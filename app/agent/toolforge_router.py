"""ToolForge Model-Based Router — replaces heuristic classify_query().

This module provides a fine-tuned model alternative to the regex-based
query classifier. When a GPU is available and the adapter is loaded,
it routes queries using a QLoRA fine-tuned Qwen2.5-7B model that was
trained on 1.1K synthetic tool-call traces (86.2% accuracy vs ~75%
heuristic baseline).

Usage:
    The module is designed as a drop-in replacement. Import and use
    in executor.py via the feature flag:

        from app.agent.toolforge_router import toolforge_classify, is_toolforge_available

        if is_toolforge_available():
            decision = toolforge_classify(query, memory_hits, has_memory)
        else:
            decision = classify_query(query, memory_hits, has_memory)

Environment:
    TOOLFORGE_ENABLED=true       Enable model-based routing (default: false)
    TOOLFORGE_ADAPTER_PATH=...   Path to the LoRA adapter directory

Requires GPU + torch + peft + transformers. Falls back gracefully
if dependencies are missing.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from app.agent.agent import QueryDecision

logger = logging.getLogger(__name__)

# ── Lazy-loaded model state ─────────────────────────────────────────────────

_router = None
_init_attempted = False

# Tool name → agent decision type mapping
# ToolForge predicts specific tools; the agent pipeline needs decision types.
_TOOL_TO_DECISION = {
    "web_search": "needs_search",
    "wikipedia": "needs_search",
    "weather": "needs_search",
    "calculator": "needs_search",      # needs pipeline for tool execution
    "dictionary": "needs_search",
    "translate": "needs_search",
    "unit_converter": "needs_search",
    "datetime": "needs_search",
    "web_reader": "needs_search",
}

_TOOL_NAMES = set(_TOOL_TO_DECISION.keys())

_SYSTEM_PROMPT = (
    "You are a tool-routing assistant. Given a user query, decide which tool(s) "
    "to call and with what arguments. If no tool is needed, respond directly. "
    "You have access to: web_search, calculator, weather, wikipedia, datetime, "
    "dictionary, translate, unit_converter, web_reader. "
    'Output tool calls as: <tool_calls>[{"name": "tool", "arguments": {...}}]</tool_calls>'
)


def is_toolforge_available() -> bool:
    """Check if ToolForge model-based routing is available.

    Returns True only if:
    1. TOOLFORGE_ENABLED=true in environment
    2. GPU + required packages are available
    3. Adapter files exist at the configured path
    """
    enabled = os.getenv("TOOLFORGE_ENABLED", "false").lower() == "true"
    if not enabled:
        return False

    adapter_path = os.getenv("TOOLFORGE_ADAPTER_PATH", "")
    if not adapter_path or not os.path.isdir(adapter_path):
        logger.debug("ToolForge adapter path not found: %s", adapter_path)
        return False

    # Check for GPU + dependencies
    try:
        import torch
        if not torch.cuda.is_available():
            logger.debug("ToolForge requires GPU — CUDA not available")
            return False
    except ImportError:
        return False

    return True


def _init_router() -> bool:
    """Lazy-initialize the ToolForge model. Called once on first query."""
    global _router, _init_attempted

    if _init_attempted:
        return _router is not None
    _init_attempted = True

    adapter_path = os.getenv("TOOLFORGE_ADAPTER_PATH", "")

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Read base model from adapter config
        config_path = os.path.join(adapter_path, "adapter_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model_id = config["base_model_name_or_path"]

        logger.info("ToolForge: Loading base model %s...", model_id)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _router = {"model": model, "tokenizer": tokenizer, "torch": torch}
        logger.info("ToolForge: Model loaded successfully!")
        return True

    except Exception as exc:
        logger.error("ToolForge: Failed to load model: %s", exc)
        _router = None
        return False


def _parse_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model output."""
    match = re.search(r'<tool_calls>(.*?)</tool_calls>', text, re.DOTALL)
    if match:
        try:
            calls = json.loads(match.group(1))
            if isinstance(calls, list):
                return [tc for tc in calls if tc.get("name") in _TOOL_NAMES]
        except json.JSONDecodeError:
            pass

    # Fallback: try raw JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            return [tc for tc in data if tc.get("name") in _TOOL_NAMES]
    except json.JSONDecodeError:
        pass

    return []


def toolforge_classify(
    query: str,
    memory_hits: int = 0,
    has_memory: bool = False,
) -> Optional[QueryDecision]:
    """Classify a query using the fine-tuned ToolForge model.

    Returns a QueryDecision compatible with the existing pipeline,
    or None if the model fails (caller should fall back to heuristic).

    The model predicts specific tools; this function maps them to
    decision types the executor understands:
        - tool predicted → "needs_search" (so the pipeline executes tools)
        - no tool predicted → "direct_answer"
        - memory available + no tool → "memory_sufficient"
    """
    if not _init_router():
        return None

    torch = _router["torch"]
    model = _router["model"]
    tokenizer = _router["tokenizer"]

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

        tool_calls = _parse_tool_calls(raw_output)

        if tool_calls:
            # Model says use a tool → route through pipeline
            primary_tool = tool_calls[0]["name"]
            decision_type = _TOOL_TO_DECISION.get(primary_tool, "needs_search")
            return QueryDecision(
                decision_type=decision_type,
                reasoning=f"ToolForge model predicted tool: {primary_tool}",
                confidence=0.92,
            )
        else:
            # Model says no tool needed
            if has_memory and memory_hits >= 2:
                return QueryDecision(
                    decision_type="memory_sufficient",
                    reasoning=f"ToolForge: no tool needed, {memory_hits} memory hits available",
                    confidence=0.88,
                )
            return QueryDecision(
                decision_type="direct_answer",
                reasoning="ToolForge model: no tool needed — direct answer",
                confidence=0.88,
            )

    except Exception as exc:
        logger.error("ToolForge inference failed: %s", exc)
        return None
