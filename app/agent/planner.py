"""Planner module — decomposes a user query into a structured execution plan.

The Planner ONLY generates a plan.  It does NOT answer the question
and does NOT call tools.  The Executor handles execution.

Phase 4 — Adaptive planning:
- Accepts a decision type from the classifier
- Skips LLM call for direct_answer and memory_sufficient queries
- Only calls LLM for needs_search (full planning needed)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.llm.groq_client import GroqClient
from app.agent.parser import _extract_json_object, _strip_markdown_fences

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_PLAN_STEPS = 5
_VALID_STEP_TYPES = {"tool", "reasoning"}

_FALLBACK_PLAN_STEP: dict[str, str] = {
    "step": "Answer the query directly",
    "type": "reasoning",
}

# ── Pre-built plans for fast-path decisions ────────────────────────────────

_DIRECT_ANSWER_PLAN_STEP: dict[str, str] = {
    "step": "Respond to the user directly and conversationally — no tools needed. "
            "If the user shared personal information, acknowledge it warmly. "
            "If they asked a knowledge question, answer from training data.",
    "type": "reasoning",
}

_MEMORY_PLAN_STEP: dict[str, str] = {
    "step": "Answer the query using relevant past information from memory",
    "type": "reasoning",
}

_AUTONOMOUS_PLAN_STEP: dict[str, str] = {
    "step": "Autonomous execution — handled by autonomous executor",
    "type": "reasoning",
}

# ── Data structures ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlanStep:
    """One step in a structured execution plan."""
    step: str    # e.g. "Search for latest AI news articles from 2024"
    type: str    # "tool" | "reasoning"


# ── Planner system prompt ──────────────────────────────────────────────────

_PLANNER_SYSTEM_PROMPT = """\
You are a planning assistant. Your ONLY job is to decompose a user query \
into a structured execution plan.

## STRICT RULES
1. Respond with ONLY valid JSON. No markdown, no explanation outside JSON.
2. You MUST NOT answer the user's question.
3. You MUST NOT call any tools.
4. Minimum 1 step, maximum 5 steps.
5. Do NOT introduce specific dates, numbers, or constraints unless \
they are explicitly mentioned in the user's query.

## STEP QUALITY REQUIREMENTS
Each step MUST include:
- An explicit SUBJECT (what entity/data is being acted on)
- A clear ACTION VERB (search, extract, summarize, compare, list)
- CONTEXT (what data it operates on, where it comes from)

REJECTED (vague):
- "analyze"
- "explore"
- "find trends"
- "understand the topic"

ACCEPTED (specific):
- "Search for recent AI news articles and breakthroughs"
- "Extract the top 3 breakthroughs from the retrieved search results"
- "Summarize the key pricing differences between the two products"

## STEP TYPES
- "tool": step requires fetching external data (web search, API call)
- "reasoning": step involves processing, summarizing, or analyzing existing data

## RESPONSE FORMAT
{
  "plan": [
    {"step": "specific action description", "type": "tool" or "reasoning"},
    ...
  ]
}

## EXAMPLES

Query: "What are the latest developments in AI?"
{
  "plan": [
    {"step": "Search for the latest AI news articles and recent breakthroughs", "type": "tool"},
    {"step": "Extract the key developments and announcements from the search results", "type": "reasoning"},
    {"step": "Organize the findings into a clear summary of major AI trends", "type": "reasoning"}
  ]
}

Query: "Explain how photosynthesis works"
{
  "plan": [
    {"step": "Provide a detailed explanation of the photosynthesis process including light-dependent and light-independent reactions", "type": "reasoning"}
  ]
}

Query: "Compare Python and Rust for backend development"
{
  "plan": [
    {"step": "Search for recent comparisons of Python vs Rust for backend development", "type": "tool"},
    {"step": "List the strengths and weaknesses of each language for backend use based on the search results", "type": "reasoning"},
    {"step": "Provide a recommendation based on different use-case scenarios", "type": "reasoning"}
  ]
}

Query: "What did OpenAI announce at their latest event?"
{
  "plan": [
    {"step": "Search for OpenAI's latest event announcements and news coverage", "type": "tool"},
    {"step": "Read the full article from the most relevant search result to extract detailed information", "type": "tool"},
    {"step": "Summarize the key announcements, products, and details from the event", "type": "reasoning"}
  ]
}
"""


# ── Planner class ──────────────────────────────────────────────────────────

class Planner:
    """Generates a structured, typed execution plan from a user query.

    The planner calls the LLM once with a planning-only prompt and
    returns a validated list of ``PlanStep`` objects.

    Adaptive behaviour (Phase 4):
    - decision == "direct_answer"     → instant reasoning step (no LLM call)
    - decision == "memory_sufficient" → instant reasoning step (no LLM call)
    - decision == "needs_search"      → full LLM-generated plan
    """

    def __init__(
        self,
        llm: GroqClient,
        max_plan_steps: int = _DEFAULT_MAX_PLAN_STEPS,
    ) -> None:
        self._llm = llm
        self._max_plan_steps = max_plan_steps

    def create_plan(
        self,
        query: str,
        decision: str | None = None,
    ) -> list[PlanStep]:
        """Decompose *query* into a list of typed plan steps.

        Args:
            query: The user's query.
            decision: Optional decision type from classify_query().
                      If "direct_answer" or "memory_sufficient", returns
                      a fast-path plan without calling the LLM.

        Returns:
            A validated list of ``PlanStep`` (1–max_plan_steps items).
            Falls back to a single reasoning step on any failure.
        """
        # ── Fast-path: skip LLM call for simple decisions ──
        if decision == "direct_answer":
            plan = [PlanStep(**_DIRECT_ANSWER_PLAN_STEP)]
            fast_msg = "[PLANNER] ⚡ Fast-path: direct_answer — skipping LLM planning"
            print(fast_msg)
            logger.info(fast_msg)
            self._log_plan(plan)
            return plan

        if decision == "memory_sufficient":
            plan = [PlanStep(**_MEMORY_PLAN_STEP)]
            fast_msg = "[PLANNER] ⚡ Fast-path: memory_sufficient — skipping LLM planning"
            print(fast_msg)
            logger.info(fast_msg)
            self._log_plan(plan)
            return plan

        if decision == "autonomous_task":
            plan = [PlanStep(**_AUTONOMOUS_PLAN_STEP)]
            fast_msg = "[PLANNER] ⚡ Fast-path: autonomous_task — delegating to autonomous executor"
            print(fast_msg)
            logger.info(fast_msg)
            self._log_plan(plan)
            return plan

        # ── Full planning: call LLM ──
        logger.info("[PLANNER] Creating plan for query: %r (decision=%s)", query, decision)

        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": _PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.2,
                max_tokens=1024,
                json_mode=True,
            )

            logger.info("[PLANNER] LLM raw output (%d chars): %s", len(raw), raw)

            plan = self._parse_plan(raw)
            self._log_plan(plan)
            return plan

        except Exception as exc:
            logger.error("[PLANNER] Plan generation failed: %s", exc, exc_info=True)
            fallback = [PlanStep(**_FALLBACK_PLAN_STEP)]
            self._log_plan(fallback, fallback=True)
            return fallback

    # ── Internals ──────────────────────────────────────────────────────

    def _parse_plan(self, raw: str) -> list[PlanStep]:
        """Extract and validate a plan from LLM raw text."""

        cleaned = _strip_markdown_fences(raw.strip())
        json_str = _extract_json_object(cleaned)
        if json_str is None:
            logger.warning("[PLANNER] No JSON object found in output")
            return [PlanStep(**_FALLBACK_PLAN_STEP)]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("[PLANNER] Invalid JSON: %s", exc)
            return [PlanStep(**_FALLBACK_PLAN_STEP)]

        if not isinstance(data, dict):
            logger.warning("[PLANNER] Expected dict, got %s", type(data).__name__)
            return [PlanStep(**_FALLBACK_PLAN_STEP)]

        raw_plan = data.get("plan")
        if not isinstance(raw_plan, list):
            logger.warning("[PLANNER] Missing or invalid 'plan' key")
            return [PlanStep(**_FALLBACK_PLAN_STEP)]

        return self._validate_steps(raw_plan)

    def _validate_steps(self, raw_plan: list[Any]) -> list[PlanStep]:
        """Validate each step has a non-empty 'step' string and a valid 'type'."""
        validated: list[PlanStep] = []

        for item in raw_plan:
            if not isinstance(item, dict):
                logger.warning("[PLANNER] Skipping non-dict step: %s", item)
                continue

            step_text = item.get("step")
            step_type = item.get("type")

            if not isinstance(step_text, str) or not step_text.strip():
                logger.warning("[PLANNER] Skipping step with missing/empty 'step': %s", item)
                continue

            if step_type not in _VALID_STEP_TYPES:
                logger.warning(
                    "[PLANNER] Skipping step with invalid type %r (must be %s): %s",
                    step_type, _VALID_STEP_TYPES, item,
                )
                continue

            validated.append(PlanStep(step=step_text.strip(), type=step_type))

        if not validated:
            logger.warning("[PLANNER] No valid steps after validation — using fallback")
            return [PlanStep(**_FALLBACK_PLAN_STEP)]

        return validated[: self._max_plan_steps]

    @staticmethod
    def _log_plan(plan: list[PlanStep], *, fallback: bool = False) -> None:
        """Pretty-print the plan to logs."""
        tag = "FALLBACK " if fallback else ""
        header = f"\n[PLANNER] {tag}Generated plan ({len(plan)} steps):"
        lines = [header]
        for i, ps in enumerate(plan, 1):
            lines.append(f"  {i}. [{ps.type}] {ps.step}")
        msg = "\n".join(lines)
        print(msg)
        logger.info(msg)
