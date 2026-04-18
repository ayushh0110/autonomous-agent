"""Robust JSON action parser for LLM outputs.

The LLM is instructed to respond with structured JSON in one of two forms:

    {"action": "tool_call", "reasoning": "...", "tool_name": "...", "tool_input": {...}}
    {"action": "final_answer", "reasoning": "...", "answer": "..."}

The 'reasoning' field is optional but logged for observability.
It is NEVER returned in the API response.

In practice, LLMs (especially smaller ones) produce:
- Markdown fences around JSON
- Preamble text before the JSON
- Trailing explanations after the JSON
- Slightly malformed JSON (trailing commas, single quotes)

This module handles all of that and returns typed dataclasses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Result types ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallAction:
    """LLM decided to call a tool."""
    tool_name: str
    tool_input: dict[str, Any]
    reasoning: str = ""


@dataclass(frozen=True)
class FinalAnswerAction:
    """LLM decided to give the final answer."""
    answer: str
    reasoning: str = ""


@dataclass(frozen=True)
class ParseError:
    """Failed to parse the LLM output into a valid action."""
    raw_output: str
    error: str


# Union type for convenience
ActionResult = ToolCallAction | FinalAnswerAction | ParseError


# ── Parser ─────────────────────────────────────────────────────────────────

def parse_llm_action(raw: str) -> ActionResult:
    """Parse LLM raw text into a typed action.

    Extraction strategy:
    1. Strip markdown code fences if present
    2. Find the first top-level JSON object via brace matching
    3. Parse and validate required fields

    Args:
        raw: Raw string output from the LLM.

    Returns:
        ToolCallAction, FinalAnswerAction, or ParseError.
    """
    if not raw or not raw.strip():
        return ParseError(raw_output=raw or "", error="Empty LLM output")

    cleaned = _strip_markdown_fences(raw.strip())
    json_str = _extract_json_object(cleaned)

    if json_str is None:
        return ParseError(
            raw_output=raw,
            error="No JSON object found in LLM output",
        )

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return ParseError(
            raw_output=raw,
            error=f"Invalid JSON: {exc}",
        )

    if not isinstance(data, dict):
        return ParseError(
            raw_output=raw,
            error=f"Expected JSON object, got {type(data).__name__}",
        )

    return _validate_action(data, raw)


# ── Internals ──────────────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    # Match ```json\n...\n``` or ```\n...\n```
    pattern = r"^```(?:json)?\s*\n?(.*?)\n?\s*```$"
    match = re.match(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced top-level { ... } in the text.

    Uses brace-depth counting to handle nested objects correctly,
    unlike naive regex that breaks on nested braces.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None  # Unbalanced braces


def _validate_action(data: dict[str, Any], raw: str) -> ActionResult:
    """Validate the parsed JSON against expected action schemas."""
    action = data.get("action")

    if action is None:
        return ParseError(
            raw_output=raw,
            error="Missing 'action' field in JSON",
        )

    if action == "tool_call":
        return _validate_tool_call(data, raw)

    if action == "final_answer":
        return _validate_final_answer(data, raw)

    return ParseError(
        raw_output=raw,
        error=f"Unknown action type: '{action}'. Must be 'tool_call' or 'final_answer'.",
    )


def _validate_tool_call(data: dict[str, Any], raw: str) -> ToolCallAction | ParseError:
    """Validate a tool_call action."""
    tool_name = data.get("tool_name")
    if not tool_name or not isinstance(tool_name, str):
        return ParseError(
            raw_output=raw,
            error="'tool_call' action requires a non-empty 'tool_name' string",
        )

    tool_input = data.get("tool_input")
    if tool_input is None:
        tool_input = {}
    if not isinstance(tool_input, dict):
        return ParseError(
            raw_output=raw,
            error=f"'tool_input' must be an object, got {type(tool_input).__name__}",
        )

    return ToolCallAction(tool_name=tool_name, tool_input=tool_input, reasoning=data.get("reasoning", ""))


def _stringify_answer(answer: Any) -> str:
    """Convert a non-string answer (list, dict) into clean bullet-point text.

    Handles common LLM patterns:
    - ["fact1", "fact2"]                    → "- fact1\\n- fact2"
    - {"facts": ["fact1", "fact2"]}         → "- fact1\\n- fact2"
    - {"key": "value", "key2": ["a", "b"]} → "- key: value\\n- key2:\\n  - a\\n  - b"
    """
    if isinstance(answer, list):
        items = [str(item).strip() for item in answer if item]
        return "\n".join(f"- {item}" for item in items) if items else str(answer)

    if isinstance(answer, dict):
        # Single-key dict wrapping a list → unwrap (e.g., {"facts": [...]})
        values = list(answer.values())
        if len(values) == 1 and isinstance(values[0], list):
            return _stringify_answer(values[0])
        # Multi-key dict → key-value bullets
        lines: list[str] = []
        for key, value in answer.items():
            if isinstance(value, list):
                lines.append(f"- {key}:")
                for item in value:
                    lines.append(f"  - {str(item).strip()}")
            else:
                lines.append(f"- {key}: {str(value).strip()}")
        return "\n".join(lines) if lines else str(answer)

    return str(answer)


def _validate_final_answer(data: dict[str, Any], raw: str) -> FinalAnswerAction | ParseError:
    """Validate a final_answer action."""
    answer = data.get("answer")
    if answer is None:
        return ParseError(
            raw_output=raw,
            error="'final_answer' action requires an 'answer' field",
        )

    if not isinstance(answer, str):
        # Be lenient — convert structured data to bullet-point text
        answer = _stringify_answer(answer)

    return FinalAnswerAction(answer=answer, reasoning=data.get("reasoning", ""))
