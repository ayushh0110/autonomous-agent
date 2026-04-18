"""Memory analyzer — LLM-based extraction of profile and session facts.

Analyzes every user message to detect:
- Profile facts: stable identity, preferences, events, habits
- Session facts: contextual references, ongoing tasks

Quality controls:
- Confidence gate: ≥ 0.8 required
- Speculative language rejection (maybe, might, someday, etc.)
- Profile restricted to: identity, preference, event
- Session restricted to: reference, task
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_MIN_CONFIDENCE = 0.8

# Regex gate — reject speculative language before trusting the LLM output
_SPECULATIVE_RE = re.compile(
    r"\b(maybe|might|might be|could be|possibly|perhaps|probably|"
    r"thinking about|considering|someday|one day|not sure|"
    r"i guess|i think maybe|wondering if)\b",
    re.IGNORECASE,
)

_VALID_PROFILE_INTENTS = frozenset({"identity", "preference", "event"})
_VALID_SESSION_INTENTS = frozenset({"reference", "task"})

_EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction engine. Analyze the user's message and extract \
any meaningful personal or contextual information.

You MUST return ONLY valid JSON with this exact structure:
{
  "store": true or false,
  "memory_type": "profile" or "session" or "none",
  "intent": "identity" or "preference" or "event" or "reference" or "task",
  "data": {"key": "...", "value": "..."},
  "confidence": 0.0 to 1.0
}

## PROFILE (memory_type = "profile")
Store ONLY stable, reusable user information:
- identity: name, job title, role, profession, location
- preference: likes, dislikes, habits, recurring choices
- event: birthdays, anniversaries, important dates

Examples:
- "My name is Ayush" → profile/identity, key="name", value="Ayush", confidence=0.95
- "I usually travel solo" → profile/preference, key="travel_style", value="solo travel", confidence=0.85
- "1st Dec is my birthday" → profile/event, key="birthday", value="December 1st", confidence=0.95

## SESSION (memory_type = "session")
Store ONLY temporary contextual information:
- reference: entities being discussed (hotels, companies, places)
- task: ongoing actions or plans in this conversation

Examples:
- "compare those two hotels" → session/reference, key="comparison", value="two hotels", confidence=0.80
- "plan a trip to Miami" → session/task, key="trip_planning", value="Miami trip", confidence=0.85

## RULES — STRICTLY FOLLOW
- Do NOT store one-time actions ("search for X", "tell me about Y")
- Do NOT store generic factual statements ("Python is a language")
- Do NOT store speculative statements ("I might go", "maybe I'll try")
- Do NOT store questions or commands directed at the assistant
- Only store when confidence ≥ 0.8
- If nothing is extractable, return {"store": false, "memory_type": "none", "intent": "identity", "data": {}, "confidence": 0.0}
- The "key" should be a short identifier: name, job, birthday, travel_style, diet, hobby, etc.
- The "value" should be the extracted fact in plain language

Return ONLY the JSON object. No explanation, no markdown."""


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class MemoryExtraction:
    """Result of analyzing a user message for memory-worthy content."""
    store: bool
    memory_type: str      # "profile" | "session" | "none"
    intent: str           # "identity" | "preference" | "event" | "reference" | "task"
    data: dict[str, str]  # {"key": ..., "value": ...}
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "store": self.store,
            "memory_type": self.memory_type,
            "intent": self.intent,
            "data": self.data,
            "confidence": self.confidence,
        }


_NO_EXTRACTION = MemoryExtraction(
    store=False, memory_type="none", intent="identity",
    data={}, confidence=0.0,
)


# ── MemoryAnalyzer ─────────────────────────────────────────────────────────

class MemoryAnalyzer:
    """Extracts profile and session facts from user messages via LLM.

    Quality pipeline:
    1. LLM extraction → structured JSON
    2. Speculative language gate (regex)
    3. Confidence threshold (≥ 0.8)
    4. Intent validation (profile vs session)
    """

    def __init__(self, llm: GroqClient) -> None:
        self._llm = llm

    def analyze(self, user_message: str) -> MemoryExtraction:
        """Analyze a user message and extract memory-worthy content.

        Args:
            user_message: The raw user input.

        Returns:
            MemoryExtraction with store decision and extracted data.
        """
        if not user_message or not user_message.strip():
            return _NO_EXTRACTION

        # ── Pre-filter: skip very short or question-only messages ──
        stripped = user_message.strip()
        if len(stripped) < 5:
            return _NO_EXTRACTION

        # ── LLM extraction ──
        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": stripped},
                ],
                temperature=0.1,
                max_tokens=256,
                json_mode=True,
            )
        except Exception as exc:
            logger.warning("[MEMORY-ANALYZER] LLM call failed: %s", exc)
            return _NO_EXTRACTION

        # ── Parse JSON ──
        extraction = self._parse_extraction(raw)
        if extraction is None:
            return _NO_EXTRACTION

        # ── Quality gates ──
        if not extraction.store:
            logger.debug("[MEMORY-ANALYZER] LLM decided not to store")
            return _NO_EXTRACTION

        # Gate 1: Confidence threshold
        if extraction.confidence < _MIN_CONFIDENCE:
            logger.info(
                "[MEMORY-ANALYZER] Rejected — confidence %.2f < %.2f",
                extraction.confidence, _MIN_CONFIDENCE,
            )
            return _NO_EXTRACTION

        # Gate 2: Speculative language
        value = extraction.data.get("value", "")
        if _SPECULATIVE_RE.search(user_message) and extraction.memory_type == "profile":
            logger.info(
                "[MEMORY-ANALYZER] Rejected — speculative language in profile extraction: %r",
                user_message[:80],
            )
            return _NO_EXTRACTION

        # Gate 3: Intent validation
        if extraction.memory_type == "profile" and extraction.intent not in _VALID_PROFILE_INTENTS:
            logger.info(
                "[MEMORY-ANALYZER] Rejected — invalid profile intent: %s",
                extraction.intent,
            )
            return _NO_EXTRACTION

        if extraction.memory_type == "session" and extraction.intent not in _VALID_SESSION_INTENTS:
            logger.info(
                "[MEMORY-ANALYZER] Rejected — invalid session intent: %s",
                extraction.intent,
            )
            return _NO_EXTRACTION

        # Gate 4: Must have key and value
        if not extraction.data.get("key") or not extraction.data.get("value"):
            logger.info("[MEMORY-ANALYZER] Rejected — missing key or value")
            return _NO_EXTRACTION

        logger.info(
            "[MEMORY-ANALYZER] ✅ Extracted %s/%s: %s=%r (confidence=%.2f)",
            extraction.memory_type, extraction.intent,
            extraction.data.get("key"), value[:60],
            extraction.confidence,
        )
        return extraction

    # ── Parsing ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_extraction(raw: str) -> MemoryExtraction | None:
        """Parse LLM output into a MemoryExtraction, with fallback handling.

        Handles flexible LLM output formats:
        - Expected:  {"data": {"key": "birthday", "value": "December 1st"}}
        - Also OK:   {"data": {"birthday": "December 1st"}}  (auto-normalised)
        """
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

            parsed = json.loads(cleaned)

            raw_data = parsed.get("data", {})
            if not isinstance(raw_data, dict):
                raw_data = {}

            # ── Normalise data to {"key": ..., "value": ...} ──
            # If LLM already used the expected format, keep it.
            # Otherwise, take the first key-value pair from the dict.
            if "key" not in raw_data or "value" not in raw_data:
                # Filter out meta-fields that aren't actual data
                data_entries = {
                    k: v for k, v in raw_data.items()
                    if k not in ("key", "value", "store", "memory_type",
                                 "intent", "confidence", "type")
                    and isinstance(v, (str, int, float, bool))
                }
                if data_entries:
                    first_key = next(iter(data_entries))
                    raw_data = {
                        "key": first_key,
                        "value": str(data_entries[first_key]),
                    }
                    logger.debug(
                        "[MEMORY-ANALYZER] Normalised data: %s=%r",
                        first_key, raw_data["value"],
                    )

            return MemoryExtraction(
                store=bool(parsed.get("store", False)),
                memory_type=str(parsed.get("memory_type", "none")),
                intent=str(parsed.get("intent", "identity")),
                data=raw_data,
                confidence=float(parsed.get("confidence", 0.0)),
            )

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "[MEMORY-ANALYZER] Failed to parse LLM output: %s — raw: %r",
                exc, raw[:200],
            )
            return None
