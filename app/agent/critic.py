"""Critic module — evaluates synthesized answers for quality and grounding.

The Critic ONLY evaluates.  It does NOT rewrite answers.
It returns structured feedback that the Executor uses to decide
whether to refine the answer or accept it.

Phase 3.2 — Enhanced evaluation with:
- Typed + severity-tagged issues (grounding, completeness, specificity, redundancy, faithfulness)
- Severity-aware refinement decisions
- Plan-step alignment check
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.llm.groq_client import GroqClient
from app.agent.parser import _extract_json_object, _strip_markdown_fences

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_VALID_CONFIDENCE_LEVELS = {"high", "medium", "low"}
_VALID_ISSUE_TYPES = {"grounding", "completeness", "specificity", "redundancy", "faithfulness"}
_VALID_SEVERITIES = {"high", "medium", "low"}

_DEFAULT_CRITIQUE: dict[str, Any] = {
    "is_valid": True,
    "issues": [],
    "suggestions": [],
    "confidence": "medium",
}

# ── Data structures ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CriticIssue:
    """A single typed, severity-tagged issue found by the Critic."""
    type: str       # "grounding" | "completeness" | "specificity" | "redundancy" | "faithfulness"
    severity: str   # "high" | "medium" | "low"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "severity": self.severity, "detail": self.detail}


@dataclass(frozen=True)
class CriticResult:
    """Structured evaluation from the Critic."""
    is_valid: bool
    issues: list[CriticIssue]
    suggestions: list[str]
    confidence: str  # "high" | "medium" | "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }

    def has_high_severity(self) -> bool:
        """Return True if any issue has severity 'high'."""
        return any(issue.severity == "high" for issue in self.issues)

    def has_only_low_severity(self) -> bool:
        """Return True if all issues are severity 'low' (and there is at least one)."""
        return bool(self.issues) and all(issue.severity == "low" for issue in self.issues)


# ── Critic system prompt ──────────────────────────────────────────────────

_CRITIC_SYSTEM_PROMPT = """\
You are a strict answer-quality evaluator. Your ONLY job is to assess \
whether an answer adequately addresses the user's query.

## STRICT RULES
1. Respond with ONLY valid JSON. No markdown, no explanation outside JSON.
2. You MUST NOT rewrite or improve the answer.
3. You MUST NOT answer the user's question yourself.
4. Be STRICT — do NOT rubber-stamp every answer as valid.

## EVALUATION CRITERIA
Check the answer against ALL of these criteria:

1. **Grounding**: Are ALL claims in the answer supported by the step results? \
   Does the answer reference data that actually exists in the results?
2. **Completeness**: Does the answer fully address EVERY part of the user's query? \
   Are there key aspects left unanswered?
3. **Specificity**: Does the answer include concrete details (names, numbers, dates, \
   sources) when such details are available in the step results? \
   Or is it vague and generic?
4. **Redundancy**: Is the answer concise and non-repetitive? \
   Does it restate the same point in multiple ways unnecessarily?
5. **Faithfulness**: Are there any hallucinated or fabricated facts \
   NOT present in the step results?
6. **Plan alignment**: Does the answer logically reflect the work done \
   in each plan step? Are any step results ignored or misrepresented?

## ISSUE TYPES
Each issue MUST have a type from this list:
- "grounding": claim not supported by step results
- "completeness": part of the query not addressed
- "specificity": vague when concrete details were available
- "redundancy": repetitive or unnecessarily verbose content
- "faithfulness": fabricated or hallucinated facts

## ISSUE SEVERITY
- "high": Critical problem — factual error, hallucination, or major omission. \
  Answer MUST be refined.
- "medium": Noticeable gap — answer is usable but clearly improvable.
- "low": Minor stylistic or cosmetic issue — acceptable as-is.

## AUTOMATIC HIGH SEVERITY RULES
- If the answer contains generic statements that could be written WITHOUT \
the step results (e.g., "AI is rapidly evolving", "technology continues to \
advance"), mark as "specificity" with severity "high".
- If the answer does NOT clearly reflect the specific data, facts, or details \
from the step results, mark as "grounding" with severity "high".
- If the answer could have been written by someone who never saw the step \
results, it MUST be marked invalid (is_valid = false).

## CONFIDENCE LEVELS
- "high": Answer is correct, complete, well-grounded, specific, and clear.
- "medium": Answer is mostly correct but has minor gaps or could be improved.
- "low": Answer has significant issues — factual errors, missing key info, \
  or poor grounding.

## RESPONSE FORMAT
{
  "is_valid": true or false,
  "issues": [
    {
      "type": "grounding | completeness | specificity | redundancy | faithfulness",
      "severity": "high | medium | low",
      "detail": "specific description of the issue"
    }
  ],
  "suggestions": ["actionable suggestion 1", "actionable suggestion 2"],
  "confidence": "high" or "medium" or "low"
}

## WHEN TO MARK AS INVALID (is_valid = false)
- Any issue with severity "high" exists
- Answer contains fabricated facts not in the step results
- Answer misses a major part of the user's question
- Answer contradicts the step results
- Answer is vague when specific data was available in the results

## WHEN TO MARK AS VALID (is_valid = true)
- Answer correctly addresses the query using step results
- Only "low" severity issues remain (cosmetic)
- Answer acknowledges when information was not found (honest)
"""

_CRITIC_USER_TEMPLATE = """\
Evaluate the following answer to the user's query.

## Original Query
{query}

## Plan Steps Executed
{plan_steps}

## Step Results (ground truth data the answer should be based on)
{step_results}

## Answer to Evaluate
{answer}

Evaluate this answer NOW. Respond with ONLY JSON."""


# ── Critic class ───────────────────────────────────────────────────────────

class Critic:
    """Evaluates a synthesized answer for quality, completeness, and grounding.

    The Critic calls the LLM once with an evaluation-only prompt and
    returns a validated ``CriticResult`` with typed, severity-tagged issues.
    """

    def __init__(self, llm: GroqClient) -> None:
        self._llm = llm

    def evaluate(
        self,
        query: str,
        answer: str,
        step_results: list[dict[str, Any]],
        plan_steps: list[str] | None = None,
    ) -> CriticResult:
        """Evaluate the synthesized answer against the query and step results.

        Args:
            query:        The original user query.
            answer:       The synthesized answer to evaluate.
            step_results: List of serialized StepResult dicts (ground truth).
            plan_steps:   Optional list of plan step descriptions for alignment check.

        Returns:
            A validated ``CriticResult``.
            Falls back to a default "valid / medium confidence" on any failure.
        """
        logger.info("[CRITIC] Evaluating answer for query: %r", query)

        formatted_steps = self._format_step_results(step_results)
        formatted_plan = self._format_plan_steps(plan_steps)

        user_msg = _CRITIC_USER_TEMPLATE.format(
            query=query,
            plan_steps=formatted_plan,
            step_results=formatted_steps,
            answer=answer,
        )

        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=1024,
                json_mode=True,
            )

            logger.info("[CRITIC] LLM raw output (%d chars): %s", len(raw), raw)

            result = self._parse_critique(raw)
            self._log_result(result)
            return result

        except Exception as exc:
            logger.error("[CRITIC] Evaluation failed: %s", exc, exc_info=True)
            fallback = CriticResult(**_DEFAULT_CRITIQUE)
            self._log_result(fallback, fallback=True)
            return fallback

    # ── Internals ──────────────────────────────────────────────────────

    def _parse_critique(self, raw: str) -> CriticResult:
        """Extract and validate a critique from LLM raw text."""

        cleaned = _strip_markdown_fences(raw.strip())
        json_str = _extract_json_object(cleaned)
        if json_str is None:
            logger.warning("[CRITIC] No JSON object found in output")
            return CriticResult(**_DEFAULT_CRITIQUE)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning("[CRITIC] Invalid JSON: %s", exc)
            return CriticResult(**_DEFAULT_CRITIQUE)

        if not isinstance(data, dict):
            logger.warning("[CRITIC] Expected dict, got %s", type(data).__name__)
            return CriticResult(**_DEFAULT_CRITIQUE)

        return self._validate_critique(data)

    def _validate_critique(self, data: dict[str, Any]) -> CriticResult:
        """Validate and normalize the parsed critique fields."""

        # ── is_valid ──
        is_valid = data.get("is_valid")
        if not isinstance(is_valid, bool):
            logger.warning("[CRITIC] 'is_valid' missing or not bool, defaulting to True")
            is_valid = True

        # ── issues (new structured format) ──
        raw_issues = data.get("issues", [])
        issues = self._validate_issues(raw_issues)

        # ── suggestions ──
        suggestions = data.get("suggestions", [])
        if not isinstance(suggestions, list):
            suggestions = [str(suggestions)] if suggestions else []
        suggestions = [str(s) for s in suggestions if s]

        # ── confidence ──
        confidence = data.get("confidence", "medium")
        if not isinstance(confidence, str) or confidence.lower() not in _VALID_CONFIDENCE_LEVELS:
            logger.warning(
                "[CRITIC] Invalid confidence %r, defaulting to 'medium'",
                confidence,
            )
            confidence = "medium"
        else:
            confidence = confidence.lower()

        return CriticResult(
            is_valid=is_valid,
            issues=issues,
            suggestions=suggestions,
            confidence=confidence,
        )

    @staticmethod
    def _validate_issues(raw_issues: Any) -> list[CriticIssue]:
        """Parse and validate the issues list, handling both old and new formats."""
        if not isinstance(raw_issues, list):
            return []

        validated: list[CriticIssue] = []
        for item in raw_issues:
            if isinstance(item, dict):
                # ── New structured format ──
                issue_type = item.get("type", "")
                if isinstance(issue_type, str):
                    issue_type = issue_type.lower()
                if issue_type not in _VALID_ISSUE_TYPES:
                    issue_type = "completeness"  # safe default

                severity = item.get("severity", "medium")
                if isinstance(severity, str):
                    severity = severity.lower()
                if severity not in _VALID_SEVERITIES:
                    severity = "medium"

                detail = str(item.get("detail", "")).strip()
                if not detail:
                    continue

                validated.append(CriticIssue(type=issue_type, severity=severity, detail=detail))

            elif isinstance(item, str) and item.strip():
                # ── Backward compat: plain string issues → default type/severity ──
                validated.append(CriticIssue(
                    type="completeness",
                    severity="medium",
                    detail=item.strip(),
                ))
            # else: skip non-string, non-dict items

        return validated

    @staticmethod
    def _format_step_results(step_results: list[dict[str, Any]]) -> str:
        """Format step results into a readable block for the LLM."""
        if not step_results:
            return "(No step results available)"

        parts: list[str] = []
        for i, sr in enumerate(step_results, 1):
            status = "✓" if sr.get("success", False) else "✗"
            step_type = sr.get("type", "unknown")
            step_desc = sr.get("step", "No description")
            result_text = sr.get("result", "No result")
            parts.append(
                f"{i}. [{status}] [{step_type}] {step_desc}\n"
                f"   Result: {result_text}"
            )
        return "\n\n".join(parts)

    @staticmethod
    def _format_plan_steps(plan_steps: list[str] | None) -> str:
        """Format plan step descriptions for the critic prompt."""
        if not plan_steps:
            return "(Plan steps not provided)"
        lines = [f"  {i}. {step}" for i, step in enumerate(plan_steps, 1)]
        return "\n".join(lines)

    @staticmethod
    def _log_result(result: CriticResult, *, fallback: bool = False) -> None:
        """Pretty-print the critic result to logs."""
        tag = "FALLBACK " if fallback else ""
        verdict = "✅ VALID" if result.is_valid else "❌ INVALID"

        header = f"\n[CRITIC] {tag}Evaluation complete: {verdict} (confidence: {result.confidence})"
        lines = [header]

        if result.issues:
            lines.append("  Issues:")
            for issue in result.issues:
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
                lines.append(f"    {severity_icon} [{issue.severity.upper()}] [{issue.type}] {issue.detail}")

        if result.suggestions:
            lines.append("  Suggestions:")
            for suggestion in result.suggestions:
                lines.append(f"    → {suggestion}")

        msg = "\n".join(lines)
        print(msg)
        logger.info(msg)
