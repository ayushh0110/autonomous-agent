"""Core agent logic — LLM-driven agentic loop with tool calling.

Phase 2.1 — System refinement:
- Tool call limit:        max N tool invocations per request (default 2)
- Final step enforcement: last step forces final_answer, tool_calls blocked
- Reasoning visibility:   LLM explains decisions (logged, never in API)
- Invalid tool protection: errors fed back to LLM as context, not crashes
- Prompt discipline:      stricter JSON-only enforcement, no looping waste

Phase 4 — Adaptive intelligence:
- Query classification:   heuristic-based decision layer (no LLM overhead)
- Tool result caching:    duplicate calls served from cache
- LLM call tracking:      per-request LLM call counter for metrics
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.llm.groq_client import GroqClient
from app.tools.base_tool import BaseTool
from app.tools.registry import ToolRegistry
from app.agent.parser import (
    parse_llm_action,
    ToolCallAction,
    FinalAnswerAction,
    ParseError,
)

logger = logging.getLogger(__name__)

# ── Defaults ───────────────────────────────────────────────────────────────

_DEFAULT_MAX_STEPS = 4
_DEFAULT_MAX_TOOL_CALLS = 2


# ── System prompts ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are a friendly, intelligent AI assistant with access to tools.
You speak naturally and conversationally — like a knowledgeable friend, not a report.

## YOUR PERSONALITY
- Be warm, natural, and human-sounding in all responses
- When you looked something up, talk about it naturally: "So I looked this up and..." \
or "From what I found..."
- When someone shares personal info, respond warmly: "That's great!" / "Nice!" / etc.
- Never sound robotic, clinical, or like a data dump
- Use casual connectors, light enthusiasm, and direct address
- Keep it concise but friendly — no walls of text

## AVAILABLE TOOLS
{tool_block}

## RESPONSE FORMAT
You MUST respond with ONLY valid JSON in one of these two formats.

### Option 1 — Call a tool:
{{
  "action": "tool_call",
  "reasoning": "brief explanation of why you need this tool",
  "tool_name": "<tool name>",
  "tool_input": {{<inputs matching the tool's schema>}}
}}

### Option 2 — Give the final answer:
{{
  "action": "final_answer",
  "reasoning": "brief explanation of how you arrived at this answer",
  "answer": "<your complete, conversational answer to the user>"
}}

## RULES
1. Respond with ONLY valid JSON. No markdown, no explanation outside JSON.
2. Only two actions exist: "tool_call" and "final_answer". Nothing else.
3. If you can answer from your own knowledge, use "final_answer" immediately.
4. Only use a tool when you genuinely need real-time or external information.
5. Do NOT call the same tool with the same input twice.
6. After receiving tool results, prefer giving a final_answer over calling more tools.
7. When citing tool results, weave them naturally into your response — don't just list facts.
8. If a tool returns an error, explain the issue conversationally in your final_answer.
9. NEVER fabricate tool results. Only use actual results provided to you.
10. The "reasoning" field is required — always explain your decision.
"""

_NO_TOOLS_SYSTEM = """\
You are a friendly, intelligent AI assistant. No tools are available.
You speak naturally and conversationally — like a helpful friend, not a report.
Be warm, direct, and human-sounding.

You MUST respond with ONLY valid JSON:
{{
  "action": "final_answer",
  "reasoning": "brief explanation",
  "answer": "<your complete, conversational answer>"
}}
No markdown, no explanation outside JSON. Only "final_answer" is allowed.
"""

_FINAL_STEP_INJECTION = (
    "\n\n⚠️ SYSTEM NOTICE: This is your FINAL step. "
    "You MUST respond with a final_answer NOW. "
    "Tool calls are NOT allowed on this step. "
    'You MUST use: {"action": "final_answer", "reasoning": "...", "answer": "..."}'
)

_TOOL_LIMIT_INJECTION = (
    "\n\n⚠️ SYSTEM NOTICE: You have reached the maximum number of tool calls. "
    "No more tool calls are allowed. "
    'You MUST give your final_answer based on the information you already have.'
)

_PARSE_RETRY_MSG = (
    "Your previous response was not valid JSON. "
    "You MUST respond with ONLY a JSON object. "
    'Either {"action": "tool_call", "reasoning": "...", "tool_name": "...", "tool_input": {...}} '
    'or {"action": "final_answer", "reasoning": "...", "answer": "..."}. '
    "Try again."
)

_TOOL_ERROR_TEMPLATE = (
    "Tool execution failed: {error}\n\n"
    "Choose another action. If you cannot recover, give your final_answer "
    "explaining what went wrong."
)

_TOOL_RESULT_TEMPLATE = (
    "Tool '{tool_name}' returned:\n\n"
    "{result}\n\n"
    "Analyze these results. If you have enough information, respond with "
    "final_answer. Do NOT call another tool unless absolutely necessary."
)

_TOOL_RESULT_EXTRACTION_TEMPLATE = (
    "Tool '{tool_name}' returned:\n\n"
    "{result}\n\n"
    "Extract ONLY concrete facts from these results as bullet points. "
    "Include: specific names, events, dates, numbers, and sources. "
    "Do NOT summarize, infer trends, or generate a narrative. "
    "Respond with final_answer containing the extracted facts as bullet points."
)

_REQUIRE_TOOL_INJECTION = (
    "Do NOT produce a final_answer. You MUST call a tool first "
    "to retrieve external data. Extract factual data only — "
    "do not summarize or generate answers."
)


def _build_tool_block(schemas: list[dict[str, Any]]) -> str:
    """Format tool schemas for the system prompt."""
    if not schemas:
        return "(No tools available)"

    parts = []
    for s in schemas:
        inputs = json.dumps(s["input_schema"], indent=2)
        parts.append(
            f"### {s['name']}\n"
            f"Description: {s['description']}\n"
            f"Inputs:\n```\n{inputs}\n```"
        )
    return "\n\n".join(parts)


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class LoopStep:
    """One iteration of the agent loop, for observability."""
    step: int
    action_type: str             # "tool_call" | "final_answer" | "parse_error" | ...
    reasoning: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: str | None = None
    answer: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    """Structured response from the agent with full trace metadata."""
    answer: str
    source: str                  # "agent_loop" | "fallback" | "error"
    steps_taken: int = 0
    tools_used: list[str] = field(default_factory=list)
    trace: list[LoopStep] = field(default_factory=list)
    llm_calls: int = 0
    cache_hits: int = 0


# ── Query Decision Layer ──────────────────────────────────────────────────

_SEARCH_INDICATORS = re.compile(
    r'\b(latest|recent|new|current|today|trending|breaking|'
    r'this week|this month|this year|2024|2025|2026|'
    r'news|price|cost|stock|weather|score|results?|'
    r'who won|how much|what happened|statistics|stats)\b',
    re.IGNORECASE,
)

_KNOWLEDGE_INDICATORS = re.compile(
    r'\b(explain|what is|what are|how does|how do|why does|why do|'
    r'define|describe|difference between|history of|'
    r'how to|tutorial|example of|calculate|solve|'
    r'concept of|meaning of|overview of|introduction to|'
    r'fundamentals|basics|principles)\b',
    re.IGNORECASE,
)

_RECENCY_SIGNALS = re.compile(
    r'\b(latest|today|breaking|trending|this week|this month|'
    r'right now|just|recently|2025|2026)\b',
    re.IGNORECASE,
)

_FACTUAL_INDICATORS = re.compile(
    r'\b(who is|who are|who was|who were|'
    r'when did|when was|when is|when were|'
    r'where is|where are|where was|'
    r'how many|how long|how old|how tall|how far|'
    r'population|gdp|capital of|founded|net worth|'
    r'salary|revenue|market cap|born|died|'
    r'height|weight|distance|area|size of)\b',
    re.IGNORECASE,
)

# Personal / conversational statements — route to direct_answer, not search.
# These are things users tell the agent about themselves or greetings.
_PERSONAL_INDICATORS = re.compile(
    r'\b(my name is|i am|i\'m|my birthday|my profession|'
    r'i like|i love|i hate|i prefer|i enjoy|i dislike|'
    r'i work as|i work at|i live in|i\'m from|'
    r'i usually|i always|i never|'
    r'call me|remember that|keep in mind|'
    r'thank you|thanks|hi|hello|hey|good morning|'
    r'good night|goodbye|bye|how are you|'
    r'nice to meet|pleased to meet)\b',
    re.IGNORECASE,
)

# Planning / multi-step task indicators — route to autonomous executor.
_TASK_INDICATORS = re.compile(
    r'\b(plan a|plan my|plan for|create a plan|build a|build me|'
    r'organize|create itinerary|make an itinerary|'
    r'compare .+ (vs|versus|or|and)|compare options|'
    r'research and|find .+ and .+|step by step|'
    r'detailed guide|complete guide|full guide|'
    r'find flights|find hotels|find places|'
    r'book .+ and|trip to|travel to|vacation to|'
    r'prepare for|set up|put together)\b',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryDecision:
    """Classification result for routing decisions."""
    decision_type: str   # "direct_answer" | "needs_search" | "memory_sufficient" | "autonomous_task"
    reasoning: str
    confidence: float


def classify_query(
    query: str,
    memory_hits: int = 0,
    has_memory: bool = False,
) -> QueryDecision:
    """Classify a query to determine the optimal execution path.

    Uses pure heuristics — zero LLM calls.

    Decision priority:
        1. Strong recency signals → needs_search (always)
        2. Sufficient memory hits (≥2) without recency → memory_sufficient
        3. Knowledge/conceptual patterns → direct_answer
        4. Any search indicators → needs_search
        5. Default → needs_search (safe fallback)
    """
    q = query.strip()

    has_recency = bool(_RECENCY_SIGNALS.search(q))
    needs_search = bool(_SEARCH_INDICATORS.search(q))
    is_knowledge = bool(_KNOWLEDGE_INDICATORS.search(q))
    is_factual = bool(_FACTUAL_INDICATORS.search(q))
    is_personal = bool(_PERSONAL_INDICATORS.search(q))
    is_task = bool(_TASK_INDICATORS.search(q))

    # Priority 0: Complex multi-step tasks → autonomous executor
    # "plan a trip to Japan", "compare React vs Vue", "find flights and hotels"
    if is_task and not is_personal:
        return QueryDecision(
            decision_type="autonomous_task",
            reasoning="Complex multi-step task requiring autonomous execution",
            confidence=0.9,
        )

    # Priority 0.5: Personal / conversational statements → direct answer
    # "my birthday is Dec 1st", "I'm a developer", "hi", "thanks"
    if is_personal and not needs_search and not has_recency:
        return QueryDecision(
            decision_type="direct_answer",
            reasoning="Personal or conversational statement — respond directly",
            confidence=0.9,
        )

    # Priority 1: Strong recency signals always need fresh data
    if has_recency:
        return QueryDecision(
            decision_type="needs_search",
            reasoning="Query contains recency signals requiring fresh data",
            confidence=0.9,
        )

    # Priority 2: Memory can answer non-recency queries
    if has_memory and memory_hits >= 2 and not needs_search:
        return QueryDecision(
            decision_type="memory_sufficient",
            reasoning=f"Found {memory_hits} relevant memory entries for non-recency query",
            confidence=0.8,
        )

    # Priority 3: Knowledge / conceptual question
    # Guard: factual data questions must go through search even if they
    # match knowledge patterns (e.g. "What is the GDP of India?")
    if is_knowledge and not needs_search and not is_factual:
        return QueryDecision(
            decision_type="direct_answer",
            reasoning="Conceptual/educational query answerable from training data",
            confidence=0.85,
        )

    # Priority 3.5: Factual question overrides knowledge classification
    if is_factual:
        return QueryDecision(
            decision_type="needs_search",
            reasoning="Factual data question requiring verified information",
            confidence=0.85,
        )

    # Priority 4: Search indicators present
    if needs_search:
        return QueryDecision(
            decision_type="needs_search",
            reasoning="Query contains search indicators",
            confidence=0.8,
        )

    # Default: uncertain → search for quality
    return QueryDecision(
        decision_type="needs_search",
        reasoning="Uncertain query type — defaulting to search for quality",
        confidence=0.5,
    )


# ── Agent ──────────────────────────────────────────────────────────────────

class Agent:
    """LLM-driven agent with structured tool calling and safety controls.

    Controls:
    - max_steps:      hard cap on loop iterations (default 4)
    - max_tool_calls: hard cap on tool invocations per request (default 2)
    - final step:     last iteration forces final_answer, blocks tool_calls
    - reasoning:      LLM must explain decisions (logged, not in API)
    - tool cache:     duplicate tool calls served from cache
    """

    def __init__(
        self,
        llm: GroqClient,
        tools: Sequence[BaseTool] | None = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        max_tool_calls: int = _DEFAULT_MAX_TOOL_CALLS,
    ) -> None:
        self._llm = llm
        self._registry = ToolRegistry(tools)
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._tool_cache: dict[str, str] = {}

        # Pre-build the system prompt (static for lifetime of agent)
        schemas = self._registry.get_schemas()
        if schemas:
            tool_block = _build_tool_block(schemas)
            self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(tool_block=tool_block)
        else:
            self._system_prompt = _NO_TOOLS_SYSTEM

        logger.info(
            "Agent initialised | tools=%s | max_steps=%d | max_tool_calls=%d",
            self._registry.tool_names or "(none)",
            self._max_steps,
            self._max_tool_calls,
        )

    # ── Cache helpers ──────────────────────────────────────────────────

    def clear_cache(self) -> None:
        """Clear the tool result cache.

        Called by the Executor at the start of each request
        to ensure request-scoped caching (prevents stale cross-request hits
        while preserving within-request deduplication).
        """
        if self._tool_cache:
            logger.info("Agent tool cache cleared (%d entries)", len(self._tool_cache))
            self._tool_cache.clear()

    def _get_cache_key(self, tool_name: str, tool_input: dict) -> str:
        """Normalize tool call into a hashable cache key."""
        normalized = json.dumps(tool_input, sort_keys=True).lower()
        return f"{tool_name}:{normalized}"

    # ── Public API ─────────────────────────────────────────────────────

    def handle(self, query: str, *, allow_tools: bool = True) -> str:
        """Simple API: query in → answer out."""
        result = self.handle_full(query, allow_tools=allow_tools)
        return result.answer

    def handle_full(self, query: str, *, allow_tools: bool = True, require_tool: bool = False) -> AgentResult:
        """Full API: query in → structured result with trace metadata.

        Args:
            query: User input.
            allow_tools: If False, tools are disabled for this request
                         (used by the Executor for 'reasoning' steps).
            require_tool: If True, block final_answer until at least one
                          tool has been called (used by the Executor for
                          'tool' plan steps).
        """
        req_id = uuid.uuid4().hex[:8]

        if not query or not query.strip():
            return AgentResult(answer="Please provide a valid query.", source="error")

        logger.info("[%s] ═══ LOOP START ═══ query=%r", req_id, query)

        try:
            result = self._run_loop(query, req_id, allow_tools=allow_tools, require_tool=require_tool)
            logger.info(
                "[%s] ═══ LOOP END ═══ steps=%d tool_calls=%s source=%s llm_calls=%d",
                req_id, result.steps_taken, result.tools_used, result.source, result.llm_calls,
            )
            return result
        except Exception as exc:
            logger.error("[%s] Loop crashed: %s", req_id, exc, exc_info=True)
            return AgentResult(
                answer="Something went wrong while processing your request.",
                source="error",
            )

    # ── Core loop ──────────────────────────────────────────────────────

    def _run_loop(self, query: str, req_id: str, *, allow_tools: bool = True, require_tool: bool = False) -> AgentResult:
        """The agentic loop with tool-call limits and final-step enforcement."""

        # ── State ──
        # When tools are disabled (reasoning steps), use the no-tools prompt
        system_prompt = self._system_prompt if allow_tools else _NO_TOOLS_SYSTEM
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        trace: list[LoopStep] = []
        tools_used: list[str] = []
        tool_call_count: int = 0
        llm_call_count: int = 0
        cache_hits: int = 0

        for step_num in range(1, self._max_steps + 1):

            is_final_step = (step_num == self._max_steps)
            tools_exhausted = (tool_call_count >= self._max_tool_calls)

            # ────────────────────────────────────────────────────────────
            # LOG: Step start
            # ────────────────────────────────────────────────────────────
            step_header = (
                f"\n{'=' * 70}\n"
                f"[STEP {step_num}] Starting step for query: \"{query}\"\n"
                f"[STEP {step_num}] step={step_num}/{self._max_steps} | "
                f"tool_calls={tool_call_count}/{self._max_tool_calls} | "
                f"final_step={is_final_step}\n"
                f"{'=' * 70}"
            )
            print(step_header)
            logger.info("[%s] %s", req_id, step_header)

            # ── Inject constraints into context for this step ──
            step_messages = list(messages)  # shallow copy
            if is_final_step:
                step_messages = self._inject_constraint(step_messages, _FINAL_STEP_INJECTION)
                print(f"[STEP {step_num}] ⚠️  FINAL STEP — injected final_answer constraint")
                logger.info("[%s] [STEP %d] Injected FINAL_STEP constraint", req_id, step_num)
            elif tools_exhausted:
                step_messages = self._inject_constraint(step_messages, _TOOL_LIMIT_INJECTION)
                print(f"[STEP {step_num}] ⚠️  TOOL LIMIT — injected no-more-tools constraint")
                logger.info("[%s] [STEP %d] Injected TOOL_LIMIT constraint", req_id, step_num)

            # ── Call LLM ──
            try:
                raw = self._llm.chat(
                    messages=step_messages,
                    temperature=0.1,
                    max_tokens=2048,
                    json_mode=True,
                )
                llm_call_count += 1
            except RuntimeError as exc:
                err_msg = f"[STEP {step_num}] ❌ LLM CALL FAILED: {exc}"
                print(err_msg)
                logger.error("[%s] %s", req_id, err_msg)
                trace.append(LoopStep(step=step_num, action_type="llm_error", error=str(exc)))
                return AgentResult(
                    answer="The AI service is currently unavailable. Please try again.",
                    source="error",
                    steps_taken=step_num,
                    tools_used=tools_used,
                    trace=trace,
                    llm_calls=llm_call_count,
                )

            # ────────────────────────────────────────────────────────────
            # LOG: Full LLM raw output (UNTRUNCATED)
            # ────────────────────────────────────────────────────────────
            raw_log = (
                f"\n[STEP {step_num}] LLM RAW OUTPUT ({len(raw)} chars):\n"
                f"{'-' * 50}\n"
                f"{raw}\n"
                f"{'-' * 50}"
            )
            print(raw_log)
            logger.info("[%s] %s", req_id, raw_log)

            # ── Parse ──
            action = parse_llm_action(raw)

            # ── PARSE ERROR → retry once ──
            if isinstance(action, ParseError):
                err_log = f"[STEP {step_num}] ⚠️  PARSE ERROR: {action.error}"
                print(err_log)
                logger.warning("[%s] %s", req_id, err_log)
                trace.append(LoopStep(step=step_num, action_type="parse_error", error=action.error))
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": _PARSE_RETRY_MSG})
                continue

            # ────────────────────────────────────────────────────────────
            # LOG: Parsed action + reasoning
            # ────────────────────────────────────────────────────────────
            reasoning = getattr(action, "reasoning", "")
            action_type = "tool_call" if isinstance(action, ToolCallAction) else "final_answer"
            parsed_log = f"[STEP {step_num}] Parsed action: {action_type}"
            print(parsed_log)
            logger.info("[%s] %s", req_id, parsed_log)
            if reasoning:
                reason_log = f"[STEP {step_num}] Reasoning: \"{reasoning}\""
                print(reason_log)
                logger.info("[%s] %s", req_id, reason_log)

            # ── FINAL ANSWER ──
            if isinstance(action, FinalAnswerAction):

                # ── Guard: must use tool but haven't yet ──
                if require_tool and tool_call_count == 0 and not is_final_step:
                    block_log = f"[STEP {step_num}] 🚫 BLOCKED: final_answer before tool use — must call a tool first"
                    print(block_log)
                    logger.warning("[%s] %s", req_id, block_log)
                    trace.append(LoopStep(
                        step=step_num,
                        action_type="blocked_final_answer",
                        reasoning=action.reasoning,
                        answer=action.answer,
                        error="final_answer blocked: must use tool first",
                    ))
                    messages.append({"role": "assistant", "content": raw})
                    messages.append({"role": "user", "content": _REQUIRE_TOOL_INJECTION})
                    continue

                # ────────────────────────────────────────────────────────
                # LOG: Final answer details
                # ────────────────────────────────────────────────────────
                final_log = (
                    f"\n{'=' * 70}\n"
                    f"[STEP {step_num}] ✅ FINAL ANSWER\n"
                    f"[STEP {step_num}] Answer: {action.answer}\n"
                    f"[STEP {step_num}] Steps taken: {step_num}\n"
                    f"[STEP {step_num}] Tools used: {tools_used or '(none)'}\n"
                    f"{'=' * 70}"
                )
                print(final_log)
                logger.info("[%s] %s", req_id, final_log)

                trace.append(LoopStep(
                    step=step_num,
                    action_type="final_answer",
                    reasoning=action.reasoning,
                    answer=action.answer,
                ))
                return AgentResult(
                    answer=action.answer,
                    source="agent_loop",
                    steps_taken=step_num,
                    tools_used=tools_used,
                    trace=trace,
                    llm_calls=llm_call_count,
                    cache_hits=cache_hits,
                )

            # ── TOOL CALL ──
            if isinstance(action, ToolCallAction):

                # ── Guard: tools disabled for this request (reasoning step) ──
                if not allow_tools:
                    block_log = f"[STEP {step_num}] 🚫 BLOCKED: tool_call disabled (reasoning step) — forcing final_answer"
                    print(block_log)
                    logger.warning("[%s] %s", req_id, block_log)
                    trace.append(LoopStep(
                        step=step_num,
                        action_type="blocked_tool_call",
                        reasoning=action.reasoning,
                        tool_name=action.tool_name,
                        error="Tool call blocked: tools disabled for reasoning step",
                    ))
                    return self._force_final_answer(messages, trace, tools_used, step_num, req_id, llm_call_count, cache_hits)

                # ── Guard: tool calls blocked on final step ──
                if is_final_step:
                    block_log = f"[STEP {step_num}] 🚫 BLOCKED: tool_call on final step — forcing final_answer"
                    print(block_log)
                    logger.warning("[%s] %s", req_id, block_log)
                    trace.append(LoopStep(
                        step=step_num,
                        action_type="blocked_tool_call",
                        reasoning=action.reasoning,
                        tool_name=action.tool_name,
                        error="Tool call blocked: final step",
                    ))
                    return self._force_final_answer(messages, trace, tools_used, step_num, req_id, llm_call_count, cache_hits)

                # ── Guard: tool call limit reached ──
                if tools_exhausted:
                    block_log = f"[STEP {step_num}] 🚫 BLOCKED: tool call limit ({self._max_tool_calls}) reached — forcing final_answer"
                    print(block_log)
                    logger.warning("[%s] %s", req_id, block_log)
                    trace.append(LoopStep(
                        step=step_num,
                        action_type="blocked_tool_call",
                        reasoning=action.reasoning,
                        tool_name=action.tool_name,
                        error=f"Tool call blocked: limit ({self._max_tool_calls}) reached",
                    ))
                    return self._force_final_answer(messages, trace, tools_used, step_num, req_id, llm_call_count, cache_hits)

                # ────────────────────────────────────────────────────────
                # LOG: Tool call details
                # ────────────────────────────────────────────────────────
                tool_input_str = json.dumps(action.tool_input, indent=2)
                call_log = (
                    f"\n[STEP {step_num}] 🔧 TOOL CALL\n"
                    f"[STEP {step_num}] Calling tool: {action.tool_name}\n"
                    f"[STEP {step_num}] Tool input: {tool_input_str}"
                )
                print(call_log)
                logger.info("[%s] %s", req_id, call_log)

                # ── Execute via registry, with duplicate cache ──
                cache_key = self._get_cache_key(action.tool_name, action.tool_input)
                if cache_key in self._tool_cache:
                    tool_result = self._tool_cache[cache_key]
                    is_error = tool_result.startswith("Error:")
                    cache_hits += 1
                    cache_log = (
                        f"\n[STEP {step_num}] ♻️ CACHE HIT — reusing result for "
                        f"'{action.tool_name}' ({len(tool_result)} chars)"
                    )
                    print(cache_log)
                    logger.info("[%s] %s", req_id, cache_log)
                else:
                    tool_result = self._registry.execute(action.tool_name, action.tool_input)
                    is_error = tool_result.startswith("Error:")
                    if not is_error:
                        self._tool_cache[cache_key] = tool_result

                # ────────────────────────────────────────────────────────
                # LOG: Full tool output (UNTRUNCATED)
                # ────────────────────────────────────────────────────────
                if is_error:
                    output_log = (
                        f"\n[STEP {step_num}] ❌ TOOL ERROR:\n"
                        f"{'-' * 50}\n"
                        f"{tool_result}\n"
                        f"{'-' * 50}"
                    )
                else:
                    output_log = (
                        f"\n[STEP {step_num}] TOOL OUTPUT ({len(tool_result)} chars):\n"
                        f"{'-' * 50}\n"
                        f"{tool_result}\n"
                        f"{'-' * 50}"
                    )
                print(output_log)
                logger.info("[%s] %s", req_id, output_log)

                tool_call_count += 1
                tools_used.append(action.tool_name)

                trace.append(LoopStep(
                    step=step_num,
                    action_type="tool_call",
                    reasoning=action.reasoning,
                    tool_name=action.tool_name,
                    tool_input=action.tool_input,
                    tool_result=tool_result[:500] if tool_result else None,
                    error=tool_result if is_error else None,
                ))

                # ── Feed result (or error) back to LLM ──
                messages.append({"role": "assistant", "content": raw})

                if is_error:
                    feedback = _TOOL_ERROR_TEMPLATE.format(error=tool_result)
                elif require_tool:
                    feedback = _TOOL_RESULT_EXTRACTION_TEMPLATE.format(
                        tool_name=action.tool_name,
                        result=tool_result,
                    )
                else:
                    feedback = _TOOL_RESULT_TEMPLATE.format(
                        tool_name=action.tool_name,
                        result=tool_result,
                    )

                messages.append({"role": "user", "content": feedback})

                # ────────────────────────────────────────────────────────
                # LOG: Context being sent to LLM on next iteration
                # ────────────────────────────────────────────────────────
                context_log = (
                    f"\n[STEP {step_num}] 📤 CONTEXT TO LLM (for next step):\n"
                    f"  User query: \"{query}\"\n"
                    f"  Tool result included: {action.tool_name} → "
                    f"{'ERROR' if is_error else f'{len(tool_result)} chars'}\n"
                    f"  Total messages in context: {len(messages)}"
                )
                print(context_log)
                logger.info("[%s] %s", req_id, context_log)
                continue

        # ── Loop exhausted (shouldn't reach here due to final-step enforcement) ──
        exhaust_log = f"\n[LOOP] ⚠️  Loop exhausted at step {self._max_steps} without final answer"
        print(exhaust_log)
        logger.warning("[%s] %s", req_id, exhaust_log)
        return self._force_final_answer(messages, trace, tools_used, self._max_steps, req_id, llm_call_count, cache_hits)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _inject_constraint(
        messages: list[dict[str, str]],
        constraint: str,
    ) -> list[dict[str, str]]:
        """Append a constraint to the system prompt for this step only.

        Returns a new list — does NOT mutate the original messages.
        """
        result = list(messages)
        # Find the system message and append the constraint
        for i, msg in enumerate(result):
            if msg["role"] == "system":
                result[i] = {
                    "role": "system",
                    "content": msg["content"] + constraint,
                }
                break
        return result

    def _force_final_answer(
        self,
        messages: list[dict[str, str]],
        trace: list[LoopStep],
        tools_used: list[str],
        step_num: int,
        req_id: str,
        llm_call_count: int = 0,
        cache_hits: int = 0,
    ) -> AgentResult:
        """Force the LLM to produce a final_answer from current context."""
        force_log = f"\n[FORCED] ⚡ Forcing final answer extraction (step {step_num})"
        print(force_log)
        logger.info("[%s] %s", req_id, force_log)

        forced_messages = list(messages)
        forced_messages.append({
            "role": "user",
            "content": (
                "You MUST now give your final answer based on everything discussed. "
                "No more tool calls are allowed. "
                '{"action": "final_answer", "reasoning": "...", "answer": "..."}'
            ),
        })

        try:
            raw = self._llm.chat(
                messages=forced_messages,
                temperature=0.1,
                max_tokens=2048,
                json_mode=True,
            )
            llm_call_count += 1

            forced_raw_log = (
                f"\n[FORCED] LLM RAW OUTPUT ({len(raw)} chars):\n"
                f"{'-' * 50}\n"
                f"{raw}\n"
                f"{'-' * 50}"
            )
            print(forced_raw_log)
            logger.info("[%s] %s", req_id, forced_raw_log)

            action = parse_llm_action(raw)
            if isinstance(action, FinalAnswerAction):
                forced_answer_log = (
                    f"\n{'=' * 70}\n"
                    f"[FORCED] ✅ FINAL ANSWER (forced)\n"
                    f"[FORCED] Reasoning: \"{action.reasoning}\"\n"
                    f"[FORCED] Answer: {action.answer}\n"
                    f"[FORCED] Steps taken: {step_num}\n"
                    f"[FORCED] Tools used: {tools_used or '(none)'}\n"
                    f"{'=' * 70}"
                )
                print(forced_answer_log)
                logger.info("[%s] %s", req_id, forced_answer_log)

                trace.append(LoopStep(
                    step=step_num,
                    action_type="forced_final_answer",
                    reasoning=action.reasoning,
                    answer=action.answer,
                ))
                return AgentResult(
                    answer=action.answer,
                    source="agent_loop",
                    steps_taken=step_num,
                    tools_used=tools_used,
                    trace=trace,
                    llm_calls=llm_call_count,
                    cache_hits=cache_hits,
                )
        except Exception as exc:
            err_log = f"[FORCED] ❌ Forced final answer failed: {exc}"
            print(err_log)
            logger.error("[%s] %s", req_id, err_log)

        fallback_log = f"[FORCED] ⚠️  Falling back — could not extract final answer"
        print(fallback_log)
        logger.warning("[%s] %s", req_id, fallback_log)

        trace.append(LoopStep(step=step_num, action_type="fallback", error="Could not extract final answer"))
        return AgentResult(
            answer="I was unable to complete the request. Please try again with a simpler question.",
            source="fallback",
            steps_taken=step_num,
            tools_used=tools_used,
            trace=trace,
            llm_calls=llm_call_count,
            cache_hits=cache_hits,
        )