"""Autonomous executor — goal-driven adaptive execution loop.

Handles complex multi-step tasks (e.g. "plan a 5-day trip to Japan")
with:
- Aspect decomposition: LLM breaks goal into ordered, prioritized aspects
- Adaptive execution: each step decided by LLM based on collected data
- Coverage gate: all required aspects must have data before finishing
- Structured synthesis: output organized by section, not paragraphs
- Tool dedup: prevents repeating the same tool call

This is a separate code path from the standard Executor, activated when
the classifier detects an "autonomous_task" query.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.agent.agent import Agent
from app.agent.executor import StepResult, PlannerExecutorResult
from app.llm.groq_client import GroqClient
from app.memory.memory_store import MemoryStore, MemoryEntry

logger = logging.getLogger(__name__)

# ── Limits ─────────────────────────────────────────────────────────────────

_MAX_AUTONOMOUS_STEPS = 12
_MAX_TOOL_CALLS = 6
_MAX_RETRIES_PER_ASPECT = 2
_MAX_ASPECTS = 7

# ── Prompts ────────────────────────────────────────────────────────────────

_DECOMPOSITION_PROMPT = """\
You are a task decomposition engine. Break the user's goal into ordered, \
prioritized aspects that must be researched or reasoned about.

RULES:
- Order aspects by dependency (e.g., flights before itinerary)
- Mark each as required (true) or optional (false)
- Provide a search_hint for aspects that need web search (null for reasoning-only aspects)
- Reasoning-only aspects (like "budget_breakdown", "itinerary", "comparison") synthesize from \
previously collected data — they don't need search
- Maximum 7 aspects
- Keep aspect names short and snake_case

Return ONLY valid JSON:
{{
  "aspects": [
    {{"name": "aspect_name", "priority": 1, "required": true, "search_hint": "search query or null"}},
    ...
  ]
}}

Examples:
- "plan a trip to Japan" → flights, hotels, places_to_visit, budget_breakdown, itinerary
- "compare React vs Vue" → react_features, vue_features, performance, community, recommendation
- "research AI trends 2026" → current_trends, key_players, breakthroughs, predictions, summary

User goal: "{goal}"

Return ONLY the JSON object:"""

_DECISION_PROMPT = """\
You are an execution controller. Based on the current state, decide the next action.

Goal: "{goal}"

## Aspects (ordered by priority):
{aspects_status}

## Data collected so far:
{collected_summary}

## Current aspect to work on: {current_aspect}
## Search hint: {search_hint}

RULES:
- If current aspect needs data and has a search_hint, use a tool step
- If current aspect has enough data, move to next
- If a search returned weak results, try a different search query
- Do NOT repeat the same search query
- For reasoning-only aspects, synthesize from collected data

Previous tool calls (DO NOT repeat these):
{previous_calls}

Return ONLY valid JSON:
{{
  "action": "search" or "reason" or "next_aspect",
  "reasoning": "why this decision",
  "search_query": "query if action=search, null otherwise"
}}"""

_STRUCTURED_SYNTHESIS_PROMPT = """\
You are a friendly, conversational assistant producing a structured deliverable.

Goal: "{goal}"

Organize the collected data into these sections. Use the EXACT section headers shown.
Be conversational and helpful — talk like a knowledgeable friend who did the research.

{sections_with_data}

RULES:
- Use ONLY the data provided above for each section
- Be specific — include names, prices, dates, and details from the data
- If a section has limited data, be honest: "I couldn't find much here, but..."
- Do NOT invent facts not present in the data
- Keep each section concise but informative
- Use bullet points, numbers, or short paragraphs as appropriate
- Add a brief intro and conclusion

Write the structured response now:"""


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class AspectSpec:
    """One aspect of a complex task to be researched or reasoned about."""
    name: str
    priority: int
    required: bool
    search_hint: str | None
    status: str = "pending"  # "pending" | "done" | "skipped" | "unavailable"


@dataclass
class AutonomousState:
    """Mutable state tracked across autonomous execution."""
    goal: str
    aspects: list[AspectSpec] = field(default_factory=list)
    collected_data: dict[str, list[str]] = field(default_factory=dict)
    completed_steps: list[StepResult] = field(default_factory=list)
    tools_called: set = field(default_factory=set)  # (tool_name, input) tuples
    step_descriptions: list[str] = field(default_factory=list)
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    total_steps: int = 0


# ── AutonomousExecutor ─────────────────────────────────────────────────────

class AutonomousExecutor:
    """Goal-driven autonomous execution with adaptive step planning.

    Pipeline:
    1. Decompose goal into prioritized aspects
    2. Execute aspects in priority order (search or reasoning)
    3. Coverage gate — ensure all required aspects have data
    4. Structured synthesis — organize output by section
    """

    def __init__(
        self,
        agent: Agent,
        llm: GroqClient,
        memory: MemoryStore | None = None,
    ) -> None:
        self._agent = agent
        self._llm = llm
        self._memory = memory

    def execute(
        self,
        query: str,
        profile_context: list[dict[str, Any]] | None = None,
        session_context: list[dict[str, Any]] | None = None,
    ) -> PlannerExecutorResult:
        """Execute a complex task autonomously.

        Args:
            query: The user's complex goal/task.
            profile_context: User profile from frontend.
            session_context: Session context from frontend.

        Returns:
            PlannerExecutorResult with structured answer.
        """
        exec_id = uuid.uuid4().hex[:8]
        logger.info("[%s] ═══ AUTONOMOUS START ═══ goal=%r", exec_id, query)

        state = AutonomousState(goal=query)

        # ── Phase 0: Decompose goal into aspects ──
        self._decompose_goal(state, exec_id)

        if not state.aspects:
            logger.warning("[%s] Decomposition failed — falling back to single step", exec_id)
            state.aspects = [
                AspectSpec(name="research", priority=1, required=True,
                           search_hint=query),
            ]

        aspects_msg = (
            f"[AUTONOMOUS] Decomposed into {len(state.aspects)} aspects: "
            + ", ".join(f"{a.name}({'req' if a.required else 'opt'})" for a in state.aspects)
        )
        print(aspects_msg)
        logger.info("[%s] %s", exec_id, aspects_msg)

        # ── Phase 1: Execute aspects in priority order ──
        self._execute_aspects(state, exec_id)

        # ── Phase 2: Coverage gate ──
        self._coverage_gate(state, exec_id)

        # ── Phase 3: Execute reasoning-only aspects ──
        self._execute_reasoning_aspects(state, exec_id)

        # ── Phase 4: Structured synthesis ──
        final_answer = self._structured_synthesis(state, exec_id)

        # ── Build result ──
        result = PlannerExecutorResult(
            answer=final_answer,
            plan=state.step_descriptions,
            step_results=[sr.to_dict() for sr in state.completed_steps],
            steps_taken=state.total_steps,
            tools_used=list({t[0] for t in state.tools_called}),
            source="autonomous_executor",
            confidence="medium",
            refinements=0,
            memory_used=False,
            memory_hits=0,
            decision="autonomous_task",
            llm_calls=state.total_llm_calls,
            steps_skipped=0,
            early_stopped=False,
            cache_hits=0,
        )

        logger.info(
            "[%s] ═══ AUTONOMOUS END ═══ steps=%d tools=%d llm_calls=%d aspects=%d",
            exec_id, state.total_steps, state.total_tool_calls,
            state.total_llm_calls, len(state.aspects),
        )
        return result

    # ── Phase 0: Decomposition ─────────────────────────────────────────

    def _decompose_goal(self, state: AutonomousState, exec_id: str) -> None:
        """Break the goal into ordered, prioritized aspects via LLM."""
        prompt = _DECOMPOSITION_PROMPT.format(goal=state.goal)

        try:
            raw = self._llm.chat(
                messages=[
                    {"role": "system", "content": "You are a task decomposition engine. Return only JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
                json_mode=True,
            )
            state.total_llm_calls += 1

            parsed = json.loads(raw)
            aspects_raw = parsed.get("aspects", [])

            for a in aspects_raw[:_MAX_ASPECTS]:
                state.aspects.append(AspectSpec(
                    name=str(a.get("name", "unknown")),
                    priority=int(a.get("priority", 99)),
                    required=bool(a.get("required", True)),
                    search_hint=a.get("search_hint"),
                ))

            # Sort by priority
            state.aspects.sort(key=lambda x: x.priority)

            logger.info(
                "[%s] [DECOMPOSITION] Parsed %d aspects from LLM",
                exec_id, len(state.aspects),
            )

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "[%s] [DECOMPOSITION] Failed to parse: %s", exec_id, exc,
            )

    # ── Phase 1: Adaptive execution ────────────────────────────────────

    def _execute_aspects(self, state: AutonomousState, exec_id: str) -> None:
        """Execute tool-based aspects in priority order."""

        for aspect in state.aspects:
            # Skip reasoning-only aspects (handled in phase 3)
            if aspect.search_hint is None:
                continue

            if state.total_steps >= _MAX_AUTONOMOUS_STEPS:
                limit_msg = f"[AUTONOMOUS] Step limit ({_MAX_AUTONOMOUS_STEPS}) reached"
                print(limit_msg)
                logger.warning("[%s] %s", exec_id, limit_msg)
                break

            if state.total_tool_calls >= _MAX_TOOL_CALLS:
                limit_msg = f"[AUTONOMOUS] Tool call limit ({_MAX_TOOL_CALLS}) reached"
                print(limit_msg)
                logger.warning("[%s] %s", exec_id, limit_msg)
                break

            self._execute_single_aspect(state, aspect, exec_id)

    def _execute_single_aspect(
        self,
        state: AutonomousState,
        aspect: AspectSpec,
        exec_id: str,
    ) -> None:
        """Execute a single tool-based aspect with retry logic."""
        retries = 0
        search_query = aspect.search_hint or aspect.name.replace("_", " ")

        while retries <= _MAX_RETRIES_PER_ASPECT:
            if state.total_tool_calls >= _MAX_TOOL_CALLS:
                break

            # Check for tool dedup
            call_key = ("web_search", search_query)
            if call_key in state.tools_called and retries > 0:
                # Generate alternative query
                search_query = f"{search_query} detailed guide recommendations"

            step_msg = (
                f"\n[AUTONOMOUS] Step {state.total_steps + 1} — "
                f"Aspect: {aspect.name} (priority {aspect.priority})\n"
                f"[AUTONOMOUS] → Search: {search_query}"
            )
            print(step_msg)
            logger.info("[%s] %s", exec_id, step_msg)

            # Build context for the agent
            context = (
                f'Goal: "{state.goal}"\n'
                f"Current aspect: {aspect.name}\n"
                f"Task: Search for information about this aspect.\n\n"
                f"Search query to use: {search_query}\n\n"
                f"After getting results, extract the most relevant facts as bullet points.\n"
                f"Focus on concrete details: names, prices, dates, ratings, addresses."
            )

            # Execute via agent
            self._agent.clear_cache()
            agent_result = self._agent.handle_full(
                context, allow_tools=True, require_tool=True,
            )
            state.total_llm_calls += agent_result.llm_calls
            state.total_steps += 1
            state.total_tool_calls += 1
            state.tools_called.add(call_key)

            # Record step
            success = agent_result.source != "error"
            step_result = StepResult(
                step=f"[{aspect.name}] {search_query}",
                type="tool",
                result=agent_result.answer,
                tools_used=agent_result.tools_used,
                success=success,
            )
            state.completed_steps.append(step_result)
            state.step_descriptions.append(
                f"Search for {aspect.name}: {search_query}"
            )

            # Store collected data
            if success and agent_result.answer and len(agent_result.answer.strip()) > 50:
                if aspect.name not in state.collected_data:
                    state.collected_data[aspect.name] = []
                state.collected_data[aspect.name].append(agent_result.answer)
                aspect.status = "done"

                done_msg = (
                    f"[AUTONOMOUS] ✅ Aspect '{aspect.name}' — "
                    f"collected {len(agent_result.answer)} chars"
                )
                print(done_msg)
                logger.info("[%s] %s", exec_id, done_msg)
                break  # Aspect done, move to next

            else:
                retries += 1
                retry_msg = (
                    f"[AUTONOMOUS] ⚠️ Weak result for '{aspect.name}' — "
                    f"retry {retries}/{_MAX_RETRIES_PER_ASPECT}"
                )
                print(retry_msg)
                logger.warning("[%s] %s", exec_id, retry_msg)

                # Try a different search query
                search_query = self._generate_alt_query(
                    state.goal, aspect.name, search_query,
                )

        # If all retries exhausted
        if aspect.status != "done":
            if aspect.required:
                aspect.status = "unavailable"
                unavail_msg = (
                    f"[AUTONOMOUS] ❌ Required aspect '{aspect.name}' "
                    f"marked unavailable after {retries} retries"
                )
                print(unavail_msg)
                logger.warning("[%s] %s", exec_id, unavail_msg)
            else:
                aspect.status = "skipped"

    def _generate_alt_query(
        self, goal: str, aspect_name: str, previous_query: str,
    ) -> str:
        """Generate an alternative search query for a failed aspect."""
        label = aspect_name.replace("_", " ")
        # Simple heuristic: add specificity
        alt = f"{label} for {goal} best options recommendations 2026"
        if alt == previous_query:
            alt = f"top {label} guide tips {goal}"
        return alt

    # ── Phase 2: Coverage gate ─────────────────────────────────────────

    def _coverage_gate(self, state: AutonomousState, exec_id: str) -> None:
        """Ensure all required aspects have data before synthesis."""
        missing = [
            a for a in state.aspects
            if a.required
            and a.search_hint is not None  # only tool aspects
            and a.name not in state.collected_data
        ]

        if not missing:
            gate_msg = "[AUTONOMOUS] ✅ Coverage gate passed — all required aspects covered"
            print(gate_msg)
            logger.info("[%s] %s", exec_id, gate_msg)
            return

        gate_msg = (
            f"[AUTONOMOUS] ⚠️ Coverage gate: {len(missing)} required aspects "
            f"missing: {[a.name for a in missing]}"
        )
        print(gate_msg)
        logger.warning("[%s] %s", exec_id, gate_msg)

        # Attempt to fill missing aspects (if limits allow)
        for aspect in missing:
            if state.total_tool_calls >= _MAX_TOOL_CALLS:
                break
            if state.total_steps >= _MAX_AUTONOMOUS_STEPS:
                break

            retry_msg = f"[AUTONOMOUS] 🔄 Coverage retry for: {aspect.name}"
            print(retry_msg)
            logger.info("[%s] %s", exec_id, retry_msg)

            aspect.search_hint = self._generate_alt_query(
                state.goal, aspect.name,
                aspect.search_hint or aspect.name.replace("_", " "),
            )
            self._execute_single_aspect(state, aspect, exec_id)

    # ── Phase 3: Reasoning aspects ─────────────────────────────────────

    def _execute_reasoning_aspects(
        self, state: AutonomousState, exec_id: str,
    ) -> None:
        """Execute reasoning-only aspects (budget, itinerary, comparison)."""

        reasoning_aspects = [
            a for a in state.aspects
            if a.search_hint is None and a.status == "pending"
        ]

        if not reasoning_aspects:
            return

        for aspect in reasoning_aspects:
            if state.total_steps >= _MAX_AUTONOMOUS_STEPS:
                break

            reason_msg = (
                f"\n[AUTONOMOUS] Step {state.total_steps + 1} — "
                f"Reasoning: {aspect.name}"
            )
            print(reason_msg)
            logger.info("[%s] %s", exec_id, reason_msg)

            # Build context with all collected data
            data_summary = self._format_collected_data(state)
            context = (
                f'Goal: "{state.goal}"\n\n'
                f"## Data collected so far:\n{data_summary}\n\n"
                f"Current task: Analyze the data above and create a "
                f"{aspect.name.replace('_', ' ')} section.\n"
                f"Use ONLY the data provided above. Be specific and detailed."
            )

            agent_result = self._agent.handle_full(
                context, allow_tools=False, require_tool=False,
            )
            state.total_llm_calls += agent_result.llm_calls
            state.total_steps += 1

            step_result = StepResult(
                step=f"[{aspect.name}] Reasoning/synthesis",
                type="reasoning",
                result=agent_result.answer,
                tools_used=[],
                success=agent_result.source != "error",
            )
            state.completed_steps.append(step_result)
            state.step_descriptions.append(f"Synthesize {aspect.name}")

            if agent_result.answer and len(agent_result.answer.strip()) > 20:
                if aspect.name not in state.collected_data:
                    state.collected_data[aspect.name] = []
                state.collected_data[aspect.name].append(agent_result.answer)
                aspect.status = "done"

    # ── Phase 4: Structured synthesis ──────────────────────────────────

    def _structured_synthesis(
        self, state: AutonomousState, exec_id: str,
    ) -> str:
        """Produce final structured output organized by aspect sections."""

        synth_msg = "[AUTONOMOUS] 📝 Generating structured synthesis..."
        print(synth_msg)
        logger.info("[%s] %s", exec_id, synth_msg)

        # Build sections with their data
        sections: list[str] = []
        for aspect in state.aspects:
            label = aspect.name.replace("_", " ").title()
            data_items = state.collected_data.get(aspect.name, [])

            if data_items:
                combined = "\n".join(data_items)
                # Truncate very long data
                if len(combined) > 2000:
                    combined = combined[:2000] + "..."
                sections.append(f"## {label}\nData:\n{combined}")
            elif aspect.required:
                sections.append(
                    f"## {label}\nData: No data available — "
                    f"could not find information for this section."
                )

        sections_with_data = "\n\n".join(sections)

        prompt = _STRUCTURED_SYNTHESIS_PROMPT.format(
            goal=state.goal,
            sections_with_data=sections_with_data,
        )

        try:
            answer = self._llm.generate_response(
                prompt, temperature=0.4, max_tokens=3000,
            )
            state.total_llm_calls += 1

            synth_done_msg = f"[AUTONOMOUS] ✅ Synthesis complete ({len(answer)} chars)"
            print(synth_done_msg)
            logger.info("[%s] %s", exec_id, synth_done_msg)

            return answer

        except Exception as exc:
            logger.error(
                "[%s] [AUTONOMOUS] Synthesis failed: %s", exec_id, exc,
            )
            # Fallback: dump raw data
            return self._fallback_synthesis(state)

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _format_collected_data(state: AutonomousState) -> str:
        """Format all collected data for injection into prompts."""
        if not state.collected_data:
            return "(no data collected yet)"

        parts: list[str] = []
        for aspect_name, items in state.collected_data.items():
            label = aspect_name.replace("_", " ").title()
            combined = "\n".join(items)
            # Truncate per-aspect
            if len(combined) > 1500:
                combined = combined[:1500] + "..."
            parts.append(f"### {label}\n{combined}")

        return "\n\n".join(parts)

    @staticmethod
    def _fallback_synthesis(state: AutonomousState) -> str:
        """Emergency fallback if synthesis LLM call fails."""
        parts = [f"# Results for: {state.goal}\n"]
        for aspect in state.aspects:
            label = aspect.name.replace("_", " ").title()
            data = state.collected_data.get(aspect.name, [])
            parts.append(f"## {label}")
            if data:
                parts.append("\n".join(data))
            else:
                parts.append("No data available.")
            parts.append("")
        return "\n".join(parts)
