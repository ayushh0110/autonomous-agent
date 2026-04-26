"""Executor module — runs a structured plan through the Agent, step by step.

Orchestrates:
1. Query classification (via Decision Layer)
2. Plan creation (via Planner, adaptive)
3. Sequential step execution (via Agent.handle_full)
4. Context passing between steps (compressed)
5. Final synthesis of all step results
6. Critic evaluation + refinement loop (Phase 3.2)
   - Severity-aware decisions
   - Regression guard on refinements
   - Confidence-driven loop control

Phase 4 — Adaptive execution:
7. Early stopping when intermediate results are strong
8. Dynamic plan adjustment (trim/expand)
9. Weak-result recovery (retry + reasoning fallback)
10. Execution metrics tracking
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.agent.agent import Agent, classify_query, QueryDecision
from app.agent.toolforge_router import toolforge_classify, is_toolforge_available
from app.agent.planner import Planner, PlanStep
from app.agent.critic import Critic, CriticResult, CriticIssue
from app.llm.groq_client import GroqClient
from app.memory.memory_store import MemoryStore, MemoryEntry

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

_DEFAULT_MAX_PLAN_STEPS = 5
_DEFAULT_MAX_EXECUTION_STEPS = 10
_DEFAULT_MAX_REFINEMENTS = 2
_CONTEXT_SUMMARY_LIMIT = 500  # chars passed per step result in context
_TOOL_CONTEXT_SUMMARY_LIMIT = 1500  # higher limit for tool steps to preserve extracted facts
_REGRESSION_LENGTH_RATIO = 0.7  # refined answer must be ≥ 70% of previous length

# ── Early stopping thresholds ─────────────────────────────────────────────
_EARLY_STOP_MIN_LENGTH = 200   # minimum chars in result to consider "strong"
_EARLY_STOP_MIN_ENTITIES = 3   # minimum distinct proper-noun entities
_EARLY_STOP_MIN_NUMBERS = 2    # minimum distinct numeric values
_MAX_DYNAMIC_STEPS = 2         # cap on steps added via plan adjustment

_SYNTHESIS_PROMPT = """\
You are a friendly, conversational assistant. Combine the step results below into \
a natural, warm answer to the user's question.

You MUST:
- Speak like a knowledgeable friend, NOT like a report or data dump
- Use natural phrases like "So from what I found..." or "Here's what I got..." \
or "Turns out..." to introduce information
- Include concrete details from the results: specific names, dates, numbers, \
events, statistics
- Remove redundancy between steps
- Do NOT introduce new facts, claims, or examples not present in the results
- Do NOT use generic filler statements like "AI is evolving rapidly" — \
every sentence must be traceable to the step results
- Do NOT mention steps, planning, or the execution process
- If the step results are weak or incomplete, be honest about it casually: \
"I couldn't find much on this, but here's what I got..."
- Keep it concise, engaging, and direct

Original query: "{query}"

Step results:
{step_results}

Write your conversational answer now:"""

_REFINEMENT_PROMPT = """\
Improve the following answer based on the critique provided.
Keep the conversational, friendly tone — the answer should sound like a person talking.

You MUST:
- Fix the issues identified by the critic
- Apply suggestions ONLY if they are consistent with the step results below
- IGNORE any suggestion that would introduce new or unsupported facts
- Use ONLY information from the original step results below
- Do NOT introduce new facts not present in the step results
- Do NOT discard correct parts of the previous answer
- Preserve the length, detail level, and conversational tone of the previous answer
- Write the improved answer directly — no meta-commentary

## Original Query
{query}

## Step Results (ground truth — this is your ONLY source of facts)
{step_results}

## Previous Answer
{previous_answer}

## Issues Found
{issues}

## Suggestions
{suggestions}

Write the improved conversational answer now:"""


# ── Data structures ────────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Outcome of executing one plan step."""
    step: str               # step description
    type: str               # "tool" | "reasoning"
    result: str             # full output from agent
    tools_used: list[str]   # tools used in this step
    success: bool           # whether step succeeded

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "type": self.type,
            "result": self.result,
            "tools_used": self.tools_used,
            "success": self.success,
        }


@dataclass
class ExecutionMetrics:
    """Tracks efficiency metrics across the execution pipeline."""
    tool_calls: int = 0        # actual tool executions (not cached)
    llm_calls: int = 0        # total LLM API calls across all phases
    steps_taken: int = 0
    steps_skipped: int = 0
    early_stopped: bool = False
    decision: str = "needs_search"
    cache_hits: int = 0        # tool calls served from cache
    dynamic_steps_added: int = 0  # steps added via plan adjustment


@dataclass
class ExecutionState:
    """Mutable state tracked across plan execution."""
    plan: list[PlanStep]
    current_step_index: int = 0
    step_results: list[StepResult] = field(default_factory=list)
    all_tools_used: list[str] = field(default_factory=list)
    memory_context: str | None = None  # injected memory from past queries
    profile_context: list[dict[str, Any]] | None = None  # user profile from frontend
    session_context: list[dict[str, Any]] | None = None  # session context from frontend
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)


@dataclass
class PlannerExecutorResult:
    """Final result returned from the Planner–Executor pipeline."""
    answer: str
    plan: list[str]             # step descriptions for API response
    step_results: list[dict]    # serialized StepResults
    steps_taken: int
    tools_used: list[str]
    source: str = "planner_executor"
    confidence: str = "medium"  # from Critic: "high" | "medium" | "low"
    refinements: int = 0        # number of refinement iterations applied
    memory_used: bool = False   # whether past memory was retrieved
    memory_hits: int = 0        # number of relevant memory entries found
    # ── Phase 4: Execution metrics ──
    decision: str = "needs_search"
    llm_calls: int = 0
    steps_skipped: int = 0
    early_stopped: bool = False
    cache_hits: int = 0


# ── Executor class ─────────────────────────────────────────────────────────

class Executor:
    """Runs a structured plan through the Agent with context passing,
    synthesis, critic evaluation, and refinement.

    Pipeline:
        query → classify → Planner (adaptive) → execute each step
              → early stop? → synthesize → Critic → refine → final answer
    """

    def __init__(
        self,
        agent: Agent,
        llm: GroqClient,
        max_plan_steps: int = _DEFAULT_MAX_PLAN_STEPS,
        max_execution_steps: int = _DEFAULT_MAX_EXECUTION_STEPS,
        max_refinements: int = _DEFAULT_MAX_REFINEMENTS,
        memory: MemoryStore | None = None,
        autonomous_executor: Any | None = None,
    ) -> None:
        self._agent = agent
        self._llm = llm
        self._planner = Planner(llm=llm, max_plan_steps=max_plan_steps)
        self._critic = Critic(llm=llm)
        self._max_execution_steps = max_execution_steps
        self._max_refinements = max_refinements
        self._memory = memory
        self._autonomous = autonomous_executor

    # ── Public API ─────────────────────────────────────────────────────

    def execute(
        self,
        query: str,
        profile_context: list[dict[str, Any]] | None = None,
        session_context: list[dict[str, Any]] | None = None,
    ) -> PlannerExecutorResult:
        """Full pipeline: classify → plan → execute → synthesize → critique → refine.

        Args:
            query: Raw user input.
            profile_context: User profile entries from frontend localStorage.
            session_context: Session context entries from frontend sessionStorage.

        Returns:
            PlannerExecutorResult with answer, plan, and metadata.
        """
        exec_id = uuid.uuid4().hex[:8]
        logger.info("[%s] ═══ EXECUTOR START ═══ query=%r", exec_id, query)

        # ── Phase 0: Memory retrieval ──
        memory_entries: list[MemoryEntry] = []
        memory_context: str | None = None
        if self._memory:
            memory_entries = self._memory.retrieve(query)
            if memory_entries:
                memory_context = self._format_memory_context(memory_entries)
                mem_msg = (
                    f"[MEMORY] Retrieved {len(memory_entries)} relevant "
                    f"entries for query"
                )
                print(mem_msg)
                logger.info("[%s] %s", exec_id, mem_msg)

        # ── Phase 0.5: Query classification (Decision Layer) ──
        # Try ToolForge model-based router first (if enabled + GPU available),
        # fall back to heuristic classifier.
        decision = None
        router_source = "heuristic"

        if is_toolforge_available():
            decision = toolforge_classify(
                query=query,
                memory_hits=len(memory_entries),
                has_memory=bool(self._memory and memory_entries),
            )
            if decision is not None:
                router_source = "toolforge"

        if decision is None:
            decision = classify_query(
                query=query,
                memory_hits=len(memory_entries),
                has_memory=bool(self._memory and memory_entries),
            )

        decision_msg = (
            f"[DECISION] type={decision.decision_type} | "
            f"confidence={decision.confidence:.2f} | "
            f"router={router_source} | "
            f"reason={decision.reasoning}"
        )
        print(decision_msg)
        logger.info("[%s] %s", exec_id, decision_msg)

        # ── Phase 1: Plan (adaptive based on decision) ──
        plan = self._planner.create_plan(query, decision=decision.decision_type)

        # ── Autonomous delegation: complex tasks go to AutonomousExecutor ──
        if decision.decision_type == "autonomous_task" and self._autonomous:
            auto_msg = "[EXECUTOR] 🚀 Delegating to AutonomousExecutor"
            print(auto_msg)
            logger.info("[%s] %s", exec_id, auto_msg)
            return self._autonomous.execute(
                query,
                profile_context=profile_context,
                session_context=session_context,
            )

        # ── Phase 2: Execute (adaptive with early stopping) ──
        state = ExecutionState(
            plan=plan,
            memory_context=memory_context,
            profile_context=profile_context,
            session_context=session_context,
        )
        state.metrics.decision = decision.decision_type

        # ── Request-scoped cache: clear stale results from prior requests ──
        self._agent.clear_cache()

        # Use while-loop so plan mutations are visible
        step_idx = 0
        while step_idx < len(state.plan):
            if step_idx >= self._max_execution_steps:
                logger.warning(
                    "[%s] [EXECUTOR] Execution step limit (%d) reached, stopping",
                    exec_id, self._max_execution_steps,
                )
                break

            plan_step = state.plan[step_idx]
            state.current_step_index = step_idx
            self._execute_step(query, state, plan_step, step_idx, exec_id)

            # ── Early stopping check ──
            should_stop, stop_reason = self._should_stop_early(state, step_idx)
            if should_stop:
                skipped = len(state.plan) - step_idx - 1
                state.metrics.steps_skipped += skipped
                state.metrics.early_stopped = True
                stop_msg = (
                    f"[EARLY-STOP] ⚡ Stopping early after step {step_idx + 1} — "
                    f"{stop_reason} (skipping {skipped} remaining steps)"
                )
                print(stop_msg)
                logger.info("[%s] %s", exec_id, stop_msg)
                break

            # ── Dynamic plan adjustment ──
            self._maybe_adjust_plan(state, step_idx, exec_id)

            step_idx += 1

        state.metrics.steps_taken = len(state.step_results)

        # ── Fast-path for direct answers: skip synthesis + critic ──
        # Simple queries like "2+2", "hello", "what is X" already have
        # a concise answer from the LLM. Running synthesis + critic just
        # bloats them unnecessarily.
        plan_descriptions = [ps.step for ps in plan]
        serialized_steps = [sr.to_dict() for sr in state.step_results]

        if decision.decision_type == "direct_answer" and state.step_results:
            final_answer = state.step_results[-1].result
            confidence = "high"
            refinements = 0
            fast_msg = (
                "[EXECUTOR] ⚡ Direct answer — skipping synthesis/critic"
            )
            print(fast_msg)
            logger.info("[%s] %s", exec_id, fast_msg)
        else:
            # ── Phase 3: Synthesize ──
            final_answer = self._synthesize(query, state, exec_id)
            state.metrics.llm_calls += 1  # synthesis LLM call

            # ── Phase 3.5: Generic answer safety check ──
            final_answer = self._generic_answer_check(
                query, final_answer, state, exec_id,
            )

            # ── Phase 4: Critic evaluate + refine ──
            final_answer, confidence, refinements = self._critique_and_refine(
                query=query,
                answer=final_answer,
                step_results=serialized_steps,
                plan_steps=plan_descriptions,
                state=state,
                exec_id=exec_id,
            )

        # ── Phase 5: Memory storage (conditional) ──
        self._store_in_memory(query, final_answer, state, confidence, exec_id)

        result = PlannerExecutorResult(
            answer=final_answer,
            plan=plan_descriptions,
            step_results=serialized_steps,
            steps_taken=state.metrics.steps_taken,
            tools_used=list(dict.fromkeys(state.all_tools_used)),
            source="planner_executor",
            confidence=confidence,
            refinements=refinements,
            memory_used=bool(memory_entries),
            memory_hits=len(memory_entries),
            decision=state.metrics.decision,
            llm_calls=state.metrics.llm_calls,
            steps_skipped=state.metrics.steps_skipped,
            early_stopped=state.metrics.early_stopped,
            cache_hits=state.metrics.cache_hits,
        )

        logger.info(
            "[%s] ═══ EXECUTOR END ═══ steps=%d tools=%s confidence=%s "
            "refinements=%d memory_used=%s decision=%s llm_calls=%d "
            "steps_skipped=%d early_stopped=%s cache_hits=%d",
            exec_id, result.steps_taken, result.tools_used,
            result.confidence, result.refinements,
            result.memory_used, result.decision, result.llm_calls,
            result.steps_skipped, result.early_stopped, result.cache_hits,
        )
        return result

    # ── Step execution ─────────────────────────────────────────────────

    def _execute_step(
        self,
        query: str,
        state: ExecutionState,
        plan_step: PlanStep,
        step_index: int,
        exec_id: str,
    ) -> None:
        """Execute a single plan step, with retry and weak-result recovery."""

        step_num = step_index + 1
        total = len(state.plan)
        # Always allow tools — the LLM decides whether to use one.
        # Tool-typed steps hint that a tool is preferred, but reasoning
        # steps can also reach for a tool when it would help (calculator,
        # wikipedia, dictionary, etc.)
        allow_tools = True
        require_tool = (plan_step.type == "tool")

        step_header = (
            f"\n[EXECUTOR] Step {step_num}/{total} "
            f"[{plan_step.type}] allow_tools={allow_tools}\n"
            f"[EXECUTOR] → {plan_step.step}"
        )
        print(step_header)
        logger.info("[%s] %s", exec_id, step_header)

        context = self._build_context(query, state, plan_step)

        # ── First attempt ──
        agent_result = self._agent.handle_full(
            context, allow_tools=allow_tools, require_tool=require_tool,
        )
        state.metrics.llm_calls += agent_result.llm_calls
        state.metrics.cache_hits += agent_result.cache_hits
        if agent_result.tools_used:
            state.metrics.tool_calls += len(agent_result.tools_used)

        # ── Retry once on error ──
        if agent_result.source == "error":
            retry_msg = f"[EXECUTOR] Step {step_num} failed, retrying once..."
            print(retry_msg)
            logger.warning("[%s] %s", exec_id, retry_msg)
            agent_result = self._agent.handle_full(
                context, allow_tools=allow_tools, require_tool=require_tool,
            )
            state.metrics.llm_calls += agent_result.llm_calls
            state.metrics.cache_hits += agent_result.cache_hits
            if agent_result.tools_used:
                state.metrics.tool_calls += len(agent_result.tools_used)

        # ── Weak-result recovery for tool steps ──
        if (
            plan_step.type == "tool"
            and agent_result.source != "error"
            and self._is_weak_tool_result(agent_result.answer)
        ):
            weak_msg = f"[EXECUTOR] Step {step_num} returned weak result — retrying with enhanced context"
            print(weak_msg)
            logger.warning("[%s] %s", exec_id, weak_msg)

            retry_context = context + (
                f"\n\n⚠️ IMPORTANT: The previous search returned weak/incomplete results:\n"
                f"\"{agent_result.answer[:200]}\"\n\n"
                "You MUST use a DIFFERENT, more specific search query this time.\n"
                "Strategies:\n"
                "- Add specific entity names, product names, or organization names\n"
                "- Add year or date constraints (e.g., '2025', '2026')\n"
                "- Use a narrower, more focused search term\n"
                "Do NOT repeat the same search query."
            )
            agent_result = self._agent.handle_full(
                retry_context, allow_tools=True, require_tool=True,
            )
            state.metrics.llm_calls += agent_result.llm_calls
            state.metrics.cache_hits += agent_result.cache_hits
            if agent_result.tools_used:
                state.metrics.tool_calls += len(agent_result.tools_used)

            # If still weak, fall back to reasoning
            if self._is_weak_tool_result(agent_result.answer):
                fallback_msg = f"[EXECUTOR] Step {step_num} still weak — falling back to reasoning"
                print(fallback_msg)
                logger.warning("[%s] %s", exec_id, fallback_msg)
                agent_result = self._agent.handle_full(
                    context, allow_tools=False, require_tool=False,
                )
                state.metrics.llm_calls += agent_result.llm_calls

        # ── Record result ──
        success = agent_result.source != "error"
        step_result = StepResult(
            step=plan_step.step,
            type=plan_step.type,
            result=agent_result.answer,
            tools_used=agent_result.tools_used,
            success=success,
        )
        state.step_results.append(step_result)
        state.all_tools_used.extend(agent_result.tools_used)

        status = "✓" if success else "✗ (failed)"
        memory_tag = " [memory-injected]" if state.memory_context else ""
        done_msg = (
            f"[EXECUTOR] Step {step_num} complete {status} "
            f"(tools: {agent_result.tools_used or 'none'}) "
            f"(result: {len(agent_result.answer)} chars){memory_tag}"
        )
        print(done_msg)
        logger.info("[%s] %s", exec_id, done_msg)

    # ── Early stopping ─────────────────────────────────────────────────

    @staticmethod
    def _should_stop_early(
        state: ExecutionState,
        step_index: int,
    ) -> tuple[bool, str]:
        """Check if remaining steps can be skipped.

        Conditions for early stop:
        1. All remaining steps are 'reasoning' type (no pending tool calls)
        2. Latest step was successful
        3. Latest result is strong:
           - Length > 200 chars
           - At least 3 distinct proper-noun entities
           - At least 2 distinct numeric values
        """
        remaining = state.plan[step_index + 1:]
        if not remaining:
            return False, ""

        # Only stop if remaining steps are all reasoning
        if not all(s.type == "reasoning" for s in remaining):
            return False, "remaining steps include tool calls"

        latest = state.step_results[-1]
        if not latest.success:
            return False, "latest step failed"

        result = latest.result
        has_detail = len(result) > _EARLY_STOP_MIN_LENGTH

        # Count distinct proper-noun entities (e.g. "OpenAI", "Google")
        entities = set(re.findall(r'[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*', result))
        has_enough_entities = len(entities) >= _EARLY_STOP_MIN_ENTITIES

        # Count distinct numeric values (e.g. "2026", "$4.5B", "150")
        numbers = set(re.findall(r'\b\d[\d,.]*\b', result))
        has_enough_numbers = len(numbers) >= _EARLY_STOP_MIN_NUMBERS

        if has_detail and has_enough_entities and has_enough_numbers:
            return True, (
                f"strong intermediate result with {len(entities)} entities "
                f"and {len(numbers)} numeric values"
            )

        return False, "result not strong enough for early stop"

    # ── Dynamic plan adjustment ────────────────────────────────────────

    def _maybe_adjust_plan(
        self,
        state: ExecutionState,
        step_index: int,
        exec_id: str,
    ) -> None:
        """Adjust remaining plan based on intermediate results.

        Rules:
        - Strong reasoning result (after tool step) + multiple reasoning remaining
          → trim to single synthesis step
        - Weak tool result + no reasoning remaining → append fallback reasoning
          (capped at _MAX_DYNAMIC_STEPS additions)
        """
        latest = state.step_results[-1]
        remaining = state.plan[step_index + 1:]

        if not remaining:
            return

        # ── Strong result after tool → trim redundant reasoning ──
        # Only trim when latest completed step is reasoning AND the
        # step before it was a tool step (ensures at least one
        # reasoning step has processed the tool output before trimming)
        if (
            latest.type == "reasoning"
            and latest.success
            and len(state.step_results) >= 2
            and state.step_results[-2].type == "tool"
        ):
            result = latest.result
            is_strong = (
                len(result) > 300
                and bool(re.search(r'\d', result))
                and bool(re.search(r'[A-Z][a-z]{2,}', result))
            )

            if is_strong and all(s.type == "reasoning" for s in remaining) and len(remaining) > 1:
                trim_count = len(remaining) - 1
                synth_step = PlanStep(
                    step="Synthesize the key findings from the search results into a comprehensive answer",
                    type="reasoning",
                )
                state.plan[step_index + 1:] = [synth_step]
                state.metrics.steps_skipped += trim_count
                adjust_msg = (
                    f"[PLAN-ADJUST] ✂️ Trimmed {trim_count} redundant reasoning steps "
                    f"→ single synthesis step"
                )
                print(adjust_msg)
                logger.info("[%s] %s", exec_id, adjust_msg)
                return

        # ── Weak tool result → append fallback reasoning (capped) ──
        if latest.type == "tool" and latest.success and self._is_weak_tool_result(latest.result):
            has_reasoning = any(s.type == "reasoning" for s in remaining)
            if not has_reasoning and state.metrics.dynamic_steps_added < _MAX_DYNAMIC_STEPS:
                fallback = PlanStep(
                    step="Provide the best answer using available information and general knowledge",
                    type="reasoning",
                )
                state.plan.append(fallback)
                state.metrics.dynamic_steps_added += 1
                adjust_msg = (
                    f"[PLAN-ADJUST] ➕ Added fallback reasoning step after weak tool result "
                    f"(dynamic additions: {state.metrics.dynamic_steps_added}/{_MAX_DYNAMIC_STEPS})"
                )
                print(adjust_msg)
                logger.info("[%s] %s", exec_id, adjust_msg)
            elif not has_reasoning:
                cap_msg = (
                    f"[PLAN-ADJUST] ⛔ Dynamic step cap reached "
                    f"({_MAX_DYNAMIC_STEPS}) — not adding fallback"
                )
                print(cap_msg)
                logger.warning("[%s] %s", exec_id, cap_msg)

    # ── Weak result detection ──────────────────────────────────────────

    @staticmethod
    def _is_weak_tool_result(result: str) -> bool:
        """Check if a tool result is too weak to be useful."""
        if not result or len(result.strip()) < 50:
            return True
        if result.startswith("Error:"):
            return True
        if result.startswith("No search results"):
            return True
        # Check for minimal factual content
        has_numbers = bool(re.search(r'\d', result))
        has_proper_nouns = bool(re.search(r'[A-Z][a-z]{2,}', result))
        if not has_numbers and not has_proper_nouns:
            return True
        return False

    # ── Context building ───────────────────────────────────────────────

    @staticmethod
    def _build_context(
        query: str,
        state: ExecutionState,
        current_step: PlanStep,
    ) -> str:
        """Build the context message for the current step.

        Includes the original query, user profile, session context,
        compressed previous step results, and the current step instruction.
        """
        parts = [f'Original user query: "{query}"']

        # ── Inject user profile (structured) ──
        profile_block = Executor._format_profile_context(state.profile_context)
        if profile_block:
            parts.append(f"\n{profile_block}")

        # ── Inject session context (structured) ──
        session_block = Executor._format_session_context(state.session_context)
        if session_block:
            parts.append(f"\n{session_block}")

        # ── Inject semantic memory context (existing RAG memory) ──
        if state.memory_context:
            parts.append(f"\n{state.memory_context}")

        # ── Memory enforcement rules (if any memory is present) ──
        has_any_memory = (
            profile_block or session_block or state.memory_context
        )
        if has_any_memory:
            parts.append("\n## MEMORY USAGE RULES (MANDATORY)")
            parts.append(
                "- If the user's profile contains relevant information (name, preferences, habits), "
                "you MUST incorporate it naturally into your response."
            )
            parts.append(
                "- Use session context to resolve references like 'those two', "
                "'the one I mentioned', etc."
            )
            parts.append(
                "- Do NOT repeat memory back verbatim — weave it into your answer naturally."
            )
            parts.append(
                "- If profile data conflicts with new information in the current query, "
                "prioritize the current query but acknowledge both."
            )
            parts.append(
                "- Semantic memory (past queries) is an OPTIONAL HINT, not ground truth."
            )
            parts.append(
                "- If tool results CONFLICT with any memory, ALWAYS trust tool results."
            )

        if state.step_results:
            parts.append("\nPrevious steps completed:")
            for i, sr in enumerate(state.step_results, 1):
                # Use higher limit for tool steps to preserve extracted facts
                limit = _TOOL_CONTEXT_SUMMARY_LIMIT if sr.type == "tool" else _CONTEXT_SUMMARY_LIMIT
                summary = sr.result[:limit]
                if len(sr.result) > limit:
                    summary += "..."
                status = "✓" if sr.success else "✗"
                parts.append(f"  {i}. [{sr.type}] {sr.step} → [{status}] {summary}")

        parts.append("\nCurrent step to execute:")
        parts.append(f"  [{current_step.type}] {current_step.step}")
        parts.append("\nFocus ONLY on completing this specific step.")
        parts.append("Do NOT jump ahead to other steps.")

        # ── Step-type-specific rules ──
        if current_step.type == "tool":
            parts.append("\n## EXTRACTION RULES (MANDATORY)")
            parts.append("- You MUST call a tool to retrieve external data")
            parts.append("- After receiving results, extract ONLY concrete facts")
            parts.append("- Output as bullet points: specific names, dates, numbers, events")
            parts.append("- Do NOT summarize, infer trends, or generate a narrative")
            parts.append("- Do NOT produce a final answer without calling a tool first")
        elif current_step.type == "reasoning" and state.step_results:
            parts.append("\n## GROUNDING RULES (MANDATORY)")
            parts.append("- Use ONLY the information from the previous step results above")
            parts.append("- Do NOT introduce new facts not present in those results")
            parts.append("- Do NOT rely on prior knowledge or training data")

        return "\n".join(parts)

    @staticmethod
    def _format_profile_context(
        profile: list[dict[str, Any]] | None,
    ) -> str:
        """Format frontend profile memory into a structured block.

        Output format:
            ## USER PROFILE (use this when relevant to the query):
            - Name: Ayush
            - Role: Developer
        """
        if not profile:
            return ""

        lines = ["## USER PROFILE (use this when relevant to the query):"]
        for entry in profile:
            key = entry.get("key", "unknown")
            value = entry.get("value", "")
            # Format key nicely
            label = key.replace("_", " ").title()
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            lines.append(f"- {label}: {value}")

        return "\n".join(lines)

    @staticmethod
    def _format_session_context(
        session: list[dict[str, Any]] | None,
    ) -> str:
        """Format frontend session memory into a structured block.

        Output format:
            ## SESSION CONTEXT (recent conversation references):
            - [task] Planning a trip to Miami
            - [reference] Comparing Hotel A vs Hotel B
        """
        if not session:
            return ""

        lines = ["## SESSION CONTEXT (recent conversation references):"]
        for entry in session:
            intent = entry.get("intent", "reference")
            value = entry.get("value", "")
            lines.append(f"- [{intent}] {value}")

        return "\n".join(lines)

    # ── Synthesis ──────────────────────────────────────────────────────

    def _synthesize(
        self,
        query: str,
        state: ExecutionState,
        exec_id: str,
    ) -> str:
        """Synthesize all step results into a single coherent final answer."""

        synth_msg = f"[SYNTHESIZER] Generating final answer from {len(state.step_results)} step results..."
        print(synth_msg)
        logger.info("[%s] %s", exec_id, synth_msg)

        # Format step results for the synthesis prompt
        formatted: list[str] = []
        for i, sr in enumerate(state.step_results, 1):
            status = "✓" if sr.success else "✗"
            formatted.append(f"{i}. [{status}] [{sr.type}] {sr.step}\n   {sr.result}")

        prompt = _SYNTHESIS_PROMPT.format(
            query=query,
            step_results="\n\n".join(formatted),
        )

        try:
            answer = self._llm.generate_response(prompt, temperature=0.3, max_tokens=2048)

            done_msg = f"[SYNTHESIZER] Final answer generated ({len(answer)} chars)"
            print(done_msg)
            logger.info("[%s] %s", exec_id, done_msg)
            return answer

        except Exception as exc:
            err_msg = f"[SYNTHESIZER] Synthesis failed: {exc} — falling back to concatenation"
            print(err_msg)
            logger.error("[%s] %s", exec_id, err_msg, exc_info=True)

            # Fallback: concatenate successful step results
            fallback = "\n\n".join(
                sr.result for sr in state.step_results if sr.success
            )
            return fallback or "I was unable to generate a complete answer. Please try again."

    # ── Generic answer safety check ───────────────────────────────────

    def _generic_answer_check(
        self,
        query: str,
        answer: str,
        state: ExecutionState,
        exec_id: str,
    ) -> str:
        """Lightweight heuristic: if the answer lacks concrete details,
        trigger one pre-refinement before the critic evaluates it."""

        if not self._is_generic_answer(answer):
            return answer

        safety_msg = (
            "[SAFETY] \u26a0\ufe0f  Generic answer detected (lacks concrete details) "
            "\u2014 triggering pre-refinement"
        )
        print(safety_msg)
        logger.warning("[%s] %s", exec_id, safety_msg)

        synthetic_critic = CriticResult(
            is_valid=False,
            issues=[CriticIssue(
                type="specificity",
                severity="high",
                detail="Answer lacks concrete details (names, numbers, dates, facts) from step results",
            )],
            suggestions=[
                "Replace generic statements with specific facts from the step results",
                "Include names, numbers, dates, and sources from the search results",
            ],
            confidence="low",
        )

        refined = self._refine_answer(
            query=query,
            previous_answer=answer,
            critic_result=synthetic_critic,
            state=state,
            exec_id=exec_id,
        )

        if refined is not None:
            ok_msg = f"[SAFETY] \u2705 Pre-refinement applied ({len(refined)} chars)"
            print(ok_msg)
            logger.info("[%s] %s", exec_id, ok_msg)
            return refined

        skip_msg = "[SAFETY] Pre-refinement failed or regressed \u2014 keeping original"
        print(skip_msg)
        logger.warning("[%s] %s", exec_id, skip_msg)
        return answer

    @staticmethod
    def _is_generic_answer(answer: str) -> bool:
        """Return True if the answer appears to lack concrete details."""
        if not answer or len(answer.strip()) < 50:
            return True

        text = answer.strip()

        has_numbers = bool(re.search(r'\d', text))
        has_specific_markers = bool(re.search(
            r'(?:according to|source:|reported|announced|published|'
            r'released|launched|named|called|titled|founded|developed by|'
            r'created by|introduced|unveiled|percent|%|\$)',
            text, re.IGNORECASE,
        ))

        return not has_numbers and not has_specific_markers

    # ── Critic evaluation + refinement loop ────────────────────────────

    def _critique_and_refine(
        self,
        query: str,
        answer: str,
        step_results: list[dict[str, Any]],
        plan_steps: list[str],
        state: ExecutionState,
        exec_id: str,
    ) -> tuple[str, str, int]:
        """Run the Critic → refine loop with severity-aware decisions."""
        best_answer = answer
        confidence = "medium"
        refinement_count = 0

        for attempt in range(1, self._max_refinements + 1):

            # ── Critic evaluation ──
            critic_header = (
                f"\n{'=' * 70}\n"
                f"[CRITIC] Evaluation round {attempt}/{self._max_refinements}\n"
                f"{'=' * 70}"
            )
            print(critic_header)
            logger.info("[%s] %s", exec_id, critic_header)

            critic_result = self._critic.evaluate(
                query=query,
                answer=best_answer,
                step_results=step_results,
                plan_steps=plan_steps,
            )
            state.metrics.llm_calls += 1  # critic LLM call
            confidence = critic_result.confidence

            # ── Decision: should we refine? ──
            should_refine, reason = self._should_refine(critic_result, attempt)

            if not should_refine:
                accept_msg = (
                    f"[CRITIC] ✅ Answer ACCEPTED on round {attempt} "
                    f"(confidence: {confidence}) — {reason}"
                )
                print(accept_msg)
                logger.info("[%s] %s", exec_id, accept_msg)
                break

            # ── Must refine ──
            refine_msg = (
                f"[CRITIC] ❌ Answer NEEDS REFINEMENT on round {attempt} — {reason}\n"
                f"[CRITIC] Issues ({len(critic_result.issues)}):"
            )
            for issue in critic_result.issues:
                refine_msg += f"\n    [{issue.severity.upper()}] [{issue.type}] {issue.detail}"
            refine_msg += f"\n[CRITIC] Suggestions: {critic_result.suggestions}"
            print(refine_msg)
            logger.info("[%s] %s", exec_id, refine_msg)

            refined = self._refine_answer(
                query=query,
                previous_answer=best_answer,
                critic_result=critic_result,
                state=state,
                exec_id=exec_id,
            )

            if refined is not None:
                best_answer = refined
                refinement_count += 1
                refined_msg = (
                    f"[REFINEMENT] ✅ Refinement {refinement_count} complete "
                    f"({len(best_answer)} chars)"
                )
                print(refined_msg)
                logger.info("[%s] %s", exec_id, refined_msg)
            else:
                fail_msg = (
                    f"[REFINEMENT] ⚠️ Refinement failed on round {attempt} — "
                    f"keeping previous best answer"
                )
                print(fail_msg)
                logger.warning("[%s] %s", exec_id, fail_msg)
                break  # Don't keep trying if refinement itself fails

        else:
            # Loop exhausted without a valid answer
            exhaust_msg = (
                f"[CRITIC] ⚠️ Max refinements ({self._max_refinements}) reached — "
                f"returning best available answer (confidence: {confidence})"
            )
            print(exhaust_msg)
            logger.warning("[%s] %s", exec_id, exhaust_msg)

        return best_answer, confidence, refinement_count

    @staticmethod
    def _should_refine(critic_result: CriticResult, attempt: int) -> tuple[bool, str]:
        """Decide whether to refine based on severity, validity, and confidence."""
        if critic_result.has_high_severity():
            high_count = sum(1 for i in critic_result.issues if i.severity == "high")
            return True, f"{high_count} high-severity issue(s) found"

        if critic_result.has_only_low_severity():
            return False, "only low-severity issues remain"

        if critic_result.is_valid:
            return False, "critic marked as valid"

        if critic_result.confidence == "low":
            return True, "confidence is low"

        if not critic_result.is_valid:
            return True, "critic marked as invalid with medium-severity issues"

        return False, "no actionable issues"

    def _refine_answer(
        self,
        query: str,
        previous_answer: str,
        critic_result: CriticResult,
        state: ExecutionState,
        exec_id: str,
    ) -> str | None:
        """Attempt to refine the answer using critic feedback."""
        logger.info("[%s] [REFINEMENT] Generating refined answer...", exec_id)

        formatted_steps: list[str] = []
        for i, sr in enumerate(state.step_results, 1):
            status = "✓" if sr.success else "✗"
            formatted_steps.append(
                f"{i}. [{status}] [{sr.type}] {sr.step}\n   {sr.result}"
            )

        issues_lines: list[str] = []
        for issue in critic_result.issues:
            issues_lines.append(f"• [{issue.severity.upper()}] [{issue.type}] {issue.detail}")
        issues_text = "\n".join(issues_lines) or "(none)"

        suggestions_text = "\n".join(f"→ {s}" for s in critic_result.suggestions) or "(none)"

        prompt = _REFINEMENT_PROMPT.format(
            query=query,
            step_results="\n\n".join(formatted_steps),
            previous_answer=previous_answer,
            issues=issues_text,
            suggestions=suggestions_text,
        )

        try:
            refined = self._llm.generate_response(
                prompt, temperature=0.3, max_tokens=2048,
            )
            state.metrics.llm_calls += 1  # refinement LLM call

            if not refined or len(refined.strip()) < 20:
                logger.warning(
                    "[%s] [REFINEMENT] Refined answer too short (%d chars), discarding",
                    exec_id, len(refined) if refined else 0,
                )
                return None

            # ── Regression guard ──
            previous_len = len(previous_answer.strip())
            refined_len = len(refined.strip())

            if previous_len > 0 and refined_len < _REGRESSION_LENGTH_RATIO * previous_len:
                regression_msg = (
                    f"[REFINEMENT] ⚠️ REGRESSION DETECTED — refined answer is too short "
                    f"({refined_len} chars vs previous {previous_len} chars, "
                    f"ratio {refined_len / previous_len:.2f} < {_REGRESSION_LENGTH_RATIO}). "
                    f"Discarding refinement."
                )
                print(regression_msg)
                logger.warning("[%s] %s", exec_id, regression_msg)
                return None

            refine_log = (
                f"\n[REFINEMENT] LLM output ({len(refined)} chars):\n"
                f"{'-' * 50}\n"
                f"{refined[:500]}{'...' if len(refined) > 500 else ''}\n"
                f"{'-' * 50}"
            )
            print(refine_log)
            logger.info("[%s] %s", exec_id, refine_log)

            return refined

        except Exception as exc:
            logger.error(
                "[%s] [REFINEMENT] LLM call failed: %s", exec_id, exc, exc_info=True,
            )
            return None

    # ── Memory helpers ─────────────────────────────────────────────────

    _MAX_MEMORY_FACTS = 10

    @staticmethod
    def _format_memory_context(entries: list[MemoryEntry]) -> str:
        """Format memory entries into a context string for injection."""
        lines = [
            "## RELEVANT PAST INFORMATION",
            "The following was retrieved from past queries.",
            "Treat as optional hints — NOT as verified facts.",
        ]
        total_facts_used = 0
        for i, entry in enumerate(entries, 1):
            remaining = Executor._MAX_MEMORY_FACTS - total_facts_used
            if remaining <= 0:
                facts_text = "(fact limit reached)"
            elif entry.facts:
                capped_facts = entry.facts[:remaining]
                total_facts_used += len(capped_facts)
                facts_text = "; ".join(capped_facts)
            else:
                facts_text = "No specific facts"
            summary_text = ""
            if entry.summary:
                truncated = entry.summary[:200]
                if len(entry.summary) > 200:
                    truncated += "..."
                summary_text = f"\n     Past answer (excerpt): {truncated}"
            lines.append(
                f"  {i}. [Past query: \"{entry.query}\"] "
                f"Facts: {facts_text}{summary_text}"
            )
        return "\n".join(lines)

    _GENERIC_PREFIXES: list[str] = [
        "ai is", "technology is", "this shows", "this means",
        "it is clear", "it is important", "in conclusion",
        "overall", "in summary", "as we can see",
        "there are many", "there is a", "it has been",
        "the world", "the future", "experts say",
        "read the", "read full", "find in-depth", "explore the",
        "browse thousands", "cover all the", "watch videos",
        "no title", "no description",
    ]

    _MIN_FACT_LENGTH = 10
    _MIN_FACT_WORDS = 4

    @staticmethod
    def _extract_facts(state: ExecutionState) -> list[str]:
        """Extract concrete facts from successful tool step results."""
        facts: list[str] = []
        for sr in state.step_results:
            if sr.success and sr.type == "tool" and sr.result:
                for line in sr.result.split("\n"):
                    cleaned = line.strip().lstrip("- ").strip()
                    if not cleaned:
                        continue

                    if cleaned.startswith(("http://", "https://", "www.")):
                        continue

                    if cleaned.lower().startswith("source:"):
                        after = cleaned[len("source:"):].strip()
                        if not after or after.startswith(("http://", "https://", "www.")):
                            continue
                        stripped = re.sub(r'^[\w\s]+\s*[–—\-:]\s*', '', after).strip()
                        cleaned = stripped if stripped else after

                    words = cleaned.split()
                    has_number = any(c.isdigit() for c in cleaned)
                    has_proper_noun = any(
                        w[0].isupper() for w in words
                        if len(w) > 1
                    )
                    has_source_marker = any(
                        marker in cleaned.lower()
                        for marker in ("according", "reported", "announced",
                                       "published", "released", "launched")
                    )

                    if len(words) < Executor._MIN_FACT_WORDS:
                        if not has_number:
                            continue

                    if len(cleaned) < Executor._MIN_FACT_LENGTH:
                        continue

                    lower = cleaned.lower()
                    if any(lower.startswith(prefix) for prefix in Executor._GENERIC_PREFIXES):
                        continue

                    if not has_number and not has_proper_noun and not has_source_marker:
                        continue

                    facts.append(cleaned)

        return Executor._deduplicate_facts(facts)

    @staticmethod
    def _deduplicate_facts(
        facts: list[str],
        threshold: float = 0.70,
    ) -> list[str]:
        """Remove near-duplicate facts using token overlap."""
        if len(facts) <= 1:
            return facts

        deduplicated: list[str] = []
        for fact in facts:
            fact_tokens = set(fact.lower().split())
            is_dup = False
            for kept in deduplicated:
                kept_tokens = set(kept.lower().split())
                smaller = min(len(fact_tokens), len(kept_tokens))
                if smaller == 0:
                    continue
                overlap = len(fact_tokens & kept_tokens) / smaller
                if overlap >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                deduplicated.append(fact)

        return deduplicated

    def _store_in_memory(
        self,
        query: str,
        answer: str,
        state: ExecutionState,
        confidence: str,
        exec_id: str,
    ) -> None:
        """Conditionally store the interaction in memory."""
        if not self._memory:
            return

        facts = self._extract_facts(state)

        stored = self._memory.add_entry(
            query=query,
            answer=answer,
            facts=facts,
            confidence=confidence,
        )

        if stored:
            store_msg = (
                f"[MEMORY] ✅ Stored in memory "
                f"(confidence={confidence}, facts={len(facts)}, "
                f"total_entries={self._memory.size})"
            )
        else:
            store_msg = f"[MEMORY] ⏭️ Skipped storage (confidence={confidence})"
        print(store_msg)
        logger.info("[%s] %s", exec_id, store_msg)
