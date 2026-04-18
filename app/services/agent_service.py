"""Service layer — wires together config, LLM, tools, and agent.

Phase 6: Routes queries through the Planner–Executor pipeline
with intelligent memory extraction and autonomous task execution.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.agent.agent import Agent
from app.agent.autonomous_executor import AutonomousExecutor
from app.agent.executor import Executor, PlannerExecutorResult
from app.agent.memory_analyzer import MemoryAnalyzer, MemoryExtraction
from app.config import Settings, get_settings
from app.llm.groq_client import GroqClient
from app.memory.memory_store import MemoryStore
from app.tools.calculator_tool import CalculatorTool
from app.tools.datetime_tool import DateTimeTool
from app.tools.dictionary_tool import DictionaryTool
from app.tools.search_tool import SearchTool
from app.tools.translation_tool import TranslationTool
from app.tools.unit_converter_tool import UnitConverterTool
from app.tools.weather_tool import WeatherTool
from app.tools.web_reader import WebReaderTool
from app.tools.wikipedia_tool import WikipediaTool

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _build_executor() -> tuple[Executor, MemoryAnalyzer]:
    """Construct the Executor + AutonomousExecutor + MemoryAnalyzer."""
    settings: Settings = get_settings()
    llm = GroqClient(settings)
    tools = [
        SearchTool(),
        WebReaderTool(),
        WeatherTool(),
        CalculatorTool(),
        WikipediaTool(),
        DictionaryTool(),
        UnitConverterTool(),
        DateTimeTool(),
        TranslationTool(),
    ]
    agent = Agent(
        llm=llm,
        tools=tools,
        max_steps=settings.max_agent_steps,
        max_tool_calls=settings.max_tool_calls,
    )
    memory = MemoryStore()

    # Autonomous executor for complex multi-step tasks
    autonomous = AutonomousExecutor(
        agent=agent,
        llm=llm,
        memory=memory,
    )

    executor = Executor(
        agent=agent,
        llm=llm,
        max_plan_steps=settings.max_plan_steps,
        max_execution_steps=settings.max_execution_steps,
        max_refinements=settings.max_refinements,
        memory=memory,
        autonomous_executor=autonomous,
    )
    analyzer = MemoryAnalyzer(llm=llm)

    tool_names = [t.name for t in tools]
    logger.info(
        "Executor ready (max_plan_steps=%d, max_execution_steps=%d, "
        "max_refinements=%d, agent_max_steps=%d, agent_max_tool_calls=%d, "
        "memory_enabled=True, memory_analyzer=True, autonomous=True, "
        "tools=%s)",
        settings.max_plan_steps,
        settings.max_execution_steps,
        settings.max_refinements,
        settings.max_agent_steps,
        settings.max_tool_calls,
        tool_names,
    )
    return executor, analyzer


def chat(
    query: str,
    profile_context: list[dict[str, Any]] | None = None,
    session_context: list[dict[str, Any]] | None = None,
) -> tuple[PlannerExecutorResult, dict[str, Any] | None]:
    """Handle a user query through the Planner–Executor pipeline.

    Args:
        query: Raw user input.
        profile_context: User profile entries from frontend localStorage.
        session_context: Session context entries from frontend sessionStorage.

    Returns:
        Tuple of (PlannerExecutorResult, memory_extraction dict or None).
    """
    executor, analyzer = _build_executor()

    # ── Memory extraction (runs in parallel with main pipeline conceptually) ──
    extraction: MemoryExtraction | None = None
    extraction_dict: dict[str, Any] | None = None
    try:
        extraction = analyzer.analyze(query)
        if extraction.store:
            extraction_dict = extraction.to_dict()
            logger.info(
                "[MEMORY-ANALYZER] Extraction result: %s/%s key=%s confidence=%.2f",
                extraction.memory_type, extraction.intent,
                extraction.data.get("key"), extraction.confidence,
            )
    except Exception as exc:
        logger.warning("[MEMORY-ANALYZER] Analysis failed (non-fatal): %s", exc)

    # ── Execute pipeline with injected context ──
    result = executor.execute(
        query,
        profile_context=profile_context,
        session_context=session_context,
    )

    return result, extraction_dict


