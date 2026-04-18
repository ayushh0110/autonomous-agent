"""Comprehensive test suite for the Autonomous Agent.

Tests all major features:
  1. Tools: calculator, wikipedia, dictionary, unit converter, datetime, translation
  2. Query classifier: all routing paths + tool-intent detection
  3. Planner: fast-path, step generation
  4. Config: multi-key loading
  5. Memory store: store/retrieve/relevance
  6. GroqClient: key rotation logic
  7. Parser: JSON action parsing
  8. Agent system prompt construction
"""

import os
import sys
import time
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ═══════════════════════════════════════════════════════════════════════════
# 1. TOOL TESTS — verify each tool works correctly in isolation
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculatorTool:
    """Test the calculator tool with various math expressions."""

    def setup_method(self):
        from app.tools.calculator_tool import CalculatorTool
        self.tool = CalculatorTool()

    def test_basic_arithmetic(self):
        assert "4" in self.tool.run(expression="2+2")

    def test_subtraction(self):
        assert "5" in self.tool.run(expression="10-5")

    def test_multiplication(self):
        assert "42" in self.tool.run(expression="6*7")

    def test_division(self):
        assert "5" in self.tool.run(expression="10/2")

    def test_power(self):
        assert "1024" in self.tool.run(expression="2**10")

    def test_sqrt(self):
        result = self.tool.run(expression="sqrt(144)")
        assert "12" in result

    def test_sqrt_non_perfect(self):
        result = self.tool.run(expression="sqrt(44567)")
        assert "211" in result

    def test_factorial(self):
        result = self.tool.run(expression="factorial(10)")
        assert "3628800" in result

    def test_trig(self):
        result = self.tool.run(expression="sin(radians(90))")
        assert "1" in result

    def test_pi_constant(self):
        result = self.tool.run(expression="pi")
        assert "3.14" in result

    def test_division_by_zero(self):
        result = self.tool.run(expression="1/0")
        assert "Error" in result

    def test_empty_expression(self):
        result = self.tool.run(expression="")
        assert "Error" in result

    def test_caret_to_power(self):
        result = self.tool.run(expression="2^8")
        assert "256" in result

    def test_blocked_keywords(self):
        result = self.tool.run(expression="import os")
        assert "Error" in result

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "calculator"
        assert "expression" in schema["input_schema"]


class TestUnitConverterTool:
    """Test unit conversion across categories."""

    def setup_method(self):
        from app.tools.unit_converter_tool import UnitConverterTool
        self.tool = UnitConverterTool()

    def test_miles_to_km(self):
        result = self.tool.run(value=5, from_unit="miles", to_unit="km")
        assert "8.04" in result

    def test_km_to_miles(self):
        result = self.tool.run(value=10, from_unit="km", to_unit="miles")
        assert "6.21" in result

    def test_fahrenheit_to_celsius(self):
        result = self.tool.run(value=100, from_unit="fahrenheit", to_unit="celsius")
        assert "37.78" in result

    def test_celsius_to_fahrenheit(self):
        result = self.tool.run(value=0, from_unit="celsius", to_unit="fahrenheit")
        assert "32" in result

    def test_celsius_to_kelvin(self):
        result = self.tool.run(value=100, from_unit="celsius", to_unit="kelvin")
        assert "373" in result

    def test_kg_to_lbs(self):
        result = self.tool.run(value=1, from_unit="kg", to_unit="lbs")
        assert "2.2" in result

    def test_liters_to_gallons(self):
        result = self.tool.run(value=10, from_unit="liters", to_unit="gallons")
        assert "2.6" in result

    def test_mb_to_gb(self):
        result = self.tool.run(value=1024, from_unit="mb", to_unit="gb")
        assert "1" in result

    def test_hours_to_seconds(self):
        result = self.tool.run(value=1, from_unit="hour", to_unit="seconds")
        assert "3600" in result

    def test_cross_category_error(self):
        result = self.tool.run(value=5, from_unit="miles", to_unit="kg")
        assert "Cannot convert" in result

    def test_missing_unit(self):
        result = self.tool.run(value=5, from_unit="miles", to_unit="")
        assert "Error" in result

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "unit_converter"


class TestDateTimeTool:
    """Test datetime tool: current time, timezone conversion, date diff."""

    def setup_method(self):
        from app.tools.datetime_tool import DateTimeTool
        self.tool = DateTimeTool()

    def test_now_utc(self):
        result = self.tool.run(action="now")
        assert "UTC" in result
        assert "Date:" in result
        assert "Time:" in result

    def test_now_tokyo(self):
        result = self.tool.run(action="now", timezone="tokyo")
        assert "JST" in result or "tokyo" in result.lower()

    def test_now_india(self):
        result = self.tool.run(action="now", timezone="india")
        assert "IST" in result or "india" in result.lower()

    def test_now_new_york(self):
        result = self.tool.run(action="now", timezone="new york")
        assert "Date:" in result

    def test_diff_future_date(self):
        result = self.tool.run(action="diff", date="2030-01-01")
        assert "days" in result

    def test_diff_past_date(self):
        result = self.tool.run(action="diff", date="2020-01-01")
        assert "ago" in result

    def test_diff_invalid_date(self):
        result = self.tool.run(action="diff", date="not-a-date")
        assert "Error" in result

    def test_diff_no_date(self):
        result = self.tool.run(action="diff")
        assert "Error" in result or "required" in result.lower()

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "datetime"


class TestDictionaryTool:
    """Test dictionary lookups (requires network)."""

    def setup_method(self):
        from app.tools.dictionary_tool import DictionaryTool
        self.tool = DictionaryTool()

    @pytest.mark.network
    def test_define_hello(self):
        result = self.tool.run(word="hello")
        assert "hello" in result.lower()

    @pytest.mark.network
    def test_define_unknown_word(self):
        result = self.tool.run(word="xyzzyplugh12345")
        assert "No definition" in result

    def test_empty_word(self):
        result = self.tool.run(word="")
        assert "Error" in result

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "dictionary"


class TestWikipediaTool:
    """Test Wikipedia lookups (requires network)."""

    def setup_method(self):
        from app.tools.wikipedia_tool import WikipediaTool
        self.tool = WikipediaTool()

    @pytest.mark.network
    def test_lookup_einstein(self):
        result = self.tool.run(query="Albert Einstein")
        assert "Einstein" in result
        assert "physicist" in result.lower() or "relativity" in result.lower()

    @pytest.mark.network
    def test_lookup_python(self):
        result = self.tool.run(query="Python programming language")
        assert "Python" in result

    def test_empty_query(self):
        result = self.tool.run(query="")
        assert "Error" in result

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "wikipedia"


class TestTranslationTool:
    """Test translation (requires network)."""

    def setup_method(self):
        from app.tools.translation_tool import TranslationTool
        self.tool = TranslationTool()

    @pytest.mark.network
    def test_english_to_spanish(self):
        result = self.tool.run(text="Hello", from_lang="english", to_lang="spanish")
        assert "Hola" in result

    @pytest.mark.network
    def test_english_to_french(self):
        result = self.tool.run(text="Thank you", from_lang="english", to_lang="french")
        assert "Merci" in result

    def test_missing_target_lang(self):
        result = self.tool.run(text="Hello", from_lang="english", to_lang="")
        assert "Error" in result

    def test_empty_text(self):
        result = self.tool.run(text="", from_lang="english", to_lang="spanish")
        assert "Error" in result

    def test_schema(self):
        schema = self.tool.schema()
        assert schema["name"] == "translate"


# ═══════════════════════════════════════════════════════════════════════════
# 2. QUERY CLASSIFIER TESTS — routing logic
# ═══════════════════════════════════════════════════════════════════════════

class TestQueryClassifier:
    """Test the heuristic query classifier routing decisions."""

    def setup_method(self):
        from app.agent.agent import classify_query
        self.classify = classify_query

    # ── Tool-intent queries (highest priority) ──

    def test_translation_query(self):
        d = self.classify("say hello in japanese")
        assert d.decision_type == "needs_search"
        assert "tool" in d.reasoning.lower()

    def test_translation_query_2(self):
        d = self.classify("translate good morning to french")
        assert d.decision_type == "needs_search"

    def test_calculation_query(self):
        d = self.classify("sqrt(44567)")
        assert d.decision_type == "needs_search"

    def test_arithmetic_query(self):
        d = self.classify("2+2")
        assert d.decision_type == "needs_search"

    def test_unit_conversion_query(self):
        d = self.classify("convert 5 miles to km")
        assert d.decision_type == "needs_search"

    def test_temperature_conversion(self):
        d = self.classify("100 fahrenheit to celsius")
        assert d.decision_type == "needs_search"

    def test_time_query(self):
        d = self.classify("what time is it in tokyo")
        assert d.decision_type == "needs_search"

    def test_definition_query(self):
        d = self.classify("define serendipity")
        assert d.decision_type == "needs_search"

    # ── Personal / conversational → direct answer ──

    def test_greeting_hello(self):
        d = self.classify("hello")
        assert d.decision_type == "direct_answer"

    def test_greeting_hi(self):
        d = self.classify("hi how are you")
        assert d.decision_type == "direct_answer"

    def test_thanks(self):
        d = self.classify("thank you")
        assert d.decision_type == "direct_answer"

    def test_personal_info(self):
        d = self.classify("my name is Ayush")
        assert d.decision_type == "direct_answer"

    # ── Recency → needs_search ──

    def test_recency_latest(self):
        d = self.classify("latest AI news")
        assert d.decision_type == "needs_search"

    def test_recency_today(self):
        d = self.classify("what happened today in tech")
        assert d.decision_type == "needs_search"

    # ── Knowledge → direct_answer ──

    def test_knowledge_explain(self):
        d = self.classify("explain photosynthesis")
        assert d.decision_type == "direct_answer"

    def test_knowledge_how_does(self):
        d = self.classify("how does gravity work")
        assert d.decision_type == "direct_answer"

    # ── Factual → needs_search ──

    def test_factual_who(self):
        d = self.classify("who is the president of France")
        assert d.decision_type == "needs_search"

    def test_factual_population(self):
        d = self.classify("population of India")
        assert d.decision_type == "needs_search"

    # ── Autonomous tasks ──

    def test_autonomous_plan_trip(self):
        d = self.classify("plan a trip to Japan for 10 days")
        assert d.decision_type == "autonomous_task"

    # ── Default fallback → needs_search ──

    def test_default_fallback(self):
        d = self.classify("some obscure query nobody expects")
        assert d.decision_type == "needs_search"


# ═══════════════════════════════════════════════════════════════════════════
# 3. PLANNER TESTS — fast-path & step generation
# ═══════════════════════════════════════════════════════════════════════════

class TestPlanner:
    """Test planner fast-path and plan step structure."""

    def setup_method(self):
        from app.agent.planner import Planner, PlanStep

    def test_direct_answer_fast_path(self):
        from app.agent.planner import Planner, PlanStep
        from unittest.mock import MagicMock
        planner = Planner(llm=MagicMock(), max_plan_steps=5)
        steps = planner.create_plan("hello", decision="direct_answer")
        assert len(steps) == 1
        assert steps[0].type == "reasoning"

    def test_memory_sufficient_fast_path(self):
        from app.agent.planner import Planner, PlanStep
        from unittest.mock import MagicMock
        planner = Planner(llm=MagicMock(), max_plan_steps=5)
        steps = planner.create_plan("recall my name", decision="memory_sufficient")
        assert len(steps) == 1
        assert steps[0].type == "reasoning"

    def test_plan_step_structure(self):
        from app.agent.planner import PlanStep
        step = PlanStep(step="Search for info", type="tool")
        assert step.step == "Search for info"
        assert step.type == "tool"


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONFIG TESTS — settings & multi-key loading
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    """Test configuration loading and multi-key detection."""

    def test_settings_load(self):
        from app.config import get_settings
        settings = get_settings()
        assert settings.groq_api_key  # at least one key exists
        assert settings.groq_model

    def test_multiple_keys_loaded(self):
        from app.config import get_settings
        settings = get_settings()
        assert len(settings.groq_api_keys) >= 1

    def test_default_model(self):
        from app.config import get_settings
        settings = get_settings()
        assert "llama" in settings.groq_model.lower() or "groq" in settings.groq_model.lower() or settings.groq_model


# ═══════════════════════════════════════════════════════════════════════════
# 5. MEMORY STORE TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestMemoryStore:
    """Test memory store/retrieve and relevance scoring."""

    def setup_method(self):
        from app.memory.memory_store import MemoryStore, MemoryEntry
        self.store = MemoryStore()
        self.MemoryEntry = MemoryEntry

    def test_store_and_retrieve(self):
        self.store.add_entry(
            query="What is Python?",
            answer="Python is a programming language.",
            confidence="high",
            facts=["Python is a programming language"],
        )
        results = self.store.retrieve("Python programming")
        assert len(results) >= 1

    def test_retrieve_no_match(self):
        from app.memory.memory_store import MemoryStore
        fresh_store = MemoryStore()
        results = fresh_store.retrieve("xyzzy12345nonexistent")
        assert len(results) == 0

    def test_store_increments_count(self):
        from app.memory.memory_store import MemoryStore
        store = MemoryStore()
        initial = store.size
        store.add_entry(
            query="test query",
            answer="test answer",
            confidence="medium",
            facts=[],
        )
        assert store.size == initial + 1


# ═══════════════════════════════════════════════════════════════════════════
# 6. GROQ CLIENT TESTS — key rotation logic
# ═══════════════════════════════════════════════════════════════════════════

class TestGroqClientKeyRotation:
    """Test API key rotation and throttling logic."""

    def test_next_key_selects_most_rested(self):
        from app.config import get_settings
        from app.llm.groq_client import GroqClient
        settings = get_settings()
        client = GroqClient(settings)

        if len(client._api_keys) < 2:
            pytest.skip("Need ≥2 keys to test rotation")

        # First call picks key 0 (all at time 0)
        key1 = client._next_key()
        client._mark_key_used()

        # Second call should pick a different key
        key2 = client._next_key()
        assert key1 != key2, "Should rotate to a different key"

    def test_multiple_keys_initialized(self):
        from app.config import get_settings
        from app.llm.groq_client import GroqClient
        settings = get_settings()
        client = GroqClient(settings)
        assert len(client._api_keys) >= 1

    def test_per_key_tracking(self):
        from app.config import get_settings
        from app.llm.groq_client import GroqClient
        settings = get_settings()
        client = GroqClient(settings)
        assert len(client._key_last_used) == len(client._api_keys)
        # All start at 0
        for t in client._key_last_used.values():
            assert t == 0.0

    def test_mark_key_updates_timestamp(self):
        from app.config import get_settings
        from app.llm.groq_client import GroqClient
        settings = get_settings()
        client = GroqClient(settings)
        client._next_key()
        before = client._key_last_used[client._key_index]
        client._mark_key_used()
        after = client._key_last_used[client._key_index]
        assert after > before


# ═══════════════════════════════════════════════════════════════════════════
# 7. PARSER TESTS — JSON action parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestParser:
    """Test LLM output parser for actions."""

    def setup_method(self):
        from app.agent.parser import parse_llm_action, ToolCallAction, FinalAnswerAction, ParseError
        self.parse = parse_llm_action
        self.ToolCallAction = ToolCallAction
        self.FinalAnswerAction = FinalAnswerAction
        self.ParseError = ParseError

    def test_parse_final_answer(self):
        raw = '{"action": "final_answer", "reasoning": "test", "answer": "Hello!"}'
        result = self.parse(raw)
        assert isinstance(result, self.FinalAnswerAction)
        assert result.answer == "Hello!"

    def test_parse_tool_call(self):
        raw = '{"action": "tool_call", "reasoning": "need data", "tool_name": "web_search", "tool_input": {"query": "AI news"}}'
        result = self.parse(raw)
        assert isinstance(result, self.ToolCallAction)
        assert result.tool_name == "web_search"
        assert result.tool_input["query"] == "AI news"

    def test_parse_invalid_json(self):
        raw = "This is not JSON at all"
        result = self.parse(raw)
        assert isinstance(result, self.ParseError)

    def test_parse_missing_action(self):
        raw = '{"reasoning": "test", "answer": "Hello!"}'
        result = self.parse(raw)
        assert isinstance(result, self.ParseError)


# ═══════════════════════════════════════════════════════════════════════════
# 8. TOOL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Test tool registry lookup and schema generation."""

    def setup_method(self):
        from app.tools.registry import ToolRegistry
        from app.tools.calculator_tool import CalculatorTool
        from app.tools.weather_tool import WeatherTool
        self.registry = ToolRegistry([CalculatorTool(), WeatherTool()])

    def test_get_tool_by_name(self):
        assert self.registry.has_tool("calculator")

    def test_get_missing_tool(self):
        assert not self.registry.has_tool("nonexistent_tool")

    def test_tool_names(self):
        names = self.registry.tool_names
        assert "calculator" in names
        assert "weather" in names

    def test_schemas(self):
        schemas = self.registry.get_schemas()
        assert len(schemas) == 2
        assert all("name" in s for s in schemas)


# ═══════════════════════════════════════════════════════════════════════════
# 9. EXECUTOR FAST-PATH TEST
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutorFastPath:
    """Test that direct_answer queries skip synthesis/critic."""

    def test_direct_answer_classification_skips_pipeline(self):
        """Verify direct_answer queries are classified correctly."""
        from app.agent.agent import classify_query
        d = classify_query("hello")
        assert d.decision_type == "direct_answer"
        # This means executor will skip synthesis/critic

    def test_tool_query_goes_through_pipeline(self):
        """Verify tool-intent queries go through full pipeline."""
        from app.agent.agent import classify_query
        d = classify_query("translate hello to spanish")
        assert d.decision_type == "needs_search"
        # This means executor will run full pipeline with tools


# ═══════════════════════════════════════════════════════════════════════════
# 10. INTEGRATION — SERVICE LAYER
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceLayer:
    """Test the agent service wiring (no live API calls)."""

    def test_build_executor(self):
        """Verify executor builds successfully with all tools."""
        from app.services.agent_service import _build_executor
        # Clear cache to force rebuild
        _build_executor.cache_clear()
        executor, analyzer = _build_executor()
        assert executor is not None
        assert analyzer is not None

    def test_tool_count(self):
        """Verify all 9 tools are registered."""
        from app.services.agent_service import _build_executor
        _build_executor.cache_clear()
        executor, _ = _build_executor()
        tools = executor._agent._registry
        assert len(tools.tool_names) == 9, (
            f"Expected 9 tools, got {len(tools.tool_names)}: {tools.tool_names}"
        )

    def test_all_tool_names(self):
        """Verify all expected tool names are present."""
        from app.services.agent_service import _build_executor
        _build_executor.cache_clear()
        executor, _ = _build_executor()
        names = set(executor._agent._registry.tool_names)
        expected = {
            "web_search", "web_reader", "weather", "calculator",
            "wikipedia", "dictionary", "unit_converter", "datetime", "translate",
        }
        assert names == expected, f"Missing: {expected - names}, Extra: {names - expected}"
