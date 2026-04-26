---
title: Autonomous Agent API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

# 🤖 Autonomous AI Agent

**A production-grade, multi-phase AI agent with adaptive planning, critic evaluation, semantic memory, and a fine-tuned tool router.**

[Live Demo](https://autonomous-agent-one.vercel.app) · [Backend (HF Spaces)](https://huggingface.co/spaces/Ayush0110/autonomousagent) · [ToolForge Fine-Tuning](https://github.com/ayushh0110/toolforge)

</div>

---

## Why This Exists

Most AI "agents" are thin wrappers around a single LLM call. This project builds a **real agentic system** with:

- A **Planner** that decomposes queries into executable steps
- An **Executor** that runs each step through 9 real-world tools
- A **Critic** that evaluates answers for grounding, completeness, and faithfulness
- **Semantic memory** with hybrid retrieval (vector + keyword) that learns from every interaction
- A **fine-tuned ToolForge router** (QLoRA, 86% accuracy) that can replace the heuristic classifier

The result is an agent that can handle everything from "what's 15% of 2850?" to "plan a 5-day trip to Japan" — with full observability, structured traces, and self-improving answer quality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                              │
│                                                                     │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────────┐   │
│  │  Query    │───▶│  Classifier │───▶│  Planner (LLM-powered)   │   │
│  │          │    │ (Heuristic / │    │  Decomposes into steps    │   │
│  │          │    │  ToolForge)  │    └───────────┬──────────────┘   │
│  │          │    └─────────────┘                │                   │
│  │          │         │                         ▼                   │
│  │          │    ┌────┴───┐         ┌──────────────────────────┐   │
│  │          │    │Memory  │         │  Executor                 │   │
│  │          │    │Retrieval│         │  ├─ Tool steps → Agent   │   │
│  │          │    │(Hybrid)│         │  ├─ Reasoning steps       │   │
│  │          │    └────────┘         │  ├─ Early stopping        │   │
│  │          │                       │  └─ Dynamic plan adjust   │   │
│  │          │                       └───────────┬──────────────┘   │
│  │          │                                   │                   │
│  │          │         ┌─────────────────────────┤                   │
│  │          │         ▼                         ▼                   │
│  │          │    ┌──────────┐         ┌───────────────┐            │
│  │          │    │ Critic   │◀────────│  Synthesizer  │            │
│  │          │    │ (5-axis  │         │  (Natural     │            │
│  │          │    │  eval)   │────────▶│   language)   │            │
│  │          │    └──────────┘  refine └───────────────┘            │
│  │          │                                                       │
│  └──────────┘    ┌─────────────────────────────────────────────┐   │
│                  │  9 Tools: web_search, calculator, weather,   │   │
│                  │  wikipedia, datetime, dictionary, translate,  │   │
│                  │  unit_converter, web_reader                   │   │
│                  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
         ▲                                              │
         │              HTTP / JSON API                  │
         ▼                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    React Frontend (Vite)                             │
│  ├─ ChatUI with typing indicator + animated responses               │
│  ├─ AgentPanel (live execution trace, tools used, confidence)       │
│  ├─ Client-side memory (localStorage profile + sessionStorage)      │
│  └─ FloatingBackground (glassmorphism, animated particles)          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Features Deep-Dive

### 1. 🧠 Query Classification — Decision Layer

Every query is classified **before** any LLM call, using zero-cost regex heuristics:

| Decision | Example | What Happens |
|----------|---------|--------------|
| `direct_answer` | "Hi", "Explain recursion" | Skips planning + tools entirely → 1 LLM call |
| `needs_search` | "Latest AI news", "Weather in Tokyo" | Full pipeline: plan → tool → synthesize → critique |
| `memory_sufficient` | "What's my name?" (after telling it) | Answers from stored memory, no tools |
| `autonomous_task` | "Plan a trip to Japan" | Delegates to AutonomousExecutor (multi-aspect) |

**ToolForge upgrade**: The heuristic classifier can be replaced by a fine-tuned Qwen2.5-7B model (86.2% accuracy, trained on 1.1K synthetic tool-call traces). Toggle with `TOOLFORGE_ENABLED=true`.

### 2. 📋 Adaptive Planner

The Planner decomposes queries into typed steps via LLM:

```json
{
  "plan": [
    {"step": "Search for latest AI breakthroughs and announcements", "type": "tool"},
    {"step": "Extract key developments from search results", "type": "reasoning"},
    {"step": "Organize findings into a clear summary", "type": "reasoning"}
  ]
}
```

- **Fast-path**: `direct_answer` and `memory_sufficient` skip the LLM planning call entirely
- **Adaptive**: Plan length adjusts based on query complexity (1–5 steps)
- **Validated**: Each step must have a specific subject + action verb (vague steps like "explore" are rejected)

### 3. ⚡ Executor — Adaptive Execution Engine

The Executor runs each plan step through the Agent with:

- **Early stopping**: If an intermediate result is strong enough (200+ chars, 3+ entities, 2+ numbers), remaining reasoning steps are skipped
- **Dynamic plan adjustment**: Trims redundant reasoning steps after strong tool results, or appends fallback reasoning after weak ones
- **Weak-result recovery**: If a tool returns weak results, retries with a different query, then falls back to reasoning
- **Request-scoped tool cache**: Prevents duplicate tool calls within a single request
- **Context compression**: Previous step results are truncated (500–1500 chars) before injection

### 4. 🔧 Agent Loop — Structured Tool Calling

The core Agent runs a bounded loop (max 4 iterations, max 2 tool calls):

```
Step 1: LLM → tool_call("web_search", {"query": "..."})
Step 2: Tool result → LLM → final_answer
```

Safety controls:
- **Final step enforcement**: Last iteration forces `final_answer`, blocks tool calls
- **Tool call limit**: Hard cap on tool invocations per request
- **Parse retry**: If LLM outputs invalid JSON, feeds error back for one retry
- **Invalid tool protection**: Unknown tools return structured errors, not crashes
- **Reasoning visibility**: Every LLM decision includes a `reasoning` field (logged, never in API)

### 5. 🔍 Critic — 5-Axis Answer Evaluation

Every synthesized answer is evaluated by the Critic on **5 dimensions**:

| Axis | What It Checks |
|------|---------------|
| **Grounding** | Are all claims supported by the step results? |
| **Completeness** | Does the answer address every part of the query? |
| **Specificity** | Does it include concrete details (names, numbers, dates)? |
| **Redundancy** | Is it concise without unnecessary repetition? |
| **Faithfulness** | Are there any hallucinated facts not in the results? |

Each issue is **severity-tagged** (`high` / `medium` / `low`):
- `high` → answer is refined (up to 2 iterations)
- `low` → accepted as-is
- **Regression guard**: refined answer must be ≥70% the length of the previous (prevents oversimplification)

### 6. 💾 Semantic Memory — Hybrid RAG

The memory system gives the agent **persistent context** across conversations:

**Storage pipeline:**
1. Every query runs through the **MemoryAnalyzer** (LLM-based fact extraction)
2. Extracted facts pass through 4 quality gates:
   - Confidence ≥ 0.8
   - No speculative language ("maybe", "might", "someday")
   - Valid intent type (profile: identity/preference/event, session: reference/task)
   - Must have key + value
3. Stored with critic confidence level (low → rejected, medium/high → stored)
4. Deduplication: 80% token overlap → skip

**Retrieval pipeline (two-stage hybrid):**
1. **Stage 1 — Vector search**: `all-MiniLM-L6-v2` embeddings → cosine similarity → top 5 candidates
2. **Stage 2 — Keyword re-rank**: Weighted Jaccard (0.8 facts + 0.2 summary) + synonym expansion + substring boost + time decay (48h half-life) + confidence multiplier
3. **Blended score**: 0.6 × keyword + 0.4 × vector
4. **Semantic dedup**: 80% overlap between results → keep higher-scoring one

### 7. 🚀 Autonomous Executor — Goal-Driven Multi-Step Tasks

Complex tasks like "plan a trip to Japan" or "compare React vs Vue" are handled by a separate execution engine:

1. **Aspect decomposition**: LLM breaks the goal into prioritized aspects (e.g., flights, hotels, places, budget, itinerary)
2. **Adaptive execution**: Each aspect is researched with tool calls, with retry + alternative query generation on failure
3. **Coverage gate**: All required aspects must have data before synthesis — missing aspects trigger retry
4. **Reasoning synthesis**: Budget breakdowns, comparisons, itineraries are synthesized from collected data
5. **Structured output**: Final answer is organized by section headers, not flat paragraphs

Limits: max 12 steps, max 6 tool calls, max 2 retries per aspect, max 7 aspects.

### 8. 🔌 9 Integrated Tools

| Tool | Source | What It Does |
|------|--------|-------------|
| `web_search` | DuckDuckGo | Search the internet for current information |
| `web_reader` | httpx + BeautifulSoup | Extract content from specific URLs |
| `weather` | OpenWeatherMap API | Current weather for any city |
| `calculator` | Python `eval` (sandboxed) | Math expressions, sqrt, log, trig, etc. |
| `wikipedia` | Wikipedia API | Encyclopedic summaries |
| `dictionary` | Free Dictionary API | Definitions, phonetics, examples |
| `translate` | MyMemory API | Translation between 30+ languages |
| `unit_converter` | Built-in conversion tables | Length, weight, temperature, volume, etc. |
| `datetime` | Python `datetime` | Current time, timezone conversion, date math |

All tools are registered via `ToolRegistry` with schema validation (required inputs, type checking) before execution.

### 9. 🎭 LLM Client — Smart Rate Limiting

The Groq client includes production-grade resilience:

- **Multi-key rotation**: 5 API keys with most-rested-first selection (not round-robin)
- **Proactive throttling**: 1s minimum gap per key (prevents 429s before they happen)
- **Automatic retry**: Parses Groq's `Retry-After` header, falls back to exponential backoff
- **Up to 5 retries** with key rotation on rate limit

### 10. 🧪 ToolForge Integration — Fine-Tuned Tool Router

The heuristic classifier (`classify_query()`) uses 200+ lines of regex patterns. [**ToolForge**](https://github.com/ayushh0110/toolforge) replaces it with a fine-tuned model:

| Metric | Heuristic | ToolForge Model |
|--------|:---------:|:---------------:|
| Accuracy | ~75% | **86.2%** |
| Approach | Regex patterns | QLoRA fine-tuned Qwen2.5-7B |
| Training data | — | 1,173 synthetic examples (Gemini distillation) |
| Ablation runs | — | 4 (tracked on W&B) |
| Latency | 0ms (regex) | ~200ms (GPU inference) |

The integration is a **feature flag** — set `TOOLFORGE_ENABLED=true` + provide the adapter path. Falls back gracefully if GPU/dependencies unavailable.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.12, FastAPI, Uvicorn |
| **LLM** | Groq API (Llama 3.1 8B Instant) |
| **Frontend** | React 19, Vite, vanilla CSS |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Tool Router** | QLoRA fine-tuned Qwen2.5-7B (optional) |
| **Deployment** | HuggingFace Spaces (Docker) + Vercel |
| **Monitoring** | Structured logging, full execution traces |

---

## Project Structure

```
autonomous-agent/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── config.py                  # Settings from environment
│   ├── agent/
│   │   ├── agent.py               # Core agent loop (4-step, 2-tool limit)
│   │   ├── executor.py            # Planner-Executor pipeline
│   │   ├── planner.py             # LLM-powered query decomposition
│   │   ├── critic.py              # 5-axis answer evaluation
│   │   ├── autonomous_executor.py # Goal-driven multi-step execution
│   │   ├── memory_analyzer.py     # LLM-based fact extraction (4 quality gates)
│   │   ├── parser.py              # Robust JSON parser for LLM outputs
│   │   └── toolforge_router.py    # Fine-tuned model router (feature flag)
│   ├── llm/
│   │   └── groq_client.py         # Multi-key rotation + throttling
│   ├── memory/
│   │   ├── memory_store.py        # Hybrid RAG (vector + keyword)
│   │   ├── embedding.py           # Sentence-transformer embeddings
│   │   └── vector_store.py        # In-memory HNSW vector index
│   ├── tools/                     # 9 tool implementations
│   │   ├── registry.py            # Schema validation + execution
│   │   ├── search_tool.py         # DuckDuckGo web search
│   │   ├── web_reader.py          # URL content extraction
│   │   ├── weather_tool.py        # OpenWeatherMap integration
│   │   ├── calculator_tool.py     # Sandboxed math evaluation
│   │   ├── wikipedia_tool.py      # Wikipedia summaries
│   │   ├── dictionary_tool.py     # Word definitions
│   │   ├── translation_tool.py    # Multi-language translation
│   │   ├── unit_converter_tool.py # Unit conversion tables
│   │   └── datetime_tool.py       # Timezone-aware date/time
│   ├── services/
│   │   └── agent_service.py       # Wires everything together
│   └── schemas/
│       └── request.py             # Pydantic request/response models
├── frontend/                      # React + Vite
│   └── src/
│       ├── components/
│       │   ├── ChatUI.jsx         # Chat interface
│       │   ├── AgentPanel.jsx     # Live execution trace
│       │   └── FloatingBackground.jsx # Animated particles
│       └── services/              # API client
├── tests/
│   ├── test_agent_features.py     # Unit tests (agent, tools, memory)
│   └── test_prompts.py            # Prompt-level regression tests
├── Dockerfile                     # Production container
├── Procfile                       # HuggingFace Spaces entrypoint
└── requirements.txt
```

---

## Quick Start

### Backend

```bash
# Clone
git clone https://github.com/ayushh0110/autonomous-agent.git
cd autonomous-agent

# Install
pip install -r requirements.txt

# Configure
cp app/.env.example app/.env
# Add your GROQ_API_KEY(s)

# Run
uvicorn app.main:app --reload --port 7860
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Enable ToolForge (optional, requires GPU)

```bash
# In app/.env
TOOLFORGE_ENABLED=true
TOOLFORGE_ADAPTER_PATH=./checkpoints/qwen7b-r64-lr2e4/final
```

---

## API

### `POST /chat`

```json
{
  "query": "What's the weather in Tokyo?",
  "profile_context": [{"key": "name", "value": "Ayush"}],
  "session_context": [{"intent": "task", "value": "trip planning"}]
}
```

**Response:**

```json
{
  "response": "Hey! So Tokyo right now is around 18°C with clear skies...",
  "source": "planner_executor",
  "tools_used": ["weather"],
  "steps_taken": 2,
  "plan": ["Get weather data for Tokyo", "Format into conversational response"],
  "confidence": "high",
  "refinements": 0,
  "memory_used": false,
  "decision": "needs_search",
  "llm_calls": 4,
  "early_stopped": false,
  "cache_hits": 0,
  "memory_extraction": null
}
```

---

## Development Phases

| Phase | Focus | Key Addition |
|-------|-------|-------------|
| 1 | Foundation | FastAPI + Groq + basic agent loop |
| 2 | Tool System | 9 tools + registry + schema validation |
| 2.1 | Safety | Tool limits, final step enforcement, parse retry |
| 3 | Planning | Planner-Executor pipeline + synthesis |
| 3.2 | Quality | Critic (5-axis eval) + severity-aware refinement |
| 4 | Intelligence | Query classifier, early stopping, dynamic planning, tool cache |
| 5 | Memory | Hybrid RAG, memory analyzer, 4 quality gates |
| 6 | Autonomy | AutonomousExecutor for multi-step tasks |
| 7 | Fine-Tuning | ToolForge integration (QLoRA Qwen2.5-7B router) |

---

## License

MIT
