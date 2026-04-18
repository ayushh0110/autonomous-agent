"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas.request import ChatRequest, ChatResponse
from app.services import agent_service

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown hooks."""
    logger.info("🚀 AI Agent API starting up")
    yield
    logger.info("🛑 AI Agent API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Agent API",
    description="Phase 6 — Planner–Executor agent with autonomous task execution.",
    version="0.6.0",
    lifespan=lifespan,
)

# ── CORS — allow frontend (Vercel) to call backend (Render) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: lock down to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Accept a user query and return the agent's response.

    The agent decides whether to answer directly via the LLM
    or to invoke a tool (e.g., web search) first.
    """
    logger.info("POST /chat — query=%r", request.query)

    try:
        result, memory_extraction = agent_service.chat(
            request.query,
            profile_context=request.profile_context,
            session_context=request.session_context,
        )
        return ChatResponse(
            response=result.answer,
            source=result.source,
            tools_used=result.tools_used,
            steps_taken=result.steps_taken,
            plan=result.plan,
            confidence=result.confidence,
            refinements=result.refinements,
            memory_used=result.memory_used,
            memory_hits=result.memory_hits,
            decision=result.decision,
            llm_calls=result.llm_calls,
            steps_skipped=result.steps_skipped,
            early_stopped=result.early_stopped,
            cache_hits=result.cache_hits,
            memory_extraction=memory_extraction,
        )

    except RuntimeError as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
