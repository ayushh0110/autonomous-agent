"""Pydantic models for API request / response payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or instruction.",
        examples=["What is Python?"],
    )
    profile_context: list[dict[str, Any]] | None = Field(
        default=None,
        description="User profile memory entries from localStorage (name, preferences, etc.).",
    )
    session_context: list[dict[str, Any]] | None = Field(
        default=None,
        description="Session context entries from sessionStorage (references, tasks).",
    )


class ChatResponse(BaseModel):
    """Outgoing chat response with agent metadata."""

    response: str = Field(
        ...,
        description="The agent's answer.",
    )
    source: str = Field(
        default="agent_loop",
        description="How the answer was produced: 'agent_loop', 'fallback', or 'error'.",
    )
    tools_used: list[str] = Field(
        default_factory=list,
        description="List of tool names used during this request.",
    )
    steps_taken: int = Field(
        default=0,
        description="Number of agent loop iterations taken.",
    )
    plan: list[str] | None = Field(
        default=None,
        description="The generated execution plan (step descriptions), if planner was used.",
    )
    confidence: str = Field(
        default="medium",
        description="Critic's confidence in the answer: 'high', 'medium', or 'low'.",
    )
    refinements: int = Field(
        default=0,
        description="Number of refinement iterations the answer went through.",
    )
    memory_used: bool = Field(
        default=False,
        description="Whether relevant past memory was used to enrich context.",
    )
    memory_hits: int = Field(
        default=0,
        description="Number of relevant memory entries retrieved.",
    )
    decision: str = Field(
        default="needs_search",
        description="Query classification: 'direct_answer', 'needs_search', or 'memory_sufficient'.",
    )
    llm_calls: int = Field(
        default=0,
        description="Total number of LLM API calls made during this request.",
    )
    steps_skipped: int = Field(
        default=0,
        description="Number of plan steps skipped (via early stopping or plan trimming).",
    )
    early_stopped: bool = Field(
        default=False,
        description="Whether execution stopped early due to high-confidence intermediate result.",
    )
    cache_hits: int = Field(
        default=0,
        description="Number of tool calls served from cache instead of re-executing.",
    )
    memory_extraction: dict[str, Any] | None = Field(
        default=None,
        description="Extracted memory from this message (for frontend to store in localStorage/sessionStorage).",
    )
