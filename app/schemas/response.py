"""Outbound response schemas for chat / search / health endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Response from the RAG chat pipeline."""

    question: str
    answer: str
    sources: list[str]
    retries: int
    clarification_needed: bool = False
    clarification_question: str | None = None


class SearchResult(BaseModel):
    """A single semantic search hit."""

    text: str
    score: float
    metadata: dict[str, str]


class SearchResponse(BaseModel):
    """Response from the semantic search endpoint."""

    results: list[SearchResult]


class HealthResponse(BaseModel):
    """Health / readiness check response."""

    status: str
    version: str = "0.2.0"
    llm_backend: str
    llm_model: str
    qdrant: str
    collection: str
    redis: str = "not_configured"
    llm: str = "skipped"
    auth_enabled: bool = False


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str


class ReadinessResponse(BaseModel):
    """Readiness probe response with per-dependency checks."""

    status: str
    checks: dict[str, dict[str, str]]
