"""Outbound response schemas for chat / search / health endpoints."""

from __future__ import annotations

from pydantic import BaseModel


class SourceCitation(BaseModel):
    """Structured citation for a retrieved document chunk."""

    text: str
    source: str = "unknown"
    page: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    """Response from the RAG chat pipeline."""

    question: str
    answer: str
    sources: list[SourceCitation]
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
    llm_backend: str
    llm_model: str
    llm: str
    qdrant: str
    collection: str
