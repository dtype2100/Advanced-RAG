"""Inbound request schemas for chat / search endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.constants import DEFAULT_TOP_K, MAX_TOP_K


class ChatRequest(BaseModel):
    """Request body for the chat / RAG query endpoint."""

    question: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K, description="Documents to retrieve"
    )
    session_id: str | None = Field(default=None, description="Optional chat session identifier")


class SearchRequest(BaseModel):
    """Request body for standalone semantic search."""

    query: str = Field(..., min_length=1)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)
