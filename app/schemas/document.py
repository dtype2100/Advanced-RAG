"""Schemas related to document ingestion."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """A single document to be ingested into the vector store."""

    text: str = Field(..., min_length=1, description="Document text to ingest")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Optional key-value metadata"
    )


class IngestRequest(BaseModel):
    """Request body for the document ingestion endpoint."""

    documents: list[DocumentInput] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    """Response body returned after successful ingestion."""

    message: str
    count: int
