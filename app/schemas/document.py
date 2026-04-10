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


class IngestJobResponse(BaseModel):
    """Accepted async ingest job — poll GET /jobs/{job_id} for status."""

    job_id: str
    status: str = Field(default="pending", description="Initial job state")
    message: str = Field(default="Ingest job queued")


class JobStatusResponse(BaseModel):
    """Status of a background ARQ job."""

    job_id: str
    status: str
    count: int | None = None
    message: str | None = None
    error: str | None = None
