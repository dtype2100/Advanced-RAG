"""Document ingestion and semantic search endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.queue.pool import get_arq_pool
from app.schemas.document import IngestJobResponse, IngestRequest, IngestResponse
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, SearchResult
from app.services.document_service import search_documents
from app.services.ingest_service import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()


async def _enqueue_ingest(raw_docs: list[dict]) -> IngestJobResponse:
    """Push ingest payload to ARQ; returns job metadata."""
    pool = await get_arq_pool()
    job = await pool.enqueue_job(
        "ingest_documents_job",
        raw_docs,
        _queue_name=settings.arq_queue_name,
    )
    if job is None:
        raise HTTPException(status_code=500, detail="Failed to enqueue ingest job")
    return IngestJobResponse(job_id=job.job_id, status="pending", message="Ingest job queued")


@router.post(
    "/documents",
    response_model=IngestResponse,
    responses={
        202: {"description": "Queued for background processing", "model": IngestJobResponse},
    },
    tags=["documents"],
)
async def ingest(req: IngestRequest):
    """Ingest documents (sync) or enqueue when ``INGEST_QUEUE_ASYNC`` and ``REDIS_URL`` are set."""
    raw_docs = [{"text": d.text, "metadata": d.metadata} for d in req.documents]

    if settings.ingest_queue_async and settings.redis_url:
        accepted = await _enqueue_ingest(raw_docs)
        return JSONResponse(status_code=202, content=accepted.model_dump())

    try:
        count = ingest_documents(raw_docs)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(message="Documents ingested successfully", count=count)


@router.post(
    "/documents/async",
    response_model=IngestJobResponse,
    status_code=202,
    tags=["documents"],
)
async def ingest_async(req: IngestRequest) -> IngestJobResponse:
    """Always enqueue ingest (requires ``REDIS_URL``)."""
    if not settings.redis_url:
        raise HTTPException(
            status_code=503,
            detail="Async ingest requires REDIS_URL and a running ARQ worker.",
        )
    raw_docs = [{"text": d.text, "metadata": d.metadata} for d in req.documents]
    return await _enqueue_ingest(raw_docs)


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(req: SearchRequest) -> SearchResponse:
    """Perform a semantic similarity search without RAG generation."""
    try:
        results = search_documents(req.query, top_k=req.top_k)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        results=[
            SearchResult(text=r["text"], score=r["score"], metadata=r["metadata"]) for r in results
        ]
    )
