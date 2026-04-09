"""Document ingestion and semantic search endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.document import IngestRequest, IngestResponse
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, SearchResult
from app.services.document_service import search_documents
from app.services.ingest_service import ingest_documents

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def ingest(req: IngestRequest) -> IngestResponse:
    """Ingest documents through the full pre-processing + indexing pipeline."""
    raw_docs = [{"text": d.text, "metadata": d.metadata} for d in req.documents]
    try:
        count = ingest_documents(raw_docs)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestResponse(message="Documents ingested successfully", count=count)


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
