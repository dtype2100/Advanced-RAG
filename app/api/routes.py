"""FastAPI route handlers."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.config import settings
from app.vectorstore import store

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check including Qdrant connection status."""
    try:
        client = store.get_client()
        collections = [c.name for c in client.get_collections().collections]
        qdrant_status = "connected"
        collection_status = "exists" if settings.collection_name in collections else "not_created"
    except Exception as e:
        qdrant_status = f"error: {e}"
        collection_status = "unknown"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        collection=collection_status,
    )


# ── Document Ingestion ──────────────────────────────────────────────────────


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def ingest_documents(req: IngestRequest):
    """Ingest documents into the Qdrant vector store."""
    texts = [d.text for d in req.documents]
    metadatas = [d.metadata for d in req.documents]

    try:
        count = store.add_documents(texts, metadatas)
    except Exception as e:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return IngestResponse(message="Documents ingested successfully", count=count)


# ── Semantic Search ──────────────────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(req: SearchRequest):
    """Perform semantic search without RAG generation."""
    try:
        results = store.search(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return SearchResponse(
        results=[
            SearchResult(text=r["text"], score=r["score"], metadata=r["metadata"]) for r in results
        ]
    )


# ── RAG Query ────────────────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def rag_query(req: QueryRequest):
    """Run the full self-corrective RAG pipeline (retrieve → grade → rewrite → generate)."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY not configured. Set it in .env or environment.",
        )

    from app.rag.graph import rag_chain

    initial_state = {
        "question": req.question,
        "rewritten_question": "",
        "documents": [],
        "scores": [],
        "generation": "",
        "retries": 0,
        "is_relevant": False,
    }

    try:
        result = rag_chain.invoke(initial_state)
    except Exception as e:
        logger.exception("RAG query failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return QueryResponse(
        question=req.question,
        answer=result.get("generation", ""),
        sources=result.get("documents", []),
        retries=result.get("retries", 0),
    )
