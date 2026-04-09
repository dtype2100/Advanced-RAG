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
from app.core.vectorstore import get_vectorstore
from app.services import retrieval

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check including vector store connection status."""
    try:
        store = get_vectorstore()
        client = store.get_client()
        if settings.vectorstore_provider == "qdrant":
            collections = [c.name for c in client.get_collections().collections]
            vs_status = "connected"
            collection_status = (
                "exists" if settings.collection_name in collections else "not_created"
            )
        else:
            vs_status = "connected"
            collection_status = "exists"
    except Exception as e:
        vs_status = f"error: {e}"
        collection_status = "unknown"

    return HealthResponse(
        status="ok",
        llm_backend=settings.llm_backend,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        vectorstore_provider=settings.vectorstore_provider,
        reranker_provider=settings.reranker_provider,
        vectorstore=vs_status,
        collection=collection_status,
    )


# ── Document Ingestion ──────────────────────────────────────────────────────


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def ingest_documents(req: IngestRequest):
    """Ingest documents into the vector store."""
    texts = [d.text for d in req.documents]
    metadatas = [d.metadata for d in req.documents]

    try:
        count = retrieval.add_documents(texts, metadatas)
    except Exception as e:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return IngestResponse(message="Documents ingested successfully", count=count)


# ── Semantic Search ──────────────────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(req: SearchRequest):
    """Perform semantic search without RAG generation."""
    try:
        results = retrieval.search(req.query, top_k=req.top_k)
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
    """Run the full self-corrective RAG pipeline (retrieve -> grade -> rewrite -> generate)."""
    if not settings.using_vllm and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM backend not configured. Set LLM_BACKEND=vllm or provide OPENAI_API_KEY.",
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
