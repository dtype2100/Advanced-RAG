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

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ───────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check including vectorstore connection status."""
    try:
        store = get_vectorstore()
        raw = store.get_raw_client()
        if settings.vectorstore_provider == "qdrant":
            collections = [c.name for c in raw.get_collections().collections]
            vs_status = "connected"
            collection_status = (
                "exists" if settings.collection_name in collections else "not_created"
            )
        elif settings.vectorstore_provider == "chroma":
            names = [c.name for c in raw.list_collections()]
            vs_status = "connected"
            collection_status = "exists" if settings.collection_name in names else "not_created"
        else:
            vs_status = "connected"
            collection_status = "unknown"
    except Exception as e:
        vs_status = f"error: {e}"
        collection_status = "unknown"

    return HealthResponse(
        status="ok",
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        reranker_provider=settings.reranker_provider,
        vectorstore_provider=settings.vectorstore_provider,
        vectorstore=vs_status,
        collection=collection_status,
    )


# ── Document Ingestion ──────────────────────────────────────────────────────


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def ingest_documents(req: IngestRequest):
    """Ingest documents into the vectorstore."""
    texts = [d.text for d in req.documents]
    metadatas = [d.metadata for d in req.documents]

    try:
        store = get_vectorstore()
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
        store = get_vectorstore()
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
    if settings.llm_provider == "vllm":
        pass  # vLLM doesn't need extra key validation
    elif settings.llm_provider == "openai" and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured. Set OPENAI_API_KEY.",
        )
    elif settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured. Set ANTHROPIC_API_KEY.",
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
