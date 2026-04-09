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
from app.providers import vectorstore as vs

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check including vector store connection status."""
    try:
        client = vs.get_client()
        # Qdrant and Chroma both expose collection listing, but via different APIs.
        if settings.vectorstore_backend == "qdrant":
            collections = [c.name for c in client.get_collections().collections]
        else:
            collections = [c.name for c in client.list_collections()]
        vs_status = "connected"
        collection_status = "exists" if settings.collection_name in collections else "not_created"
    except Exception as e:
        vs_status = f"error: {e}"
        collection_status = "unknown"

    return HealthResponse(
        status="ok",
        llm_backend=settings.llm_backend,
        llm_model=settings.llm_model,
        embedding_provider=settings.embedding_provider,
        embedding_model=(
            settings.openai_embedding_model
            if settings.embedding_provider == "openai"
            else settings.embedding_model
        ),
        reranker_provider=settings.reranker_provider,
        vectorstore_backend=settings.vectorstore_backend,
        vectorstore=vs_status,
        collection=collection_status,
    )


# ── Document Ingestion ────────────────────────────────────────────────────────


@router.post("/documents", response_model=IngestResponse, tags=["documents"])
async def ingest_documents(req: IngestRequest):
    """Ingest documents into the configured vector store."""
    texts = [d.text for d in req.documents]
    metadatas = [d.metadata for d in req.documents]

    try:
        count = vs.add_documents(texts, metadatas)
    except Exception as e:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return IngestResponse(message="Documents ingested successfully", count=count)


# ── Semantic Search ───────────────────────────────────────────────────────────


@router.post("/search", response_model=SearchResponse, tags=["search"])
async def semantic_search(req: SearchRequest):
    """Perform semantic search without RAG generation."""
    try:
        results = vs.search(req.query, top_k=req.top_k)
    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return SearchResponse(
        results=[
            SearchResult(text=r["text"], score=r["score"], metadata=r["metadata"]) for r in results
        ]
    )


# ── RAG Query ─────────────────────────────────────────────────────────────────


@router.post("/query", response_model=QueryResponse, tags=["rag"])
async def rag_query(req: QueryRequest):
    """Run the full self-corrective RAG pipeline.

    Steps: retrieve → rerank → grade → rewrite → generate.
    """
    if settings.using_vllm:
        pass  # vLLM is configured – proceed
    elif settings.using_openai_llm and not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM_BACKEND=openai requires OPENAI_API_KEY to be set.",
        )
    elif settings.using_anthropic and not settings.anthropic_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM_BACKEND=anthropic requires ANTHROPIC_API_KEY to be set.",
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
