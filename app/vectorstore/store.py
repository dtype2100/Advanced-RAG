"""Vector store facade with pluggable backend implementations."""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.embeddings.provider import reset_embedding_provider_cache
from app.reranker.provider import reset_reranker_cache
from app.vectorstore.backends.factory import (
    get_vectorstore_backend,
    reset_vectorstore_backend_cache,
)


def ensure_collection() -> None:
    """Create collection in the active vector backend when missing."""
    backend = get_vectorstore_backend()
    backend.ensure_collection()


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    """Embed and upsert documents into the active vector backend."""
    backend = get_vectorstore_backend()
    return backend.add_documents(texts=texts, metadatas=metadatas)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Run semantic search on the active vector backend."""
    backend = get_vectorstore_backend()
    effective_top_k = top_k or settings.max_retrieval_docs
    return backend.search(query=query, top_k=effective_top_k)


def healthcheck() -> dict[str, str]:
    """Return health state for the active vector backend."""
    try:
        status = get_vectorstore_backend().healthcheck()
        return {"backend": settings.vector_db_backend, **status}
    except Exception as exc:
        return {
            "backend": settings.vector_db_backend,
            "status": f"error: {exc}",
            "collection": "unknown",
        }


def get_client() -> Any:
    """Expose raw Qdrant client when backend supports it."""
    backend = get_vectorstore_backend()
    get_raw_client = getattr(backend, "get_raw_client", None)
    if callable(get_raw_client):
        return get_raw_client()
    raise RuntimeError(
        f"Current vector backend '{settings.vector_db_backend}' does not expose a raw client"
    )


def reset_runtime_state() -> None:
    """Reset backend/provider caches to apply runtime setting changes."""
    reset_vectorstore_backend_cache()
    reset_embedding_provider_cache()
    reset_reranker_cache()
