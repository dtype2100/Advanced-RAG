"""Vector store backend selection via VECTOR_BACKEND."""

from __future__ import annotations

from app.core.config import settings
from app.vectorstore.base import VectorStore

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """Return the configured vector store singleton."""
    global _store
    if _store is not None:
        return _store

    backend = settings.vector_backend.lower()
    if backend == "qdrant":
        from app.vectorstore.qdrant_backend import QdrantVectorStore

        _store = QdrantVectorStore()
    elif backend == "memory":
        from app.vectorstore.memory_backend import InMemoryVectorStore

        _store = InMemoryVectorStore()
    else:
        raise ValueError(
            f"Unsupported VECTOR_BACKEND={settings.vector_backend!r}. Supported: qdrant, memory"
        )
    return _store


def reset_vector_store_for_tests() -> None:
    global _store
    _store = None
