"""Vector store backend factory."""

from __future__ import annotations

from app.config import settings
from app.embeddings.provider import get_embedding_provider
from app.vectorstore.backends.base import VectorStoreBackend
from app.vectorstore.backends.memory import MemoryVectorStoreBackend
from app.vectorstore.backends.qdrant import QdrantVectorStoreBackend

_backend_key: tuple[str, str, str, str, str] | None = None
_backend_cache: VectorStoreBackend | None = None


def get_vectorstore_backend() -> VectorStoreBackend:
    """Build or reuse vector store backend from current environment configuration."""
    global _backend_key, _backend_cache

    key = (
        settings.vector_db_backend,
        settings.vector_db_collection_name,
        settings.qdrant_url,
        settings.embedding_backend,
        settings.embedding_model,
    )
    if _backend_cache is not None and _backend_key == key:
        return _backend_cache

    embedding_provider = get_embedding_provider()
    if settings.vector_db_backend == "qdrant":
        _backend_cache = QdrantVectorStoreBackend(embedding_provider=embedding_provider)
    elif settings.vector_db_backend == "memory":
        _backend_cache = MemoryVectorStoreBackend(embedding_provider=embedding_provider)
    else:
        raise ValueError(f"Unsupported VECTOR_DB_BACKEND: {settings.vector_db_backend}")

    _backend_key = key
    return _backend_cache


def reset_vectorstore_backend_cache() -> None:
    """Reset cached backend to pick up runtime configuration changes."""
    global _backend_key, _backend_cache
    _backend_key = None
    _backend_cache = None
