"""Vector store provider — resolves the active ``VectorStorePort`` implementation.

Change the ``VECTOR_BACKEND`` environment variable to swap implementations:
- ``qdrant``   → ``QdrantStore``  (default)
- ``pgvector`` → ``PgVectorStore`` (stub, implement when ready)
"""

from __future__ import annotations

import logging
import os

from app.storage.vectorstores.base import VectorStorePort

logger = logging.getLogger(__name__)

_store: VectorStorePort | None = None


def get_vectorstore() -> VectorStorePort:
    """Return the singleton vector store instance for the configured backend.

    Reads the ``VECTOR_BACKEND`` env var (default: ``qdrant``).
    """
    global _store
    if _store is not None:
        return _store

    backend = os.getenv("VECTOR_BACKEND", "qdrant").lower()

    if backend == "qdrant":
        from app.storage.vectorstores.qdrant_store import QdrantStore

        logger.info("Vector store provider: Qdrant")
        _store = QdrantStore()
    elif backend == "pgvector":
        from app.storage.vectorstores.pgvector_store import PgVectorStore

        logger.info("Vector store provider: pgvector")
        _store = PgVectorStore()
    else:
        raise ValueError(f"Unknown VECTOR_BACKEND: '{backend}'. Choose 'qdrant' or 'pgvector'.")

    return _store
