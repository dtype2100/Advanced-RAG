"""Vector store factory — creates the correct ``VectorStorePort`` from settings.

Centralises backend selection so that ``vectorstore_provider.py`` and any
other caller can obtain a store instance without knowing the concrete class.
"""

from __future__ import annotations

import logging
import os

from app.storage.vectorstores.base import VectorStorePort

logger = logging.getLogger(__name__)


def create_vectorstore(backend: str | None = None) -> VectorStorePort:
    """Instantiate and return the configured vector store implementation.

    Args:
        backend: Override the backend name.  If ``None``, reads the
                 ``VECTOR_BACKEND`` environment variable (default: ``qdrant``).

    Returns:
        A concrete ``VectorStorePort`` instance.

    Raises:
        ValueError: If ``backend`` is not a known implementation name.
    """
    name = (backend or os.getenv("VECTOR_BACKEND", "qdrant")).lower()

    if name == "qdrant":
        from app.storage.vectorstores.qdrant_store import QdrantStore

        logger.info("VectorStore factory: creating QdrantStore")
        return QdrantStore()

    if name == "pgvector":
        from app.storage.vectorstores.pgvector_store import PgVectorStore

        logger.info("VectorStore factory: creating PgVectorStore")
        return PgVectorStore()

    raise ValueError(f"Unknown VECTOR_BACKEND: '{name}'. Supported values: 'qdrant', 'pgvector'.")
