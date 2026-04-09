"""Factory – returns the singleton vectorstore based on VECTORSTORE_PROVIDER."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

_instance: BaseVectorStore | None = None


def get_vectorstore() -> BaseVectorStore:
    """Return a cached vectorstore provider instance.

    Provider is selected by ``settings.vectorstore_provider``:
      - ``qdrant`` → Qdrant (default, in-memory or remote)
      - ``chroma`` → ChromaDB (ephemeral, persistent, or HTTP)
    """
    global _instance
    if _instance is not None:
        return _instance

    provider = settings.vectorstore_provider

    if provider == "qdrant":
        from app.core.vectorstore.qdrant_provider import QdrantVectorStore

        _instance = QdrantVectorStore()
    elif provider == "chroma":
        from app.core.vectorstore.chroma_provider import ChromaVectorStore

        _instance = ChromaVectorStore()
    else:
        raise ValueError(f"Unknown vectorstore provider: {provider}")

    logger.info("VectorStore provider initialised: %s", provider)
    return _instance


def reset_vectorstore() -> None:
    """Reset the cached instance (useful in tests)."""
    global _instance
    _instance = None
