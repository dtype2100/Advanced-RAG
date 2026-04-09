"""Factory – returns the singleton embedding provider based on EMBEDDING_PROVIDER."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

_instance: BaseEmbedding | None = None


def get_embedding() -> BaseEmbedding:
    """Return a cached embedding provider instance.

    Provider is selected by ``settings.embedding_provider``:
      - ``fastembed``   → local FastEmbed (default)
      - ``openai``      → OpenAI Embeddings API
      - ``huggingface`` → sentence-transformers
    """
    global _instance
    if _instance is not None:
        return _instance

    provider = settings.embedding_provider

    if provider == "fastembed":
        from app.core.embedding.fastembed_provider import FastEmbedProvider

        _instance = FastEmbedProvider()
    elif provider == "openai":
        from app.core.embedding.openai_provider import OpenAIEmbeddingProvider

        _instance = OpenAIEmbeddingProvider()
    elif provider == "huggingface":
        from app.core.embedding.huggingface_provider import HuggingFaceEmbeddingProvider

        _instance = HuggingFaceEmbeddingProvider()
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")

    logger.info("Embedding provider initialised: %s", provider)
    return _instance


def reset_embedding() -> None:
    """Reset the cached instance (useful in tests)."""
    global _instance
    _instance = None
