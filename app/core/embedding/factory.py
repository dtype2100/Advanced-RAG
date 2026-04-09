"""Embedding provider factory – returns a singleton based on ``EMBEDDING_PROVIDER``."""

from __future__ import annotations

from app.core.embedding.base import BaseEmbedding

_instance: BaseEmbedding | None = None


def get_embedding() -> BaseEmbedding:
    """Return a cached embedding provider instance configured via settings."""
    global _instance
    if _instance is not None:
        return _instance

    from app.config import settings

    if settings.embedding_provider == "openai":
        from app.core.embedding.openai import OpenAIEmbedding

        _instance = OpenAIEmbedding(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )
    else:
        from app.core.embedding.fastembed import FastEmbedEmbedding

        _instance = FastEmbedEmbedding(model_name=settings.embedding_model)

    return _instance


def reset() -> None:
    """Clear the singleton – only used in tests."""
    global _instance
    _instance = None
