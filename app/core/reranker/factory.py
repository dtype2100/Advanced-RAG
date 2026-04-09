"""Reranker factory – returns a singleton based on ``RERANKER_PROVIDER``."""

from __future__ import annotations

from app.core.reranker.base import BaseReranker

_instance: BaseReranker | None = None


def get_reranker() -> BaseReranker:
    """Return a cached reranker instance configured via settings."""
    global _instance
    if _instance is not None:
        return _instance

    from app.config import settings

    if settings.reranker_provider == "flashrank":
        from app.core.reranker.flashrank import FlashRankReranker

        _instance = FlashRankReranker(model_name=settings.reranker_model)
    else:
        from app.core.reranker.noop import NoOpReranker

        _instance = NoOpReranker()

    return _instance


def reset() -> None:
    """Clear the singleton – only used in tests."""
    global _instance
    _instance = None
