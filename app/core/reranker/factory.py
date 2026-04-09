"""Factory – returns the singleton reranker based on RERANKER_PROVIDER."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.reranker.base import BaseReranker

logger = logging.getLogger(__name__)

_instance: BaseReranker | None = None


def get_reranker() -> BaseReranker | None:
    """Return a cached reranker instance, or None when disabled.

    Provider is selected by ``settings.reranker_provider``:
      - ``none``         → disabled (returns None)
      - ``flashrank``    → FlashRank (CPU-friendly)
      - ``crossencoder`` → sentence-transformers CrossEncoder
    """
    global _instance
    if not settings.reranker_enabled:
        return None
    if _instance is not None:
        return _instance

    provider = settings.reranker_provider

    if provider == "flashrank":
        from app.core.reranker.flashrank_provider import FlashRankReranker

        _instance = FlashRankReranker()
    elif provider == "crossencoder":
        from app.core.reranker.crossencoder_provider import CrossEncoderReranker

        _instance = CrossEncoderReranker()
    else:
        raise ValueError(f"Unknown reranker provider: {provider}")

    logger.info("Reranker provider initialised: %s", provider)
    return _instance


def reset_reranker() -> None:
    """Reset the cached instance (useful in tests)."""
    global _instance
    _instance = None
