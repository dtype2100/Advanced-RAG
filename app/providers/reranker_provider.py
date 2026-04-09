"""Reranker provider — returns the configured reranker instance.

Placeholder: wire up a ``CrossEncoderReranker`` or API-based reranker here.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def get_reranker():
    """Return the active reranker, or ``None`` if reranking is disabled.

    Set ``RERANKER_BACKEND`` env var to ``cross_encoder`` or ``llm`` to enable.
    """
    backend = os.getenv("RERANKER_BACKEND", "none").lower()

    if backend == "none":
        return None

    if backend == "cross_encoder":
        from app.rag.rerankers.cross_encoder import CrossEncoderReranker

        logger.info("Reranker provider: CrossEncoder")
        return CrossEncoderReranker()

    if backend == "llm":
        from app.rag.rerankers.llm_reranker import LLMReranker

        logger.info("Reranker provider: LLM reranker")
        return LLMReranker()

    raise ValueError(f"Unknown RERANKER_BACKEND: '{backend}'")
