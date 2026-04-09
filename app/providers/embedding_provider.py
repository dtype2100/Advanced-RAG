"""Embedding model provider.

Currently wraps the FastEmbed ``TextEmbedding`` model configured in settings.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)

_embedder: Any = None


def get_embedder() -> Any:
    """Return a cached ``fastembed.TextEmbedding`` instance.

    First call downloads the model (~130 MB) to ``~/.cache/fastembed/``.
    """
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding

        logger.info("Loading embedding model: %s", settings.embedding_model)
        _embedder = TextEmbedding(model_name=settings.embedding_model)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input string.
    """
    embedder = get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    Args:
        text: Query string.

    Returns:
        Float vector representation.
    """
    return embed_texts([text])[0]
