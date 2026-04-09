"""FastEmbed embedding provider – runs locally, no API key required."""

from __future__ import annotations

import logging
from typing import Any

from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

_DIMENSION_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedEmbedding(BaseEmbedding):
    """Wraps ``fastembed.TextEmbedding`` for local CPU inference."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._embedder: Any = None

    def _get_embedder(self) -> Any:
        """Lazy-load the fastembed model on first use."""
        if self._embedder is None:
            from fastembed import TextEmbedding

            logger.info("Loading FastEmbed model: %s", self._model_name)
            self._embedder = TextEmbedding(model_name=self._model_name)
        return self._embedder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        return [vec.tolist() for vec in embedder.embed(texts)]

    def dimension(self) -> int:
        return _DIMENSION_MAP.get(self._model_name, 384)
