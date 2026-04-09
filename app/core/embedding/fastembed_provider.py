"""FastEmbed embedding provider – runs locally without external API calls."""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

_DIMENSION_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedProvider(BaseEmbedding):
    """Wraps fastembed.TextEmbedding for local CPU-based embedding."""

    def __init__(self) -> None:
        from fastembed import TextEmbedding

        self._model: Any = TextEmbedding(model_name=settings.embedding_model)
        self._dimension = _DIMENSION_MAP.get(settings.embedding_model, settings.embedding_dimension)
        logger.info("FastEmbed loaded: model=%s dim=%d", settings.embedding_model, self._dimension)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def get_dimension(self) -> int:
        return self._dimension
