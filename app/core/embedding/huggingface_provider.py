"""HuggingFace sentence-transformers embedding provider."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)


class HuggingFaceEmbeddingProvider(BaseEmbedding):
    """Wraps sentence_transformers.SentenceTransformer for local embedding."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(settings.embedding_model)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "HuggingFace Embedding loaded: model=%s dim=%d",
            settings.embedding_model,
            self._dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self._dimension
