"""OpenAI embedding provider – uses the OpenAI Embeddings API."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

_DIMENSION_MAP: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbeddingProvider(BaseEmbedding):
    """Wraps langchain_openai.OpenAIEmbeddings."""

    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._model = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        self._dimension = _DIMENSION_MAP.get(settings.embedding_model, settings.embedding_dimension)
        logger.info(
            "OpenAI Embedding loaded: model=%s dim=%d",
            settings.embedding_model,
            self._dimension,
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)

    def get_dimension(self) -> int:
        return self._dimension
