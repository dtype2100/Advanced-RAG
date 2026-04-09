"""OpenAI embedding provider – requires ``OPENAI_API_KEY``."""

from __future__ import annotations

import logging

from app.core.embedding.base import BaseEmbedding

logger = logging.getLogger(__name__)

_DIMENSION_MAP: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedding(BaseEmbedding):
    """Uses the OpenAI embeddings API."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: str = "") -> None:
        self._model = model
        self._api_key = api_key
        self._client: object | None = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI

            logger.info("Initializing OpenAI embedding client (model=%s)", self._model)
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        response = client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    def dimension(self) -> int:
        return _DIMENSION_MAP.get(self._model, 1536)
