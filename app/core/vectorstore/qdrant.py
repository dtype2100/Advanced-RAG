"""Qdrant vector store implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Wraps ``qdrant_client`` – supports in-memory and remote modes."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        url: str = "",
        api_key: str = "",
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._url = url
        self._api_key = api_key
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if not self._url:
                logger.info("Initializing Qdrant in-memory client")
                self._client = QdrantClient(":memory:")
            else:
                logger.info("Connecting to Qdrant at %s", self._url)
                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key or None,
                )
        return self._client

    def get_client(self) -> Any:
        return self._get_client()

    def ensure_collection(self) -> None:
        client = self._get_client()
        collections = [c.name for c in client.get_collections().collections]
        if self._collection_name not in collections:
            client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
            logger.info(
                "Created collection '%s' (dim=%d)", self._collection_name, self._vector_size
            )

    @staticmethod
    def _deterministic_id(text: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))

    def add_documents(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not texts:
            return 0

        self.ensure_collection()
        client = self._get_client()
        metadatas = metadatas or [{} for _ in texts]

        points = [
            PointStruct(
                id=self._deterministic_id(text),
                vector=vec,
                payload={"text": text, **meta},
            )
            for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
        ]
        client.upsert(collection_name=self._collection_name, points=points)
        logger.info("Upserted %d documents into '%s'", len(points), self._collection_name)
        return len(points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_collection()
        client = self._get_client()

        response = client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": hit.payload.get("text", "") if hit.payload else "",
                "score": hit.score,
                "metadata": {key: val for key, val in (hit.payload or {}).items() if key != "text"},
            }
            for hit in response.points
        ]
