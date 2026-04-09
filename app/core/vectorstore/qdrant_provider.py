"""Qdrant vectorstore provider."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.core.embedding import get_embedding
from app.core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Wraps qdrant-client with automatic embedding via the configured provider."""

    def __init__(self) -> None:
        if settings.qdrant_in_memory:
            logger.info("Initializing Qdrant in-memory client")
            self._client = QdrantClient(":memory:")
        else:
            logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )

    def ensure_collection(self) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if settings.collection_name not in collections:
            embedder = get_embedding()
            vec_size = embedder.get_dimension()
            self._client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                settings.collection_name,
                vec_size,
            )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not texts:
            return 0

        self.ensure_collection()
        embedder = get_embedding()
        vectors = embedder.embed_texts(texts)
        metadatas = metadatas or [{} for _ in texts]

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
                vector=vec,
                payload={"text": text, **meta},
            )
            for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
        ]
        self._client.upsert(collection_name=settings.collection_name, points=points)
        logger.info("Upserted %d documents into '%s'", len(points), settings.collection_name)
        return len(points)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        self.ensure_collection()
        embedder = get_embedding()
        query_vec = embedder.embed_query(query)

        response = self._client.query_points(
            collection_name=settings.collection_name,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "text": hit.payload.get("text", "") if hit.payload else "",
                "score": hit.score,
                "metadata": {k: v for k, v in (hit.payload or {}).items() if k != "text"},
            }
            for hit in response.points
        ]

    def get_raw_client(self) -> QdrantClient:
        return self._client
