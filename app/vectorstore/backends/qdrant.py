"""Qdrant vector store backend implementation."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings
from app.embeddings.provider import EmbeddingProvider

logger = logging.getLogger(__name__)


class QdrantVectorStoreBackend:
    """Qdrant backend using configurable embedding provider."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        """Store embedding provider and defer Qdrant client creation."""
        self.embedding_provider = embedding_provider
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        """Create or reuse Qdrant client."""
        if self._client is not None:
            return self._client

        if settings.qdrant_in_memory:
            logger.info("Initializing Qdrant in-memory client")
            self._client = QdrantClient(":memory:")
        else:
            logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
            self._client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
            )
        return self._client

    def ensure_collection(self) -> None:
        """Create collection when missing."""
        client = self._get_client()
        collections = [collection.name for collection in client.get_collections().collections]
        if settings.vector_db_collection_name in collections:
            return

        vector_size = self.embedding_provider.vector_size()
        client.create_collection(
            collection_name=settings.vector_db_collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        logger.info(
            "Created collection '%s' (dim=%d)",
            settings.vector_db_collection_name,
            vector_size,
        )

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        """Embed and upsert documents to Qdrant."""
        if not texts:
            return 0

        self.ensure_collection()
        client = self._get_client()
        vectors = self.embedding_provider.embed_texts(texts)

        merged_metadatas = metadatas or [{} for _ in texts]
        if len(merged_metadatas) < len(texts):
            merged_metadatas.extend({} for _ in range(len(texts) - len(merged_metadatas)))

        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, text)),
                vector=vector,
                payload={"text": text, **metadata},
            )
            for text, vector, metadata in zip(texts, vectors, merged_metadatas, strict=True)
        ]
        client.upsert(collection_name=settings.vector_db_collection_name, points=points)
        logger.info(
            "Upserted %d documents into '%s'",
            len(points),
            settings.vector_db_collection_name,
        )
        return len(points)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run semantic search using Qdrant query_points API."""
        self.ensure_collection()
        client = self._get_client()
        query_vector = self.embedding_provider.embed_query(query)

        response = client.query_points(
            collection_name=settings.vector_db_collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "text": point.payload.get("text", "") if point.payload else "",
                "score": point.score,
                "metadata": {
                    key: value for key, value in (point.payload or {}).items() if key != "text"
                },
            }
            for point in response.points
        ]

    def healthcheck(self) -> dict[str, str]:
        """Return Qdrant connection and collection status."""
        client = self._get_client()
        collections = [collection.name for collection in client.get_collections().collections]
        collection_status = (
            "exists" if settings.vector_db_collection_name in collections else "not_created"
        )
        return {"status": "connected", "collection": collection_status}

    def get_raw_client(self) -> QdrantClient:
        """Expose underlying Qdrant client for legacy health checks."""
        return self._get_client()
