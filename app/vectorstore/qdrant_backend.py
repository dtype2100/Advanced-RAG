"""Qdrant vector store (remote or in-process :memory:)."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.providers.embeddings import get_embedding_provider
from app.vectorstore.base import VectorStore
from app.vectorstore.embedding_dims import embedding_vector_size

logger = logging.getLogger(__name__)


class QdrantVectorStore(VectorStore):
    """Qdrant-backed storage with pluggable embedding provider."""

    def __init__(self) -> None:
        self._client: QdrantClient | None = None

    def get_client(self) -> QdrantClient:
        """Return singleton Qdrant client (in-memory or remote)."""
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
        client = self.get_client()
        collections = [c.name for c in client.get_collections().collections]
        if settings.collection_name not in collections:
            dim = embedding_vector_size()
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info("Created collection '%s' (dim=%d)", settings.collection_name, dim)

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        if not texts:
            return 0

        self.ensure_collection()
        client = self.get_client()
        embedder = get_embedding_provider()
        vectors = embedder.embed_texts(texts)
        metadatas = metadatas or [{} for _ in texts]

        points = [
            PointStruct(
                id=_deterministic_id(text),
                vector=vec,
                payload={"text": text, **meta},
            )
            for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
        ]
        client.upsert(collection_name=settings.collection_name, points=points)
        logger.info("Upserted %d documents into '%s'", len(points), settings.collection_name)
        return len(points)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        self.ensure_collection()
        client = self.get_client()
        k = top_k or settings.max_retrieval_docs
        embedder = get_embedding_provider()
        query_vec = embedder.embed_query(query)

        response = client.query_points(
            collection_name=settings.collection_name,
            query=query_vec,
            limit=k,
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

    def health_snapshot(self) -> dict[str, str]:
        try:
            client = self.get_client()
            collections = [c.name for c in client.get_collections().collections]
            qdrant_status = "connected"
            collection_status = (
                "exists" if settings.collection_name in collections else "not_created"
            )
        except Exception as e:
            qdrant_status = f"error: {e}"
            collection_status = "unknown"
        return {"qdrant": qdrant_status, "collection": collection_status}


def _deterministic_id(text: str) -> str:
    """Generate a deterministic UUID v5 from text content for dedup."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
