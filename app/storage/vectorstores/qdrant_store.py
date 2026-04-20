"""Qdrant-backed vector store implementing ``VectorStorePort``.

Uses FastEmbed for local embedding and ``query_points()`` (qdrant-client ≥ 1.12).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings
from app.core.constants import VECTOR_SIZE_MAP
from app.storage.vectorstores.base import VectorStorePort

logger = logging.getLogger(__name__)


class QdrantStore(VectorStorePort):
    """Concrete Qdrant implementation of ``VectorStorePort``.

    Manages a singleton ``QdrantClient`` and a lazy-loaded FastEmbed model.
    """

    def __init__(self) -> None:
        self._client: QdrantClient | None = None
        self._embedder: Any = None

    # ── Client / embedder lifecycle ───────────────────────────────────────────

    def get_client(self) -> QdrantClient:
        """Return (or create) the singleton Qdrant client."""
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

    def get_embedder(self) -> Any:
        """Return (or initialise) the FastEmbed TextEmbedding model."""
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=settings.embedding_model)
        return self._embedder

    # ── Embedding helpers ─────────────────────────────────────────────────────

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; returns a list of float vectors."""
        embedder = self.get_embedder()
        return [vec.tolist() for vec in embedder.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        return self.embed_texts([text])[0]

    @staticmethod
    def _deterministic_id(text: str, metadata: dict[str, Any]) -> str:
        """Generate a stable UUID-v5 using text + provenance metadata."""
        key_parts = [
            text,
            str(metadata.get("source", "")),
            str(metadata.get("doc_id", "")),
            str(metadata.get("parent_id", "")),
            str(metadata.get("chunk_index", "")),
        ]
        key = "|".join(key_parts)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    # ── VectorStorePort interface ─────────────────────────────────────────────

    def ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not yet exist."""
        client = self.get_client()
        existing = [c.name for c in client.get_collections().collections]
        if settings.collection_name not in existing:
            vec_size = VECTOR_SIZE_MAP.get(settings.embedding_model, 384)
            client.create_collection(
                collection_name=settings.collection_name,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
            logger.info(
                "Created collection '%s' (dim=%d)",
                settings.collection_name,
                vec_size,
            )

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed and upsert documents into the Qdrant collection."""
        if not texts:
            return 0

        self.ensure_collection()
        client = self.get_client()
        vectors = self.embed_texts(texts)
        metadatas = metadatas or [{} for _ in texts]

        points = [
            PointStruct(
                id=self._deterministic_id(text, meta),
                vector=vec,
                payload={"text": text, **meta},
            )
            for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
        ]
        client.upsert(collection_name=settings.collection_name, points=points)
        logger.info("Upserted %d documents into '%s'", len(points), settings.collection_name)
        return len(points)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search via ``query_points``; returns list of result dicts."""
        self.ensure_collection()
        client = self.get_client()
        query_vec = self.embed_query(query)

        response = client.query_points(
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

    def delete_collection(self) -> None:
        """Drop the Qdrant collection (for tests / migrations)."""
        client = self.get_client()
        client.delete_collection(collection_name=settings.collection_name)
        logger.info("Deleted collection '%s'", settings.collection_name)
