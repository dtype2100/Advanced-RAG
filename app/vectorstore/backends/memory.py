"""In-process memory vector store backend implementation."""

from __future__ import annotations

import math
import uuid
from typing import Any

from app.config import settings
from app.embeddings.provider import EmbeddingProvider


class MemoryVectorStoreBackend:
    """Simple in-memory vector store for local development and tests."""

    def __init__(self, embedding_provider: EmbeddingProvider):
        """Initialize backend state with provided embedding provider."""
        self.embedding_provider = embedding_provider
        self._collections: dict[str, dict[str, dict[str, Any]]] = {}

    def ensure_collection(self) -> None:
        """Create collection bucket when missing."""
        self._collections.setdefault(settings.vector_db_collection_name, {})

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        """Embed and upsert documents into in-memory collection."""
        if not texts:
            return 0

        self.ensure_collection()
        collection = self._collections[settings.vector_db_collection_name]
        vectors = self.embedding_provider.embed_texts(texts)

        merged_metadatas = metadatas or [{} for _ in texts]
        if len(merged_metadatas) < len(texts):
            merged_metadatas.extend({} for _ in range(len(texts) - len(merged_metadatas)))

        for text, vector, metadata in zip(texts, vectors, merged_metadatas, strict=True):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
            collection[point_id] = {"text": text, "vector": vector, "metadata": metadata}

        return len(texts)

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run cosine similarity search against in-memory vectors."""
        self.ensure_collection()
        collection = self._collections[settings.vector_db_collection_name]
        query_vector = self.embedding_provider.embed_query(query)

        ranked = sorted(
            (
                {
                    "text": payload["text"],
                    "score": self._cosine_similarity(query_vector, payload["vector"]),
                    "metadata": payload["metadata"],
                }
                for payload in collection.values()
            ),
            key=lambda item: item["score"],
            reverse=True,
        )
        return ranked[:top_k]

    def healthcheck(self) -> dict[str, str]:
        """Return memory backend health and collection status."""
        collection_status = (
            "exists"
            if settings.vector_db_collection_name in self._collections
            else "not_created"
        )
        return {"status": "connected", "collection": collection_status}

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
