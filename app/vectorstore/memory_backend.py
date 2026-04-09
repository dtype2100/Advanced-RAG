"""In-process vector store (cosine similarity, no external DB)."""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any

from app.core.config import settings
from app.providers.embeddings import get_embedding_provider
from app.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorStore(VectorStore):
    """Simple RAM-backed store for dev/tests when VECTOR_BACKEND=memory."""

    def __init__(self) -> None:
        self._points: dict[str, dict[str, Any]] = {}

    def ensure_collection(self) -> None:
        return

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        if not texts:
            return 0

        embedder = get_embedding_provider()
        vectors = embedder.embed_texts(texts)
        metadatas = metadatas or [{} for _ in texts]

        for text, vec, meta in zip(texts, vectors, metadatas, strict=True):
            pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
            self._points[pid] = {"vector": vec, "text": text, "metadata": meta}

        logger.info("Stored %d documents in memory (%d total)", len(texts), len(self._points))
        return len(texts)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = top_k or settings.max_retrieval_docs
        if not self._points:
            return []

        embedder = get_embedding_provider()
        qvec = embedder.embed_query(query)

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for pdata in self._points.values():
            sim = _cosine_sim(qvec, pdata["vector"])
            scored.append((sim, pdata["text"], pdata.get("metadata") or {}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": text, "score": score, "metadata": meta} for score, text, meta in scored[:k]
        ]

    def health_snapshot(self) -> dict[str, str]:
        return {
            "qdrant": "n/a (memory backend)",
            "collection": "ready" if self._points else "empty",
        }
