"""Semantic search with optional cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.providers.reranker import get_reranker_provider
from app.vectorstore.factory import get_vector_store


def search_documents(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Vector search, then optional reranking (RERANKER_BACKEND / RERANKER_MODEL)."""
    store = get_vector_store()
    k = top_k if top_k is not None else settings.max_retrieval_docs

    if not settings.reranker_enabled:
        return store.search(query, k)

    n = max(k, settings.rerank_candidates)
    candidates = store.search(query, n)
    if not candidates:
        return []

    texts = [c["text"] for c in candidates]
    scores = get_reranker_provider().score_pairs(query, texts)
    ranked = sorted(
        zip(scores, candidates, strict=True),
        key=lambda pair: pair[0],
        reverse=True,
    )
    return [{**cand, "score": float(score)} for score, cand in ranked[:k]]
