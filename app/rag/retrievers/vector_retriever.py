"""Dense vector retriever wrapping the active vector store."""

from __future__ import annotations

from typing import Any

from app.providers.vectorstore_provider import get_vectorstore


def vector_retrieve(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve top-k documents using dense vector similarity.

    Args:
        query: Query string.
        top_k: Number of results to return.

    Returns:
        List of ``{text, score, metadata}`` dicts sorted by descending score.
    """
    store = get_vectorstore()
    return store.search(query, top_k=top_k)
