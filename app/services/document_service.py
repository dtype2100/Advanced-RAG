"""Document service — search and retrieval operations."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.providers.vectorstore_provider import get_vectorstore

logger = logging.getLogger(__name__)


def search_documents(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Perform a semantic search against the vector store.

    Args:
        query: Query string.
        top_k: Number of results (defaults to settings value).

    Returns:
        List of ``{text, score, metadata}`` dicts.
    """
    k = top_k or settings.max_retrieval_docs
    store = get_vectorstore()
    results = store.search(query, top_k=k)
    logger.info("Document search: %d results for '%s'", len(results), query[:60])
    return results
