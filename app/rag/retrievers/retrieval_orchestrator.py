"""Retrieval orchestrator — selects and coordinates retrieval strategies.

Acts as the single entry point for all retrieval calls, applying hybrid
retrieval, expansion, and optional reranking in sequence.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    corpus_docs: list[dict[str, Any]] | None = None,
    docstore: dict[str, str] | None = None,
    all_chunks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run the full retrieval pipeline and return ranked documents.

    Steps:
    1. Hybrid retrieval (vector + BM25 if corpus_docs provided).
    2. Parent-child expansion (if docstore provided).
    3. Small-to-big expansion (if all_chunks provided).

    Args:
        query:       Search query string.
        top_k:       Maximum number of final results (defaults to settings value).
        corpus_docs: In-memory document list for BM25.
        docstore:    Parent chunk mapping for parent-child expansion.
        all_chunks:  Full chunk list for small-to-big expansion.

    Returns:
        Ranked list of ``{text, score, metadata}`` dicts.
    """
    k = top_k or settings.max_retrieval_docs

    from app.rag.retrievers.hybrid_retriever import hybrid_retrieve

    results = hybrid_retrieve(query, corpus_docs=corpus_docs, top_k=k)
    logger.info("Hybrid retrieve: %d results", len(results))

    if docstore:
        from app.rag.retrievers.parent_child_retriever import fetch_parents

        results = fetch_parents(results, docstore)
        logger.info("After parent expansion: %d results", len(results))

    if all_chunks:
        from app.rag.retrievers.small_to_big_retriever import small_to_big_expand

        results = small_to_big_expand(results, all_chunks)
        logger.info("After small-to-big expansion: %d results", len(results))

    return results[:k]
