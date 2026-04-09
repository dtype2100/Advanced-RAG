"""Retrieval pipeline — convenience wrapper for the retrieval orchestrator."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.rag.retrievers.retrieval_orchestrator import retrieve

logger = logging.getLogger(__name__)


def run_retrieval(
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Run the full retrieval pipeline for a query.

    Currently delegates to the retrieval orchestrator which performs
    hybrid (vector + BM25) retrieval.  Extend with docstore / chunk list
    arguments as the pipeline grows.

    Args:
        query:  User query string.
        top_k:  Number of results to return (defaults to settings value).

    Returns:
        Ranked list of ``{text, score, metadata}`` dicts.
    """
    k = top_k or settings.max_retrieval_docs
    results = retrieve(query, top_k=k)
    logger.info("Retrieval pipeline: %d results for query '%s'", len(results), query[:60])
    return results
