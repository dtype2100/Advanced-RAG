"""Retrieval quality evaluator — measures top-k coverage and relevance."""

from __future__ import annotations

from typing import Any


def evaluate_retrieval(
    query: str,
    results: list[dict[str, Any]],
    relevance_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate the quality of a retrieval result set.

    Args:
        query:               The query that generated these results.
        results:             List of ``{text, score, metadata}`` dicts.
        relevance_threshold: Minimum score to consider a result relevant.

    Returns:
        Dict with ``total``, ``relevant_count``, ``coverage_ratio``, ``avg_score``.
    """
    if not results:
        return {"total": 0, "relevant_count": 0, "coverage_ratio": 0.0, "avg_score": 0.0}

    relevant = [r for r in results if r.get("score", 0) >= relevance_threshold]
    avg_score = sum(r.get("score", 0.0) for r in results) / len(results)

    return {
        "total": len(results),
        "relevant_count": len(relevant),
        "coverage_ratio": len(relevant) / len(results),
        "avg_score": round(avg_score, 4),
    }
