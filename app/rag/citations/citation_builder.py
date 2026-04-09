"""Citation builder — extracts source references from retrieved documents."""

from __future__ import annotations

from typing import Any


def build_citations(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract citation metadata from retrieval result dicts.

    Args:
        results: List of ``{text, score, metadata}`` dicts.

    Returns:
        List of citation dicts with ``source``, ``page``, and ``score`` keys.
    """
    citations: list[dict[str, str]] = []
    for result in results:
        meta = result.get("metadata", {})
        citations.append(
            {
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", ""),
                "score": str(round(result.get("score", 0.0), 4)),
            }
        )
    return citations
