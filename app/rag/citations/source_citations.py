"""Build API-facing citation objects from retrieval results."""

from __future__ import annotations

from typing import Any

from app.schemas.response import SourceCitation


def build_source_citations(results: list[dict[str, Any]]) -> list[SourceCitation]:
    """Convert retrieval dicts into structured ``SourceCitation`` models.

    Args:
        results: List of ``{text, score, metadata}`` retrieval hits.

    Returns:
        Ordered list of citations for the API response.
    """
    citations: list[SourceCitation] = []
    for result in results:
        meta = result.get("metadata") or {}
        citations.append(
            SourceCitation(
                text=str(result.get("text", "")),
                source=str(meta.get("source", "unknown")),
                page=str(meta.get("page", "")),
                score=float(result.get("score", 0.0)),
            )
        )
    return citations
