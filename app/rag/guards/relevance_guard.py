"""Relevance guard — filters out low-relevance retrieval results."""

from __future__ import annotations

from typing import Any


def filter_relevant(
    results: list[dict[str, Any]],
    threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Return only results with a score above ``threshold``.

    Args:
        results:   List of ``{text, score, metadata}`` dicts.
        threshold: Minimum score to pass (inclusive).

    Returns:
        Filtered list; preserves original order.
    """
    return [r for r in results if r.get("score", 0.0) >= threshold]
