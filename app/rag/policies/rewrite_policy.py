"""Rewrite policy — decides whether a query should be reformulated.

Separating the *decision* (this module) from the *rewriting* (query_rewriter)
makes it easy to adjust criteria without touching the rewrite logic.
"""

from __future__ import annotations

from typing import Any


def needs_rewrite(query: str, state: dict[str, Any] | None = None) -> bool:
    """Determine whether the current query should be rewritten.

    Criteria (heuristic, extendable):
    - Query is very short (< 4 tokens) — likely to produce poor recall.
    - Retrieval attempt > 0 AND previous results were empty — force rewrite
      on retry cycles.

    Args:
        query: Current query string.
        state: Optional graph state for context (e.g. retrieval attempt count).

    Returns:
        ``True`` if the query should be rewritten before retrieval.
    """
    tokens = query.split()
    if len(tokens) < 4:
        return True

    if state:
        attempt = state.get("retrieval_attempt", 0)
        children = state.get("retrieved_children", [])
        if attempt > 0 and not children:
            return True

    return False
