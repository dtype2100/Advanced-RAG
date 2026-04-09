"""Clarification policy — decides whether to ask the user for more information.

Separating the *decision* (this module) from *question generation*
(query/clarification.py) keeps each responsibility isolated.
"""

from __future__ import annotations

from typing import Any


def needs_clarification(analysis: dict[str, Any]) -> bool:
    """Decide if the query requires clarification before retrieval.

    Triggers clarification when:
    - The query has missing required slots (e.g. time period, location).
    - The query is ambiguous AND there are no retrieved results yet.

    Args:
        analysis: Dict returned by ``query_analyzer.analyze()``.

    Returns:
        ``True`` if the pipeline should pause and ask the user for more info.
    """
    return bool(analysis.get("missing_slots"))
