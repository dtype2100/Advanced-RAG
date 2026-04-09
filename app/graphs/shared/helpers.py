"""Utility helpers shared across graph node implementations."""

from __future__ import annotations


def get_active_query(state: dict) -> str:
    """Return the most refined query string available in ``state``.

    Priority: rewritten_query > clarified_query > user_query.
    """
    return (
        state.get("rewritten_query") or state.get("clarified_query") or state.get("user_query", "")
    )
