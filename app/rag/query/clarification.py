"""Clarification question generator.

Converts a query analysis dict (from ``query_analyzer``) into a concrete
question to surface to the user when information is missing.
"""

from __future__ import annotations

from typing import Any


def generate_clarification_question(analysis: dict[str, Any]) -> str:
    """Build a natural-language clarification question from the analysis.

    Args:
        analysis: Dict returned by ``query_analyzer.analyze()``.

    Returns:
        A clarification question string directed at the user.
    """
    query = analysis.get("query", "")
    missing = analysis.get("missing_slots", [])

    if "time_period" in missing:
        return f"Could you specify the time period you're asking about in: '{query}'?"
    if "location" in missing:
        return f"Could you specify the location or region you're referring to in: '{query}'?"
    if analysis.get("is_ambiguous"):
        return (
            f"Your question '{query}' seems ambiguous. "
            "Could you provide more context or clarify what you mean?"
        )
    return "Could you provide more details to help answer your question accurately?"
