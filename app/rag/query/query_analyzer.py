"""Query analyzer — detects ambiguity, missing slots, and intent.

The result dict is consumed by ``clarification_policy`` and ``rewrite_policy``.
"""

from __future__ import annotations

import re
from typing import Any


def analyze(query: str) -> dict[str, Any]:
    """Analyse a query and return a structured analysis dict.

    Current heuristics (rule-based stub):
    - ``is_ambiguous``:  True when the query contains pronouns without clear
      antecedent or is very short (< 5 tokens).
    - ``missing_slots``: List of detected missing information types.
    - ``intent``:        Rough intent category (``"factual"`` | ``"comparison"``
      | ``"procedural"`` | ``"unknown"``).

    Args:
        query: Raw user query string.

    Returns:
        Analysis dict suitable for downstream policy evaluation.
    """
    tokens = query.split()
    is_ambiguous = len(tokens) < 5 or bool(re.search(r"\b(it|this|that|they|them)\b", query, re.I))

    missing_slots: list[str] = []
    if re.search(r"\bwhen\b", query, re.I) and not re.search(r"\d{4}", query):
        missing_slots.append("time_period")
    if re.search(r"\bwhere\b", query, re.I) and not re.search(r"\bin\b|\bat\b", query, re.I):
        missing_slots.append("location")

    if re.search(r"\bhow to\b|\bsteps\b|\bguide\b", query, re.I):
        intent = "procedural"
    elif re.search(r"\bvs\b|\bcompare\b|\bdifference\b", query, re.I):
        intent = "comparison"
    elif re.search(r"\bwhat is\b|\bdefine\b|\bexplain\b", query, re.I):
        intent = "factual"
    else:
        intent = "unknown"

    return {
        "query": query,
        "is_ambiguous": is_ambiguous,
        "missing_slots": missing_slots,
        "intent": intent,
    }
