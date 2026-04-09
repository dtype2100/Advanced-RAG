"""Policy guard — blocks unsafe or out-of-scope queries.

Placeholder: extend with topic-restriction rules, PII filters, etc.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_BLOCKED_PATTERNS: list[str] = [
    r"\bhow\s+to\s+(make|build|create)\s+(bomb|weapon|virus)\b",
    r"\bpassword\s+for\b",
]


def is_allowed(query: str) -> bool:
    """Check whether the query is permitted by content policy.

    Args:
        query: User query string.

    Returns:
        ``True`` if the query passes all policy checks.
        ``False`` if the query should be blocked.
    """
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            logger.warning("Policy guard blocked query: %s", query[:80])
            return False
    return True
