"""Hallucination guard — raises a warning when risk exceeds a threshold."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.7


def check_hallucination(
    score: float,
    threshold: float = _DEFAULT_THRESHOLD,
) -> bool:
    """Return True if the hallucination risk score is acceptable.

    Args:
        score:     Hallucination risk score in [0, 1].
        threshold: Maximum acceptable risk (exclusive).

    Returns:
        ``True`` if ``score < threshold`` (answer is safe to return).
        ``False`` if the answer should be suppressed or retried.
    """
    if score >= threshold:
        logger.warning(
            "Hallucination guard triggered: score=%.2f >= threshold=%.2f",
            score,
            threshold,
        )
        return False
    return True
