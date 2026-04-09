"""Retry policy — controls the hallucination feedback loop.

Enforces a maximum of ``settings.max_retries`` (default: 3) retry cycles.
Separating this logic from the graph makes the threshold easy to tune.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings

_GROUNDING_THRESHOLD = 0.6


def should_retry(state: dict[str, Any]) -> bool:
    """Determine whether the pipeline should retry generation.

    Retry when:
    - Grounding score is below ``_GROUNDING_THRESHOLD``.
    - AND hallucination attempts are below ``settings.max_retries``.

    Args:
        state: Current CRAG graph state containing ``grounding_score``
               and ``hallucination_attempt``.

    Returns:
        ``True`` if the pipeline should loop back through retrieval and
        re-generation (at most ``settings.max_retries`` times).
    """
    grounding_score = state.get("grounding_score", 1.0)
    attempts = state.get("hallucination_attempt", 0)

    return grounding_score < _GROUNDING_THRESHOLD and attempts < settings.max_retries
