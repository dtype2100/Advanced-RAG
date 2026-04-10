"""Retry policy — controls the hallucination feedback loop.

Enforces a maximum of ``settings.max_retries`` (default: 3) retry cycles.
Separating this logic from the graph makes the threshold easy to tune.
"""

from __future__ import annotations

from typing import Any

from app.core.runtime_config import get_grounding_threshold, get_max_retries


def should_retry(state: dict[str, Any]) -> bool:
    """Determine whether the pipeline should retry generation.

    Retry when:
    - Grounding score is below the runtime (or default) grounding threshold.
    - AND hallucination attempts are below the runtime (or settings) max retries.

    Args:
        state: Current CRAG graph state containing ``grounding_score``
               and ``hallucination_attempt``.

    Returns:
        ``True`` if the pipeline should loop back through retrieval and
        re-generation (at most ``settings.max_retries`` times).
    """
    grounding_score = state.get("grounding_score", 1.0)
    attempts = state.get("hallucination_attempt", 0)
    threshold = get_grounding_threshold()
    max_r = get_max_retries()

    return grounding_score < threshold and attempts < max_r
