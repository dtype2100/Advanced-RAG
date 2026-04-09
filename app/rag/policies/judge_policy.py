"""Judge policy — translates a ``JudgeVerdict`` into a next-action decision.

Separates the *interpretation* of judge scores from the *evaluation* logic
in ``llm_judge_evaluator``, keeping each responsibility isolated.
"""

from __future__ import annotations

from typing import Literal

NextAction = Literal["accept", "retry_retrieval", "retry_generation", "reject"]

_MIN_FAITHFULNESS = 0.5
_MIN_OVERALL = 0.6


def decide_next_action(
    verdict,
    hallucination_attempt: int = 0,
    max_retries: int = 3,
) -> NextAction:
    """Map a ``JudgeVerdict`` to the appropriate next pipeline action.

    Decision rules (in priority order):
    1. If faithfulness is critically low → ``retry_retrieval`` (more context needed).
    2. If overall score is low and retries remain → ``retry_generation``.
    3. If retries exhausted → ``reject`` (surface warning to user).
    4. Otherwise → ``accept``.

    Args:
        verdict:              ``JudgeVerdict`` from ``llm_judge_evaluator.judge()``.
        hallucination_attempt: Number of retries already attempted.
        max_retries:          Maximum allowed retries.

    Returns:
        One of ``"accept"``, ``"retry_retrieval"``, ``"retry_generation"``, ``"reject"``.
    """
    if verdict.error:
        return "accept"

    if hallucination_attempt >= max_retries:
        return "reject" if not verdict.passed else "accept"

    if verdict.faithfulness < _MIN_FAITHFULNESS:
        return "retry_retrieval"

    if verdict.overall_score < _MIN_OVERALL:
        return "retry_generation"

    return "accept"
