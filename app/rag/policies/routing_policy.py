"""Routing policy — consolidates all graph branching rules in one place.

Each function receives the current graph state and returns a routing key.
Having all branching decisions here (rather than scattered across nodes or
``routes.py``) makes the full decision tree easy to audit and test.
"""

from __future__ import annotations

from typing import Any


def route_query_entry(state: dict[str, Any]) -> str:
    """Decide the first step after the user submits a query.

    Returns:
        ``"analyze_query"`` always (entry point of the main pipeline).
    """
    return "analyze_query"


def route_after_clarification_check(state: dict[str, Any]) -> str:
    """After query analysis, decide whether to ask for clarification.

    Returns:
        ``"ask_clarification"`` if ``needs_clarification`` is set.
        ``"decide_rewrite"``    otherwise.
    """
    return "ask_clarification" if state.get("needs_clarification") else "decide_rewrite"


def route_after_rewrite_check(state: dict[str, Any]) -> str:
    """After rewrite decision, decide whether to rewrite the query.

    Returns:
        ``"rewrite_query"``   if ``needs_rewrite`` is set.
        ``"hybrid_retrieve"`` otherwise.
    """
    return "rewrite_query" if state.get("needs_rewrite") else "hybrid_retrieve"


def route_after_retrieval(state: dict[str, Any]) -> str:
    """After retrieval, decide whether to expand child chunks.

    Returns:
        ``"expand_context"`` when the expansion policy recommends it.
        ``"rerank_context"`` otherwise.
    """
    from app.rag.policies.expansion_policy import should_expand

    return "expand_context" if should_expand(state) else "rerank_context"


def route_after_judge(state: dict[str, Any]) -> str:
    """After judge evaluation, decide the next action.

    Returns:
        ``"retry_retrieval"``  if faithfulness is critically low.
        ``"retry_generation"`` if overall score is low and retries remain.
        ``"reject"``           if retries exhausted and answer fails.
        ``"accept"``           if the answer passes all checks.
    """
    from app.core.config import settings
    from app.rag.policies.judge_policy import decide_next_action

    verdict = state.get("judge_verdict")
    attempt = state.get("hallucination_attempt", 0)

    if verdict is None:
        return "accept"

    action = decide_next_action(
        verdict,
        hallucination_attempt=attempt,
        max_retries=settings.max_retries,
    )
    return action


def route_after_grounding(state: dict[str, Any]) -> str:
    """After grounding evaluation, decide whether to retry.

    Delegates to ``retry_policy.should_retry`` for the loop decision.

    Returns:
        ``"retry_with_policy"`` when grounding score is too low and retries remain.
        ``"end"``               otherwise.
    """
    from app.rag.policies.retry_policy import should_retry

    return "retry_with_policy" if should_retry(state) else "end"
