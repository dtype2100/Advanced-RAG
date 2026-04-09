"""Conditional edge functions for the CRAG graph.

Each function reads the current ``CRAGState`` and returns a routing key
that LangGraph uses to select the next node.  Keeping routing logic here
(separate from node implementations) makes branching decisions explicit
and independently testable.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.graphs.crag.state import CRAGState

logger = logging.getLogger(__name__)


def route_after_analyze(state: CRAGState) -> str:
    """Decide whether to ask for clarification or proceed to rewrite check.

    Returns:
        ``"ask_clarification"`` if the analyzer flagged missing information.
        ``"decide_rewrite"``    otherwise.
    """
    if state.get("needs_clarification"):
        logger.info("Route → ask_clarification")
        return "ask_clarification"
    logger.info("Route → decide_rewrite")
    return "decide_rewrite"


def route_after_clarification(state: CRAGState) -> str:
    """After clarification is surfaced, always proceed to rewrite check.

    Returns:
        ``"decide_rewrite"`` unconditionally.
    """
    return "decide_rewrite"


def route_after_rewrite_decision(state: CRAGState) -> str:
    """Decide whether to rewrite the query or go straight to retrieval.

    Returns:
        ``"rewrite_query"``    if the rewrite policy flagged this query.
        ``"hybrid_retrieve"``  otherwise.
    """
    if state.get("needs_rewrite"):
        logger.info("Route → rewrite_query")
        return "rewrite_query"
    logger.info("Route → hybrid_retrieve")
    return "hybrid_retrieve"


def route_after_retrieve(state: CRAGState) -> str:
    """Decide whether to expand retrieved child chunks to parent context.

    Returns:
        ``"expand_context"`` when the expansion policy recommends it.
        ``"rerank_context"`` otherwise (skip expansion).
    """
    from app.rag.policies.expansion_policy import should_expand

    if should_expand(state):
        logger.info("Route → expand_context")
        return "expand_context"
    logger.info("Route → rerank_context")
    return "rerank_context"


def route_after_grounding(state: CRAGState) -> str:
    """Decide whether to retry generation or accept the current answer.

    Returns:
        ``"retry_with_policy"`` when grounding score is too low and retries
        remain (up to ``settings.max_retries``).
        ``"end"``               when the answer is acceptable or retries are
        exhausted.
    """
    from app.rag.policies.retry_policy import should_retry

    if should_retry(state):
        logger.info(
            "Route → retry_with_policy (attempt %d/%d)",
            state.get("hallucination_attempt", 0),
            settings.max_retries,
        )
        return "retry_with_policy"
    logger.info("Route → end")
    return "end"
