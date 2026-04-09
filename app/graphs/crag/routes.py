"""Conditional edge functions for the CRAG graph.

Each function reads the current ``CRAGState`` and returns a routing key that
LangGraph uses to select the next node.

Routing logic is delegated to ``rag/policies/routing_policy.py`` so that
branching decisions can be tested independently of LangGraph graph construction.
"""

from __future__ import annotations

import logging

from app.core.config import settings
from app.graphs.crag.state import CRAGState
from app.rag.policies.routing_policy import (
    route_after_clarification_check,
    route_after_grounding,
    route_after_judge,
    route_after_retrieval,
    route_after_rewrite_check,
)

logger = logging.getLogger(__name__)


def route_after_analyze(state: CRAGState) -> str:
    """Route after query analysis — clarification vs rewrite check."""
    result = route_after_clarification_check(state)
    logger.info("Route after analyze → %s", result)
    return result


def route_after_clarification(state: CRAGState) -> str:
    """After clarification is surfaced, always proceed to rewrite check."""
    return "decide_rewrite"


def route_after_rewrite_decision(state: CRAGState) -> str:
    """Route after rewrite decision — rewrite vs direct retrieval."""
    result = route_after_rewrite_check(state)
    logger.info("Route after rewrite decision → %s", result)
    return result


def route_after_retrieve(state: CRAGState) -> str:
    """Route after retrieval — context expansion vs reranking."""
    result = route_after_retrieval(state)
    logger.info("Route after retrieve → %s", result)
    return result


def route_after_judge_eval(state: CRAGState) -> str:
    """Route after judge evaluation — accept / retry / reject."""
    result = route_after_judge(state)
    logger.info(
        "Route after judge → %s (attempt %d/%d)",
        result,
        state.get("hallucination_attempt", 0),
        settings.max_retries,
    )
    return result


def route_after_grounding_eval(state: CRAGState) -> str:
    """Route after grounding evaluation — retry or end."""
    result = route_after_grounding(state)
    logger.info(
        "Route after grounding → %s (score=%.2f, attempt %d/%d)",
        result,
        state.get("grounding_score", 0.0),
        state.get("hallucination_attempt", 0),
        settings.max_retries,
    )
    return result
