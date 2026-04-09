"""Integration tests for the CRAG graph state and routing logic."""

from __future__ import annotations

from app.graphs.crag.routes import (
    route_after_analyze,
    route_after_grounding,
    route_after_rewrite_decision,
)


def test_route_after_analyze_no_clarification():
    state = {"user_query": "What is RAG?", "needs_clarification": False}
    assert route_after_analyze(state) == "decide_rewrite"


def test_route_after_analyze_needs_clarification():
    state = {"user_query": "What happened when?", "needs_clarification": True}
    assert route_after_analyze(state) == "ask_clarification"


def test_route_after_rewrite_decision_no_rewrite():
    state = {"needs_rewrite": False}
    assert route_after_rewrite_decision(state) == "hybrid_retrieve"


def test_route_after_rewrite_decision_needs_rewrite():
    state = {"needs_rewrite": True}
    assert route_after_rewrite_decision(state) == "rewrite_query"


def test_route_after_grounding_high_score_ends():
    state = {"grounding_score": 0.9, "hallucination_attempt": 0}
    assert route_after_grounding(state) == "end"


def test_route_after_grounding_low_score_retries():
    state = {"grounding_score": 0.2, "hallucination_attempt": 0}
    assert route_after_grounding(state) == "retry_with_policy"


def test_route_after_grounding_max_retries_ends():
    state = {"grounding_score": 0.1, "hallucination_attempt": 3}
    assert route_after_grounding(state) == "end"
