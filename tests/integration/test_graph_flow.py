"""Integration tests for the CRAG graph state and routing logic."""

from __future__ import annotations

from app.graphs.crag.routes import (
    route_after_analyze,
    route_after_grounding,
    route_after_judge_eval,
    route_after_rewrite_decision,
)
from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict


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


def test_route_after_grounding_low_score_runs_judge():
    state = {"grounding_score": 0.2, "hallucination_attempt": 0}
    assert route_after_grounding(state) == "run_judge"


def test_route_after_judge_retry_maps_to_retry_with_policy():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.2, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "hallucination_attempt": 0}
    assert route_after_judge_eval(state) == "retry_with_policy"


def test_route_after_grounding_max_retries_ends():
    state = {"grounding_score": 0.1, "hallucination_attempt": 3}
    assert route_after_grounding(state) == "end"
