"""Integration tests for the hallucination feedback loop and routing policy."""

from __future__ import annotations

from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict
from app.rag.policies.routing_policy import route_after_grounding, route_after_judge


def test_route_after_judge_accept():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.9, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "hallucination_attempt": 0}
    assert route_after_judge(state) == "accept"


def test_route_after_judge_retry_retrieval_low_faithfulness():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.2, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "hallucination_attempt": 0}
    assert route_after_judge(state) == "retry_retrieval"


def test_route_after_judge_retry_generation_low_overall():
    verdict = JudgeVerdict(correctness=0.2, faithfulness=0.6, completeness=0.2, conciseness=0.2)
    state = {"judge_verdict": verdict, "hallucination_attempt": 0}
    assert route_after_judge(state) == "retry_generation"


def test_route_after_judge_reject_at_max_retries():
    verdict = JudgeVerdict(correctness=0.1, faithfulness=0.1, completeness=0.1, conciseness=0.1)
    state = {"judge_verdict": verdict, "hallucination_attempt": 3}
    assert route_after_judge(state) == "reject"


def test_route_after_grounding_high_score_ends():
    state = {"grounding_score": 0.9, "hallucination_attempt": 0}
    assert route_after_grounding(state) == "end"


def test_route_after_grounding_low_score_retries():
    state = {"grounding_score": 0.2, "hallucination_attempt": 0}
    assert route_after_grounding(state) == "retry_with_policy"


def test_route_after_grounding_max_retries_ends():
    state = {"grounding_score": 0.1, "hallucination_attempt": 3}
    assert route_after_grounding(state) == "end"


def test_route_after_judge_no_verdict_accepts():
    state = {"judge_verdict": None, "hallucination_attempt": 0}
    assert route_after_judge(state) == "accept"
