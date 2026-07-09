"""Tests for the post-verification feedback router."""

from __future__ import annotations

from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict
from app.rag.policies.routing_policy import route_after_feedback


def test_feedback_accepts_good_answer():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.9, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "grounding_score": 0.9, "hallucination_attempt": 0}
    assert route_after_feedback(state) == "end"


def test_feedback_retry_retrieval_low_faithfulness():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.2, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "grounding_score": 0.9, "hallucination_attempt": 0}
    assert route_after_feedback(state) == "retry_retrieval"


def test_feedback_retry_generation_low_overall():
    verdict = JudgeVerdict(correctness=0.2, faithfulness=0.6, completeness=0.2, conciseness=0.2)
    state = {"judge_verdict": verdict, "grounding_score": 0.9, "hallucination_attempt": 0}
    assert route_after_feedback(state) == "retry_generation"


def test_feedback_grounding_retry_when_judge_passes():
    verdict = JudgeVerdict(correctness=0.9, faithfulness=0.9, completeness=0.9, conciseness=0.9)
    state = {"judge_verdict": verdict, "grounding_score": 0.2, "hallucination_attempt": 0}
    assert route_after_feedback(state) == "retry_with_policy"


def test_feedback_reject_at_max_retries():
    verdict = JudgeVerdict(correctness=0.1, faithfulness=0.1, completeness=0.1, conciseness=0.1)
    state = {"judge_verdict": verdict, "grounding_score": 0.1, "hallucination_attempt": 3}
    assert route_after_feedback(state) == "reject"
