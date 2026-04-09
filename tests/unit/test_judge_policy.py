"""Unit tests for the judge policy decision logic."""

from __future__ import annotations

from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict
from app.rag.policies.judge_policy import decide_next_action


def _verdict(**kwargs) -> JudgeVerdict:
    defaults = dict(correctness=0.9, faithfulness=0.9, completeness=0.9, conciseness=0.9)
    defaults.update(kwargs)
    return JudgeVerdict(**defaults)


def test_good_verdict_is_accepted():
    v = _verdict()
    assert decide_next_action(v) == "accept"


def test_low_faithfulness_triggers_retrieval_retry():
    v = _verdict(faithfulness=0.3)
    assert decide_next_action(v) == "retry_retrieval"


def test_low_overall_triggers_generation_retry():
    v = _verdict(correctness=0.2, faithfulness=0.6, completeness=0.2, conciseness=0.2)
    assert decide_next_action(v) == "retry_generation"


def test_max_retries_passed_verdict_is_accepted():
    v = _verdict()
    assert decide_next_action(v, hallucination_attempt=3, max_retries=3) == "accept"


def test_max_retries_failed_verdict_is_rejected():
    v = _verdict(correctness=0.1, faithfulness=0.1, completeness=0.1, conciseness=0.1)
    assert decide_next_action(v, hallucination_attempt=3, max_retries=3) == "reject"


def test_error_verdict_is_accepted_gracefully():
    v = JudgeVerdict(error="LLM timeout")
    assert decide_next_action(v) == "accept"
