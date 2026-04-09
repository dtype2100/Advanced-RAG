"""Unit tests for the LLM judge evaluator dataclass and scoring logic."""

from __future__ import annotations

from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict


def test_overall_score_perfect():
    v = JudgeVerdict(correctness=1.0, faithfulness=1.0, completeness=1.0, conciseness=1.0)
    assert abs(v.overall_score - 1.0) < 0.01


def test_overall_score_zero():
    v = JudgeVerdict(correctness=0.0, faithfulness=0.0, completeness=0.0, conciseness=0.0)
    assert v.overall_score == 0.0


def test_faithfulness_weighted_heavier():
    v_faith = JudgeVerdict(correctness=0.0, faithfulness=1.0, completeness=0.0, conciseness=0.0)
    v_corr = JudgeVerdict(correctness=1.0, faithfulness=0.0, completeness=0.0, conciseness=0.0)
    assert v_faith.overall_score > v_corr.overall_score


def test_passed_threshold():
    v_pass = JudgeVerdict(correctness=0.8, faithfulness=0.8, completeness=0.8, conciseness=0.8)
    v_fail = JudgeVerdict(correctness=0.2, faithfulness=0.2, completeness=0.2, conciseness=0.2)
    assert v_pass.passed is True
    assert v_fail.passed is False


def test_error_verdict_defaults_zero():
    v = JudgeVerdict(error="timeout")
    assert v.overall_score == 0.0
    assert v.passed is False
