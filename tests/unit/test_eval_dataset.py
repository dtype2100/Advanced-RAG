"""Schema and consistency checks for the frozen eval corpus and golden cases."""

from __future__ import annotations

from evals.loader import load_cases, load_corpus, retrieval_cases


def test_corpus_has_unique_section_keys():
    docs = load_corpus()
    keys = [(d["metadata"]["source_id"], d["metadata"]["section_id"]) for d in docs]
    assert len(docs) >= 20
    assert len(keys) == len(set(keys))


def test_cases_validate_against_corpus():
    cases = load_cases()
    ids = [case["id"] for case in cases]
    assert len(cases) >= 50
    assert len(ids) == len(set(ids))


def test_retrieval_slice_has_facts_and_qrels():
    cases = retrieval_cases(load_cases())
    assert len(cases) >= 40
    for case in cases:
        assert case["qrels"]
        assert case["must_cite_facts"]
        assert case["reference_answer"]


def test_unanswerable_cases_have_no_qrels():
    unanswerable = [case for case in load_cases() if case["unanswerable"]]
    assert len(unanswerable) >= 5
    assert all(not case["qrels"] for case in unanswerable)


def test_clarification_cases_declare_a_slot():
    clarify = [case for case in load_cases() if case["expected_clarification"]]
    assert len(clarify) >= 3
    assert all(case.get("expected_slot") for case in clarify)
