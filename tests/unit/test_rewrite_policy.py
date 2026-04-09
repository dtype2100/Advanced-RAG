"""Unit tests for the rewrite policy."""

from __future__ import annotations

from app.rag.policies.rewrite_policy import needs_rewrite


def test_short_query_triggers_rewrite():
    assert needs_rewrite("AI") is True


def test_long_query_no_rewrite():
    assert needs_rewrite("What are the main benefits of retrieval augmented generation?") is False


def test_empty_retrieval_result_triggers_rewrite():
    state = {"retrieval_attempt": 1, "retrieved_children": []}
    assert needs_rewrite("some long enough query here", state) is True


def test_successful_previous_retrieval_no_rewrite():
    state = {"retrieval_attempt": 1, "retrieved_children": ["doc1", "doc2"]}
    assert needs_rewrite("some long enough query here", state) is False
