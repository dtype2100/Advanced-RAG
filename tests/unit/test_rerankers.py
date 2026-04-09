"""Unit tests for reranker utilities (no model required)."""

from __future__ import annotations

from app.rag.guards.relevance_guard import filter_relevant


def test_filter_relevant_keeps_high_score():
    results = [
        {"text": "relevant", "score": 0.8},
        {"text": "borderline", "score": 0.3},
        {"text": "irrelevant", "score": 0.1},
    ]
    filtered = filter_relevant(results, threshold=0.3)
    assert len(filtered) == 2
    assert all(r["score"] >= 0.3 for r in filtered)


def test_filter_relevant_empty_input():
    assert filter_relevant([]) == []
