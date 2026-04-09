"""Unit tests for retriever utilities."""

from __future__ import annotations

from app.rag.retrievers.hybrid_retriever import reciprocal_rank_fusion


def test_rrf_merges_two_lists():
    list_a = [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.7}]
    list_b = [{"text": "doc2", "score": 0.8}, {"text": "doc3", "score": 0.6}]
    merged = reciprocal_rank_fusion([list_a, list_b])
    texts = [d["text"] for d in merged]
    assert "doc2" in texts
    assert len(set(texts)) == len(texts), "No duplicate texts in merged results"


def test_rrf_single_list():
    list_a = [{"text": "a", "score": 1.0}, {"text": "b", "score": 0.5}]
    merged = reciprocal_rank_fusion([list_a])
    assert len(merged) == 2
    assert merged[0]["text"] == "a"


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([]) == []
