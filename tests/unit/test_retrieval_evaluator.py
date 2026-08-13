"""Unit tests for qrels-based retrieval metrics."""

from __future__ import annotations

from app.rag.evaluators.retrieval_evaluator import evaluate_retrieval


def _hit(source: str, section: str, score: float = 0.9) -> dict:
    """Build a retrieval hit with stable section metadata."""
    return {
        "text": f"{source}/{section}",
        "score": score,
        "metadata": {"source_id": source, "section_id": section},
    }


def test_perfect_top_hit_has_unit_metrics():
    results = [_hit("rag_overview", "definition"), _hit("qdrant", "storage")]
    qrels = [{"source_id": "rag_overview", "section_id": "definition", "relevance": 2}]
    metrics = evaluate_retrieval(results, qrels, k=5)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_k"] == 1.0
    assert metrics["skipped"] is False


def test_relevant_at_rank_two_sets_mrr_half():
    results = [_hit("other", "x"), _hit("rag_overview", "definition")]
    qrels = [{"source_id": "rag_overview", "section_id": "definition", "relevance": 2}]
    metrics = evaluate_retrieval(results, qrels, k=5)
    assert metrics["hit_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_k"] < 1.0


def test_duplicate_section_keys_do_not_inflate_recall():
    results = [
        _hit("rag_overview", "definition"),
        _hit("rag_overview", "definition"),
        _hit("qdrant", "storage"),
    ]
    qrels = [
        {"source_id": "rag_overview", "section_id": "definition", "relevance": 2},
        {"source_id": "qdrant", "section_id": "storage", "relevance": 1},
    ]
    metrics = evaluate_retrieval(results, qrels, k=5)
    assert metrics["relevant_retrieved"] == 2
    assert metrics["recall_at_k"] == 1.0


def test_empty_qrels_are_skipped():
    metrics = evaluate_retrieval([_hit("x", "y")], qrels=[], k=5)
    assert metrics["skipped"] is True
    assert metrics["recall_at_k"] == 0.0


def test_miss_has_zero_hit():
    results = [_hit("other", "x")]
    qrels = [{"source_id": "rag_overview", "section_id": "definition", "relevance": 2}]
    metrics = evaluate_retrieval(results, qrels, k=5)
    assert metrics["hit_at_k"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["recall_at_k"] == 0.0
