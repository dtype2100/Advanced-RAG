"""Retrieval quality evaluator — qrels-based ranking metrics.

Scores retrieved chunks against graded relevance labels (qrels), not against
the retriever's own similarity scores.  Matching is done on stable
``source_id`` + ``section_id`` metadata so labels survive re-chunking.
"""

from __future__ import annotations

import math
from typing import Any

Qrel = dict[str, Any]
Result = dict[str, Any]


def _section_key(meta: dict[str, Any]) -> tuple[str, str]:
    """Return the stable (source_id, section_id) identity of a chunk or qrel."""
    return (str(meta.get("source_id", "")), str(meta.get("section_id", "")))


def _result_key(result: Result) -> tuple[str, str]:
    """Extract the section key from a retrieval result dict."""
    meta = result.get("metadata") or {}
    return _section_key(meta)


def _qrel_map(qrels: list[Qrel], min_relevance: int) -> dict[tuple[str, str], int]:
    """Build a section-key → relevance map, keeping the highest label per key."""
    mapping: dict[tuple[str, str], int] = {}
    for qrel in qrels:
        key = _section_key(qrel)
        rel = int(qrel.get("relevance", 0))
        if rel >= min_relevance:
            mapping[key] = max(mapping.get(key, 0), rel)
    return mapping


def _dcg(gains: list[int]) -> float:
    """Discounted cumulative gain for a list of relevance grades."""
    return sum((2**gain - 1) / math.log2(rank + 2) for rank, gain in enumerate(gains))


def evaluate_retrieval(
    results: list[Result],
    qrels: list[Qrel] | None = None,
    k: int = 5,
    min_relevance: int = 1,
    *,
    query: str | None = None,
    relevance_threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate a ranked retrieval list against graded qrels.

    Metrics:
        hit_at_k:      1.0 if any relevant section appears in the top-k.
        recall_at_k:   Unique relevant sections retrieved / all relevant sections.
        precision_at_k: Relevant items in top-k / k.
        mrr:           Reciprocal rank of the first relevant section.
        ndcg_at_k:     nDCG with graded gains (relevance 1 or 2).

    Duplicate chunks that share the same section key are collapsed to the
    first rank so a split section cannot inflate recall.

    The ``query`` and ``relevance_threshold`` keyword-only arguments are
    accepted for backward compatibility with older callers and ignored.

    Args:
        results:          Ranked ``{text, score, metadata}`` dicts.
        qrels:            Graded labels ``{source_id, section_id, relevance}``.
        k:                Cutoff (defaults to ``MAX_RETRIEVAL_DOCS`` = 5).
        min_relevance:    Minimum grade counted as relevant (default 1).
        query:            Unused; kept so old positional call sites do not break
                          if they switch to keywords.
        relevance_threshold: Unused legacy score cutoff.

    Returns:
        Dict of ranking metrics.  When ``qrels`` is empty, recall/nDCG are 0.0
        and ``skipped`` is True so unanswerable items can be excluded upstream.
    """
    del query, relevance_threshold  # legacy kwargs, not used for qrels scoring

    ranked = results[:k]
    seen: set[tuple[str, str]] = set()
    unique_ranked: list[Result] = []
    for result in ranked:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        unique_ranked.append(result)

    labels = _qrel_map(qrels or [], min_relevance=min_relevance)
    if not labels:
        return {
            "k": k,
            "total": len(unique_ranked),
            "hit_at_k": 0.0,
            "recall_at_k": 0.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "relevant_retrieved": 0,
            "relevant_total": 0,
            "skipped": True,
        }

    gains = [labels.get(_result_key(result), 0) for result in unique_ranked]
    relevant_retrieved = sum(1 for gain in gains if gain >= min_relevance)
    first_rank = next((i + 1 for i, gain in enumerate(gains) if gain >= min_relevance), None)

    ideal = sorted(labels.values(), reverse=True)[:k]
    dcg = _dcg(gains)
    idcg = _dcg(ideal)
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {
        "k": k,
        "total": len(unique_ranked),
        "hit_at_k": 1.0 if first_rank is not None else 0.0,
        "recall_at_k": relevant_retrieved / len(labels),
        "precision_at_k": relevant_retrieved / k if k else 0.0,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "ndcg_at_k": round(ndcg, 4),
        "relevant_retrieved": relevant_retrieved,
        "relevant_total": len(labels),
        "skipped": False,
    }
