"""Hybrid retriever — fuses dense vector results with BM25 sparse results.

Uses Reciprocal Rank Fusion (RRF) to combine ranked lists from both retrievers.
RRF is backend-agnostic and avoids the need to calibrate score scales.
"""

from __future__ import annotations

from typing import Any


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a ranked list of result dicts with a
                      ``text`` key used as the merge key.
        k:            RRF constant (higher k reduces the impact of top ranks).

    Returns:
        Fused, re-ranked list of unique result dicts.
    """
    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            key = doc["text"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    merged = sorted(doc_map.values(), key=lambda d: rrf_scores[d["text"]], reverse=True)
    for doc in merged:
        doc["score"] = rrf_scores[doc["text"]]
    return merged


def hybrid_retrieve(
    query: str,
    corpus_docs: list[dict[str, Any]] | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Combine dense and sparse retrieval via RRF.

    Args:
        query:       Search query.
        corpus_docs: Optional in-memory corpus for BM25.  When ``None``, only
                     vector retrieval is performed.
        top_k:       Number of final fused results to return.

    Returns:
        Fused top-k result dicts.
    """
    from app.rag.retrievers.vector_retriever import vector_retrieve

    vector_results = vector_retrieve(query, top_k=top_k * 2)
    lists_to_fuse: list[list[dict[str, Any]]] = [vector_results]

    if corpus_docs:
        from app.rag.retrievers.bm25_retriever import BM25Retriever

        bm25 = BM25Retriever(corpus_docs)
        bm25_results = bm25.search(query, top_k=top_k * 2)
        lists_to_fuse.append(bm25_results)

    fused = reciprocal_rank_fusion(lists_to_fuse)
    return fused[:top_k]
