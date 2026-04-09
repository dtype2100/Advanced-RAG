"""BM25 sparse retriever for keyword-based retrieval.

Requires: ``pip install rank_bm25``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BM25Retriever:
    """Thin wrapper around ``rank_bm25.BM25Okapi`` for document retrieval.

    Usage:
        retriever = BM25Retriever(corpus_docs)
        results = retriever.search("my query", top_k=5)
    """

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError("Install rank_bm25: pip install rank_bm25") from exc

        self._docs = docs
        tokenised = [d["text"].lower().split() for d in docs]
        self._bm25 = BM25Okapi(tokenised)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Return top-k documents ranked by BM25 score.

        Args:
            query: Query string.
            top_k: Maximum number of results.

        Returns:
            List of ``{text, score, metadata}`` dicts.
        """
        from rank_bm25 import BM25Okapi  # noqa: F401 — needed for type narrowing

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in ranked[:top_k]:
            doc = dict(self._docs[idx])
            doc["score"] = float(score)
            results.append(doc)
        return results
