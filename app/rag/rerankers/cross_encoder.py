"""Cross-encoder reranker using a sentence-transformers cross-encoder model.

Requires: ``pip install sentence-transformers``
Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranker backed by a HuggingFace cross-encoder model.

    Args:
        model_name: HuggingFace model identifier.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(model_name)
        except ImportError as exc:
            raise ImportError(
                "Install sentence-transformers: pip install sentence-transformers"
            ) from exc
        logger.info("CrossEncoderReranker loaded: %s", model_name)

    def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank ``docs`` relative to ``query`` and return sorted results.

        Args:
            query:  Query string.
            docs:   List of ``{text, ...}`` dicts to rerank.
            top_k:  Return only the top-k after reranking (``None`` = all).

        Returns:
            Re-sorted list of result dicts with updated ``score`` field.
        """
        if not docs:
            return docs

        pairs = [(query, d["text"]) for d in docs]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(docs, scores, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )
        results = [{**doc, "score": float(score)} for doc, score in ranked]
        return results[:top_k] if top_k else results
