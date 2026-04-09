"""Cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.reranker.base import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Wraps sentence_transformers.CrossEncoder for semantic reranking."""

    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(settings.reranker_model)
        logger.info("CrossEncoder reranker loaded: model=%s", settings.reranker_model)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[RerankResult]:
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs).tolist()

        scored = [
            RerankResult(text=doc, score=score, index=idx)
            for idx, (doc, score) in enumerate(zip(documents, scores, strict=True))
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
