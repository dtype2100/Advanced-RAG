"""FlashRank reranker – lightweight, CPU-friendly cross-encoder reranking."""

from __future__ import annotations

import logging

from app.config import settings
from app.core.reranker.base import BaseReranker, RerankResult

logger = logging.getLogger(__name__)


class FlashRankReranker(BaseReranker):
    """Wraps flashrank.Ranker for fast CPU-based reranking."""

    def __init__(self) -> None:
        from flashrank import Ranker

        self._ranker = Ranker(model_name=settings.reranker_model)
        logger.info("FlashRank reranker loaded: model=%s", settings.reranker_model)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[RerankResult]:
        from flashrank import RerankRequest

        passages = [{"id": i, "text": doc} for i, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = self._ranker.rerank(request)

        return [
            RerankResult(text=r["text"], score=r["score"], index=r["id"]) for r in results[:top_k]
        ]
