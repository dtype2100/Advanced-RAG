"""FlashRank cross-encoder reranker – lightweight, runs locally on CPU."""

from __future__ import annotations

import logging
from typing import Any

from app.core.reranker.base import BaseReranker

logger = logging.getLogger(__name__)


class FlashRankReranker(BaseReranker):
    """Reranks via ``flashrank.Ranker`` (CPU-only, ~30 MB model)."""

    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2") -> None:
        self._model_name = model_name
        self._ranker: Any = None

    def _get_ranker(self) -> Any:
        if self._ranker is None:
            from flashrank import Ranker

            logger.info("Loading FlashRank model: %s", self._model_name)
            self._ranker = Ranker(model_name=self._model_name)
        return self._ranker

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        from flashrank import RerankRequest

        ranker = self._get_ranker()
        passages = [{"id": idx, "text": doc["text"]} for idx, doc in enumerate(documents)]
        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)

        ranked_indices = [int(r["id"]) for r in results]
        reranked = [documents[i] for i in ranked_indices]

        if top_k is not None:
            reranked = reranked[:top_k]
        return reranked
