"""No-op reranker – passes documents through unchanged."""

from __future__ import annotations

from typing import Any

from app.core.reranker.base import BaseReranker


class NoOpReranker(BaseReranker):
    """Returns documents as-is, optionally truncated to ``top_k``."""

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if top_k is not None:
            return documents[:top_k]
        return documents
