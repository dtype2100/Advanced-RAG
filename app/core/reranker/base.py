"""Abstract base class for reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseReranker(ABC):
    """Unified interface for document rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank documents by relevance to ``query``.

        Args:
            query: The user query.
            documents: List of dicts with at least a ``"text"`` key.
            top_k: Maximum number of results to return.

        Returns:
            Reranked (and possibly truncated) list of document dicts.
        """
