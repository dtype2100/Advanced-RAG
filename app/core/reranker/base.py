"""Abstract base class for all reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RerankResult:
    """A single reranked document with its new relevance score."""

    text: str
    score: float
    index: int


class BaseReranker(ABC):
    """Unified interface that every reranker provider must implement."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[RerankResult]:
        """Re-score and re-order *documents* by relevance to *query*.

        Returns up to *top_k* results sorted by descending score.
        """
