"""Abstract base class for all embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Unified interface that every embedding provider must implement."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Defaults to batch-of-one."""
        return self.embed_texts([text])[0]

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimension for this model."""
