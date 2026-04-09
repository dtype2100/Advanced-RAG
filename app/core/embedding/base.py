"""Abstract base class for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """Unified interface for text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return their vector representations."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string. Default delegates to ``embed_texts``."""
        return self.embed_texts([text])[0]

    @abstractmethod
    def dimension(self) -> int:
        """Return the output vector dimensionality."""
