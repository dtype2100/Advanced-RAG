"""Abstract base class for vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Unified interface for vector databases."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the target collection / index if it does not exist."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Upsert documents with pre-computed vectors. Returns count of upserted points."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the top-k nearest documents as ``{text, score, metadata}`` dicts."""

    @abstractmethod
    def get_client(self) -> Any:
        """Return the underlying DB client for health-checks and diagnostics."""
