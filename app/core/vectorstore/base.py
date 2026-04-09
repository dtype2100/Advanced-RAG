"""Abstract base class for all vectorstore providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseVectorStore(ABC):
    """Unified interface that every vectorstore provider must implement."""

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the backing collection/index if it does not exist."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed and upsert *texts*. Returns the number of upserted items."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic search. Returns list of ``{text, score, metadata}``."""

    @abstractmethod
    def get_raw_client(self) -> Any:
        """Expose the underlying client for health-checks and admin ops."""
