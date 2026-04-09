"""Abstract interface (port) for vector store implementations.

All concrete vector store adapters must subclass ``VectorStorePort`` so
that the rest of the application can swap between Qdrant, pgvector, etc.
without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStorePort(ABC):
    """Protocol-style base class for vector stores.

    Implementors:
        - ``QdrantStore``  (default)
        - ``PgVectorStore`` (future drop-in replacement)
    """

    @abstractmethod
    def ensure_collection(self) -> None:
        """Create the target collection / index if it does not already exist."""

    @abstractmethod
    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        """Embed and upsert documents.

        Args:
            texts: Raw document strings.
            metadatas: Optional per-document metadata dicts.

        Returns:
            Number of successfully upserted documents.
        """

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Perform a semantic similarity search.

        Args:
            query: Query string to embed and search with.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys ``text``, ``score``, ``metadata``.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """Drop the collection / index (used in tests and migrations)."""
