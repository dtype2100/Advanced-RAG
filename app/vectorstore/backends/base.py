"""Shared vector store backend contract."""

from __future__ import annotations

from typing import Any, Protocol


class VectorStoreBackend(Protocol):
    """Contract for vector store backends."""

    def ensure_collection(self) -> None:
        """Create collection if needed."""

    def add_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
        """Embed and upsert documents."""

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Run vector similarity search."""

    def healthcheck(self) -> dict[str, str]:
        """Return backend health status and collection state."""
