"""pgvector-backed vector store — future drop-in replacement for Qdrant.

This module is a **stub** that documents the intended interface.
Implement when migrating to PostgreSQL + pgvector.
"""

from __future__ import annotations

from typing import Any

from app.storage.vectorstores.base import VectorStorePort


class PgVectorStore(VectorStorePort):
    """pgvector implementation of ``VectorStorePort`` (not yet implemented).

    Swap this in via ``vectorstore_provider.py`` when ready.
    """

    def ensure_collection(self) -> None:
        raise NotImplementedError("PgVectorStore is not yet implemented")

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        raise NotImplementedError("PgVectorStore is not yet implemented")

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise NotImplementedError("PgVectorStore is not yet implemented")

    def delete_collection(self) -> None:
        raise NotImplementedError("PgVectorStore is not yet implemented")
