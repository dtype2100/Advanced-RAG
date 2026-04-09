"""Backward-compatibility shim for app.vectorstore.store.

All new code should use ``app.providers.vectorstore_provider.get_vectorstore()``
or service-layer functions in ``app.services.*``.
"""

from __future__ import annotations

from typing import Any

from app.storage.vectorstores.qdrant_store import QdrantStore

_store = QdrantStore()


def get_client():
    """Return the singleton Qdrant client (backward compat)."""
    return _store.get_client()


def ensure_collection() -> None:
    """Ensure the Qdrant collection exists (backward compat)."""
    _store.ensure_collection()


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    """Embed and upsert documents (backward compat)."""
    return _store.add_documents(texts, metadatas)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Semantic search (backward compat)."""
    from app.core.config import settings

    return _store.search(query, top_k=top_k or settings.max_retrieval_docs)
