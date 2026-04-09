"""Backward-compatible facade: delegates to the active vector store backend."""

from __future__ import annotations

from typing import Any

from app.vectorstore.factory import get_vector_store


def get_client() -> Any:
    """Qdrant client when using VECTOR_BACKEND=qdrant; raises otherwise."""
    store = get_vector_store()
    getter = getattr(store, "get_client", None)
    if callable(getter):
        return getter()
    raise RuntimeError("get_client() requires VECTOR_BACKEND=qdrant")


def ensure_collection() -> None:
    get_vector_store().ensure_collection()


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    return get_vector_store().add_documents(texts, metadatas)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    return get_vector_store().search(query, top_k)
