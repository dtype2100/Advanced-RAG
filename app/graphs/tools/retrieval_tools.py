"""LangGraph-compatible retrieval tool definitions."""

from __future__ import annotations

from langchain_core.tools import tool

from app.providers.vectorstore_provider import get_vectorstore


@tool
def vector_search(query: str, top_k: int = 5) -> list[dict]:
    """Search the vector store for documents relevant to ``query``.

    Args:
        query: Query string.
        top_k: Maximum number of results to return.

    Returns:
        List of ``{text, score, metadata}`` dicts.
    """
    store = get_vectorstore()
    return store.search(query, top_k=top_k)
