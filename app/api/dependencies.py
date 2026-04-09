"""FastAPI dependency injection assembly.

Provides reusable ``Depends()`` callables for common resources such as the
vector store, LLM, and optional API-key auth guard.
"""

from __future__ import annotations

from app.providers.vectorstore_provider import get_vectorstore
from app.storage.vectorstores.base import VectorStorePort


def get_store() -> VectorStorePort:
    """Dependency: return the active vector store instance."""
    return get_vectorstore()
