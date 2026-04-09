"""Vector store provider — singleton accessor for the active ``VectorStorePort``.

Delegates instantiation to ``storage.vectorstores.factory`` so that the
selection logic lives in one place.

Change the ``VECTOR_BACKEND`` environment variable to swap implementations:
- ``qdrant``   → ``QdrantStore``  (default)
- ``pgvector`` → ``PgVectorStore`` (stub, implement when ready)
"""

from __future__ import annotations

import logging

from app.storage.vectorstores.base import VectorStorePort

logger = logging.getLogger(__name__)

_store: VectorStorePort | None = None


def get_vectorstore() -> VectorStorePort:
    """Return the singleton vector store for the configured backend.

    Delegates to ``storage.vectorstores.factory.create_vectorstore()``.
    """
    global _store
    if _store is not None:
        return _store

    from app.storage.vectorstores.factory import create_vectorstore

    _store = create_vectorstore()
    return _store
