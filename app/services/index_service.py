"""Index service — manage vector store collection lifecycle."""

from __future__ import annotations

import logging

from app.providers.vectorstore_provider import get_vectorstore

logger = logging.getLogger(__name__)


def ensure_index() -> None:
    """Ensure the vector store collection exists.  Called at application startup."""
    store = get_vectorstore()
    store.ensure_collection()
    logger.info("Index service: collection ready")


def rebuild_index(documents: list[dict]) -> int:
    """Drop and rebuild the vector store index from scratch.

    Args:
        documents: Full document list to re-index.

    Returns:
        Number of chunks indexed after rebuild.
    """
    from app.rag.pipelines.ingest_pipeline import run_ingest

    store = get_vectorstore()
    store.delete_collection()
    store.ensure_collection()
    logger.info("Index service: collection rebuilt")
    return run_ingest(documents)
