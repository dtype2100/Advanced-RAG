"""Ingest service — handles document ingestion requests."""

from __future__ import annotations

import logging
from typing import Any

from app.rag.pipelines.ingest_pipeline import run_ingest

logger = logging.getLogger(__name__)


def ingest_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """Ingest a list of documents through the full pre-processing pipeline.

    Args:
        documents:    List of ``{text, metadata}`` dicts.
        chunk_size:   Characters per chunk.
        chunk_overlap: Overlap between adjacent chunks.

    Returns:
        Number of chunks indexed in the vector store.
    """
    logger.info("Ingest service: processing %d documents", len(documents))
    return run_ingest(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
