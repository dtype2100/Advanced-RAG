"""Ingest pipeline — orchestrates document loading, pre-processing, chunking, and indexing."""

from __future__ import annotations

import logging
from typing import Any

from app.providers.vectorstore_provider import get_vectorstore
from app.rag.preprocess.cleaner import clean_text
from app.rag.preprocess.deduplicator import dedup_documents
from app.rag.preprocess.metadata_extractor import extract_metadata

logger = logging.getLogger(__name__)


def run_ingest(
    raw_docs: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> int:
    """Run the full ingestion pipeline.

    Steps:
    1. Clean text.
    2. Extract metadata.
    3. Deduplicate.
    4. Chunk with recursive splitter.
    5. Upsert to vector store.

    Args:
        raw_docs:      List of ``{text, metadata}`` dicts.
        chunk_size:    Characters per chunk.
        chunk_overlap: Chunk overlap in characters.

    Returns:
        Total number of chunks upserted.
    """
    from app.rag.chunkers.recursive_chunker import recursive_chunk

    cleaned = [{"text": clean_text(d["text"]), "metadata": d.get("metadata", {})} for d in raw_docs]
    enriched = [extract_metadata(d) for d in cleaned]
    unique = dedup_documents(enriched)

    chunks = recursive_chunk(unique, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning("No chunks produced from %d docs", len(raw_docs))
        return 0

    store = get_vectorstore()
    count = store.add_documents(
        texts=[c["text"] for c in chunks],
        metadatas=[c.get("metadata", {}) for c in chunks],
    )
    logger.info("Ingest pipeline: %d chunks indexed", count)
    return count
