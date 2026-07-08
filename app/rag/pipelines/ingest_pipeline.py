"""Ingest pipeline — orchestrates document loading, pre-processing, chunking, and indexing."""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.providers.vectorstore_provider import get_vectorstore
from app.rag.preprocess.cleaner import clean_text
from app.rag.preprocess.deduplicator import dedup_documents
from app.rag.preprocess.metadata_extractor import extract_metadata
from app.rag.retrievers.corpus_registry import register_docstore, register_documents

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
    4. Chunk (parent-child or recursive).
    5. Upsert to vector store and sync BM25 corpus.

    Args:
        raw_docs:      List of ``{text, metadata}`` dicts.
        chunk_size:    Characters per chunk (recursive mode).
        chunk_overlap: Chunk overlap in characters.

    Returns:
        Total number of chunks upserted.
    """
    cleaned = [{"text": clean_text(d["text"]), "metadata": d.get("metadata", {})} for d in raw_docs]
    enriched = [extract_metadata(d) for d in cleaned]
    unique = dedup_documents(enriched)

    texts = [d["text"] for d in unique]
    chunks: list[dict[str, Any]] = []

    if settings.use_parent_child_chunking:
        from app.rag.chunkers.parent_child_chunker import parent_child_chunk

        parents, children = parent_child_chunk(
            texts,
            parent_chunk_size=chunk_size,
            child_chunk_size=max(128, chunk_size // 2),
            chunk_overlap=chunk_overlap,
        )
        docstore = {p["metadata"]["chunk_id"]: p["text"] for p in parents}
        register_docstore(docstore)
        for child in children:
            parent_id = child["metadata"].get("parent_id")
            if parent_id and parent_id in docstore:
                child["metadata"]["parent_text"] = docstore[parent_id]
        chunks = children
    else:
        from app.rag.chunkers.recursive_chunker import recursive_chunk

        chunks = recursive_chunk(texts, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if not chunks:
        logger.warning("No chunks produced from %d docs", len(raw_docs))
        return 0

    store = get_vectorstore()
    count = store.add_documents(
        texts=[c["text"] for c in chunks],
        metadatas=[c.get("metadata", {}) for c in chunks],
    )
    register_documents(chunks)
    logger.info("Ingest pipeline: %d chunks indexed", count)
    return count
