"""ARQ job functions — must be importable by the worker process."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def ingest_documents_job(
    ctx: dict[str, Any],
    documents: list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict[str, Any]:
    """Run the ingest pipeline in a thread pool (blocking FastEmbed/Qdrant).

    Args:
        ctx:        ARQ context dict (unused).
        documents:  List of ``{text, metadata}`` dicts.
        chunk_size: Recursive chunker size.
        chunk_overlap: Chunk overlap.

    Returns:
        Dict with ``count`` (chunks indexed) and ``message``.
    """
    from app.rag.pipelines.ingest_pipeline import run_ingest

    def _run() -> int:
        return run_ingest(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    count = await asyncio.to_thread(_run)
    logger.info("ingest_documents_job: indexed %d chunks from %d docs", count, len(documents))
    return {"count": count, "message": "Documents ingested successfully"}
