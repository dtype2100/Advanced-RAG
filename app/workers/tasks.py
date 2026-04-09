"""Background task definitions.

Placeholder for Celery / ARQ / FastAPI BackgroundTasks integration.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def async_ingest_task(documents: list[dict]) -> None:
    """Background ingestion task (stub).

    Args:
        documents: List of ``{text, metadata}`` dicts to ingest.
    """
    from app.services.ingest_service import ingest_documents

    logger.info("Background task: ingesting %d documents", len(documents))
    ingest_documents(documents)
