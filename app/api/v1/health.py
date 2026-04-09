"""Health / readiness check endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.core.config import settings
from app.providers.vectorstore_provider import get_vectorstore
from app.schemas.response import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Return service health including vector store connection status."""
    try:
        store = get_vectorstore()
        # QdrantStore exposes get_client(); fall back gracefully for other adapters.
        if hasattr(store, "get_client"):
            client = store.get_client()
            collections = [c.name for c in client.get_collections().collections]
            qdrant_status = "connected"
            collection_status = (
                "exists" if settings.collection_name in collections else "not_created"
            )
        else:
            qdrant_status = "connected"
            collection_status = "unknown"
    except Exception as exc:
        qdrant_status = f"error: {exc}"
        collection_status = "unknown"

    return HealthResponse(
        status="ok",
        llm_backend=settings.llm_backend,
        llm_model=settings.llm_model,
        qdrant=qdrant_status,
        collection=collection_status,
    )
