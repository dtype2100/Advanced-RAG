"""Studio API — read-only deployment config and service probes."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter

from app.core.config import settings
from app.core.llm_health import probe_llm
from app.schemas.studio import ReadOnlyConfigResponse, ServiceProbeResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/studio", tags=["studio"])


@router.get("/config", response_model=ReadOnlyConfigResponse)
async def get_readonly_config() -> ReadOnlyConfigResponse:
    """Return env-backed settings (change via .env and restart)."""
    return ReadOnlyConfigResponse(
        llm_backend=settings.llm_backend,
        llm_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        vllm_base_url=settings.vllm_base_url,
        openai_api_key_set=bool(settings.openai_api_key),
        embedding_model=settings.embedding_model,
        qdrant_url=settings.qdrant_url or "(empty = in-memory)",
        qdrant_in_memory=settings.qdrant_in_memory,
        collection_name=settings.collection_name,
        redis_url_set=bool(settings.redis_url),
        ingest_queue_async=settings.ingest_queue_async,
        arq_queue_name=settings.arq_queue_name,
        reranker_backend=settings.reranker_backend,
        multi_query=settings.multi_query,
        use_parent_child_chunking=settings.use_parent_child_chunking,
        grounding_threshold=settings.grounding_threshold,
        max_retrieval_docs=settings.max_retrieval_docs,
        max_retries=settings.max_retries,
    )


@router.post("/probe", response_model=ServiceProbeResponse)
async def probe_services() -> ServiceProbeResponse:
    """Lightweight reachability checks for LLM and Qdrant."""
    llm_status = probe_llm()
    qdrant: dict[str, object] = {"note": "in-memory mode in this process"}

    async with httpx.AsyncClient(timeout=5.0) as client:
        if settings.qdrant_url:
            try:
                response = await client.get(f"{settings.qdrant_url.rstrip('/')}/healthz")
                qdrant = {"ok": response.status_code < 400, "status": response.status_code}
            except Exception as exc:
                qdrant = {"ok": False, "error": str(exc)}

    return ServiceProbeResponse(llm=llm_status, qdrant=qdrant)
