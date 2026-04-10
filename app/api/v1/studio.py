"""Studio API: runtime overrides and read-only deployment config."""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import APIRouter

from app.core import runtime_config as rc
from app.core.config import settings
from app.schemas.studio import ReadOnlyConfigResponse, RuntimeOverrides, RuntimeStateResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/studio", tags=["studio"])


@router.get("/runtime", response_model=RuntimeStateResponse)
async def get_runtime() -> RuntimeStateResponse:
    """Current process-local overrides and effective RAG parameters."""
    return RuntimeStateResponse(overrides=rc.snapshot(), effective=rc.effective_dict())


@router.patch("/runtime", response_model=RuntimeStateResponse)
async def patch_runtime(body: RuntimeOverrides) -> RuntimeStateResponse:
    """Update runtime overrides (null field = clear override)."""
    data = body.model_dump(exclude_unset=True)
    if "multi_query" in data:
        rc.set_multi_query_enabled(data["multi_query"])
    if "max_retrieval_docs" in data:
        rc.set_max_retrieval_docs(data["max_retrieval_docs"])
    if "grounding_threshold" in data:
        rc.set_grounding_threshold(data["grounding_threshold"])
    if "max_retries" in data:
        rc.set_max_retries(data["max_retries"])
    if "rerank_top_k" in data:
        rc.set_rerank_top_k(data["rerank_top_k"])
    return RuntimeStateResponse(overrides=rc.snapshot(), effective=rc.effective_dict())


@router.get("/config", response_model=ReadOnlyConfigResponse)
async def get_readonly_config() -> ReadOnlyConfigResponse:
    """Env-backed settings (change via .env / container env + restart)."""
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
        reranker_backend=os.getenv("RERANKER_BACKEND", "none"),
        multi_query_env=os.getenv("MULTI_QUERY", "0"),
        tei_embedding_url=settings.tei_embedding_url or "(not set)",
        tei_rerank_url=settings.tei_rerank_url or "(not set)",
        log_llm_io=settings.log_llm_io,
    )


@router.post("/probe")
async def probe_services() -> dict:
    """Lightweight reachability checks for LLM / Qdrant / optional TEI URLs."""
    out: dict[str, dict] = {}

    async with httpx.AsyncClient(timeout=5.0) as client:
        if settings.using_vllm:
            base = settings.vllm_base_url.rstrip("/")
            url = f"{base}/models"
            try:
                r = await client.get(url)
                out["vllm"] = {"ok": r.status_code < 400, "status": r.status_code}
            except Exception as e:
                out["vllm"] = {"ok": False, "error": str(e)}
        else:
            out["vllm"] = {"skipped": True, "reason": "LLM_BACKEND is not vllm"}

        if settings.qdrant_url:
            try:
                r = await client.get(f"{settings.qdrant_url.rstrip('/')}/healthz")
                out["qdrant"] = {"ok": r.status_code < 400, "status": r.status_code}
            except Exception as e:
                out["qdrant"] = {"ok": False, "error": str(e)}
        else:
            out["qdrant"] = {"note": "in-memory mode in this process"}

        if settings.tei_embedding_url:
            u = settings.tei_embedding_url.rstrip("/") + "/health"
            try:
                r = await client.get(u)
                out["tei_embedding"] = {"ok": r.status_code < 400, "status": r.status_code}
            except Exception as e:
                out["tei_embedding"] = {"ok": False, "error": str(e)}
        else:
            out["tei_embedding"] = {"skipped": True}

        if settings.tei_rerank_url:
            u = settings.tei_rerank_url.rstrip("/") + "/health"
            try:
                r = await client.get(u)
                out["tei_rerank"] = {"ok": r.status_code < 400, "status": r.status_code}
            except Exception as e:
                out["tei_rerank"] = {"ok": False, "error": str(e)}
        else:
            out["tei_rerank"] = {"skipped": True}

    return out
