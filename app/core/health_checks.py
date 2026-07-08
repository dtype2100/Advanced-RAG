"""Health check helpers for liveness and readiness probes."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.providers.vectorstore_provider import get_vectorstore

logger = logging.getLogger(__name__)


async def check_vectorstore() -> dict[str, str]:
    """Verify vector store connectivity and collection status."""
    try:
        store = get_vectorstore()
        if hasattr(store, "get_client"):
            client = store.get_client()
            collections = [c.name for c in client.get_collections().collections]
            collection_status = (
                "exists" if settings.collection_name in collections else "not_created"
            )
            return {"status": "connected", "collection": collection_status}
        return {"status": "connected", "collection": "unknown"}
    except Exception as exc:
        logger.warning("Vector store health check failed: %s", exc)
        return {"status": f"error: {exc}", "collection": "unknown"}


async def check_redis() -> dict[str, str]:
    """Ping Redis when configured for async ingest."""
    if not settings.redis_url:
        return {"status": "not_configured"}

    try:
        from app.queue.pool import get_arq_pool

        pool = await get_arq_pool()
        await pool.ping()
        return {"status": "connected"}
    except Exception as exc:
        logger.warning("Redis health check failed: %s", exc)
        return {"status": f"error: {exc}"}


async def check_llm() -> dict[str, str]:
    """Optionally verify LLM backend reachability."""
    if not settings.health_check_llm:
        return {"status": "skipped"}

    try:
        if settings.using_vllm:
            url = f"{settings.vllm_base_url.rstrip('/')}/models"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
            return {"status": "connected"}
        if settings.openai_api_key:
            return {"status": "configured"}
        return {"status": "not_configured"}
    except Exception as exc:
        logger.warning("LLM health check failed: %s", exc)
        return {"status": f"error: {exc}"}


def is_ready(checks: dict[str, Any]) -> bool:
    """Return True when all critical dependencies are healthy."""
    vector = checks.get("vectorstore", {})
    if not str(vector.get("status", "")).startswith("connected"):
        return False

    collection = vector.get("collection")
    if collection == "not_created":
        return False

    redis = checks.get("redis", {})
    if settings.redis_url and not str(redis.get("status", "")).startswith("connected"):
        return False

    llm = checks.get("llm", {})
    llm_status = str(llm.get("status", ""))
    return not (settings.health_check_llm and llm_status.startswith("error"))
