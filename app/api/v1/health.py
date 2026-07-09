"""Health / readiness check endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from app.core.config import settings
from app.core.health_checks import check_llm, check_redis, check_vectorstore, is_ready
from app.schemas.response import HealthResponse, LivenessResponse, ReadinessResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Return service health including dependency status (backward compatible)."""
    vector = await check_vectorstore()
    redis = await check_redis()
    llm = await check_llm()

    ready = is_ready({"vectorstore": vector, "redis": redis, "llm": llm})
    return HealthResponse(
        status="ok" if ready else "degraded",
        version="0.2.0",
        llm_backend=settings.llm_backend,
        llm_model=settings.llm_model,
        qdrant=vector["status"],
        collection=vector["collection"],
        redis=redis["status"],
        llm=llm["status"],
        auth_enabled=settings.auth_enabled,
    )


@router.get("/health/live", response_model=LivenessResponse, tags=["system"])
async def liveness() -> LivenessResponse:
    """Kubernetes liveness probe — process is running."""
    return LivenessResponse(status="alive")


@router.get("/health/ready", response_model=ReadinessResponse, tags=["system"])
async def readiness(response: Response) -> ReadinessResponse:
    """Kubernetes readiness probe — dependencies are available for traffic."""
    vector = await check_vectorstore()
    redis = await check_redis()
    llm = await check_llm()
    checks = {"vectorstore": vector, "redis": redis, "llm": llm}
    ready = is_ready(checks)

    if not ready:
        response.status_code = 503

    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        checks={
            "vectorstore": vector,
            "redis": redis,
            "llm": llm,
        },
    )
