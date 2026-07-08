"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.ingest import router as ingest_router
from app.api.v1.jobs import router as jobs_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.metrics import MetricsMiddleware, metrics_response
from app.core.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    SecurityHeadersMiddleware,
    get_cors_origins,
)
from app.queue.pool import close_arq_pool
from app.services.index_service import ensure_index

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup and clean up on shutdown."""
    logger.info("Starting Advanced RAG service")
    logger.info(
        "Qdrant mode: %s",
        "in-memory" if settings.qdrant_in_memory else settings.qdrant_url,
    )
    logger.info("Embedding model: %s", settings.embedding_model)
    logger.info("LLM backend: %s  model: %s", settings.llm_backend, settings.llm_model)
    logger.info("API auth: %s", "enabled" if settings.auth_enabled else "disabled")
    if settings.using_vllm:
        logger.info("vLLM endpoint: %s", settings.vllm_base_url)

    ensure_index()
    logger.info("Vector index ready (collection: %s)", settings.collection_name)

    yield

    await close_arq_pool()
    logger.info("Shutting down Advanced RAG service")


app = FastAPI(
    title="Advanced RAG API",
    description=(
        "Agentic RAG system: conditional clarification, conditional rewrite, "
        "parent/child + small-to-big retrieval, hallucination feedback loop (max 3×). "
        "Powered by FastAPI, LangGraph, and Qdrant."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

register_exception_handlers(app)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, limit_per_minute=settings.rate_limit_per_minute)
if settings.enable_metrics:
    app.add_middleware(MetricsMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/api/v1")
app.include_router(ingest_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint — basic service info."""
    return {
        "service": "Advanced RAG API",
        "version": "0.2.0",
        "docs": "/docs",
    }


@app.get("/metrics", tags=["system"], include_in_schema=settings.enable_metrics)
async def metrics():
    """Prometheus metrics exposition endpoint."""
    if not settings.enable_metrics:
        return Response(status_code=404, content="Metrics disabled")
    return metrics_response()
