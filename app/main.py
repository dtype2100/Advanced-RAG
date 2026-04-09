"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.services.retrieval import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Advanced RAG service")
    logger.info("LLM backend: %s (%s)", settings.llm_backend, settings.llm_model)
    logger.info(
        "Embedding provider: %s (%s)", settings.embedding_provider, settings.embedding_model
    )
    logger.info("Vector store: %s", settings.vectorstore_provider)
    logger.info("Reranker: %s", settings.reranker_provider)

    if settings.using_vllm:
        logger.info("vLLM endpoint: %s", settings.vllm_base_url)

    ensure_collection()
    logger.info("Collection '%s' ready", settings.collection_name)

    yield

    logger.info("Shutting down Advanced RAG service")


app = FastAPI(
    title="Advanced RAG API",
    description="Self-corrective RAG system powered by FastAPI, LangGraph, and Qdrant",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Advanced RAG API",
        "version": "0.1.0",
        "docs": "/docs",
    }
