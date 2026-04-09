"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.vectorstore.store import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Advanced RAG service")
    if settings.vector_db_backend == "qdrant":
        qdrant_mode = "in-memory" if settings.qdrant_in_memory else settings.qdrant_url
        logger.info("Qdrant mode: %s", qdrant_mode)
    logger.info("Embedding backend: %s", settings.embedding_backend)
    logger.info("Embedding model: %s", settings.embedding_model)
    logger.info("Reranker backend: %s", settings.reranker_backend)
    logger.info("Reranker model: %s", settings.effective_reranker_model)
    logger.info("Vector DB backend: %s", settings.vector_db_backend)
    logger.info("LLM backend: %s", settings.llm_backend)
    logger.info("LLM model: %s", settings.llm_model)
    if settings.using_vllm:
        logger.info("vLLM endpoint: %s", settings.vllm_base_url)

    ensure_collection()
    logger.info("Collection '%s' ready", settings.vector_db_collection_name)

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
