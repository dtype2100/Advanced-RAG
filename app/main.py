"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.core.vectorstore import get_vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Advanced RAG service")
    logger.info("LLM provider: %s  model: %s", settings.llm_provider, settings.llm_model)
    logger.info(
        "Embedding provider: %s  model: %s",
        settings.embedding_provider,
        settings.embedding_model,
    )
    logger.info("Reranker provider: %s", settings.reranker_provider)
    logger.info("VectorStore provider: %s", settings.vectorstore_provider)

    if settings.using_vllm:
        logger.info("vLLM endpoint: %s", settings.vllm_base_url)

    store = get_vectorstore()
    store.ensure_collection()
    logger.info("Collection '%s' ready", settings.collection_name)

    yield

    logger.info("Shutting down Advanced RAG service")


app = FastAPI(
    title="Advanced RAG API",
    description="Self-corrective RAG system powered by FastAPI, LangGraph, and pluggable providers",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "Advanced RAG API",
        "version": "0.2.0",
        "docs": "/docs",
    }
