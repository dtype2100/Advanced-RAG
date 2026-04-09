"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.providers.vectorstore import ensure_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise resources on startup, clean up on shutdown."""
    logger.info("Starting Advanced RAG service")
    logger.info("VectorStore backend : %s", settings.vectorstore_backend)
    logger.info(
        "Embedding provider  : %s / %s", settings.embedding_provider, settings.embedding_model
    )
    logger.info("LLM backend         : %s / %s", settings.llm_backend, settings.llm_model)
    logger.info("Reranker provider   : %s", settings.reranker_provider)
    if settings.using_reranker:
        if settings.reranker_provider == "cross-encoder":
            logger.info("Reranker model      : %s", settings.reranker_model)
        elif settings.reranker_provider == "cohere":
            logger.info("Cohere rerank model : %s", settings.cohere_rerank_model)
    if settings.vectorstore_backend == "qdrant":
        qdrant_mode = "in-memory" if settings.qdrant_in_memory else settings.qdrant_url
        logger.info("Qdrant mode         : %s", qdrant_mode)

    ensure_collection()
    logger.info("Collection '%s' ready", settings.collection_name)

    yield

    logger.info("Shutting down Advanced RAG service")


app = FastAPI(
    title="Advanced RAG API",
    description="Self-corrective RAG system powered by FastAPI, LangGraph, and Qdrant",
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
