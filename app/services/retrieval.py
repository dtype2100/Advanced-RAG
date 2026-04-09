"""Retrieval service – orchestrates embedding, vector search, and reranking.

This module is the single entry-point used by the RAG pipeline and the
``/search`` + ``/documents`` API routes. It delegates to whichever
concrete providers are configured via environment variables.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings
from app.core.embedding import get_embedding
from app.core.reranker import get_reranker
from app.core.vectorstore import get_vectorstore

logger = logging.getLogger(__name__)


def ensure_collection() -> None:
    """Initialise the vector store collection (called at startup)."""
    store = get_vectorstore()
    store.ensure_collection()


def add_documents(
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
) -> int:
    """Embed texts, then upsert into the configured vector store."""
    if not texts:
        return 0

    embedding = get_embedding()
    vectors = embedding.embed_texts(texts)

    store = get_vectorstore()
    return store.add_documents(texts, vectors, metadatas)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Embed query → vector search → optional reranking."""
    k = top_k or settings.max_retrieval_docs

    embedding = get_embedding()
    query_vector = embedding.embed_query(query)

    store = get_vectorstore()
    results = store.search(query_vector, top_k=k)

    reranker = get_reranker()
    reranker_top_k = settings.reranker_top_k if settings.reranker_provider != "none" else None
    results = reranker.rerank(query, results, top_k=reranker_top_k)

    return results
