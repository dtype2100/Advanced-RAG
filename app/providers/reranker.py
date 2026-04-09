"""Reranker provider abstraction.

Supported providers (selected via RERANKER_PROVIDER env var):
- ``none``          : No reranking (default, pass-through)
- ``cross-encoder`` : Local cross-encoder model via sentence-transformers
                      (requires RERANKER_MODEL, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2)
- ``cohere``        : Cohere Rerank API (requires COHERE_API_KEY)

Usage::

    from app.providers.reranker import rerank

    ranked = rerank(query="What is Python?", documents=["doc1", "doc2"], top_k=3)
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

_reranker: Any = None


def _build_reranker() -> Any | None:
    """Instantiate the reranker model based on RERANKER_PROVIDER.

    Returns:
        Reranker instance, or None if provider is 'none'.
    """
    provider = settings.reranker_provider.lower()

    if provider == "none":
        return None

    if provider == "cross-encoder":
        logger.info("Loading cross-encoder reranker: %s", settings.reranker_model)
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for RERANKER_PROVIDER=cross-encoder. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        return CrossEncoder(settings.reranker_model)

    if provider == "cohere":
        logger.info("Using Cohere reranker, model=%s", settings.cohere_rerank_model)
        try:
            import cohere  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "cohere is required for RERANKER_PROVIDER=cohere. "
                "Install it with: pip install cohere"
            ) from exc

        return cohere.Client(api_key=settings.cohere_api_key)

    raise ValueError(
        f"Unsupported RERANKER_PROVIDER='{provider}'. Supported: 'none', 'cross-encoder', 'cohere'."
    )


def get_reranker() -> Any | None:
    """Return the singleton reranker, initialising it on first call.

    Returns:
        Reranker instance or None when provider is 'none'.
    """
    global _reranker
    if not settings.using_reranker:
        return None
    if _reranker is None:
        _reranker = _build_reranker()
    return _reranker


def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[str]:
    """Rerank *documents* by relevance to *query* and return top-k results.

    When RERANKER_PROVIDER is 'none' the original order is preserved.

    Args:
        query:     The user query string.
        documents: List of candidate document texts.
        top_k:     Maximum number of results to return (None = all).

    Returns:
        Reranked list of document texts (best first).
    """
    if not documents:
        return documents

    k = top_k or len(documents)
    provider = settings.reranker_provider.lower()

    if provider == "none":
        return documents[:k]

    reranker = get_reranker()

    if provider == "cross-encoder":
        pairs = [(query, doc) for doc in documents]
        scores: list[float] = reranker.predict(pairs).tolist()
        ranked = sorted(zip(scores, documents, strict=True), key=lambda x: x[0], reverse=True)
        logger.info("CrossEncoder reranked %d docs", len(ranked))
        return [doc for _, doc in ranked[:k]]

    if provider == "cohere":
        response = reranker.rerank(
            query=query,
            documents=documents,
            model=settings.cohere_rerank_model,
            top_n=k,
        )
        result = [documents[r.index] for r in response.results]
        logger.info("Cohere reranked %d -> %d docs", len(documents), len(result))
        return result

    return documents[:k]
