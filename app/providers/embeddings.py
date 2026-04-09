"""Embedding provider abstraction.

Supported providers (selected via EMBEDDING_PROVIDER env var):
- ``fastembed``  : Local embedding via FastEmbed (default, no API key needed)
- ``openai``     : OpenAI embedding API (requires OPENAI_API_KEY)

Usage::

    from app.providers.embeddings import embed_texts, embed_query
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# ── Dimension lookup for known models ────────────────────────────────────────
# Used to size Qdrant / Chroma collections correctly at creation time.
EMBEDDING_DIM_MAP: dict[str, int] = {
    # FastEmbed models
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


def get_embedding_dim() -> int:
    """Return the vector dimension for the currently configured embedding model."""
    if settings.embedding_provider == "openai":
        return EMBEDDING_DIM_MAP.get(settings.openai_embedding_model, 1536)
    return EMBEDDING_DIM_MAP.get(settings.embedding_model, 384)


# ── Singleton embedder ────────────────────────────────────────────────────────

_embedder: Any = None


def _build_embedder() -> Any:
    """Instantiate the embedder based on EMBEDDING_PROVIDER."""
    provider = settings.embedding_provider.lower()

    if provider == "fastembed":
        logger.info("Loading FastEmbed model: %s", settings.embedding_model)
        from fastembed import TextEmbedding  # type: ignore[import-untyped]

        return TextEmbedding(model_name=settings.embedding_model)

    if provider == "openai":
        logger.info("Using OpenAI embedding model: %s", settings.openai_embedding_model)
        from openai import OpenAI  # type: ignore[import-untyped]

        client = OpenAI(api_key=settings.openai_api_key)

        class _OpenAIEmbedder:
            """Thin wrapper to unify the embed() interface with FastEmbed."""

            def embed(self, texts: list[str]) -> list[list[float]]:
                response = client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=texts,
                )
                return [item.embedding for item in response.data]

        return _OpenAIEmbedder()

    raise ValueError(
        f"Unsupported EMBEDDING_PROVIDER='{provider}'. Supported: 'fastembed', 'openai'."
    )


def get_embedder() -> Any:
    """Return the singleton embedder, initialising it on first call."""
    global _embedder
    if _embedder is None:
        _embedder = _build_embedder()
    return _embedder


# ── Public API ────────────────────────────────────────────────────────────────


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using the configured provider.

    Args:
        texts: List of strings to embed.

    Returns:
        List of float vectors, one per input text.
    """
    embedder = get_embedder()
    result = embedder.embed(texts)
    # FastEmbed returns numpy arrays; normalise to plain Python lists
    return [v.tolist() if hasattr(v, "tolist") else v for v in result]


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    Args:
        text: Query string.

    Returns:
        Float vector.
    """
    return embed_texts([text])[0]
