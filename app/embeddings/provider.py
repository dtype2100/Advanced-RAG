"""Embedding provider abstractions and factory."""

from __future__ import annotations

from typing import Protocol

from app.config import settings


class EmbeddingProvider(Protocol):
    """Contract for embedding implementations."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""

    def vector_size(self) -> int:
        """Return embedding vector dimension."""


class FastEmbedProvider:
    """FastEmbed-backed embedding provider."""

    def __init__(self, model_name: str):
        """Store model configuration and defer model loading."""
        self.model_name = model_name
        self._embedder = None
        self._vector_size: int | None = None

    def _get_embedder(self):
        """Lazy-load FastEmbed model instance."""
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self.model_name)
        return self._embedder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts using FastEmbed."""
        if not texts:
            return []
        embedder = self._get_embedder()
        return [vector.tolist() for vector in embedder.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text query."""
        vectors = self.embed_texts([text])
        return vectors[0] if vectors else []

    def vector_size(self) -> int:
        """Resolve vector dimension from model output with fallback."""
        if self._vector_size is None:
            probe_vector = self.embed_query("vector-size-probe")
            self._vector_size = len(probe_vector) or settings.embedding_size_fallback
        return self._vector_size


_provider_key: tuple[str, str] | None = None
_provider_cache: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Build or reuse embedding provider from current environment configuration."""
    global _provider_key, _provider_cache

    key = (settings.embedding_backend, settings.embedding_model)
    if _provider_cache is not None and _provider_key == key:
        return _provider_cache

    if settings.embedding_backend == "fastembed":
        _provider_cache = FastEmbedProvider(model_name=settings.embedding_model)
    else:
        raise ValueError(f"Unsupported EMBEDDING_BACKEND: {settings.embedding_backend}")

    _provider_key = key
    return _provider_cache


def reset_embedding_provider_cache() -> None:
    """Reset cached provider to pick up runtime configuration changes."""
    global _provider_key, _provider_cache
    _provider_key = None
    _provider_cache = None
