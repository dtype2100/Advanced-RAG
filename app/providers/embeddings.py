"""Embedding backends (selected via EMBEDDING_BACKEND / EMBEDDING_MODEL)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.config import settings


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embeds text batches for indexing and search."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider:
    """FastEmbed (ONNX) bi-encoder embeddings."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._embedder: Any = None

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(model_name=self._model_name)
        return self._embedder

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embedder = self._get_embedder()
        return [vec.tolist() for vec in embedder.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


_embed_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the configured embedding provider singleton."""
    global _embed_provider
    if _embed_provider is not None:
        return _embed_provider

    backend = settings.embedding_backend.lower()
    if backend == "fastembed":
        _embed_provider = FastEmbedProvider(settings.embedding_model)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_BACKEND={settings.embedding_backend!r}. Supported: fastembed"
        )
    return _embed_provider


def reset_embedding_provider_for_tests() -> None:
    """Clear singleton (tests only)."""
    global _embed_provider
    _embed_provider = None
