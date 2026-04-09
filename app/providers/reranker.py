"""Reranker backends (selected via RERANKER_BACKEND / RERANKER_MODEL)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.core.config import settings


@runtime_checkable
class RerankerProvider(Protocol):
    """Scores query–document pairs for second-stage ranking."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]: ...


class NoOpReranker:
    """Passthrough scores (caller should use vector similarity only)."""

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        return [1.0] * len(documents)


class FastEmbedCrossEncoderReranker:
    """FastEmbed ONNX cross-encoder reranking."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._encoder: Any = None

    def _get_encoder(self) -> Any:
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._encoder = TextCrossEncoder(model_name=self._model_name)
        return self._encoder

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        encoder = self._get_encoder()
        return list(encoder.rerank(query, documents))


_rerank_provider: RerankerProvider | None = None


def get_reranker_provider() -> RerankerProvider:
    """Return the configured reranker singleton."""
    global _rerank_provider
    if _rerank_provider is not None:
        return _rerank_provider

    backend = settings.reranker_backend.lower()
    if backend in ("", "none", "off", "disabled"):
        _rerank_provider = NoOpReranker()
    elif backend == "fastembed":
        _rerank_provider = FastEmbedCrossEncoderReranker(settings.reranker_model)
    else:
        raise ValueError(
            f"Unsupported RERANKER_BACKEND={settings.reranker_backend!r}. "
            "Supported: none, fastembed"
        )
    return _rerank_provider


def reset_reranker_provider_for_tests() -> None:
    global _rerank_provider
    _rerank_provider = None
