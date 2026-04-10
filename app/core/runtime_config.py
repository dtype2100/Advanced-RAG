"""Process-local runtime overrides for RAG / graph behaviour (Studio UI).

Values set here override env-backed defaults until cleared (set back to ``None``).
Thread-safe for concurrent API requests.
"""

from __future__ import annotations

import os
from threading import Lock
from typing import Any

from app.core.config import settings

_lock = Lock()
_state: dict[str, Any] = {
    "multi_query": None,
    "max_retrieval_docs": None,
    "grounding_threshold": None,
    "max_retries": None,
    "rerank_top_k": None,
}


def get_multi_query_enabled() -> bool:
    """Return whether multi-query retrieval is on (runtime override or ``MULTI_QUERY`` env)."""
    with _lock:
        v = _state["multi_query"]
    if isinstance(v, bool):
        return v
    return os.getenv("MULTI_QUERY", "0") == "1"


def set_multi_query_enabled(value: bool | None) -> None:
    """Set multi-query; ``None`` clears override (use env again)."""
    with _lock:
        _state["multi_query"] = value


def get_max_retrieval_docs() -> int:
    with _lock:
        v = _state["max_retrieval_docs"]
    if isinstance(v, int) and v > 0:
        return v
    return settings.max_retrieval_docs


def set_max_retrieval_docs(value: int | None) -> None:
    with _lock:
        _state["max_retrieval_docs"] = value


def get_grounding_threshold() -> float:
    with _lock:
        v = _state["grounding_threshold"]
    if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
        return float(v)
    return 0.6


def set_grounding_threshold(value: float | None) -> None:
    with _lock:
        _state["grounding_threshold"] = value


def get_max_retries() -> int:
    with _lock:
        v = _state["max_retries"]
    if isinstance(v, int) and v >= 0:
        return v
    return settings.max_retries


def set_max_retries(value: int | None) -> None:
    with _lock:
        _state["max_retries"] = value


def get_rerank_top_k() -> int | None:
    """If set, cap reranked chunk count."""
    with _lock:
        v = _state["rerank_top_k"]
    if v is None:
        return None
    if isinstance(v, int) and v > 0:
        return v
    return None


def set_rerank_top_k(value: int | None) -> None:
    with _lock:
        _state["rerank_top_k"] = value


def snapshot() -> dict[str, Any]:
    """Return current override values (``None`` means fall back to env/settings)."""
    with _lock:
        return dict(_state)


def effective_dict() -> dict[str, Any]:
    """Resolved values as used by the pipeline."""
    return {
        "multi_query": get_multi_query_enabled(),
        "max_retrieval_docs": get_max_retrieval_docs(),
        "grounding_threshold": get_grounding_threshold(),
        "max_retries": get_max_retries(),
        "rerank_top_k": get_rerank_top_k(),
    }
