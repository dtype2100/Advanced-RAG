"""Schemas for Studio (runtime + read-only config) API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuntimeOverrides(BaseModel):
    """Nullable fields: null clears override and restores env/settings default."""

    multi_query: bool | None = None
    max_retrieval_docs: int | None = Field(default=None, ge=1, le=100)
    grounding_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_retries: int | None = Field(default=None, ge=0, le=20)
    rerank_top_k: int | None = Field(default=None, ge=1, le=50)


class RuntimeStateResponse(BaseModel):
    """Overrides (null = default) and effective values used by the graph."""

    overrides: dict
    effective: dict


class ReadOnlyConfigResponse(BaseModel):
    """Env-backed settings (restart / redeploy to change)."""

    llm_backend: str
    llm_model: str
    llm_temperature: float
    vllm_base_url: str
    openai_api_key_set: bool
    embedding_model: str
    qdrant_url: str
    qdrant_in_memory: bool
    collection_name: str
    redis_url_set: bool
    ingest_queue_async: bool
    arq_queue_name: str
    reranker_backend: str
    multi_query_env: str
    tei_embedding_url: str
    tei_rerank_url: str
    log_llm_io: bool
