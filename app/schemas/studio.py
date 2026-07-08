"""Schemas for the Studio configuration API."""

from __future__ import annotations

from pydantic import BaseModel


class ReadOnlyConfigResponse(BaseModel):
    """Read-only deployment configuration snapshot."""

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
    multi_query: bool
    use_parent_child_chunking: bool
    grounding_threshold: float
    max_retrieval_docs: int
    max_retries: int


class ServiceProbeResponse(BaseModel):
    """Result of optional dependency reachability probes."""

    llm: str
    qdrant: dict[str, object]
