"""Application settings loaded from environment variables / .env file."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised configuration for the Advanced RAG service.

    All values are overridable via environment variables or a .env file.
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    llm_backend: str = "vllm"  # "openai" | "vllm"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # ── vLLM ─────────────────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # ── Embedding ────────────────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    collection_name: str = "advanced_rag"

    # ── RAG pipeline ─────────────────────────────────────────────────────────
    max_retrieval_docs: int = 5
    max_retries: int = 3  # hallucination feedback loop 최대 횟수

    # ── Server ───────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Queue (ARQ + Redis) ───────────────────────────────────────────────────
    redis_url: str = ""
    arq_queue_name: str = "arq:queue"
    ingest_queue_async: bool = False

    # ── External inference (optional; TEI / custom endpoints) ─────────────────
    tei_embedding_url: str = ""
    tei_rerank_url: str = ""

    # ── Observability ──────────────────────────────────────────────────────────
    # When True, log full prompts and LLM outputs under logger ``app.llm_io`` (PII risk).
    log_llm_io: bool = Field(default=False, validation_alias="LOG_LLM_IO")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def qdrant_in_memory(self) -> bool:
        """True when no external Qdrant URL is configured (uses :memory: mode)."""
        return not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        """True when the active LLM backend is vLLM."""
        return self.llm_backend == "vllm"


settings = Settings()
