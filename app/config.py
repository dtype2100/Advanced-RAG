"""Application settings loaded from environment variables / .env file.

All provider backends (LLM, embedding, vector store, reranker) can be
switched by setting a single environment variable per component.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Centralised settings – every tunable knob lives here."""

    # ── LLM ──────────────────────────────────────────────────────────────
    llm_backend: Literal["vllm", "openai"] = "vllm"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # vLLM-specific
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # OpenAI-specific
    openai_api_key: str = ""

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_provider: Literal["fastembed", "openai"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Vector Store ─────────────────────────────────────────────────────
    vectorstore_provider: Literal["qdrant", "chroma"] = "qdrant"

    # Qdrant
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Chroma
    chroma_host: str = "localhost"
    chroma_port: int = 8500
    chroma_persist_dir: str = ""

    collection_name: str = "advanced_rag"

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_provider: Literal["none", "flashrank"] = "none"
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"
    reranker_top_k: int = 5

    # ── RAG ──────────────────────────────────────────────────────────────
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Derived helpers ──────────────────────────────────────────────────

    @property
    def qdrant_in_memory(self) -> bool:
        return not self.qdrant_url

    @property
    def chroma_in_memory(self) -> bool:
        return not self.chroma_persist_dir

    @property
    def using_vllm(self) -> bool:
        return self.llm_backend == "vllm"


settings = Settings()
