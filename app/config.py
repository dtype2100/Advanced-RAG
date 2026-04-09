"""Application settings loaded from environment variables / .env file.

All provider backends (LLM, Embedding, Reranker, VectorStore) are
configurable via environment variables so that switching between
providers requires zero code changes.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────
    llm_provider: Literal["openai", "vllm", "anthropic"] = "vllm"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # OpenAI / vLLM (OpenAI-compatible) credentials
    openai_api_key: str = ""
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # Anthropic credentials
    anthropic_api_key: str = ""

    # ── Embedding ────────────────────────────────────────────────────────
    embedding_provider: Literal["fastembed", "openai", "huggingface"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # ── Reranker ─────────────────────────────────────────────────────────
    reranker_provider: Literal["none", "flashrank", "crossencoder"] = "none"
    reranker_model: str = "ms-marco-MultiBERT-L-12"
    reranker_top_k: int = 5

    # ── VectorStore ──────────────────────────────────────────────────────
    vectorstore_provider: Literal["qdrant", "chroma"] = "qdrant"
    collection_name: str = "advanced_rag"

    # Qdrant-specific
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Chroma-specific
    chroma_host: str = "localhost"
    chroma_port: int = 8200
    chroma_persist_dir: str = ""

    # ── RAG ──────────────────────────────────────────────────────────────
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def qdrant_in_memory(self) -> bool:
        return self.vectorstore_provider == "qdrant" and not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        return self.llm_provider == "vllm"

    @property
    def reranker_enabled(self) -> bool:
        return self.reranker_provider != "none"


settings = Settings()
