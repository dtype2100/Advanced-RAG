"""Application settings loaded from environment variables / .env file.

Provider selection via environment variables:
- EMBEDDING_PROVIDER: "fastembed" (default) | "openai"
- LLM_BACKEND:        "vllm" (default) | "openai" | "anthropic"
- RERANKER_PROVIDER:  "none" (default) | "cross-encoder" | "cohere"
- VECTORSTORE_BACKEND:"qdrant" (default) | "chroma"
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    # Provider: "vllm" | "openai" | "anthropic"
    llm_backend: str = "vllm"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_base_url: str = ""  # optional custom base URL (e.g. Azure OpenAI)

    # ── Anthropic ─────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── vLLM ──────────────────────────────────────────────────────────────────
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # ── Embedding ─────────────────────────────────────────────────────────────
    # Provider: "fastembed" | "openai"
    embedding_provider: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # OpenAI embedding model (used when embedding_provider=openai)
    openai_embedding_model: str = "text-embedding-3-small"

    # ── Reranker ──────────────────────────────────────────────────────────────
    # Provider: "none" | "cross-encoder" | "cohere"
    reranker_provider: str = "none"
    # cross-encoder model (used when reranker_provider=cross-encoder)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Cohere API key (used when reranker_provider=cohere)
    cohere_api_key: str = ""
    # Cohere rerank model
    cohere_rerank_model: str = "rerank-english-v3.0"

    # ── Vector Store ──────────────────────────────────────────────────────────
    # Backend: "qdrant" | "chroma"
    vectorstore_backend: str = "qdrant"
    collection_name: str = "advanced_rag"

    # Qdrant – empty url triggers in-memory mode
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Chroma – empty host triggers in-memory/local mode
    chroma_host: str = ""
    chroma_port: int = 8002
    chroma_persist_dir: str = "./chroma_db"

    # ── RAG ───────────────────────────────────────────────────────────────────
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # ── Server ────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def qdrant_in_memory(self) -> bool:
        """True when Qdrant should run in-memory (no URL configured)."""
        return not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        """True when the vLLM backend is selected."""
        return self.llm_backend == "vllm"

    @property
    def using_openai_llm(self) -> bool:
        """True when the OpenAI LLM backend is selected."""
        return self.llm_backend == "openai"

    @property
    def using_anthropic(self) -> bool:
        """True when the Anthropic LLM backend is selected."""
        return self.llm_backend == "anthropic"

    @property
    def using_reranker(self) -> bool:
        """True when any reranker is configured."""
        return self.reranker_provider != "none"


settings = Settings()
