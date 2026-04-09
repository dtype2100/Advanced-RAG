from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI (used when llm_backend=openai)
    openai_api_key: str = ""

    # LLM backend: "openai" or "vllm"
    llm_backend: str = "vllm"

    # LLM model name (OpenAI model or vLLM model name)
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # vLLM server settings
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # Vector store backend: "qdrant" (in-memory :memory: if QDRANT_URL empty) or "memory"
    vector_backend: str = "qdrant"

    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Embedding provider (extensible; only "fastembed" implemented)
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # If > 0, overrides automatic dimension lookup for the embedding model
    embedding_vector_size: int = 0

    collection_name: str = "advanced_rag"

    # Reranker: "none" or "fastembed" (cross-encoder)
    reranker_backend: str = "none"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # First-stage retrieval size before reranking (only used when reranker is enabled)
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    # RAG
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    @property
    def qdrant_in_memory(self) -> bool:
        return self.vector_backend == "qdrant" and not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        return self.llm_backend == "vllm"

    @property
    def reranker_enabled(self) -> bool:
        return self.reranker_backend.lower() not in ("", "none", "off", "disabled")


settings = Settings()
