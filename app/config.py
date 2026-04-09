from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # OpenAI (used when llm_backend=openai)
    openai_api_key: str = ""

    # LLM backend and model
    llm_backend: Literal["openai", "vllm"] = "vllm"
    llm_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_temperature: float = 0.0

    # vLLM server settings
    vllm_base_url: str = "http://localhost:8001/v1"
    vllm_model_path: str = "/workspace/models/Qwen2.5-0.5B-Instruct"
    vllm_max_model_len: int = 2048

    # Embedding backend and model
    embedding_backend: Literal["fastembed"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_size_fallback: int = 384

    # Reranker backend and model (empty model falls back to llm_model)
    reranker_backend: Literal["llm", "none"] = "llm"
    reranker_model: str = ""

    # Vector DB backend and collection
    vector_db_backend: Literal["qdrant", "memory"] = "qdrant"
    vector_db_collection_name: str = Field(
        default="advanced_rag",
        validation_alias=AliasChoices("VECTOR_DB_COLLECTION_NAME", "COLLECTION_NAME"),
    )

    # Qdrant settings (used when vector_db_backend=qdrant)
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # RAG controls
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def qdrant_in_memory(self) -> bool:
        """Return True if Qdrant should run in in-memory mode."""
        return not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        """Return True when the selected LLM backend is vLLM."""
        return self.llm_backend == "vllm"

    @property
    def collection_name(self) -> str:
        """Backward-compatible alias for legacy collection_name usage."""
        return self.vector_db_collection_name

    @property
    def effective_reranker_model(self) -> str:
        """Return explicit reranker model, or fallback to LLM model when omitted."""
        return self.reranker_model or self.llm_model


settings = Settings()
