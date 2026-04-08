from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

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

    # Qdrant – empty url triggers in-memory mode
    qdrant_url: str = ""
    qdrant_api_key: str = ""

    # Embedding (FastEmbed model name)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    collection_name: str = "advanced_rag"

    # RAG
    max_retrieval_docs: int = 5
    max_retries: int = 3

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def qdrant_in_memory(self) -> bool:
        return not self.qdrant_url

    @property
    def using_vllm(self) -> bool:
        return self.llm_backend == "vllm"


settings = Settings()
