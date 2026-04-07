from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # OpenAI
    openai_api_key: str = ""

    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

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


settings = Settings()
