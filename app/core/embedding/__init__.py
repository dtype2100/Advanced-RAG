"""Embedding provider abstraction.

Switch providers by setting ``EMBEDDING_PROVIDER`` env var:
  - ``fastembed`` (default) – local FastEmbed models
  - ``openai`` – OpenAI embedding API
"""

from app.core.embedding.base import BaseEmbedding
from app.core.embedding.factory import get_embedding

__all__ = ["BaseEmbedding", "get_embedding"]
