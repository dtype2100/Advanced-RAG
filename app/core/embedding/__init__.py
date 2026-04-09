"""Embedding providers – switchable via EMBEDDING_PROVIDER env var."""

from app.core.embedding.base import BaseEmbedding
from app.core.embedding.factory import get_embedding

__all__ = ["BaseEmbedding", "get_embedding"]
