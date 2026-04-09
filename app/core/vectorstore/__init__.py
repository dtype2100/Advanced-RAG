"""VectorStore providers – switchable via VECTORSTORE_PROVIDER env var."""

from app.core.vectorstore.base import BaseVectorStore
from app.core.vectorstore.factory import get_vectorstore

__all__ = ["BaseVectorStore", "get_vectorstore"]
