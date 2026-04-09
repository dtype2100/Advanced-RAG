"""Vector store provider abstraction.

Switch providers by setting ``VECTORSTORE_PROVIDER`` env var:
  - ``qdrant`` (default) – Qdrant vector database
  - ``chroma`` – ChromaDB
"""

from app.core.vectorstore.base import BaseVectorStore
from app.core.vectorstore.factory import get_vectorstore

__all__ = ["BaseVectorStore", "get_vectorstore"]
