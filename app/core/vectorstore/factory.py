"""Vector store factory – returns a singleton based on ``VECTORSTORE_PROVIDER``."""

from __future__ import annotations

from app.core.vectorstore.base import BaseVectorStore

_instance: BaseVectorStore | None = None


def get_vectorstore() -> BaseVectorStore:
    """Return a cached vector store instance configured via settings."""
    global _instance
    if _instance is not None:
        return _instance

    from app.config import settings
    from app.core.embedding import get_embedding

    embedding = get_embedding()
    dim = embedding.dimension()

    if settings.vectorstore_provider == "chroma":
        from app.core.vectorstore.chroma import ChromaVectorStore

        _instance = ChromaVectorStore(
            collection_name=settings.collection_name,
            vector_size=dim,
            host=settings.chroma_host,
            port=settings.chroma_port,
            persist_dir=settings.chroma_persist_dir,
        )
    else:
        from app.core.vectorstore.qdrant import QdrantVectorStore

        _instance = QdrantVectorStore(
            collection_name=settings.collection_name,
            vector_size=dim,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    return _instance


def reset() -> None:
    """Clear the singleton – only used in tests."""
    global _instance
    _instance = None
