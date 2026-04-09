"""ChromaDB vectorstore provider."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.config import settings
from app.core.embedding import get_embedding
from app.core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Wraps chromadb client with automatic embedding via the configured provider."""

    def __init__(self) -> None:
        import chromadb

        if settings.chroma_persist_dir:
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
            logger.info("ChromaDB persistent client: %s", settings.chroma_persist_dir)
        elif settings.chroma_host:
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
            logger.info("ChromaDB HTTP client: %s:%d", settings.chroma_host, settings.chroma_port)
        else:
            self._client = chromadb.EphemeralClient()
            logger.info("ChromaDB ephemeral (in-memory) client")

        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ensure_collection(self) -> None:
        self._get_collection()

    def add_documents(
        self,
        texts: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not texts:
            return 0

        embedder = get_embedding()
        vectors = embedder.embed_texts(texts)
        metadatas = metadatas or [{} for _ in texts]
        ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]

        collection = self._get_collection()
        collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        logger.info("Upserted %d documents into '%s'", len(texts), settings.collection_name)
        return len(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        embedder = get_embedding()
        query_vec = embedder.embed_query(query)

        collection = self._get_collection()
        results = collection.query(query_embeddings=[query_vec], n_results=top_k)

        out: list[dict[str, Any]] = []
        if results and results.get("documents"):
            docs = results["documents"][0]
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            metas = results["metadatas"][0] if results.get("metadatas") else [{} for _ in docs]
            for doc, dist, meta in zip(docs, distances, metas, strict=True):
                out.append(
                    {
                        "text": doc or "",
                        "score": 1.0 - dist,
                        "metadata": meta or {},
                    }
                )
        return out

    def get_raw_client(self) -> Any:
        return self._client
