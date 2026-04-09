"""ChromaDB vector store implementation."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from app.core.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Wraps ``chromadb`` – supports in-memory, persistent, and HTTP modes."""

    def __init__(
        self,
        collection_name: str,
        vector_size: int,
        host: str = "localhost",
        port: int = 8500,
        persist_dir: str = "",
    ) -> None:
        self._collection_name = collection_name
        self._vector_size = vector_size
        self._host = host
        self._port = port
        self._persist_dir = persist_dir
        self._client: Any = None
        self._collection: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import chromadb

            if self._persist_dir:
                logger.info("Initializing Chroma persistent client at %s", self._persist_dir)
                self._client = chromadb.PersistentClient(path=self._persist_dir)
            else:
                logger.info("Initializing Chroma in-memory client")
                self._client = chromadb.Client()
        return self._client

    def get_client(self) -> Any:
        return self._get_client()

    def _get_collection(self) -> Any:
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def ensure_collection(self) -> None:
        self._get_collection()
        logger.info("Collection '%s' ready (Chroma)", self._collection_name)

    @staticmethod
    def _deterministic_id(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def add_documents(
        self,
        texts: list[str],
        vectors: list[list[float]],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> int:
        if not texts:
            return 0

        collection = self._get_collection()
        metadatas = metadatas or [{} for _ in texts]
        ids = [self._deterministic_id(t) for t in texts]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Upserted %d documents into '%s' (Chroma)", len(texts), self._collection_name)
        return len(texts)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict[str, Any]]:
        collection = self._get_collection()

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {
                "text": doc or "",
                "score": 1.0 - dist,
                "metadata": meta or {},
            }
            for doc, meta, dist in zip(documents, metadatas, distances, strict=True)
        ]
