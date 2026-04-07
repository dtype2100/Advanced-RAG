"""Qdrant vector store wrapper using FastEmbed for local embedding."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.config import settings

logger = logging.getLogger(__name__)

_VECTOR_SIZE_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return singleton Qdrant client (in-memory or remote)."""
    global _client
    if _client is not None:
        return _client

    if settings.qdrant_in_memory:
        logger.info("Initializing Qdrant in-memory client")
        _client = QdrantClient(":memory:")
    else:
        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _client


def _embedding_fn() -> Any:
    """Lazy-load FastEmbed model."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embedding_model)


_embedder: Any = None


def get_embedder() -> Any:
    global _embedder
    if _embedder is None:
        _embedder = _embedding_fn()
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using FastEmbed."""
    embedder = get_embedder()
    return [vec.tolist() for vec in embedder.embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed a single query text."""
    return embed_texts([text])[0]


def ensure_collection() -> None:
    """Create the collection if it does not exist."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in collections:
        vec_size = _VECTOR_SIZE_MAP.get(settings.embedding_model, 384)
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
        )
        logger.info("Created collection '%s' (dim=%d)", settings.collection_name, vec_size)


def _deterministic_id(text: str) -> str:
    """Generate a deterministic UUID v5 from text content for dedup."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    """Embed and upsert documents. Returns count of upserted points."""
    if not texts:
        return 0

    ensure_collection()
    client = get_client()
    vectors = embed_texts(texts)
    metadatas = metadatas or [{} for _ in texts]

    points = [
        PointStruct(
            id=_deterministic_id(text),
            vector=vec,
            payload={"text": text, **meta},
        )
        for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
    ]
    client.upsert(collection_name=settings.collection_name, points=points)
    logger.info("Upserted %d documents into '%s'", len(points), settings.collection_name)
    return len(points)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Semantic search via query_points. Returns list of {text, score, metadata}."""
    ensure_collection()
    client = get_client()
    k = top_k or settings.max_retrieval_docs
    query_vec = embed_query(query)

    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vec,
        limit=k,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", "") if hit.payload else "",
            "score": hit.score,
            "metadata": {
                key: val for key, val in (hit.payload or {}).items() if key != "text"
            },
        }
        for hit in response.points
    ]
