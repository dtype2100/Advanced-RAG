"""Vector store provider abstraction.

Supported backends (selected via VECTORSTORE_BACKEND env var):
- ``qdrant``  : Qdrant (default, in-memory or remote via QDRANT_URL)
- ``chroma``  : ChromaDB (in-memory, local persistent, or remote via CHROMA_HOST)

The module exposes four public functions used by the rest of the application:
    - ``get_client()``      → raw backend client (for health checks)
    - ``ensure_collection()``
    - ``add_documents()``
    - ``search()``
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.config import settings
from app.providers.embeddings import embed_query, embed_texts, get_embedding_dim

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _deterministic_id(text: str) -> str:
    """Generate a deterministic UUID v5 from text content for deduplication."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


# ── Qdrant backend ───────────────────────────────────────────────────────────

_qdrant_client: Any = None


def _get_qdrant_client() -> Any:
    """Return singleton Qdrant client (in-memory or remote)."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client

    from qdrant_client import QdrantClient  # type: ignore[import-untyped]

    if settings.qdrant_in_memory:
        logger.info("Initialising Qdrant in-memory client")
        _qdrant_client = QdrantClient(":memory:")
    else:
        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        _qdrant_client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
        )
    return _qdrant_client


def _qdrant_ensure_collection() -> None:
    from qdrant_client.models import Distance, VectorParams  # type: ignore[import-untyped]

    client = _get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.collection_name not in collections:
        dim = get_embedding_dim()
        client.create_collection(
            collection_name=settings.collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", settings.collection_name, dim)


def _qdrant_add_documents(texts: list[str], metadatas: list[dict[str, Any]]) -> int:
    from qdrant_client.models import PointStruct  # type: ignore[import-untyped]

    _qdrant_ensure_collection()
    client = _get_qdrant_client()
    vectors = embed_texts(texts)
    points = [
        PointStruct(
            id=_deterministic_id(text),
            vector=vec,
            payload={"text": text, **meta},
        )
        for text, vec, meta in zip(texts, vectors, metadatas, strict=True)
    ]
    client.upsert(collection_name=settings.collection_name, points=points)
    logger.info("Qdrant: upserted %d documents into '%s'", len(points), settings.collection_name)
    return len(points)


def _qdrant_search(query: str, top_k: int) -> list[dict[str, Any]]:
    _qdrant_ensure_collection()
    client = _get_qdrant_client()
    query_vec = embed_query(query)
    response = client.query_points(
        collection_name=settings.collection_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
    )
    return [
        {
            "text": hit.payload.get("text", "") if hit.payload else "",
            "score": hit.score,
            "metadata": {k: v for k, v in (hit.payload or {}).items() if k != "text"},
        }
        for hit in response.points
    ]


# ── Chroma backend ───────────────────────────────────────────────────────────

_chroma_client: Any = None
_chroma_collection: Any = None


def _get_chroma_client() -> Any:
    """Return singleton ChromaDB client."""
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    try:
        import chromadb  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for VECTORSTORE_BACKEND=chroma. "
            "Install it with: pip install chromadb"
        ) from exc

    if settings.chroma_host:
        logger.info("Connecting to Chroma at %s:%d", settings.chroma_host, settings.chroma_port)
        _chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
    elif settings.chroma_persist_dir:
        logger.info("Using Chroma with persistence dir: %s", settings.chroma_persist_dir)
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    else:
        logger.info("Initialising Chroma in-memory client")
        _chroma_client = chromadb.EphemeralClient()

    return _chroma_client


def _get_chroma_collection() -> Any:
    """Return (or create) the Chroma collection singleton."""
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    client = _get_chroma_client()
    _chroma_collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Chroma collection '%s' ready", settings.collection_name)
    return _chroma_collection


def _chroma_ensure_collection() -> None:
    _get_chroma_collection()


def _chroma_add_documents(texts: list[str], metadatas: list[dict[str, Any]]) -> int:
    collection = _get_chroma_collection()
    vectors = embed_texts(texts)
    ids = [_deterministic_id(t) for t in texts]
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )
    logger.info("Chroma: upserted %d documents into '%s'", len(texts), settings.collection_name)
    return len(texts)


def _chroma_search(query: str, top_k: int) -> list[dict[str, Any]]:
    collection = _get_chroma_collection()
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    # Chroma returns L2 or cosine distance (lower = more similar).
    # Convert distance to a 0-1 similarity score: score = 1 - distance/2 (for cosine).
    return [
        {
            "text": doc,
            "score": max(0.0, 1.0 - dist / 2.0),
            "metadata": meta or {},
        }
        for doc, meta, dist in zip(docs, metas, distances, strict=True)
    ]


# ── Public API ────────────────────────────────────────────────────────────────


def get_client() -> Any:
    """Return the raw backend client (used by health-check endpoints).

    Returns:
        QdrantClient or chromadb.Client instance.
    """
    backend = settings.vectorstore_backend.lower()
    if backend == "qdrant":
        return _get_qdrant_client()
    if backend == "chroma":
        return _get_chroma_client()
    raise ValueError(f"Unsupported VECTORSTORE_BACKEND='{backend}'.")


def ensure_collection() -> None:
    """Create the vector store collection if it does not exist."""
    backend = settings.vectorstore_backend.lower()
    if backend == "qdrant":
        _qdrant_ensure_collection()
    elif backend == "chroma":
        _chroma_ensure_collection()
    else:
        raise ValueError(f"Unsupported VECTORSTORE_BACKEND='{backend}'.")


def add_documents(texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> int:
    """Embed and upsert documents into the configured vector store.

    Args:
        texts:     List of document strings.
        metadatas: Optional list of metadata dicts (one per document).

    Returns:
        Number of documents upserted.
    """
    if not texts:
        return 0
    metas = metadatas or [{} for _ in texts]
    backend = settings.vectorstore_backend.lower()
    if backend == "qdrant":
        return _qdrant_add_documents(texts, metas)
    if backend == "chroma":
        return _chroma_add_documents(texts, metas)
    raise ValueError(f"Unsupported VECTORSTORE_BACKEND='{backend}'.")


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """Semantic search against the configured vector store.

    Args:
        query:  Query string.
        top_k:  Maximum number of results (defaults to settings.max_retrieval_docs).

    Returns:
        List of dicts with keys ``text``, ``score``, ``metadata``.
    """
    k = top_k or settings.max_retrieval_docs
    backend = settings.vectorstore_backend.lower()
    if backend == "qdrant":
        return _qdrant_search(query, k)
    if backend == "chroma":
        return _chroma_search(query, k)
    raise ValueError(f"Unsupported VECTORSTORE_BACKEND='{backend}'.")
