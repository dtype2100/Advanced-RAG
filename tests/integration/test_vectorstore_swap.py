"""Integration tests for vector store port — verifies the abstraction works."""

from __future__ import annotations

from app.storage.vectorstores.base import VectorStorePort
from app.storage.vectorstores.qdrant_store import QdrantStore


def test_qdrant_store_implements_port():
    store = QdrantStore()
    assert isinstance(store, VectorStorePort)


def test_qdrant_store_add_and_search():
    store = QdrantStore()
    store.ensure_collection()
    texts = ["Machine learning is a field of AI.", "Paris is the capital of France."]
    count = store.add_documents(texts)
    assert count == 2

    results = store.search("artificial intelligence", top_k=2)
    assert len(results) >= 1
    assert all("text" in r and "score" in r for r in results)


def test_qdrant_store_dedup():
    store = QdrantStore()
    text = "Deduplication test for the vectorstore swap test."
    store.add_documents([text])
    store.add_documents([text])

    results = store.search(text, top_k=10)
    matching = [r for r in results if r["text"] == text]
    assert len(matching) == 1


def test_provider_returns_qdrant_by_default(monkeypatch):
    monkeypatch.delenv("VECTOR_BACKEND", raising=False)

    import app.providers.vectorstore_provider as vp

    vp._store = None
    store = vp.get_vectorstore()
    assert isinstance(store, QdrantStore)
    vp._store = None
