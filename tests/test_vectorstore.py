"""Vector store unit tests (in-memory Qdrant, no external deps)."""

from __future__ import annotations

from app.core.vectorstore import get_vectorstore


def test_add_and_search():
    """Ingest docs and verify semantic search returns ranked results."""
    store = get_vectorstore()
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Deep learning uses neural networks with many layers.",
    ]
    count = store.add_documents(texts)
    assert count == 3

    results = store.search("AI and neural networks", top_k=2)
    assert len(results) == 2
    assert all(r["score"] > 0 for r in results)


def test_dedup_on_re_add():
    """Same text inserted twice should not duplicate (deterministic ID)."""
    store = get_vectorstore()
    text = "Duplicate test document for deduplication."
    store.add_documents([text])
    store.add_documents([text])

    results = store.search(text, top_k=10)
    matching = [r for r in results if r["text"] == text]
    assert len(matching) == 1
