"""Vector store unit tests (in-memory Qdrant, no external deps)."""

from __future__ import annotations

from app.providers.vectorstore import add_documents, search


def test_add_and_search():
    """Ingest docs and verify semantic search returns ranked results."""
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "The Eiffel Tower is located in Paris, France.",
        "Deep learning uses neural networks with many layers.",
    ]
    count = add_documents(texts)
    assert count == 3

    results = search("AI and neural networks", top_k=2)
    assert len(results) == 2
    assert all(r["score"] > 0 for r in results)


def test_dedup_on_re_add():
    """Same text inserted twice should not duplicate (deterministic ID)."""
    text = "Duplicate test document for deduplication."
    add_documents([text])
    add_documents([text])

    results = search(text, top_k=10)
    matching = [r for r in results if r["text"] == text]
    assert len(matching) == 1
