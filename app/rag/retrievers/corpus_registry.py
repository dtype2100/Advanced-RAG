"""In-memory BM25 corpus and parent docstore synced during ingest."""

from __future__ import annotations

from typing import Any

_corpus_docs: list[dict[str, Any]] = []
_docstore: dict[str, str] = {}


def register_documents(docs: list[dict[str, Any]]) -> None:
    """Append indexed chunks to the BM25 corpus, deduplicating by text."""
    seen = {d["text"] for d in _corpus_docs}
    for doc in docs:
        text = doc.get("text", "")
        if text and text not in seen:
            seen.add(text)
            _corpus_docs.append(
                {
                    "text": text,
                    "score": 0.0,
                    "metadata": dict(doc.get("metadata") or {}),
                }
            )


def register_docstore(entries: dict[str, str]) -> None:
    """Merge parent-id → parent-text mappings into the docstore."""
    _docstore.update(entries)


def get_corpus_docs() -> list[dict[str, Any]]:
    """Return a snapshot of the in-memory BM25 corpus."""
    return list(_corpus_docs)


def get_docstore() -> dict[str, str]:
    """Return a snapshot of the parent docstore."""
    return dict(_docstore)


def clear() -> None:
    """Reset corpus and docstore (primarily for tests)."""
    _corpus_docs.clear()
    _docstore.clear()
