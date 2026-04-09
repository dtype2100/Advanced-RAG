"""Document deduplication based on content hashes."""

from __future__ import annotations

import hashlib


def dedup_documents(docs: list[dict]) -> list[dict]:
    """Remove duplicate documents by SHA-256 content hash.

    Args:
        docs: List of document dicts, each with a ``text`` key.

    Returns:
        De-duplicated list preserving original order (first occurrence kept).
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for doc in docs:
        h = hashlib.sha256(doc.get("text", "").encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique
