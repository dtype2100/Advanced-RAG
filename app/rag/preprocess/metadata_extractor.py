"""Metadata extraction from raw document dicts."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime


def extract_metadata(doc: dict) -> dict:
    """Enrich a document dict with auto-generated metadata fields.

    Adds:
    - ``content_hash``: SHA-256 of the text (for deduplication).
    - ``ingested_at``:  ISO 8601 UTC timestamp.
    - ``char_count``:   Character length of the text.

    Args:
        doc: Dict containing at least a ``text`` key.

    Returns:
        Updated dict with enriched ``metadata`` sub-dict.
    """
    text = doc.get("text", "")
    meta = dict(doc.get("metadata", {}))
    meta["content_hash"] = hashlib.sha256(text.encode()).hexdigest()
    meta["ingested_at"] = datetime.now(tz=UTC).isoformat()
    meta["char_count"] = str(len(text))
    return {**doc, "metadata": meta}
