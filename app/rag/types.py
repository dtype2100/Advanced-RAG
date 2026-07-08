"""Shared retrieval chunk types and normalisation helpers."""

from __future__ import annotations

from typing import Any, TypedDict


class ChunkHit(TypedDict, total=False):
    """A single retrieval result passed through the CRAG graph."""

    text: str
    score: float
    metadata: dict[str, Any]


def chunk_text(chunk: str | dict[str, Any]) -> str:
    """Extract plain text from a chunk string or ``{text, ...}`` dict."""
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(chunk)


def normalize_chunks(chunks: list[str | dict[str, Any]]) -> list[ChunkHit]:
    """Coerce mixed chunk representations into uniform ``ChunkHit`` dicts."""
    normalised: list[ChunkHit] = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            normalised.append(
                {
                    "text": str(chunk.get("text", "")),
                    "score": float(chunk.get("score", 0.0)),
                    "metadata": dict(chunk.get("metadata") or {}),
                }
            )
        else:
            normalised.append({"text": str(chunk), "score": 0.0, "metadata": {}})
    return normalised


def chunks_to_texts(chunks: list[str | dict[str, Any]]) -> list[str]:
    """Return non-empty text strings from heterogeneous chunk lists."""
    texts = [chunk_text(c) for c in chunks]
    return [t for t in texts if t]
