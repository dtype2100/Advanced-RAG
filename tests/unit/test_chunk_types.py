"""Unit tests for chunk type helpers."""

from __future__ import annotations

from app.rag.types import chunk_text, chunks_to_texts, normalize_chunks


def test_chunk_text_from_dict():
    assert chunk_text({"text": "hello", "score": 0.9}) == "hello"


def test_normalize_chunks_mixed_input():
    result = normalize_chunks(["plain", {"text": "dict", "score": 1.0}])
    assert result[0]["text"] == "plain"
    assert result[1]["text"] == "dict"


def test_chunks_to_texts_skips_empty():
    assert chunks_to_texts([{"text": ""}, {"text": "ok"}]) == ["ok"]
