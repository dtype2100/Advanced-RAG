"""Unit tests for the context expansion policy."""

from __future__ import annotations

from app.rag.policies.expansion_policy import should_expand


def test_empty_children_no_expand():
    state = {"retrieved_children": []}
    assert should_expand(state) is False


def test_short_chunks_trigger_expansion():
    state = {
        "retrieved_children": [
            {"text": "short", "score": 0.5, "metadata": {}},
            {"text": "tiny", "score": 0.4, "metadata": {}},
        ]
    }
    assert should_expand(state) is True


def test_short_chunks_trigger_expansion_string_form():
    state = {"retrieved_children": ["short", "tiny", "small chunk"]}
    assert should_expand(state) is True


def test_long_chunks_no_expansion():
    long_chunk = "x" * 500
    state = {"retrieved_children": [{"text": long_chunk, "score": 0.5, "metadata": {}}]}
    assert should_expand(state) is False
