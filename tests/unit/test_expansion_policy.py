"""Unit tests for the context expansion policy."""

from __future__ import annotations

from app.rag.policies.expansion_policy import should_expand


def test_empty_children_no_expand():
    state = {"retrieved_children": []}
    assert should_expand(state) is False


def test_short_chunks_trigger_expansion():
    state = {"retrieved_children": ["short", "tiny", "small chunk"]}
    assert should_expand(state) is True


def test_long_chunks_no_expansion():
    long_chunk = "x" * 500
    state = {"retrieved_children": [long_chunk, long_chunk]}
    assert should_expand(state) is False


def test_dict_children_uses_text_length():
    state = {
        "retrieved_children": [
            {"text": "x" * 20, "metadata": {}},
            {"text": "y" * 25, "metadata": {}},
        ]
    }
    assert should_expand(state) is True


def test_dict_children_long_text_no_expand():
    state = {
        "retrieved_children": [
            {"text": "x" * 400, "metadata": {}},
            {"text": "y" * 500, "metadata": {}},
        ]
    }
    assert should_expand(state) is False
