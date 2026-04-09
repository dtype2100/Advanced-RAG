"""Unit tests for graph node utility functions."""

from __future__ import annotations

from app.graphs.shared.helpers import get_active_query


def test_get_active_query_prefers_rewritten():
    state = {
        "user_query": "original",
        "rewritten_query": "rewritten",
        "clarified_query": "clarified",
    }
    assert get_active_query(state) == "rewritten"


def test_get_active_query_falls_back_to_clarified():
    state = {"user_query": "original", "clarified_query": "clarified"}
    assert get_active_query(state) == "clarified"


def test_get_active_query_falls_back_to_user_query():
    state = {"user_query": "original"}
    assert get_active_query(state) == "original"


def test_get_active_query_empty_state():
    assert get_active_query({}) == ""
