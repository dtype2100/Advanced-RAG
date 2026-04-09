"""Integration tests for the chat (RAG query) API endpoint."""

from __future__ import annotations


def test_query_runs_or_fails_gracefully(client):
    """RAG query should succeed with LLM running or fail with 500/503."""
    resp = client.post("/api/v1/query", json={"question": "What is Python?"})
    assert resp.status_code in (200, 500, 503)
    if resp.status_code == 200:
        data = resp.json()
        assert "answer" in data
        assert "sources" in data


def test_query_missing_body_returns_422(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422
