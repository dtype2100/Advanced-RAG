"""API endpoint tests (no OpenAI key required for non-RAG routes)."""

from __future__ import annotations


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["service"] == "Advanced RAG API"


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"
    assert "llm_backend" in data
    assert "llm_model" in data


def test_ingest_and_search(client):
    """Full cycle: ingest documents then search them."""
    docs = {
        "documents": [
            {"text": "Python is a popular programming language created by Guido van Rossum."},
            {"text": "FastAPI is a modern web framework for building APIs with Python."},
            {"text": "Qdrant is a vector similarity search engine written in Rust."},
        ]
    }
    resp = client.post("/api/v1/documents", json=docs)
    assert resp.status_code == 200
    assert resp.json()["count"] == 3

    resp = client.post("/api/v1/search", json={"query": "web framework", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) <= 2
    assert any("FastAPI" in r["text"] for r in results)


def test_ingest_empty_fails(client):
    resp = client.post("/api/v1/documents", json={"documents": []})
    assert resp.status_code == 422


def test_query_without_llm_backend(client):
    """RAG query should fail or return error when LLM backend is unreachable."""
    resp = client.post("/api/v1/query", json={"question": "What is Python?"})
    assert resp.status_code in (500, 503)
