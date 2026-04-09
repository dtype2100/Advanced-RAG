"""Integration tests for the document ingestion and search API endpoints."""

from __future__ import annotations


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["service"] == "Advanced RAG API"


def test_health(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["qdrant"] == "connected"
    assert "llm_backend" in data


def test_ingest_and_search(client):
    docs = {
        "documents": [
            {"text": "Python is a popular programming language created by Guido van Rossum."},
            {"text": "FastAPI is a modern web framework for building APIs with Python."},
            {"text": "Qdrant is a vector similarity search engine written in Rust."},
        ]
    }
    resp = client.post("/api/v1/documents", json=docs)
    assert resp.status_code == 200
    assert resp.json()["count"] >= 1

    resp = client.post("/api/v1/search", json={"query": "web framework", "top_k": 2})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) <= 2


def test_ingest_empty_fails(client):
    resp = client.post("/api/v1/documents", json={"documents": []})
    assert resp.status_code == 422
