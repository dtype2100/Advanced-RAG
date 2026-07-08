"""Integration tests for the Studio configuration API."""

from __future__ import annotations


def test_studio_config(client):
    resp = client.get("/api/v1/studio/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm_backend" in data
    assert "grounding_threshold" in data


def test_studio_probe(client):
    resp = client.post("/api/v1/studio/probe")
    assert resp.status_code == 200
    data = resp.json()
    assert "llm" in data
    assert "qdrant" in data
