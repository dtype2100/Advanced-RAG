"""Integration tests for operational endpoints (health, metrics, auth, rate limit)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import app


@pytest.fixture
def authed_client(monkeypatch):
    """Test client with API key auth enabled."""
    monkeypatch.setattr(settings, "api_key", "test-secret-key")
    with TestClient(app) as client:
        yield client


def test_liveness(client):
    resp = client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "alive"


def test_readiness(client):
    resp = client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["checks"]["vectorstore"]["status"] == "connected"


def test_health_includes_ops_fields(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in {"ok", "degraded"}
    assert data["version"] == "0.2.0"
    assert "redis" in data
    assert "auth_enabled" in data


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text


def test_request_id_header(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers


def test_security_headers(client):
    resp = client.get("/")
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"


def test_api_key_required_when_configured(authed_client):
    resp = authed_client.post("/api/v1/search", json={"query": "test", "top_k": 1})
    assert resp.status_code == 401

    resp = authed_client.post(
        "/api/v1/search",
        json={"query": "test", "top_k": 1},
        headers={"X-API-Key": "test-secret-key"},
    )
    assert resp.status_code == 200


def test_health_endpoints_skip_auth_when_configured(authed_client):
    resp = authed_client.get("/api/v1/health/live")
    assert resp.status_code == 200

    resp = authed_client.get("/metrics")
    assert resp.status_code == 200


def test_error_response_includes_request_id(client):
    resp = client.post("/api/v1/documents", json={"documents": []})
    assert resp.status_code == 422
    body = resp.json()
    assert "request_id" in body
