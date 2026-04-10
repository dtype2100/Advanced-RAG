"""Integration tests for ARQ-backed async ingest (no Redis required for 503 paths)."""

from __future__ import annotations

import pytest


def test_documents_async_returns_503_without_redis(client, monkeypatch):
    """Without REDIS_URL, async ingest must not silently fail."""
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod.settings, "redis_url", "")
    resp = client.post(
        "/api/v1/documents/async",
        json={"documents": [{"text": "hello world"}]},
    )
    assert resp.status_code == 503


def test_job_status_returns_503_without_redis(client, monkeypatch):
    from app.core import config as config_mod

    monkeypatch.setattr(config_mod.settings, "redis_url", "")
    resp = client.get("/api/v1/jobs/fake-job-id")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_ingest_documents_job_empty_docs():
    """Worker job runs run_ingest in a thread; empty input returns count 0."""
    from app.workers.jobs import ingest_documents_job

    out = await ingest_documents_job({}, [])
    assert out["count"] == 0
    assert "message" in out
