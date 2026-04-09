"""Shared test fixtures for the Advanced RAG test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """Synchronous FastAPI test client (session-scoped for performance)."""
    with TestClient(app) as c:
        yield c
