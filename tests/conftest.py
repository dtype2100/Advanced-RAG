"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client():
    """Provide a synchronous test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c
