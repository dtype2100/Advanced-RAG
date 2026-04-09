"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.embedding.factory import reset as reset_embedding
from app.core.reranker.factory import reset as reset_reranker
from app.core.vectorstore.factory import reset as reset_vectorstore
from app.main import app


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all provider singletons between tests for isolation."""
    reset_embedding()
    reset_reranker()
    reset_vectorstore()
    yield
    reset_embedding()
    reset_reranker()
    reset_vectorstore()


@pytest.fixture()
def client():
    """Provide a synchronous test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c
