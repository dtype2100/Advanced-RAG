"""Shared test fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.embedding.factory import reset_embedding
from app.core.reranker.factory import reset_reranker
from app.core.vectorstore.factory import reset_vectorstore


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset all cached provider singletons between test sessions."""
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
    from app.main import app

    with TestClient(app) as c:
        yield c
