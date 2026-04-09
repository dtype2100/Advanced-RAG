"""Backward-compatibility shim — delegates to app.storage.vectorstores.qdrant_store."""

from __future__ import annotations

from app.storage.vectorstores.qdrant_store import QdrantStore

_store = QdrantStore()
