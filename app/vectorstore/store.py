"""Vector store public interface.

This module is a thin compatibility shim that delegates all operations to
``app.providers.vectorstore``.  Application code that already imports from
this module continues to work unchanged; the concrete backend is selected by
the ``VECTORSTORE_BACKEND`` environment variable.
"""

from __future__ import annotations

from app.providers.vectorstore import (  # noqa: F401  re-export for callers
    add_documents,
    ensure_collection,
    get_client,
    search,
)

__all__ = [
    "get_client",
    "ensure_collection",
    "add_documents",
    "search",
]
