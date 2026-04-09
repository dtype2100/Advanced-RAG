"""Backward-compatibility re-exports for legacy code using app.api.schemas.

New code should import directly from app.schemas.*.
"""

from __future__ import annotations

from app.schemas.document import DocumentInput, IngestRequest, IngestResponse
from app.schemas.request import ChatRequest as QueryRequest
from app.schemas.request import SearchRequest
from app.schemas.response import ChatResponse as QueryResponse
from app.schemas.response import HealthResponse, SearchResponse, SearchResult

__all__ = [
    "DocumentInput",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SearchRequest",
    "SearchResult",
    "SearchResponse",
    "HealthResponse",
]
