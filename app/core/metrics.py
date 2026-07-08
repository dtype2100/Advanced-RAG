"""Prometheus metrics for HTTP and RAG pipeline observability."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total RAG query invocations",
    ["status"],
)

RAG_QUERY_DURATION_SECONDS = Histogram(
    "rag_query_duration_seconds",
    "RAG query latency in seconds",
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

INGEST_DOCUMENTS_TOTAL = Counter(
    "ingest_documents_total",
    "Total document ingest operations",
    ["mode", "status"],
)


def metrics_response() -> Response:
    """Return the Prometheus metrics exposition payload."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record HTTP request count and latency for Prometheus."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        path = _normalize_path(request.url.path)
        HTTP_REQUESTS_TOTAL.labels(
            method=request.method,
            path=path,
            status=str(response.status_code),
        ).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path).observe(duration)
        return response


def _normalize_path(path: str) -> str:
    """Collapse dynamic path segments to reduce metric cardinality."""
    if path.startswith("/api/v1/jobs/"):
        return "/api/v1/jobs/{job_id}"
    return path


def record_rag_query(*, status: str, duration_seconds: float) -> None:
    """Increment RAG query counters after a pipeline run."""
    RAG_QUERIES_TOTAL.labels(status=status).inc()
    RAG_QUERY_DURATION_SECONDS.observe(duration_seconds)


def record_ingest(*, mode: str, status: str) -> None:
    """Increment ingest counters after a document ingest operation."""
    INGEST_DOCUMENTS_TOTAL.labels(mode=mode, status=status).inc()
