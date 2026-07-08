"""HTTP middleware for request tracing, rate limiting, and security headers."""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import settings

logger = logging.getLogger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every request and response."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        request_id = getattr(request.state, "request_id", "-")
        logger.info(
            "request completed method=%s path=%s status=%s duration_ms=%.1f request_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per client IP."""

    def __init__(self, app, limit_per_minute: int) -> None:
        super().__init__(app)
        self._limit = max(limit_per_minute, 0)
        self._windows: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        if self._limit <= 0 or request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        window = self._windows[client_ip]

        while window and now - window[0] > 60:
            window.popleft()

        if len(window) >= self._limit:
            request_id = getattr(request.state, "request_id", "-")
            logger.warning(
                "rate limit exceeded client=%s path=%s request_id=%s",
                client_ip,
                request.url.path,
                request_id,
            )
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
                headers={"Retry-After": "60"},
            )

        window.append(now)
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add standard security headers to every response."""

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("X-XSS-Protection", "1; mode=block")
        return response


_EXEMPT_PATHS = frozenset(
    {
        "/",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/health/ready",
        "/metrics",
    }
)


def get_cors_origins() -> list[str]:
    """Parse CORS origins from settings (comma-separated string)."""
    raw = settings.cors_origins.strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
