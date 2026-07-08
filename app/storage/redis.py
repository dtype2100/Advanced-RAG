"""Redis connection helper for caching, chat history, and session management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    import redis

_redis_client: redis.Redis | None = None


def get_redis_client():
    """Return a sync Redis client when ``REDIS_URL`` is configured.

    Returns:
        ``redis.Redis`` instance, or ``None`` when Redis is not configured.
    """
    global _redis_client

    if not settings.redis_url:
        return None

    if _redis_client is None:
        import redis

        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)

    return _redis_client
