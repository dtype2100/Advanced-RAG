"""Build ``arq.connections.RedisSettings`` from application config."""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import settings


def get_redis_settings() -> RedisSettings:
    """Return ARQ ``RedisSettings`` from ``settings.redis_url`` (DSN string)."""
    if not settings.redis_url:
        raise ValueError("REDIS_URL is not configured; async ingest is unavailable.")
    return RedisSettings.from_dsn(settings.redis_url)
