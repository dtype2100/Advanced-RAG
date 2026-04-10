"""Singleton ARQ Redis connection pool for the API process."""

from __future__ import annotations

import logging

from arq.connections import ArqRedis

from app.queue.redis_settings import get_redis_settings

logger = logging.getLogger(__name__)

_pool: ArqRedis | None = None


async def get_arq_pool() -> ArqRedis:
    """Return (or create) the shared ``ArqRedis`` pool used to enqueue jobs."""
    global _pool
    if _pool is None:
        from arq import create_pool

        from app.core.config import settings

        _pool = await create_pool(
            get_redis_settings(),
            default_queue_name=settings.arq_queue_name,
        )
        logger.info("ARQ pool connected (queue=%s)", settings.arq_queue_name)
    return _pool


async def close_arq_pool() -> None:
    """Close the ARQ pool (call from app shutdown)."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("ARQ pool closed")
