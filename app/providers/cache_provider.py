"""Cache provider — returns an in-process dict cache or a Redis-backed cache.

Placeholder: replace ``SimpleCache`` with ``RedisCache`` in production.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class SimpleCache:
    """Minimal in-memory key-value cache for development / testing."""

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        """Return cached value or ``None`` if not present."""
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        """Store a value under ``key``."""
        self._data[key] = value

    def delete(self, key: str) -> None:
        """Remove a key from the cache."""
        self._data.pop(key, None)


_cache: SimpleCache | None = None


def get_cache() -> SimpleCache:
    """Return the singleton cache instance."""
    global _cache
    if _cache is None:
        _cache = SimpleCache()
        logger.info("Cache provider: SimpleCache (in-memory)")
    return _cache
