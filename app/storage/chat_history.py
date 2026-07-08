"""Conversation history persistence (Redis-backed with in-memory fallback)."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import settings
from app.storage.redis import get_redis_client

logger = logging.getLogger(__name__)


class ChatHistoryStore:
    """In-memory chat history store for development and testing."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a message dict to the given session's history."""
        self._store.setdefault(session_id, []).append(message)
        self._trim(session_id)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a session."""
        return list(self._store.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Remove all messages for a session."""
        self._store.pop(session_id, None)

    def _trim(self, session_id: str) -> None:
        max_messages = settings.chat_history_max_messages
        if session_id in self._store:
            self._store[session_id] = self._store[session_id][-max_messages:]


class RedisChatHistoryStore:
    """Redis-backed chat history using list operations per session."""

    def __init__(self, prefix: str = "chat:") -> None:
        self._prefix = prefix
        self._client = get_redis_client()
        if self._client is None:
            raise RuntimeError("Redis client is not configured")

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a message to the Redis list for the session."""
        key = self._key(session_id)
        self._client.rpush(key, json.dumps(message))
        self._client.ltrim(key, -settings.chat_history_max_messages, -1)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        """Load session messages from Redis."""
        raw = self._client.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    def clear(self, session_id: str) -> None:
        """Delete the session key from Redis."""
        self._client.delete(self._key(session_id))


_store: ChatHistoryStore | RedisChatHistoryStore | None = None


def get_chat_history_store() -> ChatHistoryStore | RedisChatHistoryStore:
    """Return the process-wide chat history store (Redis when available)."""
    global _store
    if _store is not None:
        return _store

    if settings.redis_url and get_redis_client() is not None:
        try:
            _store = RedisChatHistoryStore()
            logger.info("Chat history: Redis backend (%s)", settings.redis_url)
            return _store
        except Exception:
            logger.warning("Redis chat history unavailable; using in-memory store", exc_info=True)

    _store = ChatHistoryStore()
    logger.info("Chat history: in-memory backend")
    return _store
