"""Conversation history persistence.

Placeholder — integrate with Redis or PostgreSQL when multi-turn chat is required.
"""

from __future__ import annotations

from typing import Any


class ChatHistoryStore:
    """In-memory chat history store (development / testing only).

    Replace with a persistent backend (Redis, PostgreSQL) for production.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[dict[str, Any]]] = {}

    def append(self, session_id: str, message: dict[str, Any]) -> None:
        """Append a message dict to the given session's history."""
        self._store.setdefault(session_id, []).append(message)

    def get(self, session_id: str) -> list[dict[str, Any]]:
        """Retrieve all messages for a session."""
        return self._store.get(session_id, [])

    def clear(self, session_id: str) -> None:
        """Remove all messages for a session."""
        self._store.pop(session_id, None)
