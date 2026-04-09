"""LangGraph checkpointer factory.

Returns an in-memory checkpointer by default; swap to a PostgreSQL-backed
checkpointer when persistent graph state is required.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer() -> MemorySaver:
    """Return the active LangGraph checkpointer.

    Returns:
        ``MemorySaver`` for development; replace with ``AsyncPostgresSaver``
        for production multi-turn state persistence.
    """
    return MemorySaver()
