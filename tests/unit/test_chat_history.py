"""Unit tests for chat history persistence."""

from __future__ import annotations

from app.storage.chat_history import ChatHistoryStore, get_chat_history_store


def test_in_memory_chat_history_roundtrip():
    store = ChatHistoryStore()
    store.append("s1", {"role": "user", "content": "hello"})
    store.append("s1", {"role": "assistant", "content": "hi"})
    messages = store.get("s1")
    assert len(messages) == 2
    assert messages[0]["content"] == "hello"


def test_get_chat_history_store_returns_singleton():
    a = get_chat_history_store()
    b = get_chat_history_store()
    assert a is b
