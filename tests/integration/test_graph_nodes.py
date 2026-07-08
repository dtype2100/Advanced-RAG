"""Integration tests for CRAG graph nodes with mocked LLM."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.graphs.crag.nodes import expand_context, hybrid_retrieve
from app.rag.retrievers.corpus_registry import clear, register_docstore


@pytest.fixture(autouse=True)
def reset_corpus():
    clear()
    yield
    clear()


def test_hybrid_retrieve_returns_dict_chunks():
    fake_results = [{"text": "doc one", "score": 0.9, "metadata": {"source": "a"}}]

    with patch(
        "app.rag.retrievers.vector_retriever.vector_retrieve",
        return_value=fake_results,
    ):
        state = {"user_query": "What is RAG?", "retrieval_attempt": 0}
        result = hybrid_retrieve(state)

    assert result["retrieval_attempt"] == 1
    assert result["retrieved_children"][0]["text"] == "doc one"


def test_expand_context_uses_parent_text_metadata():
    state = {
        "retrieved_children": [
            {
                "text": "child snippet",
                "score": 0.8,
                "metadata": {"parent_text": "full parent document text"},
            }
        ]
    }
    result = expand_context(state)
    assert result["expanded_contexts"][0]["text"] == "full parent document text"


def test_expand_context_uses_docstore():
    register_docstore({"p1": "parent from docstore"})
    state = {
        "retrieved_children": [
            {"text": "child", "score": 0.7, "metadata": {"parent_id": "p1"}},
        ]
    }
    result = expand_context(state)
    assert result["expanded_contexts"][0]["text"] == "parent from docstore"
