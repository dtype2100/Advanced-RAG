"""Provider abstraction unit tests.

Tests that the provider layer correctly routes to the right backend
based on config values, without requiring external services.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Embedding provider tests ──────────────────────────────────────────────────


def test_embedding_dim_fastembed():
    """get_embedding_dim() returns the correct dimension for a FastEmbed model."""
    from app.providers.embeddings import get_embedding_dim

    assert get_embedding_dim() == 384  # BAAI/bge-small-en-v1.5 default


def test_embed_texts_returns_list_of_vectors():
    """embed_texts() returns one vector per input text."""
    from app.providers.embeddings import embed_texts

    vecs = embed_texts(["Hello world", "How are you?"])
    assert len(vecs) == 2
    assert all(isinstance(v, list) for v in vecs)
    assert all(len(v) > 0 for v in vecs)


def test_embed_query_returns_single_vector():
    """embed_query() returns a single flat list."""
    from app.providers.embeddings import embed_query

    vec = embed_query("test query")
    assert isinstance(vec, list)
    assert len(vec) > 0


# ── Reranker provider tests ───────────────────────────────────────────────────


def test_rerank_none_passthrough():
    """When RERANKER_PROVIDER=none, rerank() preserves order and applies top_k."""
    from app.providers.reranker import rerank

    docs = ["doc_a", "doc_b", "doc_c"]
    result = rerank("query", docs, top_k=2)
    assert result == ["doc_a", "doc_b"]


def test_rerank_empty_input():
    """rerank() with empty input returns empty list."""
    from app.providers.reranker import rerank

    assert rerank("q", [], top_k=5) == []


def test_rerank_cross_encoder_mocked():
    """cross-encoder rerank() calls CrossEncoder.predict and re-sorts docs."""
    mock_ce = MagicMock()
    import numpy as np

    mock_ce.predict.return_value = np.array([0.1, 0.9, 0.5])

    with (
        patch("app.providers.reranker.settings") as mock_settings,
        patch("app.providers.reranker._reranker", mock_ce),
    ):
        mock_settings.reranker_provider = "cross-encoder"
        mock_settings.using_reranker = True

        from app.providers import reranker as reranker_mod

        docs = ["low_score", "high_score", "mid_score"]
        result = reranker_mod.rerank("query", docs, top_k=3)
        assert result[0] == "high_score"
        assert result[1] == "mid_score"
        assert result[2] == "low_score"


# ── Vector store provider tests ───────────────────────────────────────────────


def test_vectorstore_add_and_search():
    """add_documents() and search() work end-to-end via the provider abstraction."""
    from app.providers.vectorstore import add_documents, search

    texts = ["Provider abstraction test document one.", "Provider abstraction test document two."]
    count = add_documents(texts)
    assert count == 2

    results = search("abstraction test", top_k=2)
    assert len(results) <= 2
    assert all("score" in r and "text" in r and "metadata" in r for r in results)


# ── LLM provider factory test ─────────────────────────────────────────────────


def test_get_llm_vllm_returns_chat_openai():
    """get_llm() with LLM_BACKEND=vllm returns a ChatOpenAI instance."""
    from app.providers.llm import get_llm

    with patch("app.providers.llm.settings") as mock_settings:
        mock_settings.llm_backend = "vllm"
        mock_settings.llm_model = "Qwen/Qwen2.5-0.5B-Instruct"
        mock_settings.llm_temperature = 0.0
        mock_settings.vllm_base_url = "http://localhost:8001/v1"

        from langchain_openai import ChatOpenAI

        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)


def test_get_llm_unsupported_raises():
    """get_llm() raises ValueError for unknown LLM_BACKEND values."""
    from app.providers import llm as llm_mod

    with patch("app.providers.llm.settings") as mock_settings:
        mock_settings.llm_backend = "unknown_backend"
        with pytest.raises(ValueError, match="Unsupported LLM_BACKEND"):
            llm_mod.get_llm()
