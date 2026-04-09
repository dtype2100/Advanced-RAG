"""Tests for env-driven component switching."""

from __future__ import annotations

from app.config import Settings, settings
from app.vectorstore import store


def test_settings_support_env_only_component_switch(monkeypatch):
    """Component selection should be fully configurable through environment variables."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "fastembed")
    monkeypatch.setenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    monkeypatch.setenv("RERANKER_BACKEND", "none")
    monkeypatch.setenv("RERANKER_MODEL", "custom-reranker-model")
    monkeypatch.setenv("VECTOR_DB_BACKEND", "memory")
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    monkeypatch.setenv("COLLECTION_NAME", "legacy_collection_alias")

    cfg = Settings()

    assert cfg.embedding_backend == "fastembed"
    assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.reranker_backend == "none"
    assert cfg.effective_reranker_model == "custom-reranker-model"
    assert cfg.vector_db_backend == "memory"
    assert cfg.vector_db_collection_name == "legacy_collection_alias"
    assert cfg.collection_name == "legacy_collection_alias"


def test_vector_db_can_switch_to_memory_backend():
    """VECTOR_DB_BACKEND=memory should work without changing code."""
    original_values = {
        "vector_db_backend": settings.vector_db_backend,
        "vector_db_collection_name": settings.vector_db_collection_name,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model,
    }

    try:
        settings.vector_db_backend = "memory"
        settings.vector_db_collection_name = "memory_switch_test"
        settings.embedding_backend = "fastembed"
        settings.embedding_model = "BAAI/bge-small-en-v1.5"
        store.reset_runtime_state()

        count = store.add_documents(
            [
                "RAG combines retrieval with generation.",
                "Qdrant stores vectors efficiently.",
            ]
        )
        assert count == 2

        results = store.search("retrieval and generation", top_k=2)
        assert len(results) == 2
        assert all("text" in item and "score" in item for item in results)
    finally:
        settings.vector_db_backend = original_values["vector_db_backend"]
        settings.vector_db_collection_name = original_values["vector_db_collection_name"]
        settings.embedding_backend = original_values["embedding_backend"]
        settings.embedding_model = original_values["embedding_model"]
        store.reset_runtime_state()
