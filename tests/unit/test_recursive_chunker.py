"""Unit tests for recursive chunk metadata propagation."""

from __future__ import annotations

import pytest

from app.rag.chunkers.recursive_chunker import recursive_chunk


def test_metadata_is_copied_onto_every_chunk():
    chunks = recursive_chunk(
        ["alpha " * 80, "beta " * 80],
        chunk_size=64,
        chunk_overlap=0,
        metadatas=[
            {"source_id": "a", "section_id": "one"},
            {"source_id": "b", "section_id": "two"},
        ],
    )
    assert chunks
    assert all(chunk["metadata"]["source_id"] in {"a", "b"} for chunk in chunks)
    assert chunks[0]["metadata"]["section_id"] == "one"
    assert chunks[-1]["metadata"]["section_id"] == "two"


def test_chunk_metadata_is_not_aliased_to_input_dict():
    meta = {"source_id": "a", "section_id": "one"}
    chunks = recursive_chunk(["hello world"], metadatas=[meta])
    chunks[0]["metadata"]["source_id"] = "mutated"
    assert meta["source_id"] == "a"


def test_metadatas_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        recursive_chunk(["one", "two"], metadatas=[{"source_id": "a"}])


def test_ingest_pipeline_keeps_source_labels(monkeypatch):
    captured: dict = {}

    class FakeStore:
        def add_documents(self, texts, metadatas=None):
            captured["metadatas"] = metadatas
            return len(texts)

    monkeypatch.setattr(
        "app.rag.pipelines.ingest_pipeline.get_vectorstore",
        lambda: FakeStore(),
    )
    from app.rag.pipelines.ingest_pipeline import run_ingest

    count = run_ingest(
        [
            {
                "text": "Qdrant stores vectors with a JSON payload.",
                "metadata": {"source_id": "qdrant", "section_id": "storage"},
            }
        ]
    )
    assert count >= 1
    assert captured["metadatas"][0]["source_id"] == "qdrant"
    assert captured["metadatas"][0]["section_id"] == "storage"
