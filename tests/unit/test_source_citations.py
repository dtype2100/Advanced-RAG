"""Unit tests for structured source citations."""

from __future__ import annotations

import pytest

from app.rag.citations.source_citations import build_source_citations


def test_build_source_citations():
    results = [
        {
            "text": "chunk text",
            "score": 0.9123,
            "metadata": {"source": "doc.pdf", "page": "3"},
        }
    ]
    citations = build_source_citations(results)
    assert citations[0].text == "chunk text"
    assert citations[0].source == "doc.pdf"
    assert citations[0].page == "3"
    assert citations[0].score == pytest.approx(0.9123)
