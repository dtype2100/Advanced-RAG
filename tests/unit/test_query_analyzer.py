"""Unit tests for the query analyzer."""

from __future__ import annotations

from app.rag.query.query_analyzer import analyze


def test_short_query_is_ambiguous():
    result = analyze("AI")
    assert result["is_ambiguous"] is True


def test_factual_intent():
    result = analyze("What is retrieval augmented generation?")
    assert result["intent"] == "factual"


def test_comparison_intent():
    result = analyze("Compare Qdrant vs Pinecone for vector search")
    assert result["intent"] == "comparison"


def test_procedural_intent():
    result = analyze("How to build a RAG pipeline with LangChain step by step")
    assert result["intent"] == "procedural"


def test_missing_time_slot():
    result = analyze("What were the best AI models released when?")
    assert "time_period" in result["missing_slots"]


def test_no_missing_slots_for_complete_query():
    result = analyze("What is the capital of France?")
    assert "time_period" not in result["missing_slots"]
    assert "location" not in result["missing_slots"]
