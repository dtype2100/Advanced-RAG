"""LangGraph state definition for the Advanced RAG pipeline."""

from __future__ import annotations

from typing import TypedDict


class RAGState(TypedDict, total=False):
    """Shared state flowing through the RAG graph.

    Attributes:
        question: Original user question.
        rewritten_question: Rewritten question for better retrieval.
        documents: Retrieved document texts.
        scores: Relevance scores from vector search.
        generation: LLM-generated answer.
        retries: Number of retrieval retries so far.
        is_relevant: Whether retrieved documents are relevant.
    """

    question: str
    rewritten_question: str
    documents: list[str]
    scores: list[float]
    generation: str
    retries: int
    is_relevant: bool
