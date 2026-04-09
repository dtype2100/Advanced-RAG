"""LangGraph node functions for the self-corrective RAG pipeline."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.config import settings
from app.providers.llm import get_chat_llm
from app.rag.prompts import (
    GENERATE_HUMAN,
    GENERATE_SYSTEM,
    GRADER_HUMAN,
    GRADER_SYSTEM,
    REWRITE_HUMAN,
    REWRITE_SYSTEM,
)
from app.rag.state import RAGState
from app.services.search import search_documents

logger = logging.getLogger(__name__)


# ── Node: Retrieve ──────────────────────────────────────────────────────────


def retrieve(state: RAGState) -> RAGState:
    """Retrieve documents from Qdrant for the current question."""
    query = state.get("rewritten_question") or state["question"]
    logger.info("Retrieving documents for: %s", query)

    results = search_documents(query, top_k=settings.max_retrieval_docs)
    documents = [r["text"] for r in results]
    scores = [r["score"] for r in results]

    return {**state, "documents": documents, "scores": scores}


# ── Node: Grade Documents ───────────────────────────────────────────────────


def grade_documents(state: RAGState) -> RAGState:
    """Grade each retrieved document for relevance, keeping only relevant ones."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    if not documents:
        logger.warning("No documents to grade")
        return {**state, "is_relevant": False, "documents": []}

    llm = get_chat_llm()
    relevant_docs: list[str] = []

    for doc in documents:
        response = llm.invoke(
            [
                SystemMessage(content=GRADER_SYSTEM),
                HumanMessage(content=GRADER_HUMAN.format(question=question, document=doc)),
            ]
        )
        verdict = response.content.strip().lower()
        if "relevant" in verdict and "irrelevant" not in verdict:
            relevant_docs.append(doc)

    is_relevant = len(relevant_docs) > 0
    logger.info("Grading: %d/%d documents relevant", len(relevant_docs), len(documents))

    return {**state, "documents": relevant_docs, "is_relevant": is_relevant}


# ── Node: Rewrite Query ─────────────────────────────────────────────────────


def rewrite_query(state: RAGState) -> RAGState:
    """Rewrite the question for better retrieval on retry."""
    question = state.get("rewritten_question") or state["question"]
    retries = state.get("retries", 0) + 1
    logger.info("Rewriting query (retry %d): %s", retries, question)

    llm = get_chat_llm()
    response = llm.invoke(
        [
            SystemMessage(content=REWRITE_SYSTEM),
            HumanMessage(content=REWRITE_HUMAN.format(question=question)),
        ]
    )
    rewritten = response.content.strip()
    logger.info("Rewritten to: %s", rewritten)

    return {**state, "rewritten_question": rewritten, "retries": retries}


# ── Node: Generate Answer ───────────────────────────────────────────────────


def generate(state: RAGState) -> RAGState:
    """Generate an answer from the relevant documents."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    context = "\n\n---\n\n".join(documents) if documents else "(No relevant documents found)"
    logger.info("Generating answer from %d documents", len(documents))

    llm = get_chat_llm()
    response = llm.invoke(
        [
            SystemMessage(content=GENERATE_SYSTEM),
            HumanMessage(content=GENERATE_HUMAN.format(context=context, question=question)),
        ]
    )
    return {**state, "generation": response.content.strip()}


# ── Routing Logic ────────────────────────────────────────────────────────────


def should_retry(state: RAGState) -> str:
    """Decide whether to retry retrieval or proceed to generation.

    Returns:
        "rewrite" if documents are irrelevant and retries remain.
        "generate" if documents are relevant or retries exhausted.
    """
    is_relevant = state.get("is_relevant", False)
    retries = state.get("retries", 0)

    if is_relevant:
        return "generate"
    if retries >= settings.max_retries:
        logger.warning("Max retries (%d) reached, generating with available context", retries)
        return "generate"
    return "rewrite"
