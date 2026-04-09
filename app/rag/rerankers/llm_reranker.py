"""LLM-based reranker — asks the LLM to score each document for relevance.

Slower than a cross-encoder but works without a dedicated reranking model.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a relevance judge. Given a question and a document excerpt, "
    "rate the document's relevance to the question on a scale from 0.0 (irrelevant) "
    "to 1.0 (highly relevant). Output ONLY a single float number."
)


class LLMReranker:
    """Reranker that uses the configured LLM to score document relevance."""

    def rerank(
        self, query: str, docs: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Rerank ``docs`` by LLM-assigned relevance scores.

        Args:
            query:  Query string.
            docs:   Candidate document dicts.
            top_k:  Maximum results to return (``None`` = all).

        Returns:
            Re-sorted list with updated ``score`` field.
        """
        if not docs:
            return docs

        llm = get_llm()
        scored: list[tuple[dict[str, Any], float]] = []

        for doc in docs:
            try:
                response = llm.invoke(
                    [
                        SystemMessage(content=_SYSTEM),
                        HumanMessage(content=f"Question: {query}\n\nDocument: {doc['text'][:500]}"),
                    ]
                )
                score = float(response.content.strip())
            except (ValueError, Exception):
                score = 0.0
            scored.append((doc, score))

        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        results = [{**doc, "score": s} for doc, s in ranked]
        return results[:top_k] if top_k else results
