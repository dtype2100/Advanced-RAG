"""Reranker provider abstractions and factory."""

from __future__ import annotations

from typing import Protocol

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.llm.provider import get_chat_model, message_content_to_text
from app.rag.prompts import GRADER_HUMAN, GRADER_SYSTEM


class Reranker(Protocol):
    """Contract for reranker implementations."""

    def filter_relevant(self, question: str, documents: list[str]) -> list[str]:
        """Filter documents and keep only relevant context."""


class NoopReranker:
    """Pass-through reranker that keeps all documents."""

    def filter_relevant(self, question: str, documents: list[str]) -> list[str]:
        """Return input documents without reranking."""
        _ = question
        return documents


class LLMReranker:
    """LLM-based binary relevance reranker."""

    def __init__(self, model_name: str):
        """Store reranker model configuration."""
        self.model_name = model_name

    def filter_relevant(self, question: str, documents: list[str]) -> list[str]:
        """Keep documents classified as relevant by the configured LLM reranker."""
        if not documents:
            return []

        llm = get_chat_model(model_name=self.model_name)
        relevant_docs: list[str] = []

        for document in documents:
            response = llm.invoke(
                [
                    SystemMessage(content=GRADER_SYSTEM),
                    HumanMessage(content=GRADER_HUMAN.format(question=question, document=document)),
                ]
            )
            verdict = message_content_to_text(response.content).lower()
            if "relevant" in verdict and "irrelevant" not in verdict:
                relevant_docs.append(document)

        return relevant_docs


_reranker_key: tuple[str, str] | None = None
_reranker_cache: Reranker | None = None


def get_reranker() -> Reranker:
    """Build or reuse reranker from current environment configuration."""
    global _reranker_key, _reranker_cache

    key = (settings.reranker_backend, settings.effective_reranker_model)
    if _reranker_cache is not None and _reranker_key == key:
        return _reranker_cache

    if settings.reranker_backend == "none":
        _reranker_cache = NoopReranker()
    elif settings.reranker_backend == "llm":
        _reranker_cache = LLMReranker(model_name=settings.effective_reranker_model)
    else:
        raise ValueError(f"Unsupported RERANKER_BACKEND: {settings.reranker_backend}")

    _reranker_key = key
    return _reranker_cache


def reset_reranker_cache() -> None:
    """Reset cached reranker to pick up runtime configuration changes."""
    global _reranker_key, _reranker_cache
    _reranker_key = None
    _reranker_cache = None
