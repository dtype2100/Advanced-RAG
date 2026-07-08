"""Chat service — entry point for the CRAG pipeline from the API layer."""

from __future__ import annotations

import logging
from typing import Any

from app.graphs.crag.graph import crag_chain
from app.rag.guards.policy_guard import is_allowed

logger = logging.getLogger(__name__)


def _build_initial_state(
    question: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Build the initial CRAG graph state for a user question."""
    state: dict[str, Any] = {
        "user_query": question,
        "retrieval_attempt": 0,
        "hallucination_attempt": 0,
    }
    if top_k is not None:
        state["top_k"] = top_k
    return state


def run_chat(
    question: str,
    session_id: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Run the full CRAG pipeline for a user question.

    Args:
        question:   Raw user question.
        session_id: Optional chat session identifier for history tracking.
        top_k:      Override for number of documents to retrieve.

    Returns:
        Result dict with ``answer``, ``final_status``, and graph state fields.
    """
    if not is_allowed(question):
        return {
            "answer": "I'm sorry, but I can't help with that request.",
            "final_status": "blocked",
        }

    initial_state = _build_initial_state(question, top_k=top_k)

    try:
        result = crag_chain.invoke(initial_state)
        logger.info("Chat completed, status=%s", result.get("final_status", "ok"))
        return result
    except Exception:
        logger.exception("Chat service error for question: %s", question[:80])
        raise


def stream_chat(
    question: str,
    session_id: str | None = None,
    top_k: int | None = None,
):
    """Yield CRAG graph node outputs as they complete (for SSE streaming).

    Args:
        question:   Raw user question.
        session_id: Optional chat session identifier for history tracking.
        top_k:      Override for number of documents to retrieve.

    Yields:
        Dicts mapping node names to partial state updates.
    """
    if not is_allowed(question):
        yield {
            "blocked": {
                "answer": "I'm sorry, but I can't help with that request.",
                "final_status": "blocked",
            }
        }
        return

    yield from crag_chain.stream(_build_initial_state(question, top_k=top_k))
