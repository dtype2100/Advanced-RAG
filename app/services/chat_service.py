"""Chat service — entry point for the CRAG pipeline from the API layer."""

from __future__ import annotations

import logging
from typing import Any

from app.graphs.crag.graph import crag_chain
from app.rag.guards.policy_guard import is_allowed

logger = logging.getLogger(__name__)


def run_chat(
    question: str,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run the full CRAG pipeline for a user question.

    Args:
        question:   Raw user question.
        session_id: Optional chat session identifier for history tracking.

    Returns:
        Result dict with ``answer``, ``final_status``, and graph state fields.
    """
    if not is_allowed(question):
        return {
            "answer": "I'm sorry, but I can't help with that request.",
            "final_status": "blocked",
        }

    initial_state = {
        "user_query": question,
        "retrieval_attempt": 0,
        "hallucination_attempt": 0,
    }

    try:
        result = crag_chain.invoke(initial_state)
        logger.info("Chat completed, status=%s", result.get("final_status", "ok"))
        return result
    except Exception:
        logger.exception("Chat service error for question: %s", question[:80])
        raise
