"""Chat service — entry point for the CRAG pipeline from the API layer."""

from __future__ import annotations

import logging
import time
from typing import Any

from app.core.metrics import record_rag_query
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
        record_rag_query(status="blocked", duration_seconds=0.0)
        return {
            "answer": "I'm sorry, but I can't help with that request.",
            "final_status": "blocked",
        }

    initial_state = {
        "user_query": question,
        "retrieval_attempt": 0,
        "hallucination_attempt": 0,
    }

    start = time.perf_counter()
    try:
        result = crag_chain.invoke(initial_state)
        status = str(result.get("final_status", "ok"))
        record_rag_query(status=status, duration_seconds=time.perf_counter() - start)
        logger.info("Chat completed, status=%s", status)
        return result
    except Exception:
        record_rag_query(status="error", duration_seconds=time.perf_counter() - start)
        logger.exception("Chat service error for question: %s", question[:80])
        raise
