"""Grounding evaluator — checks how well the answer is supported by context.

Uses an LLM judge to produce a score in [0, 1].
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a grounding evaluator. Given a question, a set of context documents, "
    "and a generated answer, output a single float between 0.0 and 1.0 indicating "
    "how well the answer is supported by the context. "
    "1.0 = fully grounded, 0.0 = not grounded at all. "
    "Output ONLY the float number."
)


def evaluate(answer: str, contexts: list[str], question: str = "") -> float:
    """Compute a grounding score for the answer relative to the retrieved context.

    Args:
        answer:   LLM-generated answer string.
        contexts: List of retrieved context strings.
        question: Optional original question for context.

    Returns:
        Float grounding score in [0.0, 1.0].
    """
    if not answer or not contexts:
        return 0.0

    context_str = "\n\n---\n\n".join(contexts[:5])
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=(f"Question: {question}\n\nContext:\n{context_str}\n\nAnswer: {answer}")
                ),
            ]
        )
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        logger.warning("Grounding evaluator failed; defaulting to 0.5", exc_info=True)
        return 0.5
