"""Hallucination evaluator — detects claims not supported by the retrieved context.

Returns a risk score in [0, 1]; higher scores indicate more hallucination risk.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a hallucination detector. Given a context and a generated answer, "
    "output a single float between 0.0 and 1.0 representing the probability that "
    "the answer contains claims NOT supported by the context. "
    "0.0 = fully supported, 1.0 = highly hallucinated. "
    "Output ONLY the float number."
)


def score_hallucination(answer: str, contexts: list[str]) -> float:
    """Compute a hallucination risk score for the given answer.

    Args:
        answer:   LLM-generated answer to evaluate.
        contexts: Retrieved context chunks used to generate the answer.

    Returns:
        Float hallucination risk score in [0.0, 1.0].
    """
    if not answer or not contexts:
        return 1.0

    context_str = "\n\n---\n\n".join(contexts[:5])
    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(content=f"Context:\n{context_str}\n\nAnswer: {answer}"),
            ]
        )
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        logger.warning("Hallucination evaluator failed; defaulting to 0.5", exc_info=True)
        return 0.5
