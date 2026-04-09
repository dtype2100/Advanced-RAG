"""Multi-query generator — produces N paraphrased variants of a query.

Running parallel retrieval on multiple reformulations and fusing the results
typically improves recall, especially for complex or under-specified queries.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a query diversification assistant. Given a user question, generate "
    "{n} distinct paraphrased versions that preserve the original intent but use "
    "different wording, synonyms, or perspectives. This helps improve document "
    "retrieval coverage.\n"
    "Output each variant on its own line — no numbering, no extra explanation."
)


def generate_multi_query(query: str, n: int = 3) -> list[str]:
    """Generate ``n`` paraphrased variants of ``query`` for parallel retrieval.

    Args:
        query: Original user query.
        n:     Number of query variants to generate (default: 3).

    Returns:
        List of query strings including the original as the first element.
        If LLM generation fails, returns ``[query]`` (graceful degradation).
    """
    if n <= 0:
        return [query]

    try:
        llm = get_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM.format(n=n)),
                HumanMessage(content=f"Original question: {query}"),
            ]
        )
        variants = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
        unique_variants = list(dict.fromkeys(variants))[:n]
        result = [query, *unique_variants]
        logger.info("Multi-query: generated %d variants for '%s'", len(result), query[:60])
        return result
    except Exception:
        logger.warning("Multi-query generation failed; falling back to single query", exc_info=True)
        return [query]
