"""Query rewriter — reformulates a query for better vector-store retrieval."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.core.llm_io_log import log_llm_io
from app.providers.llm_provider import get_llm

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a query rewriter. Reformulate the user question to improve retrieval "
    "from a vector database. Make it more specific, add relevant keywords, and remove "
    "ambiguity while preserving the original intent.\n"
    "Output ONLY the rewritten question — no explanation."
)


def rewrite(query: str) -> str:
    """Rewrite ``query`` for improved semantic retrieval.

    Args:
        query: Current query string (original or previously rewritten).

    Returns:
        Rewritten query string.
    """
    user_text = f"Original question: {query}\n\nRewrite this question:"
    log_llm_io("query_rewrite", user_query=query, system=_SYSTEM, user=user_text)
    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_text),
        ]
    )
    rewritten = response.content.strip()
    log_llm_io("query_rewrite", assistant=rewritten)
    logger.info("Rewriter: '%s' → '%s'", query, rewritten)
    return rewritten
