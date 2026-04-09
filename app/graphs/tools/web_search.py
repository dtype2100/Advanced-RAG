"""Web search tool for fallback retrieval when the vector store is insufficient.

Placeholder: wire up Tavily, SerpAPI, or another search API here.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def web_search(query: str, num_results: int = 3) -> list[dict]:
    """Perform a web search and return structured results.

    Args:
        query: Search query string.
        num_results: Maximum number of web results to return.

    Returns:
        List of ``{title, url, snippet}`` dicts.

    Raises:
        NotImplementedError: Until a search API is configured.
    """
    raise NotImplementedError(
        "Web search is not yet configured. Set SEARCH_BACKEND env var and implement this tool."
    )
