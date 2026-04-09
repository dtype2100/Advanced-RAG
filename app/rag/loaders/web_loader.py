"""Web page loader using httpx + BeautifulSoup.

Install: ``pip install httpx beautifulsoup4``
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_url(url: str) -> dict:
    """Fetch a URL and extract its main text content.

    Args:
        url: Target web URL.

    Returns:
        ``{text, metadata}`` dict with the page's visible text.
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError("Install dependencies: pip install httpx beautifulsoup4") from exc

    resp = httpx.get(url, follow_redirects=True, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    logger.info("Loaded %d chars from %s", len(text), url)
    return {"text": text, "metadata": {"source": url}}
