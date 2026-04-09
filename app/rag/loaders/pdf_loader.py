"""PDF document loader using PyMuPDF (fitz).

Install: ``pip install pymupdf``
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        List of ``{text, metadata}`` dicts, one per page.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("Install pymupdf: pip install pymupdf") from exc

    path = Path(path)
    doc = fitz.open(str(path))
    pages = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"text": text, "metadata": {"source": path.name, "page": str(page_num)}})
    logger.info("Loaded %d pages from %s", len(pages), path.name)
    return pages
