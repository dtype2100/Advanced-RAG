"""Text cleaning utilities applied before chunking."""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalise whitespace and remove common noise patterns from raw text.

    Steps applied:
    1. Collapse multiple blank lines to at most one.
    2. Strip leading/trailing whitespace from each line.
    3. Remove null bytes and other control characters.

    Args:
        text: Raw document text.

    Returns:
        Cleaned text string.
    """
    text = re.sub(r"\x00", "", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
