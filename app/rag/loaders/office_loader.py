"""Office document loader (DOCX, PPTX) — stub for future implementation.

Install: ``pip install python-docx python-pptx``
"""

from __future__ import annotations

from pathlib import Path


def load_docx(path: str | Path) -> list[dict]:
    """Extract paragraphs from a DOCX file (not yet implemented).

    Raises:
        NotImplementedError: Until python-docx integration is complete.
    """
    raise NotImplementedError("DOCX loading is not yet implemented")


def load_pptx(path: str | Path) -> list[dict]:
    """Extract slide text from a PPTX file (not yet implemented).

    Raises:
        NotImplementedError: Until python-pptx integration is complete.
    """
    raise NotImplementedError("PPTX loading is not yet implemented")
