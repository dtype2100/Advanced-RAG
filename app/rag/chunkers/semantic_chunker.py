"""Semantic chunker — groups sentences by embedding similarity.

Requires: ``pip install langchain-experimental``
"""

from __future__ import annotations


def semantic_chunk(texts: list[str], breakpoint_percentile: int = 95) -> list[dict]:
    """Split texts at semantically significant boundaries.

    Uses LangChain's ``SemanticChunker`` with a percentile-based breakpoint.

    Args:
        texts:                  List of raw document strings.
        breakpoint_percentile:  Percentile threshold for detecting topic shifts.

    Returns:
        List of ``{text, metadata}`` dicts.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:
        raise ImportError("Install: pip install langchain-experimental langchain-openai") from exc

    embeddings = OpenAIEmbeddings()
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=breakpoint_percentile,
    )
    chunks = []
    for text in texts:
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "metadata": {}})
    return chunks
