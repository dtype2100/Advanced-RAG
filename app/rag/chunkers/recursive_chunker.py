"""Recursive character-based text chunker."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk(
    texts: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
    """Split texts using LangChain's ``RecursiveCharacterTextSplitter``.

    Args:
        texts:         List of raw document strings.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List of ``{text, metadata}`` dicts for each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for text in texts:
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "metadata": {}})
    return chunks
