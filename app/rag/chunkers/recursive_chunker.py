"""Recursive character-based text chunker."""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk(
    docs: list[str] | list[dict[str, Any]],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict[str, Any]]:
    """Split documents using LangChain's ``RecursiveCharacterTextSplitter``.

    Args:
        docs:          Either raw text strings or ``{text, metadata}`` dicts.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap between adjacent chunks.

    Returns:
        List of ``{text, metadata}`` dicts for each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[dict[str, Any]] = []
    for doc_idx, item in enumerate(docs):
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            base_meta = dict(item.get("metadata", {}))
        else:
            text = str(item)
            base_meta = {}

        if not text:
            continue

        split_chunks = splitter.split_text(text)
        for chunk_idx, chunk in enumerate(split_chunks):
            meta = {
                **base_meta,
                "doc_index": base_meta.get("doc_index", doc_idx),
                "chunk_index": chunk_idx,
            }
            chunks.append({"text": chunk, "metadata": meta})
    return chunks
