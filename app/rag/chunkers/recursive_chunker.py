"""Recursive character-based text chunker."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk(
    texts: list[str],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    metadatas: list[dict] | None = None,
) -> list[dict]:
    """Split texts using LangChain's ``RecursiveCharacterTextSplitter``.

    Parent-document metadata is copied onto every child chunk so source labels
    (``source_id``, ``section_id``) survive ingestion and can be used as qrels.

    Args:
        texts:         List of raw document strings.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Character overlap between adjacent chunks.
        metadatas:     Optional per-document metadata aligned with ``texts``.
                       When omitted, each chunk gets an empty metadata dict.

    Returns:
        List of ``{text, metadata}`` dicts for each chunk.
    """
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must be the same length as texts")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for index, text in enumerate(texts):
        base_meta = dict(metadatas[index]) if metadatas is not None else {}
        for chunk in splitter.split_text(text):
            chunks.append({"text": chunk, "metadata": dict(base_meta)})
    return chunks
