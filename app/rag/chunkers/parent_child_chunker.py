"""Parent-child chunker — creates large parent chunks and small child chunks.

The parent chunks are stored in the docstore for context expansion.
The child chunks are indexed in the vector store for precise retrieval.
"""

from __future__ import annotations

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter


def parent_child_chunk(
    texts: list[str],
    parent_chunk_size: int = 1024,
    child_chunk_size: int = 256,
    chunk_overlap: int = 32,
) -> tuple[list[dict], list[dict]]:
    """Split texts into (parent, child) chunk pairs.

    Args:
        texts:             Source document strings.
        parent_chunk_size: Characters per parent chunk.
        child_chunk_size:  Characters per child chunk.
        chunk_overlap:     Overlap applied to both splitters.

    Returns:
        Tuple ``(parents, children)`` where each element is a list of
        ``{text, metadata}`` dicts.  Child dicts include a ``parent_id``
        metadata key linking back to their parent.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=chunk_overlap,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=chunk_overlap,
    )

    parents: list[dict] = []
    children: list[dict] = []

    for text in texts:
        for parent_text in parent_splitter.split_text(text):
            parent_id = str(uuid.uuid4())
            parents.append({"text": parent_text, "metadata": {"chunk_id": parent_id}})
            for child_text in child_splitter.split_text(parent_text):
                children.append({"text": child_text, "metadata": {"parent_id": parent_id}})

    return parents, children
