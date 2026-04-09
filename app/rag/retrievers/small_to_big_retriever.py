"""Small-to-big retriever — expands small matched chunks to surrounding context.

Unlike parent-child (which uses a pre-built docstore), this approach
reconstructs larger windows from a flat list of chunks using their position.
"""

from __future__ import annotations

from typing import Any


def small_to_big_expand(
    matched_chunks: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
    window: int = 2,
) -> list[dict[str, Any]]:
    """Expand each matched chunk by including ``window`` neighbours on each side.

    Args:
        matched_chunks: Vector-retrieved small chunks.
        all_chunks:     Full ordered chunk list from the same document.
        window:         Number of adjacent chunks to include on each side.

    Returns:
        List of expanded context dicts with merged text and updated metadata.
    """
    all_texts = [c["text"] for c in all_chunks]
    expanded: list[dict[str, Any]] = []
    seen_indices: set[int] = set()

    for match in matched_chunks:
        try:
            idx = all_texts.index(match["text"])
        except ValueError:
            expanded.append(match)
            continue

        start = max(0, idx - window)
        end = min(len(all_chunks), idx + window + 1)
        window_indices = list(range(start, end))

        if any(i in seen_indices for i in window_indices):
            continue

        seen_indices.update(window_indices)
        merged_text = "\n".join(all_chunks[i]["text"] for i in window_indices)
        expanded.append(
            {
                "text": merged_text,
                "score": match.get("score", 0.0),
                "metadata": {**match.get("metadata", {}), "expanded": "true"},
            }
        )

    return expanded
