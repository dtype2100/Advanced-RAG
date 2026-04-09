"""Expansion policy — decides whether to expand child chunks to parent context.

Isolates the expansion decision from the expansion implementation
in ``parent_child_retriever`` and ``small_to_big_retriever``.
"""

from __future__ import annotations

from typing import Any


def should_expand(state: dict[str, Any]) -> bool:
    """Determine whether retrieved child chunks should be expanded.

    Triggers expansion when:
    - Child chunks are present (no expansion if retrieval was empty).
    - The average chunk length is below a threshold, indicating small chunks
      that likely need surrounding context for coherent answers.

    Args:
        state: Current CRAG graph state.

    Returns:
        ``True`` if context expansion should be applied.
    """
    children = state.get("retrieved_children", [])
    if not children:
        return False

    avg_len = sum(len(c) for c in children) / len(children)
    # Expand if average child chunk is shorter than 300 characters
    return avg_len < 300
