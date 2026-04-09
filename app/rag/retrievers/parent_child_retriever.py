"""Parent-child retriever — retrieves child chunks then fetches parent context.

After vector search returns small child chunks, this retriever looks up the
corresponding parent documents from the docstore for richer context.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def fetch_parents(
    child_results: list[dict[str, Any]],
    docstore: dict[str, str],
) -> list[dict[str, Any]]:
    """Replace child chunks with their parent documents from the docstore.

    Args:
        child_results: Retrieval results from the vector store (child chunks).
        docstore:      Dict mapping ``parent_id`` → parent text.

    Returns:
        List of result dicts where ``text`` is replaced by the parent chunk.
        Children without a ``parent_id`` are returned as-is.
    """
    expanded: list[dict[str, Any]] = []
    seen_parent_ids: set[str] = set()

    for result in child_results:
        parent_id = result.get("metadata", {}).get("parent_id")
        if parent_id and parent_id in docstore:
            if parent_id not in seen_parent_ids:
                seen_parent_ids.add(parent_id)
                expanded.append(
                    {
                        "text": docstore[parent_id],
                        "score": result["score"],
                        "metadata": {**result.get("metadata", {}), "expanded": "true"},
                    }
                )
            logger.debug("Expanded child → parent: %s", parent_id)
        else:
            expanded.append(result)

    return expanded
