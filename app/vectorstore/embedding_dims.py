"""Known embedding model output dimensions (for collection creation)."""

from __future__ import annotations

from app.core.config import settings

# Extend when adding popular FastEmbed / sentence-transformers ids.
KNOWN_EMBEDDING_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


def embedding_vector_size() -> int:
    """Vector dimension for the current embedding model (explicit env or lookup)."""
    if settings.embedding_vector_size > 0:
        return settings.embedding_vector_size
    return KNOWN_EMBEDDING_DIMS.get(settings.embedding_model, 384)
