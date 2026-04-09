"""Project-wide constants."""

from __future__ import annotations

# ── Vector dimension map ──────────────────────────────────────────────────────
VECTOR_SIZE_MAP: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

# ── Hallucination feedback loop ───────────────────────────────────────────────
MAX_HALLUCINATION_RETRIES = 3
