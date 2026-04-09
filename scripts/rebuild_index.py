"""CLI script — drop and rebuild the vector store index from a JSONL file.

Usage:
    python scripts/rebuild_index.py --input data/processed/docs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Parse CLI args and rebuild the index."""
    parser = argparse.ArgumentParser(description="Rebuild the vector store index")
    parser.add_argument("--input", required=True, help="Path to JSONL source file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    docs = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    from app.services.index_service import rebuild_index

    count = rebuild_index(docs)
    logger.info("Index rebuilt: %d chunks indexed", count)


if __name__ == "__main__":
    main()
