"""CLI script — ingest documents from a JSONL file into the vector store.

Usage:
    python scripts/ingest.py --input data/raw/docs.jsonl
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
    """Parse CLI args and run the ingest pipeline."""
    parser = argparse.ArgumentParser(description="Ingest documents into the vector store")
    parser.add_argument(
        "--input", required=True, help="Path to JSONL file with {text, metadata} records"
    )
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

    from app.services.ingest_service import ingest_documents

    count = ingest_documents(docs)
    logger.info("Ingested %d chunks from %s", count, input_path)


if __name__ == "__main__":
    main()
