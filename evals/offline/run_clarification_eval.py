"""Offline clarification policy evaluation script."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.rag.policies.clarification_policy import needs_clarification
from app.rag.query.query_analyzer import analyze

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = Path(__file__).parent.parent / "datasets" / "clarification_eval.jsonl"


def run() -> None:
    """Evaluate clarification policy accuracy against the JSONL dataset."""
    correct = 0
    total = 0
    with DATASET.open() as f:
        for line in f:
            item = json.loads(line)
            analysis = analyze(item["query"])
            predicted = needs_clarification(analysis)
            expected = item["expected_clarification"]
            match = predicted == expected
            correct += int(match)
            total += 1
            status = "✓" if match else "✗"
            logger.info("%s Query: %s | expected=%s predicted=%s", status, item["query"], expected, predicted)

    print(f"\nAccuracy: {correct}/{total} = {correct/total:.0%}" if total else "No data")


if __name__ == "__main__":
    run()
