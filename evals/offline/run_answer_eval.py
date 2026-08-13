"""Offline answer-support evaluation script.

Scores whether retrieved context contains each ``must_cite_facts`` string.
This does not call an LLM; it measures whether generation *could* be grounded.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.offline.harness import run_golden_eval  # noqa: E402


def run() -> None:
    """Print per-case fact-support scores from the golden retrieval run."""
    report = run_golden_eval()
    rows = report["retrieval"]["cases"]
    for row in rows:
        logging.info(
            "Question: %s | fact_support: %.2f | recall@5: %.2f",
            row["query"],
            row["fact_support"],
            row["recall_at_k"],
        )
    overall = report["retrieval"]["overall"]
    print(f"\nOverall fact support: {overall['fact_support']:.2%}  n={overall['n']}")


if __name__ == "__main__":
    run()
