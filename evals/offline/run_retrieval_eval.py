"""Offline retrieval quality evaluation script."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.offline.harness import format_summary, run_golden_eval  # noqa: E402


def run() -> None:
    """Ingest the frozen corpus and print retrieval ranking metrics."""
    report = run_golden_eval()
    print("\n" + format_summary(report))
    overall = report["retrieval"]["overall"]
    print(
        f"\nOverall recall@5: {overall['recall_at_k']:.2%}  "
        f"nDCG@5: {overall['ndcg_at_k']:.3f}  hit@5: {overall['hit_at_k']:.2%}"
    )


if __name__ == "__main__":
    run()
