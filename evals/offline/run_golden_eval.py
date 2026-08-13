"""Offline golden-set evaluation: retrieval qrels, fact support, and policies.

Usage:
    python evals/offline/run_golden_eval.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Allow `python evals/offline/run_golden_eval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.offline.harness import format_summary, run_golden_eval, write_report  # noqa: E402


def run() -> dict:
    """Run the golden eval, print a summary, and persist a JSON report."""
    report = run_golden_eval()
    print("\n" + format_summary(report))

    repo_report = Path(__file__).resolve().parents[1] / "results" / "golden_eval.json"
    write_report(report, repo_report)
    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        write_report(report, artifacts / "golden_eval.json")
    return report


if __name__ == "__main__":
    run()
