"""Offline golden-set evaluation: retrieval qrels, fact support, and policies.

Usage:
    python evals/offline/run_golden_eval.py
    python evals/offline/run_golden_eval.py --note "after hybrid retrieval"
    python evals/offline/run_golden_eval.py --no-record
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.offline.cli import add_record_args  # noqa: E402
from evals.offline.harness import format_summary, run_golden_eval  # noqa: E402
from evals.recorder import format_run_markdown, record_run  # noqa: E402


def run(
    note: str = "",
    record: bool = True,
    results_dir: Path | None = None,
) -> dict:
    """Run the golden eval, print a summary, and optionally persist a ledger row."""
    started = time.perf_counter()
    report = run_golden_eval()
    duration_ms = int((time.perf_counter() - started) * 1000)
    print("\n" + format_summary(report))

    if not record:
        return report

    envelope = record_run(
        "golden",
        report,
        duration_ms=duration_ms,
        note=note,
        results_dir=results_dir,
    )
    print(f"\nRecorded {envelope['run_id']}")
    print(format_run_markdown(envelope))

    artifacts = Path("/opt/cursor/artifacts")
    if artifacts.is_dir():
        artifacts.joinpath("golden_eval_latest.md").write_text(
            format_run_markdown(envelope), encoding="utf-8"
        )
        from evals.offline.harness import write_report

        write_report(envelope, artifacts / "golden_eval_latest.json")
    return report


if __name__ == "__main__":
    parser = add_record_args(argparse.ArgumentParser(description="Run and record golden eval"))
    args = parser.parse_args()
    run(note=args.note, record=not args.no_record, results_dir=args.results_dir)
