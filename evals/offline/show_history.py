"""Print the eval history ledger.

Usage:
    python evals/offline/show_history.py
    python evals/offline/show_history.py --kind golden
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.recorder import HISTORY_PATH, format_history_table, load_history  # noqa: E402


def run(kind: str | None = None, history_path: Path | None = None) -> str:
    """Load ``history.jsonl`` and return a printable table."""
    records = load_history(history_path or HISTORY_PATH)
    return format_history_table(records, kind=kind)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show recorded eval runs")
    parser.add_argument("--kind", default=None, help="Filter by kind (golden, judge, policy)")
    args = parser.parse_args()
    print(run(kind=args.kind))
