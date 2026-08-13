"""CLI helpers shared by offline eval runners."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_record_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach ``--note`` / ``--no-record`` flags used by every eval runner."""
    parser.add_argument(
        "--note",
        default="",
        help="Optional note stored on the history ledger row",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run the eval without writing evals/results/",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Override evals/results (used in tests)",
    )
    return parser
