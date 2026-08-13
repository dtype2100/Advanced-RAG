"""Offline clarification and rewrite policy evaluation script."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.loader import load_cases  # noqa: E402
from evals.offline.harness import evaluate_policies  # noqa: E402


def run() -> None:
    """Evaluate clarification and rewrite policy accuracy against golden labels."""
    cases = load_cases()
    report = evaluate_policies(cases)
    for row in report["clarification"]:
        status = "✓" if row["match"] else "✗"
        logger.info(
            "%s clarify %s expected=%s predicted=%s",
            status,
            row["id"],
            row["expected"],
            row["predicted"],
        )
    for row in report["rewrite"]:
        if not row["match"]:
            logger.info(
                "✗ rewrite %s expected=%s predicted=%s",
                row["id"],
                row["expected"],
                row["predicted"],
            )
    print(
        f"\nClarification accuracy: {report['clarification_accuracy']:.0%}  "
        f"Rewrite accuracy: {report['rewrite_accuracy']:.0%}  n={len(cases)}"
    )
    if report["clarification_accuracy"] < 1.0 or report["rewrite_accuracy"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
