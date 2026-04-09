"""CLI script — run all offline evaluations.

Usage:
    python scripts/run_evals.py
"""

from __future__ import annotations

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_SCRIPTS = [
    "evals/offline/run_clarification_eval.py",
    "evals/offline/run_retrieval_eval.py",
    "evals/offline/run_answer_eval.py",
]


def main() -> None:
    """Run each eval script in sequence and report pass/fail."""
    failed = []
    for script in EVAL_SCRIPTS:
        logger.info("Running %s ...", script)
        result = subprocess.run([sys.executable, script], capture_output=False)
        if result.returncode != 0:
            failed.append(script)

    if failed:
        logger.error("Failed evals: %s", failed)
        sys.exit(1)
    logger.info("All evals passed.")


if __name__ == "__main__":
    main()
