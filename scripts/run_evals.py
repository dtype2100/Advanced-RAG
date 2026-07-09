"""CLI script — run the full improvement loop (analysis → feedback).

Usage:
    python scripts/run_evals.py           # full loop (needs LLM for judge phase)
    python scripts/run_evals.py --ci    # CI-safe subset (no LLM)
    IMPROVEMENT_LOOP_CI=1 make evals-ci
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

from app.core.improvement_loop import iter_eval_phases

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_pytest(target: str) -> int:
    """Run pytest against a test directory or file."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "--tb=short"],
        capture_output=False,
    ).returncode


def _run_script(script: str) -> int:
    """Run a Python eval script."""
    return subprocess.run([sys.executable, script], capture_output=False).returncode


def main() -> None:
    """Run each improvement-loop phase in canonical order."""
    parser = argparse.ArgumentParser(description="Run the Advanced-RAG improvement loop")
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Skip LLM-dependent phases (safe for GitHub Actions without vLLM)",
    )
    args = parser.parse_args()

    phases = iter_eval_phases(ci=args.ci)
    if args.ci:
        logger.info("CI mode: skipping LLM-dependent phases")

    failed: list[str] = []

    for phase in phases:
        logger.info("=== Phase [%s] %s ===", phase.name, phase.label)

        if phase.name == "verification" and phase.eval_target is None:
            logger.info(
                "Skipping runtime-only verification phase (covered by graph integration tests)"
            )
            continue

        target = phase.eval_target
        if target is None:
            continue

        code = _run_pytest(target) if target.startswith("tests/") else _run_script(target)

        if code != 0:
            failed.append(f"{phase.name}:{target}")

    if failed:
        logger.error("Failed phases: %s", failed)
        sys.exit(1)

    logger.info("Improvement loop complete — all phases passed.")


if __name__ == "__main__":
    main()
