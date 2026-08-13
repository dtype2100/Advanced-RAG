"""Offline LLM-as-judge evaluation script.

Runs the judge evaluator against ``fixtures/judge_eval.jsonl``.  When the LLM
backend is unreachable the script records a skipped run instead of failing.

Usage:
    python evals/offline/run_judge_eval.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.loader import JUDGE_FIXTURE_PATH, load_jsonl  # noqa: E402
from evals.offline.cli import add_record_args  # noqa: E402
from evals.recorder import record_run  # noqa: E402


def run(note: str = "", record: bool = True, results_dir: Path | None = None) -> dict[str, Any]:
    """Evaluate judge quality against canned fixtures, or skip if LLM is down."""
    from app.rag.evaluators.llm_judge_evaluator import judge
    from app.rag.policies.judge_policy import decide_next_action

    started = time.perf_counter()
    items = load_jsonl(JUDGE_FIXTURE_PATH)
    results: list[dict[str, Any]] = []
    skipped = False
    for item in items:
        if item.get("expected_error") and not item.get("context"):
            verdict = judge(question=item["question"], answer=item["answer"], contexts=[])
            action = decide_next_action(verdict)
            logger.info(
                "empty-context fixture %s error=%s action=%s",
                item.get("id"),
                bool(verdict.error),
                action,
            )
            results.append({"id": item.get("id"), "skipped": False, "action": action})
            continue

        verdict = judge(
            question=item["question"],
            answer=item["answer"],
            contexts=[item["context"]] if item.get("context") else [],
        )
        if verdict.error and "JSON parse error" not in verdict.error:
            logger.warning(
                "Judge backend unavailable (%s); skipping remaining items", verdict.error
            )
            skipped = True
            break

        action = decide_next_action(verdict)
        status = "✓" if verdict.passed else "✗"
        logger.info(
            "%s %s overall=%.2f faith=%.2f action=%s",
            status,
            item.get("id", item["question"][:40]),
            verdict.overall_score,
            verdict.faithfulness,
            action,
        )
        results.append(
            {
                "id": item.get("id"),
                "overall": verdict.overall_score,
                "faithfulness": verdict.faithfulness,
                "passed": verdict.passed,
                "action": action,
                "error": verdict.error,
            }
        )

    scored = [row for row in results if "overall" in row]
    duration_ms = int((time.perf_counter() - started) * 1000)
    if skipped and not scored:
        report: dict[str, Any] = {
            "status": "skipped",
            "avg_overall": None,
            "pass_rate": None,
            "scored": 0,
            "results": results,
        }
        print("\nJudge eval skipped: LLM backend is not reachable.")
    elif scored:
        avg_overall = sum(row["overall"] for row in scored) / len(scored)
        pass_rate = sum(1 for row in scored if row["passed"]) / len(scored)
        report = {
            "status": "ok",
            "avg_overall": avg_overall,
            "pass_rate": pass_rate,
            "scored": len(scored),
            "results": results,
        }
        print(f"\nJudge eval summary: avg_overall={avg_overall:.2f}  pass_rate={pass_rate:.0%}")
    else:
        report = {
            "status": "empty",
            "avg_overall": None,
            "pass_rate": None,
            "scored": 0,
            "results": results,
        }

    if record:
        envelope = record_run(
            "judge",
            report,
            duration_ms=duration_ms,
            note=note,
            results_dir=results_dir,
        )
        print(f"Recorded {envelope['run_id']} status={report['status']}")
    return report


if __name__ == "__main__":
    parser = add_record_args(argparse.ArgumentParser(description="Run and record judge eval"))
    args = parser.parse_args()
    run(note=args.note, record=not args.no_record, results_dir=args.results_dir)
