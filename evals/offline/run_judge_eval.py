"""Offline LLM-as-judge evaluation script.

Runs the judge evaluator against ``judge_eval.jsonl`` and reports per-item
verdicts plus summary statistics.

Usage:
    python evals/offline/run_judge_eval.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = Path(__file__).parent.parent / "datasets" / "judge_eval.jsonl"


def run() -> None:
    """Evaluate judge quality against the JSONL dataset and print results."""
    from app.rag.evaluators.llm_judge_evaluator import judge

    results = []
    with DATASET.open() as f:
        for line in f:
            item = json.loads(line)
            verdict = judge(
                question=item["question"],
                answer=item["answer"],
                contexts=[item["context"]],
            )
            status = "✓" if verdict.passed else "✗"
            logger.info(
                "%s Q: %s | overall=%.2f faith=%.2f",
                status,
                item["question"][:60],
                verdict.overall_score,
                verdict.faithfulness,
            )
            if verdict.error:
                logger.warning("  Judge error: %s", verdict.error)
            results.append(
                {
                    "question": item["question"],
                    "overall": verdict.overall_score,
                    "faithfulness": verdict.faithfulness,
                    "passed": verdict.passed,
                }
            )

    if results:
        avg_overall = sum(r["overall"] for r in results) / len(results)
        pass_rate = sum(1 for r in results if r["passed"]) / len(results)
        print(f"\nJudge eval summary: avg_overall={avg_overall:.2f}  pass_rate={pass_rate:.0%}")


if __name__ == "__main__":
    run()
