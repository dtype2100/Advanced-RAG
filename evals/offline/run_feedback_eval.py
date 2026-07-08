"""Offline feedback-loop evaluation against the golden regression set.

Validates routing policy decisions for the feedback phase of the improvement loop.

Usage:
    python evals/offline/run_feedback_eval.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET = Path(__file__).parent.parent / "regression" / "golden_set.yaml"

_BAD_VERDICT = dict(correctness=0.2, faithfulness=0.1, completeness=0.2, conciseness=0.2)
_GOOD_VERDICT = dict(correctness=0.9, faithfulness=0.9, completeness=0.9, conciseness=0.9)


def run() -> None:
    """Evaluate feedback routing against golden_set.yaml expectations."""
    from app.rag.evaluators.llm_judge_evaluator import JudgeVerdict
    from app.rag.policies.clarification_policy import needs_clarification
    from app.rag.policies.rewrite_policy import needs_rewrite
    from app.rag.policies.routing_policy import route_after_feedback
    from app.rag.query.query_analyzer import analyze

    with GOLDEN_SET.open() as f:
        data = yaml.safe_load(f)

    failed = []
    for item in data.get("golden_set", []):
        item_id = item["id"]
        query = item.get("query", "")

        if "expected_clarification" in item:
            analysis = analyze(query)
            predicted = needs_clarification(analysis)
            expected = item["expected_clarification"]
            if predicted != expected:
                failed.append(f"{item_id}: clarification expected={expected} got={predicted}")
                continue

        if item.get("expected_rewrite") is not None:
            predicted_rewrite = needs_rewrite(query, {})
            expected = item["expected_rewrite"]
            if predicted_rewrite != expected:
                failed.append(f"{item_id}: rewrite expected={expected} got={predicted_rewrite}")
                continue

        if "answer_hallucinated" in item:
            verdict = JudgeVerdict(**_BAD_VERDICT)
            state = {
                "judge_verdict": verdict,
                "grounding_score": 0.2,
                "hallucination_attempt": 0,
            }
            action = route_after_feedback(state)
            expected = item.get("expected_judge_action", "retry_retrieval")
            if action not in {expected, "retry_with_policy", "retry_retrieval"}:
                failed.append(f"{item_id}: feedback action expected~={expected} got={action}")

        if "answer_correct" in item:
            verdict = JudgeVerdict(**_GOOD_VERDICT)
            state = {
                "judge_verdict": verdict,
                "grounding_score": 0.9,
                "hallucination_attempt": 0,
            }
            action = route_after_feedback(state)
            if action != "end":
                failed.append(f"{item_id}: feedback action expected=end got={action}")

        logger.info("✓ %s", item_id)

    if failed:
        for msg in failed:
            logger.error("✗ %s", msg)
        raise SystemExit(1)

    logger.info("Feedback golden set: all %d cases passed", len(data.get("golden_set", [])))


if __name__ == "__main__":
    run()
