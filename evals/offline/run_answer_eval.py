"""Offline answer quality evaluation script."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.rag.evaluators.grounding_evaluator import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = Path(__file__).parent.parent / "datasets" / "answer_eval.jsonl"


def run() -> None:
    """Evaluate grounding scores for reference answers in the dataset."""
    with DATASET.open() as f:
        for line in f:
            item = json.loads(line)
            score = evaluate(
                answer=item["reference"],
                contexts=[item["context"]],
                question=item["question"],
            )
            logger.info("Question: %s | Grounding score: %.2f", item["question"], score)


if __name__ == "__main__":
    run()
