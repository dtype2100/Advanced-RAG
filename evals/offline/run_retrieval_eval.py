"""Offline retrieval quality evaluation script."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from app.rag.evaluators.retrieval_evaluator import evaluate_retrieval
from app.rag.retrievers.retrieval_orchestrator import retrieve

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET = Path(__file__).parent.parent / "datasets" / "retrieval_eval.jsonl"


def run() -> None:
    """Run retrieval evaluation against the JSONL dataset and print results."""
    results_summary = []
    with DATASET.open() as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            results = retrieve(query, top_k=5)
            metrics = evaluate_retrieval(query, results)
            logger.info(
                "Query: %s | Coverage: %.2f | Avg score: %.4f",
                query,
                metrics["coverage_ratio"],
                metrics["avg_score"],
            )
            results_summary.append({"query": query, **metrics})

    avg_coverage = (
        sum(r["coverage_ratio"] for r in results_summary) / len(results_summary)
        if results_summary
        else 0.0
    )
    print(f"\nOverall average coverage: {avg_coverage:.2%}")


if __name__ == "__main__":
    run()
