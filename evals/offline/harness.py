"""Shared helpers for offline golden-set evaluation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from app.core.constants import DEFAULT_TOP_K
from app.rag.evaluators.retrieval_evaluator import evaluate_retrieval
from app.rag.policies.clarification_policy import needs_clarification
from app.rag.policies.rewrite_policy import needs_rewrite
from app.rag.query.query_analyzer import analyze
from app.rag.retrievers.retrieval_orchestrator import retrieve
from app.services.ingest_service import ingest_documents
from evals.loader import (
    fact_support_score,
    load_cases,
    load_corpus,
    retrieval_cases,
)

logger = logging.getLogger(__name__)


def ingest_eval_corpus() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Load the frozen corpus, ingest it, and return (docs, cases, chunk_count)."""
    docs = load_corpus()
    cases = load_cases(corpus=docs)
    count = ingest_documents(docs)
    logger.info("Ingested eval corpus: %d source sections -> %d chunks", len(docs), count)
    return docs, cases, count


def _mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def evaluate_policies(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Score clarification and rewrite policies against golden labels."""
    clarification_rows = []
    rewrite_rows = []
    for case in cases:
        analysis = analyze(case["query"])
        predicted_clarify = needs_clarification(analysis)
        expected_clarify = bool(case["expected_clarification"])
        clarification_rows.append(
            {
                "id": case["id"],
                "match": predicted_clarify == expected_clarify,
                "expected": expected_clarify,
                "predicted": predicted_clarify,
                "expected_slot": case.get("expected_slot"),
                "missing_slots": analysis.get("missing_slots", []),
            }
        )
        predicted_rewrite = needs_rewrite(case["query"])
        expected_rewrite = bool(case["expected_rewrite"])
        rewrite_rows.append(
            {
                "id": case["id"],
                "match": predicted_rewrite == expected_rewrite,
                "expected": expected_rewrite,
                "predicted": predicted_rewrite,
            }
        )

    def _accuracy(rows: list[dict[str, Any]]) -> float:
        return sum(1 for row in rows if row["match"]) / len(rows) if rows else 0.0

    return {
        "clarification_accuracy": _accuracy(clarification_rows),
        "rewrite_accuracy": _accuracy(rewrite_rows),
        "clarification": clarification_rows,
        "rewrite": rewrite_rows,
    }


def evaluate_retrieval_cases(
    cases: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Run the retriever on labelled cases and aggregate ranking + fact metrics."""
    rows = []
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for case in retrieval_cases(cases):
        results = retrieve(case["query"], top_k=top_k)
        metrics = evaluate_retrieval(results, case["qrels"], k=top_k)
        contexts = [item.get("text", "") for item in results]
        fact_score = fact_support_score(case.get("must_cite_facts") or [], contexts)
        row = {
            "id": case["id"],
            "query": case["query"],
            "intent": case["intent"],
            "difficulty": case["difficulty"],
            "tags": case["tags"],
            "fact_support": fact_score,
            **metrics,
        }
        rows.append(row)
        for tag in case["tags"]:
            by_tag[tag].append(row)

    def _summarise(group: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "n": len(group),
            "hit_at_k": _mean([row["hit_at_k"] for row in group]),
            "recall_at_k": _mean([row["recall_at_k"] for row in group]),
            "mrr": _mean([row["mrr"] for row in group]),
            "ndcg_at_k": _mean([row["ndcg_at_k"] for row in group]),
            "fact_support": _mean([row["fact_support"] for row in group]),
        }

    tag_summary = {tag: _summarise(group) for tag, group in sorted(by_tag.items())}
    difficulty_summary = {
        level: _summarise([row for row in rows if row["difficulty"] == level])
        for level in ("easy", "medium", "hard")
        if any(row["difficulty"] == level for row in rows)
    }
    return {
        "overall": _summarise(rows),
        "by_tag": tag_summary,
        "by_difficulty": difficulty_summary,
        "cases": rows,
    }


def run_golden_eval(top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
    """Ingest the frozen corpus and evaluate retrieval, fact support, and policies."""
    docs, cases, chunk_count = ingest_eval_corpus()
    retrieval_report = evaluate_retrieval_cases(cases, top_k=top_k)
    policy_report = evaluate_policies(cases)
    return {
        "corpus_sections": len(docs),
        "chunks_indexed": chunk_count,
        "case_count": len(cases),
        "retrieval": retrieval_report,
        "policy": {
            "clarification_accuracy": policy_report["clarification_accuracy"],
            "rewrite_accuracy": policy_report["rewrite_accuracy"],
            "clarification_failures": [
                row for row in policy_report["clarification"] if not row["match"]
            ],
            "rewrite_failures": [row for row in policy_report["rewrite"] if not row["match"]],
        },
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    """Write a JSON report, omitting the bulky per-case dump's query text if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Wrote eval report to %s", path)


def format_summary(report: dict[str, Any]) -> str:
    """Render a compact human-readable summary of golden-eval metrics."""
    overall = report["retrieval"]["overall"]
    policy = report["policy"]
    lines = [
        "Golden eval summary",
        f"  corpus sections={report['corpus_sections']}  chunks={report['chunks_indexed']}"
        f"  cases={report['case_count']}",
        (
            f"  retrieval n={overall['n']}  hit@5={overall['hit_at_k']:.2%}  "
            f"recall@5={overall['recall_at_k']:.2%}  mrr={overall['mrr']:.3f}  "
            f"ndcg@5={overall['ndcg_at_k']:.3f}  fact_support={overall['fact_support']:.2%}"
        ),
        (
            f"  policy  clarification={policy['clarification_accuracy']:.0%}  "
            f"rewrite={policy['rewrite_accuracy']:.0%}"
        ),
        "  by difficulty:",
    ]
    for level, stats in report["retrieval"]["by_difficulty"].items():
        lines.append(
            f"    {level:6} n={stats['n']:2d}  hit@5={stats['hit_at_k']:.2%}  "
            f"recall@5={stats['recall_at_k']:.2%}  ndcg@5={stats['ndcg_at_k']:.3f}"
        )
    return "\n".join(lines)
