"""Unit tests for the eval run recorder and history ledger."""

from __future__ import annotations

from pathlib import Path

from evals.recorder import (
    extract_metrics,
    extract_misses,
    format_history_table,
    load_history,
    record_run,
)

_GOLDEN_REPORT = {
    "corpus_sections": 28,
    "chunks_indexed": 28,
    "case_count": 2,
    "retrieval": {
        "overall": {
            "n": 2,
            "hit_at_k": 1.0,
            "recall_at_k": 0.75,
            "mrr": 0.8,
            "ndcg_at_k": 0.7,
            "fact_support": 0.9,
        },
        "by_difficulty": {},
        "by_tag": {},
        "cases": [
            {
                "id": "ok_001",
                "query": "easy question",
                "recall_at_k": 1.0,
                "mrr": 1.0,
                "ndcg_at_k": 1.0,
                "fact_support": 1.0,
                "tags": ["easy"],
            },
            {
                "id": "miss_001",
                "query": "hard question",
                "recall_at_k": 0.5,
                "mrr": 0.5,
                "ndcg_at_k": 0.4,
                "fact_support": 0.5,
                "tags": ["hard"],
            },
        ],
    },
    "policy": {"clarification_accuracy": 1.0, "rewrite_accuracy": 1.0},
}


def test_extract_metrics_and_misses():
    metrics = extract_metrics("golden", _GOLDEN_REPORT)
    assert metrics["recall_at_k"] == 0.75
    assert metrics["clarification_accuracy"] == 1.0
    misses = extract_misses(_GOLDEN_REPORT)
    assert [row["id"] for row in misses] == ["miss_001"]


def test_record_run_appends_history_and_writes_latest(tmp_path: Path):
    first = record_run(
        "golden",
        _GOLDEN_REPORT,
        duration_ms=12,
        note="first",
        results_dir=tmp_path,
    )
    improved = {
        **_GOLDEN_REPORT,
        "retrieval": {
            **_GOLDEN_REPORT["retrieval"],
            "overall": {**_GOLDEN_REPORT["retrieval"]["overall"], "recall_at_k": 0.9},
        },
    }
    second = record_run(
        "golden",
        improved,
        duration_ms=15,
        note="second",
        results_dir=tmp_path,
    )

    history = load_history(tmp_path / "history.jsonl")
    assert len(history) == 2
    assert first["run_id"] != second["run_id"]
    assert second["delta_vs_previous"]["recall_at_k"] == 0.15
    assert (tmp_path / "runs" / first["run_id"] / "report.json").is_file()
    assert (tmp_path / "latest.json").is_file()
    assert "recall_at_k" in (tmp_path / "latest.md").read_text()
    assert "report" not in history[0]
    table = format_history_table(history, kind="golden")
    assert "0.900" in table
    assert "first" in table
