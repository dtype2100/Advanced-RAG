"""Persist offline eval runs as timestamped dumps plus an append-only ledger.

Each run writes:
- ``results/runs/<run_id>/report.json`` — full payload including per-case rows
- ``results/runs/<run_id>/summary.md`` — human-readable snapshot
- ``results/history.jsonl`` — one headline record per run (git-tracked)
- ``results/latest.json`` / ``latest.md`` — most recent golden (or last) snapshot
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evals.loader import CASES_PATH, CORPUS_PATH, EVALS_ROOT

logger = logging.getLogger(__name__)

RESULTS_DIR = EVALS_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
HISTORY_PATH = RESULTS_DIR / "history.jsonl"
LATEST_JSON = RESULTS_DIR / "latest.json"
LATEST_MD = RESULTS_DIR / "latest.md"

_HEADLINE_KEYS = (
    "hit_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "fact_support",
    "clarification_accuracy",
    "rewrite_accuracy",
    "judge_pass_rate",
    "judge_avg_overall",
)


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file, or empty string if it is missing."""
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_info() -> dict[str, Any]:
    """Capture branch, SHA, and dirty flag; never raises on git failures."""

    def _run(args: list[str]) -> str:
        result = subprocess.run(
            args,
            cwd=EVALS_ROOT.parent,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    sha = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_run(["git", "status", "--porcelain"]))
    return {
        "sha": sha,
        "short_sha": sha[:7] if sha else "",
        "branch": branch,
        "dirty": dirty,
    }


def _config_snapshot() -> dict[str, Any]:
    """Snapshot retrieval/LLM settings that can change eval numbers."""
    from app.core.config import settings

    return {
        "embedding_model": settings.embedding_model,
        "collection_name": settings.collection_name,
        "max_retrieval_docs": settings.max_retrieval_docs,
        "llm_backend": settings.llm_backend,
        "llm_model": settings.llm_model,
        "qdrant_in_memory": settings.qdrant_in_memory,
    }


def _dataset_snapshot() -> dict[str, Any]:
    """Fingerprint the frozen corpus and golden cases used for this run."""
    return {
        "corpus_path": str(CORPUS_PATH.relative_to(EVALS_ROOT.parent)),
        "cases_path": str(CASES_PATH.relative_to(EVALS_ROOT.parent)),
        "corpus_sha256": _sha256_file(CORPUS_PATH),
        "cases_sha256": _sha256_file(CASES_PATH),
    }


def extract_metrics(kind: str, report: dict[str, Any]) -> dict[str, Any]:
    """Pull comparable headline numbers out of a kind-specific report."""
    if kind == "golden":
        overall = report.get("retrieval", {}).get("overall", {})
        policy = report.get("policy", {})
        return {
            "hit_at_k": overall.get("hit_at_k"),
            "recall_at_k": overall.get("recall_at_k"),
            "mrr": overall.get("mrr"),
            "ndcg_at_k": overall.get("ndcg_at_k"),
            "fact_support": overall.get("fact_support"),
            "retrieval_n": overall.get("n"),
            "clarification_accuracy": policy.get("clarification_accuracy"),
            "rewrite_accuracy": policy.get("rewrite_accuracy"),
        }
    if kind == "judge":
        return {
            "judge_status": report.get("status", "unknown"),
            "judge_avg_overall": report.get("avg_overall"),
            "judge_pass_rate": report.get("pass_rate"),
            "judge_scored": report.get("scored", 0),
        }
    if kind == "policy":
        return {
            "clarification_accuracy": report.get("clarification_accuracy"),
            "rewrite_accuracy": report.get("rewrite_accuracy"),
        }
    return {}


def extract_misses(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return retrieval cases that missed a labelled section or a required fact."""
    rows = report.get("retrieval", {}).get("cases") or []
    misses = []
    for row in rows:
        if row.get("recall_at_k", 1) < 1.0 or row.get("fact_support", 1) < 1.0:
            misses.append(
                {
                    "id": row.get("id"),
                    "query": row.get("query"),
                    "recall_at_k": row.get("recall_at_k"),
                    "mrr": row.get("mrr"),
                    "ndcg_at_k": row.get("ndcg_at_k"),
                    "fact_support": row.get("fact_support"),
                    "tags": row.get("tags"),
                }
            )
    return misses


def load_history(path: Path | None = None) -> list[dict[str, Any]]:
    """Load the append-only history ledger, skipping blank or corrupt lines."""
    history_path = path or HISTORY_PATH
    if not history_path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in history_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError:
            logger.warning("Skipping corrupt history line in %s", history_path)
    return records


def _delta(current: dict[str, Any], previous: dict[str, Any] | None) -> dict[str, float]:
    """Subtract previous headline metrics from the current run (numeric keys only)."""
    if not previous:
        return {}
    prev_metrics = previous.get("metrics") or {}
    delta: dict[str, float] = {}
    for key in _HEADLINE_KEYS:
        now = current.get(key)
        then = prev_metrics.get(key)
        if isinstance(now, (int, float)) and isinstance(then, (int, float)):
            delta[key] = round(float(now) - float(then), 6)
    return delta


def _previous_same_kind(kind: str, history: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most recent history record of the same eval kind."""
    for record in reversed(history):
        if record.get("kind") == kind:
            return record
    return None


def format_run_markdown(envelope: dict[str, Any]) -> str:
    """Render a recorded run as Markdown for ``latest.md`` and the run folder."""
    metrics = envelope.get("metrics") or {}
    git = envelope.get("git") or {}
    delta = envelope.get("delta_vs_previous") or {}
    lines = [
        f"# Eval run `{envelope.get('run_id', '')}`",
        "",
        f"- kind: `{envelope.get('kind')}`",
        f"- recorded_at: `{envelope.get('recorded_at')}`",
        f"- duration_ms: `{envelope.get('duration_ms')}`",
        f"- git: `{git.get('branch', '')}` @ `{git.get('short_sha', '')}`"
        f"{' (dirty)' if git.get('dirty') else ''}",
    ]
    if envelope.get("note"):
        lines.append(f"- note: {envelope['note']}")
    lines += ["", "## Metrics", ""]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    if delta:
        lines += ["", "## Delta vs previous", ""]
        for key, value in delta.items():
            sign = "+" if value > 0 else ""
            lines.append(f"- {key}: {sign}{value:.4f}")
    misses = envelope.get("misses") or []
    if misses:
        lines += ["", "## Misses", ""]
        for miss in misses:
            lines.append(
                f"- `{miss.get('id')}` recall={miss.get('recall_at_k')} "
                f"fact={miss.get('fact_support')} — {miss.get('query')}"
            )
    lines.append("")
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with a trailing newline, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def record_run(
    kind: str,
    report: dict[str, Any],
    *,
    duration_ms: int = 0,
    note: str = "",
    results_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist one eval run and append a headline row to ``history.jsonl``.

    Args:
        kind:         Eval family: ``golden``, ``judge``, or ``policy``.
        report:       Raw report dict produced by the runner.
        duration_ms:  Wall time of the eval itself, excluding recording I/O.
        note:         Optional operator note stored on the ledger row.
        results_dir:  Override the default ``evals/results`` directory (tests).

    Returns:
        The ledger envelope (without the bulky per-case ``report`` payload).
    """
    base = results_dir or RESULTS_DIR
    runs_dir = base / "runs"
    history_path = base / "history.jsonl"
    latest_json = base / "latest.json"
    latest_md = base / "latest.md"

    recorded_at = datetime.now(tz=UTC)
    git = _git_info()
    stamp = recorded_at.strftime("%Y%m%dT%H%M%SZ")
    base_id = f"{stamp}-{kind}-{git.get('short_sha') or 'nogit'}"
    run_id = base_id
    suffix = 2
    while (runs_dir / run_id).exists():
        run_id = f"{base_id}-{suffix}"
        suffix += 1
    metrics = extract_metrics(kind, report)
    history = load_history(history_path)
    previous = _previous_same_kind(kind, history)

    envelope: dict[str, Any] = {
        "run_id": run_id,
        "kind": kind,
        "recorded_at": recorded_at.isoformat(),
        "duration_ms": duration_ms,
        "note": note or os.getenv("EVAL_NOTE", ""),
        "git": git,
        "config": _config_snapshot(),
        "dataset": _dataset_snapshot(),
        "metrics": metrics,
        "delta_vs_previous": _delta(metrics, previous),
        "misses": extract_misses(report) if kind == "golden" else [],
        "corpus_sections": report.get("corpus_sections"),
        "chunks_indexed": report.get("chunks_indexed"),
        "case_count": report.get("case_count"),
        "by_difficulty": (report.get("retrieval") or {}).get("by_difficulty"),
        "by_tag": (report.get("retrieval") or {}).get("by_tag"),
    }

    run_dir = runs_dir / run_id
    envelope["report_path"] = f"runs/{run_id}/report.json"
    full_payload = {**envelope, "report": report}
    _write_json(run_dir / "report.json", full_payload)
    (run_dir / "summary.md").write_text(format_run_markdown(envelope), encoding="utf-8")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(envelope, ensure_ascii=False) + "\n")

    # ``latest.*`` tracks the main golden suite so PRs show one snapshot.
    if kind == "golden" or not latest_json.exists():
        _write_json(latest_json, envelope)
        latest_md.write_text(format_run_markdown(envelope), encoding="utf-8")

    logger.info("Recorded eval run %s -> %s", run_id, history_path)
    return envelope


def format_history_table(records: list[dict[str, Any]], kind: str | None = None) -> str:
    """Render recent ledger rows as a fixed-width table."""
    rows = [row for row in records if kind is None or row.get("kind") == kind]
    if not rows:
        return "No eval runs recorded."
    header = (
        f"{'recorded_at':<22} {'kind':<8} {'recall@5':>9} {'ndcg@5':>8} "
        f"{'mrr':>6} {'fact':>6} {'sha':<8} note"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        metrics = row.get("metrics") or {}
        git = row.get("git") or {}
        recall = metrics.get("recall_at_k")
        ndcg = metrics.get("ndcg_at_k")
        mrr = metrics.get("mrr")
        fact = metrics.get("fact_support")
        lines.append(
            f"{str(row.get('recorded_at', ''))[:22]:<22} "
            f"{str(row.get('kind', '')):<8} "
            f"{'' if recall is None else f'{recall:.3f}':>9} "
            f"{'' if ndcg is None else f'{ndcg:.3f}':>8} "
            f"{'' if mrr is None else f'{mrr:.3f}':>6} "
            f"{'' if fact is None else f'{fact:.3f}':>6} "
            f"{str(git.get('short_sha', '')):<8} "
            f"{row.get('note') or ''}"
        )
    return "\n".join(lines)
