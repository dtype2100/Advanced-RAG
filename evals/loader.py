"""Load and validate the frozen eval corpus and golden cases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVALS_ROOT = Path(__file__).resolve().parent
CORPUS_PATH = EVALS_ROOT / "corpus" / "docs.jsonl"
CASES_PATH = EVALS_ROOT / "datasets" / "cases.jsonl"
JUDGE_FIXTURE_PATH = EVALS_ROOT / "datasets" / "fixtures" / "judge_eval.jsonl"

REQUIRED_CASE_FIELDS = (
    "id",
    "query",
    "intent",
    "difficulty",
    "tags",
    "expected_clarification",
    "expected_rewrite",
    "unanswerable",
    "qrels",
)

VALID_INTENTS = {"factual", "comparison", "procedural", "unknown"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
    return records


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """Load ingest-ready corpus docs ``{text, metadata}``."""
    docs = load_jsonl(path)
    for index, doc in enumerate(docs):
        if "text" not in doc or "metadata" not in doc:
            raise ValueError(f"Corpus record {index} missing text or metadata")
        meta = doc["metadata"]
        if not meta.get("source_id") or not meta.get("section_id"):
            raise ValueError(f"Corpus record {index} missing source_id or section_id")
    return docs


def validate_case(case: dict[str, Any], corpus_keys: set[tuple[str, str]]) -> None:
    """Raise ``ValueError`` if a golden case is malformed."""
    missing = [field for field in REQUIRED_CASE_FIELDS if field not in case]
    if missing:
        raise ValueError(f"Case {case.get('id', '<unknown>')} missing fields: {missing}")
    if case["intent"] not in VALID_INTENTS:
        raise ValueError(f"Case {case['id']} has invalid intent {case['intent']}")
    if case["difficulty"] not in VALID_DIFFICULTIES:
        raise ValueError(f"Case {case['id']} has invalid difficulty {case['difficulty']}")
    if not isinstance(case["tags"], list) or not case["tags"]:
        raise ValueError(f"Case {case['id']} must have a non-empty tags list")
    if not isinstance(case["qrels"], list):
        raise ValueError(f"Case {case['id']} qrels must be a list")

    for qrel in case["qrels"]:
        key = (str(qrel.get("source_id", "")), str(qrel.get("section_id", "")))
        rel = int(qrel.get("relevance", 0))
        if rel not in (1, 2):
            raise ValueError(f"Case {case['id']} qrel relevance must be 1 or 2, got {rel}")
        if key not in corpus_keys:
            raise ValueError(f"Case {case['id']} qrel {key} is not in the frozen corpus")

    if case["unanswerable"] and case["qrels"]:
        raise ValueError(f"Case {case['id']} is unanswerable but has qrels")
    if not case["unanswerable"] and not case["expected_clarification"] and not case["qrels"]:
        raise ValueError(f"Case {case['id']} needs qrels unless unanswerable or clarification-only")
    if not case["unanswerable"] and not case["expected_clarification"]:
        if not case.get("reference_answer"):
            raise ValueError(f"Case {case['id']} missing reference_answer")
        if not case.get("must_cite_facts"):
            raise ValueError(f"Case {case['id']} missing must_cite_facts")


def load_cases(
    path: Path = CASES_PATH,
    corpus: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Load golden cases and validate them against the frozen corpus."""
    docs = corpus if corpus is not None else load_corpus()
    corpus_keys = {
        (str(doc["metadata"]["source_id"]), str(doc["metadata"]["section_id"])) for doc in docs
    }
    cases = load_jsonl(path)
    seen_ids: set[str] = set()
    for case in cases:
        validate_case(case, corpus_keys)
        if case["id"] in seen_ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        seen_ids.add(case["id"])
    return cases


def retrieval_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cases that should be scored with retrieval metrics (have qrels)."""
    return [
        case
        for case in cases
        if case["qrels"] and not case["unanswerable"] and not case["expected_clarification"]
    ]


def fact_support_score(must_cite_facts: list[str], contexts: list[str]) -> float:
    """Fraction of required facts that appear in retrieved context (case-insensitive).

    This is a reference-free-of-LLM check: the answer cannot be grounded if the
    retriever never surfaced the supporting strings.
    """
    if not must_cite_facts:
        return 0.0
    blob = "\n".join(contexts).lower()
    hits = sum(1 for fact in must_cite_facts if fact.lower() in blob)
    return hits / len(must_cite_facts)
