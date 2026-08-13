# Evaluation datasets

Frozen corpus and golden cases for this CRAG stack. Retrieval labels are
**section-level qrels** (`source_id` + `section_id`), not retriever scores.

## Layout

| Path | Role |
|------|------|
| `corpus/docs.jsonl` | Ingest snapshot. Each record is one section with stable ids. |
| `datasets/cases.jsonl` | Golden cases: query, qrels, reference answer, policy labels. |
| `datasets/fixtures/judge_eval.jsonl` | Canned (question, context, answer) pairs for the LLM judge. |
| `offline/` | Scripts that ingest the corpus, score metrics, and record runs. |
| `results/history.jsonl` | Append-only ledger of headline metrics (tracked). |
| `results/latest.json` / `latest.md` | Most recent golden snapshot (tracked). |
| `results/runs/` | Full per-case dumps (gitignored). |

## Case schema

Required fields: `id`, `query`, `intent`, `difficulty`, `tags`,
`expected_clarification`, `expected_rewrite`, `unanswerable`, `qrels`.

Answerable retrieval cases also have `reference_answer` and `must_cite_facts`.
Qrels use graded relevance `2` (supports the answer) or `1` (partially useful).

## How to run

```bash
make evals
# or
python scripts/run_evals.py
python evals/offline/run_golden_eval.py --note "why this run exists"
make evals-history
```

Each golden/judge run appends one row to `results/history.jsonl`, writes
`results/latest.json` + `latest.md`, and stores the full per-case dump under
`results/runs/<run_id>/`. Use `--no-record` to skip persistence. `EVAL_NOTE`
is copied onto the ledger row when `--note` is omitted.

`run_retrieval_eval.py` / `run_golden_eval.py` ingest `corpus/docs.jsonl` into the
in-memory Qdrant client for that process, then score Recall@5 / MRR / nDCG@5.

Judge eval is skipped when the LLM backend is unreachable.

Baseline on this corpus (dense retrieval, k=5, 46 labelled queries):
Hit@5 100%, Recall@5 ~95%, nDCG@5 ~0.90. Hard / paraphrase / multi-hop cases
are the ones that miss a second supporting section. Treat this as a regression
snapshot on 28 sections, not a production-scale IR benchmark.

## Query wording vs clarification heuristics

`query_analyzer` flags `when` without a four-digit year as `time_period`, and
`where` without `in`/`at` as `location`. Retrieval cases in this set are phrased
to avoid those tokens so they actually reach search. Dedicated `clarify_*`
items exist to lock the current heuristic.

## Adding cases

1. Add or reuse a section in `corpus/docs.jsonl` (keep text short enough that
   default chunk size 512 does not split away the labelled facts).
2. Write the question **from that section**, then the reference answer from the
   same text only.
3. Point `qrels` at existing `(source_id, section_id)` pairs.
4. `python -c "from evals.loader import load_cases; load_cases()"` to validate.
