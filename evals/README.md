# Evaluation datasets

Frozen corpus and golden cases for this CRAG stack. Retrieval labels are
**section-level qrels** (`source_id` + `section_id`), not retriever scores.

## Layout

| Path | Role |
|------|------|
| `corpus/docs.jsonl` | Ingest snapshot. Each record is one section with stable ids. |
| `datasets/cases.jsonl` | Golden cases: query, qrels, reference answer, policy labels. |
| `datasets/fixtures/judge_eval.jsonl` | Canned (question, context, answer) pairs for the LLM judge. |
| `offline/` | Scripts that ingest the corpus and print metrics. |

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
```

`run_retrieval_eval.py` ingests `corpus/docs.jsonl` into the in-memory Qdrant
client for that process, then scores Recall@5 / MRR / nDCG@5.

Judge eval is skipped when the LLM backend is unreachable.

## Adding cases

1. Add or reuse a section in `corpus/docs.jsonl` (keep text short enough that
   default chunk size 512 does not split away the labelled facts).
2. Write the question **from that section**, then the reference answer from the
   same text only.
3. Point `qrels` at existing `(source_id, section_id)` pairs.
4. `python -c "from evals.loader import load_cases; load_cases()"` to validate.
