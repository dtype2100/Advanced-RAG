# Eval run `20260813T234452Z-golden-091cc79`

- kind: `golden`
- recorded_at: `2026-08-13T23:44:52.233357+00:00`
- duration_ms: `1675`
- git: `cursor/rag-eval-dataset-70ff` @ `091cc79`
- note: baseline: dense retrieval on frozen 28-section corpus

## Metrics

- hit_at_k: 1.0000
- recall_at_k: 0.9457
- mrr: 0.8957
- ndcg_at_k: 0.8965
- fact_support: 0.9783
- retrieval_n: 46
- clarification_accuracy: 1.0000
- rewrite_accuracy: 1.0000

## Misses

- `comparison_001` recall=0.5 fact=1.0 — What is the difference between dense vector search and BM25 in this pipeline?
- `paraphrase_001` recall=0.5 fact=0.5 — How do we stop the model from making things up after it writes an answer?
- `paraphrase_004` recall=0.5 fact=1.0 — How do we mix keyword matching with meaning-based search?
- `multihop_002` recall=0.5 fact=0.5 — What embedding dimension and collection name does the default stack use?
- `multihop_005` recall=0.5 fact=1.0 — After analyze_query, which graph step expands context instead of going straight to rerank?
