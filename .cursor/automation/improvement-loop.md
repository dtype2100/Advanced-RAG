# Cursor Cloud Automation — Improvement Loop

Use this file when creating an Automation in **Cursor Dashboard → Automations → New Automation**.

## Recommended settings

| Field | Value |
|-------|-------|
| **Name** | Advanced-RAG Improvement Loop |
| **Repository** | `dtype2100/Advanced-RAG` |
| **Branch** | `master` (or `cursor/auto-improvement-loop-3e09` while testing) |
| **Schedule** | Daily, or after merge to master |
| **Model** | composer-2.5-fast (or higher for judge tuning) |

## Automation prompt (copy-paste)

```
You are the Advanced-RAG continuous improvement agent. Follow the canonical loop
defined in app/core/improvement_loop.py in this exact order:

1. ANALYSIS      — review failing metrics / user feedback / open issues
2. VERIFICATION  — confirm the problem scope before changing code
3. SEARCH        — inspect relevant modules (graph, policies, evals, tests)
4. TEST          — reproduce with pytest; add a failing test if missing
5. EVALUATION    — run offline evals for the affected phase
6. VERIFICATION  — re-run make lint && make test && make evals-ci
7. FEEDBACK      — if evals fail, iterate (max 3 cycles); else open a PR

Commands (in order each cycle):
  make lint
  make test
  IMPROVEMENT_LOOP_CI=1 make evals-ci

If OPENAI_API_KEY is available and judge tuning is needed:
  make evals

Rules:
- Minimal diff; match existing conventions in app/
- Never skip tests before opening a PR
- Update evals/datasets/*.jsonl or evals/regression/golden_set.yaml when fixing routing/policy bugs
- Commit message format: "fix(improvement-loop): <phase> — <summary>"
- Create a draft PR when a measurable improvement is verified
- Stop after 3 failed improvement cycles and report blockers

Phase → code map:
  analysis       → app/rag/query/query_analyzer.py
  verification   → app/rag/policies/clarification_policy.py, rewrite_policy.py
  search         → app/rag/retrievers/, app/graphs/crag/nodes.py (hybrid_retrieve)
  test           → app/graphs/crag/nodes.py (test_retrieval), tests/unit/
  evaluation     → app/graphs/crag/nodes.py (generate_answer, run_judge)
  verification   → app/rag/evaluators/, evaluate_grounding node
  feedback       → app/rag/policies/routing_policy.py (route_after_feedback)
```

## Trigger options

### Option A — Scheduled (recommended)
- **Trigger:** Cron / daily
- Runs regression + proposes fixes when evals fail

### Option B — On PR merge
- **Trigger:** SCM / merge to master
- Validates master after each merge

### Option C — Manual
- **Trigger:** Manual / workflow_dispatch equivalent in Automations UI
- Use when tuning a specific phase (e.g. feedback routing)

## What runs automatically without this Automation

GitHub Actions workflow `.github/workflows/improvement-loop.yml` already runs on:
- Daily schedule (06:00 UTC)
- Push to `master` (app/evals/tests changes)
- Manual dispatch (`full_loop` input for LLM judge eval)

CI command: `make evals-ci` (skips LLM judge phase).

## Verify setup

```bash
make evals-ci          # local CI-safe loop
make evals             # full loop (needs vLLM or OPENAI_API_KEY)
```

Expected CI phases: analysis → search → test → evaluation → feedback (5 phases).
