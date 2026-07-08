# Advanced-RAG

Self-corrective RAG (CRAG) system built with **FastAPI**, **LangGraph**, **Qdrant**, and **vLLM**.

## Architecture

```
Question → Analyze → [Clarify?] → [Rewrite?] → Hybrid Retrieve (Vector + BM25)
    → [Expand parent context?] → Rerank → Generate → Grounding eval
    → [Retry loop, max 3×] → Response with structured citations
```

**Key features:**
- Hybrid retrieval: dense vector search + BM25 (RRF fusion) with optional multi-query
- Parent-child chunking on ingest with context expansion at query time
- LangGraph CRAG pipeline with clarification, rewrite, and hallucination feedback loops
- Structured API citations (`text`, `source`, `page`, `score`)
- Dual LLM backend: vLLM (local) or OpenAI (remote)
- ARQ + Redis async document ingest

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev,bm25]"
```

### 2. Download Model from HuggingFace

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### 3. Start vLLM Server

```bash
make vllm-serve
```

### 4. Start RAG API Server

```bash
cp .env.example .env
make run
```

### 5. Test the Pipeline

```bash
# Ingest documents
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Your document text here"}]}'

# RAG query (uses vLLM)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here", "top_k": 5}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check (Qdrant + LLM status) |
| POST | `/api/v1/documents` | Ingest documents (sync or async queue) |
| POST | `/api/v1/documents/async` | Always enqueue ingest (requires Redis) |
| GET | `/api/v1/jobs/{job_id}` | Poll async ingest job status |
| POST | `/api/v1/search` | Semantic search (no LLM required) |
| POST | `/api/v1/query` | Full CRAG pipeline |
| POST | `/api/v1/query/stream` | CRAG pipeline as SSE stream |

When `API_KEY` is set, protected endpoints require the `X-API-Key` header.

## Project Structure

```
app/
├── main.py                 # FastAPI app + lifespan
├── core/
│   ├── config.py           # Pydantic settings (single source of truth)
│   ├── security.py         # Optional API key auth
│   └── llm_health.py       # LLM readiness probe
├── api/v1/                 # REST endpoints
├── graphs/crag/            # LangGraph CRAG pipeline
├── rag/
│   ├── retrievers/         # Hybrid, BM25, parent-child
│   ├── pipelines/          # Ingest pipeline
│   ├── policies/           # Routing policies
│   └── types.py            # ChunkHit helpers
├── providers/              # LLM, vector store, reranker
└── storage/                # Qdrant, Redis, Postgres stubs
tests/
├── unit/
└── integration/
evals/                      # Offline evaluation scripts
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `vllm` | `vllm` or `openai` |
| `MULTI_QUERY` | `false` | Generate query variants for retrieval |
| `USE_PARENT_CHILD_CHUNKING` | `true` | Parent-child ingest + expansion |
| `RERANKER_BACKEND` | `none` | `none`, `cross_encoder`, or `llm` |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `pgvector` |
| `API_KEY` | (empty) | Optional API key for protected routes |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K retrieval count |
| `MAX_RETRIES` | `3` | Max hallucination retry loops |
| `REDIS_URL` | (empty) | Required for async ingest |

See `.env.example` for the full list.

## Development

```bash
make dev       # Install with dev tools
make lint      # Run linter
make format    # Auto-format
make test      # Run all tests
make test-unit # Unit tests only
make evals     # Offline evaluation scripts
make run       # Start FastAPI dev server
make worker    # Start ARQ worker (requires REDIS_URL)
make vllm-serve  # Start vLLM on port 8001
```

## Docker Compose

```bash
docker compose up redis qdrant api worker
```

vLLM must be started separately or configured via `LLM_BACKEND=openai`.
