# Advanced-RAG

Agentic RAG system built with **FastAPI**, **LangGraph (CRAG graph)**, **Qdrant**, optional **vLLM** or **OpenAI**, and **ARQ + Redis** for background document ingestion.

## Features

- **CRAG pipeline**: conditional clarification, rewrite, retrieval, expansion, grounding, optional LLM-as-judge, feedback loop (up to 3 retries).
- **Vector store**: Qdrant + FastEmbed; pluggable port (`VECTOR_BACKEND`) for future pgvector.
- **LLM**: vLLM (OpenAI-compatible local server) or OpenAI API.
- **Async ingest**: enqueue heavy indexing via **Redis + ARQ worker** (`POST /documents/async`, optional `INGEST_QUEUE_ASYNC` for `POST /documents`).
- **Docker Compose**: `api`, `qdrant`, `redis`, `worker` services for deployment.

## Quick start (local)

### 1. Install dependencies

```bash
pip install -e ".[dev]"
```

### 2. Model for vLLM (optional, for `/api/v1/query`)

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### 3. Environment

```bash
cp .env.example .env
# Edit .env — see Configuration Reference below
```

### 4. vLLM (optional)

```bash
make vllm-serve   # port 8001
```

### 5. API server

```bash
make run   # port 8000
```

### 6. Async ingest (optional)

Requires Redis and a worker process:

```bash
# Terminal 1: Redis (e.g. docker run -p 6379:6379 redis:7-alpine)
# Set in .env: REDIS_URL=redis://localhost:6379

# Terminal 2
make worker

# Enqueue ingest (202 + job_id), then poll status
curl -X POST http://localhost:8000/api/v1/documents/async \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Your document text"}]}'
curl http://localhost:8000/api/v1/jobs/<job_id>
```

### 7. Smoke test

```bash
# Ingest (synchronous if INGEST_QUEUE_ASYNC is false)
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Python is a programming language."}]}'

# Search (no LLM)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "programming", "top_k": 3}'

# RAG query (needs vLLM or OPENAI_API_KEY)
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python?"}'
```

## Web UI (Chat + Studio)

```bash
make run
make web-install   # once
make web-dev       # http://localhost:5173 — proxies `/api` → :8000
```

- **/** — chat-style UI (markdown answers, collapsible retrieval sources)
- **/studio** — runtime RAG parameters, read-only env snapshot, vLLM/Qdrant/TEI probe

Set `CORS_ORIGINS` in `.env` if the UI runs on another origin. Production: `cd web && npm run build` and serve `web/dist`.

## Docker Compose (deployment)

From the repository root:

```bash
cp .env.example .env
# Set secrets and overrides in .env as needed

docker compose up --build -d
```

Services:

| Service | Role |
|---------|------|
| `api` | FastAPI on port **8000** |
| `qdrant` | Vector DB on **6333** |
| `redis` | ARQ broker on **6379** |
| `worker` | `arq app.workers.settings.WorkerSettings` — processes ingest jobs |

Compose sets `QDRANT_URL`, `REDIS_URL`, and `VECTOR_BACKEND` for the Docker network. Ensure `.env` exists (or remove `env_file` and pass env another way in production).

**Note:** The Qdrant image healthcheck uses `curl`. If healthchecks fail, verify the image includes `curl` or adjust the healthcheck in `docker-compose.yml`.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health (Qdrant, LLM config) |
| POST | `/api/v1/documents` | Ingest (sync, or 202 + `job_id` if `INGEST_QUEUE_ASYNC=true` and `REDIS_URL` set) |
| POST | `/api/v1/documents/async` | Always enqueue ingest (202 + `job_id`; requires `REDIS_URL` + worker) |
| GET | `/api/v1/jobs/{job_id}` | ARQ job status / result |
| POST | `/api/v1/search` | Semantic search (no LLM) |
| POST | `/api/v1/query` | Full CRAG chat pipeline |
| GET | `/api/v1/studio/runtime` | Process-local RAG overrides + effective values |
| PATCH | `/api/v1/studio/runtime` | Update overrides (JSON; `null` clears) |
| GET | `/api/v1/studio/config` | Read-only env-backed settings snapshot |
| POST | `/api/v1/studio/probe` | Reachability: vLLM, Qdrant, optional TEI URLs |

OpenAPI: `http://localhost:8000/docs`

## LLM configuration

### vLLM (local)

```env
LLM_BACKEND=vllm
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
VLLM_BASE_URL=http://localhost:8001/v1
```

### OpenAI

```env
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key
```

## Configuration reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `vllm` | `vllm` or `openai` |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM OpenAI-compatible base URL |
| `OPENAI_API_KEY` | (empty) | Required if `LLM_BACKEND=openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `QDRANT_URL` | (empty = in-memory) | Qdrant server URL |
| `COLLECTION_NAME` | `advanced_rag` | Collection name |
| `VECTOR_BACKEND` | `qdrant` | `qdrant` or `pgvector` (stub) |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K for retrieval |
| `MAX_RETRIES` | `3` | Max feedback-loop retries in graph |
| `REDIS_URL` | (empty) | Redis for ARQ; required for async ingest |
| `INGEST_QUEUE_ASYNC` | `false` | If `true` and `REDIS_URL` set, `POST /documents` returns 202 + job |
| `ARQ_QUEUE_NAME` | `arq:queue` | Must match worker queue name |
| `LOG_LLM_IO` | `false` | If `true`, log prompts/outputs under logger `app.llm_io` (sensitive) |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated origins for browser UI |
| `TEI_EMBEDDING_URL` | (empty) | Text Embeddings Inference URL (for Studio probe) |
| `TEI_RERANK_URL` | (empty) | TEI reranker URL (for Studio probe) |

Judge LLM (optional): `JUDGE_LLM_BACKEND`, `JUDGE_LLM_MODEL`, `JUDGE_VLLM_BASE_URL` — see `app/providers/judge_llm_provider.py`.

## Development

```bash
make dev          # pip install -e ".[dev]"
make lint         # ruff check + format check
make format       # ruff fix + format
make test         # pytest
make run          # FastAPI dev server
make worker       # ARQ worker (needs REDIS_URL)
make vllm-serve   # vLLM on 8001
make evals        # offline eval scripts
make web-install  # web UI dependencies
make web-dev      # Vite dev server :5173
```

## Project layout (high level)

```
app/
├── main.py                 # FastAPI app, routers, lifespan
├── core/                   # config, logging, security, constants
├── api/v1/                 # health, ingest, chat, jobs
├── schemas/                # Pydantic request/response models
├── services/               # chat, ingest, index, document
├── storage/                # vectorstores (Qdrant, factory), redis/postgres stubs
├── providers/              # LLM, judge LLM, embedding, vectorstore, reranker, cache
├── queue/                  # ARQ Redis pool for API enqueue
├── workers/                # ARQ job functions + WorkerSettings
├── graphs/crag/            # LangGraph: graph, nodes, routes, state
├── rag/                    # loaders, chunkers, retrievers, policies, evaluators, …
└── prompts/                # YAML prompt templates
docker-compose.yml          # api, qdrant, redis, worker
docker/                     # Dockerfile, entrypoint
evals/, scripts/, tests/     # offline evals, CLI, unit + integration tests
web/                         # Vite React UI (chat + studio)
```

See `AGENTS.md` for Cursor Cloud / VM notes (ports, vLLM gotchas, startup order).
