# Advanced-RAG

Self-corrective RAG system built with **FastAPI**, **LangGraph**, **Qdrant**, and **vLLM**.

## Architecture

```
Question → Retrieve (Qdrant) → Grade Documents (LLM) →┐
                ↑                                       │
                └── Rewrite Query (LLM) ←── irrelevant ─┘
                                             relevant ──→ Generate Answer (LLM) → Response
```

**Key features:**
- Self-corrective retrieval: automatically rewrites queries when documents are irrelevant
- **vLLM**: local HuggingFace model serving via OpenAI-compatible API (CPU/GPU)
- Qdrant vector store with FastEmbed (local embeddings, no API calls for embedding)
- LangGraph `StateGraph` with conditional edges for the RAG loop
- FastAPI REST API with Swagger docs at `/docs`
- Dual LLM backend: vLLM (local) or OpenAI (remote)

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Download Model from HuggingFace

```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### 3. Start vLLM Server

```bash
# Install vLLM CPU wheel (if not already installed)
export VLLM_VERSION=0.19.0
pip install "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cpu

# Start vLLM server (port 8001)
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
  -d '{"question": "Your question here"}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check (Qdrant + LLM status) |
| POST | `/api/v1/documents` | Ingest documents into vector store |
| POST | `/api/v1/search` | Semantic search (no LLM required) |
| POST | `/api/v1/query` | Full self-corrective RAG pipeline |

## LLM Backend Configuration

### vLLM (default, local)

```env
LLM_BACKEND=vllm
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_MODEL_PATH=/workspace/models/Qwen2.5-0.5B-Instruct
```

### OpenAI (remote)

```env
LLM_BACKEND=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-your-key
```

## Development

```bash
make dev       # Install with dev tools
make lint      # Run linter
make format    # Auto-format
make test      # Run tests
make run       # Start FastAPI dev server
make vllm-serve  # Start vLLM on port 8001
```

## Project Structure

```
app/
├── main.py              # FastAPI app with lifespan
├── config.py            # Pydantic settings (vLLM/OpenAI dual backend)
├── api/
│   ├── routes.py        # API endpoint handlers
│   └── schemas.py       # Request/response models
├── rag/
│   ├── graph.py         # LangGraph StateGraph (self-corrective RAG)
│   ├── nodes.py         # Graph nodes (retrieve, grade, rewrite, generate)
│   ├── prompts.py       # LLM prompt templates
│   └── state.py         # RAGState TypedDict
└── vectorstore/
    └── store.py         # Qdrant wrapper with FastEmbed
models/                  # HuggingFace models (gitignored)
tests/
├── test_api.py          # API endpoint tests
└── test_vectorstore.py  # Vector store unit tests
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `vllm` | `vllm` or `openai` |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server endpoint |
| `VLLM_MODEL_PATH` | `/workspace/models/Qwen2.5-0.5B-Instruct` | Local model path |
| `VLLM_MAX_MODEL_LEN` | `2048` | Max context length |
| `OPENAI_API_KEY` | (empty) | Required if LLM_BACKEND=openai |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `QDRANT_URL` | (empty = in-memory) | Qdrant server URL |
| `COLLECTION_NAME` | `advanced_rag` | Qdrant collection name |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K retrieval count |
| `MAX_RETRIES` | `3` | Max query rewrite retries |
