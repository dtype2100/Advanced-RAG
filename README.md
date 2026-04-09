# Advanced-RAG

Self-corrective RAG system built with **FastAPI**, **LangGraph**, and pluggable providers.

## Architecture

```
Question → Retrieve (VectorStore) → Grade Documents (LLM) →┐
                ↑                                            │
                └── Rewrite Query (LLM) ←── irrelevant ─────┘
                                             relevant ──→ Generate Answer (LLM) → Response
```

**Key features:**
- Self-corrective retrieval: automatically rewrites queries when documents are irrelevant
- **Pluggable providers**: switch embedding, vector store, reranker, and LLM backends via environment variables
- **vLLM** or **OpenAI** for LLM inference
- **FastEmbed** or **OpenAI** for embeddings
- **Qdrant** or **ChromaDB** for vector storage
- **FlashRank** optional reranker for improved retrieval precision
- LangGraph `StateGraph` with conditional edges for the RAG loop
- FastAPI REST API with Swagger docs at `/docs`

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"

# Optional providers:
pip install -e ".[chroma]"         # ChromaDB support
pip install -e ".[openai-embed]"   # OpenAI embeddings
pip install -e ".[reranker]"       # FlashRank reranker
pip install -e ".[all]"            # All optional providers
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
| GET | `/api/v1/health` | Health check (all provider statuses) |
| POST | `/api/v1/documents` | Ingest documents into vector store |
| POST | `/api/v1/search` | Semantic search (no LLM required) |
| POST | `/api/v1/query` | Full self-corrective RAG pipeline |

## Provider Configuration

All providers are selected by a single environment variable each:

### LLM Backend

| `LLM_BACKEND` | Description |
|----------------|-------------|
| `vllm` (default) | Local vLLM server (OpenAI-compatible) |
| `openai` | OpenAI API |

### Embedding Provider

| `EMBEDDING_PROVIDER` | Description |
|-----------------------|-------------|
| `fastembed` (default) | Local FastEmbed models (no API key) |
| `openai` | OpenAI Embedding API |

### Vector Store Provider

| `VECTORSTORE_PROVIDER` | Description |
|-------------------------|-------------|
| `qdrant` (default) | Qdrant (in-memory or remote) |
| `chroma` | ChromaDB (in-memory or persistent) |

### Reranker Provider

| `RERANKER_PROVIDER` | Description |
|----------------------|-------------|
| `none` (default) | No reranking |
| `flashrank` | FlashRank cross-encoder |

## Development

```bash
make dev         # Install with dev tools
make lint        # Run linter
make format      # Auto-format
make test        # Run tests
make run         # Start FastAPI dev server
make vllm-serve  # Start vLLM on port 8001
```

## Project Structure

```
app/
├── main.py                    # FastAPI app with lifespan
├── config.py                  # Pydantic settings (all provider configs)
├── api/
│   ├── routes.py              # API endpoint handlers
│   └── schemas.py             # Request/response models
├── core/
│   ├── embedding/
│   │   ├── base.py            # BaseEmbedding ABC
│   │   ├── fastembed.py       # FastEmbed provider
│   │   ├── openai.py          # OpenAI embedding provider
│   │   └── factory.py         # Singleton factory
│   ├── llm/
│   │   └── factory.py         # LLM factory (vLLM / OpenAI)
│   ├── reranker/
│   │   ├── base.py            # BaseReranker ABC
│   │   ├── noop.py            # No-op passthrough
│   │   ├── flashrank.py       # FlashRank cross-encoder
│   │   └── factory.py         # Singleton factory
│   └── vectorstore/
│       ├── base.py            # BaseVectorStore ABC
│       ├── qdrant.py          # Qdrant provider
│       ├── chroma.py          # ChromaDB provider
│       └── factory.py         # Singleton factory
├── services/
│   └── retrieval.py           # Embed + search + rerank orchestration
└── rag/
    ├── graph.py               # LangGraph StateGraph (self-corrective RAG)
    ├── nodes.py               # Graph nodes (retrieve, grade, rewrite, generate)
    ├── prompts.py             # LLM prompt templates
    └── state.py               # RAGState TypedDict
models/                        # HuggingFace models (gitignored)
tests/
├── conftest.py                # Test fixtures
├── test_api.py                # API endpoint tests
└── test_vectorstore.py        # Retrieval service tests
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `vllm` | `vllm` or `openai` |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server endpoint |
| `VLLM_MODEL_PATH` | `/workspace/models/Qwen2.5-0.5B-Instruct` | Local model path |
| `VLLM_MAX_MODEL_LEN` | `2048` | Max context length |
| `OPENAI_API_KEY` | (empty) | Required if using OpenAI LLM or embeddings |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed` or `openai` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model name |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `VECTORSTORE_PROVIDER` | `qdrant` | `qdrant` or `chroma` |
| `QDRANT_URL` | (empty = in-memory) | Qdrant server URL |
| `QDRANT_API_KEY` | (empty) | Qdrant API key |
| `CHROMA_HOST` | `localhost` | Chroma server host |
| `CHROMA_PORT` | `8500` | Chroma server port |
| `CHROMA_PERSIST_DIR` | (empty = in-memory) | Chroma persistence path |
| `COLLECTION_NAME` | `advanced_rag` | Vector collection name |
| `RERANKER_PROVIDER` | `none` | `none` or `flashrank` |
| `RERANKER_MODEL` | `ms-marco-MiniLM-L-12-v2` | Reranker model |
| `RERANKER_TOP_K` | `5` | Reranker top-K |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K retrieval count |
| `MAX_RETRIES` | `3` | Max query rewrite retries |
