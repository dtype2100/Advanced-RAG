# Advanced-RAG

Self-corrective RAG system built with **FastAPI**, **LangGraph**, and pluggable providers.

## Architecture

```
Question → Retrieve (VectorStore) → [Rerank] → Grade Documents (LLM) →┐
                ↑                                                       │
                └── Rewrite Query (LLM) ←── irrelevant ────────────────┘
                                             relevant ──→ Generate Answer (LLM) → Response
```

**Key features:**
- **Pluggable providers** — switch LLM, Embedding, Reranker, VectorStore via env vars alone
- Self-corrective retrieval: automatically rewrites queries when documents are irrelevant
- Optional reranking stage (FlashRank / CrossEncoder)
- LangGraph `StateGraph` with conditional edges for the RAG loop
- FastAPI REST API with Swagger docs at `/docs`

## Supported Providers

| Component | Provider | Env Value | Extra Install |
|-----------|----------|-----------|---------------|
| **LLM** | vLLM (local) | `vllm` | — |
| | OpenAI | `openai` | — |
| | Anthropic | `anthropic` | `pip install -e ".[anthropic]"` |
| **Embedding** | FastEmbed (local) | `fastembed` | — |
| | OpenAI | `openai` | — |
| | HuggingFace | `huggingface` | `pip install -e ".[huggingface]"` |
| **Reranker** | None | `none` | — |
| | FlashRank | `flashrank` | `pip install -e ".[reranker-flashrank]"` |
| | CrossEncoder | `crossencoder` | `pip install -e ".[reranker-crossencoder]"` |
| **VectorStore** | Qdrant | `qdrant` | — |
| | ChromaDB | `chroma` | `pip install -e ".[chroma]"` |

## Quick Start

### 1. Install

```bash
pip install -e ".[dev]"        # core + dev tools
pip install -e ".[all]"        # all optional providers
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env to set your provider choices
```

### 3. Run

```bash
make run          # FastAPI on :8000
make vllm-serve   # vLLM on :8001 (if using vllm provider)
```

### 4. Test

```bash
# Ingest documents
curl -X POST http://localhost:8000/api/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"text": "Your document text here"}]}'

# RAG query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check (provider status) |
| POST | `/api/v1/documents` | Ingest documents into vectorstore |
| POST | `/api/v1/search` | Semantic search (no LLM required) |
| POST | `/api/v1/query` | Full self-corrective RAG pipeline |

## Example Configurations

### Local (vLLM + FastEmbed + Qdrant)

```env
LLM_PROVIDER=vllm
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTORSTORE_PROVIDER=qdrant
RERANKER_PROVIDER=none
```

### Cloud (OpenAI + ChromaDB + FlashRank)

```env
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
VECTORSTORE_PROVIDER=chroma
CHROMA_PERSIST_DIR=./chroma_data
RERANKER_PROVIDER=flashrank
RERANKER_MODEL=ms-marco-MultiBERT-L-12
```

### Anthropic + HuggingFace Embeddings

```env
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
VECTORSTORE_PROVIDER=qdrant
QDRANT_URL=http://localhost:6333
RERANKER_PROVIDER=crossencoder
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
```

## Development

```bash
make dev          # Install with dev tools
make lint         # Run linter
make format       # Auto-format
make test         # Run tests
make run          # Start FastAPI dev server
make vllm-serve   # Start vLLM on port 8001
```

## Project Structure

```
app/
├── main.py                          # FastAPI app with lifespan
├── config.py                        # Pydantic settings (all providers)
├── api/
│   ├── routes.py                    # API endpoint handlers
│   └── schemas.py                   # Request/response models
├── core/
│   ├── embedding/
│   │   ├── base.py                  # BaseEmbedding ABC
│   │   ├── factory.py               # get_embedding() factory
│   │   ├── fastembed_provider.py    # FastEmbed (local)
│   │   ├── openai_provider.py       # OpenAI Embeddings API
│   │   └── huggingface_provider.py  # sentence-transformers
│   ├── llm/
│   │   └── factory.py               # get_llm() → ChatOpenAI / ChatAnthropic
│   ├── reranker/
│   │   ├── base.py                  # BaseReranker ABC
│   │   ├── factory.py               # get_reranker() factory
│   │   ├── flashrank_provider.py    # FlashRank (CPU)
│   │   └── crossencoder_provider.py # sentence-transformers CrossEncoder
│   └── vectorstore/
│       ├── base.py                  # BaseVectorStore ABC
│       ├── factory.py               # get_vectorstore() factory
│       ├── qdrant_provider.py       # Qdrant
│       └── chroma_provider.py       # ChromaDB
├── rag/
│   ├── graph.py                     # LangGraph StateGraph
│   ├── nodes.py                     # Graph nodes (retrieve, grade, rewrite, generate)
│   ├── prompts.py                   # LLM prompt templates
│   └── state.py                     # RAGState TypedDict
tests/
├── conftest.py
├── test_api.py
└── test_vectorstore.py
```

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `vllm` | `vllm`, `openai`, or `anthropic` |
| `LLM_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Model name |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `VLLM_BASE_URL` | `http://localhost:8001/v1` | vLLM server endpoint |
| `OPENAI_API_KEY` | (empty) | Required if using openai provider |
| `ANTHROPIC_API_KEY` | (empty) | Required if using anthropic provider |
| `EMBEDDING_PROVIDER` | `fastembed` | `fastembed`, `openai`, or `huggingface` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model name |
| `EMBEDDING_DIMENSION` | `384` | Fallback vector dimension |
| `RERANKER_PROVIDER` | `none` | `none`, `flashrank`, or `crossencoder` |
| `RERANKER_MODEL` | `ms-marco-MultiBERT-L-12` | Reranker model name |
| `RERANKER_TOP_K` | `5` | Documents to keep after reranking |
| `VECTORSTORE_PROVIDER` | `qdrant` | `qdrant` or `chroma` |
| `COLLECTION_NAME` | `advanced_rag` | Collection/index name |
| `QDRANT_URL` | (empty = in-memory) | Qdrant server URL |
| `QDRANT_API_KEY` | (empty) | Qdrant API key |
| `CHROMA_HOST` | `localhost` | ChromaDB host |
| `CHROMA_PORT` | `8200` | ChromaDB port |
| `CHROMA_PERSIST_DIR` | (empty) | ChromaDB persistent storage path |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K retrieval count |
| `MAX_RETRIES` | `3` | Max query rewrite retries |
