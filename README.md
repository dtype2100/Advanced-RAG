# Advanced-RAG

Self-corrective RAG system built with **FastAPI**, **LangGraph**, and **Qdrant**.

## Architecture

```
Question → Retrieve (Qdrant) → Grade Documents (LLM) →┐
                ↑                                       │
                └── Rewrite Query (LLM) ←── irrelevant ─┘
                                             relevant ──→ Generate Answer (LLM) → Response
```

**Key features:**
- Self-corrective retrieval: automatically rewrites queries when documents are irrelevant
- Qdrant vector store with FastEmbed (local embeddings, no API calls for embedding)
- LangGraph `StateGraph` with conditional edges for the RAG loop
- FastAPI REST API with Swagger docs at `/docs`
- In-memory mode by default (no external services needed to start)

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy env and set your OpenAI key (required for /query RAG endpoint)
cp .env.example .env
# edit .env → set OPENAI_API_KEY

# Run dev server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or use Make
make run
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/api/v1/health` | Health check (Qdrant status) |
| POST | `/api/v1/documents` | Ingest documents into vector store |
| POST | `/api/v1/search` | Semantic search (no LLM required) |
| POST | `/api/v1/query` | Full RAG pipeline (requires OpenAI key) |

## Development

```bash
# Install with dev tools
make dev

# Run linter
make lint

# Auto-format
make format

# Run tests
make test
```

## Project Structure

```
app/
├── main.py              # FastAPI app with lifespan
├── config.py            # Pydantic settings from .env
├── api/
│   ├── routes.py        # API endpoint handlers
│   └── schemas.py       # Request/response models
├── rag/
│   ├── graph.py         # LangGraph StateGraph (self-corrective RAG)
│   ├── nodes.py         # Graph node functions (retrieve, grade, rewrite, generate)
│   ├── prompts.py       # LLM prompt templates
│   └── state.py         # RAGState TypedDict
└── vectorstore/
    └── store.py         # Qdrant wrapper with FastEmbed
tests/
├── test_api.py          # API endpoint tests
└── test_vectorstore.py  # Vector store unit tests
```

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required for /query) | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | FastEmbed model |
| `QDRANT_URL` | (empty = in-memory) | Qdrant server URL |
| `COLLECTION_NAME` | `advanced_rag` | Qdrant collection name |
| `MAX_RETRIEVAL_DOCS` | `5` | Top-K retrieval count |
| `MAX_RETRIES` | `3` | Max query rewrite retries |
