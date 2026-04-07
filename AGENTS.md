# AGENTS.md

## Cursor Cloud specific instructions

### Services

| Service | Command | Notes |
|---------|---------|-------|
| FastAPI dev server | `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` | or `make run` |
| Tests | `pytest -v` | or `make test` |
| Lint | `ruff check app/ tests/` | or `make lint` |

### Key Gotchas

- **Qdrant in-memory by default:** When `QDRANT_URL` is empty (default), the app uses `QdrantClient(":memory:")`. Data is lost on restart. This is intentional for dev/test.
- **qdrant-client v1.12+:** Uses `query_points()` not the deprecated `search()`. The response is `QueryResponse` with `.points` attribute.
- **FastEmbed downloads models on first use:** The first embedding call downloads `BAAI/bge-small-en-v1.5` (~130MB). This is cached in `~/.cache/fastembed/`.
- **OpenAI key only needed for `/api/v1/query`:** The `/documents` and `/search` endpoints work without `OPENAI_API_KEY` since they use local FastEmbed embeddings.
- **Ruff binary location:** After `pip install`, ruff is at `~/.local/bin/ruff`. Ensure `PATH` includes `$HOME/.local/bin`.
- **Singleton pattern for Qdrant client:** `app/vectorstore/store.py` uses a module-level singleton. If tests share state, the in-memory Qdrant retains data across test functions in the same process.
