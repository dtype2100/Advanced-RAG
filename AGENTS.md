# AGENTS.md

## Cursor Cloud specific instructions

### Services

| Service | Command | Port | Notes |
|---------|---------|------|-------|
| vLLM server | `make vllm-serve` | 8001 | Must start before RAG queries; see gotchas below |
| FastAPI dev server | `make run` | 8000 | Auto-reloads on code changes |
| Tests | `make test` | — | All tests work without vLLM running |
| Lint | `make lint` | — | ruff check + format check |

### Startup Order

1. Start vLLM first (`make vllm-serve`) — takes ~3-5 min on CPU to load model
2. Then start FastAPI (`make run`)
3. `/api/v1/documents` and `/api/v1/search` work without vLLM
4. `/api/v1/query` requires vLLM (or OpenAI key if `LLM_BACKEND=openai`)

### Key Gotchas

- **vLLM CPU startup is slow:** First launch on CPU takes 3-5 minutes for model loading + torch JIT. Use `TORCH_COMPILE_DISABLE=1` to skip torch.compile and speed up startup.
- **python3.12-dev required:** vLLM CPU uses torch inductor which compiles C++ kernels needing `Python.h`. Install `sudo apt-get install python3.12-dev` before running vLLM.
- **TCMalloc + Intel OpenMP:** For best CPU performance, set `LD_PRELOAD` with libtcmalloc and libiomp5. Paths: `/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4*` and `~/.local/lib/libiomp5.so`.
- **vLLM env vars:** `VLLM_CPU_KVCACHE_SPACE=4` (GB for KV cache), `VLLM_CPU_OMP_THREADS_BIND=0-3` (bind to CPU cores).
- **Qdrant in-memory by default:** Data is lost on restart. Set `QDRANT_URL` to use persistent storage.
- **qdrant-client v1.12+:** Uses `query_points()` not deprecated `search()`.
- **FastEmbed first-run download:** `BAAI/bge-small-en-v1.5` (~130MB) downloads on first use to `~/.cache/fastembed/`.
- **Model download:** Run `huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct` to get the model (~950MB).
- **Ruff binary:** After pip install, ruff is at `~/.local/bin/ruff`. Ensure `PATH` includes `$HOME/.local/bin`.
### Repository Overview
- **Project:** Advanced-RAG (Retrieval-Augmented Generation)
- **Status:** Newly initialized repository with only a `README.md`. No source code, dependencies, or services exist yet.

### Environment
- **Python:** 3.12.3 (system, `/usr/bin/python3`)
- **Node.js:** v22.22.1 (available if needed for frontend tooling)
- **Git:** 2.43.0
- **OS:** Ubuntu (linux 6.1.x)

### Notes for Future Agents
- As of this setup, the repository has no dependencies, build system, or runnable services. The update script is a no-op placeholder (`echo "No dependencies to install yet"`).
- When code is added, update the update script via `SetupVmEnvironment` to install actual dependencies (e.g., `pip install -r requirements.txt` or `uv sync`).
- For a RAG project, common dependencies will likely include: `langchain`, `openai`, `faiss-cpu` or `chromadb`, `fastapi`/`flask`, etc. Adjust accordingly once `requirements.txt` or `pyproject.toml` is committed.
