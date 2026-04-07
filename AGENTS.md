# AGENTS.md

## Cursor Cloud specific instructions

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
