.PHONY: install dev lint format test run vllm-serve worker evals clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check app/ tests/ evals/ scripts/
	ruff format --check app/ tests/ evals/ scripts/

format:
	ruff check --fix app/ tests/ evals/ scripts/
	ruff format app/ tests/ evals/ scripts/

test:
	pytest -v

test-unit:
	pytest -v tests/unit/

test-integration:
	pytest -v tests/integration/

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	arq app.workers.settings.WorkerSettings

vllm-serve:
	@echo "Starting vLLM server on port 8001..."
	VLLM_CPU_KVCACHE_SPACE=4 \
	VLLM_CPU_OMP_THREADS_BIND=0-3 \
	python3 -m vllm.entrypoints.openai.api_server \
		--model $(or $(VLLM_MODEL_PATH),/workspace/models/Qwen2.5-0.5B-Instruct) \
		--served-model-name Qwen/Qwen2.5-0.5B-Instruct \
		--host 0.0.0.0 \
		--port 8001 \
		--max-model-len $(or $(VLLM_MAX_MODEL_LEN),2048) \
		--dtype bfloat16

evals:
	python scripts/run_evals.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
