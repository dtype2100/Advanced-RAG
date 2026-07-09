.PHONY: install dev lint format test run vllm-serve worker evals clean

PYTHON ?= python3
export PATH := $(HOME)/.local/bin:$(PATH)

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check app/ tests/ evals/ scripts/
	$(PYTHON) -m ruff format --check app/ tests/ evals/ scripts/

format:
	$(PYTHON) -m ruff check --fix app/ tests/ evals/ scripts/
	$(PYTHON) -m ruff format app/ tests/ evals/ scripts/

test:
	$(PYTHON) -m pytest -v

test-unit:
	$(PYTHON) -m pytest -v tests/unit/

test-integration:
	$(PYTHON) -m pytest -v tests/integration/

run:
	$(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	$(PYTHON) -m arq app.workers.settings.WorkerSettings

vllm-serve:
	@echo "Starting vLLM server on port 8001..."
	VLLM_CPU_KVCACHE_SPACE=4 \
	VLLM_CPU_OMP_THREADS_BIND=0-3 \
	$(PYTHON) -m vllm.entrypoints.openai.api_server \
		--model $(or $(VLLM_MODEL_PATH),/workspace/models/Qwen2.5-0.5B-Instruct) \
		--served-model-name Qwen/Qwen2.5-0.5B-Instruct \
		--host 0.0.0.0 \
		--port 8001 \
		--max-model-len $(or $(VLLM_MAX_MODEL_LEN),2048) \
		--dtype bfloat16

evals:
	$(PYTHON) scripts/run_evals.py

evals-ci:
	IMPROVEMENT_LOOP_CI=1 $(PYTHON) scripts/run_evals.py --ci

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
