.PHONY: install install-all dev lint format test run vllm-serve clean

install:
	pip install -e .

install-all:
	pip install -e ".[all]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check app/ tests/
	ruff format --check app/ tests/

format:
	ruff check --fix app/ tests/
	ruff format app/ tests/

test:
	pytest -v

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
