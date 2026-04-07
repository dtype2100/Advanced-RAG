.PHONY: install dev lint format test run clean

install:
	pip install -e .

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

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
