.PHONY: install dev lint typecheck test clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

typecheck:
	mypy src/

test:
	pytest

test-cov:
	pytest --cov=trading_agent --cov-report=term-missing

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
