.PHONY: install test lint format typecheck reproduce clean

install:
	python -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:
	PYTHONPATH=. pytest

lint:
	PYTHONPATH=. ruff check src tests experiments

format:
	PYTHONPATH=. ruff check --fix src tests experiments
	PYTHONPATH=. black src tests experiments

typecheck:
	PYTHONPATH=. mypy src

reproduce: test
	PYTHONPATH=. python experiments/01_validate_market.py
	PYTHONPATH=. python experiments/02_classical_delta.py
	PYTHONPATH=. python experiments/03_neural_no_costs.py
	PYTHONPATH=. python experiments/04_neural_with_costs.py
	PYTHONPATH=. python experiments/05_cost_frontier.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f plots/*.png plots/*.pdf
