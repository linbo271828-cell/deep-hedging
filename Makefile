.PHONY: install test lint format typecheck reproduce reproduce-core reproduce-report clean run-backend run-frontend install-web

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

# Reproduce the v0.1 GBM-call baseline results (fast: no Heston, no multi-payoff).
reproduce-core: test
	PYTHONPATH=. python experiments/01_validate_market.py
	PYTHONPATH=. python experiments/02_classical_delta.py
	PYTHONPATH=. python experiments/03_neural_no_costs.py
	PYTHONPATH=. python experiments/04_neural_with_costs.py
	PYTHONPATH=. python experiments/05_cost_frontier.py

# Regenerate the markdown report from saved results artifacts.
# Requires reproduce-core to have been run first (or results/ to be populated).
reproduce-report:
	PYTHONPATH=. python -c "from src.reporting.report import generate_report; generate_report('results', 'results/reports/report.md')"

# Legacy target: same as reproduce-core for backward compatibility.
reproduce: reproduce-core

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -f plots/*.png plots/*.pdf

# --- Web / API local dev ---
# Run the FastAPI backend on port 8000 (in a separate terminal).
run-backend:
	PYTHONPATH=. .venv/bin/uvicorn services.sim.main:app --reload --port 8000

# Run the FastAPI backend in production mode (no --reload, binds 0.0.0.0).
# Override port with PORT=XXXX make run-backend-prod
run-backend-prod:
	PYTHONPATH=. PORT=$${PORT:-8000} .venv/bin/uvicorn services.sim.main:app --host 0.0.0.0 --port $${PORT:-8000}

# Run the Next.js frontend on port 3000 (in a separate terminal).
run-frontend:
	cd apps/web && npm run dev

# Install Next.js dependencies.
install-web:
	cd apps/web && npm install
