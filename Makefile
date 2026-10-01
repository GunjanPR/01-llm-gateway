# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python

.PHONY: install lint format test run

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

test:
	$(PYTHON) -m pytest

run:
	$(PYTHON) -m uvicorn llm_gateway.main:app --reload --host 0.0.0.0 --port 8000
