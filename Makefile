# Windows: install make with `choco install make`, or run from Git Bash.
PYTHON ?= python
KEY ?=

.PHONY: install lint format test test-integration run up up-all down logs bench demo

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

# Needs the stack running (make up + make run) and GATEWAY_TEST_KEY set
test-integration:
	$(PYTHON) -m pytest -m integration

# Gateway on the host (talks to native Ollama on :11434)
run:
	$(PYTHON) -m llm_gateway --reload

# Redis + PostgreSQL + Prometheus + Grafana in Docker, for use with `make run`
up:
	docker compose up -d redis postgres jaeger prometheus grafana

# Everything in Docker, including the gateway
up-all:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

# Baseline (direct to Ollama) vs gateway, cache bypassed. Usage: make bench KEY=<gateway key>
bench:
	$(PYTHON) scripts/bench.py --label direct --base-url http://localhost:11434/v1 --requests 50 --concurrency 4
	$(PYTHON) scripts/bench.py --label gateway --base-url http://localhost:8000/v1 --api-key $(KEY) --requests 50 --concurrency 4 --no-cache
	$(PYTHON) scripts/bench.py --label gateway-cache --base-url http://localhost:8000/v1 --api-key $(KEY) --requests 50 --concurrency 4 --repeat-ratio 0.5
	$(PYTHON) scripts/bench.py --label gateway-stream --base-url http://localhost:8000/v1 --api-key $(KEY) --requests 30 --concurrency 2 --stream --no-cache

# Scripted walkthrough of every feature against the running stack. Usage: make demo KEY=<gateway key>
demo:
	$(PYTHON) scripts/demo.py --api-key $(KEY) --markdown docs/demo.md
