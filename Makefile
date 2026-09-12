.PHONY: help install run test lint format format-check check up down logs build

GATEWAY := services/llm-gateway
TAX_AGENT := services/tax-agent

# Absolute path, so targets work whether or not the venv is activated
# and survive the `cd` in the test/run targets.
VENV ?= .venv
PY := $(CURDIR)/$(VENV)/bin/python

help:
	@echo "install        install llm-gateway dependencies"
	@echo "run            run llm-gateway locally with reload"
	@echo "test           run llm-gateway tests"
	@echo "lint           ruff check across the repo"
	@echo "format         ruff format across the repo"
	@echo "check          lint + format check + tests"
	@echo "build          docker compose build"
	@echo "up / down      start / stop all services"
	@echo "logs           follow all service logs"

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r $(GATEWAY)/requirements.txt

run:
	cd $(GATEWAY) && $(PY) -m uvicorn app.main:create_app --factory --reload

test:
	cd $(GATEWAY) && $(PY) -m pytest

lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

format-check:
	$(PY) -m ruff format --check .

check: lint format-check test

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f
