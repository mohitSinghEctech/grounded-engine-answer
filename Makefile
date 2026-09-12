.PHONY: help install run test lint format format-check check up down logs build

GATEWAY := services/llm-gateway

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
	python -m pip install --upgrade pip
	pip install -r $(GATEWAY)/requirements.txt

run:
	cd $(GATEWAY) && uvicorn app.main:create_app --factory --reload

test:
	cd $(GATEWAY) && pytest

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

check: lint format-check test

build:
	docker compose build

up:
	docker compose up

down:
	docker compose down

logs:
	docker compose logs -f
