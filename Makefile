.PHONY: install run test lint format format-check check

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

run:
	uvicorn app.main:create_app --factory --reload

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

check:
	ruff check .
	ruff format --check .
	pytest