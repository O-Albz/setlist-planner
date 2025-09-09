.PHONY: setup lint format test run precommit

setup:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .

format:
	black .

test:
	pytest

run:
	streamlit run src/setlistgraph/ui/app.py

precommit:
	pre-commit run --all-files
