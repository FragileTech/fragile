.PHONY: style check test

style:
	uv run ruff check --fix-only --unsafe-fixes .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --diff .

test:
	uv run pytest tests/
