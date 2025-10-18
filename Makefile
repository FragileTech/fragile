.PHONY: help install sync clean clean-docs test cov no-cov doctest debug style check lint typing build-docs serve-docs docs sphinx pdf all

# Default target - show help
help:
	@echo "Fragile Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install project with dependencies using uv"
	@echo "  make sync         - Sync dependencies"
	@echo "  make clean        - Clean build artifacts and caches"
	@echo "  make clean-docs   - Clean documentation build artifacts"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run tests"
	@echo "  make cov          - Run tests with coverage"
	@echo "  make no-cov       - Run tests without coverage (faster)"
	@echo "  make doctest      - Run doctests"
	@echo "  make debug        - Run tests with IPython debugger"
	@echo ""
	@echo "Linting:"
	@echo "  make style        - Format code with ruff"
	@echo "  make check        - Check code without fixing"
	@echo "  make typing       - Run mypy type checking"
	@echo "  make lint         - Run all lint checks (style + check + typing)"
	@echo ""
	@echo "Documentation:"
	@echo "  make build-docs   - Build Jupyter Book documentation (auto-converts mermaid blocks)"
	@echo "  make serve-docs   - Serve documentation (after building)"
	@echo "  make docs         - Build and serve documentation"
	@echo "  make sphinx       - Build with Sphinx directly"
	@echo "  make pdf          - Build PDF documentation (requires LaTeX)"
	@echo ""
	@echo "Complete Workflow:"
	@echo "  make all          - Run lint, build docs, and test"

# Installation and setup
install:
	uv sync

sync:
	uv sync

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-docs:
	rm -rf docs/_build

# Testing commands
test:
	uv run --group dev hatch run test:test

cov:
	uv run --group dev hatch run test:cov

no-cov:
	uv run --group dev hatch run test:no-cov

doctest:
	uv run --group dev hatch run test:doctest

debug:
	uv run --group dev hatch run test:debug

# Linting commands
style:
	uv run --group dev hatch run lint:style

check:
	uv run --group dev hatch run lint:check

typing:
	uv run --group dev hatch run lint:typing

lint:
	uv run --group dev hatch run lint:all

# Documentation commands
build-docs:
	uv run hatch run docs:build

serve-docs:
	uv run hatch run docs:serve

docs:
	uv run hatch run docs:docs

sphinx:
	uv run hatch run docs:sphinx

pdf:
	uv run hatch run docs:pdf
	@echo "✓ PDF generated at docs/_build/latex/book.pdf"

# Complete workflow
all: lint build-docs test
	@echo "✓ All checks passed!"
