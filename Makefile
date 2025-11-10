.PHONY: help install sync clean clean-docs test cov no-cov doctest debug style check lint typing build-docs serve-docs docs sphinx pdf transform-enriched build-registry build-chapter-registries build-all-registries clean-registries all update

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
	@echo "Mathematical Registries:"
	@echo "  make transform-enriched        - Transform all refined_data to pipeline_data format"
	@echo "  make build-registry            - Build unified combined_registry from all chapters"
	@echo "  make build-chapter-registries  - Build individual chapter registries"
	@echo "  make build-all-registries      - Full pipeline: transform → build unified registry"
	@echo "  make clean-registries          - Clean all registry outputs (pipeline_data and registries)"
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

# Mathematical Registry commands
# Transform enriched/refined data to optimized pipeline format
transform-enriched:
	@echo "Transforming refined_data to pipeline_data format..."
	@echo "Chapter 1: Euclidean Gas Framework"
	uv run python -m fragile.proofs.tools.enriched_to_math_types \
		--input docs/source/1_euclidean_gas/01_fragile_gas_framework/refined_data \
		--output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data
	@echo "✓ Transformation complete"
	@echo ""
	@echo "NOTE: Add more chapters here as refined_data becomes available:"
	@echo "  Chapter 2: docs/source/2_geometric_gas/<document>/refined_data"
	@echo "  Chapter 3: docs/source/3_brascamp_lieb/<document>/refined_data"

# Build unified registry from all chapters (auto-discovers all refined_data)
build-registry:
	@echo "Building unified combined_registry from all chapters..."
	@echo "(Automatically discovers all refined_data directories)"
	uv run python -m fragile.proofs.tools.build_refined_registry \
		--docs-root docs/source \
		--output combined_registry
	@echo "✓ Combined registry built at combined_registry/"

# Build individual per-chapter registries
build-chapter-registries:
	@echo "Building individual chapter registries..."
	@echo "Chapter 1: Euclidean Gas Framework"
	uv run python -m fragile.proofs.tools.build_pipeline_registry \
		--pipeline-dir docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_data \
		--output docs/source/1_euclidean_gas/01_fragile_gas_framework/pipeline_registry
	@echo "✓ Chapter registries built"
	@echo ""
	@echo "NOTE: Add more chapters here as pipeline_data becomes available"

# Clean all registry outputs and pipeline data
clean-registries:
	@echo "Cleaning registry outputs..."
	rm -rf combined_registry/ refined_registry/
	find docs/source -type d -name pipeline_registry -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Registry outputs cleaned"

# Full pipeline: transform → build unified registry
build-all-registries: transform-enriched build-registry
	@echo "✓ Complete registry pipeline finished!"

# Complete workflow
all: lint build-docs test
	@echo "✓ All checks passed!"

update:
	uv run mathster parse && uv run mathster preprocess && uv run mathster registry
