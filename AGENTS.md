# Repository Guidelines

## Project Structure & Module Organization
- `src/fragile/` is the installable package (built by Hatch).
- `src/fragile/fractalai/` contains the Fractal Gas/Fractal Set implementation, CLI, and experiments.
- `src/fragile/core/` holds Volume 1 deep learning layers, losses, and optimizers.
- `src/fragile/theory/` and `src/fragile/fractalai/theory/` hold analysis utilities and sweeps.
- `src/experiments/` and `src/fragile/fractalai/experiments/` contain notebooks, dashboards, and analysis scripts.
- `fragile_old/` is legacy code kept for reference.
- `tests/` contains pytest suites.
- `docs/` is the Jupyter Book (source in `docs/source/`, build output in `docs/_build/`, project docs in `docs/source/project/`).
- `examples/`, `media/`, and `outputs/` hold sample notebooks, assets, and generated artifacts.

## Current Work Expectations
- Expect to both improve the Jupyter Book content in `docs/` and implement or update code in `src/` as part of day-to-day tasks.
- Keep documentation changes and code changes aligned (APIs, examples, and narratives should match actual behavior).

## Theory-Driven Development Focus
- New deep learning algorithms should follow `docs/source/1_agent` and integrate with `src/fragile/core/` (layers, losses, optimizers), keeping notation and units aligned with the book.
- Production-grade Fractal Gas work extends `src/fragile/fractalai/` (especially `core/` modules like `euclidean_gas.py`, `cloning.py`, `kinetic_operator.py`, `fractal_set.py`, `history.py`) and should stay faithful to `docs/source/3_fractal_gas`.

## Current Implementation Principles
- Favor vectorized Torch ops with standard shapes: swarm state `[N, d]`, history arrays `[n_recorded, N, ...]`.
- Keep algorithm components modular (companion selection, fitness, cloning, kinetics) and configured via `param`/`PanelModel`.
- Preserve trace structures: `RunHistory` (Pydantic) for serialized runs and `FractalSet` for CST/IG/IA graph construction with scalar node attributes per Volume 3.

## Volume 3 Proof Standards
- When editing `docs/source/3_fractal_gas`, ground claims in Volume 3 metatheorems and appendices; cite internal theorems/permits explicitly.
- Avoid generic textbook lattice arguments or external documents unless the user requests them; prefer Fractal Set/QSD constructions.
- Do not introduce new assumptions or change the algorithm; if a claim relies on a permit/certificate, point to where it is certified (for example `docs/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md`).
- For unbounded spaces, use the established confining envelope or Safe Harbor results instead of adding compactness assumptions.

## Documentation Workflow & Style
- The Jupyter Book lives in `docs/source/`; follow `docs/CLAUDE.md` for heading/label conventions, Feynman prose rules, admonition classes, notation, and cross-references.
- When adding Feynman prose, use the `feynman-jupyter-educator` agent described in `docs/CLAUDE.md`; do not hand-edit Feynman blocks.
- For efficient updates, prefer the helper scripts in `docs/` (for example `add_subsection_labels.py`, `convert_section_refs.py`, `fix_transitions.py`, `collect_prf_directives.py`) over manual bulk edits.
- Expect to write, review, and improve docs in `docs/` and `docs/source/` alongside code changes, keeping `docs/source/project/` in sync with repo-level docs where applicable.

## Tooling & MCP Usage
- Use Claude and Gemini MCP tools only when the user explicitly instructs you to do so.

## Build, Test, and Development Commands
Use `uv run` directly (no Hatch) to invoke tooling:
- `uv run pytest -s -o log_cli=true -o log_cli_level=info tests` runs the pytest suite.
- `uv run pytest -s -o log_cli=true -o log_cli_level=info --doctest-modules --doctest-glob="*.md" -n 0 src` runs doctests in `src/` and Markdown.
- `uv run pytest -s -o log_cli=true -o log_cli_level=info -n auto --cov-report=term-missing --cov-config=pyproject.toml --cov=src/fragile --cov=tests` runs coverage with `pytest-cov`.
- `uv run ruff check .` runs ruff check; `uv run ruff format --diff .` shows format diff only.
- `uv run ruff check --fix-only --unsafe-fixes . && uv run ruff format .` applies ruff fixes + formatting.
- `uv run mypy --install-types --non-interactive src/fragile tests` runs mypy.
- `uv run jupyter-book build docs/` builds the Jupyter Book (ensure `make prompt` and copies into `docs/source/project/` are up to date).
- `uv run python3 -m http.server --directory docs/_build/html` serves the built docs.
- `uv run jupyter-book config sphinx docs/ --overwrite && uv run sphinx-build -b html docs/ docs/_build/html` builds with Sphinx directly.
- `uv run linkchecker --config .linkcheckerrc --ignore-url=/reference --ignore-url=None site` runs link validation.

## Coding Style & Naming Conventions
- Python 3.10, 4-space indentation.
- Ruff is the formatter/linter; line length is 99; docstrings use double quotes.
- Avoid relative imports (ruff enforces) and use conventional aliases like `import numpy as np`.
- Mypy type checks `src/fragile` and `tests`.

## Testing Guidelines
- Framework: pytest with files named `test_*.py`.
- Unit tests live in `tests/`; prefer small, deterministic tests.
- Coverage uses `pytest-cov` via `uv run hatch run test:cov`.
- Doctests run via `uv run hatch run test:doctest` (Markdown and modules).

## Commit & Pull Request Guidelines
- Commit history uses short, imperative, sentence-case messages (for example "Update docs", "Fix docs deploy"); no conventional prefixes.
- PRs should include passing tests (`uv run hatch run test:test` or `hatch run test:test`), updated docs when APIs change, a `CHANGELOG.md` entry, and your name in `AUTHORS.md`.
- Link issues for bug fixes or features when available.
