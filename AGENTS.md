# Repository Guidelines

## Project Structure & Module Organization
- `src/fragile/` is the installable package (built by Hatch).
- `src/` also holds supporting modules and experiments (for example `src/experiments/`, `src/dataviz.py`, `src/mathster/`).
- `tests/` contains pytest suites.
- `docs/` is the Jupyter Book/Sphinx site.
- `examples/`, `media/`, and `outputs/` hold sample notebooks, assets, and generated artifacts.

## Volume 3 Proof Standards
- When editing `docs/source/3_fractal_gas`, ground claims in Volume 3 metatheorems and appendices; cite internal theorems/permits explicitly.
- Avoid generic textbook lattice arguments or external documents unless the user requests them; prefer Fractal Set/QSD constructions.
- Do not introduce new assumptions or change the algorithm; if a claim relies on a permit/certificate, point to where it is certified (for example `docs/source/3_fractal_gas/1_the_algorithm/02_fractal_gas_latent.md`).
- For unbounded spaces, use the established confining envelope or Safe Harbor results instead of adding compactness assumptions.

## Build, Test, and Development Commands
Use uv to run Hatch environments:
- `uv run hatch run test:test` runs the pytest suite.
- `uv run hatch run test:doctest` runs doctests in `src/` and Markdown.
- `uv run hatch run test:cov` runs coverage with `pytest-cov`.
- `uv run hatch run lint:all` runs ruff check + format.
- `uv run hatch run lint:check` runs lint + format diff only.
- `uv run hatch run docs:build` / `uv run hatch run docs:docs` builds or builds+serves docs.
- `uv run hatch run lint:all && uv run hatch run docs:build && uv run hatch run test:test` runs lint + docs build + tests.

Equivalent Hatch commands live under `hatch run lint:*`, `hatch run test:*`, and `hatch run docs:*`.

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
