# Fragile

Fragile is a research codebase for the Fractal Gas optimization algorithm and the
Fractal Set data structure, with a Jupyter Book that captures the theory, derivations,
and experiments.

## What This Repo Contains

- `src/fragile/`: installable package (core implementation lives in `src/fragile/fractalai/`)
- `docs/`: Jupyter Book (source in `docs/source/`, build output in `docs/_build/`)
- `tests/`: pytest suites
- `examples/`, `media/`, `outputs/`: notebooks, assets, generated artifacts

## Code Map

- `src/fragile/fractalai/core/`: companion selection, fitness, cloning, kinetics,
  and the Fractal Set data structure
- `src/fragile/fractalai/experiments/`: simulations, dashboards, convergence studies
- `src/fragile/fractalai/experiments/gauge/`: U(1) and SU(2) gauge symmetry tests
- `src/fragile/fractalai/theory/`: theory utilities and analysis scripts
- CLI entry point: `src/fragile/fractalai/cli.py`

## Install

- `rye sync` (recommended)
- `pip install -e .`

## Documentation (Jupyter Book)

- Build: `rye run build-docs`
- Build and serve: `rye run docs`

## Development

- Tests: `rye run test`
- Doctests: `rye run doctest`
- Lint/format: `rye run lint`

## License

MIT
