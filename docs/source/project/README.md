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

- `uv sync --all-extras` (recommended)
- `uv pip install -e .`
- Tooling: `uv tool install hatch` (one-time, for `uv run hatch ...` commands)

## Documentation (Jupyter Book)

- Build: `uv run hatch run docs:build`
- Build and serve: `uv run hatch run docs:docs`

## Development

- Tests: `uv run hatch run test:test`
- Doctests: `uv run hatch run test:doctest`
- Lint/format: `uv run hatch run lint:all`

## QFT Calibration

```bash
python src/experiments/calibrate_fractal_gas_qft.py \
  --history-path outputs/fractal_gas_potential_well/20260123_164153_history.pt \
  --m-gev 91.1876 \
  --hbar-eff 1.0 \
  --d 3
```

## License

MIT
