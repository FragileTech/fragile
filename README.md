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

## QFT Ensemble Validation

Run multi-trial statistical validation of QFT predictions with full Standard Model gauge group (U(1) × SU(2) × SU(3)):

```bash
python src/experiments/run_qft_validation_ensemble.py \
  --n-trials 100 \
  --n-walkers 1000 \
  --n-steps 1000 \
  --parallel-jobs 50 \
  --use-viscous-coupling \
  --nu 0.1 \
  --viscous-length-scale 1.0 \
  --output-dir outputs/qft_ensemble
```

Quick test (runs in ~30 seconds):

```bash
python src/experiments/run_qft_validation_ensemble.py \
  --n-trials 5 \
  --n-walkers 100 \
  --n-steps 200 \
  --parallel-jobs 2 \
  --use-viscous-coupling \
  --output-dir outputs/qft_ensemble_test
```

Outputs:
- `ensemble_report.md`: Summary with 95% confidence intervals
- `ensemble_metrics.json`: Full statistics for all metrics
- `plots/`: Correlation lengths, Wilson loops, phase distributions, SU(3) color alignment

## License

MIT
