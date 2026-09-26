# Million-evaluation population benchmark

`benchmark-million.py` runs 20-dimensional Wave comparisons with five elites at
8, 16, 32, 64, 128, 256, 512 and 1,000 walkers. Defaults cover quadratic,
Rastrigin, Rosenbrock and all 24 noiseless BBOB functions (instance 1), with five
seeds. Methods are Gaussian, local covariance, bounded adaptive, cloning-guided,
cloning-guided with CMA boundary repair, and BIPOP-active CMA-ES. CMA controls its own population, so its reference is run
once per function and seed. The full default matrix contains 5,535 runs.

Each run admits at most 1,000,000 objective evaluations. It stops when
`abs(best_objective - reference_minimum) <= 10e-6` after initialization or a
complete engine step. Complete operations are never split to reach an exact
budget. Five elites apply to Wave; CMA retains its existing selection rules.

## Precision

The default `--precision fp64` checks the native engine's `fgo_precision()`
metadata before starting. Currently, fractal search coordinates and fitness are
FP32; objectives are FP64. CMA search coordinates are FP64. Float64 exported
snapshots do not upgrade the internal search arithmetic.

Consequently, the current engine supports the FP64 CMA reference, but the full
FP64 matrix is deliberately rejected. To explicitly measure existing fractal
behavior, `--precision mixed` records the real storage precision in every result
and the report. It does not change engine arithmetic.

```sh
# FP64 reference supported by the current engine.
python3 fractal-gas-web/tools/benchmark-million.py --methods cma \
  --output fractal-gas-web/tests/optimization/reports/million-20d-cma-fp64

# Inspect the mixed-precision matrix without running it.
python3 fractal-gas-web/tools/benchmark-million.py --precision mixed --dry-run \
  --output /tmp/million-mixed

# Verify stopping and precision rejection against the compiled native engine.
python3 fractal-gas-web/tests/optimization/test_benchmark_target.py
```

`runs.jsonl` saves each completed run, including its effective configuration,
evaluation count, precision, errors and final diagnostics. `results.csv` and
`report.md` summarize target success and median objective error, evaluations and
process CPU time. Failed runs are retained. Concurrent timing is not isolated
latency. Sources and the engine library are fingerprinted; `--resume` refuses
changed implementations or settings. Existing results are never overwritten.

The boundary comparison uses `--methods cloning_cma cloning --precision mixed`: 2,160 runs on the same current engine, with post-movement elite restoration in both variants. `cloning_cma` is fractal Wave with cloning-guided perturbations and boundary repair; it does not select the CMA-ES optimizer.
