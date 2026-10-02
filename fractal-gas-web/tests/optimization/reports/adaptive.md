# Adaptive exploration: measured comparison

10 seeds per case; 5000 evaluation allowance; two dimensions; initial swarm population 8 (CMA-ES uses its default population); maximum swarm population 64. CPU native build. Each method admits complete operations, so actual evaluation counts can fall below the common allowance.

The Gaussian and Local covariance rows are the pre-existing perturbation baselines. All fractal variants use Wave with identical initial population, bounds, seed, and evaluation allowance. BIPOP-active CMA-ES is an external reference. Fixed-mixture rows are ablations of the new strategy. Basin matching is heuristic; archived-basin counts are not certified local minima. On the stochastic control the expectation is zero everywhere, so raw observed best measures noise extremes and is excluded from improvement claims.

| Problem | Variant | Median regret | Median raw best | Mean seconds | Mean evaluations | Mean archived basins |
|---|---|---:|---:|---:|---:|---:|
| quadratic | Gaussian | 4.17127e-06 | 4.17127e-06 | 0.0838 | 5000.0 | 0.0 |
| quadratic | Local covariance | 2.78756e-06 | 2.78756e-06 | 1.2898 | 5000.0 | 0.0 |
| quadratic | Paths / fixed mixture | 0.000182558 | 0.000182558 | 0.1279 | 4992.0 | 0.0 |
| quadratic | Active / fixed mixture | 0.000225593 | 0.000225593 | 0.1184 | 4992.0 | 0.0 |
| quadratic | Adaptive mixture | 8.92849e-05 | 8.92849e-05 | 0.1067 | 4981.3 | 0.0 |
| quadratic | Restarts / no memory | 2.28864e-05 | 2.28864e-05 | 1.5462 | 4943.0 | 33.2 |
| quadratic | Restarts / basin memory | 5.81329e-06 | 5.81329e-06 | 1.3880 | 4920.1 | 30.4 |
| quadratic | BIPOP-active CMA-ES | 1.05229e-18 | 1.05229e-18 | 0.0809 | 4974.2 | 0.0 |
| bbob_10 | Gaussian | 0.170699 | -54.7693 | 0.1362 | 5000.0 | 0.0 |
| bbob_10 | Local covariance | 0.354318 | -54.5857 | 1.3661 | 5000.0 | 0.0 |
| bbob_10 | Paths / fixed mixture | 0.871958 | -54.068 | 0.2307 | 4992.0 | 0.0 |
| bbob_10 | Active / fixed mixture | 0.700367 | -54.2396 | 0.2313 | 4992.0 | 0.0 |
| bbob_10 | Adaptive mixture | 1.1651 | -53.7749 | 0.1741 | 4981.6 | 0.0 |
| bbob_10 | Restarts / no memory | 0.355169 | -54.5848 | 3.1405 | 4900.4 | 42.1 |
| bbob_10 | Restarts / basin memory | 0.224834 | -54.7152 | 2.8109 | 4894.9 | 44.9 |
| bbob_10 | BIPOP-active CMA-ES | 0 | -54.94 | 0.1411 | 4990.2 | 0.0 |
| rastrigin | Gaussian | 0.0800639 | 0.0800639 | 0.1355 | 5000.0 | 0.0 |
| rastrigin | Local covariance | 0.0401009 | 0.0401009 | 1.6176 | 5000.0 | 0.0 |
| rastrigin | Paths / fixed mixture | 0.255069 | 0.255069 | 0.1948 | 4992.0 | 0.0 |
| rastrigin | Active / fixed mixture | 0.389248 | 0.389248 | 0.1978 | 4992.0 | 0.0 |
| rastrigin | Adaptive mixture | 0.482094 | 0.482094 | 0.2167 | 4980.0 | 0.0 |
| rastrigin | Restarts / no memory | 0.252922 | 0.252922 | 2.3807 | 4887.2 | 42.7 |
| rastrigin | Restarts / basin memory | 0.0930465 | 0.0930465 | 2.1692 | 4873.0 | 46.9 |
| rastrigin | BIPOP-active CMA-ES | 0.0515929 | 0.0515929 | 0.1099 | 4990.9 | 0.0 |
| bbob_5 | Gaussian | 0.106094 | -9.10391 | 0.1211 | 5000.0 | 0.0 |
| bbob_5 | Local covariance | 0.112253 | -9.09775 | 1.5123 | 5000.0 | 0.0 |
| bbob_5 | Paths / fixed mixture | 0.365258 | -8.84474 | 0.2308 | 4976.0 | 0.0 |
| bbob_5 | Active / fixed mixture | 0.373895 | -8.8361 | 0.2615 | 4992.0 | 0.0 |
| bbob_5 | Adaptive mixture | 0.441511 | -8.76849 | 0.1759 | 4979.5 | 0.0 |
| bbob_5 | Restarts / no memory | 0.243946 | -8.96605 | 3.0997 | 4932.4 | 44.0 |
| bbob_5 | Restarts / basin memory | 0.245702 | -8.9643 | 2.8610 | 4912.9 | 50.0 |
| bbob_5 | BIPOP-active CMA-ES | 0 | -9.21 | 0.1045 | 4992.1 | 0.0 |
| stochastic_gaussian | Gaussian | n/a | -3.73441 | 0.1213 | 5000.0 | 0.0 |
| stochastic_gaussian | Local covariance | n/a | -3.61433 | 1.8376 | 5000.0 | 0.0 |
| stochastic_gaussian | Paths / fixed mixture | n/a | -3.47964 | 0.0398 | 4952.0 | 0.0 |
| stochastic_gaussian | Active / fixed mixture | n/a | -3.51969 | 0.0489 | 4952.0 | 0.0 |
| stochastic_gaussian | Adaptive mixture | n/a | -3.47049 | 0.0414 | 4948.7 | 0.0 |
| stochastic_gaussian | Restarts / no memory | n/a | -3.5261 | 0.3925 | 4973.1 | 20.6 |
| stochastic_gaussian | Restarts / basin memory | n/a | -3.57786 | 0.3646 | 4975.4 | 20.6 |
| stochastic_gaussian | BIPOP-active CMA-ES | n/a | -3.7279 | 0.1782 | 4997.0 | 0.0 |

## Improvement over previous perturbations

Positive percentages and positive log10 values mean lower median regret. Paired wins compare the same seeds; ties use a relative tolerance of 1e-12.

| Problem | New variant | Previous perturbation | Median-regret change | Orders of magnitude | Paired wins / ties / losses | Runtime ratio |
|---|---|---|---:|---:|---:|---:|
| quadratic | Adaptive mixture | Gaussian | -2040.5% | -1.33 | 1 / 0 / 9 | 1.27x |
| quadratic | Adaptive mixture | Local covariance | -3103.0% | -1.51 | 0 / 0 / 10 | 0.08x |
| quadratic | Restarts / basin memory | Gaussian | -39.4% | -0.14 | 3 / 0 / 7 | 16.57x |
| quadratic | Restarts / basin memory | Local covariance | -108.5% | -0.32 | 2 / 0 / 8 | 1.08x |
| bbob_10 | Adaptive mixture | Gaussian | -582.5% | -0.83 | 2 / 0 / 8 | 1.28x |
| bbob_10 | Adaptive mixture | Local covariance | -228.8% | -0.52 | 2 / 0 / 8 | 0.13x |
| bbob_10 | Restarts / basin memory | Gaussian | -31.7% | -0.12 | 5 / 0 / 5 | 20.64x |
| bbob_10 | Restarts / basin memory | Local covariance | +36.5% | +0.20 | 5 / 0 / 5 | 2.06x |
| rastrigin | Adaptive mixture | Gaussian | -502.1% | -0.78 | 2 / 0 / 8 | 1.60x |
| rastrigin | Adaptive mixture | Local covariance | -1102.2% | -1.08 | 2 / 0 / 8 | 0.13x |
| rastrigin | Restarts / basin memory | Gaussian | -16.2% | -0.07 | 3 / 0 / 7 | 16.00x |
| rastrigin | Restarts / basin memory | Local covariance | -132.0% | -0.37 | 3 / 0 / 7 | 1.34x |
| bbob_5 | Adaptive mixture | Gaussian | -316.2% | -0.62 | 0 / 0 / 10 | 1.45x |
| bbob_5 | Adaptive mixture | Local covariance | -293.3% | -0.59 | 0 / 0 / 10 | 0.12x |
| bbob_5 | Restarts / basin memory | Gaussian | -131.6% | -0.36 | 2 / 0 / 8 | 23.63x |
| bbob_5 | Restarts / basin memory | Local covariance | -118.9% | -0.34 | 1 / 0 / 9 | 1.89x |

## Aggregate deterministic comparison

The regret factor is the geometric mean of the four per-problem median-regret ratios. Values above 1 mean worse objective quality. Runtime uses the corresponding geometric mean.

| New variant | Previous perturbation | Regret factor | Runtime factor | Total paired wins / ties / losses |
|---|---|---:|---:|---:|
| Adaptive mixture | Gaussian | 7.78x | 1.39x | 5 / 0 / 35 |
| Adaptive mixture | Local covariance | 8.40x | 0.11x | 4 / 0 / 36 |
| Restarts / basin memory | Gaussian | 1.49x | 18.96x | 13 / 0 / 27 |
| Restarts / basin memory | Local covariance | 1.61x | 1.54x | 11 / 0 / 29 |

## Basin-memory contribution

This isolates the archive by comparing otherwise identical restart controllers.

| Problem | Regret change from basin memory |
|---|---:|
| quadratic | +74.6% |
| bbob_10 | +36.7% |
| rastrigin | +63.2% |
| bbob_5 | -0.7% |

Across the four deterministic problems, basin memory reduced median regret by a geometric factor of 2.02x relative to the same restart controller without memory.

Runs ending with a reported error: 1.
- bbob_5, Paths / fixed mixture, seed 3: All walkers are invalid or outside the domain. Reset or change bounds/time step.

Exact counts and timings are in adaptive.csv.

Reproduce with `python3 fractal-gas-web/tools/benchmark-adaptive.py --budget 5000 --seeds 10` after building the native optimization engine, or regenerate the Markdown from the saved CSV with `--report-only`.
