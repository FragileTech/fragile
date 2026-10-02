# Cloning-guided movement: equal-budget ablation

Six 20D problems; ten seeds per configuration; 100,000-evaluation cap. Fractal methods use Wave with 128 walkers, five elites, scale bounds [0.0001, 0.2], and no automatic restarts. Gaussian/local covariance standard deviation is 0.2. CMA-ES uses its own population and restart schedule. All results were freshly measured with the same engine. No settings were tuned after examining these results.

Error is best evaluated objective minus the known optimum; lower is better. Intervals are inclusive 25th–75th percentiles. Errors retain their last best objective. Timing is process CPU time in a concurrent benchmark, not isolated latency. Actual evaluation counts can be below the cap because complete steps must fit.

420 runs; 18 execution errors; 41,436,170 actual evaluations.

| Problem | Gaussian | Local covariance | Bounded adaptive | Cloning geometry | Cloning drift | Cloning combined | BIPOP-active CMA-ES |
|---|---:|---:|---:|---:|---:|---:|---:|
| Quadratic bowl | 0.0159512 | 0.00948842 | 0.000126097 | 6.20928 | 0.00421205 | 0.641172 | 1.53123e-15 |
| Rotated ill-conditioned ellipsoid | 13880.3 | 20029.7 | 8010.1 | 95841.7 | 30101.5 | 57569.3 | 7.10543e-15 |
| Rastrigin | 79.6846 | 83.1232 | 95.5156 | 98.0071 | 99.4981 | 98.003 | 4.47732 |
| Rotated Rastrigin | 110.802 | 111.842 | 360.697 | 419.946 | 419.947 | 406.933 | 5.96975 |
| Rosenbrock | 45.7696 | 33.5716 | 18.1796 | 151612 | 148.559 | 9973.24 | 1.879e-15 |
| Boundary optimum (linear slope) | 35.9146 | 34.7661 | 68.8694 | 98.1357 | 80.2226† | 78.7141† | 0 |

## Variability, accounting and movement diagnostics

| Problem | Method | Median error | IQR | Errors | Success ≤1e-6 | Evaluations min–max | CPU seconds median | Local models median | Independent families/model median | Fallback models median |
|---|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| Quadratic bowl | Gaussian | 0.0159512 | 0.01549–0.01793 | 0 | 0/10 | 99968–99968 | 1.784 | — | — | — |
| Quadratic bowl | Local covariance | 0.00948842 | 0.008453–0.00989 | 0 | 0/10 | 99968–99968 | 18.037 | — | — | — |
| Quadratic bowl | Bounded adaptive | 0.000126097 | 8.41e-05–0.0001873 | 0 | 0/10 | 99617–99753 | 4.939 | — | — | — |
| Quadratic bowl | Cloning geometry | 6.20928 | 4.281–7.129 | 0 | 0/10 | 99840–99840 | 3.870 | 1 | 100 | 0 |
| Quadratic bowl | Cloning drift | 0.00421205 | 0.001805–0.005653 | 0 | 0/10 | 99840–99840 | 3.889 | 1 | 99.5 | 0 |
| Quadratic bowl | Cloning combined | 0.641172 | 0.2823–1.277 | 0 | 0/10 | 99840–99840 | 3.995 | 1 | 95 | 0 |
| Quadratic bowl | BIPOP-active CMA-ES | 1.53123e-15 | 1.291e-15–2.133e-15 | 0 | 10/10 | 99812–99999 | 2.906 | — | — | — |
| Rotated ill-conditioned ellipsoid | Gaussian | 13880.3 | 9788–1.499e+04 | 0 | 0/10 | 99968–99968 | 4.070 | — | — | — |
| Rotated ill-conditioned ellipsoid | Local covariance | 20029.7 | 1.676e+04–2.366e+04 | 0 | 0/10 | 99968–99968 | 21.865 | — | — | — |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 8010.1 | 4181–1.139e+04 | 0 | 0/10 | 99621–99753 | 6.989 | — | — | — |
| Rotated ill-conditioned ellipsoid | Cloning geometry | 95841.7 | 6.478e+04–1.202e+05 | 0 | 0/10 | 99840–99840 | 6.093 | 1 | 96.5 | 0 |
| Rotated ill-conditioned ellipsoid | Cloning drift | 30101.5 | 1.686e+04–4.066e+04 | 0 | 0/10 | 99840–99840 | 5.713 | 1 | 98 | 0 |
| Rotated ill-conditioned ellipsoid | Cloning combined | 57569.3 | 3.229e+04–6.305e+04 | 0 | 0/10 | 99840–99840 | 5.737 | 1 | 97.5 | 0 |
| Rotated ill-conditioned ellipsoid | BIPOP-active CMA-ES | 7.10543e-15 | 1.776e-15–7.105e-15 | 0 | 10/10 | 99953–99996 | 6.833 | — | — | — |
| Rastrigin | Gaussian | 79.6846 | 77.11–85.82 | 0 | 0/10 | 99968–99968 | 2.173 | — | — | — |
| Rastrigin | Local covariance | 83.1232 | 68.71–91.18 | 0 | 0/10 | 99968–99968 | 20.478 | — | — | — |
| Rastrigin | Bounded adaptive | 95.5156 | 89.05–108.2 | 0 | 0/10 | 99618–99742 | 5.279 | — | — | — |
| Rastrigin | Cloning geometry | 98.0071 | 85.84–106.7 | 0 | 0/10 | 99840–99840 | 3.958 | 1 | 111 | 0 |
| Rastrigin | Cloning drift | 99.4981 | 86.57–107.2 | 0 | 0/10 | 99840–99840 | 3.953 | 1 | 110 | 0 |
| Rastrigin | Cloning combined | 98.003 | 91.78–107.5 | 0 | 0/10 | 99840–99840 | 4.163 | 1 | 108 | 0 |
| Rastrigin | BIPOP-active CMA-ES | 4.47732 | 3.234–5.721 | 0 | 0/10 | 99941–99999 | 3.629 | — | — | — |
| Rotated Rastrigin | Gaussian | 110.802 | 106.4–113 | 0 | 0/10 | 99968–99968 | 4.345 | — | — | — |
| Rotated Rastrigin | Local covariance | 111.842 | 99.48–115 | 0 | 0/10 | 99968–99968 | 22.654 | — | — | — |
| Rotated Rastrigin | Bounded adaptive | 360.697 | 316.6–394 | 0 | 0/10 | 99642–99751 | 7.493 | — | — | — |
| Rotated Rastrigin | Cloning geometry | 419.946 | 382.1–473.9 | 0 | 0/10 | 99840–99840 | 6.401 | 1 | 106 | 0 |
| Rotated Rastrigin | Cloning drift | 419.947 | 378.6–434.4 | 0 | 0/10 | 99840–99840 | 5.912 | 1 | 106 | 0 |
| Rotated Rastrigin | Cloning combined | 406.933 | 386.7–447.9 | 0 | 0/10 | 99840–99840 | 6.288 | 1 | 108 | 0 |
| Rotated Rastrigin | BIPOP-active CMA-ES | 5.96975 | 5.14–9.95 | 0 | 0/10 | 99923–99998 | 6.197 | — | — | — |
| Rosenbrock | Gaussian | 45.7696 | 45.06–52.11 | 0 | 0/10 | 99968–99968 | 1.672 | — | — | — |
| Rosenbrock | Local covariance | 33.5716 | 32.03–50.18 | 0 | 0/10 | 99968–99968 | 19.815 | — | — | — |
| Rosenbrock | Bounded adaptive | 18.1796 | 17.77–18.88 | 0 | 0/10 | 99619–99736 | 4.950 | — | — | — |
| Rosenbrock | Cloning geometry | 151612 | 6.01e+04–1.727e+05 | 0 | 0/10 | 99840–99840 | 3.841 | 1 | 93.5 | 0 |
| Rosenbrock | Cloning drift | 148.559 | 70.09–219.5 | 0 | 0/10 | 99840–99840 | 3.921 | 1 | 97 | 0 |
| Rosenbrock | Cloning combined | 9973.24 | 2231–1.324e+04 | 0 | 0/10 | 99840–99840 | 4.035 | 1 | 94.5 | 0 |
| Rosenbrock | BIPOP-active CMA-ES | 1.879e-15 | 1.499e-15–2.357e-15 | 0 | 10/10 | 99954–99990 | 4.528 | — | — | — |
| Boundary optimum (linear slope) | Gaussian | 35.9146 | 30.26–39.01 | 0 | 0/10 | 99968–99968 | 2.085 | — | — | — |
| Boundary optimum (linear slope) | Local covariance | 34.7661 | 33.56–40.86 | 0 | 0/10 | 99968–99968 | 19.698 | — | — | — |
| Boundary optimum (linear slope) | Bounded adaptive | 68.8694 | 53.22–79.71 | 0 | 0/10 | 99631–99740 | 5.108 | — | — | — |
| Boundary optimum (linear slope) | Cloning geometry | 98.1357 | 87.65–110.3 | 0 | 0/10 | 99840–99840 | 4.272 | 1 | 76.5 | 0 |
| Boundary optimum (linear slope) | Cloning drift | 80.2226 | 71.51–87.5 | 9 | 0/10 | 43392–99840 | 2.406 | 1 | 6 | 0 |
| Boundary optimum (linear slope) | Cloning combined | 78.7141 | 73.7–86.71 | 9 | 0/10 | 47104–99840 | 3.677 | 1 | 6 | 0 |
| Boundary optimum (linear slope) | BIPOP-active CMA-ES | 0 | 0–0 | 0 | 10/10 | 99831–99950 | 3.695 | — | — | — |

## Reproduction

```sh
python3 fractal-gas-web/tools/benchmark-cloning-guided.py --workers 8
```

The manifest fingerprints the engine, source files and runners. Full configurations and final diagnostics are retained in runs.jsonl. This experiment compares optimization variants; it does not establish Gibbs invariance or universal superiority.
