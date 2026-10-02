# Long-budget optimization comparison

100,000 evaluations allowed per run; 10 seeds (0–9); dimensions [20]. 300 completed runs, 49 reported errors.

All fractal variants use Wave, 8 initial walkers, maximum population 64, five protected elites, nonperiodic bounds, and unchanged default selection settings. Gaussian and local covariance are the preserved pre-existing perturbations in the current engine, not separately rebuilt historical revisions. Their standard deviation is 0.2; bounded adaptive uses [0.0001, 0.2]. The restart variant enables the existing controller and basin archive with defaults. No tuning was done for these problems. CMA uses its own selection rule, default initial population, initial sigma of 20% of domain width, and 9 large-population runs. Equal evaluation allowance does not imply equal movement scales or populations. COCO instance is 1; seeds vary optimizer randomness, not the problem instance.

Every actual initialization, trial, paired alternative, and restart evaluation is charged by the engine. Only complete operations are admitted. Runs may stop below the allowance if the next operation does not fit or the optimizer terminates. No artificial evaluations are added. Monotonic best-so-far and evaluation counts, and the hard evaluation ceiling, are checked after every step.

Regret means best objective minus the known optimum; lower is better. BBOB reference minima come from the realized COCO problem. Success means regret ≤ 1e-6, a common reporting threshold, not an engine stopping rule. IQR is the interpolated 25th–75th percentile range. These seeds are descriptive evidence, not a universal performance claim. CMA uses double-coordinate evaluation; fractal walkers use float coordinates, limiting comparisons near numerical precision.

Failed runs remain in every quality summary using their last best-so-far; they are not silently removed or restarted. A dagger (†) marks any group containing a reported error. Inspect evaluation use and the termination section before interpreting those groups as a full-budget comparison.

Process CPU time includes the native engine, snapshot checks, and initial/final status serialization. Wall times also include contention between benchmark workers; they are not isolated latency measurements. Full configurations, effective settings, final diagnostics, best coordinates, and archive state are retained in runs.jsonl. Build and source fingerprints are in manifest.json.

## Median final regret

| Problem | Dimensions | Gaussian | Local covariance | Bounded adaptive | Bounded + basins | BIPOP-active CMA-ES |
|---|---:|---:|---:|---:|---:|---:|
| Quadratic bowl | 20 | 0.00839394 | 0.00194017 | 2.00267e-09 | 3.62437e-09 | 1.53123e-15 |
| Rotated ill-conditioned ellipsoid | 20 | 4616.57† | 5447.39† | 287.356† | 8243.93 | 7.10543e-15 |
| Rastrigin | 20 | 130.807† | 136.191† | 130.836 | 98.998 | 4.47732 |
| Rotated Rastrigin | 20 | 121.447 | 243.513† | 780.527† | 418.371 | 5.96975 |
| Rosenbrock | 20 | 35.2524 | 22.4794 | 7.96438 | 16.803 | 1.879e-15 |
| Boundary optimum (linear slope) | 20 | 154.815† | 186.149† | 160.063† | 99.1904 | 0 |

## Variability, success, evaluation use, and time

| Problem | d | Variant | Median | IQR | Best–worst | Successes | Errors | Evaluations min–max | Median CPU seconds |
|---|---:|---|---:|---|---|---:|---:|---|---:|
| Quadratic bowl | 20 | Gaussian | 0.00839394 | 0.007886–0.009027 | 0.007243–0.009906 | 0/10 | 0 | 100000–100000 | 2.035 |
| Quadratic bowl | 20 | Local covariance | 0.00194017 | 0.001619–0.002133 | 0.001305–0.002433 | 0/10 | 0 | 100000–100000 | 256.191 |
| Quadratic bowl | 20 | Bounded adaptive | 2.00267e-09 | 1.747e-09–2.171e-09 | 1.558e-09–2.421e-09 | 10/10 | 0 | 99977–99986 | 6.587 |
| Quadratic bowl | 20 | Bounded + basin restarts | 3.62437e-09 | 3.382e-09–4.824e-09 | 3.091e-09–1.169e-08 | 10/10 | 0 | 99965–99991 | 7.806 |
| Quadratic bowl | 20 | BIPOP-active CMA-ES | 1.53123e-15 | 1.291e-15–2.133e-15 | 8.651e-16–3.086e-15 | 10/10 | 0 | 99812–99999 | 2.704 |
| Rotated ill-conditioned ellipsoid | 20 | Gaussian | 4616.57 | 2513–5326 | 1910–1.031e+04 | 0/10 | 1 | 27016–100000 | 4.184 |
| Rotated ill-conditioned ellipsoid | 20 | Local covariance | 5447.39 | 3018–1.025e+05 | 2224–3.747e+05 | 0/10 | 6 | 608–100000 | 84.384 |
| Rotated ill-conditioned ellipsoid | 20 | Bounded adaptive | 287.356 | 171.1–1126 | 48.49–1.516e+04 | 0/10 | 3 | 24381–99984 | 8.821 |
| Rotated ill-conditioned ellipsoid | 20 | Bounded + basin restarts | 8243.93 | 5909–9273 | 1043–2.187e+04 | 0/10 | 0 | 99970–99990 | 9.874 |
| Rotated ill-conditioned ellipsoid | 20 | BIPOP-active CMA-ES | 7.10543e-15 | 1.776e-15–7.105e-15 | 0–7.105e-15 | 10/10 | 0 | 99953–99996 | 6.256 |
| Rastrigin | 20 | Gaussian | 130.807 | 124.6–154.9 | 114.8–226.1 | 0/10 | 2 | 1032–100000 | 2.315 |
| Rastrigin | 20 | Local covariance | 136.191 | 122.1–147.5 | 94.4–159.6 | 0/10 | 3 | 10480–100000 | 252.017 |
| Rastrigin | 20 | Bounded adaptive | 130.836 | 111.2–146.3 | 87.56–159.2 | 0/10 | 0 | 99977–99984 | 6.331 |
| Rastrigin | 20 | Bounded + basin restarts | 98.998 | 91.04–107.7 | 79.6–134.3 | 0/10 | 0 | 99837–99988 | 8.146 |
| Rastrigin | 20 | BIPOP-active CMA-ES | 4.47732 | 3.234–5.721 | 2.985–6.965 | 0/10 | 0 | 99941–99999 | 3.292 |
| Rotated Rastrigin | 20 | Gaussian | 121.447 | 114.4–139.3 | 98.4–155.1 | 0/10 | 0 | 100000–100000 | 4.719 |
| Rotated Rastrigin | 20 | Local covariance | 243.513 | 202.1–262.1 | 125.6–290.4 | 0/10 | 1 | 38952–100000 | 254.169 |
| Rotated Rastrigin | 20 | Bounded adaptive | 780.527 | 673.8–804.9 | 423.8–861.6 | 0/10 | 3 | 11695–99985 | 8.894 |
| Rotated Rastrigin | 20 | Bounded + basin restarts | 418.371 | 349.7–501.4 | 294.5–587 | 0/10 | 0 | 99959–99992 | 10.364 |
| Rotated Rastrigin | 20 | BIPOP-active CMA-ES | 5.96975 | 5.14–9.95 | 3.98–13.93 | 0/10 | 0 | 99923–99998 | 5.867 |
| Rosenbrock | 20 | Gaussian | 35.2524 | 32.02–36.21 | 26.95–191.3 | 0/10 | 0 | 100000–100000 | 2.080 |
| Rosenbrock | 20 | Local covariance | 22.4794 | 21.87–22.81 | 19.98–222.3 | 0/10 | 0 | 100000–100000 | 251.225 |
| Rosenbrock | 20 | Bounded adaptive | 7.96438 | 6.942–8.956 | 0.1242–13.03 | 0/10 | 0 | 99978–99985 | 6.857 |
| Rosenbrock | 20 | Bounded + basin restarts | 16.803 | 16.31–17.62 | 13.25–20.65 | 0/10 | 0 | 99966–99989 | 7.674 |
| Rosenbrock | 20 | BIPOP-active CMA-ES | 1.879e-15 | 1.499e-15–2.357e-15 | 9.487e-16–8.418e-15 | 10/10 | 0 | 99954–99990 | 4.250 |
| Boundary optimum (linear slope) | 20 | Gaussian | 154.815 | 115.7–174.4 | 103.8–203.3 | 0/10 | 10 | 304–992 | 0.025 |
| Boundary optimum (linear slope) | 20 | Local covariance | 186.149 | 155–205.1 | 131.6–268.8 | 0/10 | 10 | 232–1080 | 0.330 |
| Boundary optimum (linear slope) | 20 | Bounded adaptive | 160.063 | 141.8–169.5 | 114.1–235.3 | 0/10 | 10 | 4044–15244 | 0.547 |
| Boundary optimum (linear slope) | 20 | Bounded + basin restarts | 99.1904 | 88.81–107.3 | 56.54–116.7 | 0/10 | 0 | 99837–99992 | 7.309 |
| Boundary optimum (linear slope) | 20 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99831–99950 | 3.380 |

## Matched-seed comparisons against previous local covariance

Wins/ties/losses compare final regrets with tolerance 1e-8 × max(1, |a|, |b|). Ratios compare medians; below 1 is better. These are descriptive comparisons without significance tests.

| Problem | d | Candidate | Median regret ratio | Wins / ties / losses |
|---|---:|---|---:|---|
| Quadratic bowl | 20 | Bounded adaptive | 1.032e-06 | 10 / 0 / 0 |
| Quadratic bowl | 20 | Bounded + basin restarts | 1.868e-06 | 10 / 0 / 0 |
| Quadratic bowl | 20 | BIPOP-active CMA-ES | 7.892e-13 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 20 | Bounded adaptive | 0.05275 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 20 | Bounded + basin restarts | 1.513 | 5 / 0 / 5 |
| Rotated ill-conditioned ellipsoid | 20 | BIPOP-active CMA-ES | 1.304e-18 | 10 / 0 / 0 |
| Rastrigin | 20 | Bounded adaptive | 0.9607 | 8 / 0 / 2 |
| Rastrigin | 20 | Bounded + basin restarts | 0.7269 | 10 / 0 / 0 |
| Rastrigin | 20 | BIPOP-active CMA-ES | 0.03288 | 10 / 0 / 0 |
| Rotated Rastrigin | 20 | Bounded adaptive | 3.205 | 0 / 0 / 10 |
| Rotated Rastrigin | 20 | Bounded + basin restarts | 1.718 | 0 / 0 / 10 |
| Rotated Rastrigin | 20 | BIPOP-active CMA-ES | 0.02452 | 10 / 0 / 0 |
| Rosenbrock | 20 | Bounded adaptive | 0.3543 | 10 / 0 / 0 |
| Rosenbrock | 20 | Bounded + basin restarts | 0.7475 | 10 / 0 / 0 |
| Rosenbrock | 20 | BIPOP-active CMA-ES | 8.359e-17 | 10 / 0 / 0 |
| Boundary optimum (linear slope) | 20 | Bounded adaptive | 0.8599 | 7 / 0 / 3 |
| Boundary optimum (linear slope) | 20 | Bounded + basin restarts | 0.5329 | 10 / 0 / 0 |
| Boundary optimum (linear slope) | 20 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |

## Termination

- bbob_10, d=20, Bounded adaptive, seed 1: 62199 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Bounded adaptive, seed 6: 24381 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Bounded adaptive, seed 8: 35012 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Gaussian, seed 9: 27016 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 0: 1648 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 4: 23448 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 6: 776 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 7: 6344 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 8: 44368 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=20, Local covariance, seed 9: 608 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=20, Bounded adaptive, seed 3: 21156 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=20, Bounded adaptive, seed 4: 11695 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=20, Bounded adaptive, seed 9: 19283 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=20, Local covariance, seed 8: 38952 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 0: 7772 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 1: 5134 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 2: 11463 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 3: 7976 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 4: 10847 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 5: 5997 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 6: 13885 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 7: 4044 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 8: 15244 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Bounded adaptive, seed 9: 7329 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 0: 872 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 1: 960 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 2: 616 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 3: 304 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 4: 992 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 5: 736 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 6: 608 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 7: 688 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 8: 952 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Gaussian, seed 9: 600 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 0: 1080 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 1: 376 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 2: 232 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 3: 936 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 4: 280 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 5: 464 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 6: 912 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 7: 416 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 8: 792 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=20, Local covariance, seed 9: 496 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=20, Gaussian, seed 0: 1032 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=20, Gaussian, seed 2: 29720 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=20, Local covariance, seed 5: 90896 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=20, Local covariance, seed 6: 88304 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=20, Local covariance, seed 8: 10480 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..

Reproduce after building the native engine:

```sh
python3 fractal-gas-web/tools/benchmark-long-budget.py --budget 100000 --seeds 10 --dimensions 20 --workers 12 --output tests/optimization/reports/long-budget-100k-20d-5elites
```

Use `--report-only` with the same output directory to regenerate this report without running optimization.
