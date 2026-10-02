# Long-budget optimization comparison

100,000 evaluations allowed per run; 10 seeds (0–9); dimensions [2, 10]. 600 completed runs, 68 reported errors.

All fractal variants use Wave, 8 initial walkers, maximum population 64, five protected elites, nonperiodic bounds, and unchanged default selection settings. Gaussian and local covariance are the preserved pre-existing perturbations in the current engine, not separately rebuilt historical revisions. Their standard deviation is 0.2; bounded adaptive uses [0.0001, 0.2]. The restart variant enables the existing controller and basin archive with defaults. No tuning was done for these problems. This follows the user's five-elite requirement and supersedes the unfinished zero-elite comparison. CMA uses its own selection rule, default initial population, initial sigma of 20% of domain width, and 9 large-population runs. Equal evaluation allowance does not imply equal movement scales or populations. COCO instance is 1; seeds vary optimizer randomness, not the problem instance.

Every actual initialization, trial, paired alternative, and restart evaluation is charged by the engine. Only complete operations are admitted. Runs may stop below the allowance if the next operation does not fit or the optimizer terminates. No artificial evaluations are added. Monotonic best-so-far and evaluation counts, and the hard evaluation ceiling, are checked after every step.

Regret means best objective minus the known optimum; lower is better. BBOB reference minima come from the realized COCO problem. Success means regret ≤ 1e-6, a common reporting threshold, not an engine stopping rule. IQR is the interpolated 25th–75th percentile range. Ten seeds are descriptive evidence, not a universal performance claim. CMA uses double-coordinate evaluation; fractal walkers use float coordinates, limiting comparisons near numerical precision.

Failed runs remain in every quality summary using their last best-so-far; they are not silently removed or restarted. A dagger (†) marks any group containing a reported error. Inspect evaluation use and the termination section before interpreting those groups as a full-budget comparison.

Process CPU time includes the native engine, snapshot checks, and initial/final status serialization. Wall times also include contention between benchmark workers; they are not isolated latency measurements. Full configurations, effective settings, final diagnostics, best coordinates, and archive state are retained in runs.jsonl. Build and source fingerprints are in manifest.json.

## Median final regret

| Problem | Dimensions | Gaussian | Local covariance | Bounded adaptive | Bounded + basins | BIPOP-active CMA-ES |
|---|---:|---:|---:|---:|---:|---:|
| Quadratic bowl | 2 | 2.39494e-08 | 1.82108e-08 | 9.29349e-15 | 4.29059e-13 | 6.49384e-21 |
| Quadratic bowl | 10 | 0.0013014 | 0.000437808 | 2.87321e-10 | 8.26152e-10 | 8.32244e-16 |
| Rotated ill-conditioned ellipsoid | 2 | 0.000379067 | 0.000199762 | 2.43598e-07 | 8.53089e-06 | 0 |
| Rotated ill-conditioned ellipsoid | 10 | 439.066† | 404.194† | 63.9327† | 1415.53 | 0 |
| Rastrigin | 2 | 0.000106975 | 0.000137606 | 4.97479 | 2.41022e-09 | 0 |
| Rastrigin | 10 | 46.0026 | 43.5388† | 42.2856 | 34.326 | 0.994959 |
| Rotated Rastrigin | 2 | 0.000476049† | 0.000449359 | 3.97983 | 1.90967e-08 | 0 |
| Rotated Rastrigin | 10 | 37.7649 | 53.1733† | 212.917 | 94.5196 | 2.98488 |
| Rosenbrock | 2 | 1.1112e-05 | 3.93254e-06 | 1.57741e-12 | 3.87186e-10 | 4.07137e-20 |
| Rosenbrock | 10 | 9.70938 | 7.01977 | 1.33897 | 7.60953 | 1.5801e-15 |
| Boundary optimum (linear slope) | 2 | 0.0919049† | 0.113781† | 0.0304036† | 2.0504e-05 | 0 |
| Boundary optimum (linear slope) | 10 | 44.8169† | 45.2233† | 41.9839† | 26.6468 | 0 |

## Variability, success, evaluation use, and time

| Problem | d | Variant | Median | IQR | Best–worst | Successes | Errors | Evaluations min–max | Median CPU seconds |
|---|---:|---|---:|---|---|---:|---:|---|---:|
| Quadratic bowl | 2 | Gaussian | 2.39494e-08 | 4.132e-09–5.545e-08 | 1.209e-09–6.643e-08 | 10/10 | 0 | 100000–100000 | 1.221 |
| Quadratic bowl | 2 | Local covariance | 1.82108e-08 | 6.602e-09–2.581e-08 | 1.495e-09–1.047e-07 | 10/10 | 0 | 100000–100000 | 96.271 |
| Quadratic bowl | 2 | Bounded adaptive | 9.29349e-15 | 3.887e-15–2.651e-14 | 2.151e-15–5.724e-14 | 10/10 | 0 | 99977–99985 | 2.522 |
| Quadratic bowl | 2 | Bounded + basin restarts | 4.29059e-13 | 2.622e-13–6.399e-13 | 3.351e-14–2.287e-12 | 10/10 | 0 | 99815–99944 | 3.876 |
| Quadratic bowl | 2 | BIPOP-active CMA-ES | 6.49384e-21 | 3.015e-21–1.137e-20 | 2.839e-22–2.977e-20 | 10/10 | 0 | 99130–99969 | 0.339 |
| Quadratic bowl | 10 | Gaussian | 0.0013014 | 0.001135–0.001559 | 0.001008–0.001833 | 0/10 | 0 | 100000–100000 | 1.737 |
| Quadratic bowl | 10 | Local covariance | 0.000437808 | 0.0004007–0.0004804 | 0.0003892–0.0005217 | 0/10 | 0 | 100000–100000 | 165.481 |
| Quadratic bowl | 10 | Bounded adaptive | 2.87321e-10 | 2.494e-10–3.52e-10 | 2.151e-10–4.554e-10 | 10/10 | 0 | 99977–99984 | 3.860 |
| Quadratic bowl | 10 | Bounded + basin restarts | 8.26152e-10 | 6.881e-10–1.198e-09 | 3.58e-10–9.559e-08 | 10/10 | 0 | 99929–99988 | 5.074 |
| Quadratic bowl | 10 | BIPOP-active CMA-ES | 8.32244e-16 | 4.828e-16–1.219e-15 | 4.379e-16–1.557e-15 | 10/10 | 0 | 99919–99992 | 1.008 |
| Rotated ill-conditioned ellipsoid | 2 | Gaussian | 0.000379067 | 0.0001893–0.001064 | 4.094e-05–0.001418 | 0/10 | 0 | 100000–100000 | 1.398 |
| Rotated ill-conditioned ellipsoid | 2 | Local covariance | 0.000199762 | 0.0001541–0.0003519 | 4.041e-05–0.001278 | 0/10 | 0 | 100000–100000 | 93.585 |
| Rotated ill-conditioned ellipsoid | 2 | Bounded adaptive | 2.43598e-07 | 5.013e-09–1.702e-06 | 1.131e-09–6.869e-06 | 7/10 | 0 | 99977–99985 | 2.790 |
| Rotated ill-conditioned ellipsoid | 2 | Bounded + basin restarts | 8.53089e-06 | 2.972e-06–0.0001497 | 1.774e-07–0.02436 | 2/10 | 0 | 99923–99987 | 4.603 |
| Rotated ill-conditioned ellipsoid | 2 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99292–99999 | 0.644 |
| Rotated ill-conditioned ellipsoid | 10 | Gaussian | 439.066 | 328–509.5 | 155.1–1729 | 0/10 | 1 | 82920–100000 | 2.670 |
| Rotated ill-conditioned ellipsoid | 10 | Local covariance | 404.194 | 303.5–459 | 60.68–1402 | 0/10 | 2 | 25040–100000 | 166.542 |
| Rotated ill-conditioned ellipsoid | 10 | Bounded adaptive | 63.9327 | 30.85–90.38 | 11.69–289.5 | 0/10 | 1 | 56338–99984 | 4.836 |
| Rotated ill-conditioned ellipsoid | 10 | Bounded + basin restarts | 1415.53 | 750.5–1580 | 114–2595 | 0/10 | 0 | 99936–99988 | 6.662 |
| Rotated ill-conditioned ellipsoid | 10 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99884–99998 | 2.358 |
| Rastrigin | 2 | Gaussian | 0.000106975 | 5.86e-05–0.0003358 | 1.239e-05–0.001124 | 0/10 | 0 | 100000–100000 | 1.462 |
| Rastrigin | 2 | Local covariance | 0.000137606 | 5.004e-05–0.0002243 | 3.176e-05–0.0004537 | 0/10 | 0 | 100000–100000 | 95.544 |
| Rastrigin | 2 | Bounded adaptive | 4.97479 | 4.229–7.213 | 8.043e-12–9.95 | 2/10 | 0 | 99977–99984 | 2.466 |
| Rastrigin | 2 | Bounded + basin restarts | 2.41022e-09 | 1.542e-09–5.911e-09 | 1.65e-10–8.255e-09 | 10/10 | 0 | 99819–99990 | 4.255 |
| Rastrigin | 2 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99287–99999 | 0.392 |
| Rastrigin | 10 | Gaussian | 46.0026 | 39.13–50.33 | 31.23–77.18 | 0/10 | 0 | 100000–100000 | 1.899 |
| Rastrigin | 10 | Local covariance | 43.5388 | 40.13–55.5 | 31.53–83.42 | 0/10 | 1 | 18424–100000 | 165.519 |
| Rastrigin | 10 | Bounded adaptive | 42.2856 | 39.05–52.24 | 31.84–97.51 | 0/10 | 0 | 99978–99984 | 3.802 |
| Rastrigin | 10 | Bounded + basin restarts | 34.326 | 27.36–37.56 | 22.88–39.8 | 0/10 | 0 | 99819–99988 | 5.424 |
| Rastrigin | 10 | BIPOP-active CMA-ES | 0.994959 | 0.2487–1.99 | 1.066e-14–2.985 | 3/10 | 0 | 99935–99999 | 1.191 |
| Rotated Rastrigin | 2 | Gaussian | 0.000476049 | 0.0001521–0.001116 | 4.514e-05–4.977 | 0/10 | 1 | 2440–100000 | 1.835 |
| Rotated Rastrigin | 2 | Local covariance | 0.000449359 | 0.0001583–0.7468 | 8.592e-05–0.9979 | 0/10 | 0 | 100000–100000 | 94.394 |
| Rotated Rastrigin | 2 | Bounded adaptive | 3.97983 | 0.995–3.98 | 0.995–8.955 | 0/10 | 0 | 99977–99982 | 2.675 |
| Rotated Rastrigin | 2 | Bounded + basin restarts | 1.90967e-08 | 1.481e-08–3.161e-08 | 1.926e-09–4.158e-08 | 10/10 | 0 | 99821–99972 | 4.567 |
| Rotated Rastrigin | 2 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99238–99995 | 0.684 |
| Rotated Rastrigin | 10 | Gaussian | 37.7649 | 27.57–42.48 | 23.1–55.27 | 0/10 | 0 | 100000–100000 | 3.054 |
| Rotated Rastrigin | 10 | Local covariance | 53.1733 | 40.26–70.72 | 25.35–90.03 | 0/10 | 2 | 13016–100000 | 164.960 |
| Rotated Rastrigin | 10 | Bounded adaptive | 212.917 | 191.3–260.2 | 84.57–367.1 | 0/10 | 0 | 99977–99986 | 4.757 |
| Rotated Rastrigin | 10 | Bounded + basin restarts | 94.5196 | 77.75–111.4 | 47.76–176.1 | 0/10 | 0 | 99935–99987 | 6.691 |
| Rotated Rastrigin | 10 | BIPOP-active CMA-ES | 2.98488 | 2.239–3.737 | 0.995–5.134 | 0/10 | 0 | 99851–99994 | 2.294 |
| Rosenbrock | 2 | Gaussian | 1.1112e-05 | 5.624e-06–2.284e-05 | 5.964e-07–4.929e-05 | 1/10 | 0 | 100000–100000 | 1.306 |
| Rosenbrock | 2 | Local covariance | 3.93254e-06 | 1.928e-06–8.653e-06 | 1.765e-08–1.635e-05 | 1/10 | 0 | 100000–100000 | 93.266 |
| Rosenbrock | 2 | Bounded adaptive | 1.57741e-12 | 5.782e-13–4.864e-12 | 1.279e-13–1.217e-11 | 10/10 | 0 | 99977–99985 | 2.541 |
| Rosenbrock | 2 | Bounded + basin restarts | 3.87186e-10 | 2.675e-10–3.123e-09 | 8.882e-12–4.008e-05 | 9/10 | 0 | 99923–99991 | 4.007 |
| Rosenbrock | 2 | BIPOP-active CMA-ES | 4.07137e-20 | 1.3e-20–9.623e-20 | 4.572e-21–1.518e-19 | 10/10 | 0 | 99275–99983 | 0.385 |
| Rosenbrock | 10 | Gaussian | 9.70938 | 8.518–10.52 | 7.531–134.3 | 0/10 | 0 | 100000–100000 | 1.634 |
| Rosenbrock | 10 | Local covariance | 7.01977 | 5.916–8.551 | 5.106–9.332 | 0/10 | 0 | 100000–100000 | 166.265 |
| Rosenbrock | 10 | Bounded adaptive | 1.33897 | 0.008955–3.992 | 0.00391–3.999 | 0/10 | 0 | 99977–99986 | 3.745 |
| Rosenbrock | 10 | Bounded + basin restarts | 7.60953 | 5.945–8.708 | 4.517–10.89 | 0/10 | 0 | 99933–99987 | 4.933 |
| Rosenbrock | 10 | BIPOP-active CMA-ES | 1.5801e-15 | 1.163e-15–1.722e-15 | 4.863e-16–2.301e-15 | 10/10 | 0 | 99875–99999 | 1.253 |
| Boundary optimum (linear slope) | 2 | Gaussian | 0.0919049 | 0.06079–0.1273 | 0.02495–0.2178 | 0/10 | 10 | 488–1968 | 0.019 |
| Boundary optimum (linear slope) | 2 | Local covariance | 0.113781 | 0.1046–0.2401 | 0.03444–1.777 | 0/10 | 10 | 128–928 | 0.235 |
| Boundary optimum (linear slope) | 2 | Bounded adaptive | 0.0304036 | 0.0001118–0.2751 | 4.625e-05–3.096 | 0/10 | 10 | 3911–19068 | 0.182 |
| Boundary optimum (linear slope) | 2 | Bounded + basin restarts | 2.0504e-05 | 8.106e-06–3.171e-05 | 5.245e-06–5.865e-05 | 0/10 | 0 | 99809–99979 | 3.879 |
| Boundary optimum (linear slope) | 2 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 98591–99965 | 0.389 |
| Boundary optimum (linear slope) | 10 | Gaussian | 44.8169 | 22.65–51.66 | 15.97–54.64 | 0/10 | 10 | 360–856 | 0.020 |
| Boundary optimum (linear slope) | 10 | Local covariance | 45.2233 | 38.83–51.07 | 15.72–82.14 | 0/10 | 10 | 456–1720 | 0.513 |
| Boundary optimum (linear slope) | 10 | Bounded adaptive | 41.9839 | 36.22–58.88 | 28.81–65.13 | 0/10 | 10 | 4277–16283 | 0.389 |
| Boundary optimum (linear slope) | 10 | Bounded + basin restarts | 26.6468 | 17.16–28.69 | 7.386–33.99 | 0/10 | 0 | 99908–99985 | 4.867 |
| Boundary optimum (linear slope) | 10 | BIPOP-active CMA-ES | 0 | 0–0 | 0–0 | 10/10 | 0 | 99702–99996 | 1.255 |

## Matched-seed comparisons against previous local covariance

Wins/ties/losses compare final regrets with tolerance 1e-8 × max(1, |a|, |b|). Ratios compare medians; below 1 is better. These are descriptive comparisons without significance tests.

| Problem | d | Candidate | Median regret ratio | Wins / ties / losses |
|---|---:|---|---:|---|
| Quadratic bowl | 2 | Bounded adaptive | 5.103e-07 | 7 / 3 / 0 |
| Quadratic bowl | 2 | Bounded + basin restarts | 2.356e-05 | 7 / 3 / 0 |
| Quadratic bowl | 2 | BIPOP-active CMA-ES | 3.566e-13 | 7 / 3 / 0 |
| Quadratic bowl | 10 | Bounded adaptive | 6.563e-07 | 10 / 0 / 0 |
| Quadratic bowl | 10 | Bounded + basin restarts | 1.887e-06 | 10 / 0 / 0 |
| Quadratic bowl | 10 | BIPOP-active CMA-ES | 1.901e-12 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 2 | Bounded adaptive | 0.001219 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 2 | Bounded + basin restarts | 0.04271 | 8 / 0 / 2 |
| Rotated ill-conditioned ellipsoid | 2 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 10 | Bounded adaptive | 0.1582 | 10 / 0 / 0 |
| Rotated ill-conditioned ellipsoid | 10 | Bounded + basin restarts | 3.502 | 2 / 0 / 8 |
| Rotated ill-conditioned ellipsoid | 10 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |
| Rastrigin | 2 | Bounded adaptive | 3.615e+04 | 2 / 0 / 8 |
| Rastrigin | 2 | Bounded + basin restarts | 1.752e-05 | 10 / 0 / 0 |
| Rastrigin | 2 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |
| Rastrigin | 10 | Bounded adaptive | 0.9712 | 6 / 0 / 4 |
| Rastrigin | 10 | Bounded + basin restarts | 0.7884 | 9 / 0 / 1 |
| Rastrigin | 10 | BIPOP-active CMA-ES | 0.02285 | 10 / 0 / 0 |
| Rotated Rastrigin | 2 | Bounded adaptive | 8857 | 0 / 0 / 10 |
| Rotated Rastrigin | 2 | Bounded + basin restarts | 4.25e-05 | 10 / 0 / 0 |
| Rotated Rastrigin | 2 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |
| Rotated Rastrigin | 10 | Bounded adaptive | 4.004 | 0 / 0 / 10 |
| Rotated Rastrigin | 10 | Bounded + basin restarts | 1.778 | 2 / 0 / 8 |
| Rotated Rastrigin | 10 | BIPOP-active CMA-ES | 0.05613 | 10 / 0 / 0 |
| Rosenbrock | 2 | Bounded adaptive | 4.011e-07 | 10 / 0 / 0 |
| Rosenbrock | 2 | Bounded + basin restarts | 9.846e-05 | 9 / 0 / 1 |
| Rosenbrock | 2 | BIPOP-active CMA-ES | 1.035e-14 | 10 / 0 / 0 |
| Rosenbrock | 10 | Bounded adaptive | 0.1907 | 10 / 0 / 0 |
| Rosenbrock | 10 | Bounded + basin restarts | 1.084 | 6 / 0 / 4 |
| Rosenbrock | 10 | BIPOP-active CMA-ES | 2.251e-16 | 10 / 0 / 0 |
| Boundary optimum (linear slope) | 2 | Bounded adaptive | 0.2672 | 7 / 0 / 3 |
| Boundary optimum (linear slope) | 2 | Bounded + basin restarts | 0.0001802 | 10 / 0 / 0 |
| Boundary optimum (linear slope) | 2 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |
| Boundary optimum (linear slope) | 10 | Bounded adaptive | 0.9284 | 4 / 0 / 6 |
| Boundary optimum (linear slope) | 10 | Bounded + basin restarts | 0.5892 | 9 / 0 / 1 |
| Boundary optimum (linear slope) | 10 | BIPOP-active CMA-ES | 0 | 10 / 0 / 0 |

## Termination

- bbob_10, d=10, Bounded adaptive, seed 8: 56338 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=10, Gaussian, seed 8: 82920 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=10, Local covariance, seed 2: 25040 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_10, d=10, Local covariance, seed 8: 27984 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=2, Gaussian, seed 5: 2440 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=10, Local covariance, seed 3: 13016 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_15, d=10, Local covariance, seed 9: 20936 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 0: 6640 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 1: 5365 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 2: 4784 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 3: 8865 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 4: 7826 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 5: 5436 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 6: 8040 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 7: 3911 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 8: 8261 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Bounded adaptive, seed 9: 19068 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 0: 1272 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 1: 560 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 2: 648 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 3: 864 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 4: 872 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 5: 656 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 6: 1248 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 7: 488 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 8: 1456 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Gaussian, seed 9: 1968 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 0: 928 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 1: 128 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 2: 736 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 3: 840 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 4: 632 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 5: 696 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 6: 576 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 7: 536 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 8: 728 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=2, Local covariance, seed 9: 824 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 0: 16283 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 1: 12363 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 2: 4277 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 3: 6660 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 4: 5098 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 5: 10123 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 6: 10000 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 7: 14867 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 8: 8531 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Bounded adaptive, seed 9: 7362 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 0: 712 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 1: 728 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 2: 712 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 3: 576 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 4: 360 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 5: 720 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 6: 680 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 7: 856 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 8: 856 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Gaussian, seed 9: 656 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 0: 456 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 1: 872 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 2: 1400 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 3: 632 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 4: 832 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 5: 584 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 6: 632 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 7: 728 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 8: 920 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- bbob_5, d=10, Local covariance, seed 9: 1720 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..
- rastrigin, d=10, Local covariance, seed 3: 18424 evaluations; All walkers are invalid or outside the domain. Reset or change bounds/time step..

Reproduce after building the native engine:

```sh
python3 fractal-gas-web/tools/benchmark-long-budget.py --budget 100000 --seeds 10 --dimensions 2 10 --workers 12 --output tests/optimization/reports/long-budget-100k-5elites
```

Use `--report-only` with the same output directory to regenerate this report without running optimization.
