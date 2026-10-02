# Ten-swarm Fractal Populations versus CMA-ES

20 dimensions; 5 seeds (0–4); 100,000 shared evaluations per run, including initialization. COCO instance 1.

Fractal: ten concurrent Wave swarms, each with 32 walkers, five protected/exported elites and five foreign imports every step. Fixed Gaussian jump scales (absolute coordinate units): 0.0001, 0.000232692, 0.000541455, 0.00125992, 0.00293173, 0.0068219, 0.015874, 0.0369375, 0.0859506, 0.2. Population and scale adaptation and basin restarts are disabled to preserve the requested swarm sizes/scales. Each exchange is checked for counts, foreign provenance, and distinct unprotected destinations.

CMA: freshly executed BIPOP-active CMA-ES, native defaults (own population, 20%-of-domain initial sigma, bounded mapping). Fractal uses nonperiodic boundaries without repair, matching the earlier comparison. Wave uses float32 coordinates; CMA uses float64. Equal evaluation caps do not imply equal populations or identical boundary handling. Complete rounds/generations may leave unused budget. Timings include Python/JSON verification overhead and are not pure optimizer speed measurements.

Error is best evaluated objective minus known optimum; lower is better. Success means error ≤ 1e-6. Five seeds are descriptive, not a significance test.

| Problem | Method | Median error | Error min–max | Successes | Evaluations min–max | Median seconds |
|---|---|---:|---|---:|---|---:|
| Quadratic bowl | Fractal Populations | 5.12736e-09 | 4.24863e-09–7.3665e-09 | 5/5 | 99840–99840 | 3.689 |
| Quadratic bowl | BIPOP-active CMA-ES | 1.29878e-15 | 8.65116e-16–1.53398e-15 | 5/5 | 99833–99999 | 0.917 |
| Rotated ill-conditioned ellipsoid | Fractal Populations | 6362.45 | 1799.25–28209 | 0/5 | 99840–99840 | 3.276 |
| Rotated ill-conditioned ellipsoid | BIPOP-active CMA-ES | 7.10543e-15 | 0–7.10543e-15 | 5/5 | 99953–99993 | 1.377 |
| Rastrigin | Fractal Populations | 79.5965 | 64.6722–116.41 | 0/5 | 99840–99840 | 2.439 |
| Rastrigin | BIPOP-active CMA-ES | 5.96975 | 2.98488–6.96471 | 0/5 | 99946–99991 | 0.674 |
| Rotated Rastrigin | Fractal Populations | 165.163 | 91.538–198.116 | 0/5 | 99840–99840 | 4.006 |
| Rotated Rastrigin | BIPOP-active CMA-ES | 9.94959 | 5.96975–13.9294 | 0/5 | 99925–99998 | 2.445 |
| Rosenbrock | Fractal Populations | 18.8594 | 17.5155–19.4995 | 0/5 | 99840–99840 | 4.160 |
| Rosenbrock | BIPOP-active CMA-ES | 1.81329e-15 | 9.48739e-16–3.02777e-15 | 5/5 | 99972–99990 | 1.759 |
| Boundary optimum (linear slope) | Fractal Populations | 83.1354 | 70.3175–87.0175 | 0/5 | 99840–99840 | 2.964 |
| Boundary optimum (linear slope) | BIPOP-active CMA-ES | 0 | 0–0 | 5/5 | 99835–99950 | 0.874 |

Reported runtime errors: 0. Total verified foreign imports: 466,500.

Full per-seed results and resolved configurations: runs.jsonl. Build/runner fingerprints: manifest.json.

Reproduce: `uv run python fractal-gas-web/tools/benchmark-fractal-populations.py --output /tmp/fractal-populations-comparison`.
