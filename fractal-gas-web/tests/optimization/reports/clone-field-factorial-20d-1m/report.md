# Clone-field drift/covariance ablation

140/140 runs completed; 0 execution errors.

Seven 20D functions, 128 walkers, five elites, seeds 0–4, CMA boundary repair, no restarts. Up to 1,000,000 evaluations or absolute error ≤1e-5. Scale bounds [0.0001, 0.2], drift strength 0.25. Mixed precision: FP32 walkers/fitness, FP64 objectives/covariance.

A predeclared 2×2 comparison: drift denominator is either fixed radius × summed absolute score response (current), or summed absolute response × comparison RMS length (normalized drift). The latter removes contraction-dependent drift damping while retaining directional cancellation. Covariance identity regularization is either 10% (current) or 0.1% (weak). All other settings and random seeds match. Production engine is unchanged.

Baseline rebuilt with the same pipeline reproduces the production snapshots exactly on seven problems after 20 steps. Each variant is also checked for evaluation bounds and finite, bounded movement diagnostics. Final diagnostics describe the maximum over local models, including retained inactive models.

| Problem | Variant | Runs | Targets | Median error | Median evaluations | Median drift/noise |
|---|---|---:|---:|---:|---:|---:|
| quadratic | current | 5 | 5 | 9.87342e-06 | 361088 | 0.0019 |
| quadratic | normalized_drift | 5 | 5 | 9.91528e-06 | 355840 | 0.049448 |
| quadratic | weak_regularization | 5 | 0 | 1.98539 | 999808 | 0.0026194 |
| quadratic | both | 5 | 0 | 1.37038 | 999808 | 0.042111 |
| bbob_2 | current | 5 | 0 | 589.121 | 999808 | 0.0054049 |
| bbob_2 | normalized_drift | 5 | 0 | 1329.27 | 999808 | 0.032275 |
| bbob_2 | weak_regularization | 5 | 0 | 53102.9 | 999808 | 0.0026719 |
| bbob_2 | both | 5 | 0 | 84376.7 | 999808 | 0.044646 |
| bbob_10 | current | 5 | 0 | 833.36 | 999808 | 0.0016237 |
| bbob_10 | normalized_drift | 5 | 0 | 842.357 | 999808 | 0.030943 |
| bbob_10 | weak_regularization | 5 | 0 | 64476.6 | 999808 | 0.0059702 |
| bbob_10 | both | 5 | 0 | 41768 | 999808 | 0.008519 |
| rosenbrock | current | 5 | 0 | 7.01955 | 999808 | 0.0023108 |
| rosenbrock | normalized_drift | 5 | 0 | 10.2457 | 999808 | 0.036407 |
| rosenbrock | weak_regularization | 5 | 0 | 91082.8 | 999808 | 0.0030207 |
| rosenbrock | both | 5 | 0 | 83264 | 999808 | 0.03871 |
| bbob_5 | current | 5 | 0 | 0.119446 | 999808 | 0.011014 |
| bbob_5 | normalized_drift | 5 | 0 | 0.212966 | 999808 | 0.17672 |
| bbob_5 | weak_regularization | 5 | 0 | 163.263 | 999808 | 0.021484 |
| bbob_5 | both | 5 | 0 | 162.03 | 999808 | 0.061321 |
| bbob_15 | current | 5 | 0 | 337.287 | 999808 | 0.0023884 |
| bbob_15 | normalized_drift | 5 | 0 | 278.583 | 999808 | 0.043831 |
| bbob_15 | weak_regularization | 5 | 0 | 294.503 | 999808 | 0.0028271 |
| bbob_15 | both | 5 | 0 | 325.964 | 999808 | 0.081249 |
| bbob_24 | current | 5 | 0 | 217.267 | 999808 | 0.0012823 |
| bbob_24 | normalized_drift | 5 | 0 | 251.722 | 999808 | 0.0469 |
| bbob_24 | weak_regularization | 5 | 0 | 184.907 | 999808 | 0.0017203 |
| bbob_24 | both | 5 | 0 | 154.02 | 999808 | 0.074584 |

## Matched comparisons against current

Both-target cases are compared by evaluations separately; sub-tolerance objective differences are not wins.

| Variant | Matched | Lower error | Higher error | Both targets | Current/new targets | Median new/current evaluations when both solve |
|---|---:|---:|---:|---:|---|---:|
| normalized_drift | 35 | 11 | 19 | 5 | 5/5 | 1.02 |
| weak_regularization | 35 | 9 | 26 | 0 | 5/0 | — |
| both | 35 | 9 | 26 | 0 | 5/0 | — |

## Interpretation

All 140 runs completed without execution errors; 133,553,152 objective evaluations were charged and all input fingerprints verified. The production engine was not modified. The broad sweep was stopped at 516 completed runs, whose results remain preserved separately.

Normalizing drift raised the median final reported drift/noise ratio from 0.00239 to 0.0438, but produced 11 lower-error and 19 higher-error outcomes, with five cases reaching the target in both variants. Those five required a median 1.02 times the baseline evaluations. This does not support treating weak drift magnitude as the sole bottleneck.

Reducing identity regularization from 10% to 0.1% produced large regressions on smooth and boundary-optimum problems. All five quadratic baseline runs reached the target; none of the weak-regularization runs did. The stronger regularization is providing useful protection in this implementation. A plausible explanation is that covariance of selection vectors does not reliably describe useful exploration directions when made highly anisotropic; this mechanism is an interpretation, not directly measured curvature evidence.

There are problem-specific gains: normalized drift reduces the rotated-Rastrigin median error from 337.3 to 278.6, and combining the changes reduces the Lunacek median from 217.3 to 154.0. Neither problem reaches the target. Neither tested intervention is a generally better replacement at 128 walkers. Keep current production defaults; these results cover seven functions, five seeds, and one population size, not every fractal algorithm or population.
