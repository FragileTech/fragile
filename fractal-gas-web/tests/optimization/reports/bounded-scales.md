# Bounded adaptive scale comparison (fgopt-7)

Wave, two dimensions, 5 seeds, 5000 evaluations per run; 8 initial walkers, 64 maximum, nonperiodic bounds. Adaptive scale range [0.0001, 0.2], Gaussian/local covariance standard deviation 0.2. Complete operations only; actual counts may be below the shared allowance. Runtime includes status extraction and controller metadata. This is a small experimental comparison, not a superiority claim.

| Problem | Variant | Median regret | Mean evaluations | Mean seconds | Errors |
|---|---|---:|---:|---:|---:|
| quadratic | Gaussian | 1.52621e-06 | 5000.0 | 0.0951 | 0 |
| quadratic | Local covariance | 2.66898e-06 | 5000.0 | 1.3497 | 0 |
| quadratic | Bounded adaptive | 8.25539e-11 | 4981.6 | 0.1446 | 0 |
| quadratic | Bounded + basin restarts | 1.17318e-08 | 4954.8 | 2.9789 | 0 |
| bbob_10 | Gaussian | 0.0587515 | 5000.0 | 0.2134 | 0 |
| bbob_10 | Local covariance | 0.383389 | 5000.0 | 1.6285 | 0 |
| bbob_10 | Bounded adaptive | 0.000559335 | 4983.4 | 0.1132 | 0 |
| bbob_10 | Bounded + basin restarts | 1.62175 | 4923.2 | 1.7196 | 0 |
| rastrigin | Gaussian | 0.0746236 | 5000.0 | 0.0860 | 0 |
| rastrigin | Local covariance | 0.0220915 | 5000.0 | 1.1626 | 0 |
| rastrigin | Bounded adaptive | 4.97479 | 4980.2 | 0.1160 | 0 |
| rastrigin | Bounded + basin restarts | 1.49715 | 4943.8 | 2.8541 | 0 |
| bbob_5 | Gaussian | 0.119891 | 5000.0 | 0.2137 | 0 |
| bbob_5 | Local covariance | 0.0805154 | 5000.0 | 1.7249 | 0 |
| bbob_5 | Bounded adaptive | 0.00110483 | 4981.6 | 0.1491 | 0 |
| bbob_5 | Bounded + basin restarts | 0.26043 | 4970.6 | 2.0746 | 0 |

Runs: 80. Errors: 0.

Reproduce: `python3 fractal-gas-web/tools/benchmark-bounded-scales.py --seeds 5 --budget 5000 --min-scale 0.0001 --max-scale 0.2`.

The earlier adaptive.md report describes the obsolete path-based scale rule. The bounded revision changes scalar learning and proposal normalization, so comparisons with that report measure the combined revision rather than isolating one coefficient.
