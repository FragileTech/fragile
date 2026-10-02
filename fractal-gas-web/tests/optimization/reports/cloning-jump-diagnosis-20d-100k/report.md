# Cloning-guided jump diagnosis

Pilot ablation: 20D, 128 walkers, five elites, two seeds, 100,000 evaluations per run, target 1e-5, CMA boundary repair, no restarts. Current scale bounds are [0.0001, 0.2]. Mixed precision: FP32 walkers, FP64 objectives. These short runs measure sensitivity, not a definitive explanation of the million-evaluation comparison.

| Variant | Quadratic median error | Rotated ellipsoid median error | Rastrigin median error |
|---|---:|---:|---:|
| Current | 1.44537 | 16281.7 | 117.902 |
| No drift | 6.77637 | 48772.5 | 116.91 |
| No covariance | 0.00253042 | 22857.1 | 115.426 |
| Scales / 10 | 16.0685 | 674547 | 117.415 |
| Scales x 10 | 0.000168651 | 15520.5 | 113.924 |

Execution errors: 0. Actual evaluations: 2,995,200.
