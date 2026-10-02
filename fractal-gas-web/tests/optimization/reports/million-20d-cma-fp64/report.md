# 20D population benchmark

135/135 runs completed. Up to 1,000,000 evaluations each; absolute objective target 1e-05. 5 seeds per case.

Fractal methods use Wave, five elites, nonperiodic bounds and no automatic restarts. Gaussian/local covariance scale is 0.2; bounded strategies use [0.0001, 0.2]. CMA uses its own population and restart schedule. BBOB instance is 1. No settings are tuned against these results.

Measured storage precision: `{"cma_coordinates_bits": 64, "objective_bits": 64, "swarm_coordinates_bits": 32, "swarm_fitness_bits": 32}`. A double-precision snapshot does not imply double-precision search coordinates.

Stopping is checked after initialization and complete steps. Unused budget is retained when another complete operation cannot fit. Failed runs retain their last best objective and count as failures. CPU timings were measured during concurrent execution, not isolated latency.

Actual evaluations: 57,753,243. Execution errors: 0.

| Function | Method | Walkers | Runs | Target reached | Median error | Median evaluations | Median CPU seconds | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bbob_1 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 8.48417e-06 | 2112 | 0.085 | 0 |
| bbob_10 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 8.59083e-06 | 16116 | 0.801 | 0 |
| bbob_11 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 8.77805e-06 | 11268 | 0.439 | 0 |
| bbob_12 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.48202e-06 | 22596 | 0.743 | 0 |
| bbob_13 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.0861e-06 | 42868 | 1.282 | 0 |
| bbob_14 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.60518e-06 | 6612 | 0.245 | 0 |
| bbob_15 | BIPOP-active CMA-ES | automatic | 5 | 1/5 | 1.98992 | 998874 | 29.581 | 0 |
| bbob_16 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 0.00439436 | 999951 | 52.525 | 0 |
| bbob_17 | BIPOP-active CMA-ES | automatic | 5 | 4/5 | 9.75048e-06 | 527946 | 18.502 | 0 |
| bbob_18 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 0.00672179 | 999923 | 30.198 | 0 |
| bbob_19 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 0.0809689 | 999959 | 24.789 | 0 |
| bbob_2 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 8.68244e-06 | 12852 | 0.598 | 0 |
| bbob_20 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 0.572455 | 999936 | 27.418 | 0 |
| bbob_21 | BIPOP-active CMA-ES | automatic | 5 | 2/5 | 0.929997 | 999679 | 24.388 | 0 |
| bbob_22 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 1.955 | 999897 | 17.593 | 0 |
| bbob_23 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 0.0409784 | 999979 | 97.465 | 0 |
| bbob_24 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 21.6623 | 999909 | 22.738 | 0 |
| bbob_3 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 5.96975 | 999792 | 33.137 | 0 |
| bbob_4 | BIPOP-active CMA-ES | automatic | 5 | 0/5 | 12.9345 | 999768 | 31.506 | 0 |
| bbob_5 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.82897e-06 | 2808 | 0.125 | 0 |
| bbob_6 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.74867e-06 | 7560 | 0.264 | 0 |
| bbob_7 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 7.8752e-08 | 49259 | 1.468 | 0 |
| bbob_8 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.25813e-06 | 16836 | 0.572 | 0 |
| bbob_9 | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.44565e-06 | 16728 | 0.699 | 0 |
| quadratic | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.27768e-06 | 1824 | 0.063 | 0 |
| rastrigin | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 8.93016e-06 | 302085 | 4.895 | 0 |
| rosenbrock | BIPOP-active CMA-ES | automatic | 5 | 5/5 | 9.234e-06 | 16680 | 0.570 | 0 |
