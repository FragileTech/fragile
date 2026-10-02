# 20D population benchmark

9/9 runs completed. Up to 1,000,000 evaluations each; absolute objective target 1e-05. 3 seeds per case.

Fractal methods use Wave, five elites, nonperiodic bounds and no automatic restarts. Gaussian/local covariance scale is 0.2; bounded strategies use [0.0001, 0.2]. CMA uses its own population and restart schedule. BBOB instance is 1. No settings are tuned against these results.

Measured storage precision: `{"cma_coordinates_bits": 64, "objective_bits": 64, "swarm_coordinates_bits": 32, "swarm_fitness_bits": 32}`. A double-precision snapshot does not imply double-precision search coordinates.

Stopping is checked after initialization and complete steps. Unused budget is retained when another complete operation cannot fit. Failed runs retain their last best objective and count as failures. CPU timings were measured during concurrent execution, not isolated latency.

Actual evaluations: 8,999,208. Execution errors: 0.

| Function | Method | Walkers | Runs | Target reached | Median error | Median evaluations | Median CPU seconds | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bbob_5 | Cloning combined | 8 | 3 | 0/3 | 69.615 | 999992 | 26.433 | 0 |
| bbob_5 | Cloning combined | 64 | 3 | 0/3 | 42.3265 | 999936 | 21.227 | 0 |
| bbob_5 | Cloning combined | 128 | 3 | 0/3 | 44.8206 | 999808 | 18.938 | 0 |
