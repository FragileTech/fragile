# 20D population benchmark

1159/2160 runs completed. Up to 1,000,000 evaluations each; absolute objective target 1e-05. 5 seeds per case.

Fractal methods use Wave, five elites, nonperiodic bounds and no automatic restarts. Gaussian/local covariance scale is 0.2; bounded strategies use [0.0001, 0.2]. CMA uses its own population and restart schedule. BBOB instance is 1. No settings are tuned against these results.

Cloning combined uses no boundary repair; Cloning + boundary mapping uses the same cloning-guided movement with libcmaes repair of outside proposals. Both use the current engine's post-movement elite restoration. These are fractal Wave variants, not runs of the CMA-ES optimizer.

Measured storage precision: `{"cma_coordinates_bits": 64, "objective_bits": 64, "swarm_coordinates_bits": 32, "swarm_fitness_bits": 32}`. A double-precision snapshot does not imply double-precision search coordinates.

Stopping is checked after initialization and complete steps. Unused budget is retained when another complete operation cannot fit. Failed runs retain their last best objective and count as failures. CPU timings were measured during concurrent execution, not isolated latency.

Actual evaluations: 1,091,894,000. Execution errors: 0.

| Function | Method | Walkers | Runs | Target reached | Median error | Median evaluations | Median CPU seconds | Errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bbob_1 | Cloning combined | 8 | 3 | 3/3 | 9.58199e-06 | 8976 | 0.173 | 0 |
| bbob_1 | Cloning combined | 16 | 3 | 3/3 | 9.85032e-06 | 16864 | 0.262 | 0 |
| bbob_1 | Cloning combined | 32 | 2 | 2/2 | 9.75898e-06 | 33680 | 0.615 | 0 |
| bbob_1 | Cloning combined | 64 | 2 | 2/2 | 9.87694e-06 | 80160 | 1.940 | 0 |
| bbob_1 | Cloning combined | 128 | 3 | 3/3 | 9.89329e-06 | 241536 | 4.451 | 0 |
| bbob_1 | Cloning combined | 256 | 2 | 1/2 | 1.24774e-05 | 950144 | 21.312 | 0 |
| bbob_1 | Cloning combined | 512 | 3 | 0/3 | 2.1103e-05 | 999424 | 16.475 | 0 |
| bbob_1 | Cloning combined | 1000 | 3 | 0/3 | 0.00391748 | 999000 | 16.064 | 0 |
| bbob_1 | Cloning + boundary mapping | 8 | 2 | 2/2 | 9.69074e-06 | 8296 | 0.230 | 0 |
| bbob_1 | Cloning + boundary mapping | 16 | 4 | 4/4 | 9.77657e-06 | 17048 | 0.392 | 0 |
| bbob_1 | Cloning + boundary mapping | 32 | 4 | 4/4 | 9.76361e-06 | 34784 | 0.904 | 0 |
| bbob_1 | Cloning + boundary mapping | 64 | 2 | 2/2 | 9.8102e-06 | 80352 | 1.434 | 0 |
| bbob_1 | Cloning + boundary mapping | 128 | 5 | 5/5 | 9.6763e-06 | 226048 | 4.914 | 0 |
| bbob_1 | Cloning + boundary mapping | 256 | 4 | 0/4 | 1.15235e-05 | 999680 | 15.971 | 0 |
| bbob_1 | Cloning + boundary mapping | 512 | 3 | 0/3 | 2.13803e-05 | 999424 | 19.110 | 0 |
| bbob_1 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 0.00455332 | 999000 | 20.003 | 0 |
| bbob_10 | Cloning combined | 8 | 2 | 0/2 | 43.6726 | 999992 | 51.552 | 0 |
| bbob_10 | Cloning combined | 16 | 2 | 0/2 | 146.84 | 999984 | 47.360 | 0 |
| bbob_10 | Cloning combined | 32 | 2 | 0/2 | 180.882 | 999968 | 23.045 | 0 |
| bbob_10 | Cloning combined | 64 | 3 | 0/3 | 110.195 | 999936 | 23.595 | 0 |
| bbob_10 | Cloning combined | 128 | 4 | 0/4 | 299.713 | 999808 | 44.518 | 0 |
| bbob_10 | Cloning combined | 256 | 2 | 0/2 | 1223.41 | 999680 | 21.885 | 0 |
| bbob_10 | Cloning combined | 512 | 1 | 0/1 | 2596.46 | 999424 | 35.596 | 0 |
| bbob_10 | Cloning combined | 1000 | 3 | 0/3 | 13376.1 | 999000 | 31.386 | 0 |
| bbob_10 | Cloning + boundary mapping | 8 | 3 | 0/3 | 52.5488 | 999992 | 25.838 | 0 |
| bbob_10 | Cloning + boundary mapping | 16 | 4 | 0/4 | 187.87 | 999984 | 34.037 | 0 |
| bbob_10 | Cloning + boundary mapping | 32 | 5 | 0/5 | 202.547 | 999968 | 30.970 | 0 |
| bbob_10 | Cloning + boundary mapping | 64 | 4 | 0/4 | 157.954 | 999936 | 24.048 | 0 |
| bbob_10 | Cloning + boundary mapping | 128 | 3 | 0/3 | 602.238 | 999808 | 25.081 | 0 |
| bbob_10 | Cloning + boundary mapping | 256 | 2 | 0/2 | 1750.76 | 999680 | 27.477 | 0 |
| bbob_10 | Cloning + boundary mapping | 512 | 3 | 0/3 | 7277.1 | 999424 | 23.415 | 0 |
| bbob_10 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 13249.4 | 999000 | 31.914 | 0 |
| bbob_11 | Cloning combined | 8 | 1 | 0/1 | 2.13593e-05 | 999992 | 40.200 | 0 |
| bbob_11 | Cloning combined | 16 | 3 | 2/3 | 9.96256e-06 | 786112 | 18.536 | 0 |
| bbob_11 | Cloning combined | 32 | 2 | 1/2 | 1.06979e-05 | 949360 | 29.545 | 0 |
| bbob_11 | Cloning combined | 64 | 2 | 0/2 | 0.000101988 | 999936 | 30.908 | 0 |
| bbob_11 | Cloning combined | 128 | 3 | 0/3 | 2.24573 | 999808 | 36.631 | 0 |
| bbob_11 | Cloning combined | 256 | 4 | 0/4 | 15.3714 | 999680 | 23.994 | 0 |
| bbob_11 | Cloning combined | 512 | 2 | 0/2 | 53.4208 | 999424 | 24.690 | 0 |
| bbob_11 | Cloning combined | 1000 | 3 | 0/3 | 93.2442 | 999000 | 34.736 | 0 |
| bbob_11 | Cloning + boundary mapping | 8 | 4 | 3/4 | 9.77428e-06 | 771468 | 30.492 | 0 |
| bbob_11 | Cloning + boundary mapping | 16 | 4 | 4/4 | 9.5192e-06 | 768696 | 22.956 | 0 |
| bbob_11 | Cloning + boundary mapping | 32 | 2 | 2/2 | 9.56645e-06 | 932416 | 22.125 | 0 |
| bbob_11 | Cloning + boundary mapping | 64 | 1 | 0/1 | 4.38831e-05 | 999936 | 42.387 | 0 |
| bbob_11 | Cloning + boundary mapping | 128 | 3 | 0/3 | 3.34704 | 999808 | 25.682 | 0 |
| bbob_11 | Cloning + boundary mapping | 256 | 2 | 0/2 | 10.2085 | 999680 | 33.474 | 0 |
| bbob_11 | Cloning + boundary mapping | 512 | 2 | 0/2 | 45.4257 | 999424 | 24.167 | 0 |
| bbob_11 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 91.7295 | 999000 | 41.108 | 0 |
| bbob_12 | Cloning combined | 8 | 3 | 0/3 | 1.93657 | 999992 | 33.553 | 0 |
| bbob_12 | Cloning combined | 16 | 2 | 0/2 | 0.39292 | 999984 | 41.734 | 0 |
| bbob_12 | Cloning combined | 32 | 1 | 0/1 | 0.649912 | 999968 | 20.798 | 0 |
| bbob_12 | Cloning combined | 64 | 2 | 0/2 | 1.82542 | 999936 | 18.199 | 0 |
| bbob_12 | Cloning combined | 128 | 5 | 0/5 | 1.33634 | 999808 | 30.013 | 0 |
| bbob_12 | Cloning combined | 256 | 2 | 0/2 | 0.366858 | 999680 | 25.118 | 0 |
| bbob_12 | Cloning combined | 512 | 5 | 0/5 | 0.655525 | 999424 | 23.833 | 0 |
| bbob_12 | Cloning combined | 1000 | 4 | 0/4 | 789.376 | 999000 | 22.058 | 0 |
| bbob_12 | Cloning + boundary mapping | 8 | 4 | 0/4 | 1.85153 | 999992 | 24.904 | 0 |
| bbob_12 | Cloning + boundary mapping | 16 | 5 | 0/5 | 0.46243 | 999984 | 21.503 | 0 |
| bbob_12 | Cloning + boundary mapping | 32 | 4 | 0/4 | 4.40045 | 999968 | 19.640 | 0 |
| bbob_12 | Cloning + boundary mapping | 64 | 1 | 0/1 | 1.9253 | 999936 | 57.634 | 0 |
| bbob_12 | Cloning + boundary mapping | 128 | 4 | 0/4 | 0.272909 | 999808 | 27.992 | 0 |
| bbob_12 | Cloning + boundary mapping | 256 | 4 | 0/4 | 0.814713 | 999680 | 18.616 | 0 |
| bbob_12 | Cloning + boundary mapping | 512 | 4 | 0/4 | 0.14746 | 999424 | 17.557 | 0 |
| bbob_12 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 893.715 | 999000 | 31.541 | 0 |
| bbob_13 | Cloning combined | 16 | 3 | 0/3 | 1.57796 | 999984 | 20.955 | 0 |
| bbob_13 | Cloning combined | 32 | 2 | 0/2 | 7.93492 | 999968 | 17.113 | 0 |
| bbob_13 | Cloning combined | 64 | 2 | 0/2 | 11.1546 | 999936 | 21.440 | 0 |
| bbob_13 | Cloning combined | 128 | 2 | 0/2 | 7.34431 | 999808 | 20.434 | 0 |
| bbob_13 | Cloning combined | 256 | 1 | 0/1 | 0.0360544 | 999680 | 16.947 | 0 |
| bbob_13 | Cloning combined | 512 | 1 | 0/1 | 0.0719947 | 999424 | 16.218 | 0 |
| bbob_13 | Cloning combined | 1000 | 1 | 0/1 | 4.04421 | 999000 | 16.750 | 0 |
| bbob_13 | Cloning + boundary mapping | 8 | 4 | 0/4 | 31.6621 | 999992 | 20.364 | 0 |
| bbob_13 | Cloning + boundary mapping | 16 | 3 | 0/3 | 0.369142 | 999984 | 18.123 | 0 |
| bbob_13 | Cloning + boundary mapping | 32 | 2 | 0/2 | 41.059 | 999968 | 29.748 | 0 |
| bbob_13 | Cloning + boundary mapping | 64 | 3 | 0/3 | 0.936972 | 999936 | 18.976 | 0 |
| bbob_13 | Cloning + boundary mapping | 128 | 3 | 0/3 | 3.86744 | 999808 | 20.225 | 0 |
| bbob_13 | Cloning + boundary mapping | 256 | 2 | 0/2 | 0.748083 | 999680 | 27.342 | 0 |
| bbob_13 | Cloning + boundary mapping | 512 | 2 | 0/2 | 0.0270876 | 999424 | 29.574 | 0 |
| bbob_13 | Cloning + boundary mapping | 1000 | 5 | 0/5 | 7.95772 | 999000 | 19.478 | 0 |
| bbob_14 | Cloning combined | 8 | 4 | 0/4 | 0.000127171 | 999992 | 24.228 | 0 |
| bbob_14 | Cloning combined | 16 | 1 | 0/1 | 0.000102837 | 999984 | 18.940 | 0 |
| bbob_14 | Cloning combined | 32 | 3 | 0/3 | 8.91117e-05 | 999968 | 30.727 | 0 |
| bbob_14 | Cloning combined | 64 | 2 | 0/2 | 6.619e-05 | 999936 | 18.869 | 0 |
| bbob_14 | Cloning combined | 128 | 3 | 0/3 | 3.77665e-05 | 999808 | 18.054 | 0 |
| bbob_14 | Cloning combined | 256 | 3 | 0/3 | 3.34092e-05 | 999680 | 26.798 | 0 |
| bbob_14 | Cloning combined | 512 | 4 | 0/4 | 0.00146406 | 999424 | 17.334 | 0 |
| bbob_14 | Cloning combined | 1000 | 4 | 0/4 | 0.252476 | 999000 | 22.228 | 0 |
| bbob_14 | Cloning + boundary mapping | 8 | 3 | 0/3 | 0.000139213 | 999992 | 25.907 | 0 |
| bbob_14 | Cloning + boundary mapping | 32 | 5 | 0/5 | 8.64677e-05 | 999968 | 30.381 | 0 |
| bbob_14 | Cloning + boundary mapping | 64 | 2 | 0/2 | 6.8442e-05 | 999936 | 20.129 | 0 |
| bbob_14 | Cloning + boundary mapping | 128 | 3 | 0/3 | 4.02782e-05 | 999808 | 28.161 | 0 |
| bbob_14 | Cloning + boundary mapping | 256 | 1 | 0/1 | 3.46051e-05 | 999680 | 19.640 | 0 |
| bbob_14 | Cloning + boundary mapping | 512 | 3 | 0/3 | 0.000518964 | 999424 | 21.286 | 0 |
| bbob_14 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 1.4632 | 999000 | 17.141 | 0 |
| bbob_15 | Cloning combined | 8 | 2 | 0/2 | 543.233 | 999992 | 28.207 | 0 |
| bbob_15 | Cloning combined | 16 | 2 | 0/2 | 766.586 | 999984 | 24.333 | 0 |
| bbob_15 | Cloning combined | 32 | 2 | 0/2 | 529.308 | 999968 | 24.245 | 0 |
| bbob_15 | Cloning combined | 64 | 3 | 0/3 | 563.132 | 999936 | 51.830 | 0 |
| bbob_15 | Cloning combined | 128 | 2 | 0/2 | 417.873 | 999808 | 48.014 | 0 |
| bbob_15 | Cloning combined | 256 | 4 | 0/4 | 333.307 | 999680 | 31.389 | 0 |
| bbob_15 | Cloning combined | 512 | 2 | 0/2 | 340.766 | 999424 | 28.173 | 0 |
| bbob_15 | Cloning combined | 1000 | 2 | 0/2 | 188.618 | 999000 | 52.480 | 0 |
| bbob_15 | Cloning + boundary mapping | 8 | 3 | 0/3 | 744.205 | 999992 | 60.691 | 0 |
| bbob_15 | Cloning + boundary mapping | 16 | 2 | 0/2 | 645.076 | 999984 | 37.160 | 0 |
| bbob_15 | Cloning + boundary mapping | 32 | 3 | 0/3 | 582.033 | 999968 | 36.932 | 0 |
| bbob_15 | Cloning + boundary mapping | 64 | 2 | 0/2 | 501.944 | 999936 | 31.658 | 0 |
| bbob_15 | Cloning + boundary mapping | 128 | 2 | 0/2 | 446.727 | 999808 | 23.778 | 0 |
| bbob_15 | Cloning + boundary mapping | 256 | 1 | 0/1 | 532.294 | 999680 | 28.283 | 0 |
| bbob_15 | Cloning + boundary mapping | 512 | 5 | 0/5 | 376.085 | 999424 | 23.327 | 0 |
| bbob_15 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 278.6 | 999000 | 23.155 | 0 |
| bbob_16 | Cloning combined | 8 | 3 | 0/3 | 10.2087 | 999992 | 36.449 | 0 |
| bbob_16 | Cloning combined | 16 | 1 | 0/1 | 10.9628 | 999984 | 35.505 | 0 |
| bbob_16 | Cloning combined | 32 | 3 | 0/3 | 10.2481 | 999968 | 37.641 | 0 |
| bbob_16 | Cloning combined | 64 | 3 | 0/3 | 4.76161 | 999936 | 36.593 | 0 |
| bbob_16 | Cloning combined | 128 | 3 | 0/3 | 7.62474 | 999808 | 37.937 | 0 |
| bbob_16 | Cloning combined | 256 | 5 | 0/5 | 3.39674 | 999680 | 37.001 | 0 |
| bbob_16 | Cloning combined | 512 | 3 | 0/3 | 1.35817 | 999424 | 33.709 | 0 |
| bbob_16 | Cloning combined | 1000 | 3 | 0/3 | 1.48204 | 999000 | 34.203 | 0 |
| bbob_16 | Cloning + boundary mapping | 8 | 3 | 0/3 | 16.7137 | 999992 | 41.727 | 0 |
| bbob_16 | Cloning + boundary mapping | 16 | 2 | 0/2 | 6.90249 | 999984 | 39.722 | 0 |
| bbob_16 | Cloning + boundary mapping | 32 | 4 | 0/4 | 5.22507 | 999968 | 39.303 | 0 |
| bbob_16 | Cloning + boundary mapping | 64 | 3 | 0/3 | 5.44885 | 999936 | 41.519 | 0 |
| bbob_16 | Cloning + boundary mapping | 128 | 4 | 0/4 | 6.21087 | 999808 | 48.610 | 0 |
| bbob_16 | Cloning + boundary mapping | 256 | 1 | 0/1 | 0.800798 | 999680 | 35.092 | 0 |
| bbob_16 | Cloning + boundary mapping | 512 | 2 | 0/2 | 0.828505 | 999424 | 76.826 | 0 |
| bbob_16 | Cloning + boundary mapping | 1000 | 1 | 0/1 | 1.85313 | 999000 | 34.749 | 0 |
| bbob_17 | Cloning combined | 8 | 4 | 0/4 | 11.189 | 999992 | 35.959 | 0 |
| bbob_17 | Cloning combined | 16 | 3 | 0/3 | 16.7221 | 999984 | 26.026 | 0 |
| bbob_17 | Cloning combined | 32 | 3 | 0/3 | 6.87481 | 999968 | 22.577 | 0 |
| bbob_17 | Cloning combined | 64 | 4 | 0/4 | 9.11432 | 999936 | 27.612 | 0 |
| bbob_17 | Cloning combined | 128 | 3 | 0/3 | 6.78435 | 999808 | 24.107 | 0 |
| bbob_17 | Cloning combined | 256 | 3 | 0/3 | 8.47228 | 999680 | 29.387 | 0 |
| bbob_17 | Cloning combined | 512 | 2 | 0/2 | 6.49026 | 999424 | 29.543 | 0 |
| bbob_17 | Cloning combined | 1000 | 1 | 0/1 | 5.91503 | 999000 | 31.894 | 0 |
| bbob_17 | Cloning + boundary mapping | 8 | 3 | 0/3 | 9.2208 | 999992 | 25.681 | 0 |
| bbob_17 | Cloning + boundary mapping | 16 | 3 | 0/3 | 10.1803 | 999984 | 34.071 | 0 |
| bbob_17 | Cloning + boundary mapping | 32 | 3 | 0/3 | 7.69067 | 999968 | 34.561 | 0 |
| bbob_17 | Cloning + boundary mapping | 64 | 3 | 0/3 | 10.594 | 999936 | 32.149 | 0 |
| bbob_17 | Cloning + boundary mapping | 256 | 5 | 0/5 | 7.79268 | 999680 | 22.441 | 0 |
| bbob_17 | Cloning + boundary mapping | 512 | 2 | 0/2 | 5.93511 | 999424 | 27.701 | 0 |
| bbob_18 | Cloning combined | 8 | 2 | 0/2 | 43.584 | 999992 | 25.595 | 0 |
| bbob_18 | Cloning combined | 16 | 4 | 0/4 | 35.6416 | 999984 | 26.487 | 0 |
| bbob_18 | Cloning combined | 32 | 3 | 0/3 | 26.9941 | 999968 | 33.403 | 0 |
| bbob_18 | Cloning combined | 64 | 4 | 0/4 | 39.7615 | 999936 | 23.168 | 0 |
| bbob_18 | Cloning combined | 128 | 5 | 0/5 | 23.8459 | 999808 | 41.749 | 0 |
| bbob_18 | Cloning combined | 256 | 3 | 0/3 | 24.6238 | 999680 | 35.545 | 0 |
| bbob_18 | Cloning combined | 512 | 3 | 0/3 | 16.8518 | 999424 | 30.305 | 0 |
| bbob_18 | Cloning combined | 1000 | 2 | 0/2 | 20.188 | 999000 | 44.151 | 0 |
| bbob_18 | Cloning + boundary mapping | 8 | 1 | 0/1 | 65.9452 | 999992 | 24.954 | 0 |
| bbob_18 | Cloning + boundary mapping | 16 | 2 | 0/2 | 26.2747 | 999984 | 30.818 | 0 |
| bbob_18 | Cloning + boundary mapping | 32 | 3 | 0/3 | 41.7253 | 999968 | 33.017 | 0 |
| bbob_18 | Cloning + boundary mapping | 64 | 4 | 0/4 | 35.3471 | 999936 | 39.147 | 0 |
| bbob_18 | Cloning + boundary mapping | 128 | 2 | 0/2 | 29.245 | 999808 | 22.030 | 0 |
| bbob_18 | Cloning + boundary mapping | 256 | 2 | 0/2 | 32.1728 | 999680 | 21.392 | 0 |
| bbob_18 | Cloning + boundary mapping | 512 | 1 | 0/1 | 19.9177 | 999424 | 59.264 | 0 |
| bbob_18 | Cloning + boundary mapping | 1000 | 1 | 0/1 | 22.4909 | 999000 | 50.281 | 0 |
| bbob_19 | Cloning combined | 8 | 2 | 0/2 | 10.5107 | 999992 | 49.567 | 0 |
| bbob_19 | Cloning combined | 16 | 3 | 0/3 | 3.11326 | 999984 | 18.861 | 0 |
| bbob_19 | Cloning combined | 32 | 2 | 0/2 | 1.67666 | 999968 | 19.638 | 0 |
| bbob_19 | Cloning combined | 64 | 1 | 0/1 | 0.763151 | 999936 | 18.077 | 0 |
| bbob_19 | Cloning combined | 128 | 2 | 0/2 | 2.06039 | 999808 | 20.268 | 0 |
| bbob_19 | Cloning combined | 256 | 4 | 0/4 | 1.87513 | 999680 | 18.123 | 0 |
| bbob_19 | Cloning combined | 512 | 3 | 0/3 | 1.72519 | 999424 | 18.017 | 0 |
| bbob_19 | Cloning + boundary mapping | 8 | 3 | 0/3 | 11.4132 | 999992 | 32.194 | 0 |
| bbob_19 | Cloning + boundary mapping | 16 | 4 | 0/4 | 3.25787 | 999984 | 19.702 | 0 |
| bbob_19 | Cloning + boundary mapping | 32 | 2 | 0/2 | 1.78295 | 999968 | 20.557 | 0 |
| bbob_19 | Cloning + boundary mapping | 64 | 3 | 0/3 | 1.34916 | 999936 | 51.810 | 0 |
| bbob_19 | Cloning + boundary mapping | 128 | 3 | 0/3 | 1.62627 | 999808 | 18.099 | 0 |
| bbob_19 | Cloning + boundary mapping | 256 | 3 | 0/3 | 1.81569 | 999680 | 17.384 | 0 |
| bbob_19 | Cloning + boundary mapping | 512 | 3 | 0/3 | 1.56919 | 999424 | 20.261 | 0 |
| bbob_19 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 1.98691 | 999000 | 17.199 | 0 |
| bbob_2 | Cloning combined | 8 | 1 | 0/1 | 180.249 | 999992 | 34.040 | 0 |
| bbob_2 | Cloning combined | 16 | 3 | 0/3 | 114.247 | 999984 | 38.745 | 0 |
| bbob_2 | Cloning combined | 32 | 3 | 0/3 | 355.077 | 999968 | 29.654 | 0 |
| bbob_2 | Cloning combined | 64 | 4 | 0/4 | 227.353 | 999936 | 22.151 | 0 |
| bbob_2 | Cloning combined | 128 | 3 | 0/3 | 564.232 | 999808 | 22.590 | 0 |
| bbob_2 | Cloning combined | 256 | 4 | 0/4 | 2189.87 | 999680 | 22.738 | 0 |
| bbob_2 | Cloning combined | 512 | 2 | 0/2 | 980.032 | 999424 | 38.300 | 0 |
| bbob_2 | Cloning combined | 1000 | 4 | 0/4 | 4700.5 | 999000 | 24.274 | 0 |
| bbob_2 | Cloning + boundary mapping | 8 | 3 | 0/3 | 150.178 | 999992 | 27.558 | 0 |
| bbob_2 | Cloning + boundary mapping | 16 | 4 | 0/4 | 130.272 | 999984 | 24.463 | 0 |
| bbob_2 | Cloning + boundary mapping | 32 | 2 | 0/2 | 133.283 | 999968 | 29.230 | 0 |
| bbob_2 | Cloning + boundary mapping | 64 | 2 | 0/2 | 185.834 | 999936 | 27.811 | 0 |
| bbob_2 | Cloning + boundary mapping | 128 | 4 | 0/4 | 1035.43 | 999808 | 25.208 | 0 |
| bbob_2 | Cloning + boundary mapping | 256 | 2 | 0/2 | 1968.24 | 999680 | 35.462 | 0 |
| bbob_2 | Cloning + boundary mapping | 512 | 2 | 0/2 | 1202.64 | 999424 | 34.144 | 0 |
| bbob_2 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 5055.59 | 999000 | 23.464 | 0 |
| bbob_20 | Cloning combined | 8 | 3 | 0/3 | 1.44162 | 999992 | 43.541 | 0 |
| bbob_20 | Cloning combined | 16 | 4 | 0/4 | 1.6835 | 999984 | 25.623 | 0 |
| bbob_20 | Cloning combined | 32 | 3 | 0/3 | 1.46091 | 999968 | 25.005 | 0 |
| bbob_20 | Cloning combined | 64 | 3 | 0/3 | 1.10554 | 999936 | 22.658 | 0 |
| bbob_20 | Cloning combined | 128 | 3 | 0/3 | 1.3523 | 999808 | 26.119 | 0 |
| bbob_20 | Cloning combined | 256 | 4 | 0/4 | 1.20669 | 999680 | 36.430 | 0 |
| bbob_20 | Cloning combined | 512 | 3 | 0/3 | 1.61202 | 999424 | 34.777 | 0 |
| bbob_20 | Cloning combined | 1000 | 3 | 0/3 | 1.92278 | 999000 | 35.390 | 0 |
| bbob_20 | Cloning + boundary mapping | 8 | 3 | 0/3 | 1.50139 | 999992 | 43.217 | 0 |
| bbob_20 | Cloning + boundary mapping | 16 | 3 | 0/3 | 1.62893 | 999984 | 24.499 | 0 |
| bbob_20 | Cloning + boundary mapping | 32 | 3 | 0/3 | 1.21426 | 999968 | 42.228 | 0 |
| bbob_20 | Cloning + boundary mapping | 64 | 3 | 0/3 | 1.13507 | 999936 | 26.835 | 0 |
| bbob_20 | Cloning + boundary mapping | 128 | 2 | 0/2 | 0.938551 | 999808 | 36.448 | 0 |
| bbob_20 | Cloning + boundary mapping | 256 | 2 | 0/2 | 1.20097 | 999680 | 33.062 | 0 |
| bbob_20 | Cloning + boundary mapping | 512 | 3 | 0/3 | 1.67148 | 999424 | 30.961 | 0 |
| bbob_20 | Cloning + boundary mapping | 1000 | 1 | 0/1 | 2.07579 | 999000 | 24.551 | 0 |
| bbob_21 | Cloning combined | 8 | 3 | 0/3 | 7.31756 | 999992 | 79.087 | 0 |
| bbob_21 | Cloning combined | 16 | 4 | 0/4 | 5.86855 | 999984 | 30.360 | 0 |
| bbob_21 | Cloning combined | 32 | 3 | 0/3 | 16.4002 | 999968 | 24.152 | 0 |
| bbob_21 | Cloning combined | 64 | 2 | 0/2 | 2.47297 | 999936 | 52.635 | 0 |
| bbob_21 | Cloning combined | 128 | 2 | 0/2 | 8.54605 | 999808 | 30.775 | 0 |
| bbob_21 | Cloning combined | 256 | 5 | 0/5 | 8.72509 | 999680 | 25.804 | 0 |
| bbob_21 | Cloning combined | 512 | 3 | 2/3 | 9.2666e-06 | 944640 | 39.427 | 0 |
| bbob_21 | Cloning combined | 1000 | 3 | 0/3 | 1.80387 | 999000 | 26.902 | 0 |
| bbob_21 | Cloning + boundary mapping | 8 | 3 | 0/3 | 27.8876 | 999992 | 28.803 | 0 |
| bbob_21 | Cloning + boundary mapping | 16 | 2 | 0/2 | 1.96083 | 999984 | 32.035 | 0 |
| bbob_21 | Cloning + boundary mapping | 32 | 1 | 0/1 | 8.72509 | 999968 | 24.644 | 0 |
| bbob_21 | Cloning + boundary mapping | 64 | 3 | 0/3 | 3.16981 | 999936 | 36.650 | 0 |
| bbob_21 | Cloning + boundary mapping | 128 | 3 | 0/3 | 27.8876 | 999808 | 25.967 | 0 |
| bbob_21 | Cloning + boundary mapping | 512 | 4 | 2/4 | 1.0464e-05 | 877568 | 20.998 | 0 |
| bbob_21 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 3.17303 | 999000 | 23.230 | 0 |
| bbob_22 | Cloning combined | 8 | 2 | 0/2 | 8.26993 | 999992 | 29.590 | 0 |
| bbob_22 | Cloning combined | 16 | 2 | 0/2 | 11.8672 | 999984 | 38.259 | 0 |
| bbob_22 | Cloning combined | 32 | 1 | 0/1 | 1.95501 | 999968 | 18.642 | 0 |
| bbob_22 | Cloning combined | 64 | 2 | 0/2 | 40.2245 | 999936 | 23.990 | 0 |
| bbob_22 | Cloning combined | 128 | 3 | 0/3 | 1.95503 | 999808 | 20.096 | 0 |
| bbob_22 | Cloning combined | 256 | 1 | 0/1 | 14.585 | 999680 | 24.326 | 0 |
| bbob_22 | Cloning combined | 512 | 3 | 0/3 | 0.00230784 | 999424 | 17.196 | 0 |
| bbob_22 | Cloning combined | 1000 | 3 | 0/3 | 0.694384 | 999000 | 26.726 | 0 |
| bbob_22 | Cloning + boundary mapping | 8 | 4 | 0/4 | 1.95501 | 999992 | 31.251 | 0 |
| bbob_22 | Cloning + boundary mapping | 16 | 4 | 0/4 | 18.2883 | 999984 | 25.661 | 0 |
| bbob_22 | Cloning + boundary mapping | 32 | 3 | 0/3 | 1.95501 | 999968 | 18.762 | 0 |
| bbob_22 | Cloning + boundary mapping | 64 | 4 | 0/4 | 1.95502 | 999936 | 19.437 | 0 |
| bbob_22 | Cloning + boundary mapping | 128 | 3 | 0/3 | 14.5849 | 999808 | 20.421 | 0 |
| bbob_22 | Cloning + boundary mapping | 256 | 4 | 0/4 | 22.3337 | 999680 | 29.538 | 0 |
| bbob_22 | Cloning + boundary mapping | 512 | 1 | 0/1 | 0.00380311 | 999424 | 30.510 | 0 |
| bbob_22 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 0.693894 | 999000 | 17.769 | 0 |
| bbob_23 | Cloning combined | 8 | 2 | 0/2 | 0.690787 | 999992 | 56.728 | 0 |
| bbob_23 | Cloning combined | 16 | 3 | 0/3 | 0.467021 | 999984 | 58.961 | 0 |
| bbob_23 | Cloning combined | 32 | 5 | 0/5 | 0.428597 | 999968 | 90.169 | 0 |
| bbob_23 | Cloning combined | 128 | 3 | 0/3 | 0.271365 | 999808 | 77.019 | 0 |
| bbob_23 | Cloning combined | 256 | 3 | 0/3 | 0.0712995 | 999680 | 81.286 | 0 |
| bbob_23 | Cloning combined | 512 | 1 | 0/1 | 0.0281634 | 999424 | 55.868 | 0 |
| bbob_23 | Cloning combined | 1000 | 3 | 0/3 | 0.0637794 | 999000 | 88.054 | 0 |
| bbob_23 | Cloning + boundary mapping | 8 | 2 | 0/2 | 0.921272 | 999992 | 80.109 | 0 |
| bbob_23 | Cloning + boundary mapping | 16 | 4 | 0/4 | 0.668535 | 999984 | 58.762 | 0 |
| bbob_23 | Cloning + boundary mapping | 32 | 4 | 0/4 | 0.276884 | 999968 | 58.005 | 0 |
| bbob_23 | Cloning + boundary mapping | 64 | 3 | 0/3 | 0.20864 | 999936 | 59.804 | 0 |
| bbob_23 | Cloning + boundary mapping | 128 | 3 | 0/3 | 0.25734 | 999808 | 160.699 | 0 |
| bbob_23 | Cloning + boundary mapping | 256 | 3 | 0/3 | 0.197078 | 999680 | 56.296 | 0 |
| bbob_23 | Cloning + boundary mapping | 512 | 3 | 0/3 | 0.0138466 | 999424 | 88.509 | 0 |
| bbob_23 | Cloning + boundary mapping | 1000 | 4 | 0/4 | 0.0280468 | 999000 | 59.545 | 0 |
| bbob_24 | Cloning combined | 8 | 3 | 0/3 | 425.256 | 999992 | 22.486 | 0 |
| bbob_24 | Cloning combined | 16 | 2 | 0/2 | 390.957 | 999984 | 27.533 | 0 |
| bbob_24 | Cloning combined | 32 | 3 | 0/3 | 353.965 | 999968 | 23.081 | 0 |
| bbob_24 | Cloning combined | 64 | 5 | 0/5 | 306.496 | 999936 | 23.573 | 0 |
| bbob_24 | Cloning combined | 128 | 4 | 0/4 | 310.879 | 999808 | 19.160 | 0 |
| bbob_24 | Cloning combined | 256 | 2 | 0/2 | 255.802 | 999680 | 24.544 | 0 |
| bbob_24 | Cloning combined | 512 | 2 | 0/2 | 280.637 | 999424 | 29.356 | 0 |
| bbob_24 | Cloning combined | 1000 | 2 | 0/2 | 198.299 | 999000 | 22.023 | 0 |
| bbob_24 | Cloning + boundary mapping | 8 | 2 | 0/2 | 433.904 | 999992 | 30.655 | 0 |
| bbob_24 | Cloning + boundary mapping | 16 | 4 | 0/4 | 382.06 | 999984 | 25.762 | 0 |
| bbob_24 | Cloning + boundary mapping | 32 | 5 | 0/5 | 338.523 | 999968 | 25.524 | 0 |
| bbob_24 | Cloning + boundary mapping | 64 | 3 | 0/3 | 316.513 | 999936 | 47.089 | 0 |
| bbob_24 | Cloning + boundary mapping | 128 | 3 | 0/3 | 294.246 | 999808 | 30.644 | 0 |
| bbob_24 | Cloning + boundary mapping | 256 | 3 | 0/3 | 272.769 | 999680 | 20.534 | 0 |
| bbob_24 | Cloning + boundary mapping | 512 | 4 | 0/4 | 251.111 | 999424 | 29.011 | 0 |
| bbob_24 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 237.987 | 999000 | 19.469 | 0 |
| bbob_3 | Cloning combined | 8 | 1 | 0/1 | 534.265 | 999992 | 27.380 | 0 |
| bbob_3 | Cloning combined | 16 | 3 | 0/3 | 570.091 | 999984 | 23.569 | 0 |
| bbob_3 | Cloning combined | 32 | 4 | 0/4 | 477.57 | 999968 | 26.007 | 0 |
| bbob_3 | Cloning combined | 64 | 5 | 0/5 | 408.922 | 999936 | 25.930 | 0 |
| bbob_3 | Cloning combined | 128 | 4 | 0/4 | 402.454 | 999808 | 38.581 | 0 |
| bbob_3 | Cloning combined | 256 | 2 | 0/2 | 334.796 | 999680 | 23.299 | 0 |
| bbob_3 | Cloning combined | 512 | 1 | 0/1 | 269.632 | 999424 | 22.589 | 0 |
| bbob_3 | Cloning combined | 1000 | 2 | 0/2 | 250.232 | 999000 | 40.822 | 0 |
| bbob_3 | Cloning + boundary mapping | 8 | 4 | 0/4 | 639.907 | 999992 | 44.305 | 0 |
| bbob_3 | Cloning + boundary mapping | 16 | 5 | 0/5 | 626.801 | 999984 | 26.124 | 0 |
| bbob_3 | Cloning + boundary mapping | 32 | 1 | 0/1 | 350.219 | 999968 | 23.842 | 0 |
| bbob_3 | Cloning + boundary mapping | 64 | 1 | 0/1 | 391.009 | 999936 | 28.540 | 0 |
| bbob_3 | Cloning + boundary mapping | 128 | 1 | 0/1 | 461.651 | 999808 | 45.335 | 0 |
| bbob_3 | Cloning + boundary mapping | 256 | 2 | 0/2 | 490.995 | 999680 | 24.271 | 0 |
| bbob_3 | Cloning + boundary mapping | 512 | 2 | 0/2 | 410.412 | 999424 | 25.959 | 0 |
| bbob_3 | Cloning + boundary mapping | 1000 | 1 | 0/1 | 323.358 | 999000 | 34.860 | 0 |
| bbob_4 | Cloning combined | 8 | 3 | 0/3 | 760.107 | 999992 | 30.754 | 0 |
| bbob_4 | Cloning combined | 16 | 2 | 0/2 | 589.498 | 999984 | 26.872 | 0 |
| bbob_4 | Cloning combined | 32 | 2 | 0/2 | 506.412 | 999968 | 25.402 | 0 |
| bbob_4 | Cloning combined | 64 | 2 | 0/2 | 527.993 | 999936 | 23.154 | 0 |
| bbob_4 | Cloning combined | 128 | 4 | 0/4 | 612.37 | 999808 | 35.855 | 0 |
| bbob_4 | Cloning combined | 256 | 4 | 0/4 | 567.101 | 999680 | 31.272 | 0 |
| bbob_4 | Cloning combined | 512 | 4 | 0/4 | 348.228 | 999424 | 23.702 | 0 |
| bbob_4 | Cloning combined | 1000 | 3 | 0/3 | 474.588 | 999000 | 49.291 | 0 |
| bbob_4 | Cloning + boundary mapping | 8 | 2 | 0/2 | 620.333 | 999992 | 47.874 | 0 |
| bbob_4 | Cloning + boundary mapping | 16 | 3 | 0/3 | 828.764 | 999984 | 26.791 | 0 |
| bbob_4 | Cloning + boundary mapping | 32 | 4 | 0/4 | 806.359 | 999968 | 24.969 | 0 |
| bbob_4 | Cloning + boundary mapping | 64 | 4 | 0/4 | 592.979 | 999936 | 45.590 | 0 |
| bbob_4 | Cloning + boundary mapping | 128 | 2 | 0/2 | 473.582 | 999808 | 24.180 | 0 |
| bbob_4 | Cloning + boundary mapping | 256 | 3 | 0/3 | 565.14 | 999680 | 41.825 | 0 |
| bbob_4 | Cloning + boundary mapping | 512 | 3 | 0/3 | 509.408 | 999424 | 38.943 | 0 |
| bbob_4 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 342.659 | 999000 | 25.850 | 0 |
| bbob_5 | Cloning combined | 8 | 3 | 0/3 | 64.2596 | 999992 | 28.868 | 0 |
| bbob_5 | Cloning combined | 16 | 3 | 0/3 | 88.0589 | 999984 | 16.370 | 0 |
| bbob_5 | Cloning combined | 32 | 2 | 0/2 | 85.7212 | 999968 | 19.683 | 0 |
| bbob_5 | Cloning combined | 64 | 3 | 0/3 | 52.2257 | 999936 | 19.788 | 0 |
| bbob_5 | Cloning combined | 128 | 3 | 0/3 | 44.8206 | 999808 | 23.666 | 0 |
| bbob_5 | Cloning combined | 256 | 3 | 0/3 | 23.657 | 999680 | 13.529 | 0 |
| bbob_5 | Cloning combined | 512 | 2 | 0/2 | 43.4328 | 999424 | 13.862 | 0 |
| bbob_5 | Cloning combined | 1000 | 2 | 0/2 | 51.0212 | 999000 | 15.593 | 0 |
| bbob_5 | Cloning + boundary mapping | 8 | 4 | 0/4 | 37.5287 | 999992 | 43.966 | 0 |
| bbob_5 | Cloning + boundary mapping | 16 | 3 | 0/3 | 33.7516 | 999984 | 17.434 | 0 |
| bbob_5 | Cloning + boundary mapping | 32 | 3 | 0/3 | 8.73857 | 999968 | 35.321 | 0 |
| bbob_5 | Cloning + boundary mapping | 64 | 2 | 0/2 | 0.392532 | 999936 | 16.341 | 0 |
| bbob_5 | Cloning + boundary mapping | 128 | 1 | 0/1 | 0.0719775 | 999808 | 17.667 | 0 |
| bbob_5 | Cloning + boundary mapping | 256 | 3 | 0/3 | 21.8763 | 999680 | 18.111 | 0 |
| bbob_5 | Cloning + boundary mapping | 512 | 5 | 0/5 | 33.5128 | 999424 | 22.637 | 0 |
| bbob_5 | Cloning + boundary mapping | 1000 | 2 | 0/2 | 59.2366 | 999000 | 22.755 | 0 |
| bbob_6 | Cloning combined | 8 | 4 | 4/4 | 9.19215e-06 | 159448 | 5.251 | 0 |
| bbob_6 | Cloning combined | 16 | 2 | 2/2 | 9.73555e-06 | 111568 | 2.258 | 0 |
| bbob_6 | Cloning combined | 32 | 3 | 3/3 | 8.81643e-06 | 280160 | 5.589 | 0 |
| bbob_6 | Cloning combined | 64 | 2 | 2/2 | 9.46745e-06 | 873600 | 25.442 | 0 |
| bbob_6 | Cloning combined | 128 | 5 | 0/5 | 0.89087 | 999808 | 20.539 | 0 |
| bbob_6 | Cloning combined | 256 | 4 | 0/4 | 6.26603 | 999680 | 32.502 | 0 |
| bbob_6 | Cloning combined | 512 | 2 | 0/2 | 13.242 | 999424 | 21.104 | 0 |
| bbob_6 | Cloning combined | 1000 | 2 | 0/2 | 20.7455 | 999000 | 26.737 | 0 |
| bbob_6 | Cloning + boundary mapping | 8 | 2 | 2/2 | 9.42503e-06 | 150408 | 3.439 | 0 |
| bbob_6 | Cloning + boundary mapping | 16 | 2 | 2/2 | 8.87963e-06 | 126584 | 3.241 | 0 |
| bbob_6 | Cloning + boundary mapping | 32 | 3 | 2/3 | 9.76552e-06 | 450624 | 8.565 | 0 |
| bbob_6 | Cloning + boundary mapping | 64 | 4 | 2/4 | 1.22898e-05 | 990016 | 26.912 | 0 |
| bbob_6 | Cloning + boundary mapping | 128 | 4 | 0/4 | 1.5977 | 999808 | 24.141 | 0 |
| bbob_6 | Cloning + boundary mapping | 256 | 1 | 0/1 | 6.70987 | 999680 | 19.327 | 0 |
| bbob_6 | Cloning + boundary mapping | 512 | 3 | 0/3 | 13.5388 | 999424 | 19.438 | 0 |
| bbob_6 | Cloning + boundary mapping | 1000 | 3 | 0/3 | 58.15 | 999000 | 18.046 | 0 |
| bbob_7 | Cloning combined | 8 | 2 | 0/2 | 12.865 | 999992 | 29.420 | 0 |
| bbob_7 | Cloning combined | 16 | 3 | 0/3 | 15.5387 | 999984 | 19.383 | 0 |
| bbob_7 | Cloning combined | 32 | 4 | 0/4 | 3.40568 | 999968 | 22.883 | 0 |
| bbob_7 | Cloning combined | 64 | 2 | 0/2 | 2.98673 | 999936 | 20.489 | 0 |
| bbob_7 | Cloning combined | 128 | 1 | 0/1 | 2.24562 | 999808 | 17.158 | 0 |
| bbob_7 | Cloning combined | 256 | 4 | 0/4 | 7.85306 | 999680 | 24.336 | 0 |
| bbob_7 | Cloning combined | 512 | 2 | 0/2 | 10.8943 | 999424 | 27.227 | 0 |
| bbob_7 | Cloning combined | 1000 | 3 | 0/3 | 19.5029 | 999000 | 19.483 | 0 |
| bbob_7 | Cloning + boundary mapping | 8 | 2 | 0/2 | 30.2722 | 999992 | 22.943 | 0 |
| bbob_7 | Cloning + boundary mapping | 16 | 1 | 0/1 | 4.27568 | 999984 | 21.042 | 0 |
| bbob_7 | Cloning + boundary mapping | 32 | 3 | 0/3 | 4.47675 | 999968 | 21.233 | 0 |
| bbob_7 | Cloning + boundary mapping | 64 | 1 | 0/1 | 1.86334 | 999936 | 48.474 | 0 |
| bbob_7 | Cloning + boundary mapping | 128 | 5 | 0/5 | 1.57797 | 999808 | 27.007 | 0 |
| bbob_7 | Cloning + boundary mapping | 256 | 2 | 0/2 | 4.98699 | 999680 | 22.951 | 0 |
| bbob_7 | Cloning + boundary mapping | 512 | 4 | 0/4 | 7.17962 | 999424 | 23.890 | 0 |
| bbob_7 | Cloning + boundary mapping | 1000 | 2 | 0/2 | 17.0624 | 999000 | 22.479 | 0 |
| bbob_8 | Cloning combined | 8 | 3 | 0/3 | 0.0177097 | 999992 | 28.077 | 0 |
| bbob_8 | Cloning combined | 16 | 2 | 0/2 | 4.03033 | 999984 | 17.309 | 0 |
| bbob_8 | Cloning combined | 32 | 1 | 0/1 | 0.0348753 | 999968 | 18.230 | 0 |
| bbob_8 | Cloning combined | 64 | 1 | 0/1 | 75.4828 | 999936 | 18.053 | 0 |
| bbob_8 | Cloning combined | 128 | 2 | 0/2 | 2.31053 | 999808 | 30.775 | 0 |
| bbob_8 | Cloning combined | 256 | 2 | 0/2 | 14.8171 | 999680 | 18.562 | 0 |
| bbob_8 | Cloning combined | 512 | 4 | 0/4 | 19.1924 | 999424 | 27.758 | 0 |
| bbob_8 | Cloning combined | 1000 | 3 | 0/3 | 37.6433 | 999000 | 19.304 | 0 |
| bbob_8 | Cloning + boundary mapping | 8 | 1 | 0/1 | 0.0181109 | 999992 | 33.376 | 0 |
| bbob_8 | Cloning + boundary mapping | 16 | 4 | 0/4 | 0.049756 | 999984 | 23.494 | 0 |
| bbob_8 | Cloning + boundary mapping | 32 | 3 | 0/3 | 3.08061 | 999968 | 18.972 | 0 |
| bbob_8 | Cloning + boundary mapping | 64 | 4 | 0/4 | 11.2245 | 999936 | 17.748 | 0 |
| bbob_8 | Cloning + boundary mapping | 128 | 1 | 0/1 | 0.591849 | 999808 | 36.147 | 0 |
| bbob_8 | Cloning + boundary mapping | 256 | 2 | 0/2 | 12.7559 | 999680 | 19.924 | 0 |
| bbob_8 | Cloning + boundary mapping | 1000 | 4 | 0/4 | 81.1741 | 999000 | 21.619 | 0 |
| bbob_9 | Cloning combined | 8 | 3 | 0/3 | 4.00184 | 999992 | 21.349 | 0 |
| bbob_9 | Cloning combined | 16 | 4 | 0/4 | 0.0396771 | 999984 | 23.040 | 0 |
| bbob_9 | Cloning combined | 32 | 2 | 0/2 | 3.1914 | 999968 | 29.813 | 0 |
| bbob_9 | Cloning combined | 64 | 1 | 0/1 | 9.25678 | 999936 | 15.553 | 0 |
| bbob_9 | Cloning combined | 128 | 3 | 0/3 | 0.616599 | 999808 | 28.549 | 0 |
| bbob_9 | Cloning combined | 256 | 3 | 0/3 | 11.3126 | 999680 | 18.319 | 0 |
| bbob_9 | Cloning combined | 512 | 1 | 0/1 | 18.3898 | 999424 | 14.926 | 0 |
| bbob_9 | Cloning combined | 1000 | 2 | 0/2 | 145.858 | 999000 | 31.723 | 0 |
| bbob_9 | Cloning + boundary mapping | 8 | 4 | 0/4 | 0.0151792 | 999992 | 21.129 | 0 |
| bbob_9 | Cloning + boundary mapping | 16 | 3 | 0/3 | 0.0386864 | 999984 | 26.956 | 0 |
| bbob_9 | Cloning + boundary mapping | 32 | 3 | 0/3 | 4.67024 | 999968 | 23.955 | 0 |
| bbob_9 | Cloning + boundary mapping | 64 | 1 | 0/1 | 6.49629 | 999936 | 27.102 | 0 |
| bbob_9 | Cloning + boundary mapping | 128 | 4 | 0/4 | 2.58651 | 999808 | 15.295 | 0 |
| bbob_9 | Cloning + boundary mapping | 256 | 1 | 0/1 | 14.2746 | 999680 | 41.590 | 0 |
| bbob_9 | Cloning + boundary mapping | 512 | 4 | 0/4 | 18.2044 | 999424 | 19.803 | 0 |
| bbob_9 | Cloning + boundary mapping | 1000 | 4 | 0/4 | 51.7006 | 999000 | 20.234 | 0 |
| quadratic | Cloning combined | 8 | 2 | 2/2 | 9.77396e-06 | 8580 | 0.200 | 0 |
| quadratic | Cloning combined | 16 | 2 | 2/2 | 9.75963e-06 | 20304 | 0.455 | 0 |
| quadratic | Cloning combined | 32 | 3 | 3/3 | 9.45598e-06 | 40160 | 0.631 | 0 |
| quadratic | Cloning combined | 64 | 2 | 2/2 | 9.83267e-06 | 94656 | 2.114 | 0 |
| quadratic | Cloning combined | 256 | 4 | 4/4 | 9.90253e-06 | 450432 | 7.932 | 0 |
| quadratic | Cloning combined | 512 | 5 | 0/5 | 1.77733e-05 | 999424 | 19.640 | 0 |
| quadratic | Cloning combined | 1000 | 2 | 0/2 | 0.580556 | 999000 | 24.767 | 0 |
| quadratic | Cloning + boundary mapping | 8 | 1 | 1/1 | 9.74142e-06 | 9488 | 0.279 | 0 |
| quadratic | Cloning + boundary mapping | 16 | 2 | 2/2 | 9.80417e-06 | 20544 | 0.486 | 0 |
| quadratic | Cloning + boundary mapping | 32 | 4 | 4/4 | 9.95459e-06 | 39456 | 0.946 | 0 |
| quadratic | Cloning + boundary mapping | 64 | 3 | 3/3 | 9.91995e-06 | 84352 | 1.774 | 0 |
| quadratic | Cloning + boundary mapping | 128 | 3 | 3/3 | 9.85333e-06 | 208640 | 3.501 | 0 |
| quadratic | Cloning + boundary mapping | 256 | 2 | 2/2 | 9.89253e-06 | 480768 | 9.934 | 0 |
| quadratic | Cloning + boundary mapping | 512 | 2 | 1/2 | 3.08863e-05 | 947712 | 13.721 | 0 |
| quadratic | Cloning + boundary mapping | 1000 | 3 | 0/3 | 1.00757 | 999000 | 17.730 | 0 |
| rastrigin | Cloning combined | 8 | 4 | 0/4 | 120.887 | 999992 | 21.589 | 0 |
| rastrigin | Cloning combined | 16 | 2 | 0/2 | 143.771 | 999984 | 17.040 | 0 |
| rastrigin | Cloning combined | 32 | 1 | 0/1 | 103.475 | 999968 | 24.250 | 0 |
| rastrigin | Cloning combined | 64 | 1 | 0/1 | 93.5258 | 999936 | 16.083 | 0 |
| rastrigin | Cloning combined | 128 | 2 | 0/2 | 111.435 | 999808 | 17.638 | 0 |
| rastrigin | Cloning combined | 256 | 2 | 0/2 | 92.0333 | 999680 | 18.772 | 0 |
| rastrigin | Cloning combined | 512 | 4 | 0/4 | 87.0586 | 999424 | 14.499 | 0 |
| rastrigin | Cloning combined | 1000 | 3 | 0/3 | 83.576 | 999000 | 23.789 | 0 |
| rastrigin | Cloning + boundary mapping | 8 | 3 | 0/3 | 109.445 | 999992 | 29.248 | 0 |
| rastrigin | Cloning + boundary mapping | 16 | 3 | 0/3 | 127.354 | 999984 | 19.042 | 0 |
| rastrigin | Cloning + boundary mapping | 32 | 2 | 0/2 | 137.304 | 999968 | 41.472 | 0 |
| rastrigin | Cloning + boundary mapping | 64 | 3 | 0/3 | 116.41 | 999936 | 17.194 | 0 |
| rastrigin | Cloning + boundary mapping | 128 | 2 | 0/2 | 112.927 | 999808 | 24.216 | 0 |
| rastrigin | Cloning + boundary mapping | 256 | 2 | 0/2 | 109.445 | 999680 | 22.198 | 0 |
| rastrigin | Cloning + boundary mapping | 512 | 5 | 0/5 | 83.5763 | 999424 | 17.725 | 0 |
| rastrigin | Cloning + boundary mapping | 1000 | 2 | 0/2 | 96.0132 | 999000 | 21.846 | 0 |
| rosenbrock | Cloning combined | 8 | 2 | 0/2 | 0.0150536 | 999992 | 18.607 | 0 |
| rosenbrock | Cloning combined | 16 | 3 | 0/3 | 0.0788952 | 999984 | 18.474 | 0 |
| rosenbrock | Cloning combined | 32 | 2 | 0/2 | 2.95852 | 999968 | 17.374 | 0 |
| rosenbrock | Cloning combined | 64 | 1 | 0/1 | 11.4616 | 999936 | 17.476 | 0 |
| rosenbrock | Cloning combined | 256 | 3 | 0/3 | 16.6197 | 999680 | 23.657 | 0 |
| rosenbrock | Cloning combined | 512 | 2 | 0/2 | 312.526 | 999424 | 14.076 | 0 |
| rosenbrock | Cloning combined | 1000 | 1 | 0/1 | 427.024 | 999000 | 21.017 | 0 |
| rosenbrock | Cloning + boundary mapping | 8 | 2 | 0/2 | 2.00964 | 999992 | 19.930 | 0 |
| rosenbrock | Cloning + boundary mapping | 16 | 2 | 0/2 | 2.03063 | 999984 | 16.791 | 0 |
| rosenbrock | Cloning + boundary mapping | 32 | 3 | 0/3 | 4.14029 | 999968 | 18.221 | 0 |
| rosenbrock | Cloning + boundary mapping | 64 | 1 | 0/1 | 10.5107 | 999936 | 22.516 | 0 |
| rosenbrock | Cloning + boundary mapping | 128 | 2 | 0/2 | 0.479969 | 999808 | 20.615 | 0 |
| rosenbrock | Cloning + boundary mapping | 256 | 2 | 0/2 | 15.8797 | 999680 | 20.078 | 0 |
| rosenbrock | Cloning + boundary mapping | 512 | 2 | 0/2 | 238.791 | 999424 | 21.903 | 0 |
