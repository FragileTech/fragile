# Native Einstein–Hilbert extended simulation

Reference preset: h=0.002, T=0.33, viscosity=3, d=3, cloning every 20 steps, f64, coincident start at rest. Each run executes 50000 steps (time 100). Snapshot windows contain dependent observations; no iid error bars or stationarity p-values are assigned to walkers or time samples.

Checked 450000 steps; 0 exact failures; maximum normalized identity residual 1.059e-12.

| N | Seed | Mean W, time 50–75 | Mean W, time 75–100 | Late curvature 10/50/90% |
|---:|---:|---:|---:|:---|
| 128 | 1729 | 22.7840 | 27.8642 | -2.908 / 0.195 / 3.056 |
| 128 | 7 | 19.5342 | 22.4773 | -2.659 / 0.148 / 2.952 |
| 128 | 991 | 22.4208 | 23.9619 | -2.480 / 0.118 / 2.667 |
| 32 | 1729 | 3.2460 | 4.1686 | -2.985 / 0.100 / 2.961 |
| 32 | 7 | 5.0842 | 8.4016 | -2.916 / 0.089 / 2.812 |
| 32 | 991 | 6.9336 | 9.1368 | -3.344 / 0.134 / 3.307 |
| 500 | 1729 | 71.4939 | 107.4479 | -2.109 / 0.095 / 2.524 |
| 500 | 7 | 68.5014 | 106.8235 | -2.057 / 0.082 / 2.411 |
| 500 | 991 | 58.8572 | 87.1110 | -2.187 / 0.085 / 2.533 |

The JSON records every window's signed cloning, transport, ballistic and thermal contributions, both martingale residuals, curvature/volume/reward quantiles, and distances between centered empirical distributions. RMS-normalized shape distances are a separate diagnostic; they do not establish convergence of the unscaled centered measure. The martingale check uses EH.D10 with a fixed grid of 420 radius/lambda pairs, a union bound over that grid, and a 1% failure probability per run in the ideal-innovation interpretation.
