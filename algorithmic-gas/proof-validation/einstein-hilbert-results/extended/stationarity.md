# Native Einstein–Hilbert extended simulation

Reference preset: h=0.002, T=0.33, viscosity=3, d=3, cloning every 20 steps, f64, coincident start at rest. Each run executes 500000 steps (time 1000). Snapshot windows contain dependent observations; no iid error bars or stationarity p-values are assigned to walkers or time samples.

Checked 1500000 steps; 0 exact failures; maximum normalized identity residual 2.299e-13.

| N | Seed | Mean W, time 500–750 | Mean W, time 750–1000 | Late curvature 10/50/90% |
|---:|---:|---:|---:|:---|
| 32 | 1729 | 25.9727 | 47.8467 | -2.735 / 0.011 / 2.969 |
| 32 | 7 | 31.3754 | 59.0732 | -3.320 / 0.068 / 3.443 |
| 32 | 991 | 19.8521 | 29.8423 | -3.641 / 0.203 / 3.595 |

The JSON records every window's signed cloning, transport, ballistic and thermal contributions, both martingale residuals, curvature/volume/reward quantiles, and distances between centered empirical distributions. RMS-normalized shape distances are a separate diagnostic; they do not establish convergence of the unscaled centered measure. The martingale check uses EH.D10 with a fixed grid of 420 radius/lambda pairs, a union bound over that grid, and a 1% failure probability per run in the ideal-innovation interpretation.
