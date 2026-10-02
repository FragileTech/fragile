# Landscape review: exploratory lab runs

These are fresh runs of the compiled CPU/WASM engine used by the Euclidean Gas
lab, called directly through BrowserGas. They are not a substitute algorithm.
The checked-in euclidean_d2_dt0.04.json configuration is used with h changed
to 0.01 and the boundary changed to unbounded. Each run has 64 walkers in 2D,
1,500 complete updates, and seed 7 or 19. Initial coordinates are uniform in
[center-0.2, center+0.2]. All other preset dynamics are retained, including
cloning, jitter, component collisions, friction, OU noise, final position noise,
and the velocity cap. Sphere uses U=|x|². Rastrigin uses the standard amplitude 10.
The benchmark supplies both the minimized objective and its gradient.

Statistics use the current full swarm, not a best-so-far archive. Observations
are every 10 updates; late means average the 50 observations at updates
1010 through 1500. Upward counts compare successive observed swarm means.

| Objective | Initial coordinate center | Seed | Initial mean energy | Late mean energy | Late mean squared radius | Upward observations |
|---|---:|---:|---:|---:|---:|---:|
| sphere | 0 | 7 | 0.0270 | 0.0468 | 0.0468 | 72/150 |
| sphere | 0 | 19 | 0.0276 | 0.0454 | 0.0454 | 77/150 |
| sphere | 3 | 7 | 17.9711 | 0.0469 | 0.0469 | 75/150 |
| sphere | 3 | 19 | 18.0241 | 0.0441 | 0.0441 | 74/150 |
| rastrigin | 0 | 7 | 4.9578 | 2.4732 | 0.0138 | 72/150 |
| rastrigin | 0 | 19 | 5.0579 | 2.4566 | 0.0137 | 71/150 |
| rastrigin | 3 | 7 | 22.9019 | 20.3507 | 17.8439 | 79/150 |
| rastrigin | 3 | 19 | 23.0545 | 20.4289 | 17.8209 | 74/150 |

## Interpretation limits

Finite runs can illustrate confinement and dependence on initialization. They
cannot establish uniform-in-time moment bounds, permanent trapping, stationary
convergence, an exact Gibbs profile, or transitions on much longer timescales.
There are only two seeds per initial region. No confidence intervals are inferred
from correlated time samples. The nearest-integer well counts saved in results.json
are rough Rastrigin basin diagnostics, not exact dynamical basins.

The unbounded boundary choice ensures apparent confinement is not imposed by
box killing or reflection. A Gibbs claim needs a declared density and temperature,
followed by a full-update invariance or residual calculation; none is assumed here.

Reproduce from the repository root with `node docs/research/landscape_review/run.mjs`.
Raw configurations and sampled trajectories are in results.json. The plots show
all eight trajectories without smoothing.
