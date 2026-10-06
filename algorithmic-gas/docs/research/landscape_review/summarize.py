"""Plot the saved lab-engine runs; no simulation or fitting is performed here."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
runs = json.loads((root / "results.json").read_text())
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for run in runs:
    s = run["summary"]
    row = int(s["benchmark"] == "rastrigin")
    color = "#156b91" if s["center"] == 0 else "#c65b25"
    style = "-" if s["seed"] == 7 else "--"
    label = f"Start near ({s['center']}, {s['center']}), seed {s['seed']}"
    xs = [t["step"] for t in run["series"]]
    axes[row, 0].plot(xs, [t["mean"] for t in run["series"]], style,
                      color=color, label=label, linewidth=1.1)
    axes[row, 1].plot(xs, [t["radius2"] for t in run["series"]], style,
                      color=color, label=label, linewidth=1.1)
for row, name in enumerate(["Quadratic bowl: U = |x|²", "Rastrigin"]):
    axes[row, 0].set_title(name + " — current mean energy")
    axes[row, 1].set_title(name + " — current mean squared radius")
    for ax in axes[row]:
        ax.set_yscale("log")
        ax.grid(alpha=.2)
        ax.set_xlabel("Complete updates (h = 0.01)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Euclidean Gas lab engine: 64 walkers, 2D, unbounded domain\n"
             "Cloning, component collisions, BAOAB, position noise and smooth cap enabled")
fig.tight_layout()
fig.savefig(root / "trajectories.png", dpi=160)
fig.savefig(root / "trajectories.pdf")

rows = []
for run in runs:
    s = run["summary"]
    rows.append(f"| {s['benchmark']} | {s['center']} | {s['seed']} | "
                f"{s['initialMean']:.4f} | {s['tailMean']:.4f} | "
                f"{s['tailRadius2']:.4f} | {s['upwardObservations']}/150 |")
text = """# Landscape review: exploratory lab runs

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
""" + "\n".join(rows) + """

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
"""
(root / "report.md").write_text(text)
print("\n".join(rows))
