"""Compare dependent time windows of native EH shape and geometry observables."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUANTILES = [0.1, 0.5, 0.9]
COLORS = ["#245A81", "#D17A22", "#44805A"]
rng = np.random.default_rng(104729)
DIRECTIONS = rng.normal(size=(48, 3))
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1)[:, None]
DIRECTIONS = np.concatenate([np.eye(3), DIRECTIONS])


def positions(snapshots, normalize=False):
    arrays = []
    for snapshot in snapshots:
        x = np.array(snapshot["centered_positions"]).reshape(-1, 3)
        if normalize:
            x = x / np.sqrt(np.mean(np.sum(x * x, axis=1)))
        arrays.append(x)
    return np.concatenate(arrays)


def sliced_w2(left, right):
    if left.shape != right.shape:
        raise ValueError("comparison requires equal numbers of snapshot rows")
    difference = np.sort(left @ DIRECTIONS.T, axis=0)
    difference -= np.sort(right @ DIRECTIONS.T, axis=0)
    return float(np.sqrt(np.mean(difference**2)))


def residual_bound(report):
    """EH.D10, unioned over 21 radii and 20 lambda values before selection."""
    n = report["config"]["walkers"]
    q = 1 / (n - 1) if n % 2 == 0 else 1 / n
    r = 1 / ((n - 1) * (n - 3)) if n % 2 == 0 else 1 / (n * (n - 2))
    cycles = report["cycles"]
    budget = sum(c["residual_variance_bound"] for c in cycles)
    # Each cycle contains one open gate. Its cloning variance proxy is at
    # least 4(q+r) W_gate^2, and all other budget summands are nonnegative.
    gate_upper = max(np.sqrt(c["residual_variance_bound"] / (4 * (q + r))) for c in cycles)
    candidates = []
    tau2 = 1e-6 * 0.33 * (1 - np.exp(-0.004))
    for radius in 2. ** np.arange(21):
        if radius < gate_upper:
            continue
        scale = max(2 * radius / 3, 2 * tau2 / n)
        for index in range(1, 21):
            parameter = 2. ** -index / scale
            bound = np.log(2 * 420 / 0.01) / parameter
            bound += parameter * budget / (2 * (1 - scale * parameter))
            candidates.append((bound, radius, parameter))
    if not candidates:
        raise ValueError("fixed radius grid does not cover this trajectory")
    bound, radius, parameter = min(candidates)
    residual = sum(c["cloning_residual"] + c["kinetic_residual"] for c in cycles)
    return {
        "scope": (
            "Per-run 99% EH.D10 comparison in the ideal-innovation model; grid union included."
        ),
        "gate_spread_upper_from_recorded_budget": float(gate_upper),
        "selected_radius": float(radius), "selected_lambda": float(parameter),
        "predictable_variance_budget": budget, "accumulated_residual": residual,
        "two_sided_bound": float(bound), "within_bound": bool(abs(residual) <= bound),
    }


def describe_run(report):
    cycles = report["cycles"]
    snapshots = report["snapshots"]
    steps = report["steps"]
    blocks, groups = [], []
    for block in range(4):
        lo, hi = steps * block / 4, steps * (block + 1) / 4
        group = [s for s in snapshots if lo < s["step"] <= hi]
        groups.append(group)
        cycle = [c for c in cycles if lo < c["step"] <= hi]
        row = {
            "time_interval": [lo * 0.002, hi * 0.002],
            "mean_spread": float(np.mean([c["final_spread"] for c in cycle])),
            "last_spread": cycle[-1]["final_spread"],
            "mean_energy": float(np.mean([c["final_energy"] for c in cycle])),
        }
        for name in [
            "cloning_drift", "transport_drift", "ballistic_drift", "thermal_drift",
            "cloning_residual", "kinetic_residual", "residual_variance_bound",
        ]:
            row["mean_cycle_" + name] = float(np.mean([c[name] for c in cycle]))
        row["mean_cycle_total_drift"] = sum(
            row["mean_cycle_" + name]
            for name in ["cloning_drift", "transport_drift", "ballistic_drift", "thermal_drift"]
        )
        x = positions(group)
        row["axis_second_moments"] = np.mean(x * x, axis=0).tolist()
        row["radial_quantiles"] = np.quantile(np.linalg.norm(x, axis=1), QUANTILES).tolist()
        for name in ["final_curvature", "final_volume", "final_rewards"]:
            values = np.concatenate([s[name] for s in group])
            row[name + "_quantiles"] = np.quantile(values, QUANTILES).tolist()
            row[name + "_mean"] = float(np.mean(values))
        blocks.append(row)
    differences = []
    for left, right in zip(groups[:-1], groups[1:]):
        curve_left = np.sort(np.concatenate([s["final_curvature"] for s in left]))
        curve_right = np.sort(np.concatenate([s["final_curvature"] for s in right]))
        differences.append({
            "centered_position_sliced_w2": sliced_w2(positions(left), positions(right)),
            "rms_normalized_position_sliced_w2": sliced_w2(
                positions(left, True), positions(right, True)
            ),
            "curvature_w1": float(np.mean(np.abs(curve_left - curve_right))),
        })
    return {
        "N": report["config"]["walkers"], "seed": report["config"]["gas"]["seed"],
        "blocks": blocks, "adjacent_window_distances": differences,
        "martingale_check": residual_bound(report),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.directory.glob("n*-s*.json"))
    reports = [json.loads(p.read_text()) for p in paths]
    if not reports:
        raise ValueError("no native reports")
    steps = reports[0]["steps"]
    if any(r["steps"] != steps or r["stride"] * 100 != steps for r in reports):
        raise ValueError("use one common horizon and 100 equally spaced snapshots per run")
    horizon = steps * 0.002
    metrics = {
        "scope": "Descriptive time-window comparisons; dependent snapshots are not iid samples.",
        "quantiles": QUANTILES, "sliced_directions": len(DIRECTIONS),
        "trajectory_steps": sum(r["steps"] for r in reports),
        "exact_failures": sum(r["exact_failures"] for r in reports),
        "max_relative_identity_residual": max(
            r["max_relative_identity_residual"] for r in reports
        ),
        "runs": [describe_run(r) for r in reports],
        "files": [
            {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths
        ],
    }
    fig, axs = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    populations = sorted({r["config"]["walkers"] for r in reports})
    for n, color in zip(populations, COLORS):
        runs = [r for r in reports if r["config"]["walkers"] == n]
        time = np.array([c["step"] for c in runs[0]["cycles"]]) * 0.002
        for ax, key in [(axs[0, 0], "final_spread"), (axs[0, 1], "final_energy")]:
            values = np.array([[c[key] for c in r["cycles"]] for r in runs])
            # Nonoverlapping time blocks; each seed stays a separate run.
            width = max(1, len(time) // 100)
            count = len(time) // width
            tv = time[:count * width].reshape(count, width).mean(axis=1)
            values = values[:, :count * width].reshape(len(runs), count, width).mean(axis=2)
            ax.plot(tv, values.mean(axis=0), color=color, label=f"N={n}")
            ax.fill_between(tv, values.min(axis=0), values.max(axis=0), color=color, alpha=0.13)
        rows = [r for r in metrics["runs"] if r["N"] == n]
        centers = [sum(b["time_interval"]) / 2 for b in rows[0]["blocks"]]
        for ax, key in [(axs[1, 0], "final_curvature"), (axs[1, 1], "final_volume")]:
            values = np.mean([[b[key + "_quantiles"] for b in r["blocks"]] for r in rows], axis=0)
            for index, style in [(0, "--"), (1, "-"), (2, "--")]:
                ax.plot(centers, values[:, index], style, marker="o", markersize=3, color=color)
        distance = np.mean([
            [d["centered_position_sliced_w2"] for d in r["adjacent_window_distances"]]
            for r in rows
        ], axis=0)
        axs[2, 0].plot(
            np.array([0.25, 0.5, 0.75]) * horizon,
            distance, marker="o", color=color, label=f"N={n}",
        )
        drift = np.mean([[b["mean_cycle_total_drift"] for b in r["blocks"]] for r in rows], axis=0)
        axs[2, 1].plot(centers, drift, marker="o", color=color, label=f"N={n}")
    axs[0, 0].set(title="Unscaled centered position spread", ylabel="W(x)")
    axs[0, 1].set(title="Mean squared speed", ylabel="E(v)")
    axs[1, 0].set(title="Curvature distribution: 10%, 50%, 90%", ylabel="Native scalar curvature")
    axs[1, 1].set(
        title="Volume distribution: 10%, 50%, 90%",
        ylabel="Native volume element", yscale="log",
    )
    axs[2, 0].set(
        title=f"Centered shape: adjacent {horizon / 4:g}-time-unit windows",
        ylabel="51-direction sliced W2", xlabel="Boundary between windows",
    )
    axs[2, 1].set(
        title="Signed drift, including the full cloning cycle",
        ylabel="Mean signed drift budget per cycle",
    )
    axs[2, 1].axhline(0, color="#555555", linewidth=0.8)
    for ax in axs.flat:
        ax.grid(alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
        if ax is not axs[2, 0]:
            ax.set_xlabel("Algorithm time")
    axs[0, 0].legend(frameon=False)
    title = f"Native Einstein–Hilbert: shape, geometry and signed drift through time {horizon:g}"
    fig.suptitle(title, fontsize=14)
    args.output.mkdir(parents=True, exist_ok=True)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=160)
    plt.close(fig)
    (args.output / "stationarity-metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = [
        "# Native Einstein–Hilbert extended simulation", "",
        "Reference preset: h=0.002, T=0.33, viscosity=3, d=3, cloning every 20 steps, "
        f"f64, coincident start at rest. Each run executes {steps} steps (time {horizon:g}). "
        "Snapshot windows contain dependent observations; no iid error bars or stationarity "
        "p-values are assigned to walkers or time samples.", "",
        f"Checked {metrics['trajectory_steps']} steps; "
        f"{metrics['exact_failures']} exact failures; maximum normalized identity residual "
        f"{metrics['max_relative_identity_residual']:.4g}.", "",
        f"| N | Seed | Mean W, time {horizon / 2:g}–{horizon * 0.75:g} "
        f"| Mean W, time {horizon * 0.75:g}–{horizon:g} | Late curvature 10/50/90% |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in metrics["runs"]:
        a, b = row["blocks"][-2:]
        quantiles = " / ".join(f"{v:.3f}" for v in b["final_curvature_quantiles"])
        lines.append(
            f"| {row['N']} | {row['seed']} | {a['mean_spread']:.4f} "
            f"| {b['mean_spread']:.4f} | {quantiles} |"
        )
    lines += ["", "The JSON records every window's signed cloning, transport, ballistic and "
              "thermal contributions, both martingale residuals, curvature/volume/reward "
              "quantiles, and distances between centered empirical distributions. RMS-normalized "
              "shape distances are a separate diagnostic; they do not establish convergence "
              "of the unscaled centered measure. The martingale check uses EH.D10 with a "
              "fixed grid of 420 radius/lambda pairs, a union bound over that grid, "
              "and a 1% failure probability per run in the ideal-innovation interpretation.", ""]
    (args.output / "stationarity.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
