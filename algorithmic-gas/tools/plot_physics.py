"""Render completed Algorithmic Gas research summaries and their numerical tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def configuration_label(config: dict) -> str:
    """Describe each controlled configuration using its actual parameter values."""
    prefix = f"{config['dimension']}D, N={config['walkers']}"
    fitness = config["fitness"]
    standardizer = fitness["reward_standardizer"]
    if standardizer["kind"] == "local":
        return f"{prefix}, width={standardizer['kernel']['width']:g}"
    metric = config["metric"]
    kinetic = config["kinetic"]
    changes = []
    for label, value, base in [
        ("epsilon", metric["epsilon"], 0.1),
        ("T", metric["temperature"], 1),
        ("gamma", kinetic["friction"], 1),
        ("h", kinetic["dt"], 0.02),
        ("sigma", standardizer["sigma_min"], 0.001),
    ]:
        if value != base:
            changes.append(f"{label}={value:g}")
    return f"{prefix}, {', '.join(changes) if changes else 'base'}"


def render(path: Path, output: Path) -> None:
    """Plot seed distributions and conditional thermostat residuals."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report["complete"]:
        raise ValueError(f"Research summary is incomplete: {path}")
    groups = sorted(
        report["groups"],
        key=lambda g: (
            g["configuration"]["dimension"],
            g["configuration"]["walkers"],
            configuration_label(g["configuration"]),
        ),
    )
    labels = [configuration_label(g["configuration"]) for g in groups]
    curves = [[r["curvature"] for r in g["runs"] if r["curvature"] is not None] for g in groups]
    residuals = [g["thermostat_residual_over_predicted_standard_deviation"] for g in groups]
    y = np.arange(len(groups)) + 1
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(
        1,
        2,
        figsize=(14, max(4.5, len(groups) * 0.5 + 2)),
        gridspec_kw={"width_ratios": [2.2, 1]},
        layout="constrained",
        sharey=True,
    )
    left.set_xscale("symlog", linthresh=1)
    left.boxplot(
        curves,
        positions=y,
        orientation="horizontal",
        widths=0.5,
        showfliers=False,
        patch_artist=True,
        boxprops={"facecolor": "#d4e4ef", "edgecolor": "#2d5676"},
        medianprops={"color": "#122f46", "linewidth": 1.7},
    )
    jitter = np.random.default_rng(7)
    for level, values in zip(y, curves, strict=True):
        left.scatter(
            values,
            level + jitter.uniform(-0.18, 0.18, len(values)),
            s=10,
            color="#2d5676",
            alpha=0.55,
            zorder=3,
        )
    left.set_yticks(y, labels)
    left.set_xlabel("Final population-mean scalar curvature (fitness units⁻¹)")
    left.set_title("Curvature across independent seeds", loc="left", fontweight="bold")
    left.grid(axis="x", alpha=0.2)
    right.axvspan(-1, 1, color="#e1eee9", label="±1 predicted SD")
    right.axvline(0, color="#6c8179", linewidth=0.8)
    right.scatter(residuals, y, s=35, color="#18624e", zorder=3)
    limit = max(2, max(abs(v) for v in residuals if v is not None) + 0.5)
    right.set_xlim(-limit, limit)
    right.set_xlabel("Residual sum / predicted SD")
    right.set_title("Thermostat energy", loc="left", fontweight="bold")
    right.grid(axis="y", alpha=0.15)
    right.legend(loc="lower right", frameon=False, fontsize=9)
    left.invert_yaxis()
    fig.suptitle(
        f"Algorithmic Gas · {report['completed_trajectories']} trajectories · "
        f"{report['trajectory_updates']:,} updates",
        fontsize=15,
        fontweight="bold",
    )
    output.mkdir(parents=True, exist_ok=True)
    name = path.parent.name
    for suffix in ["png", "pdf"]:
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    lines = [
        f"# {name}",
        "",
        (
            f"{report['completed_trajectories']} trajectories; "
            f"{report['trajectory_updates']:,} complete updates; "
            f"{report['walker_updates']:,} walker updates; "
            f"{report['conditional_replica_updates']:,} independent replica updates."
        ),
        "",
        (
            "Curvature is evaluated at each available O-stage query and then averaged over walkers. "
            "The table reports the median final average across seeds. Thermostat residuals are "
            "standardized within each configuration using the recorded conditional variances."
        ),
        "",
        "| Configuration | Seeds | Median scalar curvature | Thermostat residual / SD | Squared residual / variance |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, group in zip(labels, groups, strict=True):
        lines.append(
            f"| {label} | {group['trajectories']} | "
            f"{group['final_mean_scalar_curvature_median']:.6g} | "
            f"{group['thermostat_residual_over_predicted_standard_deviation']:.4f} | "
            f"{group['thermostat_squared_residual_over_variance']:.4f} |"
        )
    lines += [
        "",
        f"Maximum kinetic telescoping residual: {report['maximum_kinetic_telescoping_residual']:.6g}.",
        f"Unavailable curvature queries: {report['recorded_curvature_unavailable_queries']}.",
    ]
    closure = report["spatial_moment_closure"]
    if closure:
        fit = closure["fit"]
        lines += [
            "",
            "## Constitutive evaluation",
            "",
            closure["candidate"] + ".",
            "",
            "Training uses seeds 7–22; validation uses seeds 23–38 across every configuration.",
            "",
            "| Predictor | Held-out RMSE |",
            "|---|---:|",
            f"| Two-coefficient fit | {fit['validation_rmse']:.6g} |",
            f"| Constant baseline | {fit['constant_baseline_rmse']:.6g} |",
            f"| Shuffled training control | {closure['shuffled_training_control']['validation_rmse']:.6g} |",
        ]
    (output / f"{name}.md").write_text("\n".join(lines) + "\n")
    print(output / f"{name}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/physics-validation"))
    args = parser.parse_args()
    for summary in args.summaries:
        render(summary, args.output)


if __name__ == "__main__":
    main()
