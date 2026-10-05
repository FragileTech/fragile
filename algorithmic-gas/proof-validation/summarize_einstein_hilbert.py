"""Summarize native EH operator reports; seeds, not walkers, are replicates."""

import argparse
import json
import math
from pathlib import Path
import statistics

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def clone_ratio(estimate):
    cloning = estimate["cloning"]
    return cloning["expected_variance"] / cloning["entering_variance"]


def write_summary(metrics, steps, output):
    lines = [
        "# Einstein–Hilbert operator validation",
        "",
        "Reference h=0.002, T=0.33, viscosity=3, dimension=3, clone period=20, f64. "
        "All trajectories start coincident at rest. Seed means and minimum/maximum "
        "bands use three complete independent runs; the bands are not confidence intervals.",
        "",
        f"Checked {metrics['trajectory_steps']} native trajectory steps and "
        f"{metrics['replicas']} checkpoint replicas, independent within each ensemble. "
        f"Exact check failures: {metrics['exact_failures']}. Maximum normalized "
        f"algebraic residual: {metrics['max_relative_identity_residual']:.4g}.",
        "",
        "| N | Final W | Final mean squared speed | Late expected cloning ratio "
        "| Negative late gates | Maximum kappa |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["late_runs"]:
        lines.append(
            f"| {row['N']} | {row['final_W_mean']:.6f} | {row['final_energy_mean']:.6f} "
            f"| {row['late_clone_ratio_mean']:.6f} | {row['late_negative_clone_drifts']}"
            f"/{row['late_gates']} | {row['max_kappa']:.6f} |"
        )
    lines += [
        "",
        f"Late means steps {steps // 2 + 1}–{steps} in the longer report, with a gate "
        "every 20 steps. These sampled states do not certify a global contraction "
        "hypothesis. The sufficient unweighted velocity margin requires kappa < "
        "exp(0.001) = 1.0010005; the maxima in this table exceed it. This failure "
        "of a sufficient bound is not evidence that the actual velocity energy grows.",
        "",
        "| N | Gate | Independent replicas | Cloning mean / SE | Kinetic mean / SE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in metrics["replica_checks"]:
        lines.append(
            f"| {row['N']} | {row['step']} | {row['samples']} "
            f"| {row['cloning_z']:.3f} | {row['kinetic_z']:.3f} |"
        )
    lines += [
        "",
        "Replica errors subtract each replica's own exact conditional prediction, "
        "including its sampled diversity fitness and realized post-cloning geometry. "
        "Standard errors use independent completed replicas, never interacting walkers. "
        "The error bars in the figure are one estimated standard error. Simulations "
        f"support the identities; the continued spread growth through time {steps * 0.002:g} "
        "does not establish either stationary convergence or nonconvergence.",
        "",
    ]
    (output / "results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    longest = max(reports, key=lambda report: report["configuration"]["steps"])
    steps = longest["configuration"]["steps"]
    all_runs = [run for report in reports for run in report["runs"]]
    checks = [check for report in reports for check in report["replicas"]]
    all_estimates = [estimate for run in all_runs for estimate in run["estimates"]]
    metrics = {
        "native_constants": {
            "quarter_kick_coefficient": 0.0015,
            "ou_factor": math.exp(-0.002),
            "position_transport_coefficient": 0.001 * (1 + math.exp(-0.002)),
            "ou_variance": 0.33 * (1 - math.exp(-0.002) ** 2),
            "position_noise_variance": 1e-6 * 0.33 * (1 - math.exp(-0.002) ** 2),
            "strict_velocity_kappa_threshold": math.exp(0.001),
        },
        "reports": [str(path) for path in args.reports],
        "trajectory_steps": len(all_estimates),
        "replicas": sum(check["samples"] for check in checks),
        "exact_failures": sum(report["exact_failures"] for report in reports),
        "max_relative_identity_residual": max(
            estimate["exact_max_relative_residual"] for estimate in all_estimates
        ),
        "late_runs": [],
        "replica_checks": [],
    }
    colors = ["#245A81", "#D17A22", "#44805A", "#986599"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    for color, n in zip(colors, longest["configuration"]["walkers"]):
        runs = [run for run in longest["runs"] if run["config"]["walkers"] == n]
        time = np.array([estimate["step"] for estimate in runs[0]["estimates"]]) * 0.002
        observables = [(axs[0, 0], "final_position_variance")]
        observables.append((axs[0, 1], "final_velocity_energy"))
        for ax, key in observables:
            values = np.array([[estimate[key] for estimate in run["estimates"]] for run in runs])
            ax.plot(time, values.mean(axis=0), color=color, label=f"N={n}", linewidth=1.4)
            ax.fill_between(time, values.min(axis=0), values.max(axis=0), color=color, alpha=0.12)
        gates = [
            [estimate for estimate in run["estimates"] if estimate["step"] % 20 == 0]
            for run in runs
        ]
        relative = np.array([[clone_ratio(estimate) - 1 for estimate in gate] for gate in gates])
        axs[1, 0].plot(
            np.array([estimate["step"] for estimate in gates[0]]) * 0.002,
            relative.mean(axis=0), color=color, label=f"N={n}", linewidth=1.0,
        )
        late = [estimate for gate in gates for estimate in gate if estimate["step"] > steps / 2]
        metrics["late_runs"].append({
            "N": n,
            "final_W_mean": statistics.mean(
                run["estimates"][-1]["final_position_variance"] for run in runs
            ),
            "final_energy_mean": statistics.mean(
                run["estimates"][-1]["final_velocity_energy"] for run in runs
            ),
            "late_clone_ratio_mean": statistics.mean(clone_ratio(estimate) for estimate in late),
            "late_negative_clone_drifts": sum(clone_ratio(estimate) < 1 for estimate in late),
            "late_gates": len(late),
            "max_kappa": max(estimate["kappa"] for run in runs for estimate in run["estimates"]),
        })
    labels = []
    for index, check in enumerate(checks):
        labels.append(f'N={check["config"]["walkers"]}\nstep {check["warmup"] + 1}')
        series = [
            (-0.13, "clone_standardized_residual", "#245A81", "Cloning"),
            (0.13, "kinetic_standardized_residual", "#D17A22", "Kinetics"),
        ]
        for shift, key, color, name in series:
            axs[1, 1].errorbar(
                index + shift, check[key], yerr=1.0, fmt="o", color=color,
                capsize=3, label=name if index == 0 else None,
            )
        metrics["replica_checks"].append({
            "N": check["config"]["walkers"],
            "step": check["warmup"] + 1,
            "samples": check["samples"],
            "cloning_z": check["clone_standardized_residual"],
            "kinetic_z": check["kinetic_standardized_residual"],
        })
    axs[0, 0].set(title="Centered position variance", xlabel="Algorithm time", ylabel="W(x)")
    axs[0, 1].set(title="Mean squared speed", xlabel="Algorithm time", ylabel="E(v)")
    axs[1, 0].set(
        title="Exact conditional cloning drift / entering W",
        xlabel="Algorithm time (open gates)", ylabel="Expected relative increment",
    )
    axs[1, 0].axhline(0, color="#555555", linewidth=0.8)
    axs[1, 1].set(
        title="Independent checkpoint replica residuals",
        ylabel="Mean residual / estimated standard error",
        xticks=range(len(labels)), xticklabels=labels,
    )
    axs[1, 1].axhline(0, color="#555555", linewidth=0.8)
    for ax in axs.flat:
        ax.grid(alpha=0.17)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0, 0].legend(frameon=False)
    axs[1, 1].legend(frameon=False)
    title = "Native Einstein–Hilbert: exact operator estimates and Rust simulations"
    fig.suptitle(title, fontsize=14)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=170)
    plt.close(fig)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_summary(metrics, steps, args.output)


if __name__ == "__main__":
    main()
