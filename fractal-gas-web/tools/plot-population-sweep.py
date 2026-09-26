"""Plot median and interquartile error versus initial swarm population."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Gaussian": "#8795a1", "Local covariance": "#247c96",
          "Bounded adaptive": "#bb702e", "Bounded + basin restarts": "#8d5ba6"}
PROBLEMS = {"quadratic": "Quadratic bowl", "bbob_10": "Rotated ellipsoid",
            "rastrigin": "Rastrigin", "bbob_15": "Rotated Rastrigin",
            "rosenbrock": "Rosenbrock", "bbob_5": "Boundary optimum"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/optimization/reports/population-sweep-20d-100k")
    args = parser.parse_args()
    output = ROOT / args.output
    manifest = json.loads((output / "manifest.json").read_text())
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    reference = [json.loads(line) for line in (output / "cma-reference.jsonl").read_text().splitlines()]
    sizes = manifest["populations"]
    floor = 1e-10
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8), layout="constrained")
    fig.patch.set_facecolor("#fafaf8")
    fig.suptitle(f"Population sweep · 20 dimensions · five elites\n"
                 f"{manifest['budget']:,}-evaluation allowance · {manifest['seeds']} seeds per point",
                 fontsize=18, ha="left", x=.045)
    for ax, (problem, title) in zip(axes.flat, PROBLEMS.items()):
        ax.set_facecolor("#fafaf8")
        maximum = floor
        for variant, color in COLORS.items():
            medians, lower, upper, errors = [], [], [], []
            for n in sizes:
                group = [r for r in rows if r["problem"] == problem and r["variant"] == variant and int(r["initial_walkers"]) == n]
                values = [float(r["regret"]) for r in group]
                q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
                medians.append(max(floor, statistics.median(values)))
                lower.append(max(floor, q1))
                upper.append(max(floor, q3))
                errors.append(any(r["error"] for r in group))
                maximum = max(maximum, q3)
            ax.plot(sizes, medians, color=color, marker="o", linewidth=1.9, markersize=4)
            ax.fill_between(sizes, lower, upper, color=color, alpha=.13, linewidth=0)
            for n, y, failed in zip(sizes, medians, errors):
                if failed:
                    ax.scatter([n], [y], color="#bc3737", marker="x", s=70, linewidths=1.8, zorder=5)
        cma = max(floor, statistics.median(r["regret"] for r in reference if r["problem"] == problem))
        ax.axhline(cma, color="#32835e", linestyle="--", linewidth=1.7)
        maximum = max(maximum, cma)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sizes, [f"{n:,}" for n in sizes], fontsize=9)
        ax.set_ylim(cma / 3, 10 ** (math.ceil(math.log10(maximum)) + .5))
        ax.set_title(title, loc="left", pad=10)
        ax.set_xlabel("Starting walkers")
        ax.set_ylabel("Best objective − optimum")
        ax.grid(axis="y", alpha=.22)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([0], [0], color=color, linewidth=2, label=name) for name, color in COLORS.items()]
    handles += [Line2D([0], [0], color="#32835e", linestyle="--", label="CMA-ES reference"),
                Line2D([0], [0], marker="x", color="#bc3737", linestyle="none", label="Group includes errors")]
    fig.legend(handles=handles if any(r["error"] for r in rows) else handles[:-1], loc="outside lower center", ncol=3, frameon=False, fontsize=10,
               title="Lines: medians · shading: interquartile range · errors ≤ 1e-10 shown at 1e-10")
    fig.savefig(output / "population-sweep.png", dpi=170)
    fig.savefig(output / "population-sweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
