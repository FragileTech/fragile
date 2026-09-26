"""Render per-seed long-budget results; requires matplotlib, not used by the runner."""

import argparse
import csv
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [
    "Gaussian", "Local covariance", "Bounded adaptive",
    "Bounded + basin restarts", "BIPOP-active CMA-ES",
]
LABELS = ["Gaussian", "Previous\ncovariance", "Bounded\nadaptive", "Bounded\n+ basins", "BIPOP\nCMA-ES"]
COLORS = ["#8795a1", "#247c96", "#bb702e", "#8d5ba6", "#32835e"]
PROBLEMS = {
    "quadratic": "Quadratic bowl",
    "bbob_10": "Rotated ellipsoid",
    "rastrigin": "Rastrigin",
    "bbob_15": "Rotated Rastrigin",
    "rosenbrock": "Rosenbrock",
    "bbob_5": "Boundary optimum",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/optimization/reports/long-budget-100k-20d-5elites")
    args = parser.parse_args()
    output = ROOT / args.output
    with (output / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    budgets = {int(row["budget"]) for row in rows}
    if len(budgets) != 1:
        raise ValueError("Plot requires one common evaluation allowance")
    budget = budgets.pop()
    seeds = len({int(row["seed"]) for row in rows})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold"})
    for dimension in sorted({int(r["dimensions"]) for r in rows}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.6), layout="constrained")
        fig.patch.set_facecolor("#fafaf8")
        fig.suptitle(f"{budget:,}-evaluation allowance · {dimension} dimensions\n"
                     f"Five fractal elites · {seeds} seeds · lower regret is better", fontsize=19, ha="left", x=.045)
        for ax, (problem, name) in zip(axes.flat, PROBLEMS.items()):
            ax.set_facecolor("#fafaf8")
            max_log = -8
            for i, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
                group = sorted((r for r in rows if r["problem"] == problem
                                and int(r["dimensions"]) == dimension and r["variant"] == variant),
                               key=lambda r: int(r["seed"]))
                values = [float(r["regret"]) for r in group]
                logs = [math.log10(max(1e-8, value)) for value in values]
                max_log = max(max_log, *logs)
                for j, (row, value) in enumerate(zip(group, logs)):
                    x = i + .32 * ((j / max(1, len(group) - 1)) - .5)
                    ax.scatter(x, value, color="#bc3737" if row["error"] else color,
                               marker="x" if row["error"] else "o", s=28, alpha=.8,
                               linewidths=1.5 if row["error"] else 0)
                median = math.log10(max(1e-8, statistics.median(values)))
                ax.plot([i - .24, i + .24], [median, median], color=color, linewidth=3)
            ax.set_title(name, loc="left", pad=12)
            ax.set_xticks(range(len(VARIANTS)), LABELS, fontsize=9)
            ax.set_xlim(-.55, 4.55)
            ax.set_ylim(-8.5, max(0.5, math.ceil(max_log) + .5))
            ticks = list(range(-8, max(1, math.ceil(max_log)) + 1, 2))
            ax.set_yticks(ticks, ["≤10⁻⁸" if tick == -8 else f"$10^{{{tick}}}$" for tick in ticks])
            ax.set_ylabel("Best objective − optimum")
            ax.grid(axis="y", color="#ddddda", linewidth=.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        legend = [Line2D([0], [0], color="#444444", linewidth=3, label="Median"),
                  Line2D([0], [0], marker="o", linestyle="none", color="#777777", label="Individual run"),
                  Line2D([0], [0], marker="x", linestyle="none", color="#bc3737", label="Stopped with error")]
        fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
        fig.savefig(output / f"results-{dimension}d.png", dpi=170)
        fig.savefig(output / f"results-{dimension}d.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
