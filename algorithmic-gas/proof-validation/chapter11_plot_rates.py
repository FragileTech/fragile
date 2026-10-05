"""Plot measured Chapter11 physical-time transport and survival calibration."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path)
    parser.add_argument("--pooled", type=Path)
    args = parser.parse_args()
    data = json.loads((args.ledger / "native-curves.json").read_text())
    curves = [
        c
        for c in data["curves"]
        if c["left_group"].startswith("native-trajectories-v1/") and c["d"] == 2 and c["N"] == 32
    ]
    if len(curves) != 6:
        message = "Expected all six fresh native profiles at d2,N32"
        raise ValueError(message)
    pooled = {}
    if args.pooled:
        pooled = {
            c["left_group"]: c
            for c in json.loads((args.pooled / "report.json").read_text())["curves"]
        }
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, curve in zip(axes.flat, curves, strict=True):
        time = np.array(curve["physical_times"])
        for name, label, color in [
            ("grid_shape_H_squared", "Grid shape H²", "#355f94"),
            ("paired_empirical_projected_W2_squared", "Exact projected empirical W₂²", "#bf642f"),
        ]:
            metric = next(m for m in curve["metrics"] if m["metric"] == name)
            values = np.array(metric["values"])
            interval = np.array(metric["pointwise_bootstrap_95_intervals"])
            axis.plot(time, values, label=label, color=color, linewidth=1.8)
            axis.fill_between(
                time,
                np.maximum(interval[:, 0], 1e-9),
                np.maximum(interval[:, 1], 1e-9),
                color=color,
                alpha=0.18,
            )
        if curve["left_group"] in pooled:
            exact = pooled[curve["left_group"]]
            values = np.array([
                p["exact_normalized_projected_W2_squared"] for p in exact["point_curve"]
            ])
            interval = np.array(exact["pointwise_whole_trajectory_bootstrap_95_intervals"])
            axis.plot(
                time, values, label="Exact pooled alive-law W₂²", color="#298068", linewidth=1.8
            )
            axis.fill_between(
                time,
                np.maximum(interval[:, 0], 1e-9),
                np.maximum(interval[:, 1], 1e-9),
                color="#298068",
                alpha=0.18,
            )
        axis.set_yscale("log")
        axis.set_ylim(bottom=1e-5)
        axis.set_xlabel("Physical time")
        axis.set_ylabel("Probability-normalized error")
        axis.set_title(curve["left_group"].split("/")[1].removesuffix("-d2-N32"))
        axis.grid(alpha=0.2, which="both")
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(
        "Actual native initial-law contrasts: d = 2, N = 32\nWhole-trajectory bootstrap intervals; finite population floors retained",
        fontsize=14,
    )
    figure.savefig(args.ledger / "native-shape-transport-rates.png", dpi=180)
    figure.savefig(args.ledger / "native-shape-transport-rates.pdf")
    plt.close(figure)
    calibration = json.loads((args.ledger / "native-survival-calibration.json").read_text())[
        "rows"
    ]
    selected = [
        r
        for r in calibration
        if r["group"].startswith("native-trajectories-v1/")
        and r["predicted_mean_quadratic_variation"] > 0
    ]
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for condition, color in [
        ("whole_prepared_kinetics", "#355f94"),
        ("post_B2_final_Gaussian", "#bf642f"),
    ]:
        rows = [r for r in selected if r["conditioning"] == condition]
        x = np.array([r["predicted_mean_quadratic_variation"] for r in rows])
        y = np.array([r["actual_mean_squared_innovation_sum"] for r in rows])
        axis.scatter(x, y, c=color, label=condition.replace("_", " "), alpha=0.8, s=28)
    maximum = max(
        max(r["predicted_mean_quadratic_variation"], r["actual_mean_squared_innovation_sum"])
        for r in selected
    )
    minimum = min(
        min(r["predicted_mean_quadratic_variation"], r["actual_mean_squared_innovation_sum"])
        for r in selected
    )
    axis.plot(
        [minimum / 1.3, maximum * 1.3],
        [minimum / 1.3, maximum * 1.3],
        "--",
        color="#636363",
        label="Exact martingale prediction",
    )
    axis.set(
        xscale="log",
        yscale="log",
        xlabel="Mean predicted quadratic variation",
        ylabel="Mean observed squared innovation sum",
        title="Native survival variance calibration\nIndependent complete trajectories; both conditioning stages separate",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(args.ledger / "native-survival-calibration.png", dpi=180)
    figure.savefig(args.ledger / "native-survival-calibration.pdf")
    plt.close(figure)
    with gzip.open(args.ledger / "plot-source.py.gz", "wt") as file:
        file.write(Path(__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
