"""Retained continuous-reference and native-law entropy/transport figures."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("native_ledger", type=Path)
    parser.add_argument("empty_output", type=Path)
    args = parser.parse_args()
    args.empty_output.mkdir(parents=True, exist_ok=False)
    index_path = args.reference / "archive-index.json"
    index = json.loads(index_path.read_text())
    provenance = {
        "reference_index": str(index_path.resolve()),
        "reference_index_sha256": sha(index_path),
        "native_curves_sha256": sha(args.native_ledger / "native-curves.json"),
        "helper_sha256": sha(Path(__file__)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for axis, gamma in zip(axes, [0.25, 1, 4], strict=True):
        tag = f"gaussian-k1-temperature1-friction{gamma}-tail"
        entry = next(e for e in index["entries"] if e["tag"] == tag)
        path = args.reference / entry["path"]
        if sha(path) != entry["sha256"]:
            message = "Reference source SHA mismatch"
            raise ValueError(message)
        row = json.loads(gzip.decompress(path.read_bytes()))
        trajectory = row["states"]
        times = [s["time"] for s in trajectory]
        initial = trajectory[0]["Phi"]
        axis.semilogy(times, [s["Phi"] / initial for s in trajectory], label="Measured Φ / Φ₀")
        axis.semilogy(
            times,
            [s["predicted_envelope"] / initial for s in trajectory],
            linestyle="--",
            label="Proved upper bound",
        )
        axis.set(
            title=f"Gaussian reference, γ={gamma}",
            xlabel="Physical time",
            ylabel="Modified entropy",
        )
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for suffix in ["png", "pdf"]:
        fig.savefig(args.empty_output / f"reference-entropy.{suffix}", dpi=180)
    plt.close(fig)
    curves = json.loads((args.native_ledger / "native-curves.json").read_text())["curves"]
    profiles = [
        "quadratic-unbounded",
        "rastrigin-well",
        "rastrigin-saddle",
        "rastrigin-tail",
        "quadratic-boundary-revival",
        "rastrigin-boundary-revival",
    ]
    for metric_name, title in [
        ("grid_W2_squared", "Finite-resolution alive shape transport"),
        ("grid_H_squared", "Finite-resolution alive Hellinger error"),
    ]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for axis, profile in zip(axes.flat, profiles, strict=True):
            for n in [8, 32, 128]:
                candidates = [
                    c
                    for c in curves
                    if c["d"] == 2 and c["N"] == n and f"{profile}-d2-N{n}/left" in c["left_group"]
                ]
                if len(candidates) != 1:
                    message = f"Expected one native curve for {profile}, N={n}, d=2"
                    raise ValueError(message)
                curve = candidates[0]
                metric = next((m for m in curve["metrics"] if m["metric"] == metric_name), None)
                if metric is None:
                    continue
                line = axis.plot(curve["physical_times"], metric["values"], label=f"N={n}")[0]
                bounds = metric["pointwise_bootstrap_95_intervals"]
                if bounds is not None:
                    intervals = np.asarray(bounds)
                    axis.fill_between(
                        curve["physical_times"],
                        intervals[:, 0],
                        intervals[:, 1],
                        color=line.get_color(),
                        alpha=0.1,
                    )
            axis.set(
                title=profile.replace("-", " "),
                xlabel="Physical time",
                ylabel="Probability-normalized error",
            )
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle(title + "; d=2, independent native ensembles")
        for suffix in ["png", "pdf"]:
            fig.savefig(args.empty_output / f"native-{metric_name}.{suffix}", dpi=180)
        plt.close(fig)
    outputs = {p.name: sha(p) for p in args.empty_output.iterdir()}
    (args.empty_output / "provenance.json").write_text(
        json.dumps(
            {
                **provenance,
                "outputs": outputs,
                "scope": "Continuous kinetic reference density versus its proved envelope; independent native finite-grid alive-law comparisons with pointwise whole-trajectory bootstrap intervals. Native curves do not identify the full configuration-space entropy or a QSD.",
            },
            indent=2,
        )
        + "\n"
    )
    (args.empty_output / "helper.py").write_bytes(Path(__file__).read_bytes())


if __name__ == "__main__":
    main()
