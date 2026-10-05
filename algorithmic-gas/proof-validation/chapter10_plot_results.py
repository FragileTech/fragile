"""Plot retained, law-specific Chapter 10 rates and normalized population errors."""

import argparse
import gzip
import json
import operator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read(root, path):
    return json.loads(gzip.decompress((root / path).read_bytes()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("ledger", type=Path)
    args = parser.parse_args()
    reference = json.loads((args.reference / "report.json").read_text())
    ledger = json.loads((args.ledger / "report.json").read_text())
    if reference["comparisons_failed"] or ledger["comparisons_failed"]:
        msg = "Only passing retained datasets may be plotted"
        raise ValueError(msg)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for kappa in [0.25, 1, 4]:
        rows = [
            c
            for c in reference["cases"]
            if c["id"].startswith(f"gaussian-k{kappa}-temperature1-") and c["id"].endswith("tail")
        ]
        rows.sort(key=lambda c: c["constants"]["friction"])
        friction = [c["constants"]["friction"] for c in rows]
        ax.loglog(
            friction,
            [c["constants"]["rate"] for c in rows],
            "o--",
            label=f"Predicted sufficient rate, κ={kappa}",
        )
        ax.loglog(
            friction,
            [c["observed_endpoint_H_rate"] for c in rows],
            "s-",
            label=f"Measured endpoint H rate, κ={kappa}",
        )
    ax.set(
        title="Conservative kinetic Gaussian density: θ=1",
        xlabel="Friction γ",
        ylabel="Rate per unit time",
    )
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    rows = [
        c
        for c in reference["cases"]
        if c["id"].startswith("nonquadratic-")
        and c["parameters"]["theta"] == 1
        and c["parameters"]["gamma"] == 2
    ]
    for zone in ["well", "saddle", "tail"]:
        selected = sorted(
            [c for c in rows if c["id"].endswith(zone)], key=lambda c: c["parameters"]["amplitude"]
        )
        data = [read(args.reference, c["archive"]) for c in selected]
        ax.semilogy(
            [c["parameters"]["amplitude"] for c in selected],
            [-r["Phi_dot"] / r["Phi"] for r in data],
            "o-",
            label=zone,
        )
    selected = sorted(
        [c for c in rows if c["id"].endswith("well")], key=lambda c: c["parameters"]["amplitude"]
    )
    ax.semilogy(
        [c["parameters"]["amplitude"] for c in selected],
        [c["constants"]["rate"] for c in selected],
        "k--",
        label="Sufficient rate r",
    )
    ax.set(
        title="Nonquadratic Gibbs densities: instantaneous dissipation",
        xlabel="Cosine barrier amplitude A",
        ylabel="−Φ̇/Φ and predicted r",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for law in ledger["common_target_density_laws"]:
        data = read(args.ledger, law["archive"])
        times = [f["time"] for f in data["frames"]]
        phi = [f["fine"]["Phi"] if f["fine"]["Phi"] > 1e-12 else np.nan for f in data["frames"]]
        ax.semilogy(times, phi, label=f"ρ={law['rho']}, jump={law['jump_rate']}")
    ax.semilogy(
        times,
        [f["predicted_envelope"] for f in data["frames"]],
        "k--",
        label="Predicted Φ₀ exp(−rt)",
    )
    ax.axhline(1e-6, color="0.6", linestyle=":", label="Rate-fit resolution threshold")
    ax.set(
        title="Common-invariant Gaussian refresh: full density mixture",
        xlabel="Time",
        ylabel="Modified entropy Φ (per coordinate)",
    )
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for d in [1, 2, 4]:
        rows = sorted(
            [r for r in ledger["gaussian_variance"] if r["d"] == d], key=operator.itemgetter("N")
        )
        ns = np.asarray([r["N"] for r in rows])
        ax.loglog(ns, [r["variance"] for r in rows], "o-", label=f"d={d}")
    ax.loglog(ns, 1 / ns, "k--", label="Exact prediction C*L²/N = 1/N")
    ax.set(
        title="Independent Gaussian reference populations",
        xlabel="Population N",
        ylabel="Variance of mean position x₁",
    )
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.2, which="both")
    fig.suptitle(
        "Chapter 10: identified density laws, N-independent rates and normalized errors",
        fontsize=14,
    )
    fig.savefig(args.ledger / "rates.png", dpi=170)
    fig.savefig(args.ledger / "rates.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
