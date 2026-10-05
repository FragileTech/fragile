"""Reanalyze immutable native regional observations without running the engine."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(dataset, output):
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((dataset / "report.json").read_text())
    index = json.loads((dataset / "archive-index.json").read_text())
    entries = {entry["path"]: entry for entry in index["entries"]}
    cases = []
    hashes = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    timestep = report["harmonic_reference"]["h"]
    if not np.isfinite(timestep) or timestep <= 0:
        message = "A positive actual native timestep is required"
        raise ValueError(message)
    for case in report["cases"]:
        path = dataset / case["observations"]
        checksum = digest(path)
        if checksum != entries[case["observations"]]["sha256"]:
            raise ValueError(f"Changed observations: {path}")
        hashes[case["observations"]] = checksum
        with gzip.open(path, "rt") as stream:
            rows = json.load(stream)["rows"]
        samples, steps = case["samples"], case["steps"]
        optimal = np.empty((samples, steps + 1))
        coupling = np.empty_like(optimal)
        moments = np.empty((samples, steps + 1, 2))
        occupancy = np.empty((samples, steps + 1, 4))
        clones = 0
        refined = []
        for row in rows:
            rep, step = row["replicate"], row["step"]
            optimal[rep, step] = row["optimal_error"]
            coupling[rep, step] = row["coupling_error"]
            moments[rep, step] = [row["left"]["moment4"], row["left"]["moment8"]]
            occupancy[rep, step] = row["left"]["fractions"]
            if step:
                clones += row.get("accepted_clones", 0)
                if "refined_force_budget_residual" in row:
                    refined.append(row["refined_force_budget_residual"])
        mean = optimal.mean(axis=0)
        se = optimal.std(axis=0, ddof=1) / np.sqrt(samples)
        ratio = mean[-1] / mean[0]
        signed_rate = -np.log(ratio) / (timestep * steps) if ratio > 0 else None
        derived = {
            "d": case["d"],
            "N": case["N"],
            "initial_zone": case["initial_zone"],
            "selected_cloning": case.get("selected_cloning", False),
            "initial_optimal_error": float(mean[0]),
            "final_optimal_error": float(mean[-1]),
            "final_seed_standard_error": float(se[-1]),
            "final_initial_ratio": float(ratio),
            "signed_endpoint_decay_rate": float(signed_rate) if signed_rate is not None else None,
            "observed_error_decreases": bool(ratio < 1),
            "accepted_clone_count": clones,
            "maximum_refined_kinetic_residual": max(refined) if refined else None,
            "terminal_region_fractions": occupancy[:, -1].mean(axis=0).tolist(),
            "terminal_empirical_moments4_8": moments[:, -1].mean(axis=0).tolist(),
            "scope": "Optimal within-pair swarm transport under a shared-noise representative; signed endpoint rate is diagnostic, not a whole-law mixing certificate.",
        }
        cases.append(derived)
        panel = {"well": 0, "slow_zone": 1, "tail": 2}[case["initial_zone"]]
        axis = axes[panel]
        axis.plot(
            np.arange(steps + 1) * timestep, mean / mean[0], label=f"d={case['d']}, N={case['N']}"
        )
        axis.fill_between(
            np.arange(steps + 1) * timestep,
            np.maximum((mean - 2 * se) / mean[0], float(np.min(mean / mean[0])) / 10),
            (mean + 2 * se) / mean[0],
            alpha=0.12,
        )
    for axis, title in zip(axes, ["Well start", "Slow-zone start", "Tail start"]):
        axis.set(
            title=title,
            xlabel="Physical time",
            ylabel="Mean optimal swarm error / initial error",
            yscale="log",
        )
        axis.axhline(1, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.suptitle(
        "Native Rastrigin error — "
        + ("active cloning" if cases[0]["selected_cloning"] else "isolated kinetics")
    )
    fig.savefig(output / "regional-error.png", dpi=180)
    plt.close(fig)
    result = {
        "native_summary": report["summary"],
        "cases": cases,
        "local_well_rates": report["local_well_rates"],
        "source_hashes": hashes,
        "report_sha256": digest(dataset / "report.json"),
        "script_sha256": digest(Path(__file__)),
        "new_native_steps": 0,
        "plot_uncertainty": "Two seed standard errors; lower shading clipped to one tenth of the smallest plotted mean in its case for the logarithmic axis. Saved standard errors remain unchanged.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Native structural-landscape measurements",
        "",
        "Independent seeds determine uncertainty. These within-pair transport measurements test kinetic bounds; whole-law mixing requires the separate structural hypotheses.",
        "",
        "| d | N | Initial zone | Final / initial error | Signed endpoint rate | Error decreases | Accepted clones |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for c in cases:
        lines.append(
            f"| {c['d']} | {c['N']} | {c['initial_zone']} | {c['final_initial_ratio']:.6g} | {c['signed_endpoint_decay_rate']:.6g} | {'yes' if c['observed_error_decreases'] else 'no'} | {c['accepted_clone_count']} |"
        )
    lines.extend([
        "",
        "Negative signed rates mean this coupling error increased. They are retained, never reclassified as convergence.",
        "",
        "![Regional errors](regional-error.png)",
        "",
    ])
    (output / "results.md").write_text("\n".join(lines))
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.dataset, args.output)
