"""Summarize fresh Rust experiments without replacing their retained raw data."""

import argparse
import gzip
import hashlib
import json
import operator
from pathlib import Path


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_raw(dataset, relative, index):
    entry = next(row for row in index["entries"] if row["path"] == relative)
    path = dataset / relative
    if digest(path) != entry["sha256"]:
        msg = f"Checksum mismatch: {path}"
        raise ValueError(msg)
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def summarize(datasets, destination, cubature_review=None):
    destination.mkdir(parents=True, exist_ok=True)
    chapters = []
    provenance = []
    for dataset in datasets:
        index = json.loads((dataset / "archive-index.json").read_text())
        if index["status"] not in {"completed", "discrepancy"}:
            msg = f"Incomplete dataset: {dataset}"
            raise ValueError(msg)
        report = json.loads((dataset / "report.json").read_text())
        native = [entry for entry in index["entries"] if entry["kind"] == "native_run_archive"]
        provenance.append({
            "dataset": str(dataset.resolve()),
            "status": index["status"],
            "config": report["config"],
            "index_sha256": digest(dataset / "archive-index.json"),
            "report_sha256": digest(dataset / "report.json"),
            "archives": len(index["entries"]),
            "native_recorded_steps": sum(e["metadata"]["recorded_steps"] for e in native),
            "compressed_bytes": sum(e["compressed_bytes"] for e in index["entries"]),
        })
        for chapter in report["chapters"]:
            chapter["dataset"] = str(dataset.resolve())
            chapter["index"] = index
            chapters.append(chapter)
    numbers = [chapter["chapter"] for chapter in chapters]
    if len(set(numbers)) != len(numbers):
        msg = "Supply one final dataset per chapter; repeated samples must not be counted twice"
        raise ValueError(msg)
    all_checks = []
    missing = []
    rows = []
    for chapter in sorted(chapters, key=operator.itemgetter("chapter")):
        number = chapter["chapter"]
        coverage = chapter["coverage"]
        checks = chapter["comparisons"]
        failed = sum(check.get("passed") is False for check in checks)
        rows.append({
            "chapter": number,
            "comparisons": len(checks),
            "failed": failed,
            "required_expressions": coverage["required_expressions"],
            "source_expressions_checked": coverage["checked_required_expressions"],
            "unbound_quotes": len(coverage["unbound_evidence"]),
            "all_required_checked": coverage["complete"],
        })
        all_checks.extend({"chapter": number, **check} for check in checks)
        missing.extend(
            {"chapter": number, **entry}
            for entry in coverage["expression_ledger"]
            if entry["status"] == "not_checked_from_stored_data"
        )
        if number == 5:
            chapter["endpoint_rates"] = kinetic_plots(chapter, destination)
        if number == 6:
            survivor_plots(chapter, destination)
    summary = {
        "chapters": rows,
        "datasets": provenance,
        "kinetic_endpoint_rates": [
            rate for chapter in chapters for rate in chapter.get("endpoint_rates", [])
        ],
    }
    cubature = None
    if cubature_review is not None:
        cubature = json.loads(cubature_review.read_text())
        source = Path(cubature["derived_review"]["source_report"])
        if digest(source) != cubature["derived_review"]["source_report_sha256"]:
            msg = "Cubature source report checksum mismatch"
            raise ValueError(msg)
        summary["exact_gaussian_cubature"] = {
            "review": str(cubature_review.resolve()),
            "review_sha256": digest(cubature_review),
            "levels": cubature["levels"],
            "refinement_ratios": cubature["refinement_ratios"],
            "analytic_weak_prefactor": cubature["analytic_weak_prefactor"],
            "comparisons": len(cubature["comparisons"]),
            "failed": sum(c["passed"] is False for c in cubature["comparisons"]),
        }
        all_checks.extend({"chapter": 5, **check} for check in cubature["comparisons"])
        cubature_plot(cubature, destination)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "comparisons.jsonl").write_text(
        "".join(json.dumps(check) + "\n" for check in all_checks)
    )
    (destination / "unvalidated-expressions.json").write_text(json.dumps(missing, indent=2) + "\n")
    (destination / "failures.json").write_text(
        json.dumps([c for c in all_checks if c.get("passed") is False], indent=2) + "\n"
    )
    lines = [
        "# Fresh native Rust experiments for Chapters 4–6",
        "",
        (
            "Raw native populations, proposal plans, kinetic stages, field evaluations and noise "
            "remain in immutable gzip CBOR chunks. Checkpoints retain resumable engine state. "
            "All comparisons use population-normalized observables; sampled expectations retain "
            "their independent-replicate uncertainty."
        ),
        "",
        "| Chapter | Numerical comparisons | Failed | Required expressions checked | Unbound quotes |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['chapter']} | {row['comparisons']:,} | {row['failed']:,} | "
            f"{row['source_expressions_checked']}/{row['required_expressions']} | "
            f"{row['unbound_quotes']} |"
        )
    lines.extend([
        "",
        (
            "Passing finite comparisons do not certify every analytic theorem. "
            "`unvalidated-expressions.json` retains every required expression without "
            "whole-expression numerical evidence; unsupported global density, eigenfunction "
            "and QSD rate constants remain explicit in the chapter reports."
        ),
        "",
    ])
    for chapter in chapters:
        number = chapter["chapter"]
        lines.extend([f"## Chapter {number}", "", chapter["title"], ""])
        for gap in chapter.get("gaps", []):
            lines.extend([
                (
                    f"- {gap.get('id', 'analytic obligation')}: "
                    f"{gap.get('reason', gap.get('missing', gap))}"
                )
            ])
        if number == 5:
            lines.extend([
                "",
                "| N | d | Replicates | Steps | Final Q error / initial | Endpoint decay rate | Certified lower rate |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for case in chapter["capped_cases"]:
                rate = next(
                    r
                    for r in chapter["endpoint_rates"]
                    if r["N"] == case["N"] and r["d"] == case["d"]
                )
                lines.append(
                    f"| {case['N']} | {case['d']} | {case['samples']} | "
                    f"{case['steps']} | {case['last_mean_ratio']:.8g} | "
                    f"{rate['measured_endpoint_rate']:.8g} | {rate['certified_lower_rate']:.8g} |"
                )
            lines.extend([
                "",
                "| Timestep | Measured weak bias | Predicted weak bias | Measured strong MSE | Predicted strong MSE |",
                "|---:|---:|---:|---:|---:|",
            ])
            for level in chapter["refinement"]["levels"]:
                lines.append(
                    f"| {level['h']:.4g} | {level['measured_weak_bias']:.8g} | "
                    f"{level['predicted_weak_bias']:.8g} | {level['strong_MSE']:.8g} | "
                    f"{level['exact_strong_MSE']:.8g} |"
                )
        if number == 6:
            lines.extend([
                "",
                "| N | d | Surviving-law cost: final / initial | Alive-marginal cost: final / initial |",
                "|---:|---:|---:|---:|",
            ])
            for case in chapter["cases"]:
                decline = {
                    entry["measure"]: entry["terminal_to_initial_ratio"]
                    for entry in case["empirical_law_decay"]
                    if entry["cemetery_convention"] == "native_zero_alive"
                }
                values = [
                    decline.get(name)
                    for name in ("whole_swarm_law_transport", "uniform_alive_marginal_transport")
                ]
                text = ["unavailable" if value is None else f"{value:.8g}" for value in values]
                lines.append(f"| {case['N']} | {case['d']} | {text[0]} | {text[1]} |")
            lines.extend([
                "",
                (
                    "Law costs use independent surviving ensembles with their own "
                    "denominators. For N>4, the whole-law cost is an admissible block "
                    "coupling upper bound; N≤4 uses exact empirical-law transport. "
                    "An unavailable final law means the ensemble had no survivors. "
                    "These ratios measure decline without identifying a QSD mixing rate."
                ),
            ])
        lines.append("")
    if cubature is not None:
        lines.extend([
            "## Exact Gaussian integration of native timestep errors",
            "",
            (
                "The uncapped unit-quadratic experiment uses 96 weighted Gaussian cubature nodes. "
                "All degree-two moments and squared coupled errors integrate exactly, with zero "
                "sampling uncertainty. This resolves the weak bias hidden by Monte Carlo noise. "
                "It applies to the uncapped extension; the fixed-cap contraction is checked separately."
            ),
            "",
            "| h | Integrated weak bias | Analytic weak bias | Integrated strong MSE | Analytic strong MSE |",
            "|---:|---:|---:|---:|---:|",
        ])
        for level in cubature["levels"]:
            lines.append(
                f"| {level['h']:.4g} | {level['cubature_weak_bias']:.10g} | "
                f"{level['predicted_weak_bias']:.10g} | {level['cubature_strong_MSE']:.10g} | "
                f"{level['predicted_strong_MSE']:.10g} |"
            )
        coefficient = cubature["analytic_weak_prefactor"]["C_weak"]
        lines.extend([
            "",
            (
                f"The proved population-independent coefficient is C_weak={coefficient:.12g} "
                "for the normalized second moment, horizon 0.16 and 0<h≤0.04 dividing the horizon. "
                "All three |bias|≤C_weak h² checks pass."
            ),
            "",
        ])
    lines.extend(["## Retained datasets", ""])
    for data in provenance:
        lines.extend([
            (
                f"- `{data['dataset']}`: {data['archives']:,} archives, "
                f"{data['native_recorded_steps']:,} recorded native steps, "
                f"{data['compressed_bytes'] / 1024**3:.3f} GiB compressed."
            )
        ])
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def cubature_plot(cubature, destination):
    import matplotlib.pyplot as plt

    levels = cubature["levels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout="constrained")
    hs = [level["h"] for level in levels]
    for key, prediction, axis, label in (
        ("cubature_weak_bias", "predicted_weak_bias", axes[0], "Absolute weak moment bias"),
        ("cubature_strong_MSE", "predicted_strong_MSE", axes[1], "Strong mean squared error"),
    ):
        values = [abs(level[key]) for level in levels]
        axis.loglog(hs, values, "o-", label="Native Gaussian integral")
        axis.loglog(
            hs, [abs(level[prediction]) for level in levels], "x--", label="Analytic prediction"
        )
        axis.loglog(
            hs, [values[0] * (h / hs[0]) ** 2 for h in hs], ":", label="Second-order reference"
        )
        axis.set_xlabel("Timestep")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Uncapped quadratic native BAOAB: exact Gaussian integration")
    fig.savefig(destination / "native-exact-timestep-refinement.png", dpi=180)
    plt.close(fig)


def kinetic_plots(chapter, destination):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.8, 4.7), layout="constrained")
    dataset = Path(chapter["dataset"])
    rates = []
    for case in chapter["capped_cases"]:
        raw = load_raw(dataset, case["raw_plan"], chapter["index"])
        trajectory = np.asarray(raw["trajectories"])
        mean = trajectory.mean(axis=0)
        # Direction is a fixed experimental stratum (replicate index mod 3),
        # so between-direction variation is not random sampling uncertainty.
        strata = [trajectory[index::3] for index in range(3)]
        se = np.sqrt(sum(len(group) * group.var(axis=0, ddof=1) for group in strata)) / len(
            trajectory
        )
        times = np.arange(len(mean)) * 0.04
        delta = raw["hypotheses"]["constants"]["delta"]
        rates.append({
            "N": case["N"],
            "d": case["d"],
            "independent_replicates": len(trajectory),
            "physical_time": float(times[-1]),
            "mean_final_error_ratio": float(mean[-1]),
            "final_error_ratio_standard_error": float(se[-1]),
            "standard_error_method": "Independent-noise replicates within the three fixed perturbation-direction strata; sum(n_h * sample_variance_h) / total_samples^2",
            "measured_endpoint_rate": float(-np.log(mean[-1]) / times[-1]),
            "measured_endpoint_rate_delta_method_standard_error": float(
                se[-1] / (mean[-1] * times[-1])
            ),
            "certified_lower_rate": float(-np.log1p(-delta) / 0.04),
            "scope": "Endpoint rate of normalized mean optimal Q transport against its initial error. Not an asymptotic rate fit; source theorem gives a conservative uniform lower rate.",
        })
        (line,) = ax.plot(times, mean, label=f"N={case['N']}, d={case['d']}")
        ax.fill_between(
            times,
            np.maximum(mean - 2 * se, 1e-16),
            mean + 2 * se,
            color=line.get_color(),
            alpha=0.12,
        )
    bounds = [
        c
        for c in chapter["comparisons"]
        if c["id"].startswith("n4_d1/step") and c["id"].endswith("/Q_contraction")
    ]
    if bounds:
        ax.plot(
            [int(c["id"].split("/step")[1].split("/")[0]) * 0.04 for c in bounds],
            [c["bound"] for c in bounds],
            "k--",
            label="Predicted envelope, uniform in N",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Normalized Q error / initial error")
    ax.set_title("Native capped kinetic contraction; bands show ±2 replicate SE")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(destination / "native-capped-contraction.png", dpi=180)
    plt.close(fig)
    levels = chapter["refinement"]["levels"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), layout="constrained")
    hs = [level["h"] for level in levels]
    for key, prediction, axis, label in (
        ("measured_weak_bias", "predicted_weak_bias", axes[0], "Absolute weak moment bias"),
        ("strong_MSE", "exact_strong_MSE", axes[1], "Strong mean squared error"),
    ):
        axis.loglog(hs, [abs(level[key]) for level in levels], "o-", label="Measured")
        axis.loglog(
            hs, [abs(level[prediction]) for level in levels], "x--", label="Exact prediction"
        )
        axis.set_xlabel("Timestep")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle("Native uncapped BAOAB vs exact quadratic SDE, shared Brownian paths")
    fig.savefig(destination / "native-timestep-refinement.png", dpi=180)
    plt.close(fig)
    return rates


def survivor_plots(chapter, destination):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.7), layout="constrained")
    for case in chapter["cases"]:
        frames = [
            frame
            for frame in case["checkpoint_law_frames"]
            if frame["cemetery_convention"] == "native_zero_alive"
        ]
        if not frames:
            continue
        ax.plot(
            [frame["step"] * 0.04 for frame in frames],
            [frame["own_survival_probability_estimates"][0] for frame in frames],
            marker=".",
            label=f"N={case['N']}, d={case['d']}",
        )
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Own survival denominator / initial replicates")
    ax.set_title("Independent killed-chain ensembles: native k=0 survival")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.savefig(destination / "native-survival.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.8, 4.7), layout="constrained")
    for case in chapter["cases"]:
        frames = [
            frame
            for frame in case["checkpoint_law_frames"]
            if frame["cemetery_convention"] == "native_zero_alive"
            and frame["whole_swarm_law_transport"].get("normalized_cost") is not None
        ]
        if not frames:
            continue
        mode = "exact" if case["N"] <= 4 else "coupling upper bound"
        ax.plot(
            [frame["step"] * 0.04 for frame in frames],
            [frame["whole_swarm_law_transport"]["normalized_cost"] for frame in frames],
            marker=".",
            label=f"N={case['N']}, d={case['d']} ({mode})",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Normalized surviving empirical-law squared transport cost")
    ax.set_title("Independent native survivor laws; finite sampling floor retained")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.savefig(destination / "native-survivor-law-decline.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cubature-review", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.datasets, args.output, args.cubature_review), indent=2))


if __name__ == "__main__":
    main()
