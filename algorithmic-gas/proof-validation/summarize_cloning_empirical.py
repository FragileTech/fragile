"""Summarize measured chapter 3 inequalities without replacing bounds by fitted rates."""

import argparse
from collections import defaultdict
import json
import operator
from pathlib import Path


def summarize(report: dict, destination: Path) -> dict:
    """Retain case-level observations, predictions, residuals and independent uncertainty."""
    destination.mkdir(parents=True, exist_ok=True)
    families = defaultdict(list)
    rows = []
    for case in report["cases"]:
        for check in case["comparisons"]:
            families[check["id"]].append(check)
            rows.append({
                "case": case["name"],
                "N": case["walkers"],
                "d": case["dimensions"],
                "alive": case["alive"],
                **check,
            })
    for case in report["balanced_keystone_cases"]:
        for check in case["comparisons"]:
            families[check["id"]].append(check)
            # Pressure lower bounds were represented as negative upper bounds
            # in Rust. Human-facing values retain the original positive quantities.
            rows.append({
                "case": "balanced_keystone",
                "N": case["N"],
                "d": case["d"],
                "alive": case["N"],
                **check,
            })
    result = {
        **report["summary"],
        "samples_per_case": report["samples_per_case"],
        "families": {
            name: {
                "comparisons": len(checks),
                "violated": sum(not check["not_rejected"] for check in checks),
                "upper_bounds_supported_by_six_se_diagnostic": sum(
                    check["upper_bound_supported"] for check in checks
                ),
                "maximum_equality_standardized_residual": max(
                    (
                        abs(check["residual_mean"]) / check["residual_standard_error"]
                        for check in checks
                        if check["relation"] == "equal"
                        and check["residual_standard_error"] > 0
                        and check["maximum_absolute_residual"]
                        > 2e-10 * (1 + abs(check["observed_mean"]) + abs(check["prediction_mean"]))
                    ),
                    default=None,
                ),
            }
            for name, checks in families.items()
        },
    }
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (destination / "comparisons.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Chapter 3 measured inequalities",
        "",
        (
            f"Native proposals: **{result['native_proposals']:,}**. Cases: **{result['cases']}**. "
            f"Independent repetitions per case: **{result['samples_per_case']}**. "
            f"Rejected comparisons: **{result['comparisons_violated']}**."
        ),
        "",
        (
            "Every repetition starts a fresh native engine at the recorded input. Conditional "
            "predictions retain the actual sampled fitness and accepted collision graph. Residual "
            "uncertainty is computed across repetitions, never across interacting walker rows. "
            "Proposal observables are measured after revival, copying, jitter and component collision, "
            "before kinetics. The transport test compares two independent proposal realizations."
        ),
        "",
        "| Measured estimate | Case comparisons | Rejected | Upper bounds supported by six-SE diagnostic |",
        "|---|---:|---:|---:|",
    ]
    for name, entry in result["families"].items():
        lines.append(
            f"| {name} | {entry['comparisons']} | {entry['violated']} "
            f"| {entry['upper_bounds_supported_by_six_se_diagnostic']} |"
        )
    lines += [
        "",
        (
            "Compatibility with a statistical tolerance is distinct from one-sided support "
            "for an upper bound. Six-SE intervals are diagnostics, not simultaneous confidence "
            "certificates. Complete formula coverage has its separate strict gate."
        ),
        "",
        "## Direct Keystone feedback comparison",
        "",
        (
            "The coefficient is fixed before simulation: chi = (4/9) A0(0.5). Both clouds "
            "have zero velocities and canonical quadratic rewards, at radii 1 and 0.5. "
            "The displayed measured coefficient is Q/Vstruct, with the complete sampled "
            "measurement law retained. It is a feedback coefficient, not an inferred "
            "stationary-law convergence rate."
        ),
        "",
        "| N | d | Theoretical chi | Measured Q/Vstruct | SE | Difference from lower bound |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["balanced_keystone_cases"]:
        check = case["comparisons"][0]
        observed = -check["observed_mean"] / case["Vstruct"]
        se = check["residual_standard_error"] / case["Vstruct"]
        lines.append(
            f"| {case['N']} | {case['d']} | {case['chi']:.9g} | {observed:.9g} "
            f"| {se:.3g} | {observed - case['chi']:.9g} |"
        )
    lines += [
        "",
        "## Every case and observable",
        "",
        (
            "The bound includes its offset. An observed decay slope need not equal a "
            "conservative theorem coefficient. Exact identities compare observed and predicted "
            "means; upper bounds compare observed-minus-predicted residuals."
        ),
        "",
        "| Case | Observable | Observed mean | Prediction / upper bound | Residual | SE | Compatible |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["case"] == "balanced_keystone":
            continue
        lines.append(
            f"| {row['case']} | {row['id']} | {row['observed_mean']:.9g} "
            f"| {row['prediction_mean']:.9g} | {row['residual_mean']:.3g} "
            f"| {row['residual_standard_error']:.3g} | {row['not_rejected']} |"
        )
    lines += [
        "",
        (
            "The boundary experiment uses the chapter's declared-observable version with "
            "phi(x)=|x|^2. Each measurement realization retains its exposed-selection "
            "probability, including zero. This validates the conditional drift formula and "
            "integrals; it does not infer a strictly positive state-uniform boundary rate "
            "when the selection hypotheses fail. Dead coordinates of magnitude 1e9 are "
            "overwritten by eligible live donors, and never set the reset constant."
        ),
        "",
        (
            "External stationary-law and QSD convergence theorems referenced by Chapter 3 "
            "require their own law-level experiments and are not certified by proposal drift."
        ),
    ]
    (destination / "summary.md").write_text("\n".join(lines) + "\n")
    return result


def plot(report: dict, destination: Path) -> None:
    """Create standalone scientific figures with independently seeded uncertainty."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in sorted({case["d"] for case in report["balanced_keystone_cases"]}):
        cases = sorted(
            (case for case in report["balanced_keystone_cases"] if case["d"] == d),
            key=operator.itemgetter("N"),
        )
        ax.errorbar(
            [case["N"] for case in cases],
            [-case["comparisons"][0]["observed_mean"] / case["Vstruct"] for case in cases],
            yerr=[
                case["comparisons"][0]["residual_standard_error"] / case["Vstruct"]
                for case in cases
            ],
            marker="o",
            capsize=3,
            label=f"native d={d}, ±1 SE",
        )
    ax.axhline(
        report["balanced_keystone_cases"][0]["chi"],
        color="black",
        linestyle="--",
        label="proved population-independent lower bound",
    )
    ax.set(
        xlabel="Population N",
        ylabel="Measured error-weighted activity / structural error",
        title="Chapter 3 Keystone feedback: prediction versus native proposals",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "keystone-feedback.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for profile in sorted({case["name"].split("_")[1] for case in report["cases"]}):
        cases = sorted(
            (
                case
                for case in report["cases"]
                if case["benchmark"] == "quadratic"
                and case["dimensions"] == 1
                and case["name"].split("_")[1] == profile
            ),
            key=operator.itemgetter("walkers"),
        )
        if not cases:
            continue
        checks = [
            next(check for check in case["comparisons"] if check["id"] == "positional_reset_Bx")
            for case in cases
        ]
        ax.errorbar(
            [case["walkers"] for case in cases],
            [check["observed_mean"] / check["prediction_mean"] for check in checks],
            yerr=[check["residual_standard_error"] / check["prediction_mean"] for check in checks],
            marker="o",
            capsize=3,
            label=profile,
        )
    ax.axhline(1, color="black", linestyle="--", label="proved reset upper bound")
    ax.set(
        xlabel="Population N",
        ylabel="Post-cloning variance / reset upper bound",
        title="Chapter 3 positional reset, quadratic landscape, d=1",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "positional-reset.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    print(json.dumps(summarize(data, args.output), indent=2))
    if args.plot:
        plot(data, args.output)
