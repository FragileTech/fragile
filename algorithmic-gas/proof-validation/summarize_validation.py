"""Summarize a native chapter 1--3 report and plot its recorded observables."""

import argparse
from collections import defaultdict
import json
from operator import itemgetter
from pathlib import Path


def constant_ledger(report: dict) -> list[dict]:
    """Retain computed constants and their original scope without merging symbols."""
    ledger = []

    def visit(value, path):
        if isinstance(value, dict):
            if "value" in value and ("formula" in value or "symbol" in value):
                ledger.append({"report_path": path, **value})
                return
            if "log_revival_ratio" in value:
                inputs = {
                    key: value[key] for key in ("eta", "alpha", "beta", "epsilon_clone", "p_max")
                }
                for name in (
                    "fitness_floor",
                    "score_lower_bound",
                    "revival_ratio",
                    "exact_uniform_acceptance",
                    "log_fitness_floor",
                    "log_score_lower_bound",
                    "log_revival_ratio",
                    "log_uniform_acceptance",
                ):
                    if isinstance(value.get(name), (float, int)):
                        ledger.append({
                            "report_path": f"{path}/{name}",
                            "name": name,
                            "value": value[name],
                            "representation": "natural_log"
                            if name.startswith("log_")
                            else "value",
                            "scope": value["scope"],
                            "inputs": inputs,
                        })
            if "lyapunov_matrix" in value and "contraction_factor" in value:
                for name, number in value.items():
                    if isinstance(number, (float, int)):
                        ledger.append({
                            "report_path": f"{path}/{name}",
                            "name": name,
                            "value": number,
                            "scope": value["scope"],
                        })
                for name in ("map", "innovation_covariance", "lyapunov_matrix"):
                    ledger.append({
                        "report_path": f"{path}/{name}",
                        "name": name,
                        "value": value[name],
                        "scope": value["scope"],
                    })
            for key, child in value.items():
                if key in {"checks", "trajectory", "coverage", "runs"}:
                    continue
                if key in {"constants", "natural_logs", "values"} and isinstance(child, dict):
                    for name, number in child.items():
                        if isinstance(number, (float, int)):
                            ledger.append({
                                "report_path": f"{path}/{key}/{name}",
                                "name": name,
                                "value": number,
                                "representation": "natural_log"
                                if key == "natural_logs" or name.startswith("log_")
                                else "value",
                                "scope": value.get(
                                    "scope",
                                    value.get(
                                        "scope_notes", "See parent report inputs and hypotheses"
                                    ),
                                ),
                            })
                else:
                    visit(child, f"{path}/{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}/{index}")

    visit(report, "")
    return ledger


def summarize(report: dict, destination: Path) -> None:
    """Write source-scoped counts, rates, and experimental results for review."""
    destination.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    counts = report["coverage"]["formal_claim_counts"]
    ledger = constant_ledger(report)
    (destination / "constants.json").write_text(json.dumps(ledger, indent=2) + "\n")
    lines = [
        "# Rust validation of convergence chapters 1–3",
        "",
        (
            f"Native f64 simulations: **{summary['runs']} runs**. "
            f"Exact checks: **{summary['exact_checks']:,}**, "
            f"failures: **{summary['exact_checks_failed']}**. "
            f"Engine/diagnostic errors: **{summary['run_errors']}**; "
            f"auxiliary phase errors: **{summary.get('phase_errors', 0)}**; "
            f"extinctions: **{summary['extinctions']}**."
        ),
        "",
        (
            "Exact checks recompute the implemented stage laws and their conditional moments. "
            "Cloning is measured after revival, jitter and the component collision, before kinetics. "
            "The entering support-change count in the repaired Sasaki bound is not cumulative "
            "particle loss."
        ),
        "",
        (
            f"Kinetic statistical checks: **{summary['kinetic_statistical_checks_not_rejected']}** "
            f"not rejected, **{summary['kinetic_statistical_checks_violated']}** violated. "
            "These use independent frozen-input experiments and report standard errors. "
            "A six-standard-error diagnostic threshold is not a simultaneous confidence theorem."
        ),
        "",
        (
            f"Cloning statistical checks: **{summary['cloning_statistical_checks_not_rejected']}** "
            f"not rejected, **{summary['cloning_statistical_checks_violated']}** violated. "
            f"Auxiliary statistical checks: **{summary.get('auxiliary_statistical_checks_not_rejected', 0)}** "
            f"not rejected, **{summary.get('auxiliary_statistical_checks_violated', 0)}** violated."
        ),
        "",
        "| Inventory disposition | Count |",
        "|---|---:|",
    ]
    for key, value in counts.items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    lines += [
        "",
        (
            "A source label being exercised means its recorded diagnostic was evaluated under "
            "the listed scope. It does not certify every dependency or every formula in that claim. "
            "Every remaining formula and hypothesis stays visible in the inventory."
        ),
        "",
        (
            f"The [constant ledger](constants.json) contains **{len(ledger):,} scoped values** "
            "and log values. Repeated symbols retain their experiment inputs and report paths; "
            "constants from different hypotheses are not interchangeable."
        ),
        "",
        "| Canonical Keystone dimension | W0 | log(chi0) | log(N0 before rounding) |",
        "|---:|---:|---:|---:|",
    ]
    for constants in report.get("canonical_keystone_constants", []):
        logs = constants["natural_logs"]
        lines.append(
            f"| {constants['dimensions']} | {constants['error_threshold']:g} "
            f"| {logs['chi_0']:.6g} | {logs['N_0_unrounded']:.6g} |"
        )
    lines += [
        "",
        (
            "These explicit selection-pressure rates depend on the canonical box, pipeline "
            "and analytic reward modulus. The displayed finite-population theorem requires "
            "its N0 threshold. A finite swarm below that threshold cannot validate its asymptotic "
            "rate. Rates smaller than the f64 range remain in logarithmic form."
        ),
        "",
        (
            "| Independent cloning experiment | Samples | Mean variance | Exact prediction "
            "| Standard error | Result |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    calibration = report.get("independent_cloning") or {}
    for case in calibration.get("cases", []):
        estimate = case["positional_variance"]
        status = "Consistent" if estimate["consistent_with_theory"] else "Violated"
        lines.append(
            f"| {case['name']} | {estimate['samples']} | "
            f"{estimate['sample_mean']:.8g} | {estimate['exact_expectation']:.8g} | "
            f"{estimate['standard_error']:.3g} | {status} |"
        )
    lines += [
        "",
        (
            "| Complete-step revival/reset experiment | Dead coordinate magnitude | M | "
            "Measured marked moment | Standard error | Result |"
        ),
        "|---|---:|---:|---:|---:|---|",
    ]
    reset = report.get("complete_boundary_reset") or {}
    for case in reset.get("ensembles", []):
        estimate = next(
            item for item in case["estimates"] if item["id"] == "complete_marked_reset_M"
        )
        lines.append(
            f"| {case['name']} | {case['entering_dead_coordinate_magnitude']:g} | "
            f"{case['constants']['M']:.6g} | {estimate['sample_mean']:.6g} | "
            f"{estimate['standard_error']:.3g} | {estimate['status'].replace('_', ' ')} |"
        )
    population = (report.get("framework_continuity") or {}).get("population_scaling") or {}
    lines += [
        "",
        (
            "The corrected empirical standardizer has squared transport gain "
            "**1 / sigma_min²**, independent of population size, with no additive error term. "
            "The conditional-mixture measurement error has the decreasing bound "
            "**D² / (2 sigma_min² sqrt(k))**. The table uses actual realized empirical laws "
            "and preserves the sampled native global normalizer."
        ),
        "",
        "| Companion law | N | Exact standardized empirical error | Native estimate | Standard error |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in population.get("cases", []):
        lines.append(
            f"| {case['companion_law']} | {case['walkers']} | "
            f"{case['exact_standardized_normalized_empirical_error']:.6g} | "
            f"{case['observed_standardized_normalized_empirical_error']:.6g} | "
            f"{case['standardized_sample_standard_error']:.3g} |"
        )
    if population.get("cases"):
        lines += ["", "![Empirical error versus population size](population-error.png)"]
    decay = report.get("complete_update_decay") or {}
    shared_decay = report.get("shared_update_decay") or {}
    decay_cases_for_table = []
    for source in (decay, shared_decay):
        mode = source.get("config", {}).get("pair_randomness", "independent")
        decay_cases_for_table.extend(
            {**case, "pair_randomness": mode} for case in source.get("cases", [])
        )
    lines += [
        "",
        (
            "The chapter convergence error uses normalized optimal transport of the alive "
            "empirical laws. Dead coordinates overwritten by revival do not enter this error. "
            "Full marked all-slot transport remains a separate JSON diagnostic. "
            "Rates use the fixed initial time window, with uncertainty from independent "
            "seed pairs. Linear controls compare with a contraction factor computed before the "
            "runs and retain the independent-noise floor. Canonical cases keep actual selection, "
            "collision, cap, terminal deaths and revival; the control theorem is not assigned to them."
        ),
        "",
        "| Random coupling | Decay profile | N | d | Initial error | Final error | Fitted squared-error rate ± SE | Proved envelope rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in decay_cases_for_table:
        points = case["ensemble_trajectory"]
        initial = points[0]["alive_error"]
        final = points[-1]["alive_error"]
        rate = case["empirical_rate"]
        estimate = rate["estimated_squared_error_rate"]
        rate_error = rate["seed_jackknife_standard_error"]
        rate_text = f"{estimate:.5g}" if estimate is not None else "Unavailable"
        if rate_error is not None:
            rate_text += f" ± {rate_error:.2g}"
        final_text = f"{final['mean']:.6g}" if final is not None else "Incomplete ensemble"
        analytic_rate = (
            f"{case['reference_linear_constants']['squared_error_decay_rate']:.5g}"
            if case["linear_control_theorem_applicable"]
            else "Inapplicable"
        )
        lines.append(
            f"| {case['pair_randomness']} | {case['profile']} | {case['walkers']} | {case['dimensions']} | "
            f"{initial['mean']:.6g} | {final_text} | {rate_text} | "
            f"{analytic_rate} |"
        )
    if decay_cases_for_table:
        lines += ["", "![Complete-update empirical error](complete-update-error.png)"]
        lines += [
            "",
            (
                "Endpoint decline and a fitted slope answer different questions: a noisy "
                "fixed-window slope may be negative even when the final mean is below its "
                "initial value. The conditional affine contraction envelope bounds expected "
                "error and does not assert monotone realized paths. The fresh 32-seed baseline "
                "and every parameter sweep are retained in the "
                "[separate calibration comparison](../calibration/calibration-summary.md)."
            ),
        ]
    lines += ["", "![Recorded swarm variances](trajectories.png)", ""]
    (destination / "summary.md").write_text("\n".join(lines))
    (destination / "summary.json").write_text(
        json.dumps({"summary": summary, "coverage": counts}, indent=2) + "\n"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    decay_cases = defaultdict(list)
    plotted_decay = shared_decay if shared_decay.get("cases") else decay
    for case in plotted_decay.get("cases", []):
        decay_cases[case["profile"]].append(case)
    if decay_cases:
        fig, axes = plt.subplots(
            1, len(decay_cases), figsize=(4.4 * len(decay_cases), 4.3), squeeze=False
        )
        colors = {4: "#dd8545", 16: "#336b87", 64: "#5b8c5a"}
        for axis, (profile, cases) in zip(axes[0], sorted(decay_cases.items())):
            displayed_values = []
            for case in cases:
                points = case["ensemble_trajectory"]
                present = [point for point in points if point["alive_error"] is not None]
                color = colors.get(case["walkers"], "#444444")
                times = [point["time"] for point in present]
                means = [point["alive_error"]["mean"] for point in present]
                displayed_values.extend(means)
                errors = [2 * point["alive_error"]["standard_error"] for point in present]
                axis.plot(
                    times,
                    means,
                    color=color,
                    linestyle="-" if case["dimensions"] == 1 else "--",
                    label=f"N={case['walkers']}, d={case['dimensions']}",
                )
                axis.fill_between(
                    times,
                    [max(1e-12, mean - error) for mean, error in zip(means, errors)],
                    [mean + error for mean, error in zip(means, errors)],
                    color=color,
                    alpha=0.07,
                )
                if case["linear_control_theorem_applicable"] and case["dimensions"] == 1:
                    envelopes = [
                        point["noiseless_geometric_envelope"]
                        if point["noiseless_geometric_envelope"] is not None
                        else point["n_independent_geometric_noise_envelope"]
                        for point in present
                    ]
                    displayed_values.extend(value for value in envelopes if value is not None)
                    axis.plot(
                        times,
                        envelopes,
                        color=color,
                        linewidth=0.9,
                        linestyle=":",
                        alpha=0.8,
                        label="Proved envelope" if case["walkers"] == 4 else None,
                    )
            display_name = profile.replace("independent_noise", "gaussian_noise")
            axis.set_title(display_name.replace("_", "\n"), fontsize=10)
            axis.set_yscale("log")
            positive_values = [value for value in displayed_values if value > 0]
            if positive_values:
                axis.set_ylim(0.5 * min(positive_values), 1.3 * max(positive_values))
            axis.set_xlabel("Physical simulation time")
            axis.set_ylabel("Squared alive empirical transport error")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=7)
        mode = plotted_decay["config"].get("pair_randomness", "independent")
        fig.suptitle(
            f"Complete native updates with {mode} innovations; bands show ±2 SE across seed pairs"
        )
        fig.tight_layout()
        fig.savefig(destination / "complete-update-error.png", dpi=150)
        plt.close(fig)

    population_cases = defaultdict(list)
    for case in population.get("cases", []):
        population_cases[case["companion_law"]].append(case)
    if population_cases:
        fig, axes = plt.subplots(
            1, len(population_cases), figsize=(5.0 * len(population_cases), 3.8), squeeze=False
        )
        for axis, (law, cases) in zip(axes[0], sorted(population_cases.items())):
            cases.sort(key=itemgetter("walkers"))
            populations = [case["walkers"] for case in cases]
            for prefix, label, color in (
                ("raw", "Raw empirical law", "#dd8545"),
                ("standardized", "Native standardized law", "#336b87"),
            ):
                axis.plot(
                    populations,
                    [case[f"exact_{prefix}_normalized_empirical_error"] for case in cases],
                    color=color,
                    label=f"{label}: exact",
                )
                axis.errorbar(
                    populations,
                    [case[f"observed_{prefix}_normalized_empirical_error"] for case in cases],
                    yerr=[2 * case[f"{prefix}_sample_standard_error"] for case in cases],
                    fmt="o",
                    color=color,
                    markersize=4,
                )
            axis.set_title(law)
            axis.set_xscale("log", base=2)
            axis.set_yscale("log")
            axis.set_xlabel("Swarm population N")
            axis.set_ylabel("Expected squared empirical W2 error")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        fig.suptitle("Independent native measurements; points show ±2 standard errors")
        fig.tight_layout()
        fig.savefig(destination / "population-error.png", dpi=150)
        plt.close(fig)

    grouped = defaultdict(list)
    for run in report["runs"]:
        if run.get("outcome") != "completed":
            continue
        # Plot canonical and collapsed starts separately; profile sweeps remain in JSON.
        if "_canonical_" in run["id"] or "_collapsed_" in run["id"]:
            benchmark = run["config"]["benchmark"]
            label = benchmark if isinstance(benchmark, str) else benchmark["id"]
            grouped[label].append(run)
    if not grouped:
        return
    fig, axes = plt.subplots(1, len(grouped), figsize=(4.1 * len(grouped), 3.8), squeeze=False)
    for axis, (landscape, runs) in zip(axes[0], sorted(grouped.items())):
        for run in runs:
            collapsed = "_collapsed_" in run["id"]
            values = [
                point["moments"].get("position_variance", float("nan"))
                for point in run["trajectory"]
            ]
            axis.plot(
                range(len(values)), values, alpha=0.32, color="#dd8545" if collapsed else "#336b87"
            )
        axis.set_title(landscape)
        axis.set_xlabel("Native update")
        axis.set_ylabel("Alive positional variance")
        axis.grid(alpha=0.2)
    fig.suptitle("Blue: dispersed starts; orange: collapsed starts. Individual trajectories.")
    fig.tight_layout()
    fig.savefig(destination / "trajectories.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    summarize(json.loads(arguments.report.read_text()), arguments.output)


if __name__ == "__main__":
    main()
