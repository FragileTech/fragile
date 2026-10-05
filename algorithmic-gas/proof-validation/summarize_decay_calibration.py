"""Summarize every native decay calibration without selecting parameters.

The primary error is optimal physical q-W2 squared between the normalized alive
empirical measures. Legacy full-marked endpoints are recomputed from seed traces
and retained separately. Missing or extinct alive measures remain explicit.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from operator import itemgetter
from pathlib import Path


EXPLORATORY_SEEDS = frozenset((7, 173, 1729, 7919, 104729, 31337, 65537, 99991))
PRECISION_SEEDS = frozenset(range(100032, 100096))
PRIMARY = "alive_wasserstein_squared"
DIAGNOSTIC = "full_marked_error"


def number(value):
    """Accept finite numerical observations, preserving unavailable values."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def mean(values):
    return sum(values) / len(values) if values else None


def standard_error(values):
    if len(values) < 2:
        return None
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1)) / (
        len(values) ** 0.5
    )


def jackknife_error(estimates):
    if len(estimates) < 2:
        return None
    center = mean(estimates)
    size = len(estimates)
    return math.sqrt((size - 1) / size * sum((value - center) ** 2 for value in estimates))


def cohort(seeds):
    actual = set(seeds)
    if len(seeds) == 8 and actual == EXPLORATORY_SEEDS:
        return "exploratory_original_8"
    if len(seeds) == 64 and actual == PRECISION_SEEDS:
        return "fresh_precision_64"
    if len(seeds) == 32 and len(actual) == 32 and actual.isdisjoint(EXPLORATORY_SEEDS):
        return "heldout_fresh_32"
    if actual.intersection(EXPLORATORY_SEEDS):
        return "overlaps_original_exploratory_seeds"
    return "other_seed_set"


def family(profile):
    if profile == "rastrigin_diffusion_calibration":
        return "position_diffusion_grid"
    if profile == "rastrigin_cloning_calibration":
        return "clone_jitter_grid"
    if profile.startswith("canonical_"):
        return "canonical_baseline"
    if profile.startswith("linear_"):
        return "linear_control"
    return "other_profile"


def point_maps(case):
    return [
        {point["step"]: point for point in trajectory.get("points", [])}
        for trajectory in case.get("seed_trajectories", [])
    ]


def observation(point, observable):
    if not isinstance(point, dict):
        return None
    return number(point.get("observables", {}).get(observable))


def endpoint(maps, last_step, observable):
    """Compare the same pairs at both endpoints, retaining exclusion counts."""
    values = []
    for points in maps:
        first = observation(points.get(0), observable)
        last = observation(points.get(last_step), observable)
        if first is not None and last is not None:
            values.append((first, last))
    initial = mean([first for first, _ in values])
    final = mean([last for _, last in values])
    changes = [last - first for first, last in values]
    ratio = final / initial if initial is not None and initial > 0 else None
    ratio_jackknife = []
    if len(values) >= 2:
        for omitted in range(len(values)):
            retained = [pair for index, pair in enumerate(values) if index != omitted]
            denominator = mean([first for first, _ in retained])
            if denominator > 0:
                ratio_jackknife.append(mean([last for _, last in retained]) / denominator)
    return {
        "observable": observable,
        "defined_endpoint_seed_pairs": len(values),
        "mean_initial_error": initial,
        "mean_final_error": final,
        "terminal_to_initial_ratio": ratio,
        "ratio_jackknife_standard_error": jackknife_error(ratio_jackknife)
        if len(ratio_jackknife) == len(values)
        else None,
        "paired_mean_change": mean(changes),
        "paired_change_standard_error": standard_error(changes),
        "empirical_mean_decreased": final < initial
        if initial is not None and final is not None
        else None,
    }


def log_rate(times, values):
    if len(values) < 3 or any(value <= 0 for value in values):
        return None
    center_time = mean(times)
    center_log = mean([math.log(value) for value in values])
    denominator = sum((time - center_time) ** 2 for time in times)
    if denominator <= 0:
        return None
    return (
        -sum(
            (time - center_time) * (math.log(value) - center_log)
            for time, value in zip(times, values, strict=True)
        )
        / denominator
    )


def fixed_window_rate(case, config, maps, configured_pairs):
    """Use every configured pair and every step in the fixed window."""
    native = case.get("empirical_rate", {})
    start = int(native.get("fixed_start_step", 0))
    end = int(native.get("fixed_end_step", config.get("fit_end_step", 64)))
    result = {
        "observable": PRIMARY,
        "fixed_start_step": start,
        "fixed_end_step": end,
        "estimated_squared_error_rate": None,
        "seed_jackknife_standard_error": None,
        "jackknife_estimates_available": 0,
        "noise_floor_subtracted": False,
        "status": "unavailable_incomplete_fixed_window",
        "configured_seed_pairs": configured_pairs,
        "complete_window_seed_pairs": 0,
        "missing_window_seed_pairs": max(configured_pairs - len(maps), 0),
        "empty_alive_window_seed_pairs": 0,
    }
    for points in maps:
        if any(step not in points for step in range(start, end + 1)):
            result["missing_window_seed_pairs"] += 1
        elif any(observation(points[step], PRIMARY) is None for step in range(start, end + 1)):
            result["empty_alive_window_seed_pairs"] += 1
        else:
            result["complete_window_seed_pairs"] += 1
    if not maps or len(maps) != configured_pairs:
        return result
    samples = []
    times = []
    for step in range(start, end + 1):
        points = [points.get(step) for points in maps]
        values = [observation(point, PRIMARY) for point in points]
        if any(value is None for value in values):
            return result
        time = number(points[0].get("time"))
        if time is None:
            time = step * float(config.get("dt", 0.04))
        times.append(time)
        samples.append(values)
    estimate = log_rate(times, [mean(values) for values in samples])
    if estimate is None:
        result["status"] = "unavailable_nonpositive_fixed_window_error"
        return result
    estimates = []
    for omitted in range(len(maps)) if len(maps) >= 2 else ():
        retained = [
            mean([value for index, value in enumerate(values) if index != omitted])
            for values in samples
        ]
        candidate = log_rate(times, retained)
        if candidate is not None:
            estimates.append(candidate)
    result.update({
        "estimated_squared_error_rate": estimate,
        "seed_jackknife_standard_error": jackknife_error(estimates)
        if len(estimates) == len(maps)
        else None,
        "jackknife_estimates_available": len(estimates),
        "status": "descriptive_fixed_window_rate",
    })
    return result


def seed_completion(trajectory, points, last_step):
    """Retain which seed pairs supply a defined, complete alive endpoint."""
    first = observation(points.get(0), PRIMARY)
    last = observation(points.get(last_step), PRIMARY)
    if last_step not in points:
        status = "missing_terminal_step"
    elif first is None:
        status = "undefined_initial_alive_measure"
    elif last is None:
        status = "undefined_terminal_alive_measure"
    else:
        status = "complete_alive_endpoints"
    return {
        "pair_seed": trajectory.get("pair_seed"),
        "native_engine_seeds": trajectory.get("native_engine_seeds"),
        "complete_native_trajectory": last_step in points,
        "status": status,
        "initial_alive_counts": points.get(0, {}).get("observables", {}).get("alive_counts"),
        "terminal_alive_counts": points
        .get(last_step, {})
        .get("observables", {})
        .get("alive_counts"),
        "initial_alive_error": first,
        "terminal_alive_error": last,
        "termination": trajectory.get("termination"),
    }


def completion_status(configured, complete_native, defined_alive):
    if configured <= 0:
        return "configured_pair_count_unavailable"
    if complete_native > configured or defined_alive > complete_native:
        return "inconsistent_completion_metadata"
    if defined_alive == 0:
        return "no_defined_alive_endpoint_pairs"
    if complete_native != configured or defined_alive != configured:
        return "incomplete_alive_endpoint_ensemble"
    return "complete"


def summarize_case(path, digest, report, case, index):
    config = report.get("config", {})
    seeds = config.get("seeds", [])
    steps = int(config.get("steps", 0))
    maps = point_maps(case)
    complete_native = sum(steps in points for points in maps)
    primary = endpoint(maps, steps, PRIMARY)
    diagnostic = endpoint(maps, steps, DIAGNOSTIC)
    native = case.get("native_config", {})
    profile = str(case.get("profile", "unspecified"))
    rate = fixed_window_rate(case, config, maps, len(seeds))
    return {
        "source_file": path.name,
        "source_sha256": digest,
        "source_case_index": index,
        "profile": profile,
        "experiment_family": family(profile),
        "coupling": case.get("pair_randomness", config.get("pair_randomness", "unspecified")),
        "seed_cohort": cohort(seeds),
        "configured_seed_addresses": seeds,
        "original_exploratory_seed_overlap": sorted(set(seeds).intersection(EXPLORATORY_SEEDS)),
        "walkers": case.get("walkers"),
        "dimensions": case.get("dimensions"),
        "steps": steps,
        "dt": config.get("dt"),
        "position_diffusion": native.get("kinetic", {}).get("position_diffusion"),
        "clone_jitter_amplitude": native.get("clone_transform", {}).get("jitter_amplitude"),
        "configured_seed_pairs": len(seeds),
        "reported_seed_trajectories": len(maps),
        "complete_native_seed_pairs": complete_native,
        "defined_alive_endpoint_seed_pairs": primary["defined_endpoint_seed_pairs"],
        "excluded_alive_endpoint_seed_pairs": max(
            len(seeds) - primary["defined_endpoint_seed_pairs"], 0
        ),
        "seed_pair_completion": [
            seed_completion(trajectory, points, steps)
            for trajectory, points in zip(case.get("seed_trajectories", []), maps, strict=True)
        ],
        "completion_status": completion_status(
            len(seeds), complete_native, primary["defined_endpoint_seed_pairs"]
        ),
        "primary_alive_endpoint": primary,
        "primary_alive_fixed_window_rate": rate,
        "full_marked_endpoint_diagnostic": diagnostic,
        "native_endpoint_decline": case.get("endpoint_decline"),
        "native_empirical_rate": case.get("empirical_rate"),
        "native_terminal_window_mean_error": case.get("terminal_window_mean_error"),
        "linear_control_theorem_applicable": case.get("linear_control_theorem_applicable", False),
        "total_revivals": case.get("total_revivals"),
        "terminations": [
            {"pair_seed": trajectory.get("pair_seed"), "reason": trajectory.get("termination")}
            for trajectory in case.get("seed_trajectories", [])
            if trajectory.get("termination")
        ],
        "native_config": native,
        "diffusion_calibration": case.get("diffusion_calibration"),
        "cloning_calibration": case.get("cloning_calibration"),
        "scope": case.get("scope"),
        "applicability_flags": case.get("applicability_flags", []),
        "endpoint_metric_source": "Recomputed alive empirical q-W2 squared from each completed seed history, including legacy reports whose native endpoint metadata used full marked error. Native metadata is retained separately.",
    }


def format_number(value):
    return "unavailable" if value is None else f"{value:.5g}"


def format_uncertainty(value, error):
    return f"{format_number(value)} ± {format_number(error)}"


def markdown(summary):
    lines = [
        "# Native decay and calibration comparisons",
        "",
        "The primary observable is optimal physical q-W2 squared between normalized alive empirical measures. Endpoints and fixed-window rates are recomputed from seed histories, so legacy full-marked metadata does not set the convergence comparison. Full marked error remains a separate diagnostic.",
        "",
        "Each seed pair is an independent experimental unit. Independent coupling retains finite-population sampling noise; shared coupling uses native common streams on intrinsic swarm representatives. Canonical and calibrated curves have no certified floor subtraction or assigned analytic contraction rate. A descriptive fitted slope is not a proved rate.",
        "",
        "Original eight-seed explorations, disjoint fresh 32-seed experiments, and the separate 64-seed precision check (seed addresses 100032–100095) are listed separately. Parameters come from each actual native configuration. No candidate is selected and no preset is changed by this summary. Uncertainty is one standard error across seed pairs; rate and ratio uncertainty use seed jackknife estimates.",
        "",
        "An endpoint with fewer defined alive pairs than configured is incomplete. Its displayed mean uses only pairs with defined alive measures at both endpoints and does not establish an unconditional or survival-conditioned theorem. Every exclusion and termination remains in the JSON summary.",
        "",
    ]
    groups = defaultdict(list)
    for record in summary["cases"]:
        groups[record["seed_cohort"]].append(record)
    for label, records in groups.items():
        lines.extend([
            f"## {label.replace('_', ' ')}",
            "",
            "| Source / profile | Coupling | N / d | σ position / σ clone | Native complete / alive defined / configured pairs | Initial → final alive error | Alive ratio | Paired alive change ± SE | Fixed-window alive rate ± SE | Completion |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        for row in records:
            endpoint_values = row["primary_alive_endpoint"]
            rate = row["primary_alive_fixed_window_rate"]
            lines.append(
                f"| [{row['source_file']}]({row['source_file']}) / {row['profile']} "
                f"| {row['coupling']} | {row['walkers']} / {row['dimensions']} "
                f"| {format_number(number(row['position_diffusion']))} / "
                f"{format_number(number(row['clone_jitter_amplitude']))} "
                f"| {row['complete_native_seed_pairs']} / "
                f"{row['defined_alive_endpoint_seed_pairs']} / {row['configured_seed_pairs']} "
                f"| {format_number(endpoint_values['mean_initial_error'])} → "
                f"{format_number(endpoint_values['mean_final_error'])} "
                f"| {format_number(endpoint_values['terminal_to_initial_ratio'])} "
                f"| {format_uncertainty(endpoint_values['paired_mean_change'], endpoint_values['paired_change_standard_error'])} "
                f"| {format_uncertainty(rate['estimated_squared_error_rate'], rate['seed_jackknife_standard_error'])} "
                f"| {row['completion_status']} |"
            )
        lines.append("")
    lines.extend([
        "## Full marked diagnostic",
        "",
        "| Source / profile | Coupling | N / d | Full marked initial → final | Full marked ratio |",
        "|---|---|---:|---:|---:|",
    ])
    for row in summary["cases"]:
        diagnostic = row["full_marked_endpoint_diagnostic"]
        lines.append(
            f"| {row['source_file']} / {row['profile']} | {row['coupling']} "
            f"| {row['walkers']} / {row['dimensions']} "
            f"| {format_number(diagnostic['mean_initial_error'])} → "
            f"{format_number(diagnostic['mean_final_error'])} "
            f"| {format_number(diagnostic['terminal_to_initial_ratio'])} |"
        )
    if summary["read_errors"]:
        lines.extend(["", "## Reports not read", ""])
        lines.extend(
            f"- {item['source_file']}: {item['reason']}" for item in summary["read_errors"]
        )
    lines.extend(["", "![Alive-error calibration comparisons](calibration-comparison.png)", ""])
    return "\n".join(lines)


def plot(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), constrained_layout=True)
    records = summary["cases"]
    for column, coupling in enumerate(("independent", "shared")):
        for row, (experiment, parameter, xlabel) in enumerate((
            (
                "position_diffusion_grid",
                "position_diffusion",
                "Native position diffusion σ position",
            ),
            ("clone_jitter_grid", "clone_jitter_amplitude", "Accepted-copy jitter σ clone"),
        )):
            axis = axes[row, column]
            selected = [
                item
                for item in records
                if item["coupling"] == coupling
                and item["seed_cohort"] == "exploratory_original_8"
                and (
                    item["experiment_family"] == experiment
                    or item["profile"] == "canonical_rastrigin"
                )
            ]
            series = defaultdict(list)
            for item in selected:
                series[item["walkers"], item["dimensions"]].append(item)
            for group_index, ((walkers, dimensions), items) in enumerate(series.items()):
                color = f"C{group_index % 10}"
                items.sort(key=lambda item: (number(item[parameter]) or 0, item["source_file"]))
                coordinates = [
                    (
                        number(item[parameter]),
                        item["primary_alive_endpoint"]["terminal_to_initial_ratio"],
                    )
                    for item in items
                    if item["completion_status"] == "complete"
                    and number(item[parameter]) is not None
                    and item["primary_alive_endpoint"]["terminal_to_initial_ratio"] is not None
                ]
                if coordinates:
                    axis.plot(
                        *zip(*coordinates, strict=True),
                        color=color,
                        alpha=0.5,
                        label=f"N={walkers}, d={dimensions}",
                    )
                for item in items:
                    x = number(item[parameter])
                    values = item["primary_alive_endpoint"]
                    y = values["terminal_to_initial_ratio"]
                    if x is None:
                        continue
                    if y is None:
                        axis.annotate(
                            f"{item['defined_alive_endpoint_seed_pairs']}/{item['configured_seed_pairs']} alive pairs",
                            (x, 1),
                            xytext=(0, 8),
                            textcoords="offset points",
                            rotation=60,
                            fontsize=7,
                        )
                        axis.plot(x, 1, "x", color="gray")
                    else:
                        complete = item["completion_status"] == "complete"
                        marker = (
                            "*"
                            if item["profile"] == "canonical_rastrigin"
                            else ("o" if dimensions == 1 else "s")
                        )
                        axis.errorbar(
                            x,
                            y,
                            yerr=values["ratio_jackknife_standard_error"],
                            fmt=marker,
                            color=color,
                            markersize=8 if marker == "*" else 5,
                            markerfacecolor=color if complete else "none",
                            capsize=3,
                            alpha=1 if complete else 0.5,
                        )
                        if not complete:
                            axis.annotate(
                                f"{item['defined_alive_endpoint_seed_pairs']}/{item['configured_seed_pairs']}",
                                (x, y),
                                xytext=(4, 5),
                                textcoords="offset points",
                                fontsize=7,
                            )
            axis.set_title(f"{coupling.capitalize()} coupling · {experiment.replace('_', ' ')}")
            axis.set_xlabel(xlabel)
            axis.set_ylabel("Alive endpoint / initial mean error")
            axis.axhline(1, color="black", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.2)
            if series:
                axis.legend(fontsize=8)
            else:
                axis.text(0.5, 0.5, "No grid report yet", transform=axis.transAxes, ha="center")
        axis = axes[2, column]
        heldout = [
            item
            for item in records
            if item["coupling"] == coupling
            and item["seed_cohort"] in {"heldout_fresh_32", "fresh_precision_64"}
        ]
        groups = defaultdict(list)
        for item in heldout:
            groups[
                item["seed_cohort"],
                item["profile"],
                item["dimensions"],
                item["position_diffusion"],
                item["clone_jitter_amplitude"],
            ].append(item)
        for group_index, (
            (seed_cohort, profile, dimensions, diffusion, jitter),
            items,
        ) in enumerate(groups.items()):
            color = f"C{group_index % 10}"
            precision = seed_cohort == "fresh_precision_64"
            complete = [
                item
                for item in sorted(items, key=itemgetter("walkers"))
                if item["completion_status"] == "complete"
                and item["primary_alive_endpoint"]["terminal_to_initial_ratio"] is not None
            ]
            if complete:
                axis.errorbar(
                    [item["walkers"] for item in complete],
                    [
                        item["primary_alive_endpoint"]["terminal_to_initial_ratio"]
                        for item in complete
                    ],
                    yerr=[
                        item["primary_alive_endpoint"]["ratio_jackknife_standard_error"] or 0
                        for item in complete
                    ],
                    fmt="D" if precision else ("o-" if dimensions == 1 else "s--"),
                    color=color,
                    capsize=3,
                    label=f"{'64 precision' if precision else '32 heldout'}: {profile.removeprefix('canonical_')}, d={dimensions}, σx={diffusion}, σc={jitter}",
                )
            for item in items:
                if item["completion_status"] == "complete":
                    continue
                values = item["primary_alive_endpoint"]
                ratio = values["terminal_to_initial_ratio"]
                if ratio is None:
                    axis.plot(item["walkers"], 1, "x", color="gray")
                else:
                    axis.errorbar(
                        item["walkers"],
                        ratio,
                        yerr=values["ratio_jackknife_standard_error"],
                        fmt="D" if precision else ("o" if dimensions == 1 else "s"),
                        color=color,
                        markerfacecolor="none",
                        capsize=3,
                        alpha=0.5,
                    )
                axis.annotate(
                    f"{item['defined_alive_endpoint_seed_pairs']}/{item['configured_seed_pairs']} alive pairs",
                    (item["walkers"], 1 if ratio is None else ratio),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                )
        axis.set_title(
            f"{coupling.capitalize()} coupling · fresh 32 / separate precision 64 seeds"
        )
        axis.set_xlabel("Swarm population N")
        axis.set_ylabel("Alive endpoint / initial mean error")
        axis.axhline(1, color="black", linestyle="--", linewidth=0.8)
        axis.grid(alpha=0.2)
        if groups:
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "No heldout report yet", transform=axis.transAxes, ha="center")
    figure.suptitle(
        "Native alive empirical transport error · all exploratory and heldout cases", fontsize=13
    )
    figure.text(
        0.5,
        -0.012,
        "Stars: exploratory canonical baseline. Diamonds: separate fresh 64-seed precision check. Open markers: incomplete ensemble. × at 1: endpoint unavailable. Error bars: one seed-jackknife SE.",
        ha="center",
        fontsize=8,
    )
    figure.savefig(destination / "calibration-comparison.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def summarize(directory):
    records = []
    sources = []
    errors = []
    ignored = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "calibration-summary.json" or path.name.startswith("main-"):
            continue
        try:
            payload = path.read_bytes()
            report = json.loads(payload)
        except (OSError, ValueError) as error:
            errors.append({"source_file": path.name, "reason": str(error)})
            continue
        if not isinstance(report, dict) or not isinstance(report.get("cases"), list):
            ignored.append({"source_file": path.name, "reason": "not a standalone decay report"})
            continue
        digest = hashlib.sha256(payload).hexdigest()
        sources.append({
            "source_file": path.name,
            "sha256": digest,
            "case_count": len(report["cases"]),
            "config": report.get("config", {}),
        })
        for index, case in enumerate(report["cases"]):
            try:
                records.append(summarize_case(path, digest, report, case, index))
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                errors.append({
                    "source_file": path.name,
                    "source_case_index": index,
                    "reason": str(error),
                })
    summary = {
        "primary_observable": PRIMARY,
        "primary_metric": "Optimal physical q-W2 squared on normalized nonempty alive empirical measures",
        "parameter_selection": "none",
        "source_reports": sources,
        "cases": records,
        "read_errors": errors,
        "ignored_files": ignored,
        "scope": "All available standalone reports are retained. Alive endpoints and fixed-window fits are recomputed from individual native seed histories. Incomplete alive endpoint ensembles and their excluded/terminated pairs remain explicit. Full marked endpoints and native metadata are diagnostic provenance, not substitutes for alive convergence error. Standard errors describe independent seed-pair variation; canonical/calibrated fitted slopes do not certify theorem rates.",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "calibration-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (directory / "calibration-summary.md").write_text(markdown(summary))
    plot(summary, directory)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs/convergence/calibration",
    )
    args = parser.parse_args()
    summary = summarize(args.directory)
    print(
        f"Summarized {len(summary['cases'])} cases from {len(summary['source_reports'])} reports; {len(summary['read_errors'])} read errors."
    )


if __name__ == "__main__":
    main()
