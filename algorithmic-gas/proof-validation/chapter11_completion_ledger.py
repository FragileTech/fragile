"""Exact Chapter 11 expression ledger and native finite-horizon law diagnostics.

All entropy values are either closed-form Gaussian reference-law entropy or exact
entropy of explicitly declared finite-grid pushforwards. No atomic/continuous KL
is estimated by a histogram. Whole native trajectories are the resampling units.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import starmap
import json
import math
from operator import itemgetter
from pathlib import Path

import numpy as np


GRID = np.array([-5.0, -3.0, -1.5, -0.75, -0.375, -0.125, 0.125, 0.375, 0.75, 1.5, 3.0, 5.0])
ANALYTIC = {
    2: "Choice of a common dominating measure; invariance is tested by expression 0001.",
    8: "Canonical HK is an infimum over admissible measure-valued curves. Reference Dirac values and explicit reaction/transport actions are checked separately; finite paths do not certify the global infimum.",
    16: "Radon–Nikodym reference-measure symbol, with chain-rule proof in the chapter.",
    21: "Strictly positive mass lower-bound hypothesis; each matched reference fixture records m0.",
    28: "Conditional individual survival probabilities; actual native Gaussian probabilities and operands are retained.",
    31: "Positive Young parameter, checked on each reference mass-drift fixture.",
    34: "Geometric contraction requires R=(1+eta)r²<1, retained explicitly in each mass-drift fixture.",
    37: "Affine-recursion contraction hypothesis R<1; the implemented reference constants satisfy it.",
    41: "Structural affine-drift hypothesis r<1; no native zero-error claim follows when the additive floor is positive.",
    43: "Limit discussion: a positive drift floor alone does not imply convergence to zero.",
    44: "Bounded interior killing hypothesis 0<=c<=C; not boundary-absorption bounded killing.",
    45: "Positive continuous-model revival rate, declared on each continuous reference fixture.",
    48: "Integrating-factor proof device; its value is retained in every continuous mass fixture.",
    49: "Safe-walker death-probability hypothesis q<1, recorded in each exact finite survival fixture.",
    51: "Mass concentration domain 0<b<p0, enforced in reference and native operands.",
    52: "Conditional average-survival lower bound; each native post-B2 state has its own value, not a pathwise global mass lower bound.",
    55: "Unconditional bad-event probability hypothesis in a finite joint event law.",
    56: "Finite-time survival denominator s>0 in the retained joint event law.",
    61: "Normalized evolving law symbol; reference laws and empirical grid laws remain distinct.",
    65: "Radon–Nikodym density ratio; no global upper bound is required by the entropy conversion.",
    66: "Reference alive measure m_*q; exact masses and Gaussian shape parameters are recorded.",
    67: "Evolving alive measure m_t p_t; exact masses and Gaussian shape parameters are recorded.",
    68: "Joint mass interval hypothesis [m0,m1]; the Gaussian mass fixtures retain both endpoints.",
    74: "Finite-particle law conditioned on finite-time survival. Native realization histograms do not identify this full configuration-space law or its QSD.",
    78: "Positive QSD killing-rate hypothesis; infinite survival has zero probability, as proved in the chapter.",
    82: "Initial reference domination hypothesis, instantiated by all-subset finite-measure order checks.",
    85: "Positive finite-time survival mass required for normalization; every killed matrix iterate records the denominator.",
    88: "Initial lower reference domination hypothesis, instantiated by the minimum coordinate ratio.",
    89: "Positive lower domination coefficient, retained on each killed-kernel fixture.",
    99: "Positive survival denominator for the upper output-density bound; this is a kernel hypothesis, not an empirical density assumption.",
    103: "Positive mass of the Gaussian contribution; the reference mixture uses exactly unit total source mass.",
    114: "Taylor integration parameter interval [0,1]. The exact quadratic remainder is evaluated algebraically.",
    126: "Kernel derivative contribution requires its own K1 bound; it is not silently included in the fixed-kernel operator norm.",
    127: "Infinite Sobolev-order regularity implication is an explicit analytic hypothesis. Finite bracket arithmetic does not certify it globally.",
    128: "Positive local Sobolev derivative gain, required at every bootstrap iteration.",
    129: "Distributional equation Pu=Bu together with lower-order regularity hypotheses; no finite experiment certifies all local Sobolev classes.",
    130: "Starting Sobolev class in the analytic regularity bootstrap.",
    133: "Fixed smoothing delay t>=delta, checked in each finite reference fixture.",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_archive(root, entry):
    path = root / entry["path"]
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Archive checksum mismatch: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        return json.load(file)


def h2(p, q):
    return float(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2))


def w2(x, p, y, q):
    a = sorted((float(u), float(v)) for u, v in zip(x, p, strict=True) if v > 0)
    b = sorted((float(u), float(v)) for u, v in zip(y, q, strict=True) if v > 0)
    i = j = 0
    answer = 0.0
    while i < len(a) and j < len(b):
        take = min(a[i][1], b[j][1])
        answer += take * (a[i][0] - b[j][0]) ** 2
        a[i] = (a[i][0], a[i][1] - take)
        b[j] = (b[j][0], b[j][1] - take)
        if a[i][1] <= 0:
            i += 1
        if b[j][1] <= 0:
            j += 1
    return answer


def metrics(p, q):
    m, n = float(sum(p)), float(sum(q))
    result = {
        "root_mass_squared": (np.sqrt(m) - np.sqrt(n)) ** 2,
        "mass_difference_squared": (m - n) ** 2,
        "grid_H_squared": h2(p, q),
    }
    if min(m, n) <= 0:
        return result
    a, b = p / m, q / n
    shape = h2(a, b)
    transport = w2(GRID, a, GRID, b)
    result.update(
        grid_shape_H_squared=shape,
        grid_W2_squared=transport,
        grid_D_squared=result["grid_H_squared"] + transport,
    )
    return result


def packed(path, value):
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    with gzip.open(path, "wb") as file:
        file.write(raw)
    with gzip.open(path, "rb") as file:
        if file.read() != raw:
            message = "Deep archive verification failed"
            raise ValueError(message)
    return {
        "path": str(path),
        "sha256": sha(path),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "decoded_bytes": len(raw),
        "compressed_bytes": path.stat().st_size,
    }


def diagnostic_curves(groups, output, repetitions):
    rng = np.random.default_rng(11190315)
    curves = []
    archive_manifest = []
    pairs = []
    for key in sorted(groups):
        if not key.endswith("/left"):
            continue
        right = key[:-4] + "right"
        if right in groups:
            pairs.append((key, right))
    for left_key, right_key in pairs:
        left = sorted(groups[left_key], key=itemgetter("seed"))
        right = sorted(groups[right_key], key=itemgetter("seed"))
        if len(left) < 4 or len(right) < 4:
            message = "Independent whole-trajectory uncertainty requires >=4 seeds"
            raise ValueError(message)
        common_seed = [r["seed"] for r in left] == [r["seed"] for r in right]
        maximum_steps = max(len(r["observations"]) for r in left + right)
        dt = left[0]["h"]
        if any(r["h"] != dt for r in left + right):
            message = "Paired laws use different physical timesteps"
            raise ValueError(message)
        left_times = [j * dt for j in range(maximum_steps + 1)]

        def arrays(rows):
            result = []
            for row in rows:
                values = [row["initial"]["bins"]] + [
                    o["observation"]["bins"] for o in row["observations"]
                ]
                if len(values) < maximum_steps + 1:
                    if row["terminal"]["mass"] != 0:
                        message = (
                            "Only extinct native trajectories admit exact cemetery continuation"
                        )
                        raise ValueError(message)
                    values += [[0.0] * len(GRID)] * (maximum_steps + 1 - len(values))
                for j, observation in enumerate(row["observations"], start=1):
                    if abs(observation["physical_time"] - j * dt) > 1e-10:
                        message = "Native physical time does not start at zero or is discontinuous"
                        raise ValueError(message)
                result.append(values)
            return np.array(result, dtype=float)

        a, b = arrays(left), arrays(right)
        ia = rng.integers(0, len(left), (repetitions, len(left)))
        ib = ia.copy() if common_seed else rng.integers(0, len(right), (repetitions, len(right)))
        point = list(starmap(metrics, zip(a.mean(0), b.mean(0), strict=True)))
        names = sorted(set.intersection(*(set(v) for v in point)))
        samples = {name: np.empty((repetitions, len(left_times))) for name in names}
        for replicate in range(repetitions):
            sample_a, sample_b = a[ia[replicate]].mean(0), b[ib[replicate]].mean(0)
            for time, (x, y) in enumerate(zip(sample_a, sample_b, strict=True)):
                row = metrics(x, y)
                for name in names:
                    samples[name][replicate, time] = row.get(name, np.nan)
        results = []
        for name in names:
            values = np.array([v[name] for v in point])
            finite = np.isfinite(samples[name]).all(axis=1)
            draw = samples[name][finite]
            intervals = np.quantile(draw, [0.025, 0.975], axis=0).T.tolist() if len(draw) else None
            positive = values > 1e-14
            rate = None
            rate_interval = None
            fit = np.flatnonzero(positive & (np.arange(len(values)) > 0))
            if len(fit) >= 4:
                physical = np.array(left_times)[fit]
                rate = -float(np.polyfit(physical, np.log(values[fit]), 1)[0])
                accepted = draw[:, fit]
                accepted = accepted[np.all(accepted > 1e-14, axis=1)]
                if len(accepted):
                    centered = physical - np.mean(physical)
                    rates = -(np.log(accepted) @ centered) / (centered @ centered)
                    rate_interval = np.quantile(rates, [0.025, 0.975]).tolist()
            endpoint_samples = draw[:, -1] - draw[:, 0] if len(draw) else np.array([])
            results.append({
                "metric": name,
                "values": values.tolist(),
                "pointwise_bootstrap_95_intervals": intervals,
                "endpoint_decreased": bool(values[-1] < values[0]) if values[0] > 1e-14 else None,
                "endpoint_change_95_interval": np.quantile(
                    endpoint_samples, [0.025, 0.975]
                ).tolist()
                if len(endpoint_samples)
                else None,
                "physical_time_log_linear_decay_rate": rate,
                "decay_rate_95_interval": rate_interval,
                "rate_fit_times": np.array(left_times)[fit].tolist(),
                "finite_bootstrap_replicates": int(sum(finite)),
                "theorem_prediction": None,
                "scope": "Finite-horizon native law diagnostic; sign and rate depend on parameters. The reference is the independently seeded other initial-law ensemble, not a QSD or fitted stationary law.",
            })
        # Exact unquantized projected transport for each independent trajectory pair.
        # Sorting is only a monotone transport calculation; no walker labels enter.
        # This cost concerns the two realized normalized alive measures, not KL of
        # empirical atoms relative to an absolutely continuous reference law.
        paired_costs = np.full((min(len(left), len(right)), len(left_times)), np.nan)
        for pair, (left_run, right_run) in enumerate(zip(left, right, strict=False)):
            left_observations = [left_run["initial"]] + [
                o["observation"] for o in left_run["observations"]
            ]
            right_observations = [right_run["initial"]] + [
                o["observation"] for o in right_run["observations"]
            ]
            for time in range(min(len(left_observations), len(right_observations))):
                x = left_observations[time].get("projected_coordinates", [])
                y = right_observations[time].get("projected_coordinates", [])
                if x and y:
                    paired_costs[pair, time] = w2(
                        x, np.full(len(x), 1 / len(x)), y, np.full(len(y), 1 / len(y))
                    )
        counts = np.sum(np.isfinite(paired_costs), axis=0)
        values = np.divide(
            np.nansum(paired_costs, axis=0),
            counts,
            out=np.full(len(left_times), np.nan),
            where=counts > 0,
        )
        pair_indices = rng.integers(0, len(paired_costs), (repetitions, len(paired_costs)))
        chosen = paired_costs[pair_indices]
        chosen_counts = np.sum(np.isfinite(chosen), axis=1)
        transport_draws = np.divide(
            np.nansum(chosen, axis=1),
            chosen_counts,
            out=np.full((repetitions, len(left_times)), np.nan),
            where=chosen_counts > 0,
        )
        finite = np.isfinite(transport_draws).all(axis=1)
        valid_draws = transport_draws[finite]
        fit = np.flatnonzero(np.isfinite(values) & (values > 1e-14) & (np.arange(len(values)) > 0))
        rate = None
        rate_interval = None
        if len(fit) >= 4:
            physical = np.array(left_times)[fit]
            centered = physical - physical.mean()
            rate = -float(np.polyfit(physical, np.log(values[fit]), 1)[0])
            accepted = valid_draws[:, fit]
            accepted = accepted[np.all(accepted > 1e-14, axis=1)]
            if len(accepted):
                rates = -(np.log(accepted) @ centered) / (centered @ centered)
                rate_interval = np.quantile(rates, [0.025, 0.975]).tolist()
        name = "paired_empirical_projected_W2_squared"
        samples[name] = transport_draws
        if np.isfinite(values).all():
            results.append({
                "metric": name,
                "values": values.tolist(),
                "positive_mass_trajectory_pairs_by_time": counts.tolist(),
                "pointwise_bootstrap_95_intervals": np.quantile(
                    valid_draws, [0.025, 0.975], axis=0
                ).T.tolist()
                if len(valid_draws)
                else None,
                "endpoint_decreased": bool(values[-1] < values[0]) if values[0] > 1e-14 else None,
                "endpoint_change_95_interval": np.quantile(
                    valid_draws[:, -1] - valid_draws[:, 0], [0.025, 0.975]
                ).tolist()
                if len(valid_draws)
                else None,
                "physical_time_log_linear_decay_rate": rate,
                "decay_rate_95_interval": rate_interval,
                "rate_fit_times": np.array(left_times)[fit].tolist(),
                "finite_bootstrap_replicates": int(sum(finite)),
                "theorem_prediction": None,
                "scope": "Exact monotone transport between each pair's unquantized first-coordinate normalized alive empirical measures, averaged over independent trajectory pairs. Missing shapes after extinction remain excluded with their counts and conditioning visible. This measures realized swarm transport, not full-law KL or QSD attraction.",
            })
        saved = {
            "left_group": left_key,
            "right_group": right_key,
            "exact_projected_transport_pairs": [
                [None if not np.isfinite(x) else float(x) for x in row] for row in paired_costs
            ],
            "exact_transport_whole_pair_resampling_indices": pair_indices.tolist(),
            "left_seeds": [r["seed"] for r in left],
            "right_seeds": [r["seed"] for r in right],
            "left_first_killing_transition_steps": [
                r.get("first_killing_transition_step") for r in left
            ],
            "right_first_killing_transition_steps": [
                r.get("first_killing_transition_step") for r in right
            ],
            "whole_trajectory_left_resampling_indices": ia.tolist(),
            "whole_trajectory_right_resampling_indices": ib.tolist(),
            "common_intrinsic_stream_pair_resampling": common_seed,
            "left_all_time_grid_submeasures": a.tolist(),
            "right_all_time_grid_submeasures": b.tolist(),
            "physical_times": left_times,
            "grid_nodes": GRID.tolist(),
            "bootstrap_curves": {
                name: [[None if not np.isfinite(x) else float(x) for x in row] for row in array]
                for name, array in samples.items()
            },
            "results": results,
        }
        manifest = packed(output / f"native-curve-{len(curves):03}.json.gz", saved)
        archive_manifest.append(manifest)
        curves.append({
            "left_group": left_key,
            "right_group": right_key,
            "N": left[0]["N"],
            "d": left[0]["d"],
            "left_runs": len(left),
            "right_runs": len(right),
            "common_intrinsic_seed_stream": common_seed,
            "physical_times": left_times,
            "metrics": results,
            "archive": manifest,
        })
    return curves, archive_manifest


def survival_calibration(conditional, groups, output, repetitions, primary_root, primary_horizon):
    rng = np.random.default_rng(11190316)
    batches = defaultdict(list)
    run_lookup = {(r["group"], r["seed"]): r for runs in groups.values() for r in runs}
    for trajectory in conditional:
        per_condition = defaultdict(list)
        for frame in trajectory["frames"]:
            conditioning = (
                "whole_prepared_kinetics"
                if "B1_input" in frame["conditioning"]
                else "post_B2_final_Gaussian"
            )
            per_condition[conditioning].append(frame)
        for conditioning, frames in per_condition.items():
            run = run_lookup[trajectory["group"], trajectory["seed"]]
            horizon = run["declared_planned_horizon"]
            if horizon is None:
                message = "A deterministic planned horizon is required for conditional Hoeffding"
                raise ValueError(message)
            if len(run["observations"]) > horizon or len(frames) > horizon:
                message = "Native recorded horizon exceeds the declared deterministic design"
                raise ValueError(message)
            batches[trajectory["group"], conditioning].append({
                "seed": trajectory["seed"],
                "N": run["N"],
                "d": run["d"],
                "planned_horizon": horizon,
                "frames": frames,
                "M": sum(f["centered_mass_innovation"] for f in frames),
                "V": sum(f["conditional_variance"] for f in frames),
                "first_killing_transition_step": run["first_killing_transition_step"],
            })
    family = len(batches)
    delta = 0.01
    results = []
    archive_manifest = []
    for (group, conditioning), runs in sorted(batches.items()):
        count = len(runs)
        if len({r["seed"] for r in runs}) != count:
            message = "Duplicate seed within conditional survival calibration law"
            raise ValueError(message)
        n, d = runs[0]["N"], runs[0]["d"]
        horizon = max(r["planned_horizon"] for r in runs)
        measured = np.array([r["M"] for r in runs])
        predicted = np.array([r["V"] for r in runs])
        indices = rng.integers(0, count, (repetitions, count))
        draw_mean = measured[indices].mean(1)
        draw_m2 = (measured[indices] ** 2).mean(1)
        draw_v = predicted[indices].mean(1)
        draw_gap = draw_m2 - draw_v
        mean_bound = math.sqrt(horizon * math.log(6 * family / delta) / (2 * n * count))
        cdf_mean_budget = horizon * d * 1.5e-7
        gap_range = horizon**2 + horizon / (4 * n)
        gap_bound = gap_range * math.sqrt(math.log(6 * family / delta) / (2 * count))
        cdf_gap_budget = 2 * horizon**2 * d * 1.5e-7 + horizon * d * 1.5e-7 / n
        # Fixed-grid exponential supermartingale, with observed predictable variance.
        # Unlike plugging random V into a fixed-v Freedman statement, this evaluates
        # the actual test supermartingale and pays for the predeclared lambda grid.
        lambda_grid = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
        cdf_total_variance_budget = count * horizon * d * 1.5e-7 / n
        variance_total_upper = float(predicted.sum()) + cdf_total_variance_budget
        variance_sensitive_mean_bound = (
            min(
                (
                    math.log(6 * family * len(lambda_grid) / delta)
                    + n**2 * (math.expm1(lam / n) - lam / n) * variance_total_upper
                )
                / (lam * count)
                for lam in lambda_grid
            )
            + cdf_mean_budget
        )
        mean = float(measured.mean())
        m2, bracket = float(np.mean(measured**2)), float(predicted.mean())
        result = {
            "group": group,
            "conditioning": conditioning,
            "N": n,
            "d": d,
            "independent_trajectories": count,
            "deterministic_planned_horizon": horizon,
            "actual_mean_innovation_sum": mean,
            "predicted_mean_innovation_sum": 0.0,
            "mean_bootstrap_95_interval": np.quantile(draw_mean, [0.025, 0.975]).tolist(),
            "actual_mean_squared_innovation_sum": m2,
            "predicted_mean_quadratic_variation": bracket,
            "actual_to_predicted_second_moment_ratio": m2 / bracket if bracket > 0 else None,
            "second_moment_minus_bracket": m2 - bracket,
            "second_moment_minus_bracket_bootstrap_95_interval": np.quantile(
                draw_gap, [0.025, 0.975]
            ).tolist(),
            "familywise_error_budget": delta,
            "family_size": family,
            "conditional_Hoeffding_mean_allowance": mean_bound + cdf_mean_budget,
            "conditional_Hoeffding_passed": abs(mean) <= mean_bound + cdf_mean_budget,
            "variance_sensitive_mean_allowance": variance_sensitive_mean_bound,
            "variance_sensitive_mean_passed": abs(mean) <= variance_sensitive_mean_bound,
            "predeclared_lambda_grid": lambda_grid,
            "second_moment_distribution_free_allowance": gap_bound + cdf_gap_budget,
            "second_moment_distribution_free_passed": abs(m2 - bracket)
            <= gap_bound + cdf_gap_budget,
            "scope": "Martingale increments across stages/times; independent whole trajectories are the Monte Carlo units. Both conditioning laws remain separate. Extinction stops future increments at zero; the probability bound uses the deterministic designed horizon, not the observed stopping time.",
        }
        saved = {
            "result": result,
            "per_trajectory_M_and_V": [
                {k: v for k, v in r.items() if k != "frames"} for r in runs
            ],
            "retained_raw_per_frame_values": [
                {
                    "seed": r["seed"],
                    "frames": [
                        {
                            k: f.get(k)
                            for k in [
                                "step",
                                "N",
                                "d",
                                "conditional_mean",
                                "conditional_variance",
                                "actual_alive_fraction",
                                "centered_mass_innovation",
                                "conditioning",
                            ]
                        }
                        for f in r["frames"]
                    ],
                }
                for r in runs
            ],
            "whole_trajectory_resampling_indices": indices.tolist(),
            "bootstrap_mean_innovation_sums": draw_mean.tolist(),
            "bootstrap_mean_squared_innovation_sums": draw_m2.tolist(),
            "bootstrap_mean_quadratic_variations": draw_v.tolist(),
            "bootstrap_second_moment_minus_bracket": draw_gap.tolist(),
            "conditional_log_MGF_bound_per_step": "lambda²/(8N) after conditioning on the indicated prepared state",
            "CDF_bias_budget_per_step": d * 1.5e-7,
            "variance_sensitive_derivation": "For Y=I-p, E Y=0, |Y|<=1 and |E Y^k|<=E Y²=p(1-p), k>=2. Taylor expansion gives log E exp(tY)<=p(1-p)(exp|t|-1-|t|). The N independent row terms scaled by 1/N give N² V psi(|lambda|/N). Conditional iteration, stopping and independent trajectories preserve the resulting exponential supermartingale. Markov plus two signs, the predeclared lambda grid, all law/conditioning groups, and the three separately split families gives the displayed data-dependent predictable-variance bound; CDF mean/variance approximation allowances are added explicitly.",
        }
        manifest = packed(output / f"survival-calibration-{len(results):03}.json.gz", saved)
        archive_manifest.append(manifest)
        result["archive"] = manifest
        results.append(result)
    (output / "native-survival-calibration.json").write_text(
        json.dumps({"rows": results, "manifest": archive_manifest}, indent=2) + "\n"
    )
    return results, archive_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--primary-native-horizon", type=int)
    parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "native-survival-derivation.md").write_bytes(
        Path(__file__).with_name("chapter11-native-survival.md").read_bytes()
    )
    root = args.dataset.resolve()
    index = json.loads((root / "archive-index.json").read_text())
    if index["status"] != "complete":
        message = "Complete immutable output required"
        raise ValueError(message)
    entries = {e["tag"]: e for e in index["entries"]}
    source = load_archive(root, entries["source"])
    inventory = json.loads(source["inventory"])
    current = Path(__file__).resolve().parent / "chapter11_inventory.json"
    if json.loads(current.read_text()) != inventory:
        message = "Source inventory changed since native execution"
        raise ValueError(message)
    checks = load_archive(root, entries["reference-operands"])["checks"]
    if any(not c["passed"] for c in checks):
        message = "Reference estimate failed"
        raise ValueError(message)
    by_expression = defaultdict(list)
    for check in checks:
        for expression in check["expression_ids"]:
            by_expression[expression].append(check["id"])
    rows = []
    missing = []
    for expression in inventory["quantitative_expressions"]:
        number = int(expression["id"].rsplit("-", 1)[1])
        evidence = by_expression[expression["id"]]
        row = dict(expression)
        if evidence:
            row.update(
                status="matched_numeric_reference",
                numeric_checks=len(evidence),
                evidence_check_ids=evidence,
                disposition="Each cited check records this exact expression ID and its own finite-measure/kernel/matrix/Gaussian operands. Reference model scope remains distinct from the native full operator.",
            )
        elif number in ANALYTIC:
            row.update(
                status="analytic_definition_hypothesis_or_limit",
                numeric_checks=0,
                evidence_check_ids=[],
                disposition=ANALYTIC[number],
            )
        else:
            missing.append(number)
            row.update(
                status="finite_operand_binding_pending",
                numeric_checks=0,
                evidence_check_ids=[],
                disposition="Finite source expression requires its own matched numeric operands.",
            )
        rows.append(row)
    groups = defaultdict(list)
    conditional = []
    native_dataset_roots = [root] + [p.resolve() for p in args.additional_dataset]
    verified_input_archives = 0
    input_indices = []
    declared_horizon_cache = {}
    for native_root in native_dataset_roots:
        native_index = json.loads((native_root / "archive-index.json").read_text())
        if native_index["status"] != "complete":
            message = "Additional native operand archive must be complete"
            raise ValueError(message)
        input_indices.append({
            "root": str(native_root),
            "index_sha256": sha(native_root / "archive-index.json"),
        })
        for entry in native_index["entries"]:
            tag = entry["tag"]
            # Every JSON artifact is verified, including the immutable source snapshots.
            row = load_archive(native_root, entry)
            verified_input_archives += 1
            if tag.startswith("native-"):
                row["derived_execution_root"] = str(native_root)
                # Retained designs are exactly 1 or 16 steps; every retained run is complete.
                # Fresh stopped trajectories use the explicit command-declared 32-step cap.
                input_dataset = Path(row["dataset"])
                if not input_dataset.is_absolute():
                    input_dataset = Path(__file__).resolve().parent.parent / input_dataset
                if str(input_dataset) not in declared_horizon_cache:
                    source_report = input_dataset / "report.json"
                    source_data = (
                        json.loads(source_report.read_text()) if source_report.exists() else {}
                    )
                    declared_horizon_cache[str(input_dataset)] = source_data.get(
                        "summary", {}
                    ).get("requested_steps")
                declared = declared_horizon_cache[str(input_dataset)]
                if declared is None:
                    if any(o["observation"]["mass"] == 0 for o in row["observations"]):
                        declared = args.primary_native_horizon
                    else:
                        declared = len(row["observations"])
                row["declared_planned_horizon"] = declared
                row["first_killing_transition_step"] = next(
                    (o["step"] for o in row["observations"] if o["observation"]["mass"] == 0), None
                )
                groups[row["group"]].append(row)
            elif tag.startswith("conditional-survival-"):
                conditional.append(row)
    for group in groups.values():
        if len({r["seed"] for r in group}) != len(group):
            message = "Repeated native trajectory seed in one law"
            raise ValueError(message)
    extinction_rows = [
        {
            "group": r["group"],
            "seed": r["seed"],
            "N": r["N"],
            "d": r["d"],
            "first_killing_transition_step": r["first_killing_transition_step"],
            "terminal_alive_fraction": r["terminal"]["mass"],
            "retained_source_path": r["source_path"],
            "retained_source_SHA256": r["source_sha256"],
        }
        for runs in groups.values()
        for r in runs
    ]
    (args.output / "physical-extinction-steps.json").write_text(
        json.dumps(extinction_rows, indent=2) + "\n"
    )
    curves, archive_manifest = diagnostic_curves(groups, args.output, args.bootstrap)
    calibration, calibration_manifest = survival_calibration(
        conditional, groups, args.output, args.bootstrap, root, args.primary_native_horizon
    )
    archive_manifest += calibration_manifest
    ledger = {
        "chapter": 11,
        "source_inventory_sha256": sha(current),
        "execution_index_sha256": sha(root / "archive-index.json"),
        "helper_source_SHA256": sha(args.output / "source.py"),
        "native_survival_derivation_SHA256": sha(args.output / "native-survival-derivation.md"),
        "rows": rows,
        "counts": {
            "expressions": len(rows),
            "matched_numeric_expressions": sum(
                r["status"] == "matched_numeric_reference" for r in rows
            ),
            "analytic_expressions": sum(
                r["status"] == "analytic_definition_hypothesis_or_limit" for r in rows
            ),
            "pending_finite_expressions": missing,
            "reference_checks": len(checks),
        },
        "native_curve_groups": len(curves),
        "bootstrap_replicates": args.bootstrap,
    }
    (args.output / "expression-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (args.output / "native-curves.json").write_text(
        json.dumps({"curves": curves, "manifest": archive_manifest}, indent=2) + "\n"
    )
    (args.output / "statement-ledger.json").write_text(
        json.dumps(
            {
                "chapter": 11,
                "formal_items": [
                    {
                        **item,
                        "expression_ids": [
                            r["id"] for r in rows if r.get("formal_item_id") == item["id"]
                        ],
                        "scope": "Exact local hypotheses and original main-chapter proof retained; numerical checks certify the declared reference/native operands, not a global QSD, LSI or infinite-time regularity assumption.",
                    }
                    for item in inventory["formal_items"]
                ],
            },
            indent=2,
        )
        + "\n"
    )
    body = [
        "# Chapter 11: complete constants and finite-measure validation",
        "",
        "Hellinger and transport native measurements use explicitly declared permutation-invariant probability pushforwards. All full configurations, seeds, masks and stage inputs remain in immutable upstream Rust archives. Gaussian reference entropies use exact continuous-law formulas. Native time curves compare two independently seeded initial-law ensembles on the same finite quantization; their uncertainties resample whole trajectories, preserving all time dependence and common-stream pairs.",
        "",
        "| Expression | Source | Formula | Validation | Checks |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        formula = row["formula"].replace("|", "\\|").replace("\n", " ")
        body.append(
            f"| {row['id']} | {row['source_label']}:{row['source_line']} | `{formula}` | {row['status']} | {row['numeric_checks']} |"
        )
    body += [
        "",
        "## Measured finite-horizon native mass, shape and transport rates",
        "",
        "These are empirical rates of the declared grid pushforward laws. They retain sampling uncertainty and finite population floors. A positive slope estimate does not supply the full-law entropy or LSI hypotheses. A nondecreasing metric is recorded as a parameter-dependent finite diagnostic, never changed into a theorem violation or fitted away.",
        "",
        "| Native pair | d | N | Metric | Initial | Final | Decreases | Physical decay rate [95% interval] |",
        "|---|---:|---:|---|---:|---:|---|---|",
    ]
    for curve in curves:
        for metric in curve["metrics"]:
            rate = metric["physical_time_log_linear_decay_rate"]
            interval = metric["decay_rate_95_interval"]
            body.append(
                f"| {curve['left_group']} | {curve['d']} | {curve['N']} | {metric['metric']} | {metric['values'][0]:.6g} | {metric['values'][-1]:.6g} | {metric['endpoint_decreased']} | {rate} {interval} |"
            )
    body += [
        "",
        "## Actual survival mean and variance calibration",
        "",
        "Predicted E[M_T]=0 and E[M_T²]=E[sum conditional variance] are compared with actual mass innovations. Stopped extinction increments are zero. The uncertainty and distribution-free bounds resample or bound complete independent trajectories; neither walkers nor time steps are pooled as independent replicates.",
        "",
        "| Native law | Conditioning | Runs | Mean M | E M² measured | E quadratic variation predicted | Ratio | Difference [95% bootstrap interval] | Conditional Hoeffding |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in calibration:
        body.append(
            f"| {row['group']} | {row['conditioning']} | {row['independent_trajectories']} | {row['actual_mean_innovation_sum']:.6g} | {row['actual_mean_squared_innovation_sum']:.6g} | {row['predicted_mean_quadratic_variation']:.6g} | {row['actual_to_predicted_second_moment_ratio']} | {row['second_moment_minus_bracket_bootstrap_95_interval']} | {row['conditional_Hoeffding_passed']} |"
        )
    (args.output / "results.md").write_text("\n".join(body) + "\n")
    verification = {
        "indexed_input_archives": verified_input_archives,
        "input_index_SHA256": input_indices,
        "input_checksums_verified": True,
        "derived_archives": len(archive_manifest),
        "derived_compressed_bytes": sum(x["compressed_bytes"] for x in archive_manifest),
        "decoded_SHA256_verified": True,
        "missing_finite_bindings": missing,
        "native_survival_trajectories": len(conditional),
        "native_survival_frames": sum(len(x["frames"]) for x in conditional),
        "reference_numeric_checks": len(checks),
        "native_survival_calibration_groups": len(calibration),
        "native_survival_distribution_free_checks": 3 * len(calibration),
        "native_survival_distribution_free_failures": sum(
            not r["conditional_Hoeffding_passed"] for r in calibration
        )
        + sum(not r["second_moment_distribution_free_passed"] for r in calibration)
        + sum(not r["variance_sensitive_mean_passed"] for r in calibration),
    }
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(
        json.dumps({
            "ledger_counts": ledger["counts"],
            "curves": len(curves),
            "verification": verification,
        })
    )
    if missing:
        message = "Unbound finite expressions remain; close them before declaring completion"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
