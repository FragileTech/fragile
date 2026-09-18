"""Independent finite-kernel checks of the signed Keystone cluster-flow bounds.

This is algebraic verification, not a simulation of Euclidean Gas trajectories
and not a substitute for the proofs. The accepted-edge kernel is summed directly
from its clipped acceptance and row-specific normalization; the two lower bounds
are evaluated separately from retained fitness moments.
"""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import random


SEED = 493810
RANDOM_CASES = 10_000
TOLERANCE = 1e-11
F_STAR = 0.1
F_UPPER = 4.41


def accepted_kernel(fitness, weights, p_max, epsilon_clone):
    """Actual alive-recipient accepted-edge probabilities, plus no-clone mass."""
    k = len(fitness)
    normalizers = [math.fsum(row) for row in weights]
    accepted = [[0.0] * k for _ in range(k)]
    stay = [1.0] * k
    for i in range(k):
        if k == 1:
            continue
        for j in range(k):
            if j == i:
                continue
            # This evaluates the actual acceptance before taking any moments.
            probability = min(
                1.0,
                max(0.0, fitness[j] - fitness[i]) / (p_max * (fitness[i] + epsilon_clone)),
            )
            accepted[i][j] = weights[i][j] * probability / normalizers[i]
        stay[i] = 1.0 - math.fsum(accepted[i])
    return accepted, stay, normalizers


def lower_bounds(fitness, weights, normalizers, h_group, l_group, n, kappa, p_max, epsilon_clone):
    """Equations (3.S1) and (3.S2), independently of the accepted kernel."""
    k = len(fitness)
    fh = [fitness[i] for i in h_group]
    fl = [fitness[j] for j in l_group]
    h_mean = math.fsum(fh) / len(fh)
    l_mean = math.fsum(fl) / len(fl)
    delta = l_mean - h_mean
    s_squared = math.fsum((f - h_mean) ** 2 for f in fh) / len(fh)
    s_squared += math.fsum((f - l_mean) ** 2 for f in fl) / len(fl)
    a_star = max(F_UPPER - F_STAR, p_max * (F_UPPER + epsilon_clone))
    c_plus = kappa / a_star
    c_minus = 1.0 / (kappa * p_max * (F_STAR + epsilon_clone))
    assert c_plus <= c_minus
    unweighted = len(fh) * len(fl) / (n * (k - 1)) * (
        c_plus * delta
        - (c_minus - c_plus) / 2.0 * (math.sqrt(delta * delta + s_squared) - delta)
    )

    # Keep the genuine weighted pair covariance inside the variance of the gap.
    w_hl = math.fsum(weights[i][j] for i in h_group for j in l_group)
    delta_w = math.fsum(
        weights[i][j] * (fitness[j] - fitness[i]) for i in h_group for j in l_group
    ) / w_hl
    s_w_squared = math.fsum(
        weights[i][j] * (fitness[j] - fitness[i] - delta_w) ** 2
        for i in h_group for j in l_group
    ) / w_hl
    a_hl_denominator = max(max(0.0, max(fl) - min(fh)), p_max * (max(fh) + epsilon_clone))
    a_hl = 1.0 / (max(normalizers[i] for i in h_group) * a_hl_denominator)
    b_hl = 1.0 / (
        min(normalizers[j] for j in l_group) * p_max * (min(fl) + epsilon_clone)
    )
    weighted = w_hl / n * (
        a_hl * delta_w
        - max(0.0, b_hl - a_hl) / 2.0
        * (math.sqrt(delta_w * delta_w + s_w_squared) - delta_w)
    )
    return unweighted, weighted, delta, b_hl < a_hl


def weights_for(k, kappa, rng=None):
    weights = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i):
            value = rng.uniform(kappa, 1.0) if rng is not None else 1.0
            weights[i][j] = weights[j][i] = value
    return weights


def deterministic_cases():
    for name, fitness, p_max in [
        ("two_tied", [1.0, 1.0], 1.0),
        ("positive_unsaturated", [1.0, 1.1], 1.0),
        ("negative_unsaturated", [1.1, 1.0], 1.0),
        ("positive_saturated", [0.1, 4.41], 0.01),
        ("negative_saturated", [4.41, 0.1], 0.01),
        ("positive_data_coefficient_reversal", [0.1, 4.0], 10.0),
    ]:
        yield name, fitness, weights_for(2, 1.0), [0], [1], 2, 1.0, p_max, 1e-6
    yield (
        "homogeneous_clusters_with_external_donors",
        [0.3, 0.3, 1.7, 1.7, 4.0],
        weights_for(5, 0.1, random.Random(19)),
        [0, 1], [2, 3], 9, 0.1, 1.0, 1e-6,
    )
    yield (
        "all_tied_nonuniform_weights",
        [1.0] * 7,
        weights_for(7, 0.02, random.Random(23)),
        [0, 2, 4], [1, 3, 5], 10, 0.02, 1.0, 1e-6,
    )


def random_cases():
    rng = random.Random(SEED)
    modes = ["overlap", "positive_gap", "negative_gap", "homogeneous", "tied", "near_tie"]
    for case in range(RANDOM_CASES):
        k = rng.randrange(2, 35)
        n = k + rng.randrange(20)
        labels = list(range(k))
        rng.shuffle(labels)
        nh = rng.randrange(1, k)
        nl = rng.randrange(1, k - nh + 1)
        h_group, l_group = labels[:nh], labels[nh:nh + nl]
        kappa = 10.0 ** rng.uniform(-2.0, 0.0)
        p_max = 10.0 ** rng.uniform(-2.0, 1.0)
        epsilon_clone = 10.0 ** rng.uniform(-6.0, -1.0)
        fitness = [rng.uniform(F_STAR, F_UPPER) for _ in range(k)]
        mode = modes[case % len(modes)]
        if mode in {"positive_gap", "negative_gap"}:
            for group, interval in [(h_group, (0.2, 0.8)), (l_group, (2.0, 4.0))]:
                if mode == "negative_gap":
                    interval = (2.0, 4.0) if group is h_group else (0.2, 0.8)
                for i in group:
                    fitness[i] = rng.uniform(*interval)
        elif mode == "homogeneous":
            for group in [h_group, l_group]:
                value = rng.uniform(F_STAR, F_UPPER)
                for i in group:
                    fitness[i] = value
        elif mode == "tied":
            fitness = [1.0] * k
        elif mode == "near_tie":
            fitness = [1.0 + rng.uniform(-1e-9, 1e-9) for _ in range(k)]
        yield (
            f"{mode}:{case}", fitness, weights_for(k, kappa, rng), h_group, l_group,
            n, kappa, p_max, epsilon_clone,
        )


def check():
    minimum_slack = {"3.S1": math.inf, "3.S2": math.inf}
    maximum_violation = {"3.S1": 0.0, "3.S2": 0.0}
    worst_case = {}
    mode_counts = Counter()
    gap_signs = Counter()
    gate_regimes = Counter()
    coefficient_reversals = 0
    maximum_normalization_error = 0.0
    minimum_stay_probability = 1.0
    rows = 0
    fixed_results = []
    cases = list(deterministic_cases())
    fixed_count = len(cases)
    cases.extend(random_cases())
    for name, fitness, weights, h_group, l_group, n, kappa, p_max, epsilon_clone in cases:
        k = len(fitness)
        accepted, stay, normalizers = accepted_kernel(fitness, weights, p_max, epsilon_clone)
        for i in range(k):
            rows += 1
            assert kappa * (k - 1) - TOLERANCE <= normalizers[i] <= k - 1 + TOLERANCE
            donor_sum = math.fsum(weights[i][j] / normalizers[i] for j in range(k) if j != i)
            row_sum = math.fsum([stay[i], *accepted[i]])
            maximum_normalization_error = max(
                maximum_normalization_error, abs(donor_sum - 1.0), abs(row_sum - 1.0),
            )
            minimum_stay_probability = min(minimum_stay_probability, stay[i])
            assert stay[i] >= -TOLERANCE and accepted[i][i] == 0.0
            assert all(0.0 <= value <= 1.0 + TOLERANCE for value in accepted[i])
            for j in range(k):
                if i == j:
                    continue
                raw = (fitness[j] - fitness[i]) / (p_max * (fitness[i] + epsilon_clone))
                regime = "zero" if raw <= 0.0 else "saturated" if raw >= 1.0 else "partial"
                gate_regimes[regime] += 1

        # Independently sum the two directed masses, then subtract. Neither
        # moment bound nor the pairwise bounding inequality enters this sum.
        forward = math.fsum(accepted[i][j] for i in h_group for j in l_group) / n
        reverse = math.fsum(accepted[j][i] for i in h_group for j in l_group) / n
        net = forward - reverse
        s1, s2, delta, reversal = lower_bounds(
            fitness, weights, normalizers, h_group, l_group, n, kappa, p_max, epsilon_clone,
        )
        coefficient_reversals += reversal
        gap_signs["positive" if delta > 0.0 else "negative" if delta < 0.0 else "zero"] += 1
        mode_counts[name.split(":")[0] if ":" in name else "deterministic"] += 1
        for equation, bound in [("3.S1", s1), ("3.S2", s2)]:
            slack = net - bound
            if slack < minimum_slack[equation]:
                minimum_slack[equation] = slack
                worst_case[equation] = name
            maximum_violation[equation] = max(maximum_violation[equation], -slack)
            assert slack >= -TOLERANCE, (name, equation, net, bound)
        if ":" not in name:
            fixed_results.append({"case": name, "net_flow": net, "3.S1": s1, "3.S2": s2})

    # Self-exclusion leaves a singleton with exactly its no-clone outcome.
    singleton, singleton_stay, _ = accepted_kernel([1.0], [[0.0]], 1.0, 1e-6)
    assert singleton == [[0.0]] and singleton_stay == [1.0]
    assert maximum_normalization_error <= TOLERANCE
    assert all(gate_regimes[key] > 0 for key in ["zero", "partial", "saturated"])
    assert coefficient_reversals > 0
    return {
        "kind": "independent_finite_accepted_kernel_algebra",
        "scope": (
            "Algebraic verification of equations (3.S1) and (3.S2), "
            "not gas trajectories or an analytic proof."
        ),
        "random_seed": SEED,
        "random_cases": RANDOM_CASES,
        "deterministic_flux_cases": fixed_count,
        "singleton_persistence_checked": True,
        "alive_population_range": [2, 34],
        "fitness_bounds": {"F_star": F_STAR, "F_upper": F_UPPER},
        "normalization_rows_checked": rows,
        "maximum_kernel_normalization_error": maximum_normalization_error,
        "minimum_no_clone_probability": minimum_stay_probability,
        "minimum_slack": minimum_slack,
        "maximum_lower_bound_violation": maximum_violation,
        "worst_case": worst_case,
        "numerical_tolerance": TOLERANCE,
        "case_modes": dict(mode_counts),
        "fitness_gap_signs": dict(gap_signs),
        "gate_regimes": dict(gate_regimes),
        "data_coefficient_reversals_b_less_than_a": coefficient_reversals,
        "deterministic_results": fixed_results,
    }


if __name__ == "__main__":
    report = check()
    path = Path(__file__).resolve().parents[1] / "validation" / "keystone-flux-identities.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
