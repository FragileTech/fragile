"""Missing Chapter4 geometry/regional estimates on immutable native states."""

import argparse
import gzip
from itertools import starmap
import json
import math
from pathlib import Path

from complete_chapter04_estimates import (
    centered,
    comparison,
    distance2,
    mean_cloud,
    norm2,
    require,
    sha,
    variance,
    write_json,
)
import numpy as np
from scipy.optimize import linear_sum_assignment


def assignment_transport(left, right):
    require(
        len(left) == len(right) and bool(left), "Balanced nonempty empirical measures required"
    )
    cost = np.array([[distance2(x, y) for y in right] for x in left])
    rows, columns = linear_sum_assignment(cost)
    return float(cost[rows, columns].sum() / len(left))


def phase(positions, velocities):
    return [x + v for x, v in zip(positions, velocities, strict=True)]


def potential_profile(benchmark):
    return (
        (0, 0)
        if benchmark == "constant"
        else (2, 10)
        if benchmark == "rastrigin"
        else (2, 0)
        if benchmark == "sphere"
        else (1, 0)
    )


def analytic_fields(x, benchmark):
    k, a = potential_profile(benchmark)
    force = [-k * z - a * math.tau * math.sin(math.tau * z) for z in x]
    hessian = [k + a * math.tau**2 * math.cos(math.tau * z) for z in x]
    potential = 0.5 * k * norm2(x) + a * math.fsum(1 - math.cos(math.tau * z) for z in x)
    return force, hessian, potential


def regional_checks(points, benchmark, certificate, prefix):
    region = [
        x
        for x in points
        if all(
            lo <= z <= hi
            for lo, z, hi in zip(
                certificate["region_lower"], x, certificate["region_upper"], strict=True
            )
        )
    ]
    comparisons = []
    label = ["prop-w2-regional-cosine-profile"]
    k, a = potential_profile(benchmark)
    for i, x in enumerate(region):
        force, hessian, potential = analytic_fields(x, benchmark)
        comparisons.extend([
            comparison(
                f"{prefix}:{i}:force-sup",
                label,
                math.sqrt(norm2(force)),
                certificate["force_sup_upper"],
            ),
            comparison(
                f"{prefix}:{i}:hessian-lower",
                label,
                min(hessian),
                certificate["curvature_lower"],
                "lower",
            ),
            comparison(
                f"{prefix}:{i}:hessian-upper", label, max(hessian), certificate["curvature_upper"]
            ),
            comparison(
                f"{prefix}:{i}:global-reward-growth",
                label,
                abs(potential),
                certificate["reward_quadratic_growth"] * norm2(x)
                + certificate["reward_constant_growth"],
            ),
            comparison(
                f"{prefix}:{i}:radial-defect",
                label,
                max(
                    0,
                    math.fsum(xx * f for xx, f in zip(x, force, strict=True))
                    + certificate["restoring_k"] * norm2(x),
                ),
                certificate["radial_defect_upper"],
            ),
            comparison(
                f"{prefix}:{i}:bounded-nonquadratic-force",
                label,
                math.sqrt(math.fsum((f + k * z) ** 2 for f, z in zip(force, x, strict=True))),
                abs(a) * math.tau * math.sqrt(len(x)),
            ),
        ])
    for i, x in enumerate(region):
        fx, _, ux = analytic_fields(x, benchmark)
        for j in range(i):
            y = region[j]
            gap = math.sqrt(distance2(x, y))
            fy, _, uy = analytic_fields(y, benchmark)
            comparisons.append(
                comparison(
                    f"{prefix}:{i}:{j}:force-modulus",
                    label,
                    math.sqrt(distance2(fx, fy)),
                    certificate["force_lipschitz"] * gap,
                )
            )
            comparisons.append(
                comparison(
                    f"{prefix}:{i}:{j}:reward-oscillation",
                    label,
                    abs(ux - uy),
                    certificate["reward_oscillation_upper"],
                )
            )
            if gap <= certificate["scale"]:
                defect = max(
                    0,
                    math.fsum(
                        (xx - yy) * (f - g) for xx, yy, f, g in zip(x, y, fx, fy, strict=True)
                    )
                    + certificate["restoring_k"] * gap**2,
                )
                comparisons.append(
                    comparison(
                        f"{prefix}:{i}:{j}:pairwise-defect",
                        label,
                        defect,
                        certificate["pairwise_defect_upper"],
                    )
                )
    return comparisons, len(region)


def run(dataset, tightening_root, output):
    require(not output.exists(), "Use a fresh immutable geometry output directory")
    output.mkdir(parents=True)
    raw = json.loads((dataset / "chapter04-report.json").read_text())
    tightening = json.loads((tightening_root / "report.json").read_text())
    entries = {
        e["path"]: e
        for e in (
            json.loads(s)
            for s in (dataset / "archive-journal.jsonl").read_text().splitlines()
            if s
        )
    }
    comparisons, states = [], []
    for case, sharper in zip(raw["cases"], tightening["cases"], strict=True):
        require(case["case"] == sharper["case"], "Case order mismatch")
        relative = case["operator_frame_archives"][0]
        path = dataset / relative
        require(sha(path) == entries[relative]["sha256"], "Native source frame checksum mismatch")
        frame = json.loads(gzip.decompress(path.read_bytes()))["frames"][0]
        x, y = frame["proposal"]["positions"]
        v, w = frame["proposal"]["velocities"]
        z1, z2 = phase(x, v), phase(y, w)
        euc = assignment_transport(z1, z2)
        structure = assignment_transport(centered(z1), centered(z2))
        location = distance2(mean_cloud(z1), mean_cloud(z2))
        position_struct = assignment_transport(centered(x), centered(y))
        prefix = f"case{case['case']}"
        comparisons.extend([
            comparison(
                f"{prefix}:Euclidean-optimal-barycenter",
                [
                    "lem-wasserstein-barycenter-decomposition",
                    "rem-variance-wasserstein-interpretation",
                ],
                euc,
                structure + location,
                "equal",
                hypotheses="Native full proposal empirical probability measures; independent Euclidean assignment solve, no display-row matching assumption",
            ),
            comparison(
                f"{prefix}:SPD-q-to-Euclidean-structure",
                ["rem-variance-wasserstein-interpretation"],
                frame["centered_q_transport"],
                0.75 * structure,
                "lower",
                hypotheses="Actual fixed q=dx²+dv²+.5 dx.dv has eigenvalue floor .75; both optimal transports computed on the same complete physical proposal",
            ),
            comparison(
                f"{prefix}:phase-to-positional-structure",
                ["rem-variance-wasserstein-interpretation"],
                structure,
                position_struct,
                "lower",
            ),
            comparison(
                f"{prefix}:positional-product-plan",
                ["lem-centered-w2-variance-bound"],
                position_struct,
                variance(x) + variance(y),
            ),
        ])
        for side, cloud in enumerate((x, y)):
            indices = list(range(len(cloud)))
            first = [cloud[i] for i in indices if i % 2 == 0]
            second = [cloud[i] for i in indices if i % 2]
            fraction = len(first) / len(cloud)
            bound = (
                fraction * variance(first)
                + (1 - fraction) * variance(second)
                + fraction * (1 - fraction) * distance2(mean_cloud(first), mean_cloud(second))
            )
            comparisons.append(
                comparison(
                    f"{prefix}:{side}:actual-partition-variance",
                    ["lem-variance-decomposition"],
                    variance(cloud),
                    bound,
                    "equal",
                    hypotheses="Nonempty deterministic geometric comparison partition of actual native proposal; probabilities normalized by own cardinalities",
                )
            )
        input_alive = case["entering_alive"]
        alive_clouds = [
            [p for p, alive in zip(cloud, input_alive, strict=True) if alive]
            for cloud in case["initial_positions"]
        ]
        alive_velocities = [
            [p for p, alive in zip(cloud, input_alive, strict=True) if alive]
            for cloud in case["initial_velocities"]
        ]
        mass = math.fsum(1 / len(alive_clouds[0]) for _ in alive_clouds[0])
        comparisons.append(
            comparison(
                f"{prefix}:own-alive-normalization",
                ["rem-empirical-measures", "def-w2-alive-transport-targets"],
                mass,
                1,
                "equal",
                alive_count=len(alive_clouds[0]),
                allocated_slots=case["N"],
                hypotheses="Actual entering live mask; retained dead physical coordinates excluded from alive measure; 1/k normalization, not 1/N",
            )
        )
        alive1, alive2 = list(starmap(phase, zip(alive_clouds, alive_velocities, strict=True)))
        comparisons.append(
            comparison(
                f"{prefix}:alive-physical-transport-competitor",
                ["def-w2-alive-transport-targets", "prop-w2-prescribed-coupling-scope"],
                assignment_transport(alive1, alive2),
                math.fsum(starmap(distance2, zip(alive1, alive2, strict=True))) / len(alive1),
                hypotheses="Balanced actual alive probability measures of equal recorded live counts; supplied comparison is merely an admissible coupling",
            )
        )
        regional = []
        for name, certificate in sharper["regional_profiles"].items():
            checks, eligible = regional_checks(
                alive_clouds[0] + alive_clouds[1],
                case["benchmark"],
                certificate,
                f"{prefix}:{name}",
            )
            comparisons.extend(checks)
            regional.append({
                "name": name,
                "eligible_points": eligible,
                "all_alive_input_points": 2 * len(alive_clouds[0]),
                "certificate": certificate,
                "scope": "Point/pair diagnostics only inside declared analytic box; all exterior points retained, no residence/mixing probability inferred",
            })
        states.append({
            "case": case["case"],
            "N": case["N"],
            "d": case["d"],
            "source_frame": str(path),
            "source_frame_sha256": entries[relative]["sha256"],
            "repetition": frame["repetition"],
            "Euclidean_full_transport_squared": euc,
            "Euclidean_centered_transport_squared": structure,
            "Euclidean_location_squared": location,
            "regional_checks": regional,
        })
    result = {
        "chapter": 4,
        "new_native_steps": 0,
        "source_report_sha256": sha(dataset / "chapter04-report.json"),
        "tightening_report_sha256": sha(tightening_root / "report.json"),
        "helper_sha256": sha(Path(__file__)),
        "comparisons": comparisons,
        "states": states,
        "summary": {
            "states": len(states),
            "comparisons": len(comparisons),
            "failed": sum(not c["passed"] for c in comparisons),
        },
        "scope": "Independent optimal Euclidean transport and own-alive measure checks, normalized partition identities and analytic landscape profile tests at retained native points. These are finite-input subestimates, not universal theorem proofs or default nonlinear rate closure.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("tightening_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = run(args.dataset, args.tightening_root, args.output)
    print(json.dumps(result["summary"], indent=2))
    raise SystemExit(bool(result["summary"]["failed"]))


if __name__ == "__main__":
    main()
