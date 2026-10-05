"""Recompute fixed-direction kinetic uncertainty without generating trajectories.

The original archive remains immutable. This review checks the compressed raw
plan hashes and removes fixed between-direction variation from noise uncertainty.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def stratified_estimate(values):
    if len(values) < 6 or not all(math.isfinite(value) for value in values):
        msg = "Need finite values and two replicates per direction"
        raise ValueError(msg)
    variance = (
        sum(
            len(stratum) * statistics.variance(stratum)
            for direction in range(3)
            for stratum in [values[direction::3]]
        )
        / len(values) ** 2
    )
    return statistics.mean(values), math.sqrt(variance)


def comparison(identifier, values, bound):
    mean, se = stratified_estimate(values)
    tolerance = 2e-10 * (1 + abs(mean) + abs(bound))
    return {
        "id": identifier,
        "observed": mean,
        "bound": bound,
        "standard_error": se,
        "original_iid_standard_error": statistics.stdev(values) / math.sqrt(len(values)),
        "relation": "upper",
        "passed": mean - bound <= 6 * se + tolerance,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        msg = "Save the review outside the immutable raw archive root"
        raise ValueError(msg)
    index = json.loads((root / "archive-index.json").read_text())
    cases = []
    for entry in index["entries"]:
        if entry["kind"] != "raw_json" or not entry["tag"].endswith("/coupling_plan"):
            continue
        data = (root / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Compressed plan checksum mismatch: {entry['path']}")
        plan = json.loads(gzip.decompress(data))
        hypotheses = plan["hypotheses"]
        constants = hypotheses["constants"]
        delta = constants["delta"]
        steps = len(plan["trajectories"][0]) - 1
        checks = [comparison("cap_eta_dissipation", plan["cap_eta_residuals"], 0.0)]
        for name, key in [
            ("optimal_Q", "trajectories"),
            ("representative_Q", "representative_coupling_trajectories"),
        ]:
            for step in range(1, steps + 1):
                checks.append(
                    comparison(
                        f"{name}/step{step}",
                        [row[step] for row in plan[key]],
                        (1 - delta) ** step,
                    )
                )
        final, final_se = stratified_estimate([row[-1] for row in plan["trajectories"]])
        cases.append({
            "N": hypotheses["N"],
            "d": hypotheses["d"],
            "independent_replicates": len(plan["trajectories"]),
            "direction_counts": [len(plan["trajectories"][d::3]) for d in range(3)],
            "steps": steps,
            "horizon": steps * hypotheses["h"],
            "raw_plan": str(root / entry["path"]),
            "raw_plan_sha256": entry["sha256"],
            "constants": constants,
            "final_optimal_Q_ratio": final,
            "final_standard_error": final_se,
            "endpoint_squared_error_rate": -math.log(final) / (steps * hypotheses["h"]),
            "guaranteed_squared_error_rate": -math.log1p(-delta) / hypotheses["h"],
            "comparisons": checks,
        })
    if len(cases) != 6:
        raise ValueError(f"Full six-case review requires six saved plans; found {len(cases)}")
    review = {
        "chapter": 5,
        "archive_root": str(root),
        "analysis": (
            "Fixed-direction stratified noise uncertainty; saved means, bounds, "
            "and raw report preserved"
        ),
        "standard_error_formula": "sqrt(sum_h n_h * s_h^2) / m; strata h = replicate_index % 3",
        "scope": (
            "Independent complete replicates within each fixed input-direction stratum; "
            "no trajectory regeneration"
        ),
        "cases": cases,
        "summary": {
            "comparisons": sum(len(case["comparisons"]) for case in cases),
            "comparisons_failed": sum(
                not check["passed"] for case in cases for check in case["comparisons"]
            ),
            "raw_plan_checksums_verified": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2) + "\n")
    print(json.dumps(review["summary"]))


if __name__ == "__main__":
    main()
