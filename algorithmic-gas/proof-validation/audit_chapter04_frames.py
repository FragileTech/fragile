"""Independently audit retained Chapter 4 operator frames without engine updates."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def mean_cloud(points):
    return [sum(p[a] for p in points) / len(points) for a in range(len(points[0]))]


def variance(points):
    center = mean_cloud(points)
    return sum(sum((x - y) ** 2 for x, y in zip(p, center)) for p in points) / len(points)


def kernel_rows(x, v):
    # Exact archived Euclidean provider: radius-two radial squash, lambda-one
    # phase-space distance, width-two Gaussian, no self donors, current frame.
    features = []
    for position, velocity in zip(x, v):
        p = [z / (1 + math.sqrt(sum(a * a for a in position)) / 2) for z in position]
        w = [z / (1 + math.sqrt(sum(a * a for a in velocity)) / 2) for z in velocity]
        features.append(p + w)
    rows = []
    for i, a in enumerate(features):
        weights = [
            0.0 if i == j else math.exp(-sum((x - y) ** 2 for x, y in zip(a, b)) / 8)
            for j, b in enumerate(features)
        ]
        total = sum(weights)
        rows.append([w / total for w in weights])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if json.loads((args.dataset / "config.json").read_text())["compact"]:
        msg = (
            "This audit uses the recorded full-matrix case ordering; compact datasets are excluded"
        )
        raise ValueError(msg)
    # Read one durable journal snapshot; no file from an unfinished chunk is used.
    journal = [
        json.loads(s)
        for s in (args.dataset / "archive-journal.jsonl").read_text().splitlines()
        if s.strip()
    ]
    entries = [e for e in journal if re.search(r"chapter04-case\d+-frames-", e["tag"])]
    cases = {}
    profiles = ["canonical"] * 4 + ["selection", "revival", "singleton", "permutation"]
    landscapes = ["Quadratic", "Sphere", "Rastrigin", "Constant"]
    checks = {}
    constant_keys = [
        "R_star",
        "Delta_V",
        "s_star_squared",
        "f_H",
        "f_L",
        "a_star",
        "B_acc",
        "p_u",
        "c_H",
        "a_x",
        "b_x",
        "M_j",
        "B_T",
        "c_err",
        "g_err",
        "chi",
        "g_max",
        "R_V",
        "Q",
        "Q_lower",
        "target_error",
        "target_error_lower",
    ]

    def check(name, residual):
        if not math.isfinite(residual):
            raise ValueError(f"Nonfinite residual for {name}")
        row = checks.setdefault(name, {"comparisons": 0, "maximum_positive_residual": 0.0})
        row["comparisons"] += 1
        row["maximum_positive_residual"] = max(row["maximum_positive_residual"], residual)

    for entry in entries:
        data = (args.dataset / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {entry['path']}")
        content = json.loads(gzip.decompress(data))
        case_id = int(re.search(r"case(\d+)", entry["tag"])[1])
        profile = profiles[case_id % 8]
        case = cases.setdefault(
            case_id,
            {
                "case": case_id,
                "profile": profile,
                "landscape": landscapes[case_id % 8] if case_id % 8 < 4 else "Quadratic",
                "frames": 0,
                "target_applicable": 0,
                "target_positive": 0,
                "target_excluded": 0,
                "all_alive": True,
                "C_reset": None,
                "proxy": [],
                "centered_error": [],
                "proxy_residual": [],
                "target_constants": {k: [] for k in constant_keys},
            },
        )
        first = content["frames"][0]
        all_alive = all(first["before"]["alive"][0])
        if all_alive and "kernel" not in case:
            case["kernel"] = list(
                map(kernel_rows, first["before"]["positions"], first["before"]["velocities"])
            )
        case["all_alive"] &= all_alive
        saturation = 0.5 if profile == "selection" else 1.0
        for frame in content["frames"]:
            case["frames"] += 1
            x, y = frame["proposal"]["positions"]
            n = len(x)
            case["N"] = n
            case["d"] = len(x[0])
            for validity in frame["proposal"]["validity"]:
                check(
                    "post_proposal_all_alive",
                    n - sum(not s["terminated"] and not s["truncated"] for s in validity),
                )
            case["C_reset"] = frame["C_reset"]
            proxy = variance(x) + variance(y)
            conditional = sum(
                m["expected_position_variance"] for m in frame["conditional_moments"]
            )
            case["proxy"].append(proxy)
            case["centered_error"].append(frame["centered_positional_transport"])
            case["proxy_residual"].append(proxy - conditional)
            check("retained_proxy_recomputed", abs(proxy - frame["positional_proxy"]))
            check("centered_positional_bound", frame["centered_positional_transport"] - proxy)
            check(
                "q_barycenter_split",
                abs(
                    frame["full_q_transport"] - frame["centered_q_transport"] - frame["q_location"]
                ),
            )
            for name in ["full_q_plan", "centered_q_plan", "centered_positional_plan"]:
                plan = frame[name]
                check(
                    "normalized_transport_marginals",
                    max(
                        *(abs(sum(row) - 1 / n) for row in plan),
                        *(abs(sum(row[j] for row in plan) - 1 / n) for j in range(n)),
                    ),
                )
            target = frame["realized_target_certificate"]
            if not target or target.get("applicable") is not True:
                case["target_excluded"] += 1
                continue
            case["target_applicable"] += 1
            case["target_positive"] += int(target["positive_Q_lower"])
            for key in constant_keys:
                case["target_constants"][key].append(target[key])
            for side in [0, 1]:
                fitness = frame["sampled_fitness"][side]["fitness"]
                probabilities = frame["conditional_moments"][side]["acceptance_probabilities"]
                kernel = case["kernel"][side]
                for i, row in enumerate(kernel):
                    expected = sum(
                        k
                        * min(
                            1,
                            max(0, (fitness[j] - fitness[i]) / ((fitness[i] + 1e-6) * saturation)),
                        )
                        for j, k in enumerate(row)
                    )
                    check("independent_conditional_acceptance", abs(expected - probabilities[i]))
                    check(
                        "configured_donor_floor",
                        target["a_star"] / (n - 1) - min(k for j, k in enumerate(row) if j != i),
                    )
                check(
                    "bounded_sampled_fitness",
                    max(
                        target["fitness_bounds"][0] - min(fitness),
                        max(fitness) - target["fitness_bounds"][1],
                    ),
                )
            for observed, bound in [
                ("target_fraction", "definition_overlap_lower"),
                ("fitness_variance", "s_star_squared"),
                ("unfit_fraction", "f_U_F"),
                ("fit_fraction", "f_U_F"),
                ("minimum_target_pressure", "p_u"),
                ("target_error", "target_error_lower"),
                ("Q", "Q_lower"),
            ]:
                check(f"target_{observed}", target[bound] - target[observed])

    for case in cases.values():
        case.pop("kernel", None)
        for key in ["proxy", "centered_error", "proxy_residual"]:
            samples = case[key]
            case[key] = {
                "mean": statistics.mean(samples),
                "minimum": min(samples),
                "maximum": max(samples),
                "standard_error": statistics.stdev(samples) / math.sqrt(len(samples))
                if len(samples) > 1
                else 0.0,
            }
        case["target_constants"] = {
            k: {"minimum": min(v), "maximum": max(v), "mean": statistics.mean(v)} if v else None
            for k, v in case["target_constants"].items()
        }
    for row in checks.values():
        row["passed"] = row["maximum_positive_residual"] <= 2e-10
    result = {
        "dataset": str(args.dataset),
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "new_engine_steps": 0,
        "source": "SHA256-verified snapshot of journaled native operator frames",
        "raw_frame_provenance": [{"path": e["path"], "sha256": e["sha256"]} for e in entries],
        "source_and_code_provenance": [
            e for e in journal if e["tag"] in {"chapter04-source", "provenance"}
        ],
        "frame_archive_checksums_verified": len(entries),
        "checks": checks,
        "cases": [cases[k] for k in sorted(cases)],
        "scope": "Independent retained-frame arithmetic audit. Target constants are event-specific and are never averaged into a state-uniform rate. Partial-input reset uses Chapter 3; every stored transport plan has uniform probability marginals. Case profile ordering matches the recorded full-matrix module.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({
            "frame_archives": len(entries),
            "frames": sum(c["frames"] for c in cases.values()),
            "failed": [k for k, v in checks.items() if not v["passed"]],
        })
    )


if __name__ == "__main__":
    main()
