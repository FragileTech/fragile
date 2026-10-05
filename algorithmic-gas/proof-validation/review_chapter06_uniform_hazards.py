"""Check uniform Gaussian box hazard tails using committed independent trajectories."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower_logs(d, n, half_width, sigma):
    z = half_width / sigma
    log_q = math.log(2) - math.log(2 * math.pi) / 2 - z * z / 2 + math.log(z) - math.log1p(z * z)
    log_row = max(log_q, math.log(d) + log_q + (d - 1) * math.log1p(-math.exp(log_q)))
    log_native = n * log_row
    second = math.log(n) + math.log1p(-math.exp(log_row)) + (n - 1) * log_row
    maximum = max(log_native, second)
    return log_native, maximum + math.log(
        math.exp(log_native - maximum) + math.exp(second - maximum)
    )


def review(dataset, output):
    index = dataset / "archive-index.json"
    inventory = json.loads(index.read_text())
    if inventory["status"] not in {"complete", "completed"}:
        message = "Input must be committed complete."
        raise ValueError(message)
    entry = next(e for e in inventory["entries"] if e["tag"] == "chapter06-report")
    source = dataset / entry["path"]
    if sha(source) != entry["sha256"]:
        message = "Indexed native report checksum mismatch."
        raise ValueError(message)
    report = json.loads(gzip.decompress(source.read_bytes()))
    rows = []
    profiles = []
    # One bounded Bernoulli average at each fixed time and law; never treat
    # overlapping trajectories/times or paired left/right as independent units.
    count = sum(len(c["checkpoint_law_frames"]) * 2 for c in report["cases"])
    delta = 0.01 / count
    for case in report["cases"]:
        cfg = case["native_config"]
        domain = cfg["boundary"]["domain"]
        d, n = case["d"], case["N"]
        lower, upper = domain["lower"], domain["upper"]
        if (
            cfg["boundary"]["kind"] != "absorbing_box"
            or len(lower) != d
            or len(upper) != d
            or any(a != lower[0] or b != -lower[0] for a, b in zip(lower, upper))
            or cfg["kinetic"]["boundary_schedule"] != "end_of_step"
        ):
            message = "Uniform symmetric final-only absorbing box hypotheses fail."
            raise ValueError(message)
        sigma = cfg["kinetic"]["position_diffusion"] * math.sqrt(
            cfg["kinetic"]["integrator"]["dt"]
        )
        if sigma <= 0:
            message = "Positive independent final Gaussian position diffusion required."
            raise ValueError(message)
        log_native, log_chapter = lower_logs(d, n, -lower[0], sigma)
        profiles.append({
            "case": case["case"],
            "d": d,
            "N": n,
            "sigma": sigma,
            "half_width": -lower[0],
            "log_native_hazard_lower": log_native,
            "log_chapter_hazard_lower": log_chapter,
            "log_native_expected_lifetime_upper": -log_native,
            "log_chapter_expected_lifetime_upper": -log_chapter,
        })
        for frame in case["checkpoint_law_frames"]:
            log_h = (
                log_native if frame["cemetery_convention"] == "native_zero_alive" else log_chapter
            )
            hazard = math.exp(log_h)
            t = frame["step"]
            bound = math.exp(t * math.log1p(-hazard))
            samples = frame["requested_trajectories_each"]
            allowance = math.sqrt(math.log(1 / delta) / (2 * samples))
            for distribution, survivors in enumerate(frame["survivor_counts"]):
                observed = survivors / samples
                rows.append({
                    "id": f"case{case['case']}-{frame['cemetery_convention']}-law{distribution}-time{t}",
                    "case": case["case"],
                    "d": d,
                    "N": n,
                    "step": t,
                    "cemetery_convention": frame["cemetery_convention"],
                    "distribution": distribution,
                    "source_labels": ["rem-extinction-inevitable"],
                    "observed": observed,
                    "bound": bound,
                    "relation": "survival probability upper",
                    "signed_residual": observed - bound,
                    "hoeffding_allowance": allowance,
                    "passed": observed <= bound + allowance,
                    "independent_seed_units": samples,
                    "survivor_units": survivors,
                    "log_hazard_lower": log_h,
                    "hazard_float_underflowed": hazard == 0,
                    "scope": "Uniform over unrestricted random preparation means; fixed-N joint hazard, independent final coordinate/row position noise. Each initial law keeps its own original denominator. Native k=0 and externally stopped k<2 are distinct chains; no QSD law-rate or N-uniform whole-swarm hazard claim.",
                })
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "chapter": 6,
        "comparisons": rows,
        "profiles": profiles,
        "summary": {
            "comparisons": len(rows),
            "comparisons_failed": sum(not r["passed"] for r in rows),
            "new_native_steps": 0,
        },
        "provenance": {
            "input_index": str(index.resolve()),
            "input_index_sha256": sha(index),
            "indexed_report": str(source.resolve()),
            "indexed_report_sha256": sha(source),
            "executed_helper_sha256": sha(Path(__file__)),
        },
        "arithmetic_scope": "Real analytic Mills/event bounds evaluated in binary64, without directed roundoff certification. Finite log-scale lower certificates survive probability underflow; zero-float comparisons are explicitly marked uninformative.",
        "family_failure_budget": 0.01,
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    lines = [
        "# Chapter 6 uniform box hazard validation",
        "",
        result["arithmetic_scope"],
        "",
        "| d | N | Convention | Law | Step | Survivors/own denominator | Predicted survival upper | Hoeffding allowance |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["case"] == 6:
            lines.append(
                f"| {row['d']} | {row['N']} | {row['cemetery_convention']} | {row['distribution']} | {row['step']} | {row['survivor_units']}/{row['independent_seed_units']} | {row['bound']:.6g} | {row['hoeffding_allowance']:.6g} |"
            )
    (output / "hazard-table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result["summary"]))
    if result["summary"]["comparisons_failed"]:
        message = "Uniform hazard comparison discrepancy."
        raise SystemExit(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    review(args.dataset, args.output)
