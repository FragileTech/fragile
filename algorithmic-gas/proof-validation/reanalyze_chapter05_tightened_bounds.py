"""Compare certified stronger Chapter 5 bounds with immutable full-horizon runs."""

import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path

from reanalyze_chapter05_stratified_uncertainty import comparison, stratified_estimate


def equation(source, tag):
    center = source.index(r"\tag{" + tag + "}")
    start = source.rfind("$$", 0, center)
    end = source.index("$$", center) + 2
    return source[start:end]


def metric_certificate(k, beta, lower, upper):
    with localcontext() as ctx:
        ctx.prec = 80
        k, beta, lower, upper = map(Decimal.from_float, [k, beta, lower, upper])
        # Both full 2x2 Loewner inequalities, using exact retained f64 parameters.
        small = [1 - lower * k, 1 - lower, beta]
        large = [upper * k - 1, upper - 1, -beta]
        valid = all(v[0] >= 0 and v[1] >= 0 and v[0] * v[1] >= v[2] ** 2 for v in [small, large])
        return {
            "passed": valid,
            "precision": "80-digit decimal, exact binary64 inputs",
            "lower_gap": [str(v) for v in small],
            "upper_gap": [str(v) for v in large],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path)
    parser.add_argument("coefficient_json", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.archive_root.resolve()
    if args.output.resolve().is_relative_to(root):
        msg = "Retain the original native archive; write a separate derived review"
        raise ValueError(msg)
    coefficient_bytes = args.coefficient_json.read_bytes()
    coefficients = json.loads(coefficient_bytes)
    chapter = (
        Path(__file__).resolve().parents[1]
        / "docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    )
    source_bytes = chapter.read_bytes()
    source = source_bytes.decode()
    index = json.loads((root / "archive-index.json").read_text())
    cases = []
    for entry in index["entries"]:
        if entry["kind"] != "raw_json" or not entry["tag"].endswith("/coupling_plan"):
            continue
        data = (root / entry["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Saved plan checksum mismatch: {entry['path']}")
        plan = json.loads(gzip.decompress(data))
        hypotheses = plan["hypotheses"]
        constant = next(
            c for c in coefficients if c["curvature"] == 1 and c["d"] == hypotheses["d"]
        )
        diagonal, sector = constant["gaussian"], constant["sector"]
        for key in ["h", "gamma", "B", "V"]:
            if diagonal[key] != hypotheses[key]:
                raise ValueError(f"Retained parameter mismatch: {key}")
        if hypotheses["constants"]["k"] != diagonal["k"]:
            msg = "Retained physical metric differs from predicted metric"
            raise ValueError(msg)
        delta_q = diagonal["delta_exact_spectrum_lower"]
        delta_g = sector["delta_pathwise_lower"]
        metric = sector["metric_equivalence_to_Q_diagonal"]
        # Enlarge the earlier display's upper endpoint by adjacent f64 values;
        # certify the actual numerical m/M matrices independently below.
        upper = math.nextafter(metric["upper"], math.inf)
        prefactor = math.nextafter(upper / metric["lower"], math.inf)
        certificate = metric_certificate(diagonal["k"], sector["beta"], metric["lower"], upper)
        if not certificate["passed"]:
            msg = "The exact metric conversion matrices are not positive semidefinite"
            raise ValueError(msg)
        checks = []
        steps = len(plan["trajectories"][0]) - 1
        for step in range(1, steps + 1):
            for name, bound, label, tag in [
                (
                    "dimension_diagonal",
                    (1 - delta_q) ** step,
                    "cor-kinetic-dimension-iterated-transport",
                    "5.DIM9a",
                ),
                (
                    "converted_sector",
                    prefactor * (1 - delta_g) ** step,
                    "cor-kinetic-dimension-metric-equivalence",
                    "5.DIM8",
                ),
            ]:
                check = comparison(
                    f"{name}/step{step}", [row[step] for row in plan["trajectories"]], bound
                )
                check["source_labels"] = [label]
                check["source_quotes"] = [equation(source, tag)]
                check["hypotheses"] = hypotheses
                check["scope"] = (
                    "Saved optimal population-normalized physical Q transport; exact metric Loewner inequalities checked independently, and stronger sector rate transferred with explicit M/m prefactor"
                )
                checks.append(check)
        final, se = stratified_estimate([row[-1] for row in plan["trajectories"]])
        cases.append({
            "N": hypotheses["N"],
            "d": hypotheses["d"],
            "steps": steps,
            "independent_replicates": len(plan["trajectories"]),
            "raw_plan": str(root / entry["path"]),
            "raw_plan_sha256": entry["sha256"],
            "final_optimal_Q_ratio": final,
            "final_standard_error": se,
            "endpoint_descriptive_rate": -math.log(final) / (steps * hypotheses["h"]),
            "diagonal_delta": delta_q,
            "diagonal_guaranteed_rate": -math.log1p(-delta_q) / hypotheses["h"],
            "sector_delta": delta_g,
            "sector_guaranteed_rate": -math.log1p(-delta_g) / hypotheses["h"],
            "metric_conversion_prefactor": prefactor,
            "metric_certificate": certificate,
            "final_diagonal_envelope": (1 - delta_q) ** steps,
            "final_converted_sector_envelope": prefactor * (1 - delta_g) ** steps,
            "comparisons": checks,
        })
    if len(cases) != 6:
        raise ValueError(f"Expected six immutable full-horizon plans, found {len(cases)}")
    report = {
        "chapter": 5,
        "source_path": str(chapter),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "coefficient_artifact": str(args.coefficient_json.resolve()),
        "coefficient_sha256": hashlib.sha256(coefficient_bytes).hexdigest(),
        "native_steps": 0,
        "scope": "New source-bound bounds evaluated on previously saved native trajectories; no trajectories added or counted as independent new samples",
        "cases": cases,
        "summary": {
            "comparisons": sum(len(c["comparisons"]) for c in cases),
            "comparisons_failed": sum(
                not check["passed"] for c in cases for check in c["comparisons"]
            ),
            "compressed_plan_hashes_verified": True,
            "metric_conversion_certificates": len(cases),
        },
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
