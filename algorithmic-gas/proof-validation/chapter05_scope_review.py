"""Bind whole native positional formulas while preserving original retained-data analysis."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        msg = "Review output already exists; preserve immutable evidence"
        raise ValueError(msg)
    original = args.input.read_bytes()
    report = json.loads(original)
    source_path = Path(
        "../docs/source/2_fractal_gas/convergence_program/05_kinetic_contraction.md"
    )
    source_bytes = source_path.read_bytes()
    source = source_bytes.decode()

    def expression(tag):
        return next(body.strip() for body in source.split("$$")[1::2] if rf"\tag{{{tag}}}" in body)

    groups = {}
    for row in report["comparisons"]:
        if "/conditional_" in row["id"]:
            group, component = row["id"].rsplit("/conditional_", 1)
            row.pop("source_quotes", None)
            row["whole_theorem_validated"] = False
            groups.setdefault(group, []).append((component, row))
        if row["id"].endswith("/native_increment_residual"):
            row["source_formula"] = expression("5.X4")
    combined = []
    for group, rows in groups.items():
        if {component for component, _ in rows} != {"variance", "total", "barycenter"}:
            msg = "Missing complete three-component conditional moment evidence"
            raise ValueError(msg)
        maximum = max(abs(row["observed"]) / (row["standard_error"] + 5e-12) for _, row in rows)
        combined.append({
            "id": group + "/all_conditional_position_moments",
            "source_labels": ["cor-kinetic-native-positional-moments"],
            "source_formula": expression("5.X3"),
            "observed": maximum,
            "bound": 6.0,
            "relation": "less_equal",
            "passed": maximum <= 6.0,
            "components": [
                {
                    "component": component,
                    "observed_residual": row["observed"],
                    "standard_error": row["standard_error"],
                    "bound": 0.0,
                }
                for component, row in rows
            ],
            "independent_complete_replicas": rows[0][1]["independent_complete_replicas"],
            "hypotheses": rows[0][1]["hypotheses"],
            "scope": "All three whole-formula 5.X3 expectations checked jointly by maximum absolute standardized residual; original component means and complete-replica SE retained, with 5e-12 arithmetic allowance. No new data or global-domain certificate.",
        })
    # Review does not repeat the 103MiB raw observations; retain their checksum-bound source.
    report.pop("observations")
    report["comparisons"].extend(combined)
    report["provenance_review"] = {
        "original_report": str(args.input),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "original_comparisons": report["summary"]["comparisons"],
        "additional_whole_formula_reviews": len(combined),
        "numerical_results_changed": False,
        "new_native_steps": 0,
        "scope": "Original individual moment quotes narrowed; exact entire 5.X3 and 5.X4 expressions receive combined evidence. The original report/raw/index/source/executable provenance remains immutable.",
    }
    report["current_source_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    report["summary"]["comparisons"] = len(report["comparisons"])
    report["summary"]["comparisons_failed"] = sum(
        r["passed"] is not True for r in report["comparisons"]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
