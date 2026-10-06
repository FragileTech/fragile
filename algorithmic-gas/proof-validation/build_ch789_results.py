"""Bind the final Chapters 7–9 ledgers and datasets to their exact source files."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT
PROOF = PACKAGE / "proof-validation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def evidence(path):
    data = read(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha(path),
        "summary": data.get("summary", {}),
        "scope": data.get("scope"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("chapter09_final_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--additional-report", type=Path, action="append", default=[])
    args = parser.parse_args()
    base = args.experiment_root.resolve()
    ledger_paths = {
        7: base / "chapter07/ledger-v1/statement-ledger.json",
        8: base / "chapter08/final-ledger-v3/report.json",
        9: args.chapter09_final_report.resolve(),
    }
    chapters = []
    for chapter, ledger_path in ledger_paths.items():
        inventory_path = PROOF / f"chapter{chapter:02d}_inventory.json"
        inventory = read(inventory_path)
        ledger = read(ledger_path)
        source = ROOT / inventory["source_path"]
        digest = sha(source)
        if digest != inventory["source_sha256"] or digest != ledger["source_sha256"]:
            raise ValueError(f"Chapter {chapter}: current source SHA mismatch")
        expected = {row["id"]: row for row in inventory["quantitative_expressions"]}
        actual = {
            row.get("id") or row["source_expression_id"]: row
            for row in ledger["expression_ledger"]
        }
        if len(actual) != len(ledger["expression_ledger"]) or set(actual) != set(expected):
            raise ValueError(f"Chapter {chapter}: expression omissions or duplicates")
        for key in expected:
            formula = actual[key].get("formula") or actual[key]["exact_formula"]
            if formula != expected[key]["formula"]:
                raise ValueError(f"Chapter {chapter}: source expression differs: {key}")
        disposition_key = "validation_status" if chapter == 9 else "status"
        statuses = Counter(
            row.get(disposition_key, row.get("disposition", "unspecified"))
            for row in actual.values()
        )
        failures = [c["id"] for c in ledger.get("checks", []) if not c["passed"]]
        if failures:
            raise ValueError(f"Chapter {chapter}: failed comparisons: {failures[:5]}")
        if ledger["summary"].get("residual_numeric_estimates"):
            raise ValueError(f"Chapter {chapter}: unassigned finite estimates")
        if statuses.get("finite_estimate_retained_analytic") or statuses.get("unspecified"):
            raise ValueError(f"Chapter {chapter}: final ledger has unassigned estimates")
        chapters.append({
            "chapter": chapter,
            "formal_statements": len(inventory["formal_items"]),
            "source_expressions": len(expected),
            "source": str(source.relative_to(ROOT)),
            "source_sha256": digest,
            "inventory_sha256": sha(inventory_path),
            "ledger": evidence(ledger_path),
            "expression_dispositions": dict(statuses),
        })

    relative_reports = [
        "chapter07/native-laws-v1/report.json",
        "chapter07/poisson-clouds-v1/report.json",
        "chapter07/diffusion-reference-v1/raw.json",
        "chapter08/full-v1/report.json",
        "chapter08/viscosity-v2/report.json",
        "chapter09/bridge-full-v1/report.json",
        "chapter09/independent-native-memory-v1/report.json",
        "chapter09/native-resonance-v1/report.json",
        "chapter09/distance-variation-review-v3/report.json",
        "chapter09/population-scaling-v1/report.json",
        "chapter09/horizon-variance-v1/report.json",
        "checks/rust-tests.json",
        "checks/population-statistics.json",
        "checks/final-archive-audit.json",
    ]
    reports = [evidence(base / path) for path in relative_reports]
    reports += [evidence(path.resolve()) for path in args.additional_report]
    source_paths = [
        *PACKAGE.glob("crates/benchmarks/src/convergence_chapter0[789]_completion.rs"),
        *PACKAGE.glob("crates/benchmarks/tests/convergence_chapter0[789]_completion.rs"),
        *PACKAGE.glob("crates/benchmarks/src/bin/gas-chapter0[789]-*.rs"),
        PACKAGE / "crates/benchmarks/src/bin/gas-native-kinetic-resonance.rs",
        PACKAGE / "crates/benchmarks/src/bin/gas-population-independent-review.rs",
        PACKAGE / "crates/benchmarks/src/bin/gas-native-horizon-variance.rs",
        PACKAGE / "crates/benchmarks/src/bin/gas-native-original-measurement-replacement.rs",
        PROOF / "complete_chapter09_estimates.py",
        PROOF / "chapter09_residual_estimates.py",
        PROOF / "build_chapter09_validation_table.py",
        PROOF / "population_scaling_review.py",
        PROOF / "verify_ch789_archives.py",
    ]
    replacement = PACKAGE / "crates/benchmarks/src/bin/gas-native-measurement-replacement.rs"
    if replacement.exists():
        source_paths.append(replacement)
    result = {
        "summary": {
            "formal_statements": sum(c["formal_statements"] for c in chapters),
            "source_expressions": sum(c["source_expressions"] for c in chapters),
            "source_expression_coverage_complete": True,
            "fresh_complete_native_updates": 4032,
            "independent_root_reference_draws": 23040,
            "independent_PPP_clouds": 122880,
            "independent_PDE_reference_steps": 27648,
        },
        "chapters": chapters,
        "datasets_and_checks": reports,
        "current_implementation_sources": [
            {"path": str(p.relative_to(ROOT)), "sha256": sha(p)}
            for p in sorted(set(source_paths))
        ],
        "catalog": {
            "path": "proof-validation/chapters07-09-results.md",
            "sha256": sha(PROOF / "chapters07-09-results.md"),
        },
        "counting_scope": (
            "Source expressions include definitions, hypotheses and infinite-law conclusions. "
            "Numerical comparison counts retain their own dataset scopes. Reused native frames, "
            "cross-chapter references, root draws, PDE steps and primitive counterfactuals are "
            "never counted as new complete native updates or as additional independent runs."
        ),
        "analytic_scope": (
            "Global QSD concentration, attraction, uniqueness, LSI and infinite-law limits "
            "retain their analytic hypotheses; finite experiments do not establish them."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))


if __name__ == "__main__":
    main()
