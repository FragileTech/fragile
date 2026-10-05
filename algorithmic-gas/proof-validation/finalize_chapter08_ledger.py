"""Join exact Chapter8 aliases to independently retained comparison ledgers."""

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chapter_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    inputs = {
        "base": args.chapter_root / "chapter08/ledger-v3/report.json",
        "viscosity": args.chapter_root / "chapter08/viscosity-v2/report.json",
        "conditional_variance": args.chapter_root / "chapter09/bridge-full-v1/report.json",
    }
    reports = {key: json.loads(path.read_text()) for key, path in inputs.items()}
    report = reports["base"]
    rows = {r["source_expression_id"]: r for r in report["expression_ledger"]}
    tail = rows["chapter08-expression-0061"]
    tail.update({
        "status": "exact_formula_scoped_finite_evidence",
        "detail": "Same component-size Markov bound used in the one-step proof. All225 K-tail comparisons retain actual native component sizes, frozen conditional C_N and global C separately; no sampled maximum replaces the uniform kernel lower bound.",
        "evidence_check_references": [{
            "report": str(inputs["base"]), "sha256": sha(inputs["base"]),
            "source_label": "lem-mean-field-component-bound",
            "quantity": "conditional-component-tail", "available_comparisons": 225,
            "supplement_locations": [r["path"] for r in report["supplement_checks"]["archives"]],
        }],
    })
    variance = rows["chapter08-expression-0062"]
    cases = reports["conditional_variance"]["native_population_comparisons"]["cases"]
    variance_checks = [
        c for case in cases for c in case["checks"] if c["kind"] == "conditional-variance"
    ]
    assert len(variance_checks) == 315 and all(c["passed"] for c in variance_checks)
    variance.update({
        "status": "exact_formula_scoped_finite_evidence",
        "detail": "Cross-chapter A_phi/N uses Chapter9's actual canonical constants and315 conditional-variance comparisons from the same45 native inputs. Global constants, numerical overflow logs, universal bounded-test cap, and independent-replica uncertainty are retained. A vacuous analytic upper bound is not evidence of a sharp rate.",
        "evidence_check_references": [{
            "report": str(inputs["conditional_variance"]),
            "sha256": sha(inputs["conditional_variance"]),
            "source_label": "thm-chaos-canonical-conditional-variance",
            "quantity": "conditional-variance", "available_comparisons": len(variance_checks),
        }],
    })
    bias = rows["chapter08-expression-0064"]
    bias_checks = [c for case in cases for c in case["checks"] if c["kind"] == "conditional-bias"]
    assert len(bias_checks) == 315 and all(c["passed"] for c in bias_checks)
    bias.update({
        "status": "exact_formula_scoped_finite_evidence",
        "detail": "The explicit2||phi||infinity B*/sqrtN prefactor uses Chapter9's exact native-kernel constants and315 empirical-input bias comparisons, with full independent AtomicMeanField integration and independent-root Monte Carlo uncertainty. Binary64 overflow is retained as a logarithmic bound; it does not establish a tight bias rate.",
        "evidence_check_references": [{
            "report": str(inputs["conditional_variance"]),
            "sha256": sha(inputs["conditional_variance"]),
            "source_label": "thm-chaos-canonical-quantitative-bias",
            "quantity": "conditional-bias", "available_comparisons": len(bias_checks),
        }],
    })
    rows["chapter08-expression-0081"].update({
        "status": "native_state_or_parameter_contract",
        "detail": "Actual canonical nu0 quadratic and Rastrigin native forces are the negative analytic landscape gradient. Every saved B1/B2 kick compares the full configured native force and actual stage velocity to that formula.",
        "evidence_check_references": [
            *rows["chapter08-expression-0069"]["evidence_check_references"],
            *rows["chapter08-expression-0070"]["evidence_check_references"],
        ],
    })
    for expression_id in ["chapter08-expression-0127", "chapter08-expression-0128", "chapter08-expression-0130", "chapter08-expression-0131", "chapter08-expression-0132"]:
        rows[expression_id].update({
            "status": "analytic_small_step_statement_with_cap_diagnostics",
            "detail": "These expressions belong to the small-h warning: a complete native order-one cloning/cap attempt can have F0(mu) different from mu. Exact repeated-cap fixtures quantify its singular time scaling. No continuous-time generator or rate is inferred from this notation.",
            "evidence_check_references": [{
                "report": str(inputs["base"]), "sha256": sha(inputs["base"]),
                "archive": "finite-law-transfer-and-cap.json.gz", "quantity": "repeated-cap",
            }],
        })
    epsilon = rows["chapter08-expression-0135"]
    epsilon.update({
        "status": "exact_coupling_premise_instantiated",
        "detail": "Four exact independent two-atom reference fixtures N=2,4,8,16 use Xi=Yi+0.05, so the normalized quadratic coupling error is epsilon_N=0.0025 for every N. Both Wasserstein and Lipschitz transfer comparisons retain that exact operand. This instantiates a premise; it does not establish such a coupling for the native algorithm.",
        "evidence_check_references": [{
            "report": str(inputs["base"]), "sha256": sha(inputs["base"]),
            "archive": "finite-law-transfer-and-cap.json.gz",
            "quantity": "coupling-transfer-W2 and coupling-transfer-test",
            "available_comparisons": 8, "epsilon_N": 0.0025,
        }],
    })
    for expression_id in ["chapter08-expression-0075", "chapter08-expression-0076", "chapter08-expression-0077"]:
        row = rows[expression_id]
        row.update({
            "status": "exact_formula_scoped_finite_evidence",
            "detail": "Separate actual nu=0.3 native viscosity archives evaluate both B1 and B2 physical Gaussian kernels, count/row denominators, complete force, and explicit finite self-exclusion correction to the atomic population integral. These comparisons do not transfer the nu=0 canonical chaos theorem to viscosity.",
            "evidence_check_references": [{
                "report": str(inputs["viscosity"]), "sha256": sha(inputs["viscosity"]),
                "source_expression_id": expression_id,
                "available_comparisons": 783360, "scope": "1280 retained updates;0 new updates",
            }],
        })
    report["summary"]["residual_numeric_estimates"] = [
        value for value in report["summary"]["residual_numeric_estimates"]
        if rows[value]["status"] == "analytic_definition_or_hypothesis"
    ]
    report["joined_inputs"] = {
        key: {"path": str(path), "sha256": sha(path)} for key, path in inputs.items()
    }
    report["summary"]["additional_viscosity_checks"] = 783360
    report["summary"]["checks_total_including_viscosity"] = 4121639 + 783360
    report["summary"]["cross_chapter_variance_comparisons_referenced_not_double_counted"] = 315
    report["summary"]["cross_chapter_bias_comparisons_referenced_not_double_counted"] = 315
    assert not report["summary"]["residual_numeric_estimates"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    table = [
        "# Chapter8 exact expression ledger", "",
        "Every expression copies its source formula and keeps its precise evidence scope. Finite tests cannot prove global weak continuity, fixed-point existence, or asymptotic convergence.", "",
        "| Source expression | Formula | Disposition | Evidence and limitation |",
        "|---|---|---|---|",
    ]
    for row in report["expression_ledger"]:
        formula = row["exact_formula"].replace("|", "&#124;").replace("\n", " ")
        detail = row["detail"].replace("|", "&#124;")
        table.append(f"| {row['source_expression_id']} | `{formula}` | {row['status']} | {detail} |")
    (args.output / "expressions-table.md").write_text("\n".join(table) + "\n")
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    verification = {
        "input_hashes_checked": {key: sha(path) for key, path in inputs.items()},
        "expressions": len(rows), "residual_numeric_estimates": [],
        "new_native_updates": 0,
        "outputs": {p.name: sha(p) for p in args.output.iterdir() if p.is_file()},
    }
    (args.output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
