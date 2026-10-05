"""Assemble a compact statement-level estimate ledger with immutable provenance."""

import argparse
from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import shutil

from complete_chapter04_estimates import comparison, require, sha, write_json
from complete_chapter04_qsd_primitives import source_evaluator


def run(repository, root):
    pressure_path = root / "chapter04/report.json"
    geometry_path = root / "chapter04-geometry/report.json"
    primitive_path = root / "chapter04-qsd-primitives-final/report.json"
    pressure = json.loads(pressure_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    primitive = json.loads(primitive_path.read_text())
    evaluator, evaluator_path = source_evaluator(repository)
    auxiliary, bands = [], []
    for ref in primitive["references"]:
        config, p = ref["native_config"], ref["primitive"]
        mode = ref["mode"]
        force_band = evaluator.viscous_qsd_primitive_certificate(
            n_walkers=ref["N"],
            dimension=ref["d"],
            timestep=p["h"],
            viscosity=config["qft"]["viscosity"]["coefficient"],
            bandwidth=config["qft"]["viscosity"]["bandwidth"],
            force_lipschitz=100.0,
            force_at_origin=0.0,
            terminal_half_width=2.0,
            clone_jitter=config["clone_transform"]["jitter_amplitude"],
            restitution=config["clone_transform"]["restitution"],
            velocity_cap=config["kinetic"]["velocity_cap"],
            ou_coefficient=p["c"],
            ou_amplitude=p["q"],
            position_amplitude=p["s"],
            row_normalized=mode == "row",
        )
        require(
            force_band.certified and not force_band.limitations,
            "Displayed LF<=100 force family fails primitive predicates",
        )
        auxiliary.extend([
            comparison(
                f"{mode}:force-band-actual-profile",
                ["cor-w2-force-profile-band"],
                p["force_lipschitz"],
                100,
                hypotheses="Actual retained harmonic force is globally analytic F=-x; LF=1 from analytic derivative, already checked at native stages. Band envelope LF<=100 is an overestimate of this same force, never a changed experiment",
            ),
            comparison(
                f"{mode}:force-band-coercivity",
                ["cor-w2-force-profile-band"],
                0,
                force_band.coercivity_margin,
            ),
            comparison(
                f"{mode}:force-band-first-drift-margin",
                ["cor-w2-force-profile-band"],
                0,
                force_band.drift_margin,
            ),
            comparison(
                f"{mode}:force-band-survival-floor",
                ["cor-w2-force-profile-band"],
                p["log_survival_lower_bound"],
                force_band.log_survival_lower_bound,
                "lower",
            ),
            comparison(
                f"{mode}:separate-jitter-final-position-scales",
                ["rem-jitter-scale"],
                config["clone_transform"]["jitter_amplitude"] / p["s"],
                5.0,
                "equal",
                hypotheses="Actual native clone amplitude .1 and final position amplitude .1sqrt(.04)=.02 are separate consumed fields; ratio measured from config/stage scale",
            ),
        ])
        bands.append({
            "mode": mode,
            "analytic_force_class_LF_upper": 100,
            "actual_force_LF": 1,
            "coercivity_margin": force_band.coercivity_margin,
            "first_drift_margin": force_band.drift_margin,
            "log_survival_floor": force_band.log_survival_lower_bound,
            "log_epsilon": force_band.log_minorization_lower_bound,
            "log_M_N": force_band.log_two_step_density_upper_bound,
            "log_ell_N": force_band.log_output_density_lower_bound,
            "log_m_F": force_band.log_eigenfunction_min_lower_bound,
            "log_delta_F": force_band.log_doob_minorization_lower_bound,
            "scope": "Numeric assembly of the stated analytic force family envelope; no changed force/run and no measured stationary QSD law",
        })
    inputs = [(pressure_path, pressure), (geometry_path, geometry), (primitive_path, primitive)]
    original_paths = [
        repository
        / "algorithmic-gas/outputs/convergence/chapters04-06-experiments/full-20261004/chapter04/chapter04-report.json",
        repository
        / "algorithmic-gas/outputs/convergence/chapters04-06-experiments/tightening-20261004/chapter04/report.json",
    ]
    for path in original_paths:
        inputs.append((path, json.loads(path.read_text())))
    groups = defaultdict(list)
    for path, report in [*inputs, (root / "statement-ledger.json", {"comparisons": auxiliary})]:
        classified = defaultdict(list)
        for i, c in enumerate(report["comparisons"]):
            require(
                math.isfinite(c["observed"]) and math.isfinite(c["bound"]),
                "Non-numeric evidence cannot enter statement ledger",
            )
            for label in c.get("source_labels", []):
                classified[label].append((i, c))
        for label, comparisons in classified.items():
            worst = max(
                comparisons,
                key=lambda row: row[1].get("residual", row[1].get("maximum_absolute_residual", 0)),
            )
            groups[label].append({
                "report_path": str(path),
                "numerical_comparisons": len(comparisons),
                "failed": sum(not c["passed"] for _, c in comparisons),
                "source_numeric_evidence_pointer": "/comparisons",
                "first_comparison_index": comparisons[0][0],
                "last_comparison_index": comparisons[-1][0],
                "worst_record": worst[1],
                "scope": "Only named numerically evaluated subestimates; source-label presence does not credit every expression or universal claim",
            })
    items = []
    for statement in pressure["statement_ledger"]:
        item = {k: v for k, v in statement.items() if k != "numeric_evidence"}
        label = statement["source_label"]
        item["numerical_evidence_groups"] = groups[label]
        item["status"] = (
            "numerical_subestimates_checked"
            if groups[label]
            else "analytic_scope_remark_or_unclosed_statement"
        )
        if label == "def-w2-finite-population-qsd-regime":
            item["remaining_obligation"] = (
                "Same-kernel full primitive density/minorization/eigenfunction-floor assembly is now computed and native conditional survival checked. The universal analytic proof/functional kernel contract remains a theorem obligation; simulations do not certify it. Real dead coordinates and all Gaussian tails remain retained."
            )
        elif label in {
            "thm-w2-finite-n-conditioned-convergence",
            "cor-w2-reference-fitness-degeneracy",
            "cor-w2-force-profile-band",
        }:
            item["remaining_obligation"] = (
                "Full primitive rate/prefactor inputs now evaluated in logarithms; stationary QSD law/eigenfunction values not measured. Bounds stay TV<=1 at 128 steps and require astronomical time for a nontrivial decay prediction. This is finite-N positivity, not a population-uniform mixing result."
            )
        items.append(item)
    # Deep verification of every newly written compressed numerical ledger.
    verified = []
    for directory in (root / "chapter04", root / "chapter04-qsd-primitives-final"):
        for path in sorted(directory.glob("*.json.gz")):
            data = gzip.decompress(path.read_bytes())
            value = json.loads(data)
            verified.append({
                "path": str(path),
                "sha256": sha(path),
                "decoded_bytes": len(data),
                "decoded_kind": type(value).__name__,
            })
    source_hashes = [{"path": str(p), "sha256": sha(p)} for p, _ in inputs]
    source_files = [
        Path(__file__),
        evaluator_path,
        Path(__file__).with_name("complete_chapter04_estimates.py"),
        Path(__file__).with_name("complete_chapter04_geometry.py"),
        Path(__file__).with_name("complete_chapter04_qsd_primitives.py"),
        Path(__file__).with_name("read_native_cbor.py"),
        Path(__file__).with_name("audit_local_structural_kinetics.py"),
        Path(__file__).with_name("skip_native_cbor.c"),
    ]
    support = root / "reproducibility"
    require(not support.exists(), "Refuse to overwrite completion provenance")
    support.mkdir()
    for path in source_files:
        shutil.copy2(path, support / path.name)
        source_hashes.append({
            "path": str(path),
            "sha256": sha(path),
            "retained_copy": str(support / path.name),
        })
    report = {
        "chapter": 4,
        "new_native_steps": 0,
        "source_sha256": pressure["source_sha256"],
        "statement_ledger": items,
        "comparisons": auxiliary,
        "force_family_primitive_bounds": bands,
        "new_derived_archive_deep_verification": verified,
        "provenance": source_hashes,
        "summary": {
            "formal_statements": len(items),
            "statements_with_numeric_subestimate_evidence": sum(
                bool(i["numerical_evidence_groups"]) for i in items
            ),
            "new_pressure_comparisons": pressure["summary"]["comparisons"],
            "new_geometry_regional_comparisons": geometry["summary"]["comparisons"],
            "new_primitive_comparisons": primitive["summary"]["comparisons"],
            "new_auxiliary_comparisons": len(auxiliary),
            "new_numeric_failed": sum(
                sum(not c["passed"] for c in r["comparisons"]) for _, r in inputs[:3]
            )
            + sum(not c["passed"] for c in auxiliary),
            "exact_measurement_patterns": pressure["summary"]["exact_measurement_patterns"],
            "positive_complete_pressure_bounds": pressure["summary"][
                "positive_complete_pressure_bounds"
            ],
            "all_native_reference_tagged_rows": primitive["summary"]["all_tagged_rows"],
            "qualified_conditional_survival_rows": primitive["summary"]["qualified_tagged_rows"],
            "deep_verified_derived_archives": len(verified),
            "deep_verified_decoded_bytes": sum(v["decoded_bytes"] for v in verified),
        },
        "superseded_audits": [
            {
                "path": str(root / "chapter04-qsd-primitives-v2/report.json"),
                "reason": "Audit-code bug: executed_noise stores standard eta, so native position identity must multiply OU eta by q and final-position eta by s. Fixed evaluator with nonzero-noise regression; final audit has zero failures. Raw native experiment untouched.",
            }
        ],
        "scope": "Statement-level numerical subestimate ledger, not expression-catalog completion or proof of every global theorem. Primitive prediction and empirical applicability remain separate; default nonlinear selected-provider/tail closure remains explicit.",
    }
    write_json(root / "statement-ledger.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = run(args.repository, args.root)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
