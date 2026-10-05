"""Build an explicit statement ledger without turning local checks into global certificates."""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from generate_inventory import build


POLICY = {
    "thm-discretization": (
        "conditional_transfer",
        "5.D1 observable-specific certificate and a same-extension generator drift are required. Fixed native cap fails generator consistency; the affine uncapped extension has exact cubature only.",
    ),
    "prop-weak-error-variance": (
        "validated_affine_specialization",
        "5.D4 assembly is exact once M2 and barycenter coefficients are proved. Affine uncapped Cweak=12.29925819 and variance coefficient<=2*Cweak; general nonlinear moment envelopes are not supplied by samples.",
    ),
    "prop-weak-error-boundary": (
        "conditional_transfer",
        "An actual barrier, globally integrable derivatives and its same-extension weak-error coefficient are required; no general Kb is identified by the capped trajectories.",
    ),
    "prop-weak-error-wasserstein": (
        "deferred_matching_transfer",
        "Fixed representative coupling quadratics have exact affine checks. Optimal-matching stability and nonlinear C_LM are additional analytic inputs; fixed-cap continuum transfer is gated.",
    ),
    "prop-explicit-constants": (
        "validated_affine_specialization",
        "5.D6 exact Gaussian degree-two cubature, independent linear-SDE exponential/covariance and an analytic N-uniform prefactor are retained. This is not a nonlinear or fixed-cap SDE theorem.",
    ),
    "thm-inter-swarm-contraction-kinetic": (
        "conditional_macro_closure",
        "Requires actual barycenter-force closure and structural component LMI, jump-source bounds, weak-error certificates, and common decomposition. LF alone does not imply a positive nonconvex rate.",
    ),
    "thm-kinetic-exact-baoab-cap-coupling": (
        "validated_native_quadratic",
        "5.K1 actual capped harmonic Q coupling tested with independent complete pairs, optimal permutation-normalized transport and retained representative coupling upper bounds.",
    ),
    "cor-kinetic-canonical-coupling": (
        "validated_native_quadratic",
        "Complete canonical quadratic update retains terminal marks and noise laws; equal-fitness reference has zero optional copies, while killing/revival is separately scoped.",
    ),
    "lem-kinetic-shift-uniform-radial-cap": (
        "analytic_complete_finite_probes",
        "Shift-uniform Gaussian cap derivative expectation proved by Gaussian ball rearrangement; finite dimension/noise probes test its expectations, not all possible shifts.",
    ),
    "thm-kinetic-dimension-curvature-cap": (
        "validated_native_quadratic",
        "Dimension-explicit eta_d, curvature-dependent native matrix, stable exact 2D/(T+sqrt(T²-4D)) and finite-step costs are checked in d1/2/4/8, N4/64 and curvature .5/1/2.",
    ),
    "cor-kinetic-certified-chi-cap-integral": (
        "validated_numerical_certificate",
        "Directed radial Gaussian integral enclosures are mathematical numerical certificates, with independent high-precision cross-audits; ordinary quadrature is not used as a certified lower eta.",
    ),
    "thm-kinetic-dimension-curvature-sector": (
        "validated_native_quadratic",
        "Both endpoint LMIs are directed-certified and actual cap secants/native stages checked. Metric conversion prefactors are explicit and N-independent.",
    ),
    "cor-kinetic-regional-harmonic-reference": (
        "conditional_region",
        "Actual harmonic reference constants are certified; nonquadratic perturbation requires both kick queries in the declared region and separate escape/discrepancy accounting.",
    ),
    "cor-kinetic-dimension-metric-equivalence": (
        "validated_numerical_certificate",
        "Positive metric eigenvalues and normalized transport equivalence are checked. The metric prefactor is retained when changing rate metrics.",
    ),
    "cor-kinetic-dimension-iterated-transport": (
        "validated_native_quadratic",
        "Old 128-step and new 16-step raw trajectories show normalized errors decreasing. Iterated analytic envelopes use certified coefficients, never fitted endpoint slopes.",
    ),
    "lem-kinetic-terminal-status-coupling": (
        "validated_terminal_specialization",
        "Actual terminal marks and conditional Gaussian crossing probabilities/moduli are checked. Revived coordinates do not imply persistent particle identities.",
    ),
    "thm-kinetic-bounded-transport-smoothing": (
        "analytic_complete_law_probe_remaining",
        "5.K5 constants are explicit under global LF,c²LF<1,q>0. Existing common-noise trajectory pairs are not maximal-Gaussian row law couplings; a nontrivial empirical bounded-law smoothing check remains.",
    ),
    "cor-kinetic-full-cluster-smoothing": (
        "conditional_postclone_coupling",
        "5.K7 has a complete analytic proof for a supplied full cloning-law coupling. Complete conditional source-plan/cluster law coupling and row smoothing experiments are needed for empirical transfer; N-normalized weights cannot be replaced by whole-configuration TV.",
    ),
    "ex-kinetic-smoothing-force-constants": (
        "validated_parameter_specialization",
        "Global LF=1 (quadratic) and 2+40π² (Rastrigin) imply c²LF<1 at h=.04 in every dimension; sampled force queries independently match the declared native force.",
    ),
    "lem-location-error-drift-kinetic": (
        "validated_affine_LMI_conditional_closure",
        "5.L4 P=[[1,1/3],[1/3,2/3]], AᵀP+PA=−2I/3 and kappa0=.5527864 are certified. General macroforce closure, sources and numerical transfer remain prerequisites.",
    ),
    "lem-structural-error-drift-kinetic": (
        "conditional_macro_closure",
        "5.S1/S2 require proved clusterwise LMI, actual status/partition sources and global observable weak-error bounds. Partition labels are bookkeeping only; finite trajectory decline does not prove the closure.",
    ),
    "thm-velocity-variance-contraction-kinetic": (
        "pointwise_generator_and_native_cap",
        "Force-work, independent centered trace and Young residuals are evaluated at retained states. Continuum rho=2gamma-epsilon uses a uniform reachable force-square envelope. Native fixed-cap moments are independently bounded by V²; no continuum rate is borrowed.",
    ),
    "thm-velocity-barycenter-dissipation": (
        "pointwise_generator_and_native_cap",
        "Exact retained generator mean-force/noise source d sigma²/N and Young residual are evaluated. An actual global force envelope is still required to iterate the continuum estimate; native barycenter<=V² directly.",
    ),
    "cor-net-velocity-contraction": (
        "conditional_composition",
        "r_v C_Cv+b_Kv is the exact composition source when both same-kernel pointwise inputs hold. Native cap gives bounded moments but not an inherited continuum r_v.",
    ),
    "cor-net-barycenter-drift": (
        "validated_native_cap_specialization",
        "Post-cap total and barycenter squared speed<=V² pathwise; continuum composed r_mu inputs remain conditional.",
    ),
    "thm-positional-variance-bounded-expansion": (
        "repaired_exact_horizon_bound",
        "5.X1/X2 uses actual transient Mx,Mv and retains Mv h². No equilibrium autocovariance assumption or dropped remainder. Ballistic regression attains the exact bound and rejects equilibrium-only Mv substitution.",
    ),
    "assump-uniform-variance-bounds": (
        "explicit_analytic_input",
        "Mv=max(initial,C/rho) only after the continuum drift holds on the interval. Uniform Mx and full-update source/tail bounds must be independently established; empirical maxima confer no proof credit.",
    ),
    "cor-kinetic-native-positional-moments": (
        "validated_native_conditional_moments",
        "5.X3/X4 exact conditional Var, total M2, barycenter and finite-step Cauchy bound checked from actual B1 states and full Gaussian innovation laws, including the deterministic first viscosity. Physical dead coordinates stay retained.",
    ),
    "cor-net-positional-contraction": (
        "signed_closure_unresolved",
        "SCK.3/SCK.6 retain Keystone pressure and exact donor,barycenter,collision,force,cap,boundary residual. For current positive-fitness unbounded Rastrigin h=.04 no uniform upper bound on that residual or source-moment drift is proved.",
    ),
    "thm-boundary-potential-contraction-kinetic": (
        "conditional_weighted_corrector",
        "Needs force/barrier alignment, actual barrier-weighted Hessian-velocity moment and signed corrector-source bound. End-step cap and unweighted variance do not prove continuous weighted hypotheses.",
    ),
    "cor-total-boundary-safety": (
        "conditional_same_observable_composition",
        "r_K r_C and r_K b_C+b_K are exact composition algebra only for certified same-observable kernels/status convention. Chapter6 native Gaussian integrable-barrier bounds are a separate valid direct route.",
    ),
    "lem-kinetic-minorization": (
        "repaired_constructive_compact_row",
        "5.M1/M2 requires positive q and s, bounded prepared inputs, global inverse-T hypothesis and no interacting row force. Exact harmonic joint-density/cap-Jacobian lower bounds are checked at retained compact outputs. Row epsilon is N-free, whole-swarm product epsilon^N is not.",
    ),
    "prop-fokker-planck-kinetic": (
        "continuous_extension_only",
        "The displayed PDE is for the stated continuous generator. Native fixed-cap/status transitions are not the same PDE; no PDE density or equilibrium law is inferred from their histograms.",
    ),
    "rem-formal-invariant-measure": (
        "continuous_extension_only",
        "Fluctuation-dissipation Gibbs law requires its actual uncapped continuous conservative hypotheses. Native capped/killed full law cannot be assigned an unbounded Maxwell velocity distribution.",
    ),
}


def comparisons(value):
    if isinstance(value, dict):
        if isinstance(value.get("comparisons"), list):
            yield from value["comparisons"]
        for key, child in value.items():
            if key != "comparisons":
                yield from comparisons(child)
    elif isinstance(value, list):
        for child in value:
            yield from comparisons(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = build(5, "05_kinetic_contraction.md")
    evidence = {}
    retained = []
    for path in args.reports:
        raw = path.read_bytes()
        value = json.loads(raw)
        retained.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        for check in comparisons(value):
            obs, bound = check.get("observed"), check.get("bound")
            finite = (
                isinstance(obs, (int, float))
                and math.isfinite(obs)
                and isinstance(bound, (int, float))
                and math.isfinite(bound)
            )
            for label in check.get("source_labels", []):
                evidence.setdefault(label, []).append({
                    "report": str(path),
                    "id": check.get("id"),
                    "finite": finite,
                    "passed": check.get("passed"),
                    "observed": obs,
                    "bound": bound,
                    "standard_error": check.get("standard_error"),
                    "residual": check.get("residual", obs - bound if finite else None),
                    "relation": check.get("relation"),
                    "hypotheses": check.get("hypotheses"),
                    "scope": check.get("scope"),
                    "whole_theorem_validated": False,
                })
    rows = []
    expressions = {e["id"]: e for e in inventory["quantitative_expressions"]}
    for item in inventory["formal_items"]:
        label = item.get("source_label") or item["id"]
        status, action = POLICY.get(
            label,
            (
                "analytic_specification",
                "Definition, axiom, parameter relation or scope remark; validate each supplied parameter on its actual kernel and retain the complete formal hypotheses. This is not an empirical convergence theorem.",
            ),
        )
        checks = evidence.get(label, [])
        finite_passed = [c for c in checks if c["finite"] and c["passed"] is True]
        rows.append({
            **item,
            "status": status,
            "action_or_scope": action,
            "associated_numeric_checks": len(checks),
            "finite_checks_not_rejected": len(finite_passed),
            "failed_numeric_checks": sum(c["passed"] is False for c in checks),
            "whole_statement_empirically_certified": False,
            "estimates": [
                expressions[e] for e in item.get("formula_expression_ids", []) if e in expressions
            ],
            "evidence": checks,
        })
    result = {
        "chapter": 5,
        "current_source_sha256": inventory["source_sha256"],
        "inventory_counts": inventory["counts"],
        "statement_count": len(rows),
        "statement_status_counts": dict(Counter(r["status"] for r in rows)),
        "reports": retained,
        "rows": rows,
        "coverage_rule": "Numeric passing rows retain their exact scope. They never certify every input, extend a quadratic kernel to nonquadratic forces, or establish unresolved analytic hypotheses.",
        "regional_closure": {
            "existing_keystone": "SCK.6 supplies signed N-uniform pressure on its stated all-alive family; exact donor and kinetic residual remains.",
            "positive_fitness_source_drift": "Not proved for default unbounded Rastrigin h=.04; finite positive fitness and local moment decrease alone do not close it.",
            "source_pressure_owner": "Chapter4 completion ledger and donor/fitness analysis.",
            "tail_interface": "Chapter6 completion composes a certified selected-source moment drift with its native harmonic-plus-bounded-residual Gaussian tail law; absent drift is kept explicit.",
            "phase_transfer": "Regional discrepancy transfer matrix must bound transported errors, including conditioning/escape charges; empirical phase probabilities cannot be substituted.",
        },
    }
    (args.output / "statement-ledger.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# Chapter 5 statement-level estimate ledger",
        "",
        f"Current source SHA256: `{inventory['source_sha256']}`. Every formal item and its exact inventoried expressions is retained in [the JSON ledger](statement-ledger.json).",
        "",
        "A successful numeric check validates its listed specialization or statewise identity. The table keeps analytic prerequisites and native/continuous kernels explicit. Source snapshots in earlier datasets remain immutable.",
        "",
        "| Statement | Estimate status | Numeric checks (not rejected/associated) | Prediction, measured quantity and remaining hypotheses |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        label = row.get("source_label") or row["id"]
        lines.append(
            f"| `{label}` | {row['status'].replace('_', ' ')} | {row['finite_checks_not_rejected']}/{row['associated_numeric_checks']} | {row['action_or_scope']} |"
        )
    lines += [
        "",
        "The current positive-fitness unbounded Rastrigin regime has no completed global source-pressure/phase-transfer closure. The signed Keystone residual, supplied source-moment drift, escaped-region errors and tail sources must be bounded together before a complete N-uniform regional/global rate is reported. The empirical decreases and exact conditional native moment checks remain valid within their stated scopes.",
        "",
    ]
    (args.output / "statement-ledger.md").write_text("\n".join(lines))
    print(json.dumps({"statements": len(rows), "counts": result["statement_status_counts"]}))


if __name__ == "__main__":
    main()
