"""Check native Chapter 6 displacement budgets on retained prepared first kicks.

Squared realized displacement needs a cross moment not retained in this ledger.
We check its exact conditional budget, the resulting conditional variance bound,
and independent-seed final candidate variance using its exact Gaussian variance.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/06_convergence.md"
RAW_FORMULA = (
    r"\widehat D_h^2(\mathscr G_1)"
    "\n =" + r"\frac{c_h^2}{N}\sum_{i\in A}\|v_{1,i}\|^2"
    "\n  +" + r"\frac{|A|}{N}d\tau_h^2,"
)
CENT_FORMULA = (
    r"\widehat D_{h,\mathrm{cent}}^2(\mathscr G_1)"
    "\n =" + r"c_h^2 V_v(v_1;A)"
    "\n  +" + r"\frac{(|A|-1)_+}{N}d\tau_h^2,"
)
KINETIC_FORMULA = r"P_KV_x\le(1+\theta)V_x+(1+\theta^{-1})D_h^2."
CAP_QUOTES = [r"P_KV_v\le v_{\max}^2", r"P_KE_v\le v_{\max}^2"]
SCOPE = (
    "Actual retained standard-native prepared first kicks, before independent OU/final position "
    "Gaussians; all candidate rows and original N normalization. Candidate centered variance "
    "upper-dominates any terminal alive subset observable normalized by N, including cemetery "
    "zero. Conditional deterministic identities and Gaussian innovation variances are checked; "
    "realized squared displacement is not inferred from two marginal moments. No affine "
    "P_C Dh^2 closure or positional contraction rate is inferred from samples."
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text = SOURCE.read_text()
    for formula in [RAW_FORMULA, CENT_FORMULA, KINETIC_FORMULA]:
        if formula not in text:
            message = f"Source formula mismatch: {formula}"
            raise ValueError(message)
    data = json.loads(args.input.read_text())
    groups, comparisons, operands = defaultdict(list), [], []

    def add(name, observed, bound, labels, formula=None, quotes=None, **extra):
        tolerance = 64 * math.ulp(max(1.0, abs(observed), abs(bound)))
        passed = math.isfinite(observed) and math.isfinite(bound) and observed <= bound + tolerance
        comparisons.append({
            "id": name,
            "observed": observed,
            "bound": bound,
            "relation": "upper",
            "signed_residual": observed - bound,
            "passed": passed,
            "binary64_roundoff_tolerance": tolerance,
            "source_labels": labels,
            "source_formula": formula,
            "source_quotes": quotes,
            "scope": SCOPE,
            **extra,
        })

    for number, row in enumerate(data["observations"]):
        p, n, d = row["prediction"], row["N"], row["d"]
        c, t = p["b"], p["tau_squared"]
        raw = c * c * p["first_kick_velocity"]["total"] + d * t
        centered = c * c * p["first_kick_velocity"]["variance"] + (1 - 1 / n) * d * t
        cap = row["checks"]["postcap_bound"]
        for field in ["postcap_variance", "postcap_barycenter"]:
            add(
                f"{number}_{field}",
                row["checks"][field],
                cap,
                ["lem-convergence-capped-displacement"],
                quotes=[
                    r"V_v(S')\le v_{\max}^2"
                    if field == "postcap_variance"
                    else r"E_v(S')\le v_{\max}^2."
                ],
                applicability="Terminal velocity cap only; no pathwise native positional cap premise.",
            )
        add(
            f"{number}_terminal_velocity_moments",
            max(row["checks"]["postcap_variance"], row["checks"]["postcap_barycenter"]),
            cap,
            ["prop-velocity-rate-explicit"],
            quotes=[r"P_KV_v,P_KE_v\le v_{\max}^2"],
            applicability="Terminal native cap supplies both velocity moments independently of continuum friction or positional bounds.",
        )
        for theta in [0.25, 1.0, 4.0]:
            bound = (1 + theta) * p["input"]["variance"] + (1 + 1 / theta) * centered
            add(
                f"{number}_conditional_variance_theta{theta}",
                p["variance_prediction"],
                bound,
                ["cor-convergence-displacement-moment"],
                KINETIC_FORMULA,
                hypotheses={
                    "theta": theta,
                    "budget": centered,
                    "budget_formula": CENT_FORMULA,
                    "preparation": p["hypotheses"],
                    "N": n,
                    "d": d,
                },
            )
        add(
            f"{number}_centered_budget_le_raw",
            centered,
            raw,
            ["cor-convergence-displacement-moment"],
            quotes=[RAW_FORMULA, CENT_FORMULA],
        )
        variance_of_candidate_variance = (
            2 * t * t * d * (n - 1) / (n * n) + 4 * t * p["deterministic"]["variance"] / n
        )
        group = (row["root"], row["case"], row["epoch"], row["step"])
        groups[group].append((
            row["seed"],
            row["actual"]["variance"],
            p["variance_prediction"],
            variance_of_candidate_variance,
        ))
        operands.append({
            "source_observation": number,
            "N": n,
            "d": d,
            "c_h": c,
            "tau_squared": t,
            "input_variance": p["input"]["variance"],
            "first_kick_total": p["first_kick_velocity"]["total"],
            "first_kick_variance": p["first_kick_velocity"]["variance"],
            "raw_displacement_budget": raw,
            "centered_displacement_budget": centered,
            "conditional_candidate_variance": p["variance_prediction"],
            "variance_of_candidate_variance": variance_of_candidate_variance,
            "native_archive": {
                k: row[k]
                for k in [
                    "root",
                    "archive",
                    "archive_sha256",
                    "case",
                    "epoch",
                    "step",
                    "replicate",
                    "seed",
                ]
            },
        })
    alpha = 0.01 / len(groups)
    for number, (group, rows) in enumerate(sorted(groups.items())):
        if len({r[0] for r in rows}) != len(rows):
            message = f"Duplicated independent seed in {group}"
            raise ValueError(message)
        # Conditioning jointly on every independent seed's prepared state keeps
        # random per-seed budgets valid. No independence between times assumed.
        m = len(rows)
        predicted = sum(r[2] for r in rows) / m
        allowance = math.sqrt(sum(r[3] for r in rows) * (1 - alpha) / alpha) / m
        add(
            f"independent_seed_candidate_variance_{number}",
            sum(r[1] for r in rows) / m,
            predicted + allowance,
            ["cor-convergence-displacement-moment"],
            KINETIC_FORMULA,
            hypotheses={
                "group": list(group),
                "independent_seed_units": m,
                "alpha": alpha,
                "simultaneous_family_error_budget": 0.01,
                "conditional_mean": predicted,
                "one_sided_Cantelli_allowance": allowance,
                "conditional_variance_formula": "2 tau^4 d(N-1)/N^2 + 4 tau^2 V(m)/N",
                "scope": "Conditional Gaussian candidate variance, all slots; not survivor-conditioned moments.",
            },
        )
    summary = {
        "retained_native_stages": len(operands),
        "independent_groups": len(groups),
        "comparisons": len(comparisons),
        "comparisons_failed": sum(not c["passed"] for c in comparisons),
        "new_native_steps": 0,
        "realized_squared_displacement_comparisons": 0,
    }
    report = {
        "chapter": 6,
        "title": "Native prepared-state displacement and positional moment budget",
        "source_path": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "scope": SCOPE,
        "provenance": {
            "input": str(args.input.resolve()),
            "input_sha256": sha(args.input),
            "helper_sha256": sha(Path(__file__)),
            "upstream_input_indexes": data.get("input_indexes"),
            "raw_archive_hash_status": "Hashes retained from previously deep-verified Ch5 native archive ledger.",
        },
        "source_formulas": [RAW_FORMULA, CENT_FORMULA, KINETIC_FORMULA],
        "operands": operands,
        "comparisons": comparisons,
        "summary": summary,
        "required_remaining_math": [
            "Same-kernel affine P_C Dh^2 <= a_D Vx+b_D for a global positional rate; no terminal-cap replacement."
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "source-snapshot.md").write_bytes(SOURCE.read_bytes())
    (args.output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))
    if summary["comparisons_failed"]:
        message = "Native displacement estimate discrepancy"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
