"""Validate Chapter 6 barrier and tail subestimates on retained native preparations."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import chi2, ncx2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/06_convergence.md"
BARRIER = r"\mathbb E[W_b'\mid\mathscr G]" + "\n" + r"\le B_d:=d\Lambda(a)q^{d-1}"
STATE = r"U_{d,s}(m)=\sum_{j=1}^d u_s(m_j)\prod_{\ell\ne j}p_s(m_\ell)"
TAIL = r"\mathbb P(\|X\|>R\mid\mathscr G)\le T_0(d,u)"
MASS = r"\mathbb P(\|X\|>R)\le\min\{1,M_x/R^p\}"
COST = r"\mathbb E[\|X\|^r\mathbf1_{\{\|X\|>R\}}]\le M_x/R^{p-r}."


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--barrier", type=Path, action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    text, comparisons, consumed = SOURCE.read_text(), [], []

    def add(name, observed, bound, label, formula, scope, inputs):
        if formula not in text:
            message = f"Current whole subexpression mismatch: {formula}"
            raise ValueError(message)
        tolerance = 128 * math.ulp(max(1.0, abs(observed), abs(bound)))
        comparisons.append({
            "id": name,
            "observed": float(observed),
            "bound": float(bound),
            "passed": bool(
                math.isfinite(observed) and math.isfinite(bound) and observed <= bound + tolerance
            ),
            "relation": "upper",
            "signed_residual": float(observed - bound),
            "binary64_roundoff_tolerance": tolerance,
            "source_labels": [label],
            "source_formula": formula,
            "scope": scope,
            "hypotheses": inputs,
        })

    for path in args.barrier:
        j = json.loads(path.read_text())
        consumed.append({
            "path": str(path.resolve()),
            "sha256": sha(path),
            "provenance": j.get("provenance"),
        })
        for number, row in enumerate(j["comparisons"]):
            inputs = {
                k: row.get(k)
                for k in [
                    "cemetery_convention",
                    "distribution",
                    "step",
                    "requested_independent_seeds",
                    "entering_survivors",
                    "survivors",
                    "delta",
                ]
            }
            prefix = f"{path.stem}_{number}"
            scope = (
                row["scope"]
                + " Original independent seed denominator; no unchanged-bound claim after survival conditioning."
            )
            add(
                prefix + "_prepared_vs_uniform",
                row["prepared_state_mean_upper"],
                row["sharp_bound"],
                "lem-convergence-dimension-log-barrier",
                BARRIER,
                scope,
                inputs,
            )
            # The recorded state-aware sum bounds EY given all independent preparations;
            # the accompanying second moment sums certify this simultaneous Cantelli slack.
            add(
                prefix + "_observed_vs_conditional_budget",
                row["observed_unconditional_barrier"],
                row["prepared_state_mean_upper"] + row["independent_seed_Cantelli_allowance"],
                "lem-convergence-state-log-barrier",
                r"\frac1M\sum_{r=1}^M W_b'^{(r)}"
                + "\n"
                + r"\le \frac1M\sum_{r=1}^M U^{(r)}"
                + "\n"
                + r" +\frac1M\sqrt{\frac{1-\delta}{\delta}\sum_{r=1}^M R^{(r)}}.",
                scope,
                {
                    **inputs,
                    "conditional_budget": row["prepared_state_mean_upper"],
                    "Cantelli_allowance": row["independent_seed_Cantelli_allowance"],
                },
            )
    native = json.loads(args.prepared.read_text())
    consumed.append({
        "path": str(args.prepared.resolve()),
        "sha256": sha(args.prepared),
        "upstream_input_indexes": native.get("input_indexes"),
    })
    seen = set()
    for number, row in enumerate(native["observations"]):
        key = (row["root"], row["case"], row["epoch"], row["step"])
        if key in seen:
            continue
        seen.add(key)
        n, d, p = row["N"], row["d"], row["prediction"]
        means = np.array(p["mean_positions"]).reshape(n, d)
        sigma = math.sqrt(p["tau_squared"])
        norms = np.linalg.norm(means, axis=1)
        scope = "Actual native prepared first-kick position means and independent isotropic OU/final position candidate Gaussian. One retained preparation per independent group. Exact Gaussian probability comparisons evaluated by SciPy binary64, not directed interval. No global native slow-zone contraction is inferred."
        inputs = {
            "native_root": row["root"],
            "archive": row["archive"],
            "archive_sha256": row["archive_sha256"],
            "seed": row["seed"],
            "step": row["step"],
            "N": n,
            "d": d,
            "sigma": sigma,
        }
        # Box centered at each actual preparation mean, half-width sigma.
        # Translation cancels; the real formula is checked on actual means/covariance.
        exact_box = (2 * ndtr(1.0) - 1.0) ** d
        lower = (2 / math.sqrt(2 * math.pi)) ** d * math.exp(-d / 2)
        box_formula = r"\mathbb P(X\in B\mid\mathscr G)\ge L_B"
        add(
            f"prepared_{number}_box",
            lower,
            exact_box,
            "lem-convergence-gaussian-regional-tail",
            box_formula,
            scope,
            {**inputs, "box": "m+[-sigma,sigma]^d"},
        )
        for shift in [1.0, 2.0, 4.0]:
            radii = norms + shift * sigma * math.sqrt(d)
            u = shift * shift * d
            bound = 1.0 if u <= d else math.exp((d - u + d * math.log(u / d)) / 2)
            exact = ncx2.sf((radii / sigma) ** 2, d, (norms / sigma) ** 2)
            add(
                f"prepared_{number}_tail_shift{shift}",
                float(np.max(exact)),
                bound,
                "lem-convergence-gaussian-regional-tail",
                TAIL,
                scope,
                {
                    **inputs,
                    "u": u,
                    "R_by_row": radii.tolist(),
                    "prepared_center_norms": norms.tolist(),
                },
            )
            # T_k tilted Gaussian tail is the same primitive driving weighted
            # native candidate budgets; compare its exact chi-square tail integral.
            for k in [1, 2, 4]:
                product = math.prod(d + 2 * j for j in range(k))
                exact_weighted = product * chi2.sf(u, d + 2 * k)
                upper = product * (
                    1.0
                    if u <= d + 2 * k
                    else math.exp((d + 2 * k - u + (d + 2 * k) * math.log(u / (d + 2 * k))) / 2)
                )
                # Complete T_k definition owned by the current statement.
                tk_formula = text[
                    text.index(r"T_k(d,u)=") : text.index("$$", text.index(r"T_k(d,u)="))
                ].strip()
                add(
                    f"prepared_{number}_weighted_gaussian_shift{shift}_k{k}",
                    exact_weighted,
                    upper,
                    "lem-convergence-gaussian-regional-tail",
                    tk_formula,
                    scope
                    + " Weighted quantity here is the standardized driving Gaussian primitive Y^k 1[Y>u], not an unobserved survivor-conditioned position moment.",
                    {**inputs, "u": u, "k": k},
                )
        # Exact finite marginal on these retained, permutation-invariant prepared
        # means: every moment is known, not a fitted unbounded-law moment constant.
        empirical_scope = "Exact finite uniform-row law of actual retained prepared position means; N-normalized moments, deterministic fixed finite-law Markov/Holder subestimate. It does not certify a population moment bound for the stochastic native future law."
        for order in [4.0, 8.0]:
            moment = float(np.mean(norms**order))
            for radius in [0.25, 1.0, 4.0]:
                outside = norms > radius
                add(
                    f"prepared_{number}_moment_mass_p{order}_R{radius}",
                    float(np.mean(outside)),
                    min(1.0, moment / radius**order),
                    "lem-convergence-moment-tail-transfer",
                    MASS,
                    empirical_scope,
                    {**inputs, "p": order, "M_x": moment, "R": radius},
                )
                for cost_order in [1.0, 2.0]:
                    add(
                        f"prepared_{number}_moment_cost_p{order}_r{cost_order}_R{radius}",
                        float(np.mean(norms**cost_order * outside)),
                        moment / radius ** (order - cost_order),
                        "lem-convergence-moment-tail-transfer",
                        COST,
                        empirical_scope,
                        {**inputs, "p": order, "r": cost_order, "M_x": moment, "R": radius},
                    )
    summary = {
        "comparisons": len(comparisons),
        "comparisons_failed": sum(not c["passed"] for c in comparisons),
        "native_preparation_groups": len(seen),
        "new_native_steps": 0,
    }
    report = {
        "chapter": 6,
        "title": "Actual native barrier, Gaussian preparation and finite marginal tail subestimates",
        "source_path": str(SOURCE),
        "source_sha256": sha(SOURCE),
        "helper_sha256": sha(Path(__file__)),
        "consumed_reports": consumed,
        "comparisons": comparisons,
        "summary": summary,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "source-snapshot.md").write_bytes(SOURCE.read_bytes())
    (args.output / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary))
    if summary["comparisons_failed"]:
        message = "Native barrier/tail subestimate discrepancy"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
