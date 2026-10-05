"""Keep every Chapter 9 expression and formal statement separately reviewable."""

import argparse
import collections
import hashlib
import json
import operator
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "docs/source/2_fractal_gas/convergence_program/09_propagation_chaos.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algebra_report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--constants", type=Path)
    parser.add_argument("--rate-review", type=Path)
    parser.add_argument("--horizon-review", type=Path)
    args = parser.parse_args()
    data = json.loads(args.algebra_report.read_text())
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != data["source_sha256"]:
        message = "Table must bind the current exact chapter source"
        raise ValueError(message)
    args.output.mkdir(parents=True, exist_ok=False)
    checks = data["checks"]
    by_label = collections.defaultdict(list)
    formal_ids = {item["id"] for item in data["formal_items"]}
    formal_order = sorted(data["formal_items"], key=operator.itemgetter("source_line"))
    for c in checks:
        owners = {c["source_label"]} if c["source_label"] in formal_ids else set()
        for expression in c["source_expressions"]:
            previous = [item for item in formal_order if item["source_line"] <= expression["line"]]
            if previous:
                owners.add(previous[-1]["id"])
        for owner in owners:
            by_label[owner].append(c)
    expressions = data["expression_ledger"]
    lines = [
        "# Chapter 9: constants, rates and complete source ledger",
        "",
        (
            "Every one of the 65 formal statements and every extracted source expression is listed. "
            "A formula has numerical evidence only when its own exact formula and operands are bound "
            "to a check. Conditional analytic hypotheses and infinite-law conclusions retain their "
            "own dispositions. They do not inherit a pass from another line of the theorem."
        ),
        "",
        f"Source SHA256: `{data['source_sha256']}`.",
        "",
        "| Statement | Measurement / estimate | Comparisons | Disposition |",
        "|---|---|---:|---|",
    ]
    formal_rows = []
    for item in data["formal_items"]:
        selected = by_label.get(item["id"], [])
        formula_ids = set(item.get("formula_expression_ids", []))
        own = [e for e in expressions if e["id"] in formula_ids]
        expressions_checked = sum(bool(e["check_ids"]) for e in own)
        if selected:
            experiments = sorted({c.get("kind", c["id"].split("-")[0]) for c in selected})
            measurement = ", ".join(experiments)
            status = "Exact comparisons under recorded hypotheses; analytical conclusions retained"
        else:
            measurement = "Definition, operator contract or analytical statement with its complete source hypotheses"
            status = "Analytical / definition; no empirical theorem certificate"
        row = {
            "id": item["id"],
            "title": item["title"],
            "source_line": item["source_line"],
            "comparison_count": len(selected),
            "failed": sum(not c["passed"] for c in selected),
            "exact_statement_expression_count": len(own),
            "exact_statement_expressions_with_comparisons": expressions_checked,
            "check_ids": [c["id"] for c in selected],
            "disposition": status,
            "statement": item["statement"],
            "dependencies": item["dependencies"],
        }
        formal_rows.append(row)
        title = item["title"].replace("|", "\\|")
        lines.append(
            f"| [{title}]({SOURCE}:{item['source_line']}) | {measurement} | {len(selected)} | {status} |"
        )
    if args.constants:
        constants = json.loads(args.constants.read_text())
        native = constants.get("native_population_comparisons")
        if native:
            exemplar = next(
                c
                for c in native["cases"]
                if c["profile"] == "rastrigin-well" and c["d"] == 1 and c["N"] == 32
            )
            c = exemplar["constants"]
            fields = [
                (
                    "κ_D, κ_C",
                    "Global positive companion weight floors",
                    "exp(−D_alg²/(2 width²))",
                    f"{c['input']['measurement_weight_lower']:.7g}, {c['input']['clone_weight_lower']:.7g}",
                ),
                ("C", "Uniform accepted-edge intensity", "2/(κ_C m*)", f"{c['c']:.7g}"),
                ("D_D", "Measurement self-exclusion budget", "2/(κ_D m*)", f"{c['d_d']:.7g}"),
                (
                    "L_q",
                    "One changed measurement normalizer",
                    "S*/(m*σ_s)+3S*³/(2m*σ_s³)",
                    f"{c['l_q']:.7g}",
                ),
                (
                    "H_s, L_0",
                    "Fitness sensitivity to diversity normalization",
                    "H_s=Lipschitz rescale/power budget; L_0=H_s L_q",
                    f"{c['h_s']:.7g}, {c['l_0']:.7g}",
                ),
                (
                    "L_a",
                    "Two-argument clipped clone gate",
                    "max{1/[s_c(F_*+ε)], (F*+ε)/[s_c(F_*+ε)²]}",
                    f"{c['l_a']:.7g}",
                ),
                (
                    "M_1, M_2, M_3",
                    "First, second, third collision-component moments",
                    "e²ᶜ, (1+2C)e⁴ᶜ, (1+6C+3C²)e⁸ᶜ",
                    f"exp({c['log_m1_c']:.7g}), exp({c['log_m2_c']:.7g}), exp({c['log_m3_c']:.7g})",
                ),
                (
                    "A_D",
                    "Squared changed-row influence",
                    "9M_2(2C)[(1+B)²+B]+max{1,4B²}",
                    f"exp({c['log_a_d']:.7g})",
                ),
                (
                    "A_φ",
                    "Full native conditional empirical variance",
                    "2‖φ‖∞²[A_D+10M_2(C)+1]",
                    f"exp({c['log_a_phi_unit']:.7g}) for ‖φ‖∞=1",
                ),
                (
                    "B*",
                    "Full finite population map bias",
                    "3M_1(2C)L_T√A_T+4L_T²A_T+64A²+16M_3(C)+√N_0",
                    f"exp({c['log_b_star']:.7g})",
                ),
                (
                    "C_upd",
                    "Mean-square empirical update error",
                    "√(A_φ+4‖φ‖∞²B*²)",
                    f"exp({c['log_c_update']:.7g}) for ‖φ‖∞=1",
                ),
            ]
            lines += [
                "",
                "## Configuration constants and rates",
                "",
                "Example: actual Rastrigin well probe, d=1, N=32. These configuration constants contain no N dependence; the displayed rates divide by N or √N. All native cases retain their own complete parameter records.",
                "",
                "| Constant | Controls | Expression | Example value |",
                "|---|---|---|---|",
            ]
            lines.extend(
                f"| {name} | {meaning} | {formula} | {value} |"
                for name, meaning, formula, value in fields
            )
            lines += [
                "",
                "| Estimate | Predicted population rate | What is measured |",
                "|---|---|---|",
                "| Conditional variance | A_φ/N | Variance over independently seeded complete native updates from one frozen input |",
                "| One-step population bias | 2‖φ‖∞B*/√N | Native empirical mean minus independent full rooted population mean at the same input law, with Monte Carlo uncertainty |",
                "| Mean-square one-step error | (A_φ+4‖φ‖∞²B*²)/N | Native empirical squared discrepancy from the population reference, with reference sampling floor retained |",
                "| Alive normalization | (|U−u|+|v−m|)/m | Physical alive tests divide by the actual surviving mass in each native update; no dead physical displacement enters |",
                "| Finite-tag product discrepancy | ℓ(ℓ−1)/N | Exact without-replacement versus with-replacement finite swarm integrals |",
                "| Alive-tag product discrepancy | 2δ_N+ℓ(ℓ−1)/(m*N) | Separate alive mass and survival conditioning; conditional proof inputs remain explicit |",
                "| Finite-horizon error | δ_N+ω(r)+D*e_n/r | Supplied continuity modulus and actual entering error; no unproved global contraction substituted |",
                "| Full-path survival transfer | H_N,T≤1−(1−δ_N)^T≤Tδ_N | Exact finite substochastic filtering and Gaussian safe-center fixtures; native hazards remain separately measured |",
                "",
            ]
            lines += [
                "",
                "## Actual Rust population scaling",
                "",
                (
                    "The finite update and the independent atomic population reference use the same "
                    "entering empirical law. One native update is one independent seed unit. Rooted "
                    "reference samples include the full accepted component and its shared Haar rotation "
                    "before BAOAB; capacity exhaustion errors instead of truncating or retrying. "
                    "Physical alive tests divide by their own alive mass. Dead positions are retained "
                    "only in the complete marked state required by the actual revival mechanism."
                ),
                "",
                "| Landscape | d | N | Native seed units | Root draws | sin conditional variance | N × variance | sin mean discrepancy | Theory log variance upper |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for case in native["cases"]:
                v = case["native_variance"][0]
                error = abs(case["native_mean"][0] - case["reference_mean"][0])
                log_upper = case["checks"][0]["analytic_bound"]["log_upper"]
                lines.append(
                    f"| {case['profile']} | {case['d']} | {case['N']} | {case['native_replicas']} | {case['root_draws']} | {v:.7g} | {v * case['N']:.7g} | {error:.7g} | {log_upper:.7g} |"
                )
            lines += [
                "",
                (
                    "The worst-case configuration constants can be much larger than the universal "
                    "bounded-observable limits. Such an upper-bound check confirms consistency but "
                    "does not show a sharp measured rate. The separate independent population-scaling "
                    "review reports empirical slopes and bootstrap uncertainty. Sampled maxima never "
                    "replace global kernel, fitness or alive-fraction hypotheses."
                ),
            ]
    if args.rate_review:
        rates = json.loads(args.rate_review.read_text())
        measured = [r for r in rates["slopes"] if r["status"] == "measured"]
        variance = [r for r in measured if r["metric"] == "variance"]
        mse = [r for r in measured if r["metric"] == "MSE"]
        # Saved implementations may use lowercase metric identifiers.
        if not mse:
            mse = [r for r in measured if r["metric"].lower() == "mse"]
        lines += [
            "",
            "## Measured decrease with population size",
            "",
            "All rates below use the actual frozen-input Rust updates and the independently integrated complete population reference, with 64 independent native seeds and 512 independent rooted samples per case. Different entering finite laws are retained separately, and reference Monte Carlo variance remains visible.",
            "",
            "| Statistic | Defined rate rows | N=128 / N=8 range | Measured log-slope range | Prediction |",
            "|---|---:|---:|---:|---|",
        ]
        for name, rows in [
            ("Empirical variance", variance),
            ("Mean-square population discrepancy", mse),
        ]:
            if rows:
                ratios = [r["endpoint_ratio_N128_over_N8"] for r in rows]
                slopes = [r["slope"] for r in rows]
                lines.append(
                    f"| {name} | {len(rows)} | {min(ratios):.6g}–{max(ratios):.6g} | {min(slopes):.6g}–{max(slopes):.6g} | Upper O(1/N); empirical slope can be faster, and MSE can include the finite reference floor |"
                )
        lines += [
            "",
            f"{rates['summary']['rate_rows_with_decreasing_endpoint']} of {rates['summary']['measured_rate_rows']} defined endpoint comparisons decrease. Deterministic zero-variance rows and rows lacking enough positive bootstrap replicates retain their own dispositions; no logarithm of zero is fitted.",
            "",
            f"[Full slopes, 2,000-replicate bootstrap intervals and saved native/root operands]({args.rate_review.resolve()}); [rate table]({args.rate_review.resolve().parent / 'rates.md'}).",
        ]
    if args.horizon_review:
        horizon = json.loads(args.horizon_review.read_text())
        lines += [
            "",
            "## Actual native trajectories through 16 updates",
            "",
            f"The retained dataset contains {horizon['summary']['trajectories']} independent native trajectories and {horizon['summary']['retained_native_updates']} complete updates. Empirical fluctuations are measured separately by parameter profile, dimension, initial landscape zone and side at steps 1 through 16; the independence denominator is the trajectory seed count within that stratum.",
            "",
            "These are literal finite-time variance measurements. They do not replace the exact one-step population reference with an invented iterated reference, and they do not estimate a uniform-time QSD or global contraction constant.",
            "",
            f"[All per-time-step observable variances and N=4/64 comparisons]({args.horizon_review.resolve()}).",
        ]
    lines += [
        "",
        "## Explicit scope of the remaining analytic statements",
        "",
        "Native evidence compares the active canonical whole-component Haar operator. The ordered-star priority formulas and rates have separate finite fixtures; the full Gaussian rate comparison uses constant positive fitness with zero clone gates. The full-map variation certificate also uses the zero-gate complete kernel and an exact maximal input coupling; it is an upper certificate, rather than a measured full-TV distance for the active selected collision map.",
        "",
        f"[Every residual finite estimate and its specific reason]({args.algebra_report.resolve().parent / 'finite-estimate-residuals.md'}). All source-expression dispositions are retained individually, including finite prefixes whose infinite-limit clause stays analytic.",
    ]
    lines += [
        "",
        "## Saved evidence",
        "",
        f"- [{data['summary']['checks']} exact comparisons]({args.algebra_report.resolve()}).",
        f"- [Individual expression dispositions]({args.algebra_report.resolve().parent / 'expression-ledger.json'}).",
        (
            "- Full source statements, dependencies and per-expression check references are retained "
            "in the JSON tables alongside this document."
        ),
        "",
        (
            "Finite native observations and explicit finite-law fixtures do not establish a native "
            "QSD eigenvalue, joint LSI/Poincare constant, uniqueness of a nonlinear stationary population "
            "law or uniform global attraction. Those statements retain the exact additional conditions "
            "in the chapter; the quantitative finite-step operators and estimates are tested separately."
        ),
    ]
    (args.output / "chapter09-estimates-table.md").write_text("\n".join(lines) + "\n")
    (args.output / "estimates.md").write_text("\n".join(lines) + "\n")
    (args.output / "formal-table.json").write_text(json.dumps(formal_rows, indent=2) + "\n")
    (args.output / "expression-table.json").write_text(json.dumps(expressions, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(data["summary"], indent=2) + "\n")
    provenance = {
        "source_sha256": data["source_sha256"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in [
                args.algebra_report,
                args.constants,
                args.rate_review,
                args.horizon_review,
            ]
            if path is not None
        ],
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps({"formal_rows": len(formal_rows), "expression_rows": len(expressions)}))


if __name__ == "__main__":
    main()
