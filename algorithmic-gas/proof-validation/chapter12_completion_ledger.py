"""Source-exact Chapter 12 finite evidence and analytic limit dispositions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(root: Path, entries: list[dict], tag: str):
    entry = next(e for e in entries if e["tag"] == tag)
    path = root / entry["path"]
    assert sha(path) == entry["sha256"], path
    return json.loads(gzip.decompress(path.read_bytes()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_result", type=Path)
    parser.add_argument("empty_output", type=Path)
    parser.add_argument("--nonquadratic-reference", type=Path)
    args = parser.parse_args()
    root = args.native_result.resolve()
    output = args.empty_output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not any(output.iterdir()), "Existing datasets are immutable"
    repo = Path(__file__).resolve().parents[1]
    inventory_path = Path(__file__).with_name("chapter12_inventory.json")
    inventory = json.loads(inventory_path.read_text())
    source = repo / inventory["source_path"]
    assert sha(source) == inventory["source_sha256"], "Regenerate inventory after source changes"
    report = json.loads((root / "report.json").read_text())
    assert report["status"] == "complete" and report["failed"] == 0
    assert report["source_sha256"] == sha(source)
    index_path = root / "archive-index.json"
    index = json.loads(index_path.read_text())
    entries = index["entries"]
    checks = load_artifact(root, entries, "all-checks")
    refs = load_artifact(root, entries, "exact-reference-operands")
    comparisons = []

    def compare(name, lhs, rhs, operands, equality=False):
        allowance = 1e-9 * max(1.0, abs(lhs), abs(rhs))
        passed = abs(lhs - rhs) <= allowance if equality else lhs <= rhs + allowance
        comparisons.append({
            "name": name,
            "lhs": lhs,
            "rhs": rhs,
            "equality": equality,
            "allowance": allowance,
            "passed": passed,
            "operands": operands,
        })

    # Independent algebra evaluation of finite operands already generated in Rust.
    for row in refs["bounded_joint_tilts"]:
        n, q, beta = row["N"], row["q"], row["beta"]
        p = row["count_probabilities"]
        mean = math.fsum(s / n * v for s, v in enumerate(p))
        second = math.fsum((s / n) ** 2 * v for s, v in enumerate(p))
        var = second - mean**2
        mse = math.fsum((s / n - q) ** 2 * v for s, v in enumerate(p))
        entropy = math.fsum(
            v * (beta * (s / n - q) ** 2 - math.log(row["normalizer"])) for s, v in enumerate(p)
        )
        ag = 4 * (beta + math.log(2) / 2)
        diagonal = mean * (1 - mean)
        distinct = math.fsum(s * (s - 1) / (n * (n - 1)) * v for s, v in enumerate(p)) - mean**2
        compare("count-law normalization", math.fsum(p), 1.0, row, True)
        compare("exact total entropy", entropy, row["total_relative_entropy"], row, True)
        compare("N-independent joint tilt entropy budget", entropy, beta, row)
        compare(
            "empirical bounded-tilt variance A/N", var, ag / n, {"N": n, "A": ag, "beta": beta}
        )
        compare(
            "distinct covariance with diagonal",
            distinct,
            (n * var - diagonal) / (n - 1),
            row,
            True,
        )
        compare(
            "general g,h covariance bound",
            abs(distinct),
            (ag + diagonal) / (n - 1),
            {
                "N": n,
                "A_g": ag,
                "A_h": ag,
                "diagonal": diagonal,
                "g": "Bernoulli identity",
                "h": "identity or its negative",
            },
        )
        compare("variance below MSE", var, mse, row)
        compare("bounded marginal variance", diagonal, 1.0, row)
    for row in refs["explicit_killed_qsd"]:
        alpha = row["alpha"]
        mass = row["mass"]
        compare("killed survival factor", math.fsum(v * alpha for v in mass), alpha, row, True)
        compare("QSD mass normalization", math.fsum(mass), 1.0, row, True)
    # General g,h covariance uses complete native physical swarms, not row independence.
    orbit_rows = []
    rates = []
    for case in report["cases"]:
        n = case["configuration"]["walkers"]
        vectors = []
        sampling_rates = {}
        for record in case["sampling_paths"]:
            path = root / record["sampling"]
            entry = next(e for e in entries if e["path"] == record["sampling"])
            assert sha(path) == entry["sha256"]
            sample = json.loads(gzip.decompress(path.read_bytes()))
            pop = sample["physical_population"]
            field = pop["observations"]["fields"]["positions"]
            values = field["values"]
            d = len(values) // n
            g = [math.sin(values[i * d]) for i in range(n)]
            h = [math.cos(values[i * d]) for i in range(n)]
            vectors.append((g, h))
            for row in sample["rows"]:
                group = sampling_rates.setdefault(row["k"], [0, 0, 0.0])
                group[0] += sum(s["collision"] for s in row["samples"])
                group[1] += len(row["samples"])
                group[2] += math.fsum(
                    s["product_with"] - s["product_without"] for s in row["samples"]
                )
                assert all(len(set(s["without"])) == row["k"] for s in row["samples"])
                assert all(s["collision"] or s["with"] == s["without"] for s in row["samples"])
        r = len(vectors)
        mg = math.fsum(math.fsum(g) / n for g, h in vectors) / r
        mh = math.fsum(math.fsum(h) / n for g, h in vectors) / r
        empirical = (
            math.fsum(math.fsum(g) / n * math.fsum(h) / n for g, h in vectors) / r - mg * mh
        )
        diagonal = (
            math.fsum(math.fsum(a * b for a, b in zip(g, h)) / n for g, h in vectors) / r - mg * mh
        )
        distinct = (
            math.fsum(
                (math.fsum(g) * math.fsum(h) - math.fsum(a * b for a, b in zip(g, h)))
                / (n * (n - 1))
                for g, h in vectors
            )
            / r
            - mg * mh
        )
        operands = {
            "case": case["case"],
            "N": n,
            "independent_native_swarms": r,
            "g": "sin(x_1)",
            "h": "cos(x_1)",
            "empirical_covariance": empirical,
            "diagonal_covariance": diagonal,
            "distinct_covariance": distinct,
            "physical_vectors": vectors,
        }
        compare(
            "native orbit general covariance identity",
            distinct,
            (n * empirical - diagonal) / (n - 1),
            operands,
            True,
        )
        orbit_rows.append(operands)
        for k, (collisions, count, delta) in sampling_rates.items():
            collision = -math.expm1(math.fsum(math.log1p(-i / n) for i in range(k)))
            falling = math.prod(range(n - k + 1, n + 1))
            compare(
                "falling-factorial collision identity",
                collision,
                1 - falling / n**k,
                {"N": n, "k": k, "falling_factorial": str(falling)},
                True,
            )
            rates.append({
                "case": case["case"],
                "N": n,
                "d": case["configuration"]["dimensions"],
                "k": k,
                "exact_collision_probability": collision,
                "measured_collision_fraction": collisions / count,
                "independent_index_lists": count,
                "product_expectation_difference": delta / count,
                "union_bound": min(1, k * (k - 1) / (2 * n)),
            })
    nonquadratic = []
    if args.nonquadratic_reference:
        density_root = args.nonquadratic_reference.resolve()
        density_report = json.loads((density_root / "report.json").read_text())
        assert density_report["comparisons_failed"] == 0
        density_index = json.loads((density_root / "archive-index.json").read_text())
        for entry in density_index["entries"]:
            if not entry["tag"].startswith("nonquadratic-"):
                continue
            path = density_root / entry["path"]
            assert sha(path) == entry["sha256"]
            value = json.loads(gzip.decompress(path.read_bytes()))
            parameters = value["parameters"]
            theta = parameters["theta"]
            curvature = parameters["kappa"]
            amplitude = parameters["amplitude"]
            expected = max(theta, theta / curvature * math.exp(2 * amplitude / theta))
            calculated = value["constants"]["lsi"]
            compare(
                "nonquadratic one-coordinate bounded-perturbation LSI constant",
                calculated,
                expected,
                parameters,
                True,
            )
            nonquadratic.append({
                "tag": entry["tag"],
                "archive": str(path),
                "archive_sha256": entry["sha256"],
                "parameters": parameters,
                "coordinate_reference_LSI": calculated,
                "dimension_and_population_scope": value["dimension_and_population_scope"],
                "scope": "Explicit separable confining Gibbs-relative density. Coordinate perturbation cost exp(2 amplitude/theta) is paid before tensorization in d and N; native selected QSD LSI remains conditional.",
            })
        assert len(nonquadratic) == 72
    failed = sum(not c["passed"] for c in comparisons)
    assert failed == 0, "Finite algebra regression failed"
    (output / "supplementary-finite-operands.json").write_text(
        json.dumps(
            {"comparisons": comparisons, "native_general_covariances": orbit_rows, "rates": rates},
            indent=2,
        )
    )
    # Every source expression has an explicit disposition; numeric evidence does not
    # replace the assumptions of the unknown native QSD or any limiting assertion.
    mapping = {
        1: [
            "native final",
            "native measurement categorical",
            "native cloning categorical",
            "killed kernel equivariance",
        ],
        2: ["killed kernel equivariance"],
        4: ["killed QSD eigenfactor"],
        5: ["killed QSD eigenfactor"],
        6: ["killed QSD eigenfactor", "killed kernel equivariance"],
        10: ["exact physical empirical-orbit TV", "union bound"],
        11: ["union bound"],
        12: ["exact physical empirical-orbit TV"],
        15: ["native empirical-orbit covariance"],
        16: ["native empirical-orbit covariance"],
        22: ["exact empirical covariance identity", "native empirical-orbit covariance"],
        23: ["entropy MSE"],
        24: ["entropy MSE"],
        25: ["distinct covariance entropy"],
        26: ["native empirical-orbit covariance"],
        27: ["entropy MSE"],
        29: ["entropy MSE", "variance below MSE"],
        30: ["distinct covariance entropy"],
        33: ["Hoeffding log moment"],
        34: ["square exponential moment"],
        35: ["entropy variational inequality"],
        36: ["tilted-reference KL nonnegativity"],
        38: ["variance below MSE"],
        39: ["zero bounded observable"],
        40: ["Gaussian KL-Fisher convention"],
        41: ["Gaussian KL-Fisher convention"],
        42: ["one-coordinate marginal Gaussian LSI"],
        44: ["square-root density Fisher factor"],
        45: ["OU limiting covariance"],
        46: ["Gaussian KL-Fisher convention"],
        47: ["OU limiting covariance"],
        48: ["OU limiting covariance"],
        49: ["OU limiting covariance", "native stage"],
        50: ["OU limiting covariance"],
        51: ["one-coordinate marginal Gaussian LSI"],
        52: ["one-coordinate marginal Gaussian LSI"],
        53: ["one-coordinate marginal Gaussian LSI"],
    }
    explanations = {
        1: "Complete native current-donor transition replay transports independent random fields; exact native categorical kernel law also compared. Reference killed kernel has explicit symmetry.",
        2: "Exact unique rank-one killed reference QSD mass invariant under any permutation (mass depends only on count); actual native QSD uniqueness remains an analytic hypothesis.",
        4: "Strictly positive explicit alpha retained in killed reference eigenmeasure, supplementary survival sum checked.",
        10: "Full physical tuple TV computed exactly on distinct native position atoms and checked by coupled independent index-list experiments. Collision and union bound normalized as probabilities.",
        15: "Barycenter identity for finite permutation-orbit laws from physical clouds; direct uniform index law makes marginal and empirical mean identical.",
        16: "Finite uniform orbit averages preserve the exact barycenter, without any row-independence assumption.",
        22: "General sin/cos covariance independently recomputed from complete native physical clouds; exact bounded reference count-law identity also tested.",
        23: "N-independent A=4(beta+log(2)/2) from bounded total reference joint entropy; evaluated N2–512. Native empirical variance recorded, its global 1/N hypothesis is separately conditional.",
        24: "Same N-independent reference budget for identity/negative bounded observable; no data-fitted N-dependent constant used as a theorem prediction.",
        25: "Exact reference covariance budget including diagonal term, plus independent native general covariance identity. Uniform native variance premise not inferred from finite data.",
        27: "Total KL evaluated exactly on finite bounded joint Bernoulli tilts, never estimated by treating atomic swarms as continuous densities.",
        40: "Exact Gaussian KL/Fisher of a normalized exponential density tilt. Joint law is the conservative product reference, not the native selected QSD.",
        41: "C=max(theta,theta/curvature) independent of N and d, evaluated N1–128,d1–8. Bounded tilt/curvature criteria for other laws remain their stated analytic conditions.",
        42: "Gaussian reference entropy/Fisher convention and single-coordinate gradient test; only finite reference transfer gets numeric credit.",
        46: "Conservative quadratic Gibbs density factorization with velocity theta and spatial theta/curvature; normalization/reference role retained.",
        48: "Frozen OU reference only; nonlinear selected stationary/QSD law not identified with this reference.",
        49: "Frozen OU transition covariance and deterministic native O stage transport checked; integral representation valid analytically by variation of constants.",
        52: "Finite Gaussian product-to-marginal inequality checked; passing to an arbitrary weak marginal limit is the chapter's analytic proof, not an empirical claim.",
    }
    definitions = {3, 7, 8, 9, 13, 14, 17, 18, 20, 21, 28, 31, 32, 37, 43, 54}
    rows = []
    analytic_clauses = {1, 2, 19, 23, 24, 25, 40, 41, 42, 46, 48, 49, 51, 52, 53}
    for expression in inventory["quantitative_expressions"]:
        number = int(expression["id"].rsplit("-", 1)[1])
        prefixes = mapping.get(number, [])
        evidence = [
            {
                "artifact": "all-checks",
                "row": i,
                "expression": c["expression"],
                "scope": c["scope"],
            }
            for i, c in enumerate(checks)
            if any(c["expression"].startswith(p) for p in prefixes)
        ]
        if number == 19:
            status = "analytic_limit"
            note = "Weak-in-probability deterministic empirical convergence implies fixed-k chaos by the complete Polish-space product-continuity and finite-mixture proof. No finite experiment establishes the limiting hypothesis."
        elif number in definitions:
            status = "definition_or_hypothesis"
            note = "Source definition/domain condition retained exactly; finite-law operands and status masks provide the operational instances. It is not counted as a numerical theorem pass."
        else:
            assert evidence, (number, expression["formula"])
            status = (
                "finite_reference_and_native_evidence"
                if any("native" in e["expression"] for e in evidence)
                else "finite_reference_evidence"
            )
            note = explanations.get(
                number,
                "Exact finite reference-law expression and independent whole-experiment operands retained. Numeric checks retain their declared law and do not certify native global QSD, LSI or asymptotic hypotheses.",
            )
        rows.append(
            dict(
                **expression,
                status=status,
                explanation=note,
                evidence=evidence,
                analytic_clause_also_present=number in analytic_clauses,
            )
        )
    formals = []
    for formal in inventory["formal_items"]:
        sub = [r["id"] for r in rows if r.get("formal_item_id") == formal["id"]]
        formals.append(
            dict(
                **formal,
                expression_rows=sub,
                disposition="complete source audit; finite evidence and analytic hypotheses separately scoped",
            )
        )
    summary = {
        "chapter": 12,
        "helper_source_sha256": sha(Path(__file__)),
        "python_executable_sha256": sha(Path(sys.executable).resolve()),
        "source_sha256": sha(source),
        "inventory_sha256": sha(inventory_path),
        "native_report_sha256": sha(root / "report.json"),
        "input_archive_index_sha256": sha(index_path),
        "expressions": len(rows),
        "formal_items": len(formals),
        "native_checks": report["checks"],
        "supplementary_comparisons": len(comparisons),
        "failed": failed + report["failed"],
        "fresh_full_native_updates": report["fresh_full_native_updates"],
        "nonquadratic_reference_LSI_laws": len(nonquadratic),
        "independent_index_lists": report["independent_index_lists"],
        "dispositions": {
            s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})
        },
        "scope": "Native full physical transition and conditional empirical-cloud sampling; exact finite killed/Gaussian/count references; unknown native QSD/LSI and limit assumptions remain analytic.",
    }
    (output / "expression-ledger.json").write_text(
        json.dumps({"summary": summary, "expressions": rows, "formal_items": formals}, indent=2)
    )
    (output / "execution-source.py").write_text(Path(__file__).read_text(encoding="utf-8"))
    (output / "report.json").write_text(json.dumps(summary, indent=2))
    table = [
        "# Chapter 12: exact formulas, operands and scope",
        "",
        f"{len(rows)} expressions; {len(formals)} formal items; {report['checks']} native/reference comparisons and {len(comparisons)} independent algebra comparisons; zero failures.",
        "",
        "| Source expression | Formula | Validation | Scope |",
        "|---|---|---|---|",
    ]
    for r in rows:
        formula = r["formula"].replace("|", "\\|").replace("\n", " ")
        table.append(
            f"| {r['id']} (line {r['source_line']}) | `{formula}` | {r['status']}; {len(r['evidence'])} operand rows | {r['explanation']} |"
        )
    table += [
        "",
        "All numeric swarm observables are averages or probabilities. Array addresses identify archived operands and innovation coordinates; physical swarms are permutation invariant.",
        "",
        "Fresh native updates include identity replay and transported reversal replay; their stored original inputs are retained runs. Gaussian/KL/QSD reference results have their own explicit laws. Status entropy is not replaced by a continuous gradient inequality.",
    ]
    (output / "expressions-table.md").write_text("\n".join(table) + "\n")
    rate_table = [
        "# Conditional empirical-mixture sampling rates",
        "",
        "Independent index-list sampling on actual complete native outputs. This table measures finite sampling correction; it does not estimate an unknown QSD or prove an infinite-limit premise.",
        "",
        "| Case | N | d | k | Exact collision / TV (distinct physical atoms) | Measured collisions | Union bound | Index lists |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rates:
        rate_table.append(
            f"| {r['case']} | {r['N']} | {r['d']} | {r['k']} | {r['exact_collision_probability']:.8g} | {r['measured_collision_fraction']:.8g} | {r['union_bound']:.8g} | {r['independent_index_lists']} |"
        )
    (output / "sampling-rates.md").write_text("\n".join(rate_table) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
