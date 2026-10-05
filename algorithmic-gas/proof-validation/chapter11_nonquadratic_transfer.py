"""Transfer Chapter10's retained Rust nonquadratic operands to exact grid-law Hellinger checks.

The Simpson weights define an explicit finite quadrature law here. Numerical
identities are exact for that declared law; refinement tolerances do not become
certified continuous-law entropy or transport intervals. Global Gaussian tail
bounds and the continuum LSI/rate constants remain separately identified.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

from chapter11_completion_ledger import packed, sha, w2
import numpy as np


def comparison(name, ids, left, right, identity, operands):
    tolerance = 2e-10 * (1 + abs(left) + abs(right))
    return {
        "id": name,
        "expression_ids": [f"chapter11-expression-{x:04}" for x in ids],
        "lhs": left,
        "rhs": right,
        "relation": "=" if identity else "<=",
        "tolerance": tolerance,
        "passed": abs(left - right) <= tolerance if identity else left <= right + tolerance,
        "operands": operands,
        "scope": "Exact declared finite quadrature probability law; continuum LSI/tail hypotheses are recorded separately, not transferred silently to a discrete gradient operator.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter10_dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    root = args.chapter10_dataset.resolve()
    report = json.loads((root / "report.json").read_text())
    index = json.loads((root / "archive-index.json").read_text())
    if index["status"] != "complete":
        message = "Complete immutable Rust operand archive required"
        raise ValueError(message)
    entries = {e["path"]: e for e in index["entries"]}
    rows = []
    manifests = []
    total = failures = 0
    for case in report["cases"]:
        if not case["id"].startswith("nonquadratic-"):
            continue
        source = root / case["archive"]
        if sha(source) != entries[case["archive"]]["sha256"]:
            message = "Rust nonquadratic source artifact checksum mismatch"
            raise ValueError(message)
        with gzip.open(source, "rt") as file:
            operands = json.load(file)
        x = np.array(operands["x_quadrature_operands"], dtype=float)
        v = np.array(operands["v_quadrature_operands"], dtype=float)
        qx = x[:, 1] * x[:, 2]
        qv = v[:, 1] * v[:, 2]
        qx /= qx.sum()
        qv /= qv.sum()
        parameters = operands["parameters"]
        constants = operands["constants"]
        a, b, z = parameters["tilt"]
        shift = parameters["linear_position_tilt"]
        relative_log = (
            a * np.sin(x[:, 0, None])
            + b * np.sin(v[None, :, 0])
            + z * np.sin(x[:, 0, None]) * np.sin(v[None, :, 0])
            + shift * x[:, 0, None]
        )
        q = qx[:, None] * qv[None, :]
        maximum = float(relative_log.max())
        p_unnormalized = q * np.exp(relative_log - maximum)
        normalizer = float(p_unnormalized.sum())
        p = p_unnormalized / normalizer
        log_normalizer = math.log(normalizer) + maximum
        log_ratio = relative_log - log_normalizer
        entropy = float(np.sum(p * log_ratio))
        affinity = float(np.sum(np.sqrt(p * q)))
        px, pv = p.sum(axis=1), p.sum(axis=0)
        projected_transport = w2(x[:, 0], px, x[:, 0], qx) + w2(v[:, 0], pv, v[:, 0], qv)
        theta, kappa = parameters["theta"], parameters["kappa"]
        amplitude, frequency = parameters["amplitude"], parameters["frequency"]
        lsi = max(theta, theta / kappa * math.exp(2 * amplitude / theta))
        hessian_bound = kappa + amplitude * frequency**2
        kinetic_rate = constants["eta"] / (lsi / 2 + 3 * constants["eta"])
        mean_x, sigma_x, sigma_v = (
            shift * theta / kappa,
            math.sqrt(theta / kappa),
            math.sqrt(theta),
        )
        radius = parameters["radius"]
        envelope = math.exp(2 * (abs(a) + abs(b) + abs(z)) + 2 * amplitude / theta)

        def normal_tail(u):
            return math.erfc(u / math.sqrt(2)) / 2

        mixed_tilt_factor = math.exp(2 * (abs(a) + abs(b) + abs(z)))
        position_barrier_factor = math.exp(2 * amplitude / theta)
        position_tail = normal_tail((radius - mean_x) / sigma_x) + normal_tail(
            (radius + mean_x) / sigma_x
        )
        velocity_tail = 2 * normal_tail(radius / sigma_v)
        tail = mixed_tilt_factor * (position_barrier_factor * position_tail + velocity_tail)
        checks = []
        checks.append(
            comparison(
                "Rust-entropy-operand",
                [],
                entropy,
                operands["H"],
                True,
                {
                    "Rust_H": operands["H"],
                    "declared_grid_KL": entropy,
                    "log_relative_Z": log_normalizer,
                },
            )
        )
        checks.append(
            comparison(
                "global_LSI_constant",
                [],
                lsi,
                constants["lsi"],
                True,
                {
                    "theta": theta,
                    "kappa": kappa,
                    "A": amplitude,
                    "coordinatewise_tensorization": True,
                },
            )
        )
        checks.append(
            comparison(
                "global_Hessian_constant",
                [],
                hessian_bound,
                constants["hessian_bound"],
                True,
                {"kappa": kappa, "A": amplitude, "frequency": frequency},
            )
        )
        checks.append(
            comparison(
                "kinetic_entropy_rate",
                [],
                kinetic_rate,
                constants["rate"],
                True,
                {"eta": constants["eta"], "C": lsi},
            )
        )
        dimension_rows = []
        for dimension in [1, 2, 4, 8]:
            product_entropy = dimension * entropy
            product_affinity = affinity**dimension
            shape_h = 2 * (1 - product_affinity)
            m, n = 0.6, 0.9
            full_h = m + n - 2 * math.sqrt(m * n) * product_affinity
            root_mass = (math.sqrt(m) - math.sqrt(n)) ** 2
            point = {
                "physical_coordinates": dimension,
                "KL": product_entropy,
                "affinity": product_affinity,
                "H_squared": shape_h,
                "m": m,
                "n": n,
                "m0": m,
                "m1": n,
                "C": lsi,
                "conditional_kinetic_rate": kinetic_rate,
                "N_independent": True,
            }
            checks.append(
                comparison(
                    f"dimension{dimension}-mass-shape",
                    [19, 24, 25],
                    full_h,
                    root_mass + math.sqrt(m * n) * shape_h,
                    True,
                    point,
                )
            )
            checks.append(
                comparison(
                    f"dimension{dimension}-Hellinger-entropy",
                    [20, 27],
                    shape_h,
                    -2 * math.expm1(-product_entropy / 2),
                    False,
                    point,
                )
            )
            checks.append(
                comparison(
                    f"dimension{dimension}-affinity-Jensen",
                    [26],
                    -product_entropy / 2,
                    math.log(product_affinity),
                    False,
                    point,
                )
            )
            checks.append(
                comparison(
                    f"dimension{dimension}-mass-lower",
                    [22],
                    root_mass,
                    (m - n) ** 2 / (4 * m),
                    False,
                    point,
                )
            )
            dimension_rows.append(point)
        saved = {
            "source_Rust_dataset": str(root),
            "source_archive": case["archive"],
            "source_compressed_SHA256": sha(source),
            "parameters": parameters,
            "global_continuous_reference_constants": {
                "LSI_C": lsi,
                "Hessian_M": hessian_bound,
                "kinetic_entropy_lambda": kinetic_rate,
                "Gaussian_tail_envelope_factor": envelope,
                "bounded_mixed_tilt_factor": mixed_tilt_factor,
                "position_barrier_factor": position_barrier_factor,
                "Gaussian_position_tail": position_tail,
                "Gaussian_velocity_tail": velocity_tail,
                "tail_bound_derivation": "The bounded mixed tilt costs exp(2T) against the product of the shifted position Gibbs law and the unchanged Gaussian velocity law. Only the position marginal pays exp(2A/theta); union the two tails before applying exp(2T).",
                "Gaussian_tail_outside_probability_upper": tail,
                "scope": "Constants of the unbounded continuous Gibbs reference and the conditional entropy theorem. A finite-grid transport cost does not establish or substitute a discrete LSI.",
            },
            "finite_quadrature_law": {
                "x_nodes": x[:, 0].tolist(),
                "v_nodes": v[:, 0].tolist(),
                "q_x": qx.tolist(),
                "q_v": qv.tolist(),
                "relative_log_density_parameters": parameters["tilt"],
                "linear_position_tilt": shift,
                "log_normalizer": log_normalizer,
                "p_x": px.tolist(),
                "p_v": pv.tolist(),
                "KL": entropy,
                "affinity": affinity,
                "sum_exact_marginal_W2_squared": projected_transport,
                "regeneration": "q(x_i,v_j)=q_x[i]*q_v[j]; p=q*exp(a*sin(x_i)+b*sin(v_j)+z*sin(x_i)*sin(v_j)+linear_tilt*x_i-log_normalizer)",
                "scope": "All tails remain in the unbounded analytical envelope; nodes define a separately declared finite quadrature law. No continuous-law W2 assertion follows from the marginal discrete transport measurement.",
            },
            "dimension_rows": dimension_rows,
            "checks": checks,
        }
        manifest = packed(args.output / f"nonquadratic-{len(rows):03}.json.gz", saved)
        manifests.append(manifest)
        rows.append({
            "id": case["id"],
            "parameters": parameters,
            "C": lsi,
            "M": hessian_bound,
            "lambda": kinetic_rate,
            "KL": entropy,
            "H_squared": 2 * (1 - affinity),
            "finite_marginal_transport_squared": projected_transport,
            "continuous_tail_probability_upper": tail,
            "checks": len(checks),
            "failed": sum(not c["passed"] for c in checks),
            "archive": manifest,
        })
        total += len(checks)
        failures += sum(not c["passed"] for c in checks)
    result = {
        "chapter": 11,
        "nonquadratic_Rust_source_fixtures": len(rows),
        "coordinate_dimensions": [1, 2, 4, 8],
        "comparisons": total,
        "comparisons_failed": failures,
        "source_index_SHA256": sha(root / "archive-index.json"),
        "helper_SHA256": sha(args.output / "source.py"),
        "new_native_updates": 0,
        "rows": rows,
        "archive_manifest": manifests,
    }
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    body = [
        "# Nonquadratic mass–shape and entropy transfer",
        "",
        "Each row reuses full retained Rust quadrature operands. The finite quadrature law is declared explicitly and all Hellinger/KL identities concern that exact law. Global continuous LSI, Hessian and tail constants are recomputed separately; a discrete marginal transport cost is not a certificate for the continuous full-law transport inequality.",
        "",
        "| Landscape fixture | C | M | Conditional entropy rate | Grid KL | Grid H² | Exact grid marginal transport² | Continuous tail bound |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        body.append(
            f"| {row['id']} | {row['C']:.6g} | {row['M']:.6g} | {row['lambda']:.6g} | {row['KL']:.6g} | {row['H_squared']:.6g} | {row['finite_marginal_transport_squared']:.6g} | {row['continuous_tail_probability_upper']:.3g} |"
        )
    (args.output / "results.md").write_text("\n".join(body) + "\n")
    print(
        json.dumps({
            k: result[k]
            for k in ["nonquadratic_Rust_source_fixtures", "comparisons", "comparisons_failed"]
        })
    )
    if failures:
        message = "Nonquadratic transfer comparison failed"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
