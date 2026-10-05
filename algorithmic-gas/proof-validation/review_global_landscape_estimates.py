"""Independently check saved native gradients and selected-moment constants.

The exact integral uses endpoint antiderivatives at 70 decimal digits, rather
than the Rust midpoint/sinc expression. Saved gradient rows are deterministic
quadrature nodes, not independent Monte Carlo samples. No new trajectories.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import mpmath as mp


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_integral(x, y):
    """Integrate squared physical Rastrigin gradient by a separate primitive."""
    amplitude, frequency = 20 * mp.pi, 2 * mp.pi

    def primitive(r):
        return (
            4 * r**3 / 3
            + 4
            * amplitude
            * (-r * mp.cos(frequency * r) / frequency + mp.sin(frequency * r) / frequency**2)
            + amplitude**2 * (r / 2 - mp.sin(2 * frequency * r) / (4 * frequency))
        )

    result = mp.mpf(0)
    for left, right in zip(x, y, strict=True):
        a, b = mp.mpf(left), mp.mpf(right)
        if a == b:
            result += (2 * a + amplitude * mp.sin(frequency * a)) ** 2
        else:
            result += (primitive(b) - primitive(a)) / (b - a)
    return result


def fourth_derivative_upper(x, y):
    # f(r)=4r²+4A r sin(kr)+A²(1-cos(2kr))/2.
    # D_t^4(r sin(kr))=v^4(k^4 r sin(kr)-4k^3 cos(kr)).
    amplitude, frequency = 20 * mp.pi, 2 * mp.pi
    return mp.fsum(
        (mp.mpf(b) - mp.mpf(a)) ** 4
        * (
            4 * amplitude * (frequency**4 * max(abs(a), abs(b)) + 4 * frequency**3)
            + 8 * amplitude**2 * frequency**4
        )
        for a, b in zip(x, y, strict=True)
    )


def selected_constants(d, p, exponent, history):
    """Independent scalar specialization of the stated analytic formulas."""
    h = mp.mpf("0.04")
    half = h / 2
    damping = mp.exp(-h)
    drift = half * (1 + damping)
    force_factor = half * drift
    a = 1 - 2 * force_factor
    ou_sigma = mp.sqrt(1 - mp.exp(-2 * h)) / mp.sqrt(2)
    tau = mp.sqrt(half**2 * ou_sigma**2 + mp.mpf("0.02") ** 2)
    order = 2 * math.ceil(p / 2)
    gaussian_lp = mp.fprod(d + 2 * j for j in range(order // 2)) ** (mp.mpf(1) / order)
    additive_lp = (
        a * mp.mpf("0.1") * gaussian_lp
        + drift * (2 if history else 4)
        + force_factor * 20 * mp.pi * mp.sqrt(d)
        + tau * gaussian_lp
    )
    lam = (1 + a**p) / 2
    epsilon = (lam / a**p) ** (mp.mpf(1) / (p - 1)) - 1
    additive = (1 + 1 / epsilon) ** (p - 1) * additive_lp**p
    f_low = mp.mpf("0.1") ** (2 * mp.mpf(exponent))
    f_high = mp.mpf("2.1") ** (2 * mp.mpf(exponent))
    accept = min(mp.mpf(1), (f_high - f_low) / (f_low + mp.mpf("0.000001")))
    kernel = mp.exp(-mp.mpf(32) / 18)
    gain = accept / kernel
    multiplier = lam * (1 + gain * (mp.mpf(4) / 3 if history else 1))
    return {
        "dimension": d,
        "moment_order": p,
        "exponent": float(exponent),
        "history_window": history,
        "component_collision_enabled": not bool(history),
        "restitution": 0 if history else 0.5,
        "a": float(a),
        "gaussian_lp": float(gaussian_lp),
        "kernel_lower": float(kernel),
        "acceptance_upper": float(accept),
        "lambda_p": float(lam),
        "young_epsilon": float(epsilon),
        "kinetic_additive": float(additive),
        "moment_multiplier": float(multiplier),
        "closes": bool(multiplier < 1),
        "floor": float(additive / (1 - multiplier)) if multiplier < 1 else None,
        "physical_block_rate": float(-mp.log(multiplier) / (h * (history + 1)))
        if 0 < multiplier < 1
        else None,
        "hypotheses": [
            "all frames alive; bounded entering velocities",
            "historical accepted sources require component restitution disabled",
            "Gaussian noises full support and first viscous matrix stochastic",
        ],
    }


def review(dataset, output):
    mp.mp.dps = 70
    index_path = dataset / "archive-index.json"
    index = json.loads(index_path.read_text())
    rows, verified, provenance = [], [], None
    for entry in index["entries"]:
        path = dataset / entry["path"]
        actual_sha = digest(path)
        if actual_sha != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        verified.append({"path": entry["path"], "sha256": actual_sha})
        value = json.loads(gzip.decompress(path.read_bytes()))
        if entry["tag"] == "axioms/provenance":
            provenance = value
        if "native_gradients" not in value:
            continue
        x, y = value["x"], value["y"]
        n, d = value["quadrature_panels"], value["dimension"]
        gradients = value["native_gradients"]
        if len(gradients) != n + 1 or any(len(row) != d for row in gradients):
            raise ValueError(f"Gradient shape mismatch: {path}")
        max_gradient_error = 0.0
        for node, stored in enumerate(gradients):
            for axis, observed in enumerate(stored):
                position = x[axis] + (y[axis] - x[axis]) * node / n
                expected = 2 * position + 20 * math.pi * math.sin(2 * math.pi * position)
                max_gradient_error = max(max_gradient_error, abs(observed - expected))
        simpson = math.fsum(
            (1 if node in {0, n} else 4 if node % 2 else 2)
            * math.fsum(component * component for component in row)
            for node, row in enumerate(gradients)
        ) / (3 * n)
        exact = endpoint_integral(x, y)
        error = fourth_derivative_upper(x, y) / (180 * n**4)
        delta = [mp.mpf(b) - mp.mpf(a) for a, b in zip(x, y, strict=True)]
        length = mp.sqrt(mp.fsum(component**2 for component in delta))
        md = 20 * mp.pi * mp.sqrt(d)
        baseline = max(mp.mpf(0), 2 * length / mp.sqrt(12) - md) ** 2
        tolerance = mp.mpf("2e-10") * max(1, abs(exact))
        checks = {
            "native_gradient_formula": max_gradient_error <= 2e-9,
            "saved_simpson_reconstructed": abs(simpson - value["measured_segment_integral"])
            <= float(tolerance),
            "endpoint_vs_saved_midpoint_integral": abs(
                exact - value["independent_exact_segment_integral"]
            )
            <= tolerance,
            "simpson_error": abs(mp.mpf(simpson) - exact) <= error + tolerance,
            "analytic_segment_lower": exact + tolerance >= baseline,
            "saved_fourth_derivative_bound": abs(error - value["simpson_error_upper"])
            <= mp.mpf("2e-13") * max(1, error),
            "positive_length_certificate": length + mp.mpf("1e-11") >= mp.sqrt(12) * md,
        }
        rows.append({
            "tag": entry["tag"],
            "path": entry["path"],
            "sha256": actual_sha,
            "dimension": d,
            "nodes": n + 1,
            "max_gradient_formula_error": max_gradient_error,
            "simpson_reconstructed": simpson,
            "endpoint_integral": float(exact),
            "simpson_defect": float(abs(mp.mpf(simpson) - exact)),
            "simpson_error_upper": float(error),
            "analytic_segment_lower": float(baseline),
            "checks": checks,
            "passed": all(checks.values()),
        })
    source_paths = [
        Path("docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"),
        Path(
            "docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md"
        ),
        Path(__file__),
    ]
    constants = [
        selected_constants(d, p, exponent, history)
        for d in (1, 2, 4, 8)
        for p in (4, 8)
        for exponent in ("0.00001", "1")
        for history in (0, 1, 4)
    ]
    result = {
        "scope": "Independent retained-data review; no native trajectories generated.",
        "arithmetic_scope": "70-digit independent endpoint antiderivative; binary64 native-gradient and Simpson checks. This numerical audit is not a directed interval certificate.",
        "dataset": str(dataset),
        "index_sha256": digest(index_path),
        "provenance": provenance,
        "verified_entries": verified,
        "current_source_sha256": {str(path): digest(path) for path in source_paths},
        "cases": rows,
        "analytic_selected_moment_constants": constants,
        "native_compatibility": {
            "current_component_collision": "valid",
            "accepted_historical_component_collision": "unsupported by actual native kernel; restitution must be None",
            "historical_source_moment_algebra": "valid all-alive normalized pool",
        },
        "summary": {
            "verified_files": len(verified),
            "gradient_cases": len(rows),
            "native_gradient_nodes": sum(row["nodes"] for row in rows),
            "gradient_scalar_values": sum(row["nodes"] * row["dimension"] for row in rows),
            "checks": sum(len(row["checks"]) for row in rows),
            "failed_checks": sum(not passed for row in rows for passed in row["checks"].values()),
            "new_trajectories": 0,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = review(args.dataset, args.output)
    raise SystemExit(bool(report["summary"]["failed_checks"]))
