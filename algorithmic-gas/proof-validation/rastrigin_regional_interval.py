"""Independent directed certificate of a nonquadratic within-well rate."""

import argparse
import hashlib
import json
from pathlib import Path

from mpmath import iv


def certificate():
    iv.dps = 60
    h = iv.mpf(["0.03999999999999999", "0.04000000000000001"])
    c = h / 2
    a = iv.exp(-h)
    omega = 2 + 40 * iv.pi**2
    alpha = iv.mpf("0.8412863295825702")
    beta = iv.mpf("0.025")
    delta = iv.mpf("0.038")
    metric = iv.matrix([[alpha, beta], [beta, 1]])
    k = 1 - omega * c**2
    native = iv.matrix([
        [1 - omega * c**2 * (1 + a), iv.sqrt(omega) * c * (1 + a)],
        [-iv.sqrt(omega) * c * k * (1 + a), a * k - omega * c**2],
    ])
    endpoints = []
    for secant in [0, 1]:
        transformed = iv.matrix([[1, 0], [0, secant]]) * native
        difference = metric - transformed.T * metric * transformed - delta * metric
        determinant = difference[0, 0] * difference[1, 1] - difference[0, 1] ** 2
        passed = (
            bool(difference[0, 0] > 0) and bool(difference[1, 1] > 0) and bool(determinant > 0)
        )
        endpoints.append({
            "secant": secant,
            "diagonal_intervals": [str(difference[0, 0]), str(difference[1, 1])],
            "determinant_interval": str(determinant),
            "passed": passed,
        })
    tx = 1 / iv.sqrt(omega * (alpha - beta**2))
    tv = 1 / iv.sqrt(1 - beta**2 / alpha)
    largest = (alpha + 1 + iv.sqrt((alpha - 1) ** 2 + 4 * beta**2)) / 2
    rows = []
    for radius in ["0.001", "0.005", "0.01", "0.02", "0.03", "0.05", "0.15"]:
        r = iv.mpf(radius)
        modulus = 20 * iv.pi**2 * (1 - iv.cos(2 * iv.pi * r))
        omega = 2 + 40 * iv.pi**2 - modulus
        alpha = 1 - omega * iv.mpf("0.02") ** 2
        metric = iv.matrix([[alpha, beta], [beta, 1]])
        k = 1 - omega * c**2
        native = iv.matrix([
            [1 - omega * c**2 * (1 + a), iv.sqrt(omega) * c * (1 + a)],
            [-iv.sqrt(omega) * c * k * (1 + a), a * k - omega * c**2],
        ])
        regional_endpoints = []
        for secant in [0, 1]:
            transformed = iv.matrix([[1, 0], [0, secant]]) * native
            difference = metric - transformed.T * metric * transformed - delta * metric
            determinant = difference[0, 0] * difference[1, 1] - difference[0, 1] ** 2
            regional_endpoints.append({
                "secant": secant,
                "diagonal_intervals": [str(difference[0, 0]), str(difference[1, 1])],
                "determinant_interval": str(determinant),
                "passed": bool(difference[0, 0] > 0)
                and bool(difference[1, 1] > 0)
                and bool(determinant > 0),
            })
        harmonic_pass = all(e["passed"] for e in regional_endpoints)
        tx = 1 / iv.sqrt(omega * (alpha - beta**2))
        tv = 1 / iv.sqrt(1 - beta**2 / alpha)
        largest = (alpha + 1 + iv.sqrt((alpha - 1) ** 2 + 4 * beta**2)) / 2
        first = modulus * tx
        query2 = abs(1 - omega * c**2 * (1 + a)) * tx + c * (1 + a) * tv + c**2 * (1 + a) * first
        second = modulus * query2
        dx = c**2 * (1 + a) * first
        dv = c * abs(a - omega * c**2 * (1 + a)) * first + c * second
        rho = (iv.sqrt(1 - delta) + iv.sqrt(largest * (omega * dx**2 + dv**2))) ** 2
        closes = harmonic_pass and bool(rho < 1)
        rows.append({
            "well_half_width": float(radius),
            "harmonic_curvature_interval": str(omega),
            "harmonic_metric_alpha_interval": str(alpha),
            "harmonic_endpoints": regional_endpoints,
            "harmonic_certificate_pass": harmonic_pass,
            "rho_interval": str(rho),
            "conditional_rate_interval": str(-iv.ln(rho) / h) if closes else None,
            "contracts": closes,
        })
    source = (
        Path(__file__).resolve().parents[1]
        / "docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md"
    )
    return {
        "endpoints": endpoints,
        "rows": rows,
        "reference_endpoint_certificates_pass": all(e["passed"] for e in endpoints),
        "alpha": "1-omega_midpoint*(0.02)^2 for each declared region",
        "beta": str(beta),
        "delta": str(delta),
        "curvature_choice": "Midpoint of analytic regional Hessian interval; minimizes residual Lipschitz modulus",
        "h_interval": str(h),
        "method": "60-decimal directed interval arithmetic, analytic Rastrigin regional modulus and complete native cap sector endpoints",
        "source_label": "cor-slc-rastrigin-regional-sector",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "N_independent": True,
        "dimension_independent_local_rate": True,
        "scope": "Both actual kick-query pairs must stay in the same declared integer-well box. Gaussian escape and other regions remain separate. No global nonlinear mixing assertion.",
        "new_engine_steps": 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certificate()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({
            "endpoint_certificates_pass": report["reference_endpoint_certificates_pass"],
            "conditional_rates": [
                {
                    "well_half_width": row["well_half_width"],
                    "contracts": row["contracts"],
                    "rate_interval": row["conditional_rate_interval"],
                }
                for row in report["rows"]
            ],
        })
    )
    if not report["reference_endpoint_certificates_pass"]:
        message = "Harmonic endpoint certificate rejected; no local rates certified"
        raise SystemExit(message)
