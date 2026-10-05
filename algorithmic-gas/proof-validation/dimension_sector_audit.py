"""Independently certify curvature-aware whole-update cap endpoint matrices.

The interval covers decimal .04 and its native binary64 representation. The
certificate is a uniform matrix inequality, not a simulation or fitted rate.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from mpmath import iv


def describe(value):
    return {"interval": str(value), "strictly_positive": bool(value > 0)}


def certificate():
    iv.dps = 60
    h = iv.mpf(["0.03999999999999999", "0.04000000000000001"])
    beta = iv.mpf("0.04")
    a = iv.exp(-h)
    c = h / 2
    metric = iv.matrix([[1, beta], [beta, 1]])
    rows = []
    for curvature, numerator in (("0.5", 63), ("1", 156), ("2", 289)):
        omega = iv.mpf(curvature)
        delta = iv.mpf(numerator) / 100000
        root = iv.sqrt(omega)
        s = omega * c * c
        k = 1 - s
        native = iv.matrix([
            [1 - s * (1 + a), root * c * (1 + a)],
            [-root * c * k * (1 + a), a * k - s],
        ])
        endpoints = []
        for cap_secant in (0, 1):
            transformed = iv.matrix([[1, 0], [0, cap_secant]]) * native
            deficit = metric - transformed.T * metric * transformed - delta * metric
            determinant = deficit[0, 0] * deficit[1, 1] - deficit[0, 1] ** 2
            checks = {
                "first_diagonal": describe(deficit[0, 0]),
                "second_diagonal": describe(deficit[1, 1]),
                "determinant": describe(determinant),
            }
            endpoints.append({
                "cap_secant": cap_secant,
                "checks": checks,
                "passed": all(v["strictly_positive"] for v in checks.values()),
            })
        rows.append({
            "curvature": float(curvature),
            "beta": 0.04,
            "delta_rational": f"{numerator}/100000",
            "delta": numerator / 100000,
            "squared_error_rate_at_h_0_04": -math.log1p(-numerator / 100000) / 0.04,
            "endpoints": endpoints,
            "passed": all(e["passed"] for e in endpoints),
        })
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/source/2_fractal_gas/convergence_program/18a_keystone_uniform_coupled.md"
    )
    return {
        "method": "60-decimal mpmath directed interval arithmetic; positive diagonal/determinant certificate at both cap-sector endpoints",
        "h_interval": str(h),
        "new_engine_steps": 0,
        "source_label": "thm-rcap-harmonic-whole-update",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "scope": "Isotropic quadratic force -omega*x, gamma=1, shared additive noise, no cloning/viscosity/killing. Native radial cap has symmetric secant 0<=D<=I. Convexity in each secant eigenvalue and simultaneous scalar-block diagonalization give the matrix inequality in every d and N. Metric is population-normalized omega*|dx|^2+2*beta*sqrt(omega)<dx,dv>+|dv|^2.",
        "rows": rows,
        "all_passed": all(r["passed"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = certificate()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"all_passed": report["all_passed"], "rows": len(report["rows"])}))
    if not report["all_passed"]:
        msg = "Endpoint matrix certificate failed; evidence retained"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
