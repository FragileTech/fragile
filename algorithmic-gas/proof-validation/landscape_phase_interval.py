"""Directed certificates for the existing root-core rate and its refinements."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from mpmath import iv, mp


def parameter(value):
    decimal = mp.mpf(str(value))
    binary = mp.mpf(float(value))
    return iv.mpf([str(min(decimal, binary)), str(max(decimal, binary))])


def minimum(a, b):
    return iv.mpf([min(a.a, b.a), min(a.b, b.b)])


def maximum(a, b):
    return iv.mpf([max(a.a, b.a), max(a.b, b.b)])


def validate_parameters(p):
    required = {
        "dimension",
        "largest_well",
        "normalization",
        "timestep",
        "friction",
        "clone_jitter",
        "ou_amplitude",
        "position_amplitude",
        "velocity_cap",
        "restitution",
        "viscosity",
        "bandwidth",
        "core_radius",
    }
    if not isinstance(p, dict) or set(p) != required:
        message = "A complete declared native regional parameter record is required"
        raise ValueError(message)
    valid_dimension = (
        not isinstance(p["dimension"], bool)
        and isinstance(p["dimension"], int)
        and p["dimension"] > 0
    )
    valid_roots = (
        not isinstance(p["largest_well"], bool)
        and isinstance(p["largest_well"], int)
        and 0 <= p["largest_well"] <= 21
    )
    if not valid_dimension or not valid_roots or p["normalization"] not in {"count", "row"}:
        message = "Invalid dimension, root range or native viscous normalization"
        raise ValueError(message)
    for key in required - {"dimension", "largest_well", "normalization"}:
        value = p[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            message = "Finite numeric physical parameters are required"
            raise ValueError(message)
        if key in {"timestep", "velocity_cap", "bandwidth", "core_radius"} and value <= 0:
            message = "Positive timestep, cap, bandwidth and core radius are required"
            raise ValueError(message)
        if (
            key != "restitution"
            and key not in {"timestep", "velocity_cap", "bandwidth", "core_radius"}
            and value < 0
        ):
            message = "Friction, all noise amplitudes and viscosity must be nonnegative"
            raise ValueError(message)
    if not 0 <= p["restitution"] <= 1:
        message = "Actual native restitution must belong to [0,1]"
        raise ValueError(message)


def constants(p, radius):
    h, gamma, sigma = [parameter(p[k]) for k in ("timestep", "friction", "clone_jitter")]
    omega = 2 * iv.pi
    drift = h * (1 + iv.exp(-gamma * h)) / 2
    eta = h * drift / 2
    ell = 1 - 2 * eta
    amplitude = 20 * iv.pi * eta
    a = iv.exp(-(omega**2) * sigma**2 / 2)
    b = a**4
    cos_min = iv.cos(omega * radius)
    rho = maximum(abs(ell - amplitude * omega * a), abs(ell - amplitude * omega * a * cos_min))
    coefficient = (1 + rho**2) / 2
    if (
        not bool(radius < iv.mpf("0.125"))
        or not bool(rho > 0)
        or not bool(rho < 1)
        or not bool(ell >= 0)
    ):
        message = "Independent root-core premises fail"
        raise ValueError(message)
    old_kj = (
        ell**2 * sigma**2
        - 2 * ell * amplitude * omega * sigma**2 * a * cos_min
        + amplitude**2 / 2 * (1 - b * iv.cos(2 * omega * radius))
    )

    def variance(z):
        return (
            ell**2 * sigma**2
            - 2 * ell * amplitude * omega * sigma**2 * a * z
            + amplitude**2 * ((1 + b) / 2 - a**2 + (a**2 - b) * z**2)
        )

    sharp_kj = maximum(iv.mpf(0), maximum(variance(cos_min), variance(iv.mpf(1))))
    refresh = amplitude * (1 - a) * iv.sin(omega * radius)
    epsilon = iv.sqrt(coefficient / rho**2) - 1
    cap = (1 + 2 * abs(parameter(p["restitution"]))) * parameter(p["velocity_cap"])
    noise = (h / 2) ** 2 * parameter(p["ou_amplitude"]) ** 2 + parameter(
        p["position_amplitude"]
    ) ** 2
    d = p["dimension"]

    def floor(kj):
        return (
            (1 + epsilon) * ((1 + 1 / epsilon) * d * refresh**2 + d * kj)
            + (1 + 1 / epsilon) * drift**2 * cap**2
            + d * noise
        )

    return {
        "positional_coefficient": coefficient,
        "physical_multiplier_rate": -iv.ln(coefficient) / h,
        "old_jitter_variance": old_kj,
        "exact_jitter_variance_upper": sharp_kj,
        "old_floor": floor(old_kj),
        "exact_variance_floor": floor(sharp_kj),
    }


def certificate(fixtures):
    iv.dps = 60
    mp.dps = 90
    rows = []
    for case in fixtures["cases"]:
        p = case["parameters"]
        validate_parameters(p)
        if not bool(parameter(p["timestep"]) * parameter(p["viscosity"]) / 2 <= 1):
            message = "Independent convex first-kick premise fails"
            raise ValueError(message)
        displacement = 2 * p["largest_well"] / (2 + 20 * iv.sqrt(2) * iv.pi**2)
        original = constants(p, parameter(p["core_radius"]) + displacement)
        for _ in range(32):
            displacement = minimum(
                displacement,
                2 * p["largest_well"] / (2 + 40 * iv.pi**2 * iv.cos(2 * iv.pi * displacement)),
            )
        refined = constants(p, parameter(p["core_radius"]) + displacement)
        quadratures = case.get("full_gaussian_quadrature", [])
        quadrature_pass = all(
            q["variance"]
            <= float(refined["exact_jitter_variance_upper"].b) + 10 * q["error"] + 1e-12
            for q in quadratures
        )
        rows.append({
            "fixture": case["id"],
            "parameters": p,
            "root_displacement_interval": str(displacement),
            "geometry_iterations": 32,
            "original": {key: str(value) for key, value in original.items()},
            "refined": {key: str(value) for key, value in refined.items()},
            "rate_strictly_improves": bool(
                refined["physical_multiplier_rate"] > original["physical_multiplier_rate"]
            ),
            "floor_strictly_improves": bool(
                refined["exact_variance_floor"] < original["old_floor"]
            ),
            "independent_full_gaussian_quadrature_pass": quadrature_pass,
        })
    return {
        "rows": rows,
        "N_independent": True,
        "new_engine_steps": 0,
        "method": "60-decimal directed interval arithmetic; parameter intervals contain decimal and binary64 values. Exact Gaussian characteristic-function variance and finite analytic root enclosures.",
        "scope": "KUR one-step source-variance multiplier and full noise floor under its complete root-core premises. Source pressure, slow-zone passage and exterior flux remain required for iteration or global mixing.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = certificate(json.loads(args.fixtures.read_text()))
    source = (
        Path(__file__).resolve().parents[2]
        / "docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md"
    )
    report["source_label"] = "cor-slc-exact-jitter-regional-refinement"
    report["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["fixtures_sha256"] = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    failures = sum(not row["independent_full_gaussian_quadrature_pass"] for row in report["rows"])
    print(json.dumps({"cases": len(report["rows"]), "quadrature_failures": failures}))
    if failures:
        message = "Full Gaussian quadrature disagrees with the variance envelope"
        raise SystemExit(message)
