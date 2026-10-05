"""Independent numerical references for dimension-aware analytic certificates.

SciPy quadrature error estimates are diagnostic, not interval proofs. The Rust
certificate must enclose these values and supply its own rigorous enclosure.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigvalsh
from scipy.special import gammaln


def references():
    h, friction, diffusion, cap = 0.04, 1.0, 1.0, 2.0
    c = h / 2
    a = math.exp(-friction * h)
    q = diffusion * math.sqrt(-math.expm1(-2 * friction * h) / (2 * friction))
    rows = []
    for curvature in (0.5, 1.0, 2.0):
        k = 1 - curvature * c * c
        sigma = k * q
        for dimension in (1, 2, 4, 8):
            power = 4 if dimension == 1 else 2

            def integrand(radius, dimension=dimension, power=power, sigma=sigma):
                if radius <= 0:
                    return 0.0
                log_density = (
                    (dimension - 1) * math.log(radius)
                    - radius * radius / 2
                    - (dimension / 2 - 1) * math.log(2)
                    - gammaln(dimension / 2)
                )
                return (cap / (cap + sigma * radius)) ** power * math.exp(log_density)

            alpha, error = quad(integrand, 0, math.inf, epsabs=1e-12, epsrel=1e-12)
            eta = 1 - alpha
            trace = 1 - a * a + eta * (a * a + curvature * c * c * (1 - a * a))
            determinant = eta * curvature * c * c * (1 - a * a)
            delta = 2 * determinant / (trace + math.sqrt(trace * trace - 4 * determinant))
            rows.append({
                "d": dimension,
                "curvature": curvature,
                "sigma": sigma,
                "cap_derivative_power": power,
                "numerical_eta": eta,
                "quad_error_estimate": error,
                "numerical_delta_exact_eigenvalue": delta,
                "numerical_squared_error_rate": -math.log1p(-delta) / h,
            })
    sector = []
    beta = 0.04
    metric = np.array([[1.0, beta], [beta, 1.0]])
    for curvature in (0.5, 1.0, 2.0):
        s = curvature * c * c
        k = 1 - s
        physical = np.array([
            [1 - s * (1 + a), c * (1 + a)],
            [-curvature * c * k * (1 + a), a * k - s],
        ])
        scaling = np.diag([math.sqrt(curvature), 1.0])
        scaled = scaling @ physical @ np.linalg.inv(scaling)
        deficits = []
        for endpoint in (0.0, 1.0):
            transformed = np.diag([1.0, endpoint]) @ scaled
            deficits.append(
                float(eigvalsh(metric - transformed.T @ metric @ transformed, metric)[0])
            )
        delta = min(deficits)
        sector.append({
            "curvature": curvature,
            "beta": beta,
            "scaled_native_matrix": scaled.tolist(),
            "numerical_endpoint_generalized_eigenvalues": deficits,
            "numerical_pathwise_delta": delta,
            "numerical_squared_error_rate": -math.log1p(-delta) / h,
            "scope": "Numerical reference only; the Rust interval endpoint certificate proves the guarantee. Isotropic affine force; scaled cross metric differs from the diagonal Q metric.",
        })
    boundary = []
    positional_noise, half_width = 0.02, 2.0
    density_volume = 2 * half_width / (math.sqrt(2 * math.pi) * positional_noise)
    if density_volume <= 1:
        coordinate_bound = 2 * density_volume * (1 - math.log(2))
    else:
        coordinate_bound = (
            math.log(density_volume)
            - math.log(2 - 1 / density_volume)
            + 2
            + 2 * density_volume * math.log1p(-1 / (2 * density_volume))
        )
    for dimension in (1, 2, 4, 8):
        boundary.append({
            "d": dimension,
            "old_bound": density_volume**dimension * 2 * dimension * (1 - math.log(2)),
            "linear_bound": dimension * 2 * density_volume * (1 - math.log(2)),
            "layer_cake_bound": dimension * coordinate_bound,
        })
    return {
        "method": "Independent SciPy integration/eigensolver and closed-form evaluation; numerical diagnostics do not constitute rigorous interval certificates.",
        "config": {"h": h, "friction": friction, "diffusion": diffusion, "cap": cap},
        "new_engine_steps": 0,
        "Gaussian_cap_references": rows,
        "cap_sector_references": sector,
        "barrier_references": boundary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = references()
    report["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["cap_sector_references"], indent=2))


if __name__ == "__main__":
    main()
