"""Extract the existing Python landscape functions for independent Rust fixtures.

Uses their AST verbatim; importing the full package is unnecessary. The original
function source hash, SciPy roots and complete primitive inputs are retained.
"""

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.optimize import brentq


def main():
    workspace = Path(__file__).resolve().parents[1]
    source = workspace / "proof-validation/reference/landscape_phase.py"
    text = source.read_text()
    names = {
        "RastriginRegionalProfile",
        "RastriginRegionalBound",
        "rastrigin_regional_profile",
        "rastrigin_regional_bound",
    }
    tree = ast.parse(text)
    selected = ast.Module(
        body=[node for node in tree.body if getattr(node, "name", None) in names],
        type_ignores=[],
    )
    namespace = {"dataclass": dataclass, "math": math, "brentq": brentq}
    # Execute only named, trusted repository formulas to keep the reference
    # implementation independent of the Rust port and avoid package imports.
    exec(compile(selected, str(source), "exec"), namespace)  # ruff: ignore[exec-builtin]
    cases = []
    for d in [1, 2, 3, 4, 8]:
        for jitter in [0.0, 0.1]:
            for mode in ["count", "row"]:
                args = {
                    "dimension": d,
                    "timestep": 0.04,
                    "friction": 1.0,
                    "clone_jitter": jitter,
                    "ou_amplitude": math.sqrt(-math.expm1(-0.08) / 2),
                    "position_amplitude": 0.02,
                    "velocity_cap": 2.0,
                    "restitution": 0.5,
                    "viscosity": 0.3,
                    "core_radius": 1 / 16,
                    "largest_well": 2,
                }
                result = namespace["rastrigin_regional_bound"](**args)
                displacement = 4 / (2 + 20 * math.sqrt(2) * math.pi**2)
                for _ in range(32):
                    displacement = min(
                        displacement,
                        4 / (2 + 40 * math.pi**2 * math.cos(2 * math.pi * displacement)),
                    )
                radius = args["core_radius"] + displacement
                half = args["timestep"] / 2
                eta = half**2 * (1 + math.exp(-args["friction"] * args["timestep"]))
                ell, amplitude, omega = 1 - 2 * eta, 20 * math.pi * eta, 2 * math.pi
                attenuation = math.exp(-(omega**2) * jitter**2 / 2)
                quadrature = []
                for mu in [0.0, radius]:
                    mean = ell * mu - amplitude * attenuation * math.sin(omega * mu)

                    def integrand(z):
                        x = mu + jitter * z
                        transformed = ell * x - amplitude * math.sin(omega * x)
                        return (
                            (transformed - mean) ** 2
                            * math.exp(-(z**2) / 2)
                            / math.sqrt(2 * math.pi)
                        )

                    measured, error = quad(integrand, -math.inf, math.inf, epsabs=1e-12)
                    quadrature.append({"mu": mu, "variance": measured, "error": error})
                coarse_displacement = 4 / (2 + 20 * math.sqrt(2) * math.pi**2)
                adjusted = {**args, "core_radius": radius - coarse_displacement}
                geometry_result = namespace["rastrigin_regional_bound"](**adjusted)
                epsilon = (
                    math.sqrt(
                        geometry_result.positional_coefficient / geometry_result.mean_map_squared
                    )
                    - 1
                )
                exact_kj = max(q["variance"] for q in quadrature)
                tightened_floor = geometry_result.conservative_floor + (1 + epsilon) * d * (
                    exact_kj - geometry_result.accepted_coordinate_variance
                )
                cases.append({
                    "id": f"d{d}_jitter{jitter}_{mode}",
                    "parameters": {**args, "bandwidth": 1.0, "normalization": mode},
                    "expected": asdict(result),
                    "tightened_root_displacement": displacement,
                    "full_gaussian_quadrature": quadrature,
                    "expected_tightened": {
                        "geometry_iterations": 32,
                        "enlarged_core_radius": radius,
                        "accepted_coordinate_variance": exact_kj,
                        "positional_coefficient": geometry_result.positional_coefficient,
                        "epsilon": epsilon,
                        "conservative_floor": tightened_floor,
                        "physical_multiplier_rate": -math.log(
                            geometry_result.positional_coefficient
                        )
                        / args["timestep"],
                    },
                })
    result = {
        "python_source_path": str(source),
        "python_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "extraction": "Verbatim AST of existing functions/classes, using existing SciPy brentq",
        "extracted_source": "\n\n".join(
            ast.get_source_segment(text, node)
            for node in tree.body
            if getattr(node, "name", None) in names
        ),
        "cases": cases,
        "profile": asdict(namespace["rastrigin_regional_profile"](-2, 2)),
    }
    target = workspace / "crates/benchmarks/fixtures/landscape-phase-python.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(target)


if __name__ == "__main__":
    main()
