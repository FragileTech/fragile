"""Independent analytic/numeric review of the one-period Rastrigin certificate."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import unittest

import mpmath as mp


def lower(length):
    ell = mp.mpf(length)
    amplitude = 20 * mp.pi
    return (
        ell**2 / 3
        + amplitude**2 * (mp.mpf("0.5") - 1 / (4 * mp.pi * ell) - 1 / (mp.pi**2 * ell**2))
        - 4 * amplitude * (1 / (2 * mp.pi) + 1 / (2 * mp.pi**2 * ell))
    )


def lower_derivative(length):
    ell = mp.mpf(length)
    amplitude = 20 * mp.pi
    return (
        2 * ell / 3
        + amplitude**2 * (1 / (4 * mp.pi * ell**2) + 2 / (mp.pi**2 * ell**3))
        + 2 * amplitude / (mp.pi**2 * ell**2)
    )


def exact_endpoint_variance(left, right):
    """Separate antiderivatives, including subtraction at high precision."""
    a, b = mp.mpf(left), mp.mpf(right)
    length = b - a
    amplitude, frequency = 20 * mp.pi, 2 * mp.pi

    def primitive_mean(x):
        return x**2 - amplitude * mp.cos(frequency * x) / frequency

    def primitive_second(x):
        return (
            4 * x**3 / 3
            + 4
            * amplitude
            * (-x * mp.cos(frequency * x) / frequency + mp.sin(frequency * x) / frequency**2)
            + amplitude**2 * (x / 2 - mp.sin(2 * frequency * x) / (4 * frequency))
        )

    mean = (primitive_mean(b) - primitive_mean(a)) / length
    second = (primitive_second(b) - primitive_second(a)) / length
    return mean, second, second - mean**2


def interval_constant():
    mp.iv.dps = 70
    pi = mp.iv.pi
    amplitude = 20 * pi
    kappa = (
        mp.iv.mpf(1) / 3
        + amplitude**2 * (mp.iv.mpf("0.5") - 1 / (4 * pi) - 1 / pi**2)
        - 4 * amplitude * (1 / (2 * pi) + 1 / (2 * pi**2))
    )
    # Fixed rational lower endpoint, not a fitted empirical floor.
    rational = mp.iv.mpf(1207)
    return {
        "kappa_interval": str(kappa),
        "certified_rational_lower": 1207,
        "rational_margin_interval": str(kappa - rational),
        "positive_rational_margin": bool(kappa - rational > 0),
    }


def review(dataset, output):
    mp.mp.dps = 90
    rng = random.Random(713821)
    fixtures = [
        (ell, mid)
        for ell in (1.0, 1.000000001, 1.1, 1.5, 2.0, 10.0, 1000.0, 1e5)
        for mid in (0.0, 0.25, 0.5, -2.0, 10.0, 1e9)
    ]
    fixtures.extend(
        (math.exp(rng.uniform(0, math.log(1e5))), rng.uniform(-1e4, 1e4)) for _ in range(512)
    )
    cases = []
    for ell, mid in fixtures:
        # Exact mathematical inputs are the retained binary64 endpoints.
        a, b = float(mid - ell / 2), float(mid + ell / 2)
        ell = mp.mpf(b) - mp.mpf(a)
        mean, second, variance = exact_endpoint_variance(a, b)
        bound = lower(ell)
        cases.append({
            "left": a,
            "right": b,
            "length": float(ell),
            "gradient_mean": float(mean),
            "gradient_second": float(second),
            "gradient_variance": float(variance),
            "variance_lower": float(bound),
            "margin": float(variance - bound),
            "checks": {
                "length_at_least_one": bool(ell >= 1),
                "variance_lower": bool(variance >= bound),
                "monotone_lower": bool(lower_derivative(ell) > 0),
                "dimension_free_constant": bool(bound >= lower(1)),
            },
        })
    index_path = dataset / "archive-index.json"
    index = json.loads(index_path.read_text())
    if index["status"] != "complete":
        message = "Completed immutable dataset required"
        raise ValueError(message)
    verified = []
    provenance = None
    retained = []
    for entry in index["entries"]:
        path = dataset / entry["path"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        data = json.loads(gzip.decompress(path.read_bytes()))
        verified.append({"path": entry["path"], "sha256": sha})
        if entry["tag"] == "axioms/provenance":
            provenance = data
        if not entry["tag"].startswith("axioms/d"):
            continue
        x, y = data["x"], data["y"]
        axis = max(range(len(x)), key=lambda j: abs(y[j] - x[j]))
        left, right = sorted((x[axis], y[axis]))
        ell = mp.mpf(right) - mp.mpf(left)
        mean, second, variance = exact_endpoint_variance(left, right)
        n = data["quadrature_panels"]
        nodes = data["native_gradients"]
        weights = [1 if node in {0, n} else 4 if node % 2 else 2 for node in range(n + 1)]
        measured_mean = math.fsum(
            weight * row[axis] for weight, row in zip(weights, nodes, strict=True)
        ) / (3 * n)
        measured_second = math.fsum(
            weight * row[axis] ** 2 for weight, row in zip(weights, nodes, strict=True)
        ) / (3 * n)
        measured_variance = measured_second - measured_mean**2
        amplitude, frequency = 20 * mp.pi, 2 * mp.pi
        first_error = amplitude * frequency**4 * ell**4 / (180 * n**4)
        second_error = (
            ell**4
            * (
                4 * amplitude * (frequency**4 * max(abs(left), abs(right)) + 4 * frequency**3)
                + 8 * amplitude**2 * frequency**4
            )
            / (180 * n**4)
        )
        error = second_error + 2 * abs(mean) * first_error + first_error**2
        rounding = mp.mpf("2e-10") * max(1, abs(second), mean**2)
        full_second = mp.fsum(
            exact_endpoint_variance(min(a, b), max(a, b))[1]
            if a != b
            else (2 * mp.mpf(a) + amplitude * mp.sin(frequency * a)) ** 2
            for a, b in zip(x, y, strict=True)
        )
        full_simpson = math.fsum(
            weight * math.fsum(component**2 for component in row)
            for weight, row in zip(weights, nodes, strict=True)
        ) / (3 * n)
        gradient_error = 0.0
        for node, row in enumerate(nodes):
            for coordinate, observed in enumerate(row):
                point = x[coordinate] + (y[coordinate] - x[coordinate]) * node / n
                expected = 2 * point + 20 * math.pi * math.sin(2 * math.pi * point)
                gradient_error = max(gradient_error, abs(observed - expected))
        exact_tolerance = mp.mpf("2e-10") * max(1, full_second)
        retained.append({
            "tag": entry["tag"],
            "sha256": sha,
            "axis": axis,
            "coordinate_span": float(ell),
            "coordinate_variance": float(variance),
            "measured_coordinate_variance": measured_variance,
            "variance_lower": float(lower(ell)),
            "quadrature_variance_error_upper": float(error),
            "full_gradient_energy": float(full_second),
            "full_gradient_energy_Simpson": full_simpson,
            "max_native_gradient_formula_error": gradient_error,
            "retained_gradient_scalar_values": len(nodes) * len(x),
            "retained_budget": data["parameters"],
            "checks": {
                "one_period_span": bool(ell >= 1),
                "independent_coordinate_variance_lower": bool(variance >= lower(ell)),
                "retained_native_variance_quadrature": bool(
                    abs(mp.mpf(measured_variance) - variance) <= error + rounding
                ),
                "dimension_free_kappa": bool(variance >= lower(1)),
                "full_gradient_energy_rational_bound": bool(full_second >= 1207),
                "full_gradient_energy_floating_Phi1_bound": bool(full_second >= lower(1)),
                "independent_full_energy_vs_native_exact": bool(
                    abs(full_second - data["independent_exact_segment_integral"])
                    <= exact_tolerance
                ),
                "independent_native_Simpson_reconstruction": bool(
                    abs(full_simpson - data["measured_segment_integral"]) <= exact_tolerance
                ),
                "every_native_gradient_formula": gradient_error <= 2e-9,
            },
        })
    certificate = interval_constant()
    result = {
        "scope": "Independent analytic fixtures and the explicitly supplied native gradient dataset. No trajectories or gradient arrays are generated by this review.",
        "dataset": str(dataset),
        "native_run_provenance": provenance,
        "verified_entries": verified,
        "source_binding": {
            "label": "cor-eg-rastrigin-short-gradient",
            "source_path": "docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md",
            "current_source_sha256": hashlib.sha256(
                Path(
                    "docs/source/2_fractal_gas/convergence_program/02_euclidean_gas.md"
                ).read_bytes()
            ).hexdigest(),
            "bound_scope": "Complete one-period variance inequality; rational bound1207 is directed-certified separately from floating Phi1.",
        },
        "arithmetic": "90-digit endpoint antiderivatives; directed70-digit interval proof only for the fixed rational kappa>=1207.",
        "kappa": float(lower(1)),
        "length_coefficient": 1,
        "dimension_scope": "every positive integer d; L_grad=sqrt(d); independent of N",
        "interval_certificate": certificate,
        "analytic_cases": cases,
        "retained_cases": retained,
        "summary": {
            "analytic_cases": len(cases),
            "retained_native_cases": len(retained),
            "verified_files": len(verified),
            "retained_native_gradient_scalar_values": sum(
                row["retained_gradient_scalar_values"] for row in retained
            ),
            "checks": sum(len(row["checks"]) for row in cases + retained) + 1,
            "failed_checks": sum(
                not value for row in cases + retained for value in row["checks"].values()
            )
            + int(not certificate["positive_rational_margin"]),
            "native_new_trajectories": 0,
        },
        "review_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"kappa": result["kappa"], **result["summary"]}))
    return result


class ShortSegmentTests(unittest.TestCase):
    def setUp(self):
        mp.mp.dps = 90

    def test_fixed_positive_constant_is_interval_certified(self):
        self.assertTrue(interval_constant()["positive_rational_margin"])

    def test_integer_period_variance_at_extreme_centers(self):
        # At ell1: E sin=0, E sin²=.5, Cov=cos(2pi mid)/(2pi).
        for midpoint in (mp.mpf(0), mp.mpf("0.5"), mp.mpf("1000000000.25")):
            _, _, observed = exact_endpoint_variance(
                midpoint - mp.mpf("0.5"), midpoint + mp.mpf("0.5")
            )
            expected = mp.mpf(1) / 3 + (20 * mp.pi) ** 2 / 2 + 40 * mp.cos(2 * mp.pi * midpoint)
            self.assertLess(abs(observed - expected), mp.mpf("1e-60"))

    def test_derivative_and_covariance_constants(self):
        for ell in (mp.mpf(1), mp.mpf("1.13"), mp.mpf(1000)):
            self.assertLess(abs(mp.diff(lower, ell) - lower_derivative(ell)), mp.mpf("1e-70"))
            self.assertGreater(lower_derivative(ell), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__])
    else:
        result = review(args.dataset, args.output)
        raise SystemExit(bool(result["summary"]["failed_checks"]))
