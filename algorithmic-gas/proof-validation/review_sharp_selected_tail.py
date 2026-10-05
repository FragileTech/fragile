"""Independent high-precision audit of SLM.8--11 and saved Rust coefficients."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import unittest

import mpmath as mp


def number(value):
    return mp.mpf(str(value))


def constants(config, d, p):
    source = config["cloning_donors"]
    metric = source["distance"]
    fitness = config["fitness"]
    gate = config["clone_decision"]
    kinetic = config["kinetic"]
    h = number(kinetic["integrator"]["dt"])
    gamma = number(kinetic["integrator"]["friction"])
    alpha = number(config["clone_transform"]["restitution"] or 0)
    vcap = number(kinetic["velocity_cap"])
    sigma_j = number(config["clone_transform"]["jitter_amplitude"])
    amplitude = number(kinetic["noise"]["geometry"]["scale"]["values"][0])
    q = amplitude * mp.sqrt(h if gamma == 0 else (1 - mp.exp(-2 * gamma * h)) / (2 * gamma))
    s = number(kinetic["position_diffusion"]) * mp.sqrt(h)
    log_min, log_span = mp.mpf(0), mp.mpf(0)
    for channel in ("reward", "diversity"):
        mapping = fitness[f"{channel}_map"]
        exponent = number(fitness[f"{channel}_exponent"])
        floor = number(mapping["floor"])
        log_min += exponent * mp.log(floor)
        log_span += exponent * mp.log(1 + number(mapping["amplitude"]) / floor)
    accept = min(
        mp.mpf(1),
        mp.exp(log_min)
        * mp.expm1(log_span)
        / (number(gate["saturation"]) * (mp.exp(log_min) + number(gate["epsilon"]))),
    )
    diameter = 4 * (
        number(metric["position_radius"]) ** 2
        + number(metric["lambda"]) * number(metric["velocity_radius"]) ** 2
    )
    kappa = mp.exp(-diameter / (2 * number(source["kernel"]["width"]) ** 2))
    history = source["history_window"]
    coefficient = 1 + accept / kappa * (mp.mpf(4) / 3 if history else 1)
    a = abs(1 - 2 * h**2 / 4 * (1 + mp.exp(-gamma * h)))
    b = h / 2 * (1 + mp.exp(-gamma * h))
    eta = h / 2 * b
    even = 2 * math.ceil(p / 2)
    gaussian_lp = mp.fprod(d + 2 * j for j in range(even // 2)) ** (mp.mpf(1) / even)
    collision = vcap * (1 + 2 * alpha) ** max(mp.mpf(0), 1 - mp.mpf(2) / p)
    jitter = sigma_j * gaussian_lp * accept ** (mp.mpf(1) / p)
    budget = (
        a * jitter
        + b * collision
        + eta * 20 * mp.pi * mp.sqrt(d)
        + mp.sqrt((h * q / 2) ** 2 + s**2) * gaussian_lp
    )
    root = a * coefficient ** (mp.mpf(1) / p)
    lam = (1 + a**p) / 2 if p > 1 else a
    epsilon = (lam / a**p) ** (mp.mpf(1) / (p - 1)) - 1 if p > 1 else None
    additive = (1 + 1 / epsilon) ** (p - 1) * budget**p if p > 1 else budget
    young = lam * coefficient
    gate_interval = mp.log(
        1
        + number(gate["saturation"]) * kappa * (a ** (-p) - 1) * (mp.mpf(3) / 4 if history else 1)
    )
    return {
        "a": a,
        "kappa": kappa,
        "acceptance": accept,
        "source_coefficient": coefficient,
        "collision_lp": collision,
        "jitter_lp": jitter,
        "additive_lp": budget,
        "root_coefficient": root,
        "root_floor": budget / (1 - root) if root < 1 else None,
        "moment_floor": (budget / (1 - root)) ** p if root < 1 else None,
        "young_coefficient": young,
        "young_additive": additive,
        "young_floor": additive / (1 - young) if young < 1 else None,
        "eta_f": log_span,
        "sufficient_eta_f_upper": gate_interval,
    }


def close(a, b):
    return abs(number(a) - b) <= mp.mpf("2e-10") * max(mp.mpf("1e-30"), abs(b))


def review(dataset, output):
    mp.mp.dps = 80
    index_path = dataset / "archive-index.json"
    index = json.loads(index_path.read_text())
    blobs, verified = {}, []
    for entry in index["entries"]:
        path = dataset / entry["path"]
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        verified.append({"path": entry["path"], "sha256": sha})
        blobs[entry["tag"]] = json.loads(gzip.decompress(path.read_bytes()))
    report = json.loads((dataset / "report.json").read_text())
    profiles, lookup = [], {}
    for profile in report["profiles"]:
        d, p = profile["d"], profile["moment_order"]
        if not profile["law_group"].startswith("rastrigin"):
            message = "Only actual separable Rastrigin profile supported by this independent audit"
            raise ValueError(message)
        expected = constants(profile["parameters"], d, p)
        refined = profile["refined_closure"]
        root = profile["root_closure"]
        checks = {
            "actual_first_matrix_applicability": bool(
                profile["actual_native_viscosity"] == 0
                or profile["actual_native_normalization"] == "count"
            ),
            "collision_energy_interpolation": close(
                refined["collision_normalized_lp_upper"], expected["collision_lp"]
            ),
            "independent_gate_weighted_jitter": close(
                refined["gate_weighted_jitter_lp_upper"], expected["jitter_lp"]
            ),
            "Minkowski_additive_budget": close(
                root["additive_root_budget"], expected["additive_lp"]
            ),
            "root_multiplier": close(
                root["root_moment_coefficient"], expected["root_coefficient"]
            ),
            "root_floor": close(root["invariant_root_moment_upper"], expected["root_floor"]),
            "unrooted_floor_only": close(root["invariant_moment_upper"], expected["moment_floor"]),
            "floor_dominates_Young": bool(
                expected["young_floor"] is None
                or expected["moment_floor"] <= expected["young_floor"]
            ),
            "saturation_interval": bool(expected["eta_f"] < expected["sufficient_eta_f_upper"]),
        }
        row = {
            "law_group": profile["law_group"],
            "N": profile["N"],
            "dimension": d,
            "moment_order": p,
            "analytic_constants": {
                k: float(v) if v is not None else None for k, v in expected.items()
            },
            "checks": checks,
            "passed": all(checks.values()),
        }
        profiles.append(row)
        lookup[profile["law_group"], p] = expected
    rows, fails = 0, 0
    for tag, blob in blobs.items():
        if not tag.startswith("weak-moment-batch"):
            continue
        for row in blob["rows"]:
            expected = lookup[row["law_group"], row["moment_order"]]
            root_prediction = (
                expected["root_coefficient"]
                * number(row["before"]) ** (mp.mpf(1) / row["moment_order"])
                + expected["additive_lp"]
            ) ** row["moment_order"]
            fails += not close(row["root_complete_bound"], root_prediction)
            rows += 1
    source = Path(
        "docs/source/2_fractal_gas/convergence_program/06a_structural_landscape_convergence.md"
    )
    result = {
        "source_label": "cor-slc-sharp-selected-tail-budget",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dataset": str(dataset),
        "index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
        "verified_entries": verified,
        "profiles": profiles,
        "arithmetic_scope": "Independent80-digit formula audit; no directed arithmetic claim for these floors/rates.",
        "scope": "Expected root-moment recurrence only, not a linear unrooted excess or physical swarm-law rate. All full Gaussian and current-frame energy/independence hypotheses retained.",
        "summary": {
            "profiles": len(profiles),
            "retained_row_predictions": rows,
            "checks": sum(len(p["checks"]) for p in profiles) + rows,
            "failed_checks": sum(not value for p in profiles for value in p["checks"].values())
            + fails,
            "new_native_steps": 0,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"]))
    return result


class SharpTailTests(unittest.TestCase):
    def test_gate_weighted_gaussian_fourth_moment(self):
        accept, sigma = 0.13, 0.1
        nodes = [(-math.sqrt(3), 1 / 6), (0.0, 2 / 3), (math.sqrt(3), 1 / 6)]
        actual = sum(accept * weight * (sigma * z) ** 4 for z, weight in nodes)
        self.assertAlmostEqual(actual, accept * sigma**4 * 3, places=15)

    def test_component_energy_Lp_interpolation_and_need_for_columns(self):
        for n in (4, 16, 64):
            original = [-1.0] + [1.0] * (n - 1)
            center = sum(original) / n
            collided = [center - (v - center) for v in original]
            self.assertAlmostEqual(sum(v * v for v in collided), n)
            for p in (1.0, 2.0, 4.0, 8.0):
                norm = (sum(abs(v) ** p for v in collided) / n) ** (1 / p)
                self.assertLessEqual(norm, 3 ** max(0, 1 - 2 / p) + 1e-14)
            self.assertGreater(
                max(abs(v) for v in collided), math.sqrt(sum(v * v for v in collided) / n)
            )
            # A row-stochastic map that copies the extremal row into every row
            # increases energy. It must not receive a doubly-stochastic gate.
            self.assertGreater(n * max(v * v for v in collided), sum(v * v for v in collided))

    def test_root_floor_does_not_give_linear_unrooted_excess_rate(self):
        r, c, p = 0.8, 0.2, 4
        floor = c / (1 - r)
        u = floor + 0.01
        exact_next = (r * u + c) ** p - floor**p
        invalid_claim = r**p * (u**p - floor**p)
        self.assertGreater(exact_next, invalid_claim)


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
