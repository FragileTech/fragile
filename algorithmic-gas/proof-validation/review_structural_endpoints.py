"""Exact empirical-law phase transport from retained nonquadratic native runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import operator
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical_transform(states: list, n: int, d: int, omega: float, beta: float) -> np.ndarray:
    points = np.asarray(states, dtype=float)
    if points.shape != (n, 2 * d) or not np.isfinite(points).all():
        msg = "Retained endpoint does not have N finite [x...,v...] points."
        raise ValueError(msg)
    x, v = points[:, :d], points[:, d:]
    return np.concatenate([np.sqrt(omega) * x + beta * v, np.sqrt(1 - beta**2) * v], 1)


def empirical_law_cost(left: np.ndarray, right: np.ndarray) -> tuple[float, float, list]:
    if len(left) != len(right):
        msg = "Unequal replicate counts require a nonuniform transport solver."
        raise ValueError(msg)
    cost = np.zeros((len(left), len(right)))
    for i, swarm in enumerate(left):
        phase_costs = ((swarm[None, :, None, :] - right[:, None, :, :]) ** 2).sum(-1)
        for j, pair_cost in enumerate(phase_costs):
            row, col = linear_sum_assignment(pair_cost)
            cost[i, j] = pair_cost[row, col].mean()
    row, col = linear_sum_assignment(cost)
    exact = float(cost[row, col].mean())
    diagonal = float(np.diag(cost).mean())
    plan = [
        {"left_replicate": int(i), "right_replicate": int(j), "mass": 1 / len(left)}
        for i, j in zip(row, col, strict=True)
    ]
    return exact, diagonal, plan


def main() -> None:
    dataset = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    if destination.exists():
        msg = "Derived reviews are immutable; choose a fresh output directory."
        raise FileExistsError(msg)
    index_path = dataset / "archive-index.json"
    index = json.loads(index_path.read_text())
    if index["status"] != "complete":
        msg = "Wait for the native dataset to be fully committed."
        raise ValueError(msg)
    entries = {entry["path"]: entry for entry in index["entries"]}
    consumed = []

    def read_entry(entry: dict) -> dict:
        path = dataset / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"SHA256 mismatch in {entry['path']}.")
        consumed.append({"path": str(path), "tag": entry["tag"], "sha256": entry["sha256"]})
        return json.loads(gzip.decompress(path.read_bytes()))

    report_entry = next(e for e in entries.values() if e["tag"] == "structural-report")
    source_entry = next(e for e in entries.values() if e["tag"] == "structural-source")
    report = read_entry(report_entry)
    source = read_entry(source_entry)
    beta = float(report["harmonic_reference"]["beta"])
    omega = 2.0
    if not math.isclose(
        float(report["harmonic_reference"]["curvature"]), omega, rel_tol=0, abs_tol=1e-14
    ):
        msg = "This endpoint review requires the declared omega=2 reference."
        raise ValueError(msg)
    if not np.isfinite(beta) or abs(beta) >= 1:
        msg = "The declared G_omega,beta phase metric must be positive definite."
        raise ValueError(msg)
    results = []
    comparisons = []
    for case in report["cases"]:
        observations = read_entry(entries[case["observations"]])["rows"]
        n, d, samples = int(case["N"]), int(case["d"]), int(case["samples"])
        frames = []
        for step in [0, int(case["steps"])]:
            endpoints = sorted(
                [r for r in observations if r["step"] == step],
                key=operator.itemgetter("replicate"),
            )
            if len(endpoints) != samples or [r["replicate"] for r in endpoints] != list(
                range(samples)
            ):
                msg = "Every requested replicate needs its own retained endpoint."
                raise ValueError(msg)
            transformed = [
                np.stack([
                    physical_transform(r["endpoint_states"][side], n, d, omega, beta)
                    for r in endpoints
                ])
                for side in ["left", "right"]
            ]
            exact, diagonal, plan = empirical_law_cost(*transformed)
            reported_pair = float(np.mean([r["coupling_error"] for r in endpoints]))
            tolerance = 1e-10 * max(1.0, diagonal, reported_pair)
            comparisons.extend([
                {
                    "id": f"d{d}-n{n}-{case['initial_zone']}-step{step}-law_vs_pairs",
                    "observed": exact,
                    "bound": diagonal,
                    "relation": "upper",
                    "passed": exact <= diagonal + tolerance,
                    "scope": "Exact empirical-law minimum versus replicate-matched coupling.",
                },
                {
                    "id": f"d{d}-n{n}-{case['initial_zone']}-step{step}-pairs_vs_native",
                    "observed": diagonal,
                    "bound": reported_pair,
                    "relation": "upper",
                    "passed": diagonal <= reported_pair + tolerance,
                    "scope": "Permutation minimum versus the retained native pair coupling.",
                },
            ])
            frames.append({
                "step": step,
                "replicates_per_law": samples,
                "whole_swarm_exact_empirical_law_cost": exact,
                "replicate_matched_permutation_coupling_cost": diagonal,
                "reported_native_coupling_cost": reported_pair,
                "optimal_outer_plan": plan,
            })
        first, last = frames
        initial = first["whole_swarm_exact_empirical_law_cost"]
        terminal = last["whole_swarm_exact_empirical_law_cost"]
        results.append({
            "d": d,
            "N": n,
            "initial_zone": case["initial_zone"],
            "selected_cloning": case["selected_cloning"],
            "landscape": case["landscape"],
            "omega": omega,
            "beta": beta,
            "frames": frames,
            "terminal_to_initial_ratio": terminal / initial if initial > 0 else None,
            "empirical_endpoint_decline": terminal < initial,
        })
    failures = sum(not c["passed"] for c in comparisons)
    output = {
        "cases": results,
        "comparisons": comparisons,
        "summary": {
            "cases": len(results),
            "comparisons": len(comparisons),
            "comparisons_failed": failures,
            "new_native_steps": 0,
        },
        "metric": (
            "Inner N^-1 minimum-permutation G_omega,beta squared phase cost; "
            "outer minimum transport over the finite empirical seed laws. "
            "G=[[omega,beta*sqrt(omega)],[beta*sqrt(omega),1]], omega=2."
        ),
        "scope": (
            "Finite empirical whole-swarm law comparisons. Independent replicate seed "
            "units; left/right in a pair may share innovations. The matching removes "
            "walker labels. Endpoint decline is not a certified population law rate, "
            "QSD distance, or stationary-law error. Assignment solvers use f64."
        ),
        "provenance": {
            "dataset": str(dataset),
            "input_index_sha256": digest(index_path),
            "consumed_indexed_files": consumed,
            "embedded_source_sha256": {
                key: hashlib.sha256(value.encode()).hexdigest()
                for key, value in source.items()
                if isinstance(value, str)
            },
            "executed_helper_sha256": digest(Path(__file__).resolve()),
            "beta_source": "indexed structural-report.harmonic_reference.beta",
        },
    }
    destination.mkdir(parents=True)
    (destination / "exact-empirical-law-endpoints.json").write_text(json.dumps(output, indent=2))
    text = (
        "# Nonquadratic empirical whole-law endpoints\n\n"
        "Inner matching minimizes the N-normalized G phase cost over walker permutations; "
        "outer matching minimizes transport over the finite empirical seed laws. "
        "Each row retains its own replicate count. Paired left/right trajectories may "
        "share innovations; independent replicate seeds supply the sampling units.\n\n"
        "| d | N | Selected cloning | Initial zone | Seeds per law | Initial cost | "
        "Terminal cost | Terminal/initial | Observed decline |\n"
        "|---:|---:|---|---|---:|---:|---:|---:|---|\n"
    )
    for case in results:
        first, last = case["frames"]
        ratio = case["terminal_to_initial_ratio"]
        ratio_text = f"{ratio:.8g}" if ratio is not None else "Unavailable"
        text += (
            f"| {case['d']} | {case['N']} | {case['selected_cloning']} | "
            f"{case['initial_zone']} | {first['replicates_per_law']} | "
            f"{first['whole_swarm_exact_empirical_law_cost']:.9g} | "
            f"{last['whole_swarm_exact_empirical_law_cost']:.9g} | "
            f"{ratio_text} | "
            f"{case['empirical_endpoint_decline']} |\n"
        )
    text += (
        f"\n{len(comparisons)} transport-coupling consistency checks, {failures} discrepancies. "
        "These empirical endpoints do not certify a full-law population mixing rate. "
        "All consumed observations and source archives are bound by indexed SHA256; "
        "the review advances no native trajectory steps.\n"
    )
    (destination / "endpoint-table.md").write_text(text)
    (destination / "executed-helper.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(output["summary"]))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
