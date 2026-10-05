"""Exact finite empirical-law endpoint transport for native dimension probes."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.sparse import coo_matrix


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport(cost: np.ndarray) -> float:
    left, right = cost.shape
    if left == right:
        i, j = linear_sum_assignment(cost)
        return float(cost[i, j].mean())
    edges = np.arange(left * right)
    rows = np.concatenate([edges // right, left + edges % right])
    columns = np.concatenate([edges, edges])
    matrix = coo_matrix(
        (np.ones(2 * left * right), (rows, columns)),
        shape=(left + right, left * right),
    ).tocsr()
    mass = np.concatenate([np.full(left, 1 / left), np.full(right, 1 / right)])
    result = linprog(cost.ravel(), A_eq=matrix, b_eq=mass, bounds=(0, None), method="highs")
    if not result.success or np.max(np.abs(matrix @ result.x - mass)) > 1e-9:
        raise RuntimeError("Empirical-law transport solve failed its marginal check.")
    return float(result.fun)


def main() -> None:
    dataset = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    if destination.exists():
        raise FileExistsError("Endpoint reviews are immutable; choose a new destination.")
    index = json.loads((dataset / "archive-index.json").read_text())
    if index["status"] != "complete":
        raise ValueError("Native dimension dataset must be committed before review.")
    results = []
    inputs = []
    for case in [100, 101]:
        entry = next(
            e for e in index["entries"]
            if e["tag"] == f"chapter06-case{case}-trajectory-manifest"
        )
        path = dataset / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError("Trajectory manifest SHA256 differs from its committed index.")
        inputs.append({"path": str(path), "sha256": entry["sha256"]})
        manifest = json.loads(gzip.decompress(path.read_bytes()))
        config = manifest["native_config"]
        n, d = config["walkers"], config["dimensions"]
        half_width = config["gas"]["boundary"]["domain"]["upper"][0]
        cap = config["gas"]["kinetic"]["velocity_cap"]
        scale = 1 + 4 * d * half_width**2 + 4 * cap**2
        groups = [
            [t for t in manifest["trajectories"] if t["distribution"] == distribution]
            for distribution in [0, 1]
        ]
        endpoint_rows = []
        for step in [0, max(manifest["checkpoints"])]:
            states = [
                [
                    next(s for s in t["checkpoints"] if s["step"] == step)
                    for t in group
                    if t["native_extinction_step"] is None
                    or t["native_extinction_step"] > step
                ]
                for group in groups
            ]
            row = {"step": step, "own_survivors": [len(s) for s in states]}
            if any(not s for s in states):
                row.update({"available": False, "reason": "Zero own survivor denominator."})
                endpoint_rows.append(row)
                continue
            x, y = [np.asarray([s["phases"] for s in group]) for group in states]
            outer_cost = np.zeros((len(x), len(y)))
            for i, swarm in enumerate(x):
                pair_costs = ((swarm[None, :, None, :] - y[:, None, :, :]) ** 2).sum(-1)
                for j, pair_cost in enumerate(pair_costs):
                    a, b = linear_sum_assignment(pair_cost)
                    outer_cost[i, j] = pair_cost[a, b].sum() / n / scale
            alive_x, alive_y = [
                np.asarray([s["uniform_alive_sample"] for s in group]) for group in states
            ]
            alive_cost = ((alive_x[:, None, :] - alive_y[None, :, :]) ** 2).sum(-1) / scale
            row.update({
                "available": True,
                "whole_swarm_exact_empirical_law_cost": transport(outer_cost),
                "uniform_alive_exact_empirical_law_cost": transport(alive_cost),
            })
            endpoint_rows.append(row)
        ratio = {}
        if all(r["available"] for r in endpoint_rows):
            for field in [
                "whole_swarm_exact_empirical_law_cost",
                "uniform_alive_exact_empirical_law_cost",
            ]:
                ratio[field] = endpoint_rows[-1][field] / endpoint_rows[0][field]
        results.append({
            "case": case, "N": n, "d": d,
            "independent_seeds_per_initial_law": [len(g) for g in groups],
            "phase_cost_scale": scale, "endpoints": endpoint_rows,
            "terminal_to_initial_ratios": ratio,
            "metric": (
                "N^-1 minimum-permutation squared phase cost; alive-masked positions, "
                "status and original capped velocities; dead storage positions discarded"
            ),
        })
    report = {
        "cases": results,
        "provenance": {
            "input_index_sha256": sha256(dataset / "archive-index.json"),
            "input_manifests": inputs,
            "helper_path": str(Path(__file__).resolve()),
            "helper_sha256": sha256(Path(__file__).resolve()),
            "new_native_steps": 0,
        },
        "scope": (
            "Exact minima for finite surviving empirical laws, using own denominators. "
            "Independent trajectory seeds are sampling units. Endpoint decline is not "
            "a QSD distance or a certified infinite-time law convergence rate; "
            "transport solvers use floating arithmetic."
        ),
    }
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
