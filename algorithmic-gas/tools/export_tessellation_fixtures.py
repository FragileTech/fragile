"""Export reference values of the tessellation geometry for the Rust parity tests.

Runs the Python estimators of ``fragile.physics.geometry`` on small point sets
in float64 and writes one JSON document per case to
``crates/algorithmic-gas/tests/fixtures/tessellation``. Points are pseudo-random
(generic position): for cocircular sites Qhull and Spade may legitimately pick
different Delaunay diagonals, so lattices are not comparable edge by edge. The
degenerate cases (coincident, collinear and coplanar swarms) compare the edge
sets produced by the documented lifting and projection rules.

Usage: uv run python algorithmic-gas/tools/export_tessellation_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fragile.physics.fractal_gas.kinetic_operator import KineticOperator
from fragile.physics.geometry.delaunai import build_delaunay_edges, compute_delaunay_data


OUTPUT = (
    Path(__file__).resolve().parents[1] / "crates/algorithmic-gas/tests/fixtures/tessellation"
)
WEIGHT_MODES = [
    "uniform",
    "inverse_distance",
    "inverse_volume",
    "inverse_riemannian_volume",
    "inverse_riemannian_distance",
    "kernel",
    "riemannian_kernel",
    "riemannian_kernel_volume",
]


def geometry_case(name: str, n: int, d: int, seed: int, length_scale: float) -> dict:
    rng = np.random.default_rng(seed)
    positions = torch.tensor(rng.uniform(-1.0, 1.0, size=(n, d)), dtype=torch.float64)
    data = compute_delaunay_data(
        positions,
        torch.zeros(n, dtype=torch.float64),
        weight_modes=WEIGHT_MODES,
        compute_full_ricci=True,
        length_scale=length_scale,
    )
    case = {
        "name": name,
        "dimension": d,
        "length_scale": length_scale,
        "positions": positions.tolist(),
        "edges": data.edge_index.t().tolist(),
        "metric": data.metric_tensors.tolist(),
        "metric_det": data.metric_det.tolist(),
        "volume": data.riemannian_volume_weights.tolist(),
        "diffusion": data.diffusion_tensors.tolist(),
        "edge_distances": data.edge_distances.tolist(),
        "edge_geodesic": data.edge_geodesic_distances.tolist(),
        "weights": {mode: w.tolist() for mode, w in data.edge_weights.items()},
        "ricci_proxy": data.ricci_proxy.tolist(),
        "ricci_proxy_full": data.ricci_proxy_full.tolist(),
        "ricci_tensor_full": data.ricci_tensor_full.tolist(),
    }
    # Graph forces on the same graph: viscous force, curl and one Boris B step.
    velocities = torch.tensor(rng.normal(size=(n, d)), dtype=torch.float64)
    nu, beta_curl, dt = 3.0, 1.0, 0.05
    op = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=dt,
        nu=nu,
        beta_curl=beta_curl,
        use_viscous_coupling=True,
        viscous_neighbor_weighting="riemannian_kernel_volume",
        viscous_length_scale=length_scale,
    )
    edges = data.edge_index.t().contiguous()
    weights = data.edge_weights["riemannian_kernel_volume"]
    force = op._compute_viscous_force(
        positions, velocities, neighbor_edges=edges, edge_weights=weights
    )
    curl = op._compute_viscous_curl(positions, force, edges, weights)
    kicked, _ = op._apply_boris_kick(
        positions, velocities, neighbor_edges=edges, edge_weights=weights
    )
    case["forces"] = {
        "nu": nu,
        "beta_curl": beta_curl,
        "dt": dt,
        "velocities": velocities.tolist(),
        "viscous_force": force.tolist(),
        "curl": curl.tolist(),
        "kicked_velocities": kicked.tolist(),
        "rotation_angle": op._last_boris_angle.tolist(),
    }
    return case


def degenerate_case(name: str, positions: list[list[float]]) -> dict:
    edges = build_delaunay_edges(np.asarray(positions, dtype=np.float64))
    return {"name": name, "positions": positions, "edges": edges.tolist()}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        geometry_case("cloud_2d", 60, 2, seed=20260918, length_scale=1.0),
        geometry_case("cloud_3d", 80, 3, seed=20260919, length_scale=0.7),
    ]
    line = [[0.5 * k, 1.0 - 0.25 * k, 2.0 * k] for k in range(6)]
    plane = [[x, y, 0.5 * x - y] for x, y in [(0, 0), (1, 0), (0, 1), (1.2, 1.1), (0.4, 0.45)]]
    duplicates = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    degenerate = [
        degenerate_case("coincident", [[0.25, -0.5, 1.0]] * 5),
        degenerate_case("collinear_3d", line),
        degenerate_case("coplanar_3d", plane),
        degenerate_case("duplicates_2d", duplicates),
    ]
    for case in cases:
        (OUTPUT / f"{case['name']}.json").write_text(json.dumps(case))
    (OUTPUT / "degenerate.json").write_text(json.dumps(degenerate))
    print(f"wrote {len(cases) + 1} fixtures to {OUTPUT}")


if __name__ == "__main__":
    main()
