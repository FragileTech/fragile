"""Throwaway benchmark for cached-hull Voronoi measures.

Compares:
- Existing _compute_all_cell_volumes + _compute_all_facet_areas
- New compute_cached_volumes_and_facet_areas

Run:
  uv run python tools/benchmark_voronoi_hull_cache.py
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from scipy.spatial import Voronoi

from fragile.fractalai.scutoid.voronoi import (
    _compute_all_cell_volumes,
    _compute_all_facet_areas,
)
from fragile.fractalai.scutoid.voronoi_hull_cache import (
    compute_cached_volumes_and_facet_areas,
)


def _timeit(fn, repeats: int) -> list[float]:
    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cached-hull Voronoi measures.")
    parser.add_argument("--walkers", type=int, default=500)
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_walkers = args.walkers
    d = args.dim
    repeats = args.repeats
    seed = args.seed

    rng = np.random.default_rng(seed)
    positions = rng.standard_normal((n_walkers, d))

    vor = Voronoi(positions)
    ridge_points = vor.ridge_points
    ridge_vertices = vor.ridge_vertices
    n_edges = ridge_points.shape[0] * 2

    device = torch.device("cpu")
    dtype = torch.float64

    def baseline() -> tuple[torch.Tensor, torch.Tensor]:
        volumes = _compute_all_cell_volumes(vor, n_walkers, d, device, dtype)
        facets = _compute_all_facet_areas(vor, ridge_points, n_edges, d, device, dtype)
        return volumes, facets

    def cached() -> tuple[torch.Tensor, torch.Tensor]:
        return compute_cached_volumes_and_facet_areas(
            vor=vor,
            positions=positions,
            ridge_points=ridge_points,
            ridge_vertices=ridge_vertices,
            n_alive=n_walkers,
            d=d,
            device=device,
            dtype=dtype,
        )

    # Warmup
    baseline()
    cached()

    baseline_volumes, baseline_facets = baseline()
    cached_volumes, cached_facets = cached()

    vol_diff = (baseline_volumes - cached_volumes).abs()
    fac_diff = (baseline_facets - cached_facets).abs()

    vol_close = torch.isclose(baseline_volumes, cached_volumes, rtol=1e-5, atol=1e-7)
    fac_close = torch.isclose(baseline_facets, cached_facets, rtol=1e-5, atol=1e-7)

    print("Correctness checks")
    print(f"Volumes: max={vol_diff.max():.6g} mean={vol_diff.mean():.6g} mismatches={int((~vol_close).sum())}")
    print(f"Facets:  max={fac_diff.max():.6g} mean={fac_diff.mean():.6g} mismatches={int((~fac_close).sum())}")

    baseline_times = _timeit(baseline, repeats)
    cached_times = _timeit(cached, repeats)

    baseline_avg = sum(baseline_times) / len(baseline_times)
    cached_avg = sum(cached_times) / len(cached_times)
    speedup = baseline_avg / cached_avg if cached_avg > 0 else float("inf")

    print("\nTiming (seconds)")
    print(f"Baseline avg: {baseline_avg:.6f} (min {min(baseline_times):.6f}, max {max(baseline_times):.6f})")
    print(f"Cached   avg: {cached_avg:.6f} (min {min(cached_times):.6f}, max {max(cached_times):.6f})")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
