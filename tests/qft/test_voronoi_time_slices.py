import numpy as np
import torch

from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.qft.voronoi_time_slices import compute_time_sliced_voronoi


def test_time_sliced_voronoi_enforces_min_walkers() -> None:
    # Spatial dims: 2, time dim: 2
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.05],
            [0.9, 0.1, 0.05],
            [0.1, 0.9, 0.95],
            [0.9, 0.9, 0.95],
            [0.5, 0.2, 0.95],
            [0.2, 0.5, 0.95],
            [0.8, 0.3, 0.95],
            [0.3, 0.8, 0.95],
            [0.6, 0.6, 0.95],
            [0.7, 0.7, 0.95],
        ],
        dtype=torch.float32,
    )
    bounds = TorchBounds(
        low=torch.tensor([0.0, 0.0, 0.0]),
        high=torch.tensor([1.0, 1.0, 1.0]),
        shape=(3,),
    )

    result = compute_time_sliced_voronoi(
        positions=positions,
        time_dim=2,
        n_bins=2,
        min_walkers_bin=1,
        bounds=bounds,
        compute_curvature=False,
    )

    assert len(result.bins) == 1
    assert result.bins[0].indices.size == positions.shape[0]
    assert result.timelike_edges.shape[0] == 0
    assert result.bins[0].spacelike_edges.shape[1] == 2


def test_time_sliced_voronoi_bin_edges_match_bins() -> None:
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.4],
            [0.0, 1.0, 0.7],
            [1.0, 1.0, 0.9],
        ],
        dtype=torch.float32,
    )
    bounds = TorchBounds(
        low=torch.tensor([0.0, 0.0, 0.0]),
        high=torch.tensor([1.0, 1.0, 1.0]),
        shape=(3,),
    )

    result = compute_time_sliced_voronoi(
        positions=positions,
        time_dim=2,
        n_bins=2,
        min_walkers_bin=3,
        bounds=bounds,
        compute_curvature=False,
    )

    assert len(result.bin_edges) == len(result.bins) + 1
    assert np.all(np.diff(result.bin_edges) > 0)
