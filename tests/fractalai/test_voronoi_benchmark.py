import pytest
import torch

from fragile.fractalai.core.benchmarks import (
    VoronoiCellVolume,
    VoronoiManifoldVolumeElement,
    VoronoiRicciScalar,
)


def test_voronoi_cell_volume_benchmark_returns_finite_values() -> None:
    pytest.importorskip("scipy")
    benchmark = VoronoiCellVolume(dims=2)

    points = torch.tensor(
        [
            [-10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [10.0, 10.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    values = benchmark(points)

    assert values.shape == (points.shape[0],)
    assert torch.isfinite(values).all()
    assert torch.all(values <= 0.0)
    assert values[-1].item() < 0.0


def test_voronoi_manifold_volume_benchmark_returns_finite_values() -> None:
    pytest.importorskip("scipy")
    benchmark = VoronoiManifoldVolumeElement(dims=2)

    points = torch.tensor(
        [
            [-10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [10.0, 10.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    values = benchmark(points)

    assert values.shape == (points.shape[0],)
    assert torch.isfinite(values).all()
    assert torch.all(values <= 0.0)


def test_voronoi_ricci_benchmark_returns_finite_values() -> None:
    pytest.importorskip("scipy")
    benchmark = VoronoiRicciScalar(dims=2)

    points = torch.tensor(
        [
            [-10.0, -10.0],
            [-10.0, 10.0],
            [10.0, -10.0],
            [10.0, 10.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    values = benchmark(points)
    values_next = benchmark(points + 0.01)

    assert values.shape == (points.shape[0],)
    assert values_next.shape == (points.shape[0],)
    assert torch.isfinite(values).all()
    assert torch.isfinite(values_next).all()
