"""Unit tests for Einstein equation verification helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from fragile.fractalai.qft.einstein_equations import (
    _bootstrap_scalar_regression_cis,
    _compute_knn_density,
    _run_scalar_test_coarse_grained,
    compute_einstein_test,
    EinsteinConfig,
)


def test_einstein_knn_density_returns_finite_positive_values():
    rng = np.random.default_rng(1234)
    positions = rng.normal(size=(64, 3))

    rho = _compute_knn_density(positions, k=8)

    assert rho.shape == (64,)
    assert np.all(np.isfinite(rho))
    assert np.all(rho > 0)


def test_einstein_coarse_grained_scalar_regression_tracks_linear_relation():
    rng = np.random.default_rng(123)
    positions = rng.normal(size=(240, 3))
    radii = np.linalg.norm(positions, axis=1)
    density = 0.5 + radii
    ricci = 2.5 * density - 1.2
    mask = np.ones(positions.shape[0], dtype=bool)

    coarse = _run_scalar_test_coarse_grained(
        positions=positions,
        ricci_scalar=ricci,
        density=density,
        mask=mask,
        n_bins=12,
        min_points_per_bin=5,
        method="radial",
        volumes=None,
        density_mode="knn",
    )

    assert coarse is not None
    r2, slope, intercept, rho_bins, ricci_bins, counts = coarse
    assert r2 > 0.99
    assert abs(slope - 2.5) < 1e-6
    assert abs(intercept + 1.2) < 1e-6
    assert rho_bins.shape == ricci_bins.shape
    assert rho_bins.shape[0] == counts.shape[0]


def test_vectorized_bootstrap_scalar_regression_ci_on_linear_data():
    rng = np.random.default_rng(999)
    n_frames = 4
    n_walkers = 128
    slope_true = 2.0
    intercept_true = -0.7

    density_frames = rng.uniform(0.3, 3.0, size=(n_frames, n_walkers))
    ricci_frames = (
        slope_true * density_frames
        + intercept_true
        + 0.02 * rng.normal(size=(n_frames, n_walkers))
    )
    valid_frames = np.ones((n_frames, n_walkers), dtype=bool)

    (
        r2_ci,
        slope_ci,
        intercept_ci,
        _g_ci,
        _lambda_ci,
        n_valid,
        conf,
    ) = _bootstrap_scalar_regression_cis(
        density_frames=density_frames,
        ricci_frames=ricci_frames,
        valid_frames=valid_frames,
        n_samples=200,
        confidence=0.95,
        seed=123,
        frame_block_size=1,
    )

    assert conf is not None
    assert abs(conf - 0.95) < 1e-12
    assert n_valid > 50
    assert r2_ci is not None
    assert slope_ci is not None
    assert intercept_ci is not None
    assert slope_ci[0] <= slope_true <= slope_ci[1]
    assert intercept_ci[0] <= intercept_true <= intercept_ci[1]


def test_einstein_pipeline_supports_knn_density_and_coarse_graining():
    rng = np.random.default_rng(42)
    n = 40
    d = 3
    n_recorded = 3

    x_final = rng.normal(size=(n_recorded, n, d))
    v_final = rng.normal(size=(n_recorded, n, d))
    fitness = rng.normal(size=(n_recorded - 1, n))
    vol_weights = rng.uniform(0.2, 2.0, size=(n_recorded - 1, n))

    from scipy.spatial import cKDTree

    neighbor_edges: list[np.ndarray] = []
    k = 6
    for t in range(n_recorded - 1):
        tree = cKDTree(x_final[t + 1])
        _, idx = tree.query(x_final[t + 1], k=k + 1)
        edges = []
        for i in range(n):
            for j in idx[i, 1:]:
                edges.append((int(i), int(j)))
                edges.append((int(j), int(i)))

        seen: set[tuple[int, int]] = set()
        unique_edges = []
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)
        neighbor_edges.append(np.asarray(unique_edges, dtype=np.int64))

    history = SimpleNamespace(
        n_recorded=n_recorded,
        x_final=x_final,
        v_final=v_final,
        fitness=fitness,
        diffusion_tensors_full=None,
        fitness_hessians_full=None,
        riemannian_volume_weights=vol_weights,
        neighbor_edges=neighbor_edges,
        fitness_gradients=None,
        ricci_scalar_proxy=None,
        voronoi_regions=None,
    )

    cfg = EinsteinConfig(
        scalar_density_mode="knn",
        knn_k=8,
        coarse_grain_bins=8,
        coarse_grain_min_points=3,
        temporal_average_enabled=True,
        temporal_window_frames=2,
        temporal_stride=1,
        bootstrap_samples=48,
        bootstrap_confidence=0.95,
        bootstrap_seed=77,
        bootstrap_frame_block_size=1,
        fd_regularization=1e-4,
    )
    result = compute_einstein_test(history, cfg)

    assert result.scalar_density_mode == "knn"
    assert result.knn_k == 8
    assert result.scalar_density is not None
    assert result.scalar_density.shape == (n,)
    assert np.all(np.isfinite(result.scalar_density))
    assert np.all(result.scalar_density > 0)

    assert result.scalar_valid_mask is not None
    assert bool(np.any(result.scalar_valid_mask))
    assert result.scalar_r2_coarse is not None
    assert result.scalar_slope_coarse is not None
    assert result.scalar_intercept_coarse is not None
    assert result.scalar_bin_count_coarse is not None
    assert result.scalar_bin_count_coarse >= 3
    assert result.temporal_average_enabled is True
    assert result.temporal_frame_count >= 2
    assert result.temporal_frame_indices is not None
    assert result.scalar_regression_density is not None
    assert result.scalar_regression_ricci is not None
    assert result.scalar_regression_valid_mask is not None
    assert result.scalar_bootstrap_confidence is not None
    assert abs(result.scalar_bootstrap_confidence - 0.95) < 1e-12
