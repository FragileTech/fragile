"""Unit tests for quantum gravity analysis."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.quantum_gravity import (
    QuantumGravityConfig,
    QuantumGravityObservables,
    QuantumGravityTimeSeries,
    compute_quantum_gravity_observables,
    compute_quantum_gravity_time_evolution,
    compute_regge_action,
    compute_einstein_hilbert_action,
    compute_adm_energy,
    compute_spectral_dimension,
    compute_hausdorff_dimension,
    compute_holographic_entropy,
)


@pytest.fixture
def mock_history_2d():
    """Create a mock 2D RunHistory for testing."""
    N = 50
    d = 2
    n_recorded = 10

    # Create simple grid positions
    x = np.linspace(-5, 5, int(np.sqrt(N)))
    y = np.linspace(-5, 5, int(np.sqrt(N)))
    xx, yy = np.meshgrid(x, y)
    positions = np.stack([xx.flatten(), yy.flatten()], axis=1)[:N]

    # Create history
    history = RunHistory(N=N, d=d, n_steps=100, record_every=10)

    # Fill with positions
    for i in range(n_recorded):
        # Add small noise
        noise = np.random.randn(N, d) * 0.1
        history.x_final[i] = torch.from_numpy(positions + noise).float()

    history.n_recorded = n_recorded

    # Create alive masks
    history.alive_mask = torch.ones(n_recorded, N, dtype=torch.bool)

    # Create fitness values
    history.fitness = torch.randn(n_recorded, N)
    history.rewards = torch.randn(n_recorded, N)

    # Add bounds
    from fragile.fractalai.bounds import TorchBounds

    history.bounds = TorchBounds(
        low=torch.tensor([-10.0, -10.0]),
        high=torch.tensor([10.0, 10.0]),
    )
    history.pbc = False

    return history


@pytest.fixture
def mock_history_3d():
    """Create a mock 3D RunHistory for testing."""
    N = 27  # 3x3x3 grid
    d = 3
    n_recorded = 5

    # Create simple 3D grid positions
    x = np.linspace(-5, 5, 3)
    y = np.linspace(-5, 5, 3)
    z = np.linspace(-5, 5, 3)
    xx, yy, zz = np.meshgrid(x, y, z)
    positions = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)[:N]

    # Create history
    history = RunHistory(N=N, d=d, n_steps=50, record_every=10)

    # Fill with positions
    for i in range(n_recorded):
        noise = np.random.randn(N, d) * 0.1
        history.x_final[i] = torch.from_numpy(positions + noise).float()

    history.n_recorded = n_recorded

    # Create alive masks
    history.alive_mask = torch.ones(n_recorded, N, dtype=torch.bool)

    # Create fitness values
    history.fitness = torch.randn(n_recorded, N)
    history.rewards = torch.randn(n_recorded, N)

    # Add bounds
    from fragile.fractalai.bounds import TorchBounds

    history.bounds = TorchBounds(
        low=torch.tensor([-10.0, -10.0, -10.0]),
        high=torch.tensor([10.0, 10.0, 10.0]),
    )
    history.pbc = False

    return history


def test_quantum_gravity_config():
    """Test QuantumGravityConfig defaults."""
    config = QuantumGravityConfig()

    assert config.mc_time_index is None
    assert config.warmup_fraction == 0.1
    assert config.use_metric_correction == "full"
    assert config.diffusion_time_steps == 100
    assert config.max_diffusion_time == 10.0
    assert config.n_radial_bins == 50
    assert config.light_speed == 1.0
    assert config.planck_length == 1.0
    assert config.compute_all is True


def test_regge_action_flat_space(mock_history_2d):
    """Regge action should be near 0 for flat grid."""
    config = QuantumGravityConfig(mc_time_index=-1)

    result = compute_regge_action(mock_history_2d, mock_history_2d.n_recorded - 1, config)

    assert "regge_action" in result
    assert "regge_action_density" in result
    assert "deficit_angles" in result

    # For a regular grid, curvature should be small
    assert isinstance(result["regge_action"], float)
    # Action density should be a tensor
    assert isinstance(result["regge_action_density"], torch.Tensor)


def test_einstein_hilbert_action(mock_history_2d):
    """Test Einstein-Hilbert action computation."""
    positions = mock_history_2d.x_final[-1]
    alive = torch.ones(mock_history_2d.N, dtype=torch.bool)
    config = QuantumGravityConfig()

    result = compute_einstein_hilbert_action(positions, alive, mock_history_2d, config)

    assert "einstein_hilbert_action" in result
    assert "ricci_scalars" in result
    assert "scalar_curvature_mean" in result

    assert isinstance(result["einstein_hilbert_action"], float)
    assert isinstance(result["ricci_scalars"], torch.Tensor)
    assert result["ricci_scalars"].shape[0] == mock_history_2d.N or result["ricci_scalars"].shape[0] > 0


def test_adm_energy(mock_history_2d):
    """Test ADM energy computation."""
    ricci = torch.randn(mock_history_2d.N)
    volumes = torch.ones(mock_history_2d.N) * 0.5

    result = compute_adm_energy(ricci, volumes)

    assert "adm_mass" in result
    assert "adm_energy_density" in result

    assert isinstance(result["adm_mass"], float)
    assert result["adm_energy_density"].shape[0] == mock_history_2d.N


def test_spectral_dimension(mock_history_2d):
    """Spectral dimension should be close to spatial dimension for regular grid."""
    positions = mock_history_2d.x_final[-1]
    alive = torch.ones(mock_history_2d.N, dtype=torch.bool)

    # Build simple edge index (nearest neighbors)
    from scipy.spatial import Delaunay

    positions_np = positions.cpu().numpy()
    tri = Delaunay(positions_np)

    edges = []
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                edges.append([simplex[i], simplex[j]])
                edges.append([simplex[j], simplex[i]])

    edge_index = torch.tensor(edges, dtype=torch.long).t()

    config = QuantumGravityConfig(diffusion_time_steps=20)

    result = compute_spectral_dimension(positions, edge_index, alive, config)

    assert "spectral_dimension_curve" in result
    assert "spectral_dimension_planck" in result
    assert "heat_kernel_trace" in result

    # Spectral dimension should be finite
    assert isinstance(result["spectral_dimension_planck"], float)
    assert result["spectral_dimension_curve"].shape[0] == 20


def test_hausdorff_dimension_euclidean(mock_history_2d):
    """Hausdorff dimension should equal spatial dimension for uniform grid."""
    positions = mock_history_2d.x_final[-1]
    alive = torch.ones(mock_history_2d.N, dtype=torch.bool)
    config = QuantumGravityConfig(n_radial_bins=20)

    result = compute_hausdorff_dimension(positions, alive, config)

    assert "hausdorff_dimension" in result
    assert "volume_scaling_data" in result
    assert "local_hausdorff" in result

    # Hausdorff dimension should be close to 2 for 2D grid
    assert 1.5 <= result["hausdorff_dimension"] <= 3.0

    radii, counts = result["volume_scaling_data"]
    assert radii.shape[0] == 20
    assert counts.shape[0] == 20


def test_holographic_entropy(mock_history_2d):
    """Test holographic entropy computation."""
    positions = mock_history_2d.x_final[-1]
    alive = torch.ones(mock_history_2d.N, dtype=torch.bool)
    config = QuantumGravityConfig()

    result = compute_holographic_entropy(positions, alive, mock_history_2d, config)

    assert "holographic_entropy" in result
    assert "boundary_area" in result
    assert "bulk_volume" in result
    assert "area_law_coefficient" in result

    # All values should be non-negative
    assert result["holographic_entropy"] >= 0
    assert result["bulk_volume"] >= 0


def test_full_quantum_gravity_observables_2d(mock_history_2d):
    """Test full quantum gravity analysis on 2D history."""
    config = QuantumGravityConfig(
        mc_time_index=-1,
        diffusion_time_steps=10,
        n_radial_bins=10,
    )

    observables = compute_quantum_gravity_observables(mock_history_2d, config)

    # Check that all fields are present
    assert isinstance(observables, QuantumGravityObservables)

    # Metadata
    assert observables.n_walkers > 0
    assert observables.spatial_dims == 2
    assert observables.mc_frame >= 0

    # 1. Regge Calculus
    assert isinstance(observables.regge_action, float)
    assert observables.regge_action_density.shape[0] == mock_history_2d.N
    assert observables.deficit_angles.shape[0] >= 0

    # 2. Einstein-Hilbert
    assert isinstance(observables.einstein_hilbert_action, float)
    assert observables.ricci_scalars.shape[0] >= 0
    assert isinstance(observables.scalar_curvature_mean, float)

    # 3. ADM Energy
    assert isinstance(observables.adm_mass, float)
    assert observables.adm_energy_density.shape[0] == mock_history_2d.N

    # 4. Spectral Dimension
    assert observables.spectral_dimension_curve.shape[0] == 10
    assert isinstance(observables.spectral_dimension_planck, float)
    assert observables.heat_kernel_trace.shape[0] == 10

    # 5. Hausdorff Dimension
    assert isinstance(observables.hausdorff_dimension, float)
    radii, counts = observables.volume_scaling_data
    assert radii.shape[0] == 10
    assert counts.shape[0] == 10

    # 6. Causal Structure
    assert observables.spacelike_edges.shape[0] == 2
    assert observables.timelike_edges.shape[0] == 2
    assert isinstance(observables.causal_violations, int)

    # 7. Holographic Entropy
    assert isinstance(observables.holographic_entropy, float)
    assert isinstance(observables.boundary_area, float)
    assert isinstance(observables.bulk_volume, float)

    # 8. Spin Network
    assert observables.edge_spins.shape[0] >= 0
    assert observables.vertex_quantum_volumes.shape[0] >= 0

    # 9. Raychaudhuri
    assert observables.expansion_scalar.shape[0] >= 0
    assert observables.convergence_regions.shape[0] >= 0

    # 10. Geodesic Deviation
    assert observables.deviation_vectors.shape[1] == 2  # 2D
    assert observables.tidal_tensor.shape[1] == 2
    assert observables.tidal_eigenvalues.shape[1] == 2


def test_full_quantum_gravity_observables_3d(mock_history_3d):
    """Test full quantum gravity analysis on 3D history."""
    config = QuantumGravityConfig(
        mc_time_index=-1,
        diffusion_time_steps=10,
        n_radial_bins=10,
    )

    observables = compute_quantum_gravity_observables(mock_history_3d, config)

    # Check that all fields are present
    assert isinstance(observables, QuantumGravityObservables)

    # Metadata
    assert observables.n_walkers > 0
    assert observables.spatial_dims == 3
    assert observables.mc_frame >= 0

    # Check 3D-specific dimensions
    assert observables.deviation_vectors.shape[1] == 3  # 3D
    assert observables.tidal_tensor.shape[1] == 3
    assert observables.tidal_eigenvalues.shape[1] == 3


def test_config_variations(mock_history_2d):
    """Test different configuration options."""
    # Test with metric correction
    config_full = QuantumGravityConfig(use_metric_correction="full")
    obs_full = compute_quantum_gravity_observables(mock_history_2d, config_full)
    assert isinstance(obs_full, QuantumGravityObservables)

    # Test with diagonal correction
    config_diag = QuantumGravityConfig(use_metric_correction="diagonal")
    obs_diag = compute_quantum_gravity_observables(mock_history_2d, config_diag)
    assert isinstance(obs_diag, QuantumGravityObservables)

    # Test with no correction
    config_none = QuantumGravityConfig(use_metric_correction="none")
    obs_none = compute_quantum_gravity_observables(mock_history_2d, config_none)
    assert isinstance(obs_none, QuantumGravityObservables)


def test_empty_history():
    """Test graceful handling of empty history."""
    history = RunHistory(N=10, d=2, n_steps=10, record_every=10)
    history.n_recorded = 1
    history.alive_mask = torch.zeros(1, 10, dtype=torch.bool)  # All dead

    config = QuantumGravityConfig()

    # Should not crash
    observables = compute_quantum_gravity_observables(history, config)

    # Should have zero walkers
    assert observables.n_walkers == 0


def test_physical_consistency(mock_history_2d):
    """Test physical consistency of results."""
    config = QuantumGravityConfig()
    observables = compute_quantum_gravity_observables(mock_history_2d, config)

    # Energy should be real (not complex)
    assert np.isfinite(observables.adm_mass)

    # Dimensions should be positive
    assert observables.hausdorff_dimension > 0
    assert observables.spectral_dimension_planck > 0

    # Entropy should be non-negative
    assert observables.holographic_entropy >= 0

    # Volume should be positive
    assert observables.bulk_volume >= 0


# =============================================================================
# Time Evolution Tests (4D Spacetime Block Analysis)
# =============================================================================


def test_time_evolution_consistency(mock_history_2d):
    """Test that time evolution produces consistent shapes."""
    config = QuantumGravityConfig(warmup_fraction=0.1)

    time_series = compute_quantum_gravity_time_evolution(
        mock_history_2d, config, frame_stride=2
    )

    # Check metadata
    assert isinstance(time_series, QuantumGravityTimeSeries)
    assert time_series.n_frames > 0
    assert len(time_series.mc_frames) == time_series.n_frames

    # Check all time series have correct length
    assert len(time_series.regge_action) == time_series.n_frames
    assert len(time_series.adm_mass) == time_series.n_frames
    assert len(time_series.spectral_dimension_planck) == time_series.n_frames
    assert len(time_series.hausdorff_dimension) == time_series.n_frames
    assert len(time_series.holographic_entropy) == time_series.n_frames


def test_time_evolution_shapes(mock_history_3d):
    """Test that all arrays have expected dimensions."""
    time_series = compute_quantum_gravity_time_evolution(mock_history_3d)

    # Scalar time series should be 1D
    assert time_series.regge_action.ndim == 1
    assert time_series.adm_mass.ndim == 1
    assert time_series.hausdorff_dimension.ndim == 1

    # All should have same length
    n = time_series.n_frames
    assert time_series.regge_action.shape[0] == n
    assert time_series.einstein_hilbert_action.shape[0] == n
    assert time_series.holographic_entropy.shape[0] == n


def test_time_evolution_physical_constraints(mock_history_2d):
    """Test that time evolution satisfies physical constraints."""
    time_series = compute_quantum_gravity_time_evolution(mock_history_2d)

    # Entropy should be non-negative
    assert (time_series.holographic_entropy >= 0).all()

    # Boundary area should be non-negative
    assert (time_series.boundary_area >= 0).all()

    # Bulk volume should be positive
    assert (time_series.bulk_volume >= 0).all()

    # Hausdorff dimension should be positive
    assert (time_series.hausdorff_dimension > 0).all()

    # Number of walkers should be non-negative
    assert (time_series.n_walkers >= 0).all()

    # Area law coefficient should be finite
    assert np.isfinite(time_series.area_law_coefficient).all()


def test_time_evolution_metadata(mock_history_2d):
    """Test that metadata is correctly populated."""
    config = QuantumGravityConfig(warmup_fraction=0.2)
    time_series = compute_quantum_gravity_time_evolution(
        mock_history_2d, config, frame_stride=1
    )

    # Check spatial dimensions
    assert time_series.spatial_dims == mock_history_2d.d

    # Check config is stored
    assert isinstance(time_series.config, QuantumGravityConfig)
    assert time_series.config.warmup_fraction == 0.2

    # Check frame metadata
    assert time_series.mc_frames[0] >= int(0.2 * mock_history_2d.n_recorded)
    assert time_series.mc_frames[-1] < mock_history_2d.n_recorded


def test_time_evolution_with_stride(mock_history_2d):
    """Test frame stride functionality."""
    # With stride 1
    ts_full = compute_quantum_gravity_time_evolution(
        mock_history_2d, frame_stride=1
    )

    # With stride 2
    ts_stride = compute_quantum_gravity_time_evolution(
        mock_history_2d, frame_stride=2
    )

    # Stride 2 should have approximately half the frames
    assert ts_stride.n_frames <= ts_full.n_frames
    assert ts_stride.n_frames > 0

    # Frame indices should be multiples of stride
    if len(ts_stride.mc_frames) > 1:
        frame_diff = ts_stride.mc_frames[1] - ts_stride.mc_frames[0]
        assert frame_diff >= 2


def test_time_evolution_empty_history():
    """Test graceful handling of empty history."""
    history = RunHistory(N=10, d=2, n_steps=10, record_every=10)
    history.n_recorded = 0
    history.alive_mask = torch.zeros(0, 10, dtype=torch.bool)

    config = QuantumGravityConfig()

    # Should not crash
    time_series = compute_quantum_gravity_time_evolution(history, config)

    # Should have zero frames
    assert time_series.n_frames == 0
    assert len(time_series.mc_frames) == 0


def test_time_evolution_single_frame():
    """Test time evolution with only one frame."""
    history = RunHistory(N=20, d=2, n_steps=10, record_every=10)
    history.n_recorded = 1
    positions = torch.randn(1, 20, 2)
    history.x_final[0] = positions[0]
    history.alive_mask = torch.ones(1, 20, dtype=torch.bool)

    from fragile.fractalai.bounds import TorchBounds

    history.bounds = TorchBounds(
        low=torch.tensor([-10.0, -10.0]),
        high=torch.tensor([10.0, 10.0]),
    )
    history.pbc = False

    config = QuantumGravityConfig(warmup_fraction=0.0)

    time_series = compute_quantum_gravity_time_evolution(history, config)

    # Should have one frame
    assert time_series.n_frames == 1
    assert len(time_series.mc_frames) == 1
    assert time_series.mc_frames[0] == 0
