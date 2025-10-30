"""Tests for gauge symmetry observables module."""

import torch

from fragile.experiments.gauge.observables import (
    bin_by_distance,
    compute_cloning_score,
    compute_collective_fields,
    compute_field_gradients,
    fit_exponential_decay,
    ObservablesConfig,
)


def test_observables_config_defaults():
    """Test ObservablesConfig default values."""
    config = ObservablesConfig()
    assert config.h_eff == 1.0
    assert config.epsilon_clone == 1e-8
    assert config.beta == 1.0
    assert config.alpha == 1.0
    assert config.eta == 0.1
    assert config.A == 2.0
    assert config.lambda_alg == 0.0


def test_compute_collective_fields_mean_field(simple_swarm_2d):
    """Test collective fields computation in mean-field regime (rho=None)."""
    result = compute_collective_fields(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        alive=simple_swarm_2d["alive"],
        companions=simple_swarm_2d["diversity_companions"],
        rho=None,  # Mean-field
    )

    N = simple_swarm_2d["N"]

    # Check all fields present
    assert "d_prime" in result
    assert "r_prime" in result
    assert "fitness" in result

    # Check shapes
    assert result["d_prime"].shape == (N,)
    assert result["r_prime"].shape == (N,)
    assert result["fitness"].shape == (N,)

    # Check all alive walkers have non-zero values
    alive = simple_swarm_2d["alive"]
    assert (result["d_prime"][alive] > 0).all()
    assert (result["r_prime"][alive] > 0).all()

    # Dead walkers should be zero
    if not alive.all():
        assert (result["d_prime"][~alive] == 0).all()


def test_compute_collective_fields_local(simple_swarm_2d):
    """Test collective fields computation in local regime (finite rho)."""
    result = compute_collective_fields(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        alive=simple_swarm_2d["alive"],
        companions=simple_swarm_2d["diversity_companions"],
        rho=0.1,  # Local regime
    )

    N = simple_swarm_2d["N"]

    # Check all fields present
    assert "d_prime" in result
    assert "r_prime" in result
    assert "fitness" in result

    # Check shapes
    assert result["d_prime"].shape == (N,)
    assert result["r_prime"].shape == (N,)
    assert result["fitness"].shape == (N,)

    # All values should be finite
    assert torch.isfinite(result["d_prime"]).all()
    assert torch.isfinite(result["r_prime"]).all()
    assert torch.isfinite(result["fitness"]).all()


def test_compute_collective_fields_with_dead_walkers(partially_dead_swarm_2d):
    """Test collective fields handle dead walkers correctly."""
    config = ObservablesConfig()
    result = compute_collective_fields(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        alive=partially_dead_swarm_2d["alive"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        rho=0.1,
        config=config,
    )

    alive = partially_dead_swarm_2d["alive"]

    # Dead walkers should all have the same baseline value
    if (~alive).any():
        dead_d_prime = result["d_prime"][~alive]
        dead_r_prime = result["r_prime"][~alive]
        # All dead walkers should have identical values (constant baseline)
        assert torch.allclose(dead_d_prime, dead_d_prime[0] * torch.ones_like(dead_d_prime))
        assert torch.allclose(dead_r_prime, dead_r_prime[0] * torch.ones_like(dead_r_prime))
        # Fitness should be zero for dead walkers
        assert (result["fitness"][~alive] == 0).all()

    # Alive walkers should have varying fields
    if alive.sum() > 1:
        assert result["d_prime"][alive].std() > 0  # Should have variation


def test_compute_cloning_score(simple_swarm_2d):
    """Test cloning score computation S_i(j) = (V_j - V_i)/(V_i + eps)."""
    # Create simple fitness values
    fitness = torch.arange(50, dtype=torch.float32)

    scores = compute_cloning_score(
        fitness=fitness,
        alive=simple_swarm_2d["alive"],
        clone_companions=simple_swarm_2d["clone_companions"],
        epsilon_clone=1e-8,
    )

    N = simple_swarm_2d["N"]
    assert scores.shape == (N,)

    # Scores should be finite
    assert torch.isfinite(scores).all()

    # Check scoring logic for specific examples
    # If V_j > V_i, score should be positive
    # If V_j < V_i, score should be negative
    for i in range(min(5, N)):
        j = simple_swarm_2d["clone_companions"][i].item()
        expected_score = (fitness[j] - fitness[i]) / (fitness[i] + 1e-8)
        assert torch.isclose(scores[i], expected_score, rtol=1e-5)


def test_compute_cloning_score_with_dead_walkers(partially_dead_swarm_2d):
    """Test cloning score handles dead walkers."""
    fitness = torch.rand(partially_dead_swarm_2d["N"])

    scores = compute_cloning_score(
        fitness=fitness,
        alive=partially_dead_swarm_2d["alive"],
        clone_companions=partially_dead_swarm_2d["clone_companions"],
    )

    # Dead walkers should have zero score
    alive = partially_dead_swarm_2d["alive"]
    assert (scores[~alive] == 0).all()


def test_bin_by_distance(simple_swarm_2d):
    """Test distance binning for correlation functions."""
    import numpy as np

    # Create a simple field
    field_values = torch.randn(simple_swarm_2d["N"])

    r, C, counts = bin_by_distance(
        positions=simple_swarm_2d["positions"],
        values=field_values,
        alive=simple_swarm_2d["alive"],
        r_max=0.5,
        n_bins=20,
    )

    # Check outputs (returns numpy arrays)
    assert len(r) == 20
    assert len(C) == 20
    assert len(counts) == 20

    # r should be monotonically increasing
    assert np.all(r[1:] > r[:-1])

    # Counts should be non-negative integers
    assert np.all(counts >= 0)

    # C should be finite where counts > 0
    assert np.all(np.isfinite(C[counts > 0]))


def test_fit_exponential_decay():
    """Test exponential decay fitting for correlation length."""
    import numpy as np

    # Create synthetic data: C(r) = 2.0 * exp(-r²/0.1²)
    r = np.linspace(0, 0.5, 50)
    xi_true = 0.1
    C_true = 2.0 * np.exp(-(r**2) / xi_true**2)

    # Add small noise
    np.random.seed(42)
    C = C_true + np.random.randn(len(C_true)) * 0.05

    # Fit (returns dict)
    result = fit_exponential_decay(r, C)

    C_0_fit = result["C0"]
    xi_fit = result["xi"]
    r_squared = result["r_squared"]

    # Check recovery (should be close to true values)
    assert abs(xi_fit - xi_true) < 0.02  # Within 0.02
    assert abs(C_0_fit - 2.0) < 0.2  # Within 0.2
    assert r_squared > 0.9  # Good fit


def test_compute_field_gradients(simple_swarm_2d):
    """Test field gradient computation."""
    # Create a smooth field (e.g., linear gradient)
    positions = simple_swarm_2d["positions"]
    field_values = positions[:, 0] + 2 * positions[:, 1]  # f(x,y) = x + 2y

    gradients = compute_field_gradients(
        positions=positions,
        field_values=field_values,
        alive=simple_swarm_2d["alive"],
        k_neighbors=5,
    )

    # Check output (returns Tensor directly)
    N = simple_swarm_2d["N"]
    assert gradients.shape == (N,)

    # Gradients should be non-negative
    assert (gradients >= 0).all()

    # All alive walkers should have finite gradients
    alive = simple_swarm_2d["alive"]
    assert torch.isfinite(gradients[alive]).all()


def test_compute_field_gradients_with_dead_walkers(partially_dead_swarm_2d):
    """Test gradient computation handles dead walkers."""
    positions = partially_dead_swarm_2d["positions"]
    field_values = torch.randn(partially_dead_swarm_2d["N"])

    gradients = compute_field_gradients(
        positions=positions,
        field_values=field_values,
        alive=partially_dead_swarm_2d["alive"],
        k_neighbors=5,
    )

    # Dead walkers should have zero gradient
    alive = partially_dead_swarm_2d["alive"]
    assert (gradients[~alive] == 0).all()
