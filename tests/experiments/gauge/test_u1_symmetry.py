"""Tests for U(1) fitness symmetry."""

import torch

from fragile.experiments.gauge.u1_symmetry import (
    compare_u1_phases,
    compute_dressed_walker_state,
    compute_u1_amplitude,
    compute_u1_observable,
    compute_u1_phase_current,
    compute_u1_phase_proposed,
    U1Config,
)


def test_u1_config_defaults():
    """Test U1Config default values."""
    config = U1Config()
    assert config.h_eff == 1.0
    assert config.epsilon_d == 0.1
    assert config.lambda_alg == 0.0


def test_compute_u1_phase_current(simple_swarm_2d):
    """Test current U(1) phase computation: θ_ik = -d_alg²/(2ε_d² ℏ_eff)."""
    phases = compute_u1_phase_current(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    # Phases should be negative or zero (formula has negative sign)
    assert (phases <= 0).all()

    # All alive walkers should have finite phases
    alive = simple_swarm_2d["alive"]
    assert torch.isfinite(phases[alive]).all()

    # Dead walkers should be zero
    if not alive.all():
        assert (phases[~alive] == 0).all()


def test_compute_u1_phase_current_with_custom_config(simple_swarm_2d):
    """Test current U(1) phases with custom configuration."""
    config = U1Config(h_eff=2.0, epsilon_d=0.2, lambda_alg=0.5)

    phases = compute_u1_phase_current(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        config=config,
    )

    assert phases.shape == (simple_swarm_2d["N"],)
    assert torch.isfinite(phases[simple_swarm_2d["alive"]]).all()


def test_compute_u1_phase_proposed_mean_field(simple_swarm_2d):
    """Test proposed U(1) phases in mean-field regime (rho=None)."""
    phases = compute_u1_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,  # Mean-field
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    # Phases should be non-negative (d'^β / ℏ_eff)
    alive = simple_swarm_2d["alive"]
    assert (phases[alive] >= 0).all()
    assert torch.isfinite(phases[alive]).all()


def test_compute_u1_phase_proposed_local(simple_swarm_2d):
    """Test proposed U(1) phases in local regime (finite rho)."""
    phases = compute_u1_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,  # Local regime
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    alive = simple_swarm_2d["alive"]
    assert (phases[alive] >= 0).all()
    assert torch.isfinite(phases[alive]).all()


def test_compute_u1_phase_proposed_with_dead_walkers(partially_dead_swarm_2d):
    """Test proposed U(1) phases handle dead walkers."""
    phases = compute_u1_phase_proposed(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        companions=partially_dead_swarm_2d["diversity_companions"],
        alive=partially_dead_swarm_2d["alive"],
        rho=0.1,
    )

    alive = partially_dead_swarm_2d["alive"]

    # Dead walkers should have zero phase
    assert (phases[~alive] == 0).all()

    # Alive walkers should have non-zero phase
    assert (phases[alive] > 0).all()


def test_compute_u1_amplitude(simple_swarm_2d):
    """Test U(1) amplitude computation (softmax probabilities)."""
    amplitudes = compute_u1_amplitude(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert amplitudes.shape == (N, N)

    # Each row should sum to ~1 (probability distribution)
    row_sums = amplitudes.sum(dim=1)
    alive = simple_swarm_2d["alive"]
    assert torch.allclose(row_sums[alive], torch.ones(alive.sum()), atol=1e-5)

    # No self-pairing (diagonal should be zero)
    assert (torch.diag(amplitudes) == 0).all()

    # All probabilities non-negative
    assert (amplitudes >= 0).all()


def test_compare_u1_phases_mean_field(simple_swarm_2d):
    """Test comparison between current and proposed U(1) phases (mean-field)."""
    comparison = compare_u1_phases(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,
    )

    # Check all expected keys present
    assert "current" in comparison
    assert "proposed" in comparison
    assert "difference" in comparison
    assert "correlation" in comparison
    assert "current_mean" in comparison
    assert "current_std" in comparison
    assert "proposed_mean" in comparison
    assert "proposed_std" in comparison

    N = simple_swarm_2d["N"]
    assert comparison["current"].shape == (N,)
    assert comparison["proposed"].shape == (N,)
    assert comparison["difference"].shape == (N,)

    # Correlation should be between -1 and 1
    assert -1.0 <= comparison["correlation"] <= 1.0

    # Statistics should be finite
    assert isinstance(comparison["current_mean"], float)
    assert isinstance(comparison["proposed_std"], float)


def test_compare_u1_phases_local(simple_swarm_2d):
    """Test comparison between current and proposed U(1) phases (local)."""
    comparison = compare_u1_phases(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    # Check correlation is computed
    assert isinstance(comparison["correlation"], float)
    assert -1.0 <= comparison["correlation"] <= 1.0


def test_compute_dressed_walker_state(simple_swarm_2d):
    """Test dressed walker state computation |ψ_i⟩."""
    # Compute phases and amplitudes
    phases = compute_u1_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    amplitudes = compute_u1_amplitude(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    # Compute dressed state for walker 0
    walker_idx = 0
    state = compute_dressed_walker_state(
        phases=phases,
        amplitudes=amplitudes,
        walker_idx=walker_idx,
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert state.shape == (N,)
    assert state.dtype in {torch.complex64, torch.complex128}

    # State should have finite values
    assert torch.isfinite(state).all()


def test_compute_u1_observable(simple_swarm_2d):
    """Test gauge-invariant observable computation."""
    # Create a dressed state
    phases = compute_u1_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        companions=simple_swarm_2d["diversity_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    amplitudes = compute_u1_amplitude(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    state = compute_dressed_walker_state(
        phases=phases,
        amplitudes=amplitudes,
        walker_idx=0,
        alive=simple_swarm_2d["alive"],
    )

    # Compute observable
    observable = compute_u1_observable(state)

    # Should be a scalar close to 1 (normalized state)
    assert isinstance(observable, float)
    assert torch.isfinite(torch.tensor(observable))
    assert observable >= 0  # Norm squared is non-negative
    assert abs(observable - 1.0) < 0.1  # Should be ~1 for normalized state
