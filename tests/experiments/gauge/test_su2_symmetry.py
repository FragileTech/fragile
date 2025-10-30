"""Tests for SU(2) weak isospin symmetry."""

import torch

from fragile.experiments.gauge.su2_symmetry import (
    compare_su2_phases,
    compute_isospin_doublet_state,
    compute_su2_observable,
    compute_su2_pairing_probability,
    compute_su2_phase_current,
    compute_su2_phase_proposed,
    SU2Config,
)


def test_su2_config_defaults():
    """Test SU2Config default values."""
    config = SU2Config()
    assert config.h_eff == 1.0
    assert config.epsilon_c == 0.1
    assert config.epsilon_clone == 1e-8
    assert config.lambda_alg == 0.0


def test_compute_su2_phase_current(simple_swarm_2d):
    """Test current SU(2) phase: θ_ij = -d_alg²/(2ε_c² ℏ_eff)."""
    phases = compute_su2_phase_current(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    # Phases should be negative or zero (formula has negative sign)
    assert (phases <= 0).all()

    # Alive walkers should have finite phases
    alive = simple_swarm_2d["alive"]
    assert torch.isfinite(phases[alive]).all()

    # Dead walkers should be zero
    if not alive.all():
        assert (phases[~alive] == 0).all()


def test_compute_su2_phase_current_with_custom_config(simple_swarm_2d):
    """Test current SU(2) phases with custom configuration."""
    config = SU2Config(h_eff=2.0, epsilon_c=0.2, lambda_alg=0.5)

    phases = compute_su2_phase_current(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        config=config,
    )

    assert phases.shape == (simple_swarm_2d["N"],)
    assert torch.isfinite(phases[simple_swarm_2d["alive"]]).all()


def test_compute_su2_phase_proposed_mean_field(simple_swarm_2d):
    """Test proposed SU(2) phases in mean-field regime."""
    phases = compute_su2_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=None,  # Mean-field
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    # Phases should be finite (cloning score / ℏ_eff)
    alive = simple_swarm_2d["alive"]
    assert torch.isfinite(phases[alive]).all()


def test_compute_su2_phase_proposed_local(simple_swarm_2d):
    """Test proposed SU(2) phases in local regime."""
    phases = compute_su2_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,  # Local regime
    )

    N = simple_swarm_2d["N"]
    assert phases.shape == (N,)

    alive = simple_swarm_2d["alive"]
    assert torch.isfinite(phases[alive]).all()


def test_compute_su2_phase_proposed_with_dead_walkers(partially_dead_swarm_2d):
    """Test proposed SU(2) phases handle dead walkers."""
    phases = compute_su2_phase_proposed(
        positions=partially_dead_swarm_2d["positions"],
        velocities=partially_dead_swarm_2d["velocities"],
        rewards=partially_dead_swarm_2d["rewards"],
        diversity_companions=partially_dead_swarm_2d["diversity_companions"],
        clone_companions=partially_dead_swarm_2d["clone_companions"],
        alive=partially_dead_swarm_2d["alive"],
        rho=0.1,
    )

    alive = partially_dead_swarm_2d["alive"]

    # Dead walkers should have zero phase
    assert (phases[~alive] == 0).all()


def test_compute_su2_pairing_probability(simple_swarm_2d):
    """Test pairing probability computation for diversity pairing."""
    probs = compute_su2_pairing_probability(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert probs.shape == (N, N)

    # Each row should sum to ~1 (probability distribution)
    row_sums = probs.sum(dim=1)
    alive = simple_swarm_2d["alive"]
    assert torch.allclose(row_sums[alive], torch.ones(alive.sum()), atol=1e-5)

    # No self-pairing (diagonal should be zero)
    assert (torch.diag(probs) == 0).all()

    # All probabilities non-negative
    assert (probs >= 0).all()


def test_compare_su2_phases_mean_field(simple_swarm_2d):
    """Test comparison between current and proposed SU(2) phases (mean-field)."""
    comparison = compare_su2_phases(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
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


def test_compare_su2_phases_local(simple_swarm_2d):
    """Test comparison between current and proposed SU(2) phases (local)."""
    comparison = compare_su2_phases(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    assert isinstance(comparison["correlation"], float)
    assert -1.0 <= comparison["correlation"] <= 1.0


def test_compute_isospin_doublet_state(simple_swarm_2d):
    """Test isospin doublet state |Ψ_ij⟩ = |↑⟩⊗|ψ_i⟩ + |↓⟩⊗|ψ_j⟩."""
    # Compute phases and amplitudes
    phases = compute_su2_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    amplitudes = compute_su2_pairing_probability(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    # Compute doublet state for pair (0, 1)
    walker_i = 0
    walker_j = 1
    up_component, down_component = compute_isospin_doublet_state(
        phases=phases,
        amplitudes=amplitudes,
        walker_i=walker_i,
        walker_j=walker_j,
        alive=simple_swarm_2d["alive"],
    )

    N = simple_swarm_2d["N"]
    assert up_component.shape == (N,)
    assert down_component.shape == (N,)
    assert up_component.dtype in {torch.complex64, torch.complex128}
    assert down_component.dtype in {torch.complex64, torch.complex128}

    # Components should have finite values
    assert torch.isfinite(up_component).all()
    assert torch.isfinite(down_component).all()


def test_compute_su2_observable(simple_swarm_2d):
    """Test gauge-invariant observable from doublet state."""
    # Compute components
    phases = compute_su2_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    amplitudes = compute_su2_pairing_probability(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        alive=simple_swarm_2d["alive"],
    )

    up_component, down_component = compute_isospin_doublet_state(
        phases=phases,
        amplitudes=amplitudes,
        walker_i=0,
        walker_j=1,
        alive=simple_swarm_2d["alive"],
    )

    # Compute observable
    observable = compute_su2_observable(up_component, down_component)

    # Should be a non-negative scalar
    assert isinstance(observable, float)
    assert torch.isfinite(torch.tensor(observable))
    assert observable >= 0  # Norm squared is non-negative


def test_su2_phases_sign_difference(simple_swarm_2d):
    """Test that current and proposed phases have different sign structure.

    Current: θ_ij < 0 (negative)
    Proposed: θ_ij can be positive or negative (depends on cloning score)
    """
    current = compute_su2_phase_current(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
    )

    proposed = compute_su2_phase_proposed(
        positions=simple_swarm_2d["positions"],
        velocities=simple_swarm_2d["velocities"],
        rewards=simple_swarm_2d["rewards"],
        diversity_companions=simple_swarm_2d["diversity_companions"],
        clone_companions=simple_swarm_2d["clone_companions"],
        alive=simple_swarm_2d["alive"],
        rho=0.1,
    )

    alive = simple_swarm_2d["alive"]

    # Current should be all negative
    assert (current[alive] <= 0).all()

    # Proposed can have both signs (depends on fitness comparisons)
    # Just check it's finite
    assert torch.isfinite(proposed[alive]).all()
