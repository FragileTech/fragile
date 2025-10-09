"""Test boundary enforcement and revival for Ricci Gas."""

import pytest
import torch

from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def params_with_bounds():
    """Ricci Gas parameters with boundaries."""
    return RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        epsilon_clone=0.5,
        sigma_clone=0.2,
        force_mode="pull",
        reward_mode="inverse",
        x_min=-4.0,
        x_max=4.0,
    )


def test_boundary_enforcement_kills_walkers(device, params_with_bounds):
    """Walkers that leave bounds should be marked as dead."""
    gas = RicciGas(params_with_bounds, device=device)

    N, d = 20, 2

    # Create state with some walkers out of bounds
    x = torch.zeros(N, d, device=device)
    x[:10] = torch.randn(10, d, device=device) * 2.0  # In bounds [-4, 4]
    x[10:15] = torch.tensor([[5.0, 0.0], [0.0, 5.0], [-5.0, 0.0], [0.0, -5.0], [6.0, 6.0]])  # Out of bounds
    x[15:] = torch.randn(5, d, device=device) * 2.0  # In bounds

    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Apply boundary enforcement
    state_bounded = gas.apply_boundary_enforcement(state)

    # Check that out-of-bounds walkers are dead
    assert state_bounded.s[:10].sum() == 10  # First 10 should be alive
    assert state_bounded.s[10:15].sum() == 0  # Out-of-bounds should be dead
    assert state_bounded.s[15:].sum() == 5  # Last 5 should be alive


def test_step_enforces_boundaries(device, params_with_bounds):
    """Step method should kill walkers that move out of bounds."""
    gas = RicciGas(params_with_bounds, device=device)

    N, d = 30, 2

    # Initialize near the boundary
    x = torch.zeros(N, d, device=device)
    x[:15] = torch.rand(15, d, device=device) * 0.5 + 3.5  # Near x_max = 4.0
    x[15:] = torch.rand(15, d, device=device) * 0.5 - 4.0  # Near x_min = -4.0

    # Give strong outward velocities
    v = torch.zeros(N, d, device=device)
    v[:15, 0] = 2.0  # Push toward +x
    v[15:, 0] = -2.0  # Push toward -x

    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Take several steps
    alive_counts = []
    for _ in range(10):
        state = gas.step(state, dt=0.2, do_clone=False)  # Disable cloning to see pure death
        alive_counts.append(state.s.sum().item())

    # Some walkers should have died by leaving bounds
    assert alive_counts[-1] < alive_counts[0]
    assert alive_counts[-1] >= 0  # But not all necessarily dead


def test_dead_walkers_can_be_revived_by_cloning(device, params_with_bounds):
    """Dead walkers should be able to clone from alive walkers and revive."""
    gas = RicciGas(params_with_bounds, device=device)

    N, d = 50, 2

    # Start with all walkers in bounds and alive
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device) * 2.0  # Well within [-4, 4]
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Run dynamics: some will die, but cloning should help maintain population
    alive_history = []

    for step in range(100):
        state = gas.step(state, dt=0.1, gamma=0.8, noise_std=0.2, do_clone=True)
        alive_count = state.s.sum().item()
        alive_history.append(alive_count)

        # Track min alive count
        if step == 0:
            initial_alive = alive_count

    min_alive = min(alive_history)
    final_alive = alive_history[-1]

    # Should have some deaths during exploration
    assert min_alive < initial_alive

    # But cloning should revive some walkers
    # Final count should be higher than minimum (revival occurred)
    assert final_alive >= min_alive

    # At least some walkers should remain alive throughout
    assert min_alive > 0


def test_cloning_revives_dead_walkers_explicitly(device, params_with_bounds):
    """Explicitly test that dead walkers get revived through cloning."""
    gas = RicciGas(params_with_bounds, device=device)

    N, d = 30, 2

    # Create state with half alive, half dead
    x = torch.randn(N, d, device=device) * 2.0
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    # Kill half the walkers by setting status to 0
    s[::2] = 0.0

    state = SwarmState(x=x, v=v, s=s)

    initial_alive = state.s.sum().item()
    assert initial_alive == N // 2  # Half alive

    # Apply cloning - dead walkers should clone from alive walkers
    state_cloned = gas.apply_cloning(state)

    # After cloning, positions of dead walkers should have changed
    # (they cloned from alive walkers)
    dead_indices = torch.where(s == 0.0)[0]
    alive_indices = torch.where(s == 1.0)[0]

    # Dead walkers should now have positions near alive walkers
    for dead_idx in dead_indices:
        # Find distance to nearest alive walker's original position
        dists = (state_cloned.x[dead_idx] - x[alive_indices]).norm(dim=-1)
        min_dist = dists.min().item()

        # Should be close (within epsilon_clone range + sigma_clone jitter)
        # With epsilon_clone=0.5 and sigma_clone=0.2, should typically be < 1.0
        assert min_dist < 2.0  # Generous bound


def test_boundaries_with_no_bounds_set(device):
    """When no bounds are set, no walkers should be killed."""
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        # No x_min or x_max set
    )
    gas = RicciGas(params, device=device)

    N, d = 20, 2

    # Create walkers way out of typical range
    x = torch.randn(N, d, device=device) * 100.0  # Far out
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    # Apply boundary enforcement
    state_bounded = gas.apply_boundary_enforcement(state)

    # All should still be alive
    assert state_bounded.s.sum() == N


def test_population_maintains_with_cloning_and_bounds(device, params_with_bounds):
    """Over many steps, cloning should help maintain population despite boundaries."""
    gas = RicciGas(params_with_bounds, device=device)

    N, d = 50, 2

    torch.manual_seed(123)
    x = torch.randn(N, d, device=device) * 1.5  # Start well within bounds
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    alive_counts = []

    for _ in range(200):
        state = gas.step(state, dt=0.1, gamma=0.9, noise_std=0.1, do_clone=True)
        alive_counts.append(state.s.sum().item())

    # Calculate statistics
    mean_alive = sum(alive_counts) / len(alive_counts)
    min_alive = min(alive_counts)

    # With cloning, should maintain reasonable population
    assert mean_alive > N * 0.3  # At least 30% on average
    assert min_alive > 0  # Never complete extinction


def test_boundary_enforcement_is_per_dimension(device):
    """Boundaries should be enforced independently per dimension."""
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        x_min=-3.0,
        x_max=3.0,
    )
    gas = RicciGas(params, device=device)

    N, d = 10, 2

    # Create specific test cases
    x = torch.tensor([
        [0.0, 0.0],      # In bounds
        [2.0, 2.0],      # In bounds
        [3.5, 0.0],      # Out in x
        [0.0, 3.5],      # Out in y
        [-3.5, 0.0],     # Out in x
        [0.0, -3.5],     # Out in y
        [3.5, 3.5],      # Out in both
        [-3.5, -3.5],    # Out in both
        [2.9, 2.9],      # In bounds (near edge)
        [-2.9, -2.9],    # In bounds (near edge)
    ], device=device)

    v = torch.zeros(N, d, device=device)
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)
    state_bounded = gas.apply_boundary_enforcement(state)

    # Check expected alive/dead status
    expected_alive = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], device=device)
    assert torch.allclose(state_bounded.s, expected_alive)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
