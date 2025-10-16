"""Test cloning operator for Ricci Gas."""

import pytest
import torch

from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def params():
    """Standard Ricci Gas parameters."""
    return RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_Ric=0.01,
        force_mode="pull",
        reward_mode="inverse",
    )


def test_cloning_changes_positions(device, params):
    """Cloning should change walker positions."""
    gas = RicciGas(params, device=device)

    N, d = 50, 2
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Apply cloning
    state_cloned = gas.apply_cloning(state)

    # Positions should have changed (walkers clone to nearby companions)
    assert not torch.allclose(state.x, state_cloned.x)


def test_cloning_selects_nearby_walkers(device, params):
    """Walkers should preferentially clone from nearby companions."""
    gas = RicciGas(params, device=device)

    N, d = 100, 2

    # Create state with one isolated walker and a tight cluster
    x = torch.zeros(N, d, device=device)
    x[0] = torch.tensor([10.0, 10.0])  # Isolated walker
    x[1:] = torch.randn(N - 1, d, device=device) * 0.3  # Tight cluster around origin

    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Apply cloning multiple times
    n_trials = 100

    # Count how many walkers from the cluster choose other cluster members vs the isolated walker
    cluster_to_cluster = 0
    cluster_to_isolated = 0

    for _ in range(n_trials):
        state_cloned = gas.apply_cloning(state)

        # Check where cluster walkers (indices 1:) cloned to
        for i in range(1, N):
            # Did they clone near other cluster members or near the isolated walker?
            dist_to_isolated = (state_cloned.x[i] - x[0]).norm()
            dist_to_cluster = torch.min((state_cloned.x[i].unsqueeze(0) - x[1:]).norm(dim=-1))

            if dist_to_cluster < dist_to_isolated:
                cluster_to_cluster += 1
            else:
                cluster_to_isolated += 1

    # Cluster walkers should mostly clone from nearby cluster members, not the distant isolated walker
    assert cluster_to_cluster > cluster_to_isolated * 2  # At least 2:1 ratio


def test_cloning_respects_alive_status(device, params):
    """Dead walkers should never be chosen as companions."""
    gas = RicciGas(params, device=device)

    N, d = 20, 2
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    # Kill half the walkers
    s[::2] = 0.0

    state = SwarmState(x=x, v=v, s=s)

    # Apply cloning many times
    for _ in range(50):
        state_cloned = gas.apply_cloning(state)

        # No cloned position should be close to a dead walker's original position
        dead_indices = torch.where(~s.bool())[0]

        for dead_idx in dead_indices:
            dist_to_dead = (state_cloned.x - state.x[dead_idx]).norm(dim=-1)
            # Some walkers might be accidentally close, but not significantly more than random
            # We just check that dead walkers aren't systematically chosen
            assert (
                dist_to_dead < params.sigma_clone * 2
            ).sum() <= 4  # Very few should be this close


def test_step_includes_cloning(device, params):
    """Step method should apply cloning by default."""
    gas = RicciGas(params, device=device)

    N, d = 30, 2
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Take a step with cloning enabled
    state_with_clone = gas.step(state, dt=0.1, do_clone=True)

    # Take a step without cloning
    state_no_clone = gas.step(state, dt=0.1, do_clone=False)

    # Results should be different
    assert not torch.allclose(state_with_clone.x, state_no_clone.x)


def test_step_updates_all_components(device, params):
    """Step should update positions, velocities, and curvature."""
    gas = RicciGas(params, device=device)

    N, d = 20, 2
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Take a step
    state_new = gas.step(state, dt=0.1)

    # All components should be updated
    assert not torch.allclose(state.x, state_new.x)
    assert not torch.allclose(state.v, state_new.v)
    assert state_new.R is not None
    assert state_new.H is not None


def test_cloning_maintains_swarm_size(device, params):
    """Cloning should not change the number of walkers."""
    gas = RicciGas(params, device=device)

    N, d = 25, 2
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    state_cloned = gas.apply_cloning(state)

    assert state_cloned.x.shape[0] == N
    assert state_cloned.v.shape[0] == N
    assert state_cloned.s.shape[0] == N


def test_multiple_steps_with_cloning(device, params):
    """Run multiple steps and verify dynamics are stable."""
    gas = RicciGas(params, device=device)

    N, d = 30, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Run 50 steps
    for _ in range(50):
        state = gas.step(state, dt=0.1, do_clone=True)

        # Check no NaNs or Infs
        assert not torch.isnan(state.x).any()
        assert not torch.isinf(state.x).any()
        assert not torch.isnan(state.v).any()
        assert not torch.isinf(state.v).any()

        # Check curvature is computed
        assert state.R is not None
        assert not torch.isnan(state.R).any()


def test_cloning_with_large_epsilon(device):
    """Test that large epsilon_clone leads to more uniform selection."""
    # Use large epsilon for nearly uniform selection
    params = RicciGasParams(
        epsilon_R=0.1,
        kde_bandwidth=0.5,
        epsilon_Ric=0.01,
        epsilon_clone=100.0,  # Very large → nearly uniform
        sigma_clone=0.1,
    )
    gas = RicciGas(params, device=device)

    N, d = 50, 2

    # Create two distant clusters
    x = torch.zeros(N, d, device=device)
    x[:25] = torch.randn(25, d, device=device) * 0.2  # Cluster 1 at origin
    x[25:] = torch.randn(25, d, device=device) * 0.2 + torch.tensor([
        10.0,
        10.0,
    ])  # Cluster 2 far away

    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)
    state = SwarmState(x=x, v=v, s=s)

    # Count cross-cluster cloning
    cross_cluster_clones = 0
    n_trials = 100

    for _ in range(n_trials):
        state_cloned = gas.apply_cloning(state)

        # Check if cluster 1 walkers cloned to cluster 2 (or vice versa)
        for i in range(25):
            dist_to_cluster2 = (state_cloned.x[i] - x[25:].mean(dim=0)).norm()
            if dist_to_cluster2 < 5.0:  # Close to cluster 2
                cross_cluster_clones += 1

    # With large epsilon, should see significant cross-cluster cloning
    # With small epsilon, would see almost none
    assert cross_cluster_clones > n_trials * 5  # At least 5% of trials per walker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
