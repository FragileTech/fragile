"""Test that cloning actually revives dead walkers."""

import torch

from fragile.ricci_gas import RicciGas, RicciGasParams, SwarmState


def test_cloning_revives_all_walkers():
    """Cloning should set all walkers to alive (s=1)."""
    device = torch.device("cpu")
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_clone=0.5,
        sigma_clone=0.2,
    )
    gas = RicciGas(params, device=device)

    N, d = 50, 2
    x = torch.randn(N, d, device=device) * 2.0
    v = torch.randn(N, d, device=device) * 0.1
    s = torch.ones(N, device=device)

    # Kill half the walkers
    s[::2] = 0.0

    state = SwarmState(x=x, v=v, s=s)

    initial_alive = state.s.sum().item()
    assert initial_alive == N // 2  # Half alive

    # Apply cloning
    state_cloned = gas.apply_cloning(state)

    # After cloning, ALL walkers should be alive
    assert state_cloned.s.sum().item() == N
    assert (state_cloned.s == 1.0).all()


def test_step_maintains_population_with_boundaries():
    """With boundaries and cloning, population should stabilize."""
    device = torch.device("cpu")
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_clone=0.5,
        sigma_clone=0.2,
        x_min=-4.0,
        x_max=4.0,
    )
    gas = RicciGas(params, device=device)

    N, d = 100, 2
    torch.manual_seed(42)
    x = torch.randn(N, d, device=device) * 2.0
    v = torch.randn(N, d, device=device) * 0.3
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    alive_counts = []

    # Run for 50 steps
    for _ in range(50):
        state = gas.step(state, dt=0.1, gamma=0.9, noise_std=0.1, do_clone=True)
        alive_counts.append(state.s.sum().item())

    # Should maintain close to full population
    # Cloning revives walkers each step before boundary enforcement
    mean_alive = sum(alive_counts) / len(alive_counts)
    min_alive = min(alive_counts)

    # With revival, should maintain high population
    assert mean_alive > N * 0.9  # At least 90% on average
    assert min_alive > N * 0.8  # Never drop below 80%


def test_cloning_before_boundaries_maintains_population():
    """Cloning happens BEFORE boundaries, so population should stay high."""
    device = torch.device("cpu")
    params = RicciGasParams(
        epsilon_R=0.5,
        kde_bandwidth=0.4,
        epsilon_clone=0.5,
        sigma_clone=0.2,
        x_min=-4.0,
        x_max=4.0,
    )
    gas = RicciGas(params, device=device)

    N, d = 50, 2
    x = torch.randn(N, d, device=device) * 2.0  # Well within boundaries
    v = torch.randn(N, d, device=device) * 0.3  # Moderate velocities
    s = torch.ones(N, device=device)

    state = SwarmState(x=x, v=v, s=s)

    alive_counts = []

    for step in range(30):
        # Take step
        state = gas.step(state, dt=0.1, do_clone=True)
        alive_counts.append(state.s.sum().item())

    # Over time, should maintain high population due to revival
    mean_alive = sum(alive_counts) / len(alive_counts)
    min_alive = min(alive_counts)

    # With cloning revival, should maintain majority of population
    assert mean_alive > N * 0.85  # At least 85% on average
    assert min_alive > N * 0.75  # Never drop below 75%


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
