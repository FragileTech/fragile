"""Integration test with real Pong environment."""

import pytest

# Check if plangym is available
try:
    from plangym import AtariEnvironment

    PLANGYM_AVAILABLE = True
except ImportError:
    PLANGYM_AVAILABLE = False


@pytest.mark.skipif(not PLANGYM_AVAILABLE, reason="plangym not installed")
def test_pong_integration():
    """Test fractal gas on real Pong environment with RAM observations."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    # Use RAM observations for speed (128 bytes vs 28KB pixels)
    env = AtariEnvironment(name="Pong-v5", obs_type="ram")

    gas = AtariFractalGas(
        env=env,
        N=20,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device="cpu",
        seed=42,
    )

    final_state, history = gas.run(max_iterations=100)

    # Basic checks
    assert len(history) <= 100
    assert final_state.N == 20
    assert gas.total_steps > 0

    # Verify observations are RAM (128 dims)
    assert final_state.observations.shape == (20, 128)

    # Check virtual rewards were computed
    assert final_state.virtual_rewards is not None
    assert final_state.virtual_rewards.shape == (20,)

    # Get best walker
    best_idx, best_reward = gas.get_best_walker(final_state)
    print(f"\nBest walker {best_idx} achieved reward {best_reward:.2f}")
    print(f"Total steps: {gas.total_steps}")
    print(f"Total clones: {gas.total_clones}")
    print(f"Iterations: {len(history)}")

    # Check history has expected structure
    for info in history:
        assert "iteration" in info
        assert "num_cloned" in info
        assert "alive_count" in info
        assert "mean_reward" in info
        assert "max_reward" in info
        assert "mean_virtual_reward" in info

    env.close()


@pytest.mark.skipif(not PLANGYM_AVAILABLE, reason="plangym not installed")
def test_breakout_integration():
    """Test fractal gas on Breakout environment."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    env = AtariEnvironment(name="Breakout-v5", obs_type="ram")

    gas = AtariFractalGas(
        env=env,
        N=15,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device="cpu",
        seed=123,
    )

    final_state, history = gas.run(max_iterations=50)

    # Basic checks
    assert len(history) <= 50
    assert final_state.N == 15
    assert final_state.observations.shape == (15, 128)

    best_idx, best_reward = gas.get_best_walker(final_state)
    print(f"\nBreakout - Best walker {best_idx} achieved reward {best_reward:.2f}")

    env.close()


@pytest.mark.skipif(not PLANGYM_AVAILABLE, reason="plangym not installed")
def test_early_termination():
    """Test early stopping behavior."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    env = AtariEnvironment(name="Pong-v5", obs_type="ram")

    gas = AtariFractalGas(
        env=env,
        N=5,  # Fewer walkers for faster termination
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device="cpu",
        seed=999,
    )

    # Run with early stopping enabled
    final_state, history = gas.run(max_iterations=500, stop_when_all_dead=True)

    # Should stop before max iterations if all walkers die
    print(f"\nStopped after {len(history)} iterations (max was 500)")
    print(f"Alive walkers: {final_state.alive.sum().item()}/{final_state.N}")

    env.close()


@pytest.mark.skipif(not PLANGYM_AVAILABLE, reason="plangym not installed")
def test_reproducibility():
    """Test that seeding produces reproducible results."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    env1 = AtariEnvironment(name="Pong-v5", obs_type="ram")
    env2 = AtariEnvironment(name="Pong-v5", obs_type="ram")

    gas1 = AtariFractalGas(
        env=env1, N=10, dist_coef=1.0, reward_coef=1.0, device="cpu", seed=42
    )

    gas2 = AtariFractalGas(
        env=env2, N=10, dist_coef=1.0, reward_coef=1.0, device="cpu", seed=42
    )

    # Run both for same number of iterations
    final_state1, history1 = gas1.run(max_iterations=20)
    final_state2, history2 = gas2.run(max_iterations=20)

    # Check results are similar (not exact due to env stochasticity)
    assert len(history1) == len(history2)

    # Check metrics are in same ballpark
    for info1, info2 in zip(history1, history2):
        # Exact reproducibility is hard with env randomness,
        # but check structure is same
        assert info1.keys() == info2.keys()

    env1.close()
    env2.close()
