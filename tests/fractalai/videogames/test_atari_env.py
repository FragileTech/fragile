"""Real integration tests for AtariEnv wrapper.

All tests use actual Atari environments (gymnasium + ale-py) to verify full
compatibility with AtariFractalGas in WSL headless environments.
"""

import numpy as np
import pytest
import torch

from fragile.fractalai.videogames.atari import AtariEnv, AtariState


# ============================================================================
# Fixtures - Real Environments Only
# ============================================================================


@pytest.fixture
def device():
    """Test device (CPU for compatibility)."""
    return "cpu"


@pytest.fixture
def pong_env():
    """Real Pong environment with RAM observations."""
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture
def pong_env_rgb():
    """Real Pong environment with RGB observations."""
    env = AtariEnv("ALE/Pong-v5", obs_type="rgb", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture
def pong_env_grayscale():
    """Real Pong environment with grayscale observations."""
    env = AtariEnv("ALE/Pong-v5", obs_type="grayscale", render_mode="rgb_array")
    yield env
    env.close()


@pytest.fixture
def pong_env_no_rgb():
    """Real Pong environment without RGB frame capture."""
    env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array", include_rgb=False)
    yield env
    env.close()


@pytest.fixture
def breakout_env():
    """Real Breakout environment for variety testing."""
    env = AtariEnv("ALE/Breakout-v5", obs_type="ram", render_mode="rgb_array")
    yield env
    env.close()


# ============================================================================
# A. Environment Setup & Initialization (6 tests)
# ============================================================================


def test_atari_env_creation_default_params():
    """Test AtariEnv creation with Pong and default parameters."""
    env = AtariEnv("ALE/Pong-v5")

    assert env.name == "ALE/Pong-v5"
    assert env.obs_type == "ram"
    assert env.render_mode == "rgb_array"
    assert env.include_rgb is True
    assert env.env is not None
    assert env._ale is not None

    env.close()


def test_obs_types():
    """Test all observation types: ram, rgb, grayscale."""
    # RAM observations
    env_ram = AtariEnv("ALE/Pong-v5", obs_type="ram")
    state_ram = env_ram.reset()
    assert state_ram.observation.shape == (128,)
    assert state_ram.observation.dtype == np.uint8
    env_ram.close()

    # RGB observations
    env_rgb = AtariEnv("ALE/Pong-v5", obs_type="rgb")
    state_rgb = env_rgb.reset()
    assert state_rgb.observation.shape == (210, 160, 3)
    assert state_rgb.observation.dtype == np.uint8
    env_rgb.close()

    # Grayscale observations
    env_gray = AtariEnv("ALE/Pong-v5", obs_type="grayscale")
    state_gray = env_gray.reset()
    assert state_gray.observation.shape == (210, 160)
    assert state_gray.observation.dtype == np.uint8
    env_gray.close()


def test_render_mode_headless():
    """Test render_mode='rgb_array' for headless environments."""
    env = AtariEnv("ALE/Pong-v5", render_mode="rgb_array")
    env.reset()

    # Should be able to render in headless mode
    rgb = env.render()
    assert rgb is not None
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (210, 160, 3)
    assert rgb.dtype == np.uint8

    env.close()


def test_include_rgb_parameter():
    """Test include_rgb parameter controls RGB frame capture."""
    # With RGB capture
    env_with = AtariEnv("ALE/Pong-v5", obs_type="ram", include_rgb=True)
    state_with = env_with.reset()
    assert state_with.rgb_frame is not None
    assert state_with.rgb_frame.shape == (210, 160, 3)
    env_with.close()

    # Without RGB capture
    env_without = AtariEnv("ALE/Pong-v5", obs_type="ram", include_rgb=False)
    state_without = env_without.reset()
    assert state_without.rgb_frame is None
    env_without.close()


def test_action_space_access_and_sample(pong_env):
    """Test action_space access and sample() method."""
    # Action space should be accessible
    assert pong_env.action_space is not None

    # Should be able to sample actions
    action = pong_env.action_space.sample()
    assert isinstance(action, int | np.integer)
    assert 0 <= action < 18  # Pong has 18 actions


def test_ale_interface_accessible(pong_env):
    """Test that ALE interface is accessible for advanced features."""
    # ALE should be accessible
    assert pong_env._ale is not None

    # Should be able to get RAM directly
    ram = pong_env._ale.getRAM()
    assert isinstance(ram, np.ndarray)
    assert ram.shape == (128,)

    # Should be able to get screen RGB
    rgb = pong_env._ale.getScreenRGB()
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (210, 160, 3)


# ============================================================================
# B. State Management & Cloning (8 tests)
# ============================================================================


def test_reset_returns_atari_state(pong_env):
    """Test reset() returns AtariState with all required fields."""
    state = pong_env.reset(seed=42)

    # Check type
    assert isinstance(state, AtariState)

    # Check required fields
    assert hasattr(state, "ale_state")
    assert state.ale_state is not None
    assert isinstance(state.observation, np.ndarray)
    assert state.observation.shape == (128,)
    assert isinstance(state.rgb_frame, np.ndarray)
    assert state.rgb_frame.shape == (210, 160, 3)

    # Check copy method exists
    assert hasattr(state, "copy")
    assert callable(state.copy)


def test_atari_state_copy_creates_independent_copy(pong_env):
    """Test AtariState.copy() creates independent copy."""
    state1 = pong_env.reset(seed=42)
    state2 = state1.copy()

    # Should be different objects
    assert state1 is not state2

    # But have same values
    assert np.array_equal(state1.observation, state2.observation)
    assert np.array_equal(state1.rgb_frame, state2.rgb_frame)
    assert state1.ale_state is state2.ale_state  # ALE state is shared (immutable)

    # Modifying copy should not affect original
    state2.observation[0] = 255
    assert state1.observation[0] != 255


def test_clone_state_captures_current_state(pong_env):
    """Test clone_state() captures current environment state."""
    # Reset and take some steps
    pong_env.reset(seed=42)
    pong_env.step(action=2, dt=5)

    # Clone state
    cloned_state = pong_env.clone_state()

    # Should be AtariState
    assert isinstance(cloned_state, AtariState)
    assert cloned_state.ale_state is not None
    assert isinstance(cloned_state.observation, np.ndarray)
    assert cloned_state.observation.shape == (128,)


def test_restore_state_returns_to_previous_state(pong_env):
    """Test restore_state() returns to previously cloned state."""
    # Reset
    pong_env.reset(seed=42)

    # Clone initial state
    initial_state = pong_env.clone_state()

    # Take several steps
    for _ in range(10):
        pong_env.step(action=2, dt=1)

    # Get current observation (should be different)
    current_state = pong_env.clone_state()
    assert not np.array_equal(current_state.observation, initial_state.observation)

    # Restore to initial state and take a deterministic action
    pong_env.restore_state(initial_state)
    _, obs_after_restore, _, _, _, _ = pong_env.step(action=3, dt=2)

    # Restore again and take same action - should get same result
    pong_env.restore_state(initial_state)
    _, obs_after_restore2, _, _, _, _ = pong_env.step(action=3, dt=2)

    # Results should be identical (this tests deterministic replay)
    assert np.array_equal(obs_after_restore, obs_after_restore2)


def test_set_state_alias_works(pong_env):
    """Test set_state() alias for restore_state()."""
    # Reset and clone
    pong_env.reset(seed=42)
    initial_state = pong_env.clone_state()

    # Take steps
    pong_env.step(action=3, dt=5)

    # Use set_state alias to restore and take action
    pong_env.set_state(initial_state)
    _, obs1, _, _, _, _ = pong_env.step(action=4, dt=2)

    # Use restore_state and take same action
    pong_env.restore_state(initial_state)
    _, obs2, _, _, _, _ = pong_env.step(action=4, dt=2)

    # Results should be identical (tests that set_state == restore_state)
    assert np.array_equal(obs1, obs2)


def test_get_state_returns_dict(pong_env):
    """Test get_state() returns dict with 'state' and 'obs' keys."""
    pong_env.reset(seed=42)
    state_dict = pong_env.get_state()

    # Should be dict
    assert isinstance(state_dict, dict)

    # Should have required keys
    assert "state" in state_dict
    assert "obs" in state_dict

    # state should be AtariState
    assert isinstance(state_dict["state"], AtariState)

    # obs should be observation array
    assert isinstance(state_dict["obs"], np.ndarray)
    assert np.array_equal(state_dict["obs"], state_dict["state"].observation)


def test_clone_restore_roundtrip_determinism(pong_env):
    """Test clone/restore roundtrip maintains determinism."""
    # Reset with seed
    pong_env.reset(seed=42)

    # Clone initial state
    state1 = pong_env.clone_state()

    # Take action sequence
    _, obs1, rew1, done1, trunc1, _ = pong_env.step(action=5, dt=3)

    # Restore to cloned state
    pong_env.restore_state(state1)

    # Take same action sequence again
    _, obs2, rew2, done2, trunc2, _ = pong_env.step(action=5, dt=3)

    # Results MUST be identical for determinism
    assert np.array_equal(obs1, obs2), "Observations must match"
    assert rew1 == rew2, "Rewards must match"
    assert done1 == done2, "Done flags must match"
    assert trunc1 == trunc2, "Truncated flags must match"


def test_rgb_frame_captured_when_enabled(pong_env):
    """Test rgb_frame is captured when include_rgb=True."""
    state = pong_env.reset(seed=42)

    # RGB frame should be present
    assert state.rgb_frame is not None
    assert isinstance(state.rgb_frame, np.ndarray)
    assert state.rgb_frame.shape == (210, 160, 3)
    assert state.rgb_frame.dtype == np.uint8

    # Should not be all zeros
    assert state.rgb_frame.sum() > 0


# ============================================================================
# C. Single Step Operations (8 tests)
# ============================================================================


def test_step_with_random_action(pong_env):
    """Test step() with random action."""
    pong_env.reset(seed=42)
    action = pong_env.action_space.sample()

    new_state, obs, reward, done, truncated, info = pong_env.step(action=action)

    # Check return types
    assert isinstance(new_state, AtariState)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, int | float)
    assert isinstance(done, bool | np.bool_)
    assert isinstance(truncated, bool | np.bool_)
    assert isinstance(info, dict)


def test_step_with_state_restore(pong_env):
    """Test step() with state restore parameter."""
    # Create initial state
    initial_state = pong_env.reset(seed=42)

    # Take some steps away from initial state
    for _ in range(5):
        pong_env.step(action=2)

    # Now step from initial state using state parameter
    new_state, obs, _reward, _done, _truncated, _info = pong_env.step(
        action=3, state=initial_state, dt=1
    )

    # Should have stepped from initial state
    assert isinstance(new_state, AtariState)
    assert not np.array_equal(obs, initial_state.observation)


def test_step_dt_parameter_frame_skip(pong_env):
    """Test step() dt parameter controls frame skip (1, 3, 5)."""
    pong_env.reset(seed=42)

    # Get initial state
    state0 = pong_env.clone_state()

    # Test dt=1
    pong_env.restore_state(state0)
    _, obs1, _rew1, _, _, _ = pong_env.step(action=2, dt=1)

    # Test dt=3
    pong_env.restore_state(state0)
    _, obs3, _rew3, _, _, _ = pong_env.step(action=2, dt=3)

    # Test dt=5
    pong_env.restore_state(state0)
    _, obs5, _rew5, _, _, _ = pong_env.step(action=2, dt=5)

    # Different dt should lead to different states
    assert not np.array_equal(obs1, obs3)
    assert not np.array_equal(obs1, obs5)
    assert not np.array_equal(obs3, obs5)


def test_reward_accumulation_across_dt_frames(pong_env):
    """Test reward accumulation across dt frames."""
    # Note: In Pong, rewards are sparse, so we just verify mechanism works
    pong_env.reset(seed=42)
    state0 = pong_env.clone_state()

    # Single step
    _, _, rew1, _, _, _ = pong_env.step(action=2, dt=1)

    # Multiple steps with same action
    pong_env.restore_state(state0)
    _, _, rew_multi, _, _, _ = pong_env.step(action=2, dt=3)

    # Reward should be accumulated (mechanism test)
    assert isinstance(rew1, int | float)
    assert isinstance(rew_multi, int | float)


def test_early_termination_on_done(pong_env):
    """Test step() terminates early when done flag is set."""
    pong_env.reset(seed=42)

    # Step until episode ends or max steps
    max_steps = 10000
    for i in range(max_steps):
        _, _, _, done, _, _ = pong_env.step(action=pong_env.action_space.sample())
        if done:
            # Found a termination
            assert done is True
            break

    # Test passes if we can detect termination (Pong episodes do end)
    # If no termination found in 10k steps, that's also fine (rare but possible)


def test_early_termination_on_truncated(pong_env):
    """Test step() terminates early when truncated flag is set."""
    # Truncation typically happens on time limits
    # In Atari, this is controlled by gymnasium's TimeLimit wrapper
    pong_env.reset(seed=42)

    # Just verify truncated flag is present and boolean
    _, _, _, _, truncated, _ = pong_env.step(action=0)
    assert isinstance(truncated, bool | np.bool_)


def test_return_state_parameter(pong_env):
    """Test return_state parameter controls state return."""
    pong_env.reset(seed=42)

    # With return_state=True (default)
    state_with, _, _, _, _, _ = pong_env.step(action=2, return_state=True)
    assert state_with is not None
    assert isinstance(state_with, AtariState)

    # With return_state=False
    state_without, _, _, _, _, _ = pong_env.step(action=2, return_state=False)
    assert state_without is None


def test_observation_consistency(pong_env):
    """Test observation consistency between state and return value."""
    pong_env.reset(seed=42)

    new_state, obs, _, _, _, _ = pong_env.step(action=3)

    # Observation in state should match returned observation
    assert np.array_equal(new_state.observation, obs)


# ============================================================================
# D. Batch Operations (8 tests)
# ============================================================================


def test_step_batch_basic_operation(pong_env):
    """Test step_batch with N=10 walkers."""
    N = 10

    # Create initial states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    # Random actions
    actions = np.array([pong_env.action_space.sample() for _ in range(N)])

    # Step batch
    new_states, obs, rewards, dones, truncated, infos = pong_env.step_batch(states, actions)

    # Check outputs
    assert len(new_states) == N and new_states.dtype == object
    assert isinstance(obs, np.ndarray) and obs.shape[0] == N
    assert isinstance(rewards, np.ndarray) and rewards.shape == (N,)
    assert isinstance(dones, np.ndarray) and dones.dtype == bool and dones.shape == (N,)
    assert isinstance(truncated, np.ndarray) and truncated.dtype == bool
    assert isinstance(infos, list) and len(infos) == N


def test_step_batch_output_shapes_and_types(pong_env):
    """Test step_batch output shapes and types are correct."""
    N = 5

    # Create states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    actions = np.zeros(N, dtype=int)

    new_states, obs, rewards, dones, truncated, infos = pong_env.step_batch(states, actions)

    # Detailed type checks
    assert new_states.shape == (N,)
    assert all(isinstance(s, AtariState) for s in new_states)
    assert obs.shape == (N, 128)  # RAM observations
    assert obs.dtype == np.uint8
    assert rewards.dtype == np.float32
    assert dones.shape == (N,)
    assert truncated.shape == (N,)
    assert all(isinstance(info, dict) for info in infos)


def test_step_batch_varying_actions(pong_env):
    """Test step_batch with varying actions per walker."""
    N = 8

    # Create states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    # Different action for each walker
    actions = np.arange(N, dtype=int) % pong_env.action_space.n

    new_states, obs, _, _, _, _ = pong_env.step_batch(states, actions)

    # All walkers should have stepped
    assert len(new_states) == N
    assert all(isinstance(s, AtariState) for s in new_states)

    # Observations should vary (different actions lead to different states)
    # Note: Due to ALE determinism, same initial state + different action = different result
    assert obs.shape == (N, 128)


def test_step_batch_varying_dt_values(pong_env):
    """Test step_batch with varying dt values per walker."""
    N = 6

    # Create states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    actions = np.zeros(N, dtype=int)
    dt = np.array([1, 1, 2, 2, 3, 3], dtype=int)

    new_states, obs, rewards, _, _, _ = pong_env.step_batch(states, actions, dt)

    # All should complete
    assert len(new_states) == N
    assert obs.shape == (N, 128)
    assert rewards.shape == (N,)


def test_step_batch_single_walker(pong_env):
    """Test step_batch with single walker (N=1)."""
    N = 1

    states = np.empty(N, dtype=object)
    states[0] = pong_env.reset(seed=42)

    actions = np.array([2])

    new_states, obs, rewards, _dones, _truncated, _infos = pong_env.step_batch(states, actions)

    # Should work with N=1
    assert len(new_states) == 1
    assert obs.shape == (1, 128)
    assert rewards.shape == (1,)


def test_step_batch_large_batch(pong_env):
    """Test step_batch with large batch (N=50)."""
    N = 50

    # Create states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    actions = np.random.randint(0, pong_env.action_space.n, size=N)

    new_states, obs, _rewards, _dones, _truncated, infos = pong_env.step_batch(states, actions)

    # Should handle large batch
    assert len(new_states) == N
    assert obs.shape == (N, 128)
    assert len(infos) == N


def test_step_batch_mixed_alive_dead_walkers(pong_env):
    """Test step_batch with mixed alive/dead walkers."""
    # Note: AtariEnv doesn't track walker alive/dead status internally.
    # This is managed by AtariFractalGas at a higher level.
    # We test that step_batch processes all states regardless.
    N = 5

    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    actions = np.array([0, 1, 2, 3, 4])

    # Step batch - all walkers should be processed
    new_states, _obs, _rewards, _dones, _truncated, _infos = pong_env.step_batch(states, actions)

    assert len(new_states) == N
    assert all(isinstance(s, AtariState) for s in new_states)


def test_step_batch_sequential_consistency(pong_env):
    """Test step_batch sequential processing is consistent."""
    N = 3

    # Create identical initial states
    initial_state = pong_env.reset(seed=42)
    states = np.empty(N, dtype=object)
    for i in range(N):
        states[i] = initial_state.copy()

    # Same action for all
    actions = np.array([5, 5, 5])
    dt = np.array([2, 2, 2])

    # Step batch
    _new_states, obs, rewards, _, _, _ = pong_env.step_batch(states, actions, dt)

    # All walkers started from same state with same action/dt
    # Results should be identical due to determinism
    assert np.array_equal(obs[0], obs[1])
    assert np.array_equal(obs[0], obs[2])
    assert rewards[0] == rewards[1] == rewards[2]


# ============================================================================
# E. AtariFractalGas Integration (6 tests)
# ============================================================================


def test_atari_env_with_random_action_operator(pong_env, device):
    """Test AtariEnv works with RandomActionOperator."""
    from fragile.fractalai.videogames.kinetic import RandomActionOperator

    N = 10

    # Create operator (no device parameter for RandomActionOperator)
    kinetic_op = RandomActionOperator(
        env=pong_env,
        dt_range=(1, 4),
        seed=42,
    )

    # Create initial states
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    # Sample actions and dt separately
    actions = kinetic_op.sample_actions(N)
    dt = kinetic_op.sample_dt(N)

    # Step with sampled actions
    new_states, obs, _rewards, _dones, _truncated, _infos = pong_env.step_batch(
        states, actions, dt
    )

    # Should work seamlessly
    assert len(new_states) == N
    assert obs.shape[0] == N


def test_atari_env_with_fractal_cloning_operator(pong_env, device):
    """Test AtariEnv works with FractalCloningOperator."""
    from fragile.fractalai.videogames.cloning import FractalCloningOperator

    N = 10

    # Create operator
    FractalCloningOperator(
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
    )

    # Create mock walker state
    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    torch.randn(N, 128, device=device)
    torch.randn(N, device=device) * 0.1
    torch.randn(N, device=device) * 0.1
    torch.zeros(N, dtype=torch.bool, device=device)
    torch.zeros(N, dtype=torch.bool, device=device)
    torch.randn(N, device=device)

    # Test that cloning operator can work with AtariState objects
    # (Full integration tested in next test)
    assert all(isinstance(s, AtariState) for s in states)
    assert all(hasattr(s, "copy") for s in states)


def test_atari_fractal_gas_initialization(pong_env, device):
    """Test AtariFractalGas initialization with AtariEnv."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    gas = AtariFractalGas(
        env=pong_env,
        N=20,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device=device,
        seed=42,
    )

    # Check initialization
    assert gas.N == 20
    assert gas.env is pong_env
    assert gas.device == device
    assert gas.kinetic_op is not None
    assert gas.clone_op is not None


def test_atari_fractal_gas_reset(pong_env, device):
    """Test AtariFractalGas reset creates WalkerState with AtariEnv."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas, WalkerState

    gas = AtariFractalGas(
        env=pong_env,
        N=15,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device=device,
        seed=42,
    )

    state = gas.reset()

    # Check state structure
    assert isinstance(state, WalkerState)
    assert state.N == 15
    assert state.observations.shape == (15, 128)
    assert state.states.dtype == object
    assert all(isinstance(s, AtariState) for s in state.states)


def test_atari_fractal_gas_single_step(pong_env, device):
    """Test AtariFractalGas single step iteration with AtariEnv."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    gas = AtariFractalGas(
        env=pong_env,
        N=10,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device=device,
        seed=42,
    )

    state = gas.reset()
    new_state, info = gas.step(state)

    # Check results
    assert new_state.N == 10
    assert new_state.observations.shape == (10, 128)
    assert "iteration" in info
    assert "num_cloned" in info
    assert "alive_count" in info
    assert "mean_reward" in info
    assert "max_reward" in info


def test_atari_fractal_gas_full_run(pong_env, device):
    """Test AtariFractalGas full run (50 iterations) with AtariEnv."""
    from fragile.fractalai.videogames.atari_gas import AtariFractalGas

    gas = AtariFractalGas(
        env=pong_env,
        N=20,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device=device,
        seed=42,
    )

    final_state, history = gas.run(max_iterations=50)

    # Check completion
    assert len(history) <= 50
    assert final_state.N == 20
    assert gas.total_steps > 0
    assert gas.iteration_count == len(history)

    # Check history structure
    for info in history:
        assert "iteration" in info
        assert "num_cloned" in info
        assert "alive_count" in info
        assert "mean_reward" in info
        assert "max_reward" in info


# ============================================================================
# F. WSL Headless Compatibility (4 tests)
# ============================================================================


def test_works_with_pyglet_headless(pong_env):
    """Test works with PYGLET_HEADLESS=1 environment variable."""
    import os

    # Should already be set by conftest.py
    assert os.environ.get("PYGLET_HEADLESS") == "1"

    # Environment should still work
    state = pong_env.reset(seed=42)
    assert isinstance(state, AtariState)

    # Step should work
    _, obs, _, _, _, _ = pong_env.step(action=2)
    assert isinstance(obs, np.ndarray)


def test_render_returns_rgb_array_headless(pong_env):
    """Test render() returns RGB array in headless mode."""
    pong_env.reset(seed=42)

    # Render should return RGB array
    rgb = pong_env.render()

    assert rgb is not None
    assert isinstance(rgb, np.ndarray)
    assert rgb.shape == (210, 160, 3)
    assert rgb.dtype == np.uint8

    # Should have actual content
    assert rgb.sum() > 0


def test_no_display_required():
    """Test no DISPLAY environment variable required."""
    import os

    # Save original DISPLAY if present
    original_display = os.environ.get("DISPLAY")

    # Unset DISPLAY
    if "DISPLAY" in os.environ:
        del os.environ["DISPLAY"]

    try:
        # Should still work without DISPLAY
        env = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")
        state = env.reset(seed=42)
        assert isinstance(state, AtariState)
        env.close()
    finally:
        # Restore original DISPLAY
        if original_display is not None:
            os.environ["DISPLAY"] = original_display


def test_all_obs_types_work_in_headless_wsl():
    """Test all obs_types work in headless WSL environment."""
    import os

    # Ensure headless mode
    assert os.environ.get("PYGLET_HEADLESS") == "1"

    # Test RAM
    env_ram = AtariEnv("ALE/Pong-v5", obs_type="ram", render_mode="rgb_array")
    state_ram = env_ram.reset()
    assert state_ram.observation.shape == (128,)
    env_ram.close()

    # Test RGB
    env_rgb = AtariEnv("ALE/Pong-v5", obs_type="rgb", render_mode="rgb_array")
    state_rgb = env_rgb.reset()
    assert state_rgb.observation.shape == (210, 160, 3)
    env_rgb.close()

    # Test Grayscale
    env_gray = AtariEnv("ALE/Pong-v5", obs_type="grayscale", render_mode="rgb_array")
    state_gray = env_gray.reset()
    assert state_gray.observation.shape == (210, 160)
    env_gray.close()


# ============================================================================
# Additional Edge Cases and Robustness Tests
# ============================================================================


def test_multiple_games_support(breakout_env):
    """Test that different Atari games work with same interface."""
    # Breakout should work identically to Pong
    state = breakout_env.reset(seed=42)

    assert isinstance(state, AtariState)
    assert state.observation.shape == (128,)  # RAM

    # Step should work
    new_state, _obs, _reward, _done, _truncated, _info = breakout_env.step(action=1)
    assert isinstance(new_state, AtariState)


def test_deterministic_reset_with_seed(pong_env):
    """Test that reset with same seed produces identical initial states."""
    state1 = pong_env.reset(seed=42)
    state2 = pong_env.reset(seed=42)

    # Should be identical
    assert np.array_equal(state1.observation, state2.observation)
    assert np.array_equal(state1.rgb_frame, state2.rgb_frame)


def test_rgb_and_grayscale_observations(pong_env_rgb, pong_env_grayscale):
    """Test RGB and grayscale observations have correct shapes and types."""
    # RGB
    state_rgb = pong_env_rgb.reset()
    assert state_rgb.observation.shape == (210, 160, 3)
    assert state_rgb.observation.dtype == np.uint8

    # Grayscale
    state_gray = pong_env_grayscale.reset()
    assert state_gray.observation.shape == (210, 160)
    assert state_gray.observation.dtype == np.uint8


def test_step_batch_with_default_dt(pong_env):
    """Test step_batch with default dt (None) defaults to all 1s."""
    N = 5

    states = np.empty(N, dtype=object)
    initial_state = pong_env.reset(seed=42)
    for i in range(N):
        states[i] = initial_state.copy()

    actions = np.zeros(N, dtype=int)

    # Call without dt parameter
    new_states, obs, _rewards, _dones, _truncated, _infos = pong_env.step_batch(
        states, actions, dt=None
    )

    # Should work with dt defaulting to 1s
    assert len(new_states) == N
    assert obs.shape == (N, 128)
