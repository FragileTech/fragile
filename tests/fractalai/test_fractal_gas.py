"""Tests for FractalGas base class, initial_state support, and subclass hierarchy."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from fragile.fractalai.fractal_gas import FractalGas, WalkerState
from fragile.fractalai.videogames.atari_gas import AtariFractalGas


# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal mock env returning (state, observation, info) from reset."""

    def __init__(self, obs_shape=(128,), action_space_size=18):
        self.obs_shape = obs_shape
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.random.randint(0, action_space_size)
        self.action_space.shape = (action_space_size,)

    def reset(self, **kwargs):
        state = np.zeros(4, dtype=np.float32)
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        return state, observation, {}

    def sample_action(self):
        return np.random.randint(0, 18)

    def step_batch(self, states, actions, dt, **kwargs):
        N = len(states)
        new_states = np.array(
            [np.random.randn(4).astype(np.float32) for _ in range(N)],
            dtype=object,
        )
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.random.randn(N).astype(np.float32) * 0.1
        dones = np.zeros(N, dtype=bool)
        dones[np.random.choice(N, size=max(1, N // 10), replace=False)] = True
        truncated = np.zeros(N, dtype=bool)
        infos = [{} for _ in range(N)]
        return new_states, observations, rewards, dones, truncated, infos


@pytest.fixture
def mock_env():
    return MockEnv()


# ---------------------------------------------------------------------------
# Hierarchy & isinstance
# ---------------------------------------------------------------------------


class TestSubclassHierarchy:
    """Verify the inheritance chain works correctly."""

    def test_atari_is_fractal_gas(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        assert isinstance(gas, FractalGas)
        assert isinstance(gas, AtariFractalGas)

    def test_robotic_is_fractal_gas(self, mock_env):
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        gas = RoboticFractalGas(env=mock_env, N=5)
        assert isinstance(gas, FractalGas)
        assert isinstance(gas, RoboticFractalGas)

    def test_walker_state_importable_from_all_paths(self):
        from fragile.fractalai.fractal_gas import WalkerState as WS1
        from fragile.fractalai.robots.robotic_gas import WalkerState as WS4
        from fragile.fractalai.videogames import WalkerState as WS3
        from fragile.fractalai.videogames.atari_gas import WalkerState as WS2

        assert WS1 is WS2 is WS3 is WS4


# ---------------------------------------------------------------------------
# DEFAULT_DT_RANGE
# ---------------------------------------------------------------------------


class TestDefaultDtRange:
    def test_base_default(self):
        assert FractalGas.DEFAULT_DT_RANGE == (1, 4)

    def test_atari_default(self):
        assert AtariFractalGas.DEFAULT_DT_RANGE == (1, 4)

    def test_robotic_default(self):
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        assert RoboticFractalGas.DEFAULT_DT_RANGE == (1, 1)

    def test_dt_range_none_uses_class_default(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5, dt_range=None)
        assert gas.kinetic_op.dt_range == (1, 4)

    def test_dt_range_explicit_overrides(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5, dt_range=(2, 8))
        assert gas.kinetic_op.dt_range == (2, 8)


# ---------------------------------------------------------------------------
# initial_state on reset()
# ---------------------------------------------------------------------------


class TestInitialState:
    """Tests for the new initial_state parameter."""

    def test_reset_with_initial_state_uses_provided_obs(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=10)
        custom_obs = np.ones(128, dtype=np.float32) * 42.0
        custom_state = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        custom_info = {"custom_key": "hello"}

        state = gas.reset(initial_state=(custom_state, custom_obs, custom_info))

        assert state.N == 10
        # All walkers should have the provided observation
        assert torch.allclose(
            state.observations,
            torch.full((10, 128), 42.0),
        )

    def test_reset_with_initial_state_replicates_env_state(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        custom_state = np.array([9.0, 8.0, 7.0, 6.0], dtype=np.float32)
        custom_obs = np.zeros(128, dtype=np.float32)

        ws = gas.reset(initial_state=(custom_state, custom_obs, {}))

        # Each walker state should be an independent copy of custom_state
        for i in range(5):
            np.testing.assert_array_equal(ws.states[i], custom_state)
        # Mutating one walker state must not affect others
        ws.states[0][0] = -999
        assert ws.states[1][0] == 9.0

    def test_reset_with_initial_state_resets_metrics(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        # Run a few steps first to accumulate metrics
        s = gas.reset()
        for _ in range(3):
            s, _ = gas.step(s)
        assert gas.total_steps > 0

        # Reset with initial_state should zero metrics
        gas.reset(initial_state=(np.zeros(4), np.zeros(128), {}))
        assert gas.total_steps == 0
        assert gas.total_clones == 0
        assert gas.iteration_count == 0

    def test_reset_with_initial_state_zeros_rewards(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        ws = gas.reset(initial_state=(np.zeros(4), np.ones(128), {}))

        assert torch.all(ws.rewards == 0)
        assert torch.all(ws.step_rewards == 0)
        assert torch.all(~ws.dones)
        assert torch.all(~ws.truncated)

    def test_reset_with_initial_state_info_replicated(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=3)
        info = {"lives": 3}
        ws = gas.reset(initial_state=(np.zeros(4), np.zeros(128), info))

        for i in range(3):
            assert ws.infos[i] == {"lives": 3}
        # infos should be independent copies
        ws.infos[0]["lives"] = 0
        assert ws.infos[1]["lives"] == 3

    def test_reset_without_initial_state_calls_env(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        ws = gas.reset()  # no initial_state
        # Should still produce valid state from env.reset()
        assert ws.N == 5
        assert ws.observations.shape == (5, 128)

    def test_run_with_initial_state(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=10)
        custom_obs = np.ones(128, dtype=np.float32) * 7.0

        final_state, history = gas.run(
            max_iterations=5,
            initial_state=(np.zeros(4), custom_obs, {}),
        )

        assert len(history) == 5
        assert isinstance(final_state, WalkerState)
        assert gas.iteration_count == 5

    def test_run_with_tree_initial_state(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=10)
        custom_obs = np.ones(128, dtype=np.float32) * 3.0

        tree = gas.run_with_tree(
            max_iterations=5,
            game_name="test_game",
            initial_state=(np.zeros(4), custom_obs, {}),
        )

        # Tree should have recorded all steps
        assert tree.game_name == "test_game"
        assert len(tree.iterations) == 5

    def test_step_after_initial_state_reset(self, mock_env):
        """Ensure step() works normally after initial_state reset."""
        gas = AtariFractalGas(env=mock_env, N=10)
        state = gas.reset(initial_state=(np.zeros(4), np.zeros(128, dtype=np.float32), {}))

        new_state, info = gas.step(state)

        assert new_state.N == 10
        assert "iteration" in info
        assert info["iteration"] == 1
        assert gas.total_steps == 10


# ---------------------------------------------------------------------------
# Backward compatibility: game_name / task_name wrappers
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_atari_run_with_tree_game_name(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        tree = gas.run_with_tree(max_iterations=3, game_name="Pong")
        assert tree.game_name == "Pong"

    def test_atari_run_with_tree_task_label(self, mock_env):
        gas = AtariFractalGas(env=mock_env, N=5)
        tree = gas.run_with_tree(max_iterations=3, task_label="Breakout")
        assert tree.game_name == "Breakout"

    def test_robotic_run_with_tree_task_name(self, mock_env):
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        gas = RoboticFractalGas(env=mock_env, N=5)
        tree = gas.run_with_tree(max_iterations=3, task_name="cartpole")
        assert tree.game_name == "cartpole"

    def test_robotic_run_with_tree_task_label(self, mock_env):
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        gas = RoboticFractalGas(env=mock_env, N=5)
        tree = gas.run_with_tree(max_iterations=3, task_label="walker")
        assert tree.game_name == "walker"
