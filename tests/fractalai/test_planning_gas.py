"""Tests for PlanningFractalGas two-level planner."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from fragile.fractalai.planning_gas import PlanningFractalGas, PlanningTrajectory
from fragile.fractalai.videogames.atari_gas import AtariFractalGas


# ---------------------------------------------------------------------------
# Mock environments
# ---------------------------------------------------------------------------


class MockEnv:
    """Minimal mock env for discrete (Atari-like) actions."""

    def __init__(self, obs_shape=(128,), n_actions=18):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.random.randint(0, n_actions)
        self.action_space.shape = (n_actions,)
        self._step_count = 0
        self._done_after = None  # set to int to trigger done

    def reset(self, **kwargs):
        self._step_count = 0
        state = np.zeros(4, dtype=np.float32)
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        return state, observation, {}

    def sample_action(self):
        return np.random.randint(0, self.n_actions)

    def step_batch(self, states, actions, dt, **kwargs):
        N = len(states)
        self._step_count += 1
        new_states = np.array(
            [np.random.randn(4).astype(np.float32) for _ in range(N)],
            dtype=object,
        )
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.ones(N, dtype=np.float32) * 0.5
        dones = np.zeros(N, dtype=bool)
        if N > 10:
            # For inner gas: kill some walkers
            dones[np.random.choice(N, size=max(1, N // 10), replace=False)] = True
        if self._done_after is not None and self._step_count >= self._done_after:
            dones[:] = True
        truncated = np.zeros(N, dtype=bool)
        infos = [{} for _ in range(N)]
        return new_states, observations, rewards, dones, truncated, infos


class MockContinuousEnv:
    """Mock env with continuous action space."""

    def __init__(self, obs_shape=(8,), action_dim=2):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.random.randn(action_dim).astype(np.float64)
        self.action_space.shape = (action_dim,)

    def reset(self, **kwargs):
        state = np.zeros(4, dtype=np.float32)
        observation = np.zeros(self.obs_shape, dtype=np.float32)
        return state, observation, {}

    def sample_action(self):
        return np.random.randn(self.action_dim).astype(np.float64)

    def step_batch(self, states, actions, dt, **kwargs):
        N = len(states)
        new_states = np.array(
            [np.random.randn(4).astype(np.float32) for _ in range(N)],
            dtype=object,
        )
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.ones(N, dtype=np.float32) * 0.1
        dones = np.zeros(N, dtype=bool)
        truncated = np.zeros(N, dtype=bool)
        infos = [{} for _ in range(N)]
        return new_states, observations, rewards, dones, truncated, infos


@pytest.fixture
def mock_env():
    return MockEnv()


@pytest.fixture
def mock_continuous_env():
    return MockContinuousEnv()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanningGasCreation:
    def test_constructor_defaults(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=10, tau_inner=3)
        assert pg.N == 10
        assert pg.tau_inner == 3
        assert pg.outer_dt == 1
        assert isinstance(pg.inner_gas, AtariFractalGas)

    def test_constructor_custom_inner_gas_cls(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=8, tau_inner=2, inner_gas_cls=AtariFractalGas)
        assert isinstance(pg.inner_gas, AtariFractalGas)
        assert pg.inner_gas.N == 8

    def test_constructor_forwards_params(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=6, tau_inner=4, dist_coef=2.0, outer_dt=3)
        assert pg.inner_gas.clone_op.dist_coef == 2.0
        assert pg.outer_dt == 3


class TestReset:
    def test_reset_returns_3tuple(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=10, tau_inner=2)
        result = pg.reset()
        assert isinstance(result, tuple)
        assert len(result) == 3
        _state, _obs, info = result
        assert isinstance(info, dict)

    def test_reset_obs_shape(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=10, tau_inner=2)
        _, obs, _ = pg.reset()
        assert obs.shape == mock_env.obs_shape


class TestPlanAction:
    def test_returns_valid_action(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=3)
        state, obs, info = pg.reset()
        action, plan_info = pg.plan_action(state, obs, info)
        # Discrete env → action should be int
        assert isinstance(action, int | np.integer)
        assert isinstance(plan_info, dict)
        assert "alive_count" in plan_info
        assert "inner_max_reward" in plan_info

    def test_continuous_action_shape(self, mock_continuous_env):
        from fragile.fractalai.robots.robotic_gas import RoboticFractalGas

        pg = PlanningFractalGas(
            env=mock_continuous_env,
            N=20,
            tau_inner=3,
            inner_gas_cls=RoboticFractalGas,
        )
        state, obs, info = pg.reset()
        action, _ = pg.plan_action(state, obs, info)
        # Continuous env → action should be array with correct shape
        assert isinstance(action, np.ndarray)
        assert action.shape == (mock_continuous_env.action_dim,)


class TestRootActionPropagation:
    def test_root_actions_reflect_cloning(self, mock_env):
        """After inner run, root_actions should have been propagated via cloning."""
        pg = PlanningFractalGas(env=mock_env, N=30, tau_inner=5)
        state, obs, info = pg.reset()

        # Run planning manually to inspect root actions
        inner_state = pg.inner_gas.reset(initial_state=(state, obs, info))
        root_actions = None

        for k in range(pg.tau_inner):
            inner_state, step_info = pg.inner_gas.step(inner_state)
            if k == 0:
                root_actions = inner_state.actions.copy()
            else:
                wc = step_info["will_clone"].cpu().numpy()
                companions = step_info["clone_companions"].cpu().numpy()
                root_actions[wc] = root_actions[companions[wc]]

        # root_actions should exist and have correct shape
        assert root_actions is not None
        assert len(root_actions) == 30
        # Each root action should be a valid action (integer for Atari)
        for a in root_actions:
            assert isinstance(a, int | np.integer)


class TestStep:
    def test_step_returns_correct_tuple(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2)
        state, obs, info = pg.reset()
        result = pg.step(state, obs, info)
        assert len(result) == 7
        _new_state, _new_obs, reward, done, truncated, _new_info, step_info = result
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert "action" in step_info
        assert "plan_info" in step_info

    def test_step_applies_planned_action(self, mock_env):
        """The outer env should receive the planned action."""
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2)
        state, obs, info = pg.reset()
        _, _, _, _, _, _, step_info = pg.step(state, obs, info)
        # The action should be a valid discrete action
        assert isinstance(step_info["action"], int | np.integer)


class TestRun:
    def test_run_produces_trajectory(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2)
        traj = pg.run(max_steps=5)
        assert isinstance(traj, PlanningTrajectory)
        assert traj.num_steps == 5
        assert len(traj.rewards) == 5
        assert len(traj.cumulative_rewards) == 5
        assert len(traj.actions) == 5
        assert len(traj.planning_infos) == 5
        # states includes initial + one per step
        assert len(traj.states) == 6
        assert len(traj.observations) == 6
        assert len(traj.dones) == 5

    def test_run_stops_on_done(self):
        env = MockEnv()
        env._done_after = 3  # outer steps trigger done after 3 calls
        pg = PlanningFractalGas(env=env, N=20, tau_inner=2)
        traj = pg.run(max_steps=100)
        # Should stop well before 100 steps
        assert traj.num_steps <= 10
        assert traj.dones[-1] is True

    def test_run_frames_recorded(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2, record_frames=True)
        traj = pg.run(max_steps=3, render=True)
        assert traj.frames is not None
        # frames list exists (may be empty since mock doesn't render)
        assert isinstance(traj.frames, list)


class TestDiscreteActionRounding:
    def test_integer_actions_rounded(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=3)
        state, obs, info = pg.reset()
        action, _ = pg.plan_action(state, obs, info)
        assert isinstance(action, int | np.integer)

    def test_discrete_action_is_int_type(self, mock_env):
        """Verify the action is a Python int, not a float."""
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2)
        state, obs, info = pg.reset()
        action, _ = pg.plan_action(state, obs, info)
        # Should be int, not float
        assert type(action) in {int, np.int64, np.int32, np.intp}


class TestAllDeadFallback:
    def test_all_dead_uses_best_walker(self):
        """When all walkers are dead, use best walker's root action."""
        # Create env that kills all walkers immediately
        env = MockEnv()
        original_step_batch = env.step_batch

        def kill_all_step_batch(states, actions, dt, **kwargs):
            result = original_step_batch(states, actions, dt, **kwargs)
            new_states, obs, rewards, dones, trunc, infos = result
            dones[:] = True
            rewards[:] = np.random.randn(len(states)).astype(np.float32)
            return new_states, obs, rewards, dones, trunc, infos

        env.step_batch = kill_all_step_batch

        pg = PlanningFractalGas(env=env, N=20, tau_inner=3)
        state, obs, info = pg.reset()
        action, plan_info = pg.plan_action(state, obs, info)
        # Should still return a valid action despite all walkers being dead
        assert action is not None
        assert plan_info["alive_count"] == 0


class TestTrajectoryProperties:
    def test_total_reward(self):
        traj = PlanningTrajectory(
            rewards=[1.0, 2.0, 3.0],
            cumulative_rewards=[1.0, 3.0, 6.0],
            actions=[0, 1, 2],
        )
        assert traj.total_reward == 6.0

    def test_num_steps(self):
        traj = PlanningTrajectory(actions=[0, 1, 2, 3])
        assert traj.num_steps == 4

    def test_empty_trajectory(self):
        traj = PlanningTrajectory()
        assert traj.total_reward == 0.0
        assert traj.num_steps == 0

    def test_trajectory_cumulative_reward(self, mock_env):
        pg = PlanningFractalGas(env=mock_env, N=20, tau_inner=2)
        traj = pg.run(max_steps=5)
        # Cumulative rewards should be monotonically consistent
        expected_cum = 0.0
        for i, r in enumerate(traj.rewards):
            expected_cum += r
            assert abs(traj.cumulative_rewards[i] - expected_cum) < 1e-6
