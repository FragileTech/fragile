"""Tests for random action kinetic operator."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from fragile.fractalai.videogames.kinetic import RandomActionOperator


class MockEnv:
    """Mock plangym environment for testing."""

    def __init__(self, obs_shape=(128,), action_space_size=18):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.random.randint(0, action_space_size)

    def sample_action(self):
        """Sample a random action (plangym standard)."""
        return np.random.randint(0, self.action_space_size)

    def step_batch(self, states, actions, dt):
        """Mock batch stepping."""
        N = len(states)

        # Create mock outputs
        new_states = np.array([self._mock_state() for _ in range(N)], dtype=object)
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.random.randn(N).astype(np.float32)
        dones = np.random.rand(N) > 0.9  # 10% chance of done
        truncated = np.random.rand(N) > 0.95  # 5% chance of truncated
        infos = [{"step": i} for i in range(N)]

        return new_states, observations, rewards, dones, truncated, infos

    def _mock_state(self):
        """Create a mock state object."""
        state = MagicMock()
        state.copy = self._mock_state
        return state


@pytest.fixture
def mock_env():
    """Create a mock environment."""
    return MockEnv()


@pytest.fixture
def kinetic_op(mock_env):
    """Create a kinetic operator for testing."""
    return RandomActionOperator(env=mock_env, dt_range=(1, 4), seed=42)


def test_sample_actions(kinetic_op):
    """Test action sampling."""
    N = 10
    actions = kinetic_op.sample_actions(N)

    # Check output
    assert actions.shape == (N,)
    assert actions.dtype in {np.int64, np.int32, np.int_}
    assert (actions >= 0).all()
    assert (actions < 18).all()  # Mock env has 18 actions


def test_sample_actions_with_custom_sampler(mock_env):
    """Test action sampling with custom sampler."""
    N = 10

    def custom_sampler(n):
        return np.ones(n, dtype=int) * 5  # Always return action 5

    kinetic_op = RandomActionOperator(env=mock_env, action_sampler=custom_sampler, seed=42)

    actions = kinetic_op.sample_actions(N)

    # Check that custom sampler was used
    assert (actions == 5).all()


def test_sample_dt(kinetic_op):
    """Test frame skip sampling."""
    N = 20
    dt = kinetic_op.sample_dt(N)

    # Check output
    assert dt.shape == (N,)
    assert dt.dtype in {np.int64, np.int32, np.int_}
    assert (dt >= 1).all()  # Min dt
    assert (dt <= 4).all()  # Max dt

    # Check that we get some variation
    assert len(np.unique(dt)) > 1


def test_apply_basic(kinetic_op, mock_env):
    """Test full kinetic step with state updates."""
    N = 10

    # Create mock states
    states = np.array([mock_env._mock_state() for _ in range(N)], dtype=object)

    # Apply kinetic operator
    new_states, observations, rewards, dones, truncated, infos = kinetic_op.apply(states)

    # Check outputs
    assert len(new_states) == N
    assert observations.shape[0] == N
    assert rewards.shape == (N,)
    assert dones.shape == (N,)
    assert truncated.shape == (N,)
    assert len(infos) == N

    # Check types
    assert new_states.dtype == object
    assert observations.dtype == np.float32
    assert rewards.dtype == np.float32
    assert dones.dtype == bool
    assert truncated.dtype == bool


def test_apply_stores_last_values(kinetic_op, mock_env):
    """Test that last_actions and last_dt are stored."""
    N = 8
    states = np.array([mock_env._mock_state() for _ in range(N)], dtype=object)

    # Apply kinetic operator
    kinetic_op.apply(states)

    # Check that values were stored
    assert kinetic_op.last_actions is not None
    assert kinetic_op.last_dt is not None
    assert len(kinetic_op.last_actions) == N
    assert len(kinetic_op.last_dt) == N


def test_apply_with_custom_actions(kinetic_op, mock_env):
    """Test applying specific actions."""
    N = 5
    states = np.array([mock_env._mock_state() for _ in range(N)], dtype=object)

    # Provide custom actions
    custom_actions = np.array([0, 1, 2, 3, 4])
    custom_dt = np.array([1, 2, 3, 2, 1])

    # Apply kinetic operator
    kinetic_op.apply(states, actions=custom_actions, dt=custom_dt)

    # Check that custom values were stored
    assert np.array_equal(kinetic_op.last_actions, custom_actions)
    assert np.array_equal(kinetic_op.last_dt, custom_dt)


def test_apply_with_5_tuple_return(mock_env):
    """Test handling of 5-tuple return (no truncated)."""

    # Create env that returns 5-tuple
    class MockEnv5Tuple(MockEnv):
        def step_batch(self, states, actions, dt):
            N = len(states)
            new_states = np.array([self._mock_state() for _ in range(N)], dtype=object)
            observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
            rewards = np.random.randn(N).astype(np.float32)
            dones = np.random.rand(N) > 0.9
            infos = [{"step": i} for i in range(N)]
            return new_states, observations, rewards, dones, infos  # 5-tuple

    env_5tuple = MockEnv5Tuple()
    kinetic_op = RandomActionOperator(env=env_5tuple, dt_range=(1, 4))

    N = 10
    states = np.array([env_5tuple._mock_state() for _ in range(N)], dtype=object)

    # Apply kinetic operator
    _new_states, _observations, _rewards, _dones, truncated, _infos = kinetic_op.apply(states)

    # Check that truncated was filled in
    assert truncated.shape == (N,)
    assert truncated.dtype == bool
    assert not truncated.any()  # Should be all False


def test_seeding_reproducibility():
    """Test that seeding produces reproducible results for dt."""
    N = 10

    # Create first operator and sample
    mock_env1 = MockEnv()
    kinetic_op1 = RandomActionOperator(env=mock_env1, dt_range=(1, 4), seed=123)
    dt1 = kinetic_op1.sample_dt(N)

    # Create second operator with same seed and sample
    mock_env2 = MockEnv()
    kinetic_op2 = RandomActionOperator(env=mock_env2, dt_range=(1, 4), seed=123)
    dt2 = kinetic_op2.sample_dt(N)

    # Check reproducibility for dt
    assert np.array_equal(dt1, dt2), "Same seed should produce same dt sequence"


def test_fallback_to_action_space_sample():
    """Test fallback when sample_action is not available."""

    class MockEnvNoSampleAction:
        """Mock env without sample_action method."""

        def __init__(self):
            self.action_space = MagicMock()
            self.action_space.sample = lambda: np.random.randint(0, 18)

    mock_env_no_sample = MockEnvNoSampleAction()
    kinetic_op = RandomActionOperator(env=mock_env_no_sample, dt_range=(1, 4))

    N = 10
    actions = kinetic_op.sample_actions(N)

    # Should still work using action_space.sample()
    assert actions.shape == (N,)
    assert (actions >= 0).all()


def test_no_action_sampling_raises_error():
    """Test that error is raised when no sampling method available."""

    class BadEnv:
        """Environment without action sampling."""

    bad_env = BadEnv()
    kinetic_op = RandomActionOperator(env=bad_env, dt_range=(1, 4))

    with pytest.raises(AttributeError):
        kinetic_op.sample_actions(10)
