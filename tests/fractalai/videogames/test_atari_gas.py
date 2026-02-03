"""Tests for Atari fractal gas main algorithm."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from fragile.fractalai.videogames.atari_gas import AtariFractalGas, WalkerState


class MockEnv:
    """Mock plangym environment for testing."""

    def __init__(self, obs_shape=(128,), action_space_size=18):
        self.obs_shape = obs_shape
        self.action_space_size = action_space_size
        self.action_space = MagicMock()
        self.action_space.sample = lambda: np.random.randint(0, action_space_size)
        self.step_count = 0

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        state = MagicMock()
        state.copy = lambda: self._mock_state()
        return state

    def sample_action(self):
        """Sample a random action."""
        return np.random.randint(0, self.action_space_size)

    def step_batch(self, states, actions, dt):
        """Mock batch stepping."""
        N = len(states)
        self.step_count += N

        # Create mock outputs
        new_states = np.array([self._mock_state() for _ in range(N)], dtype=object)
        observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
        rewards = np.random.randn(N).astype(np.float32) * 0.1
        dones = np.zeros(N, dtype=bool)
        dones[np.random.choice(N, size=max(1, N // 10), replace=False)] = True
        truncated = np.zeros(N, dtype=bool)
        infos = [{"step": i} for i in range(N)]

        return new_states, observations, rewards, dones, truncated, infos

    def _mock_state(self):
        """Create a mock state object."""
        state = MagicMock()
        state.copy = lambda: self._mock_state()
        return state


@pytest.fixture
def mock_env():
    """Create a mock environment."""
    return MockEnv(obs_shape=(128,))


@pytest.fixture
def device():
    """Test device."""
    return "cpu"


def test_walker_state_creation(device):
    """Test WalkerState dataclass initialization."""
    N = 10
    obs_shape = (128,)

    # Create walker state
    states = np.array([MagicMock() for _ in range(N)], dtype=object)
    observations = torch.randn(N, *obs_shape, device=device)
    rewards = torch.zeros(N, device=device)
    step_rewards = torch.zeros(N, device=device)
    dones = torch.zeros(N, dtype=torch.bool, device=device)
    truncated = torch.zeros(N, dtype=torch.bool, device=device)
    actions = np.zeros(N, dtype=int)
    dt = np.ones(N, dtype=int)
    infos = [{} for _ in range(N)]

    walker_state = WalkerState(
        states=states,
        observations=observations,
        rewards=rewards,
        step_rewards=step_rewards,
        dones=dones,
        truncated=truncated,
        actions=actions,
        dt=dt,
        infos=infos,
    )

    # Check properties
    assert walker_state.N == N
    assert walker_state.alive.sum() == N  # All alive initially
    assert walker_state.device.type == device
    assert walker_state.observations.shape == (N, *obs_shape)


def test_walker_state_alive_property(device):
    """Test alive property correctly identifies living walkers."""
    N = 10
    obs_shape = (128,)

    states = np.array([MagicMock() for _ in range(N)], dtype=object)
    observations = torch.randn(N, *obs_shape, device=device)
    rewards = torch.zeros(N, device=device)
    step_rewards = torch.zeros(N, device=device)
    dones = torch.zeros(N, dtype=torch.bool, device=device)
    truncated = torch.zeros(N, dtype=torch.bool, device=device)

    # Mark some walkers as done or truncated
    dones[2] = True
    dones[5] = True
    truncated[7] = True

    actions = np.zeros(N, dtype=int)
    dt = np.ones(N, dtype=int)
    infos = [{} for _ in range(N)]

    walker_state = WalkerState(
        states=states,
        observations=observations,
        rewards=rewards,
        step_rewards=step_rewards,
        dones=dones,
        truncated=truncated,
        actions=actions,
        dt=dt,
        infos=infos,
    )

    # Check alive mask
    alive = walker_state.alive
    assert not alive[2]  # Done
    assert not alive[5]  # Done
    assert not alive[7]  # Truncated
    assert alive[0]  # Alive
    assert alive.sum() == N - 3  # 3 dead walkers


def test_walker_state_clone(device):
    """Test state cloning method."""
    N = 8
    obs_shape = (128,)

    # Create initial state
    states = np.array([MagicMock() for _ in range(N)], dtype=object)
    observations = torch.arange(N * 128, device=device, dtype=torch.float32).reshape(
        N, 128
    )
    rewards = torch.arange(N, device=device, dtype=torch.float32)
    step_rewards = torch.zeros(N, device=device)
    dones = torch.zeros(N, dtype=torch.bool, device=device)
    truncated = torch.zeros(N, dtype=torch.bool, device=device)
    actions = np.arange(N, dtype=int)
    dt = np.ones(N, dtype=int)
    infos = [{"id": i} for i in range(N)]

    walker_state = WalkerState(
        states=states,
        observations=observations,
        rewards=rewards,
        step_rewards=step_rewards,
        dones=dones,
        truncated=truncated,
        actions=actions,
        dt=dt,
        infos=infos,
    )

    # Define cloning
    companions = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=device)
    will_clone = torch.tensor(
        [True, False, True, False, False, True, False, False],
        dtype=torch.bool,
        device=device,
    )

    # Clone state
    cloned_state = walker_state.clone(companions, will_clone)

    # Check cloning worked
    assert cloned_state.N == N
    assert torch.equal(cloned_state.rewards[0], walker_state.rewards[1])  # Cloned
    assert torch.equal(cloned_state.rewards[1], walker_state.rewards[1])  # Not cloned
    assert torch.equal(cloned_state.rewards[2], walker_state.rewards[3])  # Cloned
    assert torch.equal(cloned_state.rewards[5], walker_state.rewards[4])  # Cloned


def test_atari_gas_initialization(mock_env, device):
    """Test AtariFractalGas initialization."""
    gas = AtariFractalGas(
        env=mock_env,
        N=20,
        dist_coef=1.0,
        reward_coef=1.0,
        dt_range=(1, 4),
        device=device,
        seed=42,
    )

    assert gas.N == 20
    assert gas.device == device
    assert gas.total_steps == 0
    assert gas.total_clones == 0
    assert gas.clone_op is not None
    assert gas.kinetic_op is not None


def test_atari_gas_reset(mock_env, device):
    """Test environment reset."""
    N = 15
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    state = gas.reset()

    # Check state structure
    assert isinstance(state, WalkerState)
    assert state.N == N
    assert state.observations.shape[0] == N
    assert state.observations.shape[1] == 128  # RAM observations
    assert state.alive.sum() == N  # All alive initially
    assert gas.total_steps == 0


def test_atari_gas_step(mock_env, device):
    """Test single iteration."""
    N = 10
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    state = gas.reset()
    new_state, info = gas.step(state)

    # Check new state
    assert isinstance(new_state, WalkerState)
    assert new_state.N == N
    assert gas.total_steps == N  # One step per walker

    # Check info dict
    assert "iteration" in info
    assert "num_cloned" in info
    assert "alive_count" in info
    assert "mean_reward" in info
    assert "max_reward" in info
    assert info["iteration"] == 1


def test_atari_gas_run(mock_env, device):
    """Test full run loop."""
    N = 10
    max_iterations = 50
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    final_state, history = gas.run(max_iterations=max_iterations)

    # Check completion
    assert isinstance(final_state, WalkerState)
    assert len(history) <= max_iterations
    assert gas.total_steps > 0
    assert gas.iteration_count == len(history)

    # Check history structure
    for info in history:
        assert "iteration" in info
        assert "num_cloned" in info
        assert "alive_count" in info


def test_atari_gas_termination(device):
    """Test early stopping when all walkers dead."""

    class MockEnvAllDead(MockEnv):
        """Environment that immediately terminates all walkers."""

        def step_batch(self, states, actions, dt):
            N = len(states)
            new_states = np.array([self._mock_state() for _ in range(N)], dtype=object)
            observations = np.random.rand(N, *self.obs_shape).astype(np.float32)
            rewards = np.zeros(N, dtype=np.float32)
            dones = np.ones(N, dtype=bool)  # All done
            truncated = np.zeros(N, dtype=bool)
            infos = [{} for _ in range(N)]
            return new_states, observations, rewards, dones, truncated, infos

    env_all_dead = MockEnvAllDead()
    gas = AtariFractalGas(
        env=env_all_dead,
        N=10,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    final_state, history = gas.run(max_iterations=100, stop_when_all_dead=True)

    # Should stop early when all dead
    assert len(history) < 100
    assert not final_state.alive.any()


def test_get_best_walker(mock_env, device):
    """Test best walker selection."""
    N = 10
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    # Create state with known rewards
    state = gas.reset()
    state.rewards = torch.tensor([1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 6.0, 8.0, 0.0], device=device)

    best_idx, best_reward = gas.get_best_walker(state)

    assert best_idx == 3
    assert best_reward == 9.0


def test_virtual_rewards_computed(mock_env, device):
    """Test that virtual rewards are computed and stored."""
    N = 10
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    state = gas.reset()
    new_state, info = gas.step(state)

    # Check virtual rewards were computed
    assert new_state.virtual_rewards is not None
    assert new_state.virtual_rewards.shape == (N,)
    assert not torch.isnan(new_state.virtual_rewards).any()
    assert "mean_virtual_reward" in info


def test_cloning_happens(mock_env, device):
    """Test that cloning events occur."""
    N = 20
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    final_state, history = gas.run(max_iterations=100)

    # Check that some cloning happened
    total_clones = sum(info["num_cloned"] for info in history)
    assert total_clones > 0
    assert gas.total_clones == total_clones


def test_rewards_accumulate(mock_env, device):
    """Test that rewards accumulate over steps."""
    N = 10
    gas = AtariFractalGas(
        env=mock_env,
        N=N,
        dist_coef=1.0,
        reward_coef=1.0,
        device=device,
        seed=42,
    )

    state = gas.reset()

    # Run a few steps
    for _ in range(5):
        state, info = gas.step(state)

    # Rewards should have accumulated (unless all negative)
    # Just check they're not all zero
    assert not torch.allclose(state.rewards, torch.zeros_like(state.rewards))


def test_different_coefficients(mock_env, device):
    """Test with different dist/reward coefficients."""
    N = 10

    gas1 = AtariFractalGas(
        env=mock_env, N=N, dist_coef=2.0, reward_coef=0.5, device=device, seed=42
    )

    gas2 = AtariFractalGas(
        env=mock_env, N=N, dist_coef=0.5, reward_coef=2.0, device=device, seed=42
    )

    # Both should run without errors
    state1, history1 = gas1.run(max_iterations=10)
    state2, history2 = gas2.run(max_iterations=10)

    assert len(history1) == 10
    assert len(history2) == 10
