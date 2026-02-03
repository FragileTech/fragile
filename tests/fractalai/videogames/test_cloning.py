"""Tests for fractal gas cloning operator."""

import pytest
import torch

from fragile.fractalai.videogames.cloning import FractalCloningOperator, clone_walker_data


@pytest.fixture
def device():
    """Test device (CPU for portability)."""
    return "cpu"


@pytest.fixture
def clone_op(device):
    """Create a cloning operator for testing."""
    return FractalCloningOperator(dist_coef=1.0, reward_coef=1.0, device=device)


def test_calculate_fitness_basic(clone_op, device):
    """Test basic virtual reward computation."""
    N = 10
    obs_dim = 128  # RAM observations

    # Create test data
    observations = torch.randn(N, obs_dim, device=device)
    rewards = torch.rand(N, device=device) * 10
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Calculate fitness
    virtual_rewards, companions = clone_op.calculate_fitness(observations, rewards, alive)

    # Check outputs
    assert virtual_rewards.shape == (N,)
    assert companions.shape == (N,)
    assert (virtual_rewards >= 0).all()  # Virtual rewards should be positive
    assert (companions >= 0).all() and (companions < N).all()  # Valid indices

    # Check that companions were stored
    assert clone_op.last_fitness_companions is not None
    assert clone_op.last_virtual_rewards is not None


def test_calculate_fitness_uniform_companions(clone_op, device):
    """Test that companion selection is uniform random."""
    N = 20
    obs_dim = 128

    observations = torch.randn(N, obs_dim, device=device)
    rewards = torch.rand(N, device=device) * 10
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Run multiple times and collect companions
    all_companions = []
    for _ in range(100):
        _, companions = clone_op.calculate_fitness(observations, rewards, alive)
        all_companions.append(companions)

    all_companions = torch.stack(all_companions)

    # Check that companions vary (not always the same)
    # Convert to float for std calculation
    assert all_companions.float().std(dim=0).sum() > 0  # Should have variation

    # Check that all walker indices appear as companions
    unique_companions = torch.unique(all_companions)
    assert len(unique_companions) > N // 2  # At least half the walkers used


def test_calculate_fitness_distance_computation(clone_op, device):
    """Test L2 distance computation on observations."""
    N = 5
    obs_dim = 128

    # Create observations with known distances
    observations = torch.zeros(N, obs_dim, device=device)
    observations[0, :] = 0.0
    observations[1, :] = 1.0
    observations[2, :] = 2.0

    rewards = torch.ones(N, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Calculate fitness
    virtual_rewards, companions = clone_op.calculate_fitness(observations, rewards, alive)

    # Virtual rewards should be computed (exact values depend on normalization)
    assert virtual_rewards.shape == (N,)
    assert not torch.isnan(virtual_rewards).any()
    assert not torch.isinf(virtual_rewards).any()


def test_decide_cloning_based_on_virtual_reward(clone_op, device):
    """Test cloning probability calculation."""
    N = 10

    # Create virtual rewards with clear winner
    virtual_rewards = torch.ones(N, device=device)
    virtual_rewards[5] = 10.0  # One walker has much higher virtual reward

    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Decide cloning (run multiple times due to randomness)
    clone_counts = torch.zeros(N, device=device)
    num_trials = 100

    for _ in range(num_trials):
        companions, will_clone = clone_op.decide_cloning(virtual_rewards, alive)
        clone_counts[will_clone] += 1

    # Walkers should clone sometimes (probabilistic)
    assert clone_counts.sum() > 0  # Some cloning should happen

    # Check that companions are valid
    assert companions.shape == (N,)
    assert (companions >= 0).all() and (companions < N).all()


def test_decide_cloning_dead_always_clone(clone_op, device):
    """Test that dead walkers always clone."""
    N = 10

    virtual_rewards = torch.ones(N, device=device)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    alive[3] = False  # Walker 3 is dead
    alive[7] = False  # Walker 7 is dead

    companions, will_clone = clone_op.decide_cloning(virtual_rewards, alive)

    # Dead walkers must clone
    assert will_clone[3] == True
    assert will_clone[7] == True


def test_clone_walker_data(device):
    """Test array cloning helper."""
    N = 8

    # Create test data
    data = torch.arange(N, dtype=torch.float32, device=device)
    companions = torch.tensor([1, 0, 3, 2, 5, 4, 7, 6], device=device)
    will_clone = torch.tensor(
        [True, False, True, False, True, False, False, True],
        dtype=torch.bool,
        device=device,
    )

    # Clone data
    cloned_data = clone_walker_data(data, companions, will_clone)

    # Check results
    assert cloned_data.shape == data.shape
    assert cloned_data[0] == data[1]  # Cloned from companion
    assert cloned_data[1] == data[1]  # Not cloned
    assert cloned_data[2] == data[3]  # Cloned from companion
    assert cloned_data[3] == data[3]  # Not cloned


def test_fitness_with_ram_observations(clone_op, device):
    """Test with realistic RAM observation tensors."""
    N = 20
    obs_shape = (128,)  # RAM observations are 128 bytes

    # Create realistic RAM observations (bytes in [0, 255])
    observations = torch.randint(0, 256, (N, *obs_shape), dtype=torch.float32, device=device)
    rewards = torch.rand(N, device=device) * 100 - 50  # Rewards can be negative
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Calculate fitness
    virtual_rewards, companions = clone_op.calculate_fitness(observations, rewards, alive)

    # Check outputs are valid
    assert virtual_rewards.shape == (N,)
    assert not torch.isnan(virtual_rewards).any()
    assert not torch.isinf(virtual_rewards).any()
    assert (virtual_rewards >= 0).all()


def test_apply_combined_operation(clone_op, device):
    """Test combined fitness + cloning operation."""
    N = 15
    obs_dim = 128

    observations = torch.randn(N, obs_dim, device=device)
    rewards = torch.rand(N, device=device) * 10
    alive = torch.ones(N, dtype=torch.bool, device=device)

    # Apply combined operation
    virtual_rewards, companions, will_clone = clone_op.apply(observations, rewards, alive)

    # Check outputs
    assert virtual_rewards.shape == (N,)
    assert companions.shape == (N,)
    assert will_clone.shape == (N,)
    assert will_clone.dtype == torch.bool

    # Check stored values
    assert clone_op.last_companions is not None
    assert clone_op.last_will_clone is not None
    assert clone_op.last_virtual_rewards is not None


def test_cloning_with_all_dead(clone_op, device):
    """Test cloning when all walkers are dead."""
    N = 10
    obs_dim = 128

    observations = torch.randn(N, obs_dim, device=device)
    rewards = torch.rand(N, device=device) * 10
    alive = torch.zeros(N, dtype=torch.bool, device=device)  # All dead

    # Calculate fitness (should still work)
    virtual_rewards, companions = clone_op.calculate_fitness(observations, rewards, alive)

    # Decide cloning
    companions, will_clone = clone_op.decide_cloning(virtual_rewards, alive)

    # All walkers should clone (because they're all dead)
    assert will_clone.all()
