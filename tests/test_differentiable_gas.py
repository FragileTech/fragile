"""Tests for Differentiable Fragile Gas with Gumbel-softmax cloning.

Tests cover:
1. Gumbel-softmax sampling and properties
2. Companion selection logits (fitness, diversity, combined)
3. Soft cloning operator
4. Full differentiable dynamics
5. Gradient flow through entire rollout
6. Comparison with hard cloning
"""

from __future__ import annotations

import math

import pytest
import torch

from fragile.differentiable_gas import (
    compute_combined_logits,
    compute_diversity_logits,
    compute_fitness_logits,
    create_differentiable_gas_variants,
    DifferentiableGas,
    DifferentiableGasParams,
    gumbel_softmax_sample,
    SwarmState,
)


@pytest.fixture
def device():
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_state(device):
    """Create a simple swarm state for testing."""
    torch.manual_seed(42)
    N, d = 20, 3
    x = torch.randn(N, d, device=device)
    v = torch.randn(N, d, device=device) * 0.1
    reward = torch.randn(N, device=device)
    return SwarmState(x=x, v=v, reward=reward)


# ==================== Gumbel-Softmax Tests ====================


def test_gumbel_softmax_shape(device):
    """Test Gumbel-softmax output shape."""
    logits = torch.randn(5, 10, device=device)

    # Soft samples
    samples = gumbel_softmax_sample(logits, tau=1.0, hard=False)
    assert samples.shape == logits.shape
    assert torch.allclose(samples.sum(dim=-1), torch.ones(5, device=device), atol=1e-5)

    # Hard samples
    samples_hard = gumbel_softmax_sample(logits, tau=1.0, hard=True)
    assert samples_hard.shape == logits.shape
    assert (samples_hard.sum(dim=-1) == 1).all()  # One-hot


def test_gumbel_softmax_temperature(device):
    """Test that temperature controls discreteness."""
    logits = torch.randn(100, 10, device=device)

    # High temperature → more uniform
    samples_hot = gumbel_softmax_sample(logits, tau=10.0, hard=False)
    entropy_hot = -(samples_hot * samples_hot.log()).sum(dim=-1).mean()

    # Low temperature → more discrete
    samples_cold = gumbel_softmax_sample(logits, tau=0.1, hard=False)
    entropy_cold = -(samples_cold * samples_cold.log()).sum(dim=-1).mean()

    assert entropy_hot > entropy_cold


def test_gumbel_softmax_gradient(device):
    """Test that Gumbel-softmax is differentiable."""
    logits = torch.randn(5, 10, device=device, requires_grad=True)

    samples = gumbel_softmax_sample(logits, tau=1.0, hard=False)
    loss = samples.sum()
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


# ==================== Selection Logits Tests ====================


def test_fitness_logits_shape(device):
    """Test fitness logits computation."""
    reward = torch.randn(20, device=device)

    logits = compute_fitness_logits(reward, temperature=1.0)

    assert logits.shape == reward.shape
    assert logits.device.type == device.type


def test_fitness_logits_ordering(device):
    """Test that higher reward → higher logit."""
    reward = torch.tensor([1.0, 5.0, 3.0], device=device)

    logits = compute_fitness_logits(reward, temperature=1.0)

    assert logits[1] > logits[2] > logits[0]


def test_diversity_logits_shape(device):
    """Test diversity logits shape."""
    x = torch.randn(20, 3, device=device)
    x_query = torch.randn(5, 3, device=device)

    logits = compute_diversity_logits(x, x_query, bandwidth=1.0)

    assert logits.shape == (5, 20)
    assert logits.device.type == device.type


def test_diversity_logits_distance(device):
    """Test that farther walkers get higher logits."""
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]], device=device)
    x_query = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    logits = compute_diversity_logits(x, x_query, bandwidth=1.0)

    # Farther walkers should have higher logits
    assert logits[0, 2] > logits[0, 1] > logits[0, 0]


def test_combined_logits_shape(device):
    """Test combined logits shape."""
    x = torch.randn(20, 3, device=device)
    reward = torch.randn(20, device=device)
    x_query = torch.randn(5, 3, device=device)

    logits = compute_combined_logits(
        x, reward, x_query, alpha_fitness=1.0, alpha_diversity=1.0, bandwidth=1.0
    )

    assert logits.shape == (5, 20)


# ==================== DifferentiableGas Tests ====================


def test_gas_initialization(device):
    """Test DifferentiableGas initialization."""
    params = DifferentiableGasParams()
    gas = DifferentiableGas(params, device=device)

    assert gas.device == device
    assert gas.tau == params.tau_init


def test_init_state(device):
    """Test state initialization."""
    params = DifferentiableGasParams()
    gas = DifferentiableGas(params, device=device)

    state = gas.init_state(N=50, d=3, bounds=(-2, 2))

    assert state.x.shape == (50, 3)
    assert state.v.shape == (50, 3)
    assert state.reward.shape == (50,)
    assert state.x.device.type == device.type


def test_compute_companion_weights(device):
    """Test companion weight computation."""
    params = DifferentiableGasParams(selection_mode="combined")
    gas = DifferentiableGas(params, device=device)

    state = gas.init_state(N=30, d=3, bounds=(-2, 2))

    weights = gas.compute_companion_weights(state, num_clones=5)

    assert weights.shape == (5, 30)
    # Each row should sum to 1 (probability distribution)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5, device=device), atol=1e-5)
    assert (weights >= 0).all()


def test_soft_clone(device):
    """Test soft cloning operator."""
    params = DifferentiableGasParams()
    gas = DifferentiableGas(params, device=device)

    state = gas.init_state(N=30, d=3, bounds=(-2, 2))
    weights = torch.softmax(torch.randn(5, 30, device=device), dim=-1)

    x_new, v_new = gas.soft_clone(state, weights)

    assert x_new.shape == (5, 3)
    assert v_new.shape == (5, 3)
    assert x_new.device.type == device.type


def test_langevin_step(device):
    """Test Langevin dynamics without cloning."""

    def sphere_potential(x):
        return (x**2).sum(dim=-1)

    params = DifferentiableGasParams()
    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    state = gas.init_state(N=30, d=3, bounds=(-2, 2))
    x_before = state.x.clone()

    state = gas.langevin_step(state, dt=0.1)

    # State should have changed
    assert not torch.allclose(state.x, x_before)
    assert torch.isfinite(state.x).all()
    assert torch.isfinite(state.v).all()


def test_step_with_cloning(device):
    """Test full step with cloning."""

    def sphere_potential(x):
        return (x**2).sum(dim=-1)

    params = DifferentiableGasParams(clone_rate=0.2)
    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    state = gas.init_state(N=30, d=3, bounds=(-2, 2))
    tau_before = gas.tau

    state = gas.step(state, dt=0.1, do_clone=True)

    # Temperature should have annealed
    assert gas.tau < tau_before

    # Companion weights should be stored
    assert state.companion_weights is not None
    assert state.companion_weights.shape[0] == int(0.2 * 30)


def test_rollout_convergence(device):
    """Test that rollout converges to minimum."""

    def sphere_potential(x):
        return (x**2).sum(dim=-1)

    params = DifferentiableGasParams(clone_rate=0.2, gamma=0.8)
    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    state = gas.init_state(N=50, d=3, bounds=(-5, 5))
    initial_reward = state.reward.mean()

    state, history = gas.rollout(state, T=50, dt=0.1)

    final_reward = state.reward.mean()

    # Reward should improve (for sphere, reward = -potential)
    assert final_reward > initial_reward
    assert len(history) == 50


# ==================== Differentiability Tests ====================


def test_gradient_flow_through_step(device):
    """Test that gradients flow through a single step."""

    def sphere_potential(x):
        return (x**2).sum(dim=-1)

    params = DifferentiableGasParams(clone_rate=0.1)
    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    # Create state with requires_grad
    x = torch.randn(30, 3, device=device, requires_grad=True)
    v = torch.randn(30, 3, device=device)
    reward = -sphere_potential(x)

    state = SwarmState(x=x, v=v, reward=reward)

    # Step
    state_new = gas.step(state, dt=0.1)

    # Compute loss
    loss = -state_new.reward.mean()
    loss.backward()

    # Gradients should have flowed back to initial positions
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_gradient_flow_through_rollout(device):
    """Test that gradients flow through entire rollout."""

    def sphere_potential(x):
        return (x**2).sum(dim=-1)

    params = DifferentiableGasParams(
        clone_rate=0.1,
        tau_init=1.0,
        tau_anneal=0.99,  # Slower annealing for stability
    )
    gas = DifferentiableGas(params, potential=sphere_potential, device=device)

    # Initialize with requires_grad
    x_init = torch.randn(20, 3, device=device, requires_grad=True)
    v_init = torch.randn(20, 3, device=device)
    reward_init = -sphere_potential(x_init)

    state = SwarmState(x=x_init, v=v_init, reward=reward_init)

    # Rollout
    state_final, _history = gas.rollout(state, T=10, dt=0.1)

    # Loss on final state
    loss = -state_final.reward.mean()
    loss.backward()

    # Should have gradients on initial positions
    assert x_init.grad is not None
    assert torch.isfinite(x_init.grad).all()


def test_meta_optimization_temperature(device):
    """Test meta-optimization of temperature parameter."""

    def rastrigin(x):
        """Rastrigin function (many local minima)."""
        A = 10
        d = x.shape[-1]
        pi = torch.tensor(math.pi, device=x.device, dtype=x.dtype)
        return A * d + (x**2 - A * torch.cos(2 * pi * x)).sum(dim=-1)

    # Learnable temperature
    tau_init_param = torch.tensor(2.0, device=device, requires_grad=True)

    params = DifferentiableGasParams(clone_rate=0.15)
    gas = DifferentiableGas(params, potential=rastrigin, device=device)

    # Override tau with learnable parameter
    gas.tau = tau_init_param

    state = gas.init_state(N=30, d=2, bounds=(-5, 5))

    # Short rollout
    for _ in range(5):
        state = gas.step(state, dt=0.1)

    # Loss: negative mean reward
    loss = -state.reward.mean()
    loss.backward()

    # Should have gradient on temperature
    assert tau_init_param.grad is not None
    assert torch.isfinite(tau_init_param.grad)


# ==================== Variant Tests ====================


def test_create_variants():
    """Test variant creation."""
    variants = create_differentiable_gas_variants()

    assert len(variants) == 4
    assert "fitness_only" in variants
    assert "diversity_only" in variants
    assert "balanced" in variants
    assert "fitness_heavy" in variants

    # Check configurations
    assert variants["fitness_only"].selection_mode == "fitness"
    assert variants["diversity_only"].selection_mode == "diversity"
    assert variants["balanced"].selection_mode == "combined"
    assert variants["fitness_heavy"].alpha_fitness == 2.0


def test_fitness_only_variant(device):
    """Test fitness-only selection."""
    variants = create_differentiable_gas_variants(device=device)
    params = variants["fitness_only"]

    gas = DifferentiableGas(params, device=device)
    state = gas.init_state(N=30, d=3, bounds=(-2, 2))

    # Should work
    state = gas.step(state, dt=0.1)
    assert torch.isfinite(state.x).all()


def test_diversity_only_variant(device):
    """Test diversity-only selection."""
    variants = create_differentiable_gas_variants(device=device)
    params = variants["diversity_only"]

    gas = DifferentiableGas(params, device=device)
    state = gas.init_state(N=30, d=3, bounds=(-2, 2))

    state = gas.step(state, dt=0.1)
    assert torch.isfinite(state.x).all()


# ==================== Edge Cases ====================


def test_single_walker(device):
    """Test with only one walker."""
    params = DifferentiableGasParams(clone_rate=0.5, elite_fraction=1.0)
    gas = DifferentiableGas(params, device=device)

    state = gas.init_state(N=1, d=3, bounds=(-2, 2))

    # Should handle gracefully
    state = gas.step(state, dt=0.1)
    assert state.x.shape == (1, 3)


def test_no_cloning(device):
    """Test with cloning disabled."""
    params = DifferentiableGasParams(clone_rate=0.0)
    gas = DifferentiableGas(params, device=device)

    state = gas.init_state(N=20, d=3, bounds=(-2, 2))

    state = gas.step(state, dt=0.1, do_clone=True)

    # Companion weights should not be computed
    assert state.companion_weights is None


def test_temperature_annealing(device):
    """Test temperature annealing schedule."""
    params = DifferentiableGasParams(tau_init=2.0, tau_min=0.1, tau_anneal=0.9)
    gas = DifferentiableGas(params, device=device)

    assert gas.tau == 2.0

    state = gas.init_state(N=20, d=3, bounds=(-2, 2))

    for _ in range(10):
        gas.step(state, dt=0.1)

    # Should have annealed
    assert gas.tau < 2.0
    assert gas.tau >= params.tau_min


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
