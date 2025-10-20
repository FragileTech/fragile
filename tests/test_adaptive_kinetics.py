"""Tests for adaptive Langevin dynamics in KineticOperator.

Tests cover:
1. Backward compatibility with standard BAOAB
2. Fitness-based force
3. Combined potential + fitness forces
4. Diagonal anisotropic diffusion
5. Full anisotropic diffusion
6. Parameter validation
7. Integration test with realistic swarm

Note: KineticOperator now receives precomputed fitness gradients/Hessians
      instead of owning a FitnessOperator.
"""

import pytest
import torch

from fragile.core.fitness import FitnessOperator, FitnessParams
from fragile.core.kinetics import KineticOperator, LangevinParams


class SimpleQuadraticPotential:
    """Simple quadratic potential U(x) = 0.5 * ||x||^2."""

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate U(x) = 0.5 * ||x||^2."""
        return 0.5 * torch.sum(x**2, dim=-1)


class SwarmState:
    """Minimal SwarmState for testing."""

    def __init__(self, x, v):
        self.x = x
        self.v = v

    @property
    def N(self):
        return self.x.shape[0]

    @property
    def d(self):
        return self.x.shape[1]


@pytest.fixture
def device():
    """Test device."""
    return torch.device("cpu")


@pytest.fixture
def dtype():
    """Test dtype."""
    return torch.float32


@pytest.fixture
def simple_state(device, dtype):
    """Simple 3-walker, 2D state for testing."""
    N, d = 3, 2
    x = torch.randn(N, d, device=device, dtype=dtype)
    v = torch.randn(N, d, device=device, dtype=dtype)
    return SwarmState(x, v)


@pytest.fixture
def simple_rewards_alive(device, dtype):
    """Simple rewards and alive mask with companions."""
    N = 3
    rewards = torch.randn(N, device=device, dtype=dtype)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    # Use circular pairing to avoid self-pairing
    companions = torch.tensor([1, 2, 0], dtype=torch.int64, device=device)
    return rewards, alive, companions


@pytest.fixture
def potential():
    """Simple quadratic potential."""
    return SimpleQuadraticPotential()


@pytest.fixture
def fitness_operator():
    """Fitness operator with default parameters."""
    return FitnessOperator()


# =============================================================================
# Test 1: Backward Compatibility
# =============================================================================


def test_backward_compatibility_standard_baoab(device, dtype, potential, simple_state):
    """Test that standard BAOAB is unchanged when adaptive features disabled."""
    # Standard parameters (all adaptive features off)
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.0,
        use_fitness_force=False,
        use_potential_force=True,
        use_anisotropic_diffusion=False,
    )

    # Create kinetic operator (no fitness operator needed)
    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Apply one step (no fitness derivatives needed)
    state_new = kinetic.apply(simple_state)

    # Verify state structure
    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape

    # Verify state changed (not frozen)
    assert not torch.allclose(state_new.x, simple_state.x)
    assert not torch.allclose(state_new.v, simple_state.v)


def test_no_potential_no_fitness_pure_ou(device, dtype, simple_state):
    """Test pure Ornstein-Uhlenbeck (no forces)."""
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_fitness_force=False,
        use_potential_force=False,  # No potential force
    )

    # No potential or fitness operator
    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=None,
        device=device,
        dtype=dtype,
    )

    # Should work (pure OU dynamics)
    state_new = kinetic.apply(simple_state)

    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape


# =============================================================================
# Test 2: Parameter Validation
# =============================================================================


def test_validation_potential_force_requires_potential(device, dtype):
    """Test that use_potential_force=True requires potential."""
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_potential_force=True,  # Requires potential
    )

    with pytest.raises(ValueError, match="potential required"):
        KineticOperator(
            gamma=params.gamma,
            beta=params.beta,
            delta_t=params.delta_t,
            integrator=params.integrator,
            epsilon_F=params.epsilon_F,
            use_fitness_force=params.use_fitness_force,
            use_potential_force=params.use_potential_force,
            epsilon_Sigma=params.epsilon_Sigma,
            use_anisotropic_diffusion=params.use_anisotropic_diffusion,
            diagonal_diffusion=params.diagonal_diffusion,
            potential=None,
            device=device,
            dtype=dtype,
        )


def test_validation_fitness_force_requires_gradient(device, dtype, potential, simple_state):
    """Test that use_fitness_force=True requires grad_fitness in apply()."""
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_fitness_force=True,
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Missing grad_fitness should raise error
    with pytest.raises(ValueError, match="grad_fitness required"):
        kinetic.apply(simple_state, grad_fitness=None)


def test_validation_anisotropic_diffusion_requires_hessian(device, dtype, potential, simple_state):
    """Test that use_anisotropic_diffusion=True requires hess_fitness in apply()."""
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_anisotropic_diffusion=True,
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Missing hess_fitness should raise error
    with pytest.raises(ValueError, match="hess_fitness required"):
        kinetic.apply(simple_state, hess_fitness=None)


# =============================================================================
# Test 3: Fitness Force Only
# =============================================================================


def test_fitness_force_only(device, dtype, fitness_operator, simple_state, simple_rewards_alive):
    """Test kinetics with fitness force only (no potential)."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,  # Small adaptation rate
        use_fitness_force=True,
        use_potential_force=False,  # Disable potential
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=None,
        device=device,
        dtype=dtype,
    )

    # Compute fitness gradient
    grad_fitness = fitness_operator.compute_gradient(
        simple_state.x, simple_state.v, rewards, alive, companions
    )

    # Apply step with fitness force
    state_new = kinetic.apply(simple_state, grad_fitness=grad_fitness)

    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape
    assert not torch.allclose(state_new.x, simple_state.x)


# =============================================================================
# Test 4: Combined Potential + Fitness Forces
# =============================================================================


def test_combined_forces(
    device, dtype, potential, fitness_operator, simple_state, simple_rewards_alive
):
    """Test kinetics with both potential and fitness forces."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,
        use_fitness_force=True,
        use_potential_force=True,  # Both forces enabled
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Compute fitness gradient
    grad_fitness = fitness_operator.compute_gradient(
        simple_state.x, simple_state.v, rewards, alive, companions
    )

    # Apply step with combined forces
    state_new = kinetic.apply(simple_state, grad_fitness=grad_fitness)

    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape


# =============================================================================
# Test 5: Diagonal Anisotropic Diffusion
# =============================================================================


def test_diagonal_anisotropic_diffusion(
    device, dtype, potential, fitness_operator, simple_state, simple_rewards_alive
):
    """Test diagonal Hessian-based anisotropic diffusion."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_Sigma=0.1,  # Regularization
        use_anisotropic_diffusion=True,
        diagonal_diffusion=True,  # Diagonal only
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Compute diagonal Hessian
    hess_fitness = fitness_operator.compute_hessian(
        simple_state.x, simple_state.v, rewards, alive, companions, diagonal_only=True
    )

    # Apply step with diagonal diffusion
    state_new = kinetic.apply(simple_state, hess_fitness=hess_fitness)

    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape
    assert not torch.allclose(state_new.v, simple_state.v)  # Velocity changed


# =============================================================================
# Test 6: Full Anisotropic Diffusion
# =============================================================================


def test_full_anisotropic_diffusion(
    device, dtype, potential, fitness_operator, simple_state, simple_rewards_alive
):
    """Test full Hessian-based anisotropic diffusion."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_Sigma=0.1,
        use_anisotropic_diffusion=True,
        diagonal_diffusion=False,  # Full anisotropic
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Compute full Hessian
    hess_fitness = fitness_operator.compute_hessian(
        simple_state.x, simple_state.v, rewards, alive, companions, diagonal_only=False
    )

    # Apply step with full diffusion
    state_new = kinetic.apply(simple_state, hess_fitness=hess_fitness)

    assert state_new.x.shape == simple_state.x.shape
    assert state_new.v.shape == simple_state.v.shape


# =============================================================================
# Test 7: Integration Test with All Features
# =============================================================================


def test_full_integration(device, dtype, potential, fitness_operator):
    """Integration test with all adaptive features enabled."""
    N, d = 10, 3
    x = torch.randn(N, d, device=device, dtype=dtype)
    v = torch.randn(N, d, device=device, dtype=dtype)
    rewards = torch.randn(N, device=device, dtype=dtype)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    alive[3:5] = False  # Some dead walkers
    # Circular pairing
    companions = torch.roll(torch.arange(N, dtype=torch.int64), 1)

    state = SwarmState(x, v)

    # Full adaptive Geometric Gas configuration
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,
        epsilon_Sigma=0.1,
        use_fitness_force=True,
        use_potential_force=True,
        use_anisotropic_diffusion=True,
        diagonal_diffusion=True,  # Use diagonal for speed
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    # Run 10 steps
    for _ in range(10):
        # Compute fitness derivatives
        grad_fitness = fitness_operator.compute_gradient(
            state.x, state.v, rewards, alive, companions
        )
        hess_fitness = fitness_operator.compute_hessian(
            state.x, state.v, rewards, alive, companions, diagonal_only=True
        )

        # Apply kinetic step
        state = kinetic.apply(state, grad_fitness=grad_fitness, hess_fitness=hess_fitness)

    # Verify state evolved
    assert state.x.shape == (N, d)
    assert state.v.shape == (N, d)
    assert not torch.allclose(state.x, x)  # Positions changed
    assert not torch.allclose(state.v, v)  # Velocities changed


# =============================================================================
# Test 8: Diffusion Tensor Properties
# =============================================================================


def test_diffusion_tensor_positive_definite(
    device, dtype, fitness_operator, simple_state, simple_rewards_alive
):
    """Test that regularized Hessian diffusion tensor is positive definite."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_Sigma=0.1,  # Regularization ensures positive definiteness
        use_anisotropic_diffusion=True,
        diagonal_diffusion=False,  # Test full tensor
        use_potential_force=False,  # No potential force (testing diffusion only)
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=None,
        device=device,
        dtype=dtype,
    )

    # Compute full Hessian
    hess_fitness = fitness_operator.compute_hessian(
        simple_state.x, simple_state.v, rewards, alive, companions, diagonal_only=False
    )

    # Compute diffusion tensor
    sigma = kinetic._compute_diffusion_tensor(simple_state.x, hess_fitness)

    # Check shape
    N, d = simple_state.N, simple_state.d
    assert sigma.shape == (N, d, d)

    # Check positive definiteness via eigenvalues
    for i in range(N):
        eigenvalues = torch.linalg.eigvalsh(sigma[i])
        assert torch.all(eigenvalues > 0), f"Walker {i} diffusion tensor not positive definite"


def test_diffusion_tensor_symmetric(
    device, dtype, fitness_operator, simple_state, simple_rewards_alive
):
    """Test that diffusion tensor is symmetric."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_Sigma=0.1,
        use_anisotropic_diffusion=True,
        diagonal_diffusion=False,
        use_potential_force=False,  # No potential force (testing diffusion only)
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=None,
        device=device,
        dtype=dtype,
    )

    # Compute full Hessian
    hess_fitness = fitness_operator.compute_hessian(
        simple_state.x, simple_state.v, rewards, alive, companions, diagonal_only=False
    )

    # Compute diffusion tensor
    sigma = kinetic._compute_diffusion_tensor(simple_state.x, hess_fitness)

    # Check symmetry for each walker
    for i in range(simple_state.N):
        assert torch.allclose(
            sigma[i], sigma[i].T, rtol=1e-5, atol=1e-6
        ), f"Walker {i} diffusion tensor not symmetric"


# =============================================================================
# Test 9: Force Computation
# =============================================================================


def test_force_computation_potential_only(device, dtype, potential, simple_state):
    """Test force computation with potential only."""
    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_potential_force=True,
        use_fitness_force=False,
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=potential,
        device=device,
        dtype=dtype,
    )

    force = kinetic._compute_force(simple_state.x, simple_state.v, grad_fitness=None)

    # Force should oppose position (quadratic potential)
    # F = -∇U = -x
    expected_force = -simple_state.x
    assert torch.allclose(force, expected_force, rtol=1e-5, atol=1e-6)


def test_force_computation_fitness_only(
    device, dtype, fitness_operator, simple_state, simple_rewards_alive
):
    """Test force computation with fitness force only."""
    rewards, alive, companions = simple_rewards_alive

    params = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,
        use_potential_force=False,
        use_fitness_force=True,
    )

    kinetic = KineticOperator(
        gamma=params.gamma,
        beta=params.beta,
        delta_t=params.delta_t,
        integrator=params.integrator,
        epsilon_F=params.epsilon_F,
        use_fitness_force=params.use_fitness_force,
        use_potential_force=params.use_potential_force,
        epsilon_Sigma=params.epsilon_Sigma,
        use_anisotropic_diffusion=params.use_anisotropic_diffusion,
        diagonal_diffusion=params.diagonal_diffusion,
        potential=None,
        device=device,
        dtype=dtype,
    )

    # Compute fitness gradient
    grad_fitness = fitness_operator.compute_gradient(
        simple_state.x, simple_state.v, rewards, alive, companions
    )

    force = kinetic._compute_force(simple_state.x, simple_state.v, grad_fitness=grad_fitness)

    # Force should have correct shape
    assert force.shape == simple_state.x.shape

    # Force should be non-zero (fitness gradient exists)
    assert not torch.allclose(force, torch.zeros_like(force))


def test_force_computation_combined(
    device, dtype, potential, fitness_operator, simple_state, simple_rewards_alive
):
    """Test force computation with both potential and fitness."""
    rewards, alive, companions = simple_rewards_alive

    params_combined = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,
        use_potential_force=True,
        use_fitness_force=True,
    )

    kinetic_combined = KineticOperator(
        params_combined, potential=potential, device=device, dtype=dtype
    )

    # Compute gradient
    grad_fitness = fitness_operator.compute_gradient(
        simple_state.x, simple_state.v, rewards, alive, companions
    )

    # Compute combined force
    force_combined = kinetic_combined._compute_force(
        simple_state.x, simple_state.v, grad_fitness=grad_fitness
    )

    # Compute individual forces

    kinetic_pot = KineticOperator(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        use_fitness_force=False,
        potential=potential,
        device=device,
        dtype=dtype,
    )
    force_pot = kinetic_pot._compute_force(simple_state.x, simple_state.v, grad_fitness=None)

    params_fit = LangevinParams(
        gamma=1.0,
        beta=1.0,
        delta_t=0.01,
        epsilon_F=0.1,
        use_potential_force=False,
        use_fitness_force=True,
    )
    kinetic_fit = KineticOperator(params_fit, potential=None, device=device, dtype=dtype)
    force_fit = kinetic_fit._compute_force(
        simple_state.x, simple_state.v, grad_fitness=grad_fitness
    )

    # Combined force should be sum of individual forces
    expected_combined = force_pot + force_fit
    assert torch.allclose(force_combined, expected_combined, rtol=1e-5, atol=1e-6)
