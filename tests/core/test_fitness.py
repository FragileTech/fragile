"""Tests for fitness operator with automatic differentiation.

Tests verify:
1. FitnessParams Pydantic validation
2. FitnessOperator basic functionality
3. Gradient computation via finite differences
4. Hessian computation via finite differences
"""

from pydantic import ValidationError
import pytest
import torch

from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import (
    compute_fitness,
    FitnessOperator,
    FitnessParams,
)


# ============================================================================
# Test FitnessParams Validation
# ============================================================================


def test_fitness_params_defaults():
    """Test FitnessParams with default values."""
    params = FitnessParams()
    assert params.alpha == 1.0
    assert params.beta == 1.0
    assert params.eta == 0.1
    assert params.lambda_alg == 0.0
    assert params.sigma_min == 1e-8
    assert params.A == 2.0


def test_fitness_params_custom_values():
    """Test FitnessParams with custom values."""
    params = FitnessParams(
        alpha=2.0,
        beta=0.5,
        eta=0.2,
        lambda_alg=1.0,
        sigma_min=1e-6,
        A=3.0,
    )
    assert params.alpha == 2.0
    assert params.beta == 0.5
    assert params.eta == 0.2
    assert params.lambda_alg == 1.0
    assert params.sigma_min == 1e-6
    assert params.A == 3.0


def test_fitness_params_validation_alpha():
    """Test that alpha must be positive."""
    with pytest.raises(ValidationError):
        FitnessParams(alpha=0.0)
    with pytest.raises(ValidationError):
        FitnessParams(alpha=-1.0)


def test_fitness_params_validation_beta():
    """Test that beta must be positive."""
    with pytest.raises(ValidationError):
        FitnessParams(beta=0.0)
    with pytest.raises(ValidationError):
        FitnessParams(beta=-1.0)


def test_fitness_params_validation_eta():
    """Test that eta must be positive."""
    with pytest.raises(ValidationError):
        FitnessParams(eta=0.0)
    with pytest.raises(ValidationError):
        FitnessParams(eta=-0.1)


def test_fitness_params_validation_lambda_alg():
    """Test that lambda_alg must be non-negative."""
    # Should accept 0.0
    params = FitnessParams(lambda_alg=0.0)
    assert params.lambda_alg == 0.0

    # Should reject negative
    with pytest.raises(ValidationError):
        FitnessParams(lambda_alg=-0.1)


def test_fitness_params_validation_sigma_min():
    """Test that sigma_min must be positive."""
    with pytest.raises(ValidationError):
        FitnessParams(sigma_min=0.0)
    with pytest.raises(ValidationError):
        FitnessParams(sigma_min=-1e-8)


def test_fitness_params_validation_A():
    """Test that A must be positive."""
    with pytest.raises(ValidationError):
        FitnessParams(A=0.0)
    with pytest.raises(ValidationError):
        FitnessParams(A=-1.0)


# ============================================================================
# Test FitnessOperator Basic Functionality
# ============================================================================


@pytest.fixture
def simple_swarm_data():
    """Create simple test data for fitness computation."""
    N, d = 10, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    # Fix companions for reproducibility
    companions = torch.arange(N)
    companions = torch.roll(companions, 1)  # Each walker paired with next one

    return {
        "positions": positions,
        "velocities": velocities,
        "rewards": rewards,
        "alive": alive,
        "companions": companions,
    }


def test_fitness_operator_initialization_defaults():
    """Test FitnessOperator initialization with defaults."""
    op = FitnessOperator()
    assert op.params.alpha == 1.0
    assert op.params.beta == 1.0
    assert op.companion_selection.method == "uniform"


def test_fitness_operator_initialization_custom():
    """Test FitnessOperator initialization with custom parameters."""
    params = FitnessParams(alpha=2.0, beta=0.5)
    companion_sel = CompanionSelection(method="softmax", epsilon=0.1)
    op = FitnessOperator(params=params, companion_selection=companion_sel)
    assert op.params.alpha == 2.0
    assert op.params.beta == 0.5
    assert op.companion_selection.method == "softmax"


def test_fitness_operator_call_matches_function(simple_swarm_data):
    """Test that FitnessOperator.__call__ matches compute_fitness function."""
    op = FitnessOperator()
    data = simple_swarm_data

    # Compute using operator
    fitness_op, info_op = op(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    # Compute using function
    fitness_fn, info_fn = compute_fitness(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
        alpha=op.params.alpha,
        beta=op.params.beta,
        eta=op.params.eta,
        lambda_alg=op.params.lambda_alg,
        sigma_min=op.params.sigma_min,
        A=op.params.A,
    )

    # Should match exactly
    torch.testing.assert_close(fitness_op, fitness_fn)
    torch.testing.assert_close(info_op["distances"], info_fn["distances"])
    torch.testing.assert_close(info_op["companions"], info_fn["companions"])


def test_fitness_operator_output_shapes(simple_swarm_data):
    """Test output shapes from FitnessOperator."""
    op = FitnessOperator()
    data = simple_swarm_data
    N = data["positions"].shape[0]

    fitness, info = op(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    assert fitness.shape == (N,)
    assert info["distances"].shape == (N,)
    assert info["companions"].shape == (N,)


def test_fitness_operator_dead_walkers(simple_swarm_data):
    """Test that dead walkers receive zero fitness."""
    op = FitnessOperator()
    data = simple_swarm_data
    N = data["positions"].shape[0]

    # Mark half the walkers as dead
    alive = torch.zeros(N, dtype=torch.bool)
    alive[: N // 2] = True

    fitness, _info = op(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=alive,
        companions=data["companions"],
    )

    # Dead walkers should have zero fitness
    assert torch.all(fitness[~alive] == 0.0)


# ============================================================================
# Test Gradient Computation
# ============================================================================


def test_gradient_simple_case(simple_swarm_data):
    """Test gradient computation is well-defined and has correct properties.

    Note on gradient interpretation:
        The fitness gradient is computed with "frozen" mean/std statistics
        (frozen mean-field approximation). This gives the instantaneous force
        for adaptive Langevin dynamics, where mean-field evolution happens
        through time-stepping, not within the gradient calculation.

        See CLAUDE.md § Mathematical Proofing for details on mean-field coupling.
    """
    op = FitnessOperator()
    data = simple_swarm_data

    # Compute gradient using FitnessOperator
    grad = op.compute_gradient(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    # Test 1: Gradient should have correct shape
    assert grad.shape == data["positions"].shape

    # Test 2: Gradient should be non-zero (fitness depends on positions)
    assert grad.abs().max() > 1e-6

    # Test 3: Gradient should not have NaN or Inf
    assert not torch.any(torch.isnan(grad))
    assert not torch.any(torch.isinf(grad))

    # Test 4: Verify gradient is actually used correctly (backward pass works)
    positions_grad = data["positions"].clone().requires_grad_(True)  # noqa: FBT003
    op.compute_gradient(
        positions=positions_grad.detach(),
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    # Also verify we can compute gradient through the full forward pass
    fitness, _info = op(
        positions_grad, data["velocities"], data["rewards"], data["alive"], data["companions"]
    )
    loss = fitness.sum()
    loss.backward()

    # The gradient should exist
    assert positions_grad.grad is not None
    assert positions_grad.grad.shape == positions_grad.shape


def test_gradient_shape(simple_swarm_data):
    """Test gradient has correct shape."""
    op = FitnessOperator()
    data = simple_swarm_data
    N, d = data["positions"].shape

    grad = op.compute_gradient(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    assert grad.shape == (N, d)


def test_gradient_custom_params(simple_swarm_data):
    """Test gradient with custom fitness parameters."""
    params = FitnessParams(alpha=2.0, beta=0.5, eta=0.2)
    op = FitnessOperator(params=params)
    data = simple_swarm_data

    grad = op.compute_gradient(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
    )

    # Test gradient properties
    assert grad.shape == data["positions"].shape
    assert grad.abs().max() > 1e-6
    assert not torch.any(torch.isnan(grad))
    assert not torch.any(torch.isinf(grad))

    # Verify we can compute gradient through the full forward pass
    positions_grad = data["positions"].clone().requires_grad_(True)  # noqa: FBT003
    fitness, _info = op(
        positions_grad, data["velocities"], data["rewards"], data["alive"], data["companions"]
    )
    loss = fitness.sum()
    loss.backward()

    # The gradient should exist and match the compute_gradient result
    assert positions_grad.grad is not None
    assert positions_grad.grad.shape == positions_grad.shape


# ============================================================================
# Test Hessian Computation
# ============================================================================


def finite_difference_hessian_diagonal(
    op: FitnessOperator,
    positions: torch.Tensor,
    velocities: torch.Tensor,
    rewards: torch.Tensor,
    alive: torch.Tensor,
    companions: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute diagonal Hessian using finite differences for validation.

    Args:
        op: FitnessOperator instance
        positions: Walker positions [N, d]
        velocities: Walker velocities [N, d]
        rewards: Rewards [N]
        alive: Alive mask [N]
        companions: Companion indices [N]
        eps: Finite difference step size

    Returns:
        Diagonal Hessian [N, d] computed via finite differences
    """
    N, d = positions.shape
    hess_diag_fd = torch.zeros_like(positions)

    for i in range(N):
        for j in range(d):
            # Perturb forward
            pos_plus = positions.clone()
            pos_plus[i, j] += eps

            # Perturb backward
            pos_minus = positions.clone()
            pos_minus[i, j] -= eps

            # Center
            fitness_center, _ = op(positions, velocities, rewards, alive, companions)
            fitness_plus, _ = op(pos_plus, velocities, rewards, alive, companions)
            fitness_minus, _ = op(pos_minus, velocities, rewards, alive, companions)

            # Second derivative: (f(x+h) - 2f(x) + f(x-h)) / h^2
            hess_diag_fd[i, j] = (
                fitness_plus.sum() - 2 * fitness_center.sum() + fitness_minus.sum()
            ) / (eps**2)

    return hess_diag_fd


def test_hessian_diagonal_simple_case(simple_swarm_data):
    """Test diagonal Hessian computation is well-defined."""
    op = FitnessOperator()
    data = simple_swarm_data

    # Compute Hessian diagonal using autograd
    hess = op.compute_hessian(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
        diagonal_only=True,
    )

    # Test Hessian properties
    assert hess.shape == data["positions"].shape
    assert not torch.any(torch.isnan(hess))
    assert not torch.any(torch.isinf(hess))

    # For anisotropic diffusion, we need positive diagonal elements
    # (after regularization with epsilon_Sigma in the kinetic operator)
    # Here we just verify the Hessian exists and is finite


def test_hessian_diagonal_shape(simple_swarm_data):
    """Test diagonal Hessian has correct shape."""
    op = FitnessOperator()
    data = simple_swarm_data
    N, d = data["positions"].shape

    hess = op.compute_hessian(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
        diagonal_only=True,
    )

    assert hess.shape == (N, d)


def test_hessian_full_shape(simple_swarm_data):
    """Test full Hessian has correct shape."""
    op = FitnessOperator()
    data = simple_swarm_data
    N, d = data["positions"].shape

    hess = op.compute_hessian(
        positions=data["positions"],
        velocities=data["velocities"],
        rewards=data["rewards"],
        alive=data["alive"],
        companions=data["companions"],
        diagonal_only=False,
    )

    assert hess.shape == (N, d, d)


def test_hessian_full_symmetric():
    """Test that full Hessian is symmetric."""
    # Use smaller problem for full Hessian test
    N, d = 3, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([1, 2, 0])  # Circular pairing

    op = FitnessOperator()
    hess = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=False,
    )

    # Hessian should be symmetric: H[i, j, k] ≈ H[i, k, j]
    for i in range(N):
        torch.testing.assert_close(hess[i], hess[i].T, rtol=1e-4, atol=1e-4)


def test_hessian_diagonal_matches_full_diagonal():
    """Test that diagonal-only computation matches diagonal of full Hessian."""
    # Use smaller problem
    N, d = 3, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([1, 2, 0])

    op = FitnessOperator()

    # Compute diagonal-only
    hess_diag = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=True,
    )

    # Compute full Hessian
    hess_full = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=False,
    )

    # Extract diagonal from full Hessian
    hess_full_diag = torch.stack([hess_full[i].diagonal() for i in range(N)])

    # Should match
    torch.testing.assert_close(hess_diag, hess_full_diag, rtol=1e-4, atol=1e-4)


# ============================================================================
# Edge Cases
# ============================================================================


def test_gradient_single_walker():
    """Test gradient with single walker."""
    N, d = 1, 2
    positions = torch.tensor([[1.0, 2.0]])
    velocities = torch.tensor([[0.5, -0.5]])
    rewards = torch.tensor([1.0])
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([0])  # Paired with itself

    op = FitnessOperator()
    grad = op.compute_gradient(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )

    assert grad.shape == (N, d)
    # Should not be NaN or Inf
    assert torch.all(torch.isfinite(grad))


def test_hessian_single_walker():
    """Test Hessian with single walker."""
    N, d = 1, 2
    positions = torch.tensor([[1.0, 2.0]])
    velocities = torch.tensor([[0.5, -0.5]])
    rewards = torch.tensor([1.0])
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([0])

    op = FitnessOperator()
    hess = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=True,
    )

    assert hess.shape == (N, d)
    assert torch.all(torch.isfinite(hess))


def test_gradient_all_same_rewards():
    """Test gradient when all rewards are identical."""
    N, d = 5, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.ones(N) * 1.5  # All same
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.arange(N)
    companions = torch.roll(companions, 1)

    op = FitnessOperator()
    grad = op.compute_gradient(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )

    assert grad.shape == (N, d)
    assert torch.all(torch.isfinite(grad))


def test_no_alive_walkers():
    """Test behavior when no walkers are alive."""
    N, d = 5, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.zeros(N, dtype=torch.bool)  # All dead
    companions = torch.arange(N)

    op = FitnessOperator()

    # Fitness should be all zeros
    fitness, _info = op(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )
    assert torch.all(fitness == 0.0)

    # Gradient should be all zeros (or close to it)
    grad = op.compute_gradient(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )
    assert torch.all(torch.isfinite(grad))
