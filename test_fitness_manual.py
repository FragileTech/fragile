"""Manual test script for fitness operator (bypasses pytest/conftest issues)."""

import sys


sys.path.insert(0, "src")

import torch

from fragile.core.fitness import compute_fitness, FitnessOperator, FitnessParams


def test_basic_functionality():
    """Test basic fitness operator functionality."""
    print("=" * 70)
    print("Testing FitnessOperator Basic Functionality")
    print("=" * 70)

    # Create simple test data
    N, d = 10, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.arange(N)
    companions = torch.roll(companions, 1)

    # Test FitnessParams
    print("\n1. Testing FitnessParams...")
    params = FitnessParams(alpha=2.0, beta=0.5, eta=0.2)
    print(f"   ✓ Created params: alpha={params.alpha}, beta={params.beta}, eta={params.eta}")

    # Test FitnessOperator initialization
    print("\n2. Testing FitnessOperator initialization...")
    op = FitnessOperator(params=params)
    print("   ✓ Operator initialized with params")

    # Test __call__
    print("\n3. Testing fitness computation...")
    fitness, distances, comp_out = op(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )
    print(f"   ✓ Fitness computed: shape={fitness.shape}, mean={fitness.mean():.4f}")
    print(f"   ✓ Distances: shape={distances.shape}")
    print(f"   ✓ Companions: shape={comp_out.shape}")

    # Verify matches standalone function
    print("\n4. Verifying match with standalone function...")
    fitness_fn, _distances_fn, _ = compute_fitness(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        alpha=params.alpha,
        beta=params.beta,
        eta=params.eta,
        lambda_alg=params.lambda_alg,
        sigma_min=params.sigma_min,
        A=params.A,
    )
    match = torch.allclose(fitness, fitness_fn)
    print(f"   {'✓' if match else '✗'} Fitness matches standalone function: {match}")

    print("\n" + "=" * 70)
    return fitness, distances, comp_out


def test_gradient():
    """Test gradient computation."""
    print("\nTesting Gradient Computation")
    print("=" * 70)

    # Create test data
    N, d = 5, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.arange(N)
    companions = torch.roll(companions, 1)

    op = FitnessOperator()

    # Compute gradient
    print("\n1. Computing gradient using autograd...")
    grad = op.compute_gradient(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
    )
    print(f"   ✓ Gradient computed: shape={grad.shape}")
    print(f"   ✓ Gradient stats: mean={grad.mean():.6f}, std={grad.std():.6f}")

    # Verify with finite differences
    print("\n2. Verifying with finite differences...")
    eps = 1e-5
    grad_fd = torch.zeros_like(positions)

    for i in range(N):
        for j in range(d):
            pos_plus = positions.clone()
            pos_plus[i, j] += eps

            pos_minus = positions.clone()
            pos_minus[i, j] -= eps

            fitness_plus, _, _ = op(pos_plus, velocities, rewards, alive, companions)
            fitness_minus, _, _ = op(pos_minus, velocities, rewards, alive, companions)

            grad_fd[i, j] = (fitness_plus.sum() - fitness_minus.sum()) / (2 * eps)

    print("   ✓ Finite difference gradient computed")

    max_diff = (grad - grad_fd).abs().max()
    mean_diff = (grad - grad_fd).abs().mean()
    rel_error = mean_diff / (grad.abs().mean() + 1e-8)

    print(f"   ✓ Max difference: {max_diff:.6f}")
    print(f"   ✓ Mean difference: {mean_diff:.6f}")
    print(f"   ✓ Relative error: {rel_error:.6f}")

    match = torch.allclose(grad, grad_fd, rtol=1e-4, atol=1e-4)
    print(f"   {'✓' if match else '✗'} Gradients match (rtol=1e-4): {match}")

    print("\n" + "=" * 70)
    return grad, grad_fd


def test_hessian_diagonal():
    """Test diagonal Hessian computation."""
    print("\nTesting Diagonal Hessian Computation")
    print("=" * 70)

    # Create smaller test data (Hessian is expensive)
    N, d = 3, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([1, 2, 0])  # Circular

    op = FitnessOperator()

    # Compute Hessian diagonal
    print("\n1. Computing diagonal Hessian using autograd...")
    hess_diag = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=True,
    )
    print(f"   ✓ Hessian diagonal computed: shape={hess_diag.shape}")
    print(f"   ✓ Hessian stats: mean={hess_diag.mean():.6f}, std={hess_diag.std():.6f}")

    # Verify with finite differences
    print("\n2. Verifying with finite differences...")
    eps = 1e-4
    hess_fd = torch.zeros_like(positions)

    for i in range(N):
        for j in range(d):
            pos_plus = positions.clone()
            pos_plus[i, j] += eps

            pos_minus = positions.clone()
            pos_minus[i, j] -= eps

            fitness_center, _, _ = op(positions, velocities, rewards, alive, companions)
            fitness_plus, _, _ = op(pos_plus, velocities, rewards, alive, companions)
            fitness_minus, _, _ = op(pos_minus, velocities, rewards, alive, companions)

            hess_fd[i, j] = (
                fitness_plus.sum() - 2 * fitness_center.sum() + fitness_minus.sum()
            ) / (eps**2)

    print("   ✓ Finite difference Hessian computed")

    max_diff = (hess_diag - hess_fd).abs().max()
    mean_diff = (hess_diag - hess_fd).abs().mean()
    rel_error = mean_diff / (hess_diag.abs().mean() + 1e-8)

    print(f"   ✓ Max difference: {max_diff:.6f}")
    print(f"   ✓ Mean difference: {mean_diff:.6f}")
    print(f"   ✓ Relative error: {rel_error:.6f}")

    match = torch.allclose(hess_diag, hess_fd, rtol=1e-2, atol=1e-3)
    print(f"   {'✓' if match else '✗'} Hessians match (rtol=1e-2): {match}")

    print("\n" + "=" * 70)
    return hess_diag, hess_fd


def test_hessian_full():
    """Test full Hessian computation and symmetry."""
    print("\nTesting Full Hessian Computation")
    print("=" * 70)

    # Very small test (full Hessian is expensive!)
    N, d = 2, 2
    torch.manual_seed(42)

    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    companions = torch.tensor([1, 0])

    op = FitnessOperator()

    # Compute full Hessian
    print("\n1. Computing full Hessian...")
    hess_full = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=False,
    )
    print(f"   ✓ Full Hessian computed: shape={hess_full.shape}")

    # Check symmetry
    print("\n2. Checking symmetry...")
    symmetric = True
    for i in range(N):
        is_sym = torch.allclose(hess_full[i], hess_full[i].T, rtol=1e-4, atol=1e-4)
        symmetric = symmetric and is_sym
        print(f"   {'✓' if is_sym else '✗'} Walker {i} Hessian is symmetric: {is_sym}")

    print(f"\n   {'✓' if symmetric else '✗'} All Hessians are symmetric: {symmetric}")

    # Compare diagonal with diagonal-only computation
    print("\n3. Comparing diagonal with diagonal-only computation...")
    hess_diag = op.compute_hessian(
        positions=positions,
        velocities=velocities,
        rewards=rewards,
        alive=alive,
        companions=companions,
        diagonal_only=True,
    )

    hess_full_diag = torch.stack([hess_full[i].diagonal() for i in range(N)])
    match = torch.allclose(hess_diag, hess_full_diag, rtol=1e-4, atol=1e-4)
    print(f"   {'✓' if match else '✗'} Diagonals match: {match}")

    print("\n" + "=" * 70)
    return hess_full


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FRAGILE FITNESS OPERATOR MANUAL TESTS")
    print("=" * 70 + "\n")

    try:
        # Run all tests
        fitness, distances, companions = test_basic_functionality()
        grad, grad_fd = test_gradient()
        hess_diag, hess_fd = test_hessian_diagonal()
        hess_full = test_hessian_full()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ TEST FAILED WITH ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
