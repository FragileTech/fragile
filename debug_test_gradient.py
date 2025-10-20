"""Debug the exact test case that's failing."""

import torch

from fragile.core.fitness import FitnessOperator


# Replicate the test fixture exactly
N, d = 10, 2
torch.manual_seed(42)

positions = torch.randn(N, d)
velocities = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
# Fix companions for reproducibility
companions = torch.arange(N)
companions = torch.roll(companions, 1)  # Each walker paired with next one

print(f"Companions: {companions}")
print(f"Positions:\n{positions}")

# Create operator
op = FitnessOperator()

# Compute gradient using autograd
grad_auto = op.compute_gradient(
    positions=positions,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)

print(f"\nAutograd gradient:\n{grad_auto}")

# Compute gradient using finite differences (replicate test code)
eps = 1e-5
grad_fd = torch.zeros_like(positions)

for i in range(N):
    for j in range(d):
        # Perturb position forward
        pos_plus = positions.clone()
        pos_plus[i, j] += eps

        # Perturb position backward
        pos_minus = positions.clone()
        pos_minus[i, j] -= eps

        # Compute fitness at both points
        fitness_plus, _, _ = op(pos_plus, velocities, rewards, alive, companions)
        fitness_minus, _, _ = op(pos_minus, velocities, rewards, alive, companions)

        # Central difference
        grad_fd[i, j] = (fitness_plus.sum() - fitness_minus.sum()) / (2 * eps)

print(f"\nFinite difference gradient:\n{grad_fd}")

# Compare
diff = (grad_auto - grad_fd).abs()
print(f"\nAbsolute difference:\n{diff}")
print(f"\nMax abs diff: {diff.max():.6e}")
print(f"Mean abs diff: {diff.mean():.6e}")

# Check relative difference (avoid division by zero)
rel_diff = diff / (grad_fd.abs() + 1e-10)
print(f"\nMax rel diff: {rel_diff.max():.6e}")

# Check if within test tolerance
atol, rtol = 1e-4, 1e-4
within_tolerance = torch.allclose(grad_auto, grad_fd, rtol=rtol, atol=atol)
print(f"\nWithin tolerance (atol={atol}, rtol={rtol}): {within_tolerance}")

if not within_tolerance:
    # Find which elements fail
    failures = ~torch.isclose(grad_auto, grad_fd, rtol=rtol, atol=atol)
    print(f"\nNumber of failing elements: {failures.sum()}")
    print(f"Failing indices: {torch.where(failures)}")
