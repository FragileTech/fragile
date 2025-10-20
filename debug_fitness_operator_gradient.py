"""Debug FitnessOperator.compute_gradient() vs direct autograd."""

import torch
from fragile.core.fitness import FitnessOperator, compute_fitness

# Replicate test fixture
N, d = 10, 2
torch.manual_seed(42)

positions = torch.randn(N, d)
velocities = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
companions = torch.arange(N)
companions = torch.roll(companions, 1)

op = FitnessOperator()

print("="*60)
print("METHOD 1: FitnessOperator.compute_gradient()")
print("="*60)

grad_method1 = op.compute_gradient(
    positions=positions,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)
print(f"Gradient:\n{grad_method1}")

print("\n" + "="*60)
print("METHOD 2: Direct autograd on compute_fitness")
print("="*60)

positions_grad = positions.clone().requires_grad_(True)
fitness, _, _ = compute_fitness(
    positions=positions_grad,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)
fitness_sum = fitness.sum()
grad_method2 = torch.autograd.grad(fitness_sum, positions_grad)[0]
print(f"Gradient:\n{grad_method2}")

print("\n" + "="*60)
print("METHOD 3: Finite differences")
print("="*60)

eps = 1e-5
grad_fd = torch.zeros_like(positions)

for i in range(N):
    for j in range(d):
        pos_plus = positions.clone()
        pos_plus[i, j] += eps
        pos_minus = positions.clone()
        pos_minus[i, j] -= eps

        fitness_plus, _, _ = compute_fitness(pos_plus, velocities, rewards, alive, companions)
        fitness_minus, _, _ = compute_fitness(pos_minus, velocities, rewards, alive, companions)

        grad_fd[i, j] = (fitness_plus.sum() - fitness_minus.sum()) / (2 * eps)

print(f"Gradient:\n{grad_fd}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

print(f"\nMethod 1 vs Method 2:")
diff_12 = (grad_method1 - grad_method2).abs()
print(f"  Max diff: {diff_12.max():.6e}")
print(f"  Are they the same? {torch.allclose(grad_method1, grad_method2, atol=1e-10)}")

print(f"\nMethod 2 vs Finite Diff:")
diff_2fd = (grad_method2 - grad_fd).abs()
print(f"  Max diff: {diff_2fd.max():.6e}")
print(f"  Are they close? {torch.allclose(grad_method2, grad_fd, atol=1e-4, rtol=1e-4)}")

print(f"\nMethod 1 vs Finite Diff:")
diff_1fd = (grad_method1 - grad_fd).abs()
print(f"  Max diff: {diff_1fd.max():.6e}")
print(f"  Are they close? {torch.allclose(grad_method1, grad_fd, atol=1e-4, rtol=1e-4)}")

print("\n" + "="*60)
print("DIAGNOSIS")
print("="*60)

if torch.allclose(grad_method1, grad_method2, atol=1e-10):
    print("✓ FitnessOperator.compute_gradient() matches direct autograd")
else:
    print("✗ FitnessOperator.compute_gradient() differs from direct autograd")
    print("  This would be a bug in the method implementation")

if torch.allclose(grad_method2, grad_fd, atol=1e-4):
    print("✓ Autograd matches finite differences")
    print("  The gradient implementation is CORRECT")
else:
    print("✗ Autograd does NOT match finite differences")
    print("  This suggests standardization creates mean-field coupling that")
    print("  autograd doesn't capture, OR there's a bug in compute_fitness")
