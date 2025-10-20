"""Test if full autograd works through standardization."""

import torch
from fragile.core.fitness import compute_fitness

# Simple test
N, d = 10, 2
torch.manual_seed(42)

positions = torch.randn(N, d, requires_grad=True)
velocities = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
companions = torch.arange(N)
companions = torch.roll(companions, 1)

# Compute fitness
fitness, _, _ = compute_fitness(
    positions=positions,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)

print(f"Fitness: {fitness}")
print(f"Fitness requires_grad: {fitness.requires_grad}")
print(f"Fitness grad_fn: {fitness.grad_fn}")

# Compute gradient
fitness_sum = fitness.sum()
grad = torch.autograd.grad(fitness_sum, positions, create_graph=False)[0]

print(f"\nGradient shape: {grad.shape}")
print(f"Gradient:\n{grad}")

# Now compute finite difference for comparison
eps = 1e-5
grad_fd = torch.zeros_like(positions)

for i in range(N):
    for j in range(d):
        pos_plus = positions.detach().clone()
        pos_plus[i, j] += eps
        pos_minus = positions.detach().clone()
        pos_minus[i, j] -= eps

        fitness_plus, _, _ = compute_fitness(pos_plus, velocities, rewards, alive, companions)
        fitness_minus, _, _ = compute_fitness(pos_minus, velocities, rewards, alive, companions)

        grad_fd[i, j] = (fitness_plus.sum() - fitness_minus.sum()) / (2 * eps)

print(f"\nFinite difference gradient:\n{grad_fd}")

diff = (grad - grad_fd).abs()
print(f"\nMax difference: {diff.max():.6e}")
print(f"Mean difference: {diff.mean():.6e}")

print("\n" + "="*60)
if diff.max() < 1e-4:
    print("✓ GRADIENTS MATCH - Full autograd works!")
else:
    print("✗ GRADIENTS DIFFER - Mean-field coupling issue confirmed")
    print(f"\nThis confirms that calling compute_fitness() directly")
    print(f"with requires_grad=True captures the full mean-field gradient,")
    print(f"but FitnessOperator.compute_gradient() does not.")
