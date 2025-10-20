"""Check if standardization statistics change with perturbation."""

import torch
from fragile.core.fitness import FitnessOperator

# Replicate the test fixture
N, d = 10, 2
torch.manual_seed(42)

positions = torch.randn(N, d)
velocities = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
companions = torch.arange(N)
companions = torch.roll(companions, 1)

op = FitnessOperator()

# Compute fitness at original positions
fitness_orig, dist_orig, _ = op(positions, velocities, rewards, alive, companions)

print("Original:")
print(f"  Distances: {dist_orig}")
print(f"  Dist mean: {dist_orig.mean():.6f}")
print(f"  Dist std: {dist_orig.std():.6f}")
print(f"  Fitness: {fitness_orig}")

# Perturb position of walker 0
eps = 1e-5
pos_pert = positions.clone()
pos_pert[0, 0] += eps

fitness_pert, dist_pert, _ = op(pos_pert, velocities, rewards, alive, companions)

print("\nPerturbed (walker 0, dimension 0):")
print(f"  Distances: {dist_pert}")
print(f"  Dist mean: {dist_pert.mean():.6f}")
print(f"  Dist std: {dist_pert.std():.6f}")
print(f"  Fitness: {fitness_pert}")

print("\nChanges:")
print(f"  Distance change: {(dist_pert - dist_orig).abs()}")
print(f"  Mean distance change: {abs(dist_pert.mean() - dist_orig.mean()):.6e}")
print(f"  Std distance change: {abs(dist_pert.std() - dist_orig.std()):.6e}")
print(f"  Fitness change: {(fitness_pert - fitness_orig).abs()}")

print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print("""
When we perturb walker 0's position, the distances for TWO walkers change:
1. Walker 0's distance to its companion (walker 9)
2. Walker 1's distance to its companion (walker 0)

But the mean and std of ALL distances change (even slightly), which affects
the Z-scores of ALL walkers through standardization!

This means the fitness function is NOT a sum of independent per-walker terms.
Instead, it has a MEAN-FIELD coupling through the standardization step.

The gradient ∂(Σfitness)/∂x_i involves:
- Direct effect: how fitness_i changes when x_i changes
- Mean-field effect: how ALL fitness values change due to updated statistics

Finite differences capture BOTH effects.
Autograd (with fixed statistics) captures ONLY the direct effect!
""")
