"""Debug NaN in gradient computation."""

import torch

from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import compute_fitness, FitnessOperator


# Create simple test case
torch.manual_seed(42)
N, d = 10, 2
positions = torch.randn(N, d, dtype=torch.float64)
velocities = torch.randn(N, d, dtype=torch.float64)
rewards = torch.randn(N, dtype=torch.float64)
alive = torch.ones(N, dtype=torch.bool)

# Select companions
companion_selection = CompanionSelection(method="uniform")
companions = companion_selection(positions, velocities, alive)

print(f"Companions: {companions}")
print("\nChecking for self-pairing:")
for i in range(N):
    if companions[i] == i:
        print(f"  Walker {i} paired with itself!")

# Compute fitness to see intermediate values
fitness, distances, _ = compute_fitness(
    positions=positions,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)

print(f"\nFitness: {fitness}")
print(f"Distances: {distances}")

# Check if any fitness or distance is nan or inf
print(f"\nFitness has NaN: {torch.any(torch.isnan(fitness))}")
print(f"Fitness has Inf: {torch.any(torch.isinf(fitness))}")
print(f"Distances has NaN: {torch.any(torch.isnan(distances))}")
print(f"Distances has Inf: {torch.any(torch.isinf(distances))}")

# Now compute gradient and check for NaN
op = FitnessOperator()

# Enable gradient
positions_grad = positions.clone().detach().requires_grad_(True)

fitness_grad, _, _ = compute_fitness(
    positions=positions_grad,
    velocities=velocities,
    rewards=rewards,
    alive=alive,
    companions=companions,
)

print(f"\nFitness with grad: {fitness_grad}")
print(f"Fitness grad has NaN: {torch.any(torch.isnan(fitness_grad))}")

# Compute gradient
fitness_sum = fitness_grad.sum()
print(f"\nFitness sum: {fitness_sum}")
print(f"Fitness sum has NaN: {torch.isnan(fitness_sum)}")

(grad,) = torch.autograd.grad(
    outputs=fitness_sum,
    inputs=positions_grad,
    create_graph=False,
    retain_graph=False,
)

print(f"\nGradient: {grad}")
print(f"Gradient has NaN: {torch.any(torch.isnan(grad))}")

# Find which walkers have NaN gradients
nan_mask = torch.any(torch.isnan(grad), dim=1)
nan_indices = torch.where(nan_mask)[0]
print(f"\nWalkers with NaN gradients: {nan_indices.tolist()}")

for idx in nan_indices:
    idx = int(idx.item())
    print(f"\nWalker {idx}:")
    print(f"  Position: {positions[idx]}")
    print(f"  Companion: {companions[idx]}")
    print(f"  Companion position: {positions[companions[idx]]}")
    print(f"  Distance: {distances[idx]}")
    print(f"  Fitness: {fitness[idx]}")
