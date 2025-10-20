"""Debug gradient computation issue."""

import torch
from fragile.core.companion_selection import CompanionSelection
from fragile.core.fitness import FitnessOperator, compute_fitness

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
print(f"Positions shape: {positions.shape}")
print(f"Velocities shape: {velocities.shape}")

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

print(f"\nAutograd gradient shape: {grad_auto.shape}")
print(f"Autograd gradient:\n{grad_auto}")

# Compute gradient using finite differences
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

print(f"\nFinite difference gradient shape: {grad_fd.shape}")
print(f"Finite difference gradient:\n{grad_fd}")

# Compare
diff = (grad_auto - grad_fd).abs()
print(f"\nAbsolute difference:\n{diff}")
print(f"Max absolute difference: {diff.max():.6e}")
print(f"Mean absolute difference: {diff.mean():.6e}")

# Check if companions change matters
print("\n" + "="*60)
print("Checking if companions change when positions change...")
print("="*60)

# Test: does perturbing position change companion selection?
for i in range(min(3, N)):
    pos_pert = positions.clone()
    pos_pert[i, 0] += eps

    companions_pert = companion_selection(pos_pert, velocities, alive)

    if not torch.all(companions_pert == companions):
        print(f"Walker {i}: Companions CHANGED when position perturbed!")
        print(f"  Original: {companions}")
        print(f"  Perturbed: {companions_pert}")
        print(f"  Difference: {(companions != companions_pert).sum()} walkers affected")
    else:
        print(f"Walker {i}: Companions unchanged")
