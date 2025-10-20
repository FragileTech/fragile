"""Debug Hessian computation to find NaN source."""

import sys


sys.path.insert(0, "src")

import torch

from fragile.core.fitness import FitnessOperator


N, d = 3, 2
torch.manual_seed(42)

x = torch.randn(N, d)
v = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
# Circular pairing: 0→1, 1→2, 2→0 (avoid self-pairing)
companions = torch.tensor([1, 2, 0], dtype=torch.int64)

fitness_op = FitnessOperator()

print("Testing Hessian computation...")
print(f"x:\n{x}")
print(f"v:\n{v}")
print(f"rewards: {rewards}")
print(f"alive: {alive}")
print(f"companions: {companions}")

# Compute Hessian
print("\nComputing full Hessian...")
hess_full = fitness_op.compute_hessian(x, v, rewards, alive, companions, diagonal_only=False)
print(f"Hessian shape: {hess_full.shape}")
print(f"Hessian[0]:\n{hess_full[0]}")
print(f"Hessian contains NaN: {torch.isnan(hess_full).any()}")
print(f"Hessian contains Inf: {torch.isinf(hess_full).any()}")

# Check eigenvalues
for i in range(N):
    eigenvalues = torch.linalg.eigvalsh(hess_full[i])
    print(f"\nWalker {i} eigenvalues: {eigenvalues}")

print("\n" + "=" * 70)
print("Computing diagonal Hessian...")
hess_diag = fitness_op.compute_hessian(x, v, rewards, alive, companions, diagonal_only=True)
print(f"Diagonal Hessian: {hess_diag}")
print(f"Contains NaN: {torch.isnan(hess_diag).any()}")
print(f"Contains Inf: {torch.isinf(hess_diag).any()}")
