"""Test if patched_standardization gives correct gradients."""

import torch

from fragile.core.fitness import patched_standardization


N = 5
torch.manual_seed(42)
values = torch.randn(N, requires_grad=True)
alive = torch.ones(N, dtype=torch.bool)

print(f"Values: {values}")

# Compute Z-scores
z = patched_standardization(values, alive)
print(f"Z-scores: {z}")

# Compute gradient
z_sum = z.sum()
grad_auto = torch.autograd.grad(z_sum, values)[0]
print(f"\nAutograd gradient: {grad_auto}")

# Compute finite difference
eps = 1e-5
grad_fd = torch.zeros(N)

for i in range(N):
    v_plus = values.detach().clone()
    v_plus[i] += eps
    z_plus = patched_standardization(v_plus, alive)

    v_minus = values.detach().clone()
    v_minus[i] -= eps
    z_minus = patched_standardization(v_minus, alive)

    grad_fd[i] = (z_plus.sum() - z_minus.sum()) / (2 * eps)

print(f"Finite diff gradient: {grad_fd}")

diff = (grad_auto - grad_fd).abs()
print(f"\nMax difference: {diff.max():.6e}")

if diff.max() < 1e-4:
    print("✓ patched_standardization gradient is correct!")
else:
    print("✗ patched_standardization gradient is WRONG!")
    print("\nThis means the standardization implementation has a bug.")
