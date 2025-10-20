"""Test the exact standardization computation step by step."""

import torch


N = 5
torch.manual_seed(42)
values = torch.randn(N, requires_grad=True)
alive = torch.ones(N, dtype=torch.bool)

print(f"Values: {values}")
print(f"Values requires_grad: {values.requires_grad}")

# Step 1: Convert to float
alive_mask = alive.float()
print(f"\nAlive mask: {alive_mask}")

# Step 2: Compute mean
n_alive = alive_mask.sum()
n_alive_safe = torch.clamp(n_alive, min=1.0)
mu = (values * alive_mask).sum() / n_alive_safe

print(f"n_alive: {n_alive}")
print(f"mu: {mu}")
print(f"mu grad_fn: {mu.grad_fn}")

# Step 3: Compute std
centered = values - mu
print(f"\ncentered: {centered}")
print(f"centered grad_fn: {centered.grad_fn}")

sigma_sq = ((centered**2) * alive_mask).sum() / n_alive_safe
sigma_reg = torch.sqrt(sigma_sq + 1e-8**2)

print(f"sigma_sq: {sigma_sq}")
print(f"sigma_reg: {sigma_reg}")
print(f"sigma_reg grad_fn: {sigma_reg.grad_fn}")

# Step 4: Z-scores
z_scores = centered / sigma_reg
print(f"\nz_scores: {z_scores}")
print(f"z_scores grad_fn: {z_scores.grad_fn}")

# Step 5: Mask
z_final = z_scores * alive_mask
print(f"\nz_final: {z_final}")
print(f"z_final grad_fn: {z_final.grad_fn}")

# Compute gradient
z_sum = z_final.sum()
print(f"\nz_sum: {z_sum}")
print(f"z_sum grad_fn: {z_sum.grad_fn}")

grad = torch.autograd.grad(z_sum, values)[0]
print(f"\nGradient: {grad}")

# The gradient should NOT be zero!
print(f"\nGradient is approximately zero: {grad.abs().max() < 1e-6}")

# Let's check what the gradient SHOULD be mathematically
# For z_i = (x_i - μ) / σ where μ and σ depend on all x
# ∂(Σz_i)/∂x_j involves chain rule through μ and σ
print("\nExpected: Non-zero gradient because z_i depends on x_j through μ and σ")
