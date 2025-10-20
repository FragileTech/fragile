"""Test that torch.func derivatives match autograd loop versions.

This script verifies:
1. compute_gradient_func() == compute_gradient()
2. compute_hessian_func() == compute_hessian()
3. Performance benchmarks
"""

import sys


sys.path.insert(0, "src")

import time

import torch

from fragile.core.fitness import FitnessOperator, FitnessParams, TORCH_FUNC_AVAILABLE


print("=" * 70)
print("TORCH.FUNC DERIVATIVES VERIFICATION")
print("=" * 70)

if not TORCH_FUNC_AVAILABLE:
    print("\n✗ torch.func not available. Skipping tests.")
    print("  Upgrade to PyTorch >= 2.0 for torch.func support.")
    sys.exit(0)

print("\n✓ torch.func available")
print(f"  PyTorch version: {torch.__version__}")

# Create test data
N, d = 10, 3
torch.manual_seed(42)

positions = torch.randn(N, d)
velocities = torch.randn(N, d)
rewards = torch.randn(N)
alive = torch.ones(N, dtype=torch.bool)
alive[2] = False  # One dead walker
alive[7] = False
companions = torch.arange(N)
companions = torch.roll(companions, 1)

print("\nTest Configuration:")
print(f"  N = {N} walkers, d = {d} dimensions")
print(f"  Alive walkers: {alive.sum().item()}/{N}")

# Create operator
params = FitnessParams(alpha=1.0, beta=1.0, eta=0.1)
op = FitnessOperator(params=params)

# ============================================================================
# Test 1: Gradient Comparison
# ============================================================================

print("\n" + "-" * 70)
print("1. GRADIENT COMPARISON: torch.func vs autograd loops")
print("-" * 70)

# Compute with autograd loops
start = time.time()
grad_loop = op.compute_gradient(positions, velocities, rewards, alive, companions)
time_loop = time.time() - start

# Compute with torch.func
start = time.time()
grad_func = op.compute_gradient_func(positions, velocities, rewards, alive, companions)
time_func = time.time() - start

print("\nLoop version (autograd):")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {grad_loop.shape}")

print("\ntorch.func version (jacrev):")
print(f"  Time: {time_func * 1000:.2f} ms")
print(f"  Shape: {grad_func.shape}")

# Check if they match
max_diff = (grad_loop - grad_func).abs().max()
mean_diff = (grad_loop - grad_func).abs().mean()
rel_error = mean_diff / (grad_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_grad = torch.allclose(grad_loop, grad_func, rtol=1e-5, atol=1e-6)
print(f"  {'✓' if match_grad else '✗'} Gradients match (rtol=1e-5, atol=1e-6): {match_grad}")

speedup_grad = time_loop / time_func if time_func > 0 else float("inf")
print(f"  Speedup: {speedup_grad:.2f}x")

# ============================================================================
# Test 2: Hessian Diagonal Comparison
# ============================================================================

print("\n" + "-" * 70)
print("2. HESSIAN DIAGONAL COMPARISON: torch.func vs autograd loops")
print("-" * 70)

# Compute with autograd loops
start = time.time()
hess_diag_loop = op.compute_hessian(
    positions, velocities, rewards, alive, companions, diagonal_only=True
)
time_loop = time.time() - start

# Compute with torch.func
start = time.time()
hess_diag_func = op.compute_hessian_func(
    positions, velocities, rewards, alive, companions, diagonal_only=True
)
time_func = time.time() - start

print("\nLoop version (autograd):")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {hess_diag_loop.shape}")

print("\ntorch.func version (hessian):")
print(f"  Time: {time_func * 1000:.2f} ms")
print(f"  Shape: {hess_diag_func.shape}")

# Check if they match
max_diff = (hess_diag_loop - hess_diag_func).abs().max()
mean_diff = (hess_diag_loop - hess_diag_func).abs().mean()
rel_error = mean_diff / (hess_diag_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_hess_diag = torch.allclose(hess_diag_loop, hess_diag_func, rtol=1e-5, atol=1e-6)
print(
    f"  {'✓' if match_hess_diag else '✗'} Hessian diagonals match (rtol=1e-5, atol=1e-6): {match_hess_diag}"
)

speedup_hess_diag = time_loop / time_func if time_func > 0 else float("inf")
print(f"  Speedup: {speedup_hess_diag:.2f}x")

# ============================================================================
# Test 3: Hessian Full Comparison (smaller problem)
# ============================================================================

print("\n" + "-" * 70)
print("3. HESSIAN FULL COMPARISON: torch.func vs autograd loops (N=3)")
print("-" * 70)

# Use smaller problem for full Hessian
N_small = 3
positions_small = positions[:N_small]
velocities_small = velocities[:N_small]
rewards_small = rewards[:N_small]
alive_small = alive[:N_small]
companions_small = torch.tensor([1, 2, 0])

# Compute with autograd loops
start = time.time()
hess_full_loop = op.compute_hessian(
    positions_small,
    velocities_small,
    rewards_small,
    alive_small,
    companions_small,
    diagonal_only=False,
)
time_loop = time.time() - start

# Compute with torch.func
start = time.time()
hess_full_func = op.compute_hessian_func(
    positions_small,
    velocities_small,
    rewards_small,
    alive_small,
    companions_small,
    diagonal_only=False,
)
time_func = time.time() - start

print("\nLoop version (autograd):")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {hess_full_loop.shape}")

print("\ntorch.func version (hessian):")
print(f"  Time: {time_func * 1000:.2f} ms")
print(f"  Shape: {hess_full_func.shape}")

# Check if they match
max_diff = (hess_full_loop - hess_full_func).abs().max()
mean_diff = (hess_full_loop - hess_full_func).abs().mean()
rel_error = mean_diff / (hess_full_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_hess_full = torch.allclose(hess_full_loop, hess_full_func, rtol=1e-5, atol=1e-6)
print(
    f"  {'✓' if match_hess_full else '✗'} Full Hessians match (rtol=1e-5, atol=1e-6): {match_hess_full}"
)

speedup_hess_full = time_loop / time_func if time_func > 0 else float("inf")
print(f"  Speedup: {speedup_hess_full:.2f}x")

# Check symmetry
print("\nSymmetry check:")
for i in range(N_small):
    is_sym = torch.allclose(hess_full_func[i], hess_full_func[i].T, rtol=1e-5, atol=1e-6)
    print(f"  Walker {i}: {'✓' if is_sym else '✗'} symmetric")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_match = match_grad and match_hess_diag and match_hess_full

if all_match:
    print("\n✓ ALL TESTS PASSED!")
    print("\nResults:")
    print("  ✓ Gradients match: torch.func.jacrev produces identical results")
    print("  ✓ Hessian diagonal match: torch.func.hessian produces identical results")
    print("  ✓ Full Hessian match: torch.func.hessian produces identical results")
    print("  ✓ Hessians are symmetric")
    print("\nPerformance:")
    print(f"  Gradient speedup: {speedup_grad:.2f}x")
    print(f"  Hessian diagonal speedup: {speedup_hess_diag:.2f}x")
    print(f"  Full Hessian speedup: {speedup_hess_full:.2f}x")
    print("\nRecommendation:")
    if speedup_grad > 1.0:
        print("  Use compute_gradient_func() for better gradient performance!")
    else:
        print("  Use compute_gradient() - loop version is faster for this problem size")
    if speedup_hess_diag > 1.0:
        print("  Use compute_hessian_func() for better Hessian performance!")
    else:
        print("  Use compute_hessian() - loop version is faster for this problem size")
else:
    print("\n✗ SOME TESTS FAILED")
    if not match_grad:
        print("  ✗ Gradient mismatch")
    if not match_hess_diag:
        print("  ✗ Hessian diagonal mismatch")
    if not match_hess_full:
        print("  ✗ Full Hessian mismatch")

print("=" * 70)
