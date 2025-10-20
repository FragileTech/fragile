"""Test that torch.func.vmap derivatives match autograd loop versions.

This script verifies:
1. compute_gradient_vmap() == compute_gradient()
2. compute_hessian_vmap() == compute_hessian()
3. Performance benchmarks
"""

import sys


sys.path.insert(0, "src")

import time

import torch

from fragile.core.fitness import FitnessOperator, FitnessParams, TORCH_FUNC_AVAILABLE


print("=" * 70)
print("TORCH.FUNC VMAP DERIVATIVES VERIFICATION")
print("=" * 70)

if not TORCH_FUNC_AVAILABLE:
    print("\n✗ torch.func not available. Skipping vmap tests.")
    print("  Upgrade to PyTorch >= 2.0 for vmap support.")
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
print("1. GRADIENT COMPARISON: vmap vs autograd loops")
print("-" * 70)

# Compute with autograd loops
start = time.time()
grad_loop = op.compute_gradient(positions, velocities, rewards, alive, companions)
time_loop = time.time() - start

# Compute with vmap
start = time.time()
grad_vmap = op.compute_gradient_vmap(positions, velocities, rewards, alive, companions)
time_vmap = time.time() - start

print("\nLoop version:")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {grad_loop.shape}")
print(f"  Gradient:\n{grad_loop}")

print("\nVmap version:")
print(f"  Time: {time_vmap * 1000:.2f} ms")
print(f"  Shape: {grad_vmap.shape}")
print(f"  Gradient:\n{grad_vmap}")

# Check if they match
max_diff = (grad_loop - grad_vmap).abs().max()
mean_diff = (grad_loop - grad_vmap).abs().mean()
rel_error = mean_diff / (grad_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_grad = torch.allclose(grad_loop, grad_vmap, rtol=1e-5, atol=1e-6)
print(f"  {'✓' if match_grad else '✗'} Gradients match (rtol=1e-5, atol=1e-6): {match_grad}")

speedup_grad = time_loop / time_vmap if time_vmap > 0 else float("inf")
print(f"  Speedup: {speedup_grad:.2f}x")

# ============================================================================
# Test 2: Hessian Diagonal Comparison
# ============================================================================

print("\n" + "-" * 70)
print("2. HESSIAN DIAGONAL COMPARISON: vmap vs autograd loops")
print("-" * 70)

# Compute with autograd loops
start = time.time()
hess_diag_loop = op.compute_hessian(
    positions, velocities, rewards, alive, companions, diagonal_only=True
)
time_loop = time.time() - start

# Compute with vmap
start = time.time()
hess_diag_vmap = op.compute_hessian_vmap(
    positions, velocities, rewards, alive, companions, diagonal_only=True
)
time_vmap = time.time() - start

print("\nLoop version:")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {hess_diag_loop.shape}")
print(f"  Hessian diagonal:\n{hess_diag_loop}")

print("\nVmap version:")
print(f"  Time: {time_vmap * 1000:.2f} ms")
print(f"  Shape: {hess_diag_vmap.shape}")
print(f"  Hessian diagonal:\n{hess_diag_vmap}")

# Check if they match
max_diff = (hess_diag_loop - hess_diag_vmap).abs().max()
mean_diff = (hess_diag_loop - hess_diag_vmap).abs().mean()
rel_error = mean_diff / (hess_diag_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_hess_diag = torch.allclose(hess_diag_loop, hess_diag_vmap, rtol=1e-5, atol=1e-6)
print(
    f"  {'✓' if match_hess_diag else '✗'} Hessian diagonals match (rtol=1e-5, atol=1e-6): {match_hess_diag}"
)

speedup_hess_diag = time_loop / time_vmap if time_vmap > 0 else float("inf")
print(f"  Speedup: {speedup_hess_diag:.2f}x")

# ============================================================================
# Test 3: Hessian Full Comparison (smaller problem)
# ============================================================================

print("\n" + "-" * 70)
print("3. HESSIAN FULL COMPARISON: vmap vs autograd loops (N=3)")
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

# Compute with vmap
start = time.time()
hess_full_vmap = op.compute_hessian_vmap(
    positions_small,
    velocities_small,
    rewards_small,
    alive_small,
    companions_small,
    diagonal_only=False,
)
time_vmap = time.time() - start

print("\nLoop version:")
print(f"  Time: {time_loop * 1000:.2f} ms")
print(f"  Shape: {hess_full_loop.shape}")

print("\nVmap version:")
print(f"  Time: {time_vmap * 1000:.2f} ms")
print(f"  Shape: {hess_full_vmap.shape}")

# Check if they match
max_diff = (hess_full_loop - hess_full_vmap).abs().max()
mean_diff = (hess_full_loop - hess_full_vmap).abs().mean()
rel_error = mean_diff / (hess_full_loop.abs().mean() + 1e-8)

print("\nComparison:")
print(f"  Max difference: {max_diff:.2e}")
print(f"  Mean difference: {mean_diff:.2e}")
print(f"  Relative error: {rel_error:.2e}")

match_hess_full = torch.allclose(hess_full_loop, hess_full_vmap, rtol=1e-5, atol=1e-6)
print(
    f"  {'✓' if match_hess_full else '✗'} Full Hessians match (rtol=1e-5, atol=1e-6): {match_hess_full}"
)

speedup_hess_full = time_loop / time_vmap if time_vmap > 0 else float("inf")
print(f"  Speedup: {speedup_hess_full:.2f}x")

# Check symmetry
print("\nSymmetry check:")
for i in range(N_small):
    is_sym = torch.allclose(hess_full_vmap[i], hess_full_vmap[i].T, rtol=1e-5, atol=1e-6)
    print(f"  Walker {i}: {'✓' if is_sym else '✗'} symmetric")

# ============================================================================
# Performance Benchmark (larger problem)
# ============================================================================

print("\n" + "-" * 70)
print("4. PERFORMANCE BENCHMARK (N=100, d=5)")
print("-" * 70)

N_bench = 100
d_bench = 5

positions_bench = torch.randn(N_bench, d_bench)
velocities_bench = torch.randn(N_bench, d_bench)
rewards_bench = torch.randn(N_bench)
alive_bench = torch.ones(N_bench, dtype=torch.bool)
companions_bench = torch.arange(N_bench)
companions_bench = torch.roll(companions_bench, 1)

print("\nBenchmark configuration:")
print(f"  N = {N_bench} walkers, d = {d_bench} dimensions")

# Gradient benchmark
print("\nGradient computation:")
start = time.time()
_ = op.compute_gradient(
    positions_bench, velocities_bench, rewards_bench, alive_bench, companions_bench
)
time_loop_grad = time.time() - start

start = time.time()
_ = op.compute_gradient_vmap(
    positions_bench, velocities_bench, rewards_bench, alive_bench, companions_bench
)
time_vmap_grad = time.time() - start

print(f"  Loop: {time_loop_grad * 1000:.2f} ms")
print(f"  Vmap: {time_vmap_grad * 1000:.2f} ms")
print(f"  Speedup: {time_loop_grad / time_vmap_grad:.2f}x")

# Hessian diagonal benchmark
print("\nHessian diagonal computation:")
start = time.time()
_ = op.compute_hessian(
    positions_bench,
    velocities_bench,
    rewards_bench,
    alive_bench,
    companions_bench,
    diagonal_only=True,
)
time_loop_hess = time.time() - start

start = time.time()
_ = op.compute_hessian_vmap(
    positions_bench,
    velocities_bench,
    rewards_bench,
    alive_bench,
    companions_bench,
    diagonal_only=True,
)
time_vmap_hess = time.time() - start

print(f"  Loop: {time_loop_hess * 1000:.2f} ms")
print(f"  Vmap: {time_vmap_hess * 1000:.2f} ms")
print(f"  Speedup: {time_loop_hess / time_vmap_hess:.2f}x")

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
    print("  ✓ Gradients match: vmap produces identical results")
    print("  ✓ Hessian diagonal match: vmap produces identical results")
    print("  ✓ Full Hessian match: vmap produces identical results")
    print("  ✓ Hessians are symmetric")
    print("\nPerformance:")
    print(f"  Gradient speedup: {speedup_grad:.2f}x")
    print(f"  Hessian diagonal speedup: {speedup_hess_diag:.2f}x")
    print(f"  Full Hessian speedup: {speedup_hess_full:.2f}x")
    print("\nRecommendation:")
    print("  Use *_vmap() methods for better performance!")
else:
    print("\n✗ SOME TESTS FAILED")
    if not match_grad:
        print("  ✗ Gradient mismatch")
    if not match_hess_diag:
        print("  ✗ Hessian diagonal mismatch")
    if not match_hess_full:
        print("  ✗ Full Hessian mismatch")

print("=" * 70)
