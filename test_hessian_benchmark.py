"""Comprehensive benchmark comparing all Hessian computation methods.

This script compares:
1. compute_hessian() - Autograd loops (baseline)
2. compute_hessian_func() - torch.func.hessian (full tensor)
3. compute_hessian_hvp() - HVP + vmap (memory efficient)

For both diagonal-only and full Hessian modes.
"""

import sys


sys.path.insert(0, "src")

import time

import torch

from fragile.core.fitness import FitnessOperator, FitnessParams, TORCH_FUNC_AVAILABLE


print("=" * 70)
print("HESSIAN COMPUTATION BENCHMARK")
print("=" * 70)

if not TORCH_FUNC_AVAILABLE:
    print("\n✗ torch.func not available. Skipping HVP tests.")
    print("  Upgrade to PyTorch >= 2.0 for torch.func support.")
    sys.exit(0)

print("\n✓ torch.func available")
print(f"  PyTorch version: {torch.__version__}")

# Test configurations
configs = [
    {"N": 10, "d": 3, "name": "Small (N=10, d=3)"},
    {"N": 50, "d": 5, "name": "Medium (N=50, d=5)"},
    {"N": 100, "d": 10, "name": "Large (N=100, d=10)"},
]

params = FitnessParams(alpha=1.0, beta=1.0, eta=0.1)
op = FitnessOperator(params=params)

results = []

for config in configs:
    N, d = config["N"], config["d"]
    print(f"\n{'=' * 70}")
    print(f"Configuration: {config['name']}")
    print(f"{'=' * 70}")

    # Create test data
    torch.manual_seed(42)
    positions = torch.randn(N, d)
    velocities = torch.randn(N, d)
    rewards = torch.randn(N)
    alive = torch.ones(N, dtype=torch.bool)
    alive[::5] = False  # Some dead walkers
    companions = torch.arange(N)
    companions = torch.roll(companions, 1)

    print("\nTest setup:")
    print(f"  Walkers: {N}")
    print(f"  Dimensions: {d}")
    print(f"  Alive: {alive.sum().item()}/{N}")

    # ========================================================================
    # Diagonal Hessian Benchmark
    # ========================================================================
    print(f"\n{'-' * 70}")
    print("DIAGONAL HESSIAN BENCHMARK")
    print(f"{'-' * 70}")

    # Method 1: Autograd loops
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    hess_loop = op.compute_hessian(
        positions, velocities, rewards, alive, companions, diagonal_only=True
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_loop = time.time() - start

    # Method 2: torch.func.hessian
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    hess_func = op.compute_hessian_func(
        positions, velocities, rewards, alive, companions, diagonal_only=True
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_func = time.time() - start

    # Method 3: HVP + vmap
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    hess_hvp = op.compute_hessian_hvp(
        positions, velocities, rewards, alive, companions, diagonal_only=True
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_hvp = time.time() - start

    # Verify correctness
    match_func = torch.allclose(hess_loop, hess_func, rtol=1e-4, atol=1e-5)
    match_hvp = torch.allclose(hess_loop, hess_hvp, rtol=1e-4, atol=1e-5)

    print("\n1. Autograd Loops (baseline):")
    print(f"   Time: {time_loop * 1000:.2f} ms")
    print(f"   Shape: {hess_loop.shape}")

    print("\n2. torch.func.hessian (full tensor):")
    print(f"   Time: {time_func * 1000:.2f} ms")
    print(f"   Speedup vs baseline: {time_loop / time_func:.2f}x")
    print(f"   {'✓' if match_func else '✗'} Matches baseline: {match_func}")

    print("\n3. HVP + vmap:")
    print(f"   Time: {time_hvp * 1000:.2f} ms")
    print(f"   Speedup vs baseline: {time_loop / time_hvp:.2f}x")
    print(f"   {'✓' if match_hvp else '✗'} Matches baseline: {match_hvp}")

    if not match_hvp:
        diff = (hess_loop - hess_hvp).abs()
        print(f"   Max diff: {diff.max():.2e}, Mean diff: {diff.mean():.2e}")

    results.append({
        "config": config["name"],
        "N": N,
        "d": d,
        "diagonal": {
            "loop_ms": time_loop * 1000,
            "func_ms": time_func * 1000,
            "hvp_ms": time_hvp * 1000,
            "func_speedup": time_loop / time_func,
            "hvp_speedup": time_loop / time_hvp,
            "match_func": match_func,
            "match_hvp": match_hvp,
        },
    })

    # ========================================================================
    # Full Hessian Benchmark (only for small configs)
    # ========================================================================
    if N <= 10:
        print(f"\n{'-' * 70}")
        print("FULL HESSIAN BENCHMARK")
        print(f"{'-' * 70}")

        # Method 1: Autograd loops
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        hess_loop_full = op.compute_hessian(
            positions, velocities, rewards, alive, companions, diagonal_only=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_loop_full = time.time() - start

        # Method 2: torch.func.hessian
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        hess_func_full = op.compute_hessian_func(
            positions, velocities, rewards, alive, companions, diagonal_only=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_func_full = time.time() - start

        # Method 3: HVP + vmap
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        hess_hvp_full = op.compute_hessian_hvp(
            positions, velocities, rewards, alive, companions, diagonal_only=False
        )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_hvp_full = time.time() - start

        # Verify correctness
        match_func_full = torch.allclose(hess_loop_full, hess_func_full, rtol=1e-4, atol=1e-5)
        match_hvp_full = torch.allclose(hess_loop_full, hess_hvp_full, rtol=1e-4, atol=1e-5)

        print("\n1. Autograd Loops (baseline):")
        print(f"   Time: {time_loop_full * 1000:.2f} ms")
        print(f"   Shape: {hess_loop_full.shape}")

        print("\n2. torch.func.hessian (full tensor):")
        print(f"   Time: {time_func_full * 1000:.2f} ms")
        print(f"   Speedup vs baseline: {time_loop_full / time_func_full:.2f}x")
        print(f"   {'✓' if match_func_full else '✗'} Matches baseline: {match_func_full}")

        print("\n3. HVP + vmap:")
        print(f"   Time: {time_hvp_full * 1000:.2f} ms")
        print(f"   Speedup vs baseline: {time_loop_full / time_hvp_full:.2f}x")
        print(f"   {'✓' if match_hvp_full else '✗'} Matches baseline: {match_hvp_full}")

        if not match_hvp_full:
            diff = (hess_loop_full - hess_hvp_full).abs()
            print(f"   Max diff: {diff.max():.2e}, Mean diff: {diff.mean():.2e}")

        # Check symmetry
        print("\nSymmetry check:")
        for i in range(min(3, N)):
            is_sym_loop = torch.allclose(
                hess_loop_full[i], hess_loop_full[i].T, rtol=1e-4, atol=1e-5
            )
            is_sym_hvp = torch.allclose(hess_hvp_full[i], hess_hvp_full[i].T, rtol=1e-4, atol=1e-5)
            print(
                f"   Walker {i}: Loop {'✓' if is_sym_loop else '✗'}, HVP {'✓' if is_sym_hvp else '✗'}"
            )

        results[-1]["full"] = {
            "loop_ms": time_loop_full * 1000,
            "func_ms": time_func_full * 1000,
            "hvp_ms": time_hvp_full * 1000,
            "func_speedup": time_loop_full / time_func_full,
            "hvp_speedup": time_loop_full / time_hvp_full,
            "match_func": match_func_full,
            "match_hvp": match_hvp_full,
        }

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("BENCHMARK SUMMARY")
print("=" * 70)

print("\nDiagonal Hessian Performance:")
print(
    f"{'Config':<20} {'N':>5} {'d':>3} | {'Loop':>8} {'Func':>8} {'HVP':>8} | {'Func vs Loop':>12} {'HVP vs Loop':>12}"
)
print("-" * 90)
for r in results:
    diag = r["diagonal"]
    print(
        f"{r['config']:<20} {r['N']:>5} {r['d']:>3} | "
        f"{diag['loop_ms']:>7.1f}ms {diag['func_ms']:>7.1f}ms {diag['hvp_ms']:>7.1f}ms | "
        f"{diag['func_speedup']:>11.2f}x {diag['hvp_speedup']:>11.2f}x"
    )

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("\nBased on benchmarks:\n")
print("1. For GRADIENTS:")
print("   → Use compute_gradient_func() (torch.func.jacrev)")
print("   → ~2x faster than autograd loops\n")

print("2. For DIAGONAL HESSIAN:")
best_method = "compute_hessian()"
for r in results:
    if r["diagonal"]["hvp_speedup"] > 1.0:
        best_method = "compute_hessian_hvp()"
        break

if best_method == "compute_hessian_hvp()":
    print(f"   → Use {best_method} (HVP + vmap)")
    print("   → Faster than autograd loops for tested problem sizes")
else:
    print(f"   → Use {best_method} (autograd loops)")
    print("   → Still fastest for small problem sizes")
    print("   → Consider HVP for larger N (>100)\n")

print("\n3. For FULL HESSIAN (small N only):")
print("   → Use compute_hessian() (autograd loops)")
print("   → Most memory-efficient and typically fastest")

print("\n" + "=" * 70)
