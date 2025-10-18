#!/usr/bin/env python
"""
Run convergence experiment for Geometric Gas.
Tests computational logic without visualization.
"""

import sys


sys.path.insert(0, "../../src")

import numpy as np
import torch

from fragile.euclidean_gas import LangevinParams
from fragile.experiments import (
    ConvergenceAnalyzer,
    ConvergenceExperiment,
    create_multimodal_potential,
)
from fragile.geometric_gas import (
    AdaptiveParams,
    GeometricGas,
    GeometricGasParams,
    LocalizationKernelParams,
)


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Convergence Experiment: Geometric Gas -> QSD")
print("=" * 70)

# ============================================================================
# STEP 1: Create Multimodal Potential
# ============================================================================
print("\n[1/5] Creating multimodal potential...")

potential, target_mixture = create_multimodal_potential(
    dims=2, n_gaussians=3, bounds_range=(-8.0, 8.0), seed=42
)

print("  ✓ Created 3-mode Gaussian mixture in 2D")
print(f"  ✓ Mode centers: {target_mixture.centers.numpy()}")
print(f"  ✓ Mode weights: {target_mixture.weights.numpy()}")

# ============================================================================
# STEP 2: Initialize Geometric Gas
# ============================================================================
print("\n[2/5] Initializing Geometric Gas...")

N = 100  # Number of walkers
dims = 2


# Create measurement function
def measurement_fn(x):
    return -potential.evaluate(x)


# Configure parameters
params = GeometricGasParams(
    N=N,
    d=dims,
    potential=potential,
    langevin=LangevinParams(
        gamma=1.0,  # Friction coefficient
        beta=1.0,  # Inverse temperature
        delta_t=0.05,  # Time step
    ),
    localization=LocalizationKernelParams(
        rho=2.0,  # Localization scale
        kernel_type="gaussian",
    ),
    adaptive=AdaptiveParams(
        epsilon_F=0.05,  # Adaptive force strength
        nu=0.02,  # Viscous coupling strength
        epsilon_Sigma=0.01,  # Hessian regularization
        rescale_amplitude=1.0,
        sigma_var_min=0.1,
        viscous_length_scale=2.0,
    ),
    device="cpu",
    dtype="float32",
)

gas = GeometricGas(params, measurement_fn=measurement_fn)

print(f"  ✓ Created GeometricGas with {N} walkers")
print(f"  ✓ ρ-localization scale: {params.localization.rho}")
print(f"  ✓ Adaptive force ε_F: {params.adaptive.epsilon_F}")
print(f"  ✓ Viscous coupling ν: {params.adaptive.nu}")

# ============================================================================
# STEP 3: Initialize State (Far from Equilibrium)
# ============================================================================
print("\n[3/5] Initializing swarm state...")

# Start walkers far from equilibrium (upper right corner)
x_init = torch.rand(N, dims) * 2.0 + 5.0  # [5, 7] x [5, 7]
v_init = torch.randn(N, dims) * 0.1  # Small initial velocities

print(f"  ✓ Initial position range: [{x_init.min():.2f}, {x_init.max():.2f}]")
print(f"  ✓ Target mixture center: {target_mixture.best_state.numpy()}")

# ============================================================================
# STEP 4: Run Convergence Experiment
# ============================================================================
print("\n[4/5] Running convergence experiment...")

# Create analyzer
analyzer = ConvergenceAnalyzer(
    target_mixture=target_mixture,
    target_centers=target_mixture.centers,
    target_weights=target_mixture.weights,
)

# Create experiment
save_snapshots_at = [0, 100, 500, 1000, 2000, 5000]
experiment = ConvergenceExperiment(gas=gas, analyzer=analyzer, save_snapshots_at=save_snapshots_at)

# Run!
n_steps = 5000
measure_every = 10

metrics, snapshots = experiment.run(
    n_steps=n_steps, x_init=x_init, v_init=v_init, measure_every=measure_every, verbose=True
)

# ============================================================================
# STEP 5: Analyze Results
# ============================================================================
print("\n[5/5] Analyzing convergence...")

summary = experiment.get_convergence_summary()

print("\n" + "=" * 70)
print("CONVERGENCE SUMMARY")
print("=" * 70)

# Basic info
print("\nExperiment Statistics:")
print(f"  Total steps: {summary['final_time']}")
print(f"  Measurements: {summary['n_steps']}")
print(f"  Snapshots saved: {len(snapshots)}")

# KL-divergence convergence
if "kl_convergence_rate" in summary:
    print("\nKL-Divergence Convergence:")
    print(f"  Rate κ: {summary['kl_convergence_rate']:.6f}")
    print(f"  Constant C: {summary['kl_constant']:.4f}")
    print(f"  Half-life: {summary['kl_half_life']:.2f} steps")
    print(f"  Final KL: {summary['final_kl']:.6f}")
    print(
        f"  Exponential fit: D_KL(t) ≈ {summary['kl_constant']:.4f} * exp(-{summary['kl_convergence_rate']:.6f} * t)"
    )
else:
    print("\n⚠️  KL-divergence fit failed (insufficient data)")

# Wasserstein distance
if "w2_convergence_rate" in summary:
    print("\nWasserstein-2 Distance:")
    print(f"  Rate κ: {summary['w2_convergence_rate']:.6f}")
    print(f"  Final W2: {summary['final_w2']:.6f}")
else:
    print("\n⚠️  Wasserstein fit failed")

# Lyapunov function
if "lyapunov_decay_rate" in summary:
    print("\nLyapunov Function (Total Variance):")
    print(f"  Decay rate: {summary['lyapunov_decay_rate']:.6f}")
    print(f"  Final value: {summary['final_lyapunov']:.6f}")
else:
    print("\n⚠️  Lyapunov fit failed")

# Final state statistics
print("\nFinal Swarm State:")
print(f"  Mean position: {summary['final_mean_position']}")
print(f"  Target center: {target_mixture.best_state.numpy()}")
print(
    f"  Distance to target: {np.linalg.norm(summary['final_mean_position'] - target_mixture.best_state.numpy()):.4f}"
)

# Check snapshots
print(f"\nSnapshot Times: {sorted(snapshots.keys())}")

# Validate exponential convergence
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

if "kl_convergence_rate" in summary and summary["kl_convergence_rate"] > 0:
    print("✓ Exponential convergence detected!")
    print("  The KL-divergence shows the iconic 'straight line' on log scale")
    print(f"  Convergence rate: κ = {summary['kl_convergence_rate']:.6f}")

    # Predict when KL < threshold
    threshold = 0.01
    if summary["kl_constant"] > 0:
        t_threshold = np.log(summary["kl_constant"] / threshold) / summary["kl_convergence_rate"]
        print(f"  Predicted time to reach D_KL < {threshold}: {t_threshold:.0f} steps")
else:
    print("⚠️  Exponential convergence not clearly detected")
    print("  This could indicate:")
    print("  - Insufficient simulation time")
    print("  - Strong transient effects")
    print("  - Numerical issues with KL estimation")

print("\n" + "=" * 70)
print("Experiment Complete!")
print("=" * 70)
print("\nNext steps:")
print("  - Run the notebook to visualize the results")
print("  - Check the 'straight line' on the log-linear plot")
print("  - Visualize swarm evolution through snapshots")
print("\n")
