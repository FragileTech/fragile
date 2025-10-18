#!/usr/bin/env python
"""Quick test with fewer steps for debugging."""

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


# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("Quick convergence test (500 steps)")
print("=" * 60)

# Create potential
print("\n[1/4] Creating potential...")
potential, target_mixture = create_multimodal_potential(dims=2, n_gaussians=3, seed=42)
print("  ✓ Created 3-mode mixture")

# Create GeometricGas
print("\n[2/4] Creating GeometricGas...")
N = 50  # Fewer walkers for speed


def measurement_fn(x):
    return -potential.evaluate(x)


params = GeometricGasParams(
    N=N,
    d=2,
    potential=potential,
    langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.05),
    localization=LocalizationKernelParams(rho=2.0, kernel_type="gaussian"),
    adaptive=AdaptiveParams(
        epsilon_F=0.05,
        nu=0.02,
        epsilon_Sigma=0.01,
        rescale_amplitude=1.0,
        sigma_var_min=0.1,
        viscous_length_scale=2.0,
    ),
    device="cpu",
    dtype="float32",
)

gas = GeometricGas(params, measurement_fn=measurement_fn)
print(f"  ✓ Created gas with {N} walkers")

# Initialize state
print("\n[3/4] Initializing state...")
x_init = torch.rand(N, 2) * 2.0 + 5.0
v_init = torch.randn(N, 2) * 0.1
print(f"  ✓ Initial range: [{x_init.min():.2f}, {x_init.max():.2f}]")

# Run experiment
print("\n[4/4] Running 500 steps...")
analyzer = ConvergenceAnalyzer(
    target_mixture=target_mixture,
    target_centers=target_mixture.centers,
    target_weights=target_mixture.weights,
)

experiment = ConvergenceExperiment(gas=gas, analyzer=analyzer, save_snapshots_at=[0, 100, 500])

metrics, snapshots = experiment.run(
    n_steps=500, x_init=x_init, v_init=v_init, measure_every=10, verbose=True
)

# Summary
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

summary = experiment.get_convergence_summary()

print(f"\nMeasurements: {summary['n_steps']}")
print(f"Snapshots: {len(snapshots)}")

if "final_kl" in summary:
    print("\nFinal metrics:")
    print(f"  KL divergence: {summary['final_kl']:.6f}")
    print(f"  Wasserstein-2: {summary['final_w2']:.6f}")
    print(f"  Lyapunov: {summary['final_lyapunov']:.6f}")

if "kl_convergence_rate" in summary:
    print("\nExponential fit:")
    print(f"  κ (rate): {summary['kl_convergence_rate']:.6f}")
    print(f"  Half-life: {summary['kl_half_life']:.2f} steps")
else:
    print("\n⚠️  Not enough data for exponential fit (need more steps)")

print("\n✓ Quick test complete!\n")
