#!/usr/bin/env python
"""
Test two-swarm convergence with framework-correct Lyapunov functions.

This script demonstrates that two swarms starting from different initial
conditions converge to the same QSD, with exponential Lyapunov decay.
"""

import sys
sys.path.insert(0, '../../src')

import torch
import numpy as np

from fragile.experiments import create_multimodal_potential
from fragile.geometric_gas import (
    GeometricGas,
    GeometricGasParams,
    LocalizationKernelParams,
    AdaptiveParams,
)
from fragile.euclidean_gas import LangevinParams
from fragile.lyapunov import (
    compute_internal_variance_position,
    compute_internal_variance_velocity,
    compute_total_lyapunov,
)

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Two-Swarm Convergence Test")
print("=" * 70)

# ============================================================================
# STEP 1: Create Potential
# ============================================================================
print("\n[1/5] Creating multimodal potential...")

potential, target_mixture = create_multimodal_potential(
    dims=2,
    n_gaussians=3,
    bounds_range=(-8.0, 8.0),
    seed=42
)

print(f"  ✓ Created 3-mode Gaussian mixture")
print(f"  ✓ Centers: {target_mixture.centers.numpy()}")
print(f"  ✓ Weights: {target_mixture.weights.numpy()}")

# ============================================================================
# STEP 2: Initialize Two Swarms
# ============================================================================
print("\n[2/5] Initializing two independent swarms...")

N = 50  # Walkers per swarm
dims = 2
n_steps = 1000

def measurement_fn(x):
    return -potential.evaluate(x)

params = GeometricGasParams(
    N=N,
    d=dims,
    potential=potential,
    langevin=LangevinParams(gamma=1.0, beta=1.0, delta_t=0.05),
    localization=LocalizationKernelParams(rho=2.0, kernel_type="gaussian"),
    adaptive=AdaptiveParams(
        epsilon_F=0.05,
        nu=0.02,
        epsilon_Sigma=0.01,
        rescale_amplitude=1.0,
        sigma_var_min=0.1,
        viscous_length_scale=2.0
    ),
    device="cpu",
    dtype="float32"
)

# Create two independent Gas instances
gas1 = GeometricGas(params, measurement_fn=measurement_fn)
gas2 = GeometricGas(params, measurement_fn=measurement_fn)

# Swarm 1: Upper right corner
x1_init = torch.rand(N, dims) * 2.0 + 5.0
v1_init = torch.randn(N, dims) * 0.1
state1 = gas1.initialize_state(x1_init, v1_init)

# Swarm 2: Lower left corner
x2_init = torch.rand(N, dims) * 2.0 - 7.0
v2_init = torch.randn(N, dims) * 0.1
state2 = gas2.initialize_state(x2_init, v2_init)

print(f"  ✓ Created two independent swarms")
print(f"  ✓ Swarm 1 initial: [{x1_init.min():.2f}, {x1_init.max():.2f}]")
print(f"  ✓ Swarm 2 initial: [{x2_init.min():.2f}, {x2_init.max():.2f}]")

# ============================================================================
# STEP 3: Compute Initial Metrics
# ============================================================================
print("\n[3/5] Computing initial framework-correct Lyapunov functions...")

# Swarm 1
V_var_x_1_init = compute_internal_variance_position(state1)
V_var_v_1_init = compute_internal_variance_velocity(state1)
V_total_1_init = compute_total_lyapunov(state1)

# Swarm 2
V_var_x_2_init = compute_internal_variance_position(state2)
V_var_v_2_init = compute_internal_variance_velocity(state2)
V_total_2_init = compute_total_lyapunov(state2)

# Inter-swarm distance
mu_x_1_init = state1.x.mean(dim=0)
mu_x_2_init = state2.x.mean(dim=0)
com_distance_init = torch.norm(mu_x_1_init - mu_x_2_init).item()

print(f"\n  Swarm 1 (initial):")
print(f"    V_Var,x: {V_var_x_1_init.item():.6f}")
print(f"    V_Var,v: {V_var_v_1_init.item():.6f}")
print(f"    V_total: {V_total_1_init.item():.6f}")

print(f"\n  Swarm 2 (initial):")
print(f"    V_Var,x: {V_var_x_2_init.item():.6f}")
print(f"    V_Var,v: {V_var_v_2_init.item():.6f}")
print(f"    V_total: {V_total_2_init.item():.6f}")

print(f"\n  Inter-swarm distance: {com_distance_init:.4f}")

# ============================================================================
# STEP 4: Run Simulation
# ============================================================================
print(f"\n[4/5] Running simulation for {n_steps} steps...")

metrics = {
    'V_total_1': [V_total_1_init.item()],
    'V_total_2': [V_total_2_init.item()],
    'com_distance': [com_distance_init],
}

# Run simulation
for step in range(n_steps):
    # Step both swarms
    _, state1 = gas1.step(state1)
    _, state2 = gas2.step(state2)

    # Record metrics every 100 steps
    if (step + 1) % 100 == 0:
        V_total_1 = compute_total_lyapunov(state1).item()
        V_total_2 = compute_total_lyapunov(state2).item()

        mu_x_1 = state1.x.mean(dim=0)
        mu_x_2 = state2.x.mean(dim=0)
        com_distance = torch.norm(mu_x_1 - mu_x_2).item()

        metrics['V_total_1'].append(V_total_1)
        metrics['V_total_2'].append(V_total_2)
        metrics['com_distance'].append(com_distance)

        print(f"  Step {step + 1:4d}: "
              f"V1={V_total_1:.4f}, "
              f"V2={V_total_2:.4f}, "
              f"dist={com_distance:.4f}")

print(f"\n  ✓ Simulation complete")

# ============================================================================
# STEP 5: Analyze Results
# ============================================================================
print("\n[5/5] Analyzing convergence...")

# Final metrics
V_var_x_1_final = compute_internal_variance_position(state1)
V_var_v_1_final = compute_internal_variance_velocity(state1)
V_total_1_final = compute_total_lyapunov(state1)

V_var_x_2_final = compute_internal_variance_position(state2)
V_var_v_2_final = compute_internal_variance_velocity(state2)
V_total_2_final = compute_total_lyapunov(state2)

mu_x_1_final = state1.x.mean(dim=0)
mu_x_2_final = state2.x.mean(dim=0)
com_distance_final = torch.norm(mu_x_1_final - mu_x_2_final).item()

# Final positions
pos1_final = state1.x.detach().numpy()
pos2_final = state2.x.detach().numpy()
mean1 = pos1_final.mean(axis=0)
mean2 = pos2_final.mean(axis=0)
mean_diff = np.linalg.norm(mean1 - mean2)

print("\n" + "=" * 70)
print("CONVERGENCE RESULTS")
print("=" * 70)

print(f"\nSwarm 1 (final):")
print(f"  V_Var,x: {V_var_x_1_final.item():.6f}")
print(f"  V_Var,v: {V_var_v_1_final.item():.6f}")
print(f"  V_total: {V_total_1_final.item():.6f}")
print(f"  Mean position: {mean1}")

print(f"\nSwarm 2 (final):")
print(f"  V_Var,x: {V_var_x_2_final.item():.6f}")
print(f"  V_Var,v: {V_var_v_2_final.item():.6f}")
print(f"  V_total: {V_total_2_final.item():.6f}")
print(f"  Mean position: {mean2}")

print(f"\nInter-swarm convergence:")
print(f"  Initial distance: {com_distance_init:.4f}")
print(f"  Final distance: {com_distance_final:.4f}")
print(f"  Reduction: {100 * (1 - com_distance_final/com_distance_init):.2f}%")

print(f"\nFinal distribution agreement:")
print(f"  Mean difference: {mean_diff:.6f}")

# Lyapunov decay
V1_reduction = 100 * (1 - V_total_1_final.item() / V_total_1_init.item())
V2_reduction = 100 * (1 - V_total_2_final.item() / V_total_2_init.item())

print(f"\nLyapunov function decay:")
print(f"  Swarm 1 reduction: {V1_reduction:.2f}%")
print(f"  Swarm 2 reduction: {V2_reduction:.2f}%")

# Target comparison
target_mean = (target_mixture.centers * target_mixture.weights.unsqueeze(1)).sum(dim=0).numpy()
print(f"\nTarget QSD mean: {target_mean}")
print(f"  Swarm 1 error: {np.linalg.norm(mean1 - target_mean):.6f}")
print(f"  Swarm 2 error: {np.linalg.norm(mean2 - target_mean):.6f}")

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

# Check convergence
if V1_reduction > 50 and V2_reduction > 50:
    print("✓ Both swarms show significant Lyapunov decay (>50%)")
else:
    print("⚠️  Weak Lyapunov decay - may need longer simulation")

if com_distance_final < 0.5 * com_distance_init:
    print("✓ Inter-swarm distance reduced by >50%")
else:
    print("⚠️  Swarms still far apart - may need longer simulation")

if mean_diff < 2.0:
    print("✓ Final distributions are close (mean difference < 2.0)")
else:
    print("⚠️  Final distributions differ - may need longer simulation")

print("\n" + "=" * 70)
print("Two-Swarm Test Complete!")
print("=" * 70)
print("\nNext steps:")
print("  - Run the notebook to visualize the swarm evolution")
print("  - Check Lyapunov decay plots (should be straight lines on log scale)")
print("  - Verify both swarms cover the same three modes")
print()
