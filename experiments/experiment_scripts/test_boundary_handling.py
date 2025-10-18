#!/usr/bin/env python
"""
Test boundary handling: verify walkers die and resurrect correctly.
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
from fragile.bounds import TorchBounds

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

print("=" * 70)
print("Boundary Handling Test")
print("=" * 70)

# ============================================================================
# STEP 1: Create Potential and Bounds
# ============================================================================
print("\n[1/4] Setting up potential and bounds...")

potential, target_mixture = create_multimodal_potential(
    dims=2,
    n_gaussians=3,
    bounds_range=(-8.0, 8.0),
    seed=42
)

# Define strict bounds
bounds = TorchBounds(
    low=torch.tensor([-8.0, -8.0]),
    high=torch.tensor([8.0, 8.0])
)

print(f"  ✓ Created potential")
print(f"  ✓ Bounds: [{bounds.low.tolist()}, {bounds.high.tolist()}]")

# ============================================================================
# STEP 2: Initialize Swarm Near Boundary
# ============================================================================
print("\n[2/4] Initializing swarm near boundary...")

N = 50
dims = 2

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
    bounds=bounds,  # <- Critical!
    device="cpu",
    dtype="float32"
)

gas = GeometricGas(params, measurement_fn=measurement_fn)

# Initialize near boundary (upper edge)
x_init = torch.rand(N, dims) * 2.0 + 6.0  # [6, 8] x [6, 8]
v_init = torch.randn(N, dims) * 0.5  # Larger velocities to cross boundary

state = gas.initialize_state(x_init, v_init)

# Check initial alive count
alive_mask_init = bounds.contains(state.x)
n_alive_init = alive_mask_init.sum().item()

print(f"  ✓ Initialized {N} walkers")
print(f"  ✓ Initial alive: {n_alive_init}/{N}")
print(f"  ✓ Initial position range: [{state.x.min():.2f}, {state.x.max():.2f}]")

# ============================================================================
# STEP 3: Run Steps and Track Boundary Crossings
# ============================================================================
print("\n[3/4] Running simulation and tracking boundary events...")

n_steps = 500
alive_counts = [n_alive_init]
death_events = []
resurrection_events = []

prev_alive_mask = alive_mask_init

for step in range(n_steps):
    # Step the swarm
    _, state = gas.step(state)

    # Check alive status
    alive_mask = bounds.contains(state.x)
    n_alive = alive_mask.sum().item()
    alive_counts.append(n_alive)

    # Detect deaths (was alive, now dead)
    newly_dead = prev_alive_mask & (~alive_mask)
    n_deaths = newly_dead.sum().item()
    if n_deaths > 0:
        death_events.append((step + 1, n_deaths))

    # Detect resurrections (was dead, now alive)
    newly_alive = (~prev_alive_mask) & alive_mask
    n_resurrections = newly_alive.sum().item()
    if n_resurrections > 0:
        resurrection_events.append((step + 1, n_resurrections))

    prev_alive_mask = alive_mask

    # Report every 100 steps
    if (step + 1) % 100 == 0:
        print(f"  Step {step + 1:3d}: {n_alive:2d}/{N} alive, "
              f"pos range [{state.x.min():.2f}, {state.x.max():.2f}]")

n_alive_final = alive_counts[-1]

print(f"\n  ✓ Simulation complete")
print(f"  ✓ Final alive: {n_alive_final}/{N}")

# ============================================================================
# STEP 4: Analyze Boundary Events
# ============================================================================
print("\n[4/4] Analyzing boundary events...")

print("\n" + "=" * 70)
print("BOUNDARY EVENT ANALYSIS")
print("=" * 70)

# Death events
total_deaths = sum(n for _, n in death_events)
print(f"\nDeath Events: {len(death_events)} times, {total_deaths} total deaths")
if len(death_events) > 0:
    print(f"  First deaths at step {death_events[0][0]}: {death_events[0][1]} walkers")
    if len(death_events) > 1:
        print(f"  Sample events: {death_events[:5]}")
else:
    print(f"  ⚠️  No deaths observed!")

# Resurrection events
total_resurrections = sum(n for _, n in resurrection_events)
print(f"\nResurrection Events: {len(resurrection_events)} times, {total_resurrections} total")
if len(resurrection_events) > 0:
    print(f"  First resurrection at step {resurrection_events[0][0]}: {resurrection_events[0][1]} walkers")
    if len(resurrection_events) > 1:
        print(f"  Sample events: {resurrection_events[:5]}")
else:
    print(f"  ⚠️  No resurrections observed!")

# Alive count statistics
min_alive = min(alive_counts)
max_alive = max(alive_counts)
avg_alive = np.mean(alive_counts)

print(f"\nAlive Walker Statistics:")
print(f"  Initial: {n_alive_init}/{N}")
print(f"  Minimum: {min_alive}/{N}")
print(f"  Maximum: {max_alive}/{N}")
print(f"  Average: {avg_alive:.1f}/{N}")
print(f"  Final: {n_alive_final}/{N}")

# Validation
print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

if total_deaths > 0:
    print("✓ Boundaries are enforced (walkers died)")
else:
    print("⚠️  No deaths - walkers may not have reached boundaries")
    print("  Try: larger initial velocities or longer simulation")

if total_resurrections > 0:
    print("✓ Cloning works (dead walkers resurrected)")
else:
    print("⚠️  No resurrections observed")
    print("  This is concerning if there were deaths!")

if min_alive < N and n_alive_final >= 0.5 * N:
    print("✓ System recovers (alive count bounces back)")
else:
    print("⚠️  System may not be recovering properly")

# Check if swarm is confined
final_pos = state.x.detach().numpy()
in_bounds = ((final_pos >= bounds.low.numpy()).all(axis=1) &
             (final_pos <= bounds.high.numpy()).all(axis=1))
n_in_bounds = in_bounds.sum()

print(f"\n✓ Final position check:")
print(f"  Walkers in bounds: {n_in_bounds}/{N}")
if n_in_bounds == N:
    print(f"  ✓ All walkers within boundary!")
else:
    print(f"  ⚠️  {N - n_in_bounds} walkers outside bounds (but marked as dead)")

print("\n" + "=" * 70)
print("Boundary Test Complete!")
print("=" * 70)
print("\nKey findings:")
print(f"  - {total_deaths} walkers died crossing boundaries")
print(f"  - {total_resurrections} dead walkers were resurrected by cloning")
print(f"  - Alive count ranged from {min_alive} to {max_alive}")
print(f"  - System maintains {n_alive_final}/{N} alive walkers at end")
print()
