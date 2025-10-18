"""
Debug script for lyapunov notebook WITHOUT clustering.
Single swarm with basic Lyapunov metrics only.
"""

import sys


sys.path.insert(0, "../src")

import numpy as np
import torch

from fragile.bounds import TorchBounds
from fragile.euclidean_gas import (
    CloningParams,
    EuclideanGas,
    EuclideanGasParams,
    LangevinParams,
    SimpleQuadraticPotential,
    VectorizedOps,
)


print("=" * 80)
print("LYAPUNOV DEBUG - NO CLUSTERING")
print("=" * 80)

# ============================================================================
# 1. Configure Swarm Parameters
# ============================================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

# Swarm parameters
N = 50
d = 2

# Domain bounds
bounds = TorchBounds(low=torch.tensor([-5.0, -5.0]), high=torch.tensor([5.0, 5.0]), device=device)

# Simple quadratic potential
x_opt = torch.zeros(d, device=device)
quadratic_potential = SimpleQuadraticPotential(x_opt=x_opt, reward_alpha=1.0, reward_beta=0.0)

# Langevin parameters
langevin_params = LangevinParams(gamma=1.0, beta=1.0, delta_t=0.01, integrator="baoab")

# Cloning parameters
cloning_params = CloningParams(
    sigma_x=0.5, lambda_alg=1.0, alpha_restitution=0.5, use_inelastic_collision=True
)

# Combined parameters
params = EuclideanGasParams(
    N=N,
    d=d,
    potential=quadratic_potential,
    langevin=langevin_params,
    cloning=cloning_params,
    bounds=bounds,
    device=device,
    dtype="float32",
)

print("\nSwarm Configuration:")
print(f"  N = {N} walkers")
print(f"  d = {d} dimensions")
print(f"  Domain: [{bounds.low[0]:.1f}, {bounds.high[0]:.1f}]^{d}")

# ============================================================================
# 2. Initialize Swarm
# ============================================================================

gas = EuclideanGas(params)

torch.manual_seed(42)
x_init = bounds.sample(N)
v_init = torch.randn(N, d, device=device) * 1.0

state = gas.initialize_state(x_init, v_init)

print("\n✓ Swarm initialized")

# ============================================================================
# 3. Run Swarm and Collect Data (NO CLUSTERING)
# ============================================================================

n_steps = 150
print(f"\nRunning swarm for {n_steps} steps...")

# Storage - NO CLUSTERING
history = {
    "x": [],  # Positions [n_steps+1, N, d]
    "v": [],  # Velocities
    "alive": [],  # Alive status [n_steps+1, N]
    "potential": [],  # Per-walker potential
    "dist_to_com": [],  # Distance to center of mass
    "var_x": [],  # Position variance
    "var_v": [],  # Velocity variance
    "mu_x": [],  # Center of mass position
    "mu_v": [],  # Center of mass velocity
    "n_alive": [],  # Number of alive walkers
}


def record_state(state, step):
    """Record detailed state information WITHOUT clustering."""
    alive_mask = bounds.contains(state.x)

    # Basic state
    history["x"].append(state.x.cpu().clone())
    history["v"].append(state.v.cpu().clone())
    history["alive"].append(alive_mask.cpu().clone())

    # Per-walker metrics
    history["potential"].append(quadratic_potential.evaluate(state.x).cpu().clone())

    # Center of mass (computed over alive walkers)
    if alive_mask.any():
        mu_x = state.x[alive_mask].mean(dim=0)
        mu_v = state.v[alive_mask].mean(dim=0)
    else:
        mu_x = torch.zeros(d, device=device)
        mu_v = torch.zeros(d, device=device)

    history["mu_x"].append(mu_x.cpu().clone())
    history["mu_v"].append(mu_v.cpu().clone())

    # Distance to center of mass
    dist_to_com = torch.norm(state.x - mu_x, dim=1)
    history["dist_to_com"].append(dist_to_com.cpu().clone())

    # Variance
    var_x = VectorizedOps.variance_position(state)
    var_v = VectorizedOps.variance_velocity(state)
    history["var_x"].append(var_x.item())
    history["var_v"].append(var_v.item())

    # Alive count
    history["n_alive"].append(alive_mask.sum().item())


# Record initial state
record_state(state, 0)

# Main loop
for t in range(n_steps):
    _, state = gas.step(state)
    record_state(state, t + 1)

    if (t + 1) % 30 == 0:
        print(
            f"  Step {t + 1}/{n_steps} - Var_x: {history['var_x'][-1]:.4f}, "
            f"Alive: {history['n_alive'][-1]}/{N}"
        )

print("\n✓ Simulation complete")

# ============================================================================
# 4. Convert History to Arrays
# ============================================================================

print("\nConverting history to arrays...")

history["x"] = torch.stack(history["x"]).numpy()  # [n_steps+1, N, d]
history["v"] = torch.stack(history["v"]).numpy()
history["alive"] = torch.stack(history["alive"]).numpy()  # [n_steps+1, N]
history["potential"] = torch.stack(history["potential"]).numpy()  # [n_steps+1, N]
history["dist_to_com"] = torch.stack(history["dist_to_com"]).numpy()
history["mu_x"] = torch.stack(history["mu_x"]).numpy()  # [n_steps+1, d]
history["mu_v"] = torch.stack(history["mu_v"]).numpy()
history["var_x"] = np.array(history["var_x"])  # [n_steps+1]
history["var_v"] = np.array(history["var_v"])
history["n_alive"] = np.array(history["n_alive"])

print("✓ Arrays created")
print(f"\n  Shape of x: {history['x'].shape}")
print(f"  Shape of alive: {history['alive'].shape}")
print(f"  Shape of var_x: {history['var_x'].shape}")

# ============================================================================
# 5. Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"\nFinal state (step {n_steps}):")
print(f"  Position variance: {history['var_x'][-1]:.4f}")
print(f"  Velocity variance: {history['var_v'][-1]:.4f}")
print(f"  Alive walkers: {history['n_alive'][-1]}/{N}")
print(f"  Mean potential: {history['potential'][-1].mean():.4f}")

print("\nVariance reduction:")
print(f"  Initial var_x: {history['var_x'][0]:.4f}")
print(f"  Final var_x: {history['var_x'][-1]:.4f}")
print(f"  Reduction: {(1 - history['var_x'][-1] / history['var_x'][0]) * 100:.1f}%")

print("\n" + "=" * 80)
print("SIMULATION SUCCESSFUL - NO CLUSTERING")
print("=" * 80)
