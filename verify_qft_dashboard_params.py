#!/usr/bin/env python
"""Verify QFT dashboard parameters are correctly set."""

import holoviews as hv
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

# Initialize holoviews
hv.extension("bokeh")

# Create QFT config
print("Creating QFT configuration...")
config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)

print("\n" + "=" * 60)
print("QFT Dashboard Configuration")
print("=" * 60)

print("\n### Benchmark")
print(f"  Benchmark name: {config.benchmark_name}")
print(f"  Bounds extent: {config.bounds_extent}")
print(f"  Dimensions: {config.dims}")

print("\n### Simulation")
print(f"  N (walkers): {config.gas_params['N']}")
print(f"  n_steps: {config.n_steps}")

print("\n### Initialization")
print(f"  init_offset: {config.init_offset}")
print(f"  init_spread: {config.init_spread}")
print(f"  init_velocity_scale: {config.init_velocity_scale}")

print("\n### Kinetic Operator (Langevin Dynamics)")
print(f"  gamma: {config.kinetic_op.gamma}")
print(f"  beta: {config.kinetic_op.beta}")
print(f"  delta_t: {config.kinetic_op.delta_t}")
print(f"  epsilon_F: {config.kinetic_op.epsilon_F}")
print(f"  nu: {config.kinetic_op.nu}")

print("\n### Viscous Coupling")
print(f"  use_viscous_coupling: {config.kinetic_op.use_viscous_coupling}")
print(f"  viscous_length_scale: {config.kinetic_op.viscous_length_scale}")
print(f"  viscous_neighbor_mode: {config.kinetic_op.viscous_neighbor_mode}")
print(f"  viscous_neighbor_threshold: {config.kinetic_op.viscous_neighbor_threshold}")
print(f"  viscous_neighbor_penalty: {config.kinetic_op.viscous_neighbor_penalty}")

print("\n### Companion Selection")
print(f"  method: {config.companion_selection.method}")
print(f"  epsilon (diversity): {config.companion_selection.epsilon}")
print(f"  epsilon (clone): {config.companion_selection_clone.epsilon}")
print(f"  lambda_alg: {config.companion_selection.lambda_alg}")

print("\n### Fitness Operator")
print(f"  alpha: {config.fitness_op.alpha}")
print(f"  beta: {config.fitness_op.beta}")
print(f"  rho: {config.fitness_op.rho}")
print(f"  lambda_alg: {config.fitness_op.lambda_alg}")

print("\n### Cloning")
print(f"  sigma_x: {config.cloning.sigma_x}")
print(f"  alpha_restitution: {config.cloning.alpha_restitution}")
print(f"  epsilon_clone: {config.cloning.epsilon_clone}")

print("\n" + "=" * 60)
print("QFT Configuration Comparison with Notebook")
print("=" * 60)

# Expected values from the plan
expected = {
    "N": 200,
    "n_steps": 5000,
    "dims": 3,
    "bounds_extent": 10.0,
    "delta_t": 0.1005,
    "epsilon_F": 38.6373,
    "nu": 1.10,
    "viscous_length_scale": 0.251372,
    "use_viscous_coupling": True,
    "viscous_neighbor_threshold": 0.75,
    "viscous_neighbor_penalty": 0.9,
    "companion_epsilon": 2.80,
    "companion_epsilon_clone": 1.68419,
    "fitness_rho": 0.251372,
}

# Check all values
all_match = True
for key, expected_val in expected.items():
    if key == "N":
        actual_val = config.gas_params["N"]
    elif key == "n_steps":
        actual_val = config.n_steps
    elif key == "dims":
        actual_val = config.dims
    elif key == "bounds_extent":
        actual_val = config.bounds_extent
    elif key == "companion_epsilon":
        actual_val = config.companion_selection.epsilon
    elif key == "companion_epsilon_clone":
        actual_val = config.companion_selection_clone.epsilon
    elif key == "fitness_rho":
        actual_val = config.fitness_op.rho
    else:
        actual_val = getattr(config.kinetic_op, key)

    match = actual_val == expected_val
    all_match = all_match and match
    status = "✓" if match else "✗"
    print(f"{status} {key:30s}: expected={expected_val:>12}, actual={actual_val:>12}")

print("\n" + "=" * 60)
if all_match:
    print("All parameters match QFT calibration! ✓")
else:
    print("Some parameters don't match! ✗")
print("=" * 60)
