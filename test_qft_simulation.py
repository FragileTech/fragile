#!/usr/bin/env python
"""Test that a QFT simulation can actually run (with reduced steps for speed)."""

import sys

import holoviews as hv
import torch

from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel


# Initialize holoviews
hv.extension("bokeh")

print("=" * 60)
print("QFT Simulation Test (Quick Run)")
print("=" * 60)

# Create QFT config with reduced steps for testing
print("\nCreating QFT configuration...")
config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)

# Reduce steps for quick test
original_steps = config.n_steps
config.n_steps = 100  # Quick test with 100 steps
print(f"Using {config.n_steps} steps (instead of {original_steps}) for quick test")

print("\nConfiguration:")
print(f"  Benchmark: {config.benchmark_name}")
print(f"  N: {config.gas_params['N']} walkers")
print(f"  Dimensions: {config.dims}")
print(f"  Bounds: [-{config.bounds_extent}, {config.bounds_extent}]")
print(f"  delta_t: {config.kinetic_op.delta_t}")
print(f"  epsilon_F: {config.kinetic_op.epsilon_F}")
print(f"  nu: {config.kinetic_op.nu}")
print(f"  Viscous coupling: {config.kinetic_op.use_viscous_coupling}")

# Run simulation
print("\nRunning simulation...")
try:
    history = config.run_simulation()
    print("✓ Simulation completed successfully!")

    print("\nResults:")
    print(f"  Steps: {history.n_steps}")
    print(f"  Recorded timesteps: {history.n_recorded}")
    print(f"  Walkers: {history.N}")
    print(f"  Dimensions: {history.d}")

    # Check final state
    final_x = history.x_final[-1]  # Final positions [N, d]
    final_U = history.U_final[-1]  # Final potential energies [N]

    print("\nFinal state statistics:")
    print(f"  Position mean: {final_x.mean(dim=0).tolist()}")
    print(f"  Position std: {final_x.std(dim=0).tolist()}")
    print(f"  Potential mean: {final_U.mean():.4f}")
    print(f"  Potential std: {final_U.std():.4f}")

    # Verify walkers are in quadratic well (should be near origin)
    distance_from_origin = torch.norm(final_x, dim=1).mean()
    print(f"  Mean distance from origin: {distance_from_origin:.4f}")

    if distance_from_origin < config.bounds_extent:
        print("✓ Walkers are within bounds")
    else:
        print("⚠ Warning: Walkers outside expected region")

    # Verify potential values are reasonable
    expected_U = 0.5 * 0.1 * (final_x**2).sum(dim=1)  # alpha=0.1
    actual_U = final_U

    if torch.allclose(expected_U, actual_U, rtol=1e-3):
        print("✓ Potential values match quadratic well formula")
    else:
        print("⚠ Warning: Potential mismatch")
        print(f"  Expected: {expected_U.mean():.4f}")
        print(f"  Actual: {actual_U.mean():.4f}")

    print("\n" + "=" * 60)
    print("QFT Simulation Test PASSED ✓")
    print("=" * 60)
    print("\nFull simulation ready to run:")
    print("  python -m fragile.fractalai.experiments.gas_visualization_dashboard --qft")

except Exception as e:
    print(f"✗ Simulation failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
