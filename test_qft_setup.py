#!/usr/bin/env python
"""Test script to verify QFT configuration setup."""

import holoviews as hv
import torch

from fragile.fractalai.core.benchmarks import QuadraticWell
from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel


# Initialize holoviews (required for GasConfigPanel)
hv.extension("bokeh")


def test_quadratic_well_benchmark():
    """Test QuadraticWell benchmark creation and evaluation."""
    print("Testing QuadraticWell benchmark...")

    qw = QuadraticWell(dims=3, alpha=0.1, bounds_extent=10.0)

    # Test at origin
    x_origin = torch.zeros(10, 3)
    result_origin = qw(x_origin)
    assert torch.allclose(result_origin, torch.zeros(10)), "Origin should have zero potential"

    # Test at non-zero point
    x_test = torch.ones(10, 3)
    result_test = qw(x_test)
    expected = 0.5 * 0.1 * 3.0  # 0.5 * alpha * ||x||^2
    assert torch.allclose(result_test, torch.full((10,), expected)), (
        f"Expected {expected}, got {result_test[0]}"
    )

    print("  ✓ QuadraticWell benchmark works correctly")
    print(f"  ✓ Best state: {qw.best_state}")
    print(f"  ✓ Benchmark value: {qw.benchmark}")


def test_qft_config_creation():
    """Test QFT configuration preset."""
    print("\nTesting QFT configuration preset...")

    config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)

    # Verify key parameters
    assert config.benchmark_name == "Quadratic Well", (
        f"Expected 'Quadratic Well', got {config.benchmark_name}"
    )
    assert config.bounds_extent == 10.0, f"Expected 10.0, got {config.bounds_extent}"
    assert config.n_steps == 5000, f"Expected 5000, got {config.n_steps}"
    assert config.gas_params["N"] == 200, f"Expected 200, got {config.gas_params['N']}"
    assert config.dims == 3, f"Expected 3, got {config.dims}"

    # Verify kinetic operator parameters
    assert config.kinetic_op.delta_t == 0.1005, f"Expected 0.1005, got {config.kinetic_op.delta_t}"
    assert config.kinetic_op.epsilon_F == 38.6373, (
        f"Expected 38.6373, got {config.kinetic_op.epsilon_F}"
    )
    assert config.kinetic_op.nu == 1.10, f"Expected 1.10, got {config.kinetic_op.nu}"
    assert config.kinetic_op.use_viscous_coupling is True, "Expected viscous coupling enabled"
    assert config.kinetic_op.viscous_length_scale == 0.251372
    assert config.kinetic_op.viscous_neighbor_threshold == 0.75
    assert config.kinetic_op.viscous_neighbor_penalty == 0.9

    # Verify companion selection parameters
    assert config.companion_selection.epsilon == 2.80, (
        f"Expected 2.80, got {config.companion_selection.epsilon}"
    )
    assert config.companion_selection_clone.epsilon == 1.68419, (
        f"Expected 1.68419, got {config.companion_selection_clone.epsilon}"
    )

    # Verify fitness operator
    assert config.fitness_op.rho == 0.251372, f"Expected 0.251372, got {config.fitness_op.rho}"

    print("  ✓ QFT configuration created with correct parameters")
    print(f"  ✓ Benchmark: {config.benchmark_name}")
    print(f"  ✓ N: {config.gas_params['N']}, n_steps: {config.n_steps}, dims: {config.dims}")
    print(f"  ✓ epsilon_F: {config.kinetic_op.epsilon_F}, nu: {config.kinetic_op.nu}")
    print(f"  ✓ viscous coupling: {config.kinetic_op.use_viscous_coupling}")
    print(f"  ✓ companion epsilon (diversity): {config.companion_selection.epsilon}")
    print(f"  ✓ companion epsilon (clone): {config.companion_selection_clone.epsilon}")


def test_potential_creation():
    """Test that potential is created correctly from QFT config."""
    print("\nTesting potential creation from QFT config...")

    config = GasConfigPanel.create_qft_config(dims=3, bounds_extent=10.0)

    # Verify potential is a QuadraticWell
    assert isinstance(config.potential, QuadraticWell), (
        f"Expected QuadraticWell, got {type(config.potential)}"
    )

    # Test potential evaluation
    x = torch.zeros(5, 3)
    result = config.potential(x)
    assert torch.allclose(result, torch.zeros(5)), "Potential at origin should be zero"

    print("  ✓ Potential created correctly")
    print(f"  ✓ Potential type: {type(config.potential).__name__}")


if __name__ == "__main__":
    print("=" * 60)
    print("QFT Configuration Setup Tests")
    print("=" * 60)

    test_quadratic_well_benchmark()
    test_qft_config_creation()
    test_potential_creation()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nTo run the QFT dashboard:")
    print("  python -m fragile.fractalai.experiments.gas_visualization_dashboard --qft")
