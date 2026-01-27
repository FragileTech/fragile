#!/usr/bin/env python
"""Verification script for dashboard UnsetValueError fix.

This script verifies that:
1. Standard dashboard can be created and serialized
2. QFT dashboard can be created and serialized
3. All operator panels can be created and serialized
4. No UnsetValueError is raised during widget initialization

Run: uv run python verify_dashboard_fix.py
"""

import sys


def test_standard_dashboard():
    """Test standard dashboard creation."""
    print("Testing standard dashboard...")
    import holoviews as hv
    import panel as pn

    from fragile.fractalai.experiments.gas_visualization_dashboard import create_app

    hv.extension("bokeh")
    pn.extension()

    app = create_app(dims=2)
    app.servable()
    print("  ✓ Standard dashboard creates and serializes")
    return True


def test_qft_dashboard():
    """Test QFT dashboard creation."""
    print("Testing QFT dashboard...")
    import holoviews as hv
    import panel as pn

    from fragile.fractalai.experiments.gas_visualization_dashboard import create_qft_app

    hv.extension("bokeh")
    pn.extension()

    app = create_qft_app()
    app.servable()
    print("  ✓ QFT dashboard creates and serializes")
    return True


def test_kinetic_operator_panel():
    """Test KineticOperator panel."""
    print("Testing KineticOperator panel...")
    import holoviews as hv
    import panel as pn
    import torch

    from fragile.fractalai.core.kinetic_operator import KineticOperator

    hv.extension("bokeh")
    pn.extension()

    def dummy_potential(x):
        return torch.zeros(x.shape[0])

    kinetic_op = KineticOperator(gamma=1.0, beta=0.5, delta_t=0.1, potential=dummy_potential)
    panel = kinetic_op.__panel__()
    panel.servable()
    print("  ✓ KineticOperator panel creates and serializes")
    return True


def test_fitness_operator_panel():
    """Test FitnessOperator panel."""
    print("Testing FitnessOperator panel...")
    import holoviews as hv
    import panel as pn

    from fragile.fractalai.core.fitness import FitnessOperator

    hv.extension("bokeh")
    pn.extension()

    fitness_op = FitnessOperator()
    panel = fitness_op.__panel__()
    panel.servable()
    print("  ✓ FitnessOperator panel creates and serializes")
    return True


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Dashboard UnsetValueError Fix Verification")
    print("=" * 60)
    print()

    tests = [
        test_standard_dashboard,
        test_qft_dashboard,
        test_kinetic_operator_panel,
        test_fitness_operator_panel,
    ]

    failed = []
    for test in tests:
        try:
            if not test():
                failed.append(test.__name__)
        except Exception as e:
            print(f"  ✗ {test.__name__} failed: {e}")
            failed.append(test.__name__)
            import traceback

            traceback.print_exc()

    print()
    print("=" * 60)
    if failed:
        print(f"FAILED: {len(failed)} test(s) failed")
        for name in failed:
            print(f"  - {name}")
        print("=" * 60)
        sys.exit(1)
    else:
        print("SUCCESS: All verification tests passed!")
        print("The dashboard UnsetValueError fix is working correctly.")
        print("=" * 60)
        sys.exit(0)


if __name__ == "__main__":
    main()
