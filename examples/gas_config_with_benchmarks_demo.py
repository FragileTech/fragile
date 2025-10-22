"""Demo script for GasConfig with integrated benchmark selector.

This script demonstrates the new workflow where GasConfig includes
an integrated benchmark selector, eliminating the need to manually
create potentials.
"""

# ruff: noqa: INP001, E402
from pathlib import Path
import sys


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import holoviews as hv
import panel as pn


hv.extension("bokeh")
pn.extension()

from fragile.experiments.gas_config_dashboard import GasConfig


def demo_basic_workflow():
    """Launch GasConfig with integrated benchmark selector."""
    print("Launching GasConfig with integrated benchmark selector...")
    print()
    print("Features:")
    print("  - Select from 9 benchmark potentials")
    print("  - Configure benchmark-specific parameters")
    print("  - Run simulation with selected potential")
    print("  - No manual potential creation needed!")
    print()

    # Create config without providing a potential
    config = GasConfig(dims=2)

    # Create dashboard
    dashboard = config.panel()
    dashboard.show(port=5009, title="Gas Config with Benchmark Selector")


def demo_programmatic_control():
    """Demonstrate programmatic benchmark control."""
    print("Demonstrating programmatic benchmark control...")
    print()

    # Create config
    config = GasConfig(dims=2, N=100, n_steps=100)

    # Configure Mixture of Gaussians
    config.benchmark_name = "Mixture of Gaussians"
    config.n_gaussians = 5
    config.benchmark_seed = 42

    print(f"✓ Selected: {config.benchmark_name}")
    print(f"✓ Configured: n_gaussians={config.n_gaussians}, seed={config.benchmark_seed}")
    print()

    # Run simulation
    print("Running simulation...")
    history = config.run_simulation()
    print(f"✓ Completed: {history.n_steps} steps, {history.n_recorded} recorded")
    print(f"✓ Final alive: {history.n_alive[-1].item()}")


def demo_benchmark_comparison():
    """Create tabs comparing different benchmarks."""
    print("Creating benchmark comparison dashboard...")
    print()

    benchmarks = [
        ("Sphere", {}),
        ("Rastrigin", {}),
        ("Mixture of Gaussians", {"n_gaussians": 3}),
        ("Constant (Zero)", {}),
    ]

    tabs_content = []
    for name, kwargs in benchmarks:
        config = GasConfig(dims=2, N=50, n_steps=100)
        config.benchmark_name = name

        # Set benchmark-specific parameters
        if name == "Mixture of Gaussians":
            config.n_gaussians = kwargs.get("n_gaussians", 3)

        tabs_content.append((name, config.panel()))

    tabs = pn.Tabs(*tabs_content)
    tabs.show(port=5010, title="Benchmark Comparison")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="GasConfig with Integrated Benchmark Selector Demo"
    )
    parser.add_argument(
        "mode",
        choices=["basic", "programmatic", "comparison"],
        help="""
        basic: Launch interactive dashboard with benchmark selector
        programmatic: Run simulation programmatically with selected benchmark
        comparison: Launch side-by-side benchmark comparison
        """,
    )

    args = parser.parse_args()

    if args.mode == "basic":
        demo_basic_workflow()
    elif args.mode == "programmatic":
        demo_programmatic_control()
    elif args.mode == "comparison":
        demo_benchmark_comparison()
