"""Demo script for BenchmarkSelector dashboard.

This script demonstrates how to use the BenchmarkSelector to explore
different benchmark potential functions and integrate them with SwarmExplorer.
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

from fragile.core.benchmarks import BenchmarkSelector, prepare_benchmark_for_explorer
from fragile.experiments.interactive_euclidean_gas import SwarmExplorer


def demo_benchmark_selector_standalone():
    """Launch standalone BenchmarkSelector dashboard."""
    print("Launching BenchmarkSelector dashboard...")
    print("This allows you to explore different benchmark potential functions.")
    print("You can:")
    print("  - Select from 9 benchmark functions")
    print("  - Adjust spatial dimension and bounds")
    print("  - Configure benchmark-specific parameters")
    print("  - Visualize the potential landscape")
    print()

    selector = BenchmarkSelector()
    dashboard = selector.panel()
    dashboard.show(port=5006, title="Benchmark Selector")


def demo_swarm_explorer_with_benchmarks():
    """Launch SwarmExplorer with selectable benchmarks."""
    print("Creating SwarmExplorer with benchmark selector...")
    print()

    # Create benchmark selector
    selector = BenchmarkSelector(benchmark_name="Mixture of Gaussians", n_gaussians=3)
    selector_dashboard = selector.panel()

    # Create SwarmExplorer using selected benchmark
    def create_explorer_panel(_):
        """Create SwarmExplorer panel from current benchmark selection."""
        benchmark = selector.get_benchmark()
        background = selector.get_background()
        mode_points = selector.get_mode_points()

        explorer = SwarmExplorer(
            potential=benchmark,
            background=background,
            mode_points=mode_points,
            dims=selector.dims,
            N=100,
            n_steps=100,
            auto_update=False,
        )
        return explorer.panel()

    # Watch for benchmark changes
    explorer_panel = pn.bind(
        create_explorer_panel,
        selector.param.benchmark_name,
    )

    # Combine into tabs
    tabs = pn.Tabs(
        ("Benchmark Selector", selector_dashboard),
        ("Swarm Explorer", explorer_panel),
    )

    tabs.show(port=5007, title="Swarm Explorer with Benchmarks")


def demo_quick_benchmark_comparison():
    """Create quick comparison of multiple benchmarks."""
    print("Creating benchmark comparison dashboard...")
    print()

    benchmarks_to_compare = [
        "Rastrigin",
        "Sphere",
        "Mixture of Gaussians",
        "Constant (Zero)",
    ]

    # Create explorers for each benchmark
    explorers = []
    for name in benchmarks_to_compare:
        benchmark, background, mode_points = prepare_benchmark_for_explorer(
            name, dims=2, n_gaussians=3 if name == "Mixture of Gaussians" else 1
        )

        explorer = SwarmExplorer(
            potential=benchmark,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=50,
            n_steps=50,
            auto_update=False,
        )
        explorers.append((name, explorer.panel()))

    # Create tabs for comparison
    tabs = pn.Tabs(*explorers)
    tabs.show(port=5008, title="Benchmark Comparison")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Selector Demo")
    parser.add_argument(
        "mode",
        choices=["selector", "explorer", "compare"],
        help="""
        selector: Launch standalone benchmark selector
        explorer: Launch SwarmExplorer with benchmark selection
        compare: Launch side-by-side benchmark comparison
        """,
    )

    args = parser.parse_args()

    if args.mode == "selector":
        demo_benchmark_selector_standalone()
    elif args.mode == "explorer":
        demo_swarm_explorer_with_benchmarks()
    elif args.mode == "compare":
        demo_quick_benchmark_comparison()
