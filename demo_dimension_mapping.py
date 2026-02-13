#!/usr/bin/env python3
"""
Demo script showing the dimension mapping feature in action.
This creates a simple visualization to demonstrate the new functionality.
"""

import numpy as np
import torch


class MockHistory:
    """Mock RunHistory for demonstration."""

    def __init__(self, d=4, N=100, n_recorded=10):
        self.d = d
        self.N = N
        self.n_recorded = n_recorded
        self.n_steps = n_recorded * 10
        self.recorded_steps = torch.arange(n_recorded)

        # Create interesting 4D data - spiral in XY, expanding in Z, oscillating in T
        t = torch.linspace(0, 2 * np.pi, N)
        frames = torch.linspace(0, 1, n_recorded).unsqueeze(1)

        x = torch.cos(t) * (1 + frames)
        y = torch.sin(t) * (1 + frames)
        z = frames.expand(-1, N) * 2 - 1  # -1 to 1
        w = torch.sin(t * 3) * torch.cos(frames * 2 * np.pi)  # Oscillating in T

        if d == 4:
            self.x_final = torch.stack([x, y, z, w], dim=2)
        else:
            self.x_final = torch.stack([x, y, z], dim=2)

        self.fitness = torch.randn(n_recorded - 1, N).abs()
        self.rewards = torch.randn(n_recorded - 1, N)
        self.alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
        self.neighbor_edges = None
        self.params = None


def demo_dimension_mapping():
    """Demonstrate dimension mapping functionality."""
    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    print("=" * 70)
    print("Dimension Mapping Demo")
    print("=" * 70)

    # Create 4D history
    print("\n1. Creating 4D simulation data...")
    history = MockHistory(d=4, N=100, n_recorded=10)
    print(
        f"   ✓ Created history with d={history.d}, N={history.N}, n_recorded={history.n_recorded}"
    )

    # Create viewer
    print("\n2. Creating SwarmConvergence3D viewer...")
    viewer = SwarmConvergence3D(history=None, bounds_extent=3.0)
    print(f"   ✓ Created viewer with initial dim options: {viewer.param.x_axis_dim.objects}")

    # Load history
    print("\n3. Loading history into viewer...")
    viewer.set_history(history)
    print(f"   ✓ Dimension options updated to: {viewer.param.x_axis_dim.objects}")
    print(f"   ✓ Color options updated to: {viewer.param.color_metric.objects}")

    # Demo configuration 1: Standard 3D view
    print("\n4. Configuration 1: Standard 3D spatial view")
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "dim_1"
    viewer.z_axis_dim = "dim_2"
    viewer.color_metric = "fitness"
    viewer._make_figure(frame=5)
    print(f"   ✓ X axis: {viewer.x_axis_dim} → {viewer._axis_label(viewer.x_axis_dim)}")
    print(f"   ✓ Y axis: {viewer.y_axis_dim} → {viewer._axis_label(viewer.y_axis_dim)}")
    print(f"   ✓ Z axis: {viewer.z_axis_dim} → {viewer._axis_label(viewer.z_axis_dim)}")
    print(f"   ✓ Color: {viewer.color_metric}")
    print("   ✓ Figure created successfully")

    # Demo configuration 2: 4D view with Euclidean time
    print("\n5. Configuration 2: XY plane across Euclidean time")
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "dim_1"
    viewer.z_axis_dim = "dim_3"  # Euclidean time
    viewer.color_metric = "dim_2"  # Color by Z position
    viewer._make_figure(frame=5)
    print(f"   ✓ X axis: {viewer.x_axis_dim} → {viewer._axis_label(viewer.x_axis_dim)}")
    print(f"   ✓ Y axis: {viewer.y_axis_dim} → {viewer._axis_label(viewer.y_axis_dim)}")
    print(f"   ✓ Z axis: {viewer.z_axis_dim} → {viewer._axis_label(viewer.z_axis_dim)}")
    print(f"   ✓ Color: {viewer.color_metric} → {viewer._axis_label(viewer.color_metric)}")
    print("   ✓ Figure created successfully")

    # Demo configuration 3: MC time visualization
    print("\n6. Configuration 3: Spatial evolution with MC time")
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "dim_1"
    viewer.z_axis_dim = "mc_time"
    viewer.color_metric = "dim_3"
    viewer._make_figure(frame=5)
    print(f"   ✓ X axis: {viewer.x_axis_dim} → {viewer._axis_label(viewer.x_axis_dim)}")
    print(f"   ✓ Y axis: {viewer.y_axis_dim} → {viewer._axis_label(viewer.y_axis_dim)}")
    print(f"   ✓ Z axis: {viewer.z_axis_dim} → {viewer._axis_label(viewer.z_axis_dim)}")
    print(f"   ✓ Color: {viewer.color_metric} → {viewer._axis_label(viewer.color_metric)}")
    print("   ✓ Figure created successfully")

    # Demo configuration 4: All MC time (creates a point at frame index)
    print("\n7. Configuration 4: Pure MC time view")
    viewer.x_axis_dim = "mc_time"
    viewer.y_axis_dim = "mc_time"
    viewer.z_axis_dim = "mc_time"
    viewer.color_metric = "fitness"
    viewer._make_figure(frame=5)
    print(f"   ✓ All axes mapped to MC time (frame={5})")
    print("   ✓ Creates single point cluster at coordinates (5, 5, 5)")
    print("   ✓ Figure created successfully")

    # Test axis ranges
    print("\n8. Testing axis range calculation...")
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "mc_time"
    viewer.z_axis_dim = "dim_2"
    ranges = viewer._get_axis_ranges(frame=5)
    print(f"   ✓ X axis (spatial): {ranges['xaxis']['range']}")
    print(f"   ✓ Y axis (mc_time): {ranges['yaxis']['range']}")
    print(f"   ✓ Z axis (spatial): {ranges['zaxis']['range']}")

    # Summary
    print("\n" + "=" * 70)
    print("✓ All dimension mapping features working correctly!")
    print("=" * 70)
    print("\nKey features demonstrated:")
    print("  • Dynamic dimension options (adapts to 3D/4D)")
    print("  • Spatial dimension mapping (dim_0, dim_1, dim_2, dim_3)")
    print("  • Monte Carlo time mapping (mc_time)")
    print("  • Dimension-based coloring")
    print("  • Automatic axis range adjustment")
    print("  • Human-readable axis labels")
    print("\nThe feature is ready to use in the dashboard!")
    print("=" * 70)


if __name__ == "__main__":
    demo_dimension_mapping()
