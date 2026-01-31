#!/usr/bin/env python3
"""
Test script for the dimension mapping implementation in SwarmConvergence3D.
"""
import numpy as np
import torch

# Mock the RunHistory class with minimal required attributes
class MockHistory:
    def __init__(self, d=3, N=100, n_recorded=10, n_steps=100):
        self.d = d
        self.N = N
        self.n_recorded = n_recorded
        self.n_steps = n_steps
        self.recorded_steps = torch.arange(n_recorded)

        # Create mock data
        self.x_final = torch.randn(n_recorded, N, d)
        self.fitness = torch.randn(n_recorded - 1, N)
        self.rewards = torch.randn(n_recorded - 1, N)
        self.alive_mask = torch.ones(n_recorded - 1, N, dtype=torch.bool)
        self.neighbor_edges = None
        self.params = None


def test_dimension_extraction():
    """Test the _extract_dimension method."""
    print("Testing dimension extraction...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    # Create a 4D history
    history = MockHistory(d=4, N=50, n_recorded=5)
    viewer = SwarmConvergence3D(history=history)

    # Test data
    positions_all = np.random.randn(50, 4)  # [N, d]
    alive = np.ones(50, dtype=bool)
    alive[40:] = False  # Only first 40 are alive
    frame = 2

    # Test spatial dimension extraction
    dim_0 = viewer._extract_dimension("dim_0", frame, positions_all, alive)
    assert dim_0.shape == (40,), f"Expected shape (40,), got {dim_0.shape}"
    assert np.allclose(dim_0, positions_all[alive, 0])
    print("  ✓ Spatial dimension extraction works")

    # Test MC time extraction
    mc_time = viewer._extract_dimension("mc_time", frame, positions_all, alive)
    assert mc_time.shape == (40,), f"Expected shape (40,), got {mc_time.shape}"
    assert np.all(mc_time == frame), "MC time should all equal frame index"
    print("  ✓ MC time extraction works")

    # Test invalid dimension
    invalid = viewer._extract_dimension("dim_10", frame, positions_all, alive)
    assert invalid.shape == (40,), f"Expected shape (40,), got {invalid.shape}"
    assert np.all(invalid == 0), "Invalid dimension should return zeros"
    print("  ✓ Invalid dimension handling works")


def test_dimension_options():
    """Test that dimension options update correctly."""
    print("\nTesting dimension options...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    # Test 3D history
    history_3d = MockHistory(d=3)
    viewer_3d = SwarmConvergence3D(history=None)
    viewer_3d.set_history(history_3d)

    expected_3d = ["dim_0", "dim_1", "dim_2", "mc_time"]
    assert viewer_3d.param.x_axis_dim.objects == expected_3d
    assert viewer_3d.param.y_axis_dim.objects == expected_3d
    assert viewer_3d.param.z_axis_dim.objects == expected_3d
    print("  ✓ 3D dimension options correct")

    # Test 4D history
    history_4d = MockHistory(d=4)
    viewer_4d = SwarmConvergence3D(history=None)
    viewer_4d.set_history(history_4d)

    expected_4d = ["dim_0", "dim_1", "dim_2", "dim_3", "mc_time"]
    assert viewer_4d.param.x_axis_dim.objects == expected_4d
    assert viewer_4d.param.y_axis_dim.objects == expected_4d
    assert viewer_4d.param.z_axis_dim.objects == expected_4d
    print("  ✓ 4D dimension options correct")

    # Test color options
    expected_color = expected_4d + ["fitness", "reward", "radius", "constant"]
    assert viewer_4d.param.color_metric.objects == expected_color
    print("  ✓ Color metric options correct")


def test_color_values():
    """Test the _get_color_values method."""
    print("\nTesting color value extraction...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    # Create a 4D history
    history = MockHistory(d=4, N=50, n_recorded=5)
    viewer = SwarmConvergence3D(history=history)

    # Test data
    positions_all = np.random.randn(50, 4)
    alive = np.ones(50, dtype=bool)
    alive[40:] = False
    frame = 2

    # Test dimension-based coloring
    viewer.color_metric = "dim_2"
    colors, showscale, colorbar = viewer._get_color_values(frame, positions_all, alive)
    assert colors.shape == (40,), f"Expected shape (40,), got {colors.shape}"
    assert showscale is True
    assert colorbar is not None
    print("  ✓ Dimension-based coloring works")

    # Test MC time coloring
    viewer.color_metric = "mc_time"
    colors, showscale, colorbar = viewer._get_color_values(frame, positions_all, alive)
    assert colors.shape == (40,), f"Expected shape (40,), got {colors.shape}"
    assert np.all(colors == frame)
    assert showscale is True
    print("  ✓ MC time coloring works")

    # Test radius coloring
    viewer.color_metric = "radius"
    colors, showscale, colorbar = viewer._get_color_values(frame, positions_all, alive)
    assert colors.shape == (40,), f"Expected shape (40,), got {colors.shape}"
    assert showscale is True
    print("  ✓ Radius coloring works")


def test_axis_ranges():
    """Test the _get_axis_ranges method."""
    print("\nTesting axis range calculation...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    history = MockHistory(d=4, N=50, n_recorded=5)
    viewer = SwarmConvergence3D(history=history, bounds_extent=15.0)

    # Test spatial dimensions
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "dim_1"
    viewer.z_axis_dim = "dim_2"
    ranges = viewer._get_axis_ranges(frame=2)

    assert ranges["xaxis"]["range"] == [-15.0, 15.0]
    assert ranges["yaxis"]["range"] == [-15.0, 15.0]
    assert ranges["zaxis"]["range"] == [-15.0, 15.0]
    print("  ✓ Spatial dimension ranges correct")

    # Test MC time range
    viewer.z_axis_dim = "mc_time"
    ranges = viewer._get_axis_ranges(frame=2)
    assert ranges["zaxis"]["range"] == [0, 4]  # n_recorded - 1
    print("  ✓ MC time range correct")


def test_axis_labels():
    """Test the _axis_label method."""
    print("\nTesting axis label generation...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    history = MockHistory(d=4)
    viewer = SwarmConvergence3D(history=history)

    # Test spatial dimensions
    assert viewer._axis_label("dim_0") == "Dimension 0 (X)"
    assert viewer._axis_label("dim_1") == "Dimension 1 (Y)"
    assert viewer._axis_label("dim_2") == "Dimension 2 (Z)"
    assert viewer._axis_label("dim_3") == "Dimension 3 (T)"
    print("  ✓ Spatial dimension labels correct")

    # Test MC time
    assert viewer._axis_label("mc_time") == "Monte Carlo Time (frame)"
    print("  ✓ MC time label correct")


def test_figure_creation():
    """Test that figure creation works with dimension mapping."""
    print("\nTesting figure creation...")

    from fragile.fractalai.qft.dashboard import SwarmConvergence3D

    # Create a 4D history
    history = MockHistory(d=4, N=50, n_recorded=5)
    viewer = SwarmConvergence3D(history=history)

    # Test with default mapping
    fig = viewer._make_figure(frame=2)
    assert fig is not None
    print("  ✓ Default mapping figure creation works")

    # Test with MC time mapping
    viewer.x_axis_dim = "dim_0"
    viewer.y_axis_dim = "dim_1"
    viewer.z_axis_dim = "mc_time"
    viewer.color_metric = "dim_3"
    fig = viewer._make_figure(frame=2)
    assert fig is not None
    print("  ✓ MC time mapping figure creation works")

    # Test with all MC time
    viewer.x_axis_dim = "mc_time"
    viewer.y_axis_dim = "mc_time"
    viewer.z_axis_dim = "mc_time"
    fig = viewer._make_figure(frame=2)
    assert fig is not None
    print("  ✓ All MC time mapping figure creation works")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SwarmConvergence3D Dimension Mapping Implementation")
    print("=" * 60)

    try:
        test_dimension_extraction()
        test_dimension_options()
        test_color_values()
        test_axis_ranges()
        test_axis_labels()
        test_figure_creation()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)
