"""Tests for Panel attribute safety in visualization components.

These tests ensure that HoloViews objects returned from rendering methods
do not have disallowed attributes that cause Panel errors like:
"String 'repr_markdown' is in the disallowed list of attribute names"
"""
import inspect

import holoviews as hv
import pytest
import torch

from fragile.bounds import TorchBounds
from fragile.core.benchmarks import Sphere
from fragile.core.cloning import CloneOperator
from fragile.core.companion_selection import CompanionSelection
from fragile.core.euclidean_gas import EuclideanGas
from fragile.core.fitness import FitnessOperator
from fragile.core.kinetic_operator import KineticOperator
from fragile.experiments.gas_visualization_dashboard import ConvergencePanel, GasVisualizer

# Initialize HoloViews for testing
hv.extension("bokeh")

# Disallowed attributes that cause Panel errors
DISALLOWED_ATTRS = ["trait_names", "ipython_display", "_getAttributeNames", "_repr_markdown_"]


@pytest.fixture
def test_simulation():
    """Create a small test simulation for testing."""
    bounds = TorchBounds(
        low=torch.full((2,), -5.0, dtype=torch.float32),
        high=torch.full((2,), 5.0, dtype=torch.float32),
    )
    benchmark = Sphere(dims=2)

    gas = EuclideanGas(
        N=30,
        d=2,
        companion_selection=CompanionSelection(method="softmax", epsilon=0.1, lambda_alg=1.0),
        potential=benchmark,
        kinetic_op=KineticOperator(
            gamma=1.0,
            beta=1.0,
            delta_t=0.1,
            potential=benchmark,
            device=torch.device("cpu"),
            dtype=torch.float32,
            bounds=bounds,
        ),
        cloning=CloneOperator(sigma_x=0.1, alpha_restitution=0.5, p_max=0.3, epsilon_clone=0.01),
        fitness_op=FitnessOperator(alpha=1.0, beta=0.1, eta=0.1, lambda_alg=1.0),
        bounds=bounds,
        device=torch.device("cpu"),
        dtype="float32",
        enable_cloning=True,
        enable_kinetic=True,
    )

    history = gas.run(n_steps=15)
    return history, benchmark


def check_object_safe(obj):
    """Check if an object is safe for Panel display.

    Returns True if safe, False if has disallowed attributes.
    """
    for attr in DISALLOWED_ATTRS:
        if hasattr(obj, attr):
            return False
    return True


def test_histogram_rendering_no_data(test_simulation):
    """Test histogram panes with no data (no history loaded)."""
    _, benchmark = test_simulation
    # Create visualizer without history
    visualizer = GasVisualizer(
        history=None,
        potential=benchmark,
        background=None,
        mode_points=None,
        bounds_extent=5.0,
    )

    # Test that histogram panes exist and are hidden (no data)
    assert len(visualizer.histogram_panes) == 6, "Should have 6 histogram panes"
    for metric, pane in visualizer.histogram_panes.items():
        assert check_object_safe(pane), f"{metric} pane has disallowed attributes: {type(pane)}"
        # Panes should be hidden when no data is available
        assert not pane.visible, f"{metric} pane should be hidden when no data"


def test_histogram_rendering_empty_selection(test_simulation):
    """Test histogram panes with no histograms enabled."""
    history, benchmark = test_simulation
    visualizer = GasVisualizer(
        history=history,
        potential=benchmark,
        background=None,
        mode_points=None,
        bounds_extent=5.0,
    )

    # Test with no histograms enabled
    visualizer.enabled_histograms = []
    # All panes should be hidden
    for metric, pane in visualizer.histogram_panes.items():
        assert check_object_safe(pane), f"{metric} pane has disallowed attributes: {type(pane)}"
        assert not pane.visible, f"{metric} pane should be hidden when disabled"


def test_histogram_rendering_normal(test_simulation):
    """Test histogram panes with normal histogram selection."""
    history, benchmark = test_simulation
    visualizer = GasVisualizer(
        history=history,
        potential=benchmark,
        background=None,
        mode_points=None,
        bounds_extent=5.0,
    )

    # Test with normal histograms
    visualizer.enabled_histograms = ["fitness", "reward"]
    # Check that enabled panes are visible and disabled ones are hidden
    for metric, pane in visualizer.histogram_panes.items():
        assert check_object_safe(pane), f"{metric} pane has disallowed attributes: {type(pane)}"
        if metric in ["fitness", "reward"]:
            assert pane.visible, f"{metric} pane should be visible when enabled"
        else:
            assert not pane.visible, f"{metric} pane should be hidden when disabled"


def test_convergence_panel_plot_pane_initial(test_simulation):
    """Test ConvergencePanel plot pane before computation."""
    history, benchmark = test_simulation
    panel = ConvergencePanel(
        history=history,
        potential=benchmark,
        benchmark=benchmark,
        bounds_extent=5.0,
    )

    # Check initial state - plot_pane is now a Column, check its objects
    assert isinstance(panel.plot_pane.objects, list), "plot_pane.objects should be a list"
    for obj in panel.plot_pane.objects:
        assert check_object_safe(obj), f"Initial plot pane object has disallowed attributes: {type(obj)}"


def test_convergence_panel_plot_pane_computed(test_simulation):
    """Test ConvergencePanel plot pane after computation."""
    history, benchmark = test_simulation
    panel = ConvergencePanel(
        history=history,
        potential=benchmark,
        benchmark=benchmark,
        bounds_extent=5.0,
    )

    # Compute metrics and update plots
    panel.compute_metrics()
    panel._update_plots()

    # Check after computation - plot_pane is now a Column, check its objects
    assert isinstance(panel.plot_pane.objects, list), "plot_pane.objects should be a list"
    for obj in panel.plot_pane.objects:
        assert check_object_safe(obj), f"Computed plot pane object has disallowed attributes: {type(obj)}"


def test_histogram_streaming_architecture(test_simulation):
    """Test that histogram architecture uses streaming instead of dynamic rendering."""
    history, benchmark = test_simulation
    visualizer = GasVisualizer(
        history=history,
        potential=benchmark,
        background=None,
        mode_points=None,
        bounds_extent=5.0,
    )

    # Verify streaming architecture components exist
    assert hasattr(visualizer, "histogram_panes"), "Should have histogram_panes dictionary"
    assert hasattr(visualizer, "_update_histogram_streams"), "Should have _update_histogram_streams method"
    assert not hasattr(visualizer, "_render_histograms"), "_render_histograms should not exist (replaced by streaming)"

    # Verify Shaolin Histogram instances exist (replaced old manual streams)
    assert hasattr(visualizer, "histogram_fitness"), "Should have histogram_fitness Shaolin Histogram"
    assert hasattr(visualizer, "histogram_distance"), "Should have histogram_distance Shaolin Histogram"
    assert hasattr(visualizer, "histogram_reward"), "Should have histogram_reward Shaolin Histogram"
    assert hasattr(visualizer, "histogram_hessian"), "Should have histogram_hessian Shaolin Histogram"
    assert hasattr(visualizer, "histogram_forces"), "Should have histogram_forces Shaolin Histogram"
    assert hasattr(visualizer, "histogram_velocity"), "Should have histogram_velocity Shaolin Histogram"

    # Verify they are Shaolin Histogram instances
    from fragile.shaolin.stream_plots import Histogram
    assert isinstance(visualizer.histogram_fitness, Histogram), "histogram_fitness should be Shaolin Histogram"


def test_no_hv_text_in_convergence_update(test_simulation):
    """Static code check: ensure no problematic hv.Text in _update_plots."""
    history, benchmark = test_simulation
    panel = ConvergencePanel(
        history=history,
        potential=benchmark,
        benchmark=benchmark,
        bounds_extent=5.0,
    )

    # Check source code
    source = inspect.getsource(panel._update_plots)
    # Should not have both hv.Text( and object = in same method
    has_text_creation = "hv.Text(" in source
    has_object_assignment = ".object =" in source

    if has_text_creation and has_object_assignment:
        pytest.fail(
            "_update_plots contains hv.Text assignment which causes Panel errors. "
            "Use string instead."
        )
