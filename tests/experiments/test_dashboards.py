"""Tests for interactive dashboards and visualizers.

These tests verify dashboard functionality programmatically without requiring
manual interaction or server deployment.
"""

import holoviews as hv
import panel as pn
import pytest
import torch

from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.gas_config_dashboard import GasConfig
from fragile.experiments.gas_visualization_dashboard import create_app, GasVisualizer
from fragile.experiments.interactive_euclidean_gas import (
    create_dashboard,
    SwarmExplorer,
)


# Initialize extensions once for all tests
hv.extension("bokeh")
pn.extension()


@pytest.fixture
def test_potential():
    """Create a simple test potential for fast testing."""
    potential, background, mode_points = prepare_benchmark_for_explorer(
        benchmark_name="Mixture of Gaussians",
        dims=2,
        bounds_range=(-6.0, 6.0),
        resolution=50,  # Lower resolution for faster tests
        n_gaussians=2,  # Fewer modes for faster tests
        seed=42,
    )
    return potential, background, mode_points


class TestGasConfig:
    """Test GasConfig parameter dashboard."""

    def test_gas_config_initialization(self, test_potential):
        """Test GasConfig can be initialized with a potential."""
        potential, _, _ = test_potential
        config = GasConfig(potential=potential, dims=2)

        assert config.dims == 2
        assert config.potential is not None
        assert config.history is None  # No simulation run yet

    def test_gas_config_run_simulation(self, test_potential):
        """Test running a simulation through GasConfig."""
        potential, _, _ = test_potential
        config = GasConfig(potential=potential, dims=2)

        # Override parameters for fast test
        config.N = 10  # Small swarm
        config.n_steps = 50  # Fast test with minimum steps

        # Run simulation
        history = config.run_simulation()

        # Verify history
        assert history is not None
        assert history.n_steps == 50
        assert history.N == 10
        assert history.x_before_clone.shape[0] == 51  # n_recorded = n_steps + 1 (initial state)

    def test_gas_config_panel_creation(self, test_potential):
        """Test creating Panel dashboard from GasConfig."""
        potential, _, _ = test_potential
        config = GasConfig(potential=potential, dims=2)

        # Create panel
        panel = config.panel()

        # Verify panel structure
        assert isinstance(panel, pn.Column)
        assert len(panel) > 0  # Has components

    def test_gas_config_completion_callback(self, test_potential):
        """Test simulation completion callback mechanism."""
        potential, _, _ = test_potential
        config = GasConfig(potential=potential, dims=2)
        config.N = 10
        config.n_steps = 50

        # Add callback to track completion
        callback_called = []

        def test_callback(history):
            callback_called.append(history)

        config.add_completion_callback(test_callback)

        # Run simulation
        history = config.run_simulation()

        # Verify callback was called
        assert len(callback_called) == 1
        assert callback_called[0] is history

    def test_gas_config_stores_operators(self, test_potential):
        """Test that GasConfig stores operators after simulation."""
        potential, _, _ = test_potential
        config = GasConfig(potential=potential, dims=2)
        config.N = 10
        config.n_steps = 50

        # Run simulation
        config.run_simulation()

        # Verify operators are stored
        assert hasattr(config, "companion_selection")
        assert hasattr(config, "clone_op")
        assert hasattr(config, "fitness_op")
        assert config.fitness_op is not None


class TestGasVisualizer:
    """Test GasVisualizer display component."""

    def test_gas_visualizer_initialization(self, test_potential):
        """Test GasVisualizer can be initialized."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        assert visualizer.history is None
        assert visualizer.potential is not None
        assert visualizer.background is not None

    def test_gas_visualizer_with_history(self, test_potential):
        """Test GasVisualizer with simulation history."""
        potential, background, mode_points = test_potential

        # Create and run simulation
        config = GasConfig(potential=potential, dims=2)
        config.N = 10
        config.n_steps = 50
        history = config.run_simulation()

        # Create visualizer
        visualizer = GasVisualizer(
            history=history,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Set required operators for visualization
        visualizer.companion_selection = config.companion_selection
        visualizer.fitness_op = config.fitness_op

        assert visualizer.history is not None
        assert visualizer.history.n_steps == 50

    def test_gas_visualizer_panel_creation(self, test_potential):
        """Test creating Panel display from GasVisualizer."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Create panel
        panel = visualizer.panel()

        # Verify panel structure
        assert isinstance(panel, pn.Row)

    def test_gas_visualizer_frame_data_retrieval(self, test_potential):
        """Test retrieving frame data for visualization."""
        potential, background, mode_points = test_potential

        # Create and run simulation
        config = GasConfig(potential=potential, dims=2)
        config.N = 10
        config.n_steps = 50
        history = config.run_simulation()

        # Create visualizer with operators
        visualizer = GasVisualizer(
            history=history,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )
        visualizer.companion_selection = config.companion_selection
        visualizer.fitness_op = config.fitness_op

        # Get frame data
        frame_data = visualizer._get_frame_data(0)

        # Verify frame data structure (using actual API keys)
        assert frame_data is not None
        assert "positions" in frame_data
        assert "velocity_vals" in frame_data
        assert "reward_vals" in frame_data
        assert "data" in frame_data
        assert frame_data["positions"].shape == (10, 2)


class TestSwarmExplorer:
    """Test SwarmExplorer integrated dashboard."""

    def test_swarm_explorer_initialization(self, test_potential):
        """Test SwarmExplorer initialization."""
        potential, background, mode_points = test_potential

        explorer = SwarmExplorer(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=10,
            n_steps=50,
        )

        assert explorer.dims == 2
        assert explorer.config is not None
        assert explorer.visualizer is not None

    def test_swarm_explorer_panel_creation(self, test_potential):
        """Test creating Panel from SwarmExplorer."""
        potential, background, mode_points = test_potential

        explorer = SwarmExplorer(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=10,
            n_steps=50,
        )

        # Create panel
        panel = explorer.panel()

        # Verify panel structure
        assert isinstance(panel, pn.Row)
        assert len(panel) == 2  # Config sidebar + Visualizer main

    def test_swarm_explorer_simulation_update(self, test_potential):
        """Test that simulation updates are properly connected."""
        potential, background, mode_points = test_potential

        explorer = SwarmExplorer(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=10,
            n_steps=50,
        )

        # Run simulation through config
        history = explorer.config.run_simulation()

        # Verify visualizer received the history
        assert explorer.visualizer.history is not None
        assert explorer.visualizer.history is history


class TestCreateDashboard:
    """Test create_dashboard factory function."""

    def test_create_dashboard_with_defaults(self):
        """Test creating dashboard with default parameters."""
        explorer, panel = create_dashboard(dims=2)

        assert isinstance(explorer, SwarmExplorer)
        assert isinstance(panel, pn.Row)
        assert explorer.potential is not None

    def test_create_dashboard_with_custom_potential(self, test_potential):
        """Test creating dashboard with custom potential."""
        potential, background, mode_points = test_potential

        explorer, _panel = create_dashboard(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
        )

        assert isinstance(explorer, SwarmExplorer)
        assert explorer.potential is potential

    def test_create_dashboard_with_explorer_params(self):
        """Test creating dashboard with custom explorer parameters."""
        explorer, _panel = create_dashboard(
            dims=2,
            explorer_params={"N": 20, "n_steps": 50},
        )

        assert explorer.config.N == 20
        assert explorer.config.n_steps == 50


class TestCreateApp:
    """Test create_app function for full application."""

    def test_create_app_basic(self):
        """Test creating full app with default parameters."""
        app = create_app(dims=2, n_gaussians=2, bounds_extent=6.0)

        # Verify app structure
        assert isinstance(app, pn.template.FastListTemplate)
        assert app.title == "Gas Visualization Dashboard"
        assert len(app.sidebar) > 0
        assert len(app.main) > 0

    def test_create_app_custom_params(self):
        """Test creating app with custom parameters."""
        app = create_app(dims=2, n_gaussians=3, bounds_extent=8.0)

        assert isinstance(app, pn.template.FastListTemplate)


class TestDashboardIntegration:
    """Integration tests for complete dashboard workflow."""

    def test_full_workflow(self, test_potential):
        """Test complete workflow: create, configure, run, visualize."""
        potential, background, mode_points = test_potential

        # 1. Create explorer
        explorer = SwarmExplorer(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=10,
            n_steps=50,
        )

        # 2. Configure parameters
        explorer.config.gamma = 2.0
        explorer.config.beta = 2.0

        # 3. Run simulation
        history = explorer.config.run_simulation()

        # 4. Verify visualization updated
        assert explorer.visualizer.history is history

        # 5. Get frame data for visualization
        frame_data = explorer.visualizer._get_frame_data(0)
        assert frame_data is not None

    def test_parameter_persistence_across_runs(self, test_potential):
        """Test that parameters persist across multiple runs."""
        potential, background, mode_points = test_potential

        explorer = SwarmExplorer(
            potential=potential,
            background=background,
            mode_points=mode_points,
            dims=2,
            N=10,
            n_steps=50,
        )

        # Set custom parameters
        explorer.config.gamma = 3.0
        explorer.config.sigma_x = 0.5

        # Run simulation
        history1 = explorer.config.run_simulation()

        # Parameters should persist
        assert explorer.config.gamma == 3.0
        assert explorer.config.sigma_x == 0.5

        # Run again
        history2 = explorer.config.run_simulation()

        # New history should be different object
        assert history2 is not history1

        # But parameters should still be the same
        assert explorer.config.gamma == 3.0
        assert explorer.config.sigma_x == 0.5
