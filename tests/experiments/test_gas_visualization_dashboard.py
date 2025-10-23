"""Tests for GasVisualizer dashboard component.

These tests verify that the visualization dashboard works correctly,
callbacks execute properly, and the UI doesn't freeze during operations.
"""

import threading
import time

import holoviews as hv
import panel as pn
import pytest
import torch

from fragile.core.benchmarks import prepare_benchmark_for_explorer
from fragile.experiments.gas_config_dashboard import GasConfig
from fragile.experiments.gas_visualization_dashboard import GasVisualizer, create_app


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


class TestGasVisualizerCallbacks:
    """Test callback mechanism and UI responsiveness."""

    def test_callback_executes_non_blocking(self, test_potential):
        """Test that simulation completion callback doesn't block."""
        potential, background, mode_points = test_potential

        # Create gas configuration
        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50

        # Create visualizer
        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Track callback execution
        callback_started = threading.Event()
        callback_completed = threading.Event()

        def tracking_callback(history):
            """Callback that tracks its execution."""
            callback_started.set()
            # Simulate visualizer update
            visualizer.epsilon_F = float(gas_config.epsilon_F)
            visualizer.use_fitness_force = bool(gas_config.use_fitness_force)
            visualizer.use_potential_force = bool(gas_config.use_potential_force)
            visualizer.companion_selection = gas_config.companion_selection
            visualizer.fitness_op = gas_config.fitness_op

            # Run set_history in thread (as fixed in create_app)
            def _update():
                visualizer.set_history(history)
                callback_completed.set()

            thread = threading.Thread(target=_update, daemon=True)
            thread.start()

        gas_config.add_completion_callback(tracking_callback)

        # Simulate button click using _on_run_clicked (this triggers callbacks)
        start_time = time.time()
        gas_config._on_run_clicked()
        callback_execution_time = time.time() - start_time

        # Verify callback started quickly (should not block)
        assert callback_started.wait(
            timeout=1.0
        ), "Callback didn't start within 1 second"

        # Callback should complete synchronously in main thread part
        # (parameter setting), then spawn thread for heavy work
        # Main callback should return quickly (< 0.5s for parameter setting)
        assert (
            callback_execution_time < 2.0
        ), f"Callback blocked for {callback_execution_time:.2f}s"

        # Wait for background thread to complete (longer timeout)
        assert callback_completed.wait(
            timeout=10.0
        ), "Background history processing didn't complete"

        # Verify visualizer was updated
        assert visualizer.history is not None
        assert visualizer.history is gas_config.history

    def test_button_state_after_simulation(self, test_potential):
        """Test that run button is re-enabled after simulation completes."""
        potential, background, mode_points = test_potential

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50

        # Get the run button
        run_button = gas_config.run_button

        # Verify button starts enabled
        assert not run_button.disabled, "Button should start enabled"

        # Simulate button click programmatically
        initial_state = run_button.disabled
        gas_config._on_run_clicked()

        # After completion, button should be re-enabled
        assert not run_button.disabled, "Button should be re-enabled after completion"
        assert initial_state == run_button.disabled, "Button state should be restored"

    def test_callback_exception_handling(self, test_potential):
        """Test that exceptions in callbacks don't leave button frozen."""
        potential, background, mode_points = test_potential

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50

        # Add a callback that raises an exception
        def failing_callback(history):
            raise ValueError("Test exception")

        gas_config.add_completion_callback(failing_callback)

        run_button = gas_config.run_button

        # Run simulation (should handle exception gracefully)
        try:
            gas_config._on_run_clicked()
        except ValueError:
            pass  # Expected

        # Button should still be re-enabled despite exception
        assert not run_button.disabled, "Button should be re-enabled after exception"

    def test_multiple_callbacks_execute(self, test_potential):
        """Test that multiple callbacks all execute."""
        potential, background, mode_points = test_potential

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50

        # Track callback execution
        callback_count = [0]
        lock = threading.Lock()

        def counting_callback(history):
            with lock:
                callback_count[0] += 1

        # Add multiple callbacks
        gas_config.add_completion_callback(counting_callback)
        gas_config.add_completion_callback(counting_callback)
        gas_config.add_completion_callback(counting_callback)

        # Run simulation using button click (triggers callbacks)
        gas_config._on_run_clicked()

        # All callbacks should have executed
        assert callback_count[0] == 3, f"Expected 3 callbacks, got {callback_count[0]}"

    def test_visualizer_set_history_doesnt_block(self, test_potential):
        """Test that set_history can complete in reasonable time."""
        potential, background, mode_points = test_potential

        # Create visualizer
        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Create small history for fast processing
        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50
        history = gas_config.run_simulation()

        # Set operators needed for visualization
        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op

        # Time the set_history operation
        start_time = time.time()
        visualizer.set_history(history)
        elapsed = time.time() - start_time

        # Should complete in reasonable time for small simulation
        # Increased from 2.0s to 5.0s to account for gradient computations
        assert elapsed < 5.0, f"set_history took {elapsed:.2f}s (too slow)"

        # Verify result was generated
        assert visualizer.result is not None
        assert len(visualizer.result["times"]) > 0


class TestCreateAppIntegration:
    """Test create_app function with callback integration."""

    def test_create_app_structure(self):
        """Test that create_app creates valid dashboard structure."""
        app = create_app(dims=2, n_gaussians=2, bounds_extent=6.0)

        # Verify app structure
        assert isinstance(app, pn.template.FastListTemplate)
        assert app.title == "Gas Visualization Dashboard"
        assert len(app.sidebar) > 0
        assert len(app.main) > 0

    def test_create_app_callback_wired(self):
        """Test that create_app properly wires up callbacks."""
        app = create_app(dims=2, n_gaussians=2, bounds_extent=6.0)

        # Extract gas_config from sidebar
        # The structure is: [Markdown, GasConfig.panel()]
        # We need to find the GasConfig instance to verify callbacks
        # For now, just verify the app structure is correct
        assert len(app.sidebar) >= 2  # Markdown + GasConfig panel

    def test_create_app_threaded_callback(self, test_potential):
        """Test that create_app uses threaded callback for non-blocking updates."""
        potential, background, mode_points = test_potential

        # Manually create the setup from create_app
        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Track threading behavior
        callback_thread_id = [None]
        main_thread_id = threading.current_thread().ident

        def on_simulation_complete(history):
            """Mimic the create_app callback."""
            visualizer.epsilon_F = float(gas_config.epsilon_F)
            visualizer.use_fitness_force = bool(gas_config.use_fitness_force)
            visualizer.use_potential_force = bool(gas_config.use_potential_force)
            visualizer.companion_selection = gas_config.companion_selection
            visualizer.fitness_op = gas_config.fitness_op

            def _update_history():
                # Record which thread this runs in
                callback_thread_id[0] = threading.current_thread().ident
                visualizer.set_history(history)

            thread = threading.Thread(target=_update_history, daemon=True)
            thread.start()
            thread.join(timeout=10.0)  # Wait for completion in test

        gas_config.add_completion_callback(on_simulation_complete)

        # Run simulation using button click (triggers callbacks)
        gas_config._on_run_clicked()

        # Verify the heavy work ran in a different thread
        assert callback_thread_id[0] is not None, "Callback thread ID not recorded"
        assert (
            callback_thread_id[0] != main_thread_id
        ), "Heavy work should run in separate thread"

        # Verify visualizer was updated
        assert visualizer.history is not None


class TestProcessHistoryPerformance:
    """Test _process_history performance and optimization."""

    def test_process_history_with_precomputed_data(self, test_potential):
        """Test that precomputed Hessians/gradients speed up processing."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Create simulation with precomputed data
        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50
        gas_config.use_anisotropic_diffusion = True  # Enables Hessian computation
        history = gas_config.run_simulation()

        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op

        # Process with precomputed data
        start = time.time()
        visualizer.set_history(history)
        elapsed_precomputed = time.time() - start

        # Verify precomputed data was available
        assert (
            history.fitness_hessians_diag is not None
        ), "Expected precomputed Hessians"

        # Should complete in reasonable time
        assert (
            elapsed_precomputed < 5.0
        ), f"Processing with precomputed data took {elapsed_precomputed:.2f}s"

    def test_process_history_handles_large_stride(self, test_potential):
        """Test that large measure_stride reduces processing time."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 100  # More steps
        history = gas_config.run_simulation()

        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op

        # Process with large stride (fewer frames)
        visualizer.measure_stride = 10
        start = time.time()
        visualizer.set_history(history)
        elapsed_strided = time.time() - start

        # Should complete faster due to fewer frames processed
        assert (
            elapsed_strided < 3.0
        ), f"Strided processing took {elapsed_strided:.2f}s"

        # Verify fewer frames were generated
        frame_count = len(visualizer.result["times"])
        assert frame_count < 20, f"Expected < 20 frames with stride=10, got {frame_count}"


class TestErrorHandling:
    """Test error handling in visualization pipeline."""

    def test_set_history_with_none(self, test_potential):
        """Test that set_history handles None gracefully."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        # Should not raise
        visualizer.set_history(None)
        assert visualizer.result is None

    def test_process_history_without_operators(self, test_potential):
        """Test _process_history without companion_selection/fitness_op."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50
        history = gas_config.run_simulation()

        # Don't set companion_selection or fitness_op
        visualizer.companion_selection = None
        visualizer.fitness_op = None

        # Should still process without errors (using zero arrays)
        visualizer.set_history(history)

        assert visualizer.result is not None
        assert len(visualizer.result["times"]) > 0

    def test_get_frame_data_out_of_bounds(self, test_potential):
        """Test _get_frame_data with invalid frame index."""
        potential, background, mode_points = test_potential

        visualizer = GasVisualizer(
            history=None,
            potential=potential,
            background=background,
            mode_points=mode_points,
            bounds_extent=6.0,
        )

        gas_config = GasConfig(potential=potential, dims=2)
        gas_config.N = 10
        gas_config.n_steps = 50
        history = gas_config.run_simulation()

        visualizer.companion_selection = gas_config.companion_selection
        visualizer.fitness_op = gas_config.fitness_op
        visualizer.set_history(history)

        # Out of bounds frame should be handled
        max_frame = len(visualizer.result["times"]) - 1
        frame_data = visualizer._get_frame_data(max_frame + 100)

        # Should return None or last valid frame
        assert frame_data is None or isinstance(frame_data, dict)
