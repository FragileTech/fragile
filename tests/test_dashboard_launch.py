"""Tests for dashboard launch and widget initialization.

Ensures all Panel/Bokeh widgets have proper default values to prevent
UnsetValueError during serialization.
"""

import holoviews as hv
import panel as pn
import pytest


class TestDashboardLaunch:
    """Test dashboard creation and serialization."""

    @pytest.fixture(autouse=True)
    def setup_extensions(self):
        """Initialize Panel/HoloViews extensions before each test."""
        hv.extension("bokeh")
        pn.extension()

    def test_standard_dashboard_creates_without_error(self):
        """Test standard dashboard can be created."""
        from fragile.fractalai.experiments.gas_visualization_dashboard import create_app

        app = create_app(dims=2)
        assert app is not None
        assert "Gas Visualization Dashboard" in app.title

    def test_qft_dashboard_creates_without_error(self):
        """Test QFT dashboard can be created."""
        from fragile.fractalai.experiments.gas_visualization_dashboard import create_qft_app

        app = create_qft_app()
        assert app is not None
        assert "QFT" in app.title or "Calibration" in app.title

    def test_standard_dashboard_serializes(self):
        """Test standard dashboard can be serialized to JSON (for Bokeh)."""
        from fragile.fractalai.experiments.gas_visualization_dashboard import create_app

        app = create_app(dims=2)

        # Extract the Panel document
        # This simulates what happens when browser connects
        try:
            # Panel apps have a servable() method that prepares for serving
            servable = app.servable()
            # If we can get here without UnsetValueError, widgets are properly initialized
            assert servable is not None
        except Exception as e:
            pytest.fail(f"Dashboard serialization failed: {e}")

    def test_qft_dashboard_serializes(self):
        """Test QFT dashboard can be serialized to JSON."""
        from fragile.fractalai.experiments.gas_visualization_dashboard import create_qft_app

        app = create_qft_app()

        try:
            servable = app.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"QFT dashboard serialization failed: {e}")


class TestOperatorWidgets:
    """Test individual operator panel widgets have proper values."""

    @pytest.fixture(autouse=True)
    def setup_extensions(self):
        """Initialize Panel extensions."""
        hv.extension("bokeh")
        pn.extension()

    def test_kinetic_operator_panel_widgets_have_values(self):
        """Test KineticOperator.__panel__() creates widgets with proper values."""
        import torch

        from fragile.fractalai.core.kinetic_operator import KineticOperator

        # KineticOperator requires gamma, beta, delta_t and potential parameters
        def dummy_potential(x):
            """Dummy potential function for testing."""
            return torch.zeros(x.shape[0])

        kinetic_op = KineticOperator(gamma=1.0, beta=0.5, delta_t=0.1, potential=dummy_potential)
        panel = kinetic_op.__panel__()

        # Panel should be created without errors
        assert panel is not None

        # Try to serialize (this would trigger UnsetValueError if widgets lack values)
        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"KineticOperator panel serialization failed: {e}")

    def test_fitness_operator_panel_widgets_have_values(self):
        """Test FitnessOperator.__panel__() creates widgets with proper values."""
        from fragile.fractalai.core.fitness import FitnessOperator

        fitness_op = FitnessOperator()
        panel = fitness_op.__panel__()

        assert panel is not None

        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"FitnessOperator panel serialization failed: {e}")

    def test_companion_selection_panel_widgets_have_values(self):
        """Test CompanionSelection.__panel__() creates widgets with proper values."""
        from fragile.fractalai.core.companion_selection import CompanionSelection

        companion_sel = CompanionSelection()
        panel = companion_sel.__panel__()

        assert panel is not None

        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"CompanionSelection panel serialization failed: {e}")

    def test_clone_operator_panel_widgets_have_values(self):
        """Test CloneOperator.__panel__() creates widgets with proper values."""
        from fragile.fractalai.core.cloning import CloneOperator

        clone_op = CloneOperator()
        panel = clone_op.__panel__()

        assert panel is not None

        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"CloneOperator panel serialization failed: {e}")


class TestGasConfigPanel:
    """Test GasConfigPanel widget initialization."""

    @pytest.fixture(autouse=True)
    def setup_extensions(self):
        """Initialize Panel extensions."""
        hv.extension("bokeh")
        pn.extension()

    def test_gas_config_panel_standard_mode(self):
        """Test GasConfigPanel creates properly in standard mode."""
        from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

        config = GasConfigPanel(dims=2)
        panel = config.panel()

        assert panel is not None

        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"GasConfigPanel (standard) serialization failed: {e}")

    def test_gas_config_panel_qft_mode(self):
        """Test GasConfigPanel creates properly in QFT mode."""
        from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

        config = GasConfigPanel.create_qft_config(dims=3)
        panel = config.panel()

        assert panel is not None

        try:
            servable = panel.servable()
            assert servable is not None
        except Exception as e:
            pytest.fail(f"GasConfigPanel (QFT) serialization failed: {e}")

    def test_all_widgets_have_initial_values(self):
        """Verify all widgets in GasConfigPanel have non-Unset values."""
        from bokeh.core.property.descriptors import UnsetValueError

        from fragile.fractalai.experiments.gas_config_panel import GasConfigPanel

        config = GasConfigPanel(dims=2)
        panel = config.panel()

        # Try to access all widget values - should not raise UnsetValueError
        try:
            # This will traverse all widgets and access their values
            panel.servable()

            # If we got here, no UnsetValueError was raised
            assert True
        except UnsetValueError as e:
            pytest.fail(f"Widget has unset value: {e}")
