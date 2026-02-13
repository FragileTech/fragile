"""Tests for the mass correlator plotting module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.mass_correlator_plots import (
    ChannelCorrelatorResult,
    ChannelDefinition,
    ChannelProjector,
    compute_all_channel_correlators,
    MassCorrelatorComputer,
    MassCorrelatorConfig,
    MassCorrelatorPlotter,
    STANDARD_CHANNELS,
)


class TestStandardChannels:
    """Tests for standard channel definitions."""

    def test_all_channels_defined(self):
        """All standard lattice QFT channels should be defined."""
        expected = {
            "scalar",
            "pseudoscalar",
            "vector",
            "axial_vector",
            "tensor",
            "nucleon",
            "glueball",
        }
        assert set(STANDARD_CHANNELS.keys()) == expected

    def test_channel_quantum_numbers(self):
        """Each channel should have valid quantum numbers."""
        for name, channel in STANDARD_CHANNELS.items():
            assert isinstance(channel, ChannelDefinition)
            assert channel.name == name
            assert isinstance(channel.quantum_numbers, dict)
            assert channel.color.startswith("#")

    def test_pion_channel_jp(self):
        """Pion (pseudoscalar) should have J=0, P=-."""
        pion = STANDARD_CHANNELS["pseudoscalar"]
        assert pion.quantum_numbers["J"] == 0
        assert pion.quantum_numbers["P"] == "-"

    def test_nucleon_baryon_number(self):
        """Nucleon should have B=1 (baryon number)."""
        nucleon = STANDARD_CHANNELS["nucleon"]
        assert nucleon.quantum_numbers.get("B") == 1


class TestChannelProjector:
    """Tests for gamma matrix projections."""

    @pytest.fixture
    def projector_3d(self):
        """Create a 3D channel projector."""
        return ChannelProjector(dim=3, device="cpu")

    def test_gamma_matrices_built(self, projector_3d):
        """Projector should have all gamma matrices."""
        assert "1" in projector_3d.gamma  # Identity
        assert "5" in projector_3d.gamma  # γ₅

    def test_scalar_projection(self, projector_3d):
        """Scalar projection should be identity-like."""
        color_i = torch.randn(10, 3, dtype=torch.complex128)
        color_j = torch.randn(10, 3, dtype=torch.complex128)

        result = projector_3d.project_bilinear(color_i, color_j, "scalar")
        assert result.shape == (10,)
        assert result.dtype == torch.complex128

    def test_pseudoscalar_projection(self, projector_3d):
        """Pseudoscalar projection should use γ₅."""
        color_i = torch.randn(10, 3, dtype=torch.complex128)
        color_j = torch.randn(10, 3, dtype=torch.complex128)

        result = projector_3d.project_bilinear(color_i, color_j, "pseudoscalar")
        assert result.shape == (10,)

    def test_vector_projection(self, projector_3d):
        """Vector projection should average over spatial directions."""
        color_i = torch.randn(10, 3, dtype=torch.complex128)
        color_j = torch.randn(10, 3, dtype=torch.complex128)

        result = projector_3d.project_bilinear(color_i, color_j, "vector")
        assert result.shape == (10,)


class TestMassCorrelatorConfig:
    """Tests for configuration dataclass."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = MassCorrelatorConfig()
        assert config.warmup_fraction == 0.1
        assert config.h_eff == 1.0
        assert "scalar" in config.channels
        assert "pseudoscalar" in config.channels

    def test_custom_channels(self):
        """Should accept custom channel list."""
        config = MassCorrelatorConfig(
            channels=("scalar", "nucleon"),
        )
        assert len(config.channels) == 2
        assert "nucleon" in config.channels


class TestChannelCorrelatorResult:
    """Tests for correlator result container."""

    def test_result_creation(self):
        """Should create result with all fields."""
        channel = STANDARD_CHANNELS["scalar"]
        result = ChannelCorrelatorResult(
            channel=channel,
            lags=np.arange(10),
            lag_times=np.arange(10) * 0.1,
            correlator=np.exp(-np.arange(10) * 0.5),
            correlator_err=None,
            effective_mass=np.ones(9) * 0.5,
            mass_fit={"mass": 0.5, "r_squared": 0.99},
            n_samples=100,
        )
        assert result.n_samples == 100
        assert result.mass_fit["mass"] == 0.5


class TestMassCorrelatorPlotter:
    """Tests for HoloViews plot generation."""

    @pytest.fixture
    def sample_results(self):
        """Create sample correlator results for plotting."""
        results = {}
        for name in ["scalar", "pseudoscalar"]:
            channel = STANDARD_CHANNELS[name]
            lags = np.arange(20)
            lag_times = lags * 0.1
            # Exponential decay with some noise
            correlator = np.exp(-lags * 0.3) * (1 + 0.1 * np.random.randn(20))
            correlator = np.maximum(correlator, 1e-10)  # Ensure positive
            eff_mass = np.ones(19) * 0.3

            results[name] = ChannelCorrelatorResult(
                channel=channel,
                lags=lags,
                lag_times=lag_times,
                correlator=correlator,
                correlator_err=None,
                effective_mass=eff_mass,
                mass_fit={"mass": 0.3, "amplitude": 1.0, "r_squared": 0.95},
                n_samples=50,
            )
        return results

    def test_plotter_creation(self, sample_results):
        """Plotter should accept results dictionary."""
        plotter = MassCorrelatorPlotter(sample_results)
        assert len(plotter.results) == 2

    def test_correlator_plot(self, sample_results):
        """Should build correlator plot for valid channel."""
        plotter = MassCorrelatorPlotter(sample_results)
        plot = plotter.build_correlator_plot("scalar")
        assert plot is not None

    def test_effective_mass_plot(self, sample_results):
        """Should build effective mass plot."""
        plotter = MassCorrelatorPlotter(sample_results)
        plot = plotter.build_effective_mass_plot("scalar")
        assert plot is not None

    def test_all_correlators_overlay(self, sample_results):
        """Should build overlay of all channels."""
        plotter = MassCorrelatorPlotter(sample_results)
        plot = plotter.build_all_correlators_overlay()
        assert plot is not None

    def test_mass_spectrum_bar(self, sample_results):
        """Should build mass spectrum bar chart."""
        plotter = MassCorrelatorPlotter(sample_results)
        plot = plotter.build_mass_spectrum_bar()
        assert plot is not None

    def test_dashboard(self, sample_results):
        """Should build full dashboard layout."""
        plotter = MassCorrelatorPlotter(sample_results)
        dashboard = plotter.build_dashboard()
        # Dashboard should be a HoloViews Layout
        assert dashboard is not None

    def test_empty_results_handling(self):
        """Should handle empty results gracefully."""
        empty_results = {}
        plotter = MassCorrelatorPlotter(empty_results)
        # Should not crash
        plot = plotter.build_all_correlators_overlay()
        assert plot is None


class TestIntegration:
    """Integration tests with minimal RunHistory."""

    def test_standard_channels_consistent(self):
        """Ensure channel definitions are internally consistent."""
        for channel in STANDARD_CHANNELS.values():
            # Each channel should have required attributes
            assert hasattr(channel, "name")
            assert hasattr(channel, "display_name")
            assert hasattr(channel, "quantum_numbers")
            assert hasattr(channel, "color")
            assert hasattr(channel, "operator_type")

            # Operator type should be valid
            assert channel.operator_type in {"bilinear", "trilinear", "gauge"}
