"""AIC parity tests for plotting module.

Verifies that ``fragile.physics.aic.plotting`` (verbatim copy) produces
identical HoloViews output to ``fragile.fractalai.qft.plotting`` for every
public ``build_*`` function.
"""

from __future__ import annotations

from typing import Any

import holoviews as hv
import numpy as np
import pytest
import torch

from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult

# -- Old (canonical) imports ---------------------------------------------------
from fragile.fractalai.qft.plotting import (
    build_all_channels_overlay as old_all_channels_overlay,
    build_correlation_decay_plot as old_decay_plot,
    build_correlator_plot as old_correlator_plot,
    build_effective_mass_plateau_plot as old_plateau_plot,
    build_effective_mass_plot as old_eff_mass_plot,
    build_mass_spectrum_bar as old_mass_bar,
    build_window_heatmap as old_heatmap,
)

# -- New (AIC copy) imports ----------------------------------------------------
from fragile.physics.aic.plotting import (
    build_all_channels_overlay as new_all_channels_overlay,
    build_correlation_decay_plot as new_decay_plot,
    build_correlator_plot as new_correlator_plot,
    build_effective_mass_plateau_plot as new_plateau_plot,
    build_effective_mass_plot as new_eff_mass_plot,
    build_mass_spectrum_bar as new_mass_bar,
    build_window_heatmap as new_heatmap,
)


hv.extension("bokeh")

# =============================================================================
# Helpers
# =============================================================================


def _extract_leaf_data(element: Any) -> list[Any]:
    """Recursively extract .data from HoloViews element trees."""
    leaves: list[Any] = []
    if hasattr(element, "values") and callable(element.values):
        try:
            children = list(element.values())
            if children and all(isinstance(c, hv.Element) for c in children):
                for child in children:
                    leaves.extend(_extract_leaf_data(child))
                return leaves
        except Exception:
            pass
    # Overlay iteration via __mul__ children
    if isinstance(element, hv.Overlay):
        for child in element:
            leaves.extend(_extract_leaf_data(child))
        return leaves
    # Leaf element
    if hasattr(element, "data"):
        leaves.append(element.data)
    return leaves


def _compare_data(old_data: Any, new_data: Any, label: str = "") -> None:
    """Compare two data payloads (arrays, dicts, DataFrames)."""
    prefix = f"{label}: " if label else ""
    if old_data is None and new_data is None:
        return
    assert type(old_data) is type(
        new_data
    ), f"{prefix}type mismatch {type(old_data)} vs {type(new_data)}"
    if isinstance(old_data, np.ndarray):
        np.testing.assert_array_equal(old_data, new_data, err_msg=f"{prefix}ndarray differs")
    elif isinstance(old_data, dict):
        assert set(old_data.keys()) == set(new_data.keys()), f"{prefix}dict keys differ"
        for k in old_data:
            _compare_data(old_data[k], new_data[k], label=f"{prefix}[{k}]")
    else:
        # pandas DataFrame or other - try equals method
        try:
            import pandas as pd

            if isinstance(old_data, pd.DataFrame):
                pd.testing.assert_frame_equal(old_data, new_data, obj=prefix)
                return
        except ImportError:
            pass
        # Fallback: equality
        assert old_data == new_data, f"{prefix}data differs"


def assert_holoviews_equal(old_plot: Any, new_plot: Any, label: str = "") -> None:
    """Compare two HoloViews objects by recursively comparing their leaf data."""
    prefix = f"{label}: " if label else ""
    if old_plot is None and new_plot is None:
        return
    assert (old_plot is None) == (new_plot is None), f"{prefix}one is None, the other is not"
    assert type(old_plot) is type(
        new_plot
    ), f"{prefix}type mismatch {type(old_plot).__name__} vs {type(new_plot).__name__}"
    old_leaves = _extract_leaf_data(old_plot)
    new_leaves = _extract_leaf_data(new_plot)
    assert len(old_leaves) == len(
        new_leaves
    ), f"{prefix}leaf count mismatch {len(old_leaves)} vs {len(new_leaves)}"
    for i, (od, nd) in enumerate(zip(old_leaves, new_leaves)):
        _compare_data(od, nd, label=f"{prefix}leaf[{i}]")


# =============================================================================
# Synthetic data builders
# =============================================================================


def _make_channel_result(
    channel_name: str = "scalar",
    n_lag: int = 20,
    mass: float = 0.5,
    mass_error: float = 0.01,
    r_squared: float = 0.99,
    n_samples: int = 100,
    dt: float = 1.0,
    with_windows: bool = True,
    with_correlator_err: bool = False,
    seed: int = 42,
) -> ChannelCorrelatorResult:
    """Build a synthetic ChannelCorrelatorResult with deterministic data."""
    rng = torch.Generator().manual_seed(seed)
    t = torch.arange(n_lag, dtype=torch.float32)
    correlator = torch.exp(-mass * t) + 0.01 * torch.randn(n_lag, generator=rng)
    correlator = correlator.abs()  # ensure positive for log plots
    effective_mass = torch.full((n_lag - 1,), mass) + 0.005 * torch.randn(n_lag - 1, generator=rng)
    effective_mass = effective_mass.abs()  # ensure positive
    series = torch.randn(n_samples, generator=rng)

    mass_fit: dict[str, Any] = {
        "mass": mass,
        "mass_error": mass_error,
        "r_squared": r_squared,
        "n_valid_windows": 10,
        "best_window": {"t_start": 2, "width": 8, "mass": mass, "r2": r_squared},
    }

    window_masses = None
    window_aic = None
    window_r2 = None
    window_widths = None
    if with_windows:
        n_widths = 5
        max_pos = n_lag
        rng2 = torch.Generator().manual_seed(seed + 1)
        window_masses = mass + 0.1 * torch.randn(n_widths, max_pos, generator=rng2)
        window_masses = window_masses.abs()
        window_aic = 10.0 + torch.randn(n_widths, max_pos, generator=rng2)
        window_r2 = 0.5 + 0.5 * torch.rand(n_widths, max_pos, generator=rng2)
        window_r2 = window_r2.clamp(0.0, 1.0)
        window_widths = [5, 10, 15, 20, 25]

    correlator_err = None
    if with_correlator_err:
        correlator_err = 0.01 * torch.ones(n_lag)

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=correlator,
        correlator_err=correlator_err,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=n_samples,
        dt=dt,
        window_masses=window_masses,
        window_aic=window_aic,
        window_widths=window_widths,
        window_r2=window_r2,
    )


def _make_results_dict(
    channels: list[str] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, ChannelCorrelatorResult]:
    """Build a dict of channel results for multi-channel plotting functions."""
    if channels is None:
        channels = ["scalar", "pseudoscalar", "vector"]
    results: dict[str, ChannelCorrelatorResult] = {}
    for i, ch in enumerate(channels):
        results[ch] = _make_channel_result(channel_name=ch, seed=seed + i * 100, **kwargs)
    return results


# =============================================================================
# Test classes
# =============================================================================


class TestParityBuildCorrelationDecayPlot:
    """Parity tests for build_correlation_decay_plot."""

    @pytest.fixture
    def decay_inputs(self) -> dict[str, Any]:
        rng = np.random.default_rng(123)
        r = np.arange(1, 21, dtype=np.float64)
        C = 2.0 * np.exp(-(r**2) / 25.0) + 0.01 * rng.standard_normal(20)
        C = np.abs(C)
        counts = np.ones(20, dtype=np.float64) * 100
        fit = {"C0": 2.0, "xi": 5.0}
        title = "Test Decay"
        return {"r": r, "C": C, "counts": counts, "fit": fit, "title": title}

    def test_basic_parity(self, decay_inputs: dict[str, Any]) -> None:
        old = old_decay_plot(**decay_inputs)
        new = new_decay_plot(**decay_inputs)
        assert_holoviews_equal(old, new, label="decay_basic")

    def test_no_fit_parity(self, decay_inputs: dict[str, Any]) -> None:
        """When xi=0, no fit curve is drawn."""
        decay_inputs["fit"] = {"C0": 0.0, "xi": 0.0}
        old = old_decay_plot(**decay_inputs)
        new = new_decay_plot(**decay_inputs)
        assert_holoviews_equal(old, new, label="decay_no_fit")

    def test_returns_none_on_insufficient_data(self) -> None:
        r = np.array([1.0])
        C = np.array([0.5])
        counts = np.array([0.0])  # all zero => masked out
        fit = {"C0": 1.0, "xi": 1.0}
        old = old_decay_plot(r, C, counts, fit, "empty")
        new = new_decay_plot(r, C, counts, fit, "empty")
        assert old is None and new is None


class TestParityBuildCorrelatorPlot:
    """Parity tests for build_correlator_plot (per-channel)."""

    @pytest.fixture
    def correlator_inputs(self) -> dict[str, Any]:
        n_lag = 30
        t = np.arange(n_lag, dtype=np.float64)
        correlator = np.exp(-0.3 * t) + 0.001
        mass_fit: dict[str, Any] = {
            "mass": 0.3,
            "mass_error": 0.02,
            "best_window": {"t_start": 3, "width": 10, "mass": 0.3, "r2": 0.98},
        }
        return {
            "lag_times": t,
            "correlator": correlator,
            "mass_fit": mass_fit,
            "channel_name": "scalar",
        }

    def test_basic_parity(self, correlator_inputs: dict[str, Any]) -> None:
        old = old_correlator_plot(**correlator_inputs)
        new = new_correlator_plot(**correlator_inputs)
        assert_holoviews_equal(old, new, label="correlator_basic")

    def test_logy_false_parity(self, correlator_inputs: dict[str, Any]) -> None:
        correlator_inputs["logy"] = False
        old = old_correlator_plot(**correlator_inputs)
        new = new_correlator_plot(**correlator_inputs)
        assert_holoviews_equal(old, new, label="correlator_logy_false")

    def test_no_mass_parity(self, correlator_inputs: dict[str, Any]) -> None:
        correlator_inputs["mass_fit"] = {"mass": 0.0, "mass_error": 0.0, "best_window": {}}
        old = old_correlator_plot(**correlator_inputs)
        new = new_correlator_plot(**correlator_inputs)
        assert_holoviews_equal(old, new, label="correlator_no_mass")

    def test_returns_none_when_no_valid_data(self) -> None:
        lag_times = np.array([0.0, 1.0])
        correlator = np.array([-1.0, -2.0])  # all negative => masked out
        mass_fit: dict[str, Any] = {"mass": 0.0, "best_window": {}}
        old = old_correlator_plot(lag_times, correlator, mass_fit, "empty")
        new = new_correlator_plot(lag_times, correlator, mass_fit, "empty")
        assert old is None and new is None


class TestParityBuildEffectiveMassPlot:
    """Parity tests for build_effective_mass_plot (per-channel)."""

    @pytest.fixture
    def eff_mass_inputs(self) -> dict[str, Any]:
        n_lag = 25
        t = np.arange(n_lag, dtype=np.float64)
        effective_mass = np.full(n_lag, 0.4) + 0.02 * np.sin(t)
        effective_mass = np.abs(effective_mass)
        mass_fit: dict[str, Any] = {
            "mass": 0.4,
            "mass_error": 0.015,
            "best_window": {"t_start": 3, "width": 10, "mass": 0.4, "r2": 0.97},
        }
        return {
            "lag_times": t,
            "effective_mass": effective_mass,
            "mass_fit": mass_fit,
            "channel_name": "vector",
        }

    def test_basic_parity(self, eff_mass_inputs: dict[str, Any]) -> None:
        old = old_eff_mass_plot(**eff_mass_inputs)
        new = new_eff_mass_plot(**eff_mass_inputs)
        assert_holoviews_equal(old, new, label="eff_mass_basic")

    def test_no_mass_parity(self, eff_mass_inputs: dict[str, Any]) -> None:
        eff_mass_inputs["mass_fit"] = {"mass": 0.0, "mass_error": 0.0}
        old = old_eff_mass_plot(**eff_mass_inputs)
        new = new_eff_mass_plot(**eff_mass_inputs)
        assert_holoviews_equal(old, new, label="eff_mass_no_fit")

    def test_with_error_band_parity(self, eff_mass_inputs: dict[str, Any]) -> None:
        """When mass_error > 0 and < mass, error band is drawn."""
        eff_mass_inputs["mass_fit"]["mass_error"] = 0.03
        old = old_eff_mass_plot(**eff_mass_inputs)
        new = new_eff_mass_plot(**eff_mass_inputs)
        assert_holoviews_equal(old, new, label="eff_mass_error_band")

    def test_returns_none_when_insufficient_data(self) -> None:
        t = np.array([0.0])
        meff = np.array([float("nan")])
        mass_fit: dict[str, Any] = {"mass": 0.0, "mass_error": 0.0}
        old = old_eff_mass_plot(t, meff, mass_fit, "empty")
        new = new_eff_mass_plot(t, meff, mass_fit, "empty")
        assert old is None and new is None


class TestParityBuildMassSpectrumBar:
    """Parity tests for build_mass_spectrum_bar."""

    @pytest.fixture
    def channel_results(self) -> dict[str, ChannelCorrelatorResult]:
        return _make_results_dict(channels=["scalar", "pseudoscalar", "vector"])

    def test_basic_parity(self, channel_results: dict[str, ChannelCorrelatorResult]) -> None:
        old = old_mass_bar(channel_results)
        new = new_mass_bar(channel_results)
        assert_holoviews_equal(old, new, label="mass_bar_basic")

    def test_custom_title_parity(
        self, channel_results: dict[str, ChannelCorrelatorResult]
    ) -> None:
        old = old_mass_bar(channel_results, title="Custom Title", ylabel="Custom Y")
        new = new_mass_bar(channel_results, title="Custom Title", ylabel="Custom Y")
        assert_holoviews_equal(old, new, label="mass_bar_custom_title")

    def test_custom_getters_parity(
        self, channel_results: dict[str, ChannelCorrelatorResult]
    ) -> None:
        def my_mass_getter(result: Any, _mode: str = "AIC-Weighted") -> float:
            return result.mass_fit.get("mass", 0.0) * 2.0

        def my_error_getter(result: Any, _mode: str = "AIC-Weighted") -> float:
            return result.mass_fit.get("mass_error", float("inf"))

        old = old_mass_bar(
            channel_results, mass_getter=my_mass_getter, error_getter=my_error_getter
        )
        new = new_mass_bar(
            channel_results, mass_getter=my_mass_getter, error_getter=my_error_getter
        )
        assert_holoviews_equal(old, new, label="mass_bar_custom_getters")

    def test_returns_none_on_empty_results(self) -> None:
        old = old_mass_bar({})
        new = new_mass_bar({})
        assert old is None and new is None

    def test_returns_none_when_all_zero_mass(self) -> None:
        result = _make_channel_result(channel_name="scalar", mass=0.0)
        old = old_mass_bar({"scalar": result})
        new = new_mass_bar({"scalar": result})
        assert old is None and new is None


class TestParityBuildWindowHeatmap:
    """Parity tests for build_window_heatmap."""

    @pytest.fixture
    def heatmap_inputs(self) -> dict[str, Any]:
        result = _make_channel_result(channel_name="tensor", with_windows=True, seed=99)
        window_masses = result.window_masses.cpu().numpy()
        window_aic = result.window_aic.cpu().numpy()
        window_r2 = result.window_r2.cpu().numpy()
        window_widths = result.window_widths
        best_window = result.mass_fit["best_window"]
        return {
            "window_masses": window_masses,
            "window_aic": window_aic,
            "window_widths": window_widths,
            "best_window": best_window,
            "channel_name": "tensor",
            "window_r2": window_r2,
        }

    def test_basic_parity(self, heatmap_inputs: dict[str, Any]) -> None:
        old = old_heatmap(**heatmap_inputs)
        new = new_heatmap(**heatmap_inputs)
        assert_holoviews_equal(old, new, label="heatmap_basic")

    def test_color_metric_aic_parity(self, heatmap_inputs: dict[str, Any]) -> None:
        heatmap_inputs["color_metric"] = "aic"
        old = old_heatmap(**heatmap_inputs)
        new = new_heatmap(**heatmap_inputs)
        assert_holoviews_equal(old, new, label="heatmap_color_aic")

    def test_color_metric_r2_parity(self, heatmap_inputs: dict[str, Any]) -> None:
        heatmap_inputs["color_metric"] = "r2"
        heatmap_inputs["alpha_metric"] = "r2"
        old = old_heatmap(**heatmap_inputs)
        new = new_heatmap(**heatmap_inputs)
        assert_holoviews_equal(old, new, label="heatmap_color_r2")

    def test_no_r2_fallback_parity(self, heatmap_inputs: dict[str, Any]) -> None:
        """When window_r2 is None and r2 metric requested, should fall back."""
        heatmap_inputs["window_r2"] = None
        heatmap_inputs["color_metric"] = "r2"
        heatmap_inputs["alpha_metric"] = "r2"
        old = old_heatmap(**heatmap_inputs)
        new = new_heatmap(**heatmap_inputs)
        assert_holoviews_equal(old, new, label="heatmap_no_r2_fallback")

    def test_no_best_window_parity(self, heatmap_inputs: dict[str, Any]) -> None:
        heatmap_inputs["best_window"] = {}
        old = old_heatmap(**heatmap_inputs)
        new = new_heatmap(**heatmap_inputs)
        assert_holoviews_equal(old, new, label="heatmap_no_best_window")

    def test_returns_none_on_empty(self) -> None:
        old = old_heatmap(None, None, [], {}, "empty")
        new = new_heatmap(None, None, [], {}, "empty")
        assert old is None and new is None


class TestParityBuildEffectiveMassPlateauPlot:
    """Parity tests for build_effective_mass_plateau_plot (two-panel)."""

    @pytest.fixture
    def plateau_inputs(self) -> dict[str, Any]:
        n_lag = 30
        t = np.arange(n_lag, dtype=np.float64)
        correlator = np.exp(-0.4 * t) + 0.001
        effective_mass = np.full(n_lag - 1, 0.4) + 0.01 * np.sin(np.arange(n_lag - 1))
        effective_mass = np.abs(effective_mass)
        mass_fit: dict[str, Any] = {
            "mass": 0.4,
            "mass_error": 0.02,
            "best_window": {"t_start": 3, "width": 10, "mass": 0.4, "r2": 0.97},
        }
        return {
            "lag_times": t,
            "correlator": correlator,
            "effective_mass": effective_mass,
            "mass_fit": mass_fit,
            "channel_name": "scalar",
        }

    def test_basic_parity(self, plateau_inputs: dict[str, Any]) -> None:
        old = old_plateau_plot(**plateau_inputs)
        new = new_plateau_plot(**plateau_inputs)
        assert old is not None and new is not None
        # Returns tuple of (left_panel, right_panel)
        assert len(old) == 2 and len(new) == 2
        assert_holoviews_equal(old[0], new[0], label="plateau_left")
        assert_holoviews_equal(old[1], new[1], label="plateau_right")

    def test_with_dt_parity(self, plateau_inputs: dict[str, Any]) -> None:
        plateau_inputs["dt"] = 0.5
        old = old_plateau_plot(**plateau_inputs)
        new = new_plateau_plot(**plateau_inputs)
        assert old is not None and new is not None
        assert_holoviews_equal(old[0], new[0], label="plateau_dt_left")
        assert_holoviews_equal(old[1], new[1], label="plateau_dt_right")

    def test_no_mass_parity(self, plateau_inputs: dict[str, Any]) -> None:
        plateau_inputs["mass_fit"] = {"mass": 0.0, "mass_error": 0.0, "best_window": {}}
        old = old_plateau_plot(**plateau_inputs)
        new = new_plateau_plot(**plateau_inputs)
        assert old is not None and new is not None
        assert_holoviews_equal(old[0], new[0], label="plateau_no_mass_left")
        assert_holoviews_equal(old[1], new[1], label="plateau_no_mass_right")

    def test_returns_none_when_insufficient_data(self) -> None:
        t = np.array([0.0, 1.0])
        corr = np.array([-1.0, -2.0])  # all negative => masked
        meff = np.array([float("nan")])
        mass_fit: dict[str, Any] = {"mass": 0.0, "mass_error": 0.0, "best_window": {}}
        old = old_plateau_plot(t, corr, meff, mass_fit, "empty")
        new = new_plateau_plot(t, corr, meff, mass_fit, "empty")
        assert old is None and new is None


class TestParityBuildAllChannelsOverlay:
    """Parity tests for build_all_channels_overlay."""

    @pytest.fixture
    def channel_results(self) -> dict[str, ChannelCorrelatorResult]:
        return _make_results_dict(
            channels=["scalar", "pseudoscalar", "vector"],
            with_correlator_err=True,
        )

    def test_correlator_overlay_parity(
        self, channel_results: dict[str, ChannelCorrelatorResult]
    ) -> None:
        old = old_all_channels_overlay(channel_results, plot_type="correlator")
        new = new_all_channels_overlay(channel_results, plot_type="correlator")
        assert_holoviews_equal(old, new, label="all_channels_correlator")

    def test_effective_mass_overlay_parity(
        self, channel_results: dict[str, ChannelCorrelatorResult]
    ) -> None:
        old = old_all_channels_overlay(channel_results, plot_type="effective_mass")
        new = new_all_channels_overlay(channel_results, plot_type="effective_mass")
        assert_holoviews_equal(old, new, label="all_channels_eff_mass")

    def test_no_logy_parity(self, channel_results: dict[str, ChannelCorrelatorResult]) -> None:
        old = old_all_channels_overlay(
            channel_results, plot_type="correlator", correlator_logy=False
        )
        new = new_all_channels_overlay(
            channel_results, plot_type="correlator", correlator_logy=False
        )
        assert_holoviews_equal(old, new, label="all_channels_no_logy")

    def test_returns_none_on_empty(self) -> None:
        old = old_all_channels_overlay({})
        new = new_all_channels_overlay({})
        assert old is None and new is None

    def test_returns_none_when_all_zero_samples(self) -> None:
        result = _make_channel_result(channel_name="scalar", n_samples=0)
        # Overwrite n_samples to 0
        result = ChannelCorrelatorResult(
            channel_name="scalar",
            correlator=result.correlator,
            correlator_err=None,
            effective_mass=result.effective_mass,
            mass_fit=result.mass_fit,
            series=result.series,
            n_samples=0,
            dt=result.dt,
        )
        old = old_all_channels_overlay({"scalar": result})
        new = new_all_channels_overlay({"scalar": result})
        assert old is None and new is None


class TestParityChannelPlotClass:
    """Parity tests for the ChannelPlot class."""

    @pytest.fixture
    def result_basic(self) -> ChannelCorrelatorResult:
        return _make_channel_result(
            channel_name="pseudoscalar",
            with_correlator_err=False,
            seed=77,
        )

    @pytest.fixture
    def result_with_errors(self) -> ChannelCorrelatorResult:
        return _make_channel_result(
            channel_name="vector",
            with_correlator_err=True,
            seed=88,
        )

    def test_correlator_plot_parity(self, result_basic: ChannelCorrelatorResult) -> None:
        from fragile.fractalai.qft.plotting import ChannelPlot as OldChannelPlot
        from fragile.physics.aic.plotting import ChannelPlot as NewChannelPlot

        old = OldChannelPlot(result_basic).correlator_plot()
        new = NewChannelPlot(result_basic).correlator_plot()
        assert_holoviews_equal(old, new, label="channel_plot_correlator")

    def test_effective_mass_plot_parity(self, result_basic: ChannelCorrelatorResult) -> None:
        from fragile.fractalai.qft.plotting import ChannelPlot as OldChannelPlot
        from fragile.physics.aic.plotting import ChannelPlot as NewChannelPlot

        old = OldChannelPlot(result_basic).effective_mass_plot()
        new = NewChannelPlot(result_basic).effective_mass_plot()
        assert_holoviews_equal(old, new, label="channel_plot_eff_mass")

    def test_correlator_with_errors_parity(
        self, result_with_errors: ChannelCorrelatorResult
    ) -> None:
        from fragile.fractalai.qft.plotting import ChannelPlot as OldChannelPlot
        from fragile.physics.aic.plotting import ChannelPlot as NewChannelPlot

        old = OldChannelPlot(result_with_errors).correlator_plot()
        new = NewChannelPlot(result_with_errors).correlator_plot()
        assert_holoviews_equal(old, new, label="channel_plot_corr_with_err")

    def test_effective_mass_with_errors_parity(
        self, result_with_errors: ChannelCorrelatorResult
    ) -> None:
        from fragile.fractalai.qft.plotting import ChannelPlot as OldChannelPlot
        from fragile.physics.aic.plotting import ChannelPlot as NewChannelPlot

        old = OldChannelPlot(result_with_errors).effective_mass_plot()
        new = NewChannelPlot(result_with_errors).effective_mass_plot()
        assert_holoviews_equal(old, new, label="channel_plot_eff_mass_with_err")

    def test_logy_false_parity(self, result_basic: ChannelCorrelatorResult) -> None:
        from fragile.fractalai.qft.plotting import ChannelPlot as OldChannelPlot
        from fragile.physics.aic.plotting import ChannelPlot as NewChannelPlot

        old = OldChannelPlot(result_basic, logy=False).correlator_plot()
        new = NewChannelPlot(result_basic, logy=False).correlator_plot()
        assert_holoviews_equal(old, new, label="channel_plot_logy_false")
