"""AIC parity tests for smoc_pipeline module.

Verifies that ``fragile.physics.aic.smoc_pipeline`` (verbatim copy) produces
identical results to the original ``fragile.fractalai.qft.smoc_pipeline``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.smoc_pipeline import (
    aggregate_correlators,
    AggregatedCorrelator,
    ChannelProjector,
    CorrelatorComputer,
    CorrelatorConfig,
    MassExtractionConfig,
    MassExtractor,
    ProjectorConfig,
    run_smoc_pipeline,
    SimulationConfig,
    SMoCSimulator,
)
from fragile.physics.aic.smoc_pipeline import (
    aggregate_correlators as new_aggregate,
    AggregatedCorrelator as NewAggCorr,
    ChannelProjector as NewProjector,
    CorrelatorComputer as NewCorrComputer,
    CorrelatorConfig as NewCorrConfig,
    MassExtractionConfig as NewMassConfig,
    MassExtractor as NewMassExtractor,
    ProjectorConfig as NewProjConfig,
    run_smoc_pipeline as new_run_pipeline,
    SimulationConfig as NewSimConfig,
    SMoCSimulator as NewSimulator,
)
from tests.physics.aic.conftest import (
    assert_mass_fit_equal,
    assert_tensor_or_nan_equal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_mass_extraction_equal(old: dict, new: dict, label: str = "") -> None:
    """Compare mass extraction result dicts, handling numpy arrays and nested dicts."""
    prefix = f"{label}: " if label else ""
    assert set(old.keys()) == set(
        new.keys()
    ), f"{prefix}key mismatch: {set(old.keys()) ^ set(new.keys())}"
    for key in old:
        old_val = old[key]
        new_val = new[key]
        if old_val is None and new_val is None:
            continue
        if isinstance(old_val, np.ndarray) and isinstance(new_val, np.ndarray):
            np.testing.assert_array_equal(old_val, new_val, err_msg=f"{prefix}{key}")
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            _assert_mass_extraction_equal(old_val, new_val, label=f"{prefix}{key}")
        elif isinstance(old_val, float) and isinstance(new_val, float):
            assert old_val == new_val, f"{prefix}{key}: {old_val!r} != {new_val!r}"
        else:
            assert old_val == new_val, f"{prefix}{key}: {old_val!r} != {new_val!r}"


SMALL_SIM_KWARGS = {
    "batch_size": 4,
    "grid_size": 8,
    "internal_dim": 4,
    "t_thermalization": 10,
    "t_measurement": 20,
    "seed": 42,
    "device": "cpu",
}


def _make_sim_config() -> SimulationConfig:
    return SimulationConfig(**SMALL_SIM_KWARGS)


def _make_new_sim_config() -> NewSimConfig:
    return NewSimConfig(**SMALL_SIM_KWARGS)


# ---------------------------------------------------------------------------
# 1. SMoCSimulator parity
# ---------------------------------------------------------------------------


class TestParitySMoCSimulator:
    """Verify simulator produces identical history tensors."""

    def test_simulate_parity(self) -> None:
        old_sim = SMoCSimulator(_make_sim_config())
        old_history = old_sim.run()

        new_sim = NewSimulator(_make_new_sim_config())
        new_history = new_sim.run()

        assert (
            old_history.shape == new_history.shape
        ), f"Shape mismatch: {old_history.shape} vs {new_history.shape}"
        assert torch.equal(old_history, new_history), (
            f"History tensors differ. "
            f"Max abs diff = {(old_history - new_history).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# 2. ChannelProjector parity
# ---------------------------------------------------------------------------


class TestParityChannelProjector:
    """Verify channel projection produces identical fields."""

    def test_project_all_parity(self) -> None:
        # Generate a deterministic history tensor
        torch.manual_seed(42)
        history = torch.randn(4, 20, 8, 4)

        old_proj = ChannelProjector(ProjectorConfig())
        new_proj = NewProjector(NewProjConfig())

        old_fields = old_proj.project_all(history)
        new_fields = new_proj.project_all(history)

        assert set(old_fields.keys()) == set(
            new_fields.keys()
        ), f"Channel key mismatch: {set(old_fields.keys()) ^ set(new_fields.keys())}"

        for ch in old_fields:
            assert_tensor_or_nan_equal(old_fields[ch], new_fields[ch], label=f"project_all[{ch}]")


# ---------------------------------------------------------------------------
# 3. CorrelatorComputer parity
# ---------------------------------------------------------------------------


class TestParityCorrelatorComputer:
    """Verify autocorrelation computation is identical."""

    def test_autocorrelation_parity(self) -> None:
        torch.manual_seed(42)
        signal = torch.randn(4, 50)

        old_cc = CorrelatorComputer(CorrelatorConfig())
        new_cc = NewCorrComputer(NewCorrConfig())

        old_auto = old_cc.compute_autocorrelation_fft(signal)
        new_auto = new_cc.compute_autocorrelation_fft(signal)

        assert_tensor_or_nan_equal(old_auto, new_auto, label="autocorrelation")


# ---------------------------------------------------------------------------
# 4. MassExtractor parity
# ---------------------------------------------------------------------------


class TestParityMassExtractor:
    """Verify mass extraction yields identical results."""

    def test_extract_mass_parity(self) -> None:
        # Build a synthetic AggregatedCorrelator that decays exponentially
        torch.manual_seed(42)
        t = torch.arange(30, dtype=torch.float32)
        mean = torch.exp(-0.3 * t) + 0.01 * torch.randn(30)
        mean = mean.clamp(min=1e-8)  # ensure positive for log
        std = 0.05 * torch.ones(30)
        std_err = std / (8**0.5)

        old_agg = AggregatedCorrelator(
            mean=mean.clone(), std=std.clone(), std_err=std_err.clone(), n_samples=8
        )
        new_agg = NewAggCorr(
            mean=mean.clone(), std=std.clone(), std_err=std_err.clone(), n_samples=8
        )

        old_ext = MassExtractor(MassExtractionConfig())
        new_ext = NewMassExtractor(NewMassConfig())

        old_result = old_ext.extract_mass(old_agg)
        new_result = new_ext.extract_mass(new_agg)

        _assert_mass_extraction_equal(old_result, new_result, label="extract_mass")


# ---------------------------------------------------------------------------
# 5. run_smoc_pipeline parity
# ---------------------------------------------------------------------------


class TestParityRunPipeline:
    """Verify the full pipeline produces identical results."""

    def test_pipeline_parity(self) -> None:
        pipeline_kwargs = {
            "batch_size": 4,
            "grid_size": 8,
            "internal_dim": 4,
            "t_thermalization": 10,
            "t_measurement": 20,
            "seed": 42,
            "device": "cpu",
            "verbose": False,
            "keep_history": True,
        }

        old_result = run_smoc_pipeline(**pipeline_kwargs)
        new_result = new_run_pipeline(**pipeline_kwargs)

        # Compare history
        assert old_result.history is not None
        assert new_result.history is not None
        assert torch.equal(old_result.history, new_result.history), (
            "Pipeline history tensors differ. "
            f"Max abs diff = {(old_result.history - new_result.history).abs().max().item():.2e}"
        )

        # Compare projected fields
        assert set(old_result.projected_fields.keys()) == set(new_result.projected_fields.keys())
        for ch in old_result.projected_fields:
            assert_tensor_or_nan_equal(
                old_result.projected_fields[ch],
                new_result.projected_fields[ch],
                label=f"pipeline projected_fields[{ch}]",
            )

        # Compare correlators
        assert set(old_result.correlators.keys()) == set(new_result.correlators.keys())
        for ch in old_result.correlators:
            assert_tensor_or_nan_equal(
                old_result.correlators[ch],
                new_result.correlators[ch],
                label=f"pipeline correlators[{ch}]",
            )

        # Compare aggregated correlators
        assert set(old_result.aggregated.keys()) == set(new_result.aggregated.keys())
        for ch in old_result.aggregated:
            old_agg = old_result.aggregated[ch]
            new_agg = new_result.aggregated[ch]
            assert_tensor_or_nan_equal(old_agg.mean, new_agg.mean, label=f"agg[{ch}].mean")
            assert_tensor_or_nan_equal(old_agg.std, new_agg.std, label=f"agg[{ch}].std")
            assert_tensor_or_nan_equal(
                old_agg.std_err, new_agg.std_err, label=f"agg[{ch}].std_err"
            )
            assert old_agg.n_samples == new_agg.n_samples

        # Compare masses
        assert set(old_result.masses.keys()) == set(new_result.masses.keys())
        for ch in old_result.masses:
            _assert_mass_extraction_equal(
                old_result.masses[ch],
                new_result.masses[ch],
                label=f"pipeline masses[{ch}]",
            )


# ---------------------------------------------------------------------------
# 6. aggregate_correlators parity
# ---------------------------------------------------------------------------


class TestParityAggregateCorrelators:
    """Verify correlator aggregation produces identical statistics."""

    def test_aggregate_parity(self) -> None:
        torch.manual_seed(42)
        correlators = torch.randn(8, 30)

        old_agg = aggregate_correlators(correlators.clone(), keep_raw=True)
        new_agg = new_aggregate(correlators.clone(), keep_raw=True)

        assert_tensor_or_nan_equal(old_agg.mean, new_agg.mean, label="agg.mean")
        assert_tensor_or_nan_equal(old_agg.std, new_agg.std, label="agg.std")
        assert_tensor_or_nan_equal(old_agg.std_err, new_agg.std_err, label="agg.std_err")
        assert old_agg.n_samples == new_agg.n_samples
        assert old_agg.raw is not None
        assert new_agg.raw is not None
        assert_tensor_or_nan_equal(old_agg.raw, new_agg.raw, label="agg.raw")
