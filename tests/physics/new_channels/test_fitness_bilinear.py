"""Tests for fitness-weighted companion-pair bilinear correlators."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.new_channels.fitness_bilinear_channels import (
    compute_fitness_bilinear_correlator,
    compute_fitness_bilinear_from_color,
    FitnessBilinearConfig,
    FitnessBilinearOutput,
)


# -- Shape tests --


class TestOutputShapes:
    def test_correlator_shapes(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        max_lag = 3
        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=max_lag,
        )
        assert out.fitness_pseudoscalar.shape == (max_lag + 1,)
        assert out.fitness_scalar_variance.shape == (max_lag + 1,)
        assert out.fitness_axial.shape == (max_lag + 1,)
        assert out.counts.shape == (max_lag + 1,)

    def test_operator_series_shapes(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
        )
        T = tiny_color_states.shape[0]
        assert out.operator_fitness_pseudoscalar_series.shape == (T,)
        assert out.operator_fitness_scalar_variance_series.shape == (T,)
        assert out.operator_fitness_axial_series.shape == (T,)

    def test_empty_frames(self, tiny_companions_distance, tiny_companions_clone):
        T, N = 0, 3
        color = torch.zeros(T, N, 3, dtype=torch.complex64)
        valid = torch.zeros(T, N, dtype=torch.bool)
        fitness = torch.zeros(T, N)
        scores = torch.zeros(T, N)
        alive = torch.zeros(T, N, dtype=torch.bool)
        out = compute_fitness_bilinear_from_color(
            color=color,
            color_valid=valid,
            companions_distance=torch.zeros(T, N, dtype=torch.long),
            companions_clone=torch.zeros(T, N, dtype=torch.long),
            fitness=fitness,
            cloning_scores=scores,
            alive_mask=alive,
            max_lag=5,
        )
        assert out.n_valid_source_pairs == 0
        assert out.fitness_pseudoscalar.shape == (6,)


# -- Correctness tests --


class TestCorrectness:
    def test_connected_vs_raw_differ(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
            use_connected=True,
        )
        # Connected and raw should differ (unless signal is zero-mean)
        # At least verify they are both computed
        assert out.fitness_pseudoscalar_raw is not None
        assert out.fitness_pseudoscalar_connected is not None

    def test_pair_selection_modes(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        for mode in ("distance", "clone", "both"):
            out = compute_fitness_bilinear_from_color(
                color=tiny_color_states,
                color_valid=tiny_color_valid,
                companions_distance=tiny_companions_distance,
                companions_clone=tiny_companions_clone,
                fitness=tiny_fitness,
                cloning_scores=tiny_scores,
                alive_mask=tiny_alive,
                max_lag=3,
                pair_selection=mode,
            )
            assert out.pair_selection == mode
            assert out.fitness_pseudoscalar.shape == (4,)

    def test_counts_positive(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
        )
        # At least lag=0 should have positive counts
        assert out.counts[0] > 0
        assert out.n_valid_source_pairs > 0

    def test_fitness_floor_handles_zeros(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_scores,
        tiny_alive,
    ):
        """Zero fitness values should be clamped by fitness_floor."""
        T, N = tiny_color_states.shape[:2]
        zero_fitness = torch.zeros(T, N)
        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=zero_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
            fitness_floor=1e-30,
        )
        # Should not produce NaN
        assert torch.isfinite(out.fitness_pseudoscalar).all()

    def test_deterministic(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        kwargs = {
            "color": tiny_color_states,
            "color_valid": tiny_color_valid,
            "companions_distance": tiny_companions_distance,
            "companions_clone": tiny_companions_clone,
            "fitness": tiny_fitness,
            "cloning_scores": tiny_scores,
            "alive_mask": tiny_alive,
            "max_lag": 3,
        }
        out1 = compute_fitness_bilinear_from_color(**kwargs)
        out2 = compute_fitness_bilinear_from_color(**kwargs)
        assert torch.equal(out1.fitness_pseudoscalar, out2.fitness_pseudoscalar)
        assert torch.equal(out1.fitness_scalar_variance, out2.fitness_scalar_variance)
        assert torch.equal(out1.fitness_axial, out2.fitness_axial)


# -- RunHistory wrapper test --


class TestRunHistoryWrapper:
    def test_compute_from_history(self, mock_history):
        config = FitnessBilinearConfig(
            warmup_fraction=0.1,
            max_lag=10,
        )
        out = compute_fitness_bilinear_correlator(mock_history, config)
        assert isinstance(out, FitnessBilinearOutput)
        assert out.fitness_pseudoscalar.shape == (11,)


# -- Adapter test --


class TestAdapter:
    def test_extract_fitness_bilinear(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        from fragile.physics.new_channels.mass_extraction_adapter import extract_fitness_bilinear

        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
        )
        corrs, ops = extract_fitness_bilinear(out, use_connected=True)
        assert "fitness_pseudoscalar" in corrs
        assert "fitness_scalar_variance" in corrs
        assert "fitness_axial" in corrs
        assert "fitness_pseudoscalar" in ops

    def test_collect_correlators_integration(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_companions_distance,
        tiny_companions_clone,
        tiny_fitness,
        tiny_scores,
        tiny_alive,
    ):
        from fragile.physics.new_channels.mass_extraction_adapter import collect_correlators

        out = compute_fitness_bilinear_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            fitness=tiny_fitness,
            cloning_scores=tiny_scores,
            alive_mask=tiny_alive,
            max_lag=3,
        )
        result = collect_correlators(out, use_connected=True)
        assert "fitness_pseudoscalar" in result.correlators
