"""Regression tests for vector meson channel."""

from __future__ import annotations

import torch
import pytest

from fragile.fractalai.qft.vector_meson_channels import (
    VectorMesonCorrelatorConfig,
    VectorMesonCorrelatorOutput,
    compute_vector_meson_correlator_from_color_positions,
    compute_companion_vector_meson_correlator,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.vector_meson_channels import (
    VectorMesonCorrelatorConfig as NewVectorConfig,
    compute_vector_meson_correlator_from_color_positions as new_from_color,
    compute_companion_vector_meson_correlator as new_companion,
)

from .conftest import MockRunHistory, assert_outputs_equal


# =============================================================================
# Layer A: Analytical / from_color tests
# =============================================================================


class TestFromColorPositionsOutput:
    @pytest.fixture
    def vector_output(self, tiny_color_states, tiny_color_valid, tiny_alive,
                      tiny_companions_distance, tiny_companions_clone, tiny_positions):
        return compute_vector_meson_correlator_from_color_positions(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            positions=tiny_positions,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            pair_selection="both",
            eps=1e-12,
        )

    def test_output_type(self, vector_output):
        assert isinstance(vector_output, VectorMesonCorrelatorOutput)

    def test_vector_shape(self, vector_output):
        assert vector_output.vector.shape == (4,)  # max_lag=3 -> 4

    def test_axial_vector_shape(self, vector_output):
        assert vector_output.axial_vector.shape == (4,)

    def test_mean_vector_shape(self, vector_output):
        assert vector_output.mean_vector.shape == (3,)

    def test_operator_vector_series_shape(self, vector_output):
        assert vector_output.operator_vector_series.shape == (5, 3)

    def test_dtypes(self, vector_output):
        assert vector_output.vector.dtype == torch.float32
        assert vector_output.counts.dtype == torch.int64

    def test_counts_positive(self, vector_output):
        assert vector_output.counts[0].item() > 0

    def test_connected_leq_raw(self, vector_output):
        assert vector_output.vector_connected[0].item() <= vector_output.vector_raw[0].item() + 1e-6

    def test_unit_displacement_different(self, tiny_color_states, tiny_color_valid, tiny_alive,
                                         tiny_companions_distance, tiny_companions_clone,
                                         tiny_positions):
        out_normal = compute_vector_meson_correlator_from_color_positions(
            color=tiny_color_states, color_valid=tiny_color_valid,
            positions=tiny_positions, alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3, use_unit_displacement=False,
        )
        out_unit = compute_vector_meson_correlator_from_color_positions(
            color=tiny_color_states, color_valid=tiny_color_valid,
            positions=tiny_positions, alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3, use_unit_displacement=True,
        )
        assert not torch.allclose(out_normal.vector, out_unit.vector, atol=1e-10)

    def test_regression_finite(self, vector_output):
        assert torch.isfinite(vector_output.vector).all()
        assert torch.isfinite(vector_output.axial_vector).all()
        assert torch.isfinite(vector_output.operator_vector_series).all()


class TestFromColorEmpty:
    def test_empty_input(self):
        color = torch.zeros(0, 3, 3, dtype=torch.complex64)
        valid = torch.zeros(0, 3, dtype=torch.bool)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        pos = torch.zeros(0, 3, 3)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_vector_meson_correlator_from_color_positions(
            color=color, color_valid=valid, positions=pos, alive=alive,
            companions_distance=comp_d, companions_clone=comp_c,
            max_lag=5,
        )
        assert out.vector.shape == (6,)
        assert out.n_valid_source_pairs == 0


# =============================================================================
# Layer B: Integration with MockRunHistory
# =============================================================================


class TestCompanionVector:
    @pytest.fixture
    def config(self):
        return VectorMesonCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_vector_meson_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, VectorMesonCorrelatorOutput)

    def test_output_shapes(self, output):
        assert output.vector.shape == (11,)
        assert output.axial_vector.shape == (11,)
        assert output.counts.shape == (11,)

    def test_regression_finite(self, output):
        assert torch.isfinite(output.vector.sum())
        assert torch.isfinite(output.axial_vector.sum())

    def test_score_directed_mode(self, mock_history):
        cfg = VectorMesonCorrelatorConfig(
            max_lag=5, ell0=1.0, operator_mode="score_directed",
        )
        out = compute_companion_vector_meson_correlator(mock_history, cfg)
        assert torch.isfinite(out.vector.sum())


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityVector:
    """Verify new-location vector meson functions produce identical results to originals."""

    def test_from_color_parity(self, tiny_color_states, tiny_color_valid, tiny_alive,
                               tiny_companions_distance, tiny_companions_clone, tiny_positions):
        common = dict(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            positions=tiny_positions,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            pair_selection="both",
            eps=1e-12,
        )
        old_out = compute_vector_meson_correlator_from_color_positions(alive=tiny_alive, **common)
        new_out = new_from_color(**common)
        assert_outputs_equal(old_out, new_out)

    def test_companion_parity(self, mock_history):
        cfg_old = VectorMesonCorrelatorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        cfg_new = NewVectorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        old_out = compute_companion_vector_meson_correlator(mock_history, cfg_old)
        new_out = new_companion(mock_history, cfg_new)
        assert_outputs_equal(old_out, new_out)
