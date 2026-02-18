"""Regression tests for meson phase channel."""

from __future__ import annotations

import pytest
import torch

from fragile.fractalai.qft.meson_phase_channels import (
    build_companion_pair_indices,
    compute_companion_meson_phase_correlator,
    compute_meson_phase_correlator_from_color,
    MesonPhaseCorrelatorConfig,
    MesonPhaseCorrelatorOutput,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.meson_phase_channels import (
    build_companion_pair_indices as new_build_pairs,
    compute_companion_meson_phase_correlator as new_companion,
    compute_meson_phase_correlator_from_color as new_from_color,
    MesonPhaseCorrelatorConfig as NewMesonConfig,
)

from .conftest import assert_outputs_equal, MockRunHistory


# =============================================================================
# Layer A: Analytical / from_color tests
# =============================================================================


class TestBuildCompanionPairIndices:
    def test_both_mode_shape(self):
        T, N = 5, 4
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        indices, valid = build_companion_pair_indices(comp_d, comp_c, pair_selection="both")
        assert indices.shape == (T, N, 2)
        assert valid.shape == (T, N, 2)

    def test_distance_mode_shape(self):
        T, N = 5, 4
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        indices, _valid = build_companion_pair_indices(comp_d, comp_c, pair_selection="distance")
        assert indices.shape == (T, N, 1)

    def test_clone_mode_shape(self):
        T, N = 5, 4
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        indices, _valid = build_companion_pair_indices(comp_d, comp_c, pair_selection="clone")
        assert indices.shape == (T, N, 1)


class TestFromColorOutput:
    @pytest.fixture
    def meson_output(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_alive,
        tiny_companions_distance,
        tiny_companions_clone,
    ):
        return compute_meson_phase_correlator_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            pair_selection="both",
            eps=1e-12,
        )

    def test_output_type(self, meson_output):
        assert isinstance(meson_output, MesonPhaseCorrelatorOutput)

    def test_pseudoscalar_shape(self, meson_output):
        assert meson_output.pseudoscalar.shape == (4,)

    def test_scalar_shape(self, meson_output):
        assert meson_output.scalar.shape == (4,)

    def test_counts_shape(self, meson_output):
        assert meson_output.counts.shape == (4,)

    def test_dtypes(self, meson_output):
        assert meson_output.pseudoscalar.dtype == torch.float32
        assert meson_output.scalar.dtype == torch.float32
        assert meson_output.counts.dtype == torch.int64

    def test_counts_lag0_positive(self, meson_output):
        assert meson_output.counts[0].item() > 0

    def test_connected_leq_raw_scalar(self, meson_output):
        assert meson_output.scalar_connected[0].item() <= meson_output.scalar_raw[0].item() + 1e-6

    def test_n_valid_pairs(self, meson_output):
        # T=5, N=3, pair_selection="both" => 2 pairs per walker
        assert meson_output.n_valid_source_pairs > 0

    def test_operator_series_shape(self, meson_output):
        assert meson_output.operator_pseudoscalar_series.ndim == 1
        assert meson_output.operator_pseudoscalar_series.shape[0] == 5
        assert meson_output.operator_scalar_series.shape[0] == 5

    def test_regression_finite(self, meson_output):
        assert torch.isfinite(meson_output.pseudoscalar).all()
        assert torch.isfinite(meson_output.scalar).all()
        assert torch.isfinite(meson_output.operator_pseudoscalar_series).all()
        assert torch.isfinite(meson_output.operator_scalar_series).all()


class TestFromColorEmpty:
    def test_empty_input(self):
        color = torch.zeros(0, 3, 3, dtype=torch.complex64)
        valid = torch.zeros(0, 3, dtype=torch.bool)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_meson_phase_correlator_from_color(
            color=color,
            color_valid=valid,
            alive=alive,
            companions_distance=comp_d,
            companions_clone=comp_c,
            max_lag=5,
        )
        assert out.pseudoscalar.shape == (6,)
        assert out.n_valid_source_pairs == 0


# =============================================================================
# Layer B: Integration with MockRunHistory
# =============================================================================


class TestCompanionMeson:
    @pytest.fixture
    def config(self):
        return MesonPhaseCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_meson_phase_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, MesonPhaseCorrelatorOutput)

    def test_output_shapes(self, output):
        assert output.pseudoscalar.shape == (11,)
        assert output.scalar.shape == (11,)
        assert output.counts.shape == (11,)

    def test_regression_finite(self, output):
        assert torch.isfinite(output.pseudoscalar.sum())
        assert torch.isfinite(output.scalar.sum())

    def test_operator_mode_standard_vs_score(self, mock_history):
        cfg_std = MesonPhaseCorrelatorConfig(max_lag=5, ell0=1.0, operator_mode="standard")
        cfg_score = MesonPhaseCorrelatorConfig(max_lag=5, ell0=1.0, operator_mode="score_directed")
        out_std = compute_companion_meson_phase_correlator(mock_history, cfg_std)
        out_score = compute_companion_meson_phase_correlator(mock_history, cfg_score)
        # Different modes should produce different results
        assert not torch.allclose(out_std.pseudoscalar, out_score.pseudoscalar, atol=1e-10)


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityMeson:
    """Verify new-location meson functions produce identical results to originals."""

    def test_build_pair_indices_parity(self):
        T, N = 5, 4
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        for mode in ("both", "distance", "clone"):
            old_idx, old_valid = build_companion_pair_indices(comp_d, comp_c, pair_selection=mode)
            new_idx, new_valid = new_build_pairs(comp_d, comp_c, pair_selection=mode)
            assert torch.equal(old_idx, new_idx), f"pair indices differ for mode={mode}"
            assert torch.equal(old_valid, new_valid), f"pair valid mask differs for mode={mode}"

    def test_from_color_parity(
        self,
        tiny_color_states,
        tiny_color_valid,
        tiny_alive,
        tiny_companions_distance,
        tiny_companions_clone,
    ):
        common = {
            "color": tiny_color_states,
            "color_valid": tiny_color_valid,
            "companions_distance": tiny_companions_distance,
            "companions_clone": tiny_companions_clone,
            "max_lag": 3,
            "use_connected": True,
            "pair_selection": "both",
            "eps": 1e-12,
        }
        old_out = compute_meson_phase_correlator_from_color(alive=tiny_alive, **common)
        new_out = new_from_color(**common)
        assert_outputs_equal(old_out, new_out)

    def test_companion_parity(self, mock_history):
        cfg_old = MesonPhaseCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )
        cfg_new = NewMesonConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )
        old_out = compute_companion_meson_phase_correlator(mock_history, cfg_old)
        new_out = new_companion(mock_history, cfg_new)
        assert_outputs_equal(old_out, new_out)
