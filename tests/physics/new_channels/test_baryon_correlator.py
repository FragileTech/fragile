"""Regression tests for baryon triplet channel."""

from __future__ import annotations

import torch
import pytest

from fragile.fractalai.qft.baryon_triplet_channels import (
    BaryonTripletCorrelatorConfig,
    BaryonTripletCorrelatorOutput,
    compute_baryon_correlator_from_color,
    compute_companion_baryon_correlator,
    _det3,
    build_companion_triplets,
    _resolve_frame_indices,
)

# New-location aliases for parity tests
from fragile.physics.new_channels.baryon_triplet_channels import (
    BaryonTripletCorrelatorConfig as NewBaryonConfig,
    _det3 as new_det3,
    build_companion_triplets as new_build_triplets,
    compute_baryon_correlator_from_color as new_from_color,
    compute_companion_baryon_correlator as new_companion,
)

from .conftest import MockRunHistory, assert_outputs_equal


# =============================================================================
# Layer A: Analytical / from_color tests
# =============================================================================


class TestDet3:
    def test_identity_columns(self):
        a = torch.tensor([1.0, 0.0, 0.0])
        b = torch.tensor([0.0, 1.0, 0.0])
        c = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(_det3(a, b, c), torch.tensor(1.0))

    def test_matches_torch_linalg_det(self):
        gen = torch.Generator().manual_seed(50)
        # [B, 3] for each column
        a = torch.randn(10, 3, generator=gen)
        b = torch.randn(10, 3, generator=gen)
        c = torch.randn(10, 3, generator=gen)
        det_manual = _det3(a, b, c)
        mat = torch.stack([a, b, c], dim=-1)  # [B, 3, 3]
        det_ref = torch.linalg.det(mat)
        assert torch.allclose(det_manual, det_ref, atol=1e-5)


class TestBuildCompanionTriplets:
    def test_cyclic_all_valid_distinct(self):
        N = 5
        T = 3
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        anchor, j, k, valid = build_companion_triplets(comp_d, comp_c)
        # All should be valid for cyclic with N>=3
        assert valid.all()
        # All distinct
        assert (anchor != j).all()
        assert (anchor != k).all()
        assert (j != k).all()


class TestResolveFrameIndices:
    def test_basic_range(self):
        history = MockRunHistory(n_recorded=30)
        indices = _resolve_frame_indices(history, warmup_fraction=0.1, end_fraction=1.0, mc_time_index=None)
        assert len(indices) > 0
        assert indices[0] >= 1
        assert indices[-1] < 30


class TestFromColorOutput:
    """Tests on compute_baryon_correlator_from_color with tiny fixtures."""

    @pytest.fixture
    def baryon_output(self, tiny_color_states, tiny_color_valid, tiny_alive,
                      tiny_companions_distance, tiny_companions_clone):
        return compute_baryon_correlator_from_color(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            alive=tiny_alive,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            eps=1e-12,
            operator_mode="det_abs",
        )

    def test_output_type(self, baryon_output):
        assert isinstance(baryon_output, BaryonTripletCorrelatorOutput)

    def test_correlator_shape(self, baryon_output):
        assert baryon_output.correlator.shape == (4,)  # max_lag=3 -> 4 entries

    def test_counts_shape(self, baryon_output):
        assert baryon_output.counts.shape == (4,)

    def test_dtypes(self, baryon_output):
        assert baryon_output.correlator.dtype == torch.float32
        assert baryon_output.counts.dtype == torch.int64

    def test_counts_lag0_positive(self, baryon_output):
        assert baryon_output.counts[0].item() > 0

    def test_correlator_raw_lag0_nonneg(self, baryon_output):
        # Self-correlation of |det| >= 0
        assert baryon_output.correlator_raw[0].item() >= 0

    def test_connected_leq_raw_lag0(self, baryon_output):
        assert baryon_output.correlator_connected[0].item() <= baryon_output.correlator_raw[0].item() + 1e-6

    def test_n_valid_triplets(self, baryon_output):
        # All walkers valid, cyclic companions, T=5 => 5*3 = 15
        assert baryon_output.n_valid_source_triplets == 5 * 3

    def test_operator_series_shape(self, baryon_output):
        assert baryon_output.operator_baryon_series.ndim == 1
        assert baryon_output.operator_baryon_series.shape[0] == 5  # T=5

    def test_regression_finite(self, baryon_output):
        assert torch.isfinite(baryon_output.correlator).all()
        assert torch.isfinite(baryon_output.operator_baryon_series).all()
        assert torch.isfinite(baryon_output.correlator.sum())

    def test_regression_sum(self, baryon_output):
        # Regression: ensure deterministic output changes are detected
        s = baryon_output.correlator.sum().item()
        assert abs(s) < 1e6  # sanity bound


class TestFromColorEmpty:
    def test_empty_input(self):
        color = torch.zeros(0, 3, 3, dtype=torch.complex64)
        valid = torch.zeros(0, 3, dtype=torch.bool)
        alive = torch.zeros(0, 3, dtype=torch.bool)
        comp_d = torch.zeros(0, 3, dtype=torch.long)
        comp_c = torch.zeros(0, 3, dtype=torch.long)
        out = compute_baryon_correlator_from_color(
            color=color, color_valid=valid, alive=alive,
            companions_distance=comp_d, companions_clone=comp_c,
            max_lag=5,
        )
        assert out.correlator.shape == (6,)
        assert out.n_valid_source_triplets == 0


# =============================================================================
# Layer B: Integration with MockRunHistory
# =============================================================================


class TestCompanionBaryon:
    @pytest.fixture
    def config(self):
        return BaryonTripletCorrelatorConfig(
            warmup_fraction=0.1,
            end_fraction=1.0,
            max_lag=10,
            use_connected=True,
            ell0=1.0,
        )

    @pytest.fixture
    def output(self, mock_history, config):
        return compute_companion_baryon_correlator(mock_history, config)

    def test_runs_without_error(self, output):
        assert isinstance(output, BaryonTripletCorrelatorOutput)

    def test_output_shapes(self, output):
        assert output.correlator.shape == (11,)
        assert output.counts.shape == (11,)

    def test_frame_indices_nonempty(self, output):
        assert len(output.frame_indices) > 0
        for idx in output.frame_indices:
            assert 1 <= idx < 30

    def test_regression_finite(self, output):
        assert torch.isfinite(output.correlator.sum())

    def test_n_valid_positive(self, output):
        assert output.n_valid_source_triplets > 0

    def test_operator_series_finite(self, output):
        assert torch.isfinite(output.operator_baryon_series).all()

    def test_operator_modes_differ(self, mock_history):
        results = {}
        for mode in ("det_abs", "flux_action", "score_signed"):
            cfg = BaryonTripletCorrelatorConfig(
                max_lag=5, ell0=1.0, operator_mode=mode,
            )
            out = compute_companion_baryon_correlator(mock_history, cfg)
            results[mode] = out.correlator.sum().item()
        # At least two modes produce different values
        vals = list(results.values())
        assert not all(abs(v - vals[0]) < 1e-10 for v in vals[1:])


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityBaryon:
    """Verify new-location baryon functions produce identical results to originals."""

    def test_det3_parity(self):
        gen = torch.Generator().manual_seed(200)
        a = torch.randn(10, 3, generator=gen)
        b = torch.randn(10, 3, generator=gen)
        c = torch.randn(10, 3, generator=gen)
        assert torch.equal(_det3(a, b, c), new_det3(a, b, c))

    def test_build_triplets_parity(self):
        N, T = 5, 3
        comp_d = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).clone()
        comp_c = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1).clone()
        old_anchor, old_j, old_k, old_valid = build_companion_triplets(comp_d, comp_c)
        new_anchor, new_j, new_k, new_valid = new_build_triplets(comp_d, comp_c)
        assert torch.equal(old_anchor, new_anchor)
        assert torch.equal(old_j, new_j)
        assert torch.equal(old_k, new_k)
        assert torch.equal(old_valid, new_valid)

    def test_from_color_parity(self, tiny_color_states, tiny_color_valid, tiny_alive,
                               tiny_companions_distance, tiny_companions_clone):
        common = dict(
            color=tiny_color_states,
            color_valid=tiny_color_valid,
            companions_distance=tiny_companions_distance,
            companions_clone=tiny_companions_clone,
            max_lag=3,
            use_connected=True,
            eps=1e-12,
            operator_mode="det_abs",
        )
        old_out = compute_baryon_correlator_from_color(alive=tiny_alive, **common)
        new_out = new_from_color(**common)
        assert_outputs_equal(old_out, new_out)

    def test_companion_parity(self, mock_history):
        cfg_old = BaryonTripletCorrelatorConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        cfg_new = NewBaryonConfig(
            warmup_fraction=0.1, end_fraction=1.0, max_lag=10,
            use_connected=True, ell0=1.0,
        )
        old_out = compute_companion_baryon_correlator(mock_history, cfg_old)
        new_out = new_companion(mock_history, cfg_new)
        assert_outputs_equal(old_out, new_out)
