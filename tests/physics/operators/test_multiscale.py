"""Tests for fragile.physics.operators.multiscale."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from fragile.physics.operators.multiscale import (
    gate_pair_validity_by_scale,
    gate_triplet_validity_by_scale,
    per_frame_series_multiscale,
    per_frame_vector_series_multiscale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_distance_matrix(T: int, N: int, *, scale: float = 1.0, seed: int = 0) -> Tensor:
    """Build a symmetric non-negative distance matrix [T, N, N] with zero diagonal."""
    gen = torch.Generator().manual_seed(seed)
    pos = torch.randn(T, N, 3, generator=gen) * scale
    diffs = pos.unsqueeze(2) - pos.unsqueeze(1)  # [T, N, N, 3]
    d = diffs.norm(dim=-1)
    d.diagonal(dim1=1, dim2=2).zero_()
    return d


def _cyclic_pair_indices(T: int, N: int, P: int) -> Tensor:
    """Create pair indices [T, N, P] via cyclic shifts 1..P."""
    base = torch.arange(N)
    layers = [base.roll(-s) for s in range(1, P + 1)]
    single = torch.stack(layers, dim=-1)  # [N, P]
    return single.unsqueeze(0).expand(T, -1, -1).clone()


# ===================================================================
# gate_pair_validity_by_scale
# ===================================================================


class TestGatePairValidityByScale:
    """Tests for gate_pair_validity_by_scale."""

    def test_output_shape(self) -> None:
        T, N, P, S = 4, 8, 3, 5
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N)
        scales = torch.linspace(0.5, 5.0, S)

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert result.shape == (T, S, N, P)
        assert result.dtype == torch.bool

    def test_all_valid_large_scale(self) -> None:
        """A scale larger than any distance yields all True (given base valid)."""
        T, N, P = 3, 6, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N)
        max_dist = distances.max().item()
        scales = torch.tensor([max_dist + 10.0])

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert result.all()

    def test_small_scale_restricts(self) -> None:
        """A very small scale should gate out most pairs."""
        T, N, P = 3, 10, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N, scale=5.0)
        # Tiny scale — only very close pairs survive
        scales = torch.tensor([1e-6])

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        # Most pairs should be gated out
        frac_valid = result.float().mean().item()
        assert frac_valid < 0.5

    def test_zero_distance_always_within_scale(self) -> None:
        """Pairs whose distance is exactly zero pass any positive scale."""
        T, N, P = 2, 5, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        # All pairs point to themselves -> d=0
        pair_indices = torch.arange(N).view(1, N, 1).expand(T, N, P).clone()
        distances = _make_distance_matrix(T, N)
        scales = torch.tensor([0.001, 1.0, 100.0])

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        # Distance to self is 0 -> within any scale >= 0
        assert result.all()

    def test_base_invalid_stays_invalid(self) -> None:
        """Pairs that are invalid in base remain invalid at all scales."""
        T, N, P, S = 3, 6, 2, 4
        pair_valid = torch.zeros(T, N, P, dtype=torch.bool)  # all invalid
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N)
        scales = torch.tensor([1e6] * S)  # huge scales

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert not result.any()

    def test_monotonicity(self) -> None:
        """Larger scale => superset of smaller scale validity."""
        T, N, P = 4, 10, 3
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N, scale=3.0)
        scales = torch.tensor([0.5, 1.0, 2.0, 5.0, 20.0])

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        # For each consecutive pair of scales, smaller is subset of larger
        for s in range(result.shape[1] - 1):
            smaller = result[:, s, :, :]
            larger = result[:, s + 1, :, :]
            # If smaller is True, larger must also be True
            assert (smaller & ~larger).sum().item() == 0

    def test_single_scale_single_pair(self) -> None:
        """Minimal dimensions: S=1, P=1."""
        T, N, P = 2, 4, 1
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = torch.zeros(T, N, N)  # all zero distances
        scales = torch.tensor([1.0])

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert result.shape == (T, 1, N, 1)
        assert result.all()

    def test_partial_base_validity(self) -> None:
        """Only some pairs are base-valid; scale gating further restricts."""
        T, N, P = 2, 6, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_valid[:, 0, :] = False  # walker 0 pairs always invalid
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N)
        scales = torch.tensor([1e6])  # huge scale

        result = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        # Walker 0 pairs should still be invalid
        assert not result[:, :, 0, :].any()
        # Other walkers should be valid (scale is huge)
        assert result[:, :, 1:, :].all()


# ===================================================================
# gate_triplet_validity_by_scale
# ===================================================================


class TestGateTripletValidityByScale:
    """Tests for gate_triplet_validity_by_scale."""

    def test_output_shape(self) -> None:
        T, N, S = 4, 8, 5
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        d_ij = torch.rand(T, N)
        d_ik = torch.rand(T, N)
        d_jk = torch.rand(T, N)
        scales = torch.linspace(0.5, 5.0, S)

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        assert result.shape == (T, S, N)
        assert result.dtype == torch.bool

    def test_uses_max_of_three_distances(self) -> None:
        """Gating should use max(d_ij, d_ik, d_jk) for each triplet."""
        T, N = 1, 5
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        # d_ij is large, others are small
        d_ij = torch.tensor([[10.0, 0.1, 0.1, 10.0, 0.1]])
        d_ik = torch.tensor([[0.1, 10.0, 0.1, 0.1, 0.1]])
        d_jk = torch.tensor([[0.1, 0.1, 10.0, 0.1, 0.1]])
        scales = torch.tensor([5.0])  # below 10.0

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        # Triplets 0,1,2,3 have max dist = 10.0 > 5.0 -> invalid
        # Triplet 4 has max dist = 0.1 <= 5.0 -> valid
        expected = torch.tensor([[[False, False, False, False, True]]])
        assert result.equal(expected)

    def test_base_invalid_stays_invalid(self) -> None:
        """Base-invalid triplets remain invalid regardless of scale."""
        T, N, _S = 3, 6, 3
        triplet_valid = torch.zeros(T, N, dtype=torch.bool)
        d_ij = torch.zeros(T, N)
        d_ik = torch.zeros(T, N)
        d_jk = torch.zeros(T, N)
        scales = torch.tensor([1e6, 1e7, 1e8])

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        assert not result.any()

    def test_monotonicity(self) -> None:
        """Larger scale => superset of smaller scale triplet validity."""
        T, N = 5, 10
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        gen = torch.Generator().manual_seed(77)
        d_ij = torch.rand(T, N, generator=gen) * 10
        d_ik = torch.rand(T, N, generator=gen) * 10
        d_jk = torch.rand(T, N, generator=gen) * 10
        scales = torch.tensor([1.0, 3.0, 5.0, 8.0, 15.0])

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        for s in range(result.shape[1] - 1):
            smaller = result[:, s, :]
            larger = result[:, s + 1, :]
            assert (smaller & ~larger).sum().item() == 0

    def test_all_zero_distances(self) -> None:
        """Zero distances for all triplets -> all within any positive scale."""
        T, N = 2, 5
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        d_ij = torch.zeros(T, N)
        d_ik = torch.zeros(T, N)
        d_jk = torch.zeros(T, N)
        scales = torch.tensor([0.001, 1.0])

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        assert result.all()

    def test_single_scale_single_triplet(self) -> None:
        """Minimal dimensions: S=1, N=1."""
        T, N = 2, 1
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        d_ij = torch.tensor([[0.5], [0.3]])
        d_ik = torch.tensor([[0.2], [0.1]])
        d_jk = torch.tensor([[0.4], [0.8]])
        scales = torch.tensor([0.6])

        result = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        assert result.shape == (T, 1, N)
        # t=0: max(0.5, 0.2, 0.4) = 0.5 <= 0.6 -> True
        # t=1: max(0.3, 0.1, 0.8) = 0.8 > 0.6 -> False
        expected = torch.tensor([[[True]], [[False]]])
        assert result.equal(expected)


# ===================================================================
# per_frame_series_multiscale
# ===================================================================


class TestPerFrameSeriesMultiscale:
    """Tests for per_frame_series_multiscale."""

    def test_output_shape_pair_case(self) -> None:
        """3D values [T, N, P] with 4D valid [T, S, N, P] -> [S, T]."""
        T, N, P, S = 5, 8, 3, 4
        values = torch.randn(T, N, P)
        valid = torch.ones(T, S, N, P, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert result.shape == (S, T)
        assert result.dtype == torch.float32

    def test_output_shape_triplet_case(self) -> None:
        """2D values [T, N] with 3D valid [T, S, N] -> [S, T]."""
        T, N, S = 5, 8, 4
        values = torch.randn(T, N)
        valid = torch.ones(T, S, N, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert result.shape == (S, T)
        assert result.dtype == torch.float32

    def test_all_valid_constant_values(self) -> None:
        """All valid with constant values -> output equals that constant."""
        T, N, P, S = 4, 6, 2, 3
        c = math.pi
        values = torch.full((T, N, P), c)
        valid = torch.ones(T, S, N, P, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert torch.allclose(result, torch.full((S, T), c), atol=1e-5)

    def test_all_valid_constant_triplet_case(self) -> None:
        """Triplet case: all valid constant -> output equals constant."""
        T, N, S = 4, 6, 3
        c = math.e
        values = torch.full((T, N), c)
        valid = torch.ones(T, S, N, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert torch.allclose(result, torch.full((S, T), c), atol=1e-5)

    def test_all_invalid_returns_zeros(self) -> None:
        """All invalid -> zeros everywhere."""
        T, N, P, S = 4, 6, 2, 3
        values = torch.randn(T, N, P)
        valid = torch.zeros(T, S, N, P, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert (result == 0).all()

    def test_all_invalid_triplet_returns_zeros(self) -> None:
        """Triplet case: all invalid -> zeros."""
        T, N, S = 4, 6, 3
        values = torch.randn(T, N)
        valid = torch.zeros(T, S, N, dtype=torch.bool)

        result = per_frame_series_multiscale(values, valid)
        assert (result == 0).all()

    def test_partial_validity_correct_mean(self) -> None:
        """Partial validity computes correct weighted (masked) mean."""
        _T, _N, _P, _S = 1, 4, 1, 1
        # values: [1, 4, 1] = [[1], [2], [3], [4]]
        values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        # valid: [1, 1, 4, 1] - only first two walkers valid
        valid = torch.tensor([[[[True], [True], [False], [False]]]])

        result = per_frame_series_multiscale(values, valid)
        # Mean of 1.0 and 2.0 = 1.5
        expected = torch.tensor([[1.5]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_partial_validity_triplet(self) -> None:
        """Triplet case: partial validity gives correct mean."""
        _T, _N, _S = 1, 5, 1
        values = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0]])
        valid = torch.tensor([[[True, True, False, False, True]]])

        result = per_frame_series_multiscale(values, valid)
        # Mean of 10, 20, 50 = 80/3
        expected = torch.tensor([[80.0 / 3.0]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_invalid_ndim_raises(self) -> None:
        """Values with ndim != 2 or 3 raises ValueError."""
        values_1d = torch.randn(10)
        valid_1d = torch.ones(10, dtype=torch.bool)
        with pytest.raises(ValueError, match="2D.*3D"):
            per_frame_series_multiscale(values_1d, valid_1d)

        values_4d = torch.randn(2, 3, 4, 5)
        valid_4d = torch.ones(2, 3, 4, 5, dtype=torch.bool)
        with pytest.raises(ValueError, match="2D.*3D"):
            per_frame_series_multiscale(values_4d, valid_4d)

    def test_different_scales_different_means(self) -> None:
        """Different validity per scale should yield different averages."""
        T, N, P = 1, 4, 1
        S = 2
        values = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        valid = torch.zeros(T, S, N, P, dtype=torch.bool)
        # Scale 0: only walker 0 valid
        valid[0, 0, 0, 0] = True
        # Scale 1: walkers 0 and 3 valid
        valid[0, 1, 0, 0] = True
        valid[0, 1, 3, 0] = True

        result = per_frame_series_multiscale(values, valid)
        assert result.shape == (S, T)
        assert torch.allclose(result[0, 0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(result[1, 0], torch.tensor(2.5), atol=1e-5)  # mean(1,4)


# ===================================================================
# per_frame_vector_series_multiscale
# ===================================================================


class TestPerFrameVectorSeriesMultiscale:
    """Tests for per_frame_vector_series_multiscale."""

    def test_output_shape(self) -> None:
        T, N, P, C, S = 5, 8, 3, 4, 2
        values = torch.randn(T, N, P, C)
        valid = torch.ones(T, S, N, P, dtype=torch.bool)

        result = per_frame_vector_series_multiscale(values, valid)
        assert result.shape == (S, T, C)
        assert result.dtype == torch.float32

    def test_all_valid_constant(self) -> None:
        """All valid with constant vector -> output equals that constant."""
        T, N, P, C, S = 4, 6, 2, 3, 2
        vec = torch.tensor([1.0, 2.0, 3.0])
        values = vec.view(1, 1, 1, C).expand(T, N, P, C).clone().float()
        valid = torch.ones(T, S, N, P, dtype=torch.bool)

        result = per_frame_vector_series_multiscale(values, valid)
        for s in range(S):
            for t in range(T):
                assert torch.allclose(result[s, t], vec, atol=1e-5)

    def test_all_invalid_returns_zeros(self) -> None:
        """All invalid -> zeros everywhere."""
        T, N, P, C, S = 4, 6, 2, 3, 2
        values = torch.randn(T, N, P, C)
        valid = torch.zeros(T, S, N, P, dtype=torch.bool)

        result = per_frame_vector_series_multiscale(values, valid)
        assert (result == 0).all()

    def test_wrong_values_ndim_raises(self) -> None:
        """Values not 4D raises ValueError."""
        values_3d = torch.randn(2, 3, 4)
        valid_4d = torch.ones(2, 1, 3, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match=r"\[T,N,P,C\]"):
            per_frame_vector_series_multiscale(values_3d, valid_4d)

    def test_wrong_valid_ndim_raises(self) -> None:
        """Valid not 4D raises ValueError."""
        values_4d = torch.randn(2, 3, 4, 5)
        valid_3d = torch.ones(2, 3, 4, dtype=torch.bool)
        with pytest.raises(ValueError, match=r"\[T,S,N,P\]"):
            per_frame_vector_series_multiscale(values_4d, valid_3d)

    def test_partial_validity_correct_mean(self) -> None:
        """Partial validity computes correct component-wise mean."""
        _T, _N, _P, _C, _S = 1, 2, 1, 2, 1
        # Walker 0 has [1, 2], walker 1 has [3, 4]
        values = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
        # Only walker 0 is valid
        valid = torch.tensor([[[[True], [False]]]])

        result = per_frame_vector_series_multiscale(values, valid)
        expected = torch.tensor([[[1.0, 2.0]]])
        assert torch.allclose(result, expected, atol=1e-5)

    def test_different_scales_different_means(self) -> None:
        """Different per-scale validity yields different vector averages."""
        T, N, P, C = 1, 3, 1, 2
        S = 2
        # Walkers: [1,0], [0,2], [3,3]
        values = torch.tensor([[[[1.0, 0.0]], [[0.0, 2.0]], [[3.0, 3.0]]]])
        valid = torch.zeros(T, S, N, P, dtype=torch.bool)
        # Scale 0: only walker 0
        valid[0, 0, 0, 0] = True
        # Scale 1: walkers 0 and 1
        valid[0, 1, 0, 0] = True
        valid[0, 1, 1, 0] = True

        result = per_frame_vector_series_multiscale(values, valid)
        assert result.shape == (S, T, C)
        assert torch.allclose(result[0, 0], torch.tensor([1.0, 0.0]), atol=1e-5)
        assert torch.allclose(result[1, 0], torch.tensor([0.5, 1.0]), atol=1e-5)


# ===================================================================
# Integration tests with gate + averaging pipeline
# ===================================================================


class TestMultiscaleIntegration:
    """Integration tests combining gating and averaging."""

    def test_pair_gate_then_average(self) -> None:
        """Full pipeline: pair gating followed by per_frame_series_multiscale."""
        T, N, P = 3, 10, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N, scale=2.0, seed=42)
        scales = torch.tensor([0.5, 2.0, 50.0])

        gated = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert gated.shape == (T, 3, N, P)

        values = torch.randn(T, N, P)
        series = per_frame_series_multiscale(values, gated)
        assert series.shape == (3, T)
        assert series.dtype == torch.float32

    def test_triplet_gate_then_average(self) -> None:
        """Full pipeline: triplet gating followed by per_frame_series_multiscale."""
        T, N = 3, 10
        triplet_valid = torch.ones(T, N, dtype=torch.bool)
        gen = torch.Generator().manual_seed(55)
        d_ij = torch.rand(T, N, generator=gen) * 5
        d_ik = torch.rand(T, N, generator=gen) * 5
        d_jk = torch.rand(T, N, generator=gen) * 5
        scales = torch.tensor([1.0, 3.0, 10.0])

        gated = gate_triplet_validity_by_scale(triplet_valid, d_ij, d_ik, d_jk, scales)
        assert gated.shape == (T, 3, N)

        values = torch.randn(T, N)
        series = per_frame_series_multiscale(values, gated)
        assert series.shape == (3, T)

    def test_pair_gate_then_vector_average(self) -> None:
        """Full pipeline: pair gating followed by vector averaging."""
        T, N, P, C = 3, 10, 2, 4
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N, scale=2.0, seed=42)
        scales = torch.tensor([0.5, 2.0, 50.0])

        gated = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)

        values = torch.randn(T, N, P, C)
        series = per_frame_vector_series_multiscale(values, gated)
        assert series.shape == (3, T, C)
        assert series.dtype == torch.float32

    def test_largest_scale_mean_matches_global_mean(self) -> None:
        """At a sufficiently large scale, the multiscale mean equals global mean."""
        T, N, P = 2, 6, 2
        pair_valid = torch.ones(T, N, P, dtype=torch.bool)
        pair_indices = _cyclic_pair_indices(T, N, P)
        distances = _make_distance_matrix(T, N, scale=1.0, seed=77)
        huge_scale = distances.max().item() + 100.0
        scales = torch.tensor([huge_scale])

        gated = gate_pair_validity_by_scale(pair_valid, pair_indices, distances, scales)
        assert gated.all()

        values = torch.randn(T, N, P)
        series = per_frame_series_multiscale(values, gated)  # [1, T]

        # Compare to simple global mean per frame
        for t in range(T):
            global_mean = values[t].mean().item()
            assert abs(series[0, t].item() - global_mean) < 1e-5
