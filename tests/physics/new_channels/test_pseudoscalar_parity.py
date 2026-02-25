"""Tests for pseudoscalar parity fix: Im[c_i† c_j] projection.

Verifies that:
1. Pseudoscalar operator is parity-ODD: Im[z] -> -Im[z] under parity.
2. Scalar operator is parity-EVEN: Re[z] -> Re[z] under parity.
3. Scalar and pseudoscalar outputs are independent (not trivially equal).
4. Numerical correctness with known inputs.
5. PseudoscalarChannel._apply_gamma_projection returns imaginary part.
6. compute_pseudoscalar_operators uses Im (not gamma5-weighted Re).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Parity transform: c^α → -(c^α)*
# ---------------------------------------------------------------------------


def apply_parity(color: Tensor) -> Tensor:
    """Parity transform on color states: c^α → -(c^α)*."""
    return -color.conj()


# ---------------------------------------------------------------------------
# Test 1: Pseudoscalar is parity-odd
# ---------------------------------------------------------------------------


class TestPseudoscalarParityOdd:
    """Im[c_i† c_j] must flip sign under parity."""

    def test_imag_dot_flips_sign(self):
        gen = torch.Generator().manual_seed(42)
        d = 4
        c_i = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )

        # Original pseudoscalar: Im[c_i† c_j]
        ps_orig = (c_i.conj() * c_j).sum(dim=-1).imag

        # Parity-transformed
        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        ps_parity = (c_i_p.conj() * c_j_p).sum(dim=-1).imag

        # Parity-odd means sign flip
        assert torch.allclose(ps_parity, -ps_orig, atol=1e-6), (
            f"Pseudoscalar should flip sign under parity: "
            f"orig={ps_orig.item():.6f}, parity={ps_parity.item():.6f}"
        )

    def test_imag_dot_flips_sign_batched(self):
        gen = torch.Generator().manual_seed(99)
        T, N, d = 5, 10, 4
        color = torch.complex(
            torch.randn(T, N, d, generator=gen),
            torch.randn(T, N, d, generator=gen),
        )
        # Use cyclic shift as neighbor
        idx_j = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
        color_i = color
        color_j = color.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))

        ps_orig = (color_i.conj() * color_j).sum(dim=-1).imag

        color_p = apply_parity(color)
        color_i_p = color_p
        color_j_p = color_p.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))
        ps_parity = (color_i_p.conj() * color_j_p).sum(dim=-1).imag

        assert torch.allclose(ps_parity, -ps_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Scalar is parity-even
# ---------------------------------------------------------------------------


class TestScalarParityEven:
    """Re[c_i† c_j] must be invariant under parity."""

    def test_real_dot_invariant(self):
        gen = torch.Generator().manual_seed(42)
        d = 4
        c_i = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )

        sc_orig = (c_i.conj() * c_j).sum(dim=-1).real

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        sc_parity = (c_i_p.conj() * c_j_p).sum(dim=-1).real

        assert torch.allclose(sc_parity, sc_orig, atol=1e-6), (
            f"Scalar should be invariant under parity: "
            f"orig={sc_orig.item():.6f}, parity={sc_parity.item():.6f}"
        )

    def test_real_dot_invariant_batched(self):
        gen = torch.Generator().manual_seed(99)
        T, N, d = 5, 10, 4
        color = torch.complex(
            torch.randn(T, N, d, generator=gen),
            torch.randn(T, N, d, generator=gen),
        )
        idx_j = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
        color_i = color
        color_j = color.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))

        sc_orig = (color_i.conj() * color_j).sum(dim=-1).real

        color_p = apply_parity(color)
        color_i_p = color_p
        color_j_p = color_p.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))
        sc_parity = (color_i_p.conj() * color_j_p).sum(dim=-1).real

        assert torch.allclose(sc_parity, sc_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Scalar and pseudoscalar are independent
# ---------------------------------------------------------------------------


class TestScalarPseudoscalarIndependence:
    """Scalar (Re) and pseudoscalar (Im) outputs should not be trivially equal."""

    def test_not_equal(self):
        gen = torch.Generator().manual_seed(42)
        d = 4
        c_i = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )

        dot = (c_i.conj() * c_j).sum(dim=-1)
        sc = dot.real
        ps = dot.imag

        assert not torch.allclose(sc, ps, atol=1e-6), (
            "Scalar and pseudoscalar should differ for generic complex inputs"
        )

    def test_orthogonal_parity_structure(self):
        """Under parity, scalar is unchanged while pseudoscalar flips.
        This confirms they carry different quantum numbers."""
        gen = torch.Generator().manual_seed(77)
        d = 8
        c_i = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen),
            torch.randn(d, generator=gen),
        )

        dot_orig = (c_i.conj() * c_j).sum(dim=-1)

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        dot_parity = (c_i_p.conj() * c_j_p).sum(dim=-1)

        # Re part same, Im part flipped — structurally orthogonal
        assert torch.allclose(dot_parity.real, dot_orig.real, atol=1e-6)
        assert torch.allclose(dot_parity.imag, -dot_orig.imag, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Numerical correctness with known inputs
# ---------------------------------------------------------------------------


class TestNumericalCorrectness:
    """Verify pseudoscalar and scalar projections with hand-computed values."""

    def test_known_value_single_pair(self):
        # c_i = [1+2i, 3+0i], c_j = [0+1i, 2+3i]
        c_i = torch.tensor([1 + 2j, 3 + 0j])
        c_j = torch.tensor([0 + 1j, 2 + 3j])

        # c_i† c_j = conj(c_i) * c_j
        # = [(1-2i)(0+1i), (3-0i)(2+3i)]
        # = [(0+1i-0+2), (6+9i)]
        # = [(2+1i), (6+9i)]
        # sum = 8 + 10i
        dot = (c_i.conj() * c_j).sum(dim=-1)
        assert torch.allclose(dot.real, torch.tensor(8.0), atol=1e-6)
        assert torch.allclose(dot.imag, torch.tensor(10.0), atol=1e-6)

        # Scalar = Re[dot] = 8
        sc = dot.real
        assert torch.allclose(sc, torch.tensor(8.0), atol=1e-6)

        # Pseudoscalar = Im[dot] = 10
        ps = dot.imag
        assert torch.allclose(ps, torch.tensor(10.0), atol=1e-6)

    def test_real_colors_give_zero_pseudoscalar(self):
        """When colors are purely real, Im[c_i† c_j] = 0."""
        c_i = torch.tensor([1.0, 2.0, 3.0]).to(torch.complex64)
        c_j = torch.tensor([4.0, 5.0, 6.0]).to(torch.complex64)

        ps = (c_i.conj() * c_j).sum(dim=-1).imag
        assert torch.allclose(ps, torch.tensor(0.0), atol=1e-7)

    def test_purely_imaginary_colors(self):
        """When colors are purely imaginary, c_i† c_j is real, so Im = 0."""
        c_i = torch.tensor([1j, 2j, 3j])
        c_j = torch.tensor([4j, 5j, 6j])

        # conj(ai) * (bj) = -ai * bj = -ab * i*j = -ab * (-1) = ab  (real)
        ps = (c_i.conj() * c_j).sum(dim=-1).imag
        assert torch.allclose(ps, torch.tensor(0.0), atol=1e-7)

    def test_parity_on_known_value(self):
        """Verify parity transform on the known-value pair."""
        c_i = torch.tensor([1 + 2j, 3 + 0j])
        c_j = torch.tensor([0 + 1j, 2 + 3j])

        ps_orig = (c_i.conj() * c_j).sum(dim=-1).imag  # 10

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        ps_parity = (c_i_p.conj() * c_j_p).sum(dim=-1).imag

        assert torch.allclose(ps_parity, torch.tensor(-10.0), atol=1e-6)

    def test_unit_basis_known_values(self):
        """c_i = [1, 0, 0], c_j = [1j, 0, 0] -> scalar=0, pseudoscalar=1."""
        c_i = torch.tensor([1 + 0j, 0 + 0j, 0 + 0j])
        c_j = torch.tensor([0 + 1j, 0 + 0j, 0 + 0j])

        dot = (c_i.conj() * c_j).sum(dim=-1)
        # conj(1) * 1j = 1j => Re=0, Im=1
        assert torch.allclose(dot.real, torch.tensor(0.0), atol=1e-7)
        assert torch.allclose(dot.imag, torch.tensor(1.0), atol=1e-7)


# ---------------------------------------------------------------------------
# Test 5: PseudoscalarChannel integration
# ---------------------------------------------------------------------------


class TestPseudoscalarChannelIntegration:
    """Verify PseudoscalarChannel._apply_gamma_projection returns Im part."""

    def test_apply_gamma_projection_returns_imag(self):
        """_apply_gamma_projection should return the imaginary part of the
        color dot product, matching the standalone formula."""
        from fragile.physics.new_channels.correlator_channels import PseudoscalarChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = PseudoscalarChannel(history)

        gen = torch.Generator().manual_seed(55)
        T, S, d = 4, 6, 3
        color_i = torch.complex(
            torch.randn(T, S, d, generator=gen),
            torch.randn(T, S, d, generator=gen),
        )
        color_j = torch.complex(
            torch.randn(T, S, d, generator=gen),
            torch.randn(T, S, d, generator=gen),
        )

        result = channel._apply_gamma_projection(color_i, color_j)
        expected = (color_i.conj() * color_j).sum(dim=-1).imag

        assert torch.allclose(result, expected, atol=1e-6), (
            f"_apply_gamma_projection should return Im[c_i† c_j], "
            f"max diff = {(result - expected).abs().max().item()}"
        )

    def test_scalar_channel_returns_real(self):
        """ScalarChannel._apply_gamma_projection should return the real part."""
        from fragile.physics.new_channels.correlator_channels import ScalarChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = ScalarChannel(history)

        gen = torch.Generator().manual_seed(55)
        T, S, d = 4, 6, 3
        color_i = torch.complex(
            torch.randn(T, S, d, generator=gen),
            torch.randn(T, S, d, generator=gen),
        )
        color_j = torch.complex(
            torch.randn(T, S, d, generator=gen),
            torch.randn(T, S, d, generator=gen),
        )

        result = channel._apply_gamma_projection(color_i, color_j)
        expected = (color_i.conj() * color_j).sum(dim=-1).real

        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Aggregation function test
# ---------------------------------------------------------------------------


class TestAggregationPseudoscalar:
    """Verify compute_pseudoscalar_operators uses Im (not gamma5-weighted Re)."""

    def test_pseudoscalar_differs_from_gamma5_weighted(self):
        """The new Im-based pseudoscalar must differ from the old
        gamma5-weighted Re for generic complex color states."""
        from fragile.fractalai.qft.aggregation import (
            AggregatedTimeSeries,
            build_gamma_matrices,
            compute_pseudoscalar_operators,
        )

        gen = torch.Generator().manual_seed(314)
        T, N, d = 5, 8, 3
        device = torch.device("cpu")

        color = torch.complex(
            torch.randn(T, N, d, generator=gen),
            torch.randn(T, N, d, generator=gen),
        )
        # Normalize
        norms = color.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color = color / norms

        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)

        # Cyclic shift neighbors
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = (
            torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)
        )

        agg_data = AggregatedTimeSeries(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
            n_timesteps=T,
            n_walkers=N,
            d=d,
            dt=1.0,
            device=device,
        )

        gamma = build_gamma_matrices(d, device)

        # New (correct): Im[c_i† c_j]
        ps_new = compute_pseudoscalar_operators(agg_data, gamma)

        # Old (wrong): Re[c_i† gamma5 c_j]
        # Manually compute the old gamma5-weighted version
        t_idx = torch.arange(T).unsqueeze(1).expand(-1, N)
        color_i = color[t_idx, sample_indices]
        first_neighbor = neighbor_indices[:, :, 0]
        color_j = color[t_idx, first_neighbor]
        gamma5_diag = gamma["5"]  # [d]
        gamma5_projected = color_j * gamma5_diag.unsqueeze(0).unsqueeze(0)
        old_op = (color_i.conj() * gamma5_projected).sum(dim=-1).real
        valid_mask = first_neighbor != sample_indices
        old_op = torch.where(valid_mask, old_op, torch.zeros_like(old_op))
        counts = valid_mask.sum(dim=1).clamp(min=1).float()
        ps_old = old_op.sum(dim=1) / counts

        # They must differ for generic complex input
        # Cast to same dtype for comparison (gamma matrices are complex128)
        ps_new_f = ps_new.double()
        ps_old_f = ps_old.double()
        assert not torch.allclose(ps_new_f, ps_old_f, atol=1e-4), (
            "New Im-based pseudoscalar should differ from old gamma5-weighted Re. "
            f"max diff = {(ps_new_f - ps_old_f).abs().max().item()}"
        )

    def test_pseudoscalar_parity_through_aggregation(self):
        """Verify parity-odd behavior through the aggregation function."""
        from fragile.fractalai.qft.aggregation import (
            AggregatedTimeSeries,
            build_gamma_matrices,
            compute_pseudoscalar_operators,
            compute_scalar_operators,
        )

        gen = torch.Generator().manual_seed(271)
        T, N, d = 4, 6, 3
        device = torch.device("cpu")

        color = torch.complex(
            torch.randn(T, N, d, generator=gen),
            torch.randn(T, N, d, generator=gen),
        )
        norms = color.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color = color / norms

        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = (
            torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)
        )

        gamma = build_gamma_matrices(d, device)

        def make_agg(c):
            return AggregatedTimeSeries(
                color=c,
                color_valid=color_valid,
                sample_indices=sample_indices,
                neighbor_indices=neighbor_indices,
                alive=alive,
                n_timesteps=T,
                n_walkers=N,
                d=d,
                dt=1.0,
                device=device,
            )

        agg_orig = make_agg(color)
        ps_orig = compute_pseudoscalar_operators(agg_orig, gamma)
        sc_orig = compute_scalar_operators(agg_orig, gamma)

        # Parity-transformed color
        color_p = apply_parity(color)
        norms_p = color_p.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color_p = color_p / norms_p

        agg_parity = make_agg(color_p)
        ps_parity = compute_pseudoscalar_operators(agg_parity, gamma)
        sc_parity = compute_scalar_operators(agg_parity, gamma)

        # Pseudoscalar flips sign under parity
        assert torch.allclose(ps_parity, -ps_orig, atol=1e-5), (
            f"Pseudoscalar should flip sign. max diff = "
            f"{(ps_parity + ps_orig).abs().max().item()}"
        )
        # Scalar is invariant
        assert torch.allclose(sc_parity, sc_orig, atol=1e-5), (
            f"Scalar should be invariant. max diff = "
            f"{(sc_parity - sc_orig).abs().max().item()}"
        )
