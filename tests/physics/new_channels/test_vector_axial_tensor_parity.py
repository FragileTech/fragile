"""Tests for vector, axial vector, and tensor operator parity.

Verifies that with purely imaginary gamma matrices (Levi-Civita for γ_μ,
anti-symmetric for σ_μν), the Re/Im projections select correct parity:

For purely imaginary M (M = -M*):
    z_P = c_i^T M c_j* = -c_i^T M* c_j* = -z* under parity c → -c*
    → Re[z_P] = -Re[z] (parity-odd)
    → Im[z_P] = +Im[z] (parity-even)

1. Vector (1--):   Re[c_i† (iε_μ) c_j] — parity-odd  (correct)
2. Axial (1+-):    Im[c_i† (iε_μ) c_j] — parity-even (correct)
3. Tensor (2++):   Im[c_i† σ_μν c_j]   — parity-even (correct)
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
# Test 1: Gamma matrix properties
# ---------------------------------------------------------------------------


class TestGammaMatrixProperties:
    """Verify Levi-Civita γ_μ matrices are purely imaginary, anti-symmetric, Hermitian."""

    @pytest.fixture
    def gamma_mu(self):
        d = 3
        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        return torch.stack(gamma_mu_list, dim=0)

    def test_purely_imaginary(self, gamma_mu):
        """γ_μ should have zero real part."""
        assert torch.allclose(gamma_mu.real, torch.zeros_like(gamma_mu.real), atol=1e-15)

    def test_anti_symmetric(self, gamma_mu):
        """γ_μ should be anti-symmetric: γ_μ^T = -γ_μ."""
        for mu in range(gamma_mu.shape[0]):
            assert torch.allclose(
                gamma_mu[mu].T, -gamma_mu[mu], atol=1e-15
            ), f"γ_{mu} is not anti-symmetric"

    def test_hermitian(self, gamma_mu):
        """γ_μ should be Hermitian: γ_μ† = γ_μ (since purely imaginary + anti-symmetric)."""
        for mu in range(gamma_mu.shape[0]):
            assert torch.allclose(
                gamma_mu[mu].conj().T, gamma_mu[mu], atol=1e-15
            ), f"γ_{mu} is not Hermitian"

    def test_correct_count(self, gamma_mu):
        """For d=3, should have exactly 3 gamma matrices."""
        assert gamma_mu.shape == (3, 3, 3)

    def test_general_d(self):
        """Test Levi-Civita construction for d=4, 5."""
        for d in [4, 5]:
            gamma_mu_list = []
            for mu in range(d):
                gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
                nu = (mu + 1) % d
                gamma_mu[mu, nu] = 1.0j
                gamma_mu[nu, mu] = -1.0j
                gamma_mu_list.append(gamma_mu)
            gamma = torch.stack(gamma_mu_list, dim=0)
            assert gamma.shape == (d, d, d)
            assert torch.allclose(gamma.real, torch.zeros_like(gamma.real), atol=1e-15)


# ---------------------------------------------------------------------------
# Test 2: Vector is parity-odd
# ---------------------------------------------------------------------------


class TestVectorParityOdd:
    """Re[c_i† (iε_μ) c_j] must flip sign under parity."""

    def test_real_projection_flips_sign(self):
        gen = torch.Generator().manual_seed(42)
        d = 3
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        # Build Levi-Civita γ_μ
        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        # Original vector: Re[c_i† γ_μ c_j] averaged over μ
        result = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)
        vec_orig = result.mean().real

        # Parity-transformed
        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        result_p = torch.einsum("i,mij,j->m", c_i_p.conj(), gamma_mu, c_j_p)
        vec_parity = result_p.mean().real

        assert torch.allclose(vec_parity, -vec_orig, atol=1e-6), (
            f"Vector should flip sign under parity: "
            f"orig={vec_orig.item():.6f}, parity={vec_parity.item():.6f}"
        )

    def test_real_projection_flips_sign_batched(self):
        gen = torch.Generator().manual_seed(99)
        T, N, d = 5, 10, 3
        color = torch.complex(
            torch.randn(T, N, d, generator=gen).double(),
            torch.randn(T, N, d, generator=gen).double(),
        )
        idx_j = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
        color_i = color
        color_j = color.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))

        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        vec_orig = result.mean(dim=-1).real

        color_p = apply_parity(color)
        color_i_p = color_p
        color_j_p = color_p.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))
        result_p = torch.einsum("...i,mij,...j->...m", color_i_p.conj(), gamma_mu, color_j_p)
        vec_parity = result_p.mean(dim=-1).real

        assert torch.allclose(vec_parity, -vec_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Axial vector is parity-even
# ---------------------------------------------------------------------------


class TestAxialVectorParityEven:
    """Im[c_i† (iε_μ) c_j] must be invariant under parity."""

    def test_imag_projection_invariant(self):
        gen = torch.Generator().manual_seed(42)
        d = 3
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)
        axial_orig = result.mean().imag

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        result_p = torch.einsum("i,mij,j->m", c_i_p.conj(), gamma_mu, c_j_p)
        axial_parity = result_p.mean().imag

        assert torch.allclose(axial_parity, axial_orig, atol=1e-6), (
            f"Axial vector should be invariant under parity: "
            f"orig={axial_orig.item():.6f}, parity={axial_parity.item():.6f}"
        )

    def test_imag_projection_invariant_batched(self):
        gen = torch.Generator().manual_seed(99)
        T, N, d = 5, 10, 3
        color = torch.complex(
            torch.randn(T, N, d, generator=gen).double(),
            torch.randn(T, N, d, generator=gen).double(),
        )
        idx_j = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
        color_i = color
        color_j = color.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))

        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        axial_orig = result.mean(dim=-1).imag

        color_p = apply_parity(color)
        color_i_p = color_p
        color_j_p = color_p.gather(1, idx_j.unsqueeze(-1).expand(-1, -1, d))
        result_p = torch.einsum("...i,mij,...j->...m", color_i_p.conj(), gamma_mu, color_j_p)
        axial_parity = result_p.mean(dim=-1).imag

        assert torch.allclose(axial_parity, axial_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Tensor is parity-even
# ---------------------------------------------------------------------------


class TestTensorParityEven:
    """Im[c_i† σ_μν c_j] must be invariant under parity."""

    def test_imag_projection_invariant(self):
        gen = torch.Generator().manual_seed(42)
        d = 3
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        # Build σ_μν (purely imaginary)
        sigma_list = []
        for mu in range(d):
            for nu in range(mu + 1, d):
                sigma = torch.zeros(d, d, dtype=torch.complex128)
                sigma[mu, nu] = 1.0j
                sigma[nu, mu] = -1.0j
                sigma_list.append(sigma)
        sigma = torch.stack(sigma_list, dim=0)

        result = torch.einsum("i,pij,j->p", c_i.conj(), sigma, c_j)
        tensor_orig = result.mean().imag

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        result_p = torch.einsum("i,pij,j->p", c_i_p.conj(), sigma, c_j_p)
        tensor_parity = result_p.mean().imag

        assert torch.allclose(tensor_parity, tensor_orig, atol=1e-6), (
            f"Tensor should be invariant under parity: "
            f"orig={tensor_orig.item():.6f}, parity={tensor_parity.item():.6f}"
        )

    def test_real_projection_flips_sign(self):
        """Verify that Re[c_i† σ_μν c_j] is parity-ODD (confirming old code was wrong)."""
        gen = torch.Generator().manual_seed(42)
        d = 3
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        sigma_list = []
        for mu in range(d):
            for nu in range(mu + 1, d):
                sigma = torch.zeros(d, d, dtype=torch.complex128)
                sigma[mu, nu] = 1.0j
                sigma[nu, mu] = -1.0j
                sigma_list.append(sigma)
        sigma = torch.stack(sigma_list, dim=0)

        result = torch.einsum("i,pij,j->p", c_i.conj(), sigma, c_j)
        tensor_orig = result.mean().real

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        result_p = torch.einsum("i,pij,j->p", c_i_p.conj(), sigma, c_j_p)
        tensor_parity = result_p.mean().real

        # Re should flip sign (parity-odd) — this is why old .real was wrong for 2++
        assert torch.allclose(tensor_parity, -tensor_orig, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 5: Vector and axial vector are independent
# ---------------------------------------------------------------------------


class TestVectorAxialIndependence:
    """Vector (Re) and axial (Im) projections should not be trivially equal."""

    def test_not_equal(self):
        gen = torch.Generator().manual_seed(42)
        d = 3
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)
        vec = result.mean().real
        axial = result.mean().imag

        assert not torch.allclose(vec, axial, atol=1e-6), (
            "Vector and axial vector should differ for generic complex inputs"
        )

    def test_orthogonal_parity_structure(self):
        """Under parity, Re flips while Im stays — confirming orthogonal quantum numbers."""
        gen = torch.Generator().manual_seed(77)
        d = 4
        c_i = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )
        c_j = torch.complex(
            torch.randn(d, generator=gen).double(),
            torch.randn(d, generator=gen).double(),
        )

        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, dtype=torch.complex128)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result_orig = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)

        c_i_p = apply_parity(c_i)
        c_j_p = apply_parity(c_j)
        result_parity = torch.einsum("i,mij,j->m", c_i_p.conj(), gamma_mu, c_j_p)

        # Re part flips (vector = parity-odd), Im part stays (axial = parity-even)
        assert torch.allclose(result_parity.real, -result_orig.real, atol=1e-6)
        assert torch.allclose(result_parity.imag, result_orig.imag, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Known numerical values
# ---------------------------------------------------------------------------


class TestNumericalCorrectness:
    """Verify vector/axial/tensor projections with hand-computed values."""

    def test_vector_known_value(self):
        """For specific inputs, verify Re[c_i† (iε_μ) c_j]."""
        # d=3, γ_0: [0,1]=i, [1,0]=-i
        # c_i = [1, 0, 0], c_j = [0, 1, 0]
        c_i = torch.tensor([1 + 0j, 0 + 0j, 0 + 0j], dtype=torch.complex128)
        c_j = torch.tensor([0 + 0j, 1 + 0j, 0 + 0j], dtype=torch.complex128)

        # γ_0: only non-zero at [0,1]=i, [1,0]=-i
        # c_i† γ_0 c_j = conj(c_i[0]) * γ_0[0,1] * c_j[1] = 1 * i * 1 = i
        # γ_1: only non-zero at [1,2]=i, [2,1]=-i → c_i† γ_1 c_j = 0
        # γ_2: only non-zero at [2,0]=i, [0,2]=-i → c_i† γ_2 c_j = 1 * (-i) * 0 = 0

        gamma_mu_list = []
        for mu in range(3):
            gamma_mu = torch.zeros(3, 3, dtype=torch.complex128)
            nu = (mu + 1) % 3
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)
        # μ=0: i, μ=1: 0, μ=2: 0 → mean = i/3
        vec = result.mean().real  # Re(i/3) = 0
        axial = result.mean().imag  # Im(i/3) = 1/3

        assert torch.allclose(vec, torch.tensor(0.0, dtype=torch.float64), atol=1e-7)
        assert torch.allclose(axial, torch.tensor(1.0 / 3.0, dtype=torch.float64), atol=1e-7)

    def test_tensor_known_value(self):
        """For specific inputs, verify Im[c_i† σ_μν c_j]."""
        c_i = torch.tensor([1 + 0j, 0 + 0j, 0 + 0j], dtype=torch.complex128)
        c_j = torch.tensor([0 + 0j, 1 + 0j, 0 + 0j], dtype=torch.complex128)

        # σ_{01}: [0,1]=i, [1,0]=-i
        # c_i† σ_{01} c_j = conj(c_i[0]) * σ_{01}[0,1] * c_j[1] = 1 * i * 1 = i
        # Other σ pairs give 0 for these inputs
        sigma_list = []
        for mu in range(3):
            for nu in range(mu + 1, 3):
                sigma = torch.zeros(3, 3, dtype=torch.complex128)
                sigma[mu, nu] = 1.0j
                sigma[nu, mu] = -1.0j
                sigma_list.append(sigma)
        sigma = torch.stack(sigma_list, dim=0)

        result = torch.einsum("i,pij,j->p", c_i.conj(), sigma, c_j)
        # p=0 (01): i, p=1 (02): 0, p=2 (12): 0 → mean = i/3
        tensor_val = result.mean().imag  # Im(i/3) = 1/3

        assert torch.allclose(tensor_val, torch.tensor(1.0 / 3.0, dtype=torch.float64), atol=1e-7)

    def test_real_colors_vector(self):
        """When colors are purely real, vector Re = 0 (since γ_μ is purely imaginary)."""
        c_i = torch.tensor([1.0, 2.0, 3.0]).to(torch.complex128)
        c_j = torch.tensor([4.0, 5.0, 6.0]).to(torch.complex128)

        gamma_mu_list = []
        for mu in range(3):
            gamma_mu = torch.zeros(3, 3, dtype=torch.complex128)
            nu = (mu + 1) % 3
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        gamma_mu = torch.stack(gamma_mu_list, dim=0)

        result = torch.einsum("i,mij,j->m", c_i.conj(), gamma_mu, c_j)
        vec = result.mean().real
        # For real colors and purely imaginary M: c† M c is purely imaginary → Re = 0
        assert torch.allclose(vec, torch.tensor(0.0, dtype=torch.float64), atol=1e-7)


# ---------------------------------------------------------------------------
# Test 7: Channel class integration
# ---------------------------------------------------------------------------


class TestChannelIntegration:
    """Verify VectorChannel, AxialVectorChannel, TensorChannel produce correct projections."""

    def test_vector_channel_returns_real(self):
        from fragile.physics.new_channels.correlator_channels import VectorChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = VectorChannel(history)

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

        # Manually compute expected: Re[einsum with Levi-Civita γ_μ]
        gamma_mu = channel.gamma["mu"].to(color_i.dtype)
        expected = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        expected = expected.mean(dim=-1).real

        assert torch.allclose(result, expected, atol=1e-6)

    def test_axial_vector_channel_returns_imag(self):
        from fragile.physics.new_channels.correlator_channels import AxialVectorChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = AxialVectorChannel(history)

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

        # Manually compute expected: Im[einsum with Levi-Civita γ_μ]
        gamma_mu = channel.gamma["mu"].to(color_i.dtype)
        expected = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        expected = expected.mean(dim=-1).imag

        assert torch.allclose(result, expected, atol=1e-6)

    def test_tensor_channel_returns_imag(self):
        from fragile.physics.new_channels.correlator_channels import TensorChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = TensorChannel(history)

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

        sigma = channel.gamma["sigma"].to(color_i.dtype)
        expected = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
        expected = expected.mean(dim=-1).imag

        assert torch.allclose(result, expected, atol=1e-6)

    def test_gamma_mu_is_purely_imaginary(self):
        """Verify that the built gamma matrices are purely imaginary."""
        from fragile.physics.new_channels.correlator_channels import VectorChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = VectorChannel(history)

        gamma_mu = channel.gamma["mu"]
        assert torch.allclose(gamma_mu.real, torch.zeros_like(gamma_mu.real), atol=1e-15)

    def test_no_5mu_key(self):
        """Verify that '5mu' key is no longer in gamma dict."""
        from fragile.physics.new_channels.correlator_channels import VectorChannel
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=123)
        channel = VectorChannel(history)

        assert "5mu" not in channel.gamma


# ---------------------------------------------------------------------------
# Test 8: Aggregation function tests
# ---------------------------------------------------------------------------


class TestAggregationFunctions:
    """Verify aggregation functions use correct projections."""

    def _make_agg_data(self, T=5, N=8, d=3, seed=314):
        from fragile.fractalai.qft.aggregation import AggregatedTimeSeries

        gen = torch.Generator().manual_seed(seed)
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
        return agg_data, color, device

    def test_vector_parity_through_aggregation(self):
        """Verify vector operators are parity-odd through aggregation."""
        from fragile.fractalai.qft.aggregation import (
            build_gamma_matrices,
            compute_vector_operators,
        )

        agg_data, color, device = self._make_agg_data(seed=271)
        gamma = build_gamma_matrices(3, device)

        vec_orig = compute_vector_operators(agg_data, gamma)

        # Parity-transformed
        from fragile.fractalai.qft.aggregation import AggregatedTimeSeries

        color_p = apply_parity(color)
        norms_p = color_p.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color_p = color_p / norms_p

        agg_parity = AggregatedTimeSeries(
            color=color_p,
            color_valid=agg_data.color_valid,
            sample_indices=agg_data.sample_indices,
            neighbor_indices=agg_data.neighbor_indices,
            alive=agg_data.alive,
            n_timesteps=agg_data.n_timesteps,
            n_walkers=agg_data.n_walkers,
            d=agg_data.d,
            dt=1.0,
            device=device,
        )

        vec_parity = compute_vector_operators(agg_parity, gamma)

        assert torch.allclose(vec_parity, -vec_orig, atol=1e-5), (
            f"Vector should flip sign. max diff = "
            f"{(vec_parity + vec_orig).abs().max().item()}"
        )

    def test_axial_vector_parity_through_aggregation(self):
        """Verify axial vector operators are parity-even through aggregation."""
        from fragile.fractalai.qft.aggregation import (
            build_gamma_matrices,
            compute_axial_vector_operators,
        )

        agg_data, color, device = self._make_agg_data(seed=271)
        gamma = build_gamma_matrices(3, device)

        axial_orig = compute_axial_vector_operators(agg_data, gamma)

        from fragile.fractalai.qft.aggregation import AggregatedTimeSeries

        color_p = apply_parity(color)
        norms_p = color_p.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color_p = color_p / norms_p

        agg_parity = AggregatedTimeSeries(
            color=color_p,
            color_valid=agg_data.color_valid,
            sample_indices=agg_data.sample_indices,
            neighbor_indices=agg_data.neighbor_indices,
            alive=agg_data.alive,
            n_timesteps=agg_data.n_timesteps,
            n_walkers=agg_data.n_walkers,
            d=agg_data.d,
            dt=1.0,
            device=device,
        )

        axial_parity = compute_axial_vector_operators(agg_parity, gamma)

        assert torch.allclose(axial_parity, axial_orig, atol=1e-5), (
            f"Axial vector should be invariant. max diff = "
            f"{(axial_parity - axial_orig).abs().max().item()}"
        )

    def test_tensor_parity_through_aggregation(self):
        """Verify tensor operators are parity-even through aggregation."""
        from fragile.fractalai.qft.aggregation import (
            build_gamma_matrices,
            compute_tensor_operators,
        )

        agg_data, color, device = self._make_agg_data(seed=271)
        gamma = build_gamma_matrices(3, device)

        tensor_orig = compute_tensor_operators(agg_data, gamma)

        from fragile.fractalai.qft.aggregation import AggregatedTimeSeries

        color_p = apply_parity(color)
        norms_p = color_p.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        color_p = color_p / norms_p

        agg_parity = AggregatedTimeSeries(
            color=color_p,
            color_valid=agg_data.color_valid,
            sample_indices=agg_data.sample_indices,
            neighbor_indices=agg_data.neighbor_indices,
            alive=agg_data.alive,
            n_timesteps=agg_data.n_timesteps,
            n_walkers=agg_data.n_walkers,
            d=agg_data.d,
            dt=1.0,
            device=device,
        )

        tensor_parity = compute_tensor_operators(agg_parity, gamma)

        assert torch.allclose(tensor_parity, tensor_orig, atol=1e-5), (
            f"Tensor should be invariant. max diff = "
            f"{(tensor_parity - tensor_orig).abs().max().item()}"
        )

    def test_build_gamma_no_5mu_key(self):
        """Verify build_gamma_matrices no longer produces '5mu' key."""
        from fragile.fractalai.qft.aggregation import build_gamma_matrices

        gamma = build_gamma_matrices(3, torch.device("cpu"))
        assert "5mu" not in gamma
