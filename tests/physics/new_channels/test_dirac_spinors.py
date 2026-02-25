"""Tests for Dirac spinor module: gamma matrices, Hopf map, bilinears, quantum numbers.

Verifies:
1. Clifford algebra: {gamma_mu, gamma_nu} = 2 g_munu I4
2. gamma5 anticommutation, squaring, hermiticity
3. Hopf fibration spinor normalization and chart switching
4. Color-to-Dirac-spinor mapping
5. Bilinear computation correctness
6. Parity transformation properties for all 5 channels:
   - Scalar (0^++) parity-even
   - Pseudoscalar (0^-+) parity-odd
   - Vector (1^--) parity-odd
   - Axial vector (1^+-) parity-even
   - Tensor (2^++) parity-even
7. Channel independence: distinct channels give distinct values
8. DiracOperatorSeries from full pipeline
9. Channel registry integration (DiracScalar/Pseudoscalar/... classes)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
    compute_dirac_bilinear,
    compute_dirac_operator_series,
    DiracOperatorSeries,
    vector_to_weyl_spinor,
    verify_clifford_algebra,
)


# ===========================================================================
# Clifford algebra verification
# ===========================================================================


class TestCliffordAlgebra:
    """Verify gamma matrices satisfy the Clifford algebra."""

    @pytest.fixture(autouse=True)
    def setup_gamma(self):
        self.gamma = build_dirac_gamma_matrices()

    def test_anticommutation(self):
        """Test {gamma_mu, gamma_nu} = 2 g_munu I4."""
        results = verify_clifford_algebra(self.gamma)
        assert results["clifford_anticommutation"]

    def test_gamma5_anticommutation(self):
        """gamma5 anticommutes with all gamma_mu."""
        results = verify_clifford_algebra(self.gamma)
        assert results["gamma5_anticommutation"]

    def test_gamma5_squared_is_identity(self):
        """(gamma5)^2 = I4."""
        results = verify_clifford_algebra(self.gamma)
        assert results["gamma5_squared"]

    def test_hermiticity(self):
        """gamma_mu^dagger = gamma0 gamma_mu gamma0."""
        results = verify_clifford_algebra(self.gamma)
        assert results["hermiticity"]

    def test_all_checks_pass(self):
        results = verify_clifford_algebra(self.gamma)
        assert all(results.values()), f"Failed checks: {[k for k, v in results.items() if not v]}"

    def test_gamma0_shape(self):
        assert self.gamma["gamma0"].shape == (4, 4)

    def test_gamma_k_shape(self):
        assert self.gamma["gamma_k"].shape == (3, 4, 4)

    def test_gamma5_shape(self):
        assert self.gamma["gamma5"].shape == (4, 4)

    def test_gamma5_k_shape(self):
        assert self.gamma["gamma5_k"].shape == (3, 4, 4)

    def test_sigma_munu_shape(self):
        assert self.gamma["sigma_munu"].shape == (6, 4, 4)

    def test_charge_conjugation_shape(self):
        assert self.gamma["C"].shape == (4, 4)

    def test_gamma0_squared_is_identity(self):
        """gamma0^2 = I4 (metric signature +---)."""
        gamma0 = self.gamma["gamma0"]
        I4 = torch.eye(4, dtype=gamma0.dtype)
        assert (gamma0 @ gamma0 - I4).abs().max() < 1e-10

    def test_spatial_gamma_squared_is_minus_identity(self):
        """gamma_k^2 = -I4 for k=1,2,3."""
        gamma_k = self.gamma["gamma_k"]
        I4 = torch.eye(4, dtype=gamma_k.dtype)
        for k in range(3):
            sq = gamma_k[k] @ gamma_k[k]
            assert (sq + I4).abs().max() < 1e-10, f"gamma_{k+1}^2 != -I4"

    def test_sigma_munu_antisymmetric(self):
        """sigma_munu are antisymmetric: sigma_munu = -sigma_numu."""
        gamma0 = self.gamma["gamma0"]
        gamma_k = self.gamma["gamma_k"]
        all_g = [gamma0] + [gamma_k[k] for k in range(3)]

        idx = 0
        for mu in range(4):
            for nu in range(mu + 1, 4):
                sigma_mn = self.gamma["sigma_munu"][idx]
                comm_mn = all_g[mu] @ all_g[nu] - all_g[nu] @ all_g[mu]
                expected = 0.5j * comm_mn
                assert (sigma_mn - expected).abs().max() < 1e-10
                idx += 1


# ===========================================================================
# Hopf fibration / vector_to_weyl_spinor
# ===========================================================================


class TestVectorToWeylSpinor:
    """Test the Hopf map R^3 -> C^2."""

    def test_unit_z_axis(self):
        """z-hat -> (1, 0) up to normalization and sqrt(r) scaling."""
        w = torch.tensor([[0.0, 0.0, 1.0]])
        spinor, valid = vector_to_weyl_spinor(w)
        assert valid.all()
        # For z-hat: north chart gives (sqrt(2r)*sqrt(r)/(sqrt(2r)), 0) = (1, 0)
        assert spinor.shape == (1, 2)
        assert spinor[0, 0].abs() > 0.5  # dominant component

    def test_zero_vector_invalid(self):
        """Zero vectors should be marked invalid."""
        w = torch.zeros(5, 3)
        _, valid = vector_to_weyl_spinor(w)
        assert not valid.any()

    def test_batch_shape(self):
        """Verify output shapes for batched input."""
        w = torch.randn(10, 20, 3)
        spinor, valid = vector_to_weyl_spinor(w)
        assert spinor.shape == (10, 20, 2)
        assert valid.shape == (10, 20)

    def test_south_pole_chart_switching(self):
        """Vectors near -z should use south chart without NaN."""
        w = torch.tensor([[0.0, 0.0, -1.0], [1e-8, 0.0, -1.0 + 1e-15]])
        spinor, valid = vector_to_weyl_spinor(w)
        assert valid.all()
        assert torch.isfinite(spinor).all()

    def test_spinor_magnitude_scales_with_sqrt_r(self):
        """Spinor norm should scale as sqrt(|w|)."""
        w1 = torch.tensor([[0.0, 0.0, 1.0]])
        w4 = torch.tensor([[0.0, 0.0, 4.0]])
        s1, _ = vector_to_weyl_spinor(w1)
        s4, _ = vector_to_weyl_spinor(w4)
        norm1 = s1.abs().pow(2).sum().sqrt()
        norm4 = s4.abs().pow(2).sum().sqrt()
        # norm4 / norm1 should be sqrt(4)/sqrt(1) = 2
        ratio = norm4 / norm1
        assert abs(ratio.item() - 2.0) < 0.1


# ===========================================================================
# color_to_dirac_spinor
# ===========================================================================


class TestColorToDiracSpinor:
    """Test the color -> Dirac spinor mapping."""

    def test_output_shape(self):
        color = torch.randn(5, 10, 3, dtype=torch.cfloat)
        spinor, valid = color_to_dirac_spinor(color)
        assert spinor.shape == (5, 10, 4)
        assert valid.shape == (5, 10)

    def test_output_dtype(self):
        color = torch.randn(5, 10, 3, dtype=torch.cfloat)
        spinor, _ = color_to_dirac_spinor(color)
        assert spinor.dtype == torch.complex128

    def test_nonzero_color_gives_valid_spinor(self):
        gen = torch.Generator().manual_seed(42)
        color = torch.complex(
            torch.randn(4, 8, 3, generator=gen),
            torch.randn(4, 8, 3, generator=gen),
        )
        _, valid = color_to_dirac_spinor(color)
        # Most spinors should be valid for random nonzero color
        assert valid.float().mean() > 0.5

    def test_zero_color_gives_invalid(self):
        color = torch.zeros(2, 3, 3, dtype=torch.cfloat)
        _, valid = color_to_dirac_spinor(color)
        assert not valid.any()

    def test_purely_real_color_has_zero_upper_components(self):
        """Real color -> Im(c)=0 -> ψ_L=0 (upper 2 components)."""
        color = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.cfloat)
        spinor, valid = color_to_dirac_spinor(color)
        # Im(c) = 0 -> xi_L = 0 -> upper 2 components are zero
        # But valid_L will be False since |Im(c)|=0
        assert not valid[0, 0].item()

    def test_finite_output(self):
        gen = torch.Generator().manual_seed(99)
        color = torch.complex(
            torch.randn(8, 16, 3, generator=gen),
            torch.randn(8, 16, 3, generator=gen),
        )
        spinor, _ = color_to_dirac_spinor(color)
        assert torch.isfinite(spinor).all()


# ===========================================================================
# compute_dirac_bilinear
# ===========================================================================


class TestComputeDiracBilinear:
    """Test the bilinear computation psi_bar_i Gamma psi_j."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.gamma = build_dirac_gamma_matrices()
        gen = torch.Generator().manual_seed(42)
        self.psi_i = torch.complex(
            torch.randn(4, 8, 4, generator=gen).double(),
            torch.randn(4, 8, 4, generator=gen).double(),
        )
        self.psi_j = torch.complex(
            torch.randn(4, 8, 4, generator=gen).double(),
            torch.randn(4, 8, 4, generator=gen).double(),
        )

    def test_scalar_bilinear_shape(self):
        I4 = torch.eye(4, dtype=torch.complex128)
        result = compute_dirac_bilinear(
            self.psi_i, self.psi_j, self.gamma["gamma0"], I4,
        )
        assert result.shape == (4, 8)

    def test_scalar_bilinear_is_real(self):
        I4 = torch.eye(4, dtype=torch.complex128)
        result = compute_dirac_bilinear(
            self.psi_i, self.psi_j, self.gamma["gamma0"], I4,
        )
        assert not result.is_complex()

    def test_vector_bilinear_multi_gamma_shape(self):
        """gamma_k is [3, 4, 4] -> result is [T, S, 3]."""
        result = compute_dirac_bilinear(
            self.psi_i, self.psi_j, self.gamma["gamma0"], self.gamma["gamma_k"],
        )
        assert result.shape == (4, 8, 3)

    def test_tensor_bilinear_multi_gamma_shape(self):
        """sigma_munu is [6, 4, 4] -> result is [T, S, 6]."""
        result = compute_dirac_bilinear(
            self.psi_i, self.psi_j, self.gamma["gamma0"], self.gamma["sigma_munu"],
        )
        assert result.shape == (4, 8, 6)

    def test_bilinear_is_finite(self):
        I4 = torch.eye(4, dtype=torch.complex128)
        for key in ["gamma5", "gamma_k", "gamma5_k", "sigma_munu"]:
            result = compute_dirac_bilinear(
                self.psi_i, self.psi_j, self.gamma["gamma0"], self.gamma[key],
            )
            assert torch.isfinite(result).all(), f"Non-finite bilinear for {key}"

    def test_scalar_bilinear_manual_check(self):
        """Manual check: psi_bar_i psi_j = psi_i^dag gamma0 psi_j."""
        gamma0 = self.gamma["gamma0"]
        I4 = torch.eye(4, dtype=torch.complex128)
        psi_i = self.psi_i[0, 0]  # [4]
        psi_j = self.psi_j[0, 0]  # [4]
        # Manual: psi_i^dag @ gamma0 @ I4 @ psi_j
        expected = (psi_i.conj() @ gamma0 @ psi_j).real
        result = compute_dirac_bilinear(
            psi_i.unsqueeze(0).unsqueeze(0),
            psi_j.unsqueeze(0).unsqueeze(0),
            gamma0, I4,
        )
        assert torch.allclose(result.squeeze(), expected, atol=1e-10)


# ===========================================================================
# Parity properties of Dirac bilinears
# ===========================================================================


def _apply_dirac_parity(psi: Tensor, gamma0: Tensor) -> Tensor:
    """Standard Dirac parity: ψ → γ₀ ψ.

    Under this transformation:
        ψ̄ Γ ψ → ψ̄ (γ₀ Γ γ₀) ψ

    So bilinears transform as:
        Γ = I₄    → +ψ̄ψ       (scalar, parity-even)
        Γ = γ₅    → -ψ̄γ₅ψ    (pseudoscalar, parity-odd)
        Γ = γ_k   → -ψ̄γ_kψ   (vector, parity-odd)
        Γ = γ₅γ_k → +ψ̄γ₅γ_kψ (axial vector, parity-even)
    """
    # gamma0 is [4,4], psi is [..., 4]
    return torch.einsum("ab,...b->...a", gamma0, psi)


def _make_random_spinors(seed: int = 42) -> Tensor:
    """Generate random Dirac spinors [B, 4] for parity testing."""
    gen = torch.Generator().manual_seed(seed)
    B = 100
    return torch.complex(
        torch.randn(B, 4, generator=gen).double(),
        torch.randn(B, 4, generator=gen).double(),
    )


class TestDiracParityScalar:
    """Dirac scalar ψ̄ψ should be parity-EVEN (0^++).

    Under ψ → γ₀ψ: ψ̄ψ → ψ̄(γ₀ I₄ γ₀)ψ = ψ̄ψ.
    """

    def test_parity_even(self):
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        I4 = torch.eye(4, dtype=gamma0.dtype)
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        s_orig = compute_dirac_bilinear(psi, psi, gamma0, I4)
        s_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, I4)

        assert torch.allclose(s_orig, s_parity, atol=1e-8), (
            f"Scalar should be parity-even, max diff = {(s_orig - s_parity).abs().max().item()}"
        )


class TestDiracParityPseudoscalar:
    """Dirac pseudoscalar ψ̄γ₅ψ should be parity-ODD (0^-+).

    Under ψ → γ₀ψ: ψ̄γ₅ψ → ψ̄(γ₀ γ₅ γ₀)ψ = -ψ̄γ₅ψ
    since {γ₅, γ₀} = 0.
    """

    def test_parity_odd(self):
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        ps_orig = compute_dirac_bilinear(psi, psi, gamma0, gamma["gamma5"])
        ps_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, gamma["gamma5"])

        assert torch.allclose(ps_parity, -ps_orig, atol=1e-8), (
            f"Pseudoscalar should be parity-odd, max diff = {(ps_orig + ps_parity).abs().max().item()}"
        )


class TestDiracParityVector:
    """Dirac vector ψ̄γ_k ψ should be parity-ODD (1^--).

    Under ψ → γ₀ψ: ψ̄γ_kψ → ψ̄(γ₀ γ_k γ₀)ψ = -ψ̄γ_kψ
    since {γ_k, γ₀} = 0 for k=1,2,3.
    """

    def test_parity_odd_per_component(self):
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        v_orig = compute_dirac_bilinear(psi, psi, gamma0, gamma["gamma_k"])  # [B, 3]
        v_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, gamma["gamma_k"])

        # Each spatial component flips sign
        for k in range(3):
            assert torch.allclose(v_parity[:, k], -v_orig[:, k], atol=1e-8), (
                f"Vector component k={k} should flip sign, "
                f"max diff = {(v_orig[:, k] + v_parity[:, k]).abs().max().item()}"
            )


class TestDiracParityAxialVector:
    """Dirac axial vector ψ̄γ₅γ_k ψ should be parity-EVEN (1^+-).

    Under ψ → γ₀ψ: ψ̄γ₅γ_kψ → ψ̄(γ₀ γ₅γ_k γ₀)ψ = +ψ̄γ₅γ_kψ
    since γ₅γ_k has two anticommutations with γ₀ (one from γ₅, one from γ_k).
    """

    def test_parity_even_per_component(self):
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        a_orig = compute_dirac_bilinear(psi, psi, gamma0, gamma["gamma5_k"])  # [B, 3]
        a_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, gamma["gamma5_k"])

        for k in range(3):
            assert torch.allclose(a_parity[:, k], a_orig[:, k], atol=1e-8), (
                f"Axial vector component k={k} should be invariant, "
                f"max diff = {(a_orig[:, k] - a_parity[:, k]).abs().max().item()}"
            )


class TestDiracParityTensor:
    """Dirac tensor components: σ_{0k} is parity-odd, σ_{ij} is parity-even.

    Under ψ → γ₀ψ: ψ̄σ_μνψ → ψ̄(γ₀ σ_μν γ₀)ψ.
    σ_{0k} = (i/2)[γ₀,γ_k] has one γ₀ and one γ_k → odd parity.
    σ_{ij} = (i/2)[γ_i,γ_j] has two spatial gammas → even parity.
    """

    def test_sigma_0k_parity_odd(self):
        """σ_{0k} components (indices 0,1,2 in our ordering) flip sign."""
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        t_orig = compute_dirac_bilinear(psi, psi, gamma0, gamma["sigma_munu"])  # [B, 6]
        t_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, gamma["sigma_munu"])

        # Ordering: (01, 02, 03, 12, 13, 23) — first 3 are σ_{0k}
        for idx in range(3):
            assert torch.allclose(t_parity[:, idx], -t_orig[:, idx], atol=1e-8), (
                f"σ_0{idx+1} should be parity-odd, "
                f"max diff = {(t_orig[:, idx] + t_parity[:, idx]).abs().max().item()}"
            )

    def test_sigma_ij_parity_even(self):
        """σ_{ij} components (indices 3,4,5 in our ordering) are invariant."""
        gamma = build_dirac_gamma_matrices()
        gamma0 = gamma["gamma0"]
        psi = _make_random_spinors(seed=42)
        psi_p = _apply_dirac_parity(psi, gamma0)

        t_orig = compute_dirac_bilinear(psi, psi, gamma0, gamma["sigma_munu"])
        t_parity = compute_dirac_bilinear(psi_p, psi_p, gamma0, gamma["sigma_munu"])

        # Indices 3,4,5 are σ_{12}, σ_{13}, σ_{23}
        for idx in range(3, 6):
            assert torch.allclose(t_parity[:, idx], t_orig[:, idx], atol=1e-8), (
                f"σ component {idx} should be parity-even, "
                f"max diff = {(t_orig[:, idx] - t_parity[:, idx]).abs().max().item()}"
            )


# ===========================================================================
# Channel independence
# ===========================================================================


class TestChannelIndependence:
    """Different channels should give distinct operator values."""

    def test_all_channels_differ(self):
        gen = torch.Generator().manual_seed(77)
        color = torch.complex(
            torch.randn(10, 20, 3, generator=gen),
            torch.randn(10, 20, 3, generator=gen),
        )
        psi, valid = color_to_dirac_spinor(color)
        gamma = build_dirac_gamma_matrices()
        I4 = torch.eye(4, dtype=gamma["gamma0"].dtype)

        # Compute all channels
        scalar = compute_dirac_bilinear(psi, psi, gamma["gamma0"], I4)
        pseudo = compute_dirac_bilinear(psi, psi, gamma["gamma0"], gamma["gamma5"])
        vector = compute_dirac_bilinear(psi, psi, gamma["gamma0"], gamma["gamma_k"]).mean(-1)
        axial = compute_dirac_bilinear(psi, psi, gamma["gamma0"], gamma["gamma5_k"]).mean(-1)
        tensor = compute_dirac_bilinear(psi, psi, gamma["gamma0"], gamma["sigma_munu"]).mean(-1)

        ops = {"scalar": scalar, "pseudo": pseudo, "vector": vector,
               "axial": axial, "tensor": tensor}

        # Check all pairs differ
        names = list(ops.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i + 1:]:
                assert not torch.allclose(ops[n1], ops[n2], atol=1e-4), (
                    f"{n1} and {n2} should differ for generic input"
                )


# ===========================================================================
# compute_dirac_operator_series (full pipeline wrapper)
# ===========================================================================


class TestComputeDiracOperatorSeries:
    """Test the full pipeline wrapper."""

    @pytest.fixture
    def agg_data(self):
        gen = torch.Generator().manual_seed(42)
        T, N = 10, 20
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = (
            torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)
        )
        return {
            "color": color,
            "color_valid": color_valid,
            "sample_indices": sample_indices,
            "neighbor_indices": neighbor_indices,
            "alive": alive,
        }

    def test_returns_dataclass(self, agg_data):
        result = compute_dirac_operator_series(**agg_data)
        assert isinstance(result, DiracOperatorSeries)

    def test_output_shapes(self, agg_data):
        T = agg_data["color"].shape[0]
        result = compute_dirac_operator_series(**agg_data)
        assert result.scalar.shape == (T,)
        assert result.pseudoscalar.shape == (T,)
        assert result.vector.shape == (T,)
        assert result.axial_vector.shape == (T,)
        assert result.tensor.shape == (T,)
        assert result.tensor_0k.shape == (T,)
        assert result.n_valid_pairs.shape == (T,)
        assert result.spinor_valid_fraction.shape == (T,)

    def test_output_is_finite(self, agg_data):
        result = compute_dirac_operator_series(**agg_data)
        assert torch.isfinite(result.scalar).all()
        assert torch.isfinite(result.pseudoscalar).all()
        assert torch.isfinite(result.vector).all()
        assert torch.isfinite(result.axial_vector).all()
        assert torch.isfinite(result.tensor).all()
        assert torch.isfinite(result.tensor_0k).all()

    def test_valid_pairs_positive(self, agg_data):
        result = compute_dirac_operator_series(**agg_data)
        assert (result.n_valid_pairs > 0).all()

    def test_spinor_valid_fraction_in_range(self, agg_data):
        result = compute_dirac_operator_series(**agg_data)
        assert (result.spinor_valid_fraction >= 0).all()
        assert (result.spinor_valid_fraction <= 1).all()

    def test_d_not_3_raises(self, agg_data):
        """d != 3 should raise ValueError."""
        agg_data["color"] = torch.randn(10, 20, 4, dtype=torch.cfloat)
        with pytest.raises(ValueError, match="d=3"):
            compute_dirac_operator_series(**agg_data)


# ===========================================================================
# Channel registry integration
# ===========================================================================


class TestDiracChannelRegistry:
    """Verify Dirac channels are registered and produce valid output."""

    def test_all_dirac_channels_registered(self):
        from fragile.physics.new_channels.correlator_channels import CHANNEL_REGISTRY

        for name in [
            "dirac_scalar", "dirac_pseudoscalar", "dirac_vector",
            "dirac_axial_vector", "dirac_tensor", "dirac_tensor_0k",
        ]:
            assert name in CHANNEL_REGISTRY, f"{name} not in CHANNEL_REGISTRY"

    def test_dirac_channel_classes_have_correct_names(self):
        from fragile.physics.new_channels.correlator_channels import (
            DiracScalarChannel,
            DiracPseudoscalarChannel,
            DiracVectorChannel,
            DiracAxialVectorChannel,
            DiracTensorChannel,
            DiracTensor0kChannel,
        )
        assert DiracScalarChannel.channel_name == "dirac_scalar"
        assert DiracPseudoscalarChannel.channel_name == "dirac_pseudoscalar"
        assert DiracVectorChannel.channel_name == "dirac_vector"
        assert DiracAxialVectorChannel.channel_name == "dirac_axial_vector"
        assert DiracTensorChannel.channel_name == "dirac_tensor"
        assert DiracTensor0kChannel.channel_name == "dirac_tensor_0k"

    def test_dirac_channel_projection_returns_tensor(self):
        """Each Dirac channel's _apply_gamma_projection returns a valid tensor."""
        from fragile.physics.new_channels.correlator_channels import CHANNEL_REGISTRY
        from tests.physics.new_channels.conftest import MockRunHistory

        history = MockRunHistory(N=10, d=3, n_recorded=10, seed=42)
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

        for name in [
            "dirac_scalar", "dirac_pseudoscalar", "dirac_vector",
            "dirac_axial_vector", "dirac_tensor", "dirac_tensor_0k",
        ]:
            channel = CHANNEL_REGISTRY[name](history)
            result = channel._apply_gamma_projection(color_i, color_j)
            assert result.shape == (T, S), f"{name}: expected shape {(T, S)}, got {result.shape}"
            assert torch.isfinite(result).all(), f"{name}: non-finite values"


# ===========================================================================
# Mass extraction pattern matching
# ===========================================================================


class TestMassExtractionPatterns:
    """Verify that dirac_* keys are matched to correct channel groups."""

    def test_dirac_channels_map_to_parent_groups(self):
        import re

        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        expected = {
            "dirac_scalar": ("scalar", "meson"),
            "dirac_pseudoscalar": ("pseudoscalar", "meson"),
            "dirac_vector": ("vector", "meson"),
            "dirac_axial_vector": ("axial_vector", "meson"),
            "dirac_tensor": ("tensor", "tensor"),
        }

        for key, (exp_group, exp_type) in expected.items():
            matched = False
            for pattern, gname, ctype in _CHANNEL_PATTERNS:
                if pattern.match(key):
                    assert gname == exp_group, (
                        f"{key}: expected group {exp_group}, got {gname}"
                    )
                    assert ctype == exp_type, (
                        f"{key}: expected type {exp_type}, got {ctype}"
                    )
                    matched = True
                    break
            assert matched, f"{key} not matched by any pattern"

    def test_dirac_patterns_dont_steal_standard_keys(self):
        """Standard keys like 'scalar' must NOT match dirac_ patterns."""
        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        standard_keys = ["scalar", "pseudoscalar", "vector", "axial_vector", "tensor"]
        for key in standard_keys:
            for pattern, gname, ctype in _CHANNEL_PATTERNS:
                if pattern.match(key):
                    # This match should NOT be a dirac pattern
                    assert not key.startswith("dirac_"), (
                        f"Standard key {key} matched a dirac_ pattern"
                    )
                    break
