"""Tests for Dirac baryon operators: diquark algebra, parity projection, baryon channels.

Verifies:
A. Diquark algebra: shapes, antisymmetry, multi-component Gamma
B. Parity projection: completeness, upper/lower selection, idempotence
C. DiracBaryonSeries shape validation
D. Physical properties: nucleon_det correlation, channel differences, parity structure
E. Edge cases: d<3, k<2, single-walker, all-dead
F. Integration: compute_dirac_baryons_from_agg wrapper
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from fragile.physics.new_channels.dirac_baryons import (
    compute_diquark,
    compute_dirac_baryon_operators,
    compute_dirac_baryons_from_agg,
    DiracBaryonSeries,
    parity_projection,
)
from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def gamma():
    return build_dirac_gamma_matrices()


@pytest.fixture
def random_spinors():
    """Random Dirac spinors [T, S, 4] for testing."""
    gen = torch.Generator().manual_seed(42)
    T, S = 8, 16
    return torch.complex(
        torch.randn(T, S, 4, generator=gen).double(),
        torch.randn(T, S, 4, generator=gen).double(),
    )


@pytest.fixture
def baryon_inputs():
    """Standard test inputs for compute_dirac_baryon_operators."""
    gen = torch.Generator().manual_seed(42)
    T, N = 10, 20
    color = torch.complex(
        torch.randn(T, N, 3, generator=gen),
        torch.randn(T, N, 3, generator=gen),
    )
    color_valid = torch.ones(T, N, dtype=torch.bool)
    alive = torch.ones(T, N, dtype=torch.bool)
    sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
    # 2 neighbors: cyclic shift-1 and shift-2
    n1 = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
    n2 = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1)
    neighbor_indices = torch.stack([n1, n2], dim=-1)  # [T, N, 2]
    return {
        "color": color,
        "color_valid": color_valid,
        "sample_indices": sample_indices,
        "neighbor_indices": neighbor_indices,
        "alive": alive,
    }


# ===========================================================================
# A. Diquark algebra
# ===========================================================================


class TestDiquarkAlgebra:
    """Tests for compute_diquark."""

    def test_output_shape_single_gamma(self, gamma, random_spinors):
        """compute_diquark output shape matches input batch shape for [4,4] Gamma."""
        psi_a = random_spinors
        psi_b = random_spinors.roll(1, dims=1)
        result = compute_diquark(psi_a, psi_b, gamma["C"], gamma["gamma5"])
        assert result.shape == random_spinors.shape[:-1]  # [T, S]

    def test_antisymmetry_with_identity_gamma(self, gamma, random_spinors):
        """With Gamma=I₄: (ψ_a)^T C ψ_b is antisymmetric under a↔b (C is antisymmetric)."""
        C = gamma["C"]
        I4 = torch.eye(4, dtype=C.dtype)
        psi_a = random_spinors
        psi_b = random_spinors.roll(1, dims=1)

        dq_ab = compute_diquark(psi_a, psi_b, C, I4)
        dq_ba = compute_diquark(psi_b, psi_a, C, I4)

        # C^T = -C for the standard charge conjugation matrix
        # So (ψ_b)^T C ψ_a = ψ_b^T C ψ_a = -(ψ_a)^T C ψ_b (up to transpose)
        # Actually: (ψ_b)^T C I₄ ψ_a = Σ_{st} (ψ_b)_s C_{st} (ψ_a)_t
        #         = Σ_{st} (ψ_a)_t C_{st} (ψ_b)_s
        #         = Σ_{ts} (ψ_a)_s C_{ts} (ψ_b)_t   (relabel s↔t)
        #         = Σ_{st} (ψ_a)_s C^T_{st} (ψ_b)_t
        #         = (ψ_a)^T C^T ψ_b
        # If C^T = -C, then dq_ba = -dq_ab
        assert torch.allclose(dq_ba, -dq_ab, atol=1e-8), (
            f"Diquark not antisymmetric: max diff = {(dq_ba + dq_ab).abs().max().item()}"
        )

    def test_multi_component_gamma_shape(self, gamma, random_spinors):
        """With Gamma [n, 4, 4]: output has shape [..., n]."""
        psi_a = random_spinors
        psi_b = random_spinors.roll(1, dims=1)
        # gamma_k is [3, 4, 4]
        result = compute_diquark(psi_a, psi_b, gamma["C"], gamma["gamma_k"])
        expected_shape = (*random_spinors.shape[:-1], 3)
        assert result.shape == expected_shape

    def test_diquark_is_finite(self, gamma, random_spinors):
        psi_a = random_spinors
        psi_b = random_spinors.roll(1, dims=1)
        for key in ["gamma5", "gamma_k", "gamma5_k"]:
            result = compute_diquark(psi_a, psi_b, gamma["C"], gamma[key])
            assert torch.isfinite(result).all(), f"Non-finite diquark for Gamma={key}"


# ===========================================================================
# B. Parity projection
# ===========================================================================


class TestParityProjection:
    """Tests for parity_projection."""

    def test_completeness(self, gamma, random_spinors):
        """P₊ψ + P₋ψ recovers original spinor (component sum)."""
        gamma0 = gamma["gamma0"]
        psi = random_spinors  # [T, S, 4]

        proj_plus = parity_projection(psi, gamma0, positive=True)
        proj_minus = parity_projection(psi, gamma0, positive=False)

        # Sum of projected component sums = sum of all original components
        original_sum = psi.sum(dim=-1)
        assert torch.allclose(proj_plus + proj_minus, original_sum, atol=1e-10), (
            "P₊ + P₋ does not recover original spinor sum"
        )

    def test_positive_projects_upper(self, gamma):
        """P₊ with γ₀ = diag(1,1,-1,-1) selects upper 2 components."""
        gamma0 = gamma["gamma0"]
        # Spinor with known components
        psi = torch.tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j], dtype=torch.complex128)
        proj = parity_projection(psi, gamma0, positive=True)
        # P₊ = diag(1,1,0,0), so projected = (1, 2, 0, 0), sum = 3
        assert torch.allclose(proj, torch.tensor(3.0 + 0j, dtype=torch.complex128), atol=1e-10)

    def test_negative_projects_lower(self, gamma):
        """P₋ with γ₀ = diag(1,1,-1,-1) selects lower 2 components."""
        gamma0 = gamma["gamma0"]
        psi = torch.tensor([1.0 + 0j, 2.0 + 0j, 3.0 + 0j, 4.0 + 0j], dtype=torch.complex128)
        proj = parity_projection(psi, gamma0, positive=False)
        # P₋ = diag(0,0,1,1), so projected = (0, 0, 3, 4), sum = 7
        assert torch.allclose(proj, torch.tensor(7.0 + 0j, dtype=torch.complex128), atol=1e-10)

    def test_idempotent_positive(self, gamma, random_spinors):
        """P₊² = P₊ (checked via double projection giving same result)."""
        gamma0 = gamma["gamma0"]
        I4 = torch.eye(4, device=gamma0.device, dtype=gamma0.dtype)
        P_plus = 0.5 * (I4 + gamma0)

        psi = random_spinors
        # First projection
        proj1 = torch.einsum("st,...t->...s", P_plus, psi)
        # Second projection on result
        proj2 = torch.einsum("st,...t->...s", P_plus, proj1)
        assert torch.allclose(proj1, proj2, atol=1e-10), "P₊ is not idempotent"

    def test_idempotent_negative(self, gamma, random_spinors):
        """P₋² = P₋ (checked via double projection giving same result)."""
        gamma0 = gamma["gamma0"]
        I4 = torch.eye(4, device=gamma0.device, dtype=gamma0.dtype)
        P_minus = 0.5 * (I4 - gamma0)

        psi = random_spinors
        proj1 = torch.einsum("st,...t->...s", P_minus, psi)
        proj2 = torch.einsum("st,...t->...s", P_minus, proj1)
        assert torch.allclose(proj1, proj2, atol=1e-10), "P₋ is not idempotent"


# ===========================================================================
# C. DiracBaryonSeries shape validation
# ===========================================================================


class TestDiracBaryonSeriesShapes:
    """Verify all output fields have correct shapes."""

    def test_all_operator_fields_are_T(self, baryon_inputs):
        T = baryon_inputs["color"].shape[0]
        result = compute_dirac_baryon_operators(**baryon_inputs)
        assert isinstance(result, DiracBaryonSeries)

        for field_name in ["nucleon", "delta", "n_star_scalar", "n_star_axial", "nucleon_det"]:
            field = getattr(result, field_name)
            assert field.shape == (T,), f"{field_name}: expected ({T},), got {field.shape}"

    def test_diagnostic_fields_are_T(self, baryon_inputs):
        T = baryon_inputs["color"].shape[0]
        result = compute_dirac_baryon_operators(**baryon_inputs)
        assert result.n_valid_triplets.shape == (T,)
        assert result.spinor_valid_fraction.shape == (T,)

    def test_all_values_finite(self, baryon_inputs):
        result = compute_dirac_baryon_operators(**baryon_inputs)
        for field_name in [
            "nucleon", "delta", "n_star_scalar", "n_star_axial", "nucleon_det",
            "n_valid_triplets", "spinor_valid_fraction",
        ]:
            field = getattr(result, field_name)
            assert torch.isfinite(field).all(), f"{field_name} has non-finite values"


# ===========================================================================
# D. Physical properties
# ===========================================================================


class TestPhysicalProperties:
    """Tests for physical correctness of baryon channels."""

    def test_nucleon_det_and_nucleon_both_nonzero(self, baryon_inputs):
        """nucleon_det and nucleon should both produce nonzero output."""
        result = compute_dirac_baryon_operators(**baryon_inputs)
        # Both channels should produce nonzero operators (they share the color det)
        assert result.nucleon_det.abs().max() > 1e-10, "nucleon_det is all-zero"
        assert result.nucleon.abs().max() > 1e-10, "nucleon is all-zero"

    def test_nucleon_and_delta_differ(self, baryon_inputs):
        """nucleon and delta should have different magnitudes (different diquark)."""
        result = compute_dirac_baryon_operators(**baryon_inputs)
        assert not torch.allclose(result.nucleon, result.delta, atol=1e-6), (
            "nucleon and delta should differ"
        )

    def test_positive_and_negative_parity_differ(self, baryon_inputs):
        """Positive-parity channels differ from negative-parity channels."""
        result = compute_dirac_baryon_operators(**baryon_inputs)
        # nucleon (P₊) vs n_star_scalar (P₋)
        assert not torch.allclose(result.nucleon, result.n_star_scalar, atol=1e-6), (
            "nucleon and n_star_scalar should differ (different parity)"
        )

    def test_all_dead_gives_zero(self):
        """With all-dead walkers, operators should be zero."""
        gen = torch.Generator().manual_seed(42)
        T, N = 5, 10
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.zeros(T, N, dtype=torch.bool)  # ALL dead
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        n1 = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1)
        n2 = torch.arange(N).roll(-2).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.stack([n1, n2], dim=-1)

        result = compute_dirac_baryon_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
        )
        for field_name in ["nucleon", "delta", "n_star_scalar", "n_star_axial", "nucleon_det"]:
            field = getattr(result, field_name)
            assert (field.abs() < 1e-10).all(), f"{field_name} not zero with all-dead walkers"

    def test_n_valid_triplets_bounded(self, baryon_inputs):
        """n_valid_triplets <= S (sample size)."""
        S = baryon_inputs["sample_indices"].shape[1]
        result = compute_dirac_baryon_operators(**baryon_inputs)
        assert (result.n_valid_triplets <= S).all()

    def test_spinor_valid_fraction_in_range(self, baryon_inputs):
        result = compute_dirac_baryon_operators(**baryon_inputs)
        assert (result.spinor_valid_fraction >= 0).all()
        assert (result.spinor_valid_fraction <= 1).all()


# ===========================================================================
# E. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_d_less_than_3_raises(self):
        """d < 3 should raise ValueError."""
        T, N = 5, 10
        color = torch.randn(T, N, 2, dtype=torch.cfloat)
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.zeros(T, N, 2, dtype=torch.long)

        with pytest.raises(ValueError, match="d >= 3"):
            compute_dirac_baryon_operators(
                color=color,
                color_valid=color_valid,
                sample_indices=sample_indices,
                neighbor_indices=neighbor_indices,
                alive=alive,
            )

    def test_k_less_than_2_raises(self):
        """k < 2 neighbors should raise ValueError."""
        T, N = 5, 10
        color = torch.randn(T, N, 3, dtype=torch.cfloat)
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.zeros(T, N, 1, dtype=torch.long)  # only 1 neighbor

        with pytest.raises(ValueError, match="2 neighbors"):
            compute_dirac_baryon_operators(
                color=color,
                color_valid=color_valid,
                sample_indices=sample_indices,
                neighbor_indices=neighbor_indices,
                alive=alive,
            )

    def test_single_walker_all_same_indices(self):
        """Single walker with all indices pointing to itself → valid_mask False → zero."""
        T, N = 5, 1
        gen = torch.Generator().manual_seed(42)
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        sample_indices = torch.zeros(T, N, dtype=torch.long)  # all 0
        neighbor_indices = torch.zeros(T, N, 2, dtype=torch.long)  # all 0

        result = compute_dirac_baryon_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
        )
        # All indices are the same → distinct check fails → zero output
        assert result.n_valid_triplets.sum() == 0
        for field_name in ["nucleon", "delta", "n_star_scalar", "n_star_axial", "nucleon_det"]:
            field = getattr(result, field_name)
            assert (field.abs() < 1e-10).all(), f"{field_name} not zero for single walker"


# ===========================================================================
# F. Integration: compute_dirac_baryons_from_agg
# ===========================================================================


class TestComputeDiracBaryonsFromAgg:
    """Verify the convenience wrapper calls through correctly."""

    def test_calls_compute_dirac_baryon_operators(self, baryon_inputs):
        """compute_dirac_baryons_from_agg delegates to compute_dirac_baryon_operators."""
        mock_agg = MagicMock()
        mock_agg.color = baryon_inputs["color"]
        mock_agg.color_valid = baryon_inputs["color_valid"]
        mock_agg.sample_indices = baryon_inputs["sample_indices"]
        mock_agg.neighbor_indices = baryon_inputs["neighbor_indices"]
        mock_agg.alive = baryon_inputs["alive"]
        mock_agg.sample_edge_weights = None

        result = compute_dirac_baryons_from_agg(mock_agg)
        assert isinstance(result, DiracBaryonSeries)
        T = baryon_inputs["color"].shape[0]
        assert result.nucleon.shape == (T,)


# ===========================================================================
# Mass extraction pattern matching
# ===========================================================================


class TestBaryonMassExtractionPatterns:
    """Verify that baryon_*_dirac keys are matched to correct channel groups."""

    def test_baryon_dirac_channels_map_correctly(self):
        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        expected = {
            "baryon_nucleon_dirac": ("nucleon", "baryon"),
            "baryon_delta_dirac": ("delta", "baryon"),
            "baryon_nstar_scalar_dirac": ("n_star_scalar", "baryon"),
            "baryon_nstar_axial_dirac": ("n_star_axial", "baryon"),
            "baryon_nucleon_det_dirac": ("nucleon", "baryon"),
            "baryon_delta_flux": ("delta", "baryon"),
            "baryon_nstar_something": ("n_star", "baryon"),
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

    def test_existing_baryon_nucleon_still_works(self):
        """baryon_nucleon_det_abs should still match nucleon/baryon."""
        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        key = "baryon_nucleon_det_abs"
        for pattern, gname, ctype in _CHANNEL_PATTERNS:
            if pattern.match(key):
                assert gname == "nucleon"
                assert ctype == "baryon"
                break
