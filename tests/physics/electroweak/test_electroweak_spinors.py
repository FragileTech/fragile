"""Tests for electroweak spinor module.

Verifies:
A. Chiral projector algebra (P_L + P_R = I, idempotent, orthogonal, Hermitian)
B. Gauge link properties (unit-modulus, equal fitness → 1, conjugate symmetry)
C. Chiral bilinear correctness (matches strong-force bilinear, real-valued, gauge scaling)
D. ElectroweakSpinorOutput shape validation (all fields [T], finite)
E. Physical properties (chiral decomposition, parity violation bounded, pair counts)
F. Edge cases (d != 3 raises, all-dead → zero, uniform chirality)
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import Tensor

from fragile.physics.electroweak.electroweak_spinors import (
    ElectroweakSpinorOutput,
    _compute_chiral_bilinear,
    build_chiral_projectors,
    compute_electroweak_spinor_operators,
    compute_su2_gauge_link,
    compute_u1_gauge_link,
)
from fragile.physics.new_channels.dirac_spinors import (
    build_dirac_gamma_matrices,
    color_to_dirac_spinor,
    compute_dirac_bilinear,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def gamma():
    return build_dirac_gamma_matrices()


@pytest.fixture
def projectors(gamma):
    return build_chiral_projectors(gamma["gamma5"])


@pytest.fixture
def ew_input():
    """Standard input for compute_electroweak_spinor_operators."""
    gen = torch.Generator().manual_seed(42)
    T, N = 10, 20
    color = torch.complex(
        torch.randn(T, N, 3, generator=gen),
        torch.randn(T, N, 3, generator=gen),
    )
    color_valid = torch.ones(T, N, dtype=torch.bool)
    alive = torch.ones(T, N, dtype=torch.bool)
    fitness = torch.rand(T, N, generator=gen).clamp(min=1e-6)
    # Walker chirality: random +1/-1
    chi_raw = torch.randint(0, 2, (T, N), generator=gen)
    walker_chi = torch.where(chi_raw == 1, 1.0, -1.0)
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
        "fitness": fitness,
        "walker_chi": walker_chi,
    }


# ===========================================================================
# A. Chiral projector algebra
# ===========================================================================


class TestChiralProjectors:
    """Verify P_L and P_R satisfy projector algebra."""

    def test_sum_is_identity(self, projectors):
        P_L, P_R = projectors
        I4 = torch.eye(4, dtype=P_L.dtype)
        assert (P_L + P_R - I4).abs().max() < 1e-10

    def test_PL_idempotent(self, projectors):
        P_L, _ = projectors
        assert (P_L @ P_L - P_L).abs().max() < 1e-10

    def test_PR_idempotent(self, projectors):
        _, P_R = projectors
        assert (P_R @ P_R - P_R).abs().max() < 1e-10

    def test_orthogonal(self, projectors):
        P_L, P_R = projectors
        assert (P_L @ P_R).abs().max() < 1e-10
        assert (P_R @ P_L).abs().max() < 1e-10

    def test_PL_hermitian(self, projectors):
        P_L, _ = projectors
        assert (P_L - P_L.conj().T).abs().max() < 1e-10

    def test_PR_hermitian(self, projectors):
        _, P_R = projectors
        assert (P_R - P_R.conj().T).abs().max() < 1e-10

    def test_shapes(self, projectors):
        P_L, P_R = projectors
        assert P_L.shape == (4, 4)
        assert P_R.shape == (4, 4)


# ===========================================================================
# B. Gauge link properties
# ===========================================================================


class TestU1GaugeLink:
    """Test U(1) hypercharge gauge link."""

    def test_unit_modulus(self):
        fit_i = torch.tensor([1.0, 2.0, 3.0])
        fit_j = torch.tensor([1.5, 2.5, 3.5])
        u = compute_u1_gauge_link(fit_i, fit_j, h_eff=1.0)
        assert torch.allclose(u.abs(), torch.ones_like(u.abs()), atol=1e-10)

    def test_equal_fitness_gives_one(self):
        fit = torch.tensor([1.0, 2.0, 3.0])
        u = compute_u1_gauge_link(fit, fit, h_eff=1.0)
        assert torch.allclose(u.real, torch.ones_like(u.real), atol=1e-10)
        assert torch.allclose(u.imag, torch.zeros_like(u.imag), atol=1e-10)

    def test_conjugate_symmetry(self):
        """U(i,j) = U(j,i)*."""
        fit_i = torch.tensor([1.0, 2.5, 0.3])
        fit_j = torch.tensor([1.5, 3.0, 0.8])
        u_ij = compute_u1_gauge_link(fit_i, fit_j, h_eff=0.5)
        u_ji = compute_u1_gauge_link(fit_j, fit_i, h_eff=0.5)
        assert torch.allclose(u_ij, u_ji.conj(), atol=1e-10)

    def test_batch_shape(self):
        fit_i = torch.randn(5, 10)
        fit_j = torch.randn(5, 10)
        u = compute_u1_gauge_link(fit_i, fit_j, h_eff=1.0)
        assert u.shape == (5, 10)


class TestSU2GaugeLink:
    """Test SU(2) isospin gauge link."""

    def test_unit_modulus(self):
        fit_i = torch.tensor([1.0, 2.0, 3.0])
        fit_j = torch.tensor([1.5, 2.5, 3.5])
        u = compute_su2_gauge_link(fit_i, fit_j, h_eff=1.0)
        assert torch.allclose(u.abs(), torch.ones_like(u.abs()), atol=1e-10)

    def test_equal_fitness_gives_one(self):
        fit = torch.tensor([1.0, 2.0, 3.0])
        u = compute_su2_gauge_link(fit, fit, h_eff=1.0)
        # With equal fitness, delta_s = 0, so phase = 0, U = 1
        assert torch.allclose(u.real, torch.ones_like(u.real), atol=1e-10)
        assert torch.allclose(u.imag, torch.zeros_like(u.imag), atol=1e-10)

    def test_symmetric_in_ij(self):
        """SU(2) link uses |Delta S|, so U(i,j) = U(j,i)."""
        fit_i = torch.tensor([1.0, 2.5])
        fit_j = torch.tensor([1.5, 3.0])
        u_ij = compute_su2_gauge_link(fit_i, fit_j, h_eff=1.0)
        u_ji = compute_su2_gauge_link(fit_j, fit_i, h_eff=1.0)
        assert torch.allclose(u_ij, u_ji, atol=1e-10)

    def test_batch_shape(self):
        fit_i = torch.randn(5, 10)
        fit_j = torch.randn(5, 10)
        u = compute_su2_gauge_link(fit_i, fit_j, h_eff=1.0)
        assert u.shape == (5, 10)


# ===========================================================================
# C. Chiral bilinear correctness
# ===========================================================================


class TestChiralBilinear:
    """Test _compute_chiral_bilinear."""

    @pytest.fixture(autouse=True)
    def setup(self, gamma):
        gen = torch.Generator().manual_seed(42)
        self.gamma = gamma
        self.psi_i = torch.complex(
            torch.randn(4, 8, 4, generator=gen).double(),
            torch.randn(4, 8, 4, generator=gen).double(),
        )
        self.psi_j = torch.complex(
            torch.randn(4, 8, 4, generator=gen).double(),
            torch.randn(4, 8, 4, generator=gen).double(),
        )

    def test_no_projector_no_gauge_matches_dirac_bilinear(self):
        """With P=None, gauge=None, should match compute_dirac_bilinear."""
        I4 = torch.eye(4, dtype=torch.complex128)
        gamma0 = self.gamma["gamma0"]

        chiral_result = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4,
            P_chirality=None, gauge_link=None,
        )
        dirac_result = compute_dirac_bilinear(self.psi_i, self.psi_j, gamma0, I4)

        assert torch.allclose(chiral_result, dirac_result.float(), atol=1e-5)

    def test_output_is_real(self):
        gamma0 = self.gamma["gamma0"]
        I4 = torch.eye(4, dtype=torch.complex128)
        result = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4,
        )
        assert not result.is_complex()

    def test_gauge_link_scales_bilinear(self):
        """With a pure-real gauge link, bilinear should scale."""
        gamma0 = self.gamma["gamma0"]
        I4 = torch.eye(4, dtype=torch.complex128)

        result_no_gauge = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4,
        )
        # Use a real gauge link of magnitude 1 (phase = 0)
        gauge_one = torch.ones(4, 8, dtype=torch.complex128)
        result_with_gauge = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4, gauge_link=gauge_one,
        )
        assert torch.allclose(result_no_gauge, result_with_gauge, atol=1e-5)

    def test_chiral_projection_reduces_magnitude(self):
        """With chiral projector, bilinear should generally differ from full."""
        gamma0 = self.gamma["gamma0"]
        I4 = torch.eye(4, dtype=torch.complex128)
        P_L, _ = build_chiral_projectors(self.gamma["gamma5"])

        result_full = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4,
        )
        result_L = _compute_chiral_bilinear(
            self.psi_i, self.psi_j, gamma0, I4, P_chirality=P_L,
        )
        # They should be different for generic input
        assert not torch.allclose(result_full, result_L, atol=1e-4)


# ===========================================================================
# D. ElectroweakSpinorOutput shape validation
# ===========================================================================


class TestElectroweakSpinorOutputShapes:
    """Verify all output fields have correct shapes and are finite."""

    def test_returns_dataclass(self, ew_input):
        result = compute_electroweak_spinor_operators(**ew_input)
        assert isinstance(result, ElectroweakSpinorOutput)

    def test_all_operator_fields_are_T_tensors(self, ew_input):
        T = ew_input["color"].shape[0]
        result = compute_electroweak_spinor_operators(**ew_input)

        operator_fields = [
            "j_vector_L", "j_vector_R", "j_vector_V",
            "o_scalar_L", "o_scalar_R",
            "j_vector_walkerL", "j_vector_walkerR",
            "j_vector_L_walkerL", "j_vector_R_walkerR",
            "o_yukawa_LR", "o_yukawa_RL",
            "j_vector_u1", "j_vector_L_u1",
            "j_vector_L_su2", "j_vector_R_su2",
        ]
        for name in operator_fields:
            val = getattr(result, name)
            assert val.shape == (T,), f"{name}: expected ({T},), got {val.shape}"

    def test_all_diagnostic_fields_are_T_tensors(self, ew_input):
        T = ew_input["color"].shape[0]
        result = compute_electroweak_spinor_operators(**ew_input)

        diag_fields = [
            "n_valid_pairs", "n_valid_pairs_LL", "n_valid_pairs_RR",
            "n_valid_pairs_LR",
            "parity_violation_dirac", "parity_violation_walker",
        ]
        for name in diag_fields:
            val = getattr(result, name)
            assert val.shape == (T,), f"{name}: expected ({T},), got {val.shape}"

    def test_all_values_finite(self, ew_input):
        result = compute_electroweak_spinor_operators(**ew_input)

        for f in dataclasses.fields(result):
            val = getattr(result, f.name)
            if isinstance(val, Tensor):
                assert torch.isfinite(val).all(), f"{f.name} has non-finite values"


# ===========================================================================
# E. Physical properties
# ===========================================================================


class TestPhysicalProperties:
    """Test physics constraints on the output."""

    def test_vector_chiral_decomposition(self, ew_input):
        """j_vector_V should approximately equal j_vector_L + j_vector_R.

        This holds when the projectors are applied to the same pairs with same weights.
        """
        result = compute_electroweak_spinor_operators(**ew_input)
        # V = L + R is exact when P_L + P_R = I (which it is)
        assert torch.allclose(
            result.j_vector_V,
            result.j_vector_L + result.j_vector_R,
            atol=1e-4,
        ), (
            f"V-A decomposition failed: max diff = "
            f"{(result.j_vector_V - result.j_vector_L - result.j_vector_R).abs().max()}"
        )

    def test_parity_violation_bounded(self, ew_input):
        """Parity violation diagnostic should satisfy |pv| <= 1."""
        result = compute_electroweak_spinor_operators(**ew_input)
        assert (result.parity_violation_dirac.abs() <= 1.0 + 1e-6).all()
        assert (result.parity_violation_walker.abs() <= 1.0 + 1e-6).all()

    def test_pair_counts_consistent(self, ew_input):
        """n_valid_pairs >= n_valid_pairs_LL + n_valid_pairs_RR + n_valid_pairs_LR."""
        result = compute_electroweak_spinor_operators(**ew_input)
        total = result.n_valid_pairs
        sub = result.n_valid_pairs_LL + result.n_valid_pairs_RR + result.n_valid_pairs_LR
        # The total also includes RL pairs not counted individually, so total >= sub
        assert (total >= sub - 1).all(), (
            f"Pair counts inconsistent: total = {total}, sub = {sub}"
        )

    def test_valid_pairs_positive(self, ew_input):
        result = compute_electroweak_spinor_operators(**ew_input)
        assert (result.n_valid_pairs > 0).all()

    def test_uniform_L_chirality_zeroes_walkerR(self):
        """With all walkers L-handed, j_vector_walkerR should be ~0."""
        gen = torch.Generator().manual_seed(77)
        T, N = 8, 15
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        fitness = torch.rand(T, N, generator=gen).clamp(min=1e-6)
        walker_chi = torch.ones(T, N)  # All left-handed
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)

        result = compute_electroweak_spinor_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
            fitness=fitness,
            walker_chi=walker_chi,
        )
        # With all walkers L, there are no R walkers → walkerR should be ~0
        # (the averaging denominator is clamped, so it's 0/eps ≈ 0)
        assert result.j_vector_walkerR.abs().max() < 1e-6, (
            f"j_vector_walkerR should be ~0 with uniform L chirality, "
            f"got max = {result.j_vector_walkerR.abs().max()}"
        )

    def test_uniform_L_dirac_projection_matches(self):
        """With uniform L chirality, j_vector_L_walkerL should match j_vector_L."""
        gen = torch.Generator().manual_seed(88)
        T, N = 8, 15
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        fitness = torch.rand(T, N, generator=gen).clamp(min=1e-6)
        walker_chi = torch.ones(T, N)  # All left-handed
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)

        result = compute_electroweak_spinor_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
            fitness=fitness,
            walker_chi=walker_chi,
        )
        # All walkers are L, so both_L = valid, and j_vector_L_walkerL = j_vector_L
        assert torch.allclose(result.j_vector_L_walkerL, result.j_vector_L, atol=1e-4)


# ===========================================================================
# F. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_d_not_3_raises(self, ew_input):
        """d != 3 should raise ValueError."""
        ew_input["color"] = torch.randn(10, 20, 4, dtype=torch.cfloat)
        with pytest.raises(ValueError, match="d=3"):
            compute_electroweak_spinor_operators(**ew_input)

    def test_all_dead_walkers_zero_operators(self):
        """All-dead walkers should produce zero operators."""
        gen = torch.Generator().manual_seed(55)
        T, N = 5, 10
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.zeros(T, N, dtype=torch.bool)  # All dead
        fitness = torch.rand(T, N, generator=gen)
        walker_chi = torch.zeros(T, N)  # Dead = 0 chirality
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)

        result = compute_electroweak_spinor_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
            fitness=fitness,
            walker_chi=walker_chi,
        )
        # All dead means no valid pairs → operators should be zero (or near-zero)
        assert result.n_valid_pairs.sum() == 0
        assert result.j_vector_L.abs().max() < 1e-6

    def test_no_cloning_walker_L_currents_near_zero(self):
        """With no cloning (all R walkers), walker-L currents should be ~0."""
        gen = torch.Generator().manual_seed(66)
        T, N = 8, 15
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        color_valid = torch.ones(T, N, dtype=torch.bool)
        alive = torch.ones(T, N, dtype=torch.bool)
        fitness = torch.rand(T, N, generator=gen).clamp(min=1e-6)
        walker_chi = -torch.ones(T, N)  # All R (no cloning)
        sample_indices = torch.arange(N).unsqueeze(0).expand(T, -1)
        neighbor_indices = torch.arange(N).roll(-1).unsqueeze(0).expand(T, -1).unsqueeze(-1)

        result = compute_electroweak_spinor_operators(
            color=color,
            color_valid=color_valid,
            sample_indices=sample_indices,
            neighbor_indices=neighbor_indices,
            alive=alive,
            fitness=fitness,
            walker_chi=walker_chi,
        )
        # No L walkers → walker-L restricted currents should be ~0
        assert result.j_vector_walkerL.abs().max() < 1e-6
        assert result.j_vector_L_walkerL.abs().max() < 1e-6

    def test_single_frame(self):
        """Single frame should still work."""
        gen = torch.Generator().manual_seed(99)
        T, N = 1, 10
        color = torch.complex(
            torch.randn(T, N, 3, generator=gen),
            torch.randn(T, N, 3, generator=gen),
        )
        result = compute_electroweak_spinor_operators(
            color=color,
            color_valid=torch.ones(T, N, dtype=torch.bool),
            sample_indices=torch.arange(N).unsqueeze(0),
            neighbor_indices=torch.arange(N).roll(-1).unsqueeze(0).unsqueeze(-1),
            alive=torch.ones(T, N, dtype=torch.bool),
            fitness=torch.rand(T, N, generator=gen),
            walker_chi=torch.ones(T, N),
        )
        assert result.j_vector_L.shape == (1,)
        assert torch.isfinite(result.j_vector_L).all()


# ===========================================================================
# Mass extraction pattern matching
# ===========================================================================


class TestEWSpinorMassExtractionPatterns:
    """Verify EW spinor channel keys match mass extraction patterns."""

    def test_ew_spinor_channels_matched(self):
        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        test_keys = {
            "j_vector_L_su2": "ew_spinor_w",
            "j_vector_L_u1": "ew_spinor_z",
            "j_vector_u1": "ew_spinor_photon",
            "j_vector_L": "ew_spinor_left",
            "j_vector_R": "ew_spinor_right",
            "j_vector_V": "ew_spinor_vector",
            "j_vector_walkerL": "ew_spinor_walker",
            "j_vector_walkerR": "ew_spinor_walker",
            "o_scalar_L": "ew_spinor_scalar",
            "o_scalar_R": "ew_spinor_scalar",
            "o_yukawa_LR": "ew_spinor_yukawa",
            "o_yukawa_RL": "ew_spinor_yukawa",
            "parity_violation_dirac": "ew_parity_violation",
            "parity_violation_walker": "ew_parity_violation",
        }

        for key, expected_group in test_keys.items():
            matched = False
            for pattern, gname, ctype in _CHANNEL_PATTERNS:
                if pattern.match(key):
                    assert gname == expected_group, (
                        f"{key}: expected group {expected_group!r}, got {gname!r}"
                    )
                    assert ctype == "electroweak", (
                        f"{key}: expected type 'electroweak', got {ctype!r}"
                    )
                    matched = True
                    break
            assert matched, f"{key} not matched by any pattern"

    def test_ew_spinor_patterns_dont_steal_standard_ew_keys(self):
        """Standard EW keys like 'u1_phase' must still match their original patterns."""
        from fragile.physics.mass_extraction.pipeline import _CHANNEL_PATTERNS

        standard_ew = {
            "u1_phase": "u1_hypercharge",
            "su2_phase": "su2_phase",
            "su2_doublet": "su2_doublet",
            "ew_mixed": "ew_mixed",
        }

        for key, expected_group in standard_ew.items():
            for pattern, gname, _ in _CHANNEL_PATTERNS:
                if pattern.match(key):
                    assert gname == expected_group, (
                        f"Standard key {key!r} matched wrong group: "
                        f"expected {expected_group!r}, got {gname!r}"
                    )
                    break


# ===========================================================================
# Correlator plot grouping
# ===========================================================================


class TestEWSpinorCorrelatorGrouping:
    """Verify EW spinor channels are grouped correctly in plots."""

    def test_grouping(self):
        from fragile.physics.app.correlator_plots import group_electroweak_correlator_keys

        keys = [
            "u1_phase",
            "j_vector_L", "j_vector_R", "j_vector_V",
            "o_scalar_L",
            "j_vector_walkerL", "j_vector_walkerR",
            "o_yukawa_LR",
            "j_vector_u1", "j_vector_L_su2",
            "parity_violation_dirac",
        ]
        groups = group_electroweak_correlator_keys(keys)

        assert "U(1)" in groups
        assert "u1_phase" in groups["U(1)"]

        assert "EW Chiral Currents" in groups
        assert set(groups["EW Chiral Currents"]) == {"j_vector_L", "j_vector_R", "j_vector_V", "o_scalar_L"}

        assert "EW Walker Chirality" in groups
        assert set(groups["EW Walker Chirality"]) == {"j_vector_walkerL", "j_vector_walkerR"}

        assert "EW Yukawa" in groups
        assert "o_yukawa_LR" in groups["EW Yukawa"]

        assert "EW Gauge-Dressed" in groups
        assert set(groups["EW Gauge-Dressed"]) == {"j_vector_u1", "j_vector_L_su2"}

        assert "EW Parity Violation" in groups
        assert "parity_violation_dirac" in groups["EW Parity Violation"]
