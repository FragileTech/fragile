"""AIC parity tests for pure internal radial channel functions.

Compares ``fragile.fractalai.qft.radial_channels`` (old) against
``fragile.physics.aic.radial_channels`` (new) for deterministic,
RunHistory-free helpers.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.radial_channels import (
    _apply_projection,
    _build_gamma_matrices,
    _compute_radial_correlator,
)
from fragile.physics.aic.radial_channels import (
    _apply_projection as new_projection,
    _build_gamma_matrices as new_gamma,
    _compute_radial_correlator as new_radial_correlator,
)
from tests.physics.aic.conftest import assert_tensor_or_nan_equal


# ---------------------------------------------------------------------------
# TestParityGammaMatrices
# ---------------------------------------------------------------------------


class TestParityGammaMatrices:
    """Parity: _build_gamma_matrices across old and new paths."""

    def test_dim3_parity(self) -> None:
        device = torch.device("cpu")
        dtype = torch.complex64
        old = _build_gamma_matrices(dim=3, device=device, dtype=dtype)
        new = new_gamma(dim=3, device=device, dtype=dtype)

        assert set(old.keys()) == set(
            new.keys()
        ), f"Key mismatch: {set(old.keys()) ^ set(new.keys())}"
        for key in old:
            assert torch.equal(old[key], new[key]), f"gamma[{key!r}] differs for dim=3"

    def test_dim4_parity(self) -> None:
        device = torch.device("cpu")
        dtype = torch.complex64
        old = _build_gamma_matrices(dim=4, device=device, dtype=dtype)
        new = new_gamma(dim=4, device=device, dtype=dtype)

        assert set(old.keys()) == set(
            new.keys()
        ), f"Key mismatch: {set(old.keys()) ^ set(new.keys())}"
        for key in old:
            assert torch.equal(old[key], new[key]), f"gamma[{key!r}] differs for dim=4"


# ---------------------------------------------------------------------------
# TestParityApplyProjection
# ---------------------------------------------------------------------------


CHANNELS = ["scalar", "pseudoscalar", "vector", "axial_vector", "tensor"]


class TestParityApplyProjection:
    """Parity: _apply_projection across old and new paths."""

    @pytest.mark.parametrize("channel", CHANNELS)
    def test_projection_parity(self, channel: str) -> None:
        dim = 3
        device = torch.device("cpu")

        gamma_old = _build_gamma_matrices(dim=dim, device=device, dtype=torch.complex128)
        gamma_new = new_gamma(dim=dim, device=device, dtype=torch.complex128)

        rng = torch.Generator(device=device)
        rng.manual_seed(42)

        # Synthetic complex color states: shape [5, 3, 3]
        real_i = torch.randn(5, 3, dim, generator=rng, dtype=torch.float64, device=device)
        imag_i = torch.randn(5, 3, dim, generator=rng, dtype=torch.float64, device=device)
        real_j = torch.randn(5, 3, dim, generator=rng, dtype=torch.float64, device=device)
        imag_j = torch.randn(5, 3, dim, generator=rng, dtype=torch.float64, device=device)
        color_i = torch.complex(real_i, imag_i)
        color_j = torch.complex(real_j, imag_j)

        old_result = _apply_projection(channel, color_i, color_j, gamma_old)
        new_result = new_projection(channel, color_i, color_j, gamma_new)

        assert_tensor_or_nan_equal(old_result, new_result, label=f"projection({channel})")


# ---------------------------------------------------------------------------
# TestParityRadialCorrelator
# ---------------------------------------------------------------------------


class TestParityRadialCorrelator:
    """Parity: _compute_radial_correlator across old and new paths."""

    def test_basic_correlator_parity(self) -> None:
        rng = np.random.default_rng(42)
        n_particles = 50
        operators = rng.standard_normal(n_particles)

        # Build all-pairs indices
        pair_i_list = []
        pair_j_list = []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                pair_i_list.append(i)
                pair_j_list.append(j)
        pair_i = np.array(pair_i_list, dtype=np.int64)
        pair_j = np.array(pair_j_list, dtype=np.int64)

        distances = rng.uniform(0.1, 5.0, size=len(pair_i))
        bin_edges = np.linspace(0.0, 5.0, 21)

        old_corr, old_counts, old_sum_w = _compute_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
        )
        new_corr, new_counts, new_sum_w = new_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
        )

        np.testing.assert_array_equal(old_corr, new_corr)
        np.testing.assert_array_equal(old_counts, new_counts)
        np.testing.assert_array_equal(old_sum_w, new_sum_w)

    def test_weighted_correlator_parity(self) -> None:
        rng = np.random.default_rng(99)
        n_particles = 30
        operators = rng.standard_normal(n_particles)

        pair_i_list = []
        pair_j_list = []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                pair_i_list.append(i)
                pair_j_list.append(j)
        pair_i = np.array(pair_i_list, dtype=np.int64)
        pair_j = np.array(pair_j_list, dtype=np.int64)

        distances = rng.uniform(0.0, 3.0, size=len(pair_i))
        weights = rng.uniform(0.5, 2.0, size=len(pair_i))
        bin_edges = np.linspace(0.0, 3.0, 16)

        old_corr, old_counts, old_sum_w = _compute_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
            weights=weights,
        )
        new_corr, new_counts, new_sum_w = new_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
            weights=weights,
        )

        np.testing.assert_array_equal(old_corr, new_corr)
        np.testing.assert_array_equal(old_counts, new_counts)
        np.testing.assert_array_equal(old_sum_w, new_sum_w)

    def test_empty_pairs_parity(self) -> None:
        operators = np.array([1.0, 2.0, 3.0])
        pair_i = np.array([], dtype=np.int64)
        pair_j = np.array([], dtype=np.int64)
        distances = np.array([], dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, 6)

        old_corr, old_counts, old_sum_w = _compute_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
        )
        new_corr, new_counts, new_sum_w = new_radial_correlator(
            operators,
            pair_i,
            pair_j,
            distances,
            bin_edges,
        )

        np.testing.assert_array_equal(old_corr, new_corr)
        np.testing.assert_array_equal(old_counts, new_counts)
        np.testing.assert_array_equal(old_sum_w, new_sum_w)
