"""Regression tests for multiscale_strong_force: old-vs-new parity.

Tests that the new module (fragile.physics.aic.multiscale_strong_force),
which no longer accepts an ``alive`` parameter, produces bit-for-bit identical
results to the old module (fragile.fractalai.qft.multiscale_strong_force) when
``alive`` is all-True (the invariant).
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.fractalai.qft.correlator_channels import CorrelatorConfig

# Old module (still has alive parameter)
from fragile.fractalai.qft.multiscale_strong_force import (
    _compute_channel_series_from_kernels as old_compute_series,
    _compute_companion_per_scale_results_preserving_original as old_companion_per_scale,
    _walker_bootstrap_mass_std as old_walker_bootstrap,
)

# New module (alive removed)
from fragile.physics.aic.multiscale_strong_force import (
    _compute_channel_series_from_kernels as new_compute_series,
    _compute_companion_per_scale_results_preserving_original as new_companion_per_scale,
    _walker_bootstrap_mass_std as new_walker_bootstrap,
)


# ---------------------------------------------------------------------------
# Shared test data builder
# ---------------------------------------------------------------------------


def _build_random_inputs(
    *, t_len: int, n_scales: int, n_walkers: int, seed: int = 1234
) -> dict[str, Tensor]:
    """Build deterministic random inputs for multiscale tests."""
    torch.manual_seed(seed)
    color_real = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    color_imag = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    color = torch.complex(color_real, color_imag)
    color_valid = torch.ones(t_len, n_walkers, dtype=torch.bool)
    positions = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)
    alive = torch.ones(t_len, n_walkers, dtype=torch.bool)
    force = torch.randn(t_len, n_walkers, 3, dtype=torch.float32)

    kernels = torch.rand(t_len, n_scales, n_walkers, n_walkers, dtype=torch.float32)
    eye = torch.eye(n_walkers, dtype=torch.float32).view(1, 1, n_walkers, n_walkers)
    kernels = kernels * (1.0 - eye)
    kernels = kernels / kernels.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    dx = positions[:, :, None, :] - positions[:, None, :, :]
    pairwise_distances = torch.linalg.vector_norm(dx, dim=-1)
    finite_pos = pairwise_distances[torch.isfinite(pairwise_distances) & (pairwise_distances > 0)]
    if finite_pos.numel() > 0:
        probs = torch.linspace(0.2, 0.9, n_scales, dtype=torch.float32)
        scales = torch.quantile(finite_pos, probs).clamp(min=1e-6)
    else:
        scales = torch.linspace(1e-3, 1.0, n_scales, dtype=torch.float32)

    idx = torch.arange(n_walkers, dtype=torch.long)
    companions_distance = ((idx + 1) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    companions_clone = ((idx + 2) % n_walkers).view(1, n_walkers).expand(t_len, -1).clone()
    cloning_scores = torch.randn(t_len, n_walkers, dtype=torch.float32)

    return {
        "color": color,
        "color_valid": color_valid,
        "positions": positions,
        "alive": alive,
        "force": force,
        "kernels": kernels,
        "scales": scales,
        "pairwise_distances": pairwise_distances,
        "companions_distance": companions_distance,
        "companions_clone": companions_clone,
        "cloning_scores": cloning_scores,
    }


# All channels supported by _compute_channel_series_from_kernels
BASE_CHANNELS = [
    "scalar",
    "pseudoscalar",
    "vector",
    "axial_vector",
    "nucleon",
    "glueball",
]

COMPANION_CHANNELS = [
    "scalar_companion",
    "scalar_raw_companion",
    "scalar_abs2_vacsub_companion",
    "pseudoscalar_companion",
    "scalar_score_directed_companion",
    "pseudoscalar_score_directed_companion",
    "scalar_score_weighted_companion",
    "pseudoscalar_score_weighted_companion",
    "vector_companion",
    "axial_vector_companion",
    "vector_score_directed_companion",
    "axial_vector_score_directed_companion",
    "vector_score_directed_longitudinal_companion",
    "axial_vector_score_directed_longitudinal_companion",
    "vector_score_directed_transverse_companion",
    "axial_vector_score_directed_transverse_companion",
    "vector_score_gradient_companion",
    "axial_vector_score_gradient_companion",
    "tensor_companion",
    "nucleon_companion",
    "nucleon_score_signed_companion",
    "nucleon_score_abs_companion",
    "nucleon_flux_action_companion",
    "nucleon_flux_sin2_companion",
    "nucleon_flux_exp_companion",
    "glueball_companion",
    "glueball_phase_action_companion",
    "glueball_phase_sin2_companion",
]

ALL_CHANNELS = BASE_CHANNELS + COMPANION_CHANNELS


# =============================================================================
# Layer C: Old-vs-New Parity Tests
# =============================================================================


class TestParityComputeChannelSeries:
    """old _compute_channel_series_from_kernels (with alive=all-True)
    must produce identical outputs to the new version (no alive param)."""

    @pytest.fixture
    def data(self):
        return _build_random_inputs(t_len=5, n_scales=4, n_walkers=8)

    def test_base_channels_parity(self, data):
        old_out = old_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            alive=data["alive"],
            force=data["force"],
            kernels=data["kernels"],
            channels=BASE_CHANNELS,
        )
        new_out = new_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            force=data["force"],
            kernels=data["kernels"],
            channels=BASE_CHANNELS,
        )
        for ch in BASE_CHANNELS:
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"Channel {ch}: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )

    def test_all_channels_parity(self, data):
        old_out = old_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            alive=data["alive"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            cloning_scores=data["cloning_scores"],
            channels=ALL_CHANNELS,
        )
        new_out = new_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            cloning_scores=data["cloning_scores"],
            channels=ALL_CHANNELS,
        )
        for ch in ALL_CHANNELS:
            assert ch in old_out, f"Channel {ch} missing from old output"
            assert ch in new_out, f"Channel {ch} missing from new output"
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"Channel {ch}: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )

    def test_invalid_companions_parity(self, data):
        """With invalid (-1) companions, both old and new should produce zeros."""
        bad_comp = torch.full_like(data["companions_distance"], -1)
        old_out = old_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            alive=data["alive"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=bad_comp,
            companions_clone=bad_comp,
            cloning_scores=data["cloning_scores"],
            channels=COMPANION_CHANNELS,
        )
        new_out = new_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=bad_comp,
            companions_clone=bad_comp,
            cloning_scores=data["cloning_scores"],
            channels=COMPANION_CHANNELS,
        )
        for ch in COMPANION_CHANNELS:
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"Channel {ch} with invalid companions: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )

    @pytest.mark.parametrize("seed", [42, 99, 2025])
    def test_parity_multiple_seeds(self, seed):
        """Ensure equivalence holds for different random seeds."""
        data = _build_random_inputs(t_len=4, n_scales=3, n_walkers=6, seed=seed)
        old_out = old_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            alive=data["alive"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            cloning_scores=data["cloning_scores"],
            channels=ALL_CHANNELS,
        )
        new_out = new_compute_series(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            force=data["force"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            cloning_scores=data["cloning_scores"],
            channels=ALL_CHANNELS,
        )
        for ch in ALL_CHANNELS:
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"seed={seed}, channel {ch}: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )


class TestParityCompanionPerScale:
    """old _compute_companion_per_scale_results_preserving_original (with alive)
    must produce identical correlators to the new version (no alive)."""

    # Channels tested by the companion per-scale function
    COMPANION_CORR_CHANNELS = [
        "scalar_companion",
        "scalar_raw_companion",
        "scalar_abs2_vacsub_companion",
        "pseudoscalar_companion",
        "scalar_score_directed_companion",
        "pseudoscalar_score_directed_companion",
        "vector_companion",
        "axial_vector_companion",
        "nucleon_companion",
        "nucleon_score_signed_companion",
        "nucleon_score_abs_companion",
        "glueball_companion",
        "nucleon_flux_action_companion",
        "nucleon_flux_sin2_companion",
        "nucleon_flux_exp_companion",
        "glueball_phase_action_companion",
        "glueball_phase_sin2_companion",
    ]

    @pytest.fixture
    def companion_data(self):
        t_len, n_walkers = 7, 10
        data = _build_random_inputs(t_len=t_len, n_scales=3, n_walkers=n_walkers)
        pairwise_distances = data["pairwise_distances"]
        companions_distance = data["companions_distance"]
        companions_clone = data["companions_clone"]

        d_ij = pairwise_distances.gather(
            2, companions_distance.clamp(0, n_walkers - 1).unsqueeze(-1)
        ).squeeze(-1)
        d_ik = pairwise_distances.gather(
            2, companions_clone.clamp(0, n_walkers - 1).unsqueeze(-1)
        ).squeeze(-1)
        flat_idx = companions_distance.clamp(
            0, n_walkers - 1
        ) * n_walkers + companions_clone.clamp(0, n_walkers - 1)
        d_jk = pairwise_distances.reshape(t_len, n_walkers * n_walkers).gather(1, flat_idx)

        max_radius = float(
            torch.nan_to_num(pairwise_distances, nan=0.0, posinf=0.0, neginf=0.0).max().item()
            + 1.0
        )
        scales = torch.tensor([max_radius], dtype=torch.float32)
        return {**data, "d_ij": d_ij, "d_ik": d_ik, "d_jk": d_jk, "scales_1": scales}

    def test_full_scale_correlators_parity(self, companion_data):
        d = companion_data
        cfg = CorrelatorConfig(max_lag=4, use_connected=True)

        old_out = old_companion_per_scale(
            color=d["color"],
            color_valid=d["color_valid"],
            positions=d["positions"],
            alive=d["alive"],
            cloning_scores=d["cloning_scores"],
            companions_distance=d["companions_distance"],
            companions_clone=d["companions_clone"],
            distance_ij=d["d_ij"],
            distance_ik=d["d_ik"],
            distance_jk=d["d_jk"],
            scales=d["scales_1"],
            channels=self.COMPANION_CORR_CHANNELS,
            dt=1.0,
            config=cfg,
        )
        new_out = new_companion_per_scale(
            color=d["color"],
            color_valid=d["color_valid"],
            positions=d["positions"],
            cloning_scores=d["cloning_scores"],
            companions_distance=d["companions_distance"],
            companions_clone=d["companions_clone"],
            distance_ij=d["d_ij"],
            distance_ik=d["d_ik"],
            distance_jk=d["d_jk"],
            scales=d["scales_1"],
            channels=self.COMPANION_CORR_CHANNELS,
            dt=1.0,
            config=cfg,
        )

        for ch in self.COMPANION_CORR_CHANNELS:
            assert ch in old_out, f"{ch} missing from old output"
            assert ch in new_out, f"{ch} missing from new output"
            old_corr = old_out[ch][0].correlator
            new_corr = new_out[ch][0].correlator
            assert torch.equal(old_corr, new_corr), (
                f"Channel {ch} correlator differs: max diff = "
                f"{(old_corr.float() - new_corr.float()).abs().max().item():.2e}"
            )

    def test_full_scale_series_parity(self, companion_data):
        d = companion_data
        cfg = CorrelatorConfig(max_lag=4, use_connected=True)

        old_out = old_companion_per_scale(
            color=d["color"],
            color_valid=d["color_valid"],
            positions=d["positions"],
            alive=d["alive"],
            cloning_scores=d["cloning_scores"],
            companions_distance=d["companions_distance"],
            companions_clone=d["companions_clone"],
            distance_ij=d["d_ij"],
            distance_ik=d["d_ik"],
            distance_jk=d["d_jk"],
            scales=d["scales_1"],
            channels=self.COMPANION_CORR_CHANNELS,
            dt=1.0,
            config=cfg,
        )
        new_out = new_companion_per_scale(
            color=d["color"],
            color_valid=d["color_valid"],
            positions=d["positions"],
            cloning_scores=d["cloning_scores"],
            companions_distance=d["companions_distance"],
            companions_clone=d["companions_clone"],
            distance_ij=d["d_ij"],
            distance_ik=d["d_ik"],
            distance_jk=d["d_jk"],
            scales=d["scales_1"],
            channels=self.COMPANION_CORR_CHANNELS,
            dt=1.0,
            config=cfg,
        )

        for ch in self.COMPANION_CORR_CHANNELS:
            old_series = old_out[ch][0].series
            new_series = new_out[ch][0].series
            assert torch.equal(old_series, new_series), (
                f"Channel {ch} series differs: max diff = "
                f"{(old_series.float() - new_series.float()).abs().max().item():.2e}"
            )


class TestParityWalkerBootstrap:
    """old _walker_bootstrap_mass_std (with alive) must produce identical
    mass-std tensors to the new version (no alive)."""

    def test_walker_bootstrap_parity(self):
        data = _build_random_inputs(t_len=6, n_scales=2, n_walkers=8, seed=77)
        cfg = CorrelatorConfig(max_lag=3, use_connected=True)
        channels = BASE_CHANNELS

        old_out = old_walker_bootstrap(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            alive=data["alive"],
            force=data["force"],
            cloning_scores=data["cloning_scores"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            channels=channels,
            dt=1.0,
            config=cfg,
            n_bootstrap=5,
            seed=42,
        )
        new_out = new_walker_bootstrap(
            color=data["color"],
            color_valid=data["color_valid"],
            positions=data["positions"],
            force=data["force"],
            cloning_scores=data["cloning_scores"],
            companions_distance=data["companions_distance"],
            companions_clone=data["companions_clone"],
            kernels=data["kernels"],
            scales=data["scales"],
            pairwise_distances=data["pairwise_distances"],
            channels=channels,
            dt=1.0,
            config=cfg,
            n_bootstrap=5,
            seed=42,
        )

        for ch in channels:
            assert ch in old_out, f"{ch} missing from old output"
            assert ch in new_out, f"{ch} missing from new output"
            old_val = old_out[ch]
            new_val = new_out[ch]
            assert (
                old_val.shape == new_val.shape
            ), f"Channel {ch}: shape mismatch {old_val.shape} vs {new_val.shape}"
            # NaN positions must match
            assert torch.equal(
                torch.isnan(old_val), torch.isnan(new_val)
            ), f"Channel {ch}: NaN pattern differs"
            # Finite values must be identical
            finite = torch.isfinite(old_val) & torch.isfinite(new_val)
            if finite.any():
                assert torch.equal(old_val[finite], new_val[finite]), (
                    f"Channel {ch}: max diff = "
                    f"{(old_val[finite] - new_val[finite]).abs().max().item():.2e}"
                )
