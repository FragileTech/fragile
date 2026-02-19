"""AIC parity tests for multiscale_strong_force.

Both old (fragile.fractalai.qft) and new (fragile.physics.aic) modules are
verbatim copies, so they share identical signatures (including ``alive``).
These tests verify bit-for-bit regression parity.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult, CorrelatorConfig

# Old module
from fragile.fractalai.qft.multiscale_strong_force import (
    _compute_channel_series_from_kernels as old_compute_series,
    _compute_companion_per_scale_results_preserving_original as old_companion_per_scale,
    _select_best_scale as old_select_best,
    _walker_bootstrap_mass_std as old_walker_bootstrap,
)

# New module (AIC copy -- verbatim, identical signatures)
from fragile.physics.aic.multiscale_strong_force import (
    _compute_channel_series_from_kernels as new_compute_series,
    _compute_companion_per_scale_results_preserving_original as new_companion_per_scale,
    _select_best_scale as new_select_best,
    _walker_bootstrap_mass_std as new_walker_bootstrap,
)
from tests.physics.aic.conftest import (
    assert_dict_results_equal,
    assert_tensor_or_nan_equal,
)


# ---------------------------------------------------------------------------
# Channel lists
# ---------------------------------------------------------------------------

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

# Companion channels that _compute_companion_per_scale_results_preserving_original supports
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

    # Row-stochastic kernels with zero diagonal
    kernels = torch.rand(t_len, n_scales, n_walkers, n_walkers, dtype=torch.float32)
    eye = torch.eye(n_walkers, dtype=torch.float32).view(1, 1, n_walkers, n_walkers)
    kernels = kernels * (1.0 - eye)
    kernels = kernels / kernels.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Pairwise distances
    dx = positions[:, :, None, :] - positions[:, None, :, :]
    pairwise_distances = torch.linalg.vector_norm(dx, dim=-1)

    # Scales from distance quantiles
    finite_pos = pairwise_distances[torch.isfinite(pairwise_distances) & (pairwise_distances > 0)]
    if finite_pos.numel() > 0:
        probs = torch.linspace(0.2, 0.9, n_scales, dtype=torch.float32)
        scales = torch.quantile(finite_pos, probs).clamp(min=1e-6)
    else:
        scales = torch.linspace(1e-3, 1.0, n_scales, dtype=torch.float32)

    # Companion indices (cyclic shifts)
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


# =============================================================================
# TestParityComputeChannelSeries
# =============================================================================


class TestParityComputeChannelSeries:
    """Old and new _compute_channel_series_from_kernels must produce
    bit-for-bit identical results (both accept alive)."""

    @pytest.fixture
    def data(self):
        return _build_random_inputs(t_len=5, n_scales=4, n_walkers=8)

    @pytest.mark.parametrize("seed", [42, 99, 2025])
    def test_base_channels_parity(self, seed):
        data = _build_random_inputs(t_len=5, n_scales=4, n_walkers=8, seed=seed)
        shared_kwargs = {
            "color": data["color"],
            "color_valid": data["color_valid"],
            "positions": data["positions"],
            "alive": data["alive"],
            "force": data["force"],
            "kernels": data["kernels"],
            "channels": BASE_CHANNELS,
        }
        old_out = old_compute_series(**shared_kwargs)
        new_out = new_compute_series(**shared_kwargs)
        for ch in BASE_CHANNELS:
            assert ch in old_out, f"Channel {ch} missing from old output"
            assert ch in new_out, f"Channel {ch} missing from new output"
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"seed={seed}, channel {ch}: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )

    @pytest.mark.parametrize("seed", [42, 99, 2025])
    def test_all_channels_parity(self, seed):
        data = _build_random_inputs(t_len=4, n_scales=3, n_walkers=6, seed=seed)
        shared_kwargs = {
            "color": data["color"],
            "color_valid": data["color_valid"],
            "positions": data["positions"],
            "alive": data["alive"],
            "force": data["force"],
            "kernels": data["kernels"],
            "scales": data["scales"],
            "pairwise_distances": data["pairwise_distances"],
            "companions_distance": data["companions_distance"],
            "companions_clone": data["companions_clone"],
            "cloning_scores": data["cloning_scores"],
            "channels": ALL_CHANNELS,
        }
        old_out = old_compute_series(**shared_kwargs)
        new_out = new_compute_series(**shared_kwargs)
        for ch in ALL_CHANNELS:
            assert ch in old_out, f"Channel {ch} missing from old output"
            assert ch in new_out, f"Channel {ch} missing from new output"
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"seed={seed}, channel {ch}: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )

    def test_invalid_companions_parity(self, data):
        """With invalid (-1) companions, both old and new should produce zeros."""
        bad_comp = torch.full_like(data["companions_distance"], -1)
        shared_kwargs = {
            "color": data["color"],
            "color_valid": data["color_valid"],
            "positions": data["positions"],
            "alive": data["alive"],
            "force": data["force"],
            "kernels": data["kernels"],
            "scales": data["scales"],
            "pairwise_distances": data["pairwise_distances"],
            "companions_distance": bad_comp,
            "companions_clone": bad_comp,
            "cloning_scores": data["cloning_scores"],
            "channels": COMPANION_CHANNELS,
        }
        old_out = old_compute_series(**shared_kwargs)
        new_out = new_compute_series(**shared_kwargs)
        for ch in COMPANION_CHANNELS:
            assert torch.equal(old_out[ch], new_out[ch]), (
                f"Channel {ch} with invalid companions: max diff = "
                f"{(old_out[ch].float() - new_out[ch].float()).abs().max().item():.2e}"
            )


# =============================================================================
# TestParitySelectBestScale
# =============================================================================


class TestParitySelectBestScale:
    """Old and new _select_best_scale must return the same index."""

    def _make_synthetic_results(self, n_scales: int = 5) -> list[ChannelCorrelatorResult]:
        """Build a list of synthetic ChannelCorrelatorResult with varied quality."""
        results = []
        torch.manual_seed(777)
        for i in range(n_scales):
            n_lags = 10
            corr = torch.randn(n_lags + 1, dtype=torch.float32).abs() * (0.5 + i * 0.1)
            eff_mass = torch.randn(n_lags, dtype=torch.float32).abs()
            series = torch.randn(20, dtype=torch.float32)

            mass_val = 0.5 + i * 0.15
            mass_err = 0.02 + i * 0.01
            r2_val = 0.95 - i * 0.05
            aic_val = 10.0 + i * 2.0

            mass_fit = {
                "mass": mass_val,
                "mass_error": mass_err,
                "r_squared": r2_val,
                "n_valid_windows": 3,
                "best_window": {"aic": aic_val, "width": 5, "start": 2},
            }

            results.append(
                ChannelCorrelatorResult(
                    channel_name=f"test_ch_{i}",
                    correlator=corr,
                    correlator_err=None,
                    effective_mass=eff_mass,
                    mass_fit=mass_fit,
                    series=series,
                    n_samples=20,
                    dt=1.0,
                )
            )
        return results

    def test_select_best_parity(self):
        results = self._make_synthetic_results(n_scales=5)
        for kwargs in [
            {},
            {"min_r2": 0.8},
            {"min_windows": 2},
            {"max_error_pct": 10.0},
            {"remove_artifacts": True},
            {"min_r2": 0.85, "max_error_pct": 20.0, "remove_artifacts": True},
        ]:
            old_idx = old_select_best(results, **kwargs)
            new_idx = new_select_best(results, **kwargs)
            assert (
                old_idx == new_idx
            ), f"_select_best_scale mismatch with {kwargs}: old={old_idx}, new={new_idx}"

    def test_select_best_all_bad(self):
        """When all results fail filters, both should return None."""
        results = self._make_synthetic_results(n_scales=3)
        # Set all masses to NaN to make them all fail
        for r in results:
            r.mass_fit["mass"] = float("nan")
        old_idx = old_select_best(results)
        new_idx = new_select_best(results)
        assert old_idx is None
        assert new_idx is None

    def test_select_best_empty(self):
        """Empty results list should return None for both."""
        old_idx = old_select_best([])
        new_idx = new_select_best([])
        assert old_idx is None
        assert new_idx is None


# =============================================================================
# TestParityCompanionPerScale
# =============================================================================


class TestParityCompanionPerScale:
    """Old and new _compute_companion_per_scale_results_preserving_original
    must produce identical correlators and series."""

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

    def _shared_kwargs(self, d):
        return {
            "color": d["color"],
            "color_valid": d["color_valid"],
            "positions": d["positions"],
            "alive": d["alive"],
            "cloning_scores": d["cloning_scores"],
            "companions_distance": d["companions_distance"],
            "companions_clone": d["companions_clone"],
            "distance_ij": d["d_ij"],
            "distance_ik": d["d_ik"],
            "distance_jk": d["d_jk"],
            "scales": d["scales_1"],
            "channels": COMPANION_CORR_CHANNELS,
            "dt": 1.0,
            "config": CorrelatorConfig(max_lag=4, use_connected=True),
        }

    def test_correlators_parity(self, companion_data):
        kwargs = self._shared_kwargs(companion_data)
        old_out = old_companion_per_scale(**kwargs)
        new_out = new_companion_per_scale(**kwargs)

        for ch in COMPANION_CORR_CHANNELS:
            assert ch in old_out, f"{ch} missing from old output"
            assert ch in new_out, f"{ch} missing from new output"
            assert len(old_out[ch]) == len(
                new_out[ch]
            ), f"{ch}: scale count mismatch {len(old_out[ch])} vs {len(new_out[ch])}"
            for s_idx in range(len(old_out[ch])):
                old_corr = old_out[ch][s_idx].correlator
                new_corr = new_out[ch][s_idx].correlator
                assert torch.equal(old_corr, new_corr), (
                    f"Channel {ch} scale {s_idx} correlator differs: max diff = "
                    f"{(old_corr.float() - new_corr.float()).abs().max().item():.2e}"
                )

    def test_series_parity(self, companion_data):
        kwargs = self._shared_kwargs(companion_data)
        old_out = old_companion_per_scale(**kwargs)
        new_out = new_companion_per_scale(**kwargs)

        for ch in COMPANION_CORR_CHANNELS:
            for s_idx in range(len(old_out[ch])):
                old_series = old_out[ch][s_idx].series
                new_series = new_out[ch][s_idx].series
                assert torch.equal(old_series, new_series), (
                    f"Channel {ch} scale {s_idx} series differs: max diff = "
                    f"{(old_series.float() - new_series.float()).abs().max().item():.2e}"
                )


# =============================================================================
# TestParityWalkerBootstrap
# =============================================================================


class TestParityWalkerBootstrap:
    """Old and new _walker_bootstrap_mass_std must produce identical
    mass-std tensors (NaN-tolerant)."""

    def test_walker_bootstrap_parity(self):
        data = _build_random_inputs(t_len=6, n_scales=2, n_walkers=8, seed=77)
        cfg = CorrelatorConfig(max_lag=3, use_connected=True)
        shared_kwargs = {
            "color": data["color"],
            "color_valid": data["color_valid"],
            "positions": data["positions"],
            "alive": data["alive"],
            "force": data["force"],
            "cloning_scores": data["cloning_scores"],
            "companions_distance": data["companions_distance"],
            "companions_clone": data["companions_clone"],
            "kernels": data["kernels"],
            "scales": data["scales"],
            "pairwise_distances": data["pairwise_distances"],
            "channels": BASE_CHANNELS,
            "dt": 1.0,
            "config": cfg,
            "n_bootstrap": 5,
            "seed": 42,
        }
        old_out = old_walker_bootstrap(**shared_kwargs)
        new_out = new_walker_bootstrap(**shared_kwargs)

        for ch in BASE_CHANNELS:
            assert ch in old_out, f"{ch} missing from old output"
            assert ch in new_out, f"{ch} missing from new output"
            assert_tensor_or_nan_equal(old_out[ch], new_out[ch], label=f"bootstrap {ch}")
