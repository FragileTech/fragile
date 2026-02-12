"""Tests for batched GEVP companion-channel utilities."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fragile.fractalai.qft.correlator_channels import ChannelCorrelatorResult
from fragile.fractalai.qft.gevp_channels import (
    compute_companion_channel_gevp,
    compute_companion_nucleon_gevp,
    GEVPConfig,
)


def _make_channel_result(
    name: str,
    series: torch.Tensor,
    *,
    dt: float = 1.0,
    mass: float = 0.5,
    mass_error: float = 0.05,
    r_squared: float = 0.9,
    n_valid_windows: int = 6,
) -> ChannelCorrelatorResult:
    series_f = series.float().clone()
    corr = torch.clamp(series_f, min=1e-12)
    return ChannelCorrelatorResult(
        channel_name=name,
        correlator=corr,
        correlator_err=None,
        effective_mass=torch.zeros(max(0, corr.numel() - 1), dtype=torch.float32),
        mass_fit={
            "mass": float(mass),
            "mass_error": float(mass_error),
            "r_squared": float(r_squared),
            "n_valid_windows": int(n_valid_windows),
        },
        series=series_f,
        n_samples=int(series_f.numel()),
        dt=float(dt),
    )


def _build_base_results(t_len: int = 96) -> dict[str, ChannelCorrelatorResult]:
    torch.manual_seed(1234)
    t = torch.arange(t_len, dtype=torch.float32)
    ground = torch.exp(-0.35 * t)
    excited = torch.exp(-0.95 * t)
    coeffs = {
        "nucleon": (1.0, 0.9),
        "nucleon_flux_action": (0.8, 0.6),
        "nucleon_flux_sin2": (1.2, 0.4),
        "nucleon_flux_exp": (0.7, 1.1),
        "nucleon_score_abs": (1.1, 0.3),
    }
    out: dict[str, ChannelCorrelatorResult] = {}
    for name, (a0, a1) in coeffs.items():
        noise = 0.002 * torch.randn_like(t)
        series = a0 * ground + a1 * excited + noise
        out[name] = _make_channel_result(name, series, dt=1.0)
    return out


def test_companion_nucleon_gevp_recovers_reasonable_ground_mass() -> None:
    """GEVP principal state should land in a physically plausible low-mass range."""
    base_results = _build_base_results()
    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        max_basis=10,
        compute_bootstrap_errors=False,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
    )

    mass = float(gevp.result.mass_fit.get("mass", float("nan")))
    assert mass > 0.2
    assert mass < 0.6
    assert int(gevp.result.mass_fit.get("gevp_n_basis_input", 0)) == 5


def test_companion_channel_gevp_scalar_uses_scalar_family_only() -> None:
    """Generic companion GEVP should restrict the basis to the selected channel family."""
    base_results = _build_base_results()
    t = torch.arange(96, dtype=torch.float32)
    ground = torch.exp(-0.30 * t)
    excited = torch.exp(-0.85 * t)
    scalar_coeffs = {
        "scalar": (1.0, 0.7),
        "scalar_score_directed": (0.8, 0.5),
        "scalar_score_weighted": (1.1, 0.4),
        "scalar_raw": (0.9, 0.8),
        "scalar_abs2_vacsub": (1.2, 0.3),
    }
    for name, (a0, a1) in scalar_coeffs.items():
        series = a0 * ground + a1 * excited + 0.002 * torch.randn_like(t)
        base_results[name] = _make_channel_result(name, series, dt=1.0)

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        max_basis=10,
    )
    gevp = compute_companion_channel_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
        base_channel="scalar",
    )

    labels = set(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert labels == set(scalar_coeffs)
    assert gevp.result.channel_name == "scalar_gevp"
    assert str(gevp.result.mass_fit.get("source", "")) == "gevp_scalar"
    assert str(gevp.result.mass_fit.get("base_channel", "")) == "scalar"


@pytest.mark.parametrize(
    ("base_channel", "family_channels"),
    [
        (
            "pseudoscalar",
            (
                "pseudoscalar",
                "pseudoscalar_score_directed",
                "pseudoscalar_score_weighted",
            ),
        ),
        (
            "glueball",
            (
                "glueball",
                "glueball_phase_action",
                "glueball_phase_sin2",
            ),
        ),
    ],
)
def test_companion_channel_gevp_supports_requested_families(
    base_channel: str,
    family_channels: tuple[str, ...],
) -> None:
    """Generic companion GEVP should run for pseudoscalar and glueball families."""
    base_results = _build_base_results()
    t = torch.arange(96, dtype=torch.float32)
    ground = torch.exp(-0.25 * t)
    excited = torch.exp(-0.75 * t)
    for idx, name in enumerate(family_channels):
        a0 = 1.0 + 0.1 * idx
        a1 = 0.6 - 0.1 * idx
        series = a0 * ground + a1 * excited + 0.002 * torch.randn_like(t)
        base_results[name] = _make_channel_result(name, series, dt=1.0)

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        max_basis=10,
    )
    gevp = compute_companion_channel_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
        base_channel=base_channel,
    )

    labels = set(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert labels == set(family_channels)
    assert gevp.result.channel_name == f"{base_channel}_gevp"
    assert str(gevp.result.mass_fit.get("source", "")) == f"gevp_{base_channel}"
    assert str(gevp.result.mass_fit.get("base_channel", "")) == base_channel


def test_companion_channel_gevp_rejects_unsupported_base_channel() -> None:
    """Unsupported base channels should fail fast with a clear message."""
    base_results = _build_base_results()
    cfg = GEVPConfig(
        t0=2,
        max_lag=20,
        fit_mode="linear",
        fit_start=2,
        min_fit_points=3,
        basis_strategy="base_only",
    )
    with pytest.raises(ValueError, match="Unsupported companion GEVP base channel"):
        compute_companion_channel_gevp(
            base_results=base_results,
            multiscale_output=None,
            config=cfg,
            base_channel="vector",
        )


def test_companion_nucleon_gevp_prunes_ill_conditioned_basis() -> None:
    """Nearly dependent operators should be pruned by condition-aware basis selection."""
    t = torch.arange(96, dtype=torch.float32)
    base = torch.exp(-0.45 * t)
    tiny = 1e-7
    results = {
        "nucleon": _make_channel_result("nucleon", base + tiny * torch.randn_like(base)),
        "nucleon_flux_action": _make_channel_result(
            "nucleon_flux_action", base + tiny * torch.randn_like(base)
        ),
        "nucleon_flux_sin2": _make_channel_result(
            "nucleon_flux_sin2", base + tiny * torch.randn_like(base)
        ),
        "nucleon_flux_exp": _make_channel_result(
            "nucleon_flux_exp", base + tiny * torch.randn_like(base)
        ),
        "nucleon_score_abs": _make_channel_result(
            "nucleon_score_abs", base + tiny * torch.randn_like(base)
        ),
    }
    cfg = GEVPConfig(
        t0=2,
        max_lag=32,
        use_connected=False,
        fit_mode="linear",
        fit_start=2,
        min_fit_points=3,
        basis_strategy="base_only",
        eig_rel_cutoff=1e-2,
        cond_limit=20.0,
        max_basis=10,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=results,
        multiscale_output=None,
        config=cfg,
    )

    n_input = int(gevp.result.mass_fit.get("gevp_n_basis_input", 0))
    n_kept = int(gevp.result.mass_fit.get("gevp_n_basis_kept", 0))
    assert n_input == 5
    assert n_kept < n_input


def test_companion_nucleon_gevp_uses_best_scale_augmented_basis() -> None:
    """base_plus_best_scale strategy should include multiscale best operators when present."""
    base_results = _build_base_results()

    per_scale_results = {}
    best_scale_index = {}
    for base_name in (
        "nucleon",
        "nucleon_flux_action",
        "nucleon_flux_sin2",
        "nucleon_flux_exp",
        "nucleon_score_abs",
    ):
        series = base_results[base_name].series
        scale0 = _make_channel_result(f"{base_name}_companion_s0", series)
        scale1 = _make_channel_result(f"{base_name}_companion_s1", 0.9 * series + 0.001)
        per_scale_results[f"{base_name}_companion"] = [scale0, scale1]
        best_scale_index[f"{base_name}_companion"] = 1

    multiscale = SimpleNamespace(
        per_scale_results=per_scale_results,
        best_scale_index=best_scale_index,
    )

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_plus_best_scale",
        max_basis=10,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=base_results,
        multiscale_output=multiscale,
        config=cfg,
    )

    labels = list(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert any("@best_scale[" in label for label in labels)
    assert int(gevp.result.mass_fit.get("gevp_n_basis_input", 0)) >= 8


def test_companion_nucleon_gevp_filters_basis_by_r2_and_window_count() -> None:
    """GEVP basis should include only operators that pass R² and window-count filters."""
    base_results = _build_base_results()
    base_results["nucleon"].mass_fit["r_squared"] = 0.95
    base_results["nucleon"].mass_fit["n_valid_windows"] = 7
    base_results["nucleon_flux_action"].mass_fit["r_squared"] = 0.35
    base_results["nucleon_flux_action"].mass_fit["n_valid_windows"] = 7
    base_results["nucleon_flux_sin2"].mass_fit["r_squared"] = 0.93
    base_results["nucleon_flux_sin2"].mass_fit["n_valid_windows"] = 1
    base_results["nucleon_flux_exp"].mass_fit["r_squared"] = 0.88
    base_results["nucleon_flux_exp"].mass_fit["n_valid_windows"] = 6
    base_results["nucleon_score_abs"].mass_fit["r_squared"] = 0.25
    base_results["nucleon_score_abs"].mass_fit["n_valid_windows"] = 0

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        min_operator_r2=0.8,
        min_operator_windows=3,
        max_basis=10,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
    )

    labels = list(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert set(labels) == {"nucleon", "nucleon_flux_exp"}
    assert int(gevp.result.mass_fit.get("gevp_n_basis_input", 0)) == 2
    assert int(gevp.result.mass_fit.get("gevp_min_operator_windows", -1)) == 3
    assert float(gevp.result.mass_fit.get("gevp_min_operator_r2", float("nan"))) == 0.8
    assert any("Operator quality filter excluded 3 basis vectors" in note for note in gevp.notes)


def test_companion_nucleon_gevp_filters_basis_by_max_error_pct() -> None:
    """GEVP basis should exclude operators whose relative mass error exceeds threshold."""
    base_results = _build_base_results()
    base_results["nucleon"].mass_fit["mass"] = 1.0
    base_results["nucleon"].mass_fit["mass_error"] = 0.10
    base_results["nucleon_flux_action"].mass_fit["mass"] = 1.0
    base_results["nucleon_flux_action"].mass_fit["mass_error"] = 0.45
    base_results["nucleon_flux_sin2"].mass_fit["mass"] = 1.0
    base_results["nucleon_flux_sin2"].mass_fit["mass_error"] = 0.25
    base_results["nucleon_flux_exp"].mass_fit["mass"] = 1.0
    base_results["nucleon_flux_exp"].mass_fit["mass_error"] = 0.35
    base_results["nucleon_score_abs"].mass_fit["mass"] = 1.0
    base_results["nucleon_score_abs"].mass_fit["mass_error"] = 0.05

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        max_operator_error_pct=30.0,
        max_basis=10,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
    )

    labels = set(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert labels == {"nucleon", "nucleon_flux_sin2", "nucleon_score_abs"}
    assert float(gevp.result.mass_fit.get("gevp_max_operator_error_pct", float("nan"))) == 30.0
    assert any("err_pct=" in note for note in gevp.notes)


def test_companion_nucleon_gevp_removes_artifact_operators_when_enabled() -> None:
    """Artifact filter should remove zero/invalid error channels and mass==0 channels."""
    base_results = _build_base_results()
    base_results["nucleon"] = _make_channel_result(
        "nucleon",
        base_results["nucleon"].series,
        mass=0.8,
        mass_error=0.0,
        r_squared=0.95,
        n_valid_windows=8,
    )
    base_results["nucleon_flux_action"] = _make_channel_result(
        "nucleon_flux_action",
        base_results["nucleon_flux_action"].series,
        mass=0.75,
        mass_error=0.12,
        r_squared=0.95,
        n_valid_windows=8,
    )
    base_results["nucleon_flux_sin2"] = _make_channel_result(
        "nucleon_flux_sin2",
        base_results["nucleon_flux_sin2"].series,
        mass=0.0,
        mass_error=0.15,
        r_squared=0.92,
        n_valid_windows=8,
    )
    base_results["nucleon_flux_exp"] = _make_channel_result(
        "nucleon_flux_exp",
        base_results["nucleon_flux_exp"].series,
        mass=0.73,
        mass_error=float("inf"),
        r_squared=0.9,
        n_valid_windows=8,
    )
    base_results["nucleon_score_abs"] = _make_channel_result(
        "nucleon_score_abs",
        base_results["nucleon_score_abs"].series,
        mass=0.72,
        mass_error=0.13,
        r_squared=0.91,
        n_valid_windows=8,
    )

    cfg = GEVPConfig(
        t0=2,
        max_lag=40,
        use_connected=False,
        fit_mode="linear",
        fit_start=3,
        min_fit_points=3,
        basis_strategy="base_only",
        min_operator_r2=0.0,
        min_operator_windows=0,
        remove_artifacts=True,
        max_basis=10,
    )
    gevp = compute_companion_nucleon_gevp(
        base_results=base_results,
        multiscale_output=None,
        config=cfg,
    )

    labels = set(gevp.result.mass_fit.get("gevp_basis_labels", []))
    assert "nucleon" not in labels
    assert "nucleon_flux_sin2" not in labels
    assert "nucleon_flux_exp" not in labels
    assert int(gevp.result.mass_fit.get("gevp_n_basis_input", 0)) == 2
    assert bool(gevp.result.mass_fit.get("gevp_remove_artifacts", False))
    assert any("mass_error==0" in note for note in gevp.notes)
    assert any("mass==0" in note for note in gevp.notes)
    assert any("mass_error=nan_or_inf" in note for note in gevp.notes)
