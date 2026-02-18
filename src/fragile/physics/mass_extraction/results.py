"""Result dataclasses and extraction functions for mass fitting.

Provides structured containers for per-channel mass results, fit diagnostics,
effective mass data, and the top-level extraction result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gvar
import numpy as np


@dataclass
class ChannelMassResult:
    """Mass extraction result for a single channel group."""

    name: str
    channel_type: str
    ground_state_mass: Any  # gvar.GVar
    excited_masses: list[Any] = field(default_factory=list)  # list[gvar.GVar]
    energy_levels: list[Any] = field(default_factory=list)  # list[gvar.GVar]
    amplitudes: dict[str, Any] = field(default_factory=dict)
    dE: Any = None  # gvar array
    variant_keys: list[str] = field(default_factory=list)


@dataclass
class FitDiagnostics:
    """Diagnostics from the nonlinear fit."""

    chi2: float = 0.0
    dof: int = 0
    chi2_per_dof: float = 0.0
    Q: float = 0.0
    logGBF: float = 0.0
    nit: int = 0
    svdcut: float = 0.0
    fit_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectiveMassResult:
    """Effective mass for a single correlator."""

    datatag: str
    m_eff: Any  # gvar array
    t_values: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MassExtractionResult:
    """Complete output of the mass extraction pipeline."""

    channels: dict[str, ChannelMassResult] = field(default_factory=dict)
    diagnostics: FitDiagnostics = field(default_factory=FitDiagnostics)
    effective_masses: dict[str, EffectiveMassResult] = field(default_factory=dict)
    stability_scan: list[dict] | None = None
    data: dict[str, Any] = field(default_factory=dict)
    fit: Any = None  # lsqfit.nonlinear_fit


def extract_channel_results(
    fit: Any,
    groups: list,
    keys_per_group: dict[str, list[str]],
) -> dict[str, ChannelMassResult]:
    """Extract per-channel mass results from a completed fit.

    Reconstructs energy levels as ``E_n = sum(dE[0:n+1])`` and collects
    amplitudes per variant.

    Args:
        fit: Completed ``lsqfit.nonlinear_fit``.
        groups: List of :class:`ChannelGroupConfig`.
        keys_per_group: Group name -> list of correlator keys used.

    Returns:
        Dict mapping group name to :class:`ChannelMassResult`.
    """
    results = {}
    p = fit.p

    for group in groups:
        gname = group.name
        variant_keys = keys_per_group.get(gname, [])

        # Extract dE — BufferDict auto-derives exp(log(key)) when accessed.
        # Use try/except since the derived key may not appear in keys().
        dE_key = f"{gname}.dE"
        try:
            dE = p[dE_key]
        except KeyError:
            continue

        # Reconstruct energy levels: E_n = sum(dE[0:n+1])
        energy_levels = []
        for n in range(len(dE)):
            E_n = sum(dE[: n + 1])
            energy_levels.append(E_n)

        ground = energy_levels[0] if energy_levels else gvar.gvar(0, 0)
        excited = energy_levels[1:] if len(energy_levels) > 1 else []

        # Collect amplitudes per variant
        amplitudes = {}
        for key in variant_keys:
            amp_key = f"{gname}.{key}.a"
            if amp_key in p:
                amplitudes[key] = p[amp_key]

        results[gname] = ChannelMassResult(
            name=gname,
            channel_type=group.channel_type,
            ground_state_mass=ground,
            excited_masses=excited,
            energy_levels=energy_levels,
            amplitudes=amplitudes,
            dE=dE,
            variant_keys=variant_keys,
        )

    return results


def extract_diagnostics(fit: Any) -> FitDiagnostics:
    """Extract diagnostics from a completed fit.

    Args:
        fit: Completed ``lsqfit.nonlinear_fit``.

    Returns:
        :class:`FitDiagnostics` with fit quality metrics.
    """
    return FitDiagnostics(
        chi2=float(fit.chi2),
        dof=int(fit.dof),
        chi2_per_dof=float(fit.chi2 / fit.dof) if fit.dof > 0 else 0.0,
        Q=float(fit.Q),
        logGBF=float(fit.logGBF) if hasattr(fit, "logGBF") else 0.0,
        nit=int(fit.nit) if hasattr(fit, "nit") else 0,
        svdcut=float(fit.svdcut) if fit.svdcut is not None else 0.0,
        fit_parameters=dict(fit.p.items()),
    )
