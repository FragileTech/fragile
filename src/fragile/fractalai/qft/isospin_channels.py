"""Isospin-filtered correlator channel computation.

Splits walkers by will_clone (isospin) into up-type (cloners) and
down-type (persisters), then computes separate correlator masses
for each component using the standard operator + FFT pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.aggregation import (
    AggregatedTimeSeries,
    aggregate_time_series,
    build_gamma_matrices,
    compute_glueball_operators,
    compute_scalar_operators,
    compute_pseudoscalar_operators,
    compute_vector_operators,
    compute_axial_vector_operators,
    compute_tensor_operators,
    compute_nucleon_operators,
)
from fragile.fractalai.qft.correlator_channels import (
    ChannelConfig,
    ChannelCorrelatorResult,
    CorrelatorConfig,
    compute_channel_correlator,
)


@dataclass
class IsospinChannelResult:
    """Results from isospin-split correlator analysis."""

    up_results: dict[str, ChannelCorrelatorResult]
    down_results: dict[str, ChannelCorrelatorResult]


# PDG quark mass ratios (up-type / down-type) per generation
ISOSPIN_MASS_RATIOS: dict[str, float] = {
    "Gen1 (u/d)": 0.00216 / 0.00467,     # ≈ 0.46
    "Gen2 (c/s)": 1.27 / 0.0934,         # ≈ 13.6
    "Gen3 (t/b)": 172.69 / 4.18,         # ≈ 41.3
}

# Per-channel observed isospin splittings.
# Each channel corresponds to a lightest hadron. When we split walkers by
# will_clone (up-type cloners vs down-type persisters) we effectively measure
# the mass of the same channel in two isospin sectors.  The observed ratio
# gives the PDG mass of the lightest up-type hadron over the lightest
# down-type hadron in each channel's quantum numbers (I³ = +½ vs −½).
#
# Format: channel → (up_particle, down_particle, m_up_GeV, m_down_GeV, ratio)
ISOSPIN_CHANNEL_SPLITTINGS: dict[str, tuple[str, str, float, float, float]] = {
    # Pseudoscalar: π⁺(ud̄) vs π⁻(dū) — degenerate in isospin limit
    "pseudoscalar": ("π⁺ (ud̄)", "π⁻ (dū)", 0.13957, 0.13957, 1.000),
    # Vector: ρ⁺ vs ρ⁻ — degenerate
    "vector": ("ρ⁺", "ρ⁻", 0.77526, 0.77526, 1.000),
    # Scalar: a₀⁺(980) vs a₀⁻(980) — degenerate
    "scalar": ("a₀⁺(980)", "a₀⁻(980)", 0.980, 0.980, 1.000),
    # Axial vector: a₁⁺(1260) vs a₁⁻(1260)
    "axial_vector": ("a₁⁺(1260)", "a₁⁻(1260)", 1.230, 1.230, 1.000),
    # Tensor: a₂⁺(1320) vs a₂⁻(1320)
    "tensor": ("a₂⁺(1320)", "a₂⁻(1320)", 1.3183, 1.3183, 1.000),
    # Nucleon: proton (uud) vs neutron (udd) — small isospin splitting
    "nucleon": ("proton (uud)", "neutron (udd)", 0.938272, 0.939565, 0.998624),
    # Glueball: flavour-blind, no isospin splitting expected
    "glueball": ("0⁺⁺ glueball", "0⁺⁺ glueball", 1.710, 1.710, 1.000),
}


def _compute_filtered_operators(
    agg_data: AggregatedTimeSeries,
    gamma: dict[str, Tensor],
    channels: list[str],
    walker_mask: Tensor,
) -> dict[str, Tensor]:
    """Compute operators only for walkers where walker_mask is True.

    For bilinear channels (scalar, pseudoscalar, vector, axial_vector, tensor),
    we AND the isospin mask into ``color_valid`` so that operators only average
    over the selected walkers. For glueball (force-squared), we AND into
    ``alive`` instead.

    Args:
        agg_data: Aggregated time series data (shared, not copied).
        gamma: Gamma matrices.
        channels: List of channel names to compute.
        walker_mask: [T, N] bool tensor — True for walkers to include.
    """
    # Inject isospin mask into color_valid for bilinear channels
    filtered_valid = agg_data.color_valid & walker_mask[:agg_data.color_valid.shape[0]]
    filtered_alive = agg_data.alive & walker_mask[:agg_data.alive.shape[0]]

    # Shallow copy with filtered validity (no memory duplication of large arrays)
    filtered_agg = replace(agg_data, color_valid=filtered_valid, alive=filtered_alive)

    operators: dict[str, Tensor] = {}
    for ch in channels:
        if ch == "scalar":
            operators[ch] = compute_scalar_operators(filtered_agg, gamma)
        elif ch == "pseudoscalar":
            operators[ch] = compute_pseudoscalar_operators(filtered_agg, gamma)
        elif ch == "vector":
            operators[ch] = compute_vector_operators(filtered_agg, gamma)
        elif ch == "axial_vector":
            operators[ch] = compute_axial_vector_operators(filtered_agg, gamma)
        elif ch == "tensor":
            operators[ch] = compute_tensor_operators(filtered_agg, gamma)
        elif ch == "nucleon":
            operators[ch] = compute_nucleon_operators(filtered_agg, gamma)
        elif ch == "glueball":
            operators[ch] = compute_glueball_operators(filtered_agg)
    return operators


def _operators_to_correlators(
    operators: dict[str, Tensor],
    dt: float,
    correlator_config: CorrelatorConfig,
) -> dict[str, ChannelCorrelatorResult]:
    """Run correlator analysis on each channel's operator series."""
    results: dict[str, ChannelCorrelatorResult] = {}
    for name, series in operators.items():
        results[name] = compute_channel_correlator(
            series=series,
            dt=dt,
            config=correlator_config,
            channel_name=name,
        )
    return results


def compute_isospin_channels(
    history: RunHistory,
    channel_config: ChannelConfig,
    correlator_config: CorrelatorConfig,
    channels: list[str] | None = None,
) -> IsospinChannelResult:
    """Compute correlator masses split by isospin (will_clone).

    Up-type walkers are those with ``will_clone=True`` (cloners).
    Down-type walkers are those with ``will_clone=False`` (persisters).

    The expensive aggregation pass (neighbor topology, color states) is
    done once and shared between both isospin components.

    Args:
        history: RunHistory with will_clone data.
        channel_config: Aggregation configuration (shared with strong force).
        correlator_config: Correlator fitting configuration.
        channels: Channel names to compute (None = default set).

    Returns:
        IsospinChannelResult with up_results and down_results dicts.
    """
    if channels is None:
        channels = ["scalar", "pseudoscalar", "vector", "axial_vector",
                     "tensor", "nucleon", "glueball"]

    # Filter nucleon in low dimensions
    if history.d < 3:
        channels = [ch for ch in channels if ch != "nucleon"]

    # 1. Single aggregation pass (the expensive part)
    agg_data = aggregate_time_series(history, channel_config)

    # 2. Build gamma matrices once
    gamma = build_gamma_matrices(agg_data.d, agg_data.device)

    # 3. Extract will_clone aligned to the aggregation time range
    #    aggregate_time_series uses start_idx = max(1, warmup_fraction * n_recorded)
    #    and slices history arrays as [start_idx-1 : n_recorded-1] for force,
    #    and [start_idx : n_recorded-1] for color. The alive mask shape is [T, N].
    start_idx = max(1, int(history.n_recorded * channel_config.warmup_fraction))
    T = agg_data.alive.shape[0]
    N = agg_data.n_walkers

    # will_clone[t] tells us which walkers will clone at step t.
    # The aggregation time range starts at start_idx (for color) but alive
    # comes from start_idx-1.  We align will_clone to the same T-length window.
    wc_raw = history.will_clone[start_idx - 1: start_idx - 1 + T]  # [T_raw, N]
    # Ensure we have the right length (pad with False if history is short)
    if wc_raw.shape[0] < T:
        pad = torch.zeros(T - wc_raw.shape[0], N, dtype=torch.bool, device=wc_raw.device)
        wc_raw = torch.cat([wc_raw, pad], dim=0)
    will_clone = wc_raw[:T].bool().to(agg_data.device)  # [T, N]

    # 4. Compute operators for each isospin component
    up_operators = _compute_filtered_operators(agg_data, gamma, channels, will_clone)
    down_operators = _compute_filtered_operators(agg_data, gamma, channels, ~will_clone)

    # 5. Run correlator analysis on each component
    up_results = _operators_to_correlators(up_operators, agg_data.dt, correlator_config)
    down_results = _operators_to_correlators(down_operators, agg_data.dt, correlator_config)

    return IsospinChannelResult(up_results=up_results, down_results=down_results)
