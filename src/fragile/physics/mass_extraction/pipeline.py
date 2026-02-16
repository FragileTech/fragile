"""Top-level orchestrator for the mass extraction pipeline.

Converts ``PipelineResult`` correlators into Bayesian multi-exponential fits,
extracting particle masses with proper error propagation.
"""

from __future__ import annotations

import numpy as np

from fragile.physics.operators.pipeline import PipelineResult

from .config import (
    ChannelGroupConfig,
    MassExtractionConfig,
)
from .data_preparation import correlators_to_gvar, multi_run_correlators_to_gvar
from .effective_mass import compute_effective_mass_for_all
from .fitting import run_fit
from .models import build_all_models
from .priors import build_combined_prior
from .results import (
    MassExtractionResult,
    extract_channel_results,
    extract_diagnostics,
)


# ---------------------------------------------------------------------------
# Channel auto-detection mapping
# ---------------------------------------------------------------------------

_CHANNEL_GROUP_MAP = {
    # Meson family
    "scalar": "pion",
    "pseudoscalar": "pion",
    # Vector family
    "vector": "rho",
    "axial_vector": "rho",
    # Baryon family
    "nucleon": "nucleon",
    "baryon": "nucleon",
    # Glueball family (matches glueball_*)
    "glueball": "glueball",
    # Tensor family (matches tensor_*)
    "tensor": "tensor",
}

_CHANNEL_TYPES = {
    "pion": "meson",
    "rho": "meson",
    "nucleon": "baryon",
    "glueball": "glueball",
    "tensor": "tensor",
}


def extract_masses(
    pipeline_result: PipelineResult,
    config: MassExtractionConfig | None = None,
) -> MassExtractionResult:
    """Extract particle masses from a pipeline result.

    Performs the full analysis chain:

    1. Auto-detect channel groups from correlator keys (or use config).
    2. Expand multiscale ``[S, L]`` correlators into per-scale keys.
    3. Convert torch tensors to gvar arrays with covariance estimation.
    4. Build Corr2 models with shared dE and independent amplitudes.
    5. Build priors (optionally fastfit-seeded).
    6. Run simultaneous Bayesian fit.
    7. Extract per-channel masses with gvar error propagation.
    8. Compute effective masses (optional cross-check).

    Args:
        pipeline_result: Output of ``compute_strong_force_pipeline``.
        config: Mass extraction configuration. ``None`` uses defaults.

    Returns:
        :class:`MassExtractionResult` with extracted masses and diagnostics.
    """
    if config is None:
        config = MassExtractionConfig()

    correlators = pipeline_result.correlators
    operators = pipeline_result.operators

    # Step 1: Determine channel groups
    if config.channel_groups:
        groups = config.channel_groups
    else:
        groups = _auto_detect_channel_groups(list(correlators.keys()))

    if not groups:
        return MassExtractionResult()

    # Step 2-3: Convert to gvar with covariance
    data = correlators_to_gvar(correlators, operators, config)

    if not data:
        return MassExtractionResult(data=data)

    # Determine available keys and max lags
    available_keys = list(data.keys())
    max_lags = {k: len(v) - 1 for k, v in data.items()}

    # Update group correlator_keys to only include available ones
    keys_per_group: dict[str, list[str]] = {}
    active_groups = []
    for group in groups:
        active_keys = [k for k in group.correlator_keys if k in available_keys]
        if active_keys:
            group.correlator_keys = active_keys
            keys_per_group[group.name] = active_keys
            active_groups.append(group)

    if not active_groups:
        return MassExtractionResult(data=data)

    # Step 4: Build models
    models = build_all_models(active_groups, available_keys, max_lags)

    if not models:
        return MassExtractionResult(data=data)

    # Step 5: Build priors
    prior = build_combined_prior(active_groups, data, keys_per_group)

    # Step 6: Run fit
    fit = run_fit(data, models, prior, config.fit)

    # Step 7: Extract results
    channels = extract_channel_results(fit, active_groups, keys_per_group)
    diagnostics = extract_diagnostics(fit)

    # Step 8: Effective masses (optional)
    effective_masses = {}
    if config.compute_effective_mass:
        effective_masses = compute_effective_mass_for_all(
            data,
            dt=config.effective_mass_dt,
            method=config.effective_mass_method,
        )

    return MassExtractionResult(
        channels=channels,
        diagnostics=diagnostics,
        effective_masses=effective_masses,
        data=data,
        fit=fit,
    )


def extract_masses_multi_run(
    pipeline_results: list[PipelineResult],
    config: MassExtractionConfig | None = None,
) -> MassExtractionResult:
    """Extract masses using inter-run variation for covariance.

    This is the gold-standard approach: runs multiple independent simulations
    and uses the scatter between them for error estimation.

    Args:
        pipeline_results: List of pipeline results from independent runs.
        config: Mass extraction configuration.

    Returns:
        :class:`MassExtractionResult` with inter-run error estimates.
    """
    if not pipeline_results:
        return MassExtractionResult()

    if config is None:
        config = MassExtractionConfig()

    # Collect correlators from all runs
    run_correlators = [pr.correlators for pr in pipeline_results]
    data = multi_run_correlators_to_gvar(run_correlators)

    if not data:
        return MassExtractionResult()

    # Use first result's correlator keys for group detection
    all_keys = list(data.keys())
    if config.channel_groups:
        groups = config.channel_groups
    else:
        groups = _auto_detect_channel_groups(all_keys)

    if not groups:
        return MassExtractionResult(data=data)

    available_keys = list(data.keys())
    max_lags = {k: len(v) - 1 for k, v in data.items()}

    keys_per_group: dict[str, list[str]] = {}
    active_groups = []
    for group in groups:
        active_keys = [k for k in group.correlator_keys if k in available_keys]
        if active_keys:
            group.correlator_keys = active_keys
            keys_per_group[group.name] = active_keys
            active_groups.append(group)

    if not active_groups:
        return MassExtractionResult(data=data)

    models = build_all_models(active_groups, available_keys, max_lags)
    if not models:
        return MassExtractionResult(data=data)

    prior = build_combined_prior(active_groups, data, keys_per_group)
    fit = run_fit(data, models, prior, config.fit)

    channels = extract_channel_results(fit, active_groups, keys_per_group)
    diagnostics = extract_diagnostics(fit)

    effective_masses = {}
    if config.compute_effective_mass:
        effective_masses = compute_effective_mass_for_all(
            data,
            dt=config.effective_mass_dt,
            method=config.effective_mass_method,
        )

    return MassExtractionResult(
        channels=channels,
        diagnostics=diagnostics,
        effective_masses=effective_masses,
        data=data,
        fit=fit,
    )


def _auto_detect_channel_groups(
    correlator_keys: list[str],
) -> list[ChannelGroupConfig]:
    """Auto-detect channel groups from correlator key names.

    Maps known channel names to physical particle groups:

    - scalar, pseudoscalar -> "pion" (meson)
    - vector, axial_vector -> "rho" (meson)
    - nucleon, baryon -> "nucleon" (baryon)
    - glueball* -> "glueball"
    - tensor* -> "tensor"

    Args:
        correlator_keys: List of correlator key strings.

    Returns:
        List of auto-detected :class:`ChannelGroupConfig`.
    """
    group_keys: dict[str, list[str]] = {}

    for key in correlator_keys:
        # Strip scale suffix for matching
        base = key.rsplit("_scale_", 1)[0] if "_scale_" in key else key

        # Try exact match first
        group_name = _CHANNEL_GROUP_MAP.get(base)

        # Try prefix match for glueball_*, tensor_*
        if group_name is None:
            for prefix, gname in _CHANNEL_GROUP_MAP.items():
                if base.startswith(prefix):
                    group_name = gname
                    break

        if group_name is None:
            # Unknown channel -> create its own group
            group_name = base

        if group_name not in group_keys:
            group_keys[group_name] = []
        group_keys[group_name].append(key)

    groups = []
    for gname, keys in group_keys.items():
        channel_type = _CHANNEL_TYPES.get(gname, "meson")
        groups.append(
            ChannelGroupConfig(
                name=gname,
                correlator_keys=sorted(keys),
                channel_type=channel_type,
            )
        )

    return groups
