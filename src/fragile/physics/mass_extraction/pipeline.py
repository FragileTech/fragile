"""Top-level orchestrator for the mass extraction pipeline.

Converts ``PipelineResult`` correlators into Bayesian multi-exponential fits,
extracting particle masses with proper error propagation.
"""

from __future__ import annotations

import logging
import re

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
    extract_channel_results,
    extract_diagnostics,
    MassExtractionResult,
)


_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel auto-detection mapping
# ---------------------------------------------------------------------------

# Regex patterns: match channel prefix followed by underscore (mode suffix) or
# end-of-string (bare key).  Propagator prefixes (longer strings) must precede
# bare prefixes, and ``axial_vector`` must precede ``vector`` so that
# ``axial_vector_standard`` is not mis-matched.
_CHANNEL_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # Propagator prefixes (longer → must precede bare prefixes)
    (re.compile(r"^meson_pseudoscalar(?:_|$)"), "pseudoscalar", "meson"),
    (re.compile(r"^meson_scalar(?:_|$)"), "scalar", "meson"),
    (re.compile(r"^axial_full(?:_|$)"), "axial_vector", "meson"),
    (re.compile(r"^vector_full(?:_|$)"), "vector", "meson"),
    (re.compile(r"^baryon_nucleon(?:_|$)"), "nucleon", "baryon"),
    (re.compile(r"^glueball_plaquette(?:_|$)"), "glueball", "glueball"),
    # Regular operator prefixes
    (re.compile(r"^axial_vector(?:_|$)"), "axial_vector", "meson"),
    (re.compile(r"^pseudoscalar(?:_|$)"), "pseudoscalar", "meson"),
    (re.compile(r"^scalar(?:_|$)"), "scalar", "meson"),
    (re.compile(r"^vector(?:_|$)"), "vector", "meson"),
    (re.compile(r"^nucleon(?:_|$)"), "nucleon", "baryon"),
    (re.compile(r"^baryon(?:_|$)"), "nucleon", "baryon"),
    (re.compile(r"^glueball(?:_|$)"), "glueball", "glueball"),
    (re.compile(r"^tensor(?:_|$)"), "tensor", "tensor"),
    # Electroweak operator prefixes
    (re.compile(r"^u1_"), "u1_hypercharge", "electroweak"),
    (re.compile(r"^su2_doublet_diff(?:_|$)"), "su2_doublet_diff", "electroweak"),
    (re.compile(r"^su2_doublet(?:_|$)"), "su2_doublet", "electroweak"),
    (re.compile(r"^su2_phase(?:_|$)"), "su2_phase", "electroweak"),
    (re.compile(r"^su2_component(?:_|$)"), "su2_component", "electroweak"),
    (re.compile(r"^ew_mixed(?:_|$)"), "ew_mixed", "electroweak"),
    (re.compile(r"^fitness_phase(?:_|$)"), "symmetry_breaking", "electroweak"),
    (re.compile(r"^clone_indicator(?:_|$)"), "symmetry_breaking", "electroweak"),
    (re.compile(r"^velocity_norm_"), "parity_velocity", "electroweak"),
]

_CHANNEL_GROUP_MAP = {
    "scalar": "scalar",
    "pseudoscalar": "pseudoscalar",
    "meson_scalar": "scalar",
    "meson_pseudoscalar": "pseudoscalar",
    "vector": "vector",
    "axial_vector": "axial_vector",
    "vector_full": "vector",
    "axial_full": "axial_vector",
    "nucleon": "nucleon",
    "baryon": "nucleon",
    "baryon_nucleon": "nucleon",
    "glueball": "glueball",
    "glueball_plaquette": "glueball",
    "tensor": "tensor",
}

_CHANNEL_TYPES = {
    "scalar": "meson",
    "pseudoscalar": "meson",
    "vector": "meson",
    "axial_vector": "meson",
    "nucleon": "baryon",
    "glueball": "glueball",
    "tensor": "tensor",
    "u1_hypercharge": "electroweak",
    "su2_isospin": "electroweak",
    "ew_mixed": "electroweak",
    "symmetry_breaking": "electroweak",
    "parity_velocity": "electroweak",
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
        groups = _auto_detect_channel_groups(
            list(correlators.keys()),
            include_multiscale=config.include_multiscale,
        )

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
        groups = _auto_detect_channel_groups(
            all_keys,
            include_multiscale=config.include_multiscale,
        )

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
    *,
    include_multiscale: bool = True,
) -> list[ChannelGroupConfig]:
    """Auto-detect channel groups from correlator key names.

    Uses regex patterns with ``(?:_|$)`` anchoring so that mode-suffixed keys
    like ``scalar_standard`` or ``nucleon_flux_action`` are matched correctly.

    Maps known channel names to physical particle groups:

    - scalar, meson_scalar -> "scalar" (meson, sigma/f0)
    - pseudoscalar, meson_pseudoscalar -> "pseudoscalar" (meson, pion)
    - vector, vector_full -> "vector" (meson, rho)
    - axial_vector, axial_full -> "axial_vector" (meson, a1)
    - nucleon, baryon, baryon_nucleon -> "nucleon" (baryon)
    - glueball, glueball_plaquette -> "glueball"
    - tensor -> "tensor"

    Args:
        correlator_keys: List of correlator key strings.
        include_multiscale: When ``False``, skip keys containing ``_scale_N``.

    Returns:
        List of auto-detected :class:`ChannelGroupConfig`.
    """
    group_keys: dict[str, list[str]] = {}
    group_types: dict[str, str] = {}

    for key in correlator_keys:
        # Optionally skip multiscale keys
        if not include_multiscale and "_scale_" in key:
            _log.debug("Skipping multiscale key %r (include_multiscale=False)", key)
            continue

        # Strip scale suffix for matching
        base = key.rsplit("_scale_", 1)[0] if "_scale_" in key else key

        # Try regex patterns (order matters: axial_vector before vector)
        group_name = None
        channel_type = None
        for pattern, gname, ctype in _CHANNEL_PATTERNS:
            if pattern.match(base):
                group_name = gname
                channel_type = ctype
                break

        if group_name is None:
            _log.warning("Unmatched correlator key %r — assigning to own group", key)
            group_name = base
            channel_type = "meson"

        group_keys.setdefault(group_name, []).append(key)
        group_types[group_name] = channel_type

    groups = []
    for gname, keys in group_keys.items():
        channel_type = group_types.get(gname, _CHANNEL_TYPES.get(gname, "meson"))
        groups.append(
            ChannelGroupConfig(
                name=gname,
                correlator_keys=sorted(keys),
                channel_type=channel_type,
            )
        )

    if groups:
        _log.info(
            "Auto-detected %d channel groups: %s",
            len(groups),
            ", ".join(f"{g.name}({len(g.correlator_keys)})" for g in groups),
        )

    return groups
