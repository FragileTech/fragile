"""Build corrfitter Corr2 models for simultaneous multi-channel fitting.

Constructs ``corrfitter.Corr2`` objects with shared energy (``dE``) parameters
across all operator variants in a channel group, and independent amplitudes
per variant.
"""

from __future__ import annotations

from typing import Any

import corrfitter as cf

from .config import ChannelFitConfig, ChannelGroupConfig


def build_corr2_model(
    datatag: str,
    group_name: str,
    fit_config: ChannelFitConfig,
    max_lag: int,
) -> cf.Corr2:
    """Build a single Corr2 model for one correlator variant.

    Args:
        datatag: Correlator key (e.g., ``"scalar"``).
        group_name: Channel group name for shared parameter naming.
        fit_config: Fitting parameters (tmin, tmax, nexp, etc.).
        max_lag: Maximum lag from the correlator data.

    Returns:
        A ``corrfitter.Corr2`` model.
    """
    tmax = fit_config.tmax if fit_config.tmax is not None else max_lag
    tmax = min(tmax, max_lag)
    tmin = fit_config.tmin
    tp = fit_config.tp

    # Shared dE across all variants in the group.
    # The model always uses the plain key; log-normal parameterization
    # is handled entirely in the prior dict (key "log({group}.dE)").
    dE_key = f"{group_name}.dE"

    # Per-variant amplitudes (separate source/sink keys)
    a_key = f"{group_name}.{datatag}.a"
    b_key = f"{group_name}.{datatag}.b"

    kwargs: dict[str, Any] = {
        "datatag": datatag,
        "tdata": range(max_lag + 1),
        "tfit": range(tmin, tmax + 1),
    }

    if tp is not None:
        kwargs["tp"] = tp

    if fit_config.nexp_osc > 0:
        # Include oscillating states as second element of tuples
        osc_dE_key = f"{group_name}.dE_osc"
        osc_a_key = f"{group_name}.{datatag}.ao"
        osc_b_key = f"{group_name}.{datatag}.bo"
        kwargs["a"] = (a_key, osc_a_key)
        kwargs["b"] = (b_key, osc_b_key)
        kwargs["dE"] = (dE_key, osc_dE_key)
        kwargs["s"] = (1.0, -1.0)
    else:
        # No oscillating states: pass plain strings
        kwargs["a"] = a_key
        kwargs["b"] = b_key
        kwargs["dE"] = dE_key

    return cf.Corr2(**kwargs)


def build_channel_group_models(
    group: ChannelGroupConfig,
    available_keys: list[str],
) -> list[cf.Corr2]:
    """Build Corr2 models for all variants in a channel group.

    Args:
        group: Channel group configuration.
        available_keys: Correlator keys present in the data.

    Returns:
        List of Corr2 models for this group.
    """
    models = []
    for key in group.correlator_keys:
        if key in available_keys:
            # Infer max_lag from available data later; use a large default
            max_lag = 80  # Will be overridden in build_all_models
            models.append(build_corr2_model(key, group.name, group.fit, max_lag))
    return models


def build_all_models(
    groups: list[ChannelGroupConfig],
    available_keys: list[str],
    max_lags: dict[str, int],
) -> list[cf.Corr2]:
    """Build all Corr2 models across all channel groups.

    Args:
        groups: List of channel group configurations.
        available_keys: All correlator keys present in the data.
        max_lags: Channel name -> maximum lag available.

    Returns:
        Combined list of Corr2 models for simultaneous fitting.
    """
    all_models = []
    for group in groups:
        for key in group.correlator_keys:
            if key in available_keys:
                ml = max_lags.get(key, 80)
                model = build_corr2_model(key, group.name, group.fit, ml)
                all_models.append(model)
    return all_models


def get_parameter_keys(
    groups: list[ChannelGroupConfig],
    available_keys: list[str],
) -> dict[str, dict]:
    """Get parameter key mapping for post-fit result extraction.

    Args:
        groups: List of channel group configurations.
        available_keys: All correlator keys present in the data.

    Returns:
        Dict mapping group name to ``{"dE_key": ..., "amplitude_keys": {datatag: key}}``.
    """
    result = {}
    for group in groups:
        dE_key = f"{group.name}.dE"

        amp_keys = {}
        for key in group.correlator_keys:
            if key in available_keys:
                amp_keys[key] = f"{group.name}.{key}.a"

        result[group.name] = {
            "dE_key": dE_key,
            "amplitude_keys": amp_keys,
        }
    return result
