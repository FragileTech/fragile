"""Prior construction for Bayesian multi-exponential fits.

Builds gvar priors with shared energy parameters across channel groups
and independent amplitude priors per variant. Optionally seeds the
ground-state energy from ``corrfitter.fastfit``.
"""

from __future__ import annotations

from typing import Any

import gvar
import numpy as np

from .config import ChannelGroupConfig, PriorConfig


def build_prior_for_group(
    group: ChannelGroupConfig,
    data: dict[str, np.ndarray],
    available_keys: list[str],
) -> dict[str, Any]:
    """Build priors for a single channel group.

    Creates shared ``dE`` entries and per-variant amplitude entries.

    Args:
        group: Channel group configuration.
        data: gvar correlator data dict.
        available_keys: Correlator keys present in the data.

    Returns:
        Prior dict for this group.
    """
    prior_cfg = group.prior
    fit_cfg = group.fit
    nexp = fit_cfg.nexp

    # Optionally seed ground-state energy from fastfit
    ground_energy = None
    if prior_cfg.use_fastfit_seeding:
        # Try to estimate from the first available key
        for key in group.correlator_keys:
            if key in available_keys and key in data:
                ground_energy = _estimate_ground_energy(
                    data, key, fit_cfg.tmin, fit_cfg.tp
                )
                if ground_energy is not None:
                    break

    # Build dE prior
    dE = _build_dE_prior(nexp, ground_energy, prior_cfg.dE_ground, prior_cfg.dE_excited, fit_cfg.use_log_dE)

    if fit_cfg.use_log_dE:
        prior_key = f"log({group.name}.dE)"
    else:
        prior_key = f"{group.name}.dE"
    prior = {prior_key: dE}

    # Build amplitude priors per variant (source a and sink b)
    for key in group.correlator_keys:
        if key in available_keys:
            prior[f"{group.name}.{key}.a"] = _build_amplitude_prior(nexp, prior_cfg.amplitude)
            prior[f"{group.name}.{key}.b"] = _build_amplitude_prior(nexp, prior_cfg.amplitude)

    # Oscillating states if needed
    if fit_cfg.nexp_osc > 0:
        osc_dE = _build_dE_prior(
            fit_cfg.nexp_osc, None, prior_cfg.dE_ground, prior_cfg.dE_excited, fit_cfg.use_log_dE
        )
        osc_key = f"log({group.name}.dE_osc)" if fit_cfg.use_log_dE else f"{group.name}.dE_osc"
        prior[osc_key] = osc_dE

        for key in group.correlator_keys:
            if key in available_keys:
                prior[f"{group.name}.{key}.ao"] = _build_amplitude_prior(
                    fit_cfg.nexp_osc, prior_cfg.amplitude
                )
                prior[f"{group.name}.{key}.bo"] = _build_amplitude_prior(
                    fit_cfg.nexp_osc, prior_cfg.amplitude
                )

    return prior


def build_combined_prior(
    groups: list[ChannelGroupConfig],
    data: dict[str, np.ndarray],
    keys_per_group: dict[str, list[str]],
) -> dict[str, Any]:
    """Build combined prior dict for all channel groups.

    Args:
        groups: List of channel group configurations.
        data: gvar correlator data dict.
        keys_per_group: Group name -> list of correlator keys used.

    Returns:
        Combined prior dict for the simultaneous fit.
    """
    combined = {}
    for group in groups:
        avail = keys_per_group.get(group.name, [])
        group_prior = build_prior_for_group(group, data, avail)
        combined.update(group_prior)
    return combined


def _estimate_ground_energy(
    data: dict[str, np.ndarray],
    key: str,
    tmin: int,
    tp: int | None,
) -> Any | None:
    """Estimate ground-state energy using corrfitter.fastfit.

    Args:
        data: gvar correlator data dict.
        key: Correlator key to use.
        tmin: Minimum time slice.
        tp: Periodic boundary time extent.

    Returns:
        A gvar estimate of the ground-state energy, or ``None`` on failure.
    """
    try:
        import corrfitter as cf

        corr = data[key]
        tdata = range(len(corr))

        kwargs = {"data": corr, "tdata": tdata, "tmin": tmin}
        if tp is not None:
            kwargs["tp"] = tp

        ff = cf.fastfit(**kwargs)
        return ff.E
    except Exception:
        return None


def _build_dE_prior(
    nexp: int,
    ground_energy: Any | None,
    dE_ground_str: str,
    dE_excited_str: str,
    use_log: bool,
) -> np.ndarray:
    """Build the dE prior array.

    Args:
        nexp: Number of exponentials.
        ground_energy: Optional fastfit estimate for the ground state.
        dE_ground_str: Prior string for ground-state splitting.
        dE_excited_str: Prior string for excited-state splittings.
        use_log: If True, return log(dE) priors for log-normal parameterization.

    Returns:
        Array of nexp gvar values.
    """
    dE = np.empty(nexp, dtype=object)

    # Ground state
    if ground_energy is not None:
        mean = gvar.mean(ground_energy)
        sdev = max(gvar.sdev(ground_energy), abs(mean) * 0.3)
        dE[0] = gvar.gvar(abs(mean), sdev)
    else:
        dE[0] = gvar.gvar(dE_ground_str)

    # Excited states
    for i in range(1, nexp):
        dE[i] = gvar.gvar(dE_excited_str)

    if use_log:
        # Convert to log-normal: log(dE)
        log_dE = np.empty(nexp, dtype=object)
        for i in range(nexp):
            m = gvar.mean(dE[i])
            s = gvar.sdev(dE[i])
            if m <= 0:
                m = 0.5
            log_m = np.log(m)
            log_s = s / m  # relative error -> log error
            log_dE[i] = gvar.gvar(log_m, max(log_s, 0.3))
        return log_dE

    return dE


def _build_amplitude_prior(
    nexp: int,
    amplitude_str: str,
) -> np.ndarray:
    """Build amplitude prior array.

    Args:
        nexp: Number of exponentials.
        amplitude_str: gvar string for amplitude prior.

    Returns:
        Array of nexp gvar values.
    """
    return np.array([gvar.gvar(amplitude_str) for _ in range(nexp)])
