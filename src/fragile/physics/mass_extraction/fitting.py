"""Fitting engine for Bayesian multi-exponential correlator analysis.

Wraps ``corrfitter.CorrFitter`` to run simultaneous or chained fits,
and provides stability scanning over ``(tmin, nexp)`` grids.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import corrfitter as cf
import numpy as np

from .config import ChannelGroupConfig, FitConfig
from .models import build_all_models
from .priors import build_combined_prior


def run_fit(
    data: dict[str, np.ndarray],
    models: list[cf.Corr2],
    prior: dict[str, Any],
    config: FitConfig | None = None,
) -> Any:
    """Run a simultaneous Bayesian fit.

    Args:
        data: gvar correlator data dict.
        models: List of Corr2 models.
        prior: Prior dict.
        config: Fitting engine parameters. ``None`` uses defaults.

    Returns:
        ``lsqfit.nonlinear_fit`` result.
    """
    if config is None:
        config = FitConfig()

    fitter = cf.CorrFitter(models=models)

    fit_kwargs: dict[str, Any] = {
        "data": data,
        "prior": prior,
        "svdcut": config.svdcut,
        "tol": config.tol,
        "maxit": config.maxit,
    }

    if config.nterm is not None:
        fit_kwargs["nterm"] = (config.nterm, config.nterm)

    if config.use_chained:
        fit = fitter.chained_lsqfit(**fit_kwargs)
    else:
        fit = fitter.lsqfit(**fit_kwargs)

    return fit


def run_stability_scan(
    data: dict[str, np.ndarray],
    groups: list[ChannelGroupConfig],
    available_keys: list[str],
    max_lags: dict[str, int],
    keys_per_group: dict[str, list[str]],
    fit_config: FitConfig | None = None,
    tmin_range: tuple[int, int] = (1, 8),
    nexp_range: tuple[int, int] = (1, 4),
) -> list[dict]:
    """Scan a grid of ``(tmin, nexp)`` values for fit stability.

    Args:
        data: gvar correlator data dict.
        groups: Channel group configurations.
        available_keys: Available correlator keys.
        max_lags: Channel name -> max lag.
        keys_per_group: Group name -> list of correlator keys.
        fit_config: Base fitting engine config.
        tmin_range: ``(tmin_min, tmin_max)`` inclusive.
        nexp_range: ``(nexp_min, nexp_max)`` inclusive.

    Returns:
        List of dicts with ``tmin``, ``nexp``, ``chi2_dof``, ``Q``,
        ``logGBF``, and per-group ``masses``.
    """
    if fit_config is None:
        fit_config = FitConfig()

    results = []
    for tmin in range(tmin_range[0], tmin_range[1] + 1):
        for nexp in range(nexp_range[0], nexp_range[1] + 1):
            # Deep-copy groups and override tmin/nexp
            scan_groups = []
            for g in groups:
                g2 = deepcopy(g)
                g2.fit.tmin = tmin
                g2.fit.nexp = nexp
                scan_groups.append(g2)

            try:
                models = build_all_models(scan_groups, available_keys, max_lags)
                if not models:
                    continue
                prior = build_combined_prior(scan_groups, data, keys_per_group)
                fit = run_fit(data, models, prior, fit_config)

                # Extract ground-state masses per group
                masses = {}
                for g2 in scan_groups:
                    dE_key = f"{g2.name}.dE"
                    try:
                        masses[g2.name] = fit.p[dE_key][0]
                    except KeyError:
                        pass

                results.append({
                    "tmin": tmin,
                    "nexp": nexp,
                    "chi2_dof": float(fit.chi2 / fit.dof) if fit.dof > 0 else 0.0,
                    "Q": float(fit.Q),
                    "logGBF": float(fit.logGBF) if hasattr(fit, "logGBF") else 0.0,
                    "masses": masses,
                })
            except Exception:
                # Skip failed fits in scan
                results.append({
                    "tmin": tmin,
                    "nexp": nexp,
                    "chi2_dof": float("nan"),
                    "Q": 0.0,
                    "logGBF": float("-inf"),
                    "masses": {},
                })

    return results
