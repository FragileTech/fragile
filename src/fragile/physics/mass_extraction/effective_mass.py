"""Effective mass computation with gvar error propagation.

Provides log-ratio and cosh methods for extracting effective masses
from correlator data with automatic uncertainty tracking.
"""

from __future__ import annotations

import gvar
import numpy as np

from .results import EffectiveMassResult


def compute_effective_mass(
    corr: np.ndarray,
    dt: float = 1.0,
    method: str = "log_ratio",
) -> np.ndarray:
    """Compute effective mass from a gvar correlator array.

    Args:
        corr: 1D array of gvar values, ``C(t)`` for ``t = 0, 1, ..., T``.
        dt: Time step size.
        method: ``"log_ratio"`` or ``"cosh"``.

    Returns:
        Array of gvar effective masses. Length depends on method:
        ``log_ratio`` gives ``T-1`` values, ``cosh`` gives ``T-2`` values.
    """
    T = len(corr)
    if T < 2:
        return np.array([], dtype=object)

    if method == "log_ratio":
        m_eff = []
        for t in range(T - 1):
            if gvar.mean(corr[t]) > 0 and gvar.mean(corr[t + 1]) > 0:
                m_eff.append(gvar.log(corr[t] / corr[t + 1]) / dt)
            else:
                m_eff.append(gvar.gvar(0, 0))
        return np.array(m_eff)

    elif method == "cosh":
        if T < 3:
            return np.array([], dtype=object)
        m_eff = []
        for t in range(1, T - 1):
            denom = 2.0 * corr[t]
            numer = corr[t - 1] + corr[t + 1]
            if gvar.mean(denom) != 0:
                ratio = numer / denom
                if gvar.mean(ratio) >= 1.0:
                    m_eff.append(gvar.arccosh(ratio) / dt)
                else:
                    m_eff.append(gvar.gvar(0, 0))
            else:
                m_eff.append(gvar.gvar(0, 0))
        return np.array(m_eff)

    else:
        raise ValueError(f"Unknown effective mass method: {method!r}")


def compute_effective_mass_for_all(
    data: dict[str, np.ndarray],
    dt: float = 1.0,
    method: str = "log_ratio",
) -> dict[str, EffectiveMassResult]:
    """Compute effective masses for all channels in a gvar data dict.

    Args:
        data: Channel name -> gvar correlator array.
        dt: Time step size.
        method: ``"log_ratio"`` or ``"cosh"``.

    Returns:
        Dict mapping channel name to :class:`EffectiveMassResult`.
    """
    results = {}
    for tag, corr in data.items():
        m_eff = compute_effective_mass(corr, dt=dt, method=method)
        if method == "log_ratio":
            t_vals = np.arange(len(m_eff))
        else:  # cosh
            t_vals = np.arange(1, len(m_eff) + 1)
        results[tag] = EffectiveMassResult(
            datatag=tag,
            m_eff=m_eff,
            t_values=t_vals,
        )
    return results
