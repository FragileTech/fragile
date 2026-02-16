"""Tests for effective mass computation."""

from __future__ import annotations

import gvar
import numpy as np

from fragile.physics.mass_extraction.effective_mass import (
    compute_effective_mass,
    compute_effective_mass_for_all,
)


def _make_single_exp_gvar(mass: float, amp: float, T: int) -> np.ndarray:
    """Create a single-exponential gvar correlator."""
    t = np.arange(T)
    means = amp**2 * np.exp(-mass * t)
    errors = np.abs(means) * 0.01  # 1% errors
    return gvar.gvar(means, errors)


def test_log_ratio_single_exp():
    """Log-ratio effective mass of a pure exponential should be constant."""
    mass = 0.3
    corr = _make_single_exp_gvar(mass, 1.0, 20)
    m_eff = compute_effective_mass(corr, method="log_ratio")
    assert len(m_eff) == 19
    # All values should be close to the true mass
    for m in m_eff[:15]:  # avoid noisy tail
        assert abs(gvar.mean(m) - mass) < 0.01


def test_cosh_method():
    """Cosh effective mass computation."""
    mass = 0.3
    corr = _make_single_exp_gvar(mass, 1.0, 20)
    m_eff = compute_effective_mass(corr, method="cosh")
    assert len(m_eff) == 18  # T - 2


def test_effective_mass_for_all():
    """Compute effective masses for multiple channels."""
    data = {
        "ch1": _make_single_exp_gvar(0.3, 1.0, 20),
        "ch2": _make_single_exp_gvar(0.5, 0.8, 20),
    }
    results = compute_effective_mass_for_all(data)
    assert "ch1" in results
    assert "ch2" in results
    assert results["ch1"].datatag == "ch1"
    assert len(results["ch1"].m_eff) == 19


def test_effective_mass_short_correlator():
    """Short correlator should not crash."""
    corr = gvar.gvar([1.0], [0.1])
    m_eff = compute_effective_mass(corr, method="log_ratio")
    assert len(m_eff) == 0
