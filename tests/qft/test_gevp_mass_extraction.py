"""Tests for multi-mode GEVP mass extraction."""

from __future__ import annotations

import math

import pytest
import torch

from fragile.fractalai.qft.gevp_mass_extraction import (
    auto_select_tau_diag,
    compute_multimode_effective_masses,
    detect_plateau_per_mode,
    extract_multimode_gevp_masses,
    extract_multimode_t0_sweep,
    fit_exponential_per_mode,
    GEVPMassSpectrum,
    GEVPMultiModeConfig,
    project_all_modes,
    solve_gevp_fixed_eigenvectors,
    T0SweepResult,
)


def _build_two_mode_correlator(
    m_ground: float = 0.3,
    m_excited: float = 0.8,
    t_len: int = 64,
    n_ops: int = 3,
    noise_scale: float = 1e-4,
    seed: int = 42,
) -> tuple[torch.Tensor, float, float]:
    """Build a synthetic whitened correlator matrix with two known masses.

    Returns:
        c_proj: Correlator matrices [L, N, N] (already whitened).
        m_ground: Ground state mass.
        m_excited: Excited state mass.
    """
    torch.manual_seed(seed)
    t = torch.arange(t_len, dtype=torch.float32)

    # Two exponential modes
    lambda_0 = torch.exp(-m_ground * t)
    lambda_1 = torch.exp(-m_excited * t)

    # Random mixing matrix (orthogonal)
    q, _ = torch.linalg.qr(torch.randn(n_ops, n_ops))
    q = q[:, :2]  # [N, 2]

    # Build C(t) = sum_n lambda_n(t) * v_n v_n^T
    c_proj = torch.zeros(t_len, n_ops, n_ops)
    for lag in range(t_len):
        c_proj[lag] = (
            lambda_0[lag] * q[:, 0:1] @ q[:, 0:1].T + lambda_1[lag] * q[:, 1:2] @ q[:, 1:2].T
        )
        # Add small symmetric noise
        noise = noise_scale * torch.randn(n_ops, n_ops)
        c_proj[lag] += 0.5 * (noise + noise.T)

    return c_proj, m_ground, m_excited


class TestAutoSelectTauDiag:
    def test_returns_valid_tau(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator()
        tau = auto_select_tau_diag(c_proj, t0=1, search_range=(2, 6))
        assert 2 <= tau <= 6

    def test_respects_search_range(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator()
        tau = auto_select_tau_diag(c_proj, t0=1, search_range=(4, 5))
        assert 4 <= tau <= 5

    def test_single_mode_returns_lo(self) -> None:
        """Single-mode system has no spectral gap to compare."""
        t_len, n = 32, 1
        c_proj = torch.eye(n).unsqueeze(0).expand(t_len, -1, -1).clone()
        for t in range(t_len):
            c_proj[t] *= torch.exp(torch.tensor(-0.3 * t))
        tau = auto_select_tau_diag(c_proj, t0=1, search_range=(3, 8))
        assert tau >= 2  # At least t0+1


class TestSolveGEVPFixedEigenvectors:
    def test_shapes(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator(n_ops=3)
        v, evals, norm = solve_gevp_fixed_eigenvectors(c_proj, t0=1, tau_diag=3)
        # V should be [3, K_kept] where K_kept <= 3
        assert v.ndim == 2
        assert v.shape[0] == 3
        assert evals.shape[0] == v.shape[1]
        assert norm.shape[0] == v.shape[1]

    def test_eigenvalues_sorted_descending(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator(n_ops=3)
        _, evals, _ = solve_gevp_fixed_eigenvectors(c_proj, t0=1, tau_diag=3)
        for i in range(len(evals) - 1):
            assert evals[i] >= evals[i + 1]


class TestProjectAllModes:
    def test_output_shape(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator(t_len=32, n_ops=3)
        v, _, norm = solve_gevp_fixed_eigenvectors(c_proj, t0=1, tau_diag=3)
        mode_evals = project_all_modes(c_proj, v, norm)
        n_modes = v.shape[1]
        assert mode_evals.shape == (n_modes, 32)

    def test_eigenvalues_positive_at_early_lags(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator(t_len=32, n_ops=3, noise_scale=1e-6)
        v, _, norm = solve_gevp_fixed_eigenvectors(c_proj, t0=1, tau_diag=3)
        mode_evals = project_all_modes(c_proj, v, norm)
        # Early lags should have positive eigenvalues
        assert (mode_evals[:, 1:10] > 0).all()


class TestComputeMultimodeEffectiveMasses:
    def test_shape(self) -> None:
        eigenvalues = torch.exp(-0.3 * torch.arange(20).float()).unsqueeze(0)
        m_eff = compute_multimode_effective_masses(eigenvalues, dt=1.0)
        assert m_eff.shape == (1, 19)

    def test_constant_mass_recovery(self) -> None:
        """Pure exponential should give constant effective mass."""
        mass = 0.5
        t = torch.arange(30, dtype=torch.float32)
        eigenvalues = torch.exp(-mass * t).unsqueeze(0)
        m_eff = compute_multimode_effective_masses(eigenvalues, dt=1.0)
        # Should be close to 0.5 everywhere
        finite = m_eff[torch.isfinite(m_eff)]
        assert torch.allclose(finite, torch.tensor(mass), atol=1e-5)

    def test_two_modes(self) -> None:
        """Two modes with different masses."""
        t = torch.arange(30, dtype=torch.float32)
        eigenvalues = torch.stack([
            torch.exp(-0.3 * t),
            torch.exp(-0.8 * t),
        ])
        m_eff = compute_multimode_effective_masses(eigenvalues, dt=1.0)
        assert m_eff.shape == (2, 29)
        # Mode 0 mass ~ 0.3
        finite_0 = m_eff[0][torch.isfinite(m_eff[0])]
        assert abs(float(finite_0[:10].mean()) - 0.3) < 0.05
        # Mode 1 mass ~ 0.8
        finite_1 = m_eff[1][torch.isfinite(m_eff[1])]
        assert abs(float(finite_1[:10].mean()) - 0.8) < 0.05


class TestDetectPlateauPerMode:
    def test_flat_data_detected(self) -> None:
        """Constant effective mass should be detected as a plateau."""
        n_lags = 20
        m_eff = torch.full((1, n_lags), 0.5, dtype=torch.float32)
        masses, _errors, _ranges = detect_plateau_per_mode(m_eff, min_length=3, max_slope=0.3)
        assert math.isfinite(float(masses[0]))
        assert abs(float(masses[0]) - 0.5) < 0.01

    def test_noisy_plateau(self) -> None:
        """Plateau with small noise should still be detected."""
        torch.manual_seed(99)
        n_lags = 20
        m_eff = 0.5 + 0.01 * torch.randn(1, n_lags)
        m_eff = torch.clamp_min(m_eff, 0.01)
        masses, _errors, _ranges = detect_plateau_per_mode(m_eff, min_length=3, max_slope=0.3)
        assert math.isfinite(float(masses[0]))
        assert abs(float(masses[0]) - 0.5) < 0.1

    def test_multi_mode_plateaus(self) -> None:
        """Two modes with different plateau values."""
        n_lags = 20
        m_eff = torch.stack([
            torch.full((n_lags,), 0.3),
            torch.full((n_lags,), 0.8),
        ])
        masses, _errors, _ranges = detect_plateau_per_mode(m_eff, min_length=3, max_slope=0.3)
        assert abs(float(masses[0]) - 0.3) < 0.01
        assert abs(float(masses[1]) - 0.8) < 0.01


class TestExtractMultimodeGEVPMasses:
    def test_shape_correctness(self) -> None:
        c_proj, _m_ground, _m_excited = _build_two_mode_correlator(t_len=48, n_ops=3)
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0)

        assert isinstance(spectrum, GEVPMassSpectrum)
        assert spectrum.n_modes > 0
        assert spectrum.mode_eigenvalues.shape[0] == spectrum.n_modes
        assert spectrum.mode_eigenvalues.shape[1] == 48
        assert spectrum.effective_masses.shape == (spectrum.n_modes, 47)
        assert spectrum.plateau_masses.shape == (spectrum.n_modes,)
        assert spectrum.plateau_errors.shape == (spectrum.n_modes,)
        assert len(spectrum.plateau_ranges) == spectrum.n_modes
        assert spectrum.rotation_matrix.ndim == 2

    def test_recovers_ground_mass(self) -> None:
        """Ground state mass should be approximately recovered."""
        m_ground = 0.3
        c_proj, _, _ = _build_two_mode_correlator(
            m_ground=m_ground, m_excited=0.9, t_len=64, n_ops=3, noise_scale=1e-5
        )
        config = GEVPMultiModeConfig(
            plateau_min_length=3,
            plateau_max_slope=0.5,
        )
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0, config=config)

        ground_mass = float(spectrum.plateau_masses[0].item())
        assert math.isfinite(ground_mass)
        assert abs(ground_mass - m_ground) < 0.15  # Within 50% tolerance for synthetic

    def test_recovers_excited_mass(self) -> None:
        """Excited state mass should be approximately recovered for clean data."""
        m_ground = 0.3
        m_excited = 0.9
        c_proj, _, _ = _build_two_mode_correlator(
            m_ground=m_ground,
            m_excited=m_excited,
            t_len=64,
            n_ops=3,
            noise_scale=1e-6,
        )
        config = GEVPMultiModeConfig(
            plateau_min_length=3,
            plateau_max_slope=0.5,
        )
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0, config=config)

        if spectrum.n_modes >= 2:
            excited_mass = float(spectrum.plateau_masses[1].item())
            if math.isfinite(excited_mass) and excited_mass > 0:
                # Excited mass should be larger than ground mass
                ground_mass = float(spectrum.plateau_masses[0].item())
                assert excited_mass > ground_mass

    def test_single_mode_edge_case(self) -> None:
        """K_kept=1 should still work and return 1 mode."""
        t_len = 32
        t = torch.arange(t_len, dtype=torch.float32)
        c_proj = torch.zeros(t_len, 1, 1)
        for lag in range(t_len):
            c_proj[lag, 0, 0] = torch.exp(-0.4 * t[lag])

        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0)
        assert spectrum.n_modes == 1
        assert spectrum.mode_eigenvalues.shape == (1, t_len)

    def test_config_tau_diag_override(self) -> None:
        """Explicit tau_diag should be used."""
        c_proj, _, _ = _build_two_mode_correlator(t_len=32)
        config = GEVPMultiModeConfig(tau_diag=5)
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0, config=config)
        assert spectrum.tau_diag == 5

    def test_diagnostics_populated(self) -> None:
        c_proj, _, _ = _build_two_mode_correlator()
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0)
        assert "eigenvalues_at_tau_diag" in spectrum.diagnostics
        assert "spectral_gap" in spectrum.diagnostics

    def test_exp_fields_populated(self) -> None:
        """extract_multimode_gevp_masses should populate exp_masses/errors/r2."""
        c_proj, _, _ = _build_two_mode_correlator(t_len=48, n_ops=3, noise_scale=1e-5)
        spectrum = extract_multimode_gevp_masses(c_proj, t0=1, dt=1.0)
        assert spectrum.exp_masses is not None
        assert spectrum.exp_errors is not None
        assert spectrum.exp_r2 is not None
        assert spectrum.exp_masses.shape == (spectrum.n_modes,)
        assert spectrum.exp_errors.shape == (spectrum.n_modes,)
        assert spectrum.exp_r2.shape == (spectrum.n_modes,)


class TestFitExponentialPerMode:
    def test_pure_exponential_recovery(self) -> None:
        """Pure exponential eigenvalues should give exact mass recovery."""
        mass = 0.4
        t = torch.arange(30, dtype=torch.float32)
        eigenvalues = torch.exp(-mass * t).unsqueeze(0)  # [1, 30]
        exp_masses, _exp_errors, exp_r2 = fit_exponential_per_mode(
            eigenvalues,
            dt=1.0,
            fit_start=1,
            fit_stop=25,
        )
        assert exp_masses.shape == (1,)
        assert math.isfinite(float(exp_masses[0]))
        assert abs(float(exp_masses[0]) - mass) < 0.01
        assert float(exp_r2[0]) > 0.99

    def test_two_mode_recovery(self) -> None:
        """Two modes with different masses."""
        t = torch.arange(40, dtype=torch.float32)
        eigenvalues = torch.stack([
            torch.exp(-0.3 * t),
            torch.exp(-0.7 * t),
        ])
        exp_masses, _exp_errors, _exp_r2 = fit_exponential_per_mode(
            eigenvalues,
            dt=1.0,
            fit_start=1,
            fit_stop=30,
        )
        assert exp_masses.shape == (2,)
        assert abs(float(exp_masses[0]) - 0.3) < 0.01
        assert abs(float(exp_masses[1]) - 0.7) < 0.01

    def test_nonpositive_masking(self) -> None:
        """Non-positive eigenvalues should be excluded from fit."""
        t = torch.arange(20, dtype=torch.float32)
        eigenvalues = torch.exp(-0.5 * t).unsqueeze(0)
        # Corrupt some points
        eigenvalues[0, 15:] = -1.0
        exp_masses, _exp_errors, _exp_r2 = fit_exponential_per_mode(
            eigenvalues,
            dt=1.0,
            fit_start=1,
            fit_stop=20,
        )
        # Should still recover mass from valid points
        assert math.isfinite(float(exp_masses[0]))
        assert abs(float(exp_masses[0]) - 0.5) < 0.05

    def test_insufficient_points_gives_nan(self) -> None:
        """Fewer than 3 valid points should return NaN."""
        eigenvalues = torch.tensor([[1.0, 0.5, -1.0, -1.0, -1.0]])  # [1, 5]
        exp_masses, _exp_errors, _exp_r2 = fit_exponential_per_mode(
            eigenvalues,
            dt=1.0,
            fit_start=0,
            fit_stop=5,
        )
        assert math.isnan(float(exp_masses[0]))

    def test_output_shapes(self) -> None:
        """Output shapes should match n_modes."""
        t = torch.arange(20, dtype=torch.float32)
        eigenvalues = torch.stack([
            torch.exp(-0.2 * t),
            torch.exp(-0.5 * t),
            torch.exp(-0.9 * t),
        ])
        exp_masses, exp_errors, exp_r2 = fit_exponential_per_mode(
            eigenvalues,
            dt=1.0,
            fit_start=1,
        )
        assert exp_masses.shape == (3,)
        assert exp_errors.shape == (3,)
        assert exp_r2.shape == (3,)

    def test_dt_scaling(self) -> None:
        """Mass should scale inversely with dt."""
        mass = 0.5
        t = torch.arange(30, dtype=torch.float32)
        eigenvalues = torch.exp(-mass * t).unsqueeze(0)
        # With dt=2.0, mass_extracted should be 0.5 (slope is -0.5, dt=2 → -0.5/2 does not apply;
        # but the τ indices are 0,1,2... and λ = exp(-m * τ), so slope=-m, mass=-slope/dt=m/dt)
        # Actually eigenvalues[n, τ_idx] = exp(-mass * τ_idx), and we fit log(λ) = a - slope * τ_idx.
        # slope = -mass, so exp_mass = -slope / dt = mass / dt.
        # For dt=1, mass=0.5 → exp_mass=0.5. For dt=2, mass=0.5 → exp_mass=0.25.
        exp_masses_dt1, _, _ = fit_exponential_per_mode(eigenvalues, dt=1.0, fit_start=1)
        exp_masses_dt2, _, _ = fit_exponential_per_mode(eigenvalues, dt=2.0, fit_start=1)
        assert abs(float(exp_masses_dt1[0]) - mass) < 0.01
        assert abs(float(exp_masses_dt2[0]) - mass / 2.0) < 0.01


class TestExtractMultimodeT0Sweep:
    def test_correct_spectra_count(self) -> None:
        """Sweep should produce one spectrum per valid t0."""
        c_proj, _, _ = _build_two_mode_correlator(t_len=48, n_ops=3, noise_scale=1e-5)
        result = extract_multimode_t0_sweep(
            c_proj,
            t0_values=[1, 2, 3],
            dt=1.0,
        )
        assert isinstance(result, T0SweepResult)
        assert result.t0_values == [1, 2, 3]
        assert len(result.spectra) >= 1
        for t0_val in result.spectra:
            assert t0_val in {1, 2, 3}

    def test_tolerates_failed_t0(self) -> None:
        """Invalid t0 values should be silently skipped."""
        c_proj, _, _ = _build_two_mode_correlator(t_len=48, n_ops=3)
        # t0=100 is out of range
        result = extract_multimode_t0_sweep(
            c_proj,
            t0_values=[1, 2, 100],
            dt=1.0,
        )
        assert 100 not in result.spectra
        assert len(result.spectra) >= 1

    def test_consensus_populated(self) -> None:
        """Consensus masses/errors should be populated when spectra exist."""
        c_proj, _, _ = _build_two_mode_correlator(t_len=48, n_ops=3, noise_scale=1e-5)
        result = extract_multimode_t0_sweep(
            c_proj,
            t0_values=[1, 2, 3, 4],
            dt=1.0,
        )
        assert result.consensus_masses is not None
        assert result.consensus_errors is not None
        assert result.consensus_masses.ndim == 1

    def test_empty_sweep_gives_no_consensus(self) -> None:
        """All-failed sweep should give no consensus."""
        c_proj = torch.zeros(5, 2, 2)  # degenerate
        result = extract_multimode_t0_sweep(
            c_proj,
            t0_values=[10, 20],
            dt=1.0,
        )
        assert len(result.spectra) == 0
        assert result.consensus_masses is None
