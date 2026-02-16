"""Multi-mode GEVP mass extraction using fixed-eigenvector projection.

Implements the standard lattice QCD "fixed eigenvector" method:
  1. Solve GEVP at a single reference lag τ_diag to obtain rotation matrix V
  2. Project C(τ) onto V for all τ to get per-mode eigenvalues λ_n(τ)
  3. Extract effective masses m_eff_n(τ) = log(λ_n(τ)/λ_n(τ+1)) / dt
  4. Detect plateaus per mode to obtain ground + excited state masses

All core computations are fully vectorized PyTorch — no Python loops
over modes or lags.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class GEVPMultiModeConfig:
    """Configuration for multi-mode GEVP mass extraction."""

    tau_diag: int | None = None
    tau_diag_search_range: tuple[int, int] = (3, 8)
    plateau_min_length: int = 3
    plateau_max_slope: float = 0.3
    clamp_floor: float = 1e-12
    fit_start: int | None = None  # None = auto (tau_diag + 1)
    fit_stop: int | None = None  # None = use all available lags


@dataclass
class GEVPMassSpectrum:
    """Full multi-mode GEVP results.

    Attributes:
        mode_eigenvalues: Projected eigenvalues λ_n(τ) with shape [N_modes, L].
        effective_masses: Effective masses m_eff_n(τ) with shape [N_modes, L-1].
        plateau_masses: Best plateau mass per mode [N_modes].
        plateau_errors: Plateau mass uncertainty per mode [N_modes].
        plateau_ranges: (start, end) lag indices for each mode's plateau.
        rotation_matrix: Eigenvector matrix V from GEVP at τ_diag [M, M].
        tau_diag: Reference lag used for the GEVP solve.
        n_modes: Number of modes extracted.
        diagnostics: Extra diagnostic information.
    """

    mode_eigenvalues: Tensor
    effective_masses: Tensor
    plateau_masses: Tensor
    plateau_errors: Tensor
    plateau_ranges: list[tuple[int, int]]
    rotation_matrix: Tensor
    tau_diag: int
    n_modes: int
    diagnostics: dict[str, object] = field(default_factory=dict)
    exp_masses: Tensor | None = None  # [N_modes]
    exp_errors: Tensor | None = None  # [N_modes]
    exp_r2: Tensor | None = None  # [N_modes]


def auto_select_tau_diag(
    c_proj: Tensor,
    *,
    t0: int,
    search_range: tuple[int, int] = (3, 8),
) -> int:
    """Select τ_diag that maximizes spectral gap ratio.

    For each candidate τ in [search_range[0], search_range[1]], solve the
    GEVP C(τ)v = λ C(t0)v and compute the ratio λ_0/λ_1 (spectral gap).
    Returns the τ with the largest gap.

    Args:
        c_proj: Whitened correlator matrices [L, M, M].
        t0: Reference lag for C(t0).
        search_range: (min_tau, max_tau) to search over.

    Returns:
        Best τ_diag value.
    """
    lo, hi = int(search_range[0]), int(search_range[1])
    l_max = c_proj.shape[0]
    lo = max(lo, t0 + 1)
    hi = min(hi, l_max - 1)
    if hi < lo:
        return lo

    m = c_proj.shape[-1]
    if m < 2:
        return lo

    c0 = c_proj[t0]
    c0_sym = 0.5 * (c0 + c0.T)

    best_tau = lo
    best_gap = -1.0
    # Candidate τ values are few (typically 3-8), so a short loop is fine.
    for tau in range(lo, hi + 1):
        ct = c_proj[tau]
        ct_sym = 0.5 * (ct + ct.T)
        try:
            evals = torch.linalg.eigvalsh(torch.linalg.solve(c0_sym, ct_sym))
        except Exception:
            continue
        evals_sorted = evals.real.sort(descending=True).values
        if evals_sorted[0] > 0 and evals_sorted[1] > 0:
            gap = float((evals_sorted[0] / evals_sorted[1]).item())
        elif evals_sorted[0] > 0:
            gap = float("inf")
        else:
            continue
        if gap > best_gap:
            best_gap = gap
            best_tau = tau

    return best_tau


def solve_gevp_fixed_eigenvectors(
    c_proj: Tensor,
    *,
    t0: int,
    tau_diag: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Solve GEVP at τ_diag and return fixed eigenvectors.

    Computes V from C(τ_diag) v = λ C(t0) v using symmetric eigendecomposition
    of C(t0)^{-1/2} C(τ_diag) C(t0)^{-1/2}.

    Args:
        c_proj: Whitened correlator matrices [L, M, M].
        t0: Reference lag index.
        tau_diag: Lag at which to solve the GEVP.

    Returns:
        V: Rotation matrix [M, M], columns sorted by descending eigenvalue.
        eigenvalues: GEVP eigenvalues at τ_diag [M], sorted descending.
        norm_diag: Diagonal of V^T C(t0) V, used for normalization [M].
    """
    c0 = c_proj[t0]
    ct = c_proj[tau_diag]
    c0_sym = 0.5 * (c0 + c0.T)
    ct_sym = 0.5 * (ct + ct.T)

    # Whitening: C(t0) = U D U^T → W = U D^{-1/2}
    evals_0, evecs_0 = torch.linalg.eigh(c0_sym)
    pos_mask = evals_0 > 1e-12
    evals_pos = evals_0[pos_mask]
    evecs_pos = evecs_0[:, pos_mask]
    w = evecs_pos / torch.sqrt(evals_pos).unsqueeze(0)  # [M, K]

    # Transformed problem: eigh(W^T C(τ) W)
    ct_w = w.T @ ct_sym @ w  # [K, K]
    ct_w = 0.5 * (ct_w + ct_w.T)
    evals_tau, evecs_tau = torch.linalg.eigh(ct_w)

    # Sort descending
    order = torch.argsort(evals_tau, descending=True)
    evals_sorted = evals_tau[order].real.float()
    evecs_sorted = evecs_tau[:, order].real.float()

    # Map back to original basis: V = W @ evecs
    v = w @ evecs_sorted  # [M, N_kept]

    # Normalization diagonal: diag(V^T C(t0) V)
    norm_diag = (v.T @ c0_sym @ v).diagonal().real.float()
    norm_diag = torch.clamp_min(norm_diag.abs(), 1e-12)

    return v, evals_sorted, norm_diag


def project_all_modes(
    c_proj: Tensor,
    v: Tensor,
    norm: Tensor,
) -> Tensor:
    """Project C(τ) onto fixed eigenvectors for all modes and lags.

    Fully vectorized: computes V^T C(τ) V diagonal for all τ at once.

    Args:
        c_proj: Whitened correlator matrices [L, M, M].
        v: Rotation matrix from solve_gevp_fixed_eigenvectors [M, N_modes].
        norm: Normalization diagonal [N_modes].

    Returns:
        mode_eigenvalues: Projected eigenvalues [N_modes, L].
    """
    # c_rotated[t, n, m] = V[i,n]^T @ C(t)[i,j] @ V[j,m] → [L, N, N]
    c_rotated = torch.einsum("in,tij,jm->tnm", v, c_proj, v)  # [L, N, N]
    # Extract diagonal → [L, N]
    lambdas = c_rotated.diagonal(dim1=-2, dim2=-1)  # [L, N]
    # Normalize
    lambdas = lambdas / norm.unsqueeze(0)  # [L, N]
    # Transpose to [N, L]
    return lambdas.T.contiguous()


def compute_multimode_effective_masses(
    eigenvalues: Tensor,
    *,
    dt: float = 1.0,
    clamp_floor: float = 1e-12,
) -> Tensor:
    """Compute effective masses from eigenvalue ratios.

    m_eff[n, τ] = log(λ[n, τ] / λ[n, τ+1]) / dt

    Fully vectorized over both mode and lag dimensions.

    Args:
        eigenvalues: Mode eigenvalues [N_modes, L].
        dt: Time step.
        clamp_floor: Floor for clamping eigenvalues before log.

    Returns:
        Effective masses [N_modes, L-1].
    """
    lam_cur = eigenvalues[:, :-1]
    lam_next = eigenvalues[:, 1:]
    # Clamp to avoid log of non-positive
    ratio = torch.clamp_min(lam_cur, clamp_floor) / torch.clamp_min(lam_next, clamp_floor)
    m_eff = torch.log(ratio) / dt
    # Mark non-physical as NaN
    return torch.where(
        (lam_cur > clamp_floor) & (lam_next > clamp_floor) & torch.isfinite(m_eff),
        m_eff,
        torch.full_like(m_eff, float("nan")),
    )


def detect_plateau_per_mode(
    m_eff: Tensor,
    *,
    min_length: int = 3,
    max_slope: float = 0.3,
) -> tuple[Tensor, Tensor, list[tuple[int, int]]]:
    """Detect effective-mass plateaus for each mode.

    Uses a sliding-window approach: finds the longest contiguous region
    where |m_eff[τ+1] - m_eff[τ]| < max_slope * mean(m_eff) and
    all values are finite.

    Args:
        m_eff: Effective masses [N_modes, L-1].
        min_length: Minimum plateau length.
        max_slope: Maximum relative slope for plateau acceptance.

    Returns:
        masses: Plateau-averaged mass per mode [N_modes].
        errors: Standard deviation within plateau per mode [N_modes].
        ranges: (start, end) indices of the plateau for each mode.
    """
    n_modes, _n_lags = m_eff.shape
    device = m_eff.device
    masses = torch.full((n_modes,), float("nan"), dtype=torch.float32, device=device)
    errors = torch.full((n_modes,), float("nan"), dtype=torch.float32, device=device)
    ranges: list[tuple[int, int]] = []

    # Per-mode plateau detection (unavoidable loop — plateau bounds differ per mode)
    for mode_idx in range(n_modes):
        row = m_eff[mode_idx]
        finite_mask = torch.isfinite(row) & (row > 0)
        if finite_mask.sum() < min_length:
            ranges.append((0, 0))
            continue

        best_start, best_end = 0, 0
        best_length = 0

        # Scan for contiguous plateaus
        finite_idx = torch.nonzero(finite_mask, as_tuple=False).flatten()
        if finite_idx.numel() == 0:
            ranges.append((0, 0))
            continue

        start = int(finite_idx[0].item())
        for pos in range(len(finite_idx)):
            idx = int(finite_idx[pos].item())
            # Check if contiguous with previous
            if pos > 0 and idx != int(finite_idx[pos - 1].item()) + 1:
                start = idx

            # Check slope from start to current
            window = row[start : idx + 1]
            window_finite = window[torch.isfinite(window) & (window > 0)]
            if window_finite.numel() < 2:
                continue

            mean_val = window_finite.mean()
            if mean_val <= 0:
                continue
            diffs = (window_finite[1:] - window_finite[:-1]).abs()
            rel_slope = diffs.max() / mean_val

            if float(rel_slope.item()) <= max_slope:
                length = idx - start + 1
                if length > best_length:
                    best_length = length
                    best_start = start
                    best_end = idx + 1
            else:
                # Reset start to current position
                start = idx

        if best_length >= min_length:
            plateau = row[best_start:best_end]
            plateau_finite = plateau[torch.isfinite(plateau) & (plateau > 0)]
            if plateau_finite.numel() >= 1:
                masses[mode_idx] = plateau_finite.mean()
                if plateau_finite.numel() >= 2:
                    errors[mode_idx] = plateau_finite.std()
                else:
                    errors[mode_idx] = 0.0
            ranges.append((best_start, best_end))
        else:
            # Fallback: use all finite values
            finite_vals = row[finite_mask]
            if finite_vals.numel() >= 1:
                masses[mode_idx] = finite_vals.mean()
                errors[mode_idx] = finite_vals.std() if finite_vals.numel() >= 2 else 0.0
            ranges.append((0, 0))

    return masses, errors, ranges


def fit_exponential_per_mode(
    eigenvalues: Tensor,
    *,
    dt: float = 1.0,
    fit_start: int = 2,
    fit_stop: int | None = None,
    clamp_floor: float = 1e-12,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit exponential decay to per-mode eigenvalues via log-linear OLS.

    For each mode, fits log(λ_n(τ)) = a - m·τ using closed-form OLS.
    The mass is extracted from the slope: m_n = -slope / dt.

    Args:
        eigenvalues: Mode eigenvalues [N_modes, L].
        dt: Time step between lags.
        fit_start: First lag index to include in fit.
        fit_stop: Last lag index (exclusive). None = use all.
        clamp_floor: Floor for positive eigenvalue selection.

    Returns:
        exp_masses: Fitted masses [N_modes], NaN for modes with <3 valid points.
        exp_errors: Fit uncertainty per mode [N_modes].
        exp_r2: Coefficient of determination per mode [N_modes].
    """
    n_modes, n_lags = eigenvalues.shape
    device = eigenvalues.device

    fit_start = max(0, int(fit_start))
    if fit_stop is None:
        fit_stop = n_lags
    else:
        fit_stop = min(int(fit_stop), n_lags)

    if fit_stop <= fit_start:
        nan = torch.full((n_modes,), float("nan"), dtype=torch.float32, device=device)
        return nan, nan.clone(), nan.clone()

    # Slice to fit range
    ev_fit = eigenvalues[:, fit_start:fit_stop]  # [N, F]
    n_fit = ev_fit.shape[1]
    tau = torch.arange(fit_start, fit_start + n_fit, dtype=torch.float32, device=device)

    # Mask: only positive & finite eigenvalues
    valid = (ev_fit > clamp_floor) & torch.isfinite(ev_fit)  # [N, F]
    log_ev = torch.where(valid, torch.log(ev_fit.clamp(min=clamp_floor)), torch.zeros_like(ev_fit))

    # Masked sums for closed-form OLS per mode
    n_valid = valid.float().sum(dim=1)  # [N]
    tau_exp = tau.unsqueeze(0).expand(n_modes, -1)  # [N, F]
    mask_f = valid.float()

    sum_x = (tau_exp * mask_f).sum(dim=1)
    sum_y = (log_ev * mask_f).sum(dim=1)
    sum_xx = (tau_exp.square() * mask_f).sum(dim=1)
    sum_xy = (tau_exp * log_ev * mask_f).sum(dim=1)

    denom = n_valid * sum_xx - sum_x.square()
    # Avoid division by zero
    safe_denom = torch.where(denom.abs() > 1e-30, denom, torch.ones_like(denom))

    slope = (n_valid * sum_xy - sum_x * sum_y) / safe_denom  # [N]
    intercept = (sum_y * sum_xx - sum_x * sum_xy) / safe_denom  # [N]

    # Mass from negative slope
    exp_masses = -slope / dt

    # R² computation
    y_mean = sum_y / n_valid.clamp(min=1)
    ss_tot = ((log_ev - y_mean.unsqueeze(1)).square() * mask_f).sum(dim=1)
    y_pred = intercept.unsqueeze(1) + slope.unsqueeze(1) * tau_exp
    ss_res = ((log_ev - y_pred).square() * mask_f).sum(dim=1)
    safe_ss_tot = torch.where(ss_tot > 1e-30, ss_tot, torch.ones_like(ss_tot))
    exp_r2 = 1.0 - ss_res / safe_ss_tot

    # Error estimate from residual standard error
    dof = (n_valid - 2).clamp(min=1)
    mse = ss_res / dof
    se_slope = torch.sqrt(mse * n_valid / safe_denom.abs().clamp(min=1e-30))
    exp_errors = se_slope / dt

    # Invalidate modes with too few points
    insufficient = n_valid < 3
    nan_val = torch.tensor(float("nan"), dtype=torch.float32, device=device)
    exp_masses = torch.where(insufficient | (denom.abs() < 1e-30), nan_val, exp_masses)
    exp_errors = torch.where(insufficient | (denom.abs() < 1e-30), nan_val, exp_errors)
    exp_r2 = torch.where(insufficient | (ss_tot < 1e-30), nan_val, exp_r2)

    return exp_masses, exp_errors, exp_r2


@dataclass
class T0SweepResult:
    """Results from running GEVP extraction across multiple t0 values."""

    t0_values: list[int]
    spectra: dict[int, GEVPMassSpectrum]  # only successful runs
    consensus_masses: Tensor | None = None  # median across t0 [N_modes]
    consensus_errors: Tensor | None = None  # std across t0 [N_modes]


def extract_multimode_t0_sweep(
    c_proj: Tensor,
    *,
    t0_values: list[int],
    dt: float = 1.0,
    config: GEVPMultiModeConfig | None = None,
) -> T0SweepResult:
    """Run multi-mode GEVP extraction for multiple t0 values.

    For each t0, calls extract_multimode_gevp_masses with tau_diag=None
    (auto-selected per t0). Failed t0 values are silently skipped.
    Computes consensus masses as nanmedian/nanstd across successful runs.

    Args:
        c_proj: Whitened correlator matrices [L, M, M].
        t0_values: List of t0 values to sweep.
        dt: Time step between lags.
        config: Multi-mode configuration (tau_diag is overridden to None).

    Returns:
        T0SweepResult with per-t0 spectra and consensus.
    """
    if config is None:
        config = GEVPMultiModeConfig()

    # Override tau_diag to None for auto-selection per t0
    sweep_config = GEVPMultiModeConfig(
        tau_diag=None,
        tau_diag_search_range=config.tau_diag_search_range,
        plateau_min_length=config.plateau_min_length,
        plateau_max_slope=config.plateau_max_slope,
        clamp_floor=config.clamp_floor,
        fit_start=config.fit_start,
        fit_stop=config.fit_stop,
    )

    spectra: dict[int, GEVPMassSpectrum] = {}
    for t0_val in t0_values:
        try:
            spectrum = extract_multimode_gevp_masses(
                c_proj,
                t0=t0_val,
                dt=dt,
                config=sweep_config,
            )
            spectra[t0_val] = spectrum
        except Exception:
            continue

    if not spectra:
        return T0SweepResult(
            t0_values=list(t0_values),
            spectra=spectra,
        )

    # Compute consensus across successful t0 values
    # Find max n_modes across spectra
    max_modes = max(s.n_modes for s in spectra.values())
    device = next(iter(spectra.values())).plateau_masses.device

    # Stack plateau masses: [n_successful, max_modes], pad with NaN
    mass_stack = torch.full(
        (len(spectra), max_modes),
        float("nan"),
        dtype=torch.float32,
        device=device,
    )
    for i, s in enumerate(spectra.values()):
        n = min(s.n_modes, max_modes)
        mass_stack[i, :n] = s.plateau_masses[:n]

    consensus_masses = torch.nanmedian(mass_stack, dim=0).values
    # nanstd: manual computation
    finite = torch.isfinite(mass_stack)
    count = finite.float().sum(dim=0).clamp(min=1)
    safe = torch.where(finite, mass_stack, torch.zeros_like(mass_stack))
    mean = safe.sum(dim=0) / count
    centered = torch.where(finite, mass_stack - mean, torch.zeros_like(mass_stack))
    variance = centered.square().sum(dim=0) / count.clamp(min=1)
    consensus_errors = torch.sqrt(variance.clamp(min=0))
    consensus_errors = torch.where(
        count > 1,
        consensus_errors,
        torch.full_like(consensus_errors, float("nan")),
    )

    return T0SweepResult(
        t0_values=list(t0_values),
        spectra=spectra,
        consensus_masses=consensus_masses,
        consensus_errors=consensus_errors,
    )


def extract_multimode_gevp_masses(
    c_proj: Tensor,
    *,
    t0: int,
    dt: float = 1.0,
    config: GEVPMultiModeConfig | None = None,
) -> GEVPMassSpectrum:
    """Top-level orchestrator for multi-mode GEVP mass extraction.

    Steps:
      1. Auto-select τ_diag (or use config value)
      2. Solve GEVP at τ_diag for fixed eigenvectors
      3. Project C(τ) onto eigenvectors for all modes/lags
      4. Compute effective masses
      5. Detect plateaus per mode

    Args:
        c_proj: Whitened correlator matrices [L, M, M].
        t0: Reference lag for C(t0).
        dt: Time step between lags.
        config: Multi-mode configuration.

    Returns:
        GEVPMassSpectrum with full multi-mode results.
    """
    if config is None:
        config = GEVPMultiModeConfig()

    l_len, _m_dim, _ = c_proj.shape

    # 1. Select τ_diag
    if config.tau_diag is not None:
        tau_diag = config.tau_diag
    else:
        tau_diag = auto_select_tau_diag(
            c_proj,
            t0=t0,
            search_range=config.tau_diag_search_range,
        )
    tau_diag = max(t0 + 1, min(tau_diag, l_len - 1))

    # 2. Solve GEVP
    v, evals_diag, norm_diag = solve_gevp_fixed_eigenvectors(c_proj, t0=t0, tau_diag=tau_diag)

    n_modes = v.shape[1]

    # 3. Project all modes
    mode_eigenvalues = project_all_modes(c_proj, v, norm_diag)

    # 4. Effective masses
    m_eff = compute_multimode_effective_masses(
        mode_eigenvalues, dt=dt, clamp_floor=config.clamp_floor
    )

    # 5. Plateau detection
    plateau_masses, plateau_errors, plateau_ranges = detect_plateau_per_mode(
        m_eff,
        min_length=config.plateau_min_length,
        max_slope=config.plateau_max_slope,
    )

    # 6. Exponential decay fitting
    fit_start = config.fit_start
    if fit_start is None:
        fit_start = max(2, tau_diag + 1)
    exp_masses, exp_errors, exp_r2 = fit_exponential_per_mode(
        mode_eigenvalues,
        dt=dt,
        fit_start=fit_start,
        fit_stop=config.fit_stop,
        clamp_floor=config.clamp_floor,
    )

    return GEVPMassSpectrum(
        mode_eigenvalues=mode_eigenvalues,
        effective_masses=m_eff,
        plateau_masses=plateau_masses,
        plateau_errors=plateau_errors,
        plateau_ranges=plateau_ranges,
        rotation_matrix=v,
        tau_diag=tau_diag,
        n_modes=n_modes,
        diagnostics={
            "eigenvalues_at_tau_diag": evals_diag,
            "norm_diag": norm_diag,
            "spectral_gap": float((evals_diag[0] / evals_diag[1]).item())
            if n_modes >= 2 and evals_diag[1] > 0
            else float("inf"),
        },
        exp_masses=exp_masses,
        exp_errors=exp_errors,
        exp_r2=exp_r2,
    )


def extract_multimode_gevp_masses_bootstrap(
    basis: Tensor,
    whitener: Tensor,
    *,
    t0: int,
    dt: float,
    max_lag: int,
    use_connected: bool = True,
    n_bootstrap: int = 100,
    seed: int = 12345,
    config: GEVPMultiModeConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Bootstrap multi-mode GEVP masses with fixed V from central estimate.

    Computes the central GEVP rotation matrix V once, then applies it to
    each bootstrap replicate to get per-mode mass distributions.

    Args:
        basis: Operator series [K, T].
        whitener: Whitening matrix from _build_whitener [K_orig, K_kept].
        t0: Reference lag.
        dt: Time step.
        max_lag: Maximum lag for correlator computation.
        use_connected: Subtract mean before correlation.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed.
        config: Multi-mode configuration.

    Returns:
        bootstrap_masses: Per-bootstrap, per-mode masses [B, N_modes].
        errors: Bootstrap standard deviation per mode [N_modes].
    """
    from fragile.physics.app.qft.gevp_channels import (
        _fft_cross_correlator_lags,
        _fft_cross_correlator_lags_batched,
    )

    if config is None:
        config = GEVPMultiModeConfig()

    # Central estimate
    c_lags = _fft_cross_correlator_lags(basis, max_lag=max_lag, use_connected=use_connected)
    c_proj = torch.einsum("ki,tkj,jm->tim", whitener, c_lags, whitener)
    c_proj = 0.5 * (c_proj + c_proj.transpose(-1, -2))

    spectrum = extract_multimode_gevp_masses(c_proj, t0=t0, dt=dt, config=config)
    v = spectrum.rotation_matrix
    norm = spectrum.diagnostics.get("norm_diag")
    if not isinstance(norm, Tensor):
        norm = torch.ones(v.shape[1], dtype=v.dtype, device=v.device)

    n_modes = v.shape[1]
    _, t_len = basis.shape
    device = basis.device

    # Generate bootstrap resamples
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    idx = torch.randint(0, t_len, (n_bootstrap, t_len), generator=gen, device=device)
    sampled = torch.gather(
        basis.unsqueeze(0).expand(n_bootstrap, -1, -1),
        dim=2,
        index=idx.unsqueeze(1).expand(-1, basis.shape[0], -1),
    )

    # Batched correlator computation
    c_boot = _fft_cross_correlator_lags_batched(
        sampled, max_lag=max_lag, use_connected=use_connected
    )

    # Whiten: [B, L, K, K] → [B, L, M, M]
    c_proj_boot = torch.einsum("ki,btkj,jm->btim", whitener, c_boot, whitener)
    c_proj_boot = 0.5 * (c_proj_boot + c_proj_boot.transpose(-1, -2))

    # Project with FIXED V: V^T C_boot(t) V → diagonal → [B, L, N]
    c_rot_boot = torch.einsum("ni,btij,jm->btnm", v, c_proj_boot, v)
    lam_boot = c_rot_boot.diagonal(dim1=-2, dim2=-1)  # [B, L, N]
    lam_boot = lam_boot / norm.unsqueeze(0).unsqueeze(0)

    # Effective masses: [B, N, L-1]
    lam_mode = lam_boot.permute(0, 2, 1)  # [B, N, L]
    floor = config.clamp_floor
    ratio = torch.clamp_min(lam_mode[:, :, :-1], floor) / torch.clamp_min(
        lam_mode[:, :, 1:], floor
    )
    m_eff_boot = torch.log(ratio) / dt
    m_eff_boot = torch.where(
        (lam_mode[:, :, :-1] > floor) & (lam_mode[:, :, 1:] > floor) & torch.isfinite(m_eff_boot),
        m_eff_boot,
        torch.full_like(m_eff_boot, float("nan")),
    )

    # Extract plateau mass per bootstrap sample
    bootstrap_masses = torch.full(
        (n_bootstrap, n_modes), float("nan"), dtype=torch.float32, device=device
    )
    for b in range(n_bootstrap):
        for n in range(n_modes):
            row = m_eff_boot[b, n]
            finite = row[torch.isfinite(row) & (row > 0)]
            if finite.numel() >= 1:
                bootstrap_masses[b, n] = finite.mean()

    # Errors
    finite_mask = torch.isfinite(bootstrap_masses)
    count = finite_mask.sum(dim=0).float().clamp(min=1)
    safe = torch.where(finite_mask, bootstrap_masses, torch.zeros_like(bootstrap_masses))
    mean_mass = safe.sum(dim=0) / count
    centered = torch.where(
        finite_mask, bootstrap_masses - mean_mass, torch.zeros_like(bootstrap_masses)
    )
    errors = torch.sqrt((centered.square().sum(dim=0) / count).clamp(min=0))
    errors = torch.where(count > 1, errors, torch.full_like(errors, float("nan")))

    return bootstrap_masses, errors
