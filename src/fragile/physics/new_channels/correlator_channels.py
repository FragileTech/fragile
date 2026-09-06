"""Correlator computation and mass fitting from operator time series.

This module computes two-point correlators and extracts effective masses
from operator time series. Time series preprocessing is handled by the
aggregation module.

Workflow:
    1. Preprocessing: aggregation.py generates operator time series
    2. Correlator: compute_correlator_fft() via FFT
    3. Mass extraction: ConvolutionalAICExtractor with AIC weighting
    4. Results: ChannelCorrelatorResult with mass, errors, diagnostics

Class Hierarchy:
    ChannelCorrelator (ABC)
    ├── BilinearChannelCorrelator (ABC)
    │   ├── ScalarChannel        - Identity projection
    │   ├── PseudoscalarChannel  - γ₅ diagonal projection
    │   ├── VectorChannel        - γ_μ sum projection
    │   ├── AxialVectorChannel   - γ₅γ_μ projection
    │   └── TensorChannel        - σ_μν antisymmetric
    ├── TrilinearChannelCorrelator
    │   └── NucleonChannel       - 3×3 determinant
    └── GaugeChannelCorrelator
        └── GlueballChannel      - ||force||² norm

Main entry points:
    - compute_all_channels(history, config): Compute all particle channels
    - ScalarChannel(history, config).compute(): Single channel analysis

For custom time series analysis without RunHistory:
    - Use compute_correlator_fft() directly with your series
    - Use ConvolutionalAICExtractor for mass fitting

Usage:
    from fragile.fractalai.qft.correlator_channels import (
        ChannelConfig,
        compute_all_channels,
        ScalarChannel,
        PseudoscalarChannel,
    )

    # Compute all channels
    config = ChannelConfig()
    results = compute_all_channels(history, config=config)

    # Or compute a single channel
    scalar = ScalarChannel(history, config)
    result = scalar.compute()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F

from fragile.physics.qft_utils import _fft_correlator_batched
from fragile.physics.qft_utils.helpers import recorded_time_step


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


# =============================================================================
# Configuration and Result Dataclasses
# =============================================================================


@dataclass
class ChannelConfig:
    """Configuration for aggregation (operator computation).

    This config is used for preprocessing and operator computation only.
    For correlator analysis configuration, use CorrelatorConfig.

    Neighbor Method:
        The neighbor_method parameter controls how neighbor topology is computed:
        - "auto" (default): Auto-detect best available (recorded -> companions)
        - "recorded": Explicitly use pre-computed neighbor_edges from simulation
        - "companions": Use companion walker indices

        Using "auto" provides optimal performance by prioritizing pre-computed data
        when available, with automatic fallback to slower methods as needed.
    """

    # Time parameters
    warmup_fraction: float = 0.1
    end_fraction: float = 1.0

    # Time axis selection (Monte Carlo vs Euclidean)
    time_axis: str = (
        "mc"  # "mc" (Monte Carlo timesteps) or "euclidean" (spatial dimension as time)
    )
    euclidean_time_dim: int = 3  # Which spatial dimension to use as Euclidean time (0-indexed)
    euclidean_time_bins: int = 50  # Number of time bins for Euclidean time analysis
    euclidean_time_range: tuple[float, float] | None = None  # (t_min, t_max) or None for auto

    # Color state parameters
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None

    # Neighbor selection
    neighbor_method: str = "auto"  # Auto-detect: recorded -> companions
    neighbor_k: int = 100

    # Edge weight mode for operator averaging (from pre-computed scutoid weights)
    edge_weight_mode: str = "uniform"
    # Options: "uniform", "inverse_distance", "inverse_volume",
    #   "inverse_riemannian_volume", "inverse_riemannian_distance",
    #   "kernel", "riemannian_kernel"


@dataclass
class CorrelatorConfig:
    """Configuration for correlator analysis (mass fitting).

    Separated from ChannelConfig which is for aggregation.
    """

    # Correlator computation
    max_lag: int = 80
    use_connected: bool = True

    # AIC fitting parameters
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")

    # Linear fit parameters (legacy)
    fit_mode: str = "aic"  # "aic", "linear", "linear_abs"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2

    # Bootstrap error estimation
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class ChannelCorrelatorResult:
    """Result of computing correlator for a single channel."""

    channel_name: str
    correlator: Tensor  # C(t) values [max_lag+1]
    correlator_err: Tensor | None  # Bootstrap errors
    effective_mass: Tensor  # m_eff(t) [max_lag]
    mass_fit: dict[str, Any]  # AIC-weighted mass extraction
    series: Tensor  # Raw operator time series [T]
    n_samples: int  # Number of time samples
    dt: float  # Time step
    # Per-window data for visualization
    window_masses: Tensor | None = None  # [num_widths, max_positions]
    window_aic: Tensor | None = None  # [num_widths, max_positions]
    window_widths: list[int] | None = None  # List of window widths used
    window_r2: Tensor | None = None  # [num_widths, max_positions]


# =============================================================================
# Convolutional AIC Mass Extractor
# =============================================================================


class ConvolutionalAICExtractor:
    """Extract a mass from every fit window of ``log C(t)`` at once.

    For each window width ``W`` and start ``t0`` a weighted straight line is
    fitted to ``log C(t)`` with weights ``1/sigma_t^2``.  All sufficient
    statistics are convolution sums, so every window is processed in one pass.

    Window quality is compared with the information criterion used for fit
    range averaging in lattice spectroscopy,

        AIC_i = chi2_i + 2 k + 2 N_cut,

    where ``k = 2`` parameters and ``N_cut`` is the number of valid points left
    out of window ``i``.  Without the ``N_cut`` term every window whose chi2 is
    consistent with its size receives the same weight, so pure-noise windows at
    large ``t`` dominate the average.  Sums are accumulated in float64; the
    naive float32 evaluation loses every digit of chi2 once ``|log C| > 10``.
    Points with relative error above ``max_log_error`` are excluded before any
    window is formed: below that signal-to-noise the logarithm of a noisy
    correlator is biased, and a wide window over such a tail would otherwise be
    preferred by the excluded-point penalty.
    """

    def __init__(
        self,
        window_widths: list[int] | None = None,
        min_mass: float = 0.0,
        max_mass: float = float("inf"),
        penalize_excluded_points: bool = True,
        max_log_error: float | None = 0.5,
    ):
        self.window_widths = window_widths or list(range(5, 51))
        self.min_mass = min_mass
        self.max_mass = max_mass
        self.penalize_excluded_points = penalize_excluded_points
        # Points whose relative error exceeds this are not resolved from zero;
        # log C is then biased and non-Gaussian, so they never enter a window.
        self.max_log_error = max_log_error

    @staticmethod
    def _conv(x: Tensor, kernel: Tensor) -> Tensor:
        return F.conv1d(x, kernel)

    def _fit_single_width_full(
        self,
        log_corr: Tensor,
        log_err: Tensor,
        W: int,
    ) -> dict[str, Tensor]:
        """Weighted least squares for all windows of width ``W``.

        Args:
            log_corr: Log correlator ``[1, 1, T]``; non-finite entries are invalid.
            log_err: Log-space errors ``[1, 1, T]``; non-positive or non-finite
                entries mark invalid points.
            W: Window width.

        Returns:
            Dict of ``[1, 1, T - W + 1]`` tensors: ``mass``, ``aic``, ``r2``,
            ``chi2``, ``slope_var``, ``valid``.
        """
        device = log_corr.device
        y = log_corr.to(torch.float64)
        err = log_err.to(torch.float64)
        point_valid = torch.isfinite(y) & torch.isfinite(err) & (err > 0)
        if self.max_log_error is not None and math.isfinite(self.max_log_error):
            point_valid &= err <= self.max_log_error
        w = torch.where(point_valid, 1.0 / err.clamp_min(1e-300) ** 2, torch.zeros_like(err))
        y = torch.where(point_valid, y, torch.zeros_like(y))

        t_vec = torch.arange(W, device=device, dtype=torch.float64)
        k_1 = torch.ones(1, 1, W, device=device, dtype=torch.float64)
        k_t = t_vec.view(1, 1, W)
        k_tt = (t_vec**2).view(1, 1, W)

        n_valid = self._conv(point_valid.to(torch.float64), k_1)
        S_w = self._conv(w, k_1)
        S_wt = self._conv(w, k_t)
        S_wtt = self._conv(w, k_tt)
        S_wy = self._conv(w * y, k_1)
        S_wty = self._conv(w * y, k_t)
        S_wyy = self._conv(w * y * y, k_1)

        det = S_w * S_wtt - S_wt**2
        window_valid = (n_valid == W) & (det > 0)
        safe_det = torch.where(window_valid, det, torch.ones_like(det))
        safe_S_w = torch.where(window_valid, S_w, torch.ones_like(S_w))

        slope = (S_w * S_wty - S_wt * S_wy) / safe_det
        intercept = (S_wy - slope * S_wt) / safe_S_w
        mass = -slope

        # Weighted residual sum of squares via the normal-equation identity.
        chi2 = (S_wyy - intercept * S_wy - slope * S_wty).clamp_min(0.0)
        chi2_tot = S_wyy - S_wy**2 / safe_S_w
        r2 = torch.where(
            chi2_tot > 0,
            1.0 - chi2 / chi2_tot.clamp_min(1e-300),
            torch.full_like(chi2, float("nan")),
        )
        slope_var = S_w / safe_det

        n_total_valid = point_valid.to(torch.float64).sum()
        n_cut = (n_total_valid - W).clamp_min(0.0) if self.penalize_excluded_points else 0.0
        aic = chi2 + 4.0 + 2.0 * n_cut

        invalid = (
            ~window_valid | (mass < self.min_mass) | (mass > self.max_mass) | ~torch.isfinite(mass)
        )
        aic = torch.where(invalid, torch.full_like(aic, float("inf")), aic)
        r2 = torch.where(invalid, torch.full_like(r2, float("nan")), r2)
        mass = torch.where(invalid, torch.full_like(mass, float("nan")), mass)
        slope_var = torch.where(invalid, torch.full_like(slope_var, float("nan")), slope_var)
        return {
            "mass": mass,
            "aic": aic,
            "r2": r2,
            "chi2": chi2,
            "slope_var": slope_var,
            "valid": ~invalid,
        }

    def fit_single_width(
        self,
        log_corr: Tensor,
        log_err: Tensor,
        W: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(mass, aic, r2)`` for all windows of width ``W``.

        Shapes follow ``conv1d``: ``[1, 1, T - W + 1]``.  Invalid windows carry
        ``nan`` mass, ``inf`` AIC and ``nan`` R².
        """
        out = self._fit_single_width_full(log_corr, log_err, W)
        return out["mass"], out["aic"], out["r2"]

    def fit_all_widths(
        self,
        log_corr: Tensor,
        log_err: Tensor,
    ) -> dict[str, Any]:
        """Fit every window of every width and form the AIC-weighted average.

        Args:
            log_corr: Log correlator ``[T]``.
            log_err: Log-space errors ``[T]``.

        Returns:
            Dict with ``mass`` (AIC-weighted), ``mass_error`` (statistical
            error of the weighted average combined with the window spread),
            ``window_spread``, ``statistical_error``, ``r_squared``,
            ``n_valid_windows``, ``best_window`` and the per-window tensors.
        """
        T = int(log_corr.shape[0])
        log_corr = log_corr.reshape(1, 1, -1)
        log_err = log_err.reshape(1, 1, -1)

        masses, aics, r2s, variances, valid_widths = [], [], [], [], []
        for W in self.window_widths:
            if W > T or W < 3:
                continue
            out = self._fit_single_width_full(log_corr, log_err, W)
            pad_right = T - out["mass"].shape[-1]
            masses.append(F.pad(out["mass"], (0, pad_right), value=float("nan")))
            aics.append(F.pad(out["aic"], (0, pad_right), value=float("inf")))
            r2s.append(F.pad(out["r2"], (0, pad_right), value=float("nan")))
            variances.append(F.pad(out["slope_var"], (0, pad_right), value=float("nan")))
            valid_widths.append(W)

        if not masses:
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "window_spread": float("inf"),
                "statistical_error": float("inf"),
                "r_squared": float("nan"),
                "n_valid_windows": 0,
                "window_masses": None,
                "window_aic": None,
                "window_widths": [],
                "window_r2": None,
                "window_mass_variance": None,
            }

        mass_stack = torch.cat(masses, dim=0).squeeze(1)  # [num_widths, T]
        aic_stack = torch.cat(aics, dim=0).squeeze(1)
        r2_stack = torch.cat(r2s, dim=0).squeeze(1)
        var_stack = torch.cat(variances, dim=0).squeeze(1)

        flat_mass = mass_stack.flatten()
        flat_aic = aic_stack.flatten()
        flat_r2 = r2_stack.flatten()
        flat_var = var_stack.flatten()
        valid = torch.isfinite(flat_aic) & torch.isfinite(flat_mass) & (flat_mass > 0)

        if not valid.any():
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "window_spread": float("inf"),
                "statistical_error": float("inf"),
                "r_squared": float("nan"),
                "n_valid_windows": 0,
                "window_masses": mass_stack,
                "window_aic": aic_stack,
                "window_widths": valid_widths,
                "window_r2": r2_stack,
                "window_mass_variance": var_stack,
            }

        aic_valid = flat_aic[valid]
        mass_valid = flat_mass[valid]
        var_valid = flat_var[valid]
        weights = torch.exp(-0.5 * (aic_valid - aic_valid.min()))
        weights = weights / weights.sum()

        mass_final = float((weights * mass_valid).sum().item())
        window_spread = float((weights * (mass_valid - mass_final) ** 2).sum().sqrt().item())
        stat_var = torch.where(torch.isfinite(var_valid), var_valid, torch.zeros_like(var_valid))
        statistical_error = float((weights * stat_var).sum().sqrt().item())
        mass_error = float(math.sqrt(statistical_error**2 + window_spread**2))

        r2_final = float("nan")
        r2_ok = torch.isfinite(flat_r2[valid])
        if r2_ok.any():
            w_r2 = weights[r2_ok] / weights[r2_ok].sum()
            r2_final = float((w_r2 * flat_r2[valid][r2_ok]).sum().item())

        best_flat_idx = int(
            torch.where(valid, flat_aic, torch.full_like(flat_aic, float("inf"))).argmin().item()
        )
        best_w_idx, best_t_idx = best_flat_idx // T, best_flat_idx % T
        best_var = float(flat_var[best_flat_idx].item())
        best_r2 = float(flat_r2[best_flat_idx].item())

        return {
            "mass": mass_final,
            "mass_error": mass_error,
            "window_spread": window_spread,
            "statistical_error": statistical_error,
            "r_squared": r2_final,
            "n_valid_windows": int(valid.sum().item()),
            "best_window": {
                "width": valid_widths[best_w_idx] if best_w_idx < len(valid_widths) else 0,
                "t_start": best_t_idx,
                "mass": float(flat_mass[best_flat_idx].item()),
                "mass_error": math.sqrt(best_var) if math.isfinite(best_var) else float("nan"),
                "aic": float(flat_aic[best_flat_idx].item()),
                "r2": best_r2 if math.isfinite(best_r2) else float("nan"),
            },
            "window_masses": mass_stack,
            "window_aic": aic_stack,
            "window_widths": valid_widths,
            "window_r2": r2_stack,
            "window_mass_variance": var_stack,
        }


# =============================================================================
# FFT-Based Correlator Computation
# =============================================================================


def _fft_correlator_single(
    series: Tensor,
    max_lag: int,
    use_connected: bool = True,
) -> Tensor:
    """Compute time correlator using FFT (internal helper).

    Args:
        series: Operator time series [T].
        max_lag: Maximum lag to compute.
        use_connected: Subtract mean (connected correlator).

    Returns:
        Correlator C(t) for t=0 to max_lag [max_lag+1].
    """
    return _fft_correlator_batched(series.unsqueeze(0), max_lag, use_connected).squeeze(0)


def bootstrap_correlator_error(
    series: Tensor,
    max_lag: int,
    n_bootstrap: int = 100,
    use_connected: bool = True,
) -> Tensor:
    """Block-bootstrap the original lag products without shuffling time points."""
    import numpy as np

    from fragile.physics.qft_utils.statistics import (
        resample_statistics,
        sample_covariance,
        series_statistics,
    )

    if len(series) < 4:
        return torch.full((max_lag + 1,), float("nan"), device=series.device, dtype=series.dtype)
    block_size = max(1, int(len(series) ** 0.5))
    safe_lag = min(max_lag, max(0, len(series) - 2 * block_size - 1))
    stats = series_statistics(series, safe_lag, use_connected)
    samples = resample_statistics(stats, "bootstrap", block_size, n_bootstrap, 42)
    errors = np.sqrt(np.maximum(0, np.diag(sample_covariance(samples, "bootstrap"))))
    result = torch.full((max_lag + 1,), float("nan"), device=series.device, dtype=series.dtype)
    result[: safe_lag + 1] = torch.as_tensor(errors, device=series.device, dtype=series.dtype)
    return result


def compute_correlator_fft(
    series: Tensor,
    max_lag: int,
    use_connected: bool = True,
) -> Tensor:
    """Compute time correlator using FFT.

    Args:
        series: Operator time series [T].
        max_lag: Maximum lag to compute.
        use_connected: Subtract mean (connected correlator).

    Returns:
        Correlator C(t) for t=0 to max_lag [max_lag+1].
    """
    return _fft_correlator_single(series, max_lag, use_connected)


def compute_effective_mass_torch(
    correlator: Tensor,
    dt: float,
) -> Tensor:
    """Compute effective mass from correlator.

    m_eff(t) = -d/dt log(C(t)) ≈ log(C(t)/C(t+1)) / dt

    Args:
        correlator: Correlator C(t) [T].
        dt: Time step.

    Returns:
        Effective mass [T-1].
    """
    if correlator.numel() < 2 or dt <= 0:
        return torch.tensor([], device=correlator.device)

    c0 = correlator[:-1]
    c1 = correlator[1:]

    # Only compute where both are positive
    valid = (c0 > 0) & (c1 > 0)
    eff = torch.full_like(c0, float("nan"))
    eff[valid] = torch.log(c0[valid] / c1[valid]) / dt

    return eff


# =============================================================================
# Pure Function API for Correlator Analysis
# =============================================================================


def extract_mass_aic(
    correlator: Tensor,
    dt: float,
    config: CorrelatorConfig,
    correlator_err: Tensor | None = None,
) -> dict[str, Any]:
    """Fit decay windows in the supplied time unit.

    Statistical error uses measured origin-block covariance where available.
    Without measurement errors, only a point estimate and window spread are
    reported; ``mass_error`` is NaN, never a fabricated confidence interval.
    A correlator whose statistics are too sparse for covariance estimation is
    still fitted; the result is flagged with ``uncertainty_method="unavailable"``
    and an ``uncertainty_note`` instead of aborting the caller.
    """
    import numpy as np

    from fragile.physics.qft_utils.statistics import resample_statistics, sample_covariance

    if not dt > 0:
        msg = "dt must be positive"
        raise ValueError(msg)
    stats = getattr(correlator, "correlator_statistics", None)
    covariance = None
    uncertainty_note: str | None = None
    if stats is not None:
        try:
            samples = resample_statistics(stats, "block_jackknife", 10, 200, 42)
        except ValueError as exc:
            samples = None
            uncertainty_note = str(exc)
        if samples is not None:
            supported = np.isfinite(samples).all(0)
            last_supported = (
                int(np.flatnonzero(~supported)[0]) if not supported.all() else len(correlator)
            )
            if last_supported >= 3:
                correlator = correlator[:last_supported]
                samples = samples[:, :last_supported]
                covariance = sample_covariance(samples, "block_jackknife")
                correlator_err = torch.as_tensor(
                    np.sqrt(np.maximum(0, np.diag(covariance))),
                    device=correlator.device,
                    dtype=correlator.dtype,
                )
            else:
                uncertainty_note = (
                    "fewer than three lags have origin statistics in every jackknife block"
                )
    measured = correlator_err is not None
    mask = torch.isfinite(correlator) & (correlator > 0)
    if measured:
        correlator_err = correlator_err.to(correlator)
        mask &= torch.isfinite(correlator_err)
    unavailable = {
        "mass": 0.0,
        "mass_error": float("nan"),
        "window_spread": float("nan"),
        "statistical_error": float("nan"),
        "r_squared": float("nan"),
        "n_valid_windows": 0,
        "uncertainty_method": "unavailable",
    }
    if not mask.any():
        if uncertainty_note:
            unavailable["uncertainty_note"] = uncertainty_note
        return unavailable
    log_corr = torch.full_like(correlator, float("nan"), dtype=torch.float64)
    log_corr[mask] = correlator[mask].double().log()
    if measured:
        log_err = torch.full_like(log_corr, float("nan"))
        log_err[mask] = correlator_err[mask].double() / correlator[mask].double().abs()
        # A zero error would carry infinite weight; treat such points as unusable.
        log_err = torch.where(log_err > 0, log_err, torch.full_like(log_err, float("nan")))
    else:
        # Unit residual weights define an unweighted point fit, not 10% errors.
        log_err = torch.ones_like(log_corr)
    last = int(mask.nonzero()[-1]) + 1
    log_corr, log_err = log_corr[:last], log_err[:last]
    extractor = ConvolutionalAICExtractor(
        window_widths=config.window_widths,
        min_mass=config.min_mass * dt,
        max_mass=config.max_mass * dt,
        # Unit weights carry no signal-to-noise information to cut on.
        max_log_error=None if not measured else 0.5,
    )
    result = extractor.fit_all_widths(log_corr, log_err)
    for key in ("mass", "mass_error", "window_spread", "statistical_error"):
        if key in result and math.isfinite(result[key]):
            result[key] /= dt
    if result.get("window_masses") is not None:
        result["window_masses"] = result["window_masses"] / dt
    if result.get("window_mass_variance") is not None:
        result["window_mass_variance"] = result["window_mass_variance"] / (dt * dt)
    best_window = result.get("best_window")
    if isinstance(best_window, dict):
        for key in ("mass", "mass_error"):
            if key in best_window and math.isfinite(best_window[key]):
                best_window[key] /= dt
    result["uncertainty_method"] = (
        "origin_block_jackknife"
        if covariance is not None
        else ("supplied_diagonal_errors" if measured else "unavailable")
    )
    if uncertainty_note:
        result["uncertainty_note"] = uncertainty_note
    if not measured:
        result["mass_error"] = float("nan")
        result["statistical_error"] = float("nan")
        if isinstance(best_window, dict):
            best_window["mass_error"] = float("nan")
    elif covariance is not None and result.get("n_valid_windows", 0):
        # Propagate the full lag covariance through the weighted-slope
        # coefficients of every valid window, then AIC-average the variances.
        corr_np = correlator[:last].detach().double().cpu().numpy()
        denominator = corr_np[:, None] * corr_np[None, :]
        log_cov = np.divide(
            covariance[:last, :last],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator != 0,
        )
        w_np = log_err.detach().cpu().numpy()
        w_np = np.where(np.isfinite(w_np) & (w_np > 0), 1.0 / w_np**2, 0.0)
        variances, aics = [], []
        for wi, width in enumerate(result["window_widths"]):
            t = np.arange(width, dtype=float)
            for start in range(last - width + 1):
                aic = float(result["window_aic"][wi, start])
                mass = float(result["window_masses"][wi, start])
                if not (np.isfinite(aic) and np.isfinite(mass) and mass > 0):
                    continue
                w = w_np[start : start + width]
                if not np.all(w > 0):
                    continue
                t_bar = np.sum(w * t) / np.sum(w)
                coeff = w * (t - t_bar) / np.sum(w * (t - t_bar) ** 2) / dt
                sub = log_cov[start : start + width, start : start + width]
                variances.append(max(0.0, float(coeff @ sub @ coeff)))
                aics.append(aic)
        if aics:
            weights = np.exp(-0.5 * (np.asarray(aics) - min(aics)))
            weights /= weights.sum()
            statistical = float(np.sqrt(weights @ np.asarray(variances)))
            result["statistical_error"] = statistical
            result["mass_error"] = float(np.sqrt(statistical**2 + result["window_spread"] ** 2))
    return result


def extract_mass_linear(
    correlator: Tensor,
    dt: float,
    config: CorrelatorConfig,
) -> dict[str, Any]:
    """Extract mass using a simple linear fit on log(C(t)).

    Extracted from ChannelCorrelator.extract_mass_linear().

    Args:
        correlator: Correlator C(t).
        dt: Time step.
        config: CorrelatorConfig.

    Returns:
        Dict with mass, amplitude, r_squared, fit_points.
    """
    if correlator.numel() == 0:
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": 0.0,
        }

    n = correlator.shape[0]
    fit_start = max(0, int(config.fit_start))
    fit_stop = config.fit_stop
    if fit_stop is None:
        fit_stop = n - 1
    fit_stop = min(int(fit_stop), n - 1)
    if fit_stop < fit_start:
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": 0.0,
        }

    idx = torch.arange(n, device=correlator.device, dtype=torch.float32)
    mask = (idx >= fit_start) & (idx <= fit_stop) & (correlator > 0)
    n_points = int(mask.sum().item())
    if n_points < max(2, int(config.min_fit_points)):
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": float(n_points),
        }

    x = idx[mask]
    y = torch.log(correlator[mask])
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xx = (x * x).sum()
    sum_xy = (x * y).sum()
    denom = n_points * sum_xx - sum_x * sum_x
    if denom.abs() < 1e-12:
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": float(n_points),
        }

    slope = (n_points * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n_points
    mass = -slope
    amplitude = torch.exp(intercept)

    y_pred = intercept + slope * x
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mass": float(mass.item()),
        "amplitude": float(amplitude.item()),
        "r_squared": float(r_squared.item()),
        "fit_points": float(n_points),
    }


def compute_channel_correlator(
    series: Tensor,
    dt: float,
    config: CorrelatorConfig,
    channel_name: str = "unknown",
) -> ChannelCorrelatorResult:
    """Compute correlator and mass for a single channel.

    PURE FUNCTION: Takes pre-computed series, no RunHistory access.

    Args:
        series: Operator time series [T].
        dt: Time step.
        config: CorrelatorConfig (analysis configuration).
        channel_name: Channel name for result.

    Returns:
        ChannelCorrelatorResult with correlator, mass, diagnostics.
    """
    if series.numel() == 0:
        return ChannelCorrelatorResult(
            channel_name=channel_name,
            correlator=torch.zeros(config.max_lag + 1),
            correlator_err=None,
            effective_mass=torch.zeros(config.max_lag),
            mass_fit={"mass": 0.0, "mass_error": float("inf")},
            series=series,
            n_samples=0,
            dt=dt,
        )

    # Compute correlator
    real_series = series.real if series.is_complex() else series
    correlator = compute_correlator_fft(
        real_series,
        max_lag=config.max_lag,
        use_connected=config.use_connected,
    )

    # Bootstrap errors if requested
    correlator_err = None
    if config.compute_bootstrap_errors:
        correlator_err = bootstrap_correlator_error(
            real_series,
            max_lag=config.max_lag,
            n_bootstrap=config.n_bootstrap,
            use_connected=config.use_connected,
        )

    # Effective mass
    from fragile.physics.qft_utils.statistics import ensure_statistics

    correlator = ensure_statistics(correlator, series)
    effective_mass = compute_effective_mass_torch(correlator, dt)

    # Mass extraction
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(correlator.abs(), dt, config)
        window_data = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(correlator, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(correlator, dt, config, correlator_err)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=correlator,
        correlator_err=correlator_err,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=int(series.numel()),
        dt=dt,
        **window_data,
    )


def compute_all_correlators(
    operator_series: OperatorTimeSeries,
    config: CorrelatorConfig,
    channels: list[str] | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators for all channels from pre-computed operators.

    PURE FUNCTION: No RunHistory access, just processes operator series.

    Args:
        operator_series: Pre-computed operator series.
        config: CorrelatorConfig (analysis configuration).
        channels: List of channel names (None = all in operator_series).

    Returns:
        Dictionary mapping channel names to results.
    """

    if channels is None:
        channels = list(operator_series.operators.keys())

    results = {}
    for channel_name in channels:
        if channel_name not in operator_series.operators:
            continue

        series = operator_series.operators[channel_name]

        try:
            result = compute_channel_correlator(
                series=series,
                dt=operator_series.dt,
                config=config,
                channel_name=channel_name,
            )
            results[channel_name] = result
        except Exception as e:
            # Create empty result on error
            results[channel_name] = ChannelCorrelatorResult(
                channel_name=channel_name,
                correlator=torch.zeros(config.max_lag + 1),
                correlator_err=None,
                effective_mass=torch.zeros(config.max_lag),
                mass_fit={"mass": 0.0, "mass_error": float("inf"), "error": str(e)},
                series=torch.zeros(0),
                n_samples=0,
                dt=operator_series.dt,
            )

    return results


# =============================================================================
# Base Channel Correlator Classes
# =============================================================================


class ChannelCorrelator(ABC):
    """Abstract base class for channel correlator computation.

    Subclasses implement specific particle channels with vectorized operations.
    """

    channel_name: str = "base"

    def __init__(
        self,
        history: RunHistory,
        config: ChannelConfig | None = None,
        correlator_config: CorrelatorConfig | None = None,
    ):
        """Initialize the channel correlator.

        Args:
            history: Fractal Gas run history.
            config: Configuration parameters for aggregation.
            correlator_config: Configuration for correlator analysis (optional).
        """
        self.history = history
        self.config = config or ChannelConfig()

        # Create correlator config from aggregation config if not provided
        # This maintains backward compatibility
        if correlator_config is None:
            correlator_config = CorrelatorConfig(
                max_lag=getattr(config, "max_lag", 80) if config else 80,
                use_connected=getattr(config, "use_connected", True) if config else True,
                window_widths=getattr(config, "window_widths", None) if config else None,
                min_mass=getattr(config, "min_mass", 0.0) if config else 0.0,
                max_mass=getattr(config, "max_mass", float("inf")) if config else float("inf"),
                fit_mode=getattr(config, "fit_mode", "aic") if config else "aic",
                fit_start=getattr(config, "fit_start", 2) if config else 2,
                fit_stop=getattr(config, "fit_stop", None) if config else None,
                min_fit_points=getattr(config, "min_fit_points", 2) if config else 2,
                compute_bootstrap_errors=getattr(config, "compute_bootstrap_errors", False)
                if config
                else False,
                n_bootstrap=getattr(config, "n_bootstrap", 100) if config else 100,
            )
        self.correlator_config = correlator_config

        self._validate_config()
        self._build_gamma_matrices()

    def _validate_config(self) -> None:
        """Validate and fill missing config values."""
        from fragile.fractalai.qft.aggregation import estimate_ell0

        # Handle deprecated "uniform" alias
        if self.config.neighbor_method == "uniform":
            self.config.neighbor_method = "companions"

        if self.config.neighbor_method not in {"companions", "recorded", "auto"}:
            msg = "neighbor_method must be 'auto', 'companions', or 'recorded'"
            raise ValueError(msg)
        if self.config.ell0 is None:
            self.config.ell0 = estimate_ell0(self.history)

    def _build_gamma_matrices(self) -> None:
        """Build gamma matrices for bilinear projections."""
        d = self.history.d
        device = self.history.x_final.device
        dtype = torch.complex128

        self.gamma: dict[str, Tensor] = {}

        # Identity (scalar channel)
        self.gamma["1"] = torch.eye(d, device=device, dtype=dtype)

        # γ₅ diagonal (pseudoscalar) - alternating signs
        gamma5_diag = torch.tensor(
            [(-1.0) ** i for i in range(d)],
            device=device,
            dtype=dtype,
        )
        self.gamma["5"] = gamma5_diag  # Store just diagonal for efficiency
        self.gamma["5_matrix"] = torch.diag(gamma5_diag)

        # γ_μ matrices (vector) — purely imaginary, anti-symmetric (Levi-Civita)
        # γ_μ[α,β] = i * ε_{μαβ}: angular momentum generators × i
        # Purely imaginary M → Re = parity-odd (vector 1--), Im = parity-even (axial 1+-)
        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, device=device, dtype=dtype)
            nu = (mu + 1) % d
            gamma_mu[mu, nu] = 1.0j
            gamma_mu[nu, mu] = -1.0j
            gamma_mu_list.append(gamma_mu)
        self.gamma["mu"] = torch.stack(gamma_mu_list, dim=0)  # [d, d, d]

        # σ_μν matrices (tensor)
        sigma_list = []
        for mu in range(d):
            for nu in range(mu + 1, d):
                sigma = torch.zeros(d, d, device=device, dtype=dtype)
                sigma[mu, nu] = 1.0j
                sigma[nu, mu] = -1.0j
                sigma_list.append(sigma)
        if sigma_list:
            self.gamma["sigma"] = torch.stack(sigma_list, dim=0)  # [n_pairs, d, d]
        else:
            self.gamma["sigma"] = torch.zeros(0, d, d, device=device, dtype=dtype)

    @abstractmethod
    def _compute_operators_vectorized(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
        sample_indices: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute operators for all timesteps (vectorized).

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].
            sample_indices: Sample indices [T, S].
            neighbor_indices: Neighbor indices [T, S, k].

        Returns:
            Operator time series [T].
        """
        ...

    def compute_series(self) -> Tensor:
        """Compute operator time series.

        Delegates to aggregation.py for all operator computation.

        Returns:
            Series [T] of operator values (MC time) or [n_bins] (Euclidean time).
        """
        from fragile.fractalai.qft.aggregation import compute_all_operator_series

        # Compute all operators using aggregation module (handles both MC and Euclidean time)
        operator_series = compute_all_operator_series(
            self.history,
            self.config,
            channels=[self.channel_name],
        )

        # Extract this channel's series (may not exist if filtered out, e.g., nucleon in d<3)
        if self.channel_name not in operator_series.operators:
            # Channel was filtered out (e.g., nucleon requires d>=3)
            device = self.history.x_final.device
            n_timesteps = operator_series.n_timesteps if operator_series.n_timesteps > 0 else 1
            return torch.zeros(n_timesteps, device=device)

        return operator_series.operators[self.channel_name]

    def compute_correlator(self) -> Tensor:
        """Compute time correlator using FFT.

        Returns:
            Correlator C(t) [max_lag+1].
        """
        series = self.compute_series()
        return compute_correlator_fft(
            series.real if series.is_complex() else series,
            max_lag=self.correlator_config.max_lag,
            use_connected=self.correlator_config.use_connected,
        )

    def compute(self) -> ChannelCorrelatorResult:
        """Compute full channel analysis.

        Returns:
            ChannelCorrelatorResult with all computed quantities.
        """
        series = self.compute_series()
        dt = recorded_time_step(self.history)

        # Use the new pure function API
        return compute_channel_correlator(
            series=series,
            dt=dt,
            config=self.correlator_config,
            channel_name=self.channel_name,
        )


# =============================================================================
# Bilinear Channel Correlators
# =============================================================================


class BilinearChannelCorrelator(ChannelCorrelator):
    """Base class for bilinear (meson) channel correlators.

    Computes ψ̄_i Γ ψ_j operators with different gamma matrix projections.
    """

    @abstractmethod
    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """Apply gamma matrix projection for this channel.

        Args:
            color_i: Color states for site i [T, S, d].
            color_j: Color states for site j [T, S, d].

        Returns:
            Projected bilinear [T, S].
        """
        ...

    def _compute_operators_vectorized(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
        sample_indices: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute bilinear operators for all timesteps.

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].
            sample_indices: Sample indices [T, S].
            neighbor_indices: Neighbor indices [T, S, k].

        Returns:
            Operator time series [T].
        """
        T, _N, _d = color.shape
        S = sample_indices.shape[1]
        device = color.device

        # Gather color states for samples and first neighbors
        # color_i: [T, S, d]
        t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)
        color_i = color[t_idx, sample_indices]

        # Use first neighbor
        first_neighbor = neighbor_indices[:, :, 0]  # [T, S]
        color_j = color[t_idx, first_neighbor]

        # Validity masks
        valid_i = (
            valid[t_idx, sample_indices]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
        )
        valid_j = (
            valid[t_idx, first_neighbor]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
        )
        valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

        # Apply channel-specific projection
        op_values = self._apply_gamma_projection(color_i, color_j)  # [T, S]

        # Mask invalid
        op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

        # Mean over samples per timestep
        counts = valid_mask.sum(dim=1).clamp(min=1)
        return op_values.sum(dim=1) / counts


class ScalarChannel(BilinearChannelCorrelator):
    """Scalar channel (σ): Identity projection.

    J^PC = 0^++

    Operator: ψ̄_i · ψ_j = Σ_a (color_i^a)* · color_j^a
    """

    channel_name = "scalar"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """Identity projection: simple dot product.

        Returns: (color_i.conj() * color_j).sum(dim=-1)
        """
        return (color_i.conj() * color_j).sum(dim=-1).real


class PseudoscalarChannel(BilinearChannelCorrelator):
    """Pseudoscalar channel (π): parity-odd projection.

    J^PC = 0^-+

    Operator: Im[c_i† c_j] — parity-odd under c^α → -(c^α)*.
    """

    channel_name = "pseudoscalar"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """Parity-odd projection: imaginary part of color dot product.

        Under parity, c^α → -(c^α)*, so the bilinear c_i† c_j → (c_i† c_j)*.
        Im[z] → Im[z*] = -Im[z], i.e. parity-odd (pseudoscalar, 0⁻⁺).

        Returns: Im[(color_i.conj() * color_j).sum(dim=-1)]
        """
        return (color_i.conj() * color_j).sum(dim=-1).imag


class VectorChannel(BilinearChannelCorrelator):
    """Vector channel (ρ): Re[c_i† (iε_μ) c_j] projection.

    J^PC = 1^--

    Uses Levi-Civita γ_μ matrices (purely imaginary, anti-symmetric).
    For purely imaginary M: Re[c_i† M c_j] is parity-odd → vector (1--).
    """

    channel_name = "vector"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """γ_μ projection: Re[c_i† (iε_μ) c_j] averaged over directions.

        Purely imaginary γ_μ → Re is parity-odd (vector 1--).
        """
        gamma_mu = self.gamma["mu"].to(color_i.device, dtype=color_i.dtype)  # [d, d, d]
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        return result.mean(dim=-1).real


class AxialVectorChannel(BilinearChannelCorrelator):
    """Axial vector channel (a₁): Im[c_i† (iε_μ) c_j] projection.

    J^PC = 1^+-

    Uses the same Levi-Civita γ_μ matrices as VectorChannel.
    For purely imaginary M: Im[c_i† M c_j] is parity-even → axial vector (1+-).
    """

    channel_name = "axial_vector"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """γ_μ projection: Im[c_i† (iε_μ) c_j] averaged over directions.

        Purely imaginary γ_μ → Im is parity-even (axial vector 1+-).
        """
        gamma_mu = self.gamma["mu"].to(color_i.device, dtype=color_i.dtype)  # [d, d, d]
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        return result.mean(dim=-1).imag


class TensorChannel(BilinearChannelCorrelator):
    """Tensor channel (f₂): Im[c_i† σ_μν c_j] projection.

    J^PC = 2^++

    σ_μν is purely imaginary. Im[c_i† M c_j] is parity-even → tensor (2++).
    """

    channel_name = "tensor"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """σ_μν projection: Im[c_i† σ_μν c_j] averaged over pairs.

        Purely imaginary σ_μν → Im is parity-even (tensor 2++).
        """
        sigma = self.gamma["sigma"].to(color_i.device, dtype=color_i.dtype)  # [n_pairs, d, d]
        if sigma.shape[0] == 0:
            return torch.zeros(color_i.shape[:-1], device=color_i.device)
        result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
        return result.mean(dim=-1).imag


# =============================================================================
# Trilinear (Baryon) Channel Correlator
# =============================================================================


class TrilinearChannelCorrelator(ChannelCorrelator):
    """Base class for trilinear (baryon) channel correlators.

    Computes εᵃᵇᶜ ψᵃ ψᵇ ψᶜ operators using determinant of color matrix.
    """


class NucleonChannel(TrilinearChannelCorrelator):
    """Nucleon channel: 3×3 determinant of color states.

    Requires d>=3 (uses first 3 spatial components).

    Operator: det([ψ_i, ψ_j, ψ_k]) for triplets (i, j, k)
    """

    channel_name = "nucleon"

    def _compute_operators_vectorized(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
        sample_indices: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute nucleon operators using determinant.

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].
            sample_indices: Sample indices [T, S].
            neighbor_indices: Neighbor indices [T, S, k].

        Returns:
            Operator time series [T].
        """
        T, _N, d = color.shape
        device = color.device

        if d < 3:
            # Nucleon requires at least 3 spatial dimensions
            return torch.zeros(T, device=device)

        # Use only first 3 components (spatial dimensions, excluding Euclidean time)
        color = color[..., :3]

        S = sample_indices.shape[1]
        k = neighbor_indices.shape[2]

        if k < 2:
            return torch.zeros(T, device=device)

        # Gather indices
        t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, S)

        # Color states
        color_i = color[t_idx, sample_indices]  # [T, S, d]
        color_j = color[t_idx, neighbor_indices[:, :, 0]]  # [T, S, d]
        color_k = color[t_idx, neighbor_indices[:, :, 1]]  # [T, S, d]

        # Stack to form 3x3 matrix: [T, S, d, 3]
        matrix = torch.stack([color_i, color_j, color_k], dim=-1)

        # Compute determinant: [T, S]
        det = torch.linalg.det(matrix)

        # Validity mask
        valid_i = (
            valid[t_idx, sample_indices]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
        )
        valid_j = (
            valid[t_idx, neighbor_indices[:, :, 0]]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]]
        )
        valid_k = (
            valid[t_idx, neighbor_indices[:, :, 1]]
            & alive[t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]]
        )
        valid_mask = valid_i & valid_j & valid_k

        # Mask invalid
        det = torch.where(valid_mask, det, torch.zeros_like(det))

        # Mean over samples
        counts = valid_mask.sum(dim=1).clamp(min=1)
        series = det.sum(dim=1) / counts

        return series.real if series.is_complex() else series


# =============================================================================
# Gauge (Glueball) Channel Correlator
# =============================================================================


class GaugeChannelCorrelator(ChannelCorrelator):
    """Base class for gauge field correlators."""


class GlueballChannel(GaugeChannelCorrelator):
    """Glueball channel: ||force||² norm.

    J^PC = 0^++

    Operator: Σ_i ||F_i||² (sum of force squared norms)
    """

    channel_name = "glueball"

    def _compute_operators_vectorized(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
        sample_indices: Tensor,
        neighbor_indices: Tensor,
    ) -> Tensor:
        """Compute glueball operators from force field.

        Args:
            color: Color states [T, N, d] (unused, we use force directly).
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].
            sample_indices: Sample indices [T, S].
            neighbor_indices: Neighbor indices [T, S, k].

        Returns:
            Operator time series [T].
        """
        start_idx = max(1, int(self.history.n_recorded * self.config.warmup_fraction))
        end_fraction = getattr(self.config, "end_fraction", 1.0)
        end_idx = max(start_idx + 1, int(self.history.n_recorded * end_fraction))

        # Get force field
        force = self.history.force_viscous[start_idx - 1 : end_idx - 1]  # [T, N, d]
        T = force.shape[0]
        device = force.device

        # Force squared norm: [T, N]
        force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

        # Average over alive walkers per timestep
        series = []
        for t in range(T):
            alive_t = (
                alive[t]
                if t < alive.shape[0]
                else torch.ones(force.shape[1], dtype=torch.bool, device=device)
            )
            if alive_t.any():
                series.append(force_sq[t, alive_t].mean())
            else:
                series.append(torch.tensor(0.0, device=device))

        return torch.stack(series)


# =============================================================================
# Dirac Spinor Bilinear Channel Correlators
# =============================================================================


class _DiracBilinearBase(BilinearChannelCorrelator):
    """Base for Dirac spinor bilinear channels.

    Converts color states c ∈ ℂ³ to Dirac spinors ψ ∈ ℂ⁴ via the Hopf
    fibration, then computes ψ̄_i Γ ψ_j.  Subclasses specify the Γ matrix.
    """

    _dirac_gamma: dict[str, Tensor] | None = None

    def _get_dirac_gamma(self, device: torch.device) -> dict[str, Tensor]:
        if self._dirac_gamma is None:
            from .dirac_spinors import build_dirac_gamma_matrices

            self.__class__._dirac_gamma = build_dirac_gamma_matrices(device=device)
        return self._dirac_gamma

    def _color_to_spinor_pair(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert color pair to Dirac spinor pair, returning (psi_i, psi_j, valid)."""
        from .dirac_spinors import color_to_dirac_spinor

        shape = color_i.shape[:-1]  # [T, S]
        psi_i, vi = color_to_dirac_spinor(color_i.reshape(-1, 3))
        psi_j, vj = color_to_dirac_spinor(color_j.reshape(-1, 3))
        psi_i = psi_i.reshape(*shape, 4)
        psi_j = psi_j.reshape(*shape, 4)
        valid = vi.reshape(shape) & vj.reshape(shape)
        return psi_i, psi_j, valid


class DiracScalarChannel(_DiracBilinearBase):
    """Dirac scalar channel: ψ̄ψ (Γ = I₄).  J^PC = 0^++."""

    channel_name = "dirac_scalar"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        from .dirac_spinors import compute_dirac_bilinear

        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        I4 = torch.eye(4, device=color_i.device, dtype=g["gamma0"].dtype)
        result = compute_dirac_bilinear(psi_i, psi_j, g["gamma0"], I4)
        return torch.where(valid, result, torch.zeros_like(result))


class DiracPseudoscalarChannel(_DiracBilinearBase):
    """Dirac pseudoscalar channel: Im[ψ̄ψ].  J^PC = 0^-+."""

    channel_name = "dirac_pseudoscalar"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        result = torch.einsum("...a,ab,...b->...", psi_i.conj(), g["gamma0"], psi_j).imag
        return torch.where(valid, result, torch.zeros_like(result))


class DiracVectorChannel(_DiracBilinearBase):
    """Dirac vector channel: (1/3)Σ_k ψ̄γ_k ψ.  J^PC = 1^--."""

    channel_name = "dirac_vector"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        from .dirac_spinors import compute_dirac_bilinear

        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        result = compute_dirac_bilinear(psi_i, psi_j, g["gamma0"], g["gamma_k"])
        result = result.mean(dim=-1)
        return torch.where(valid, result, torch.zeros_like(result))


class DiracAxialVectorChannel(_DiracBilinearBase):
    """Dirac axial vector channel: (1/3)Σ_k ψ̄γ₅γ_k ψ.  J^PC = 1^+-."""

    channel_name = "dirac_axial_vector"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        from .dirac_spinors import compute_dirac_bilinear

        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        result = compute_dirac_bilinear(psi_i, psi_j, g["gamma0"], g["gamma5_k"])
        result = result.mean(dim=-1)
        return torch.where(valid, result, torch.zeros_like(result))


class DiracTensorChannel(_DiracBilinearBase):
    """Dirac tensor channel: (1/3)Σ_{j<k} ψ̄σ_jk ψ.  J^P = 1^+ (parity-even).

    Uses only spatial-spatial σ_jk components (indices 3,4,5: σ_12, σ_13, σ_23).
    Couples to the a₁ meson (~1260 MeV).
    """

    channel_name = "dirac_tensor"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        from .dirac_spinors import compute_dirac_bilinear

        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        result = compute_dirac_bilinear(psi_i, psi_j, g["gamma0"], g["sigma_munu"][3:])
        result = result.mean(dim=-1)
        return torch.where(valid, result, torch.zeros_like(result))


class DiracTensor0kChannel(_DiracBilinearBase):
    """Dirac tensor σ_0k channel: (1/3)Σ_k ψ̄σ_{0k} ψ.  J^P = 1^- (parity-odd).

    Uses only temporal-spatial σ_0k components (indices 0,1,2: σ_01, σ_02, σ_03).
    Couples to the pion (pseudoscalar) and provides a cross-check for the
    pseudoscalar mass.
    """

    channel_name = "dirac_tensor_0k"

    def _apply_gamma_projection(self, color_i: Tensor, color_j: Tensor) -> Tensor:
        from .dirac_spinors import compute_dirac_bilinear

        psi_i, psi_j, valid = self._color_to_spinor_pair(color_i, color_j)
        g = self._get_dirac_gamma(color_i.device)
        result = compute_dirac_bilinear(psi_i, psi_j, g["gamma0"], g["sigma_munu"][:3])
        result = result.mean(dim=-1)
        return torch.where(valid, result, torch.zeros_like(result))


# =============================================================================
# Channel Registry and Factory
# =============================================================================


CHANNEL_REGISTRY: dict[str, type[ChannelCorrelator]] = {
    "scalar": ScalarChannel,
    "pseudoscalar": PseudoscalarChannel,
    "vector": VectorChannel,
    "axial_vector": AxialVectorChannel,
    "tensor": TensorChannel,
    "nucleon": NucleonChannel,
    "glueball": GlueballChannel,
    "dirac_scalar": DiracScalarChannel,
    "dirac_pseudoscalar": DiracPseudoscalarChannel,
    "dirac_vector": DiracVectorChannel,
    "dirac_axial_vector": DiracAxialVectorChannel,
    "dirac_tensor": DiracTensorChannel,
    "dirac_tensor_0k": DiracTensor0kChannel,
}


def compute_all_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ChannelConfig | None = None,
    spatial_dims: int | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators for multiple channels.

    BACKWARD COMPATIBLE API: Delegates to new implementation.

    Args:
        history: Fractal Gas run history.
        channels: List of channel names (default: all registered).
        config: Configuration parameters.
        spatial_dims: Number of spatial dimensions (2 or 3). If provided, filters out
            channels that require specific dimensionality (e.g., nucleon requires d=3).

    Returns:
        Dictionary mapping channel names to results.
    """
    from fragile.fractalai.qft.aggregation import compute_all_operator_series

    if channels is None:
        channels = list(CHANNEL_REGISTRY.keys())

    # Filter out baryon channels in 2D mode (they require d=3)
    if spatial_dims is not None and spatial_dims < 3:
        channels = [ch for ch in channels if ch != "nucleon"]

    config = config or ChannelConfig()

    # Step 1: Compute operators (aggregation phase)
    operator_series = compute_all_operator_series(history, config, channels)

    # Step 2: Create correlator config from channel config
    # Extract correlator parameters from config (for backward compatibility)
    correlator_config = CorrelatorConfig(
        max_lag=getattr(config, "max_lag", 80),
        use_connected=getattr(config, "use_connected", True),
        window_widths=getattr(config, "window_widths", None),
        min_mass=getattr(config, "min_mass", 0.0),
        max_mass=getattr(config, "max_mass", float("inf")),
        fit_mode=getattr(config, "fit_mode", "aic"),
        fit_start=getattr(config, "fit_start", 2),
        fit_stop=getattr(config, "fit_stop", None),
        min_fit_points=getattr(config, "min_fit_points", 2),
        compute_bootstrap_errors=getattr(config, "compute_bootstrap_errors", False),
        n_bootstrap=getattr(config, "n_bootstrap", 100),
    )

    # Step 3: Compute correlators (analysis phase)
    return compute_all_correlators(operator_series, correlator_config, channels)


def get_channel_class(channel_name: str) -> type[ChannelCorrelator]:
    """Get channel correlator class by name.

    Args:
        channel_name: Name of the channel.

    Returns:
        Channel correlator class.

    Raises:
        ValueError: If channel name is not registered.
    """
    if channel_name not in CHANNEL_REGISTRY:
        msg = f"Unknown channel: {channel_name}. Available: {list(CHANNEL_REGISTRY.keys())}"
        raise ValueError(msg)
    return CHANNEL_REGISTRY[channel_name]
