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
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from fragile.fractalai.core.history import RunHistory


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
        - "auto" (default): Auto-detect best available (recorded → companions → voronoi)
        - "recorded": Explicitly use pre-computed neighbor_edges from simulation
        - "companions": Use companion walker indices
        - "voronoi": Recompute Delaunay/Voronoi tessellation (slowest but most flexible)

        Using "auto" provides optimal performance by prioritizing pre-computed data
        when available, with automatic fallback to slower methods as needed.
    """

    # Time parameters
    warmup_fraction: float = 0.1

    # Monte Carlo slice selection (for Euclidean time analysis)
    mc_time_index: int | None = None  # Recorded index; None => last recorded slice

    # Time axis selection (Monte Carlo vs Euclidean)
    time_axis: str = "mc"  # "mc" (Monte Carlo timesteps) or "euclidean" (spatial dimension as time)
    euclidean_time_dim: int = 3  # Which spatial dimension to use as Euclidean time (0-indexed)
    euclidean_time_bins: int = 50  # Number of time bins for Euclidean time analysis
    euclidean_time_range: tuple[float, float] | None = None  # (t_min, t_max) or None for auto

    # Color state parameters
    h_eff: float = 1.0
    mass: float = 1.0
    ell0: float | None = None

    # Neighbor selection
    neighbor_method: str = "auto"  # Auto-detect: recorded → companions → voronoi
    neighbor_k: int = 100
    voronoi_pbc_mode: str = "mirror"
    voronoi_exclude_boundary: bool = True
    voronoi_boundary_tolerance: float = 1e-6
    use_time_sliced_tessellation: bool = True
    time_sliced_neighbor_mode: str = "spacelike"


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
    """Extract mass using 1D convolutions for ALL windows simultaneously.

    This class transforms the fitting problem into signal processing.
    For a window of size W, it computes:
    - Mass = -slope of linear fit to log(C(t))
    - AIC = χ² + 2k (k=2 parameters)

    All quantities decompose into convolution sums, enabling processing
    of millions of windows per second.

    Complexity: O(T * num_widths)
    """

    def __init__(
        self,
        window_widths: list[int] | None = None,
        min_mass: float = 0.0,
        max_mass: float = float("inf"),
    ):
        """Initialize the AIC extractor.

        Args:
            window_widths: List of window sizes to try (default: 5 to 50).
            min_mass: Minimum valid mass value.
            max_mass: Maximum valid mass value.
        """
        self.window_widths = window_widths or list(range(5, 51))
        self.min_mass = min_mass
        self.max_mass = max_mass

    def _build_kernels(
        self, W: int, device: torch.device
    ) -> tuple[Tensor, Tensor]:
        """Build convolution kernels for window size W.

        Args:
            W: Window size.
            device: Compute device.

        Returns:
            Tuple of (sum kernel, time moment kernel).
        """
        # Sum kernel: [1, 1, W] of ones
        k_1 = torch.ones(1, 1, W, device=device)

        # Time moment kernel: [0, 1, ..., W-1]
        # conv1d is cross-correlation (no flip), so kernel[k] * input[j+k]
        # We want sum_k k * input[j+k], so kernel = [0, 1, 2, ..., W-1]
        t_vec = torch.arange(W, device=device, dtype=torch.float32)
        k_t = t_vec.view(1, 1, W)

        return k_1, k_t

    def fit_single_width(
        self,
        log_corr: Tensor,
        log_err: Tensor,
        W: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute mass and AIC for all start positions with window width W.

        Uses OLS fitting via convolutions:
            slope = (W * S_ty - S_t * S_y) / denom
            mass = -slope

        Args:
            log_corr: Log correlator [1, 1, T].
            log_err: Log errors [1, 1, T].
            W: Window width.

        Returns:
            Tuple of (mass [1, 1, T-W+1], aic [1, 1, T-W+1], r2 [1, 1, T-W+1]).
        """
        device = log_corr.device
        k_1, k_t = self._build_kernels(W, device)

        # Precompute constants
        t_vec = torch.arange(W, device=device, dtype=torch.float32)
        S_t = t_vec.sum()
        S_tt = (t_vec**2).sum()
        denom = W * S_tt - S_t**2

        if denom.abs() < 1e-12:
            # Degenerate case
            inf_tensor = torch.full_like(log_corr[:, :, : log_corr.shape[-1] - W + 1], float("inf"))
            nan_tensor = torch.full_like(inf_tensor, float("nan"))
            return torch.zeros_like(inf_tensor), inf_tensor, nan_tensor

        # Convolutions for sufficient statistics
        S_y = F.conv1d(log_corr, k_1)  # [1, 1, T-W+1]
        S_ty = F.conv1d(log_corr, k_t)  # [1, 1, T-W+1]
        S_yy = F.conv1d(log_corr**2, k_1)  # [1, 1, T-W+1]

        # OLS fit parameters
        slope = (W * S_ty - S_t * S_y) / denom
        intercept = (S_y - slope * S_t) / W
        mass = -slope

        # Sum of Squared Residuals
        sse = (
            S_yy
            + slope**2 * S_tt
            + W * intercept**2
            - 2 * slope * S_ty
            - 2 * intercept * S_y
            + 2 * slope * intercept * S_t
        )

        # Total Sum of Squares for R^2 (unweighted, log-space)
        sst = S_yy - (S_y**2) / W
        r2 = 1.0 - sse / (sst + 1e-12)
        r2 = torch.where(sst > 0, r2, torch.full_like(r2, float("nan")))

        # Chi² with variance weighting
        avg_var = F.conv1d(log_err**2, k_1) / W
        chi2 = sse / (avg_var + 1e-9)

        # AIC = χ² + 2k (k=2)
        aic = chi2 + 4.0

        # Invalidate non-physical masses
        invalid = (mass < self.min_mass) | (mass > self.max_mass) | ~torch.isfinite(mass)
        aic = torch.where(invalid, torch.full_like(aic, float("inf")), aic)
        r2 = torch.where(invalid, torch.full_like(r2, float("nan")), r2)

        return mass, aic, r2

    def fit_all_widths(
        self,
        log_corr: Tensor,
        log_err: Tensor,
    ) -> dict[str, Any]:
        """Fit ALL windows for ALL widths and compute AIC-weighted average.

        Args:
            log_corr: Log correlator [T].
            log_err: Log errors [T].

        Returns:
            Dict with mass, mass_error, best_window, n_valid_windows.
        """
        T = log_corr.shape[0]
        device = log_corr.device

        # Reshape for conv1d: [1, 1, T]
        log_corr = log_corr.view(1, 1, -1)
        log_err = log_err.view(1, 1, -1)

        all_masses = []
        all_aics = []
        all_r2 = []
        valid_widths = []

        for W in self.window_widths:
            if W > T:
                continue

            mass, aic, r2 = self.fit_single_width(log_corr, log_err, W)

            # Pad to length T (conv output is T-W+1)
            pad_right = T - mass.shape[-1]
            mass = F.pad(mass, (0, pad_right), value=float("nan"))
            aic = F.pad(aic, (0, pad_right), value=float("inf"))
            r2 = F.pad(r2, (0, pad_right), value=float("nan"))

            all_masses.append(mass)
            all_aics.append(aic)
            all_r2.append(r2)
            valid_widths.append(W)

        if not all_masses:
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "r_squared": float("nan"),
                "n_valid_windows": 0,
                "window_masses": None,
                "window_aic": None,
                "window_widths": [],
                "window_r2": None,
            }

        # Stack: [num_widths, T]
        mass_stack = torch.cat(all_masses, dim=0).squeeze(1)  # [num_widths, T]
        aic_stack = torch.cat(all_aics, dim=0).squeeze(1)  # [num_widths, T]
        r2_stack = torch.cat(all_r2, dim=0).squeeze(1)  # [num_widths, T]

        # Flatten to [num_widths * T]
        flat_mass = mass_stack.flatten()
        flat_aic = aic_stack.flatten()
        flat_r2 = r2_stack.flatten()

        # Filter valid (finite AIC, positive mass)
        valid = torch.isfinite(flat_aic) & (flat_mass > 0)

        if not valid.any():
            return {
                "mass": 0.0,
                "mass_error": float("inf"),
                "r_squared": float("nan"),
                "n_valid_windows": 0,
                "window_masses": mass_stack,
                "window_aic": aic_stack,
                "window_widths": valid_widths,
                "window_r2": r2_stack,
            }

        # AIC weights: w_i = exp(-0.5 * (AIC_i - AIC_min))
        aic_valid = flat_aic[valid]
        mass_valid = flat_mass[valid]

        aic_min = aic_valid.min()
        delta_aic = aic_valid - aic_min
        weights = torch.exp(-0.5 * delta_aic)
        weights = weights / weights.sum()

        # Weighted average
        mass_final = (weights * mass_valid).sum().item()
        mass_var = (weights * (mass_valid - mass_final) ** 2).sum()
        mass_error = mass_var.sqrt().item()

        r2_final = float("nan")
        valid_r2 = valid & torch.isfinite(flat_r2)
        if valid_r2.any():
            aic_r2 = flat_aic[valid_r2]
            r2_vals = flat_r2[valid_r2]
            aic_min_r2 = aic_r2.min()
            delta_aic_r2 = aic_r2 - aic_min_r2
            weights_r2 = torch.exp(-0.5 * delta_aic_r2)
            weights_r2 = weights_r2 / weights_r2.sum()
            r2_final = (weights_r2 * r2_vals).sum().item()

        # Best window
        best_flat_idx = flat_aic.argmin().item()
        best_w_idx = best_flat_idx // T
        best_t_idx = best_flat_idx % T
        best_r2 = flat_r2[best_flat_idx].item() if torch.isfinite(flat_r2[best_flat_idx]) else float("nan")

        return {
            "mass": mass_final,
            "mass_error": mass_error,
            "r_squared": r2_final,
            "n_valid_windows": int(valid.sum().item()),
            "best_window": {
                "width": valid_widths[best_w_idx] if best_w_idx < len(valid_widths) else 0,
                "t_start": best_t_idx,
                "mass": flat_mass[best_flat_idx].item(),
                "aic": flat_aic[best_flat_idx].item(),
                "r2": best_r2,
            },
            "window_masses": mass_stack,
            "window_aic": aic_stack,
            "window_widths": valid_widths,
            "window_r2": r2_stack,
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
    if series.numel() == 0:
        return torch.zeros(max_lag + 1, device=series.device, dtype=series.dtype)

    T = series.shape[0]
    if use_connected:
        series = series - series.mean()

    # Zero-pad for FFT convolution
    padded = F.pad(series.float(), (0, T))
    fft_s = torch.fft.fft(padded)
    corr = torch.fft.ifft(fft_s * fft_s.conj()).real

    # Normalize by number of overlapping samples
    effective_lag = min(max_lag, T - 1)
    counts = torch.arange(T, T - effective_lag - 1, -1, device=series.device, dtype=torch.float32)
    result = corr[: effective_lag + 1] / counts

    # Pad with zeros if max_lag > T-1
    if effective_lag < max_lag:
        result = F.pad(result, (0, max_lag - effective_lag), value=0.0)

    return result


def bootstrap_correlator_error(
    series: Tensor,
    max_lag: int,
    n_bootstrap: int = 100,
    use_connected: bool = True,
) -> Tensor:
    """Compute bootstrap standard error for correlator.

    Uses block bootstrap resampling to estimate uncertainty in the correlator.
    The series is resampled with replacement and the correlator is computed
    for each resample. The standard deviation across resamples gives the
    standard error estimate.

    Args:
        series: Operator time series [T].
        max_lag: Maximum lag to compute.
        n_bootstrap: Number of bootstrap resamples.
        use_connected: Subtract mean (connected correlator).

    Returns:
        Bootstrap standard error for C(t) [max_lag+1].
    """
    if series.numel() == 0:
        return torch.zeros(max_lag + 1, device=series.device, dtype=series.dtype)

    T = series.shape[0]
    device = series.device
    dtype = series.dtype

    # Collect bootstrap correlator samples
    bootstrap_corrs = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = torch.randint(0, T, (T,), device=device)
        resampled = series[indices]
        # Compute correlator for this resample
        corr = _fft_correlator_single(resampled, max_lag, use_connected)
        bootstrap_corrs.append(corr)

    # Stack and compute standard deviation (standard error)
    stacked = torch.stack(bootstrap_corrs)  # [n_bootstrap, max_lag+1]
    return stacked.std(dim=0)


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
) -> dict[str, Any]:
    """Extract mass using convolutional AIC.

    Extracted from ChannelCorrelator.extract_mass_aic().

    Args:
        correlator: Correlator C(t) [max_lag+1].
        dt: Time step.
        config: CorrelatorConfig.

    Returns:
        Dict with mass, mass_error, and fitting details.
    """
    # Filter positive values for log
    mask = correlator > 0
    if not mask.any():
        return {"mass": 0.0, "mass_error": float("inf"), "n_valid_windows": 0}

    log_corr = torch.full_like(correlator, float("nan"))
    log_corr[mask] = torch.log(correlator[mask])

    # Estimate errors (simple bootstrap proxy)
    log_err = torch.ones_like(log_corr) * 0.1

    # Find first NaN to trim series
    finite_mask = torch.isfinite(log_corr)
    if not finite_mask.any():
        return {"mass": 0.0, "mass_error": float("inf"), "n_valid_windows": 0}

    last_valid = finite_mask.nonzero()[-1].item()
    log_corr = log_corr[: last_valid + 1]
    log_err = log_err[: last_valid + 1]

    extractor = ConvolutionalAICExtractor(
        window_widths=config.window_widths,
        min_mass=config.min_mass,
        max_mass=config.max_mass,
    )

    return extractor.fit_all_widths(log_corr, log_err)


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
    effective_mass = compute_effective_mass_torch(correlator, dt)

    # Mass extraction
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(correlator.abs(), dt, config)
        window_data = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(correlator, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(correlator, dt, config)
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
    from fragile.fractalai.qft.aggregation import OperatorTimeSeries

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
                max_lag=getattr(config, 'max_lag', 80) if config else 80,
                use_connected=getattr(config, 'use_connected', True) if config else True,
                window_widths=getattr(config, 'window_widths', None) if config else None,
                min_mass=getattr(config, 'min_mass', 0.0) if config else 0.0,
                max_mass=getattr(config, 'max_mass', float('inf')) if config else float('inf'),
                fit_mode=getattr(config, 'fit_mode', 'aic') if config else 'aic',
                fit_start=getattr(config, 'fit_start', 2) if config else 2,
                fit_stop=getattr(config, 'fit_stop', None) if config else None,
                min_fit_points=getattr(config, 'min_fit_points', 2) if config else 2,
                compute_bootstrap_errors=getattr(config, 'compute_bootstrap_errors', False) if config else False,
                n_bootstrap=getattr(config, 'n_bootstrap', 100) if config else 100,
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

        if self.config.neighbor_method not in {"companions", "voronoi", "recorded", "auto"}:
            msg = "neighbor_method must be 'auto', 'companions', 'voronoi', or 'recorded'"
            raise ValueError(msg)
        if self.config.use_time_sliced_tessellation:
            if self.config.time_sliced_neighbor_mode not in {
                "spacelike",
                "timelike",
                "spacelike+timelike",
            }:
                msg = (
                    "time_sliced_neighbor_mode must be "
                    "'spacelike', 'timelike', or 'spacelike+timelike'"
                )
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

        # γ_μ matrices (vector)
        gamma_mu_list = []
        for mu in range(d):
            gamma_mu = torch.zeros(d, d, device=device, dtype=dtype)
            gamma_mu[mu, mu] = 1.0
            if mu > 0:
                gamma_mu[mu, 0] = 0.5j
                gamma_mu[0, mu] = -0.5j
            gamma_mu_list.append(gamma_mu)
        self.gamma["mu"] = torch.stack(gamma_mu_list, dim=0)  # [d, d, d]

        # γ₅γ_μ matrices (axial vector)
        gamma_5mu_list = []
        for mu in range(d):
            gamma_5mu = self.gamma["5_matrix"] @ gamma_mu_list[mu]
            gamma_5mu_list.append(gamma_5mu)
        self.gamma["5mu"] = torch.stack(gamma_5mu_list, dim=0)  # [d, d, d]

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
        dt = float(self.history.delta_t * self.history.record_every)

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
        T, N, d = color.shape
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
        valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
        valid_j = valid[t_idx, first_neighbor] & alive[t_idx.clamp(max=alive.shape[0] - 1), first_neighbor]
        valid_mask = valid_i & valid_j & (first_neighbor != sample_indices)

        # Apply channel-specific projection
        op_values = self._apply_gamma_projection(color_i, color_j)  # [T, S]

        # Mask invalid
        op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

        # Mean over samples per timestep
        counts = valid_mask.sum(dim=1).clamp(min=1)
        series = op_values.sum(dim=1) / counts

        return series


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
    """Pseudoscalar channel (π): γ₅ diagonal projection.

    J^PC = 0^-+

    Operator: ψ̄_i γ₅ ψ_j with γ₅ = diag(1, -1, 1, -1, ...)
    """

    channel_name = "pseudoscalar"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """γ₅ projection: alternating sign dot product.

        Returns: (color_i.conj() * gamma5_diag * color_j).sum(dim=-1)
        """
        gamma5_diag = self.gamma["5"].to(color_i.device)  # [d]
        return (color_i.conj() * gamma5_diag * color_j).sum(dim=-1).real


class VectorChannel(BilinearChannelCorrelator):
    """Vector channel (ρ): γ_μ sum projection.

    J^PC = 1^--

    Operator: Σ_μ ψ̄_i γ_μ ψ_j averaged over spatial directions
    """

    channel_name = "vector"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """γ_μ projection: einsum over gamma matrices.

        Returns: einsum("...i,mij,...j->...m", ci.conj(), gamma_mu, cj).mean(-1)
        """
        gamma_mu = self.gamma["mu"].to(color_i.device, dtype=color_i.dtype)  # [d, d, d]
        # einsum: batch dims ..., color i, matrix (m, i, j), color j -> batch, mu
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_mu, color_j)
        return result.mean(dim=-1).real


class AxialVectorChannel(BilinearChannelCorrelator):
    """Axial vector channel (a₁): γ₅γ_μ projection.

    J^PC = 1^+-

    Operator: Σ_μ ψ̄_i γ₅γ_μ ψ_j averaged over spatial directions
    """

    channel_name = "axial_vector"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """γ₅γ_μ projection.

        Returns: einsum("...i,mij,...j->...m", ci.conj(), gamma5_mu, cj).mean(-1)
        """
        gamma_5mu = self.gamma["5mu"].to(color_i.device, dtype=color_i.dtype)  # [d, d, d]
        result = torch.einsum("...i,mij,...j->...m", color_i.conj(), gamma_5mu, color_j)
        return result.mean(dim=-1).real


class TensorChannel(BilinearChannelCorrelator):
    """Tensor channel (f₂): σ_μν antisymmetric projection.

    J^PC = 2^++

    Operator: Σ_{μ<ν} ψ̄_i σ_μν ψ_j averaged over antisymmetric pairs
    """

    channel_name = "tensor"

    def _apply_gamma_projection(
        self,
        color_i: Tensor,
        color_j: Tensor,
    ) -> Tensor:
        """σ_μν projection: antisymmetric tensor.

        Returns: einsum("...i,pij,...j->...p", ci.conj(), sigma_munu, cj).mean(-1)
        """
        sigma = self.gamma["sigma"].to(color_i.device, dtype=color_i.dtype)  # [n_pairs, d, d]
        if sigma.shape[0] == 0:
            return torch.zeros(color_i.shape[:-1], device=color_i.device)
        result = torch.einsum("...i,pij,...j->...p", color_i.conj(), sigma, color_j)
        return result.mean(dim=-1).real


# =============================================================================
# Trilinear (Baryon) Channel Correlator
# =============================================================================


class TrilinearChannelCorrelator(ChannelCorrelator):
    """Base class for trilinear (baryon) channel correlators.

    Computes εᵃᵇᶜ ψᵃ ψᵇ ψᶜ operators using determinant of color matrix.
    """

    pass


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
        T, N, d = color.shape
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
        valid_i = valid[t_idx, sample_indices] & alive[t_idx.clamp(max=alive.shape[0] - 1), sample_indices]
        valid_j = valid[t_idx, neighbor_indices[:, :, 0]] & alive[
            t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 0]
        ]
        valid_k = valid[t_idx, neighbor_indices[:, :, 1]] & alive[
            t_idx.clamp(max=alive.shape[0] - 1), neighbor_indices[:, :, 1]
        ]
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

    pass


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
        n_recorded = self.history.n_recorded

        # Get force field
        force = self.history.force_viscous[start_idx - 1 : n_recorded - 1]  # [T, N, d]
        T = force.shape[0]
        device = force.device

        # Force squared norm: [T, N]
        force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

        # Average over alive walkers per timestep
        series = []
        for t in range(T):
            alive_t = alive[t] if t < alive.shape[0] else torch.ones(force.shape[1], dtype=torch.bool, device=device)
            if alive_t.any():
                series.append(force_sq[t, alive_t].mean())
            else:
                series.append(torch.tensor(0.0, device=device))

        return torch.stack(series)


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
        channels = [ch for ch in channels if ch not in {"nucleon"}]

    config = config or ChannelConfig()

    # Step 1: Compute operators (aggregation phase)
    operator_series = compute_all_operator_series(history, config, channels)

    # Step 2: Create correlator config from channel config
    # Extract correlator parameters from config (for backward compatibility)
    correlator_config = CorrelatorConfig(
        max_lag=getattr(config, 'max_lag', 80),
        use_connected=getattr(config, 'use_connected', True),
        window_widths=getattr(config, 'window_widths', None),
        min_mass=getattr(config, 'min_mass', 0.0),
        max_mass=getattr(config, 'max_mass', float('inf')),
        fit_mode=getattr(config, 'fit_mode', 'aic'),
        fit_start=getattr(config, 'fit_start', 2),
        fit_stop=getattr(config, 'fit_stop', None),
        min_fit_points=getattr(config, 'min_fit_points', 2),
        compute_bootstrap_errors=getattr(config, 'compute_bootstrap_errors', False),
        n_bootstrap=getattr(config, 'n_bootstrap', 100),
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
