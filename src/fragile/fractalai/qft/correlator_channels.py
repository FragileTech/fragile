"""Vectorized channel correlator computation for lattice QFT analysis.

This module provides a class hierarchy for computing two-point correlators
across different particle channels, with PyTorch-based vectorization and
convolutional AIC fitting for efficient mass extraction.

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
from dataclasses import dataclass, field
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
    """Configuration for channel correlator computation."""

    # Time parameters
    warmup_fraction: float = 0.1
    max_lag: int = 80

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
    neighbor_method: str = "knn"
    knn_k: int = 4
    knn_sample: int | None = 512
    voronoi_pbc_mode: str = "mirror"
    voronoi_exclude_boundary: bool = True
    voronoi_boundary_tolerance: float = 1e-6

    # AIC fitting parameters
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")
    # Linear fit parameters (legacy-style)
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2

    # Correlator options
    use_connected: bool = True


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


def _resolve_mc_time_index(history, mc_time_index: int | None) -> int:
    """Resolve a Monte Carlo slice index from either recorded index or step."""
    if history.n_recorded < 2:
        raise ValueError("Need at least 2 recorded timesteps for Euclidean analysis.")
    if mc_time_index is None:
        resolved = history.n_recorded - 1
    else:
        try:
            raw = int(mc_time_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid mc_time_index: {mc_time_index}") from exc
        if raw in history.recorded_steps:
            resolved = history.get_step_index(raw)
        else:
            resolved = raw
    if resolved < 1 or resolved >= history.n_recorded:
        msg = (
            f"mc_time_index {resolved} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1} "
            "or a recorded step value)."
        )
        raise ValueError(msg)
    return resolved


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


def bin_by_euclidean_time(
    positions: Tensor,
    operators: Tensor,
    alive: Tensor,
    time_dim: int = 3,
    n_bins: int = 50,
    time_range: tuple[float, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Bin walkers by Euclidean time coordinate and compute mean operator per bin.
    
    In 4D simulations (3 spatial + 1 Euclidean time), this function treats one
    spatial dimension as a time coordinate and computes operator averages within
    time bins. This enables lattice QFT analysis where correlators are computed
    over spatial separation in the time dimension rather than Monte Carlo timesteps.
    
    Args:
        positions: Walker positions over MC time [T, N, d]
        operators: Operator values to average [T, N]
        alive: Alive mask [T, N]
        time_dim: Which spatial dimension is Euclidean time (0-indexed, default 3)
        n_bins: Number of time bins
        time_range: (t_min, t_max) or None for auto from data
        
    Returns:
        time_coords: Bin centers [n_bins]
        operator_series: Mean operator vs Euclidean time [n_bins]
        
    Example:
        >>> # 4D simulation with d=4, treat 4th dim as time
        >>> positions = history.x_before_clone  # [T, N, 4]
        >>> operators = compute_scalar_operators(...)  # [T, N]
        >>> alive = history.alive_mask  # [T, N]
        >>> time_coords, series = bin_by_euclidean_time(positions, operators, alive)
        >>> correlator = compute_correlator_fft(series, max_lag=40)
    """
    # Extract Euclidean time coordinate
    t_euc = positions[:, :, time_dim]  # [T, N]
    
    # Flatten over MC time dimension to treat all snapshots as ensemble
    t_euc_flat = t_euc[alive]  # [total_alive_walkers]
    ops_flat = operators[alive]
    
    if t_euc_flat.numel() == 0:
        # No alive walkers
        device = positions.device
        return torch.zeros(n_bins, device=device), torch.zeros(n_bins, device=device)
    
    # Determine time range
    if time_range is None:
        t_min, t_max = t_euc_flat.min().item(), t_euc_flat.max().item()
        # Add small padding to avoid edge effects
        padding = (t_max - t_min) * 0.01
        t_min -= padding
        t_max += padding
    else:
        t_min, t_max = time_range
    
    # Create bins
    edges = torch.linspace(t_min, t_max, n_bins + 1, device=positions.device)
    bin_centers = (edges[:-1] + edges[1:]) / 2
    
    # Bin operators using vectorized histogram
    operator_series = torch.zeros(n_bins, device=positions.device)
    counts = torch.zeros(n_bins, device=positions.device)
    
    for i in range(n_bins):
        mask = (t_euc_flat >= edges[i]) & (t_euc_flat < edges[i + 1])
        count = mask.sum()
        if count > 0:
            operator_series[i] = ops_flat[mask].sum()
            counts[i] = count.float()
    
    # Handle last bin inclusively
    mask = t_euc_flat == edges[-1]
    if mask.sum() > 0:
        operator_series[-1] += ops_flat[mask].sum()
        counts[-1] += mask.sum().float()
    
    # Average
    valid = counts > 0
    operator_series[valid] = operator_series[valid] / counts[valid]
    operator_series[~valid] = 0.0
    
    return bin_centers, operator_series


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
    ):
        """Initialize the channel correlator.

        Args:
            history: Fractal Gas run history.
            config: Configuration parameters.
        """
        self.history = history
        self.config = config or ChannelConfig()
        self._validate_config()
        self._build_gamma_matrices()

    def _validate_config(self) -> None:
        """Validate and fill missing config values."""
        if self.config.neighbor_method not in {"uniform", "knn", "voronoi", "recorded"}:
            msg = "neighbor_method must be 'uniform', 'knn', 'voronoi', or 'recorded'"
            raise ValueError(msg)
        if self.config.ell0 is None:
            self._estimate_ell0()

    def _estimate_ell0(self) -> None:
        """Estimate ell0 from median companion distance."""
        mid_idx = self.history.n_recorded // 2
        if mid_idx == 0:
            self.config.ell0 = 1.0
            return

        x_pre = self.history.x_before_clone[mid_idx]
        comp_idx = self.history.companions_distance[mid_idx - 1]
        alive = self.history.alive_mask[mid_idx - 1]

        # Compute distances
        diff = x_pre - x_pre[comp_idx]
        if self.history.pbc and self.history.bounds is not None:
            high = self.history.bounds.high.to(x_pre)
            low = self.history.bounds.low.to(x_pre)
            span = high - low
            diff = diff - span * torch.round(diff / span)
        dist = torch.linalg.vector_norm(diff, dim=-1)

        if dist.numel() > 0 and alive.any():
            self.config.ell0 = float(dist[alive].median().item())
        else:
            self.config.ell0 = 1.0

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

    def _compute_color_states_batch(
        self,
        start_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Compute color states for all timesteps from start_idx onward.

        Vectorized across T dimension.

        Args:
            start_idx: Starting time index.

        Returns:
            Tuple of (color [T, N, d], valid [T, N]).
        """
        n_recorded = self.history.n_recorded
        T = n_recorded - start_idx

        # Extract batched tensors
        v_pre = self.history.v_before_clone[start_idx:]  # [T, N, d]
        force_visc = self.history.force_viscous[start_idx - 1 : n_recorded - 1]  # [T, N, d]

        # Color state computation (vectorized)
        h_eff = self.config.h_eff
        mass = self.config.mass
        ell0 = self.config.ell0

        phase = (mass * v_pre * ell0) / h_eff
        complex_phase = torch.polar(torch.ones_like(phase), phase.float())

        if force_visc.dtype == torch.float64:
            complex_dtype = torch.complex128
        else:
            complex_dtype = torch.complex64

        tilde = force_visc.to(complex_dtype) * complex_phase.to(complex_dtype)
        norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)
        color = tilde / norm
        valid = norm.squeeze(-1) > 1e-12

        return color, valid

    def _compute_knn_batch(
        self,
        start_idx: int,
        sample_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute k-NN indices for all timesteps.

        Args:
            start_idx: Starting time index.
            sample_size: Number of walkers to sample per timestep.

        Returns:
            Tuple of (sample_indices [T, S], neighbor_indices [T, S, k], alive [T, N]).
        """
        n_recorded = self.history.n_recorded
        T = n_recorded - start_idx
        N = self.history.N
        k = self.config.knn_k
        sample_size = sample_size or self.config.knn_sample or N
        device = self.history.x_final.device

        # Get positions and alive masks
        x_pre = self.history.x_before_clone[start_idx:]  # [T, N, d]
        alive = self.history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]

        all_sample_idx = []
        all_neighbor_idx = []

        for t in range(T):
            alive_t = alive[t]
            alive_indices = torch.where(alive_t)[0]

            if alive_indices.numel() == 0:
                # No alive walkers
                all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
                all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
                continue

            # Sample indices
            if alive_indices.numel() <= sample_size:
                sample_idx = alive_indices
            else:
                sample_idx = alive_indices[:sample_size]

            actual_sample_size = sample_idx.numel()

            # Compute pairwise distances
            pos_sample = x_pre[t, sample_idx]  # [S, d]
            pos_all = x_pre[t]  # [N, d]

            diff = pos_sample.unsqueeze(1) - pos_all.unsqueeze(0)  # [S, N, d]
            if self.history.pbc and self.history.bounds is not None:
                high = self.history.bounds.high.to(pos_sample)
                low = self.history.bounds.low.to(pos_sample)
                span = high - low
                diff = diff - span * torch.round(diff / span)

            dist_sq = (diff**2).sum(dim=-1)  # [S, N]

            # Mask out dead walkers and self
            alive_mask = alive_t.unsqueeze(0).expand(actual_sample_size, -1)
            dist_sq = dist_sq.masked_fill(~alive_mask, float("inf"))
            dist_sq[torch.arange(actual_sample_size, device=device), sample_idx] = float("inf")

            # Get k nearest neighbors
            k_eff = min(k, alive_indices.numel() - 1)
            if k_eff <= 0:
                neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)
            else:
                _, indices = torch.topk(dist_sq, k=k_eff, largest=False)
                # Pad to k if needed
                if k_eff < k:
                    indices = F.pad(indices, (0, k - k_eff), value=0)
                neighbor_idx = indices

            # Pad sample to sample_size if needed
            if actual_sample_size < sample_size:
                sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
                neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

            all_sample_idx.append(sample_idx)
            all_neighbor_idx.append(neighbor_idx)

        sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
        neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

        return sample_indices, neighbor_indices, alive

    def _compute_companion_batch(
        self,
        start_idx: int,
        sample_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use stored companion indices as neighbors (uniform companion selection)."""
        n_recorded = self.history.n_recorded
        T = n_recorded - start_idx
        N = self.history.N
        k = max(2, int(self.config.knn_k))
        sample_size = sample_size or self.config.knn_sample or N
        device = self.history.x_final.device

        alive = self.history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]
        companions_distance = self.history.companions_distance[start_idx - 1 : n_recorded - 1]
        companions_clone = self.history.companions_clone[start_idx - 1 : n_recorded - 1]

        all_sample_idx = []
        all_neighbor_idx = []

        for t in range(T):
            alive_t = alive[t]
            alive_indices = torch.where(alive_t)[0]

            if alive_indices.numel() == 0:
                all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
                all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
                continue

            if alive_indices.numel() <= sample_size:
                sample_idx = alive_indices
            else:
                sample_idx = alive_indices[:sample_size]

            actual_sample_size = sample_idx.numel()
            comp_d = companions_distance[t, sample_idx]
            comp_c = companions_clone[t, sample_idx]
            neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)
            neighbor_idx[:, 0] = comp_d
            neighbor_idx[:, 1] = comp_c

            if actual_sample_size < sample_size:
                sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
                neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

            all_sample_idx.append(sample_idx)
            all_neighbor_idx.append(neighbor_idx)

        sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
        neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

        return sample_indices, neighbor_indices, alive

    def _compute_voronoi_batch(
        self,
        start_idx: int,
        sample_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute Voronoi neighbor indices for all timesteps."""
        try:
            from fragile.fractalai.qft.voronoi_observables import compute_voronoi_tessellation
        except Exception:
            return self._compute_knn_batch(start_idx, sample_size=sample_size)

        n_recorded = self.history.n_recorded
        T = n_recorded - start_idx
        N = self.history.N
        k = int(self.config.knn_k)
        sample_size = sample_size or self.config.knn_sample or N
        device = self.history.x_final.device

        x_pre = self.history.x_before_clone[start_idx:]  # [T, N, d]
        alive = self.history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]

        all_sample_idx = []
        all_neighbor_idx = []

        for t in range(T):
            alive_t = alive[t]
            alive_indices = torch.where(alive_t)[0]

            if alive_indices.numel() == 0:
                all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
                all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
                continue

            if alive_indices.numel() <= sample_size:
                sample_idx = alive_indices
            else:
                sample_idx = alive_indices[:sample_size]

            actual_sample_size = sample_idx.numel()
            neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)

            vor_data = compute_voronoi_tessellation(
                positions=x_pre[t],
                alive=alive_t,
                bounds=self.history.bounds,
                pbc=self.history.pbc,
                pbc_mode=self.config.voronoi_pbc_mode,
                exclude_boundary=self.config.voronoi_exclude_boundary,
                boundary_tolerance=self.config.voronoi_boundary_tolerance,
            )
            neighbor_lists = vor_data.get("neighbor_lists", {})
            index_map = vor_data.get("index_map", {})
            reverse_map = {v: k for k, v in index_map.items()}

            for s_idx, i_idx in enumerate(sample_idx):
                i_orig = int(i_idx.item())
                i_vor = reverse_map.get(i_orig)
                if i_vor is None:
                    continue
                neighbors_vor = neighbor_lists.get(i_vor, [])
                if not neighbors_vor:
                    continue
                neighbors_orig = [index_map[n] for n in neighbors_vor if n in index_map]
                if not neighbors_orig:
                    continue
                chosen = neighbors_orig[:k]
                if len(chosen) < k:
                    chosen.extend([i_orig] * (k - len(chosen)))
                neighbor_idx[s_idx] = torch.tensor(chosen, device=device)

            if actual_sample_size < sample_size:
                sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
                neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

            all_sample_idx.append(sample_idx)
            all_neighbor_idx.append(neighbor_idx)

        sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
        neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

        return sample_indices, neighbor_indices, alive

    def _compute_recorded_neighbors_batch(
        self,
        start_idx: int,
        sample_size: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Use recorded neighbor edges from RunHistory (uniform over neighbors)."""
        if self.history.neighbor_edges is None:
            return self._compute_companion_batch(start_idx, sample_size=sample_size)

        n_recorded = self.history.n_recorded
        T = n_recorded - start_idx
        N = self.history.N
        k = max(1, int(self.config.knn_k))
        sample_size = sample_size or self.config.knn_sample or N
        device = self.history.x_final.device

        alive = self.history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]

        all_sample_idx = []
        all_neighbor_idx = []

        for t in range(T):
            alive_t = alive[t]
            alive_indices = torch.where(alive_t)[0]
            if alive_indices.numel() == 0:
                all_sample_idx.append(torch.zeros(sample_size, device=device, dtype=torch.long))
                all_neighbor_idx.append(torch.zeros(sample_size, k, device=device, dtype=torch.long))
                continue

            if alive_indices.numel() <= sample_size:
                sample_idx = alive_indices
            else:
                sample_idx = alive_indices[:sample_size]

            actual_sample_size = sample_idx.numel()
            neighbor_idx = torch.zeros(actual_sample_size, k, device=device, dtype=torch.long)

            record_idx = start_idx + t
            edges = self.history.neighbor_edges[record_idx]
            if not torch.is_tensor(edges) or edges.numel() == 0:
                # Fallback to self-padding
                neighbor_idx[:] = sample_idx.unsqueeze(1)
            else:
                edge_list = edges.tolist()
                neighbor_map: dict[int, list[int]] = {}
                for i, j in edge_list:
                    if i == j:
                        continue
                    if i not in neighbor_map:
                        neighbor_map[i] = [j]
                    else:
                        neighbor_map[i].append(j)

                for s_i, i_idx in enumerate(sample_idx.tolist()):
                    neighbors = neighbor_map.get(i_idx, [])
                    if not neighbors:
                        neighbor_idx[s_i] = i_idx
                        continue
                    chosen = neighbors[:k]
                    if len(chosen) < k:
                        chosen.extend([i_idx] * (k - len(chosen)))
                    neighbor_idx[s_i] = torch.tensor(chosen, device=device)

            if actual_sample_size < sample_size:
                sample_idx = F.pad(sample_idx, (0, sample_size - actual_sample_size), value=0)
                neighbor_idx = F.pad(neighbor_idx, (0, 0, 0, sample_size - actual_sample_size), value=0)

            all_sample_idx.append(sample_idx)
            all_neighbor_idx.append(neighbor_idx)

        sample_indices = torch.stack(all_sample_idx, dim=0)  # [T, S]
        neighbor_indices = torch.stack(all_neighbor_idx, dim=0)  # [T, S, k]

        return sample_indices, neighbor_indices, alive

    def _compute_neighbor_batch(
        self,
        start_idx: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Dispatch neighbor selection based on config."""
        if self.config.neighbor_method == "uniform":
            return self._compute_companion_batch(start_idx)
        if self.config.neighbor_method == "recorded":
            return self._compute_recorded_neighbors_batch(start_idx)
        if self.config.neighbor_method == "voronoi":
            return self._compute_voronoi_batch(start_idx)
        return self._compute_knn_batch(start_idx)

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

        Returns:
            Series [T] of operator values (MC time) or [n_bins] (Euclidean time).
        """
        if self.config.time_axis == "euclidean":
            # Euclidean time analysis: bin operators by spatial time coordinate
            return self._compute_series_euclidean()
        else:
            # Standard Monte Carlo time analysis
            return self._compute_series_mc()
    
    def _compute_series_mc(self) -> Tensor:
        """Compute operator time series over Monte Carlo timesteps."""
        start_idx = max(1, int(self.history.n_recorded * self.config.warmup_fraction))

        # Batch compute color states
        color, valid = self._compute_color_states_batch(start_idx)

        # Batch compute neighbors
        sample_indices, neighbor_indices, alive = self._compute_neighbor_batch(start_idx)

        # Compute operators (implemented by subclass)
        series = self._compute_operators_vectorized(
            color, valid, alive, sample_indices, neighbor_indices
        )

        return series
    
    def _compute_series_euclidean(self) -> Tensor:
        """Compute operator time series over Euclidean time coordinate.

        This method bins walkers by their Euclidean time coordinate and computes
        operator averages within each time bin.
        """
        # Check dimension
        if self.history.d < self.config.euclidean_time_dim + 1:
            msg = (
                f"Cannot use dimension {self.config.euclidean_time_dim} as Euclidean time "
                f"(only {self.history.d} dimensions available)"
            )
            raise ValueError(msg)
        
        start_idx = _resolve_mc_time_index(self.history, self.config.mc_time_index)

        # Get positions for Euclidean time extraction
        positions = self.history.x_before_clone[start_idx : start_idx + 1]  # [1, N, d]
        alive = self.history.alive_mask[start_idx - 1 : start_idx]  # [1, N]

        # Compute operators for all walkers (not averaged)
        operators = self._compute_operators_per_walker(start_idx)[:1]  # [1, N]

        # Bin by Euclidean time
        time_coords, series = bin_by_euclidean_time(
            positions=positions,
            operators=operators,
            alive=alive,
            time_dim=self.config.euclidean_time_dim,
            n_bins=self.config.euclidean_time_bins,
            time_range=self.config.euclidean_time_range,
        )
        
        return series
    
    def _compute_operators_per_walker(self, start_idx: int) -> Tensor:
        """Compute operators for each walker (not averaged).
        
        Args:
            start_idx: Starting time index.
            
        Returns:
            Operators [T, N] for each walker at each timestep.
        """
        # Batch compute color states
        color, valid = self._compute_color_states_batch(start_idx)
        
        # Get alive mask
        n_recorded = self.history.n_recorded
        alive = self.history.alive_mask[start_idx - 1 : n_recorded - 1]  # [T, N]
        
        # Compute operators without averaging (subclass-specific)
        return self._compute_operators_all_walkers(color, valid, alive)

    def compute_correlator(self) -> Tensor:
        """Compute time correlator using FFT.

        Returns:
            Correlator C(t) [max_lag+1].
        """
        series = self.compute_series()
        return compute_correlator_fft(
            series.real if series.is_complex() else series,
            max_lag=self.config.max_lag,
            use_connected=self.config.use_connected,
        )

    def extract_mass_aic(self, correlator: Tensor) -> dict[str, Any]:
        """Extract mass using convolutional AIC.

        Args:
            correlator: Correlator C(t) [max_lag+1].

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
            window_widths=self.config.window_widths,
            min_mass=self.config.min_mass,
            max_mass=self.config.max_mass,
        )

        return extractor.fit_all_widths(log_corr, log_err)

    def extract_mass_linear(self, correlator: Tensor) -> dict[str, Any]:
        """Extract mass using a simple linear fit on log(C(t))."""
        if correlator.numel() == 0:
            return {
                "mass": 0.0,
                "amplitude": 0.0,
                "r_squared": 0.0,
                "fit_points": 0.0,
            }

        n = correlator.shape[0]
        fit_start = max(0, int(self.config.fit_start))
        fit_stop = self.config.fit_stop
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
        if n_points < max(2, int(self.config.min_fit_points)):
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

    def extract_mass_linear_abs(self, correlator: Tensor) -> dict[str, Any]:
        """Extract mass using a linear fit on log(|C(t)|)."""
        return self.extract_mass_linear(correlator.abs())
    def compute(self) -> ChannelCorrelatorResult:
        """Compute full channel analysis.

        Returns:
            ChannelCorrelatorResult with all computed quantities.
        """
        series = self.compute_series()

        if series.numel() == 0:
            return ChannelCorrelatorResult(
                channel_name=self.channel_name,
                correlator=torch.zeros(self.config.max_lag + 1),
                correlator_err=None,
                effective_mass=torch.zeros(self.config.max_lag),
                mass_fit={"mass": 0.0, "mass_error": float("inf")},
                series=series,
                n_samples=0,
                dt=float(self.history.delta_t * self.history.record_every),
                window_masses=None,
                window_aic=None,
                window_widths=None,
                window_r2=None,
            )

        correlator = compute_correlator_fft(
            series.real if series.is_complex() else series,
            max_lag=self.config.max_lag,
            use_connected=self.config.use_connected,
        )

        dt = float(self.history.delta_t * self.history.record_every)
        effective_mass = compute_effective_mass_torch(correlator, dt)
        if self.config.fit_mode == "linear_abs":
            mass_fit = self.extract_mass_linear_abs(correlator)
            window_masses = None
            window_aic = None
            window_widths = None
            window_r2 = None
        elif self.config.fit_mode == "linear":
            mass_fit = self.extract_mass_linear(correlator)
            window_masses = None
            window_aic = None
            window_widths = None
            window_r2 = None
        else:
            mass_fit = self.extract_mass_aic(correlator)
            window_masses = mass_fit.pop("window_masses", None)
            window_aic = mass_fit.pop("window_aic", None)
            window_widths = mass_fit.pop("window_widths", None)
            window_r2 = mass_fit.pop("window_r2", None)

        return ChannelCorrelatorResult(
            channel_name=self.channel_name,
            correlator=correlator,
            correlator_err=None,
            effective_mass=effective_mass,
            mass_fit=mass_fit,
            series=series,
            n_samples=int(series.numel()),
            dt=dt,
            window_masses=window_masses,
            window_aic=window_aic,
            window_widths=window_widths,
            window_r2=window_r2,
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

    def _compute_operators_all_walkers(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
    ) -> Tensor:
        """Compute bilinear operators for all walkers without averaging.

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].

        Returns:
            Operator values [T, N] for each walker.
        """
        T, N, d = color.shape
        device = color.device

        # Get neighbors for all walkers at all timesteps
        neighbor_indices = self._get_all_neighbors(color, valid, alive)  # [T, N, k]

        # Use first neighbor for each walker
        first_neighbor = neighbor_indices[:, :, 0]  # [T, N]

        # Gather color states
        t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)  # [T, N]
        color_i = color  # [T, N, d] (walkers themselves)
        color_j = color[t_idx, first_neighbor]  # [T, N, d] (their neighbors)

        # Validity masks
        valid_i = valid & alive
        valid_j_idx = t_idx.clamp(max=alive.shape[0] - 1)
        valid_j = valid[t_idx, first_neighbor] & alive[valid_j_idx, first_neighbor]
        valid_mask = valid_i & valid_j & (first_neighbor != torch.arange(N, device=device).unsqueeze(0))

        # Apply channel-specific projection
        op_values = self._apply_gamma_projection(color_i, color_j)  # [T, N]

        # Mask invalid
        op_values = torch.where(valid_mask, op_values, torch.zeros_like(op_values))

        return op_values

    def _get_all_neighbors(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
    ) -> Tensor:
        """Get neighbor indices for all walkers at all timesteps.

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].

        Returns:
            Neighbor indices [T, N, k] where k is number of neighbors.
        """
        T, N, d = color.shape
        device = color.device
        k = self.config.knn_k

        neighbor_indices = torch.zeros((T, N, k), dtype=torch.long, device=device)

        for t in range(T):
            # Get neighbors for this timestep
            alive_t = alive[min(t, alive.shape[0] - 1)]

            if self.config.neighbor_method == "euclidean":
                # Euclidean distance neighbors
                x_t = self.history.x_before_clone[t + max(1, int(self.history.n_recorded * self.config.warmup_fraction))]
                dists = torch.cdist(x_t, x_t)  # [N, N]
                dists = dists + torch.eye(N, device=device) * 1e10  # Exclude self
                dists[:, ~alive_t] = 1e10  # Exclude dead walkers
                neighbor_indices[t] = torch.topk(dists, k, largest=False, dim=1).indices
            else:
                # Voronoi neighbors (not implemented for per-walker)
                # For now, fall back to random alive walkers
                alive_idx = torch.where(alive_t)[0]
                if len(alive_idx) > 0:
                    for i in range(N):
                        if alive_t[i]:
                            # Get k random alive neighbors (excluding self)
                            other_alive = alive_idx[alive_idx != i]
                            if len(other_alive) >= k:
                                perm = torch.randperm(len(other_alive), device=device)[:k]
                                neighbor_indices[t, i] = other_alive[perm]
                            elif len(other_alive) > 0:
                                # Repeat if not enough neighbors
                                neighbor_indices[t, i] = other_alive[torch.randint(0, len(other_alive), (k,), device=device)]
                            else:
                                neighbor_indices[t, i] = i  # Self-neighbor if no others

        return neighbor_indices


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

    Requires d=3.

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

        if d != 3:
            # Nucleon requires d=3
            return torch.zeros(T, device=device)

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

    def _compute_operators_all_walkers(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
    ) -> Tensor:
        """Compute nucleon operators for all walkers without averaging.

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].

        Returns:
            Operator values [T, N] for each walker.
        """
        T, N, d = color.shape
        device = color.device

        if d != 3:
            # Nucleon requires d=3
            return torch.zeros((T, N), device=device)

        # Get neighbors for all walkers
        neighbor_indices = self._get_all_neighbors_nucleon(color, valid, alive)  # [T, N, k]

        if neighbor_indices.shape[2] < 2:
            return torch.zeros((T, N), device=device)

        # Gather indices
        t_idx = torch.arange(T, device=device).unsqueeze(1).expand(-1, N)

        # Color states for triplets
        color_i = color  # [T, N, d]
        color_j = color[t_idx, neighbor_indices[:, :, 0]]  # [T, N, d]
        color_k = color[t_idx, neighbor_indices[:, :, 1]]  # [T, N, d]

        # Stack to form 3x3 matrix: [T, N, d, 3]
        matrix = torch.stack([color_i, color_j, color_k], dim=-1)

        # Compute determinant: [T, N]
        det = torch.linalg.det(matrix)

        # Validity mask
        valid_i = valid & alive
        valid_j_idx = t_idx.clamp(max=alive.shape[0] - 1)
        valid_j = valid[t_idx, neighbor_indices[:, :, 0]] & alive[valid_j_idx, neighbor_indices[:, :, 0]]
        valid_k = valid[t_idx, neighbor_indices[:, :, 1]] & alive[valid_j_idx, neighbor_indices[:, :, 1]]
        valid_mask = valid_i & valid_j & valid_k

        # Mask invalid
        det = torch.where(valid_mask, det, torch.zeros_like(det))

        return det.real if det.is_complex() else det

    def _get_all_neighbors_nucleon(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
    ) -> Tensor:
        """Get neighbor indices for all walkers (nucleon needs 2 neighbors).

        Args:
            color: Color states [T, N, d].
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].

        Returns:
            Neighbor indices [T, N, 2] for first two neighbors.
        """
        T, N, d = color.shape
        device = color.device
        k = max(2, self.config.n_neighbors)

        neighbor_indices = torch.zeros((T, N, k), dtype=torch.long, device=device)

        for t in range(T):
            alive_t = alive[min(t, alive.shape[0] - 1)]

            if self.config.neighbor_method == "euclidean":
                # Euclidean distance neighbors
                start_idx = max(1, int(self.history.n_recorded * self.config.warmup_fraction))
                x_t = self.history.x_before_clone[min(t + start_idx, len(self.history.x_before_clone) - 1)]
                dists = torch.cdist(x_t, x_t)  # [N, N]
                dists = dists + torch.eye(N, device=device) * 1e10  # Exclude self
                dists[:, ~alive_t] = 1e10  # Exclude dead walkers
                neighbor_indices[t] = torch.topk(dists, k, largest=False, dim=1).indices
            else:
                # Fall back to random alive walkers
                alive_idx = torch.where(alive_t)[0]
                if len(alive_idx) > 0:
                    for i in range(N):
                        if alive_t[i]:
                            other_alive = alive_idx[alive_idx != i]
                            if len(other_alive) >= k:
                                perm = torch.randperm(len(other_alive), device=device)[:k]
                                neighbor_indices[t, i] = other_alive[perm]
                            elif len(other_alive) > 0:
                                neighbor_indices[t, i] = other_alive[torch.randint(0, len(other_alive), (k,), device=device)]
                            else:
                                neighbor_indices[t, i] = i

        return neighbor_indices


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

    def _compute_operators_all_walkers(
        self,
        color: Tensor,
        valid: Tensor,
        alive: Tensor,
    ) -> Tensor:
        """Compute glueball operators for all walkers without averaging.

        Args:
            color: Color states [T, N, d] (unused for glueball).
            valid: Valid color flags [T, N].
            alive: Alive walker flags [T, N].

        Returns:
            Operator values [T, N] for each walker.
        """
        start_idx = max(1, int(self.history.n_recorded * self.config.warmup_fraction))
        n_recorded = self.history.n_recorded

        # Get force field
        force = self.history.force_viscous[start_idx - 1 : n_recorded - 1]  # [T, N, d]

        # Force squared norm: [T, N]
        force_sq = torch.linalg.vector_norm(force, dim=-1).pow(2)

        # Mask invalid walkers
        force_sq = torch.where(alive, force_sq, torch.zeros_like(force_sq))

        return force_sq


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
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators for multiple channels.

    Args:
        history: Fractal Gas run history.
        channels: List of channel names (default: all registered).
        config: Configuration parameters.

    Returns:
        Dictionary mapping channel names to results.
    """
    if channels is None:
        channels = list(CHANNEL_REGISTRY.keys())

    config = config or ChannelConfig()
    results = {}

    for channel_name in channels:
        if channel_name not in CHANNEL_REGISTRY:
            continue

        channel_class = CHANNEL_REGISTRY[channel_name]
        try:
            correlator = channel_class(history, config)
            results[channel_name] = correlator.compute()
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
                dt=float(history.delta_t * history.record_every),
                window_masses=None,
                window_aic=None,
                window_widths=None,
                window_r2=None,
            )

    return results


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
