"""Spectral gap estimation for fractal gas simulations.

Provides multiple methods to estimate the spectral gap of the walker dynamics,
which controls mixing time and correlation length.  A larger spectral gap
indicates faster mixing and better ergodic sampling.

Methods
-------
1. **Graph Laplacian (Fiedler value)** — second-smallest eigenvalue of the
   normalised graph Laplacian built from the Delaunay tessellation.
2. **Autocorrelation decay** — fit an exponential to the autocorrelation of
   an observable (position spread, energy, …) to extract the integrated
   autocorrelation time τ; then gap ≈ 1/τ.
3. **Transfer-matrix proxy** — ratio C(t+1)/C(t) of the position-position
   autocorrelation gives an effective mass / gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor


if TYPE_CHECKING:
    from fragile.physics.fractal_gas.history import RunHistory


@dataclass
class SpectralGapResult:
    """Container for spectral gap estimates from all methods."""

    fiedler_value: float
    """Graph Laplacian spectral gap (median over sampled frames)."""

    fiedler_std: float
    """Standard deviation of Fiedler values across sampled frames."""

    autocorrelation_gap: float
    """Gap from integrated autocorrelation time: 1 / τ_int."""

    autocorrelation_tau: float
    """Integrated autocorrelation time (in recorded frames)."""

    transfer_matrix_gap: float
    """Effective gap from log-ratio of successive correlator values."""

    transfer_matrix_gap_std: float
    """Standard deviation of the transfer-matrix gap estimate."""


# ---------------------------------------------------------------------------
# Method 1: Graph Laplacian (Fiedler value)
# ---------------------------------------------------------------------------


def _graph_laplacian_fiedler(
    positions: Tensor,
    edges: Tensor,
    edge_weights: Tensor | None = None,
) -> float:
    """Compute the Fiedler value (algebraic connectivity) of the graph Laplacian.

    The Fiedler value λ₂ is the second-smallest eigenvalue of the normalised
    graph Laplacian L = I − D⁻¹/²AD⁻¹/².  It measures how well-connected the
    graph is; larger λ₂ means faster diffusion / mixing.

    Args:
        positions: Walker positions [N, d] (unused but kept for API symmetry).
        edges: Directed edge list [E, 2].
        edge_weights: Optional weights [E].  Defaults to uniform.

    Returns:
        Fiedler value (λ₂).  Returns 0.0 if the graph is disconnected or
        has fewer than 3 nodes.
    """
    N = int(positions.shape[0])
    if N < 3 or edges.numel() == 0:
        return 0.0

    device = positions.device
    src, dst = edges[:, 0].long(), edges[:, 1].long()

    if edge_weights is None:
        w = torch.ones(src.shape[0], device=device, dtype=positions.dtype)
    else:
        w = edge_weights.to(positions.dtype)

    # Build adjacency matrix (symmetric)
    A = torch.zeros(N, N, device=device, dtype=positions.dtype)
    A[src, dst] += w
    # Symmetrise
    A = (A + A.T) / 2

    # Degree
    deg = A.sum(dim=1)
    disconnected = deg < 1e-12
    if disconnected.all():
        return 0.0
    deg = deg.clamp(min=1e-12)

    # Normalised Laplacian: L = I - D^{-1/2} A D^{-1/2}
    inv_sqrt_deg = 1.0 / deg.sqrt()
    L_norm = torch.eye(N, device=device, dtype=positions.dtype) - (
        inv_sqrt_deg.unsqueeze(1) * A * inv_sqrt_deg.unsqueeze(0)
    )

    # Eigenvalues (symmetric → real)
    eigvals = torch.linalg.eigvalsh(L_norm)
    # λ₂ is the second-smallest
    sorted_eigvals = eigvals.sort().values
    if sorted_eigvals.numel() < 2:
        return 0.0
    return max(0.0, float(sorted_eigvals[1].item()))


def estimate_fiedler_from_history(
    history: RunHistory,
    n_frames: int = 8,
    warmup_fraction: float = 0.15,
) -> tuple[float, float]:
    """Estimate graph Laplacian spectral gap from multiple frames.

    Samples ``n_frames`` evenly-spaced frames after warmup, computes the
    Fiedler value of the Delaunay neighbor graph at each, and returns
    (median, std).

    Returns (0.0, 0.0) if neighbor graph data is not available.
    """
    if history.neighbor_edges is None or len(history.neighbor_edges) == 0:
        return 0.0, 0.0

    n_rec = history.n_recorded
    start = max(1, int(n_rec * warmup_fraction))
    # neighbor_edges is indexed by idx-1 (one fewer than recorded frames)
    n_edge_frames = len(history.neighbor_edges)
    if start - 1 >= n_edge_frames:
        return 0.0, 0.0

    frame_pool = list(range(start - 1, min(n_edge_frames, n_rec - 1)))
    if not frame_pool:
        return 0.0, 0.0

    step = max(1, len(frame_pool) // n_frames)
    sampled = frame_pool[::step][:n_frames]

    fiedler_vals = []
    for fi in sampled:
        edges = history.neighbor_edges[fi]
        if edges is None or edges.numel() == 0:
            continue
        # Use positions after cloning (which the tessellation was built on)
        pos = history.x_after_clone[fi]

        # Try to get edge weights
        ew = None
        if history.edge_weights is not None and fi < len(history.edge_weights):
            wd = history.edge_weights[fi]
            if isinstance(wd, dict):
                # Prefer riemannian_kernel_volume, fallback to first available
                for key in ("riemannian_kernel_volume", "kernel", "inverse_riemannian_distance"):
                    if key in wd:
                        ew = wd[key]
                        break
                if ew is None and wd:
                    ew = next(iter(wd.values()))
            elif isinstance(wd, Tensor):
                ew = wd

        val = _graph_laplacian_fiedler(pos, edges, ew)
        fiedler_vals.append(val)

    if not fiedler_vals:
        return 0.0, 0.0
    arr = np.array(fiedler_vals)
    return float(np.median(arr)), float(np.std(arr))


# ---------------------------------------------------------------------------
# Method 2: Autocorrelation decay
# ---------------------------------------------------------------------------


def _autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Normalised autocorrelation of a 1-D signal via FFT."""
    n = len(x)
    if n < 2:
        return np.array([1.0])
    x = x - x.mean()
    var = np.var(x)
    if var < 1e-30:
        return np.ones(min(n, max_lag or n))

    # Zero-pad to avoid circular correlation
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    fft_x = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(fft_x * np.conj(fft_x), n=fft_size)[:n]
    acf /= acf[0]

    if max_lag is not None:
        acf = acf[: max_lag + 1]
    return acf


def estimate_autocorrelation_gap(
    history: RunHistory,
    warmup_fraction: float = 0.15,
    max_lag_fraction: float = 0.5,
) -> tuple[float, float]:
    """Estimate spectral gap from integrated autocorrelation time.

    Uses the position-spread observable R²(t) = mean_i(|x_i - x̄|²) as the
    primary signal.  The integrated autocorrelation time is computed with
    automatic windowing (Sokal's method: truncate when τ_int > lag/6).

    Returns:
        (gap, tau_int) where gap = 1/tau_int.  Returns (0, inf) if
        estimation fails.
    """
    n_rec = history.n_recorded
    start = max(1, int(n_rec * warmup_fraction))
    if n_rec - start < 10:
        return 0.0, float("inf")

    # Observable: position spread per frame
    x = history.x_before_clone[start:n_rec]  # [T, N, d]
    centroid = x.mean(dim=1, keepdim=True)  # [T, 1, d]
    r2 = ((x - centroid) ** 2).sum(dim=-1).mean(dim=1)  # [T]
    signal = r2.cpu().numpy().astype(np.float64)

    max_lag = max(5, int(len(signal) * max_lag_fraction))
    acf = _autocorrelation(signal, max_lag=max_lag)

    # Integrated autocorrelation time with Sokal windowing
    tau_int = 0.5  # contribution from lag 0 is 0.5 * acf[0] = 0.5
    for lag in range(1, len(acf)):
        if acf[lag] <= 0:
            break
        tau_int += acf[lag]
        # Sokal's criterion: stop when window > 6*tau_int
        if lag > 6 * tau_int:
            break

    tau_int = max(tau_int, 0.5)
    gap = 1.0 / tau_int
    return float(gap), float(tau_int)


# ---------------------------------------------------------------------------
# Method 3: Transfer-matrix proxy (effective mass)
# ---------------------------------------------------------------------------


def estimate_transfer_matrix_gap(
    history: RunHistory,
    warmup_fraction: float = 0.15,
) -> tuple[float, float]:
    """Estimate spectral gap from log-ratio of position autocorrelation.

    Computes C(t) = <x(0)·x(t)> (spatially averaged), then the effective
    gap m_eff(t) = -log(C(t+1)/C(t)).  Returns (median, std) of m_eff over
    the plateau region.

    Returns:
        (gap, gap_std).  Returns (0, 0) if estimation fails.
    """
    n_rec = history.n_recorded
    start = max(1, int(n_rec * warmup_fraction))
    T = n_rec - start
    if T < 20:
        return 0.0, 0.0

    x = history.x_before_clone[start:n_rec]  # [T, N, d]
    # Flatten to [T, N*d] for correlation
    x_flat = x.reshape(T, -1).float()
    x_flat = x_flat - x_flat.mean(dim=0, keepdim=True)

    # C(t) = <x(0) · x(t)> averaged over reference times
    max_lag = min(T // 3, 100)
    if max_lag < 5:
        return 0.0, 0.0

    correlator = torch.zeros(max_lag, device=x.device, dtype=torch.float32)
    n_avg = T - max_lag
    for lag in range(max_lag):
        # Average over all starting points
        dot = (x_flat[:n_avg] * x_flat[lag : lag + n_avg]).sum(dim=1)
        correlator[lag] = dot.mean()

    correlator = correlator.cpu().numpy()

    # Normalise
    if abs(correlator[0]) < 1e-30:
        return 0.0, 0.0
    correlator = correlator / correlator[0]

    # Effective mass: m_eff(t) = -log(C(t+1) / C(t))
    m_eff = []
    for t in range(len(correlator) - 1):
        if correlator[t] > 1e-12 and correlator[t + 1] > 1e-12:
            ratio = correlator[t + 1] / correlator[t]
            if 0 < ratio < 1:
                m_eff.append(-np.log(ratio))

    if len(m_eff) < 3:
        return 0.0, 0.0

    arr = np.array(m_eff)
    # Use the latter half as plateau region
    plateau = arr[len(arr) // 3 :]
    if len(plateau) < 2:
        plateau = arr

    return float(np.median(plateau)), float(np.std(plateau))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_spectral_gap(
    history: RunHistory,
    warmup_fraction: float = 0.15,
) -> SpectralGapResult:
    """Compute spectral gap estimates using all available methods.

    Args:
        history: RunHistory from a completed simulation run.
        warmup_fraction: Fraction of initial frames to discard.

    Returns:
        SpectralGapResult with estimates from each method.
    """
    fiedler, fiedler_std = estimate_fiedler_from_history(
        history,
        warmup_fraction=warmup_fraction,
    )
    ac_gap, ac_tau = estimate_autocorrelation_gap(
        history,
        warmup_fraction=warmup_fraction,
    )
    tm_gap, tm_std = estimate_transfer_matrix_gap(
        history,
        warmup_fraction=warmup_fraction,
    )

    return SpectralGapResult(
        fiedler_value=fiedler,
        fiedler_std=fiedler_std,
        autocorrelation_gap=ac_gap,
        autocorrelation_tau=ac_tau,
        transfer_matrix_gap=tm_gap,
        transfer_matrix_gap_std=tm_std,
    )
