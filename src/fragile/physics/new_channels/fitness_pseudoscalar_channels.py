"""Fitness pseudoscalar channel correlators.

Extracts correlators from log-fitness fluctuations weighted by dominant/subordinate
parity (cloning score sign).  This measures whether dominant walkers (S <= 0) have
systematically different log-fitness fluctuations than subordinate walkers (S > 0).

Three correlators:
- C_PP(tau): pseudoscalar-pseudoscalar (pion mass)
- C_SS(tau): scalar variance (radial mode mass)
- C_JP(tau): axial current x pseudoscalar cross-correlator (PCAC mass check)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.qft_utils import _fft_correlator_batched, resolve_frame_indices


@dataclass
class FitnessPseudoscalarConfig:
    """Configuration for fitness pseudoscalar correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    fitness_floor: float = 1e-30


@dataclass
class FitnessPseudoscalarOutput:
    """Fitness pseudoscalar correlator output and diagnostics."""

    # PP correlator [max_lag+1]
    cpp: Tensor
    cpp_raw: Tensor
    cpp_connected: Tensor

    # Scalar variance correlator [max_lag+1]
    css: Tensor
    css_raw: Tensor
    css_connected: Tensor

    # JP cross-correlator [max_lag+1]
    cjp: Tensor
    cjp_raw: Tensor
    cjp_connected: Tensor

    # PCAC mass [max_lag+1], NaN at tau=0
    pcac_mass: Tensor

    # Operator time series
    operator_pseudoscalar_series: Tensor  # P(t) [T]
    operator_scalar_variance_series: Tensor  # Sigma(t) [T]
    operator_axial_series: Tensor  # J(t) [T-1]

    frame_indices: list[int]
    n_valid_frames: int
    mean_pseudoscalar: float
    mean_scalar_variance: float


def _fft_cross_correlator(
    a: Tensor,
    b: Tensor,
    max_lag: int,
    use_connected: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Asymmetric cross-correlator C_ab(tau) = <a(t) * b(t+tau)>.

    Uses FFT: IFFT(conj(FFT(a)) * FFT(b)).real with zero-padding.

    Args:
        a: First series [T].
        b: Second series [T].  Must have same length as *a*.
        max_lag: Maximum lag to compute.
        use_connected: Subtract means before correlation.

    Returns:
        (correlator, raw, connected) each [max_lag+1].
    """
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"Expected 1D tensors, got shapes {tuple(a.shape)} and {tuple(b.shape)}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Series must have same length, got {a.shape[0]} and {b.shape[0]}"
        )

    T = a.shape[0]
    a_work = a.float()
    b_work = b.float()
    mean_a = a_work.mean()
    mean_b = b_work.mean()

    def _compute(x: Tensor, y: Tensor) -> Tensor:
        effective_lag = min(max_lag, T - 1)
        n_fft = T + effective_lag + 1
        n_fft = 1 << (n_fft - 1).bit_length()
        x_pad = F.pad(x.unsqueeze(0), (0, n_fft - T)).squeeze(0)
        y_pad = F.pad(y.unsqueeze(0), (0, n_fft - T)).squeeze(0)
        fft_x = torch.fft.fft(x_pad)
        fft_y = torch.fft.fft(y_pad)
        corr = torch.fft.ifft(fft_x.conj() * fft_y).real
        counts = torch.arange(
            T, T - effective_lag - 1, -1, device=a.device, dtype=torch.float32
        )
        result = corr[:effective_lag + 1] / counts
        if effective_lag < max_lag:
            result = F.pad(result, (0, max_lag - effective_lag), value=0.0)
        return result

    raw = _compute(a_work, b_work)
    connected = _compute(a_work - mean_a, b_work - mean_b)
    correlator = connected if use_connected else raw
    return correlator, raw, connected


def compute_fitness_pseudoscalar_from_data(
    fitness: Tensor,
    cloning_scores: Tensor,
    alive_mask: Tensor,
    *,
    max_lag: int = 80,
    use_connected: bool = True,
    fitness_floor: float = 1e-30,
    frame_indices: list[int] | None = None,
) -> FitnessPseudoscalarOutput:
    """Compute fitness pseudoscalar correlators from raw data tensors.

    Args:
        fitness: Fitness values [T, N].
        cloning_scores: Cloning scores [T, N].
        alive_mask: Boolean alive mask [T, N].
        max_lag: Maximum lag.
        use_connected: Use connected correlators.
        fitness_floor: Floor for fitness before log.
        frame_indices: Frame indices for diagnostics.

    Returns:
        FitnessPseudoscalarOutput with all correlators and diagnostics.
    """
    T, N = fitness.shape
    device = fitness.device

    if frame_indices is None:
        frame_indices = list(range(T))

    # Empty output helper
    def _empty_output() -> FitnessPseudoscalarOutput:
        empty = torch.zeros(max_lag + 1, device=device)
        pcac = torch.full((max_lag + 1,), float("nan"), device=device)
        return FitnessPseudoscalarOutput(
            cpp=empty, cpp_raw=empty.clone(), cpp_connected=empty.clone(),
            css=empty.clone(), css_raw=empty.clone(), css_connected=empty.clone(),
            cjp=empty.clone(), cjp_raw=empty.clone(), cjp_connected=empty.clone(),
            pcac_mass=pcac,
            operator_pseudoscalar_series=torch.zeros(0, device=device),
            operator_scalar_variance_series=torch.zeros(0, device=device),
            operator_axial_series=torch.zeros(0, device=device),
            frame_indices=frame_indices,
            n_valid_frames=0,
            mean_pseudoscalar=0.0,
            mean_scalar_variance=0.0,
        )

    if T < 2:
        return _empty_output()

    alive = alive_mask.bool()
    n_alive = alive.float().sum(dim=1)  # [T]
    n_alive_safe = n_alive.clamp(min=1.0)

    # 1. log_v = log(clamp(fitness, min=fitness_floor)) [T, N]
    log_v = torch.log(fitness.float().clamp(min=fitness_floor))

    # 2. mean_log_v: per-frame alive-weighted mean [T]
    log_v_masked = log_v * alive.float()
    mean_log_v = log_v_masked.sum(dim=1) / n_alive_safe  # [T]

    # 3. delta = log_v - mean_log_v [T, N]
    delta = log_v - mean_log_v.unsqueeze(1)

    # 4. sigma = where(scores <= 0, +1, -1) [T, N]
    sigma = torch.where(cloning_scores.float() <= 0, 1.0, -1.0)

    # 5. P(t) = mean(sigma * delta) over alive walkers [T]
    P = (sigma * delta * alive.float()).sum(dim=1) / n_alive_safe  # [T]

    # 6. Sigma(t) = mean(delta^2) over alive walkers [T]
    Sigma = (delta**2 * alive.float()).sum(dim=1) / n_alive_safe  # [T]

    # 7. J(t) = P(t+1) - P(t) [T-1]
    J = P[1:] - P[:-1]

    # Correlators via FFT
    # C_PP: pseudoscalar autocorrelation
    cpp_result = _fft_correlator_batched(P.unsqueeze(0), max_lag, use_connected=False)
    cpp_raw = cpp_result.squeeze(0)
    cpp_conn_result = _fft_correlator_batched(P.unsqueeze(0), max_lag, use_connected=True)
    cpp_connected = cpp_conn_result.squeeze(0)
    cpp = cpp_connected if use_connected else cpp_raw

    # C_SS: scalar variance autocorrelation
    css_result = _fft_correlator_batched(Sigma.unsqueeze(0), max_lag, use_connected=False)
    css_raw = css_result.squeeze(0)
    css_conn_result = _fft_correlator_batched(Sigma.unsqueeze(0), max_lag, use_connected=True)
    css_connected = css_conn_result.squeeze(0)
    css = css_connected if use_connected else css_raw

    # C_JP: axial current x pseudoscalar cross-correlator
    # J and P[:-1] have same length T-1
    P_trunc = P[:-1]
    cjp, cjp_raw, cjp_connected = _fft_cross_correlator(J, P_trunc, max_lag, use_connected)

    # PCAC mass: (C_JP(tau) - C_JP(tau-1)) / (2 * C_PP(tau)) for tau >= 1
    pcac_mass = torch.full((max_lag + 1,), float("nan"), device=device)
    if max_lag >= 1:
        delta_cjp = cjp[1:] - cjp[:-1]  # [max_lag]
        cpp_denom = 2.0 * cpp[1:]  # [max_lag]
        # Avoid division by zero
        safe_denom = cpp_denom.clone()
        safe_denom[safe_denom.abs() < 1e-30] = float("nan")
        pcac_mass[1:] = delta_cjp / safe_denom

    return FitnessPseudoscalarOutput(
        cpp=cpp,
        cpp_raw=cpp_raw,
        cpp_connected=cpp_connected,
        css=css,
        css_raw=css_raw,
        css_connected=css_connected,
        cjp=cjp,
        cjp_raw=cjp_raw,
        cjp_connected=cjp_connected,
        pcac_mass=pcac_mass,
        operator_pseudoscalar_series=P,
        operator_scalar_variance_series=Sigma,
        operator_axial_series=J,
        frame_indices=frame_indices,
        n_valid_frames=T,
        mean_pseudoscalar=float(P.mean().item()),
        mean_scalar_variance=float(Sigma.mean().item()),
    )


def compute_fitness_pseudoscalar_correlator(
    history: RunHistory,
    config: FitnessPseudoscalarConfig | None = None,
) -> FitnessPseudoscalarOutput:
    """Compute fitness pseudoscalar correlators from a RunHistory.

    Args:
        history: Run history with fitness, cloning_scores, alive_mask.
        config: Configuration; None uses defaults.

    Returns:
        FitnessPseudoscalarOutput with all correlators and diagnostics.
    """
    if config is None:
        config = FitnessPseudoscalarConfig()

    frame_indices = resolve_frame_indices(
        history,
        warmup_fraction=config.warmup_fraction,
        end_fraction=config.end_fraction,
    )

    max_lag = int(config.max_lag)
    device = history.fitness.device

    if not frame_indices:
        empty = torch.zeros(max_lag + 1, device=device)
        pcac = torch.full((max_lag + 1,), float("nan"), device=device)
        return FitnessPseudoscalarOutput(
            cpp=empty, cpp_raw=empty.clone(), cpp_connected=empty.clone(),
            css=empty.clone(), css_raw=empty.clone(), css_connected=empty.clone(),
            cjp=empty.clone(), cjp_raw=empty.clone(), cjp_connected=empty.clone(),
            pcac_mass=pcac,
            operator_pseudoscalar_series=torch.zeros(0, device=device),
            operator_scalar_variance_series=torch.zeros(0, device=device),
            operator_axial_series=torch.zeros(0, device=device),
            frame_indices=[],
            n_valid_frames=0,
            mean_pseudoscalar=0.0,
            mean_scalar_variance=0.0,
        )

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1

    # Same indexing as meson_phase_channels: [start_idx-1 : end_idx-1]
    fitness = torch.as_tensor(
        history.fitness[start_idx - 1 : end_idx - 1],
        dtype=torch.float32,
        device=device,
    )
    cloning_scores = torch.as_tensor(
        history.cloning_scores[start_idx - 1 : end_idx - 1],
        dtype=torch.float32,
        device=device,
    )
    alive_mask = torch.as_tensor(
        history.alive_mask[start_idx - 1 : end_idx - 1],
        dtype=torch.bool,
        device=device,
    )

    return compute_fitness_pseudoscalar_from_data(
        fitness=fitness,
        cloning_scores=cloning_scores,
        alive_mask=alive_mask,
        max_lag=max_lag,
        use_connected=bool(config.use_connected),
        fitness_floor=float(config.fitness_floor),
        frame_indices=frame_indices,
    )
