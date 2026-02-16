"""FFT-based correlator computation.

Ported from fragile.fractalai.qft.correlator_channels._fft_correlator_batched.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _fft_correlator_batched(
    series: Tensor,
    max_lag: int,
    use_connected: bool = True,
) -> Tensor:
    """Compute time correlator for a batch of series using FFT.

    Args:
        series: Operator time series [B, T].
        max_lag: Maximum lag to compute.
        use_connected: Subtract per-series mean (connected correlator).

    Returns:
        Correlator C(t) for t=0 to max_lag [B, max_lag+1].
    """
    if series.ndim != 2:
        raise ValueError(f"Expected 2D tensor [B, T], got shape {tuple(series.shape)}")

    if series.numel() == 0:
        return torch.zeros(series.shape[0], max_lag + 1, device=series.device, dtype=series.dtype)

    _, T = series.shape
    work = series.float()
    if use_connected:
        work = work - work.mean(dim=1, keepdim=True)

    # Zero-pad each sample for FFT convolution along the time axis.
    padded = F.pad(work, (0, T))  # [B, 2T]
    fft_s = torch.fft.fft(padded, dim=1)
    corr = torch.fft.ifft(fft_s * fft_s.conj(), dim=1).real

    # Normalize by number of overlapping samples.
    effective_lag = min(max_lag, T - 1)
    counts = torch.arange(T, T - effective_lag - 1, -1, device=series.device, dtype=torch.float32)
    result = corr[:, : effective_lag + 1] / counts.unsqueeze(0)

    # Pad with zeros if max_lag > T-1.
    if effective_lag < max_lag:
        result = F.pad(result, (0, max_lag - effective_lag), value=0.0)

    return result
