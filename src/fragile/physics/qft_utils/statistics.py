"""Time-origin statistics for correlation estimates.

Statistics travel with the returned tensor through ``correlator_statistics``.
Tensor arithmetic does not preserve this metadata: use ``stack_correlators``
for scales. Consumers validate that the statistics reproduce the input mean
before using them. Missing statistics must never change the measured observable.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class CorrelatorStatistics:
    sums: Tensor  # [time origin, lag], already component-contracted
    counts: Tensor  # same shape; missing source/sink pairs have count zero

    def mean(self) -> Tensor:
        return self.sums.sum(0) / self.counts.sum(0).clamp_min(1)


def attach_statistics(correlator: Tensor, sums: Tensor, counts: Tensor) -> Tensor:
    """Attach origin contributions without changing any correlator values."""
    correlator.correlator_statistics = CorrelatorStatistics(sums, counts)
    return correlator


def record_lag(sums: Tensor, counts: Tensor, lag: int, products: Tensor, valid: Tensor):
    """Accumulate products over particles/components, retaining time origins."""
    n = products.shape[0]
    sums[:n, lag] = torch.where(valid, products, 0).reshape(n, -1).sum(-1)
    counts[:n, lag] = valid.reshape(n, -1).sum(-1)


def series_statistics(
    series: Tensor, max_lag: int, connected: bool = True
) -> CorrelatorStatistics:
    """Dot-product correlation contributions for [T] or [T, components]."""
    work = series.detach().to(torch.float64).reshape(series.shape[0], -1)
    if connected:
        work = work - work.mean(0, keepdim=True)
    T = work.shape[0]
    sums = work.new_zeros(T, max_lag + 1)
    counts = torch.zeros_like(sums)
    for lag in range(min(max_lag + 1, T)):
        sums[: T - lag, lag] = (work[: T - lag] * work[lag:]).sum(-1)
        counts[: T - lag, lag] = 1
    return CorrelatorStatistics(sums, counts)


def batched_series_statistics(
    series: Tensor, max_lag: int, connected: bool = True
) -> list[CorrelatorStatistics]:
    """Origin statistics for every series of a batch ``[B, T]`` or ``[B, T, C]``.

    Equivalent to calling :func:`series_statistics` on each row, but the lag
    loop runs once for the whole batch.
    """
    work = series.detach().to(torch.float64)
    if work.ndim == 2:
        work = work.unsqueeze(-1)
    if work.ndim != 3:
        msg = f"series must have shape [B, T] or [B, T, C], got {tuple(series.shape)}"
        raise ValueError(msg)
    B, T, _ = work.shape
    if connected:
        work = work - work.mean(1, keepdim=True)
    sums = work.new_zeros(B, T, max_lag + 1)
    counts = torch.zeros_like(sums)
    for lag in range(min(max_lag + 1, T)):
        sums[:, : T - lag, lag] = (work[:, : T - lag] * work[:, lag:]).sum(-1)
        counts[:, : T - lag, lag] = 1
    return [CorrelatorStatistics(sums[b], counts[b]) for b in range(B)]


def correlators_with_statistics(
    series: Tensor,
    max_lag: int,
    connected: bool = True,
    dtype: torch.dtype = torch.float32,
) -> list[Tensor]:
    """Time correlators of a series batch with their origin statistics attached.

    The central value is the mean of the origin contributions, so it agrees
    with the FFT estimator (normalised by ``T - lag``, zero beyond ``T - 1``)
    up to rounding, while the statistics are exact by construction.
    """
    out = []
    for stats in batched_series_statistics(series, max_lag, connected):
        out.append(attach_statistics(stats.mean().to(dtype), stats.sums, stats.counts))
    return out


def stack_correlators(correlators: list[Tensor]) -> Tensor:
    """Stack scale correlators, retaining each scale's origin statistics."""
    result = torch.stack(correlators)
    result.correlator_statistics = [getattr(c, "correlator_statistics", None) for c in correlators]
    return result


def ensure_statistics(correlator: Tensor, series: Tensor) -> Tensor:
    """Recover series-based statistics only if they reproduce the measurement."""
    if getattr(correlator, "correlator_statistics", None) is not None:
        return correlator
    if series.ndim not in (1, 2) or not len(series):
        return correlator
    for connected in (True, False):
        stats = series_statistics(series, len(correlator) - 1, connected)
        if torch.allclose(stats.mean().to(correlator), correlator, rtol=2e-4, atol=1e-7):
            return attach_statistics(correlator, stats.sums, stats.counts)
    return correlator


def resample_statistics(
    stats: CorrelatorStatistics, method: str, block_size: int, n_bootstrap: int, seed: int
) -> np.ndarray:
    """Resample contiguous blocks of original-time lag contributions.

    Blocks contain equal numbers of origins (truncate only the incomplete
    last block for resampling). No artificial temporal adjacency is introduced.
    The central estimate continues to use all original observations.
    """
    sums = stats.sums.detach().double().cpu().numpy()
    counts = stats.counts.detach().double().cpu().numpy()
    T = len(sums)
    if T < 4:
        msg = "At least four time origins are required for covariance estimation"
        raise ValueError(msg)
    block_size = min(max(1, int(block_size)), max(1, T // 4))
    K = T // block_size
    block_sums = sums[: K * block_size].reshape(K, block_size, -1).sum(1)
    block_counts = counts[: K * block_size].reshape(K, block_size, -1).sum(1)
    if method in {"block_jackknife", "uncorrelated"}:
        numerator = block_sums.sum(0) - block_sums
        denominator = block_counts.sum(0) - block_counts
    elif method == "bootstrap":
        if n_bootstrap < 2:
            msg = "n_bootstrap must be at least two"
            raise ValueError(msg)
        idx = np.random.default_rng(seed).integers(K, size=(n_bootstrap, K))
        numerator = block_sums[idx].sum(1)
        denominator = block_counts[idx].sum(1)
    else:
        msg = f"Unknown covariance method: {method}"
        raise ValueError(msg)
    return np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )


def sample_covariance(samples: np.ndarray, method: str) -> np.ndarray:
    """Covariance of the estimator, not of the mean of its replicas."""
    centered = samples - samples.mean(0)
    n = len(samples)
    factor = 1 / (n - 1) if method == "bootstrap" else (n - 1) / n
    return factor * centered.T @ centered
