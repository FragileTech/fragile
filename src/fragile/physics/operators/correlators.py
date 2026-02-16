"""Unified batched FFT correlator computation for all operator channels.

Takes operator series from all channels, stacks them into batches, and
computes temporal correlators in a single FFT pass.  Vector and tensor
channels are contracted (dot-product) into scalar correlators.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.physics.qft_utils.fft import _fft_correlator_batched


def compute_correlators_batched(
    operators: dict[str, Tensor],
    max_lag: int,
    use_connected: bool = True,
    n_scales: int = 1,
) -> dict[str, Tensor]:
    """Compute temporal correlators for all operator series in one FFT pass.

    Scalar series ``[T]`` are batched directly.  Vector series ``[T, 3]``
    produce a dot-product contracted correlator
    ``C(tau) = sum_mu <O_mu(t) O_mu(t+tau)>``.  Tensor series ``[T, 5]``
    are handled similarly with 5-component contraction.

    When ``n_scales > 1``, multiscale operators have shapes ``[S, T]`` or
    ``[S, T, C]`` and correlators are computed per scale, producing
    ``[S, max_lag + 1]``.

    Args:
        operators: Channel name -> operator series tensor.
        max_lag: Maximum temporal lag.
        use_connected: Subtract mean before FFT (connected correlator).
        n_scales: Number of scales (1 = single-scale mode).

    Returns:
        Channel name -> correlator tensor ``[max_lag + 1]`` or
        ``[S, max_lag + 1]`` when multiscale.
    """
    if not operators:
        return {}

    n_lags = max(0, int(max_lag)) + 1
    device = next(iter(operators.values())).device
    results: dict[str, Tensor] = {}

    if n_scales > 1:
        # Multiscale mode: operators are [S, T] or [S, T, C]
        for name, series in operators.items():
            if series.numel() == 0:
                continue
            if series.ndim == 2:
                # Scalar [S, T]: treat S as batch dimension
                corr = _fft_correlator_batched(
                    series, max_lag=max_lag, use_connected=use_connected,
                )  # [S, max_lag+1]
                results[name] = corr
            elif series.ndim == 3:
                # Multi-component [S, T, C]: reshape to [S*C, T], FFT, contract
                S, T, C = series.shape
                flat = series.reshape(S * C, T)
                corr = _fft_correlator_batched(
                    flat, max_lag=max_lag, use_connected=use_connected,
                )  # [S*C, max_lag+1]
                corr = corr.reshape(S, C, -1).sum(dim=1)  # [S, max_lag+1]
                results[name] = corr
    else:
        # Single-scale mode (unchanged)
        scalar_names: list[str] = []
        scalar_series: list[Tensor] = []
        multi_names: list[str] = []
        multi_series: list[Tensor] = []

        for name, series in operators.items():
            if series.numel() == 0:
                continue
            if series.ndim == 1:
                scalar_names.append(name)
                scalar_series.append(series)
            elif series.ndim == 2:
                multi_names.append(name)
                multi_series.append(series)

        if scalar_series:
            batch = torch.stack(scalar_series, dim=0)
            corr = _fft_correlator_batched(
                batch, max_lag=max_lag, use_connected=use_connected,
            )
            for i, name in enumerate(scalar_names):
                results[name] = corr[i]

        if multi_series:
            component_list: list[Tensor] = []
            channel_component_counts: list[int] = []
            for series in multi_series:
                n_comp = series.shape[1]
                channel_component_counts.append(n_comp)
                for c in range(n_comp):
                    component_list.append(series[:, c])

            if component_list:
                batch = torch.stack(component_list, dim=0)
                corr = _fft_correlator_batched(
                    batch, max_lag=max_lag, use_connected=use_connected,
                )
                offset = 0
                for idx, name in enumerate(multi_names):
                    n_comp = channel_component_counts[idx]
                    contracted = corr[offset : offset + n_comp].sum(dim=0)
                    results[name] = contracted
                    offset += n_comp

    # Fill in empty entries for channels that were skipped (0-length)
    for name in operators:
        if name not in results:
            if n_scales > 1:
                results[name] = torch.zeros(
                    n_scales, n_lags, dtype=torch.float32, device=device,
                )
            else:
                results[name] = torch.zeros(
                    n_lags, dtype=torch.float32, device=device,
                )

    return results
