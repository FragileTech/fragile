"""Unified batched FFT correlator computation for all operator channels.

Takes operator series from all channels, stacks them into batches, and
computes temporal correlators in a single FFT pass.  Vector and tensor
channels are contracted (dot-product) into scalar correlators.
"""

from __future__ import annotations

import torch
from torch import Tensor

from fragile.fractalai.qft.correlator_channels import _fft_correlator_batched


def compute_correlators_batched(
    operators: dict[str, Tensor],
    max_lag: int,
    use_connected: bool = True,
) -> dict[str, Tensor]:
    """Compute temporal correlators for all operator series in one FFT pass.

    Scalar series ``[T]`` are batched directly.  Vector series ``[T, 3]``
    produce a dot-product contracted correlator
    ``C(tau) = sum_mu <O_mu(t) O_mu(t+tau)>``.  Tensor series ``[T, 5]``
    are handled similarly with 5-component contraction.

    Args:
        operators: Channel name -> operator series tensor.
        max_lag: Maximum temporal lag.
        use_connected: Subtract mean before FFT (connected correlator).

    Returns:
        Channel name -> correlator tensor ``[max_lag + 1]``.
    """
    if not operators:
        return {}

    n_lags = max(0, int(max_lag)) + 1
    device = next(iter(operators.values())).device

    # Classify channels by shape
    scalar_names: list[str] = []
    scalar_series: list[Tensor] = []
    multi_names: list[str] = []
    multi_series: list[Tensor] = []  # [T, C] where C > 1

    for name, series in operators.items():
        if series.numel() == 0:
            continue
        if series.ndim == 1:
            scalar_names.append(name)
            scalar_series.append(series)
        elif series.ndim == 2:
            multi_names.append(name)
            multi_series.append(series)

    results: dict[str, Tensor] = {}

    # --- Scalar channels: single batched FFT call ---
    if scalar_series:
        batch = torch.stack(scalar_series, dim=0)  # [B, T]
        corr = _fft_correlator_batched(batch, max_lag=max_lag, use_connected=use_connected)
        for i, name in enumerate(scalar_names):
            results[name] = corr[i]

    # --- Multi-component channels: batch all components, then contract ---
    if multi_series:
        # Collect all component series across all multi-component channels
        component_list: list[Tensor] = []
        channel_component_counts: list[int] = []
        for series in multi_series:
            n_comp = series.shape[1]
            channel_component_counts.append(n_comp)
            for c in range(n_comp):
                component_list.append(series[:, c])

        if component_list:
            batch = torch.stack(component_list, dim=0)  # [B_total, T]
            corr = _fft_correlator_batched(batch, max_lag=max_lag, use_connected=use_connected)

            # Unpack and contract per channel
            offset = 0
            for idx, name in enumerate(multi_names):
                n_comp = channel_component_counts[idx]
                # Sum over components: C(tau) = sum_c C_c(tau)
                contracted = corr[offset : offset + n_comp].sum(dim=0)
                results[name] = contracted
                offset += n_comp

    # Fill in empty entries for channels that were skipped (0-length)
    for name in operators:
        if name not in results:
            results[name] = torch.zeros(n_lags, dtype=torch.float32, device=device)

    return results
