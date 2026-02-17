"""Convert torch correlator tensors to gvar arrays with covariance estimation.

Provides utilities for converting the torch-based ``PipelineResult`` output
into gvar data suitable for corrfitter, including optional resampling-based
covariance estimation.
"""

from __future__ import annotations

import gvar
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .config import CovarianceConfig, MassExtractionConfig


def correlator_tensor_to_numpy(corr: Tensor) -> NDArray:
    """Detach a torch correlator tensor and convert to float64 numpy array.

    Args:
        corr: Correlator tensor of any shape.

    Returns:
        NumPy float64 array with same shape.
    """
    return corr.detach().cpu().to(torch.float64).numpy()


def operator_series_to_correlator_samples(
    series: Tensor,
    max_lag: int,
    method: str = "block_jackknife",
    block_size: int = 10,
    n_bootstrap: int = 200,
    seed: int = 42,
    use_connected: bool = True,
) -> NDArray:
    """Resample an operator time series to produce correlator samples.

    Each resample computes a correlator via FFT, giving ``N_samples``
    independent correlator estimates for covariance estimation.

    Args:
        series: Operator time series ``[T]`` (1D).
        max_lag: Maximum lag.
        method: ``"block_jackknife"`` or ``"bootstrap"``.
        block_size: Block size for jackknife/bootstrap.
        n_bootstrap: Number of bootstrap samples.
        seed: Random seed for bootstrap.
        use_connected: Subtract mean (connected correlator).

    Returns:
        Array of shape ``[N_samples, max_lag + 1]``.
    """
    from fragile.physics.qft_utils.fft import _fft_correlator_batched

    if series.ndim != 1:
        raise ValueError(f"Expected 1D series, got shape {tuple(series.shape)}")

    T = len(series)
    if T < 2 * block_size:
        block_size = max(1, T // 4)

    n_blocks = T // block_size

    if method == "block_jackknife":
        samples = []
        for i in range(n_blocks):
            # Leave out block i
            mask = torch.ones(T, dtype=torch.bool)
            start = i * block_size
            end = min(start + block_size, T)
            mask[start:end] = False
            sub = series[mask].unsqueeze(0)  # [1, T-block]
            corr = _fft_correlator_batched(sub, max_lag=max_lag, use_connected=use_connected)
            samples.append(corr.squeeze(0).cpu().to(torch.float64).numpy())
        return np.array(samples)

    elif method == "bootstrap":
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(n_bootstrap):
            # Draw n_blocks block indices with replacement
            block_idx = rng.integers(0, n_blocks, size=n_blocks)
            parts = []
            for bi in block_idx:
                start = bi * block_size
                end = min(start + block_size, T)
                parts.append(series[start:end])
            resampled = torch.cat(parts).unsqueeze(0)  # [1, ~T]
            corr = _fft_correlator_batched(
                resampled, max_lag=max_lag, use_connected=use_connected
            )
            samples.append(corr.squeeze(0).cpu().to(torch.float64).numpy())
        return np.array(samples)

    else:
        raise ValueError(f"Unknown resampling method: {method!r}")


def correlators_to_gvar(
    correlators: dict[str, Tensor],
    operators: dict[str, Tensor] | None = None,
    config: MassExtractionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Convert torch correlator tensors to gvar arrays.

    For the ``"uncorrelated"`` method, diagonal errors are estimated from the
    correlator values directly. For ``"block_jackknife"`` and ``"bootstrap"``,
    the operator time series are resampled to estimate the full covariance
    matrix via ``gvar.dataset.avg_data``.

    Args:
        correlators: Channel name -> correlator tensor ``[L]`` or ``[S, L]``.
        operators: Channel name -> operator time series (needed for resampling).
        config: Pipeline config. ``None`` uses defaults.

    Returns:
        Dict mapping channel name to 1D gvar array.
    """
    if config is None:
        from .config import MassExtractionConfig

        config = MassExtractionConfig()

    cov_config = config.covariance
    result: dict[str, np.ndarray] = {}

    for key, corr_t in correlators.items():
        corr_np = correlator_tensor_to_numpy(corr_t)

        if corr_np.ndim == 2:
            # Multiscale [S, L] -> expand to separate keys
            for s in range(corr_np.shape[0]):
                scale_key = f"{key}_scale_{s}"
                result[scale_key] = _single_correlator_to_gvar(
                    scale_key,
                    corr_np[s],
                    operators,
                    cov_config,
                )
        else:
            result[key] = _single_correlator_to_gvar(
                key,
                corr_np,
                operators,
                cov_config,
            )

    return result


def _single_correlator_to_gvar(
    key: str,
    corr_np: NDArray,
    operators: dict[str, Tensor] | None,
    cov_config: CovarianceConfig,
) -> np.ndarray:
    """Convert a single 1D correlator to a gvar array."""
    if cov_config.method == "uncorrelated" or operators is None:
        # Use diagonal errors: relative error ~ 10% as conservative default
        means = corr_np
        errors = np.abs(means) * 0.1 + 1e-15
        return gvar.gvar(means, errors)

    # Resampling-based covariance
    # Find matching operator (strip scale suffix if needed)
    base_key = key.rsplit("_scale_", 1)[0] if "_scale_" in key else key
    if base_key not in operators:
        means = corr_np
        errors = np.abs(means) * 0.1 + 1e-15
        return gvar.gvar(means, errors)

    series = operators[base_key]

    # Step 1: Handle multiscale — extract specific scale from first axis.
    # Operators may be [S, T] or [S, T, D]; slicing gives [T] or [T, D].
    if "_scale_" in key and series.ndim >= 2:
        scale_index = 0
        try:
            scale_index = int(key.rsplit("_scale_", 1)[1])
        except (ValueError, IndexError):
            pass
        if scale_index < series.shape[0]:
            series = series[scale_index]

    # Step 2: Handle multi-component operators — average over trailing
    # dimensions.  E.g. vector operators have shape [T, 3]; after scale
    # slicing a [S, T, D] tensor we may still have [T, D].
    while series.ndim > 1:
        series = series.mean(dim=-1)

    if series.ndim != 1 or len(series) < 2:
        means = corr_np
        errors = np.abs(means) * 0.1 + 1e-15
        return gvar.gvar(means, errors)

    max_lag = len(corr_np) - 1
    samples = operator_series_to_correlator_samples(
        series,
        max_lag=max_lag,
        method=cov_config.method,
        block_size=cov_config.block_size,
        n_bootstrap=cov_config.n_bootstrap,
        seed=cov_config.seed,
    )

    # Use gvar.dataset.avg_data for proper covariance
    dataset = {key: samples}
    avg = gvar.dataset.avg_data(dataset)
    return avg[key]


def multi_run_correlators_to_gvar(
    run_correlators: list[dict[str, Tensor]],
) -> dict[str, np.ndarray]:
    """Convert correlators from multiple runs to gvar arrays.

    Uses inter-run variation for covariance estimation (gold standard).

    Args:
        run_correlators: List of correlator dicts from independent runs.

    Returns:
        Dict mapping channel name to gvar array with inter-run covariance.
    """
    if not run_correlators:
        return {}

    # Collect all keys
    all_keys = set()
    for run_corr in run_correlators:
        all_keys.update(run_corr.keys())

    # Build dataset: key -> [N_runs, T] array
    dataset: dict[str, list[NDArray]] = {}
    for key in all_keys:
        samples = []
        for run_corr in run_correlators:
            if key in run_corr:
                corr_t = run_corr[key]
                corr_np = correlator_tensor_to_numpy(corr_t)
                if corr_np.ndim == 2:
                    # Multiscale: expand
                    for s in range(corr_np.shape[0]):
                        scale_key = f"{key}_scale_{s}"
                        if scale_key not in dataset:
                            dataset[scale_key] = []
                        dataset[scale_key].append(corr_np[s])
                else:
                    if key not in dataset:
                        dataset[key] = []
                    dataset[key].append(corr_np)

    # Convert to numpy arrays and use avg_data
    np_dataset = {k: np.array(v) for k, v in dataset.items() if len(v) > 1}

    if not np_dataset:
        # Fall back to uncorrelated if only one run
        result = {}
        for key in all_keys:
            corr_t = run_correlators[0][key]
            corr_np = correlator_tensor_to_numpy(corr_t)
            if corr_np.ndim == 2:
                for s in range(corr_np.shape[0]):
                    scale_key = f"{key}_scale_{s}"
                    means = corr_np[s]
                    result[scale_key] = gvar.gvar(means, np.abs(means) * 0.1 + 1e-15)
            else:
                result[key] = gvar.gvar(corr_np, np.abs(corr_np) * 0.1 + 1e-15)
        return result

    return gvar.dataset.avg_data(np_dataset)
