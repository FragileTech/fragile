"""Correlator conversion with measured, jointly resampled covariance."""

from __future__ import annotations

import warnings

import gvar
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

from fragile.physics.mass_extraction.config import MassExtractionConfig
from fragile.physics.qft_utils.statistics import (
    CorrelatorStatistics,
    resample_statistics,
    sample_covariance,
    series_statistics,
)


def correlator_tensor_to_numpy(corr: Tensor) -> NDArray:
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
    """Resample blocks of lag products at their original temporal separations.

    Vector components are contracted after multiplication. Replica means never
    replace the original measured correlator. Connected products use the full
    sample mean; this is an origin-block estimate of covariance of that statistic.
    """
    if series.ndim not in (1, 2):
        msg = "Expected [T] or [T, components] operator series"
        raise ValueError(msg)
    stats = series_statistics(series, max_lag, use_connected)
    return resample_statistics(stats, method, block_size, n_bootstrap, seed)


def correlators_to_gvar(
    correlators: dict[str, Tensor],
    operators: dict[str, Tensor] | None = None,
    config: MassExtractionConfig | None = None,
    *,
    selected_keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Preserve measured correlators and estimate their joint covariance.

    Uncorrelated means the diagonal of measured jackknife covariance. Assumed
    relative errors require the explicit method 'assumed_relative'. Pair-based
    estimates require their original lag contributions, not averaged operators.
    selected_keys, when supplied, restricts conversion to expanded fit keys.
    """
    config = config or MassExtractionConfig()
    cfg = config.covariance
    entries = {}
    for key, corr in correlators.items():
        metadata = getattr(corr, "correlator_statistics", None)
        if corr.ndim == 2:
            for s in range(corr.shape[0]):
                op = None if operators is None else operators.get(key)
                entries[f"{key}_scale_{s}"] = (
                    corr[s],
                    metadata[s] if isinstance(metadata, list) else None,
                    op[s] if op is not None else None,
                )
        else:
            entries[key] = (corr, metadata, None if operators is None else operators.get(key))
    if selected_keys is not None:
        entries = {key: value for key, value in entries.items() if key in selected_keys}
    if not entries:
        return {}
    if cfg.method == "assumed_relative":
        warnings.warn(
            "Fit errors are assumed relative errors, not measured uncertainties.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            k: gvar.gvar(
                correlator_tensor_to_numpy(c),
                np.abs(correlator_tensor_to_numpy(c)) * cfg.relative_error + 1e-15,
            )
            for k, (c, _, _) in entries.items()
        }

    means, replicas, sizes, keys = [], [], [], []
    origin_lengths = set()
    for key, (corr, stats, series) in entries.items():
        if stats is None and series is not None:
            for connected in (True, False):
                candidate = series_statistics(series, len(corr) - 1, connected)
                if torch.allclose(candidate.mean().to(corr), corr, rtol=2e-4, atol=1e-7):
                    stats = candidate
                    break
        if not isinstance(stats, CorrelatorStatistics):
            msg = (
                f"{key}: original lag statistics or a matching operator series "
                "are required for measured covariance"
            )
            raise ValueError(msg)
        support = stats.counts.sum(0) > 0
        n = int(support.sum().item())
        if n == 0 or not bool(support[:n].all()):
            msg = f"{key}: missing lag support"
            raise ValueError(msg)
        stats = CorrelatorStatistics(stats.sums[:, :n], stats.counts[:, :n])
        corr = corr[:n]
        if not torch.allclose(stats.mean().to(corr), corr, rtol=2e-4, atol=1e-7):
            msg = f"{key}: lag statistics do not reproduce the supplied correlator"
            raise ValueError(msg)
        origin_lengths.add(stats.sums.shape[0])
        sample = resample_statistics(stats, cfg.method, cfg.block_size, cfg.n_bootstrap, cfg.seed)
        supported = np.isfinite(sample).all(0)
        n = int(np.flatnonzero(~supported)[0]) if not supported.all() else n
        if n < 3:
            msg = f"{key}: insufficient supported lags for covariance"
            raise ValueError(msg)
        replicas.append(sample[:, :n])
        corr = corr[:n]
        means.append(correlator_tensor_to_numpy(corr))
        keys.append(key)
        sizes.append(n)
    if len(origin_lengths) != 1:
        msg = "Joint covariance requires aligned channels with identical time origins"
        raise ValueError(msg)
    # Channels whose replicas do not vary at some lag (a constant series, or
    # too few independent origins) have a singular covariance and would make
    # the fit residuals non-finite. Exclude them with a warning so the other
    # channels can still be fitted; fail only if nothing usable remains.
    degenerate = [
        key
        for key, sample in zip(keys, replicas)
        if not np.all(np.isfinite(sample.var(axis=0)) & (sample.var(axis=0) > 0))
    ]
    if degenerate:
        n_origins = next(iter(origin_lengths))
        warnings.warn(
            f"Excluded {degenerate} from the fit: {n_origins} time origins give zero "
            "variance at some lags (constant series or too few independent frames; with "
            "cloning_frames_only the series has one frame per cloning step).",
            RuntimeWarning,
            stacklevel=2,
        )
        kept = [i for i, key in enumerate(keys) if key not in degenerate]
        if not kept:
            msg = f"Covariance is singular for every selected channel: {degenerate}"
            raise ValueError(msg)
        keys = [keys[i] for i in kept]
        sizes = [sizes[i] for i in kept]
        means = [means[i] for i in kept]
        replicas = [replicas[i] for i in kept]
    samples = np.concatenate(replicas, axis=1)
    cov = sample_covariance(samples, cfg.method)
    if cfg.method == "uncorrelated":
        cov = np.diag(np.diag(cov))
    joint = gvar.gvar(np.concatenate(means), cov)
    output = {}
    offset = 0
    for key, size in zip(keys, sizes):
        output[key] = joint[offset : offset + size]
        offset += size
    return output


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

    if not np_dataset or any(len(v) != len(run_correlators) for v in dataset.values()):
        msg = "Inter-run covariance requires at least two aligned runs for every channel"
        raise ValueError(msg)

    return gvar.dataset.avg_data(np_dataset)
