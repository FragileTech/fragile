"""Particle-level QFT observables from Fractal Gas run data."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def compute_companion_distance(
    positions: torch.Tensor,
    companions: torch.Tensor,
    pbc: bool,
    bounds: Any | None,
) -> torch.Tensor:
    diff = positions - positions[companions]
    if pbc and bounds is not None:
        high = bounds.high.to(positions)
        low = bounds.low.to(positions)
        span = high - low
        diff = diff - span * torch.round(diff / span)
    return torch.linalg.vector_norm(diff, dim=-1)


def compute_knn_indices(
    positions: torch.Tensor,
    alive: torch.Tensor,
    k: int,
    pbc: bool,
    bounds: Any | None,
    sample_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if k <= 0:
        raise ValueError("k must be positive.")

    if sample_indices is None:
        sample_indices = torch.where(alive)[0]

    if sample_indices.numel() == 0:
        return torch.empty((0, 0), device=positions.device, dtype=torch.long)

    pos_sample = positions[sample_indices]
    pos_all = positions

    diff = pos_sample.unsqueeze(1) - pos_all.unsqueeze(0)
    if pbc and bounds is not None:
        high = bounds.high.to(positions)
        low = bounds.low.to(positions)
        span = high - low
        diff = diff - span * torch.round(diff / span)

    dist_sq = (diff**2).sum(dim=-1)
    alive_mask = alive.unsqueeze(0).expand(dist_sq.shape[0], -1)
    dist_sq = dist_sq.masked_fill(~alive_mask, float("inf"))
    dist_sq[torch.arange(dist_sq.shape[0], device=positions.device), sample_indices] = float(
        "inf"
    )

    k_eff = min(k, positions.shape[0] - 1)
    if k_eff <= 0:
        return torch.empty((dist_sq.shape[0], 0), device=positions.device, dtype=torch.long)

    _, indices = torch.topk(dist_sq, k=k_eff, largest=False)
    return indices


def compute_color_state(
    force_viscous: torch.Tensor,
    velocities: torch.Tensor,
    h_eff: float,
    mass: float,
    ell0: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if h_eff <= 0:
        raise ValueError("h_eff must be positive.")
    if mass <= 0:
        raise ValueError("mass must be positive.")
    if ell0 <= 0:
        raise ValueError("ell0 must be positive.")
    if force_viscous.shape != velocities.shape:
        msg = "force_viscous and velocities must have the same shape."
        raise ValueError(msg)

    phase = (mass * velocities * ell0) / h_eff
    phase = phase.to(force_viscous.dtype)
    complex_phase = torch.polar(torch.ones_like(phase), phase)

    if force_viscous.dtype == torch.float64:
        complex_dtype = torch.complex128
    else:
        complex_dtype = torch.complex64

    tilde = force_viscous.to(complex_dtype) * complex_phase
    norm = torch.linalg.vector_norm(tilde, dim=-1, keepdim=True)
    norm = torch.clamp(norm, min=eps)
    color = tilde / norm
    valid = norm.squeeze(-1) > eps
    return color, valid


def compute_meson_operator_knn(
    color: torch.Tensor,
    sample_indices: torch.Tensor,
    neighbor_indices: torch.Tensor,
    alive: torch.Tensor,
    color_valid: torch.Tensor,
    reduce: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor]:
    if neighbor_indices.shape[1] < 1:
        raise ValueError("Need at least 1 neighbor for meson operator.")

    i = sample_indices
    neighbors = neighbor_indices

    color_i = color[i]
    color_j = color[neighbors]
    dots = (color_i.conj().unsqueeze(1) * color_j).sum(dim=-1)

    valid_i = alive[i] & color_valid[i]
    valid_j = alive[neighbors] & color_valid[neighbors]
    valid = valid_i.unsqueeze(1) & valid_j & (neighbors != i.unsqueeze(1))

    if reduce == "first":
        dots_first = dots[:, 0]
        valid_first = valid[:, 0]
        meson = torch.where(valid_first, dots_first, torch.zeros_like(dots_first))
        return meson, valid_first

    if reduce != "mean":
        raise ValueError("reduce must be 'mean' or 'first'")

    dots = torch.where(valid, dots, torch.zeros_like(dots))
    counts = valid.sum(dim=1)
    counts_clamped = torch.clamp(counts, min=1)
    meson = dots.sum(dim=1) / counts_clamped
    return meson, counts > 0


def compute_baryon_operator_knn(
    color: torch.Tensor,
    sample_indices: torch.Tensor,
    neighbor_indices: torch.Tensor,
    alive: torch.Tensor,
    color_valid: torch.Tensor,
    max_pairs: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if color.shape[-1] != 3:
        msg = "Baryon operator requires d=3 color vectors."
        raise ValueError(msg)
    if neighbor_indices.shape[1] < 2:
        raise ValueError("Need at least 2 neighbors for baryon operator.")

    i = sample_indices
    k_neighbors = neighbor_indices.shape[1]
    pair_idx = torch.combinations(
        torch.arange(k_neighbors, device=neighbor_indices.device), r=2
    )
    if max_pairs is not None and pair_idx.shape[0] > max_pairs:
        pair_idx = pair_idx[:max_pairs]

    j = neighbor_indices[:, pair_idx[:, 0]]
    k = neighbor_indices[:, pair_idx[:, 1]]

    valid = (
        alive[i].unsqueeze(1)
        & color_valid[i].unsqueeze(1)
        & alive[j]
        & color_valid[j]
        & alive[k]
        & color_valid[k]
        & (j != i.unsqueeze(1))
        & (k != i.unsqueeze(1))
        & (j != k)
    )

    color_i = color[i].unsqueeze(1).expand(-1, j.shape[1], -1)
    matrix = torch.stack([color_i, color[j], color[k]], dim=-1)
    det = torch.linalg.det(matrix)
    det = torch.where(valid, det, torch.zeros_like(det))
    counts = valid.sum(dim=1)
    counts_clamped = torch.clamp(counts, min=1)
    det_mean = det.sum(dim=1) / counts_clamped
    return det_mean, counts > 0


def compute_meson_operator(
    color: torch.Tensor,
    companions: torch.Tensor,
    alive: torch.Tensor,
    color_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    companion_color = color[companions]
    meson = (color.conj() * companion_color).sum(dim=-1)
    valid = alive & color_valid & alive[companions] & color_valid[companions]
    meson = torch.where(valid, meson, torch.zeros_like(meson))
    return meson, valid


def compute_baryon_operator(
    color: torch.Tensor,
    companions_distance: torch.Tensor,
    companions_clone: torch.Tensor,
    alive: torch.Tensor,
    color_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if color.shape[-1] != 3:
        msg = "Baryon operator requires d=3 color vectors."
        raise ValueError(msg)

    i = torch.arange(color.shape[0], device=color.device)
    j = companions_distance
    k = companions_clone

    valid = alive & color_valid & alive[j] & color_valid[j] & alive[k] & color_valid[k]
    valid = valid & (j != i) & (k != i) & (j != k)

    color_j = color[j]
    color_k = color[k]
    matrix = torch.stack([color, color_j, color_k], dim=-1)
    det = torch.linalg.det(matrix)
    det = torch.where(valid, det, torch.zeros_like(det))
    return det, valid


def compute_time_correlator(
    series: np.ndarray,
    max_lag: int | None = None,
    use_connected: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(series.shape[0])
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.complex128)

    if max_lag is None or max_lag >= n:
        max_lag = n - 1

    data = series
    if use_connected:
        data = data - data.mean()

    corr = np.zeros(max_lag + 1, dtype=np.complex128)
    for lag in range(max_lag + 1):
        corr[lag] = np.mean(data[: n - lag] * np.conjugate(data[lag:]))
    lags = np.arange(max_lag + 1, dtype=np.int64)
    return lags, corr


def fit_mass_exponential(
    lag_times: np.ndarray,
    corr: np.ndarray,
    fit_start: int = 1,
    fit_stop: int | None = None,
) -> dict[str, float]:
    if corr.size == 0:
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": 0.0,
        }

    if fit_stop is None:
        fit_stop = corr.size - 1

    fit_stop = min(fit_stop, corr.size - 1)
    indices = np.arange(corr.size)
    corr_real = np.real(corr)
    mask = (indices >= fit_start) & (indices <= fit_stop) & (corr_real > 0)

    if mask.sum() < 2:
        return {
            "mass": 0.0,
            "amplitude": 0.0,
            "r_squared": 0.0,
            "fit_points": float(mask.sum()),
        }

    x = lag_times[mask]
    y = np.log(corr_real[mask])

    slope, intercept = np.polyfit(x, y, 1)
    mass = float(-slope)
    amplitude = float(np.exp(intercept))

    y_pred = intercept + slope * x
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        "mass": mass,
        "amplitude": amplitude,
        "r_squared": r_squared,
        "fit_points": float(mask.sum()),
    }


def compute_effective_mass(corr: np.ndarray, delta_t: float) -> np.ndarray:
    if corr.size < 2 or delta_t <= 0:
        return np.array([], dtype=np.float64)

    corr_real = np.real(corr)
    eff = np.full(corr_real.size - 1, np.nan, dtype=np.float64)
    for idx in range(corr_real.size - 1):
        c0 = corr_real[idx]
        c1 = corr_real[idx + 1]
        if c0 > 0 and c1 > 0:
            eff[idx] = float(np.log(c0 / c1) / delta_t)
    return eff


def select_mass_plateau(
    lag_times: np.ndarray,
    corr: np.ndarray,
    fit_start: int = 1,
    fit_stop: int | None = None,
    min_points: int = 3,
    max_points: int | None = None,
    max_cv: float | None = 0.2,
) -> dict[str, float | int] | None:
    if corr.size < 3 or lag_times.size < 2:
        return None

    if min_points < 2:
        min_points = 2
    if max_points is not None and max_points < min_points:
        max_points = min_points

    diffs = np.diff(lag_times)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    delta_t = float(np.median(diffs))
    if delta_t <= 0 or not np.isfinite(delta_t):
        return None

    eff = compute_effective_mass(corr, delta_t)
    if eff.size == 0:
        return None

    fit_stop_idx = corr.size - 1 if fit_stop is None else min(fit_stop, corr.size - 1)
    fit_start_idx = max(0, fit_start)
    eff_start = fit_start_idx
    eff_stop = min(eff.size - 1, fit_stop_idx - 1)
    if eff_stop < eff_start:
        return None

    valid = np.isfinite(eff) & (eff > 0)
    valid[:eff_start] = False
    valid[eff_stop + 1 :] = False
    if not valid.any():
        return None

    def _better(candidate: dict[str, float], best: dict[str, float] | None) -> bool:
        if best is None:
            return True
        if candidate["cv"] < best["cv"]:
            return True
        if candidate["cv"] == best["cv"] and candidate["n"] > best["n"]:
            return True
        if (
            candidate["cv"] == best["cv"]
            and candidate["n"] == best["n"]
            and candidate["start"] < best["start"]
        ):
            return True
        return False

    best = None
    best_any = None
    idx = 0
    while idx < valid.size:
        if not valid[idx]:
            idx += 1
            continue
        seg_start = idx
        while idx < valid.size and valid[idx]:
            idx += 1
        seg_end = idx - 1
        seg_len = seg_end - seg_start + 1
        if seg_len < min_points:
            continue
        for start in range(seg_start, seg_end - min_points + 2):
            window_max = min(seg_end, start + (max_points - 1) if max_points else seg_end)
            for end in range(start + min_points - 1, window_max + 1):
                window = eff[start : end + 1]
                mean = float(np.mean(window))
                if mean <= 0:
                    continue
                std = float(np.std(window, ddof=1)) if window.size > 1 else 0.0
                cv = std / mean if mean > 0 else float("inf")
                candidate = {
                    "start": int(start),
                    "end": int(end),
                    "mean": mean,
                    "std": std,
                    "cv": cv,
                    "n": int(window.size),
                }
                if _better(candidate, best_any):
                    best_any = candidate
                if max_cv is None or cv <= max_cv:
                    if _better(candidate, best):
                        best = candidate

    selected = best or best_any
    if selected is None:
        return None

    fit_start_idx = int(selected["start"])
    fit_stop_idx = int(selected["end"]) + 1
    if fit_stop_idx <= fit_start_idx:
        return None

    lag_start = float(lag_times[fit_start_idx])
    lag_stop = float(lag_times[fit_stop_idx])

    return {
        "fit_start": fit_start_idx,
        "fit_stop": fit_stop_idx,
        "eff_start": int(selected["start"]),
        "eff_stop": int(selected["end"]),
        "eff_mean": float(selected["mean"]),
        "eff_std": float(selected["std"]),
        "eff_cv": float(selected["cv"]),
        "eff_n": int(selected["n"]),
        "delta_t": float(delta_t),
        "lag_start": lag_start,
        "lag_stop": lag_stop,
    }
