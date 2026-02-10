"""Electroweak (U1/SU2) channel correlators for Fractal Gas runs.

This module computes electroweak operator series from simulation-recorded
neighbor/companion data and extracts correlator masses through the shared
correlator pipeline used by other QFT channel modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from fragile.fractalai.core.history import RunHistory
from fragile.fractalai.qft.aggregation import (
    bin_by_euclidean_time,
)
from fragile.fractalai.qft.correlator_channels import (
    _fft_correlator_batched,
    ChannelCorrelatorResult,
    compute_channel_correlator,
    compute_effective_mass_torch,
    CorrelatorConfig,
    extract_mass_aic,
    extract_mass_linear,
)
from fragile.fractalai.qft.electroweak_observables import (
    compute_weighted_electroweak_ops_vectorized,
    pack_neighbors_from_edges,
    PackedNeighbors,
)


ELECTROWEAK_CHANNELS = (
    "u1_phase",
    "u1_dressed",
    "u1_phase_q2",
    "u1_dressed_q2",
    "su2_phase",
    "su2_component",
    "su2_doublet",
    "su2_doublet_diff",
    "ew_mixed",
)


@dataclass
class ElectroweakChannelConfig:
    """Configuration for electroweak channel correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    h_eff: float = 1.0
    use_connected: bool = True
    neighbor_method: str = "auto"
    edge_weight_mode: str = "inverse_riemannian_distance"
    neighbor_weighting: str = "inv_geodesic_full"
    companion_topology: str = "distance"
    neighbor_k: int = 0
    kernel_length_scale: float = 1.0
    voronoi_pbc_mode: str = "mirror"
    voronoi_exclude_boundary: bool = True
    voronoi_boundary_tolerance: float = 1e-6
    use_time_sliced_tessellation: bool = True
    time_sliced_neighbor_mode: str = "spacelike"

    # Time axis selection (for 4D Euclidean time analysis)
    time_axis: str = "mc"  # "mc" or "euclidean"
    euclidean_time_dim: int = 3  # Which dimension is Euclidean time (0-indexed)
    euclidean_time_bins: int = 50  # Number of time bins for Euclidean analysis
    euclidean_time_range: tuple[float, float] | None = None  # (t_min, t_max) or None for auto
    mc_time_index: int | None = None  # Recorded index for Euclidean slice; None => last

    # Fit settings
    window_widths: list[int] | None = None
    min_mass: float = 0.0
    max_mass: float = float("inf")
    fit_mode: str = "aic"
    fit_start: int = 2
    fit_stop: int | None = None
    min_fit_points: int = 2

    # Electroweak parameters (None => infer from history.params)
    epsilon_d: float | None = None
    epsilon_c: float | None = None
    epsilon_clone: float | None = None
    lambda_alg: float | None = None

    # Bootstrap error estimation
    compute_bootstrap_errors: bool = False
    n_bootstrap: int = 100


@dataclass
class ElectroweakChannelOutput:
    """Computed electroweak correlators and diagnostics."""

    channel_results: dict[str, ChannelCorrelatorResult]
    frame_indices: list[int]
    n_valid_frames: int
    avg_alive_walkers: float
    avg_edges: float


@dataclass
class _ElectroweakSeriesBundle:
    """Internal carrier for per-channel series and frame diagnostics."""

    series_map: dict[str, Tensor]
    frame_indices: list[int]
    n_valid_frames: int
    avg_alive_walkers: float
    avg_edges: float


EDGE_WEIGHT_MODE_ALIASES: dict[str, tuple[str, ...]] = {
    "uniform": ("uniform",),
    "inverse_distance": ("inverse_riemannian_distance", "inv_geodesic_iso"),
    "inverse_volume": ("inverse_riemannian_volume",),
    "kernel": ("kernel", "riemannian_kernel", "riemannian_kernel_volume"),
}


def _resolve_neighbor_method_strict(method: str) -> str:
    method_norm = str(method).strip().lower()
    if method_norm == "uniform":
        method_norm = "companions"
    if method_norm == "voronoi":
        msg = (
            "neighbor_method='voronoi' is disabled for electroweak channels. "
            "Use 'recorded', 'companions', or 'auto' to reuse simulation-recorded data."
        )
        raise ValueError(msg)
    if method_norm not in {"auto", "recorded", "companions"}:
        msg = "neighbor_method must be 'auto', 'recorded', or 'companions'."
        raise ValueError(msg)
    return method_norm


def _resolve_companion_topology(mode: str) -> str:
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"distance", "clone", "both"}:
        msg = "companion_topology must be 'distance', 'clone', or 'both'."
        raise ValueError(msg)
    return mode_norm


def _resolve_edge_weight_mode(
    requested_mode: str,
    edge_dict: dict[str, Tensor],
) -> str | None:
    candidates = [requested_mode]
    candidates.extend(EDGE_WEIGHT_MODE_ALIASES.get(requested_mode, ()))
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate in edge_dict:
            return candidate
    return None


def _nested_param(params: dict[str, Any] | None, *keys: str, default: float | None) -> float | None:
    if params is None:
        return default
    current: Any = params
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    if current is None:
        return default
    try:
        return float(current)
    except (TypeError, ValueError):
        return default


def _resolve_electroweak_params(history: RunHistory, cfg: ElectroweakChannelConfig) -> dict[str, float]:
    params = history.params if isinstance(history.params, dict) else None
    epsilon_d = cfg.epsilon_d
    if epsilon_d is None:
        epsilon_d = _nested_param(params, "companion_selection", "epsilon", default=None)
    epsilon_c = cfg.epsilon_c
    if epsilon_c is None:
        epsilon_c = _nested_param(params, "companion_selection_clone", "epsilon", default=None)
    lambda_alg = cfg.lambda_alg
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "companion_selection", "lambda_alg", default=None)
    if lambda_alg is None:
        lambda_alg = _nested_param(params, "fitness", "lambda_alg", default=0.0)
    epsilon_clone = cfg.epsilon_clone
    if epsilon_clone is None:
        epsilon_clone = _nested_param(params, "cloning", "epsilon_clone", default=1e-8)

    if epsilon_d is None:
        epsilon_d = 1.0
    if epsilon_c is None:
        epsilon_c = float(epsilon_d)
    if lambda_alg is None:
        lambda_alg = 0.0
    if epsilon_clone is None:
        epsilon_clone = 1e-8

    epsilon_d = float(max(epsilon_d, 1e-8))
    epsilon_c = float(max(epsilon_c, 1e-8))
    epsilon_clone = float(max(epsilon_clone, 1e-8))
    lambda_alg = float(max(lambda_alg, 0.0))

    return {
        "epsilon_d": epsilon_d,
        "epsilon_c": epsilon_c,
        "epsilon_clone": epsilon_clone,
        "lambda_alg": lambda_alg,
    }


def _resolve_mc_time_index(history: RunHistory, mc_time_index: int | None) -> int:
    """Resolve a Monte Carlo slice index from either recorded index or step."""
    if history.n_recorded < 2:
        msg = "Need at least 2 recorded timesteps for Euclidean analysis."
        raise ValueError(msg)
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


def _build_neighbor_data_from_history(
    history: RunHistory,
    frame_idx: int,
    mode: str,
    alive: Tensor,
    max_neighbors: int = 0,
) -> PackedNeighbors | None:
    """Build per-walker neighbor indices and weights from pre-computed RunHistory data."""
    if (
        history.neighbor_edges is None
        or history.edge_weights is None
        or frame_idx >= len(history.neighbor_edges)
        or frame_idx >= len(history.edge_weights)
    ):
        return None

    edges = history.neighbor_edges[frame_idx]
    ew_dict = history.edge_weights[frame_idx]
    if not torch.is_tensor(edges) or edges.numel() == 0:
        return None
    if mode == "uniform":
        weights_flat = torch.ones(edges.shape[0], device=edges.device, dtype=torch.float32)
    else:
        if not isinstance(ew_dict, dict):
            return None
        resolved_mode = _resolve_edge_weight_mode(mode, ew_dict)
        if resolved_mode is None:
            return None
        weights_flat = ew_dict[resolved_mode]
    return pack_neighbors_from_edges(
        edges=edges,
        edge_weights=weights_flat,
        alive=alive,
        n_walkers=int(alive.shape[0]),
        max_neighbors=int(max_neighbors),
        device=alive.device,
        dtype=torch.float32,
    )


def _build_neighbor_data_from_companions(
    history: RunHistory,
    frame_idx: int,
    alive: Tensor,
    companion_topology: str = "distance",
    max_neighbors: int = 0,
) -> PackedNeighbors | None:
    if frame_idx < 1:
        return None
    mode = _resolve_companion_topology(companion_topology)
    companions_distance = getattr(history, "companions_distance", None)
    companions_clone = getattr(history, "companions_clone", None)
    if companions_distance is None:
        return None
    if mode in {"clone", "both"} and companions_clone is None:
        return None
    info_idx = frame_idx - 1
    if info_idx < 0 or info_idx >= companions_distance.shape[0]:
        return None
    if companions_clone is not None and info_idx >= companions_clone.shape[0]:
        return None

    comp_d = companions_distance[info_idx]
    if not torch.is_tensor(comp_d):
        comp_d = torch.as_tensor(comp_d)
    comp_d = comp_d.to(device=alive.device, dtype=torch.long)

    comp_c: Tensor | None = None
    if companions_clone is not None:
        comp_c = companions_clone[info_idx]
        if not torch.is_tensor(comp_c):
            comp_c = torch.as_tensor(comp_c)
        comp_c = comp_c.to(device=alive.device, dtype=torch.long)

    n_walkers = int(alive.shape[0])
    if comp_d.numel() != n_walkers:
        return None
    if mode in {"clone", "both"} and (comp_c is None or comp_c.numel() != n_walkers):
        return None
    comp_d = comp_d.clamp(min=0, max=max(n_walkers - 1, 0))
    if comp_c is not None:
        comp_c = comp_c.clamp(min=0, max=max(n_walkers - 1, 0))

    src = torch.arange(n_walkers, device=alive.device, dtype=torch.long)
    if mode == "distance":
        dst = comp_d
    elif mode == "clone":
        if comp_c is None:
            return None
        dst = comp_c
    else:
        if comp_c is None:
            return None
        src = torch.cat([src, src], dim=0)
        dst = torch.cat([comp_d, comp_c], dim=0)

    edges = torch.stack([src, dst], dim=1)
    weights = torch.ones(edges.shape[0], device=alive.device, dtype=torch.float32)
    return pack_neighbors_from_edges(
        edges=edges,
        edge_weights=weights,
        alive=alive,
        n_walkers=n_walkers,
        max_neighbors=int(max_neighbors),
        device=alive.device,
        dtype=torch.float32,
    )


def _require_recorded_or_companion_neighbors(
    history: RunHistory,
    frame_idx: int,
    alive: Tensor,
    edge_weight_mode: str,
    neighbor_method: str,
    companion_topology: str = "distance",
    max_neighbors: int = 0,
) -> PackedNeighbors:
    method = _resolve_neighbor_method_strict(neighbor_method)
    if method in {"auto", "recorded"}:
        packed = _build_neighbor_data_from_history(
            history=history,
            frame_idx=frame_idx,
            mode=edge_weight_mode,
            alive=alive,
            max_neighbors=max_neighbors,
        )
        if packed is not None:
            return packed
        if method == "recorded":
            edge_weights_history = getattr(history, "edge_weights", None)
            if edge_weights_history is None:
                msg = (
                    "neighbor_method='recorded' requires RunHistory.edge_weights "
                    "to be recorded during simulation."
                )
                raise ValueError(msg)
            if frame_idx >= len(edge_weights_history):
                msg = (
                    f"Recorded frame {frame_idx} missing in edge_weights "
                    f"(available 0..{len(edge_weights_history) - 1})."
                )
                raise ValueError(msg)
            edge_dict = edge_weights_history[frame_idx]
            available = (
                ", ".join(sorted(str(k) for k in edge_dict.keys()))
                if isinstance(edge_dict, dict)
                else ""
            )
            msg = (
                f"edge_weights[{frame_idx}] does not contain mode '{edge_weight_mode}'. "
                f"Available modes: [{available}]"
            )
            raise ValueError(msg)

    if method in {"auto", "companions"}:
        packed = _build_neighbor_data_from_companions(
            history=history,
            frame_idx=frame_idx,
            alive=alive,
            companion_topology=companion_topology,
            max_neighbors=max_neighbors,
        )
        if packed is not None:
            return packed

    msg = (
        "Electroweak channels require simulation-recorded neighbor or companion data. "
        "Enable neighbor graph recording (neighbor_edges + edge_weights) or companions in RunHistory."
    )
    raise ValueError(msg)


def _masked_mean(values: Tensor, alive: Tensor) -> Tensor:
    masked = torch.where(alive, values, torch.zeros_like(values))
    counts = alive.sum(dim=1).clamp(min=1)
    return masked.sum(dim=1) / counts


def _compute_electroweak_series(
    history: RunHistory,
    cfg: ElectroweakChannelConfig,
) -> _ElectroweakSeriesBundle:
    frame_indices: list[int] = []
    alive_counts: list[float] = []
    edge_counts: list[float] = []

    start_idx = max(1, int(history.n_recorded * cfg.warmup_fraction))
    end_fraction = getattr(cfg, "end_fraction", 1.0)
    end_idx = max(start_idx + 1, int(history.n_recorded * end_fraction))
    if cfg.time_axis == "euclidean":
        start_idx = _resolve_mc_time_index(history, cfg.mc_time_index)
    if start_idx >= end_idx:
        return _ElectroweakSeriesBundle(
            series_map={},
            frame_indices=[],
            n_valid_frames=0,
            avg_alive_walkers=0.0,
            avg_edges=0.0,
        )

    h_eff = float(max(cfg.h_eff, 1e-8))
    edge_weight_mode = getattr(cfg, "edge_weight_mode", "inverse_riemannian_distance")
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    if cfg.time_axis == "euclidean":
        frame_idx = start_idx
        positions = history.x_before_clone[frame_idx]
        velocities = history.v_before_clone[frame_idx]
        fitness = history.fitness[frame_idx - 1]
        alive = history.alive_mask[frame_idx - 1]
        packed_neighbors = _require_recorded_or_companion_neighbors(
            history=history,
            frame_idx=frame_idx,
            alive=alive,
            edge_weight_mode=edge_weight_mode,
            neighbor_method=cfg.neighbor_method,
            companion_topology=cfg.companion_topology,
            max_neighbors=int(cfg.neighbor_k),
        )
        operators = compute_weighted_electroweak_ops_vectorized(
            positions=positions,
            velocities=velocities,
            fitness=fitness,
            alive=alive,
            neighbors=packed_neighbors,
            h_eff=h_eff,
            epsilon_d=epsilon_d,
            epsilon_c=epsilon_c,
            epsilon_clone=epsilon_clone,
            lambda_alg=lambda_alg,
            bounds=history.bounds,
            pbc=bool(history.pbc),
        )
        frame_indices.append(int(frame_idx))
        alive_counts.append(float(alive.sum().item()))
        edge_counts.append(float(packed_neighbors.valid.sum().item()))
        alive = alive.unsqueeze(0)
    else:
        T = end_idx - start_idx
        if T <= 0:
            return _ElectroweakSeriesBundle(
                series_map={},
                frame_indices=[],
                n_valid_frames=0,
                avg_alive_walkers=0.0,
                avg_edges=0.0,
            )
        operators = {name: [] for name in ELECTROWEAK_CHANNELS}
        alive_series = []
        for frame_idx in range(start_idx, end_idx):
            positions = history.x_before_clone[frame_idx]
            velocities = history.v_before_clone[frame_idx]
            fitness = history.fitness[frame_idx - 1]
            alive = history.alive_mask[frame_idx - 1]
            packed_neighbors = _require_recorded_or_companion_neighbors(
                history=history,
                frame_idx=frame_idx,
                alive=alive,
                edge_weight_mode=edge_weight_mode,
                neighbor_method=cfg.neighbor_method,
                companion_topology=cfg.companion_topology,
                max_neighbors=int(cfg.neighbor_k),
            )
            frame_ops = compute_weighted_electroweak_ops_vectorized(
                positions=positions,
                velocities=velocities,
                fitness=fitness,
                alive=alive,
                neighbors=packed_neighbors,
                h_eff=h_eff,
                epsilon_d=epsilon_d,
                epsilon_c=epsilon_c,
                epsilon_clone=epsilon_clone,
                lambda_alg=lambda_alg,
                bounds=history.bounds,
                pbc=bool(history.pbc),
            )
            frame_indices.append(int(frame_idx))
            alive_counts.append(float(alive.sum().item()))
            edge_counts.append(float(packed_neighbors.valid.sum().item()))
            for name in operators:
                operators[name].append(frame_ops[name])
            alive_series.append(alive)
        operators = {name: torch.stack(values, dim=0) for name, values in operators.items()}
        alive = torch.stack(alive_series, dim=0)

    if cfg.time_axis == "euclidean":
        if history.d < cfg.euclidean_time_dim + 1:
            msg = (
                f"Cannot use dimension {cfg.euclidean_time_dim} as Euclidean time "
                f"(only {history.d} dimensions available)"
            )
            raise ValueError(msg)

        positions = history.x_before_clone[start_idx : start_idx + 1]
        alive_slice = alive[:1]

        series: dict[str, Tensor] = {}
        for name, op_values in operators.items():
            if op_values.dim() == 1:
                op_values = op_values.unsqueeze(0)
            if op_values.is_complex():
                _coords_r, series_real = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.real[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                _coords_i, series_imag = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values.imag[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
                series[name] = series_real + 1j * series_imag
            else:
                _coords, series[name] = bin_by_euclidean_time(
                    positions=positions,
                    operators=op_values[:1],
                    alive=alive_slice,
                    time_dim=cfg.euclidean_time_dim,
                    n_bins=cfg.euclidean_time_bins,
                    time_range=cfg.euclidean_time_range,
                )
    else:
        series = {
            name: _masked_mean(op_values, alive)
            for name, op_values in operators.items()
        }

    n_valid_frames = len(frame_indices)
    avg_alive_walkers = float(sum(alive_counts) / n_valid_frames) if n_valid_frames > 0 else 0.0
    avg_edges = float(sum(edge_counts) / n_valid_frames) if n_valid_frames > 0 else 0.0
    return _ElectroweakSeriesBundle(
        series_map=series,
        frame_indices=frame_indices,
        n_valid_frames=n_valid_frames,
        avg_alive_walkers=avg_alive_walkers,
        avg_edges=avg_edges,
    )


def _to_correlator_config(cfg: ElectroweakChannelConfig) -> CorrelatorConfig:
    return CorrelatorConfig(
        max_lag=int(cfg.max_lag),
        use_connected=bool(cfg.use_connected),
        window_widths=cfg.window_widths,
        min_mass=float(cfg.min_mass),
        max_mass=float(cfg.max_mass),
        fit_mode=str(cfg.fit_mode),
        fit_start=int(cfg.fit_start),
        fit_stop=cfg.fit_stop,
        min_fit_points=int(cfg.min_fit_points),
        compute_bootstrap_errors=bool(cfg.compute_bootstrap_errors),
        n_bootstrap=int(cfg.n_bootstrap),
    )


def _build_result_from_precomputed(
    channel_name: str,
    series: Tensor,
    correlator: Tensor,
    correlator_err: Tensor | None,
    dt: float,
    config: CorrelatorConfig,
) -> ChannelCorrelatorResult:
    effective_mass = compute_effective_mass_torch(correlator, dt)
    if config.fit_mode == "linear_abs":
        mass_fit = extract_mass_linear(correlator.abs(), dt, config)
        window_data = {}
    elif config.fit_mode == "linear":
        mass_fit = extract_mass_linear(correlator, dt, config)
        window_data = {}
    else:
        mass_fit = extract_mass_aic(correlator, dt, config)
        window_data = {
            "window_masses": mass_fit.pop("window_masses", None),
            "window_aic": mass_fit.pop("window_aic", None),
            "window_widths": mass_fit.pop("window_widths", None),
            "window_r2": mass_fit.pop("window_r2", None),
        }

    return ChannelCorrelatorResult(
        channel_name=channel_name,
        correlator=correlator,
        correlator_err=correlator_err,
        effective_mass=effective_mass,
        mass_fit=mass_fit,
        series=series,
        n_samples=int(series.numel()),
        dt=dt,
        **window_data,
    )


def _compute_channel_results_batched(
    series_map: dict[str, Tensor],
    dt: float,
    config: CorrelatorConfig,
) -> dict[str, ChannelCorrelatorResult]:
    """Compute correlators and masses using grouped batched FFT operations."""
    results: dict[str, ChannelCorrelatorResult] = {}
    if not series_map:
        return results

    series_buffers: dict[str, Tensor] = {}
    groups: dict[bytes, tuple[Tensor, list[str]]] = {}

    for name, series in series_map.items():
        real_series = series.real if series.is_complex() else series
        real_series = real_series.float()
        valid = torch.isfinite(real_series)
        series_buffers[name] = real_series

        key = valid.detach().cpu().numpy().tobytes()
        if key in groups:
            groups[key][1].append(name)
        else:
            groups[key] = (valid, [name])

    for valid_t, names in groups.values():
        if not bool(torch.any(valid_t)):
            for name in names:
                empty_series = torch.zeros(0, device=valid_t.device, dtype=torch.float32)
                results[name] = compute_channel_correlator(
                    series=empty_series,
                    dt=dt,
                    config=config,
                    channel_name=name,
                )
            continue

        series_stack = torch.stack([series_buffers[name][valid_t] for name in names], dim=0).float()
        correlators = _fft_correlator_batched(
            series_stack,
            max_lag=int(config.max_lag),
            use_connected=bool(config.use_connected),
        )

        correlator_errs: Tensor | None = None
        if config.compute_bootstrap_errors:
            n_bootstrap = int(max(1, config.n_bootstrap))
            t_len = int(series_stack.shape[1])
            idx = torch.randint(0, t_len, (n_bootstrap, t_len), device=series_stack.device)
            idx = idx.unsqueeze(1).expand(-1, series_stack.shape[0], -1)
            sampled = torch.gather(
                series_stack.unsqueeze(0).expand(n_bootstrap, -1, -1),
                dim=2,
                index=idx,
            )
            boot_corr = _fft_correlator_batched(
                sampled.reshape(-1, t_len),
                max_lag=int(config.max_lag),
                use_connected=bool(config.use_connected),
            )
            correlator_errs = boot_corr.reshape(n_bootstrap, series_stack.shape[0], -1).std(dim=0)

        for idx_name, name in enumerate(names):
            err = correlator_errs[idx_name] if correlator_errs is not None else None
            results[name] = _build_result_from_precomputed(
                channel_name=name,
                series=series_stack[idx_name],
                correlator=correlators[idx_name],
                correlator_err=err,
                dt=dt,
                config=config,
            )

    return results


def _resolve_requested_channels(channels: list[str] | None) -> list[str]:
    if channels is None:
        return list(ELECTROWEAK_CHANNELS)
    requested = [str(name) for name in channels]
    unknown = sorted(set(requested) - set(ELECTROWEAK_CHANNELS))
    if unknown:
        msg = f"Unsupported electroweak channels {unknown}; supported: {list(ELECTROWEAK_CHANNELS)}."
        raise ValueError(msg)
    return requested


def compute_electroweak_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ElectroweakChannelConfig | None = None,
) -> ElectroweakChannelOutput:
    """Compute electroweak correlators and return results plus diagnostics."""
    cfg = config or ElectroweakChannelConfig()
    requested_channels = _resolve_requested_channels(channels)
    series_bundle = _compute_electroweak_series(history, cfg)
    selected_series = {
        name: series_bundle.series_map[name]
        for name in requested_channels
        if name in series_bundle.series_map
    }
    correlator_cfg = _to_correlator_config(cfg)
    dt = float(history.delta_t * history.record_every)
    channel_results = _compute_channel_results_batched(
        series_map=selected_series,
        dt=dt,
        config=correlator_cfg,
    )
    return ElectroweakChannelOutput(
        channel_results=channel_results,
        frame_indices=series_bundle.frame_indices,
        n_valid_frames=series_bundle.n_valid_frames,
        avg_alive_walkers=series_bundle.avg_alive_walkers,
        avg_edges=series_bundle.avg_edges,
    )


def compute_all_electroweak_channels(
    history: RunHistory,
    channels: list[str] | None = None,
    config: ElectroweakChannelConfig | None = None,
) -> dict[str, ChannelCorrelatorResult]:
    """Compatibility wrapper returning only channel results."""
    output = compute_electroweak_channels(history, channels=channels, config=config)
    return output.channel_results


def compute_electroweak_snapshot_operators(
    history: RunHistory,
    config: ElectroweakChannelConfig | None = None,
    channels: list[str] | None = None,
    frame_idx: int | None = None,
) -> dict[str, Tensor]:
    """Compute per-walker electroweak operators at a single MC snapshot."""
    cfg = config or ElectroweakChannelConfig()
    requested_channels = _resolve_requested_channels(channels)

    if frame_idx is None:
        frame_idx = _resolve_mc_time_index(history, cfg.mc_time_index)
    if frame_idx < 1 or frame_idx >= history.n_recorded:
        msg = (
            f"frame_idx {frame_idx} out of bounds "
            f"(valid recorded index 1..{history.n_recorded - 1})."
        )
        raise ValueError(msg)

    h_eff = float(max(cfg.h_eff, 1e-8))
    params = _resolve_electroweak_params(history, cfg)
    epsilon_d = float(params["epsilon_d"])
    epsilon_c = float(params["epsilon_c"])
    epsilon_clone = float(params["epsilon_clone"])
    lambda_alg = float(params["lambda_alg"])

    positions = history.x_before_clone[frame_idx]
    velocities = history.v_before_clone[frame_idx]
    fitness = history.fitness[frame_idx - 1]
    alive = history.alive_mask[frame_idx - 1]

    edge_weight_mode = getattr(cfg, "edge_weight_mode", "inverse_riemannian_distance")
    packed_neighbors = _require_recorded_or_companion_neighbors(
        history=history,
        frame_idx=frame_idx,
        alive=alive,
        edge_weight_mode=edge_weight_mode,
        neighbor_method=cfg.neighbor_method,
        companion_topology=cfg.companion_topology,
        max_neighbors=int(cfg.neighbor_k),
    )
    operators = compute_weighted_electroweak_ops_vectorized(
        positions=positions,
        velocities=velocities,
        fitness=fitness,
        alive=alive,
        neighbors=packed_neighbors,
        h_eff=h_eff,
        epsilon_d=epsilon_d,
        epsilon_c=epsilon_c,
        epsilon_clone=epsilon_clone,
        lambda_alg=lambda_alg,
        bounds=history.bounds,
        pbc=bool(history.pbc),
    )

    return {name: operators[name] for name in requested_channels if name in operators}
