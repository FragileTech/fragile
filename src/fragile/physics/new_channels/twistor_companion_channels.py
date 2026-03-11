"""Companion-triplet twistor-inspired correlator channels.

These channels treat walker/companion geometry as an effective twistor operator
family. They follow the same source-frame triplet propagation rule used by the
companion glueball channel: the distance/clone companion assignments are fixed
at the source frame and the resulting triplet indices are evaluated at sink
times with those same indices.

The local operator family is obtained from two effective twistors built from
the source walker's distance and clone edges:

    tau_i(t) = <lambda_ij(t), lambda_ik(t)>
    V_i^a(t) = lambda_ij(t)^dagger sigma^a lambda_ik(t)

Channels:
- scalar: Re(tau_i)
- pseudoscalar: Im(tau_i)
- glueball-like scalar: |tau_i|^2
- vector: Re(V_i)
- axial-vector: Im(V_i)
- tensor: mean spin-2-like component built from V_i
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory
from fragile.physics.operators.twistor_operators import compute_twistor_triplet_fields
from fragile.physics.qft_utils import resolve_frame_indices


@dataclass
class TwistorCompanionCorrelatorConfig:
    """Configuration for companion-triplet twistor correlators."""

    warmup_fraction: float = 0.1
    end_fraction: float = 1.0
    max_lag: int = 80
    use_connected: bool = True
    delta_t: float | None = None
    spatial_dims: tuple[int, int, int] | None = None
    velocity_scale: float = 1.0
    eps: float = 1e-12


@dataclass
class TwistorCompanionCorrelatorOutput:
    """Companion-triplet twistor correlators and diagnostics."""

    scalar: Tensor
    scalar_raw: Tensor
    scalar_connected: Tensor
    pseudoscalar: Tensor
    pseudoscalar_raw: Tensor
    pseudoscalar_connected: Tensor
    glueball: Tensor
    glueball_raw: Tensor
    glueball_connected: Tensor
    vector: Tensor
    vector_raw: Tensor
    vector_connected: Tensor
    axial_vector: Tensor
    axial_vector_raw: Tensor
    axial_vector_connected: Tensor
    tensor: Tensor
    tensor_raw: Tensor
    tensor_connected: Tensor
    counts: Tensor
    frame_indices: list[int]
    triplet_counts_per_frame: Tensor
    mean_scalar: float
    mean_pseudoscalar: float
    mean_glueball: float
    mean_vector: Tensor
    mean_axial_vector: Tensor
    mean_tensor: float
    disconnected_scalar: float
    disconnected_pseudoscalar: float
    disconnected_glueball: float
    disconnected_vector: float
    disconnected_axial_vector: float
    disconnected_tensor: float
    n_valid_source_triplets: int
    operator_scalar_series: Tensor
    operator_pseudoscalar_series: Tensor
    operator_glueball_series: Tensor
    operator_vector_series: Tensor
    operator_axial_vector_series: Tensor
    operator_tensor_series: Tensor


def _per_frame_triplet_series(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Average ``[T, N]`` triplet observables per frame with masking."""
    weights = valid.to(values.dtype)
    counts = valid.sum(dim=1).to(torch.int64)
    sums = (values * weights).sum(dim=1)
    series = torch.zeros(values.shape[0], device=values.device, dtype=torch.float32)
    valid_t = counts > 0
    if torch.any(valid_t):
        series[valid_t] = (sums[valid_t] / counts[valid_t].to(values.dtype)).float()
    return series, counts


def _per_frame_vector_series(values: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Average ``[T, N, 3]`` vector observables per frame with masking."""
    weights = valid.to(values.dtype).unsqueeze(-1)
    counts = valid.sum(dim=1).to(torch.int64)
    sums = (values * weights).sum(dim=1)
    series = torch.zeros(values.shape[0], 3, device=values.device, dtype=torch.float32)
    valid_t = counts > 0
    if torch.any(valid_t):
        denom = counts[valid_t].to(values.dtype).unsqueeze(-1)
        series[valid_t] = (sums[valid_t] / denom).float()
    return series, counts


def _empty_output(
    *,
    n_lags: int,
    device: torch.device,
    frame_indices: list[int] | None = None,
) -> TwistorCompanionCorrelatorOutput:
    """Return an empty twistor output with consistent tensor shapes."""
    empty_f = torch.zeros(n_lags, dtype=torch.float32, device=device)
    empty_i = torch.zeros(n_lags, dtype=torch.int64, device=device)
    empty_t = torch.zeros(0, dtype=torch.int64, device=device)
    empty_vec = torch.zeros(3, dtype=torch.float32, device=device)
    empty_vec_series = torch.zeros(0, 3, dtype=torch.float32, device=device)

    return TwistorCompanionCorrelatorOutput(
        scalar=empty_f,
        scalar_raw=empty_f.clone(),
        scalar_connected=empty_f.clone(),
        pseudoscalar=empty_f.clone(),
        pseudoscalar_raw=empty_f.clone(),
        pseudoscalar_connected=empty_f.clone(),
        glueball=empty_f.clone(),
        glueball_raw=empty_f.clone(),
        glueball_connected=empty_f.clone(),
        vector=empty_f.clone(),
        vector_raw=empty_f.clone(),
        vector_connected=empty_f.clone(),
        axial_vector=empty_f.clone(),
        axial_vector_raw=empty_f.clone(),
        axial_vector_connected=empty_f.clone(),
        tensor=empty_f.clone(),
        tensor_raw=empty_f.clone(),
        tensor_connected=empty_f.clone(),
        counts=empty_i,
        frame_indices=[] if frame_indices is None else frame_indices,
        triplet_counts_per_frame=empty_t,
        mean_scalar=0.0,
        mean_pseudoscalar=0.0,
        mean_glueball=0.0,
        mean_vector=empty_vec.clone(),
        mean_axial_vector=empty_vec.clone(),
        mean_tensor=0.0,
        disconnected_scalar=0.0,
        disconnected_pseudoscalar=0.0,
        disconnected_glueball=0.0,
        disconnected_vector=0.0,
        disconnected_axial_vector=0.0,
        disconnected_tensor=0.0,
        n_valid_source_triplets=0,
        operator_scalar_series=empty_t.float(),
        operator_pseudoscalar_series=empty_t.float(),
        operator_glueball_series=empty_t.float(),
        operator_vector_series=empty_vec_series.clone(),
        operator_axial_vector_series=empty_vec_series.clone(),
        operator_tensor_series=empty_t.float(),
    )


def compute_twistor_companion_correlator_from_geometry(
    positions: Tensor,
    velocities: Tensor,
    alive_mask: Tensor,
    companions_distance: Tensor,
    companions_clone: Tensor,
    *,
    delta_t: float | Tensor,
    max_lag: int = 80,
    use_connected: bool = True,
    spatial_dims: tuple[int, int, int] | None = None,
    velocity_scale: float = 1.0,
    eps: float = 1e-12,
    frame_indices: list[int] | None = None,
) -> TwistorCompanionCorrelatorOutput:
    """Compute the full twistor companion correlator family."""
    if positions.ndim != 3 or velocities.ndim != 3:
        raise ValueError(
            f"positions and velocities must have shape [T, N, d], got "
            f"{tuple(positions.shape)} and {tuple(velocities.shape)}."
        )
    if positions.shape != velocities.shape:
        raise ValueError(
            f"positions and velocities must share the same shape, got "
            f"{tuple(positions.shape)} vs {tuple(velocities.shape)}."
        )
    if alive_mask.shape != positions.shape[:2]:
        raise ValueError(f"alive_mask must have shape [T, N], got {tuple(alive_mask.shape)}.")
    if companions_distance.shape != positions.shape[:2] or companions_clone.shape != positions.shape[:2]:
        raise ValueError(
            "companion arrays must have shape [T, N] aligned with positions, got "
            f"{tuple(companions_distance.shape)} and {tuple(companions_clone.shape)}."
        )

    t_total = int(positions.shape[0])
    max_lag = max(0, int(max_lag))
    effective_lag = min(max_lag, max(0, t_total - 1))
    n_lags = max_lag + 1
    device = positions.device

    if t_total == 0:
        return _empty_output(n_lags=n_lags, device=device, frame_indices=frame_indices)

    (
        source_scalar,
        source_pseudoscalar,
        source_glueball,
        source_vector,
        source_axial_vector,
        source_tensor,
        source_valid,
    ) = compute_twistor_triplet_fields(
        positions=positions,
        velocities=velocities,
        alive_mask=alive_mask,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        delta_t=delta_t,
        spatial_dims=spatial_dims,
        velocity_scale=velocity_scale,
        eps=eps,
    )

    operator_scalar_series, triplet_counts_per_frame = _per_frame_triplet_series(
        source_scalar,
        source_valid,
    )
    operator_pseudoscalar_series, _ = _per_frame_triplet_series(source_pseudoscalar, source_valid)
    operator_glueball_series, _ = _per_frame_triplet_series(source_glueball, source_valid)
    operator_vector_series, _ = _per_frame_vector_series(source_vector, source_valid)
    operator_axial_vector_series, _ = _per_frame_vector_series(source_axial_vector, source_valid)
    operator_tensor_series, _ = _per_frame_triplet_series(source_tensor, source_valid)

    n_valid_source_triplets = int(source_valid.sum().item())
    if n_valid_source_triplets > 0:
        mean_scalar_t = source_scalar[source_valid].mean()
        mean_pseudoscalar_t = source_pseudoscalar[source_valid].mean()
        mean_glueball_t = source_glueball[source_valid].mean()
        mean_vector_t = source_vector[source_valid].mean(dim=0)
        mean_axial_vector_t = source_axial_vector[source_valid].mean(dim=0)
        mean_tensor_t = source_tensor[source_valid].mean()
    else:
        mean_scalar_t = torch.zeros((), dtype=torch.float32, device=device)
        mean_pseudoscalar_t = torch.zeros((), dtype=torch.float32, device=device)
        mean_glueball_t = torch.zeros((), dtype=torch.float32, device=device)
        mean_vector_t = torch.zeros(3, dtype=torch.float32, device=device)
        mean_axial_vector_t = torch.zeros(3, dtype=torch.float32, device=device)
        mean_tensor_t = torch.zeros((), dtype=torch.float32, device=device)

    disconnected_scalar = float((mean_scalar_t * mean_scalar_t).item())
    disconnected_pseudoscalar = float((mean_pseudoscalar_t * mean_pseudoscalar_t).item())
    disconnected_glueball = float((mean_glueball_t * mean_glueball_t).item())
    disconnected_vector = float((mean_vector_t * mean_vector_t).sum().item())
    disconnected_axial_vector = float((mean_axial_vector_t * mean_axial_vector_t).sum().item())
    disconnected_tensor = float((mean_tensor_t * mean_tensor_t).item())

    scalar_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    scalar_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    pseudoscalar_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    pseudoscalar_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    glueball_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    glueball_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    vector_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    vector_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    axial_vector_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    axial_vector_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    tensor_raw = torch.zeros(n_lags, dtype=torch.float32, device=device)
    tensor_connected = torch.zeros(n_lags, dtype=torch.float32, device=device)
    counts = torch.zeros(n_lags, dtype=torch.int64, device=device)

    for lag in range(effective_lag + 1):
        source_len = t_total - lag
        (
            sink_scalar,
            sink_pseudoscalar,
            sink_glueball,
            sink_vector,
            sink_axial_vector,
            sink_tensor,
            sink_valid,
        ) = compute_twistor_triplet_fields(
            positions=positions[lag : lag + source_len],
            velocities=velocities[lag : lag + source_len],
            alive_mask=alive_mask[lag : lag + source_len],
            companions_distance=companions_distance[:source_len],
            companions_clone=companions_clone[:source_len],
            delta_t=delta_t,
            spatial_dims=spatial_dims,
            velocity_scale=velocity_scale,
            eps=eps,
        )

        valid_pair = source_valid[:source_len] & sink_valid
        count = int(valid_pair.sum().item())
        counts[lag] = count
        if count == 0:
            continue

        src_scalar_l = source_scalar[:source_len]
        src_pseudoscalar_l = source_pseudoscalar[:source_len]
        src_glueball_l = source_glueball[:source_len]
        src_vector_l = source_vector[:source_len]
        src_axial_vector_l = source_axial_vector[:source_len]
        src_tensor_l = source_tensor[:source_len]

        scalar_raw[lag] = (src_scalar_l * sink_scalar)[valid_pair].mean().float()
        pseudoscalar_raw[lag] = (src_pseudoscalar_l * sink_pseudoscalar)[valid_pair].mean().float()
        glueball_raw[lag] = (src_glueball_l * sink_glueball)[valid_pair].mean().float()
        vector_raw[lag] = ((src_vector_l * sink_vector).sum(dim=-1))[valid_pair].mean().float()
        axial_vector_raw[lag] = (
            (src_axial_vector_l * sink_axial_vector).sum(dim=-1)
        )[valid_pair].mean().float()
        tensor_raw[lag] = (src_tensor_l * sink_tensor)[valid_pair].mean().float()

        scalar_connected[lag] = (
            (src_scalar_l - mean_scalar_t) * (sink_scalar - mean_scalar_t)
        )[valid_pair].mean().float()
        pseudoscalar_connected[lag] = (
            (src_pseudoscalar_l - mean_pseudoscalar_t)
            * (sink_pseudoscalar - mean_pseudoscalar_t)
        )[valid_pair].mean().float()
        glueball_connected[lag] = (
            (src_glueball_l - mean_glueball_t) * (sink_glueball - mean_glueball_t)
        )[valid_pair].mean().float()
        vector_connected[lag] = (
            ((src_vector_l - mean_vector_t) * (sink_vector - mean_vector_t)).sum(dim=-1)
        )[valid_pair].mean().float()
        axial_vector_connected[lag] = (
            (
                (src_axial_vector_l - mean_axial_vector_t)
                * (sink_axial_vector - mean_axial_vector_t)
            ).sum(dim=-1)
        )[valid_pair].mean().float()
        tensor_connected[lag] = (
            (src_tensor_l - mean_tensor_t) * (sink_tensor - mean_tensor_t)
        )[valid_pair].mean().float()

    return TwistorCompanionCorrelatorOutput(
        scalar=scalar_connected if use_connected else scalar_raw,
        scalar_raw=scalar_raw,
        scalar_connected=scalar_connected,
        pseudoscalar=pseudoscalar_connected if use_connected else pseudoscalar_raw,
        pseudoscalar_raw=pseudoscalar_raw,
        pseudoscalar_connected=pseudoscalar_connected,
        glueball=glueball_connected if use_connected else glueball_raw,
        glueball_raw=glueball_raw,
        glueball_connected=glueball_connected,
        vector=vector_connected if use_connected else vector_raw,
        vector_raw=vector_raw,
        vector_connected=vector_connected,
        axial_vector=axial_vector_connected if use_connected else axial_vector_raw,
        axial_vector_raw=axial_vector_raw,
        axial_vector_connected=axial_vector_connected,
        tensor=tensor_connected if use_connected else tensor_raw,
        tensor_raw=tensor_raw,
        tensor_connected=tensor_connected,
        counts=counts,
        frame_indices=list(range(t_total)) if frame_indices is None else frame_indices,
        triplet_counts_per_frame=triplet_counts_per_frame,
        mean_scalar=float(mean_scalar_t.item()),
        mean_pseudoscalar=float(mean_pseudoscalar_t.item()),
        mean_glueball=float(mean_glueball_t.item()),
        mean_vector=mean_vector_t.float(),
        mean_axial_vector=mean_axial_vector_t.float(),
        mean_tensor=float(mean_tensor_t.item()),
        disconnected_scalar=disconnected_scalar,
        disconnected_pseudoscalar=disconnected_pseudoscalar,
        disconnected_glueball=disconnected_glueball,
        disconnected_vector=disconnected_vector,
        disconnected_axial_vector=disconnected_axial_vector,
        disconnected_tensor=disconnected_tensor,
        n_valid_source_triplets=n_valid_source_triplets,
        operator_scalar_series=operator_scalar_series,
        operator_pseudoscalar_series=operator_pseudoscalar_series,
        operator_glueball_series=operator_glueball_series,
        operator_vector_series=operator_vector_series,
        operator_axial_vector_series=operator_axial_vector_series,
        operator_tensor_series=operator_tensor_series,
    )


def compute_companion_twistor_correlator(
    history: RunHistory,
    config: TwistorCompanionCorrelatorConfig | None = None,
) -> TwistorCompanionCorrelatorOutput:
    """Compute companion-triplet twistor correlators from a ``RunHistory``."""
    config = config or TwistorCompanionCorrelatorConfig()
    frame_indices = resolve_frame_indices(
        history=history,
        warmup_fraction=float(config.warmup_fraction),
        end_fraction=float(config.end_fraction),
    )
    n_lags = int(max(0, config.max_lag)) + 1
    if not frame_indices:
        return _empty_output(n_lags=n_lags, device=torch.device("cpu"), frame_indices=[])

    start_idx = frame_indices[0]
    end_idx = frame_indices[-1] + 1
    positions = torch.as_tensor(
        history.x_before_clone[start_idx:end_idx],
        dtype=torch.float32,
    )
    velocities = torch.as_tensor(
        history.v_before_clone[start_idx:end_idx],
        dtype=torch.float32,
    )
    companions_distance = torch.as_tensor(
        history.companions_distance[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=positions.device,
    )
    companions_clone = torch.as_tensor(
        history.companions_clone[start_idx - 1 : end_idx - 1],
        dtype=torch.long,
        device=positions.device,
    )
    alive_mask = torch.as_tensor(
        history.alive_mask[start_idx - 1 : end_idx - 1],
        dtype=torch.bool,
        device=positions.device,
    )

    delta_t = float(config.delta_t) if config.delta_t is not None else float(
        getattr(history, "delta_t", 1.0)
    )

    return compute_twistor_companion_correlator_from_geometry(
        positions=positions,
        velocities=velocities,
        alive_mask=alive_mask,
        companions_distance=companions_distance,
        companions_clone=companions_clone,
        delta_t=delta_t,
        max_lag=int(config.max_lag),
        use_connected=bool(config.use_connected),
        spatial_dims=config.spatial_dims,
        velocity_scale=float(config.velocity_scale),
        eps=float(max(config.eps, 0.0)),
        frame_indices=frame_indices,
    )
