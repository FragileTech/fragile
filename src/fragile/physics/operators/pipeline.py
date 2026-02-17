"""Orchestrator: preparation -> operators -> correlators in one call.

Provides :func:`compute_strong_force_pipeline` which runs the full
companion-channel analysis pipeline:

1. Prepare channel data once from RunHistory.
2. Dispatch to each operator module for requested channels.
3. Collect operator series and compute batched correlators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from fragile.physics.fractal_gas.history import RunHistory

from .config import (
    BaryonOperatorConfig,
    ChannelConfigBase,
    CorrelatorConfig,
    ElectroweakOperatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    MultiscaleConfig,
    TensorOperatorConfig,
    VectorOperatorConfig,
)
from .correlators import compute_correlators_batched
from .preparation import prepare_channel_data, PreparedChannelData


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    """Output of :func:`compute_strong_force_pipeline`.

    Attributes:
        operators: Channel name -> operator time series tensor.
        correlators: Channel name -> correlator tensor ``[max_lag + 1]``.
        prepared_data: The shared :class:`PreparedChannelData` used by all
            operator modules.
    """

    operators: dict[str, Tensor] = field(default_factory=dict)
    correlators: dict[str, Tensor] = field(default_factory=dict)
    prepared_data: PreparedChannelData | None = None
    scales: Tensor | None = None  # [S] for reference


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Unified configuration for the full strong-force pipeline.

    Combines a base channel config, per-channel operator configs,
    a correlator config, and a list of requested channels.
    """

    base: ChannelConfigBase = field(default_factory=ChannelConfigBase)
    meson: MesonOperatorConfig = field(default_factory=MesonOperatorConfig)
    vector: VectorOperatorConfig = field(default_factory=VectorOperatorConfig)
    baryon: BaryonOperatorConfig = field(default_factory=BaryonOperatorConfig)
    glueball: GlueballOperatorConfig = field(default_factory=GlueballOperatorConfig)
    tensor: TensorOperatorConfig = field(default_factory=TensorOperatorConfig)
    electroweak: ElectroweakOperatorConfig = field(default_factory=ElectroweakOperatorConfig)
    correlator: CorrelatorConfig = field(default_factory=CorrelatorConfig)
    multiscale: MultiscaleConfig = field(default_factory=MultiscaleConfig)
    channels: list[str] | None = None  # None = all


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

# Channel sets that require specific PreparedChannelData flags
_NEEDS_POSITIONS = {"vector", "tensor"}
_NEEDS_SCORES = {"meson", "vector"}  # when score-directed modes are used
_NEEDS_MOMENTUM = {"glueball", "tensor"}
_NEEDS_FITNESS = {"electroweak"}
_NEEDS_ALIVE = {"electroweak"}
_NEEDS_VELOCITIES = {"electroweak"}
_NEEDS_POSITIONS_FULL = {"electroweak"}
_NEEDS_WILL_CLONE = {"electroweak"}


def compute_strong_force_pipeline(
    history: RunHistory,
    config: PipelineConfig | None = None,
) -> PipelineResult:
    """Run the full strong-force companion-channel pipeline.

    1. Calls :func:`prepare_channel_data` **once** from *history*.
    2. Dispatches to each requested operator module.
    3. Passes all operator series to :func:`compute_correlators_batched`.

    Args:
        history: Simulation run history.
        config: Pipeline configuration.  ``None`` uses all defaults.

    Returns:
        :class:`PipelineResult` with operator series, correlators, and the
        shared prepared data.
    """
    if config is None:
        config = PipelineConfig()

    requested = config.channels
    if requested is None:
        requested = ["meson", "vector", "baryon", "glueball", "tensor"]

    # Determine what optional data the preparation step must extract
    requested_set = set(requested)
    need_positions = bool(_NEEDS_POSITIONS & requested_set)
    need_scores = _any_score_mode(config, requested)
    need_momentum = bool(_NEEDS_MOMENTUM & requested_set) and _any_momentum_mode(
        config, requested
    )
    need_fitness = bool(_NEEDS_FITNESS & requested_set)
    need_alive = bool(_NEEDS_ALIVE & requested_set)
    need_velocities = bool(_NEEDS_VELOCITIES & requested_set)
    need_positions_full = bool(_NEEDS_POSITIONS_FULL & requested_set)
    need_will_clone = bool(_NEEDS_WILL_CLONE & requested_set)

    momentum_axis = 0
    if "glueball" in requested:
        momentum_axis = config.glueball.momentum_axis
    elif "tensor" in requested:
        momentum_axis = config.tensor.momentum_axis

    # Use the base config for preparation (shared physics params)
    data = prepare_channel_data(
        history,
        config.base,
        need_positions=need_positions,
        need_scores=need_scores,
        need_momentum_axis=need_momentum,
        momentum_axis=momentum_axis,
        need_fitness=need_fitness,
        need_alive=need_alive,
        need_velocities=need_velocities,
        need_positions_full=need_positions_full,
        need_will_clone=need_will_clone,
    )

    # Stage 1.5: Multiscale preparation
    ms = config.multiscale
    if ms.n_scales > 1 and data.frame_indices:
        from .geodesics import (
            compute_pairwise_distances,
            compute_smeared_kernels,
            gather_companion_distances,
            select_scales,
        )

        data.pairwise_distances = compute_pairwise_distances(
            history,
            data.frame_indices,
            method=ms.distance_method,
            edge_weight_mode=ms.edge_weight_mode,
            batch_size=ms.distance_batch_size,
        )

        if ms.scales is not None:
            data.scales = torch.as_tensor(
                ms.scales,
                dtype=torch.float32,
                device=data.device,
            )
        else:
            data.scales = select_scales(
                data.pairwise_distances,
                ms.n_scales,
                q_low=ms.scale_q_low,
                q_high=ms.scale_q_high,
                max_samples=ms.max_scale_samples,
                min_scale=ms.min_scale,
            )

        data.companion_d_ij, data.companion_d_ik, data.companion_d_jk = gather_companion_distances(
            data.pairwise_distances,
            data.companions_distance,
            data.companions_clone,
        )

        if ms.mode in {"kernel", "both"}:
            data.kernels = compute_smeared_kernels(
                data.pairwise_distances,
                data.scales,
                kernel_type=ms.kernel_type,
            )

    # Stage 2: Dispatch to operator modules
    all_operators: dict[str, Tensor] = {}

    if "meson" in requested:
        from .meson_operators import compute_meson_operators

        ops = compute_meson_operators(data, config.meson)
        all_operators.update(ops)

    if "vector" in requested:
        from .vector_operators import compute_vector_operators

        ops = compute_vector_operators(data, config.vector)
        all_operators.update(ops)

    if "baryon" in requested:
        from .baryon_operators import compute_baryon_operators

        ops = compute_baryon_operators(data, config.baryon)
        all_operators.update(ops)

    if "glueball" in requested:
        from .glueball_operators import compute_glueball_operators

        ops = compute_glueball_operators(data, config.glueball)
        all_operators.update(ops)

    if "tensor" in requested:
        from .tensor_operators import compute_tensor_operators

        ops = compute_tensor_operators(data, config.tensor)
        all_operators.update(ops)

    if "electroweak" in requested:
        from .electroweak_operators import compute_electroweak_operators

        ew_cfg = _resolve_electroweak_params(history, config.electroweak)
        ops = compute_electroweak_operators(data, ew_cfg)
        all_operators.update(ops)

    # Stage 3: Compute correlators
    effective_n_scales = ms.n_scales if data.scales is not None else 1
    correlators = compute_correlators_batched(
        all_operators,
        max_lag=config.correlator.max_lag,
        use_connected=config.correlator.use_connected,
        n_scales=effective_n_scales,
    )

    return PipelineResult(
        operators=all_operators,
        correlators=correlators,
        prepared_data=data,
        scales=data.scales,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _any_score_mode(config: PipelineConfig, requested: list[str]) -> bool:
    """Check whether any requested channel uses a score-directed mode."""
    if "meson" in requested:
        mode = str(config.meson.operator_mode).strip().lower()
        if mode in {"score_directed", "score_weighted"}:
            return True
    if "vector" in requested:
        mode = str(config.vector.operator_mode).strip().lower()
        if mode in {"score_directed", "score_gradient"}:
            return True
    if "baryon" in requested:
        mode = str(config.baryon.operator_mode).strip().lower()
        if mode in {"score_signed", "score_abs"}:
            return True
    return False


def _resolve_electroweak_params(
    history: RunHistory,
    cfg: ElectroweakOperatorConfig,
) -> ElectroweakOperatorConfig:
    """Auto-resolve epsilon_d / epsilon_clone from history.params when None."""
    from dataclasses import replace

    epsilon_d = cfg.epsilon_d
    epsilon_clone = cfg.epsilon_clone

    params = getattr(history, "params", None) or {}

    if epsilon_d is None:
        # Try history params, then fall back to std of fitness
        epsilon_d = params.get("epsilon_d")
        if epsilon_d is None and hasattr(history, "fitness") and history.fitness.numel() > 0:
            epsilon_d = float(history.fitness.std().clamp(min=1e-6).item())
        if epsilon_d is None:
            epsilon_d = 1.0

    if epsilon_clone is None:
        epsilon_clone = params.get("epsilon_clone")
        if epsilon_clone is None:
            epsilon_clone = 1e-8

    return replace(cfg, epsilon_d=float(epsilon_d), epsilon_clone=float(epsilon_clone))


def _any_momentum_mode(config: PipelineConfig, requested: list[str]) -> bool:
    """Check whether any requested channel uses momentum projection."""
    if "glueball" in requested and config.glueball.use_momentum_projection:
        return True
    if "tensor" in requested:
        # Tensor always uses momentum projection when positions_axis is present
        return True
    return False
