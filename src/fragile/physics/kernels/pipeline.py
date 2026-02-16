"""Orchestrator: preparation -> operators -> correlators in one call.

Provides :func:`compute_strong_force_pipeline` which runs the full
companion-channel analysis pipeline:

1. Prepare channel data once from RunHistory.
2. Dispatch to each operator module for requested channels.
3. Collect operator series and compute batched correlators.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from fragile.fractalai.core.history import RunHistory

from .config import (
    BaryonOperatorConfig,
    ChannelConfigBase,
    CorrelatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
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
    correlator: CorrelatorConfig = field(default_factory=CorrelatorConfig)
    channels: list[str] | None = None  # None = all


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

# Channel sets that require specific PreparedChannelData flags
_NEEDS_POSITIONS = {"vector", "tensor"}
_NEEDS_SCORES = {"meson", "vector"}  # when score-directed modes are used
_NEEDS_MOMENTUM = {"glueball", "tensor"}


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
    need_positions = bool(_NEEDS_POSITIONS & set(requested))
    need_scores = _any_score_mode(config, requested)
    need_momentum = bool(_NEEDS_MOMENTUM & set(requested)) and _any_momentum_mode(
        config, requested
    )

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
    )

    # Dispatch to operator modules
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

    # Compute correlators
    correlators = compute_correlators_batched(
        all_operators,
        max_lag=config.correlator.max_lag,
        use_connected=config.correlator.use_connected,
    )

    return PipelineResult(
        operators=all_operators,
        correlators=correlators,
        prepared_data=data,
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


def _any_momentum_mode(config: PipelineConfig, requested: list[str]) -> bool:
    """Check whether any requested channel uses momentum projection."""
    if "glueball" in requested and config.glueball.use_momentum_projection:
        return True
    if "tensor" in requested:
        # Tensor always uses momentum projection when positions_axis is present
        return True
    return False
