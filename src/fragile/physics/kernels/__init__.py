"""Strong-force companion-channel operators and correlators.

Public API
----------
Data preparation:
    :func:`prepare_channel_data`, :class:`PreparedChannelData`

Operator modules:
    :func:`compute_meson_operators`
    :func:`compute_vector_operators`
    :func:`compute_baryon_operators`
    :func:`compute_glueball_operators`
    :func:`compute_tensor_operators`

Correlators:
    :func:`compute_correlators_batched`

Pipeline:
    :func:`compute_strong_force_pipeline`, :class:`PipelineResult`, :class:`PipelineConfig`

Configuration:
    :class:`ChannelConfigBase`,
    :class:`MesonOperatorConfig`, :class:`VectorOperatorConfig`,
    :class:`BaryonOperatorConfig`, :class:`GlueballOperatorConfig`,
    :class:`TensorOperatorConfig`, :class:`CorrelatorConfig`
"""

from .baryon_operators import compute_baryon_operators
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
from .glueball_operators import compute_glueball_operators
from .meson_operators import compute_meson_operators
from .pipeline import compute_strong_force_pipeline, PipelineConfig, PipelineResult
from .preparation import prepare_channel_data, PreparedChannelData
from .tensor_operators import compute_tensor_operators
from .vector_operators import compute_vector_operators


__all__ = [
    "BaryonOperatorConfig",
    # Config
    "ChannelConfigBase",
    "CorrelatorConfig",
    "GlueballOperatorConfig",
    "MesonOperatorConfig",
    "PipelineConfig",
    "PipelineResult",
    "PreparedChannelData",
    "TensorOperatorConfig",
    "VectorOperatorConfig",
    "compute_baryon_operators",
    # Correlators
    "compute_correlators_batched",
    "compute_glueball_operators",
    # Operators
    "compute_meson_operators",
    # Pipeline
    "compute_strong_force_pipeline",
    "compute_tensor_operators",
    "compute_vector_operators",
    # Preparation
    "prepare_channel_data",
]
