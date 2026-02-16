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
from .pipeline import PipelineConfig, PipelineResult, compute_strong_force_pipeline
from .preparation import PreparedChannelData, prepare_channel_data
from .tensor_operators import compute_tensor_operators
from .vector_operators import compute_vector_operators

__all__ = [
    # Preparation
    "prepare_channel_data",
    "PreparedChannelData",
    # Operators
    "compute_meson_operators",
    "compute_vector_operators",
    "compute_baryon_operators",
    "compute_glueball_operators",
    "compute_tensor_operators",
    # Correlators
    "compute_correlators_batched",
    # Pipeline
    "compute_strong_force_pipeline",
    "PipelineResult",
    "PipelineConfig",
    # Config
    "ChannelConfigBase",
    "MesonOperatorConfig",
    "VectorOperatorConfig",
    "BaryonOperatorConfig",
    "GlueballOperatorConfig",
    "TensorOperatorConfig",
    "CorrelatorConfig",
]
