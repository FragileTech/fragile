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
    :class:`TensorOperatorConfig`, :class:`CorrelatorConfig`,
    :class:`MultiscaleConfig`

Geodesics:
    :func:`compute_pairwise_distances`, :func:`select_scales`,
    :func:`compute_smeared_kernels`, :func:`gather_companion_distances`

Multiscale:
    :func:`gate_pair_validity_by_scale`, :func:`gate_triplet_validity_by_scale`,
    :func:`per_frame_series_multiscale`, :func:`per_frame_vector_series_multiscale`
"""

from .baryon_operators import compute_baryon_operators
from .config import (
    BaryonOperatorConfig,
    ChannelConfigBase,
    CorrelatorConfig,
    GlueballOperatorConfig,
    MesonOperatorConfig,
    MultiscaleConfig,
    TensorOperatorConfig,
    VectorOperatorConfig,
)
from .correlators import compute_correlators_batched
from .geodesics import (
    compute_pairwise_distances,
    compute_smeared_kernels,
    gather_companion_distances,
    select_scales,
)
from .glueball_operators import compute_glueball_operators
from .meson_operators import compute_meson_operators
from .multiscale import (
    gate_pair_validity_by_scale,
    gate_triplet_validity_by_scale,
    per_frame_series_multiscale,
    per_frame_vector_series_multiscale,
)
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
    "MultiscaleConfig",
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
    # Geodesics
    "compute_pairwise_distances",
    "compute_smeared_kernels",
    # Pipeline
    "compute_strong_force_pipeline",
    "compute_tensor_operators",
    "compute_vector_operators",
    # Multiscale
    "gate_pair_validity_by_scale",
    "gate_triplet_validity_by_scale",
    "gather_companion_distances",
    "per_frame_series_multiscale",
    "per_frame_vector_series_multiscale",
    # Preparation
    "prepare_channel_data",
    "select_scales",
]
