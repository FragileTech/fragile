"""QFT utility functions for the physics module.

Self-contained building blocks (FFT correlator, color states, gather helpers,
companion builders, PBC helpers, neighbor analysis, aggregation) ported from
fragile.fractalai.qft to eliminate cross-package dependencies.
"""

from fragile.physics.qft_utils.fft import _fft_correlator_batched
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0
from fragile.physics.qft_utils.helpers import (
    _apply_pbc_diff_torch,
    _slice_bounds,
    _safe_gather_2d,
    _safe_gather_3d,
    _resolve_3d_dims,
    _resolve_mc_time_index,
    _resolve_frame_indices,
)
from fragile.physics.qft_utils.companions import (
    PAIR_SELECTION_MODES,
    build_companion_triplets,
    build_companion_pair_indices,
)
from fragile.physics.qft_utils.neighbors import (
    compute_companion_batch,
    compute_recorded_neighbors_batch,
    compute_neighbors_auto,
    compute_neighbor_topology,
    compute_full_neighbor_matrix,
)
from fragile.physics.qft_utils.aggregation import (
    AggregatedTimeSeries,
    OperatorTimeSeries,
    aggregate_time_series,
    extract_precomputed_edge_weights,
    _compute_operator_weights,
)

__all__ = [
    "_fft_correlator_batched",
    "compute_color_states_batch",
    "estimate_ell0",
    "_apply_pbc_diff_torch",
    "_slice_bounds",
    "_safe_gather_2d",
    "_safe_gather_3d",
    "_resolve_3d_dims",
    "_resolve_mc_time_index",
    "_resolve_frame_indices",
    "PAIR_SELECTION_MODES",
    "build_companion_triplets",
    "build_companion_pair_indices",
    "compute_companion_batch",
    "compute_recorded_neighbors_batch",
    "compute_neighbors_auto",
    "compute_neighbor_topology",
    "compute_full_neighbor_matrix",
    "AggregatedTimeSeries",
    "OperatorTimeSeries",
    "aggregate_time_series",
    "extract_precomputed_edge_weights",
    "_compute_operator_weights",
]
