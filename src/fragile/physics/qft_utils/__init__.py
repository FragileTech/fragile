"""QFT utility functions for the physics module.

Self-contained building blocks (FFT correlator, color states, gather helpers,
companion builders) ported from fragile.fractalai.qft to eliminate
cross-package dependencies.
"""

from fragile.physics.qft_utils.aggregation import bin_by_euclidean_time
from fragile.physics.qft_utils.color_states import compute_color_states_batch, estimate_ell0
from fragile.physics.qft_utils.companions import (
    build_companion_pair_indices,
    build_companion_triplets,
    PAIR_SELECTION_MODES,
)
from fragile.physics.qft_utils.fft import _fft_correlator_batched
from fragile.physics.qft_utils.helpers import (
    resolve_3d_dims,
    resolve_frame_indices,
    safe_gather_2d,
    safe_gather_3d,
)


__all__ = [
    "PAIR_SELECTION_MODES",
    "_fft_correlator_batched",
    "bin_by_euclidean_time",
    "build_companion_pair_indices",
    "build_companion_triplets",
    "compute_color_states_batch",
    "estimate_ell0",
    "resolve_3d_dims",
    "resolve_frame_indices",
    "safe_gather_2d",
    "safe_gather_3d",
]
