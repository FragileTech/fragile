"""Tests for companion triplet and pair index builders."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from fragile.physics.qft_utils.companions import (
    build_companion_pair_indices,
    build_companion_triplets,
    PAIR_SELECTION_MODES,
)


# ===========================================================================
# TestPairSelectionModes
# ===========================================================================


class TestPairSelectionModes:
    """Test the PAIR_SELECTION_MODES constant."""

    def test_is_tuple_of_3_strings(self):
        """PAIR_SELECTION_MODES should be a tuple of 3 strings."""
        assert isinstance(PAIR_SELECTION_MODES, tuple)
        assert len(PAIR_SELECTION_MODES) == 3
        assert all(isinstance(mode, str) for mode in PAIR_SELECTION_MODES)

    def test_contains_expected_modes(self):
        """PAIR_SELECTION_MODES should contain exactly distance, clone, both."""
        assert set(PAIR_SELECTION_MODES) == {"distance", "clone", "both"}


# ===========================================================================
# TestBuildCompanionTriplets
# ===========================================================================


class TestBuildCompanionTriplets:
    """Test build_companion_triplets function."""

    def test_output_shapes(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """All outputs should have shape [T, N], structural_valid should be bool."""
        anchor_idx, companion_j, companion_k, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        assert anchor_idx.shape == (T, N)
        assert companion_j.shape == (T, N)
        assert companion_k.shape == (T, N)
        assert structural_valid.shape == (T, N)
        assert structural_valid.dtype == torch.bool

    def test_anchor_idx_values(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """anchor_idx[t, i] should equal i for all t, i."""
        anchor_idx, _, _, _ = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        for t in range(T):
            expected = torch.arange(N, device=anchor_idx.device)
            assert torch.equal(anchor_idx[t], expected)

    def test_companion_j_equals_distance(
        self, companions_distance: Tensor, companions_clone: Tensor
    ):
        """companion_j should equal companions_distance."""
        _, companion_j, _, _ = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        assert torch.equal(companion_j, companions_distance)

    def test_companion_k_equals_clone(self, companions_distance: Tensor, companions_clone: Tensor):
        """companion_k should equal companions_clone."""
        _, _, companion_k, _ = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        assert torch.equal(companion_k, companions_clone)

    def test_structural_validity_cyclic_companions(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """With cyclic companions (shift by 1 and 2), all should be valid when N >= 3."""
        # Fixtures use cyclic shift by 1 and 2
        # For N >= 3, all indices are distinct and in range
        if N >= 3:
            _, _, _, structural_valid = build_companion_triplets(
                companions_distance=companions_distance,
                companions_clone=companions_clone,
            )
            assert structural_valid.all()

    def test_out_of_range_negative_indices(self):
        """Out-of-range negative indices should make structural_valid False."""
        companions_distance = torch.tensor([[1, 2, -1, 0]])
        companions_clone = torch.tensor([[2, 3, 0, 1]])

        _, _, _, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        # Index 2 has companion_j = -1, which is out of range
        assert not structural_valid[0, 2]
        # Others should be valid (all distinct and in range)
        assert structural_valid[0, 0]
        assert structural_valid[0, 1]
        assert structural_valid[0, 3]

    def test_out_of_range_large_indices(self):
        """Indices >= N should make structural_valid False."""
        companions_distance = torch.tensor([[1, 2, 3, 5]])  # 5 >= N
        companions_clone = torch.tensor([[2, 3, 0, 1]])

        _, _, _, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        # Index 3 has companion_j = 5 >= N
        assert not structural_valid[0, 3]
        # Others should be valid
        assert structural_valid[0, 0]
        assert structural_valid[0, 1]
        assert structural_valid[0, 2]

    def test_self_referencing_anchor(self):
        """When companion equals anchor, structural_valid should be False."""
        # companion_j[0, 1] == 1 (self-reference)
        companions_distance = torch.tensor([[1, 1, 2, 3]])
        companions_clone = torch.tensor([[2, 3, 0, 1]])

        _, _, _, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        # Index 1 has companion_j == anchor == 1
        assert not structural_valid[0, 1]

    def test_j_equals_k(self):
        """When companion_j equals companion_k, structural_valid should be False."""
        # At index 1: companion_j = 2, companion_k = 2
        companions_distance = torch.tensor([[1, 2, 3, 0]])
        companions_clone = torch.tensor([[3, 2, 1, 0]])

        _, _, _, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        # Index 1 has companion_j == companion_k == 2
        assert not structural_valid[0, 1]

    def test_shape_mismatch_raises_value_error(self):
        """Shape mismatch between distance and clone should raise ValueError."""
        companions_distance = torch.zeros(10, 20)
        companions_clone = torch.zeros(10, 15)  # Different N

        with pytest.raises(ValueError, match="must have the same shape"):
            build_companion_triplets(
                companions_distance=companions_distance,
                companions_clone=companions_clone,
            )

    def test_wrong_ndim_1d_raises_value_error(self):
        """1D input should raise ValueError."""
        companions_distance = torch.zeros(20)
        companions_clone = torch.zeros(20)

        with pytest.raises(ValueError, match="Expected companion arrays with shape"):
            build_companion_triplets(
                companions_distance=companions_distance,
                companions_clone=companions_clone,
            )

    def test_wrong_ndim_3d_raises_value_error(self):
        """3D input should raise ValueError."""
        companions_distance = torch.zeros(10, 20, 5)
        companions_clone = torch.zeros(10, 20, 5)

        with pytest.raises(ValueError, match="Expected companion arrays with shape"):
            build_companion_triplets(
                companions_distance=companions_distance,
                companions_clone=companions_clone,
            )

    def test_t_equals_1_works(self):
        """T=1 (single time frame) should work correctly."""
        N = 5
        companions_distance = torch.arange(N).roll(-1).unsqueeze(0)
        companions_clone = torch.arange(N).roll(-2).unsqueeze(0)

        anchor_idx, companion_j, companion_k, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        assert anchor_idx.shape == (1, N)
        assert companion_j.shape == (1, N)
        assert companion_k.shape == (1, N)
        assert structural_valid.shape == (1, N)
        assert structural_valid.all()  # All valid for N=5 with distinct cyclic shifts

    def test_n_equals_1_all_invalid(self):
        """N=1 should result in all structural_valid being False (all companions are self)."""
        T = 5
        N = 1
        # With N=1, roll gives same index (0)
        companions_distance = torch.zeros(T, N, dtype=torch.long)
        companions_clone = torch.zeros(T, N, dtype=torch.long)

        _, _, _, structural_valid = build_companion_triplets(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
        )

        # All companions reference self (index 0)
        assert not structural_valid.any()


# ===========================================================================
# TestBuildCompanionPairIndices
# ===========================================================================


class TestBuildCompanionPairIndices:
    """Test build_companion_pair_indices function."""

    def test_mode_both_output_shapes(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """mode='both' should output [T,N,2] for pair_indices and structural_valid."""
        pair_indices, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        assert pair_indices.shape == (T, N, 2)
        assert structural_valid.shape == (T, N, 2)
        assert structural_valid.dtype == torch.bool

    def test_mode_distance_output_shapes(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """mode='distance' should output [T,N,1]."""
        pair_indices, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="distance",
        )

        assert pair_indices.shape == (T, N, 1)
        assert structural_valid.shape == (T, N, 1)
        assert structural_valid.dtype == torch.bool

    def test_mode_clone_output_shapes(
        self, T: int, N: int, companions_distance: Tensor, companions_clone: Tensor
    ):
        """mode='clone' should output [T,N,1]."""
        pair_indices, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="clone",
        )

        assert pair_indices.shape == (T, N, 1)
        assert structural_valid.shape == (T, N, 1)
        assert structural_valid.dtype == torch.bool

    def test_mode_both_values(self, companions_distance: Tensor, companions_clone: Tensor):
        """mode='both' should have pair_indices[:,:,0] = distance, [:,:,1] = clone."""
        pair_indices, _ = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        assert torch.equal(pair_indices[:, :, 0], companions_distance)
        assert torch.equal(pair_indices[:, :, 1], companions_clone)

    def test_mode_distance_values(self, companions_distance: Tensor, companions_clone: Tensor):
        """mode='distance' should have pair_indices[:,:,0] = distance."""
        pair_indices, _ = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="distance",
        )

        assert torch.equal(pair_indices[:, :, 0], companions_distance)

    def test_mode_clone_values(self, companions_distance: Tensor, companions_clone: Tensor):
        """mode='clone' should have pair_indices[:,:,0] = clone."""
        pair_indices, _ = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="clone",
        )

        assert torch.equal(pair_indices[:, :, 0], companions_clone)

    def test_invalid_mode_raises_value_error(
        self, companions_distance: Tensor, companions_clone: Tensor
    ):
        """Invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="pair_selection must be one of"):
            build_companion_pair_indices(
                companions_distance=companions_distance,
                companions_clone=companions_clone,
                pair_selection="invalid_mode",
            )

    def test_case_insensitive_mode_both(
        self, companions_distance: Tensor, companions_clone: Tensor
    ):
        """Mode should be case insensitive: 'BOTH' should work."""
        pair_indices, _ = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="BOTH",
        )

        assert pair_indices.shape[2] == 2

    def test_case_insensitive_mode_distance_with_whitespace(
        self, companions_distance: Tensor, companions_clone: Tensor
    ):
        """Mode should handle whitespace: ' Distance ' should work."""
        pair_indices, _ = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection=" Distance ",
        )

        assert pair_indices.shape[2] == 1
        assert torch.equal(pair_indices[:, :, 0], companions_distance)

    def test_out_of_range_companions_invalid(self):
        """Out-of-range companions should have structural_valid False."""
        # Index 2 has out-of-range distance companion
        companions_distance = torch.tensor([[1, 2, -1, 0]])
        companions_clone = torch.tensor([[2, 3, 1, 2]])

        _, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        # Index 2 has invalid distance companion
        assert not structural_valid[0, 2, 0]
        # But clone companion should be valid
        assert structural_valid[0, 2, 1]

    def test_self_referencing_invalid(self):
        """Self-referencing companions should have structural_valid False."""
        # Index 1 has companion_distance = 1 (self)
        companions_distance = torch.tensor([[1, 1, 2, 3]])
        companions_clone = torch.tensor([[2, 3, 0, 1]])

        _, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        # Index 1 has self-referencing distance companion
        assert not structural_valid[0, 1, 0]
        # Clone companion should be valid
        assert structural_valid[0, 1, 1]

    def test_small_case_known_values(self):
        """T=1, N=3 small case with known values."""
        # companions_distance: [0->1, 1->2, 2->0]
        # companions_clone: [0->2, 1->0, 2->1]
        companions_distance = torch.tensor([[1, 2, 0]])
        companions_clone = torch.tensor([[2, 0, 1]])

        pair_indices, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        # Check indices
        assert pair_indices[0, 0, 0] == 1  # distance
        assert pair_indices[0, 0, 1] == 2  # clone
        assert pair_indices[0, 1, 0] == 2
        assert pair_indices[0, 1, 1] == 0
        assert pair_indices[0, 2, 0] == 0
        assert pair_indices[0, 2, 1] == 1

        # All should be valid (distinct, in range, non-self)
        assert structural_valid.all()

    def test_all_companions_self_invalid(self):
        """When all companions equal self, structural_valid should be all False."""
        T = 3
        N = 5
        # All companions point to themselves
        companions_distance = torch.arange(N).unsqueeze(0).expand(T, -1)
        companions_clone = torch.arange(N).unsqueeze(0).expand(T, -1)

        _, structural_valid = build_companion_pair_indices(
            companions_distance=companions_distance,
            companions_clone=companions_clone,
            pair_selection="both",
        )

        # All companions are self-referencing
        assert not structural_valid.any()
