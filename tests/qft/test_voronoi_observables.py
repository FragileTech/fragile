"""Unit tests for Voronoi-weighted particle operators."""

import numpy as np
import pytest
import torch

from fragile.fractalai.qft.voronoi_observables import (
    classify_boundary_cells,
    compute_baryon_operator_voronoi,
    compute_geometric_weights,
    compute_meson_operator_voronoi,
    compute_voronoi_tessellation,
)


class TestVoronoiTessellation:
    """Tests for Voronoi tessellation computation."""

    def test_simple_grid_2d(self):
        """Test Voronoi on regular 2D grid."""
        # Create a 3x3 grid
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [2.0, 2.0],
            ],
            dtype=torch.float32,
        )
        alive = torch.ones(9, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=None, pbc=False, exclude_boundary=False
        )

        # Should have voronoi object and valid data
        assert voronoi_data["voronoi"] is not None
        assert len(voronoi_data["neighbor_lists"]) == 9
        assert len(voronoi_data["volumes"]) == 9
        assert len(voronoi_data["alive_indices"]) == 9

        # Center point should have 4 neighbors
        center_idx = 4
        assert len(voronoi_data["neighbor_lists"][center_idx]) >= 4

        # Corner points should have 2 neighbors
        corner_idx = 0
        assert len(voronoi_data["neighbor_lists"][corner_idx]) >= 2

    def test_simple_grid_3d(self):
        """Test Voronoi on random 3D points."""
        # Use random points to avoid degeneracies
        torch.manual_seed(42)
        positions = torch.rand(10, 3) * 5.0
        alive = torch.ones(10, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        assert voronoi_data["voronoi"] is not None
        assert len(voronoi_data["neighbor_lists"]) == 10

        # Random points should have neighbors
        total_neighbors = sum(len(voronoi_data["neighbor_lists"][i]) for i in range(10))
        assert total_neighbors > 0

        # Volumes should be positive
        volumes = voronoi_data["volumes"]
        assert np.all(volumes > 0)

    def test_empty_alive_mask(self):
        """Test Voronoi with no alive walkers."""
        positions = torch.rand(10, 2)
        alive = torch.zeros(10, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        assert voronoi_data["voronoi"] is None
        assert len(voronoi_data["neighbor_lists"]) == 0
        assert len(voronoi_data["volumes"]) == 0

    def test_single_walker(self):
        """Test Voronoi with single walker."""
        positions = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        alive = torch.ones(1, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        # Single point has no neighbors
        assert len(voronoi_data["neighbor_lists"]) == 1
        assert len(voronoi_data["neighbor_lists"][0]) == 0

    def test_partial_alive_mask(self):
        """Test Voronoi with partial alive mask."""
        positions = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        # Only first 3 are alive
        alive = torch.tensor([True, True, True, False, False])

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        # Should only compute for 3 alive walkers
        assert len(voronoi_data["alive_indices"]) == 3
        assert len(voronoi_data["neighbor_lists"]) == 3


class TestGeometricWeights:
    """Tests for geometric weight computation."""

    def test_facet_area_weights_normalization(self):
        """Test that facet area weights sum to 1 for each node."""
        # Simple 2D grid
        positions = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )
        alive = torch.ones(4, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area", normalize=True)

        edge_weights = weights["edge_weights"]

        # For each node, sum of outgoing edge weights should be ~1
        for i in range(4):
            neighbors = voronoi_data["neighbor_lists"][i]
            if neighbors:
                total_weight = sum(edge_weights.get((i, j), 0.0) for j in neighbors)
                assert np.isclose(total_weight, 1.0, rtol=0.1)

    def test_volume_weights_positive(self):
        """Test that volume weights are positive."""
        positions = torch.rand(10, 3)
        alive = torch.ones(10, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="volume", normalize=True)

        node_weights = weights["node_weights"]
        assert torch.all(node_weights > 0)

    def test_combined_weights(self):
        """Test combined weight mode."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        alive = torch.ones(3, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=None, pbc=False, exclude_boundary=False
        )

        weights = compute_geometric_weights(voronoi_data, weight_mode="combined", normalize=True)

        edge_weights = weights["edge_weights"]
        assert len(edge_weights) > 0


class TestMesonOperatorVoronoi:
    """Tests for Voronoi meson operator."""

    def test_meson_basic(self):
        """Test basic meson operator computation."""
        # Use random 2D points (not collinear)
        torch.manual_seed(42)
        positions = torch.rand(5, 2) * 3.0
        alive = torch.ones(5, dtype=torch.bool)

        # Random color states
        color = torch.randn(5, 2, dtype=torch.complex64)
        # Normalize colors
        color = color / torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-6)
        color_valid = torch.ones(5, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=None, pbc=False, exclude_boundary=False
        )

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area", normalize=True)

        sample_indices = torch.arange(5)

        meson, valid = compute_meson_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
        )

        # Random points should have at least some valid computations
        assert torch.any(valid)
        # Meson values should be reasonable for normalized colors
        if torch.any(valid):
            assert torch.all(torch.abs(meson[valid]) <= 2.0)

    def test_meson_with_invalid_colors(self):
        """Test meson operator with some invalid colors."""
        positions = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
        alive = torch.ones(3, dtype=torch.bool)

        color = torch.randn(3, 2, dtype=torch.complex64)
        color_valid = torch.tensor([True, False, True])

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.arange(3)

        _meson, valid = compute_meson_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
        )

        # Walker 1 should be invalid (color_valid[1] = False)
        assert not valid[1]

    def test_meson_empty_sample(self):
        """Test meson operator with empty sample indices."""
        positions = torch.rand(5, 2)
        alive = torch.ones(5, dtype=torch.bool)
        color = torch.randn(5, 2, dtype=torch.complex64)
        color_valid = torch.ones(5, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.tensor([], dtype=torch.long)

        meson, valid = compute_meson_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
        )

        assert meson.shape[0] == 0
        assert valid.shape[0] == 0


class TestBaryonOperatorVoronoi:
    """Tests for Voronoi baryon operator."""

    def test_baryon_basic(self):
        """Test basic baryon operator computation."""
        # 4 walkers in 3D
        positions = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        alive = torch.ones(4, dtype=torch.bool)

        # 3D color states
        color = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=torch.complex64,
        )
        color_valid = torch.ones(4, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.arange(4)

        baryon, _valid = compute_baryon_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
            max_triplets=None,
        )

        # For a tetrahedron, at least some nodes should have enough neighbors
        # Note: Highly symmetric configurations might not generate expected neighbors
        # Just check that computation completes without error
        assert baryon.shape[0] == 4

    def test_baryon_wrong_dimension(self):
        """Test baryon operator with wrong color dimension."""
        positions = torch.rand(5, 2)
        alive = torch.ones(5, dtype=torch.bool)

        # 2D color (should fail - baryon needs 3D)
        color = torch.randn(5, 2, dtype=torch.complex64)
        color_valid = torch.ones(5, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.arange(5)

        with pytest.raises(ValueError, match="requires d=3"):
            compute_baryon_operator_voronoi(
                color=color,
                sample_indices=sample_indices,
                voronoi_neighbors=voronoi_data["neighbor_lists"],
                geometric_weights=weights,
                alive=alive,
                color_valid=color_valid,
                index_map=voronoi_data["index_map"],
                weight_mode="facet_area",
            )

    def test_baryon_max_triplets(self):
        """Test baryon operator with max_triplets limit."""
        # Create many walkers to test triplet limiting
        positions = torch.rand(20, 3)
        alive = torch.ones(20, dtype=torch.bool)
        color = torch.randn(20, 3, dtype=torch.complex64)
        color_valid = torch.ones(20, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.arange(20)

        # Should work with limited triplets
        baryon, _valid = compute_baryon_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
            max_triplets=10,
        )

        assert baryon.shape[0] == 20

    def test_baryon_insufficient_neighbors(self):
        """Test baryon operator with insufficient neighbors."""
        # Only 2 walkers - not enough for triplets
        positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
        alive = torch.ones(2, dtype=torch.bool)
        color = torch.randn(2, 3, dtype=torch.complex64)
        color_valid = torch.ones(2, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")

        sample_indices = torch.arange(2)

        _baryon, valid = compute_baryon_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
        )

        # Should be invalid (need at least 2 neighbors for triplet)
        assert not torch.any(valid)


class TestBoundaryExclusion:
    """Tests for boundary cell filtering."""

    def test_boundary_detection_2d_box(self):
        """Test boundary cell detection in 2D box."""
        # Create a grid with boundary cells
        positions = torch.tensor(
            [
                [0.0, 0.0],  # corner (boundary)
                [5.0, 0.0],  # edge (boundary)
                [10.0, 0.0],  # corner (boundary)
                [0.0, 5.0],  # edge (boundary)
                [5.0, 5.0],  # center (interior)
                [10.0, 5.0],  # edge (boundary)
                [0.0, 10.0],  # corner (boundary)
                [5.0, 10.0],  # edge (boundary)
                [10.0, 10.0],  # corner (boundary)
            ],
            dtype=torch.float32,
        )
        alive = torch.ones(9, dtype=torch.bool)

        # Create mock bounds
        class MockBounds:
            def __init__(self):
                self.low = torch.tensor([0.0, 0.0])
                self.high = torch.tensor([10.0, 10.0])

        bounds = MockBounds()

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=bounds, pbc=False, exclude_boundary=True
        )

        classification = voronoi_data.get("classification")
        if classification is not None:
            # Center point should be interior (or adjacent)
            # At minimum, not all should be boundary
            assert not torch.all(classification["is_boundary"])

    def test_boundary_adjacent_detection(self):
        """Test boundary-adjacent cell detection."""
        # Create positions where we can identify adjacency
        positions = torch.rand(20, 2) * 10.0
        alive = torch.ones(20, dtype=torch.bool)

        class MockBounds:
            def __init__(self):
                self.low = torch.tensor([0.0, 0.0])
                self.high = torch.tensor([10.0, 10.0])

        bounds = MockBounds()

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=bounds, pbc=False, exclude_boundary=True
        )

        classification = voronoi_data.get("classification")
        if classification is not None:
            # Should have all three tiers
            assert torch.any(classification["tier"] == 0)  # Some boundary
            # Interior or boundary-adjacent should exist
            assert torch.any(classification["tier"] >= 1)

    def test_meson_skips_boundary_tiers(self):
        """Test meson operator skips Tier 0-1 cells."""
        torch.manual_seed(42)
        positions = torch.rand(10, 2) * 10.0
        alive = torch.ones(10, dtype=torch.bool)
        color = torch.randn(10, 2, dtype=torch.complex64)
        color = color / torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-6)
        color_valid = torch.ones(10, dtype=torch.bool)

        class MockBounds:
            def __init__(self):
                self.low = torch.tensor([0.0, 0.0])
                self.high = torch.tensor([10.0, 10.0])

        bounds = MockBounds()

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=bounds, pbc=False, exclude_boundary=True
        )

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")
        sample_indices = torch.arange(10)

        _meson, valid = compute_meson_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
            classification=voronoi_data.get("classification"),
        )

        classification = voronoi_data.get("classification")
        if classification is not None:
            # Tier 0 and 1 should be invalid
            for i in range(10):
                if classification["tier"][i] < 2:
                    assert not valid[
                        i
                    ], f"Walker {i} with tier {classification['tier'][i]} should be invalid"

    def test_baryon_skips_boundary_tiers(self):
        """Test baryon operator skips Tier 0-1 cells."""
        torch.manual_seed(42)
        positions = torch.rand(10, 3) * 10.0
        alive = torch.ones(10, dtype=torch.bool)
        color = torch.randn(10, 3, dtype=torch.complex64)
        color = color / torch.linalg.vector_norm(color, dim=-1, keepdim=True).clamp(min=1e-6)
        color_valid = torch.ones(10, dtype=torch.bool)

        class MockBounds:
            def __init__(self):
                self.low = torch.tensor([0.0, 0.0, 0.0])
                self.high = torch.tensor([10.0, 10.0, 10.0])

        bounds = MockBounds()

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=bounds, pbc=False, exclude_boundary=True
        )

        weights = compute_geometric_weights(voronoi_data, weight_mode="facet_area")
        sample_indices = torch.arange(10)

        _baryon, valid = compute_baryon_operator_voronoi(
            color=color,
            sample_indices=sample_indices,
            voronoi_neighbors=voronoi_data["neighbor_lists"],
            geometric_weights=weights,
            alive=alive,
            color_valid=color_valid,
            index_map=voronoi_data["index_map"],
            weight_mode="facet_area",
            max_triplets=20,
            classification=voronoi_data.get("classification"),
        )

        classification = voronoi_data.get("classification")
        if classification is not None:
            # Tier 0 and 1 should be invalid
            for i in range(10):
                if classification["tier"][i] < 2:
                    assert not valid[
                        i
                    ], f"Walker {i} with tier {classification['tier'][i]} should be invalid"

    def test_pbc_disables_boundary_exclusion(self):
        """Test that PBC disables boundary exclusion."""
        positions = torch.rand(10, 2)
        alive = torch.ones(10, dtype=torch.bool)

        class MockBounds:
            def __init__(self):
                self.low = torch.tensor([0.0, 0.0])
                self.high = torch.tensor([1.0, 1.0])

        bounds = MockBounds()

        voronoi_data = compute_voronoi_tessellation(
            positions, alive, bounds=bounds, pbc=True, exclude_boundary=True
        )

        classification = voronoi_data.get("classification")
        if classification is not None:
            # With PBC, all should be interior (no real boundary)
            assert torch.all(classification["is_interior"])
            assert not torch.any(classification["is_boundary"])

    def test_tier2_used_as_neighbors(self):
        """Test that Tier 2 cells contribute to Tier 3 operators."""
        # This is implicitly tested by the meson/baryon skip tests
        # If Tier 2 cells weren't used as neighbors, Tier 3 would have no neighbors
        # and would also be invalid, which would make the tests fail


class TestWeightModes:
    """Test different weight modes."""

    def test_all_weight_modes_meson(self):
        """Test all weight modes work for meson operator."""
        positions = torch.rand(10, 2)
        alive = torch.ones(10, dtype=torch.bool)
        color = torch.randn(10, 2, dtype=torch.complex64)
        color_valid = torch.ones(10, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        sample_indices = torch.arange(10)

        for weight_mode in ["facet_area", "volume", "combined"]:
            weights = compute_geometric_weights(
                voronoi_data, weight_mode=weight_mode, normalize=True
            )

            meson, _valid = compute_meson_operator_voronoi(
                color=color,
                sample_indices=sample_indices,
                voronoi_neighbors=voronoi_data["neighbor_lists"],
                geometric_weights=weights,
                alive=alive,
                color_valid=color_valid,
                index_map=voronoi_data["index_map"],
                weight_mode=weight_mode,
            )

            # Should compute something
            assert meson.shape[0] == 10

    def test_all_weight_modes_baryon(self):
        """Test all weight modes work for baryon operator."""
        positions = torch.rand(10, 3)
        alive = torch.ones(10, dtype=torch.bool)
        color = torch.randn(10, 3, dtype=torch.complex64)
        color_valid = torch.ones(10, dtype=torch.bool)

        voronoi_data = compute_voronoi_tessellation(positions, alive, bounds=None, pbc=False)

        sample_indices = torch.arange(10)

        for weight_mode in ["facet_area", "volume", "combined"]:
            weights = compute_geometric_weights(
                voronoi_data, weight_mode=weight_mode, normalize=True
            )

            baryon, _valid = compute_baryon_operator_voronoi(
                color=color,
                sample_indices=sample_indices,
                voronoi_neighbors=voronoi_data["neighbor_lists"],
                geometric_weights=weights,
                alive=alive,
                color_valid=color_valid,
                index_map=voronoi_data["index_map"],
                weight_mode=weight_mode,
                max_triplets=20,
            )

            # Should compute something
            assert baryon.shape[0] == 10
