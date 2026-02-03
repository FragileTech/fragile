"""Tests for neighbor tracking system with virtual boundaries."""

import pytest
import torch
import numpy as np
from scipy.spatial import Voronoi

from fragile.fractalai.scutoid.neighbors import (
    detect_nearby_boundary_faces,
    project_to_boundary_face,
    project_faces_vectorized,
    estimate_boundary_facet_area,
    compute_boundary_neighbors,
    create_extended_edge_index,
    build_csr_from_coo,
    query_walker_neighbors,
    query_walker_neighbors_vectorized,
)
from fragile.fractalai.scutoid.voronoi import compute_vectorized_voronoi
from fragile.fractalai.bounds import TorchBounds


class TestBoundaryDetection:
    """Test boundary face detection."""

    def test_boundary_detection_2d_left(self):
        """Test detecting left boundary in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.01, 0.5])  # Near left wall

        faces = detect_nearby_boundary_faces(position, bounds, tolerance=0.05)

        assert 0 in faces  # Left wall (face 0)
        assert 1 not in faces  # Not near right wall
        assert 2 not in faces  # Not near bottom
        assert 3 not in faces  # Not near top

    def test_boundary_detection_2d_corner(self):
        """Test detecting corner (two faces) in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.01, 0.02])  # Near bottom-left corner

        faces = detect_nearby_boundary_faces(position, bounds, tolerance=0.05)

        assert 0 in faces  # Left wall
        assert 2 in faces  # Bottom wall
        assert len(faces) == 2

    def test_boundary_detection_3d(self):
        """Test detecting boundary in 3D."""
        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )
        position = torch.tensor([0.5, 0.98, 0.5])  # Near top (y-high)

        faces = detect_nearby_boundary_faces(position, bounds, tolerance=0.05)

        assert 3 in faces  # y-high wall (face 3)

    def test_boundary_detection_interior(self):
        """Test that interior points detect no boundaries."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.5, 0.5])  # Interior

        faces = detect_nearby_boundary_faces(position, bounds, tolerance=0.05)

        assert len(faces) == 0

    def test_boundary_detection_tolerance(self):
        """Test tolerance parameter."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.08, 0.5])  # 0.08 from left wall

        # Should not detect with small tolerance
        faces_small = detect_nearby_boundary_faces(position, bounds, tolerance=0.05)
        assert len(faces_small) == 0

        # Should detect with large tolerance
        faces_large = detect_nearby_boundary_faces(position, bounds, tolerance=0.1)
        assert 0 in faces_large


class TestBoundaryProjection:
    """Test projection onto boundary faces."""

    def test_projection_2d_left(self):
        """Test projection onto left wall in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.1, 0.5])

        proj_pos, normal = project_to_boundary_face(position, face_id=0, bounds=bounds)

        assert torch.allclose(proj_pos, torch.tensor([0.0, 0.5]))
        assert torch.allclose(normal, torch.tensor([-1.0, 0.0]))

    def test_projection_2d_right(self):
        """Test projection onto right wall in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.9, 0.3])

        proj_pos, normal = project_to_boundary_face(position, face_id=1, bounds=bounds)

        assert torch.allclose(proj_pos, torch.tensor([1.0, 0.3]))
        assert torch.allclose(normal, torch.tensor([1.0, 0.0]))

    def test_projection_2d_bottom(self):
        """Test projection onto bottom wall in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.5, 0.1])

        proj_pos, normal = project_to_boundary_face(position, face_id=2, bounds=bounds)

        assert torch.allclose(proj_pos, torch.tensor([0.5, 0.0]))
        assert torch.allclose(normal, torch.tensor([0.0, -1.0]))

    def test_projection_2d_top(self):
        """Test projection onto top wall in 2D."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.7, 0.9])

        proj_pos, normal = project_to_boundary_face(position, face_id=3, bounds=bounds)

        assert torch.allclose(proj_pos, torch.tensor([0.7, 1.0]))
        assert torch.allclose(normal, torch.tensor([0.0, 1.0]))

    def test_projection_3d(self):
        """Test projection in 3D."""
        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )
        position = torch.tensor([0.5, 0.5, 0.1])

        proj_pos, normal = project_to_boundary_face(position, face_id=4, bounds=bounds)

        assert torch.allclose(proj_pos, torch.tensor([0.5, 0.5, 0.0]))
        assert torch.allclose(normal, torch.tensor([0.0, 0.0, -1.0]))


class TestBoundaryFacetArea:
    """Test boundary facet area estimation."""

    def test_facet_area_regular_grid_2d(self):
        """Test facet area estimation on regular 2D grid."""
        # Create 3x3 regular grid
        positions = torch.tensor(
            [
                [0.25, 0.25],
                [0.50, 0.25],
                [0.75, 0.25],
                [0.25, 0.50],
                [0.50, 0.50],
                [0.75, 0.50],
                [0.25, 0.75],
                [0.50, 0.75],
                [0.75, 0.75],
            ],
            dtype=torch.float32,
        )
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        # Build Voronoi
        vor = Voronoi(positions.numpy())

        # Test bottom-left corner walker (index 0)
        # Should have facet area ≈ 0.25 (cell edge length)
        area = estimate_boundary_facet_area(
            walker_idx=0, face_id=0, vor=vor, positions=positions, bounds=bounds
        )

        # Allow some tolerance
        assert 0.2 < area < 0.3

    def test_facet_area_fallback(self):
        """Test that fallback returns reasonable value."""
        # Create simple positions
        positions = torch.tensor([[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]], dtype=torch.float32)
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        vor = Voronoi(positions.numpy())

        # Should not crash and return positive value
        area = estimate_boundary_facet_area(
            walker_idx=0, face_id=0, vor=vor, positions=positions, bounds=bounds
        )

        assert area > 0


class TestBoundaryNeighborComputation:
    """Test virtual boundary neighbor computation."""

    def test_compute_boundary_neighbors_2d(self):
        """Test computing boundary neighbors in 2D."""
        # Create positions near boundaries
        positions = torch.tensor(
            [
                [0.1, 0.5],  # Near left
                [0.9, 0.5],  # Near right
                [0.5, 0.1],  # Near bottom
                [0.5, 0.9],  # Near top
                [0.5, 0.5],  # Interior
            ],
            dtype=torch.float32,
        )

        # Build Voronoi
        vor = Voronoi(positions.numpy())

        # Classify tiers (assume all are tier 0 for testing)
        tier = torch.zeros(len(positions), dtype=torch.long)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        boundary_data = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2
        )

        # Should have boundary neighbors for first 4 walkers
        assert len(boundary_data.positions) >= 4
        assert boundary_data.positions.shape[1] == 2  # 2D
        assert len(boundary_data.normals) == len(boundary_data.positions)
        assert len(boundary_data.facet_areas) == len(boundary_data.positions)
        assert len(boundary_data.distances) == len(boundary_data.positions)
        assert len(boundary_data.walker_indices) == len(boundary_data.positions)

        # Check that normals are unit vectors
        norms = torch.norm(boundary_data.normals, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

        # Check that distances are positive
        assert torch.all(boundary_data.distances > 0)

    def test_compute_boundary_neighbors_tier_filtering(self):
        """Test that only tier 0/1 walkers get boundary neighbors."""
        positions = torch.tensor(
            [
                [0.1, 0.5],  # Tier 0 - near boundary
                [0.5, 0.5],  # Tier 2 - interior (will be filtered)
            ],
            dtype=torch.float32,
        )

        vor = Voronoi(positions.numpy())

        # Only first walker is tier 0
        tier = torch.tensor([0, 2], dtype=torch.long)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        boundary_data = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2
        )

        # Should only have boundary neighbor for tier 0 walker
        assert torch.all(boundary_data.walker_indices == 0)

    def test_compute_boundary_neighbors_3d(self):
        """Test computing boundary neighbors in 3D."""
        positions = torch.tensor(
            [[0.1, 0.5, 0.5], [0.5, 0.5, 0.5]],  # Near x-low  # Interior
            dtype=torch.float32,
        )

        vor = Voronoi(positions.numpy())
        tier = torch.zeros(len(positions), dtype=torch.long)

        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )

        boundary_data = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2
        )

        assert boundary_data.positions.shape[1] == 3  # 3D
        assert len(boundary_data.positions) >= 1


class TestExtendedEdgeIndex:
    """Test creating extended edge index with boundary neighbors."""

    def test_create_extended_edge_index(self):
        """Test extending edge_index with boundary neighbors."""
        # Create simple graph: 0-1, 1-2
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).T  # [2, 2]

        # Create mock boundary data (walker 0 has boundary neighbor)
        from fragile.fractalai.scutoid.neighbors import BoundaryWallData

        boundary_data = BoundaryWallData(
            positions=torch.tensor([[0.0, 0.5]], dtype=torch.float32),  # 1 wall
            normals=torch.tensor([[-1.0, 0.0]], dtype=torch.float32),
            facet_areas=torch.tensor([0.3], dtype=torch.float32),
            distances=torch.tensor([0.1], dtype=torch.float32),
            walker_indices=torch.tensor([0], dtype=torch.long),
            face_ids=torch.tensor([0], dtype=torch.long),
        )

        n_walkers = 3

        edge_index_ext, distances_ext, facets_ext, types_ext = create_extended_edge_index(
            edge_index, boundary_data, n_walkers
        )

        # Should have 3 edges (2 walker + 1 boundary)
        assert edge_index_ext.shape[1] == 3
        assert len(distances_ext) == 3
        assert len(facets_ext) == 3
        assert len(types_ext) == 3

        # Last edge should be (0, 3) - walker 0 to boundary wall at index 3
        assert edge_index_ext[0, 2] == 0
        assert edge_index_ext[1, 2] == 3  # Boundary index = N + 0

        # Types should be [0, 0, 1] (two walker edges, one boundary edge)
        assert torch.equal(types_ext, torch.tensor([0, 0, 1], dtype=torch.long))

    def test_create_extended_edge_index_multiple_boundaries(self):
        """Test with multiple boundary neighbors."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T  # [2, 1]

        from fragile.fractalai.scutoid.neighbors import BoundaryWallData

        # Two walkers, each with one boundary
        boundary_data = BoundaryWallData(
            positions=torch.tensor([[0.0, 0.5], [1.0, 0.5]], dtype=torch.float32),
            normals=torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
            facet_areas=torch.tensor([0.3, 0.3], dtype=torch.float32),
            distances=torch.tensor([0.1, 0.1], dtype=torch.float32),
            walker_indices=torch.tensor([0, 1], dtype=torch.long),
            face_ids=torch.tensor([0, 1], dtype=torch.long),
        )

        n_walkers = 2

        edge_index_ext, _, _, types_ext = create_extended_edge_index(
            edge_index, boundary_data, n_walkers
        )

        # Should have 3 edges (1 walker + 2 boundary)
        assert edge_index_ext.shape[1] == 3

        # Check boundary edges
        boundary_edges = edge_index_ext[:, types_ext == 1]
        assert boundary_edges.shape[1] == 2
        assert torch.equal(boundary_edges[0], torch.tensor([0, 1]))  # Source walkers
        assert torch.equal(boundary_edges[1], torch.tensor([2, 3]))  # Boundary indices


class TestCSRConversion:
    """Test COO to CSR conversion."""

    def test_build_csr_from_coo_simple(self):
        """Test CSR conversion on simple graph."""
        # Graph: 0->1, 0->2, 1->2
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

        n_nodes = 3

        csr_data = build_csr_from_coo(edge_index, n_nodes)

        # Check pointers
        expected_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.long)
        assert torch.equal(csr_data["csr_ptr"], expected_ptr)

        # Check indices
        expected_indices = torch.tensor([1, 2, 2], dtype=torch.long)
        assert torch.equal(csr_data["csr_indices"], expected_indices)

    def test_build_csr_with_edge_data(self):
        """Test CSR conversion with edge attributes."""
        edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)

        distances = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        facet_areas = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        edge_types = torch.tensor([0, 1, 0], dtype=torch.long)

        n_nodes = 3

        csr_data = build_csr_from_coo(
            edge_index,
            n_nodes,
            edge_distances=distances,
            edge_facet_areas=facet_areas,
            edge_types=edge_types,
        )

        # Check that edge data is reordered correctly
        assert len(csr_data["csr_distances"]) == 3
        assert len(csr_data["csr_facet_areas"]) == 3
        assert len(csr_data["csr_types"]) == 3

        # Node 0 has edges 0,1 → distances should be [1.0, 2.0]
        assert torch.equal(csr_data["csr_distances"][:2], torch.tensor([1.0, 2.0]))

    def test_build_csr_disconnected_nodes(self):
        """Test CSR with disconnected nodes."""
        # Node 1 has no outgoing edges
        edge_index = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)

        n_nodes = 3

        csr_data = build_csr_from_coo(edge_index, n_nodes)

        # Pointers: [0, 1, 1, 2]
        expected_ptr = torch.tensor([0, 1, 1, 2], dtype=torch.long)
        assert torch.equal(csr_data["csr_ptr"], expected_ptr)


class TestCSRQueries:
    """Test CSR neighbor queries."""

    def test_query_walker_neighbors(self):
        """Test querying walker neighbors."""
        # Build CSR manually
        csr_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.long)
        csr_indices = torch.tensor([1, 2, 2], dtype=torch.long)
        csr_types = torch.tensor([0, 0, 0], dtype=torch.long)

        # Query node 0
        neighbors = query_walker_neighbors(0, csr_ptr, csr_indices)
        assert torch.equal(neighbors, torch.tensor([1, 2]))

        # Query node 1
        neighbors = query_walker_neighbors(1, csr_ptr, csr_indices)
        assert torch.equal(neighbors, torch.tensor([2]))

        # Query node 2 (no neighbors)
        neighbors = query_walker_neighbors(2, csr_ptr, csr_indices)
        assert len(neighbors) == 0

    def test_query_walker_neighbors_with_filter(self):
        """Test querying with type filtering."""
        csr_ptr = torch.tensor([0, 3, 3], dtype=torch.long)
        csr_indices = torch.tensor([1, 2, 3], dtype=torch.long)
        csr_types = torch.tensor([0, 0, 1], dtype=torch.long)  # Last is boundary

        # Query all neighbors
        neighbors = query_walker_neighbors(0, csr_ptr, csr_indices, csr_types, filter_type=None)
        assert torch.equal(neighbors, torch.tensor([1, 2, 3]))

        # Query walker neighbors only
        neighbors = query_walker_neighbors(0, csr_ptr, csr_indices, csr_types, filter_type=0)
        assert torch.equal(neighbors, torch.tensor([1, 2]))

        # Query boundary neighbors only
        neighbors = query_walker_neighbors(0, csr_ptr, csr_indices, csr_types, filter_type=1)
        assert torch.equal(neighbors, torch.tensor([3]))

    def test_query_walker_neighbors_vectorized(self):
        """Test vectorized neighbor query for all walkers."""
        csr_ptr = torch.tensor([0, 2, 3, 3], dtype=torch.long)
        csr_indices = torch.tensor([1, 2, 2], dtype=torch.long)

        neighbors, mask, counts = query_walker_neighbors_vectorized(csr_ptr, csr_indices)

        assert neighbors.shape == (3, 2)
        assert torch.equal(neighbors[0], torch.tensor([1, 2]))
        assert torch.equal(neighbors[1], torch.tensor([2, -1]))
        assert torch.equal(neighbors[2], torch.tensor([-1, -1]))
        assert torch.equal(
            mask,
            torch.tensor(
                [[True, True], [True, False], [False, False]],
                dtype=torch.bool,
            ),
        )
        assert torch.equal(counts, torch.tensor([2, 1, 0]))

    def test_query_walker_neighbors_vectorized_with_filter(self):
        """Test vectorized neighbor query with type filtering."""
        csr_ptr = torch.tensor([0, 3, 3], dtype=torch.long)
        csr_indices = torch.tensor([1, 2, 3], dtype=torch.long)
        csr_types = torch.tensor([0, 0, 1], dtype=torch.long)

        neighbors, mask, counts = query_walker_neighbors_vectorized(
            csr_ptr, csr_indices, csr_types, filter_type=0
        )
        assert neighbors.shape == (2, 2)
        assert torch.equal(neighbors[0], torch.tensor([1, 2]))
        assert torch.equal(neighbors[1], torch.tensor([-1, -1]))
        assert torch.equal(
            mask,
            torch.tensor([[True, True], [False, False]], dtype=torch.bool),
        )
        assert torch.equal(counts, torch.tensor([2, 0]))

        neighbors, mask, counts = query_walker_neighbors_vectorized(
            csr_ptr, csr_indices, csr_types, filter_type=1
        )
        assert neighbors.shape == (2, 1)
        assert torch.equal(neighbors[0], torch.tensor([3]))
        assert torch.equal(neighbors[1], torch.tensor([-1]))
        assert torch.equal(
            mask,
            torch.tensor([[True], [False]], dtype=torch.bool),
        )
        assert torch.equal(counts, torch.tensor([1, 0]))


class TestVoronoiIntegration:
    """Integration tests with VoronoiTriangulation."""

    def test_voronoi_with_boundaries_2d(self):
        """Test VoronoiTriangulation with virtual boundaries in 2D."""
        # Create 3x3 grid
        x = torch.linspace(0.2, 0.8, 3)
        y = torch.linspace(0.2, 0.8, 3)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        positions = torch.stack([xx.flatten(), yy.flatten()], dim=1)

        n = len(positions)
        alive = torch.ones(n, dtype=torch.bool)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        tri = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=False)

        # Should have boundary neighbors
        assert tri.has_boundary_neighbors
        assert tri.boundary_walls is not None
        assert len(tri.boundary_walls) > 0

        # Check CSR format exists
        assert tri.neighbor_csr_ptr is not None
        assert tri.neighbor_csr_indices is not None
        assert tri.neighbor_csr_types is not None

        # Check that corner walkers have boundary neighbors
        corner_idx = 0  # Bottom-left corner
        neighbors = tri.get_walker_neighbors(corner_idx, include_boundaries=True)
        types_start = tri.neighbor_csr_ptr[corner_idx]
        types_end = tri.neighbor_csr_ptr[corner_idx + 1]
        types = tri.neighbor_csr_types[types_start:types_end]

        # Should have at least one boundary neighbor
        assert torch.any(types == 1)

    def test_voronoi_with_boundaries_3d(self):
        """Test VoronoiTriangulation with virtual boundaries in 3D."""
        # Create small 3D grid
        positions = torch.tensor(
            [
                [0.2, 0.2, 0.2],
                [0.8, 0.2, 0.2],
                [0.2, 0.8, 0.2],
                [0.2, 0.2, 0.8],
                [0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )

        alive = torch.ones(len(positions), dtype=torch.bool)

        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )

        tri = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=False)

        assert tri.has_boundary_neighbors
        assert tri.boundary_walls is not None
        assert tri.boundary_walls.shape[1] == 3  # 3D positions

    def test_voronoi_pbc_no_boundaries(self):
        """Test that PBC mode has no boundary neighbors."""
        positions = torch.tensor(
            [[0.2, 0.2], [0.8, 0.2], [0.2, 0.8], [0.8, 0.8]], dtype=torch.float32
        )

        alive = torch.ones(len(positions), dtype=torch.bool)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        tri = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=True)

        # Should NOT have boundary neighbors
        assert not tri.has_boundary_neighbors
        assert tri.boundary_walls is None

        # CSR format should still exist
        assert tri.neighbor_csr_ptr is not None
        assert tri.neighbor_csr_indices is not None

    def test_walker_neighbor_helper_method(self):
        """Test get_walker_neighbors helper method."""
        positions = torch.tensor(
            [[0.1, 0.5], [0.5, 0.5], [0.9, 0.5]], dtype=torch.float32
        )

        alive = torch.ones(len(positions), dtype=torch.bool)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        tri = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=False)

        # Test querying with boundaries
        neighbors_all = tri.get_walker_neighbors(0, include_boundaries=True)
        assert len(neighbors_all) >= 1

        # Test querying without boundaries
        neighbors_walkers = tri.get_walker_neighbors(0, include_boundaries=False)

        # Should have fewer or equal neighbors
        assert len(neighbors_walkers) <= len(neighbors_all)


class TestVectorizedProjection:
    """Test vectorized projection function."""

    def test_project_faces_vectorized_single_face(self):
        """Test vectorized projection with single face."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.1, 0.5])
        face_ids = torch.tensor([0])  # Left wall

        proj_positions, normals = project_faces_vectorized(
            position, face_ids, bounds, 2, torch.device("cpu"), torch.float32
        )

        assert proj_positions.shape == (1, 2)
        assert normals.shape == (1, 2)
        assert torch.allclose(proj_positions[0], torch.tensor([0.0, 0.5]))
        assert torch.allclose(normals[0], torch.tensor([-1.0, 0.0]))

    def test_project_faces_vectorized_multiple_faces(self):
        """Test vectorized projection with multiple faces (corner case)."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.05, 0.03])  # Near bottom-left corner
        face_ids = torch.tensor([0, 2])  # Left and bottom walls

        proj_positions, normals = project_faces_vectorized(
            position, face_ids, bounds, 2, torch.device("cpu"), torch.float32
        )

        assert proj_positions.shape == (2, 2)
        assert normals.shape == (2, 2)

        # First projection (left wall)
        assert torch.allclose(proj_positions[0], torch.tensor([0.0, 0.03]))
        assert torch.allclose(normals[0], torch.tensor([-1.0, 0.0]))

        # Second projection (bottom wall)
        assert torch.allclose(proj_positions[1], torch.tensor([0.05, 0.0]))
        assert torch.allclose(normals[1], torch.tensor([0.0, -1.0]))

    def test_project_faces_vectorized_3d(self):
        """Test vectorized projection in 3D."""
        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )
        position = torch.tensor([0.5, 0.95, 0.5])
        face_ids = torch.tensor([3])  # y-high wall

        proj_positions, normals = project_faces_vectorized(
            position, face_ids, bounds, 3, torch.device("cpu"), torch.float32
        )

        assert proj_positions.shape == (1, 3)
        assert torch.allclose(proj_positions[0], torch.tensor([0.5, 1.0, 0.5]))
        assert torch.allclose(normals[0], torch.tensor([0.0, 1.0, 0.0]))

    def test_project_faces_vectorized_matches_scalar(self):
        """Test that vectorized version matches scalar projection."""
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))
        position = torch.tensor([0.1, 0.9])

        # Test all 4 faces
        face_ids = torch.tensor([0, 1, 2, 3])

        # Vectorized version
        proj_vec, normals_vec = project_faces_vectorized(
            position, face_ids, bounds, 2, torch.device("cpu"), torch.float32
        )

        # Scalar version (for comparison)
        for i, face_id in enumerate(face_ids):
            proj_scalar, normal_scalar = project_to_boundary_face(
                position, face_id.item(), bounds
            )

            assert torch.allclose(proj_vec[i], proj_scalar)
            assert torch.allclose(normals_vec[i], normal_scalar)


class TestParallelization:
    """Tests for parallel facet area computation."""

    def test_parallel_matches_sequential(self):
        """Verify parallel computation produces identical results to sequential."""
        # Create positions near boundaries in 3D
        np.random.seed(42)
        positions = torch.from_numpy(
            np.random.randn(100, 3).astype(np.float32) * 0.3 + 0.5
        )
        positions = torch.clamp(positions, 0.05, 0.95)

        vor = Voronoi(positions.numpy())
        tier = torch.zeros(len(positions), dtype=torch.long)
        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )

        # Compute with sequential (n_jobs=1)
        result_seq = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=1
        )

        # Compute with parallel (n_jobs=4)
        result_par = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=4
        )

        # Results should match exactly
        assert len(result_seq.positions) == len(result_par.positions)
        assert torch.allclose(result_seq.positions, result_par.positions)
        assert torch.allclose(result_seq.normals, result_par.normals)
        assert torch.allclose(result_seq.distances, result_par.distances)
        assert torch.allclose(result_seq.facet_areas, result_par.facet_areas, rtol=1e-5)
        assert torch.equal(result_seq.walker_indices, result_par.walker_indices)
        assert torch.equal(result_seq.face_ids, result_par.face_ids)

    def test_parallel_small_workload(self):
        """Test that small workloads (<10 pairs) fall back to sequential."""
        # Create just a few positions (need at least d+1 non-colinear points)
        positions = torch.tensor(
            [
                [0.05, 0.5],  # Near left boundary
                [0.95, 0.5],  # Near right boundary
                [0.5, 0.2],   # Interior - lower
                [0.5, 0.8],   # Interior - upper
            ],
            dtype=torch.float32,
        )

        vor = Voronoi(positions.numpy())
        tier = torch.zeros(len(positions), dtype=torch.long)
        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        # Should work without crashing (uses sequential path)
        result = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=4
        )

        assert len(result.positions) >= 1


class TestPerformance:
    """Performance tests for CSR format and vectorization."""

    def test_vectorized_boundary_neighbors_performance(self):
        """Benchmark vectorized boundary neighbor computation."""
        import time

        # Create positions near boundaries in 3D
        np.random.seed(42)
        positions = torch.from_numpy(
            np.random.randn(200, 3).astype(np.float32) * 0.3 + 0.5
        )

        # Clip to ensure some are near boundaries
        positions = torch.clamp(positions, 0.05, 0.95)

        # Build Voronoi
        vor = Voronoi(positions.numpy())

        # All walkers are tier 0 (boundary) for maximum workload
        tier = torch.zeros(len(positions), dtype=torch.long)

        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )

        # Warm up
        _ = compute_boundary_neighbors(positions, tier, bounds, vor, boundary_tolerance=0.2)

        # Benchmark
        num_runs = 5
        times = []
        for _ in range(num_runs):
            start = time.time()
            result = compute_boundary_neighbors(
                positions, tier, bounds, vor, boundary_tolerance=0.2
            )
            times.append(time.time() - start)

        avg_time = np.mean(times)
        print(f"\nVectorized boundary neighbors (200 walkers, 3D): {avg_time*1000:.2f} ms")
        print(f"  Number of boundary pairs: {len(result.positions)}")

        # Performance should be reasonable (< 1 second for 200 walkers)
        assert avg_time < 1.0

    def test_parallel_performance_improvement(self):
        """Benchmark parallel vs sequential facet area computation.

        Note: Parallelization overhead may dominate for small workloads (<500 pairs).
        This test documents the performance characteristics but doesn't enforce
        a speedup threshold.
        """
        import time

        # Create positions with significant boundary workload
        np.random.seed(42)
        positions = torch.from_numpy(
            np.random.randn(200, 3).astype(np.float32) * 0.3 + 0.5
        )
        positions = torch.clamp(positions, 0.05, 0.95)

        vor = Voronoi(positions.numpy())
        tier = torch.zeros(len(positions), dtype=torch.long)
        bounds = TorchBounds(
            low=torch.tensor([0.0, 0.0, 0.0]), high=torch.tensor([1.0, 1.0, 1.0])
        )

        # Warm up
        _ = compute_boundary_neighbors(
            positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=1
        )

        # Benchmark sequential
        num_runs = 3
        times_seq = []
        for _ in range(num_runs):
            start = time.time()
            result_seq = compute_boundary_neighbors(
                positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=1
            )
            times_seq.append(time.time() - start)

        # Benchmark parallel (4 cores)
        times_par = []
        for _ in range(num_runs):
            start = time.time()
            result_par = compute_boundary_neighbors(
                positions, tier, bounds, vor, boundary_tolerance=0.2, n_jobs=4
            )
            times_par.append(time.time() - start)

        avg_seq = np.mean(times_seq)
        avg_par = np.mean(times_par)
        speedup = avg_seq / avg_par

        print(f"\nParallel performance (200 walkers):")
        print(f"  Sequential: {avg_seq*1000:.2f} ms")
        print(f"  Parallel (4 cores): {avg_par*1000:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Number of boundary pairs: {len(result_par.positions)}")

        # Note: For small workloads (~170 pairs), parallelization overhead
        # dominates and sequential is faster. Parallelization becomes beneficial
        # for workloads >500 pairs where facet area computation time exceeds
        # process spawning overhead.
        #
        # This test documents the behavior - no assertion on speedup since
        # small workloads are expected to be slower with parallelization.

    def test_csr_performance_improvement(self):
        """Benchmark CSR vs COO for neighbor queries."""
        import time

        # Create large random graph
        n_nodes = 1000
        n_edges = 5000

        edge_index = torch.randint(0, n_nodes, (2, n_edges), dtype=torch.long)

        # Build CSR
        csr_data = build_csr_from_coo(edge_index, n_nodes)

        # Benchmark COO queries
        num_queries = 100
        query_nodes = torch.randint(0, n_nodes, (num_queries,))

        start = time.time()
        for node in query_nodes:
            mask = edge_index[0] == node
            neighbors_coo = edge_index[1, mask]
        time_coo = time.time() - start

        # Benchmark CSR queries
        start = time.time()
        for node in query_nodes:
            neighbors_csr = query_walker_neighbors(
                node.item(), csr_data["csr_ptr"], csr_data["csr_indices"]
            )
        time_csr = time.time() - start

        # CSR should be faster
        speedup = time_coo / time_csr
        print(f"CSR speedup: {speedup:.2f}x")
        assert speedup > 1.0  # At least some improvement

    def test_boundary_overhead(self):
        """Measure memory overhead of virtual boundaries."""
        positions = torch.randn(1000, 2) * 0.8 + 0.5  # Centered around [0.5, 0.5]

        alive = torch.ones(len(positions), dtype=torch.bool)

        bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

        # With boundaries
        tri_with = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=False)

        # Without boundaries (PBC)
        tri_without = compute_vectorized_voronoi(positions, alive, bounds=bounds, pbc=True)

        # Compare memory (rough estimate via tensor sizes)
        def estimate_memory(tri):
            total = 0
            for attr in dir(tri):
                val = getattr(tri, attr)
                if isinstance(val, torch.Tensor):
                    total += val.numel() * val.element_size()
            return total

        mem_with = estimate_memory(tri_with)
        mem_without = estimate_memory(tri_without)

        overhead = (mem_with - mem_without) / mem_without
        print(f"Memory overhead: {overhead * 100:.1f}%")

        # Should be reasonable (< 50% overhead)
        assert overhead < 0.5
