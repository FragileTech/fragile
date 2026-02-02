# Neighbor Tracking System - Usage Guide

## Overview

The neighbor tracking system provides:
1. **Virtual boundary neighbors** - Domain walls treated as "virtual walkers"
2. **CSR format** - Fast O(1) neighbor queries
3. **Automatic boundary handling** - No configuration needed when `pbc=False`
4. **Multi-dimensional support** - Works in 2D, 3D, 4D, and higher dimensions

## Basic Usage

```python
import torch
from fragile.fractalai.bounds import TorchBounds
from fragile.fractalai.scutoid.voronoi import compute_vectorized_voronoi

# Create positions and bounds
positions = torch.randn(100, 2) * 0.4 + 0.5  # Centered in [0, 1]^2
alive = torch.ones(len(positions), dtype=torch.bool)
bounds = TorchBounds(low=torch.tensor([0.0, 0.0]), high=torch.tensor([1.0, 1.0]))

# Build tessellation with virtual boundaries
# Note: boundary_tolerance should be >> distance to boundary for boundary walkers
tri = compute_vectorized_voronoi(
    positions, alive,
    bounds=bounds,
    pbc=False,  # Enables virtual boundaries
    boundary_tolerance=0.3  # Detect boundaries within 0.3 units
)

# Check results
print(f"Has boundary neighbors: {tri.has_boundary_neighbors}")  # True
print(f"Number of boundary walls: {len(tri.boundary_walls)}")  # > 0
```

## Querying Neighbors

### Using Helper Methods

```python
# Get all neighbors (walkers + boundaries)
walker_idx = 0
all_neighbors = tri.get_walker_neighbors(walker_idx, include_boundaries=True)

# Get only walker neighbors
walker_neighbors = tri.get_walker_neighbors(walker_idx, include_boundaries=False)

# Neighbor indices:
# - Walker indices: 0, 1, ..., N-1
# - Boundary indices: N, N+1, ..., N+W-1
```

### Using CSR Format Directly

```python
# Direct CSR access for tight loops
for i in range(tri.n_walkers):
    # O(1) access to neighbor range
    start = tri.neighbor_csr_ptr[i]
    end = tri.neighbor_csr_ptr[i + 1]

    # Get neighbor data
    neighbors = tri.neighbor_csr_indices[start:end]
    distances = tri.neighbor_csr_distances[start:end]
    facet_areas = tri.neighbor_csr_facet_areas[start:end]
    types = tri.neighbor_csr_types[start:end]  # 0=walker, 1=boundary

    # Filter by type
    walker_mask = types == 0
    boundary_mask = types == 1

    walker_neighbors = neighbors[walker_mask]
    boundary_neighbors = neighbors[boundary_mask]

    # Process neighbors...
```

## Boundary Wall Properties

```python
# Access boundary wall data
for i in range(len(tri.boundary_walls)):
    walker_idx = tri.boundary_wall_walker_indices[i]  # Which walker
    wall_position = tri.boundary_walls[i]  # [d] position on boundary
    wall_normal = tri.boundary_wall_normals[i]  # [d] outward unit normal
    distance = tri.boundary_wall_distances[i]  # Distance from walker to wall
    facet_area = tri.boundary_wall_facet_areas[i]  # Intersection area

    # Example: Compute boundary flux
    gradient = get_gradient_at_walker(walker_idx)
    flux = torch.dot(gradient, wall_normal) * facet_area
```

## Boundary Tolerance

The `boundary_tolerance` parameter determines which walkers are considered "near" boundaries:

```python
# Small tolerance (default: 1e-6) - only walkers very close to walls
tri = compute_vectorized_voronoi(
    positions, alive, bounds=bounds, pbc=False, boundary_tolerance=1e-6
)

# Large tolerance - walkers within 0.3 units of walls
tri = compute_vectorized_voronoi(
    positions, alive, bounds=bounds, pbc=False, boundary_tolerance=0.3
)
```

**Rule of thumb**: Set `boundary_tolerance` slightly larger than the distance from your boundary walkers to the domain walls.

## Performance

CSR format provides 1.5-5x speedup over COO filtering:

```python
import time

# CSR query (fast)
start = time.time()
for i in range(1000):
    neighbors = tri.get_walker_neighbors(i % tri.n_walkers)
time_csr = time.time() - start

# COO query (slow)
start = time.time()
for i in range(1000):
    idx = i % tri.n_walkers
    mask = tri.edge_index[0] == idx
    neighbors = tri.edge_index[1, mask]
time_coo = time.time() - start

print(f"Speedup: {time_coo / time_csr:.2f}x")
```

## Gradient Estimation Example

```python
# Gradient estimation with boundary handling
gradient = torch.zeros(tri.n_walkers, 2)

for i in range(tri.n_walkers):
    start = tri.neighbor_csr_ptr[i]
    end = tri.neighbor_csr_ptr[i + 1]

    neighbors = tri.neighbor_csr_indices[start:end]
    types = tri.neighbor_csr_types[start:end]

    # Filter to walker neighbors (exclude boundaries for gradient)
    walker_mask = types == 0
    walker_neighbors = neighbors[walker_mask]

    if len(walker_neighbors) < 2:
        continue  # Need at least 2 neighbors

    # Compute finite differences
    Δx = positions[walker_neighbors] - positions[i]
    ΔV = fitness[walker_neighbors] - fitness[i]

    # Least squares gradient
    # ... (standard gradient estimation logic)
```

## Boundary Flux Computation

```python
# Compute flux through domain boundaries
total_flux = 0.0

for i in range(len(tri.boundary_walls)):
    walker_idx = tri.boundary_wall_walker_indices[i]
    wall_normal = tri.boundary_wall_normals[i]
    facet_area = tri.boundary_wall_facet_areas[i]

    # Get gradient at walker
    grad = gradient_field[walker_idx]

    # Flux = gradient · normal * area
    flux = torch.dot(grad, wall_normal) * facet_area
    total_flux += flux

print(f"Total boundary flux: {total_flux}")
```

## 4D and Higher Dimensions

The system fully supports 4D and higher dimensional spaces:

```python
# 4D example
positions_4d = torch.randn(100, 4) * 0.4 + 0.5  # [N, 4]
alive = torch.ones(len(positions_4d), dtype=torch.bool)
bounds_4d = TorchBounds(
    low=torch.zeros(4),
    high=torch.ones(4)
)

tri_4d = compute_vectorized_voronoi(
    positions_4d, alive,
    bounds=bounds_4d,
    pbc=False,
    boundary_tolerance=0.3
)

print(f"4D boundary walls: {len(tri_4d.boundary_walls)}")
print(f"Wall shape: {tri_4d.boundary_walls.shape}")  # [W, 4]

# Boundary facets in 4D are 3D polyhedra
for i in range(len(tri_4d.boundary_walls)):
    facet_volume = tri_4d.boundary_wall_facet_areas[i]  # 3D volume
    normal = tri_4d.boundary_wall_normals[i]  # [4] unit vector
```

Face ID convention for 4D:
- Face 0, 1: x-low, x-high
- Face 2, 3: y-low, y-high
- Face 4, 5: z-low, z-high
- Face 6, 7: w-low, w-high

## Periodic Boundaries (PBC)

When using PBC, no virtual boundaries are created:

```python
# PBC mode - no boundary neighbors
tri = compute_vectorized_voronoi(
    positions, alive, bounds=bounds, pbc=True
)

print(f"Has boundary neighbors: {tri.has_boundary_neighbors}")  # False
print(f"Boundary walls: {tri.boundary_walls}")  # None

# CSR format still available
neighbors = tri.get_walker_neighbors(0)  # Only walker neighbors
```

## Compatibility

The new fields are fully compatible with existing code:

```python
# Existing fields still work
edge_index = tri.edge_index  # [2, E] walker-walker edges only
edge_distances = tri.edge_distances
facet_areas = tri.facet_areas
tier = tri.tier  # Boundary classification

# New fields are optional
if tri.has_boundary_neighbors:
    # Use boundary information
    boundary_walls = tri.boundary_walls
else:
    # Fallback to walker-only edges
    pass
```

## Edge Index Conventions

- **`edge_index`**: [2, E] walker-to-walker edges only (original)
- **`edge_index_extended`**: [2, E+W] includes boundary edges (if `pbc=False`)
  - Walker-walker edges: `(i, j)` where `i, j < N`
  - Walker-boundary edges: `(i, N+k)` where `i < N`, `k` is wall index

```python
if tri.edge_index_extended is not None:
    # Use extended edges (includes boundaries)
    edge_index = tri.edge_index_extended
else:
    # Use original edges (walker-walker only)
    edge_index = tri.edge_index
```

## Memory Overhead

Typical memory overhead: 10-20% for boundary data + CSR format

```python
import sys

def get_tensor_memory(tri):
    total = 0
    for attr in dir(tri):
        val = getattr(tri, attr)
        if isinstance(val, torch.Tensor):
            total += val.numel() * val.element_size()
    return total

mem_with_boundaries = get_tensor_memory(tri)
# Compare with PBC tessellation to estimate overhead
```

## Troubleshooting

### No boundary neighbors detected

**Problem**: `tri.boundary_walls` is empty
**Solution**: Increase `boundary_tolerance` parameter

```python
# Too small - walkers at 0.2 won't be detected
tri = compute_vectorized_voronoi(..., boundary_tolerance=0.1)

# Good - walkers within 0.25 units detected
tri = compute_vectorized_voronoi(..., boundary_tolerance=0.25)
```

### Index out of bounds

**Problem**: Trying to access boundary indices as walkers
**Solution**: Filter by `neighbor_csr_types` first

```python
# WRONG - boundary indices may be >= N
neighbors = tri.neighbor_csr_indices[start:end]
values = fitness[neighbors]  # ERROR if neighbors includes N+k

# CORRECT - filter to walker indices
types = tri.neighbor_csr_types[start:end]
walker_mask = types == 0
walker_neighbors = neighbors[walker_mask]
values = fitness[walker_neighbors]  # OK
```
