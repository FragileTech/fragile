# Viscous Force Weighting Update

## Summary

Added two new Gaussian kernel weighting options for geodesic distances in the viscous force calculation. These options use the scutoid/Voronoi neighbor graph to compute shortest path (geodesic) distances between walkers, then apply Gaussian kernels to these distances.

## Changes Made

### 1. New Weighting Options

Added to `KineticOperator.viscous_neighbor_weighting`:

- **`kernel_geodesic_euclidean`**: Gaussian kernel on geodesic distance using Euclidean edge lengths
  - Computes shortest paths on the neighbor graph using standard Euclidean distances
  - Applies Gaussian kernel: K(d_geodesic) = exp(-d²/(2l²))
  - Best for uniform spatial metrics

- **`kernel_geodesic_metric`**: Gaussian kernel on geodesic distance using metric-weighted edge lengths
  - Computes shortest paths using fitness Hessian-based metric distances
  - Accounts for anisotropic geometry from the fitness landscape
  - Best for capturing geometric effects in curved spaces

### 2. Implementation Details

#### New Method: `_compute_geodesic_distances`
Location: `src/fragile/fractalai/core/kinetic_operator.py:917`

- Uses Floyd-Warshall algorithm for all-pairs shortest paths (O(N³))
- Computes edge lengths from neighbor graph
- Supports periodic boundary conditions (PBC)
- Optional metric weighting using diffusion tensor

#### Updated Method: `_compute_viscous_force`
Locations:
- Line 826: Added geodesic kernel handling for neighbor_edges case
- Line 730: Added fallback warnings for all-pairs case

### 3. Dashboard Integration

The new options are automatically available in the dashboard through:
- Parameter definition in `KineticOperator` (line 184)
- Widget configuration (line 358)
- Gas configuration panel auto-discovery

## Usage

### Via Python API

```python
from fragile.fractalai.core.kinetic_operator import KineticOperator

kinetic_op = KineticOperator(
    gamma=1.0,
    beta=1.0,
    delta_t=0.1,
    nu=1.0,  # Enable viscous coupling
    use_viscous_coupling=True,
    viscous_neighbor_weighting="kernel_geodesic_euclidean",  # New option!
    viscous_length_scale=0.5,  # Length scale for Gaussian kernel
)
```

### Via Dashboard

1. Open the QFT dashboard or gas configuration panel
2. Navigate to "Langevin Dynamics" section
3. Enable "Use viscous coupling"
4. Select weighting from dropdown:
   - `kernel_geodesic_euclidean` - for uniform metric
   - `kernel_geodesic_metric` - for fitness-based metric

## Requirements

- **Neighbor graph must be recorded**: Set `neighbor_graph_method="delaunay"` or `"voronoi"`
- **Neighbor edges required**: Geodesic options only work when neighbor_edges are available
  - If neighbor_edges is None, the code will fall back to Euclidean kernel with a warning

## Existing Weighting Options (for reference)

- `kernel`: Gaussian kernel on direct Euclidean distance (original)
- `uniform`: Equal weights for all neighbors
- `inverse_distance`: 1/r weighting
- `metric_diag`: Metric-weighted 1/r (diagonal diffusion tensor)
- `metric_full`: Metric-weighted 1/r (full diffusion tensor)

## Performance Considerations

- **Geodesic distance computation**: O(N³) using Floyd-Warshall
  - Suitable for N ≤ 500 walkers (typical QFT simulations use N=200)
  - Computed once per viscous force evaluation
  - Cached within each kinetic step

- **Memory**: O(N²) for distance matrix

## Mathematical Framework

### Euclidean Geodesic Kernel

For neighbor edges (i,j), compute:
- Edge length: d_ij = ||x_i - x_j||
- Geodesic distance: d_geodesic[i,j] = shortest path using Floyd-Warshall
- Kernel weight: w_ij = exp(-d²_geodesic / (2l²))

### Metric-Weighted Geodesic Kernel

Using emergent metric g = H + ε_Σ I (where H is fitness Hessian):
- Metric distance: d²_metric = (x_i - x_j)ᵀ g (x_i - x_j)
- Edge length: d_ij = √(d²_metric)
- Geodesic distance: d_geodesic[i,j] = shortest path with metric edge lengths
- Kernel weight: w_ij = exp(-d²_geodesic / (2l²))

### Viscous Force with Geodesic Kernel

F_viscous(x_i) = ν ∑_j [w_ij / deg(i)] (v_j - v_i)

where:
- ν: viscous coupling strength
- w_ij: geodesic kernel weight
- deg(i) = ∑_k w_ik: local degree (normalization)

## Testing

To verify the implementation:

```python
import torch
from fragile.fractalai.core.kinetic_operator import KineticOperator
from fragile.fractalai.core.euclidean_gas import EuclideanGas, SwarmState

# Create operator with geodesic weighting
kinetic_op = KineticOperator(
    gamma=1.0,
    beta=1.0,
    delta_t=0.1,
    nu=1.0,
    use_viscous_coupling=True,
    viscous_neighbor_weighting="kernel_geodesic_euclidean",
    viscous_length_scale=0.5,
)

# Create simple state
N, d = 50, 3
state = SwarmState(
    x=torch.randn(N, d),
    v=torch.randn(N, d)
)

# Create neighbor edges (e.g., from Delaunay)
neighbor_edges = torch.randint(0, N, (100, 2))

# Compute viscous force
viscous_force = kinetic_op._compute_viscous_force(
    state.x,
    state.v,
    neighbor_edges=neighbor_edges
)

print(f"Viscous force shape: {viscous_force.shape}")
print(f"Viscous force norm: {viscous_force.norm():.4f}")
```

## Files Modified

1. `src/fragile/fractalai/core/kinetic_operator.py`
   - Added `viscous_neighbor_weighting` options (lines 184-193)
   - Updated widget configuration (lines 358-369)
   - Added `_compute_geodesic_distances` method (lines 917-1014)
   - Updated `_compute_viscous_force` with geodesic handling (lines 826-842, 730-757)

## Backward Compatibility

All existing weighting options remain unchanged. The default is still `"kernel"` (Euclidean distance), so no breaking changes.
