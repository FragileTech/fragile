# QFT Dashboard Performance Profile Report

## Configuration Profiled
- **N walkers**: 50
- **n_steps**: 10
- **dims**: 4 (3D spatial + 1D Euclidean time)
- **Benchmark**: Voronoi Ricci Scalar
- **Neighbor graph**: Delaunay
- **Viscous coupling**: enabled with geodesic weighting
- **Anisotropic diffusion**: enabled (full tensor)

## Total Runtime
**154 seconds** for 10 steps with 50 walkers (~15.4 seconds per step)

---

## Top Performance Bottlenecks

### 1. **Hessian Computation via Autograd** (67.5% of total time)
| Location | Total Time | Self Time | Calls |
|----------|-----------|-----------|-------|
| `torch.autograd.grad` / `run_backward` | 104.8s | 103.9s | 60,000 |
| `fitness.py:822(compute_hessian)` | 106.9s | 1.8s | 240 |

**Root cause**: The Hessian is computed via nested autograd differentiation. Each step performs 250 backward passes (50 walkers × 5 Hessian diagonal elements).

**File location**: `src/fragile/fractalai/core/fitness.py:822`

### 2. **Voronoi Tessellation** (24.5% of total time)
| Location | Total Time | Self Time | Calls |
|----------|-----------|-----------|-------|
| `voronoi_observables.py:175(compute_voronoi_tessellation)` | 37.8s | 8.2s | 961 |
| `voronoi_observables.py:432(_compute_facet_area)` | 17.3s | 10.7s | 254,597 |
| `voronoi_observables.py:477(_compute_cell_volume)` | 8.3s | 8.1s | 17,677 |

**Root cause**: The VoronoiRicciScalar benchmark calls `compute_voronoi_tessellation` for every potential evaluation (~961 times for 10 steps). Each call computes facet areas and cell volumes using scipy ConvexHull operations.

**File locations**:
- `src/fragile/fractalai/qft/voronoi_observables.py:175`
- `src/fragile/fractalai/qft/voronoi_observables.py:432`
- `src/fragile/fractalai/qft/voronoi_observables.py:477`

### 3. **Neighbor Graph Computation** (4.1% of total time)
| Location | Total Time | Self Time | Calls |
|----------|-----------|-----------|-------|
| `euclidean_gas.py:337(_compute_neighbor_graph)` | 6.3s | 5.0s | 240 |

**Root cause**: Delaunay tessellation is recomputed every step. The scipy.spatial.Delaunay call and edge extraction dominate.

**File location**: `src/fragile/fractalai/core/euclidean_gas.py:337`

### 4. **Numpy Cross Products** (3.4% of total time)
| Location | Total Time | Calls |
|----------|-----------|-------|
| `numpy.core.numeric.cross` | 5.3s | 147,488 |

**Root cause**: Used heavily in Voronoi facet area calculations for 4D geometry.

---

## Call Frequency Analysis

| Function | Calls per Step | Purpose |
|----------|---------------|---------|
| `torch.autograd.grad` | 6,000 | Hessian computation |
| `_compute_facet_area` | 25,460 | Voronoi cell geometry |
| `_compute_cell_volume` | 1,768 | Voronoi cell volumes |
| `compute_voronoi_tessellation` | 96 | Ricci scalar benchmark |

---

## Bottleneck Summary by Component

```
Component                          Time      Percentage
─────────────────────────────────────────────────────────
Autograd Hessian (run_backward)   103.9s      67.5%
Voronoi tessellation               37.8s      24.5%
Neighbor graph (Delaunay)           6.3s       4.1%
Viscous force computation           0.4s       0.3%
Delaunay scutoid computation        1.3s       0.9%
Other                               4.4s       2.7%
─────────────────────────────────────────────────────────
Total                             154.1s     100.0%
```

---

## Optimization Recommendations

### Priority 1: Hessian Computation (67.5% impact)
1. **Use finite-difference Hessian estimation** instead of autograd for diagonal Hessian
2. **Cache Hessian values** between steps when positions change slowly
3. **Consider JAX with vmap** for batched gradient computation

### Priority 2: Voronoi Tessellation (24.5% impact)
1. **Increase `update_every`** parameter for VoronoiRicciScalar to reduce recomputation
2. **Cache Voronoi data** when walker positions haven't changed significantly
3. **Use the scutoid Delaunay path** which is faster (1.3s vs 37.8s per 240 calls)

### Priority 3: Neighbor Graph (4.1% impact)
1. **Increase `neighbor_graph_update_every`** to skip recomputation on every step
2. **Use incremental Delaunay updates** when few walkers move significantly

---

## Key File Locations for Optimization

| File | Line | Function | Issue |
|------|------|----------|-------|
| `core/fitness.py` | 822 | `compute_hessian` | Autograd backward passes |
| `qft/voronoi_observables.py` | 175 | `compute_voronoi_tessellation` | Redundant tessellation |
| `qft/voronoi_observables.py` | 432 | `_compute_facet_area` | Per-facet ConvexHull |
| `qft/voronoi_observables.py` | 477 | `_compute_cell_volume` | Per-cell ConvexHull |
| `core/euclidean_gas.py` | 337 | `_compute_neighbor_graph` | Delaunay per step |
| `core/benchmarks.py` | 1076 | `voronoi_ricci_potential` | Triggers tessellation |

---

## Scaling Estimate

With N=200 and n_steps=100 (the target configuration):
- Hessian: ~4× more work per step (800 walkers vs 50) → ~400s per step
- Voronoi: ~4× more tessellation work → ~150s per step
- **Estimated total**: ~55,000s (~15 hours) for 100 steps

This explains why the original profiling attempt with N=200, n_steps=100 timed out after 4+ hours.
