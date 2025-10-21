# Scutoid Metric Correction Implementation Summary

## Overview

This document explains the metric correction feature added to the scutoids module to bridge the gap between flat-space deficit angle curvature and the emergent Riemannian geometry induced by anisotropic diffusion.

## The Problem

The original scutoid implementation had a fundamental conceptual issue:

1. **Deficit angles** computed from Euclidean Voronoi tessellation measure **intrinsic curvature** of walker configurations
2. **Anisotropic diffusion** tensor `Σ = (H + ε_Σ I)^{-1/2}` induces **extrinsic curvature** from the fitness landscape
3. These were **completely disconnected** - the deficit angles didn't know about the diffusion tensor

The framework documents claim these should converge to the same Ricci scalar (Theorem 5.4.1), but only if:
- Voronoi tessellation uses **Riemannian geodesic distances** (computationally expensive)
- Walkers have **adapted to anisotropic diffusion** over time

## The Solution: Metric Correction

We implement **first-order perturbation theory** to correct flat-space deficit angles with local metric information:

```
R^manifold(x_i) ≈ R^flat(x_i) + ΔR^metric(x_i)
```

This avoids expensive Riemannian Voronoi computation while capturing essential metric effects.

## Three Correction Modes

### 1. No Correction (`metric_correction='none'`)

**What it computes:**
- Pure flat-space deficit angles from Euclidean Voronoi tessellation
- Intrinsic curvature of walker configuration, independent of fitness landscape

**When to use:**
- Analyzing pure geometric properties of walker distributions
- When fitness landscape has no anisotropic diffusion
- As a baseline for comparison

**Complexity:** O(N log N) (Voronoi tessellation)

### 2. Diagonal Correction (`metric_correction='diagonal'`)

**What it computes:**
- Flat deficit angles + diagonal metric correction
- Formula: `ΔR ≈ (1/2)Σ_k ∂²g_kk/∂x_k²`
- Uses only diagonal components of metric tensor `g = H + ε_Σ I`

**How it works:**
- Estimates local density scale from neighbor distances
- Approximates second derivatives along coordinate axes
- No explicit Hessian computation needed (uses proxy)

**When to use:**
- Need metric correction but computational budget is tight
- Fitness landscape is not strongly anisotropic
- Quick exploratory analysis

**Complexity:** O(N) correction + O(N log N) tessellation

**Limitations:**
- Less accurate for highly anisotropic fitness landscapes
- Uses density as proxy for true metric (simplified)

### 3. Full Correction (`metric_correction='full'`)

**What it computes:**
- Flat deficit angles + full metric tensor correction
- Formula: `ΔR = (1/2)∇²(tr h) - (1/4)||∇h||²`
- Uses neighbor finite differences to estimate metric gradients

**How it works:**
- For each walker, examines all k ≈ 6 neighbors
- Computes metric differences in each direction
- Estimates gradient and Laplacian of metric perturbation
- Applies full perturbation formula

**When to use:**
- Need accurate coupling between walker configuration and fitness geometry
- Fitness landscape has strong anisotropic features
- Publication-quality results required

**Complexity:** O(N·k) correction + O(N log N) tessellation

**Limitations:**
- More expensive than diagonal correction
- Still approximation (not true Riemannian Voronoi)

## Implementation Details

### Key Methods

1. **`compute_ricci_scalars()`**
   - Computes flat-space deficit angles (always)
   - Automatically calls `compute_metric_corrected_ricci()` if correction enabled
   - Stores both flat and corrected values

2. **`compute_metric_corrected_ricci()`**
   - Applies correction based on `self.metric_correction` mode
   - Calls `_compute_diagonal_metric_correction()` or `_compute_full_metric_correction()`
   - Stores results in `self.ricci_scalars_corrected`

3. **`get_ricci_scalars()`**
   - Returns appropriate array based on correction mode
   - Returns corrected values if available, otherwise flat

### Usage Example

```python
from fragile.core.history import RunHistory
from fragile.core.scutoids import create_scutoid_history

# Load experiment data
history = RunHistory.load("experiment.pt")

# Create with metric correction
scutoid_hist = create_scutoid_history(
    history,
    metric_correction='diagonal'  # or 'full' or 'none'
)

# Build tessellation and compute curvature
scutoid_hist.build_tessellation()
scutoid_hist.compute_ricci_scalars()  # Automatically applies correction

# Get corrected Ricci scalars
ricci = scutoid_hist.get_ricci_scalars()
```

## Mathematical Justification

### From Differential Geometry

For small metric perturbations `g_ij = δ_ij + h_ij`:

```
R = R^flat + (1/2)∇²(tr h) - (1/4)h^ij ∂_i∂_j h_kk + O(||h||²)
```

Where:
- `R^flat`: Flat-space curvature (from deficit angles)
- `h = g - I`: Metric perturbation
- `∇²(tr h)`: Laplacian of metric trace (captured by diagonal correction)
- Second term: Full tensor coupling (captured by full correction)

### Connection to Framework

The emergent metric from anisotropic diffusion is:

```
g_ij(x,t) = H_ij(x,t) + ε_Σ δ_ij
```

Where `H = ∇²V_fit` is the fitness Hessian.

**Theorem 5.4.1** states deficit angles converge to Ricci scalar of this metric:

```
lim_{diam(V_i) → 0} δ_i / Vol(∂V_i) = C(d) · R(x_i)
```

Our corrections provide **first-order approximation** to this limit without requiring true Riemannian Voronoi.

## Physical Interpretation

### What Each Component Measures

| Component | Measures | Source |
|-----------|----------|--------|
| R^flat | Intrinsic curvature | Walker spatial configuration |
| ΔR^metric | Extrinsic curvature | Fitness landscape geometry |
| R^corrected | Total curvature | Both coupled together |

### Evolution Dynamics

- **Short-time**: Deficit angles and diffusion curvature are independent
- **Medium-time**: Walkers begin adapting to anisotropic diffusion
- **Long-time (equilibrium)**: Deficit angles reflect the curved metric
- **Correction**: Provides explicit coupling at all timescales

## Validation Strategy

To verify the corrections work:

1. **Flat fitness** (`V = 0`): All corrections should be ≈ 0
2. **Quadratic bowl** (`V = ½x^T A x`): Compare to analytical `R = tr(A)`
3. **High curvature regions**: Corrections should be significant where `||H||` is large
4. **Equilibrium runs**: Corrected angles should stabilize to consistent values

## Future Enhancements

### For Production Use

1. **True Hessian evaluation**: Replace density proxy with actual `∇²V_fit`
2. **Adaptive correction**: Choose mode based on local anisotropy
3. **GPU acceleration**: Move correction computation to GPU for large N
4. **Validation metrics**: Add convergence checks for correction accuracy

### For Mathematical Rigor

1. **Error bounds**: Prove `||R^corrected - R^true|| = O(ε²)`
2. **Convergence proof**: Show correction → 0 as walkers equilibrate
3. **Stability analysis**: Characterize numerical stability of finite differences

## References

- `old_docs/source/14_scutoid_geometry_framework.md` §5.4.1: Deficit angle convergence
- `old_docs/source/14_scutoid_geometry_framework.md` §5.4: Emergent metric tensor
- `src/fragile/core/scutoids.py`: Implementation
- `src/fragile/core/kinetic_operator.py`: Anisotropic diffusion tensor

## Summary

The metric correction feature:
- ✅ **Bridges** flat-space deficit angles and emergent Riemannian geometry
- ✅ **Cheap**: O(N) diagonal or O(N·k) full, much faster than Riemannian Voronoi
- ✅ **Physically meaningful**: Couples intrinsic and extrinsic curvature
- ✅ **Flexible**: Three modes for different accuracy/cost trade-offs
- ✅ **Documented**: Clear mathematical justification and usage examples

This enables meaningful curvature analysis of Fragile Gas evolution under anisotropic diffusion.
