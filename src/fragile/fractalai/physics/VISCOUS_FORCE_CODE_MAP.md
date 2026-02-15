# Viscous Force & Riemannian Kernel Volume Weights: Code Map

This document traces every file and function involved in computing the viscous force
with Riemannian kernel volume (RKV) weights, as used in the EuclideanGas simulation
loop and the QFT dashboard.

---

## Weight Variants (WeightMode)

Defined in `scutoid/weights.py`:

| Variant | Function | Formula |
|---|---|---|
| `uniform` | `compute_uniform_weights` | `w_ij = 1/degree_i` |
| `inverse_distance` | `compute_inverse_distance_weights` | `w_ij = 1/(d_E(i,j) + eps)` |
| `inverse_volume` | `compute_inverse_volume_weights` | `w_ij = 1/(V_j + eps)` |
| `inverse_riemannian_volume` | `compute_inverse_riemannian_volume_weights` | `w_ij = 1/(V_j^R + eps)` |
| `inverse_riemannian_distance` | `compute_inverse_riemannian_distance_weights` | `w_ij = 1/(d_g(i,j) + eps)` |
| `kernel` | `compute_gaussian_kernel_weights` | `w_ij = exp(-d_E^2/(2l^2))` |
| `riemannian_kernel` | `compute_riemannian_kernel_weights` | `w_ij = exp(-d_g^2/(2l^2))` |
| `riemannian_kernel_volume` | `compute_riemannian_kernel_volume_weights` | `w_ij = exp(-d_g^2/(2l^2)) * sqrt(det g_j)` |

---

## Files & Functions Involved

### 1. `scutoid/weights.py` — Core weight computation

| Function | Description |
|---|---|
| `compute_edge_weights()` | Dispatcher: routes weight mode string to the correct `compute_*_weights()` function |
| `compute_riemannian_kernel_volume_weights()` | Main RKV: metric -> geodesic distance -> Gaussian kernel * Riemannian volume |
| `compute_riemannian_volumes()` | `V^R = V^E * sqrt(det g)` via `torch.linalg.slogdet` |
| `clamp_metric_eigenvalues()` | Eigenvalue clamping via `torch.linalg.eigh` to ensure positive-definiteness |
| `_normalize_edge_weights()` | Row-normalization via `scatter_add_` so each source's outgoing weights sum to 1 |
| `_apply_alive_mask()` | Zeros out weights for dead walkers |
| `compute_uniform_weights()` | Uniform weights |
| `compute_inverse_distance_weights()` | Inverse Euclidean distance weights |
| `compute_inverse_volume_weights()` | Inverse Euclidean volume weights |
| `compute_inverse_riemannian_volume_weights()` | Inverse Riemannian volume weights |
| `compute_inverse_riemannian_distance_weights()` | Inverse geodesic distance weights |
| `compute_gaussian_kernel_weights()` | Gaussian kernel (Euclidean distance) |
| `compute_riemannian_kernel_weights()` | Gaussian kernel (geodesic distance) |

### 2. `scutoid/hessian_estimation.py` — Emergent metric

| Function | Description |
|---|---|
| `compute_emergent_metric()` | Public wrapper |
| `_compute_emergent_metric()` | Computes `g = Cov(Delta_x)^{-1}` from neighbor displacements using `scatter_add` + `linalg.pinv` |

### 3. `scutoid/delaunai.py` — Delaunay graph + scutoid pipeline

| Symbol | Description |
|---|---|
| `DelaunayScutoidData` (dataclass) | Container: `edge_weights: dict[str, Tensor]`, `metric_tensors`, `riemannian_volume_weights`, `diffusion_tensors`, `edge_geodesic_distances` |
| `compute_delaunay_scutoid()` | Main entry: builds Delaunay graph, computes metric, calls `compute_edge_weights()` for each requested mode |
| `_build_delaunay_edges()` | Scipy Delaunay triangulation -> sparse COO edge list |
| `_compute_diffusion_tensor()` | `Sigma = g^{-1/2}` via eigendecomposition |
| `_compute_geodesic_distances()` | Edge-averaged Riemannian distances |

### 4. `core/kinetic_operator.py` — BAOAB integrator + viscous force

| Symbol | Description |
|---|---|
| `KineticOperator` (class) | BAOAB integrator. Params: `viscous_neighbor_weighting` (default `"riemannian_kernel_volume"`), `nu`, `viscous_length_scale`, `viscous_volume_weighting`, etc. |
| `KineticOperator.apply()` | Main BAOAB loop. Calls `_apply_boris_kick()` twice (B-steps) |
| `KineticOperator._apply_boris_kick()` | Calls `_compute_viscous_force()` before and after Boris rotation |
| `KineticOperator._compute_viscous_force()` | `F_visc_i = nu * sum_j w_ij * (v_j - v_i)` using sparse edge weights |
| `KineticOperator._get_viscous_weights()` | Selects precomputed edge weights (for RKV) or computes on-the-fly (for `kernel`/`uniform`/`inverse_distance`) |
| `KineticOperator._compute_volume_weights()` | Riemannian volume weights from Voronoi data (optional extra weighting) |
| `KineticOperator._compute_diffusion_tensor()` | Anisotropic diffusion tensor |
| `psi_v()` (module-level) | Velocity squashing function applied during integration |

### 5. `core/euclidean_gas.py` — Simulation driver

| Symbol | Description |
|---|---|
| `EuclideanGas` (class) | Simulation driver. Param: `neighbor_weight_modes` (list of weight modes to precompute) |
| `EuclideanGas.step()` | Calls `compute_delaunay_scutoid()`, extracts precomputed weights, passes to `KineticOperator.apply()` |
| Records `edge_weights` dict into `RunHistory` info for dashboard analysis |

### 6. `qft/voronoi_observables.py` — Volume weights for KineticOperator

| Function | Description |
|---|---|
| `compute_riemannian_volume_weights()` | `V_i^R = V_i^E * sqrt(det g_i)` where `sqrt(det g) = c2^d / det(sigma)`. Used by `KineticOperator._compute_volume_weights()` |

### 7. `qft/particle_observables.py` — Color state from viscous force

| Function | Description |
|---|---|
| `compute_color_state()` | Takes `force_viscous` tensor, computes QFT color/phase state: `tilde = F_visc * exp(i * m * v * l0 / h_eff)` |

### 8. `qft/simulation.py` — CLI config

| Symbol | Description |
|---|---|
| `OperatorConfig` (dataclass) | Configuration with `viscous_neighbor_weighting` (default `"kernel"` here, overridden to `"riemannian_kernel_volume"` by dashboard) |
| `build_kinetic_operator()` | Builds `KineticOperator` from `OperatorConfig` |

### 9. `qft/new_dashboard.py` — Dashboard (offline analysis)

| Symbol | Description |
|---|---|
| `FractalSetSettings` (class) | Contains `edge_weight_mode`, `geometry_*` params |
| `_compute_companion_geometric_weights()` | Numpy-only RKV implementation for companion-edge graph-cut analysis |
| `_sum_cross_boundary_weights()` | Uses precomputed edge weights for graph-cut |
| `_count_crossing_lineages_weighted()` | Uses volume weights for area computation |
| Algorithm analysis callbacks | Read `history.edge_weights["riemannian_kernel_volume"]` for plotting |

### 10. `core/fractal_set.py` — Older dense-matrix implementation (separate path)

| Symbol | Description |
|---|---|
| `FractalSet._compute_viscous_weights()` | Older dense-matrix Gaussian kernel viscous weights (no Riemannian geometry). Separate from the scutoid-based sparse pipeline. |

---

## Main Call Chain (Simulation Time)

```
EuclideanGas.step()
  |
  +-> compute_delaunay_scutoid()                    [scutoid/delaunai.py]
  |     |
  |     +-> _build_delaunay_edges()                  (scipy Delaunay -> COO edges)
  |     |
  |     +-> compute_emergent_metric()                [scutoid/hessian_estimation.py]
  |     |     +-> _compute_emergent_metric()          (Cov(Delta_x)^{-1} via pinv)
  |     |
  |     +-> clamp_metric_eigenvalues()               [scutoid/weights.py]
  |     |
  |     +-> _compute_geodesic_distances()            [scutoid/delaunai.py]
  |     |
  |     +-> compute_edge_weights("riemannian_kernel_volume")  [scutoid/weights.py]
  |           |
  |           +-> compute_riemannian_kernel_volume_weights()
  |                 |
  |                 +-> clamp_metric_eigenvalues()
  |                 +-> g_edge = 0.5 * (g_src + g_dst)
  |                 +-> d_g^2 = einsum("ei,eij,ej->e", delta, g_edge, delta)
  |                 +-> kernel = exp(-d_g^2 / (2*l^2))
  |                 +-> compute_riemannian_volumes()  (V^R = V^E * sqrt(det g))
  |                 +-> raw_w = kernel * V_dst^R
  |                 +-> _apply_alive_mask()
  |                 +-> _normalize_edge_weights()
  |
  +-> edge_weights = scutoid_data.edge_weights["riemannian_kernel_volume"]
  |
  +-> KineticOperator.apply(edge_weights=...)        [core/kinetic_operator.py]
        |
        +-> _apply_boris_kick()                      (1st B-step)
        |     |
        |     +-> _compute_viscous_force()
        |           |
        |           +-> _get_viscous_weights()        (returns precomputed RKV weights)
        |           +-> v_diff = v[j] - v[i]
        |           +-> force.index_add_(0, src, w * v_diff)
        |           +-> return nu * force
        |
        +-> O-step (Ornstein-Uhlenbeck noise)
        |
        +-> _apply_boris_kick()                      (2nd B-step, same structure)
```

---

## Design Notes

1. **Precomputed vs on-the-fly**: Modes requiring metric tensors (`riemannian_*` and `inverse_riemannian_*`) MUST be precomputed in `compute_delaunay_scutoid()`. Only `kernel`, `uniform`, and `inverse_distance` support on-the-fly computation in `_get_viscous_weights()`.

2. **Double-counting warning**: `KineticOperator.viscous_volume_weighting` multiplies weights by `sqrt(det g_j)`. This MUST be disabled when using `riemannian_kernel_volume` to avoid double-counting the volume element.

3. **Two parallel implementations**: The scutoid pipeline uses torch + sparse COO edges; the dashboard's `_compute_companion_geometric_weights()` reimplements RKV in numpy for offline companion-graph analysis.

4. **Default mismatch**: `KineticOperator` defaults to `"riemannian_kernel_volume"`, `simulation.py`'s `OperatorConfig` defaults to `"kernel"`, and the dashboard overrides to `"riemannian_kernel_volume"`.

---

## Files to Copy to `physics/`

Priority order for the refactoring:

| Priority | Source File | Key Contents |
|---|---|---|
| 1 | `scutoid/weights.py` | All weight computation functions, `WeightMode` type |
| 2 | `scutoid/hessian_estimation.py` | `compute_emergent_metric()` |
| 3 | `scutoid/delaunai.py` | `DelaunayScutoidData`, `compute_delaunay_scutoid()`, edge building |
| 4 | `core/kinetic_operator.py` | `KineticOperator` class (BAOAB + viscous force) |
| 5 | `core/euclidean_gas.py` | `EuclideanGas` simulation driver |
| 6 | `qft/voronoi_observables.py` | `compute_riemannian_volume_weights()` (subset) |
| 7 | `qft/particle_observables.py` | `compute_color_state()` (subset) |
