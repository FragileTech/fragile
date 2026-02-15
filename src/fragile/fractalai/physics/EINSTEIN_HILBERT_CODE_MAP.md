# Einstein-Hilbert Action Reward: Code Map

Traces every file and function involved in computing the Einstein-Hilbert action
`S_EH = integral R * sqrt(det g) * d^d x` used as reward in the `RiemannianMix` benchmark.

---

## High-Level Formula

Per walker the reward is:

```
U_i = -(volume_weight * ricci_weight) * R_i * sqrt(det g_i)
```

Langevin dynamics minimizes `U`, so walkers maximize `R * sqrt(det g)`,
matching the Euclidean path-integral weight `exp(+S_EH)`.

---

## Geometric Quantities Pipeline

```
Walker positions x_i
    |
    v
Delaunay triangulation --> edge_index (neighbor graph)
    |
    v
Neighbor displacement covariance: Cov_i = mean_j[(x_j - x_i)(x_j - x_i)^T]
    |
    v
Emergent metric: g_i = (Cov_i + eps*I)^{-1}
    |
    +---> det(g_i) --> sqrt(det g_i)           [volume element]
    |
    +---> u_i = (1/(2d)) * ln(det g_i)        [conformal factor]
    |
    +---> geodesic dist d_g(i,j) = sqrt(dx^T * g_edge * dx)
    |         --> edge weights w_ij = (1/d_g) / sum_k(1/d_g)
    |
    v
Graph Laplacian: (Delta u)_i = sum_j w_ij * (u_j - u_i)
    |
    v
Ricci scalar proxy: R_i = -2(d-1) * (Delta u)_i
    |
    v
Reward: U_i = -(volume_weight * ricci_weight) * R_i * sqrt(det g_i)
```

---

## Files & Functions Involved

### 1. `core/benchmarks.py` — `RiemannianMix` (reward entry point)

| Symbol | Description |
|---|---|
| `RiemannianMix` (class, line ~1054) | `OptimBenchmark` subclass. Params: `volume_weight`, `ricci_weight` |
| `riemannian_mix_potential()` (line ~1087) | Callable: `U = -scale * _cache_ricci * (-_cache_volume)` = `-scale * R * sqrt(det g)` |
| `update_scutoid_cache()` (line ~1112) | Receives scutoid data dict, stores `_cache_volume = -volume_weights`, `_cache_ricci = ricci_scalar` |

### 2. `core/euclidean_gas.py` — Simulation driver

| Symbol | Description |
|---|---|
| `EuclideanGas.step()` (line ~649) | Calls `compute_delaunay_scutoid()`, scatters alive-only data to full-N tensors |
| Lines ~862-878 | Calls `potential.update_scutoid_cache()` with `riemannian_volume_weights` and `ricci_scalar` |

### 3. `scutoid/delaunai.py` — Scutoid pipeline orchestrator

| Symbol | Description |
|---|---|
| `compute_delaunay_scutoid()` (line 172) | Main entry: Delaunay graph -> metric -> volumes -> Ricci proxy |
| `_build_delaunay_edges()` (line 71) | `scipy.spatial.Delaunay` -> symmetric COO edge list |
| `_compute_det_and_volume()` (line 136) | `slogdet(g)` -> `det(g)`, `sqrt(det g)` |
| `_compute_geodesic_distances()` (line 155) | `d_g = sqrt(dx^T * g_edge * dx)` with edge-averaged metric |
| `_compute_diffusion_tensor()` (line 147) | `sigma = g^{-1/2}` via eigendecomposition |
| `DelaunayScutoidData` (dataclass, line 30) | Container: `riemannian_volume_weights`, `ricci_proxy`, `metric_tensors`, etc. |

### 4. `scutoid/hessian_estimation.py` — Emergent metric

| Symbol | Description |
|---|---|
| `compute_emergent_metric()` (line 574) | Public wrapper |
| `_compute_emergent_metric()` (line 583) | `Cov_i = mean_j[(x_j-x_i)(x_j-x_i)^T]`, `g_i = pinv(Cov_i + eps*I)` |

### 5. `scutoid/weights.py` — Edge weights & eigenvalue clamping

| Symbol | Description |
|---|---|
| `clamp_metric_eigenvalues()` (line 131) | `eigh` -> clamp eigenvalues to `[min_eig, max_eig]` -> reconstruct |
| `compute_edge_weights()` (line 335) | Dispatcher for all weight modes |
| `compute_inverse_riemannian_distance_weights()` | `w_ij = 1/d_g(i,j)`, normalized. Used by Ricci proxy |
| `_normalize_edge_weights()` (line 23) | Row-normalize via `scatter_add_` |

### 6. `scutoid/ricci.py` — Ricci scalar computation

| Symbol | Description |
|---|---|
| `_prepare_u()` (line 9) | `u_i = (1/(2d)) * ln(det g_i)` — conformal factor |
| `compute_ricci_proxy()` (line 71) | **Default path.** Conformal formula: `R = -2(d-1) * Delta(u)` where `Delta` is the weighted graph Laplacian |
| `compute_ricci_proxy_full_metric()` (line 100) | **Full path** (when `compute_full_ricci=True`). Adds gradient-norm correction: `R = -2(d-1) * e^{-2u} * [Delta(u) + (d-2)*|grad u|^2_g]` |
| `_fit_local_quadratic()` (line 14) | Weighted least-squares quadratic fit for `grad(u)` and `Hessian(u)` |

---

## The Ricci Scalar: Mathematical Detail

### Default proxy (conformal, graph-Laplacian only)

For a conformally flat metric `g = e^{2u} * delta`:

```
R = -2(d-1) * [Delta(u) + (d-2) * |grad u|^2]
```

The **default proxy drops the gradient-norm term** and uses:

```
R_i ≈ -2(d-1) * (Delta u)_i
    = -2(d-1) * sum_j w_ij * (u_j - u_i)
```

where:
- `u_i = (1/(2d)) * ln(det g_i)`
- `w_ij = (1/d_g(i,j)) / sum_k (1/d_g(i,k))` (normalized inverse geodesic distance)
- `d_g(i,j) = sqrt(dx^T * 0.5*(g_i + g_j) * dx)` (edge-averaged geodesic distance)

### Full metric proxy (optional)

When `compute_full_ricci=True`, includes the gradient-norm term via local quadratic fitting:

```
R_i = -2(d-1) * e^{-2u_i} * [(Delta u)_i + (d-2) * |grad u_i|^2_g]
```

where `grad u` is estimated by weighted least-squares fit of `u(x)` to a local quadratic model around each node.

---

## How Volume Weighting Works

1. **Metric tensor:** `g_i = [Cov_i + eps*I]^{-1}`
   - Dense clusters -> small displacements -> small Cov -> large g -> large det(g)
   - Sparse regions -> large displacements -> large Cov -> small g -> small det(g)

2. **Volume element:** `sqrt(det g_i)` via `slogdet` + `exp(0.5 * logdet)`

3. **In the reward:** `U_i = -scale * R_i * sqrt(det g_i)`, discretizing the continuum integral `S_EH = integral R * sqrt(det g) * d^d x`

---

## Main Call Chain

```
EuclideanGas.step()
  |
  +-> compute_delaunay_scutoid()                    [scutoid/delaunai.py]
  |     |
  |     +-> _build_delaunay_edges()                  (scipy Delaunay -> COO edges)
  |     |
  |     +-> compute_emergent_metric()                [scutoid/hessian_estimation.py]
  |     |     g_i = pinv(Cov_i + eps*I)
  |     |
  |     +-> clamp_metric_eigenvalues()               [scutoid/weights.py]
  |     |
  |     +-> _compute_det_and_volume()                [scutoid/delaunai.py]
  |     |     det(g_i) via slogdet
  |     |     volume_weights = sqrt(det g_i)
  |     |
  |     +-> _compute_geodesic_distances()            [scutoid/delaunai.py]
  |     |     d_g(i,j) = sqrt(dx^T * g_edge * dx)
  |     |
  |     +-> compute_edge_weights("inverse_riemannian_distance")  [scutoid/weights.py]
  |     |     w_ij = (1/d_g) / sum_k(1/d_g)
  |     |
  |     +-> compute_ricci_proxy()                    [scutoid/ricci.py]
  |           u_i = (1/(2d)) * ln(det g_i)
  |           R_i = -2(d-1) * sum_j w_ij * (u_j - u_i)
  |
  +-> Scatter to full-N tensors:
  |     scutoid_volume_full[alive_idx] = volume_weights
  |     scutoid_ricci_full[alive_idx]  = ricci_proxy
  |
  +-> RiemannianMix.update_scutoid_cache({volume, ricci})
  |     _cache_volume = -volume_weights
  |     _cache_ricci  = ricci_proxy
  |
  +-> RiemannianMix.__call__(x) = riemannian_mix_potential(x)
        U_i = -(vol_w * ricci_w) * R_i * sqrt(det g_i)
```

---

## Alternative Code Paths (analysis, not reward)

### `qft/quantum_gravity.py` — `compute_einstein_hilbert_action()`

Analysis-only function (not used as reward). Uses a different Ricci estimation:
1. **Best:** Raychaudhuri expansion `theta = (1/V) dV/dt`, then `R ≈ -theta`
2. **Fallback:** Volume distortion `R ≈ (1 - V/<V>)` (spatial-only)

### `qft/einstein_equations.py` — Full differential geometry pipeline

Verification-only. Implements the textbook chain:

```
metric derivatives (dg/dx via weighted LS)
  -> Christoffel symbols (Gamma^a_bc)
    -> Christoffel derivatives (dGamma/dx via LS)
      -> Riemann tensor (R^a_bcd)
        -> Ricci tensor (R_bd = trace of Riemann)
          -> Ricci scalar (R = g^bd * R_bd)
            -> Einstein tensor (G_bd = R_bd - 0.5*R*g_bd)
```

### `qft/voronoi_observables.py` — `compute_curvature_proxies()`

Volume distortion / Raychaudhuri expansion curvature proxies (for QFT analysis).

### `geometry/curvature.py`

Alternative curvature methods (Laplacian spectrum, FractalSet Hessian).

---

## Files to Copy to `physics/` (for this pipeline)

| Priority | Source File | Key Contents |
|---|---|---|
| 1 | `scutoid/ricci.py` | `compute_ricci_proxy()`, `compute_ricci_proxy_full_metric()` |
| 2 | `scutoid/weights.py` | `compute_inverse_riemannian_distance_weights()`, `clamp_metric_eigenvalues()`, `_normalize_edge_weights()` |
| 3 | `scutoid/hessian_estimation.py` | `compute_emergent_metric()` |
| 4 | `scutoid/delaunai.py` | `compute_delaunay_scutoid()`, `_build_delaunay_edges()`, `_compute_det_and_volume()`, `_compute_geodesic_distances()` |
| 5 | `core/benchmarks.py` | `RiemannianMix` class (subset) |
| 6 | `core/euclidean_gas.py` | Scutoid integration in `step()` (subset) |
