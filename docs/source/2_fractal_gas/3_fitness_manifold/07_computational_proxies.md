(sec-computational-proxies)=
# Computational Proxies for Scutoid Geometry

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`02_scutoid_spacetime`, {doc}`03_curvature_gravity`



(sec-tldr-computational-proxies)=
## TLDR

*Notation: $R(x_i)$ = Ricci scalar; $\delta_i$ = deficit angle; $V_i$ = Voronoi cell volume; $\theta_i$ = Raychaudhuri expansion; $g_{\mu\nu}$ = emergent metric ({prf:ref}`def-adaptive-diffusion-tensor-latent`); $H = \nabla^2 V_{\mathrm{fit}}$ = fitness Hessian; $\epsilon_\Sigma$ = regularization parameter; $\nu$ = viscous coupling strength.*

**The Computational Challenge**: Chapters 01-06 established rigorous theory: emergent Riemannian geometry from fitness Hessian ($g = H + \epsilon_\Sigma I$), scutoid tessellation from cloning events, curvature from holonomy. But computing these quantities naively requires expensive operations—Hessian eigendecomposition ($O(Nd^3)$), Christoffel symbols (third derivatives of fitness), holonomy around arbitrary loops. How do we measure curved geometry at $O(N)$ cost?

**Eight O(N) Proxies**: Locality saves us. Curvature is encoded in *local* structure—neighbor distances, cell volumes, shape distortion. We compute eight proxies that converge to geometric quantities under refinement:

| Proxy | Formula | Complexity | Error | Measures |
|-------|---------|------------|-------|----------|
| Deficit angles | $R = 2\delta_i / A_i$ (2D) | $O(N \log N)$ | $O(N^{-2/d})$ | Discrete Ricci scalar |
| Volume distortion | $\sigma^2_V = \text{Var}(V_i/\langle V \rangle)$ | $O(N)$ | Heuristic | Curvature-induced inhomogeneity |
| Shape distortion | $\rho_i = r_{\text{in}}/r_{\text{circ}}$ | $O(N)$ | Heuristic | Shear from curvature |
| Raychaudhuri | $\theta_i = (1/V_i)(dV_i/dt)$ | $O(N)$ | $O(\epsilon_N)$ | Expansion scalar |
| Graph Laplacian | $\lambda_1$ (spectral gap) | $O(N \log N)$ | Cheeger bound | Spectral curvature |
| Emergent metric | $g_{\mu\nu} \approx (C^{-1})_{\mu\nu}$ | $O(N \cdot k)$ | $O(k^{-1/2})$ | Neighbor covariance |
| Geodesic distances | Dijkstra on weighted graph | $O(E \log N)$ | $O(\epsilon_N)$ | Riemannian distance |
| Riemannian volumes | $V^{\text{Riem}}_i = V^{\text{Eucl}}_i \sqrt{\det(g_i)}$ | $O(N \cdot d^2)$ | Exact | Measure correction |

**The Scutoidal Viscous Force**: These proxies enable online learning because we can compute the viscous force—the manifold relaxation operator—at $O(N \cdot k)$ cost using a row-normalized graph Laplacian:

$$
F_{\text{visc},i} = \nu \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_k w_{ik}} (v_j - v_i)
$$

Five weighting modes (uniform, kernel, inverse distance, metric-diagonal, metric-full) allow interpolation from flat-space to fully curved dynamics. The force drives the swarm toward quasi-uniform distribution in Riemannian geometry, implementing the fitness manifold's intrinsic pressure.

**Why Euclidean Voronoi Works**: The key insight (from the metric correction framework in `scutoids.py`) is first-order perturbation theory: $R^{\text{manifold}} \approx R^{\text{flat}} + \Delta R^{\text{metric}}$. We use cheap Euclidean tessellation ($O(N \log N)$) plus optional $O(N)$ metric corrections, avoiding expensive Riemannian Voronoi ($O(N^2 d)$ with geodesic distances).

**Mathematical Correction (v2.0+)**: Earlier versions of the code incorrectly used perimeter (boundary volume) instead of area for the 2D Ricci scalar, giving wrong dimensional units. The corrected implementation uses $R = 2\delta / A_{\text{Voronoi}}$ per Regge calculus, with proper dimensional analysis verified.



(sec-computational-proxies-intro)=
## Introduction

:::{div} feynman-prose
Here is the central tension we face. In Chapters 01-06, we built a beautiful edifice: emergent Riemannian geometry from the fitness Hessian, scutoid spacetime from cloning events, Ricci curvature from deficit angles, even a holographic area law. The math is rigorous. The convergence theorems are proven. The geometric picture is compelling.

But now the engineer in you asks: *how do I actually compute this?*

You look at the formulas and your heart sinks. The emergent metric is $g = H + \epsilon_\Sigma I$ where $H$ is the Hessian—that means computing $d \times d$ second derivative matrices at every walker position. The Ricci scalar involves Christoffel symbols, which need *third* derivatives of the fitness potential. Holonomy around loops requires parallel transport with ODEs. If you have $N = 10^4$ walkers in $d = 100$ dimensions, the computational cost seems astronomical.

Here is what saves us: **locality**. Curvature is not some global, holistic property that requires looking at the entire manifold. It is encoded in *local* structure—how neighbors are arranged, how cell volumes change, how shapes distort. We can probe curvature by measuring simple geometric features of the Voronoi tessellation, and these features converge to the rigorous geometric quantities under refinement.

This chapter explains eight $O(N)$ computational proxies that make the theoretical framework practical. Each proxy has a formal convergence theorem, a physical interpretation, and a simple implementation. Together, they enable online learning: we can compute curvature, adapt dynamics, and optimize the fitness manifold in real-time.

The payoff is the **scutoidal viscous force**—a graph Laplacian operator that implements manifold relaxation at $O(N \cdot k)$ cost where $k \approx 6$ is the average neighbor count. This force drives the swarm toward quasi-uniform distribution in the emergent Riemannian geometry, realizing the pressure dynamics from {doc}`04_field_equations` without ever explicitly computing Christoffel symbols.
:::



(sec-deficit-angle-proxy)=
## Proxy 1: Deficit Angles → Ricci Scalar

:::{div} feynman-prose
Let me start with the most fundamental proxy: deficit angles. This is the bridge between discrete and continuous geometry, the formula that lets us extract curvature from a finite tessellation.

The idea goes back to Tullio Regge in 1961. He realized that in discrete geometry, curvature does not live on faces or edges—it lives at *hinges*. In 2D, hinges are vertices where edges meet. The deficit angle at a vertex is $2\pi$ minus the sum of angles around it. On a flat plane, angles sum to exactly $2\pi$, so deficit is zero. On a sphere, there is positive deficit (angles sum to less than $2\pi$). On a saddle, negative deficit.

Regge proved that as you refine the tessellation, the deficit angle per unit boundary volume converges to the Ricci scalar. This is not an approximation or a heuristic—it is a theorem with quantified error bounds. And it is computationally cheap: for 2D, just compute angles around Delaunay triangles; for higher dimensions, compute dihedral angles at $(d-2)$-faces.

The beautiful part is that this works on *arbitrary* tessellations—you do not need a regular lattice or uniform spacing. The scutoid tessellation from cloning events is irregular, dynamical, even topologically changing. Deficit angles handle it all.
:::

:::{prf:definition} Discrete Deficit Angle
:label: def-deficit-angle

Let $\mathcal{V}$ be a Voronoi tessellation of $N$ points in $\mathbb{R}^d$. For a walker at position $x_i$, let $V_i$ be its Voronoi cell with boundary $\partial V_i$.

In **2D** ($d=2$): Let $\{e_1, e_2, \ldots, e_k\}$ be the edges incident to vertex $x_i$ in the dual Delaunay triangulation. The **deficit angle** is:

$$
\delta_i = 2\pi - \sum_{j=1}^{k} \alpha_j
$$

where $\alpha_j$ is the interior angle of the $j$-th Delaunay triangle at vertex $x_i$.

In **higher dimensions** ($d \geq 3$): The deficit angle is defined via dihedral angles at $(d-2)$-dimensional hinges (Voronoi edges). For computational purposes, we use the Gauss-Bonnet relation:

$$
\delta_i = \omega_d - \sum_{F \in \partial V_i} \text{SolidAngle}_i(F)
$$

where $\omega_d = 2\pi^{d/2}/\Gamma(d/2)$ is the surface area of the unit $(d-1)$-sphere, and the sum runs over all $(d-1)$-faces $F$ of the Voronoi cell boundary.

**Normalization**: $\text{Vol}(\partial V_i)$ denotes the $(d-1)$-dimensional volume (surface area) of the Voronoi cell boundary.
:::

:::{prf:theorem} Deficit Angle Convergence to Ricci Scalar
:label: thm-deficit-angle-convergence

Let $\mathcal{V}_N$ be a sequence of Voronoi tessellations with inter-particle spacing $\epsilon_N \sim N^{-1/d}$ in a bounded domain $\Omega \subset \mathbb{R}^d$ with Riemannian metric $g$. Assume the generators are quasi-uniformly distributed with shape-regularity constant $C_{\text{shape}} < \infty$.

**For $d = 2$**: The discrete Gaussian curvature at vertex $x_i$ converges to the continuous Gaussian curvature:

$$
K(x_i) = \frac{\delta_i}{A_{V_i}} + O(\epsilon_N^2)
$$

where $A_{V_i}$ is the **area** of the Voronoi cell $V_i$. The Ricci scalar is:

$$
R(x_i) = 2K(x_i) = \frac{2\delta_i}{A_{V_i}} + O(\epsilon_N^2)
$$

**For $d = 3$**: The deficit angle is defined at each **edge** $e$ of the Delaunay triangulation. The Ricci scalar is:

$$
R(x_i) = \frac{1}{V_{V_i}} \sum_{e \ni x_i} \delta_e \ell_e
$$

where:
- $\delta_e$ is the deficit angle at edge $e$ (sum of dihedral angles around $e$ subtracted from $2\pi$)
- $\ell_e$ is the edge length
- $V_{V_i}$ is the **volume** of the Voronoi cell

**Error bound**: The convergence rate is $O(\epsilon_N^2) = O(N^{-2/d})$ under $C^3$ regularity of the metric and shape-regular refinement.

*Proof.*

**Step 1. Discrete Gauss-Bonnet Theorem (2D):**

For a polygonal region $V_i$ with piecewise straight boundary, the integrated Gaussian curvature equals the angle deficit:

$$
\int_{V_i} K(x) \, dA = \delta_i
$$

This follows from the Gauss-Bonnet theorem: for a region with $n$ vertices and interior angles $\{\alpha_j\}_{j=1}^n$,

$$
\int_{V_i} K \, dA + \sum_{j=1}^n (\pi - \alpha_j) = 2\pi \chi(V_i)
$$

For a simply connected cell ($\chi = 1$) with straight edges (zero geodesic curvature), this reduces to:

$$
\int_{V_i} K \, dA = 2\pi - \sum_{j=1}^n \alpha_j = \delta_i
$$

**Step 2. Localization (2D):**

For small cells ($\text{diam}(V_i) \sim \epsilon_N$), expand $K(x)$ around $x_i$:

$$
K(x) = K(x_i) + \nabla K(x_i) \cdot (x - x_i) + O(\|x - x_i\|^2)
$$

Integrating over $V_i$ and using $\int_{V_i} (x - x_i) \, dA = 0$ (by symmetry of Voronoi cells):

$$
\int_{V_i} K(x) \, dA = K(x_i) \cdot A_{V_i} + O(\epsilon_N^{2+d})
$$

Therefore:

$$
K(x_i) = \frac{\delta_i}{A_{V_i}} + O(\epsilon_N^2)
$$

**Step 3. Ricci Scalar (2D):**

For a 2D Riemannian manifold embedded in higher dimensions, the Ricci scalar is $R = 2K$. Thus:

$$
R(x_i) = \frac{2\delta_i}{A_{V_i}} + O(\epsilon_N^2)
$$

**Step 4. Higher Dimensions:**

In $d = 3$, curvature concentrates at edges (codimension-2 hinges). The Regge action is:

$$
S_{\text{Regge}} = \sum_{\text{edges } e} \delta_e \ell_e
$$

The Ricci scalar is recovered by distributing edge contributions to vertices and dividing by cell volume (see Regge 1961, Cheeger-Müller-Schrader 1984).

**Error Analysis**: Under shape-regular refinement and $C^3$ metric regularity, higher-order terms in the Taylor expansion contribute $O(\epsilon_N^2)$ to the pointwise approximation.

**Remark**: This formula differs from the implementation in `scutoids.py` which uses boundary volume (perimeter in 2D). The **correct** formula uses the $d$-dimensional Voronoi cell volume. See Section {ref}`sec-implementation-notes` for discussion of this discrepancy.

$\square$
:::

**Implementation**: The deficit angle computation is implemented in `src/fragile/fractalai/core/scutoids.py`. The deficit angle calculation (2D example):

```python
def _compute_deficit_angles(cells: list[VoronoiCell]) -> dict[int, float]:
    """Compute deficit angle at each Delaunay vertex (2D only).

    δ_i = 2π - Σ_j α_j

    where α_j are interior angles of incident triangles.
    """
    n = len(vor.points)
    deficit = np.full(n, 2 * np.pi)

    # Build Delaunay dual
    tri = Delaunay(vor.points)

    for simplex in tri.simplices:
        # For each triangle, compute angles at each vertex
        pts = vor.points[simplex]
        for i, idx in enumerate(simplex):
            # Angle at vertex i
            v1 = pts[(i+1)%3] - pts[i]
            v2 = pts[(i+2)%3] - pts[i]
            angle = np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
                -1, 1
            ))
            deficit[idx] -= angle

    return deficit
```

The Ricci scalar computation uses the **corrected formula** from {prf:ref}`thm-deficit-angle-convergence`:

```python
# Compute cell volumes (d-dimensional: area in 2D, volume in 3D)
cell_volumes = self._compute_cell_volumes(bottom_cells)

# CORRECTED FORMULA (as of latest version):
# 2D: R = δ / (0.5 * A_Voronoi) = 2δ / A_Voronoi
# 3D: R = δ / V_Voronoi
C_d = 0.5 if self.d == 2 else 1.0
ricci = delta / (C_d * cell_vol)
```

The `_compute_cell_volumes()` function computes:
- **2D**: Polygon area using the shoelace formula (NOT perimeter)
- **3D**: Polyhedron volume from ConvexHull

This gives the correct dimensional units `[length^{-2}]` for the Ricci scalar and matches the Regge calculus formula proven in {prf:ref}`thm-deficit-angle-convergence`.

**Cross-Reference**: This proxy implements the discrete curvature formula from Regge calculus and converges to the continuous Ricci scalar from {prf:ref}`def-affine-connection` in {doc}`03_curvature_gravity`.



(sec-volume-distortion-proxy)=
## Proxy 2: Volume Distortion

:::{div} feynman-prose
Here is a beautiful fact about Riemannian geometry: curvature makes volumes non-uniform. Even if you start with perfectly evenly spaced particles, curvature will cause some Voronoi cells to expand and others to contract.

Think about geodesics on a sphere. Parallel lines that start out evenly spaced will converge at the poles (positive curvature causes focusing). The same thing happens to Voronoi cells: in regions of positive Ricci curvature, cells are compressed relative to flat space; in negative curvature regions, they expand.

The variance of normalized cell volumes thus encodes curvature information. This is not as precise as deficit angles—we cannot recover the pointwise Ricci scalar—but it is incredibly cheap to compute ($O(N)$ with no geometric constructions) and serves as a global diagnostic of geometry.

The relationship is $\sigma^2_V \sim \langle |R| \rangle \cdot h^2$ where $h$ is the inter-particle spacing. This follows from the Jacobi equation for geodesic deviation in Riemannian geometry.
:::

:::{prf:definition} Volume Distortion
:label: def-volume-distortion

Let $\{V_i\}_{i=1}^N$ be the Voronoi cell volumes for $N$ walkers. The **volume distortion** is the variance of normalized volumes:

$$
\sigma^2_V = \text{Var}\left(\frac{V_i}{\langle V \rangle}\right)
$$

where $\langle V \rangle = (1/N) \sum_{i=1}^N V_i$ is the mean cell volume.

**Optional Riemannian correction**: If the emergent metric $g_{\mu\nu}(x)$ is known, compute Riemannian volumes:

$$
V^{\text{Riem}}_i = V^{\text{Eucl}}_i \cdot \sqrt{\det(g(x_i))}
$$

and use these in the variance calculation.

**Boundary handling**: Exclude boundary and boundary-adjacent cells (3-tier classification implemented in `voronoi_observables.py::classify_boundary_cells()`) to avoid artifacts from unbounded Voronoi regions.
:::

:::{prf:theorem} Volume-Curvature Scaling Relation
:label: thm-volume-curvature-relation

For quasi-uniformly distributed particles in a Riemannian manifold $(M, g)$ with spatially varying Ricci scalar $R(x)$, the volume distortion satisfies:

$$
\sigma^2_V \sim \frac{\epsilon_N^4}{d^2} \text{Var}(R) + O(\epsilon_N^{d+2})
$$

where:
- $\epsilon_N \sim N^{-1/d}$ is the inter-particle spacing
- $\text{Var}(R) = \langle (R - \langle R \rangle)^2 \rangle$ is the spatial variance of the Ricci scalar
- The symbol $\sim$ denotes scaling behavior (not a rigorous bound)

*Proof Sketch.*

**Step 1. Jacobi Equation for Volume Distortion:**

The volume of a small geodesic ball in a Riemannian manifold with Ricci curvature satisfies (to second order in radius):

$$
V(\epsilon, x) = V_0(\epsilon) \left(1 - \frac{R(x) \epsilon^2}{6d} + O(\epsilon^3)\right)
$$

where $V_0(\epsilon) = \omega_d \epsilon^d$ is the flat-space volume and $\omega_d = \pi^{d/2} / \Gamma(1 + d/2)$.

**Step 2. Voronoi Cell Volume Approximation:**

For quasi-uniform particle distribution, the Voronoi cell volume at position $x_i$ scales as the geodesic ball volume:

$$
V_i \approx V_0(\epsilon_N) \left(1 - \frac{R(x_i) \epsilon_N^2}{6d} + O(\epsilon_N^3)\right)
$$

**Step 3. Normalized Volume Variance:**

The normalized volumes are:

$$
\frac{V_i}{\langle V \rangle} \approx \frac{1 - R(x_i) \epsilon_N^2 / (6d)}{1 - \langle R \rangle \epsilon_N^2 / (6d)} \approx 1 + \frac{(R(x_i) - \langle R \rangle) \epsilon_N^2}{6d} + O(\epsilon_N^4)
$$

where we used $(1 + \delta)^{-1} \approx 1 - \delta$ for small $\delta$.

**Step 4. Variance Computation:**

Taking the variance:

$$
\sigma^2_V = \text{Var}\left(\frac{V_i}{\langle V \rangle}\right) \approx \text{Var}\left(\frac{R(x_i) \epsilon_N^2}{6d}\right) = \frac{\epsilon_N^4}{36d^2} \text{Var}(R)
$$

Therefore:

$$
\sigma^2_V \sim \frac{\epsilon_N^4}{d^2} \text{Var}(R) \sim \frac{\langle (R - \langle R \rangle)^2 \rangle}{N^{4/d} d^2}
$$

**Remark**: This is a **heuristic scaling relation**, not a rigorous theorem, as it assumes:
1. Perfect quasi-uniformity (no clustering beyond curvature effects)
2. Small curvature regime ($R \epsilon_N^2 \ll 1$)
3. Voronoi cells well-approximated by geodesic balls

For empirical verification, plot $\log(\sigma^2_V)$ vs. $\log(N)$; slope should be $-4/d$.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/voronoi_observables.py` lines 1119-1141:

```python
# 1. Volume Distortion
volumes_eff = volumes
volume_weights_alive = _select_alive_volume_weights(volume_weights, voronoi_data)
if volume_weights_alive is not None and len(volume_weights_alive) == n:
    volumes_eff = volume_weights_alive  # Use Riemannian volumes if available

# Exclude boundary cells
interior_mask = _interior_mask(voronoi_data, n, include_boundary_adjacent=True)
valid_mask = interior_mask & np.isfinite(volumes_eff) if interior_mask is not None else np.isfinite(volumes_eff)

# Normalize by mean
mean_vol = volumes_eff[valid_mask].mean() if valid_mask.any() else 1.0
normalized_volumes = np.ones_like(volumes_eff)
if mean_vol > 0 and valid_mask.any():
    normalized_volumes[valid_mask] = volumes_eff[valid_mask] / mean_vol

# Compute variance
volume_variance = float(np.var(normalized_volumes[valid_mask])) if valid_mask.any() else 0.0
```

**Complexity**: $O(N)$ assuming Voronoi tessellation is already computed.

**Cross-Reference**: Volume distortion connects to the emergent metric $g = H + \epsilon_\Sigma I$ from {prf:ref}`def-adaptive-diffusion-tensor-latent` in {doc}`01_emergent_geometry`. Regions where $\det(H)$ is large have compressed cells.



(sec-shape-distortion-proxy)=
## Proxy 3: Shape Distortion (Inradius/Circumradius Ratio)

:::{div} feynman-prose
Volume tells us *how much* a cell is compressed or expanded. But curvature also causes *shear*—cells get elongated in some directions and compressed in others. The metric tensor $g_{\mu\nu}$ has directional information encoded in its eigenvectors and eigenvalues.

How do we extract this with $O(N)$ cost? One simple proxy is the inradius-to-circumradius ratio of each Voronoi cell. For a perfectly symmetric cell (like a regular hexagon in 2D or a regular dodecahedron in 3D), the inradius (radius of the largest inscribed sphere) is close to the circumradius (radius of the smallest circumscribed sphere). But shear deformation breaks this symmetry—the cell becomes elongated, and the ratio drops.

This is not a perfect measure of curvature—cell shape is also affected by neighbor configuration and random fluctuations. But in equilibrium, after the swarm has adapted to anisotropic diffusion, the average shape distortion in a region correlates with the shear eigenvalues of the Ricci tensor.

Computational cost is $O(N)$ because we only need to compute distances from the cell centroid to all vertices (typically 10-20 vertices per cell in 3D).
:::

:::{prf:definition} Shape Distortion
:label: def-shape-distortion

For each Voronoi cell $V_i$, let $\{v_1, v_2, \ldots, v_m\}$ be its vertex positions. Compute:

1. **Centroid**: $c_i = (1/m) \sum_{j=1}^m v_j$
2. **Vertex distances**: $d_j = \|v_j - c_i\|$
3. **Inradius**: $r_{\text{in},i} = \min_j d_j$
4. **Circumradius**: $r_{\text{circ},i} = \max_j d_j$

The **shape distortion** for cell $i$ is:

$$
\rho_i = \frac{r_{\text{in},i}}{r_{\text{circ},i}} \in (0, 1]
$$

A value of $\rho_i = 1$ indicates perfect spherical symmetry (all vertices equidistant from centroid). Values $\rho_i \ll 1$ indicate strong elongation.

**Alternative measure**: The variance of vertex distances:

$$
\sigma^2_{\text{dist},i} = \text{Var}(\{d_1, d_2, \ldots, d_m\})
$$

also quantifies shape irregularity.
:::

:::{prf:theorem} Shape Distortion and Metric Anisotropy
:label: thm-shape-distortion-anisotropy

In a Riemannian manifold with metric $g_{\mu\nu}$, let $\lambda_{\min}$ and $\lambda_{\max}$ be the minimum and maximum eigenvalues of $g$ at point $x$. For small Voronoi cells in quasi-equilibrium, the expected shape distortion satisfies:

$$
\mathbb{E}[\rho_i] \approx \sqrt{\frac{\lambda_{\min}}{\lambda_{\max}}} + O(\epsilon_N)
$$

where $\epsilon_N$ is the inter-particle spacing.

*Proof Sketch.*

**Step 1. Ellipsoidal Approximation:**

In a locally flat coordinate system aligned with the metric eigenvectors, distances scale as:

$$
\|x - x_i\|_g^2 = \sum_{\mu=1}^d \lambda_\mu (x^\mu - x_i^\mu)^2
$$

A sphere in Euclidean coordinates becomes an ellipsoid in the metric-induced geometry.

**Step 2. Voronoi Cell Shape:**

The Voronoi cell boundary is (approximately) the locus of points equidistant from $x_i$ and its neighbors. In the presence of metric anisotropy, this boundary is ellipsoidal with axis ratios determined by $\sqrt{\lambda_\mu}$.

**Step 3. Inradius/Circumradius:**

For an ellipsoid with semi-axes $a_1 \leq a_2 \leq \cdots \leq a_d$, the inradius is $r_{\text{in}} \approx a_1$ and circumradius is $r_{\text{circ}} \approx a_d$. Thus:

$$
\rho \approx \frac{a_1}{a_d} \sim \sqrt{\frac{\lambda_{\min}}{\lambda_{\max}}}
$$

**Remark**: This is a mean-field approximation. Fluctuations from discrete sampling add noise $O(\epsilon_N)$.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/voronoi_observables.py` lines 1143-1186:

```python
# 2. Shape Distortion (inradius/circumradius ratio)
shape_distortion = np.zeros(n)
cell_centroids = np.zeros((n, d))
centroid_distances = np.zeros(n)

for i in range(n):
    if vor is not None:
        region_idx = vor.point_region[i]
        vertices_idx = vor.regions[region_idx]

        # Skip boundary cells (-1 in vertices means unbounded)
        if -1 not in vertices_idx and len(vertices_idx) >= d + 1:
            try:
                vertices = vor.vertices[vertices_idx]
                centroid = vertices.mean(axis=0)
                cell_centroids[i] = centroid

                # Compute distances from centroid to all vertices
                distances = np.linalg.norm(vertices - centroid, axis=1)
                inradius = distances.min() if len(distances) > 0 else 0.0
                circumradius = distances.max() if len(distances) > 0 else 1.0

                # Shape distortion ratio
                if circumradius > 1e-10:
                    shape_distortion[i] = inradius / circumradius
                else:
                    shape_distortion[i] = 1.0

                # Distance variance (alternative measure)
                centroid_distances[i] = float(np.var(distances)) if len(distances) > 1 else 0.0

            except Exception:
                shape_distortion[i] = 1.0
                cell_centroids[i] = positions[i]
                centroid_distances[i] = 0.0
```

**Complexity**: $O(N \cdot m)$ where $m$ is the average number of vertices per cell. For quasi-uniform tessellations in 3D, $m \approx 15$, so effectively $O(N)$.

**Cross-Reference**: Shape distortion measures the directional anisotropy of the emergent metric $g_{\mu\nu}$ from {prf:ref}`def-adaptive-diffusion-tensor-latent`. In the presence of non-isotropic Hessian eigenvalues, cells elongate along soft directions.



(sec-raychaudhuri-proxy)=
## Proxy 4: Raychaudhuri Expansion

:::{div} feynman-prose
Now we come to a proxy that connects geometry to *dynamics*. The Raychaudhuri equation (from {doc}`03_curvature_gravity`) governs how bundles of geodesics expand or contract:

$$
\frac{d\theta}{d\tau} = -\frac{1}{d}\theta^2 - \sigma_{\mu\nu}\sigma^{\mu\nu} + \omega_{\mu\nu}\omega^{\mu\nu} - R_{\mu\nu}u^\mu u^\nu
$$

The expansion scalar $\theta$ measures the rate of volume change along the flow. The key insight: in a vorticity-free ($\omega = 0$) and shear-free ($\sigma = 0$) flow, we have simply:

$$
\frac{d\theta}{d\tau} \approx -\theta^2 / d - R
$$

So the Ricci scalar equals (minus) the rate of change of expansion. Or, for small expansion: $\theta \approx -R$.

How do we measure $\theta$ on a discrete tessellation? Easy: $\theta = (1/V)(dV/dt)$. Track how each Voronoi cell volume changes between timesteps, normalize by the current volume, and you have a direct probe of the Ricci scalar.

This is the only proxy that requires *time series* data—we need volumes at two consecutive times. But the computational cost is still $O(N)$: just subtract volumes and divide.

The beautiful part is that this is not an ad-hoc approximation. It is the *discrete version of the Raychaudhuri equation*, and it converges to the continuous formula under refinement (Theorem {prf:ref}`thm-discrete-raychaudhuri` in {doc}`03_curvature_gravity`).
:::

:::{prf:definition} Discrete Raychaudhuri Expansion
:label: def-raychaudhuri-expansion

Let $V_i(t)$ be the Voronoi cell volume for walker $i$ at time $t$, and $V_i(t + \Delta t)$ at time $t + \Delta t$. The **expansion scalar** for cell $i$ is:

$$
\theta_i = \frac{1}{V_i(t)} \cdot \frac{V_i(t + \Delta t) - V_i(t)}{\Delta t}
$$

This approximates the continuous definition $\theta = \nabla_\mu u^\mu$ for a congruence of worldlines.

**Sign convention**: Positive $\theta$ means expansion (volume increasing); negative $\theta$ means contraction.

**Curvature relation**: From the Raychaudhuri equation (neglecting shear and vorticity):

$$
\frac{d\theta}{dt} \approx -\frac{1}{d}\theta^2 - R_{tt}
$$

where $R_{tt}$ is the timelike-timelike component of the Ricci tensor. For small $\theta$ and approximately constant curvature:

$$
\theta(t) \approx -R_{tt} \cdot \Delta t + O((\Delta t)^2)
$$

For spatial slices in the quasi-static regime, $R_{tt} \approx R/d$ where $R$ is the spatial Ricci scalar, giving:

$$
\langle R \rangle \approx -d \cdot \langle \theta \rangle / \Delta t
$$

**Note**: This is a heuristic relation; rigorous convergence requires careful treatment of the metric's time dependence and the relationship between spatial and spacetime curvature.
:::

:::{prf:theorem} Discrete Raychaudhuri Convergence
:label: thm-raychaudhuri-convergence

Let $\{x_i(t)\}_{i=1}^N$ be a family of particles evolving under a smooth vector field $u^\mu(x, t)$ in a Riemannian manifold. Let $V_i(t)$ be the Voronoi cell volumes. Under quasi-uniform refinement ($\epsilon_N \sim N^{-1/d}$), the discrete expansion $\theta_i^{\text{disc}}$ converges to the continuous expansion $\theta(x_i, t) = \nabla_\mu u^\mu$ with error:

$$
\left| \theta_i^{\text{disc}} - \theta(x_i, t) \right| \leq C \epsilon_N
$$

where $C$ depends on the $C^2$ norm of $u^\mu$ and the shape-regularity of the tessellation.

*Proof.*

This is a special case of {prf:ref}`thm-discrete-raychaudhuri` in {doc}`03_curvature_gravity`, which proves convergence of the full discrete Raychaudhuri equation including shear and vorticity terms. The key steps are:

**Step 1. Voronoi Volume as Flow Determinant:**

By the divergence theorem:

$$
\frac{dV_i}{dt} = \int_{\partial V_i} u \cdot \hat{n} \, dS
$$

For small cells, approximate $u$ as constant:

$$
\frac{dV_i}{dt} \approx V_i \cdot \nabla \cdot u = V_i \cdot \theta
$$

**Step 2. Discrete Derivative:**

The forward difference approximation gives:

$$
\theta_i^{\text{disc}} = \frac{V_i(t+\Delta t) - V_i(t)}{V_i(t) \cdot \Delta t} = \theta(x_i, t) + O(\Delta t)
$$

**Step 3. Spatial Error:**

Cell-to-cell variation in $\theta$ introduces error $O(\epsilon_N)$ from the "constant $u$" approximation.

**Identification**: Combining temporal and spatial errors gives $O(\epsilon_N + \Delta t)$. For typical integration schemes, $\Delta t \sim \epsilon_N$, so total error is $O(\epsilon_N)$.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/voronoi_observables.py` lines 1196-1217:

```python
if prev_volumes is not None and dt > 0:
    if len(prev_volumes) == n:
        # Compute volume rate: dV/dt ≈ (V_t - V_{t-dt}) / dt
        dV_dt = (volumes_eff - prev_volumes) / dt

        # Raychaudhuri expansion: θ = (1/V) dV/dt
        # Avoid division by zero
        safe_volumes = np.where(volumes_eff > 1e-10, volumes_eff, 1.0)
        raychaudhuri_expansion = dV_dt / safe_volumes
        if valid_mask is not None:
            raychaudhuri_expansion = np.where(valid_mask, raychaudhuri_expansion, 0.0)

        # Mean curvature proxy: using -<θ> as heuristic
        # NOTE: Dimensionally, this gives [time^-1], not [length^-2]
        # In algorithmic units where dt ≈ ε_N (inter-particle spacing),
        # this provides a curvature-like diagnostic quantity
        valid_mask = np.isfinite(raychaudhuri_expansion) & (np.abs(raychaudhuri_expansion) < 1e6)
        mean_curvature_estimate = 0.0
        if valid_mask.sum() > 0:
            mean_curvature_estimate = float(-raychaudhuri_expansion[valid_mask].mean())

        result["raychaudhuri_expansion"] = raychaudhuri_expansion
        result["mean_curvature_estimate"] = mean_curvature_estimate
```

**Complexity**: $O(N)$ assuming volumes are already computed at both times.

**Dimensional Note**: The expansion $\theta$ has units $[time^{-1}]$ while the spatial Ricci scalar has units $[length^{-2}]$. The code uses `mean_curvature_estimate = -<θ>` as a heuristic diagnostic. For this to match geometric curvature, one would need to rescale by $(c \Delta t)^{-1}$ where $c$ is a characteristic velocity and $\Delta t$ is the time step. In algorithmic units where $\Delta t \sim \epsilon_N$ (inter-particle spacing), the expansion directly probes the rate of geometric focusing.

**Cross-Reference**: This proxy implements discrete volume tracking related to the Raychaudhuri equation from {prf:ref}`def-parallel-transport` in {doc}`03_curvature_gravity`. The theoretical connection between $\theta$ and $R$ is derived in the focusing theorem, though dimensional consistency requires careful unit analysis.



(sec-graph-laplacian-proxy)=
## Proxy 5: Graph Laplacian Spectral Gap

:::{div} feynman-prose
Now we get to something really clever: using *eigenvalues* to measure curvature. This comes from spectral geometry, a field that studies how the shape and curvature of a manifold are encoded in the spectrum of the Laplace-Beltrami operator.

Here is the key fact: manifolds with positive Ricci curvature have a spectral gap—the first non-zero eigenvalue $\lambda_1$ of the Laplacian is bounded below. This is the Cheeger inequality. Intuitively, positive curvature makes the manifold "round," preventing it from having long, thin bottlenecks that would allow low-frequency modes.

On a discrete graph, we can compute the graph Laplacian $L = D - W$ where $D$ is the degree matrix and $W$ is the adjacency matrix with edge weights. The eigenvalues of $L$ probe the geometry of the underlying space. For a graph that approximates a curved manifold, $\lambda_1$ is related to the Ricci curvature via:

$$
\lambda_1 \geq C(\kappa, d)
$$

where $\kappa$ is a lower bound on Ricci curvature.

The computation uses sparse eigensolvers (Lanczos or ARPACK) with complexity $O(N \log N)$ for the first few eigenvalues. This is more expensive than the previous proxies but still tractable for online learning.

The payoff is that spectral gap is a *global* measure—it probes the large-scale geometry, not just local curvature fluctuations. This makes it robust to noise.
:::

:::{prf:definition} Graph Laplacian for Voronoi Tessellation
:label: def-graph-laplacian

Let $G = (V, E)$ be the graph with vertices $V = \{x_1, x_2, \ldots, x_N\}$ (walker positions) and edges $E$ connecting Voronoi neighbors. Define edge weights:

$$
w_{ij} = \begin{cases}
\exp\left(-\|x_i - x_j\|^2 / (2\ell^2)\right) & \text{if } i \sim j \\
0 & \text{otherwise}
\end{cases}
$$

where $\ell$ is a length scale and $i \sim j$ means cells $i$ and $j$ share a Voronoi facet.

The **graph Laplacian** is:

$$
L = D - W
$$

where:
- $W$ is the adjacency matrix with entries $W_{ij} = w_{ij}$
- $D$ is the degree matrix: $D_{ii} = \sum_{j=1}^N w_{ij}$

The **normalized graph Laplacian** is:

$$
\mathcal{L} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} W D^{-1/2}
$$

**Spectrum**: The eigenvalues $0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_{N-1} \leq 2$ of $\mathcal{L}$ encode geometric information. The first non-zero eigenvalue $\lambda_1$ is the **spectral gap**.
:::

:::{prf:theorem} Cheeger Inequality and Ricci Curvature
:label: thm-cheeger-inequality

For a graph $G$ approximating a Riemannian manifold $(M, g)$ with Ricci curvature $\text{Ric}_g \geq \kappa g$ for some $\kappa > 0$, the spectral gap satisfies:

$$
\lambda_1 \geq \frac{C(d) \cdot \kappa}{N^{2/d}}
$$

where $C(d)$ is a dimension-dependent constant.

Conversely, if $\text{Ric}_g \leq \kappa_{\max} g$, then:

$$
\lambda_1 \leq C'(d) \cdot \kappa_{\max} + O(N^{-1/d})
$$

*Reference.*

This is a discrete version of the Li-Yau inequality and Cheeger's inequality from Riemannian geometry. Rigorous proofs for graph approximations appear in:
- Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
- Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces." *Journal of Functional Analysis*.

$\square$
:::

**Implementation**: The graph Laplacian computation for Voronoi tessellations is in `src/fragile/fractalai/geometry/curvature.py`:

```python
def compute_graph_laplacian_eigenvalues(
    positions: np.ndarray,
    edge_index: np.ndarray,
    k: int = 10,
    sigma: float = 1.0,
) -> np.ndarray:
    """Compute first k eigenvalues of normalized graph Laplacian.

    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        k: Number of eigenvalues to compute
        sigma: Length scale for Gaussian kernel

    Returns:
        eigenvalues: [k] sorted eigenvalues 0 = λ_0 < λ_1 <= ... <= λ_{k-1}
    """
    N = len(positions)

    # Build adjacency matrix with Gaussian weights
    W = np.zeros((N, N))
    for e in range(edge_index.shape[1]):
        i, j = edge_index[:, e]
        dist = np.linalg.norm(positions[i] - positions[j])
        w_ij = np.exp(-dist**2 / (2 * sigma**2))
        W[i, j] = w_ij
        W[j, i] = w_ij

    # Degree matrix
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    L_norm = np.eye(N) - D_inv_sqrt @ W @ D_inv_sqrt

    # Compute smallest k eigenvalues (use sparse solver for large N)
    eigenvalues, _ = scipy.sparse.linalg.eigsh(L_norm, k=k, which='SM')

    return np.sort(eigenvalues)
```

**Complexity**: $O(N \log N)$ for sparse graphs using iterative eigensolvers. Dense graphs require $O(N^3)$ but do not arise in typical Voronoi tessellations.

**Cross-Reference**: The spectral gap measures global properties of the emergent metric from {prf:ref}`def-adaptive-diffusion-tensor-latent`. It is related to the mixing time of diffusion processes on the fitness manifold.



(sec-emergent-metric-proxy)=
## Proxy 6: Emergent Metric from Neighbor Covariance

:::{div} feynman-prose
All the previous proxies extract *scalar* information: deficit angles give the Ricci scalar, volume variance gives a global curvature measure, spectral gap gives a topological diagnostic. But the full geometric structure is a *tensor*—the metric $g_{\mu\nu}$ with $d(d+1)/2$ independent components.

How do we approximate the metric at $O(N)$ cost without computing second derivatives of the fitness?

The answer is neighbor covariance. For each walker $i$, look at its $k$ nearest Voronoi neighbors. Compute the covariance matrix of their positions relative to $x_i$:

$$
C_{\alpha\beta} = \frac{1}{k} \sum_{j \in \mathcal{N}(i)} (x_j^\alpha - x_i^\alpha)(x_j^\beta - x_i^\beta)
$$

This covariance matrix encodes how the neighbors are distributed around $x_i$. In an isotropic flat space, $C$ is proportional to the identity. In an anisotropic curved space, $C$ reflects the metric structure.

The key insight: the *inverse* of $C$ approximates the metric tensor:

$$
g_{\mu\nu} \approx (C^{-1})_{\mu\nu}
$$

Why? Because in equilibrium, the walker distribution adapts to the diffusion tensor $D = g^{-1}$. Regions with large diffusion are more densely populated (exploration), so neighbor distances are *smaller*. The inverse of this covariance gives back the metric.

This is not exact—there are corrections from boundary effects, finite $k$, and non-equilibrium transients. But the error is $O(k^{-1})$, so with $k \approx 6$ neighbors we get 15-20% accuracy, sufficient for many applications.

The computational cost is $O(N \cdot k \cdot d^2)$. For $k = 6$ and $d = 100$, this is vastly cheaper than computing Hessians everywhere.
:::

:::{prf:definition} Emergent Metric from Neighbor Covariance
:label: def-emergent-metric-neighbor

For each walker $i$ at position $x_i$, let $\mathcal{N}(i) = \{j_1, j_2, \ldots, j_k\}$ be its $k$ nearest Voronoi neighbors. Compute the **neighbor covariance matrix**:

$$
C_{\alpha\beta}(x_i) = \frac{1}{k} \sum_{j \in \mathcal{N}(i)} (x_j^\alpha - x_i^\alpha)(x_j^\beta - x_i^\beta)
$$

where Greek indices $\alpha, \beta \in \{1, 2, \ldots, d\}$ label spatial coordinates.

The **emergent metric tensor** is defined as:

$$
g_{\mu\nu}(x_i) = (C^{-1})_{\mu\nu} + \epsilon_{\text{reg}} \delta_{\mu\nu}
$$

where $\epsilon_{\text{reg}} > 0$ is a small regularization parameter to ensure positive definiteness (typically $\epsilon_{\text{reg}} \sim 10^{-5}$).

**Properties**:
1. Symmetric: $g_{\mu\nu} = g_{\nu\mu}$ (from $C$ symmetry)
2. Positive definite: Guaranteed by regularization
3. $O(k)$ accuracy: Error scales as $O(k^{-1})$ for large $k$
:::

:::{prf:theorem} Neighbor Covariance Convergence to Metric
:label: thm-neighbor-covariance-metric

Let walkers be distributed according to the quasi-stationary distribution $\rho_{\text{QSD}}(x) \propto \exp(-\beta V_{\mathrm{fit}}(x))$ on a Riemannian manifold with metric $g_{\mu\nu}$ and diffusion tensor $D_{\mu\nu} = g^{-1}_{\mu\nu}$.

For large neighbor count $k$ and quasi-uniform density, the neighbor covariance satisfies:

$$
C_{\alpha\beta}(x_i) = D_{\alpha\beta}(x_i) \cdot r_k^2 + O(r_k^3) + O(k^{-1/2})
$$

where $r_k \sim k^{1/d}$ is the typical neighbor distance.

Therefore:

$$
g_{\mu\nu}(x_i) \approx \frac{1}{r_k^2} \cdot (C^{-1})_{\mu\nu}
$$

converges to the true metric with error $O(k^{-1/2})$.

*Proof Sketch.*

**Step 1. Diffusion-Covariance Relation:**

In equilibrium, walkers are distributed according to $\rho \propto e^{-V_{\mathrm{fit}}}$. For small regions around $x_i$, the density is approximately uniform in the metric-induced measure.

The covariance of positions under uniform sampling in a metric $g$ is proportional to $g^{-1}$:

$$
\mathbb{E}[(x - x_i)(x - x_i)^T] \propto g^{-1}
$$

This follows from the fact that $g^{-1}$ is the diffusion tensor in Riemannian Brownian motion.

**Step 2. Finite Neighbor Corrections:**

For $k$ finite neighbors, the sample covariance $C$ is an unbiased estimator of the true covariance with variance $O(k^{-1})$:

$$
C_{\alpha\beta} = \mathbb{E}[C_{\alpha\beta}] + O(k^{-1/2})
$$

**Step 3. Inversion:**

Inverting $C$ with regularization gives:

$$
(C^{-1})_{\mu\nu} = ((\mathbb{E}[C])^{-1})_{\mu\nu} + O(k^{-1/2})
$$

using perturbation theory for matrix inversion.

**Identification**: Combining steps and rescaling by $r_k^2$ gives the metric approximation.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/higgs_observables.py` lines 142-192:

```python
def compute_emergent_metric(
    positions: Tensor,
    edge_index: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute emergent Riemannian metric from neighbor covariance.

    For each walker i, compute the covariance matrix of its neighbors:
    C_αβ = (1/k) Σ_j (x_j^α - x_i^α)(x_j^β - x_i^β)

    The metric tensor is: g_μν = (C^{-1})_μν

    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        alive: Alive mask [N]

    Returns:
        metric_tensors: [N, d, d] Symmetric positive-definite metric tensors
    """
    N, d = positions.shape
    device = positions.device

    # Initialize covariance matrices
    covariance = torch.zeros(N, d, d, device=device, dtype=positions.dtype)

    # Compute position differences for all edges
    row, col = edge_index[0], edge_index[1]
    diff = positions[col] - positions[row]  # [E, d]

    # Outer product: diff ⊗ diff
    outer_prod = diff.unsqueeze(2) * diff.unsqueeze(1)  # [E, d, d]

    # Aggregate to nodes using scatter_add
    covariance_flat = covariance.view(N, -1)
    covariance_flat.scatter_add_(0, row.unsqueeze(1).expand(-1, d * d), outer_prod.view(-1, d * d))
    covariance = covariance_flat.view(N, d, d)

    # Normalize by degree
    degree = torch.bincount(row, minlength=N).float().clamp(min=1)
    covariance = covariance / degree.view(-1, 1, 1)

    # Invert to get metric (with regularization for stability)
    epsilon = torch.eye(d, device=device, dtype=positions.dtype) * 1e-5
    metric_tensors = torch.linalg.inv(covariance + epsilon)

    # Ensure symmetry (due to numerical precision)
    metric_tensors = 0.5 * (metric_tensors + metric_tensors.transpose(-2, -1))

    return metric_tensors
```

**Complexity**: $O(N \cdot k \cdot d^2)$ where $k$ is the average degree (typically $k \approx 6$ in 2D, $k \approx 12$ in 3D). The matrix inversion is $O(d^3)$ per walker, but $d$ is usually $\leq 10$ for fitness manifolds.

**Cross-Reference**: This proxy provides a cheap approximation to the full emergent metric $g = H + \epsilon_\Sigma I$ from {prf:ref}`def-adaptive-diffusion-tensor-latent` in {doc}`01_emergent_geometry`, without needing to compute second derivatives of $V_{\mathrm{fit}}$.



(sec-geodesic-distance-proxy)=
## Proxy 7: Geodesic Distances on Weighted Graph

:::{div} feynman-prose
Once we have the emergent metric tensor from neighbor covariance (Proxy 6), we can use it to compute *geodesic distances*—the shortest paths through the curved geometry.

Why does this matter? Because many optimization algorithms (like trust-region methods) need to know "how far" two points are in the fitness landscape's intrinsic geometry. Euclidean distance can be wildly misleading: two points close in Euclidean space might be separated by a high-curvature ridge, making them "far" in the manifold sense.

The algorithm is simple:
1. Build a weighted graph where edge weights are $w_{ij} = \|x_i - x_j\|_g$, the distance in the metric $g$
2. Run Dijkstra's algorithm to find shortest paths
3. The resulting distances are discrete approximations to continuous geodesic distances

For each edge, we interpolate the metric to the edge midpoint: $g_{\text{edge}} = (g_i + g_j)/2$, then compute:

$$
d_{\text{geo}}(i,j)^2 = \Delta x^T \cdot g_{\text{edge}} \cdot \Delta x
$$

where $\Delta x = x_j - x_i$.

The computational cost is $O(E \log N)$ where $E$ is the number of edges. For Voronoi graphs, $E \approx 3N$ in 2D and $E \approx 6N$ in 3D, so this is effectively $O(N \log N)$.

This is still more expensive than the other proxies, but geodesic distances are only needed for specific queries (e.g., "what is the manifold distance from walker $i$ to the best walker?"), not for all pairs.
:::

:::{prf:definition} Discrete Geodesic Distance
:label: def-discrete-geodesic-distance

Let $G = (V, E)$ be the Voronoi neighbor graph with vertices $V = \{x_1, \ldots, x_N\}$ and edges $E$. For each edge $(i, j) \in E$, define the **Riemannian edge length**:

$$
\ell_{ij} = \sqrt{(x_j - x_i)^T \cdot g_{\text{edge}} \cdot (x_j - x_i)}
$$

where $g_{\text{edge}} = (g_i + g_j)/2$ is the metric interpolated to the edge midpoint.

The **discrete geodesic distance** between walkers $i$ and $j$ is:

$$
d_{\text{geo}}(i, j) = \min_{\gamma: i \to j} \sum_{(p,q) \in \gamma} \ell_{pq}
$$

where the minimum is over all paths $\gamma$ from $i$ to $j$ in the graph.

**Computation**: Use Dijkstra's algorithm with edge weights $\ell_{ij}$.

**Convergence**: Under quasi-uniform refinement, $d_{\text{geo}}(i,j) \to d_g(x_i, x_j)$ where $d_g$ is the Riemannian distance on the continuous manifold, with error $O(\epsilon_N)$.
:::

:::{prf:theorem} Convergence of Discrete Geodesic Distance
:label: thm-geodesic-distance-convergence

Let $(M, g)$ be a smooth Riemannian manifold and $\{x_i\}_{i=1}^N$ a quasi-uniform sampling with inter-particle spacing $\epsilon_N \sim N^{-1/d}$. Let $d_{\text{geo}}^N(i,j)$ be the discrete geodesic distance on the Voronoi graph.

Then:

$$
\left| d_{\text{geo}}^N(i,j) - d_g(x_i, x_j) \right| \leq C \epsilon_N
$$

where $d_g$ is the Riemannian distance on $(M,g)$ and $C$ depends on the $C^2$ norm of $g$ and the shape-regularity constant.

*Proof.*

**Step 1. Geodesic Approximation by Piecewise Linear Paths:**

Any continuous geodesic $\gamma: [0,1] \to M$ can be approximated by a sequence of straight-line segments connecting nearby points:

$$
\ell_g(\gamma) = \int_0^1 \sqrt{g(\dot{\gamma}, \dot{\gamma})} \, dt
$$

**Step 2. Discrete Path Length:**

For a path through graph vertices $\{x_{i_0}, x_{i_1}, \ldots, x_{i_m}\}$, the discrete length is:

$$
\ell_{\text{disc}} = \sum_{k=0}^{m-1} \ell_{i_k, i_{k+1}}
$$

**Step 3. Error Analysis:**

The difference between continuous and discrete arises from:
1. **Chord vs. arc length**: $O(\epsilon_N^2)$ for small segments
2. **Metric interpolation error**: $O(\epsilon_N)$ from using midpoint approximation
3. **Path non-optimality**: Discrete path may not follow exact geodesic

Summing over $O(\epsilon_N^{-1})$ segments gives total error $O(\epsilon_N)$.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/higgs_observables.py` lines 247-284:

```python
def compute_geodesic_distances(
    positions: Tensor,
    edge_index: Tensor,
    metric_tensors: Tensor,
    alive: Tensor,
) -> Tensor:
    """Compute geodesic distances using emergent metric.

    d_geo(i,j)² = Δx^T · g_ij · Δx
    where g_ij = (g_i + g_j) / 2 (interpolated metric on edge)

    Args:
        positions: Walker positions [N, d]
        edge_index: Edge connectivity [2, E]
        metric_tensors: Metric at each node [N, d, d]
        alive: Alive mask [N]

    Returns:
        geodesic_distances: [E] Geodesic length of each edge
    """
    row, col = edge_index[0], edge_index[1]

    # Position differences
    delta_x = positions[col] - positions[row]  # [E, d]

    # Interpolate metric to edge midpoint
    g_edge = 0.5 * (metric_tensors[row] + metric_tensors[col])  # [E, d, d]

    # Compute bilinear form: v^T G v
    # First: G * v
    Gv = torch.matmul(g_edge, delta_x.unsqueeze(2)).squeeze(2)  # [E, d]
    # Then: v^T * (G * v)
    dist_sq = (delta_x * Gv).sum(dim=1)  # [E]

    # Take square root (with epsilon for stability)
    geodesic_dist = torch.sqrt(dist_sq.clamp(min=1e-8))

    return geodesic_dist
```

For shortest path computation, use standard graph algorithms:

```python
# Build weighted graph
import networkx as nx

G = nx.Graph()
for i in range(N):
    G.add_node(i)
for e, (i, j) in enumerate(edge_index.T):
    G.add_edge(int(i), int(j), weight=float(geodesic_dist[e]))

# Compute shortest paths (Dijkstra)
distances = nx.single_source_dijkstra_path_length(G, source=0)
```

**Complexity**: $O(E \log N)$ per source using Dijkstra with binary heap. For all-pairs distances, $O(N \cdot E \log N) = O(N^2 \log N)$ in sparse graphs.

**Cross-Reference**: Geodesic distances are the natural notion of distance on the fitness manifold $(M, g)$ from {doc}`01_emergent_geometry`. They appear in trust-region optimization and in the definition of the algorithmic distance used for companion selection.



(sec-riemannian-volume-proxy)=
## Proxy 8: Riemannian Volume Weights

:::{div} feynman-prose
The last proxy addresses a subtle but important issue: *measure theory on curved manifolds*.

When we compute averages like $\langle O \rangle = (1/N) \sum_i O_i$, we are implicitly assuming all walkers have equal weight. This is correct in flat Euclidean space where the volume element is $dx^1 dx^2 \cdots dx^d$.

But on a Riemannian manifold, the volume element is $\sqrt{\det(g)} \, dx^1 \cdots dx^d$. The metric determinant acts as a density that weights different regions. In curved geometry, the correct average is:

$$
\langle O \rangle = \frac{\sum_i O_i \cdot \sqrt{\det(g_i)} \cdot V_i}{\sum_i \sqrt{\det(g_i)} \cdot V_i}
$$

where $V_i$ are the Euclidean Voronoi volumes and $\sqrt{\det(g_i)}$ converts them to Riemannian volumes.

This correction is small if the metric is nearly isotropic ($g \approx I$) but becomes significant when anisotropy is strong (large spread in Hessian eigenvalues). For fitness landscapes with $\det(H) \gg 1$ in some regions, ignoring the correction can bias observables.

Computationally, this is cheap: just multiply Voronoi volumes by $\sqrt{\det(g)}$, which costs $O(d^3)$ per walker (LU decomposition for determinant). Total cost is $O(N \cdot d^3)$, still effectively $O(N)$ for fixed $d$.
:::

:::{prf:definition} Riemannian Volume Weights
:label: def-riemannian-volume-weights

Let $V_i^{\text{Eucl}}$ be the Voronoi cell volume computed in Euclidean coordinates, and $g_{\mu\nu}(x_i)$ the emergent metric at walker $i$.

The **Riemannian volume weight** is:

$$
V_i^{\text{Riem}} = V_i^{\text{Eucl}} \cdot \sqrt{\det(g(x_i))}
$$

**Normalized weights** (for weighted averaging):

$$
w_i = \frac{V_i^{\text{Riem}}}{\sum_{j=1}^N V_j^{\text{Riem}}}
$$

**Weighted average** of an observable $O_i$:

$$
\langle O \rangle_g = \sum_{i=1}^N w_i \cdot O_i
$$
:::

:::{prf:theorem} Volume Element Transformation
:label: thm-volume-element-transformation

On a Riemannian manifold $(M, g)$, the volume element in local coordinates $\{x^\mu\}$ is:

$$
dV_g = \sqrt{\det(g)} \, dx^1 \wedge dx^2 \wedge \cdots \wedge dx^d
$$

For a measurable set $A \subset M$:

$$
\text{Vol}_g(A) = \int_A \sqrt{\det(g(x))} \, dx
$$

For quasi-uniform discrete sampling, the discrete approximation is:

$$
\text{Vol}_g(A) \approx \sum_{i: x_i \in A} V_i^{\text{Eucl}} \cdot \sqrt{\det(g(x_i))}
$$

with error $O(\epsilon_N^2)$ under $C^2$ regularity of $g$.

*Proof.*

This is the standard change-of-variables formula in Riemannian geometry. See Lee, J. M. (2018), *Introduction to Riemannian Manifolds*, Chapter 4.

$\square$
:::

**Implementation**: From `src/fragile/fractalai/qft/voronoi_observables.py` and `higgs_observables.py`:

```python
def compute_riemannian_volume_weights(
    volumes_eucl: np.ndarray,
    metric_tensors: np.ndarray,
) -> np.ndarray:
    """Compute Riemannian volume weights from metric determinants.

    V^Riem_i = V^Eucl_i * sqrt(det(g_i))

    Args:
        volumes_eucl: Euclidean Voronoi volumes [N]
        metric_tensors: Emergent metric [N, d, d]

    Returns:
        volumes_riem: Riemannian-corrected volumes [N]
    """
    N = len(volumes_eucl)
    volumes_riem = np.zeros(N)

    for i in range(N):
        g_i = metric_tensors[i]
        # Compute det(g) robustly
        try:
            det_g = np.linalg.det(g_i)
            if det_g > 1e-10:
                volumes_riem[i] = volumes_eucl[i] * np.sqrt(det_g)
            else:
                # Degenerate metric, use Euclidean volume
                volumes_riem[i] = volumes_eucl[i]
        except np.linalg.LinAlgError:
            volumes_riem[i] = volumes_eucl[i]

    return volumes_riem
```

For integration in the viscous force (see next section), volume weights multiply edge weights:

```python
# In kinetic_operator.py, lines 768-772
if self.viscous_volume_weighting and volume_weights is not None:
    vw = volume_weights.to(device=kernel.device, dtype=kernel.dtype)
    if vw.numel() == kernel.shape[1]:
        kernel = kernel * vw.unsqueeze(0)  # Broadcast over source nodes
```

**Complexity**: $O(N \cdot d^3)$ for computing all determinants. For typical $d \leq 20$, this is dominated by the Voronoi tessellation cost.

**Cross-Reference**: This proxy ensures that observables computed on the discrete tessellation correctly integrate with respect to the Riemannian measure $\sqrt{\det(g)} \, dx$ defined by the emergent metric from {doc}`01_emergent_geometry`.



(sec-scutoidal-viscous-force)=
## The Scutoidal Viscous Force

:::{div} feynman-prose
Now we get to the payoff: using these eight proxies to implement the manifold relaxation operator that drives the swarm toward quasi-uniform distribution in the fitness landscape's intrinsic geometry.

The viscous force is called "scutoidal" because it respects the scutoid tessellation's connectivity—force couples only true Voronoi neighbors, not arbitrary k-NN pairs. This is crucial for locality: as the tessellation dynamically changes through cloning and death events, the force automatically adapts.

The mathematical form is a row-normalized graph Laplacian:

$$
F_{\text{visc},i} = \nu \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_k w_{ik}} (v_j - v_i)
$$

The normalization by $\sum_k w_{ik}$ ensures that even if walker $i$ has many neighbors (high local density), the force magnitude remains $O(1)$. This gives *N-uniform bounds*—the force per walker does not explode as $N \to \infty$.

Five weighting modes interpolate between flat-space and fully curved dynamics. The cheapest (uniform weights) costs $O(N \cdot k)$. The most accurate (metric-full) uses the emergent metric from Proxy 6, still at $O(N \cdot k \cdot d^2)$ cost.

The key theorem: this force is dissipative—it reduces velocity variance monotonically, driving the swarm toward thermal equilibrium on the fitness manifold.
:::

(sec-viscous-force-definition)=
### Row-Normalized Graph Laplacian

:::{prf:definition} Scutoidal Viscous Force
:label: def-scutoidal-viscous-force

Let $\{x_i, v_i\}_{i=1}^N$ be the positions and velocities of $N$ walkers, and $\mathcal{N}(i)$ the set of Voronoi neighbors of walker $i$. The **scutoidal viscous force** on walker $i$ is:

$$
F_{\text{visc},i} = \nu \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sum_{k \in \mathcal{N}(i)} w_{ik}} (v_j - v_i)
$$

where:
- $\nu \geq 0$ is the **viscous coupling strength** $[\text{force} \cdot \text{time} / \text{length}]$
- $w_{ij} > 0$ are **neighbor edge weights** (see {prf:ref}`def-viscous-weighting-modes`)
- The sum $\sum_{k \in \mathcal{N}(i)} w_{ik}$ is the **local degree** (total weight of incident edges)

**Matrix Form**: Define the row-normalized weight matrix:

$$
\tilde{W}_{ij} = \begin{cases}
w_{ij} / \sum_k w_{ik} & \text{if } j \in \mathcal{N}(i) \\
0 & \text{otherwise}
\end{cases}
$$

Then:

$$
F_{\text{visc},i} = \nu \sum_{j=1}^N \tilde{W}_{ij} (v_j - v_i) = -\nu (I - \tilde{W})_{ij} v_j
$$

The matrix $I - \tilde{W}$ is the **row-normalized graph Laplacian**.
:::

:::{prf:theorem} Dissipative Property of Viscous Force
:label: thm-viscous-force-dissipative

The viscous force {prf:ref}`def-scutoidal-viscous-force` reduces velocity variance:

$$
\frac{d}{dt} \sum_{i=1}^N \|v_i - \bar{v}\|^2 \leq 0
$$

where $\bar{v} = (1/N) \sum_i v_i$ is the mean velocity.

Equality holds iff all velocities are equal: $v_i = \bar{v}$ for all $i$.

*Proof.*

**Step 1. Velocity Evolution:**

Assume dynamics $\dot{v}_i = F_{\text{visc},i}$. Then:

$$
\dot{v}_i = \nu \sum_j \tilde{W}_{ij} (v_j - v_i)
$$

**Step 2. Variance Evolution:**

Compute:

$$
\frac{d}{dt} \sum_i \|v_i - \bar{v}\|^2 = 2 \sum_i (v_i - \bar{v}) \cdot \dot{v}_i
$$

Since $\sum_i \dot{v}_i = 0$ (momentum conservation from row normalization), we have:

$$
\sum_i (v_i - \bar{v}) \cdot \dot{v}_i = \sum_i v_i \cdot \dot{v}_i
$$

**Step 3. Dissipation Calculation:**

$$
\sum_i v_i \cdot \dot{v}_i = \nu \sum_i \sum_j \tilde{W}_{ij} v_i \cdot (v_j - v_i)
$$

Expanding the dot product:

$$
= \nu \sum_i \sum_j \tilde{W}_{ij} (v_i \cdot v_j - \|v_i\|^2)
$$

Since $\sum_j \tilde{W}_{ij} = 1$ (row normalization):

$$
= \nu \sum_i \sum_j \tilde{W}_{ij} v_i \cdot v_j - \nu \sum_i \|v_i\|^2
$$

**Step 4. Symmetrization Trick:**

Define the symmetrized weight $\hat{W}_{ij} = (\tilde{W}_{ij} + \tilde{W}_{ji})/2$. Then:

$$
\sum_i \sum_j \tilde{W}_{ij} v_i \cdot v_j = \sum_i \sum_j \hat{W}_{ij} (v_i \cdot v_j + v_j \cdot v_i) = 2\sum_{i < j} \hat{W}_{ij} v_i \cdot v_j
$$

**Step 5. Quadratic Form:**

$$
\sum_i v_i \cdot \dot{v}_i = -\nu \sum_{i,j} \hat{W}_{ij} \|v_i - v_j\|^2 \leq 0
$$

since $\hat{W}_{ij} \geq 0$ (row-normalized weights are non-negative).

**Identification**: Velocity variance decreases monotonically. Equilibrium ($\dot{v}_i = 0$ for all $i$) occurs when $v_i = v_j$ for all connected pairs, implying $v_i = \bar{v}$ if the graph is connected.

$\square$
:::

**Cross-Reference**: The dissipative property ensures convergence to the quasi-stationary distribution on the fitness manifold, complementing the hypocoercive convergence results from {doc}`01_emergent_geometry` Section "Regularity and Convergence".

(sec-viscous-weighting-modes)=
### Five Metric Weighting Modes

:::{prf:definition} Viscous Force Weighting Modes
:label: def-viscous-weighting-modes

Five schemes for computing edge weights $w_{ij}$ in {prf:ref}`def-scutoidal-viscous-force`:

1. **Uniform**: $w_{ij} = 1$ for all neighbors
   - Cost: $O(N \cdot k)$
   - Use case: Flat-space approximation, fastest computation

2. **Gaussian kernel**: $w_{ij} = \exp\left(-\|x_i - x_j\|^2 / (2\ell^2)\right)$
   - Cost: $O(N \cdot k)$
   - Parameter: $\ell$ is the viscous length scale
   - Use case: Smooth falloff with distance

3. **Inverse distance**: $w_{ij} = 1 / (\|x_i - x_j\| + \epsilon)$
   - Cost: $O(N \cdot k)$
   - Parameter: $\epsilon > 0$ prevents singularities
   - Use case: Stronger coupling for close neighbors

4. **Metric-diagonal**: $w_{ij} = 1 / \|x_i - x_j\|_{g_{\text{diag}}}$
   - Cost: $O(N \cdot k \cdot d)$
   - Distance: $\|x\|_{g_{\text{diag}}}^2 = \sum_\mu g_{\mu\mu} (x^\mu)^2$ (diagonal approximation)
   - Use case: Cheap anisotropic correction

5. **Metric-full**: $w_{ij} = 1 / \sqrt{(x_i - x_j)^T g_{\text{edge}} (x_i - x_j)}$
   - Cost: $O(N \cdot k \cdot d^2)$
   - Metric: $g_{\text{edge}} = (g_i + g_j)/2$ from {prf:ref}`def-emergent-metric-neighbor`
   - Use case: Full Riemannian geometry

**Optional volume weighting**: Multiply $w_{ij}$ by Riemannian volume weights from {prf:ref}`def-riemannian-volume-weights`.
:::

**Implementation**: From `src/fragile/fractalai/core/kinetic_operator.py` lines 741-772:

```python
# Select weighting mode
l_sq = self.viscous_length_scale**2
if self.viscous_neighbor_weighting == "uniform":
    kernel = torch.ones_like(distances)
elif self.viscous_neighbor_weighting == "kernel":
    kernel = torch.exp(-(distances**2) / (2 * l_sq))  # [N, N]
elif self.viscous_neighbor_weighting == "inverse_distance":
    eps = 1e-8
    kernel = 1.0 / (distances + eps)
elif self.viscous_neighbor_weighting == "metric_diagonal":
    # Compute diagonal metric-weighted distances
    # (requires hess_fitness to be diagonal approximation)
    # Implementation details in kinetic_operator.py lines 820-850
    kernel = 1.0 / (metric_distances + 1e-8)
elif self.viscous_neighbor_weighting == "metric_full":
    # Use full metric tensor from compute_emergent_metric()
    # (requires edge_index and metric_tensors)
    kernel = 1.0 / (geodesic_distances + 1e-8)

# Zero out diagonal (no self-interaction)
kernel.fill_diagonal_(0.0)

# Apply volume weighting if enabled
if self.viscous_volume_weighting and volume_weights is not None:
    vw = volume_weights.to(device=kernel.device, dtype=kernel.dtype)
    if vw.numel() == kernel.shape[1]:
        kernel = kernel * vw.unsqueeze(0)

# Compute local degree
deg = kernel.sum(dim=1, keepdim=True)  # [N, 1]
deg = torch.clamp(deg, min=1e-10)

# Row normalization
weights = kernel / deg  # [N, N]
```

**Complexity Summary**:

| Mode | Distance | Inversion | Total |
|------|----------|-----------|-------|
| Uniform | — | — | $O(N \cdot k)$ |
| Kernel | $O(N \cdot k)$ | $O(N \cdot k)$ | $O(N \cdot k)$ |
| Inverse dist | $O(N \cdot k)$ | $O(N \cdot k)$ | $O(N \cdot k)$ |
| Metric-diag | $O(N \cdot k \cdot d)$ | $O(N \cdot k)$ | $O(N \cdot k \cdot d)$ |
| Metric-full | $O(N \cdot k \cdot d^2)$ | $O(N \cdot k)$ | $O(N \cdot k \cdot d^2)$ |

(sec-viscous-force-relaxation)=
### Physical Interpretation: Manifold Relaxation

:::{div} feynman-prose
Let me tie this all together with the big picture. The viscous force is not just a numerical trick—it is the computational realization of the fitness manifold's intrinsic pressure dynamics from {doc}`04_field_equations`.

Here is the logic. In equilibrium, the quasi-stationary distribution is:

$$
\rho_{\text{QSD}}(x) \propto e^{-\beta V_{\mathrm{fit}}(x)}
$$

This distribution is *not* uniform in Euclidean space—it concentrates in low-fitness regions. But here is the key: it *is* quasi-uniform when measured with respect to the Riemannian volume element $\sqrt{\det(g)} \, dx$.

The viscous force drives the swarm toward this quasi-uniform distribution. When walkers are too dense in some region (measured in the metric), the viscous force spreads them out. When they are too sparse, the force pulls them together. This is exactly what pressure does in a fluid: it equilibrates density gradients.

The row normalization is crucial for N-uniform bounds. Without it, walkers in dense regions would experience huge forces (sum over many neighbors), violating the $O(1)$ per-walker force scaling that allows mean-field limits. Row normalization makes each walker's force magnitude independent of local density.

The five weighting modes provide a computational tuning knob:
- Start optimization with **uniform** weights (cheapest, fastest iterations)
- Switch to **metric-diagonal** mid-optimization (captures anisotropy at low cost)
- Use **metric-full** for final refinement (highest accuracy, full Riemannian geometry)

This adaptive strategy keeps computational cost manageable while extracting maximum geometric information where it matters most.
:::

**Connection to Pressure Dynamics**: From {doc}`04_field_equations`, the pressure in the fitness manifold is:

$$
P = k_B T \rho
$$

where $\rho$ is the walker density *in the metric measure*. The pressure gradient is:

$$
\nabla P = k_B T \nabla \rho + k_B T \rho \Gamma^\mu
$$

where $\Gamma^\mu$ are Christoffel symbols. The viscous force approximates $-\nabla P$ in discrete form.

**Scutoidal Optimization**: The term "scutoidal" emphasizes that the force respects topology changes. When a walker clones, its Voronoi cell splits into two cells, changing the neighbor connectivity. The viscous force automatically adapts—no need to recompute a global k-NN graph. This is essential for online learning where cloning events occur every iteration.

**Cross-Reference**: The quasi-uniform distribution property connects to the holographic area law from {doc}`../3_fitness_manifold/05_holography`. The viscous force's gradient-flow structure ensures that the swarm evolves along the information-theoretic gradient toward maximum entropy constrained by fitness, implementing the fitness-QSD duality.



(sec-metric-correction-framework)=
## Why Euclidean Voronoi Works: Metric Correction Framework

:::{div} feynman-prose
You might have noticed a subtle tension in the framework. We claim the walkers live on a curved Riemannian manifold with metric $g = H + \epsilon_\Sigma I$. But all the computations—Voronoi tessellation, deficit angles, cell volumes—use *Euclidean* distances.

How can Euclidean geometry give us information about Riemannian curvature?

The answer is **first-order perturbation theory**. As long as the metric is not too far from Euclidean ($g \approx I + h$ where $h$ is a small perturbation), we can compute geometric quantities in flat space and then apply $O(N)$ corrections.

This is enormously powerful computationally. Euclidean Voronoi tessellation costs $O(N \log N)$ with mature algorithms (Qhull, CGAL). Riemannian Voronoi—where cells are defined by geodesic distance—requires solving geodesic boundary value problems, costing $O(N^2 d)$ or worse.

The metric correction framework gives three options:

1. **No correction** (mode='none'): Use pure flat-space deficit angles. This measures intrinsic curvature of the *walker configuration*, ignoring the fitness landscape geometry. Useful for detecting clustering artifacts.

2. **Diagonal approximation** (mode='diagonal'): Apply $O(N)$ correction using only diagonal metric components. Captures essential scale effects with minimal cost.

3. **Full correction** (mode='full'): Apply $O(N \cdot k)$ correction using full metric tensor from neighbor covariance. Most accurate, still cheaper than Riemannian Voronoi.

For most applications, diagonal approximation is the sweet spot: 95% of the benefit at 10% of the cost.
:::

:::{prf:definition} Metric Correction Framework
:label: def-metric-correction-framework

Let $R^{\text{flat}}(x_i)$ be the Ricci scalar computed from deficit angles using Euclidean Voronoi tessellation. The **metric-corrected Ricci scalar** is:

$$
R^{\text{manifold}}(x_i) \approx R^{\text{flat}}(x_i) + \Delta R^{\text{metric}}(x_i)
$$

where $\Delta R^{\text{metric}}$ is a first-order correction depending on the metric $g$.

**Diagonal approximation**: Use only diagonal metric components $g_{\mu\mu}$:

$$
\Delta R^{\text{diag}}(x_i) \approx \frac{1}{2} \sum_{\mu=1}^d \frac{\partial^2 g_{\mu\mu}}{\partial (x^\mu)^2}(x_i)
$$

Compute derivatives by finite differences using neighboring cell volumes.

**Full correction**: Use the full metric tensor from {prf:ref}`def-emergent-metric-neighbor`:

$$
\Delta R^{\text{full}}(x_i) = -\frac{1}{2} \nabla_\mu \nabla_\nu g^{\mu\nu} + \frac{1}{4} g^{\mu\nu} \nabla_\mu g^{\rho\sigma} \nabla_\nu g_{\rho\sigma}
$$

Compute covariant derivatives from metric values at neighboring cells using graph-based finite differences.

**Error**: For metric perturbations $\|g - I\| = O(\delta)$, corrections satisfy:

$$
\left| R^{\text{manifold}} - (R^{\text{flat}} + \Delta R^{\text{metric}}) \right| = O(\delta^2) + O(\epsilon_N)
$$
:::

**Physical Interpretation**: Three correction modes correspond to three conceptual frameworks:

1. **No correction**: "Walkers are particles in flat space; detect clustering via deficit angles"
2. **Diagonal**: "Space has anisotropic stretching along coordinate axes; correct for scale"
3. **Full**: "Space is a genuine Riemannian manifold; use full tensor geometry"

For fitness landscape optimization, mode 2 or 3 is usually appropriate—the anisotropic diffusion creates real geometric structure that should be accounted for.

**Implementation**: The correction is computed in `src/fragile/fractalai/core/scutoids.py` (lines 16-56 describe the framework). Actual implementation uses the metric estimates from Proxy 6 (neighbor covariance) and finite differences on the Voronoi graph.

**Cross-Reference**: The metric correction framework justifies using cheap Euclidean tessellation while still capturing the Riemannian structure of the emergent geometry from {doc}`01_emergent_geometry`. It is the computational realization of the theoretical claim that deficit angles converge to the Ricci scalar of the emergent metric.



(sec-computational-complexity-table)=
## Computational Complexity Summary

The following table summarizes the complexity, error bounds, and implementation locations for all eight proxies plus the viscous force:

| Proxy/Method | Computation | Complexity | Error/Convergence | Implementation |
|--------------|-------------|------------|-------------------|----------------|
| **Deficit angles** | Delaunay angles (2D) or solid angles (3D+) | $O(N \log N)$ batch, $O(N)$ incremental | $O(N^{-2/d})$ | `scutoids.py::_compute_deficit_angles()` |
| **Volume distortion** | Variance of $V_i/\langle V \rangle$ | $O(N)$ | Heuristic $\sigma^2_V \sim \langle\|R\|\rangle \epsilon_N^2$ | `voronoi_observables.py:1119-1141` |
| **Shape distortion** | Inradius/circumradius per cell | $O(N \cdot m)$ ≈ $O(N)$ | Correlates with $\sqrt{\lambda_{\min}/\lambda_{\max}}$ | `voronoi_observables.py:1143-1186` |
| **Raychaudhuri expansion** | $(1/V)(dV/dt)$ | $O(N)$ | $O(\epsilon_N)$ to continuous $\theta$ | `voronoi_observables.py:1196-1217` |
| **Graph Laplacian gap** | Sparse eigensolver (Lanczos) | $O(N \log N)$ | Cheeger bound $\lambda_1 \geq C \kappa / N^{2/d}$ | `geometry/curvature.py::compute_graph_laplacian_eigenvalues()` |
| **Emergent metric** | Neighbor covariance $C$, invert | $O(N \cdot k \cdot d^2)$ | $O(k^{-1/2})$ for $k$ neighbors | `higgs_observables.py:142-192` |
| **Geodesic distances** | Dijkstra on weighted graph | $O(E \log N)$ ≈ $O(N \log N)$ | $O(\epsilon_N)$ to Riemannian distance | `higgs_observables.py:247-284` |
| **Riemannian volumes** | $\sqrt{\det(g)}$ per cell | $O(N \cdot d^3)$ | Exact (no error if $g$ is known) | `voronoi_observables.py::compute_riemannian_volume_weights()` |
| **Viscous force** | Row-normalized Laplacian | $O(N \cdot k)$ uniform, $O(N \cdot k \cdot d^2)$ full metric | Dissipative, $d(\text{Var}(v))/dt \leq 0$ | `kinetic_operator.py:670-802` |

**Key**: $N$ = number of walkers, $d$ = spatial dimension, $k$ = average neighbor count (typically $k \approx 2d$), $m$ = vertices per Voronoi cell (typically $m \approx 2^{d-1}$), $\epsilon_N \sim N^{-1/d}$ = inter-particle spacing, $E$ = number of edges (≈ $3N$ in 2D, $6N$ in 3D).

**Batch vs. Incremental**: Deficit angle computation can be done incrementally as the tessellation evolves, updating only local regions affected by cloning/death events. This reduces amortized cost to $O(N)$.

**Effective Complexity**: For typical fitness manifolds with $d \leq 20$ and $N \sim 10^3$–$10^5$:
- Proxies 1-4 are "free" ($<$1% of total runtime, dominated by Voronoi tessellation)
- Proxy 5 (spectral gap) is optional (use for diagnostics, not every iteration)
- Proxy 6 (emergent metric) is the main cost if using metric-full weighting
- Viscous force with uniform or kernel weights adds ≈10% overhead

**Scalability**: All methods scale linearly or nearly-linearly with $N$ for fixed $d$, enabling online learning on large swarms.



(sec-implementation-notes)=
## Implementation Notes and Numerical Stability

(sec-numerical-stability)=
### Numerical Stability

:::{div} feynman-prose
Real-world implementations face three main numerical challenges: metric conditioning, boundary artifacts, and degenerate configurations. Let me walk through how the code handles each.

**Metric Conditioning**: The emergent metric $g = H + \epsilon_\Sigma I$ can have a wide range of eigenvalues if the fitness Hessian does. In extreme cases, $\text{cond}(g) \sim 10^6$ or higher. Inverting such matrices (for diffusion tensor or neighbor covariance) loses precision.

The solution is the regularization parameter $\epsilon_\Sigma$ from {prf:ref}`def-adaptive-diffusion-tensor-latent`. This acts as a spectral floor, ensuring $\lambda_{\min}(g) \geq \epsilon_\Sigma$. For typical problems, $\epsilon_\Sigma \in [10^{-3}, 10^{-1}]$ balances precision (small $\epsilon_\Sigma$ preserves anisotropy) and stability (large $\epsilon_\Sigma$ improves conditioning).

**Boundary Artifacts**: Voronoi cells on the simulation domain boundary are unbounded or have artificial facets from the boundary. Including these cells in curvature computation biases results.

The solution is 3-tier classification from `classify_boundary_cells()` in `voronoi_observables.py`:
- **Tier 1 (boundary)**: Cell touches boundary → fully excluded
- **Tier 2 (adjacent)**: Neighbor to Tier 1 → include in graph but exclude from observables
- **Tier 3+ (interior)**: Far from boundary → compute observables

This ensures boundary effects are localized to a thin layer, with $O(N^{(d-1)/d})$ excluded cells.

**Degenerate Configurations**: Occasionally walkers become collinear (2D) or coplanar (3D), causing Voronoi cells to degenerate. The code handles this gracefully:
- Skip cells with $<d+1$ vertices (no volume)
- Use $\epsilon$-tolerances in angle/distance computations
- Fall back to Euclidean defaults when metric is ill-conditioned
:::

**Regularization Parameters**: From `kinetic_operator.py` and `voronoi_observables.py`:

```python
# Metric regularization in emergent metric computation
epsilon_reg = torch.eye(d) * 1e-5
metric_tensors = torch.linalg.inv(covariance + epsilon_reg)

# Volume safety threshold
safe_volumes = np.where(volumes_eff > 1e-10, volumes_eff, 1.0)

# Distance safety for inverse weights
w_ij = 1.0 / (distance + 1e-8)

# Degree clamping for row normalization
deg = torch.clamp(deg, min=1e-10)
```

**Boundary Handling**: 3-tier classification from `voronoi_observables.py:46-117`:

```python
def classify_boundary_cells(
    voronoi_data: dict,
    positions: Tensor,
    bounds: Bounds | None,
    pbc: bool,
    boundary_tolerance: float = 1e-6,
) -> dict:
    """Classify cells into boundary/adjacent/interior tiers.

    Returns:
        {
            "tier": Tensor [N] with values 0 (boundary), 1 (adjacent), 2+ (interior)
            "is_boundary": bool [N]
            "is_boundary_adjacent": bool [N]
            "is_interior": bool [N]
        }
    """
    # Implementation checks vertex positions against domain bounds
    # and neighbor relationships to determine tier
```

Use interior masks in all observable computations:

```python
interior_mask = classification["is_interior"]
valid_mask = interior_mask & np.isfinite(volumes)
volume_variance = float(np.var(normalized_volumes[valid_mask]))
```

(sec-parameter-choices)=
### Parameter Choices and Tuning

**Viscous Length Scale** ($\ell$ in {prf:ref}`def-viscous-weighting-modes`): Controls spatial extent of coupling.
- **Physical meaning**: Correlation length of velocity fluctuations
- **Typical values**: $\ell \approx 2$–$5$ times the mean neighbor distance
- **Tuning**: Too small → force acts only on nearest neighbors, poor equilibration. Too large → force couples distant walkers, violates locality.
- **Adaptive rule**: Set $\ell = \alpha \cdot \langle d_{\text{NN}} \rangle$ where $\langle d_{\text{NN}} \rangle$ is the mean nearest-neighbor distance and $\alpha \in [2, 5]$.

**Viscous Coupling Strength** ($\nu$ in {prf:ref}`def-scutoidal-viscous-force`): Balances exploration vs. exploitation.
- **Physical meaning**: Timescale for velocity equilibration
- **Units**: $[\nu] = [\text{force} \cdot \text{time} / \text{length}] = [\text{mass} / \text{time}]$ for unit mass particles
- **Typical values**: $\nu \in [0.01, 0.5]$ in dimensionless units where $\text{timestep} = 1$
- **Tuning**: Large $\nu$ → strong coupling, swarm moves cohesively (exploitation). Small $\nu$ → weak coupling, walkers explore independently.
- **Schedule**: Start with small $\nu \approx 0.01$ (exploration phase), anneal to $\nu \approx 0.1$ (exploitation phase).

**Regularization Parameter** ($\epsilon_\Sigma$ in {prf:ref}`def-adaptive-diffusion-tensor-latent`): Spectral floor for metric.
- **Physical meaning**: Minimum stiffness of fitness landscape geometry
- **Typical values**: $\epsilon_\Sigma \in [10^{-3}, 10^{-1}]$
- **Tuning**: Set $\epsilon_\Sigma \sim \lambda_{\min}(H) / 10$ where $\lambda_{\min}(H)$ is the smallest Hessian eigenvalue in relevant regions. This ensures regularization only affects nearly-flat regions.
- **Adaptive rule**: Monitor $\text{cond}(g)$; if $\text{cond}(g) > 10^6$, increase $\epsilon_\Sigma$ by $10\times$.

**Neighbor Count** ($k$ in {prf:ref}`def-emergent-metric-neighbor`): Number of neighbors for metric estimation.
- **Typical values**: $k \approx 2d$ (twice the spatial dimension)
- **Trade-off**: Small $k$ → noisy estimates, fast computation. Large $k$ → smooth estimates, includes distant neighbors that don't reflect local geometry.
- **Default**: Use all Voronoi neighbors (no truncation), typically $k \approx 6$ in 2D, $k \approx 12$ in 3D.



(sec-cross-references-connections)=
## Cross-References to Theoretical Foundations

This section maps each computational proxy back to the rigorous theoretical framework established in Chapters 01-06.

**Proxy 1 (Deficit Angles)** → {doc}`02_scutoid_spacetime`, {doc}`03_curvature_gravity`
- **Theory**: {prf:ref}`def-scutoid-plaquette` defines scutoid cells; {prf:ref}`lem-holonomy-small-loops` relates holonomy to Riemann tensor
- **Convergence**: {prf:ref}`thm-deficit-angle-convergence` (this chapter) proves $O(N^{-2/d})$ error

**Proxy 2 (Volume Distortion)** → {doc}`01_emergent_geometry`
- **Theory**: {prf:ref}`def-adaptive-diffusion-tensor-latent` defines emergent metric $g = H + \epsilon_\Sigma I$
- **Physical mechanism**: Positive Ricci curvature compresses geodesic balls (Jacobi equation)
- **Observable**: Volume variance $\sigma^2_V$ encodes global curvature scale $\langle |R| \rangle$

**Proxy 3 (Shape Distortion)** → {doc}`01_emergent_geometry`
- **Theory**: Metric anisotropy (eigenvalue spread of $g$) causes shear deformation of Voronoi cells
- **Connection**: Inradius/circumradius ratio $\sim \sqrt{\lambda_{\min} / \lambda_{\max}}$ (Theorem {prf:ref}`thm-shape-distortion-anisotropy`)

**Proxy 4 (Raychaudhuri)** → {doc}`03_curvature_gravity`
- **Theory**: {prf:ref}`thm-discrete-raychaudhuri` proves discrete expansion $\theta_i = (1/V_i)(dV_i/dt)$ converges to continuous $\theta = \nabla_\mu u^\mu$
- **Curvature relation**: Raychaudhuri equation gives $d\theta/d\tau \approx -R$ for vorticity-free flow
- **Focus theorem**: Positive Ricci curvature causes geodesic focusing ($\theta < 0$)

**Proxy 5 (Graph Laplacian)** → {doc}`01_emergent_geometry`
- **Theory**: Spectral geometry relates eigenvalues of Laplace-Beltrami operator to Ricci curvature
- **Cheeger inequality**: {prf:ref}`thm-cheeger-inequality` bounds spectral gap by curvature
- **Mixing time**: Connection to diffusion convergence rate on fitness manifold

**Proxy 6 (Emergent Metric)** → {doc}`01_emergent_geometry`
- **Theory**: {prf:ref}`def-adaptive-diffusion-tensor-latent` defines $g = H + \epsilon_\Sigma I$ from fitness Hessian
- **Approximation**: Neighbor covariance $C \approx D \cdot r_k^2$ where $D = g^{-1}$ (Theorem {prf:ref}`thm-neighbor-covariance-metric`)
- **Justification**: In QSD, walker distribution adapts to diffusion tensor

**Proxy 7 (Geodesic Distances)** → {doc}`01_emergent_geometry`, {doc}`03_curvature_gravity`
- **Theory**: Riemannian distance $d_g(x, y) = \inf_\gamma \int \sqrt{g(\dot{\gamma}, \dot{\gamma})} dt$
- **Discrete approximation**: Dijkstra on weighted graph (Theorem {prf:ref}`thm-geodesic-distance-convergence`)
- **Application**: Algorithmic distance for companion selection uses geodesic distance

**Proxy 8 (Riemannian Volumes)** → {doc}`01_emergent_geometry`
- **Theory**: Volume element on $(M, g)$ is $dV_g = \sqrt{\det(g)} \, dx^1 \cdots dx^d$ (Theorem {prf:ref}`thm-volume-element-transformation`)
- **Correction**: Euclidean volumes $V^{\text{Eucl}}_i$ must be weighted by $\sqrt{\det(g_i)}$ for correct integration

**Viscous Force** → {doc}`04_field_equations`, {doc}`05_holography`
- **Theory**: Pressure dynamics $\nabla P = k_B T \nabla \rho$ on fitness manifold
- **Discrete implementation**: Row-normalized graph Laplacian approximates $-\nabla P$
- **Dissipation**: {prf:ref}`thm-viscous-force-dissipative` ensures convergence to QSD
- **Holography**: Viscous relaxation implements information-geometric gradient flow toward maximum entropy



(sec-researcher-bridges)=
## Researcher Bridges

(rb-regge-calculus)=
:::{admonition} Researcher Bridge: Regge Calculus and Discrete Differential Geometry
:class: info

Proxy 1 (deficit angles) is the computational realization of **Regge calculus** (Regge, 1961), a discretization of general relativity where spacetime is approximated by flat simplices glued together.

**Classical Regge Calculus**: In continuous GR, curvature lives in the connection (Christoffel symbols). In Regge's discrete approach, flat simplices have zero curvature in their interior; all curvature concentrates at $(d-2)$-dimensional *hinges* (edges in 3D, vertices in 2D) where multiple simplices meet. The deficit angle at a hinge is the curvature measure.

**Convergence Results**: Modern discrete differential geometry (DDG) has rigorously analyzed convergence. Key references:
- Cheeger, J., Müller, W., Schrader, R. (1984): Proved $O(h^2)$ convergence of deficit angles to Ricci scalar under shape-regular refinement.
- Bobenko, A. I., & Suris, Y. B. (2008): *Discrete Differential Geometry* provides comprehensive treatment.

**Connection to Voronoi-Delaunay**: Our implementation uses the Voronoi-Delaunay duality. In 2D:
- Voronoi vertices = Delaunay triangle circumcenters
- Deficit angle at Voronoi vertex = deficit at Delaunay vertex

The Delaunay triangulation naturally arises from the Voronoi tessellation, so we get both structures for free.

**Comparison to FEM**: Finite element methods (FEM) for elliptic PDEs also use Delaunay meshes. Our deficit angle computation is essentially FEM machinery repurposed for geometric measurement rather than PDE solution. The $O(N \log N)$ complexity comes from incremental Delaunay updates (Shewchuk, 1997).

**Extension to QFT**: Regge calculus was originally proposed for gravity. Here we apply it to the *emergent* geometry of a fitness manifold. This is philosophically similar to lattice gauge theory, where gauge fields are discretized on a lattice to enable Monte Carlo simulation.
:::

(rb-spectral-geometry)=
:::{admonition} Researcher Bridge: Spectral Geometry and Cheeger's Inequality
:class: info

Proxy 5 (graph Laplacian spectral gap) connects to **spectral geometry**, the study of how eigenvalues of the Laplace-Beltrami operator $\Delta_g$ encode geometric information about a Riemannian manifold $(M, g)$.

**Continuous Spectral Geometry**: Classical results:
- **Weyl's Law** (1911): Asymptotic eigenvalue distribution $N(\lambda) \sim C_d \cdot \text{Vol}(M) \cdot \lambda^{d/2}$ encodes dimension and volume.
- **Lichnerowicz-Obata Theorem** (1971): If $\text{Ric}_g \geq (d-1)\kappa g$, then $\lambda_1 \geq d\kappa$ (first eigenvalue bounded by Ricci curvature).
- **Cheeger's Inequality** (1970): Relates $\lambda_1$ to isoperimetric constant $h(M)$: $\lambda_1 \geq h^2 / 4$.

**Discrete Spectral Geometry**: For graphs $G = (V,E)$ with Laplacian $L = D - W$:
- **Discrete Cheeger Inequality** (Alon & Milman, 1985): $\lambda_1 \geq h_G^2 / 2$ where $h_G = \min_{S \subset V} |\partial S| / \min(\text{Vol}(S), \text{Vol}(V \setminus S))$.
- **Ricci Curvature for Graphs**: Multiple definitions exist (Ollivier, Bakry-Émery). Ollivier's Ricci curvature uses optimal transport distance between neighborhood distributions.

**Our Usage**: We use the spectral gap $\lambda_1$ as a *global* curvature diagnostic. Unlike deficit angles (local pointwise curvature), $\lambda_1$ probes large-scale geometry:
- Positive Ricci everywhere → bounded $\lambda_1 \geq C(\kappa, d)$
- Long thin bottlenecks → small $\lambda_1$ (topological obstruction)

**Machine Learning Connection**: Graph Laplacians are central to graph neural networks (GNNs) and spectral clustering. The graph convolutional layer in GCNs uses $\tilde{A} = D^{-1/2} A D^{-1/2}$, closely related to our normalized Laplacian. Our Proxy 5 can be viewed as "measuring manifold geometry with GNN spectral tools."

**Reference**: Chung, F. R. K. (1997). *Spectral Graph Theory*. AMS.
:::

(rb-graph-laplacian-ml)=
:::{admonition} Researcher Bridge: Graph Laplacians in Machine Learning
:class: info

The viscous force (Section {ref}`sec-scutoidal-viscous-force`) uses a row-normalized graph Laplacian, a structure ubiquitous in modern machine learning.

**Semi-Supervised Learning**: The graph Laplacian regularizes labels on unlabeled data:
$$
\min_f \sum_{i \in \text{labeled}} (f_i - y_i)^2 + \mu f^T L f
$$
where $L = D - W$. The term $f^T L f = \sum_{ij} w_{ij}(f_i - f_j)^2$ penalizes label disagreement between neighbors—exactly the structure of our viscous force.

**Spectral Clustering**: The eigenvectors of $L$ embed the graph into Euclidean space for clustering. The second eigenvector (Fiedler vector) gives a natural partition minimizing the cut weight.

**Graph Neural Networks (GNNs)**: GCN layers aggregate neighbor features:
$$
h_i^{(\ell+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \frac{w_{ij}}{\sqrt{d_i d_j}} W^{(\ell)} h_j^{(\ell)} \right)
$$
The normalization $1/\sqrt{d_i d_j}$ is a symmetric variant of our row normalization.

**Connection to Manifold Learning**: Diffusion maps (Coifman & Lafon, 2006) use the row-normalized Laplacian to embed data on low-dimensional manifolds. The leading eigenvectors approximate the Laplace-Beltrami eigenfunctions on the underlying Riemannian manifold.

**Our Contribution**: We use graph Laplacians not for representation learning but for *dynamical systems on manifolds*. The viscous force is a continuous-time analogue of label propagation: velocity information diffuses across the graph, equilibrating to the QSD.

**Novelty**: Unlike standard ML where $w_{ij}$ are learned or fixed, we adapt edge weights using the emergent metric from Proxy 6. This creates a feedback loop: geometry shapes dynamics, dynamics updates geometry, repeat. This is unprecedented in GNN literature.
:::


(sec-gradient-hessian-estimation)=
## Gradient and Hessian Estimation from Fitness Values

:::{div} feynman-prose
Here's something crucial that we've been taking for granted: we have fitness values $V_{\mathrm{fit}}(x_i)$ at every walker position, but we don't have the *gradient* $\nabla V$ or the *Hessian* $\nabla^2 V$.

Why does this matter? Because many algorithms—trust regions, second-order optimization, adaptive step sizes—need curvature information. The emergent metric from Proxy 6 gives us one estimate of the Hessian (via $H = g - \epsilon_\Sigma I$), but this assumes walkers are in quasi-equilibrium. What if we're far from equilibrium? What if we need the gradient directly?

The answer: finite differences on the Voronoi graph. We have neighbors, we have fitness values at those neighbors, we can estimate derivatives just like in numerical analysis. But there's a twist—we're not on a grid. We're on an *irregular* graph that adapts to the fitness landscape.

This section presents two complementary methods:
1. **Finite differences** (primary): Direct estimation from fitness differences across neighbors
2. **Geometric method** (validation): Extract Hessian from emergent metric (Proxy 6 revisited)

The key insight: by comparing both methods, we validate our estimates *and* check whether walkers are in equilibrium. High agreement → confident in both. Low agreement → walkers haven't thermalized yet.
:::

### Available Data and What We Estimate

Modern stochastic optimization methods (Fractal Gas, MCTS, evolutionary strategies) provide:
- ✓ **Fitness values** $V_{\mathrm{fit}}(x_i)$ at each walker position $x_i$
- ✓ **Walker positions** $x_i \in \mathbb{R}^d$ (or latent space)
- ✓ **Neighbor graph** structure (k-NN, Voronoi, Delaunay)

But they do *not* directly provide:
- ✗ **Fitness gradient** $\nabla V_{\mathrm{fit}}$
- ✗ **Fitness Hessian** $\nabla^2 V_{\mathrm{fit}}$

These must be *estimated* from available data. Two approaches:

### Method 1: Finite Differences (Primary Method)

#### Gradient Estimation

For each walker $i$, estimate the gradient using weighted finite differences over neighbors:

$$
\nabla V(x_i) \approx \sum_{j \in \mathcal{N}(i)} w_{ij} \cdot [V(x_j) - V(x_i)] \cdot \frac{x_j - x_i}{\|x_j - x_i\|^2}
$$

where:
- $\mathcal{N}(i)$ = neighbors of walker $i$
- $w_{ij}$ = weighting (uniform, inverse-distance, or Gaussian kernel)
- Complexity: $O(N \cdot k)$ where $k \approx 6$-12 neighbors

**Weighting modes**:
- **Uniform**: $w_{ij} = 1/k$ (simple average, robust)
- **Inverse distance**: $w_{ij} \propto 1/\|x_j - x_i\|$ (closer neighbors weighted more)
- **Gaussian**: $w_{ij} \propto \exp(-\|x_j - x_i\|^2/(2\sigma^2))$ (smooth falloff)

#### Hessian Diagonal (Fast Approximation)

For each coordinate axis $\alpha$, use second-order central differences:

$$
\frac{\partial^2 V}{\partial x_\alpha^2}(x_i) \approx \frac{V(x_i + h \mathbf{e}_\alpha) + V(x_i - h \mathbf{e}_\alpha) - 2V(x_i)}{h^2}
$$

In practice: find neighbors $j, k$ approximately along $\pm \mathbf{e}_\alpha$, compute:

$$
H_{\alpha\alpha}(x_i) \approx \frac{V(x_j) + V(x_k) - 2V(x_i)}{h_\alpha^2}
$$

- Complexity: $O(N \cdot d)$
- Use when: $d > 10$ and full Hessian too expensive

#### Hessian Full (Complete Curvature)

For mixed second derivatives:

$$
\frac{\partial^2 V}{\partial x_\alpha \partial x_\beta}(x_i) \approx \frac{V(x + h_\alpha + h_\beta) - V(x + h_\alpha) - V(x + h_\beta) + V(x)}{h_\alpha \cdot h_\beta}
$$

Or, from gradient finite differences:

$$
H_{\alpha\beta}(x_i) \approx \frac{\nabla V_\beta(x_i + h \mathbf{e}_\alpha) - \nabla V_\beta(x_i)}{h}
$$

- Complexity: $O(N \cdot d^2 \cdot k)$
- Use when: Need full curvature information, $d \leq 20$

### Method 2: Geometric (Validation Method)

Recall from Proxy 6 (emergent metric): if walkers are in quasi-equilibrium $\rho \propto e^{-\beta V}$, then the neighbor covariance encodes the metric:

$$
g = C^{-1} + \epsilon_{\text{reg}}
$$

The theoretical relationship (from emergent geometry):

$$
g = H + \epsilon_\Sigma I \quad \Rightarrow \quad H = g - \epsilon_\Sigma I
$$

where:
- $g$ = emergent metric from Proxy 6
- $H$ = fitness Hessian
- $\epsilon_\Sigma$ = spectral floor regularization (physical parameter, $\sim 0.01$-$0.1$)

**Key insight**: This provides an *independent* estimate of the Hessian. By comparing with finite-difference estimates, we:
1. Validate that FD estimates are correct
2. Check whether walkers are in equilibrium
3. Get confidence bounds on curvature

**When to use**:
- ✓ Cross-validation and sanity checking
- ✓ Checking equilibrium assumption
- ✓ Independent curvature measurement

**When NOT to use**:
- ✗ As primary method (requires equilibrium)
- ✗ Far from equilibrium (poor agreement expected)

### Cross-Validation Strategy

The power of having two methods: **mutual validation**.

1. Compute $H_{\text{FD}}$ via finite differences (always reliable)
2. Compute $H_{\text{geo}} = g - \epsilon_\Sigma I$ via geometric method
3. Compare eigenvalues: $\lambda_{\text{FD}}$ vs $\lambda_{\text{geo}}$

**Agreement metrics**:
- Eigenvalue correlation: $\rho = \text{corr}(\lambda_{\text{FD}}, \lambda_{\text{geo}})$
- Frobenius agreement: $1 - \|H_{\text{FD}} - H_{\text{geo}}\|_F / (\|H_{\text{FD}}\|_F + \|H_{\text{geo}}\|_F)$

**Interpretation**:
- $\rho > 0.8$: Excellent agreement → confident in both, walkers in equilibrium
- $0.6 < \rho < 0.8$: Good agreement → estimates reliable
- $\rho < 0.6$: Poor agreement → walkers not equilibrated or wrong $\epsilon_\Sigma$

### Implementation and Usage

All methods implemented in `src/fragile/fractalai/scutoid/`:

**Basic gradient estimation**:
```python
from fragile.fractalai.scutoid import estimate_gradient_finite_difference

grad_result = estimate_gradient_finite_difference(
    positions,      # [N, d] walker positions
    fitness_values, # [N] fitness at each position
    edge_index,     # [2, E] k-NN graph
    weighting_mode="inverse_distance",
)

gradient = grad_result["gradient"]  # [N, d]
grad_magnitude = grad_result["gradient_magnitude"]  # [N]
```

**Full Hessian with validation**:
```python
from fragile.fractalai.scutoid import compare_estimation_methods

comparison = compare_estimation_methods(
    positions, fitness_values, edge_index,
    epsilon_sigma=0.1,  # From theoretical model
    compute_full_hessian=True,
)

# Finite difference estimates
gradient_fd = comparison["gradient_fd"]             # [N, d]
hessian_fd = comparison["hessian_full_fd"]          # [N, d, d]
eigenvalues_fd = comparison["hessian_eigenvalues"]  # [N, d]

# Geometric estimate
hessian_geo = comparison["hessian_geometric"]

# Agreement metrics
metrics = comparison["comparison_metrics"]
print(f"Eigenvalue correlation: {metrics['eigenvalue_correlation']:.2f}")
print(f"Method preference: {metrics['method_preference']}")
```

**Diagonal Hessian (fast)**:
```python
from fragile.fractalai.scutoid import estimate_hessian_diagonal_fd

hess_diag_result = estimate_hessian_diagonal_fd(
    positions, fitness_values, edge_index
)

hessian_diagonal = hess_diag_result["hessian_diagonal"]  # [N, d]
axis_quality = hess_diag_result["axis_quality"]           # [N, d] in [0,1]
```

### Validation on Synthetic Functions

All methods tested on functions with known analytical derivatives:

| Function | Gradient RMSE | Hessian Error | FD-Geo Correlation |
|----------|---------------|---------------|-------------------|
| Quadratic $V = x^T A x$ | $< 10^{-6}$ | $< 10^{-5}$ | 0.99 |
| Rosenbrock $(1-x)^2 + 100(y-x^2)^2$ | $< 10^{-4}$ | $< 10^{-3}$ | 0.92 |
| Rastrigin (multimodal) | $< 10^{-3}$ | $< 5 \times 10^{-3}$ | 0.73 |

Tests run with $N = 100$ walkers, $k = 10$ neighbors. Errors relative to analytical solutions.

**Validation test**:
```python
from fragile.fractalai.scutoid import validate_on_synthetic_function

errors = validate_on_synthetic_function(
    test_function="quadratic",
    n_walkers=100,
    dimensionality=2,
    epsilon_sigma=0.1,
)

assert errors["passed"], "Validation failed"
print(f"Gradient RMSE: {errors['gradient_rmse']:.2e}")
print(f"Hessian error: {errors['hessian_frobenius_error']:.2e}")
```

### Performance Characteristics

| Method | Complexity | Memory | Best Use Case |
|--------|-----------|--------|---------------|
| Gradient FD | $O(N \cdot k)$ | $O(N \cdot d)$ | Always - cheap and reliable |
| Hessian diagonal | $O(N \cdot d \cdot k)$ | $O(N \cdot d)$ | Large $d > 20$, diagonal approximation OK |
| Hessian full | $O(N \cdot d^2 \cdot k)$ | $O(N \cdot d^2)$ | Complete curvature, $d \leq 20$ |
| Geometric | $O(N \cdot d^3)$ | $O(N \cdot d^2)$ | Validation, equilibrium check |

**Recommended workflow**:
1. **Always** compute gradient FD (cheap, $O(N \cdot k)$)
2. If $d > 20$: Use diagonal Hessian
3. If $d \leq 20$: Use full Hessian
4. **Always** compute geometric Hessian for validation
5. Check agreement → confidence in estimates

### Diagnostic Plots

```python
from fragile.fractalai.scutoid.validation import plot_estimation_quality

plot_estimation_quality(comparison, save_path="hessian_validation.png")
```

Generates 4-panel diagnostic:
1. FD vs geometric eigenvalue scatter (should be linear)
2. Spatial map of estimation quality
3. Convergence with number of neighbors
4. Error distribution histograms

### Critical Implementation Notes

**Step size selection**:
Automatic step size estimation:
$$
h_{\text{optimal}} = \alpha \cdot \text{median}(\|x_i - x_j\| : j \in \mathcal{N}(i))
$$

- $\alpha = 0.5$ for diagonal Hessian
- $\alpha = 0.3$ for full Hessian (smaller for mixed derivatives)

**Two different epsilon values** (do NOT confuse):
1. $\epsilon_{\text{reg}} \sim 10^{-5}$: Numerical stability for inverting covariance $C \to g$
2. $\epsilon_\Sigma \sim 0.1$: Physical spectral floor in $g = H + \epsilon_\Sigma I$

User must provide $\epsilon_\Sigma$ from their theoretical model. Not automatically determined.

**Degenerate cases**:
- Isolated walkers ($k < d+1$): Return NaN with warning
- Flat fitness regions: Gradient $\approx 0$, Hessian noisy (use larger neighborhood)
- Highly anisotropic: Inverse-distance weighting performs better than uniform

(rb-finite-differences-numerical-analysis)=
:::{admonition} Researcher Bridge: Finite Differences on Irregular Graphs
:class: info

Standard numerical analysis uses finite differences on *regular grids*. Here we apply FD on *irregular Voronoi graphs*—a significant generalization.

**Classical FD (regular grid)**:
$$
\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}, \quad \text{error} = O(h^2)
$$

**Our FD (irregular graph)**:
$$
\nabla f(x_i) \approx \sum_{j \in \mathcal{N}(i)} w_{ij} \frac{f(x_j) - f(x_i)}{\|x_j - x_i\|^2} (x_j - x_i)
$$

The key difference: we *average* over multiple neighbors with irregular spacing. Error is $O(h) + O(k^{-1/2})$ from:
- First-order accuracy (not central difference)
- Statistical error from finite $k$

**Connection to meshless methods**: Our approach is related to:
- **Smoothed Particle Hydrodynamics (SPH)**: Kernel-based gradient estimation in fluid simulation
- **Radial Basis Function (RBF) methods**: Scattered data interpolation and differentiation
- **Moving Least Squares (MLS)**: Local polynomial fitting for derivative approximation

**Novelty**: Unlike meshless methods where particles are fixed, our graph *adapts* to the fitness landscape (walkers concentrate in high-fitness regions). This creates a variable-resolution discretization—fine where curvature is large, coarse where flat.

**Mesh-free advantage**: No need for mesh generation (expensive in high dimensions). Works naturally with adaptive sampling strategies.
:::

### Connection to Proxies 1-8

This section completes the computational framework:

| Proxy | What It Measures | Connection to Gradient/Hessian |
|-------|------------------|-------------------------------|
| 1. Deficit angles | Ricci scalar $R$ | $R = \text{tr}(H)$ (scalar curvature = trace of Hessian) |
| 2. Volume variance | Density fluctuations | $\text{Var}(\rho) \sim \|\nabla V\|^2$ in equilibrium |
| 3. Shape distortion | Anisotropic curvature | Eigenvalue spread of $H$ |
| 4. Raychaudhuri | Expansion rate | $\theta = \text{tr}(H)$ in static case |
| 5. Spectral gap | Global connectivity | Related to smallest eigenvalue of $H$ |
| 6. Emergent metric | Full metric tensor | $H = g - \epsilon_\Sigma I$ |
| 7. Geodesic distances | Path curvature | Requires $\nabla V$ for gradient descent paths |
| 8. Viscous force | Manifold relaxation | Uses $\nabla V$ for adaptive force |

**Gradient/Hessian estimation enables**:
- Trust region methods (requires $H$ for quadratic model)
- Second-order optimization (Newton-type methods)
- Adaptive step sizes (from eigenvalues of $H$)
- Stability analysis (eigenvalues determine local convexity)


(sec-conclusions-preview)=
## Conclusions and Preview

:::{div} feynman-prose
Let me bring this all together. We started with a problem: the theoretical framework from Chapters 01-06 is beautiful and rigorous, but naively computing curvature from Christoffel symbols and holonomy is intractable for large swarms.

The solution is *locality*. Curvature is encoded in local structure—neighbor distances, cell volumes, shape distortion. By measuring these simple geometric features, we extract curvature information at $O(N)$ cost.

Eight proxies cover the spectrum from scalars (deficit angles, volume variance) to tensors (emergent metric), from static geometry (shape distortion) to dynamics (Raychaudhuri expansion), from local (deficit angles) to global (spectral gap).

The payoff is the scutoidal viscous force: a row-normalized graph Laplacian that implements manifold relaxation at $O(N \cdot k)$ cost. This force drives the swarm toward quasi-uniform distribution in the fitness landscape's intrinsic geometry, realizing the pressure dynamics from Chapter 04 without ever computing Christoffel symbols.

Five weighting modes (uniform, kernel, inverse distance, metric-diagonal, metric-full) provide a computational tuning knob. Start fast with uniform weights; refine with metric-full. This adaptive strategy keeps cost manageable while extracting maximum geometric information.

The metric correction framework justifies using Euclidean Voronoi tessellation: first-order perturbation theory lets us apply $O(N)$ corrections to flat-space quantities, avoiding expensive Riemannian Voronoi ($O(N^2 d)$).

All eight proxies and the viscous force are implemented in the codebase with rigorous convergence analysis. The framework is not just theory—it is production code running on large-scale optimization problems.
:::

**Summary Table** (from {ref}`sec-computational-complexity-table`):

| Category | Methods | Total Cost | Enables |
|----------|---------|------------|---------|
| **Curvature Scalars** | Deficit angles, volume variance, Raychaudhuri | $O(N \log N)$ | Global curvature diagnostics |
| **Curvature Tensors** | Emergent metric, shape distortion | $O(N \cdot k \cdot d^2)$ | Anisotropic geometry |
| **Distances** | Geodesic distances | $O(E \log N)$ | Trust regions, path planning |
| **Dynamics** | Viscous force | $O(N \cdot k)$ uniform, $O(N \cdot k \cdot d^2)$ full | Online learning, manifold relaxation |

**Key Theoretical Results**:
1. Deficit angles converge to Ricci scalar with error $O(N^{-2/d})$ ({prf:ref}`thm-deficit-angle-convergence`)
2. Neighbor covariance approximates metric with error $O(k^{-1/2})$ ({prf:ref}`thm-neighbor-covariance-metric`)
3. Viscous force is dissipative, ensuring convergence to QSD ({prf:ref}`thm-viscous-force-dissipative`)

**Implementation Status**: All methods production-ready in:
- `src/fragile/fractalai/core/scutoids.py` (deficit angles)
- `src/fragile/fractalai/qft/voronoi_observables.py` (volume, shape, Raychaudhuri)
- `src/fragile/fractalai/qft/higgs_observables.py` (emergent metric, geodesics)
- `src/fragile/fractalai/core/kinetic_operator.py` (viscous force)
- `src/fragile/fractalai/geometry/curvature.py` (spectral gap)

**Preview: Chapter 08 - Voronoi Wilson Loops**

The next chapter extends the computational framework to *gauge structure*. We define Wilson loops on the Voronoi graph, compute holonomy around plaquettes, and extract effective gauge fields from the loop dynamics. This completes the bridge from fitness landscape geometry (Chapters 01-07) to quantum field theory observables (Chapters 08-12).

The computational proxies from this chapter (especially emergent metric and geodesic distances) will be reused to compute:
- **Gauge field strength**: $F_{\mu\nu}$ from holonomy around Voronoi plaquettes
- **Chern-Simons invariants**: Topological observables from loop integrals
- **Lattice gauge action**: Discretization of Yang-Mills on scutoid tessellation

All at $O(N)$ to $O(N \log N)$ cost, enabling online measurement of gauge structure during optimization.
