(sec-voronoi-wilson-loops)=
# Voronoi Wilson Loops on the Fitness Manifold

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`02_scutoid_spacetime`, {doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`


(sec-tldr-voronoi-wilson)=
## TLDR

*Notation: $\mathrm{Vor}_i(t)$ = Voronoi cell of walker $i$ at time $t$; $\mathcal{N}_i(t)$ = neighbor set from Delaunay/Voronoi adjacency ({prf:ref}`def-neighbor-set`); $g = H + \epsilon_\Sigma I$ = emergent metric ({prf:ref}`def-adaptive-diffusion-tensor-latent`); $d_g$ = geodesic distance; $U(e)$ = edge parallel transport; $W[\gamma]$ = Wilson loop ({prf:ref}`def-wilson-loop-lqft`); $\theta_{ij}$ = fitness phase on neighbor edge.*

**Goal**: define Wilson loops directly on the Voronoi/Delaunay complex that is recorded in `RunHistory` (neighbor edges + Voronoi metadata), without requiring a Fractal Set build. This gives a Voronoi-cell version of the same gauge-loop observables used in the Fractal Set, computed from the data we already collect at runtime.

**Plan**: use (1) per-slice Delaunay neighbor edges as spacelike links, (2) time-sliced Voronoi adjacency as timelike links, (3) edge phases from fitness/cloning/viscous data in `RunHistory`, and (4) minimal cycles (triangles and 4-cycles) to form Wilson loops and plaquettes. The resulting loop statistics can be computed in $O(N)$ per recorded slice when the Delaunay complex is linear size.


(sec-voronoi-wilson-intro)=
## Introduction

The Fractal Set provides a canonical lattice for gauge theory, and its Wilson loops are built from interaction triangles and plaquettes in the CST-IG-IA complex ({doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`). This chapter defines a **Voronoi-cell analog** of those loops using the data already stored in `RunHistory`: per-slice Delaunay neighbor edges and optional Voronoi region metadata. The intent is not to change the algorithm, but to add a parallel observable that lives entirely on the fitness-manifold tessellation described in {doc}`02_scutoid_spacetime`.

We take the emergent metric $g = H + \epsilon_\Sigma I$ ({prf:ref}`def-adaptive-diffusion-tensor-latent`) as the background geometry and use the **Voronoi/Delaunay complex** as the discrete substrate. Loops are then defined as closed cycles of neighbor edges (spacelike) and time-slice adjacencies (timelike), with edge phases derived from the same fitness/cloning/viscous quantities used in the Fractal Set.


(sec-voronoi-loop-data)=
## Data Available from Runs

The Voronoi-loop construction is restricted to quantities already present in `RunHistory`:

- **Positions and velocities**: $x_t$, $v_t$ from `x_before_clone`, `x_final`, `v_before_clone`, `v_final`.
- **Fitness and cloning channel**: `fitness`, `cloning_scores`, `cloning_probs`, `companions_distance`, `companions_clone`, `will_clone`.
- **Neighbor graph** (if `neighbor_graph_record=True`):
  - `neighbor_edges[t]`: Delaunay/Voronoi neighbor edges (directed) for slice $t$.
  - `voronoi_regions[t]`: Voronoi vertices/regions metadata (for cell volumes and facet areas).
- **Adaptive geometry**:
  - `fitness_hessians_diag/full` if anisotropic diffusion is enabled.
  - `sigma_reg_diag/full` (diffusion tensor) for geometric weighting.
  - `riemannian_volume_weights` for the fitness-manifold volume element (see {prf:ref}`def-adaptive-diffusion-tensor-latent`).

For spacetime loops, we additionally reuse the time-sliced Voronoi construction in
`voronoi_time_slices.py`, which returns **spacelike edges** (within a Euclidean-time bin) and
**timelike edges** (between adjacent bins) on the same Voronoi/Delaunay substrate.


(sec-voronoi-loop-links)=
## Voronoi Gauge Links (Edge Phases)

We define link variables on Delaunay edges and timelike edges using recorded fields. The following
choices mirror the Fractal Set constructions while staying within run data:

1. **U(1) fitness phase (default)**

For a neighbor edge $(i \rightarrow j)$ at time slice $t$:

$$
\theta_{ij}(t) := -\frac{V_{\mathrm{fit}}(j,t) - V_{\mathrm{fit}}(i,t)}{\hbar_{\mathrm{eff}}}
$$

$$
U_{ij}(t) := \exp(i\,\theta_{ij}(t))
$$

This is the Voronoi analog of the IG phase in the Fractal Set and uses only `fitness`.

2. **SU(2) cloning doublet phase (optional)**

Use cloning scores $S_i(j)$ and probabilities $P_i^{\mathrm{clone}}(j)$ (from
`cloning_scores`/`cloning_probs`) to define an $SU(2)$ doublet at each neighbor pair. The link
variable between pairs is then the $SU(2)$ transport between doublets at adjacent edges. This
matches the Fractal Set $SU(2)$ construction (see {doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`) but
is restricted to neighbors in the Voronoi graph.

3. **SU(N) viscous-force phase (optional)**

Use the viscous coupling vectors (from `force_viscous`) to build a complexified normalized vector
at each node, then define $SU(N)$ links as the change of basis between neighbor vectors. This is
the Voronoi analog of the Fractal Set $SU(N)$ construction, but with adjacency given by the
Delaunay graph.

**Note**: In the current implementation, Fractal Set phases $\phi_{\mathrm{CST}}$ and
$\phi_{\mathrm{IA}}$ are set to zero in code. The Voronoi construction can adopt the same default
(only $\theta_{ij}$ contributes) until explicit CST/IA phase conventions are promoted to a
certified rule.


(sec-voronoi-loop-families)=
## Loop Families (Voronoi Analog of Fractal-Set Loops)

The Fractal Set uses **interaction triangles** and **hourglass plaquettes**. We propose the
following Voronoi-cell analogs that can be computed directly from recorded data.

### 1) Spatial Delaunay Triangles (Voronoi interaction triangles)

At a fixed slice $t$, the Delaunay triangulation provides simplices whose **2-faces** are
triangles. For each oriented triangle $(i, j, k)$ we define the Wilson loop:

$$
W_{\triangle}(i,j,k) = \mathrm{Tr}\left(U_{ij} U_{jk} U_{ki}\right)
$$

For $U(1)$, this reduces to $\cos(\theta_{ij} + \theta_{jk} + \theta_{ki})$.

This is the pure-spatial Voronoi analog of the Fractal Set interaction triangle (IG + CST + IA),
but now built entirely inside a spatial slice using neighbor adjacency.

### 2) Time-Sliced Hourglass Plaquettes

Using `compute_time_sliced_voronoi`, each Euclidean-time bin supplies spacelike edges and the
adjacent bins supply timelike edges. For a pair of neighbors $(i,j)$ that appear in consecutive
bins, define the 4-cycle:

$$
(i_t \rightarrow j_t) \rightarrow (j_t \rightarrow j_{t+1}) \rightarrow (j_{t+1} \rightarrow i_{t+1}) \rightarrow (i_{t+1} \rightarrow i_t)
$$

This is the Voronoi analog of the hourglass plaquette (two interaction triangles glued along an
edge). The Wilson loop is the ordered product around this 4-cycle. For $U(1)$, it is the cosine
of the summed phases along the loop.

### 3) Spacetime Interaction Triangles (CST-IG-IA analog)

For each walker $i$, use the worldline edge $(i_t \rightarrow i_{t+1})$ as the CST analog and any
Delaunay neighbor $j$ at time $t$ as the IG analog. Define the IA analog as the influence edge
$(i_{t+1} \rightarrow j_t)$ when $j$ appears in the viscous or companion influence set for $i$.
The resulting triangle is:

$$
(i_t \rightarrow j_t) \rightarrow (j_t \rightarrow i_{t+1}) \rightarrow (i_{t+1} \rightarrow i_t)
$$

This matches the Fractal Set interaction-triangle template, but uses only Voronoi adjacency and
recorded influence weights.


(sec-voronoi-loop-plan)=
## Computation Plan (Voronoi Loop Pipeline)

The following plan computes the same loop families as the Fractal Set, but on the Voronoi graph
recorded in `RunHistory`:

1. **Select a recorded slice** $t$ (or a list of slices).
2. **Build adjacency**:
   - If `neighbor_edges[t]` exists, use it directly.
   - Otherwise, recompute Delaunay/Voronoi neighbors from positions (same logic as
     `compute_voronoi_tessellation`).
3. **Compute edge weights and lengths** (optional):
   - Euclidean length: $\|x_i - x_j\|$.
   - Geodesic edge length: $d_g(i,j)^2 = \Delta x^T g_{ij} \Delta x$, with
     $g_{ij} = (g_i + g_j)/2$ from the emergent metric ({prf:ref}`def-adaptive-diffusion-tensor-latent`).
4. **Assign gauge links**:
   - $U(1)$: use $\theta_{ij}$ from fitness differences (default).
   - $SU(2)$: use cloning-score doublets (optional).
   - $SU(N)$: use viscous-force phases (optional).
5. **Enumerate minimal loops**:
   - **Triangles** from Delaunay simplices (spatial loops).
   - **Hourglass 4-cycles** from time-sliced Voronoi (spacetime loops).
   - **Interaction triangles** via worldline + neighbor + influence edges.
6. **Compute Wilson statistics**:
   - For each loop, compute $W[\gamma]$ and action $1 - \mathrm{Re}\,W[\gamma]$.
   - Aggregate mean, variance, histograms, and time series (parallel to
     `_compute_wilson_loops` in `qft/analysis.py`).

This pipeline is linear in the number of Delaunay simplices and loop primitives per slice, which
is $O(N)$ in low dimensions under the same assumptions used in the scutoid tessellation.


(sec-voronoi-loop-weights)=
## Weighting, Boundary Handling, and Normalization

To avoid boundary artifacts, reuse the Voronoi boundary-tier classification in
`voronoi_observables.classify_boundary_cells` (Tier 1 excluded, Tier 2 used only as neighbors).
Loop contributions can be weighted by:

- **Facet areas** $A_{ij}$ (dual to Delaunay edges) when available from Voronoi metadata.
- **Dual cell volumes** $V_i$ as a normalization for node-based averages.
- **Geodesic edge length** $d_g(i,j)$ if the emergent metric is enabled.

Weights should be recorded alongside loop values so that weighted and unweighted Wilson statistics
can be compared directly.


(sec-voronoi-volume-element)=
## Fitness-Manifold Volume Element (Riemannian Weights)

The emergent metric is defined by the adaptive diffusion tensor
({prf:ref}`def-adaptive-diffusion-tensor-latent`). To weight Voronoi neighbors with respect to the
fitness manifold, we use the Riemannian volume element

$$
V_i^{(R)} := V_i^{(E)}\,\sqrt{\det g_i}.
$$

Using $g \propto (\Sigma \Sigma^T)^{-1}$ from the diffusion definition, the code implements the
runtime approximation

$$
\sqrt{\det g_i} \;\approx\; \frac{c_2^{d}}{\det \Sigma_i},
$$

with $\Sigma_i$ taken from `sigma_reg_diag/full` and $V_i^{(E)}$ from the Euclidean Voronoi cell.
This produces the recorded `riemannian_volume_weights`, which are used whenever scutoid neighbors
are weighted in analysis and operators on the Voronoi/Delaunay complex. The approximation uses only
quantities already defined in the framework and does not introduce new assumptions beyond
{prf:ref}`def-adaptive-diffusion-tensor-latent`.


(sec-voronoi-curvature-current)=
## Current Curvature and Geometry Observables (Code Summary)

The existing code computes curvature and geometric properties using three complementary routes:

1. **Scutoid/Regge curvature (deficit angles)**
   - `core/scutoids.py` builds the time-varying scutoid tessellation from `RunHistory` and computes
     Ricci scalars via deficit angles (Regge calculus). This is used in
     `qft/quantum_gravity.compute_regge_action`.

2. **Voronoi curvature proxies (fast O(N))**
   - `qft/voronoi_observables.compute_curvature_proxies` provides volume distortion, shape
     distortion, and Raychaudhuri expansion $\theta = (1/V)dV/dt$ as curvature proxies.
   - `qft/quantum_gravity.compute_einstein_hilbert_action` uses these proxies to approximate the
     Einstein-Hilbert action and Ricci scalar statistics.

3. **Emergent metric and geodesics**
   - The emergent metric $g = H + \epsilon_\Sigma I$ is defined in
     {prf:ref}`def-adaptive-diffusion-tensor-latent` and is realized in code through recorded
     Hessians or via `qft/higgs_observables.compute_emergent_metric` (neighbor covariance).
   - `qft/higgs_observables.compute_geodesic_distances` computes edge-wise geodesic lengths.

Additional geometry/QG observables built on these include spectral dimension, Hausdorff dimension,
causal structure via time-sliced Voronoi, Raychaudhuri expansion, and tidal/geodesic deviation
(`qft/quantum_gravity.py`).

The Voronoi Wilson loop construction proposed in this chapter is designed to be **compatible with
these existing observables** and to reuse the same neighbor graphs and metric proxies already
computed during analysis.
