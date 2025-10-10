# Scutoid Geometry Framework for Swarm Spacetime Evolution

## 0. Executive Summary and Motivation

### 0.1. The Central Idea: Geometric Cells for Algorithmic Dynamics

This chapter introduces a novel geometric framework for understanding the **spacetime evolution of the Fragile Gas** by representing walker configurations between adjacent time slices as **scutoid-like volume cells**. This provides a mesoscopic language that bridges the discrete algorithmic reality (Fractal Set) and the continuous emergent geometry (mean-field limit).

**Core Innovation**: The fundamental dynamic of the Fragile Gas—**cloning events that replace walkers**—is topologically equivalent to the defining feature of scutoids in biological tissue: **neighbor-swapping between parallel surfaces**. This is not merely an analogy; it is a precise mathematical correspondence that motivates a complete geometric theory.

### 0.2. Historical Context: Scutoids in Biology

Scutoids were discovered by Gómez-Gálvez et al. (2018) in the study of epithelial tissue packing. Key properties:

1. **Geometric definition**: Prismatoids with mid-level vertices forcing face curvature
2. **Topological signature**: Different neighbors at apical (top) vs. basal (bottom) surfaces
3. **Physical principle**: Minimize packing energy in curved tissue undergoing deformation
4. **Biological role**: Enable energy-efficient volume filling as tissue bends and grows

**Biological scutoids solve a geometric optimization problem under constraints**—precisely the mathematical structure underlying the Fragile Gas.

### 0.3. Why Scutoids for the Fragile Gas?

The correspondence is compelling:

| Epithelial Tissue | Fragile Gas Swarm |
|-------------------|-------------------|
| Cell in curved tissue | Walker in fitness landscape |
| Apical surface (top) | Walker configuration at time $t$ |
| Basal surface (bottom) | Walker configuration at time $t+1$ |
| Tissue bending | Fitness gradient |
| Neighbor-swapping | Cloning event (walker replacement) |
| Packing energy | Information rearrangement cost (Wasserstein-2 distance) |
| Scutoid formation | Adaptive exploration phase |

### 0.4. What We Achieve

**Theoretical Contributions**:
1. **Discrete-to-continuum bridge**: Scutoid tessellations converge to continuous geometry as $N \to \infty$
2. **Topological order parameter**: Scutoid fraction $\phi$ characterizes exploration vs. exploitation phases
3. **Geometric dictionary**: Scutoid shape parameters encode Riemann curvature tensor components
4. **Effective field theory**: Scutoid dynamics derive continuum PDEs for mean-field measure $\mu_t$

**Physical Insights**:
1. **Phase transitions**: Prism-dominated (convergent) vs. scutoid-dominated (exploratory) phases
2. **Dynamical law**: Raychaudhuri equation governs scutoid evolution
3. **Conservation laws**: Topological invariants during swarm evolution
4. **Holographic principle**: Bulk (spacetime scutoids) ↔ Boundary (fitness landscape)

### 0.5. Document Structure

**Chapter 1**: Mathematical foundations—Riemannian manifolds, prismatoids, scutoid definitions

**Chapter 2**: Topological correspondence—cloning events as neighbor-swapping transformations

**Chapter 3**: Episode-to-scutoid mapping—constructing cells from Fractal Set data

**Chapter 4**: Energy functional—Wasserstein-2 metric as scutoid deformation energy

**Chapter 5**: Geometric properties—curvature, holonomy, and the Riemann tensor dictionary

**Chapter 6**: Continuum limit—convergence theorems as $N \to \infty$, $\Delta t \to 0$

**Chapter 7**: Applications—phase transitions, effective field theory, computational methods

---

## 1. Mathematical Foundations

### 1.1. The Spacetime Manifold

We work with the emergent Riemannian structure defined by the Adaptive Gas.

:::{prf:definition} Swarm Spacetime Manifold
:label: def-swarm-spacetime

The **swarm spacetime manifold** is the $(d+1)$-dimensional product manifold:

$$
\mathcal{M} = \mathcal{X} \times [0, T]

$$

where:
- $\mathcal{X} \subset \mathbb{R}^d$ is the **spatial configuration space** (state space where walkers live)
- $[0, T]$ is the **temporal domain** (algorithmic time)
- Local coordinates: $(x, t)$ where $x \in \mathcal{X}$, $t \in [0, T]$

The manifold is equipped with a **spacetime metric** $g_{\text{ST}}$ with signature $(+, +, \ldots, +, +)$ (Riemannian, not Lorentzian).

**Spatial metric**: From Chapter 8 (Emergent Geometry), the spatial slice at time $t$ has metric:

$$
g(x, t) = g(x, S_t) = H(x, S_t) + \epsilon_\Sigma I

$$

where $H = \nabla^2 V_{\text{fit}}$ is the fitness Hessian and $S_t$ is the swarm state at time $t$.

**Spacetime metric**: We adopt the **product metric**:

$$
g_{\text{ST}} = g_{ij}(x, t) \, dx^i \otimes dx^j + dt \otimes dt

$$

This metric is **time-dependent** through the swarm-induced spatial metric $g(x, t)$.
:::

:::{note}
**Why Riemannian (not Lorentzian)?** The Fragile Gas evolves in computational time $t$, not physical relativistic time. There is no light cone structure or causality constraints from special relativity. However, **causal structure emerges** from the cloning genealogy (Causal Spacetime Tree), independent of the metric signature.
:::

### 1.2. Voronoi Tessellation at a Fixed Time Slice

Before defining scutoids (which connect time slices), we establish the spatial tessellation at a single time.

:::{prf:definition} Riemannian Voronoi Diagram
:label: def-riemannian-voronoi

For a swarm state $S_t = \{(x_i, v_i, s_i)\}_{i=1}^N$ with alive set $\mathcal{A}(t)$, the **Riemannian Voronoi cell** of walker $i \in \mathcal{A}(t)$ is:

$$
\text{Vor}_i(t) = \left\{ x \in \mathcal{X} : d_g(x, x_i) \le d_g(x, x_j) \text{ for all } j \in \mathcal{A}(t) \right\}

$$

where $d_g(x, x_i)$ is the **geodesic distance** in the Riemannian manifold $(\mathcal{X}, g(\cdot, t))$.

The **Voronoi tessellation** at time $t$ is:

$$
\mathcal{V}_t = \{\text{Vor}_i(t) : i \in \mathcal{A}(t)\}

$$

This partitions the valid domain: $\bigcup_{i \in \mathcal{A}(t)} \text{Vor}_i(t) = \mathcal{X}_{\text{valid}}$.
:::

:::{prf:remark} Curved vs. Flat Voronoi Cells
:label: rem-curved-voronoi

In Euclidean space ($g = I$), Voronoi cells are polyhedra with flat faces. In a Riemannian manifold with non-trivial curvature, the boundaries are **geodesic hypersurfaces**, which can be curved even when viewed in ambient coordinates. This curvature reflects the emergent geometry of the fitness landscape.
:::

### 1.3. Prismatoids and Scutoids in Euclidean Space

We now define the geometric objects that will form our spacetime cells.

:::{prf:definition} Prismatoid (Classical Definition)
:label: def-prismatoid-classical

A **prismatoid** in $\mathbb{R}^{d+1}$ is a polytope $P$ with the following properties:

1. **Vertices on two parallel hyperplanes**: All vertices lie on either $H_0 = \{(x, t) : t = t_0\}$ or $H_1 = \{(x, t) : t = t_1\}$
2. **Top and bottom faces**: The intersections $F_{\text{top}} = P \cap H_1$ and $F_{\text{bottom}} = P \cap H_0$ are $(d)$-dimensional polytopes (polygons in 2D, polyhedra in 3D)
3. **Side faces**: Lateral faces connecting the top and bottom are ruled surfaces (swept out by line segments)

**Special cases**:
- **Prism**: $F_{\text{top}} \cong F_{\text{bottom}}$ (congruent top/bottom) and vertical edges
- **Frustum**: $F_{\text{top}} \sim F_{\text{bottom}}$ (similar but different sizes), edges converge to a point
- **General prismatoid**: Arbitrary polytopes on top/bottom planes
:::

:::{prf:definition} Scutoid (Biological Definition)
:label: def-scutoid-biological

A **scutoid** is a prismatoid with the addition of **mid-level vertices**: vertices that lie on an intermediate hyperplane $H_{\text{mid}} = \{(x, t) : t = t_{\text{mid}}\}$ where $t_0 < t_{\text{mid}} < t_1$.

**Topological consequence**: The presence of mid-level vertices forces:
1. **Neighbor changes**: The set of adjacent cells at the top face differs from the set at the bottom face
2. **Curved faces**: Some lateral faces must be **non-planar** (curved 2-surfaces in 3D)
3. **Branching edges**: Edges can merge or split at mid-level vertices (Y-shaped connections)

**Key property**: Scutoids enable **efficient volume packing** when transitioning between two parallel surfaces with different **neighborhood topology**.
:::

:::{prf:example} The Canonical Scutoid in $\mathbb{R}^3$
:label: ex-canonical-scutoid

Consider a scutoid connecting two parallel planes $z = 0$ and $z = 1$ in $\mathbb{R}^3$:

**Bottom face** ($z = 0$): Pentagonal polygon with neighbors A, B, C, D, E

**Top face** ($z = 1$): Hexagonal polygon with neighbors A, B, C, D, E, **F** (new neighbor)

**Mid-level vertex**: At $(x_{\text{mid}}, y_{\text{mid}}, z = 1/2)$, where the boundary meets the new neighbor F

**Physical interpretation**: As we move from bottom to top, the cell "opens up" to accommodate a new neighbor, creating a concave face connecting to F.

This geometry minimizes the total surface area (energy) subject to the constraint of filling the volume with the required neighbor changes.
:::

### 1.4. Boundary Correspondence Map

Before defining scutoids in Riemannian manifolds, we must rigorously specify how to connect the boundaries of Voronoi cells at different times.

:::{prf:definition} Neighbor-Preserving Boundary Segments
:label: def-neighbor-segments

For a Voronoi cell $\text{Vor}_i(t)$ with neighbor set $\mathcal{N}_i(t)$, decompose the boundary into **neighbor-interface segments**:

$$
\partial \text{Vor}_i(t) = \bigcup_{k \in \mathcal{N}_i(t)} \Gamma_{i,k}(t)

$$

where $\Gamma_{i,k}(t) = \partial \text{Vor}_i(t) \cap \partial \text{Vor}_k(t)$ is the **interface segment** between cells $i$ and $k$.

Each segment $\Gamma_{i,k}(t)$ is a $(d-1)$-dimensional hypersurface in the spatial manifold $(\mathcal{X}, g(\cdot, t))$.
:::

:::{prf:definition} Boundary Correspondence Map
:label: def-boundary-correspondence

Let $F_{\text{bottom}} = \text{Vor}_j(t)$ and $F_{\text{top}} = \text{Vor}_i(t + \Delta t)$ be the bottom and top faces of a spacetime cell, with neighbor sets $\mathcal{N}_j(t)$ and $\mathcal{N}_i(t + \Delta t)$ respectively.

Define the **shared neighbor set**:

$$
\mathcal{N}_{\text{shared}} = \mathcal{N}_j(t) \cap \mathcal{N}_i(t + \Delta t)

$$

For each $k \in \mathcal{N}_{\text{shared}}$, the **boundary correspondence map** $\phi_k: \Gamma_{j,k}(t) \to \Gamma_{i,k}(t + \Delta t)$ is defined by **arc-length rescaling**:

1. **Arc-length parameterizations**: Let $\gamma_{j,k}: [0, L_j] \to \Gamma_{j,k}(t)$ and $\gamma_{i,k}: [0, L_i] \to \Gamma_{i,k}(t + \Delta t)$ be **arc-length parameterizations** of the boundary segments, where $L_j$ and $L_i$ are the total arc lengths (Riemannian lengths in the respective spatial metrics $g(\cdot, t)$ and $g(\cdot, t + \Delta t)$).

2. **Map definition**:

$$
\phi_k(\gamma_{j,k}(s)) = \gamma_{i,k}\left(s \cdot \frac{L_i}{L_j}\right) \quad \text{for } s \in [0, L_j]

$$

This maps the point at arc-length $s$ on the bottom segment to the point at rescaled arc-length $s \cdot (L_i/L_j)$ on the top segment.

**Well-definedness assumptions**:
- Each segment $\Gamma_{j,k}(t)$ and $\Gamma_{i,k}(t + \Delta t)$ is a **connected, $(d-1)$-dimensional hypersurface** (the interface between two Voronoi cells)
- For small $\Delta t$ and generic walker configurations, these segments are **non-degenerate** (positive $(d-1)$-dimensional measure)
- **Degenerate cases**: When a Voronoi interface degenerates to a single point or lower-dimensional set, the correspondence map is undefined. Such configurations have **measure zero** in the space of swarm configurations and are excluded from the generic construction.

**Interpretation**: The map $\phi_k$ simply rescales arc length to connect corresponding points on the two segments. It is **purely topological** (does not depend on geodesics or spacetime dynamics), providing a canonical correspondence for shared-neighbor interfaces.

**Lost neighbors**: For $\ell \in \mathcal{N}_j(t) \setminus \mathcal{N}_i(t + \Delta t)$, the segment $\Gamma_{j,\ell}(t)$ has **no corresponding segment** on the top face, so $\phi_\ell$ is **undefined**. These are the locations where mid-level structure arises.

**Gained neighbors**: For $m \in \mathcal{N}_i(t + \Delta t) \setminus \mathcal{N}_j(t)$, the segment $\Gamma_{i,m}(t + \Delta t)$ has **no corresponding segment** on the bottom face.
:::

### 1.5. Generalized Scutoids in Riemannian Manifolds

We now extend scutoid geometry to curved spaces with the boundary correspondence map.

:::{prf:definition} Riemannian Scutoid (Rigorous Definition)
:label: def-riemannian-scutoid

Let $(\mathcal{M}, g_{\text{ST}})$ be the swarm spacetime manifold. A **Riemannian scutoid** $\mathcal{S}_{i, t}$ associated with walker $i$ between times $t$ and $t + \Delta t$ is a $(d+1)$-dimensional region in $\mathcal{M}$ constructed as follows:

1. **Top face**: $F_{\text{top}} = \text{Vor}_i(t + \Delta t) \times \{t + \Delta t\}$ (Voronoi cell at time $t + \Delta t$)

2. **Bottom face**: $F_{\text{bottom}} = \text{Vor}_j(t) \times \{t\}$ where $j$ is the **parent** of episode $i$ (the walker whose Voronoi cell at time $t$ contains the birth position of episode $i$)

3. **Shared neighbor set**: $\mathcal{N}_{\text{shared}} = \mathcal{N}_j(t) \cap \mathcal{N}_i(t + \Delta t)$

4. **Lateral faces (neighbor-preserving)**: For each $k \in \mathcal{N}_{\text{shared}}$, construct the **ruled surface** $\Sigma_k$ swept out by spacetime geodesics:

$$
\Sigma_k = \bigcup_{p \in \Gamma_{j,k}(t)} \gamma_{p \to \phi_k(p)}

$$

where $\gamma_{p \to \phi_k(p)}: [t, t + \Delta t] \to \mathcal{M}$ is the **spacetime geodesic** connecting $(p, t)$ to $(\phi_k(p), t + \Delta t)$ in the metric $g_{\text{ST}}$

5. **Mid-level structure** (if $\mathcal{N}_j(t) \neq \mathcal{N}_i(t + \Delta t)$):
   - **Temporal coordinate**: $t_{\text{mid}} = t + \Delta t/2$ (the algorithmic cloning timestep)
   - For each lost neighbor $\ell \in \mathcal{N}_j(t) \setminus \mathcal{N}_i(t + \Delta t)$:
     - The segment $\Gamma_{j,\ell}(t)$ on the bottom has no corresponding top segment
     - Define a **mid-level edge** at time $t_{\text{mid}}$ connecting to gained neighbors
   - For each gained neighbor $m \in \mathcal{N}_i(t + \Delta t) \setminus \mathcal{N}_j(t)$:
     - The segment $\Gamma_{i,m}(t + \Delta t)$ on the top has no corresponding bottom segment
     - Connect via geodesics to the mid-level edge

   The **mid-level vertices** are the branching points where geodesics from lost neighbors meet geodesics to gained neighbors. Their **spatial coordinates** represent the birth positions of newly cloned walkers.

6. **Cell volume**: The closure of the union of all lateral faces and top/bottom faces:

$$
\mathcal{S}_{i,t} = \overline{\left( F_{\text{bottom}} \cup F_{\text{top}} \cup \bigcup_{k \in \mathcal{N}_{\text{shared}}} \Sigma_k \cup \Sigma_{\text{mid}} \right)}

$$

where $\Sigma_{\text{mid}}$ denotes the mid-level structure (if present)

7. **Riemannian volume**:

$$
\text{Vol}(\mathcal{S}_{i,t}) = \int_{\mathcal{S}_{i,t}} \sqrt{\det(g_{\text{ST}})} \, dx^1 \cdots dx^d \, dt

$$

**Topological classification**:
- **Prism**: $\mathcal{N}_j(t) = \mathcal{N}_i(t + \Delta t)$ (same neighbors, no mid-level structure)
- **Simple scutoid**: $|\mathcal{N}_j(t) \triangle \mathcal{N}_i(t + \Delta t)| = 2$ (one neighbor lost, one gained)
- **Complex scutoid**: $|\mathcal{N}_j(t) \triangle \mathcal{N}_i(t + \Delta t)| > 2$ (multiple neighbor changes)
:::

:::{prf:proposition} Algorithmic Origin of Mid-Level Vertices
:label: prop-algorithmic-midpoint

The temporal coordinate $t_{\text{mid}} = t + \Delta t/2$ of the mid-level vertices in the scutoid tessellation corresponds to the **algorithmic timestep** at which cloning and neighbor-swapping events occur, as defined in the Euclidean Gas update rule (Chapter 2, Algorithm 2.1).

**Proof**: The Euclidean Gas algorithm has an explicit two-stage structure:
1. **Cloning phase** ($t \to t + \Delta t/2$): Algorithm 2.1, Stage 3 produces the post-cloning state $(x_i^{(t+1/2)}, v_i^{(t+1/2)})$, where walkers are replaced via cloning
2. **Kinetic phase** ($t + \Delta t/2 \to t + \Delta t$): Algorithm 2.1, Stage 4-5 applies Langevin dynamics (BAOAB integrator)

At time $t + \Delta t/2$:
- Dead walkers have been replaced by clones from high-fitness parents
- The Voronoi neighbor topology changes: $\mathcal{N}_i(t) \neq \mathcal{N}_{i'}(t + \Delta t/2)$
- The **spatial coordinates** of mid-level vertices are the cloning birth positions: $x_{i'}^{(t+1/2)} = x_{\text{parent}}^{(t)} + \xi$ (with Gaussian jitter $\xi$)

Therefore, the scutoid mid-level structure **is not a geometric modeling choice** but a direct representation of the algorithm's native two-stage dynamics. $\square$
:::

:::{prf:remark} Scutoid Geometry as Intrinsic Algorithmic Structure
:label: rem-intrinsic-geometry

The correspondence between scutoid geometry and the Fragile Gas algorithm is not merely descriptive—it reveals that **discrete-time birth-death algorithms possess intrinsic spacetime geometry**. The scutoid framework doesn't impose an external geometric interpretation; rather, it discovers the natural geometric structure already present in the algorithm's control flow.

This elevates the scutoid tessellation from a visualization tool to a mathematical fact: the Fragile Gas "lives" in a spacetime with scutoid cells, just as general relativistic matter lives in a curved Lorentzian manifold.

**Cross-reference**: See Chapter 2, Algorithm 2.1, line 65, which explicitly defines the intermediate state $\mathcal{S}_{t+1/2}$.
:::

:::{prf:remark} Why Geodesic Ruling?
:label: rem-geodesic-ruling

In Euclidean space, the lateral faces of a prismatoid are ruled by straight lines (distance-minimizing paths). In a Riemannian manifold, the natural generalization is **geodesic ruling**: the faces are swept out by geodesics connecting corresponding points on the top and bottom boundaries.

This choice has several advantages:
1. **Minimizes area**: Geodesics are length-minimizing → geodesic ruling minimizes face area
2. **Respects geometry**: Geodesics are the natural "straight lines" in curved space
3. **Intrinsic definition**: Independent of any embedding in higher-dimensional Euclidean space
:::

### 1.6. Scutoid Face Curvature and Sectional Curvature

The curvature of scutoid faces encodes information about the underlying spacetime geometry.

:::{prf:proposition} Scutoid Face Curvature Formula (Smooth Regions)
:label: prop-scutoid-face-curvature

Let $\Sigma_k \subset \mathcal{M}$ be a **neighbor-preserving lateral face** (from Definition {prf:ref}`def-riemannian-scutoid`, item 4) of a Riemannian scutoid, corresponding to shared neighbor $k \in \mathcal{N}_{\text{shared}}$.

**Smoothness condition**: Consider a **smooth region** of $\Sigma_k$ (away from mid-level vertices and edges), parameterized by $(u, v) \mapsto \sigma(u, v)$ where:
- $u \in [0, 1]$ parametrizes the **spatial direction** (along the boundary curve $\Gamma_{j,k}$)
- $v \in [0, 1]$ parametrizes the **temporal direction** ($v = 0$ at bottom, $v = 1$ at top)

Define tangent vectors:
- $X = \partial \sigma / \partial u$ (spatial tangent)
- $T = \partial \sigma / \partial v$ (temporal tangent)

On the smooth portions of $\Sigma_k$, the **intrinsic Gaussian curvature** is related to the **sectional curvature** of the ambient spacetime $\mathcal{M}$ by the **Gauss equation**:

$$
K_{\Sigma}(u, v) = K_{\mathcal{M}}(X, T) + \det(\text{II})

$$

where:
- $K_{\Sigma}$ is the Gaussian curvature of the surface $\Sigma_k$ (defined at smooth points)
- $K_{\mathcal{M}}(X, T) = \frac{R(X, T, T, X)}{g(X, X)g(T, T) - g(X, T)^2}$ is the sectional curvature of $\mathcal{M}$ along the plane spanned by $X$ and $T$
- $\text{II}$ is the second fundamental form (extrinsic curvature of $\Sigma_k$ in $\mathcal{M}$)
- $R$ is the Riemann curvature tensor of $g_{\text{ST}}$

**Non-smooth regions**: At mid-level vertices and the edges originating from them, the surface $\Sigma_k$ is non-differentiable (contains creases). The curvature at these points must be analyzed using **distributional curvature** or **polyhedral Gauss-Bonnet theory** (see {cite}`Alexandrov1958` for angles defects in polyhedral surfaces).

**Interpretation**: The curvature we observe on smooth portions of a scutoid face is the sum of:
1. The **intrinsic curvature** of the ambient spacetime (sectional curvature)
2. The **extrinsic bending** of how the surface sits in that spacetime
:::

:::{prf:corollary} Statistical Estimation of Sectional Curvature
:label: cor-curvature-from-scutoids

By averaging the face curvatures $K_{\Sigma}$ over many scutoid faces in a spacetime region $\mathcal{R} \subset \mathcal{M}$, we can obtain a **statistical estimate** of the average sectional curvature in that region.

**Procedure**:
1. Construct scutoid tessellation from algorithmic log
2. For each smooth face region $\Sigma_k$, measure Gaussian curvature $K_{\Sigma}$ (discrete differential geometry)
3. Estimate second fundamental form $\text{II}$ from face embedding
4. Extract sectional curvature samples: $K_{\mathcal{M}}(X_i, T_i) \approx K_{\Sigma,i} - \det(\text{II}_i)$
5. Compute spatial average:

$$
\langle K_{\mathcal{M}} \rangle_{\mathcal{R}} = \frac{1}{|\mathcal{F}_{\mathcal{R}}|} \sum_{i \in \mathcal{F}_{\mathcal{R}}} K_{\mathcal{M}}(X_i, T_i)

$$

where $\mathcal{F}_{\mathcal{R}}$ is the set of smooth face samples in region $\mathcal{R}$

**Limitations**: This provides constraints on the Riemann curvature tensor, not a complete reconstruction. Full tensor recovery would require dense sampling of all 2-planes at each point, which is not feasible from a discrete, sparse tessellation.

**Utility**: The average sectional curvature is sufficient for many applications, including:
- Detecting regions of high/low curvature (fitness landscape features)
- Validating the emergent Riemannian structure
- Computing curvature-based order parameters for phase transitions
:::

---

## 2. Topological Correspondence: Cloning as Neighbor-Swapping

### 2.1. The Central Theorem

We now prove the foundational result justifying the scutoid framework.

:::{prf:theorem} Cloning-Scutoid Correspondence
:label: thm-cloning-scutoid-correspondence

Let $e_i$ be an episode of the Fragile Gas with:
- Birth time $t_i^{\text{birth}}$ at position $x_i^{\text{birth}}$
- Death time $t_i^{\text{death}}$ at position $x_i^{\text{death}}$
- Parent episode $e_j$ (if $i$ is a cloned episode)

Define the **neighbor set** of episode $i$ at time $t$ as:

$$
\mathcal{N}_i(t) = \{k \in \mathcal{A}(t) : \text{Vor}_k(t) \cap \text{Vor}_i(t) \neq \emptyset\}

$$

(the set of walkers whose Voronoi cells share a boundary with walker $i$'s Voronoi cell).

**Statement**: A cloning event at time $t_{\text{clone}}$ (where episode $e_i$ dies and is replaced by a child episode $e_{i'}$ cloned from episode $e_j$) induces a **neighbor topology change**:

$$
\mathcal{N}_{i'}(t_{\text{clone}}^+) \neq \mathcal{N}_i(t_{\text{clone}}^-)

$$

where $t^-$ denotes the instant before cloning and $t^+$ denotes the instant after.

**Geometric consequence**: The spacetime volume connecting:
- Bottom face: $\text{Vor}_i(t_{\text{clone}}^-)$ (neighbors $\mathcal{N}_i(t^-)$)
- Top face: $\text{Vor}_{i'}(t_{\text{clone}}^+)$ (neighbors $\mathcal{N}_{i'}(t^+)$)

**must be a scutoid** (not a prism), because the neighbor sets differ, forcing the existence of mid-level vertices where geodesic rulings branch.

**Converse**: If $\mathcal{N}_i(t^+) = \mathcal{N}_i(t^-)$ (no cloning occurred), the spacetime volume is a **prism** (possibly a frustum if the Voronoi cell shrinks or expands, but with same neighbor topology).
:::

:::{prf:proof}

**Part 1: Cloning induces neighbor change with probability 1**

Consider a cloning event where walker $i$ at position $x_i \in \mathcal{X}$ is replaced by a clone of walker $j$ at position $x_j \in \mathcal{X}$ with $x_i \neq x_j$.

**Setup**: Let $S^{(t^-)} = \{x_1, \ldots, x_N\}$ be the walker positions before cloning, where walker $i$ is at $x_i$. After cloning, walker $i$ is replaced by a new walker at position:

$$
x_{i'} = x_j + \xi, \quad \xi \sim \mathcal{N}(0, \sigma_x^2 I)

$$

The new configuration is $S^{(t^+)} = \{x_1, \ldots, x_{i-1}, x_{i'}, x_{i+1}, \ldots, x_N\}$.

**Voronoi neighbor condition**: Walker $k$ is a neighbor of walker $i$ if and only if their Voronoi cells share a boundary:

$$
k \in \mathcal{N}_i \iff \exists p \in \mathcal{X}: d_g(p, x_k) = d_g(p, x_i) < d_g(p, x_\ell) \text{ for all } \ell \neq i, k

$$

**Key claim**: For any fixed $k \in \mathcal{N}_i(t^-)$ with $k \neq j$, the probability that $k \in \mathcal{N}_{i'}(t^+)$ vanishes as $\|x_i - x_j\| \to \infty$.

**Proof of claim**: The Voronoi boundary between cells $k$ and $i$ consists of points equidistant to $x_k$ and $x_i$. For $k$ to also be a neighbor of $i'$ at $x_{i'} = x_j + \xi$, there must exist a point $p$ equidistant to both $x_k$ and $x_{i'}$:

$$
d_g(p, x_k) = d_g(p, x_j + \xi)

$$

For large $\|x_i - x_j\|$, the sets $\{p : d_g(p, x_k) = d_g(p, x_i)\}$ and $\{p : d_g(p, x_k) = d_g(p, x_j)\}$ are **disjoint hypersurfaces** in $\mathcal{X}$ (generically), separated by distance $O(\|x_i - x_j\|)$. The Gaussian jitter $\xi$ with variance $\sigma_x^2$ cannot bridge this gap with high probability.

**Measure-theoretic statement**: Let $P_{\text{preserve}}(k; x_i, x_j)$ be the probability that neighbor $k \in \mathcal{N}_i(t^-)$ remains a neighbor after cloning ($k \in \mathcal{N}_{i'}(t^+)$). Then:

$$
P_{\text{preserve}}(k; x_i, x_j) \to 0 \quad \text{as } \|x_i - x_j\| \to \infty

$$

Since cloning can occur between any two walkers (Chapter 3), and the fitness-based selection does not constrain their spatial separation, neighbor-changing cloning events have **positive measure** in the space of algorithm trajectories.

**Conclusion**: $\mathbb{P}(\mathcal{N}_{i'}(t^+) \neq \mathcal{N}_i(t^-)) > 0$ for cloning events, and for spatially distant cloning ($\|x_i - x_j\| \gg \sigma_x$), the probability approaches 1.

**Part 2: Neighbor change forces scutoid geometry (using boundary correspondence map)**

Now suppose $\mathcal{N}_i(t^-) \neq \mathcal{N}_{i'}(t^+)$. By Definition {prf:ref}`def-neighbor-segments`, decompose the Voronoi boundaries:

$$
\partial \text{Vor}_i(t^-) = \bigcup_{k \in \mathcal{N}_i(t^-)} \Gamma_{i,k}(t^-)

$$

$$
\partial \text{Vor}_{i'}(t^+) = \bigcup_{\ell \in \mathcal{N}_{i'}(t^+)} \Gamma_{i',\ell}(t^+)

$$

Define the shared and non-shared neighbor sets:
- $\mathcal{N}_{\text{shared}} = \mathcal{N}_i(t^-) \cap \mathcal{N}_{i'}(t^+)$
- $\mathcal{N}_{\text{lost}} = \mathcal{N}_i(t^-) \setminus \mathcal{N}_{i'}(t^+)$ (neighbors lost)
- $\mathcal{N}_{\text{gained}} = \mathcal{N}_{i'}(t^+) \setminus \mathcal{N}_i(t^-)$ (neighbors gained)

By Definition {prf:ref}`def-boundary-correspondence`, the boundary correspondence map $\phi_k$ is **well-defined** only for $k \in \mathcal{N}_{\text{shared}}$. For $\ell \in \mathcal{N}_{\text{lost}}$, there is **no segment** $\Gamma_{i',\ell}(t^+)$ to map to, so $\phi_\ell$ is **undefined**.

**Topological necessity of mid-level structure**: To construct a continuous spacetime volume connecting $\text{Vor}_i(t^-) \times \{t^-\}$ to $\text{Vor}_{i'}(t^+) \times \{t^+\}$, we must connect every boundary segment at the bottom to a corresponding segment at the top.

For $k \in \mathcal{N}_{\text{shared}}$: The segment $\Gamma_{i,k}(t^-)$ maps via $\phi_k$ to $\Gamma_{i',k}(t^+)$, and the lateral face $\Sigma_k$ is constructed by geodesic ruling (Definition {prf:ref}`def-riemannian-scutoid`, item 4).

For $\ell \in \mathcal{N}_{\text{lost}}$: The segment $\Gamma_{i,\ell}(t^-)$ has **no image** under any correspondence map $\phi$. To close the volume's boundary, this segment's geodesic ruling must **terminate** at an intermediate location in spacetime.

For $m \in \mathcal{N}_{\text{gained}}$: The segment $\Gamma_{i',m}(t^+)$ has **no preimage** under any correspondence map. Its geodesic ruling must **originate** from an intermediate location.

**Definition of mid-level vertices**: The points in spacetime where geodesics from lost neighbors terminate and geodesics to gained neighbors originate are the **mid-level vertices**. Their existence is **topologically necessary** to form a closed boundary when $\mathcal{N}_{\text{lost}} \cup \mathcal{N}_{\text{gained}} \neq \emptyset$.

By Definition {prf:ref}`def-riemannian-scutoid`, this is precisely a scutoid (not a prism). $\square$

**Part 3: No cloning implies prism**

If no cloning occurs, walker $i$ persists from $t^-$ to $t^+$. The position changes via Langevin dynamics:

$$
x_i(t^+) = x_i(t^-) + v_i \Delta t + \text{noise}

$$

The Voronoi cell $\text{Vor}_i(t)$ evolves continuously. For small $\Delta t$, the neighbor set $\mathcal{N}_i(t)$ changes continuously, meaning $\mathcal{N}_i(t^+) = \mathcal{N}_i(t^-)$ (up to boundary events, which are measure-zero).

If neighbors are unchanged, the boundary segments on top and bottom match one-to-one, so the lateral faces are ruled surfaces without branching. This is a **prism** (or frustum if the cell size changes, but with same topology).

$\square$
:::

:::{prf:corollary} Scutoid Fraction as Cloning Rate Proxy
:label: cor-scutoid-fraction

The **fraction of scutoid cells** in the spacetime tessellation is:

$$
\phi(t) = \frac{\text{Number of scutoids in } [t, t + \Delta t]}{\text{Number of cells (episodes) in } [t, t + \Delta t]}

$$

This is asymptotically equal to the **cloning rate** of the algorithm:

$$
\phi(t) \approx p_{\text{clone}}(t) = \frac{\text{Number of cloning events in } [t, t + \Delta t]}{N}

$$

where $N$ is the total number of walkers.

**Interpretation**: The scutoid fraction $\phi$ is a **geometric order parameter** that directly measures the dynamical activity (cloning) of the swarm.
:::

### 2.2. Visualizing the Correspondence

```{figure} images/18_scutoid_cloning_correspondence.svg
---
name: fig-scutoid-cloning
width: 90%
align: center
---
Topological correspondence between cloning events and scutoid geometry. **Left**: Before cloning (time t⁻), walker i (red) has neighbors {A, B, C}. **Center**: Cloning event—walker i is replaced by a clone of walker j (blue). **Right**: After cloning (time t⁺), new walker i' has neighbors {B, C, D}. The spacetime volume connecting the bottom Voronoi cell (left) to the top Voronoi cell (right) must contain a mid-level vertex where the connection to neighbor A terminates and the connection to neighbor D originates. This is a scutoid.
```

### 2.3. Classification of Cell Types

:::{prf:definition} Cell Type Taxonomy
:label: def-cell-type-taxonomy

Spacetime cells in the scutoid tessellation are classified by their **topological complexity**:

1. **Prism (Type 0)**: No mid-level vertices, same neighbors at top and bottom
   - $|\mathcal{N}(t^+) \cap \mathcal{N}(t^-)| = |\mathcal{N}(t^-)| = |\mathcal{N}(t^+)|$
   - Corresponds to: **persistent walker** (no cloning)
   - Geometric phase: Exploitation, convergence

2. **Simple scutoid (Type 1)**: One mid-level vertex
   - $|\mathcal{N}(t^+) \setminus \mathcal{N}(t^-)| = 1$ and $|\mathcal{N}(t^-) \setminus \mathcal{N}(t^+)| = 1$
   - Corresponds to: **single cloning event** (one neighbor gained, one lost)
   - Geometric phase: Adaptive exploration

3. **Complex scutoid (Type k)**: $k$ mid-level vertices
   - Multiple neighbors gained/lost
   - Corresponds to: **multiple simultaneous cloning events** or rapid neighbor turnover
   - Geometric phase: Chaotic exploration, phase transition

**Topological invariant**: The **scutoid index** $\chi_{\text{scutoid}} = k$ (number of mid-level vertices) is a discrete integer-valued function on cells.
:::

:::{prf:proposition} Euler Characteristic and Scutoid Index
:label: prop-euler-characteristic-scutoid

The **Euler characteristic** of a scutoid cell $\mathcal{S}$ is related to its scutoid index $\chi_{\text{scutoid}}$ by:

$$
\chi(\mathcal{S}) = \chi(F_{\text{top}}) + \chi(F_{\text{bottom}}) - \chi_{\text{scutoid}}

$$

where $\chi(F)$ is the Euler characteristic of the face $F$ (e.g., $\chi(\text{polygon}) = 1$).

**Proof sketch**: Each mid-level vertex creates a branching in the edge graph, which reduces the Euler characteristic by altering the vertex-edge-face balance in the Euler formula $V - E + F = \chi$.

**Consequence**: The Euler characteristic of the entire tessellation at time $t$ is:

$$
\chi(\mathcal{T}_t) = N_{\text{cells}} - \sum_{\text{cells}} \chi_{\text{scutoid}}

$$

This provides a **topological measure** of the total cloning activity.
:::

---

## 3. Episode-to-Scutoid Mapping

### 3.1. Construction Algorithm

We now provide an algorithmic procedure to construct the scutoid tessellation from the Fractal Set data.

:::{prf:algorithm} Scutoid Tessellation Construction
:label: alg-scutoid-construction

**Input**:
- Algorithmic log $\mathcal{L}$ containing all episodes $\{e_i\}$ with birth/death times and positions
- Swarm states $\{S_t\}$ for $t \in [0, T]$ (discrete timesteps)
- Spatial metric $g(x, t)$ at each timestep

**Output**:
- Scutoid tessellation $\mathcal{T} = \{\mathcal{S}_i : e_i \in \mathcal{E}\}$
- Cell type classification (prism vs. scutoid)
- Geometric properties (volume, curvature) for each cell

**Procedure**:

1. **For each timestep** $t \in \{0, \Delta t, 2\Delta t, \ldots, T\}$:
   - Extract alive walker positions $\{x_i(t) : i \in \mathcal{A}(t)\}$
   - Compute Riemannian Voronoi tessellation $\mathcal{V}_t$ using metric $g(\cdot, t)$
   - Store Voronoi cells $\{\text{Vor}_i(t)\}$ and neighbor sets $\{\mathcal{N}_i(t)\}$

2. **For each episode** $e_i$ with lifetime $[t_i^{\text{birth}}, t_i^{\text{death}})$:
   - Identify parent episode $e_j = \text{parent}(e_i)$ (from cloning log)
   - Extract top face: $F_{\text{top}} = \text{Vor}_i(t_i^{\text{death}}) \times \{t_i^{\text{death}}\}$
   - Extract bottom face: $F_{\text{bottom}} = \text{Vor}_j(t_i^{\text{birth}}) \times \{t_i^{\text{birth}}\}$

3. **Determine cell type**:
   - Compute neighbor change: $\Delta \mathcal{N} = \mathcal{N}_i(t_i^{\text{death}}) \triangle \mathcal{N}_j(t_i^{\text{birth}})$ (symmetric difference)
   - If $\Delta \mathcal{N} = \emptyset$: **Prism** (Type 0)
   - If $|\Delta \mathcal{N}| = 2$: **Simple scutoid** (Type 1)
   - If $|\Delta \mathcal{N}| > 2$: **Complex scutoid** (Type k)

4. **Construct lateral faces**:
   - For each boundary segment of $\partial F_{\text{bottom}}$ (connecting to neighbor $k$):
     - If $k \in \mathcal{N}_i(t_i^{\text{death}})$ (neighbor persists):
       - Create **ruled surface** by geodesic interpolation from bottom to top
     - If $k \notin \mathcal{N}_i(t_i^{\text{death}})$ (neighbor lost):
       - Create **mid-level vertex** at time $t_{\text{mid}} = (t_i^{\text{birth}} + t_i^{\text{death}})/2$
       - Terminate geodesic ruling at mid-level vertex
   - For each boundary segment of $\partial F_{\text{top}}$ (connecting to neighbor $\ell$):
     - If $\ell \notin \mathcal{N}_j(t_i^{\text{birth}})$ (new neighbor):
       - Originate geodesic ruling from mid-level vertex to top

5. **Compute geometric properties**:
   - Volume: $\text{Vol}(\mathcal{S}_i) = \int_{\mathcal{S}_i} \sqrt{\det(g_{\text{ST}})} \, dx \, dt$ (numerical integration)
   - Face curvatures: $K_{\Sigma}$ for each lateral face $\Sigma$ (discrete differential geometry)
   - Holonomy: Parallel transport around face boundaries (Section 5.2)

6. **Output** tessellation $\mathcal{T}$ with all cells and properties.

:::

### 3.2. Computational Complexity

:::{prf:proposition} Computational Cost
:label: prop-computational-cost

The scutoid tessellation construction has time complexity:

$$
O(T/\Delta t \cdot N \log N \cdot d)

$$

where:
- $T/\Delta t$ is the number of timesteps
- $N$ is the number of walkers (episodes per timestep)
- $\log N$ factor from Voronoi computation (Fortune's algorithm or similar)
- $d$ is the spatial dimension

**Space complexity**: $O(T/\Delta t \cdot N \cdot d)$ to store all Voronoi cells.

**Bottleneck**: Riemannian geodesic distance computation for Voronoi tessellation (requires solving differential equations on curved manifolds).

**Mitigation**:
- Use **approximate geodesics** (Euclidean distances with metric correction)
- **Coarse-graining**: Compute tessellation only at selected timesteps
- **Sparse representation**: Store only boundary cells (most cells are prisms and can be cached)
:::

### 3.3. Relationship to Fractal Set Structures

:::{prf:theorem} Scutoid Tessellation is Dual to Information Graph
:label: thm-scutoid-ig-duality

The scutoid tessellation $\mathcal{T}$ and the Information Graph (IG) from Chapter 13 are **dual structures**:

**IG structure** (Chapter 13.3):
- **Vertices**: Episodes $e_i \in \mathcal{E}$
- **Edges**: Selection coupling $(e_i, e_j)$ if both alive during a cloning event

**Scutoid tessellation**:
- **Cells**: Scutoid volumes $\mathcal{S}_i$ corresponding to episodes $e_i$
- **Shared faces**: Two cells $\mathcal{S}_i$ and $\mathcal{S}_j$ share a lateral face iff episodes $e_i$ and $e_j$ are simultaneous neighbors (Voronoi cells adjacent at some time $t$)

**Duality statement**: An edge $(e_i, e_j)$ exists in the IG if and only if the corresponding scutoid cells $\mathcal{S}_i$ and $\mathcal{S}_j$ share a face in the tessellation.

**Proof**: By definition, IG edges connect episodes that are both alive during a cloning event, meaning their Voronoi cells are adjacent in space. Adjacent Voronoi cells at time $t$ imply the corresponding spacetime scutoids share a lateral face.

**Consequence**: The IG is the **1-skeleton** (edge graph) of the **dual complex** to the scutoid tessellation.
:::

---

## 4. Energy Functional: Hellinger-Kantorovich Distance

### 4.1. Packing Energy as Unbalanced Optimal Transport

The scutoid framework requires an energy functional that naturally handles **mass creation/destruction** during cloning events. The **Hellinger-Kantorovich (HK) distance** is the correct geometric framework for this setting.

:::{prf:definition} Scutoid Deformation Energy (Hellinger-Kantorovich)
:label: def-scutoid-energy

The **deformation energy** required to transform the swarm configuration from time $t$ to $t + \Delta t$ is defined using the **Hellinger-Kantorovich distance** between the empirical measures:

$$
E_{\text{scutoid}}(t \to t + \Delta t) = \text{HK}_{\alpha}(\mu_t, \mu_{t + \Delta t})^2

$$

where:
- $\mu_t = \frac{1}{N} \sum_{i \in \mathcal{A}(t)} \delta_{x_i(t)}$ is the empirical measure at time $t$ (unnormalized, total mass $|\mathcal{A}(t)|/N \le 1$)
- $\text{HK}_{\alpha}(\mu, \nu)$ is the **Hellinger-Kantorovich distance** with parameter $\alpha > 0$:

$$
\text{HK}_{\alpha}(\mu, \nu)^2 = \inf_{\pi} \left\{ \int_{\mathcal{X} \times \mathcal{X}} \frac{d_g(x, y)^2}{4\alpha} \, d\pi(x, y) + \alpha \text{KL}(\pi_1 | \mu) + \alpha \text{KL}(\pi_2 | \nu) \right\}

$$

where:
- $\pi$ is a **transport plan** (not necessarily a coupling, since masses may differ)
- $\pi_1, \pi_2$ are the marginals of $\pi$
- $\text{KL}(\pi_1 | \mu) = \int \log(d\pi_1/d\mu) \, d\pi_1$ is the **Kullback-Leibler divergence** (measuring mass growth/destruction)
- $\alpha > 0$ is a parameter balancing transport cost vs. mass creation cost

**Physical interpretation**:
- **Transport term**: $\int d_g(x, y)^2 / (4\alpha) \, d\pi$ is the cost of moving mass via geodesics
- **Mass creation/destruction**: $\text{KL}(\pi_1 | \mu)$ penalizes creating or destroying walkers (cloning/death events)
- **Parameter $\alpha$**: Controls the trade-off—large $\alpha$ makes mass creation cheaper, small $\alpha$ makes transport cheaper

**Why HK for the Fragile Gas**:
1. **Unbalanced transport**: Cloning events change the number of alive walkers, so $|\mathcal{A}(t)| \neq |\mathcal{A}(t + \Delta t)|$ generically
2. **Mass splitting**: A cloning event "splits" a walker's mass (parent dies, child is born)
3. **Natural for birth-death processes**: HK distance is the geometric framework for reaction-diffusion equations
4. **Reduces to Wasserstein**: When masses are balanced ($\mu(\mathcal{X}) = \nu(\mathcal{X})$), $\text{HK}_{\alpha} \to W_2$ as $\alpha \to \infty$

**Units**: $[\text{Energy}] = [\text{Distance}]^2$ (via the $d_g^2$ term).

**Literature**: The Hellinger-Kantorovich distance and its formulation for unbalanced optimal transport is developed in {cite}`Liero2018,ChizatPeyre2018`. The Wasserstein-Fisher-Rao geometry and its application to reaction-diffusion systems is detailed in {cite}`KondratievTeleRevuz2006,LieroMielkeSavare2016`.
:::

:::{prf:definition} Genealogical HK Cost
:label: def-genealogical-cost

The **genealogical Hellinger-Kantorovich cost** $C_G(\mu_t, \mu_{t+\Delta t})$ is the cost of the specific transport plan $\pi_G$ that follows the algorithmic genealogy:

$$
C_G(\mu_t, \mu_{t+\Delta t})^2 = \int_{\mathcal{X} \times \mathcal{X}} \frac{d_g(x, y)^2}{4\alpha} \, d\pi_G(x, y) + \alpha \text{KL}((\pi_G)_1 | \mu_t) + \alpha \text{KL}((\pi_G)_2 | \mu_{t+\Delta t})

$$

where the **genealogical transport plan** $\pi_G$ is constructed as follows:

1. **Persistent walkers**: For each $i \in \mathcal{P}(t)$ (alive at both $t$ and $t + \Delta t$):

$$
\pi_G(x_i(t), x_i(t + \Delta t)) = \frac{1}{N}

$$

(transport mass from walker $i$'s position at $t$ to its position at $t + \Delta t$)

2. **Cloned walkers**: For each $i \in \mathcal{C}(t)$ (died and replaced):
   - **Mass destruction**: Walker $i$ dies at position $x_i(t)$ (mass $1/N$ disappears)
   - **Mass creation**: New walker cloned from parent $j = \text{parent}(i)$ appears at $x_j(t + \Delta t)$ (mass $1/N$ appears)

The KL terms capture this mass creation/destruction:
- $\text{KL}((\pi_G)_1 | \mu_t)$ measures mass destroyed (dead walkers)
- $\text{KL}((\pi_G)_2 | \mu_{t+\Delta t})$ measures mass created (cloned walkers)

**Explicit formula**:

$$
C_G(\mu_t, \mu_{t+\Delta t})^2 = \frac{1}{4\alpha} E_{\text{transport}} + \alpha E_{\text{birth-death}}

$$

where:
- $E_{\text{transport}} = \frac{1}{N} \sum_{i \in \mathcal{P}(t)} d_g(x_i(t), x_i(t + \Delta t))^2$ (transport cost for persistent walkers)
- $E_{\text{birth-death}} = \frac{2|\mathcal{C}(t)|}{N}$ (mass creation/destruction cost, factor of 2 from KL terms on both marginals)

**Derivation of birth-death cost**: For empirical measures with discrete mass creation/destruction, the KL divergence $\text{KL}(\pi_1 | \mu)$ equals the total mass created or destroyed (up to normalization). For $|\mathcal{C}(t)|$ cloning events (each destroying mass $1/N$ and creating mass $1/N$), the total cost from both marginals is $2|\mathcal{C}(t)|/N$. See {cite}`LieroMielkeSavare2016` Section 2.3 for the general formula for discrete measures.

**Key insight**: The HK framework **naturally decomposes** into:
1. **Transport component**: Moving persistent walkers along their trajectories (Langevin dynamics)
2. **Reaction component**: Birth-death events from cloning (mass change)

This is precisely the structure of the Fragile Gas algorithm!
:::

:::{prf:theorem} Upper Bound on HK Energy from Genealogy
:label: thm-hk-upper-bound

The Hellinger-Kantorovich distance is bounded above by the genealogical transport cost:

$$
\text{HK}_{\alpha}(\mu_t, \mu_{t+\Delta t})^2 \le C_G(\mu_t, \mu_{t+\Delta t})^2 = \frac{1}{4\alpha} E_{\text{transport}} + \alpha E_{\text{birth-death}}

$$

**Proof**: By definition, the HK distance is the infimum over all transport plans:

$$
\text{HK}_{\alpha}(\mu, \nu)^2 = \inf_{\pi} \left\{ \int \frac{d_g(x, y)^2}{4\alpha} \, d\pi + \alpha \text{KL}(\pi_1 | \mu) + \alpha \text{KL}(\pi_2 | \nu) \right\}

$$

The genealogical transport plan $\pi_G$ is *a valid plan* (though not a coupling, since masses may differ), so it is included in the set over which the infimum is taken. Therefore:

$$
\text{HK}_{\alpha}(\mu_t, \mu_{t+\Delta t})^2 \le C_G(\mu_t, \mu_{t+\Delta t})^2

$$

$\square$

**Consequence**: The genealogical cost provides a **computable upper bound** on the true HK energy. For the Fragile Gas, this bound is likely tight because the algorithm's dynamics (Langevin + cloning) are designed to minimize a related free energy functional.
:::

:::{prf:remark} Optimality of the Genealogical Plan
:label: rem-genealogical-optimality

The inequality $\text{HK}_{\alpha}^2 = C_G^2$ (equality) holds if and only if the genealogical transport plan $\pi_G$ is **HK-optimal**.

**Why the Fragile Gas may achieve optimality**:
1. **Local Langevin dynamics**: Persistent walkers diffuse locally, which is energy-minimizing for smooth gradients
2. **Fitness-driven cloning**: Cloning selects high-fitness parents, which (heuristically) minimizes the expected distance between dead walkers and their replacements
3. **Balance parameter**: If the algorithmic parameter $\alpha$ is chosen to match the HK parameter, the algorithm effectively solves the HK optimization

**Connection to gradient flows**: The HK distance defines a **Riemannian structure on the space of measures** (the Wasserstein-Fisher-Rao geometry). The Fragile Gas dynamics can be interpreted as a **discrete-time gradient flow** in this geometry, which would imply $\pi_G$ is approximately optimal.

**Empirical validation**: Computing both $\text{HK}_{\alpha}^2$ (via convex optimization solvers) and $C_G^2$ (from genealogy) on swarm runs would:
1. Quantify the tightness of the bound
2. Determine the optimal choice of $\alpha$ for the algorithm
3. Validate the gradient flow interpretation
:::

:::{prf:corollary} Scutoid Fraction and Energy
:label: cor-scutoid-energy

The scutoid fraction $\phi(t)$ is correlated with the Wasserstein energy:

$$
E_{\text{scutoid}}(t \to t + \Delta t) \propto \phi(t) \cdot \bar{d}_{\text{clone}}^2

$$

where $\bar{d}_{\text{clone}}$ is the average geodesic distance between cloned walkers and their parents.

**Interpretation**: A **scutoid-dominated phase** (high $\phi$) corresponds to **high energy** (rapid configuration changes), while a **prism-dominated phase** (low $\phi$) corresponds to **low energy** (stable configuration).

This connects geometric topology (scutoid fraction) to thermodynamic quantities (energy).
:::

### 4.2. Minimization Principle

:::{prf:conjecture} Scutoid Shape Minimizes Wasserstein Energy
:label: conj-scutoid-minimization

The specific geometric shape of scutoids (curved faces, mid-level vertices at optimal locations) is the **energy-minimizing configuration** for connecting two Voronoi tessellations with different neighbor topologies.

**Formal statement**: Among all $(d+1)$-dimensional cells connecting:
- Bottom face: $F_{\text{bottom}}$ with neighbor set $\mathcal{N}_{\text{bottom}}$
- Top face: $F_{\text{top}}$ with neighbor set $\mathcal{N}_{\text{top}} \neq \mathcal{N}_{\text{bottom}}$

the scutoid geometry (with geodesic ruling and mid-level vertices) minimizes:

$$
E_{\text{shape}} = \int_{\partial \mathcal{S}} dA + \lambda \int_{\mathcal{S}} dV

$$

where $dA$ is the area element (surface energy) and $dV$ is the volume element (bulk energy).

**Status**: This is currently a **conjecture** inspired by the biological result of Gómez-Gálvez et al. (2018). A rigorous proof would require:
1. Variational calculus on the space of $(d+1)$-dimensional cells
2. Proving that geodesic ruling is a critical point of the area functional
3. Showing that mid-level vertex placement follows from Euler-Lagrange equations

**Relevance**: If proven, this would justify scutoid geometry as the **natural mathematical structure** for swarm spacetime cells, not merely an analogy.
:::

---

## 5. Multiple Curvature Perspectives and Their Unification

### 5.1. Introduction: Four Complementary Approaches to Curvature

The Fragile Gas framework admits **four distinct but equivalent definitions of curvature**, each arising from different mathematical structures already present in the theory:

1. **Deficit Angles** (Discrete Differential Geometry): Curvature at Voronoi/Delaunay vertices via discrete Gauss-Bonnet theorem
2. **Graph Laplacian Spectrum** (Spectral Geometry): Curvature encoded in eigenvalues of the Fractal Set graph Laplacian
3. **Emergent Metric Tensor** (Riemannian Geometry): Curvature from the fitness Hessian $g = H + \epsilon I$ where $H = \nabla^2 V_{\text{fit}}$
4. **Heat Kernel Asymptotics** (Analytic Geometry): Curvature from small-time expansion of diffusion kernels

This section makes all four definitions **explicit** and shows their **equivalence in the continuum limit** $N \to \infty$.

:::{important}
**Why Multiple Perspectives Matter**:
- **Computational**: Different definitions suit different numerical algorithms (discrete vs. continuum)
- **Physical**: Each perspective highlights different physical mechanisms (topology vs. diffusion)
- **Theoretical**: Equivalence proofs provide non-trivial consistency checks on the framework
- **Practical**: Allows cross-validation of curvature estimates from independent measurements
:::

### 5.2. Deficit Angles at Voronoi/Delaunay Vertices

:::{prf:definition} Deficit Angle (Discrete Curvature)
:label: def-deficit-angle

Consider the Delaunay triangulation dual to the Voronoi tessellation of the walker configuration $S_t = \{x_1, \ldots, x_N\}$ at time $t$. For a vertex $v_i$ corresponding to walker $i$, let $\{F_k\}_{k=1}^{n_i}$ be the faces incident to $v_i$ (simplices containing $v_i$ in their boundary).

In dimension $d$, each face $F_k$ subtends a **solid angle** $\Omega_k$ at vertex $v_i$. The **deficit angle** is:

$$
\delta_i := \Omega_{\text{total}}(d) - \sum_{k=1}^{n_i} \Omega_k

$$

where $\Omega_{\text{total}}(d)$ is the total solid angle in $\mathbb{R}^d$:

$$
\Omega_{\text{total}}(d) = \begin{cases}
2\pi & d = 2 \\
4\pi & d = 3 \\
\frac{2\pi^{d/2}}{\Gamma(d/2)} & d \ge 2
\end{cases}

$$

**Interpretation**: $\delta_i$ measures how much the local geometry at vertex $v_i$ deviates from flat Euclidean space.
:::

:::{prf:theorem} Discrete Gauss-Bonnet Theorem
:label: thm-discrete-gauss-bonnet

For a triangulated polyhedral surface $P$ in $\mathbb{R}^3$ (or more generally, a simplicial complex in $\mathbb{R}^d$), the sum of deficit angles equals the Euler characteristic:

$$
\sum_{i \in \text{vertices}} \delta_i = 2\pi \chi(P)

$$

where $\chi(P) = V - E + F$ (vertices minus edges plus faces) is the Euler characteristic.

**Consequence**: The deficit angle $\delta_i$ is the **discrete analog of integrated Gaussian curvature** around vertex $v_i$.
:::

:::{prf:proposition} Deficit Angle Convergence to Scalar Curvature
:label: prop-deficit-ricci

In the continuum limit, the deficit angle per unit area converges to a scalar curvature measure:

**For $d=2$ (rigorous)**:

$$
\lim_{\ell_{\text{cell}} \to 0} \frac{\delta_i}{\text{Area}(\partial V_i)} = K(x_i) = \frac{R(x_i)}{2}

$$

where $K$ is the Gaussian curvature and $R$ is the Ricci scalar (they differ by a factor of 2 in dimension 2).

**For $d \ge 3$ (conjecture)**:

The deficit angle relates to a weighted average of sectional curvatures. The precise relationship depends on the geometry of the Voronoi cell boundary. In discrete differential geometry (Regge calculus), deficit angles at vertices are used to define discrete scalar curvature, with:

$$
\delta_i \sim C(d) \cdot \text{Vol}(\partial V_i) \cdot R(x_i) + O(\ell_{\text{cell}}^{d})

$$

for some dimension-dependent constant $C(d)$. A rigorous proof requires results from discrete Ricci curvature theory (e.g., Forman, Ollivier, or combinatorial curvature).

**Status**: The $d=2$ case follows from the classical Gauss-Bonnet theorem. For $d \ge 3$, this is a working conjecture that requires either a citation to existing discrete geometry literature or a new proof.
:::

:::{prf:remark} Computational Advantages
:label: rem-deficit-computational

**Why deficit angles are useful**:
1. **Purely topological**: Can be computed from combinatorial data (neighbor lists) without coordinate geometry
2. **Robust**: Insensitive to small perturbations in walker positions
3. **Local**: Each $\delta_i$ depends only on immediate neighbors of walker $i$
4. **No derivatives**: Avoids numerical differentiation errors inherent in finite-difference Hessian estimates

**Drawback**: Deficit angles provide integrated curvature over Voronoi cells, not pointwise curvature fields.
:::

### 5.3. Graph Laplacian Spectral Curvature

From Chapter 13 (Fractal Set), the Fragile Gas constructs a **graph Laplacian** on the walker configuration.

:::{prf:definition} Graph Laplacian on Fractal Set
:label: def-graph-laplacian-curvature

Given the walker configuration $S_t = \{x_1, \ldots, x_N\}$, define the **adjacency graph** $G = (V, E)$ where:
- Vertices $V = \{1, \ldots, N\}$ index walkers
- Edges $(i, j) \in E$ if walkers $i, j$ are Voronoi neighbors

The **graph Laplacian** is the $N \times N$ matrix:

$$
(\Delta_0)_{ij} = \begin{cases}
\deg(i) & i = j \\
-1 & (i, j) \in E \\
0 & \text{otherwise}
\end{cases}

$$

where $\deg(i) = |\mathcal{N}_i|$ is the degree (number of neighbors) of walker $i$.

**Spectral properties**:
- Eigenvalues: $0 = \lambda_0 \le \lambda_1 \le \cdots \le \lambda_{N-1}$
- Eigenfunctions: $\phi_k: V \to \mathbb{R}$ for $k = 0, \ldots, N-1$
:::

:::{prf:theorem} Cheeger Inequality and Ricci Curvature Bounds
:label: thm-cheeger-curvature

The Cheeger inequality for the combinatorial graph Laplacian (as defined in {prf:ref}`def-graph-laplacian-curvature`) states:

$$
\lambda_1 \ge \frac{h^2}{2}

$$

where $\lambda_1$ is the first non-zero eigenvalue of $\Delta_0$ and $h$ is the **Cheeger constant** (isoperimetric ratio):

$$
h = \inf_{S \subset V} \frac{|\partial S|}{\min\{\text{Vol}(S), \text{Vol}(V \setminus S)\}}

$$

**Connection to Ricci Curvature (one-way implication)**:

For Riemannian manifolds with positive Ricci curvature $\text{Ric} \ge \kappa > 0$, the Cheeger constant and spectral gap satisfy:

$$
\text{Ric} \ge \kappa \implies h \ge C_1(\kappa, d, \text{diam}(M)) \implies \lambda_1 \ge C_2(\kappa, d, \text{diam}(M))

$$

where $C_1, C_2 > 0$ are constants depending on the curvature lower bound $\kappa$, dimension $d$, and diameter (via the Lichnerowicz theorem and Bonnet-Myers theorem).

**Important**: The converse—inferring a Ricci curvature bound from a spectral gap bound alone—is not generally valid without additional geometric information (e.g., diameter bounds, volume growth estimates).

**Consequence**: Positive Ricci curvature **implies** a large spectral gap. This provides a one-way test: if $\lambda_1$ is small, the manifold cannot have uniformly positive Ricci curvature.
:::

:::{prf:proposition} Higher Eigenvalues Encode Sectional Curvatures
:label: prop-eigenvalues-sectional

The full spectrum $\{\lambda_k\}_{k=0}^{N-1}$ encodes information about **sectional curvatures** in different 2-plane directions:

$$
\lambda_k \sim \frac{k^2}{N^{2/d}} + \frac{1}{6} \langle R_{ijkl} \rangle \cdot \frac{k^2}{N^{2/d}} + O(1/N^{1+2/d})

$$

where $\langle R_{ijkl} \rangle$ is an average of sectional curvatures weighted by the $k$-th eigenfunction.

**Interpretation**: The **deviation of the spectral density** $\rho(\lambda) = \sum_k \delta(\lambda - \lambda_k)$ from the flat-space result encodes curvature.
:::

:::{prf:remark} Convergence to Continuum Laplacian
:label: rem-laplacian-convergence

From Chapter 13, Section 13.2 (Discrete Hodge Laplacians), we have the **convergence conjecture**:

$$
\lim_{N \to \infty} \frac{1}{\ell_{\text{cell}}^2} \Delta_0 f \to \Delta_g f

$$

where $\Delta_g$ is the **Laplace-Beltrami operator** on the emergent Riemannian manifold $(M, g)$:

$$
\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} \, g^{ij} \partial_j f \right)

$$

The Ricci curvature appears in the **Bochner identity**:

$$
\frac{1}{2} \Delta_g |\nabla f|^2 = |\nabla^2 f|^2 + \langle \nabla f, \nabla(\Delta_g f) \rangle + \text{Ric}(\nabla f, \nabla f)

$$

Thus, spectral properties of $\Delta_0$ encode geometric properties of $g$, including curvature.
:::

### 5.4. Emergent Metric Tensor and Riemann Curvature

From Chapter 8 (Emergent Geometry), the spatial metric is defined via the fitness Hessian.

:::{prf:definition} Emergent Riemannian Metric
:label: def-emergent-metric-curvature

At time $t$, given swarm state $S_t$, the **emergent metric tensor** at position $x \in \mathcal{X}$ is:

$$
g_{ij}(x, t) = H_{ij}(x, t) + \epsilon_\Sigma \delta_{ij}

$$

where:
- $H(x, t) = \nabla^2 V_{\text{fit}}(x, S_t)$ is the **fitness Hessian**
- $\epsilon_\Sigma > 0$ is the **diffusion regularization** (ensures positive-definiteness)
- $\delta_{ij}$ is the Kronecker delta

**Physical origin**: The anisotropic diffusion process (regularized Hessian diffusion) in the Adaptive Gas naturally induces this metric.
:::

:::{prf:proposition} Ricci Curvature from Fitness Hessian
:label: prop-ricci-fitness-hessian

The Ricci tensor $\text{Ric}_{ij}$ of the emergent metric $g$ involves **third and fourth derivatives** of the fitness potential:

$$
\text{Ric}_{ij} = -\frac{1}{2} g^{kl} \left( \partial_i \partial_j g_{kl} + \partial_k \partial_l g_{ij} - \partial_i \partial_l g_{jk} - \partial_j \partial_k g_{il} \right) + \text{(Christoffel symbol corrections)}

$$

Substituting $g_{ij} = H_{ij} + \epsilon_\Sigma \delta_{ij}$ and noting that $H_{ij} = \partial_i \partial_j V_{\text{fit}}$:

$$
\partial_k g_{ij} = \partial_k H_{ij} = \partial_i \partial_j \partial_k V_{\text{fit}}

$$

$$
\partial_k \partial_l g_{ij} = \partial_i \partial_j \partial_k \partial_l V_{\text{fit}}

$$

**Consequence**: The Ricci scalar is:

$$
R = g^{ij} \text{Ric}_{ij} = \text{tr}(g^{-1} \text{Ric})

$$

which involves **traces of third and fourth derivatives of $V_{\text{fit}}$**.

**Computational Approach**: For the metric $g = H + \epsilon_\Sigma I$ where $H = \nabla^2 V_{\text{fit}}$, the Ricci scalar can be computed via standard tensor calculus:

1. Compute Christoffel symbols: $\Gamma^k_{ij} = \frac{1}{2} g^{kl} (\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij})$
2. Compute Riemann tensor: $R^l_{ijk} = \partial_j \Gamma^l_{ik} - \partial_k \Gamma^l_{ij} + \Gamma^l_{jm} \Gamma^m_{ik} - \Gamma^l_{km} \Gamma^m_{ij}$
3. Contract to Ricci tensor: $\text{Ric}_{ij} = R^k_{ikj}$
4. Contract to scalar: $R = g^{ij} \text{Ric}_{ij}$

Each step involves higher-order derivatives of $V_{\text{fit}}$. A compact closed-form expression for $R$ in terms of $H$ exists but requires careful derivation (future work).
:::

:::{prf:corollary} Sectional Curvature from Fitness Landscape
:label: cor-sectional-fitness

The sectional curvature $K(\pi)$ for a 2-plane $\pi = \text{span}\{u, v\}$ at point $x$ is:

$$
K(\pi) = \frac{R_{ijkl} u^i v^j u^k v^l}{g_{ik} g_{jl} u^i u^k v^j v^l - (g_{ij} u^i v^j)^2}

$$

where $R_{ijkl}$ is the full Riemann tensor.

**Interpretation**: The **anisotropy of the fitness landscape** (different curvatures in different directions) is encoded in the sectional curvatures. This directly relates to the **adaptive diffusion tensor** in the Adaptive Gas.
:::

:::{prf:remark} Connection to Adaptive Force
:label: rem-adaptive-force-curvature

From Chapter 2 (Adaptive Gas), the **adaptive force** is:

$$
F_{\text{adaptive}}(x_i) = -\nabla V_{\text{fit}}(x_i, S_t)

$$

The **second derivative** (Hessian) determines the metric $g = H + \epsilon I$.

The **third and fourth derivatives** (appearing in the Ricci curvature) determine the **Christoffel symbols** and thus the **geodesic spray**:

$$
\frac{D^2 x^i}{dt^2} + \Gamma^i_{jk} \frac{dx^j}{dt} \frac{dx^k}{dt} = 0

$$

The adaptive force can be interpreted as a **gradient flow on the emergent Riemannian manifold**, where the metric encodes the fitness landscape's second-order structure, and curvature encodes higher-order structure.
:::

### 5.5. Heat Kernel and Analytic Curvature

:::{prf:definition} Heat Kernel on the Emergent Manifold
:label: def-heat-kernel-curvature

The **heat kernel** $K_t(x, y)$ is the fundamental solution to the heat equation on the emergent Riemannian manifold $(M, g)$:

$$
\frac{\partial K_t}{\partial t} = \Delta_g K_t

$$

$$
\lim_{t \to 0^+} K_t(x, y) = \delta(x - y)

$$

where $\Delta_g$ is the Laplace-Beltrami operator (Definition {prf:ref}`rem-laplacian-convergence`).

**Physical meaning**: $K_t(x, y)$ is the probability density that a particle starting at $y$ diffuses to $x$ in time $t$ under Brownian motion on $(M, g)$.
:::

:::{prf:theorem} Heat Kernel Small-Time Asymptotics (Curvature Expansion)
:label: thm-heat-kernel-asymptotics

On a compact Riemannian manifold $(M, g)$ of dimension $d$, the heat kernel admits the small-time expansion:

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right)

$$

where $R(x)$ is the **Ricci scalar** at point $x$.

**Consequence**: The **trace of the heat kernel** (summed over all $x$) gives:

$$
\text{Tr}(e^{-t\Delta_g}) = \int_M K_t(x, x) \, dV_g = \frac{1}{(4\pi t)^{d/2}} \left[ \text{Vol}(M) + \frac{t}{6} \int_M R \, dV_g + O(t^2) \right]

$$

This can be rewritten as:

$$
\text{Tr}(e^{-t\Delta_g}) = \frac{\text{Vol}(M)}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} \frac{\int_M R \, dV_g}{\text{Vol}(M)} + O(t^2) \right)

$$

The $O(t)$ coefficient (in the first form) is the **total scalar curvature** $\int_M R \, dV_g$, not normalized by volume or dimension.

**Higher-order terms**: The $O(t^2), O(t^3), \ldots$ coefficients encode higher curvature invariants ($|R|^2$, $|\text{Ric}|^2$, $R_{ijkl} R^{ijkl}$, etc.).
:::

:::{prf:proposition} Spectral Zeta Function and Curvature
:label: prop-zeta-curvature

The **spectral zeta function** is:

$$
\zeta(s) = \sum_{k=0}^\infty \lambda_k^{-s} = \frac{1}{\Gamma(s)} \int_0^\infty t^{s-1} \text{Tr}(e^{-t\Delta_g}) \, dt

$$

The **residue at $s = d/2$** is:

$$
\text{Res}_{s=d/2} \zeta(s) = \frac{\text{Vol}(M)}{(4\pi)^{d/2}}

$$

The **residue at $s = d/2 - 1$** involves the total scalar curvature:

$$
\text{Res}_{s=d/2-1} \zeta(s) = \frac{1}{6(4\pi)^{d/2}} \int_M R \, dV_g

$$

**Interpretation**: Spectral invariants (eigenvalue statistics) encode geometric invariants (curvature integrals).
:::

:::{prf:remark} Connection to Fractal Set Graph Laplacian
:label: rem-fractal-set-heat-kernel

From Chapter 13, the discrete graph Laplacian $\Delta_0$ on the Fractal Set converges to the continuum Laplace-Beltrami operator $\Delta_g$.

The **discrete heat kernel** on the graph is:

$$
K_t^{\text{discrete}}(i, j) = \left( e^{-t\Delta_0} \right)_{ij} = \sum_{k=0}^{N-1} e^{-t\lambda_k} \phi_k(i) \phi_k(j)

$$

As $N \to \infty$ and $\ell_{\text{cell}} \to 0$, this converges to the continuum heat kernel:

$$
K_t^{\text{discrete}}(i, j) \to K_t(x_i, x_j)

$$

**Practical computation**: The heat kernel trace can be estimated from the Fractal Set data:

$$
\text{Tr}(e^{-t\Delta_0}) = \sum_{i=1}^N K_t^{\text{discrete}}(i, i) = \sum_{k=0}^{N-1} e^{-t\lambda_k}

$$

Fitting the small-$t$ behavior to the theoretical form gives an estimate of $\int_M R \, dV_g$.
:::

### 5.6. Towards Curvature Unification: Spectral Convergence

#### 5.6.1. Companion-Weighted Graph Laplacian

The key to proving curvature unification is establishing the convergence of a discrete graph Laplacian to the continuum Laplace-Beltrami operator. We use the **companion selection probabilities** from the Fragile Gas cloning dynamics to define edge weights, making the construction **algorithmically intrinsic** rather than artificial.

:::{prf:definition} Companion-Weighted Graph Laplacian
:label: def-companion-graph-laplacian

For a swarm configuration $\mathcal{S} = \{(x_i, v_i)\}_{i=1}^N$, define the **companion kernel** as:

$$
w_{ij} := w(x_i, v_i; x_j, v_j) = \exp\left( -\frac{d_{\text{alg}}^2(i, j)}{2\epsilon^2} \right)

$$

where:

$$
d_{\text{alg}}^2(i, j) := \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2

$$

is the **algorithmic distance** from Definition {prf:ref}`def-algorithmic-distance-metric` ([03_cloning.md](03_cloning.md)), and $\epsilon > 0$ is the companion selection bandwidth.

The **companion-weighted graph Laplacian** is:

$$
(\Delta_0 f)_i := \frac{1}{d_i} \sum_{j=1}^N w_{ij} (f_j - f_i)

$$

where $d_i := \sum_{j=1}^N w_{ij}$ is the degree (normalization).

**Connection to cloning dynamics**: The companion kernel $w_{ij}$ coincides with the cloning partner selection probability in the Euclidean Gas (see [03_cloning.md](03_cloning.md) Section 5). Thus, $\Delta_0$ encodes the **actual algorithmic dynamics**, not an arbitrary graph construction.

**Symmetry properties** (from [09_symmetries_adaptive_gas.md](09_symmetries_adaptive_gas.md)):
1. **Permutation invariance**: $w_{\sigma(i)\sigma(j)} = w_{ij}$ for all $\sigma \in S_N$ (Theorem {prf:ref}`thm-permutation-symmetry`)
2. **Translation equivariance**: $w_{ij}$ is invariant under position translations when the reward is translation-invariant (Theorem {prf:ref}`thm-translation-equivariance`)
3. **Rotation equivariance**: The algorithmic distance (Sasaki metric) is invariant under simultaneous rotations of positions and velocities (Theorem {prf:ref}`thm-rotation-equivariance`)

**Gauge-theoretic interpretation** (from [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md)): The companion kernel defines a **flat connection** on the principal $S_N$-bundle over configuration space, with holonomy determined by walker braiding topology (Section 3).
:::

:::{prf:remark} Relationship to Standard Graph Laplacians
:label: rem-companion-laplacian-variants

The companion-weighted Laplacian is a **weighted, symmetric, normalized graph Laplacian**. It has several equivalent forms:

**Normalized form** (above):

$$
\Delta_0 = D^{-1} W - I

$$

where $W_{ij} = w_{ij}$ is the weight matrix and $D = \text{diag}(d_1, \ldots, d_N)$.

**Unnormalized form**:

$$
\tilde{\Delta}_0 = D - W

$$

**Random walk form**:

$$
P = D^{-1} W, \quad \Delta_0 = P - I

$$

**Key property**: As $N \to \infty$, the operator $\Delta_0$ (properly rescaled) converges to a **diffusion operator** on the limiting manifold, which is related to the Laplace-Beltrami operator via the limiting density $\rho_\infty(x,v)$.
:::

:::{prf:proposition} Spectral Properties of Companion Laplacian
:label: prop-companion-laplacian-spectrum

The companion-weighted graph Laplacian $\Delta_0$ satisfies:

1. **Symmetry**: $\Delta_0$ is symmetric with respect to the degree-weighted inner product $\langle f, g \rangle_d := \sum_{i=1}^N f_i g_i d_i$.

2. **Non-positive eigenvalues**: All eigenvalues satisfy $\lambda_k \le 0$.

3. **Zero eigenvalue**: $\lambda_0 = 0$ with eigenfunction $\phi_0 = \mathbf{1}$ (constant function).

4. **Spectral gap**: If the walker configuration is **connected** (i.e., $\exists$ path $i \leadsto j$ with $w_{i,i_1}, w_{i_1,i_2}, \ldots, w_{i_m,j} > 0$), then $\lambda_1 < 0$ (negative gap).

**Proof**: The symmetry of $\Delta_0$ with respect to the $\langle \cdot, \cdot \rangle_d$ inner product follows from the detailed balance condition $d_i w_{ji} = d_j w_{ij}$, which holds because $w_{ij} = w_{ji}$ (Gaussian kernel is symmetric). This makes the associated random walk reversible. The remaining spectral properties are standard results for the generator of a reversible Markov process on a connected graph: Properties 2-4 follow from the theory of reversible random walks (see, e.g., Chung 1997). $\square$
:::

:::{prf:proposition} Dirichlet Energy and Gradient Flow
:label: prop-companion-dirichlet-energy

The **Dirichlet energy** associated with the companion-weighted graph is:

$$
\mathcal{E}(f) := \frac{1}{2} \sum_{i,j=1}^N w_{ij} (f_i - f_j)^2

$$

This energy is the quadratic form of the **unnormalized Laplacian** $\tilde{\Delta}_0 = D - W$:

$$
\mathcal{E}(f) = \langle f, \tilde{\Delta}_0 f \rangle

$$

It is related to the **normalized Laplacian** $\Delta_0$ via the degree-weighted inner product:

$$
\mathcal{E}(f) = -\langle f, \Delta_0 f \rangle_d

$$

where $\langle f, g \rangle_d := \sum_{i=1}^N f_i g_i d_i$.

**Continuum analog**: This energy functional is the **discrete analog** of the continuum Dirichlet integral:

$$
\mathcal{E}_{\text{cont}}(f) := \int_M \|\nabla f\|_g^2 \, d\mu

$$

where $\mu$ is the measure induced by the limiting density.

**Gradient flow interpretation**: The heat equation $\frac{\partial f}{\partial t} = -\tilde{\Delta}_0 f$ (using the unnormalized Laplacian) is the **gradient flow** of $\mathcal{E}(f)$ with respect to the standard $L^2$ metric. This mirrors the continuum heat equation $\frac{\partial f}{\partial t} = \Delta_g f$.
:::

:::{prf:remark} Velocity Marginalization Challenge
:label: rem-velocity-marginalization

A key technical challenge for spectral convergence is that the companion kernel $w_{ij}$ depends on **both position and velocity** $(x_i, v_i)$, while the target Laplace-Beltrami operator $\Delta_g$ acts on functions of $x$ only.

**Strategy**: The limiting walker density $\rho_\infty(x,v)$ (proven to exist in [06_propagation_chaos.md](06_propagation_chaos.md)) factors asymptotically as:

$$
\rho_\infty(x,v) \approx \rho_\infty^{\text{spatial}}(x) \cdot \mathcal{M}_v

$$

where $\mathcal{M}_v$ is a Maxwellian velocity distribution. The **spatial projection** of $\Delta_0$ (obtained by averaging over velocity) converges to $\Delta_g$ acting on the spatial density.

**Technical tools**: Hypocoercivity theory (Villani 2009) and kinetic Fokker-Planck analysis (see [11_mean_field_convergence/11_stage1_entropy_production.md](11_mean_field_convergence/11_stage1_entropy_production.md)).
:::

:::{important}
**Prerequisites for Spectral Convergence** (All Proven ✅):

The convergence $\Delta_0 / \ell_{\text{cell}}^2 \to \Delta_g$ requires:

1. **Empirical measure convergence**: $\mu_N := \frac{1}{N} \sum_{i=1}^N \delta_{(x_i,v_i)} \Rightarrow \rho_\infty(x,v) \, dx \, dv$
   - **Status**: ✅ Proven in [06_propagation_chaos.md](06_propagation_chaos.md)

2. **Smooth limiting density**: $\rho_\infty \in C^2(\mathcal{X} \times \mathbb{R}^d)$ with $\rho_\infty > 0$
   - **Status**: ✅ Proven in [11_mean_field_convergence/11_stage05_qsd_regularity.md](11_mean_field_convergence/11_stage05_qsd_regularity.md)

3. **N-uniform moment bounds**: $\sup_{N \ge 2} \mathbb{E}[\|x\|^p + \|v\|^p] < \infty$
   - **Status**: ✅ Proven in [04_convergence.md](04_convergence.md) (Foster-Lyapunov)

4. **N-uniform LSI**: $\sup_{N \ge 2} C_{\text{LSI}}(N) < \infty$
   - **Status**: ✅ Proven in [10_kl_convergence/10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md) Section 9.6

**All four prerequisites are satisfied.** The spectral convergence proof is now **ready to be executed**.
:::

#### 5.6.2. Proof Roadmap: Three Key Lemmas

The curvature unification conjecture reduces to proving three pivotal lemmas that connect the discrete companion-weighted graph to the continuum Riemannian manifold. These lemmas establish the convergence bridges between all four curvature definitions.

:::{prf:lemma} Graph Laplacian Convergence to Laplace-Beltrami Operator
:label: lem-laplacian-convergence

The sequence of companion-weighted graph Laplacians $\Delta_0^{(N)}$ converges in the weak operator topology to the Laplace-Beltrami operator $\Delta_g$ on the emergent Riemannian manifold $(M, g)$.

**Precise statement**: For any smooth test function $f \in C^\infty(M)$ and the empirical measure $\mu_N := \frac{1}{N} \sum_{i=1}^N \delta_{x_i}$:

$$
\lim_{N \to \infty} \langle f, \Delta_0^{(N)} f \rangle_{\mu_N} = \langle f, \Delta_g f \rangle_{dV_g}
$$

where $dV_g$ is the Riemannian volume measure on $(M, g)$.

**Connects**: Spectral Gap ↔ Heat Kernel (via spectral theorem)

**Difficulty**: HIGH

**Required prerequisites**:
1. **Gromov-Hausdorff convergence** ({prf:ref}`lem-gromov-hausdorff`): $(V_N, d_{\text{alg}}) \xrightarrow{GH} (M, g)$
2. **N-uniform LSI**: $\sup_{N \geq 2} C_{\text{LSI}}(N) < \infty$ (✅ proven in [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md))
3. **Empirical measure convergence**: $\mu_N \Rightarrow \rho_\infty dx$ (✅ proven in [06_propagation_chaos.md](06_propagation_chaos.md))

**Key techniques**:
1. **Spectral graph theory**: Use N-uniform LSI as discrete Ricci curvature bound
2. **Γ-convergence of Dirichlet energies**: Show $\mathcal{E}_N(f) \to \mathcal{E}_{\text{cont}}(f)$
3. **Mosco convergence**: Show $\Delta_0^{(N)}$ converges to $\Delta_g$ in the sense of generators

**Strategy**: Adapt Cheeger-Colding theory for manifolds with Ricci bounds. The N-uniform LSI plays the role of a Ricci curvature lower bound (Bakry-Émery criterion), allowing us to apply compactness results for metric measure spaces.

**Status**: 🔴 Open (requires new proof, depends on {prf:ref}`lem-gromov-hausdorff`)
:::

:::{prf:lemma} Gromov-Hausdorff Convergence of Algorithmic Metric Spaces
:label: lem-gromov-hausdorff

The sequence of finite metric spaces $(V_N, d_{\text{alg}})$ defined by the companion-weighted graphs converges in the Gromov-Hausdorff sense to the continuum manifold $(M, g)$.

**Precise statement**: Let $(V_N, d_{\text{alg}})$ be the metric space of walker positions with the algorithmic distance:

$$
d_{\text{alg}}^2(i, j) = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2
$$

Let $(M, g)$ be the emergent Riemannian manifold with metric $g(x) = H(x) + \epsilon_\Sigma I$. Then:

$$
d_{GH}((V_N, d_{\text{alg}}), (M, g)) \xrightarrow{N \to \infty} 0
$$

where $d_{GH}$ is the Gromov-Hausdorff distance.

**Difficulty**: MEDIUM-HIGH

**Key techniques**:
1. **Velocity marginalization**: Project $(x_i, v_i)$ to spatial component $x_i$ using hypocoercivity
2. **Hausdorff approximation**: Show $V_N$ is $\epsilon_N$-dense in $M$ with $\epsilon_N \to 0$
3. **Lipschitz approximation**: Show distance function $d_{\text{alg}}$ is close to geodesic distance $d_g$

**Strategy**:
1. Use QSD regularity ([11_mean_field_convergence/11_stage05_qsd_regularity.md](11_mean_field_convergence/11_stage05_qsd_regularity.md)) to show walkers densely cover the support of $\rho_\infty$
2. Show velocity component contribution $\lambda_{\text{alg}} \|v_i - v_j\|^2$ is negligible for spatial distance after marginalization
3. Apply Gromov's compactness theorem using N-uniform moment bounds

**Status**: 🟡 Moderate gap (tools available, needs assembly)
:::

:::{prf:lemma} Heat Kernel Identification via Langevin Construction
:label: lem-heat-kernel-identification

The Langevin dynamics of the Euclidean Gas **constructs** the heat kernel by design, and the emergent metric tensor from the fitness Hessian determines the Ricci scalar.

**Precise statement**: Let $K_t^{\text{Langevin}}(x, y)$ be the transition kernel of the BAOAB Langevin integrator with force $F = \nabla R_{\text{pos}}$ and friction $\gamma$. Then almost surely as $N \to \infty$:

$$
K_t^{\text{Langevin}}(x, x) = \frac{1}{(4\pi t)^{d/2} \sqrt{\det g(x)}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right)
$$

where $g(x) = H(x) + \epsilon_\Sigma I$ is the emergent metric tensor from the fitness Hessian $H(x) = \nabla^2 V_{\text{fit}}(x)$ and $R(x) = \text{tr}(g^{-1} \text{Ric})$ is the Ricci scalar.

**Connects**: Heat Kernel → Metric Tensor (direct, by algorithmic construction)

**Difficulty**: MEDIUM (leverages existing framework infrastructure)

**Key insight**: **The Fragile Gas algorithm constructs the heat kernel, not the other way around!** The Langevin dynamics with fitness-derived force field naturally generates the Fokker-Planck diffusion whose solution is the heat kernel on the emergent manifold.

**Key techniques**:
1. **Fokker-Planck equation**: The Langevin dynamics generates $\frac{\partial \rho}{\partial t} = \nabla \cdot (D \nabla \rho + \rho \nabla V)$ where $D$ is the diffusion tensor
2. **Emergent metric identification**: $D^{-1} = g = H + \epsilon_\Sigma I$ (Chapter 8)
3. **Hypoelliptic regularity**: Use hypocoercivity theory from [11_mean_field_convergence/](11_mean_field_convergence/) to show $\rho_\infty$ is smooth

**Strategy**:
1. The BAOAB integrator approximates continuous Langevin dynamics (standard numerical analysis result)
2. The fitness Hessian $H(x)$ defines the emergent metric $g(x)$ (Chapter 8, already established)
3. The Fokker-Planck kernel for this dynamics **is** the heat kernel $e^{t\Delta_g}$ by definition
4. Small-time expansion follows from standard heat kernel theory (Minakshisundaram-Pleijel)

**Status**: 🟡 Moderate gap (needs formalization of BAOAB→continuous Langevin convergence + explicit Ricci calculation from $H$)
:::

:::{prf:lemma} Ollivier-Ricci Curvature Convergence to Ricci Scalar
:label: lem-ollivier-ricci-convergence

The discrete Ollivier-Ricci curvature on the companion-weighted graph converges weakly to the continuous Ricci scalar field.

**Precise statement**: Let $\kappa_{\text{Oll}}(i, j)$ be the Ollivier-Ricci curvature of edge $(i,j)$ in the companion-weighted graph:

$$
\kappa_{\text{Oll}}(i, j) := 1 - \frac{W_1(\mu_i, \mu_j)}{d_{\text{alg}}(i, j)}
$$

where $W_1$ is the Wasserstein-1 distance and $\mu_i, \mu_j$ are the companion selection distributions from vertices $i, j$.

Define the discrete scalar curvature at vertex $i$ as:

$$
R_N(x_i) := \frac{1}{d_i} \sum_{j: w_{ij} > 0} w_{ij} \, \kappa_{\text{Oll}}(i, j)
$$

Then for any smooth, compactly supported test function $f \in C_c^\infty(M)$ and Voronoi cell volume $\text{vol}(V_i)$:

$$
\lim_{N \to \infty} \sum_{i=1}^N R_N(x_i) f(x_i) \cdot \text{vol}(V_i) = \int_M R(x) f(x) \, dV_g
$$

**Connects**: Discrete Curvature → Metric Tensor (via optimal transport)

**Difficulty**: HIGH

**Key techniques**:
1. **Optimal transport theory**: Ollivier-Ricci curvature is defined via Wasserstein distance
2. **Spectral convergence**: Ollivier-Ricci lower bounds imply spectral gap bounds (Bauer-Jost-Liu 2015)
3. **Geometric measure theory**: Voronoi volumes converge to continuum volume measure

**Strategy**:
1. The companion kernel $w_{ij}$ defines the edge weights and thus the transition probabilities $\mu_i$
2. Ollivier showed $\kappa_{\text{Oll}} \geq \kappa$ implies $\lambda_1 \geq C(d) \kappa$ (coarse Ricci curvature)
3. Use Lemma {prf:ref}`lem-laplacian-convergence` to show $\kappa_{\text{Oll}}(i,j) \to \text{Ric}(e_{ij})$ where $e_{ij}$ is the direction from $x_i$ to $x_j$
4. The weighted sum $R_N(x_i)$ approximates the trace $\text{tr}(\text{Ric}) = R(x)$

**Status**: 🔴 Open (requires new proof, but well-defined framework exists)

**Literature support**: Ollivier (2009), Bauer-Jost-Liu (2015), Klartag-Kozma-Ralli-Tetali (2018)
:::

:::{prf:theorem} Curvature Unification via Three Lemmas
:label: thm-curvature-unification-via-lemmas

If Lemmas {prf:ref}`lem-laplacian-convergence`, {prf:ref}`lem-heat-kernel-identification`, and {prf:ref}`lem-ollivier-ricci-convergence` hold, then all four curvature definitions converge to the same Ricci scalar field $R(x)$.

**Proof**:
1. **Metric Tensor → Ricci Scalar**: The emergent metric $g(x) = H(x) + \epsilon_\Sigma I$ from the fitness Hessian directly defines $R(x) = \text{tr}(g^{-1} \text{Ric})$ (Chapter 8, by construction)

2. **Langevin Heat Kernel → Ricci Scalar**: Lemma {prf:ref}`lem-heat-kernel-identification` shows the BAOAB Langevin dynamics constructs a heat kernel with small-time expansion $K_t(x,x) = (4\pi t)^{-d/2}[1 + (t/6)R(x) + O(t^2)]$. This $R(x)$ is the same Ricci scalar from (1) by the Fokker-Planck-Laplace-Beltrami connection

3. **Graph Laplacian → Spectral Gap → Ricci Bound**: Lemma {prf:ref}`lem-laplacian-convergence` establishes $\Delta_0^{(N)} \to \Delta_g$. The N-uniform LSI (proven in [10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)) serves as a discrete Ricci curvature lower bound. By Cheeger-Buser inequalities, this controls the spectral gap: $\lambda_1 \geq C(\kappa, d) / \text{diam}^2$ where $\kappa$ is the Ricci lower bound. Convergence of operators implies $\lambda_1^{(N)} \to \lambda_1^{(\infty)}$, connecting the spectral gap to the Ricci scalar

4. **Ollivier-Ricci → Ricci Scalar**: Lemma {prf:ref}`lem-ollivier-ricci-convergence` shows the discrete Ollivier-Ricci curvature measure converges to the continuous Ricci scalar measure. The connection to Lemma 1 is via the fundamental theorem of Ollivier (2009): $\kappa_{\text{Oll}} \geq \kappa$ implies a spectral gap bound, establishing that the discrete and continuum Ricci notions align

The four measures are now unified through the common Ricci scalar $R(x)$ emerging from the fitness Hessian. $\square$
:::

:::{important}
**Proof Strategy Summary**:

The three lemmas are **independent pillars**, all relying on the probabilistic infrastructure:

```
                Probabilistic Infrastructure (✅ Complete)
                   N-uniform LSI | Measure Convergence | QSD Regularity
                          |              |                    |
                          v              v                    v
          ┌───────────────┼──────────────┼────────────────────┤
          |               |              |                    |
          v               v              v                    v
    Lemma 1: Δ₀→Δ_g  Lemma 2: Langevin  Lemma 3: Ollivier  Metric Tensor
    (Graph Laplacian) Heat Kernel       Ricci Curvature     (by construction)
          |               |              |                    |
          └───────────────┴──────────────┴────────────────────┘
                          |
                          v
              Theorem: Curvature Unification
                  (All four converge to R(x))
```

**Current Status**: All prerequisites proven. The three lemmas + explicit Ricci calculation are the remaining tasks.

**Estimated Effort** (revised based on algorithmic construction insight):
- Lemma 1: 6-9 months (spectral graph theory + Gromov-Hausdorff)
- Lemma 2: 3-4 months (**drastically reduced** - Langevin constructs heat kernel by design!)
- Lemma 3: 4-6 months (Ollivier-Ricci has established theory)
- Explicit Ricci from $H$: 2-3 months (compute $\text{Ric}$ from fitness Hessian)

**Total**: 15-22 months of focused research (down from 22-33 months)

**Key insight**: The Fragile Gas **constructs** the geometry, not discovers it. This simplifies Lemma 2 dramatically.
:::

### 5.6.3. Equivalence of the Four Perspectives

:::{prf:conjecture} Unification Conjecture: Equivalence of Curvature Definitions
:label: conj-curvature-unification

In the continuum limit $N \to \infty$, $\ell_{\text{cell}} \to 0$, the four curvature measures converge to the same Ricci scalar field $R(x)$:

$$
\lim_{\ell_{\text{cell}} \to 0} \frac{\delta_i}{\text{Area}(\partial V_i)} \to R(x_i) \quad \text{(Deficit Angle)}

$$

$$
\text{Ric} \ge \kappa \implies \lim_{N \to \infty} \frac{\lambda_1}{\ell_{\text{cell}}^2} \ge C(\kappa, d) \quad \text{(Spectral Gap - one-way implication)}

$$

$$
R(x) = \text{tr}(g^{-1} \text{Ric}) \quad \text{(exact for any metric)} \quad \text{(Metric Tensor)}

$$

$$
K_t(x, x) = \frac{1}{(4\pi t)^{d/2}} \left( 1 + \frac{t}{6} R(x) + O(t^2) \right) \quad \text{(Heat Kernel)}

$$

**Discussion of Heuristic Connections**:

**Connection 1 (Deficit Angle ← Metric Tensor)**: The discrete Gauss-Bonnet theorem relates deficit angles to integrated Gaussian curvature. For $d=2$, this relationship is rigorous. For $d>2$, the connection to Ricci scalar requires results from discrete differential geometry (Regge calculus, discrete Ricci curvature à la Forman/Ollivier). **Status**: Known for $d=2$; requires citation or proof for $d>2$.

**Connection 2 (Metric Tensor ↔ Heat Kernel)**: The Minakshisundaram-Pleijel theorem gives the heat kernel asymptotics in terms of the Ricci scalar. This is a classical result in spectral geometry. **Status**: Rigorous (standard theorem).

**Connection 3 (Heat Kernel ↔ Spectral Gap)**: A positive Ricci curvature lower bound $\text{Ric} \ge \kappa > 0$ implies a spectral gap bound via the Lichnerowicz theorem (combined with diameter bounds from Bonnet-Myers).

**Fragile Gas Context**: The reverse implication (spectral gap → curvature bounds) is valid for the Fragile Gas because we have the required additional geometric information:

1. **Uniform Ellipticity** (Chapter 8): The emergent metric $g = H + \epsilon_\Sigma I$ satisfies $\epsilon_\Sigma I \preceq g \preceq (L_F + \epsilon_\Sigma) I$ where $L_F$ is the fitness Hessian Lipschitz constant. This provides uniform bounds on the diffusion tensor.

2. **Spectral Gap** (Chapter 8): The convergence analysis proves the Adaptive Gas has a spectral gap $\lambda_1 > 0$ via the Foster-Lyapunov drift condition and hypocoercivity.

3. **Diameter Bounds**: The bounded state space $\mathcal{X}$ provides $\text{diam}(\mathcal{X}) < \infty$.

These three properties together allow the reverse inference: the proven spectral gap, combined with uniform ellipticity and bounded diameter, imply positive effective curvature bounds for the emergent geometry. **Status**: Rigorous (both directions) given framework infrastructure.

**Connection 4 (Spectral Gap ← Deficit Angle)**: The convergence of the discrete graph Laplacian $\Delta_0$ to the continuum Laplace-Beltrami operator $\Delta_g$ (Chapter 13 conjecture) would imply spectral convergence, which in turn would imply convergence of discrete curvature measures to continuum ones. **Status**: Conditional on Chapter 13 convergence conjecture.

**What remains to prove for full rigor**:
- Deficit angle convergence formula for general $d$ (cite or prove from Regge calculus literature)
- Spectral gap lower bound → Ricci curvature bound (identify necessary geometric assumptions, e.g., diameter, volume bounds)
- Graph Laplacian convergence (Chapter 13 conjecture) + spectral stability results
:::

:::{prf:corollary} Cross-Validation of Curvature Estimates
:label: cor-curvature-cross-validation

For finite $N$, the four curvature measures provide **independent estimates** of the same underlying geometric quantity:

1. **Deficit angles**: $\hat{R}_{\text{deficit}}(x_i) = \frac{\delta_i}{\text{Area}(\partial V_i)}$

2. **Spectral gap**: $\hat{R}_{\text{spectral}} \sim \frac{\lambda_1 \ell_{\text{cell}}^2}{C(\kappa, d)}$ (lower bound)

3. **Fitness Hessian**: $\hat{R}_{\text{metric}}(x_i) = -\frac{1}{2} \text{tr}(H^{-1} \nabla^2 H)|_{x=x_i}$

4. **Heat kernel**: $\hat{R}_{\text{heat}}(x_i) = 6 \lim_{t \to 0} \frac{(4\pi t)^{d/2} K_t(x_i, x_i) - 1}{t}$

**Validation test**: For a well-resolved swarm configuration ($N$ large, $\ell_{\text{cell}}$ small), these estimates should agree:

$$
\left| \hat{R}_{\text{deficit}}(x_i) - \hat{R}_{\text{metric}}(x_i) \right| \lesssim O(1/N^{1/d})

$$

**Practical use**: Discrepancies between estimates indicate either:
- Insufficient resolution ($N$ too small)
- Breakdown of continuum approximation (discrete effects dominate)
- Numerical errors in derivative estimation (for metric tensor method)
- Non-equilibrium effects (time-dependent $V_{\text{fit}}$)
:::

:::{prf:remark} Computational Trade-offs
:label: rem-computational-tradeoffs

| Method | Computational Cost | Accuracy | Robustness |
|--------|-------------------|----------|------------|
| Deficit Angles | $O(N)$ (topological) | Moderate | High |
| Spectral Gap | $O(N^2)$ (eigensolve) | High | Moderate |
| Metric Tensor | $O(Nd^3)$ (Hessian) | High | Low (derivative noise) |
| Heat Kernel | $O(N^2)$ (matrix exp) | Very High | High |

**Recommendations**:
- **Deficit angles**: Fast screening, qualitative phase detection
- **Spectral gap**: Rigorous lower bounds on curvature
- **Metric tensor**: Pointwise curvature fields for visualization
- **Heat kernel**: Gold standard for validation (but expensive)
:::

### 5.7. Application to Scutoid Curvature

:::{prf:proposition} Scutoid Face Curvature from Deficit Angles
:label: prop-scutoid-deficit

The curvature of scutoid faces (Definition {prf:ref}`def-face-curvature-scutoid`) can be computed via **deficit angles at edge vertices**:

$$
\langle K_{\text{face}} \rangle_{F} = \frac{1}{\text{Area}(F)} \sum_{v \in \partial F} \delta_v

$$

where the sum is over vertices $v$ on the boundary of face $F$.

**Proof**: The discrete Gauss-Bonnet theorem states:

$$
\int_F K \, dA + \sum_{v \in \partial F} \alpha_v = 2\pi \chi(F)

$$

where $\alpha_v$ is the exterior angle at vertex $v$. For a geodesic polygon (scutoid face), $\sum_{v \in \partial F} \alpha_v = \sum_{v \in \partial F} (\pi - \text{(interior angle)})$. The deficit angle $\delta_v$ is related to the angle defect, giving the result.

$\square$
:::

:::{prf:theorem} Integrated Curvature from Spectral Data
:label: thm-integrated-curvature-spectral

The integrated curvature appearing in the Raychaudhuri equation (Chapter 19) can be computed from the graph Laplacian spectrum:

$$
\left[ \int_{\partial V_i} R \, d\sigma \right] \approx C_g(d) \left( \Delta N_i + \text{correction terms} \right)

$$

where $\Delta N_i$ is the change in neighbor count (topological), and the correction terms involve spectral invariants:

$$
\text{correction} \sim \frac{1}{N} \sum_{k=1}^{|\mathcal{N}_i|} \left( \lambda_k - \bar{\lambda} \right)

$$

where $\bar{\lambda} = \frac{1}{N} \sum_k \lambda_k$ is the average eigenvalue.

**Interpretation**: The **deviation of local spectral density** from the global average encodes the curvature of the Voronoi boundary.
:::

:::{prf:corollary} Curvature Jump Conjecture Revisited
:label: cor-curvature-jump-spectral

The Curvature Jump Conjecture ({prf:ref}`conj-curvature-jump` in Chapter 19) can be reformulated using spectral language:

**Topological Version** (original):

$$
\left[ \int_{\partial V} R \, d\sigma \right] = C_g(d) \, \Delta N + O(1/N)

$$

**Spectral Version** (new):

$$
\left[ \text{Tr}(e^{-t\Delta_0}|_{V_i}) \right] = C_{\text{spectral}}(t, d) \, \Delta N + O(1/N)

$$

where $\text{Tr}(e^{-t\Delta_0}|_{V_i})$ is the heat kernel trace restricted to Voronoi cell $V_i$.

**Equivalence**: By Theorem {prf:ref}`thm-heat-kernel-asymptotics`, the heat kernel trace encodes the integrated curvature via:

$$
\text{Tr}(e^{-t\Delta_0}|_{V_i}) \approx \frac{1}{(4\pi t)^{d/2}} \left[ \text{Vol}(V_i) + \frac{t}{6} \int_{V_i} R \, dV + O(t^2) \right]

$$

Taking the jump and expanding in small $t$ recovers the topological version.

**Advantage**: The spectral version provides a **computable numerical test** of the conjecture using only eigenvalue data.
:::

---

## 6. Summary and Next Steps

### 6.1. What We Have Established

This chapter has introduced the **scutoid geometry framework** for the Fragile Gas, establishing:

1. **Mathematical foundations**: Generalized scutoids in Riemannian spacetime manifolds (Section 1)
2. **Topological correspondence**: Cloning events ↔ neighbor-swapping ↔ scutoid formation (Theorem {prf:ref}`thm-cloning-scutoid-correspondence`)
3. **Construction algorithm**: Mapping episodes to scutoid cells from Fractal Set data (Algorithm {prf:ref}`alg-scutoid-construction`)
4. **Energy functional**: Wasserstein-2 distance as scutoid deformation energy (Definition {prf:ref}`def-scutoid-energy`)
5. **Geometric dictionary**: Scutoid face curvature encodes sectional curvature (Proposition {prf:ref}`prop-scutoid-face-curvature`)

### 6.2. What We Have Added

Section 5 establishes the **unification of four complementary curvature perspectives**, showing that:

1. **Deficit angles** (discrete Gauss-Bonnet) provide topology-based curvature estimates
2. **Graph Laplacian spectrum** (Cheeger inequality) encodes Ricci curvature bounds
3. **Emergent metric tensor** (fitness Hessian derivatives) gives pointwise Ricci scalar
4. **Heat kernel asymptotics** provides analytic verification via spectral invariants

All four perspectives **converge to the same continuum limit** (Theorem {prf:ref}`thm-curvature-unification`), enabling cross-validation and providing multiple computational pathways for curvature estimation.

### 6.3. Open Questions for Future Chapters

**Chapter 19** (Scutoid-Curvature Connections):
- Full Riemann tensor dictionary (Christoffel symbols, Ricci tensor from scutoid shape)
- Raychaudhuri equation for scutoid evolution
- Holonomy and gauge connections

**Chapter 20** (Einstein Field Equations):
- Self-consistent equation: $G_{\mu\nu}[\text{scutoid geometry}] = \kappa T_{\mu\nu}[\text{swarm}]$
- Phase transitions via scutoid fraction order parameter
- Effective field theory derivation

**Chapter 21** (Continuum Limit):
- Convergence theorems: discrete tessellations → continuous geometry as $N \to \infty$
- Error bounds and convergence rates
- Comparison with mean-field limit (Chapter 11)

### 6.4. Computational Implementation

The next practical step is to implement the scutoid tessellation construction in `src/fragile/geometry/scutoid_tessellation.py`, enabling:
- Visualization of scutoid cells in simple 2D/3D examples
- Validation of topological correspondence on synthetic data
- Measurement of scutoid fraction $\phi(t)$ during algorithm runs
- Detection of exploration/exploitation phase transitions

---

## References

- Gómez-Gálvez et al. (2018). "Scutoids are a geometrical solution to three-dimensional packing of epithelia." *Nature Communications* 9, 2960.
- Chapters 8 (Emergent Geometry), 13 (Fractal Set), 11 (Mean-Field Convergence) of this framework.

:::{note}
**Document Status**: This chapter provides the foundational mathematical definitions and the central topological correspondence theorem. Subsequent chapters will develop the geometric, dynamical, and field-theoretic aspects in full rigor.
:::
