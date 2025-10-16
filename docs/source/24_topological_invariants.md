# Topological Invariants for the Fragile Gas Framework

## Executive Summary

### Purpose and Scope

This document provides a unified treatment of **topological invariants** in the Fragile Gas framework, consolidating computational methods for characterizing the global topological structure of the fitness landscape and walker configuration space. While [curvature.md](curvature.md) focuses on local geometric properties, this document addresses global topological features that remain invariant under continuous deformations.

**Key Achievement**: We establish that the Fragile Gas framework admits **multiple computational pathways** for computing topological invariants from the discrete Delaunay triangulation and Fractal Set structure, enabling both classical algebraic topology methods and modern computational topology techniques.

### Core Topological Invariants

This document covers the following fundamental topological invariants:

1. **Euler Characteristic** ($\chi$)
   - Combinatorial formula: $V - E + F - T$ (for 3D simplicial complexes)
   - Relation to Gauss-Bonnet theorem via deficit angles
   - **Computed via Delaunay triangulation** (⚠️ yields point cloud topology, not landscape topology; see § 1.1 for critical limitations)

2. **Betti Numbers** ($\beta_0, \beta_1, \beta_2, \ldots$)
   - $\beta_0$: Connected components
   - $\beta_1$: Independent cycles (loops)
   - $\beta_k$: $k$-dimensional holes
   - Computable via persistent homology

3. **Homology Groups** ($H_k$)
   - Formal algebraic structure encoding topological features
   - Relation: $\beta_k = \text{rank}(H_k)$
   - Requires boundary operator computation on simplicial complex

4. **Homotopy Groups** ($\pi_k$)
   - Fundamental group $\pi_1$: Loop equivalence classes
   - Higher homotopy groups: Homotopy classes of sphere maps
   - Framework connection: already defined for configuration space

5. **Genus** ($g$)
   - For 2D surfaces: $\chi = 2 - 2g$ (orientable) or $\chi = 2 - g$ (non-orientable)
   - Characterizes "number of holes" in surfaces
   - Computable from Euler characteristic and orientability

6. **Chern Numbers** (for gauge bundles)
   - Topological charges from gauge field holonomy
   - Connection to Yang-Mills theory in the framework
   - Related to Chern-Gauss-Bonnet theorem

### Why Topological Invariants Matter

**Complementarity with Curvature**:
- **Curvature** (local): How space bends at each point
- **Topology** (global): Overall shape and connectivity of space
- Example: Flat torus has zero curvature everywhere but non-trivial topology ($\beta_1 = 2$)

**Physical Interpretation**:
- **$\beta_0$**: Number of disconnected components in the walker configuration (⚠️ use persistent homology, not Delaunay χ, for landscape basin detection)
- **$\beta_1$**: Number of independent cycles in the configuration graph (exploration pathways)
- **$\chi$**: Combinatorial invariant of walker triangulation; related to total curvature via Gauss-Bonnet
- **Genus**: Surface classification (requires persistent homology or α-complexes for landscape topology)

**Algorithmic Applications**:
- Detect phase transitions when $\beta_k$ changes (topological data analysis)
- Identify bottlenecks in state space exploration ($\beta_1$ changes)
- Validate topological consistency via Euler characteristic
- Measure "ruggedness" of fitness landscape via persistent homology

### Document Structure

- **Part 1** ({ref}`part-euler-characteristic`): Euler characteristic from Delaunay triangulation
- **Part 2** ({ref}`part-betti-numbers`): Betti numbers and persistent homology
- **Part 3** ({ref}`part-homology-groups`): Homology groups and boundary operators
- **Part 4** ({ref}`part-homotopy-groups`): Homotopy groups and fundamental group
- **Part 5** ({ref}`part-genus-classification`): Genus and surface classification
- **Part 6** ({ref}`part-chern-numbers`): Chern numbers and gauge topology
- **Part 7** ({ref}`part-computational-methods`): Computational algorithms and libraries
- **Part 8** ({ref}`part-framework-connections`): Cross-references to existing framework results

---

(part-euler-characteristic)=
## Part 1: Euler Characteristic

### 1.1. Definition and Computation

The **Euler characteristic** is the most fundamental topological invariant, relating combinatorial structure to global topology.

:::{prf:definition} Euler Characteristic (Simplicial Complex)
:label: def-euler-characteristic

For a simplicial complex $K$ with $V$ vertices, $E$ edges, $F$ faces (2-simplices), and $T$ tetrahedra (3-simplices) in dimension $d=3$, the **Euler characteristic** is:

$$
\chi(K) := V - E + F - T

$$

More generally, for a $d$-dimensional simplicial complex:

$$
\chi(K) := \sum_{k=0}^{d} (-1)^k n_k

$$

where $n_k$ is the number of $k$-simplices.

**Fundamental Property**: $\chi$ is a **topological invariant** — it is unchanged under homeomorphisms (continuous deformations with continuous inverse).
:::

:::{prf:theorem} Euler Characteristic from Delaunay Triangulation
:label: thm-euler-delaunay

For the walker configuration $S_t = \{x_1, \ldots, x_N\}$ at time $t$, let $\text{DT}(S_t)$ be the Delaunay triangulation. The Euler characteristic of the triangulated domain is:

$$
\chi_t := V_t - E_t + F_t - T_t

$$

where:
- $V_t = N$: Number of walkers (vertices)
- $E_t$: Number of Delaunay edges
- $F_t$: Number of Delaunay faces
- $T_t$: Number of Delaunay tetrahedra (for $d=3$)

**CRITICAL LIMITATION**: The Delaunay triangulation of a finite point cloud is homeomorphic to the **convex hull** of those points, NOT necessarily the underlying fitness landscape topology. For a point cloud in $\mathbb{R}^3$, the Delaunay complex of the convex hull has $\chi = 1$ regardless of whether the sampled surface has genus $g > 0$. This formula computes the Euler characteristic of the **triangulated point cloud**, not the **manifold being sampled**.

**To capture landscape topology**: Use α-complexes with appropriate radius thresholds or persistent homology (see § 2.3) to recover the topology of the underlying space from the sample. The Nerve Lemma guarantees that sufficiently dense sampling with appropriate filtration recovers the correct homotopy type.

**Valid interpretations of Delaunay $\chi_t$**:
1. **Combinatorial complexity**: Number of simplices in the triangulation (algorithmic bookkeeping)
2. **Gauss-Bonnet verification**: Check $\sum_i \delta_i = 2\pi \chi_t$ for internal consistency
3. **NOT valid**: Inferring genus, detecting topological phase transitions, or measuring landscape topology directly from Delaunay $\chi$ alone

**Computational Complexity**: $O(|\text{simplices}|)$ for scanning all simplices in the triangulation. In $\mathbb{R}^d$, a Delaunay triangulation of $N$ points contains $O(N)$ simplices for $d=2$, $O(N)$ for $d=3$ (typically), but up to $\Theta(N^{\lceil d/2 \rceil})$ in the worst case. For practical walker configurations in $\mathbb{R}^3$, the simplex count is usually $O(N)$, making the Euler characteristic computation linear in the number of walkers.

**Framework Status**: Already computed in the Fragile framework via the Delaunay triangulation maintained for curvature computations. This provides the Euler characteristic of the **walker point cloud triangulation**, not the fitness landscape manifold.
:::

:::{prf:remark} Euler Characteristic for Different Dimensions
:label: rem-euler-dimensions

**For $d=2$** (triangulated surfaces):

$$
\chi = V - E + F

$$

**Standard values**:
- Sphere: $\chi = 2$
- Torus: $\chi = 0$
- Klein bottle: $\chi = 0$
- Disk: $\chi = 1$

**For $d=4$** (4D simplicial complexes):

$$
\chi = V - E + F - T + P

$$

where $P$ is the number of 4-simplices (pentatopes).
:::

### 1.2. Connection to Gauss-Bonnet Theorem

The Euler characteristic is deeply connected to curvature via the **Gauss-Bonnet theorem**, which bridges local geometry and global topology.

:::{prf:theorem} Discrete Gauss-Bonnet Theorem
:label: thm-discrete-gauss-bonnet-topology

For a **closed** triangulated polyhedral surface $P$ (simplicial 2-complex **without boundary**), the sum of deficit angles (see [curvature.md](curvature.md) § 1.1) equals:

$$
\sum_{i \in \text{vertices}} \delta_i = 2\pi \chi(P)

$$

where $\delta_i$ is the deficit angle at vertex $i$ and $\chi(P)$ is the Euler characteristic.

**Boundary case**: For surfaces with boundary $\partial M$, the formula becomes:

$$
\sum_{i \in \text{vertices}} \delta_i + \int_{\partial M} \kappa_g \, ds = 2\pi \chi(M)

$$

where $\kappa_g$ is the geodesic curvature along the boundary curves and $ds$ is the arc length element. See [curvature.md](curvature.md) § 1.1 for full treatment of boundary contributions.

**Consequence**: The total integrated Gaussian curvature is determined by topology:

$$
\int_M K \, dA = 2\pi \chi(M)

$$

This is the **classical Gauss-Bonnet theorem** for smooth surfaces.
:::

:::{prf:remark} Euler Characteristic and Scutoid Cells
:label: rem-euler-characteristic-scutoid-connection

Each individual scutoid cell $\mathcal{S}$ is topologically homeomorphic to a **3-ball** (a convex polyhedron), and therefore has Euler characteristic:

$$
\chi(\mathcal{S}) = 1

$$

regardless of its combinatorial structure (number of faces, mid-level vertices, etc.).

The **scutoid index** $\chi_{\text{scutoid}}$ defined in [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) is a **separate topological invariant** that measures cloning activity at mid-level vertices, not the Euler characteristic of the cell itself. See {prf:ref}`prop-euler-characteristic-scutoid` for the relationship between $\chi_{\text{scutoid}}$ and the combinatorial structure of scutoid faces.

**Critical**: The Euler characteristic of the entire spacetime tessellation is **not** the sum of individual cell Euler characteristics. The global Euler characteristic must be computed directly from the full Delaunay triangulation using {prf:ref}`thm-euler-delaunay`:

$$
\chi_{\text{total}}(t) = V_t - E_t + F_t - T_t

$$

where the counts are taken over the entire simplicial complex. The Euler characteristic is not additive over individual cells due to shared faces, edges, and vertices between cells.
:::

---

(part-betti-numbers)=
## Part 2: Betti Numbers and Persistent Homology

### 2.1. Betti Numbers: Counting Topological Features

**Betti numbers** count independent topological features at each dimension.

:::{prf:definition} Betti Numbers
:label: def-betti-numbers

For a topological space $X$ (or simplicial complex $K$), the **$k$-th Betti number** is:

$$
\beta_k(X) := \text{rank}(H_k(X))

$$

where $H_k(X)$ is the $k$-th homology group (see Part 3).

**Interpretation**:
- $\beta_0$: Number of **connected components**
- $\beta_1$: Number of independent **1-dimensional holes** (loops, tunnels)
- $\beta_2$: Number of independent **2-dimensional voids** (cavities)
- $\beta_k$: Number of independent **$k$-dimensional holes**

**Relation to Euler characteristic**:

$$
\chi(X) = \sum_{k=0}^{\dim(X)} (-1)^k \beta_k(X)

$$
:::

:::{prf:example} Betti Numbers for Standard Spaces
:label: ex-betti-standard

1. **Sphere $S^2$**:
   - $\beta_0 = 1$ (connected)
   - $\beta_1 = 0$ (no loops)
   - $\beta_2 = 1$ (one void)
   - Euler characteristic: $\chi = 1 - 0 + 1 = 2$ ✓

2. **Torus $T^2$**:
   - $\beta_0 = 1$ (connected)
   - $\beta_1 = 2$ (two independent loops: meridian and parallel)
   - $\beta_2 = 1$ (one void)
   - Euler characteristic: $\chi = 1 - 2 + 1 = 0$ ✓

3. **Figure-eight $S^1 \vee S^1$**:
   - $\beta_0 = 1$ (connected)
   - $\beta_1 = 2$ (two independent loops)
   - Euler characteristic: $\chi = 1 - 2 = -1$
:::

### 2.2. Computation from Delaunay Triangulation

:::{prf:theorem} Betti Numbers from Fractal Set 2-Complex
:label: thm-betti-fractal-set

The Fractal Set is formally defined as a **2-dimensional simplicial complex** $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}}, P)$ where $P$ is the set of plaquettes (CST-IG 4-cycles), as specified in {prf:ref}`def-fractal-set-simplicial-complex` from [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md).

::::{important} **Cellular Homology Required**
The 2-cells (plaquettes) are **quadrilaterals** (CST-IG 4-cycles), NOT triangles (2-simplices). Therefore, the Fractal Set is technically a **CW complex** (or cellular complex), not a simplicial complex in the strict sense.

**Consequences**:
1. Standard simplicial boundary operators $\partial_k$ do NOT apply directly without modification
2. Must use **cellular homology** with boundary operators defined for arbitrary cell attachments
3. Alternatively, **triangulate each plaquette** via barycentric subdivision to obtain a true simplicial complex

The formulas below assume either:
- (A) Cellular boundary operators properly defined for quadrilateral 2-cells, OR
- (B) Barycentric subdivision has been performed to triangulate all plaquettes

**Status**: The framework document [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md) defines the 2-complex structure but **does not provide explicit cellular boundary operators** for quadrilateral plaquettes. The boundary maps $\partial_1$ and $\partial_2$ are structurally implied by the CW complex definition but not computed. Implementation of $\beta_1, \beta_2$ computation requires either:
1. Explicitly defining cellular boundary operators (mathematically straightforward but not yet done), OR
2. Triangulating each plaquette via barycentric subdivision to obtain a simplicial complex where standard $\partial_k$ formulas apply
::::

**$\beta_0$: Connected Components**

$$
\beta_0 = \text{number of connected components in } \mathcal{F}

$$

Computed via depth-first search (DFS) or union-find in $O(|\mathcal{E}| + |E|)$ on the 1-skeleton.

**$\beta_1$: Independent Cycles (1-dimensional holes)**

For the full 2-complex, the first Betti number is defined as:

$$
\beta_1 = \dim(\ker \partial_1) - \dim(\text{im } \partial_2)

$$

where $\partial_1: C_1 \to C_0$ and $\partial_2: C_2 \to C_1$ are the boundary operators.

**Coefficient field**: When working over a **field** $\mathbb{F}$ (e.g., $\mathbb{Z}_2$, $\mathbb{Q}$), the dimension formula above applies directly. Working over $\mathbb{Z}_2$ is computationally efficient (bitwise XOR operations) and sufficient for computing Betti numbers (counts of topological holes), but it loses torsion information.

To capture **torsion** (finite-order elements in homology groups), work over the integers $\mathbb{Z}$. However, $\mathbb{Z}$ is not a field, so "dimension" is replaced by "rank" and the Smith Normal Form algorithm is required (see {prf:ref}`alg-boundary-matrix` for the full integer homology workflow).

**Computational formula via rank-nullity theorem**: Using $\dim(C_k)$ to denote the dimension of the $k$-chain group:

$$
\beta_1 = \dim(C_1) - \text{rank}(\partial_1) - \text{rank}(\partial_2)

$$

For the Fractal Set 2-complex:
- **Vertex set**: $V := \mathcal{E}$ (episodes from Fractal Set)
- **Edge set**: $E' := \{\{u, v\} \mid (u, v) \in E_{\text{CST}} \cup E_{\text{IG}} \text{ or } (v, u) \in E_{\text{CST}} \cup E_{\text{IG}}\}$ (undirected 1-skeleton)
- **Plaquette set**: $P$ is the set of CST-IG 4-cycles (2-cells)

Then $\dim(C_1) = |E'|$, and:

$$
\beta_1 = |E'| - \text{rank}(\partial_1) - \text{rank}(\partial_2)

$$

**1-skeleton upper bound**: If we ignore the 2-complex structure and treat the Fractal Set as a graph (1-skeleton only), we get an **upper bound** by setting $\text{rank}(\partial_2) = 0$:

$$
\beta_1^{\text{graph}} = |E'| - \text{rank}(\partial_1) = |E'| - (|\mathcal{E}| - \beta_0)

$$

This simplifies to $\beta_1^{\text{graph}} = |E'| - |\mathcal{E}| + \beta_0$, which overcounts cycles that bound plaquettes.

**Critical distinction**: The correct formula requires computing $\text{rank}(\partial_2)$, which is done via boundary matrix reduction (see {prf:ref}`alg-boundary-matrix`). The graph formula is computationally cheaper but mathematically gives only an upper bound: $\beta_1 \le \beta_1^{\text{graph}}$.

**Framework Status**: The full 2-complex structure is defined in {prf:ref}`def-fractal-set-simplicial-complex`, but **cellular boundary operators are not explicitly provided**. Computational algorithms for $\beta_1$ including plaquette contributions require defining these operators. Currently, only the 1-skeleton upper bound $\beta_1^{\text{graph}}$ is computable via graph algorithms.
:::

:::{prf:remark} Physical Interpretation in Fragile Gas
:label: rem-betti-physical

**$\beta_0$ (Connected Components)**:
- Number of **disconnected fitness basins** explored by the swarm
- Phase transition indicator: $\beta_0$ decreases when walkers merge into a single basin
- Related to **multimodal optimization** performance

**$\beta_1$ (Independent Cycles)**:
- Number of **independent exploration pathways** in the configuration graph
- High $\beta_1$ → rich connectivity, many alternative paths
- Low $\beta_1$ → bottlenecks, sparse connectivity
- Connection to **propagation of chaos**: mean-field limit homogenizes cycles

**Example**: If walkers split into two disjoint groups exploring separate basins, $\beta_0 = 2$. If they later reconnect via cloning, $\beta_0$ drops to 1 (topological transition).
:::

### 2.3. Persistent Homology and TDA

**Persistent homology** tracks how Betti numbers change across multiple scales, revealing the "lifetime" of topological features.

:::{prf:definition} Persistent Homology (Filtration)
:label: def-persistent-homology

A **filtration** is a nested sequence of simplicial complexes:

$$
K_0 \subseteq K_1 \subseteq K_2 \subseteq \cdots \subseteq K_m

$$

indexed by a scale parameter (e.g., radius $r$ in Vietoris-Rips complex).

For each $K_i$, compute Betti numbers $\beta_k^{(i)}$. A topological feature (connected component, loop, void) that appears at scale $r_{\text{birth}}$ and disappears at $r_{\text{death}}$ has **persistence**:

$$
\text{persistence} := r_{\text{death}} - r_{\text{birth}}

$$

**Persistent homology** is the collection of all (birth, death) pairs across scales, typically visualized as a **persistence diagram** or **barcode**.

**Key Insight**: Features with high persistence are "real" topological structure; features with low persistence are noise.
:::

:::{prf:definition} Vietoris-Rips Complex (for TDA)
:label: def-vietoris-rips

For a point cloud $\{x_1, \ldots, x_N\} \subset \mathbb{R}^d$ and radius $r > 0$, the **Vietoris-Rips complex** $\text{VR}(r)$ is the simplicial complex where:

- $k+1$ points $\{x_{i_0}, \ldots, x_{i_k}\}$ form a $k$-simplex if and only if:

$$
d(x_i, x_j) \le r \quad \text{for all } i, j \in \{i_0, \ldots, i_k\}

$$

**Filtration**: As $r$ increases from 0 to $\infty$, we get a filtration:

$$
\text{VR}(0) \subseteq \text{VR}(r_1) \subseteq \text{VR}(r_2) \subseteq \cdots \subseteq \text{VR}(\infty)

$$

**Computational note**: Vietoris-Rips is easier to compute than Čech complex (only pairwise distances needed) but may create spurious higher-dimensional simplices. Theoretically, the Čech complex is often preferred as the **Nerve Lemma** guarantees that it is homotopy equivalent to the union of balls, faithfully capturing the topology of the underlying continuous space.

However, the VR complex also has strong theoretical guarantees: **Hausmann's theorem** establishes that for a finite metric space with sufficiently dense sampling, the Vietoris-Rips complex is homotopy equivalent to the Čech complex at appropriate scales, and both capture the topology of the underlying space. This, combined with its computational efficiency, makes the VR complex the standard choice in practice for persistent homology.
:::

:::{prf:algorithm} Computing Persistent Homology for Fragile Gas
:label: alg-persistent-homology

**Input**: Walker positions $\{x_1(t), \ldots, x_N(t)\}$ at time $t$

**Output**: Persistence diagrams for $H_0, H_1, H_2$

**Steps**:

1. **Construct filtration**: Build Vietoris-Rips filtration $\{\text{VR}(r_i)\}_{i=0}^m$ with radii $0 = r_0 < r_1 < \cdots < r_m$

2. **Compute boundary matrices**: For each $k \in \{0, 1, 2\}$, construct boundary operator $\partial_k: C_k \to C_{k-1}$ at each scale

3. **Matrix reduction**: Apply persistent homology algorithm (e.g., standard reduction, twist reduction, or clearing optimization)

4. **Extract persistence pairs**: Identify (birth, death) pairs for each homology class

5. **Filter by persistence**: Remove short-lived features (noise) by thresholding:

$$
\text{keep feature if } r_{\text{death}} - r_{\text{birth}} > \epsilon_{\text{pers}}

$$

**Complexity**: $O(s^3)$ where $s$ is the total number of simplices in the final complex of the filtration. **Important**: For a Vietoris-Rips complex on $N$ points with maximum dimension $k_{\max}$, the number of simplices is bounded by $s \le \sum_{k=0}^{k_{\max}} \binom{N}{k+1} = O(N^{k_{\max}+1})$. Thus the complexity is $O(N^{3(k_{\max}+1)})$ in the worst case, which is polynomial in $N$ for fixed $k_{\max}$ but exponential for large $k_{\max}$.

**Feasibility assumption**: The worst-case complexity is $O(s^3)$ where $s$ can be exponential in $N$ for high-dimensional data. In practice, libraries such as **Ripser** exploit sparsity and geometric structure of many real-world point clouds to achieve much better performance, but there is **no general near-linear guarantee**. The complexity is practical only under the assumption that the walker point cloud has **well-behaved geometric structure** (e.g., low intrinsic dimension, sparse distribution, or clustered configuration). For arbitrary or adversarially-chosen point configurations in high dimensions, the method becomes computationally infeasible. For large $N$ (>1000), use sparse matrix methods, dimension capping ($k \le 2$), or approximate algorithms.

**Reference**: Bauer et al., "Ripser: Efficient computation of Vietoris-Rips persistence barcodes" (2021) — practical optimizations for geometric data.

**Libraries**: Use existing TDA libraries:
- **Python**: `gudhi`, `ripser`, `scikit-tda`
- **C++**: `GUDHI`, `Ripser`, `PHAT`
- **Julia**: `Ripserer.jl`, `PersistenceDiagrams.jl`
:::

---

(part-homology-groups)=
## Part 3: Homology Groups

### 3.1. Formal Definition

Homology groups provide the full algebraic structure underlying Betti numbers.

:::{prf:definition} Simplicial Homology
:label: def-simplicial-homology

For a simplicial complex $K$, define:

**Chain groups**: $C_k(K)$ is the free abelian group generated by $k$-simplices:

$$
C_k(K) := \bigoplus_{\sigma \in K_k} \mathbb{Z} \cdot \sigma

$$

where $K_k$ is the set of all $k$-simplices.

**Boundary operator**: $\partial_k: C_k \to C_{k-1}$ is the linear map:

$$
\partial_k([v_0, v_1, \ldots, v_k]) := \sum_{i=0}^{k} (-1)^i [v_0, \ldots, \hat{v}_i, \ldots, v_k]

$$

where $\hat{v}_i$ means omit vertex $v_i$.

**Fundamental property**: $\partial_{k-1} \circ \partial_k = 0$ (boundary of boundary is zero)

**Homology groups**:

$$
H_k(K) := \frac{\ker(\partial_k)}{\text{im}(\partial_{k+1})} = \frac{Z_k}{B_k}

$$

where:
- $Z_k := \ker(\partial_k)$ is the group of **$k$-cycles** (closed chains)
- $B_k := \text{im}(\partial_{k+1})$ is the group of **$k$-boundaries** (exact chains)

**Betti number recovery**:

$$
\beta_k := \text{rank}(H_k(K; \mathbb{Z}))

$$

This is the rank of the free part of the homology group $H_k(K)$ over integers $\mathbb{Z}$. For homology with coefficients in a field $\mathbb{F}$ (e.g., $\mathbb{Q}$ or $\mathbb{Z}_p$):

$$
\beta_k = \dim(H_k(K; \mathbb{F})) = \dim(Z_k) - \dim(B_k)

$$

where the dimensions are over the field $\mathbb{F}$. Working over a field simplifies computation but loses torsion information (finite-order elements in the homology group).
:::

:::{prf:example} Homology Computation for Triangle
:label: ex-homology-triangle

Consider a triangle $K$ with vertices $\{a, b, c\}$ and edges $\{[a,b], [b,c], [c,a]\}$ but **no 2-simplex** (hollow triangle).

**$H_0$ (0-dimensional homology)**:
- $C_0 = \langle a, b, c \rangle \cong \mathbb{Z}^3$
- $\partial_1([a,b]) = b - a$, $\partial_1([b,c]) = c - b$, $\partial_1([c,a]) = a - c$
- $Z_0 = C_0$ (all 0-chains are cycles)
- $B_0 = \text{span}\{b - a, c - b\} \cong \mathbb{Z}^2$
- $H_0 = \mathbb{Z}^3 / \mathbb{Z}^2 \cong \mathbb{Z}$ → $\beta_0 = 1$ (one connected component) ✓

**$H_1$ (1-dimensional homology)**:
- $C_1 = \langle [a,b], [b,c], [c,a] \rangle \cong \mathbb{Z}^3$
- $Z_1 = \ker(\partial_1) = \text{span}\{[a,b] + [b,c] + [c,a]\} \cong \mathbb{Z}$ (the loop)
- $B_1 = \{0\}$ (no 2-simplices, so no boundaries from higher dimension)
- $H_1 = Z_1 / B_1 \cong \mathbb{Z}$ → $\beta_1 = 1$ (one independent loop) ✓

If we **fill** the triangle (add 2-simplex $[a,b,c]$):
- $\partial_2([a,b,c]) = [b,c] - [a,c] + [a,b]$ (the loop is now a boundary)
- $B_1 = Z_1$ → $H_1 = 0$ → $\beta_1 = 0$ (no holes) ✓
:::

### 3.2. Computation from Fractal Set

:::{prf:algorithm} Boundary Matrix Method for Homology
:label: alg-boundary-matrix

**Input**: Simplicial complex $K$ (e.g., Delaunay triangulation of walkers)

**Output**: Betti numbers $\beta_0, \beta_1, \ldots, \beta_d$

**Steps**:

1. **Enumerate simplices**: List all $k$-simplices for $k = 0, 1, \ldots, d$

2. **Construct boundary matrices**: For each $k$, build matrix $\partial_k$ where:
   - Rows index $(k-1)$-simplices
   - Columns index $k$-simplices
   - Entry $(i, j)$ is the coefficient of simplex $i$ in $\partial_k(\text{simplex } j)$

3. **Compute ranks (over a field)**:
   - To compute Betti numbers over a field $\mathbb{F}$ (which is sufficient for counting holes), use Gaussian elimination:
   - $\text{rank}_\mathbb{F}(\partial_k)$ via row reduction over $\mathbb{F}$
   - $\dim(Z_k) = \dim(C_k) - \text{rank}_\mathbb{F}(\partial_k)$
   - $\dim(B_k) = \text{rank}_\mathbb{F}(\partial_{k+1})$

4. **Betti numbers**:

$$
\beta_k = \dim(Z_k) - \dim(B_k) = \dim(C_k) - \text{rank}_\mathbb{F}(\partial_k) - \text{rank}_\mathbb{F}(\partial_{k+1})

$$

**For full integer homology** (including torsion): Use **Smith Normal Form** on the boundary matrices $\partial_k$ over $\mathbb{Z}$. This gives the torsion coefficients as well as the free rank.

**Complexity**: $O(n^3)$ where $n$ is the total number of cells (dominated by matrix reduction via Gaussian elimination over a field). Smith Normal Form over $\mathbb{Z}$ for full integer homology has comparable or worse complexity but provides torsion information. Fast matrix multiplication algorithms (ω < 3) do not apply to sparse boundary matrices in practical homology computations.

**Optimization**: Use sparse matrix methods; boundary matrices are extremely sparse for Delaunay triangulations.
:::

:::{prf:remark} Cohomology vs Homology
:label: rem-cohomology-vs-homology

**Cohomology** groups $H^k(K)$ are the dual of homology:

$$
H^k(K) := \frac{\ker(\delta^k)}{\text{im}(\delta^{k-1})}

$$

where $\delta^k: C^k \to C^{k+1}$ is the **coboundary operator** on cochains $C^k := \text{Hom}(C_k, \mathbb{Z})$.

**Why cohomology matters**:
- Cohomology has a **ring structure** (cup product) → richer algebraic invariants
- More natural for differential forms and de Rham cohomology
- Connection to gauge theory: cohomology classes classify gauge bundles

**For Fragile framework**:
- Homology is typically sufficient for counting topological features
- Cohomology becomes relevant for gauge-theoretic formulations (see Part 6)

**Framework Status**: Cohomology groups are defined in [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md) § 3 for the Fractal Set as a 2-dimensional simplicial complex.
:::

---

(part-homotopy-groups)=
## Part 4: Homotopy Groups

### 4.1. Fundamental Group

The **fundamental group** $\pi_1(X, x_0)$ captures information about loops that cannot be detected by homology alone.

:::{prf:definition} Fundamental Group
:label: def-fundamental-group

For a pointed topological space $(X, x_0)$, the **fundamental group** $\pi_1(X, x_0)$ is the group of homotopy classes of loops based at $x_0$:

$$
\pi_1(X, x_0) := \{\text{loops } \gamma: [0,1] \to X \text{ with } \gamma(0) = \gamma(1) = x_0\} / \sim

$$

where $\gamma_1 \sim \gamma_2$ if $\gamma_1$ can be continuously deformed to $\gamma_2$ (homotopy).

**Group operation**: Concatenation of loops

$$
(\gamma_1 \cdot \gamma_2)(t) := \begin{cases}
\gamma_1(2t) & 0 \le t \le 1/2 \\
\gamma_2(2t - 1) & 1/2 \le t \le 1
\end{cases}

$$

**Identity**: Constant loop at $x_0$

**Inverse**: Reverse loop $\gamma^{-1}(t) := \gamma(1 - t)$
:::

:::{prf:theorem} Hurewicz Theorem (Abelianization)
:label: thm-hurewicz

There is a natural homomorphism (Hurewicz map):

$$
h: \pi_1(X, x_0) \to H_1(X)

$$

from the fundamental group to the first homology group. This map is **abelianization** — it quotients out the non-abelian structure:

$$
H_1(X) \cong \pi_1(X, x_0) / [\pi_1, \pi_1]

$$

where $[\pi_1, \pi_1]$ is the commutator subgroup.

**Consequence**: Homology $H_1$ counts independent loops but forgets their order. Fundamental group $\pi_1$ retains non-commutative information.

**Example**: Torus has $\pi_1(T^2) \cong \mathbb{Z} \times \mathbb{Z}$ (abelian) and $H_1(T^2) \cong \mathbb{Z}^2$ (isomorphic in this case). But the figure-eight has $\pi_1(S^1 \vee S^1) \cong \mathbb{Z} * \mathbb{Z}$ (free group, non-abelian) while $H_1(S^1 \vee S^1) \cong \mathbb{Z}^2$ (abelian).
:::

### 4.2. Homotopy Groups in the Fragile Framework

:::{prf:theorem} Configuration Space Homotopy (Product Space)
:label: thm-homotopy-configuration-space

For the Fragile Gas configuration space $\mathcal{X}^N = \mathcal{X} \times \cdots \times \mathcal{X}$ ($N$ copies), the homotopy groups satisfy:

$$
\pi_k(\mathcal{X}^N) \cong \prod_{i=1}^N \pi_k(\mathcal{X})

$$

where the right-hand side is the direct product of $N$ copies of $\pi_k(\mathcal{X})$.

**For $k \ge 2$** (higher homotopy groups are abelian), this simplifies to:

$$
\pi_k(\mathcal{X}^N) \cong \bigoplus_{i=1}^N \pi_k(\mathcal{X}) \cong \pi_k(\mathcal{X})^N

$$

using the direct sum notation for abelian groups.

**Proof**: This follows from the universal property of the product topology. A map $f: S^k \to \mathcal{X}^N$ is continuous if and only if its component maps $f_i = p_i \circ f: S^k \to \mathcal{X}$ are continuous, where $p_i: \mathcal{X}^N \to \mathcal{X}$ is the projection onto the $i$-th factor. Similarly, a homotopy $H: S^k \times [0,1] \to \mathcal{X}^N$ is continuous if and only if its component homotopies $H_i = p_i \circ H$ are continuous. This establishes a one-to-one correspondence between homotopy classes of maps into the product space and the product of the sets of homotopy classes of maps into each factor: $[S^k, \mathcal{X}^N] \cong \prod_{i=1}^N [S^k, \mathcal{X}]$. The group structure is defined component-wise, yielding the isomorphism of homotopy groups. $\square$

**Physical Interpretation**: The configuration space $\mathcal{X}^N$ itself remains a **fixed topological space** throughout the algorithm's execution, with homotopy groups $\pi_k(\mathcal{X}^N) \cong \pi_k(\mathcal{X})^N$ as stated above. The **dynamics** of the Fragile Gas, however, do not explore this full space uniformly. When a cloning event occurs (setting $x_j \leftarrow x_i$), the swarm **state** is instantaneously mapped to the **diagonal submanifold** $\Delta_{ij} := \{w \in \mathcal{X}^N \mid w_i = w_j\}$, which is homeomorphic to $\mathcal{X}^{N-1}$. Crucially, this is a statement about where the state **is** at that instant, not a change to the ambient space itself. The subsequent dynamics (kinetic operator, fitness forces) evolve within $\mathcal{X}^N$, potentially moving off the diagonal. Thus, cloning and selection constrain the **accessible regions** of the configuration space along lower-dimensional submanifolds, but the topology of the **space** $\mathcal{X}^N$ and its homotopy groups remain invariant. The algorithm's trajectory traces a path through various such submanifolds embedded in the unchanging topological structure of $\mathcal{X}^N$.

**Framework Status**: This is a standard result in algebraic topology for product spaces. See Hatcher, "Algebraic Topology" (2002), §4.1, for the general theory. The application to the Fragile framework is mentioned in [00_reference.md](00_reference.md) (search for "homotopy groups" in entry index).
:::

:::{prf:remark} Accessible State Space Topology
:label: rem-accessible-state-space

The **accessible state space** $\mathcal{A} \subset \mathcal{X}^N$ is the subset of configuration space that the Fragile Gas algorithm can actually reach through its dynamics (kinetic operator + cloning + selection). This is a strict subset of the full product space $\mathcal{X}^N$ due to:

1. **Cloning constraints**: After cloning events, the state lies on diagonal submanifolds $\Delta_{ij} = \{w \in \mathcal{X}^N \mid w_i = w_j\}$
2. **Finite-time reachability**: Only states within a bounded distance (in the algorithmic metric) from the initial condition are accessible
3. **Selection barriers**: Low-fitness regions may be inaccessible due to cloning thresholds

The accessible space $\mathcal{A}$ is **conjectured to have a stratified structure** that can be decomposed into manifolds of varying dimensions, though this has not been rigorously proven within the framework. Its topology is far more complex than $\mathcal{X}^N$ and depends on:
- The cloning history (which walkers have been cloned)
- The fitness landscape structure (which regions are accessible)
- The algorithmic parameters (temperature, cloning threshold, etc.)

**Open Problem**: Characterize the homotopy groups $\pi_k(\mathcal{A})$ of the accessible state space. These may differ significantly from $\pi_k(\mathcal{X}^N)$ and provide deeper insight into the algorithm's exploration capabilities.

**Practical Implication**: While the ambient space $\mathcal{X}^N$ has trivial higher homotopy (if $\mathcal{X}$ is contractible), the accessible space $\mathcal{A}$ may have non-trivial topology due to fitness barriers creating "holes" in the reachable region. This topological obstruction can prevent the algorithm from reaching global optima.

**Future Work**: A rigorous analysis of $\mathcal{A}$'s topology would require:
- Formal definition of the accessible set as the forward-reachable set from initial conditions
- **Proof** that $\mathcal{A}$ admits a stratified manifold decomposition (currently conjectured but unproven)
- Tools from stratified space topology to characterize the stratum structure
- Connection to Morse theory (fitness landscape critical points create topological features in $\mathcal{A}$)
- Analysis of singularities and non-smooth regions induced by cloning/selection discontinuities
:::

:::{prf:remark} When Homotopy Matters More Than Homology
:label: rem-homotopy-vs-homology

**Homology is easier to compute** (linear algebra) but **loses information**:
- Homology: Abelian groups, counts holes
- Homotopy: Potentially non-abelian groups, tracks loop structure

**Example**: The **Klein bottle** $K$ has:
- $H_1(K) \cong \mathbb{Z} \oplus \mathbb{Z}_2$ (first homology)
- $\pi_1(K) \cong \langle a, b \mid aba^{-1}b = 1 \rangle$ (non-abelian presentation)

The homotopy group reveals the **twist** in the Klein bottle that homology cannot detect.

**For Fragile framework**: Homology is typically sufficient for practical computations. Homotopy groups become relevant for:
- Detecting non-trivial loop structures in high-dimensional state spaces
- Verifying topological properties of gauge bundles (see Part 6)
- Understanding obstruction theory for continuous extensions
:::

---

(part-genus-classification)=
## Part 5: Genus and Surface Classification

### 5.1. Genus from Euler Characteristic

For 2-dimensional surfaces, the **genus** is the "number of holes" and completely determines the topology (up to orientability).

:::{prf:definition} Genus of a Surface
:label: def-genus

For a compact, connected, **orientable** surface $\Sigma$ without boundary, the **genus** $g$ is related to the Euler characteristic by:

$$
\chi(\Sigma) = 2 - 2g

$$

**Standard surfaces**:
- Sphere: $\chi = 2$, $g = 0$ (no holes)
- Torus: $\chi = 0$, $g = 1$ (one hole)
- Double torus (pretzel): $\chi = -2$, $g = 2$ (two holes)
- $g$-holed torus: $\chi = 2 - 2g$

For **non-orientable** surfaces (e.g., Klein bottle, real projective plane):

$$
\chi(\Sigma) = 2 - g

$$

where $g$ is the **non-orientable genus** (number of crosscaps).

**Classification Theorem**: Any compact, connected surface without boundary is homeomorphic to either:
1. A sphere with $g$ handles (orientable genus $g$), or
2. A sphere with $g$ crosscaps (non-orientable genus $g$)
:::

:::{prf:theorem} Computing Genus from Triangulated 2-Manifold
:label: thm-genus-from-triangulation

::::{important} **Prerequisites: Actual 2-Manifold Required**
This algorithm applies ONLY to a genuine **2-dimensional combinatorial manifold** (a triangulated surface), NOT to the raw Delaunay triangulation of a 3D point cloud.

**Why**: The Delaunay triangulation of walkers in $\mathbb{R}^3$ produces a 3-dimensional simplicial complex (convex hull), not a 2-surface. To apply this genus computation, you must first:
1. Extract a 2-manifold via **α-complex** at appropriate radius (see § 2.3), OR
2. Use **persistent homology** to identify the correct scale where the 2-surface emerges

**See § 1.1 warnings**: The Delaunay Euler characteristic captures point cloud topology, not landscape topology.
::::

For a triangulated 2-surface $\Sigma$ with $V$ vertices, $E$ edges, and $F$ faces:

1. **Compute Euler characteristic**: $\chi = V - E + F$

2. **Determine orientability**: Check if the triangulation admits a consistent orientation using a **propagation-based algorithm**:

   **Preconditions**: The input simplicial complex must be a **combinatorial 2-manifold**, meaning:
   - For a **closed surface**: Every edge is shared by exactly two faces
   - For a **surface with boundary**: Boundary edges are shared by exactly one face; interior edges by exactly two faces

   If the complex has edges shared by more than two faces (pinch points) or other non-manifold features, the algorithm may fail or produce incorrect results. For robustness, verify the manifold property before running this algorithm.

   a. **Initialize**: Pick an arbitrary face $F_0$ and assign it an orientation (e.g., order its vertices as $(v_1, v_2, v_3)$). Mark $F_0$ as "oriented". Create a queue $Q := \{F_0\}$.

   b. **Propagate orientations**: While $Q$ is non-empty:
      - Dequeue a face $F$ from $Q$
      - For each edge $e$ of $F$, find the adjacent face $F'$ sharing edge $e$ (if it exists and is interior, i.e., not a boundary edge)
      - If $F'$ is not yet oriented:
        * Orient $F'$ so that the shared edge $e$ has **opposite orientation** in $F$ and $F'$ (if $F$ has edge oriented $(u,v)$, then $F'$ must have $(v,u)$)
        * Mark $F'$ as "oriented" and add it to $Q$
      - If $F'$ is already oriented:
        * Check if the shared edge $e$ has opposite orientations in $F$ and $F'$
        * If they have the **same orientation**, the surface is **non-orientable** → STOP

   c. If the algorithm completes without conflicts, the surface is **orientable**.

   **Important restriction**: This algorithm assumes the surface is **closed (without boundary)** or properly handles boundary edges (edges adjacent to only one face should be ignored in the consistency check). For surfaces with boundary, use the boundary-aware Euler characteristic formula: $\chi = 2 - 2g - b$ for orientable surfaces with $b$ boundary components, or $\chi = 2 - g - b$ for non-orientable surfaces.

   **Complexity**: $O(F)$ where $F$ is the number of faces (each face visited at most once). This is equivalent to $O(E)$ since $F = O(E)$ for a 2-complex.

   **Correctness**: A closed surface is orientable if and only if its triangulation admits a consistent orientation. The propagation algorithm tests for this by attempting to build such an orientation; a conflict indicates a "twist" characteristic of non-orientable surfaces (e.g., Möbius strip, Klein bottle).

3. **Compute genus**:
   - If orientable: $g = (2 - \chi) / 2$
   - If non-orientable: $g = 2 - \chi$

**Complexity**: $O(E)$ for orientation check, $O(1)$ for genus computation given $\chi$.

**Framework Application**: For 2D fitness landscapes embedded as surfaces in $\mathbb{R}^3$, this gives a topological classification of landscape complexity.
:::

### 5.2. Genus and Gauss-Bonnet

:::{prf:theorem} Gauss-Bonnet and Genus
:label: thm-gauss-bonnet-genus

For a compact orientable surface $\Sigma$ without boundary:

$$
\int_\Sigma K \, dA = 2\pi \chi(\Sigma) = 2\pi(2 - 2g) = 4\pi(1 - g)

$$

where $K$ is the Gaussian curvature.

**Physical Interpretation**:
- **Genus 0** (sphere): Total curvature is $4\pi > 0$ (positive on average)
- **Genus 1** (torus): Total curvature is $0$ (can be flat everywhere)
- **Genus $g > 1$**: Total curvature is $4\pi(1 - g) < 0$ (negative on average, hyperbolic)

**Framework Connection**: This relates the **topological invariant** (genus) to the **geometric quantity** (integrated curvature). See [curvature.md](curvature.md) § 1.2 for deficit angle formulation.
:::

---

(part-chern-numbers)=
## Part 6: Chern Numbers and Gauge Topology

### 6.1. Chern Classes and Characteristic Classes

**Chern numbers** are topological invariants of complex vector bundles, arising naturally in gauge theory.

:::{prf:definition} First Chern Class
:label: def-first-chern-class

For a complex line bundle $L \to M$ over a manifold $M$, the **first Chern class** is a cohomology class:

$$
c_1(L) \in H^2(M; \mathbb{Z})

$$

defined via the curvature 2-form $F$ of a connection on $L$:

$$
c_1(L) = \left[\frac{i F}{2\pi}\right] \in H^2(M; \mathbb{Z})

$$

where $[\cdot]$ denotes cohomology class.

**First Chern number**: For a compact surface $\Sigma$, integrate:

$$
c_1 := \int_\Sigma c_1(L) = \frac{1}{2\pi} \int_\Sigma F

$$

This is an **integer** (topological charge).
:::

:::{prf:theorem} Chern-Gauss-Bonnet Theorem
:label: thm-chern-gauss-bonnet

For a compact orientable Riemannian surface $(\Sigma, g)$ without boundary, the Euler characteristic is the integral of Gaussian curvature:

$$
\chi(\Sigma) = \frac{1}{2\pi} \int_\Sigma K \, dA

$$

Identifying the tangent bundle $T\Sigma$ with a complex line bundle (via an **almost complex structure**, which on an oriented surface allows one to define a consistent rotation by 90° in each tangent plane, effectively turning each real $\mathbb{R}^2$ into a copy of $\mathbb{C}$), the **first Chern number** equals the Euler characteristic:

$$
\int_\Sigma c_1(T\Sigma) = \chi(\Sigma)

$$

where $c_1(T\Sigma) \in H^2(\Sigma; \mathbb{Z})$ is the first Chern class (a cohomology class). The integral evaluates the class on the fundamental cycle $[\Sigma]$.

**Generalization**: For higher-dimensional compact orientable manifolds, the Euler characteristic is the integral of the **Euler class**:

$$
\chi(M) = \int_M e(TM) = \langle e(TM), [M] \rangle

$$

where $e(TM) \in H^{\dim(M)}(M; \mathbb{Z})$ is the Euler class and $[M]$ is the fundamental class.
:::

### 6.2. Chern Numbers in Yang-Mills Theory

:::{prf:definition} Yang-Mills Topological Charge
:label: def-yang-mills-topological-charge

For a Yang-Mills gauge field $A$ on a 4-dimensional manifold $M$ with field strength $F = dA + A \wedge A$, the **topological charge** (second Chern number) is:

$$
Q := \frac{1}{8\pi^2} \int_M \text{tr}(F \wedge F)

$$

This is the **instanton number**, an integer topological invariant.

**Physical Interpretation**: Counts the winding number of the gauge field configuration. In QCD, instantons are non-perturbative solutions with $Q \neq 0$.

**Framework Connection**: In [15_yang_mills/yang_mills_geometry.md](15_yang_mills/yang_mills_geometry.md), the Fragile framework's gauge structure admits Yang-Mills configurations. The discrete version uses **plaquette holonomy** (see [curvature.md](curvature.md) § 3 for Riemann tensor from plaquettes).
:::

:::{prf:algorithm} Naive Lattice Topological Charge Approximation
:label: alg-chern-from-plaquettes

**Input**: Gauge field $A$ on Fractal Set edges (lattice gauge field), lattice spacing $a$

**Output**: Approximate topological charge $Q_{\text{approx}}$ (discrete Chern number)

::::{warning} **Non-Integer Results for Non-Smooth Fields**
The formula presented below is a **first-order lattice approximation** that is **not guaranteed to produce integer values** for arbitrary gauge configurations. Topological charge is fundamentally an integer (instanton number), but this discrete formula will deviate from integer values for:
- **Strongly-coupled regimes** (large gauge coupling)
- **Rough or non-smooth gauge fields** (where $U_{\mu\nu}$ is far from the identity)
- **Coarse lattice spacing** (large $a$)

For generic lattice configurations, expect $Q_{\text{approx}} \notin \mathbb{Z}$. This is a **discretization artifact**, not a physical result. Only in the continuum limit ($a \to 0$) for sufficiently smooth field configurations does $Q_{\text{approx}} \to Q \in \mathbb{Z}$.

**For rigorous integer quantization** at fixed lattice spacing, use geometrically exact charge definitions (e.g., Lüscher's geometric charge, field-theoretic index theorem methods). See Lüscher, "Topology of Lattice Gauge Fields", Comm. Math. Phys. 85, 39 (1982).
::::

**Steps**:

1. **Compute plaquette holonomies**: For each oriented plaquette $P_{\mu\nu}(x)$ in the $\mu$-$\nu$ plane at lattice site $x$, compute:

$$
U_{\mu\nu}(x) := U_\mu(x) U_\nu(x + \hat{\mu}) U_\mu^\dagger(x + \hat{\nu}) U_\nu^\dagger(x)

$$

where $U_\mu(x) = \exp(ia A_\mu(x))$ is the link variable and $\hat{\mu}$ is the unit lattice vector.

2. **Approximate field strength tensor** (first-order discretization): For each lattice site $x$, compute:

$$
F_{\mu\nu}(x) := \frac{1}{2ia^2} \left[ U_{\mu\nu}(x) - U_{\mu\nu}^\dagger(x) - \frac{1}{N}\text{tr}\left(U_{\mu\nu}(x) - U_{\mu\nu}^\dagger(x)\right) \mathbf{1} \right]

$$

**Crucial—Lie algebra projection**: This formula **projects onto the Lie algebra** $\mathfrak{su}(N)$ by removing the trace (identity component). For non-abelian gauge groups, omitting this projection produces a matrix in $\mathfrak{gl}(N)$ instead of $\mathfrak{su}(N)$, breaking integer quantization. For U(1) (abelian) gauge theory, the trace term vanishes.

   **Critical—Approximation validity**: This formula is a **first-order approximation** to the exact field strength $F_{\mu\nu} = \frac{i}{a^2}\log(U_{\mu\nu})$. It is only valid for sufficiently smooth gauge fields where $U_{\mu\nu} \approx \mathbf{1} + O(a)$ (weak coupling, small lattice spacing). For strongly-coupled or rough configurations, this approximation breaks down and the resulting topological charge $Q_{\text{approx}}$ will deviate significantly from an integer, violating the fundamental principle that topological charge must be quantized. Use improved discretizations (e.g., clover-average, geometric charge definitions) for general configurations.

3. **Sum over hypercubes with Levi-Civita contraction**: For 4D Yang-Mills, the discrete topological charge is:

$$
Q_{\text{approx}} := \frac{1}{32\pi^2} \sum_{x \in \text{lattice}} \epsilon_{\mu\nu\rho\sigma} \text{tr}\left[ F_{\mu\nu}(x) F_{\rho\sigma}(x) \right]

$$

where $\epsilon_{\mu\nu\rho\sigma}$ is the Levi-Civita symbol and the sum runs over all lattice sites. Each term combines all six independent plaquette orientations at site $x$.

4. **Continuum limit and integer quantization**: As $a \to 0$ for a sequence of smooth fields, $Q_{\text{approx}} \to Q \in \mathbb{Z}$ (integer instanton number).

   **Critical caveat**: For **strongly-coupled or "rough" gauge configurations** where $U_{\mu\nu}$ is far from the identity, this first-order discrete formula can produce $Q_{\text{approx}}$ that deviates significantly from an integer, violating the principle of topological quantization. Only a geometrically exact charge definition (e.g., based on fiber bundle transition functions) guarantees integer-valuedness for all configurations at fixed lattice spacing. More sophisticated "improved" actions or geometric charge definitions are required to recover integer quantization in strongly-coupled regimes.

**Mathematical Note**: The key difference from naive plaquette summation is that the Levi-Civita tensor $\epsilon_{\mu\nu\rho\sigma}$ ensures gauge invariance by properly combining orthogonal plaquettes. A single plaquette $F_{\mu\nu}$ cannot form $F \wedge F$ alone; the wedge product requires contributions from all four spacetime directions.

**Framework Status**: **NOT IMPLEMENTED**. Plaquette holonomy machinery exists in [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) § 5 for computing the Riemann tensor, and gauge connections are defined in [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md) § 4. However, the **Levi-Civita ε-contraction over 4D hypercubes** (step 3 above) and the **geometric charge definition** required for integer quantization are **not yet implemented**. The existing plaquette infrastructure provides only the $F_{\mu\nu}$ components; assembling these into the full topological charge $Q$ via $\text{tr}(F \wedge F)$ remains to be done.

**References**:
- Lüscher, "Topology of Lattice Gauge Fields", Comm. Math. Phys. 85, 39 (1982) — geometric charge
- Montvay & Münster, "Quantum Fields on a Lattice" (1994), Chapter 4 — lattice QCD
- Wilson, "Confinement of quarks", Phys. Rev. D 10, 2445 (1974) — Wilson loops
:::

---

(part-computational-methods)=
## Part 7: Computational Methods and Libraries

### 7.1. Python Libraries for TDA

:::{prf:remark} Recommended Libraries
:label: rem-tda-libraries

**GUDHI** (Geometry Understanding in Higher Dimensions):
- Comprehensive persistent homology, Rips complexes, alpha complexes
- Python bindings to optimized C++ core
- Installation: `pip install gudhi`
- Documentation: [gudhi.inria.fr](http://gudhi.inria.fr)

**Ripser**:
- Ultra-fast persistent homology (sparse matrix optimizations)
- Python bindings: `pip install ripser`
- Ideal for large point clouds ($N > 1000$)
- GitHub: [Ripser](https://github.com/scikit-tda/ripser.py)

**scikit-tda**:
- High-level TDA toolkit built on Ripser
- Includes visualization, preprocessing, and machine learning integration
- Installation: `pip install scikit-tda`
- Includes **persim** for persistence diagram metrics

**Giotto-tda**:
- TDA for time series and machine learning pipelines
- scikit-learn compatible
- Installation: `pip install giotto-tda`

**Dionysus 2**:
- Zigzag persistence, multi-parameter persistence
- Python bindings: `pip install dionysus`
- Advanced features beyond standard persistent homology
:::

### 7.2. Integration with Fragile Framework

:::{prf:algorithm} TDA Pipeline for Fragile Gas
:label: alg-tda-pipeline

**Input**: Swarm trajectory $\{S_t\}_{t=0}^T$ where $S_t = \{x_1(t), \ldots, x_N(t)\}$

**Output**: Persistence diagrams, bottleneck distances, topological features

```python
import numpy as np
from ripser import ripser
from persim import plot_diagrams, bottleneck
import gudhi

def compute_tda_features(trajectory, max_radius=1.0, max_dimension=2):
    """
    Compute topological data analysis features for swarm trajectory.

    Parameters
    ----------
    trajectory : ndarray, shape (T, N, d)
        Walker positions over time
    max_radius : float
        Maximum filtration radius
    max_dimension : int
        Maximum homology dimension to compute

    Returns
    -------
    diagrams : list of ndarrays
        Persistence diagrams for each timestep
    features : dict
        Topological feature summary
    """
    T, N, d = trajectory.shape
    diagrams = []

    for t in range(T):
        # Get walker positions at time t
        positions = trajectory[t]  # shape (N, d)

        # Compute persistent homology using Ripser
        result = ripser(
            positions,
            maxdim=max_dimension,
            thresh=max_radius,
            coeff=2  # Use Z/2Z coefficients: much faster (bitwise XOR operations)
            # but loses torsion information (e.g., cannot distinguish Klein bottle
            # from torus). For counting holes (Betti numbers), this is sufficient.
        )
        diagrams.append(result['dgms'])

    # Compute topological features
    features = {
        'betti_0': [count_features(d[0]) for d in diagrams],
        'betti_1': [count_features(d[1]) for d in diagrams],
        'persistence_entropy': [entropy(d[1]) for d in diagrams],
        'max_persistence': [max_lifetime(d[1]) for d in diagrams]
    }

    return diagrams, features

def count_features(diagram, threshold=0.1):
    """Count persistent features above threshold."""
    if len(diagram) == 0:
        return 0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    return np.sum(lifetimes > threshold)

def entropy(diagram):
    """Compute persistent entropy."""
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return 0.0
    lifetimes = lifetimes / np.sum(lifetimes)
    return -np.sum(lifetimes * np.log(lifetimes + 1e-12))

def max_lifetime(diagram):
    """Maximum persistence (longest-lived feature)."""
    if len(diagram) == 0:
        return 0.0
    lifetimes = diagram[:, 1] - diagram[:, 0]
    finite_lifetimes = lifetimes[np.isfinite(lifetimes)]
    # Handle case where all classes are essential (infinite death)
    if finite_lifetimes.size == 0:
        return float('inf')  # All features are essential
    return np.max(finite_lifetimes)
```

**Integration points**:
1. **SwarmState extension**: Add TDA features to swarm metadata
2. **Visualization**: Use `fragile.shaolin` to plot persistence diagrams
3. **Adaptive operators**: Use $\beta_1$ to detect bottlenecks and trigger exploration
:::

### 7.3. Computational Complexity Summary

:::{prf:table} Complexity of Topological Invariants
:label: table-complexity

| Invariant | Method | Complexity | Framework Status |
|-----------|--------|------------|------------------|
| $\chi$ (Euler char.) | Delaunay count | $O(N)$ | ⚠️ Computed (point cloud only; see § 1.1) |
| $\beta_0$ (components) | DFS/Union-Find | $O(N + E)$ | ✅ Computable from Fractal Set |
| $\beta_1$ (cycles) | Cellular homology | $O(s^3)$ via Gaussian elimination over field; Smith Normal Form for $\mathbb{Z}$ | ⚠️ Requires cellular boundary operators |
| $\beta_k$ ($k \ge 2$) | Boundary matrix | $O(s^3)$ | ⚠️ Requires full CW complex structure |
| Persistent $H_*$ | Ripser | $O(s^3)$ | 🔧 External library; $s$ = #simplices (exp. in $N$) |
| $\pi_1$ (fund. group) | Simplicial approx. | Exponential | ❌ Hard in general |
| Genus $g$ | From $\chi$ + TDA | $O(1)$ given $\chi$ | ⚠️ Requires persistent homology for landscape |
| Chern numbers | Geometric charge | Not implemented | ❌ Plaquette holonomy defined; no Levi-Civita ε-contraction |

**Legend**:
- ✅ Already computed or trivially computable
- ⚠️ Feasible but requires additional computation
- 🔧 Requires external library
- ❌ Computationally intractable for large systems
:::

---

(part-framework-connections)=
## Part 8: Framework Connections and Cross-References

### 8.1. Existing Topological Results

:::{prf:summary} Topological Invariants Already in Framework
:label: sum-existing-topology

**Euler Characteristic**:
- **Location**: {prf:ref}`prop-euler-characteristic-scutoid` in [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)
- **Status**: Fully specified for scutoid tessellation
- **Computation**: From Delaunay triangulation vertex/edge/face counts

**First Betti Number** ($\beta_1$):
- **Location**: [00_reference.md](00_reference.md) (search "Betti number"); {prf:ref}`def-fractal-set-simplicial-complex` in [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md)
- **Status**: Computable from Fractal Set 2-complex (see {prf:ref}`thm-betti-fractal-set`)
- **Computation**: $\beta_1 = |E'| - |\mathcal{E}| + \beta_0 - \text{rank}(\partial_2)$ where $E'$ is the undirected 1-skeleton and $\partial_2$ accounts for plaquette boundaries. The 1-skeleton-only formula $\beta_1 = |E'| - |\mathcal{E}| + \beta_0$ provides an upper bound.

**Homotopy Groups**:
- **Location**: [00_reference.md](00_reference.md) entry on configuration space homotopy
- **Status**: Theoretical result ($\pi_k(\mathcal{X}^N) \cong \pi_k(\mathcal{X})^N$)
- **Computation**: Not directly computed; used for topological consistency checks

**Cohomology Groups**:
- **Location**: {prf:ref}`def-fractal-set-simplicial-complex` in [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md)
- **Status**: Defined for Fractal Set as 2-complex
- **Computation**: Not yet implemented; boundary operators defined

**Gauge Bundle Topology**:
- **Location**: [12_gauge_theory_adaptive_gas.md](12_gauge_theory_adaptive_gas.md) and [15_yang_mills/yang_mills_geometry.md](15_yang_mills/yang_mills_geometry.md)
- **Status**: Gauge connections defined; Chern numbers implicit via curvature
- **Computation**: Plaquette holonomy machinery exists (see [curvature.md](curvature.md) § 3)
:::

### 8.2. Connections to Curvature

:::{prf:remark} Curvature-Topology Dictionary
:label: rem-curvature-topology-bridge

This document ([24_topological_invariants.md](24_topological_invariants.md)) is complementary to [curvature.md](curvature.md):

**Curvature → Topology** (local to global):
- **Gauss-Bonnet**: Integrated curvature gives Euler characteristic

$$
\int_M K \, dA = 2\pi \chi(M)

$$

- **Deficit angles**: Discrete curvature at vertices sums to topology

$$
\sum_i \delta_i = 2\pi \chi

$$

- **Chern-Gauss-Bonnet**: Chern classes computed from curvature 2-forms

**Topology → Curvature** (global constraints on local geometry):
- **Myers Theorem**: Positive Ricci curvature + finite diameter → $\pi_1$ finite
- **Bonnet-Myers**: $\text{Ric} \ge \kappa > 0 \implies \text{diam}(M) \le \pi/\sqrt{\kappa}$
- **Hyperbolic surfaces**: Genus $g \ge 2 \implies$ admits metric with constant $K = -1$

**Both frameworks use**:
- Delaunay triangulation (simplicial complex structure)
- Voronoi tessellation (dual structure for volumes)
- Deficit angles (link curvature to Euler characteristic)
- Plaquette holonomy (Riemann tensor and Chern numbers)

**Computational synergy**: Computing curvature (via deficit angles) automatically gives Euler characteristic for free via Gauss-Bonnet.
:::

### 8.3. Applications to Fragile Gas Dynamics

:::{prf:remark} Topological Phase Transitions in Optimization
:label: rem-topological-phase-transitions

**Hypothesis**: During optimization, the swarm configuration undergoes **topological phase transitions** detectable via Betti numbers.

**Phase 1: Exploration** ($t \approx 0$):
- $\beta_0$ large (many disconnected clusters)
- $\beta_1$ small (sparse connectivity)
- High genus (rugged landscape exploration)

**Phase 2: Clustering** ($t \approx T/2$):
- $\beta_0$ decreases (clusters merge)
- $\beta_1$ increases (walkers form connected network)
- Topology simplifies (focus on promising regions)

**Phase 3: Convergence** ($t \approx T$):
- $\beta_0 = 1$ (single connected component)
- $\beta_1 \to 0$ (tree-like structure, no cycles)
- Genus $g \to 0$ (contractible configuration)

**Algorithmic use**:
- **Early termination**: If $\beta_0 = 1$ and $\beta_1 = 0$ early, swarm has converged
- **Stagnation detection**: If $\beta_k$ unchanging, trigger exploration boost
- **Multimodal indicator**: Persistent $\beta_0 > 1$ indicates multiple basins
:::

### 8.4. Open Questions and Future Work

:::{prf:remark} Future Directions
:label: rem-future-topology

**Theoretical**:
1. **Topological convergence theorem**: Prove that $\beta_k(S_t) \to \beta_k(\mathcal{A})$ as $t \to \infty$, where $\mathcal{A}$ is the attractor set
2. **Bottleneck distance bounds**: Relate algorithmic performance to bottleneck distance $d_B(H_1(S_t), H_1(S_{t'}))$
3. **Homotopy invariants**: Characterize when configuration space homotopy non-trivial

**Computational**:
1. **Online TDA**: Incremental persistent homology updates as walkers move
2. **Sparse methods**: Exploit Delaunay sparsity for faster homology computation
3. **GPU acceleration**: Parallelize boundary matrix reduction

**Applications**:
1. **Adaptive cloning**: Use $\beta_1$ to detect bottlenecks and focus cloning
2. **Landscape classification**: Classify fitness functions by persistent homology signatures
3. **Multimodal optimization**: Use $\beta_0$ tracking to ensure multi-basin coverage
:::

---

## References and Further Reading

**Textbooks**:
- Hatcher, "Algebraic Topology" (2002) — comprehensive introduction to homology and homotopy
- Edelsbrunner & Harer, "Computational Topology" (2010) — persistent homology and TDA
- Munkres, "Elements of Algebraic Topology" (1984) — classic reference

**Computational Topology**:
- Zomorodian & Carlsson, "Computing Persistent Homology" (2005) — foundational algorithm
- Bauer et al., "Ripser: Efficient computation of Vietoris-Rips persistence barcodes" (2021)
- Maria et al., "The Gudhi Library" (2014) — software reference

**Applications to Optimization**:
- Nicolau et al., "Topology based data analysis identifies a subgroup of breast cancers with a unique mutational profile" (2011) — TDA in biology
- Carlsson, "Topology and Data" (2009) — foundational TDA overview

**Framework Documents**:
- [curvature.md](curvature.md) — complementary geometric invariants
- [14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md) — discrete differential geometry
- [13_fractal_set_new/01_fractal_set.md](13_fractal_set_new/01_fractal_set.md) — data structure specification
- [13_fractal_set_new/08_lattice_qft_framework.md](13_fractal_set_new/08_lattice_qft_framework.md) — simplicial complex formulation

---

## Summary

This document establishes computational methods for **six core topological invariants** in the Fragile Gas framework:

1. **Euler characteristic** $\chi$ — computed from Delaunay triangulation (⚠️ yields point cloud topology only; see § 1.1 for critical limitations)
2. **Betti numbers** $\beta_k$ — $\beta_0$ computable from Fractal Set; $\beta_1, \beta_2$ require cellular homology (⚠️ boundary operators not yet defined)
3. **Homology groups** $H_k$ — formal algebraic structure via boundary operators (⚠️ requires cellular vs. simplicial clarification)
4. **Homotopy groups** $\pi_k$ — theoretical results for configuration space (configuration space itself, not accessible dynamics)
5. **Genus** $g$ — surface classification requires persistent homology or α-complexes with **sufficiently dense sampling** to satisfy Nerve Lemma guarantees (see § 2.3); not computable from Delaunay χ alone
6. **Chern numbers** — plaquette holonomy defined; Levi-Civita ε-contraction **not yet implemented** (❌ see § 6.2)

These invariants complement the **curvature theory** in [curvature.md](curvature.md) by providing global topological constraints on the emergent geometry. Together, curvature and topology form a complete picture of the Fragile Gas fitness landscape structure.

**Key Contributions**:
- Unified conceptual framework for all major topological invariants
- Integration plan with existing Delaunay triangulation and Fractal Set infrastructure
- Practical algorithms with complexity analysis and library recommendations
- Physical interpretation for optimization applications (phase transitions, bottleneck detection)
- Cross-references to framework documents showing existing topological results
- **Critical corrections**: Identified gaps between defined structures and actual topology capture

**Implementation Status and Gaps**:
- **Delaunay Euler characteristic**: Computed but captures point cloud topology, not landscape manifold
- **Fractal Set Betti numbers**: Requires cellular boundary operators for quadrilateral plaquettes
- **Chern numbers**: Plaquette holonomy exists; topological charge assembly not implemented

**Implementation Priority**:
1. **High**: Persistent homology via Ripser for landscape topology (external library; replaces misleading Delaunay χ interpretations)
2. **High**: $\beta_0$ from Fractal Set 1-skeleton (DFS/union-find; already feasible)
3. **Medium**: Cellular boundary operators for Fractal Set to enable $\beta_1, \beta_2$ computation
4. **Medium**: Chern number computation via Levi-Civita ε-contraction (extends existing plaquette infrastructure)
5. **Low**: Higher homotopy groups (expensive, limited practical value)
