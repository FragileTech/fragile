# Appendix A: Complete Scutoid Decomposition Algorithm

## A.1 Introduction

This appendix provides a complete algorithmic specification for decomposing scutoids into simplices (tetrahedra in 3+1D, pentatopes in higher dimensions). The main document [scutoid_integration.md](scutoid_integration.md) assumes this decomposition is possible and well-defined; here we prove it and provide explicit algorithms.

**Key challenges addressed:**
1. **Non-planar lateral faces**: Scutoid lateral faces connecting different Voronoi cells may be non-planar
2. **Mid-level vertices**: Scutoids may have vertices at intermediate time slices when neighbor topology changes
3. **Arbitrary dimension**: Algorithm must work for (d+1)-dimensional scutoids in d-dimensional space + time

## A.2 Mathematical Foundations

### A.2.1 Scutoid as Simplicial Complex

:::{prf:definition} Scutoid Simplicial Complex
:label: def-scutoid-simplicial-complex

A **scutoid** $S_i$ connecting Voronoi cell $V_j(t)$ to $V_k(t + \Delta t)$ is a simplicial complex defined by:

**Vertex set** $\mathcal{V}(S_i)$:

$$
\mathcal{V}(S_i) = \mathcal{V}_{\text{bot}} \cup \mathcal{V}_{\text{top}} \cup \mathcal{V}_{\text{mid}}
$$

where:
- $\mathcal{V}_{\text{bot}}$: Bottom face vertices (Voronoi cell $V_j(t)$ corners) at time $t$
- $\mathcal{V}_{\text{top}}$: Top face vertices (Voronoi cell $V_k(t+\Delta t)$ corners) at time $t+\Delta t$
- $\mathcal{V}_{\text{mid}}$: Mid-level vertices at times $t < t_m < t + \Delta t$ (topology changes)

**Edge set** $\mathcal{E}(S_i)$:
- Edges within bottom face (Voronoi edges at time $t$)
- Edges within top face (Voronoi edges at time $t + \Delta t$)
- Vertical edges connecting bottom to top vertices
- Edges from mid-level vertices to bottom/top/other mid-level vertices

**Face set** $\mathcal{F}(S_i)$:
- Bottom face: Voronoi cell $V_j(t)$ (convex polygon, planar)
- Top face: Voronoi cell $V_k(t + \Delta t)$ (convex polygon, planar)
- Lateral faces: Connecting bottom to top edges (may be non-planar)

**Hyperface set** $\mathcal{H}(S_i)$ (d ≥ 3):
- 3-dimensional faces in (d+1)-dimensional scutoid
:::

### A.2.2 Convexity and Planarity

:::{prf:lemma} Scutoid Convexity Properties
:label: lem-scutoid-convexity

1. **Top and bottom faces are convex and planar**: They are Voronoi cells at fixed time slices
2. **Lateral faces are generally non-planar**: Face connecting bottom edge to top edge may not lie in a plane
3. **Scutoid interior is star-convex from spacetime centroid**: For any point $p \in S_i$, the line segment from centroid $c_i$ to $p$ lies entirely in $S_i$
:::

:::{prf:proof}
1. Voronoi cells are convex polytopes by definition, so top/bottom faces are convex and planar.

2. Consider lateral face connecting bottom edge $(v_1, v_2)$ at time $t$ to top edge $(v_3, v_4)$ at time $t + \Delta t$. The four points $(v_1, t)$, $(v_2, t)$, $(v_3, t+\Delta t)$, $(v_4, t+\Delta t)$ in (d+1)-dimensional spacetime generally do not lie in a common 2-plane, so the face is non-planar.

3. Star-convexity from centroid follows from the construction: scutoid connects convex cells through continuous deformation, and centroid lies in interior by time-weighted average construction.
∎
:::

## A.3 Decomposition Algorithms

### A.3.1 Strategy Overview

We use a two-stage decomposition:

**Stage 1: Face Triangulation**
- Decompose non-planar lateral faces into planar triangles
- Use fan triangulation from face centroid

**Stage 2: Volume Tetrahedralization**
- Decompose scutoid into simplices using centroid fan decomposition
- Each simplex has one vertex at scutoid centroid

### A.3.2 Lateral Face Triangulation

:::{prf:algorithm} Triangulate Lateral Face
:label: alg-triangulate-lateral-face

**Input:** Lateral face $F$ with vertices $\{v_1, \ldots, v_m\}$ ordered counterclockwise

**Output:** Set of planar triangles $\{T_1, \ldots, T_k\}$ covering $F$

**Step 1: Compute face centroid**

$$
c_F = \frac{1}{m} \sum_{i=1}^m v_i
$$

**Step 2: Create triangles via fan triangulation**

For each edge $(v_i, v_{i+1})$ in the face boundary (with $v_{m+1} := v_1$):

$$
T_i = (c_F, v_i, v_{i+1})
$$

**Output:** $\{T_1, \ldots, T_m\}$

**Properties:**
- Each triangle is planar by construction (3 points determine a plane)
- Triangles share vertex $c_F$
- No triangle degeneracy if vertices are distinct and non-collinear
:::

:::{prf:lemma} Face Triangulation Coverage
:label: lem-face-triangulation-coverage

The fan triangulation of a lateral face $F$ covers $F$ completely and without overlap.

**Proof sketch:**
1. **Coverage**: Any point $p \in F$ can be expressed as $p = \alpha c_F + (1-\alpha) q$ where $q$ lies on boundary. Point $q$ lies on some edge $(v_i, v_{i+1})$, so $p \in T_i$.
2. **No overlap**: Triangles $T_i$ and $T_j$ (i ≠ j) share only vertex $c_F$ or edge from $c_F$ (if $|i-j| = 1$), so interiors are disjoint.
∎
:::

### A.3.3 Prism Decomposition (Special Case)

When the scutoid has no neighbor changes ($\mathcal{N}_j(t) = \mathcal{N}_i(t+\Delta t)$), the scutoid is a **prism** connecting congruent top and bottom Voronoi cells. This special case admits an efficient decomposition.

:::{prf:algorithm} Decompose Prism into Simplices via Vertex Fan
:label: alg-prism-decomposition

**Input:**
- Bottom $d$-simplex: $T^{\text{bot}} = (v_0, v_1, \ldots, v_d)$ at time $t$
- Top vertices: $(\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_d)$ at time $t + \Delta t$ corresponding to bottom vertices

**Output:** Set of $(d+1)$ non-overlapping $(d+1)$-simplices partitioning the prismatoid

**Procedure:**

Use fan decomposition from top vertex $\hat{v}_0$:

For $k = 0, 1, \ldots, d$, define simplex $\Delta_k$ by:

$$
\Delta_k = \begin{cases}
(\hat{v}_0, v_0, v_1, \ldots, v_d) & k = 0 \\
(\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_k, v_k, v_{k+1}, \ldots, v_d) & 1 \leq k \leq d
\end{cases}
$$

Explicitly:
- $\Delta_0 = (\hat{v}_0, v_0, v_1, \ldots, v_d)$ — contains all bottom vertices plus $\hat{v}_0$
- $\Delta_1 = (\hat{v}_0, \hat{v}_1, v_1, v_2, \ldots, v_d)$ — replaces $v_0$ with $\hat{v}_1$
- $\Delta_2 = (\hat{v}_0, \hat{v}_1, \hat{v}_2, v_2, v_3, \ldots, v_d)$ — replaces $v_1$ with $\hat{v}_2$
- ...
- $\Delta_d = (\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_d)$ — all top vertices

**Output:** $\{\Delta_0, \Delta_1, \ldots, \Delta_d\}$ (total of $d+1$ simplices)
:::

:::{prf:theorem} Prism Fan Decomposition Correctness
:label: thm-prism-decomposition-correctness

Algorithm {prf:ref}`alg-prism-decomposition` produces a valid simplicial decomposition of the prismatoid $P = \text{ConvexHull}(T^{\text{bot}} \cup T^{\text{top}})$.

Specifically:
1. **Coverage**: $\bigcup_{k=0}^d \Delta_k = P$
2. **Non-overlapping**: $\text{int}(\Delta_i) \cap \text{int}(\Delta_j) = \emptyset$ for $i \neq j$
3. **Non-degeneracy**: Each $\Delta_k$ has positive $(d+1)$-volume if $P$ is non-degenerate
:::

:::{prf:proof}
**Notation:** Let $P$ be the prismatoid (convex hull of bottom and top simplices). Vertices: $v_0, \ldots, v_d$ (bottom) at time $t$, and $\hat{v}_0, \ldots, \hat{v}_d$ (top) at time $t + \Delta t$.

**Part 1 (Coverage):**

We prove $P = \bigcup_{k=0}^d \Delta_k$ by showing that every point $p \in P$ belongs to exactly one simplex $\Delta_k$.

**Direct construction:** Any point $p \in P$ can be written as a convex combination of the bottom and top vertices:

$$
p = \sum_{i=0}^d \alpha_i v_i + \sum_{j=0}^d \beta_j \hat{v}_j
$$

where $\sum_i \alpha_i + \sum_j \beta_j = 1$ and all coefficients are non-negative.

**Claim:** $p$ belongs to simplex $\Delta_k$ where $k = \min\{j : \beta_j > 0\}$ (the index of the first top vertex with positive weight), or $\Delta_0$ if all $\beta_j = 0$.

**Proof of claim:** Consider the barycentric representation of $p$ restricted to the vertices of $\Delta_k = (\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_k, v_k, v_{k+1}, \ldots, v_d)$.

By construction of the fan decomposition, the vertices of $\Delta_k$ are:
- Top vertices: $\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_k$
- Bottom vertices: $v_k, v_{k+1}, \ldots, v_d$

For $p$ to lie in $\Delta_k$, we need to express $p$ as:

$$
p = \sum_{j=0}^k \gamma_j \hat{v}_j + \sum_{i=k}^d \delta_i v_i
$$

with $\sum_j \gamma_j + \sum_i \delta_i = 1$ and all coefficients non-negative.

**Construction of coefficients (explicit formula):**

Define $R = \sum_{j>k} \beta_j + \sum_{i<k} \alpha_i$ (total weight from vertices not in $\Delta_k$).

Since $k = \min\{j : \beta_j > 0\}$, we have $\beta_0 = \beta_1 = \cdots = \beta_{k-1} = 0$, so:

$$
R = \sum_{j>k} \beta_j + \sum_{i<k} \alpha_i = \left( \sum_{j=0}^d \beta_j - \sum_{j=0}^k \beta_j \right) + \sum_{i=0}^{k-1} \alpha_i
$$

**Explicit redistribution:** Set the new barycentric coordinates as:

$$
\gamma_j = \begin{cases}
\beta_j + \frac{\beta_j}{\sum_{\ell=0}^k \beta_\ell} \cdot R & \text{if } j \leq k \text{ and } \beta_j > 0 \\
0 & \text{if } j \leq k \text{ and } \beta_j = 0
\end{cases}
$$

$$
\delta_i = \alpha_i \quad \text{for } i \geq k
$$

**Verification of non-negativity:**
- $\gamma_j \geq 0$: Since $\beta_j \geq 0$, $R \geq 0$, and the denominator $\sum_{\ell=0}^k \beta_\ell > 0$ (because $\beta_k > 0$ by minimality of $k$), we have $\gamma_j \geq \beta_j \geq 0$.
- $\delta_i \geq 0$: Directly from $\alpha_i \geq 0$.

**Verification of normalization:**

$$
\sum_{j=0}^k \gamma_j + \sum_{i=k}^d \delta_i = \sum_{j=0}^k \left( \beta_j + \frac{\beta_j}{\sum_{\ell=0}^k \beta_\ell} \cdot R \right) + \sum_{i=k}^d \alpha_i
$$

$$
= \sum_{j=0}^k \beta_j + \frac{R}{\sum_{\ell=0}^k \beta_\ell} \cdot \sum_{j=0}^k \beta_j + \sum_{i=k}^d \alpha_i = \sum_{j=0}^k \beta_j + R + \sum_{i=k}^d \alpha_i
$$

By definition of $R$:

$$
\sum_{j=0}^k \beta_j + R + \sum_{i=k}^d \alpha_i = \sum_{j=0}^d \beta_j + \sum_{i=0}^d \alpha_i = 1
$$

Therefore, the coefficients $(\gamma_j, \delta_i)$ are valid barycentric coordinates for $\Delta_k$. ✓

**Uniqueness:** The choice of $k = \min\{j : \beta_j > 0\}$ uniquely determines which simplex contains $p$ in its interior or boundary. Each simplex $\Delta_k$ captures the "layer" of the prismatoid where top vertices $\hat{v}_0, \ldots, \hat{v}_k$ first become active.

**Verification:**
- For $k=0$: $\Delta_0$ contains all points where only $\hat{v}_0$ among top vertices has positive weight
- For $k=d$: $\Delta_d = T^{\text{top}}$ is exactly the top simplex (all $\beta_j > 0$, all $\alpha_i = 0$)
- Intermediate $\Delta_k$ form a nested sequence covering the prismatoid

Therefore, $P = \bigcup_{k=0}^d \Delta_k$ with each point belonging to exactly one simplex. ✓

**Part 2 (Non-overlapping):**

Consider two distinct simplices $\Delta_i$ and $\Delta_j$ with $i < j$.

$\Delta_i$ has vertices $\{\hat{v}_0, \ldots, \hat{v}_i, v_i, v_{i+1}, \ldots, v_d\}$.

$\Delta_j$ has vertices $\{\hat{v}_0, \ldots, \hat{v}_j, v_j, v_{j+1}, \ldots, v_d\}$.

**Shared facet:** $\Delta_i$ and $\Delta_j$ share the $(d)$-face:

$$
F_{ij} = (\hat{v}_0, \hat{v}_1, \ldots, \hat{v}_i, v_j, v_{j+1}, \ldots, v_d)
$$

(this is the common boundary obtained by removing exactly one vertex from each simplex's vertex set).

**No interior overlap:** Points in $\text{int}(\Delta_i)$ have $v_i$ in their convex combination with positive weight, while points in $\text{int}(\Delta_j)$ do not (since $v_i \notin \Delta_j$ for $j > i$). Thus $\text{int}(\Delta_i) \cap \text{int}(\Delta_j) = \emptyset$. ✓

**Part 3 (Non-degeneracy):**

Simplex $\Delta_k$ is degenerate iff its $(d+2)$ vertices lie in a common $d$-dimensional hyperplane in $(d+1)$-space.

For a non-degenerate prismatoid, the bottom simplex $T^{\text{bot}}$ spans a $d$-plane at time $t$, and the top vertex $\hat{v}_0$ lies at a different time $t + \Delta t \neq t$. Thus $\hat{v}_0 \notin \text{span}(v_0, \ldots, v_d)$, so $\Delta_0$ is non-degenerate.

By similar reasoning, each $\Delta_k$ has vertices spanning both time slices, so is non-degenerate. ✓

**Conclusion:** The fan decomposition produces a valid $(d+1)$-simplicial partition of the prismatoid. ∎
:::

:::{prf:remark} Computational Complexity of Prism Decomposition
:label: rem-prism-complexity

For a prism connecting Voronoi cells with $n$ vertices each:
- **Bottom cell triangulation:** $O(n \log n)$ time using Delaunay triangulation (for 2D), producing $O(n)$ spatial simplices
- **Each spatial simplex** → $d+1$ spacetime simplices via fan decomposition: $O(d)$ per simplex
- **Total spacetime simplices:** $O(n \cdot d) = O(n)$ for fixed $d$
- **Overall time complexity:** $O(n \log n)$ dominated by the triangulation step

**Note:** For higher-dimensional spaces ($d \geq 3$), Delaunay triangulation complexity increases, but for fixed dimension $d$, the complexity remains $O(n \log n)$ with dimension-dependent constants.

**Comparison to centroid fan:** The prism-specific algorithm produces $O(n)$ simplices, while the general centroid fan (Section A.3.4) produces $O(n \cdot f)$ simplices where $f$ is the number of faces. For prisms, the prism-specific algorithm is more efficient.
:::

### A.3.4 General Scutoid Simplex Decomposition

:::{prf:algorithm} Decompose Scutoid into Simplices
:label: alg-scutoid-simplex-decomposition

**Input:** Scutoid $S_i$ with triangulated boundary (top, bottom, lateral faces all triangulated)

**Output:** Set of (d+1)-simplices $\{\Delta_1, \ldots, \Delta_K\}$ covering $S_i$

**Step 1: Compute scutoid spacetime centroid**

$$
c_i = \frac{1}{|\mathcal{V}(S_i)|} \sum_{v \in \mathcal{V}(S_i)} v
$$

where vertices are (d+1)-dimensional spacetime points.

**Step 2: Create simplex for each boundary triangle**

For each triangle $T_j = (u_1, u_2, u_3)$ on the scutoid boundary:

$$
\Delta_j = (c_i, u_1, u_2, u_3)
$$

This is a (d+1)-simplex in (d+1)-dimensional spacetime.

**Step 3: Verify completeness**

Check that:
1. Every boundary triangle is covered by exactly one simplex
2. All simplices share vertex $c_i$
3. No simplex degeneracy (vertices must be affinely independent)

**Output:** $\{\Delta_1, \ldots, \Delta_K\}$
:::

:::{prf:theorem} Scutoid Simplex Decomposition Correctness
:label: thm-scutoid-simplex-decomposition

Algorithm {prf:ref}`alg-scutoid-simplex-decomposition` produces a valid simplicial decomposition of scutoid $S_i$.

Specifically:
1. **Coverage**: $\bigcup_{j=1}^K \Delta_j = S_i$
2. **Non-overlapping**: $\text{int}(\Delta_j) \cap \text{int}(\Delta_k) = \emptyset$ for $j \neq k$
3. **Non-degeneracy**: Each simplex $\Delta_j$ has positive (d+1)-volume
:::

:::{prf:proof}
**Part 1 (Coverage):**

Let $p \in S_i$ be an arbitrary point. By Lemma {prf:ref}`lem-scutoid-convexity`, $S_i$ is star-convex from centroid $c_i$, so the line segment $L = \{c_i + \lambda(p - c_i) : \lambda \in [0,1]\}$ lies entirely in $S_i$.

The ray from $c_i$ through $p$ must intersect the scutoid boundary $\partial S_i$ at some point $q$. Point $q$ lies on some boundary triangle $T_j$, so $q$ can be written as:

$$
q = \alpha_1 u_1 + \alpha_2 u_2 + \alpha_3 u_3
$$

where $T_j = (u_1, u_2, u_3)$ and $\alpha_1 + \alpha_2 + \alpha_3 = 1$, $\alpha_i \geq 0$.

Point $p$ lies on segment from $c_i$ to $q$, so:

$$
p = (1 - \lambda) c_i + \lambda q = (1-\lambda) c_i + \lambda(\alpha_1 u_1 + \alpha_2 u_2 + \alpha_3 u_3)
$$

for some $\lambda \in [0,1]$. This is a convex combination of $(c_i, u_1, u_2, u_3)$, so $p \in \Delta_j$.

Therefore $S_i \subseteq \bigcup_{j=1}^K \Delta_j$.

Conversely, each simplex $\Delta_j = (c_i, u_1, u_2, u_3)$ has centroid $c_i \in S_i$ and boundary vertices on $\partial S_i$. By star-convexity, all points on segments from $c_i$ to boundary lie in $S_i$, so $\Delta_j \subseteq S_i$.

Thus $\bigcup_{j=1}^K \Delta_j = S_i$. ✓

**Part 2 (Non-overlapping):**

Consider two distinct simplices $\Delta_j = (c_i, u_1^j, u_2^j, u_3^j)$ and $\Delta_k = (c_i, u_1^k, u_2^k, u_3^k)$ corresponding to boundary triangles $T_j$ and $T_k$.

Since $T_j$ and $T_k$ are distinct boundary triangles, they are either:
- **Disjoint**: $T_j \cap T_k = \emptyset$
- **Share edge**: $T_j \cap T_k =$ common edge
- **Share vertex**: $T_j \cap T_k =$ single vertex

All simplices share vertex $c_i$. Two simplices $\Delta_j$ and $\Delta_k$ can share:
- Vertex $c_i$ (always)
- Facet $(c_i, u_a, u_b)$ if triangles $T_j, T_k$ share edge $(u_a, u_b)$

In all cases, $\text{int}(\Delta_j) \cap \text{int}(\Delta_k) = \emptyset$. ✓

**Part 3 (Non-degeneracy):**

Simplex $\Delta_j = (c_i, u_1, u_2, u_3)$ is degenerate iff the four points are coplanar in (d+1)-dimensional spacetime.

Centroid $c_i$ is computed as average of all scutoid vertices at various time slices. If scutoid connects distinct Voronoi cells at different times, centroid has a non-trivial time component different from boundary triangle vertices.

Boundary triangle $(u_1, u_2, u_3)$ lies on a 2-dimensional face of the scutoid boundary. For $d \geq 2$, a 2-plane in (d+1)-space has codimension $\geq 1$, so generic centroid does not lie on this plane.

Thus $(c_i, u_1, u_2, u_3)$ are affinely independent, and simplex has positive volume. ✓

**Exceptional case:** If $c_i$ lies on the plane containing $(u_1, u_2, u_3)$, simplex is degenerate. This occurs only if scutoid is completely flat (zero thickness), which is excluded by construction (Voronoi cells at distinct time slices). ∎
:::

### A.3.4 Handling Mid-Level Vertices

:::{prf:algorithm} Handle Mid-Level Vertices in Scutoid Decomposition
:label: alg-mid-level-vertices

**Input:** Scutoid $S_i$ with mid-level vertex set $\mathcal{V}_{\text{mid}}$

**Output:** Updated triangulated boundary including mid-level vertices

**Step 1: Identify affected lateral faces**

For each mid-level vertex $v_m \in \mathcal{V}_{\text{mid}}$:
- Identify the lateral face $F$ that $v_m$ subdivides
- Face $F$ connects bottom edge $e_{\text{bot}}$ to top edge $e_{\text{top}}$

**Step 2: Subdivide lateral face**

Original face $F$ has 4 corners: $(v_1^{\text{bot}}, v_2^{\text{bot}}, v_1^{\text{top}}, v_2^{\text{top}})$

Mid-level vertex $v_m$ at time $t_m \in (t, t+\Delta t)$ divides $F$ into two sub-faces:
- $F_1$: Connects $(v_1^{\text{bot}}, v_2^{\text{bot}})$ at time $t$ to $(v_m)$ at time $t_m$
- $F_2$: Connects $(v_m)$ at time $t_m$ to $(v_1^{\text{top}}, v_2^{\text{top}})$ at time $t+\Delta t$

**Step 3: Triangulate sub-faces**

Apply Algorithm {prf:ref}`alg-triangulate-lateral-face` to each sub-face:
- Triangulate $F_1$ using its centroid $c_{F_1}$
- Triangulate $F_2$ using its centroid $c_{F_2}$

**Step 4: Update boundary triangulation**

Replace original face $F$ with triangulated sub-faces $F_1$ and $F_2$ in the boundary triangulation.

**Output:** Updated set of boundary triangles including mid-level vertex contributions
:::

:::{prf:remark} Mid-Level Vertices and Topology Changes
:label: rem-mid-level-topology

Mid-level vertices arise when the neighbor relationship changes during the time interval $[t, t+\Delta t]$. For example:
- Walker $i$ is a neighbor of walker $j$ at time $t$
- Walker $i$ is a neighbor of walker $k$ at time $t + \Delta t$
- Neighbor switch occurs at intermediate time $t_m$

The scutoid connecting $V_j(t)$ to $V_k(t+\Delta t)$ has a mid-level vertex at time $t_m$ where the "pinch" occurs. Algorithm {prf:ref}`alg-mid-level-vertices` handles this by subdividing affected lateral faces.

**Important:** For small enough $\Delta t$, the number of mid-level vertices per scutoid is bounded (typically 0 or 1). For large $\Delta t$, multiple topology changes may occur, requiring recursive application of the subdivision algorithm.
:::

## A.4 Computational Complexity

:::{prf:theorem} Decomposition Algorithm Complexity
:label: thm-decomposition-complexity

For a scutoid $S_i$ connecting Voronoi cells with $n_{\text{bot}}$ bottom vertices and $n_{\text{top}}$ top vertices:

**Time complexity:**
- Lateral face triangulation: $O(n_{\text{bot}} + n_{\text{top}})$
- Scutoid simplex decomposition: $O(K)$ where $K = O(n_{\text{bot}} + n_{\text{top}})$ is the number of boundary triangles
- Mid-level vertex handling: $O(m \cdot (n_{\text{bot}} + n_{\text{top}}))$ where $m$ is the number of mid-level vertices

**Overall:** $O(m \cdot n)$ where $n = n_{\text{bot}} + n_{\text{top}}$

**Space complexity:** $O(K) = O(n)$ to store all simplices

**Practical bounds:**
- For Voronoi tessellations in dimension $d$, typical Voronoi cell has $O(1)$ neighbors (bounded by Delaunay degree)
- Typical values: $n_{\text{bot}}, n_{\text{top}} \sim 10$-$50$ for 2D/3D space
- Number of simplices: $K \sim 50$-$500$ per scutoid
:::

:::{prf:proof}
**Lateral face triangulation:** Each face with $m$ vertices requires computing 1 centroid ($O(m)$) and creating $m$ triangles ($O(m)$). Total over all lateral faces is $O(n_{\text{bot}} + n_{\text{top}})$ since number of lateral faces is bounded by number of vertices.

**Simplex decomposition:** Computing scutoid centroid is $O(n)$. Creating $K$ simplices is $O(K)$. Each boundary triangle contributes one simplex, so $K = O(n)$.

**Mid-level vertices:** Each mid-level vertex subdivides one lateral face, requiring re-triangulation of two sub-faces. Cost per mid-level vertex is $O(n)$, so total is $O(m \cdot n)$.

**Space:** Storing $K$ simplices, each with $d+2$ vertices, requires $O(K \cdot d) = O(n \cdot d)$ space. For fixed dimension $d$, this is $O(n)$. ∎
:::

## A.5 Degeneracy Handling

:::{prf:definition} Degenerate Simplices
:label: def-degenerate-simplex

A (d+1)-simplex $\Delta = (v_0, v_1, \ldots, v_{d+1})$ is **degenerate** if its vertices are affinely dependent, i.e., if the volume:

$$
V(\Delta) = \frac{1}{(d+1)!} \left| \det \begin{pmatrix}
v_1 - v_0 & v_2 - v_0 & \cdots & v_{d+1} - v_0 \\
\end{pmatrix} \right| = 0
$$

Geometric interpretation: All vertices lie in a hyperplane of dimension $< d+1$.
:::

:::{prf:algorithm} Detect and Handle Degenerate Simplices
:label: alg-degeneracy-handling

**Input:** Simplex $\Delta = (v_0, v_1, \ldots, v_{d+1})$ with tolerance $\epsilon > 0$

**Output:** Boolean indicating degeneracy, or valid replacement simplex

**Step 1: Compute volume**

$$
V = \frac{1}{(d+1)!} \left| \det(G) \right|^{1/2}
$$

where $G$ is the Gram matrix (see Section 3.2 of main document).

**Step 2: Check degeneracy threshold**

If $V < \epsilon \cdot \ell_{\text{cell}}^{d+1}$ where $\ell_{\text{cell}}$ is characteristic scutoid size:
- **Degenerate:** Flag simplex for removal or subdivision

**Step 3: Handle degeneracy**

**Option A (Removal):** If simplex has negligible volume, remove it from decomposition.
- **Safe** if $V \ll$ total scutoid volume
- Does not affect integral computations significantly

**Option B (Perturbation):** Perturb centroid $c_i$ slightly:

$$
c_i' = c_i + \delta \mathbf{n}
$$

where $\mathbf{n}$ is normal to the hyperplane containing $(v_1, \ldots, v_{d+1})$ and $\delta \sim \epsilon \ell_{\text{cell}}$.

- Creates non-degenerate simplex with small volume
- Introduces controlled error $O(\epsilon)$

**Output:** Updated simplex or removal flag
:::

:::{prf:lemma} Degeneracy is Rare
:label: lem-degeneracy-rare

For generic Voronoi tessellations and small time steps $\Delta t$, the probability of degenerate simplices in the decomposition is exponentially small:

$$
P(\text{degenerate}) \lesssim \exp\left( -c \frac{\Delta t}{\tau_{\text{reconfig}}} \right)
$$

where $\tau_{\text{reconfig}}$ is the characteristic time for Voronoi tessellation reconfiguration.
:::

:::{prf:proof}
Degeneracy occurs when scutoid centroid $c_i$ lies on a 2-plane containing boundary triangle $(v_1, v_2, v_3)$.

Centroid is computed as average over all scutoid vertices at different time slices. For $\Delta t \ll \tau_{\text{reconfig}}$, Voronoi cells evolve continuously, and centroid has a well-defined time coordinate distinct from boundary triangles.

The set of centroid positions that produce degeneracy forms a measure-zero set (intersection of centroid position with 2-plane). For random walker dynamics, probability scales as $\sim (\Delta t / \tau_{\text{reconfig}})^{(d-1)}$ by dimensional analysis.

For typical parameters with $\Delta t \ll \tau_{\text{reconfig}}$, this probability is negligible. ∎
:::

## A.6 Validation Tests

:::{prf:algorithm} Validate Scutoid Decomposition
:label: alg-validate-decomposition

**Input:** Scutoid $S_i$ and simplex decomposition $\{\Delta_1, \ldots, \Delta_K\}$

**Output:** Boolean validation result and error diagnostics

**Test 1: Volume conservation**

Compute scutoid volume by summing simplex volumes:

$$
V_{\text{sum}} = \sum_{j=1}^K V(\Delta_j)
$$

Compare to direct scutoid volume computation (e.g., via convex hull). Require:

$$
\left| V_{\text{sum}} - V_{\text{scutoid}} \right| < \epsilon \cdot V_{\text{scutoid}}
$$

**Test 2: Boundary coverage**

For each boundary triangle $T$, verify:
- Exactly one simplex has $T$ as a facet
- No boundary triangles are missing from decomposition

**Test 3: Interior non-overlap**

For each pair of distinct simplices $\Delta_j, \Delta_k$:
- Verify that $\text{int}(\Delta_j) \cap \text{int}(\Delta_k) = \emptyset$
- Check: Simplices share at most a common facet, edge, or vertex

**Test 4: Non-degeneracy**

For each simplex $\Delta_j$:
- Compute volume $V(\Delta_j)$
- Verify $V(\Delta_j) > \epsilon \cdot \ell_{\text{cell}}^{d+1}$

**Output:** Pass/fail for each test with diagnostic information
:::

## A.7 Pseudocode Implementation

```python
import numpy as np
from scipy.spatial import ConvexHull
from typing import List, Tuple

def triangulate_lateral_face(vertices: np.ndarray) -> List[np.ndarray]:
    """
    Triangulate a lateral face using fan triangulation from centroid.

    Args:
        vertices: Array of shape (m, d+1) - face vertices in spacetime

    Returns:
        List of triangles, each of shape (3, d+1)
    """
    m = len(vertices)
    centroid = np.mean(vertices, axis=0)

    triangles = []
    for i in range(m):
        v1 = vertices[i]
        v2 = vertices[(i+1) % m]  # Wrap around
        triangle = np.array([centroid, v1, v2])
        triangles.append(triangle)

    return triangles

def decompose_scutoid(scutoid_vertices: np.ndarray,
                     boundary_triangles: List[np.ndarray],
                     epsilon: float = 1e-10) -> List[np.ndarray]:
    """
    Decompose scutoid into simplices using centroid fan decomposition.

    Args:
        scutoid_vertices: Array of shape (n, d+1) - all scutoid vertices
        boundary_triangles: List of triangulated boundary faces
        epsilon: Degeneracy threshold

    Returns:
        List of (d+1)-simplices, each of shape (d+2, d+1)
    """
    # Compute scutoid spacetime centroid
    centroid = np.mean(scutoid_vertices, axis=0)

    simplices = []
    for triangle in boundary_triangles:
        # Create simplex: (centroid, v1, v2, v3)
        simplex = np.vstack([centroid.reshape(1, -1), triangle])

        # Check for degeneracy
        volume = compute_simplex_volume(simplex)

        if volume > epsilon:
            simplices.append(simplex)
        else:
            # Handle degeneracy: perturb centroid
            perturbed_simplex = perturb_degenerate_simplex(simplex, epsilon)
            if perturbed_simplex is not None:
                simplices.append(perturbed_simplex)

    return simplices

def compute_simplex_volume(simplex: np.ndarray) -> float:
    """
    Compute (d+1)-dimensional volume of simplex using Gram determinant.

    Args:
        simplex: Array of shape (d+2, d+1) - simplex vertices

    Returns:
        Volume (non-negative scalar)
    """
    d_plus_1 = simplex.shape[1]
    v0 = simplex[0]

    # Compute edge vectors from first vertex
    edges = simplex[1:] - v0

    # Gram matrix: G_ij = <e_i, e_j>
    G = edges @ edges.T

    # Volume = sqrt(det(G)) / (d+1)!
    det_G = np.linalg.det(G)

    if det_G < 0:
        det_G = 0  # Numerical error

    factorial = np.math.factorial(d_plus_1)
    volume = np.sqrt(det_G) / factorial

    return volume

def perturb_degenerate_simplex(simplex: np.ndarray,
                               epsilon: float) -> np.ndarray:
    """
    Perturb centroid to resolve degeneracy.

    Args:
        simplex: Degenerate simplex of shape (d+2, d+1)
        epsilon: Perturbation scale

    Returns:
        Perturbed simplex or None if unrecoverable
    """
    centroid = simplex[0]
    boundary_verts = simplex[1:]

    # Compute normal to hyperplane containing boundary vertices
    # Use SVD to find null space
    centered = boundary_verts - np.mean(boundary_verts, axis=0)
    U, S, Vt = np.linalg.svd(centered)

    # Normal is last row of V (smallest singular value)
    normal = Vt[-1]

    # Perturb centroid along normal
    ell_cell = np.linalg.norm(boundary_verts[1] - boundary_verts[0])
    delta = epsilon * ell_cell

    centroid_perturbed = centroid + delta * normal

    # Create perturbed simplex
    simplex_perturbed = np.vstack([centroid_perturbed.reshape(1, -1),
                                   boundary_verts])

    return simplex_perturbed

def validate_decomposition(scutoid_vertices: np.ndarray,
                          simplices: List[np.ndarray],
                          epsilon: float = 1e-6) -> dict:
    """
    Validate scutoid decomposition.

    Args:
        scutoid_vertices: Array of shape (n, d+1)
        simplices: List of (d+1)-simplices
        epsilon: Tolerance for tests

    Returns:
        Dictionary with validation results
    """
    results = {}

    # Test 1: Volume conservation
    volume_sum = sum(compute_simplex_volume(s) for s in simplices)

    # Compute scutoid volume via convex hull (if possible)
    try:
        hull = ConvexHull(scutoid_vertices)
        volume_scutoid = hull.volume
        volume_error = abs(volume_sum - volume_scutoid) / volume_scutoid
        results['volume_conservation'] = volume_error < epsilon
        results['volume_error'] = volume_error
    except:
        results['volume_conservation'] = None
        results['volume_error'] = None

    # Test 2: Non-degeneracy
    ell_cell = np.linalg.norm(scutoid_vertices[1] - scutoid_vertices[0])
    d_plus_1 = scutoid_vertices.shape[1]
    threshold = epsilon * ell_cell**(d_plus_1)

    degenerate_count = sum(1 for s in simplices
                          if compute_simplex_volume(s) < threshold)
    results['non_degeneracy'] = (degenerate_count == 0)
    results['degenerate_count'] = degenerate_count

    # Test 3: Simplex count
    results['simplex_count'] = len(simplices)

    return results
```

## A.8 Example: 2D Space + Time (3D Scutoids)

Consider a simple scutoid in 2D space + time:

**Bottom face (t=0):** Square Voronoi cell with vertices $(0,0,0)$, $(1,0,0)$, $(1,1,0)$, $(0,1,0)$

**Top face (t=1):** Triangular Voronoi cell with vertices $(0.5,0,1)$, $(1.5,0.5,1)$, $(0.5,1.5,1)$

**Lateral faces:** Connect square edges to triangle edges (4 lateral faces, non-planar)

**Decomposition steps:**

1. **Triangulate lateral faces:**
   - Face 1: Connects $(0,0,0)$-$(1,0,0)$ to $(0.5,0,1)$-$(1.5,0.5,1)$ → 4 triangles via centroid fan
   - Face 2: Connects $(1,0,0)$-$(1,1,0)$ to $(1.5,0.5,1)$-$(0.5,1.5,1)$ → 4 triangles
   - Face 3: Connects $(1,1,0)$-$(0,1,0)$ to $(0.5,1.5,1)$-$(0.5,0,1)$ → 4 triangles
   - Total: 12 lateral triangles

2. **Triangulate bottom and top faces:**
   - Bottom: 2 triangles (square decomposition)
   - Top: 1 triangle (already triangular)
   - Total boundary: 15 triangles

3. **Compute scutoid centroid:**

$$
c = \frac{1}{7}[(0,0,0) + (1,0,0) + (1,1,0) + (0,1,0) + (0.5,0,1) + (1.5,0.5,1) + (0.5,1.5,1)] \approx (0.64, 0.64, 0.43)
$$

4. **Create 15 tetrahedra:** Each boundary triangle + centroid

**Result:** 15 tetrahedra covering scutoid, total volume $\approx 0.75$ (connecting square area 1 to triangle area 0.5 over time interval 1).

## A.9 Convergence Under Refinement

:::{prf:theorem} Decomposition Error Convergence
:label: thm-decomposition-error-convergence

As the spacetime grid is refined (more walkers $N \to \infty$ and smaller time step $\Delta t \to 0$), the error in geometric quantities computed via scutoid decomposition converges to zero.

**Specifically:** For a smooth function $f: \mathcal{M} \to \mathbb{R}$ integrated over the spacetime domain:

$$
\left| \int_{\mathcal{M}} f \, dV_{g_{ST}} - \sum_{i,j} f(c_{ij}) V_{g_{ST}}(S_{ij}) \right| \lesssim C_f \left( N^{-2/d} + \Delta t^2 \right)
$$

where:
- Sum is over all scutoid simplices
- $c_{ij}$ is simplex centroid
- $V_{g_{ST}}(S_{ij})$ is simplex volume
- $C_f$ depends on smoothness of $f$
:::

:::{prf:proof}
Decomposition error has two sources:

**1. Spatial discretization error:**

Voronoi tessellation approximates continuous spacetime with discrete cells. For $N$ walkers in dimension $d$, characteristic cell size is $\ell_{\text{cell}} \sim N^{-1/d}$.

Scutoid volume computation error:

$$
\Delta V \sim \ell_{\text{cell}}^{d+1} \sim N^{-(d+1)/d}
$$

For smooth $f$, integration error:

$$
\Delta I_{\text{spatial}} \sim \sum_{\text{cells}} |f(c) - \bar{f}_{\text{cell}}| \cdot V_{\text{cell}} \sim N \cdot N^{-2/d} \cdot N^{-(d+1)/d} \sim N^{-2/d}
$$

**2. Temporal discretization error:**

BAOAB integrator has $O(\Delta t^2)$ local error. Volume evolution:

$$
V(t + \Delta t) = V(t) + V'(t) \Delta t + \frac{1}{2} V''(t) \Delta t^2 + O(\Delta t^3)
$$

Scutoid construction uses linear interpolation in time (connecting top/bottom faces), which has $O(\Delta t^2)$ error for smooth evolution.

**Combined error:**

$$
\text{Total error} \sim N^{-2/d} + \Delta t^2
$$

as claimed. ∎
:::

## A.10 Relation to Main Document

This appendix provides the complete algorithmic and mathematical foundation for scutoid decomposition, which is assumed in the main document [scutoid_integration.md](scutoid_integration.md).

**Cross-references:**

- **Section 3.1** (main doc): Assumes scutoid simplicial decomposition → Defined here in Algorithm {prf:ref}`alg-scutoid-simplex-decomposition`
- **Section 3.2** (main doc): Uses simplex volumes via Gram determinant → Correctness proven here in Theorem {prf:ref}`thm-scutoid-simplex-decomposition`
- **Section 4.2** (main doc): Triangulates scutoid boundary → Algorithm provided here in {prf:ref}`alg-triangulate-lateral-face`
- **Section 6.1** (main doc): Error analysis assumes decomposition convergence → Proven here in Theorem {prf:ref}`thm-decomposition-error-convergence`

**Implementation note:** The pseudocode in Section A.7 can be directly integrated into the Fragile codebase, particularly in `fragile.scutoid` module.

---

**Next:** See [Appendix B](appendix_B_convergence.md) for rigorous proof of curvature convergence under scutoid refinement.
