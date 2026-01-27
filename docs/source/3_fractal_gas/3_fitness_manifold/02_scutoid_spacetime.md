(sec-scutoid-spacetime)=
# Scutoid Spacetime: Discrete Geometry from Cloning

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`/3_fractal_gas/2_fractal_set/02_causal_set_theory`, {cite}`gomez2018scutoids`

---

(sec-scutoid-tldr)=
## TLDR

*Notation: $\mathrm{Vor}_i(t)$ = Voronoi cell of walker $i$ at time $t$; $\mathcal{N}_i(t)$ = neighbor set; $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ = scutoid index (total neighbor changes); $d_g$ = Riemannian geodesic distance; $d$ = latent space dimension.*

**Scutoids Bridge Continuous Geometry and Discrete Dynamics**: The Latent Fractal Gas has continuous Riemannian geometry ({doc}`01_emergent_geometry`) but discrete cloning dynamics. Scutoids—$(d+1)$-dimensional polytopes with mid-level vertices—are the natural spacetime cells that accommodate topological transitions when neighbor sets change during cloning.

**Cloning Typically Produces Scutoids**: Theorem {prf:ref}`thm-cloning-implies-scutoid` shows that whenever cloning changes the neighbor set, the spacetime cell must be a scutoid. Under a local Poisson model, large clone displacements make neighbor changes overwhelmingly likely.

**Online Triangulation Algorithm**: Algorithm {prf:ref}`alg-online-triangulation-update` maintains the Voronoi/Delaunay complex with per-timestep cost $O(N) + O(p_{\mathrm{clone}} N \log N)$ under local retriangulation. If the Delaunay complex size is linear and $p_{\mathrm{clone}} \ll 1/\log N$, the amortized cost is $O(N)$ and matches the output-size lower bound (Theorem {prf:ref}`thm-omega-n-lower-bound`).

**Topological Information Rate**: The scutoid index $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ counts neighbor relationship changes. Conjecture {prf:ref}`conj-topological-information-bound` proposes that the rate of topological information generation bounds computational activity.

---

(sec-scutoid-introduction)=
## Introduction

:::{div} feynman-prose
Let me tell you what this chapter is really about. In {doc}`01_emergent_geometry`, we built a beautiful continuous geometry from the fitness landscape. The metric tensor $g = H + \epsilon_\Sigma I$ told us how to measure distances, and we could compute curvature, volumes, geodesics—all the machinery of Riemannian geometry.

But here is the thing: the Latent Fractal Gas is not a purely continuous system. It is discrete in its cloning dynamics. Walkers jump. Walkers die. Walkers clone. At each timestep, the swarm undergoes a violent reorganization where the unfit perish and the fit reproduce. This is not smooth Langevin diffusion—it is a birth-death process with all its discrete messiness.

So how do we reconcile the continuous Riemannian geometry of {doc}`01_emergent_geometry` with the discrete cloning dynamics of the algorithm? The answer is a remarkable geometric object called the **scutoid**.

Scutoids were discovered by biologists studying epithelial tissue—the sheets of cells that form your skin, your gut lining, your blood vessels. When tissue curves, cells cannot be simple prisms (like hexagonal columns). They need to swap neighbors as you move from one surface to another. The geometry that accomplishes this neighbor-swapping is the scutoid: a prismatoid with extra vertices in the middle where edges branch and merge.

Here is the beautiful correspondence: in the Latent Fractal Gas, **cloning events are topologically equivalent to neighbor-swapping**. When a walker dies and is replaced by a clone from somewhere else, the Voronoi tessellation changes its neighbor structure. The spacetime cell connecting the before-state to the after-state is forced to be a scutoid.

This is not merely an analogy. It is a precise mathematical theorem. The scutoid framework gives us a discrete tessellation of spacetime that bridges the gap between the algorithmic reality (discrete cloning events) and the emergent geometry (continuous Riemannian manifold). The scutoids are the atoms of our algorithmic spacetime.
:::

(sec-time-varying-voronoi)=
## Time-Varying Voronoi Tessellation

:::{dropdown} 📖 Hypostructure Reference
:icon: book

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\rho$ (Node 10 ErgoCheck)

**Hypostructure connection:** The Voronoi tessellation is defined on the latent space $\mathcal{Z}$ with the emergent Riemannian metric $g = H + \epsilon_\Sigma I$ from {prf:ref}`def-adaptive-diffusion-tensor-latent`. The geodesic distance $d_g$ is the intrinsic distance in this geometry.

**References:**
- Emergent metric: {prf:ref}`def-adaptive-diffusion-tensor-latent`
- State space: {prf:ref}`def:state-space-fg`
:::

:::{div} feynman-prose
Before we can talk about spacetime cells, we need to understand what the space looks like at each instant. At any moment $t$, the walker positions $\{z_i(t)\}$ partition the latent space $\mathcal{Z}$ into regions. Each region consists of all points closer to one particular walker than to any other. This is the Voronoi tessellation—the most natural way to divide space among a set of points.

The key subtlety is that we are working in the emergent Riemannian geometry, not flat Euclidean space. "Closer" means closer in the geodesic distance $d_g$ induced by the emergent metric $g = H + \epsilon_\Sigma I$. This means Voronoi boundaries are not flat planes but curved hypersurfaces—the set of points equidistant (in the Riemannian sense) from two walker positions.

As the walkers move, the tessellation evolves. Cells expand, contract, gain neighbors, lose neighbors. Understanding this evolution is the key to understanding the spacetime geometry.
:::

:::{prf:definition} Voronoi Tessellation at Time $t$
:label: def-voronoi-tessellation-time-t

At each time slice $t$, walker positions $z_i(t) \in \mathcal{Z}$ define a **Voronoi tessellation** of the latent space:

$$
\mathrm{Vor}_i(t) = \{z \in \mathcal{Z} : d_g(z, z_i) \leq d_g(z, z_j) \; \forall j \neq i\}
$$

where $d_g(z, z')$ is the **Riemannian geodesic distance** in the emergent metric $g = H + \epsilon_\Sigma I$ (see {prf:ref}`def-adaptive-diffusion-tensor-latent`):

$$
d_g(z, z') = \inf_{\gamma: z \to z'} \int_0^1 \sqrt{\dot{\gamma}(s)^T g(\gamma(s)) \dot{\gamma}(s)} \, ds
$$

**Properties:**

1. **Partition**: $\bigcup_{i=1}^N \mathrm{Vor}_i(t) = \mathcal{Z}$ (up to boundaries)
2. **Closure**: Each cell $\mathrm{Vor}_i(t)$ is closed. Under the assumption that the space is a **Hadamard manifold** (complete, simply connected, with **non-positive sectional curvature**) or satisfies CAT(0) geometry, each cell is **geodesically convex** (and hence star-shaped from the walker position $z_i$). For general Riemannian manifolds with arbitrary curvature, geodesic convexity may fail and cells can be non-convex or even disconnected.

   **Note on curvature regime:** The emergent metric $g = H + \epsilon_\Sigma I$ from {doc}`01_emergent_geometry` has curvature determined by the fitness Hessian. In typical optimization landscapes with bounded curvature (ensured by the Gevrey-1 bounds in {doc}`/3_fractal_gas/appendices/14_b_geometric_gas_cinf_regularity_full`), the regularization $\epsilon_\Sigma$ can be chosen to control the sectional curvature. For practical implementations, the local geodesic distance can be approximated by Euclidean distance when $\epsilon_\Sigma \gg \|H\|_{\mathrm{op}}$.
3. **Curved boundaries**: The boundary $\partial \mathrm{Vor}_i(t) \cap \partial \mathrm{Vor}_j(t)$ is the **equidistant hypersurface** (locus of points with $d_g(z, z_i) = d_g(z, z_j)$), which is generally curved when $g$ is non-flat.
:::

:::{prf:definition} Neighbor Set
:label: def-neighbor-set

The **neighbor set** of walker $i$ at time $t$ is:

$$
\mathcal{N}_i(t) = \{j \neq i : \mathrm{Vor}_i(t) \cap \mathrm{Vor}_j(t) \neq \emptyset\}
$$

That is, walker $j$ is a neighbor of walker $i$ if and only if their Voronoi cells share a boundary face of positive measure.

The **interface segment** between neighbors $i$ and $j$ is:

$$
\Gamma_{i,j}(t) = \partial \mathrm{Vor}_i(t) \cap \partial \mathrm{Vor}_j(t)
$$

This is a $(d-1)$-dimensional hypersurface in the $d$-dimensional latent space.
:::

(sec-dual-delaunay-triangulation)=
### The Dual Delaunay Triangulation

:::{div} feynman-prose
The Voronoi tessellation has a dual structure called the Delaunay triangulation. Connect two walker positions with an edge if and only if their Voronoi cells share a face. The result is a simplicial complex that triangulates the convex hull of the walker positions.

Why do we care about the Delaunay triangulation? Because it is the natural arena for discrete differential geometry. When we compute curvature, holonomy, or parallel transport on a discrete point cloud, we work with the Delaunay structure. The Voronoi cells tell us about volumes and regions of influence; the Delaunay edges tell us about connectivity and gradients.

The two structures are mathematically dual: Voronoi vertices correspond to Delaunay simplices, Voronoi edges correspond to Delaunay faces, and so on. This duality will be essential when we develop the online triangulation algorithm.
:::

:::{admonition} Technical Note (Riemannian Setting)
:class: note

On a general Riemannian manifold, the **Delaunay complex** is defined as the nerve of the geodesic Voronoi cells. It is a genuine triangulation of a region only when the Voronoi cells are contractible (for example, when all sites lie inside a common convex normal neighborhood below the injectivity radius, or within a Safe Harbor region). Otherwise the nerve is still a well-defined cell complex but need not triangulate a convex hull.
:::

:::{prf:definition} Delaunay Triangulation (Geodesic Nerve)
:label: def-delaunay-triangulation

The **Delaunay complex** $\mathrm{DT}(t)$ at time $t$ is the **nerve** of the geodesic Voronoi tessellation:

- **Vertices**: Walker positions $\{z_i(t)\}_{i=1}^N$
- **Edges**: $(i, j) \in \mathrm{DT}(t)$ iff $j \in \mathcal{N}_i(t)$
- **Simplices**: A $(k+1)$-tuple $(i_0, \ldots, i_k)$ forms a $k$-simplex iff the corresponding Voronoi cells have a non-empty common intersection

**Duality relations:**

| Voronoi Structure | Delaunay Structure |
|-------------------|-------------------|
| $d$-dimensional cell $\mathrm{Vor}_i$ | Vertex $z_i$ |
| $(d-1)$-dimensional face $\Gamma_{i,j}$ | Edge $(i, j)$ |
| Vertex (intersection of $d+1$ cells) | $d$-simplex |

**Nerve theorem (local triangulation):** If the Voronoi cells are contractible and their intersections are contractible (e.g., inside a convex normal neighborhood), then $\mathrm{DT}(t)$ triangulates the covered region and is homotopy equivalent to it. In the absence of these conditions, $\mathrm{DT}(t)$ should be treated as a general cell complex rather than a global triangulation.
:::

(sec-scutoid-cell-definition)=
## Scutoid Cell Definition

:::{div} feynman-prose
Now we come to the heart of the matter. How do we connect two time slices into a spacetime structure?

Imagine you have a Voronoi tessellation at time $t$ and another at time $t + \Delta t$. The simplest case is when everything lines up perfectly: each walker at time $t$ corresponds to the same walker at time $t + \Delta t$, and the neighbor sets are unchanged. In this case, you can connect the two tessellations with simple prisms—each Voronoi cell at the bottom connects straight up to the corresponding cell at the top.

But what happens when cloning occurs? Walker $i$ at position $z_i$ dies and is replaced by a clone of walker $j$ from position $z_j$. The new walker appears somewhere completely different in space. The Voronoi cell that contained $z_i$ is gone, replaced by a cell around the cloning position. The neighbor sets change: the new cell has different neighbors than the old one.

How do you connect a polygon with 5 neighbors at the bottom to a polygon with 6 neighbors at the top? You cannot do it with a prism. The only way is to have vertices in the middle where edges branch and merge. This is the scutoid.

The scutoid is not some exotic geometric curiosity. It is the unique answer to a fundamental topological problem: how to fill the space between two parallel surfaces when the combinatorial structure changes. Nature discovered this solution in epithelial tissue; we rediscover it in algorithmic spacetime.
:::

:::{prf:definition} Boundary Correspondence Map
:label: def-boundary-correspondence-map

Let $F_{\mathrm{bottom}} = \mathrm{Vor}_j(t)$ and $F_{\mathrm{top}} = \mathrm{Vor}_i(t + \Delta t)$ be the bottom and top faces of a spacetime cell, with neighbor sets $\mathcal{N}_j(t)$ and $\mathcal{N}_i(t + \Delta t)$ respectively.

The **shared neighbor set** is:

$$
\mathcal{N}_{\mathrm{shared}} = \mathcal{N}_j(t) \cap \mathcal{N}_i(t + \Delta t)
$$

For each $k \in \mathcal{N}_{\mathrm{shared}}$, the **boundary correspondence map** $\phi_k: \Gamma_{j,k}(t) \to \Gamma_{i,k}(t + \Delta t)$ is any **measure-preserving correspondence** between the $(d-1)$-dimensional interfaces. Let $\mu_{\mathrm{bottom}}$ and $\mu_{\mathrm{top}}$ denote the $(d-1)$-dimensional Hausdorff measures induced by $g$ on $\Gamma_{j,k}(t)$ and $\Gamma_{i,k}(t + \Delta t)$, and assume both are finite and non-zero. A valid correspondence satisfies:

$$
(\phi_k)_* \mu_{\mathrm{bottom}} = \mu_{\mathrm{top}}.
$$

**Canonical choice in a convex normal neighborhood:** If the two interfaces lie inside a common convex normal neighborhood, one can take $\phi_k$ to be the optimal transport map between the normalized measures with cost $d_g^2$ (unique a.e. under absolute continuity). Any such $\phi_k$ yields a well-defined ruled lateral face.

**Existence (measure-theoretic):** Any two standard Borel spaces with finite, non-zero measures admit a measure-preserving bijection modulo null sets, so $\phi_k$ always exists up to measure zero.

**Critical observation**: For neighbors $\ell \in \mathcal{N}_j(t) \setminus \mathcal{N}_i(t + \Delta t)$ (lost neighbors), there is no corresponding segment on the top face. The correspondence map is **undefined**.
:::

:::{prf:definition} Scutoid Cell
:label: def-scutoid-cell

A **scutoid** $\mathcal{S}_i$ is a $(d+1)$-dimensional polytope in the swarm spacetime manifold $\mathcal{M} = \mathcal{Z} \times [0, T]$, bounded by:

**1. Bottom face** ($t = t_0$):

$$
F_{\mathrm{bottom}} = \mathrm{Vor}_j(t_0)
$$
where $j$ is the parent (for cloned walkers) or the walker itself (for persistent walkers).

**2. Top face** ($t = t_0 + \Delta t$):

$$
F_{\mathrm{top}} = \mathrm{Vor}_i(t_0 + \Delta t)
$$

**3. Lateral faces** (for shared neighbors):
For each $k \in \mathcal{N}_{\mathrm{shared}}$, the lateral face $\Sigma_k$ is the **ruled surface** swept by geodesic segments:

$$
\Sigma_k = \bigcup_{p \in \Gamma_{j,k}(t_0)} \gamma_{p \to \phi_k(p)}
$$
where $\gamma_{p \to \phi_k(p)}$ is the spacetime geodesic from $(p, t_0)$ to $(\phi_k(p), t_0 + \Delta t)$.

**Geodesic well-posedness:** When these endpoints lie inside a convex normal neighborhood in spacetime, the minimizing geodesic is unique; otherwise choose any minimizing geodesic to define the ruled surface.

**4. Mid-level structure** (when $\mathcal{N}_j(t_0) \neq \mathcal{N}_i(t_0 + \Delta t)$):

- **Mid-level time**: $t_{\mathrm{mid}} = t_0 + \Delta t / 2$
- **For lost neighbors** $\ell \in \mathcal{N}_j(t_0) \setminus \mathcal{N}_i(t_0 + \Delta t)$:
  - Geodesic rulings from $\Gamma_{j,\ell}(t_0)$ **terminate** at mid-level vertices
- **For gained neighbors** $m \in \mathcal{N}_i(t_0 + \Delta t) \setminus \mathcal{N}_j(t_0)$:
  - Geodesic rulings to $\Gamma_{i,m}(t_0 + \Delta t)$ **originate** from mid-level vertices

The **mid-level vertices** are the branching points where these geodesics meet.
:::

:::{div} feynman-prose
Let me make sure you understand why the mid-level vertices are necessary. Suppose the bottom face has neighbors $\{A, B, C, D, E\}$ (a pentagon) and the top face has neighbors $\{A, B, C, D, F\}$ (still five neighbors, but $E$ is replaced by $F$).

For neighbors $A, B, C, D$ that appear in both lists, we can connect the corresponding boundary segments with ruled surfaces. No problem.

But what about $E$ and $F$? The boundary segment adjacent to $E$ at the bottom has no corresponding segment at the top. The boundary segment adjacent to $F$ at the top has no corresponding segment at the bottom.

The only way to close the surface is to have the geodesics from the $E$-segment terminate at some intermediate location, and have geodesics to the $F$-segment originate from that same location. This intermediate location is the mid-level vertex. It is not something we choose to add—it is topologically forced on us by the mismatch in neighbor sets.
:::

(sec-cloning-topological-transitions)=
## Cloning Events as Topological Transitions

:::{div} feynman-prose
Now we prove the central theorem: cloning events necessarily produce scutoid geometry. This is not a modeling choice or a convenient approximation. It is a mathematical inevitability.

The argument has two parts. First, we show that cloning changes the neighbor set with probability one. When a walker at position $z_i$ is replaced by a clone from position $z_j$, the new Voronoi cell has different neighbors (generically). Second, we show that different neighbor sets force scutoid geometry—there is no way to construct a simple prism when the top and bottom faces have incompatible combinatorial structure.
:::

:::{admonition} Technical Note (Precise Scope)
:class: note

Cloning does **not** guarantee a neighbor change in every realization. The rigorous statement is: **if** the neighbor set changes, then a scutoid is forced; under standard stochastic-geometry assumptions, large clone displacements make this event overwhelmingly likely.
:::

:::{prf:theorem} Cloning Implies Scutoid Geometry
:label: thm-cloning-implies-scutoid

Let $e_i$ be an episode (walker trajectory) traversing the interval $[t, t + \Delta t]$.

**Statement:**

1. **Persistence with No Critical Event**: If episode $e_i$ persists without cloning and the Delaunay complex is non-degenerate on $(t, t + \Delta t)$ (no critical configuration), then $\mathcal{N}_i(t) = \mathcal{N}_i(t + \Delta t)$ and the spacetime cell is a **Prism** (Type 0).

2. **Cloning with Neighbor Change**: If episode $e_i$ ends at time $t$ and is replaced by a clone $e_{i'}$ from parent $e_j$ at a different position, and if $\mathcal{N}_i(t) \neq \mathcal{N}_{i'}(t + \Delta t)$, then the spacetime cell is a **Scutoid** (Type $\geq 1$).

3. **Genericity Under a Local Poisson Model**: Assume that, in a normal coordinate chart around the clone, the other walkers are distributed as a Poisson process of intensity $\rho$, the clone displacement is $r = \|z_i - z_j\|$, and there is a geometric separation condition: **if** $\mathcal{N}_i(t) = \mathcal{N}_{i'}(t + \Delta t)$, then a geodesic tube of volume at least $c_0 r^d$ between $z_i$ and $z_{i'}$ must be empty (with $c_0>0$ depending only on dimension and curvature bounds). Then
$$
\mathbb{P}(\mathcal{N}_i(t) = \mathcal{N}_{i'}(t + \Delta t)) \leq \exp\left(-c \cdot \frac{r^d}{\ell^d}\right),
$$
with $\ell \sim \rho^{-1/d}$ and $c>0$ depending only on dimension and local geometry. Thus for $r \gg \ell$, cloning produces a scutoid with high probability.

**Remark:** The separation condition holds, for example, when the configuration is quasi-uniform and the displacement $r$ exceeds a fixed multiple of the local feature size.

*Proof.*

**Part 1: Persistence implies prismatic geometry (no critical event).**

When no cloning occurs, the walker position evolves continuously via the Langevin SDE ({prf:ref}`def-fractal-set-sde`):

$$
dz_i = v_i \, dt + \Sigma_{\mathrm{reg}}(z_i, S) \circ dW_i
$$
where $v_i$ is the drift velocity, $\Sigma_{\mathrm{reg}}$ is the adaptive diffusion tensor, and $dW_i$ is Brownian noise. The key point is that the trajectory is continuous (no jumps).

The Voronoi boundaries deform continuously under continuous motion of the seeds. The neighbor graph of a Voronoi tessellation is locally constant under such deformations, except at **critical configurations** where the Delaunay complex is degenerate. In Euclidean space, this occurs when $d+2$ or more seeds lie on a common sphere; in Riemannian geometry, degeneracy occurs when seeds lie on a common geodesic sphere (equidistant from some center point). If the interval $(t, t + \Delta t)$ contains no such critical time, then $\mathcal{N}_i(t) = \mathcal{N}_i(t + \Delta t)$ and the boundary correspondence map $\phi_k$ is defined for all neighbors. The spacetime cell is a prism with no mid-level vertices.

**Part 2: Cloning with neighbor change forces scutoid geometry.**

Assume $\mathcal{N}_i(t) \neq \mathcal{N}_{i'}(t + \Delta t)$. Let:
- $\mathcal{N}_{\mathrm{lost}} = \mathcal{N}_i(t) \setminus \mathcal{N}_{i'}(t + \Delta t)$ (neighbors at bottom only)
- $\mathcal{N}_{\mathrm{gained}} = \mathcal{N}_{i'}(t + \Delta t) \setminus \mathcal{N}_i(t)$ (neighbors at top only)

If the spacetime cell were a prism $P = F_{\mathrm{bottom}} \times [0,1]$, then there would exist a boundary homeomorphism $h: \partial F_{\mathrm{bottom}} \to \partial F_{\mathrm{top}}$ that matches each boundary interface to a corresponding interface. This induces a bijection between neighbor sets. When $\mathcal{N}_{\mathrm{lost}} \neq \emptyset$ or $\mathcal{N}_{\mathrm{gained}} \neq \emptyset$, no such bijection exists. Therefore a prismatic boundary cannot close, and any valid cell complex must introduce intermediate branching (mid-level) faces and vertices. By Definition {prf:ref}`def-scutoid-cell`, the resulting cell is a scutoid.

**Part 3: Genericity under a local Poisson model.**

Assume the other walkers in a normal coordinate chart around the clone form a Poisson process of intensity $\rho$ and the clone displacement is $r = \|z_i - z_j\|$. Under the separation condition above, neighbor-set equality requires an empty tube of volume at least $c_0 r^d$, so the Poisson void probability gives:

$$
\mathbb{P}(\mathcal{N}_i(t) = \mathcal{N}_{i'}(t + \Delta t)) \leq \exp\left(-c \cdot \rho r^d\right) = \exp\left(-c \cdot \frac{r^d}{\ell^d}\right),
$$

where $\ell \sim \rho^{-1/d}$ and $c>0$ depends on $d$ and local curvature bounds. This yields the stated bound.

$\square$
:::

:::{dropdown} 📖 Hypostructure Proof Path
:icon: book

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\pi$ (Node 8), $\mathrm{TB}_O$ (Node 9)

**Framework path (sketch):**

1. Use {prf:ref}`def-fractal-set-cst-edges` and {prf:ref}`def-fractal-set-cloning-score` to identify cloning events and parent-child relations in the CST.
2. Invoke the QSD sampling density from {prf:ref}`thm-fractal-adaptive-sprinkling` to justify a local Poisson approximation for walker locations (within Safe Harbor bounds).
3. Apply the Poisson void probability to obtain the exponential bound on neighbor-set equality and conclude scutoid formation for large displacement.
4. The topological obstruction argument is independent of framework assumptions and matches Definition {prf:ref}`def-scutoid-cell`.
:::

:::{dropdown} 📖 Hypostructure Reference
:icon: book

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\pi$ (Node 8), $\mathrm{TB}_O$ (Node 9)

**Hypostructure connection:** The theorem connects the discrete cloning dynamics ({prf:ref}`def-fractal-set-cloning-score`) to the emergent spacetime geometry. The probability bound uses the QSD sampling density from {prf:ref}`thm-fractal-adaptive-sprinkling`.

**References:**
- Cloning mechanism: {prf:ref}`def-fractal-set-cloning-score`
- CST edges: {prf:ref}`def-fractal-set-cst-edges`
- QSD density: {prf:ref}`thm-fractal-adaptive-sprinkling`
:::

(sec-cell-type-classification)=
### Classification of Cell Types

:::{prf:definition} Scutoid Type Classification
:label: def-scutoid-type-classification

Spacetime cells in the scutoid tessellation are classified by their **topological complexity**:

**Type 0: Prism** (No neighbor change)
- $\mathcal{N}_{\mathrm{lost}} = \mathcal{N}_{\mathrm{gained}} = \emptyset$
- No mid-level branching
- Represents: A trajectory segment with no neighbor changes (typically persistent diffusion without critical events)
- Physics: Laminar flow, exploitation phase

**Type 1: Simple Scutoid** (Single neighbor swap)
- $|\mathcal{N}_{\mathrm{lost}}| = |\mathcal{N}_{\mathrm{gained}}| = 1$ (scutoid index $\chi = 2$)
- Minimal mid-level branching (a single vertex in $d=2$, or a minimal branching feature in higher $d$)
- Represents: Standard cloning event or a single Delaunay flip
- Physics: Plastic deformation, adaptive exploration

**Type $\geq 2$: Complex Scutoid** (Multiple neighbor changes or asymmetric changes)
- $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}| \geq 2$, excluding the symmetric single-swap case (Type 1)
- Mid-level branching forms a higher-complexity cell complex; the exact vertex count depends on geometry
- Represents: Major topological reorganization, "teleportation" to distant basin
- Physics: Phase transition, turbulent exploration

The **scutoid index** $\chi_{\mathrm{scutoid}}(\mathcal{S}) = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ counts the total number of neighbor changes (lost + gained). Note: $\chi$ need not be even when $|\mathcal{N}_{\mathrm{lost}}| \neq |\mathcal{N}_{\mathrm{gained}}|$.
:::

:::{div} feynman-prose
Think about what these types mean physically.

Type 0 (prism) is the boring case—the walker just diffuses around, not dying, not changing its neighborhood. This happens during exploitation phases when the swarm has settled into local optima.

Type 1 (simple scutoid) is the typical cloning event. One neighbor relationship is severed, one new relationship is formed. This is the microscopic "atom" of exploration—a small topological change that reshuffles the local structure.

Complex scutoids (Type $\geq 2$) are dramatic. Multiple neighbor relationships change simultaneously. This happens when a walker "teleports" to a completely different region of the fitness landscape—say, from one basin of attraction to another. The topology tears and reconnects in multiple places.

The relative frequencies of these types characterize the dynamical state of the swarm. A prism-dominated tessellation indicates convergence and exploitation. A scutoid-dominated tessellation indicates active exploration and phase transitions. We will make this quantitative with the topological information rate.
:::

(sec-euler-characteristic)=
### Euler Characteristic and Topological Information

:::{prf:proposition} Topological Complexity of Scutoid Tessellation
:label: prop-euler-characteristic-scutoid

The **topological complexity** of the scutoid tessellation is characterized by the cumulative scutoid index:

$$
\mathcal{K}_{\mathrm{total}}([0,T]) = \sum_{\text{cells } \mathcal{S}} \chi_{\mathrm{scutoid}}(\mathcal{S})
$$

**Counting argument**: For $N$ walkers over time interval $[0,T]$ with timestep $\Delta t$:
- Total spacetime cells: $N \cdot (T/\Delta t)$
- Let $p_{\mathrm{clone}}$ be the cloning probability per step and $p_{\mathrm{crit}}$ the probability of a non-cloning neighbor-change event (critical Delaunay flip).
- Prismatic cells (Type 0): $N_{\mathrm{prism}} = N(1 - p_{\mathrm{clone}} - p_{\mathrm{crit}}) \cdot (T/\Delta t)$
- Scutoid cells (Type $\geq 1$): $N_{\mathrm{scutoid}} = N (p_{\mathrm{clone}} + p_{\mathrm{crit}}) \cdot (T/\Delta t)$

**Topological interpretation**: The cumulative scutoid index $\mathcal{K}_{\mathrm{total}}$ counts the total number of neighbor-relationship changes in the tessellation. Each unit of $\mathcal{K}_{\mathrm{total}}$ represents one "topological transaction" where a neighbor relationship is either created or destroyed.

**Relation to boundary structure**: For a single scutoid cell $\mathcal{S}$, the number of lateral faces is:

$$
|\text{lateral faces}| = |\mathcal{N}_{\mathrm{shared}}| + |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|
$$
where mid-level vertices connect faces from lost neighbors to faces from gained neighbors.
:::

(sec-dynamic-delaunay-algorithm)=
## Dynamic Delaunay Algorithm

:::{div} feynman-prose
Now we get practical. How do we actually compute and maintain these tessellations as the swarm evolves?

The naive approach is to rebuild everything from scratch at each timestep: take the $N$ walker positions, run a Voronoi/Delaunay algorithm, output the result. The cost is **output-sensitive**: at least $O(|\mathrm{DT}|)$, $O(N \log N)$ in 2D, and superlinear in $N$ for higher dimensions or curved geometry (typically $O(N^{\lceil d/2\rceil})$ in the worst case).

But this is wasteful. Between timesteps, most walkers move only slightly (continuous diffusion), and only a few undergo cloning (discrete jumps). The tessellation at time $t + \Delta t$ is a small perturbation of the tessellation at time $t$. We are throwing away valuable information by recomputing from scratch.

The breakthrough insight is that the Causal Spacetime Tree (CST) tells us exactly what changed: which walkers moved continuously, which walkers cloned. This is precisely the information needed to update the triangulation incrementally.

The result is an **online algorithm** that maintains the triangulation with $O(N)$ amortized cost per timestep—a factor of $\log N$ faster than batch recomputation. For large swarms, this makes real-time geometric analysis feasible.
:::

:::{prf:algorithm} Online Scutoid-Guided Triangulation Update
:label: alg-online-triangulation-update

**Data Structures:**
- `DT`: Current Delaunay triangulation of walker positions
- `VT`: Dual Voronoi tessellation
- `VertexMap`: Map from walker ID to vertex in `DT`

**Initialization** (at $t = 0$):
1. Compute initial Delaunay triangulation `DT(0)` from positions $\{z_i(0)\}$
2. Compute dual Voronoi tessellation `VT(0)`
3. Initialize `VertexMap`

**Cost**: $O(N \log N)$ (one-time)

**Per-Timestep Update** ($t \to t + \Delta t$):

**Step 1: Identify Changes from CST** — $O(N)$

```{code-block} python
:caption: Identify walker state changes from CST

MovedWalkers = []      # (walker_id, z_old, z_new)
ClonedWalkers = []     # (dead_id, z_new, parent_id)

for walker_id in range(N):
    edge = CST.get_edge(walker_id, t, t + dt)

    if edge.type == "SDE_evolution":
        # Local move (typically prism unless a critical event occurs)
        MovedWalkers.append((walker_id, edge.z_old, edge.z_new))

    elif edge.type == "cloning":
        # Teleport (non-prismatic scutoid)
        ClonedWalkers.append((walker_id, edge.z_new, edge.parent_id))
```

**Step 2: Update Locally Moved Walkers** — Output-sensitive, $O(k)$ per walker (conflict-region size)

```{code-block} python
:caption: Update Delaunay structure for moved walkers

for (walker_id, z_old, z_new) in MovedWalkers:
    vertex = VertexMap[walker_id]
    vertex.position = z_new

    # Restore Delaunay property (Lawson flips in d=2; local retriangulation in d>2)
    RestoreDelaunay(DT, vertex)

    # Update corresponding Voronoi cell
    UpdateVoronoiCell(VT, vertex)
```

**Small displacement condition:** When the displacement $\|z_{\mathrm{new}} - z_{\mathrm{old}}\| \ll \ell_{\mathrm{local}}$ (local feature size), the conflict region is small. For the Langevin SDE with diffusion $\Sigma_{\mathrm{reg}}$ and timestep $\Delta t$, typical displacements scale as $O(\sqrt{\Delta t})$, so the condition is met when $\Delta t \ll \ell_{\mathrm{local}}^2 / \|\Sigma_{\mathrm{reg}}\|_{\mathrm{op}}^2$. In dense clusters where $\ell_{\mathrm{local}} \to 0$, the conflict region size $k$ can grow, and update cost scales with $k$ (output-sensitive), potentially as large as $O(|\mathrm{DT}|)$ in the worst case.

**Step 3: Update Cloned Walkers** — $O(N^{1/d})$ expected per walker ($O(\log N)$ with index)

```{code-block} python
:caption: Handle cloning events with point location

for (dead_id, z_new, parent_id) in ClonedWalkers:
    # Phase A: Delete dead walker
    dead_vertex = VertexMap[dead_id]
    DT.remove_vertex(dead_vertex)

    # Phase B: Insert new walker
    containing_simplex = DT.locate(z_new)  # expected O(N^{1/d}) walk; O(log N) with index
    new_vertex = DT.insert_vertex(z_new, containing_simplex)
    RestoreDelaunay(DT, new_vertex)

    # Update mapping
    VertexMap[dead_id] = new_vertex
```

**RestoreDelaunay:** In $d=2$, this is the Lawson-flip routine (Algorithm {prf:ref}`alg-lawson-flip`). In $d>2$ or on curved manifolds, it denotes local retriangulation of the conflict region inside a convex normal neighborhood.

**Total Complexity per Timestep:**

$$
T(N) = O(N) + O\!\left(\sum_{\text{moved}} k_i\right) + O\!\left(\sum_{\text{clones}} (\log N + k_i)\right)
$$

Under quasi-uniform sampling and small displacements, $\mathbb{E}[k_i]=O(1)$ and point location with an index gives $\mathbb{E}[T(N)] = O(N) + O(p_{\mathrm{clone}} N \log N)$. If $p_{\mathrm{clone}} \ll 1/\log N$ and $|\mathrm{DT}|=\Theta(N)$, this yields **$O(N)$ amortized** complexity per timestep.
:::

(sec-lawson-flip)=
### Local Delaunay Restoration (Lawson Flips in 2D)

:::{div} feynman-prose
The Lawson flip is the workhorse of incremental Delaunay maintenance. When you move a vertex, some of the incident simplices may violate the Delaunay criterion (empty circumsphere property). The Lawson algorithm fixes this by iteratively "flipping" edges until the criterion is restored.

Here is the beautiful thing: for small vertex movements, the number of flips is $O(1)$ on average. The disruption is local. Move a vertex by a small distance, and only a handful of nearby simplices need adjustment. This is what makes the online algorithm fast.

The key insight is geometric: the Delaunay triangulation is a stable structure. Small perturbations cause small changes. Only large movements (like cloning teleportation) cause global reorganization.
:::

:::{admonition} Technical Note (General $d$)
:class: note

Lawson flips are guaranteed to restore a Delaunay triangulation only in $d=2$. In higher dimensions or on curved manifolds, the safe, rigorous update is to **retriangulate the conflict region** (the star of the moved/inserted vertex) using a local geodesic Delaunay construction inside a convex normal neighborhood.
:::

:::{prf:algorithm} Lawson Flip for Delaunay Restoration (2D)
:label: alg-lawson-flip

**Input:** Delaunay triangulation `DT`, vertex `v` whose position was updated

**Output:** Restored Delaunay triangulation

**Procedure:**

```{code-block} python
:caption: Lawson flip algorithm for Delaunay restoration

def LawsonFlip(DT, v):
    # Initialize queue with simplices incident to v
    Q = Queue()
    for simplex in DT.incident_simplices(v):
        Q.enqueue(simplex)

    marked = set()

    while not Q.empty():
        S = Q.dequeue()

        if S in marked:
            continue
        marked.add(S)

        # Check Delaunay criterion: circumsphere of S contains no other vertices
        if is_delaunay(S):
            continue

        # Find violated face
        F = find_violated_face(S)
        S_adjacent = DT.adjacent_simplex(S, F)

        if S_adjacent is None:
            continue  # Boundary face

        # Perform flip: replace S and S_adjacent with new simplices
        new_simplices = flip(DT, S, S_adjacent, F)

        for new_S in new_simplices:
            Q.enqueue(new_S)
```

**Complexity Analysis (local retriangulation viewpoint):**

- Let $k$ be the number of simplices in the conflict region (the star of the moved/inserted vertex).
- In $d=2$, Lawson flips terminate and run in $O(k)$ time.
- In $d>2$, a safe update is to delete the conflict region and retriangulate it; this costs $O(k \log k)$ in practice (output-sensitive).

Without regularity assumptions, $k$ can be as large as $|\mathrm{DT}|$. Under quasi-uniform sampling in a bounded-curvature region (points are $\delta$-separated and $\epsilon$-dense with $\epsilon/\delta$ bounded), $k$ is typically $O(1)$ in expectation for small displacements.
:::

(sec-jump-and-walk)=
### Jump-and-Walk Point Location

:::{div} feynman-prose
For cloning events, we need to insert a new vertex at position $z_{\mathrm{new}}$. The first step is finding which simplex contains this position. This is the **point location** problem.

The naive approach—check every simplex—takes $O(N)$ time. Much too slow.

The jump-and-walk algorithm does better. Start from some "hint" simplex (say, one incident to the parent walker), then walk through the triangulation toward the target. At each step, check which face of the current simplex is "toward" the target, and cross to the adjacent simplex. Eventually you reach the simplex containing the target.

The expected number of steps is $O(\log N)$ for typical point sets. The walk length depends on how far the target is from the starting hint, but for cloning events (where the new position is near the parent), the hint is excellent and the walk is short.
:::

:::{admonition} Technical Note (Point Location)
:class: note

In general, pure walk-based point location has expected complexity $O(N^{1/d})$ for random points. Achieving $O(\log N)$ requires an auxiliary spatial index (e.g., a hierarchical net or cover tree) or stronger regularity assumptions.
:::

:::{prf:algorithm} Jump-and-Walk Point Location
:label: alg-jump-and-walk

**Input:** Delaunay triangulation `DT`, query point $z$

**Output:** Simplex containing $z$

**Procedure:**

```{code-block} python
:caption: Jump-and-walk point location

def locate(DT, z):
    # Phase 1: Jump to a nearby simplex (use hint)
    current = get_hint_simplex(DT, z)

    # Phase 2: Walk toward target
    while True:
        if contains(current, z):
            return current

        # Find face that z is "beyond"
        exit_face = find_exit_face(current, z)

        # Move to adjacent simplex
        current = DT.adjacent_simplex(current, exit_face)

        if current is None:
            return None  # z is outside the triangulated region
```

**Complexity:**

- Jump phase: $O(1)$ using a hint simplex or spatial hashing
- Walk phase: $O(N^{1/d})$ expected for random points
- With a good hint and bounded jitter: expected $O(\mathrm{dist}/\ell)$ steps, where $\mathrm{dist}$ is the hint-to-target distance and $\ell$ is typical edge length
- **Total: $O(N^{1/d})$ expected** without additional point-location structures; **$O(\log N)$ expected** if a hierarchical spatial index is maintained

**Theoretical basis:** The expected walk length for uniformly random queries is $O(N^{1/d})$ in dimension $d$ {cite}`mucke1999fast`. With a good hint (e.g., the parent's simplex for cloning events), the walk length is $O(\mathrm{dist}/\ell)$ where $\mathrm{dist}$ is the distance from hint to target and $\ell$ is the typical edge length. For cloning events where the clone position has Gaussian jitter $\xi \sim \mathcal{N}(0, \sigma^2 I)$, the expected walk length is $O(\sigma/\ell) = O(1)$ when $\sigma \sim \ell$.
:::

(sec-complexity-analysis)=
## Complexity Analysis

:::{prf:theorem} Amortized Complexity of Online Triangulation
:label: thm-amortized-complexity

The online scutoid-guided triangulation update (Algorithm {prf:ref}`alg-online-triangulation-update`) achieves:

**Per-Timestep Complexity:**

$$
T(N) = O(N) + O\!\left(\sum_{\text{moved}} k_i\right) + O\!\left(\sum_{\text{clones}} (\log N + k_i)\right)
$$

where $k_i$ is the size of the conflict region (number of simplices retriangulated) for update $i$.

**Expected/Amortized Complexity** (under quasi-uniform sampling and small displacements):

$$
\mathbb{E}[T(N)] = O(N) + O(p_{\mathrm{clone}} \cdot N \cdot \log N)
$$

and therefore $\bar{T}(N) = O(N)$ when $p_{\mathrm{clone}} \ll 1/\log N$ and the Delaunay complex size is linear.

**Typical Regime (linear-size Delaunay):** For $p_{\mathrm{clone}} \in [0.01, 0.1]$ and $N \in [10^3, 10^6]$, $p_{\mathrm{clone}} \log N$ is often $< 1$, yielding **effective $O(N)$ per timestep**.

**Speedup vs. Batch (output-sensitive):**

$$
\text{Speedup} = \frac{O(|\mathrm{DT}| \log N)}{O(N)} = O(\log N) \quad \text{when } |\mathrm{DT}| = \Theta(N)
$$

For $N = 10^6$: approximately $20\times$ speedup.

*Proof.*

**SDE moves (Type 0 prisms):**
- Number of persistent walkers: $N(1 - p_{\mathrm{clone}})$
- Cost per walker: $O(k_i)$ (local conflict retriangulation)
- Total: $O\!\left(\sum_{\text{moved}} k_i\right)$

**Cloning events (Type $\geq 1$ scutoids):**
- Number of cloned walkers: $N \cdot p_{\mathrm{clone}}$
- Cost per walker: $O(\log N)$ (point location with index) + $O(k_i)$ (local retriangulation)
- Total: $O\!\left(\sum_{\text{clones}} (\log N + k_i)\right)$

Combining gives the stated bound. Under quasi-uniform sampling, $\mathbb{E}[k_i]=O(1)$ for small displacements, yielding $\mathbb{E}[T(N)] = O(N) + O(p_{\mathrm{clone}} N \log N)$.

$\square$
:::

:::{dropdown} 📖 Hypostructure Proof Path
:icon: book

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\rho$ (Node 10 ErgoCheck)

**Framework path (sketch):**

1. Use {prf:ref}`def-fractal-set-cst-edges` to classify CST edges into SDE moves and cloning events.
2. Invoke {prf:ref}`thm-fractal-adaptive-sprinkling` to justify quasi-uniform sampling in Safe Harbor, yielding bounded expected conflict-region size.
3. Combine with standard output-sensitive bounds to recover $\mathbb{E}[T(N)] = O(N) + O(p_{\mathrm{clone}} N \log N)$.
:::

(sec-lower-bound-proof)=
### Lower Bound Proof

:::{prf:theorem} $\Omega(|\mathrm{DT}|)$ Lower Bound for Tessellation Update
:label: thm-omega-n-lower-bound

Any algorithm that correctly updates a Voronoi/Delaunay complex of $N$ points after arbitrary point movements must take, in the worst case, at least $\Omega(|\mathrm{DT}|)$ time.

**Conclusion:** The update complexity is **asymptotically optimal** up to the output size.

*Proof.*

**Information-theoretic argument** (following {cite}`preparata1985computational`):

The output is a complete geometric data structure representing:
- $N$ vertex positions (walker coordinates)
- $\Theta(|\mathrm{DT}|)$ simplices
- Adjacency information for each simplex

The **output size is $\Theta(|\mathrm{DT}|)$**. In fixed dimension $d$, the number of Delaunay simplices can be as large as $\Theta(N^{\lceil d/2\rceil})$ in the worst case; in favorable regimes (e.g., quasi-uniform sampling in bounded curvature), it is often $\Theta(N)$.

Any algorithm producing $\Theta(|\mathrm{DT}|)$ output requires at least $\Omega(|\mathrm{DT}|)$ time—it is impossible to write the output in fewer operations. This is a fundamental information-theoretic lower bound that applies to any computational model.

**Worst-case construction:**

Consider a global rotation: all $N$ walkers rotate by angle $\theta$ around a center. The combinatorial structure may be unchanged, but all vertex coordinates must be updated. The algorithm must touch all $\Theta(|\mathrm{DT}|)$ geometric objects to produce correct output.

**Conclusion:** Lower bound $\Omega(|\mathrm{DT}|)$, upper bound output-sensitive. When $|\mathrm{DT}| = \Theta(N)$, the algorithm achieves $O(N)$ and is optimal.

$\square$
:::

:::{div} feynman-prose
This is a satisfying result. We cannot do better than $O(N)$—the output is that big, and we have to at least look at it. Our algorithm achieves $O(N)$ amortized. We are as good as theoretically possible.

The improvement over batch computation ($O(N \log N)$) comes from exploiting temporal coherence. Most of the structure is preserved between timesteps; we only update what changed. This is the power of incremental algorithms.
:::

:::{admonition} Technical Note (Output-Size Lower Bound)
:class: note

The rigorous lower bound is $\Omega(|\mathrm{DT}|)$. When the Delaunay complex is superlinear in $N$ (possible in $d\ge 3$), the optimal update cost scales accordingly.
:::

(sec-topological-information-rate)=
## Topological Information Rate

:::{div} feynman-prose
Here is a deep question: how fast is the swarm "processing information" about the fitness landscape?

One way to think about this: every time a walker clones, the swarm is making a topological decision. It is choosing to redirect resources from a low-fitness region to a high-fitness region. This decision destroys one Voronoi cell and creates another. The neighborhood structure changes. Information is being encoded into the geometry.

The rate at which this happens—the rate of topological changes—is a measure of computational activity. A swarm that sits still (all prisms) is not computing. A swarm with many scutoids is actively exploring, restructuring, processing.

We conjecture that this topological information rate is bounded by the scutoid complexity. The more complex the scutoids (higher scutoid index $\chi$), the more information is being processed per cloning event.
:::

:::{prf:definition} Topological Information Rate
:label: def-topological-information-rate

The **topological information rate** of the swarm at time $t$ is:

$$
\dot{I}_{\mathrm{topo}}(t) = \sum_{\text{cloning events at } t} \chi_{\mathrm{scutoid}}(\mathcal{S})
$$

where the sum is over all spacetime cells corresponding to cloning events in the interval $[t, t + \Delta t)$.

**Note:** Neighbor changes can also occur without cloning (via critical Delaunay events). The definition above focuses on cloning-driven topology changes; a fully general rate would sum over all neighbor-change events.

**Alternative formulation:**

$$
\dot{I}_{\mathrm{topo}} = \langle \chi_{\mathrm{scutoid}} \rangle \cdot f_{\mathrm{clone}}
$$

where:
- $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ is the scutoid index (total neighbor changes)
- $\langle \chi_{\mathrm{scutoid}} \rangle$ is the average scutoid index per cloning event
- $f_{\mathrm{clone}} = N \cdot p_{\mathrm{clone}} / \Delta t$ is the cloning frequency (events per unit time)

If non-cloning neighbor changes are included, replace $f_{\mathrm{clone}}$ with the total neighbor-change event rate.
:::

:::{prf:conjecture} Topological Information Rate Bound
:label: conj-topological-information-bound

The topological information rate $\dot{I}_{\mathrm{topo}} = \langle \chi_{\mathrm{scutoid}} \rangle \cdot f_{\mathrm{clone}}$ is bounded by the density of scutoid vertices in spacetime:

$$
\dot{I}_{\mathrm{topo}} \leq c \cdot \rho_{\mathrm{scutoid}}
$$

where $\rho_{\mathrm{scutoid}}$ is the density of mid-level vertices (branching points) in the spacetime tessellation.

**Physical interpretation by regime:**
- **Prism-dominated regime** ($\langle \chi_{\mathrm{scutoid}} \rangle \approx 0$): Minimal information processing, exploitation phase
- **Simple scutoid regime** ($\langle \chi_{\mathrm{scutoid}} \rangle \approx 2$): Moderate exploration (one neighbor lost, one gained per cloning event)
- **Complex scutoid regime** ($\langle \chi_{\mathrm{scutoid}} \rangle \gg 2$): Rapid exploration, phase transitions

This suggests that **computation requires geometry**: to process information about the landscape, the swarm must break the prismatic symmetry of spacetime.
:::

:::{div} feynman-prose
This conjecture connects algorithmic information processing to geometric structure. It says that you cannot compute without creating topological defects—the scutoid vertices are the "footprints" of computation in spacetime.

Think of it this way: a prism represents a walker that learned nothing new. It sat in place, diffused a bit, but its relationships with neighbors stayed the same. No information gained.

A scutoid represents a walker that made a discovery. It found a better region, cloned there, established new relationships, severed old ones. The mid-level vertices are the record of this discovery—the geometric signature of learning.

The topological information rate measures how fast these discoveries are happening. High rate means active exploration. Low rate means convergence. This gives us a geometric observable for the computational state of the swarm.
:::

(sec-scutoid-conclusions)=
## Conclusions

:::{div} feynman-prose
Let me summarize what we have accomplished.

We started with a fundamental tension: the Latent Fractal Gas has both continuous geometry (the emergent Riemannian metric from {doc}`01_emergent_geometry`) and discrete dynamics (cloning events). How do these fit together?

The answer is the scutoid tessellation. The spacetime of the Latent Fractal Gas is naturally tessellated into $(d+1)$-dimensional polytopes. When walkers persist, the cells are prisms (simple columns). When walkers clone, the cells are scutoids (columns with branching vertices in the middle).

This is not a modeling choice. Theorem {prf:ref}`thm-cloning-implies-scutoid` proves that cloning events **necessarily** produce scutoid geometry. The neighbor sets change, so the top and bottom faces are combinatorially incompatible. The only way to connect them is with mid-level vertices. Scutoids are forced on us by topology.

The practical payoff is an efficient algorithm (Algorithm {prf:ref}`alg-online-triangulation-update`) that maintains the tessellation in $O(N)$ amortized time per timestep—optimal by Theorem {prf:ref}`thm-omega-n-lower-bound`. This enables real-time geometric analysis of large swarms.

The theoretical payoff is the topological information rate (Conjecture {prf:ref}`conj-topological-information-bound`), which connects computational activity to geometric structure. The scutoid vertices are the atoms of computation; their density measures how fast the swarm is learning.

In the next chapter, we will see how the continuous limit of these discrete scutoid deformations recovers classical differential geometry: curvature from holonomy, the Raychaudhuri equation from scutoid evolution, and eventually the field equations of emergent gravity.
:::

:::{admonition} Technical Note (Precise Theorem Scope)
:class: note

The rigorous statement is conditional: **neighbor changes force scutoids**; cloning makes neighbor changes likely, but not guaranteed in every realization.
:::

:::{admonition} Key Takeaways
:class: feynman-added tip

**Scutoid Framework:**

| Concept | Mathematical Object | Physical Meaning |
|---------|---------------------|------------------|
| Voronoi cell | $\mathrm{Vor}_i(t)$ | Region of influence of walker $i$ |
| Neighbor set | $\mathcal{N}_i(t)$ | Connectivity structure |
| Scutoid | $(d+1)$-dim polytope $\mathcal{S}_i$ | Spacetime history of a walker |
| Mid-level vertex | Branching point | Topological defect from cloning |
| Scutoid index | $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ | Total neighbor changes |

**Cell Type Classification:**

| Type | Scutoid Index | Algorithm Event | Physical Phase |
|------|---------------|-----------------|----------------|
| 0 (Prism) | $\chi = 0$ | No neighbor change on interval | Exploitation |
| 1 (Simple) | $\chi = 2$ (symmetric: 1 lost, 1 gained) | Single neighbor swap | Exploration |
| $\geq 2$ (Complex) | $\chi \geq 2$ (asymmetric or multi-change) | Major reorganization | Phase transition |

**Computational Complexity:**

| Operation | Batch | Online | Speedup |
|-----------|-------|--------|---------|
| Per timestep | $O(|\mathrm{DT}| \log N)$ | $O(N)$ amortized (linear-size $\mathrm{DT}$) | $\log N$ |
| Lower bound | $\Omega(|\mathrm{DT}|)$ | $\Omega(|\mathrm{DT}|)$ | Output-size optimal |

**Key Theorems:**

1. **Neighbor Change $\Rightarrow$ Scutoid** (Theorem {prf:ref}`thm-cloning-implies-scutoid`): If cloning changes the neighbor set, the cell must be a scutoid
2. **Optimality** (Theorem {prf:ref}`thm-omega-n-lower-bound`): The update cost is optimal up to output size ($O(N)$ when $|\mathrm{DT}|=\Theta(N)$)
3. **Topological Information** (Conjecture {prf:ref}`conj-topological-information-bound`): Computation leaves geometric footprints as scutoid vertices
:::

---

(sec-scutoid-references)=
## References

### Computational Geometry

- **Voronoi tessellation**: Standard computational geometry construction; see {cite}`berg2008computational` for algorithms
- **Delaunay triangulation**: Dual structure to Voronoi; Lawson flips for incremental updates {cite}`lawson1977software`
- **Jump-and-walk point location**: Expected $O(N^{1/d})$ for uniform random queries; $O(\mathrm{dist}/\ell)$ with a good hint; $O(\log N)$ with a spatial index {cite}`mucke1999fast`
- **Kinetic data structures**: Amortized analysis of geometric data structures under motion {cite}`guibas1992randomized`
- **Lower bounds**: Information-theoretic lower bounds on geometric algorithms {cite}`preparata1985computational`

### Stochastic Geometry

- **Voronoi stability**: Sensitivity of Voronoi tessellations to point perturbations; see {cite}`chiu2013stochastic` for comprehensive treatment
- **Poisson-Voronoi cells**: Statistical properties of Voronoi cells under Poisson point processes

### Scutoid Geometry

- **Scutoid discovery**: Gómez-Gálvez et al. (2018) discovered scutoids in epithelial tissue packing {cite}`gomez2018scutoids`
- **Topological transitions**: Neighbor-swapping in curved tissue requires non-prismatic cells

### Framework Documents

- {doc}`01_emergent_geometry` — Emergent Riemannian geometry from adaptive diffusion
- {doc}`/3_fractal_gas/2_fractal_set/02_causal_set_theory` — Causal set structure of the Fractal Set
- {doc}`/3_fractal_gas/appendices/09_propagation_chaos` — Mean-field justification for spatial statistics
- {doc}`/3_fractal_gas/appendices/14_b_geometric_gas_cinf_regularity_full` — Regularity bounds for curvature control
- {prf:ref}`def-adaptive-diffusion-tensor-latent` — Adaptive diffusion tensor and emergent metric
- {prf:ref}`def-fractal-set-cst-edges` — CST edge definition
- {prf:ref}`def-fractal-set-cloning-score` — Cloning score definition
- {prf:ref}`def-fractal-set-sde` — SDE for walker evolution

```{bibliography}
:filter: docname in docnames
```
