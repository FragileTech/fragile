(sec-scutoid-spacetime)=
# Scutoid Spacetime: Discrete Geometry from Cloning

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`/3_fractal_gas/2_fractal_set/02_causal_set_theory`, {cite}`gomez2018scutoids`

---

(sec-scutoid-tldr)=
## TLDR

*Notation: $\mathrm{Vor}_i(t)$ = Voronoi cell of walker $i$ at time $t$; $\mathcal{N}_i(t)$ = neighbor set; $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ = scutoid index (total neighbor changes); $d_g$ = Riemannian geodesic distance; $d$ = latent space dimension.*

**Scutoids Bridge Continuous Geometry and Discrete Dynamics**: The Latent Fractal Gas has continuous Riemannian geometry ({doc}`01_emergent_geometry`) but discrete cloning dynamics. Scutoids—$(d+1)$-dimensional polytopes with mid-level vertices—are the natural spacetime cells that accommodate topological transitions when neighbor sets change during cloning.

**Cloning Forces Scutoid Geometry**: Theorem {prf:ref}`thm-cloning-implies-scutoid` proves that cloning events *necessarily* produce scutoid cells. When a walker dies and is replaced by a clone at a different position, the Voronoi neighbor sets change (generically), and connecting incompatible top/bottom faces requires mid-level branching vertices.

**Online Triangulation Algorithm**: Algorithm {prf:ref}`alg-online-triangulation-update` maintains the Voronoi/Delaunay tessellation in $O(N)$ amortized time per timestep, optimal by Theorem {prf:ref}`thm-omega-n-lower-bound`. This is a factor of $\log N$ faster than batch recomputation.

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

:::{prf:definition} Delaunay Triangulation
:label: def-delaunay-triangulation

The **Delaunay triangulation** $\mathrm{DT}(t)$ at time $t$ is the simplicial complex dual to the Voronoi tessellation:

- **Vertices**: Walker positions $\{z_i(t)\}_{i=1}^N$
- **Edges**: $(i, j) \in \mathrm{DT}(t)$ iff $j \in \mathcal{N}_i(t)$
- **Simplices**: A $(k+1)$-tuple $(i_0, \ldots, i_k)$ forms a $k$-simplex iff the corresponding Voronoi cells have a common intersection point

**Duality relations:**

| Voronoi Structure | Delaunay Structure |
|-------------------|-------------------|
| $d$-dimensional cell $\mathrm{Vor}_i$ | Vertex $z_i$ |
| $(d-1)$-dimensional face $\Gamma_{i,j}$ | Edge $(i, j)$ |
| Vertex (intersection of $d+1$ cells) | $d$-simplex |
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

For each $k \in \mathcal{N}_{\mathrm{shared}}$, the **boundary correspondence map** $\phi_k: \Gamma_{j,k}(t) \to \Gamma_{i,k}(t + \Delta t)$ is defined by **arc-length rescaling**:

1. Parameterize $\Gamma_{j,k}(t)$ by arc length: $\gamma_{\mathrm{bottom}}: [0, L_{\mathrm{bottom}}] \to \Gamma_{j,k}(t)$
2. Parameterize $\Gamma_{i,k}(t + \Delta t)$ by arc length: $\gamma_{\mathrm{top}}: [0, L_{\mathrm{top}}] \to \Gamma_{i,k}(t + \Delta t)$
3. Define the correspondence map:

$$
\phi_k(\gamma_{\mathrm{bottom}}(s)) = \gamma_{\mathrm{top}}\left(s \cdot \frac{L_{\mathrm{top}}}{L_{\mathrm{bottom}}}\right)
$$

This maps corresponding fractions of the boundary: the point at 30% along the bottom segment maps to the point at 30% along the top segment.

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

:::{prf:theorem} Cloning Implies Scutoid Geometry
:label: thm-cloning-implies-scutoid

Let $e_i$ be an episode (walker trajectory) traversing the interval $[t, t + \Delta t]$.

**Statement:**

1. **No Cloning (Persistence)**: If episode $e_i$ persists without cloning, then with probability 1, $\mathcal{N}_i(t) = \mathcal{N}_i(t + \Delta t)$. The spacetime cell is a **Prism** (Type 0).

2. **Cloning Event**: If episode $e_i$ ends at time $t$ and is replaced by a clone $e_{i'}$ from parent $e_j$ at a different position, then generically $\mathcal{N}_i(t) \neq \mathcal{N}_{i'}(t + \Delta t)$ (probability approaching 1 as $\|z_i - z_j\|/\ell \to \infty$). The spacetime cell is a **Scutoid** (Type $\geq 1$).

*Proof.*

**Part 1: Persistence implies prismatic geometry.**

When no cloning occurs, the walker position evolves continuously via the Langevin SDE ({prf:ref}`def-fractal-set-sde`):

$$
dz_i = v_i \, dt + \Sigma_{\mathrm{reg}}(z_i, S) \circ dW_i
$$
where $v_i$ is the drift velocity, $\Sigma_{\mathrm{reg}}$ is the adaptive diffusion tensor, and $dW_i$ is Brownian noise. The key point is that the trajectory is continuous (no jumps).

For infinitesimal $\Delta t$, the Voronoi boundaries deform continuously. The neighbor graph of a Voronoi tessellation is constant under continuous deformation of seeds, except at **critical configurations** where the Delaunay triangulation is degenerate. In Euclidean space, this occurs when $d+2$ or more seeds lie on a common sphere; in Riemannian geometry, degeneracy occurs when seeds lie on a common geodesic sphere (equidistant from some center point). These critical configurations form a set of measure zero in the space of walker configurations.

Therefore, $\mathcal{N}_i(t) = \mathcal{N}_i(t + \Delta t)$ almost surely. The boundary correspondence map $\phi_k$ is defined for all neighbors, and the spacetime cell is a prism with no mid-level vertices.

**Part 2: Cloning induces neighbor change.**

A cloning event replaces walker $i$ at position $z_i$ with a clone at position $z_{i'} = z_j + \xi$, where $z_j$ is the parent's position and $\xi \sim \mathcal{N}(0, \sigma^2 I)$ is Gaussian jitter.

Consider a neighbor $k \in \mathcal{N}_i(t)$. For $k$ to remain a neighbor of the clone at $z_{i'}$, there must exist points equidistant from both $z_k$ and $z_{i'}$. The locus of such points (the Voronoi boundary) depends on the positions $z_k$ and $z_{i'}$.

Since $z_i$ and $z_j$ are generically at different locations (the parent is a high-fitness walker, the dying walker is low-fitness), the Voronoi boundaries around $z_i$ and $z_j$ are different hypersurfaces. The probability that a neighbor of $z_i$ is also a neighbor of $z_j$ decreases as $\|z_i - z_j\| / \ell_{\mathrm{local}} \to \infty$, where $\ell_{\mathrm{local}}$ is the typical inter-walker spacing.

**Quantitative bound**: For walker density $\rho \sim N / \mathrm{Vol}(\mathcal{Z})$ and typical spacing $\ell \sim \rho^{-1/d}$:

$$
\mathbb{P}(\mathcal{N}_i(t) = \mathcal{N}_{i'}(t + \Delta t)) \leq \exp\left(-c \cdot \frac{\|z_i - z_j\|^d}{\ell^d}\right)
$$
for some constant $c > 0$.

*Justification*: The expected number of walkers in the region between $z_i$ and $z_j$ scales as $\rho \cdot \|z_i - z_j\|^d = (\|z_i - z_j\|/\ell)^d$. For $z_i$ and $z_{i'}$ to share the same neighbor set, no intervening walkers can disrupt the Voronoi structure. By a Poisson-type argument, this probability decays exponentially with the expected number of intervening walkers.

**Part 3: Neighbor change forces scutoid geometry.**

Suppose $\mathcal{N}_j(t) \neq \mathcal{N}_{i'}(t + \Delta t)$. Let:
- $\mathcal{N}_{\mathrm{lost}} = \mathcal{N}_j(t) \setminus \mathcal{N}_{i'}(t + \Delta t)$ (neighbors at bottom only)
- $\mathcal{N}_{\mathrm{gained}} = \mathcal{N}_{i'}(t + \Delta t) \setminus \mathcal{N}_j(t)$ (neighbors at top only)

For $\ell \in \mathcal{N}_{\mathrm{lost}}$, the boundary segment $\Gamma_{j,\ell}(t)$ has no corresponding segment on the top face. The geodesic ruling from this segment cannot continue to the top—it must terminate somewhere in between.

For $m \in \mathcal{N}_{\mathrm{gained}}$, the boundary segment $\Gamma_{i',m}(t + \Delta t)$ has no corresponding segment on the bottom face. The geodesic ruling to this segment must originate somewhere in between.

The intermediate locations where geodesics terminate and originate are the **mid-level vertices**. Their existence is topologically necessary to form a closed boundary.

By Definition {prf:ref}`def-scutoid-cell`, any spacetime cell with mid-level vertices is a scutoid, not a prism.

$\square$
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

**Type 0: Prism** (No cloning)
- $\mathcal{N}_{\mathrm{lost}} = \mathcal{N}_{\mathrm{gained}} = \emptyset$
- No mid-level vertices
- Represents: Persistent walker undergoing continuous diffusion
- Physics: Laminar flow, exploitation phase

**Type 1: Simple Scutoid** (Single neighbor swap)
- $|\mathcal{N}_{\mathrm{lost}}| = |\mathcal{N}_{\mathrm{gained}}| = 1$ (scutoid index $\chi = 2$)
- One mid-level vertex (or edge in higher dimensions)
- Represents: Standard cloning event with local reorganization
- Physics: Plastic deformation, adaptive exploration

**Type $\geq 2$: Complex Scutoid** (Multiple neighbor changes or asymmetric changes)
- $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}| \geq 2$, excluding the symmetric single-swap case (Type 1)
- Number of mid-level vertices: $\max(|\mathcal{N}_{\mathrm{lost}}|, |\mathcal{N}_{\mathrm{gained}}|)$
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
- Prismatic cells (Type 0): $N_{\mathrm{prism}} = N(1 - p_{\mathrm{clone}}) \cdot (T/\Delta t)$
- Scutoid cells (Type $\geq 1$): $N_{\mathrm{scutoid}} = N \cdot p_{\mathrm{clone}} \cdot (T/\Delta t)$

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

The naive approach is to rebuild everything from scratch at each timestep: take the $N$ walker positions, run a Voronoi/Delaunay algorithm, output the result. This costs $O(N \log N)$ per timestep using Fortune's algorithm or Bowyer-Watson.

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
        # Type 1: Local move (prismatic scutoid)
        MovedWalkers.append((walker_id, edge.z_old, edge.z_new))

    elif edge.type == "cloning":
        # Type 2: Teleport (non-prismatic scutoid)
        ClonedWalkers.append((walker_id, edge.z_new, edge.parent_id))
```

**Step 2: Update Locally Moved Walkers** — Amortized $O(1)$ per walker

```{code-block} python
:caption: Update Delaunay structure for moved walkers

for (walker_id, z_old, z_new) in MovedWalkers:
    vertex = VertexMap[walker_id]
    vertex.position = z_new

    # Restore Delaunay property via Lawson flips
    LawsonFlip(DT, vertex)

    # Update corresponding Voronoi cell
    UpdateVoronoiCell(VT, vertex)
```

**Step 3: Update Cloned Walkers** — $O(\log N)$ per walker

```{code-block} python
:caption: Handle cloning events with point location

for (dead_id, z_new, parent_id) in ClonedWalkers:
    # Phase A: Delete dead walker
    dead_vertex = VertexMap[dead_id]
    DT.remove_vertex(dead_vertex)

    # Phase B: Insert new walker
    containing_simplex = DT.locate(z_new)  # O(log N) via jump-and-walk
    new_vertex = DT.insert_vertex(z_new, containing_simplex)
    LawsonFlip(DT, new_vertex)

    # Update mapping
    VertexMap[dead_id] = new_vertex
```

**Total Complexity per Timestep:**

$$
T(N) = O(N) + O(p_{\mathrm{clone}} \cdot N \cdot \log N)
$$

For typical cloning probabilities $p_{\mathrm{clone}} \ll 1/\log N$, the second term is dominated by the first, yielding **$O(N)$ amortized** complexity per timestep.
:::

(sec-lawson-flip)=
### Lawson Flip Algorithm

:::{div} feynman-prose
The Lawson flip is the workhorse of incremental Delaunay maintenance. When you move a vertex, some of the incident simplices may violate the Delaunay criterion (empty circumsphere property). The Lawson algorithm fixes this by iteratively "flipping" edges until the criterion is restored.

Here is the beautiful thing: for small vertex movements, the number of flips is $O(1)$ on average. The disruption is local. Move a vertex by a small distance, and only a handful of nearby simplices need adjustment. This is what makes the online algorithm fast.

The key insight is geometric: the Delaunay triangulation is a stable structure. Small perturbations cause small changes. Only large movements (like cloning teleportation) cause global reorganization.
:::

:::{prf:algorithm} Lawson Flip for Delaunay Restoration
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

**Complexity Analysis:**

- For vertex displacement $\delta$, affected simplices lie within distance $O(\delta)$
- Number of affected simplices: $O(1)$ for small $\delta$
- Each flip may cascade to $O(1)$ neighbors
- **Total flips: $O(1)$ amortized** for small displacements

**Key Property:** Lawson flips preserve the Delaunay structure incrementally. The algorithm terminates because each flip reduces a potential function (total circumradius).
:::

(sec-jump-and-walk)=
### Jump-and-Walk Point Location

:::{div} feynman-prose
For cloning events, we need to insert a new vertex at position $z_{\mathrm{new}}$. The first step is finding which simplex contains this position. This is the **point location** problem.

The naive approach—check every simplex—takes $O(N)$ time. Much too slow.

The jump-and-walk algorithm does better. Start from some "hint" simplex (say, one incident to the parent walker), then walk through the triangulation toward the target. At each step, check which face of the current simplex is "toward" the target, and cross to the adjacent simplex. Eventually you reach the simplex containing the target.

The expected number of steps is $O(\log N)$ for typical point sets. The walk length depends on how far the target is from the starting hint, but for cloning events (where the new position is near the parent), the hint is excellent and the walk is short.
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
            return None  # z is outside convex hull
```

**Complexity:**

- Jump phase: $O(1)$ using spatial hashing or parent simplex
- Walk phase: $O(\sqrt[d]{N})$ expected for random points, $O(\log N)$ with good hint
- **Total: $O(\log N)$ expected** when hint simplex is near target
:::

(sec-complexity-analysis)=
## Complexity Analysis

:::{prf:theorem} Amortized Complexity of Online Triangulation
:label: thm-amortized-complexity

The online scutoid-guided triangulation update (Algorithm {prf:ref}`alg-online-triangulation-update`) achieves:

**Per-Timestep Complexity:**

$$
T(N) = O(N) + O(p_{\mathrm{clone}} \cdot N \cdot \log N)
$$

**Amortized Complexity** (over $T$ timesteps):

$$
\bar{T}(N) = O(N) \quad \text{if } p_{\mathrm{clone}} \ll \frac{1}{\log N}
$$

**Typical Regime:** For $p_{\mathrm{clone}} \in [0.01, 0.1]$ and $N \in [10^3, 10^6]$:
- $p_{\mathrm{clone}} \cdot \log N \approx 0.01 \times 20 = 0.2 \ll 1$
- **Effective complexity: $O(N)$ per timestep**

**Speedup vs. Batch:**

$$
\text{Speedup} = \frac{O(N \log N)}{O(N)} = O(\log N)
$$

For $N = 10^6$: approximately $20\times$ speedup.

*Proof.*

**SDE moves (Type 0 prisms):**
- Number of persistent walkers: $N(1 - p_{\mathrm{clone}})$
- Cost per walker: $O(1)$ amortized (Lawson flips for small displacement)
- Total: $O(N)$

**Cloning events (Type $\geq 1$ scutoids):**
- Number of cloned walkers: $N \cdot p_{\mathrm{clone}}$
- Cost per walker: $O(\log N)$ (point location) + $O(1)$ (Lawson flips)
- Total: $O(p_{\mathrm{clone}} \cdot N \cdot \log N)$

Combining: $T(N) = O(N) + O(p_{\mathrm{clone}} \cdot N \cdot \log N)$.

For $p_{\mathrm{clone}} \ll 1/\log N$, the second term is dominated by the first.

$\square$
:::

(sec-lower-bound-proof)=
### Lower Bound Proof

:::{prf:theorem} $\Omega(N)$ Lower Bound for Tessellation Update
:label: thm-omega-n-lower-bound

Any algorithm that correctly updates a Voronoi/Delaunay tessellation of $N$ points after arbitrary point movements must take, in the worst case, at least $\Omega(N)$ time.

**Conclusion:** The $O(N)$ amortized complexity of Algorithm {prf:ref}`alg-online-triangulation-update` is **asymptotically optimal**.

*Proof.*

**Information-theoretic argument:**

The output is a complete geometric data structure representing:
- $N$ vertex positions (walker coordinates)
- $\Theta(N)$ simplices (in fixed dimension $d$)
- Adjacency information for each simplex

The **output size is $\Theta(N)$**.

Any algorithm producing $\Theta(N)$ output requires at least $\Omega(N)$ time—it is impossible to write $N$ pieces of information in fewer than $N$ operations.

**Worst-case construction:**

Consider a global rotation: all $N$ walkers rotate by angle $\theta$ around a center. The combinatorial structure may be unchanged, but all vertex coordinates must be updated. The algorithm must touch all $\Theta(N)$ geometric objects.

**Conclusion:** Lower bound $\Omega(N)$, upper bound $O(N)$ amortized. The algorithm is optimal.

$\square$
:::

:::{div} feynman-prose
This is a satisfying result. We cannot do better than $O(N)$—the output is that big, and we have to at least look at it. Our algorithm achieves $O(N)$ amortized. We are as good as theoretically possible.

The improvement over batch computation ($O(N \log N)$) comes from exploiting temporal coherence. Most of the structure is preserved between timesteps; we only update what changed. This is the power of incremental algorithms.
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

**Alternative formulation:**

$$
\dot{I}_{\mathrm{topo}} = \langle \chi_{\mathrm{scutoid}} \rangle \cdot f_{\mathrm{clone}}
$$

where:
- $\chi_{\mathrm{scutoid}} = |\mathcal{N}_{\mathrm{lost}}| + |\mathcal{N}_{\mathrm{gained}}|$ is the scutoid index (total neighbor changes)
- $\langle \chi_{\mathrm{scutoid}} \rangle$ is the average scutoid index per cloning event
- $f_{\mathrm{clone}} = N \cdot p_{\mathrm{clone}} / \Delta t$ is the cloning frequency (events per unit time)
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
| 0 (Prism) | $\chi = 0$ | Persistence | Exploitation |
| 1 (Simple) | $\chi = 2$ (symmetric: 1 lost, 1 gained) | Single neighbor swap | Exploration |
| $\geq 2$ (Complex) | $\chi \geq 2$ (asymmetric or multi-change) | Major reorganization | Phase transition |

**Computational Complexity:**

| Operation | Batch | Online | Speedup |
|-----------|-------|--------|---------|
| Per timestep | $O(N \log N)$ | $O(N)$ amortized | $\log N$ |
| Lower bound | $\Omega(N)$ | $\Omega(N)$ | Optimal |

**Key Theorems:**

1. **Cloning $\Rightarrow$ Scutoid** (Theorem {prf:ref}`thm-cloning-implies-scutoid`): Cloning events necessarily produce scutoid geometry
2. **Optimality** (Theorem {prf:ref}`thm-omega-n-lower-bound`): The $O(N)$ online algorithm is asymptotically optimal
3. **Topological Information** (Conjecture {prf:ref}`conj-topological-information-bound`): Computation leaves geometric footprints as scutoid vertices
:::

---

(sec-scutoid-references)=
## References

### Computational Geometry

- **Voronoi tessellation**: Standard computational geometry construction; see {cite}`berg2008computational` for algorithms
- **Delaunay triangulation**: Dual structure to Voronoi; Lawson flips for incremental updates {cite}`lawson1977software`
- **Jump-and-walk point location**: Expected $O(\log N)$ complexity for typical point sets {cite}`mucke1999fast`

### Scutoid Geometry

- **Scutoid discovery**: Gómez-Gálvez et al. (2018) discovered scutoids in epithelial tissue packing {cite}`gomez2018scutoids`
- **Topological transitions**: Neighbor-swapping in curved tissue requires non-prismatic cells

### Framework Documents

- {doc}`01_emergent_geometry` — Emergent Riemannian geometry from adaptive diffusion
- {doc}`/3_fractal_gas/2_fractal_set/02_causal_set_theory` — Causal set structure of the Fractal Set
- {prf:ref}`def-adaptive-diffusion-tensor-latent` — Adaptive diffusion tensor and emergent metric
- {prf:ref}`def-fractal-set-cst-edges` — CST edge definition
- {prf:ref}`def-fractal-set-cloning-score` — Cloning score definition

```{bibliography}
:filter: docname in docnames
```
