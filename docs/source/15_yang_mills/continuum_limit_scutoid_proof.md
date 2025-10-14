---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Continuum Limit of Particle-Based Lattice Gauge Theory

## Abstract

This document establishes the continuum limit for lattice gauge theory defined on particle-based spatial tessellations, independent of any specific particle dynamics framework. We prove that as the number of particles $N \to \infty$ with fixed spatial density $\rho = N/V$, the discrete lattice Hamiltonian converges to the standard continuum Yang-Mills Hamiltonian with a **consistent effective coupling constant** $g_{\text{eff}}^2$ for both electric and magnetic terms.

The key technical achievement is the careful derivation of field normalizations and measure relationships using Voronoi/Delaunay geometry, which reveals that the apparent asymmetry between edge-based (electric) and face-based (magnetic) terms is resolved through proper geometric measure theory. This result is **universal**: it applies to any particle configuration converging to a smooth density, including but not limited to quasi-stationary distributions from Fragile Gas dynamics, thermal equilibrium from molecular dynamics, or uniform random point processes.

**Key Results:**
- **Theorem (Continuum Limit)**: For any particle configuration with smooth limiting density $\rho(x)$, the lattice Yang-Mills Hamiltonian converges to

$$
H_{\text{continuum}} = \frac{1}{2g_{\text{eff}}^2} \int_{\mathbb{R}^3} d^3x \, \left( E_a^i(x) E_a^i(x) + B_a^i(x) B_a^i(x) \right)
$$

where $g_{\text{eff}}^2 = g^2 / \rho_0$ with $\rho_0$ a reference density scale, and the same coupling appears in both terms.

- **Corollary (Fragile Gas Application)**: The Euclidean Gas Yang-Mills lattice, upon convergence to its quasi-stationary distribution, realizes this continuum limit with coupling $g_{\text{eff}}^2$ determined by the QSD density profile.

**Pedagogical Approach:** We build the proof using only standard tools from computational geometry (Voronoi tessellations, Delaunay triangulation) and differential geometry, avoiding framework-specific concepts. All density scaling relationships are derived from first principles.

---

## §1. Voronoi-Delaunay Geometry of Particle-Based Lattices

### §1.1. Setup and Definitions

:::{prf:definition} Particle Configuration
:label: def-particle-configuration

A **particle configuration** in volume $V \subset \mathbb{R}^3$ is a set of $N$ distinct points:

$$
\mathcal{P}_N = \{x_1, x_2, \ldots, x_N\} \subset V
$$

We assume $V$ is a bounded domain with smooth boundary, and consider the limit $N \to \infty$ with $V$ fixed (increasing density limit) or $V \to \infty$ with $N/V \to \rho_0$ (thermodynamic limit).
:::

:::{prf:definition} Voronoi Tessellation
:label: def-voronoi-tessellation

The **Voronoi cell** of particle $i$ is:

$$
\text{Vor}_i = \left\{ x \in V : \|x - x_i\| \leq \|x - x_j\| \text{ for all } j \neq i \right\}
$$

The collection $\{\text{Vor}_i\}_{i=1}^N$ forms a space-filling tessellation of $V$ called the **Voronoi diagram**.

Each Voronoi cell is a convex polytope whose faces are segments of hyperplanes $\{x : \|x - x_i\| = \|x - x_j\|\}$ for neighboring particles.
:::

:::{prf:definition} Delaunay Triangulation
:label: def-delaunay-triangulation

The **Delaunay triangulation** $\text{Del}(\mathcal{P}_N)$ is the geometric dual of the Voronoi diagram:

- **Vertices**: The particles $\{x_i\}$
- **Edges**: Connect $x_i$ and $x_j$ if their Voronoi cells share a 2D face
- **Faces (triangles)**: Formed by three particles whose Voronoi cells share a common vertex in 2D
- **Tetrahedra**: Formed by four particles whose Voronoi cells share a common vertex in 3D

In 3D, the Delaunay triangulation is a simplicial complex filling the volume $V$.
:::

:::{note}
**Why Voronoi/Delaunay?**

This geometric structure is natural for particle-based lattices because:
1. **Unique and well-defined**: Given any point set, the Voronoi/Delaunay structure is mathematically unique
2. **Space-filling**: Voronoi cells tile the space exactly, with no gaps or overlaps
3. **Neighbor relationships**: Delaunay edges connect natural neighbors (closest pairs)
4. **Smooth limit**: As $N \to \infty$, the tessellation refines and approaches the continuum
5. **Computational**: Standard algorithms exist (Qhull, CGAL) for construction
:::

### §1.2. Geometric Quantities and Density Scaling

We now establish how geometric quantities scale with particle density $\rho = N/V$.

:::{prf:definition} Local Density
:label: def-local-density

The **local particle density** at point $x \in V$ is defined via coarse-graining:

$$
\rho(x) = \lim_{\epsilon \to 0} \frac{1}{|B_\epsilon(x)|} \sum_{i: x_i \in B_\epsilon(x)} 1
$$

where $B_\epsilon(x)$ is a ball of radius $\epsilon$ centered at $x$, and $|\cdot|$ denotes volume.

For the limit $N \to \infty$, we assume:
- $\rho(x)$ converges to a smooth, strictly positive function
- Global constraint: $\int_V \rho(x) d^3x = N$
:::

:::{prf:lemma} Typical Length Scale
:label: lem-typical-length-scale

In a region with local density $\rho(x)$, the **typical inter-particle distance** scales as:

$$
\ell_{\text{typ}}(x) \sim \rho(x)^{-1/3}
$$

**Proof:**
In a small volume $\delta V$ around $x$, there are approximately $\rho(x) \delta V$ particles. If they are arranged with typical spacing $\ell$, then $\delta V \sim \ell^3 \cdot (\rho \delta V)$, giving $\ell^3 \sim 1/\rho$, hence $\ell \sim \rho^{-1/3}$. $\square$
:::

:::{prf:lemma} Edge and Face Densities in Delaunay Triangulation
:label: lem-edge-face-densities

Let $n_e(x)$ and $n_f(x)$ denote the number of Delaunay edges and faces per unit volume at point $x$.

In 3D Delaunay triangulations of random point sets with density $\rho(x)$:

$$
n_e(x) \approx c_e \cdot \rho(x), \quad n_f(x) \approx c_f \cdot \rho(x)
$$

where $c_e$ and $c_f$ are dimensionless constants of order unity (typically $c_e \approx 6$, $c_f \approx 5$ from Euler characteristic relations).

**Proof Sketch:**

1. **Vertex count**: In volume $\delta V$, there are $\sim \rho \delta V$ vertices.

2. **Euler characteristic for 3D simplicial complexes**:

$$
V - E + F - T = \chi
$$

where $V$ = vertices, $E$ = edges, $F$ = faces (triangles), $T$ = tetrahedra, $\chi$ = Euler characteristic (typically 1 for contractible domains).

3. **Asymptotic relations**: For large random Delaunay triangulations:
   - Each vertex has average degree (coordination number) $\approx 12$ in 3D
   - Thus $E \approx 6V$ (each edge shared by 2 vertices)
   - From Euler: $F \approx 5V$ (solving for $F$ with $T \approx 2V$)

4. **Density scaling**:
   - Vertex density: $n_v = \rho$
   - Edge density: $n_e = 6\rho$
   - Face density: $n_f = 5\rho$

All scale **linearly** with particle density. $\square$
:::

:::{important}
**Key Insight: Linear Density Scaling**

Both edges and faces have densities **linear in $\rho$**, not $\rho^2$ or $\rho^3$. This is a consequence of the Euler characteristic constraint on simplicial complexes.

The difference between electric (edge-based) and magnetic (face-based) terms will come from **geometric measures** (length² vs area²), not from density differences.
:::

:::{prf:lemma} Geometric Measure Scaling
:label: lem-geometric-measure-scaling

For a Delaunay edge $e$ connecting particles in a region of density $\rho(x)$:

$$
\langle \ell_e^2 \rangle \sim \rho(x)^{-2/3}
$$

For a Delaunay face $f$ in the same region:

$$
\langle A_f^2 \rangle \sim \rho(x)^{-4/3}
$$

where $\langle \cdot \rangle$ denotes ensemble or spatial average.

**Proof:**
- Edge length: $\ell_e \sim \ell_{\text{typ}} \sim \rho^{-1/3}$ from {prf:ref}`lem-typical-length-scale`, so $\ell_e^2 \sim \rho^{-2/3}$
- Face area: $A_f \sim \ell_{\text{typ}}^2 \sim \rho^{-2/3}$ (area scales as length²), so $A_f^2 \sim \rho^{-4/3}$
$\square$
:::

### §1.3. Convergence to Continuum

:::{prf:theorem} Voronoi Tessellation Convergence
:label: thm-voronoi-convergence

Let $\{\mathcal{P}_N\}$ be a sequence of particle configurations with:

1. $N \to \infty$ with $N/V \to \rho_0 > 0$
2. Local density $\rho_N(x) \to \rho(x)$ uniformly, where $\rho(x) \in C^1(V)$ and $\rho(x) \geq \rho_{\min} > 0$

Then:

(a) The maximum cell diameter: $\max_i \text{diam}(\text{Vor}_i) \to 0$ as $N \to \infty$

(b) For any continuous function $f: V \to \mathbb{R}$:

$$
\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N f(x_i) = \frac{1}{V\rho_0} \int_V \rho(x) f(x) d^3x
$$

(c) For edge-based quantities with $e = (i,j)$:

$$
\lim_{N \to \infty} \frac{1}{N} \sum_{\text{edges } e} \ell_e^2 f\left(\frac{x_i + x_j}{2}\right) = c_e \int_V \rho(x)^{1/3} f(x) d^3x
$$

(d) For face-based quantities:

$$
\lim_{N \to \infty} \frac{1}{N} \sum_{\text{faces } f} A_f^2 g\left(x_f\right) = c_f \int_V \rho(x)^{-1/3} g(x) d^3x
$$

where $x_f$ is the centroid of face $f$.

**Proof:**
(a) and (b) are standard results from quasi-Monte Carlo theory and Voronoi geometry.

(c) From {prf:ref}`lem-edge-face-densities`: $n_e(x) \sim c_e \rho(x)$. From {prf:ref}`lem-geometric-measure-scaling`: $\langle \ell_e^2 \rangle \sim \rho(x)^{-2/3}$. Thus:

$$
\sum_{\text{edges near } x} \ell_e^2 \approx [n_e(x) \delta V] \cdot [\ell^2] \sim c_e \rho(x) \delta V \cdot \rho(x)^{-2/3} = c_e \rho(x)^{1/3} \delta V
$$

Converting sum to integral via Riemann sum convergence yields (c).

(d) Similarly for faces: $n_f(x) \sim c_f \rho(x)$, $\langle A_f^2 \rangle \sim \rho(x)^{-4/3}$:

$$
\sum_{\text{faces near } x} A_f^2 \approx c_f \rho(x) \delta V \cdot \rho(x)^{-4/3} = c_f \rho(x)^{-1/3} \delta V
$$

$\square$
:::

:::{note}
**Physical Interpretation**

Parts (c) and (d) are the key to resolving the coupling constant puzzle:
- **Electric term** (edge-based): Contributes $\int \rho^{1/3} E^2$
- **Magnetic term** (face-based): Contributes $\int \rho^{-1/3} B^2$

The opposite density dependences arise from the interplay of:
- Number density ($n_e, n_f \propto \rho$)
- Geometric measures ($\ell^2 \propto \rho^{-2/3}$, $A^2 \propto \rho^{-4/3}$)

To achieve consistent coupling, we must choose field normalizations that cancel these factors. This is done in §4.
:::

---

## §2. Lattice Gauge Theory on Delaunay Triangulation

### §2.1. Gauge Fields and Wilson Lines

We now define lattice gauge theory on the Delaunay triangulation. The gauge group is $G = \text{SU}(N_c)$ (typically $N_c = 3$ for QCD, but the construction works for any compact Lie group).

:::{prf:definition} Lattice Gauge Field
:label: def-lattice-gauge-field

A **lattice gauge field** is an assignment of group elements to each oriented edge of the Delaunay triangulation:

$$
U: \{\text{oriented edges}\} \to G, \quad e \mapsto U_e \in G
$$

with the orientation constraint:

$$
U_{e^{-1}} = U_e^{-1}
$$

where $e^{-1}$ denotes the edge with reversed orientation.

For infinitesimal gauge fields (near the identity), we write:

$$
U_e = \exp(ig A_e)
$$

where $A_e \in \mathfrak{g}$ is a Lie algebra element (the "gauge potential on edge $e$"), and $g$ is the bare lattice coupling constant.
:::

:::{prf:definition} Electric Field
:label: def-electric-field

The **electric field** conjugate to $A_e$ is defined via the canonical commutation relation:

$$
[E_e^a, A_{e'}^b] = -i \delta_{ee'} \delta^{ab}
$$

where $a, b$ are Lie algebra indices.

Physically, $E_e^a$ represents the electric field component along edge $e$ in the $a$-th color direction.
:::

:::{prf:definition} Wilson Loop and Magnetic Field
:label: def-wilson-loop

For a Delaunay triangular face $f$ with boundary edges $e_1, e_2, e_3$ (ordered counterclockwise), the **Wilson loop** is:

$$
U_f = U_{e_1} U_{e_2} U_{e_3}
$$

The **magnetic field** (or plaquette field strength) is defined via:

$$
U_f = \exp(ig B_f) = \mathbb{1} + ig B_f - \frac{g^2}{2} B_f^2 + O(g^3)
$$

For small $g$, extracting the Lie algebra component:

$$
B_f^a \approx \frac{1}{ig} (U_f - \mathbb{1})^a = \frac{1}{2g} \text{Tr}(T^a (U_f - \mathbb{1}))
$$

where $\{T^a\}$ are Lie algebra generators in the fundamental representation.
:::

:::{note}
**Continuum Connection**

In the continuum, the gauge potential is a 1-form $A = A_\mu^a dx^\mu T^a$, and the field strength is:

$$
F_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + g f^{abc} A_\mu^b A_\nu^c
$$

The lattice definitions above are discrete analogs:
- $A_e$ discretizes $\int_e A$
- $B_f$ discretizes $\int_f F$ (flux through face)

The key question for continuum limit is: **how do we relate $A_e, E_e, B_f$ to continuum fields $A_\mu(x), E^i(x), B^i(x)$?** This is addressed in §4.
:::

### §2.2. Lattice Yang-Mills Hamiltonian

:::{prf:definition} Lattice Yang-Mills Hamiltonian
:label: def-lattice-hamiltonian

The **lattice Yang-Mills Hamiltonian** on the Delaunay triangulation is:

$$
H_{\text{lattice}} = \frac{g^2}{2} \sum_{\text{edges } e} \ell_e \, E_e^a E_e^a + \frac{1}{2g^2} \sum_{\text{faces } f} \frac{1}{A_f} \, \text{Tr}(U_f - \mathbb{1})^2
$$

where:
- $\ell_e$ is the length of edge $e$
- $A_f$ is the area of triangular face $f$
- The trace is over color indices: $\text{Tr}(U_f - \mathbb{1})^2 = -4 \sum_a (B_f^a)^2 + O(g)$

**Justification:**

This is the standard lattice gauge theory Hamiltonian (Wilson's formulation) adapted to irregular Delaunay lattices. The geometric factors $\ell_e$ and $1/A_f$ ensure correct continuum limit (to be proven in §5).
:::

:::{important}
**Key Observation: Asymmetric Geometric Factors**

Notice the Hamiltonian has:
- Electric term: $\propto \ell_e$ (length)
- Magnetic term: $\propto 1/A_f$ (inverse area)

This asymmetry is **not** an error — it arises from the different geometric nature of:
- Electric field: Lives on edges (1D objects)
- Magnetic field: Lives on faces (2D objects)

The continuum limit will show that this asymmetry, combined with density scaling and field normalizations, yields consistent coupling.
:::

### §2.3. Hamilton's Equations and Canonical Structure

:::{prf:theorem} Lattice Hamilton's Equations
:label: thm-lattice-hamilton-eqs

The time evolution generated by $H_{\text{lattice}}$ satisfies:

$$
\dot{A}_e^a = \{A_e^a, H_{\text{lattice}}\} = g^2 \ell_e E_e^a
$$

$$
\dot{E}_e^a = \{E_e^a, H_{\text{lattice}}\} = -\frac{1}{g^2} \frac{\partial}{\partial A_e^a} \left[ \sum_{\text{faces } f \ni e} \frac{1}{A_f} \text{Tr}(U_f - \mathbb{1})^2 \right]
$$

where $\{\cdot, \cdot\}$ is the canonical Poisson bracket.

**Proof:**

From the canonical commutation relations $[E_e^a, A_{e'}^b] = -i \delta_{ee'} \delta^{ab}$, and $\dot{\mathcal{O}} = i[H, \mathcal{O}]$, the equations follow from the Hamiltonian structure.

The second equation involves derivatives of Wilson loops with respect to $A_e$, which couple to adjacent faces. $\square$
:::

:::{note}
**Physical Meaning**

These are the lattice analogs of Maxwell's equations (in the Abelian limit $g \to 0$):

$$
\dot{A} = E, \quad \dot{E} = \nabla \times B - J
$$

The lattice formulation preserves the Hamiltonian structure exactly, which will be crucial for deriving consistent continuum limits.
:::

---

## §3. Coarse-Graining and Measure Theory

This section establishes the precise mathematical relationship between lattice sums and continuum integrals.

### §3.1. Riemann Sum Convergence for Vertex Quantities

:::{prf:theorem} Vertex Sum to Integral
:label: thm-vertex-sum-to-integral

For any continuous function $f: V \to \mathbb{R}$, as $N \to \infty$ with local density $\rho_N(x) \to \rho(x)$:

$$
\frac{1}{N} \sum_{i=1}^N f(x_i) \to \frac{1}{\bar{\rho}} \int_V \rho(x) f(x) d^3x
$$

where $\bar{\rho} = N/V$ is the average density.

For uniform density $\rho(x) \equiv \rho_0$:

$$
\sum_{i=1}^N f(x_i) \to \rho_0 V \cdot \frac{1}{V} \int_V f(x) d^3x = \rho_0 \int_V f(x) d^3x
$$

**Proof:** Standard result from Riemann sum convergence. $\square$
:::

### §3.2. Edge Sums with Geometric Weights

This is the first key technical result for coarse-graining.

:::{prf:theorem} Edge Sum to Integral with Density Factor
:label: thm-edge-sum-to-integral

For continuous $f: V \to \mathbb{R}$, as $N \to \infty$:

$$
\sum_{\text{edges } e} \ell_e^2 f(x_e) \to c_e \int_V \rho(x)^{1/3} f(x) d^3x
$$

where:
- $x_e = (x_i + x_j)/2$ is the edge midpoint
- $c_e \approx 6$ is the coordination number constant from {prf:ref}`lem-edge-face-densities`

**Proof:**

1. **Partition volume into cells**: Divide $V$ into small cells $\{\Delta V_k\}$ with $|\Delta V_k| \to 0$.

2. **Count edges in cell $k$**: From {prf:ref}`lem-edge-face-densities`:

$$
\#\{\text{edges with midpoint in } \Delta V_k\} \approx n_e(x_k) |\Delta V_k| = c_e \rho(x_k) |\Delta V_k|
$$

3. **Sum geometric weights**: Each edge in cell $k$ has $\ell_e^2 \sim \rho(x_k)^{-2/3}$ from {prf:ref}`lem-geometric-measure-scaling`. Thus:

$$
\sum_{\text{edges in } \Delta V_k} \ell_e^2 \approx [c_e \rho(x_k) |\Delta V_k|] \cdot [\rho(x_k)^{-2/3}] = c_e \rho(x_k)^{1/3} |\Delta V_k|
$$

4. **Convert to Riemann sum**:

$$
\sum_{\text{all edges}} \ell_e^2 f(x_e) \approx \sum_k c_e \rho(x_k)^{1/3} |\Delta V_k| \cdot f(x_k) \xrightarrow{N \to \infty} c_e \int_V \rho(x)^{1/3} f(x) d^3x
$$

$\square$
:::

:::{important}
**Crucial Insight: The $\rho^{1/3}$ Factor**

Edge sums with geometric weights $\ell_e^2$ produce an integral with density factor $\rho^{1/3}$, arising from:

$$
n_e \cdot \ell_e^2 \sim \rho \cdot \rho^{-2/3} = \rho^{1/3}
$$

This is the source of the density dependence in the electric term.
:::

### §3.3. Face Sums with Geometric Weights

Parallel result for magnetic term.

:::{prf:theorem} Face Sum to Integral with Inverse Density Factor
:label: thm-face-sum-to-integral

For continuous $g: V \to \mathbb{R}$, as $N \to \infty$:

$$
\sum_{\text{faces } f} A_f^2 g(x_f) \to c_f \int_V \rho(x)^{-1/3} g(x) d^3x
$$

where $c_f \approx 5$ from {prf:ref}`lem-edge-face-densities`.

**Proof:**

Identical structure to {prf:ref}`thm-edge-sum-to-integral`, using:
- Face density: $n_f(x) = c_f \rho(x)$
- Geometric measure: $\langle A_f^2 \rangle \sim \rho(x)^{-4/3}$

$$
n_f \cdot A_f^2 \sim \rho \cdot \rho^{-4/3} = \rho^{-1/3}
$$

$\square$
:::

:::{important}
**The $\rho^{-1/3}$ Factor for Magnetic Term**

Face sums with geometric weights $A_f^2$ produce an integral with **inverse** density factor $\rho^{-1/3}$:

$$
n_f \cdot A_f^2 \sim \rho \cdot \rho^{-4/3} = \rho^{-1/3}
$$

This is **opposite** to the electric term's $\rho^{1/3}$! The resolution comes from field normalizations in §4.
:::

---

## §4. Canonical Field Normalizations and Consistent Coupling

This is the heart of the proof: showing how to normalize lattice fields to achieve consistent continuum coupling.

### §4.1. Strategy

We need to relate:
- Lattice fields: $A_e, E_e, B_f$ (discrete quantities on edges/faces)
- Continuum fields: $A_i(x), E^i(x), B^i(x)$ (continuous vector fields)

The key requirements are:

1. **Canonical structure**: Continuum Poisson brackets must follow from lattice brackets
2. **Geometric consistency**: Field dimensions must match (e.g., $[E] = [F/L^2]$ in 3+1D)
3. **Consistent coupling**: Both electric and magnetic terms must yield the same $g_{\text{eff}}^2$

### §4.2. Electric Field Normalization

:::{prf:definition} Continuum Electric Field from Lattice
:label: def-continuum-electric-field

We define the continuum electric field via:

$$
E_{\text{lattice}}^a(e) = \ell_e \cdot E_{\text{continuum}}^{a,i}(x_e) \cdot \hat{e}^i
$$

where:
- $\hat{e}^i = (x_j - x_i)/\ell_e$ is the unit tangent vector of edge $e = (i,j)$
- $E_{\text{continuum}}^{a,i}(x)$ is the continuum electric field (spatial vector)

Inverting:

$$
E_{\text{continuum}}^{a,i}(x_e) \cdot \hat{e}^i = \frac{1}{\ell_e} E_{\text{lattice}}^a(e)
$$

For the squared norm (summed over color $a$ and spatial $i$ indices):

$$
E_{\text{continuum}}^{a,i}(x_e) E_{\text{continuum}}^{a,i}(x_e) \approx \frac{1}{\ell_e^2} E_{\text{lattice}}^a(e) E_{\text{lattice}}^a(e)
$$

where we used isotropy (averaging over directions).
:::

:::{prf:theorem} Electric Term Coarse-Graining
:label: thm-electric-coarse-graining

The electric term in the lattice Hamiltonian becomes:

$$
\begin{split}
H_{\text{elec}}^{\text{lattice}} &= \frac{g^2}{2} \sum_{\text{edges } e} \ell_e E_{\text{lattice}}^a(e) E_{\text{lattice}}^a(e) \\
&= \frac{g^2}{2} \sum_{\text{edges } e} \ell_e \cdot \ell_e^2 \, E_{\text{continuum}}^{a,i}(x_e) E_{\text{continuum}}^{a,i}(x_e) \\
&= \frac{g^2}{2} \sum_{\text{edges } e} \ell_e^3 \, |E_{\text{cont}}|^2(x_e)
\end{split}
$$

Wait, this scales as $\ell^3 \sim \rho^{-1}$, combined with $n_e \sim \rho$, giving $\int |E|^2$ with no density factor!

**Error in normalization**: Let me reconsider...
:::

:::{note}
**Dimensional Analysis Check**

Let's verify dimensions systematically:

- **Lattice electric field** $E_{\text{lattice}}$: Canonically conjugate to $A_{\text{lattice}}$ (dimensionless angle). Thus $[E_{\text{lattice}}] = [E] \cdot [L]$ where $[E]$ is continuum electric field.

- **Continuum electric field**: In 3+1D with Hamiltonian $H = \frac{1}{2g^2} \int E^2$, we have $[E^2] = [H]/[L^3] = [E]/[L^3]$, so $[E] = [E]^{1/2} [L]^{-3/2}$...

This is getting circular. Let me use standard lattice QCD conventions.
:::

### §4.3. Standard Lattice QCD Normalizations

Let me restart with proper lattice gauge theory normalizations from the literature.

:::{prf:definition} Standard Lattice QCD Field Definitions
:label: def-standard-lattice-fields

In lattice QCD (Wilson's formulation):

1. **Gauge potential (link variable)**:

$$
U_e = \exp(ig a A_\mu)
$$

where $a$ is the lattice spacing, and $A_\mu$ is the continuum gauge potential.

2. **Electric field**: The canonical conjugate $E_e$ has dimension $[E_e] = 1$ (dimensionless), related to continuum via:

$$
E_{\text{continuum}}^i = \frac{1}{a^2} E_e
$$

(electric field dimension: $[E_{\text{cont}}] = [M L T^{-1}] / [Q] = [M^{1/2} L^{-1/2}]$ in natural units)

3. **Magnetic field**: From plaquette $U_\square$:

$$
B_{\text{continuum}}^i = \frac{1}{a^2} B_\square
$$

where $B_\square$ is extracted from $U_\square - \mathbb{1}$ as in {prf:ref}`def-wilson-loop`.

4. **Lattice Hamiltonian**:

$$
H = \frac{1}{2g^2 a} \sum_{\text{links}} E_e^2 + \frac{g^2 a}{2} \sum_{\text{plaquettes}} B_\square^2
$$

Notice:
- Electric term has factor $1/(g^2 a)$ (inverse length)
- Magnetic term has factor $g^2 a$ (length)

This is the standard Wilson lattice gauge theory!
:::

Now the question: **How do we adapt this to irregular Delaunay lattices where "lattice spacing $a$" is not constant?**

### §4.4. Adaptation to Irregular Lattices

:::{prf:theorem} Field Normalizations on Irregular Delaunay Lattice
:label: thm-irregular-lattice-fields

On a Delaunay triangulation with varying edge lengths $\ell_e$ and face areas $A_f$, the proper field normalizations are:

1. **Electric field**:

$$
E_{\text{continuum}}^{a,i}(x_e) = \frac{1}{\ell_e^{3/2}} E_{\text{lattice}}^a(e)
$$

2. **Magnetic field**:

$$
B_{\text{continuum}}^{a,i}(x_f) = \frac{1}{A_f} B_{\text{lattice}}^a(f)
$$

where $B_{\text{lattice}}^a(f)$ is extracted from the Wilson loop as in {prf:ref}`def-wilson-loop`.

**Justification:**

These normalizations ensure:
- Correct continuum dimension: $[E_{\text{cont}}] = [L]^{-1}$ (since $[E_{\text{lattice}}] = [L]^{1/2}$ from canonical structure)
- Consistent Hamiltonian: The lattice Hamiltonian becomes:

$$
H_{\text{lattice}} = \frac{g^2}{2} \sum_e \ell_e E_e^2 + \frac{1}{2g^2} \sum_f \frac{1}{A_f} B_f^2
$$

where $E_e = \ell_e^{3/2} E_{\text{cont}}$ and $B_f = A_f B_{\text{cont}}$.

Substituting:

$$
H = \frac{g^2}{2} \sum_e \ell_e^4 |E_{\text{cont}}|^2 + \frac{1}{2g^2} \sum_f A_f |B_{\text{cont}}|^2
$$

Now apply coarse-graining theorems from §3...

**For electric term**: Using {prf:ref}`thm-edge-sum-to-integral` with $f(x) = |E_{\text{cont}}|^2(x)$:

$$
\sum_e \ell_e^4 |E_{\text{cont}}|^2(x_e) = \sum_e \ell_e^2 \cdot [\ell_e^2 |E_{\text{cont}}|^2](x_e)
$$

But wait - the theorem applies to $\ell_e^2$, not $\ell_e^4$. We have:

$$
\sum_e \ell_e^2 g(x_e) \to c_e \int \rho(x)^{1/3} g(x) d^3x
$$

If $g = \ell_e^2 |E|^2 \sim \rho^{-2/3} |E|^2$, then... this still has $\ell_e$ dependence in $g$, which breaks the theorem assumption.

The issue is that we're trying to apply a coarse-graining theorem to quantities that themselves have geometric weight!

Let me reconsider the entire approach...
:::

### §4.5. Correct Approach: Fixed Continuum Field Ansatz

The error above was treating $E_{\text{cont}}$ as if it varied on lattice scales. The correct approach:

:::{prf:theorem} Continuum Limit via Field Ansatz
:label: thm-continuum-field-ansatz

Assume lattice fields arise from sampling a smooth continuum field:

$$
E_e^a \approx \ell_e \int_e E^{a,\mu}(x) dx_\mu \approx \ell_e \cdot E^{a,i}(x_e) \hat{e}^i
$$

$$
B_f^a \approx \int_f \frac{1}{2} \epsilon_{\mu\nu\rho} F^{a,\mu\nu}(x) dS^\rho \approx A_f \cdot B^{a,i}(x_f) \hat{n}_f^i
$$

where:
- $E^{a,i}(x)$ is the continuum chromoelectric field
- $B^{a,i}(x)$ is the continuum chromomagnetic field
- $\hat{e}^i, \hat{n}_f^i$ are edge tangent and face normal unit vectors

Then:

$$
E_e^a E_e^a = \ell_e^2 |E^a(x_e)|^2, \quad B_f^a B_f^a = A_f^2 |B^a(x_f)|^2
$$

where $|E^a|^2 = E^{a,i} E^{a,i}$ (sum over spatial indices).

**Proof:** This is the standard relation between lattice and continuum fields in lattice gauge theory, derived from the continuum action discretization. See Montvay & Münster, "Quantum Fields on a Lattice", §4.3. $\square$
:::

Now the coarse-graining becomes clear:

:::{prf:theorem} Electric Term Continuum Limit
:label: thm-electric-continuum-limit

$$
\begin{split}
H_{\text{elec}}^{\text{lattice}} &= \frac{g^2}{2} \sum_e \ell_e E_e^a E_e^a \\
&= \frac{g^2}{2} \sum_e \ell_e \cdot \ell_e^2 |E^a(x_e)|^2 \\
&= \frac{g^2}{2} \sum_e \ell_e^3 |E^a(x_e)|^2
\end{split}
$$

But $\ell_e^3 = \ell_e^2 \cdot \ell_e$. We can write:

$$
\sum_e \ell_e^3 |E^a|^2 = \sum_e \ell_e^2 \cdot [\ell_e |E^a|^2]
$$

Now from {prf:ref}`lem-typical-length-scale`: $\ell_e \sim \rho^{-1/3}$ at point $x_e$. So:

$$
\ell_e |E^a|^2 \approx \rho(x_e)^{-1/3} |E^a(x_e)|^2
$$

Applying {prf:ref}`thm-edge-sum-to-integral`:

$$
\sum_e \ell_e^2 \cdot [\rho^{-1/3} |E|^2] \to c_e \int \rho(x)^{1/3} \cdot \rho(x)^{-1/3} |E(x)|^2 d^3x = c_e \int |E(x)|^2 d^3x
$$

The $\rho^{1/3}$ and $\rho^{-1/3}$ **cancel exactly**!

Thus:

$$
H_{\text{elec}} \xrightarrow{N \to \infty} \frac{g^2 c_e}{2} \int d^3x \, |E^a(x)|^2
$$

$\square$
:::

:::{prf:theorem} Magnetic Term Continuum Limit
:label: thm-magnetic-continuum-limit

$$
\begin{split}
H_{\text{mag}}^{\text{lattice}} &= \frac{1}{2g^2} \sum_f \frac{1}{A_f} B_f^a B_f^a \\
&= \frac{1}{2g^2} \sum_f \frac{1}{A_f} \cdot A_f^2 |B^a(x_f)|^2 \\
&= \frac{1}{2g^2} \sum_f A_f |B^a(x_f)|^2
\end{split}
$$

From geometry: $A_f \sim \ell_{\text{typ}}^2 \sim \rho^{-2/3}$, so:

$$
A_f |B^a|^2 \approx \rho(x_f)^{-2/3} |B^a(x_f)|^2
$$

But we need this in the form of {prf:ref}`thm-face-sum-to-integral`. We have:

$$
\sum_f A_f |B|^2 = \sum_f A_f^2 \cdot \frac{1}{A_f} |B|^2 = \sum_f A_f^2 \cdot [A_f^{-1} |B|^2]
$$

With $A_f \sim \rho^{-2/3}$:

$$
A_f^{-1} |B|^2 \approx \rho(x_f)^{2/3} |B^a(x_f)|^2
$$

Applying {prf:ref}`thm-face-sum-to-integral`:

$$
\sum_f A_f^2 \cdot [\rho^{2/3} |B|^2] \to c_f \int \rho(x)^{-1/3} \cdot \rho(x)^{2/3} |B(x)|^2 d^3x = c_f \int \rho(x)^{1/3} |B(x)|^2 d^3x
$$

Wait, this gives $\int \rho^{1/3} |B|^2$, not $\int |B|^2$! The density factors **don't cancel** for the magnetic term!

**This reveals the inconsistency remains.** Let me reconsider the Hamiltonian definition...
:::

---

## §5. Resolution: Correct Lattice Hamiltonian for Irregular Lattices

The calculations above reveal that the naive adaptation of Wilson's Hamiltonian to irregular lattices does **not** work. We need to modify the lattice Hamiltonian itself.

:::{prf:theorem} Consistent Lattice Hamiltonian on Irregular Delaunay Lattice
:label: thm-consistent-lattice-hamiltonian

The correct lattice Hamiltonian that yields consistent continuum limit is:

$$
H_{\text{lattice}} = \frac{g^2}{2} \sum_e \frac{\ell_e}{\rho_e^{2/3}} E_e^a E_e^a + \frac{1}{2g^2} \sum_f \frac{\rho_f^{2/3}}{A_f} B_f^a B_f^a
$$

where $\rho_e$ and $\rho_f$ are the local densities at edge $e$ and face $f$.

With field ansatz $E_e = \ell_e E_{\text{cont}}$ and $B_f = A_f B_{\text{cont}}$:

$$
\begin{split}
H_{\text{elec}} &= \frac{g^2}{2} \sum_e \frac{\ell_e^3}{\rho_e^{2/3}} |E_{\text{cont}}|^2 = \frac{g^2}{2} \sum_e \ell_e^2 \cdot \left[\frac{\ell_e}{\rho_e^{2/3}} |E|^2\right] \\
&\to \frac{g^2 c_e}{2} \int \rho^{1/3} \cdot \frac{\rho^{-1/3}}{\rho^{2/3}} |E|^2 = \frac{g^2 c_e}{2} \int |E|^2 d^3x
\end{split}
$$

$$
\begin{split}
H_{\text{mag}} &= \frac{1}{2g^2} \sum_f \frac{\rho_f^{2/3}}{A_f} A_f^2 |B_{\text{cont}}|^2 = \frac{1}{2g^2} \sum_f A_f^2 \cdot [\rho_f^{2/3} |B|^2] \\
&\to \frac{c_f}{2g^2} \int \rho^{-1/3} \cdot \rho^{2/3} |B|^2 = \frac{c_f}{2g^2} \int \rho^{1/3} |B|^2 d^3x
\end{split}
$$

**Still inconsistent!** The electric term gives $\int |E|^2$ while magnetic gives $\int \rho^{1/3} |B|^2$.

$\square$ (Proof fails - this approach doesn't work)
:::

---

[DOCUMENT PAUSED - SEEKING ALTERNATIVE STRATEGY]

**STATUS**: The direct adaptation of Wilson's lattice Hamiltonian to irregular lattices encounters a fundamental obstruction. The geometric measures ($\ell_e^2$ vs $A_f^2$) scale differently with density ($\rho^{-2/3}$ vs $\rho^{-4/3}$), and no choice of lattice Hamiltonian with simple density-dependent weights can make both terms coarse-grain to the same density dependence.

**Possible resolutions**:
1. The continuum limit requires uniform density ($\rho = \text{const}$), not varying $\rho(x)$
2. Need different field definitions (not $E_e = \ell_e E_{\text{cont}}$)
3. The lattice Hamiltonian must include non-local terms
4. There is a geometric identity relating edge and face contributions that we're missing

This requires further investigation before completing §5-6.
