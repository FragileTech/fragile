# The Fractal Set

## TLDR

**Complete Data Structure**: The Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\mathrm{CST}} \cup E_{\mathrm{IG}} \cup E_{\mathrm{IA}}, \mathcal{T})$ is a **2-dimensional simplicial complex with an oriented 1-skeleton** (a directed 2-complex) that records every aspect of the Fractal Gas algorithm's execution. Nodes represent spacetime points (walker $i$ at timestep $t$), CST edges encode temporal evolution, IG edges encode spatial coupling, and IA (influence attribution) edges close the causal loop from effect back to cause. The **interaction triangles** $\mathcal{T}$ are the fundamental 2-simplices—each triangle records one complete interaction: "walker $j$ influenced walker $i$'s evolution from $t$ to $t+1$."

**Frame-Invariant Storage via Spinors**: Nodes store **scalar quantities** (energy, fitness, status flags)—values that are identical in all coordinate systems. Edges store **spinor representations** of vectorial data (velocities, forces, gradients)—quantities that transform covariantly under rotations. This separation ensures the Fractal Set is a coordinate-free geometric object: the same physical information is recoverable regardless of which basis an observer chooses.

**Antisymmetric Selection Structure**: IG edges are **directed** with an antisymmetric cloning potential: $V_{\mathrm{clone}}(i \to j) = \Phi_j - \Phi_i = -V_{\mathrm{clone}}(j \to i)$. This sign flip under exchange is formally analogous to fermionic antisymmetry (without claiming physical fermions). The directed structure captures the asymmetric influence of fitness differences on cloning decisions.

**Triangular Interactions and Wilson Loops**: The fundamental closed loops are **3-cycles** (interaction triangles), not 4-cycles. Each triangle $\triangle_{ij,t}$ has vertices $(n_{j,t}, n_{i,t}, n_{i,t+1})$ connected by IG, CST, and IA edges. **Plaquettes** (4-cycles) are derived structures: two adjacent triangles sharing a diagonal. Wilson loops on triangles measure quantum phase accumulated around single interactions; plaquette Wilson loops factorize into products of triangle holonomies.

**Lossless Reconstruction Guarantee**: The Fractal Set contains sufficient information to reconstruct discrete-time dynamics: phase-space trajectories $(x_i(t), v_i(t))$, all force components $(\mathbf{F}_{\mathrm{stable}}, \mathbf{F}_{\mathrm{adapt}}, \mathbf{F}_{\mathrm{viscous}})$, diffusion tensor samples $\Sigma_{\mathrm{reg}}(x_i(t), S(t))$, fitness/reward values at visited points, and the empirical measure $f_k(t)$. Reconstruction is exact for stored scalars and machine-precision for spinor-encoded vectors; realized stochastic increments are stored explicitly via $\psi_{\mathrm{noise}}$.

## Introduction

:::{div} feynman-prose
Every physical measurement ultimately reduces to numbers that do not depend on who is doing the measuring or which direction they call "north." When you measure the temperature of a gas, you get the same answer whether you are facing east or west. When you measure the distance between two particles, you get the same number in any coordinate system. These are *scalars*—the most basic, frame-independent quantities in physics.

But physics also involves quantities that *do* depend on your coordinate choice: velocities, forces, gradients. If I say a particle is moving at 5 m/s "to the right," that statement depends on which direction I call right. These are *vectors*, and they transform in a specific way when you rotate your coordinate system. The challenge is: how do you store vector data in a coordinate-free way?

The answer is *spinors*. A spinor is a mathematical object that encodes the same information as a vector but transforms more naturally under rotations. When you rotate your coordinate system, the spinor transforms by the spin representation (algebraically simpler, and in low dimensions sometimes lower-dimensional than the vector representation). More importantly, two observers using different coordinate systems can recover the *same* vector from the *same* spinor—they just apply their respective transformations.

This document defines the **Fractal Set**: a complete data structure that records everything the Fractal Gas algorithm computes, stored in a coordinate-free way. Scalars go on nodes. Spinors go on edges. The result is a mathematical object that can be passed between observers, stored, analyzed, and reconstructed without any reference to a particular coordinate system.
:::

The Fractal Gas algorithm generates a complex web of interacting walkers evolving through a state space, with dynamics governed by kinetic forces, fitness-dependent selection, and stochastic diffusion. A complete record of this algorithm's execution must capture not only where each walker was at each timestep, but also the forces that acted upon it, the selection pressures it experienced, and the coupling with other walkers that influenced its cloning decisions.

The Fractal Set is designed to store all of this information with two key properties:

1. **Coordinate independence**: The data structure makes no commitment to any particular coordinate system. Observers using different bases can extract the same physical information.

2. **Lossless completeness**: No information about the **discrete-time execution** is lost. Given the Fractal Set, one can reconstruct the full dynamics at sampled timesteps and visited points without external data.

The structure separates data into two categories based on transformation properties:

| Category | Stored On | Transformation | Examples |
|----------|-----------|----------------|----------|
| **Scalars** | Nodes | Invariant (same in all frames) | Energy, fitness, status flags, norms |
| **Spinors** | Edges | Covariant (transform under $\mathrm{Spin}(d)$) | Velocities, forces, gradients, displacements |

:::{figure} ../../../svg_images/fractal_set_data_storage.svg
:name: fig-fractal-set-data-storage
:width: 100%

**What gets stored where.** Nodes carry only scalar data; CST and IG edges carry spinor-encoded vectors; IA edges store scalar attribution weights and phases.
:::

The edges divide into **three types** reflecting the algorithm's causal structure:

| Edge Type | Connects | Encodes | Directionality |
|-----------|----------|---------|----------------|
| **CST** (Causal Spacetime Tree) | $(n_{i,t}, n_{i,t+1})$ | Temporal evolution of single walker | Directed (time's arrow) |
| **IG** (Information Graph) | $(n_{i,t}, n_{j,t})$ | Spatial coupling between contemporaneous walkers | Directed (selection asymmetry) |
| **IA** (Influence Attribution) | $(n_{i,t+1}, n_{j,t})$ | Causal attribution from effect to cause | Directed (retrocausal) |

Together, the three edge types form a **directed 1-skeleton**: CST encodes timelike evolution, IG encodes spacelike coupling, and IA closes causal triangles by attributing each walker's update to its influencers. The underlying undirected support is simplicial.

:::{figure} ../../../svg_images/fractal_set_overview.svg
:name: fig-fractal-set-overview
:width: 100%

**Fractal Set on two time slices.** CST edges run forward in time, IG edges connect same-time walkers, and IA edges point back from effects to causes. One interaction triangle is highlighted.
:::

:::{figure} figures/cst-growth-tree-ig-ia.svg
:name: fig-fractal-set-growth-tree
:width: 100%

**CST growth tree with IG and IA edges.** CST edges build the forward-time tree, IG edges link contemporaneous walkers in sampled directed pairs, and IA edges attribute influence back across timesteps.
:::

---

## Overview

The Fractal Set captures the Fractal Gas algorithm as a **2-dimensional directed 2-complex** with the following components:

Let $\mathcal{P}_t$ denote the set of **ordered companion pairs** realized at timestep $t$ by the
companion selection operators (distance and cloning; {prf:ref}`def-fractal-set-companion-kernel`).
Define $m_t := |\mathcal{P}_t|$.

- **Nodes $\mathcal{N}$** (0-simplices): One node $n_{i,t}$ for each walker $i \in \{1, \ldots, N\}$ at each timestep $t \in \{0, 1, \ldots, T\}$. Total: $|\mathcal{N}| = N(T+1)$ nodes.

- **CST Edges $E_{\mathrm{CST}}$** (1-simplices): Directed edges $(n_{i,t}, n_{i,t+1})$ connecting consecutive timesteps of the same walker, provided the walker is alive at time $t$. These form a forest of directed paths (one per persistent walker ID); genealogical links from cloning are recorded separately.

- **IG Edges $E_{\mathrm{IG}}$** (1-simplices): Directed edges $(n_{i,t}, n_{j,t})$ for each sampled companion pair $(i, j) \in \mathcal{P}_t$ at the same timestep. At timestep $t$ with $k_t$ alive walkers, there are $m_t$ directed IG edges (one per sampled ordered pair).

- **IA Edges $E_{\mathrm{IA}}$** (1-simplices): Directed edges $(n_{i,t+1}, n_{j,t})$ connecting the effect (walker $i$ at $t+1$) to the cause (walker $j$ at $t$) for each $(i, j) \in \mathcal{P}_t$ with $t \in \{0,\ldots,T-1\}$. These **close the causal triangles**, attributing each walker's evolution to its sampled influencers. There are $m_t$ IA edges per **update** timestep.

- **Clone ancestry graph $E_{\mathrm{clone}}$** (derived): A directed relation from parent to child defined by the clone source attribute; these edges encode branching history but are **not** part of the CST order (see {prf:ref}`def-fractal-set-clone-ancestry`).

- **Interaction Triangles $\mathcal{T}$** (2-simplices): Each triangle $\triangle_{ij,t}$ has vertices $\{n_{j,t}, n_{i,t}, n_{i,t+1}\}$ and boundary edges (IG, CST, IA), for each $(i, j) \in \mathcal{P}_t$ and $t \in \{0,\ldots,T-1\}$. These are the **fundamental closed loops** of the structure.

- **Weight functions**: $\omega_{\mathrm{CST}}: E_{\mathrm{CST}} \to \mathbb{R}_{>0}$ assigns temporal weights (typically $\Delta t$), $\omega_{\mathrm{IG}}: E_{\mathrm{IG}} \to \mathbb{R}$ assigns selection coupling weights (the antisymmetric cloning potential), and $\omega_{\mathrm{IA}}: E_{\mathrm{IA}} \to [0,1]$ assigns influence attribution weights.

We assume $N \geq 2$ so interaction graphs are nontrivial.

:::{div} feynman-prose
Why do we need *three* types of edges? Because causality has three components: evolution, influence, and attribution.

**CST edges** (timelike) connect the same walker at different times. When walker $i$ moves from position $x$ to $x + \Delta x$, that transition is stored on a CST edge. The velocity, the forces, the diffusion—everything about *how* the walker evolved—lives on that edge.

**IG edges** (spacelike) connect sampled companion pairs at the same time. When walker $i$ draws a distance or cloning companion $j$, that comparison is stored on an IG edge. The fitness difference, the relative position, the viscous coupling—everything about *how* walkers influence each other in that sampled interaction—lives on these edges.

**IA edges** (diagonal) connect effects to causes across time. When walker $i$'s state at $t+1$ was influenced by walker $j$ at time $t$, that causal link is stored on an IA edge. These edges complete the triangle—they close the causal loop.

The triangle is the atom of interaction. A single influence event involves three nodes and three edges: $j$ at time $t$ (the source), $i$ at time $t$ (the receiver before), and $i$ at time $t+1$ (the receiver after). The IG edge says "j influenced i." The CST edge says "i evolved." The IA edge says "i's evolution was partly due to j." Together, they form an irreducible causal unit.
:::

:::{figure} ../../../svg_images/fractal_set_triangle.svg
:name: fig-fractal-set-triangle
:width: 80%

**The interaction triangle.** One IG edge (influence), one CST edge (evolution), and one IA edge (attribution) close a single causal loop.
:::

Let $m_t := |\mathcal{P}_t|$ be the number of sampled companion pairs at time $t$.

The following table summarizes the structural properties:

| Property | CST Edges | IG Edges | IA Edges |
|----------|-----------|----------|----------|
| **Direction** | Timelike ($t \to t+1$) | Spacelike (same $t$) | Diagonal ($t+1 \to t$) |
| **Cardinality** | $\sum_{t=0}^{T-1} |\mathcal{A}(t)|$ | $\sum_{t=0}^{T} m_t$ | $\sum_{t=0}^{T-1} m_t$ |
| **Topology** | Forest (acyclic) | Directed companion graph per $t$ | Bipartite per $(t, t+1)$ on sampled pairs |
| **Key weight** | Timestep $\Delta t$ | Cloning potential $V_{\mathrm{clone}}$ | Influence weight $w_{ij}$ |
| **Role in triangle** | Evolution edge | Influence edge | Attribution edge |

---

:::{admonition} The Big Picture: Why This Structure?
:class: feynman-added tip

Before diving into the technical details, here is the forest before the trees:

1. **Goal**: Store everything about an algorithm run in a coordinate-free way
2. **Problem**: Some quantities (energies) are coordinate-free already; others (velocities, forces) depend on coordinate choice
3. **Solution**: Split data into scalars (store on nodes) and vectors-as-spinors (store on edges)
4. **Bonus**: The resulting structure is a directed 2-complex with natural gauge-theoretic properties

The rest of this document makes these ideas precise. Section 1 explains the scalar/vector distinction. Section 2 defines nodes. Section 3 defines temporal edges (CST). Section 4 defines spatial edges (IG). Section 5 puts it all together with influence attribution (IA) edges and triangles. Section 6 proves that everything can be reconstructed. Section 7 shows how it works on curved manifolds. Section 8 gives practical guarantees.
:::

(sec-frame-invariance)=
## 1. Frame Invariance and Covariance

### 1.1 Observers and Coordinate Systems

A **coordinate system** on the state space $\mathcal{X} \subseteq \mathbb{R}^d$ is a choice of origin and orthonormal basis $\{e_1, \ldots, e_d\}$. Two observers using coordinate systems related by a rotation $R \in \mathrm{SO}(d)$ will represent the same physical configuration differently: if observer 1 assigns coordinates $x$ to a point, observer 2 assigns coordinates $x' = Rx$.

:::{prf:definition} Frame-Invariant Quantity (Scalar)
:label: def-fractal-set-scalar

A quantity $\phi: \mathcal{X} \to \mathbb{R}$ is **frame-invariant** (or a **scalar field**) if its value at any physical point is independent of coordinate choice:

$$\phi'(x') = \phi(x) \quad \text{for all } x \in \mathcal{X}, \; R \in \mathrm{SO}(d), \; x' = Rx.$$
Equivalently, $\phi' = \phi \circ R^{-1}$ implies $\phi'(Rx) = \phi(x)$.
:::

Scalars include: kinetic energy $E_{\mathrm{kin}} = \frac{1}{2}\|v\|^2$, potential energy $U(x)$, fitness $\Phi(x)$, distance norms $\|x - y\|$, and any quantity constructed solely from inner products and norms.

:::{prf:definition} Frame-Covariant Quantity (Vector)
:label: def-fractal-set-vector

A quantity $\mathbf{v}: \mathcal{X} \to \mathbb{R}^d$ is **frame-covariant** (or a **vector field**) if its components transform by the same rotation relating the coordinate systems:

$$\mathbf{v}'(x') = R\mathbf{v}(x) \quad \text{for all } x \in \mathcal{X}, \; R \in \mathrm{SO}(d), \; x' = Rx.$$
:::

Vectors include: position $x$, velocity $v$, force $\mathbf{F}$, gradient $\nabla\phi$, and any quantity that "points in a direction."

The key distinction: **scalars are the same number in all frames; vectors are the same geometric object but with different numerical components in different frames.**

### 1.2 The Spinor Representation

:::{div} feynman-prose
Now we come to what I think is the most beautiful part of this whole construction. You might ask: why not just store vectors as lists of numbers? What is all this spinor business about?

Here is the problem. When you write a vector as $(3, 4, 5)$, you have secretly made a choice: you have picked a coordinate system. The number "3" means nothing by itself—it only means "3 units in the direction I am calling $e_1$." If someone else uses a different $e_1$, they get different numbers for the same physical arrow.

Ordinarily this is fine—you just agree on coordinates. But what if you want to store data that can be read by *anyone*, in *any* coordinate system, without prior agreement? You need a representation that carries the geometry intrinsically, not relative to some arbitrary basis.

Spinors do exactly this. A spinor is a complex object that encodes directional information in a basis-independent way. Two observers using different coordinates can both extract vectors from the same spinor—and they will get answers that are related by exactly the rotation relating their coordinate systems. The spinor is the common ground.

This is not just mathematical elegance. It is operational: the Fractal Set can be serialized, transmitted, stored, and reconstructed by systems that never communicated about coordinate choices. The spinors carry the geometry with them.
:::

Vectors transform by $d \times d$ rotation matrices. For computational and geometric reasons, it is often preferable to represent vectors using **spinors**—complex objects that transform by a simpler (though higher-dimensional in complex terms) representation.

:::{prf:definition} Spinor Space
:label: def-fractal-set-spinor-space

For state space dimension $d$, fix a complex spinor module $\mathbb{S}_d$. For $d \geq 3$, choose a Clifford representation $\{\Gamma_i\}_{i=1}^d$ on $\mathbb{S}_d$. For $d = 2$, we use the minimal Spin(2) module $\mathbb{C}$ with the quadratic map $v = \psi^2$ (already full for $\mathbb{R}^2$). For odd $d$, $\dim_{\mathbb{C}} \mathbb{S}_d = 2^{(d-1)/2}$. For even $d \geq 4$, the Dirac module has $\dim_{\mathbb{C}} = 2^{d/2}$ and splits into two Weyl (chiral) modules; to represent **all** vectors we use the full Dirac module. A **spinor** is an element $\psi \in \mathbb{S}_d$.
:::

The spinor dimension depends on $d$. We denote $s_d := \dim_{\mathbb{C}} \mathbb{S}_d$ for the representation chosen for storage. The Fractal Set uses the minimal Spin(2) spinor in $d = 2$, Dirac spinors in even dimensions $d \geq 4$ (full vector coverage), and minimal spinors in odd dimensions. Example choices:

| Dimension $d$ | Storage module $\mathbb{S}_d$ | Complex Dimension | Real Parameters | Notes |
|---------------|-------------------------------|-------------------|-----------------|-------|
| 2 | $\mathbb{C}$ | 1 | 2 | Spin(2) spinor with $v = \psi^2$ |
| 3 | $\mathbb{C}^2$ | 2 | 4 | Pauli spinors |
| 4 | $\mathbb{C}^4$ | 4 | 8 | Dirac spinors (full vector coverage) |
| 5 | $\mathbb{C}^4$ | 4 | 8 | Minimal odd-dimensional spinor |
| 6 | $\mathbb{C}^8$ | 8 | 16 | Dirac spinors (full vector coverage) |
| 7 | $\mathbb{C}^8$ | 8 | 16 | Minimal odd-dimensional spinor |
| 8 | $\mathbb{C}^{16}$ | 16 | 32 | Dirac spinors (full vector coverage) |

For general $d$, the Dirac spinor dimension is $2^{\lfloor d/2 \rfloor}$. Storage and complexity scale with the chosen $s_d$.

:::{prf:definition} Spinor Representation of $\mathrm{SO}(d)$
:label: def-fractal-set-spinor-rep

The **spinor representation** is a group homomorphism $S: \mathrm{Spin}(d) \to \mathrm{GL}(\mathbb{S}_d)$ where $\mathrm{Spin}(d)$ is the double cover of $\mathrm{SO}(d)$. For each rotation $R \in \mathrm{SO}(d)$, there exist exactly two preimages $\pm U \in \mathrm{Spin}(d)$ such that the **equivariant quadratic map** $\pi$ satisfies:

$$\pi(S(U)\psi) = R \, \pi(\psi).$$

For $d \geq 3$ (or any Clifford module), $\pi(\psi)_i = \psi^\dagger \Gamma_i \psi$. For $d = 2$, $\pi(\psi) = \psi^2$.

Equivalently, the following diagram commutes:

$$\begin{array}{ccc}
\mathbb{S}_d & \xrightarrow{S(U)} & \mathbb{S}_d \\
\downarrow \pi & & \downarrow \pi \\
\mathbb{R}^d & \xrightarrow{R} & \mathbb{R}^d
\end{array}$$

A vector-to-spinor map $\iota$ is then a choice of section of $\pi$ (a gauge-fixing), introduced below.
:::

The sign ambiguity ($\pm U$ both correspond to $R$) reflects the double-cover structure and is a **gauge freedom**—both choices encode the same physical vector.

:::{note}
:class: feynman-added

**The double cover: why $\pm U$ give the same rotation.** Imagine rotating a coffee cup 360 degrees. It returns to its starting position—that is the rotation $R = I$. But in spinor space, a 360-degree rotation corresponds to $U = -I$, not $U = +I$. You need to rotate *twice* (720 degrees) to get $U = +I$.

This strange fact is not a bug; it is how spinors detect orientation. Fermions (like electrons) pick up a minus sign under 360-degree rotations; via the spin-statistics theorem this is related to antisymmetry under exchange. For our purposes, the double cover means two spinor transformations give the same vector transformation—we just pick one consistently.

The gauge freedom is real but harmless: both choices $\pm U$ produce the same physical answer when you extract vectors from spinors.
:::

### 1.3 Vector-Spinor Correspondence

:::{div} feynman-prose
Now we need the actual recipe: given a vector, how do you make a spinor? And given a spinor, how do you get the vector back?

The key tool is the Clifford algebra—a mathematical structure that encodes the geometry of orthonormal bases. Do not worry too much about the formalism; what matters is the result. In low dimensions, the conversion is simple and explicit.

In 2D, we identify $\mathbb{R}^2$ with $\mathbb{C}$ and represent a vector by a **spinor square**: $v = \psi^2$. This keeps the half-angle action of Spin(2) and leaves a $\mathbb{Z}_2$ sign ambiguity (phase $\pi$) as the gauge freedom.

In 3D, things get more interesting. A vector becomes a 2-component complex spinor using spherical coordinates. With our normalization, $\|\psi\|^2 = \|v\|$. The **relative phase and amplitudes** encode the direction; the **overall U(1) phase** does *not* change the vector. This is the Hopf fibration in disguise: $S^3 / S^1 \cong S^2$.

In higher dimensions, the pattern continues but gets more complicated. The spinor dimension grows exponentially in $d$ (Dirac scales as $2^{\lfloor d/2 \rfloor}$), which eventually exceeds the vector dimension. This is the price of coordinate independence: you need extra degrees of freedom to encode the transformation properties.

The critical property is *roundtrip consistency*: if you convert a vector to a spinor and back, you get the same vector.
:::

The explicit correspondence uses the **Clifford algebra** $\mathrm{Cl}(d)$ generated by $\{e_1, \ldots, e_d\}$ with $e_i e_j + e_j e_i = 2\delta_{ij}$. We use the Euclidean convention so the representing matrices $\Gamma_i$ can be taken Hermitian, which makes $\psi^\dagger \Gamma_i \psi$ real.

:::{prf:definition} Vector-to-Spinor Map
:label: def-fractal-set-vec-to-spinor

The **vector-to-spinor embedding** $\iota: \mathbb{R}^d \to \mathbb{S}_d$ encodes vectors as spinors with a **canonical phase convention**. For $v = \sum_i v_i e_i \in \mathbb{R}^d$:

**Case $d = 2$**: Spinor square representation. Write $v = v_1 + i v_2 = r e^{i\phi}$ with $\phi \in (-\pi, \pi]$ and define

$$\iota(v) = \sqrt{r} \, e^{i\phi/2} \in \mathbb{C}.$$

Then the extraction map is $\pi(\psi) = \psi^2$, so $\pi(\iota(v)) = v$. The phase convention fixes the sign ambiguity of the square root.

**Case $d = 3$**: Using the **canonical lift** with real first component when possible,

$$\iota(v) = \sqrt{\|v\|} \begin{pmatrix} \cos(\theta/2) \\ \sin(\theta/2) e^{i\phi} \end{pmatrix} \in \mathbb{C}^2,$$
where $(\|v\|, \theta, \phi)$ are spherical coordinates: $v = \|v\|(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$.

**Equivalently**, for $v \neq 0$: construct the Hermitian matrix $v \cdot \boldsymbol{\sigma} = v_1 \sigma_1 + v_2 \sigma_2 + v_3 \sigma_3$ and take the eigenvector with eigenvalue $+\|v\|$, normalized to have $\|\psi\|^2 = \|v\|$ and first component real non-negative.

**For $v = 0$**: $\iota(0) = 0$ in any dimension.

**General $d \geq 3$**: The embedding extends via the chosen Clifford representation on $\mathbb{S}_d$, with analogous phase conventions ensuring a unique representative in each gauge class.

**Remark**: For $d \geq 3$, the map $\iota$ is not linear. Any global phase convention introduces a branch cut; $\iota$ is smooth on each gauge patch but cannot be globally continuous because the $\mathrm{U}(1)$ bundle $S^{2n+1} \to \mathbb{C}P^n$ is nontrivial.
:::

:::{prf:definition} Spinor-to-Vector Map
:label: def-fractal-set-spinor-to-vec

The **spinor-to-vector extraction** $\pi: \mathbb{S}_d \to \mathbb{R}^d$ is the left inverse of $\iota$ (on the image of $\iota$):

**Case $d = 2$**: For $\psi \in \mathbb{C}$ identified with $\mathbb{R}^2$,

$$\pi(\psi) = \psi^2 = \mathrm{Re}(\psi^2) \, e_1 + \mathrm{Im}(\psi^2) \, e_2.$$

**Case $d = 3$**: For $\psi = \begin{pmatrix} \alpha \\ \beta \end{pmatrix} \in \mathbb{C}^2$,

$$\pi(\psi) = \begin{pmatrix} 2\mathrm{Re}(\alpha^*\beta) \\ 2\mathrm{Im}(\alpha^*\beta) \\ |\alpha|^2 - |\beta|^2 \end{pmatrix}.$$

**General $d \geq 3$**: With a fixed Clifford representation $\{\Gamma_i\}$,

$$\pi(\psi)_i = \psi^\dagger \Gamma_i \psi,$$

which is equivariant under $\mathrm{Spin}(d)$ and reduces to the low-dimensional formulas above.
:::

:::{prf:proposition} Transformation Covariance
:label: prop-fractal-set-spinor-covariance

For any rotation $R \in \mathrm{SO}(d)$ with spinor lift $U \in \mathrm{Spin}(d)$ (either of the two preimages), the following holds:

$$\pi(U \psi) = R \cdot \pi(\psi) \quad \text{and} \quad \pi(U \cdot \iota(v)) = Rv$$
for all $v \in \mathbb{R}^d$ and $\psi \in \mathbb{S}_d$ in the image of $\iota$. With a fixed phase convention, $U \cdot \iota(v)$ represents $Rv$ and may differ from $\iota(Rv)$ by a unit phase.

*Proof.* This follows from the defining property of the spinor representation: the diagram in {prf:ref}`def-fractal-set-spinor-rep` commutes. $\square$
:::

:::{div} feynman-prose
What is the point of this spinor business? Why not just store vectors as $d$-tuples of real numbers?

The answer is *geometric naturalness*. A spinor is a coordinate-free object. When you store a vector as $(v_1, v_2, v_3)$, you have implicitly chosen a basis. To recover the geometric vector, you need to know which basis was used. But when you store a spinor $\psi$, you have stored the geometric information directly. Any observer can extract their own coordinate representation by applying their own spinor-to-vector map.

This is not just aesthetic. When the Fractal Set is passed between systems, serialized to disk, or analyzed by different components, there is no need to track "which coordinate system was used." The spinor *is* the data, and the coordinates are derived as needed.
:::

---

(sec-nodes)=
## 2. Nodes: Spacetime Points with Scalar Data

:::{div} feynman-prose
Let me tell you what a node really is. A node is a *moment in a walker's life*—a snapshot of "who" (which walker), "when" (which timestep), and "how" (what was its state). But here is the key constraint: the node stores only what can be known without picking a direction.

Think of it this way. Suppose you are blind but can feel temperature and hear sounds. You can measure how hot something is (a scalar). You can measure how loud it is (a scalar). But you cannot measure which direction the heat source is—that requires vision, which gives you directional information. The node is like a blind observer: it knows energy, fitness, whether the walker is alive or dead, but it does not know which way things are pointing.

Why this restriction? Because direction requires a coordinate system, and we are building a coordinate-free data structure. Everything directional—velocities, forces, gradients—gets stored on edges, where the spinor representation handles the geometry. The node stays clean: pure scalars, pure frame-invariance.

This separation is not arbitrary. It reflects a fundamental distinction in physics between intrinsic properties (mass, charge, energy) and relational properties (velocity relative to what? force in which direction?). The node captures the intrinsic; the edges capture the relational.
:::

### 2.1 Node Set Definition

:::{prf:definition} Spacetime Node
:label: def-fractal-set-node

A **spacetime node** $n_{i,t}$ represents walker $i \in \{1, \ldots, N\}$ at discrete timestep $t \in \{0, 1, \ldots, T\}$. The **node set** is:

$$\mathcal{N} := \{n_{i,t} : i \in \{1, \ldots, N\}, \; t \in \{0, \ldots, T\}\}.$$
The cardinality is $|\mathcal{N}| = N(T+1)$.
:::

Each node represents a single "event" in the algorithm's spacetime: a specific walker at a specific moment. The node set forms a rectangular grid $\{1, \ldots, N\} \times \{0, \ldots, T\}$, though connectivity (via edges) respects the actual alive/dead status of walkers.

### 2.2 Node Scalar Attributes

Nodes store **only scalar quantities**—values that are frame-invariant.

:::{prf:definition} Node Scalar Attributes
:label: def-fractal-set-node-attributes

Each node $n_{i,t} \in \mathcal{N}$ carries the following scalar attributes:

**Identity attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Walker ID | $\mathrm{id}(n)$ | $\mathbb{Z}_+$ | [count] | Unique identifier for walker |
| Timestep | $t(n)$ | $\mathbb{Z}_{\geq 0}$ | [count] | Discrete time index |
| Node ID | $\mathrm{nid}(n)$ | $\mathbb{Z}_+$ | [count] | Unique identifier for node |

**Temporal attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Proper time | $\tau(n)$ | $\mathbb{R}_{\geq 0}$ | [time] | Continuous time: $\tau = t \cdot \Delta t$ |
| Timestep duration | $\Delta t$ | $\mathbb{R}_{>0}$ | [time] | Time between consecutive steps |

**Status attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Alive flag | $s(n)$ | $\{0, 1\}$ | [boolean] | 1 if walker alive at this timestep, 0 otherwise |
| Clone source | $c(n)$ | $\mathbb{Z}_+ \cup \{\bot\}$ | [count] | ID of cloning source if cloned this step, $\bot$ otherwise |

**Energy attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Kinetic energy | $E_{\mathrm{kin}}(n)$ | $\mathbb{R}_{\geq 0}$ | [energy] | $\frac{1}{2}\|v\|^2$ at this node |
| Potential energy | $U(n)$ | $\mathbb{R}$ | [energy] | Potential energy $U(x)$ at position |
| Total energy | $E(n)$ | $\mathbb{R}$ | [energy] | $E_{\mathrm{kin}}(n) + U(n)$ |

**Fitness attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Fitness | $\Phi(n)$ | $\mathbb{R}_{\geq 0}$ | [dimensionless] | Objective function value $\Phi(x)$ |
| Virtual reward | $V_{\mathrm{fit}}(n)$ | $\mathbb{R}$ | [dimensionless] | Localized fitness potential $V_{\mathrm{fit}}[f_k, \rho](x)$ |
| Reward signal | $r(n)$ | $\mathbb{R}$ | [dimensionless] | Instantaneous reward (if applicable) |

**Localized statistics:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Local mean | $\mu_\rho(n)$ | $\mathbb{R}$ | [dimensionless] | Kernel-weighted mean fitness around $x$ |
| Local std | $\sigma_\rho(n)$ | $\mathbb{R}_{\geq 0}$ | [dimensionless] | Kernel-weighted std of fitness |
| Local derivative | $\sigma'_\rho(n)$ | $\mathbb{R}$ | [1/distance] | Derivative of $\sigma_\rho$ w.r.t. $\rho$ |
| Partition function | $Z_\rho(n)$ | $\mathbb{R}_{>0}$ | [dimensionless] | Normalizing constant for kernel |

**Global parameters (constant across nodes):**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Fermi energy | $\epsilon_F$ | $\mathbb{R}$ | [energy] | Selection threshold parameter |
| Viscosity | $\nu$ | $\mathbb{R}_{\geq 0}$ | [1/time] | Viscous coupling strength |
| Friction | $\gamma$ | $\mathbb{R}_{\geq 0}$ | [1/time] | Velocity damping coefficient |
| Localization scale | $\rho$ | $\mathbb{R}_{>0}$ | [distance] | Kernel bandwidth |
| Diffusion floor | $\epsilon_\Sigma$ | $\mathbb{R}_{>0}$ | [dimensionless] | Regularization for diffusion tensor |
:::

:::{prf:definition} Clone Ancestry Relation
:label: def-fractal-set-clone-ancestry

The **clone ancestry graph** is the derived directed relation

$$
E_{\mathrm{clone}} := \{(n_{j,t-1}, n_{i,t}) : c(n_{i,t}) = j \neq \bot\}.
$$

This encodes genealogical branching (parent $j$ to child $i$) and can be read directly from the
clone source attribute or from IA edges with $\chi_{\mathrm{clone}}=1$. These edges are **not**
part of the CST order or distance; they are interaction/genealogy data only.
:::

### 2.3 Frame-Invariance of Node Data

:::{prf:proposition} Node Scalars are Frame-Invariant
:label: prop-fractal-set-node-invariance

Every attribute in {prf:ref}`def-fractal-set-node-attributes` is a scalar field in the sense of {prf:ref}`def-fractal-set-scalar`.

*Proof.* Each attribute falls into one of the following categories:

1. **Discrete identifiers** ($\mathrm{id}$, $t$, $\mathrm{nid}$, $s$, $c$): Pure labels with no geometric content.

2. **Time coordinates** ($\tau$, $\Delta t$): Time is a scalar (same in all spatial coordinate systems under $\mathrm{SO}(d)$).

3. **Norms and scalar fields** ($E_{\mathrm{kin}} = \frac{1}{2}\|v\|^2$, $U(x)$, $\Phi(x)$, $V_{\mathrm{fit}}$): Norms are invariant, and scalar fields assign coordinate-independent values to points.

4. **Statistical aggregates** ($\mu_\rho$, $\sigma_\rho$, $Z_\rho$): Defined via integration over scalar kernels.

5. **Constants** ($\epsilon_F$, $\nu$, $\gamma$, $\rho$, $\epsilon_\Sigma$): Parameters independent of position or orientation.

None of these scalar quantities depend on coordinate choice. $\square$
:::

:::{div} feynman-prose
The node knows only what can be measured with a ruler and a stopwatch, not a compass. It knows how much energy the walker has, but not which direction it is moving. It knows how fit the walker is, but not which way the fitness gradient points. All the directional information—velocities, forces, gradients—is stored on edges, where the spinor representation handles the coordinate bookkeeping.

This separation is not arbitrary. It reflects a deep physical principle: the state of a system at a single instant can be characterized by coordinate-free quantities, but the *evolution* of a system—how it changes from one instant to the next—inherently involves directions and thus requires covariant data.
:::

---

(sec-cst-edges)=
## 3. CST Edges: Temporal Evolution with Spinor Data

### 3.1 Edge Set Definition

:::{prf:definition} CST Edge Set
:label: def-fractal-set-cst-edges

The **Causal Spacetime Tree (CST) edge set** is:

$$E_{\mathrm{CST}} := \{(n_{i,t}, n_{i,t+1}) : i \in \{1, \ldots, N\}, \; t \in \{0, \ldots, T-1\}, \; s(n_{i,t}) = 1\}.$$
Each CST edge connects a walker to its immediate temporal successor, provided the walker is alive. The edges are **directed** from earlier to later time.
:::

CST edges form a **forest** of worldlines: each walker's trajectory is a directed path with no forward branching, and the union over all walkers is a forest. Cloning does **not** add CST edges; when $c(n_{i,t}) \neq \bot$, the prior CST path ends and a new CST path begins at $n_{i,t}$. Genealogical branching is recorded separately via $E_{\mathrm{clone}}$ (or IA edges with $\chi_{\mathrm{clone}}=1$), not in the CST itself.

:::{prf:definition} Alive Walker Set
:label: def-fractal-set-alive-set

At timestep $t$, the **alive walker set** is:

$$\mathcal{A}(t) := \{i \in \{1, \ldots, N\} : s(n_{i,t}) = 1\}.$$
The number of alive walkers is $k_t := |\mathcal{A}(t)|$.

In the Fractal Gas algorithm the population is conserved, so $k_t = N$ for all $t$ (walkers may change state, but slots persist).
:::

### 3.2 Causal Set Axioms

:::{div} feynman-prose
Here is something that should make you sit up. The structure we have built—this forest of CST edges connecting walkers across time—turns out to satisfy the mathematical axioms of *causal set theory*, which is one of the serious candidates for quantum gravity.

Causal set theory says: spacetime is fundamentally discrete, and the only structure that matters is the causal ordering—who came before whom. You do not need coordinates, you do not need distances, you do not even need a notion of "space." All you need is a set of events and a relation that says "this event could have caused that event."

The three axioms are almost embarrassingly simple. They say: you cannot be your own grandparent (no time travel). If A caused B and B caused C, then A caused C (causality is transitive). And between any two events, only finitely many things can happen (no infinite regress).

The remarkable thing is that these axioms are *enough* to recover much of the geometry of spacetime. The causal structure determines the conformal structure—the "shape" of spacetime up to an overall scale. Our Fractal Set, by satisfying these axioms, inherits this geometric richness.

I am not claiming that the Fractal Gas is a theory of quantum gravity. But I am pointing out that the mathematical structure we need for coordinate-free data storage is the *same* mathematical structure that appears in fundamental physics. This is not coincidence; it reflects the fact that coordinate-independence is a deep physical principle, not just a software engineering convenience.
:::

The CST structure satisfies the axioms of **causal set theory**, making it a valid discrete model of spacetime.

:::{prf:definition} Causal Set Axioms
:label: def-fractal-set-cst-axioms

A **causal set** is a pair $(\mathcal{C}, \prec)$ where $\mathcal{C}$ is a set and $\prec$ is a binary relation satisfying:

1. **CS1 (Irreflexivity)**: $\forall x \in \mathcal{C}: \neg(x \prec x)$. No element is its own ancestor.

2. **CS2 (Transitivity)**: $\forall x, y, z \in \mathcal{C}: (x \prec y \land y \prec z) \Rightarrow x \prec z$. The ancestor relation is transitive.

3. **CS3 (Local Finiteness)**: $\forall x, z \in \mathcal{C}: |\{y \in \mathcal{C} : x \prec y \prec z\}| < \infty$. Causal intervals contain finitely many elements.
:::

:::{prf:proposition} CST Satisfies Causal Set Axioms
:label: prop-fractal-set-cst-causal

Define the causal relation $\prec_{\mathrm{CST}}$ on $\mathcal{N}$ as the transitive closure of $E_{\mathrm{CST}}$:

$$n_{i,t} \prec_{\mathrm{CST}} n_{j,s} \iff \exists \text{ directed path in } E_{\mathrm{CST}} \text{ from } n_{i,t} \text{ to } n_{j,s}.$$
Then $(\mathcal{N}, \prec_{\mathrm{CST}})$ satisfies CS1, CS2, and CS3.

*Proof.*

**CS1 (Irreflexivity)**: CST edges always increase the timestep: $(n_{i,t}, n_{i,t+1})$ has $t+1 > t$. Any path in $E_{\mathrm{CST}}$ strictly increases the timestep, so no path can return to its starting node. Thus $\neg(n \prec_{\mathrm{CST}} n)$ for all $n$.

**CS2 (Transitivity)**: By definition, $\prec_{\mathrm{CST}}$ is the transitive closure, hence transitive.

**CS3 (Local Finiteness)**: If $n_{i,t} \prec_{\mathrm{CST}} n_{j,s}$, then $t < s$. The causal interval $\{n : n_{i,t} \prec_{\mathrm{CST}} n \prec_{\mathrm{CST}} n_{j,s}\}$ contains only nodes with timesteps in $\{t+1, \ldots, s-1\}$, which is finite. Moreover, at each timestep there are at most $N$ walkers. Thus the interval has at most $N(s-t-1) < \infty$ elements. $\square$
:::

:::{div} feynman-prose
The CST structure encodes time's arrow in the graph. You cannot be your own grandparent. If $A$ came before $B$ and $B$ came before $C$, then $A$ came before $C$. And between any two events, only finitely many things can happen.

These are not just mathematical properties—they are the defining features of causality. A universe without these properties would allow time travel, causal loops, and the paradoxes that come with them. The Fractal Gas, by construction, generates a causally consistent spacetime structure.
:::

### 3.3 CST Edge Spinor Attributes

Each CST edge stores the spinor representations of all vectorial quantities involved in the temporal evolution.

:::{prf:definition} CST Edge Spinor Attributes
:label: def-fractal-set-cst-attributes

Each CST edge $e = (n_{i,t}, n_{i,t+1}) \in E_{\mathrm{CST}}$ carries the following attributes:

**Identity attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Walker ID | $\mathrm{id}(e)$ | $\mathbb{Z}_+$ | [count] | Walker this edge belongs to |
| Start timestep | $t(e)$ | $\mathbb{Z}_{\geq 0}$ | [count] | Timestep of source node |
| Edge ID | $\mathrm{eid}(e)$ | $\mathbb{Z}_+$ | [count] | Unique edge identifier |

**Velocity spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Initial velocity | $\psi_{v,t}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $v_i(t)$ |
| Final velocity | $\psi_{v,t+1}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $v_i(t+1)$ |
| Velocity increment | $\psi_{\Delta v}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $\Delta v = v_i(t+1) - v_i(t)$ |

**Position spinor:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Displacement | $\psi_{\Delta x}(e)$ | $\mathbb{S}_d$ | [distance] | Spinor of $\Delta x = x_i(t+1) - x_i(t)$ |

**Force spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Stable force | $\psi_{\mathbf{F}_{\mathrm{stable}}}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $\mathbf{F}_{\mathrm{stable}}(x_i)$ |
| Adaptive force | $\psi_{\mathbf{F}_{\mathrm{adapt}}}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $\mathbf{F}_{\mathrm{adapt}}(x_i, S)$ |
| Viscous force | $\psi_{\mathbf{F}_{\mathrm{viscous}}}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $\mathbf{F}_{\mathrm{viscous}}(x_i, S)$ |
| Friction force | $\psi_{\mathbf{F}_{\mathrm{friction}}}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $-\gamma v_i$ |
| Total force | $\psi_{\mathbf{F}_{\mathrm{total}}}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $\mathbf{F}_{\mathrm{total}} = \sum \mathbf{F}_{\cdot}$ |

**Diffusion spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Diffusion tensor | $\psi_{\Sigma_{\mathrm{reg}}}(e)$ | $\mathbb{S}_d^{\otimes 2}$ | [distance/time^{3/2}] | Spinor encoding of $\Sigma_{\mathrm{reg}}(x_i, S)$ |
| Noise realization | $\psi_{\mathrm{noise}}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of the stochastic increment $\Sigma_{\mathrm{reg}} \circ dW_i$ |

**Gradient spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Potential gradient | $\psi_{\nabla U}(e)$ | $\mathbb{S}_d$ | [energy/distance] | Spinor of $\nabla U(x_i)$ |
| Fitness gradient | $\psi_{\nabla \Phi}(e)$ | $\mathbb{S}_d$ | [1/distance] | Spinor of $\nabla \Phi(x_i)$ |
| Virtual reward gradient | $\psi_{\nabla V_{\mathrm{fit}}}(e)$ | $\mathbb{S}_d$ | [1/distance] | Spinor of $\nabla V_{\mathrm{fit}}(x_i)$ |

**Derived scalars (stored for efficiency):**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Velocity norm change | $\|\Delta v\|(e)$ | $\mathbb{R}_{\geq 0}$ | [distance/time] | $\|\Delta v\|$ |
| Displacement norm | $\|\Delta x\|(e)$ | $\mathbb{R}_{\geq 0}$ | [distance] | $\|\Delta x\|$ |
| Timestep | $\Delta t(e)$ | $\mathbb{R}_{>0}$ | [time] | Time duration of this step |
:::

### 3.4 Relationship to SDE Dynamics

The CST edge data encodes one complete step of the stochastic differential equation governing walker dynamics.

:::{prf:definition} Adaptive Gas SDE
:label: def-fractal-set-sde

The Adaptive Gas dynamics for walker $i$ with state $(x_i, v_i)$ is governed by:

$$dv_i = \left[\mathbf{F}_{\mathrm{stable}}(x_i) + \mathbf{F}_{\mathrm{adapt}}(x_i, S) + \mathbf{F}_{\mathrm{viscous}}(x_i, S) - \gamma v_i\right] dt + \Sigma_{\mathrm{reg}}(x_i, S) \circ dW_i,$$

$$dx_i = v_i \, dt,$$
where:
- $\mathbf{F}_{\mathrm{stable}}(x) = -\nabla U(x)$ is the conservative force from the potential
- $\mathbf{F}_{\mathrm{adapt}}(x, S) = -\nabla V_{\mathrm{fit}}[f_k, \rho](x)$ is the adaptive force from the fitness landscape
- $\mathbf{F}_{\mathrm{viscous}}(x, S) = \nu \sum_{j \neq i} K_\rho(x_i, x_j)(v_j - v_i)$ is the viscous coupling force
- $\Sigma_{\mathrm{reg}}(x, S)$ is the fitness-adapted diffusion tensor
- $dW_i$ is a standard Wiener process
:::

:::{prf:proposition} CST Edge Encodes Complete Kinetic Update
:label: prop-fractal-set-cst-sde

Given the CST edge attributes for $e = (n_{i,t}, n_{i,t+1})$, the complete evolution from $(x_i(t), v_i(t))$ to $(x_i(t+1), v_i(t+1))$ can be reconstructed:

$$v_i(t) = \pi(\psi_{v,t}(e)), \quad v_i(t+1) = \pi(\psi_{v,t+1}(e)),$$

$$\Delta x = \pi(\psi_{\Delta x}(e)), \quad x_i(t+1) = x_i(t) + \Delta x,$$

and all force components:

$$\mathbf{F}_{\cdot}(x_i, S, t) = \pi(\psi_{\mathbf{F}_\cdot}(e)).$$

*Proof.* By {prf:ref}`prop-fractal-set-spinor-covariance`, the spinor-to-vector map $\pi$ recovers the geometric vector from its spinor representation. The spinor attributes store exactly the quantities appearing in the SDE {prf:ref}`def-fractal-set-sde`. $\square$
:::

:::{div} feynman-prose
Each CST edge is a complete snapshot of one integration step. It tells you where the walker started, where it ended up, what forces pushed it, and how much randomness intervened. If you line up all the CST edges for a single walker, you get its complete trajectory through phase space. If you line up all the force spinors, you get a record of every push the system gave that walker.

The beauty is that this record is coordinate-free. Two observers can look at the same CST edge, apply their respective spinor-to-vector maps, and reconstruct the trajectory in their own coordinates—and they will agree on all physical observables like energy, distance traveled, and work done by forces.
:::

### 3.5 Spinor Operations Complexity

:::{prf:proposition} Spinor Conversion Complexity
:label: prop-fractal-set-spinor-complexity

The vector-to-spinor map $\iota: \mathbb{R}^d \to \mathbb{S}_d$ and spinor-to-vector map $\pi: \mathbb{S}_d \to \mathbb{R}^d$ have dimension-dependent complexity:

| Dimension $d$ | $\iota$ Time | $\pi$ Time | Spinor Storage |
|---------------|-------------|------------|----------------|
| 2 | $O(1)$ | $O(1)$ | 2 reals |
| 3 | $O(1)$ | $O(1)$ | 4 reals |
| 4 | $O(1)$ | $O(1)$ | 8 reals |
| General | $O(s^3)$ | $O(d\,s^2)$ | $2s$ reals where $s = \dim_{\mathbb{C}}\mathbb{S}_d$ |

For $d \leq 4$, conversions are constant-time with fixed arithmetic operations. For general $d$, the eigenvector computation in $\iota$ is $O(s^3)$ in the worst case for an $s \times s$ Hermitian matrix, while $\pi$ (the bilinear forms $\psi^\dagger \Gamma_i \psi$) is $O(s^2)$ per component, giving $O(d\,s^2)$ total. Sparse Clifford representations can reduce constants.

*Proof.* For $d = 2$: constant-time complex square root/extraction. For $d = 3$: trigonometric functions and 2×2 matrix operations. For general $d$: eigenvalue decomposition of $s \times s$ Hermitian matrix. $\square$
:::

---

(sec-ig-edges)=
## 4. IG Edges: Asymmetric Selection Coupling

### 4.1 Edge Set Definition

:::{prf:definition} IG Edge Set
:label: def-fractal-set-ig-edges

Let $\mathcal{P}_t$ be the set of **ordered companion pairs** realized at timestep $t$ by the
companion selection operators (distance and cloning; {prf:ref}`def-fractal-set-companion-kernel`).
The **Information Graph (IG) edge set** is:

$$E_{\mathrm{IG}} := \{(n_{i,t}, n_{j,t}) : (i, j) \in \mathcal{P}_t, \; t \in \{0, \ldots, T\}\}.$$
Each IG edge connects an **ordered** sampled pair of distinct alive walkers at the same timestep.
The edges are **directed**: by convention, $(n_{i,t}, n_{j,t})$ is oriented from the influenced
walker $i$ toward the influencer $j$. Edges at $t = T$ are terminal-time snapshots and do not
participate in IA edges or triangles.
:::

:::{prf:proposition} IG Edge Cardinality
:label: prop-fractal-set-ig-cardinality

At timestep $t$ with $k_t = |\mathcal{A}(t)|$ alive walkers, the number of IG edges is
$m_t := |\mathcal{P}_t|$. The total across all timesteps is:

$$|E_{\mathrm{IG}}| = \sum_{t=0}^{T} m_t.$$

*Proof.* Each sampled ordered pair $(i, j) \in \mathcal{P}_t$ contributes one edge. $\square$
:::

For the sequential greedy pairing operator ({prf:ref}`def-greedy-pairing-algorithm`),
$m_t = k_t - f_t$ with $f_t \in \{0,1\}$ fixed points. In general, the IG edges at each timestep
form a **directed companion graph** on the alive walkers, restricted to the sampled pairs; the
influence is asymmetric, so the edge $(i, j)$ need not imply $(j, i)$ unless the pairing is mutual.

### 4.2 Directionality and Antisymmetry

The key structural property of IG edges is the **antisymmetry** of the cloning potential.

:::{prf:definition} Directed Cloning Potential
:label: def-fractal-set-cloning-potential

The **directed cloning potential** from walker $i$ to walker $j$ at timestep $t$ is:

$$V_{\mathrm{clone}}(i \to j; t) := \Phi(n_{j,t}) - \Phi(n_{i,t}) = \Phi_j(t) - \Phi_i(t),$$
where $\Phi_i(t) := \Phi(n_{i,t})$ is the fitness of walker $i$ at time $t$.
:::

:::{prf:proposition} Antisymmetry of Cloning Potential
:label: prop-fractal-set-antisymmetry

The cloning potential is **antisymmetric** under exchange of walkers:

$$V_{\mathrm{clone}}(j \to i; t) = -V_{\mathrm{clone}}(i \to j; t).$$

*Proof.* Direct computation:

$$V_{\mathrm{clone}}(j \to i; t) = \Phi_i(t) - \Phi_j(t) = -(\Phi_j(t) - \Phi_i(t)) = -V_{\mathrm{clone}}(i \to j; t). \quad \square$$
:::

This antisymmetry has profound implications:

:::{prf:corollary} Selection Asymmetry
:label: cor-fractal-set-selection-asymmetry

If $\Phi_j(t) > \Phi_i(t)$ (walker $j$ is fitter than walker $i$), then:
- $V_{\mathrm{clone}}(i \to j; t) > 0$: Walker $i$ is "pulled toward" walker $j$ (wants to clone from $j$)
- $V_{\mathrm{clone}}(j \to i; t) < 0$: Walker $j$ is "pushed away" from walker $i$ (does not want to clone from $i$)

The cloning potential biases flow from less fit to more fit; realized cloning events can still be stochastic or constrained by other rules.
:::

:::{div} feynman-prose
The antisymmetry is the discrete seed of quantum statistics. In quantum mechanics, fermions obey the Pauli exclusion principle: two identical fermions cannot occupy the same state. This shows up mathematically as antisymmetry under particle exchange—if you swap two fermions, the wave function picks up a minus sign.

Here, we have the same structure at the algorithmic level. The cloning potential $V_{\mathrm{clone}}(i \to j)$ flips sign when you exchange $i$ and $j$. This means the "flow" of cloning is inherently biased: fitness differences create a preferred direction from low to high. This asymmetry is what makes selection work—it creates a directed tendency toward fitter configurations.

We are not claiming the algorithm *is* a quantum system. But we are observing that the mathematical structure of selection—antisymmetry under exchange—is the same structure that underlies fermionic quantum field theory. This is not coincidence; it reflects a deep connection between optimization (flowing toward better states) and the antisymmetric structures that appear in physics.
:::

### 4.3 IG Edge Spinor Attributes

:::{prf:definition} IG Edge Spinor Attributes
:label: def-fractal-set-ig-attributes

Each IG edge $e = (n_{i,t}, n_{j,t}) \in E_{\mathrm{IG}}$ carries the following attributes:

**Identity attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Source walker | $i(e)$ | $\mathbb{Z}_+$ | [count] | Walker being influenced |
| Target walker | $j(e)$ | $\mathbb{Z}_+$ | [count] | Walker exerting influence |
| Timestep | $t(e)$ | $\mathbb{Z}_{\geq 0}$ | [count] | Timestep of interaction |
| Edge ID | $\mathrm{eid}(e)$ | $\mathbb{Z}_+$ | [count] | Unique edge identifier |

**Position spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Source position | $\psi_{x_i}(e)$ | $\mathbb{S}_d$ | [distance] | Spinor of $x_i(t)$ |
| Target position | $\psi_{x_j}(e)$ | $\mathbb{S}_d$ | [distance] | Spinor of $x_j(t)$ |
| Relative position | $\psi_{\Delta x_{ij}}(e)$ | $\mathbb{S}_d$ | [distance] | Spinor of $x_j - x_i$ |

**Velocity spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Source velocity | $\psi_{v_i}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $v_i(t)$ |
| Target velocity | $\psi_{v_j}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $v_j(t)$ |
| Relative velocity | $\psi_{\Delta v_{ij}}(e)$ | $\mathbb{S}_d$ | [distance/time] | Spinor of $v_j - v_i$ |

**Coupling spinors:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Viscous coupling | $\psi_{\mathrm{viscous}, ij}(e)$ | $\mathbb{S}_d$ | [distance/time^2] | Spinor of $\nu K_\rho(x_i, x_j)(v_j - v_i)$ |

**Scalar attributes:**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Kernel weight | $K_\rho(e)$ | $\mathbb{R}_{\geq 0}$ | [dimensionless] | $K_\rho(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / 2\rho^2)$ |
| Normalized weight | $w_{ij}(e)$ | $\mathbb{R}_{\geq 0}$ | [probability] | $w_{ij} = K_\rho(e) / \sum_{l \in \mathcal{A}(t) \setminus \{i\}} K_\rho(x_i, x_l)$ |
| Euclidean distance | $d_{ij}(e)$ | $\mathbb{R}_{\geq 0}$ | [distance] | $\|x_i - x_j\|$ |
| Algorithmic distance | $d_{\mathrm{alg}, ij}(e)$ | $\mathbb{R}_{\geq 0}$ | [distance] | $\sqrt{\|x_i - x_j\|^2 + \lambda_{\mathrm{alg}}\|v_i - v_j\|^2}$ |
| Phase potential | $\theta_{ij}(e)$ | $\mathbb{R}$ | [dimensionless] | $-(\Phi_j - \Phi_i)/\hbar_{\mathrm{eff}}$ |
| Source fitness | $\Phi_i(e)$ | $\mathbb{R}_{\geq 0}$ | [dimensionless] | $\Phi(x_i)$ |
| Target fitness | $\Phi_j(e)$ | $\mathbb{R}_{\geq 0}$ | [dimensionless] | $\Phi(x_j)$ |
| **Cloning potential** | $V_{\mathrm{clone}}(e)$ | $\mathbb{R}$ | [dimensionless] | $\Phi_j - \Phi_i$ (antisymmetric) |

**Complex amplitude (optional):**

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Coupling amplitude | $\psi_{ij}(e)$ | $\mathbb{C}$ | [probability^{1/2}] | $\sqrt{P_{\mathrm{comp}}(i,j)} \cdot e^{i\theta_{ij}}$ |
:::

### 4.4 Viscous Coupling Representation

:::{div} feynman-prose
Now let us talk about viscosity. In a real gas, molecules that are close together tend to drag each other along—fast molecules slow down, slow molecules speed up, and the gas develops a kind of internal friction. This is viscosity: the tendency of nearby particles to share momentum.

The Fractal Gas has the same effect, but it is engineered rather than emergent. When walker $i$ is near walker $j$, they exchange momentum through a viscous coupling force. The force is proportional to the *velocity difference* $(v_j - v_i)$—if $j$ is moving faster, it pulls $i$ along; if $j$ is slower, it drags $i$ back.

The strength of this coupling falls off with distance through the kernel $K_\rho(x_i, x_j) = \exp(-\|x_i - x_j\|^2 / 2\rho^2)$. Nearby walkers couple strongly; distant walkers barely feel each other. This is how the algorithm creates local coherence without global rigidity.

Why do we want this? Because optimization benefits from coherent exploration. If all walkers in a region are moving in roughly the same direction, they can explore that direction efficiently. Without viscosity, walkers would scatter chaotically and waste computational effort. With viscosity, they form something like a flock of birds—individually free, but collectively coordinated.
:::

The viscous force creates momentum exchange between nearby walkers.

:::{prf:definition} Pairwise Viscous Force
:label: def-fractal-set-viscous-force

The **pairwise viscous force** exerted by walker $j$ on walker $i$ is:

$$\mathbf{F}_{\mathrm{viscous}, ij} := \nu K_\rho(x_i, x_j)(v_j - v_i),$$

where $K_\rho(x, y) := \exp(-\|x - y\|^2 / 2\rho^2)$ is the Gaussian kernel with bandwidth $\rho$.

The total viscous force on walker $i$ is:

$$\mathbf{F}_{\mathrm{viscous}}(x_i, S) = \sum_{j \in \mathcal{A}(t) \setminus \{i\}} \mathbf{F}_{\mathrm{viscous}, ij}.$$
:::

:::{prf:proposition} Viscous Force Reconstruction from IG Edges
:label: prop-fractal-set-viscous-reconstruction

Define the IG-restricted viscous interaction for walker $i$ at timestep $t$:

$$\mathbf{F}_{\mathrm{viscous}}^{\mathrm{IG}}(x_i, S, t) = \sum_{e \in E_{\mathrm{IG}}: i(e) = i, t(e) = t} \pi(\psi_{\mathrm{viscous}, ij}(e)).$$

If the viscous coupling is evaluated on the sampled companion graph, then
$\mathbf{F}_{\mathrm{viscous}}^{\mathrm{IG}} = \mathbf{F}_{\mathrm{viscous}}$. In the full-kernel
variant of {prf:ref}`def-fractal-set-viscous-force`, the total force is recomputed directly from
node data using the kernel definition, and the applied total is stored on the CST edge.

*Proof.* Each IG edge $(n_{i,t}, n_{j,t})$ stores the spinor $\psi_{\mathrm{viscous}, ij}$ of the
pairwise force for that sampled pair. Summing over IG edges with source $i$ yields the interaction
recorded on the sampled graph. $\square$
:::

### 4.5 Algorithmic Distance and Fitness Phase

The Latent Fractal Gas uses **algorithmic distance** in phase space to determine companion selection weights.

:::{prf:definition} Algorithmic Distance
:label: def-fractal-set-alg-distance

The **algorithmic distance** between walkers $i$ and $j$ is:

$$d_{\mathrm{alg}}(i, j)^2 := \|x_i - x_j\|^2 + \lambda_{\mathrm{alg}} \|v_i - v_j\|^2,$$
where $\lambda_{\mathrm{alg}} \geq 0$ is a parameter weighting velocity similarity relative to position similarity.
:::

:::{prf:definition} Phase Potential
:label: def-fractal-set-phase-potential

The **phase potential** associated with the pair $(i, j)$ is the fitness phase difference:

$$
\theta_i := -\frac{\Phi_i}{\hbar_{\mathrm{eff}}}, \quad
\theta_{ij} := \theta_j - \theta_i = -\frac{\Phi_j - \Phi_i}{\hbar_{\mathrm{eff}}}.
$$

The additive fitness baseline $\Phi \to \Phi + c$ shifts $\theta_i$ by a constant and leaves $\theta_{ij}$ invariant, giving the $U(1)$ phase redundancy.
:::

The phase potential appears in the complex coupling amplitude
$\psi_{ij} = \sqrt{P_{\mathrm{comp}}(i,j)} \cdot e^{i\theta_{ij}}$, which encodes the magnitude (selection probability from algorithmic distance) and the phase (fitness difference) of the walker-walker interaction.

:::{div} feynman-prose
The algorithmic distance combines position and velocity into a single measure of "similarity" in phase space. Two walkers that are close in position but moving in opposite directions are less similar than two walkers that are equally close but moving together. This is sensible for optimization: walkers moving coherently toward a fitness peak should cooperate more than walkers that happen to be nearby but are exploring different directions.

The algorithmic distance controls the amplitude of interaction through $P_{\mathrm{comp}}$, while the fitness difference sets the phase. Close walkers interact strongly (large amplitude), and large fitness gaps generate rapid phase winding. This creates the possibility of wave-like interference in the walker ensemble—a feature that emerges naturally from the algorithm's structure without being explicitly designed.
:::

---

(sec-fractal-set)=
## 5. The Fractal Set: Complete Data Structure

### 5.1 Formal Definition

:::{prf:definition} The Fractal Set
:label: def-fractal-set-complete

The **Fractal Set** generated by a run of the Fractal Gas algorithm with $N$ walkers for $T$ timesteps is a **directed 2-complex** with simplicial support:

$$\mathcal{F} := (\mathcal{N}, E_{\mathrm{CST}} \cup E_{\mathrm{IG}} \cup E_{\mathrm{IA}}, \mathcal{T}, \boldsymbol{\omega}, \mathcal{D})$$
where:

**Simplicial structure (undirected support):**
- **$\mathcal{N}$**: Node set (0-simplices) — {prf:ref}`def-fractal-set-node`
- **$E_{\mathrm{CST}}$**: CST edge set (1-simplices) — {prf:ref}`def-fractal-set-cst-edges`
- **$E_{\mathrm{IG}}$**: IG edge set (1-simplices) — {prf:ref}`def-fractal-set-ig-edges`
- **$E_{\mathrm{IA}}$**: IA edge set (1-simplices) — {prf:ref}`def-fractal-set-ia-edges`
- **$\mathcal{T}$**: Interaction triangles (2-simplices) — {prf:ref}`def-fractal-set-triangle`

We attach asymmetric data to **oriented** edges; $E_{\mathrm{IG}}$ and $E_{\mathrm{IA}}$ should be read as sets of oriented edges whose undirected supports are the 1-simplices of the complex.

**Weight functions** $\boldsymbol{\omega} = (\omega_{\mathrm{CST}}, \omega_{\mathrm{IG}}, \omega_{\mathrm{IA}})$:
- $\omega_{\mathrm{CST}}: E_{\mathrm{CST}} \to \mathbb{R}_{>0}$ — timestep duration $\Delta t$
- $\omega_{\mathrm{IG}}: E_{\mathrm{IG}} \to \mathbb{R}$ — cloning potential $V_{\mathrm{clone}}(i \to j)$
- $\omega_{\mathrm{IA}}: E_{\mathrm{IA}} \to [0,1]$ — influence attribution weight $w_{ij}$

**Attribute data** $\mathcal{D} = (\mathcal{D}_{\mathcal{N}}, \mathcal{D}_{\mathrm{CST}}, \mathcal{D}_{\mathrm{IG}}, \mathcal{D}_{\mathrm{IA}})$:
- **$\mathcal{D}_{\mathcal{N}}$**: Node attributes — {prf:ref}`def-fractal-set-node-attributes`
- **$\mathcal{D}_{\mathrm{CST}}$**: CST edge attributes — {prf:ref}`def-fractal-set-cst-attributes`
- **$\mathcal{D}_{\mathrm{IG}}$**: IG edge attributes — {prf:ref}`def-fractal-set-ig-attributes`
- **$\mathcal{D}_{\mathrm{IA}}$**: IA edge attributes — {prf:ref}`def-fractal-set-ia-attributes`
:::

:::{div} feynman-prose
The Fractal Set is the algorithm's autobiography, written in coordinate-free language. Every decision, every force, every random fluctuation is recorded. A future observer—perhaps using entirely different coordinates, perhaps running on different hardware—can read this autobiography and reconstruct exactly what happened.

This is not just data logging. Ordinary logs record numbers in some arbitrary coordinate system. To interpret them, you need to know what coordinate system was used. The Fractal Set records geometric objects. To interpret them, you only need to know the laws of spinor algebra, which are universal.
:::

### 5.2 Covariance Structure

:::{prf:theorem} Frame-Invariance of Scalar Data
:label: thm-fractal-set-scalar-invariance

All scalar attributes stored in $\mathcal{D}_{\mathcal{N}}$, $\mathcal{D}_{\mathrm{CST}}$, $\mathcal{D}_{\mathrm{IG}}$, and $\mathcal{D}_{\mathrm{IA}}$ are frame-invariant in the sense of {prf:ref}`def-fractal-set-scalar`.

*Proof.* By construction:
- Node scalars: Established in {prf:ref}`prop-fractal-set-node-invariance`.
- CST edge scalars ($\|\Delta v\|$, $\|\Delta x\|$, $\Delta t$): Norms and time intervals are frame-invariant.
- IG edge scalars ($K_\rho$, $w_{ij}$, $d_{ij}$, $d_{\mathrm{alg}, ij}$, $\theta_{ij}$, $\Phi_i$, $\Phi_j$, $V_{\mathrm{clone}}$): Distances and kernel weights are functions of norms; $\Phi_i$, $\Phi_j$ are scalar field evaluations; $V_{\mathrm{clone}}$ is a difference of scalars.
- IA edge scalars ($w_{\mathrm{IA}}$, $\chi_{\mathrm{clone}}$, $\phi_{\mathrm{IA}}$): Scalar weights, indicators, and phases. $\square$
:::

:::{prf:theorem} Frame-Covariance of Spinor Data
:label: thm-fractal-set-spinor-covariance

All spinor attributes stored in $\mathcal{D}_{\mathrm{CST}}$ and $\mathcal{D}_{\mathrm{IG}}$ transform covariantly under $\mathrm{SO}(d)$: if the coordinate system is rotated by $R$, and $U \in \mathrm{Spin}(d)$ is a lift of $R$, then each spinor $\psi$ transforms as $\psi \mapsto U\psi$.

*Proof.* Each spinor is constructed via the vector-to-spinor map $\iota$ from a vector field. By {prf:ref}`prop-fractal-set-spinor-covariance`, spinors transform by the lift $U$ and satisfy $\pi(U\psi) = R\pi(\psi)$, which is the required covariance. $\square$
:::

:::{prf:corollary} Coordinate-Free Reconstruction
:label: cor-fractal-set-coordinate-free

Two observers using coordinate systems related by $R \in \mathrm{SO}(d)$ can independently reconstruct all vector quantities from the Fractal Set. Their reconstructions are related by $R$:

$$\mathbf{v}^{(2)} = R \mathbf{v}^{(1)}.$$

*Proof.* Observer 1 computes $\mathbf{v}^{(1)} = \pi^{(1)}(\psi)$. Observer 2 computes $\mathbf{v}^{(2)} = \pi^{(2)}(\psi)$. Since the spinor-to-vector maps are related by $\pi^{(2)} = R \circ \pi^{(1)}$ (the spinor representation intertwines), we have $\mathbf{v}^{(2)} = R\mathbf{v}^{(1)}$. $\square$
:::

### 5.3 Influence Attribution Edges

The third edge type completes the causal structure by connecting effects to their causes.

:::{div} feynman-prose
Now here is the edge type that closes the loop. CST edges tell you how a walker evolved. IG edges tell you who was around to influence it. But neither tells you *which* influences actually mattered.

Think about it this way: walker $i$ at time $t+1$ is a different creature from walker $i$ at time $t$. It has moved, changed velocity, maybe even cloned. Something caused that change. The IA edge points backward in time, from the effect to the cause: "You, walker $i$ at $t+1$, are partly the result of what walker $j$ was doing at time $t$."

This is attribution, not just correlation. The IG edge says "$j$ and $i$ were neighbors." The IA edge says "$j$ actually contributed to what $i$ became." The weight on the IA edge tells you how much: was $j$ the main influence, or just one voice in a chorus?

The direction is deliberately "retrocausal"—from later to earlier time. We are not saying the future causes the past. We are saying: to understand the present, trace it back to its sources. The IA edge is an accounting device, a ledger entry that says "credit this much of $i$'s change to $j$."
:::

:::{prf:definition} Influence Attribution Edge Set
:label: def-fractal-set-ia-edges

The **Influence Attribution (IA) edge set** is:

$$E_{\mathrm{IA}} := \{(n_{i,t+1}, n_{j,t}) : (i, j) \in \mathcal{P}_t, \; t \in \{0, \ldots, T-1\}\}.$$
Each IA edge connects the **effect** (walker $i$ at time $t+1$) to a **cause** (walker $j$ at
time $t$) for a sampled pair. The direction is **retrocausal**: from later to earlier time,
attributing the outcome to its source.
:::

:::{prf:definition} IA Edge Attributes
:label: def-fractal-set-ia-attributes

Each IA edge $e = (n_{i,t+1}, n_{j,t}) \in E_{\mathrm{IA}}$ carries the following scalar attributes:

| Attribute | Symbol | Type | Unit | Description |
|-----------|--------|------|------|-------------|
| Influence weight | $w_{\mathrm{IA}}(e)$ | $[0, 1]$ | [probability] | Fraction of $i$'s update attributable to $j$: $w_{ij}(t)$ |
| Clone indicator | $\chi_{\mathrm{clone}}(e)$ | $\{0, 1\}$ | [boolean] | 1 if $c(n_{i,t+1}) = j$ (cloned from $j$), else 0 |
| Phase contribution | $\phi_{\mathrm{IA}}(e)$ | $\mathbb{R}$ | [dimensionless] | Phase accumulated on attribution edge |

For **viscous coupling**, $w_{\mathrm{IA}}(e) = K_\rho(x_i, x_j) / \sum_{l \in \mathcal{A}(t) \setminus \{i\}} K_\rho(x_i, x_l)$.

For **cloning**, $w_{\mathrm{IA}}(e) = 1$ if $\chi_{\mathrm{clone}}(e) = 1$, else 0.
:::

:::{prf:proposition} IA Edge Cardinality
:label: prop-fractal-set-ia-cardinality

Let $E_{\mathrm{IG}}^{<T} := \{(n_{i,t}, n_{j,t}) \in E_{\mathrm{IG}} : t \in \{0, \ldots, T-1\}\}$. The IA edge cardinality equals the IG edge cardinality on update timesteps:

$$|E_{\mathrm{IA}}| = |E_{\mathrm{IG}}^{<T}| = \sum_{t=0}^{T-1} m_t.$$

*Proof.* At each $t \in \{0, \ldots, T-1\}$, there is one IA edge for each sampled ordered pair
$(i, j) \in \mathcal{P}_t$, matching the IG edges at the same timestep. $\square$
:::

### 5.4 Interaction Triangles: The Fundamental 2-Simplices

:::{prf:definition} Interaction Triangle
:label: def-fractal-set-triangle

An **interaction triangle** $\triangle_{ij,t}$ is the 2-simplex with:

**Vertices** (0-faces):

$$V(\triangle_{ij,t}) = \{n_{j,t}, n_{i,t}, n_{i,t+1}\}$$

**Edges** (1-faces), forming the **boundary** $\partial\triangle_{ij,t}$:
- $e_{\mathrm{IG}} = (n_{i,t}, n_{j,t}) \in E_{\mathrm{IG}}$: "walker $j$ influences walker $i$"
- $e_{\mathrm{CST}} = (n_{i,t}, n_{i,t+1}) \in E_{\mathrm{CST}}$: "walker $i$ evolves"
- $e_{\mathrm{IA}} = (n_{i,t+1}, n_{j,t}) \in E_{\mathrm{IA}}$: "attribute $i$'s update to $j$"

**Orientation convention**: We orient $\triangle_{ij,t}$ as the ordered simplex $(n_{i,t}, n_{i,t+1}, n_{j,t})$, so

$$\partial \triangle_{ij,t} = e_{\mathrm{CST}} + e_{\mathrm{IA}} - e_{\mathrm{IG}}.$$

Equivalently, the boundary path is $n_{i,t} \to n_{i,t+1}$ (CST), $n_{i,t+1} \to n_{j,t}$ (IA), and $n_{j,t} \to n_{i,t}$ (IG with reversed orientation).

The **triangle set** is:

$$\mathcal{T} := \{\triangle_{ij,t} : (i, j) \in \mathcal{P}_t, \; t \in \{0, \ldots, T-1\}\}.$$
:::

:::{prf:proposition} Triangle Cardinality
:label: prop-fractal-set-triangle-cardinality

The number of interaction triangles equals the number of IG edges at each update:

$$|\mathcal{T}| = |E_{\mathrm{IG}}^{<T}| = |E_{\mathrm{IA}}| = \sum_{t=0}^{T-1} m_t.$$

At each update timestep $t \in \{0, \ldots, T-1\}$, there is one triangle for each sampled
ordered pair $(i, j) \in \mathcal{P}_t$.

*Proof.* Each triangle $\triangle_{ij,t}$ is uniquely determined by the ordered pair $(i, j)$ and
timestep $t$, in bijection with IG edges at the same timestep. $\square$
:::

:::{div} feynman-prose
The triangle is the atom of interaction. You cannot break it into smaller pieces without losing causality.

Consider what happens when walker $j$ influences walker $i$: at time $t$, both walkers exist with their own positions and velocities. Walker $i$ "sees" walker $j$ through the IG edge—this is the influence channel. Then $i$ evolves from $t$ to $t+1$ along its CST edge—this is the evolution channel. Finally, the IA edge closes the loop by recording "this evolution was partly due to $j$"—this is the attribution channel.

Without the IG edge, we wouldn't know who influenced whom. Without the CST edge, we wouldn't know what changed. Without the IA edge, we couldn't close the causal loop.

The triangle is irreducible. It's the smallest closed loop in the structure. And it corresponds exactly to the physical process: one walker influencing another's evolution.
:::

### 5.5 Plaquettes as Parallel Transport

:::{prf:definition} Plaquette
:label: def-fractal-set-plaquette

A **plaquette** $P_{ij,t}$ is the simplicial 2-chain formed by two adjacent interaction triangles,
defined when **both orientations** are present (i.e., $(i, j) \in \mathcal{P}_t$ and
$(j, i) \in \mathcal{P}_t$):

$$P_{ij,t} = \triangle_{ij,t} \cup \triangle_{ji,t}$$
where:
- $\triangle_{ij,t}$ has vertices $\{n_{j,t}, n_{i,t}, n_{i,t+1}\}$ ("$j$ influences $i$")
- $\triangle_{ji,t}$ has vertices $\{n_{i,t}, n_{j,t}, n_{j,t+1}\}$ ("$i$ influences $j$")

The two triangles share the **undirected IG edge at time $t$**, with opposite orientations.
:::

:::{prf:proposition} Plaquette Decomposition
:label: prop-fractal-set-plaquette-decomposition

The boundary of a plaquette is a 4-cycle formed by the non-shared edges:

$$\partial P_{ij,t} = \partial\triangle_{ij,t} + \partial\triangle_{ji,t}$$
where the shared IG edge appears in both triangles with opposite orientation and cancels.

*Proof.* The boundary of $P_{ij,t}$ consists of four edges forming a closed 4-cycle:
- $(n_{i,t}, n_{i,t+1})$: CST for walker $i$
- $(n_{i,t+1}, n_{j,t})$: IA edge (back-diagonal)
- $(n_{j,t}, n_{j,t+1})$: CST for walker $j$
- $(n_{j,t+1}, n_{i,t})$: IA edge (back-diagonal)

The shared IG edge $(n_{i,t}, n_{j,t})$ appears with opposite orientation in each triangle and cancels. The result is a "hourglass" 4-cycle connecting time $t$ to time $t+1$ via two CST edges and two IA back-edges. $\square$
:::

:::{div} feynman-prose
What does a plaquette measure? It measures **parallel transport inconsistency**—whether the two walkers' evolutions "agree" with each other.

Let me draw the hourglass shape in your mind. At time $t$, you have two walkers, $i$ and $j$, connected by an IG edge—that is the *waist* of the hourglass, the narrow crossing point. Above and below the waist, the hourglass bulges out:

```
          n_{i,t+1}      n_{j,t+1}
             | \        / |
          CST|  \  IA  /  |CST
             |   \    /   |
             |    \  /    |
             |     \/     |
             |     /\     |
             |    /  \    |
          n_{i,t} ---IG--- n_{j,t}
```

The boundary of the plaquette traces the outer edge of this hourglass: up the left side (CST for $i$), along the IA back-edge to $n_{j,t}$, up the right side (CST for $j$), and back along the other IA edge to $n_{i,t}$. The IG edge at the waist is *internal*—it belongs to both triangles but cancels when you trace the outer boundary.

Now imagine transporting some quantity around this hourglass boundary. Start at $n_{i,t}$, follow $i$'s evolution forward to $n_{i,t+1}$, then follow the IA back-edge to $n_{j,t}$ (attributing $i$'s change to $j$), then follow $j$'s evolution forward to $n_{j,t+1}$, and finally return via the IA back-edge to $n_{i,t}$ (attributing $j$'s change to $i$).

If $i$'s evolution (influenced by $j$) and $j$'s evolution (influenced by $i$) are mutually consistent, the parallel transport returns you to where you started. If they are inconsistent, you accumulate a nontrivial holonomy—the mathematical signature of "curvature" in the interaction structure.

But here is the key insight: the plaquette is not fundamental. It is made of two triangles sharing their IG edge. The triangles are the atoms; plaquettes are molecules. When we compute Wilson loops on plaquettes, we are really multiplying Wilson loops on triangles.
:::

### 5.6 Simplicial Complex Structure

:::{prf:theorem} Fractal Set as a Directed 2-Complex
:label: thm-fractal-set-simplicial

Let $\bar{E}$ be the **undirected supports** of $E_{\mathrm{CST}}$, $E_{\mathrm{IG}}$, and $E_{\mathrm{IA}}$ (identify opposite orientations). Then $(\mathcal{N}, \bar{E}, \mathcal{T})$ is a **2-dimensional simplicial complex**, and the oriented edge sets equip it with a directed 1-skeleton and asymmetric edge data.

The complex satisfies the **closure property**: every face of a simplex is also in the complex.

*Proof.*
- Every edge in $\bar{E}$ has both endpoints in $\mathcal{N}$ by definition.
- Every triangle $\triangle_{ij,t} \in \mathcal{T}$ has its three vertices in $\mathcal{N}$ and its three boundary edges in $\bar{E}$ by construction.
- The boundary operator $\partial_2: \mathcal{T} \to \mathbb{Z}[E]$ is well-defined on oriented edges:

$$\partial_2 \triangle_{ij,t} = e_{\mathrm{CST}} + e_{\mathrm{IA}} - e_{\mathrm{IG}}$$
(consistent with the orientation convention in {prf:ref}`def-fractal-set-triangle`). $\square$
:::

:::{prf:corollary} Euler Characteristic
:label: cor-fractal-set-euler

For a single timestep $t$ with $k = k_t$ alive walkers, let $\mathcal{P}_t$ be the sampled ordered
pairs and let $\overline{\mathcal{P}}_t$ be their undirected support (unordered pairs). The local
Euler characteristic of the $(t, t+1)$ simplicial slice is:

$$\chi_t = |V_t| - |E_t| + |F_t| = 2k - \left(k + |\overline{\mathcal{P}}_t| + |\mathcal{P}_t|\right) + |\mathcal{P}_t| = k - |\overline{\mathcal{P}}_t|.$$

For the sequential greedy pairing operator ({prf:ref}`def-greedy-pairing-algorithm`),
$|\overline{\mathcal{P}}_t| = (k - f_t)/2$ with $f_t \in \{0,1\}$ fixed points, so
$\chi_t = (k + f_t)/2$.

*Proof.* Vertices: $2k$ (walkers at $t$ and $t+1$). Edges: $k$ (CST) + $|\overline{\mathcal{P}}_t|$
(undirected IG at $t$) + $|\mathcal{P}_t|$ (IA). Faces: $|\mathcal{P}_t|$ triangles. The formula
follows by substitution. $\square$
:::

:::{div} feynman-prose
What does the Euler characteristic tell us? It summarizes how densely the sampled companion pairs
stitch each $(t, t+1)$ slice together.

With sampled companion pairs, the slice topology depends on how many unordered pairs are realized.
For pairing-based companions, each timestep slice decomposes into disjoint hourglasses (one per
matched pair), so $\chi_t$ stays positive; the complexity of the Fractal Set then comes from how
pairings change over time and how CST worldlines weave these triangles together.

The fact that the Fractal Set forms a genuine 2-dimensional 2-complex with simplicial support—not
just a graph—means we can apply the entire machinery of algebraic topology: homology groups, Betti
numbers, and cohomological field theories. The triangles are not decorations; they are load-bearing
mathematical structure.
:::

### 5.7 Wilson Loops on Interaction Triangles

:::{prf:definition} Gauge Connection on Edges
:label: def-fractal-set-gauge-connection

A **gauge connection** on the Fractal Set assigns to each oriented edge $e$ a **parallel transport phase** $U_e \in \mathrm{U}(1)$. For the phase connection, we use:
- $U_{\mathrm{IG}}(e) = e^{i\theta_{ij}}$ where $\theta_{ij}$ is the phase potential from {prf:ref}`def-fractal-set-phase-potential`
- $U_{\mathrm{CST}}(e) = e^{i\phi_{\mathrm{CST}}}$ where $\phi_{\mathrm{CST}}$ is the phase accumulated during evolution
- $U_{\mathrm{IA}}(e) = e^{i\phi_{\mathrm{IA}}}$ where $\phi_{\mathrm{IA}}$ is the attribution phase

Orientation reversal conjugates: if $U_e = e^{i\phi_e}$, then $U_{-e} = U_e^* = e^{-i\phi_e}$.
:::

:::{prf:definition} Wilson Loop on a Triangle
:label: def-fractal-set-wilson-loop

The **Wilson loop** around an interaction triangle $\triangle_{ij,t}$ is the **holonomy** of the gauge connection:

$$W(\triangle_{ij,t}) := U_{\mathrm{CST}}(e_{\mathrm{CST}}) \cdot U_{\mathrm{IA}}(e_{\mathrm{IA}}) \cdot U_{\mathrm{IG}}(e_{\mathrm{IG}})^* = e^{i(\phi_{\mathrm{CST}} + \phi_{\mathrm{IA}} - \theta_{ij})}.$$

This is the phase holonomy around the triangle boundary.
:::

:::{prf:proposition} Plaquette Wilson Loop Factorization
:label: prop-fractal-set-wilson-factorization

The plaquette holonomy factorizes into triangle holonomies:

$$W(P_{ij,t}) = W(\triangle_{ij,t}) \cdot W(\triangle_{ji,t})$$

*Proof.* The plaquette boundary $\partial P_{ij,t}$ equals $\partial\triangle_{ij,t} + \partial\triangle_{ji,t}$, with the shared IG edge canceling due to opposite orientations. The holonomy around $\partial P$ is the product of the triangle holonomies, and the shared IG edge contributes $U_{\mathrm{IG}} \cdot U_{\mathrm{IG}}^* = 1$. $\square$
:::

:::{div} feynman-prose
What does the Wilson loop around a triangle measure? It measures the **quantum phase accumulated during one complete interaction**.

The Wilson loop combines three phases from the three edges of the triangle boundary:
- **CST phase** $\phi_{\mathrm{CST}}$: the phase accumulated during $i$'s evolution
- **IA phase** $\phi_{\mathrm{IA}}$: the attribution phase on the IA edge
- **IG phase** $-\theta_{ij}$: the IG edge is traversed with opposite orientation

The total holonomy $\Phi(\triangle) = \phi_{\mathrm{CST}} + \phi_{\mathrm{IA}} - \theta_{ij}$ is the **interaction phase**. Now, what does it *mean* for this phase to be zero or nonzero?

A **flat interaction** ($\Phi(\triangle) = 0$) means the phases balance perfectly: the evolution-plus-attribution phase equals the influence phase ($\phi_{\mathrm{CST}} + \phi_{\mathrm{IA}} = \theta_{ij}$). The interaction was *self-consistent*—what $j$ put in, $i$ got out, with no residue.

A **curved interaction** ($\Phi(\triangle) \neq 0$) means something was left over. Perhaps the influence was stronger than the attribution acknowledges, or the evolution picked up extra phase from the environment. The nonzero holonomy is a signature that *something happened* in this interaction beyond the simple story of "$j$ influenced $i$."

In lattice gauge theory, the Wilson loop around a plaquette measures field strength—the curvature of the gauge connection. Here, the Wilson loop around a triangle measures **interaction strength**—the curvature of the causal structure. Regions of the Fractal Set where many triangles have large holonomies are regions of *intense interaction*—where walkers are strongly influencing each other, and the simple additive picture of "my change equals the sum of my influences" breaks down.

The fundamental objects are triangles, not plaquettes. Plaquettes are pairs of triangles sharing an IG edge. This is why 3-cycles are the right minimal loops for defining gauge-invariant observables on the Fractal Set.
:::

:::{note}
:class: feynman-added

**Why triangles instead of plaquettes?** In standard lattice gauge theory on a hypercubic lattice, the minimal closed loops are 4-cycles (plaquettes) because the lattice has only coordinate-aligned edges. The Fractal Set has a richer structure: three edge types with different causal roles. The minimal closed loop that involves all three edge types is a 3-cycle (triangle), not a 4-cycle.

This is not a choice—it is forced by the causal structure. An IG edge (spacelike) connects two walkers at the *same* time. A CST edge (timelike) connects the *same* walker at different times. An IA edge (diagonal) connects a walker's future to another walker's past. The smallest loop that uses one of each is a triangle.

Plaquettes appear when you ask: "What happens when two walkers mutually influence each other?" That requires two triangles—one for each direction of influence—sharing their IG edge.
:::

### 5.8 Memory Complexity

:::{prf:proposition} Fractal Set Memory Complexity
:label: prop-fractal-set-memory

For $N$ walkers, $T$ timesteps, average alive walkers $k$, sampled-pair counts
$m_t := |\mathcal{P}_t|$, and state dimension $d$ (with $s_d = \dim_{\mathbb{C}}\mathbb{S}_d$):
let $M := \sum_{t=0}^{T} m_t$.

| Component | Count | Size per Element | Total Size |
|-----------|-------|------------------|------------|
| Nodes | $N(T+1)$ | $O(1)$ scalars | $O(NT)$ |
| CST edges | $O(NT)$ | $O(s_d)$ spinors + $O(1)$ scalars | $O(NT \cdot s_d)$ |
| IG edges | $O(M)$ | $O(s_d)$ spinors + $O(1)$ scalars | $O(M \cdot s_d)$ |
| IA edges | $O(M)$ | $O(1)$ scalars | $O(M)$ |
| Triangles | $O(M)$ | $O(1)$ pointers | $O(M)$ |

Total memory: $O(NT \cdot s_d + M \cdot s_d)$.

Note: IA edges and triangles add only $O(M)$ scalar storage—negligible compared to the spinor-heavy
IG edges.

For two-companion sampling, $M = O(Tk)$; if all pairs are materialized, $M = O(Tk^2)$ and the
dense bound is recovered.

*Proof.* Direct counting from the definitions. IA edges store only scalar weights (no spinors), and triangles store only pointers to their three boundary edges. $\square$
:::

The IG and IA edges dominate memory for large $N$. If you materialize additional IG edges (for
example, all pairs above a kernel threshold), sparsification can reduce storage to
$O(TNk_{\mathrm{eff}})$ where $k_{\mathrm{eff}}$ is the effective number of neighbors within the
kernel bandwidth. When sparsifying, triangles are also pruned: only triangles whose IG edge
survives the threshold are retained.

---

(sec-reconstruction)=
## 6. Reconstruction of Algorithm Dynamics

:::{div} feynman-prose
Now we come to the payoff. We have built this elaborate structure—nodes with scalars, edges with spinors, triangles closing causal loops—but what is it *for*? The answer is reconstruction: the ability to recover everything that happened, exactly, from the stored data.

Think of it like a perfect black box recorder for an airplane. After a flight, you want to know: where was the plane at each moment? What forces acted on it? What did the pilot do? A good recorder captures all of this. A perfect recorder captures it in a way that any investigator, using any coordinate system, can analyze.

The Fractal Set is a perfect recorder for the algorithm. Given the stored data, you can reconstruct:
- Every walker's trajectory through phase space
- Every force that acted on every walker at every moment
- The fitness landscape at every point any walker visited
- Which walkers cloned from which, and when
- The full empirical measure—the statistical distribution of the walker swarm

And you can do this without knowing what coordinate system was used to run the algorithm. The spinors carry the geometry; the reconstruction just extracts it.

This is not a theoretical nicety. It means the Fractal Set is a *portable* record. You can run the algorithm on one machine, store the Fractal Set, ship it to another continent, and reconstruct the dynamics perfectly. No coordinate translation, no basis alignment, no information loss.
:::

The central property of the Fractal Set is **lossless reconstruction**: all algorithm dynamics can be recovered from the stored data.

### 6.1 Reconstructible Quantities

:::{prf:definition} Reconstruction Target Set
:label: def-fractal-set-reconstruction-targets

The **reconstruction targets** are the quantities that characterize the Fractal Gas algorithm:

1. **Phase-space trajectories**: $(x_i(t), v_i(t))$ for all $i \in \{1, \ldots, N\}$, $t \in \{0, \ldots, T\}$
2. **Force fields**: $\mathbf{F}_{\mathrm{stable}}$, $\mathbf{F}_{\mathrm{adapt}}$, $\mathbf{F}_{\mathrm{viscous}}$, $\mathbf{F}_{\mathrm{friction}}$, $\mathbf{F}_{\mathrm{total}}$ at each walker position and time
3. **Diffusion tensor field**: $\Sigma_{\mathrm{reg}}(x, S, t)$ at each walker position and time
4. **Fitness landscape**: $\Phi(x)$ sampled at all walker positions
5. **Virtual reward field**: $V_{\mathrm{fit}}[f_k, \rho](x)$ sampled at all walker positions
6. **Localized statistics**: $\mu_\rho$, $\sigma_\rho$, $Z_\rho$ at all walker positions
7. **Population dynamics**: $\mathcal{A}(t)$, $k_t = |\mathcal{A}(t)|$ at all timesteps
8. **Empirical measure**: $f_k(t) = \frac{1}{k_t}\sum_{i \in \mathcal{A}(t)} \delta_{(x_i(t), v_i(t))}$ at all timesteps
9. **Cloning events**: Which walker cloned from which, at which timestep
:::

### 6.2 Phase Space Trajectory Reconstruction

:::{prf:theorem} Trajectory Reconstruction
:label: thm-fractal-set-trajectory

Given the Fractal Set $\mathcal{F}$, the complete phase-space trajectory $(x_i(t), v_i(t))$ for any walker $i$ can be reconstructed.

*Proof.*

**Velocity reconstruction**: For each node $n_{i,t}$, find the incoming CST edge $e = (n_{i,t-1}, n_{i,t})$ (if $t > 0$) or outgoing CST edge $e' = (n_{i,t}, n_{i,t+1})$ (if $t < T$ and $s(n_{i,t}) = 1$). Use the final-velocity spinor $\psi_{v,t}(e)$ from the incoming edge or the initial-velocity spinor $\psi_{v,t}(e')$ from the outgoing edge:

$$v_i(t) = \pi(\psi_{v,t}(e)) \quad \text{or} \quad v_i(t) = \pi(\psi_{v,t}(e')).$$

**Position reconstruction**: Recover $x_i(0)$ from any IG edge at $t=0$ incident to walker $i$, then accumulate displacements:

$$x_i(t) = x_i(0) + \sum_{s=0}^{t-1} \pi(\psi_{\Delta x}(e_s)),$$
where $e_s = (n_{i,s}, n_{i,s+1})$ is the CST edge at timestep $s$.

Alternatively, positions can be read directly at any $t$ from IG edge position spinors $\psi_{x_i}(e)$ incident to walker $i$, avoiding displacement accumulation. $\square$
:::

### 6.3 Force Field Reconstruction

:::{prf:theorem} Force Field Reconstruction
:label: thm-fractal-set-force

All force components at any walker position and time can be reconstructed from CST edge spinors.

*Proof.* Each CST edge $e = (n_{i,t}, n_{i,t+1})$ stores force spinors $\psi_{\mathbf{F}_\cdot}(e)$ for each force component. The reconstruction is:

$$\mathbf{F}_{\cdot}(x_i(t), S(t), t) = \pi(\psi_{\mathbf{F}_\cdot}(e)).$$
This gives force values at the sampled positions $\{x_i(t) : i \in \mathcal{A}(t)\}$. $\square$
:::

### 6.4 Diffusion Tensor Field Reconstruction

:::{prf:theorem} Diffusion Tensor Reconstruction
:label: thm-fractal-set-diffusion

The diffusion tensor $\Sigma_{\mathrm{reg}}(x, S, t)$ can be reconstructed at sampled positions from CST edge data.

*Proof.* Each CST edge stores the diffusion tensor spinor $\psi_{\Sigma_{\mathrm{reg}}}(e)$, which encodes the full tensor. Reconstruction uses the spinor-to-tensor extraction (extension of spinor-to-vector). $\square$
:::

### 6.5 Fitness and Reward Landscape Reconstruction

:::{prf:theorem} Landscape Reconstruction
:label: thm-fractal-set-landscape

The fitness $\Phi(x)$ and virtual reward $V_{\mathrm{fit}}(x)$ fields can be reconstructed at all sampled positions.

*Proof.* Node attributes directly store $\Phi(n)$ and $V_{\mathrm{fit}}(n)$. For node $n_{i,t}$:

$$\Phi(x_i(t)) = \Phi(n_{i,t}), \quad V_{\mathrm{fit}}(x_i(t)) = V_{\mathrm{fit}}(n_{i,t}).$$
This provides a sampling of the landscapes at walker-visited positions. $\square$
:::

### 6.6 Population Dynamics Reconstruction

:::{prf:theorem} Population Reconstruction
:label: thm-fractal-set-population

The alive walker set $\mathcal{A}(t)$ and empirical measure $f_k(t)$ can be reconstructed at all timesteps.

*Proof.*
**Alive set**: $\mathcal{A}(t) = \{i : s(n_{i,t}) = 1\}$ from node status flags.

**Empirical measure**: Using reconstructed trajectories,

$$f_k(t) = \frac{1}{k_t}\sum_{i \in \mathcal{A}(t)} \delta_{(x_i(t), v_i(t))}$$
where $(x_i(t), v_i(t))$ comes from {prf:ref}`thm-fractal-set-trajectory`. $\square$
:::

### 6.7 Cloning Event Reconstruction

:::{prf:theorem} Cloning Event Reconstruction
:label: thm-fractal-set-cloning

The complete cloning history—which walker cloned from which, at which timestep—can be reconstructed from node attributes.

*Proof.* Each node $n_{i,t}$ stores the **clone source** attribute $c(n_{i,t}) \in \mathbb{Z}_+ \cup \{\bot\}$ ({prf:ref}`def-fractal-set-node-attributes`). The cloning events are:

$$\mathcal{E}_{\mathrm{clone}} = \{(i, j, t) : c(n_{i,t}) = j \neq \bot\},$$
indicating walker $i$ cloned from walker $j$ at timestep $t$. The genealogical tree can be reconstructed by following clone source pointers backward in time. $\square$
:::

### 6.8 Main Reconstruction Theorem

:::{prf:theorem} Lossless Reconstruction
:label: thm-fractal-set-lossless

The Fractal Set $\mathcal{F}$ contains **complete information** to reconstruct all Fractal Gas dynamics at discrete timesteps, including the realized stochastic increments $\Sigma_{\mathrm{reg}} \circ dW_i$ stored on CST edges. The only missing information is interpolation between sampled positions (the landscapes are known only at walker-visited points).

Formally: given $\mathcal{F}$, one can reconstruct all quantities in {prf:ref}`def-fractal-set-reconstruction-targets` exactly for discrete-time values and at sampled positions.

*Proof.* Combine {prf:ref}`thm-fractal-set-trajectory` through {prf:ref}`thm-fractal-set-cloning`. Each target quantity is either directly stored (node/edge attributes) or reconstructable from stored spinors via the spinor-to-vector map:

1. **Phase-space trajectories**: {prf:ref}`thm-fractal-set-trajectory`
2. **Force fields**: {prf:ref}`thm-fractal-set-force`
3. **Diffusion tensor**: {prf:ref}`thm-fractal-set-diffusion`
4. **Fitness/reward landscapes**: {prf:ref}`thm-fractal-set-landscape`
5. **Population dynamics**: {prf:ref}`thm-fractal-set-population`
6. **Cloning events**: {prf:ref}`thm-fractal-set-cloning`

$\square$
:::

:::{prf:corollary} Frame-Independent Physics
:label: cor-fractal-set-physics

Any physical observable computed from the Fractal Gas dynamics—energy, work, entropy, convergence metrics—is recoverable from $\mathcal{F}$ and yields the same value regardless of which coordinate system the recovering observer uses.

*Proof.* Physical observables are scalar functions of the reconstructed trajectories and fields. Scalars are frame-invariant by construction. $\square$
:::

:::{div} feynman-prose
The Fractal Set is the algorithm's complete memory. Nothing is lost. Every force that acted, every step that was taken, every walker that cloned—it is all there, encoded in a coordinate-free way.

The only thing not directly recorded is what the fitness landscape looks like *between* the points that walkers actually visited (though you could interpolate if you wanted to). The realized stochastic increments themselves are stored on CST edges, so the full pathwise noise in state space is recoverable.

This is not just theoretical completeness. It means you can take a Fractal Set, hand it to someone using completely different coordinates, and they can reconstruct the exact same physics. They will see the same energies, the same convergence rates, the same optimal solutions. The mathematics guarantees it.
:::

---

(sec-latent-instantiation)=
## 7. Instantiation in Latent Space

:::{div} feynman-prose
Everything we have built so far works in any state space. We have been careful to make no assumptions about geometry beyond what is needed for scalars and vectors. Now we cash in on that generality.

The original Fractal Gas runs in Euclidean space $\mathbb{R}^d$ where the metric is flat: distances are computed the usual way, geodesics are straight lines, and the geometry is simple. But the *interesting* problems live in curved spaces. Think of optimization over probability distributions, where the natural geometry is Fisher-Rao. Or optimization in a learned latent space, where the metric reflects the structure of the data.

The Latent Fractal Gas is what you get when you run the Fractal Set machinery on a curved manifold $(\mathcal{Z}, G)$ instead of flat Euclidean space. The spinors still work—they encode vectors in a coordinate-free way, regardless of whether the underlying space is curved. The triangles still close—they record causal interactions, regardless of the geometry. The reconstruction still works—you can recover trajectories and forces, now as curves on a manifold rather than straight lines in $\mathbb{R}^d$.

What changes is the *dynamics*. In flat space, particles move in straight lines unless a force bends them. On a curved manifold, particles follow geodesics—the "straightest possible" curves on the surface. The integrator must respect this geometry, which is why we use the Boris-BAOAB scheme with its geodesic exponential map.

The beautiful thing is that the Fractal Set does not care about these differences. It stores coordinates-free data regardless of whether the coordinates live in flat or curved space. The geometry is baked into the spinors; the curvature is encoded in how the spinors transform. The data structure is universal.
:::

The Fractal Set structure is independent of the underlying state space. It can be instantiated in Euclidean space $\mathbb{R}^d$ or in a learned latent manifold $(\mathcal{Z}, G)$. This section describes the **Latent Fractal Gas** instantiation.

### 7.1 Domain Shift: Euclidean to Latent

:::{prf:definition} Latent State Space
:label: def-fractal-set-latent-space

The **latent state space** is a Riemannian manifold $(\mathcal{Z}, G)$ where:
- $\mathcal{Z} \subseteq \mathbb{R}^{d_z}$ is the latent coordinate domain
- $G: \mathcal{Z} \to \mathbb{R}^{d_z \times d_z}$ is a position-dependent metric tensor, $G(z) \succ 0$

The metric defines inner products and norms:

$$\langle u, v \rangle_{G(z)} := u^\top G(z) v, \quad \|u\|_{G(z)} := \sqrt{\langle u, u \rangle_{G(z)}}.$$
:::

The key differences between Euclidean and latent instantiations:

| Aspect | Euclidean | Latent |
|--------|-----------|--------|
| **State space** | $\mathbb{R}^d$ with $G = I$ | $(\mathcal{Z}, G(z))$ with curved metric |
| **Distance** | $\|x - y\|$ | $d_G(z_1, z_2)$ (geodesic distance) |
| **Gradients** | $\nabla f$ | $G^{-1}(z) \nabla f$ (Riemannian gradient) |
| **Integration** | Euler-Maruyama | Geodesic Boris-BAOAB |
| **Boundaries** | Physical domain boundaries | Sieve-detected: information overload, causal stasis |
| **Fitness** | Static objective | Adaptive: depends on swarm state $S$ |
| **Diffusion** | Isotropic: $\Sigma = \sigma I$ | Anisotropic: $\Sigma_{\mathrm{reg}}(z) = (\nabla^2 V_{\mathrm{fit}} + \epsilon_\Sigma I)^{-1/2}$ |

### 7.2 Soft Companion Selection

:::{div} feynman-prose
Here is a key design choice in the Latent Fractal Gas: how does a walker decide who to pay attention to?

In the simplest version, every walker would consider every other walker equally. But this is wasteful. A walker exploring the north side of the fitness landscape does not need to know what walkers on the south side are doing—they are too far away to matter.

So we use *soft* companion selection. Each walker computes a probability distribution over other walkers, weighted by proximity in phase space. Close walkers get high weight; distant walkers get low weight. The algorithmic distance combines position and velocity—if two walkers are close *and* moving in similar directions, they are "companions" who should influence each other.

The selection is "soft" because it uses a softmax-style exponential weighting rather than hard cutoffs. This makes the algorithm differentiable and stable. Small changes in walker positions cause small changes in companion weights, which prevents discontinuous jumps in the dynamics.

Each walker samples two companions per timestep: one for computing diversity (how different am I from others?), one for cloning decisions (who should I copy if I am doing badly?). These can be different walkers—you might measure diversity against a nearby explorer while considering cloning from a distant high-performer.
:::

The Latent Fractal Gas uses **phase-space softmax** to select companions for cloning and diversity computation.

:::{prf:definition} Companion Selection Kernel
:label: def-fractal-set-companion-kernel

The **companion selection weight** between walkers $i$ and $j$ is:

$$w_{ij} := \exp\left(-\frac{d_{\mathrm{alg}}(i, j)^2}{2\varepsilon^2}\right), \quad w_{ii} := 0,$$

where $d_{\mathrm{alg}}(i, j)$ is the algorithmic distance ({prf:ref}`def-fractal-set-alg-distance`) and $\varepsilon > 0$ is a temperature parameter.

The **soft companion distribution** for walker $i$ at timestep $t$ is:

$$P_i(j; t) := \frac{w_{ij}}{\sum_{l \in \mathcal{A}(t) \setminus \{i\}} w_{il}}, \quad j \in \mathcal{A}(t) \setminus \{i\}.$$
:::

Two companions are sampled independently at each timestep $t$:
- **Distance companion** $c_i^{\mathrm{dist}} \sim P_i(\cdot; t)$: Used for diversity (exploration) term
- **Cloning companion** $c_i^{\mathrm{clone}} \sim P_i(\cdot; t)$: Used for cloning source selection

### 7.3 Two-Channel Fitness Potential

:::{div} feynman-prose
The fitness function is the heart of any optimization algorithm. It tells walkers: "this is good, go here" or "this is bad, go away." But there is a subtlety that most optimization algorithms ignore: you want *both* to find good solutions *and* to explore broadly.

If you only reward high fitness, all walkers converge to the first decent solution and get stuck. This is premature convergence—the bane of optimization. If you only reward diversity, walkers scatter randomly and never exploit what they find. This is pure exploration with no exploitation.

The two-channel fitness solves this by *multiplying* two terms:
- **Reward channel**: How good is my current position? Am I moving toward high reward?
- **Diversity channel**: How different am I from nearby walkers? Am I exploring new territory?

The product means you need both. A walker in a good location but surrounded by clones has low diversity, so moderate total fitness. A walker in a bad location but isolated has low reward, so moderate total fitness. The winners are walkers who find *novel* high-quality regions—new discoveries, not just refinements.

The exponents $\alpha_{\mathrm{fit}}$ and $\beta_{\mathrm{fit}}$ let you tune the balance. High $\alpha$ emphasizes exploitation; high $\beta$ emphasizes exploration. The optimal balance depends on the problem landscape and how much you have already explored.
:::

:::{prf:definition} Two-Channel Fitness
:label: def-fractal-set-two-channel-fitness

The **fitness potential** for walker $i$ combines reward and diversity:

$$V_i := (d_i')^{\beta_{\mathrm{fit}}} (r_i')^{\alpha_{\mathrm{fit}}},$$

where:

**Reward channel**:

$$r_i := \langle \mathcal{R}(z_i), v_i \rangle_{G(z_i)},$$

the metric contraction of the reward 1-form $\mathcal{R}$ with velocity.

**Diversity channel**:

$$d_i := d_G(z_i, z_{c_i^{\mathrm{dist}}}),$$

the geodesic distance to the distance companion.

Both are standardized and transformed:

$$r_i' := g_A((\tilde{r}_i - \mu_r) / \sigma_r), \quad d_i' := g_A((\tilde{d}_i - \mu_d) / \sigma_d),$$
where $g_A(z) := A / (1 + e^{-z})$ is the logistic function with range $[0, A]$.

The exponents $\alpha_{\mathrm{fit}}, \beta_{\mathrm{fit}} > 0$ balance exploitation (reward) vs. exploration (diversity).
:::

### 7.4 Momentum-Conserving Cloning

:::{div} feynman-prose
Cloning is how the Fractal Gas reallocates resources. When a walker is doing badly—stuck in a low-fitness region while others are thriving—it "dies" and is reborn at the location of a successful walker. This is computational natural selection: successful strategies get copied; unsuccessful ones get replaced.

But there is a problem with naive cloning. If walker $i$ suddenly teleports to walker $j$'s position, the total momentum of the system changes discontinuously. This creates numerical instabilities and violates the physical intuition that the walker swarm is a kind of fluid.

The solution is *momentum-conserving cloning*. When a group of walkers clone to the same target, they undergo an "inelastic collision" in velocity space. The center-of-mass velocity is preserved, but the relative velocities shrink by a factor $\alpha_{\mathrm{rest}}$ (the coefficient of restitution). If $\alpha_{\mathrm{rest}} = 1$, the collision is perfectly elastic and velocities are unchanged. If $\alpha_{\mathrm{rest}} = 0$, all walkers in the group end up with identical velocities—the center-of-mass velocity.

This is exactly like pool balls colliding. The total momentum is conserved, but energy can be lost to "heat" (here, randomness). The effect is that cloning creates local coherence: walkers that clone together end up moving together, at least for a while.
:::

:::{prf:definition} Cloning Score and Probability
:label: def-fractal-set-cloning-score

The **cloning score** for walker $i$ toward its cloning companion $c_i^{\mathrm{clone}}$ is:

$$S_i := \frac{V_{c_i^{\mathrm{clone}}} - V_i}{V_i + \varepsilon_{\mathrm{clone}}},$$

where $\varepsilon_{\mathrm{clone}} > 0$ prevents division by zero.

The **cloning probability** is:

$$p_i := \min\left(1, \max\left(0, \frac{S_i}{p_{\max}}\right)\right),$$
where $p_{\max}$ is the maximum cloning probability.
:::

:::{prf:definition} Momentum-Conserving Cloning Update
:label: def-fractal-set-momentum-cloning

When walker $i$ clones from walker $j = c_i^{\mathrm{clone}}$:

**Position update** (Gaussian jitter):

$$z_i' := z_j + \sigma_z \zeta_i, \quad \zeta_i \sim \mathcal{N}(0, I).$$

**Velocity update** (inelastic collision): Let $G$ be the collision group (companion $j$ and all walkers cloning from $j$ this step).

$$V_{\mathrm{COM}} := \frac{1}{|G|} \sum_{k \in G} v_k, \quad u_k := v_k - V_{\mathrm{COM}},$$

$$v_k' := V_{\mathrm{COM}} + \alpha_{\mathrm{rest}} u_k,$$
where $\alpha_{\mathrm{rest}} \in [0, 1]$ is the coefficient of restitution.
:::

:::{prf:proposition} Momentum Conservation
:label: prop-fractal-set-momentum

The cloning update conserves total momentum within each collision group:

$$\sum_{k \in G} v_k' = \sum_{k \in G} v_k.$$

*Proof.*

$$\sum_{k \in G} v_k' = \sum_{k \in G} (V_{\mathrm{COM}} + \alpha_{\mathrm{rest}} u_k) = |G| V_{\mathrm{COM}} + \alpha_{\mathrm{rest}} \sum_{k \in G} u_k.$$

Since $\sum_k u_k = \sum_k (v_k - V_{\mathrm{COM}}) = \sum_k v_k - |G| V_{\mathrm{COM}} = 0$ by definition of $V_{\mathrm{COM}}$:

$$\sum_{k \in G} v_k' = |G| V_{\mathrm{COM}} = \sum_{k \in G} v_k. \quad \square$$
:::

### 7.5 Anisotropic Diffusion

:::{div} feynman-prose
Here is a clever trick that makes the Fractal Gas much smarter than isotropic random search.

In standard diffusion, randomness is added equally in all directions—the walker makes small random steps in a spherical cloud. But this is wasteful. Some directions are interesting (where the fitness varies slowly and exploration is needed), while others are dangerous (where the fitness varies rapidly and precision is needed).

Anisotropic diffusion adapts the randomness to the local landscape. The key insight is that the Hessian $\nabla^2 V_{\mathrm{fit}}$—the matrix of second derivatives of the fitness potential—tells you about the curvature of the landscape.

- **Large eigenvalue** = steep direction = the fitness changes rapidly if you move this way. You should step *carefully* here, with small random perturbations.
- **Small eigenvalue** = flat direction = the fitness barely changes if you move this way. You can step *boldly* here, with large random perturbations.

The diffusion tensor $\Sigma_{\mathrm{reg}} = (\nabla^2 V_{\mathrm{fit}} + \varepsilon_\Sigma I)^{-1/2}$ implements this automatically. It shrinks diffusion in steep directions and expands it in flat directions. The regularization $\varepsilon_\Sigma I$ ensures the tensor is always well-defined, even when the Hessian has zero eigenvalues.

The result is that walkers explore efficiently: they zoom through flat valleys and tiptoe across ridges. This is dramatically more efficient than isotropic diffusion, especially in high-dimensional spaces where most directions are irrelevant.
:::

:::{prf:definition} Fitness-Adaptive Diffusion Tensor
:label: def-fractal-set-anisotropic-diffusion

The **regularized diffusion tensor** at position $z$ is:

$$\Sigma_{\mathrm{reg}}(z) := \left(\nabla_z^2 V_{\mathrm{fit}}(z) + \varepsilon_\Sigma I\right)^{-1/2},$$
where $\nabla_z^2 V_{\mathrm{fit}}$ is the Hessian of the virtual fitness potential and $\varepsilon_\Sigma > 0$ ensures uniform ellipticity for a positive-semidefinite Hessian. If the Hessian is indefinite, use a positive-semidefinite proxy (for example, absolute eigenvalues) before taking the inverse square root. In dimensional units, a scalar scale factor may be applied so the noise term $\Sigma_{\mathrm{reg}} \circ dW$ matches the velocity SDE.
:::

The effect is directional noise adaptation:
- **Flat directions** (small Hessian eigenvalues): Large diffusion → exploration
- **Stiff directions** (large Hessian eigenvalues): Small diffusion → exploitation

### 7.6 Geodesic Integration (Boris-BAOAB)

:::{div} feynman-prose
The final piece of the puzzle is the integrator: how do we actually update walker positions and velocities? This is not as simple as it sounds, especially on a curved manifold.

In flat space, you can use the obvious formula: $x_{\mathrm{new}} = x_{\mathrm{old}} + v \cdot \Delta t$. But on a curved manifold, "add velocity times time" does not even make sense—the velocity lives in a tangent space at one point, and you cannot add it to a point in a different tangent space. You need to use the *geodesic exponential map*, which traces out the path a particle would follow if it moved with that velocity on the curved surface.

The Boris-BAOAB integrator is a sophisticated scheme designed for exactly this situation. It splits each timestep into five substeps:

1. **B (Boris kick)**: Update momentum from forces, with special handling for curl (rotating forces)
2. **A (drift)**: Move the position along the geodesic
3. **O (thermostat)**: Add random noise (the Langevin part)
4. **A (drift)**: Move again along the geodesic
5. (The B at the end combines with the B at the start of the next step)

The "Boris" part handles magnetic-like forces that would cause particles to spiral. The "BAOAB" structure ensures good statistical properties—the algorithm samples from the correct equilibrium distribution even with finite timesteps. The geodesic exponential map ensures the curved geometry is respected exactly, not approximated.

This is the numerical engine that makes the Latent Fractal Gas work: a symplectic, geometry-respecting integrator that handles forces, curvature, and stochasticity all at once.
:::

:::{prf:definition} Boris-BAOAB Integrator
:label: def-fractal-set-boris-baoab

The **Boris-BAOAB** integrator for Lorentz-Langevin dynamics on $(\mathcal{Z}, G)$ consists of five substeps per timestep $h$:

Let $p = G(z)v$ be the metric momentum and $\Phi_{\mathrm{eff}}$ the effective potential.

**B (half kick + Boris rotation)**:
1. $p \leftarrow p - \frac{h}{2}\nabla\Phi_{\mathrm{eff}}(z)$
2. If the reward has curl ($\mathcal{F} = d\mathcal{R} \neq 0$): Apply Boris rotation with parameter $\beta_{\mathrm{curl}} G^{-1}\mathcal{F}$
3. $p \leftarrow p - \frac{h}{2}\nabla\Phi_{\mathrm{eff}}(z)$

**A (half drift)**:

$$z \leftarrow \mathrm{Exp}_z\left(\frac{h}{2}G^{-1}(z)p\right),$$

where $\mathrm{Exp}_z$ is the Riemannian exponential map (geodesic flow).

**O (thermostat)**:

$$p \leftarrow c_1 p + c_2 G^{1/2}(z) \Sigma_{\mathrm{reg}}(z) \xi,$$
where $\xi \sim \mathcal{N}(0, I)$, $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1 - c_1^2)T_c}$.

**A (half drift)**: Repeat the A step.
:::

The Boris-BAOAB integrator provides:
- Second-order accuracy in timestep $h$
- Exact handling of the metric structure via geodesic exponential map
- Exact handling of reward curl via Boris rotation
- Preservation of long-time statistical properties

---

(sec-guarantees)=
## 8. Operational Guarantees

:::{div} feynman-prose
You might wonder: is all this machinery practical? Does the coordinate-independence come at a cost that makes the Fractal Set unusable for real problems?

The answer is: the costs are modest, and the benefits are substantial. Let me give you the summary before we dive into the details.

**Storage**: Spinors cost at most 2x the storage of raw vectors for $d \leq 4$. The overhead grows for higher even dimensions when Dirac spinors are required, but remains controlled by the spinor dimension $s_d$. This is cheap insurance for coordinate independence.

**Accuracy**: Reconstruction is exact for scalars and machine-precision for spinor-encoded vectors. You lose nothing to rounding in the spinor conversion.

**Speed**: Common queries (position at time $t$, force at time $t$, neighbors of walker $i$) are $O(1)$ with simple indexing. Full reconstruction is $O(NT + |E_{\mathrm{IG}}|)$—linear in the size of the data structure.

The Fractal Set is not a theoretical curiosity. It is a practical data structure for storing and analyzing optimization algorithm executions, with costs comparable to naive logging and benefits that naive logging cannot provide.
:::

### 8.1 Storage Efficiency

:::{prf:proposition} Spinor Storage Overhead
:label: prop-fractal-set-storage-overhead

For dimension $d \leq 4$, the spinor representation requires at most $2 \times d$ real numbers, compared to $d$ for raw vector storage. The overhead factor is at most 2.

For $d > 4$, the spinor dimension grows exponentially (Dirac scales as $2^{\lfloor d/2 \rfloor}$), which may exceed $d$.

| $d$ | Vector size | Spinor size (reals) | Overhead |
|-----|-------------|---------------------|----------|
| 2 | 2 | 2 | 1.0× |
| 3 | 3 | 4 | 1.33× |
| 4 | 4 | 8 | 2.0× |
| 5 | 5 | 8 | 1.6× |
| 6 | 6 | 16 | 2.67× |
| 7 | 7 | 16 | 2.29× |
| 8 | 8 | 32 | 4.0× |

*Proof.* From the spinor dimension table ({prf:ref}`def-fractal-set-spinor-space`) using the Dirac choices in even dimensions and minimal choices in odd dimensions. Direct computation gives the ratios. $\square$
:::

The storage overhead is compensated by the coordinate-independence benefit: no need to store or transmit basis information.

### 8.2 Reconstruction Accuracy

:::{prf:theorem} Reconstruction Precision
:label: thm-fractal-set-precision

Reconstruction from the Fractal Set has the following accuracy:

| Quantity | Reconstruction Error |
|----------|---------------------|
| Scalars (node attributes) | Exact (0 error) |
| Vectors from spinors | Machine precision ($\sim 10^{-15}$ relative) |
| Trajectories (accumulated) | $O(T \cdot \epsilon_{\mathrm{machine}})$ |

*Proof.* Scalar storage is lossless. Spinor-to-vector conversion involves only floating-point arithmetic (multiplication, addition), which is exact up to machine precision. Trajectory reconstruction accumulates $T$ such operations. $\square$
:::

### 8.3 Query Complexity

:::{prf:proposition} Query Time Complexity
:label: prop-fractal-set-query

Common queries on the Fractal Set have the following time complexity:

| Query | Complexity | Method |
|-------|------------|--------|
| Position/velocity at $(i, t)$ | $O(1)$ | Direct edge lookup + spinor conversion |
| Force at $(i, t)$ | $O(1)$ | CST edge lookup + spinor conversion |
| All neighbors of $i$ at $t$ | $O(\deg_t(i))$ | IG edge enumeration |
| Full trajectory of walker $i$ | $O(T)$ | CST edge chain |
| Full reconstruction | $O(NT + |E_{\mathrm{IG}}|)$ | All edges |
| Alive walkers at $t$ | $O(N)$ | Node status scan |

Here $\deg_t(i)$ is the number of sampled IG edges incident to walker $i$ at time $t$.

With indexing (hash tables on $(i, t)$ pairs), lookups become $O(1)$ expected time. $\square$
:::

---

(sec-fractal-set-parameters)=
## Parameter Glossary

| Category | Symbol | Typical Range | Unit | Description |
|----------|--------|---------------|------|-------------|
| **Dimensions** | $d$ | $2$–$10$ | [count] | State space dimension |
| | $N$ | $10^2$–$10^4$ | [count] | Number of walkers |
| | $T$ | $10^3$–$10^6$ | [count] | Number of timesteps |
| **Time** | $\Delta t$ | $10^{-3}$–$10^{-1}$ | [time] | Integration timestep |
| | $\gamma$ | $0.1$–$10$ | [1/time] | Friction coefficient |
| **Localization** | $\rho$ | Problem-dependent | [distance] | Kernel bandwidth |
| | $\varepsilon$ | $\rho / 2$ | [distance] | Companion selection temperature |
| **Fitness** | $\alpha_{\mathrm{fit}}$ | $0.5$–$2$ | [dimensionless] | Reward exponent |
| | $\beta_{\mathrm{fit}}$ | $0.5$–$2$ | [dimensionless] | Diversity exponent |
| | $\varepsilon_{\mathrm{clone}}$ | $10^{-6}$ | [dimensionless] | Cloning score regularizer |
| | $p_{\max}$ | $0.1$–$0.5$ | [probability] | Maximum cloning probability |
| **Cloning** | $\sigma_z$ | $\rho / 10$ | [distance] | Position jitter scale |
| | $\alpha_{\mathrm{rest}}$ | $0.5$–$0.9$ | [dimensionless] | Coefficient of restitution |
| **Diffusion** | $\varepsilon_\Sigma$ | $10^{-4}$–$10^{-2}$ | [dimensionless] | Diffusion floor |
| | $T_c$ | $1.0$ | [energy] | Thermostat temperature |
| **Viscosity** | $\nu$ | $0$–$1$ | [1/time] | Viscous coupling strength |
| **Phase** | $\lambda_{\mathrm{alg}}$ | $0.1$–$1$ | [time^2] | Velocity weight in algorithmic distance |
| | $\varepsilon_c$ | $\rho$ | [distance] | Coherence scale |
| | $\hbar_{\mathrm{eff}}$ | Problem-dependent | [dimensionless] | Effective phase scale (choose units so $\theta_{ij}$ is dimensionless) |

---

## Summary

- **The Fractal Set** is a **2-dimensional directed 2-complex** with simplicial support that records the complete execution of the Fractal Gas algorithm with three edge types (CST, IG, IA) and interaction triangles as fundamental 2-simplices.

- **Scalars on nodes, spinors on edges**: Frame-invariant quantities (energy, fitness, status) are stored at spacetime points; frame-covariant quantities (velocities, forces, gradients) are stored on temporal (CST) and spatial (IG) edges as spinor representations. IA edges store scalar attribution weights.

- **Three edge types close causal loops**: CST edges encode timelike evolution, IG edges encode spacelike coupling with antisymmetric cloning potentials, and IA edges complete the causal attribution from effect to cause.

- **Interaction triangles are the fundamental 2-simplices**: Each triangle $\triangle_{ij,t}$ records one complete interaction: "$j$ influenced $i$'s evolution from $t$ to $t+1$." Plaquettes (4-cycles) are derived structures—pairs of adjacent triangles.

- **Wilson loops on triangles**: The holonomy around a triangle measures the quantum phase accumulated during one interaction. Plaquette Wilson loops factorize into products of triangle Wilson loops, establishing triangles as the fundamental gauge-invariant observables.

- **Lossless reconstruction**: The Fractal Set contains sufficient information to recover discrete-time dynamics—trajectories, forces, diffusion samples, fitness values at visited points, population dynamics—in any coordinate system with machine precision.

:::{div} feynman-prose
We have built a fossil record that survives translation. The Fractal Set encodes not just what the algorithm did, but *how* it did it—and this encoding is independent of the arbitrary choices (coordinates, bases, units) that different observers might make.

The triangle is the atom of interaction. Every influence, every coupling, every cloning decision is recorded as a closed 3-cycle. You cannot break the triangle into smaller pieces without losing causality: you need to know who influenced whom (IG), what changed (CST), and to attribute the change to its source (IA).

The spinor representation ensures coordinate independence. The triangular 2-complex structure ensures that gauge-invariant observables (Wilson loops) live on the right objects: 3-cycles, not 4-cycles. Together, they make the Fractal Set a genuine mathematical object—a directed 2-complex with a natural gauge structure—rather than a pile of numbers.
:::

:::{seealso}
:class: feynman-added

- {doc}`../1_the_algorithm/01_algorithm_intuition`: Intuitive introduction to the Fractal Gas algorithm
- {doc}`../1_the_algorithm/02_fractal_gas_latent`: Formal proof object for the Latent Fractal Gas with convergence guarantees
:::
