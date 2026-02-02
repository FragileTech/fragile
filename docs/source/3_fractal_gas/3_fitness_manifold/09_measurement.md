(sec-measurement)=
# Measurement Operators on Scutoid Spacetime

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`02_scutoid_spacetime`, {doc}`/source/3_fractal_gas/2_fractal_set/02_causal_set_theory`



(sec-measurement-tldr)=
## TLDR

*Notation: $t_k$ = discrete measurement times; $H(n,b)$ = hyperplane probe with unit normal $n$ and offset $b$; $L(p,u)$ = line probe with anchor $p$ and unit direction $u$; $\mathrm{Vor}_i(t)$ = Voronoi cell; $\mathcal{N}_i(t)$ = neighbor set; $\mathrm{DT}(t)$ = Delaunay complex; $S_{i,k}$ = scutoid connecting walker $i$ across $[t_k,t_{k+1}]$; $\bar{x}(S)$ = scutoid barycenter; $d_g$ = geodesic distance in emergent metric; $d_{\mathrm{DT}}$ = Delaunay-graph distance; $\phi$ = phase; $A$ = amplitude.*

**Measurement operator**: A probe (hyperplane or line) intersects a time slice and selects a pierced set of Voronoi cells. The **pierced neighbor graph** is the Delaunay graph restricted to that pierced set.

**Scutoid-to-wave mapping**: Each scutoid defines a phase from signed distance to the probe and an amplitude from scutoid volume or QSD weight. The measurement functional aggregates these phases over pierced scutoids into a discrete signal.

**Geodesic proxies**: When the Riemannian geodesic distance is unavailable, use shortest-path distances on the Delaunay graph with Euclidean edge weights as a computational proxy.



(sec-measurement-setup)=
## 1. Measurement Setup and Probes

We measure the scutoid spacetime by intersecting discrete time slices with probing operators. The probes are geometric objects in the spatial slice and are evaluated at discrete times.

:::{prf:definition} Measurement Schedule
:label: def-measurement-schedule

Fix $T>0$ and define measurement times

$$
t_k = t_0 + k T, \quad k \in \mathbb{Z}_{\ge 0}.
$$

At each $t_k$, define a spatial probe in the latent space $\mathcal{Z}$ and evaluate which Voronoi cells are pierced.
:::

:::{prf:definition} Hyperplane Probe
:label: def-hyperplane-probe

A **hyperplane probe** is specified by a unit normal $n \in \mathbb{R}^d$ and offset $b \in \mathbb{R}$, defining

$$
H(n,b) := \{x \in \mathcal{Z} : n \cdot x = b\}.
$$

The signed Euclidean distance from $x$ to $H$ is

$$
\mathrm{dist}_E(x,H) = n \cdot x - b.
$$
:::

:::{prf:definition} Line Probe
:label: def-line-probe

A **line probe** is specified by an anchor point $p \in \mathbb{R}^d$ and unit direction $u \in \mathbb{R}^d$, defining

$$
L(p,u) := \{p + \lambda u : \lambda \in \mathbb{R}\}.
$$

The Euclidean distance from $x$ to $L$ is

$$
\mathrm{dist}_E(x,L) = \| (I - u u^T)(x - p) \|.
$$
:::



(sec-measurement-piercing)=
## 2. Pierced Cells and Pierced Neighbors

We define piercing at the Voronoi-cell level, then restrict the Delaunay graph to the pierced set.

:::{prf:definition} Pierced Voronoi Cell (Hyperplane)
:label: def-pierced-cell-hyperplane

At time $t_k$, the Voronoi cell $\mathrm{Vor}_i(t_k)$ is **pierced** by $H(n,b)$ if

$$
H(n,b) \cap \mathrm{Vor}_i(t_k) \neq \emptyset.
$$

Define the pierced set

$$
P^{H}_{t_k} := \{ i : H(n,b) \cap \mathrm{Vor}_i(t_k) \neq \emptyset \}.
$$
:::

:::{prf:definition} Pierced Voronoi Cell (Line)
:label: def-pierced-cell-line

At time $t_k$, the Voronoi cell $\mathrm{Vor}_i(t_k)$ is **pierced** by $L(p,u)$ if

$$
L(p,u) \cap \mathrm{Vor}_i(t_k) \neq \emptyset.
$$

Define the pierced set

$$
P^{L}_{t_k} := \{ i : L(p,u) \cap \mathrm{Vor}_i(t_k) \neq \emptyset \}.
$$
:::

:::{prf:definition} Pierced Neighbor Graph
:label: def-pierced-neighbor-graph

Let $\mathrm{DT}(t_k)$ be the Delaunay complex at time $t_k$ with edge set
$E_{\mathrm{DT}}(t_k)$. The **pierced neighbor graph** is the induced subgraph on the pierced set

$$
E^{\mathrm{pierced}}_{t_k} := \{(i,j) \in E_{\mathrm{DT}}(t_k) : i,j \in P_{t_k}\},
$$

where $P_{t_k}$ is either $P^{H}_{t_k}$ or $P^{L}_{t_k}$.
:::

**Interpretation**: this is precisely "pierced and already a neighbor".



(sec-measurement-scutoids)=
## 3. Scutoid Barycenters and Phase Maps

We now lift the slice-based piercing to scutoid cells and define phase functions.

:::{prf:definition} Scutoid Indexing on a Measurement Slice
:label: def-scutoid-indexing-measurement

Let $S_{i,k}$ denote the scutoid connecting walker $i$ across $[t_k,t_{k+1}]$. We associate $S_{i,k}$ to the slice $t_k$ by its bottom cell (walker index at $t_k$), and to the slice $t_{k+1}$ by its top cell. A scutoid is **pierced at time $t_k$** if its associated bottom (or top) Voronoi cell is pierced by the probe.
:::

:::{prf:definition} Scutoid Barycenter (Center-Only)
:label: def-scutoid-barycenter-center

Let $x^-_{i,k}$ and $x^+_{i,k}$ be the bottom and top walker positions of scutoid $S_{i,k}$. The **center-only barycenter** is

$$
\bar{x}_{\mathrm{ctr}}(S_{i,k}) := \tfrac{1}{2}(x^-_{i,k} + x^+_{i,k}).
$$

The spacetime barycenter is

$$
\bar{X}_{\mathrm{ctr}}(S_{i,k}) := \big(\bar{x}_{\mathrm{ctr}}(S_{i,k}),\; \tfrac{1}{2}(t_k + t_{k+1})\big).
$$
:::

:::{prf:definition} Scutoid Barycenter (Vertex-Augmented)
:label: def-scutoid-barycenter-vertex

Let $V^-_{i,k}$ and $V^+_{i,k}$ be the sets of Voronoi vertices at the bottom and top. The **vertex-augmented barycenter** is

$$
\bar{x}_{\mathrm{vx}}(S_{i,k}) := \frac{1}{2 + |V^-_{i,k}| + |V^+_{i,k}|}
\Big(x^-_{i,k} + x^+_{i,k} + \sum_{v \in V^-_{i,k}} v + \sum_{v \in V^+_{i,k}} v\Big).
$$
:::

:::{prf:definition} Phase from Hyperplane (Euclidean)
:label: def-phase-hyperplane-euclidean

Given a barycenter $\bar{x}(S)$, the Euclidean phase induced by the hyperplane probe is

$$
\phi^{H}_E(S) := k_H\, \mathrm{dist}_E(\bar{x}(S), H(n,b))
\quad \text{with} \quad \mathrm{dist}_E(\bar{x},H) = n \cdot \bar{x} - b.
$$

Here $k_H$ is a fixed scale factor (e.g., inverse length).
:::

:::{prf:definition} Phase from Line (Euclidean)
:label: def-phase-line-euclidean

Given a barycenter $\bar{x}(S)$, the Euclidean phase induced by the line probe is

$$
\phi^{L}_E(S) := k_L\, \mathrm{dist}_E(\bar{x}(S), L(p,u))
\quad \text{with} \quad \mathrm{dist}_E(\bar{x},L) = \| (I - u u^T)(\bar{x} - p) \|.
$$
:::

:::{prf:definition} Geodesic Phase Proxies
:label: def-phase-geodesic-proxy

When the geodesic distance $d_g$ is available, define

$$
\phi^{H}_g(S) := k_H\, \mathrm{dist}_g(\bar{x}(S), H),
\quad
\phi^{L}_g(S) := k_L\, \mathrm{dist}_g(\bar{x}(S), L),
$$

where $\mathrm{dist}_g$ is the Riemannian distance induced by the emergent metric
$g = H + \epsilon_\Sigma I$ on the slice. When $d_g$ is not explicitly computed, use the Delaunay-graph proxy

$$
\mathrm{dist}_g(\bar{x}, \cdot) \approx d_{\mathrm{DT}}(i, \cdot),
$$

where $i$ is the walker index associated to $S$ and $d_{\mathrm{DT}}$ is the shortest-path distance on $\mathrm{DT}(t_k)$ with Euclidean edge lengths.
:::



(sec-measurement-walker-relation)=
## 4. Walker-Relative Phases

The phase can be tied to the walker lineage associated with the scutoid, providing a direct relation between the probe and the walker state.

:::{prf:definition} Walker-Relative Phase (Euclidean)
:label: def-walker-relative-phase-euclidean

Let $x^-_{i,k}$, $x^+_{i,k}$ be the bottom and top walker positions of scutoid $S_{i,k}$. Define

$$
\phi^{-}_E(S_{i,k}) := k_W\, \|\bar{x}(S_{i,k}) - x^-_{i,k}\|,
\quad
\phi^{+}_E(S_{i,k}) := k_W\, \|\bar{x}(S_{i,k}) - x^+_{i,k}\|.
$$
:::

:::{prf:definition} Walker-Relative Phase (Geodesic Proxy)
:label: def-walker-relative-phase-geo

When $d_g$ is not explicitly available, approximate by the Delaunay-graph distance between the walker and the pierced set or the probe intersection on the slice:

$$
\phi^{-}_g(S_{i,k}) := k_W\, d_{\mathrm{DT}}(i, P_{t_k}),
\quad
\phi^{+}_g(S_{i,k}) := k_W\, d_{\mathrm{DT}}(i, P_{t_{k+1}}).
$$
:::



(sec-measurement-amplitude)=
## 5. Amplitude Maps and Measurement Functional

:::{prf:definition} Scutoid Amplitude
:label: def-scutoid-amplitude

Define the scutoid amplitude by one of the following choices:

1. **Volume amplitude**: $A_V(S) := \mathrm{Vol}_{d+1}(S)$.
2. **QSD amplitude**: $A_Q(S) := w_{\mathrm{geo}}(S)$, the geometric reweighting from
   {prf:ref}`def-cst-volume` applied to the scutoid volume or its slice-wise proxy.

Either choice yields a nonnegative amplitude and is compatible with the scutoid tessellation.
:::

:::{prf:definition} Measurement Functional
:label: def-measurement-functional

Given a probe at time $t_k$, define the measured signal as

$$
\mathcal{M}(t_k) := \sum_{S \in \mathcal{S}_{t_k}} A(S)\, e^{i\phi(S)},
$$

where $\mathcal{S}_{t_k}$ is the set of scutoids pierced at time $t_k$ (by bottom association),
$A(S)$ is a chosen amplitude map, and $\phi(S)$ is a chosen phase map from Sections 3–4.
:::

**Choice of phase map**: For hyperplane probes use $\phi^H_E$ or $\phi^H_g$; for line probes use $\phi^L_E$ or $\phi^L_g$. Walker-relative phases can be combined additively with probe phases.



(sec-measurement-gauge-compatibility)=
## 6. Gauge-Phase Compatibility

The probe-induced phase is a geometric scalar. The gauge phases in the Fractal Set live on edges and are defined by parallel transport operators (see {doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`, {prf:ref}`def-fractal-set-gauge-connection`, {prf:ref}`def-wilson-loop-lqft`). To compare scutoid phases across different walkers without fixing a gauge, we use transport to a reference and extract gauge-invariant combinations.

:::{prf:definition} Edge Parallel Transport on the Pierced Graph
:label: def-pierced-edge-transport

Let $E^{\mathrm{pierced}}_{t_k}$ be the pierced neighbor graph at time $t_k$. For each edge $(i,j) \in E^{\mathrm{pierced}}_{t_k}$, assign a parallel transport operator

$$
U_{ij} := U(e_i, e_j),
$$

with $U_{ij} \in U(1)$ for abelian transport or $U_{ij} \in SU(N)$ for non-abelian transport, as in {doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`. For $SU(2)$ interaction-pair transport, replace $U_{ij}$ by the pairwise transport $U_{ij\to i'j'}$ acting on local doublets along interaction edges (see Section 2.2 in {doc}`/source/3_fractal_gas/2_fractal_set/03_lattice_qft`). Under a local gauge transformation $G_i$ at node $i$, the transport transforms by

$$
U_{ij} \mapsto G_i\, U_{ij}\, G_j^{-1}.
$$
:::

:::{prf:definition} Scutoid Field and Transport to Reference
:label: def-scutoid-field-transport

Let $S_{i,k}$ be a scutoid associated to walker $i$ on slice $t_k$. Define its complex field value

$$
\psi(S_{i,k}) := A(S_{i,k})\, e^{i\phi(S_{i,k})}.
$$

Fix a reference walker $r$ on the same slice and a path $\gamma : i \to r$ along IG/Delaunay edges. Let $U(\gamma)$ be the parallel transport operator (path-ordered product of edge transports). The reference-transported field is

$$
\psi_r(S_{i,k}) := U(\gamma)\, \psi(S_{i,k}).
$$

Changes of local gauge at nodes act by phase rotations on $\psi$ and conjugation on $U(\gamma)$, so $\psi_r$ is gauge-covariant and can be compared at a common reference.
:::

:::{prf:definition} Reference-Transported Measurement
:label: def-reference-transported-measurement

Let $\mathcal{S}_{t_k}$ be the pierced scutoid set at time $t_k$. Define the reference-transported measurement

$$
\mathcal{M}_r(t_k) := \sum_{S \in \mathcal{S}_{t_k}} \psi_r(S).
$$

For the abelian case ($U(1)$), define the gauge-invariant scalar

$$
\mathcal{I}_r(t_k) := |\mathcal{M}_r(t_k)|.
$$

For the non-abelian case ($SU(N)$), define

$$
\mathcal{I}_r(t_k) := \mathrm{tr}\,\big(\mathcal{M}_r(t_k)\,\mathcal{M}_r(t_k)^{\dagger}\big).
$$
:::

:::{prf:lemma} Reference-Transport Covariance
:label: lem-reference-transport-covariance

Under a local gauge transformation $G_i$ at each node, the reference-transported fields satisfy

$$
\psi_r(S_{i,k}) \mapsto G_r\,\psi_r(S_{i,k}),
$$

and hence

$$
\mathcal{M}_r(t_k) \mapsto G_r\,\mathcal{M}_r(t_k).
$$

Consequently, $\mathcal{I}_r(t_k)$ is gauge-invariant in both the abelian and non-abelian cases.

*Proof.*

**Step 1. Edge transport covariance:** For any path $\gamma:i\to r$, the path-ordered transport transforms as
$U(\gamma) \mapsto G_r\,U(\gamma)\,G_i^{-1}$ by repeated application of
$U_{ij} \mapsto G_i U_{ij} G_j^{-1}$.

**Step 2. Field covariance:** Since $\psi(S_{i,k}) \mapsto G_i\,\psi(S_{i,k})$, we have
$\psi_r(S_{i,k}) = U(\gamma)\psi(S_{i,k}) \mapsto G_r\,\psi_r(S_{i,k})$.

**Step 3. Measurement covariance:** Linearity of the sum yields
$\mathcal{M}_r(t_k) \mapsto G_r\,\mathcal{M}_r(t_k)$.

**Step 4. Invariants:** In $U(1)$, $|\mathcal{M}_r|$ is invariant under multiplication by $e^{i\alpha}$. In $SU(N)$, $\mathrm{tr}(\mathcal{M}_r\mathcal{M}_r^{\dagger})$ is invariant under conjugation. $\square$
:::

:::{prf:definition} Gauge-Compatible Probe Phase
:label: def-gauge-compatible-probe-phase

A probe phase assignment $\phi(S)$ is **gauge-compatible** if, for every IG/Delaunay edge $(i,j)$ in a pierced neighbor graph at $t_k$, the relative phase matches the edge transport in the abelian case:

$$
\exp\big(i(\phi(S_{j,k})-\phi(S_{i,k}))\big) = U_{ij},
$$

or, for non-abelian transport, the scutoid field transforms by

$$
\psi(S_{j,k}) = U_{ij}\, \psi(S_{i,k})
$$

up to the chosen reference transport. This condition is optional; if it is not enforced, the probe phase is an external scalar field independent of the gauge phases.
:::

:::{prf:definition} Gauge-Invariant Measurement via Holonomy
:label: def-gauge-invariant-measurement

Let $\mathcal{C}$ be a closed loop in the pierced neighbor graph. The gauge-invariant holonomy is

$$
W(\mathcal{C}) := \mathrm{tr}\, \mathcal{P} \prod_{(i,j)\in \mathcal{C}} U_{ij}.
$$

This is the Wilson loop {prf:ref}`def-wilson-loop-lqft` evaluated on a loop in the pierced neighbor graph.

A gauge-invariant probe statistic can be formed by combining the measured field with holonomy along loops that traverse pierced cells. This isolates the gauge content from the probe-induced scalar phase.
:::



(sec-measurement-implementation-notes)=
## 7. Implementation Notes (Computable Proxies)

1. **Piercing test**: On bounded Voronoi cells, hyperplane piercing can be detected by sign changes of $n \cdot v - b$ over cell vertices $v$. For line probes, test the minimum distance from vertices to the line. When vertices are unavailable (unbounded cells), approximate using a local radius derived from neighbor distances.

2. **Geodesic proxy**: The Delaunay graph $\mathrm{DT}(t_k)$ provides a computable shortest-path metric with Euclidean edge lengths. This is a proxy for $d_g$; it introduces no new theoretical assumptions and is consistent with the Euclidean tessellation used in code.

3. **Scutoid association**: A scutoid $S_{i,k}$ can be indexed by bottom or top association depending on the measurement time. For the last time slice, use top association.

4. **Normalization**: Phase scales $k_H,k_L,k_W$ and amplitude normalization should be recorded explicitly to permit comparison across windows.
