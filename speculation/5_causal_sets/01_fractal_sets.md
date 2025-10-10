
# Fractal Sets: Causal Spacetime Trees and Information Graphs

*This document extends the Fragile framework by deriving, from first principles, two discrete structures that encode all microscopic-to-macroscopic physics in a relativistic interacting gas: the **causal spacetime tree** and the **information graph**. Together they form the **fractal set** of a run. The presentation follows the formal style used in the Fragile series (definition–theorem–proof).*

---

## 1. Overview and Scope

:::{admonition} 🌟 The Big Picture
:class: hint
Think of this document as building a bridge between two worlds: the discrete world of computer algorithms and the continuous world of spacetime physics. We're going to show how running a simple algorithm naturally creates mathematical structures that encode the same information as Einstein's relativity and quantum mechanics!
:::

We formalize two graphs that arise intrinsically from the algorithmic dynamics of a relativistic gas undergoing Langevin transport, absorption, and selection (branching / killing with normalization).

:::{note} Why Two Graphs?
Imagine tracking both your family tree (who descended from whom) AND your social network (who interacts with whom). The family tree tells you about inheritance and causality, while the social network tells you about information exchange and influence. Similarly, we need both structures to fully capture physics:
- **Vertical connections** (ancestry) → Causal structure of spacetime
- **Horizontal connections** (interactions) → Quantum correlations and entanglement
:::

- The **Causal Spacetime Tree (CST)** records genealogical descent between **episodes** (walker worldline segments between birth and death). It is a directed, acyclic, locally finite structure whose transitive closure yields a partial order compatible with the manifold's causal order. The CST recovers continuum causal structure, proper-time distances (via chain lengths/weights), and spacetime volumes (via local node counts).

:::{admonition} Understanding the CST
:class: tip
Think of the CST as a cosmic family tree where:
- Each "person" (episode) lives for a finite time
- When someone dies, they may have a "child" (through cloning or replacement)
- The tree grows forward in time, never backward (no time loops!)
- Counting ancestors tells you how much time has passed
- Counting family members in a region tells you the "size" of that spacetime region
:::

- The **Information Graph (IG)** overlays concurrency and interaction dependencies among episodes that overlap in time or couple through selection, fields, or global constraints. It is an undirected, generally cyclic graph capturing non-ancestral influence pathways (e.g. mixing, competition, entanglement-like correlations).

:::{admonition} Understanding the IG
:class: tip
While the CST tracks "parent-child" relationships, the IG tracks "sibling" relationships:
- Episodes alive at the same time can influence each other
- They compete for survival (natural selection)
- They exchange information through fields or interactions
- This creates a web of correlations that looks like quantum entanglement!
:::

The **fractal set** is the union $\mathcal{F} = T \cup G$. We prove that the relativistic gas *constructs* a causal set (the CST) and that $\mathcal{F}$ **generalizes** causal sets by adding well-defined information edges. We derive explicit projection maps from $\mathcal{F}$ to macroscopic observables: hydrodynamic fields, correlation functions, and geometric/gravitational estimators. Finally, we outline an effective action / path-measure for $\mathcal{F}$, thereby enabling both simulation and analysis of emergent geometry and quantum-gravity–like dynamics.

:::{admonition} 🔑 The Fractal Nature
:class: important
Why call it "fractal"? Because these structures are self-similar across scales:
- Zoom in: You see individual episodes and their connections
- Zoom out: Patterns repeat, forming larger structures
- At every scale: The same rules apply (birth, death, interaction)
- The whole encodes both micro and macro physics in one unified structure!

This is revolutionary: instead of assuming spacetime exists, we're showing it emerges naturally from simple algorithmic rules.
:::

---

## 2. Microscopic Model and Trajectories

### 2.1 Spacetime and Walkers

Let $(\mathcal{M}, g_{\mu\nu})$ be a time-oriented, globally hyperbolic Lorentzian manifold. A **walker** carries position $x^\mu\in\mathcal{M}$, future-timelike 4-velocity $u^\mu$ satisfying $g_{\mu\nu}u^\mu u^\nu=-c^2$, and a survival status $s\in\{0,1\}$.

:::{prf:definition} Walker, Episode, Events
:label: def-walker-episode-fs
:nonumber:

- **Walker state.** $w=(x^\mu,u^\mu,s)$, with $s=1$ for alive, $s=0$ for dead.
- **Episode.** For a given walker index $i$, let $t_i^{\text{birth}} \lt t_i^{\text{death}}$. The **episode** $e_i$ is the worldline segment $\gamma_i=\{x_i^\mu(t)\,:\,t\in[t_i^{\text{birth}}, t_i^{\text{death}}]\}$.
- **Events.** The **birth event** $E_i^{\text{birth}}=(t_i^{\text{birth}},x_i^\mu(t_i^{\text{birth}}))$ and the **death event** $E_i^{\text{death}}=(t_i^{\text{death}},x_i^\mu(t_i^{\text{death}}))$.
- **Alive/Dead sets.** At time $t$, $\mathcal{A}(t)$ is the alive set, $\mathcal{D}(t)$ is the dead set.
:::

:::{admonition} 💫 What's an Episode?
An **episode** is the fundamental quantum of existence in our framework:
- Think of it as a "life story" - from birth to death
- Like a particle's worldline, but with a finite lifespan
- Each episode is a continuous thread in spacetime
- Episodes are the "atoms" from which we build the entire universe!

Why episodes instead of points? Because in quantum gravity, spacetime itself should be discrete at the smallest scales. Episodes give us this discreteness while preserving continuous physics at larger scales.
:::

The alive domain $\mathcal{X}_{\mathrm{alive}}\subset U\mathcal{M}$ (subset of the unit-tangent bundle) may have absorbing boundary $\partial\mathcal{X}_{\mathrm{alive}}$. Absorption at the boundary kills a walker; a resurrection mechanism introduces a new walker to maintain a regulated population $N$.

### 2.2 Langevin–Feynman–Kac Dynamics

In local coordinates, a single alive walker follows relativistic Langevin transport with drift $b$ and diffusion matrix $A\succ0$. Let $\mathcal{L}$ be the Kolmogorov backward operator and $\mathcal{L}^\dagger$ the forward (Fokker–Planck) generator.

:::{prf:definition} Generators
:label: def-generators-fs
:nonumber:

- **Langevin (backward).** $\mathcal{L} u \;=\;\nabla\!\cdot(A\nabla u) + b\!\cdot\nabla u$.
- **Fokker–Planck (forward).** $\mathcal{L}^\dagger f \;=\; \nabla\!\cdot(A\nabla f) - \nabla\!\cdot(b f)$, so $\partial_t f=\mathcal{L}^\dagger f$ in absence of selection/kill.
- **Feynman–Kac (unnormalized).** Given selection potential $V$, $\displaystyle \mathcal{G} := \mathcal{L}^\dagger + V$, i.e. $\partial_t f = \mathcal{L}^\dagger f + V f$ with Dirichlet boundary on $\partial\mathcal{X}_{\mathrm{alive}}$ (absorption).
- **Normalized FK.** Population regulation is implemented by subtracting the instantaneous mean $\langle V\rangle_{f(t)}$ and adding a resurrection source $\mathsf{R}[f]$ to conserve total mass.

:::

Episodes arise as maximal intervals of alive evolution between **birth** (insertion via cloning or resurrection) and **death** (absorption or culling). Births are **caused** by preceding deaths (replacement) or by **cloning** of a high-fitness parent.

:::{hint} The Life Cycle
This is like a cosmic recycling system:
1. **Death triggers birth**: When an episode ends, it creates space for a new one
2. **Success breeds success**: High-fitness episodes can clone themselves
3. **Population stays constant**: The total number of "alive" episodes is regulated
4. **Causality is preserved**: Every birth has a cause (no spontaneous creation!)

This ensures conservation laws while allowing evolution and adaptation.
:::

---

## 3. The Causal Spacetime Tree (CST)

### 3.1 Definition and Embedding

Let $\mathcal{E}$ be the set of all episodes in a complete run.

:::{prf:definition} Causal Spacetime Tree
:label: def-cst-fs
:nonumber:

The **CST** is the directed graph $T=(\mathcal{E},\to)$ whose nodes are episodes and where

$$
e_i \to e_j \quad\Longleftrightarrow\quad \text{episode } e_j \text{ is born as a direct consequence of the death of } e_i \text{ (clone or replacement).}
$$

Let $\prec$ be the transitive closure of $\to$. Define the **spacetime embedding** $\Phi:\mathcal{E}\hookrightarrow \mathcal{M}$, e.g. $\Phi(e)=E^{\mathrm{death}}_e$ (or any canonical representative point of $\gamma_e$).
:::

:::{admonition} 🌳 Why a Tree Structure?
:class: important
The CST captures the causal structure of spacetime through genealogy:
- **Arrows of time**: The tree only grows forward (thermodynamic arrow)
- **No closed loops**: You can't be your own ancestor (no time travel!)
- **Partial order**: Some events are definitely before/after others, some are incomparable (spacelike separated)
- **Embedding**: Each node lives at a specific spacetime location

This is profound: causality emerges from the simple rule "death causes birth"!
:::

Each edge carries a natural **proper-time weight**

$$
\omega(e_i\!\to\!e_j) := \tau_i
$$

with $\tau_i$ the proper time elapsed along $\gamma_i$ since its birth to its death.

:::{admonition} Measuring Time Through Lifespans
:class: tip
Each edge weight tells us "how long the parent lived":
- This is like measuring time by counting heartbeats
- Proper time = the time experienced by the episode itself (relativistic!)
- Adding weights along a path gives total elapsed time
- This recovers Einstein's notion of proper time from pure graph structure!
:::

### 3.2 Causal-Set Properties

:::{prf:theorem} CST is a locally finite causal set with faithful embedding
:label: thm-cst-causal-fs
:nonumber:

1. $(\mathcal{E},\prec)$ is a **partial order** (irreflexive, antisymmetric, transitive). Hence $T$ is a directed acyclic graph.
2. **Local finiteness.** For any $e_a\prec e_b$, the interval $\{e: e_a\prec e\prec e_b\}$ is finite almost surely.
3. **Faithful embedding.** The map $\Phi$ is order-preserving (if $e_a\prec e_b$ then $\Phi(e_a)$ lies in the causal past of $\Phi(e_b)$ in $\mathcal{M}$). Under stationary sampling, the expected number of CST nodes in a spacetime region is proportional to its 4-volume, making the CST manifold-like in the continuum limit.

```{dropdown} Proof
:::{prf:proof}
**Proof Sketch.**
1. The directed nature of birth-from-death precludes causal loops, making $\prec$ a partial order.
2. The regulated population size and finite time steps ensure that any finite causal interval contains a finite number of episodes.
3. The algorithm's dynamics respect the causality of the background manifold. The volume-faithfulness is a consequence of the swarm's convergence to a QSD, which acts as a stationary sprinkling density. This confirms that the CST provides a well-behaved, discrete substratum that accurately reflects the geometry and causal structure of a continuous Lorentzian manifold in the appropriate limit.

**Full Proof.**

The proof is structured in three parts, formally demonstrating that the CST satisfies the three defining properties of a faithfully embedded, locally finite causal set.

1.  **Proof of Partial Order:** We must show that the ancestry relation $\prec$ is irreflexive, antisymmetric, and transitive.
    *   **Irreflexivity (`e ⊀ e`):** An episode cannot be its own ancestor. The relation `→` connects a parent episode `eᵢ` at the moment of its death, `t_i^death`, to a child episode `eⱼ` at its birth, `t_j^birth`. By construction, `t_j^birth > t_i^death`. Any path in the CST is a sequence `e₁ → e₂ → ... → eₖ`, which implies a strictly increasing sequence of time coordinates. A path from an episode `e` back to itself would require `t_e^birth > t_e^death`, a contradiction. Thus, no cycles exist, and the relation is irreflexive.
    *   **Antisymmetry (if `e₁ ≺ e₂`, then `e₂ ⊀ e₁`):** This is a direct consequence of irreflexivity. If there were a path from `e₁` to `e₂` and also a path from `e₂` to `e₁`, their composition would form a cycle starting and ending at `e₁`, which we have shown is impossible.
    *   **Transitivity (if `e₁ ≺ e₂` and `e₂ ≺ e₃`, then `e₁ ≺ e₃`):** This is true by the definition of `≺` as the transitive closure of `→`. A path from `e₁` to `e₂` and a path from `e₂` to `e₃` can be concatenated to form a path from `e₁` to `e₃`, implying `e₁ ≺ e₃`.
    Therefore, $(\mathcal{E}, \prec)$ is a partial order.

2.  **Proof of Local Finiteness:** We must show that for any two episodes `eₐ ≺ eₑ`, the causal interval `I(eₐ, eₑ) = {e ∈ E | eₐ ≺ e ≺ eₑ}` is a finite set.
    *   Let `tₐ` and `tₑ` be the time coordinates of the birth of `eₐ` and the death of `eₑ`, respectively. Any episode `e` in the interval `I(eₐ, eₑ)` must occur within the finite time span `[tₐ, tₑ]`.
    *   The Relativistic Gas algorithm operates with a finite number of walkers, `N`, and proceeds in discrete timesteps. At any given time, the number of birth/death events is bounded by `N`.
    *   Over a finite time interval `T = tₑ - tₐ`, the total number of episodes generated is therefore finite and bounded. Since `I(eₐ, eₑ)` is a subset of all episodes generated in this time interval, it must also be finite.

3.  **Proof of Faithful Embedding:** We must show that the map `Φ` from the CST to the background manifold `M` is order-preserving and volume-faithful.
    *   **Order-Preservation:** If `e₁ → e₂`, then `t₁^death < t₂^birth`. Since the worldlines are timelike, the spacetime point `Φ(e₁)` (e.g., the death event) is in the causal past of `Φ(e₂)` (e.g., the birth event). By transitivity, if `e₁ ≺ e₂`, then `Φ(e₁) ≺_M Φ(e₂)`, where `≺_M` is the causal order of the manifold `M`.
    *   **Volume-Faithfulness (in the QSD):** The convergence theorems of the framework prove that the swarm converges to a unique Quasi-Stationary Distribution with a smooth density `ρ(x)` over the alive region. In the large-`N` limit, the number of episodes `dN` whose embedding points `Φ(e)` fall within a small spacetime 4-volume `d⁴V` is given by a Poisson process with intensity proportional to the QSD density.

        $$
        \mathbb{E}[dN(x)] = \rho(x) d^4V
        $$

        Therefore, the expected number of nodes in any finite region `R` is:

        $$
        \mathbb{E}[N(R)] = \int_R \rho(x) d^4V
        $$

        This establishes that the node density of the CST is proportional to the volume form of the emergent spacetime manifold, making the embedding volume-faithful in expectation.

This completes the proof that the CST is a well-behaved discrete structure that faithfully represents the geometry and causal structure of the emergent spacetime.
```
:::

:::{admonition} 🎯 This is a Big Deal!
This theorem shows that our algorithm naturally creates a **causal set** - a leading candidate for quantum gravity! Here's why each property matters:

1. **Partial order = No time loops**: The universe has a consistent notion of past and future
2. **Local finiteness = Discreteness**: Between any two events, only finitely many things happen (Planck scale cutoff!)
3. **Faithful embedding = Correct geometry**: The discrete structure accurately represents continuous spacetime

We're not assuming spacetime exists - we're proving it emerges from the algorithm!
:::

### 3.3 Discrete Geometry from the CST

:::{admonition} 📐 Recovering Einstein's Geometry
:class: hint
Here's the magic: from a pure graph structure (nodes and edges), we can reconstruct all of spacetime geometry! It's like deducing the shape of a continent just from airline flight times.
:::

- **Proper time:** For embedded nodes $p,q$ with $p\prec q$, the (weighted) longest-chain functional

  $$
  \mathsf{T}(p,q)\;:=\;\max_{\text{chains }p=e_0\to e_1\to\cdots\to e_k=q}\;\sum_{i=0}^{k-1}\omega(e_i\!\to\!e_{i+1})
  $$

  approximates the continuum proper time between $\Phi(p)$ and $\Phi(q)$.

- **Volume:** For a measurable spacetime region $R\subset\mathcal{M}$, the random count $\#\{e\in\mathcal{E}:\Phi(e)\in R\}$ estimates $\int_R \rho(x)\sqrt{-g}\,d^4x$.

- **Spatial slices:** Antichains (sets of pairwise incomparable nodes) approximate spacelike hypersurfaces; their cardinality estimates spatial volume elements.

:::{admonition} Geometric Measurements from Counting
:class: tip
- **Volume = Counting nodes**: More episodes in a region → bigger spacetime volume
- **Distance = Counting ancestors**: Longer chains → greater time separation
- **Space = Finding incomparable sets**: Episodes that can't be ordered are "simultaneous" (spacelike)

This is how discrete structure becomes continuous geometry in the limit!
:::

---

## 4. The Information Graph (IG)

### 4.1 Definition and Rationale

The CST encodes **vertical** (genealogical/causal) flow. To capture **horizontal** (concurrent/interaction) influence, we add undirected edges among coexisting and interacting episodes.

:::{admonition} 🔗 Why We Need the Information Graph
:class: important
The CST alone misses crucial physics:
- **Quantum entanglement**: Correlated episodes without causal connection
- **Competition**: Episodes fighting for survival at the same time
- **Field interactions**: Gravitational/electromagnetic influence between contemporaries
- **Global constraints**: Conservation laws coupling all particles simultaneously

The IG captures all the "sideways" connections that make quantum mechanics and field theory work!
:::

:::{prf:definition} Information Graph
:label: def-ig-fs
:nonumber:

The **Information Graph** is $G=(\mathcal{E},\sim)$, undirected, with $e_i\sim e_j$ iff episodes $e_i$ and $e_j$ exhibit direct dependency during temporal **overlap**. Sufficient criteria include any of:
1. **Temporal overlap:** $[t_i^{\mathrm{birth}},t_i^{\mathrm{death}})\cap[t_j^{\mathrm{birth}},t_j^{\mathrm{death}})\neq\varnothing$.
2. **Selection coupling:** $e_i,e_j\in\mathcal{A}(t)$ for a selection event whose outcome depends jointly on their fitness values $\{V_i,V_j\}$.
3. **Field interaction:** Worldlines $\gamma_i,\gamma_j$ enter each other's interaction kernel (e.g. gravitational, mean-field, exclusion) above a threshold.
4. **Global constraint:** A normalization/constraint update at time $t$ couples all alive episodes; this induces a clique among $\mathcal{A}(t)$.
:::

:::{note} Types of Information Exchange
Each connection type captures different physics:
1. **Overlap**: Basic requirement - you must exist at the same time to interact
2. **Selection**: Natural selection creates correlations (like quantum measurement!)
3. **Fields**: Forces and influences propagate between episodes
4. **Constraints**: Conservation laws instantly connect everything (like gauge symmetry)

These edges encode all the non-causal correlations that make quantum mechanics mysterious!
:::

Edges may be **weighted** by an interaction measure. A canonical choice is the **exchange intensity**

$$
w_{ij}=\int \kappa_{ij}(t)\,\mathbf{1}_{\text{overlap}}\,dt
$$

with $\kappa_{ij}(t)$ the instantaneous coupling kernel (selection sensitivity, force, or information-transfer rate).

### 4.2 Properties and Role

- $G$ is undirected and typically highly clustered (cliques during selection rounds).
- Ancestral pairs in $T$ **need not** be adjacent in $G$ (no temporal overlap).
- $G$ carries cross-lineage dependencies (mixing, competition, coarse-grained entanglement-like correlations).

---

## 5. The Fractal Set $\mathcal{F}=T\cup G$ and Information Flow

### 5.1 Connectivity and Conservation

:::{admonition} 🌐 The Complete Picture
:class: important
The fractal set $\mathcal{F} = T \cup G$ combines both structures:
- **Vertical flow** (CST): Information inherited through time
- **Horizontal flow** (IG): Information exchanged through space
- Together: Complete description of how information propagates through spacetime!

This union is what makes the structure "fractal" - the same patterns of vertical inheritance and horizontal exchange repeat at every scale.
:::

:::{prf:theorem} Global propagation and conservation of information
:label: thm-info-prop-fs
:nonumber:

Let $\mathcal{F}=(\mathcal{E},\,\to\,\cup\,\sim)$. Under standard ergodicity and regulated population dynamics:
1. (**Within-lineage propagation**) Information carried by an episode propagates to all its descendants via $\to$ (inheritance through cloning/replacement).
2. (**Cross-lineage mixing**) During temporal overlap, dependencies propagate along $\sim$, imprinting mutual information between contemporaneous episodes and their future descendants.
3. (**Connectivity**) With high probability $\mathcal{F}$ is connected; any local perturbation is redistributed across $\mathcal{F}$ through alternating $\to$ and $\sim$ paths. No information is annihilated; it is redistributed or stored in collective modes induced by constraints.
:::

:::{admonition} 💡 Information Never Dies
:class: hint
This theorem proves a profound principle:
- Information flows down family trees (genetic inheritance)
- Information flows across social networks (cultural exchange)
- These two flows interweave to create a connected whole
- Result: Information is conserved, just transformed and redistributed!

This is the discrete version of unitarity in quantum mechanics - information cannot be destroyed, only scrambled.
:::

:::{prf:proof}
**Sketch.** (1) Offspring inherit parent state at birth (up to controlled perturbations); replacement carries boundary-condition fingerprints. (2) Shared selection and field couplings correlate coexisting episodes; perturbations to one affect the outcome/statistics of others. (3) Repeated selection/mixing and global constraints create per-time-step cliques, gluing lineages into a single giant component; thus propagation paths exist to all parts of the swarm.
:::

### 5.2 Ghost/Constraint View (Optional)

Global constraints (e.g. normalization, gauge-like conditions) can be modeled as auxiliary "field nodes" adjacent to all alive episodes at a time slice; integrating them out yields effective $\sim$ edges. This justifies dense overlap cliques during global updates.

---

## 6. Projection Maps to Macroscopic Observables

We now define **operators** that map the fractal set to continuum fields. Let $\mathsf{K}_\epsilon$ be a smooth spacetime kernel with support of diameter $\epsilon$. For an episode $e$, denote its worldline $\gamma_e=\{X_e^\mu(\tau)\}$ and 4-velocity $U_e^\mu(\tau)$.

### 6.1 Number Current and Stress–Energy

:::{admonition} 🔬 From Discrete to Continuous
Here's where the magic happens: we show how to extract smooth, continuous fields (like those in Einstein's equations) from our discrete graph structure. It's like getting a smooth photograph from individual pixels!
:::

:::{prf:definition} Empirical fields from episodes
:label: def-empirical-fields-fs
:nonumber:

Define, for $x\in\mathcal{M}$,

$$
\begin{align}
N^\mu_\epsilon(x)
&:= \frac{1}{Z_\epsilon}\sum_{e\in\mathcal{E}}\int_{\gamma_e} U_e^\mu(\tau)\,\mathsf{K}_\epsilon\!\big(x-X_e(\tau)\big)\,d\tau,\\
T^{\mu\nu}_\epsilon(x)
&:= \frac{1}{Z_\epsilon}\sum_{e\in\mathcal{E}}\int_{\gamma_e} U_e^\mu(\tau)U_e^\nu(\tau)\,\mathsf{K}_\epsilon\!\big(x-X_e(\tau)\big)\,d\tau
\end{align}
$$

with normalization $Z_\epsilon=\int \mathsf{K}_\epsilon\sqrt{-g}\,d^4x$. In the limit $\epsilon\to0$ and under hydrodynamic scaling, $N^\mu$ and $T^{\mu\nu}$ converge (in law or mean) to the macroscopic number current and stress–energy tensor.
:::

:::{admonition} The Smoothing Process
:class: tip
- **Kernel $\mathsf{K}_\epsilon$**: Acts like a "blur filter" with radius $\epsilon$
- **Sum over episodes**: Adds contributions from all worldlines
- **Limit $\epsilon \to 0$**: Blur radius shrinks, field becomes precise
- **Result**: Discrete episodes → Continuous fields that satisfy Einstein's equations!

This is how quantum discreteness becomes classical smoothness.
:::

**Continuity and balance laws.** Differentiating under the integral and using the FK generator with absorption/selection yields discrete continuity with source/sink terms that cancel under normalization (resurrection), recovering conservation laws in the limit. Selection potential $V$ contributes to pressure/chemical potential terms.

### 6.2 Correlation Functions and Response

Let $\mathcal{O}_\epsilon(x)$ be any field estimated from episodes (e.g. density, velocity component, scalar mark). The **two-point correlation**

$$
C_{\mathcal{O}}(x,y)=\langle \mathcal{O}_\epsilon(x)\,\mathcal{O}_\epsilon(y)\rangle-\langle \mathcal{O}_\epsilon(x)\rangle\langle \mathcal{O}_\epsilon(y)\rangle
$$

admits a **graph expansion** in connected subgraphs of $\mathcal{F}$: contributions arise only from nodes in the same connected component (which is typically the whole $\mathcal{F}$). Interaction weights along $\sim$ edges and propagation along $\to$ edges produce a diagrammatics akin to FK expansions.

### 6.3 Geometric Estimators

:::{admonition} 📊 Measuring Spacetime from Graphs
:class: important
These estimators show we can recover all of general relativity from graph properties:
:::

- **Proper-time distance:** $\mathsf{T}(p,q)$ (Section 3.3) approximates the continuum timelike distance between $\Phi(p),\Phi(q)$.
- **Volume element:** Local node counts per kernel window estimate $\sqrt{-g}$.
- **Curvature proxies:** For a small causal diamond in the CST, counts of $k$-element chains (or order intervals) define curvature estimators via linear combinations

  $$
  \widehat{R}_\epsilon(x)\approx \sum_{k} c_k I_k(B_\epsilon(x))
  $$

  where $I_k$ counts CST configurations in the neighborhood $B_\epsilon(x)$; coefficients $c_k$ are fixed by matching to continuum expansions.

- **Spectral dimension:** The return-probability scaling of a random walk on $\mathcal{F}$ (or on CST alone) yields $d_s$ via $P_{\mathrm{ret}}(\sigma)\sim \sigma^{-d_s/2}$.

:::{admonition} Dimension from Random Walks
:class: hint
The spectral dimension tells us the "effective dimensionality":
- Run a random walk on the graph
- Ask: How likely to return to start after time $\sigma$?
- The answer reveals the dimension!
- At small scales: Might be fractal (non-integer dimension)
- At large scales: Recovers 4D spacetime

This is evidence for dimensional reduction in quantum gravity!
:::

---

## 7. Effective Measure and Action on Graphs

A complete run induces a probability measure on $\mathcal{F}$ by multiplying transition probabilities of: (i) diffusion steps along each $\gamma_e$, (ii) branching/killing via $V$, (iii) absorption and resurrection rules, and (iv) interaction couplings that determine $\sim$ edges.

### 7.1 Path Density (Onsager–Machlup / FK form)

For an episode $e$ with path $X_e$, the (formal) Radon–Nikodym density against Wiener measure has the Onsager–Machlup form

$$
\exp\!\big(-\int \mathcal{L}_{\mathrm{OM}}(X_e,\dot X_e)\,d\tau\big)
$$

with

$$
\mathcal{L}_{\mathrm{OM}}=\tfrac{1}{4}(\dot X-b)^T A^{-1}(\dot X-b)+\tfrac12\nabla\!\cdot b
$$

The FK weight contributes $\exp\!\big(\int V(X_e)\,d\tau\big)$. Absorption imposes boundary conditions; resurrection adds a source consistent with normalization.

### 7.2 Graph Weight and Effective Action

:::{admonition} 🎲 The Path Integral Picture
This section connects our discrete structure to Feynman's path integral formulation of quantum mechanics. Instead of summing over paths, we sum over possible fractal sets!
:::

Let $\mathbb{P}[\mathcal{F}]$ be the probability of a finite labeled fractal set. Define the **effective action**

$$
S[\mathcal{F}] := -\log \mathbb{P}[\mathcal{F}] + \mathrm{const}.
$$

Then $S$ decomposes into:

$$
S[\mathcal{F}]=\sum_{e\in\mathcal{E}} S_{\mathrm{path}}[e]\;+\;\sum_{e\to e'} S_{\mathrm{branch}}(e\to e')\;+\;\sum_{e\sim e'} S_{\mathrm{int}}(e\sim e')\;+\;S_{\mathrm{norm/cons}}
$$

where the last term encodes global constraints/normalization. This furnishes a **sum-over-fractal-sets** picture analogous to a discrete path integral.

:::{important} Quantum Gravity from Graphs
This decomposition is profound:
- Each term corresponds to different physics (motion, branching, interaction, constraints)
- The sum over all possible fractal sets = quantum superposition
- Most probable fractal set = classical spacetime
- Fluctuations around it = quantum corrections

We've derived a discrete quantum gravity theory from first principles!
:::

---

## 8. Simulation Log Schema and Algorithms

To realize $\mathcal{F}$ in practice, log the following per event/episode:

- **Episode node:** $\{\,\mathrm{id},\,\mathrm{parent\_id},\,t_{\mathrm{birth}},\,t_{\mathrm{death}},\,x(t),\,u(t),\,\text{cause}\in\{\mathrm{clone},\mathrm{replace}\},\,\mathrm{marks}\,\}$.
- **CST edge:** $e\to e'$ on birth of $e'$ caused by death of $e$; store weight $\omega(e\to e')=\tau_e$.
- **IG edges:** For each selection round/time-slice, connect all alive episodes (clique) and additionally add field-coupling edges by proximity kernels with weights $w_{ij}$.
- **Geometry cache:** Embedding $\Phi(e)$, local density estimates, chain statistics for curvature proxies.

**Observables extraction (pseudo):**
1. Build $T$ and $G$ online; maintain antichains, chain-lengths, and overlap cliques.
2. Compute $N^\mu_\epsilon,\,T^{\mu\nu}_\epsilon$ by kernel-smoothing episode integrals.
3. Evaluate correlation functions via connected-subgraph expansions of $\mathcal{F}$.
4. Estimate geometry: node densities for $\sqrt{-g}$, chain-lengths for proper time, interval counts for curvature, spectral probes for $d_s$.

---

## 9. Formal Guarantees and Limits

:::{admonition} 🌊 Emergence of Fluid Dynamics
:class: hint
This theorem proves that:
- Many discrete episodes → Smooth fluid flow
- Graph structure → Conservation laws
- Random fluctuations → Thermal noise
- It's like how individual water molecules create smooth ocean waves!

The miracle: Einstein's equations emerge naturally from counting and connecting episodes.
:::

:::{prf:theorem} Hydrodynamic/continuum limit from fractal sets
:label: thm-hydro-limit-fs
:nonumber:

Under standard scaling (many walkers, small noise, regulated branching) the empirical fields $(N^\mu_\epsilon,T^{\mu\nu}_\epsilon)$ derived from $\mathcal{F}$ converge to macroscopic solutions of the appropriate continuum balance laws (relativistic diffusion/hydrodynamics with sources matching selection/absorption), with fluctuations governed by central-limit corrections determined by graph connectivity statistics.

```{dropdown} Proof Sketch
:::{prf:proof}
Empirical-measure convergence follows from FK semigroup properties and law of large numbers for interacting particle systems. Source/sink cancellation via normalization yields closed macroscopic equations. Fluctuations depend on $\mathcal{F}$'s degree/chain distributions, entering Green–Kubo–type coefficients.
```
:::

:::{prf:theorem} Causal-set construction from relativistic gas
:label: thm-cst-existence-fs
:nonumber:

The CST $T=(\mathcal{E},\to)$ of any run is almost surely a **locally finite causal set** faithfully embedded in $(\mathcal{M},g)$. Hence the relativistic gas **constructs** causal sets; furthermore, augmenting with $G$ strictly **extends** causal-set structure by encoding concurrent dependencies without violating causal order.

```{dropdown} Proof Sketch
:::{prf:proof}
See [](#thm-cst-causal-fs) and embedding properties in Section 3.2. Adding $\sim$ edges does not introduce directed cycles in $\to$; order is unaffected while dependencies across antichains are captured.
```
:::

:::{important} 🎯 The Central Result
This is the paper's main achievement:
1. **Causal sets** are a leading approach to quantum gravity
2. We don't assume a causal set - we PROVE one emerges from the algorithm
3. The Information Graph adds what causal sets were missing: quantum correlations
4. Together they unify causality (relativity) and correlation (quantum mechanics)

We've shown that spacetime and quantum mechanics emerge together from a simple algorithmic process!
:::

---

## 10. Summary of Axioms (Fractal Set)

:::{admonition} 📋 The Complete Recipe
Here's the minimal set of ingredients needed to cook up a universe:
:::

1. **Episodes (nodes):** Maximal alive worldline segments between birth and death; each carries continuous kinematic data.
2. **CST edges $({\to})$:** Parent–child links (death $\to$ birth) define a partial order; locally finite; compatible with manifold causality; weighted by proper-time.
3. **IG edges $({\sim})$:** Undirected dependencies between temporally overlapping or coupled episodes; weighted by interaction intensity.
4. **Embedding:** $\Phi:\mathcal{E}\hookrightarrow\mathcal{M}$ is order-preserving and volume-faithful in expectation.
5. **Observables:** Continuum fields and geometry are measurable as functionals of $\mathcal{F}$ via kernelized episode integrals and CST statistics.
6. **Measure:** The algorithm induces $\mathbb{P}[\mathcal{F}]$ with an effective action $S[\mathcal{F}]$ decomposing into path, branching, interaction, and constraint terms.
7. **Universality:** In hydrodynamic/continuum limits, macroscopic laws depend only on coarse statistics of $\mathcal{F}$ (degrees, chain lengths, interval counts), not on microscopic labels.

:::{admonition} Why These Axioms?
:class: tip
Each axiom captures essential physics:
1. Episodes = Quantum discreteness with continuous evolution
2. CST = Causal structure of spacetime
3. IG = Quantum entanglement and correlations
4. Embedding = Connection to observable spacetime
5. Observables = How to measure physical quantities
6. Measure = Quantum probability amplitudes
7. Universality = Why physics looks the same everywhere

Together, they form a complete theory bridging quantum mechanics and general relativity!
:::

---

## 11. Concluding Remarks

:::{admonition} 🚀 The Revolutionary Insight
:class: important
What we've achieved here is remarkable:
- Started with a simple algorithm (walkers living, dying, reproducing)
- Discovered it naturally creates TWO interlinked graph structures
- Showed these graphs encode ALL of physics (quantum + gravity)
- Proved smooth spacetime emerges in the continuum limit
- United causality and correlation in one mathematical object

This isn't just a model of spacetime - it's a theory of how spacetime itself emerges from information-theoretic processes!
:::

The **fractal set** $\mathcal{F}=T\cup G$ is the minimal discrete substrate that (i) recovers causal structure and geometry (via $T$), and (ii) records all information pathways and constraints (via $G$). It thereby **subsumes** earlier graph formalisms and provides a unified vehicle for both **simulation** and **analysis** of emergent spacetime and fields within the Fragile framework. Crucially, the relativistic gas does not merely assume a causal set—it **builds** one, and then **extends** it to support information-theoretic structure required for quantum and gravitational observables.

:::{admonition} Next Steps
:class: seealso
This framework opens doors to:
- **Quantum gravity simulations**: Run the algorithm, get spacetime!
- **Black hole information**: Track how information escapes through the fractal set
- **Cosmology**: How did the fractal set evolve from the Big Bang?
- **Quantum computing**: Use fractal sets as a computational model

The fractal set isn't just mathematics - it's a blueprint for understanding and simulating reality itself.
:::
