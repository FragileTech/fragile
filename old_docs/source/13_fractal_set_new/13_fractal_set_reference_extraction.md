# Mathematical Objects Extraction: Fractal Set Chapter

**Source**: `/home/guillem/fragile/docs/source/13_fractal_set/13_A_fractal_set.md`
**Date**: 2025-10-10
**Purpose**: Complete extraction of all mathematical objects for insertion into reference document

---

## Section 0: Foundations

### 0.1 State Space

#### Definition: State Space Manifold
- **Label**: `def-d-state-space-manifold`
- **Type**: Definition
- **Section**: 0.1 - The spacetime arena
- **Tags**: `state-space`, `riemannian-manifold`, `metric-tensor`, `fitness-landscape`, `foundations`

**Statement**:
The **state space** is a Riemannian manifold $(\mathcal{X}, G)$ where:
- $\mathcal{X} \subseteq \mathbb{R}^d$: The configuration space (positions accessible to walkers)
- $G = G(x, S)$: The metric tensor induced by the fitness landscape Hessian

For the Adaptive Gas, the metric tensor is:

$$
G_{\text{reg}}(x, S) = (H_{\Phi}(x, S) + \epsilon_\Sigma I)^{-1},
$$

where $H_{\Phi}$ is the regularized Hessian of the fitness potential and $\epsilon_\Sigma > 0$ ensures uniform ellipticity.

We denote:
- ${\rm dVol}_G$: The volume element induced by $G$
- $\mathcal{X}_{\text{valid}} \subset \mathcal{X}$: The valid domain where walkers remain alive
- Time parameter $t \in [0, T]$: Discrete algorithmic timesteps indexed by $t = n\Delta t$

**Standing assumption**: $\mathcal{X}$ is a bounded domain with smooth boundary, and the metric $G$ is uniformly elliptic.

**Related results**: Connects to Chapter 7 (Adaptive Gas), ensures well-posed SDE dynamics

---

### 0.2 Walker Episodes

#### Definition: Walker Status and Episodes
- **Label**: `def-d-walker-status-episodes`
- **Type**: Definition
- **Section**: 0.2 - Walker episodes and algorithmic log
- **Tags**: `walker`, `episode`, `trajectory`, `birth-death`, `survival-status`

**Statement**:
A **walker** at timestep $t$ is a tuple $(x_t, v_t, s_t)$ where:
- $x_t \in \mathcal{X}$: Position in configuration space
- $v_t \in \mathbb{R}^d$: Velocity vector
- $s_t \in \{0,1\}$: Survival status (1 = alive, 0 = dead/absorbed)

An **episode** $e$ is a maximal contiguous alive interval $[t^{\rm b}_e, t^{\rm d}_e)$ of a walker's trajectory. The episode comprises:
- **Trajectory**: $\gamma_e : [t^{\rm b}_e, t^{\rm d}_e) \to \mathcal{X}$
- **Birth event**: $(t^{\rm b}_e, x_{t^{\rm b}_e})$
- **Death event**: $(t^{\rm d}_e, x_{t^{\rm d}_e})$
- **Episode duration**: $\tau_e := t^{\rm d}_e - t^{\rm b}_e$

**Status transitions:**
- $0 \to 1$ (birth): Episode begins
- $1 \to 0$ (death): Episode ends by exiting $\mathcal{X}_{\rm valid}$ or being selected for cloning

**Related results**: Foundation for CST construction

---

#### Definition: The Algorithmic Log and Episode Embedding
- **Label**: `def-d-algorithmic-log`
- **Type**: Definition
- **Section**: 0.2 - Walker episodes and algorithmic log
- **Tags**: `algorithmic-log`, `episode-embedding`, `death-position`, `canonical-embedding`

**Statement**:
The **algorithmic log** $\mathcal{L}$ is the complete record of all episodes $e \in \mathcal{E}$, storing:
1. Unique episode identifier $\text{id}(e) \in \mathbb{N}$
2. Parent identifier $\text{parent}(e) \in \mathcal{E} \cup \{\text{root}\}$
3. Birth timestep $t^{\rm b}_e$ and death timestep $t^{\rm d}_e$
4. Episode duration $\tau_e = t^{\rm d}_e - t^{\rm b}_e$
5. (Optionally) Trajectory sample $\{(t_i, x_i, v_i)\}_{i=1}^{K_e}$

The **canonical embedding** $\Phi : \mathcal{E} \hookrightarrow \mathcal{X}$ assigns:

$$
\Phi(e) := x_{t^{\rm d}_e} \quad \text{(death position)}.
$$

**Remark**: Alternative embeddings (birth position, trajectory centroid) yield different properties.

**Related results**: `def-d-cst` uses this embedding, `def-d-genealogy-causal-structure`

---

#### Definition: Genealogy and Causal Graph Structure
- **Label**: `def-d-genealogy-causal-structure`
- **Type**: Definition
- **Section**: 0.2 - Walker episodes and algorithmic log
- **Tags**: `genealogy`, `parent-child`, `ancestry`, `causal-order`, `edge-weight`

**Statement**:
The **parent-child relation** $e_i \to e_j$ holds iff:

$$
t^{\rm b}_j = t^{\rm d}_i \quad \text{and} \quad \text{parent}(e_j) = \text{id}(e_i).
$$

**Edge multiplicity**: A single episode may have multiple children, but exactly one parent (or none if root).

The **CST edge weight** is:

$$
\omega(e_i \to e_j) := \tau_i.
$$

The **ancestry relation** $e \prec e'$ is the transitive closure of $\to$:

$$
e \prec e' \iff \exists \, \text{path} \, e = e_0 \to e_1 \to \cdots \to e_k = e'.
$$

**Related results**: `def-d-cst`, `prop-cst-forest`

---

### 0.3 Foundational Axioms

#### Axiom: Temporal Causality (No Future Dependence)
- **Label**: `def-ax-locality`
- **Type**: Axiom
- **Section**: 0.3 - Foundational axioms
- **Tags**: `causality`, `no-future-dependence`, `filtration`, `algorithmic-axiom`

**Statement**:
For any timestep $t$, cloning decisions depend only on past history, not future states.

**Formally**: Let $\mathcal{F}_t := \sigma(\{(s, x_s, v_s, s_s) : s \leq t\})$ be the natural filtration. Then:

$$
P(\text{walker } i \text{ clones at step } t) = f(\mathcal{S}_t, \mathcal{H}_{\leq t}),
$$

where $\mathcal{S}_t$ is the current swarm state and $\mathcal{H}_{\leq t}$ is history up to time $t$.

**Verification**: Euclidean Gas cloning probability depends only on current fitness values and algorithmic distances (all $\mathcal{F}_t$-measurable).

**Related results**: Required for `def-d-cst` to be well-defined DAG

---

#### Axiom: Exchangeability and Label-Invariance
- **Label**: `def-ax-covariance`
- **Type**: Axiom
- **Section**: 0.3 - Foundational axioms
- **Tags**: `exchangeability`, `label-invariance`, `permutation-symmetry`, `mean-field`

**Statement**:
The law of the algorithmic log is invariant under:

1. **Walker permutations**: For any permutation $\pi : \{1, \ldots, N\} \to \{1, \ldots, N\}$,
   $$
   \mathcal{L}(\mathcal{S}_t) = \mathcal{L}(\pi(\mathcal{S}_t))
   $$

2. **Episode label permutations**: For any bijection $\sigma : \mathcal{E} \to \mathcal{E}$,
   $$
   \mathcal{L}(\mathcal{L}) = \mathcal{L}(\sigma(\mathcal{L}))
   $$

**Interpretation**: Walker exchangeability enables mean-field limits (Chapters 5-6). Episode labels are arbitrary bookkeeping.

**Related results**: Enables `def-d-ig-order-invariant-construction`

---

#### Axiom: Population-Regulated Stationarity
- **Label**: `def-ax-population-regulation`
- **Type**: Axiom
- **Section**: 0.3 - Foundational axioms
- **Tags**: `qsd`, `quasi-stationary-distribution`, `martingale`, `intensity-process`

**Statement**:
There exists a **quasi-stationary distribution (QSD)** $\mu$ on living swarms $\Sigma_N^{\text{alive}}$ such that:

1. **Conditional stationarity**:
   $$
   \lim_{t \to \infty} \mathcal{L}(\mathcal{S}_t \mid \tau_{\text{ext}} > t) = \mu
   $$

2. **Previsible compensator**: The birth counting process has compensator $\Lambda_t(A)$:
   $$
   M_t(A) := N(A \times [0,t]) - \int_0^t \Lambda_s(A) \, ds
   $$
   is a martingale w.r.t. $\mathcal{F}_t$.

3. **Locally integrable intensity**: For compact $K \subset M$,
   $$
   \int_0^T \Lambda_t(K) \, dt < \infty \quad \text{a.s. for all } T < \infty
   $$

**Verification**: Chapter 3, Section 3.3 proves cloning operator with fitness $\Phi(x) = -U(x)$ induces QSD.

**Related results**: Chapter 3 Theorem 3.4, `def-d-discrete-function-spaces` (node weights use QSD density)

---

#### Axiom: Finite Range or Rapid Decay
- **Label**: `def-ax-finite-range`
- **Type**: Axiom
- **Section**: 0.3 - Foundational axioms
- **Tags**: `cloning-kernel`, `finite-range`, `rapid-decay`, `locality`

**Statement**:
The cloning kernel $Q_\delta(x' | x)$ satisfies:

1. **Finite range**: $\exists R < \infty$ such that $Q_\delta(x' | x) = 0$ for $d_g(x, x') > R$, or
2. **Rapid decay**: $Q_\delta(x' | x) \leq C e^{-\alpha d_g(x, x')^\beta}$ for $C, \alpha, \beta > 0$

where $d_g$ is geodesic distance on $(M, g)$.

**Interpretation**: Children born "near" parents—no arbitrarily large spatial jumps.

**Verification**: Chapter 3, Definition 3.3 specifies Gaussian kernel with bandwidth $\delta$.

**Related results**: Ensures `def-d-information-graph` edges connect spatially proximate walkers

---

#### Axiom: Local Finiteness (Non-Explosion)
- **Label**: `def-ax-local-finiteness`
- **Type**: Axiom
- **Section**: 0.3 - Foundational axioms
- **Tags**: `local-finiteness`, `non-explosion`, `bounded-births`, `point-process`

**Statement**:
For any compact $K \subset M$ and finite time $[0,T]$,

$$
\mathbb{E}\big[N(K \times [0,T])\big] < \infty,
$$

where $N(A)$ counts births in spacetime region $A$.

**Interpretation**: Algorithm does not produce infinite births in finite spacetime volume.

**Verification**: Follows from finite population size $N$ and bounded cloning rate. Births bounded by $O(NT)$.

**Related results**: Required for `def-d-discrete-function-spaces` to be well-defined

---

## Section 2: Causal Spacetime Tree (CST)

#### Definition: Causal Spacetime Tree (CST)
- **Label**: `def-d-cst`
- **Type**: Definition
- **Section**: 2.1 - Definition and graph structure
- **Tags**: `cst`, `directed-graph`, `genealogy`, `edge-weight`, `ancestry`

**Statement**:
The **Causal Spacetime Tree** is the directed graph

$$
\mathcal{T} := (\mathcal{E}, \to, \omega),
$$

where:
- $\mathcal{E}$ is the set of episodes (nodes)
- $\to \subset \mathcal{E} \times \mathcal{E}$ is the parent-child relation (edges) from `def-d-genealogy-causal-structure`
- $\omega : \to \to \mathbb{R}_{>0}$ assigns to each edge $e_i \to e_j$ the weight $\omega(e_i \to e_j) = \tau_i$

The **transitive closure** of $\to$ is the ancestry partial order $\prec$ on $\mathcal{E}$:

$$
(\mathcal{E}, \prec) \quad \text{with} \quad e \prec e' \iff \exists \, \text{directed path from } e \text{ to } e' \text{ in } \mathcal{T}.
$$

**Related results**: `prop-cst-forest`, `def-d-cst-observables`, `def-d-retarded-propagator`

---

#### Proposition: CST is a Forest
- **Label**: `prop-cst-forest`
- **Type**: Proposition
- **Section**: 2.1 - Definition and graph structure
- **Tags**: `dag`, `tree`, `forest`, `acyclicity`, `graph-structure`

**Statement**:
The CST $\mathcal{T} = (\mathcal{E}, \to)$ is a **directed acyclic graph (DAG)**. Moreover:
1. If all walkers originate from a single initial condition, $\mathcal{T}$ is a **tree** (connected, with one root)
2. If walkers have $k$ distinct initial ancestors, $\mathcal{T}$ is a **forest** of $k$ trees

**Proof**:
1. **Acyclicity**: Suppose $e \prec e'$ and $e' \prec e$. Then concatenating paths yields:
   $$
   t^{\rm b}_e < t^{\rm d}_e = t^{\rm b}_{e_1} < \cdots < t^{\rm d}_{e'_n} = t^{\rm b}_e,
   $$
   contradicting $t^{\rm b}_e < t^{\rm b}_e$. Thus $\mathcal{T}$ is acyclic.

2. **Tree/forest structure**: Each episode has at most one parent. Acyclicity + unique parent ⇒ tree structure from each root. ∎

**Edge count**: Tree with $|\mathcal{E}|$ nodes has $|\mathcal{E}| - 1$ edges. Forest with $k$ components has $|\mathcal{E}| - k$ edges.

**Related results**: Foundation for `def-d-retarded-propagator` causality

---

#### Definition: CST Discrete Causal Structure
- **Label**: `def-d-cst-causal-structure`
- **Type**: Definition
- **Section**: 2.3 - CST faithfulness and causal set properties
- **Tags**: `causal-order`, `partial-order`, `finite-population`, `embedding`

**Statement**:
Define the relation $\prec$ on $\mathcal{E}$ by:

$$
e \prec e' \iff t^{\rm d}_e \leq t^{\rm b}_{e'},
$$

where $\Phi(e)$ is the death location of episode $e$ in spacetime.

**Provable properties**:

1. **Partial order**: $(\mathcal{E}, \prec)$ is a partial order (reflexive, antisymmetric, transitive). CST $\mathcal{T}$ is a DAG.

2. **Finite population**: For any run with population $N$ and $T$ timesteps, $|\mathcal{E}| \leq NT$.

3. **Order-preserving embedding**: By construction,
   $$
   e \prec e' \implies t^{\rm d}_e \leq t^{\rm b}_{e'}.
   $$

**Related results**: `conj-cst-faithful-embedding`, causal set theory connections

---

#### Conjecture: CST Continuum Limit Properties
- **Label**: `conj-cst-faithful-embedding`
- **Type**: Conjecture
- **Section**: 2.3 - CST faithfulness and causal set properties
- **Tags**: `continuum-limit`, `local-finiteness`, `volume-faithful`, `poisson-scaling`

**Statement**:
If discrete Euclidean Gas structures converge to continuum objects as $N \to \infty$, $\Delta t \to 0$, the limiting CST may satisfy:

1. **Local finiteness**: For temporal interval $I(t_1,t_2) := [t_1, t_2] \subset M$,
   $$
   \#\{e \in \mathcal{E} : \Phi(e) \in I(p,q)\} < \infty.
   $$

2. **Volume-faithful embedding**: Episode counts may have Poisson-like scaling,
   $$
   \#\{e : \Phi(e) \in A\} \sim \rho \cdot \text{Vol}_g(A)
   $$
   for measurable $A \subset M$, where $\rho$ is emergent density parameter.

**Open question**: Characterizing limiting distribution of episodes requires analysis beyond discrete algorithmic construction.

**Related results**: Connection to locally finite causal sets (Bombelli 1987, Surya 2019)

---

#### Definition: CST Observables
- **Label**: `def-d-cst-observables`
- **Type**: Definition
- **Section**: 2.4 - CST observables
- **Tags**: `proper-time`, `volume-estimator`, `spatial-slices`, `antichain`, `cauchy-surface`

**Statement**:
From CST $\mathcal{T} = (\mathcal{E}, \to, \omega)$, define:

1. **Proper-time estimator**: For $e, e' \in \mathcal{E}$ with $e \prec e'$,
   $$
   \mathsf{T}(e, e') := \max_{\text{paths } \pi : e \leadsto e'} \sum_{(e_i \to e_{i+1}) \in \pi} \omega(e_i \to e_{i+1}),
   $$
   where maximum is over directed paths from $e$ to $e'$. Estimates maximal proper time along ancestral trajectories.

2. **Volume estimator**: For measurable $R \subset M$,
   $$
   \widehat{{\rm Vol}}(R) := \frac{1}{\rho} \cdot \#\{e : \Phi(e) \in R\},
   $$
   where $\rho$ is QSD density. Estimates ${\rm Vol}_g(R)$ in densification limit.

3. **Spatial slices**: Set $\mathcal{S} \subset \mathcal{E}$ is a **spatial slice** (or **antichain**) if:
   $$
   \forall \, e, e' \in \mathcal{S}, \, e \neq e' \implies e \not\prec e' \text{ and } e' \not\prec e.
   $$
   Maximal antichains correspond to Cauchy surfaces in continuum limit.

**Related results**: `conj-cst-faithful-embedding`, continuum limit analysis

---

## Section 3: Information Graph (IG)

#### Definition: Information Graph (IG)
- **Label**: `def-d-information-graph`
- **Type**: Definition
- **Section**: 3.2 - Definition of Information Graph
- **Tags**: `ig`, `interaction-graph`, `selection-coupling`, `field-interaction`, `undirected-graph`

**Statement**:
The **Information Graph** is an undirected weighted graph

$$
\mathcal{G} := (\mathcal{E}, \sim, w),
$$

where:
- $\mathcal{E}$ is the episode set (same nodes as CST)
- $\sim \subset \mathcal{E} \times \mathcal{E}$ is the **interaction edge relation**
- $w : \sim \to \mathbb{R}_{>0}$ assigns **exchange intensity weights**

**Edge criterion**: Episodes $e_i, e_j$ satisfy $e_i \sim e_j$ iff they exhibit **direct algorithmic dependency**, via:

1. **Selection coupling**: Both participate in same selection/cloning event. At time $t \in [t_i^{\rm b}, t_i^{\rm d}) \cap [t_j^{\rm b}, t_j^{\rm d})$:
   - Both $e_i, e_j$ alive at $t$
   - Fitness values $\Phi(x_i(t))$, $\Phi(x_j(t))$ contribute to cloning probability (Chapter 3, Definition 3.2)

2. **Field interaction**: Trajectories $\gamma_i, \gamma_j$ sufficiently close. Using interaction kernel $\kappa : M \times M \to \mathbb{R}_{\geq 0}$:
   $$
   \int_{t_i^{\rm b}}^{t_i^{\rm d}} \int_{t_j^{\rm b}}^{t_j^{\rm d}} \kappa(\gamma_i(\tau), \gamma_j(\tau')) \, d\tau \, d\tau' > \epsilon
   $$

3. **Global constraint coupling**: Episodes share contribution to global constraint (e.g., total charge, momentum). All alive episodes at time $t$ mutually connected.

**Remark**: In Euclidean Gas, criterion (1) is primary—cloning operator creates selection coupling, yielding **time-sliced clique structure**.

**Related results**: `def-d-ig-edge-weights`, `prop-ig-connectivity`, `def-d-fractal-set`

---

#### Definition: IG Edge Weights
- **Label**: `def-d-ig-edge-weights`
- **Type**: Definition
- **Section**: 3.2 - Definition of Information Graph
- **Tags**: `edge-weights`, `exchange-intensity`, `order-invariant`, `coupling-strength`

**Statement**:
For each edge $e_i \sim e_j$, define **exchange intensity weight**

$$
w_{ij} := \int_{\mathbb{R}} \kappa_{ij}(t) \, \mathbf{1}_{T_{ij}}(t) \, dt,
$$

where:
- $T_{ij} := [t_i^{\rm b}, t_i^{\rm d}) \cap [t_j^{\rm b}, t_j^{\rm d})$ is **temporal overlap interval**
- $\kappa_{ij}(t) : \mathbb{R} \to \mathbb{R}_{>0}$ is **time-dependent coupling strength**, constructed from **order-invariant CST features**

**Canonical choices for $\kappa_{ij}(t)$**:

1. **Interval cardinality weight**: Let $I(e_i, e_j) := \{e \in \mathcal{E} : \Phi(e_i) \prec \Phi(e) \prec \Phi(e_j)\}$. Define:
   $$
   \kappa_{ij}(t) := \exp\big(-\alpha \, |I(e_i, e_j)|\big)
   $$
   Stronger coupling to "causally close" episodes (few intervening episodes).

2. **Layer count weight**: Partition $\mathcal{E}$ into layers $L_0, L_1, \ldots$ where $L_{k+1} = \{e : \exists \, e' \in L_k \text{ with } e' \to e\}$:
   $$
   \kappa_{ij}(t) := \exp\big(-\beta \, |L(e_i) - L(e_j)|\big)
   $$

3. **Maximal chain weight**: Let $\ell(e_i, e_j)$ be longest directed path in CST connecting $e_i, e_j$:
   $$
   \kappa_{ij}(t) := (\ell(e_i, e_j) + 1)^{-\gamma}
   $$

**Key property**: All three choices are **order-invariant** (depend only on $\prec$, not labels/coordinates).

**Related results**: `def-d-order-invariant-functionals`, `def-d-ig-order-invariant-construction`

---

#### Definition: Order-Invariant Functionals
- **Label**: `def-d-order-invariant-functionals`
- **Type**: Definition
- **Section**: 3.2 - Definition of Information Graph
- **Tags**: `order-invariance`, `causal-automorphism`, `intrinsic-structure`, `lorentz-invariance`

**Statement**:
A functional $F : \mathcal{C} \to \mathbb{R}$ on CST configurations $\mathcal{C}$ is **order-invariant** if:

$$
F(\psi(\mathcal{T})) = F(\mathcal{T})
$$

for every **causal automorphism** $\psi : (\mathcal{E}, \prec) \to (\mathcal{E}', \prec')$ (order-preserving bijection).

**Examples of order-invariant functionals**:
1. Interval cardinalities $|I(e, e')|$
2. Longest chain lengths $\ell(e, e')$
3. Antichain sizes $\max |\mathcal{A}|$ for $\mathcal{A} \subset \mathcal{E}$ antichain
4. Causal diamond counts $\#\{e : e_1 \prec e \prec e_2\}$

**Examples that are NOT order-invariant**:
1. Episode death times $t^{\rm d}_e$ (depend on coordinate choice)
2. Proper lifetimes $\tau_e$ (depend on metric)
3. Spatial distances $d_g(\Phi(e), \Phi(e'))$ (depend on embedding)

**Key principle**: Order-invariant functionals capture intrinsic graph structure independent of coordinates.

**Related results**: Foundation for `conj-ig-lorentz-invariance`

---

#### Definition: IG Order-Invariant Construction
- **Label**: `def-d-ig-order-invariant-construction`
- **Type**: Definition
- **Section**: 3.3 - Temporal symmetry theorem
- **Tags**: `ig-construction`, `label-invariance`, `bhs-compatibility`, `unbounded-valency`

**Statement**:
The IG $\mathcal{G} = (\mathcal{E}, \sim, w)$ can be constructed using:
1. Edge relation $\sim$ as functional of **causal order data** from CST
2. Weights $w_{ij}$ as functionals of **selection coupling strength**

**Provable properties**:

**(Label-invariance)**: For episode relabeling $\sigma : \mathcal{E} \to \mathcal{E}$ (bijection),
$$
\sigma(\mathcal{G}) \cong \mathcal{G}.
$$

**(BHS compatibility)**: IG construction avoids BHS no-go theorem (BHS2009) by:
1. Allowing **unbounded valency** (episodes connect to arbitrarily many others)
2. Not selecting finite neighborhoods via geometric rules
3. Basing edges on **algorithmic coupling** from cloning mechanism

**Related results**: `conj-ig-lorentz-invariance`, gauge theory (Chapter 9)

---

#### Conjecture: IG Lorentz Invariance in Continuum Limit
- **Label**: `conj-ig-lorentz-invariance`
- **Type**: Conjecture
- **Section**: 3.3 - Temporal symmetry theorem
- **Tags**: `lorentz-invariance`, `continuum-limit`, `spacetime-isometry`, `covariance`

**Statement**:
If discrete IG structures converge as $N \to \infty$, $\Delta t \to 0$, the limiting law may be Lorentz invariant:

For spacetime isometry $\psi : M \to M$ (i.e., $\psi^* g = g$),
$$
\mathcal{L}(\mathcal{G}_{\infty}) = \mathcal{L}(\psi_* \mathcal{G}_{\infty}),
$$
where $\mathcal{G}_{\infty}$ denotes limiting continuum object.

**Heuristic**: If IG edges depend only on causal order (time-translation invariant) and scalar fields (covariant), discrete structure should inherit these symmetries.

**Open question**: Proving this requires showing discrete algorithmic construction converges and limit respects spacetime symmetries.

**Potential implications**: No artificial frame dependence, gauge theory compatibility, unbounded connectivity.

**Related results**: `def-d-order-invariant-functionals`, gauge theory (Section 7, Chapter 9)

---

#### Proposition: IG Connectivity in the Selection Regime
- **Label**: `prop-ig-connectivity`
- **Type**: Proposition
- **Section**: 3.4 - IG connectivity and percolation
- **Tags**: `connectivity`, `percolation`, `clique-structure`, `graph-expansion`

**Statement**:
Assume:
1. Cloning operator induces selection coupling (Definition `def-d-information-graph`, criterion 1)
2. Selection rate $\lambda_{\text{sel}} > 0$ (at least one cloning event per unit time)
3. Population size $N \geq N_0$ for threshold $N_0 \sim O(\log(1/\delta))$

Then with probability at least $1 - \delta$, the IG $\mathcal{G}$ is **connected**: there exists a path of IG edges between any two episodes alive at overlapping times.

**Proof (sketch)**:
1. Each cloning event at time $t$ creates **clique** (complete subgraph) among all alive episodes at $t$
2. Over interval $[0, T]$, there are $\sim \lambda_{\text{sel}} T$ cloning events
3. Episodes alive across multiple cloning events bridge cliques
4. **Percolation argument**: For $N$ large enough, union of cliques forms giant connected component with high probability
5. Standard bond percolation theory (Grimmett 1999) gives $\mathbb{P}[\mathcal{G} \text{ connected}] \geq 1 - \delta$ for $N \geq C \log(1/\delta)$ ∎

**Related results**: `prop-info-propagation` (enables information propagation)

---

## Section 4: Fractal Set and Information Flow

#### Definition: Fractal Set
- **Label**: `def-d-fractal-set`
- **Type**: Definition
- **Section**: 4.1 - Definition of Fractal Set
- **Tags**: `fractal-set`, `mixed-graph`, `cst-union-ig`, `weighted-graph`, `composite-structure`

**Statement**:
The **Fractal Set** of an algorithmic run is the composite graph

$$
\mathcal{F} := (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}}, \omega \cup w),
$$

where:
- $\mathcal{E}$ is the episode set (common node set)
- $E_{\text{CST}} = \{(e, e') : e \to e'\}$ is CST edges (directed)
- $E_{\text{IG}} = \{(e, e') : e \sim e'\}$ is IG edges (undirected)
- $\omega \cup w$ assigns weights: CST edges get $\omega(e \to e')$, IG edges get $w(e \sim e')$

**Notation**: Often write $\mathcal{F} = \mathcal{T} \cup \mathcal{G}$.

**Edge types**:
1. **CST edges** ($\to$): **Timelike/causal/inheritance channels**
   - Direction: Past → Future
   - Interpretation: Information flows along genealogical lines
   - Weight: Proper lifetime (propagator "length")

2. **IG edges** ($\sim$): **Spacelike/acausal/interaction channels**
   - Direction: Undirected (or bidirectional)
   - Interpretation: Contemporaneous coupling (selection, fields, constraints)
   - Weight: Exchange intensity (interaction "strength")

**Graph properties**:
- **Mixed graph**: Combines directed (CST) and undirected (IG) edges
- **Weighted**: Both edge types carry physical weights
- **Locally finite**: Each episode has finite CST valency but potentially unbounded IG valency
- **Connected** (high probability): `prop-ig-connectivity` + CST tree structure ⇒ giant component

**Related results**: `prop-info-propagation`, `def-d-discrete-function-spaces`, field theory (Section 7)

---

#### Proposition: Information Propagation on Discrete Fractal Set
- **Label**: `prop-info-propagation`
- **Type**: Proposition
- **Section**: 4.2 - Information propagation on $\mathcal{F}$
- **Tags**: `information-propagation`, `inheritance`, `mixing`, `connectivity`, `unitarity`

**Statement**:
For discrete Fractal Set $\mathcal{F}$ with:
1. **Bounded population**: Fixed $N$ walkers maintained by cloning
2. **Positive cloning rate**: At least one cloning event per timestep (high probability)
3. **Finite walker speeds**: Walker velocities bounded

The following discrete propagation properties hold:

**(1) Inheritance along CST edges**:
Let $f : \mathcal{E} \to \mathbb{R}$ be scalar observable. For edge $e \to e'$:
$$
\mathbb{E}[f(e') \mid \mathcal{F}_{t^{\rm d}_e}] = \mathcal{K}(f(e)) + \text{noise},
$$
where $\mathcal{K}$ is kinetic evolution operator (Chapter 4), noise is stochastic increment from kinetic SDE.

**Interpretation**: Children inherit information from parents, modulated by stochastic evolution.

**(2) Mixing along IG edges**:
For $e_i \sim e_j$ (IG edge), fitness values are **correlated**:
$$
\text{Cov}(\Phi(x_i(t)), \Phi(x_j(t))) > 0 \quad \text{for } t \in T_{ij}.
$$
Correlation propagates to descendants:
$$
\text{Cov}(f(e'_i), f(e'_j)) \geq \alpha \cdot w_{ij} \cdot \text{Cov}(f(e_i), f(e_j)) \cdot e^{-\gamma (\tau_i + \tau_j)}
$$

**Interpretation**: Interactions (IG edges) create cross-lineage correlations inherited by descendants—discrete analog of **field-mediated entanglement** in QFT.

**(3) Connectivity and information conservation**:
With probability $\geq 1 - \delta$ (for $N \geq N_0(\delta)$), $\mathcal{F}$ is **connected**: for any two episodes with overlapping lifetimes, there exists **mixed path**
$$
\pi : e = e_0 \xrightarrow{\to \text{ or } \sim} e_1 \xrightarrow{\to \text{ or } \sim} \cdots \xrightarrow{\to \text{ or } \sim} e_k = e'
$$
alternating CST and IG edges.

**Information redistribution**: Localized perturbation $\delta f$ at $e_0$ propagates:
- **Causally** (CST edges) to descendants
- **Acausally** (IG edges) to contemporaneous episodes
- **Recursively** (alternating CST/IG) to entire connected component

**Conjectural continuum limit**: If discrete structures converge, redistribution yields unitary-like evolution.

**Proof**: See Section 4.2.1 (uses Karatzas-Shreve stochastic process theory, Chung graph theory). ∎

**Related results**: Discrete analog of quantum unitarity, foundation for QFT (Section 7)

---

## Section 5: Projection to Continuum Physics

#### Definition: Mollified Empirical Fields
- **Label**: `def-d-empirical-fields`
- **Type**: Definition
- **Section**: 5.1 - Empirical fields from Fractal Set
- **Tags**: `mollification`, `empirical-fields`, `density`, `velocity-field`, `kinetic-energy`

**Statement**:
Let $\mathsf{K}_\varepsilon : M \times M \to \mathbb{R}_{\geq 0}$ be **spacetime mollification kernel** with bandwidth $\varepsilon > 0$, satisfying:

1. **Smoothness**: $\mathsf{K}_\varepsilon \in C^\infty_c(M \times M)$
2. **Normalization**: $\int_M \mathsf{K}_\varepsilon(x, y) \, {\rm dVol}_g(y) = 1$ for all $x \in M$
3. **Scaling**: $\mathsf{K}_\varepsilon(x, y) = \varepsilon^{-d} K(d_g(x, y) / \varepsilon)$
4. **Delta convergence**: As $\varepsilon \to 0$, $\mathsf{K}_\varepsilon(x, \cdot) \to \delta_x$

For episode $e \in \mathcal{E}$ with trajectory $\gamma_e : [t^{\rm b}_e, t^{\rm d}_e) \to \mathcal{X}$ and velocity $v_e(t) = \dot{\gamma}_e(t)$:

1. **Spatial density**:
   $$
   \rho_\varepsilon(x) := \frac{1}{N} \sum_{e \in \mathcal{E}} \int_{t^{\rm b}_e}^{t^{\rm d}_e} \mathsf{K}_\varepsilon\big(x, \gamma_e(t)\big) \, dt
   $$

2. **Velocity field**:
   $$
   u_\varepsilon(x) := \frac{1}{\rho_\varepsilon(x)} \frac{1}{N} \sum_{e \in \mathcal{E}} \int_{t^{\rm b}_e}^{t^{\rm d}_e} v_e(t) \, \mathsf{K}_\varepsilon\big(x, \gamma_e(t)\big) \, dt
   $$

3. **Kinetic energy density**:
   $$
   E_\varepsilon(x) := \frac{1}{N} \sum_{e \in \mathcal{E}} \int_{t^{\rm b}_e}^{t^{\rm d}_e} \frac{1}{2}\|v_e(t)\|^2 \, \mathsf{K}_\varepsilon\big(x, \gamma_e(t)\big) \, dt
   $$

**Interpretation**: As $\varepsilon \to 0$ and $N \to \infty$, these may converge to mean-field density $f(t,x,v)$ from Chapter 5.

**Related results**: Chapter 5 mean-field limits, continuum field theory

---

## Section 6: Discrete Differential Geometry

#### Definition: Discrete Function Spaces
- **Label**: `def-d-discrete-function-spaces`
- **Type**: Definition
- **Section**: 6.1 - Discrete function spaces and inner products
- **Tags**: `function-spaces`, `cochains`, `inner-products`, `node-weights`, `edge-weights`

**Statement**:
Define function spaces on $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}}, \omega \cup w)$:

1. **0-cochains** (functions on nodes):
   $$
   C^0(\mathcal{F}) := \{f : \mathcal{E} \to \mathbb{R}\} \cong \mathbb{R}^{|\mathcal{E}|}.
   $$

2. **1-cochains** (functions on edges):
   $$
   C^1(\mathcal{F}) := \{\alpha : E_{\text{CST}} \cup E_{\text{IG}} \to \mathbb{R}\}
   $$

**Edge orientation**: For IG edges (undirected), choose consistent orientation. For $(e, e')$ with $e < e'$, define $\alpha(e, e') = -\alpha(e', e)$.

**Inner products**: Using physical weights:

1. **Node weights**: $m_v := \rho(\Phi(v))$ where $\rho$ is QSD density

2. **Edge weights**:
   - CST edges: $w_{e \to e'} = \omega(e \to e') = \tau_e$
   - IG edges: $w_{e \sim e'} = w_{ee'}$ (exchange intensity)

$$
\langle f, g \rangle_{C^0} := \sum_{v \in \mathcal{E}} m_v \, f(v) \, g(v)
$$

$$
\langle \alpha, \beta \rangle_{C^1} := \sum_{e \in E_{\text{CST}} \cup E_{\text{IG}}} w_e \, \alpha(e) \, \beta(e)
$$

**Physical interpretation**: $m_v$ measures "volume", $w_e$ measures "length" (CST) or "strength" (IG).

**Related results**: `def-d-discrete-exterior-derivative`, `def-d-discrete-hodge-laplacians`

---

#### Definition: Discrete Exterior Derivative
- **Label**: `def-d-discrete-exterior-derivative`
- **Type**: Definition
- **Section**: 6.2 - Exterior derivative and codifferential
- **Tags**: `exterior-derivative`, `graph-gradient`, `incidence-matrix`, `differential-operator`

**Statement**:
The **exterior derivative** $d : C^0(\mathcal{F}) \to C^1(\mathcal{F})$ is the **graph gradient**, defined for $f \in C^0$ and edge $e = (u, v)$:

$$
(df)(e) := \begin{cases}
f(v) - f(u) & \text{if } e \text{ is CST edge } u \to v, \\
f(v) - f(u) & \text{if } e \text{ is IG edge } u \sim v \text{ with } u < v.
\end{cases}
$$

**Matrix form**: If nodes ordered as $v_1, \ldots, v_{|\mathcal{E}|}$ and edges as $e_1, \ldots, e_{|E|}$, then $d$ is **incidence matrix** $B \in \mathbb{R}^{|E| \times |\mathcal{E}|}$:

$$
B_{ev} = \begin{cases}
+1 & \text{if edge } e \text{ points to node } v, \\
-1 & \text{if edge } e \text{ points from node } v, \\
0 & \text{otherwise}.
\end{cases}
$$

Then $(df) = B f$ (matrix-vector product).

**Related results**: `def-d-discrete-codifferential`, `def-d-discrete-hodge-laplacians`

---

#### Definition: Discrete Codifferential
- **Label**: `def-d-discrete-codifferential`
- **Type**: Definition
- **Section**: 6.2 - Exterior derivative and codifferential
- **Tags**: `codifferential`, `adjoint-operator`, `divergence`, `graph-laplacian`

**Statement**:
The **codifferential** $d^* : C^1(\mathcal{F}) \to C^0(\mathcal{F})$ is **formal adjoint** of $d$ w.r.t. inner products:

$$
\langle d^* \alpha, f \rangle_{C^0} = \langle \alpha, df \rangle_{C^1} \quad \text{for all } \alpha \in C^1, \, f \in C^0.
$$

**Explicit formula**: For $\alpha \in C^1$ and node $v \in \mathcal{E}$,

$$
(d^* \alpha)(v) = \frac{1}{m_v} \sum_{e \ni v} w_e \, \alpha(e) \, \text{sgn}(e, v),
$$

where sum is over edges $e$ incident to $v$, and

$$
\text{sgn}(e, v) := \begin{cases}
+1 & \text{if } e \text{ points to } v, \\
-1 & \text{if } e \text{ points from } v.
\end{cases}
$$

**Matrix form**: $d^* = M^{-1} B^T W$ where:
- $M = \text{diag}(m_{v_1}, \ldots, m_{v_{|\mathcal{E}|}})$ (node weights)
- $W = \text{diag}(w_{e_1}, \ldots, w_{e_{|E|}})$ (edge weights)
- $B^T$ is transpose of incidence matrix

**Verification**: Direct computation shows adjoint property holds. ✓

**Related results**: `def-d-discrete-hodge-laplacians`

---

#### Definition: Discrete Hodge Laplacians
- **Label**: `def-d-discrete-hodge-laplacians`
- **Type**: Definition
- **Section**: 6.3 - Hodge Laplacians and convergence
- **Tags**: `hodge-laplacian`, `graph-laplacian`, `laplace-beltrami`, `discrete-calculus`

**Statement**:
Define **Hodge Laplacians**:

1. **Laplacian on 0-cochains** (functions on nodes):
   $$
   \Delta_0 := d^* d : C^0(\mathcal{F}) \to C^0(\mathcal{F}).
   $$

   **Explicit formula**: For $f \in C^0$ and node $v$,
   $$
   (\Delta_0 f)(v) = \frac{1}{m_v} \sum_{e = (v, v')} w_e \, \big(f(v) - f(v')\big)
   $$

   **Interpretation**: Discrete analog of $-\nabla^2$ (negative Laplacian on $(M, g)$). Measures how $f$ differs from weighted average over neighbors.

2. **Laplacian on 1-cochains** (functions on edges):
   $$
   \Delta_1 := d d^* + d^* d : C^1(\mathcal{F}) \to C^1(\mathcal{F}).
   $$

   **Interpretation**: Discrete Hodge Laplacian on 1-forms. Decomposition $\Delta_1 = d d^* + d^* d$ corresponds to Hodge decomposition (gradient + curl).

**Matrix forms**:
$$
\Delta_0 = M^{-1} B^T W B, \quad \Delta_1 = B M^{-1} B^T W + W^{-1} B M B^T.
$$

**Related results**: `prop-discrete-laplacian-properties`, `conj-continuum-limit-laplacian`

---

#### Proposition: Discrete Laplacian Properties
- **Label**: `prop-discrete-laplacian-properties`
- **Type**: Proposition
- **Section**: 6.3 - Hodge Laplacians and convergence
- **Tags**: `self-adjoint`, `positive-semi-definite`, `kernel`, `spectrum`, `cheeger-inequality`

**Statement**:
The discrete Laplacian $\Delta_0$ satisfies:

1. **Symmetry**: $\langle \Delta_0 f, g \rangle_{C^0} = \langle f, \Delta_0 g \rangle_{C^0}$ (self-adjoint)

2. **Non-negativity**: $\langle \Delta_0 f, f \rangle_{C^0} = \langle df, df \rangle_{C^1} \geq 0$ (positive semi-definite)

3. **Kernel**: $\ker(\Delta_0) = \{\text{constant functions on each connected component of } \mathcal{F}\}$

4. **Spectrum**: If $\mathcal{F}$ is connected, eigenvalues $0 = \lambda_0 < \lambda_1 \leq \lambda_2 \leq \cdots$ satisfy **Cheeger inequality** (Chung 1997):
   $$
   \frac{h^2}{2} \leq \lambda_1 \leq 2h,
   $$
   where $h$ is **isoperimetric constant** (graph expansion):
   $$
   h := \min_{\substack{S \subset \mathcal{E} \\ |S| \leq |\mathcal{E}|/2}} \frac{\sum_{e : e \cap S \neq \emptyset, \, e \cap S^c \neq \emptyset} w_e}{\sum_{v \in S} m_v}.
   $$

**Proof**: (1)-(3) are standard properties of graph Laplacians. (4) is discrete Cheeger inequality. ∎

**Related results**: `conj-continuum-limit-laplacian`, spectral graph theory

---

#### Conjecture: Continuum Limit of Discrete Laplacian
- **Label**: `conj-continuum-limit-laplacian`
- **Type**: Conjecture
- **Section**: 6.3 - Hodge Laplacians and convergence
- **Tags**: `continuum-limit`, `dalembertian`, `convergence`, `discrete-to-continuum`

**Statement**:
In densification limit as $N \to \infty$ and $\Delta t \to 0$, discrete Laplacian $\Delta_0$ may converge to **d'Alembertian** $\Box_g = -\nabla^\mu \nabla_\mu$ on $(M, g)$:

For smooth $\phi : M \to \mathbb{R}$, define $\phi^{\rm disc}(v) := \phi(\Phi(v))$ (restriction to CST nodes). Conjecture:

$$
\lim_{\substack{N \to \infty \\ \Delta t \to 0}} \mathbb{E}\left[\sum_{v \in \mathcal{E}} m_v \big|(\Delta_0 \phi^{\rm disc})(v) - (\Box_g \phi)(\Phi(v))\big|^2\right] = 0.
$$

**Heuristic argument**: Sum $\sum_{e \ni v} w_e (f(v) - f(v'))$ should approximate Riemann sum for $\int_{\text{nbhd}(v)} \nabla f \cdot \nabla \phi \, {\rm dVol}_g$. If discrete nodes become dense and edge weights capture local geometry, discrete operator should converge to continuum integral, which equals $-\Box_g \phi$ by integration by parts.

**Open questions**: Rigorous proof requires characterizing limiting distribution of CST nodes, controlling boundary terms, justifying mollification. See Wardetzky 2007, Hirani 2003 for related discrete exterior calculus convergence results.

**Related results**: `def-d-retarded-propagator`, continuum QFT

---

## Section 7: Propagators and Interaction Vertices

#### Definition: Discrete Retarded d'Alembertian
- **Label**: `def-d-retarded-propagator`
- **Type**: Definition
- **Section**: 7.1 - The retarded propagator on CST
- **Tags**: `retarded-propagator`, `dalembertian`, `causal-operator`, `cst-operator`

**Statement**:
Define **retarded discrete d'Alembertian** $\Box_{\text{CST}} : C^0(\mathcal{F}) \to C^0(\mathcal{F})$ by restricting Hodge Laplacian $\Delta_0$ to propagation along **CST edges only**:

$$
(\Box_{\text{CST}} f)(v) := \frac{1}{m_v} \sum_{\substack{e = (v', v) \\ e \in E_{\text{CST}}}} \omega(e) \, \big(f(v) - f(v')\big),
$$

where sum is over CST edges $e$ **pointing to** $v$ (i.e., $v' \to v$ with $v' \prec v$).

**Key property**: $\Box_{\text{CST}}$ is **causal** (retarded)—depends only on values $f(v')$ for $v' \prec v$ (ancestors of $v$).

**Alternative formulation via order-invariants**: Following Aslanbeigi-Saravani-Sorkin 2014, define using interval cardinalities:

$$
(\Box_{\text{CST}}^{\text{AS}} f)(v) := \sum_{v' \prec v} \kappa(|I(v', v)|) \, \big(f(v) - f(v')\big),
$$

where $\kappa(n) = \exp(-\alpha n)$ is decay kernel. This formulation is **order-invariant** and thus automatically temporally covariant.

**Connection**: For small $|I(v', v)| \sim O(1)$, both formulations agree. For large cardinalities, AS version decays exponentially, implementing **finite causal horizon**.

**Related results**: `def-d-retarded-green-operator`, `prop-green-operator-existence`

---

#### Definition: Retarded Green Operator
- **Label**: `def-d-retarded-green-operator`
- **Type**: Definition
- **Section**: 7.1 - The retarded propagator on CST
- **Tags**: `green-function`, `propagator`, `inverse-operator`, `retarded`

**Statement**:
The **retarded Green operator** (or **propagator**) $\mathsf{G}_{\mathcal{T}} : C^0(\mathcal{F}) \to C^0(\mathcal{F})$ is **inverse** of $\Box_{\text{CST}}$ on orthogonal complement of constants:

$$
\Box_{\text{CST}} \, \mathsf{G}_{\mathcal{T}} = \text{Id} - \Pi_0,
$$

where $\Pi_0$ is projection onto constant functions (kernel of $\Box_{\text{CST}}$).

**Retardation**: $\mathsf{G}_{\mathcal{T}}$ is **causal**: $(\mathsf{G}_{\mathcal{T}} f)(v)$ depends only on $f(v')$ for $v' \prec v$.

**Explicit form**: For episodes $v, v' \in \mathcal{E}$ with $v' \prec v$, propagator $G(v, v') := \mathsf{G}_{\mathcal{T}}(\delta_{v'})(v)$ (Green's function from $v'$ to $v$):

$$
G(v, v') = \sum_{\text{paths } \pi : v' \leadsto v} \prod_{e \in \pi} \frac{\omega(e)}{m_{\text{target}(e)}} \cdot W_\pi,
$$

where sum is over directed paths in CST from $v'$ to $v$, and $W_\pi$ is combinatorial weight factor.

**Physical interpretation**: $G(v, v')$ is **amplitude** for information/excitation to propagate from episode $v'$ to $v$ along genealogical tree.

**Related results**: `prop-green-operator-existence`, `def-p-diagrammatics`

---

#### Proposition: Existence and Uniqueness of Retarded Green Operator
- **Label**: `prop-green-operator-existence`
- **Type**: Proposition
- **Section**: 7.1 - The retarded propagator on CST
- **Tags**: `existence`, `uniqueness`, `greens-identity`, `causality`, `positive-definite`

**Statement**:
The retarded Green operator $\mathsf{G}_{\mathcal{T}}$ exists and is unique, satisfying:

1. **Causality**: $G(v, v') = 0$ if $v' \not\prec v$
2. **Green's identity**: $\Box_{\text{CST}} G(\cdot, v') = \delta_{v'} - \bar{\delta}$ where $\bar{\delta} = 1/|\mathcal{E}|$ (constant)
3. **Symmetry**: $G(v, v') = G(v', v)$ (for appropriately defined symmetric version)
4. **Positive definiteness**: $\sum_{v, v'} G(v, v') f(v) f(v') \geq 0$ for all $f \in C^0$

**Proof (sketch)**:
1. **Existence**: CST is DAG (`prop-cst-forest`), so $\Box_{\text{CST}}$ is upper-triangular matrix (after ordering by causal order). Restriction to orthogonal complement of constants is invertible (eigenvalues $\lambda_i > 0$ by positive semi-definiteness). Thus $\mathsf{G}_{\mathcal{T}}$ exists.

2. **Causality**: By construction, $G(v, v')$ defined via paths $v' \leadsto v$ in CST, which exist only if $v' \prec v$. ✓

3. **Green's identity**: Follows from $\Box_{\text{CST}} \mathsf{G}_{\mathcal{T}} = \text{Id} - \Pi_0$ by applying to $\delta_{v'}$. ✓

4. **Symmetry**: Naive Green's function not symmetric due to CST directionality. **Symmetrized version** $G_{\text{sym}}(v, v') := \frac{1}{2}(G(v, v') + G(v', v))$ is analog of Feynman propagator in QFT. ✓

5. **Positive definiteness**: Follows from $\langle f, \mathsf{G}_{\mathcal{T}} f \rangle_{C^0} = \langle \Box_{\text{CST}}^{-1} f, f \rangle_{C^0} \geq 0$. ✓ ∎

**Remark on continuum limit**: As $\rho \to \infty$, `conj-continuum-limit-laplacian` implies $G(v, v')$ converges to **retarded Green's function** $G_{\text{ret}}(x, y)$ solving $\Box_g G_{\text{ret}}(\cdot, y) = \delta_y$ with $G_{\text{ret}}(x, y) = 0$ for $x \not\in J^+(y)$.

**Related results**: `def-d-retarded-propagator`, `def-p-diagrammatics`, continuum QFT

---

#### Definition: Discrete Interaction Vertex Functional
- **Label**: `def-d-interaction-vertex`
- **Type**: Definition
- **Section**: 7.2 - Interaction vertices on IG
- **Tags**: `interaction-vertex`, `ig-functional`, `coupling`, `gauge-theory`

**Statement**:
For IG edges $e \sim e'$, define **interaction vertex functional** $\mathsf{V}_{\mathcal{G}} : C^0(\mathcal{F})^{\otimes k} \to \mathbb{R}$:

**2-point interaction** (simplest case):
$$
\mathsf{V}_{\mathcal{G}}^{(2)}[\phi] := \sum_{e \sim e'} w_{ee'} \, \phi(e) \, \phi(e'),
$$
where $\phi \in C^0(\mathcal{F})$ is scalar field on episodes.

**k-point interaction** (higher-order):
For cliques $C = \{e_1, \ldots, e_k\} \subset \mathcal{E}$ (complete subgraphs of IG):
$$
\mathsf{V}_{\mathcal{G}}^{(k)}[\phi] := \sum_{\text{cliques } C} W_C \prod_{i \in C} \phi(e_i),
$$
where $W_C$ is clique weight, e.g., $W_C = \prod_{i < j \in C} w_{e_i e_j}^{1/(k-1)}$ (geometric mean).

**Gauge theory generalization**: For gauge fields $A^\mu : E_{\text{IG}} \to \mathbb{R}^d$ (Lie algebra valued):
$$
\mathsf{V}_{\mathcal{G}}^{\text{gauge}}[A] := \sum_{\text{plaquettes } p} W_p \, \text{Tr}\big[U_p\big],
$$
where $U_p = \exp(\oint_{\partial p} A)$ is holonomy around plaquette $p$, and $W_p$ is plaquette weight.

**Physical interpretation**:
- $\mathsf{V}_{\mathcal{G}}^{(2)}$: Scalar field interaction ($\phi^4$ theory vertex)
- $\mathsf{V}_{\mathcal{G}}^{(k)}$: Higher-point interactions ($\phi^6$, $\phi^8$)
- $\mathsf{V}_{\mathcal{G}}^{\text{gauge}}$: Non-abelian gauge field interaction (Yang-Mills action)

**Related results**: `def-p-diagrammatics`, Chapter 9 (Yang-Mills), Appendix B (plaquettes)

---

#### Proposition: Feynman Diagrams on the Fractal Set
- **Label**: `def-p-diagrammatics`
- **Type**: Proposition
- **Section**: 7.3 - Diagrammatics and correlation functions
- **Tags**: `feynman-diagrams`, `correlation-functions`, `path-integral`, `qft`

**Statement**:
Let $\phi : \mathcal{E} \to \mathbb{R}$ be scalar field on Fractal Set. **Correlation functions**

$$
\langle \phi(v_1) \cdots \phi(v_k) \rangle := \mathbb{E}_{\mu}[\phi(v_1) \cdots \phi(v_k)]
$$

(expectation w.r.t. QSD $\mu$) admit expansion as sum over **connected subgraphs** $\Gamma \subset \mathcal{F}$:

$$
\langle \phi(v_1) \cdots \phi(v_k) \rangle = \sum_{\substack{\Gamma \subset \mathcal{F} \\ \Gamma \text{ connected} \\ v_1, \ldots, v_k \in \Gamma}} W_\Gamma \prod_{e \in E_{\text{CST}}(\Gamma)} G(e) \prod_{e' \in E_{\text{IG}}(\Gamma)} V(e'),
$$

where:
- $W_\Gamma$ is combinatorial symmetry factor
- $G(e) = G(v, v')$ for CST edge $e = (v \to v')$ (propagator from `def-d-retarded-green-operator`)
- $V(e') = w_{ee'}$ for IG edge $e' = (e \sim e')$ (interaction from `def-d-interaction-vertex`)

**Diagrammatic rules**:
1. **Propagators** (lines): CST edges $\to$ carry $G(v, v')$
2. **Vertices** (interaction points): IG edges $\sim$ or cliques carry $\mathsf{V}_{\mathcal{G}}$
3. **External legs**: Correlation function arguments $v_1, \ldots, v_k$
4. **Sum over all connected diagrams** connecting external legs via propagators and vertices

**Proof (sketch)**:
Discrete analog of Feynman path integral expansion in QFT. Follows standard field theory arguments:

**Step 1**: Define generating functional $Z[J] := \mathbb{E}_\mu[\exp(\sum_v J(v) \phi(v))]$ for source $J$.

**Step 2**: Expand around classical solution $\phi_{\text{cl}}$ (minimizing action $S[\phi] = \frac{1}{2}\langle \phi, \Box_{\text{CST}} \phi \rangle - \mathsf{V}_{\mathcal{G}}[\phi] - \langle J, \phi \rangle$):
$$
Z[J] \approx e^{-S[\phi_{\text{cl}}]} \int D\phi \, e^{-\frac{1}{2}\langle \phi, \Box_{\text{CST}} \phi \rangle + \mathsf{V}_{\mathcal{G}}[\phi]}
$$

**Step 3**: Gaussian integral yields contractions (Wick's theorem), each contraction a propagator $G(v, v')$. Interaction term $\mathsf{V}_{\mathcal{G}}[\phi]$ provides vertices.

**Step 4**: Logarithm $W[J] = \log Z[J]$ generates **connected** correlation functions (cumulants). Disconnected diagrams factor out.

**Step 5**: Each term corresponds to graph $\Gamma$ with:
- Nodes = episode locations
- CST edges = propagators $G$
- IG edges = interaction vertices $\mathsf{V}_{\mathcal{G}}$

Summing over all graphs yields correlation function. ✓ ∎

**Remark on continuum limit**: As $\rho \to \infty$, discrete correlation functions converge to **continuum QFT correlation functions** (Glimm-Jaffe 1987):
$$
\langle \phi(x_1) \cdots \phi(x_k) \rangle_{\text{continuum}} = \lim_{\rho \to \infty} \langle \phi(v_1) \cdots \phi(v_k) \rangle_{\mathcal{F}},
$$
where $\Phi(v_i) \to x_i$. Discrete Feynman diagrams become continuum Feynman diagrams.

This establishes $\mathcal{F}$ as **non-perturbative lattice regularization** of relativistic QFT.

**Related results**: `def-d-retarded-green-operator`, `def-d-interaction-vertex`, Chapter 9 (gauge theory)

---

## Appendix B: Plaquettes and Fundamental Cycles

#### Definition: Fundamental Cycles
- **Label**: `def-d-fundamental-cycles`
- **Type**: Definition
- **Section**: B.2 - Fundamental Cycles
- **Tags**: `fundamental-cycles`, `cst-paths`, `closed-loops`, `cycle-space`

**Statement**:
For each IG edge $e \sim e'$, there exists unique path $\pi_T(e, e')$ in CST connecting $e$ and $e'$. The **fundamental cycle** associated with this IG edge is:

$$
C(e, e') = (e \sim e') \cup \pi_T(e, e')
$$

This is closed loop: starting at $e$, traversing IG edge to $e'$, then following CST path back to $e$.

**Related results**: `prop-cycle-basis`, `def-d-plaquettes`

---

#### Proposition: Cycle Basis
- **Label**: `prop-cycle-basis`
- **Type**: Proposition
- **Section**: B.2 - Fundamental Cycles
- **Tags**: `cycle-basis`, `algebraic-graph-theory`, `homology`

**Statement**:
The fundamental cycles $\{C(e, e') : e \sim e' \in G \setminus T\}$ form a basis for the cycle space. Every cycle in full graph $T \cup G$ can be written as sum (symmetric difference) of fundamental cycles.

**Proof sketch**: Standard result from algebraic graph theory. Number of fundamental cycles equals $|E_G| - |\mathcal{E}| + 1$ where $E_G$ is total edges. □

**Related results**: `def-d-plaquettes`, topological invariants

---

#### Definition: Plaquettes
- **Label**: `def-d-plaquettes`
- **Type**: Definition
- **Section**: B.3 - Plaquette Definition
- **Tags**: `plaquettes`, `fundamental-cycles`, `boundary`, `orientation`, `spatial-embedding`

**Statement**:
The **plaquettes** of CST/IG structure are fundamental cycles $C(e, e')$. Denote set of all plaquettes by $\mathcal{P}$.

Each plaquette has:
- A **boundary** $\partial p = \{e_1, e_2, \ldots, e_k\}$ (ordered sequence of episodes)
- An **orientation** induced by orienting IG edge consistently
- A **spatial embedding** via position embeddings $\Phi(e_i) : \mathcal{E} \to \mathcal{X}$ (death positions in configuration space)

**Related results**: `def-d-plaquette-weight`, `def-d-interaction-vertex` (gauge theory)

---

#### Definition: Plaquette Weight
- **Label**: `def-d-plaquette-weight`
- **Type**: Definition
- **Section**: B.4 - Discrete Weights
- **Tags**: `plaquette-weight`, `area`, `discrete-action`, `gauge-theory`

**Statement**:
For plaquette $p$ with boundary episodes $\{e_1, \ldots, e_k\}$, define **plaquette weight**:

$$
w(p) = \sum_{i=1}^k \tau_{e_i} \cdot d_{\text{alg}}(e_i, e_{i+1})
$$

where:
- $\tau_{e_i} = t^{\rm d}_{e_i} - t^{\rm b}_{e_i}$ is episode duration
- $d_{\text{alg}}(e_i, e_{i+1})$ is algorithmic distance between adjacent episodes in cycle
- Sum runs over all edges in boundary $\partial p$

**Alternative**: For configuration space visualization, use enclosed area in $\mathcal{X}$:

$$
A_{\mathcal{X}}(p) = \frac{1}{2}\left|\sum_{i=1}^k (x_i \times x_{i+1})\right|
$$

where $x_i = \Phi(e_i)$ are death positions projected to configuration space.

**Remark**: Plaquette definitions provide canonical closed loops in Fractal Set. Weights $w(p)$ can be used for:
1. Graph-based field theory (action contributions to cycles)
2. Information flow analysis (quantifying feedback loops in IG)
3. Topological invariants (computing homology groups of $\mathcal{F}$)

Speculative connection to gauge theory (Chapter 9) requires extending episodes with internal color charges.

**Related results**: `def-d-interaction-vertex` (gauge theory generalization), Chapter 9 (Yang-Mills)

---

## Summary Statistics

**Total mathematical objects**: 37
- **Definitions**: 24
- **Propositions**: 5
- **Theorems**: 0
- **Lemmas**: 0
- **Axioms**: 5
- **Conjectures**: 3
- **Corollaries**: 0

**Major thematic areas**:
1. **Foundations** (8 objects): State space, episodes, axioms
2. **CST Construction** (6 objects): Tree structure, causal order, observables
3. **IG Construction** (6 objects): Selection coupling, order-invariance, connectivity
4. **Fractal Set** (2 objects): Composite structure, information propagation
5. **Continuum Projection** (1 object): Mollified empirical fields
6. **Discrete Calculus** (6 objects): Function spaces, exterior derivative, Laplacians
7. **Propagators** (4 objects): Green operators, interaction vertices, Feynman diagrams
8. **Plaquettes** (4 objects): Fundamental cycles, plaquette weights

**Key continuum limit conjectures** (require future work):
- `conj-cst-faithful-embedding`: CST converges to locally finite causal set
- `conj-ig-lorentz-invariance`: IG construction preserves Lorentz symmetry
- `conj-continuum-limit-laplacian`: Discrete Laplacian converges to d'Alembertian

**Cross-references to other chapters**:
- Chapter 2: Euclidean Gas (algorithmic dynamics)
- Chapter 3: Cloning operator (QSD, fitness mechanism)
- Chapter 4: Convergence (kinetic operator)
- Chapter 5: Mean-field limits (continuum density)
- Chapter 7: Adaptive Gas (metric tensor, Hessian)
- Chapter 9: Yang-Mills (gauge theory, plaquettes)

---

## Notes for Reference Document Integration

1. **Label consistency**: All labels use `def-d-*`, `prop-*`, `conj-*` prefixes. Maintain when inserting into reference.

2. **Cross-reference format**: Uses `{prf:ref}` directive. Ensure all internal references resolve after integration.

3. **Equation labels**: Some equations have `:label:` tags (e.g., `eq-cst-proper-time`, `eq-ig-exchange-intensity`). Preserve for cross-referencing.

4. **Dependencies**: Mathematical objects build sequentially. Maintain section order when extracting subsets.

5. **Proof status**:
   - Axioms (5): Verifiable algorithmic properties
   - Definitions (24): Well-defined discrete constructions
   - Propositions (5): Proofs sketched or referenced (standard results)
   - Conjectures (3): Require continuum limit analysis (future work)

6. **Figure references**: Document includes 6 figures (`fig-episode-trajectorys`, `fig-cst-tree`, `fig-information-graph`, `fig-fractal-set`, `fig-order-invariant`). Not extracted but referenced in text.

7. **Notation conventions**:
   - $\mathcal{E}$: Episode set
   - $\mathcal{T}$: CST (Causal Spacetime Tree)
   - $\mathcal{G}$: IG (Information Graph)
   - $\mathcal{F}$: Fractal Set ($\mathcal{T} \cup \mathcal{G}$)
   - $\prec$: Causal order / ancestry relation
   - $\to$: CST edges (parent-child)
   - $\sim$: IG edges (interaction)
   - $\Phi$: Episode embedding (death position)
   - $\omega$: CST edge weights (proper time)
   - $w$: IG edge weights (exchange intensity)

---

**End of extraction**
