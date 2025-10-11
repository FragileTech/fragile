# Fractal Set Theory - Mathematical Reference

This document contains all mathematical definitions, theorems, lemmas, propositions, axioms, and corollaries from the Fractal Set theory documents (13_fractal_set_new/).

**Total mathematical objects:** 195

**Source documents:**
- 01_fractal_set.md: 32 objects
- 02_computational_equivalence.md: 14 objects
- 03_yang_mills_noether.md: 42 objects
- 04_rigorous_additions.md: 21 objects
- 05_qsd_stratonovich_foundations.md: 12 objects
- 06_continuum_limit_theory.md: 12 objects
- 07_discrete_symmetries_gauge.md: 18 objects
- 08_lattice_qft_framework.md: 24 objects
- 09_geometric_algorithms.md: 4 objects
- 10_areas_volumes_integration.md: 16 objects

---

## 01_fractal_set.md

**Objects in this document:** 32

### Definitions (16)

### Spacetime Node

**Type:** Definition
**Label:** `def-node-spacetime`
**Source:** [13_fractal_set_new/01_fractal_set.md § 1.1. Node Definition](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
A **node** $n \in \mathcal{N}$ represents a single walker at a single discrete timestep. The node set is:

$$
\mathcal{N} := \{n_{i,t} : i \in \{1, \ldots, N\}, \, t \in \{0, 1, \ldots, T\}\},

$$

where:
- $N$: Total number of walkers (fixed population size)
- $T$: Total number of timesteps
- $n_{i,t}$: Node representing walker $i$ at timestep $t$

**Node cardinality:** $|\mathcal{N}| = N \times (T+1)$ (total spacetime points)

Each node $n_{i,t}$ stores the **complete scalar state** of walker $i$ at timestep $t$ - all frame-invariant quantities from the Adaptive Gas dynamics.

---

### Node Scalar Attributes

**Type:** Definition
**Label:** `def-node-attributes`
**Source:** [13_fractal_set_new/01_fractal_set.md § 1.2. Node Scalar Data Structure](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
Each node $n_{i,t} \in \mathcal{N}$ contains the following **scalar (frame-invariant)** quantities explicitly present in the Adaptive Gas SDE:

**Identity:**
- $\text{walker\_id}(n) = i \in \{1, \ldots, N\}$: Walker index
- $\text{timestep}(n) = t \in \{0, 1, \ldots, T\}$: Discrete timestep
- $\text{node\_id}(n) = i + t \cdot N \in \mathbb{N}$: Global unique identifier

**Temporal Scalar:**
- $t(n) := t \cdot \Delta t \in \mathbb{R}_{\geq 0}$: Continuous time coordinate

**Status Scalar:**
- $s(n) \in \{0, 1\}$: Survival status (1=alive in $A_k$, 0=dead/out-of-bounds)

**Energy Scalars:**
- $E_{\text{kin}}(n) := \frac{1}{2} \|v(n)\|^2 \in \mathbb{R}_{\geq 0}$: Kinetic energy (scalar norm of velocity)
- $U(n) := U(x(n)) \in \mathbb{R}$: Confining potential energy

**Fitness Scalar:**
- $\Phi(n) := \Phi(x(n)) \in \mathbb{R}$: Fitness value (objective function)

**Virtual Reward Scalar:**
- $V_{\text{fit}}(n) := V_{\text{fit}}[f_k, \rho](x(n)) \in \mathbb{R}$: ρ-localized fitness potential

**Localized Statistical Scalars:**
- $\mu_\rho(n) := \mu_\rho[f_k, \Phi, x(n)] \in \mathbb{R}$: Localized mean fitness in ρ-neighborhood
- $\sigma_\rho(n) := \sigma_\rho[f_k, \Phi, x(n)] \in \mathbb{R}_{\geq 0}$: Localized standard deviation
- $\sigma'_\rho(n) := \sigma'_{\text{patch}}(\sigma^2_\rho(n)) \in \mathbb{R}_{> 0}$: Regularized C¹ standard deviation
- $Z_\rho(n) := Z_\rho[f_k, \Phi, x(n)] \in \mathbb{R}$: ρ-localized Z-score

**Global Algorithm Parameters (scalars, fixed throughout):**
- $\epsilon_F \in \mathbb{R}_{\geq 0}$: Adaptive force strength
- $\nu \in \mathbb{R}_{\geq 0}$: Viscosity coefficient
- $\gamma \in \mathbb{R}_{> 0}$: Friction coefficient
- $\rho \in \mathbb{R}_{> 0}$: Localization scale
- $\epsilon_\Sigma \in \mathbb{R}_{> 0}$: Diffusion regularization

---

### Causal Spacetime Tree (CST) Edges

**Type:** Definition
**Label:** `def-cst-edges`
**Source:** [13_fractal_set_new/01_fractal_set.md § 2.1. CST Edge Definition](13_fractal_set_new/01_fractal_set.md)
**Tags:** `causal-tree`, `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `spinor`

**Statement:**
The **CST edge set** encodes temporal transitions between consecutive timesteps:

$$
E_{\text{CST}} := \{(n_{i,t}, n_{i,t+1}) : i \in \{1, \ldots, N\}, \, t \in \{0, \ldots, T-1\}, \, s(n_{i,t}) = 1\},

$$

where each edge represents the continuous evolution of a single walker from timestep $t$ to $t+1$ (alive walkers only).

Each edge $(n_{i,t}, n_{i,t+1})$ stores **spinor data** representing the frame-covariant vectorial transition between the source and target nodes according to the Adaptive Gas SDE.

---

### Spinor Representation

**Type:** Definition
**Label:** `def-spinor-representation`
**Source:** [13_fractal_set_new/01_fractal_set.md § 2.2. Spinors: Frame-Covariant Vector Representation](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `metric-tensor`, `nodes`, `spinor`

**Statement:**
A **spinor** $\psi \in \mathbb{C}^{2^{[d/2]}}$ is a frame-covariant representation of a vector $v \in \mathbb{R}^d$ that transforms under the spinor representation of the rotation group $\text{SO}(d)$.

**Key property:** Under a rotation $R \in \text{SO}(d)$ of the vector:

$$
v \mapsto v' = Rv,

$$

the corresponding spinor transforms via:

$$
\psi \mapsto \psi' = S(R) \psi,

$$

where $S: \text{SO}(d) \to \text{Spin}(d)$ is the spinor representation (double cover of $\text{SO}(d)$).

**Advantages for the Fractal Set:**
1. **Frame independence**: Spinors encode vectors without reference to a preferred basis
2. **Geometric naturalness**: Spinor formalism is intrinsic to the manifold structure
3. **Covariant derivatives**: Force and velocity transitions are geometric objects
4. **Sign ambiguity**: Spinors have $\psi \equiv -\psi$, reflecting gauge freedom

**Practical encoding:** For computational purposes, we store both the vector $v \in \mathbb{R}^d$ and its spinor representation $\psi$, related by the Clifford algebra encoding:

$$
\psi = \text{exp}\left(\frac{1}{4} \sum_{j<k} \theta_{jk} \gamma_j \gamma_k\right) \psi_0

$$

where $\gamma_j$ are Clifford generators and $\theta_{jk}$ are rotation angles encoded in $v$.

---

### CST Edge Spinor Attributes

**Type:** Definition
**Label:** `def-cst-edge-attributes`
**Source:** [13_fractal_set_new/01_fractal_set.md § 2.3. CST Edge Spinor Data Structure](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `spinor`

**Statement:**
Each CST edge $(n_{i,t}, n_{i,t+1}) \in E_{\text{CST}}$ contains the following **spinor (frame-covariant)** quantities from the Adaptive Gas SDE:

**Identity:**
- $\text{walker\_id} = i$: Walker ID
- $\text{timestep} = t$: When transition occurred

**Velocity Spinors:**
- $\psi_{v,t} \in \mathbb{C}^{2^{[d/2]}}$: Velocity spinor at source timestep $t$
- $\psi_{v,t+1} \in \mathbb{C}^{2^{[d/2]}}$: Velocity spinor at target timestep $t+1$
- $\psi_{\Delta v} := \psi_{v,t+1} \ominus \psi_{v,t} \in \mathbb{C}^{2^{[d/2]}}$: Velocity increment spinor

**Position Displacement Spinor:**
- $\psi_{\Delta x} \in \mathbb{C}^{2^{[d/2]}}$: Position displacement spinor $x_{t+1} - x_t$

**Force Spinors (from Adaptive Gas SDE):**

From [../07_adaptative_gas.md](../07_adaptative_gas.md#def-hybrid-sde), the velocity evolution is:

$$
dv_i = \left[ \mathbf{F}_{\text{stable}}(x_i) + \mathbf{F}_{\text{adapt}}(x_i, S) + \mathbf{F}_{\text{viscous}}(x_i, S) - \gamma v_i \right] dt + \Sigma_{\text{reg}}(x_i, S) \circ dW_i

$$

We store spinor representations of each force component evaluated at timestep $t$:

**1. Stable Force Spinor:**
- $\psi_{\mathbf{F}_{\text{stable}}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor encoding $\mathbf{F}_{\text{stable}}(x_t) = -\nabla U(x_t)$

**2. Adaptive Force Spinor:**
- $\psi_{\mathbf{F}_{\text{adapt}}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor encoding $\mathbf{F}_{\text{adapt}}(x_t, S) = \epsilon_F \nabla_{x_i} V_{\text{fit}}[f_k, \rho](x_t)$

**3. Viscous Force Spinor:**
- $\psi_{\mathbf{F}_{\text{viscous}}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor encoding $\mathbf{F}_{\text{viscous}}(x_t, S) = \nu \sum_{j \neq i} K_\rho(x_i - x_j) (v_j - v_i)$

**4. Friction Force Spinor:**
- $\psi_{\mathbf{F}_{\text{friction}}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor encoding $-\gamma v_t$

**5. Total Drift Spinor:**
- $\psi_{\mathbf{F}_{\text{total}}} := \psi_{\mathbf{F}_{\text{stable}}} \oplus \psi_{\mathbf{F}_{\text{adapt}}} \oplus \psi_{\mathbf{F}_{\text{viscous}}} \oplus \psi_{\mathbf{F}_{\text{friction}}} \in \mathbb{C}^{2^{[d/2]}}$

**Diffusion Tensor Spinor:**
- $\psi_{\Sigma_{\text{reg}}} \in \mathbb{C}^{2^{[d/2]} \times 2^{[d/2]}}$: Spinor tensor encoding $\Sigma_{\text{reg}}(x_t, S) = (H_i(S) + \epsilon_\Sigma I)^{-1/2}$

**Stochastic Increment Spinor:**
- $\psi_{\text{noise}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor encoding the Stratonovich increment $\Sigma_{\text{reg}}(x_t, S) \circ dW_t$

**Gradient Spinors:**
- $\psi_{\nabla U} \in \mathbb{C}^{2^{[d/2]}}$: Spinor of potential gradient $\nabla U(x_t)$
- $\psi_{\nabla \Phi} \in \mathbb{C}^{2^{[d/2]}}$: Spinor of fitness gradient $\nabla \Phi(x_t)$
- $\psi_{\nabla V_{\text{fit}}} \in \mathbb{C}^{2^{[d/2]}}$: Spinor of fitness potential gradient $\nabla V_{\text{fit}}[f_k, \rho](x_t)$

**Derived Scalar Quantities:**
- $\|\Delta v\| = |\psi_{\Delta v}| \in \mathbb{R}_{\geq 0}$: Magnitude of velocity change
- $\|\Delta x\| = |\psi_{\Delta x}| \in \mathbb{R}_{\geq 0}$: Magnitude of displacement
- $\Delta t \in \mathbb{R}_{> 0}$: Timestep size (global constant)

---

### Information Graph (IG) Edges (Directed)

**Type:** Definition
**Label:** `def-ig-edges`
**Source:** [13_fractal_set_new/01_fractal_set.md § 3.1. IG Edge Definition](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `spinor`, `symmetry`

**Statement:**
The **IG edge set** encodes **directed, asymmetric selection coupling** between walkers at the **same timestep**:

$$
E_{\text{IG}} := \{(n_{i,t}, n_{j,t}) : i, j \in \{1, \ldots, N\}, \, i \neq j, \, s(n_{i,t}) = 1, \, s(n_{j,t}) = 1\},

$$

where:
- Edges are **directed**: $(n_{i,t}, n_{j,t})$ represents the influence of walker $j$ on walker $i$'s cloning decision
- Both walkers must be **alive** at timestep $t$
- Edges form a **complete directed graph** (tournament) at each timestep among alive walkers
- For $k = |\mathcal{A}(t)|$ alive walkers, there are $k(k-1)$ directed IG edges (all ordered pairs)

**Asymmetry of cloning:**

The cloning mechanism is fundamentally **antisymmetric**:
- Walker $i$ can clone **to** walker $j$ (overwriting $j$'s state)
- Walker $j$ cannot simultaneously clone to walker $i$
- The edge $(n_{i,t}, n_{j,t})$ stores the **directed cloning potential** $V_{\text{clone}}(i \to j)$

**Interpretation:** The directed edge $(n_{i,t}, n_{j,t})$ encodes:
1. **Viscous coupling**: Walker $j$'s velocity influences walker $i$ via $\mathbf{F}_{\text{viscous}}$
2. **Cloning potential**: Fitness difference $\Phi_j - \Phi_i$ determines cloning probability
3. **Companion selection**: Algorithmic distance for virtual reward computation

Each edge $(n_{i,t}, n_{j,t})$ stores **spinor data** representing the frame-covariant directional coupling from walker $j$ to walker $i$.

---

### Directed Cloning Potential

**Type:** Definition
**Label:** `def-cloning-potential`
**Source:** [13_fractal_set_new/01_fractal_set.md § 3.2. Antisymmetric Cloning Potential](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fermionic`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `symmetry`

**Statement:**
For a directed IG edge $(n_{i,t}, n_{j,t})$ from $j$ to $i$, define the **antisymmetric cloning potential**:

$$
V_{\text{clone}}(i \to j) := \Phi_j - \Phi_i

$$

This potential is **antisymmetric** under particle exchange:

$$
V_{\text{clone}}(j \to i) = \Phi_i - \Phi_j = -V_{\text{clone}}(i \to j)

$$

**Physical interpretation:**
- $V_{\text{clone}}(i \to j) > 0$: Walker $j$ is **fitter** than $i$ → increases $i$'s desire to clone from $j$
- $V_{\text{clone}}(i \to j) < 0$: Walker $j$ is **less fit** than $i$ → decreases $i$'s cloning probability
- $V_{\text{clone}}(i \to j) = 0$: Equal fitness → neutral influence

**Connection to fermionic antisymmetry:**

The antisymmetry $V_{\text{clone}}(i \to j) = -V_{\text{clone}}(j \to i)$ is the **key property** for deriving fermionic propagators:
- In QFT, fermion exchange introduces a **minus sign** (Pauli exclusion)
- In the Fractal Set, cloning asymmetry introduces a **fitness sign flip**
- This structure will enable antisymmetric Green's functions in the continuum limit

---

### IG Edge Spinor Attributes (Directed)

**Type:** Definition
**Label:** `def-ig-edge-attributes`
**Source:** [13_fractal_set_new/01_fractal_set.md § 3.3. IG Edge Spinor Data Structure](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `spinor`

**Statement:**
Each **directed** IG edge $(n_{i,t}, n_{j,t}) \in E_{\text{IG}}$ (from walker $j$ to walker $i$) contains data from the Adaptive Gas SDE:

**Identity:**
- $\text{source\_walker} = j$: Walker ID providing influence (source)
- $\text{target\_walker} = i$: Walker ID receiving influence (target)
- $\text{timestep} = t$: Shared timestep

**Spatial Spinors:**
- $\psi_{x_i} \in \mathbb{C}^{2^{[d/2]}}$: Position spinor of walker $i$ (encoding $x_i$)
- $\psi_{x_j} \in \mathbb{C}^{2^{[d/2]}}$: Position spinor of walker $j$ (encoding $x_j$)
- $\psi_{\Delta x_{ij}} := \psi_{x_j} \ominus \psi_{x_i} \in \mathbb{C}^{2^{[d/2]}}$: Relative position spinor (encodes $x_j - x_i$)

**Velocity Spinors:**
- $\psi_{v_i} \in \mathbb{C}^{2^{[d/2]}}$: Velocity spinor of walker $i$
- $\psi_{v_j} \in \mathbb{C}^{2^{[d/2]}}$: Velocity spinor of walker $j$
- $\psi_{\Delta v_{ij}} := \psi_{v_j} \ominus \psi_{v_i} \in \mathbb{C}^{2^{[d/2]}}$: Relative velocity spinor (encodes $v_j - v_i$)

**Viscous Coupling Spinor:**
- $\psi_{\text{viscous},ij} \in \mathbb{C}^{2^{[d/2]}}$: Viscous force contribution from $j$ to $i$:

  $$
  \psi_{\text{viscous},ij} := \text{spinor}\left[\nu K_\rho(x_i, x_j) (v_j - v_i)\right]

  $$

  This is the pairwise contribution to $\mathbf{F}_{\text{viscous}}(x_i, S)$ from walker $j$.

**Localization Kernel Weight (Complex Scalar):**
- $K_\rho(x_i, x_j) \in \mathbb{R}_{> 0}$: Localization kernel weight
- $w_{ij}(\rho) := \frac{K_\rho(x_i, x_j)}{\sum_{\ell \in A_k} K_\rho(x_i, x_\ell)} \in [0,1]$: Normalized weight for localized moments

**Phase Potential and Complex Amplitude:**
- $\theta_{ij} := -\frac{d_{\text{alg}}(i,j)^2}{2 \varepsilon_c^2 \hbar_{\text{eff}}} \in \mathbb{R}$: Phase potential encoding algorithmic distance
- $\psi_{ij} := \sqrt{P_{\text{comp}}(i,j)} \cdot \exp(i \theta_{ij}) \in \mathbb{C}$: Complex amplitude for quantum-like coupling

where:
- $d_{\text{alg}}(i,j)$: Algorithmic distance (from fitness-weighted genealogy)
- $\varepsilon_c$: Cloning interaction range (characteristic length scale)
- $\hbar_{\text{eff}}$: Effective Planck constant (fundamental algorithmic quantum)
- $P_{\text{comp}}(i,j)$: Companion selection probability

**Derived Scalar Quantities:**
- $d_{ij} := \|x_i - x_j\| = |\psi_{\Delta x_{ij}}| \in \mathbb{R}_{\geq 0}$: Euclidean distance

**Antisymmetric Fitness Potential (KEY SCALAR):**
- $V_{\text{clone}}(i \to j) := \Phi_j - \Phi_i \in \mathbb{R}$: **Directed fitness difference** (antisymmetric!)

  This is the **primary edge weight** encoding directed influence. It satisfies:

  $$
  V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)

  $$

**Fitness Scalars (from nodes):**
- $\Phi_i := \Phi(n_{i,t}) \in \mathbb{R}$: Fitness of target walker $i$
- $\Phi_j := \Phi(n_{j,t}) \in \mathbb{R}$: Fitness of source walker $j$

---

### The Fractal Set

**Type:** Definition
**Label:** `def-fractal-set`
**Source:** [13_fractal_set_new/01_fractal_set.md § 4.1. Fractal Set Definition](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fermionic`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `spinor`, `symmetry`

**Statement:**
The **Fractal Set** is the union of CST and IG:

$$
\mathcal{F} := (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}}, \omega_{\text{CST}}, \omega_{\text{IG}}),

$$

where:
- $\mathcal{N}$: Node set (all spacetime points, $|\mathcal{N}| = N \times (T+1)$)
  - **Nodes store scalars**: All frame-invariant quantities
- $E_{\text{CST}}$: Directed temporal edges (timestep transitions)
  - **Edges store spinors**: Velocity, force, displacement spinors
- $E_{\text{IG}}$: **Directed spatial edges** (asymmetric selection coupling at each timestep)
  - **Edges store spinors**: Relative position, velocity, viscous coupling spinors
  - **Key scalar**: Antisymmetric cloning potential $V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i$
- $\omega_{\text{CST}} : E_{\text{CST}} \to \mathbb{R}_{>0}$: CST edge weights (typically $\Delta t$)
- $\omega_{\text{IG}} : E_{\text{IG}} \to \mathbb{R}_{>0}$: IG edge weights (selection coupling strength)

**Covariance structure:**
- **Scalar node data** is **frame-invariant** (same in all coordinate systems)
- **Spinor edge data** is **frame-covariant** (transforms under the spinor representation)
- Together, they provide a **complete, coordinate-free description** of the Adaptive Gas dynamics

**Separation of concerns:**
- **CST**: Encodes **timelike structure** (how state evolves, who clones whom)
- **IG**: Encodes **spacelike structure** (contemporaneous **directed** coupling through selection and viscous forces)

**Antisymmetry and fermionic structure:**
- IG edges are **directed** with antisymmetric cloning potential: $V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)$
- This antisymmetry is the **discrete precursor** to fermionic exchange symmetry
- In the continuum limit, will yield antisymmetric propagators (fermionic Green's functions)

---

### Two Distinct Companion Selection Events

**Type:** Definition
**Label:** `def-two-companion-selections`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.2. The Two Independent Random Selections](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
The cloning operator involves exactly **two independent random companion selections**:

**1. Diversity Companion Selection** (for computing fitness):

For each walker i at timestep t, sample diversity companion k from alive set:

$$
k \sim P_{\text{comp}}^{(\text{div})}(\cdot | i)

$$

The diversity companion k is used to compute walker i's fitness:

$$
V_{\text{fit}}(i|k) = V_{\text{fit}}[f_k, \rho, \tilde{d}_i(k)](x_i)

$$

where $\tilde{d}_i(k)$ is the Z-score of distance $d(x_i, x_k)$ in the ρ-neighborhood of i.

**Physical interpretation**: Measures "how diverse is walker i relative to randomly sampled companion k?"

**2. Cloning Companion Selection** (for comparison):

For walker i attempting to clone, sample cloning companion j from alive set:

$$
j \sim P_{\text{comp}}^{(\text{clone})}(\cdot | i)

$$

The cloning companion j is the target that i will compare fitness against.

**Physical interpretation**: "Which neighbor should i attempt to displace through cloning?"

**Independence**: These two selections are **statistically independent**:

$$
P(j, k | i) = P_{\text{comp}}^{(\text{clone})}(j|i) \cdot P_{\text{comp}}^{(\text{div})}(k|i)

$$

**Four-walker correlation**: The cloning score depends on **both walkers' diversity companions**:

$$
S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)

$$

where k is i's diversity companion and m is j's diversity companion.

---

### Complete Set of Phase Potentials and Amplitudes

**Type:** Definition
**Label:** `def-complete-phase-amplitudes`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.3. Phase Potentials and Complex Amplitudes](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
For the four-walker cloning event $(i \to j, k, m)$, define the following phase potentials and complex amplitudes:

**1. Diversity Companion Amplitude:**

Phase potential:

$$
\theta_{ik}^{(\text{div})} := -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2 \hbar_{\text{eff}}}

$$

Complex amplitude:

$$
\psi_{ik}^{(\text{div})} := \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot \exp\left(i\theta_{ik}^{(\text{div})}\right) \in \mathbb{C}

$$

**Unitarity**: $\sum_{k \in A_t \setminus \{i\}} \left|\psi_{ik}^{(\text{div})}\right|^2 = 1$

**2. Cloning Companion Amplitude:**

Phase potential:

$$
\theta_{ij}^{(\text{clone})} := -\frac{d_{\text{alg}}(i,j)^2}{2\epsilon_c^2 \hbar_{\text{eff}}}

$$

Complex amplitude:

$$
\psi_{ij}^{(\text{clone})} := \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} \cdot \exp\left(i\theta_{ij}^{(\text{clone})}\right) \in \mathbb{C}

$$

**Unitarity**: $\sum_{j \in A_t \setminus \{i\}} \left|\psi_{ij}^{(\text{clone})}\right|^2 = 1$

**3. Fitness Amplitude (Pure Phase):**

For walker i with diversity companion k:

$$
\psi_i^{(\text{fit})}(k) := \exp\left(\frac{iV_{\text{fit}}(i|k)}{\hbar_{\text{eff}}}\right) \in S^1 \subset \mathbb{C}

$$

**Note**: Modulus = 1 (pure phase) since fitness is deterministic given diversity companion k.

**4. Success Amplitude (Cloning Decision):**

For cloning score $S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$:

$$
\psi_{\text{succ}}(S) := \sqrt{P_{\text{succ}}(S)} \cdot \exp\left(\frac{iS}{\hbar_{\text{eff}}}\right) \in \mathbb{C}

$$

where $P_{\text{succ}}(S) = \sigma(S/T_{\text{clone}})$ is the cloning success probability (sigmoid function).

**Physical interpretation**: Success probability as modulus, score difference as phase.

**Constants:**
- $\epsilon_d > 0$: Diversity interaction range
- $\epsilon_c > 0$: Cloning interaction range
- $\hbar_{\text{eff}} > 0$: Effective Planck constant (fundamental action scale)
- $T_{\text{clone}} > 0$: Cloning temperature (sigmoid steepness parameter)

---

### IG Edge Attributes with Gauge Symmetry Assignment

**Type:** Definition
**Label:** `def-ig-edge-gauge-symmetries`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.8. Storage in Fractal Set: IG Edge Structure with Gauge Symmetries](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `nodes`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
**Extension to Table 3 (IG Edge Attributes):**

Add the following fields to each directed IG edge $(n_{i,t}, n_{j,t})$:

| **Category** | **Field** | **Type** | **Description** | **Symmetry** |
|--------------|-----------|----------|-----------------|--------------|
| **Edge Type** | `edge_type` | `enum` | `"diversity"` or `"cloning"` | |
| **Diversity Edge (U(1))** | `theta_U1` | `float` | $\theta_{ik}^{(\text{U(1)})} = -d_{\text{alg}}(i,k)^2/(2\epsilon_d^2 \hbar_{\text{eff}})$ | U(1) fitness |
| | `psi_U1` | `complex` | $\psi_{ik}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k\|i)} e^{i\theta_{ik}^{(\text{U(1)})}}$ | U(1) fitness |
| | `diversity_companion_id` | `int` | Walker ID k (diversity companion) | |
| | `V_fit_ik` | `float` | Fitness potential $V_{\text{fit}}(i\|k)$ | |
| **Cloning Edge (SU(2))** | `theta_SU2` | `float` | $\theta_{ij}^{(\text{SU(2)})} = -d_{\text{alg}}(i,j)^2/(2\epsilon_c^2 \hbar_{\text{eff}})$ | SU(2) weak |
| | `A_SU2` | `complex` | $A_{ij}^{\text{SU(2)}} = \sqrt{P_{\text{comp}}^{(\text{clone})}(j\|i)} e^{i\theta_{ij}^{(\text{SU(2)})}}$ | SU(2) weak |
| | `cloning_companion_id` | `int` | Walker ID j (cloning companion) | |
| | `cloning_score` | `float` | $S(i,j,k,m) = V_{\text{fit}}(i\|k) - V_{\text{fit}}(j\|m)$ | |
| | `psi_success` | `complex` | $\psi_{\text{succ}}(S) = \sqrt{P_{\text{succ}}(S)} e^{iS/\hbar_{\text{eff}}}$ | |

**Symmetry-to-Edge Mapping:**

| **Symmetry** | **Algorithmic Origin** | **IG Edge Type** | **Count per Walker** |
|--------------|------------------------|------------------|---------------------|
| **U(1) fitness** | Diversity self-measurement | Diversity edge $(i, k)$ | $N-1$ |
| **SU(2) weak** | Cloning interaction | Cloning edge $(i, j)$ | $1$ |

**Physical Interpretation:**

1. **Diversity edges** are U(1) gauge connections representing fitness self-measurement probes
2. **Cloning edges** are SU(2) interaction vertices representing weak isospin coupling

**Storage for Dressed SU(2) Interaction:**

To reconstruct $\Psi(i \to j) = A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)$:
- **SU(2) vertex**: Stored on cloning edge $(i, j)$ as `A_SU2`
- **U(1) dressing of i**: Sum over all diversity edges $(i, k)$ to compute $|\psi_i\rangle$
- **U(1) dressing of j**: Sum over all diversity edges $(j, m)$ to compute $|\psi_j\rangle$
- **Effective kernel**: $K_{\text{eff}}$ computed from path integral over (k, m)

**Storage Efficiency:**

For N alive walkers:
- **Diversity edges (U(1))**: $N(N-1)$ total
- **Cloning edges (SU(2))**: $N$ total
- **Total IG edges per timestep**: $N^2$ edges

**Factorized Tensor Storage:**

$$
\Psi(i \to j) = \underbrace{A_{ij}^{\text{SU(2)}}}_{\text{Cloning edge}} \cdot \underbrace{\sum_{k,m} \psi_{ik}^{(\text{div})} \cdot \psi_{jm}^{(\text{div})} \cdot \psi_{\text{succ}}(S)}_{\text{Diversity edges + outcome}}

$$

This is a rank-4 tensor with dimension $N \times N \times N \times N$, but only $(N-1)^2$ non-zero entries per (i,j) pair.

---

### Fitness Operator on Diversity Space

**Type:** Definition
**Label:** `def-fitness-operator`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.10. SU(2) Weak Isospin Symmetry from Four-Walker Interaction](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
The **fitness operator** $\hat{V}_{\text{fit},i}$ for walker i acts on the diversity Hilbert space $\mathcal{H}_{\text{div}}$ as:

$$
\hat{V}_{\text{fit},i} |k\rangle := V_{\text{fit}}(i|k) |k\rangle

$$

This operator is **diagonal** in the companion basis $\{|k\rangle\}$ and depends on which walker's context (i or j) defines the fitness evaluation.

**Expectation value** for dressed walker $|\psi_i\rangle$:

$$
\langle \psi_i | \hat{V}_{\text{fit},i} | \psi_i \rangle = \sum_{k \in A_t} \left|\psi_{ik}^{(\text{div})}\right|^2 V_{\text{fit}}(i|k)

$$

This is the average fitness of walker i over all possible diversity companion choices.

**Cloning score operator** on the interaction doublet space $\mathcal{H}_{\text{int}}(i,j) = \mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}}$:

Define isospin projection operators:
- $\hat{P}_{\uparrow} = |{\uparrow}\rangle\langle{\uparrow}|$: projects onto "cloner" role
- $\hat{P}_{\downarrow} = |{\downarrow}\rangle\langle{\downarrow}|$: projects onto "target" role

The **cloning score operator** is:

$$
\hat{S}_{ij} := (\hat{P}_{\uparrow} \otimes \hat{V}_{\text{fit},i}) - (\hat{P}_{\downarrow} \otimes \hat{V}_{\text{fit},j})

$$

This operator acts correctly on $\mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}}$ by:
1. Projecting onto the $|{\uparrow}\rangle$ component and applying walker i's fitness evaluation
2. Projecting onto the $|{\downarrow}\rangle$ component and applying walker j's fitness evaluation
3. Computing their difference

**Expected score**:

$$
\langle \Psi_{ij} | \hat{S}_{ij} | \Psi_{ij} \rangle = \langle \psi_i | \hat{V}_{\text{fit},i} | \psi_i \rangle - \langle \psi_j | \hat{V}_{\text{fit},j} | \psi_j \rangle

$$

This operator encodes the fitness comparison between the two walkers in their respective roles.

---

### The Reward Scalar Field

**Type:** Definition
**Label:** `def-reward-scalar-field`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.11. Scalar Higgs-Like Reward Field](13_fractal_set_new/01_fractal_set.md)
**Tags:** `convergence`, `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `metric-tensor`, `nodes`, `spinor`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The **reward function** $r(x)$ acts as a fundamental **real scalar field** analogous to the Higgs field in the Standard Model.

**1. Field Definition:**

For each position $x \in \mathcal{X}$ in state space:

$$
r: \mathcal{X} \to \mathbb{R}

$$

The reward field is **position-dependent** but **companion-independent** (unlike fitness $V_{\text{fit}}(i|k)$).

**2. Coupling to Gauge Sector:**

The reward field couples to the U(1)_fitness × SU(2)_weak electroweak sector through the fitness functional:

$$
V_{\text{fit}}(i|k) = V_{\text{fit}}[r, \rho, \tilde{d}_i(k)](x_i)

$$

where:
- $r(x_i)$: Local reward value (Higgs scalar)
- $\tilde{d}_i(k)$: Diversity score (Z-score of distance to companion k)
- $\rho$: Localization scale

The reward field provides the scalar potential that the U(1) fitness gauge symmetry probes via diversity measurements.

**3. Higgs Mechanism Analogue:**

The reward field provides "mass" to walkers through the fitness potential:

$$
m_{\text{eff}}(i) \propto V_{\text{fit}}(i|k) \propto r(x_i)

$$

**Phase transition**: When the algorithm converges, $\langle r \rangle \neq 0$ (symmetry breaking):
- **Pre-convergence**: $\langle r \rangle \approx 0$ (symmetric phase, walkers explore uniformly)
- **Post-convergence**: $\langle r \rangle > 0$ (broken phase, walkers concentrate near optima)

**4. Yukawa Coupling:**

The scalar field couples to the "weak isospin doublet" $|\psi_i\rangle$ through:

$$
\mathcal{L}_{\text{Yukawa}} \sim r(x_i) \cdot \bar{\psi}_i \psi_i = r(x_i) \cdot (p_i + (1-p_i))

$$

This gives the cloning probability a "mass term":

$$
p_i(S) \sim \frac{1}{1 + e^{-S/T_{\text{clone}}}} \quad \text{where} \quad S \propto V_{\text{fit}}(i|k) \propto r(x_i)

$$

**5. Storage in Fractal Set:**

The reward field is stored as a **node scalar attribute**:
- Node attribute: $r(n_{i,t}) = r(x_i(t))$
- CST edge attribute: Gradient $\nabla r(x_i)$ stored as spinor $\psi_{\nabla r}$

---

### SU(3) Gluon Field Components from Manifold Geometry

**Type:** Definition
**Label:** `def-su3-gluon-field`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.13. SU(3) Strong Sector from Viscous Force](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `riemannian`, `spinor`, `su3-symmetry`

**Statement:**
The eight gluon field components $A_{ij}^a$ for $a = 1, \ldots, 8$ are constructed from the emergent Riemannian metric $g_{\mu\nu}(x)$:

**1. Metric Tensor:**

$$
g_{\mu\nu}(x) = \delta_{\mu\nu} + \frac{1}{\epsilon_\Sigma^2} \frac{\partial^2 V_{\text{fit}}}{\partial x^\mu \partial x^\nu}(x)

$$

from the regularized Hessian diffusion (see {prf:ref}`def-hybrid-sde`).

**2. Christoffel Symbols (Connection Coefficients):**

$$
\Gamma^\lambda_{\mu\nu}(x) = \frac{1}{2} g^{\lambda\rho}(x) \left(\frac{\partial g_{\rho\mu}}{\partial x^\nu} + \frac{\partial g_{\rho\nu}}{\partial x^\mu} - \frac{\partial g_{\mu\nu}}{\partial x^\rho}\right)

$$

**3. Gluon Field Components:**

The SU(3) gluon field $A_{ij}^a$ is derived from the Christoffel symbols projected onto the Gell-Mann basis:

$$
A_{ij}^a = \text{Tr}\left[\lambda_a \cdot \Gamma(x_i, x_j)\right]

$$

where $\Gamma(x_i, x_j)$ is the parallel transport operator from walker i to walker j along geodesic.

**Storage in Fractal Set:**

Each IG edge $(n_{i,t}, n_{j,t})$ stores:
- Gluon field: $\mathbf{A}_{ij} = (A_{ij}^1, \ldots, A_{ij}^8) \in \mathbb{R}^8$
- Color state: $|\Psi_i^{(\text{color})}\rangle \in \mathbb{C}^3$ (node attribute)
- Viscous force spinor: $\psi_{\mathbf{F}_{\text{viscous}},ij}$ (already stored)

**Related Results:** `def-hybrid-sde`

---

### Storage of Curvature Tensor in Fractal Set

**Type:** Definition
**Label:** `def-curvature-storage`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.14. Emergent General Relativity and Curved Manifold](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `curvature`, `data-structure`, `edges`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `spinor`

**Statement:**
The Riemann curvature tensor $R^\rho_{\sigma\mu\nu}(x_i)$ at each walker position is stored in the Fractal Set:

**Node attributes** (scalars):
- Ricci scalar: $R(n_{i,t}) = R(x_i(t))$
- Metric determinant: $\det(g)(n_{i,t}) = \det(g_{\mu\nu}(x_i))$

**CST edge attributes** (spinors):
- Christoffel symbols: $\Gamma^\lambda_{\mu\nu}(n_{i,t})$ (rank-3 tensor, stored as spinor components)
- Ricci tensor: $R_{\mu\nu}(n_{i,t})$ (symmetric 2-tensor, stored as spinor)

**Full Riemann tensor**: Stored as auxiliary data structure (rank-4 tensor, $d^4$ components for d-dimensional space).

**Dimension**: For $d=3$ spatial dimensions, $R^\rho_{\sigma\mu\nu}$ has $3^4 = 81$ components, but symmetries reduce to **20 independent components**.

---

### Propositions (2)

### Frame Independence of the Fractal Set

**Type:** Proposition
**Label:** `prop-frame-independence`
**Source:** [13_fractal_set_new/01_fractal_set.md § 6.2. Covariance Verification](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `spinor`

**Statement:**
Let $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ be the Fractal Set. Under a coordinate transformation $x \mapsto x' = R(x)$ with $R \in \text{SO}(d)$:

1. **Node scalars** are invariant:
   $$
   \Phi'(n) = \Phi(n), \quad E'_{\text{kin}}(n) = E_{\text{kin}}(n), \quad \text{etc.}
   $$

2. **Edge spinors** transform covariantly:
   $$
   \psi'_v = S(R) \psi_v, \quad \psi'_{\mathbf{F}} = S(R) \psi_{\mathbf{F}}, \quad \text{etc.}
   $$

   where $S: \text{SO}(d) \to \text{Spin}(d)$ is the spinor representation.

3. **Physical observables** (derived from spinors) are invariant:
   $$
   \|\mathbf{F}_{\text{total}}\|' = \|\mathbf{F}_{\text{total}}\|, \quad \theta'_{\text{deflection}} = \theta_{\text{deflection}}
   $$

**Consequence:** The Fractal Set $\mathcal{F}$ provides a **coordinate-free description** of the Adaptive Gas algorithm. Any physical quantity computed from $\mathcal{F}$ is independent of the choice of reference frame.

---

### SU(2) Invariance of Total Interaction Probability

**Type:** Proposition
**Label:** `prop-su2-invariance`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.10. SU(2) Weak Isospin Symmetry from Four-Walker Interaction](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `su2-symmetry`, `symmetry`

**Statement:**
The SU(2) symmetry has a direct physical consequence: the **total cloning interaction probability** for the pair $(i, j)$ is invariant under SU(2) transformations.

**Definition of invariant observable**:

$$
P_{\text{total}}(i, j) := P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)

$$

**Invariance statement**:

$$
P_{\text{total}}(i, j) = P'_{\text{total}}(i, j)

$$

where primed quantities are computed from the rotated state $|\Psi'_{ij}\rangle = (U \otimes I)|\Psi_{ij}\rangle$.

**Physical interpretation**:

An SU(2) rotation changes the "viewpoint" of the interaction. In one basis, walker i might seem more likely to clone j. After rotation, j might seem more likely to clone i. However, the **total propensity for the pair to interact via cloning remains constant**.

The interaction potential is conserved within the pair, even if its directionality changes from a different perspective.

---

### Theorems (14)

### Reconstruction Theorem: The Fractal Set as Complete Algorithm Representation

**Type:** Theorem
**Label:** `thm-fractal-set-reconstruction`
**Source:** [13_fractal_set_new/01_fractal_set.md § 4.2. Completeness Property](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `spinor`

**Statement:**
The Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}}, \omega_{\text{CST}}, \omega_{\text{IG}})$ with scalar node data and spinor edge data contains **complete information** to reconstruct all dynamics of the Adaptive Viscous Fluid Model.

**Statement:** Given the Fractal Set $\mathcal{F}$ storing data from [../07_adaptative_gas.md](../07_adaptative_gas.md), we can reconstruct:

1. **Full phase space trajectory**: $(x_i(t), v_i(t))$ for all walkers $i \in \{1, \ldots, N\}$ and timesteps $t \in \{0, \ldots, T\}$
2. **All force fields**: $\mathbf{F}_{\text{stable}}(x,t), \mathbf{F}_{\text{adapt}}(x,S,t), \mathbf{F}_{\text{viscous}}(x,S,t)$ at each spacetime point
3. **Diffusion tensor field**: $\Sigma_{\text{reg}}(x,S,t)$ at all points
4. **Fitness landscape**: $\Phi(x)$ sampled at walker positions
5. **Virtual reward field**: $V_{\text{fit}}[f_k, \rho](x)$ at all positions
6. **Localized statistical moments**: $\mu_\rho[f_k, \Phi, x], \sigma_\rho[f_k, \Phi, x], Z_\rho[f_k, \Phi, x]$ at all spacetime points
7. **Viscous coupling structure**: Pairwise interactions via IG edge spinors
8. **Alive walker set**: $A_k(t)$ at each timestep via node status flags
9. **Empirical measure**: $f_k(t) = \frac{1}{k}\sum_{i \in A_k(t)} \delta_{(x_i(t), v_i(t))}$ at all timesteps

**Proof:**

1. **Phase space reconstruction**:
   - Positions: Integrate $\psi_{\Delta x}$ along CST edges from initial conditions
   - Velocities: Stored as $\psi_{v,t}$ on CST edges, extract via spinor-to-vector map
   - Both yield frame-covariant $(x_i(t), v_i(t))$ in any coordinate system

2. **Force field reconstruction**: CST edges store complete spinor decomposition:
   $$
   \psi_{\mathbf{F}_{\text{total}}} = \psi_{\mathbf{F}_{\text{stable}}} \oplus \psi_{\mathbf{F}_{\text{adapt}}} \oplus \psi_{\mathbf{F}_{\text{viscous}}} \oplus \psi_{\mathbf{F}_{\text{friction}}}
   $$
   Extract each component: $\mathbf{F}_{\text{stable}} = -\nabla U$, $\mathbf{F}_{\text{adapt}} = \epsilon_F \nabla V_{\text{fit}}$, etc.

3. **Viscous force reconstruction**: IG edges store pairwise contributions $\psi_{\text{viscous},ij}$:
   $$
   \mathbf{F}_{\text{viscous}}(x_i, S, t) = \sum_{j \in A_k(t), j \neq i} \nu K_\rho(x_i, x_j)(v_j - v_i) = \bigoplus_{j \in A_k(t), j \neq i} \psi_{\text{viscous},ij}
   $$

4. **Diffusion tensor reconstruction**: CST edges store $\psi_{\Sigma_{\text{reg}}}$, extract:
   $$
   \Sigma_{\text{reg}}(x_i, S, t) = (H_i(S) + \epsilon_\Sigma I)^{-1/2}
   $$
   where Hessian $H_i$ can be computed from fitness gradient spinors $\psi_{\nabla V_{\text{fit}}}$

5. **Fitness and virtual reward**: Node scalars $\Phi(n_{i,t}), V_{\text{fit}}(n_{i,t})$ provide direct sampling

6. **Statistical moments**: Node scalars $\mu_\rho(n), \sigma_\rho(n), \sigma'_\rho(n), Z_\rho(n)$ stored at each spacetime point

7. **Localization weights**: IG edges store $K_\rho(x_i, x_j)$ and $w_{ij}(\rho)$, enabling:
   $$
   \mu_\rho[f_k, \Phi, x_i] = \sum_{j \in A_k} w_{ij}(\rho) \Phi_j
   $$

8. **Alive walker set**: Node status $s(n_{i,t}) \in \{0,1\}$ determines $A_k(t) = \{i : s(n_{i,t}) = 1\}$

9. **Empirical measure**: Combine alive set with trajectories:
   $$
   f_k(t) = \frac{1}{|A_k(t)|} \sum_{i \in A_k(t)} \delta_{(x_i(t), v_i(t))}
   $$

10. **SDE verification**: The reconstructed $(x_i(t), v_i(t))$ satisfy the Adaptive Gas SDE:
    $$
    \begin{aligned}
    dx_i &= v_i \, dt \\
    dv_i &= \left[\mathbf{F}_{\text{stable}}(x_i) + \mathbf{F}_{\text{adapt}}(x_i,S) + \mathbf{F}_{\text{viscous}}(x_i,S) - \gamma v_i\right] dt + \Sigma_{\text{reg}}(x_i,S) \circ dW_i
    \end{aligned}
    $$
    up to the stochastic noise realization $dW_i$

11. **Frame independence**: All reconstructions use spinor operations $\psi \mapsto v$ that are covariant under $\text{SO}(d)$ transformations ∎

---

### Tensor Product Structure for Cloning Interaction

**Type:** Theorem
**Label:** `thm-interaction-hilbert-space`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.4. Interaction Hilbert Space and Four-Walker States](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`

**Statement:**
For a cloning event between walkers i and j, the **interaction Hilbert space** is:

$$
\mathcal{H}_{\text{int}}(i,j) = \mathcal{H}_i^{(\text{div})} \otimes \mathcal{H}_j^{(\text{div})} = \mathbb{C}^{N-1} \otimes \mathbb{C}^{N-1}

$$

where N = |A_t| is the number of alive walkers.

**Dimension**: $\dim \mathcal{H}_{\text{int}}(i,j) = (N-1)^2$

**Basis states**: $|k, m\rangle = |k\rangle_i \otimes |m\rangle_j$ for $k, m \in A_t \setminus \{i,j\}$

**Physical interpretation**: Each basis state corresponds to a specific choice of diversity companions (k for walker i, m for walker j).

**State vector for cloning event i→j:**

$$
|\Psi(i \to j)\rangle = \sum_{k,m \in A_t} \psi_{ik}^{(\text{div})} \psi_{jm}^{(\text{div})} |k, m\rangle

$$

**Normalization:**

$$
\langle \Psi(i \to j) | \Psi(i \to j) \rangle = \left(\sum_k \left|\psi_{ik}^{(\text{div})}\right|^2\right) \left(\sum_m \left|\psi_{jm}^{(\text{div})}\right|^2\right) = 1

$$

using unitarity of diversity companion amplitudes.

---

### Path Integral Formulation of the Dressed SU(2) Interaction

**Type:** Theorem
**Label:** `thm-path-integral-dressed-su2`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.5. Complete Path Integral Formula: U(1) Dressing and SU(2) Interaction](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `su2-symmetry`, `u1-symmetry`

**Statement:**
The **total cloning amplitude** for walker i to clone over walker j is the product of an **SU(2) interaction amplitude** and an **effective interaction kernel** computed by a path integral over all U(1) diversity-dressing configurations:

$$
\Psi(i \to j) = A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)

$$

where:

**1. SU(2) Interaction Amplitude:**

$$
A_{ij}^{\text{SU(2)}} := \psi_{ij}^{(\text{clone})} = \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} \cdot e^{i\theta_{ij}^{(\text{SU(2)})}}

$$

This is the amplitude for selecting walker j as the cloning companion, **initiating the SU(2) interaction**. The phase $\theta_{ij}^{(\text{SU(2)})}$ is the phase of the SU(2) vertex itself (previously denoted $\theta_{ij}^{(\text{clone})}$).

**2. Effective Interaction Kernel (Path Integral over U(1) Dressings):**

$$
K_{\text{eff}}(i, j) := \sum_{k,m \in A_t} \left[ \underbrace{\psi_{ik}^{(\text{div})}}_{\text{U(1) dressing of } i} \cdot \underbrace{\psi_{jm}^{(\text{div})}}_{\text{U(1) dressing of } j} \cdot \underbrace{\psi_{\text{succ}}(S(i,j,k,m))}_{\text{Interaction outcome}} \right]

$$

where:
- $\psi_{ik}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}^{(\text{U(1)})}}$: **U(1) self-measurement** amplitude (walker i probing its fitness via diversity companion k)
- $\psi_{jm}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(m|j)} \cdot e^{i\theta_{jm}^{(\text{U(1)})}}$: **U(1) self-measurement** amplitude (walker j probing its fitness via diversity companion m)
- $\psi_{\text{succ}}(S) = \sqrt{P_{\text{succ}}(S)} \cdot e^{iS/\hbar_{\text{eff}}}$: Success amplitude for cloning score $S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$

**Physical Interpretation:**

The SU(2) interaction between walkers i and j does not occur in isolation. Its effective strength and outcome, encoded in $K_{\text{eff}}$, are determined by **quantum interference** of all the ways i and j are simultaneously interacting with their respective diversity environments (k and m) via **U(1) self-measurement**.

The path integral calculates how the U(1) "charges" (fitness measurements) of the walkers modify their effective SU(2) interaction potential.

**Explicit form**:

$$
\Psi(i \to j) = \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} \cdot e^{i\theta_{ij}^{(\text{SU(2)})}} \cdot \sum_{k,m} \sqrt{P_k P_m P_{\text{succ}}(S_{km})} \cdot e^{i(\theta_{ik}^{(\text{U(1)})} + \theta_{jm}^{(\text{U(1)})} + S_{km}/\hbar_{\text{eff}})}

$$

where $P_k = P_{\text{comp}}^{(\text{div})}(k|i)$ and $P_m = P_{\text{comp}}^{(\text{div})}(m|j)$.

**Total cloning probability**:

$$
P_{\text{clone}}(i \to j) = \left|\Psi(i \to j)\right|^2 = \left|A_{ij}^{\text{SU(2)}}\right|^2 \cdot \left|K_{\text{eff}}(i, j)\right|^2

$$

---

### Global U(1)_fitness Symmetry

**Type:** Theorem
**Label:** `thm-u1-fitness-global`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.6. Global U(1) Fitness Symmetry of Diversity Measurement](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `nodes`, `noether`, `symmetry`, `u1-symmetry`

**Statement:**
The diversity companion selection mechanism defines a **global U(1) symmetry** governing fitness self-measurement:

$$
G_{\text{fitness}} = \text{U}(1)_{\text{fitness}}^{\text{global}}

$$

acting on the diversity Hilbert space $\mathcal{H}_{\text{div}} = \mathbb{C}^{N-1}$.

**Algorithmic Origin:**

A walker i determines its fitness potential $V_{\text{fit}}(i)$ by **sampling a diversity companion k**. This is an act of self-measurement against the environmental backdrop (the swarm). The complex amplitude:

$$
\psi_{ik}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}^{(\text{U(1)})}}

$$

where $\theta_{ik}^{(\text{U(1)})} = -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2\hbar_{\text{eff}}}$ depends only on the algorithmic distance.

**Global Symmetry Transformation:**

All walkers undergo the **same phase rotation**:

$$
\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}, \quad \alpha \in [0, 2\pi) \text{ (same for all i)}

$$

Equivalently, all phases shift uniformly:

$$
\theta_{ik}^{(\text{U(1)})} \to \theta_{ik}^{(\text{U(1)})} + \alpha

$$

**Invariant Observables:**

Under global phase rotation, all physical observables are invariant:

$$
P_{\text{clone}}(i \to j) = |\Psi(i \to j)|^2 = |A_{ij}^{\text{SU}(2)}|^2 \cdot |K_{\text{eff}}(i,j)|^2

$$

transforms as:

$$
|K_{\text{eff}}'(i,j)|^2 = |e^{i \cdot 2\alpha}|^2 \cdot |K_{\text{eff}}(i,j)|^2 = |K_{\text{eff}}(i,j)|^2

$$

**Why Global, Not Local:**

The symmetry is global (not local/gauged) because:

1. **No gauge field**: Phases $\theta_{ik}$ are fixed by algorithmic distances, not dynamical gauge fields
2. **Path integral structure**: $K_{\text{eff}} = \sum_{k,m} [\ldots]$ - different terms get different phase factors under local transformation
3. **Physical meaning**: Absolute fitness scale is unphysical, only **relative** fitness matters

**Conserved Charge (Noether's Theorem):**

From Theorem 4.2 in {doc}`../09_symmetries_adaptive_gas.md`, global U(1) symmetry implies a **conserved fitness current**:

$$
J_{\text{fitness}}^{\mu} = \sum_{i \in A_t} \text{Im}(\psi_i^* \partial^\mu \psi_i)

$$

satisfying $\partial_\mu J^\mu = 0$ (conservation law).

**Physical Interpretation:**

1. **Fitness charge conservation**: Total fitness "charge" of swarm is conserved
2. **Selection rules**: Processes must conserve U(1) charge (like baryon number in QCD)
3. **Higgs coupling**: Reward field $r(x)$ couples to fitness charge (mass generation)
4. **IG Edge**: Generates **Diversity Edge $(i, k)$** for fitness self-measurement

**Analogy to Particle Physics:**

| **Adaptive Gas** | **Standard Model** |
|------------------|--------------------|
| Global U(1)_fitness | Global U(1)_B (baryon number) |
| Fitness charge | Baryon charge |
| Conserved current J_fitness | Baryon current J_B |
| Diversity companion k | Virtual meson exchange |
| Fitness potential V_fit | Effective potential |
| No gauge boson | No gauge boson (global) |

**Note on Gauge vs Global:**

This U(1) is **global** (like baryon/lepton number), not **gauged** (like electromagnetism). The **true local gauge symmetry** of the Fragile Gas is the discrete **S_N permutation group**, which gives Wilson loops via braid holonomy (see §7.7 and {doc}`../12_gauge_theory_adaptive_gas.md`).

---

### S_N Permutation Gauge Symmetry and Braid Holonomy

**Type:** Theorem
**Label:** `thm-sn-braid-holonomy`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.7. S_N Braid Holonomy: The True Gauge Structure](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `curvature`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `metric-tensor`, `nodes`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The **fundamental local gauge symmetry** of the Fragile Gas is the **symmetric group S_N** acting by permuting walker labels. This gives rise to **braid group topology** and **Wilson loops** via holonomy.

**Fundamental Gauge Group:**

$$
G_{\text{gauge}} = S_N \quad \text{(permutation group of N walkers)}

$$

**Physical Origin:**

Walker labels $\{1, 2, \ldots, N\}$ are **arbitrary** - permuting indices does not change physical state:

$$
\mathcal{S} = (w_1, w_2, \ldots, w_N) \sim \sigma \cdot \mathcal{S} = (w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(N)})

$$

for any $\sigma \in S_N$. This **gauge redundancy** is the source of gauge structure.

**Configuration Space as Orbifold:**

The physical configuration space is the **quotient**:

$$
\mathcal{M}_{\text{config}} = \Sigma_N / S_N

$$

This is an **orbifold** with fundamental group:

$$
\pi_1(\mathcal{M}_{\text{config}}) \cong B_N \quad \text{(braid group)}

$$

**Gauge Connection via Braid Homomorphism:**

Parallel transport along paths in configuration space is defined by:

$$
\rho: B_N \to S_N

$$

the canonical homomorphism from the braid group to the symmetric group.

**Wilson Loops = Holonomy:**

For a closed loop $\gamma: [0, T] \to \mathcal{M}_{\text{config}}$ with $\gamma(0) = \gamma(T) = [\mathcal{S}_0]$, the **holonomy** is:

$$
\text{Hol}(\gamma) = \rho([\gamma]) \in S_N

$$

This is a permutation $\sigma \in S_N$ representing the net relabeling of walkers after following the closed path $\gamma$.

**Example: Two-Walker Exchange**

For $N=2$, an elementary braid $\sigma_1 \in B_2$ (walker 1 passes above walker 2) has holonomy:

$$
\text{Hol}(\sigma_1) = (1 \, 2) \in S_2

$$

the transposition swapping labels 1 and 2.

**Gauge-Invariant Observables:**

Physical observables must be **S_N-invariant** functions on $\Sigma_N$:

$$
f(\sigma \cdot \mathcal{S}) = f(\mathcal{S}) \quad \forall \sigma \in S_N

$$

Examples:
- Cloning probability: $P_{\text{clone}}(i \to j)$ (depends on unordered pair)
- Fitness histograms: $\{V_{\text{fit}}(i) : i \in A_t\}$ (unordered set)
- Diversity distances: $\{d_{\text{alg}}(i,j) : i, j \in A_t\}$ (pairwise, symmetric)

**Connection to Diversity/Cloning Edges:**

When walker i selects diversity companion k, or cloning companion j, the system traces paths in configuration space:

- **Diversity cycle**: $i \to k \to i$ creates closed loop $\gamma_{ik}$
- **Cloning cycle**: $i \to j \to i$ creates closed loop $\gamma_{ij}$

Each cycle has holonomy $\text{Hol}(\gamma) \in S_N$ depending on the braid class of the trajectory.

**Curvature = Non-Trivial Holonomy:**

For a triangle of walkers $(i, j, k)$, define the **discrete curvature**:

$$
F_{ijk} = \mathbb{P}(\text{Hol}(\gamma_{ijk}) = \sigma) \quad \text{for } \sigma \in S_3

$$

If $F_{ijk} = \delta_e$ (all probability on identity), the connection is **flat**. Otherwise, the anisotropic algorithmic metric induces **non-trivial curvature**.

**Relationship to U(1) and SU(2):**

The emergent U(1) and SU(2) symmetries arise as **effective theories** in the continuum limit:

- **S_N braid holonomy** → Wilson loops (discrete, topological)
- **Global U(1)_fitness** → Conserved fitness charge (continuous, global)
- **Local SU(2)_weak** → Weak isospin gauge (continuous, local)

See {doc}`../12_gauge_theory_adaptive_gas.md` for full rigorous treatment of S_N gauge structure, including:
- Principal orbifold bundle formulation (§1-2)
- Braid group topology π₁(M_config) ≅ B_N (§3.1-3.2)
- Parallel transport and holonomy (§3.3-3.4)
- Dynamics generating braids (§4)
- Curvature and anisotropic metrics (§4.3)

---

### Hybrid Discrete-Continuum Gauge Theory on the Fractal Set

**Type:** Theorem
**Label:** `thm-sn-su2-lattice-qft`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.9. Lattice QFT Structure: S_N Discrete Gauge + SU(2) Continuum](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `curvature`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `nodes`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The Fractal Set realizes a **hybrid gauge theory** combining discrete S_N topology with continuum SU(2) dynamics:

**1. Gauge Structure:**

$$
G_{\text{total}} = S_N^{\text{discrete}} \times \text{SU}(2)_{\text{weak}}^{\text{local}} \times \text{U}(1)_{\text{fitness}}^{\text{global}}

$$

**Three-tier hierarchy:**
- **S_N**: Fundamental discrete gauge group (permutations, braid holonomy)
- **SU(2)_weak**: Emergent local gauge symmetry (weak isospin)
- **U(1)_fitness**: Emergent global symmetry (conserved fitness charge)

This structure **generalizes** the Standard Model by including discrete gauge topology.

**2. Connections and Fields:**

**S_N Discrete Connection** (on all edges):
- Braid holonomy ρ: B_N → S_N defines parallel transport
- Holonomy Hol(γ) ∈ S_N for closed loops γ
- **Wilson loops**: Gauge-invariant observables from braid topology

**U(1) Global Phase** (on diversity edges):
$$
\theta_{ik}^{(\text{U}(1)}} = -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2 \hbar_{\text{eff}}}
$$

**Not a gauge field** - fixed by algorithmic distances. Transforms globally: θ → θ + α (same α for all).

**SU(2) Local Gauge Field** (on cloning edges):
$$
A_{\text{SU}(2)}[i, j] = \theta_{ij}^{(\text{SU}(2))} = -\frac{d_{\text{alg}}(i,j)^2}{2\epsilon_c^2 \hbar_{\text{eff}}}
$$

**True local gauge field** - transforms locally: $U_{ij} \to G_i U_{ij} G_j^\dagger$ for $G_i \in \text{SU}(2)$.

**3. Curvature and Field Strength:**

**S_N Discrete Curvature**:
For triangle (i, j, k), the curvature is the holonomy distribution:
$$
F_{ijk}^{S_N} = \mathbb{P}(\text{Hol}(\gamma_{ijk}) = \sigma) \quad \forall \sigma \in S_3
$$

**SU(2) Field Strength**:
$$
F_{\text{SU}(2)}^{(a)}[P] = \oint_{\partial P} A_{\text{SU}(2)}^{(a)} + g \epsilon^{abc} A_{\text{SU}(2)}^{(b)} A_{\text{SU}(2)}^{(c)}
$$

where $a, b, c \in \{1,2,3\}$ are SU(2) generator indices (non-Abelian).

**4. Wilson Action:**

$$
S_{\text{Wilson}}[\mathcal{F}] = \sum_{\text{plaquettes } P} \left[\beta_{S_N} \mathcal{D}_{\text{KL}}(\text{Hol}[P] \| \delta_e) + \beta_{\text{SU}(2)}\left(1 - \frac{1}{2}\text{Re Tr}\, W_{\text{SU}(2)}[P]\right)\right]

$$

where:
- **S_N term**: KL divergence from trivial holonomy (flatness)
- **SU(2) Wilson loop**: $W_{\text{SU}(2)}[P] = \mathcal{P}\exp\left(i\oint_{\partial P} A_{\text{SU}(2)}\right)$ (path-ordered)
- $\beta_{S_N}$, $\beta_{\text{SU}(2)}$: Coupling constants

**5. Path Integral Formulation:**

The total partition function factorizes:

$$
Z = \sum_{\text{configs}} \exp\left(-S_{\text{Wilson}}[\mathcal{F}]\right) \prod_{i,j} \left|A_{ij}^{\text{SU}(2)}\right|^2 \cdot \left|K_{\text{eff}}(i, j)\right|^2

$$

where:
- Sum over all **U(1) diversity configurations** (k companions)
- Sum over all **SU(2) cloning configurations** (j targets)
- $A_{ij}^{\text{SU}(2)}$: SU(2) interaction vertex amplitude
- $K_{\text{eff}}(i, j)$: Effective kernel from U(1) dressing

**6. Feynman Diagrams:**

Each cloning event $\Psi(i \to j) = A_{ij}^{\text{SU}(2)} \cdot K_{\text{eff}}(i, j)$ corresponds to:
- **External legs**: Walkers i, j (bare states)
- **Self-energy loops**: U(1) diversity dressing (k, m virtual companions)
- **Central vertex**: SU(2) weak interaction (2-to-2 scattering)
- **Path integral**: $K_{\text{eff}} = \sum_{k,m}$ sums over all U(1)-dressed configurations

This is a standard QFT structure with renormalized vertex from environmental coupling.

---

### SU(2) Symmetry from Dressed Walker Interaction

**Type:** Theorem
**Label:** `thm-su2-interaction-symmetry`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.10. SU(2) Weak Isospin Symmetry from Four-Walker Interaction](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `su2-symmetry`, `symmetry`

**Statement:**
The cloning interaction between walkers i and j reveals a fundamental **SU(2) weak isospin symmetry** acting not on the outcome, but on the **dressed states** of the interacting walkers before measurement.

**1. Dressed Walker State:**

A walker i is "dressed" by its quantum superposition with all possible diversity companions. Its state $|\psi_i\rangle$ lives in the diversity Hilbert space $\mathcal{H}_{\text{div}} = \mathbb{C}^{N-1}$:

$$
|\psi_i\rangle := \sum_{k \in A_t \setminus \{i\}} \psi_{ik}^{(\text{div})} |k\rangle = \sum_{k \in A_t \setminus \{i\}} \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}^{(\text{div})}} |k\rangle

$$

where $\{|k\rangle\}$ forms an orthonormal basis for $\mathcal{H}_{\text{div}}$.

**Physical interpretation**: This state vector encodes the complete information about how walker i perceives its diversity environment.

**2. Interaction Hilbert Space (Tensor Product Structure):**

The total interaction space is the **tensor product** of an "isospin" space $\mathcal{H}_{\text{iso}} = \mathbb{C}^2$ (spanning the cloner/target roles) and the diversity space:

$$
\mathcal{H}_{\text{int}}(i,j) = \mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}} = \mathbb{C}^2 \otimes \mathbb{C}^{N-1}

$$

**Basis**: Isospin basis $\{|↑\rangle, |↓\rangle\}$ where:
- $|↑\rangle = (1, 0)^T$: "cloner" role
- $|↓\rangle = (0, 1)^T$: "target" role

**3. Weak Isospin Doublet:**

The state of the interacting pair $(i, j)$ is a vector in $\mathcal{H}_{\text{int}}$, constructed from the isospin basis and the dressed walker states:

$$
|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}

$$

**Explicit form**:

$$
|\Psi_{ij}\rangle = \sum_{k \in A_t} \left(\psi_{ik}^{(\text{div})} |↑, k\rangle + \psi_{jk}^{(\text{div})} |↓, k\rangle\right)

$$

where $|↑, k\rangle = |↑\rangle \otimes |k\rangle$.

**4. SU(2) Transformations:**

An SU(2) transformation $U \in \text{SU}(2)$ acts on the isospin space, mixing the roles of cloner and target, while leaving the diversity space untouched:

$$
|\Psi_{ij}\rangle \mapsto |\Psi'_{ij}\rangle = (U \otimes I_{\text{div}}) |\Psi_{ij}\rangle

$$

For a rotation by angle $\theta$ around axis $\hat{n}$:

$$
U = \exp\left(i\frac{\theta}{2}\boldsymbol{\sigma} \cdot \hat{n}\right), \quad \boldsymbol{\sigma} = (\sigma_1, \sigma_2, \sigma_3)

$$

**Example (rotation in isospin-3 direction)**:

$$
\begin{pmatrix} |\psi'_i\rangle \\ |\psi'_j\rangle \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} |\psi_i\rangle \\ |\psi_j\rangle \end{pmatrix}

$$

This implies: $|\psi'_i\rangle = \cos\theta |\psi_i\rangle - \sin\theta |\psi_j\rangle$

The rotated "cloner" state is a quantum superposition of the original cloner and target dressed states.

**5. Physical Interpretation:**

The SU(2) symmetry represents a fundamental equivalence between the roles of cloner and target **at the level of the interaction dynamics**, before any measurement collapses the state.

- **Mixing of roles**: An SU(2) rotation coherently mixes "walker i as cloner" with "walker j as cloner"
- **Preservation of interaction**: The total interaction potential between the pair is conserved
- **Connection to measurement**: The final binary outcome (Clone|Persist) emerges from measurement collapse (§7.10.3)

---

### Global U(1)_fitness Invariance

**Type:** Theorem
**Label:** `thm-u1-global-invariance`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.12. Formal Gauge Invariance: Proofs](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `cst`, `data-structure`, `edges`, `fractal-set`, `ig`, `nodes`, `noether`, `symmetry`, `u1-symmetry`

**Statement:**
The cloning probability $P_{\text{clone}}(i \to j)$ is invariant under **global U(1)_fitness transformations**.

**Setup:**

A **global U(1) transformation** applies the same phase rotation to all walkers:

$$
\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}, \quad \alpha \in [0, 2\pi) \text{ (same for all } i, k \text{)}
$$

**Proof:**

The effective kernel transforms as:

$$
K_{\text{eff}}'(i,j) = \sum_{k,m} e^{i\alpha} \psi_{ik}^{(\text{div})} \cdot e^{i\alpha} \psi_{jm}^{(\text{div})} \cdot \psi_{\text{succ}}(S)
$$

$$
= e^{i \cdot 2\alpha} \sum_{k,m} \psi_{ik}^{(\text{div})} \cdot \psi_{jm}^{(\text{div})} \cdot \psi_{\text{succ}}(S) = e^{i \cdot 2\alpha} K_{\text{eff}}(i,j)
$$

Taking the modulus squared:

$$
|K_{\text{eff}}'(i,j)|^2 = |e^{i \cdot 2\alpha}|^2 \cdot |K_{\text{eff}}(i,j)|^2 = |K_{\text{eff}}(i,j)|^2
$$

since $|e^{i\phi}| = 1$ for any real phase $\phi$.

Therefore, the cloning probability is invariant:

$$
P_{\text{clone}}'(i \to j) = |A_{ij}^{\text{SU}(2)}|^2 \cdot |K_{\text{eff}}'(i,j)|^2 = |A_{ij}^{\text{SU}(2)}|^2 \cdot |K_{\text{eff}}(i,j)|^2 = P_{\text{clone}}(i \to j)
$$

**Conclusion:**

Global U(1)_fitness is a genuine symmetry. By Noether's theorem (see {doc}`../09_symmetries_adaptive_gas.md` §4.2), this implies a **conserved fitness current**. ∎

---

### Local SU(2)_weak Gauge Invariance

**Type:** Theorem
**Label:** `thm-su2-local-gauge-invariance`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.12. Formal Gauge Invariance: Proofs](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `metric-tensor`, `nodes`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The path integral for the cloning amplitude is **locally gauge invariant** under SU(2)_weak transformations at each walker vertex.

**Setup:**

Define a **local SU(2) gauge transformation** at each walker vertex $i \in A_t$:

$$
G_i \in \text{SU}(2), \quad G_i = \exp\left(i\sum_{a=1}^3 \beta^{(a)}(i) \frac{\sigma^a}{2}\right)
$$

where $\beta^{(a)}(i) \in \mathbb{R}$ are three arbitrary real parameters and $\sigma^a$ are Pauli matrices.

**Transformation of Link Variables:**

The SU(2) phase on cloning edges is encoded in the **link variable**:

$$
U_{ij} = \exp(i\theta_{ij}^{(\text{SU}(2))} \mathbf{n} \cdot \boldsymbol{\sigma})
$$

where $\mathbf{n}$ is a unit vector in SU(2) space.

Under local gauge transformation:

$$
U_{ij} \to U_{ij}' = G_i \cdot U_{ij} \cdot G_j^\dagger
$$

**Physical Interpretation:**

The SU(2) gauge transformation rotates the weak isospin doublet:

$$
|\Psi_{ij}\rangle = |{\uparrow}\rangle \otimes |\psi_i\rangle + |{\downarrow}\rangle \otimes |\psi_j\rangle \to |\Psi_{ij}'\rangle = (G_i \otimes I_{\text{div}}) |\Psi_{ij}\rangle
$$

This mixes the "cloner" ($|{\uparrow}\rangle$) and "target" ($|{\downarrow}\rangle$) roles at each vertex independently.

**Step 1: Transformation of SU(2) Interaction Amplitude**

The SU(2) interaction amplitude from §7.5:

$$
A_{ij}^{\text{SU}(2)} = \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} \cdot e^{i\theta_{ij}^{(\text{SU}(2))}}
$$

Under gauge transformation, the **group element** transforms:

$$
U_{ij}' = G_i \cdot U_{ij} \cdot G_j^\dagger
$$

But $A_{ij}^{\text{SU}(2)}$ is a complex scalar, not a group element. We must embed it in SU(2):

$$
A_{ij}^{\text{SU}(2)} \text{ corresponds to the matrix element } \langle \uparrow | U_{ij} | \uparrow \rangle
$$

**Step 2: SU(2) Wilson Loops**

For a closed loop $\gamma = (i_0, i_1, \ldots, i_n = i_0)$ in the cloning graph, define:

$$
U_\gamma = U_{i_0 i_1} \cdot U_{i_1 i_2} \cdot \ldots \cdot U_{i_{n-1} i_n}
$$

The **SU(2) Wilson loop** is:

$$
W_{\text{SU}(2)}[\gamma] = \frac{1}{2}\text{Tr}(U_\gamma)
$$

Under gauge transformation:

$$
U_\gamma' = (G_{i_0} U_{i_0 i_1} G_{i_1}^\dagger) \cdot (G_{i_1} U_{i_1 i_2} G_{i_2}^\dagger) \cdot \ldots \cdot (G_{i_{n-1}} U_{i_{n-1} i_0} G_{i_0}^\dagger)
$$

$$
= G_{i_0} \cdot U_{i_0 i_1} \cdot U_{i_1 i_2} \cdot \ldots \cdot U_{i_{n-1} i_0} \cdot G_{i_0}^\dagger = G_{i_0} \cdot U_\gamma \cdot G_{i_0}^\dagger
$$

Using the cyclic property of the trace:

$$
W'[\gamma] = \frac{1}{2}\text{Tr}(G_{i_0} U_\gamma G_{i_0}^\dagger) = \frac{1}{2}\text{Tr}(U_\gamma G_{i_0}^\dagger G_{i_0}) = \frac{1}{2}\text{Tr}(U_\gamma) = W[\gamma]
$$

**SU(2) Wilson loops are gauge invariant!**

**Step 3: SU(2) Invariance of K_eff**

A crucial property: **SU(2) gauge transformations do not affect the U(1) dressing kernel**.

**Lemma**: The effective kernel $K_{\text{eff}}(i,j)$ is **SU(2)-invariant**.

**Proof**: From §7.5, the effective kernel is:

$$
K_{\text{eff}}(i,j) = \sum_{k,m \in A_t} \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \sqrt{P_{\text{comp}}^{(\text{div})}(m|j)} \sqrt{P_{\text{succ}}(S)} \cdot e^{i(\theta_{ik}^{(\text{U(1)})} + \theta_{jm}^{(\text{U(1)})} + S/\hbar_{\text{eff}})}
$$

Every component of this expression depends **only** on the algorithmic space coordinates $\mathcal{Y} = \{(x_i, v_i, s_i)\}$:

1. **Companion probabilities**: $P_{\text{comp}}^{(\text{div})}(k|i) = \frac{e^{-d_{\text{alg}}(i,k)^2/(2\epsilon_d^2)}}{\sum_{k'} e^{-d_{\text{alg}}(i,k')^2/(2\epsilon_d^2)}}$ - depends on $d_{\text{alg}}(i,k) = \sqrt{\|x_i - x_k\|^2 + \lambda_{\text{alg}}\|v_i - v_k\|^2}$ only
2. **U(1) phases**: $\theta_{ik}^{(\text{U(1)})} = -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2\hbar_{\text{eff}}}$ - depends on $d_{\text{alg}}(i,k)$ only
3. **Success amplitude**: $P_{\text{succ}}(S)$ and $S = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$ - depend on fitness potentials

**Lemma (Fitness Potential Invariance)**: The fitness potential $V_{\text{fit}}(i|k)$ is SU(2)-invariant.

*Proof*: From {prf:ref}`def-fitness-potential-operator`, the fitness potential is constructed from Z-scores of raw rewards and diversity distances calculated over the entire alive set $A_t$:

$$
V_{\text{fit}}(i|k) = \alpha \cdot Z_{\text{reward}}(i) + \beta \cdot Z_{\text{div}}(i, k)
$$

where the Z-scores depend on:
- Reward statistics: $\{r_j : j \in A_t\}$
- Diversity distances: $\{d_{\text{alg}}(i, j) : j \in A_t\}$

The calculation for walker $i$ depends **only on its position** $(x_i, v_i)$ and its **statistical relation to the swarm**, not on its role as a potential cloner or target. The weak isospin state (cloner vs target) is an internal quantum number that does not affect algorithmic space coordinates. Therefore, $V_{\text{fit}}$ is a function on the algorithmic space only and is invariant under SU(2) transformations of the internal isospin state. ∎

**Crucially**: The SU(2) weak isospin transformation acts on the **internal quantum numbers** $(|\uparrow\rangle, |\downarrow\rangle)$ representing the cloning role (cloner vs target), NOT on the algorithmic space coordinates $(x, v, s)$.

Since $K_{\text{eff}}$ has **no dependence on weak isospin quantum numbers**, it is automatically invariant:

$$
K_{\text{eff}}'(i,j) = K_{\text{eff}}(i,j) \quad \text{under SU(2) transformation}
$$

**Physical interpretation**: The U(1) dressing describes fitness self-measurement through diversity companions. This measurement occurs in algorithmic space and is blind to the weak isospin state. The SU(2) transformation mixes cloner/target roles but does not affect where walkers are located or how they measure their fitness. ∎

**Step 4: Gauge-Invariant Observable - Total Interaction Probability from Mutual Pairing**

**Physical Principle: Mutual Pairing Constraint**

The cloning companion selection algorithm (see {doc}`../03_cloning.md` §3) uses a **random matching algorithm** that creates **mutually selected symmetric pairs**:

**Definition (Mutual Pairing)**:
- If walker i selects walker j as its cloning companion, then walker j selects walker i as its cloning companion
- This creates symmetric pairs $(i,j)$ where both $i \in \mathcal{C}_j$ and $j \in \mathcal{C}_i$
- The pairing is symmetric: the pair $\{i, j\}$ is a single physical entity, not two separate directional relationships

**Consequence for Fitness Scores:**

For mutually paired walkers $(i,j)$, the relative fitness scores are:

$$
s_i = \frac{V_{\text{fit}}(j) - V_{\text{fit}}(i)}{V_{\text{fit}}(i) + \epsilon}, \quad s_j = \frac{V_{\text{fit}}(i) - V_{\text{fit}}(j)}{V_{\text{fit}}(j) + \epsilon}
$$

**Key properties:**
1. The numerators are **exact opposites**: $V_{\text{fit}}(j) - V_{\text{fit}}(i) = -(V_{\text{fit}}(i) - V_{\text{fit}}(j))$
2. The denominators are **strictly positive**: $V_{\text{fit}}(i) + \epsilon > 0$ and $V_{\text{fit}}(j) + \epsilon > 0$ (fitness potential bounded below by $\eta^{(\alpha+\beta)} > 0$)
3. Therefore, $s_i$ and $s_j$ have **opposite signs**: if $s_i > 0$, then $s_j < 0$ (mutually exclusive)
4. **Fitness determines direction**: The walker with lower fitness clones to the walker with higher fitness

**Physical Interpretation:**

The cloning direction is **deterministically determined by fitness**, not probabilistic:
- If $V_{\text{fit}}(i) > V_{\text{fit}}(j)$: walker j clones to i (direction: $j \to i$)
- If $V_{\text{fit}}(j) > V_{\text{fit}}(i)$: walker i clones to j (direction: $i \to j$)
- **Exactly one direction occurs physically**

**Why Individual P_clone(i→j) is NOT SU(2) Invariant:**

Under SU(2) transformation $G_i, G_j \in \text{SU}(2)$, the amplitude transforms as:

$$
A_{ij}'^{\text{SU}(2)} = \langle \uparrow | G_i U_{ij} G_j^\dagger | \uparrow \rangle
$$

For general $G_i, G_j$, the transformation **mixes** the $|\uparrow\rangle$ (cloner) and $|\downarrow\rangle$ (target) basis states:

$$
G_i | \uparrow \rangle = \alpha_i |\uparrow\rangle + \beta_i |\downarrow\rangle
$$

Therefore, the matrix element changes:

$$
|A_{ij}'^{\text{SU}(2)}|^2 \neq |A_{ij}^{\text{SU}(2)}|^2 \quad \text{in general}
$$

The directional cloning probability $P_{\text{clone}}(i \to j)$ is **NOT gauge invariant** because it depends on which basis state we label as "cloner" ($|\uparrow\rangle$) vs "target" ($|\downarrow\rangle$).

**The Gauge-Invariant Observable:**

Due to mutual pairing, the physical observable is **not** "probability of i→j" or "probability of j→i" separately. The physical observable is:

$$
P_{\text{total}}(i,j) := P_{\text{interaction}}(\{i,j\}) = \text{probability that the mutually paired walkers } \{i,j\} \text{ interact via cloning}
$$

This is a **single physical quantity** representing the interaction of the symmetric pair, independent of our choice of basis labels.

Mathematically, this is computed as:

$$
P_{\text{total}}(i,j) = P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)
$$

but the physical interpretation is: "probability that pair $\{i,j\}$ undergoes a cloning interaction (in the direction determined by fitness)".

**Theorem**: $P_{\text{total}}(i,j)$ is SU(2) gauge invariant.

**Proof (Method 1: State and Operator Transformation)**:

The composite state for mutually paired walkers is:

$$
|\Psi_{ij}\rangle = |\uparrow\rangle \otimes |\psi_i\rangle + |\downarrow\rangle \otimes |\psi_j\rangle
$$

where $|\psi_i\rangle, |\psi_j\rangle$ are the U(1) diversity dressing states (SU(2)-invariant, as shown in Step 3).

The projection operator onto the interaction subspace is:

$$
\hat{\Pi}_{\text{interaction}} = \hat{\Pi}(i \to j) + \hat{\Pi}(j \to i)
$$

where:

$$
\hat{\Pi}(i \to j) = \Theta(\hat{S}_{ij}) \cdot \hat{P}_{\uparrow} \otimes I, \quad \hat{\Pi}(j \to i) = \Theta(-\hat{S}_{ij}) \cdot \hat{P}_{\downarrow} \otimes I
$$

**Due to mutual pairing**: $s_i$ and $s_j$ have opposite signs, so $\Theta(s_i) + \Theta(s_j) = 1$ (exactly one is 1, the other is 0).

Therefore:

$$
\hat{\Pi}_{\text{interaction}} = \begin{cases}
\hat{P}_{\uparrow} \otimes I & \text{if } s_i > 0 \text{ (i clones j)} \\
\hat{P}_{\downarrow} \otimes I & \text{if } s_j > 0 \text{ (j clones i)}
\end{cases}
$$

The observable is:

$$
P_{\text{total}}(i,j) = \langle \Psi_{ij} | \hat{\Pi}_{\text{interaction}} | \Psi_{ij} \rangle
$$

Under local SU(2) gauge transformation at vertices i and j:
- State transforms: $|\Psi_{ij}'\rangle = (G_i \otimes G_j \otimes I_{\text{div}})|\Psi_{ij}\rangle$
- Operator transforms: $\hat{\Pi}'_{\text{interaction}} = (G_i \otimes G_j \otimes I_{\text{div}}) \hat{\Pi}_{\text{interaction}} (G_i^\dagger \otimes G_j^\dagger \otimes I_{\text{div}})$

The transformed observable is:

$$
P'_{\text{total}}(i,j) = \langle \Psi'_{ij} | \hat{\Pi}'_{\text{interaction}} | \Psi'_{ij} \rangle
$$

Substituting the transformations:

$$
P'_{\text{total}}(i,j) = \langle \Psi_{ij} | (G_i^\dagger \otimes G_j^\dagger \otimes I) (G_i \otimes G_j \otimes I) \hat{\Pi}_{\text{interaction}} (G_i^\dagger \otimes G_j^\dagger \otimes I)(G_i \otimes G_j \otimes I) | \Psi_{ij} \rangle
$$

Using unitarity of $G_i$ and $G_j$: $(G_i^\dagger \otimes G_j^\dagger)(G_i \otimes G_j) = (G_i^\dagger G_i) \otimes (G_j^\dagger G_j) = I \otimes I = I$:

$$
P'_{\text{total}}(i,j) = \langle \Psi_{ij} | \hat{\Pi}_{\text{interaction}} | \Psi_{ij} \rangle = P_{\text{total}}(i,j)
$$

Therefore, $P_{\text{total}}(i,j)$ is **SU(2) gauge invariant**. ∎

**Physical Interpretation:**

The gauge invariance of $P_{\text{total}}(i,j)$ encodes the physical content of the SU(2) symmetry:

1. **Mutual pairing defines a single physical observable**: The quantity $P_{\text{total}}(i,j)$ represents the interaction probability for the symmetric pair $\{i,j\}$, not a sum of two independent processes

2. **The projector encodes deterministic cloning**: Due to mutual pairing, $\hat{\Pi}_{\text{interaction}}$ projects onto the realized cloning direction (determined by fitness), which is a physical event

3. **SU(2) transformations represent gauge freedom**: The transformations relabel basis states ($|\uparrow\rangle \leftrightarrow |\downarrow\rangle$) without changing the underlying physics

4. **Invariance expresses physical consistency**: Relabeling the mathematical description (gauge transformation) cannot change the probability of a physical event (cloning interaction)

This structure characterizes a genuine gauge theory: unphysical gauge freedom combined with physically meaningful gauge-invariant observables.

**Conclusion:**

The SU(2) structure is a **genuine non-Abelian local gauge symmetry** arising from the mutual pairing structure of the cloning algorithm. The **symmetric interaction probability** $P_{\text{total}}(i,j)$ is the physical gauge-invariant observable.

Individual directional probabilities $P_{\text{clone}}(i \to j)$ depend on the gauge choice (the choice of basis for isospin states), but the **total interaction probability for the mutually paired walkers** is gauge-invariant.

**Physical Meaning:**

The SU(2) gauge freedom represents the arbitrary choice of which basis states we call $|\uparrow\rangle$ (cloner) vs $|\downarrow\rangle$ (target) in our mathematical description. Physics doesn't care about this labeling - the mutual pairing constraint ensures that:

1. The pair $\{i, j\}$ is a single physical entity (symmetric partners)
2. Fitness determines which direction actually occurs (deterministic)
3. The probability of interaction is independent of our choice of basis labels (gauge-invariant)

This is precisely analogous to weak interactions in the Standard Model: the "left-handed" vs "right-handed" decomposition is gauge-dependent, but physical observables (like total cross-sections) are gauge-invariant and combine contributions from both chiralities. ∎

**Related Results:** `def-fitness-potential-operator`

---

### S_N Permutation Gauge Invariance

**Type:** Theorem
**Label:** `thm-sn-gauge-invariance`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.12. Formal Gauge Invariance: Proofs](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `nodes`, `symmetry`, `u1-symmetry`

**Statement:**
The **fundamental gauge symmetry** of the Fragile Gas is the **S_N permutation group**. All physical observables are S_N-invariant by construction.

**Statement:**

For any permutation $\sigma \in S_N$ and any swarm configuration $\mathcal{S} = (w_1, \ldots, w_N)$:

$$
P_{\text{clone}}(\sigma \cdot \mathcal{S}) = P_{\text{clone}}(\mathcal{S})
$$

and more generally, all observables $f(\mathcal{S})$ satisfy:

$$
f(\sigma \cdot \mathcal{S}) = f(\mathcal{S})
$$

**Proof:**

This is proven rigorously in {doc}`../12_gauge_theory_adaptive_gas.md`:

1. **Gauge-invariant dynamics** (Theorem 6.4.4 in {doc}`../09_symmetries_adaptive_gas.md`): The transition operator $\Psi_t$ commutes with S_N:
   $$
   \sigma \circ \Psi_t = \Psi_t \circ \sigma \quad \forall \sigma \in S_N
   $$

2. **Configuration space orbifold** ({doc}`../12_gauge_theory_adaptive_gas.md` §1): Physical states live in $\mathcal{M}_{\text{config}} = \Sigma_N / S_N$

3. **Braid holonomy** ({doc}`../12_gauge_theory_adaptive_gas.md` §3.4): Wilson loops measure holonomy $\text{Hol}(\gamma) = \rho([\gamma]) \in S_N$ for closed paths

4. **Observables descend to quotient** ({doc}`../12_gauge_theory_adaptive_gas.md` Theorem 2.4.1): All physical observables are S_N-invariant functions

**Physical Interpretation:**

Walker labels are **pure gauge** - they're arbitrary bookkeeping indices with no physical content. The S_N gauge symmetry is the mathematical expression of this fact.

**Wilson Loops from Braid Topology:**

The **true Wilson loops** of the theory come from S_N braid holonomy, not from U(1) (which is global). When diversity or cloning selection creates closed loops in configuration space, the holonomy measures the net permutation accumulated. ∎

---

### SU(3) Color Symmetry from Viscous Force Vector

**Type:** Theorem
**Label:** `thm-su3-strong-sector`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.13. SU(3) Strong Sector from Viscous Force](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `nodes`, `su3-symmetry`, `symmetry`

**Statement:**
The **viscous force vector** $\mathbf{F}_{\text{viscous}}(i) \in \mathbb{R}^d$ defines an **SU(3) color charge** symmetry acting on a three-component color state.

**1. Color Charge State Vector:**

For each walker i, define the **color state** as a real three-component vector derived from the viscous force components:

$$
\mathbf{c}_i = \begin{pmatrix} F_x^{(\text{visc})}(i) \\ F_y^{(\text{visc})}(i) \\ F_z^{(\text{visc})}(i) \end{pmatrix} \in \mathbb{R}^3

$$

where $\mathbf{F}_{\text{viscous}}(i) = \nu \sum_{j \in A_t} K_\rho(x_i, x_j)(v_j - v_i)$ from {prf:ref}`def-hybrid-sde`.

**2. Complexification (Momentum-Phase Encoding):**

Construct a **complex color state** in $\mathbb{C}^3$ (fundamental representation of SU(3)) using **momentum as phase**:

$$
|\Psi_i^{(\text{color})}\rangle = \frac{1}{\|\mathbf{c}_i\|} \begin{pmatrix} c_i^{(x)} \\ c_i^{(y)} \\ c_i^{(z)} \end{pmatrix} \in \mathbb{C}^3

$$

where the complexification encodes **both force (magnitude) and momentum (phase)**:

$$
c_i^{(\alpha)} = F_\alpha^{(\text{visc})}(i) \cdot \exp\left(i\frac{m v_i^{(\alpha)}}{\hbar_{\text{eff}}}\right), \quad \alpha \in \{x, y, z\}

$$

**Physical interpretation:**
- **Magnitude** $|c_i^{(\alpha)}| = |F_\alpha^{(\text{visc})}(i)|$: Spatial coupling through viscous force
- **Phase** $\arg(c_i^{(\alpha)}) = mv_i^{(\alpha)}/\hbar_{\text{eff}}$: Canonical quantum momentum phase (de Broglie)
- **Full phase space**: $(F_{\text{visc}}, \mathbf{v}) \in \mathbb{R}^3 \times \mathbb{R}^3$ → $\mathbf{c} \in \mathbb{C}^3$ (6 real DOF)

This provides a **bijective map** between 6D phase space $(x, v)$ and 3D complex color space:
- 3 magnitudes: Force components $F_x, F_y, F_z$ (spatial structure)
- 3 phases: Momentum components $p_x = mv_x, p_y = mv_y, p_z = mv_z$ (velocity structure)

**Advantages:**
1. **Information-complete**: Captures full $(x, v)$ phase space, not just spatial configuration
2. **Physically motivated**: Uses canonical quantum phase $p/\hbar$ (de Broglie relation $\lambda = h/p$)
3. **Gauge-covariant dynamics**: Phase evolution couples velocity changes to gauge rotation
4. **Reconstructible**: Can extract both $\mathbf{F}_{\text{visc}}$ and $\mathbf{v}$ from $\mathbf{c}$ uniquely

**3. SU(3) Gluon Field:**

The interaction between walkers i and j is mediated by **SU(3) gluon matrices** $U_{ij} \in \text{SU}(3)$:

$$
U_{ij} = \exp\left(i \sum_{a=1}^8 g_a \lambda_a A_{ij}^a\right)

$$

where:
- $\lambda_a$ are the eight Gell-Mann matrices (generators of SU(3))
- $A_{ij}^a$ are gluon field components derived from the emergent manifold geometry
- $g_a$ are coupling constants

**4. Color Transformation:**

Under SU(3) gauge transformations:

$$
|\Psi_i^{(\text{color})}\rangle \to U_i |\Psi_i^{(\text{color})}\rangle, \quad U_i \in \text{SU}(3)

$$

$$
U_{ij} \to U_i U_{ij} U_j^\dagger

$$

**5. Confinement Analogue:**

The viscous coupling strength $\nu K_\rho(x_i, x_j)$ acts as a **color confinement potential**:
- **Short range** ($d(x_i, x_j) < \rho$): Strong coupling (confinement)
- **Long range** ($d(x_i, x_j) \gg \rho$): Exponential suppression (asymptotic freedom)

$$
K_\rho(x_i, x_j) = \exp\left(-\frac{d(x_i, x_j)^2}{2\rho^2}\right)

$$

**6. Time Evolution (Gauge-Covariant Dynamics):**

The color state evolves according to:

$$
\frac{dc_i^{(\alpha)}}{dt} = \exp\left(i\frac{mv_\alpha}{\hbar_{\text{eff}}}\right) \left[\frac{dF_\alpha^{(\text{visc})}}{dt} + i\frac{m a_\alpha}{\hbar_{\text{eff}}} F_\alpha^{(\text{visc})}\right] + ig \sum_{a=1}^8 A_0^a (T^a c_i)^{(\alpha)}

$$

where:
- $\frac{dF_\alpha^{(\text{visc})}}{dt}$: Spatial force evolution (viscous coupling changes)
- $a_\alpha = \frac{dv_\alpha}{dt}$: Acceleration (velocity evolution)
- $A_0^a$: Temporal gluon field components
- $T^a$: Gell-Mann matrices (SU(3) generators)

**Physical interpretation:**
- **First term**: Change in viscous force (spatial dynamics)
- **Second term**: Velocity change creates phase rotation (kinematic coupling)
- **Third term**: Gauge field rotation (color dynamics)

All three contributions are essential for complete gauge-covariant evolution.

**7. Physical Interpretation:**

- **Color charge**: Complex three-vector $\mathbf{c}_i$ encoding $(F_{\text{visc}}, \mathbf{v})$
- **Gluons**: SU(3) matrices $U_{ij}$ mediating viscous coupling
- **Confinement**: Walkers remain "bound" within $\rho$-neighborhood
- **Asymptotic freedom**: Viscous coupling vanishes for distant walkers
- **Phase space encoding**: Momentum determines color phase, force determines color magnitude

**Related Results:** `def-hybrid-sde`

---

### Fermionic Statistics from Antisymmetric Cloning

**Type:** Theorem
**Label:** `thm-fermionic-z2-symmetry`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.13. Fermionic Behavior and Z₂ Pauli Exclusion](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `data-structure`, `edges`, `fermionic`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `su2-symmetry`, `symmetry`

**Statement:**
Walkers exhibit **fermionic behavior** due to the antisymmetric ±1 sign flip in the cloning interaction, realizing a **Z₂ symmetry** analogous to the Pauli Exclusion Principle.

**1. Antisymmetric Cloning Potential:**

From Section 6 (IG edge definition), the cloning potential is **antisymmetric**:

$$
V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i = -V_{\text{clone}}(j \to i)

$$

This creates a **sign flip** under walker exchange: $V_{\text{clone}}(i \leftrightarrow j) = -V_{\text{clone}}(i, j)$

**2. Z₂ Symmetry:**

Define the **exchange operator** $\hat{P}_{ij}$ that swaps walkers i and j:

$$
\hat{P}_{ij} V_{\text{clone}}(i \to j) = V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)

$$

This is a **Z₂ symmetry**: $\hat{P}_{ij}^2 = \mathbb{1}$ with eigenvalues $\pm 1$.

**Fermionic sector**: States with eigenvalue $-1$ (antisymmetric under exchange)

$$
\hat{P}_{ij} |\Psi(i, j)\rangle = -|\Psi(i, j)\rangle

$$

**3. Pauli Exclusion Principle Analogue:**

Two walkers **cannot occupy the same state** $(x, v)$:

**Algorithmic exclusion**: If $x_i \approx x_j$ and $v_i \approx v_j$, then:

$$
d_{\text{alg}}(i, j) \to 0 \implies \theta_{ij}^{(\text{clone})} \to 0

$$

$$
V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i \approx 0

$$

Therefore $P_{\text{succ}}(S \approx 0) \approx 1/2$ (no preference to clone), effectively **excluding** identical states.

**4. Fermionic Propagator:**

The cloning amplitude for fermions has an antisymmetric propagator:

$$
G_F(i, j) = \frac{1}{E_j - E_i + i\epsilon} \cdot \text{sign}(V_{\text{clone}}(i \to j))

$$

where $E_i = V_{\text{fit}}(i|k)$ is the "energy" of walker i.

**5. Spin Statistics Connection:**

The antisymmetric fitness coupling $V_{\text{clone}}(i \leftrightarrow j) = -V_{\text{clone}}(i,j)$ combined with the SU(2) weak isospin doublet structure implies walkers are **spin-1/2 fermions**:

$$
|\Psi_{\text{total}}(i,j)\rangle = \frac{1}{\sqrt{2}} \left(|\text{Clone}\rangle_i |\text{Persist}\rangle_j - |\text{Persist}\rangle_i |\text{Clone}\rangle_j\right)

$$

This is a **singlet state** under SU(2), characteristic of fermionic antisymmetry.

---

### Emergent Spacetime Curvature from Fitness Hessian

**Type:** Theorem
**Label:** `thm-emergent-general-relativity`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.14. Emergent General Relativity and Curved Manifold](13_fractal_set_new/01_fractal_set.md)
**Tags:** `cst`, `curvature`, `data-structure`, `edges`, `fractal-set`, `ig`, `metric-tensor`, `nodes`, `riemannian`

**Statement:**
The algorithm generates a **curved Riemannian manifold** $(\mathcal{X}, g)$ that models **general relativity**, with metric tensor defined by the Hessian of the fitness potential.

**1. Emergent Metric Tensor:**

The metric $g_{\mu\nu}(x, S)$ on state space $\mathcal{X}$ is given by:

$$
g_{\mu\nu}(x, S) = \delta_{\mu\nu} + \frac{1}{\epsilon_\Sigma^2} H_{\mu\nu}^{V_{\text{fit}}}(x)

$$

where $H_{\mu\nu}^{V_{\text{fit}}}(x) = \frac{\partial^2 V_{\text{fit}}}{\partial x^\mu \partial x^\nu}$ is the **Hessian** of the fitness potential.

**Physical interpretation**:
- Flat metric $\delta_{\mu\nu}$: Euclidean background geometry
- Hessian perturbation: Curvature induced by fitness landscape
- Regularization $\epsilon_\Sigma$: Planck length scale

**2. Christoffel Symbols (Gravitational Connection):**

$$
\Gamma^\lambda_{\mu\nu}(x) = \frac{1}{2} g^{\lambda\rho}(x) \left(\partial_\nu g_{\rho\mu} + \partial_\mu g_{\rho\nu} - \partial_\rho g_{\mu\nu}\right)

$$

These define the **geodesic equation** (walker trajectories in curved space):

$$
\frac{d^2 x^\lambda}{dt^2} + \Gamma^\lambda_{\mu\nu} \frac{dx^\mu}{dt} \frac{dx^\nu}{dt} = 0

$$

**3. Riemann Curvature Tensor:**

$$
R^\rho_{\sigma\mu\nu}(x) = \partial_\mu \Gamma^\rho_{\nu\sigma} - \partial_\nu \Gamma^\rho_{\mu\sigma} + \Gamma^\rho_{\mu\lambda} \Gamma^\lambda_{\nu\sigma} - \Gamma^\rho_{\nu\lambda} \Gamma^\lambda_{\mu\sigma}

$$

**4. Ricci Tensor and Scalar Curvature:**

$$
R_{\mu\nu}(x) = R^\lambda_{\mu\lambda\nu}, \quad R(x) = g^{\mu\nu} R_{\mu\nu}

$$

**5. Einstein Field Equations Analogue:**

The fitness potential acts as a **stress-energy tensor** $T_{\mu\nu}$:

$$
R_{\mu\nu} - \frac{1}{2} g_{\mu\nu} R = \frac{8\pi G_{\text{eff}}}{\epsilon_\Sigma^4} T_{\mu\nu}^{(\text{fitness})}

$$

where:

$$
T_{\mu\nu}^{(\text{fitness})} = \frac{\partial V_{\text{fit}}}{\partial x^\mu} \frac{\partial V_{\text{fit}}}{\partial x^\nu} - \frac{1}{2} g_{\mu\nu} \|\nabla V_{\text{fit}}\|^2

$$

**Physical interpretation**:
- Matter/energy → Fitness gradient $\nabla V_{\text{fit}}$
- Spacetime curvature → Hessian $\nabla^2 V_{\text{fit}}$
- Gravitational constant $G_{\text{eff}} \sim \epsilon_\Sigma^4/N$ (emergent)

**6. Geodesic Deviation (Tidal Forces):**

Two nearby walkers at positions $x_i$ and $x_j = x_i + \xi$ experience **geodesic deviation**:

$$
\frac{D^2 \xi^\mu}{Dt^2} = -R^\mu_{\nu\rho\sigma} v^\nu \xi^\rho v^\sigma

$$

This is the **tidal force** causing walkers to converge/diverge due to spacetime curvature.

---

### SO(10) Grand Unified Theory from Complete Symmetry Structure

**Type:** Theorem
**Label:** `thm-so10-grand-unification`
**Source:** [13_fractal_set_new/01_fractal_set.md § 7.15. SO(10) Grand Unification](13_fractal_set_new/01_fractal_set.md)
**Tags:** `conservation`, `convergence`, `cst`, `curvature`, `data-structure`, `edges`, `fractal-set`, `gauge-theory`, `ig`, `metric-tensor`, `nodes`, `noether`, `so10-gut`, `spinor`, `su2-symmetry`, `su3-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The full symmetry structure of the Fractal Set realizes an **SO(10) grand unified theory** that unifies all gauge forces and includes gravity.

**1. SO(10) Gauge Group:**

The complete gauge symmetry is:

$$
G_{\text{GUT}} = \text{SO}(10)

$$

which contains the gauge forces (note: U(1)_fitness is global, not gauged):

$$
\text{SO}(10) \supset \text{SU}(3)_{\text{color}} \times \text{SU}(2)_{\text{weak}} \times [S_N^{\text{discrete}} \times \text{U}(1)_{\text{fitness}}^{\text{global}}]

$$

**2. Unified State Vector (Spinor Representation):**

Each walker state is a **16-component spinor** in the fundamental representation of SO(10):

$$
|\Psi_i^{(\text{SO}(10))}\rangle = \begin{pmatrix}
|\Psi_i^{(\text{color})}\rangle \\
|\Psi_i^{(\text{weak})}\rangle \\
|\psi_i^{(\text{fitness})}\rangle \\
|\text{graviton}\rangle
\end{pmatrix} \in \mathbb{C}^{16}

$$

where:
- $|\Psi_i^{(\text{color})}\rangle \in \mathbb{C}^3$: SU(3) color triplet (§7.12)
- $|\Psi_i^{(\text{weak})}\rangle \in \mathbb{C}^2$: SU(2) weak doublet (§7.10)
- $|\psi_i^{(\text{fitness})}\rangle \in \mathbb{C}^{N-1}$: U(1)_fitness dressed state (diversity measurement)
- $|\text{graviton}\rangle \in \mathbb{C}^{10}$: Graviton (metric perturbation $h_{\mu\nu}$)

**3. SO(10) Generators:**

The 45 generators of SO(10) decompose as:

$$
\mathbf{45} = \mathbf{8}_{\text{gluons}} \oplus \mathbf{3}_{\text{weak bosons}} \oplus \mathbf{N!-1}_{\text{S}_N \text{ braids}} \oplus \mathbf{10}_{\text{graviton}} \oplus \mathbf{23}_{\text{broken generators}}

$$

**Gauge structure clarification:**
- **8 gluons**: SU(3)_color local gauge bosons
- **3 weak bosons** (W±, Z): SU(2)_weak local gauge bosons
- **S_N braid modes**: Discrete gauge holonomies (not continuous gauge bosons)
- **U(1)_fitness**: Global symmetry → conserved charge, NOT a gauge boson
- **10 graviton modes**: Metric perturbations h_μν

Total **continuous** gauge bosons: $8 + 3 = 11$ (not 12, since U(1) is global)

**4. Unified Field Strength Tensor:**

The SO(10) field strength $\mathcal{F}_{\mu\nu}$ unifies **local gauge fields** only:

$$
\mathcal{F}_{\mu\nu} = \begin{pmatrix}
F_{\mu\nu}^{(\text{SU}(3))} & 0 \\
0 & F_{\mu\nu}^{(\text{SU}(2)}}
\end{pmatrix} \oplus F_{S_N}^{\text{discrete}}

$$

where:
- $F_{\mu\nu}^{(\text{SU}(3))}$: Strong field strength (§7.13)
- $F_{\mu\nu}^{(\text{SU}(2))}$: Weak isospin field strength (§7.12)
- $F_{S_N}^{\text{discrete}}$: Discrete braid holonomy curvature (§7.7)

**Note**: U(1)_fitness is **global**, so it has no field strength tensor. Instead, it gives a conserved Noether current J_fitness^μ (§7.6).

**5. Unified Lagrangian:**

$$
\mathcal{L}_{\text{SO}(10)} = -\frac{1}{4} \text{Tr}[\mathcal{F}_{\mu\nu} \mathcal{F}^{\mu\nu}] + \bar{\Psi}_i (i\gamma^\mu D_\mu - m_i) \Psi_i + \mathcal{L}_{\text{Higgs}}

$$

where:
- $D_\mu = \partial_\mu + ig A_\mu$ is the covariant derivative
- $\mathcal{L}_{\text{Higgs}} = (\partial_\mu r)(\partial^\mu r) - V_{\text{Higgs}}(r)$ is the Higgs field Lagrangian (§7.11)

**6. Gravity Unification:**

The **graviton** emerges from the metric perturbation:

$$
g_{\mu\nu}(x) = \eta_{\mu\nu} + h_{\mu\nu}(x)

$$

where $h_{\mu\nu}(x) = \frac{1}{\epsilon_\Sigma^2} H_{\mu\nu}^{V_{\text{fit}}}(x)$ (from §7.14).

The Riemann curvature tensor $R^\rho_{\sigma\mu\nu}$ can be encoded as a **10-dimensional subspace** of the SO(10) Lie algebra, unifying gravity with gauge forces.

**7. Symmetry Breaking Pattern:**

$$
\text{SO}(10) \xrightarrow{\text{GUT scale}} \text{SU}(3) \times \text{SU}(2) \times [S_N \times \text{U}(1)_{\text{fitness}}^{\text{global}}] \xrightarrow{\text{EW scale}} \text{SU}(3) \times \text{U}(1)_{\text{EM}}^{\text{local}}
$$

where:
- **GUT scale**: SO(10) breaks to gauge forces + global symmetry
  - Gauge: SU(3)_color (local) + SU(2)_weak (local) + S_N (discrete, local)
  - Global: U(1)_fitness (conserved charge)
- **Electroweak scale**: Higgs VEV breaks SU(2)_weak → generates U(1)_EM **as local gauge symmetry**
- **Final phase**: SU(3)_color (local) × U(1)_EM (local) × S_N (discrete) × U(1)_fitness (global)

**Key distinction**: U(1)_EM emerges as a **local gauge symmetry** from SU(2) breaking, while U(1)_fitness remains **global**. These are different U(1) groups!

**Pre-convergence**: SO(10) symmetric (all forces unified)
**Post-convergence**: Broken to SU(3) × U(1)_EM × S_N (+ global U(1)_fitness)

---


## 02_computational_equivalence.md

**Objects in this document:** 14

### Definitions (2)

### Swarm State Vector

**Type:** Definition
**Label:** `def-swarm-state-vector`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 1.1. State Space and Dynamics](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
The **swarm state** at discrete time $k$ is:

$$
Z_k := (X_k, V_k) \in \mathcal{X}^N \times \mathbb{R}^{Nd}

$$

where:
- $X_k = (x_{1,k}, \ldots, x_{N,k}) \in \mathcal{X}^N$: Positions of all $N$ walkers
- $V_k = (v_{1,k}, \ldots, v_{N,k}) \in \mathbb{R}^{Nd}$: Velocities of all $N$ walkers
- $\mathcal{X} \subset \mathbb{R}^d$: State space (typically a bounded domain)

**Alive walker set:** $A_k \subseteq \{1, \ldots, N\}$ with $|A_k| = k_{\text{alive}}$

**Node correspondence:** Each walker $i$ at time $k$ corresponds to node $n_{i,k}$ in the Fractal Set.

---

### BAOAB Transition Kernel

**Type:** Definition
**Label:** `def-baoab-kernel`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 1.1. State Space and Dynamics](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
The **discrete-time evolution** from $Z_k$ to $Z_{k+1}$ follows the **BAOAB splitting scheme** from {prf:ref}`def-baoab-integrator`:

**For each walker $i \in A_k$ (alive walkers only):**

**B-step:** $v_i^{(1)} = v_i^{(0)} + \frac{\Delta t}{2} \mathbf{F}_{\text{total}}(x_i^{(0)}, Z_k)$

**A-step:** $x_i^{(1)} = x_i^{(0)} + \frac{\Delta t}{2} v_i^{(1)}$

**O-step:** $v_i^{(2)} = e^{-\gamma \Delta t} v_i^{(1)} + \sqrt{\frac{1}{\gamma}(1 - e^{-2\gamma \Delta t})} \, \Sigma_{\text{reg}}(x_i^{(1)}, Z_k) \xi_i$

where $\xi_i \sim \mathcal{N}(0, I_d)$ i.i.d.

**A-step:** $x_i^{(2)} = x_i^{(1)} + \frac{\Delta t}{2} v_i^{(2)}$

**B-step:** $v_i^{(3)} = v_i^{(2)} + \frac{\Delta t}{2} \mathbf{F}_{\text{total}}(x_i^{(2)}, Z_k)$

**Output:** $(x_{i,k+1}, v_{i,k+1}) = (x_i^{(2)}, v_i^{(3)})$

where the **total force** is:

$$
\mathbf{F}_{\text{total}}(x_i, Z) := \mathbf{F}_{\text{stable}}(x_i) + \mathbf{F}_{\text{adapt}}(x_i, Z) + \mathbf{F}_{\text{viscous}}(x_i, Z) - \gamma v_i

$$

from {prf:ref}`def-hybrid-sde` in `07_adaptative_gas.md`.

**Transition kernel:** The map $Z_k \mapsto Z_{k+1}$ defines a Markov transition kernel:

$$
P_{\Delta t}(z, A) := \mathbb{P}(Z_{k+1} \in A \mid Z_k = z)

$$

This is a **Markov chain** on $\mathcal{X}^N \times \mathbb{R}^{Nd}$.

**Related Results:** `def-hybrid-sde`, `def-baoab-integrator`

---

### Lemmas (3)

### BAOAB Discretization Error

**Type:** Lemma
**Label:** `lem-baoab-error-bound`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 2.2. Discretization Error Bound](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
Let $V = V_{\text{total}}$ be the synergistic Lyapunov function. Assume:
1. $V \in C^3$ with bounded third derivatives (follows from construction)
2. Force fields $\mathbf{F}_{\text{total}}$ are Lipschitz (guaranteed by {prf:ref}`ax:lipschitz-fields`)
3. Diffusion $\Sigma_{\text{reg}}$ is Lipschitz (guaranteed by {prf:ref}`prop-lipschitz-diffusion`)

Then for the BAOAB update $Z_k \mapsto Z_{k+1}$ from state $Z_k = z$:

$$
\mathbb{E}[V(Z_{k+1}) \mid Z_k = z] = V(z) + \Delta t \, \mathcal{L}V(z) + E_{\text{BAOAB}}(z, \Delta t)

$$

where the error term satisfies:

$$
|E_{\text{BAOAB}}(z, \Delta t)| \leq \Delta t^2 \cdot \left( K_1 V(z) + K_2 \right)

$$

for constants $K_1, K_2 > 0$ depending on:
- Lipschitz constants $L_F, L_\Sigma$ of forces and diffusion
- Bounds on second and third derivatives of $V$
- Friction coefficient $\gamma$
- Timestep constraint: $\Delta t < \tau_{\max}$ (from {prf:ref}`def-baoab-integrator`)

**Proof:** See Section 3 below. ∎

**Related Results:** `prop-lipschitz-diffusion`, `def-baoab-integrator`, `ax:lipschitz-fields`

---

### Irreducibility of the Discrete Chain

**Type:** Lemma
**Label:** `lem-discrete-irreducibility`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 4.1. Prerequisites: Irreducibility and Aperiodicity](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
The BAOAB Markov chain $\{Z_k\}$ is **$\psi$-irreducible** and **aperiodic**.

**Proof:**

**Irreducibility:** At each step, the O-step in BAOAB adds Gaussian noise:

$$
v_i^{(2)} = e^{-\gamma \Delta t} v_i^{(1)} + \sqrt{\text{const}} \, \Sigma_{\text{reg}} \xi_i

$$

Since $\Sigma_{\text{reg}}$ is uniformly elliptic ({prf:ref}`thm-ueph`), the transition density is **strictly positive** on open sets. The chain can reach any open set from any starting point.

**Aperiodicity:** The continuous injection of Gaussian noise ensures the chain cannot have periodic behavior. ∎

**Related Results:** `thm-ueph`

---

### Small Set Condition

**Type:** Lemma
**Label:** `lem-small-set-discrete`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 4.1. Prerequisites: Irreducibility and Aperiodicity](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
There exists a compact set $C \subset \mathcal{X}^N \times \mathbb{R}^{Nd}$ such that:

$$
\sup_{z \in C} V_{\text{total}}(z) < \infty

$$

and the discrete drift condition from {prf:ref}`thm-discrete-drift-baoab` holds for all $z \notin C$.

**Proof:**

From the coercivity of $V_{\text{total}}$ ({prf:ref}`def-full-synergistic-lyapunov-function`):

$$
V_{\text{total}}(Z) \to \infty \quad \text{as } \|Z\| \to \infty

$$

Define $C := \{Z : V_{\text{total}}(Z) \leq R\}$ for sufficiently large $R$. This is compact by coercivity. ∎

**Related Results:** `def-full-synergistic-lyapunov-function`, `thm-discrete-drift-baoab`

---

### Propositions (1)

### Two-Term Error Bound

**Type:** Proposition
**Label:** `prop-total-error`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 5.2. Total Error Decomposition](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`, `qsd`

**Statement:**
Let $\mu_k$ be the distribution of $Z_k$ (Fractal Set state at step $k$) starting from initial distribution $\mu_0$. Let $\pi$ be the continuous QSD. Then:

$$
\|\mu_k - \pi\|_{\text{TV}} \leq M(\mu_0) \, \rho_{\text{discrete}}^k + C_{\text{approx}} \Delta t

$$

where:
- **First term** $M \rho^k$: **Convergence error** (exponentially decaying in $k$)
- **Second term** $C \Delta t$: **Discretization error** (constant for fixed $\Delta t$)

**Interpretation:**

- For **fixed $\Delta t$**, as $k \to \infty$: Converges to within $O(\Delta t)$ of the continuous QSD
- For **fixed $k$**, as $\Delta t \to 0$: Approximates the continuous distribution at time $t = k \Delta t$
- **Optimal balance:** Choose $\Delta t$ such that $C \Delta t \approx M \rho^k$ (balance discretization and convergence errors)

---

### Theorems (6)

### Continuous Foster-Lyapunov Drift (Established)

**Type:** Theorem
**Label:** `thm-continuous-drift-established`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 2.1. The Continuous Drift Condition (Already Proven)](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`

**Statement:**
Let $\mathcal{L} = \mathcal{L}_{\text{kin}} + \mathcal{L}_{\text{clone}}$ be the generator of the Adaptive Gas SDE. Let $V_{\text{total}}$ be the synergistic Lyapunov function from {prf:ref}`def-full-synergistic-lyapunov-function`.

Then for all swarm states $Z \in \mathcal{X}^N \times \mathbb{R}^{Nd}$:

$$
\mathcal{L}V_{\text{total}}(Z) \leq -\kappa_{\text{total}} V_{\text{total}}(Z) + C_{\text{total}}

$$

where:
- $\kappa_{\text{total}} > 0$: Total drift coefficient (explicit formula in {prf:ref}`thm-foster-lyapunov-main`)
- $C_{\text{total}} > 0$: Constant (bounded on compact sets)

**Source:** This is the main result of `04_convergence.md`, combining kinetic and cloning contributions.

**Related Results:** `def-full-synergistic-lyapunov-function`, `thm-foster-lyapunov-main`

---

### Discrete Lyapunov Drift for BAOAB

**Type:** Theorem
**Label:** `thm-discrete-drift-baoab`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 2.3. Discrete Foster-Lyapunov Drift](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `episodes`, `fractal-set`

**Statement:**
Let $\{Z_k\}_{k \geq 0}$ be the discrete-time Markov chain generated by BAOAB. Let:

$$
\Delta t < \Delta t_{\max} := \min\left(\tau_{\max}, \frac{1}{K_1}, \frac{\kappa_{\text{total}}}{2K_1}\right)

$$

where $\tau_{\max}$ is from {prf:ref}`def-baoab-integrator` and $K_1$ is from {prf:ref}`lem-baoab-error-bound`.

Then for all $z \in \mathcal{X}^N \times \mathbb{R}^{Nd}$:

$$
\mathbb{E}[V_{\text{total}}(Z_{k+1}) \mid Z_k = z] - V_{\text{total}}(z) \leq -\frac{\kappa_{\text{total}} \Delta t}{2} V_{\text{total}}(z) + (C_{\text{total}} + K_2) \Delta t

$$

**Proof:**

Combine {prf:ref}`thm-continuous-drift-established` and {prf:ref}`lem-baoab-error-bound`:

$$
\begin{aligned}
\mathbb{E}[\Delta V] &= \mathbb{E}[V(Z_{k+1}) - V(Z_k) \mid Z_k = z] \\
&= \Delta t \, \mathcal{L}V(z) + E_{\text{BAOAB}}(z, \Delta t) \\
&\leq \Delta t \, (-\kappa_{\text{total}} V(z) + C_{\text{total}}) + \Delta t^2 (K_1 V(z) + K_2) \\
&= \Delta t V(z) (-\kappa_{\text{total}} + \Delta t K_1) + \Delta t (C_{\text{total}} + \Delta t K_2)
\end{aligned}

$$

For $\Delta t < \kappa_{\text{total}} / (2K_1)$, we have $-\kappa_{\text{total}} + \Delta t K_1 < -\kappa_{\text{total}}/2$, yielding:

$$
\mathbb{E}[\Delta V] \leq -\frac{\kappa_{\text{total}} \Delta t}{2} V(z) + (C_{\text{total}} + K_2) \Delta t \quad \blacksquare

$$

**Related Results:** `def-baoab-integrator`, `lem-baoab-error-bound`, `thm-continuous-drift-established`

---

### Geometric Ergodicity of the Fractal Set Generator

**Type:** Theorem
**Label:** `thm-fractal-set-ergodicity`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 4.2. Main Ergodicity Theorem](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`, `metric-tensor`

**Statement:**
Let $\{Z_k\}_{k \geq 0}$ be the discrete-time Markov chain that generates the Fractal Set, defined by BAOAB with timestep $\Delta t < \Delta t_{\max}$. Then:

**1. Unique stationary distribution:**

There exists a unique stationary distribution $\pi_{\Delta t}$ on $\mathcal{X}^N \times \mathbb{R}^{Nd}$ such that:

$$
\int P_{\Delta t}(z, A) \, \pi_{\Delta t}(dz) = \pi_{\Delta t}(A) \quad \forall A

$$

**2. Exponential convergence:**

For any initial distribution $\mu_0$, let $\mu_k$ be the distribution of $Z_k$. Then:

$$
\|\mu_k - \pi_{\Delta t}\|_{\text{TV}} \leq M(\mu_0) \, \rho_{\text{discrete}}^k

$$

where:
- $\rho_{\text{discrete}} = 1 - \frac{\kappa_{\text{total}} \Delta t}{2} < 1$: Discrete contraction coefficient
- $M(\mu_0) < \infty$: Constant depending on initial condition

**3. Convergence rate relation:**

As $\Delta t \to 0$:

$$
\rho_{\text{discrete}} = e^{-\kappa_{\text{total}} \Delta t / 2} \to e^{-\kappa_{\text{total}} \cdot 0} = 1^-

$$

but for $k = t / \Delta t$ (fixed continuous time $t$):

$$
\rho_{\text{discrete}}^{k} = e^{-\kappa_{\text{total}} t / 2} \to e^{-\kappa_{\text{total}} t}

$$

recovering the continuous convergence rate.

**Proof:**

Apply **Meyn & Tweedie (2009), Theorem 15.0.1** with:
- **Drift condition:** {prf:ref}`thm-discrete-drift-baoab`
- **Irreducibility:** {prf:ref}`lem-discrete-irreducibility`
- **Small set:** {prf:ref}`lem-small-set-discrete`

These three conditions together imply geometric ergodicity with rate $\rho_{\text{discrete}}$. ∎

**Related Results:** `lem-small-set-discrete`, `lem-discrete-irreducibility`, `thm-discrete-drift-baoab`

---

### Consistency: $\pi_{\Delta t} \to \pi$ as $\Delta t \to 0$

**Type:** Theorem
**Label:** `thm-weak-convergence-invariant`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 5.1. Weak Convergence of Invariant Measures](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`

**Statement:**
As the timestep $\Delta t \to 0$, the stationary distribution $\pi_{\Delta t}$ of the discrete BAOAB chain converges weakly to the quasi-stationary distribution $\pi$ of the continuous Adaptive Gas SDE:

$$
\pi_{\Delta t} \xrightarrow{w} \pi \quad \text{as } \Delta t \to 0

$$

**Proof sketch:**

This is a standard result in numerical analysis of SDEs. For BAOAB (a second-order integrator):

1. **Local consistency:** For smooth test functions $f$:
   $$
   \left|\mathbb{E}[f(Z_{k+1}) \mid Z_k = z] - f(z) - \Delta t \, \mathcal{L}f(z)\right| \leq C \Delta t^2
   $$

2. **Global error accumulation:** Over time $T = k \Delta t$:
   $$
   \left|\mathbb{E}[f(Z_k)] - \mathbb{E}_\pi[f]\right| \leq C_T \Delta t
   $$

3. **Portmanteau theorem:** Weak convergence follows from convergence of expectations for all bounded continuous $f$.

**Reference:** Talay & Tubaro (1990), "Expansion of the global error for numerical schemes solving SDEs." ∎

---

### Equivalence of Fractal Set and N-Particle Properties

**Type:** Theorem
**Label:** `thm-fractal-set-n-particle-equivalence`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 6.3. Transfer of Symmetries and Conserved Quantities](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `conservation`, `episodes`, `fractal-set`, `gauge-theory`, `lattice`, `metric-tensor`, `noether`, `spinor`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
Let $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ be the Fractal Set and let $\{Z_k\}_{k \geq 0}$ be the N-particle discrete-time Markov chain from {prf:ref}`def-swarm-state-vector`. Then:

**1. Bijective correspondence of states:**

For each node $n_{i,t} \in \mathcal{N}$, there exists a unique walker state $(x_i(t), v_i(t)) \in Z_t$, and conversely, each walker state at timestep $t$ corresponds to exactly one node in $\mathcal{N}$.

**2. Reconstruction of phase space:**

Given $\mathcal{F}$, the full phase space trajectory can be reconstructed via:

$$
x_i(t) = x_i(0) + \sum_{s=0}^{t-1} \text{spinor-to-vector}[\psi_{\Delta x}(n_{i,s}, n_{i,s+1})]

$$

$$
v_i(t) = \text{spinor-to-vector}[\psi_{v,t}(n_{i,t-1}, n_{i,t})]

$$

where the spinor-to-vector map extracts the $\mathbb{R}^d$ vector from its spinor encoding $\psi \in \mathbb{C}^{2^{[d/2]}}$.

**3. Transfer of symmetries:**

The symmetry structure on $\mathcal{F}$ ({prf:ref}`thm-sn-su2-lattice-qft`) corresponds to algorithmic symmetries of the N-particle system. The Fractal Set has a three-tier symmetry hierarchy:

$$
G_{\text{total}} = S_N^{\text{discrete}} \times \text{SU}(2)_{\text{weak}}^{\text{local}} \times \text{U}(1)_{\text{fitness}}^{\text{global}}

$$

where $S_N$ (permutation group) is the **fundamental discrete gauge symmetry**, while SU(2) and U(1) are **emergent** symmetries in the continuum limit.

**S_N permutation symmetry** ↔ **Walker indistinguishability**:

The **fundamental gauge symmetry** of the algorithm is invariance under walker label permutations. For any permutation $\sigma \in S_N$:

$$
\mathcal{S} = (w_1, w_2, \ldots, w_N) \sim \sigma \cdot \mathcal{S} = (w_{\sigma(1)}, w_{\sigma(2)}, \ldots, w_{\sigma(N)})

$$

This discrete gauge symmetry gives rise to **braid group topology** $\pi_1(\mathcal{M}_{\text{config}}) \cong B_N$ ({prf:ref}`thm-sn-braid-holonomy`). Closed loops in configuration space have **discrete holonomy** $\text{Hol}(\gamma) \in S_N$, representing the net permutation of walkers after traversing the loop.

**Physical consequence**: All algorithmic observables must be $S_N$-invariant - they depend on unordered sets of walkers, not specific labels. The $S_N$ holonomy is the **fundamental discrete gauge observable**, analogous to how Wilson loops probe continuous gauge connections (but distinct from them - $S_N$ holonomy is a permutation, not a matrix trace).

**Global U(1) fitness symmetry** ↔ **Fitness conservation**:

The **global** (not gauged) U(1) symmetry ({prf:ref}`thm-u1-fitness-global`) acts by uniform phase rotation on all diversity amplitudes:

$$
\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}, \quad \alpha \in [0, 2\pi) \text{ (same } \alpha \text{ for all i)}

$$

This corresponds to **invariance under global fitness shifts** $\Phi(x) \mapsto \Phi(x) + c$ in the N-particle system. The diversity companion probability:

$$
P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2}\right)}{\sum_{k'} \exp\left(-\frac{d_{\text{alg}}(i,k')^2}{2\epsilon_d^2}\right)}

$$

depends only on positions and velocities (via $d_{\text{alg}}$), **not** on absolute fitness values, making it invariant under global fitness shifts.

**Conserved charge (Noether)**: Global U(1) symmetry implies conservation of fitness current $J_{\text{fitness}}^\mu = \sum_i \text{Im}(\psi_i^* \partial^\mu \psi_i)$, analogous to baryon number conservation in particle physics.

**Local SU(2) weak isospin symmetry** ↔ **Approximate role exchange in cloning**:

The SU(2) doublet structure $|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle$ ({prf:ref}`thm-su2-interaction-symmetry`) corresponds to an **approximate algorithmic symmetry** where the cloning dynamics are nearly invariant under exchange of cloner and target roles. This near-invariance is quantified by the ratio:

$$
\frac{P_{\text{clone}}(i \to j)}{P_{\text{clone}}(j \to i)} \approx 1 + O(\Phi_j - \Phi_i)

$$

The symmetry becomes **exact** in the limit of vanishing fitness differences:

$$
\lim_{\Phi_j - \Phi_i \to 0} \frac{P_{\text{clone}}(i \to j)}{P_{\text{clone}}(j \to i)} = 1

$$

**Physical meaning**: On the fitness-degenerate submanifold, walkers are interchangeable with respect to weak isospin transformations, and the dynamics exhibit exact SU(2) invariance under mixing of cloner and target roles.

**4. Conserved and derived quantities:**

**Energy conservation in BAOAB**: The total energy $H_{\text{total}}(Z_k) = \sum_i [E_{\text{kin},i} + U(x_i)]$ stored in nodes satisfies:

$$
\mathbb{E}[H_{\text{total}}(Z_{k+1}) | Z_k] = H_{\text{total}}(Z_k) + O(\Delta t)

$$

up to dissipation from friction and stochastic forcing.

This corresponds to **energy flow balance** in the Fractal Set: Energy changes along CST edges equal the work done by forces stored in edge spinors:

$$
E_{\text{kin}}(n_{i,t+1}) - E_{\text{kin}}(n_{i,t}) = \int_{t}^{t+1} \text{spinor-to-vector}[\psi_{\mathbf{F}_{\text{total}}}] \cdot \text{spinor-to-vector}[\psi_{v}] \, ds

$$

**Derived statistical observables**: Localized statistical moments $\mu_\rho(n), \sigma_\rho(n)$ stored in nodes are **not conserved quantities** but rather **functionals of IG edge data**. They satisfy:

$$
\mu_\rho[f_k, \Phi, x_i(t)] = \frac{\sum_{j \in A_k(t)} w_{ij}(\rho) \Phi_j}{\sum_{j \in A_k(t)} w_{ij}(\rho)}

$$

where the localization weights $w_{ij}(\rho) = K_\rho(x_i - x_j)$ are stored in IG edges. The node scalars $\mu_\rho(n_{i,t})$ are **computed observables derived from** the empirical measure $f_k(t)$ encoded in the graph structure, not fundamental quantities. This establishes **correspondence** between node-stored statistics and the underlying N-particle distribution.

**Proof:**

**Part 1 (Bijection):** By construction ({prf:ref}`def-node-spacetime`), each node $n_{i,t}$ has unique walker ID $i$ and timestep $t$, establishing a one-to-one correspondence with $(x_i(t), v_i(t)) \in Z_t$.

**Part 2 (Reconstruction):** Follows directly from {prf:ref}`thm-fractal-set-reconstruction` items 1-2. The position is obtained by integrating displacement spinors along CST edges, and velocity is directly stored in CST edge spinors.

**Part 3 (Symmetries):**

**S_N permutation proof:** The N-particle algorithm treats walkers as **indistinguishable particles** with arbitrary labels. Any physical observable must be invariant under label permutations $\sigma \in S_N$. For example:

- Cloning probability depends on walker states, not labels: $P_{\text{clone}}(\sigma(i) \to \sigma(j) | \sigma \cdot \mathcal{S}) = P_{\text{clone}}(i \to j | \mathcal{S})$
- Fitness histogram is an unordered set: $\{V_{\text{fit}}(i) : i \in A_t\}$ is $S_N$-invariant
- Diversity distances are symmetric: $d_{\text{alg}}(i, j) = d_{\text{alg}}(j, i)$

In the Fractal Set, this fundamental gauge redundancy manifests as **braid holonomy**. When walkers exchange positions during dynamics, their worldlines form braids in spacetime. Closed loops $\gamma$ in configuration space have holonomy $\text{Hol}(\gamma) \in S_N$, measuring the net permutation acquired.

**Discrete holonomy**: For a closed path in CST+IG, the holonomy is computed via the braid homomorphism $\rho: B_N \to S_N$. This **discrete gauge structure** is fundamental - all other symmetries (U(1), SU(2)) are emergent in the continuum limit. Unlike continuous Wilson loops (which are matrix traces), the $S_N$ holonomy is a discrete permutation, making this a hybrid discrete-continuous gauge theory.

**Global U(1) proof:** The diversity companion selection follows a Gibbs distribution. From the Fractal Set specification, the probability is:

$$
P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2}\right)}{\sum_{k' \in A_t \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}(i,k')^2}{2\epsilon_d^2}\right)}

$$

The algorithmic distance is defined as:

$$
d_{\text{alg}}(i, k)^2 = \|x_i - x_k\|^2 + \lambda_v \|v_i - v_k\|^2

$$

where $\lambda_v$ is a constant weighting parameter. **Crucially, $d_{\text{alg}}$ depends only on positions and velocities, NOT on fitness values $\Phi(x_i), \Phi(x_k)$.**

Under a global fitness shift $\Phi(x) \mapsto \Phi(x) + c$ for constant $c \in \mathbb{R}$:

$$
\begin{aligned}
d_{\text{alg}}(i,k)^2 &= \|x_i - x_k\|^2 + \lambda_v \|v_i - v_k\|^2 \\
&\quad \text{(unchanged, as positions and velocities are independent of fitness labeling)} \\
\Rightarrow P_{\text{comp}}^{(\text{div})}(k|i) &= \frac{\exp(-d_{\text{alg}}^2/(2\epsilon_d^2))}{\sum_{k'} \exp(-d_{\text{alg}}(i,k')^2/(2\epsilon_d^2))} \\
&\quad \text{(invariant)}
\end{aligned}

$$

The U(1) phase $\theta_{ik}^{(\text{U(1)})} = -d_{\text{alg}}(i,k)^2/(2\epsilon_d^2 \hbar_{\text{eff}})$ is likewise invariant. **Global** phase transformations $\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}$ (same $\alpha$ for all $i, k$) leave all observables invariant:

$$
\left|\psi_{ik}^{(\text{div})}\right|^2 = P_{\text{comp}}^{(\text{div})}(k|i) \quad \text{(global U(1) invariant)}

$$

This is **not a gauge symmetry** - there is no dynamical gauge field. The phases $\theta_{ik}$ are fixed by algorithmic distances. It's a **global symmetry** like baryon number conservation, giving a conserved Noether current $J_{\text{fitness}}^\mu$.

**SU(2) proof:** The cloning amplitude $\Psi(i \to j)$ factorizes ({prf:ref}`def-cloning-amplitude-factorization`) as:

$$
\Psi(i \to j) = A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)

$$

where $A_{ij}^{\text{SU(2)}} = \sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)} e^{i\theta_{ij}^{(\text{SU(2)})}}$ is the bare SU(2) vertex and $K_{\text{eff}}$ is the U(1)-dressed effective kernel.

An SU(2) transformation $U \in \text{SU}(2)$ acts on the isospin doublet $(|\psi_i\rangle, |\psi_j\rangle)^T$ as a rotation, mixing cloner and target roles. In the N-particle system, the amplitude ratio:

$$
\frac{\Psi(i \to j)}{\Psi(j \to i)} = \frac{A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)}{A_{ji}^{\text{SU(2)}} \cdot K_{\text{eff}}(j, i)}

$$

The SU(2) vertex satisfies $A_{ij}^{\text{SU(2)}} \propto \exp(i\theta_{ij}^{(\text{SU(2)})})$ where the phase depends on algorithmic distance, which is **symmetric** under $(i,j)$ exchange. However, the probability amplitude $\sqrt{P_{\text{comp}}^{(\text{clone})}(j|i)}$ contains **fitness asymmetry** through the cloning score $S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$.

Therefore:

$$
\frac{\Psi(i \to j)}{\Psi(j \to i)} = \exp\left(\frac{i}{\hbar_{\text{eff}}} [S(i,j,k,m) - S(j,i,m,k)] + O(\Phi_j - \Phi_i)\right)

$$

In the limit $\Phi_j - \Phi_i \to 0$, the fitness differences vanish and the ratio approaches unity, establishing **exact SU(2) symmetry on the fitness-degenerate submanifold**.

**Part 4 (Conserved and derived quantities):**

**Energy conservation:** From the BAOAB O-step ({prf:ref}`def-baoab-kernel`), the velocity update includes stochastic forcing:

$$
v^{(2)} = e^{-\gamma \Delta t} v^{(1)} + \sqrt{\frac{1}{\gamma}(1 - e^{-2\gamma \Delta t})} \, \Sigma_{\text{reg}} \xi

$$

where $\xi \sim \mathcal{N}(0, I_d)$. The kinetic energy evolves as:

$$
\begin{aligned}
E_{\text{kin}}(v_{k+1}) &= \frac{1}{2}\|v_{k+1}\|^2 \\
&= \frac{1}{2}\|v_k + \Delta t \, \mathbf{F}_{\text{total}} + \sqrt{\Delta t} \, \Sigma_{\text{reg}} \tilde{\xi}\|^2 \\
&\quad \text{(where } \tilde{\xi} \text{ is the rescaled noise)} \\
&= \frac{1}{2}\|v_k\|^2 + \Delta t \, v_k \cdot \mathbf{F}_{\text{total}} + \sqrt{\Delta t} \, v_k \cdot \Sigma_{\text{reg}} \tilde{\xi} \\
&\quad + \frac{\Delta t^2}{2}\|\mathbf{F}_{\text{total}}\|^2 + \frac{\Delta t}{2}\|\Sigma_{\text{reg}} \tilde{\xi}\|^2 + O(\Delta t^{3/2})
\end{aligned}

$$

Taking expectations over the noise $\mathbb{E}[\tilde{\xi}] = 0, \mathbb{E}[\|\tilde{\xi}\|^2] = d$:

$$
\begin{aligned}
\mathbb{E}[E_{\text{kin}}(v_{k+1}) | v_k] &= E_{\text{kin}}(v_k) + \Delta t \, v_k \cdot \mathbf{F}_{\text{total}} \\
&\quad + \frac{\Delta t}{2} \text{Tr}(\Sigma_{\text{reg}} \Sigma_{\text{reg}}^T) + O(\Delta t^2)
\end{aligned}

$$

The $O(\Delta t)$ terms are:
- $v_k \cdot \mathbf{F}_{\text{total}}$: **Power from forces** (work done)
- $\frac{1}{2}\text{Tr}(\Sigma_{\text{reg}}^2)$: **Stochastic heating**

The $O(\Delta t^2)$ terms include friction dissipation $-\gamma \Delta t \|v_k\|^2$ and higher-order force-noise coupling. For the **total energy** $H_{\text{total}} = \sum_i [E_{\text{kin},i} + U(x_i)]$:

$$
\mathbb{E}[H_{\text{total}}(Z_{k+1}) | Z_k] = H_{\text{total}}(Z_k) + \Delta t \, (\text{heating} - \text{dissipation}) + O(\Delta t^2)

$$

where the $O(\Delta t)$ term represents the **balance between stochastic forcing and friction**, not exact conservation.

The term $v_k \cdot \mathbf{F}_{\text{total}}$ equals:

$$
v_k \cdot \mathbf{F}_{\text{total}} = \text{spinor-to-vector}[\psi_v] \cdot \text{spinor-to-vector}[\psi_{\mathbf{F}_{\text{total}}}]

$$

using the spinor representation. The node scalars $E_{\text{kin}}(n_{i,t+1})$ and $E_{\text{kin}}(n_{i,t})$ store the energies, while the CST edge spinors $\psi_v, \psi_{\mathbf{F}_{\text{total}}}, \psi_{\Sigma_{\text{reg}}}$ enable computing the energy flow components.

**Derived statistical observables:** The localized mean $\mu_\rho[f_k, \Phi, x_i]$ is computed from the empirical measure $f_k(t)$:

$$
f_k(t) = \frac{1}{k} \sum_{j \in A_k(t)} \delta_{(x_j(t), v_j(t))}

$$

where $A_k(t) = \{j : s(n_{j,t}) = 1\}$ is determined from node status flags. The localization kernel $K_\rho(x_i - x_j)$ stored in IG edges provides the weighting:

$$
\mu_\rho[f_k, \Phi, x_i] = \sum_{j \in A_k} w_{ij}(\rho) \Phi_j, \quad w_{ij}(\rho) = \frac{K_\rho(x_i - x_j)}{\sum_{m \in A_k} K_\rho(x_i - x_m)}

$$

The node scalar $\mu_\rho(n_{i,t})$ equals the right-hand side. This proves that localized statistics are **derived observables** (functionals of the N-particle configuration), not independent quantities or conservation laws. The correspondence establishes that node-stored moments accurately represent the algorithmic state's local statistical structure. ∎

**Related Results:** `thm-su2-interaction-symmetry`, `def-node-spacetime`, `thm-sn-su2-lattice-qft`, `thm-sn-braid-holonomy`, `def-cloning-amplitude-factorization`, `def-baoab-kernel`, `thm-fractal-set-reconstruction`, `def-swarm-state-vector`, `thm-u1-fitness-global`

---

### Complete Convergence Fidelity

**Type:** Theorem
**Label:** `thm-complete-fidelity`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 7.1. What We Have Proven](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`, `metric-tensor`, `qsd`

**Statement:**
The Fractal Set $\mathcal{F} = (\mathcal{N}, E_{\text{CST}} \cup E_{\text{IG}})$ is the path history of a **geometrically ergodic discrete-time Markov chain**. Specifically:

**1. Information:** {prf:ref}`thm-fractal-set-reconstruction` proves the Fractal Set contains complete SDE data (reconstruction).

**2. Dynamics:** {prf:ref}`thm-fractal-set-ergodicity` proves the generator is geometrically ergodic (convergence).

**3. Fidelity:** {prf:ref}`thm-weak-convergence-invariant` proves the discrete invariant measure approximates the continuous QSD.

**Combined:** The Fractal Set is a **faithful discrete representation** of the Adaptive Gas SDE, inheriting all convergence guarantees.

**Related Results:** `thm-weak-convergence-invariant`, `thm-fractal-set-reconstruction`, `thm-fractal-set-ergodicity`

---

### Corollarys (2)

### Convergence Inheritance

**Type:** Corollary
**Label:** `cor-convergence-inheritance`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 4.2. Main Ergodicity Theorem](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `convergence`, `episodes`, `fractal-set`, `metric-tensor`, `qsd`

**Statement:**
All convergence guarantees from the continuous Adaptive Gas SDE are **inherited** by the discrete Fractal Set generator:

1. **Geometric ergodicity** ({prf:ref}`thm-main-convergence`) → {prf:ref}`thm-fractal-set-ergodicity`
2. **Foster-Lyapunov stability** ({prf:ref}`thm-foster-lyapunov-main`) → {prf:ref}`thm-discrete-drift-baoab`
3. **Exponential convergence rate** $\kappa_{\text{total}}$ → $\kappa_{\text{total}}/2$ (discrete, for small $\Delta t$)
4. **Keystone targeting** ({prf:ref}`lem-quantitative-keystone`) → Built into $\mathbf{F}_{\text{adapt}}$
5. **Uniform ellipticity** ({prf:ref}`thm-ueph`) → Preserved in O-step

**Practical implication:** The empirical distribution of nodes in the Fractal Set converges exponentially fast to a distribution $\pi_{\Delta t}$ that is close to the continuous QSD $\pi$.

**Related Results:** `thm-ueph`, `thm-main-convergence`, `lem-quantitative-keystone`, `thm-foster-lyapunov-main`, `thm-fractal-set-ergodicity`, `thm-discrete-drift-baoab`

---

### Fractal Set as Complete Algorithmic Representation

**Type:** Corollary
**Label:** `cor-fractal-set-complete-representation`
**Source:** [13_fractal_set_new/02_computational_equivalence.md § 6.3. Transfer of Symmetries and Conserved Quantities](13_fractal_set_new/02_computational_equivalence.md)
**Tags:** `computational-equivalence`, `conservation`, `convergence`, `episodes`, `fermionic`, `fractal-set`, `gauge-theory`, `lattice`, `spinor`, `symmetry`

**Statement:**
The Fractal Set $\mathcal{F}$ is **informationally and dynamically complete** for the N-particle Adaptive Gas algorithm:

**1. Informational completeness**: Every scalar, spinor, and graph structure element in $\mathcal{F}$ corresponds to a unique quantity or operation in the N-particle system (Reconstruction Theorem {prf:ref}`thm-fractal-set-reconstruction`).

**2. Dynamical completeness**: Every symmetry and conservation law in the N-particle system has a unique representation in $\mathcal{F}$ ({prf:ref}`thm-fractal-set-n-particle-equivalence`).

**3. Convergence equivalence**: The discrete-time Markov chain generating $\mathcal{F}$ converges to the same long-term distribution as the continuous SDE, up to $O(\Delta t)$ discretization error ({prf:ref}`thm-fractal-set-ergodicity`, {prf:ref}`thm-weak-convergence-invariant`).

**Practical implication**: Analyzing the Fractal Set graph structure is **equivalent** to analyzing the N-particle algorithm. Properties proven for $\mathcal{F}$ (gauge symmetries, lattice QFT structure, fermionic propagators) are **properties of the algorithm**, not artifacts of the representation.

**Related Results:** `thm-weak-convergence-invariant`, `thm-fractal-set-n-particle-equivalence`, `thm-fractal-set-reconstruction`, `thm-fractal-set-ergodicity`

---


## 03_yang_mills_noether.md

**Objects in this document:** 42

### Definitions (16)

### Hybrid Gauge Structure

**Type:** Definition
**Label:** `def-hybrid-gauge-structure`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 1.1. Three-Tier Gauge Hierarchy](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The Fractal Set gauge group is the product:

$$
G_{\text{total}} = S_N \times_{\text{semi}} (\text{SU}(2)_{\text{weak}} \times U(1)_{\text{fitness}})
$$

where:

**1. S_N Permutation Gauge Symmetry** (fundamental, discrete):
- **Origin**: Walker labels $\{1, \ldots, N\}$ are arbitrary bookkeeping indices
- **Gauge transformation**: $\sigma \cdot \mathcal{S} = (w_{\sigma(1)}, \ldots, w_{\sigma(N)})$ for $\sigma \in S_N$
- **Connection**: Braid holonomy $\text{Hol}(\gamma) = \rho([\gamma]) \in S_N$ for closed loops $\gamma$
- **Wilson loops**: Gauge-invariant observables from braid topology

**2. SU(2)_weak Local Gauge Symmetry** (emergent, continuous):
- **Origin**: Cloning interaction creates weak isospin doublet structure
- **Hilbert space**: $\mathcal{H}_{\text{int}}(i,j) = \mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}} = \mathbb{C}^2 \otimes \mathbb{C}^{N-1}$
- **Gauge transformation**: Acts on isospin factor only
- **Physical invariant**: Total interaction probability (see §1.3)

**3. U(1)_fitness Global Symmetry** (emergent, continuous):
- **Origin**: Absolute fitness scale is unphysical; only relative fitness matters
- **Global transformation**: $\psi_{ik}^{(\text{div})} \to e^{i\alpha} \psi_{ik}^{(\text{div})}$ (same $\alpha$ for all)
- **Physical invariant**: $|K_{\text{eff}}(i,j)|^2$ (cloning kernel modulus squared)
- **Conserved charge**: Fitness current $J_{\text{fitness}}^\mu$ (Noether's theorem)

**Hierarchy:**
- **S_N is fundamental**: Present from first principles (walker indistinguishability)
- **SU(2) is local but emergent**: Arises from cloning doublet structure
- **U(1) is global and emergent**: Consequence of fitness potential structure

**Semi-direct product structure:** The S_N action permutes walker indices, giving $S_N \ltimes \text{SU}(2)^N$. The U(1) factor commutes with both.

---

### Dressed Walker State and Interaction Hilbert Space

**Type:** Definition
**Label:** `def-dressed-walker-state`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 1.2. Dressed Walker States and Tensor Product Structure](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
Following {prf:ref}`thm-su2-interaction-symmetry` in {doc}`13_fractal_set/00_full_set.md`, the quantum state of the cloning interaction involves a tensor product structure.

**1. Diversity Hilbert Space:**

For walker $i$, the **diversity Hilbert space** is:

$$
\mathcal{H}_{\text{div}} = \mathbb{C}^{N-1}
$$

with orthonormal basis $\{|k\rangle : k \in A_t \setminus \{i\}\}$ labeling diversity companions.

**2. Dressed Walker State:**

Walker $i$ is "dressed" by superposition over all possible diversity companions:

$$
|\psi_i\rangle := \sum_{k \in A_t \setminus \{i\}} \psi_{ik}^{(\text{div})} |k\rangle \in \mathcal{H}_{\text{div}}
$$

where:

$$
\psi_{ik}^{(\text{div})} = \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}^{(\text{div})}}
$$

with:
- $P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp(-d_{\text{alg}}^2(i,k)/(2\epsilon_d^2))}{\sum_{k'} \exp(-d_{\text{alg}}^2(i,k')/(2\epsilon_d^2))}$: diversity companion probability
- $\theta_{ik}^{(\text{div})} = -d_{\text{alg}}^2(i,k)/(2\epsilon_d^2 \hbar_{\text{eff}})$: U(1) phase from algorithmic distance

**Physical interpretation**: $|\psi_i\rangle$ encodes how walker $i$ perceives its diversity environment.

**3. Isospin Hilbert Space:**

The **weak isospin space** is:

$$
\mathcal{H}_{\text{iso}} = \mathbb{C}^2
$$

with basis:
- $|↑\rangle = (1, 0)^T$: "cloner" role
- $|↓\rangle = (0, 1)^T$: "target" role

**4. Interaction Hilbert Space (Tensor Product):**

The full interaction space for pair $(i,j)$ is:

$$
\mathcal{H}_{\text{int}}(i,j) = \mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}} = \mathbb{C}^2 \otimes \mathbb{C}^{N-1}
$$

**5. Weak Isospin Doublet State:**

The state of the interacting pair is:

$$
|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}
$$

**Explicit form:**

$$
|\Psi_{ij}\rangle = \sum_{k \in A_t} \left(\psi_{ik}^{(\text{div})} |↑, k\rangle + \psi_{jk}^{(\text{div})} |↓, k\rangle\right)
$$

where $|↑, k\rangle = |↑\rangle \otimes |k\rangle$.

**Critical distinction**:
- $|\Psi_{ij}\rangle$: Full interaction state in $\mathbb{C}^2 \otimes \mathbb{C}^{N-1}$ (dimension $2(N-1)$)
- $|\psi_i\rangle$: Dressed walker state in $\mathbb{C}^{N-1}$ (dimension $N-1$)
- The SU(2) symmetry acts only on the first factor (isospin $\mathbb{C}^2$)

**Related Results:** `thm-su2-interaction-symmetry`

---

### SU(2) Transformation on Interaction State

**Type:** Definition
**Label:** `def-su2-transformation`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 1.2. Dressed Walker States and Tensor Product Structure](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
An SU(2) gauge transformation acts on the isospin factor only:

$$
|\Psi_{ij}\rangle \mapsto |\Psi'_{ij}\rangle = (U \otimes I_{\text{div}}) |\Psi_{ij}\rangle
$$

where $U \in \text{SU}(2)$ acts on $\mathcal{H}_{\text{iso}} = \mathbb{C}^2$ and $I_{\text{div}}$ is the identity on $\mathcal{H}_{\text{div}}$.

**Matrix form for $U = \begin{pmatrix} a & b \\ -b^* & a^* \end{pmatrix}$ with $|a|^2 + |b|^2 = 1$:**

$$
|↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle \to (a|↑\rangle + b|↓\rangle) \otimes |\psi_i\rangle + (-b^*|↑\rangle + a^*|↓\rangle) \otimes |\psi_j\rangle
$$

$$
= |↑\rangle \otimes (a|\psi_i\rangle - b^*|\psi_j\rangle) + |↓\rangle \otimes (b|\psi_i\rangle + a^*|\psi_j\rangle)
$$

This mixes the cloner and target roles through isospin rotation.

---

### Fitness Operator on Diversity Space

**Type:** Definition
**Label:** `def-fitness-operator`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 1.3. Gauge-Invariant Physical Observables](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
From {prf:ref}`def-fitness-operator` in {doc}`13_fractal_set/00_full_set.md`, the **fitness operator** for walker $i$ acts on $\mathcal{H}_{\text{div}}$ as:

$$
\hat{V}_{\text{fit},i} |k\rangle := V_{\text{fit}}(i|k) |k\rangle
$$

This is diagonal in the companion basis.

**Expectation value** for dressed walker $|\psi_i\rangle$:

$$
\langle \psi_i | \hat{V}_{\text{fit},i} | \psi_i \rangle = \sum_{k \in A_t} \left|\psi_{ik}^{(\text{div})}\right|^2 V_{\text{fit}}(i|k)
$$

**Cloning score operator** on $\mathcal{H}_{\text{int}} = \mathcal{H}_{\text{iso}} \otimes \mathcal{H}_{\text{div}}$:

Define isospin projectors $\hat{P}_{\uparrow} = |↑\rangle\langle↑|$ and $\hat{P}_{\downarrow} = |↓\rangle\langle↓|$.

The **cloning score operator** is:

$$
\hat{S}_{ij} := (\hat{P}_{\uparrow} \otimes \hat{V}_{\text{fit},i}) - (\hat{P}_{\downarrow} \otimes \hat{V}_{\text{fit},j})
$$

**Expected score:**

$$
\langle \Psi_{ij} | \hat{S}_{ij} | \Psi_{ij} \rangle = \langle \psi_i | \hat{V}_{\text{fit},i} | \psi_i \rangle - \langle \psi_j | \hat{V}_{\text{fit},j} | \psi_j \rangle
$$

**Related Results:** `def-fitness-operator`

---

### Effective Matter Lagrangian

**Type:** Definition
**Label:** `def-effective-matter-lagrangian`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 2.2. Matter Action on Interaction Hilbert Space](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `metric-tensor`, `noether`, `spinor`, `symmetry`, `yang-mills`

**Statement:**
For each cloning interaction pair $(i,j)$ with interaction state $|\Psi_{ij}\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}$, we postulate the **matter Lagrangian**:

$$
\mathcal{L}_{\text{matter}}(i,j) = \bar{\Psi}_{ij} (i\gamma^\mu \partial_\mu - m_{\text{eff}}) \Psi_{ij}
$$

where:

**1. Field content:**
- $\Psi_{ij}$: Interaction spinor field on $\mathbb{C}^2 \otimes \mathbb{C}^{N-1}$
- $\bar{\Psi}_{ij} = \Psi_{ij}^\dagger \gamma^0$: Dirac adjoint

**2. Dirac operator in $(1+d)$-dimensional spacetime:**
- $\gamma^\mu$: Dirac gamma matrices satisfying $\{\gamma^\mu, \gamma^\nu\} = 2\eta^{\mu\nu}$
- $\eta^{\mu\nu} = \text{diag}(1, -1, \ldots, -1)$: Lorentzian metric
- $\partial_\mu$: Discrete derivative on Fractal Set lattice (§2.4)

**3. Effective mass:**

$$
m_{\text{eff}} = \langle \Psi_{ij} | \hat{S}_{ij} | \Psi_{ij} \rangle
$$

where $\hat{S}_{ij}$ is the cloning score operator ({prf:ref}`def-fitness-operator`).

**Physical interpretation**:
- Kinetic term $\bar{\Psi} i\gamma^\mu \partial_\mu \Psi$: Evolution along Fractal Set edges
- Mass term $m_{\text{eff}} \bar{\Psi}\Psi$: Fitness comparison determines "mass" (stability)
- Positive $m_{\text{eff}}$: Walker $i$ is fitter (favors $i \to j$ cloning)
- Negative $m_{\text{eff}}$: Walker $j$ is fitter (favors $j \to i$ cloning)

**Related Results:** `def-fitness-operator`

---

### SU(2) Gauge Field from Algorithmic Phases

**Type:** Definition
**Label:** `def-su2-gauge-field`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 2.3. Gauge Field and Covariant Derivative](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The SU(2) gauge field is constructed from the algorithmic geometry of the Fractal Set.

**1. Connection 1-form:**

From the cloning amplitude phase $\theta_{ij}^{(\text{SU}(2))}$ in {doc}`13_fractal_set/00_full_set.md` §7.10, define the **connection 1-form**:

$$
A_e^{(a)} T^a := \frac{i}{\tau} \log U_e
$$

where $U_e$ is the parallel transport operator along edge $e$ (to be defined precisely below) and $T^a = \sigma^a/2$ are the SU(2) generators (Pauli matrices).

**2. Link variable (parallel transport):**

For edge $e = (n_j, n_i)$ in the Fractal Set, the **link variable** is:

$$
U_{ij} := \mathcal{P} \exp\left(i \int_j^i A_\mu dx^\mu\right) \in \text{SU}(2)
$$

where $\mathcal{P}$ denotes path ordering.

**Discrete approximation** on Fractal Set:

$$
U_{ij} = \exp\left(i\tau \sum_{a=1}^3 A_e^{(a)} T^a\right)
$$

where $\tau$ is the lattice coupling parameter (related to edge length).

**3. Algorithmic phase identification:**

From {doc}`13_fractal_set/00_full_set.md`, the cloning amplitude has phase:

$$
\theta_{ij}^{(\text{SU}(2))} = -\frac{d_{\text{alg}}^2(i,j)}{2\epsilon_c^2 \hbar_{\text{eff}}}
$$

We identify:

$$
\tau \sum_{a=1}^3 A_e^{(a)} T^a \approx \theta_{ij}^{(\text{SU}(2))} \cdot I_2 + O(\tau^2)
$$

For small $\tau$, the gauge field is approximately:

$$
A_e^{(3)} \approx \frac{\theta_{ij}^{(\text{SU}(2))}}{\tau}, \quad A_e^{(1)} \approx A_e^{(2)} \approx 0
$$

(dominant contribution in the $\sigma^3$ direction).

**4. Gauge transformation:**

Under local SU(2) transformation $U_i \in \text{SU}(2)$ at node $i$:

$$
U_{ij} \to U_i U_{ij} U_j^\dagger
$$

$$
A_\mu \to U A_\mu U^\dagger + \frac{i}{g} U \partial_\mu U^\dagger
$$

**5. Covariant derivative:**

The gauge-covariant derivative acting on $\Psi_{ij}$ is:

$$
D_\mu \Psi_{ij} = \partial_\mu \Psi_{ij} - ig \sum_{a=1}^3 A_\mu^{(a)} (T^a \otimes I_{\text{div}}) \Psi_{ij}
$$

where $T^a$ acts on the isospin $\mathbb{C}^2$ factor and $I_{\text{div}}$ is identity on $\mathbb{C}^{N-1}$.

**Gauge covariance:**

$$
D_\mu \Psi_{ij} \to (U \otimes I_{\text{div}}) D_\mu \Psi_{ij}
$$

under $\Psi_{ij} \to (U \otimes I_{\text{div}}) \Psi_{ij}$.

---

### Discrete Derivatives on Fractal Set Lattice

**Type:** Definition
**Label:** `def-discrete-derivatives`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 2.4. Discrete Spacetime Derivatives](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The Fractal Set is a discrete spacetime lattice with non-uniform structure. Derivatives must be defined carefully.

**1. Temporal derivative (along CST edges):**

For a field $\Phi(n_{i,t})$ on nodes, the temporal derivative is:

$$
\partial_0 \Phi(n_{i,t}) := \frac{\Phi(n_{i,t+1}) - \Phi(n_{i,t})}{\Delta t}
$$

**2. Spatial derivative (emergent manifold):**

Spatial derivatives use the localization kernel $K_\rho$ to define a discrete gradient:

$$
\partial_k \Phi(n_{i,t}) := \frac{1}{\rho} \sum_{j \in A_t} w_{ij} K_\rho(y_i, y_j) \cdot (\Phi(n_{j,t}) - \Phi(n_{i,t})) \cdot \hat{e}_k(y_i \to y_j)
$$

where:
- $K_\rho(y_i, y_j) = \exp(-d_{\text{alg}}^2(y_i, y_j)/(2\rho^2))$: localization kernel
- $\hat{e}_k(y_i \to y_j)$: unit vector from $y_i$ to $y_j$ projected onto axis $k$
- $w_{ij}$: normalization weights ensuring $\sum_j w_{ij} = 1$

**3. Discrete divergence:**

For a 4-current $J^\mu$:

$$
\nabla_{\text{discrete}} \cdot J := \partial_0 J^0 + \sum_{k=1}^d \partial_k J^k
$$

**4. Discrete d'Alembertian:**

$$
\Box_{\text{discrete}} = \partial_0^2 - \sum_{k=1}^d \partial_k^2
$$

**Continuum limit**: As $\Delta t, \rho \to 0$, these discrete operators converge to standard continuum derivatives $\partial_\mu$.

---

### SU(2) Link Variables on Fractal Set

**Type:** Definition
**Label:** `def-su2-link-variables`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 4.1. SU(2) Link Variables and Parallel Transport](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
For each edge $e = (n_j, n_i)$ in the Fractal Set (CST or IG edge), define the **SU(2) link variable**:

$$
U_{ij}(e) = \exp\left(i\tau \sum_{a=1}^3 A_e^{(a)} T^a\right) \in \text{SU}(2)
$$

where:
- $A_e^{(a)}$: Gauge field components along edge $e$ (three real numbers)
- $T^a = \sigma^a/2$: Pauli matrices (SU(2) generators)
- $\tau$: Lattice coupling parameter

**Physical interpretation**: $U_{ij}$ is the parallel transport operator from node $j$ to node $i$.

**Gauge transformation:**

$$
U_{ij} \to U_i U_{ij} U_j^\dagger
$$

**Connection to cloning amplitude:**

From {doc}`13_fractal_set/00_full_set.md` §7.10, the cloning amplitude factorizes as:

$$
\Psi(i \to j) = A_{ij}^{\text{SU}(2)} \cdot K_{\text{eff}}(i,j)
$$

where $A_{ij}^{\text{SU}(2)}$ contains the SU(2) structure. We relate:

$$
A_{ij}^{\text{SU}(2)} = \langle ↓ | \otimes \langle \psi_j | U_{ij} | ↑ \rangle \otimes | \psi_i \rangle
$$

This makes the cloning amplitude manifestly gauge-covariant.

---

### Discrete Field Strength (Plaquette Curvature)

**Type:** Definition
**Label:** `def-discrete-field-strength`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 4.2. Field Strength and Plaquettes](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `curvature`, `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
For each **plaquette** $\square$ (elementary closed loop with 4 edges) in the Fractal Set, define the **plaquette variable**:

$$
U_{\square} = U_{12} U_{23} U_{34} U_{41}
$$

where $1,2,3,4$ are the four nodes in cyclic order and $U_{ij}$ are the link variables.

The **field strength** is:

$$
F_{\square} = \frac{1}{i\tau^2} \log U_{\square} \in \mathfrak{su}(2)
$$

**Expansion in generators:**

$$
F_{\square} = \sum_{a=1}^3 F_{\square}^{(a)} T^a
$$

where:

$$
F_{\square}^{(a)} = 2 \text{Tr}(T^a F_{\square}) = \text{Tr}(\sigma^a F_{\square})
$$

**Continuum limit**: As $\tau \to 0$:

$$
F_{\square}^{(a)} \to F_{\mu\nu}^{(a)} = \partial_\mu A_\nu^{(a)} - \partial_\nu A_\mu^{(a)} + g \epsilon^{abc} A_\mu^{(b)} A_\nu^{(c)}
$$

(Yang-Mills field strength).

**Gauge invariance:**

$$
U_{\square} \to V_1 U_{\square} V_1^\dagger \quad \Rightarrow \quad \text{Tr}(U_{\square}) = \text{Tr}(V_1 U_{\square} V_1^\dagger) = \text{Tr}(U_{\square})
$$

The trace is gauge-invariant.

---

### Fractal Set Plaquette Types

**Type:** Definition
**Label:** `def-fractal-set-plaquettes`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 4.2. Field Strength and Plaquettes](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The Fractal Set has three types of plaquettes:

**1. Temporal plaquettes** (CST × CST):

$$
\square_{\text{temp}}(i,j,t) = (n_{i,t}, n_{i,t+1}, n_{j,t+1}, n_{j,t})
$$

- **Edges**: Two CST edges (time evolution of $i$ and $j$) plus two spatial connections
- **Count**: $\binom{N}{2} \times T$

**2. Spatial plaquettes** (IG × CST):

$$
\square_{\text{space}}(i \to j, t) = (n_{i,t}, n_{j,t}, n_{j,t+1}, n_{i,t+1})
$$

- **Edges**:
  - $(n_{i,t}, n_{j,t})$: IG edge (cloning interaction)
  - $(n_{j,t}, n_{j,t+1})$: CST edge (time evolution of $j$)
  - $(n_{j,t+1}, n_{i,t+1})$: IG edge at $t+1$ or spatial connection
  - $(n_{i,t+1}, n_{i,t})$: CST edge backwards (time evolution of $i$, reversed)
- **Count**: $O(N^2 T)$

**3. Mixed plaquettes** (IG × IG):

$$
\square_{\text{mixed}}(i,j,k,t) = (n_{i,t}, n_{j,t}, n_{k,t}, n_{i,t})
$$

- **Edges**: Three IG edges forming a triangle
- **Count**: $O(N^3)$ per timestep (rare, requires dense interaction graph)

**Total plaquette count**: $O(N^2 T)$

---

### Discrete Yang-Mills Action

**Type:** Definition
**Label:** `def-discrete-ym-action`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 4.3. Wilson Plaquette Action](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `curvature`, `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The **discrete Yang-Mills action** on the Fractal Set is:

$$
S_{\text{YM}} = \frac{1}{g^2} \sum_{\square} S_{\square}
$$

where:

$$
S_{\square} = 2\left(1 - \frac{1}{2}\text{Tr}(U_{\square})\right)
$$

is the **plaquette action** (Wilson action).

**Expansion for small fields**: For $||A_e|| \ll 1$:

$$
U_{\square} \approx I + i\tau^2 \sum_a F_{\square}^{(a)} T^a - \frac{\tau^4}{2} \sum_a (F_{\square}^{(a)})^2 T^a T^a + O(\tau^6)
$$

Taking the trace:

$$
\text{Tr}(U_{\square}) = 2 - \frac{\tau^4}{4}\sum_{a=1}^3 (F_{\square}^{(a)})^2 + O(\tau^6)
$$

Therefore:

$$
S_{\square} = \frac{\tau^4}{4}\sum_{a=1}^3 (F_{\square}^{(a)})^2 + O(\tau^6)
$$

**Continuum limit**: As $\tau \to 0$:

$$
S_{\text{YM}} \to \frac{1}{4g^2} \int d^{d+1}x \sum_{a=1}^3 F_{\mu\nu}^{(a)} F^{(a),\mu\nu}
$$

(standard Yang-Mills action).

**Physical interpretation**:
- $\text{Tr}(U_{\square}) = 2$: Flat connection (no curvature)
- $\text{Tr}(U_{\square}) < 2$: Non-zero field strength (gauge field present)
- $g$: Coupling constant

---

### Gauge-Invariant Observables

**Type:** Definition
**Label:** `def-physical-observables`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 6. Physical Observables and Wilson Loops](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `curvature`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
**1. Wilson loops** (holonomy around closed paths):

For closed loop $\gamma = (e_1, e_2, \ldots, e_L, e_1)$:

$$
W_\gamma = \text{Tr}\left(\prod_{k=1}^L U_{e_k}\right) \in \mathbb{R}
$$

**Properties**:
- Gauge-invariant
- $|W_\gamma| \le 2$ (dimension of SU(2))
- Measures enclosed gauge field curvature

**2. Total interaction probability** (SU(2)-invariant):

From {prf:ref}`prop-su2-invariance`:

$$
P_{\text{total}}(i,j) = P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)
$$

This is the **correct** SU(2)-invariant observable, not individual directional probabilities.

**3. Plaquette expectation values**:

$$
\langle W_{\square} \rangle = \frac{1}{Z} \int \mathcal{D}[A] \, \text{Tr}(U_{\square}) \, e^{-S_{\text{YM}}}
$$

**Interpretation**:
- $\langle W_{\square} \rangle \approx 2$: Weak coupling (small field)
- $\langle W_{\square} \rangle \ll 2$: Strong coupling (large field)

**4. Noether charge expectation values**:

$$
\langle Q_{\text{fitness}} \rangle = \frac{1}{Z} \int \mathcal{D}[\Psi] \mathcal{D}[A] \, Q_{\text{fitness}} \, e^{iS}
$$

where $Q_{\text{fitness}} = \sum_{i \in A_t} \Phi(x_i)$.

**Related Results:** `prop-su2-invariance`

---

### Discrete Hamiltonian on Fractal Set

**Type:** Definition
**Label:** `def-discrete-hamiltonian-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.5. Hamiltonian Formulation in Algorithmic Parameters](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `curvature`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The Hamiltonian formulation separates time from space, expressing dynamics as evolution under a Hamiltonian operator.

**1. Phase Space Coordinates:**

- **Positions** (nodes): $\{x_i(t), v_i(t)\}_{i \in A_t}$
- **Gauge fields** (edges): $\{A_k^{(a)}(e)\}$ (spatial components, $k=1,\ldots,d$)
- **Electric fields** (edges): $\{E_k^{(a)}(e)\}$ (conjugate momenta)

**2. Canonical Hamiltonian:**

$$
H = H_{\text{matter}} + H_{\text{gauge}} + H_{\text{interaction}}
$$

**3. Matter Hamiltonian (Algorithmic):**

$$
H_{\text{matter}} = \sum_{i \in A_t} \left[\frac{1}{2}m v_i^2 + U(x_i) - \beta r(x_i) + V_{\text{adaptive}}(x_i, S_t) + V_{\text{viscous}}(x_i, S_t)\right]
$$

where:

- **Kinetic energy**: $\frac{1}{2}m v_i^2$
- **Confining potential**: $U(x_i)$ (Valid Domain)
- **Reward**: $-\beta r(x_i)$ (fitness landscape)
- **Adaptive potential**:

$$
V_{\text{adaptive}} = -\epsilon_F \sum_{i,j} K_\rho(y_i, y_j) V_{\text{fit}}(i|j)
$$

- **Viscous potential**:

$$
V_{\text{viscous}} = -\frac{\nu}{2} \sum_{i,j} K_\rho(y_i, y_j) \|v_i - v_j\|^2
$$

**Algorithmic parameters**: $(m, \beta, \epsilon_F, \nu, \rho, \lambda_v)$

**4. Gauge Field Hamiltonian:**

$$
H_{\text{gauge}} = \frac{g^2}{2} \sum_{e \in \text{Edges}} (E_e^{(a)})^2 + \frac{1}{g^2} \sum_{\square \in \text{Plaquettes}} \left(1 - \frac{1}{2}\text{Tr}(U_{\square})\right)
$$

**Electric field term**: Kinetic energy of gauge field

**Magnetic field term**: Potential energy from curvature

**In algorithmic parameters**:

$$
E_e^{(a)} \sim \frac{\partial A_e^{(a)}}{\partial t} \sim \frac{1}{\Delta t} \frac{d_{\text{alg}}^2}{\epsilon_c^2 \hbar_{\text{eff}}}
$$

$$
U_{\square} - I \sim \left(\frac{d_{\text{alg}}^2}{\epsilon_c^2 \hbar_{\text{eff}}}\right)^2
$$

**5. Interaction Hamiltonian:**

$$
H_{\text{interaction}} = g \sum_{e \in E_{\text{IG}}} J_k^{(a)}(e) A_k^{(a)}(e)
$$

where $J_k^{(a)}$ is the spatial component of the SU(2) current.

**In algorithmic parameters**:

$$
J_k^{(a)} \sim P_{\text{comp}}^{(\text{div})}(\epsilon_d) \cdot v_k \sim \exp\left(-\frac{d_{\text{alg}}^2}{2\epsilon_d^2}\right) v_k
$$

**6. Hamilton's Equations:**

**For matter fields**:

$$
\frac{dx_i}{dt} = \frac{\partial H}{\partial (m v_i)} = v_i
$$

$$
\frac{dv_i}{dt} = -\frac{\partial H}{\partial x_i} = -\gamma v_i - \nabla U + \epsilon_F F_{\text{adaptive}} + \nu F_{\text{viscous}} + \sqrt{2D} \xi_i
$$

(where friction $-\gamma v_i$ and diffusion $\sqrt{2D}\xi_i$ are added phenomenologically).

**For gauge fields**:

$$
\frac{\partial A_k^{(a)}}{\partial t} = \frac{\delta H}{\delta E_k^{(a)}} = g^2 E_k^{(a)}
$$

$$
\frac{\partial E_k^{(a)}}{\partial t} = -\frac{\delta H}{\delta A_k^{(a)}} = \sum_{\square \ni e} F_{\square}^{(a)} - g J_k^{(a)}
$$

**7. Total Energy Evolution:**

$$
\frac{dH}{dt} = -\gamma \sum_i v_i^2 + \sqrt{2D} \sum_i \xi_i \cdot \frac{\partial H}{\partial v_i} + \Delta H_{\text{cloning}}
$$

**Dissipation**: $-\gamma \sum v_i^2$ (parameter $\gamma$)

**Stochastic injection**: $\sqrt{2D}$ term (parameter $D$)

**Cloning events**: $\Delta H_{\text{cloning}}$

**Energy conservation**: $dH/dt = 0$ when $\gamma, D \to 0$ and no cloning.

---

### Dimensions of Algorithmic Parameters

**Type:** Definition
**Label:** `def-parameter-dimensions`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.1. Dimensional Analysis of Algorithmic Parameters](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
We establish the physical dimensions of all algorithmic parameters from their definitions in the Adaptive Gas SDEs:

**Primary dimensional parameters:**

1. **Position**: $x_i \in \mathcal{X} \subset \mathbb{R}^d$ has dimension $[L]$ (Length)

2. **Velocity**: $v_i$ has dimension $[V] = [L][T]^{-1}$

3. **Timestep**: $\tau$ (cloning period) has dimension $[T]$ (Time)

4. **Fitness potential**: $V_{\text{fit}}(x)$ is dimensionless (normalized scores)

5. **External potential**: $U(x)$ has dimension $[E] := [M][L]^2[T]^{-2}$ (Energy)

6. **Walker mass**: From kinetic energy $K = \frac{1}{2}m v^2$, we have $[m] = [M]$ (Mass)

**Derived dimensional parameters:**

7. **Friction coefficient**: $\gamma$ from $dv = -\gamma v dt$ has dimension $[\gamma] = [T]^{-1}$

8. **Diffusion coefficient**: $D$ from $\sqrt{2D} dW$ has dimension $[D] = [L]^2[T]^{-1}$

9. **Localization scale**: $\rho$ (Gaussian kernel width) has dimension $[\rho] = [L]$

10. **Cloning selection scale**: $\epsilon_c$ (kernel width in $d_{\text{alg}}$) has dimension $[\epsilon_c] = [L]$ (since $d_{\text{alg}}$ has units of length for $\lambda_v = 1$)

11. **Velocity weight**: $\lambda_v$ in $d_{\text{alg}}^2 = \|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2$ has dimension $[\lambda_v] = [T]^2$ to make $d_{\text{alg}}$ have units $[L]$

12. **Viscosity coefficient**: $\nu$ from viscous force $\nu \sum_j K_\rho (v_j - v_i)$ has dimension $[\nu] = [T]^{-1}$

13. **Adaptive force strength**: $\epsilon_F$ from $F = \epsilon_F \sum_j K_\rho \nabla V_{\text{fit}}$ where $\nabla V_{\text{fit}}$ has dimension $[L]^{-1}$ (gradient of dimensionless potential), and $K_\rho$ is dimensionless (normalized kernel), gives $[\epsilon_F] = [M][L]^2[T]^{-2}$ to make $F$ have units $[M][L][T]^{-2}$

**Dimensionless parameters:**

14. **Number of walkers**: $N$ (dimensionless)

15. **State space dimension**: $d$ (dimensionless)

16. **Exploitation weights**: $\alpha, \beta$ (dimensionless)

17. **Cloning temperature**: $T_{\text{clone}}$ (dimensionless, normalizes fitness scores)

18. **Diversity noise scale**: $\epsilon_d$ normalized such that amplitudes are dimensionless

---

### Fine Structure Constant for Fractal Set

**Type:** Definition
**Label:** `def-fine-structure-constant`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.8. Fine Structure Constant and Dimensionless Ratios](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The dimensionless fine structure constant is:

$$
\alpha_{\text{FS}} = g_{\text{weak}}^2 = \frac{\tau \rho^2}{m \epsilon_c^2}
$$

**Dimensional check:** Using natural units where $[m] = [M] = [T]^{-1}$ and $[\rho] = [L] = [T]$:

$$
[\alpha_{\text{FS}}] = \frac{[T] [T]^2}{[T]^{-1} [T]^2} = \frac{[T]^3}{[T]} = [T]^2
$$

Wait, this is still not dimensionless. In natural units with $\hbar = c = 1$, we have $[m] = [E] = [L]^{-1}$. Then:

$$
[\alpha_{\text{FS}}] = \frac{[T] [L]^2}{[L]^{-1} [L]^2} = [T] [L] = [1] \quad \checkmark
$$

when we set $c = 1$ so $[T] = [L]$.

---

### Rigorous Dictionary: Physical Constants ↔ Algorithmic Parameters

**Type:** Definition
**Label:** `def-constant-dictionary-corrected`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.9. Summary: Complete Dictionary of Fundamental Constants](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
| Physical Constant | Algorithmic Expression (Corrected) | Dimensions | Reference |
|-------------------|-----------------------------------|------------|-----------|
| $\hbar_{\text{eff}}$ | $m\epsilon_c^2/\tau$ | $[M][L]^2[T]^{-1}$ | {prf:ref}`thm-effective-planck-constant` |
| $g_{\text{weak}}^2$ | $\tau\rho^2/(m\epsilon_c^2)$ | Dimensionless | {prf:ref}`thm-su2-coupling-constant` |
| $e_{\text{fitness}}^2$ | $m^2/\epsilon_F$ | Dimensionless | {prf:ref}`thm-u1-coupling-constant` |
| $m_{\text{clone}}$ | $1/\epsilon_c$ | $[M]$ | {prf:ref}`thm-mass-scales` |
| $m_{\text{MF}}$ | $1/\rho$ | $[M]$ | {prf:ref}`thm-mass-scales` |
| $m_{\text{gap}}$ | $\sqrt{\lambda_{\text{gap}}/\tau}$ | $[M]$ | {prf:ref}`thm-mass-scales` |
| $m_{\text{friction}}$ | $\gamma$ | $[M]$ | {prf:ref}`thm-mass-scales` |
| $\xi$ (correlation length) | $\epsilon_c/\sqrt{\tau\lambda_{\text{gap}}}$ | $[L]$ | {prf:ref}`thm-correlation-length` |
| $\alpha_{\text{FS}}$ | $\tau\rho^2/(m\epsilon_c^2)$ | Dimensionless | {prf:ref}`def-fine-structure-constant` |

**Key dimensional parameters (from {prf:ref}`def-parameter-dimensions`):**
- $m$: Walker mass, dimension $[M]$
- $\epsilon_c$: Cloning selection scale, dimension $[L]$
- $\rho$: Localization scale, dimension $[L]$
- $\tau$: Cloning timestep, dimension $[T]$
- $\gamma$: Friction coefficient, dimension $[T]^{-1} = [M]$ (natural units)
- $\epsilon_F$: Adaptive force strength, dimension $[M][L]^2[T]^{-2}$
- $\lambda_v$: Velocity weight, dimension $[T]^2$

**Dimensionless parameters:**
- $N$: Number of walkers
- $d$: State space dimension
- $\alpha, \beta$: Exploitation weights
- $T_{\text{clone}}$: Cloning temperature (normalizes dimensionless fitness)

**Dimensionless ratios characterizing algorithm regime ({prf:ref}`thm-dimensionless-ratios`):**
- $\sigma_{\text{sep}} = \epsilon_c/\rho$: Scale separation
- $\eta_{\text{time}} = \tau\lambda_{\text{gap}}$: Timescale ratio
- $\kappa = \xi/\rho$: Correlation-to-interaction ratio

**Mass hierarchy for efficient operation:**

$$
m_{\text{friction}} \ll m_{\text{gap}} < m_{\text{MF}} < m_{\text{clone}}
$$

Equivalently:

$$
\gamma \ll \sqrt{\frac{\lambda_{\text{gap}}}{\tau}} < \frac{1}{\rho} < \frac{1}{\epsilon_c}
$$

**Related Results:** `def-parameter-dimensions`, `thm-dimensionless-ratios`, `thm-effective-planck-constant`, `thm-u1-coupling-constant`, `thm-su2-coupling-constant`, `thm-mass-scales`, `thm-correlation-length`, `def-fine-structure-constant`

---

### Axioms (1)

### Effective Lagrangian Postulate

**Type:** Axiom
**Label:** `axiom-effective-lagrangian`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 2.1. Postulates and Scope](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
We postulate an **effective Lagrangian** for the cloning interaction based on symmetry principles. This Lagrangian is not derived from first principles but is constructed to:

1. Respect the SU(2)_weak × U(1)_fitness symmetry structure
2. Reproduce the cloning probabilities in appropriate limits
3. Enable systematic Noether current derivation

**Status**: This is a **phenomenological model** (effective field theory), not a fundamental derivation. Future work should derive this from the stochastic path integral in {doc}`13_fractal_set/00_full_set.md` §7.5.

**Justification**: Effective field theories are a standard tool in physics when microscopic dynamics are complex. The validity is tested by comparing predictions with the actual algorithm behavior.

---

### Propositions (1)

### SU(2) Invariance of Total Interaction Probability

**Type:** Proposition
**Label:** `prop-su2-invariance`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 1.3. Gauge-Invariant Physical Observables](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
From {prf:ref}`prop-su2-invariance` in {doc}`13_fractal_set/00_full_set.md`, the SU(2) gauge-invariant observable is the **total interaction probability**:

$$
P_{\text{total}}(i, j) := P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)
$$

**Invariance statement:**

$$
P_{\text{total}}(i, j) = P'_{\text{total}}(i, j)
$$

where primed quantities are computed from the rotated state $|\Psi'_{ij}\rangle = (U \otimes I_{\text{div}})|\Psi_{ij}\rangle$.

**Physical interpretation**: An SU(2) rotation changes the "viewpoint" of the interaction (which walker is cloner vs target), but the **total propensity for the pair to interact** is gauge-invariant.

**Note**: Individual directional probabilities $P_{\text{clone}}(i \to j)$ are **not** gauge-invariant; only their sum is.

**Related Results:** `prop-su2-invariance`

---

### Theorems (22)

### U(1) Fitness Noether Current (Rigorous Derivation)

**Type:** Theorem
**Label:** `thm-u1-noether-current`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 3.1. U(1)_fitness Global Current](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The global U(1)_fitness symmetry implies a conserved fitness current on the Fractal Set.

**1. Current Definition:**

For each node $n_{i,t}$ (walker $i$ at time $t$), define the **fitness 4-current**:

$$
J_{\text{fitness}}^\mu(n_{i,t}) := \rho_{\text{fitness}}(n_{i,t}) \cdot u^\mu(n_{i,t})
$$

where:
- $\rho_{\text{fitness}}(n_{i,t}) := \Phi(x_i(t)) \cdot s(n_{i,t})$: fitness charge density
- $u^\mu = (\gamma^{-1}, v^k)$: 4-velocity with $\gamma = (1 - v^2)^{-1/2}$ (in units where $c=1$)
- $\Phi(x) = -U(x) + \beta r(x)$: fitness function
- $s \in \{0,1\}$: survival status

**Components:**
- Temporal: $J^0_{\text{fitness}} = \rho_{\text{fitness}} / \gamma \approx \rho_{\text{fitness}}$ (non-relativistic)
- Spatial: $J^k_{\text{fitness}} = \rho_{\text{fitness}} \cdot v^k$

**2. Discrete Continuity Equation:**

The current satisfies:

$$
\frac{J^0_{\text{fitness}}(n_{i,t+1}) - J^0_{\text{fitness}}(n_{i,t})}{\Delta t} + \sum_{k=1}^d \partial_k J^k_{\text{fitness}}(n_{i,t}) = \mathcal{S}_{\text{fitness}}(n_{i,t})
$$

where $\mathcal{S}_{\text{fitness}}$ is the source term from cloning events.

**3. Global Conservation:**

Summing over all alive walkers $i \in A_t$:

$$
\frac{d}{dt} Q_{\text{fitness}}(t) = \sum_{i \in A_t} \mathcal{S}_{\text{fitness}}(n_{i,t})
$$

where:

$$
Q_{\text{fitness}}(t) := \sum_{i \in A_t} \Phi(x_i(t))
$$

is the total fitness charge.

**Between cloning events** ($\mathcal{S}_{\text{fitness}} = 0$), the total fitness charge is conserved:

$$
Q_{\text{fitness}}(t) = \text{constant}
$$

---

### SU(2) Weak Isospin Noether Current

**Type:** Theorem
**Label:** `thm-su2-noether-current`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 3.2. SU(2)_weak Local Current](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The local SU(2)_weak gauge symmetry implies three conserved weak isospin currents, one for each generator $T^a = \sigma^a/2$ ($a = 1,2,3$).

**1. Current Definition:**

For each cloning pair $(i,j)$ with interaction state $|\Psi_{ij}\rangle \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}$, define the **weak isospin 4-current**:

$$
J_\mu^{(a)}(i,j) = \bar{\Psi}_{ij} \gamma_\mu (T^a \otimes I_{\text{div}}) \Psi_{ij}
$$

where:
- $\Psi_{ij} = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle$: interaction doublet
- $\gamma_\mu$: Dirac gamma matrices in $(1+d)$ dimensions
- $T^a = \sigma^a/2$: Pauli matrices (SU(2) generators acting on isospin space)
- $I_{\text{div}}$: identity on diversity space $\mathbb{C}^{N-1}$
- $\bar{\Psi}_{ij} = \Psi_{ij}^\dagger \gamma^0$: Dirac adjoint

**2. Explicit Component Form:**

Writing out the tensor product structure:

$$
J_\mu^{(a)}(i,j) = \langle ↑| \otimes \langle \psi_i| + \langle ↓| \otimes \langle \psi_j| \Big) \gamma_\mu (T^a \otimes I) \Big(|↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle \Big)
$$

For $a=3$ (diagonal generator $T^3 = \text{diag}(1/2, -1/2)$):

$$
J_\mu^{(3)}(i,j) = \frac{1}{2} \bar{\psi}_i \gamma_\mu \psi_i - \frac{1}{2} \bar{\psi}_j \gamma_\mu \psi_j
$$

This is the **isospin charge difference** between cloner and target.

For $a=1,2$ (off-diagonal generators):

$$
J_\mu^{(1,2)}(i,j) \propto \bar{\psi}_i \gamma_\mu \psi_j + \bar{\psi}_j \gamma_\mu \psi_i
$$

These are **mixing currents** coupling the two dressed walker states.

**3. Conservation Law:**

Assuming the effective Lagrangian ({prf:ref}`def-effective-matter-lagrangian`) governs the dynamics, the current satisfies:

$$
\partial^\mu J_\mu^{(a)}(i,j) = 0
$$

(discrete divergence vanishes on-shell, i.e., when equations of motion are satisfied).

**4. Gauge Transformation:**

Under local SU(2) transformation $U_i, U_j \in \text{SU}(2)$ at nodes $i$ and $j$:

$$
J_\mu^{(a)} \to (U_i)_{ab} J_\mu^{(b)} (U_j^\dagger)_{bc}
$$

(adjoint representation transformation).

**Physical observables** are traces:

$$
\mathcal{O}_{\text{weak}} = \sum_{a=1}^3 \text{Tr}(J_\mu^{(a)} J^{(a),\mu})
$$

which are gauge-invariant.

**Related Results:** `def-effective-matter-lagrangian`

---

### Gauge Invariance of Yang-Mills Action

**Type:** Theorem
**Label:** `thm-ym-action-gauge-invariant`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 4.3. Wilson Plaquette Action](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The discrete Yang-Mills action $S_{\text{YM}}$ is **exactly gauge-invariant** under local SU(2) transformations.

**Proof**: Under $U_i \to V_i U_i V_i^\dagger$ at each node:

$$
U_{ij} \to V_i U_{ij} V_j^\dagger
$$

For plaquette $\square = (1,2,3,4)$:

$$
U_{\square}' = (V_1 U_{12} V_2^\dagger)(V_2 U_{23} V_3^\dagger)(V_3 U_{34} V_4^\dagger)(V_4 U_{41} V_1^\dagger)
$$

$$
= V_1 U_{12} U_{23} U_{34} U_{41} V_1^\dagger = V_1 U_{\square} V_1^\dagger
$$

Therefore:

$$
\text{Tr}(U_{\square}') = \text{Tr}(V_1 U_{\square} V_1^\dagger) = \text{Tr}(U_{\square})
$$

Hence $S_{\square}' = S_{\square}$ and $S_{\text{YM}}' = S_{\text{YM}}$. ∎

---

### Complete Gauge-Covariant Path Integral

**Type:** Theorem
**Label:** `thm-gauge-covariant-path-integral`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 5. Gauge-Covariant Path Integral](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `integration`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The full effective field theory for the Adaptive Gas on the Fractal Set has partition function:

$$
Z = \int \mathcal{D}[\Psi] \mathcal{D}[A] \exp\left(i(S_{\text{matter}} + S_{\text{coupling}} + S_{\text{YM}})\right)
$$

where:

**1. Matter action**:

$$
S_{\text{matter}} = \sum_{(i,j) \in \text{IG}} \int d\tau \, \bar{\Psi}_{ij} (i\gamma^\mu D_\mu - m_{\text{eff}}) \Psi_{ij}
$$

with $D_\mu = \partial_\mu - ig A_\mu^{(a)} (T^a \otimes I_{\text{div}})$ (covariant derivative).

**2. Coupling action**:

$$
S_{\text{coupling}} = g \sum_{a=1}^3 \sum_{(i,j)} \int d\tau \, J_\mu^{(a)}(i,j) \cdot A_\mu^{(a)}
$$

where $J_\mu^{(a)}$ is the weak isospin current ({prf:ref}`thm-su2-noether-current`).

**3. Yang-Mills action**:

$$
S_{\text{YM}} = \frac{1}{g^2} \sum_{\square} 2\left(1 - \frac{1}{2}\text{Tr}(U_{\square})\right)
$$

**Integration measure**:
- $\mathcal{D}[\Psi]$: Functional integral over all interaction states $\Psi_{ij} \in \mathbb{C}^2 \otimes \mathbb{C}^{N-1}$
- $\mathcal{D}[A]$: Functional integral over all gauge field configurations $A_e^{(a)}$

**Gauge invariance**: The path integral is gauge-invariant:

$$
Z' = Z
$$

under local SU(2) transformations $\Psi_{ij} \to (U_i \otimes I) \Psi_{ij}$ and $A_\mu \to U A_\mu U^\dagger + (i/g) U \partial_\mu U^\dagger$.

**Related Results:** `thm-su2-noether-current`

---

### Cluster Decomposition for Wilson Loops

**Type:** Theorem
**Label:** `thm-cluster-decomposition`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 6. Physical Observables and Wilson Loops](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
For non-overlapping Wilson loops $\gamma_1$, $\gamma_2$ separated by $d \gg \rho$:

$$
\langle W_{\gamma_1} W_{\gamma_2} \rangle \approx \langle W_{\gamma_1} \rangle \langle W_{\gamma_2} \rangle \quad \text{as } d \to \infty
$$

**Physical significance**: Ensures locality of gauge interactions.

**Fractal Set context**: Correlation decays exponentially:

$$
\langle W_{\gamma_1} W_{\gamma_2} \rangle - \langle W_{\gamma_1} \rangle \langle W_{\gamma_2} \rangle \sim e^{-d_{\text{alg}}(\gamma_1, \gamma_2)/\rho}
$$

due to localization kernel $K_\rho(x,y) = \exp(-d_{\text{alg}}^2/(2\rho^2))$.

---

### Continuum Limit of Discrete Yang-Mills

**Type:** Theorem
**Label:** `thm-continuum-limit-ym`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 7. Continuum Limit and Asymptotic Freedom](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
As lattice parameters $\Delta t, \rho \to 0$ with fixed physical coupling $g_{\text{phys}}$:

$$
S_{\text{YM}}^{\text{discrete}} \to S_{\text{YM}}^{\text{continuum}} = \frac{1}{4g_{\text{phys}}^2} \int d^{d+1}x \sum_{a=1}^3 F_{\mu\nu}^{(a)} F^{(a),\mu\nu}
$$

**Lattice renormalization**:

$$
\frac{1}{g_{\text{phys}}^2} = \frac{1}{g^2} + b_0 \log(\Lambda \tau) + O(g^2)
$$

where $b_0 = -11/(48\pi^2)$ for SU(2) (beta function coefficient).

**Asymptotic freedom**: As $\tau \to 0$, $g_{\text{phys}} \to 0$ (coupling vanishes at short distances).

---

### U(1) Fitness Charge Flow in Algorithmic Parameters

**Type:** Theorem
**Label:** `thm-u1-flow-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.1. U(1) Fitness Flow Equations](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `metric-tensor`, `noether`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The U(1) fitness charge evolves according to flow equations expressed entirely in algorithmic parameters.

**1. Total Fitness Charge:**

$$
Q_{\text{fitness}}(t) = \sum_{i \in A_t} \Phi(x_i(t))
$$

where $\Phi(x) = -U(x) + \beta r(x)$ with:
- $U(x)$: Confining potential (defines Valid Domain boundary)
- $r(x)$: Reward function
- $\beta$: Temperature parameter

**2. Discrete Flow Equation:**

$$
\frac{dQ_{\text{fitness}}}{dt} = \underbrace{\sum_{i \in A_t} \nabla \Phi(x_i) \cdot v_i}_{\text{Transport}} + \underbrace{\sum_{i \in A_t} \mathcal{S}_{\text{cloning}}(i,t)}_{\text{Cloning source}}
$$

**3. Transport Term (Algorithmic Parameters):**

From the Adaptive Gas SDE ({prf:ref}`def-hybrid-sde` in {doc}`07_adaptative_gas.md`):

$$
\nabla \Phi \cdot v_i = \nabla \Phi \cdot \left(v_i^{\text{drift}} + v_i^{\text{diffusion}}\right)
$$

where:

$$
v_i^{\text{drift}} = -\gamma v_i + F_{\text{confine}}(x_i) + F_{\text{adaptive}}(x_i, S_t) + F_{\text{viscous}}(x_i, S_t)
$$

**Components in algorithmic parameters:**

- **Friction**: $-\gamma v_i$ (parameter $\gamma > 0$)
- **Confining force**: $F_{\text{confine}} = -\nabla U(x)$
- **Adaptive force**:

$$
F_{\text{adaptive}}(x_i, S_t) = \epsilon_F \sum_{j \in A_t} K_\rho(y_i, y_j) \nabla V_{\text{fit}}(i|j)
$$

with localization kernel:

$$
K_\rho(y_i, y_j) = \frac{1}{Z_\rho(i)} \exp\left(-\frac{d_{\text{alg}}^2(y_i, y_j)}{2\rho^2}\right)
$$

and algorithmic distance:

$$
d_{\text{alg}}^2(y_i, y_j) = \|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2
$$

- **Viscous force**:

$$
F_{\text{viscous}}(x_i, S_t) = \nu \sum_{j \in A_t} K_\rho(y_i, y_j) (v_j - v_i)
$$

**Algorithmic parameters:**
- $\epsilon_F$: Adaptive force strength
- $\rho$: Localization scale
- $\lambda_v$: Velocity weight in algorithmic metric
- $\nu$: Viscosity coefficient

**4. Cloning Source Term:**

At each cloning event at time $t$:

$$
\mathcal{S}_{\text{cloning}}(i,t) = \begin{cases}
+\Phi(x_j) & \text{if walker } j \text{ born (clones } i) \\
-\Phi(x_i) & \text{if walker } i \text{ dies} \\
0 & \text{otherwise}
\end{cases}
$$

Cloning probability in algorithmic parameters ({doc}`13_fractal_set/00_full_set.md`):

$$
P_{\text{clone}}(i \to j) = P_{\text{comp}}^{(\text{clone})}(j|i) \cdot P_{\text{succ}}(S_{ij})
$$

where:

$$
P_{\text{comp}}^{(\text{clone})}(j|i) = \frac{\exp(-d_{\text{alg}}^2(i,j)/(2\epsilon_c^2))}{\sum_{j'} \exp(-d_{\text{alg}}^2(i,j')/(2\epsilon_c^2))}
$$

$$
P_{\text{succ}}(S_{ij}) = \sigma\left(\frac{S_{ij}}{T_{\text{clone}}}\right)
$$

with cloning score:

$$
S_{ij} = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)
$$

**Algorithmic parameters:**
- $\epsilon_c$: Cloning selection scale
- $T_{\text{clone}}$: Cloning temperature

**5. Complete Flow in Algorithmic Parameters:**

$$
\frac{dQ_{\text{fitness}}}{dt} = \sum_{i \in A_t} \nabla \Phi \cdot \left[-\gamma v_i - \nabla U + \epsilon_F \sum_j K_\rho \nabla V_{\text{fit}} + \nu \sum_j K_\rho (v_j - v_i)\right] + \mathcal{S}_{\text{cloning}}
$$

**Physical interpretation:**
- **Friction term**: Dissipates fitness charge via velocity damping
- **Confining force**: Redistributes fitness within Valid Domain
- **Adaptive force**: Drives fitness charge towards high-fitness regions (parameter $\epsilon_F$)
- **Viscous force**: Couples fitness flow between nearby walkers (parameter $\nu$)
- **Cloning source**: Injects/removes fitness at cloning events

**Related Results:** `def-hybrid-sde`

---

### SU(2) Isospin Current Flow in Algorithmic Parameters

**Type:** Theorem
**Label:** `thm-su2-flow-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.2. SU(2) Weak Isospin Flow Equations](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The SU(2) weak isospin currents evolve according to flow equations expressed in algorithmic parameters.

**1. Isospin Current for Pair (i,j):**

From {prf:ref}`thm-su2-noether-current`:

$$
J_\mu^{(a)}(i,j) = \bar{\Psi}_{ij} \gamma_\mu (T^a \otimes I_{\text{div}}) \Psi_{ij}
$$

where $|\Psi_{ij}\rangle = |↑\rangle \otimes |\psi_i\rangle + |↓\rangle \otimes |\psi_j\rangle$ and:

$$
|\psi_i\rangle = \sum_{k \in A_t \setminus \{i\}} \sqrt{P_{\text{comp}}^{(\text{div})}(k|i)} \cdot e^{i\theta_{ik}} |k\rangle
$$

**2. Diversity Companion Probability (Algorithmic):**

$$
P_{\text{comp}}^{(\text{div})}(k|i) = \frac{\exp\left(-\frac{d_{\text{alg}}^2(i,k)}{2\epsilon_d^2}\right)}{Z_i^{(\text{div})}}
$$

with partition function:

$$
Z_i^{(\text{div})} = \sum_{k' \in A_t \setminus \{i\}} \exp\left(-\frac{d_{\text{alg}}^2(i,k')}{2\epsilon_d^2}\right)
$$

**Algorithmic parameter**: $\epsilon_d$ (diversity selection scale)

**3. U(1) Phase (Algorithmic):**

$$
\theta_{ik} = -\frac{d_{\text{alg}}^2(i,k)}{2\epsilon_d^2 \hbar_{\text{eff}}}
$$

**Algorithmic parameters**:
- $\epsilon_d$: Sets phase scale
- $\hbar_{\text{eff}}$: Effective Planck constant (emergent quantum scale)

**4. Discrete Conservation Law:**

The isospin current satisfies (approximately, see {prf:ref}`thm-su2-noether-current`):

$$
\partial_0 J_0^{(a)}(i,j) + \sum_{k=1}^d \partial_k J_k^{(a)}(i,j) \approx 0
$$

**Temporal derivative** (along CST edges):

$$
\partial_0 J_0^{(a)} = \frac{J_0^{(a)}(t+\Delta t) - J_0^{(a)}(t)}{\Delta t}
$$

**Spatial derivative** (using localization kernel):

$$
\partial_k J_k^{(a)}(i,j) = \frac{1}{\rho} \sum_{(i',j') \in \text{IG}} K_\rho(y_{ij}, y_{i'j'}) \cdot (J_k^{(a)}(i',j') - J_k^{(a)}(i,j)) \cdot \hat{e}_k
$$

where $y_{ij} = (x_i + x_j)/2$ is the midpoint of the pair.

**5. Breaking of Exact Conservation:**

The effective mass $m_{\text{eff}} = \langle \Psi_{ij} | \hat{S}_{ij} | \Psi_{ij} \rangle$ breaks exact SU(2) invariance.

**Fitness comparison** (algorithmic):

$$
m_{\text{eff}}(i,j) = \sum_k P_{\text{comp}}^{(\text{div})}(k|i) V_{\text{fit}}(i|k) - \sum_m P_{\text{comp}}^{(\text{div})}(m|j) V_{\text{fit}}(j|m)
$$

where:

$$
V_{\text{fit}}(i|k) = \Phi(x_i) + \alpha \tilde{d}_i(k)
$$

with:
- $\Phi(x_i) = -U(x_i) + \beta r(x_i)$: Fitness function
- $\tilde{d}_i(k) = \frac{d_{\text{alg}}(i,k) - \mu_\rho(i)}{\sigma_\rho'(i)}$: ρ-localized Z-score

**Algorithmic parameters**:
- $\alpha$: Diversity weight
- $\rho$: Localization scale for statistics

**Conservation breaking:**

$$
\partial^\mu J_\mu^{(a)} = \bar{\Psi}_{ij} [m_{\text{eff}}, T^a \otimes I] \Psi_{ij}
$$

The commutator vanishes when $m_{\text{eff}}$ is constant (fitness-independent), but in general:

$$
|\partial^\mu J_\mu^{(a)}| \lesssim \frac{|\Delta m_{\text{eff}}|}{\Delta t}
$$

where $\Delta m_{\text{eff}}$ is the fitness variation scale.

**Algorithmic control of conservation quality**:
- Small $\alpha$: Weak diversity weight → better conservation
- Small $\epsilon_F$: Weak adaptive force → smaller fitness gradients → better conservation
- Large $\rho$: Large localization scale → smoother fitness landscape → better conservation

**Related Results:** `thm-su2-noether-current`

---

### Discrete Yang-Mills Equations in Algorithmic Parameters

**Type:** Theorem
**Label:** `thm-ym-eom-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.3. Yang-Mills Equations of Motion on the Lattice](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `curvature`, `fractal-set`, `gauge-theory`, `lattice`, `noether`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The Yang-Mills equations of motion on the Fractal Set are derived by varying the action with respect to the gauge field.

**1. Yang-Mills Action (from {prf:ref}`def-discrete-ym-action`):**

$$
S_{\text{YM}} = \frac{1}{g^2} \sum_{\square \in \text{Plaquettes}} 2\left(1 - \frac{1}{2}\text{Tr}(U_{\square})\right)
$$

where $U_{\square} = U_{12} U_{23} U_{34} U_{41}$ and:

$$
U_{ij} = \exp\left(i\tau \sum_{a=1}^3 A_e^{(a)} T^a\right)
$$

**2. Gauge Field in Algorithmic Parameters:**

From {prf:ref}`def-su2-gauge-field`:

$$
A_e^{(a)} \approx \frac{1}{\tau} \theta_{ij}^{(\text{SU}(2))} \delta^{a3} = -\frac{1}{\tau} \frac{d_{\text{alg}}^2(i,j)}{2\epsilon_c^2 \hbar_{\text{eff}}} \delta^{a3}
$$

**Algorithmic parameters**:
- $d_{\text{alg}}^2(i,j) = \|x_i - x_j\|^2 + \lambda_v \|v_i - v_j\|^2$
- $\epsilon_c$: Cloning selection scale
- $\hbar_{\text{eff}}$: Effective Planck constant
- $\tau$: Lattice coupling (related to edge length)

**3. Variation of Action:**

The equation of motion is obtained from $\delta S_{\text{YM}} / \delta A_e^{(a)} = 0$.

For small field:

$$
U_{\square} \approx I + i\tau^2 F_{\square} + O(\tau^4)
$$

where $F_{\square} = \sum_a F_{\square}^{(a)} T^a$ with:

$$
F_{\square}^{(a)} = \frac{1}{\tau^2} \left(A_{12}^{(a)} + A_{23}^{(a)} + A_{34}^{(a)} + A_{41}^{(a)}\right) + O(\tau^2)
$$

**4. Discrete Yang-Mills Equation:**

Varying the action gives:

$$
\sum_{\square \ni e} \left[\partial_\square U_e\right] = 0
$$

where the sum is over all plaquettes containing edge $e$.

**Explicit form** (weak field approximation):

$$
\sum_{\square \ni e} F_{\square}^{(a)} = g^2 J_e^{(a)}
$$

where $J_e^{(a)}$ is the current source on edge $e$ from matter coupling.

**5. Matter Current in Algorithmic Parameters:**

From the coupling action $S_{\text{coupling}} = g \sum_e J_e^{(a)} A_e^{(a)}$:

$$
J_e^{(a)} = \bar{\Psi}_{ij} \gamma_\mu (T^a \otimes I_{\text{div}}) \Psi_{ij}\Big|_{\text{edge } e}
$$

The dressed walker states $|\psi_i\rangle$ depend on:
- Diversity probabilities $P_{\text{comp}}^{(\text{div})}(k|i)$ (parameter $\epsilon_d$)
- Algorithmic distances $d_{\text{alg}}(i,k)$ (parameters $\lambda_v$, $\rho$)
- U(1) phases $\theta_{ik}$ (parameters $\epsilon_d$, $\hbar_{\text{eff}}$)

**6. Complete Discrete Equation (Algorithmic):**

$$
\sum_{\square \ni e} \left[\frac{A_{12}^{(a)} + A_{23}^{(a)} + A_{34}^{(a)} + A_{41}^{(a)}}{\tau^2}\right] = g^2 \bar{\Psi}_{ij} \gamma_\mu (T^a \otimes I) \Psi_{ij}
$$

where:
- Left side: Discrete curvature (depends on $d_{\text{alg}}$ via $A_e$)
- Right side: Matter current (depends on $P_{\text{comp}}, \theta_{ik}$)

**Algorithmic parameter dependence**:
- **Gauge field** $A_e$: $(\epsilon_c, \hbar_{\text{eff}}, \lambda_v, \rho, \tau)$
- **Matter current** $J_e$: $(\epsilon_d, \alpha, \beta, \epsilon_F, \nu)$
- **Coupling**: $g$

**7. Continuum Limit:**

As $\tau \to 0$, this becomes the standard Yang-Mills equation:

$$
D_\nu F^{(a),\mu\nu} = g^2 J^{(a),\mu}
$$

where $D_\nu$ is the covariant derivative in adjoint representation.

**Related Results:** `def-su2-gauge-field`, `def-discrete-ym-action`

---

### Ward Identities from Gauge Invariance

**Type:** Theorem
**Label:** `thm-ward-identities-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.4. Ward Identities and Conserved Charges](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
Gauge invariance of the path integral implies **Ward identities** relating correlation functions. These can be expressed in algorithmic parameters.

**1. U(1) Ward Identity:**

For any gauge-invariant observable $\mathcal{O}$:

$$
\langle \frac{\delta \mathcal{O}}{\delta \alpha} \rangle = 0
$$

where $\alpha$ is the global U(1) phase.

**Explicit form** (fitness charge conservation):

$$
\frac{d}{dt}\langle Q_{\text{fitness}}(t) \rangle = \langle \mathcal{S}_{\text{cloning}}(t) \rangle
$$

In algorithmic parameters:

$$
\frac{d}{dt}\left\langle \sum_{i \in A_t} \Phi(x_i) \right\rangle = \sum_{i,j} P_{\text{clone}}(i \to j) (\Phi(x_j) - \Phi(x_i))
$$

where $P_{\text{clone}}$ depends on $(\epsilon_c, T_{\text{clone}}, \alpha, \beta, \rho)$.

**2. SU(2) Ward Identity:**

For infinitesimal SU(2) transformation $\delta_a \Psi = i\epsilon^a (T^a \otimes I) \Psi$:

$$
\langle \delta_a \mathcal{O} \rangle = 0
$$

**Current conservation** (approximately):

$$
\left\langle \partial^\mu J_\mu^{(a)} \right\rangle \approx \left\langle \bar{\Psi}_{ij} [m_{\text{eff}}, T^a] \Psi_{ij} \right\rangle
$$

The right side vanishes when fitness is uniform ($m_{\text{eff}} = \text{const}$).

**In algorithmic parameters**:

$$
\left\langle \partial^\mu J_\mu^{(a)} \right\rangle \sim \epsilon_F \cdot (\text{fitness gradient scale})
$$

Setting $\epsilon_F \to 0$ (no adaptive force) gives exact Ward identity.

**3. Generalized Ward-Takahashi Identity:**

For gauge field correlations:

$$
\langle A_e^{(a)} A_{e'}^{(b)} \rangle - \langle A_e^{(a)} \rangle \langle A_{e'}^{(b)} \rangle = \delta_{ab} G(e, e')
$$

where $G(e,e')$ is the propagator.

**In algorithmic parameters**:

$$
G(e,e') \propto K_\rho(y_e, y_{e'}) = \exp\left(-\frac{d_{\text{alg}}^2(y_e, y_{e'})}{2\rho^2}\right)
$$

The gauge field correlations decay exponentially with algorithmic distance, with decay length $\rho$.

**Physical interpretation**: The localization scale $\rho$ sets the correlation length of gauge field fluctuations.

---

### Discrete Gauss Law Constraint

**Type:** Theorem
**Label:** `thm-discrete-gauss-law`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.4. Ward Identities and Conserved Charges](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
On the Fractal Set lattice, the Gauss law constraint relates the electric field to the charge density.

**1. Electric Field (Temporal Component):**

Define the **discrete electric field** on edge $e$:

$$
E_e^{(a)} := \frac{1}{\tau} (A_0^{(a)}(n_{i,t+1}) - A_0^{(a)}(n_{i,t}))
$$

where $A_0$ is the temporal component of the gauge field.

**2. Charge Density at Node:**

The **SU(2) charge density** at node $n_{i,t}$ is:

$$
\rho^{(a)}(n_{i,t}) := J_0^{(a)}(n_{i,t}) = \bar{\Psi}_{ij} \gamma_0 (T^a \otimes I) \Psi_{ij}
$$

**In algorithmic parameters**:

$$
\rho^{(a)} \sim P_{\text{comp}}^{(\text{div})}(\epsilon_d) \cdot (\text{fitness asymmetry})
$$

**3. Discrete Gauss Law:**

$$
\sum_{e \text{ from } n} E_e^{(a)} = g^2 \rho^{(a)}(n)
$$

where the sum is over all edges emanating from node $n$.

**Explicit form**:

$$
\sum_{j \text{ neighbors of } i} \frac{A_0^{(a)}(n_{j,t+1}) - A_0^{(a)}(n_{i,t})}{\tau} = g^2 \bar{\Psi}_{ij} \gamma_0 T^a \Psi_{ij}
$$

**4. Algorithmic Interpretation:**

The Gauss law states that the **flux of the electric field** out of a node equals the **SU(2) charge** at that node.

**Left side** (flux): Depends on gauge field variations $\sim d_{\text{alg}}^2/(\tau \epsilon_c^2 \hbar_{\text{eff}})$

**Right side** (charge): Depends on matter distribution $\sim P_{\text{comp}}(\epsilon_d)$

**Balance condition**:

$$
\frac{d_{\text{alg}}^2}{\tau \epsilon_c^2 \hbar_{\text{eff}}} \sim g^2 P_{\text{comp}}(\epsilon_d)
$$

This determines the self-consistent gauge field configuration.

**5. Constraint on Algorithm Parameters:**

For consistent gauge structure:

$$
\frac{\epsilon_d^2}{\epsilon_c^2} \sim g^2 \tau \hbar_{\text{eff}}
$$

This relates diversity selection scale $\epsilon_d$ to cloning selection scale $\epsilon_c$ via the gauge coupling $g$.

---

### Effective Planck Constant from Walker Dynamics

**Type:** Theorem
**Label:** `thm-effective-planck-constant`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.2. Effective Planck Constant from Kinetic Energy](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The effective Planck constant $\hbar_{\text{eff}}$ that governs quantum-like interference in the Fractal Set emerges from the kinetic energy scale and cloning timescale:

$$
\hbar_{\text{eff}} = m \epsilon_c^2 / \tau
$$

where:
- $m$: walker mass
- $\epsilon_c$: cloning selection scale
- $\tau$: cloning timestep

**Dimensional verification:**

$$
[\hbar_{\text{eff}}] = [M] \cdot [L]^2 \cdot [T]^{-1} = [M][L]^2[T]^{-1} = [\text{Action}] \quad \checkmark
$$

---

### Discrete Field Strength in Terms of Algorithmic Distance

**Type:** Theorem
**Label:** `thm-field-strength-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.3. Field Strength Tensor from Lattice Geometry](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The SU(2) field strength tensor $F_{\square}$ for a plaquette $\square = (i,j,k,\ell)$ is given by:

$$
F_{\square} = \frac{m}{2\tau^3 \hbar_{\text{eff}}} \left[ d_{\text{alg}}^2(i,j) + d_{\text{alg}}^2(j,k) - d_{\text{alg}}^2(k,\ell) - d_{\text{alg}}^2(\ell,i) + d_{\text{alg}}^2(i,k) - d_{\text{alg}}^2(j,\ell) \right] + O(\tau^2)
$$

where this is the trace part of the field strength matrix.

---

### SU(2) Gauge Coupling Constant (Dimensionally Correct)

**Type:** Theorem
**Label:** `thm-su2-coupling-constant`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.4. SU(2) Gauge Coupling from Wilson Action](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `yang-mills`

**Statement:**
The dimensionless SU(2) gauge coupling constant is:

$$
g_{\text{weak}}^2 = \frac{\tau \rho^2}{m \epsilon_c^2}
$$

**Dimensional verification:**

$$
[g_{\text{weak}}^2] = \frac{[T] [L]^2}{[M] [L]^2} = \frac{[T]}{[M]} = [1] \quad \text{if we set } [M] = [T]
$$

In natural units where $\hbar = c = 1$, mass and inverse time have the same dimension, so $g^2$ is dimensionless. $\checkmark$

---

### U(1) Fitness Gauge Coupling (Dimensionally Correct)

**Type:** Theorem
**Label:** `thm-u1-coupling-constant`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.5. U(1) Fitness Gauge Coupling from Adaptive Force](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The dimensionless U(1) fitness gauge coupling is:

$$
e_{\text{fitness}}^2 = \frac{m}{\epsilon_F \rho}
$$

**Dimensional verification (natural units $\hbar = c = 1$):**

Using $[\epsilon_F] = [M][L]^2[T]^{-2} = [M]^3$ in natural units where $[L] = [M]^{-1}$ and $[T] = [M]^{-1}$:

$$
[e_{\text{fitness}}^2] = \frac{[M]}{[M]^3 \cdot [M]^{-1}} = \frac{[M]}{[M]^2} = [M]^{-1} = [L]
$$

To make this dimensionless, we need one more factor. The correct dimensionless form is:

$$
e_{\text{fitness}}^2 = \frac{m^2}{\epsilon_F}
$$

giving $[e^2] = [M]^2 / [M]^3 = [M]^{-1} = [L]$, which equals $[1]$ when $[L]$ is treated as dimensionless ratio. $\checkmark$

---

### Emergent Mass Scales from Operator Spectra (Dimensionally Correct)

**Type:** Theorem
**Label:** `thm-mass-scales`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.6. Mass Scales from Spectral Properties](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `convergence`, `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The Fractal Set framework generates multiple mass scales from the spectral gaps of different operators:

**1. Cloning mass scale:**

$$
m_{\text{clone}} = \frac{1}{\epsilon_c}
$$

**Dimensional check:** $[m_{\text{clone}}] = [L]^{-1} = [M]$ in natural units. $\checkmark$

**Derivation**: The cloning kernel $P_{\text{clone}}(i \to j) \propto \exp(-d_{\text{alg}}^2/(2\epsilon_c^2))$ has Fourier transform:

$$
\widetilde{P}_{\text{clone}}(k) \sim \exp\left(-\frac{\epsilon_c^2 k^2}{2}\right)
$$

For $k \ll 1/\epsilon_c$, this behaves as a massive propagator $1/(k^2 + m^2)$ with $m = 1/\epsilon_c$.

**2. Mean-field mass scale:**

$$
m_{\text{MF}} = \frac{1}{\rho}
$$

**Dimensional check:** $[m_{\text{MF}}] = [L]^{-1} = [M]$ in natural units. $\checkmark$

**Derivation**: The localization kernel $K_\rho(r) = \exp(-r^2/(2\rho^2))$ has the same Fourier structure, giving mass scale $m_{\text{MF}} = 1/\rho$.

**3. Spectral gap mass (convergence):**

$$
m_{\text{gap}} = \sqrt{\frac{\lambda_{\text{gap}}}{\tau}}
$$

where $\lambda_{\text{gap}}$ is the spectral gap of the generator (see {doc}`04_convergence.md`).

**Dimensional check:** $[m_{\text{gap}}] = \sqrt{[T]^{-1}/[T]} = [T]^{-1} = [M]$ in natural units. $\checkmark$

**Interpretation**: This is the mass scale associated with relaxation to quasi-stationary distribution. The correlation time is $\tau_{\text{corr}} \sim 1/\lambda_{\text{gap}}$, giving a mass $m_{\text{gap}} = \sqrt{1/(\tau \tau_{\text{corr}})}$.

**4. Friction mass:**

$$
m_{\text{friction}} = \gamma
$$

**Dimensional check:** $[m_{\text{friction}}] = [T]^{-1} = [M]$ in natural units. $\checkmark$

**Interpretation**: The friction coefficient directly sets an inverse timescale for velocity relaxation.

**5. Mass hierarchy:**

For efficient algorithm operation with separation of scales:

$$
m_{\text{friction}} \ll m_{\text{gap}} < m_{\text{MF}} < m_{\text{clone}}
$$

This corresponds to:

$$
\gamma \ll \sqrt{\frac{\lambda_{\text{gap}}}{\tau}} < \frac{1}{\rho} < \frac{1}{\epsilon_c}
$$

$$
\gamma \ll \frac{\nu}{\rho^4} < \frac{\epsilon_F}{\rho^4} \ll \frac{T_{\text{clone}}}{\epsilon_c^2}
$$

**Physical meaning:**
- Friction acts on longest timescales (small mass)
- Spectral gap sets convergence rate
- Mean-field range controls collective behavior
- Cloning has shortest range (large mass)

---

### Correlation Length (Rigorous Derivation)

**Type:** Theorem
**Label:** `thm-correlation-length`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.7. Correlation Length from Spectral Gap](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The spatial correlation length is determined by the ratio of mass scales:

$$
\xi = \frac{m_{\text{clone}}}{m_{\text{gap}}} = \frac{\epsilon_c^{-1}}{\sqrt{\lambda_{\text{gap}}/\tau}} = \frac{1}{\epsilon_c} \sqrt{\frac{\tau}{\lambda_{\text{gap}}}}
$$

**Dimensional check:** $[\xi] = [L]^{-1} \cdot \sqrt{[T]/[T]^{-1}} = [L]^{-1} \cdot [T] = [L]$ in natural units. $\checkmark$

---

### Dimensionless Parameter Ratios

**Type:** Theorem
**Label:** `thm-dimensionless-ratios`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.8. Fine Structure Constant and Dimensionless Ratios](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The following dimensionless combinations characterize the algorithm regime:

**1. Coupling strength:**

$$
\alpha_{\text{FS}} = \frac{\tau \rho^2}{m \epsilon_c^2}
$$

**2. Scale separation:**

$$
\sigma_{\text{sep}} = \frac{\epsilon_c}{\rho}
$$

**3. Timescale ratio:**

$$
\eta_{\text{time}} = \tau \lambda_{\text{gap}}
$$

**4. Correlation-to-interaction ratio:**

$$
\kappa = \frac{\xi}{\rho} = \frac{\epsilon_c}{\rho \sqrt{\tau \lambda_{\text{gap}}}}
$$

These ratios determine the algorithmic phase:
- Perturbative: $\alpha_{\text{FS}} \ll 1$, $\sigma_{\text{sep}} \ll 1$, $\eta_{\text{time}} \ll 1$
- Non-perturbative: $\alpha_{\text{FS}} \sim 1$, $\sigma_{\text{sep}} \sim 1$, $\eta_{\text{time}} \sim 1$
- Strong coupling: $\alpha_{\text{FS}} \gg 1$, $\sigma_{\text{sep}} \gg 1$, $\kappa \gg 1$

---

### Renormalization Group Flow in Algorithmic Parameters

**Type:** Theorem
**Label:** `thm-rg-flow-algorithmic`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.7. Coupling Constants and Renormalization](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The renormalization group (RG) flow of the effective gauge theory can be expressed entirely in terms of algorithmic parameters:

**1. SU(2) beta function:**

$$
\frac{d g_{\text{weak}}^2}{d \ln \mu} = -\beta_0 g_{\text{weak}}^4 + O(g_{\text{weak}}^6)
$$

where $\beta_0 = 22/(48\pi^2)$ for SU(2), and the scale is $\mu \sim 1/\tau$.

Substituting $g_{\text{weak}}^2 = \epsilon_c^2 \hbar_{\text{eff}}/(\tau^2 \rho^4)$:

$$
\frac{d}{d \ln \mu}\left(\frac{\epsilon_c^2 \hbar_{\text{eff}}}{\tau^2 \rho^4}\right) = -\beta_0 \left(\frac{\epsilon_c^2 \hbar_{\text{eff}}}{\tau^2 \rho^4}\right)^2
$$

**2. Algorithmic parameter flow:**

If we assume $\hbar_{\text{eff}} = \epsilon_c^2 T_{\text{clone}}$ is fixed, then:

$$
\frac{d \epsilon_c^2}{d \ln \mu} = -\beta_0 \frac{\epsilon_c^6 T_{\text{clone}}^2}{\tau^4 \rho^8}
$$

or equivalently:

$$
\frac{d \rho^4}{d \ln \mu} = \beta_0 \frac{\epsilon_c^4 T_{\text{clone}} \rho^4}{\tau^4}
$$

**3. Asymptotic freedom:**

At high energy scales ($\mu \to \infty$, $\tau \to 0$):

$$
g_{\text{weak}}^2(\mu) \to 0
$$

This requires either:
- $\epsilon_c \to 0$ (cloning becomes ultra-local)
- $\rho \to \infty$ (mean-field range increases)
- $\tau \to 0$ (time discretization refined)

**4. Infrared behavior:**

At low energy scales ($\mu \to 0$, $\tau \to \infty$):

$$
g_{\text{weak}}^2(\mu) \to \infty
$$

suggesting strong coupling or confinement.

**5. U(1) running:**

The U(1) coupling has opposite sign beta function (asymptotic growth):

$$
\frac{d e_{\text{fitness}}^2}{d \ln \mu} = \beta_0^{(\text{U}(1))} e_{\text{fitness}}^4
$$

where $\beta_0^{(\text{U}(1))} > 0$. Substituting $e_{\text{fitness}}^2 = 1/(\epsilon_F \rho^2)$:

$$
\frac{d}{d \ln \mu}\left(\frac{1}{\epsilon_F \rho^2}\right) = \beta_0^{(\text{U}(1))} \frac{1}{\epsilon_F^2 \rho^4}
$$

This implies $\epsilon_F$ must decrease or $\rho$ must increase at higher scales to maintain the U(1) coupling growth.

---

### Measurable Signatures of Fundamental Constants

**Type:** Theorem
**Label:** `thm-measurable-signatures`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.9. Experimental Predictions and Observables](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `area`, `conservation`, `convergence`, `fractal-set`, `gauge-theory`, `noether`, `su2-symmetry`, `symmetry`, `u1-symmetry`, `yang-mills`

**Statement:**
The fundamental constants derived above lead to experimentally verifiable predictions in algorithmic observables:

**1. Correlation length scaling:**

The two-point correlation function of walker positions should decay as:

$$
\langle x_i(t) x_j(t) \rangle - \langle x_i(t) \rangle \langle x_j(t) \rangle \sim e^{-|i-j|/\xi}
$$

where the correlation length is:

$$
\xi \sim \frac{\lambda_{\text{MF}}}{\alpha_{\text{FS}}} = \frac{\rho \tau^2 \rho^2}{\epsilon_c^2} = \frac{\tau^2 \rho^3}{\epsilon_c^2}
$$

**Prediction**: Plotting $\ln(\text{correlation})$ vs walker distance should yield a straight line with slope $-1/\xi$.

**2. Critical slowing down near convergence:**

As the algorithm approaches convergence, the relaxation time should diverge as:

$$
\tau_{\text{relax}}(\epsilon) \sim \tau_{\text{relax}}(0) \left(\frac{\epsilon}{\epsilon_0}\right)^{-z}
$$

where $\epsilon$ is the distance to optimum and $z$ is the dynamical critical exponent:

$$
z = \frac{m_{\text{clone}}^2}{m_{\text{loc}}^2} = \frac{T_{\text{clone}} \rho^4}{\epsilon_c^2 \epsilon_F}
$$

**Prediction**: Log-log plot of relaxation time vs $\epsilon$ near convergence should have slope $-z$.

**3. Wilson loop area law:**

For a rectangular loop of size $L \times T$, the Wilson loop expectation should scale as:

$$
\langle W_{L \times T} \rangle \sim \exp\left(-\sigma L T\right)
$$

where the string tension is:

$$
\sigma = \frac{g_{\text{weak}}^2}{\lambda_{\text{clone}}^2} = \frac{\epsilon_c^2 \hbar_{\text{eff}}}{\tau^2 \rho^4 \epsilon_c^2} = \frac{T_{\text{clone}}}{\tau^2 \rho^4}
$$

**Prediction**: Plotting $-\ln(\langle W \rangle)$ vs area $LT$ should yield slope $\sigma$.

**4. Asymptotic freedom signature:**

The effective coupling should decrease at short distances (high energies):

$$
g_{\text{weak}}^2(\tau') = \frac{g_{\text{weak}}^2(\tau)}{1 + \beta_0 g_{\text{weak}}^2(\tau) \ln(\tau/\tau')}
$$

**Prediction**: Measuring cloning correlation strength at different time discretizations should show logarithmic running.

**5. Noether charge conservation:**

The U(1) fitness charge and SU(2) isospin charges should be conserved up to cloning source terms:

$$
\frac{dQ_{\text{fitness}}}{dt} \approx \mathcal{S}_{\text{cloning}}
$$

**Prediction**: Tracking fitness charge over time and comparing to cloning event frequency should verify Ward identity.

---

### UV Safety from Uniform Ellipticity

**Type:** Theorem
**Label:** `thm-uv-safety-elliptic-diffusion`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.10. UV Safety and Mass Gap Survival in Continuum Limit](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The regularized diffusion tensor $\Sigma_{\text{reg}}(x, S) = (H(x, S) + \epsilon_\Sigma I)^{-1/2}$ provides **UV protection** for the spectral gap, preventing its collapse as $\tau \to 0$.

**Mechanism**: The uniform ellipticity bounds from {prf:ref}`thm-uniform-ellipticity` in {doc}`08_emergent_geometry.md`:

$$
c_{\min}(\rho) I \preceq D_{\text{reg}}(x, S) \preceq c_{\max}(\rho) I
$$

ensure that the generator $\mathcal{L}$ has spectral gap bounded below:

$$
\lambda_{\text{gap}} \geq \lambda_{\text{gap,min}}(\gamma, c_{\min}, \kappa_{\text{conf}}) > 0
$$

**independent of the timestep** $\tau$ (which only appears in the discretized evolution, not in the continuous-time generator).

**Key insight**: The spectral gap $\lambda_{\text{gap}}$ is a property of the **continuous-time generator**:

$$
\mathcal{L} = v \cdot \nabla_x - \nabla U(x) \cdot \nabla_v - \gamma v \cdot \nabla_v + \text{Tr}(D_{\text{reg}} \nabla_v^2)
$$

and does NOT depend on the discrete timestep $\tau$. The timestep only enters through the **BAOAB integrator** for numerical implementation (see {doc}`02_euclidean_gas.md` §3.2).

**Related Results:** `thm-uniform-ellipticity`

---

### Mass Gap Survival via RG Fixed Point

**Type:** Theorem
**Label:** `thm-mass-gap-rg-fixed-point`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 9.10. UV Safety and Mass Gap Survival in Continuum Limit](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The mass gap survives the continuum limit $\tau \to 0$ if the renormalization group (RG) flow has a **UV fixed point** where physical masses remain finite.

**RG flow equations** (from {prf:ref}`thm-rg-flow-algorithmic`):

$$
\frac{d g_{\text{weak}}^2}{d \ln \mu} = -\beta_0 (g_{\text{weak}}^2)^2, \quad \beta_0 = \frac{11}{3} > 0
$$

This is **asymptotic freedom**: the gauge coupling $g_{\text{weak}}^2 \to 0$ as $\mu \to \infty$ (UV).

**Fixed point analysis**:

At the UV fixed point $g_{\text{weak}}^* = 0$, the Yang-Mills theory becomes a **free theory**, and the mass gap is protected by the conformal symmetry of the free theory (broken by quantum corrections at finite coupling).

**However**, the spectral gap $\lambda_{\text{gap}}$ depends on the **full generator**, including:
1. The kinetic operator (hypocoercive diffusion)
2. The cloning operator (competitive selection)
3. The adaptive force (fitness-driven drift)

**Claim**: The uniform ellipticity of $D_{\text{reg}}$ ensures that $\lambda_{\text{gap}} \geq c_{\min}(\rho) \gamma > 0$ for all scales, preventing UV collapse.

**Related Results:** `thm-rg-flow-algorithmic`

---

### Corollarys (2)

### Conservation Between Cloning Events

**Type:** Corollary
**Label:** `cor-u1-conservation-between-cloning`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.1. U(1) Fitness Flow Equations](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `conservation`, `fractal-set`, `gauge-theory`, `noether`, `symmetry`, `yang-mills`

**Statement:**
Between cloning events, the total fitness charge satisfies:

$$
\frac{dQ_{\text{fitness}}}{dt} = -\gamma \sum_{i \in A_t} \nabla \Phi \cdot v_i + \text{O}(D)
$$

where the first term is dissipation and $D$ is the diffusion coefficient.

**In the limit $\gamma, D \to 0$ (Hamiltonian dynamics):**

$$
\frac{dQ_{\text{fitness}}}{dt} = 0
$$

The fitness charge is **exactly conserved** in this limit.

**Algorithmic control**: Conservation quality is controlled by:
- Friction $\gamma$ (smaller = better conservation)
- Diffusion $D$ (smaller = better conservation)
- Timestep $\Delta t$ (smaller = better conservation)

---

### Algorithmic Control of Yang-Mills Dynamics

**Type:** Corollary
**Label:** `cor-ym-algorithmic-control`
**Source:** [13_fractal_set_new/03_yang_mills_noether.md § 8.3. Yang-Mills Equations of Motion on the Lattice](13_fractal_set_new/03_yang_mills_noether.md)
**Tags:** `convergence`, `fractal-set`, `gauge-theory`, `lattice`, `noether`, `symmetry`, `yang-mills`

**Statement:**
The Yang-Mills field dynamics can be controlled via algorithmic parameters:

**1. Coupling strength:**

The effective Yang-Mills coupling is:

$$
g_{\text{eff}}^2 \propto \frac{\epsilon_c^2 \hbar_{\text{eff}}^2}{\tau^2 \rho^4}
$$

**Increasing** $g_{\text{eff}}$ (stronger gauge interaction):
- Decrease $\epsilon_c$ (tighter cloning selection)
- Decrease $\hbar_{\text{eff}}$ (smaller quantum scale)
- Decrease $\rho$ (smaller localization)
- Increase $\tau$ (coarser lattice)

**2. Field strength scale:**

The typical field strength is:

$$
|F| \sim \frac{d_{\text{alg}}^2}{\tau^2 \epsilon_c^2 \hbar_{\text{eff}}}
$$

**Increasing** field strength:
- Increase walker separation (larger $d_{\text{alg}}$)
- Decrease $\epsilon_c$ or $\hbar_{\text{eff}}$

**3. Matter coupling:**

The matter current magnitude is:

$$
|J| \sim P_{\text{comp}}^{(\text{div})} \sim \exp\left(-\frac{d_{\text{alg}}^2}{2\epsilon_d^2}\right)
$$

**Increasing** matter coupling:
- Increase $\epsilon_d$ (broader diversity selection)
- Increase walker density (more companions)

**Practical implications**:
- **Weak coupling regime** ($g_{\text{eff}} \ll 1$): Large $\rho$, small $\epsilon_c$ → perturbative Yang-Mills
- **Strong coupling regime** ($g_{\text{eff}} \gg 1$): Small $\rho$, large $\epsilon_c$ → non-perturbative, confinement

The algorithm naturally explores both regimes by varying $\rho$ during convergence.

---


## 04_rigorous_additions.md

**Objects in this document:** 21

### Definitions (4)

### Episode Relabeling Gauge Group

**Type:** Definition
**Label:** `def-episode-relabeling-group-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.1. Episode Permutation Group as Gauge Symmetry](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `gauge-theory`, `metric-tensor`, `proofs`, `rigor`

**Statement:**
Let $\mathcal{E} = \\{e_1, \ldots, e_{|\mathcal{E}|}\\}$ be the set of episodes in the Fractal Set. The **episode relabeling group** is:

$$
G_{\text{epi}} = S_{|\mathcal{E}|}
$$

the symmetric group on $|\mathcal{E}|$ elements.

**Gauge transformation:** A permutation $\sigma \in S_{|\mathcal{E}|}$ acts on the Fractal Set by:

$$
\sigma: e_i \mapsto e_{\sigma(i)}
$$

relabeling all episodes.

**Physical equivalence:** Two Fractal Sets $\mathcal{F}$ and $\mathcal{F}'$ are **physically equivalent** if $\mathcal{F}' = \sigma \cdot \mathcal{F}$ for some $\sigma \in S_{|\mathcal{E}|}$.

**Source:** 13_B §1.1 Definition 1.1.1 (`def-episode-relabeling-group`); extracted_mathematics_13B.md.

---

### Discrete Gauge Connection on Fractal Set

**Type:** Definition
**Label:** `def-discrete-gauge-connection-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.2. Discrete Parallel Transport and Connection](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `causal-tree`, `fractal-set`, `gauge-theory`, `proofs`, `rigor`

**Statement:**
The **discrete gauge connection** defines parallel transport along CST and IG edges.

**Gauge group:** $G_{\text{gauge}} = S_{|\mathcal{E}|}$ (episode permutations)

**Parallel transport operators:**

1. **CST edges** (timelike, causal):
   $$\mathcal{T}^{\text{CST}}(e_i^t \to e_i^{t+1}) = \text{id} \in S_{|\mathcal{E}|}$$

   **Interpretation:** Episode index is **preserved** along temporal evolution (same walker)

2. **IG edges** (spacelike, non-causal):
   $$\mathcal{T}^{\text{IG}}(e_i \to e_j) = (i \, j) \in S_{|\mathcal{E}|}$$

   **Interpretation:** IG edge represents **exchange/correlation** between episodes $i$ and $j$, encoded as transposition

**Path-ordered product:** For a path $\gamma = (e_1 \to e_2 \to \cdots \to e_n)$:

$$
\mathcal{T}(\gamma) = \mathcal{T}(e_{n-1} \to e_n) \circ \cdots \circ \mathcal{T}(e_1 \to e_2) \in S_{|\mathcal{E}|}
$$

**Source:** 13_B §2.1 Definition 2.1.2 (`def-discrete-gauge-connection`); extracted_mathematics_13B.md.

---

### Discrete Curvature Functional

**Type:** Definition
**Label:** `def-discrete-curvature-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.3. Wilson Loops and Holonomy Observables](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `curvature`, `fractal-set`, `proofs`, `rigor`

**Statement:**
For a closed loop (plaquette) $P$ in the Fractal Set, define the **discrete curvature** as:

$$
\kappa(P) = \begin{cases}
0 & \text{if } \text{Hol}(P) = \text{id} \\
1 & \text{if } \text{Hol}(P) \neq \text{id}
\end{cases}
$$

**Interpretation:** The loop has non-trivial holonomy if parallel transport around it produces a non-identity permutation.

**Source:** 13_B §2.2 Definition 2.2.2 (`def-discrete-curvature`); extracted_mathematics_13B.md.

---

### Order-Invariant Functional

**Type:** Definition
**Label:** `def-order-invariant-functional-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 6.1. Order-Invariant Functionals](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `causal-tree`, `fractal-set`, `gauge-theory`, `proofs`, `rigor`

**Statement:**
A functional $F: \mathcal{F} \to \mathbb{R}$ on Fractal Sets is **order-invariant** (or **causal-automorphism-invariant**) if:

$$
F(\psi(\mathcal{F})) = F(\mathcal{F})
$$

for all **causal automorphisms** $\psi$ - graph isomorphisms that preserve:
1. CST temporal ordering (causality)
2. IG edge structure

**Examples:**
- Total fitness: $F(\mathcal{F}) = \sum_{e \in \mathcal{E}} \Phi(e)$
- Episode measure: $\mu_{\text{epi}}(A) = |\{e : x_e \in A\}| / |\mathcal{E}|$
- Wilson loops: $W[\gamma] = \text{Tr}[\text{Hol}(\gamma)]$

**Physical interpretation:** Order-invariant functionals are the **gauge-invariant observables** of the theory.

**Source:** 13_A §3 Definition `def-d-order-invariant-functionals`.

---

### Lemmas (3)

### Fast Velocity Thermalization Justifies Annealed Approximation

**Type:** Lemma
**Label:** `lem-velocity-marginalization`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.2. Velocity Marginalization and Timescale Separation](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `metric-tensor`, `proofs`, `rigor`

**Statement:**
Consider the full Langevin dynamics on phase space $(x, v) \in \mathcal{X} \times \mathbb{R}^d$. Under the assumptions of Chapter 04 (geometric ergodicity of kinetic operator), there is a **timescale separation**:

$$
\tau_v \ll \tau_x
$$

where:
- $\tau_v \sim \gamma^{-1}$: Velocity thermalization time
- $\tau_x \sim \epsilon_c^{-2}$: Spatial exploration time (diffusion)

with $\epsilon_c = \sqrt{T/\gamma}$ the thermal coherence length.

**Consequence:** On the timescale of spatial diffusion, velocities are effectively in thermal equilibrium at each position $x$. This justifies the **annealed approximation** where IG edge weights (which depend on temporal overlap) can be approximated by a position-dependent kernel:

$$
W(x_i, x_j) = C_v \exp\left( -\frac{\|x_i - x_j\|^2}{2\epsilon_c^2} \right)
$$

**Source:** `velocity_marginalization_rigorous.md` §2 Theorem `thm-timescale-separation`.

---

### Riemann Sum Convergence for Episodes

**Type:** Lemma
**Label:** `lem-sum-to-integral-episodes`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.3. Covariance Matrix Convergence to Inverse Metric](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `convergence`, `fractal-set`, `proofs`, `qsd`, `riemannian`, `rigor`, `volume`

**Statement:**
For any continuous function $h: \mathcal{X} \to \mathbb{R}$:

$$
\frac{1}{N_{\text{local}}} \sum_{e_j \in \mathcal{N}_{\epsilon}(e_i)} h(x_j) \xrightarrow{N \to \infty} \int_{B_{\epsilon}(x_i)} h(x) \, \rho_{\text{spatial}}(x) \, dx / \int_{B_{\epsilon}(x_i)} \rho_{\text{spatial}}(x) \, dx
$$

where $B_{\epsilon}(x_i)$ is the $\epsilon$-ball centered at $x_i$.

**Proof:** This is Riemann sum convergence. Episodes sample from $\rho_{\text{spatial}} \propto \sqrt{\det g} \, e^{-U_{\text{eff}}/T}$ by {prf:ref}`thm-qsd-spatial-riemannian-volume`. Standard concentration inequalities (Hoeffding, Bernstein) give the rate $O(N_{\text{local}}^{-1/2})$. ∎

**Related Results:** `thm-qsd-spatial-riemannian-volume`

---

### Gaussian Covariance in Riemannian Normal Coordinates

**Type:** Lemma
**Label:** `lem-gaussian-covariance-curved`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.3. Covariance Matrix Convergence to Inverse Metric](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `metric-tensor`, `proofs`, `riemannian`, `rigor`, `volume`

**Statement:**
In Riemannian normal coordinates centered at $x_0$ with metric $g_{ij}(x_0) + O(\|x - x_0\|^2)$:

$$
\Sigma(\epsilon) = \epsilon^2 g(x_0)^{-1} + O(\epsilon^3)
$$

**Proof sketch:**
1. Change to normal coordinates $y = \exp_{x_0}^{-1}(x)$
2. Metric becomes $g_{ij}(y) = \delta_{ij} + O(\|y\|^2)$
3. Volume measure becomes $\sqrt{\det g(y)} = 1 + O(\|y\|^2)$
4. Integral reduces to Gaussian moment:

$$
\Sigma(\epsilon) \approx \int_{B_{\epsilon}(0)} w(\|y\|) \, y y^T \, dy / \int_{B_{\epsilon}(0)} w(\|y\|) \, dy
$$

5. For Gaussian weight $w(\|y\|) = \exp(-\|y\|^2 / (2\epsilon_c^2))$:

$$
\int y_i y_j \, e^{-\|y\|^2 / (2\epsilon_c^2)} \, dy = \delta_{ij} \cdot \epsilon_c^2 \cdot (\text{const})
$$

6. Rescaling by $\epsilon$ gives $\Sigma(\epsilon) = \epsilon^2 I + O(\epsilon^3)$
7. Transforming back to original coordinates: $I \to g(x_0)^{-1}$ ∎

---

### Propositions (4)

### Continuum Covariance Formula

**Type:** Proposition
**Label:** `prop-continuum-covariance-integral`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.3. Covariance Matrix Convergence to Inverse Metric](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `proofs`, `rigor`

**Statement:**
The continuum limit of the covariance sum is:

$$
\Sigma(\epsilon) := \frac{\int_{B_{\epsilon}(x_0)} w(x - x_0) \, (x - x_0)(x - x_0)^T \, \rho(x) \, dx}{\int_{B_{\epsilon}(x_0)} w(x - x_0) \, \rho(x) \, dx}
$$

where $w(x - x_0) = \exp(-\|x - x_0\|^2 / (2\epsilon_c^2))$ is the annealed kernel weight.

For small $\epsilon$, this integral can be evaluated via Taylor expansion.

---

### Diffusion Tensor from Fokker-Planck

**Type:** Proposition
**Label:** `prop-diffusion-from-fokker-planck`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.3. Covariance Matrix Convergence to Inverse Metric](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `metric-tensor`, `proofs`, `rigor`

**Statement:**
The diffusion tensor $D(x)$ in the Fokker-Planck equation for spatial marginal is related to $\Sigma_{\text{reg}}$ by:

$$
D(x) = \Sigma_{\text{reg}}(x) \Sigma_{\text{reg}}(x)^T
$$

and satisfies $D(x) = g(x)^{-1}$ up to a constant factor.

This is the emergent metric definition from Chapter 08.

---

### Discrete-Continuous Symmetry Correspondence

**Type:** Proposition
**Label:** `prop-symmetry-correspondence-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 5.2. Symmetry Correspondence Table](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `causal-tree`, `fractal-set`, `proofs`, `rigor`, `su2-symmetry`, `symmetry`, `u1-symmetry`

**Statement:**
The following table establishes the correspondence between **discrete symmetries** of the Fractal Set and **continuous symmetries** of the emergent continuum theory:

| **Discrete (Fractal Set)** | **Continuous (Continuum Limit)** | **Physical Meaning** |
|----------------------------|----------------------------------|----------------------|
| Episode relabeling $S_{|\mathcal{E}|}$ | Particle permutation $S_N$ / Braid group $B_N$ | Walker indistinguishability |
| CST temporal ordering | Time translation $\mathbb{R}$ | Causality |
| IG spatial coupling | Diffeomorphisms Diff($\mathcal{X}$) | Coordinate freedom |
| Reward shift $R \to R + c$ | Global U(1)_fitness phase | Fitness scale invariance |
| Episode pair $(i, j)$ exchange | SU(2)_weak isospin rotation | Cloning role exchange |

**Key insight:** Discrete symmetries are **exact** on the Fractal Set, while continuous symmetries are **emergent** in the $N \to \infty$ limit.

**Source:** 13_B §6.3 Proposition 6.3.1 (symmetry correspondence table); extracted_mathematics_13B.md.

---

### CST as Causal Set

**Type:** Proposition
**Label:** `prop-cst-causal-set-axioms`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 6.2. CST Satisfies Causal Set Axioms](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `causal-tree`, `fractal-set`, `metric-tensor`, `proofs`, `rigor`

**Statement:**
The Causal Spacetime Tree (CST) of the Fractal Set satisfies the **axioms of causal set theory** (Bombelli, Lee, Meyer, Sorkin):

1. **Partial order:** The temporal ordering $e_i \prec e_j$ (episode $i$ precedes $j$) is:
   - Reflexive: $e \prec e$
   - Antisymmetric: $e_i \prec e_j$ and $e_j \prec e_i$ implies $i = j$
   - Transitive: $e_i \prec e_j \prec e_k$ implies $e_i \prec e_k$

2. **Local finiteness:** For any two episodes $e_i \prec e_k$, the set $\{e_j : e_i \prec e_j \prec e_k\}$ is finite.

3. **Manifoldlikeness:** In the continuum limit $N \to \infty$, the causal structure approximates that of a Lorentzian manifold.

**Consequence:** The CST can be interpreted as a **discrete spacetime** in the sense of causal set quantum gravity.

**Source:** 13_E §1 Proposition `prop-cst-satisfies-axioms`.

---

### Theorems (10)

### Graph Laplacian Convergence to Laplace-Beltrami

**Type:** Theorem
**Label:** `thm-graph-laplacian-convergence-complete`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.2. Main Convergence Theorem](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `convergence`, `fractal-set`, `laplacian`, `metric-tensor`, `proofs`, `qsd`, `riemannian`, `rigor`

**Statement:**
Let $\mathcal{F}_N$ be the Fractal Set with $N$ total nodes (episodes). Let $f_\phi: \mathcal{E}_N \to \mathbb{R}$ be a smooth test function on episodes induced by $\phi: \mathcal{X} \to \mathbb{R}$ via $f_\phi(e) = \phi(x_e)$ where $x_e$ is the spatial location of episode $e$.

Define the **graph Laplacian** on the Fractal Set as:

$$
(\Delta_{\mathcal{F}_N} f)(e_i) := \frac{1}{d_i} \sum_{e_j \sim e_i} w_{ij} \left( \frac{f(e_j) - f(e_i)}{\|x_j - x_i\|^2} \right)
$$

where:
- $e_j \sim e_i$ denotes IG neighbors (episodes connected by IG edges)
- $w_{ij}$ are IG edge weights (see §2.1)
- $d_i = \sum_{e_j \sim e_i} w_{ij}$ is the weighted degree

Let $(\mathcal{X}, g)$ be the emergent Riemannian manifold with metric $g$ and let $\mu$ be the QSD spatial marginal. Then as $N \to \infty$:

$$
\frac{1}{N} \sum_{e \in \mathcal{E}_N} (\Delta_{\mathcal{F}_N} f_\phi)(e) \xrightarrow{p} \int_{\mathcal{X}} (\Delta_g \phi)(x) \, d\mu(x)
$$

where $\Delta_g$ is the Laplace-Beltrami operator:

$$
\Delta_g \phi = \frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} \, g^{ij} \partial_j \phi \right)
$$

**Convergence rate:** With probability at least $1 - \delta$:

$$
\left| \frac{1}{N} \sum_{e \in \mathcal{E}_N} (\Delta_{\mathcal{F}_N} f_\phi)(e) - \int_{\mathcal{X}} (\Delta_g \phi)(x) \, d\mu(x) \right| \leq C(\phi) \cdot N^{-1/4} \log(1/\delta)
$$

for a constant $C(\phi)$ depending on smoothness of $\phi$.

**Source:** 13_B §3.2 Theorem 3.2.1 (`thm-graph-laplacian-convergence`), with complete proof provided by combining lemmas from discussion documents.

---

### QSD Spatial Marginal is Riemannian Volume Measure

**Type:** Theorem
**Label:** `thm-qsd-spatial-riemannian-volume`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.1. QSD Spatial Marginal Equals Riemannian Volume](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `metric-tensor`, `proofs`, `qsd`, `riemannian`, `rigor`, `volume`

**Statement:**
Consider the Adaptive Gas SDE from {doc}`../07_adaptative_gas.md` with state space $\mathcal{X} \times \mathbb{R}^d$ and quasi-stationary distribution $\pi_{\text{QSD}}$.

The **spatial marginal** of the QSD (integrating out velocities) is:

$$
\rho_{\text{spatial}}(x) = \int_{\mathbb{R}^d} \pi_{\text{QSD}}(x, v) \, dv \propto \sqrt{\det g(x)} \, \exp\left( -\frac{U_{\text{eff}}(x)}{T} \right)
$$

where:
- $g(x) = D(x)^{-1}$ is the emergent Riemannian metric (inverse diffusion tensor)
- $D(x) = \Sigma_{\text{reg}}(x) \Sigma_{\text{reg}}(x)^T$ is the position-dependent diffusion tensor
- $U_{\text{eff}}(x) = U(x) + T \log Z_{\text{kin}}$ is the effective potential (confining + entropic)
- $T = 1/\gamma$ is the effective temperature

**Critical insight:** The $\sqrt{\det g(x)}$ factor arises because the Langevin SDE in Chapter 07 uses **Stratonovich calculus**, not Itô calculus.

**Source:** `qsd_stratonovich_final.md` Theorem `thm-main-result-final` (Gemini validated as publication-ready).

---

### Local Covariance Converges to Inverse Metric Tensor

**Type:** Theorem
**Label:** `thm-covariance-convergence-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 1.3.3. Covariance Matrix Convergence to Inverse Metric](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `convergence`, `fractal-set`, `metric-tensor`, `proofs`, `riemannian`, `rigor`

**Statement:**
Let $e_i$ be an episode at position $x_i \in \mathcal{X}$. Define the **local covariance matrix** from IG edge displacements:

$$
\Sigma_i := \frac{\sum_{e_j \in \mathcal{N}_{\epsilon}(e_i)} w_{ij} \Delta x_{ij} \Delta x_{ij}^T}{\sum_{e_j \in \mathcal{N}_{\epsilon}(e_i)} w_{ij}}
$$

where:
- $\mathcal{N}_{\epsilon}(e_i) = \{e_j : \|x_j - x_i\| < \epsilon\}$ is the $\epsilon$-neighborhood
- $w_{ij}$ are IG edge weights (see §2.1)
- $\Delta x_{ij} = x_j - x_i$

Let $g(x_i)$ be the Riemannian metric at $x_i$ (inverse of diffusion tensor $D(x_i)$). Then as $N \to \infty$ and $\epsilon \to 0$ (with $N_{\text{local}} := |\\mathcal{N}_{\epsilon}(e_i)| \to \infty$):

$$
\Sigma_i \xrightarrow{a.s.} \epsilon^2 g(x_i)^{-1}
$$

**Convergence rate:** For bounded geometry assumptions:

$$
\mathbb{E}\left[ \|\Sigma_i - \epsilon^2 g(x_i)^{-1}\|_F \right] \leq C \left( \epsilon + N_{\text{local}}^{-1/2} \right)
$$

where $\|\cdot\|_F$ is the Frobenius norm.

**Source:** `covariance_convergence_rigorous_proof.md` Theorem `thm-covariance-convergence-rigorous` (complete 4-step proof).

---

### IG Edge Weights from Temporal Overlap

**Type:** Theorem
**Label:** `thm-ig-edge-weights-algorithmic`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 2.1. IG Edge Weights from Companion Selection Dynamics](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `proofs`, `rigor`

**Statement:**
Let $e_i$ and $e_j$ be two episodes with temporal overlap interval:

$$
T_{\text{overlap}}(i, j) = [t^b_{\max}, t^d_{\min}]
$$

where:
- $t^b_{\max} = \max(t^b_i, t^b_j)$: Later birth time
- $t^d_{\min} = \min(t^d_i, t^d_j)$: Earlier death time

Let $P(c_i(t) = j \mid i)$ be the probability that episode $i$ selects episode $j$ as its companion (diversity or cloning) at time $t \in T_{\text{overlap}}(i, j)$.

Then the **IG edge weight** between episodes $i$ and $j$ is:

$$
w_{ij} = \int_{T_{\text{overlap}}(i, j)} P(c_i(t) = j \mid i) \, dt
$$

**Interpretation:** The edge weight is the **expected number of selection events** between episodes $i$ and $j$ during their temporal overlap.

**Consequence:** This removes all arbitrariness from IG edge construction. The weights are **computed** from the algorithm, not **imposed** as hyperparameters.

**Source:** 13_B §3.3 Theorem 3.3.1 (`thm-ig-edge-weights-algorithmic`); 13_E §2.1b Theorem `thm-ig-edge-weights-from-companion-selection`.

---

### Weighted First Moment Encodes Christoffel Symbols

**Type:** Theorem
**Label:** `thm-weighted-first-moment-connection`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 2.2. Christoffel Symbols Emerge from Weighted Moments](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `metric-tensor`, `proofs`, `rigor`, `volume`

**Statement:**
Let $e_i$ be an episode at position $x_i$. Define the **weighted first moment** of IG edge displacements:

$$
M_i := \sum_{e_j \in \text{IG}(e_i)} w_{ij} \Delta x_{ij}
$$

where $\text{IG}(e_i)$ are IG neighbors of episode $i$ and $\Delta x_{ij} = x_j - x_i$.

Let $\varepsilon_c = \sqrt{T/\gamma}$ be the thermal coherence length (typical IG edge length). Then:

$$
M_i = \varepsilon_c^2 \cdot D_{\text{reg}}(x_i) \cdot \nabla \log \sqrt{\det g(x_i)} + O(\varepsilon_c^3)
$$

where:
- $D_{\text{reg}}(x_i) = g(x_i)^{-1}$ is the regularized diffusion tensor
- $\nabla \log \sqrt{\det g} = \frac{1}{2\sqrt{\det g}} \nabla(\det g) = \frac{1}{2} g^{ij} \partial_k g_{ij}$ is the **divergence of the metric connection**

**Physical interpretation:** The first moment encodes the **drift term** from the volume factor in the stationary distribution, which is precisely the Christoffel symbol of the second kind:

$$
\Gamma^i_{jj} = \frac{1}{2} g^{ik} \partial_j g_{kj}
$$

**Consequence:** Connection coefficients are **not put in by hand** but **emerge from the algorithm**.

**Source:** 13_B §3.4 Theorem 3.4.1 (`thm-weighted-first-moment-connection`).

---

### Discrete Permutation Invariance

**Type:** Theorem
**Label:** `thm-discrete-permutation-invariance-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.1. Episode Permutation Group as Gauge Symmetry](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `gauge-theory`, `metric-tensor`, `proofs`, `rigor`

**Statement:**
The probability distribution over Fractal Sets is **invariant** under episode relabeling:

$$
\mathcal{L}(\mathcal{F}) = \mathcal{L}(\sigma \cdot \mathcal{F}) \quad \forall \sigma \in S_{|\mathcal{E}|}
$$

where $\mathcal{L}(\mathcal{F})$ denotes the law of the random Fractal Set.

**Proof:** Episodes are labeled arbitrarily during simulation. The underlying stochastic process treats all episodes symmetrically - labels are bookkeeping, not physical.

**Consequence:** All **gauge-invariant observables** must be symmetric functions of episodes:

$$
O(\sigma \cdot \mathcal{F}) = O(\mathcal{F}) \quad \forall \sigma \in S_{|\mathcal{E}|}
$$

**Source:** 13_B §1.1 Theorem 1.1.2 (`thm-discrete-permutation-invariance`); extracted_mathematics_13B.md.

---

### Connection to Braid Holonomy

**Type:** Theorem
**Label:** `thm-connection-to-braid-holonomy-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.2. Discrete Parallel Transport and Connection](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `gauge-theory`, `lattice`, `proofs`, `rigor`

**Statement:**
The discrete gauge connection on the Fractal Set is **compatible** with the braid group holonomy from Chapter 12.

Specifically, for a closed path $\gamma$ in the Fractal Set, the discrete holonomy:

$$
\text{Hol}_{\mathcal{F}}(\gamma) = \mathcal{T}(\gamma) \in S_{|\mathcal{E}|}
$$

agrees with the braid holonomy projected to $S_N$ via the natural homomorphism $\rho: B_N \to S_N$.

**Consequence:** The Fractal Set discrete gauge structure is the **lattice regularization** of the continuous braid gauge theory.

**Source:** 13_B §2.1 Theorem 2.1.3 (`thm-connection-to-braid-holonomy`); extracted_mathematics_13B.md.

---

### IG Edges Close Fundamental Cycles

**Type:** Theorem
**Label:** `thm-ig-fundamental-cycles-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 3.3. Wilson Loops and Holonomy Observables](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `proofs`, `rigor`

**Statement:**
If the CST is a **tree** (single root, no cycles), then each IG edge $(e_i, e_j)$ closes **exactly one fundamental cycle**:

$$
C_{ij} = \text{CST path from root to } e_i + (e_i \to e_j)_{\text{IG}} + \text{CST path from } e_j \text{ back to root}
$$

**Consequence:** The number of independent Wilson loops equals the number of IG edges.

**Graph theory:** This is the standard result that in a tree with $V$ vertices and $E_{\text{tree}} = V-1$ edges, adding $E_{\text{IG}}$ edges creates $E_{\text{IG}}$ independent cycles (first Betti number).

**Source:** 13_D Part III §3 Theorem `thm-ig-fundamental-cycles`.

---

### Fan Triangulation for Riemannian Area

**Type:** Theorem
**Label:** `thm-fan-triangulation-area`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 4.1. Fan Triangulation Algorithm for Riemannian Area](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `area`, `fractal-set`, `metric-tensor`, `proofs`, `riemannian`, `rigor`

**Statement:**
Let $C = (e_1, e_2, \ldots, e_n, e_1)$ be a closed cycle in the Fractal Set with episodes at positions $x_1, \ldots, x_n \in \mathcal{X} \subset \mathbb{R}^d$.

**Algorithm: Fan Triangulation**

1. Choose a base episode $e_1$ (arbitrary)
2. Triangulate the cycle into $n-2$ triangles:
   $$T_k = (e_1, e_{k+1}, e_{k+2}) \quad \text{for } k = 1, \ldots, n-2$$

3. Compute the **Riemannian area** of each triangle using the metric $g$:
   $$A(T_k) = \frac{1}{2} \sqrt{\det\begin{pmatrix} g(v_{k+1}, v_{k+1}) & g(v_{k+1}, v_{k+2}) \\ g(v_{k+1}, v_{k+2}) & g(v_{k+2}, v_{k+2}) \end{pmatrix}}$$
   where $v_{k+1} = x_{k+1} - x_1$, $v_{k+2} = x_{k+2} - x_1$ are edge vectors, and $g(v, w) = v^T g(x_1) w$ is the metric inner product.

4. Sum the areas:
   $$A(C) = \sum_{k=1}^{n-2} A(T_k)$$

**Key property:** This area is **base-independent** (choice of $e_1$ doesn't matter) and equals the **Riemannian surface area** enclosed by the cycle.

**Source:** 13_D Part III Theorem `thm-comprehensive-fan-triangulation`.

---

### Discrete Translation Equivariance

**Type:** Theorem
**Label:** `thm-discrete-translation-equivariance-rigorous`
**Source:** [13_fractal_set_new/04_rigorous_additions.md § 5.1. Discrete Symmetry Theorems](13_fractal_set_new/04_rigorous_additions.md)
**Tags:** `fractal-set`, `proofs`, `rigor`

**Statement:**
If the reward function $R: \mathcal{X} \to \mathbb{R}$ is translation-invariant:

$$
R(x + a) = R(x) \quad \forall x \in \mathcal{X}, a \in \mathbb{R}^d
$$

then the Fractal Set distribution is **translation-equivariant**:

$$
\mathcal{L}(\mathcal{F}) = \mathcal{L}(T_a(\mathcal{F}))
$$

where $T_a(\mathcal{F})$ translates all episode positions by $a$.

**Source:** 13_B §1.2 Theorem 1.2.2 (`thm-discrete-translation-equivariance`); extracted_mathematics_13B.md.

---


## 05_qsd_stratonovich_foundations.md

**Objects in this document:** 12

### Definitions (1)

### Adaptive Gas Langevin SDE (Stratonovich Form)

**Type:** Definition
**Label:** `def-adaptive-gas-stratonovich-sde`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 2.1. Primary Formulation from Chapter 07](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`, `metric-tensor`

**Statement:**
From {prf:ref}`def-adaptive-sde` in {doc}`../07_adaptative_gas.md`, the Adaptive Gas dynamics on phase space $(x, v) \in \mathcal{X} \times \mathbb{R}^d$ is:

$$
\begin{aligned}
dx_i &= v_i \, dt \\
dv_i &= F_{\text{total}}(x_i, \mathcal{S}_t) \, dt - \gamma v_i \, dt + \Sigma_{\text{reg}}(x_i) \circ dW_i
\end{aligned}
$$

where:

**Drift terms:**
- $F_{\text{total}}(x, \mathcal{S}) = F_{\text{conf}}(x) + F_{\text{adap}}(x, \mathcal{S})$ is the total force
- $F_{\text{conf}}(x) = -\nabla U(x)$ from confining potential
- $F_{\text{adap}}(x, \mathcal{S}) = \epsilon_F \nabla V_{\text{fit}}[f_k, \rho](x)$ from adaptive mean-field fitness potential
- Combined: $F_{\text{total}} = -\nabla U_{\text{eff}}$ where $U_{\text{eff}}(x) = U(x) - \epsilon_F V_{\text{fit}}(x)$
- $\gamma v_i$ is friction (Stokes drag)

**Diffusion term:**
- $\Sigma_{\text{reg}}(x) = (H(x) + \epsilon_\Sigma I)^{-1/2}$ is the **regularized Hessian square root**
- $H(x) = -\nabla^2 \Phi(x)$ is the negative fitness Hessian
- $\epsilon_\Sigma > 0$ is regularization ensuring positive definiteness
- **Metric:** $g(x) = \Sigma_{\text{reg}}^{-2}(x) = H(x) + \epsilon_\Sigma I$

**Critical notation:** The **$\circ dW_i$** denotes **Stratonovich stochastic integral**, not Itô.

**Noise strength:** Implicit in $\Sigma_{\text{reg}}$ normalization is temperature $T = \sigma^2/(2\gamma)$ from friction-diffusion balance.

**Related Results:** `def-adaptive-sde`

---

### Propositions (3)

### Stationary Solution Satisfies Stratonovich Fokker-Planck

**Type:** Proposition
**Label:** `prop-stationary-verification-stratonovich`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 5.2. Direct Verification via Fokker-Planck](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`

**Statement:**
The distribution $\rho(x) = C \sqrt{\det g(x)} \exp(-U_{\text{eff}}/T)$ satisfies the **stationary condition** for the Stratonovich Fokker-Planck equation:

$$
0 = -\nabla \cdot (b \rho) + \frac{1}{2} \nabla \cdot (D \nabla \rho) - \frac{1}{4} \nabla \cdot (D \nabla \log \det D \cdot \rho)
$$

where $b = -D \nabla (U_{\text{eff}}/T)$ and $D = (T/\gamma) g^{-1}$.

---

### QSD Spatial Marginal for Non-Equilibrium System

**Type:** Proposition
**Label:** `prop-qsd-spatial-nonequilibrium`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § Appendix A: Timescale Separation and QSD Spatial Marginal](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`, `qsd`

**Statement:**
Consider the full Adaptive Gas dynamics with cloning and death operators. Despite the system being **non-equilibrium** and **non-reversible**, the spatial marginal of the quasi-stationary distribution (QSD) is given by:

$$
\rho_{\text{spatial}}(x) = \int \rho_\infty(x, v) \, dv = C \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

where $\rho_\infty(x, v)$ is the full phase-space QSD.

**Key insight:** Even though the full system violates detailed balance, the **spatial marginal** follows the Stratonovich stationary distribution formula because of timescale separation.

---

### Stationary Distribution of Effective Spatial SDE

**Type:** Proposition
**Label:** `prop-spatial-sde-stationary`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § Appendix B: Effective Spatial Dynamics via Kramers-Smoluchowski Reduction](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`

**Statement:**
The stationary distribution of the effective spatial SDE (Theorem {prf:ref}`thm-kramers-smoluchowski-standard`) is:

$$
\rho_{\text{st}}(x) = \frac{1}{Z} \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

where $Z$ is the normalization constant.

**Related Results:** `thm-kramers-smoluchowski-standard`

---

### Theorems (8)

### QSD Spatial Marginal Equals Riemannian Volume Measure

**Type:** Theorem
**Label:** `thm-qsd-riemannian-volume-main`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 1.1. Statement of Main Theorem](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`, `metric-tensor`, `qsd`, `riemannian`, `volume`

**Statement:**
Let $\pi_{\text{QSD}}(x, v)$ be the quasi-stationary distribution of the Adaptive Gas on the alive set $\mathcal{A} = \mathcal{X} \times \mathbb{R}^d$. The **spatial marginal** (integrating out velocities) is:

$$
\rho_{\text{spatial}}(x) = \int_{\mathbb{R}^d} \pi_{\text{QSD}}(x, v) \, dv = \frac{1}{Z} \sqrt{\det g(x)} \, \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

where:
- $g(x) = (H(x) + \epsilon_\Sigma I)$ is the **emergent Riemannian metric** tensor from {prf:ref}`def-regularized-hessian-tensor` in {doc}`../08_emergent_geometry.md`
- $U_{\text{eff}}(x) = U(x) - \epsilon_F V_{\text{fit}}(x)$ is the **effective potential** combining confinement $U$ and adaptive force virtual reward $V_{\text{fit}}$
- $T = \sigma^2/(2\gamma)$ is the **effective temperature** from friction-diffusion balance
- $Z$ is the normalization constant ensuring $\int_{\mathcal{X}} \rho_{\text{spatial}}(x) \, dx = 1$

**Geometric interpretation:** Episodes sample from the **canonical Gibbs measure with respect to the Riemannian volume element**:

$$
d\mu_{\text{Riem}}(x) = \sqrt{\det g(x)} \, dx
$$

This is the natural volume measure on the emergent Riemannian manifold $(\mathcal{X}, g)$.

**Source:** `docs/source/13_fractal_set_old/discussions/qsd_stratonovich_final.md` Theorem `thm-main-result-final` (Gemini validated).

**Related Results:** `def-regularized-hessian-tensor`

---

### Timescale Separation in Overdamped Limit

**Type:** Theorem
**Label:** `thm-timescale-separation-overdamped`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 3.1. High-Friction Limit and Timescale Separation](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`

**Statement:**
Consider the Stratonovich Langevin system {prf:ref}`def-adaptive-gas-stratonovich-sde` in the regime where the friction coefficient is large: $\gamma \gg 1$.

There is a **timescale separation** between velocity and position dynamics:

$$
\tau_v \ll \tau_x
$$

where:
- $\tau_v \sim \gamma^{-1}$: Velocity thermalization time (fast)
- $\tau_x \sim \ell^2 \gamma / T$: Spatial diffusion time over length scale $\ell$ (slow)

**Consequence:** On timescales $t \gg \gamma^{-1}$, velocities have equilibrated to a **quasi-equilibrium** Maxwell-Boltzmann distribution at each position $x$, and the spatial dynamics can be described by an **effective overdamped Langevin equation** (Kramers-Smoluchowski limit).

**Source:** Standard result in stochastic process theory. See Pavliotis (2014) *Stochastic Processes and Applications*, Chapter 7; Pavliotis & Stuart (2008) *Multiscale Methods*, Chapters 6-7.

**Related Results:** `def-adaptive-gas-stratonovich-sde`

---

### Kramers-Smoluchowski Reduction in Stratonovich Form

**Type:** Theorem
**Label:** `thm-stratonovich-kramers-smoluchowski`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 3.2. Effective Spatial Stratonovich SDE](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`

**Statement:**
In the high-friction limit $\gamma \gg 1$, the Stratonovich Langevin system {prf:ref}`def-adaptive-gas-stratonovich-sde` reduces to an **effective Stratonovich SDE** for the spatial marginal:

$$
dx = b_{\text{eff}}(x) \, dt + \sigma_{\text{eff}}(x) \circ dW_t^{\text{spatial}}
$$

where:

$$
b_{\text{eff}}(x) = \frac{1}{\gamma} F_{\text{total}}(x) = -\frac{1}{\gamma} \nabla U_{\text{eff}}(x)
$$

$$
\sigma_{\text{eff}}(x) = \sqrt{\frac{2T}{\gamma}} \Sigma_{\text{reg}}(x) = \sqrt{\frac{2T}{\gamma}} g(x)^{-1/2}
$$

**Diffusion tensor:** $D(x) = \frac{1}{2} \sigma_{\text{eff}} \sigma_{\text{eff}}^T = \frac{T}{\gamma} g(x)^{-1}$

**Critical property:** This SDE is **Stratonovich** (symbol $\circ$), preserving the interpretation from the original Langevin equation.

**Source:** Standard result. See Graham (1977) Z. Physik B **26**, 397; Pavliotis (2014) Chapter 7.

**Related Results:** `def-adaptive-gas-stratonovich-sde`

---

### Stratonovich Stationary Distribution with State-Dependent Diffusion

**Type:** Theorem
**Label:** `thm-stratonovich-stationary-general`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 4.1. General Theorem (Graham 1977)](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`, `metric-tensor`, `riemannian`, `volume`

**Statement:**
Consider a **Stratonovich SDE** on $\mathbb{R}^d$:

$$
dx = b(x) \, dt + \sigma(x) \circ dW
$$

with:
- Drift: $b(x) = -D(x) \nabla U(x)$ (gradient flow)
- Diffusion tensor: $D(x) = \frac{1}{2} \sigma(x) \sigma(x)^T$ (symmetric, positive definite)
- Potential: $U: \mathbb{R}^d \to \mathbb{R}$ (smooth, grows at infinity)

Assume:
1. **Detailed balance:** The system is in thermal equilibrium (no external driving)
2. **Ergodicity:** The SDE admits a unique stationary distribution
3. **Integrability:** $\int_{\mathbb{R}^d} (\det D(x))^{-1/2} e^{-U(x)} dx < \infty$

Then the **stationary distribution** is:

$$
\rho_{\text{st}}(x) = \frac{1}{Z} \frac{1}{\sqrt{\det D(x)}} \, \exp(-U(x))
$$

where $Z = \int_{\mathbb{R}^d} (\det D)^{-1/2} e^{-U} dx$ is the normalization constant.

**Geometric form:** Define the **metric tensor** $g(x) := D(x)^{-1}$. Then:

$$
\boxed{\rho_{\text{st}}(x) = \frac{1}{Z} \sqrt{\det g(x)} \, \exp(-U(x))}
$$

This is the **Gibbs measure with respect to the Riemannian volume element** $dV_g = \sqrt{\det g(x)} \, dx$.

**Source:** Graham, R. (1977) "Covariant formulation of non-equilibrium statistical thermodynamics", *Zeitschrift für Physik B* **26**, 397-405, Equation (3.13).

---

### QSD Spatial Marginal is Riemannian Volume (Detailed Proof)

**Type:** Theorem
**Label:** `thm-qsd-spatial-marginal-detailed`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 5.1. Main Theorem Proof](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`, `qsd`, `riemannian`, `volume`

**Statement:**
The spatial marginal of the Adaptive Gas quasi-stationary distribution satisfies:

$$
\rho_{\text{spatial}}(x) = \frac{1}{Z} \sqrt{\det g(x)} \, \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

where $g(x) = H(x) + \epsilon_\Sigma I$, $U_{\text{eff}}(x) = U(x) - \epsilon_F V_{\text{fit}}(x)$, and $T = \sigma^2/(2\gamma)$.

---

### Graph Laplacian Converges to Laplace-Beltrami

**Type:** Theorem
**Label:** `thm-graph-laplacian-convergence-consequence`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 6.1. Belkin-Niyogi Theorem Application](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `convergence`, `fractal-set`, `laplacian`, `qsd`, `riemannian`

**Statement:**
Let episodes $\{x_i\}_{i=1}^N$ be sampled from $\rho_{\text{spatial}}(x) \propto \sqrt{\det g(x)} e^{-U_{\text{eff}}/T}$ (Theorem {prf:ref}`thm-qsd-spatial-marginal-detailed`).

Define the **companion-weighted graph Laplacian**:

$$
(\Delta_{\text{graph}} f)(x_i) = \frac{1}{d_i} \sum_{j: \text{IG neighbor}} w_{ij} \frac{f(x_j) - f(x_i)}{\|x_j - x_i\|^2}
$$

where $w_{ij}$ are IG edge weights and $d_i = \sum_j w_{ij}$ is weighted degree.

Then as $N \to \infty$:

$$
\frac{1}{N} \sum_{i=1}^N (\Delta_{\text{graph}} f)(x_i) \xrightarrow{p} C \int_{\mathcal{X}} (\Delta_g f)(x) \, \rho_{\text{spatial}}(x) \, dx
$$

where $\Delta_g$ is the **Laplace-Beltrami operator** on the Riemannian manifold $(\mathcal{X}, g)$:

$$
\Delta_g f = \frac{1}{\sqrt{\det g}} \partial_i \left( \sqrt{\det g} \, g^{ij} \partial_j f \right)
$$

**Convergence rate:** With probability $\geq 1 - \delta$:

$$
\left| \frac{1}{N} \sum_i \Delta_{\text{graph}} f(x_i) - C \int \Delta_g f \, \rho \, dx \right| \leq O(N^{-1/4} \log(1/\delta))
$$

**Source:** Belkin, M. & Niyogi, P. (2006) "Convergence of Laplacian Eigenmaps", *NIPS*; combined with Theorem {prf:ref}`thm-qsd-spatial-marginal-detailed`.

**Related Results:** `thm-qsd-spatial-marginal-detailed`

---

### Complete Summary - Episodes Sample Riemannian Volume

**Type:** Theorem
**Label:** `thm-complete-summary-riemannian-sampling`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § 7.1. Main Results Proven](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `convergence`, `fractal-set`, `laplacian`, `qsd`, `riemannian`, `volume`

**Statement:**
The following statements have been proven rigorously:

**1. Stratonovich formulation (Chapter 07)**

The Adaptive Gas Langevin SDE uses Stratonovich interpretation $\circ dW$, which is physically correct for state-dependent diffusion from fast degrees of freedom.

**2. Kramers-Smoluchowski reduction (Theorem {prf:ref}`thm-stratonovich-kramers-smoluchowski`)**

In high-friction limit $\gamma \gg 1$, spatial dynamics reduces to:

$$
dx = -\frac{1}{\gamma} \nabla U_{\text{eff}} \, dt + \sqrt{\frac{2T}{\gamma}} g^{-1/2} \circ dW
$$

**3. Stationary distribution (Graham 1977, Theorem {prf:ref}`thm-stratonovich-stationary-general`)**

The Stratonovich stationary distribution is:

$$
\rho_{\text{st}} \propto (\det D)^{-1/2} e^{-U} = \sqrt{\det g} \, e^{-U}
$$

**4. QSD spatial marginal (Theorem {prf:ref}`thm-qsd-spatial-marginal-detailed`)**

$$
\boxed{\rho_{\text{spatial}}(x) \propto \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)}
$$

**5. Graph Laplacian convergence (Theorem {prf:ref}`thm-graph-laplacian-convergence-consequence`)**

$$
\Delta_{\text{graph}} f \xrightarrow{N \to \infty} C \Delta_g f
$$

**Chain of implications:**

Stratonovich SDE → Kramers-Smoluchowski → Graham's theorem → $\rho \propto \sqrt{\det g}$ → Belkin-Niyogi → $\Delta_{\text{graph}} \to \Delta_g$

**Status:** All steps rigorously proven. Publication-ready.

**Related Results:** `thm-stratonovich-kramers-smoluchowski`, `thm-qsd-spatial-marginal-detailed`, `thm-graph-laplacian-convergence-consequence`, `thm-stratonovich-stationary-general`

---

### Effective Spatial SDE in High-Friction Limit

**Type:** Theorem
**Label:** `thm-kramers-smoluchowski-standard`
**Source:** [13_fractal_set_new/05_qsd_stratonovich_foundations.md § Appendix B: Effective Spatial Dynamics via Kramers-Smoluchowski Reduction](13_fractal_set_new/05_qsd_stratonovich_foundations.md)
**Tags:** `fractal-set`

**Statement:**
Consider the Stratonovich Langevin system:

$$
\begin{aligned}
dx &= v \, dt \\
dv &= [-\nabla U_{\text{eff}}(x) - \gamma v] dt + \sqrt{2\gamma T} \, g(x)^{-1/2} \circ dW
\end{aligned}
$$

In the high-friction limit $\gamma \gg 1$, the effective spatial dynamics is governed by the **Stratonovich SDE**:

$$
dx = -\frac{1}{\gamma} \nabla U_{\text{eff}} \, dt + \sqrt{\frac{2T}{\gamma}} g(x)^{-1/2} \circ dW + O(\gamma^{-2})
$$

The spatial marginal density $\rho_s(x, t) := \int \rho(x, v, t) dv$ satisfies the **spatial Fokker-Planck equation**:

$$
\frac{\partial \rho_s}{\partial t} = \nabla \cdot \left[\frac{T}{\gamma} g(x)^{-1} \nabla \rho_s + \frac{1}{\gamma} \rho_s \nabla U_{\text{eff}}\right] + O(\gamma^{-2})
$$

**Critical feature:** The Stratonovich interpretation is **preserved** in the high-friction limit.

---


## 06_continuum_limit_theory.md

**Objects in this document:** 12

### Lemmas (3)

### QSD Marginal Equals Riemannian Volume Measure

**Type:** Lemma
**Label:** `lem-qsd-riemannian-volume`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 2.1. QSD = Riemannian Volume](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`, `metric-tensor`, `qsd`, `riemannian`, `volume`

**Statement:**
Let $\rho_\infty(x, v)$ be the quasi-stationary distribution (QSD) of the Adaptive Gas with emergent metric $g(x) = H(x) + \epsilon_\Sigma I$.

The spatial marginal density is:

$$
\rho_{\text{spatial}}(x) := \int \rho_\infty(x, v) \, dv = C \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right)
$$

where:
- $g(x) = H(x) + \epsilon_\Sigma I$ is the emergent Riemannian metric
- $U_{\text{eff}}(x) = U(x) - \epsilon_F V_{\text{fit}}(x)$ is the effective potential
- $T = \sigma^2/(2\gamma)$ is the effective temperature
- $C$ is a normalization constant
- $\sqrt{\det g(x)}$ is the **Riemannian volume element**

**Key insight**: The $\sqrt{\det g(x)}$ factor arises because the Langevin dynamics uses **Stratonovich calculus**, which preserves Riemannian geometric structure.

---

### Timescale Separation and Annealed Approximation

**Type:** Lemma
**Label:** `lem-velocity-marginalization`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 2.2. Velocity Marginalization and Annealed Approximation](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`

**Statement:**
The Euclidean Gas dynamics with Langevin kinetic operator exhibits timescale separation:

**Fast timescale** (velocity relaxation): $\tau_v = O(\gamma^{-1})$

**Slow timescale** (spatial diffusion): $\tau_x = O(L^2 D^{-1})$ where $L$ is domain size and $D \sim \gamma^{-1}$ is diffusion coefficient

**Separation ratio**: $\frac{\tau_x}{\tau_v} = O(\gamma L^2) \gg 1$ for $\gamma L^2 \gg 1$

**Consequence**: For spatial observables (e.g., graph Laplacian), velocities can be replaced by their equilibrium distribution:

$$
\langle A(x, v) \rangle_{\text{quenched}} \approx \langle A(x, v) \rangle_{\text{annealed}} = \int A(x, v) \mathcal{M}_\gamma(v) \, dv
$$

where $\mathcal{M}_\gamma(v) = (2\pi \gamma^{-1})^{-d/2} \exp(-\gamma \|v\|^2 / 2)$ is the Maxwellian velocity distribution.

**Error bound**: The annealing error is $O(e^{-c \gamma t})$ for $t \gg \gamma^{-1}$, where $c > 0$ depends on the spectral gap of the kinetic operator.

---

### Convergence of Discrete Covariance to Inverse Metric

**Type:** Lemma
**Label:** `lem-covariance-convergence`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 2.3. Covariance Matrix Convergence](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `convergence`, `fractal-set`, `laplacian`, `metric-tensor`, `qsd`

**Statement:**
Let $\{e_i\}_{i=1}^{N_{\text{epi}}}$ be the episodes generated by the Adaptive Gas algorithm with $N$ walkers over time $[0, T]$. Assume:

1. **QSD convergence**: Walker density converges to QSD with rate $\|μ_N - μ_{QSD}\|_{TV} = O(N^{-1/4})$
2. **Regularity**: Metric $g(x) = H(x) + \epsilon_\Sigma I$ is $C^2$ with $\lambda_{\min}(g) \geq c_0 > 0$ uniformly
3. **Localization**: Companion bandwidth $\epsilon \ll 1$ satisfies $\epsilon = O(N^{-1/(2d)})$ (optimal)

Fix an episode $e_i$ at position $x_i = \Phi(e_i)$ sampled from the QSD. Define the local covariance matrix:

$$
\Sigma_i := \frac{\sum_{e_j \in \mathcal{N}_{\epsilon}(e_i)} w_{ij} \Delta x_{ij} \Delta x_{ij}^T}{\sum_{e_j \in \mathcal{N}_{\epsilon}(e_i)} w_{ij}}
$$

where $\Delta x_{ij} = \Phi(e_j) - \Phi(e_i)$, $\mathcal{N}_{\epsilon}(e_i)$ is the local neighborhood of episode $e_i$, and $N_{\text{local}} := |\mathcal{N}_{\epsilon}(e_i)|$ is the number of episodes in the local neighborhood.

Then almost surely as $N \to \infty$:

$$
\Sigma_i \xrightarrow{a.s.} \epsilon^2 g(x_i)^{-1}
$$

with convergence rate:

$$
\mathbb{E}[\|\Sigma_i - \epsilon^2 g(x_i)^{-1}\|_F] \leq C \left( \epsilon + \frac{1}{\sqrt{N_{\text{local}}}} \right)
$$

where $\|\cdot\|_F$ is the Frobenius norm.

---

### Theorems (6)

### Graph Laplacian Converges to Laplace-Beltrami Operator

**Type:** Theorem
**Label:** `thm-graph-laplacian-convergence`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 1. Main Convergence Theorem](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `convergence`, `fractal-set`, `laplacian`, `metric-tensor`, `qsd`, `riemannian`

**Statement:**
Let $\mathcal{F}_N$ be the Fractal Set generated by $N$ walkers over time $[0, T]$. Let $(\mathcal{X}, g)$ be the emergent Riemannian manifold with metric:

$$
g(x) = \mathbb{E}[H(x, S) + \epsilon_\Sigma I]^{-1}
$$

time-averaged over the quasi-stationary distribution (QSD).

For any smooth function $\phi \in C^2(\mathcal{X})$, define $f_\phi(e) = \phi(\Phi(e))$ where $\Phi(e)$ is the episode spatial embedding (death position).

Then, for a suitable normalization of the graph Laplacian $\tilde{\Delta}_{\mathcal{F}_N}$, as $N \to \infty$:

$$
\tilde{\Delta}_{\mathcal{F}_N} f_\phi \to \Delta_g \phi
$$

where the convergence is in $L^2(\mathcal{X}, d\mu)$ and:
- $\tilde{\Delta}_{\mathcal{F}_N}$: Normalized Fractal Set graph Laplacian (Definition 3.1.1)
- $\Delta_g$: Laplace-Beltrami operator on $(\mathcal{X}, g)$
- $d\mu$: Quasi-stationary distribution measure

**Convergence rate**: Under regularity conditions (Assumption 3.2.2),

$$
\left\| \tilde{\Delta}_{\mathcal{F}_N} f_\phi - \Delta_g \phi \right\|_{L^2(\mu)} \leq C(\mathcal{X}, g, T) \|\phi\|_{C^2} \cdot N^{-1/4}
$$

with probability at least $1 - \delta$ for $C = C(\delta, T, \mathcal{X})$.

---

### Convergence of Graph Laplacian to Generator (Belkin-Niyogi 2006)

**Type:** Theorem
**Label:** `thm-belkin-niyogi-convergence`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 3.1. Standard Graph Laplacian Convergence Theorem](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `convergence`, `fractal-set`, `laplacian`, `riemannian`, `volume`

**Statement:**
Let $\{x_i\}_{i=1}^N$ be i.i.d. samples from a probability measure $p(x) dx$ on a compact Riemannian manifold $(M, g)$. Define the graph Laplacian:

$$
(L_N f)_i = \frac{1}{N \epsilon^{d+2}} \sum_{j=1}^N K_\epsilon(x_i, x_j) (f_j - f_i)
$$

where $K_\epsilon(x, y) = \exp(-\|x - y\|^2 / (4\epsilon^2))$ is a Gaussian kernel.

Then as $N \to \infty$ and $\epsilon \to 0$ with $N \epsilon^d \to \infty$:

$$
L_N f \xrightarrow{P} \frac{1}{2p(x)} \nabla \cdot (p(x) \nabla f) + O(\epsilon^2)
$$

pointwise in probability.

**Riemannian case**: If $p(x) = \sqrt{\det g(x)} q(x)$ for some smooth $q(x) > 0$, then:

$$
L_N f \to \frac{1}{2q(x)} \left[\frac{1}{\sqrt{\det g}} \nabla \cdot (\sqrt{\det g} \, \nabla f)\right] + O(\epsilon^2)
$$

For $q(x) = 1$ (sampling according to Riemannian volume), this is:

$$
L_N f \to \frac{1}{2} \Delta_g f
$$

where $\Delta_g$ is the Laplace-Beltrami operator.

---

### Graph Laplacian Converges to Laplace-Beltrami for Fractal Set

**Type:** Theorem
**Label:** `thm-fractal-set-laplacian-convergence`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 3.2. Application to Fractal Set Episodes](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`

**Statement:**
Let $\{e_i\}$ be episodes generated by the Euclidean Gas with $N$ walkers. Define the spatial graph Laplacian using the annealed kernel (Corollary {prf:ref}`cor-annealed-kernel`):

$$
(L_{\text{spatial}} f)(x_i) = \frac{1}{N_{\text{local}} \epsilon^{d+2}} \sum_{j \in \mathcal{N}_\epsilon(i)} W(x_i, x_j) (f(x_j) - f(x_i))
$$

where $x_i = \Phi(e_i)$ are episode positions.

Then as $N \to \infty$ and $\epsilon \to 0$:

$$
L_{\text{spatial}} f \xrightarrow{a.s.} \frac{C_v}{2} \Delta_g f
$$

where:
- $\Delta_g$ is the Laplace-Beltrami operator on $(\mathcal{X}, g)$ with $g(x) = H(x) + \epsilon_\Sigma I$
- $C_v$ is the velocity prefactor from Corollary {prf:ref}`cor-annealed-kernel` (can be absorbed into operator normalization)

**Related Results:** `cor-annealed-kernel`

---

### IG Edge Weights from Companion Selection

**Type:** Theorem
**Label:** `thm-ig-edge-weights-algorithmic`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 4.1. IG Edge Weights from Companion Selection Dynamics](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`

**Statement:**
For episodes $e_i$ and $e_j$ with temporal overlap period $T_{\text{overlap}}(i,j)$, the IG edge weight is the time-integrated companion selection probability:

$$
w_{ij} = \int_{T_{\text{overlap}}(i,j)} P(c_i(t) = j \mid i \in \mathcal{A}(t)) \, dt
$$

where the companion selection probability is:

$$
P(c_i(t) = j \mid i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,j;t)^2}{2\varepsilon_c^2}\right)}{Z_i(t)}
$$

with:
- **Algorithmic distance**: $d_{\text{alg}}(i,j)^2 = \|x_i(t) - x_j(t)\|^2 + \lambda_v \|v_i(t) - v_j(t)\|^2$
- **Partition function**: $Z_i(t) = \sum_{l \in \mathcal{A}(t) \setminus \{i\}} \exp(-d_{\text{alg}}(i,l;t)^2 / 2\varepsilon_c^2)$

**Discrete form**:

$$
w_{ij} \approx \tau \sum_{t_k \in T_{\text{overlap}}(i,j)} \frac{\exp\left(-d_{\text{alg}}(i,j; t_k)^2 / 2\varepsilon_c^2\right)}{Z_i(t_k)}
$$

**Conclusion**: Edge weights are **algorithmically determined**, not design choices.

---

### Weighted First Moment and Connection Term

**Type:** Theorem
**Label:** `thm-weighted-first-moment-connection`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 4.2. Christoffel Symbols Emerge from Weighted First Moment](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`, `metric-tensor`, `qsd`, `riemannian`, `volume`

**Statement:**
Let $g(x) = H(x) + \varepsilon_{\Sigma} I$ be the emergent Riemannian metric.

For an episode $e_i$ at position $x_i = \Phi(e_i)$ in a swarm state converged to QSD, the weighted first moment of the IG edge distribution is:

$$
\sum_{e_j \in \text{IG}(e_i)} w_{ij} \Delta x_{ij} = \varepsilon_c^2 D_{\text{reg}}(x_i) \nabla \log \sqrt{\det g(x_i)} + O(\varepsilon_c^3)
$$

where:
- $\Delta x_{ij} = \Phi(e_j) - \Phi(e_i)$: Spatial separation
- $D_{\text{reg}}(x_i) = g(x_i)^{-1}$: Diffusion tensor (inverse metric)
- $\varepsilon_c$: Companion selection bandwidth

**Physical interpretation**: The drift term $\nabla \log \sqrt{\det g}$ is the **connection term** (Christoffel symbols) in the Laplace-Beltrami operator, arising from **volume measure variation** on the Riemannian manifold.

---

### Explicit Convergence Rate

**Type:** Theorem
**Label:** `thm-explicit-convergence-rate`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 5.2. Convergence Rate Summary](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `convergence`, `fractal-set`, `laplacian`, `qsd`

**Statement:**
Under regularity conditions (Assumption 3.2.2), for any smooth function $\phi \in C^2(\mathcal{X})$:

$$
\left\| \Delta_{\mathcal{F}_N} f_\phi - \Delta_g \phi \right\|_{L^2(\mu)} \leq C(\mathcal{X}, g, T) \|\phi\|_{C^2} \cdot N^{-1/4}
$$

with probability at least $1 - \delta$ for $C = C(\delta)$.

**Component-wise breakdown**:

| **Error Source** | **Rate** | **Condition** |
|------------------|----------|---------------|
| QSD convergence | $O(N^{-1/4})$ | Hypocoercivity (Ch 11) |
| Finite sampling | $O(N^{-1/4})$ | Optimal bandwidth $\epsilon \sim N^{-1/(2d)}$ |
| Bandwidth bias | $O(\epsilon^2) = O(N^{-1/d})$ | Smoothness assumption |
| Velocity thermalization | $O(e^{-c\gamma t})$ | Fast friction limit |

**Dominant term**: For $d \geq 2$, the convergence rate is $O(N^{-1/4})$, dominated by **both** QSD convergence and finite-sample discretization error.

---

### Corollarys (3)

### Annealed Companion Kernel

**Type:** Corollary
**Label:** `cor-annealed-kernel`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 2.2. Velocity Marginalization and Annealed Approximation](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`

**Statement:**
For episodes $e_i$ and $e_j$ at positions $x_i, x_j$ with velocities drawn from the equilibrium distribution $\mathcal{M}_\gamma(v)$, the effective companion kernel is:

$$
W(x_i, x_j) := \mathbb{E}_{v_i, v_j \sim \mathcal{M}_\gamma}[w_{ij}] = C_v(\lambda_v, \epsilon, \gamma) \cdot \exp\left(-\frac{\|x_i - x_j\|^2}{2\epsilon^2}\right)
$$

where the velocity prefactor is:

$$
C_v = \left(1 + \frac{2\lambda_v}{\gamma \epsilon^2}\right)^{-d/2}
$$

**Key insight**: The velocity term $\lambda_v \|v_i - v_j\|^2$ integrates out to a constant prefactor, leaving a **purely spatial Gaussian kernel**.

---

### Exponential Decay of Edge Weights

**Type:** Corollary
**Label:** `cor-edge-weight-decay`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 4.1. IG Edge Weights from Companion Selection Dynamics](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`

**Statement:**
For episodes $e_i, e_j$ with spatial separation $\|\Phi(e_i) - \Phi(e_j)\| = r$ large compared to $\varepsilon_c$:

$$
w_{ij} \lesssim T_{\text{overlap}} \cdot \exp\left(-\frac{r^2}{2\varepsilon_c^2}\right)
$$

**Implication**: The IG is **sparse**—only episodes within interaction range $O(\varepsilon_c)$ in phase space have significant edge weights.

---

### Christoffel Symbols from Algorithmic Dynamics

**Type:** Corollary
**Label:** `cor-christoffel-from-algorithm`
**Source:** [13_fractal_set_new/06_continuum_limit_theory.md § 4.2. Christoffel Symbols Emerge from Weighted First Moment](13_fractal_set_new/06_continuum_limit_theory.md)
**Tags:** `continuum-limit`, `fractal-set`, `laplacian`, `metric-tensor`, `qsd`, `volume`

**Statement:**
The Christoffel symbols $\Gamma^{\lambda}_{\mu\nu}$ of the emergent metric $g(x)$ satisfy:

$$
\Gamma^{\lambda}_{\mu\nu} g^{\nu\rho} = \partial_{\mu} \log \sqrt{\det g}
$$

By Theorem {prf:ref}`thm-weighted-first-moment-connection`, these symbols emerge from the **algorithmic companion selection dynamics** through the weighted first moment of edge distributions. They are **not imposed externally** but are **intrinsic consequences** of the QSD volume measure.

**Related Results:** `thm-weighted-first-moment-connection`

---


## 07_discrete_symmetries_gauge.md

**Objects in this document:** 18

### Definitions (9)

### Episode Relabeling Group

**Type:** Definition
**Label:** `def-episode-relabeling-group-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.1. Episode Permutation Group $S_{|\mathcal{E}|}$](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `metric-tensor`

**Statement:**
Let $\mathcal{E}$ be the set of all episodes generated by the Adaptive Gas algorithm. The **episode relabeling group** is:

$$
G_{\text{epi}} = \{\sigma : \mathcal{E} \to \mathcal{E} \mid \sigma \text{ is a bijection}\}
$$

This is the **symmetric group** on $\mathcal{E}$:

$$
G_{\text{epi}} \cong S_{|\mathcal{E}|}
$$

**Action on Fractal Set structures**:

1. **CST edges**: $\sigma(e_i \to e_j) := \sigma(e_i) \to \sigma(e_j)$
2. **IG edges**: $\sigma(e_i \sim e_j) := \sigma(e_i) \sim \sigma(e_j)$
3. **Episode durations**: $\tau_{\sigma(e)} = \tau_e$ (intrinsic property)
4. **Spatial embedding**: $\Phi(\sigma(e)) = \Phi(e)$ (canonical choice: death position)

**Physical interpretation**: Episode labels (ID numbers) are arbitrary bookkeeping. The genealogical and interaction structure is label-independent.

---

### Spatial Transformation of Fractal Set

**Type:** Definition
**Label:** `def-spatial-transformation-fractal-set-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.3. Spatial Transformation Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
For a spatial transformation $\Psi : \mathcal{X} \to \mathcal{X}$ (e.g., translation $T_a$, rotation $R \in SO(d)$), define the **push-forward** on the Fractal Set:

$$
\Psi_*(\mathcal{F}) = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}}, \Psi \circ \Phi)
$$

where:
- Episodes $\mathcal{E}$ are unchanged (graph vertices)
- CST/IG edges are unchanged (genealogy is spatial-transformation-invariant)
- Spatial embedding is transformed: $\Phi(e) \mapsto \Psi(\Phi(e))$

**Functional transformation**: For functionals $F[\mathcal{F}]$ depending on spatial positions:

$$
(\Psi_* F)[\mathcal{F}] = F[\Psi_*(\mathcal{F})]
$$

---

### Fractal Set Paths and Loops

**Type:** Definition
**Label:** `def-fractal-set-paths-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 2.1. Parallel Transport on CST and IG](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
A **path** in the Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ is a sequence of episodes:

$$
\gamma = (e_0, e_1, \ldots, e_k)
$$

where consecutive episodes are connected by either:
- **CST edge** (timelike): $e_i \to e_{i+1}$ (parent → child)
- **IG edge** (spacelike): $e_i \sim e_{i+1}$ (selection coupling)

A path is a **loop** if $e_0 = e_k$ (returns to starting episode).

**Path type**:
- **Purely timelike**: all edges are CST edges (follows genealogy)
- **Purely spacelike**: all edges are IG edges (follows selection coupling)
- **Mixed**: contains both CST and IG edges

---

### Discrete Gauge Connection via Episode Permutations

**Type:** Definition
**Label:** `def-discrete-gauge-connection-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 2.1. Parallel Transport on CST and IG](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `gauge-theory`, `metric-tensor`

**Statement:**
The **discrete gauge group** acting on the Fractal Set is the symmetric group:

$$
G_{\text{gauge}} = S_{|\mathcal{E}|}
$$

For a path $\gamma = (e_0, e_1, \ldots, e_k)$ in $\mathcal{F}$, define the **parallel transport permutation** as an element:

$$
U(\gamma) \in S_{|\mathcal{E}|}
$$

This permutation is the ordered product of elementary transports:

$$
U(\gamma) = U(e_{k-1}, e_k) \circ U(e_{k-2}, e_{k-1}) \circ \cdots \circ U(e_0, e_1)
$$

**Elementary transport** $U(e_i, e_j) \in S_{|\mathcal{E}|}$:

1. **CST edge** ($e_i \to e_j$): Identity permutation (timelike transport preserves labels)

   $$
   U(e_i, e_j) = \text{id}_{S_{|\mathcal{E}|}}
   $$

2. **IG edge** ($e_i \sim e_j$): Transposition (spacelike transport swaps episodes)

   $$
   U(e_i, e_j) = (e_i \, e_j) \in S_{|\mathcal{E}|}
   $$

   where $(e_i \, e_j)$ is the transposition swapping the two episodes (acting on episodes themselves, not their labels).

**Holonomy**: For a loop $\gamma$ with $e_0 = e_k$, the **holonomy** is simply the parallel transport permutation:

$$
\text{Hol}(\gamma) = U(\gamma) \in S_{|\mathcal{E}|}
$$

This is the net permutation accumulated by traversing the loop.

---

### Wilson Loops on Fractal Set

**Type:** Definition
**Label:** `def-wilson-loops-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 2.2. Holonomy and Wilson Loops](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `curvature`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `gauge-theory`

**Statement:**
For a loop $\gamma$ in $\mathcal{F}$ and a representation $\rho : S_{|\mathcal{E}|} \to \text{GL}(V)$, the **Wilson loop** is:

$$
W_\rho(\gamma) = \text{Tr}_V[\rho(\text{Hol}(\gamma))]
$$

**Standard representations**:

1. **Trivial representation**: $\rho(\sigma) = 1$ for all $\sigma \in S_{|\mathcal{E}|}$

   $$
   W_{\text{triv}}(\gamma) = 1 \quad \text{(always)}
   $$

2. **Sign representation**: $\rho(\sigma) = \text{sgn}(\sigma)$ (parity of permutation)

   $$
   W_{\text{sign}}(\gamma) = (-1)^{\# \text{IG edges in } \gamma}
   $$

3. **Fundamental representation**: $\rho = \text{id}$ (permutation matrices)

   $$
   W_{\text{fund}}(\gamma) = \text{Tr}[\text{Hol}(\gamma)]
   $$

**Physical interpretation**: Wilson loops are gauge-invariant observables detecting non-trivial holonomy (curvature) in the discrete gauge connection.

---

### Plaquettes in the Fractal Set

**Type:** Definition
**Label:** `def-plaquettes-fractal-set-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 2.3. Discrete Curvature Functional](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `curvature`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
A **plaquette** $P$ in $\mathcal{F}$ is a closed loop of length 4 (square):

$$
P = (e_0, e_1, e_2, e_3, e_0)
$$

where edges alternate between CST and IG:

$$
e_0 \xrightarrow{\text{CST}} e_1 \xrightarrow{\text{IG}} e_2 \xrightarrow{\text{CST}} e_3 \xrightarrow{\text{IG}} e_0
$$

**Plaquette holonomy**:

$$
\text{Hol}(P) = U(e_3, e_0) \circ U(e_2, e_3) \circ U(e_1, e_2) \circ U(e_0, e_1)
$$

$$
= (e_0 \, e_3) \circ \text{id} \circ (e_1 \, e_2) \circ \text{id} = (e_0 \, e_3) \circ (e_1 \, e_2)
$$

**Trivial vs. non-trivial plaquettes**:
- **Trivial**: $\text{Hol}(P) = \text{id}$ (flat curvature)
- **Non-trivial**: $\text{Hol}(P) \neq \text{id}$ (non-zero curvature)

---

### Discrete Curvature Functional

**Type:** Definition
**Label:** `def-discrete-curvature-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 2.3. Discrete Curvature Functional](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `curvature`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
The **discrete curvature** at a plaquette $P$ is:

$$
\kappa(P) = \begin{cases}
0 & \text{if } \text{Hol}(P) = \text{id} \\
1 & \text{if } \text{Hol}(P) \neq \text{id}
\end{cases}
$$

**Alternative (cycle index)**:

$$
\kappa_{\text{cycle}}(P) = \frac{1}{|\mathcal{E}|!} \sum_{g \in S_{|\mathcal{E}|}} |\text{Fix}(g \cdot \text{Hol}(P))|
$$

where $\text{Fix}(g)$ is the number of fixed points of permutation $g$.

**Integrated curvature**: The total curvature in spacetime region $\mathcal{R} \subset \mathcal{X} \times [0, T]$ is:

$$
K(\mathcal{R}) = \sum_{\substack{P \text{ plaquette} \\ P \subset \mathcal{R}}} \kappa(P)
$$

---

### Order-Invariant Functionals

**Type:** Definition
**Label:** `def-order-invariant-functionals-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 4. Order-Invariant Functionals](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `causal-tree`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `metric-tensor`

**Statement:**
A functional $F : \mathcal{C} \to \mathbb{R}$ on the space of CST configurations $\mathcal{C}$ is **order-invariant** if:

$$
F(\psi(\mathcal{T})) = F(\mathcal{T})
$$

for every **causal automorphism** $\psi : (\mathcal{E}, \prec) \to (\mathcal{E}', \prec')$ (order-preserving bijection).

**Examples of order-invariant functionals**:

1. **Interval cardinalities**: $|I(e, e')| := |\{e'' : e \prec e'' \prec e'\}|$
2. **Longest chain lengths**: $\ell(e, e') := \max\{\text{length of chains from } e \text{ to } e'\}$
3. **Antichain sizes**: $\max |\mathcal{A}|$ for $\mathcal{A} \subset \mathcal{E}$ an antichain
4. **Causal diamond counts**: $\#\{e : e_1 \prec e \prec e_2\}$

**Examples that are NOT order-invariant**:

1. **Episode death times** $t^{\rm d}_e$ (depend on coordinate choice)
2. **Proper lifetimes** $\tau_e$ (depend on metric, not just order)
3. **Spatial distances** $d_g(\Phi(e), \Phi(e'))$ (depend on embedding, not just order)

**Key principle**: Order-invariant functionals capture intrinsic graph structure independent of coordinate choices or metric details.

---

### Graph-Theoretic Conserved Quantities

**Type:** Definition
**Label:** `def-graph-conserved-quantities-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 5. Conservation Laws from Discrete Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `causal-tree`, `conservation`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
A functional $Q : \mathcal{F}_t \to \mathbb{R}$ is **conserved** if:

$$
\mathbb{E}[Q(\mathcal{F}_{t+\Delta t}) \mid \mathcal{F}_t] = Q(\mathcal{F}_t)
$$

for all timesteps $t$.

**Examples of conserved graph quantities**:

1. **Total episode count in a time slice**:

   $$
   N_{\text{alive}}(t) = |\mathcal{A}(t)| = |\{e \in \mathcal{E} : t^{\rm b}_e \leq t < t^{\rm d}_e\}|
   $$

2. **Causal order**:

   $$
   \text{Order}(\mathcal{F}) = \{(e_i, e_j) : e_i \prec e_j\}
   $$

   The partial order $\prec$ is **monotonically increasing**: once $e_i \prec e_j$, this relation persists.

3. **Total edge count** (CST + IG):

   $$
   E_{\text{total}}(t) = |E_{\text{CST}}| + |E_{\text{IG}}|
   $$

   This is non-conserved but has **bounded growth rate**.

---

### Propositions (4)

### Symmetry Correspondence Across Layers

**Type:** Proposition
**Label:** `prop-symmetry-correspondence-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 3. Discrete-Continuous Symmetry Correspondence](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `convergence`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `gauge-theory`, `symmetry`

**Statement:**
The symmetries established in Chapter 9 have exact discrete analogues in the Fractal Set:

| **Symmetry** | **Layer 3 (Ch 9)** | **Layer 2 (Ch 12)** | **Layer 1 (This Ch)** | **Convergence** |
|--------------|-------------------|---------------------|----------------------|-----------------|
| **Permutation** | $S_N$ | Gauge group $S_N$ | Episode relabeling $S_{\|\mathcal{E}\|}$ | {prf:ref}`thm-discrete-permutation-invariance-discrete-sym` |
| **Translation** | $T_a$ (Thm 9.2.2) | Config space symmetry | $T_a$-invariance ({prf:ref}`thm-discrete-translation-equivariance-discrete-sym`) | Exact |
| **Rotation** | $SO(d)$ (Thm 9.2.3) | Braid group $B_N$ | Plaquette symmetry | {prf:ref}`cor-discrete-rotational-equivariance-discrete-sym` |
| **Time-reversal** | Broken (Thm 9.2.5) | CST acyclicity | No timelike loops | {prf:ref}`conj-connection-to-braid-holonomy-discrete-sym` |

**Key insight**: Every continuous symmetry has a **discrete graph-theoretic analogue** in the Fractal Set.

**Related Results:** `thm-discrete-permutation-invariance-discrete-sym`, `conj-connection-to-braid-holonomy-discrete-sym`, `thm-discrete-translation-equivariance-discrete-sym`, `cor-discrete-rotational-equivariance-discrete-sym`

---

### Episode Count Conservation

**Type:** Proposition
**Label:** `prop-episode-count-conservation-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 5. Conservation Laws from Discrete Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `conservation`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
At each timestep $t$, the number of alive episodes equals the number of walkers:

$$
|\mathcal{A}(t)| = N
$$

This is **exactly conserved** by the cloning operator: every cloning event replaces one dead walker with one alive walker, maintaining $N$ alive walkers at all times.

---

### Edge Growth Bound

**Type:** Proposition
**Label:** `prop-edge-growth-bound-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 5. Conservation Laws from Discrete Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
The rate of edge creation in the Fractal Set is bounded by:

$$
\frac{d}{dt} |E_{\text{CST}} \cup E_{\text{IG}}| \leq C_{\text{clone}} N + C_{\text{select}} N^2
$$

where:
- $C_{\text{clone}}$: Cloning rate per walker (CST edges)
- $C_{\text{select}}$: Selection coupling rate (IG edges)

**CST edges**:

$$
\frac{d}{dt} |E_{\text{CST}}| \leq C_{\text{clone}} N
$$

**IG edges**:

$$
\frac{d}{dt} |E_{\text{IG}}| \leq C_{\text{select}} \binom{N}{2} \sim C_{\text{select}} N^2
$$

---

### Correspondence between Discrete Symmetries and Conserved Quantities

**Type:** Proposition
**Label:** `prop-noether-correspondence-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 5. Conservation Laws from Discrete Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `conservation`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `metric-tensor`, `noether`, `symmetry`

**Statement:**
Each discrete symmetry of the Fractal Set has a corresponding conserved or quasi-conserved quantity, in analogy with Noether's theorem for continuous symmetries.

| **Symmetry** | **Conserved Quantity** | **Justification** |
|--------------|------------------------|-------------------|
| Episode permutation $S_{|\mathcal{E}|}$ | Graph isomorphism class | {prf:ref}`thm-discrete-permutation-invariance-discrete-sym`: All graph functionals are invariant under relabeling |
| Translation $T_a$ | Center of mass of spatial embedding | {prf:ref}`thm-discrete-translation-equivariance-discrete-sym`: Translation equivariance implies $\mathbb{E}[\Phi_{\text{COM}}(\mathcal{F}_{t+\Delta t})] = \mathbb{E}[\Phi_{\text{COM}}(\mathcal{F}_t)] + a$ |
| Rotation $SO(d)$ | Angular momentum (if velocity included) | {prf:ref}`cor-discrete-rotational-equivariance-discrete-sym`: Rotational equivariance implies conservation of total angular momentum about the origin |
| Time-translation | Episode count per time slice | {prf:ref}`prop-episode-count-conservation-discrete-sym`: Exactly $N$ alive episodes at each timestep |

**Heuristic arguments**:

1. **Translation**: Define the center of mass as $\Phi_{\text{COM}}(\mathcal{F}) = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \Phi(e)$. Translation equivariance of the dynamics ensures this quantity shifts by exactly $a$ under $T_a$, implying conservation of the center of mass in the comoving frame.

2. **Rotation**: If velocity is included in the embedding, define total angular momentum as $L(\mathcal{F}) = \sum_{e \in \mathcal{E}} \Phi(e) \times v_e$. Rotational equivariance implies this quantity is conserved (up to dissipation from friction).

3. **Time-translation**: The population regulation mechanism ({prf:ref}`prop-episode-count-conservation-discrete-sym`) ensures exactly $N$ episodes are alive at each time, a discrete analogue of probability conservation.

**Key distinction**: Discrete conservation laws are **combinatorial** (counting episodes/edges), while continuous conservation laws are **geometric** (momentum, energy). The connection is established through the continuum limit.

**Related Results:** `thm-discrete-permutation-invariance-discrete-sym`, `thm-discrete-translation-equivariance-discrete-sym`, `cor-discrete-rotational-equivariance-discrete-sym`, `prop-episode-count-conservation-discrete-sym`

---

### Theorems (3)

### Discrete Permutation Invariance

**Type:** Theorem
**Label:** `thm-discrete-permutation-invariance-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.2. Permutation Invariance Theorem](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
The Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ is **statistically invariant** under episode relabeling.

**Precise statement**: For any bijection $\sigma : \mathcal{E} \to \mathcal{E}$ and any graph-theoretic functional $F : \mathcal{F} \to \mathbb{R}$,

$$
\mathbb{E}[F(\mathcal{F})] = \mathbb{E}[F(\sigma(\mathcal{F}))]
$$

where $\sigma(\mathcal{F}) = (\sigma(\mathcal{E}), \{\sigma(e_i) \to \sigma(e_j) : e_i \to e_j \in E_{\text{CST}}\} \cup \{\sigma(e_i) \sim \sigma(e_j) : e_i \sim e_j \in E_{\text{IG}}\})$.

**Graph isomorphism**: The relabeled Fractal Set is isomorphic to the original:

$$
\sigma(\mathcal{F}) \cong \mathcal{F}
$$

as labeled graphs (same structure, different vertex labels).

---

### Discrete Translation Equivariance

**Type:** Theorem
**Label:** `thm-discrete-translation-equivariance-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.3. Spatial Transformation Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `symmetry`

**Statement:**
Suppose the reward function $R(x, v)$ and valid domain $\mathcal{X}_{\text{valid}}$ satisfy:

$$
R(x + a, v) = R(x, v), \quad x + a \in \mathcal{X}_{\text{valid}} \iff x \in \mathcal{X}_{\text{valid}}
$$

for some $a \in \mathbb{R}^d$ (translation symmetry of the environment).

Then the Fractal Set distribution is **translation-equivariant**:

$$
\mathcal{L}(\mathcal{F}) = \mathcal{L}(T_a(\mathcal{F}))
$$

**Graph structure preservation**: The CST and IG edge sets are **invariant** under translation:

$$
E_{\text{CST}}[T_a(\mathcal{F})] = E_{\text{CST}}[\mathcal{F}], \quad E_{\text{IG}}[T_a(\mathcal{F})] = E_{\text{IG}}[\mathcal{F}]
$$

Only the spatial embedding $\Phi$ changes: $\Phi(e) \mapsto \Phi(e) + a$.

---

### Order-Invariance Implies Lorentz Invariance (Continuum Limit)

**Type:** Theorem
**Label:** `thm-order-invariance-lorentz-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 4. Order-Invariant Functionals](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `causal-tree`, `convergence`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`

**Statement:**
**Conjecture**: If a discrete functional $F$ on the Fractal Set is order-invariant, then its continuum limit (if it exists) is Lorentz-invariant.

**Heuristic argument**:

1. **Discrete**: Order-invariance means $F$ depends only on causal structure $\prec$, not on coordinates or embeddings.

2. **Continuum**: In Minkowski spacetime, the causal structure (light cone ordering) is preserved by Lorentz transformations:

   $$
   x \prec y \iff (y - x) \text{ is future-directed timelike or null}
   $$

3. **Convergence**: If $F_N \to F_\infty$ as $N \to \infty$, and each $F_N$ is order-invariant, then $F_\infty$ must be Lorentz-invariant (causal structure is the only preserved data).

**Rigorous proof**: Requires characterizing the continuum limit of the Fractal Set and proving convergence of order-invariant functionals. This is future work.

---

### Corollarys (2)

### Continuous Limit Recovers Swarm Permutation Symmetry

**Type:** Corollary
**Label:** `cor-continuous-limit-permutation-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.2. Permutation Invariance Theorem](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `convergence`, `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `symmetry`

**Statement:**
In the limit $N \to \infty$, the episode relabeling symmetry $G_{\text{epi}} \cong S_{|\mathcal{E}|}$ converges to the **walker permutation symmetry** $S_N$ from Chapter 9.

The episode permutation group $S_{|\mathcal{E}_N|}$ restricts to $S_N$ via the **walker projection map**:

$$
\pi : S_{|\mathcal{E}_N|} \to S_N, \quad \sigma \mapsto \tilde{\sigma}
$$

where $\tilde{\sigma}(i) = j$ if $\sigma$ maps the current episode of walker $i$ to the current episode of walker $j$.

**Convergence**: For functionals $F$ that factor through walker indices,

$$
\lim_{N \to \infty} \mathbb{E}[F(\mathcal{F}_N)] = \mathbb{E}[F(\text{swarm configuration})]
$$

recovering Chapter 9's permutation symmetry in the continuum.

---

### Rotational Equivariance

**Type:** Corollary
**Label:** `cor-discrete-rotational-equivariance-discrete-sym`
**Source:** [13_fractal_set_new/07_discrete_symmetries_gauge.md § 1.3. Spatial Transformation Symmetries](13_fractal_set_new/07_discrete_symmetries_gauge.md)
**Tags:** `discrete-symmetries`, `fractal-set`, `gauge-symmetry`, `metric-tensor`

**Statement:**
For rotationally symmetric reward $R(Rx, Rv) = R(x, v)$ and domain ($R \in SO(d)$), the Fractal Set is rotationally equivariant:

$$
\mathcal{L}(\mathcal{F}) = \mathcal{L}(R_*(\mathcal{F}))
$$

The proof follows the same structure as {prf:ref}`thm-discrete-translation-equivariance-discrete-sym`.

**Related Results:** `thm-discrete-translation-equivariance-discrete-sym`

---


## 08_lattice_qft_framework.md

**Objects in this document:** 24

### Definitions (13)

### Effective Speed of Causation

**Type:** Definition
**Label:** `def-effective-causal-speed`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 1.2. Temporal Structure and Global Hyperbolicity](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `causal-tree`, `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 1.3, lines 187-213]

From cloning dynamics (Chapter 3), spatial perturbation $\Delta \mathbf{x} \sim \mathcal{N}(0, \delta^2 I)$ over episode duration $\tau_e$ defines:

$$
c_{\text{eff}} = \frac{\delta}{\langle \tau \rangle}
$$

where:
- $\delta$: Cloning noise scale (Chapter 3, Definition 3.2.2.2)
- $\langle \tau \rangle$: Mean episode duration

**Lightcone condition:** Episode $e'$ is in the causal future of $e$ only if:

$$
e \prec e' \implies d_g(\mathbf{x}_e, \mathbf{x}_{e'}) \leq c_{\text{eff}} (t_{e'} - t_e)
$$

**Physical interpretation:** $c_{\text{eff}}$ is the maximum information propagation speed through the CST. The CST has a **built-in UV cutoff** at scale $\delta$ (analogous to Planck length).

---

### Fractal Set as Simplicial Complex

**Type:** Definition
**Label:** `def-fractal-set-simplicial-complex`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 3.1. Fractal Set as 2-Complex](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 3.2, lines 447-478]

The Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ is a **2-dimensional simplicial complex**:

**0-cells (vertices):** Episodes $e \in \mathcal{E}$

**1-cells (edges):**
- **Timelike edges:** CST edges $e_i \to e_j$ (directed)
- **Spacelike edges:** IG edges $e_i \sim e_j$ (undirected)

**2-cells (faces):** Plaquettes $P$ (alternating CST-IG 4-cycles):

$$
P = (e_0 \xrightarrow{\text{CST}} e_1 \xrightarrow{\text{IG}} e_2 \xrightarrow{\text{CST}} e_3 \xrightarrow{\text{IG}} e_0)
$$

**Boundary operator:** For a 2-cell $P = (e_0, e_1, e_2, e_3)$:

$$
\partial P = (e_0 \to e_1) + (e_1 \sim e_2) + (e_2 \to e_3) + (e_3 \sim e_0)
$$

**Cohomology:** The cohomology groups $H^k(\mathcal{F})$ classify topological features:
- $H^0$: Connected components
- $H^1$: Loops (1-cycles not bounding 2-chains)
- $H^2$: Voids (2-cycles)

---

### Paths and Wilson Loops

**Type:** Definition
**Label:** `def-paths-wilson-loops`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 3.1. Fractal Set as 2-Complex](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 3.3, lines 482-520]

A **path** in $\mathcal{F}$ is a sequence:

$$
\gamma = (e_0, e_1, \ldots, e_n)
$$

where consecutive episodes are connected by either CST edge (timelike step) or IG edge (spacelike step).

**Path classification:**
1. **Timelike path:** All edges are CST edges (follows genealogy)
2. **Spacelike path:** All edges are IG edges (stays within one time slice)
3. **Mixed path:** Contains both CST and IG edges

**Closed loop:** A path with $e_n = e_0$ (returns to starting episode).

**Physical interpretation:** Closed loops are precisely the **Wilson loops** in lattice gauge theory.

---

### U(1) Gauge Field on Fractal Set

**Type:** Definition
**Label:** `def-u1-gauge-field`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 4.1. Gauge Group and Parallel Transport](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `metric-tensor`, `qft`, `u1-symmetry`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 4.1, lines 527-561]

A **U(1) gauge field** (electromagnetic field) on $\mathcal{F}$ assigns parallel transport operators to edges:

**CST edges (timelike):**

$$
U_{\text{time}}(e_i \to e_j) = \exp\left(-i q A_0(e_i, e_j) \tau_{ij}\right) \in U(1)
$$

where:
- $A_0$: Temporal component of gauge potential (electric potential)
- $\tau_{ij}$: Proper time along edge (episode duration)
- $q$: Electric charge (coupling constant)

**IG edges (spacelike):**

$$
U_{\text{space}}(e_i \sim e_j) = \exp\left(i q \int_{e_i}^{e_j} \mathbf{A} \cdot d\mathbf{x}\right) \in U(1)
$$

where:
- $\mathbf{A}$: Spatial components of gauge potential (vector potential)
- Integral along geodesic from $\mathbf{x}_i = \Phi(e_i)$ to $\mathbf{x}_j = \Phi(e_j)$ in emergent metric $g$

**Gauge transformation:** Under $U : \mathcal{E} \to U(1)$:

$$
U(e_i, e_j) \mapsto U(e_i) U(e_i, e_j) U(e_j)^{-1}
$$

---

### SU(N) Gauge Field on Fractal Set

**Type:** Definition
**Label:** `def-sun-gauge-field`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 4.1. Gauge Group and Parallel Transport](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`, `su2-symmetry`, `su3-symmetry`, `u1-symmetry`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 4.2, lines 571-610]

For **non-abelian gauge group** $G = SU(N)$ (e.g., $SU(3)$ for QCD), parallel transport operators are **$N \times N$ unitary matrices**:

**CST edges:**

$$
U_{\text{time}}(e_i \to e_j) = \mathcal{P} \exp\left(-i g \int_{e_i}^{e_j} A_0^a T^a dt\right) \in SU(N)
$$

**IG edges:**

$$
U_{\text{space}}(e_i \sim e_j) = \mathcal{P} \exp\left(i g \int_{e_i}^{e_j} A_i^a T^a dx^i\right) \in SU(N)
$$

where:
- $A_\mu^a$: Gauge field components ($a = 1, \ldots, N^2 - 1$)
- $T^a$: Generators of $SU(N)$ (Lie algebra basis)
- $g$: Gauge coupling constant
- $\mathcal{P}$: Path-ordered exponential

**Physical interpretation:**
- $SU(2)$: Weak interaction
- $SU(3)$: Strong interaction (QCD)
- $SU(3) \times SU(2) \times U(1)$: Standard Model gauge group

---

### Discrete Field Strength Tensor

**Type:** Definition
**Label:** `def-discrete-field-strength`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 4.2. Plaquette Field Strength](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `area`, `curvature`, `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`, `u1-symmetry`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 5.2, lines 680-715]

For an **oriented** plaquette $P = (e_0, e_1, e_2, e_3, e_0)$ in $\mathcal{F}$, the **field strength** (plaquette holonomy) is:

$$
F[P] = U(e_0 \to e_1) U(e_1 \sim e_2) U(e_2 \to e_3)^{\dagger} U(e_3 \sim e_0)^{\dagger}
$$

where:
- Forward transport: $U(e_i \to e_j)$ for CST edges, $U(e_i \sim e_j)$ for IG edges
- Reverse transport: $U(e_i \to e_j)^{\dagger} = U(e_j \to e_i)$ (adjoint/inverse)
- Plaquette orientation: Chosen consistently (e.g., counterclockwise)

**Physical interpretation:** This measures the **commutator** of parallel transport around a closed loop, which is the discrete analog of the curvature $[D_\mu, D_\nu] = F_{\mu\nu}$.

**Small loop expansion:** For small plaquettes (lattice spacing $a \to 0$):

$$
F[P] = \mathbb{I} + i a^2 F_{\mu\nu} + O(a^3)
$$

where $F_{\mu\nu}$ is the continuum **field strength tensor** (electromagnetic tensor or Yang-Mills curvature).

**For U(1) gauge theory:**

$$
F[P] = e^{i q \Phi[P]}
$$

where $\Phi[P]$ is the magnetic flux through plaquette $P$:

$$
\Phi[P] = \int_P F_{\mu\nu} dx^\mu \wedge dx^\nu \approx F_{\mu\nu} \cdot \text{Area}(P)
$$

**Note:** The use of adjoints in reverse directions ensures $F[P] \to \mathbb{I}$ as $a \to 0$, with non-trivial first-order correction $\propto F_{\mu\nu}$.

---

### Wilson Lattice Gauge Action

**Type:** Definition
**Label:** `def-wilson-gauge-action`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 4.3. Wilson Action](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`, `u1-symmetry`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 6.1, lines 772-806]

The **Wilson action** on the Fractal Set is:

$$
S_{\text{Wilson}}[A] = \beta \sum_{\text{plaquettes } P \subset \mathcal{F}} \left(1 - \frac{1}{N} \text{Re} \, \text{Tr} \, U[P]\right)
$$

where:
- $\beta = 2N / g^2$: Inverse coupling constant (for gauge group $SU(N)$ or $U(1)$)
- $U[P]$: Plaquette holonomy (Definition {prf:ref}`def-discrete-field-strength`)
- $N = \dim(G)$: Dimension of gauge group representation
- Sum over all plaquettes in $\mathcal{F}$ (mixed CST-IG 4-cycles)

**Continuum limit:** As lattice spacing $a \to 0$:

$$
S_{\text{Wilson}} \to \frac{1}{4g^2} \int d^4x \, F_{\mu\nu} F^{\mu\nu}
$$

(Yang-Mills action in continuum).

**For U(1) theory:** $N = 1$, so:

$$
S_{\text{Wilson}} = \beta \sum_P (1 - \cos(\Phi[P]))
$$

where $\Phi[P]$ is the magnetic flux through plaquette $P$.

**Related Results:** `def-discrete-field-strength`

---

### Wilson Loop Operator

**Type:** Definition
**Label:** `def-wilson-loop-operator`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 5.1. Wilson Loop Observable](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 5.1, lines 650-676]

For a closed loop $\gamma$ in $\mathcal{F}$ and gauge group $G$, the **Wilson loop** is:

$$
W[\gamma] = \text{Tr}\left[\prod_{\text{edges } e \in \gamma} U(e)\right]
$$

where:
- $U(e)$: Parallel transport operator on edge $e$
- Product taken in **path order** (starting at arbitrary base point)
- Trace over gauge group representation (ensures gauge invariance)

**Gauge invariance:** Under gauge transformation $\Omega : \mathcal{E} \to G$:

$$
W[\gamma] \mapsto \text{Tr}\left[\Omega(e_0) \left(\prod U(e)\right) \Omega(e_0)^\dagger\right] = W[\gamma]
$$

(trace is cyclic, so base point cancels).

**Physical interpretation:**
- $W[\gamma]$ measures **gauge field flux** through any surface bounded by $\gamma$
- In QED: $W[\gamma] = e^{i q \Phi_B}$ (Aharonov-Bohm effect)
- In QCD: $W[\gamma]$ gives **quark confinement potential** $V(R) \sim R$ for large loops

---

### Antisymmetric Fermionic Kernel

**Type:** Definition
**Label:** `def-fermionic-kernel`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 7.1. Antisymmetric Cloning Kernel](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `fermionic`, `field-theory`, `fractal-set`, `lattice-qft`, `metric-tensor`, `qft`, `symmetry`

**Statement:**
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 4.3, lines 329-356]

Define the **antisymmetric kernel**:

$$
\tilde{K}(i, j) := K(i, j) - K(j, i)
$$

where $K(i,j)$ is the cloning kernel (probability $i$ clones from $j$).

From cloning scores:

$$
K(i, j) \propto \max(0, S_i(j))
$$

**Antisymmetric part:**

$$
\tilde{K}(i, j) = K(i, j) - K(j, i) \propto S_i(j) - S_j(i)
$$

For non-zero fitness differences ($V_i \neq V_j$):

$$
\tilde{K}(i, j) \neq 0 \quad \text{(non-trivial antisymmetry)}
$$

This kernel has the **mathematical structure of fermionic propagators**.

---

### Discrete Fermionic Action

**Type:** Definition
**Label:** `def-discrete-fermionic-action`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 7.3. Fermionic Action on Fractal Set](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `fermionic`, `field-theory`, `fractal-set`, `lattice-qft`, `metric-tensor`, `qft`

**Statement:**
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 6.1, lines 426-450]

On the Fractal Set $\mathcal{F}$, the fermionic action has **spatial** and **temporal** components:

$$
\boxed{S_{\text{fermion}} = S_{\text{fermion}}^{\text{spatial}} + S_{\text{fermion}}^{\text{temporal}}}
$$

**Spatial component** (IG edges):

$$
S_{\text{fermion}}^{\text{spatial}} = -\sum_{(i,j) \in E_{\text{IG}}} \bar{\psi}_i \tilde{K}_{ij} \psi_j
$$

where:
- $\bar{\psi}_i, \psi_j$: Grassmann fields on episodes $i, j$
- $\tilde{K}_{ij} = K_{ij} - K_{ji}$: Antisymmetric cloning kernel (spatial correlations)
- Sum over IG edges (simultaneously alive episodes)

**Temporal component** (CST edges):

$$
S_{\text{fermion}}^{\text{temporal}} = -\sum_{(i \to j) \in E_{\text{CST}}} \bar{\psi}_i D_t \psi_j
$$

where:
- $(i \to j)$: CST edges (parent-child relations)
- $D_t$: Discrete time derivative (forward difference)
- Encodes temporal propagation of fermions along genealogy

**Note:** The current framework (Chapter 13) only defines the **spatial component** explicitly. The temporal component is proposed here to achieve a complete fermionic theory. Future work must derive $D_t$ from cloning dynamics across timesteps.

**Propagator:**

$$
G(i, j) = \langle \psi_i \bar{\psi}_j \rangle = ((\tilde{K} + D_t)^{-1})_{ij}
$$

**Path integral:**

$$
Z = \int \mathcal{D}[\bar{\psi}] \mathcal{D}[\psi] \, e^{-S_{\text{fermion}}}
$$

---

### Lattice Scalar Field Action

**Type:** Definition
**Label:** `def-lattice-scalar-action`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 8.1. Scalar Field Action on Fractal Set](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`, `qsd`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 7.1, lines 886-925]

A **real scalar field** $\phi : \mathcal{E} \to \mathbb{R}$ on the Fractal Set has action:

$$
S_{\text{scalar}}[\phi] = \sum_{e \in \mathcal{E}} \left[\frac{1}{2} (\partial_\mu \phi)^2(e) + \frac{m^2}{2} \phi(e)^2 + V(\phi(e))\right]
$$

where:
- $(\partial_\mu \phi)^2$: Discrete derivative (finite difference on CST+IG edges)
- $m$: Scalar mass
- $V(\phi)$: Self-interaction potential (e.g., $V(\phi) = \frac{\lambda}{4!} \phi^4$)

**Discrete derivatives:**

**Timelike direction (CST edges):** Since CST is a tree structure where a parent episode can have multiple children, we average over all immediate children:

$$
(\partial_0 \phi)(e) = \frac{1}{|\text{Children}(e)|} \sum_{e_c \in \text{Children}(e)} \frac{\phi(e_c) - \phi(e)}{\tau_e}
$$

where $\text{Children}(e) = \{e' : e \to e' \in E_{\text{CST}}\}$ is the set of immediate children of episode $e$.

**Special case:** If $|\text{Children}(e)| = 0$ (leaf node, episode dies without cloning), define $(\partial_0 \phi)(e) = 0$ (no temporal evolution).

**Rationale:** Averaging over children treats all branches of the genealogical tree democratically, consistent with the QSD equilibrium measure.

**Spacelike directions (IG edges):**

$$
(\partial_i \phi)(e) = \frac{1}{|\text{IG}(e)|} \sum_{e' \sim e} \frac{\phi(e') - \phi(e)}{d_g(\mathbf{x}_e, \mathbf{x}_{e'})}
$$

**Kinetic term:**

$$
(\partial_\mu \phi)^2 = -(\partial_0 \phi)^2 + \sum_{i=1}^d (\partial_i \phi)^2
$$

(Lorentzian signature: negative time, positive space).

---

### Graph Laplacian on Fractal Set

**Type:** Definition
**Label:** `def-graph-laplacian-fractal-set`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 8.2. Graph Laplacian Equals Laplace-Beltrami Operator](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `laplacian`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 7.2b, lines 1007-1025]

For scalar field $\phi: \mathcal{E} \to \mathbb{R}$, the **Graph Laplacian** is:

$$
(\Delta_{\text{graph}} \phi)(e_i) := \sum_{e_j \in \text{IG}(e_i)} w_{ij} \left[\phi(e_j) - \phi(e_i)\right]
$$

where:
- $\text{IG}(e_i) = \{e_j : (e_i, e_j) \in E_{\text{IG}}\}$: Set of IG neighbors
- $w_{ij}$: Edge weights (Theorem {prf:ref}`thm-ig-edge-weights-algorithmic`)

**Normalized version:**

$$
(\Delta_{\text{norm}} \phi)(e_i) = \frac{1}{d_i} \sum_{e_j \in \text{IG}(e_i)} w_{ij} \left[\phi(e_j) - \phi(e_i)\right]
$$

where $d_i = \sum_{e_j} w_{ij}$ is the weighted degree.

**Related Results:** `thm-ig-edge-weights-algorithmic`

---

### Total QFT Action on Fractal Set

**Type:** Definition
**Label:** `def-total-qft-action`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 9.1. Unified Action Functional](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `fermionic`, `field-theory`, `fractal-set`, `gauge-theory`, `lattice-qft`, `qft`

**Statement:**
The complete quantum field theory on $\mathcal{F}$ has action:

$$
S_{\text{total}} = S_{\text{gauge}} + S_{\text{fermion}} + S_{\text{scalar}}
$$

where:

**Gauge sector:**

$$
S_{\text{gauge}} = \beta \sum_{P \subset \mathcal{F}} \left(1 - \frac{1}{N} \text{Re} \, \text{Tr} \, U[P]\right)
$$

**Fermion sector:**

$$
S_{\text{fermion}} = -\sum_{(i,j) \in E_{\text{IG}}} \bar{\psi}_i \tilde{K}_{ij} \psi_j
$$

**Scalar sector:**

$$
S_{\text{scalar}} = \sum_{e \in \mathcal{E}} \left[\frac{1}{2} (\partial_\mu \phi)^2(e) + \frac{m^2}{2} \phi(e)^2 + V(\phi(e))\right]
$$

**Partition function:**

$$
Z = \int \mathcal{D}[U] \mathcal{D}[\bar{\psi}] \mathcal{D}[\psi] \mathcal{D}[\phi] \, e^{-S_{\text{total}}}
$$

---

### Propositions (4)

### CST Satisfies Bombelli-Lee-Meyer-Sorkin Axioms

**Type:** Proposition
**Label:** `prop-cst-causal-set-axioms`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 1.1. Causal Set Axioms Verification](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `causal-tree`, `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`, `riemannian`, `volume`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 1.1, lines 133-145]

The Causal Spacetime Tree $\mathcal{T} = (\mathcal{E}, E_{\text{CST}}, \prec)$ satisfies the axiomatic requirements for a causal set in quantum gravity:

**CS1 (Partial Order):** The ancestry relation $\prec$ is irreflexive, transitive, and acyclic.

**Proof:** By construction (Chapter 13), CST edges represent parent-child relations $e_i \to e_j$ with $t^{\rm b}_j = t^{\rm d}_i$. This defines a directed acyclic graph (DAG):
- **Irreflexive:** No episode is its own ancestor
- **Transitive:** If $e_i \prec e_j$ and $e_j \prec e_k$, then $e_i \prec e_k$ by genealogy
- **Acyclic:** Time flows forward; no closed timelike curves

**CS2 (Local Finiteness):** Causal intervals $I(e, e') = \{e'' : e \prec e'' \prec e'\}$ are finite.

**Proof:** Each episode has finite lifetime and finite number of descendants between any two time points. The alive set $|\mathcal{A}(t)| = N < \infty$ bounds the number of episodes per time slice.

**CS3 (Manifoldlikeness):** Episode density approximates spacetime volume.

**Proof:** From Chapter 14, Theorem 14.4.1, the density of episodes per unit volume converges to the Riemannian volume measure $\sqrt{\det g(x)} dx$ in the continuum limit.

**Conclusion:** CST is a valid causal set in the Bombelli-Lee-Meyer-Sorkin framework for causal set quantum gravity. ∎

---

### CST Admits Global Time Function

**Type:** Proposition
**Label:** `prop-cst-global-time`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 1.2. Temporal Structure and Global Hyperbolicity](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 1.2, lines 149-171]

The CST is **globally hyperbolic**: there exists a continuous function $t : \mathcal{E} \to \mathbb{R}$ such that:

$$
e \prec e' \implies t(e) < t(e')
$$

**Explicit construction:** Use death time $t(e) = t^{\rm d}_e$.

**Proof:** By construction, CST edges $e_i \to e_j$ satisfy $t^{\rm b}_j = t^{\rm d}_i$, so:

$$
e_i \to e_j \implies t^{\rm d}_i < t^{\rm d}_j
$$

Transitivity of $\prec$ extends this to all ancestors/descendants. ∎

**Consequence:** Can define **Cauchy surfaces** $\Sigma_t = \{e : t^{\rm b}_e \leq t < t^{\rm d}_e\}$ (set of alive episodes at time $t$).

---

### IG Edges Connect Causally Disconnected Events

**Type:** Proposition
**Label:** `prop-ig-spacelike-separation`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 2.2. Spacelike vs Timelike Edges](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `causal-tree`, `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 2.1, lines 259-281]

For any IG edge $e_i \sim e_j$:

$$
e_i \not\prec e_j \quad \text{and} \quad e_j \not\prec e_i
$$

(neither is an ancestor of the other in the CST).

**Proof:** By construction, IG edges connect episodes that are **simultaneously alive**:

$$
e_i \sim e_j \implies \exists t : e_i, e_j \in \mathcal{A}(t)
$$

If $e_i \prec e_j$, then $t^{\rm d}_i < t^{\rm b}_j$ (child born after parent dies), contradicting simultaneous existence. Similarly, $e_j \prec e_i$ leads to contradiction. Therefore, $e_i$ and $e_j$ are causally independent in the CST. ∎

**Spacelike separation:** For $e_i \sim e_j$, define the interval:

$$
\Delta s^2(e_i, e_j) = -c^2 (t_i - t_j)^2 + d_g(\mathbf{x}_i, \mathbf{x}_j)^2 > 0
$$

(positive $\implies$ spacelike).

**Physical significance:** IG edges provide **non-local connections** between causally separated regions—exactly the structure needed for **quantum entanglement** in spacetime.

---

### Wilson Loop Area Law

**Type:** Proposition
**Label:** `prop-wilson-loop-area-law`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 5.2. Area Law and Confinement](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `area`, `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 5.3, lines 720-742]

In **confining gauge theories** (e.g., QCD), large Wilson loops exhibit **area law behavior**:

$$
\langle W[\gamma] \rangle \sim e^{-\sigma \, \text{Area}(\gamma)}
$$

where:
- $\sigma$: String tension (physical constant, $\sigma \sim 1 \, \text{GeV}^2$ for QCD)
- $\text{Area}(\gamma)$: Minimal area of surface bounded by loop $\gamma$
- $\langle \cdot \rangle$: Expectation value in quantum vacuum state

**Physical interpretation:** The area law arises from **flux tube formation** between quark-antiquark pairs—flux is confined to a narrow tube, giving energy $\propto$ length $\propto$ area.

**In CST+IG:** We can compute $\langle W[\gamma] \rangle$ by:
1. Summing over all CST+IG realizations (different algorithm runs)
2. Computing $W[\gamma]$ for each realization
3. Taking empirical average

**Prediction:** If the Adaptive Gas exhibits **confinement-like behavior** (walkers trapped in fitness basins), we expect area law scaling.

---

### Theorems (7)

### Algorithmic Determination of IG Edge Weights

**Type:** Theorem
**Label:** `thm-ig-edge-weights-algorithmic`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 2.1. IG Edge Weights from Selection Dynamics](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `lattice`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 2.1b, lines 284-345]

For episodes $e_i$ and $e_j$ with temporal overlap period $T_{\text{overlap}} = \{t : e_i, e_j \in \mathcal{A}(t)\}$, the IG edge weight is **algorithmically determined** (not arbitrary):

$$
w_{ij} = \int_{T_{\text{overlap}}} P(c_i(t) = j \mid i) \, dt
$$

where $P(c_i(t) = j \mid i)$ is the **companion selection probability** from Chapter 03, Definition 5.7.1:

$$
P(c_i(t) = j \mid i) = \frac{\exp\left(-\frac{d_{\text{alg}}(i,j;t)^2}{2\varepsilon_c^2}\right)}{Z_i(t)}
$$

with:
- **Algorithmic distance:** $d_{\text{alg}}(i,j)^2 = \|x_i - x_j\|^2 + \lambda_{\text{alg}} \|v_i - v_j\|^2$
- **Cloning interaction range:** $\varepsilon_c > 0$ (algorithm parameter)
- **Partition function:** $Z_i(t) = \sum_{l \in \mathcal{A}(t) \setminus \{i\}} \exp(-d_{\text{alg}}(i,l;t)^2 / 2\varepsilon_c^2)$

**Discrete approximation:**

$$
w_{ij} \approx \tau \sum_{k=1}^{n} \frac{\exp\left(-\frac{d_{\text{alg}}(i,j; t_k)^2}{2\varepsilon_c^2}\right)}{Z_i(t_k)}
$$

where $\tau$ is the discrete timestep size and $n$ is the number of timesteps in the overlap period.

**Proof:** See [13_E_cst_ig_lattice_qft.md § 2.1b, lines 316-345] for complete derivation.

**Conclusion:** Edge weights are **fully determined** by algorithmic distance, cloning interaction range, and episode overlap dynamics. No arbitrary choices. ∎

---

### CST+IG as Lattice for Gauge Theory and QFT

**Type:** Theorem
**Label:** `thm-cst-ig-lattice-qft-main`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 6. Complete Lattice QFT Framework](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `causal-tree`, `field-theory`, `fractal-set`, `gauge-theory`, `lattice`, `lattice-qft`, `qft`, `u1-symmetry`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 0.2, lines 24-61]

The Fractal Set $\mathcal{F} = (\mathcal{E}, E_{\text{CST}} \cup E_{\text{IG}})$ admits a **complete lattice gauge theory** structure with:

**1. Gauge group:** $G = S_{|\mathcal{E}|}$ (episode permutations) or $G = U(1)$ (electromagnetic) or $G = SU(N)$ (Yang-Mills)

**2. Gauge connection:** Parallel transport operators on edges:
- **CST edges (timelike):** $U_{\text{time}}(e_i \to e_j) = e^{-i \int_{e_i}^{e_j} A_0 dt}$
- **IG edges (spacelike):** $U_{\text{space}}(e_i \sim e_j) = e^{i \int_{e_i}^{e_j} \mathbf{A} \cdot d\mathbf{x}}$

**3. Wilson loops:** For closed path $\gamma = (e_0, e_1, \ldots, e_k = e_0)$:

$$
W[\gamma] = \text{Tr}\left[\mathcal{P} \exp\left(i \oint_\gamma A\right)\right] = \text{Tr}\left[\prod_{i=0}^{k-1} U(e_i, e_{i+1})\right]
$$

**4. Field strength:** Plaquette holonomy (with proper orientation):

$$
F[P] = U(e_0 \to e_1) U(e_1 \sim e_2) U(e_2 \to e_3)^{\dagger} U(e_3 \sim e_0)^{\dagger}
$$

for plaquette $P = (e_0, e_1, e_2, e_3)$ with alternating CST/IG edges, where $U^\dagger$ denotes adjoint (reverse transport).

**5. Action functional:**

$$
S_{\text{gauge}}[\mathcal{F}] = \sum_{\text{plaquettes } P} w_P \left(1 - \frac{1}{|G|}\text{Re}\,\text{Tr}\,F[P]\right)
$$

**Status:** All components **rigorously defined** through Chapters 13-14. This theorem **synthesizes** existing structures into unified QFT framework.

**Physical significance:**
- ✅ First **dynamics-driven lattice** for QFT (not hand-designed)
- ✅ Causal structure from **optimization dynamics**, not background geometry
- ✅ Quantum correlations from **algorithmic selection coupling**
- ✅ Enables **non-perturbative QFT** calculations on emergent spacetime

---

### Cloning Scores Exhibit Antisymmetric Structure

**Type:** Theorem
**Label:** `thm-cloning-antisymmetry`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 7.1. Antisymmetric Cloning Kernel](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `fermionic`, `field-theory`, `fractal-set`, `lattice-qft`, `metric-tensor`, `qft`, `symmetry`

**Statement:**
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 4.2, lines 294-322] ✅ **GEMINI VALIDATED**

The cloning scores (Chapter 3) satisfy:

$$
S_i(j) := \frac{V_j - V_i}{V_i + \varepsilon_{\text{clone}}}
$$

**Antisymmetry in numerators:**

$$
S_i(j) \propto (V_j - V_i), \quad S_j(i) \propto (V_i - V_j)
$$

$$
\boxed{S_i(j) + S_j(i) \cdot \frac{V_i + \varepsilon}{V_j + \varepsilon} \propto 0}
$$

**For small $\varepsilon$** ($\varepsilon \ll V_i, V_j$):

$$
S_i(j) \approx -S_j(i) \quad \text{(approximately antisymmetric)}
$$

**Exact antisymmetry:**

$$
\text{numerator of } S_i(j) = -(\text{numerator of } S_j(i))
$$

This is the **algorithmic signature of fermionic structure**.

**Gemini's validation:** "You have resolved the core of my original Issue #1. The antisymmetric structure is the correct dynamical signature of a fermionic system."

---

### Algorithmic Exclusion Principle

**Type:** Theorem
**Label:** `thm-algorithmic-exclusion`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 7.2. Algorithmic Exclusion Principle](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `fermionic`, `field-theory`, `fractal-set`, `lattice-qft`, `qft`

**Statement:**
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 5.1, lines 364-386] ✅ **GEMINI VALIDATED**

For any walker pair $(i, j)$:

**Case 1:** $V_i < V_j$ (i less fit)
- $S_i(j) > 0$ → Walker $i$ **can** clone from $j$
- $S_j(i) < 0$ → Walker $j$ **cannot** clone from $i$

**Case 2:** $V_i > V_j$ (j less fit)
- $S_i(j) < 0$ → Walker $i$ **cannot** clone from $j$
- $S_j(i) > 0$ → Walker $j$ **can** clone from $i$

**Case 3:** $V_i = V_j$ (equal fitness)
- $S_i(j) = 0$, $S_j(i) = 0$ → Neither clones

**Exclusion principle:** **At most one walker per pair can clone in any given direction.**

This is analogous to Pauli exclusion: "Two fermions cannot occupy the same state."

**Gemini's validation:** "The algorithmic exclusion principle is a strong analogue to the Pauli Exclusion Principle."

---

### Exclusion Requires Anticommuting Fields

**Type:** Theorem
**Label:** `thm-exclusion-anticommuting`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 7.2. Algorithmic Exclusion Principle](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `lattice-qft`, `metric-tensor`, `qft`

**Statement:**
**Source:** [13_D_fractal_set_emergent_qft_comprehensive.md § 5.2, lines 393-418]

The algorithmic exclusion principle **requires** anticommuting (Grassmann) field variables for correct path integral formulation.

**Argument:**

1. **Cloning event** $i \to j$: Represents transition amplitude
2. **Double counting problem:** Naively, both $i \to j$ and $j \to i$ are "possible transitions"
3. **Exclusion resolves it:** Only one direction allowed (determined by fitness comparison)
4. **Path integral:** To avoid overcounting, must use antisymmetric variables

**Grassmann variables:** $\psi_i, \psi_j$ with $\{\psi_i, \psi_j\} = 0$

**Amplitude for $i \to j$:**

$$
\mathcal{A}(i \to j) \propto \bar{\psi}_i S_i(j) \psi_j
$$

**Amplitude for $j \to i$:**

$$
\mathcal{A}(j \to i) \propto \bar{\psi}_j S_j(i) \psi_i = -\bar{\psi}_i S_j(i) \psi_j
$$

The anticommutation $\{\psi_i, \psi_j\} = 0$ **automatically** enforces exclusion.

---

### Graph Laplacian Converges to Laplace-Beltrami Operator

**Type:** Theorem
**Label:** `thm-laplacian-convergence-curved`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 8.2. Graph Laplacian Equals Laplace-Beltrami Operator](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `field-theory`, `fractal-set`, `laplacian`, `lattice`, `lattice-qft`, `metric-tensor`, `qft`, `qsd`, `riemannian`

**Statement:**
**Source:** [13_E_cst_ig_lattice_qft.md § 7.2b, lines 1027-1131]

Let $g(x, S)$ be the emergent Riemannian metric from Chapter 08:

$$
g(x, S) = H(x, S) + \varepsilon_{\Sigma} I
$$

where $H(x, S) = \nabla^2_x V_{\text{fit}}$ is the fitness Hessian.

In the continuum limit $\varepsilon_c \to 0$, $N \to \infty$ with scaling $\varepsilon_c \sim \sqrt{2 D_{\text{reg}} \tau}$:

$$
\lim_{\substack{\varepsilon_c \to 0 \\ N \to \infty}} (\Delta_{\text{graph}} \phi)(e_i)
= \Delta_{\text{LB}} \phi(x_i)
:= \frac{1}{\sqrt{\det g(x_i)}} \partial_{\mu} \left(\sqrt{\det g(x_i)} \, g^{\mu\nu}(x_i) \partial_{\nu} \phi(x_i)\right)
$$

In coordinates:

$$
\Delta_{\text{LB}} \phi = g^{\mu\nu} \nabla_{\mu} \nabla_{\nu} \phi
= g^{\mu\nu} \left[\partial_{\mu} \partial_{\nu} \phi - \Gamma^{\lambda}_{\mu\nu} \partial_{\lambda} \phi\right]
$$

where $\Gamma^{\lambda}_{\mu\nu}$ are the Christoffel symbols of metric $g$.

**Proof:** See [13_E_cst_ig_lattice_qft.md § 7.2b, lines 1056-1131] for complete derivation involving:
1. Taylor expansion of field
2. Weighted first moment (connection term from QSD density)
3. Weighted second moment (Laplacian term)
4. Scaling and continuum limit

**Conclusion:** The graph Laplacian on IG converges to the Laplace-Beltrami operator on the emergent Riemannian manifold. ∎

---

### Fragile Gas Framework Generates Complete Lattice QFT

**Type:** Theorem
**Label:** `thm-fragile-gas-generates-qft`
**Source:** [13_fractal_set_new/08_lattice_qft_framework.md § 9.2. Main Result: Emergent QFT from Algorithmic Dynamics](13_fractal_set_new/08_lattice_qft_framework.md)
**Tags:** `causal-tree`, `fermionic`, `field-theory`, `fractal-set`, `gauge-theory`, `laplacian`, `lattice`, `lattice-qft`, `metric-tensor`, `qft`, `riemannian`, `u1-symmetry`

**Statement:**
The Fragile Gas optimization dynamics (Chapters 1-12) naturally generates:

**1. Discrete spacetime:** CST+IG forms a causal set with:
- Timelike structure from genealogy (CST)
- Spacelike correlations from selection (IG)
- Emergent Riemannian geometry from fitness landscape

**2. Gauge fields:** Parallel transport on edges:
- Gauge group: $U(1)$, $SU(N)$, or $S_{|\mathcal{E}|}$
- Wilson loops: Gauge-invariant observables
- Field strength: Plaquette holonomy

**3. Fermions:** Antisymmetric cloning kernel:
- Exclusion principle from fitness comparison
- Grassmann fields on IG edges
- Dirac-like structure (conjectured continuum limit)

**4. Scalar fields:** Graph Laplacian = Laplace-Beltrami operator:
- Fitness potential as background field
- Discrete derivatives converge to Riemannian derivatives
- Correct continuum limit to curved-space QFT

**Physical significance:** This is the first **unified framework** connecting:
- Stochastic optimization (Chapters 1-7)
- Mean-field theory (Chapters 5-6)
- Emergent geometry (Chapter 8)
- Gauge theory (Chapter 12)
- Discrete spacetime (Chapter 13)
- Lattice QFT (this document)

**Novel capability:** Extract QFT observables directly from optimization runs—no external lattice design required.

---


## 09_geometric_algorithms.md

**Objects in this document:** 4

### Definitions (2)

### Empirical Metric from IG Edges

**Type:** Definition
**Label:** `def-empirical-metric`
**Source:** [13_fractal_set_new/09_geometric_algorithms.md § 3.1. Local Covariance Matrix Computation](13_fractal_set_new/09_geometric_algorithms.md)
**Tags:** `algorithms`, `fractal-set`, `geometric-algorithms`, `metric-tensor`

**Statement:**
For episode $e_i$ at position $x_i = \Phi(e_i)$, the **local covariance matrix** from IG neighbors is:

$$
\Sigma_i = \frac{1}{|\mathcal{N}_{\text{IG}}(e_i)|} \sum_{e_j \in \mathcal{N}_{\text{IG}}(e_i)} w_{ij} \cdot \Delta x_{ij} \Delta x_{ij}^T
$$

where:
- $\mathcal{N}_{\text{IG}}(e_i) = \{e_j : (e_i \sim e_j) \in E_{\text{IG}}\}$ (IG neighbors)
- $\Delta x_{ij} = \Phi(e_j) - \Phi(e_i)$ (displacement vector)
- $w_{ij} = 1 / ((\Delta t_{ij})^2 + \|\Delta x_{ij}\|^2)$ (algorithmic weight)

**Relationship to metric:**

$$
\Sigma_i^{-1} \approx G(x_i, S) + O(1/N)
$$

The **inverse covariance** approximates the metric tensor in the continuum limit.

---

### Parallel Transport on Fractal Set

**Type:** Definition
**Label:** `def-parallel-transport-fractal`
**Source:** [13_fractal_set_new/09_geometric_algorithms.md § 4.1. Path Tracing on CST+IG](13_fractal_set_new/09_geometric_algorithms.md)
**Tags:** `algorithms`, `fractal-set`, `gauge-theory`, `geometric-algorithms`

**Statement:**
For gauge field $U_e \in SU(N_c)$ on each edge $e$:

**CST edge** $e_p \to e_c$ (parent → child):
- Forward: $U(e_p, e_c)$
- Backward: $U(e_c, e_p) = U(e_p, e_c)^\dagger$

**IG edge** $e_i \sim e_j$ (undirected):
- Either direction: $U(e_i, e_j)$ or $U(e_j, e_i) = U(e_i, e_j)^\dagger$

**Path-ordered product** along path $P = \{e_0 \to e_1 \to \cdots \to e_n\}$:

$$
U(P) = U(e_{n-1}, e_n) \times U(e_{n-2}, e_{n-1}) \times \cdots \times U(e_0, e_1)
$$

(Rightmost operator acts first, consistent with quantum mechanics)

---

### Theorems (2)

### Fan Triangulation is Base-Independent

**Type:** Theorem
**Label:** `thm-fan-base-independence`
**Source:** [13_fractal_set_new/09_geometric_algorithms.md § 1.2. Proof of Base Independence](13_fractal_set_new/09_geometric_algorithms.md)
**Tags:** `algorithms`, `area`, `curvature`, `fractal-set`, `geometric-algorithms`, `metric-tensor`, `riemannian`

**Statement:**
The choice of base vertex (centroid vs. arbitrary vertex) does **not** affect the total area, up to $O(d^2)$ curvature corrections.

**Statement:** For cycle $C$ embedded in Riemannian manifold $(\mathcal{X}, G)$:

$$
A_{\text{centroid}}(C) \approx A_{\text{vertex}}(C) \quad \text{(error } O(R \cdot \text{diam}(C)^3) \text{)}
$$

where $R = \|\text{Riemann}(x)\|$ is the curvature scale.

**Proof Sketch:**

1. **Flat space:** Exact equality (shoelace formula independent of base)
2. **Curved space:** Metric varies as $G(x + \delta x) = G(x) + O(\|\nabla G\| \cdot \|\delta x\|)$
3. **Centroid minimizes** $\sum_i \|x - \Phi(e_i)\|^2$ → smallest metric variation
4. **Error bound:** Using Taylor expansion of $G$ around base point:

$$
|A_{\text{base}_1}(C) - A_{\text{base}_2}(C)| \leq C_d \cdot R \cdot \text{diam}(C)^3
$$

**Conclusion:** For small cycles (typical in IG), choice of base vertex negligible.

---

### IG Edges Close Fundamental Cycles

**Type:** Theorem
**Label:** `thm-ig-cycles`
**Source:** [13_fractal_set_new/09_geometric_algorithms.md § 2.1. Cycle Basis Construction](13_fractal_set_new/09_geometric_algorithms.md)
**Tags:** `algorithms`, `fractal-set`, `geometric-algorithms`

**Statement:**
**Assumption:** CST is a rooted spanning tree (single common ancestor)

**Claim:** For IG graph with $k$ edges $E_{\text{IG}} = \{e_1, \ldots, e_k\}$:

1. Each IG edge $e_i = (e_a \sim e_b)$ closes exactly one fundamental cycle $C(e_i)$
2. The cycles $\{C(e_1), \ldots, C(e_k)\}$ form a complete basis for the cycle space

**Construction:** For IG edge $e_i = (e_a \sim e_b)$:

$$
C(e_i) := e_i \cup P_{\text{CST}}(e_a, e_b)
$$

where $P_{\text{CST}}(e_a, e_b)$ is the unique path from $e_a$ to $e_b$ in the CST tree.

**Proof:**

*Part 1 (Unique path):* CST is tree → unique path between any two vertices

*Part 2 (Closed cycle):* $e_i$ connects $e_a \to e_b$ (IG), $P_{\text{CST}}$ connects $e_b \to e_a$ (CST) → closed loop

*Part 3 (Linear independence):* Each $C(e_i)$ contains IG edge $e_i$, no other cycle contains $e_i$ → independent

*Part 4 (Completeness):* Cycle space dimension = $|E_{\text{total}}| - |V| + 1 = (|E_{\text{CST}}| + k) - |V| + 1$. Since $|E_{\text{CST}}| = |V| - 1$ (tree property), dimension = $k$. We have $k$ independent cycles → complete basis. ∎

---


## 10_areas_volumes_integration.md

**Objects in this document:** 16

### Definitions (8)

### Riemannian Volume Element on Emergent Manifold

**Type:** Definition
**Label:** `def-riemannian-volume-element`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 1.1. Riemannian Volume Element](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `curvature`, `fractal-set`, `integration`, `metric-tensor`, `qsd`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $(\mathcal{X}, g)$ be the emergent Riemannian manifold from the Adaptive Gas, where:

- $\mathcal{X} \subseteq \mathbb{R}^d$ is the state space
- $g(x) = H(x) + \epsilon_\Sigma I$ is the emergent metric tensor (inverse diffusion)
- $H(x) = -\nabla^2 \Phi(x)$ is the fitness Hessian

The **Riemannian volume element** at point $x \in \mathcal{X}$ is:

$$
dV_g(x) := \sqrt{\det g(x)} \, dx
$$

where $dx = dx_1 \wedge \cdots \wedge dx_d$ is the Euclidean volume element.

**Physical interpretation:**
- $\sqrt{\det g(x)}$: Jacobian factor relating Euclidean to Riemannian volume
- Large $\sqrt{\det g}$: "Stretched" region (high curvature, hard to explore)
- Small $\sqrt{\det g}$: "Compressed" region (low curvature, easy to explore)

**Connection to QSD:** From {prf:ref}`thm-qsd-riemannian-volume-main` in {doc}`05_qsd_stratonovich_foundations.md`:

$$
\rho_{\text{spatial}}(x) \propto \sqrt{\det g(x)} \, e^{-U_{\text{eff}}(x)/T}
$$

Episodes naturally sample from this measure.

**Source:** Standard differential geometry. See Lee (2018) *Introduction to Riemannian Manifolds*, Chapter 6.

**Related Results:** `thm-qsd-riemannian-volume-main`

---

### Induced Volume Form on k-Dimensional Submanifold

**Type:** Definition
**Label:** `def-induced-volume-form`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 1.2. Submanifold Volume Forms](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $\Sigma \subset \mathcal{X}$ be a $k$-dimensional submanifold embedded in the $d$-dimensional Riemannian manifold $(\mathcal{X}, g)$.

**Parametrization:** Let $\varphi: U \subset \mathbb{R}^k \to \Sigma \subset \mathcal{X}$ be a smooth parametrization with $\varphi(u) = x(u_1, \ldots, u_k)$.

**Tangent vectors:**

$$
\frac{\partial x}{\partial u_i} =: \partial_i x \in T_x \mathcal{X}, \quad i = 1, \ldots, k
$$

**Induced metric tensor:** The **pullback metric** $g_\Sigma$ on $\Sigma$ is:

$$
(g_\Sigma)_{ij}(u) := g\left(\partial_i x, \partial_j x\right) = (\partial_i x)^T g(x(u)) (\partial_j x)
$$

**Induced volume form:**

$$
dV_{g_\Sigma}(u) := \sqrt{\det g_\Sigma(u)} \, du_1 \wedge \cdots \wedge du_k
$$

**Riemannian volume of $\Sigma$:**

$$
\text{Vol}_g(\Sigma) := \int_U \sqrt{\det g_\Sigma(u)} \, du
$$

**Special cases:**
- $k=1$ (curve): Arc length $L = \int \sqrt{g(\dot{\gamma}, \dot{\gamma})} \, dt$
- $k=2$ (surface): Area $A = \int \sqrt{\det g_\Sigma} \, du \, dv$
- $k=3$ (3D region): Volume $V = \int \sqrt{\det g_\Sigma} \, du \, dv \, dw$
- $k=d$ (full manifold): Total volume $\text{Vol}_g(\mathcal{X}) = \int \sqrt{\det g} \, dx$

**Source:** Lee (2018) Chapter 10; Spivak (1979) *Differential Geometry Vol. 3*.

---

### Discrete Approximation of Riemannian Integrals

**Type:** Definition
**Label:** `def-discrete-riemannian-integration`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 1.3. Discrete Approximation via Episode Sampling](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `convergence`, `fractal-set`, `integration`, `qsd`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $\{e_i\}_{i=1}^N$ be episodes with positions $x_i = \Phi(e_i) \in \mathcal{X}$ sampled from the QSD.

For a continuous function $f: \mathcal{X} \to \mathbb{R}$, the **discrete approximation** of the Riemannian integral is:

$$
\int_{\mathcal{X}} f(x) \, dV_g(x) \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \cdot w_{\text{vol}}(x_i)
$$

where $w_{\text{vol}}(x_i)$ is a **volume weight** accounting for local sampling density.

**Two methods for volume weights:**

**Method 1: Uniform weighting (if episodes sample from $\sqrt{\det g}$)**

If $\rho_{\text{episodes}}(x) \propto \sqrt{\det g(x)}$ (proven in {prf:ref}`thm-qsd-riemannian-volume-main`):

$$
w_{\text{vol}}(x_i) = 1
$$

The factor $\sqrt{\det g}$ is **already incorporated** in the sampling density.

**Method 2: Explicit reweighting (for non-QSD distributions)**

If episodes sample from arbitrary density $\rho(x)$:

$$
w_{\text{vol}}(x_i) = \frac{\sqrt{\det g(x_i)}}{\rho(x_i)}
$$

Importance sampling correction.

**Convergence:** As $N \to \infty$:

$$
\frac{1}{N} \sum_{i=1}^N f(x_i) \xrightarrow{a.s.} \int_{\mathcal{X}} f(x) \, \rho(x) \, dx
$$

By strong law of large numbers. If $\rho \propto \sqrt{\det g}$, this gives the Riemannian integral.

**Source:** Standard Monte Carlo integration. See Robert & Casella (2004) *Monte Carlo Statistical Methods*, Chapter 3.

**Related Results:** `thm-qsd-riemannian-volume-main`

---

### Plaquette Area in Lattice Gauge Theory

**Type:** Definition
**Label:** `def-plaquette-area-gauge`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 2.2. Riemannian Area for Plaquettes in Lattice QFT](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `fractal-set`, `gauge-theory`, `integration`, `lattice`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
In the lattice QFT framework ({doc}`08_lattice_qft_framework.md`), a **plaquette** $P$ is an elementary closed loop formed by:

$$
P = (e_{i_0} \xrightarrow{\text{CST}} e_{i_1} \xleftarrow{\text{IG}} e_{j_1} \xleftarrow{\text{CST}} e_{j_0} \xrightarrow{\text{IG}} e_{i_0})
$$

**Four vertices:**
- $x_{i_0}, x_{i_1}, x_{j_1}, x_{j_0}$ (positions of the four episodes)

**Riemannian area:** Apply fan triangulation:

$$
A_g(P) = \text{FanTriangulation}([x_{i_0}, x_{i_1}, x_{j_1}, x_{j_0}], g)
$$

**Use in Wilson action:** From {prf:ref}`def-wilson-action-lattice` in {doc}`08_lattice_qft_framework.md`:

$$
S_{\text{Wilson}}[\mathcal{F}] = \sum_{P \in \text{Plaquettes}} \left(1 - \text{Re}[W[P]]\right) \cdot A_g(P)
$$

where $W[P]$ is the Wilson loop around plaquette $P$.

**Physical interpretation:**
- Larger $A_g(P)$: Stronger penalty for non-trivial holonomy
- Mimics continuum Yang-Mills action $\sim \int \text{Tr}[F_{\mu\nu}^2] \sqrt{\det g} \, d^4x$

**Related Results:** `def-wilson-action-lattice`

---

### Riemannian Volume of Tetrahedron

**Type:** Definition
**Label:** `def-tetrahedron-volume`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 3.1. Cayley-Menger Determinant for Tetrahedron Volume](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $T = (x_0, x_1, x_2, x_3)$ be a tetrahedron with vertices $x_i \in \mathbb{R}^d$ (typically $d \geq 3$).

**Metric tensor:** Evaluate at centroid:

$$
g_c := g\left(\frac{x_0 + x_1 + x_2 + x_3}{4}\right)
$$

**Edge vectors from base vertex $x_0$:**

$$
v_1 := x_1 - x_0, \quad v_2 := x_2 - x_0, \quad v_3 := x_3 - x_0
$$

**Gram matrix:** $3 \times 3$ matrix of Riemannian inner products:

$$
G := \begin{pmatrix}
\langle v_1, v_1 \rangle_g & \langle v_1, v_2 \rangle_g & \langle v_1, v_3 \rangle_g \\
\langle v_2, v_1 \rangle_g & \langle v_2, v_2 \rangle_g & \langle v_2, v_3 \rangle_g \\
\langle v_3, v_1 \rangle_g & \langle v_3, v_2 \rangle_g & \langle v_3, v_3 \rangle_g
\end{pmatrix}
$$

where $\langle v_i, v_j \rangle_g := v_i^T g_c v_j$.

**Riemannian volume:**

$$
V_g(T) := \frac{1}{6} \sqrt{\det G}
$$

**Geometric interpretation:** Generalization of $\frac{1}{6} |v_1 \cdot (v_2 \times v_3)|$ from Euclidean space to Riemannian manifold.

**Error:** If $\text{diam}(T) = O(\epsilon)$, then $V_g(T) = V_{\text{true}} + O(\epsilon^4)$ (assuming $g$ is $C^3$).

**Source:** Standard Riemannian geometry. See do Carmo (1992) *Riemannian Geometry*, Chapter 9.

---

### Riemannian Volume of d-Simplex

**Type:** Definition
**Label:** `def-d-simplex-volume`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 4.1. Simplex Volume in Arbitrary Dimension](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $S = (x_0, x_1, \ldots, x_d)$ be a $d$-simplex in $\mathbb{R}^d$ with vertices $x_i \in \mathcal{X}$.

**Metric at centroid:**

$$
g_c := g\left(\frac{1}{d+1} \sum_{i=0}^d x_i\right)
$$

**Edge vectors:**

$$
v_i := x_i - x_0 \quad \text{for } i = 1, \ldots, d
$$

**Gram matrix:** $d \times d$ matrix:

$$
G_{ij} := \langle v_i, v_j \rangle_g = v_i^T g_c v_j
$$

**Riemannian volume:**

$$
V_g(S) := \frac{1}{d!} \sqrt{\det G}
$$

**Special cases:**
- $d=1$: Length $L = \sqrt{\langle v_1, v_1 \rangle_g}$
- $d=2$: Area $A = \frac{1}{2} \sqrt{\det G}$ (triangle)
- $d=3$: Volume $V = \frac{1}{6} \sqrt{\det G}$ (tetrahedron)
- $d=4$: Hypervolume $V = \frac{1}{24} \sqrt{\det G}$ (pentatope)

**Generalization:** Works for any $d \leq \dim(\mathcal{X})$.

---

### Surface Integral on Riemannian Manifold

**Type:** Definition
**Label:** `def-surface-integral-scalar`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 8.1. Surface Integrals of Scalar Fields](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $\Sigma \subset (\mathcal{X}, g)$ be a 2D surface (submanifold) embedded in the $d$-dimensional Riemannian manifold $(\mathcal{X}, g)$.

Let $f: \Sigma \to \mathbb{R}$ be a continuous scalar function defined on $\Sigma$.

The **surface integral** of $f$ over $\Sigma$ is:

$$
\iint_\Sigma f \, dS_g := \int_\Sigma f(x) \, dV_{g_\Sigma}(x)
$$

where $dV_{g_\Sigma}$ is the induced Riemannian volume element on $\Sigma$ from {prf:ref}`def-induced-volume-form`.

**Parametric form:** If $\Sigma$ is parametrized by $\varphi: U \subset \mathbb{R}^2 \to \Sigma$ with $\varphi(u,v) = x(u,v)$:

$$
\iint_\Sigma f \, dS_g = \int_U f(x(u,v)) \sqrt{\det g_\Sigma(u,v)} \, du \, dv
$$

where the induced metric is:

$$
g_\Sigma = \begin{pmatrix}
\langle \partial_u x, \partial_u x \rangle_g & \langle \partial_u x, \partial_v x \rangle_g \\
\langle \partial_v x, \partial_u x \rangle_g & \langle \partial_v x, \partial_v x \rangle_g
\end{pmatrix}
$$

**Discrete approximation:** For a triangulated surface $\Sigma \approx \bigcup_i T_i$ (triangles):

$$
\iint_\Sigma f \, dS_g \approx \sum_i f(x_{c_i}) \cdot A_g(T_i)
$$

where $x_{c_i}$ is the centroid of triangle $T_i$ and $A_g(T_i)$ is computed via fan triangulation.

**Source:** Standard surface integration. See Lee (2018) Chapter 16.

**Related Results:** `def-induced-volume-form`

---

### Flux of Vector Field Through Surface

**Type:** Definition
**Label:** `def-flux-vector-field`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 8.2. Vector Field Flux Through Surfaces](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
Let $\Sigma \subset (\mathcal{X}, g)$ be an oriented 2D surface with **unit normal vector field** $\mathbf{n}: \Sigma \to T\mathcal{X}$ (where $\|\mathbf{n}\|_g = 1$).

Let $\mathbf{F}: \mathcal{X} \to T\mathcal{X}$ be a vector field.

The **flux** of $\mathbf{F}$ through $\Sigma$ is:

$$
\Phi[\mathbf{F}, \Sigma] := \iint_\Sigma \langle \mathbf{F}, \mathbf{n} \rangle_g \, dS_g
$$

where $\langle \cdot, \cdot \rangle_g$ is the Riemannian inner product: $\langle \mathbf{F}, \mathbf{n} \rangle_g = \mathbf{F}^T g \mathbf{n}$.

**Physical interpretation:**
- $\Phi > 0$: Net flow of $\mathbf{F}$ outward through $\Sigma$
- $\Phi < 0$: Net flow inward
- $\Phi = 0$: No net flux (source-free or balanced)

**Parametric form:** For parametrization $x(u,v)$:

$$
\mathbf{n} = \frac{(\partial_u x) \times_g (\partial_v x)}{\|(\partial_u x) \times_g (\partial_v x)\|_g}
$$

where $\times_g$ is the Riemannian cross product (requires $d \geq 3$).

**Simplified for embedded surface in $\mathbb{R}^3$:**

If $\Sigma \subset \mathbb{R}^3$ and $g(x) \approx I$ (nearly Euclidean):

$$
\mathbf{n} \approx \frac{(\partial_u x) \times (\partial_v x)}{\|(\partial_u x) \times (\partial_v x)\|}
$$

Standard Euclidean cross product.

**Source:** Vector calculus on manifolds. See Marsden & Tromba (2011) *Vector Calculus*, Chapter 7.

---

### Propositions (3)

### Curvature from Area Defect

**Type:** Proposition
**Label:** `prop-curvature-from-area-defect`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 2.3. Euclidean vs Riemannian Area Comparison](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `curvature`, `fractal-set`, `integration`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
For a plaquette $P$ with Euclidean area $A_{\text{Euclid}}(P)$ and Riemannian area $A_g(P)$, define the **area ratio**:

$$
r_A(P) := \frac{A_g(P)}{A_{\text{Euclid}}(P)}
$$

Then:

$$
r_A(P) \approx 1 + \frac{1}{6} R_{\text{scalar}} \cdot A_{\text{Euclid}}(P) + O(A^2)
$$

where $R_{\text{scalar}}$ is the scalar curvature at the plaquette centroid.

**Interpretation:**
- $r_A > 1$: Positive curvature (sphere-like, "stretched")
- $r_A < 1$: Negative curvature (hyperbolic, "compressed")
- $r_A \approx 1$: Flat (Euclidean geometry)

**Use:** Measure curvature empirically by computing $r_A$ for plaquettes.

**Source:** Regge calculus. See Barrett et al. (2009) "Tullio Regge's Legacy: Regge Calculus and Discrete Gravity".

---

### Cayley-Menger Determinant Formula

**Type:** Proposition
**Label:** `prop-cayley-menger-volume`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 3.1. Cayley-Menger Determinant for Tetrahedron Volume](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `fractal-set`, `integration`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
An alternative formula using pairwise Riemannian distances:

Define **Riemannian distance matrix** $D \in \mathbb{R}^{4 \times 4}$:

$$
D_{ij} := \sqrt{(x_i - x_j)^T g_c (x_i - x_j)} \quad \text{for } i, j = 0, 1, 2, 3
$$

The **Cayley-Menger determinant** is:

$$
\text{CM}(T) := \begin{vmatrix}
0 & 1 & 1 & 1 & 1 \\
1 & 0 & D_{01}^2 & D_{02}^2 & D_{03}^2 \\
1 & D_{10}^2 & 0 & D_{12}^2 & D_{13}^2 \\
1 & D_{20}^2 & D_{21}^2 & 0 & D_{23}^2 \\
1 & D_{30}^2 & D_{31}^2 & D_{32}^2 & 0
\end{vmatrix}
$$

Then:

$$
V_g(T) = \frac{1}{6\sqrt{2}} \sqrt{|\text{CM}(T)|}
$$

**Advantage:** Only requires pairwise distances, not full coordinate representation.

**Disadvantage:** Numerically less stable than Gram matrix method for high dimensions.

**Source:** Blumenthal (1970) *Theory and Applications of Distance Geometry*, Chapter 4.

---

### Monte Carlo Integration with Riemannian Measure

**Type:** Proposition
**Label:** `prop-monte-carlo-riemannian`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 4.2. Direct Integration Over Episode Distribution](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `convergence`, `fractal-set`, `integration`, `qsd`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
Let $f: \mathcal{X} \to \mathbb{R}$ be a continuous function. The **Riemannian integral** is:

$$
I[f] := \int_{\mathcal{X}} f(x) \, dV_g(x) = \int_{\mathcal{X}} f(x) \sqrt{\det g(x)} \, dx
$$

**Method 1: Direct Monte Carlo (if episodes sample from $\sqrt{\det g}$)**

If $\{x_i\}_{i=1}^N$ are sampled i.i.d. from $\rho(x) \propto \sqrt{\det g(x)} e^{-U_{\text{eff}}/T}$ (QSD):

$$
I[f] \approx \frac{Z}{N} \sum_{i=1}^N f(x_i)
$$

where $Z = \int \sqrt{\det g} e^{-U_{\text{eff}}/T} dx$ is unknown but can be estimated.

**Method 2: Importance Sampling Correction**

If episodes sample from arbitrary $\rho(x) \neq \sqrt{\det g}$:

$$
I[f] = \int f(x) \frac{\sqrt{\det g(x)}}{\rho(x)} \rho(x) \, dx \approx \frac{1}{N} \sum_{i=1}^N f(x_i) \frac{\sqrt{\det g(x_i)}}{\rho(x_i)}
$$

**Variance:**

$$
\text{Var}\left[\frac{1}{N} \sum f(x_i)\right] = \frac{1}{N} \text{Var}_\rho[f] = O(N^{-1})
$$

Standard Monte Carlo convergence rate.

**Adaptive importance sampling:** Choose $\rho(x) \propto |f(x)| \sqrt{\det g(x)}$ to minimize variance (not always feasible).

---

### Theorems (5)

### Fan Triangulation Gives Correct Riemannian Area

**Type:** Theorem
**Label:** `thm-fan-triangulation-correct`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 2.1. Fan Triangulation Algorithm](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `curvature`, `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
Let $C = (x_0, \ldots, x_{n-1}, x_0)$ be a simple closed curve in $(\mathcal{X}, g)$ bounding a simply-connected region $\Sigma \subset \mathcal{X}$.

Assume:
1. $\Sigma$ is nearly planar (small geodesic curvature)
2. $\text{diam}(\Sigma) \ll R_g$ where $R_g$ is the radius of curvature of $(\mathcal{X}, g)$
3. Metric $g$ is $C^2$ on $\Sigma$

Then the fan triangulation algorithm {prf:ref}`alg-fan-triangulation-area` computes:

$$
A_g(C) = \text{Vol}_g(\Sigma) + O(\text{diam}(\Sigma)^3)
$$

where $\text{Vol}_g(\Sigma) = \int_\Sigma dV_{g_\Sigma}$ is the true Riemannian area.

**Proof idea:**
1. Each triangle area $A_i$ approximates the Riemannian area of the corresponding region
2. Using $g_c$ (metric at centroid) instead of variable $g(x)$ introduces error $O(\|\nabla g\| \cdot \text{diam}^2)$
3. Summing $n$ triangles accumulates error $O(n \cdot \text{diam}^3)$
4. For fixed $\text{diam}(\Sigma)$, increasing $n$ improves accuracy

**Related Results:** `alg-fan-triangulation-area`

---

### Fan Triangulation Error Bound

**Type:** Theorem
**Label:** `thm-fan-triangulation-error`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 6.1. Discretization Error for Fan Triangulation](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `curvature`, `fractal-set`, `integration`, `metric-tensor`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
Let $\Sigma \subset (\mathcal{X}, g)$ be a smooth 2D surface with true Riemannian area $A_{\text{true}}$.

Let $C = (x_0, \ldots, x_{n-1})$ be a polygonal approximation of $\partial \Sigma$ with $n$ vertices and maximum edge length $h := \max_i \|x_{i+1} - x_i\|$.

Assume:
1. Metric $g \in C^3(\Sigma)$
2. Hausdorff distance $d_H(C, \partial \Sigma) \leq h$
3. Surface curvature bounded: $\|K\|_\infty \leq K_{\max}$

Then the fan triangulation area $A_{\text{fan}}$ satisfies:

$$
|A_{\text{fan}} - A_{\text{true}}| \leq C \cdot h^2 \cdot (1 + K_{\max} A_{\text{true}})
$$

where $C$ depends on $\|g\|_{C^3}$ and $n$.

**Proof sketch:**
1. Each triangle introduces error from replacing $g(x)$ with $g(x_c)$: $O(h^2 \|\nabla g\|)$
2. Curvature of $\Sigma$ causes geometric error: $O(h^2 K_{\max})$
3. Summing $n \sim h^{-1}$ triangles gives total error $O(h^2)$

**Practical implication:** To achieve accuracy $\epsilon$, need $h = O(\sqrt{\epsilon})$ ⇒ $n = O(\epsilon^{-1/2})$ vertices.

---

### Monte Carlo Error for Riemannian Integrals

**Type:** Theorem
**Label:** `thm-monte-carlo-error-riemannian`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 6.2. Monte Carlo Integration Error](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `convergence`, `fractal-set`, `integration`, `riemannian`, `riemannian-geometry`, `volumes`

**Statement:**
Let $I[f] = \int f(x) \sqrt{\det g(x)} \, dx$ and let $\{x_i\}_{i=1}^N$ be i.i.d. samples from $\rho \propto \sqrt{\det g} e^{-U/T}$.

The Monte Carlo estimator is:

$$
\hat{I}_N[f] := \frac{Z}{N} \sum_{i=1}^N f(x_i)
$$

where $Z$ is estimated separately.

**Error bound:** With probability $\geq 1 - \delta$:

$$
|I[f] - \hat{I}_N[f]| \leq \frac{\sigma[f]}{\sqrt{N}} \cdot t_{\alpha/2}
$$

where:
- $\sigma[f] := \sqrt{\text{Var}_\rho[f]}$ is standard deviation
- $t_{\alpha/2}$ is critical value for confidence level $1-\delta$

**Convergence rate:** $O(N^{-1/2})$ regardless of dimension (Monte Carlo advantage)

**Variance reduction:** Use stratified sampling or control variates to reduce $\sigma[f]$.

---

### Discrete Divergence Theorem

**Type:** Theorem
**Label:** `thm-discrete-divergence-theorem`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 8.3. Divergence Theorem on Fractal Set](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `fractal-set`, `integration`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
Let $\Omega \subset \mathcal{X}$ be a 3D region with smooth boundary $\partial \Omega = \Sigma$ (closed surface).

Let $\mathbf{F}: \mathcal{X} \to T\mathcal{X}$ be a smooth vector field.

The **divergence theorem** (Gauss's theorem) states:

$$
\iiint_\Omega (\nabla_g \cdot \mathbf{F}) \, dV_g = \iint_{\partial \Omega} \langle \mathbf{F}, \mathbf{n} \rangle_g \, dS_g
$$

where:
- $\nabla_g \cdot \mathbf{F} = \frac{1}{\sqrt{\det g}} \partial_i(\sqrt{\det g} \, F^i)$ is the **Riemannian divergence**
- $\mathbf{n}$ is the outward-pointing unit normal to $\partial \Omega$

**Discrete approximation:**

1. **Volume integral (LHS):**

   Tetrahedralize $\Omega = \bigcup_k T_k$:

   $$
   \iiint_\Omega (\nabla_g \cdot \mathbf{F}) \, dV_g \approx \sum_k (\nabla_g \cdot \mathbf{F})(x_{c_k}) \cdot V_g(T_k)
   $$

2. **Surface integral (RHS):**

   Triangulate $\partial \Omega$:

   $$
   \iint_{\partial \Omega} \langle \mathbf{F}, \mathbf{n} \rangle_g \, dS_g \approx \sum_i \langle \mathbf{F}(x_{c_i}), \mathbf{n}_i \rangle_g \cdot A_g(T_i)
   $$

**Validation test:** Compute both sides independently and check:

$$
\left| \text{LHS} - \text{RHS} \right| / \text{max}(|\text{LHS}|, |\text{RHS}|) < \epsilon_{\text{tol}}
$$

This tests both:
- Correctness of volume/area algorithms
- Consistency of divergence computation

**Source:** Standard theorem in Riemannian geometry. See Lee (2018) Chapter 16; Spivak (1979) Vol. 5.

---

### Complete Framework for Riemannian Volumes on Fractal Set

**Type:** Theorem
**Label:** `thm-complete-volume-framework`
**Source:** [13_fractal_set_new/10_areas_volumes_integration.md § 9.1. Main Results](13_fractal_set_new/10_areas_volumes_integration.md)
**Tags:** `area`, `convergence`, `curvature`, `fractal-set`, `integration`, `qsd`, `riemannian`, `riemannian-geometry`, `volume`, `volumes`

**Statement:**
The following computational framework is established:

**1. Two-dimensional areas (plaquettes, surfaces):**
- Algorithm: Fan triangulation {prf:ref}`alg-fan-triangulation-area`
- Error: $O(h^2)$ for mesh size $h$
- Use: Wilson loop action, curvature estimation

**2. Three-dimensional volumes (regions, cells):**
- Algorithm: Delaunay tetrahedral decomposition {prf:ref}`alg-tetrahedral-volume`
- Formula: Gram determinant for tetrahedra
- Use: Phase space volume, entropy

**3. General d-dimensional volumes:**
- Algorithm: Simplicial decomposition {prf:ref}`alg-simplicial-decomposition`
- Formula: $V_g(S) = \frac{1}{d!} \sqrt{\det G}$ for d-simplex
- Use: High-dimensional state spaces

**4. Monte Carlo integration:**
- Direct integration if episodes sample from $\sqrt{\det g}$ (QSD)
- Importance sampling correction for arbitrary distributions
- Convergence: $O(N^{-1/2})$ regardless of dimension

All algorithms have working Python implementations with numerical stability guarantees.

**Related Results:** `alg-tetrahedral-volume`, `alg-simplicial-decomposition`, `alg-fan-triangulation-area`

---

