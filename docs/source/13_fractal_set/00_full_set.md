# The Fractal Set: Complete Data Structure Specification

**Document purpose.** This document provides a complete specification of the **Fractal Set** $\mathcal{F}$ as a computational representation of the **Adaptive Viscous Fluid Model** from [07_adaptative_gas.md](../07_adaptative_gas.md) at **maximum temporal granularity**. Each node represents a **single walker at a single timestep**, and edges represent **transitions between timesteps** or **selection coupling**.

**Key principle:** The Fractal Set is built at the **finest possible resolution** - one node per walker per timestep - making it a **perfect fossil record** with no temporal aggregation.

**Covariance principle:** The data structure is designed to be **frame-independent**:
- **Nodes store scalars**: Frame-invariant quantities (fitness, energy, localized statistics)
- **Edges store spinors**: Frame-covariant vectorial quantities representing transitions (velocity spinors, force spinors, displacements)

This ensures the Fractal Set represents the **intrinsic geometry** of the algorithm without dependence on a preferred coordinate system or reference frame.

---

## 1. Nodes: Spacetime Points with Scalar Data

### 1.1. Node Definition

:::{prf:definition} Spacetime Node
:label: def-node-spacetime

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
:::

### 1.2. Node Scalar Data Structure

:::{prf:definition} Node Scalar Attributes
:label: def-node-attributes

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
:::

**Remark on covariance.** All quantities in nodes are **scalars** under coordinate transformations:
- Energies, fitness, and statistical moments are frame-invariant
- Norms like $\|v\|$ are scalar invariants derived from vector spinors
- Parameters $\epsilon_F, \nu, \gamma, \rho, \epsilon_\Sigma$ are global constants

---

## 2. CST Edges: Temporal Evolution with Spinor Data

### 2.1. CST Edge Definition

:::{prf:definition} Causal Spacetime Tree (CST) Edges
:label: def-cst-edges

The **CST edge set** encodes temporal transitions between consecutive timesteps:

$$
E_{\text{CST}} := \{(n_{i,t}, n_{i,t+1}) : i \in \{1, \ldots, N\}, \, t \in \{0, \ldots, T-1\}, \, s(n_{i,t}) = 1\},

$$

where each edge represents the continuous evolution of a single walker from timestep $t$ to $t+1$ (alive walkers only).

Each edge $(n_{i,t}, n_{i,t+1})$ stores **spinor data** representing the frame-covariant vectorial transition between the source and target nodes according to the Adaptive Gas SDE.
:::

### 2.2. Spinors: Frame-Covariant Vector Representation

:::{prf:definition} Spinor Representation
:label: def-spinor-representation

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
:::

**Remark.** In 3D, spinors are elements of $\mathbb{C}^2$ (Pauli spinors). In 2D, they are elements of $\mathbb{C}$ (complex phase). The dimension scales as $2^{[d/2]}$ for general $d$.

### 2.3. CST Edge Spinor Data Structure

:::{prf:definition} CST Edge Spinor Attributes
:label: def-cst-edge-attributes

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
:::

**Remark on covariance.** The CST edge stores:
- **Spinors** for all vectorial quantities (velocities, forces, displacements)
- **Scalars** for derived norms, angles, energies, and fitness values
- This makes the edge data **covariant**: it transforms correctly under coordinate changes

---

## 3. IG Edges: Asymmetric Selection Coupling with Spinor Data

### 3.1. IG Edge Definition

:::{prf:definition} Information Graph (IG) Edges (Directed)
:label: def-ig-edges

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
:::

### 3.2. Antisymmetric Cloning Potential

:::{prf:definition} Directed Cloning Potential
:label: def-cloning-potential

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
:::

### 3.3. IG Edge Spinor Data Structure

:::{prf:definition} IG Edge Spinor Attributes (Directed)
:label: def-ig-edge-attributes

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
:::

**Remark on antisymmetry and directed graphs.** The IG edge structure is:
- **Directed**: $(n_{i,t}, n_{j,t})$ and $(n_{j,t}, n_{i,t})$ are **distinct edges** with opposite potentials
- **Antisymmetric**: $V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)$ (fitness sign flip under exchange)
- **Complete tournament**: For $k$ alive walkers, there are $k(k-1)$ directed IG edges (not $\binom{k}{2}$)
- **Fermionic precursor**: This is the **key structural property** for deriving fermionic field theory

**Remark on covariance.** The IG edge stores:
- **Spinors** for positions, velocities, relative positions, viscous forces
- **Scalars** for distances, weights, fitness differences, and **antisymmetric cloning potential**
- The spinor representation makes the IG edges **geometrically intrinsic** to the manifold structure

---

## 4. The Fractal Set

### 4.1. Fractal Set Definition

:::{prf:definition} The Fractal Set
:label: def-fractal-set

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
:::

### 4.2. Completeness Property

:::{prf:theorem} Reconstruction Theorem: The Fractal Set as Complete Algorithm Representation
:label: thm-fractal-set-reconstruction

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
:::

**Corollary.** The Fractal Set is **lossless** with respect to Adaptive Gas dynamics (modulo random noise realizations) and **frame-independent**: reconstructions in any coordinate system yield equivalent physics.

---

## 5. Data Structure Summary Tables

### 5.1. Node Data Structure (Scalars Only)

**Table 1: Node Scalar Attributes (frame-invariant, from Adaptive Gas SDE)**

| **Category** | **Field** | **Type** | **Description** | **SDE Source** |
|--------------|-----------|----------|-----------------|----------------|
| **Identity** | `walker_id` | `int` | Walker index $i$ | - |
| | `timestep` | `int` | Discrete timestep $t$ | - |
| | `node_id` | `int` | Global ID: $i + t \cdot N$ | - |
| **Temporal** | `t` | `float` | Continuous time: $t \cdot \Delta t$ | - |
| **Status** | `s` | `{0,1}` | Survival status (alive in $A_k$) | Def. 1.0.3 |
| **Energy** | `E_kin` | `float` | Kinetic energy $\frac{1}{2}\|v\|^2$ | From $v_i$ |
| | `U` | `float` | Confining potential $U(x)$ | Eq. (2.1) |
| **Fitness** | `Phi` | `float` | Fitness $\Phi(x)$ | Def. 2.1 |
| **Virtual Reward** | `V_fit` | `float` | $V_{\text{fit}}[f_k,\rho](x)$ | Def. 2.1 |
| **Localized Stats** | `mu_rho` | `float` | $\mu_\rho[f_k,\Phi,x]$ | Def. 1.0.3 |
| | `sigma_rho` | `float` | $\sigma_\rho[f_k,\Phi,x]$ | Def. 1.0.3 |
| | `sigma_prime_rho` | `float` | $\sigma'_\rho[f_k,\Phi,x]$ (C¹ patched) | Def. 1.0.4 |
| | `Z_rho` | `float` | $Z_\rho[f_k,\Phi,x]$ | Def. 1.0.4 |
| **Global Params** | `eps_F` | `float` | Adaptive force strength $\epsilon_F$ | Eq. (2.2) |
| | `nu` | `float` | Viscosity coefficient $\nu$ | Eq. (2.3) |
| | `gamma` | `float` | Friction coefficient $\gamma$ | Eq. (2.4) |
| | `rho` | `float` | Localization scale $\rho$ | Def. 1.0.2 |
| | `eps_Sigma` | `float` | Diffusion regularization $\epsilon_\Sigma$ | Def. 2.5 |

### 5.2. CST Edge Data Structure (Spinors + Scalars)

**Table 2: CST Edge Attributes (frame-covariant, from Adaptive Gas SDE)**

| **Category** | **Field** | **Type** | **Description** | **SDE Source** |
|--------------|-----------|----------|-----------------|----------------|
| **Identity** | `walker_id` | `int` | Walker ID | - |
| | `timestep` | `int` | Transition timestep $t \to t+1$ | - |
| **Velocity Spinors** | `psi_v_t` | `Spinor` | Velocity spinor at $t$ | From $v_i(t)$ |
| | `psi_v_t+1` | `Spinor` | Velocity spinor at $t+1$ | From $v_i(t+1)$ |
| | `psi_Delta_v` | `Spinor` | Velocity increment | $dv_i$ |
| **Position Spinor** | `psi_Delta_x` | `Spinor` | Displacement | $dx_i = v_i dt$ |
| **Force Spinors** | `psi_F_stable` | `Spinor` | $\mathbf{F}_{\text{stable}}(x_t)$ | Eq. (2.1) |
| | `psi_F_adapt` | `Spinor` | $\mathbf{F}_{\text{adapt}}(x_t,S)$ | Eq. (2.2) |
| | `psi_F_viscous` | `Spinor` | $\mathbf{F}_{\text{viscous}}(x_t,S)$ | Eq. (2.3) |
| | `psi_F_friction` | `Spinor` | $-\gamma v_t$ | Eq. (2.4) |
| | `psi_F_total` | `Spinor` | Total drift | Sum of above |
| **Diffusion** | `psi_Sigma_reg` | `Tensor` | $\Sigma_{\text{reg}}(x_t,S)$ | Def. 2.5 |
| | `psi_noise` | `Spinor` | $\Sigma_{\text{reg}} \circ dW_t$ | Stratonovich |
| **Gradient Spinors** | `psi_grad_U` | `Spinor` | $\nabla U(x_t)$ | From $U$ |
| | `psi_grad_Phi` | `Spinor` | $\nabla \Phi(x_t)$ | From $\Phi$ |
| | `psi_grad_V_fit` | `Spinor` | $\nabla V_{\text{fit}}[f_k,\rho](x_t)$ | Appendix A |
| **Derived Scalars** | `norm_Delta_v` | `float` | $\|\Delta v\|$ | From spinor norm |
| | `norm_Delta_x` | `float` | $\|\Delta x\|$ | From spinor norm |
| | `dt` | `float` | Timestep size $\Delta t$ | Global constant |

### 5.3. IG Edge Data Structure (Directed, Spinors + Scalars)

**Table 3: IG Edge Attributes (directed, frame-covariant, from Adaptive Gas SDE)**

| **Category** | **Field** | **Type** | **Description** | **SDE Source** |
|--------------|-----------|----------|-----------------|----------------|
| **Identity** | `source_walker` | `int` | Source walker $j$ | - |
| | `target_walker` | `int` | Target walker $i$ | - |
| | `timestep` | `int` | Shared timestep $t$ | - |
| **Spatial Spinors** | `psi_x_i` | `Spinor` | Position spinor of $i$ | From $x_i(t)$ |
| | `psi_x_j` | `Spinor` | Position spinor of $j$ | From $x_j(t)$ |
| | `psi_Delta_x_ij` | `Spinor` | Relative position $x_j - x_i$ | Computed |
| **Velocity Spinors** | `psi_v_i` | `Spinor` | Velocity spinor of $i$ | From $v_i(t)$ |
| | `psi_v_j` | `Spinor` | Velocity spinor of $j$ | From $v_j(t)$ |
| | `psi_Delta_v_ij` | `Spinor` | Relative velocity $v_j - v_i$ | Computed |
| **Viscous Spinor** | `psi_viscous_ij` | `Spinor` | $\nu K_\rho(x_i,x_j)(v_j-v_i)$ | Eq. (2.3) |
| **Localization** | `K_rho_ij` | `float` | $K_\rho(x_i, x_j)$ | Def. 1.0.2 |
| | `w_ij` | `float` | $w_{ij}(\rho)$ normalized weight | Def. 1.0.3 |
| **Phase Potential** | `theta_ij` | `float` | $\theta_{ij} = -\frac{d_{\text{alg}}(i,j)^2}{2\varepsilon_c^2\hbar_{\text{eff}}}$ | Phase encoding |
| | `psi_ij` | `complex` | $\psi_{ij} = \sqrt{P_{\text{comp}}(i,j)} e^{i\theta_{ij}}$ | Complex amplitude |
| **Derived Scalars** | `distance` | `float` | $d_{ij} = \|x_i - x_j\|$ | Euclidean norm |
| **Antisymmetric** | `V_clone` | `float` | $V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i$ | Fitness difference |
| **Fitness** | `Phi_i` | `float` | $\Phi(x_i)$ from node | From node |
| | `Phi_j` | `float` | $\Phi(x_j)$ from node | From node |

---

## 6. Spinor Encoding and Covariance

### 6.1. Practical Spinor Representation

For computational implementation, spinors are stored as:

**3D case** (most common):
- Spinor $\psi \in \mathbb{C}^2$ (2 complex components = 4 real numbers)
- Related to vector $v \in \mathbb{R}^3$ via Pauli matrices:

  $$
  v \cdot \sigma = \begin{pmatrix} v_z & v_x - iv_y \\ v_x + iv_y & -v_z \end{pmatrix}

  $$

  Eigenspinors of $v \cdot \sigma$ encode $v$ covariantly.

**2D case**:
- Spinor $\psi \in \mathbb{C}$ (1 complex number = 2 real numbers)
- Phase $\arg(\psi)$ encodes direction

**General d dimensions**:
- Spinor $\psi \in \mathbb{C}^{2^{[d/2]}}$ (Clifford algebra representation)

**Storage format:**
```python
class Spinor:
    def __init__(self, components: np.ndarray):
        self.psi = components  # Complex array of shape (2^{d//2},)

    def to_vector(self) -> np.ndarray:
        """Convert spinor to vector via Clifford algebra."""
        return clifford_to_vector(self.psi)

    @staticmethod
    def from_vector(v: np.ndarray) -> 'Spinor':
        """Encode vector as spinor."""
        return Spinor(vector_to_clifford(v))
```

### 6.2. Covariance Verification

:::{prf:proposition} Frame Independence of the Fractal Set
:label: prop-frame-independence

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
:::

---

## 7. Quantum Embedding: Four-Walker Interference and Path Integral Structure

### 7.1. Overview: The Complete Four-Walker Quantum Structure

The cloning operator in the Adaptive Gas exhibits a **four-walker quantum interference pattern** arising from two independent random companion selections. Every cloning event involves:

1. **Walker i** (the cloning walker)
2. **Walker j** (the target that i may clone over)
3. **Walker k** (i's diversity companion, used to compute i's fitness)
4. **Walker m** (j's diversity companion, used to compute j's fitness)

The cloning decision compares **two fitness values computed with different diversity companions**:

$$
S(i,j,k,m) = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)

$$

This creates quantum interference over **(N-1)² paths** corresponding to all possible (k,m) combinations, where N is the number of alive walkers.

**Mathematical framework**: Following the quantum embedding discovered in [19_mean_field_gauge.md § 5](../19_mean_field_gauge.md#5-complex-structure-and-unitary-symmetries-the-quantum-embedding), we formalize the complete path integral structure of cloning amplitudes.

### 7.2. The Two Independent Random Selections

:::{prf:definition} Two Distinct Companion Selection Events
:label: def-two-companion-selections

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
:::

### 7.3. Phase Potentials and Complex Amplitudes

:::{prf:definition} Complete Set of Phase Potentials and Amplitudes
:label: def-complete-phase-amplitudes

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
:::

### 7.4. Interaction Hilbert Space and Four-Walker States

:::{prf:theorem} Tensor Product Structure for Cloning Interaction
:label: thm-interaction-hilbert-space

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
:::

:::{prf:proof}
The two diversity companion selections are **statistically independent**, so the joint probability factors:

$$
P(k, m | i, j) = P_{\text{comp}}^{(\text{div})}(k|i) \cdot P_{\text{comp}}^{(\text{div})}(m|j)

$$

Therefore the joint amplitude is the product:

$$
\psi_{ik}^{(\text{div})} \psi_{jm}^{(\text{div})} = \sqrt{P(k|i)P(m|j)} \cdot e^{i(\theta_{ik}^{(\text{div})} + \theta_{jm}^{(\text{div})})}

$$

This corresponds to a state in the tensor product Hilbert space $\mathbb{C}^{N-1} \otimes \mathbb{C}^{N-1}$. ∎
:::

### 7.5. Complete Path Integral Formula: U(1) Dressing and SU(2) Interaction

:::{prf:theorem} Path Integral Formulation of the Dressed SU(2) Interaction
:label: thm-path-integral-dressed-su2

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
:::

:::{prf:remark} Summary: Hierarchical Structure of Symmetries
:class: important

The cloning amplitude has a **two-level hierarchical structure** reflecting the separation of U(1) and SU(2) symmetries:

| **Component** | **Symmetry** | **Algorithmic Origin** | **Modulus** | **Phase** |
|---------------|--------------|------------------------|-------------|-----------|
| $A_{ij}^{\text{SU(2)}}$ | **SU(2) weak** | Cloning companion selection | $\sqrt{P_{\text{comp}}^{(\text{clone})}(j\|i)}$ | $\theta_{ij}^{(\text{SU(2)})} = -d_{\text{alg}}(i,j)^2/(2\epsilon_c^2\hbar_{\text{eff}})$ |
| $\psi_{ik}^{(\text{div})}$ | **U(1) fitness** | Diversity self-measurement (i) | $\sqrt{P_{\text{comp}}^{(\text{div})}(k\|i)}$ | $\theta_{ik}^{(\text{U(1)})} = -d_{\text{alg}}(i,k)^2/(2\epsilon_d^2\hbar_{\text{eff}})$ |
| $\psi_{jm}^{(\text{div})}$ | **U(1) fitness** | Diversity self-measurement (j) | $\sqrt{P_{\text{comp}}^{(\text{div})}(m\|j)}$ | $\theta_{jm}^{(\text{U(1)})} = -d_{\text{alg}}(j,m)^2/(2\epsilon_d^2\hbar_{\text{eff}})$ |
| $\psi_{\text{succ}}(S)$ | *(outcome)* | Success measurement | $\sqrt{P_{\text{succ}}(S)}$ | $S/\hbar_{\text{eff}}$ |

**Factorized structure:**

$$
\Psi(i \to j) = \underbrace{A_{ij}^{\text{SU(2)}}}_{\text{Interaction vertex}} \cdot \underbrace{K_{\text{eff}}(i, j)}_{\text{Dressed by U(1)}}

$$

**Physical narrative:**
1. **U(1) Dressing**: Walkers i and j independently probe their fitness via diversity companions (k and m), acquiring U(1) "charges"
2. **SU(2) Vertex**: The dressed walkers interact through cloning selection, forming a weak isospin doublet
3. **Path Integral**: The sum over (k, m) computes quantum interference of all possible U(1)-dressed configurations

**Note**: Fitness amplitudes $\psi_i^{(\text{fit})}(k)$ and $\psi_j^{(\text{fit})}(m)$ are **absorbed into** $\psi_{\text{succ}}(S)$ since $S = V_{\text{fit}}(i|k) - V_{\text{fit}}(j|m)$.
:::

:::{prf:remark} Feynman Diagram Interpretation
:class: important

The factorized structure $\Psi(i \to j) = A_{ij}^{\text{SU(2)}} \cdot K_{\text{eff}}(i, j)$ corresponds to a standard **QFT Feynman diagram** with self-energy corrections:

**Diagram Structure:**

```
    [i] ───○───╲              ╱───○─── [j]
           │    ╲            ╱    │
    U(1)   k     ╲__SU(2)__╱     m   U(1)
  dressing       ╱  vertex  ╲        dressing
```

**Components:**

1. **External lines**: Walkers i (incoming) and j (incoming/target)
2. **Self-energy loops** (U(1) dressing):
   - Walker i has a **virtual diversity loop** connecting to all possible companions k
   - Walker j has an independent **virtual diversity loop** connecting to all possible companions m
   - These loops "dress" the bare walker states with environmental fitness information
3. **Central vertex** (SU(2) interaction):
   - The two dressed walkers meet at a **2-to-2 scattering vertex**
   - Amplitude: $A_{ij}^{\text{SU(2)}}$
   - Cloning score $S(i,j,k,m)$ emerges from this vertex
4. **Outgoing state**: Either i or j survives (measurement collapse)

**Comparison to Standard QFT:**

| **Adaptive Gas** | **Standard QFT** |
|------------------|------------------|
| Diversity companion k, m | Virtual photons/gluons |
| U(1) dressing loops | Self-energy corrections |
| SU(2) vertex $A_{ij}$ | Weak interaction vertex |
| Path integral over (k,m) | Loop integration |
| $(N-1)^2$ paths | Continuum of virtual states |

**Physical Interpretation:**

This is not a primitive 4-point interaction. It is an **effective 2-to-2 interaction** whose strength is renormalized by environmental coupling (U(1) loops). The path integral $K_{\text{eff}}$ computes the quantum corrections to the bare SU(2) vertex from summing over all virtual diversity states.

The Adaptive Gas **naturally implements dressed perturbation theory** through its multi-stage stochastic sampling!
:::

### 7.6. Global U(1) Fitness Symmetry of Diversity Measurement

:::{prf:theorem} Global U(1)_fitness Symmetry
:label: thm-u1-fitness-global

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
:::

### 7.7. S_N Braid Holonomy: The True Gauge Structure

:::{prf:theorem} S_N Permutation Gauge Symmetry and Braid Holonomy
:label: thm-sn-braid-holonomy

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
:::

### 7.8. Storage in Fractal Set: IG Edge Structure with Gauge Symmetries

:::{prf:definition} IG Edge Attributes with Gauge Symmetry Assignment
:label: def-ig-edge-gauge-symmetries

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
:::

### 7.9. Lattice QFT Structure: S_N Discrete Gauge + SU(2) Continuum

:::{prf:theorem} Hybrid Discrete-Continuum Gauge Theory on the Fractal Set
:label: thm-sn-su2-lattice-qft

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
:::

### 7.10. SU(2) Weak Isospin Symmetry from Four-Walker Interaction

:::{prf:theorem} SU(2) Symmetry from Dressed Walker Interaction
:label: thm-su2-interaction-symmetry

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
:::

:::{prf:definition} Fitness Operator on Diversity Space
:label: def-fitness-operator

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
:::

:::{prf:proposition} SU(2) Invariance of Total Interaction Probability
:label: prop-su2-invariance

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
:::

:::{prf:proof}
We prove that $P_{\text{total}}(i,j) = P_{\text{clone}}(i \to j) + P_{\text{clone}}(j \to i)$ is invariant under SU(2) transformations.

**Step 1: Define Projection Operators**

Define projection operators onto the cloning subspaces:

$$
\hat{\Pi}(i \to j) := \Theta(\hat{S}_{ij}) \cdot \hat{P}_{\uparrow} \otimes I_{\text{div}}

$$

$$
\hat{\Pi}(j \to i) := \Theta(-\hat{S}_{ij}) \cdot \hat{P}_{\downarrow} \otimes I_{\text{div}}

$$

where:
- $\Theta$ is the Heaviside step function operator: $\Theta(\hat{A}) = \int_0^\infty d\lambda |\lambda\rangle\langle\lambda|$ projects onto positive eigenvalues of $\hat{A}$
- $\hat{S}_{ij} = (\hat{P}_{\uparrow} \otimes \hat{V}_{\text{fit},i}) - (\hat{P}_{\downarrow} \otimes \hat{V}_{\text{fit},j})$ is the cloning score operator

**Physical meaning**:
- $\hat{\Pi}(i \to j)$ projects onto states where walker i (in $|{\uparrow}\rangle$ role) has higher fitness than walker j, i.e., cloning i→j is favorable
- $\hat{\Pi}(j \to i)$ projects onto states where walker j (in $|{\downarrow}\rangle$ role) has higher fitness, i.e., cloning j→i is favorable

**Step 2: Express Probabilities as Expectation Values**

The cloning probabilities are:

$$
P_{\text{clone}}(i \to j) = \langle \Psi_{ij} | \hat{\Pi}(i \to j) | \Psi_{ij} \rangle

$$

$$
P_{\text{clone}}(j \to i) = \langle \Psi_{ij} | \hat{\Pi}(j \to i) | \Psi_{ij} \rangle

$$

Therefore:

$$
P_{\text{total}}(i,j) = \langle \Psi_{ij} | \left(\hat{\Pi}(i \to j) + \hat{\Pi}(j \to i)\right) | \Psi_{ij} \rangle

$$

**Step 3: Define Total Interaction Operator**

$$
\hat{\Pi}_{\text{interaction}} := \hat{\Pi}(i \to j) + \hat{\Pi}(j \to i)

$$

This operator measures the total propensity for the pair to interact via cloning (in either direction).

**Step 4: Mutual Pairing Constraint and Physical Observable**

**Critical Physical Constraint:**

The cloning companion selection algorithm (see {doc}`../03_cloning.md` §3) uses a **random matching algorithm** that enforces **mutual pairing**:

**Definition**: If walker i selects walker j as its cloning companion, then walker j selects walker i as its cloning companion. This creates symmetric pairs $(i,j)$ where both $i \in \mathcal{C}_j$ and $j \in \mathcal{C}_i$.

**Consequence for Fitness Scores:**

For mutually paired walkers $(i,j)$:

$$
s_i = \frac{V_{\text{fit}}(j) - V_{\text{fit}}(i)}{V_{\text{fit}}(i) + \epsilon}, \quad s_j = \frac{V_{\text{fit}}(i) - V_{\text{fit}}(j)}{V_{\text{fit}}(j) + \epsilon}
$$

**Key observation**: The numerators, $V_{\text{fit}}(j) - V_{\text{fit}}(i)$ and $V_{\text{fit}}(i) - V_{\text{fit}}(j)$, are **exact opposites**.

The denominators, $V_{\text{fit}}(i) + \epsilon$ and $V_{\text{fit}}(j) + \epsilon$, are **strictly positive**, as the fitness potential $V_{\text{fit}}$ is bounded below by $\eta^{(\alpha+\beta)} > 0$ (see {prf:ref}`lem-potential-bounds`).

Therefore, $s_i$ and $s_j$ must have **opposite signs** (assuming $V_{\text{fit}}(i) \neq V_{\text{fit}}(j)$).

**Physical Implication:**

Since $s_i$ and $s_j$ have opposite signs, exactly one fitness score is positive. Therefore:

$$
\Theta(s_i) + \Theta(s_j) = 1 \quad \text{(exactly one equals 1, the other equals 0)}
$$

This means:

$$
\hat{\Pi}_{\text{interaction}} = \hat{\Pi}(i \to j) + \hat{\Pi}(j \to i) = \begin{cases}
\hat{P}_{\uparrow} \otimes I & \text{if } s_i > 0 \text{ (lower fitness j clones to higher fitness i)} \\
\hat{P}_{\downarrow} \otimes I & \text{if } s_j > 0 \text{ (lower fitness i clones to higher fitness j)}
\end{cases}
$$

The cloning direction is **deterministically determined by fitness**: the walker with lower fitness always clones to the walker with higher fitness.

**Physical Interpretation of $P_{\text{total}}(i,j)$:**

Due to mutual pairing, $P_{\text{total}}(i,j)$ is **NOT** "sum of two independent probabilities." It is:

$$
P_{\text{total}}(i,j) = P_{\text{interaction}}(\{i,j\}) = \text{probability that the mutually paired walkers } \{i,j\} \text{ interact via cloning}
$$

This is a **single physical quantity** representing the interaction of the symmetric pair, independent of our choice of which basis state we call $|\uparrow\rangle$ (cloner) vs $|\downarrow\rangle$ (target).

**Step 5: SU(2) Gauge Invariance Proof**

Under local SU(2) gauge transformation at vertices i and j:
- State transforms: $|\Psi'_{ij}\rangle = (G_i \otimes G_j \otimes I_{\text{div}})|\Psi_{ij}\rangle$
- Operator transforms: $\hat{\Pi}'_{\text{interaction}} = (G_i \otimes G_j \otimes I) \hat{\Pi}_{\text{interaction}} (G_i^\dagger \otimes G_j^\dagger \otimes I)$

The transformed observable is:

$$
P'_{\text{total}}(i,j) = \langle \Psi'_{ij} | \hat{\Pi}'_{\text{interaction}} | \Psi'_{ij} \rangle
$$

Substituting the transformations:

$$
P'_{\text{total}}(i,j) = \langle \Psi_{ij} | (G_i^\dagger \otimes G_j^\dagger \otimes I)(G_i \otimes G_j \otimes I) \hat{\Pi}_{\text{interaction}} (G_i^\dagger \otimes G_j^\dagger \otimes I)(G_i \otimes G_j \otimes I) | \Psi_{ij} \rangle
$$

Using unitarity of $G_i$ and $G_j$: $(G_i^\dagger \otimes G_j^\dagger)(G_i \otimes G_j) = (G_i^\dagger G_i) \otimes (G_j^\dagger G_j) = I \otimes I = I$:

$$
P'_{\text{total}}(i,j) = \langle \Psi_{ij} | \hat{\Pi}_{\text{interaction}} | \Psi_{ij} \rangle = P_{\text{total}}(i,j)
$$

Therefore, $P_{\text{total}}(i,j)$ is **SU(2) gauge invariant**.

**Physical Interpretation:**

The gauge invariance of $P_{\text{total}}(i,j)$ reflects the physical content of the SU(2) symmetry:

1. **Mutual pairing defines a single physical observable**: The quantity $P_{\text{total}}(i,j)$ represents the interaction probability for the symmetric pair $\{i,j\}$, not a sum of two independent processes

2. **The projector encodes deterministic cloning**: Due to mutual pairing, $\hat{\Pi}_{\text{interaction}}$ projects onto the realized cloning direction (determined by fitness), which is a physical event

3. **SU(2) transformations represent gauge freedom**: The transformations relabel basis states ($|\uparrow\rangle \leftrightarrow |\downarrow\rangle$) without changing the underlying physics

4. **Invariance expresses physical consistency**: Relabeling the mathematical description (gauge transformation) cannot change the probability of a physical event (cloning interaction)

This structure characterizes a genuine gauge theory: unphysical gauge freedom combined with physically meaningful gauge-invariant observables. ∎
:::

:::{prf:remark} Connection to U(1)_fitness Symmetry
:class: important

The SU(2) interaction symmetry is **structurally linked** to the U(1)_fitness symmetry (§7.6).

**Key observation**: The SU(2) doublet $|\Psi_{ij}\rangle$ is constructed from dressed walker states $|\psi_i\rangle$ and $|\psi_j\rangle$, whose components $\psi_{ik}^{(\text{div})}$ are precisely the objects carrying U(1)_fitness charges:

$$
\psi_{ik}^{(\text{div})} \to e^{i\lambda_i} \psi_{ik}^{(\text{div})} \quad \text{(U(1)}_{\text{fitness}} \text{ transformation)}

$$

**Implication**: The SU(2) symmetry does not exist independently of the U(1)_fitness structure. It acts on objects that are **already U(1)-dressed** through diversity self-measurement. This provides a coherent framework where different gauge symmetries are intrinsically connected.

**Analogy to electroweak theory**: Similar to how SU(2)_L and U(1)_Y in the Standard Model act on fermion doublets (e.g., electron-neutrino), our SU(2)_weak acts on doublets whose components carry U(1)_fitness charges.

**Physical interpretation**:
- **U(1) dressing**: Each walker measures its fitness via diversity companions, acquiring U(1) charge
- **SU(2) interaction**: The **selection of walker j as cloning companion** initiates an SU(2) interaction between the two U(1)-dressed walkers
- **Dressed vertex**: The effective SU(2) interaction strength is modified by the U(1) dressing (path integral $K_{\text{eff}}$)

**Status**: This is a **structural link**, not a full "electroweak unification" in the sense of a unified gauge group with mixed generators. A complete unification would require demonstrating a unified covariant derivative $D_\mu = \partial_\mu - igA_\mu^a T^a - ig'B_\mu Y$ and deriving the gauge field dynamics from a single action.
:::

:::{prf:remark} Measurement and Wave Function Collapse
:class: important

The connection to quantum measurement is operationally concrete:

**1. Prepared State (Superposition)**:

Before measurement, the system exists in the interaction doublet $|\Psi_{ij}\rangle$, a superposition of potential outcomes.

**2. Measurement Apparatus**:

The algorithmic implementation:
- Random threshold: $T_i \sim U(0, p_{\text{max}})$
- Comparator: $S_i(c_i) > T_i$ (where $S_i$ is the cloning score)

**3. Wave Function Collapse**:

The moment the comparison is executed, the state collapses:
- If $S_i > T_i$: State collapses to $|↑\rangle$ (Clone occurs)
- If $S_i \leq T_i$: State collapses to $|↓\rangle$ (Persist)

This maps directly to Born rule: $P(\text{Clone}) = |\langle ↑ | \Psi_{ij} \rangle|^2$

The random sampling implements quantum measurement, transforming potential (amplitude) into actuality (definite outcome).
:::

:::{prf:remark} Gauge Covariant Path Integral and SU(2) Gauge Field
:class: dropdown

The SU(2) symmetry acting on pairs of walkers $(i, j)$ represents a **local gauge symmetry**. To fully integrate with the path integral formalism (§7.5), which sums over individual diversity companion paths $(k, m)$, the theory must be elevated to a true lattice gauge theory.

This requires two further theoretical steps:

**1. Derivation of the SU(2) Noether Current:**

The SU(2) symmetry, via Noether's theorem, implies the existence of a **conserved current** $J_\mu^{(a)}(i, j)$ for $a = 1, 2, 3$ (three generators of SU(2)).

**Form of the current** (to be derived):

$$
J_\mu^{(a)}(i,j) = \bar{\Psi}_{ij} \gamma_\mu T^a \Psi_{ij}
$$

where $T^a = \sigma^a/2$ are the SU(2) generators (Pauli matrices).

**Conservation**: $\partial^\mu J_\mu^{(a)} = 0$ under the discrete equations of motion.

**2. Introduction of SU(2) Gauge Field:**

A complete gauge-covariant description requires a new dynamical **SU(2) gauge field** (connection) $A_\mu^{(a)}$ living on the edges of the Fractal Set.

**Gauge-covariant derivative**:

$$
D_\mu \Psi_{ij} = \partial_\mu \Psi_{ij} - ig \sum_{a=1}^3 A_\mu^{(a)} (T^a \otimes I_{\text{div}}) \Psi_{ij}
$$

**Coupling term in action**:

$$
S_{\text{coupling}} = \int d\tau \sum_{a=1}^3 J_\mu^{(a)} A^{(a,\mu)}
$$

**Yang-Mills field strength**:

$$
F_{\mu\nu}^{(a)} = \partial_\mu A_\nu^{(a)} - \partial_\nu A_\mu^{(a)} + g \epsilon^{abc} A_\mu^{(b)} A_\nu^{(c)}
$$

**Gauge field action**:

$$
S_{\text{YM}} = -\frac{1}{4} \int d^4x \sum_{a=1}^3 F_{\mu\nu}^{(a)} F^{(a,\mu\nu)}
$$

**Total path integral**:

$$
Z = \int \mathcal{D}[\Psi] \mathcal{D}[A_\mu] \exp\left(i(S_{\text{matter}} + S_{\text{coupling}} + S_{\text{YM}})\right)
$$

where the integral is over:
- **Matter fields**: All diversity companion configurations $(k, m)$
- **Gauge fields**: All SU(2) connection configurations $A_\mu^{(a)}$

**Status**: This is a **necessary extension** for complete gauge-covariant formulation. Requires:
- Explicit derivation of conserved currents from discrete dynamics
- Specification of gauge field dynamics on the Fractal Set lattice
- Proof that gauge invariance is preserved under discretization

**Recommendation**: Develop as **priority research program** following validation of current interaction space formulation.
:::

### 7.11. Scalar Higgs-Like Reward Field

:::{prf:definition} The Reward Scalar Field
:label: def-reward-scalar-field

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
:::

:::{prf:remark} Higgs Vacuum Expectation Value (VEV)
:class: important

The **vacuum expectation value** of the reward field is the global average:

$$
\langle r \rangle = \frac{1}{N} \sum_{i \in A_t} r(x_i)

$$

**Symmetry breaking condition**: $\langle r \rangle \neq 0$ breaks the U(1)_fitness × SU(2)_weak symmetry, giving "mass" (stability) to walkers concentrated near high-reward regions.

**Mexican hat potential**: The reward landscape $U(x) - \beta r(x)$ exhibits a characteristic double-well structure during phase transition, analogous to the Higgs potential:

$$
V_{\text{Higgs}}(r) = -\mu^2 |r|^2 + \lambda |r|^4

$$

where $\mu^2 < 0$ triggers spontaneous symmetry breaking.
:::

### 7.12. Formal Gauge Invariance: Proofs

This section proves that the three symmetries (S_N, Global U(1), Local SU(2)) give gauge-invariant observables.

:::{prf:theorem} Global U(1)_fitness Invariance
:label: thm-u1-global-invariance

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
:::

:::{prf:remark} Why Not Local U(1)?
:class: important

**Critical insight from rigorous analysis:**

We cannot have **local** U(1) gauge invariance (different α(i) at each walker) because the U(1) phases are **not dynamical gauge fields** - they are **fixed by algorithmic distances**.

**Step-by-step demonstration:**

Recall from §7.5 that the companion link amplitudes are:

$$
\psi_{ik} = \sqrt{P_{\text{comp}}(k|i)} \cdot e^{i\theta_{ik}^{(\text{U}(1))}}
$$

where the phase is determined by algorithmic distance:

$$
\theta_{ik}^{(\text{U}(1))} = -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2 \hbar_{\text{eff}}}
$$

**Attempt a local U(1) transformation:**

Consider transforming each companion link amplitude:

$$
\psi_{ik} \to \psi_{ik}' = e^{i\alpha(i)} \psi_{ik} = \sqrt{P_{\text{comp}}(k|i)} \cdot e^{i(\theta_{ik}^{(\text{U}(1))} + \alpha(i))}
$$

This would correspond to shifting the phase:

$$
\theta_{ik}^{(\text{U}(1))} \to \theta_{ik}^{(\text{U}(1))} + \alpha(i)
$$

**Why this fails:**

For this to be a **gauge transformation**, we would need a **gauge connection** $A_i$ such that:

$$
\theta_{ik}^{(\text{U}(1))} \to \theta_{ik}^{(\text{U}(1))} + \alpha(i) - \alpha(k)
$$

(analogous to the covariant derivative $\partial_\mu + iA_\mu$ in standard gauge theory).

However, the phases θ_ik^(U(1)) are **NOT free gauge degrees of freedom** - they are **fixed by the algorithmic distance metric**:

$$
\theta_{ik}^{(\text{U}(1))} = f(d_{\text{alg}}(i,k))
$$

where $f(d) = -d^2/(2\epsilon_d^2 \hbar_{\text{eff}})$ is a **deterministic function** of the algorithmic distance.

**The contradiction:**

If we perform a local transformation $\theta_{ik} \to \theta_{ik} + \alpha(i)$, the transformed phase is **no longer consistent** with the algorithmic distance structure:

$$
\theta_{ik}' = -\frac{d_{\text{alg}}(i,k)^2}{2\epsilon_d^2 \hbar_{\text{eff}}} + \alpha(i) \neq f(d_{\text{alg}}(i,k))
$$

unless $\alpha(i) = 0$ for all $i$ (the trivial transformation).

**Physical interpretation:**

1. **Standard gauge theory**: Gauge fields (like the photon field $A_\mu$ in QED) are **dynamical** - they can be freely transformed by gauge transformations, and the physics (observables) remains unchanged.

2. **Our U(1) phases**: The phases θ_ik^(U(1)) are **kinematic** - they are completely determined by the swarm's algorithmic geometry. They cannot be independently transformed without breaking the consistency with the distance metric.

**Why global U(1) works:**

A **global** transformation (same α for all walkers):

$$
\psi_{ik} \to e^{i\alpha} \psi_{ik} \quad \text{for all } i,k
$$

preserves all observables because:

$$
K_{\text{eff}}'(i,j) = \sum_{k,m} (e^{i\alpha} \psi_{ik})(e^{i\alpha} \psi_{jm})^* \cdot f(d_{\text{alg}}(k,m)) = |e^{i\alpha}|^2 K_{\text{eff}}(i,j) = K_{\text{eff}}(i,j)
$$

The global phase cancels in all probability amplitudes: $|K_{\text{eff}}'|^2 = |K_{\text{eff}}|^2$.

**The resolution:**

U(1)_fitness is a **global symmetry** (like baryon number conservation in the Standard Model), **not a local gauge symmetry** (like electromagnetism). The phases are **background geometric data**, not dynamical gauge fields.

The **true local gauge symmetry** is the discrete S_N permutation group, which acts by relabeling walker indices - a transformation that is consistent with the permutation invariance of the algorithmic distance metric (see §7.7).

**Analogy:**

This situation is analogous to:
- **Flat spacetime special relativity**: Global Poincaré symmetry (translations/rotations apply uniformly)
- **General relativity**: Local diffeomorphism symmetry (can transform differently at each point)

Our U(1)_fitness is like the global Poincaré group - it's a **rigid symmetry**, not a **local gauge symmetry**.
:::

:::{prf:theorem} Local SU(2)_weak Gauge Invariance
:label: thm-su2-local-gauge-invariance

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
:::

:::{prf:remark} Non-Abelian Nature of SU(2)
:class: important

**Key differences from U(1):**

1. **Matrix-valued**: SU(2) link variables are 2×2 matrices, not just phases
2. **Non-commutative**: $U_{ij} U_{jk} \neq U_{jk} U_{ij}$ - **path ordering matters**
3. **Self-interaction**: The field strength has cubic terms: $F_{\mu\nu} \sim \partial A + g[A, A]$

**Physical consequence:**

Non-Abelian gauge bosons (W/Z bosons) **interact with themselves**, unlike photons. This leads to:
- Asymptotic freedom (strong force weakens at short distances)
- Confinement (quarks cannot be isolated)
- Mass generation via Higgs mechanism

Our SU(2)_weak symmetry exhibits all these features through the cloning interaction structure.
:::

:::{prf:theorem} S_N Permutation Gauge Invariance
:label: thm-sn-gauge-invariance

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
:::

:::{prf:remark} Three-Tier Gauge Structure
:class: important

**Summary of the complete gauge structure:**

| Symmetry | Type | Origin | Observables | Wilson Loops |
|----------|------|--------|-------------|--------------|
| **S_N** | Local, discrete | Walker labels arbitrary | S_N-invariant functions | Braid holonomy $\text{Hol}(\gamma)$ |
| **SU(2)_weak** | Local, continuous | Cloning isospin doublet | Trace over weak isospin | SU(2) path-ordered exponential |
| **U(1)_fitness** | Global, continuous | Absolute fitness scale | None (global phase) | None (global symmetry) |

**Hierarchy:**

1. **S_N is fundamental**: Discrete gauge group from first principles
2. **SU(2) is emergent local**: Continuum effective theory for cloning
3. **U(1) is emergent global**: Conserved fitness charge (Noether)

This structure **generalizes the Standard Model** by incorporating discrete gauge topology alongside continuum gauge fields.
:::
### 7.13. SU(3) Strong Sector from Viscous Force

:::{prf:theorem} SU(3) Color Symmetry from Viscous Force Vector
:label: thm-su3-strong-sector

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
:::

:::{prf:definition} SU(3) Gluon Field Components from Manifold Geometry
:label: def-su3-gluon-field

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
:::

### 7.13. Fermionic Behavior and Z₂ Pauli Exclusion

:::{prf:theorem} Fermionic Statistics from Antisymmetric Cloning
:label: thm-fermionic-z2-symmetry

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
:::

:::{prf:proof}
**Antisymmetry proof**:

Under walker exchange $i \leftrightarrow j$:

$$
\Psi(i \to j) = \psi_{ij}^{(\text{clone})} \cdot \sum_{k,m} [\psi_{ik}^{(\text{div})} \cdot \psi_{jm}^{(\text{div})} \cdot \psi_{\text{succ}}(S(i,j,k,m))]

$$

$$
\Psi(j \to i) = \psi_{ji}^{(\text{clone})} \cdot \sum_{k,m} [\psi_{jk}^{(\text{div})} \cdot \psi_{im}^{(\text{div})} \cdot \psi_{\text{succ}}(S(j,i,k,m))]

$$

Since $S(j,i,k,m) = V_{\text{fit}}(j|k) - V_{\text{fit}}(i|m) = -S(i,j,m,k)$:

$$
\psi_{\text{succ}}(S(j,i,k,m)) = \sqrt{P_{\text{succ}}(-S)} \cdot e^{-iS/\hbar_{\text{eff}}}

$$

For a symmetric sigmoid $P_{\text{succ}}(-S) = 1 - P_{\text{succ}}(S)$, the total amplitude transforms as:

$$
\Psi(j \to i) \approx -\Psi(i \to j) \quad \text{(antisymmetric)}

$$

proving fermionic exchange statistics. ∎
:::

### 7.14. Emergent General Relativity and Curved Manifold

:::{prf:theorem} Emergent Spacetime Curvature from Fitness Hessian
:label: thm-emergent-general-relativity

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
:::

:::{prf:definition} Storage of Curvature Tensor in Fractal Set
:label: def-curvature-storage

The Riemann curvature tensor $R^\rho_{\sigma\mu\nu}(x_i)$ at each walker position is stored in the Fractal Set:

**Node attributes** (scalars):
- Ricci scalar: $R(n_{i,t}) = R(x_i(t))$
- Metric determinant: $\det(g)(n_{i,t}) = \det(g_{\mu\nu}(x_i))$

**CST edge attributes** (spinors):
- Christoffel symbols: $\Gamma^\lambda_{\mu\nu}(n_{i,t})$ (rank-3 tensor, stored as spinor components)
- Ricci tensor: $R_{\mu\nu}(n_{i,t})$ (symmetric 2-tensor, stored as spinor)

**Full Riemann tensor**: Stored as auxiliary data structure (rank-4 tensor, $d^4$ components for d-dimensional space).

**Dimension**: For $d=3$ spatial dimensions, $R^\rho_{\sigma\mu\nu}$ has $3^4 = 81$ components, but symmetries reduce to **20 independent components**.
:::

### 7.15. SO(10) Grand Unification

:::{prf:theorem} SO(10) Grand Unified Theory from Complete Symmetry Structure
:label: thm-so10-grand-unification

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
:::

:::{prf:remark} Complete Fractal Set as Grand Unified Field Theory
:class: important

The **Fractal Set** encodes a complete **SU(3) × SU(2) × U(1) grand unified field theory** with the following identifications:

| **Physics Concept** | **Fractal Set Realization** | **Algorithmic Origin** | **Section** |
|---------------------|------------------------------|------------------------|-------------|
| **U(1)_fitness** | Diversity self-measurement phase | Diversity companion selection | §7.6 |
| **SU(2)_weak** | Weak isospin doublet interaction | Cloning companion selection | §7.10 |
| **Higgs field** | Reward scalar $r(x)$ | Position-dependent fitness potential | §7.11 |
| **SU(3)_color** | Viscous force three-vector | Momentum-phase encoding | §7.12 |
| **Fermions** | Walkers (antisymmetric cloning) | Pauli exclusion from antisymmetry | §7.13 |
| **General Relativity** | Emergent curved manifold $g_{\mu\nu}$ | Fitness Hessian curvature | §7.14 |
| **SO(10) GUT** | Unified 16-component spinor | Complete gauge unification | §7.15 |

**Gauge Group Structure**:
$$
G = \text{SU}(3)_{\text{color}} \times \text{SU}(2)_{\text{weak}} \times \text{U}(1)_{\text{fitness}}
$$

This directly parallels the Standard Model gauge group (with U(1)_fitness replacing U(1)_Y).

**Status**: This is a **complete particle physics analogue** embedded in the algorithmic structure of the Adaptive Gas.

**Testable predictions**:
1. Wilson loops measure confinement/deconfinement phase transition
2. Higgs VEV $\langle r \rangle$ correlates with algorithm convergence
3. Geodesic deviation measures effective "gravitational" coupling between walkers
4. SO(10) unification predicts specific relationships between coupling constants $\epsilon_d, \epsilon_c, \nu, \gamma$
:::

---

## 8. Conclusion

The Fractal Set defined here provides a **discrete, fine-grained, complete, frame-independent, quantum-embedded representation** of the Adaptive Viscous Fluid Model from [../07_adaptative_gas.md](../07_adaptative_gas.md):

**Key properties:**
1. **Maximum granularity**: One node per walker per timestep (no temporal aggregation)
2. **Complete SDE representation**: All data from the Adaptive Gas SDE stored (positions, velocities, forces, diffusion, localized statistics)
3. **Dual directed graph structure**: CST (timelike temporal evolution) + IG (spacelike viscous coupling, **directed**)
4. **Frame independence**: Scalars in nodes, spinors in edges
5. **Covariant dynamics**: Force decomposition $\mathbf{F}_{\text{stable}}, \mathbf{F}_{\text{adapt}}, \mathbf{F}_{\text{viscous}}$ stored as spinors
6. **Antisymmetric fitness coupling**: IG edges carry antisymmetric potential $V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)$
7. **Fermionic precursor**: Antisymmetry enables fermionic propagators in continuum limit
8. **Four-walker quantum embedding**: Dressed SU(2) interaction with U(1) self-energy loops → effective interaction kernel from $(N-1)^2$ path interference (Section 7)
9. **Path integral structure**: Cloning amplitude $\Psi(i \to j) = A_{ij}^{\text{SU}(2)} \cdot K_{\text{eff}}(i,j)$ factorizes into SU(2) vertex and U(1) dressing (Theorem {prf:ref}`thm-path-integral-dressed-su2`)
10. **Gauge theory**: Standard Model-like gauge group $\text{SU}(3)_{\text{color}} \times \text{SU}(2)_{\text{weak}} \times \text{U}(1)_{\text{fitness}}$ with clear algorithmic origins (Theorem {prf:ref}`thm-u1-fitness-gauge`, {prf:ref}`thm-su2-interaction-symmetry`)
11. **Lattice QFT**: Wilson action for U(1) × SU(2) with Feynman diagram structure showing self-energy loops + central vertex (Theorem {prf:ref}`thm-u1-su2-lattice-qft`)
12. **Perfect fossil record**: Algorithm can be fully reconstructed from $\mathcal{F}$ (Theorem {prf:ref}`thm-fractal-set-reconstruction`)

**Data from Adaptive Gas SDE ({prf:ref}`def-hybrid-sde`):**

**Nodes (scalars):**
- Phase space: $E_{\text{kin}} = \frac{1}{2}\|v\|^2$, $U(x)$
- Fitness: $\Phi(x)$, $V_{\text{fit}}[f_k,\rho](x)$
- Localized statistics: $\mu_\rho, \sigma_\rho, \sigma'_\rho, Z_\rho$ from {prf:ref}`def-localized-mean-field-moments` and {prf:ref}`def-unified-z-score`
- Status: $s \in \{0,1\}$ (alive in $A_k$)
- Parameters: $\epsilon_F, \nu, \gamma, \rho, \epsilon_\Sigma$

**CST edges (spinors):**
- Velocity evolution: $\psi_{v,t}, \psi_{v,t+1}, \psi_{\Delta v}$
- Force decomposition: $\psi_{\mathbf{F}_{\text{stable}}}, \psi_{\mathbf{F}_{\text{adapt}}}, \psi_{\mathbf{F}_{\text{viscous}}}, \psi_{\mathbf{F}_{\text{friction}}}$
- Diffusion: $\psi_{\Sigma_{\text{reg}}}, \psi_{\text{noise}}$
- Gradients: $\psi_{\nabla U}, \psi_{\nabla \Phi}, \psi_{\nabla V_{\text{fit}}}$

**IG edges (spinors, directed, two-channel):**
- Pairwise coupling: $\psi_{\text{viscous},ij} = \text{spinor}[\nu K_\rho(x_i,x_j)(v_j-v_i)]$
- Localization: $K_\rho(x_i,x_j), w_{ij}(\rho)$ from {prf:ref}`def-localization-kernel`
- Antisymmetric fitness: $V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i$
- **Two-channel phases** (Section 7): $\theta_{ik}^{(\text{div})}, \theta_{ij}^{(\text{clone})}$
- **Complex amplitudes**: $\psi_{ik}^{(\text{div})}, \psi_{ij}^{(\text{clone})}, \psi_{\text{succ}}(S(i,j,k,m))$
- **Four-walker amplitude**: $\Psi(i \to j) = \psi_{ij}^{(\text{clone})} \cdot \sum_{k,m} [\psi_{ik}^{(\text{div})} \cdot \psi_{jm}^{(\text{div})} \cdot \psi_{\text{succ}}(S_{km})]$

**Reconstruction:** Theorem {prf:ref}`thm-fractal-set-reconstruction` proves the Fractal Set enables complete reconstruction of:
- Full trajectories $(x_i(t), v_i(t))$
- All force fields $\mathbf{F}_{\text{stable}}, \mathbf{F}_{\text{adapt}}, \mathbf{F}_{\text{viscous}}$
- Diffusion tensor $\Sigma_{\text{reg}}(x,S,t)$
- Empirical measure $f_k(t)$ and localized statistics
- SDE verification up to noise realization $dW_i$

The Fractal Set is the **computational substrate** for analyzing Adaptive Gas dynamics in a coordinate-free, frame-independent manner.