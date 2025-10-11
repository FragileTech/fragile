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

## 7. Conclusion

The Fractal Set defined here provides a **discrete, fine-grained, complete, frame-independent representation** of the Adaptive Viscous Fluid Model from [../07_adaptative_gas.md](../07_adaptative_gas.md):

**Key properties:**
1. **Maximum granularity**: One node per walker per timestep (no temporal aggregation)
2. **Complete SDE representation**: All data from the Adaptive Gas SDE stored (positions, velocities, forces, diffusion, localized statistics)
3. **Dual directed graph structure**: CST (timelike temporal evolution) + IG (spacelike viscous coupling, **directed**)
4. **Frame independence**: Scalars in nodes, spinors in edges
5. **Covariant dynamics**: Force decomposition $\mathbf{F}_{\text{stable}}, \mathbf{F}_{\text{adapt}}, \mathbf{F}_{\text{viscous}}$ stored as spinors
6. **Antisymmetric fitness coupling**: IG edges carry antisymmetric potential $V_{\text{clone}}(j \to i) = -V_{\text{clone}}(i \to j)$
7. **Fermionic precursor**: Antisymmetry enables fermionic propagators in continuum limit
8. **Perfect fossil record**: Algorithm can be fully reconstructed from $\mathcal{F}$ (Theorem {prf:ref}`thm-fractal-set-reconstruction`)

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

**IG edges (spinors, directed):**
- Pairwise coupling: $\psi_{\text{viscous},ij} = \text{spinor}[\nu K_\rho(x_i,x_j)(v_j-v_i)]$
- Localization: $K_\rho(x_i,x_j), w_{ij}(\rho)$ from {prf:ref}`def-localization-kernel`
- Antisymmetric fitness: $V_{\text{clone}}(i \to j) = \Phi_j - \Phi_i$

**Reconstruction:** Theorem {prf:ref}`thm-fractal-set-reconstruction` proves the Fractal Set enables complete reconstruction of:
- Full trajectories $(x_i(t), v_i(t))$
- All force fields $\mathbf{F}_{\text{stable}}, \mathbf{F}_{\text{adapt}}, \mathbf{F}_{\text{viscous}}$
- Diffusion tensor $\Sigma_{\text{reg}}(x,S,t)$
- Empirical measure $f_k(t)$ and localized statistics
- SDE verification up to noise realization $dW_i$

The Fractal Set is the **computational substrate** for analyzing Adaptive Gas dynamics in a coordinate-free, frame-independent manner.
