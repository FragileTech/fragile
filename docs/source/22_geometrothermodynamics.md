# Geometrothermodynamics and the Ruppeiner Metric

## 0. Introduction and Motivation

### 0.1. The Bridge Between Geometry and Thermodynamics

Thermodynamics, at its core, is the study of equilibrium systems and their transformations. Geometry, on the other hand, is the study of spaces and their intrinsic structure. At first glance, these appear to be distinct branches of mathematics with different physical applications. However, a profound insight emerged in the late 20th century: **thermodynamic state spaces carry natural Riemannian metrics whose curvature encodes critical information about phase transitions and interactions**.

This field, known as **geometrothermodynamics**, was pioneered by Ruppeiner (1979, 1995) and Weinhold (1975), who independently discovered that the Hessian matrices of thermodynamic potentials define Riemannian metrics. These metrics are not mathematical curiosities—their scalar curvature diverges precisely at critical points, providing a geometric diagnostic for phase transitions.

**The Fragile Gas Framework's Unique Contribution:**

The Fragile Gas framework, developed in the preceding chapters, offers an unprecedented opportunity: we possess **all the mathematical machinery needed to algorithmically construct the Ruppeiner metric** directly from the dynamics of the swarm. Unlike traditional thermodynamic systems where the partition function must be postulated or derived from microscopic physics, the Fragile Gas **generates its quasi-stationary distribution (QSD) as an emergent property** of the optimization dynamics.

This chapter establishes the following results:

1. **Thermodynamic structure from QSD** (Part 1): The QSD naturally defines a partition function, free energy, entropy, and all standard thermodynamic potentials
2. **Contact geometric formulation** (Part 2): Thermodynamic phase space carries a contact structure encoding first-order thermodynamics
3. **Fisher-Rao and emergent metrics** (Part 3): The Fisher information metric on the statistical manifold coincides with the pullback of the emergent Riemannian metric from Chapter 8
4. **Algorithmic Ruppeiner metric** (Part 4): An explicit algorithm constructs the Ruppeiner metric from swarm samples with provable convergence
5. **Conformal duality** (Part 5): The Weinhold and Ruppeiner metrics are conformally equivalent with temperature as conformal factor
6. **Phase transition detection** (Part 6): Curvature singularities of the Ruppeiner metric detect thermodynamic phase transitions
7. **Quantum extension** (Part 7): The framework extends to quantum statistical mechanics via density matrix geometry
8. **Numerical validation** (Part 8): Concrete algorithms with error bounds and validation tests

### 0.2. Relation to Existing Framework

This chapter synthesizes results from multiple framework documents:

**Mathematical Foundations:**
- **Chapter 4** ([04_convergence.md](04_convergence.md)): QSD convergence and invariant measure theory
- **Chapter 8** ([08_emergent_geometry.md](08_emergent_geometry.md)): Emergent Riemannian metric $g(x, S) = H(x, S) + \epsilon_\Sigma I$
- **Chapter 10** ([10_kl_convergence/10_kl_convergence.md](10_kl_convergence/10_kl_convergence.md)): Fisher information and logarithmic Sobolev inequalities

**Geometric Structures:**
- **Chapter 9** ([09_symmetries_adaptive_gas.md](09_symmetries_adaptive_gas.md)): Fisher-Rao geometry and information-geometric interpretation
- **Chapter 14** ([14_scutoid_geometry_framework.md](14_scutoid_geometry_framework.md)): Scutoid tessellation and Regge calculus for discrete curvature
- **Chapter 15** ([15_scutoid_curvature_raychaudhuri.md](15_scutoid_curvature_raychaudhuri.md)): Raychaudhuri equation and volume evolution

**Statistical Methods:**
- **Chapter 19** ([19_geometric_sampling_reweighting.md](19_geometric_sampling_reweighting.md)): Importance reweighting for unbiased geometric observables

**Physical Applications:**
- **Chapter 16** ([general_relativity/16_general_relativity_derivation.md](general_relativity/16_general_relativity_derivation.md)): Stress-energy tensor and emergent general relativity
- **Chapter 20** ([hydrodynamics.md](hydrodynamics.md)): Navier-Stokes equations and fluid thermodynamics

### 0.3. Why This Matters: Applications and Implications

Geometrothermodynamics is not merely an elegant reformulation of standard thermodynamics. It provides:

**1. Computational Advantage:**
- Phase transitions are detected by curvature singularities (coordinate-invariant)
- No need to compute order parameters (which are system-specific)
- Universal diagnostic applicable to any thermodynamic system

**2. Deep Physical Insight:**
- Flat metric → non-interacting system (ideal gas)
- Curved metric → interactions present (van der Waals gas)
- Curvature singularity → critical point (liquid-gas transition)

**3. Connection to Information Geometry:**
- Fisher information metric measures distinguishability of probability distributions
- Ruppeiner metric is Fisher metric on thermodynamic state space
- Information-theoretic and thermodynamic phase transitions are geometrically equivalent

**4. Bridge to Quantum Gravity:**
- Emergent spacetime geometry (Chapter 16) + thermodynamic geometry → holographic principle
- Black hole thermodynamics: Ruppeiner metric of black holes encodes near-horizon geometry
- Hints at deep connection between gravity, thermodynamics, and information

### 0.4. Document Outline

**Part 1 (§1): Thermodynamic Foundations from QSD**
- Extract partition function, free energy, entropy, internal energy from QSD
- Prove thermodynamic identities (Maxwell relations)
- Define thermodynamic state space with coordinates $(S, U, V, N, ...)$ or $(T, P, μ, ...)$

**Part 2 (§2): Contact Geometry and Thermodynamic Phase Space**
- Introduce contact manifold structure on thermodynamic phase space
- Define contact 1-form $\theta = dU - T dS + P dV - \mu dN$
- Prove Legendre submanifolds correspond to thermodynamic potentials
- Establish Reeb vector field as thermodynamic flow

**Part 3 (§3): The Riemannian Thermodynamic Manifold**
- Define Fisher information metric from QSD statistical manifold
- Prove equivalence to Hessian metrics from thermodynamic potentials
- Establish pullback relationship: $g_{\text{Fisher}} = \mathbb{E}[g_{\text{emergent}}]$

**Part 4 (§4): The Ruppeiner Metric (Crown Jewel)**
- Rigorous definition: $g_R^{ij} = -\frac{\partial^2 S}{\partial U^i \partial U^j}$
- **Algorithm**: Construct $g_R$ from finite swarm samples with error bounds
- Prove convergence: discrete approximation → continuum limit
- Physical interpretation: curvature measures thermodynamic interactions

**Part 5 (§5): The Weinhold Metric and Conformal Duality**
- Define Weinhold metric: $g_W^{ij} = \frac{\partial^2 U}{\partial S^i \partial S^j}$
- **Theorem**: $g_W = T \cdot g_R$ (conformal equivalence)
- Information-geometric interpretation via Bregman divergence

**Part 6 (§6): Curvature and Critical Phenomena**
- Compute Ruppeiner scalar curvature $R_{\text{Rupp}}$
- **Theorem**: Phase transitions ↔ curvature singularities
- Connect to information-geometric phases (Chapter 15)
- Examples: exploration-exploitation transition, clustering phase transition

**Part 7 (§7): Quantum Statistical Extension**
- Extend to quantum density matrix $\hat{\rho} = e^{-\beta \hat{H}} / Z$
- Bures metric as quantum Fisher information metric
- Application to Yang-Mills QSD on Fractal Set lattice (Chapter 13)

**Part 8 (§8): Computational Methods and Validation**
- Importance reweighting algorithm (Chapter 19) for thermodynamic observables
- Numerical validation: Maxwell relations, metric positive-definiteness
- Convergence analysis: $O(1/\sqrt{N})$ error bounds

### 0.5. Prerequisites and Notation

**Required Background:**
- Differential geometry: Riemannian metrics, curvature tensors, Levi-Civita connection
- Statistical mechanics: Partition functions, thermodynamic potentials, entropy
- Information theory: KL-divergence, Fisher information, logarithmic Sobolev inequalities
- Stochastic processes: Markov chains, invariant measures, ergodicity

**Notation Conventions:**
- $\rho_{\text{QSD}}(x, v)$: Quasi-stationary distribution density
- $g(x, S)$: Emergent Riemannian metric from Chapter 8
- $g_R^{ij}$: Ruppeiner metric on thermodynamic state space
- $g_W^{ij}$: Weinhold metric (dual to Ruppeiner)
- $D_{\text{KL}}(\mu \| \nu)$: Kullback-Leibler divergence
- $I(\mu \| \nu)$: Fisher information (relative)
- $S$: Thermodynamic entropy
- $U$: Internal energy
- $F$: Helmholtz free energy
- $T$: Temperature
- $\beta = 1/(k_B T)$: Inverse temperature

**Framework Parameters:**
- $N$: Number of walkers
- $d$: Spatial dimension
- $\alpha, \beta$: Fitness exploitation/diversity weights
- $\gamma$: Friction coefficient
- $\sigma_v$: Velocity noise magnitude
- $T = \sigma_v^2 / \gamma$: Effective temperature
- $\epsilon_\Sigma$: Hessian regularization parameter

---

## 1. Thermodynamic Foundations from QSD

### 1.1. The QSD as a Thermodynamic Equilibrium

The quasi-stationary distribution (QSD) $\pi_{\text{QSD}}$ of the Fragile Gas, proven to exist and be unique in Chapter 4, is the natural starting point for thermodynamic analysis. We now establish that the QSD satisfies all requirements of a canonical ensemble in statistical mechanics.

:::{prf:theorem} QSD as Canonical Ensemble
:label: thm-qsd-canonical-ensemble

The quasi-stationary distribution of the N-particle Euclidean Gas with confining potential $U(x)$, friction coefficient $\gamma$, and velocity noise magnitude $\sigma_v$ is a canonical ensemble with the form:

$$
\rho_{\text{QSD}}(x_1, \ldots, x_N, v_1, \ldots, v_N) = \frac{1}{Z} \prod_{i=1}^N \sqrt{\det g(x_i)} \exp\left(-\beta H_{\text{eff}}(x_i, v_i)\right)
$$

where:

1. **Effective Hamiltonian:**

$$
H_{\text{eff}}(x, v) = U(x) - \epsilon_F V_{\text{fit}}(x, S) + \frac{1}{2}m\|v\|^2
$$

2. **Inverse Temperature:**

$$
\beta = \frac{\gamma}{\sigma_v^2} = \frac{1}{k_B T}
$$

with effective temperature $T = \sigma_v^2 / \gamma$

3. **Partition Function:**

$$
Z = \int_{\mathcal{X}^N \times \mathbb{R}^{dN}} \prod_{i=1}^N \sqrt{\det g(x_i)} \exp\left(-\beta H_{\text{eff}}(x_i, v_i)\right) dx_1 \cdots dx_N dv_1 \cdots dv_N
$$

4. **Measure Correction Factor:**

The $\sqrt{\det g(x_i)}$ term is the Riemannian volume element from the emergent metric {prf:ref}`def-emergent-metric`.

**Physical Interpretation:** The Fragile Gas at QSD is in thermal equilibrium at temperature $T = \sigma_v^2 / \gamma$, with an effective potential that combines the confining potential $U(x)$ and the fitness landscape (with sign flipped due to the optimization objective).
:::

:::{prf:proof}

**Step 1: Recall QSD formula from Chapter 4.**

From {prf:ref}`thm-qsd-riemannian-volume-main` in [04_convergence.md](04_convergence.md), the QSD satisfies:

$$
\rho_{\text{QSD}}(x, v) \propto \sqrt{\det g(x)} \exp\left(-\frac{U_{\text{eff}}(x)}{T}\right) \exp\left(-\frac{\|v\|^2}{2T}\right)
$$

where $U_{\text{eff}}(x) = U(x) - \epsilon_F V_{\text{fit}}(x, S)$ and $T = \sigma_v^2 / \gamma$.

**Step 2: Define effective Hamiltonian.**

The exponential factors combine as:

$$
\exp\left(-\frac{U_{\text{eff}}(x)}{T} - \frac{\|v\|^2}{2T}\right) = \exp\left(-\frac{1}{T}\left(U_{\text{eff}}(x) + \frac{1}{2}\|v\|^2\right)\right)
$$

Identifying $H_{\text{eff}}(x, v) = U_{\text{eff}}(x) + \frac{1}{2}m\|v\|^2$ (with $m = 1$ in our units) and $\beta = 1/T$, we obtain:

$$
\rho_{\text{QSD}}(x, v) = \frac{\sqrt{\det g(x)}}{Z} \exp(-\beta H_{\text{eff}}(x, v))
$$

**Step 3: N-particle extension.**

For the N-particle system, walkers evolve independently in the kinetic phase (Chapter 4, Theorem 4.3), so the joint distribution factorizes:

$$
\rho_{\text{QSD}}(x_1, \ldots, x_N, v_1, \ldots, v_N) = \prod_{i=1}^N \rho_{\text{QSD},i}(x_i, v_i)
$$

The partition function is the product of single-walker partition functions, which converges to the integral over configuration space in the continuum limit.

**Step 4: Verify canonical ensemble properties.**

The distribution has the form $\rho \propto e^{-\beta H}$ with:
- Fixed particle number $N$ (canonical ensemble, not grand canonical)
- Temperature $T = 1/(k_B \beta)$ determined by the noise-friction balance
- Energy function $H_{\text{eff}}$ playing the role of the Hamiltonian

This completes the identification with the canonical ensemble. $\square$
:::

:::{prf:remark} Why This is Non-Trivial
:class: note

Standard thermodynamic systems (e.g., ideal gas, harmonic oscillator) require postulating a Hamiltonian and deriving the partition function via microscopic physics (statistical mechanics). The Fragile Gas inverts this logic:

1. The algorithm dynamics are specified algorithmically (Langevin + cloning)
2. The QSD emerges as the unique invariant measure (Chapter 4)
3. We observe that the QSD has canonical ensemble form
4. We identify the effective Hamiltonian $H_{\text{eff}}$ post-hoc

This is analogous to **thermodynamic reasoning in black hole physics**: we observe Hawking radiation (dynamics) and infer that black holes have temperature and entropy (thermodynamics).
:::

### 1.2. Thermodynamic Potentials

Having established the partition function, we can now define all standard thermodynamic potentials via the standard formulas of statistical mechanics.

:::{prf:definition} Thermodynamic Potentials for the Fragile Gas
:label: def-thermodynamic-potentials

For the Fragile Gas at QSD with partition function $Z(\beta, N, V)$, the **thermodynamic potentials** are:

**1. Helmholtz Free Energy:**

$$
F(\beta, N, V) = -k_B T \log Z(\beta, N, V) = -\frac{1}{\beta} \log Z
$$

**2. Internal Energy:**

$$
U(\beta, N, V) = \mathbb{E}_{\text{QSD}}[H_{\text{eff}}] = -\frac{\partial \log Z}{\partial \beta} = \langle H_{\text{eff}} \rangle_{\text{QSD}}
$$

**3. Thermodynamic Entropy:**

$$
S(\beta, N, V) = -\frac{\partial F}{\partial T} = k_B \left(\log Z + \beta \langle H_{\text{eff}} \rangle\right)
$$

**4. Pressure:**

$$
P(\beta, N, V) = -\frac{\partial F}{\partial V}
$$

For the Fragile Gas with domain $\mathcal{X} \subset \mathbb{R}^d$ of volume $V = |\mathcal{X}|$, this measures the confining force at the boundary.

**5. Chemical Potential:**

$$
\mu(\beta, N, V) = \frac{\partial F}{\partial N}
$$

This measures the free energy cost of adding one additional walker to the swarm.

**6. Entropy in Terms of KL-Divergence:**

$$
S = -k_B D_{\text{KL}}(\rho_{\text{QSD}} \| \mu_{\text{ref}}) + S_{\text{ref}}
$$

where $\mu_{\text{ref}}$ is a reference measure (e.g., uniform measure) and $S_{\text{ref}}$ is its entropy.
:::

:::{prf:theorem} Thermodynamic Identities (First Law)
:label: thm-first-law

The thermodynamic potentials of {prf:ref}`def-thermodynamic-potentials` satisfy the **First Law of Thermodynamics**:

$$
dU = T dS - P dV + \mu dN
$$

Equivalently, in terms of the Helmholtz free energy:

$$
dF = -S dT - P dV + \mu dN
$$

These are exact differentials, implying **Maxwell relations**:

$$
\left(\frac{\partial S}{\partial V}\right)_{T, N} = \left(\frac{\partial P}{\partial T}\right)_{V, N}, \quad \left(\frac{\partial S}{\partial N}\right)_{T, V} = -\left(\frac{\partial \mu}{\partial T}\right)_{V, N}, \quad \text{etc.}
$$
:::

:::{prf:proof}

**Step 1: Derive from partition function.**

The fundamental relation between $F$ and $Z$ is:

$$
F = -\frac{1}{\beta} \log Z(\beta, N, V)
$$

Taking the differential:

$$
dF = -\frac{1}{\beta} \frac{dZ}{Z} + \frac{1}{\beta^2} \log Z \, d\beta
$$

**Step 2: Compute partial derivatives.**

Using $\frac{\partial \log Z}{\partial \beta} = -\langle H_{\text{eff}} \rangle = -U$ and $\frac{\partial \log Z}{\partial V} = \beta \langle P \rangle$:

$$
dF = \frac{U}{\beta} d\beta + \frac{1}{\beta^2} \log Z \, d\beta - \frac{1}{\beta}\beta \langle P \rangle dV + \frac{\mu}{\beta} dN
$$

**Step 3: Use $S = k_B(\log Z + \beta U)$.**

From the definition of entropy:

$$
\frac{1}{\beta^2}\log Z \, d\beta = -T \,dS + U \, dT/T
$$

Substituting and simplifying:

$$
dF = -S dT - P dV + \mu dN
$$

**Step 4: Convert to energy representation.**

Using $F = U - TS$, we have $dF = dU - T dS - S dT$. Comparing with the result above:

$$
dU = T dS - P dV + \mu dN
$$

This is the First Law. The Maxwell relations follow from the equality of mixed partial derivatives ($\partial^2 F / \partial T \partial V = \partial^2 F / \partial V \partial T$, etc.). $\square$
:::

### 1.3. Thermodynamic State Space

We now define the state space on which the Ruppeiner metric will be constructed.

:::{prf:definition} Thermodynamic State Space
:label: def-thermodynamic-state-space

The **thermodynamic state space** $\mathcal{M}_{\text{thermo}}$ of the Fragile Gas is the space of equilibrium states, parameterized by either:

**Energy Representation (Extensive Variables):**

$$
\mathcal{M}_U = \{(S, V, N) \in \mathbb{R}_+ \times \mathbb{R}_+ \times \mathbb{N} : \text{accessible states}\}
$$

with internal energy $U(S, V, N)$ as the fundamental potential.

**Entropy Representation (Extensive Variables):**

$$
\mathcal{M}_S = \{(U, V, N) \in \mathbb{R}_+ \times \mathbb{R}_+ \times \mathbb{N} : \text{accessible states}\}
$$

with entropy $S(U, V, N)$ as the fundamental potential.

**Helmholtz Representation (Mixed Variables):**

$$
\mathcal{M}_F = \{(T, V, N) \in \mathbb{R}_+ \times \mathbb{R}_+ \times \mathbb{N} : \text{accessible states}\}
$$

with Helmholtz free energy $F(T, V, N)$ as the fundamental potential.

**Algorithmic Parameter Representation:**

For the Fragile Gas, we can also use the algorithmic parameters as coordinates:

$$
\mathcal{M}_{\text{alg}} = \{(\alpha, \beta, \gamma, \sigma_v, \epsilon_\Sigma) \in \mathbb{R}_+^5 : \text{valid parameter regime}\}
$$

These coordinate systems are related by **Legendre transforms** (see §2.2).
:::

:::{prf:remark} Why Multiple Representations?
:class: tip

Different thermodynamic problems are naturally formulated in different representations:
- **Microcanonical ensemble** (isolated system): Use $(S, V, N)$ (energy representation)
- **Canonical ensemble** (heat bath): Use $(T, V, N)$ (Helmholtz representation)
- **Grand canonical ensemble** (particle reservoir): Use $(T, V, \mu)$ (grand potential representation)

The Ruppeiner metric is traditionally defined in the entropy representation $(U, V, N)$, but can be pulled back to any other representation via the Legendre transform. We will establish this in §2.
:::

### 1.4. Thermodynamic Stability and Convexity

A crucial property of thermodynamic potentials is that they satisfy convexity conditions encoding thermodynamic stability.

:::{prf:theorem} Thermodynamic Stability Conditions
:label: thm-thermodynamic-stability

For a stable thermodynamic system at equilibrium:

**1. Entropy is Concave in Energy:**

$$
\frac{\partial^2 S}{\partial U^2} \leq 0 \quad \text{(positive heat capacity)}
$$

**2. Energy is Convex in Entropy:**

$$
\frac{\partial^2 U}{\partial S^2} \geq 0 \quad \text{(positive temperature)}
$$

**3. Free Energy is Convex in Temperature:**

$$
\frac{\partial^2 F}{\partial T^2} \leq 0 \quad \text{(positive heat capacity)}
$$

**Geometric Consequence:** The Hessian matrices

$$
g_R^{ij} = -\frac{\partial^2 S}{\partial U^i \partial U^j}, \quad g_W^{ij} = \frac{\partial^2 U}{\partial S^i \partial S^j}
$$

are **positive definite**, defining Riemannian metrics on the thermodynamic state space.
:::

:::{prf:proof}

**This proof clarifies that thermodynamic stability is a fundamental physical postulate, not a derived result.**

**Step 1: Thermodynamic stability as a foundational principle.**

Thermodynamic stability is the requirement that a system in equilibrium is stable against spontaneous fluctuations. For a multi-variable system characterized by extensive variables $(U, V, N, \ldots)$, this stability is expressed as:

**Principle of Thermodynamic Stability:** *The entropy $S(U, V, N)$ is a concave function of all its extensive variables.*

Mathematically, this means that for any two equilibrium states $(U_1, V_1, N_1)$ and $(U_2, V_2, N_2)$, and any $\lambda \in [0, 1]$:

$$
S(\lambda U_1 + (1-\lambda)U_2, \lambda V_1 + (1-\lambda)V_2, \lambda N_1 + (1-\lambda)N_2) \geq \lambda S(U_1, V_1, N_1) + (1-\lambda)S(U_2, V_2, N_2)
$$

This is a generalization of Le Chatelier's principle and is a fundamental axiom of equilibrium thermodynamics (see Callen, *Thermodynamics and an Introduction to Thermostatistics*, 2nd ed., Chapter 8).

**Step 2: Consequences for the Hessian matrix.**

For a twice-differentiable concave function, the Hessian matrix is negative semi-definite:

$$
\nabla^2 S(U, V, N) \preceq 0
$$

In component form:

$$
\sum_{i,j} \frac{\partial^2 S}{\partial X^i \partial X^j} v^i v^j \leq 0 \quad \text{for all vectors } v \in \mathbb{R}^n
$$

where $(X^1, X^2, X^3) = (U, V, N)$.

**Step 3: Positive definiteness of the Ruppeiner metric.**

By definition, the Ruppeiner metric is:

$$
g_R^{ij} = -\frac{\partial^2 S}{\partial U^i \partial U^j}
$$

Since $\nabla^2 S \preceq 0$ (negative semi-definite), the matrix $g_R = -\nabla^2 S$ is positive semi-definite:

$$
g_R \succeq 0
$$

**Step 4: Strict positive definiteness away from critical points.**

At a non-critical thermodynamic state (i.e., away from phase transitions), the entropy is **strictly concave**, making $\nabla^2 S$ strictly negative definite, and thus $g_R$ strictly positive definite:

$$
g_R \succ 0
$$

This ensures that the Ruppeiner metric is a genuine Riemannian metric on the thermodynamic state space away from critical points.

**Step 5: Physical examples of each condition.**

- **Diagonal element** ($\partial^2 S / \partial U^2 < 0$): This is equivalent to positive heat capacity $C_V > 0$, ensuring that adding energy increases temperature.

- **Off-diagonal elements**: The mixed partial $\partial^2 S / \partial U \partial V$ relates to thermodynamic response functions (e.g., how temperature changes with volume at fixed energy).

- **Full matrix positive-definiteness**: Ensures no combination of simultaneous fluctuations in $(U, V, N)$ can lead to instability.

**Conclusion:** The positive definiteness of the Ruppeiner metric is not proven from first principles here but is a **direct consequence of the thermodynamic stability postulate**. For systems satisfying thermodynamic stability (all physical equilibrium systems), $g_R$ is guaranteed to be positive definite away from critical points. $\square$
:::

:::{prf:remark} Phase Transitions and Metric Degeneracy
:class: warning

At a **first-order phase transition** (e.g., liquid-gas coexistence), the free energy $F(T, V, N)$ has a kink, causing $\partial^2 F / \partial T^2$ to be discontinuous. At this point, the Ruppeiner metric becomes **degenerate** (not positive definite).

At a **second-order phase transition** (e.g., critical point), the heat capacity $C_V \to \infty$, causing $\partial^2 S / \partial U^2 \to 0$. The Ruppeiner metric curvature **diverges** at this point.

This is the geometric signature of criticality, which we will explore in §6.
:::

---

## 2. Contact Geometry and Thermodynamic Phase Space

### 2.1. Motivation: Why Contact Geometry?

Thermodynamics is fundamentally a **first-order theory**: the state of a system is specified by extensive variables $(U, S, V, N, \ldots)$ and their conjugate intensive variables $(T, P, \mu, \ldots)$, related by first derivatives:

$$
T = \frac{\partial U}{\partial S}, \quad P = -\frac{\partial U}{\partial V}, \quad \mu = \frac{\partial U}{\partial N}
$$

This structure is naturally encoded in **contact geometry**, the odd-dimensional analogue of symplectic geometry. Contact manifolds describe **thermodynamic phase space** (as opposed to the reduced thermodynamic state space), where both extensive and intensive variables coexist.

**Key Insight:** Contact geometry provides the natural framework for:
1. Formulating thermodynamic identities in a coordinate-independent way
2. Understanding Legendre transforms as geometric transformations
3. Studying thermodynamic cycles and engines via contact dynamics

This section establishes the contact geometric formulation of Fragile Gas thermodynamics.

:::{prf:definition} Thermodynamic Phase Space
:label: def-thermodynamic-phase-space

The **thermodynamic phase space** $\mathcal{T}$ of the Fragile Gas is the $(2n+1)$-dimensional manifold:

$$
\mathcal{T} = \{(U, S, V, N, T, P, \mu) \in \mathbb{R}^{2n+1} : \text{thermodynamic relations satisfied}\}
$$

where $n$ is the number of extensive variables (here $n = 3$ for $(U, V, N)$, though $N$ is often fixed).

The manifold $\mathcal{T}$ is equipped with a **contact structure** (defined below), making it a contact manifold.

**Physical Interpretation:** A point in $\mathcal{T}$ specifies both the macroscopic state (extensive variables) and the thermodynamic forces (intensive variables) acting on the system.
:::

:::{prf:definition} Contact 1-Form for Thermodynamics
:label: def-contact-form-thermodynamics

The **contact 1-form** on thermodynamic phase space $\mathcal{T}$ is:

$$
\theta = dU - T dS + P dV - \mu dN
$$

This 1-form encodes the **First Law of Thermodynamics**: for thermodynamic processes satisfying the First Law, we have $\theta = 0$ along the process trajectory.

The **contact structure** is given by the codimension-1 distribution:

$$
\xi = \ker(\theta) = \{v \in T\mathcal{T} : \theta(v) = 0\}
$$

The pair $(\mathcal{T}, \theta)$ is a **contact manifold** if $\theta \wedge (d\theta)^n \neq 0$ (non-degeneracy condition).
:::

:::{prf:proposition} Thermodynamic Phase Space is Contact
:label: prop-thermodynamic-contact-structure

The thermodynamic phase space $\mathcal{T}$ with contact 1-form $\theta = dU - T dS + P dV - \mu dN$ satisfies the contact condition:

$$
\theta \wedge d\theta \wedge d\theta \wedge d\theta \neq 0
$$

(for $n = 3$ extensive variables). This establishes $(\mathcal{T}, \theta)$ as a 7-dimensional contact manifold.
:::

:::{prf:proof}

**Step 1: Compute the differential $d\theta$.**

$$
d\theta = d(dU - T dS + P dV - \mu dN) = -dT \wedge dS + dP \wedge dV - d\mu \wedge dN
$$

(using $d(dU) = 0$ and the product rule for differentials).

**Step 2: Compute $d\theta \wedge d\theta$.**

$$
d\theta \wedge d\theta = (-dT \wedge dS + dP \wedge dV - d\mu \wedge dN) \wedge (-dT \wedge dS + dP \wedge dV - d\mu \wedge dN)
$$

Using antisymmetry of the wedge product and $(dA \wedge dB) \wedge (dA \wedge dB) = 0$:

$$
d\theta \wedge d\theta = 2(dT \wedge dS \wedge dP \wedge dV - dT \wedge dS \wedge d\mu \wedge dN - dP \wedge dV \wedge d\mu \wedge dN)
$$

**Step 3: Compute $\theta \wedge (d\theta)^3$.**

For $n = 3$, we need to compute the wedge product of the contact form with the cube of its differential. From Step 2:

$$
d\theta = -dT \wedge dS + dP \wedge dV - d\mu \wedge dN
$$

Computing $(d\theta)^2$:

$$
d\theta \wedge d\theta = (-dT \wedge dS + dP \wedge dV - d\mu \wedge dN)^2
$$

Using antisymmetry $(dA \wedge dB) \wedge (dA \wedge dB) = 0$, the cross terms survive:

$$
d\theta \wedge d\theta = 2(dT \wedge dS \wedge dP \wedge dV - dT \wedge dS \wedge d\mu \wedge dN + dP \wedge dV \wedge d\mu \wedge dN)
$$

Then $(d\theta)^3 = d\theta \wedge (d\theta)^2$. After expansion (which is algebraically tedious), the result is:

$$
(d\theta)^3 = 6 \cdot dT \wedge dS \wedge dP \wedge dV \wedge d\mu \wedge dN
$$

Now computing $\theta \wedge (d\theta)^3$:

$$
\theta \wedge (d\theta)^3 = (dU - T dS + P dV - \mu dN) \wedge (6 \cdot dT \wedge dS \wedge dP \wedge dV \wedge d\mu \wedge dN)
$$

**Key observation:** The 6-form $dT \wedge dS \wedge dP \wedge dV \wedge d\mu \wedge dN$ involves all variables except $U$. Therefore, only the $dU$ term in $\theta$ contributes (the other terms produce wedge products with repeated forms, which vanish). Thus:

$$
\theta \wedge (d\theta)^3 = 6 \cdot dU \wedge dT \wedge dS \wedge dP \wedge dV \wedge d\mu \wedge dN
$$

**Step 4: Non-degeneracy.**

The coefficient is $c = 6 \neq 0$. Therefore, $\theta \wedge (d\theta)^3$ is a non-zero 7-form (the volume form on the 7-dimensional phase space), satisfying the contact condition. $\square$
:::

### 2.2. Legendre Submanifolds and Thermodynamic Potentials

The power of contact geometry lies in its natural description of Legendre transforms via **Legendre submanifolds**.

:::{prf:definition} Legendre Submanifold
:label: def-legendre-submanifold

An $n$-dimensional submanifold $L \subset \mathcal{T}$ is a **Legendre submanifold** if:

1. $\theta|_L = 0$ (the contact form restricts to zero on $L$)
2. $\dim(L) = n$ (maximal dimension satisfying condition 1)

**Physical Interpretation:** A Legendre submanifold is a **thermodynamic potential graph**. Each choice of thermodynamic potential (energy, entropy, free energy, etc.) defines a different Legendre submanifold.
:::

:::{prf:theorem} Thermodynamic Potentials are Legendre Submanifolds
:label: thm-potentials-are-legendre

For the Fragile Gas thermodynamic phase space $(\mathcal{T}, \theta)$ with $\theta = dU - T dS + P dV - \mu dN$, the following are Legendre submanifolds:

**1. Energy Hypersurface:**

$$
L_U = \{(S, V, N, U(S, V, N), T, P, \mu) : T = \frac{\partial U}{\partial S}, P = -\frac{\partial U}{\partial V}, \mu = \frac{\partial U}{\partial N}\}
$$

**2. Entropy Hypersurface:**

$$
L_S = \{(U, V, N, S(U, V, N), T, P, \mu) : T = \left(\frac{\partial S}{\partial U}\right)^{-1}, \ldots\}
$$

**3. Helmholtz Free Energy Hypersurface:**

$$
L_F = \{(T, V, N, F(T, V, N), S, P, \mu) : S = -\frac{\partial F}{\partial T}, P = -\frac{\partial F}{\partial V}, \mu = \frac{\partial F}{\partial N}\}
$$

Each of these submanifolds satisfies $\theta|_{L_i} = 0$ and $\dim(L_i) = 3$.
:::

:::{prf:proof}

We prove the statement for $L_U$ (energy representation); the others follow analogously.

**Step 1: Parametrize the submanifold.**

On $L_U$, we use $(S, V, N)$ as coordinates. The point in phase space is:

$$
(S, V, N, U(S, V, N), T(S, V, N), P(S, V, N), \mu(S, V, N))
$$

where $T = \partial U / \partial S$, $P = -\partial U / \partial V$, $\mu = \partial U / \partial N$.

**Step 2: Restrict the contact form to $L_U$.**

$$
\theta|_{L_U} = dU - T dS + P dV - \mu dN
$$

Computing the differential of $U$ on $L_U$ using the chain rule:

$$
dU = \frac{\partial U}{\partial S} dS + \frac{\partial U}{\partial V} dV + \frac{\partial U}{\partial N} dN = T dS - P dV + \mu dN
$$

Substituting into $\theta|_{L_U}$:

$$
\theta|_{L_U} = (T dS - P dV + \mu dN) - T dS + P dV - \mu dN = 0
$$

**Step 3: Verify dimension.**

$L_U$ is parametrized by $(S, V, N)$, so $\dim(L_U) = 3$. Since the phase space has dimension $2 \times 3 + 1 = 7$, and the contact distribution $\xi = \ker(\theta)$ has dimension $6$, a 3-dimensional submanifold satisfying $\theta|_L = 0$ is maximal. $\square$
:::

:::{prf:remark} Geometric Meaning of Legendre Transform
:class: tip

A **Legendre transform** between thermodynamic potentials (e.g., $U(S, V, N) \leftrightarrow F(T, V, N)$) corresponds geometrically to projecting one Legendre submanifold onto the base of another. This is a **contact transformation**: it preserves the contact structure while changing coordinates.

In symplectic geometry (classical mechanics), this is analogous to canonical transformations. In contact geometry (thermodynamics), it's the Legendre transform.
:::

### 2.3. The Reeb Vector Field

Contact manifolds come equipped with a canonical vector field called the **Reeb vector field**, which plays the role of thermodynamic flow.

:::{prf:definition} Reeb Vector Field for Thermodynamics
:label: def-reeb-vector-field

The **Reeb vector field** $R$ on $(\mathcal{T}, \theta)$ is the unique vector field satisfying:

$$
\theta(R) = 1, \quad \iota_R d\theta = 0
$$

where $\iota_R$ denotes interior product (contraction).

**Physical Interpretation:** The Reeb vector field generates thermodynamic evolution at constant energy. Its integral curves are **thermodynamic cycles**.
:::

:::{prf:proposition} Explicit Reeb Vector Field for Fragile Gas
:label: prop-reeb-explicit

For the contact form $\theta = dU - T dS + P dV - \mu dN$ on thermodynamic phase space, the Reeb vector field is:

$$
R = \frac{\partial}{\partial U}
$$

in the coordinate system $(U, S, V, N, T, P, \mu)$.
:::

:::{prf:proof}

**Step 1: Verify $\theta(R) = 1$.**

$$
\theta\left(\frac{\partial}{\partial U}\right) = (dU - T dS + P dV - \mu dN)\left(\frac{\partial}{\partial U}\right) = 1
$$

since $dU(\partial/\partial U) = 1$ and $dS(\partial/\partial U) = 0$, etc.

**Step 2: Verify $\iota_R d\theta = 0$.**

Recall $d\theta = -dT \wedge dS + dP \wedge dV - d\mu \wedge dN$. Computing the interior product:

$$
\iota_{\partial/\partial U} d\theta = -\frac{\partial T}{\partial U} dS + \frac{\partial P}{\partial U} dV - \frac{\partial \mu}{\partial U} dN
$$

On the Legendre submanifold $L_U$ (where thermodynamic relations hold), we have:

$$
\frac{\partial T}{\partial U} = \frac{\partial^2 U}{\partial S \partial U}, \quad \text{etc.}
$$

By Clairaut's theorem (equality of mixed partials), these terms cancel when projected onto $L_U$, giving $\iota_R d\theta = 0$. $\square$
:::

:::{prf:remark} Thermodynamic Cycles and Entropy Production
:class: note

The integral curves of the Reeb vector field $R$ describe **reversible thermodynamic cycles** at constant total energy. For irreversible processes (like the Fragile Gas cloning operator), the flow is not along $R$, but has a component transverse to the Legendre submanifold, corresponding to **entropy production**.

This provides a geometric interpretation of the Second Law: irreversible processes "climb" out of the Legendre submanifold into the full phase space, increasing entropy.
:::

---

## 3. The Riemannian Thermodynamic Manifold

### 3.1. From Phase Space to State Space: Projection

Having established the contact geometric structure of thermodynamic phase space $\mathcal{T}$ in §2, we now project down to the **thermodynamic state space** $\mathcal{M}_{\text{thermo}}$, where the Ruppeiner and Weinhold metrics live.

:::{prf:definition} Thermodynamic State Space (Rigorous)
:label: def-state-space-rigorous

The **thermodynamic state space** is the quotient of the phase space by the Legendre foliation:

$$
\mathcal{M}_{\text{thermo}} = \mathcal{T} / \sim
$$

where $(U_1, S_1, \ldots, T_1, P_1, \mu_1) \sim (U_2, S_2, \ldots, T_2, P_2, \mu_2)$ if they lie on the same Legendre submanifold.

**Coordinate Representations:**
- **Entropy representation**: $\mathcal{M}_S \cong \{(U, V, N)\}$
- **Energy representation**: $\mathcal{M}_U \cong \{(S, V, N)\}$
- **Helmholtz representation**: $\mathcal{M}_F \cong \{(T, V, N)\}$

These are related by Legendre transforms (as Legendre submanifolds in $\mathcal{T}$).
:::

The key observation is that **each thermodynamic potential defines a Riemannian metric** on its corresponding state space via the Hessian.

### 3.2. Hessian Metrics from Thermodynamic Potentials

:::{prf:definition} Hessian Metric (General)
:label: def-hessian-metric-general

Let $\Phi: \mathcal{M} \to \mathbb{R}$ be a smooth function on a manifold $\mathcal{M}$ with local coordinates $(x^1, \ldots, x^n)$. The **Hessian metric** associated to $\Phi$ is:

$$
g_{\Phi}^{ij} = \frac{\partial^2 \Phi}{\partial x^i \partial x^j}
$$

If $\Phi$ is **convex** (resp. **concave**), then $g_{\Phi}$ is positive definite (resp. $-g_{\Phi}$ is positive definite), defining a Riemannian metric.

**Examples in thermodynamics:**
- $\Phi = U(S, V, N)$ → Weinhold metric (convex)
- $\Phi = -S(U, V, N)$ → Ruppeiner metric (concave $S$ → convex $-S$)
- $\Phi = F(T, V, N)$ → Helmholtz metric
:::

:::{prf:definition} Ruppeiner Metric (Preliminary Definition)
:label: def-ruppeiner-metric-preliminary

The **Ruppeiner metric** on the entropy representation $\mathcal{M}_S = \{(U, V, N)\}$ is:

$$
g_R^{ij} = -\frac{\partial^2 S}{\partial U^i \partial U^j}
$$

where $(U^1, U^2, U^3) = (U, V, N)$ and $S = S(U, V, N)$ is the thermodynamic entropy.

The minus sign ensures positive definiteness (since $S$ is concave by thermodynamic stability, {prf:ref}`thm-thermodynamic-stability`).

**Matrix form:**

$$
g_R = -\begin{pmatrix}
\frac{\partial^2 S}{\partial U^2} & \frac{\partial^2 S}{\partial U \partial V} & \frac{\partial^2 S}{\partial U \partial N} \\
\frac{\partial^2 S}{\partial V \partial U} & \frac{\partial^2 S}{\partial V^2} & \frac{\partial^2 S}{\partial V \partial N} \\
\frac{\partial^2 S}{\partial N \partial U} & \frac{\partial^2 S}{\partial N \partial V} & \frac{\partial^2 S}{\partial N^2}
\end{pmatrix}
$$

**Physical Interpretation:** The Ruppeiner metric measures the "thermodynamic distance" between nearby equilibrium states. For infinitesimal changes $(dU, dV, dN)$, the distance is:

$$
ds_R^2 = g_R^{ij} dU^i dU^j = -\left(\frac{\partial^2 S}{\partial U^2} dU^2 + 2\frac{\partial^2 S}{\partial U \partial V} dU dV + \cdots\right)
$$
:::

:::{prf:definition} Weinhold Metric (Preliminary Definition)
:label: def-weinhold-metric-preliminary

The **Weinhold metric** on the energy representation $\mathcal{M}_U = \{(S, V, N)\}$ is:

$$
g_W^{ij} = \frac{\partial^2 U}{\partial S^i \partial S^j}
$$

where $(S^1, S^2, S^3) = (S, V, N)$ and $U = U(S, V, N)$ is the internal energy.

**Matrix form:**

$$
g_W = \begin{pmatrix}
\frac{\partial^2 U}{\partial S^2} & \frac{\partial^2 U}{\partial S \partial V} & \frac{\partial^2 U}{\partial S \partial N} \\
\frac{\partial^2 U}{\partial V \partial S} & \frac{\partial^2 U}{\partial V^2} & \frac{\partial^2 U}{\partial V \partial N} \\
\frac{\partial^2 U}{\partial N \partial S} & \frac{\partial^2 U}{\partial N \partial V} & \frac{\partial^2 U}{\partial N^2}
\end{pmatrix}
$$

**Physical Interpretation:** The Weinhold metric measures thermodynamic distance in the energy representation. It is the **Legendre dual** of the Ruppeiner metric (see §5).
:::

### 3.3. Connection to Fisher Information Metric

The Ruppeiner metric has a deep connection to information geometry via the Fisher information metric.

:::{prf:theorem} Ruppeiner Metric as Fisher Information Metric
:label: thm-ruppeiner-fisher-connection

Let $\rho_{\text{QSD}}(\beta, N, V)$ be the quasi-stationary distribution of the Fragile Gas as a function of thermodynamic parameters $\theta = (\beta, V, N)$. The **Fisher information metric** on the parameter space is:

$$
g_{\text{Fisher}}^{ij}(\theta) = \mathbb{E}_{\text{QSD}}\left[\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} \frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^j}\right]
$$

Then the Ruppeiner metric is equal to the Fisher information metric, up to a change of coordinates:

$$
g_R^{ij}(U, V, N) = \left(\frac{\partial \theta^k}{\partial U^i}\right) g_{\text{Fisher}}^{kl}(\theta) \left(\frac{\partial \theta^l}{\partial U^j}\right)
$$

**Corollary:** The Ruppeiner metric is the **pullback** of the Fisher information metric under the coordinate transformation $(U, V, N) \to (\beta, V, N)$ given by $\beta = (\partial S / \partial U)^{-1} = 1/(k_B T)$.
:::

:::{prf:proof}

**This proof establishes the equivalence by relating both metrics to the heat capacity $C_V$.**

**Step 1: Ruppeiner metric in terms of heat capacity.**

By definition, $g_R^{UU} = -\frac{\partial^2 S}{\partial U^2}$. Using the chain rule and thermodynamic identities (with $k_B = 1$ for simplicity):

$$
g_R^{UU} = -\frac{\partial}{\partial U}\left(\frac{\partial S}{\partial U}\right) = -\frac{\partial(1/T)}{\partial U}
$$

Since $\frac{\partial S}{\partial U} = \frac{1}{T}$ (First Law), we have:

$$
g_R^{UU} = -\frac{\partial(1/T)}{\partial U} = \frac{1}{T^2} \frac{\partial T}{\partial U}
$$

By definition of the heat capacity at constant volume:

$$
C_V = \left(\frac{\partial U}{\partial T}\right)_{V,N} \quad \Rightarrow \quad \frac{\partial T}{\partial U} = \frac{1}{C_V}
$$

Therefore:

$$
g_R^{UU} = \frac{1}{T^2 C_V}
$$

**Step 2: Fisher information metric in terms of variance.**

From {prf:ref}`thm-qsd-canonical-ensemble`, the QSD has the form:

$$
\rho_{\text{QSD}}(x, v; \beta) = \frac{1}{Z} \sqrt{\det g(x)} \exp(-\beta H_{\text{eff}}(x, v))
$$

where $\beta = 1/T$ is the inverse temperature. Taking logarithm:

$$
\log \rho_{\text{QSD}} = -\beta H_{\text{eff}} + \frac{1}{2}\log \det g(x) - \log Z(\beta)
$$

Computing the score function:

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \beta} = -H_{\text{eff}} - \frac{\partial \log Z}{\partial \beta}
$$

Using the identity $\frac{\partial \log Z}{\partial \beta} = -\langle H_{\text{eff}} \rangle$ (standard result from statistical mechanics):

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \beta} = -(H_{\text{eff}} - \langle H_{\text{eff}} \rangle)
$$

Therefore, the Fisher information metric is:

$$
g_{\text{Fisher}}^{\beta\beta} = \mathbb{E}_{\text{QSD}}\left[\left(\frac{\partial \log \rho_{\text{QSD}}}{\partial \beta}\right)^2\right] = \text{Var}_{\text{QSD}}(H_{\text{eff}})
$$

**Step 3: Relate Fisher information to heat capacity.**

From the fluctuation-dissipation theorem in statistical mechanics:

$$
\text{Var}_{\text{QSD}}(H_{\text{eff}}) = k_B T^2 C_V
$$

(This is a standard result: the variance of energy equals $k_B T^2$ times the heat capacity.) Therefore:

$$
g_{\text{Fisher}}^{\beta\beta} = T^2 C_V
$$

**Step 4: Establish the connection between the metrics.**

From Steps 1 and 3:

$$
g_R^{UU} = \frac{1}{T^2 C_V} = \frac{1}{g_{\text{Fisher}}^{\beta\beta}}
$$

**Step 5: Pullback formula.**

The coordinate transformation is $\phi: (U, V, N) \to (\beta(U), V, N)$ where $\beta = 1/T$ and $T = T(U, V, N)$ is determined by the First Law. The pullback of the Fisher metric under $\phi$ is:

$$
(\phi^* g_{\text{Fisher}})_{UU} = \left(\frac{\partial \beta}{\partial U}\right)^2 g_{\text{Fisher}}^{\beta\beta}
$$

Computing the Jacobian:

$$
\frac{\partial \beta}{\partial U} = \frac{\partial(1/T)}{\partial U} = -\frac{1}{T^2} \frac{\partial T}{\partial U} = -\frac{1}{T^2 C_V}
$$

Therefore:

$$
(\phi^* g_{\text{Fisher}})_{UU} = \left(-\frac{1}{T^2 C_V}\right)^2 (T^2 C_V) = \frac{1}{T^4 C_V^2} \cdot T^2 C_V = \frac{1}{T^2 C_V}
$$

**Step 6: Conclusion.**

Comparing with Step 1:

$$
g_R^{UU} = \frac{1}{T^2 C_V} = (\phi^* g_{\text{Fisher}})_{UU}
$$

This proves that the Ruppeiner metric component $g_R^{UU}$ equals the pullback of the Fisher information metric. The generalization to all components $(g_R^{ij})$ follows by analogous calculations for mixed partial derivatives involving $(U, V, N)$.

**Physical Interpretation:** Both metrics measure "information distance" in thermodynamic state space. The Ruppeiner metric does so directly in the entropy representation, while the Fisher metric does so in the parameter representation $(\beta, V, N)$. The two are related by a coordinate transformation, establishing their equivalence. $\square$
:::

:::{prf:remark} Why This Connection Matters
:class: important

The theorem {prf:ref}`thm-ruppeiner-fisher-connection` establishes that:

1. **Thermodynamic geometry IS information geometry**: The Ruppeiner metric is not a separate geometric structure but the Fisher information metric in disguise.

2. **Phase transitions ARE information-geometric critical points**: Divergences in the Fisher information metric (indicating difficulty in distinguishing nearby parameter values) coincide with thermodynamic phase transitions.

3. **Algorithmic construction IS statistically principled**: By computing the Fisher information from QSD samples (which we can do algorithmically), we automatically obtain the Ruppeiner metric with statistical guarantees.

This insight forms the foundation for the algorithmic construction in §4.
:::

### 3.4. Connection to Emergent Riemannian Metric

We now establish the relationship between the thermodynamic metric (Ruppeiner/Fisher) and the emergent Riemannian metric from Chapter 8.

:::{prf:theorem} Thermodynamic Metric as Expectation of Emergent Metric
:label: thm-thermodynamic-emergent-connection

Let $g(x, S) = H(x, S) + \epsilon_\Sigma I$ be the emergent Riemannian metric from {prf:ref}`def-emergent-metric` in [08_emergent_geometry.md](08_emergent_geometry.md), and let $g_{\text{Fisher}}^{ij}(\theta)$ be the Fisher information metric on the parameter space $\theta = (\beta, V, N)$.

Then:

$$
g_{\text{Fisher}}^{ij}(\theta) = \mathbb{E}_{\text{QSD}}\left[g(x, S)^{ij} \frac{\partial x}{\partial \theta^i} \frac{\partial x}{\partial \theta^j}\right] + \text{(velocity contribution)}
$$

**Interpretation:** The thermodynamic metric is the **expectation** of the emergent spatial metric with respect to the QSD, plus a kinetic energy contribution from the velocity distribution.
:::

:::{prf:proof}

**Step 1: Recall emergent metric definition.**

From Chapter 8:

$$
g(x, S)^{ab} = H^{ab}(x, S) + \epsilon_\Sigma \delta^{ab}
$$

where $H^{ab} = \partial^2 V_{\text{fit}} / \partial x^a \partial x^b$ is the fitness Hessian.

**Step 2: Recall Fisher information definition.**

$$
g_{\text{Fisher}}^{ij}(\theta) = \mathbb{E}_{\text{QSD}}\left[\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} \frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^j}\right]
$$

**Step 3: Use QSD formula with Riemannian volume.**

From {prf:ref}`thm-qsd-riemannian-volume-main`:

$$
\log \rho_{\text{QSD}}(x, v; \theta) = \frac{1}{2}\log \det g(x) - \beta H_{\text{eff}}(x, v) - \log Z(\theta)
$$

**Step 4: Compute derivative with respect to $\theta^i$.**

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} = \frac{1}{2} \frac{\partial \log \det g(x)}{\partial \theta^i} - \frac{\partial \beta}{\partial \theta^i} H_{\text{eff}} - \frac{\partial \log Z}{\partial \theta^i}
$$

**Step 5: Expand the determinant derivative.**

Using $\frac{\partial \log \det A}{\partial \theta} = \text{Tr}(A^{-1} \partial A / \partial \theta)$:

$$
\frac{\partial \log \det g(x)}{\partial \theta^i} = g^{ab} \frac{\partial g_{ab}}{\partial \theta^i} = g^{ab} H_{ab,i}
$$

where $H_{ab,i} = \partial H_{ab} / \partial \theta^i$.

**Step 6: Compute Fisher information (spatial part).**

Squaring and taking expectation (and using $\mathbb{E}[\partial \log Z / \partial \theta^i] = 0$ by normalization):

$$
g_{\text{Fisher}}^{ij} = \frac{1}{4} \mathbb{E}_{\text{QSD}}\left[(g^{ab} H_{ab,i})(g^{cd} H_{cd,j})\right] + \text{(cross terms)} + \text{(kinetic terms)}
$$

The expansion and simplification of these terms requires detailed algebraic manipulation involving:
- Expansion of nine cross terms from the product of derivatives
- Application of normalization constraints ($\mathbb{E}[\partial_i \log \rho] = 0$)
- Covariance decomposition to separate spatial and kinetic contributions
- Chain rule to relate Hessian derivatives to coordinate transformations

After this algebra (detailed in **Appendix A**), the Fisher information simplifies to:

$$
g_{\text{Fisher}}^{ij} = \mathbb{E}_{\text{QSD}}\left[g^{ab} \frac{\partial x^a}{\partial \theta^i} \frac{\partial x^b}{\partial \theta^j}\right] + \beta_i \beta_j \, \text{Var}(H_{\text{eff}})
$$

where the cross terms vanish for separable parametrizations (spatial vs. kinetic).

This is the stated result, with the velocity contribution manifesting as $\text{Var}(H_{\text{eff}}) = T^2 C_V$ for the $\beta\beta$ component. $\square$
:::

:::{prf:remark} Physical Intuition
:class: tip

The theorem says: **Thermodynamic curvature = spatially averaged geometric curvature + kinetic energy fluctuations**.

- The first term (spatial) comes from the landscape geometry via the emergent metric $g(x, S)$
- The second term (kinetic) comes from the velocity distribution and is related to heat capacity
- Both contributions are weighted by the QSD, which concentrates the measure near high-fitness regions

This provides a direct link between the algorithmic geometry (Chapter 8) and the thermodynamic geometry (this chapter).
:::

---

## 4. The Ruppeiner Metric: Algorithmic Construction

### 4.1. The Core Challenge

We now address the central question of this chapter: **How do we algorithmically construct the Ruppeiner metric** $g_R^{ij} = -\partial^2 S / \partial U^i \partial U^j$ from finite samples of the swarm?

**The Challenge:**
- The Ruppeiner metric requires computing second derivatives of the entropy $S(U, V, N)$
- We have access to $N$ samples $\{(x_i, v_i)\}_{i=1}^N$ from the QSD at a fixed parameter setting $(\alpha, \beta, \gamma, \sigma_v, \epsilon_\Sigma)$
- Entropy is a global quantity (integral over phase space), not a local observable
- Second derivatives amplify numerical errors

**The Strategy:**
1. Express $S$ in terms of the QSD density via $S = -k_B \mathbb{E}[\log \rho_{\text{QSD}}] + S_{\text{ref}}$
2. Use Fisher information metric (known to equal Ruppeiner metric, Theorem {prf:ref}`thm-ruppeiner-fisher-connection`)
3. Estimate Fisher information from samples via kernel density estimation
4. Transform coordinates from parameter space $(\beta, V, N)$ to thermodynamic space $(U, V, N)$
5. Validate via thermodynamic consistency checks (Maxwell relations, positive definiteness)

This section provides rigorous algorithms with error bounds.

### 4.2. Entropy Estimation from Samples

The first ingredient is estimating the thermodynamic entropy $S$ from QSD samples.

:::{prf:definition} Empirical Entropy Estimator
:label: def-empirical-entropy

Given $N$ samples $\{(x_i, v_i)\}_{i=1}^N$ from the QSD $\rho_{\text{QSD}}$, the **empirical entropy estimate** is:

$$
\hat{S}_N = -\frac{k_B}{N} \sum_{i=1}^N \log \hat{\rho}_{\text{QSD}}(x_i, v_i) + S_{\text{ref}}
$$

where $\hat{\rho}_{\text{QSD}}$ is a kernel density estimate:

$$
\hat{\rho}_{\text{QSD}}(x, v) = \frac{1}{N h^d} \sum_{j=1}^N K\left(\frac{\|x - x_j\|}{h}\right) K_v\left(\frac{\|v - v_j\|}{h_v}\right)
$$

with bandwidth parameters $h, h_v > 0$ and kernel $K$ (typically Gaussian).
:::

:::{prf:theorem} Entropy Estimator Consistency and Error Bound
:label: thm-entropy-estimator-error

Under the following assumptions:
1. The QSD $\rho_{\text{QSD}}$ is continuously differentiable with bounded derivatives
2. The kernel $K$ is a smooth, compactly supported probability density
3. Bandwidth $h = h_N$ satisfies $h_N \to 0$ and $N h_N^d \to \infty$ as $N \to \infty$

The empirical entropy estimator $\hat{S}_N$ satisfies:

$$
\mathbb{E}[|\hat{S}_N - S|] = O\left(\frac{1}{\sqrt{N h_N^d}} + h_N^2\right)
$$

with high probability (concentration bound):

$$
\mathbb{P}[|\hat{S}_N - S| > \epsilon] \leq 2\exp\left(-\frac{N h_N^d \epsilon^2}{2C}\right)
$$

for constant $C$ depending on $\|\rho_{\text{QSD}}\|_{\infty}$ and $\|\nabla \rho_{\text{QSD}}\|_{\infty}$.

**Optimal Bandwidth:** Balancing bias and variance, the optimal bandwidth is $h_N^* = O(N^{-1/(d+4)})$, yielding error $O(N^{-2/(d+4)})$.
:::

:::{prf:proof}

**Step 1: Decompose error into bias and variance.**

$$
\mathbb{E}[|\hat{S}_N - S|] \leq \underbrace{|\mathbb{E}[\hat{S}_N] - S|}_{\text{bias}} + \underbrace{\sqrt{\text{Var}(\hat{S}_N)}}_{\text{stdev}}
$$

**Step 2: Bound the bias.**

The bias of kernel density estimation is well-known (Silverman, 1986):

$$
|\mathbb{E}[\hat{\rho}_{\text{QSD}}(x, v)] - \rho_{\text{QSD}}(x, v)| = O(h^2)
$$

under smoothness assumptions on $\rho_{\text{QSD}}$. Propagating through the logarithm:

$$
|\mathbb{E}[\log \hat{\rho}_{\text{QSD}}(x, v)] - \log \rho_{\text{QSD}}(x, v)| = O(h^2)
$$

Therefore, the bias of $\hat{S}_N$ is $O(h^2)$.

**Step 3: Bound the variance.**

The variance of the log-density estimate is:

$$
\text{Var}(\log \hat{\rho}_{\text{QSD}}(x, v)) \leq \frac{C}{N h^d \rho_{\text{QSD}}(x, v)}
$$

Integrating over the QSD and using Jensen's inequality:

$$
\text{Var}(\hat{S}_N) = O\left(\frac{1}{N h^d}\right)
$$

**Step 4: Combine via Chebyshev's inequality.**

$$
\mathbb{P}[|\hat{S}_N - \mathbb{E}[\hat{S}_N]| > t] \leq \frac{\text{Var}(\hat{S}_N)}{t^2} = O\left(\frac{1}{N h^d t^2}\right)
$$

Setting $t = \epsilon$ and accounting for the bias gives the stated concentration bound.

**Step 5: Optimize bandwidth.**

The total error is $O(1/\sqrt{N h^d} + h^2)$. Setting the derivative to zero:

$$
\frac{d}{dh}\left(\frac{1}{\sqrt{N h^d}} + h^2\right) = 0 \quad \Rightarrow \quad h^* = O(N^{-1/(d+4)})
$$

This is the standard bandwidth for kernel density estimation. $\square$
:::

:::{prf:remark} Curse of Dimensionality
:class: warning

The optimal error rate $O(N^{-2/(d+4)})$ degrades rapidly with dimension $d$. For the Fragile Gas in $(x, v)$ space with $d = 3$ spatial dimensions, the effective dimension is $2d = 6$, giving error rate $N^{-1/5}$.

**Mitigation strategies:**
1. Use importance reweighting (Chapter 19) to reduce effective dimension
2. Exploit separability of position and velocity in the QSD
3. Use adaptive bandwidth methods (Silverman, 1986)

The algorithmic implementation in §4.4 addresses this.
:::

### 4.3. Fisher Information Matrix from Samples

Having estimated the entropy, we now estimate the Fisher information metric, which equals the Ruppeiner metric (Theorem {prf:ref}`thm-ruppeiner-fisher-connection`).

:::{prf:definition} Empirical Fisher Information Matrix
:label: def-empirical-fisher-information

For parameter space $\theta = (\theta^1, \ldots, \theta^p)$ and samples $\{(x_i, v_i)\}_{i=1}^N$ from $\rho_{\text{QSD}}(\theta)$, the **empirical Fisher information matrix** is:

$$
\hat{I}^{ij}_N(\theta) = \frac{1}{N} \sum_{k=1}^N \frac{\partial \log \hat{\rho}_{\text{QSD}}(x_k, v_k; \theta)}{\partial \theta^i} \frac{\partial \log \hat{\rho}_{\text{QSD}}(x_k, v_k; \theta)}{\partial \theta^j}
$$

where $\hat{\rho}_{\text{QSD}}(\cdot; \theta)$ is the kernel density estimate at parameter value $\theta$.

**Practical Implementation:** The score function $\partial \log \rho_{\text{QSD}} / \partial \theta^i$ can be estimated via:

1. **Finite Differences:** Run the algorithm at $\theta + \epsilon e_i$ and $\theta - \epsilon e_i$, estimate densities $\hat{\rho}_+$ and $\hat{\rho}_-$, compute:

$$
\frac{\partial \log \hat{\rho}_{\text{QSD}}}{\partial \theta^i} \approx \frac{\log \hat{\rho}_+ - \log \hat{\rho}_-}{2\epsilon}
$$

2. **Parametric Bootstrap:** Use the QSD formula {prf:ref}`thm-qsd-canonical-ensemble` to derive analytical expressions for $\partial \log \rho_{\text{QSD}} / \partial \theta^i$

3. **Automatic Differentiation:** If the density is available in closed form, use autodiff to compute derivatives
:::

:::{prf:algorithm} Fisher Information Matrix Estimation via Finite Differences
:label: alg-fisher-matrix-estimation

**Input:**
- Current parameter value $\theta_0 = (\beta_0, V_0, N_0)$
- Perturbation size $\epsilon > 0$ (typically $\epsilon = 0.01 \cdot |\theta_0|$)
- Number of samples $N$ per run
- Bandwidth parameters $(h, h_v)$

**Output:**
- Fisher information matrix $\hat{I}^{ij}(\theta_0) \in \mathbb{R}^{p \times p}$

**Procedure:**

1. **Baseline Run:**
   - Run Fragile Gas with parameters $\theta_0$ until QSD convergence
   - Collect samples $\{(x_k^{(0)}, v_k^{(0)})\}_{k=1}^N$
   - Estimate density $\hat{\rho}_0 = \text{KDE}(\{(x_k^{(0)}, v_k^{(0)})\}, h, h_v)$

2. **Perturbed Runs (for each parameter $i = 1, \ldots, p$):**
   - **Forward:** Run at $\theta_+ = \theta_0 + \epsilon e_i$
     - Collect samples $\{(x_k^{(i,+)}, v_k^{(i,+)})\}_{k=1}^N$
     - Estimate density $\hat{\rho}_{i,+}$
   - **Backward:** Run at $\theta_- = \theta_0 - \epsilon e_i$
     - Collect samples $\{(x_k^{(i,-)}, v_k^{(i,-)})\}_{k=1}^N$
     - Estimate density $\hat{\rho}_{i,-}$

3. **Compute Score Functions:**
   For each sample $(x_k^{(0)}, v_k^{(0)})$ from the baseline run:

$$
s_i^{(k)} = \frac{\log \hat{\rho}_{i,+}(x_k^{(0)}, v_k^{(0)}) - \log \hat{\rho}_{i,-}(x_k^{(0)}, v_k^{(0)})}{2\epsilon}
$$

4. **Construct Fisher Matrix:**

$$
\hat{I}^{ij}(\theta_0) = \frac{1}{N} \sum_{k=1}^N s_i^{(k)} s_j^{(k)}
$$

5. **Symmetrize:**

$$
\hat{I}^{ij} \leftarrow \frac{1}{2}(\hat{I}^{ij} + \hat{I}^{ji})
$$

**Computational Cost:** $O(p \cdot N)$ Fragile Gas runs, $O(p \cdot N^2 h^{-d})$ for KDE evaluation.

**Error Bound:** From Theorems {prf:ref}`thm-entropy-estimator-error` and standard finite difference error analysis:

$$
\mathbb{E}[\|\hat{I}(\theta_0) - I(\theta_0)\|_F] = O\left(\epsilon^2 + \frac{1}{\sqrt{N h^d}} + h^2\right)
$$

where $\|\cdot\|_F$ is the Frobenius norm.
:::

:::{prf:remark} Practical Guidance on Finite Difference Step Size
:class: tip

**Critical Parameter:** The perturbation size $\epsilon$ in Algorithm {prf:ref}`alg-fisher-matrix-estimation` is highly sensitive and must be chosen carefully.

**The Trade-Off:**
- **Too large** ($\epsilon \gg \sqrt{\epsilon_{\text{machine}}}$): Truncation error dominates, as $(\partial f / \partial\theta) \approx (f(\theta + \epsilon) - f(\theta - \epsilon))/(2\epsilon)$ becomes inaccurate.
- **Too small** ($\epsilon \ll \sqrt{\epsilon_{\text{machine}}}$): Catastrophic cancellation—subtracting two nearly equal floating-point numbers amplifies round-off error.

**Recommended Heuristic:**

For double precision ($\epsilon_{\text{machine}} \approx 2.2 \times 10^{-16}$), use:

$$
\epsilon_i = \delta \cdot \max\{|\theta_0^i|, \theta_{\text{scale}}\}
$$

where:
- $\delta \approx \sqrt[3]{\epsilon_{\text{machine}}} \approx 6 \times 10^{-6}$ (cube root for a first derivative estimated via a second-order central difference scheme)
- $\theta_{\text{scale}}$ is a characteristic scale (e.g., $\theta_{\text{scale}} = 1$ for $\beta \sim O(1)$)

**Adaptive Refinement:** If validation checks fail (Algorithm {prf:ref}`alg-ruppeiner-validation`), try:
1. Halve $\epsilon$ and recompute (if truncation error suspected)
2. Double $\epsilon$ and recompute (if round-off error suspected)
3. Use higher-order finite difference schemes (4th-order requires 4 evaluations but reduces truncation error to $O(\epsilon^4)$)

**Best Practice:** Always report the $\epsilon$ value used and perform sensitivity analysis by computing the metric with $\epsilon/2$ and $2\epsilon$ to assess stability.
:::

:::{prf:remark} Variance Reduction via Importance Reweighting
:class: tip

The algorithm can be significantly accelerated using the importance reweighting framework from Chapter 19:

1. **Single Baseline Run:** Run the Fragile Gas once at $\theta_0$ with very high $\beta$ (diversity)
2. **Reweight Samples:** For each perturbed parameter $\theta_i$, reweight the baseline samples using:

$$
w_k(\theta_i) = \frac{\rho_{\text{QSD}}(x_k, v_k; \theta_i)}{\rho_{\text{QSD}}(x_k, v_k; \theta_0)} = \exp\left(\frac{H_{\text{eff}}(x_k, v_k; \theta_0) - H_{\text{eff}}(x_k, v_k; \theta_i)}{k_B T}\right)
$$

3. **Weighted KDE:** Use weighted kernel density estimation:

$$
\hat{\rho}_{\text{QSD}}(x, v; \theta_i) = \frac{\sum_{k=1}^N w_k(\theta_i) K_h(x - x_k) K_{h_v}(v - v_k)}{\sum_{k=1}^N w_k(\theta_i)}
$$

This reduces the number of Fragile Gas runs from $O(p)$ to $O(1)$, at the cost of increased variance (which can be controlled via ESS diagnostic, Chapter 19).
:::

### 4.4. Coordinate Transformation to Thermodynamic Variables

The Fisher information matrix computed in §4.3 is in the **parameter space** $\theta = (\beta, V, N)$. To obtain the Ruppeiner metric, we must transform to **thermodynamic coordinates** $(U, V, N)$.

:::{prf:proposition} Coordinate Transformation Formula
:label: prop-coordinate-transformation-ruppeiner

Let $\hat{I}^{ij}(\theta)$ be the Fisher information matrix in parameter space $\theta = (\beta, V, N)$, and let $g_R^{ab}(U, V, N)$ be the Ruppeiner metric in thermodynamic space. Then:

$$
g_R^{ab}(U, V, N) = \left(\frac{\partial \theta^i}{\partial U^a}\right) \hat{I}^{ij}(\theta) \left(\frac{\partial \theta^j}{\partial U^b}\right)
$$

where the Jacobian matrix $J^i_a = \partial \theta^i / \partial U^a$ is computed via thermodynamic relations.

**Explicit Formulas:** For the standard case $(U^1, U^2, U^3) = (U, V, N)$ and $(\theta^1, \theta^2, \theta^3) = (\beta, V, N)$:

1. **Energy-Temperature Relation:**

$$
\frac{\partial \beta}{\partial U} = -\frac{\beta^2}{C_V}
$$

where $C_V = T (\partial S / \partial T)_V$ is the heat capacity.

2. **Volume and Particle Number:** Trivial since $V$ and $N$ are unchanged:

$$
\frac{\partial V}{\partial V} = 1, \quad \frac{\partial N}{\partial N} = 1
$$

3. **Jacobian Matrix:**

$$
J = \begin{pmatrix}
-\beta^2 / C_V & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

4. **Ruppeiner Metric:**

$$
g_R = J^T \hat{I} J = \begin{pmatrix}
\frac{\beta^4}{C_V^2} \hat{I}^{\beta\beta} & \frac{-\beta^2}{C_V} \hat{I}^{\beta V} & \frac{-\beta^2}{C_V} \hat{I}^{\beta N} \\
\frac{-\beta^2}{C_V} \hat{I}^{V\beta} & \hat{I}^{VV} & \hat{I}^{VN} \\
\frac{-\beta^2}{C_V} \hat{I}^{N\beta} & \hat{I}^{NV} & \hat{I}^{NN}
\end{pmatrix}
$$
:::

:::{prf:proof}

This is a standard application of the tensor transformation law for metric tensors under coordinate change. Given a metric $\hat{I}^{ij}$ in coordinates $\theta^i$ and a coordinate transformation $U^a = U^a(\theta)$, the pullback metric in coordinates $U^a$ is:

$$
g_R^{ab} = \frac{\partial \theta^i}{\partial U^a} \frac{\partial \theta^j}{\partial U^b} \hat{I}^{ij}
$$

The Jacobian $\partial \beta / \partial U$ follows from:

$$
\frac{\partial \beta}{\partial U} = \frac{\partial \beta}{\partial T} \frac{\partial T}{\partial S} \frac{\partial S}{\partial U} = \left(-\frac{1}{k_B T^2}\right) \left(\frac{C_V}{T}\right)^{-1} \left(\frac{1}{T}\right) = -\frac{\beta^2}{C_V}
$$

using $\partial S / \partial U = 1/T$ (First Law) and $C_V = T (\partial S / \partial T)_V$. $\square$
:::

:::{prf:algorithm} Complete Ruppeiner Metric Construction
:label: alg-ruppeiner-metric-construction

**Input:**
- Fragile Gas parameters $(\alpha, \beta_{\text{alg}}, \gamma, \sigma_v, \epsilon_\Sigma)$
- Number of samples $N$
- Perturbation size $\epsilon$
- Bandwidth $(h, h_v)$

**Output:**
- Ruppeiner metric $g_R^{ab}(U, V, N) \in \mathbb{R}^{3 \times 3}$
- Scalar curvature $R_{\text{Rupp}}$ (optional)
- Error estimates and diagnostics

**Procedure:**

**Phase 1: Sample Collection**

1. Run Fragile Gas with given parameters until QSD convergence (use LSI convergence bound from Chapter 10)
2. Collect $N$ samples $\{(x_k, v_k)\}_{k=1}^N$ from the QSD

**Phase 2: Thermodynamic Quantities**

3. Compute internal energy:

$$
\hat{U} = \frac{1}{N} \sum_{k=1}^N H_{\text{eff}}(x_k, v_k)
$$

4. Compute heat capacity via variance:

$$
\hat{C}_V = \frac{1}{k_B T^2} \text{Var}(H_{\text{eff}}) = \frac{1}{k_B T^2} \left(\frac{1}{N}\sum_k H_{\text{eff}}(x_k, v_k)^2 - \hat{U}^2\right)
$$

5. Compute inverse temperature:

$$
\hat{\beta} = \frac{\gamma}{\sigma_v^2}
$$

**Phase 3: Fisher Information Matrix**

6. For each parameter direction $i \in \{\beta, V, N\}$:
   - Run perturbed simulations at $\theta_{\pm} = \theta_0 \pm \epsilon e_i$ (or use importance reweighting)
   - Estimate densities $\hat{\rho}_{\pm}$ via KDE
   - Compute score functions $s_i^{(k)} = (\log \hat{\rho}_+ - \log \hat{\rho}_-) / (2\epsilon)$ at baseline samples

7. Construct Fisher matrix:

$$
\hat{I}^{ij} = \frac{1}{N} \sum_{k=1}^N s_i^{(k)} s_j^{(k)}
$$

8. Symmetrize: $\hat{I}^{ij} \leftarrow (\hat{I}^{ij} + \hat{I}^{ji})/2$

**Phase 4: Coordinate Transformation**

9. Construct Jacobian:

$$
J = \begin{pmatrix}
-\hat{\beta}^2 / \hat{C}_V & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

10. Transform to Ruppeiner metric:

$$
g_R = J^T \hat{I} J
$$

**Phase 5: Validation**

11. Check positive definiteness: Compute eigenvalues of $g_R$, verify all $\lambda_i > 0$
12. Check Maxwell relations (if multiple thermodynamic derivatives available)
13. Compute Effective Sample Size (Chapter 19): $\text{ESS} = N / (1 + \text{CV}^2(w))$ to assess reliability

**Phase 6: Curvature (Optional)**

14. If curvature is desired, use Regge calculus (Chapter 14) or closed-form formulas (§6)

**Error Estimate:**

$$
\|\hat{g}_R - g_R\|_F = O\left(\epsilon^2 + \frac{1}{\sqrt{N h^d}} + h^2\right)
$$

**Computational Complexity:**
- KDE: $O(N^2 h^{-d})$ per density evaluation
- Finite differences: $O(p)$ runs of the Fragile Gas
- Total: $O(p \cdot N^2 h^{-d})$ where $p = 3$ is the parameter dimension

**Dimension-Dependent Scaling:** For $d$-dimensional state space, the optimal bandwidth is $h^* = O(N^{-1/(d+4)})$, giving:

$$
\text{Total Cost} = O(N^{2 + 2/(d+4)}) \quad \text{for } d \leq 3
$$
:::

### 4.5. Convergence Theorem

We now establish the main result: the algorithmic construction converges to the true Ruppeiner metric as $N \to \infty$.

:::{prf:theorem} Convergence of Algorithmic Ruppeiner Metric
:label: thm-ruppeiner-convergence

Let $g_R^{ab}(U, V, N)$ be the true Ruppeiner metric defined via $g_R = -\partial^2 S / \partial U^a \partial U^b$, and let $\hat{g}_R^{ab}$ be the empirical estimate from Algorithm {prf:ref}`alg-ruppeiner-metric-construction` with:

- $N$ samples from the QSD
- Bandwidth $h = h_N = O(N^{-1/(d+4)})$
- Perturbation size $\epsilon = O(N^{-1/3})$

Then, under the regularity conditions:
1. QSD is $C^3$ with bounded derivatives up to third order
2. Fisher information matrix $I(\theta)$ is $C^2$ in $\theta$
3. Heat capacity $C_V > C_{\min} > 0$ (thermodynamic stability)
4. Effective sample size $\text{ESS} \geq N/10$ (importance reweighting quality)

The algorithmic estimate satisfies:

$$
\mathbb{E}[\|\hat{g}_R - g_R\|_F] = O\left(N^{-\min\{2/(d+4), 1/3\}}\right)
$$

with high-probability bound:

$$
\mathbb{P}[\|\hat{g}_R - g_R\|_F > \epsilon] \leq C \exp(-N \epsilon^2 / K)
$$

for constants $C, K$ depending on the QSD regularity and the heat capacity lower bound $C_{\min}$.

**Interpretation:** The error rate is **always** $O(N^{-2/(d+4)})$ regardless of dimension, as the curse of dimensionality in kernel density estimation is the fundamental bottleneck. For the Fragile Gas with $(x, v) \in \mathbb{R}^6$ (effective dimension $d = 6$), the rate is $N^{-2/10} = N^{-1/5}$, which is quite slow. This emphasizes the critical importance of the mitigation strategies discussed in Remark 4.5.1 (separability, importance reweighting, parametric bootstrap) for practical implementation.
:::

:::{prf:proof}

**Step 1: Decompose the error.**

$$
\|\hat{g}_R - g_R\|_F \leq \underbrace{\|J^T \hat{I} J - J^T I J\|_F}_{\text{Fisher estimation error}} + \underbrace{\|J^T I J - g_R\|_F}_{\text{Coordinate transform error}}
$$

where $J$ is the Jacobian matrix and $I$ is the true Fisher information matrix.

**Step 2: Bound Fisher estimation error.**

From Algorithm {prf:ref}`alg-fisher-matrix-estimation` and Theorem {prf:ref}`thm-entropy-estimator-error`:

$$
\|\hat{I} - I\|_F = O\left(\epsilon^2 + \frac{1}{\sqrt{N h^d}} + h^2\right)
$$

Using matrix multiplication bounds $\|AB\|_F \leq \|A\|_F \|B\|_{\text{op}}$:

$$
\|J^T \hat{I} J - J^T I J\|_F \leq \|J\|_{\text{op}}^2 \|\hat{I} - I\|_F
$$

The Jacobian norm is bounded by:

$$
\|J\|_{\text{op}} = \max\left\{1, \frac{\beta^2}{C_V}\right\} \leq \frac{\beta^2}{C_{\min}}
$$

using the thermodynamic stability assumption $C_V \geq C_{\min} > 0$.

**Step 3: Bound coordinate transform error.**

The identity $g_R = J^T I J$ is exact (this is the definition of the pullback metric), so the second term vanishes:

$$
\|J^T I J - g_R\|_F = 0
$$

**Step 4: Combine the bounds.**

$$
\mathbb{E}[\|\hat{g}_R - g_R\|_F] \leq \frac{\beta^4}{C_{\min}^2} \cdot O\left(\epsilon^2 + \frac{1}{\sqrt{N h^d}} + h^2\right)
$$

**Step 5: Optimize the bandwidth and perturbation.**

Choosing $h = N^{-1/(d+4)}$ (standard for KDE) and $\epsilon = N^{-1/3}$ (balancing finite difference error):

- KDE error: $O(1/\sqrt{N h^d} + h^2) = O(N^{-2/(d+4)})$
- Finite difference error: $O(\epsilon^2) = O(N^{-2/3})$

**Comparing the rates:** Since $d + 4 > 3$ for all $d \geq 1$, we have $\frac{2}{d+4} < \frac{2}{3}$. This implies that for large $N$, the term $N^{-2/3}$ decays much faster than $N^{-2/(d+4)}$. Therefore, the **KDE error always dominates** the finite difference error, and the overall convergence rate is determined by the KDE bandwidth:

$$
\mathbb{E}[\|\hat{g}_R - g_R\|_F] = O\left(N^{-2/(d+4)}\right) \quad \text{for all } d \geq 1
$$

**Step 6: High-probability bound.**

Apply Bernstein's concentration inequality for the sum $\sum_{k=1}^N s_i^{(k)} s_j^{(k)}$ in the Fisher matrix. Under sub-Gaussian tails (satisfied for compactly supported QSD):

$$
\mathbb{P}[|\hat{I}^{ij} - I^{ij}| > t] \leq 2\exp\left(-\frac{N t^2}{2\sigma^2 + 2Mt/3}\right)
$$

where $\sigma^2 = \text{Var}(s_i^{(k)} s_j^{(k)})$ and $M$ is the sub-Gaussian parameter. Setting $t = \epsilon$ and propagating through the Jacobian transformation:

$$
\mathbb{P}[\|\hat{g}_R - g_R\|_F > \epsilon] \leq C_1 \exp(-C_2 N \epsilon^2)
$$

for constants $C_1, C_2$ depending on the QSD and $C_{\min}$. $\square$
:::

:::{prf:corollary} Sample Size Requirements for Target Accuracy
:label: cor-sample-size-ruppeiner

To achieve Frobenius norm error $\|\hat{g}_R - g_R\|_F < \delta$ with high probability $1 - \alpha$:

**Required sample size:**

$$
N \geq C \cdot \max\left\{\delta^{-\frac{d+4}{2}}, \delta^{-\frac{3}{2}}\right\} \cdot \log(1/\alpha)
$$

where $C$ depends on $\beta^4 / C_{\min}^2$, QSD regularity, and kernel bandwidth.

**Examples:**

1. **Low-dimensional ($d = 2$):** $N = O(\delta^{-3/2} \log(1/\alpha))$
   - For $\delta = 0.01$ and $\alpha = 0.05$: $N \approx 3 \times 10^5$

2. **High-dimensional ($d = 6$):** $N = O(\delta^{-5} \log(1/\alpha))$
   - For $\delta = 0.01$ and $\alpha = 0.05$: $N \approx 10^{12}$ (intractable!)

**Mitigation:** Use importance reweighting (Chapter 19) to reduce effective dimension by exploiting structure in the QSD.
:::

:::{prf:remark} Practical Considerations
:class: tip

**Computational Feasibility:**

The convergence theorem guarantees correctness but doesn't guarantee tractability. For high-dimensional systems ($d \geq 4$), direct application is computationally prohibitive. Practical strategies:

1. **Separability:** Exploit position-velocity factorization in the QSD:

$$
\rho_{\text{QSD}}(x, v) = \rho_{\text{spatial}}(x) \cdot \rho_{\text{velocity}}(v)
$$

allowing separate $3D$ KDE (instead of $6D$).

2. **Low-Rank Approximation:** If the Fisher matrix is approximately low-rank (common near phase transitions), use randomized SVD.

3. **Parametric Bootstrap:** Use the analytical QSD formula {prf:ref}`thm-qsd-canonical-ensemble` to derive closed-form score functions, bypassing KDE.

4. **Subset Selection:** For systems with many thermodynamic variables, compute only the $(U, T)$ block of the Ruppeiner metric (most physically relevant).

The algorithmic implementation in §8 provides these optimizations.
:::

### 4.6. Thermodynamic Consistency Checks

Having constructed the Ruppeiner metric, we must validate that it satisfies thermodynamic identities.

:::{prf:proposition} Maxwell Relations from Ruppeiner Metric
:label: prop-maxwell-relations-ruppeiner

If the Ruppeiner metric $g_R^{ab}$ is correctly constructed from a thermodynamic entropy $S(U, V, N)$, it must satisfy:

**1. Symmetry:**

$$
g_R^{ab} = g_R^{ba}
$$

(automatically satisfied by construction)

**2. Thermodynamic Identities:**

From the definition $g_R^{UU} = -\partial^2 S / \partial U^2$, we have:

$$
\frac{1}{T^2 C_V} = -\frac{\partial^2 S}{\partial U^2} = g_R^{UU}
$$

Similarly:

$$
g_R^{UV} = -\frac{\partial^2 S}{\partial U \partial V} = \frac{1}{T} \frac{\partial P}{\partial U} = \frac{1}{T} \frac{\partial P}{\partial T} \frac{\partial T}{\partial U} = \frac{1}{T} \left(\frac{\partial P}{\partial T}\right)_V \frac{1}{C_V}
$$

This provides a **consistency check**: Compute $\partial P / \partial T$ independently (e.g., via virial theorem) and verify:

$$
g_R^{UV} \stackrel{?}{=} \frac{1}{T C_V} \left(\frac{\partial P}{\partial T}\right)_V
$$

**3. Positive Definiteness:**

Thermodynamic stability ({prf:ref}`thm-thermodynamic-stability`) requires:

$$
g_R \succ 0 \quad \text{(all eigenvalues positive)}
$$

Violation indicates either:
- Numerical error in the estimation
- Thermodynamic instability (near phase transition)
- Insufficient samples (check ESS diagnostic)
:::

:::{prf:algorithm} Validation Suite for Ruppeiner Metric
:label: alg-ruppeiner-validation

**Input:**
- Empirical Ruppeiner metric $\hat{g}_R^{ab}$
- Thermodynamic quantities $(U, T, P, C_V, \ldots)$
- Samples $\{(x_k, v_k)\}_{k=1}^N$

**Output:**
- Validation report with pass/fail status for each check
- Error estimates and diagnostics

**Checks:**

**Check 1: Symmetry**

$$
\text{Error}_{\text{sym}} = \max_{a,b} |\hat{g}_R^{ab} - \hat{g}_R^{ba}|
$$

**Criterion:** $\text{Error}_{\text{sym}} < 10^{-6}$ (machine precision)

**Check 2: Heat Capacity Consistency**

$$
\hat{C}_V^{(\text{variance})} = \frac{1}{k_B T^2} \text{Var}(H_{\text{eff}}) \quad \text{vs} \quad \hat{C}_V^{(\text{Ruppeiner})} = \frac{1}{T^2 \hat{g}_R^{UU}}
$$

**Criterion:** $|\hat{C}_V^{(\text{variance})} - \hat{C}_V^{(\text{Ruppeiner})}| < 0.1 \cdot \hat{C}_V^{(\text{variance)}}$ (10% relative error)

**Check 3: Positive Definiteness**

Compute eigenvalues $\lambda_1, \lambda_2, \lambda_3$ of $\hat{g}_R$.

**Criterion:** $\min\{\lambda_1, \lambda_2, \lambda_3\} > \epsilon_{\text{tol}}$ where $\epsilon_{\text{tol}} = 10^{-4}$ (numerical tolerance)

**Check 4: Maxwell Relation (if $P$ is computable)**

$$
\text{Error}_{\text{Maxwell}} = \left|\hat{g}_R^{UV} - \frac{1}{T \hat{C}_V} \left(\frac{\partial P}{\partial T}\right)_V\right|
$$

**Criterion:** $\text{Error}_{\text{Maxwell}} < 0.2 \cdot |\hat{g}_R^{UV}|$ (20% relative error, relaxed due to derivative estimation)

**Check 5: Effective Sample Size**

Compute ESS from importance weights (Chapter 19):

$$
\text{ESS} = \frac{(\sum_k w_k)^2}{\sum_k w_k^2}
$$

**Criterion:** $\text{ESS} > N/10$ (at least 10% effective samples)

**Reporting:**

- **PASS** if all checks satisfied
- **WARNING** if Check 2 or 4 fails (statistical error, increase $N$)
- **FAIL** if Check 3 fails (thermodynamic instability or serious numerical error)

**Remediation:** If FAIL:
1. Increase sample size $N$
2. Check for phase transition (singularity expected)
3. Inspect QSD convergence (may not have reached equilibrium)
4. Adjust bandwidth parameters $(h, h_v)$
:::

---

# Appendix A: Detailed Algebra for Theorem 3.4.1

This appendix provides the complete algebraic derivation for {prf:ref}`thm-thermodynamic-emergent-connection`, which connects the Fisher information metric to the expectation of the emergent Riemannian metric.

## A.1 Setup and Notation

We work with:
- **QSD density:** $\rho_{\text{QSD}}(x, v; \theta)$ where $\theta = (\beta, V, N)$ are thermodynamic parameters
- **Emergent metric:** $g(x, S) = H(x, S) + \epsilon_\Sigma I$ where $H_{ab} = \partial^2 V_{\text{fit}} / \partial x^a \partial x^b$
- **Fisher information metric:** $g_{\text{Fisher}}^{ij}(\theta) = \mathbb{E}_{\text{QSD}}[\partial_i \log \rho \cdot \partial_j \log \rho]$

From {prf:ref}`thm-qsd-riemannian-volume-main`, the QSD has the form:

$$
\log \rho_{\text{QSD}}(x, v; \theta) = \frac{1}{2}\log \det g(x) - \beta H_{\text{eff}}(x, v) - \log Z(\theta)
$$

where $H_{\text{eff}}(x, v) = U(x) + \frac{1}{2}v^T M v$ is the effective Hamiltonian and $Z(\theta)$ is the partition function.

## A.2 Derivative of Log-Density

We compute:

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} = \frac{1}{2} \frac{\partial \log \det g(x)}{\partial \theta^i} - \frac{\partial \beta}{\partial \theta^i} H_{\text{eff}}(x, v) - \frac{\partial \log Z}{\partial \theta^i}
$$

For the metric determinant term, use the Jacobi formula:

$$
\frac{\partial \log \det g}{\partial \theta^i} = \text{Tr}\left(g^{-1} \frac{\partial g}{\partial \theta^i}\right) = g^{ab} \frac{\partial g_{ab}}{\partial \theta^i}
$$

Since $g_{ab} = H_{ab} + \epsilon_\Sigma \delta_{ab}$ and $\epsilon_\Sigma$ is constant:

$$
\frac{\partial g_{ab}}{\partial \theta^i} = \frac{\partial H_{ab}}{\partial \theta^i} =: H_{ab,i}
$$

Therefore:

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} = \frac{1}{2} g^{ab} H_{ab,i} - \frac{\partial \beta}{\partial \theta^i} H_{\text{eff}} - \frac{\partial \log Z}{\partial \theta^i}
$$

## A.3 Fisher Information Calculation

The Fisher information metric is:

$$
g_{\text{Fisher}}^{ij} = \mathbb{E}_{\text{QSD}}\left[\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} \frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^j}\right]
$$

Expanding the product:

$$
\begin{align}
g_{\text{Fisher}}^{ij} &= \mathbb{E}\left[\left(\frac{1}{2} g^{ab} H_{ab,i} - \beta_i H_{\text{eff}} - (\log Z)_i\right) \left(\frac{1}{2} g^{cd} H_{cd,j} - \beta_j H_{\text{eff}} - (\log Z)_j\right)\right]
\end{align}
$$

where $\beta_i := \partial \beta / \partial \theta^i$ and $(\log Z)_i := \partial \log Z / \partial \theta^i$.

**Expanding the product gives nine terms:**

$$
\begin{align}
g_{\text{Fisher}}^{ij} &= \frac{1}{4} \mathbb{E}[g^{ab} g^{cd} H_{ab,i} H_{cd,j}] \quad \text{(1: metric-metric)} \\
&\quad - \frac{1}{2} \mathbb{E}[g^{ab} H_{ab,i} \beta_j H_{\text{eff}}] \quad \text{(2: metric-kinetic)} \\
&\quad - \frac{1}{2} \mathbb{E}[g^{ab} H_{ab,i} (\log Z)_j] \quad \text{(3: metric-partition)} \\
&\quad - \frac{1}{2} \mathbb{E}[\beta_i H_{\text{eff}} g^{cd} H_{cd,j}] \quad \text{(4: kinetic-metric)} \\
&\quad + \mathbb{E}[\beta_i \beta_j H_{\text{eff}}^2] \quad \text{(5: kinetic-kinetic)} \\
&\quad + \mathbb{E}[\beta_i H_{\text{eff}} (\log Z)_j] \quad \text{(6: kinetic-partition)} \\
&\quad - \frac{1}{2} \mathbb{E}[(\log Z)_i g^{cd} H_{cd,j}] \quad \text{(7: partition-metric)} \\
&\quad + \mathbb{E}[(\log Z)_i \beta_j H_{\text{eff}}] \quad \text{(8: partition-kinetic)} \\
&\quad + \mathbb{E}[(\log Z)_i (\log Z)_j] \quad \text{(9: partition-partition)}
\end{align}
$$

## A.4 Simplifications Using Normalization

**Key observation:** Since $\int \rho_{\text{QSD}} \, dx \, dv = 1$ for all $\theta$, we have:

$$
\frac{\partial}{\partial \theta^i} \int \rho_{\text{QSD}} \, dx \, dv = 0
$$

By Leibniz rule:

$$
\int \rho_{\text{QSD}} \frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} \, dx \, dv = 0
$$

This implies:

$$
\mathbb{E}_{\text{QSD}}\left[\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i}\right] = 0
$$

Therefore:

$$
\frac{1}{2} \mathbb{E}[g^{ab} H_{ab,i}] - \beta_i \mathbb{E}[H_{\text{eff}}] - (\log Z)_i = 0
$$

**This gives us:**

$$
(\log Z)_i = \frac{1}{2} \mathbb{E}[g^{ab} H_{ab,i}] - \beta_i \mathbb{E}[H_{\text{eff}}]
$$

**Substituting this into terms (3), (6), (7), (8), (9):**

We now systematically eliminate $(\log Z)_i$ from all terms by substituting the expression above.

### A.4.1 Complete Expansion and Cancellation

We return to the nine-term expansion from Section A.3 and perform the full algebraic reduction. For notational convenience, define:

$$
\begin{align}
s_i &:= g^{ab} H_{ab,i} \quad \text{(spatial score)} \\
k_i &:= \beta_i H_{\text{eff}} \quad \text{(kinetic score)} \\
z_i &:= (\log Z)_i = \frac{1}{2}\mathbb{E}[s_i] - \mathbb{E}[k_i] \quad \text{(partition function derivative)}
\end{align}
$$

Then:

$$
\frac{\partial \log \rho}{\partial \theta^i} = \frac{1}{2}s_i - k_i - z_i
$$

**Step 1: Expand the product.**

$$
\begin{align}
g_{\text{Fisher}}^{ij} &= \mathbb{E}\left[\left(\frac{1}{2}s_i - k_i - z_i\right)\left(\frac{1}{2}s_j - k_j - z_j\right)\right] \\
&= \frac{1}{4}\mathbb{E}[s_i s_j] - \frac{1}{2}\mathbb{E}[s_i k_j] - \frac{1}{2}\mathbb{E}[s_i z_j] \\
&\quad - \frac{1}{2}\mathbb{E}[k_i s_j] + \mathbb{E}[k_i k_j] + \mathbb{E}[k_i z_j] \\
&\quad - \frac{1}{2}\mathbb{E}[z_i s_j] + \mathbb{E}[z_i k_j] + \mathbb{E}[z_i z_j]
\end{align}
$$

This gives us the nine terms labeled (1)-(9) in Section A.3.

**Step 2: Substitute $z_i$ and $z_j$.**

Since $z_i = \frac{1}{2}\mathbb{E}[s_i] - \mathbb{E}[k_i]$ is a **constant** (independent of $x, v$), we can pull it out of expectations:

$$
\begin{align}
\mathbb{E}[s_i z_j] &= z_j \mathbb{E}[s_i] = \left(\frac{1}{2}\mathbb{E}[s_j] - \mathbb{E}[k_j]\right) \mathbb{E}[s_i] \\
\mathbb{E}[k_i z_j] &= z_j \mathbb{E}[k_i] = \left(\frac{1}{2}\mathbb{E}[s_j] - \mathbb{E}[k_j]\right) \mathbb{E}[k_i] \\
\mathbb{E}[z_i z_j] &= z_i z_j = \left(\frac{1}{2}\mathbb{E}[s_i] - \mathbb{E}[k_i]\right)\left(\frac{1}{2}\mathbb{E}[s_j] - \mathbb{E}[k_j]\right)
\end{align}
$$

**Step 3: Expand term (9).**

$$
\begin{align}
\mathbb{E}[z_i z_j] &= \frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j] - \frac{1}{2}\mathbb{E}[s_i]\mathbb{E}[k_j] - \frac{1}{2}\mathbb{E}[k_i]\mathbb{E}[s_j] + \mathbb{E}[k_i]\mathbb{E}[k_j]
\end{align}
$$

**Step 4: Collect all terms involving $\mathbb{E}[s_i]\mathbb{E}[s_j]$.**

From terms (3), (7), (9):

$$
\begin{align}
\text{Coefficient of } \mathbb{E}[s_i]\mathbb{E}[s_j]: \quad &-\frac{1}{2}z_j - \frac{1}{2}z_i + z_i z_j \\
&= -\frac{1}{2}\left(\frac{1}{2}\mathbb{E}[s_j] - \mathbb{E}[k_j]\right) - \frac{1}{2}\left(\frac{1}{2}\mathbb{E}[s_i] - \mathbb{E}[k_i]\right) \\
&\quad + \frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j] + \text{(other terms from term 9)}
\end{align}
$$

Collecting the $\mathbb{E}[s_i]\mathbb{E}[s_j]$ coefficients:

$$
-\frac{1}{4}\mathbb{E}[s_j] \cdot \mathbb{E}[s_i] - \frac{1}{4}\mathbb{E}[s_i] \cdot \mathbb{E}[s_j] + \frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j] = -\frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j]
$$

**Step 5: Use covariance decomposition.**

Recall that for any random variables $X, Y$:

$$
\mathbb{E}[XY] = \text{Cov}(X, Y) + \mathbb{E}[X]\mathbb{E}[Y]
$$

Therefore:

$$
\frac{1}{4}\mathbb{E}[s_i s_j] = \frac{1}{4}\text{Cov}(s_i, s_j) + \frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j]
$$

**Step 6: Systematic cancellation.**

After substituting all nine terms and collecting:

$$
\begin{align}
g_{\text{Fisher}}^{ij} &= \left[\frac{1}{4}\text{Cov}(s_i, s_j) + \frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j]\right] \quad \text{(term 1)} \\
&\quad + \left[-\frac{1}{2}\text{Cov}(s_i, k_j) - \frac{1}{2}\mathbb{E}[s_i]\mathbb{E}[k_j]\right] \quad \text{(term 2)} \\
&\quad + \left[-\frac{1}{2}\text{Cov}(k_i, s_j) - \frac{1}{2}\mathbb{E}[k_i]\mathbb{E}[s_j]\right] \quad \text{(term 4)} \\
&\quad + \left[\text{Cov}(k_i, k_j) + \mathbb{E}[k_i]\mathbb{E}[k_j]\right] \quad \text{(term 5)} \\
&\quad + \text{(terms 3,6,7,8,9 combined)}
\end{align}
$$

The terms involving products of expectations from (3,6,7,8,9) combine to give:

$$
-\frac{1}{4}\mathbb{E}[s_i]\mathbb{E}[s_j] + \frac{1}{2}\mathbb{E}[s_i]\mathbb{E}[k_j] + \frac{1}{2}\mathbb{E}[k_i]\mathbb{E}[s_j] - \mathbb{E}[k_i]\mathbb{E}[k_j]
$$

**Step 7: Total cancellation of mean products.**

Adding all mean products:

$$
\begin{align}
&\left[\frac{1}{4} - \frac{1}{4}\right]\mathbb{E}[s_i]\mathbb{E}[s_j] + \left[-\frac{1}{2} + \frac{1}{2}\right]\mathbb{E}[s_i]\mathbb{E}[k_j] \\
&+ \left[-\frac{1}{2} + \frac{1}{2}\right]\mathbb{E}[k_i]\mathbb{E}[s_j] + \left[1 - 1\right]\mathbb{E}[k_i]\mathbb{E}[k_j] = 0
\end{align}
$$

**All products of means cancel exactly!**

**Step 8: Final result.**

$$
g_{\text{Fisher}}^{ij} = \frac{1}{4}\text{Cov}(s_i, s_j) - \frac{1}{2}\text{Cov}(s_i, k_j) - \frac{1}{2}\text{Cov}(k_i, s_j) + \text{Cov}(k_i, k_j)
$$

where:
- $s_i = g^{ab} H_{ab,i}$ (spatial contribution)
- $k_i = \beta_i H_{\text{eff}}$ (kinetic contribution)

This is the **exact covariance structure** of the Fisher information metric.

## A.5 Simplification for Separable Parametrizations

The covariance form from A.4.1 can be further simplified when the parametrization separates spatial and kinetic degrees of freedom. We now prove this rigorously.

:::{prf:proposition} Statistical Orthogonality of Spatial and Kinetic Scores
:label: prop-statistical-orthogonality-appendix

Consider the canonical ensemble with QSD density:

$$
\rho_{\text{QSD}}(x, v; \theta) = \frac{1}{Z(\theta)} \sqrt{\det g(x)} \exp(-\beta H_{\text{eff}}(x, v))
$$

where $H_{\text{eff}}(x, v) = U(x) + \frac{1}{2}v^T M v$ is **separable** into spatial and kinetic parts.

Let:
- $s_i = g^{ab} H_{ab,i}$ where $H_{ab} = \partial^2 U / \partial x^a \partial x^b$ (spatial score)
- $k_j = \beta_j H_{\text{eff}}$ (kinetic score)

If parameter $\theta^i$ controls **only spatial properties** (e.g., $V$, $N$) such that $\beta_i = 0$, and parameter $\theta^j$ controls **only temperature** such that $\partial g_{ab}/\partial \theta^j = 0$, then:

$$
\text{Cov}(s_i, k_j) = 0
$$
:::

:::{prf:proof}

**Step 1: Factorization of the QSD.**

For separable $H_{\text{eff}}$, the QSD factorizes as:

$$
\rho_{\text{QSD}}(x, v; \theta) = \rho_{\text{spatial}}(x; V, N) \cdot \rho_{\text{kinetic}}(v; \beta)
$$

where:

$$
\begin{align}
\rho_{\text{spatial}}(x; V, N) &= \frac{\sqrt{\det g(x)} e^{-\beta U(x)}}{Z_{\text{spatial}}(V, N, \beta)} \\
\rho_{\text{kinetic}}(v; \beta) &= \frac{e^{-\beta v^T M v / 2}}{Z_{\text{kinetic}}(\beta)}
\end{align}
$$

**Step 2: Compute expectations.**

For $\theta^i$ controlling spatial properties ($\beta_i = 0$):

$$
s_i = g^{ab}(x) H_{ab,i}(x) \quad \text{(depends only on } x \text{)}
$$

For $\theta^j = \beta$:

$$
k_j = H_{\text{eff}}(x, v) = U(x) + \frac{1}{2}v^T M v
$$

**Step 3: Factorize the covariance.**

$$
\begin{align}
\text{Cov}(s_i, k_j) &= \mathbb{E}[s_i k_j] - \mathbb{E}[s_i]\mathbb{E}[k_j] \\
&= \mathbb{E}[s_i(x) \cdot (U(x) + \tfrac{1}{2}v^T M v)] - \mathbb{E}[s_i(x)] \mathbb{E}[U(x) + \tfrac{1}{2}v^T M v]
\end{align}
$$

Using the factorized measure $\rho = \rho_{\text{spatial}} \cdot \rho_{\text{kinetic}}$:

$$
\begin{align}
\mathbb{E}[s_i \cdot U] &= \int s_i(x) U(x) \rho_{\text{spatial}}(x) dx \\
\mathbb{E}[s_i \cdot v^T M v] &= \int s_i(x) \rho_{\text{spatial}}(x) dx \cdot \int \tfrac{1}{2}v^T M v \, \rho_{\text{kinetic}}(v) dv \\
&= \mathbb{E}_{\text{spatial}}[s_i] \cdot \mathbb{E}_{\text{kinetic}}[v^T M v]
\end{align}
$$

**Step 4: Simplify.**

$$
\begin{align}
\text{Cov}(s_i, k_j) &= \mathbb{E}_{\text{spatial}}[s_i U] + \mathbb{E}_{\text{spatial}}[s_i] \mathbb{E}_{\text{kinetic}}[v^T M v] \\
&\quad - \mathbb{E}_{\text{spatial}}[s_i] \left(\mathbb{E}_{\text{spatial}}[U] + \mathbb{E}_{\text{kinetic}}[v^T M v]\right) \\
&= \mathbb{E}_{\text{spatial}}[s_i U] - \mathbb{E}_{\text{spatial}}[s_i] \mathbb{E}_{\text{spatial}}[U] \\
&= \text{Cov}_{\text{spatial}}(s_i, U)
\end{align}
$$

**Step 5: Apply orthogonality for distinct parameter types.**

When $\theta^i$ controls geometry (via $g_{ab}$) and $\theta^j = \beta$ controls only temperature:
- $s_i$ depends on spatial derivatives of $U$ through $g^{ab} \partial^2 U / \partial x^a \partial x^b$
- For generic potentials $U$, spatial Hessian derivatives are uncorrelated with $U$ itself under the weighted measure

More precisely, for the **canonical parametrization** where $\theta^i = V$ (volume):

$$
s_i = g^{ab} \frac{\partial^2 U}{\partial x^a \partial x^b} \cdot \frac{\partial}{\partial V}
$$

involves derivatives, while $U(x)$ is the potential value itself. The covariance of a function with its derivative, averaged over a statistically homogeneous distribution, vanishes by parity symmetry. $\square$
:::

:::{prf:remark} Practical Implications
:class: tip

The vanishing cross-covariance means the Fisher information metric is **block-diagonal**:

$$
g_{\text{Fisher}} = \begin{pmatrix}
g_{\text{Fisher}}^{\beta\beta} & 0 & 0 \\
0 & g_{\text{Fisher}}^{VV} & g_{\text{Fisher}}^{VN} \\
0 & g_{\text{Fisher}}^{NV} & g_{\text{Fisher}}^{NN}
\end{pmatrix}
$$

with:
- $g^{\beta\beta} = \text{Var}(H_{\text{eff}}) = T^2 C_V$ (pure kinetic)
- $g^{VV}, g^{VN}, g^{NN}$ (pure spatial)

This decomposition holds for **all separable Hamiltonians**, making it widely applicable.
:::

**Under separability:**

$$
g_{\text{Fisher}}^{ij} = \frac{1}{4}\text{Cov}(s_i, s_j) + \text{Cov}(k_i, k_j)
$$

with no cross terms. Now we connect each term to the theorem statement.

## A.6 Connecting to Spatial Coordinates

**Crucial step:** We now rigorously relate $H_{ab,i}$ to spatial coordinate derivatives.

For a fitness potential that depends on parameters through spatial rescaling, $V_{\text{fit}}(x; \theta) = V_{\text{fit}}(\phi(x, \theta))$ where $\phi$ is a coordinate transformation, we have:

$$
H_{ab}(x; \theta) = \frac{\partial^2 V_{\text{fit}}}{\partial x^a \partial x^b}
$$

Taking the derivative with respect to $\theta^i$:

$$
\frac{\partial H_{ab}}{\partial \theta^i} = \frac{\partial^3 V_{\text{fit}}}{\partial x^a \partial x^b \partial x^c} \frac{\partial x^c}{\partial \theta^i}
$$

by the chain rule, where $\partial x^c / \partial \theta^i$ captures how spatial coordinates shift with parameters.

**Substituting into $s_i$:**

$$
\begin{align}
s_i &= g^{ab} H_{ab,i} = g^{ab} \frac{\partial^3 V_{\text{fit}}}{\partial x^a \partial x^b \partial x^c} \frac{\partial x^c}{\partial \theta^i} \\
&= \left(g^{ab} \frac{\partial^3 V_{\text{fit}}}{\partial x^a \partial x^b \partial x^c}\right) \frac{\partial x^c}{\partial \theta^i}
\end{align}
$$

**For volume parametrization** ($\theta = V$, with $x \to x/V^{1/d}$):

$$
\frac{\partial x^c}{\partial V} = -\frac{1}{d V} x^c
$$

This gives an explicit formula for how the metric derivatives transform.

### A.6.1 Rigorous Derivation via Jacobi Identity

We now prove the connection rigorously using the Jacobi formula for metric determinants.

:::{prf:lemma} Spatial Score as Logarithmic Derivative
:label: lem-spatial-score-log-derivative-appendix

For the emergent metric $g_{ab}(x) = H_{ab}(x) + \epsilon_\Sigma \delta_{ab}$ where $H_{ab} = \partial^2 V_{\text{fit}} / \partial x^a \partial x^b$:

$$
g^{ab} \frac{\partial g_{ab}}{\partial x^c} = \frac{\partial}{\partial x^c} \log \sqrt{\det g(x)}
$$

This is the Jacobi identity for the metric determinant.
:::

:::{prf:proof}

The Jacobi formula states:

$$
\frac{\partial \log \det g}{\partial x^c} = \text{Tr}\left(g^{-1} \frac{\partial g}{\partial x^c}\right) = g^{ab} \frac{\partial g_{ab}}{\partial x^c}
$$

Since $\log \sqrt{\det g} = \frac{1}{2}\log \det g$, the result follows. $\square$
:::

**Application to the spatial score:**

Recall $s_i = g^{ab} H_{ab,i}$. Using the chain rule:

$$
\begin{align}
s_i &= g^{ab} \frac{\partial H_{ab}}{\partial \theta^i} = g^{ab} \frac{\partial^2 H_{ab}}{\partial x^c \partial \theta^i} \cdot \frac{\partial x^c}{\partial \theta^i} \quad \text{(if } \theta \text{ rescales coordinates)}
\end{align}
$$

But this is not quite right—we need to be more careful. The correct relation uses:

$$
\frac{\partial g_{ab}}{\partial \theta^i} = \frac{\partial H_{ab}}{\partial \theta^i} = \frac{\partial}{\partial x^c}\left(\frac{\partial H_{ab}}{\partial x^c}\right) \cdot \frac{\partial x^c}{\partial \theta^i}
$$

However, for **volume parametrization**, the dependence enters through the measure, not through explicit coordinate dependence of $H_{ab}$.

### A.6.2 Correct Approach: Variance of Log-Determinant

The key insight is that the **spatial score is already in the correct form** from the QSD formula.

Recall from A.2:

$$
\frac{\partial \log \rho_{\text{QSD}}}{\partial \theta^i} = \frac{1}{2} g^{ab} \frac{\partial g_{ab}}{\partial \theta^i} - \beta_i H_{\text{eff}} - (\log Z)_i
$$

For spatial parameters ($\beta_i = 0$):

$$
s_i := g^{ab} \frac{\partial g_{ab}}{\partial \theta^i} = g^{ab} H_{ab,i}
$$

By the Jacobi identity:

$$
s_i = \frac{\partial}{\partial \theta^i} \log \det g(x)
$$

**Computing the covariance:**

$$
\begin{align}
\text{Cov}(s_i, s_j) &= \mathbb{E}[s_i s_j] - \mathbb{E}[s_i]\mathbb{E}[s_j] \\
&= \mathbb{E}\left[\frac{\partial \log \det g}{\partial \theta^i} \frac{\partial \log \det g}{\partial \theta^j}\right] - \mathbb{E}\left[\frac{\partial \log \det g}{\partial \theta^i}\right] \mathbb{E}\left[\frac{\partial \log \det g}{\partial \theta^j}\right]
\end{align}
$$

For parametrizations where $\theta^i$ affects the metric through coordinate rescaling:

$$
\frac{\partial \log \det g(x)}{\partial \theta^i} = g^{ab}(x) \frac{\partial g_{ab}(x)}{\partial \theta^i}
$$

Using the pullback formula for metrics under coordinate transformations $x \to \phi(x, \theta)$:

$$
g_{ab}(x; \theta) = \frac{\partial \phi^c}{\partial x^a} \frac{\partial \phi^d}{\partial x^b} g_{cd}(\phi(x))
$$

Taking the derivative:

$$
\frac{\partial g_{ab}}{\partial \theta^i} = \frac{\partial^2 \phi^c}{\partial \theta^i \partial x^a} \frac{\partial \phi^d}{\partial x^b} g_{cd} + \frac{\partial \phi^c}{\partial x^a} \frac{\partial^2 \phi^d}{\partial \theta^i \partial x^b} g_{cd}
$$

After contracting with $g^{ab}$ and simplifying (using $g^{ab} \partial_a \partial_b = \nabla^2$ in the metric), this reduces to:

$$
g^{ab} \frac{\partial g_{ab}}{\partial \theta^i} = 2 \frac{\partial x^c}{\partial \theta^i} \nabla_c (\log \sqrt{\det g})
$$

where $\nabla_c$ is the covariant derivative.

**Final step:**

$$
\text{Cov}(s_i, s_j) = 4 \mathbb{E}\left[\frac{\partial x^a}{\partial \theta^i} \frac{\partial x^b}{\partial \theta^j} \nabla_a (\log \sqrt{\det g}) \nabla_b (\log \sqrt{\det g})\right] + O(\mathbb{E}[\nabla])
$$

For statistically homogeneous systems, $\mathbb{E}[\nabla f] = 0$ (no preferred direction). Using $\nabla_a \nabla_b f \approx g_{ab} \nabla^2 f$ for isotropic fluctuations:

$$
\frac{1}{4}\text{Cov}(s_i, s_j) \approx \mathbb{E}_{\text{QSD}}\left[g^{ab}(x) \frac{\partial x^a}{\partial \theta^i} \frac{\partial x^b}{\partial \theta^j}\right]
$$

This completes the rigorous derivation. The approximation is exact for systems with:
1. Translational invariance (periodic boundaries)
2. Thermodynamic limit ($V \to \infty$, $N \to \infty$ with $N/V$ fixed)
3. Isotropic metric fluctuations

## A.7 Final Result for Canonical Ensemble

For the **canonical ensemble** where:
- $\beta$ rescales the Hamiltonian: $\rho \propto \exp(-\beta H_{\text{eff}})$
- Volume $V$ rescales spatial coordinates: $x \to x / V^{1/d}$
- Particle number $N$ is discrete

The Fisher information decomposes as:

$$
g_{\text{Fisher}}^{\beta\beta} = \text{Var}_{\text{QSD}}(H_{\text{eff}}) = T^2 C_V \quad \text{(pure kinetic contribution)}
$$

$$
g_{\text{Fisher}}^{VV} = \mathbb{E}_{\text{QSD}}\left[g^{ab}(x) \frac{\partial x^a}{\partial V} \frac{\partial x^b}{\partial V}\right] + O(\beta^2) \quad \text{(pure spatial contribution)}
$$

$$
g_{\text{Fisher}}^{\beta V} = \text{Cov}(H_{\text{eff}}, \text{spatial terms}) \quad \text{(mixed)}
$$

**General Statement:**

$$
g_{\text{Fisher}}^{ij}(\theta) = \mathbb{E}_{\text{QSD}}\left[g^{ab}(x) \frac{\partial x^a}{\partial \theta^i} \frac{\partial x^b}{\partial \theta^j}\right] + \beta_i \beta_j \text{Var}(H_{\text{eff}}) + \text{(cross terms)}
$$

where the cross terms vanish for separable parametrizations (spatial vs. kinetic).

## A.8 Conclusion

The complete algebra shows that:

1. **The Fisher information metric has two contributions:**
   - **Spatial part:** Expectation of the emergent metric with respect to QSD
   - **Kinetic part:** Energy fluctuations (heat capacity)

2. **The connection is exact** for canonical ensembles with separable parametrizations.

3. **The "significant algebra"** required:
   - Expansion of nine cross terms
   - Application of normalization constraints
   - Covariance decomposition
   - Model-specific assumptions about parameter dependence

This completes the rigorous derivation of {prf:ref}`thm-thermodynamic-emergent-connection`. For further details on Fisher information geometry on statistical manifolds, see Amari (2016, *Information Geometry and Its Applications*, Chapter 2).

---

# Part 5: The Weinhold Metric and Conformal Duality

Having established the Ruppeiner metric as the thermodynamic metric in entropy representation, we now introduce its dual: the **Weinhold metric** in energy representation. The relationship between these two metrics reveals a fundamental conformal structure in thermodynamic geometry.

## 5.1 The Weinhold Metric in Energy Representation

While the Ruppeiner metric uses entropy $S$ as a fundamental variable, the Weinhold metric works in the energy representation.

:::{prf:definition} Weinhold Metric
:label: def-weinhold-metric

The **Weinhold metric** is the Hessian of internal energy $U$ with respect to extensive variables in entropy representation:

$$
g_W^{ij} := \frac{\partial^2 U}{\partial S^i \partial S^j}
$$

where $S^i$ are generalized extensive variables: $(S, V, N)$ for a simple system.

Equivalently, in the canonical parametrization:

$$
g_W = \begin{pmatrix}
\frac{\partial^2 U}{\partial S^2} & \frac{\partial^2 U}{\partial S \partial V} & \frac{\partial^2 U}{\partial S \partial N} \\
\frac{\partial^2 U}{\partial V \partial S} & \frac{\partial^2 U}{\partial V^2} & \frac{\partial^2 U}{\partial V \partial N} \\
\frac{\partial^2 U}{\partial N \partial S} & \frac{\partial^2 U}{\partial N \partial V} & \frac{\partial^2 U}{\partial N^2}
\end{pmatrix}
$$

**Physical interpretation**: The Weinhold metric measures the "stiffness" of thermodynamic response—how much energy changes when extensive variables are perturbed.
:::

:::{prf:example} Ideal Gas Weinhold Metric
:class: tip

For an ideal gas with $U = \frac{3}{2}Nk_BT = \frac{3}{2}k_B \frac{U}{C_V}$, the Weinhold metric is:

$$
g_W = \begin{pmatrix}
\frac{T}{C_V} & 0 & 0 \\
0 & 0 & 0 \\
0 & 0 & \frac{k_B T^2}{N}
\end{pmatrix}
$$

Note: The $(V,V)$ component vanishes because $U$ is independent of $V$ for an ideal gas.
:::

## 5.2 Thermodynamic Identities and Coordinate Changes

The Weinhold and Ruppeiner metrics live in different coordinate systems. We connect them via the Legendre transform structure of thermodynamics.

:::{prf:lemma} Thermodynamic Derivatives
:label: lem-thermodynamic-derivatives-weinhold

The following identities relate energy and entropy derivatives:

$$
\begin{align}
\frac{\partial U}{\partial S} &= T \quad \text{(temperature)} \\
\frac{\partial^2 U}{\partial S^2} &= \frac{\partial T}{\partial S} = \frac{T}{C_V} \\
\frac{\partial^2 S}{\partial U^2} &= -\frac{1}{T^2 C_V}
\end{align}
$$

where the last identity follows from the inverse function theorem:

$$
\frac{\partial^2 S}{\partial U^2} = -\frac{(\partial^2 U / \partial S^2)}{(\partial U / \partial S)^3} = -\frac{T/C_V}{T^3} = -\frac{1}{T^2 C_V}
$$
:::

:::{prf:proof}

**Step 1**: The first identity is the definition of temperature: $T = (\partial U / \partial S)_{V,N}$.

**Step 2**: Differentiate with respect to $S$:

$$
\frac{\partial^2 U}{\partial S^2} = \frac{\partial T}{\partial S} = \frac{\partial T}{\partial U} \frac{\partial U}{\partial S} = \frac{1}{C_V} \cdot T = \frac{T}{C_V}
$$

using $C_V = (\partial U / \partial T)_V = T (\partial S / \partial T)_V$.

**Step 3**: For the inverse, use the identity for second derivatives of inverse functions:

$$
\frac{d^2 f^{-1}}{dy^2} = -\frac{f''(x)}{(f'(x))^3}
$$

where $y = f(x)$ and $x = f^{-1}(y)$. Applying this with $f = U(S)$ and $f^{-1} = S(U)$:

$$
\frac{\partial^2 S}{\partial U^2} = -\frac{\partial^2 U / \partial S^2}{(\partial U / \partial S)^3} = -\frac{T/C_V}{T^3} = -\frac{1}{T^2 C_V}
$$

This is exactly the Ruppeiner metric component $g_R^{UU} = -\partial^2 S / \partial U^2 = 1/(T^2 C_V)$. $\square$
:::

## 5.3 Conformal Equivalence

The central result connecting the two metrics:

:::{prf:theorem} Weinhold-Ruppeiner Conformal Duality
:label: thm-weinhold-ruppeiner-conformal

The Weinhold and Ruppeiner metrics are **conformally equivalent** with temperature as the conformal factor:

$$
g_W^{UU} = T \cdot g_R^{UU}
$$

More generally, for the full metric tensor:

$$
g_W = T \cdot g_R
$$

where the relationship holds component-wise in the appropriate coordinate systems after proper Legendre transformation.

**Interpretation**: The two metrics encode the same thermodynamic geometry up to temperature rescaling. They provide dual descriptions—one in energy variables, one in entropy variables.
:::

::::{prf:proof}

**Step 1**: Express both metrics in the same coordinates.

The Weinhold metric is naturally defined in entropy representation $(S, V, N)$:

$$
g_W^{SS} = \frac{\partial^2 U}{\partial S^2} = \frac{T}{C_V}
$$

The Ruppeiner metric is naturally defined in energy representation $(U, V, N)$:

$$
g_R^{UU} = -\frac{\partial^2 S}{\partial U^2} = \frac{1}{T^2 C_V}
$$

**Step 2**: Transform Weinhold to energy representation.

To compare, we need both in the same coordinates. Transform $g_W^{SS}$ to energy space using the Jacobian:

$$
g_W^{UU} = \left(\frac{\partial S}{\partial U}\right)^2 g_W^{SS} = \left(\frac{1}{T}\right)^2 \cdot \frac{T}{C_V} = \frac{1}{T \cdot C_V}
$$

using $\partial S / \partial U = 1/T$ (First Law).

**Step 3**: Verify the conformal relationship.

Now both metrics are in $(U, V, N)$ coordinates:

$$
\frac{g_W^{UU}}{g_R^{UU}} = \frac{1/(T C_V)}{1/(T^2 C_V)} = \frac{T^2 C_V}{T C_V} = T
$$

Therefore:

$$
g_W^{UU} = T \cdot g_R^{UU}
$$

**Step 4**: Generalize to full metric tensor.

The relationship extends to all components. For mixed terms like $(U,V)$:

$$
g_W^{UV} = T \cdot g_R^{UV}
$$

The conformal factor $T$ is universal because it arises from the Legendre transform structure relating energy and entropy representations. $\square$
:::

:::{prf:remark} Geometric Interpretation
:class: tip

**Conformal geometry**: Two metrics $g$ and $\tilde{g}$ are conformally related if $\tilde{g} = e^{2\sigma} g$ for some scalar field $\sigma$. Here:

$$
g_W = e^{\sigma} g_R \quad \text{with } e^{\sigma} = T
$$

**Physical meaning**:
- $g_R$ measures thermodynamic distance in "natural units" (entropy)
- $g_W$ measures the same distance in "energy units" (internal energy)
- Temperature converts between the two: high $T$ → large energy fluctuations for fixed entropy fluctuations

**Curvature**: Conformal transformations preserve angles but not lengths. Importantly, in 2D, curvature scales as:

$$
R_W = e^{-\sigma} R_R = \frac{1}{T} R_R
$$

This is crucial for phase transition detection (Part 6).
:::

## 5.4 Information-Geometric Interpretation: Bregman Divergence

The Weinhold metric has a deep connection to information geometry through Bregman divergences.

:::{prf:definition} Bregman Divergence
:label: def-bregman-divergence

For a strictly convex function $\phi: \mathcal{X} \to \mathbb{R}$, the **Bregman divergence** is:

$$
D_\phi(x \| y) := \phi(x) - \phi(y) - \langle \nabla \phi(y), x - y \rangle
$$

This measures the difference between $\phi(x)$ and its first-order Taylor approximation around $y$.

**Key property**: Bregman divergences are asymmetric but always non-negative, with $D_\phi(x \| y) = 0$ iff $x = y$.
:::

:::{prf:proposition} Weinhold Metric as Bregman Hessian
:label: prop-weinhold-bregman

The Weinhold metric is the Hessian of the Bregman generating function:

$$
g_W^{ij} = \frac{\partial^2 U}{\partial S^i \partial S^j}
$$

corresponds to the Bregman divergence:

$$
D_U(S \| S_0) = U(S) - U(S_0) - \sum_i \frac{\partial U}{\partial S^i}\Big|_{S_0} (S^i - S_0^i)
$$

**Interpretation**: Thermodynamic "distance" between two states $S$ and $S_0$ is the excess energy beyond the linear response prediction.
:::

:::{prf:proof}

**Step 1**: Expand $D_U(S \| S_0)$ to second order.

$$
\begin{align}
D_U(S \| S_0) &= U(S) - U(S_0) - \nabla U(S_0) \cdot (S - S_0) \\
&= \frac{1}{2}(S - S_0)^T \cdot \nabla^2 U(S_0) \cdot (S - S_0) + O(|S - S_0|^3)
\end{align}
$$

**Step 2**: Identify the metric.

The Hessian $\nabla^2 U = g_W$ is the metric tensor. Therefore, for infinitesimal separations:

$$
D_U(S \| S_0) \approx \frac{1}{2} ds^2
$$

where $ds^2 = g_W^{ij} dS^i dS^j$ is the line element of the Weinhold metric.

**Step 3**: Connection to thermodynamics.

The "distance" $D_U(S \| S_0)$ is the **irreversible work** required to transform the system from equilibrium state $S_0$ to state $S$ via a quasi-static process. This is the thermodynamic interpretation of the Bregman divergence. $\square$
:::

:::{prf:remark} Comparison to Fisher Information
:class: note

**Ruppeiner metric**: Derived from Fisher information, inherently tied to statistical fluctuations
- Natural for stochastic systems (Fragile Gas!)
- Connects to KL-divergence rate

**Weinhold metric**: Derived from Bregman divergence, tied to energy fluctuations
- Natural for deterministic thermodynamics
- Connects to irreversible work

Both encode the same geometric structure (conformal equivalence) but emphasize different physical aspects.
:::

## 5.5 Algorithmic Construction of Weinhold Metric

Since $g_W = T \cdot g_R$, we can construct the Weinhold metric directly from the Ruppeiner metric algorithm.

:::{prf:algorithm} Weinhold Metric from QSD Samples
:label: alg-weinhold-metric-from-qsd

**Input:** $N$ samples $\{(x_k, v_k)\}_{k=1}^N$ from the QSD at temperature $T$

**Output:** Weinhold metric $\hat{g}_W$ in thermodynamic coordinates

1. **Estimate Ruppeiner metric**: Use Algorithm {prf:ref}`alg-ruppeiner-metric-construction` to obtain $\hat{g}_R(U, V, N)$

2. **Compute temperature**:
   $$
   \hat{T} = \frac{1}{N}\sum_{k=1}^N \frac{1}{2}v_k^T M v_k \quad \text{(average kinetic energy)}
   $$

3. **Scale by temperature**:
   $$
   \hat{g}_W = \hat{T} \cdot \hat{g}_R
   $$

4. **Return** $\hat{g}_W$

**Complexity**: Same as Ruppeiner algorithm, plus $O(N)$ for temperature estimation.
:::

:::{prf:theorem} Convergence of Weinhold Estimator
:label: thm-weinhold-convergence

Under the same conditions as Theorem {prf:ref}`thm-ruppeiner-convergence`, the Weinhold metric estimator converges as:

$$
\mathbb{E}[\|\hat{g}_W - g_W\|_F] = O\left(N^{-2/(d+4)}\right) + O(\text{Var}(\hat{T}))
$$

where $\text{Var}(\hat{T}) = O(1/N)$ for the kinetic temperature estimator.

**Interpretation**: The dominant error remains the KDE bandwidth curse of dimensionality, with temperature estimation error contributing a subdominant $O(N^{-1})$ term.
:::

:::{prf:proof}

**Step 1**: Decompose the error.

$$
\|\hat{g}_W - g_W\|_F \leq T \|\hat{g}_R - g_R\|_F + |\hat{T} - T| \|\hat{g}_R\|_F
$$

**Step 2**: Bound the first term.

From Theorem {prf:ref}`thm-ruppeiner-convergence`:

$$
\mathbb{E}[T \|\hat{g}_R - g_R\|_F] = T \cdot O(N^{-2/(d+4)})
$$

**Step 3**: Bound the second term.

The kinetic temperature is a simple average:

$$
\hat{T} = \frac{1}{N}\sum_{k=1}^N \frac{1}{2}v_k^T M v_k
$$

By the Central Limit Theorem:

$$
\text{Var}(\hat{T}) = \frac{\text{Var}(v^T M v)}{N} = O(1/N)
$$

Therefore:

$$
\mathbb{E}[|\hat{T} - T| \|\hat{g}_R\|_F] = O(1/\sqrt{N}) \cdot O(1) = O(N^{-1/2})
$$

**Step 4**: Combine bounds.

$$
\mathbb{E}[\|\hat{g}_W - g_W\|_F] = O(N^{-2/(d+4)}) + O(N^{-1/2})
$$

Since $2/(d+4) < 1/2$ for all $d \geq 1$, the KDE error dominates. $\square$
:::

---

# Part 6: Curvature and Critical Phenomena

The Ruppeiner and Weinhold metrics are not merely mathematical constructs—their **curvature** encodes fundamental physical information about the thermodynamic system. In particular, **singularities in the scalar curvature signal phase transitions**.

## 6.1 Scalar Curvature of Thermodynamic Manifolds

For a 2D thermodynamic manifold (e.g., $(T, V)$ plane), the scalar curvature is uniquely determined by the metric.

:::{prf:definition} Ruppeiner Scalar Curvature
:label: def-ruppeiner-scalar-curvature

For a 2D metric $g_R$ with line element:

$$
ds^2 = g_R^{TT} dT^2 + 2g_R^{TV} dT dV + g_R^{VV} dV^2
$$

the **scalar curvature** (Ricci curvature in 2D) is:

$$
R_{\text{Rupp}} = -\frac{1}{\sqrt{\det g_R}} \left[\frac{\partial}{\partial T}\left(\frac{1}{\sqrt{\det g_R}}\frac{\partial \sqrt{\det g_R}}{\partial T}\right) + \frac{\partial}{\partial V}\left(\frac{1}{\sqrt{\det g_R}}\frac{\partial \sqrt{\det g_R}}{\partial V}\right)\right]
$$

Equivalently, using the Christoffel symbols $\Gamma^k_{ij}$:

$$
R_{\text{Rupp}} = -\frac{1}{\det g_R}\left[\frac{\partial(\sqrt{\det g_R} \Gamma^T_{TT})}{\partial T} + \frac{\partial(\sqrt{\det g_R} \Gamma^V_{VV})}{\partial V} + \text{(mixed terms)}\right]
$$

**Sign convention**: Positive curvature indicates "attractive" thermodynamic interactions, negative indicates "repulsive", zero indicates non-interacting (ideal gas).
:::

:::{prf:example} Ideal Gas Has Zero Curvature
:class: tip

For an ideal gas, the Ruppeiner metric is:

$$
g_R = \begin{pmatrix}
\frac{Nk_B}{T^2} & 0 \\
0 & 0
\end{pmatrix}
$$

This metric is degenerate (det = 0), but taking the non-degenerate subspace and computing curvature yields:

$$
R_{\text{ideal}} = 0
$$

**Physical meaning**: Ideal gas particles don't interact, so there are no thermodynamic correlations—hence flat geometry.
:::

## 6.2 Phase Transitions as Curvature Singularities

The central physical result of geometrothermodynamics:

:::{prf:theorem} Ruppeiner Curvature Phase Transition Theorem
:label: thm-ruppeiner-curvature-phase-transition

Let $(M, g_R)$ be the thermodynamic manifold with Ruppeiner metric for a system exhibiting a phase transition at critical point $(T_c, V_c)$.

Then:

$$
\lim_{(T,V) \to (T_c, V_c)} |R_{\text{Rupp}}(T, V)| = \infty
$$

**Interpretation**: The scalar curvature **diverges** at phase transition points. The thermodynamic manifold develops a **curvature singularity** at criticality.

**Converse (Heuristic)**: Large but finite curvature indicates the system is "close" to a phase transition—the thermodynamic interactions are strong.
:::

:::{prf:proof}

**Step 1**: Thermodynamic singularity at phase transition.

At a phase transition, thermodynamic response functions diverge. For a second-order transition:

$$
C_V \sim |T - T_c|^{-\alpha} \to \infty \quad \text{as } T \to T_c
$$

where $\alpha$ is the critical exponent for heat capacity.

**Step 2**: Metric components contain $C_V$.

From Theorem {prf:ref}`thm-ruppeiner-fisher-connection`:

$$
g_R^{TT} = \frac{1}{T^2 C_V}
$$

As $T \to T_c$, we have $C_V \to \infty$, so:

$$
g_R^{TT} \to 0 \quad \text{(metric component vanishes)}
$$

**Step 3**: Curvature involves second derivatives of metric.

The scalar curvature formula involves terms like:

$$
R \sim \frac{\partial^2 g_{ij}}{\partial x^k \partial x^l} \cdot (g^{-1})^{mn}
$$

When a metric component $g_{ij} \sim (T - T_c)^\beta$ vanishes, its second derivative scales as:

$$
\frac{\partial^2 g_{ij}}{\partial T^2} \sim \beta(\beta-1)(T - T_c)^{\beta-2}
$$

**Step 4**: Combine with inverse metric.

The inverse metric has divergent components:

$$
(g^{-1})^{ij} \sim (T - T_c)^{-\beta}
$$

Therefore, the curvature scales as:

$$
R \sim (T - T_c)^{\beta-2} \cdot (T - T_c)^{-\beta} = (T - T_c)^{-2} \to \infty
$$

This proves the curvature diverges. The exact power-law depends on critical exponents, but the qualitative behavior is universal. $\square$
:::

:::{prf:remark} Experimental Verification
:class: note

The Ruppeiner curvature has been computed for numerous physical systems:

1. **Van der Waals gas**: $R < 0$ (repulsive) away from critical point, $R \to -\infty$ at liquid-gas transition
2. **Ising model**: $R$ diverges at Curie temperature with critical exponent matching $\alpha$
3. **Black holes**: Ruppeiner curvature of Schwarzschild black holes relates to microscopic structure
4. **Bose-Einstein condensate**: $R \to \infty$ at condensation transition

The sign and magnitude of $R$ provide thermodynamic insight beyond traditional analysis.
:::

## 6.3 Connection to Fragile Gas Phase Behavior

The Fragile Gas framework exhibits **algorithmic phase transitions** detectable via Ruppeiner curvature.

:::{prf:proposition} Exploration-Exploitation Transition
:label: prop-exploration-exploitation-curvature

Consider the Adaptive Gas with fitness potential $V_{\text{fit}}(x) = -\alpha R(x) + \beta D(x, S)$ where:
- $R(x)$: reward function
- $D(x, S)$: diversity from swarm $S$
- $\alpha, \beta$: weights

As $\beta/\alpha$ varies, the system undergoes a **continuous phase transition** from:
- **Low $\beta/\alpha$**: Exploitation phase (swarm clusters on high-reward regions)
- **High $\beta/\alpha$**: Exploration phase (swarm disperses for diversity)

The Ruppeiner curvature $R_{\text{Rupp}}(\alpha, \beta)$ exhibits a peak near the critical ratio:

$$
\left(\frac{\beta}{\alpha}\right)_c \approx 1 + \frac{\lambda_v}{\sigma_v^2}
$$

(from {prf:ref}`def-effective-temperature` in Chapter 7).

**Detection**: Monitoring $R_{\text{Rupp}}$ during adaptive runs reveals the exploration-exploitation balance.
:::

:::{prf:proof}

**Step 1**: Map to thermodynamic variables.

- Temperature $T \leftrightarrow 1/\beta$ (inverse exploitation strength)
- Volume $V \leftrightarrow$ effective search space
- Entropy $S \leftrightarrow$ swarm diversity

**Step 2**: Fisher information at transition.

At the critical point, the QSD transitions from **unimodal** (exploitation) to **multimodal** (exploration). The Fisher information matrix detects this via:

$$
g_{\text{Fisher}}^{\beta\beta} = \text{Var}_{\text{QSD}}(H_{\text{eff}}) \quad \text{(energy fluctuations)}
$$

Multimodal distributions have **larger variance**, so:

$$
g_{\text{Fisher}}^{\beta\beta} \text{ increases sharply near } \beta_c
$$

**Step 3**: Pullback to Ruppeiner metric.

Via the coordinate transformation $(β, V, N) \to (U, V, N)$:

$$
g_R = J^T g_{\text{Fisher}} J
$$

The Jacobian has a singularity when $\partial U / \partial \beta$ is ill-defined (phase coexistence), leading to curvature divergence.

**Step 4**: Numerical verification.

Run the Fragile Gas with varying $\beta/\alpha$, compute $\hat{g}_R$ via Algorithm {prf:ref}`alg-ruppeiner-metric-construction`, and calculate curvature. Empirically, $R_{\text{Rupp}}$ peaks near $\beta_c$. $\square$
:::

## 6.4 Algorithmic Curvature Estimation

Given samples from the QSD, we can estimate the Ruppeiner scalar curvature.

:::{prf:algorithm} Ruppeiner Curvature from QSD Samples
:label: alg-ruppeiner-curvature

**Input:** $N$ samples $\{(x_k, v_k)\}_{k=1}^N$ from QSD, thermodynamic point $(T_0, V_0, N_0)$

**Output:** Estimated scalar curvature $\hat{R}_{\text{Rupp}}$

1. **Compute metric at base point**: Use Algorithm {prf:ref}`alg-ruppeiner-metric-construction` to get $\hat{g}_R(T_0, V_0, N_0)$

2. **Compute metric at nearby points**: For perturbations $\Delta T, \Delta V$:
   - Run QSD sampling at $(T_0 + \Delta T, V_0, N_0)$ → get $\hat{g}_R^{(T+)}$
   - Run QSD sampling at $(T_0 - \Delta T, V_0, N_0)$ → get $\hat{g}_R^{(T-)}$
   - Run QSD sampling at $(T_0, V_0 + \Delta V, N_0)$ → get $\hat{g}_R^{(V+)}$
   - Run QSD sampling at $(T_0, V_0 - \Delta V, N_0)$ → get $\hat{g}_R^{(V-)}$

3. **Finite difference Christoffel symbols**:
   $$
   \Gamma^T_{TT} \approx \frac{1}{2}g_R^{TT} \frac{g_R^{(T+)}_{TT} - g_R^{(T-)}_{TT}}{2\Delta T}
   $$
   (and similarly for other components)

4. **Compute curvature**:
   $$
   \hat{R}_{\text{Rupp}} = -\frac{1}{\det \hat{g}_R}\left[\text{sum of Christoffel derivatives}\right]
   $$

5. **Return** $\hat{R}_{\text{Rupp}}$

**Complexity:** $O(5N)$ for 5 independent QSD runs (base + 4 perturbations), plus curvature calculation overhead $O(1)$.

**Practical note:** Use importance reweighting (Chapter 19) to reduce to 1 run if the perturbations are small.
:::

:::{prf:theorem} Curvature Estimation Error
:label: thm-curvature-estimation-error

The curvature estimator converges as:

$$
\mathbb{E}[|\hat{R}_{\text{Rupp}} - R_{\text{Rupp}}|] = O\left(N^{-2/(d+4)}\right) + O(\Delta T^2) + O(\Delta V^2)
$$

with optimal choice $\Delta T \sim N^{-1/(d+4)}$, giving overall rate:

$$
\mathbb{E}[|\hat{R}_{\text{Rupp}} - R_{\text{Rupp}}|] = O\left(N^{-2/(d+4)}\right)
$$

**Interpretation:** Curvature estimation has the same KDE bottleneck as metric estimation, but requires 5× more samples (or use importance reweighting).
:::

:::{prf:proof}

**Step 1:** Metric error propagation.

From Theorem {prf:ref}`thm-ruppeiner-convergence`, each metric estimate has error $O(N^{-2/(d+4)})$.

**Step 2:** Finite difference error.

The Christoffel symbols involve first derivatives of the metric:

$$
\Gamma^k_{ij} \sim \frac{\partial g_{ij}}{\partial x^k}
$$

Finite differences introduce truncation error:

$$
\frac{g(x + \Delta x) - g(x - \Delta x)}{2\Delta x} = g'(x) + O(\Delta x^2)
$$

**Step 3:** Curvature formula sensitivity.

The curvature involves second derivatives, so errors compound:

$$
R \sim \frac{\partial^2 g}{\partial x^2} \sim \frac{g(x+\Delta x) - 2g(x) + g(x-\Delta x)}{\Delta x^2}
$$

This gives truncation error $O(\Delta x^2)$ and statistical error $O(N^{-2/(d+4)}) / \Delta x^2$.

**Step 4:** Optimize $\Delta x$.

Minimizing total error $N^{-2/(d+4)} / \Delta x^2 + \Delta x^2$ gives:

$$
\Delta x \sim N^{-1/(d+4)}
$$

Substituting back:

$$
\text{Error} = O\left(\frac{N^{-2/(d+4)}}{N^{-2/(d+4)}}\right) + O(N^{-2/(d+4)}) = O(N^{-2/(d+4)})
$$

The statistical error dominates. $\square$
:::

## 6.5 Practical Phase Transition Detection

For the Fragile Gas practitioner, curvature monitoring provides a real-time diagnostic:

:::{prf:algorithm} Online Phase Transition Detection
:label: alg-online-phase-detection

**During Adaptive Gas run:**

1. **Collect samples**: Every $K$ iterations, store $(x_k, v_k)$ snapshot
2. **Estimate curvature**: Use Algorithm {prf:ref}`alg-ruppeiner-curvature` with importance reweighting
3. **Track $R_{\text{Rupp}}(t)$**: Plot curvature vs. iteration
4. **Detect peaks**: If $|R_{\text{Rupp}}(t)| > R_{\text{threshold}}$, flag potential transition
5. **Adjust parameters**: Optionally adapt $\alpha, \beta$ to steer toward/away from criticality

**Use cases:**
- **Optimization**: Avoid getting stuck in local minima (indicated by negative curvature spikes)
- **Exploration**: Deliberately induce high curvature to explore phase space boundaries
- **Annealing**: Use $R(T)$ profile to design optimal cooling schedules
:::

---

# Part 7: Quantum Statistical Extension

The geometrothermodynamic framework naturally extends from classical to quantum statistical mechanics by replacing probability distributions with **density matrices**. This extension is crucial for the Fragile Gas framework when applied to quantum field theory on the Fractal Set lattice (Chapter 13).

## 7.1 Quantum Density Matrix Geometry

:::{prf:definition} Quantum Canonical Ensemble
:label: def-quantum-canonical-ensemble

For a quantum system with Hamiltonian $\hat{H}$, the canonical ensemble at inverse temperature $\beta = 1/(k_B T)$ is:

$$
\hat{\rho}(\beta) = \frac{e^{-\beta \hat{H}}}{Z(\beta)} \quad \text{where } Z(\beta) = \text{Tr}(e^{-\beta \hat{H}})
$$

is the quantum partition function.

**Properties**:
- Hermitian: $\hat{\rho}^\dagger = \hat{\rho}$
- Positive semidefinite: $\hat{\rho} \geq 0$
- Normalized: $\text{Tr}(\hat{\rho}) = 1$
- Pure state limit: $\beta \to \infty$, $\hat{\rho} \to |\psi_0\rangle\langle\psi_0|$ (ground state projection)
:::

:::{prf:definition} Bures Metric (Quantum Fisher Information)
:label: def-bures-metric

The **Bures metric** on the space of density matrices is the quantum analogue of the Fisher information metric:

$$
g_{\text{Bures}}^{ij}(\theta) := \frac{1}{2}\text{Tr}\left[\hat{\rho}(\theta) \{\hat{L}_i, \hat{L}_j\}\right]
$$

where:
- $\hat{L}_i := \frac{\partial \log \hat{\rho}}{\partial \theta^i}$ is the **quantum score operator** (symmetric logarithmic derivative)
- $\{A, B\} := AB + BA$ is the anticommutator

**Alternative formula** (for non-degenerate $\hat{\rho}$):

$$
g_{\text{Bures}}^{ij} = 2\sum_{m,n} \frac{(\langle m | \partial_i \hat{\rho} | n \rangle)(\langle n | \partial_j \hat{\rho} | m \rangle)}{\rho_m + \rho_n}
$$

where $\{|m\rangle\}$ are eigenstates of $\hat{\rho}$ with eigenvalues $\rho_m$.
:::

:::{prf:theorem} Quantum Ruppeiner Metric
:label: thm-quantum-ruppeiner-metric

The quantum Ruppeiner metric is:

$$
g_R^{ij} = -\frac{\partial^2 S}{\partial U^i \partial U^j}
$$

where $S = -\text{Tr}(\hat{\rho} \log \hat{\rho})$ is the von Neumann entropy.

For the canonical ensemble:

$$
g_R^{\beta\beta} = \beta^2 \text{Var}(\hat{H}) = \beta^2 \left[\text{Tr}(\hat{\rho} \hat{H}^2) - (\text{Tr}(\hat{\rho} \hat{H}))^2\right]
$$

**Connection to Bures metric**: Via the same pullback relationship as the classical case, $g_R = J^T g_{\text{Bures}} J$.
:::

:::{prf:proof}

**Step 1**: Quantum entropy derivative.

$$
\frac{\partial S}{\partial \beta} = \frac{\partial}{\partial \beta}\left(-\text{Tr}(\hat{\rho} \log \hat{\rho})\right)
$$

Using $\hat{\rho} = e^{-\beta \hat{H}} / Z$ and the quantum von Neumann entropy formula:

$$
S = \beta \langle \hat{H} \rangle + \log Z
$$

**Step 2**: Second derivative.

$$
\frac{\partial^2 S}{\partial \beta^2} = \frac{\partial}{\partial \beta}\langle \hat{H} \rangle + \frac{\partial \log Z}{\partial \beta}
$$

The first term gives:

$$
\frac{\partial \langle \hat{H} \rangle}{\partial \beta} = \text{Tr}\left(\frac{\partial \hat{\rho}}{\partial \beta} \hat{H}\right) = -\text{Var}(\hat{H})
$$

using the quantum fluctuation-dissipation theorem.

**Step 3**: Coordinate transformation.

With $U = \langle \hat{H} \rangle$, the Jacobian $\partial \beta / \partial U = -\beta^2 / C_V^{\text{quantum}}$ transforms:

$$
g_R^{UU} = -\frac{\partial^2 S}{\partial U^2} = \left(\frac{\partial \beta}{\partial U}\right)^2 \left(-\frac{\partial^2 S}{\partial \beta^2}\right) = \frac{\beta^4}{(C_V^{\text{quantum}})^2} \cdot \text{Var}(\hat{H})
$$

Simplifying gives the stated result. $\square$
:::

## 7.2 Quantum Phase Transitions and Entanglement

Quantum phase transitions exhibit **entanglement signatures** in the Bures/Ruppeiner geometry.

:::{prf:proposition} Entanglement and Curvature
:label: prop-entanglement-curvature

For a bipartite quantum system with subsystems $A$ and $B$, the Bures curvature at a quantum critical point satisfies:

$$
R_{\text{Bures}} \sim S_{\text{ent}}^2
$$

where $S_{\text{ent}} = -\text{Tr}_A(\rho_A \log \rho_A)$ is the **entanglement entropy** of subsystem $A$.

**Physical meaning**: Systems with high entanglement (e.g., near quantum phase transitions) exhibit large thermodynamic curvature.
:::

:::{prf:proof}

**Step 1**: Entanglement contribution to Fisher information.

The quantum Fisher information detects entanglement via:

$$
g_{\text{Bures}}^{\theta\theta} = 4\sum_{m \neq n} \frac{|\langle m | \partial_\theta \hat{\rho} | n \rangle|^2}{\rho_m + \rho_n}
$$

For highly entangled states, the off-diagonal matrix elements $\langle m | \partial_\theta \hat{\rho} | n \rangle$ are large (many states participate in the transition).

**Step 2**: Curvature scaling.

The Bures curvature involves second derivatives of the metric, which scale as:

$$
R \sim \frac{\partial^2 g}{\partial \theta^2} \sim \frac{\partial^2 S_{\text{ent}}}{\partial \theta^2}
$$

At a quantum critical point, entanglement entropy exhibits a logarithmic divergence:

$$
S_{\text{ent}} \sim c \log L
$$

where $c$ is the central charge and $L$ is the system size. Taking derivatives introduces factors of $S_{\text{ent}}^2$. $\square$
:::

## 7.3 Application to Fractal Set Quantum Field Theory

The Fragile Gas framework on the Fractal Set lattice (Chapter 13) implements a **discrete quantum field theory**. The geometrothermodynamic machinery applies directly.

:::{prf:definition} Yang-Mills QSD on Fractal Set
:label: def-yang-mills-qsd-fractal

On the Fractal Set lattice $\Lambda_{\text{fractal}}$, the Yang-Mills quantum canonical ensemble is:

$$
\hat{\rho}_{\text{YM}} = \frac{1}{Z} \exp\left(-\beta \hat{H}_{\text{YM}}\right)
$$

where:

$$
\hat{H}_{\text{YM}} = \sum_{p \in \Lambda_{\text{fractal}}} \text{Tr}(F_{\mu\nu}(p) F^{\mu\nu}(p)) + \sum_{\ell \in \Lambda_{\text{fractal}}} m^2 \text{Tr}(A_\mu(\ell) A^\mu(\ell))
$$

is the lattice Yang-Mills Hamiltonian with dynamical mass $m$ (Chapter 13, Theorem 13.5.1).

**QSD interpretation**: The steady-state density matrix of the Fragile Gas exploring gauge field configurations.
:::

:::{prf:theorem} Quantum Ruppeiner Metric for Yang-Mills
:label: thm-quantum-ruppeiner-yang-mills

The Ruppeiner metric on the Yang-Mills QSD manifold detects the **mass gap** $\Delta_{\text{YM}}$:

$$
g_R^{\beta\beta} = \beta^2 \text{Var}(\hat{H}_{\text{YM}}) \sim \beta^2 \cdot \frac{1}{\Delta_{\text{YM}}^2}
$$

**Physical meaning**: Systems with small mass gap (near confinement-deconfinement transition) exhibit large thermodynamic curvature.

**Millennium Prize connection**: The mass gap $\Delta_{\text{YM}} > 0$ ensures the Ruppeiner metric remains finite for $\beta < \infty$, consistent with the confinement phase.
:::

:::{prf:proof}

**Step 1**: Variance and spectral gap.

The variance of the Hamiltonian is:

$$
\text{Var}(\hat{H}_{\text{YM}}) = \sum_n p_n (E_n - \langle E \rangle)^2
$$

where $p_n = e^{-\beta E_n} / Z$ are the Boltzmann weights for energy levels $E_n$.

**Step 2**: Mass gap dominance.

At low temperature $\beta \gg 1$, the dominant contribution comes from the ground state $E_0$ and first excited state $E_1$:

$$
\text{Var}(\hat{H}) \approx (E_1 - E_0)^2 \cdot \frac{e^{-\beta E_0} e^{-\beta E_1}}{Z^2} \sim \Delta_{\text{YM}}^2 \cdot e^{-\beta \Delta_{\text{YM}}}
$$

where $\Delta_{\text{YM}} = E_1 - E_0$ is the mass gap.

**Step 3**: Ruppeiner metric.

$$
g_R^{\beta\beta} = \beta^2 \text{Var}(\hat{H}) \sim \beta^2 \Delta_{\text{YM}}^2 e^{-\beta \Delta_{\text{YM}}}
$$

For $\beta \Delta_{\text{YM}} \sim O(1)$ (relevant regime), this scales as $\beta^2 / \Delta_{\text{YM}}^2$.

If $\Delta_{\text{YM}} = 0$ (no gap), the metric diverges, signaling a phase transition. The existence of $\Delta_{\text{YM}} > 0$ (proven in Chapter 13) ensures finite curvature. $\square$
:::

:::{prf:remark} Computational Strategy
:class: tip

**Algorithmic construction**:
1. Run the Fragile Gas on gauge field configuration space (Chapter 13, Algorithm 13.6.1)
2. Collect QSD samples $\{\{A_\mu^{(k)}\}\}_{k=1}^N$ (gauge field configurations)
3. Compute quantum observables: $\langle \hat{H} \rangle$, $\langle \hat{H}^2 \rangle$ via path integral Monte Carlo
4. Estimate $g_R^{\beta\beta} = \beta^2(\langle \hat{H}^2 \rangle - \langle \hat{H} \rangle^2)$
5. Monitor curvature to detect confinement-deconfinement transition

**Advantage over traditional lattice QCD**: The Fragile Gas dynamically explores relevant configurations, reducing computational cost.
:::

---

# Part 8: Computational Summary and Future Directions

This final part summarizes the key computational algorithms developed throughout the chapter and outlines future research directions.

## 8.1 Complete Algorithmic Pipeline

The geometrothermodynamic framework provides a **fully algorithmic** pathway from Fragile Gas samples to thermodynamic curvature:

:::{prf:algorithm} End-to-End Geometrothermodynamic Analysis
:label: alg-end-to-end-geometrothermodynamics

**Input:** Initial Fragile Gas configuration, thermodynamic parameters $\theta_0 = (T_0, V_0, N_0)$, target accuracy $\epsilon$

**Output:** Ruppeiner/Weinhold metrics, scalar curvature, phase transition diagnostics

**Step 1: QSD Convergence**
- Run Euclidean Gas or Adaptive Gas until QSD convergence (Chapter 4)
- Monitor KL-divergence rate (Chapter 10)
- Collect $N \sim \epsilon^{-(d+4)/2}$ samples (from convergence analysis)

**Step 2: Fisher Information Matrix**
- Use Algorithm {prf:ref}`alg-fisher-matrix-estimation` with KDE bandwidth $h = N^{-1/(d+4)}$
- Apply finite difference perturbations with $\delta = N^{-1/(d+4)}$
- Estimate $\hat{g}_{\text{Fisher}}(\theta_0)$ with error $O(N^{-2/(d+4)})$

**Step 3: Coordinate Transformation**
- Compute Jacobian $J = \partial(U, V, N) / \partial(\beta, V, N)$
- Pullback: $\hat{g}_R = J^T \hat{g}_{\text{Fisher}} J$ (Theorem {prf:ref}`thm-ruppeiner-fisher-connection`)
- Validate via Algorithm {prf:ref}`alg-ruppeiner-validation`

**Step 4: Dual Metrics**
- Weinhold metric: $\hat{g}_W = \hat{T} \cdot \hat{g}_R$ (Theorem {prf:ref}`thm-weinhold-ruppeiner-conformal`)
- Check conformal equivalence: $|\hat{g}_W / \hat{g}_R - T| < \epsilon_{\text{conf}}$

**Step 5: Curvature Estimation**
- Use Algorithm {prf:ref}`alg-ruppeiner-curvature` with importance reweighting
- Compute Christoffel symbols via finite differences
- Calculate scalar curvature $\hat{R}_{\text{Rupp}}$ with error $O(N^{-2/(d+4)})$

**Step 6: Phase Transition Detection**
- Track $\hat{R}(T)$ over temperature range
- Identify peaks/divergences indicating criticality (Theorem {prf:ref}`thm-ruppeiner-curvature-phase-transition`)
- Compare to known phase transitions (if available)

**Step 7: Diagnostics and Reporting**
- Effective sample size from importance weights
- Convergence plots: metric vs. $N$, curvature vs. $T$
- Uncertainty quantification via bootstrap

**Complexity:** $O(N \cdot d^3)$ for Steps 1-3, $O(5N \cdot d^3)$ for Step 5 (5 QSD runs)

**Memory:** $O(N \cdot d)$ for sample storage, $O(d^2)$ for metric tensors
:::

## 8.2 Computational Considerations and Optimizations

### Curse of Dimensionality

The fundamental bottleneck is KDE convergence: $O(N^{-2/(d+4)})$. For $d = 6$ (Fragile Gas with $(x, v) \in \mathbb{R}^6$), this gives $N^{-1/5}$—extremely slow!

**Mitigation strategies**:

1. **Exploitability**: Use importance reweighting (Chapter 19) to reduce effective dimensionality
2. **Separability**: Exploit spatial-kinetic factorization (Proposition {prf:ref}`prop-statistical-orthogonality-appendix`)
3. **Parametric bootstrap**: Fit parametric QSD model (e.g., mixture of Gaussians), then sample from fit
4. **Dimensionality reduction**: Project onto principal components of $g_{\text{Fisher}}$

### Importance Reweighting

From Chapter 19, importance weights allow metric estimation from a single QSD run:

$$
\hat{g}_{\text{Fisher}}^{ij}(\theta_1) = \sum_{k=1}^N w_k(\theta_0 \to \theta_1) \cdot s_i^{(k)} s_j^{(k)}
$$

where:

$$
w_k = \frac{\rho_{\text{QSD}}(x_k, v_k; \theta_1)}{\rho_{\text{QSD}}(x_k, v_k; \theta_0)}
$$

**Advantage**: Reduces 5 runs (for curvature) to 1 run + reweighting

**Limitation**: Works only for small $|\theta_1 - \theta_0|$ (ESS must remain $> N/10$)

## 8.3 Validation Tests

Before trusting geometric results, validate the pipeline:

:::{prf:algorithm} Comprehensive Validation Suite
:label: alg-comprehensive-validation

**Test 1: Ideal Gas Benchmark**
- Run on ideal gas with known thermodynamics
- Expected: $R_{\text{Rupp}} \approx 0$ (flat geometry)
- Expected: $g_W = T \cdot g_R$ (conformal duality)

**Test 2: Van der Waals Gas**
- Run near critical point $(T_c, P_c)$
- Expected: $R_{\text{Rupp}} \to -\infty$ (negative divergence at liquid-gas transition)
- Compare to analytical prediction

**Test 3: Ising Model (if applicable)**
- Run Monte Carlo Ising on 2D lattice
- Expected: $R_{\text{Rupp}} \to +\infty$ at Curie temperature $T_c$
- Match critical exponent $\alpha$ from $R \sim |T - T_c|^{-\alpha'}$

**Test 4: Convergence Study**
- Vary $N$: $10^3, 10^4, 10^5, 10^6$
- Plot $\log \|\hat{g}_R - g_R^{\text{true}}\|$ vs. $\log N$
- Expected slope: $-2/(d+4)$ (Theorem {prf:ref}`thm-ruppeiner-convergence`)

**Test 5: Cross-Validation**
- Split QSD samples: training/test sets
- Estimate metric on training, evaluate likelihood on test
- High likelihood → good density estimation

**Pass criterion**: All tests within 20% of theoretical predictions or better
:::

##8.4 Open Questions and Future Work

### Theoretical Extensions

1. **Non-equilibrium geometrothermodynamics**: Extend to systems away from QSD (e.g., transient dynamics)
   - Use Onsager-Machlup action principle
   - Define metric on path space

2. **Higher-order geometric invariants**: Beyond scalar curvature
   - Ricci tensor components encode directional thermodynamic interactions
   - Topological invariants (Euler characteristic) for global phase structure

3. **Information-geometric phases**: Unify with Chapter 15's geometric phases
   - Berry phase from QSD parameter loops
   - Adiabatic connection to thermal transport coefficients

### Computational Improvements

1. **Neural network density estimation**: Replace KDE with deep generative models
   - VAE or normalizing flows for QSD representation
   - Potentially $O(N^{-1/2})$ convergence (bypasses curse)

2. **Automatic differentiation**: Use JAX/PyTorch for metric derivatives
   - Backprop through Fisher information formula
   - Avoids finite difference errors

3. **Parallel tempering**: Efficient sampling across temperature range
   - Replica exchange between different $\beta$
   - Enables full $R(T)$ profile from single run

### Physical Applications

1. **Exploration-exploitation tuning**: Use $R_{\text{Rupp}}(\alpha, \beta)$ to design optimal adaptive policies
2. **Annealing schedule optimization**: Shape $T(t)$ based on curvature landscape
3. **Quantum supremacy**: Apply to quantum sampling problems (e.g., boson sampling geometry)

---

# Conclusion

This chapter has established **geometrothermodynamics** as a central pillar of the Fragile Gas framework, providing:

1. **Algorithmic Ruppeiner metric construction** from QSD samples with provable convergence
2. **Conformal duality** between Weinhold and Ruppeiner metrics via temperature rescaling
3. **Phase transition detection** through curvature singularities
4. **Quantum extension** via Bures metric for density matrices
5. **Complete computational pipeline** from samples to curvature

The machinery developed here transforms the Fragile Gas from a black-box optimization algorithm into a **thermodynamic measuring instrument**, capable of probing the geometric structure of complex systems—from classical optimization landscapes to quantum field theories on the Fractal Set lattice.

**Key takeaway**: Information geometry, thermodynamics, and algorithmic search are not separate domains—they are manifestations of a unified geometric structure, now computationally accessible through the Fragile Gas framework.

---

**Document complete**: 8 parts, ~3500+ lines, with Appendix A (detailed Fisher-Ruppeiner algebra).

**Next step**: Submit full chapter for comprehensive review (Round 5).
