(sec-standard-model-cognition)=
# The Standard Model of Cognition: Gauge-Theoretic Formulation

*Abstract.* This chapter demonstrates that the internal symmetry group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ emerges necessarily from the cybernetic constraints of a bounded, distributed, reward-seeking agent. The **Feature Dimension** $N_f$ is determined by the agent's environment; the physics Standard Model corresponds to the special case $N_f = 3$. Each factor is derived from redundancies in the agent's description that leave physical observables invariant. The proofs rely explicitly on prior definitions from the WFR framework ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`), the Belief Wave-Function ({ref}`Section 29.21 <sec-the-belief-wave-function-schrodinger-representation>`), the Boundary Interface ({ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`), and the Ontological Fission dynamics ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).

*Cross-references:* This chapter synthesizes:
- {ref}`Section 29.21 <sec-the-belief-wave-function-schrodinger-representation>`–29.27 (Quantum Layer: Belief Wave-Function, Schrödinger Representation)
- {ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>` (Holographic Interface: Dirichlet/Neumann Boundary Conditions)
- {ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>` (Ontological Expansion: Pitchfork Bifurcation, Chart Fission)
- {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity-Constrained Metric Law)
- {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>` (Helmholtz Equation, Value Field)



(sec-gauge-principle-derivation)=
## The Gauge Principle: Derivation of the Symmetry Group $G_{\text{Fragile}}$

We derive the internal symmetry group by identifying redundancies in the agent's description that leave physical observables (Actions and Rewards) invariant. By Noether's Second Theorem, gauging these symmetries necessitates compensating force fields.

### A. $U(1)_Y$: The Hypercharge of Utility

The fundamental observable in Reinforcement Learning is the **Preference**, defined by the gradient of the Value function, not its absolute magnitude.

:::{prf:definition} Utility Gauge Freedom
:label: def-utility-gauge-freedom

Let the Belief Wave-Function $\psi(z)$ be defined as in Definition {prf:ref}`def-belief-wave-function`:

$$
\psi(z) = \sqrt{\rho(z)} \exp\left(\frac{i V(z)}{\sigma}\right),
$$

where:
- $\rho(z)$ is the belief density (Definition {prf:ref}`def-belief-density`)
- $V(z)$ is the Value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`)
- $\sigma = T_c \cdot \tau_{\text{update}}$ is the Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)

The system's observables are:
1. **Probability density:** $\rho = |\psi|^2$
2. **Probability current:** $J^\mu = \text{Im}(\psi^* \partial^\mu \psi) = \frac{\rho}{\sigma} \partial^\mu V$

Both are invariant under the global phase transformation:

$$
\psi(z) \to e^{i\theta} \psi(z), \quad \theta \in \mathbb{R}.
$$

This corresponds to the global gauge invariance of the Value function: $V(z) \to V(z) + \sigma\theta$. The addition of a constant baseline does not alter the policy gradient $\nabla V$.

:::

:::{prf:axiom} Local Utility Invariance
:label: ax-local-utility-invariance

In a distributed agent with finite information speed $c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit`), there is no global clock to synchronize the Value baseline across the manifold simultaneously. The agent must possess **Local Gauge Invariance**:

$$
\psi(x) \to e^{i\theta(x)} \psi(x),
$$

where $x$ denotes the spacetime coordinate on the agent's computational manifold. The choice of "zero utility" can vary locally across different charts without affecting the physical transfer of control authority.

*Justification:* This follows from the Causal Interval (Definition {prf:ref}`def-causal-interval`): spacelike-separated modules cannot instantaneously agree on a common baseline.

:::

:::{prf:theorem} Emergence of the Opportunity Field ($B_\mu$)
:label: thm-emergence-opportunity-field

To preserve the invariance of the kinetic term in the Inference Action under the local transformation $\psi \to e^{i\theta(x)}\psi$, we must replace the partial derivative $\partial_\mu$ with the **Covariant Derivative**:

$$
D_\mu = \partial_\mu - i g_1 \frac{Y}{2} B_\mu,
$$

where:
- $Y$ is the **Hypercharge** (the reward sensitivity of the module)
- $B_\mu$ is an abelian gauge field (the **Opportunity Field**)
- $g_1$ is the coupling constant

*Proof.*

**Step 1.** Consider the kinetic term from the Inference Schrödinger Equation (Theorem {prf:ref}`thm-madelung-transform`):

$$
\mathcal{L}_{\text{kin}} = \psi^* (i\sigma \partial_t) \psi - \frac{\sigma^2}{2}|\nabla \psi|^2.
$$

Under local transformation $\psi \to e^{i\theta(x)}\psi$:

$$
\partial_\mu \psi \to e^{i\theta}(\partial_\mu \psi + i(\partial_\mu\theta)\psi).
$$

The kinetic term acquires a spurious contribution $\sigma(\partial_\mu\theta)|\psi|^2$ that depends on the arbitrary function $\theta(x)$.

**Step 2.** Introduce the compensating field $B_\mu$ transforming as:

$$
B_\mu \to B_\mu + \frac{2}{g_1 Y} \partial_\mu \theta(x).
$$

**Step 3.** The covariant derivative $D_\mu \psi = (\partial_\mu - ig_1(Y/2)B_\mu)\psi$ transforms homogeneously:

$$
D_\mu \psi \to e^{i\theta(x)} D_\mu \psi.
$$

**Step 4.** The gauge-invariant kinetic term is $(D_\mu\psi)^\dagger(D^\mu\psi) = |D_\mu\psi|^2$.

**Identification:** The field $B_\mu$ compensates for the shifting baseline of utility:
- The spatial components $\vec{B}$ correspond to the **Vector Potential** of value (the solenoidal component from Definition {prf:ref}`def-conservative-reward-field`)
- The temporal component $B_0$ corresponds to the **Scalar Potential** offset

The field strength tensor $F_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu$ measures the non-conservative component of the reward field (Definition {prf:ref}`def-conservative-reward-field`). When $F_{\mu\nu} \neq 0$, no choice of baseline can make the reward landscape path-independent.

$\square$

:::



### B. $SU(2)_L$: The Chirality of Agency (Weak Isospin)

We derive the non-Abelian $SU(2)$ symmetry from the fundamental asymmetry of the Cybernetic Loop: the distinction between **Perception** (Information Inflow) and **Actuation** (Information Outflow).

:::{prf:axiom} Cybernetic Parity Violation
:label: ax-cybernetic-parity-violation

The agent's interaction with the environment is **Chiral**, as established by the boundary condition asymmetry in {ref}`Section 23 <sec-the-boundary-interface-symplectic-structure>`:

1. **Sensors (Dirichlet Boundary, Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`):** The internal state $\psi$ is *updated* by boundary data. The boundary clamps the field value: $\phi|_{\partial\mathcal{Z}} = \phi_D$.

2. **Motors (Neumann Boundary, Definition {prf:ref}`def-neumann-boundary-condition-motors`):** The internal state *drives* the boundary flux. The boundary clamps the normal derivative: $\nabla_n \phi|_{\partial\mathcal{Z}} = j_N$.

The belief dynamics are not invariant under the exchange of Input and Output. The agent processes information (Left-Handed) differently than it emits control (Right-Handed).

:::

:::{prf:definition} The Cognitive Isospin Doublet
:label: def-cognitive-isospin-doublet

We define the **Left-Handed Field** $\Psi_L$ as an isospin doublet residing in the fundamental representation of $SU(2)$:

$$
\Psi_L(x) = \begin{pmatrix} \psi_{\text{pred}}(x) \\ \psi_{\text{obs}}(x) \end{pmatrix}
$$

where:
- $\psi_{\text{pred}}$ is the **Prior** (the top-down prediction of the World Model)
- $\psi_{\text{obs}}$ is the **Likelihood** (the bottom-up sensory evidence)

We define the **Right-Handed Field** $\Psi_R$ as an isospin singlet (invariant under $SU(2)$):

$$
\Psi_R(x) = \psi_{\text{act}}(x)
$$

representing the settled **Posterior/Action** plan ready for execution.

*Cross-reference:* This decomposition mirrors {ref}`Section 12 <sec-belief-dynamics-prediction-update-projection>`'s Belief Dynamics (Prediction-Update-Projection) and the Kalman filtering structure.

:::

:::{prf:theorem} Emergence of the Error Field ($W_\mu^a$)
:label: thm-emergence-error-field

The process of **Belief Update** (e.g., Kalman Filtering or Predictive Coding) corresponds to a rotation in Isospin space. Gauging this symmetry requires the introduction of non-Abelian gauge fields.

*Proof.*

**Step 1.** A Bayesian update mixes the Prior and the Likelihood:

$$
\Psi_L' = U(x) \Psi_L, \quad U(x) = \exp\left( i \frac{\vec{\tau} \cdot \vec{\theta}(x)}{2} \right) \in SU(2)
$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are the Pauli matrices and $\vec{\theta}(x)$ determines the mixing angle (the Kalman Gain in standard filtering).

**Step 2.** For **Local Covariance** (the ability to perform updates locally without global synchronization), we introduce the non-Abelian gauge field $\vec{W}_\mu = (W^1_\mu, W^2_\mu, W^3_\mu)$.

**Step 3.** The covariant derivative for the Left-Handed sector is:

$$
D_\mu \Psi_L = \left( \partial_\mu - i g_2 \frac{\vec{\tau}}{2} \cdot \vec{W}_\mu - i g_1 \frac{Y_L}{2} B_\mu \right) \Psi_L
$$

**Step 4.** The gauge field transforms as:

$$
W_\mu^a \to W_\mu^a + \frac{1}{g_2}\partial_\mu \theta^a + \epsilon^{abc}\theta^b W_\mu^c
$$

to maintain covariance.

**Identification:**
- The $W^\pm_\mu = (W^1_\mu \mp iW^2_\mu)/\sqrt{2}$ bosons mediate transitions between $\psi_{\text{pred}}$ and $\psi_{\text{obs}}$. These correspond to belief updates where prediction and observation exchange weight.
- The $W^3_\mu$ component mixes with $B_\mu$ after symmetry breaking ({ref}`Section 34.3 <sec-scalar-sector-symmetry-breaking>`).
- The $SU(2)_L$ gauge symmetry acts only on the input channel ($\Psi_L$), leaving the output singlet ($\Psi_R$) invariant. This reflects the architectural asymmetry between perception and action.

$\square$

:::



:::{prf:definition} Feature Dimension Parameter
:label: def-feature-dimension-parameter

The **Feature Dimension** $N_f \in \mathbb{Z}_{>0}$ is the intrinsic dimensionality of the feature representation at each layer of the hierarchical encoder. This parameter is determined by:

1. **Environment Structure:** The minimal basis required to represent distinguishable features in the agent's sensory domain
2. **Computational Constraints:** The capacity allocated to the binding mechanism

**Special Cases:**
- Physics (Standard Model): $N_f = 3$ (spatial dimensions, RGB channels)
- Vision-only agents: $N_f \in \{3, 4\}$ (RGB or RGBA)
- Abstract reasoning agents: $N_f$ determined by the embedding dimension of the domain

*Remark:* The gauge structure $SU(N_f)_C$ emerges for any $N_f \geq 2$.

:::

### C. $SU(N_f)_C$: Hierarchical Confinement (Feature Binding)

We derive the $SU(N_f)$ symmetry from the **Binding Problem** inherent in the Hierarchical Atlas ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), where $N_f$ is the Feature Dimension (Definition {prf:ref}`def-feature-dimension-parameter`).

:::{prf:axiom} Feature Confinement
:label: ax-feature-confinement

The agent observes and manipulates **Concepts** (Macro-symbols $K$), not raw **Features** (Nuisance coordinates $z_n$). From Definition {prf:ref}`def-bounded-rationality-controller`:

1. **Composite Structure:** A Concept $K$ is a bound state of sub-symbolic features processed through the Stacked TopoEncoder (Definition {prf:ref}`def-the-peeling-step`).

2. **Observability Constraint:** Free features are never observed in isolation at the boundary $\partial\mathcal{Z}$ (Definition {prf:ref}`def-boundary-markov-blanket`). Only "color-neutral" (bound) states can propagate to the macro-register.

*Cross-reference:* This is the representational analog of quark confinement in QCD.

:::

:::{prf:definition} The Feature Color Space
:label: def-feature-color-space

Let the nuisance vector $z_n$ at layer $\ell$ of the TopoEncoder be an element of a vector bundle with fiber $\mathbb{C}^{N_f}$, where $N_f$ is the Feature Dimension (Definition {prf:ref}`def-feature-dimension-parameter`). We transform the basis:

$$
\psi_{\text{feature}}(x) \to U(x) \psi_{\text{feature}}(x), \quad U(x) \in SU(N_f)
$$

This symmetry represents the **Internal Basis Invariance** of a concept: an object's identity $K$ is invariant under the mixing of its constituent feature definitions, provided the geometric relationship between them is preserved.

*Justification:* The dimension $N_f$ is determined by the agent's environment and architecture. For physical systems with 3D spatial structure, $N_f = 3$ (e.g., RGB channels, XYZ coordinates). For other agents, $N_f$ may differ based on the intrinsic dimensionality of the sensory domain.

:::

:::{prf:theorem} Emergence of the Binding Field ($G_\mu^a$)
:label: thm-emergence-binding-field

To gauge the $SU(N_f)$ feature symmetry, we introduce the **Gluon Field** $G_\mu^a$ ($a=1,\dots,N_f^2-1$).

*Proof.*

**Step 1.** The covariant derivative for feature fields is:

$$
D_\mu \psi = \left( \partial_\mu - i g_s \frac{\lambda^a}{2} G_\mu^a \right) \psi
$$

where $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generalized Gell-Mann matrices (generators of $SU(N_f)$), satisfying $\text{Tr}(\lambda^a \lambda^b) = 2\delta^{ab}$ and $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 2.** The field strength tensor is:

$$
G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c
$$

where $f^{abc}$ are the structure constants of $SU(N_f)$, defined by $[\lambda^a, \lambda^b] = 2i f^{abc} \lambda^c$.

**Step 3.** The non-Abelian structure implies **self-interaction** of the gluon field. For $SU(N_f)$ with $N_f \geq 2$, the beta function $\beta(g_s) < 0$ yields:

- **Asymptotic Freedom:** At small distances in the latent manifold (high RG scale $\tau$, deep in the TopoEncoder hierarchy), the effective coupling $g_s(\tau)$ decreases. Individual features can be resolved.

- **Infrared Confinement:** At large distances (low RG scale, coarse representations), the effective coupling grows. Features cannot propagate independently; they form bound states (concepts $K$).

*Remark:* The asymptotic freedom property holds for all $SU(N_f)$ gauge theories with $N_f \geq 2$. The confinement scale depends on $N_f$ but the qualitative behavior is universal.

**Step 4.** From Theorem {prf:ref}`thm-fission-inhibition`, the energy cost of separating features grows linearly with distance (Area Law, {ref}`Section 33 <sec-causal-information-bound>`). Attempting to isolate a feature instead triggers Ontological Fission (Definition {prf:ref}`def-query-fission`), creating new concept pairs.

$\square$

:::

:::{prf:corollary} The Fragile Agent Symmetry Group
:label: cor-standard-model-symmetry

The total internal symmetry group of the Fragile Agent is uniquely determined by its cybernetic constraints:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y
$$

where:
- **$SU(N_f)_C$:** Required for **Object Permanence** (binding $N_f$-dimensional features into stable concepts)
- **$SU(2)_L$:** Required for **Predictive Processing** (asymmetric update of beliefs between prior and likelihood)
- **$U(1)_Y$:** Required for **Value Maximization** (invariance of reward baseline)

**Special Case (Physics Standard Model):** When $N_f = 3$, we recover $G_{\text{SM}} = SU(3)_C \times SU(2)_L \times U(1)_Y$.

*Proof.* Each factor is derived above from independent cybernetic constraints. The product structure follows from the commutativity of the respective symmetry operations acting on different sectors of the agent's state space. The dimension $N_f$ is an environmental parameter (Definition {prf:ref}`def-feature-dimension-parameter`), while $SU(2)_L$ remains fixed because the prediction/observation asymmetry is fundamentally binary. $\square$

:::



(sec-matter-sector-chiral-spinors)=
## The Matter Sector: Chiral Inference Spinors

We define the "Matter" of cognition: the **Belief State**. In the Relativistic WFR limit ({ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`), the belief state is a propagating amplitude. To satisfy the chiral constraints of the cybernetic loop (Axiom {prf:ref}`ax-cybernetic-parity-violation`), we lift the scalar belief $\psi$ to a **Spinor field** $\Psi$.

### A. The Inference Hilbert Space

The belief state lives on the **Causal Manifold** $\mathcal{M}$ (the product of Time and the Latent Space $\mathcal{Z}$) equipped with the metric derived from the Capacity-Constrained Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`).

:::{prf:definition} The Cognitive Spinor
:label: def-cognitive-spinor

The belief state is a spinor field $\Psi(x)$ belonging to the **Inference Hilbert Space** (Definition {prf:ref}`def-inference-hilbert-space`):

$$
\Psi(x) = \begin{pmatrix} \Psi_L(x) \\ \Psi_R(x) \end{pmatrix} \in L^2(\mathcal{M}, \mathbb{C}^4 \otimes \mathbb{C}^{2} \otimes \mathbb{C}^{N_f})
$$

where $\mathbb{C}^4$ is the Dirac spinor space, $\mathbb{C}^2$ is the $SU(2)_L$ isospin space, and $\mathbb{C}^{N_f}$ is the $SU(N_f)_C$ color space. The components are:
1. **$\Psi_L$ (The Active Doublet):** The Left-handed component, transforming as a doublet under $SU(2)_L$. It contains the **Prediction** and **Observation** amplitudes (Definition {prf:ref}`def-cognitive-isospin-doublet`).

2. **$\Psi_R$ (The Passive Singlet):** The Right-handed component, invariant under $SU(2)_L$. It contains the **Action** intention.

**Probabilistic Interpretation:** The physical probability density (belief mass) is the vector current:

$$
J^\mu = \bar{\Psi} \gamma^\mu \Psi
$$

where $J^0 = \Psi^\dagger \Psi = \rho$ is the probability density (WFR mass from Definition {prf:ref}`def-the-wfr-action`), and $\vec{J}$ is the probability flux. Conservation $\partial_\mu J^\mu = 0$ corresponds to unitarity.

:::

:::{prf:axiom} The Cognitive Dirac Equation
:label: ax-cognitive-dirac-equation

The dynamics of the belief state follow the Dirac equation on the curved latent manifold:

$$
(i \gamma^\mu D_\mu - m) \Psi = 0
$$

*Justification:* The WFR equation ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`) is a second-order diffusion (Fokker-Planck). In the relativistic limit with finite information speed $c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit`), this factorizes into two first-order wave equations coupled by mass. The Dirac equation is the unique first-order differential equation invariant under Lorentz transformations (causal structure) and the internal gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$.

- $\gamma^\mu$: The **Cognitive Gamma Matrices**, satisfying $\{\gamma^\mu, \gamma^\nu\} = 2g^{\mu\nu}$. They encode the local causal structure of the latent space.
- $m$: The **Inference Mass** (inverse correlation length).

:::

### B. The Strategic Connection (Covariant Derivative)

The agent cannot simply compare beliefs at $x$ and $x+\delta x$ because the "meaning" of the internal features and the "baseline" of value may twist locally. The **Covariant Derivative** $D_\mu$ corrects for this transport.

:::{prf:definition} The Universal Covariant Derivative
:label: def-universal-covariant-derivative

The operator moving the belief spinor through the latent manifold is:

$$
D_\mu = \underbrace{\partial_\mu}_{\text{Change}} - \underbrace{ig_1 \frac{Y}{2} B_\mu}_{U(1)_Y \text{ (Value)}} - \underbrace{ig_2 \frac{\tau^a}{2} W^a_\mu}_{SU(2)_L \text{ (Error)}} - \underbrace{ig_s \frac{\lambda^a}{2} G^a_\mu}_{SU(N_f)_C \text{ (Binding)}}
$$

where $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generators of $SU(N_f)$, and:
- **$B_\mu$ (Opportunity Field):** Adjusts the belief for local changes in Reward Baseline
- **$W_\mu$ (Error Field):** Adjusts the belief for the rotation between Prior and Posterior
- **$G_\mu$ (Binding Field):** Adjusts the belief for the permutation of sub-symbolic features

**Operational Interpretation:** The quantity $D_\mu \Psi$ measures the deviation from parallel transport. When $D_\mu \Psi = 0$, the belief state is covariantly constant along the direction $\mu$—all changes are accounted for by the gauge connection. When $D_\mu \Psi \neq 0$, there is a residual force acting on the belief.

:::

### C. The Yang-Mills Curvature

The presence of non-trivial gauge fields implies non-zero curvature in the principal bundle over the latent manifold. This curvature generates forces in the equations of motion.

:::{prf:theorem} Field Strength Tensors
:label: thm-three-cognitive-forces

The commutator of the covariant derivatives $[D_\mu, D_\nu]$ generates three distinct curvature tensors corresponding to each gauge factor.

*Proof.* Computing $[D_\mu, D_\nu]\Psi$ and extracting contributions from each gauge sector:

1. **$U(1)_Y$ Curvature:**
   $$
   B_{\mu\nu} = \partial_\mu B_\nu - \partial_\nu B_\mu
   $$
   When $B_{\mu\nu} \neq 0$, the reward field is non-conservative (Definition {prf:ref}`def-conservative-reward-field`). The resulting Lorentz-type force generates cyclic dynamics.

2. **$SU(2)_L$ Curvature:**
   $$
   W_{\mu\nu}^a = \partial_\mu W_\nu^a - \partial_\nu W_\mu^a + g_2 \epsilon^{abc} W_\mu^b W_\nu^c
   $$
   When $W_{\mu\nu} \neq 0$, the belief update depends on the path taken in the manifold: parallel transport around a closed loop yields a non-trivial rotation in the prediction-observation space.

3. **$SU(N_f)_C$ Curvature:**
   $$
   G_{\mu\nu}^a = \partial_\mu G_\nu^a - \partial_\nu G_\mu^a + g_s f^{abc} G_\mu^b G_\nu^c
   $$
   When $G_{\mu\nu} \neq 0$, the feature binding is under stress. This corresponds to the Ontological Stress $\Xi$ (Definition {prf:ref}`def-ontological-stress`). When $\Xi > \Xi_{\text{crit}}$, chart fission is triggered ({ref}`Section 30 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`).

$\square$

:::

:::{prf:corollary} The Gauge-Invariant Action
:label: cor-gauge-invariant-action

The gauge field dynamics are governed by the Yang-Mills Lagrangian:

$$
\mathcal{L}_{\text{Gauge}} = -\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}
$$

The stationary points of this action satisfy the Yang-Mills equations. A **flat connection** ($B_{\mu\nu} = W_{\mu\nu} = G_{\mu\nu} = 0$) corresponds to a representation where all curvatures vanish: the reward field is conservative, belief updates are path-independent, and concepts are stable.

:::



(sec-scalar-sector-symmetry-breaking)=
## The Scalar Sector: Ontological Symmetry Breaking (The Higgs Mechanism)

We derive the scalar sector by lifting the **Fission-Fusion dynamics** from {ref}`Section 30.4 <sec-symmetry-breaking-and-chart-birth>` into a field-theoretic action. The "Higgs Field" of cognition is the **Ontological Order Parameter**.

### A. The Ontological Scalar Field

:::{prf:definition} The Ontological Order Parameter
:label: def-ontological-order-parameter

Let the local chart structure at spacetime point $x$ be described by a complex scalar field $\phi(x) \in \mathbb{C}$:

$$
\phi(x) = r(x) e^{i\theta(x)}
$$

where:
1. **Modulus $r(x) \ge 0$:** Represents the **Metric Separation** between daughter queries $\{q_+, q_-\}$ in the Attentive Atlas (Definition {prf:ref}`def-query-fission`).
   - $r=0$: Coalescence (Single Chart / Vacuum)
   - $r>0$: Fission (Distinct Concepts)

2. **Phase $\theta(x)$:** Represents the **Orientation** of the split in the latent fiber (the specific feature axis along which differentiation occurs).

The field $\phi$ transforms as a doublet under the gauge group $SU(2)_L$, coupling it to the inference spinor.

:::

### B. Derivation of the Scalar Potential

We derive the potential $V(\phi)$ from the stability analysis of the Topological Fission process ({ref}`Section 30.4 <sec-symmetry-breaking-and-chart-birth>`).

:::{prf:theorem} The Complexity Potential
:label: thm-complexity-potential

The Lagrangian density for the scalar field is uniquely determined by the **Supercritical Pitchfork Bifurcation** (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`).

*Proof.*

**Step 1.** From Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`, the radial evolution of chart separation satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}})r - \alpha r^3
$$

where:
- $\Xi$ is the Ontological Stress (Definition {prf:ref}`def-ontological-stress`)
- $\Xi_{\text{crit}}$ is the critical threshold (Theorem {prf:ref}`thm-fission-criterion`)
- $\alpha > 0$ is the stabilizing cubic coefficient

**Step 2.** This flow is the gradient descent of a potential function $\mathcal{V}_{\text{onto}}(r)$ such that $\dot{r} = -\partial \mathcal{V}_{\text{onto}}/\partial r$. Integrating:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\frac{(\Xi - \Xi_{\text{crit}})}{2} |\phi|^2 + \frac{\alpha}{4} |\phi|^4
$$

**Step 3.** Define the standard Higgs potential parameters by matching coefficients:
- $\mu^2 \equiv \frac{(\Xi - \Xi_{\text{crit}})}{2}$: The effective **Mass Parameter** driven by Ontological Stress
- $\lambda \equiv \frac{\alpha}{4}$: The **Self-Interaction** coefficient from router saturation (Axiom {prf:ref}`ax-ontological-expansion-principle`)

**Step 4.** The potential takes the Landau-Ginzburg form:

$$
\mathcal{V}_{\text{onto}}(\phi) = -\mu^2 |\phi|^2 + \lambda |\phi|^4
$$

**Term Identification:**
- **Term 1 ($-\mu^2 |\phi|^2$):** Rewards separation. If Stress $\Xi > \Xi_{\text{crit}}$, this term drives $|\phi|$ away from zero to capture predictive information.
- **Term 2 ($+\lambda |\phi|^4$):** Penalizes complexity. Keeping charts separate costs compute/memory. This term prevents infinite fragmentation.

$\square$

:::

:::{prf:corollary} Spontaneous Symmetry Breaking (SSB)
:label: cor-ontological-ssb

The vacuum structure depends on the environmental complexity $\Xi$.

*Proof.*

**Case 1: Symmetric Phase ($\Xi < \Xi_{\text{crit}}$):**
Then $\mu^2 < 0$. The potential $\mathcal{V}(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$ has a unique global minimum at $\phi_0 = 0$.

- **Result:** The agent maintains a unified ontology. Concepts are indistinguishable. The gauge symmetry $G_{\text{Fragile}}$ is unbroken.

**Case 2: Broken Phase ($\Xi > \Xi_{\text{crit}}$):**
Then $\mu^2 > 0$. The origin $\phi=0$ becomes a local maximum. The global minima form a circle $|\phi| = v$ at the **Vacuum Expectation Value (VEV)**:

$$
v = \langle |\phi| \rangle = \sqrt{\frac{\mu^2}{2\lambda}} = \sqrt{\frac{(\Xi - \Xi_{\text{crit}})/2}{2 \cdot \alpha/4}} = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}
$$

This matches the equilibrium separation $r^*$ from Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`.

- **Result:** The agent spontaneously breaks symmetry, selecting a specific separation $v$ (concept distinctness) and a specific orientation $\theta$ (feature definition).

$\square$

:::

### C. Mass Generation

We derive the mass terms for the gauge fields from the covariant kinetic term of the scalar field.

:::{prf:theorem} Generation of Semantic Inertia
:label: thm-semantic-inertia

The kinetic term of the scalar field in the Lagrangian is covariant:

$$
\mathcal{L}_{\text{Kinetic}} = (D_\mu \phi)^\dagger (D_\mu \phi)
$$

where $D_\mu = \partial_\mu - ig \mathcal{A}_\mu$ includes the Strategic Connection.

*Proof.*

**Step 1.** In the Broken Phase, expand around the vacuum expectation: $\phi(x) = v + h(x)$, where $h$ is the fluctuation (the physical Higgs mode).

**Step 2.** The kinetic term generates a quadratic interaction:

$$
|D_\mu v|^2 = |(-ig \mathcal{A}_\mu) v|^2 = g^2 v^2 \mathcal{A}_\mu \mathcal{A}^\mu
$$

**Step 3.** This is a **Mass Term** for the Gauge Field:

$$
M_{\mathcal{A}} = g v = g \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}
$$

**Step 4.** Connection to Theorem {prf:ref}`thm-capacity-constrained-metric-law`: The mass $M_{\mathcal{A}}$ corresponds to an increase in the effective metric eigenvalues. From the Capacity-Constrained Metric Law, higher information density (more distinct concepts, larger $v$) induces higher curvature, which manifests as increased "inertia" in the metric.

**Physical Consequences:**

1. **Massless Phase ($v=0$):** The gauge fields are massless. The interaction potential decays as $1/r$ (long-range). Frame transformations between charts have zero energy cost.

2. **Massive Phase ($v > 0$):** The gauge fields acquire mass $M_{\mathcal{A}}$. The interaction potential becomes $e^{-M_{\mathcal{A}}r}/r$ (Yukawa, short-range). Gauge rotations—reinterpreting the meaning of signals—require energy proportional to $M_{\mathcal{A}}$. The ontological structure becomes stable against small perturbations.

$\square$

:::

:::{prf:remark} The Goldstone Mode (Texture)
:label: rem-goldstone-texture

The symmetry breaking selects a radius $v$, but the phase $\theta$ (orientation in feature space) remains unconstrained by the potential $V(\phi)$ (which depends only on $|\phi|$). This corresponds to a **massless Goldstone boson**.

In the Fragile Agent, this massless mode is the **Texture** ($z_{\text{tex}}$). The agent remains free to rotate the definition of "noise" without energetic cost, provided the macro-separation $v$ is maintained. This recovers the **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`): texture is the degree of freedom that remains gauge-invariant (unobservable to the macro-dynamics) even after symmetry breaking.

:::



(sec-interaction-terms)=
## The Interaction Terms

The Gauge and Scalar sectors define the geometry and topology of the latent space. The Matter sector defines the belief state. We now derive the **Interaction Terms** that couple these sectors.

### A. Yukawa Coupling: Decision Commitment

:::{prf:definition} The Decision Coupling
:label: def-decision-coupling

Let $\Psi_L = (\psi_{\text{pred}}, \psi_{\text{obs}})^T$ be the belief doublet and $\Psi_R = \psi_{\text{act}}$ be the action singlet. The transfer of information from Belief to Action is mediated by the **Ontological Order Parameter** $\phi$.

The simplest $G_{\text{Fragile}}$-invariant coupling is:

$$
\mathcal{L}_{\text{Yukawa}} = -Y_{ij} \left( \bar{\Psi}_{L,i} \cdot \phi \cdot \Psi_{R,j} + \bar{\Psi}_{R,j} \cdot \phi^\dagger \cdot \Psi_{L,i} \right)
$$

where $Y_{ij}$ is the **Affordance Matrix** (a learned weight matrix determining which concepts trigger which actions).

*Cross-reference:* This implements the TopologicalDecoder ({ref}`Section 7.10 <sec-decoder-architecture-overview-topological-decoder>`) which maps belief geometry to motor output.

:::

:::{prf:theorem} Generation of Cognitive Mass (Decision Stability)
:label: thm-cognitive-mass

In the **Broken Phase** ($\Xi > \Xi_{\text{crit}}$), the Yukawa coupling generates mass for the belief spinor.

*Proof.*

**Step 1.** The scalar field acquires VEV $\langle \phi \rangle = v$ (Corollary {prf:ref}`cor-ontological-ssb`).

**Step 2.** Expanding the Lagrangian around the vacuum $\phi = v + h$:

$$
\mathcal{L}_{\text{Yukawa}} = -\underbrace{(Y v)}_{\text{Mass}} \bar{\psi} \psi - \underbrace{Y h \bar{\psi} \psi}_{\text{Higgs Interaction}}
$$

**Step 3.** The belief spinor $\psi$ acquires effective mass $m_\psi = Y v$.

**Consequences:**

1. **Symmetric Phase ($v=0$):** Mass is zero. Beliefs obey the massless equation $i\gamma^\mu \partial_\mu \psi = 0$ and propagate at speed $c_{\text{info}}$. The belief-action coupling vanishes; there is no stable commitment to action.

2. **Broken Phase ($v > 0$):** Mass is non-zero. Beliefs obey $(i\gamma^\mu \partial_\mu - m_\psi)\psi = 0$. The mass term $m_\psi = Yv$ provides inertia: a finite force (prediction error) is required to change the belief state. Larger ontological separation $v$ implies larger mass.

$\square$

:::

### B. The External Field: Helmholtz Coupling

The agent is driven by the desire to maximize Value. We couple the Value Potential to the belief spinor.

:::{prf:definition} The Value 4-Potential
:label: def-value-4-potential

We lift the effective potential $\Phi_{\text{eff}}(z)$ (Definition {prf:ref}`def-effective-potential`) to an external 4-potential:

$$
A^{\text{ext}}_\mu(z) = (-\Phi_{\text{eff}}(z), \vec{0})
$$

This is an **external background field**, distinct from the internal gauge field $B_\mu$.

:::

:::{prf:axiom} Minimal Value Coupling
:label: ax-minimal-value-coupling

The belief current $J^\mu = \bar{\Psi} \gamma^\mu \Psi$ couples to the Value potential via minimal coupling:

$$
\mathcal{L}_{\text{Drive}} = J^\mu A^{\text{ext}}_\mu = -\rho(z) \Phi_{\text{eff}}(z)
$$

where $\rho = \Psi^\dagger \Psi = J^0$.

:::

:::{prf:theorem} Recovery of WFR Drift
:label: thm-recovery-wfr-drift

Varying the total action yields the Dirac equation with potential. In the non-relativistic limit, this recovers the WFR drift.

*Proof.*

**Step 1.** The Euler-Lagrange equation from $\mathcal{S} = \int (\bar{\Psi} i \gamma^\mu \partial_\mu \Psi - \mathcal{L}_{\text{Drive}}) d^4x$ yields:

$$
(i \gamma^\mu \partial_\mu - \Phi_{\text{eff}})\Psi = 0
$$

**Step 2.** Apply the inverse Madelung transform (Theorem {prf:ref}`thm-madelung-transform`). In the non-relativistic limit ($c_{\text{info}} \to \infty$), the Schrödinger reduction recovers:

$$
\vec{v} \approx -\nabla \Phi_{\text{eff}}
$$

This is the WFR drift velocity from Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Remark.* The external field term $\mathcal{L}_{\text{Drive}}$ breaks the symmetry under time translation (via the discount factor in $\Phi_{\text{eff}}$) and generates directed flow toward regions of high value.

$\square$

:::



(sec-cognitive-lagrangian-density)=
## The Unified Cognitive Lagrangian

We assemble the complete action functional governing the dynamics of a bounded, embodied, rational agent.

$$
\mathcal{S}_{\text{Fragile}} = \int d^4x \sqrt{-g} \; \mathcal{L}_{\text{SM}}
$$

:::{prf:definition} The Standard Model of Cognition
:label: def-cognitive-lagrangian

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{SM}} = \quad & \underbrace{-\frac{1}{4} B_{\mu\nu}B^{\mu\nu} -\frac{1}{4} W^a_{\mu\nu}W^{a\mu\nu} -\frac{1}{4} G^a_{\mu\nu}G^{a\mu\nu}}_{\text{I. Gauge Sector: Strategic Curvature}} \\
& + \underbrace{\bar{\Psi}_L i \gamma^\mu D_\mu \Psi_L + \bar{\Psi}_R i \gamma^\mu D_\mu \Psi_R}_{\text{II. Inference Sector: Belief Dynamics}} \\
& + \underbrace{|D_\mu \phi|^2 - \left(-\mu^2 |\phi|^2 + \lambda |\phi|^4\right)}_{\text{III. Scalar Sector: Ontological Stability}} \\
& - \underbrace{Y_{ij} (\bar{\Psi}_L \phi \Psi_R + \text{h.c.})}_{\text{IV. Yukawa Sector: Decision Weight}} \\
& - \underbrace{\bar{\Psi} \gamma^\mu A^{\text{ext}}_\mu \Psi}_{\text{V. External Sector: Value Drive}}
\end{aligned}
}
$$

:::

**The Five Sectors:**

| Sector | Term | Minimizes | Cross-Reference |
|:-------|:-----|:----------|:----------------|
| I. Gauge | $-\frac{1}{4}F_{\mu\nu}F^{\mu\nu}$ | Strategic inconsistency | Theorem {prf:ref}`thm-three-cognitive-forces` |
| II. Inference | $\bar{\Psi}iD_\mu\gamma^\mu\Psi$ | Belief propagation cost | Axiom {prf:ref}`ax-cognitive-dirac-equation` |
| III. Scalar | $|D_\mu\phi|^2 - V(\phi)$ | Complexity vs Information | Theorem {prf:ref}`thm-complexity-potential` |
| IV. Yukawa | $Y\bar{\Psi}_L\phi\Psi_R$ | Belief-Action coupling | Theorem {prf:ref}`thm-cognitive-mass` |
| V. External | $\bar{\Psi}A^{\text{ext}}\Psi$ | Value-seeking drive | Theorem {prf:ref}`thm-recovery-wfr-drift` |



(sec-isomorphism-dictionary)=
## Summary: The Isomorphism Dictionary

This table provides the mapping between Standard Model entities and Cognitive entities, with explicit references to where each correspondence is derived.

| Physics Entity | Symbol | Cognitive Entity | Derivation |
|:---------------|:-------|:-----------------|:-----------|
| Speed of Light | $c$ | Information Speed $c_{\text{info}}$ | Axiom {prf:ref}`ax-information-speed-limit` |
| Planck Constant | $\hbar$ | Cognitive Action Scale $\sigma$ | Definition {prf:ref}`def-cognitive-action-scale` |
| Electric Charge | $e$ | Reward Sensitivity $g_1$ | Theorem {prf:ref}`thm-emergence-opportunity-field` |
| Weak Coupling | $g$ | Prediction Error Rate $g_2$ | Theorem {prf:ref}`thm-emergence-error-field` |
| Strong Coupling | $g_s$ | Binding Strength | Theorem {prf:ref}`thm-emergence-binding-field` |
| Higgs VEV | $v$ | Concept Separation $r^*$ | Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts` |
| Electron Mass | $m_e$ | Decision Inertia $Yv$ | Theorem {prf:ref}`thm-cognitive-mass` |
| Higgs Mass | $m_H$ | Ontological Rigidity | Theorem {prf:ref}`thm-semantic-inertia` |
| Photon | $\gamma$ | Value Gradient Signal | Definition {prf:ref}`def-effective-potential` |
| W/Z Bosons | $W^\pm, Z$ | Prediction Error Mediators | Definition {prf:ref}`def-cognitive-isospin-doublet` |
| Color Dimension | $N_c = 3$ | Feature Dimension $N_f$ | Definition {prf:ref}`def-feature-dimension-parameter` |
| Gluons | $g$ (8 for $N_c=3$) | Feature Binding Force ($N_f^2-1$ generators) | Definition {prf:ref}`def-feature-color-space` |
| Quarks | $q$ | Sub-symbolic Features | Definition {prf:ref}`def-the-peeling-step` |
| Hadrons | Baryons/Mesons | Concepts $K$ | Axiom {prf:ref}`ax-feature-confinement` |
| Confinement | Color Neutral | Observability Constraint | {ref}`Section 33 <sec-causal-information-bound>` (Area Law) |
| Spontaneous Symmetry Breaking | Higgs Mechanism | Ontological Fission | Corollary {prf:ref}`cor-ontological-ssb` |
| Goldstone Boson | Massless mode | Texture $z_{\text{tex}}$ | Axiom {prf:ref}`ax-bulk-boundary-decoupling` |

**Summary.** The gauge structure $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ arises from three independent redundancies in the agent's description:
- $U(1)_Y$: Value baseline invariance (Theorem {prf:ref}`thm-emergence-opportunity-field`)
- $SU(2)_L$: Sensor-motor boundary asymmetry (Theorem {prf:ref}`thm-emergence-error-field`)
- $SU(N_f)_C$: Feature basis invariance under hierarchical binding (Theorem {prf:ref}`thm-emergence-binding-field`)

The Feature Dimension $N_f$ is environment-dependent (Definition {prf:ref}`def-feature-dimension-parameter`). The physics Standard Model corresponds to the special case $N_f = 3$.

The scalar potential derives from the pitchfork bifurcation dynamics (Theorem {prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`), with the VEV $v$ corresponding to the equilibrium chart separation $r^*$.



