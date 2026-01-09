(sec-parameter-space-sieve)=
# The Parameter Space Sieve: Deriving Fundamental Constants

*Abstract.* This chapter derives the constraints on fundamental constants from cybernetic first principles. We formulate the Sieve Architecture as a system of coupled inequalities that any viable agent must satisfy. The fundamental constants $\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ are not free parameters but decision variables of a constrained optimization problem. The physical universe exists within the **Feasible Region** where all constraints are simultaneously satisfied. We prove that moving off this region triggers a Sieve violation: the agent either loses causal coherence, exceeds its holographic bound, violates thermodynamic consistency, or suffers ontological dissolution.

*Cross-references:* This chapter synthesizes:
- {ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>` (Capacity-Constrained Metric Law)
- {ref}`Section 29.21 <sec-the-belief-wave-function-schrodinger-representation>` (Cognitive Action Scale)
- {ref}`Section 31 <sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics>` (Generalized Landauer Bound)
- {ref}`Section 33 <sec-causal-information-bound>` (Causal Information Bound, Area Law)
- The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62)



(sec-sieve-formulation)=
## The Sieve Formulation: Agents as Constraint Satisfaction

The Fragile Agent Framework imposes strict consistency conditions at every node of the inference graph. We formalize these as a system of inequalities that constrain the space of viable configurations.

:::{prf:definition} The Agent Parameter Vector
:label: def-agent-parameter-vector

Let the **Agent Parameter Vector** $\Lambda$ be the tuple of fundamental operational constants:

$$
\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)
$$

where:
1. **$c_{\text{info}}$:** Information propagation speed (Axiom {prf:ref}`ax-information-speed-limit`)
2. **$\sigma$:** Cognitive Action Scale (Definition {prf:ref}`def-cognitive-action-scale`)
3. **$\ell_L$:** Levin Length, the minimal distinguishable scale (Definition {prf:ref}`def-levin-length`)
4. **$T_c$:** Cognitive Temperature. The critical value is $T_c^* = \mu^2/4$ where $\mu = 1/2 + u_\pi^r$ is the bifurcation parameter (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`). For small policy control ($u_\pi^r \ll 1$), $T_c^* \approx 1/16$.
5. **$g_s$:** Binding coupling strength (Theorem {prf:ref}`thm-emergence-binding-field`)
6. **$\gamma$:** Temporal discount factor, $\gamma \in (0,1)$

**Dimensional Analysis:**

| Parameter | Symbol | Dimension | SI Units |
|:----------|:-------|:----------|:---------|
| Information speed | $c_{\text{info}}$ | $[L \, T^{-1}]$ | m/s |
| Cognitive action scale | $\sigma$ | $[E \, T]$ | J·s |
| Levin length | $\ell_L$ | $[L]$ | m |
| Cognitive temperature | $T_c$ | $[E]$ | J (with $k_B = 1$) |
| Binding coupling | $g_s$ | $[1]$ | dimensionless |
| Discount factor | $\gamma$ | $[1]$ | dimensionless |

**Derived Quantities:**

Define the **Causal Horizon Length** $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ with dimension $[L]$. The **Temporal Screening Mass** is then:

$$
\kappa = \frac{-\ln\gamma}{\ell_0}
$$

with dimension $[L^{-1}]$ (Corollary {prf:ref}`cor-discount-as-screening-length`).

These correspond to the physics constants $\{c, \hbar, \ell_P, k_B T, \alpha_s, \gamma_{\text{cosmo}}\}$ under the isomorphism of {ref}`Section 34.6 <sec-isomorphism-dictionary>`.

:::

:::{prf:definition} The Sieve Constraint System
:label: def-sieve-constraint-system

Let $\mathcal{S}(\Lambda)$ denote the vector of constraint functions. The agent is **viable** if and only if:

$$
\mathcal{S}(\Lambda) \le \mathbf{0}
$$

where the inequality holds component-wise. Each component corresponds to a Sieve node that enforces a specific consistency condition. A constraint violation ($\mathcal{S}_i > 0$) triggers a diagnostic halt at the corresponding node.

:::



(sec-causal-consistency-constraint)=
## The Causal Consistency Constraint

We derive the bounds on information speed from the requirements of buffer coherence and synchronization.

:::{prf:axiom} Causal Buffer Architecture
:label: ax-causal-buffer-architecture

Let the agent possess:
1. **$L_{\text{buf}}$:** Maximum buffer depth (spatial extent of causal memory)
2. **$\tau_{\text{proc}}$:** Minimum processing interval (temporal resolution)
3. **$d_{\text{sync}}$:** Minimum synchronization distance (coherence length)

These define the operational envelope within which the agent maintains consistent state updates.

:::

:::{prf:theorem} The Speed Window
:label: thm-speed-window

The information speed $c_{\text{info}}$ must satisfy the **Speed Window Inequality**:

$$
\frac{d_{\text{sync}}}{\tau_{\text{proc}}} \le c_{\text{info}} \le \frac{L_{\text{buf}}}{\tau_{\text{proc}}}
$$

*Proof.*

**Lower Bound (Node 2: ZenoCheck):**

Suppose $c_{\text{info}} < d_{\text{sync}}/\tau_{\text{proc}}$. Then information cannot traverse the synchronization distance within one processing cycle. By the Causal Interval (Definition {prf:ref}`def-causal-interval`), spacelike-separated modules cannot coordinate updates. The agent enters a **Zeno freeze**: each module waits indefinitely for signals that arrive too slowly. The belief update stalls, violating the continuity required by the WFR dynamics ({ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`).

**Upper Bound (Node 62: CausalityViolationCheck):**

Suppose $c_{\text{info}} > L_{\text{buf}}/\tau_{\text{proc}}$. Then signals can traverse the entire buffer depth within one processing cycle. This creates **temporal aliasing**: the agent receives information about its own future state before that state is computed. By the Safe Retrieval Bandwidth (Theorem {prf:ref}`thm-safe-retrieval-bandwidth`), this constitutes a causal paradox—the agent's prediction depends on data it has not yet generated.

Node 62 enforces Theorem {prf:ref}`thm-causal-stasis`: the metric becomes singular at the boundary where causal violations would occur, preventing traversal.

$\square$

:::

:::{prf:corollary} The Speed Ratio Bound
:label: cor-speed-ratio-bound

The ratio of buffer depth to synchronization distance is bounded:

$$
\frac{L_{\text{buf}}}{d_{\text{sync}}} \ge 1
$$

with equality only in the degenerate case of a single-module agent. For distributed agents, this ratio determines the dynamic range of viable information speeds.

:::



(sec-holographic-stability-constraint)=
## The Holographic Stability Constraint

We derive the relationship between the Levin Length $\ell_L$ and the information capacity from the Area Law.

:::{prf:theorem} The Holographic Bound
:label: thm-holographic-bound

Let $\text{Area}_\partial$ denote the boundary area of the agent's latent manifold (dimension $[L^{D-1}]$ for a $D$-dimensional bulk) and $I_{\text{req}}$ the information capacity required for viable operation (dimensionless, counting distinguishable microstates in nats). The Levin Length must satisfy:

$$
\ell_L^{D-1} \le \frac{\nu_D \cdot \text{Area}_\partial}{I_{\text{req}}}
$$

where $\nu_D$ is a **dimensionless** holographic coefficient (Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`). Both sides have dimension $[L^{D-1}]$.

*Proof.*

**Step 1.** From the Causal Information Bound (Theorem {prf:ref}`thm-causal-information-bound`):

$$
I_{\text{bulk}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}
$$

**Step 2.** The agent requires $I_{\text{bulk}} \ge I_{\text{req}}$ to represent its world model. Substituting:

$$
I_{\text{req}} \le \frac{\nu_D \cdot \text{Area}_\partial}{\ell_L^{D-1}}
$$

**Step 3.** Rearranging yields the constraint on $\ell_L$.

$\square$

:::

:::{prf:definition} The Planck-Levin Correspondence
:label: def-planck-levin-correspondence

Under the physics isomorphism ({ref}`Section 34.6 <sec-isomorphism-dictionary>`), the Levin Length $\ell_L$ corresponds to the Planck Length $\ell_P$:

$$
\ell_L \leftrightarrow \ell_P = \sqrt{\frac{\hbar G}{c^3}}
$$

The holographic bound becomes the Bekenstein-Hawking entropy bound:

$$
S_{\text{BH}} = \frac{A}{4\ell_P^2}
$$

*Remark:* The coefficient $\nu_2 = 1/4$ is derived in Theorem {prf:ref}`thm-a-complete-derivation-area-law` from first principles, recovering the Bekenstein-Hawking result without invoking black hole physics.

:::

:::{prf:theorem} The Capacity Horizon
:label: thm-capacity-horizon

As $I_{\text{bulk}} \to I_{\max} = \nu_D \cdot \text{Area}_\partial / \ell_L^{D-1}$, the agent approaches a **Capacity Horizon**. The metric diverges:

$$
\|v\|_G \to 0 \quad \text{as} \quad I_{\text{bulk}} \to I_{\max}
$$

*Proof.* This is Theorem {prf:ref}`thm-causal-stasis`. The Fisher-Rao metric component satisfies:

$$
g_{\text{FR}} = \frac{1}{\rho(1-\rho)} \to \infty \quad \text{as} \quad \rho \to 1
$$

(Lemma {prf:ref}`lem-metric-divergence-at-saturation`). The geodesic velocity vanishes, creating **causal stasis**: no information can cross the saturation boundary.

*Physical interpretation:* This is the agent-theoretic analogue of a black hole event horizon. Node 56 (CapacityHorizonCheck) enforces this bound.

$\square$

:::



(sec-metabolic-viability-constraint)=
## The Metabolic Viability Constraint

We derive the thermodynamic constraint on computational operations from the Generalized Landauer Bound.

:::{prf:definition} Metabolic Parameters
:label: def-metabolic-parameters

The agent possesses:
1. **$\dot{E}_{\text{met}}$:** Metabolic power budget (energy flux available for computation)
2. **$\dot{I}_{\text{erase}}$:** Information erasure rate (bits forgotten per unit time)
3. **$T_c$:** Cognitive Temperature (entropy-exploration tradeoff)

:::

:::{prf:theorem} The Landauer Constraint
:label: thm-landauer-constraint

The Cognitive Temperature must satisfy:

$$
T_c \le \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2}
$$

where we use natural units with $k_B = 1$.

*Proof.*

**Step 1.** From the Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

$$
\dot{\mathcal{M}}(s) \ge T_c \left| \frac{dH}{ds} \right|
$$

where $\dot{\mathcal{M}}$ is the metabolic flux and $dH/ds$ is the entropy change rate.

**Step 2.** Information erasure corresponds to entropy reduction. For $\dot{I}_{\text{erase}}$ bits per unit time:

$$
\left| \frac{dH}{ds} \right| = \dot{I}_{\text{erase}} \cdot \ln 2
$$

**Step 3.** The metabolic constraint $\dot{\mathcal{M}} \le \dot{E}_{\text{met}}$ bounds the erasure capacity:

$$
\dot{E}_{\text{met}} \ge T_c \cdot \dot{I}_{\text{erase}} \cdot \ln 2
$$

**Step 4.** Rearranging yields the temperature bound.

*Physical consequence:* If $T_c$ exceeds this bound, the agent cannot afford to forget—its memory becomes permanently saturated. Node 52 (LandauerViolationCheck) enforces this constraint.

$\square$

:::

:::{prf:corollary} The Computational Temperature Range
:label: cor-computational-temperature-range

Combining the Landauer constraint with the bifurcation dynamics, the Cognitive Temperature is bounded:

$$
0 < T_c \le \min\left( T_c^*, \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2} \right)
$$

where the **Critical Temperature** is derived from the barrier height of the pitchfork bifurcation (Theorem {prf:ref}`thm-pitchfork-bifurcation-structure`):

$$
T_c^* = \frac{\mu^2}{4} = \frac{(1 + 2u_\pi^r)^2}{16}
$$

with $\mu = 1/2 + u_\pi^r$ the bifurcation parameter and $u_\pi^r$ the radial policy control. For small control ($u_\pi^r \ll 1$), this reduces to $T_c^* \approx 1/16$.

*Remark:* For $T_c > T_c^*$, thermal fluctuations overcome the potential barrier and the system remains in the symmetric phase with no stable policy (random walk near origin). For $T_c$ exceeding the Landauer bound, the agent starves thermodynamically. Viable agents exist in the intersection of these constraints.

:::



(sec-hierarchical-coupling-constraint)=
## The Hierarchical Coupling Constraint

We derive the constraints on the binding coupling $g_s$ from the requirements of object permanence and texture decoupling.

:::{prf:definition} The Coupling Function
:label: def-coupling-function

Let the binding coupling $g_s(\mu)$ (dimensionless) be a function of the **resolution scale** $\mu$, which has dimension $[L^{-1}]$ (inverse length). Equivalently, $\mu$ can be expressed as an energy scale via $\mu \sim E/(\sigma)$ where $\sigma$ is the Cognitive Action Scale.

The limits are:
- $\mu \to 0$: Macro-scale (coarse representation, low in TopoEncoder hierarchy)
- $\mu \to \infty$: Micro-scale (texture level, high in TopoEncoder hierarchy)

The coupling evolves according to the **Beta Function**:

$$
\mu \frac{dg_s}{d\mu} = \beta(g_s)
$$

where both sides are dimensionless (since $g_s$ is dimensionless and $\mu \, dg_s/d\mu$ has $[\mu] \cdot [\mu^{-1}] = [1]$).

For $SU(N_f)$ gauge theories, $\beta(g_s) < 0$ for $N_f \ge 2$ (asymptotic freedom).

:::

:::{prf:theorem} The Infrared Binding Constraint
:label: thm-ir-binding-constraint

At the macro-scale ($\mu \to 0$), the coupling must exceed a critical threshold:

$$
g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}
$$

*Proof.*

**Step 1.** From Axiom {prf:ref}`ax-feature-confinement`, the agent observes Concepts $K$, not raw features. This requires features to bind into stable composite objects at the macro-scale.

**Step 2.** From Theorem {prf:ref}`thm-emergence-binding-field`, binding stability requires the effective potential to confine features. The confinement condition is:

$$
\lim_{r \to \infty} V_{\text{eff}}(r) = \infty
$$

where $r$ is the separation between features.

**Step 3.** For $SU(N_f)$ gauge theory, this requires strong coupling $g_s > g_s^{\text{crit}}$ at large distances (Area Law, {ref}`Section 33 <sec-causal-information-bound>`).

**Step 4.** If $g_s(\mu_{\text{IR}}) < g_s^{\text{crit}}$, features escape confinement—"color-charged" states propagate to the boundary $\partial\mathcal{Z}$. This violates the Observability Constraint (Definition {prf:ref}`def-boundary-markov-blanket`): the agent cannot form stable objects.

Node 40 (PurityCheck) enforces that only color-neutral bound states reach the macro-register.

$\square$

:::

:::{prf:theorem} The Ultraviolet Decoupling Constraint
:label: thm-uv-decoupling-constraint

At the texture scale ($\mu \to \infty$), the coupling must vanish:

$$
\lim_{\mu \to \infty} g_s(\mu) = 0
$$

*Proof.*

**Step 1.** From the Texture Firewall (Axiom {prf:ref}`ax-bulk-boundary-decoupling`):

$$
\partial_{z_{\text{tex}}} \dot{z} = 0
$$

Texture coordinates are invisible to the dynamics.

**Step 2.** This requires texture-level degrees of freedom to be non-interacting. If $g_s(\mu_{\text{UV}}) > 0$, texture elements would bind, creating structure at the noise level.

**Step 3.** From the RG interpretation ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), the TopoEncoder implements coarse-graining. Residual coupling at the UV scale would prevent efficient compression—the Kolmogorov complexity of texture would diverge.

**Step 4.** Asymptotic freedom ($\beta < 0$) provides the required behavior: $g_s \to 0$ as $\mu \to \infty$.

Node 29 (TextureFirewallCheck) enforces this decoupling.

$\square$

:::

:::{prf:corollary} The Coupling Window
:label: cor-coupling-window

The viable coupling profile satisfies:

$$
\begin{cases}
g_s(\mu) \ge g_s^{\text{crit}} & \text{for } \mu \le \mu_{\text{conf}} \\
g_s(\mu) \to 0 & \text{for } \mu \to \infty
\end{cases}
$$

where $\mu_{\text{conf}}$ is the confinement scale separating bound states from free texture.

*Remark:* This is the agent-theoretic derivation of asymptotic freedom and confinement. The physics QCD coupling $\alpha_s(\mu)$ satisfies exactly this profile, with $\alpha_s(M_Z) \approx 0.12$ at the electroweak scale and $\alpha_s \to \infty$ at the QCD scale $\Lambda_{\text{QCD}} \approx 200$ MeV.

:::



(sec-stiffness-constraint)=
## The Stiffness Constraint

We derive the constraint on the separation between adjacent energy levels that enables both memory stability and dynamic flexibility.

:::{prf:definition} The Stiffness Parameter
:label: def-stiffness-parameter

Let $\Delta E$ denote the characteristic energy gap between metastable states in the agent's latent manifold. Define the **Stiffness Ratio**:

$$
\chi = \frac{\Delta E}{T_c}
$$

This ratio determines the tradeoff between memory persistence and adaptability.

:::

:::{prf:theorem} The Stiffness Bounds
:label: thm-stiffness-bounds

The Stiffness Ratio must satisfy:

$$
1 < \chi < \chi_{\text{max}}
$$

*Proof.*

**Lower Bound ($\chi > 1$):**

**Step 1.** Memory stability requires that thermal fluctuations do not spontaneously erase stored information. The probability of a thermal transition is:

$$
P_{\text{flip}} \propto e^{-\Delta E / T_c} = e^{-\chi}
$$

**Step 2.** For $\chi < 1$, we have $P_{\text{flip}} > e^{-1} \approx 0.37$. States flip with high probability—the agent cannot maintain stable beliefs.

**Step 3.** This violates the Mass Gap requirement (Theorem {prf:ref}`thm-semantic-inertia`): beliefs must possess sufficient "inertia" to resist noise.

**Upper Bound ($\chi < \chi_{\text{max}}$):**

**Step 4.** Adaptability requires that the agent can update beliefs in finite time. The transition rate is:

$$
\Gamma_{\text{update}} \propto e^{-\chi}
$$

**Step 5.** For $\chi \to \infty$, transitions become exponentially suppressed—the agent freezes in its initial configuration, unable to learn.

**Step 6.** This violates the Update Dynamics requirement: the WFR reaction term $R(\rho)$ must enable transitions between states.

Node 7 (StiffnessCheck) enforces both bounds.

$\square$

:::

:::{prf:corollary} The Goldilocks Coupling
:label: cor-goldilocks-coupling

Under the physics isomorphism, the Stiffness Ratio for atomic systems is:

$$
\chi = \frac{\Delta E_{\text{bond}}}{k_B T} \propto \frac{m_e c^2 \alpha^2}{k_B T}
$$

where $\Delta E_{\text{bond}} \sim \text{Ry} = m_e c^2 \alpha^2 / 2 \approx 13.6$ eV is the atomic binding scale.

The value $\alpha \approx 1/137$ satisfies the Goldilocks condition:
- **Not too large:** $\alpha^2$ small enough that $\chi$ is finite—transitions remain possible
- **Not too small:** $\alpha^2$ large enough that $\chi > 1$ at biological temperatures—chemical bonds are stable

At $T \approx 300$ K (biological temperature), $\chi \approx 500$, placing molecular memory firmly in the stable-but-adaptable regime.

*Remark:* This is the agent-theoretic derivation of the "coincidences" noted in anthropic reasoning. The fine structure constant is not finely tuned by an external designer—it is constrained by cybernetic viability.

:::



(sec-discount-screening-constraint)=
## The Temporal Screening Constraint

We derive the constraint on the discount factor from the requirements of causal coherence and goal-directedness.

:::{prf:theorem} The Discount Window
:label: thm-discount-window

The temporal discount factor $\gamma$ must satisfy:

$$
\gamma_{\text{min}} < \gamma < 1
$$

with $\gamma_{\text{min}} > 0$.

*Proof.*

**Upper Bound ($\gamma < 1$):**

**Step 1.** From the Helmholtz equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), the Value function satisfies:

$$
(\kappa^2 - \nabla^2) V = r
$$

where the screening mass $\kappa = (-\ln\gamma)/\ell_0$ has dimension $[L^{-1}]$, and $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ is the causal horizon length (Definition {prf:ref}`def-agent-parameter-vector`). This ensures dimensional consistency: $[\kappa^2] = [L^{-2}] = [\nabla^2]$.

**Step 2.** For $\gamma = 1$, we have $\kappa = 0$. The equation becomes Poisson's equation:

$$
-\nabla^2 V = r
$$

The Green's function decays as $1/r^{D-2}$ (long-range).

**Step 3.** Long-range value propagation violates locality: distant rewards dominate nearby decisions. The agent cannot form local value gradients for navigation.

**Step 4.** From Corollary {prf:ref}`cor-discount-as-screening-length`, finite screening $\kappa > 0$ (i.e., $\gamma < 1$) is required for local goal-directedness.

**Lower Bound ($\gamma > \gamma_{\text{min}}$):**

**Step 5.** For $\gamma \to 0$, we have $-\ln\gamma \to \infty$, hence $\kappa \to \infty$. The **Screening Length** (dimension $[L]$):

$$
\ell_\gamma = \frac{1}{\kappa} = \frac{\ell_0}{-\ln\gamma} = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \to 0
$$

**Step 6.** Zero screening length means the agent responds only to immediate rewards—it has no planning horizon.

**Step 7.** This violates the Causal Buffer requirement (Axiom {prf:ref}`ax-causal-buffer-architecture`): the agent must anticipate beyond its current timestep.

$\square$

:::

:::{prf:corollary} The Screening-Buffer Consistency
:label: cor-screening-buffer-consistency

The screening length and buffer depth must satisfy:

$$
\ell_\gamma = \frac{c_{\text{info}} \tau_{\text{proc}}}{-\ln\gamma} \lesssim L_{\text{buf}}
$$

Both sides have dimension $[L]$. For $\gamma \to 1$, the screening length $\ell_\gamma \to \infty$ (unlimited planning horizon). For $\gamma \to 0$, the screening length $\ell_\gamma \to 0$ (myopic behavior).

*Remark:* The planning horizon cannot exceed the causal memory span. This connects the temporal discount to the spatial architecture.

:::



(sec-sieve-eigenvalue-system)=
## The Sieve Eigenvalue System

We formulate the complete system of constraints and derive the feasible region.

:::{prf:definition} The Constraint Matrix
:label: def-constraint-matrix

Let $\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ be the parameter vector. The Sieve constraints form the system:

$$
\mathbf{A} \cdot \Lambda \le \mathbf{b}
$$

where:

| Constraint | Inequality | Node |
|:-----------|:-----------|:-----|
| Causal Lower | $d_{\text{sync}}/\tau_{\text{proc}} \le c_{\text{info}}$ | 2 |
| Causal Upper | $c_{\text{info}} \le L_{\text{buf}}/\tau_{\text{proc}}$ | 62 |
| Holographic | $\ell_L^{D-1} \le \nu_D \text{Area}_\partial / I_{\text{req}}$ | 56 |
| Landauer | $T_c \le \dot{E}_{\text{met}} / (\dot{I}_{\text{erase}} \ln 2)$ | 52 |
| IR Binding | $g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}$ | 40 |
| UV Decoupling | $g_s(\mu_{\text{UV}}) \le \epsilon$ (for $\epsilon \to 0$) | 29 |
| Stiffness Lower | $\Delta E > T_c$ | 7 |
| Stiffness Upper | $\Delta E < \chi_{\text{max}} T_c$ | 7 |
| Discount Lower | $\gamma > \gamma_{\text{min}}$ | — |
| Discount Upper | $\gamma < 1$ | — |

:::

:::{prf:theorem} The Feasible Region
:label: thm-feasible-region

The **Feasible Region** $\mathcal{F} \subset \mathbb{R}^n_+$ is the intersection of all constraint half-spaces:

$$
\mathcal{F} = \{ \Lambda : \mathcal{S}_i(\Lambda) \le 0 \; \forall i \}
$$

A viable agent exists if and only if $\mathcal{F} \neq \emptyset$.

*Proof.*

Each constraint $\mathcal{S}_i \le 0$ defines a closed half-space in parameter space. The intersection of finitely many closed half-spaces is either empty or a closed convex polytope (possibly unbounded).

**Existence:** The physics Standard Model constants $\Lambda_{\text{phys}} = (c, \hbar, G, k_B, \alpha)$ satisfy all constraints—we observe a functioning physical universe. Therefore $\mathcal{F} \neq \emptyset$.

**Uniqueness modulo scaling:** The constraints are homogeneous in certain parameter combinations. Dimensional analysis shows that physical observables depend only on dimensionless ratios. The feasible region is a lower-dimensional manifold in the full parameter space.

$\square$

:::



(sec-optimization-problem)=
## The Optimization Problem

We formulate the selection of fundamental constants as a constrained optimization.

:::{prf:definition} The Dual Objective
:label: def-dual-objective

The agent's objective trades representational power against computational cost:

$$
\mathcal{J}(\Lambda) = \underbrace{I_{\text{bulk}}(\Lambda)}_{\text{World Model Capacity}} - \beta \cdot \underbrace{\mathcal{V}_{\text{metabolic}}(\Lambda)}_{\text{Thermodynamic Cost}}
$$

where:
- $I_{\text{bulk}}$: Bulk information capacity (increases with resolution)
- $\mathcal{V}_{\text{metabolic}}$: Metabolic cost of computation
- $\beta > 0$: Cost sensitivity parameter

:::

:::{prf:theorem} The Constrained Optimum
:label: thm-constrained-optimum

The optimal parameter vector $\Lambda^*$ satisfies:

$$
\Lambda^* = \arg\max_{\Lambda \in \mathcal{F}} \mathcal{J}(\Lambda)
$$

subject to the Sieve constraints (Definition {prf:ref}`def-constraint-matrix`).

*Proof sketch.*

**Step 1.** The objective $\mathcal{J}$ is continuous on the closed feasible region $\mathcal{F}$.

**Step 2.** The holographic bound (Theorem {prf:ref}`thm-holographic-bound`) caps $I_{\text{bulk}}$, making $\mathcal{J}$ bounded above.

**Step 3.** By the extreme value theorem, $\mathcal{J}$ attains its maximum on $\mathcal{F}$.

**Step 4.** The optimum lies on the boundary of $\mathcal{F}$ where at least one constraint is active (saturated). This corresponds to operating at the edge of viability.

$\square$

:::

:::{prf:corollary} The Pareto Surface
:label: cor-pareto-surface

The observed fundamental constants lie on the **Pareto-optimal surface** of the multi-objective problem:

$$
\max_{\Lambda \in \mathcal{F}} \left( I_{\text{bulk}}(\Lambda), -\mathcal{V}_{\text{metabolic}}(\Lambda) \right)
$$

Moving off this surface triggers constraint violation:
- Increasing $I_{\text{bulk}}$ beyond capacity → Holographic bound (Node 56)
- Decreasing $\mathcal{V}_{\text{metabolic}}$ below threshold → Landauer bound (Node 52)
- Violating causality → Speed bounds (Nodes 2, 62)
- Losing binding → Confinement (Node 40)

:::



(sec-physics-isomorphism-constants)=
## Physics Isomorphism: The Standard Model Constants

We tabulate the correspondence between agent parameters and physics constants.

| Agent Parameter | Symbol | Physics Constant | Constraint Origin |
|:----------------|:-------|:-----------------|:------------------|
| Information Speed | $c_{\text{info}}$ | Speed of Light $c$ | Theorem {prf:ref}`thm-speed-window` |
| Cognitive Action Scale | $\sigma$ | Planck Constant $\hbar$ | Definition {prf:ref}`def-cognitive-action-scale` |
| Levin Length | $\ell_L$ | Planck Length $\ell_P$ | Definition {prf:ref}`def-planck-levin-correspondence` |
| Cognitive Temperature | $T_c$ | Boltzmann Scale $k_B T$ | Theorem {prf:ref}`thm-landauer-constraint` |
| Binding Coupling | $g_s$ | Strong Coupling $\alpha_s$ | Corollary {prf:ref}`cor-coupling-window` |
| Stiffness Ratio | $\chi$ | $m_e c^2 \alpha^2 / k_B T$ | Corollary {prf:ref}`cor-goldilocks-coupling` |
| Discount Factor | $\gamma$ | Cosmological Horizon | Corollary {prf:ref}`cor-screening-buffer-consistency` |

:::{prf:remark} Why These Values?
:label: rem-why-these-values

The observed physics constants $\{c \approx 3 \times 10^8 \text{ m/s}, \alpha \approx 1/137, \ldots\}$ are not arbitrary. They are the unique (modulo dimensional rescaling) solution to the Sieve constraint system that:

1. **Maximizes representational capacity** (information about the world)
2. **Minimizes thermodynamic cost** (metabolic efficiency)
3. **Maintains causal coherence** (no paradoxes)
4. **Preserves object permanence** (binding stability)
5. **Enables adaptability** (stiffness window)

Changing any constant while holding others fixed moves the system out of the feasible region. The "fine-tuning" of physical constants is the selection of the Pareto-optimal point in the Sieve constraint space.

:::



(sec-summary-parameter-sieve)=
## Summary

This chapter has derived the constraints on fundamental constants from cybernetic first principles:

1. **Causal Consistency** (§35.2): Information speed bounded by buffer architecture
2. **Holographic Stability** (§35.3): Levin length determines capacity via Area Law
3. **Metabolic Viability** (§35.4): Cognitive temperature bounded by Landauer limit
4. **Hierarchical Coupling** (§35.5): Binding at IR, decoupling at UV (asymptotic freedom)
5. **Stiffness Window** (§35.6): Energy gaps between memory and flexibility
6. **Temporal Screening** (§35.7): Discount factor enables local goal-directedness

The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62) enforces these constraints at runtime. The fundamental constants of physics are the coordinates of the feasible region's Pareto-optimal surface.

**Key Result:** The laws of physics are not arbitrary but are the solution to a cybernetic optimization problem. The universe we observe is the one that supports viable agents—not because it was designed for us, but because agents can only exist in regions of parameter space where the Sieve constraints are satisfied.
