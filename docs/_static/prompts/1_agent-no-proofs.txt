## intro_agent.md

:::{prf:theorem} The RL Degeneracy Theorem
:label: thm-rl-degeneracy

Standard Reinforcement Learning is recovered from the Fragile Agent framework under the joint limit:

$$
\text{Standard RL} = \lim_{\substack{G \to I \\ |\mathcal{K}| \to \infty \\ \Xi_{\text{crit}} \to \infty}} \text{Fragile Agent}
$$
where:
1. **Flat Geometry** ($G \to I$): The state-space metric becomes Euclidean, eliminating coordinate-invariant updates
2. **Infinite Capacity** ($|\mathcal{K}| \to \infty$): No information bottleneck, continuous state space without quantization
3. **No Safety Constraints** ($\Xi_{\text{crit}} \to \infty$): The Sieve is disabled, all actions permitted

:::

## 01_foundations/01_definitions.md

:::{prf:definition} Bounded-Rationality Controller
:label: def-bounded-rationality-controller

The agent is a controller with internal state

$$
Z_t := (K_t, z_{n,t}, z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}},

$$
and internal components (Encoder/Shutter, World Model, Critic, Policy). Its evolution is driven only by the observable interaction stream at the interface (observations/feedback) and by its own outgoing control signals (actions).

:::

:::{prf:definition} Boundary / Markov Blanket
:label: def-boundary-markov-blanket

The boundary variables at time $t$ are the interface tuple

$$
B_t := (x_t,\ r_t,\ d_t,\ \iota_t,\ a_t),

$$
where:
- $x_t\in\mathcal{X}$ is the observation (input sample),
- $r_t\in\mathbb{R}$ is the boundary reward sample (the pairing of the reward 1-form with the latent step; its cumulative integral is path-independent only in the conservative case),
- $d_t\in\{0,1\}$ is termination (an exogenous absorbing-event flag; on the latent side it is represented by $\Gamma_{\text{term}}$ only when termination is $\sigma(Z_t)$-measurable),
- $\iota_t$ denotes any additional side channels (costs, constraints, termination reasons, privileged signals),
- $a_t\in\mathcal{A}$ is action (control signal sent outward).

:::

:::{prf:definition} Environment as Generative Process
:label: def-environment-as-generative-process

The "environment" is the conditional law of future interface signals given past interface history. Concretely it is a (possibly history-dependent) kernel on incoming boundary signals conditional on the boundary history:

$$
P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_{\le t}).

$$
In the Markov case this reduces to the familiar RL kernel

$$
P_{\partial}(x_{t+1}, r_{t+1}, d_{t+1}, \iota_{t+1}\mid B_t),

$$
but the **interpretation changes**: $P_{\partial}$ is not "a dataset generator"; it is the **input-output law** that the controller must cope with under partial observability and model mismatch.

This is the categorical move: we do not assume access to the environment's latent variables; we work only with the **law over observable interface variables**.

:::

:::{prf:definition} Agent symmetry group; operational
:label: def-agent-symmetry-group-operational

Let:
- $G_{\text{obj}}$ be an **objective/feedback gauge** acting on scalar feedback signals (e.g., change of units or baseline shift). A common choice is the positive affine group

  $$
  G_{\text{obj}} := \{(\alpha,\beta_0): \alpha>0,\ r\mapsto \alpha r+\beta_0\}.

  $$
  (For a fixed-temperature entropy-regularized objective, this is an operational reparameterization: the scale must be co-scaled with $T_c$, and an offset is a symmetry only when terminal data and horizon conventions are co-transformed.)
- $G_{\text{spatial}}$ be an **observation gauge** acting on raw observations $x$ (e.g., pose/translation/rotation; choose $SE(3)$, $SE(2)$, $\mathrm{Sim}(2)$, or a task-specific subgroup depending on sensors).
- $S_{\mathcal K}$ be the **symbol-permutation symmetry** of the discrete macro register: it is $S_{|\mathcal K|}$ for a flat codebook and $S_{N_v}\wr S_{N_c}$ (chart permutations and within-chart code permutations) for the two-level Attentive Atlas register.
- $\mathrm{Symp}(2n,\mathbb{R})$ be an optional **phase-space symmetry** acting on canonical latent coordinates $z=(q,p)\in\mathbb{R}^{2n}$ when the world model is parameterized as a symplectic/Hamiltonian system {cite}`greydanus2019hamiltonian` ({ref}`sec-defect-functionals-implementing-regulation`.B).

The (candidate) total symmetry group is the direct product

$$
\mathcal{G}_{\mathbb{A}}
:=
G_{\text{obj}}
\times
G_{\text{spatial}}
\times
S_{\mathcal K}
\times
\mathrm{Symp}(2n,\mathbb{R}).

$$
**Internal vs. external symmetries.**
- **Internal (objective) gauge:** co-transformations of the scalar feedback and its temperature/terminal conventions that preserve the chosen objective; a fixed-$T_c$ or fixed-terminal task need not be invariant under the full affine group.
- **External (observation) gauge:** transformations of the input stream that change *pose* but not *identity*.

**Principle of covariance (engineering requirement).** The internal maps of the agent should be equivariant under $\mathcal{G}_{\mathbb{A}}$ in the following typed sense:
- **Shutter $E$**: canonicalize or quotient $G_{\text{spatial}}$ before discretization, so the macro register is approximately invariant:

  $$
  K(x)\approx K(g\cdot x)\quad (g\in G_{\text{spatial}}),

  $$
  while $z_n$ carries structured nuisance parameters (pose/basis/disturbance coordinates) and $z_{\mathrm{tex}}$ carries reconstruction-only texture ({ref}`sec-the-shutter-as-a-vq-vae`, {ref}`sec-defect-functionals-implementing-regulation`.A).
- **World model $S$ and policy $\pi$:** be covariant to symbol permutations $S_{\mathcal K}$ by treating $K$ only through its embedding $e_K$ (not the integer label) and by using permutation-invariant diagnostics.
- **Critic/value and dual variables:** enforce stability and constraint satisfaction in a way that is robust to re-scaling/offset of the scalar feedback ({ref}`sec-defect-functionals-implementing-regulation`.C, {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`).

These are *requirements on representations and interfaces*, not philosophical claims: if an invariance is not enforced, the corresponding failure modes (symmetry blindness, brittle scaling, uncontrolled drift) become more likely and harder to debug.

:::

## 01_foundations/02_control_loop.md

:::{prf:definition} State-Space Sensitivity Metric
:label: def-state-space-sensitivity-metric

The **value-curvature component** of the state-space sensitivity metric at a point $z$ in the latent space is defined
from the symmetric Hessian of the value function, with a PSD proxy used when enforcing metric positivity:

$$
(G_V)_{ij} = \frac{\partial^2 V}{\partial z_i \partial z_j} \quad \text{(theory)}, \qquad
(G_V)_{ij} \approx c_V\,\frac{\partial V}{\partial z_i}\frac{\partial V}{\partial z_j} \quad \text{(Gauss--Newton proxy)}.

$$

The **complete** metric used elsewhere is $G = G_V + \lambda_G G_\pi$ with $G_\pi$ the state-space Fisher component (Definition {prf:ref}`def-complete-latent-space-metric`). Units: $[(G_V)_{ij}]=\mathrm{nat}\,[z]^{-2}$ if $z$ is measured in units $[z]$; in the proxy form, $c_V$ carries units $\mathrm{nat}^{-1}$.
:::

:::{prf:definition} Complete Latent Space Metric
:label: def-complete-latent-space-metric

The complete state-space sensitivity metric on $\mathcal{Z}$ is defined as:

$$
G_{ij}(z) = \underbrace{(G_V)_{ij}(z)}_{\text{Hessian (value curvature)}} + \lambda_G \underbrace{(G_\pi)_{ij}(z)}_{\text{Fisher (control sensitivity)}},
\qquad
(G_\pi)_{ij}(z) := \mathbb{E}_{a \sim \pi} \left[ \frac{\partial \log \pi(a|z)}{\partial z_i} \frac{\partial \log \pi(a|z)}{\partial z_j} \right].

$$

Units follow the book's tracked-nat convention: if log-probabilities carry the tracked unit, then
$[G_\pi]=\mathrm{nat}^2[z]^{-2}$ and $[\lambda_G]=\mathrm{nat}^{-1}$ so that
$\lambda_GG_\pi$ has units $\mathrm{nat}[z]^{-2}$ like $G_V$. If log-probabilities are treated as dimensionless,
the corresponding nat factors are omitted consistently from both terms.
:::

:::{prf:definition} Causal Enclosure Condition
:label: def-causal-enclosure-condition

**Causal Enclosure Condition (Markov sufficiency).** With the nuisance/texture split ({ref}`sec-the-shutter-as-a-vq-vae`), let $(K_t, z_{n,t}, z_{\mathrm{tex},t}, K^{\text{act}}_t)$ be the internal state/action process and define the macrostate $K_t:=\Pi(Z_t)$ (projection to the discrete register). The macro-model requirement is the conditional independence

$$
K_{t+1}\ \perp\!\!\!\perp\ (z_{n,t}, z_{\mathrm{tex},t})\ \big|\ (K_t,K^{\text{act}}_t),

$$
equivalently the vanishing of a conditional mutual information:

$$
I(K_{t+1};z_{n,t},z_{\mathrm{tex},t}\mid K_t,K^{\text{act}}_t)=0.

$$
:::

:::{prf:definition} Closure Defect
:label: def-closure-defect

**Closure Defect (kernel-level).** Write the micro-dynamics as a Markov kernel $P(dz'\mid z,a)$ and let $P_\Pi(\cdot\mid z,a)$ be the pushforward kernel on $\mathcal{K}$ induced by $\Pi$. A learned macro-dynamics kernel $\bar{P}(\cdot\mid k,a)$ is enclosure-correct iff

$$
P_\Pi(\cdot\mid z,a)=\bar{P}(\cdot\mid \Pi(z),a)
\quad\text{for }P\text{-a.e. }z.

$$
A canonical defect functional is the expected divergence

$$
\delta_{\text{CE}}
:=
\mathbb{E}_{z,a}\Big[D_{\mathrm{KL}}\big(P_\Pi(\cdot\mid z,a)\ \Vert\ \bar{P}(\cdot\mid \Pi(z),a)\big)\Big].

$$
:::

:::{prf:assumption} Regularity Conditions for the Fragile Agent
:label: asm-regularity-conditions

1. **Smoothness:** $V \in C^2(\mathcal{Z})$ --- the Hessian exists and is continuous
2. **Positive Definiteness:** $G(z) \succ 0$ for all $z \in \mathcal{Z}$ --- the metric is non-degenerate
3. **Lipschitz Dynamics:** $\|f(z_1, a) - f(z_2, a)\| \leq L\|z_1 - z_2\|$ --- no discontinuities
4. **Bounded State Space:** $\mathcal{Z}$ is compact, or $V$ has appropriate growth at infinity
:::

:::{prf:definition} Local Conditioning Scale
:label: def-local-conditioning-scale

Let $(\mathcal{Z}, G)$ be the Riemannian latent manifold. Define a local conditioning scale $\vartheta: \mathcal{Z} \to \mathbb{R}^+$ as the trace of the inverse metric:

$$
\vartheta(z) := \frac{1}{d} \operatorname{Tr}\left( G^{-1}(z) \right)

$$
where $d = \dim(\mathcal{Z})$. The corresponding **precision / coupling coefficient** is
$\beta_{\text{cpl}}(z) = [\vartheta(z)]^{-1}$. When entropy regularization is tied to geometry, interpret
$\beta_{\text{cpl}}$ as a local inverse temperature; in an isothermal approximation where $\beta_{\text{cpl}}$ is
constant, set $\beta_{\text{cpl}} = 1/T_c$.
Units: for a general coordinate scale introduce $\ell_0$ with $[\ell_0^2]=[z]^2/\mathrm{nat}$ and set $\vartheta(z):=\operatorname{Tr}(G^{-1})/(d\ell_0^2)$; then $\beta_{\text{cpl}}$ is dimensionless and $\beta_{\text{cpl}}=1/T_c$ is meaningful in the normalised convention.

:::

:::{prf:remark} Variance-Curvature Correspondence (Scaling Ansatz)
:label: lem-variance-curvature-correspondence

Assume the action space is identified with latent displacements and that a stationary Gaussian policy is generated by
the same diffusion model as the latent dynamics. In entropy-regularized control, one useful scaling ansatz is:

$$
\Sigma_{\text{step}}(z) \propto T_c\,G^{-1}(z)

$$
This is a calibration relation, not a theorem implied by the metric definition. In maximum-entropy control / exponential-family models, stationary distributions over latent states
often take an exponential form $p(z)\propto \exp(-V(z)/T_c)$. In an isothermal approximation where
$\beta_{\text{cpl}}$ is constant, identify $T_c = \beta_{\text{cpl}}^{-1}$ only after the reference-scale convention
for $\vartheta$ has been fixed. Deviations can be measured by the covector defect
**$\mathcal{D}_{\beta_{\text{cpl}}} := \|d\log p + \beta_{\text{cpl}}(dV-A)\|_{G^{-1}}^2$**.

:::

:::{prf:definition} Entropy-Regularized Objective Functional
:label: def-entropy-regularized-objective-functional

Let $d\mu_G:=\sqrt{|G|}\,dz$ be the Riemannian volume form on $\mathcal{Z}$ and let $p(z)$ be a probability density with respect to $d\mu_G$. For a (dimensionless) trade-off coefficient $T_c\ge 0$, define

$$
\mathcal{F}[p,\pi]
:=
\int_{\mathcal{Z}} p(z)\Big(V(z) - T_c\,H(\pi(\cdot\mid z))\Big)\,d\mu_G,

$$
where $H(\pi(\cdot\mid z)) := -\mathbb{E}_{a\sim \pi(\cdot\mid z)}[\log \pi(a\mid z)]$ is the per-state policy entropy (in nats). Because $V$ and $H$ are measured in nats ({ref}`sec-units-and-dimensional-conventions`), $T_c$ is dimensionless.

:::

:::{prf:definition} Belief Density
:label: def-belief-density

Let $p(z,s)\ge 0$ be a density with respect to $d\mu_G$ representing the agent's belief (or belief-weight) over latent coordinates. In closed-system idealizations one may impose $\int_{\mathcal{Z}}p(z,s)\,d\mu_G=1$; in open-system implementations with explicit projections/reweightings we track the unnormalized mass and renormalize when needed ({ref}`sec-intrinsic-motivation-maximum-entropy-exploration`).

:::

:::{prf:definition} Transport Field
:label: def-transport-field

Let $v\in\Gamma(T\mathcal{Z})$ be a vector field describing the instantaneous transport of belief mass on $\mathcal{Z}$. In a value-gradient-flow idealization (used only for intuition), one may take

$$
v^i(z) := -G^{ij}(z)\frac{\partial V}{\partial z^j},

$$
so transport points in the direction of decreasing $V$ (Riemannian steepest descent). Units: if computation time is measured in solver units, then $[v]=[z]/\mathrm{solver\ time}$ (map to $\mathrm{step}$ using the $t \leftrightarrow s$ budget in {ref}`sec-the-chronology-temporal-distinctions`).

:::

:::{prf:lemma} Continuity Equation for Transport
:label: lem-continuity-equation-for-transport

If the belief density evolves only by deterministic transport under $v$ (no internal sources/sinks), then it satisfies the continuity equation

$$
\frac{\partial p}{\partial s} + \nabla_i \left( p v^i \right) = 0

$$
where $\nabla_i$ denotes the Levi-Civita covariant derivative associated with $G$.

:::

:::{prf:definition} Source Residual
:label: def-source-residual

In general, belief evolution may include additional update effects (e.g. approximation error, off-manifold steps, or explicit projection/reweighting). We collect these into a residual/source term $\sigma(z,s)$:

$$
\frac{\partial p}{\partial s} + \operatorname{div}_G(p v) = \sigma

$$
Interpreting $\sigma$:
1. If $\sigma>0$ on a region, belief mass is being created there beyond pure transport; this indicates an **ungrounded internal update** relative to the transport model.
2. If $\sigma<0$, belief mass is being removed beyond pure transport (aggressive forgetting or projection).
3. Integrating over any measurable region $U\subseteq\mathcal{Z}$ and applying the divergence theorem yields the exact mass balance

   $$
   \frac{d}{ds}\int_U p\,d\mu_G
   =
   -\oint_{\partial U}\langle p v,n\rangle\,dA_G
   +\int_U \sigma\,d\mu_G.

   $$
   For {math}`U=\mathcal{Z}` this relates net mass change to boundary flux and the integrated residual.

:::

:::{prf:proposition} Mass Conservation in a Closed Enclosure
:label: prop-mass-conservation-in-a-closed-enclosure

If $\sigma\equiv 0$ and the boundary flux vanishes (e.g. $\langle p v,n\rangle=0$ on $\partial\mathcal{Z}$), then the total belief mass

$$
\mathcal{V}(s):=\int_{\mathcal{Z}}p(z,s)\,d\mu_G

$$
is constant in time.

:::

:::{prf:definition} Observation Inflow Form
:label: def-observation-inflow-form

Let $j \in \Omega^{d-1}(\partial \mathcal{Z})$ be the **observation inflow form**. This form represents the rate of information entering the model through the interface.

:::

:::{prf:theorem} Generalized Conservation of Belief
:label: thm-generalized-conservation-of-belief

The evolution of the belief density $p$ satisfies the **Global Balance Equation**:

$$
\frac{d}{ds}\int_{\mathcal{Z}}p\,d\mu_G
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G
\;+\;
\int_{\mathcal{Z}} \sigma\,d\mu_G.

$$
where $n$ is the outward unit normal and $dA_G$ is the induced boundary area element. (Equivalently, if $\iota:\partial\mathcal{Z}\hookrightarrow \mathcal{Z}$ is the inclusion map, then the boundary flux is the pullback $\iota^*(p v\;\lrcorner\; d\mu_G)$.)

**The Architectural Sieve Condition (Node 13: BoundaryCheck).** The idealized "fully grounded" regime corresponds to $\sigma\approx 0$ in the interior: net changes in internal belief mass should be attributable to boundary influx and explicit projection events. Operationally we do not estimate $\sigma$ pointwise; instead Node 13 and the coupling-window diagnostics (Definition {prf:ref}`thm-information-stability-window-operational`) enforce non-collapse of $I(X;K)$ and bound posterior dispersion $H(p_t)$. Marginal code-usage entropy $H(\bar p(K))$ is monitored separately for codebook liveness.

$$
\frac{d\mathcal{V}}{ds}
=
-\oint_{\partial \mathcal{Z}} \langle p v,n\rangle\,dA_G,

$$
in the case $\sigma\equiv 0$.

Here $\langle p v,n\rangle$ is the outward flux density across the boundary (negative values correspond to net inflow).

**Distinction: boundary-driven updates vs ungrounded updates**

1.  **Valid learning (boundary-driven):** The belief changes because there is non-negligible boundary flux, i.e. new observations justify updating the internal state.
2.  **Ungrounded update (internal source):** The belief changes despite negligible boundary flux, corresponding to $\sigma>0$ under the transport model. Operationally, this is a warning sign that internal rollouts are decoupled from the data stream and should be treated as unreliable for control until re-grounded.

:::

:::{prf:corollary} Boundary filter interpretation
:label: cor-boundary-filter-interpretation

Sieve Nodes 13-16 (BoundaryCheck / InputSaturationCheck / SNRCheck / AlignCheck) can be interpreted as monitoring a trace-like coupling between bulk and boundary (informally: whether internal degrees of freedom remain supported by boundary evidence), analogous in spirit to the trace map $\operatorname{Tr}: H^1(\mathcal{Z}) \to H^{1/2}(\partial \mathcal{Z})$:

*   **Mode B.O (Overload):** Occurs when interface inflow exceeds the effective capacity of the manifold (Levin capacity), breaking the assumed operating regime.
*   **Mode B.D (Starvation):** Occurs when interface inflow is too weak, causing the internal information volume to decay (catastrophic forgetting).
*   **Mode B.C (Control Deficit):** A boundary-control mismatch at the critic/policy interface. AlignCheck detects value-boundary misalignment; when boundary information is present but the action repertoire is insufficient for the disturbance process, BarrierVariety records the requisite-variety subcase. The intervention mapping is summarized in {ref}`the failure-mode table <sec-failure-modes>`.

:::

## 02_sieve/01_diagnostics.md

:::{prf:definition} Component interfaces and diagnostic losses
A **thin interface** specifies the minimal coupling between components and the conditions that coupling must satisfy. The corresponding **defect functionals** ($\mathcal{L}_{\text{check}}$) measure departures from these conditions and provide the diagnostic losses used to enforce the interface checks during training.
:::

## 03_architecture/01_compute_tiers.md

:::{prf:definition} Attentive Routing Law
:label: def-attentive-routing-law

$$
w_i(x) := \frac{\exp\left(\frac{\langle k_i(z), q(z,f) \rangle}{\tau(z)}\right)}{\sum_{j=1}^{N_c} \exp\left(\frac{\langle k_j(z), q(z,f) \rangle}{\tau(z)}\right)}

$$
where $k_i(z) = U(z)\,\text{base\_query}_i$ and $\tau(z)$ is the metric-aware temperature. With `covariant_attn=False`, $U(z)=I$ and $\text{base\_query}_i = c_i$, reducing to dot-product routing on chart centers. This mechanism is **permutation equivariant**: shuffling the memory order of the chart tokens shuffles the output indices without changing the underlying topology or geometry.

:::

:::{prf:definition} The Macro-State Tree
:label: def-the-macro-state-tree

Let $\mathcal{T}$ be a rooted tree representing the hierarchical partition of the state space.

1. The **root** represents the entire observation space $\mathcal{X}$.
2. **Level 1 nodes** correspond to charts $K_{\text{chart}} \in \{1, \dots, N_c\}$.
3. **Level 2 nodes** correspond to codes $K_{\text{code}} \in \{1, \dots, N_v\}$ within a chart.
4. Edges represent the containment relationship (refinement of the partition).

Equip the vertex set $V(\mathcal{T})$ with the graph metric $d_{\mathcal{T}}$ (shortest path length).

:::

:::{prf:lemma} Gromov Hyperbolicity
:label: lem-gromov-hyperbolicity

The tree metric space $(\mathcal{T}, d_{\mathcal{T}})$ is $0$-hyperbolic in the sense of Gromov. That is, for any geodesic triangle, each side is contained in the $0$-neighborhood of the union of the other two sides.
:::

:::{prf:corollary} The Hyperbolic Embedding
:label: cor-the-hyperbolic-embedding

There exists a quasi-isometric embedding $\iota: V(\mathcal{T}) \hookrightarrow \mathbb{H}^n$ into $n$-dimensional hyperbolic space such that the depth in the tree correlates with the hyperbolic distance from a basepoint. In the upper half-space model $\mathbb{H}^n = \{(x, y) : y > 0\}$ with metric $ds^2 = (dx^2 + dy^2)/y^2$, tree depth $\ell$ maps to $\log(1/y)$; equivalently, in the Poincare ball model, depth maps to $2\tanh^{-1}(r)$ where $r \in [0,1)$ is the radial coordinate.

This identifies the **discrete macro-register** $K_t = (K_{\text{chart}}, K_{\text{code}})$ as the bulk of a hyperbolic geometry. Navigating from the root to a leaf corresponds to moving from the interior of $\mathbb{H}^n$ toward the ideal boundary $\partial_\infty \mathbb{H}^n$, increasing information resolution at each step.

:::

:::{prf:definition} The Local Fibre Structure
:label: def-the-local-fibre-structure

We model the latent space $\mathcal{Z}$ as a disjoint union of fibres over the discrete index set $\mathcal{K}$:

$$
\mathcal{Z} = \bigsqcup_{k \in \mathcal{K}} \mathcal{Z}_n^{(k)}, \qquad \mathcal{Z}_n^{(k)} \cong \mathbb{R}^{d_n}.

$$
For each macro-symbol $k \in \mathcal{K}$, the fibre $\mathcal{Z}_n^{(k)}$ represents the **structured nuisance** space (local pose/basis coordinates).

The interpolation of this discrete structure into a continuous manifold is achieved by the Attentive Atlas ({ref}`sec-tier-the-attentive-atlas`), which provides soft transition functions (partitions of unity) $\{w_i(x)\}$ that interpolate between fibres in overlap regions.

:::

:::{prf:remark} Texture as an Idealized Boundary
:label: prop-texture-as-the-ideal-boundary

The intended architecture treats the **texture residual** $z_{\text{tex}}$ as a boundary-like,
reconstruction-only channel, while $(K,z_n)$ carry the operational macro state. The current
implementation has a finite stack of refinement blocks, so it does not construct an infinite tree,
its limit set, or a conformal boundary at infinity. A literal identification of $z_{\text{tex}}$
with such a boundary requires an explicit infinite-depth refinement and a convergence theorem.

The causal-enclosure condition is therefore an architectural constraint imposed on the decoder and
transition model, not a consequence of the finite stack alone. This idealized boundary picture is
useful for organizing the latent channels, but it should not be read as a theorem about the finite
implementation.

:::

:::{prf:definition} The Latent Metric Tensor
:label: def-the-latent-metric-tensor

Working in the upper half-space model where depth $\rho \in [0, \infty)$ corresponds to $y = e^{-\rho}$, the metric $ds^2$ on the global latent space $\mathcal{Z}$ takes the form:

$$
ds^2 = d\rho^2 + d\sigma_{\mathcal{K}}^2 + e^{2\rho} \|dz_n\|^2

$$
where:

* $\rho$ is the resolution depth (hierarchy level), with $\rho = 0$ at the root and $\rho \to \infty$ at the boundary.
* $d\sigma_{\mathcal{K}}^2$ is the (discrete) metric on tree branches at fixed depth—operationally, it counts the number of chart/code transitions.
* $\|dz_n\|^2$ is the Euclidean metric on the structured nuisance $z_n$.
* A unit metric displacement in the horospherical coordinate $z_n$ is an $e^{-\rho}$ coordinate displacement;
  the factor $e^{2\rho}$ records the corresponding metric stretching at greater depth.

**Rigorous Interpretation of $z_n$:**
The structured nuisance $z_n$ is not stochastic noise; it is the **tangent space coordinate** on the horosphere (surface of constant depth $\rho$) determined by the active macro-symbol $K$. Horospheres in hyperbolic space are intrinsically flat (zero curvature), which is why local linear control theory (LTI approximations) applies within a single chart, even though the global geometry is hyperbolic.

This upper-half-space metric is a geometric model for the latent hierarchy; it is distinct from the
learned sensitivity metric $G$ used to precondition control updates.

:::

:::{prf:definition} The Peeling Step
:label: def-the-peeling-step

At layer $\ell$, the input signal $x^{(\ell)}$ is decomposed into a structural component (the **Effective Theory** at scale $\ell$) and a residual component (the **High-Frequency Fluctuations**).

1. **Analysis (Encoding):** The block identifies the macro-symbol $K^{(\ell)}$ and structured nuisance $z_n^{(\ell)}$ that best approximate $x^{(\ell)}$:

$$
(K^{(\ell)}, z_n^{(\ell)}) = \mathcal{E}^{(\ell)}(x^{(\ell)})

$$
2. **Synthesis (Effective Reconstruction):** The block generates the signal explained by this structure:

$$
\hat{x}^{(\ell)} = \mathcal{D}^{(\ell)}(K^{(\ell)}, z_n^{(\ell)})

$$
3. **Residual Computation (Texture Extraction):** The unexplained signal is isolated:

$$
z_{\text{tex}}^{(\ell)} = x^{(\ell)} - \hat{x}^{(\ell)}

$$
:::

:::{prf:definition} The Rescaling Operator / Renormalization
:label: def-the-rescaling-operator-renormalization

To prevent signal decay (vanishing activations) without using skip connections, we explicitly renormalize the residual to unit variance before passing it to the next scale:

$$
x^{(\ell+1)} = \frac{z_{\text{tex}}^{(\ell)}}{\sigma^{(\ell)} + \epsilon}, \qquad \sigma^{(\ell)} = \sqrt{\mathrm{Var}(z_{\text{tex}}^{(\ell)}) + \epsilon}

$$
The scalar $\sigma^{(\ell)}$ is stored as a state variable (the **scale factor**) for the decoding pass.

:::

:::{prf:definition} Total Reconstruction
:label: def-total-reconstruction

The original signal is reconstructed by summing the contributions of all scales, modulated by their respective scale factors. Define $\Pi^{(\ell)} := \prod_{j=0}^{\ell-1} \sigma^{(j)}$ with the convention $\Pi^{(0)} = 1$ (empty product). Then:

$$
\hat{x} = \sum_{\ell=0}^{L-1} \Pi^{(\ell)} \cdot \hat{x}^{(\ell)} + \Pi^{(L)} \cdot x^{(L)}

$$
:::

:::{prf:proposition} Gradient Preservation via Square Orthogonality
:label: prop-gradient-preservation-via-orthogonality

Let $W$ be a **square** weight matrix satisfying $W^T W = I$. Then:
1. All singular values of $W$ equal 1.
2. The backward gradient $\nabla_x \mathcal{L} = W^T \nabla_y \mathcal{L}$ satisfies $\|\nabla_x \mathcal{L}\| = \|\nabla_y \mathcal{L}\|$.
3. Neither explosion nor vanishing occurs across the layer.

:::

:::{prf:proposition} Forward Activation Stability
:label: prop-forward-activation-stability

Assume the input is standardized and each rescaling factor is computed from the preceding residual. Then:
1. $\mathrm{Var}(x^{(\ell)}) \approx 1$ for $\ell\ge1$ (up to the stabilizer and batch-estimation error).
2. Non-linearities (GELU) operate in their active region, avoiding saturation.
3. The backward gradient is scaled by $1/\sigma^{(\ell)}$, amplifying gradients for fine-scale layers.

**Gradient Amplification Analysis:** Let the loss $\mathcal{L}$ depend on the output of block $\ell$. The gradient flowing back to block $\ell-1$ includes the factor:

$$
\frac{\partial x^{(\ell)}}{\partial z_{\text{tex}}^{(\ell-1)}} = \frac{1}{\sigma^{(\ell-1)}}

$$
This scalar expression is exact only when $\sigma^{(\ell-1)}$ is detached from the forward graph. If it is estimated with gradients enabled, the Jacobian also contains the rank-one derivative of the variance estimate.
If training achieves variance reduction, assume $\sigma^{(\ell)}<1$ (the texture has less variance than the unit-normalized input). Under this additional hypothesis:
- **Without rescaling:** inputs to deeper layers decay exponentially ($\|x^{(\ell)}\| \to 0$), killing activations.
- **With rescaling:** inputs $x^{(\ell)}$ remain $O(1)$ (unit variance), keeping non-linearities in their active region.
- **Gradient amplification:** the backward gradient includes the factor $1/\sigma^{(\ell-1)} > 1$, counteracting the natural decay of fine-scale influence on the global loss.

This prevents the **Spectral Bias** where neural networks preferentially learn low frequencies and ignore high-frequency structure.

:::

:::{prf:proposition} Conditional upper bound for the continuous encoder path
:label: thm-dynamical-isometry-without-skip-connections

Assume the continuous encoder path (with the VQ step replaced by its straight-through surrogate) has square or explicitly isometric linear maps, each with operator norm at most $K$, and activations with Lipschitz constant at most $L_\phi$. Then its Jacobian $J_{\mathrm{enc}}$ satisfies the one-sided bound

$$
\sigma_{\max}(J_{\mathrm{enc}}) \le (K L_\phi)^L\prod_\ell(1+\epsilon_{\mathrm{orth}})^{1/2}.
$$

These hypotheses do not supply a positive lower singular-value bound. In particular, the full reconstruction map can be an identity by the algebraic peeling/reconstruction definitions, while the encoder path can still be contractive.

:::

:::{prf:definition} Factorized Jump Operator
:label: def-factorized-jump-operator

For each chart $i$, define:
- An **encoder** $B_i: \mathbb{R}^{d_n} \to \mathbb{R}^r$ that lifts local coordinates to the global tangent space.
- A **decoder** $A_j: \mathbb{R}^r \to \mathbb{R}^{d_n}$ that projects from the global tangent space to chart $j$'s coordinates.
- Bias terms $c_i \in \mathbb{R}^r$ and $d_j \in \mathbb{R}^{d_n}$.

The transition $L_{i \to j}$ is then:

$$
L_{i \to j}(z) = A_j(B_i z + c_i) + d_j

$$
:::

:::{prf:proposition} Parameter Efficiency
:label: prop-parameter-efficiency

The factorized parameterization requires $O(K \cdot r \cdot d_n)$ parameters instead of $O(K^2 \cdot d_n^2)$.

:::

:::{prf:definition} Overlap Consistency Loss
:label: def-overlap-consistency-loss

For a pair of charts $(i, j)$ with non-empty overlap, define the pairwise consistency loss as:

$$
\mathcal{L}_{\text{jump}}^{(i,j)} = \mathbb{E}_{x : w_i(x) > \tau, \, w_j(x) > \tau} \left[ \left\| z_n^{(j)} - L_{i \to j}(z_n^{(i)}) \right\|^2 \right]

$$
where $z_n^{(i)}$ and $z_n^{(j)}$ are the nuisance coordinates computed independently by chart $i$ and chart $j$'s encoders, and $w_i(x), w_j(x)$ are the soft router weights. The total overlap consistency loss sums over all overlapping pairs:

$$
\mathcal{L}_{\text{jump}} = \sum_{i < j} \mathcal{L}_{\text{jump}}^{(i,j)}

$$
**Intuition:** If the encoder correctly identifies that $x$ belongs to both charts, then applying the jump operator to chart $i$'s encoding should yield chart $j$'s encoding. Any discrepancy indicates that the transition functions are inconsistent with the actual data manifold.

**Implementation Details:**

1. **Overlap Detection:** A point $x$ is in the overlap $U_i \cap U_j$ if both router weights exceed a threshold:

   $$
   \mathbf{1}[x \in U_i \cap U_j] \approx \mathbf{1}[w_i(x) > \tau] \cdot \mathbf{1}[w_j(x) > \tau]

   $$
   With soft routers ({ref}`sec-tier-the-attentive-atlas`), we use the product $w_i(x) \cdot w_j(x)$ as a soft indicator.

2. **Sampling Overlaps:** Computing all $K^2$ pairs is expensive. We sample:
   - The top-2 charts per point (from router weights).
   - Random chart pairs with probability proportional to their co-activation frequency.

3. **Symmetry Penalty (Optional):** To encourage approximate invertibility:

   $$
   \mathcal{L}_{\text{inv}} = \mathbb{E}_{x, i, j} \left[ \left\| z_n^{(i)} - L_{j \to i}(L_{i \to j}(z_n^{(i)})) \right\|^2 \right]

   $$
:::

## 03_architecture/02_disentangled_vae.md

:::{prf:definition} The Three-Channel Latent Decomposition
:label: def-three-channel-latent

The internal state at time $t$ decomposes as:

$$
Z_t = (K_t, z_{n,t}, z_{\mathrm{tex},t})
$$

where:

1. $K_t = (K_{\mathrm{chart}}, K_{\mathrm{code}})$ is the discrete macro state. $K_{\mathrm{chart}}$
   selects an atlas chart, and $K_{\mathrm{code}}$ selects a local code within that chart.
2. $z_{n,t} \in \mathbb{R}^{d_n}$ is the structured nuisance (pose, basis, gauge residual).
3. $z_{\mathrm{tex},t} \in \mathbb{R}^{d_{\mathrm{tex}}}$ is reconstruction-only texture.

The implementation diagrams below use a common width $d_n=d_{\mathrm{tex}}=D$. If the nuisance or
texture widths differ, insert learned embeddings into the $D$-dimensional decoder input.

The geometry latent used by the decoder is

$$
z_{\mathrm{geo}} = c_{\mathrm{bar}} + z_{q,\mathrm{st}} + z_n
$$

where $c_{\mathrm{bar}}$ is the chart center mixture and $z_{q,\mathrm{st}}$ is the straight-through
quantized code.
:::

:::{prf:definition} The Golden Rule of Causal Enclosure
:label: def-causal-enclosure

The macro symbol must satisfy the causal enclosure property:

$$
I\!\left(K_{t+1};(z_{n,t},z_{\mathrm{tex},t})\mid K_t,a_t\right)=0.
$$

This is the joint enclosure condition: once the current macro state and action are given, neither
residual channel carries additional predictive information about the next macro symbol. Predictive
concentration of $P(K_{t+1}\mid K_t,a_t)$ is a separate model-quality diagnostic.
:::

:::{prf:definition} The Total TopoEncoder Loss
:label: def-total-disentangled-loss

The compound loss is:

$$
\mathcal{L}_{\text{total}} =
\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} + \lambda_{\text{ent}}\,\mathcal{L}_{\text{entropy}} +
\lambda_{\text{cons}}\,\mathcal{L}_{\text{consistency}} +
\sum_{i \in \text{tiers}} \lambda_i \mathcal{L}_i +
\lambda_{\text{jump}}\,\mathcal{L}_{\text{jump}} +
\lambda_{\text{sup}}\,\mathcal{L}_{\text{sup}}.
$$

Where:

- $\mathcal{L}_{\text{recon}} = \|x - \hat{x}\|^2$ (MSE reconstruction).
- $\mathcal{L}_{\text{vq}}$ is the codebook + commitment loss.
- $\mathcal{L}_{\text{entropy}}=\log N_c-\frac1B\sum_bH(w_b)$ is an entropy-raising anti-collapse
  regularizer; chart usage/diversity terms prevent dead charts.
- $\mathcal{L}_{\text{consistency}}$ aligns encoder and decoder routing.
- Tiered losses include variance, diversity, separation, codebook centering, chart center
  separation, residual scale, window, disentangle, orthogonality, code entropy, per-chart code
  entropy, KL prior, orbit, and VICReg invariance.
- $\mathcal{L}_{\text{jump}}$ enforces chart transition consistency when the jump operator is
  enabled.
- $\mathcal{L}_{\text{sup}}$ applies supervised topology when labels are available.

Learned precisions can reweight reconstruction, VQ, and supervised terms when enabled.
:::

:::{prf:definition} Routing Sharpness
:label: def-routing-sharpness

Let $K$ be the chart assignment and $N_c$ the number of charts. Define

$$
\rho_{\text{route}} = 1 - \frac{H(K \mid X)}{\log N_c}.
$$

Values near 1 indicate deterministic soft routing and values near 0 indicate diffuse routing. The
quantity equals $I(X;K)/\log N_c$ only when the marginal chart usage is uniform.
:::

:::{prf:definition} Hierarchical Latent Stack
:label: def-hierarchical-latent

A multi-scale atlas uses a hierarchy of discrete chart codes:

$$
Z_t = (K_t^{(0)}, K_t^{(1)}, \ldots, K_t^{(L)}, z_{n,t}, z_{\mathrm{tex},t})
$$

where each level $\ell$ has its own chart set and codebook, and higher levels capture coarser
structure.
:::

## 03_architecture/03_optimization.md

:::{prf:definition} Preconditioned Update
:label: def-preconditioned-update
The preconditioned update is

$$
\theta_{t+1} = \theta_t - \eta_t M_t g_t,
$$
with $M_t$ SPD and $g_t = \nabla\mathcal{V}(\theta_t)$.
:::

:::{prf:theorem} Preconditioned Descent (Sufficient Condition)
:label: thm-preconditioned-descent
Under A1--A2, if

$$
0 < \eta_t < \frac{2 m_{\min}}{L m_{\max}^2},
$$
then the update in Definition {prf:ref}`def-preconditioned-update` satisfies

$$
\mathcal{V}(\theta_{t+1}) \le \mathcal{V}(\theta_t) - \left(\eta_t m_{\min} - \frac{L}{2}\eta_t^2 m_{\max}^2\right)
\|g_t\|^2.
$$
In particular, $\mathcal{V}$ decreases whenever $g_t \ne 0$. If
$\eta_t = 2 m_{\min} / (L m_{\max}^2)$, the right-hand side yields nonincrease.
:::

:::{prf:definition} Relative Trust Region (Mach Limit)
:label: def-relative-trust-region
A relative trust region is the constraint

$$
\|\theta_{t+1} - \theta_t\| \le \kappa\,(\|\theta_t\| + \epsilon_\theta),
$$
with $\kappa \in (0,1)$ and $\epsilon_\theta \ge 0$. The case $\epsilon_\theta = 0$ recovers the strict relative form,
while $\epsilon_\theta > 0$ avoids degeneracy at $\|\theta_t\| = 0$.
:::

:::{prf:lemma} Trust-Region Scaling Preserves Descent
:label: lem-trust-region-scaling
Let $d_t = \eta_t M_t g_t$ with $\eta_t$ satisfying Theorem {prf:ref}`thm-preconditioned-descent`. Define the scaled step

$$
\tilde d_t = s d_t, \qquad s := \min\left(1, \frac{\kappa(\|\theta_t\| + \epsilon_\theta)}{\|d_t\|} \right).
$$
Then $\mathcal{V}(\theta_t - \tilde d_t) \le \mathcal{V}(\theta_t)$.
:::

:::{prf:proposition} Discrete Varentropy Brake
:label: prop-varentropy-brake-discrete
Let $T_t > 0$ be the cognitive temperature and $V_H(\theta_t)$ the varentropy. Define

$$
T_{t+1} = T_t\left(1 - \frac{\eta_T}{1 + \gamma V_H(\theta_t)}\right),
$$
with $\eta_T \in (0,1)$ and $\gamma > 0$. If a relaxation constant $C>0$ is available and
$\eta_T\le 2C\sqrt{\gamma}$, then

1. $0 < T_{t+1} \le T_t$ (temperature is positive and nonincreasing), and
2. $|T_{t+1} - T_t| \le C T_t / \sqrt{V_H(\theta_t)}$ whenever $V_H(\theta_t)>0$ (cooling is slowed when
   $V_H$ is large).
:::

:::{prf:definition} Gradient-Momentum Alignment
:label: def-gradient-alignment
Let $g_t$ be the gradient and $m_t$ a momentum estimate. Define the alignment score

$$
a_t := g_t^\top m_t.
$$
:::

:::{prf:proposition} Alignment-Triggered Step Damping
:label: prop-alignment-step-damping
Let $a_t := g_t^\top m_t$ and fix a damping factor $\rho \in (0,1]$. Define

$$
\eta_t^{+} :=
\begin{cases}
\rho\,\eta_t & \text{if } a_t < 0, \\
\eta_t & \text{otherwise}.
\end{cases}
$$
If $\eta_t$ satisfies Theorem {prf:ref}`thm-preconditioned-descent`, then the damped step with $\eta_t^{+}$ also
decreases $\mathcal{V}$.
:::

:::{prf:proposition} SNR-Gated Step Size
:label: prop-snr-gate
Under A1--A3 and the stochastic update $\theta_{t+1} = \theta_t - \eta_t M_t \hat g_t$, assume $g_t \ne 0$. The
expected Lyapunov change satisfies

$$
\mathbb{E}[\mathcal{V}(\theta_{t+1})\mid\theta_t]
\le \mathcal{V}(\theta_t)
- \eta_t m_{\min}\|g_t\|^2
+ \frac{L}{2}\eta_t^2 m_{\max}^2\left(\|g_t\|^2 + \sigma^2\right).
$$
Consequently, a sufficient condition for expected descent is

$$
\eta_t
\le
\frac{2 m_{\min}}{L m_{\max}^2\left(1 + \sigma^2 / \|g_t\|^2\right)}
= \frac{2 m_{\min}}{L m_{\max}^2}\cdot\frac{\mathrm{SNR}}{1+\mathrm{SNR}},
$$
with $\mathrm{SNR} := \|g_t\|^2 / \sigma^2$. If $g_t = 0$, the sufficient bound reduces to $\eta_t = 0$.
:::

:::{prf:definition} Log-LR Conduction Update
:label: def-log-lr-conduction
Let $\eta_i > 0$ be per-group learning rates and $x_i := \log(\eta_i)$. Define

$$
 x_i^{+} = x_i + \frac{k}{2}(x_{i-1} - 2 x_i + x_{i+1}),
$$
with Neumann boundary conditions $x_0 = x_1$, $x_{n+1} = x_n$ and conductivity $k \in [0,1]$.
:::

:::{prf:proposition} Conduction Contracts LR Disparities
:label: prop-conduction-contracts
Let $L$ be the path-graph Laplacian on $n$ groups and define the energy

$$
E(x) := \frac{1}{2} x^\top L x = \frac{1}{2}\sum_{i=1}^{n-1}(x_{i+1} - x_i)^2.
$$
The update in Definition {prf:ref}`def-log-lr-conduction` is gradient descent on $E$ with step size $k/2$, and for
$k \in [0,1]$ it satisfies $E(x^{+}) \le E(x)$.
:::

:::{prf:theorem} Thermodynamic Governor Stability (Conditional)
:label: thm-optimizer-conditional-stability
Under A1--A5, with updates that apply in the order: conduction on log learning rates, SNR/alignment gates,
the preconditioned step, and trust-region clipping last, and with $M_t$ block-diagonal across parameter groups,
assume:
1. preconditioned descent (Theorem {prf:ref}`thm-preconditioned-descent`),
2. trust-region scaling (Lemma {prf:ref}`lem-trust-region-scaling`),
3. alignment-triggered step damping (Proposition {prf:ref}`prop-alignment-step-damping`),
4. varentropy brake (Proposition {prf:ref}`prop-varentropy-brake-discrete`),
5. SNR gating (Proposition {prf:ref}`prop-snr-gate`), and
6. log-LR conduction (Proposition {prf:ref}`prop-conduction-contracts`),

the optimizer produces a nonincreasing Lyapunov objective in the deterministic case. The expected-descent claim
under noise applies only to the unscaled stochastic step in Proposition {prf:ref}`prop-snr-gate`; trust-region
clipping and alignment damping are covered there only in the deterministic regime. Learning rates remain positive
and coherent across adjacent groups, and the temperature schedule obeys the adiabatic constraint during annealing
phases with no Governor-initiated heating.
:::

:::{prf:remark}
:label: rem-agent-optimization-conditional
These guarantees are conditional and local. They ensure stability and controlled descent, not global optimality.
Violations of A1--A5 (e.g., non-smooth losses, unbounded noise, misordered groups) void the guarantees.
:::

## 04_control/01_exploration.md

:::{prf:definition} Macro Path Distribution
:label: def-macro-path-distribution

Fix a horizon $H\in\mathbb{N}$ and a (possibly stochastic) policy $\pi(a\mid k)$. The induced distribution over length-$H$ macro state-action trajectories

$$
\xi := (K_t, A_t, K_{t+1}, A_{t+1}, \dots, A_{t+H-1}, K_{t+H})
    \in \mathcal{K}\times(\mathcal{A}\times\mathcal{K})^H

$$
conditioned on $K_t=k$ is

$$
P_\pi(\xi\mid k)
:=
\prod_{h=0}^{H-1}\pi(A_{t+h}\mid K_{t+h})\ \bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h}).

$$
(For continuous $\mathcal{A}$, interpret $P_\pi(\xi\mid k)$ as a density with respect to the action reference measure.)

:::

:::{prf:definition} Causal Path Entropy
:label: def-causal-path-entropy

The causal path entropy at $(k,H)$ under $\pi$ is the cumulative policy entropy along paths
$\xi\in\Gamma_H(k)$ induced by $\pi$ and $\bar{P}$:

$$
S_c(k,H;\pi)
:= \sum_{h=0}^{H-1} \mathbb{E}_{\xi\sim P_\pi(\cdot\mid k)}
\left[ \mathcal H\!\left(\pi(\cdot\mid K_{t+h})\right) \right].

$$
Only policy randomness contributes; stochasticity in $\bar{P}$ does not add entropy credit.
The expectation is taken under the path law induced by $\pi$ and $\bar{P}$.
This quantity is well-typed because the macro register is discrete; for continuous $\mathcal{A}$, interpret
$\mathcal H(\pi(\cdot\mid k))$ as a differential entropy with respect to the action reference measure.

:::

:::{prf:definition} Exploration Gradient, metric form
:label: def-exploration-gradient-metric-form

Let $z_{\text{macro}}=e_k\in\mathbb{R}^{d_m}$ denote the code embedding of $k$ ({ref}`sec-the-shutter-as-a-vq-vae`), and let $G$ be the relevant metric on the macro chart ({ref}`sec-second-order-sensitivity-value-defines-a-local-metric`). Assume smooth policy and kernel heads $\pi_\theta(a\mid z)$ and $\bar P_\phi(k'\mid z,a)$, and let $\widetilde S_c(z,H;\pi)$ be the causal-entropy formula above with the initial code $k$ replaced by $z$ and these heads evaluated at $z$.

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,G(e_k)^{-1}\nabla_z\widetilde S_c(z,H;\pi)\big|_{z=e_k},

$$
where $T_c>0$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`). The straight-through VQ estimator transports this continuous gradient to the pre-quantization coordinates. In the strictly symbolic limit there is no tangent vector; use the separate preference ordering obtained by ranking $S_c(k,H;\pi)$ over $k$.

**Interpretation (Exploration / Reachability).** $S_c(k,H;\pi)$ measures how much action-level randomness the
agent injects along trajectories from $k$ under $\pi$. Increasing $S_c$ preserves **agent-controlled reachability**:
the policy avoids committing to a narrow action sequence, independent of environmental stochasticity.

:::

:::{prf:definition} MaxEnt RL objective on macrostates
:label: def-maxent-rl-objective-on-macrostates

Let $\mathcal{R}(k,a)$ be an instantaneous reward/cost-rate term ({ref}`sec-re-typing-standard-rl-primitives-as-interface-signals`, {ref}`sec-the-hjb-correspondence`) and let $\gamma\in(0,1)$ be the discount factor (dimensionless). The maximum-entropy objective is

$$
J_{T_c}(\pi)
:=
\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,K^{\text{act}}_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right],

$$
where $\mathcal{H}$ is Shannon entropy. This is the standard "utility + entropy regularization" objective.

**Regimes.**
- $T_c\to 0$: $\pi$ collapses toward determinism; behavior can be brittle under distribution shift.
- $T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat).
- The useful regime is intermediate: enough entropy to remain robust, enough utility to remain directed.

:::

:::{prf:proposition} Soft Bellman form, discrete actions
:label: prop-soft-bellman-form-discrete-actions

Assume finite $\mathcal{A}$. Define the soft state value

$$
V^*(k) := \max_{\pi} \ \mathbb{E}\Big[\sum_{t\ge 0}\gamma^t(\mathcal{R}+T_c\mathcal{H})\ \Big|\ K_0=k\Big].

$$
Then $V^*$ satisfies the entropic Bellman fixed point

$$
V^*(k)
=
T_c \log \sum_{a\in\mathcal{A}}
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right),

$$
and the corresponding optimal policy is the softmax policy

$$
\pi^*(a\mid k)\propto
\exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}[V^*(k')]\right)\right).

$$
:::

:::{prf:definition} Causal Path Space
:label: def-causal-path-space

For a macrostate $k\in\mathcal{K}$ and horizon $H$, define the future macro state-action path space

$$
\Gamma_H(k)
:=
\left\{(k_0,a_0,k_1,a_1,\dots,a_{H-1},k_H)\in\mathcal{K}^{H+1}\times\mathcal{A}^H : k_0 = k\right\}.

$$
:::

:::{prf:definition} Path Probability
:label: def-path-probability

$P_\pi(\xi\mid k)$ is the induced state-action path probability from {prf:ref}`def-macro-path-distribution`.

:::

:::{prf:definition} Causal Entropy
:label: def-causal-entropy

$S_c(k,H;\pi)$ is the causal path entropy from {prf:ref}`def-causal-path-entropy`, i.e., the cumulative policy
entropy along the induced path measure $P_\pi(\cdot\mid k)$.

:::

:::{prf:definition} Exploration gradient, covariant form
:label: def-exploration-gradient-covariant-form

On a macro chart with metric $G$ ({ref}`sec-second-order-sensitivity-value-defines-a-local-metric`),

$$
\mathbf{g}_{\text{expl}}(e_k) := T_c\,G(e_k)^{-1}\nabla_z\widetilde S_c(z,H;\pi)\big|_{z=e_k},

$$
:::

:::{prf:theorem} Finite-Horizon Equivalence for a Deterministic Macro Kernel
:label: thm-equivalence-of-entropy-regularized-control-forms-discrete-macro

Assume:
1. finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$,
2. a deterministic enclosure-consistent macro kernel $\bar{P}(k'\mid k,a)$,
3. bounded reward flux $\mathcal{R}(k,a)$,
4. a finite horizon $H$ and undiscounted objective ($\gamma=1$).

Then the following are equivalent characterizations of the same finite-horizon optimal control law:

1. **Finite-horizon MaxEnt control:** $\pi^*$ maximizes
   $\mathbb E_\pi[\sum_{h=0}^{H-1}(\mathcal R(K_{t+h},A_{t+h})+T_c\mathcal H(\pi(\cdot\mid K_{t+h})))]$ from the initial state $K_t=k$.
2. **Exponentially tilted trajectory measure (KL-regularization).** Fix a uniform reference (prior) policy $\pi_0(a\mid k)$. For the deterministic kernel, the length-$H$ optimal path law admits
   the exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar P$:

   $$
   P^*(\omega\mid K_t=k)\ \propto\
   P_0(\omega \mid k)\,
   \exp\!\left(\frac{1}{T_c}\sum_{h=0}^{H-1}\mathcal{R}(K_{t+h},A_{t+h})\right),

   $$
   where $P_0(\omega \mid k) := \prod_{h=0}^{H-1}\pi_0(A_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},A_{t+h})$ is the finite-horizon reference measure.
3. **Finite-horizon soft Bellman optimality:** with $V_H^*\equiv0$,

   $$
   V_h^*(k)=T_c\log\sum_{a\in\mathcal A}\exp\!\left(\frac{\mathcal R(k,a)+\mathbb E_{k'\sim\bar P(\cdot\mid k,a)}V_{h+1}^*(k')}{T_c}\right),
   $$

   and $\pi_h^*(a\mid k)$ is the corresponding softmax policy.

Moreover, for the kernel-consistent family $P=P_\pi$ and a uniform prior $\pi_0$, the link is the KL-regularized variational identity

$$
\log Z_H(k)
=
\sup_{\pi}
\left\{
\frac{1}{T_c}\,\mathbb{E}_{P_\pi}\!\left[\sum_{h=0}^{H-1}\mathcal{R}\right]
-D_{\mathrm{KL}}(P_\pi\Vert P_0)
\right\},

$$
and the optimizer is the policy-induced law in item 2. For uniform $\pi_0$,
$D_{\mathrm{KL}}(P_\pi\Vert P_0)=H\log|\mathcal A|-S_c(k,H;\pi)$, so this is maximization of expected reward plus $T_c$ times the causal path entropy. The normalization satisfies
$T_c\log Z_H(k)=V_0^*(k)-T_cH\log|\mathcal A|$.

For stochastic kernels or discounted objectives, this equivalence does not hold in this form. One must either
restrict the admissible laws to $P=\pi\cdot\bar P$ and use the discounted per-step policy KL, or treat the full
path tilt as a distinct risk-sensitive control problem.

:::

## 04_control/02_belief_dynamics.md

:::{prf:definition} Belief operator
:label: def-belief-operator

Let $d=|\mathcal K|$ and let $\varrho_t\in\mathbb{C}^{d\times d}$ satisfy $\varrho_t\succeq 0$ and $\mathrm{Tr}(\varrho_t)=1$. Diagonal $\varrho_t$ in the macro basis reduces to a classical probability vector; non-diagonal terms can be used to encode correlations/uncertainty structure in a learned feature basis.

:::

:::{prf:definition} GKSL generator
:label: def-gksl-generator

A time-homogeneous, norm-continuous CPTP semigroup on $\mathbb C^{d\times d}$ has a generator of the Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) form {cite}`gorini1976completely,lindblad1976generators`:

$$
\frac{d\varrho}{dt}
=
\underbrace{-i[H,\varrho]}_{\text{conservative drift}}
\;+\;
\underbrace{\sum_{j} \gamma_j\left(L_j\varrho L_j^\dagger-\frac12\{L_j^\dagger L_j,\varrho\}\right)}_{\text{dissipative update}},

$$
where {math}`H=H^\dagger` is Hermitian, {math}`\gamma_j\ge 0` are rates per interaction time, and {math}`\{L_j\}` are (learned) operators.

**Operational interpretation (within this document).**
- The commutator term is a structured way to represent **reversible internal prediction** (it preserves $\mathrm{Tr}(\varrho)$ and the spectrum of $\varrho$).
- The dissipator is a structured way to represent **irreversible disturbance / decoherence** while preserving positivity and trace.

This is a modeling choice, not a claim about literal quantum physics: it is used here purely as a convenient, well-posed parametrization of CPTP belief updates.

*Note (WFR Correspondence).* If $H$ is diagonal in the macro basis and the $L_j$ are jump operators
$|j\rangle\langle k|$, diagonal states are invariant and the GKSL equation reduces to a classical master
equation with rates $W_{jk}$. If, in addition, $W$ satisfies detailed balance with respect to a stationary
law $\pi$, the resulting chain is a gradient flow of relative entropy in the discrete transport metric of
{cite}`maas2011gradient,mielke2011gradient`. Identifying that metric with the full WFR action
({prf:ref}`def-the-wfr-action`) requires a separate metric comparison. For diagonal $\varrho$, the commutator
vanishes only under the stated diagonal-$H$ hypothesis; otherwise it generates coherences.

:::

## 04_control/03_coupling_window.md

:::{prf:definition} Grounding rate
:label: def-grounding-rate

Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). For a time window or minibatch, the *grounding rate* is the corresponding average information inflow per step:

$$
\lambda_{\text{in}} := \mathbb{E}[G_t].

$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

:::

:::{prf:definition} Mixing rate
:label: def-mixing-rate

Let $p_t\in\Delta^{|\mathcal K|-1}$ be the macro posterior from the belief update and set $S_t:=H(p_t)$. The *mixing rate* is the average positive growth of posterior uncertainty:

$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].

$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

:::

:::{prf:definition} Information-stability window; operational
:label: thm-information-stability-window-operational

Choose information and posterior-dispersion margins $\epsilon_I>0$ and $\epsilon_H>0$. The *operational coupling window* is the set of time windows for which

$$
\epsilon_I \le I(X_t;K_t) \quad\text{and}\quad H(p_t)\le \log|\mathcal{K}|-\epsilon_H,

$$
and the net entropy balance satisfies

$$
\lambda_{\text{in}}\ge \lambda_{\text{mix}}-\delta.

$$
Violations correspond to identifiable barrier modes:
- If $I(X;K)\approx 0$: under-coupling - ungrounded inference / decoupling (Mode D.C).
- If $H(p_t)\approx \log|\mathcal{K}|$: over-aggressive updating or posterior dispersion (BarrierScat).

*Remark.* This definition is stated at the level of measurable information quantities so it can be audited online. It is an operating criterion, not a sufficiency theorem for stability; such a theorem would require a specified macro-kernel class and a contraction inequality (for example, a log-Sobolev or Doeblin condition).

:::

:::{prf:proposition} Information upper bound for the grounding margin
:label: prop-grounding-information-upper-bound

For a discrete macro register,

$$
I(X_t;K_t)=H(K_t)-H(K_t\mid X_t)\le H(K_t).
$$

Consequently, the lower window condition $I(X_t;K_t)\ge\epsilon_I$ implies
$H(K_t)\ge\epsilon_I$ and $H(K_t\mid X_t)\le H(K_t)-\epsilon_I$. This implication concerns the marginal code-usage entropy $H(K_t)$; it does not identify that entropy with the posterior entropy $H(p_t)$ used in the dispersion condition.
:::

## 05_geometry/01_metric_law.md

:::{prf:definition} DPI / boundary-capacity constraint
:label: def-dpi-boundary-capacity-constraint

Consider the boundary stream $(X_t)_{t\ge 0}$ and the induced internal state process $(Z_t)_{t\ge 0}$ produced by the shutter (Definition {prf:ref}`def-bounded-rationality-controller`). Because all internal state is computed from boundary influx and internal memory, any information in the bulk must be mediated by a finite-capacity channel. Operationally, the data-processing constraint is:

$$
I_{\text{bulk}} \;\le\; C_{\partial},

$$
where $C_{\partial}$ is the effective information capacity of the boundary channel and $I_{\text{bulk}}$ is a declared grounded-information proxy for the internal state. The inequality is an operational capacity postulate; the data-processing inequality supplies it only after $I_{\text{bulk}}$ has been defined as mutual information with a boundary-history variable. Units are nats for a fixed observation window and nat/step for the corresponding rate.

:::

:::{prf:definition} Information density and bulk information volume
:label: def-information-density-and-bulk-information-volume

Let $\rho(z,s)$ denote the probability density of the agent's belief state at position $z \in \mathcal{Z}$ and computation time $s$, **defined with respect to the Riemannian volume measure** $d\mu_G = \sqrt{|G|}\,dz^n$. Let $\rho_{\mathrm{ref}}(z,s)>0$ be a reference belief density with respect to the same measure. The **relative-information density** is

$$
\iota_{\mathrm{bulk}}(z,s) := \rho(z,s)\log\frac{\rho(z,s)}{\rho_{\mathrm{ref}}(z,s)},

$$
with units of nats per unit Riemannian volume ($n=\dim\mathcal{Z}$). The local integrand need not be non-negative, but its integral is a KL divergence.

*Remark.* The differential entropy $h_G[\rho]:=-\int \rho\log\rho\,d\mu_G$ is a separate coordinate-invariant quantity. The relative-information integral below is the non-negative proxy used for the capacity diagnostic.

:::

:::{prf:definition} Bulk Information Volume
:label: def-a-bulk-information-volume

Define the bulk information volume over a region $\Omega\subseteq\mathcal{Z}$ by

$$
I_{\text{bulk}}(\Omega) := \int_{\Omega} \iota_{\mathrm{bulk}}(z,s)\, d\mu_G
 = D_{\mathrm{KL}}\!\left(\rho\,\middle\|\,\rho_{\mathrm{ref}}\right)_{\Omega}.

$$
When $\Omega=\mathcal{Z}$ we write $I_{\text{bulk}}:=I_{\text{bulk}}(\mathcal{Z})$. This is conceptually distinct from the probability-mass balance in {ref}`Section 2.11 <sec-variance-value-duality-and-information-conservation>`; it measures deviation from the declared reference belief in nats.

:::

:::{prf:definition} Boundary capacity: area law at finite resolution
:label: def-boundary-capacity-area-law-at-finite-resolution

Let $\partial_{\varepsilon}\mathcal{Z}$ be an explicitly chosen cutoff hypersurface (for example $|z|=1-\varepsilon$ in the Poincaré chart), and let $dA_G$ be its induced $(n-1)$-dimensional area form. If the boundary interface has a minimal resolvable scale $\ell>0$ (pixel/token floor), then an operational capacity model is

$$
C_{\partial}(\partial_{\varepsilon}\mathcal{Z})
:=
\frac{1}{\eta_\ell}\oint_{\partial_{\varepsilon}\mathcal{Z}} dA_G,

$$
where $\eta_\ell$ is the effective boundary area-per-nat at resolution $\ell$ (a resolution-dependent constant set by the interface). The capacity is therefore cutoff-dependent; if no geometric cutoff is declared, use the interface-channel definition instead of an area integral.
Units: $[\eta_\ell]=[dA_G]/\mathrm{nat}$ and $[\ell]$ is the chosen boundary resolution length scale.

*Remark (discrete macro specialization).* For the split shutter, use distinct rate proxies over a declared observation window:

$$
C_{\partial}^{\mathrm{rate}}\ :=\ \sup_{p(x)} I(X;K)\ \le\ \log|\mathcal{K}|,

$$
while the realised inflow is $\lambda_{\mathrm{in}}=\mathbb{E}[I(X_t;K_t)]$ in nat/step (Definition {prf:ref}`def-grounding-rate`). Multiplying the rate by a window length converts it to a stock in nats; Node 13 monitors the realised inflow rather than identifying it with channel capacity.

:::

:::{prf:theorem} Capacity-constrained metric law
:label: thm-capacity-constrained-metric-law

Under the regularity and boundary-clamping hypotheses stated in {ref}`Appendix A <sec-appendix-a-full-derivations>`, stationarity of the curvature-plus-risk functional implies

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
where $\Lambda$ and $\kappa$ are constants and $T_{ij}$ is the **total Risk Tensor** induced by the reward field. The equation is consistent with the value equation only when the value and (if present) curl fields are on shell for the same action; this is an explicit hypothesis, not a consequence of the boundary capacity. *Units:* $\Lambda$ has the same units as curvature ($[R]\sim [z]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* The equality supplies a geometric consistency residual. Interpreting it as a mechanism that enforces $I_{\text{bulk}}\le C_{\partial}$ requires the separate capacity postulate (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`) and is not derived by this variation.

**Implementation hook.** The squared residual of this identity defines the metric-law regularizer $\mathcal{L}_{\text{EFE}}$; use the metric-contracted norm specified in {ref}`Appendix F <sec-appendix-f-loss-terms-reference>`.

:::

:::{prf:definition} Extended Risk Tensor with Maxwell Stress
:label: def-extended-risk-tensor

The total Risk Tensor $T_{ij}$ decomposes into gradient and curl contributions:

$$
T_{ij} = T_{ij}^{\text{gradient}} + T_{ij}^{\text{Maxwell}},

$$
where:

1. **Gradient Stress** (from scalar potential $\Phi$):

$$
T_{ij}^{\text{gradient}} = \partial_i \Phi \, \partial_j \Phi - G_{ij}\left(\frac{1}{2}\|\nabla\Phi\|_G^2+U(\Phi)\right)

$$
2. **Maxwell Stress** (from {prf:ref}`def-value-curl` $\mathcal{F}$):

$$
T_{ij}^{\text{Maxwell}} = \alpha_{\mathcal F}\left(\mathcal{F}_{ik}\mathcal{F}_j^{\;k} - \frac{1}{4}G_{ij}\mathcal{F}^{kl}\mathcal{F}_{kl}\right)

$$
*Units:* $[T_{ij}^{\text{gradient}}] = \mathrm{nat}^2/[z]^2$ when $U$ has the same units as the gradient term. Under the volume's convention $[\mathcal F]=\mathrm{nat}/[z]^2$, choose $[\alpha_{\mathcal F}]=[z]^2$ so the Maxwell contribution has the same units. This is an optional action extension; the scalar derivation in Appendix A does not include it.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Maxwell term vanishes and we recover the standard gradient-only risk tensor.

**Non-Conservative Case:** When $\mathcal{F} \neq 0$, the Maxwell stress contributes additional terms to the curvature equation.

:::

:::{prf:definition} Capacity saturation diagnostic
:label: def-capacity-saturation-diagnostic

Compute the capacity saturation ratio:

$$
\nu_{\text{cap}}(s) := \frac{I_{\text{bulk}}(s)}{C_{\partial}},

$$
where $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \iota_{\mathrm{bulk}}(z,s)\, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; the agent may be compressing excessively (lossy representation).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the declared capacity postulate (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`); this is a diagnostic flag, not a direct consequence of the DPI unless $I_{\text{bulk}}$ is a mutual information.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature residual (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) is insufficient. This triggers a representation reflow---reduce the KL-to-reference proxy through coarsening or stronger regularization, then re-evaluate the metric residual.

:::

## 05_geometry/02_wfr_geometry.md

:::{prf:definition} The Generalized WFR Action
:label: def-the-wfr-action

The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ is the infimum of the undriven generalized energy functional:

$$
\mathcal{E}_{\mathrm{WFR}}[\rho, v, r] = \int_0^1 \int_{\mathcal{Z}} \left( \underbrace{\|v_s(z)\|_G^2}_{\text{Transport Cost}} + \underbrace{\lambda^2 |r_s(z)|^2}_{\text{Reaction Cost}} \right) d\rho_s(z) \, ds

$$
subject to the **Unbalanced Continuity Equation**:

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r

$$
where:
- $v_s(z) \in T_z\mathcal{Z}$ is the **velocity field** (transport/flow)
- $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass)
- $\lambda > 0$ is the **length-scale parameter** balancing transport and reaction
- $G$ is the Riemannian metric on the continuous fibres ({ref}`Section 2.5 <sec-second-order-sensitivity-value-defines-a-local-metric>`)
**Conservative and driven cases.** The displayed distance contains no vector potential. A non-conservative reward field is represented by the separately defined driven action
$\mathcal S_{\mathbf A}:=\mathcal E_{\mathrm{WFR}}-2\beta_{\mathrm{curl}}\int\!\langle\mathbf A,v\rangle\,d\rho\,ds$; it is a control objective, not a squared distance. Only the curvature $\mathcal F=d\mathbf A$ is gauge invariant. If $H^1_{\mathrm{dR}}(\mathcal Z)=0$ (or the harmonic component is set to zero) and $\mathcal F=0$, a gauge with $\mathbf A=0$ may be chosen; otherwise a harmonic component remains.

The driven action is not claimed to be gauge invariant or to yield the second-order Lorentz--Langevin SDE without an additional reduction and noise model. The SDE is defined independently in Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Forward reference (Boundary Conditions).* {ref}`Section 23.5 <sec-wfr-boundary-conditions-waking-vs-dreaming>` specifies how boundary conditions on $\partial\mathcal{Z}$ (sensory and motor boundaries) constrain the WFR dynamics: **Waking** imposes Dirichlet (sensors) + Neumann (motors) BCs; **Dreaming** imposes reflective BCs on both, enabling recirculating flow without external input.

:::

:::{prf:remark} Units
:label: rem-units

$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$ after taking the metric coordinates as length units. The ratio $\|v\|/(\lambda |r|)$ is a local cost ratio; the exact crossover for a pair of Dirac masses depends on the normalization of the Hellinger term.

:::

:::{prf:definition} Canonical length-scale
:label: def-canonical-length-scale

Let $G_n(\cdot;K)$ be the metric on the continuous fibre of chart $K$. If the
interior fibres have a positive uniform injectivity-radius lower bound, a
geometric calibration is

$$
\lambda_{\mathrm{inj}} := \inf_{K}\inf_{z\in \mathcal Z_{n,K}^{\circ}}
\operatorname{inj}_{G_n(\cdot;K)}(z),

$$
where the infimum is taken over the chosen chart interiors. The discrete
factor has no exponential map, and on an untrimmed open chart the infimum may
be zero; in either case use a calibrated positive hyperparameter instead of
calling this quantity canonical.

*Default value.* If the injectivity radius is unknown or the metric is learned, a practical default is:

$$
\lambda_{\text{default}} := \ell_{\mathrm{step}}\sqrt{\frac{1}{n}\operatorname{tr}\!\left(\bar G_n\right)},

$$
where $\ell_{\mathrm{step}}$ is a declared coordinate step and $\bar G_n$ is
the metric averaged over the sampled chart interiors. This is a calibration
heuristic, not an intrinsic injectivity-radius identity.

*Cross-reference:* The screening length $\ell_{\text{screen}} =
1/\kappa_{\mathrm{scr}}$ from {ref}`Section 24.2
<sec-the-bulk-potential-screened-poisson-equation>` is a spatial
discount-induced/value-field scale. It is distinct from the WFR reaction
length $\lambda$ (and from the temporal discount rate used to define
$\kappa_{\mathrm{scr}}$).

:::

:::{prf:proposition} Limiting Regimes
:label: prop-limiting-regimes

The WFR metric seamlessly unifies discrete and continuous dynamics:

1. **Continuous Movement (Flow):** When moving within a chart, $r \approx 0$. The dynamics are dominated by $\nabla \cdot (\rho v)$, and the metric reduces to $W_2$ (Wasserstein-2). This recovers the Riemannian manifold structure of the nuisance fibres.

2. **Discrete Movement (Jump):** When the flow reaches a topological obstruction (chart boundary without overlap), transport can become prohibitively expensive. It can then be cheaper to use the source term $r$:
   - $r < 0$ on the old chart (mass destruction)
   - $r > 0$ on the new chart (mass creation)
   On the full non-negative-measure cone this gives the Hellinger/Fisher--Rao-type pure-reaction metric; a Fisher--Rao simplex is obtained only after restricting to a finite normalized chart register.

3. **Mixed Regime (Overlap):** In chart overlaps, both $v$ and $r$ are active. The optimal path smoothly interpolates between transport and reaction.

:::

:::{prf:theorem} Classical Master Equation as a Discrete-Wasserstein Gradient Flow
:label: thm-classical-master-equation-wfr

Let $\mathcal{K} = \{1, \ldots, K\}$ be a finite state space with transition rates $W_{jk} \geq 0$ (rate of jumping from $k$ to $j$). The classical master equation

$$
\dot{p}_j = \sum_{k} W_{jk} p_k - W_{kj} p_j
$$

If the chain is reversible with respect to a strictly positive stationary distribution $\pi$ (detailed balance $W_{jk}\pi_k=W_{kj}\pi_j$), this is the **gradient flow** of the relative entropy $H(p \| \pi) = \sum_j p_j \log(p_j / \pi_j)$ with respect to a discrete Wasserstein-type metric. It is not, by this statement alone, a WFR reaction flow {cite}`maas2011gradient,mielke2011gradient,chow2012fokker`.

:::

:::{prf:corollary} GKSL Classical Limit
:label: cor-gksl-classical-limit

Assume that $\varrho = \mathrm{diag}(p_1, \ldots, p_K)$, that $H$ is diagonal in the same basis, and that the jump operators preserve diagonal matrices (for example, they are linear combinations of matrix units). Then the GKSL equation reduces to a classical master equation with rates

$$
W_{jk} = \sum_\ell \gamma_\ell |\langle j | L_\ell | k \rangle|^2.
$$

The commutator term vanishes under the diagonal-$H$ hypothesis. If the resulting rates satisfy detailed balance, Theorem {prf:ref}`thm-classical-master-equation-wfr` identifies the evolution as a gradient flow in the discrete-Wasserstein geometry.

:::

:::{prf:remark} Full Quantum Case
:label: rem-full-quantum-wfr

For non-diagonal density matrices (quantum coherences), the appropriate geometric structure is the **quantum Wasserstein distance** of Carlen \& Maas {cite}`carlen2014wasserstein,carlen2017gradient`. The GKSL equation is the gradient flow of quantum relative entropy with respect to this metric. This framework handles coherences but is more complex than the classical WFR theory used here.

:::

:::{prf:definition} WFR World Model
:label: def-wfr-world-model

The action-conditioned world model outputs a generalized velocity field $(v,r)$; the policy selects actions, and a separate planner or loss can train it to minimize WFR path length to a target distribution.

```python
import torch
import torch.nn as nn
from typing import Tuple

class WFRWorldModel(nn.Module):
    """
    Unified World Model using Unbalanced Optimal Transport dynamics.

    Predicts the 'Generalized Velocity' (v, r) for belief particles.
    No separate 'discrete' and 'continuous' modules.
    """

    def __init__(
        self,
        macro_embed_dim: int,
        nuisance_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        # Input: particle state + action
        # State includes: macro embedding, nuisance coords, mass (weight)
        input_dim = macro_embed_dim + nuisance_dim + 1 + action_dim

        # Single MLP backbone for unified dynamics
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Head 1: Transport velocity (Riemannian motion on fibre)
        self.head_v = nn.Linear(hidden_dim, nuisance_dim)

        # Head 2: Reaction rate (Fisher-Rao mass creation/destruction)
        self.head_r = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        z_t: torch.Tensor,           # [B, D] latent state (macro_embed + nuisance)
        mass_t: torch.Tensor,        # [B, 1] particle weight (belief mass)
        action_t: torch.Tensor,      # [B, A] action
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next state via WFR dynamics.

        Returns:
            z_next: [B, D] next latent state
            mass_next: [B, 1] next particle mass
            v_t: [B, nuisance_dim] transport velocity
            r_t: [B, 1] reaction rate
        """
        # Unified prediction
        inp = torch.cat([z_t, mass_t, action_t], dim=-1)
        feat = self.dynamics_net(inp)

        v_t = self.head_v(feat)  # Transport velocity
        r_t = self.head_r(feat)  # Reaction rate (log-growth)

        # Integrate dynamics (Euler step)
        # Position update (Transport): z' = z + v * dt
        z_next = z_t.clone()
        z_next[..., -self.head_v.out_features:] += v_t * dt

        # Mass update (Reaction): m' = m * exp(r * dt)
        # If r > 0: hypothesis gaining probability (jumping in)
        # If r < 0: hypothesis losing probability (jumping out)
        mass_next = mass_t * torch.exp(r_t * dt)

        return z_next, mass_next, v_t, r_t
```

**How this handles the "Jump" seamlessly:**

- **Deep inside a Chart:** Model predicts $r \approx 0$ and $v \neq 0$. Particle moves normally.
- **Approaching a Boundary:** Model sees invalid description (high prediction error). Predicts $r < 0$ for current chart, $r > 0$ for neighboring chart particles.
- **Result:** Probability mass smoothly "tunnels" between charts without hard discrete switching.

:::

:::{prf:definition} Scale-Dependent Teleportation Cost
:label: def-scale-dependent-teleportation-cost

$$
\lambda^{(\ell)} \propto \Pi^{(\ell)}:=\prod_{j<\ell}\sigma^{(j)} \quad \text{(cumulative residual scale)}

$$
where $\sigma^{(\ell)}$ is the scale factor from Definition {prf:ref}`def-the-rescaling-operator-renormalization`.

**Interpretation:**
- **Layer 0 (Bulk / IR):** $\Pi^{(0)}=1$ sets the reference scale. Jumping is expensive relative to later residual scales when the cumulative factors decrease.
- **Layer $L$ (Texture / UV):** the cumulative factor $\Pi^{(L)}$ carries the absolute residual scale; a per-layer ordering must be checked from the measured factors rather than assumed from a single $\sigma^{(\ell)}$.

**Correspondence with Cosmological Constant:**
If the capacity-constrained metric law is applied separately at each layer, the term $\Lambda^{(\ell)}G_{ij}$ plays the role of a baseline curvature. The correspondence is:

$$
\Lambda^{(\ell)} \sim \frac{1}{(\lambda^{(\ell)})^2}

$$
- Bulk (low $\Lambda$): Flat, rigid, transport-dominated
- Boundary (high $\Lambda$): Curved, fluid, reaction-dominated

:::

:::{prf:theorem} Auxiliary WFR variational stress tensor
:label: thm-wfr-stress-energy-tensor-variational-form

Let the WFR action be

$$
\mathcal{S}_{\mathrm{WFR}}
=
\frac12\int_0^T\int_{\mathcal{Z}}
\rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,

$$
with continuity equation

$$
\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r.

$$
Define, under the explicit density-fixed convention,

$$
T^{\mathrm{WFR}}_{ij}:=
-\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}}
\quad\text{(holding }\rho,v,r\text{ fixed).}

$$
Then

$$
T^{\mathrm{WFR}}_{ij}=\rho\,v_i v_j + P\,G_{ij},
\qquad
P=\frac12\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right),

$$
which is a perfect-fluid form under this density-fixed convention, with reaction contributing an additive pressure term
{math}`P_{\mathrm{react}}=\tfrac12\lambda^2\rho r^2`.

:::

:::{prf:definition} WFR Consistency Loss / WFRCheck
:label: def-wfr-consistency-loss-wfrcheck

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:

$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2

$$
This penalizes deviations from the unbalanced continuity equation.

**Practical implementation:**

```python
def compute_wfr_consistency_loss(
    rho_t: torch.Tensor,       # [B, K] belief over charts at time t
    rho_t1: torch.Tensor,      # [B, K] belief over charts at time t+1
    v_t: torch.Tensor,         # [B, K, d_n] transport velocity per chart
    r_t: torch.Tensor,         # [B, K] reaction rate per chart
    dt: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute WFR consistency loss (cone-space formulation).

    Penalizes violation of unbalanced continuity equation.
    """
    sqrt_rho_t = torch.sqrt(rho_t + eps)
    sqrt_rho_t1 = torch.sqrt(rho_t1 + eps)

    # Approximate divergence term (finite difference)
    # In practice, use automatic differentiation if v is differentiable
    div_rho_v = torch.zeros_like(rho_t)  # Placeholder for nabla . (rho v)

    # Predicted change in sqrt(rho) from: d/ds sqrt(rho) = (rho*r - div(rho*v)) / (2*sqrt(rho))
    predicted_delta = (dt / (2 * sqrt_rho_t + eps)) * (rho_t * r_t - div_rho_v)

    # Actual change
    actual_delta = sqrt_rho_t1 - sqrt_rho_t

    # L2 loss
    loss = ((actual_delta - predicted_delta) ** 2).mean()

    return loss
```

:::

## 05_geometry/03_holographic_gen.md

:::{prf:definition} Manifold Boundary and Interior
:label: def-manifold-boundary-and-interior

Let $\mathcal{Z}\cong\mathbb B^D$ be the latent manifold. The **ideal boundary** is the $(D-1)$-dimensional limit set:

$$
\partial\mathcal{Z} := \{z \in \mathbb{R}^D : |z| = 1\}=S^{D-1}.

$$
The **interior** (or bulk) is the open disk:

$$
\text{int}(\mathcal{Z}) := \{z \in \mathbb{R}^D : |z| < 1\}.

$$
These are standard differential geometry terms; the boundary is the ideal boundary at infinity in the hyperbolic metric.

:::

:::{prf:definition} Hyperbolic Volume Growth
:label: def-hyperbolic-volume-growth

For the two-dimensional Poincaré disk ($D=2$) with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the volume of a hyperbolic ball $B_r(0)$ grows exponentially:

$$
\mathrm{Vol}(B_r(0)) = 4\pi \sinh^2\!\left(\frac{r}{2}\right) \;\approx\; \pi e^r \quad \text{as } r \to \infty.

$$
For general $D$, $\mathrm{Vol}(B_r)=\mathrm{Vol}(S^{D-1})\int_0^r\sinh^{D-1}(s)\,ds\sim e^{(D-1)r}$; the displayed formula and the later $r(\tau)$ calculation are the $D=2$ case.

:::

:::{prf:definition} The Entropic Force
:label: def-the-entropic-force

The "Free Energy" of a state at radius $r$ is dominated by the entropic volume term $S(r) = 2 \operatorname{artanh}(r)$. To maximize entropy (fill the capacity), the agent experiences a radial force:

$$
F_{\text{entropy}}(z) = \nabla_G S(z) = \frac{(1-|z|^2)}{2} \cdot \frac{z}{|z|}

$$
This accounts for the Poincaré metric conformal factor. The drift magnitude decreases near the boundary ($|z| \to 1$), ensuring the agent asymptotically approaches but never reaches it.

Units: $[F_{\text{entropy}}] = [z]/\tau$.

:::

:::{prf:proposition} Isotropic Radial Expansion
:label: prop-isotropic-radial-expansion

If acting alone (no policy steering), the entropic drift produces the isotropic expansion:

$$
r(\tau) = \tanh(\tau/2)

$$
This represents isotropic diffusion---expanding uniformly in all directions.

:::

:::{prf:definition} Hyperbolic Information Potential
:label: def-hyperbolic-information-potential

The **information potential** $U: \mathbb{D} \to \mathbb{R}$ is the negative hyperbolic distance from the origin:

$$
U(z) := -d_{\mathbb{D}}(0, z) = -2 \operatorname{artanh}(|z|) = -\log\!\left(\frac{1+|z|}{1-|z|}\right).

$$
Units: $[U] = \mathrm{nat}$.

*Remark (Thermodynamic Interpretation).* The log-volume $S=-U$ is minimal at the origin and increases toward the ideal boundary. Thus at $z=0$, $U=0$ is the maximum potential and the state has minimal committed information; as $|z|\to1$, $U\to-\infty$ and the information depth $-U(z)$ diverges.

:::

:::{prf:proposition} Riemannian Gradient of $U$
:label: prop-riemannian-gradient-of

The gradient in the Poincaré metric is:

$$
\nabla_G U(z) = G^{-1} \nabla U = -\frac{(1-|z|^2)}{2} \hat{z}, \quad \text{where } \hat{z} = \frac{z}{|z|}.

$$
The **entropic drift** (negative gradient) pushes radially outward:

$$
-\nabla_G U(z) = \frac{(1-|z|^2)}{2} \hat{z}.

$$
*Remark (Connection to {ref}`Section 7.11 <sec-the-geometry-of-the-latent-space-a-hyperbolic-hierarchy>`).* The Poincare coordinate $z$ relates to depth via $\rho = d_{\mathbb{D}}(0, z) = 2\operatorname{artanh}(|z|)$. Chart transitions are handled by the WFR jump process ({ref}`Section 22.2 <sec-the-coupled-jump-diffusion-sde>`), governed by the {prf:ref}`def-the-wfr-action`.

**Cross-references:** Definition {prf:ref}`def-information-density-and-bulk-information-volume`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

:::{prf:proposition} SO(D) Symmetry at Origin
:label: prop-so-d-symmetry-at-origin

At $z = 0$:
1. The metric is isotropic: $G(0) = 4I$
2. The entropic field has no preferred direction at the origin; its radial magnitude has the limit $\lim_{r\downarrow0}|F_{\text{entropy}}|=1/2$.
3. The system has full rotational symmetry $SO(D)$

*Orbit calculation:* Every rotation fixes the zero vector, so its stabilizer is $SO(D)$ and its orbit is a point. The separately defined scalar vacuum has the representation-dependent mass matrix of {prf:ref}`thm-higgs-mechanism`.

:::

:::{prf:definition} The Control Field
:label: def-the-control-field

The Policy $\pi_\theta(a|z)$ outputs actions in $\mathcal A$. After a declared smooth action-to-tangent map $B(z):\mathcal A\to T_z\mathcal Z$, its mean control is a vector field

$$
u_\pi(z) := B(z)\,\mathbb{E}_{a\sim\pi_\theta(\cdot|z)}[a].

$$
This vector field represents the **Information Preference** of the agent (or the User).

Units: $[u_\pi] = [z]/\tau$.

*Remark (Context-Conditioning).* {ref}`sec-the-context-space-unified-definition` generalizes this to **context-conditioned policies** $\pi(a|z,c)$ where the context $c \in \mathcal{C}$ unifies RL action spaces, classification label spaces, and LLM prompt spaces. For a cost potential, a compatible deterministic drift is $u_\pi(z,c)=-G^{-1}(z)\nabla_z\Phi_{\text{eff}}(z,K,c)$; a learned action policy need not be a gradient field.

:::

:::{prf:definition} Control Field at Origin
:label: def-control-field-at-origin

At $\tau=0$, the total drift is:

$$
F_{\text{total}} = F_{\text{entropy}} + u_\pi(0)

$$
The entropic field selects no direction at $z=0$, so the initial *direction* is determined by the policy (or by noise); its radial magnitude is not zero.

:::

:::{prf:theorem} Unified Control Interpretation
:label: thm-unified-control-interpretation

The control field $u_\pi$ has three distinct operating modes:

| **Mode**                     | **Control Field $u_\pi$**                          | **Interpretation**                         |
|------------------------------|----------------------------------------------------|--------------------------------------------|
| **RL**                       | $u_\pi = -G^{-1} \nabla_z V_{\text{critic}}$        | Descends the cost-to-go                   |
| **Conditioned Generation**   | $u_\pi = G^{-1} \cdot \text{embed}(\text{prompt})$ | Clamped to user's prompt embedding         |
| **Unconditional (Dreaming)** | $u_\pi = 0$                                        | Pure thermal fluctuation selects direction |

*Scope.* Each row is a separate parameterization of a tangent drift. The policy-gradient theorem optimizes action parameters; it does not by itself identify $\nabla_zV$ with the policy output.

:::

:::{prf:proposition} Radial-angular SDE under the $D=2$ overdamped specialization {cite}`strogatz2015nonlinear`
:label: thm-angular-symmetry-breaking

Assume $D=2$, $\alpha=1$, $\gamma_{\mathrm{risk}}=0$, $\mathcal F=0$, and absorb the constant friction into the time unit. In the **overdamped limit** of the second-order geodesic Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`, Theorem {prf:ref}`thm-overdamped-limit`), the generation dynamics decompose into radial expansion and angular diffusion.

**Radial dynamics (monotonic expansion):** The radial coordinate $r = |z|$ satisfies:

$$
dr = \left[\frac{1-r^2}{2}+u_\pi^r+\frac{T_c(1-r^2)^2}{4r}\right]d\tau + \frac{1-r^2}{2}\sqrt{2T_c}\,dW_r

$$
with drift $\frac{1-r^2}{2} > 0$ for all $r \in [0,1)$. The origin is not a fixed point; the drift pushes trajectories outward (though stochastic fluctuations can temporarily reverse this at small $r$).

**Angular dynamics (symmetry breaking):** In polar coordinates $z = re^{i\theta}$, the angular evolution satisfies:

$$
d\theta = \frac{u_\pi^\theta}{r}\,d\tau + \frac{1-r^2}{2r}\sqrt{2T_c}\,dW_\theta

$$
where $u_\pi^\theta = u_\pi \cdot \hat{\theta}$ is the tangential component of the control field.

**Local direction-to-noise ratio:** A dimensionless local measure of angular drift relative to diffusion is

$$
\mathrm{Pe}_\theta^2(r) := \frac{2r^2|u_\pi^\theta|^2}{T_c(1-r^2)^2}.

$$
- If $\mathrm{Pe}_\theta\ll1$, angular diffusion dominates over the chosen time interval.
- If $\mathrm{Pe}_\theta\gg1$, the policy drift dominates locally.

:::

:::{prf:axiom} Bulk-Boundary Decoupling
:label: ax-bulk-boundary-decoupling

The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**. Write $\mathcal Z_{\mathrm{bulk}}:=\mathcal K\times\mathcal Z_n\subset\mathcal Z$ for the texture-free projection:

1. **Interior (Planning Domain):** The bulk projection evolves on $\mathcal Z_{\mathrm{bulk}}$ and contains no texture component. Planning depends only on geometry and topology:

$$
\Pi_{\mathrm{bulk}}\dot Z = f(\Pi_{\mathrm{bulk}}Z,u_\pi),\qquad \partial_{z_{\mathrm{tex}}}\Pi_{\mathrm{bulk}}\dot Z=0.

$$
2. **Boundary Interface:** Texture $z_{\text{tex}}$ is a stochastic component that exists **only** at the interface where the internal state meets the external observation:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma(z_{\text{final}}))

$$
Formally, the partition condition is:

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}^k, \lambda_{\text{jump}}, u_\pi \right] = 0 \quad \forall \tau \in [0, \tau_{\text{stop}})

$$
:::

:::{prf:definition} Boundary Texture Distribution
:label: def-boundary-texture-distribution

At the terminal position $z_{\text{final}}$, texture is sampled from a **geometry-dependent** Gaussian:

$$
z_{\text{tex}} \sim \mathcal{N}\big(0,\, \Sigma(z_{\text{final}})\big),

$$
where the covariance matrix is:

$$
\Sigma(z) = \sigma_{\text{tex}}^2 \cdot G^{-1}(z) = \sigma_{\text{tex}}^2 \cdot \frac{(1-|z|^2)^2}{4} I.

$$
Units: $[\Sigma] = [z_{\text{tex}}]^2$.

:::

:::{prf:proposition} Conformal Texture Scaling
:label: prop-conformal-texture-scaling

The texture variance scales with the inverse metric:

| **Region** | **$\lvert z\rvert$** | **$\Sigma(z)$**                            | **Interpretation**        |
|------------|----------------------|--------------------------------------------|---------------------------|
| Origin     | $\approx 0$          | $\sigma_{\text{tex}}^2/4 \cdot I$          | Moderate texture (coarse) |
| Mid-disk   | $\approx 0.5$        | $\sigma_{\text{tex}}^2 \cdot 9/64 \cdot I$ | Reduced texture           |
| Boundary   | $\to 1$              | $\to 0$                                    | Deterministic texture     |

*Remark (Conformal suppression).* Near the boundary (high resolution/specificity), the metric $G$ diverges, so $G^{-1} \to 0$ and texture fluctuations are suppressed.

:::

:::{prf:definition} Boundary Decoder
:label: def-boundary-decoder

The Decoder $\mathcal{D}$ is the **only** component that sees texture. It performs the **boundary synthesis**:

$$
x = \mathcal{D}(z_{\text{final}}, z_{\text{tex}})

$$
where:
- $z_{\text{final}} = (e_K, z_n)$: Determines the shape, physics, and causal structure
- $z_{\text{tex}}$: "Paints" the high-frequency details onto that structure

:::

:::{prf:definition} Stopping Criterion
:label: def-stopping-criterion

Given a declared generation horizon $\tau_{\max}$, the flow terminates when the radial coordinate exceeds a cutoff or the horizon is reached:

$$
\tau_{\text{stop}} := \min\!\left\{\tau_{\max},\ \inf\{\tau \ge 0 : |z(\tau)| \ge R_{\text{cutoff}}\}\right\}.

$$
Equivalently, in terms of the information depth $-U(z)=2\operatorname{artanh}|z|$ (Definition {prf:ref}`def-hyperbolic-information-potential`), stop when $-U(z)\ge C_{\mathrm{stop}}$ and choose $R_{\text{cutoff}}=\tanh(C_{\mathrm{stop}}/2)$. Taking $C_{\mathrm{stop}}\le C_\partial$ keeps the generated state within the declared boundary budget; this is an operational choice, not a consequence of the metric-law field equation.
In practice, choose $R_{\text{cutoff}} = 1 - \varepsilon$ with $\varepsilon$ tied to the Levin length/resolution ({ref}`Appendix A <sec-appendix-a-full-derivations>`). This is a
computational cutoff, not a terminal task boundary.

**Algorithm 21.3.7 (Boundary Texture Sampling).**

```python
import torch

def sample_boundary_texture(
    z_final: torch.Tensor,        # [B, D] final semantic position
    texture_dim: int,             # Dimension of texture space
    sigma_tex: float = 1.0,       # Base texture std dev
) -> torch.Tensor:
    """
    Sample texture with geometry-dependent variance.

    Implements Definition 21.3.2:
        z_tex ~ N(0, Sigma(z_final))
        Sigma(z) = sigma_tex^2 * G^{-1}(z)

    The partition condition (Axiom 21.3.1) ensures this is called
    ONLY at terminal time, not during interior dynamics.

    Cross-ref: Proposition 21.3.3 (Conformal Scaling)
    """
    B = z_final.shape[0]

    # Compute G^{-1}(z) = (1 - |z|^2)^2 / 4
    r_sq = (z_final ** 2).sum(dim=-1, keepdim=True)  # [B, 1]
    one_minus_r_sq = torch.clamp(1.0 - r_sq, min=1e-6)
    G_inv_scale = (one_minus_r_sq ** 2) / 4.0  # [B, 1]

    # Texture std = sigma_tex * sqrt(G^{-1})
    texture_std = sigma_tex * torch.sqrt(G_inv_scale)  # [B, 1]

    # Sample isotropic Gaussian, then scale
    z_tex = torch.randn(B, texture_dim, device=z_final.device)
    z_tex = z_tex * texture_std  # broadcast scaling

    return z_tex
```

:::

## 05_geometry/04_equations_motion.md

:::{prf:definition} Mass Tensor
:label: def-mass-tensor

We define the **inertial mass tensor** $\mathbf{M}(z)$ as the capacity-constrained metric:

$$
\mathbf{M}(z) := G(z).

$$
This definition has the following operational consequences:
- **High curvature regions** (large $G$) have larger effective mass, yielding smaller velocity updates per unit force
- **Low curvature regions** (small $G$) have smaller effective mass, yielding larger velocity updates per unit force

Units: $[\mathbf{M}_{ij}] = [z]^{-2}$ (same as metric).

*Remark (Risk-Metric Coupling).* Combined with the Metric Law (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), this yields a causal chain:

$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced step size}

$$
The metric-weighted step size decreases in high-curvature (high-risk) regions without explicit penalty terms.

:::

:::{prf:definition} Free-energy Path Action (Operational)
:label: def-extended-onsager-machlup-action

Let $(\mathcal{Z}, G)$ be the latent Riemannian manifold with the capacity-constrained metric ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`). For a path $z: [0, T] \to \mathcal{Z}$, define the following free-energy path functional:

$$
S_{\mathrm{path}}[z] = \int_0^T \left( \frac{1}{2}\mathbf{M}(z)\|\dot{z}\|^2 + \Phi_{\text{eff}}(z) + \frac{T_c}{12}\,R(z) + T_c \cdot H_{\pi}(z) \right) ds,

$$
where:
- $\mathbf{M}(z)\|\dot{z}\|^2 = G_{ij}(z)\,\dot{z}^i\,\dot{z}^j$ is the kinetic energy (mass = metric)
- $\Phi_{\text{eff}}(z)$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $R(z)$ is the scalar curvature of the metric $G$
- $H_{\pi}(z) = -\mathbb{E}_{a \sim \pi}[\log \pi(a|z)]$ is the policy entropy
- $T_c > 0$ is the {prf:ref}`def-cognitive-temperature` (cf. {ref}`Section 21.2 <sec-policy-control-field>`)

This is an operational modeling objective, not the Onsager--Machlup functional of the diffusion below; no most-probable-path claim follows from this definition. A genuine Onsager--Machlup functional also requires the drift, reference measure, and path-tube convention.

*Units.* The displayed expression uses dimensionless computational time and normalized latent coordinates. If a physical time scale is restored, all coefficients are rescaled together; from $dz^k=G^{kj}p_j\,ds$ and $[G]=[z]^{-2}$, the covector momentum has units $[p]=[z]^{-1}\,\mathrm{time}^{-1}$.

*Remark (Curvature Correction).* The term $\frac{T_c}{12}R(z)$ is a stochastic correction that accounts for the path-measure distortion on curved spaces. In flat space ($R = 0$), this term vanishes. The entropy term $T_c H_{\pi}$ ensures the agent prefers stochastic policies in uncertain regions.

:::

:::{prf:proposition} Mass Scaling Near Boundary
:label: prop-mass-scaling-near-boundary

For the Poincare disk, the mass tensor scales as:

$$
\mathbf{M}(z) = \frac{4}{(1-|z|^2)^2} I_d \quad \xrightarrow{|z| \to 1} \quad +\infty.

$$
The metric diverges as $|z| \to 1$, which bounds all finite-action trajectories to the interior of the disk.

:::

:::{prf:remark} Onsager--Machlup Scope
:label: prop-most-probable-path

For the controlled diffusion

$$
dz^k = b^k(z)\,ds + \sqrt{2T_c}\,\sigma^{kj}(z)\,dW^j_s,

$$
where $\sigma \sigma^T = G^{-1}$. The path functional defined above does not identify the most probable path of this diffusion. That identification requires the drift $b$, the reference measure, and a specified path-tube convention.

The singular-perturbation calculation in {ref}`Appendix A.4 <sec-appendix-a-full-derivations>` establishes the overdamped reduction only; it is not an Onsager--Machlup derivation.

:::

:::{prf:definition} Second-Order Geodesic Langevin Equation
:label: def-bulk-drift-continuous-flow

The agent's state evolves as a **particle with position $z$ and momentum $p$** on the Riemannian manifold $(\mathcal{Z}, G)$. The dynamics are given by the coupled **second-order Langevin SDE**:

$$
\begin{cases}
dz^k = G^{kj}(z)\, p_j\, ds \\[8pt]
dp_k = \left[ -\partial_k \Phi_{\text{eff}} - \gamma\, p_k + \beta_{\text{curl}}\, \mathcal{F}_{kj}\, G^{j\ell}\, p_\ell + \Gamma^m_{k\ell}\, G^{\ell j}\, p_j\, p_m + \gamma G_{kj}u_\pi^j \right] ds + \sqrt{2\gamma T_c}\, (G^{1/2})_{kj}\, dW^j_s
\end{cases}

$$
where:
- $\Phi_{\text{eff}}$ is the **effective potential** (Definition {prf:ref}`def-effective-potential`)
- $\gamma > 0$ is the **friction coefficient** (damping rate)
- $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$ is the **Value Curl** tensor (Definition {prf:ref}`def-value-curl`)
- $\beta_{\text{curl}} \ge 0$ is the **curl coupling strength** (dimensionless)
- $\Gamma^m_{k\ell}$ are the **Christoffel symbols** of the Levi-Civita connection (Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`)
- $u_\pi^j$ is the contravariant **policy velocity** from Definition {prf:ref}`def-the-control-field`; the momentum equation uses the covector force $\gamma G_{kj}u_\pi^j$
- $T_c$ is the **cognitive temperature** (Definition {prf:ref}`def-cognitive-temperature`)
- $W_s$ is a standard Wiener process

*Units.* In normalized computational units $s$, $\Phi_{\text{eff}}$, and $T_c$ are dimensionless. If a physical time scale $\tau$ is restored, $[p]=[z]^{-1}\tau^{-1}$ follows from the kinematic equation and the force coefficients are rescaled accordingly.

**Interpretation:** The position evolves via the momentum (kinematic relation), while the momentum evolves under:

1. **Gradient force**: $-\nabla\Phi_{\text{eff}}$ — force from effective potential
2. **Friction**: $-\gamma p$ — damping toward equilibrium
3. **Lorentz force**: $\beta_{\text{curl}} \mathcal{F} G^{-1} p$ — velocity-dependent force from Value Curl (perpendicular to velocity)
4. **Geodesic correction**: $+\Gamma^m_{k\ell}G^{\ell j}p_jp_m$ in covector momentum coordinates, which yields $-\Gamma(\dot z,\dot z)$ in the second-order position equation
5. **Control field**: $\gamma G_{kj}u_\pi^j$ — the policy velocity converted to a covector force
6. **Thermal noise**: $\sqrt{2\gamma T_c} G^{1/2} dW$ — fluctuation-dissipation balanced noise

**Hamiltonian Structure:** In the conservative, uncontrolled subcase ($T_c=0$, $\gamma=0$, $\beta_{\text{curl}}=0$, and $u_\pi=0$), the deterministic part derives from the Hamiltonian:

$$
H(z, p) = \frac{1}{2} G^{ij}(z)\, p_i\, p_j + \Phi_{\text{eff}}(z).

$$
With $\beta_{\text{curl}}=0$ and any policy force absorbed into a scalar potential, the friction and noise terms form the usual **Ornstein--Uhlenbeck thermostat** and the Boltzmann statement applies under the stated boundary and regularity conditions.

**Conservative Limit:** When $\mathcal{F} = 0$ (Definition {prf:ref}`def-conservative-reward-field`), the Lorentz term vanishes and we recover the standard geodesic Langevin equation.

**Non-Conservative Dynamics:** When $\mathcal{F} \neq 0$, the Lorentz force induces rotational dynamics. Trajectories may converge to limit cycles rather than fixed points (Theorem {prf:ref}`thm-ness-existence`).

*Remark (BAOAB Integration).* This second-order system is integrated using the Boris-BAOAB scheme (Definition {prf:ref}`def-baoab-splitting`), which preserves the Boltzmann distribution to $O(h^2)$ and handles the velocity-dependent Lorentz force via the Boris rotation.

:::

:::{prf:proposition} Explicit Christoffel Symbols for Poincaré Disk
:label: prop-explicit-christoffel-symbols-for-poincare-disk

For the Poincare disk model with metric $G_{ij} = \frac{4\delta_{ij}}{(1-|z|^2)^2}$, the Christoffel symbols in Cartesian coordinates are:

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right).

$$
The geodesic correction term $\Gamma^k_{ij}\dot{z}^i\dot{z}^j$ contracts to:

$$
\Gamma^k_{ij}\dot{z}^i\dot{z}^j = \frac{4(z \cdot \dot{z})}{1-|z|^2}\dot{z}^k - \frac{2|\dot{z}|^2}{1-|z|^2}z^k.

$$
:::

:::{prf:definition} Mass Evolution - Jump Process
:label: def-mass-evolution-jump-process

The importance weight $m(s)$ evolves according to a coupled jump-diffusion:

$$
dm = m \cdot r(z, a)\,ds + m \cdot (\eta - 1)\,dN_s,

$$
where:
- $r(z, a)$ is the **reaction rate** from the WFR dynamics ({ref}`Section 20.2 <sec-the-wfr-metric>`)
- $N_s$ is a Poisson process with target-dependent intensity
  $\lambda_{\text{jump}}(z\to z')$
- $\eta$ is the multiplicative jump factor; its value is a declared
  resampling convention rather than an implication of the critic cost

*Interpretation:* Between jumps, mass evolves smoothly via the reaction term $r$. At jump times, the mass is rescaled by factor $\eta$, and the position is teleported via the chart transition operator $L_{i \to j}$.

:::

:::{prf:definition} Target-dependent Jump Intensity
:label: prop-jump-intensity-from-value-discontinuity

For a proposed transition $z\mapsto L(z)$, one admissible target-dependent jump intensity is:

$$
\lambda_{\text{jump}}(z\to L(z)) = \lambda_0 \cdot \exp\left(\beta_{\text{ent}} \cdot \left( V_{\text{source}}(z) - V_{\text{target}}(L(z)) - c_{\text{transport}} \right) \right),

$$
where:
- $\lambda_0 > 0$ is a base jump rate
- $\beta_{\text{ent}} > 0$ is the inverse temperature (sharpness)
- $V_{\text{target}}$ and $V_{\text{source}}$ are cost-to-go functions on the target and source charts; the displayed sign favors lower target cost
- $L: \mathcal{Z}_{\text{source}} \to \mathcal{Z}_{\text{target}}$ is the chart transition operator
- $c_{\text{transport}} \ge 0$ is the transport cost (WFR term)

*Remark (SMC Interpretation).* The mass $m(s)$ is precisely the **importance weight** in Sequential Monte Carlo (SMC) / particle filtering. The agent is a single-particle realization of the WFR flow from {ref}`Section 20 <sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces>`. Multiple particles can be used for ensemble-based generation.

**Cross-references:** {ref}`Section 20.2 <sec-the-wfr-metric>` ({prf:ref}`def-the-wfr-action`), {ref}`Section 20.6 <sec-the-unified-world-model>` (WFR world model), {ref}`Section 11 <sec-intrinsic-motivation-maximum-entropy-exploration>` (Filtering and projection).

:::

:::{prf:definition} Effective Potential
:label: def-effective-potential

The unified effective potential is:

$$
\Phi_{\text{eff}}(z, K) = \alpha\, U(z) + (1 - \alpha)\, V_{\text{critic}}(z, K) + \gamma_{risk}\, \Psi_{\text{risk}}(z),

$$
where:
- $U(z) = -d_{\mathbb{D}}(0, z) = -2\operatorname{artanh}(|z|)$ is the **hyperbolic information potential** (Definition {prf:ref}`def-hyperbolic-information-potential`)
- $V_{\text{critic}}(z, K)$ is the **learned cost-to-go/critic** on chart $K$ ({ref}`Section 2.7 <sec-the-hjb-correspondence>`; lower is better)
- $\Psi_{\text{risk}}(z) = \frac{1}{2}\operatorname{tr}(T_{ij} G^{ij})$ is the **risk-stress contribution** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\alpha \in [0, 1]$ is the generation-vs-control hyperparameter
- $\gamma_{risk} \ge 0$ is the risk aversion coefficient

Units: $[\Phi_{\text{eff}}] = \mathrm{nat}$.

:::

:::{prf:proposition} Mode Interpretation
:label: prop-mode-interpretation

The parameter $\alpha$ interpolates between pure generation and pure control:

| Regime              | $\alpha$ Value      | Behavior                                                       |
|---------------------|---------------------|----------------------------------------------------------------|
| **Pure Generation** | $\alpha = 1$        | Flow follows $-\nabla_G U$ (holographic expansion, {ref}`Section 21 <sec-radial-generation-entropic-drift-and-policy-control>`) |
| **Pure Control**    | $\alpha = 0$        | Flow follows $-\nabla_G V_{\text{critic}}$ (policy gradient)   |
| **Hybrid**          | $\alpha \in (0, 1)$ | Balanced generation and control                                |

*Remark (Risk Modulation).* The $\gamma_{risk}$ term provides an additional penalty in high-stress regions (large $T_{ij}$), which further discourages risky trajectories beyond the geometric slowdown from Mass=Metric.

:::

:::{prf:corollary} Gradient Decomposition
:label: cor-gradient-decomposition

The gradient of the effective potential decomposes as:

$$
\nabla_G \Phi_{\text{eff}} = \alpha\, \nabla_G U + (1 - \alpha)\, \nabla_G V_{\text{critic}} + \gamma_{risk}\, \nabla_G \Psi_{\text{risk}}.

$$
For the Poincare disk model, the first term simplifies to:

$$
\nabla_G U = -\frac{(1-|z|^2)}{2}\, \hat{z}, \qquad \hat{z} = \frac{z}{|z|}.

$$
**Cross-references:** Definition {prf:ref}`def-hyperbolic-information-potential`, {ref}`Section 2.7 <sec-the-hjb-correspondence>` (Critic $V$), Section 14.2 (MaxEnt control), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

*Forward reference (Scalar Field Interpretation).* {ref}`Section 24 <sec-the-reward-field-value-forms-and-hodge-geometry>`
provides the complete field-theoretic interpretation of $V_{\text{critic}}$: the Critic solves the **Screened Poisson
Equation** (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) with rewards as boundary flux (scalar charges in the
conservative case; Definition {prf:ref}`def-the-reward-flux`), the Value represents **Gibbs Free Energy** (Axiom
{prf:ref}`ax-the-boltzmann-value-law`), and the Value Hessian induces a **Conformal Coupling** to the metric (Definition
{prf:ref}`def-value-metric-conformal-coupling`).

:::

:::{prf:definition} Cognitive Temperature
:label: def-cognitive-temperature

The **cognitive temperature** $T_c > 0$ is the exploration-exploitation tradeoff parameter that controls:

1. **Diffusion magnitude:** The thermal noise term in the geodesic SDE scales as $\sqrt{2T_c}\,dW$
2. **Boltzmann policy:** The softmax temperature in $\pi(a|z) \propto \exp(Q(z,a)/T_c)$
3. **Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$

*Units:* nat (dimensionless in natural units where $k_B = 1$).

*Correspondence:* $T_c$ is the agent-theoretic analogue of thermodynamic temperature $k_B T$ in statistical mechanics.
:::

:::{prf:definition} Boris-BAOAB Splitting
:label: def-baoab-splitting

The Boris-BAOAB integrator splits the Lorentz-Langevin dynamics into five substeps per time step $h$:

1. **B** (half kick + optional curl rotation):
   - Apply the covector half-kick
     $p^- \leftarrow p - \frac{h}{2}(d\Phi-\gamma G u_\pi)$.
   - If $\mathcal{F}\neq0$, apply the Cayley update for the velocity-dependent term. With
     $A=\beta_{\text{curl}}\mathcal{F}G^{-1}$,
     $$
     p^+ \leftarrow \left(I-\frac{h}{2}A\right)^{-1}
       \left(I+\frac{h}{2}A\right)p^-.
     $$
     In three dimensions this has the usual Boris cross-product form; the matrix form is the
     definition in arbitrary dimension.
   - The second half-kick is the B substep at the end of the symmetric composition.

2. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}(z)\, \xi$, where $\xi \sim \mathcal{N}(0, I)$

4. **A** (half drift): $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1}(z)\, p\right)$

5. **B** (half kick + Boris rotation): Same as step 1

where $c_1 = e^{-\gamma h}$ and $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Conservative Limit:** When $\mathcal{F} = 0$, the Boris rotation is identity and the idealized composition reduces to standard BAOAB.

The reference code below is an implementation approximation: it uses an exponential-map drift and a local Christoffel correction, and therefore should not be read as an exact geodesic substep or as a proof of the invariant measure.

*Remark (Curl Rotation).* Under the skew condition $A^{\mathsf T}G^{-1}+G^{-1}A=0$, the Cayley update is a metric-orthogonal, volume-preserving rotation and preserves the kinetic norm $p^{\mathsf T}G^{-1}p$. It need not preserve the Euclidean norm $|p|$. The Lorentz force therefore does no work in the metric kinetic energy.

*Remark (O-step).* The O-step implements the **Ornstein-Uhlenbeck thermostat**, which exactly preserves the Maxwell-Boltzmann momentum distribution $p \sim \mathcal{N}(0, T_c G)$.

:::

:::{prf:definition} Möbius Translation on the Poincaré Disk
:label: def-mobius-translation

For $c,z\in\mathbb D^d$, the Möbius translation that sends $c$ to the
origin is

$$
\phi_c(z):=(-c)\oplus z,
$$

where $\oplus$ is the Poincaré-disk Möbius addition used by the exponential
map below.  In particular $\phi_c(c)=0$ and $\phi_c$ is an isometry of the
Poincaré metric.  The map changes coordinates only; it does not alter a
potential, a router, or a probability law.

:::

:::{prf:proposition} BAOAB Preserves Boltzmann
:label: prop-baoab-preserves-boltzmann

Under the conservative hypotheses $\beta_{\text{curl}}=0$, $u_\pi=0$, constant $T_c$, reversible boundary conditions, and an exact implementation of the stated symmetric splitting, the BAOAB integrator preserves the Boltzmann distribution $\rho(z, p) \propto \exp(-H(z,p)/T_c)$ to second order in $h$.

:::

:::{prf:theorem} Overdamped Limit
:label: thm-overdamped-limit

Consider the conservative second-order SDE obtained from Definition {prf:ref}`def-bulk-drift-continuous-flow` by setting
$\beta_{\text{curl}}=0$ and $u_\pi=0$, with inertial scale $m$ and friction $\gamma$:

$$
m\,\ddot{z}^k + \gamma\,\dot{z}^k + G^{kj}\partial_j\Phi_{\text{eff}} + \Gamma^k_{ij}\dot{z}^i\dot{z}^j = \sqrt{2\gamma T_c}\,\left(G^{-1/2}\right)^{kj}\,\xi^j,

$$
where $\xi$ is white noise. In physical time, the Smoluchowski--Kramers limit gives
$dz=-(1/\gamma)G^{-1}d\Phi_{\text{eff}}\,dt+\sqrt{2T_c/\gamma}\,G^{-1/2}dW_t$.
After the declared computation-time rescaling $s=t/\gamma$, the formal limit $m/\gamma\to0$ is the Ito equation:

$$
dz^k = \left[-G^{k\ell}(z)\,\partial_\ell\Phi_{\text{eff}}(z) - T_c G^{ij}(z)\Gamma^k_{ij}(z)\right] ds + \sqrt{2T_c}\,\left(G^{-1/2}(z)\right)^{kj}\,dW^j_s.

$$
:::

:::{prf:corollary} Recovery of Holographic Flow
:label: cor-recovery-of-holographic-flow

Setting $\alpha = 1$ (pure generation), $T_c \to 0$, $\mathcal{F}=0$, $u_\pi=0$, and using the computation-time unit in the overdamped equation recovers the prescribed holographic gradient flow from {ref}`Section 21.2 <sec-policy-control-field>`:

$$
\dot{z} = -G^{-1}(z)\,\nabla U(z).

$$
For the Poincare disk and $z\neq0$, this gives $\dot{z} = \frac{(1-|z|^2)}{2}\,\frac{z}{|z|}$, which integrates to $|z(\tau)| = \tanh(\tau/2+\operatorname{artanh}r_0)$.

:::

:::{prf:corollary} Fokker-Planck Duality {cite}`risken1996fokkerplanck`
:label: cor-fokker-planck-duality

In the conservative overdamped subcase above, let $q(z,s)$ denote density
with respect to the Riemannian volume $d\mu_G=\sqrt{|G|}\,dz$.  Its stationary
density is

$$
q_*(z) \propto \exp\left(-\frac{\Phi_{\text{eff}}(z)}{T_c}\right).

$$
The corresponding density with respect to coordinate Lebesgue volume is
$p_*(z)=q_*(z)\sqrt{|G(z)|}$.  The two densities describe the same
Boltzmann law; the factor $\sqrt{|G|}$ is a change of reference measure.

:::

:::{prf:definition} Agent Lifecycle Phases
:label: def-agent-lifecycle-phases


| Phase           | Time Interval                | Dynamics                         | Texture      | Key Operations                                                                         |
|-----------------|------------------------------|----------------------------------|--------------|----------------------------------------------------------------------------------------|
| **1. Init**     | $\tau = 0$                   | $z(0) = 0$                       | None         | Initialize at origin; $p(0) \sim \mathcal{N}(0, T_c G(0))$                             |
| **2. Kick**     | $[0, \tau_{kick}]$           | Langevin at origin               | None         | Apply symmetry-breaking control $u_\pi$ (Def. {prf:ref}`def-the-control-field`)        |
| **3. Bulk**     | $[\tau_{kick}, \tau_{stop}]$ | BAOAB + Jumps                    | **Firewall** | Geodesic flow with chart transitions                                                   |
| **4. Boundary** | $\tau = \tau_{stop}$         | $\lVert z\rVert \geq R_{cutoff}$ | Sampled      | Sample texture $z_{tex} \sim \mathcal{N}(0, \Sigma(z))$                                |
| **5. Decode**   | Post-$\tau_{stop}$           | —                                | Used         | $x = \text{Decoder}(e_K, z_n, z_{tex})$                                                |

*Remark.* The **Texture Firewall** (Axiom {prf:ref}`ax-bulk-boundary-decoupling`) ensures that $\partial_{z_{tex}} \dot{z} = 0$ throughout the bulk phase—texture is completely invisible to the dynamics.

**Algorithm 22.6.2 (Full Agent Loop).**

```python
def run_agent_loop(
    policy: Policy,
    decoder: Decoder,
    T_c: float,
    gamma: float,
    h: float,
    R_cutoff: float = 0.95,
    max_steps: int = 1000,
) -> torch.Tensor:
    """
    Execute the full agent lifecycle from init to decode.

    Returns: Generated output x
    """
    B, d = 1, policy.latent_dim
    device = policy.device

    # ===== Phase 1: Init =====
    z = torch.zeros(B, d, device=device)
    p = torch.randn(B, d, device=device) * math.sqrt(T_c * 4.0)  # G(0) = 4I
    K = torch.zeros(B, dtype=torch.long, device=device)
    m = torch.ones(B, device=device)
    state = GeodesicState(z=z, p=p, K=K, m=m, s=0.0)

    # ===== Phase 2: Kick =====
    # Keep the initial symmetry-breaking velocity and include it in the first
    # bulk control update instead of overwriting it before the first step.
    u_kick = policy.symmetry_breaking_kick(z, mode='generation')

    # ===== Phase 3: Bulk (with Texture Firewall) =====
    for step in range(max_steps):
        # Compute effective potential gradient.  The implementation must return
        # the declared zero subgradient for the radial U term at z=0.
        grad_Phi = compute_effective_potential_gradient(
            state.z, state.K, policy.value_fn, alpha=0.5
        )

        # Update control field.  The kick is applied on the first bulk step.
        u_pi = policy.control_field(state.z, state.K)
        if step == 0:
            u_pi = u_pi + u_kick

        # BAOAB step (texture is invisible here)
        state = geodesic_baoab_step(
            state, grad_Phi, u_pi, T_c, gamma, h,
            jump_rate_fn=policy.jump_rate,
            chart_transition_fn=policy.chart_transition,
            grad_Phi_fn=lambda z_, K_: compute_effective_potential_gradient(
                z_, K_, policy.value_fn, alpha=0.5
            ),
            num_charts=getattr(policy, "num_charts", None),
            mass_jump_factor=getattr(policy, "mass_jump_factor", 1.0),
        )

        # Check boundary condition
        z_norm = torch.sqrt((state.z ** 2).sum(dim=-1))
        if (z_norm >= R_cutoff).all():
            break

    # ===== Phase 4: Boundary - Sample texture =====
    z_tex = sample_holographic_texture(state.z, sigma_tex=0.1)

    # ===== Phase 5: Decode =====
    embedding = policy.chart_embedding(state.K)  # e_K
    x = decoder(embedding, state.z, z_tex)

    return x
```

:::

:::{prf:proposition} Lifecycle Schedule Interpretation
:label: prop-phase-transition-interpretation

The agent lifecycle admits a thermodynamic phase-transition analogy for its schedule:

| Phase | Thermodynamic Analogy | Order Parameter |
|-------|----------------------|-----------------|
| Init (gas) | High entropy, symmetric | $\lVert z\rVert = 0$ |
| Kick (nucleation) | Symmetry breaking | $u_\pi \neq 0$ |
| Bulk (liquid) | Directed flow | $0 < \lVert z\rVert < R_{cutoff}$ |
| Boundary (solid) | Crystallization | $\lVert z\rVert \geq R_{cutoff}$ |

:::

:::{prf:definition} Einstein Relation on Manifolds
:label: def-einstein-relation-on-manifolds

The fluctuation-dissipation relation requires:

$$
\sigma^2(z) = \frac{2\gamma(z)\, T_c}{G(z)},

$$
where $\sigma^2$ is the noise variance. This fixes the fluctuation--dissipation covariance for a chosen $T_c(z)$ and $\gamma(z)$; a Boltzmann equilibrium additionally requires constant coefficients (or the corresponding variable-coefficient correction terms).

:::

:::{prf:proposition} Geometry-scaled Temperature Schedule
:label: prop-automatic-phase-transitions

For the modeling choice $T_c(z)=T_0(1-|z|^2)^2/4$ on the Poincare disk, the effective coordinate noise decreases toward the boundary:

| Regime                      | Metric $G(z)$ | Effective Noise | Phase Behavior                |
|-----------------------------|---------------|-----------------|-------------------------------|
| **Uncertain** (near origin) | Small         | Large           | Gas phase (exploration)       |
| **Certain** (near boundary) | Large         | Small           | Solid phase (crystallization) |

*Remark.* This is an explicit geometry-scaled schedule, not a phase-transition theorem. The Einstein relation alone does not imply a thermodynamic phase transition.

:::

:::{prf:definition} Fisher-Covariance Duality
:label: def-fisher-covariance-duality

The inverse relationship between uncertainty and metric:

$$
G(z) \approx \Sigma^{-1}(z),

$$
where $\Sigma(z)$ is the posterior covariance of the belief at $z$. This duality underlies the Mass=Metric principle (Definition {prf:ref}`def-mass-tensor`).

**Algorithm 22.7.4 (Adaptive Temperature).**

```python
def adaptive_temperature(
    z: torch.Tensor,
    base_T: float,
    certainty_scale: float = 1.0,
) -> torch.Tensor:
    """
    Compute adaptive temperature based on local geometry.

    T_c(z) = base_T * (1 - |z|^2)^2 / 4

    This is a modeling choice that decreases T_c toward the boundary;
    Boltzmann-equilibrium guarantees for constant temperature do not apply.
    """
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    # Conformal factor inverse: G^{-1} = (1-|z|^2)^2 / 4
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return base_T * inv_conformal * certainty_scale
```

:::

:::{prf:corollary} Deterministic Boundary
:label: cor-deterministic-boundary

As $|z| \to 1$:

$$
T_c(z) \to 0, \qquad \text{noise} \to 0.

$$
The coordinate noise in the bulk position tends to zero under this schedule. This does not make separately sampled boundary texture deterministic.

:::

## 06_fields/01_boundary_interface.md

:::{prf:definition} Symplectic Interface Lift
:label: def-symplectic-boundary-manifold

For an interface whose lifted phase space is $T^*\mathcal{Q}$, use canonical coordinates $(q,p)$ and
the symplectic form $(T^*\mathcal{Q},\omega)$ where:
- $q \in \mathcal{Q}$ is the **position bundle** (sensory configuration)
- $p \in T^*_q\mathcal{Q}$ is the **momentum bundle** (motor flux)

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.

$$
The units of $\omega$ are inherited from the chosen coordinate and momentum units; no information
unit is implied without an explicit normalization.

*Remark (Causal Structure).* The symplectic structure encodes causality: observations fix "where" the belief state is (position), while actions fix "how" it flows outward (momentum/flux). These cannot be treated symmetrically as static fields.

:::

:::{prf:definition} Sensory Assimilation Target (Dirichlet-like limit)
:label: def-dirichlet-boundary-condition-sensors

The sensory input stream $\phi(x)$ supplies an observation posterior or
target density in the latent domain:

$$
\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t)),

$$
where $q_{\text{obs}}(t) = E_\phi(x_t)$ is the encoded observation. A literal
delta is the zero-noise limit. In the WFR continuity equation this target is
normally coupled through an assimilation source

$$
S_{\mathrm{obs}}(z,t)=\kappa_{\mathrm{obs}}
\bigl(\rho_{\mathrm{obs}}(z,t)-\rho_{\mathrm{bulk}}(z,t)\bigr),
$$

with a finite-width likelihood for a noisy sensor. A literal Dirichlet trace
is recovered only as a strong-assimilation limit on a genuine geometric
boundary.

*Interpretation:* Information flow from environment to agent (observation).

:::

:::{prf:definition} Neumann Boundary Condition --- Motors
:label: def-neumann-boundary-condition-motors

The motor output stream imposes a **Neumann-like** condition on the WFR transport flux:

$$
j_{\rho}\!\cdot\mathbf{n}\big|_{\partial\mathcal{Z}_{\text{motor}}}
= j_{\text{motor}}(z,u_\pi,t),

$$
where $j_{\rho}=\rho v$ for the WFR transport equation. If a Fokker--Planck
diffusion is included, use the total probability flux
$j_{\rho}=\rho v-T_cG^{-1}\nabla\rho$. The motor current is an
information-flux functional of the texture-free interface state:

$$
j_{\text{motor}} = J_A(z,u_\pi,t),

$$
*Interpretation:* Information flow from agent to environment (action). Its
units are the density-flux units induced by the chosen WFR time coordinate.
The raw decoder output (torques or voltages) is a separate physical quantity;
an interface map is required before it can be converted into $J_A$.

:::

:::{prf:remark} Conditional Symplectic Duality
:label: prop-symplectic-duality-principle

On a genuine even-dimensional phase-space lift, the canonical transformation
$(q,p)\mapsto(p,-q)$ exchanges coordinate roles. This gives a useful analogy between sensing and
actuation. It does not, by itself, map a PDE Dirichlet trace into a Neumann flux condition; that
requires a specified Hamiltonian boundary-value problem and a Legendre transform on the same
configuration manifold.

**Cross-references:** {ref}`sec-the-interface-and-observation-inflow` (Observation inflow), Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`.

:::

:::{prf:definition} Visual Atlas — Perception
:label: def-visual-atlas-perception

The Visual Atlas $\mathcal{A}_{\text{vis}} = \{(U_\alpha, \phi_\alpha, e_\alpha^{\text{vis}})\}_{\alpha \in \mathcal{K}_{\text{vis}}}$ is a chart atlas on the sensory manifold $\mathcal{Q}$ with:
- **Charts** $U_\alpha \subset \mathcal{Q}$: Objects, Scenes, Viewpoints
- **Chart maps** $\phi_\alpha: U_\alpha \to \mathbb{R}^{d_{\text{vis}}}$: Local coordinates
- **Codebook embeddings** $e_\alpha^{\text{vis}} \in \mathbb{R}^{d_m}$: Discrete macro codes

*Input:* Raw observations $\phi_{\text{raw}}$ (pixels, sensors).
*Output:* Latent state $z \in \mathcal{Z}$ (configuration).

:::

:::{prf:definition} Action Atlas --- Actuation
:label: def-action-atlas-actuation

The Action Atlas $\mathcal{A}_{\text{act}} = \{(V_\beta, \psi_\beta, e_\beta^{\text{act}})\}_{\beta \in \mathcal{K}_{\text{act}}}$ is a chart atlas on the motor manifold $T^*\mathcal{Q}$ with:
- **Charts** $V_\beta \subset T^*\mathcal{Q}$: Gaits, Grasps, Tool Affordances (topologically distinct control regimes)
- **Chart maps** $\psi_\beta: V_\beta \to \mathbb{R}^{d_{\text{act}}}$: Local motor coordinates
- **Codebook embeddings** $e_\beta^{\text{act}} \in \mathbb{R}^{d_m}$: Action primitive codes

*Input:* Intention $u_{\text{intent}} \in T_z\mathcal{Z}$ (from Policy, {ref}`sec-policy-control-field`).
*Output:* Actuation $a_{\text{raw}}$ (torques, voltages).

*Remark (Jump Operator in Action Atlas).* The **Jump Operator** $L_{\beta \to \beta'}$ in the Action Atlas represents **Task Switching**: transitioning from one control primitive to another (e.g., "Walk" $\to$ "Jump", "Grasp" $\to$ "Release"). This mirrors the chart transition operator in the Visual Atlas ({ref}`sec-the-unified-world-model`).

:::

:::{prf:proposition} Conditional Atlas Legendre Correspondence
:label: thm-atlas-duality-via-legendre-transform

Let $\mathcal A_{\mathrm{vis}}^T$ be the tangent lift of the visual atlas to
$T\mathcal Q$, and let $G_{\mathcal Q}=E_\phi^*G$ be a positive metric on
$\mathcal Q$ pulled back from the latent metric. If
$L(q,\dot q)$ is $C^2$ and strictly convex in $\dot q$, its Legendre map

$$
\mathcal L_L:T\mathcal Q\longrightarrow T^*\mathcal Q,
\qquad (q,\dot q)\longmapsto
\left(q,\frac{\partial L}{\partial\dot q}\right)
$$

is a fibrewise diffeomorphism. For
$L=\tfrac12\|\dot q\|_{G_{\mathcal Q}}^2-V(q)$, the induced momentum is
$p=G_{\mathcal Q}(q)\dot q$. On a lifted visual chart $(U_\alpha,
T\phi_\alpha)$, an induced action chart can therefore use

$$
\psi_\beta\circ\mathcal L_L\circ(T\phi_\alpha)^{-1}(q,\dot q)
 =\bigl(q,\,G_{\mathcal Q}(q)\dot q\bigr).
$$

This construction defines the compatible lifted atlas
$\mathcal L_L(\mathcal A_{\mathrm{vis}}^T)$. An independently learned action
atlas may have a different number of charts; compatibility is a design
constraint or alignment loss, not the equality
$\mathcal A_{\mathrm{act}}=\mathcal L_L(\mathcal A_{\mathrm{vis}})$.

:::

:::{prf:definition} The Holographic Shutter — Unified Interface
:label: def-the-holographic-shutter-unified-interface

The Shutter is extended from {ref}`sec-the-shutter-as-a-vq-vae` to a symmetric tuple:

$$
\mathbb{S} = (\mathcal{A}_{\text{vis}}, \mathcal{A}_{\text{act}}),

$$
where:
- **Ingress (Perception):** $E_\phi: \mathcal{Q} \to \mathcal{Z}$ via Visual Atlas
- **Egress (Actuation):** $D_A: T_z\mathcal{Z} \times \mathcal{Z} \to T^*\mathcal{Q}$ via Action Atlas
- **Proprioception (Inverse Model):** $E_A: T^*\mathcal{Q} \to T_z\mathcal{Z}$ maps realized actions back to intentions

**Cross-references:** {ref}`sec-the-shutter-as-a-vq-vae` (VQ-VAE Shutter), {ref}`sec-tier-the-attentive-atlas` (AttentiveAtlasEncoder), {ref}`sec-decoder-architecture-overview-topological-decoder` (TopologicalDecoder).

:::

:::{prf:definition} Motor Texture Decomposition
:label: def-motor-texture-decomposition

The motor output decomposes as:

$$
a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}}),

$$
where:
- $K^{\text{act}}_t \in \mathcal{K}_{\text{act}}$ is the **discrete motor macro** (action primitive/chart index)
- $z_{n,\text{motor}} \in \mathbb{R}^{d_{\text{motor},n}}$ is **motor nuisance** (impedance, compliance, force distribution)
- $z_{\text{tex,motor}} \in \mathbb{R}^{d_{\text{motor,tex}}}$ is **motor texture** (tremor, fine-grained noise, micro-corrections)

*Remark (Parallel to Visual Decomposition).* This mirrors the visual decomposition $(K_t, z_{n,t}, z_{\text{tex},t})$ from {ref}`sec-the-shutter-as-a-vq-vae`:

| Component                 | Visual Domain                 | Motor Domain                              |
|---------------------------|-------------------------------|-------------------------------------------|
| **Macro (discrete)**      | Object/Scene chart $K$        | Action primitive $K^{\text{act}}$         |
| **Nuisance (continuous)** | Pose/viewpoint $z_n$          | Compliance/impedance $z_{n,\text{motor}}$ |
| **Texture (residual)**    | Pixel detail $z_{\text{tex}}$ | Tremor/noise $z_{\text{tex,motor}}$       |

:::

:::{prf:definition} Compliance Tensor
:label: def-compliance-tensor

The motor nuisance encodes the **compliance tensor**:

$$
C_{ij}(z_{n,\text{motor}}) = \frac{\partial a^i}{\partial f^j},

$$
where $f$ is the external force/feedback. This determines how the motor output responds to perturbations:
- **High compliance** ($C$ large): Soft, yielding response (safe interaction)
- **Low compliance** ($C$ small): Stiff, precise response (accurate positioning)

Units: $[C_{ij}] = [a]/[f]$.

:::

:::{prf:definition} Motor Texture Distribution
:label: def-motor-texture-distribution

At the motor boundary, texture is sampled from a geometry-dependent Gaussian:

$$
z_{\text{tex,motor}} \sim \mathcal{N}(0, \Sigma_{\text{motor}}(z)),

$$
where:

$$
\Sigma_{\text{motor}}(z) = \sigma_{\text{motor}}^2 \cdot G_{\text{motor}}^{-1}(z) = \sigma_{\text{motor}}^2 \cdot \frac{(1-|z|^2)^2}{4} I_{d_{\text{motor,tex}}}.

$$
This follows the same conformal scaling as visual texture (Definition {prf:ref}`def-boundary-texture-distribution`), ensuring consistent thermodynamic behavior.

:::

:::{prf:axiom} Motor Texture Firewall
:label: ax-motor-texture-firewall

Motor texture is decoupled from the Bulk dynamics:

$$
\partial_{z_{\text{tex,motor}}} \dot{z} = 0, \qquad \partial_{z_{\text{tex,motor}}} u_\pi = 0.

$$
The policy $\pi_\theta$ operates on $(K, z_n, A, z_{n,\text{motor}})$ but **never** on $(z_{\text{tex}}, z_{\text{tex,motor}})$.

*Remark (Sim-to-Real Gap).* The **motor texture variance** $\sigma_{\text{motor}}^2$ is the mathematical definition of the "Sim-to-Real gap":
- **Simulation:** $\sigma_{\text{motor}} \approx 0$ (deterministic, no tremor)
- **Reality:** $\sigma_{\text{motor}} > 0$ (friction, sensor noise, motor tremor)
- **Robustness:** The Bulk policy $u_\pi$ is invariant; only the Action Decoder learns to manage domain-specific noise.

**Cross-references:** {ref}`sec-the-retrieval-texture-firewall` (Texture Firewall), Axiom {prf:ref}`ax-bulk-boundary-decoupling`.

:::

:::{prf:definition} Cycle Phases
:label: def-cycle-phases


| Phase             | Process            | Information Flow                      | Entropy Change               |
|-------------------|--------------------|---------------------------------------|------------------------------|
| **I. Perception** | Compression        | Mutual information $I(X;K)$ extracted | $\Delta S_{\text{bulk}} < 0$ |
| **II. Dreaming**  | Internal evolution | No external exchange                  | Model-dependent; $\Delta S=0$ only in the isolated reversible limit  |
| **III. Action**   | Expansion          | Mutual information $I(A;K)$ injected  | $\Delta S_{\text{bulk}} > 0$ |

*Remark (Statistical mechanics analogy).* This cycle is structurally analogous to a Stirling cycle in thermodynamics. The analogy does not supply an entropy balance or efficiency bound without specified reservoirs and an entropy-production estimate.

:::

:::{prf:remark} Perception as Compression (Operational Identity)
:label: thm-perception-as-compression

During perception, the agent compresses external entropy into internal free energy:

$$
W_{\text{compress}} = T_c \cdot I(X_t; K_t) \geq 0,

$$
where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I(X_t; K_t)$ is the mutual information extracted from the observation $X_t$ into the macro-state $K_t$.

*Mechanism:* The Visual Encoder $E_\phi$ compresses high-entropy raw data $\phi_{\text{raw}}$ into a low-entropy macro-state $z$. The "heat" absorbed is the raw sensory stream.

*Information-theoretic interpretation:* Entropy decreases ($\Delta S < 0$). The Information Bottleneck cost bounds the compression.

:::

:::{prf:remark} Action as Expansion (Operational Identity)
:label: thm-action-as-expansion

During action, the agent expands internal free energy into external control:

$$
W_{\text{expand}} = T_c \cdot I(K^{\text{act}}_t; K_t) \geq 0,

$$
where $I(K^{\text{act}}_t; K_t)$ is the mutual information injected from the intention into the motor output.

*Mechanism:* The Action Decoder $D_A$ "expands" the low-entropy Intention $u_\pi$ into high-dimensional motor commands $a_{\text{raw}}$, injecting motor texture.

*Information-theoretic interpretation:* Entropy increases ($\Delta S > 0$). The agent injects stochastic texture into motor outputs.

:::

:::{prf:remark} Dreaming as an Isolated-Limit Approximation
:label: def-dreaming-as-unitary-evolution

In the ideal isolated, zero-friction limit, one may model the dreaming phase by a Hamiltonian flow. This is an approximation; the WFR/Langevin dreaming model remains thermal unless those terms are explicitly removed:


$$
\partial_s \rho + [H_{\text{internal}}, \rho]_{\text{Poisson}} = 0,

$$
where $H_{\text{internal}}$ is the effective Hamiltonian:

$$
H_{\text{internal}}(z, p) = \frac{1}{2}\|p\|_{G^{-1}}^2 + V_{\text{critic}}(z).

$$
*Mechanism:* The agent is decoupled from the boundary (adiabatic/isolated). The Bulk evolves under Hamiltonian dynamics (BAOAB integrator with $\gamma \to 0$).

*Information-theoretic interpretation:* Isentropic ($\Delta S = 0$). Internal planning proceeds without information exchange with the environment.

:::

:::{prf:remark} Information-Cycle Efficiency Analogy
:label: prop-carnot-efficiency-bound

A Carnot-style bound does not follow from the definitions in this volume. The ratio below is an operational efficiency diagnostic; interpreting it as a thermodynamic bound requires an explicit two-reservoir model, temperatures, and an entropy-production inequality:

$$
\eta = \frac{I(K^{\text{act}}_t; K_t)}{I(X_t; K_t)} \leq 1 - \frac{T_{\text{motor}}}{T_{\text{sensor}}},

$$
where $T_{\text{sensor}}$ and $T_{\text{motor}}$ are the effective temperatures at the sensory and motor boundaries.

*Interpretation:* Perfect efficiency ($\eta = 1$) requires $T_{\text{motor}} = 0$ (deterministic motors) or $T_{\text{sensor}} \to \infty$ (infinite sensory entropy). Real systems operate at $\eta < 1$.

**Cross-references:** {ref}`sec-adaptive-thermodynamics` (Adaptive Thermodynamics), {ref}`sec-the-equivalence-theorem` (MaxEnt Control).

*Forward reference (Reward as Heat).* {ref}`sec-the-bulk-potential-screened-poisson-equation` establishes that Reward is the thermodynamic **heat input** that drives the cycle: the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`) identifies $V(z) = E(z) - T_c S(z)$ as Gibbs Free Energy, and Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass` proves that WFR dynamics materialize the agent in high-value regions ("Value Creates Mass").

:::

:::{prf:definition} Waking: Observation Assimilation and Motor Flux
:label: def-waking-boundary-clamping

During waking, the sensory stream supplies an observation target (or posterior) to the
bulk.  Write this target as $\rho_{\mathrm{obs}}(z,t)$.  A finite-noise encoder
does not impose an exact Dirichlet trace; its coupling is represented by the
assimilation source introduced in
{prf:ref}`def-dirichlet-boundary-condition-sensors`:

$$
S_{\mathrm{obs}}(z,t)=\kappa_{\mathrm{obs}}(z,t)\bigl(\rho_{\mathrm{obs}}(z,t)-\rho_{\mathrm{bulk}}(z,t)\bigr).

$$
In the same schedule the motor interface may prescribe a WFR flux,

$$
j_{\rho}\!\cdot\mathbf{n}=J_A(z,u_\pi,t),
$$

with $J_A$ carrying the flux units specified in
{prf:ref}`def-neumann-boundary-condition-motors`.  An exact Dirichlet clamp is recovered
only as a declared strong-assimilation limit on a genuine boundary.  The
transport/reaction balance is selected by the stated WFR action and its
parameters; there is no universal threshold at which one term must dominate.

:::

:::{prf:definition} Dreaming: Reflective Boundary
:label: def-dreaming-reflective-boundary

During dreaming, the sensory stream is cut and the sensory boundary is **Reflective**. The motor boundary may still be specified separately; $u_\pi=0$ is a common closed-loop choice, not the definition of the mode:

$$
j_{\rho}\!\cdot\mathbf{n} = 0 \quad \text{(Reflective/Neumann-zero)}.

$$
The system is closed with respect to the sensory boundary.  Total mass is
conserved only when the reaction term also integrates to zero,
$\int_{\mathcal{Z}}\rho r\,d\mu_G=0$; the motor boundary and any internal
source or sink must be specified separately.  In the closed-loop choice
$u_\pi=0$, the remaining drift is driven by the internal potential
$V_{\text{critic}}(z)$, but this is a modeling choice rather than the
definition of dreaming.

:::

:::{prf:remark} WFR Mode Switching
:label: thm-wfr-mode-switching

Changing the sensory boundary from an open observation trace to a reflective trace defines the operational waking/dreaming switch. This is a boundary-condition change, not a thermodynamic phase-transition theorem:

| Mode         | Sensory coupling                         | Motor coupling                    | Internal Flow | Information Balance       |
|--------------|-------------------------------------------|-----------------------------------|---------------|---------------------------|
| **Waking**   | Assimilation source $S_{\mathrm{obs}}$     | Prescribed flux $J_A$             | Source-driven | Depends on supplied trace  |
| **Dreaming** | Reflective $j_\rho\!\cdot n=0$           | Reflective or prescribed         | Recirculating | Zero sensory flux          |
| **Pure actuation** | No sensory source                    | Prescribed flux $J_A$             | Motor-driven  | Net outward flux allowed   |

:::

:::{prf:proposition} Net Interface Flux and Grounding Rate
:label: prop-grounding-rate-via-boundary-flux

The signed net interface flux can be recorded as the operational diagnostic
$\Phi_{\mathrm{net}}(t)$:

$$
\Phi_{\mathrm{net}}(t)=
\oint_{\partial\mathcal{Z}_{\mathrm{sense}}}j_{\mathrm{obs}}\!\cdot dA
-\oint_{\partial\mathcal{Z}_{\mathrm{motor}}}j_{\mathrm{motor}}\!\cdot dA.

$$

This signed flux is distinct from the upstream grounding rate
$G_t:=I(X_t;K_t)$ and its realised rate
$\lambda_{\mathrm{in}}=\mathbb{E}[G_t]$ in
{prf:ref}`def-grounding-rate`.  A relation between expected sensory flux and
$\lambda_{\mathrm{in}}$ is an additional interface calibration assumption.
The sign of $\Phi_{\mathrm{net}}$ records the chosen boundary convention: it
can be positive in an observation-driven schedule, zero for a closed sensory
boundary, or negative when a prescribed motor flux dominates.

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR Action), {ref}`sec-the-unified-world-model` (WFR World Model), {ref}`sec-the-interface-and-observation-inflow` (Observation Inflow).

:::

:::{prf:definition} Context Space
:label: def-context-space

The **Context Space** $\mathcal{C}$ is a manifold parameterizing the control/conditioning signal for the agent:

$$
\mathcal{C} := \{c : c \text{ specifies a boundary condition on } \partial\mathcal{Z}\}.

$$
The context determines the target distribution at the motor boundary via an action-dependent effective cost:

$$
\pi(a | z, c) \propto \exp\left(-\frac{1}{T_c} \mathcal{C}_{\text{eff}}(z,a,c)\right),

$$
Units: $[\mathcal{C}]$ inherits from the task domain.

:::

:::{prf:definition} Context Instantiation Maps
:label: def-context-instantiation-functor

For each task domain, a typed instantiation map sends a task-specific
conditioning signal into $\mathcal C$.  Calling these maps a functor would
require category structures that are not part of this volume.  The three
canonical instantiations are:

| Task Domain        | Context $c \in \mathcal{C}$ | Motor Output $a$           | Action-dependent cost $\mathcal{C}_{\text{eff}}$      |
|--------------------|-----------------------------|----------------------------|----------------------------------------------|
| **RL**             | Task/context space $\mathcal{C}_{\mathrm{RL}}$ | Motor command $a$ | $Q_{\mathrm{cost}}(z,a,c)$ |
| **Classification** | Label space $\mathcal{Y}$   | Class prediction $a$ | $-\log p(a\mid z,c)$ (cross-entropy) |
| **LLM**            | Prompt space $\mathcal{P}$  | Token $a$ | $-\log p(a\mid z,c)$ (next-token loss) |

*Key Insight:* In all cases, the context $c$ functions as the **symmetry-breaking boundary condition** that determines which direction the holographic expansion takes at the origin.

:::

:::{prf:remark} Conditional Context Structure
:label: thm-universal-context-structure

When the task-specific encoder, action-dependent cost, and boundary lift are explicitly identified, the context instantiations share the following schematic structure:

1. **Embedding:** $c \mapsto e_c \in \mathbb{R}^{d_c}$ maps the context to a latent vector.
2. **Typed lift:** a declared map $\iota_c:\mathbb{R}^{d_c}\to T_0\mathcal Z$
   sends that embedding into the tangent space at the reference state.
3. **Symmetry-Breaking Kick:** $\iota_c(e_c)$ determines the initial control field:

   $$
   u_\pi(0) = G^{-1}(0)\,\iota_c(e_c) = \frac{1}{4}\,\iota_c(e_c)

   $$
   (at the Poincare disk origin where $G(0) = 4I$)
4. **Motor Distribution:** The output distribution is:

   $$
   \pi(a | z, c) = \text{softmax}_a\left(-\frac{\mathcal{C}_{\text{eff}}(z,a,c)}{T_c}\right)

   $$
*Scope.* The shared wiring is an architectural analogy. Equivalence of the three task domains requires separate task-specific encoders, action spaces, and costs; it is not implied by the geometric notation alone.

:::

:::{prf:definition} Context-Conditioned WFR
:label: def-context-conditioned-wfr

The WFR dynamics ({ref}`sec-the-wfr-metric`) generalize to context-conditioned form:

$$
\partial_s \rho + \nabla \cdot (\rho\, v_c) = \rho\, r_c,

$$
where:
- $v_c(z) = -G^{-1}(z) \nabla_z \mathcal{C}_{\text{eff}}(z,a,c) + u_\pi(z, c)$ is the context-conditioned velocity, after the action has been selected
- $r_c(z)$ is the context-conditioned reaction rate (chart jumps influenced by context)

:::

:::{prf:remark} Shared Context Interface (Conditional)
:label: cor-prompt-action-label

RL actions, classification labels, and LLM prompts can share an interface
factorization when their task-specific maps are supplied:

$$
 c \longmapsto e_c \longmapsto \iota_c(e_c) \longmapsto \pi(\cdot\mid z,c).

$$
The domains are not isomorphic by notation alone.  Each requires its own
context encoder, action/output space, cost, and boundary lift.  A chart route,
continuous target, or texture channel can be shared only when those maps are
explicitly identified and dimensionally compatible.

**Cross-references:** {ref}`sec-policy-control-field` (Control Field), Theorem {prf:ref}`thm-unified-control-interpretation`, Definition {prf:ref}`def-effective-potential`.

*Forward reference (Effective Potential Resolution).* {ref}`sec-the-bulk-potential-screened-poisson-equation`
resolves the scalar state-cost part of $\Phi_{\text{eff}}$: in the conservative
subcase the critic solves the **Screened Poisson Equation** with the cost
source convention used by the control loop.  An action-dependent context cost
$Q_{\mathrm{cost}}(z,a,c)$ is a separate policy object and is not identified
with the state potential for arbitrary tasks.  The discount factor $\gamma$
 determines the stationary-diffusion screening length
$\ell_{\mathrm{diff}}=\sqrt{T_c\Delta t/(-\ln\gamma)}$ (natural units: $\sqrt{T_c/(-\ln\gamma)}$)
(Corollary {prf:ref}`cor-discount-as-screening-length`),
explaining why distant rewards are exponentially suppressed in policy.

:::

## 06_fields/02_reward_field.md

:::{prf:definition} The Reward 1-Form
:label: def-reward-1-form

Let $\mathcal{R}$ be a differential 1-form on the latent manifold $(\mathcal{Z}, G)$. The **instantaneous reward rate** received by the agent moving with velocity $v \in T_z\mathcal{Z}$ is:

$$
r_t = \mathcal{R}(z)[v] = \mathcal{R}_i(z) \dot{z}^i.

$$

*Units:* $[\mathcal{R}] = \mathrm{nat}/[\text{length}]$ and $[r_t]=\mathrm{nat}/\text{time}$ for the
continuous-time reward rate. A discrete sample is $r_t^{\text{step}}=r_t\,\Delta t$ and has units nat.

The cumulative reward along a trajectory $\gamma: [0,T] \to \mathcal{Z}$ is the **line integral**:

$$
R_{\text{cumulative}} = \int_\gamma \mathcal{R} = \int_0^T \mathcal{R}_i(\gamma(t)) \dot{\gamma}^i(t) \, dt.

$$

*Remark.* Instantaneous reward depends on both position $z$ and velocity $\dot{z}$. A stationary agent ($\dot{z} = 0$) receives zero instantaneous reward.

:::

:::{prf:definition} The Reward Flux (Boundary Form)
:label: def-the-reward-flux

The environment provides reward via a boundary 1-form $J_r$ on $\partial\Omega$. Let
$\iota:\partial\Omega\hookrightarrow\Omega$ be the inclusion. The boundary condition is the pullback
$\iota^*\mathcal{R} = J_r$, and the cumulative boundary reward along a boundary trajectory
$\gamma_\partial$ is:

$$
\int_{\gamma_\partial} J_r = \text{Cumulative Boundary Reward}.

$$
In the discrete limit, this manifests as samples $r_t^{\text{step}} = J_r(\partial_t)\,\Delta t$ deposited at the boundary
coordinates $(t, z_{\text{boundary}})$.

*Units:* $[J_r] = \mathrm{nat}/[\text{length}]$ and $[r_t^{\text{step}}] = \mathrm{nat}$.

*Relation to 1-form:* For any surface $\Sigma$ with boundary $\partial\Sigma$, Stokes' theorem gives
$\oint_{\partial\Sigma}\mathcal{R}=\int_\Sigma d\mathcal{R}$. For a loop $\gamma=\partial\Sigma$, Stokes' theorem relates its circulation to $d\mathcal R$; a general closed loop need not bound a surface. The pullback $\iota^*\mathcal R$ is tangential boundary data;
it is not by itself a Neumann flux or a source density for a value PDE.

:::

:::{prf:definition} Terminal Boundary (End/Death Flags)
:label: def-terminal-boundary

When termination is $\sigma(Z_t)$-measurable, let $\Gamma_{\text{term}} \subset \mathcal{Z}$ denote
the terminal subset representing end/death flags. Under partial observability use the conditional
killing rate instead of identifying the flag with a latent subset.
Define the stopping time $\tau_{\text{term}} := \inf\{t \ge 0 : z_t \in \Gamma_{\text{term}}\}$ and
kill the process upon hitting $\Gamma_{\text{term}}$. For the conservative value PDE, impose a
Dirichlet condition $V|_{\Gamma_{\text{term}}} = V_{\text{term}}$ (often $0$ or a terminal payoff).
In WFR form, include a killing rate $\kappa_{\text{term}}(z) \ge 0$ or a reaction term $r<0$
concentrated on $\Gamma_{\text{term}}$.

*Terminal vs holographic boundary.* $\Gamma_{\text{term}}$ is a task boundary and is separate from the
holographic boundary of the hyperbolic space.

*Computational cutoff.* For numerics we truncate the hyperbolic disk at $\lvert z\rvert = 1-\varepsilon$,
with $\varepsilon$ tied to the Levin length/resolution. This is a computational boundary for stability,
not a physical terminal set.

:::

:::{prf:theorem} Hodge Decomposition of the Reward Field
:label: thm-hodge-decomposition

On a compact latent Riemannian manifold $(\mathcal{Z}, G)$, after choosing the absolute or relative
Hodge--Morrey--Friedrichs boundary condition, the Reward 1-form $\mathcal{R}$ decomposes into:

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}

$$
where:
1. **$\Phi \in \Omega^0(\mathcal{Z})$** (Scalar Potential): The conservative/optimizable component. $d\Phi$ is an exact form.
2. **$\Psi \in \Omega^2(\mathcal{Z})$** (Vector Potential): The rotational/cyclic component. $\delta\Psi$ is a coexact form (divergence-free).
3. **$\eta \in \mathcal{H}^1_{\mathrm{bc}}(\mathcal{Z})$** (Harmonic Flux): a boundary-conditioned harmonic
field satisfying $d\eta = 0$ and $\delta\eta = 0$. Only the corresponding absolute/relative harmonic subspace
is identified with de Rham cohomology and hence with topological cycles; on the truncated disk with the standard
relative boundary condition, $H^1$ is trivial.

We use $\Phi$ for the reward-side potential and $V:=-\Phi$ for the cost-to-go convention used
by the control-loop chapter. Thus $d\Phi=-dV$; the conservative case corresponds to $A=0$.
Define the non-exact component $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $\mathcal{F} = dA$.

*Units:* $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat}$, $[\eta] = \mathrm{nat}/[\text{length}]$.

:::

:::{prf:definition} The Value Curl (Vorticity Tensor)
:label: def-value-curl

The **Value Curl** is the exterior derivative of the reward form:

$$
\mathcal{F} := d\mathcal{R} = dA = d\delta\Psi.

$$
In coordinates: $\mathcal{F}_{ij} = \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i$.

*Units:* $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$ (curvature of value).

**Properties:**
1. $\mathcal{F}$ is antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
2. $\mathcal{F}$ satisfies the Bianchi identity: $d\mathcal{F} = 0$
3. $\mathcal{F}$ is gauge-invariant: if $\mathcal{R} \to \mathcal{R} + d\chi$, then $\mathcal{F} \to \mathcal{F}$

:::

:::{prf:definition} Conservative Reward Field
:label: def-conservative-reward-field

The reward field $\mathcal{R}$ is **conservative** when it is exact, i.e. when there is a scalar $\Phi$ with
$\mathcal{R}=d\Phi$. Equivalently, $d\mathcal R=0$ and all periods $\oint_\gamma\mathcal R$ vanish for closed
loops. On a simply connected domain with the boundary condition above, vanishing curl is sufficient:

$$
\mathcal{R}=d\Phi.

$$
**Conservative Special Case:** In the reward convention $\mathcal{R}=d\Phi$; the cost-to-go used elsewhere is
$V=-\Phi$. For a loop that bounds a surface, $\gamma=\partial\Sigma$, Stokes' theorem gives:

$$
\oint_\gamma \mathcal{R} = \int_\Sigma d\mathcal{R} = 0.

$$

*Remark.* Standard RL assumes conservative reward fields. The scalar value function $V(s)$ exists precisely because path-independence holds.

:::

:::{prf:proposition} Value Cycle Detection
:label: prop-value-cycle-detection

The Value Curl $\mathcal{F}$ can be estimated from trajectory data. For a closed loop $\gamma$ in latent space:

$$
\oint_{\partial\Sigma} \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0
\implies \text{the reward field is not exact.}

$$
**Diagnostic:** Non-zero circulation $\oint_\gamma \mathcal{R}$ indicates non-conservative structure.
Using accumulated TD-error around closed loops is a heuristic that requires approximate loop closure
and a consistent reward estimator.

:::

:::{prf:theorem} Bellman generator and stationary screening
:label: thm-the-hjb-helmholtz-correspondence

For the diffusion and discount already used in
the Bellman diffusion defined here, write
$\mathcal L=b\cdot\nabla+T_c\Delta_G$ and $\gamma_h=e^{-\lambda h}$.
In this theorem $V$ denotes the reward-side score $V_{\mathrm{rew}}=\Phi$;
the control-loop cost critic is $V_{\mathrm{cost}}=-\Phi$ and has source
$\rho_c=-\rho_r$.
The smooth Bellman equation has continuous-time form
$$
\partial_tV+\mathcal LV-\lambda V+r=0.
$$
In its stationary zero-drift sector,
$(-\Delta_G+\lambda/T_c)V=r/T_c$; denote this screening coefficient by
$\kappa_B^2=\lambda/T_c$. The scalar field action used in this chapter
instead defines the wave operator
$$
\Box_g=-|g|^{-1/2}\partial_\mu(|g|^{1/2}g^{\mu\nu}\partial_\nu),
\qquad(\Box_g+\kappa^2)V=\rho_r.
$$
The stationary operators coincide under the coefficient identification
$\kappa^2=\kappa_B^2$ and the same sources and boundary realization.

:::

:::{prf:remark} Dimensional Consistency of the Helmholtz Equation
:label: rem-helmholtz-dimensions

The screened Poisson equation $-\Delta_G V + \kappa^2 V = \rho_r$ requires careful dimensional analysis. The naive expression $\kappa = -\ln\gamma$ appears dimensionless, which would be inconsistent with $[\Delta_G] = [\text{length}]^{-2}$.

The stationary Bellman generator fixes the coefficient by
$\kappa^2=\lambda/T_c$. If a separate propagation speed or diffusion coefficient is introduced,
it must be included in the generator before converting to a spatial coefficient; it is not an
independent identification. In normalized units $T_c=\Delta t=1$, this gives
$\kappa=\sqrt{-\ln\gamma}$, not $-\ln\gamma$.

If a separate physical propagation speed is used, its coefficient must be derived from the
corresponding dimensional generator. It cannot be identified with $\lambda/c_{\text{info}}$ without
adding that model explicitly. Under the normalized stationary convention,
$\ell_{\text{screen}}=1/\kappa$ is not determined by $\gamma$ alone.

:::

:::{prf:proposition} Green's Function Interpretation
:label: prop-green-s-function-interpretation

The Critic computes the **Green's function** of the screened Laplacian on the latent geometry:

$$
V(z) = \int_{\Omega} G_\kappa(z, z') \rho_r(z') \, d\mu_G(z') + \mathcal{B}_{\partial\Omega}[G_\kappa, V],

$$
where $G_\kappa(z, z')$ is the Green's function satisfying
$(-\Delta_G + \kappa^2) G_\kappa(z, \cdot) = \delta_z$, and $\mathcal{B}_{\partial\Omega}$ encodes the
chosen boundary condition. A boundary source density $\sigma_r$ must be specified independently as a
single-layer/Neumann datum; it is not the pullback $\iota^*\mathcal R$. Under that separate choice:

$$
V(z) = \int_{\partial\Omega} G_\kappa(z, z') \sigma_r(z') \, d\Sigma(z').

$$

*Remark.* The value at $z$ is a weighted integral of bulk sources plus boundary flux, with weights
given by the Green's function. This is a superposition principle: the Helmholtz equation is linear.

:::

:::{prf:remark} Green's Function Decay Scope
:label: prop-green-s-function-decay

For the Euclidean screened operator (and for geometries with the corresponding asymptotic
analysis), the Green's function has the familiar large-distance form:

$$
G_\kappa(z, z') \sim \frac{1}{d_G(z, z')^{(d-1)/2}} \exp\left(-\kappa \cdot d_G(z, z')\right),

$$
where $d_G$ is the geodesic distance and $d$ is the dimension.

:::

:::{prf:corollary} Discount as Screening Length
:label: cor-discount-as-screening-length

The discount factor $\gamma$ determines a characteristic **screening length**:

$$
\ell_{\text{screen}} = \frac{1}{\kappa} = \sqrt{\frac{T_c}{\lambda}}.

$$
where $\lambda := -\ln\gamma / \Delta t$.
For $\gamma=0.99$, $T_c=\Delta t=1$: $\ell_{\text{screen}}\approx 9.97$ in the normalized latent-length units.

*Interpretation:* Under the displayed Green-kernel hypotheses, rewards at geodesic distance
$>\ell_{\text{screen}}$ are suppressed. On a hyperbolic or bounded domain, the decay rate and
boundary terms must be recomputed rather than inferred from the flat-space asymptotic.

*Note:* Numerical values below assume the normalized stationary convention ($T_c=\Delta t=1$).

**Table 24.2.5 (Discount-Screening Correspondence).**

| Discount $\gamma$ | Screening Mass $\kappa$ | Screening Length $\ell$ | Interpretation                    |
|-------------------|-------------------------|-------------------------|-----------------------------------|
| $\gamma \to 1$    | $\kappa \to 0$          | $\ell \to \infty$       | Infinite horizon (massless field) |
| $\gamma = 0.99$   | $\kappa \approx 0.100$   | $\ell \approx 9.97$      | Standard RL (normalized) |
| $\gamma = 0.9$    | $\kappa \approx 0.325$    | $\ell \approx 3.08$       | Short horizon (normalized)                     |
| $\gamma \to 0$    | $\kappa \to \infty$     | $\ell \to 0$            | Myopic (infinitely massive)       |

**Cross-references:** {ref}`sec-the-hjb-correspondence` (HJB Equation), Theorem {prf:ref}`thm-capacity-constrained-metric-law`.

:::

:::{prf:axiom} Boltzmann-Value Modeling Convention
:label: ax-the-boltzmann-value-law

Let $F(z):=-\Phi(z)$ be the cost/free-energy scalar associated with the Hodge potential. The thermodynamic modeling
convention is

$$
F(z) = E(z) - T_c S(z),

$$
where:
- $E(z)$ is the **task risk/cost** at state $z$
- $S(z)$ is the **exploration entropy** (measure of uncertainty/optionality)
- $T_c$ is the **cognitive temperature** ({prf:ref}`def-cognitive-temperature`, {ref}`sec-hyperbolic-volume-and-entropic-drift`)

*Units:* $[F]=[\Phi]=[E]=[T_c S]=\mathrm{nat}$.

**Conservative Case ($\mathcal{F} = 0$):** When the Value Curl vanishes and periods vanish, $\mathcal{R}=d\Phi$.
The cost convention used by the control loop is $V(z)=-\Phi(z)$.

**Non-Conservative Case ($\mathcal{F} \neq 0$):** The scalar potential $\Phi$ captures only the optimizable component of the reward field. The solenoidal component $\delta\Psi$ creates additional cyclic dynamics.

:::

:::{prf:definition} Canonical Ensemble {cite}`sutton2018rl`
:label: def-canonical-ensemble

When an invariant measure exists and the drift and boundary conditions satisfy detailed balance,
this potential induces a probability measure on the manifold via the **Canonical Ensemble**:

$$
P_{\text{stationary}}(z) = \frac{1}{Z} \exp\left(\frac{V_{\mathrm{rew}}(z)}{T_c}\right)
 = \frac{1}{Z} \exp\left(\frac{\Phi(z)}{T_c}\right),

$$
where $Z = \int_{\mathcal{Z}} \exp(V_{\mathrm{rew}}(z)/T_c) \, d\mu_G(z)$ is the partition function.

*Sign Convention:* $V_{\mathrm{rew}}=\Phi$ is the reward-like quantity (higher is better) and
$F=V_{\mathrm{cost}}=-\Phi$ is the cost/free-energy potential. The displayed ensemble therefore uses
$+V_{\mathrm{rew}}/T_c=\Phi/T_c=-F/T_c$. This is a conditional modeling convention,
not a consequence of the WFR reaction closure alone.

:::

:::{prf:definition} WFR Reaction Closure: Value Creates Mass
:label: thm-wfr-consistency-value-creates-mass

In the WFR dynamics ({prf:ref}`def-the-wfr-action`, {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`), the reaction rate $r(z)$ in the unbalanced continuity equation is determined by the value function:

$$
r(z) = \frac{1}{s_r} \left( V(z) - \bar{V} \right),

$$
where $\bar{V} = \mathbb{E}_\rho[V]$ is the mean value and $s_r$ is the reaction time scale (computation time).

*Consequence:* The total mass satisfies:

$$
\frac{d}{ds}\int_{\mathcal Z}\rho\,d\mu_G
 = \int_{\mathcal Z}\rho(z,s)r(z,s)\,d\mu_G
 = \frac{1}{s_r}\int_{\mathcal Z}\rho\,(V-\bar V)\,d\mu_G.

$$

Under this closure, the local reaction contribution is positive where $V>\bar V$ and negative
where $V<\bar V$; transport and boundary flux can change the total mass separately.

*Scope.* This is a chosen reaction closure. It becomes a consequence only after a WFR variational
problem, endpoint constraints, and a target density have been specified; the WFR action alone does
not determine $r$ from $V$.

:::

:::{prf:remark} Conditional Conservative Equilibrium
:label: cor-equilibrium-distribution

**Conditional statement:** If the transport drift, reaction law, boundary conditions, and invariant
measure are chosen to satisfy detailed balance with the reward convention, then the stationary
density is the Boltzmann form:

$$
\rho_\infty(z) \propto \exp\left(-\frac{V_{\mathrm{cost}}(z)}{T_c}\right)
 = \exp\left(\frac{\Phi(z)}{T_c}\right),

$$
which is exactly the canonical ensemble (Definition {prf:ref}`def-canonical-ensemble`).

Without those detailed-balance hypotheses, the displayed reaction closure does not by itself imply
stationarity or zero current.

:::

:::{prf:proposition} Conditional Non-Equilibrium Steady State (NESS)
:label: thm-ness-existence

If a stationary solution exists and detailed balance is broken (for example by a nonzero curl
term or boundary drive), it is a **Non-Equilibrium Steady State** satisfying:

1. **Stationarity:** $\partial_s \rho_\infty = 0$
2. **Persistent Current:** The probability current $J = \rho v - D\nabla\rho$ is non-zero and divergence-free: $\nabla \cdot J = 0$ but $J \neq 0$
3. **Entropy Production:** The system continually produces entropy at rate:

$$
\dot{S}_i = \int_{\mathcal{Z}} \frac{\|J\|_G^2}{\rho D} \, d\mu_G > 0

$$

*Remark.* The probability density $\rho_\infty$ is time-independent, but individual trajectories circulate indefinitely. This distinguishes NESS from true equilibrium (where $J = 0$).

:::

:::{prf:proposition} NESS Decomposition
:label: prop-ness-decomposition

The probability current in a NESS decomposes into:

$$
J = J_{\text{gradient}} + J_{\text{cyclic}}

$$
where:
- $J_{\text{gradient}} = -D\rho\nabla\ln\rho + \rho\nabla\Phi$ derives from the scalar potential
- $J_{\text{cyclic}} = \rho \cdot v_{\text{curl}}$ derives from the solenoidal component

At stationarity, $\nabla \cdot J = 0$, but only $J_{\text{gradient}} = 0$ at true equilibrium. NESS has $J_{\text{cyclic}} \neq 0$.

:::

:::{prf:remark} Varentropy as a Temperature-Sensitivity Diagnostic
:label: cor-varentropy-stability

Let $\mathcal{I}(a|z) = -\ln \pi(a|z)$ be the surprisal of an action. Define the **Policy Varentropy** $V_H(z)$ as the variance of the surprisal under the Boltzmann policy:

$$
V_H(z) := \mathrm{Var}_{a \sim \pi}[\mathcal{I}(a|z)] = \mathbb{E}_{\pi}\left[ \left( \ln \pi(a|z) + H(\pi) \right)^2 \right].

$$
*Units:* $\mathrm{nat}^2$.

Under the Boltzmann-Value Law (Axiom {prf:ref}`ax-the-boltzmann-value-law`), the Varentropy equals the **Heat Capacity** $C_v$ of the decision state:

$$
V_H(z) = \beta_{\text{ent}}^2 \mathrm{Var}_\pi[Q] = C_v,

$$
where $\beta_{\text{ent}} = 1/T_c$ is the inverse cognitive temperature. Equivalently:

$$
V_H(z) = T_c \frac{\partial H(\pi)}{\partial T_c}.

$$
**Operational Consequence:**
1. **Thermal Stability:** $V_H$ measures the sensitivity of the agent's exploration strategy to changes in the cognitive temperature $T_c$.
2. A spike can flag sharp temperature sensitivity, but it is not by itself a phase-transition
   theorem.
3. Any annealing-rate condition requires a separate mixing-time or spectral-gap estimate; none is
   implied by the varentropy identity.

:::

:::{prf:definition} Value-Metric Conformal Coupling
:label: def-value-metric-conformal-coupling

We model the effect of Value on the Metric $G$ as a **Conformal Transformation**:

$$
\tilde{G}_{ij}(z) = \Omega^2(z) \cdot G_{ij}(z),

$$
where the conformal factor $\Omega(z)$ depends on the **Hessian of the Value**:

$$
\Omega(z) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(z)\|_{\text{op}},

$$
with $\alpha_{\text{conf}} \ge 0$ the conformal coupling strength and $\|\cdot\|_{\text{op}}$ the operator norm.

Units: $[\Omega] = 1$ (dimensionless), $[\alpha_{\text{conf}}] = \text{length}^2/\mathrm{nat}$.

:::

:::{prf:proposition} Risk-Curvature Mechanism
:label: prop-risk-curvature-mechanism

The conformal factor encodes the local "importance" of the value landscape:

| Value Landscape           | $\lVert\nabla^2 V\rVert$ | $\Omega$    | Effect                           |
|---------------------------|--------------------------|-------------|----------------------------------|
| **Flat** (low importance) | $\approx 0$              | $\approx 1$ | Default hyperbolic bulk geometry |
| **Curved** (ridge/valley) | $\gg 0$                  | $\gg 1$     | Distances expand, mass increases |
| **Saddle** (transition)   | moderate                 | $> 1$       | Intermediate slowdown            |

:::

:::{prf:corollary} Inertia at Critical Regions
:label: cor-inertia-at-critical-regions

Near sharp ridges or valleys of $V$ (where $\|\nabla^2 V\|$ is large), the conformal factor causes:

1. **Inertia Increase:** The effective mass $\tilde{G}(z) = \Omega^2(z) G(z)$ increases, so the agent slows down near critical decision boundaries ({ref}`sec-the-coupled-jump-diffusion-sde` mass scaling).

2. **Resolution Increase:** The capacity-constrained metric allocates more volume to high-curvature regions (Theorem {prf:ref}`thm-capacity-constrained-metric-law`), allowing higher-fidelity representation of value gradients.

3. **Stability:** The agent cannot "rush through" regions of high value curvature—it is forced to carefully navigate decision boundaries.

*Remark (Physical analogy).* The conformal scaling of effective velocity is mathematically analogous to gravitational time dilation in general relativity, where proper time dilates in regions of high gravitational potential.

:::

:::{prf:remark} Conformal Laplacian Transformation
:label: prop-conformal-laplacian-transformation

Under the conformal transformation $G \to \tilde{G} = \Omega^2 G$, the Laplace-Beltrami operator acting on a scalar function $f$ transforms as:

$$
\Delta_{\tilde{G}} f = \Omega^{-2} \left( \Delta_G f + (d-2) \frac{G^{ij} \partial_i \Omega}{\Omega} \partial_j f \right),

$$
where $d$ is the dimension. For the Value function $V$ itself (which determines $\Omega$), this creates a **nonlinear coupling**. The screened Poisson equation (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`) in the conformally modified metric becomes:

$$
-\Delta_{\tilde{G}} V + \tilde{\kappa}^2 V = \tilde{\rho}_r,

$$
with effective screening mass $\tilde{\kappa}^2 = \Omega^{-2} \kappa^2$.

*Remark (Self-Consistency).* Since $\Omega$ depends on $\nabla^2 V$, the equation becomes nonlinear: the geometry adapts to the value landscape which in turn affects the geometry. In practice, we solve this iteratively or treat $\Omega$ as slowly-varying.

*Interpretation:* The displayed coefficient is a coordinate rewriting of the conformally transformed
operator. A self-focusing or longer-range-correlation conclusion requires solving the transformed
boundary-value problem; it does not follow from the rescaling alone.

**Cross-references:** Theorem {prf:ref}`thm-capacity-constrained-metric-law`, {ref}`sec-the-stochastic-action-principle` (Mass=Metric), Proposition {prf:ref}`prop-mass-scaling-near-boundary`.

:::

:::{prf:remark} RL--Electrodynamics Correspondence (Formal Analogy)
:label: thm-rl-as-electrodynamics-on-a-curved-manifold

Under the conservative geodesic Langevin convention of the equations-of-motion chapter, the analogy can be written as:

The agent is a **particle** with:
- **Position** $q \in \mathcal{Z}$ (from Perception / Dirichlet BC)
- **Momentum** $p \in T_q\mathcal{Z}$ (from Action / Neumann BC)
- **Mass** $G(q)$ (the Riemannian metric = information geometry)
- **Potential Energy** $\Phi_{\mathrm{eff}}(q)$ (from the conservative reward/cost potential)
- **External Forces** $u_\pi(q)$ (from Policy / symmetry-breaking kick)

moving according to the **geodesic SDE** (Definition {prf:ref}`def-bulk-drift-continuous-flow`):

$$
dq^k = G^{kj}(q) p_j \, ds, \qquad
dp_k = \left[-\partial_k\Phi_{\mathrm{eff}} - \gamma p_k
+\beta_{\text{curl}}\mathcal{F}_{kj}G^{j\ell}p_\ell
+\Gamma^m_{k\ell}G^{\ell j}p_jp_m + \gamma G_{kj}u_\pi^j\right]ds
+\sqrt{2\gamma T_c}\,(G^{1/2})_{kj}dW^j_s,

$$
on a **curved manifold** with metric $G$ satisfying the **capacity constraint** (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The Christoffel term is the geodesic correction in covector momentum coordinates. Treating the conservative component as a screened Helmholtz solution requires the conservative diffusion hypotheses of {prf:ref}`thm-the-hjb-helmholtz-correspondence`.

This is a physics-inspired correspondence, not an identification theorem. The standard RL components can be
organized using the following field-theory roles:

| RL Component      | Field Theory Role                                         |
|-------------------|-----------------------------------------------------------|
| Encoder           | **Coordinate Chart** (embedding from boundary to bulk)    |
| Critic            | **Field Solver** (Green's function of screened Laplacian) |
| Policy            | **External Force** (symmetry-breaking current)            |
| Discount $\gamma$ | **Screening Mass** (controls correlation length)          |
| Temperature $T_c$ | **Thermal Bath** (fluctuation-dissipation source)         |

:::

:::{prf:remark} Three Interface Roles (Operational)
:label: cor-the-three-boundary-conditions

The agent-environment interface decomposes into exactly three types of boundary conditions:

1. **Dirichlet** (Sensors): Clamp position $q = q_{\text{obs}}$. Information flows **in**.
2. **Neumann** (Motors): Clamp flux $\nabla_n \cdot p = j_{\text{motor}}$. Information flows **out**.
3. **Reward trace/source**: choose either a boundary trace for $\Phi$ or an independently specified Neumann/source
   datum $\sigma_r$. The pullback $\iota^*\mathcal R$ supplies tangential (Dirichlet-type) data and is not itself
   a charge density.

These three conditions fully specify the agent's interaction with its environment.

**Cross-references:** {ref}`sec-the-boundary-interface-symplectic-structure` (Holographic Interface), {ref}`sec-the-equations-of-motion-geodesic-jump-diffusion` (Equations of Motion), {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Geometry).

:::

## 06_fields/03_info_bound.md

:::{prf:definition} Holographic Coefficient
:label: def-holographic-coefficient

The **Holographic Coefficient** $\nu_D$ for a $D$-dimensional latent manifold with $(D-1)$-sphere boundary is:

$$
\nu_D := \frac{(D-1)\,\Omega_{D-1}}{8\pi}

$$

where $\Omega_{D-1} = \frac{2\pi^{D/2}}{\Gamma(D/2)}$ is the surface area of the unit $(D-1)$-sphere.

| $D$ | Boundary | $\Omega_{D-1}$ | $\nu_D$ | Numerical |
|-----|----------|----------------|---------|-----------|
| 2   | Circle ($S^1$) | $2\pi$ | $1/4$ | 0.250 |
| 3   | Sphere ($S^2$) | $4\pi$ | $1$ | 1.000 |
| 4   | Glome ($S^3$) | $2\pi^2$ | $3\pi/4$ | 2.356 |
| 5   | 4-sphere ($S^4$) | $8\pi^2/3$ | $4\pi/3$ | 4.189 |
| 6   | 5-sphere ($S^5$) | $\pi^3$ | $5\pi^2/8$ | 6.169 |
| $D \gg 1$ | Hyper-sphere | $\to 0$ | $\to 0$ | Capacity collapse |

*Remark (Dimensional pressure).* The coefficient $\nu_D$ is non-monotonic: it increases from $D=2$ to a peak near $D \approx 9$ ($\nu_9 \approx 9.45$), then decays to zero as $D \to \infty$. The curse of dimensionality applies to this high-dimensional tail. Dimensional reduction pressure arises beyond the peak; $D \approx 3$ lies on the rising portion of the curve.

*Remark (Physics correspondence).* For $D=2$, we recover the Bekenstein-Hawking coefficient $\nu_2 = 1/4$, making the Causal Information Bound $I_{\max} = \text{Area}/(4\ell_L)$ directly analogous to black hole entropy $S = A/(4\ell_P^2)$.

*Units:* $[\nu_D] = \text{dimensionless}$.

:::

:::{prf:definition} Levin Length
:label: def-levin-length

Let $\eta_\ell$ be the boundary $(D-1)$-volume per nat at resolution $\ell$
(Definition {prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`).
For a declared latent dimension $D\ge2$, define the **Levin Length** by

$$
\ell_L := (\nu_D\eta_\ell)^{1/(D-1)}.
$$

This convention makes $\ell_L^{D-1}$ the boundary volume per nat after the
dimension-dependent normalization $\nu_D$ is fixed.

Units: $[\ell_L]=[z]$ when the boundary measure is expressed in the
corresponding normalized coordinate units.

*Interpretation.* A boundary cell of $(D-1)$-volume
$\ell_L^{D-1}$ carries one nat under this operational normalization. In the
two-dimensional Poincaré-disk convention this reads
$C_\partial=\operatorname{Area}/(4\ell_L)$ because $\nu_2=1/4$.

*Remark (Naming).* The name honors Leonid Levin's foundational work on algorithmic information theory and the universal distribution {cite}`levin1973universal`. The Levin Length represents the floor below which distinctions cannot be computationally meaningful.

:::

:::{prf:definition} Saturation Limit
:label: def-saturation-limit

The agent is at the **Saturation Limit** when the bulk information volume (Definition {prf:ref}`def-a-bulk-information-volume`) equals the boundary capacity (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`):

$$
I_{\text{bulk}} = C_\partial.

$$
At this limit, the DPI constraint $I_{\text{bulk}} \le C_\partial$ is satisfied with equality.

:::

:::{prf:remark} Formal Spherical Saturation Ansatz
:label: lem-metric-divergence-at-saturation

The following Schwarzschild-style expression is a formal radial ansatz for exploring a
capacity-saturation regime:
$$
A(r) = \left( 1 - \frac{2\mu(r)}{(n-2)r^{n-2}}
- \frac{\Lambda_{\mathrm{eff}}r^2}{n(n-1)} \right)^{-1}.
$$
It is not a consequence of the capacity-constrained metric law without an independent
spherically symmetric field equation and boundary-value calculation. In particular, the
Poincare-disk boundary and the $n=2$ case require separate analysis. Treat $G^{rr}\to0$ at a
zero of the displayed denominator as a diagnostic ansatz, not as a theorem about the learned
metric.
:::

:::{prf:definition} Conditional Causal Information Capacity
:label: thm-causal-information-bound

Under an explicit capacity permit that identifies stable representational information with
boundary area at resolution $\ell_L$, define the operational capacity
$$
I_{\max}:=\nu_D\,\frac{\operatorname{Area}(\partial\mathcal Z)}{\ell_L^{D-1}}.
$$
Here $\nu_D$ is the dimension-dependent coefficient defined above and the boundary area is
computed in the selected induced metric. This formula is a modeling convention/diagnostic
normalization; the current metric law does not by itself prove the bulk-to-boundary identity,
the spherical saturation solution, or the Fisher normalization used in the former derivation.

For the $D=2$ normalized convention, $\nu_2=1/4$ and the formula reads
$I_{\max}=\operatorname{Area}(\partial\mathcal Z)/(4\ell_L)$.
Any use of this expression as a theorem must state the additional field equation, boundary
conditions, and dimensional normalization that establish the permit.
:::

:::{prf:proposition} Conditional Radial Causal Stasis
:label: thm-causal-stasis

Assume the conditional capacity formula above, the formal spherical ansatz, bounded radial force,
and $G^{rr}\to0$ at the selected horizon. Then the radial component of an overdamped drift
satisfies
$$
v^r=-G^{rr}\partial_r\Phi_{\mathrm{eff}}\longrightarrow0.
$$
This conclusion controls the radial component in that ansatz. It does not imply
$\|v\|_G\to0$ for the full tensor, nor does it follow from $I_{\mathrm{bulk}}\to I_{\max}$
without the additional hypotheses.
:::

:::{prf:remark} Formal Saturation-Velocity Scaling
:label: cor-saturation-velocity-tradeoff

Let $\eta_{\text{Sch}} := I_{\text{bulk}}/I_{\max}$ be the saturation ratio. If the model additionally identifies
$\eta_{\text{Sch}}=\mu/\mu_{\max}$ at fixed horizon radius, the radial update scales as:

$$
|v^r| \sim (1 - \eta_{\text{Sch}})^{1/2}.

$$
*Scope.* This square-root scaling is a consequence only of the formal radial ansatz and a
specific relation between the saturation ratio and the radial denominator; it is not established
for a general learned metric.

*Former proof sketch.* If one additionally assumes $\eta_{\text{Sch}}=\mu/\mu_{\max}$ and a linear radial denominator,
then $G^{rr}\sim1-\eta_{\text{Sch}}$ and the displayed square-root scaling follows for the selected radial component. This
identification is a modeling assumption, not a consequence of the capacity definition.

At 90% saturation ($\eta_{\text{Sch}} = 0.9$), the radial component is $\sim 32\%$ of its
reference value; at 99% it is $\sim 10\%$. These percentages do not describe angular motion or a general learned metric.

:::

:::{prf:definition} Capacity Horizon Diagnostic
:label: def-capacity-horizon-diagnostic

Compute the **Saturation Ratio**:

$$
\eta_{\text{Sch}}(s) := \frac{I_{\text{bulk}}(s)}{I_{\max}} = \frac{I_{\text{bulk}}(s)}{\nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}},

$$
where:
- $I_{\text{bulk}}(s) = \int_{\mathcal{Z}} \iota_{\mathrm{bulk}}(z,s) \, d\mu_G$ per Definition {prf:ref}`def-a-bulk-information-volume`; any empirical proxy must be calibrated to this quantity
- $\nu_D$ is the Holographic Coefficient (Definition {prf:ref}`def-holographic-coefficient`)
- $D$ is the latent manifold dimension

*Special case (Poincare disk, $D=2$):* $\eta_{\text{Sch}} = 4\ell_L \cdot I_{\text{bulk}} / \text{Area}(\partial\mathcal{Z})$.

*Interpretation:*
- $\eta_{\text{Sch}} < 0.5$: Safe operating regime. Ample capacity headroom.
- $0.5 \le \eta_{\text{Sch}} < 0.9$: Elevated utilization. Monitor for growth trends.
- $0.9 \le \eta_{\text{Sch}} < 0.99$: **Warning setpoint.** Test the radial-stasis hypotheses and monitor the measured update components.
- $\eta_{\text{Sch}} \ge 0.99$: **Critical setpoint.** Investigate the radial-stasis hypotheses and consider a
  conservative remediation; this threshold does not prove that stasis is imminent.

*Cross-reference:* Complements the metric-law CapacitySaturationCheck ({ref}`sec-diagnostic-node-capacity-saturation`)
by providing the velocity-degradation interpretation and connecting to ontological remediation.
:::

## 07_cognition/01_supervised_topo.md

:::{prf:remark} Extension, Not Replacement
:label: rem-extension-not-replacement

{ref}`sec-the-context-space-unified-definition` establishes classification as selecting a context $c \in \mathcal{Y}$ (the label space), with cross-entropy cost $-\log p(y|z)$ (Definition {prf:ref}`def-context-instantiation-functor`). This section specifies the **topological constraints** that enforce geometric coherence of this classification:

1. Charts should be semantically pure (one class per chart, modulo transition regions)
2. Different classes should be metrically separated (long geodesics between class regions)
3. Classification should be stable under dynamics (regions of attraction)

:::

:::{prf:definition} Semantic Partition
:label: def-semantic-partition

Let $\mathcal{Y} = \{1, \ldots, C\}$ be the set of class labels and $\mathcal{K}$ the macro-state register (Definition 2.2.1). A labeling $Y: \mathcal{X} \to \mathcal{Y}$ induces a **soft partition** of the chart atlas:

$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(Y=y \mid K=k) > 1 - \epsilon_{\text{purity}}\},

$$
where $\epsilon_{\text{purity}} \in (0, 0.5)$ is the purity threshold. Define the transition-chart set
\[
\mathcal{T}:=\mathcal{K}\setminus\bigcup_{y\in\mathcal{Y}}\mathcal{A}_y
 = \left\{k:\max_y P(Y=y\mid K=k)\le 1-\epsilon_{\text{purity}}\right\}.
\]

*Interpretation:* $\mathcal{A}_y$ is the **sub-atlas** of charts predominantly associated with class $y$. A chart $k$ belongs to $\mathcal{A}_y$ if, given that a sample routes to chart $k$, the probability of class $y$ exceeds $1 - \epsilon_{\text{purity}}$.

:::

:::{prf:definition} Transition Chart Set
:label: prop-soft-injectivity

For $\epsilon_{\text{purity}}<1/2$, the sets $\mathcal{A}_y$ are pairwise disjoint, because two conditional probabilities cannot both exceed $1-\epsilon_{\text{purity}}>1/2$. The transition charts are the complement $\mathcal{T}$ defined above. They are charts whose dominant class does not meet the selected purity threshold; high conditional entropy is an optional diagnostic, not part of the definition.

*Remark (Geometric Interpretation).* A transition chart may lie near a decision boundary, but this does not by itself make it a saddle or an unstable fixed point; those properties require a specified smooth potential and a dynamical analysis.

**Cross-references:** {ref}`sec-the-context-space-unified-definition` (Context-Conditioned Policies), Definition 2.2.1 (Macro-State Register), {ref}`sec-tier-the-attentive-atlas` (Router Weights).

:::

:::{prf:definition} Class-Conditioned Potential
:label: def-class-conditioned-potential

Given a target class $y \in \mathcal{Y}$ and differentiable soft-router weights $w_k(z)$, define the semantic potential:

$$
V_y(z) := -\beta_{\text{class}} \log\left(\sum_{k=1}^{N_c} w_k(z)P(Y=y \mid K=k)\right) + V_{\text{base}}(z),

$$
where:
- $P(Y=y \mid K=k) = \text{softmax}(\Theta_{k,:})_y$ with learnable parameters $\Theta \in \mathbb{R}^{N_c \times C}$
- $V_{\text{base}}(z)$ is the unconditioned critic ({ref}`sec-the-hjb-correspondence`)
- $\beta_{\text{class}} > 0$ is the **class temperature** (inverse of semantic diffusion)
- Units: $[V_y] = \mathrm{nat}$

*Remark (Chart-to-Class Mapping).* The learnable parameter $\Theta_{k,y}$ represents the log-affinity of chart $k$ for class $y$. The soft mixture is differentiable in $z$; replacing $w_k(z)$ by a hard chart index would make the semantic term piecewise constant and remove its class-dependent continuous drift.

*Remark (Alternative: Empirical Estimation).* Instead of learnable parameters, one may estimate $P(Y|K)$ empirically via exponential moving average:

$$
\hat{P}(Y=y \mid K=k) = \frac{\text{EMA}[\mathbb{I}[Y=y, K=k]]}{\text{EMA}[\mathbb{I}[K=k]]}.

$$
This is non-differentiable w.r.t. chart assignment but more grounded in observations. A hybrid approach initializes learnable $\Theta$ from empirical estimates after warmup.

:::

:::{prf:definition} Region of Attraction
:label: def-region-of-attraction

The **region of attraction** for class $y$ is:

$$
\mathcal{B}_y := \left\{z \in \mathcal{Z}: \lim_{t\to\infty}\phi_t(z)\ \text{exists and}\ K\!\left(\lim_{t\to\infty}\phi_t(z)\right)\in\mathcal{A}_y\right\},

$$
where $\phi_t$ denotes the flow of the curl-corrected system

$$
\dot{z} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z)\nabla V_y(z)\right), \qquad \mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}
$$
(conservative case: $\mathcal{F}=0$).
Here $\nabla V_y$ is the ordinary differential of the smooth semantic potential. A non-conservative reward
component can be included through the separately defined curl mobility; the class-conditioning statement below
is made in the conservative case $\mathcal{F}=0$.

*Interpretation:* $\mathcal{B}_y$ is the set of initial conditions from which the deterministic gradient flow on $V_y$ converges to the class-$y$ region.

:::

:::{prf:proposition} Conditional Classification Relaxation
:label: thm-classification-as-relaxation

Under the conservative, deterministic overdamped dynamics ({ref}`sec-the-overdamped-limit`) with the smooth potential $V_y$:

$$
dz = -G^{-1}(z)\nabla V_y(z)\,ds, \qquad T_c=0.

$$
The limiting chart assignment satisfies, whenever the trajectory converges:

$$
K\!\left(\lim_{s \to \infty} z(s)\right) \in \mathcal{A}_y,

$$
provided:
1. $z(0) \in \mathcal{B}_y$ (initial condition in the basin)
2. the trajectory remains in the domain of the smooth router and converges to a local minimum of $V_y$
3. that limiting minimum lies in $K^{-1}(\mathcal{A}_y)$.

:::

:::{prf:corollary} Inference via Relaxation
:label: cor-inference-via-relaxation

Classification inference proceeds as:
1. Encode: $z_0 = \text{Enc}(x)$
2. Relax under neutral potential $V_{\text{base}}$ (no class conditioning) to equilibrium $z^*$
3. Read out: $\hat{y} = \arg\max_y P(Y=y \mid K(z^*))$

*Remark (Fast Path).* In practice, we often skip the relaxation and use direct readout: $\hat{y} = \arg\max_y \sum_k w_k(x) \cdot P(Y=y \mid K=k)$, where $w_k(x)$ are the router weights ({ref}`sec-tier-the-attentive-atlas` (Router Weights)). This is an operational classifier; identifying it with a zero-temperature, infinite-time limit requires additional convergence and calibration assumptions.

**Cross-references:** {ref}`sec-the-overdamped-limit` (Overdamped Limit), Definition {prf:ref}`def-effective-potential`, {ref}`sec-the-hjb-correspondence` (Critic).

:::

:::{prf:definition} Class-Consistent Jump Rate
:label: def-class-consistent-jump-rate

For the WFR reaction term (Definition {prf:ref}`def-the-wfr-action`), modulate the inter-chart transition rate:

$$
\lambda_{i \to j}^{\text{sup}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{class}}(i, j)\right),

$$
where:
- $\lambda^{(0)}_{i \to j}$ is the **base transition rate** from the GKSL master equation ({prf:ref}`def-gksl-generator`, {cite}`lindblad1976gksl,gorini1976gksl`, {ref}`sec-connection-to-gksl-master-equation`), derived from the overlap consistency of jump operators (Section 7.13)
- $\gamma_{\text{sep}} \geq 0$ is the **separation strength** (hyperparameter)
- $D_{\text{class}}(i, j) = \mathbb{I}[\text{Class}(i) \neq \text{Class}(j)]$ is the class disagreement indicator
- $\text{Class}(k) := \arg\max_y P(Y=y \mid K=k)$ is the dominant class of chart $k$

*Remark (Rate vs Operator).* {ref}`sec-factorized-jump-operators-efficient-chart-transitions` defines the **transition function** $L_{i \to j}$ (the coordinate change map). The **transition rate** $\lambda_{i \to j}$ is a separate quantity from the GKSL/master equation framework ({ref}`sec-connection-to-gksl-master-equation`, Equation 20.5.2) that governs *how often* jumps occur, not *where* they go. The rate is typically derived from the overlap structure: $\lambda_{i \to j}^{(0)} \propto \mathbb{E}_{x}[w_i(x) w_j(x)]$, measuring how much probability mass lies in the overlap $U_i \cap U_j$.

*Interpretation:* Transitions between charts of the same class proceed at the base rate $\lambda^{(0)}$. Transitions between charts of different classes are exponentially suppressed by factor $e^{-\gamma_{\text{sep}}}$.

:::

:::{prf:remark} Transition-Rate Suppression Diagnostic
:label: prop-effective-disconnection

Increasing $\gamma_{\text{sep}}$ suppresses the prescribed cross-class jump rates:

$$
\frac{\lambda_{i\to j}^{\text{sup}}}{\lambda_{i\to j}^{(0)}}=e^{-\gamma_{\text{sep}}}
\quad\text{when }\operatorname{Class}(i)\ne\operatorname{Class}(j).

$$
This is a rate-suppression diagnostic. It does not imply that the WFR distance diverges: WFR paths include
reaction controls, and a pure-reaction path can have finite Hellinger-type cost independently of
$\gamma_{\text{sep}}$. A geometric disconnection claim would require a different metric whose admissible paths
explicitly exclude that reaction channel.

:::

:::{prf:remark} Tunneling as Anomaly Detection
:label: rem-tunneling-as-anomaly-detection

Cross-class transitions are not forbidden, merely exponentially suppressed. A detected cross-class jump indicates:

1. **Anomaly:** The sample lies in a transition region not well-covered by training
2. **Distribution shift:** The test distribution differs from training
3. **Adversarial input:** Deliberate perturbation to cross class boundaries

This provides a natural **out-of-distribution detection** mechanism: monitor the rate of cross-class transitions.

:::

:::{prf:definition} Class-Modulated Jump Operator
:label: def-class-modulated-jump-operator

Modify the jump operator (Definition {prf:ref}`def-factorized-jump-operator`) to incorporate class consistency:

```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,    # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor, # [N_c, C] learnable logits
    gamma_sep: float = 5.0,       # Separation strength
) -> torch.Tensor:
    """
    Compute class-modulated jump rates.

    Cross-ref:
        - Definition 25.3.1 (Class-Consistent Jump Rate)
        - Definition 7.13.1 (Jump Operator)
    """
    # Get dominant class per chart
    p_y_given_k = F.softmax(chart_to_class, dim=1)  # [N_c, C]
    dominant_class = p_y_given_k.argmax(dim=1)       # [N_c]

    # Compute class disagreement matrix
    class_match = (dominant_class.unsqueeze(1) == dominant_class.unsqueeze(0)).float()  # [N_c, N_c]
    D_class = 1.0 - class_match  # 1 if classes differ, 0 if same

    # Modulate rates
    lambda_sup = lambda_base * torch.exp(-gamma_sep * D_class)

    return lambda_sup
```

**Cross-references:** {ref}`sec-the-wfr-metric` (WFR Metric), Definition {prf:ref}`def-factorized-jump-operator`, {ref}`sec-connection-to-gksl-master-equation` (GKSL Connection).

:::

:::{prf:definition} Purity Loss
:label: def-purity-loss

The purity loss measures how well charts separate classes:

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k),

$$
where:
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ is the marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ is the class entropy within chart $k$

*Interpretation:* $\mathcal{L}_{\text{purity}} = H(Y \mid K)$, the conditional entropy of class given chart. Minimizing this encourages each chart to be associated with a single class.

:::

:::{prf:proposition} Purity-Information Duality
:label: prop-purity-information-duality

Minimizing $\mathcal{L}_{\text{purity}}$ is equivalent to maximizing the mutual information $I(K; Y)$:

$$
\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y).

$$
Since $H(Y)$ is fixed by the data, $\min \mathcal{L}_{\text{purity}} \Leftrightarrow \max I(K; Y)$.

:::

:::{prf:definition} Balance Loss
:label: def-balance-loss

Prevent degenerate solutions where all samples route to few charts:

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right),

$$
where $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ is the average router weight vector.

*Interpretation:* Encourages all charts to be used, preventing "dead charts" and ensuring the atlas covers the label space.

:::

:::{prf:definition} Contrastive Loss
:label: def-contrastive-loss

Enforce that different-class samples have low router overlap, using the following bounded proxy for separation:

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} (w_i^\top w_j)\,\max\!\left(0, m - \left(1-w_i^\top w_j\right)\right)^2,

$$
where:
- $\mathcal{P}$ is the set of sample pairs in the batch
- $w_i, w_j$ are router weight vectors
- $m \in (0,1)$ is the margin for the bounded proxy
- $1-w_i^\top w_j$ is a router-overlap proxy, not a WFR or jump distance

*Interpretation:* If two samples have different labels but high router overlap ($w_i^\top w_j$ large), the bounded proxy is below the margin and the loss penalizes the configuration. A genuine geometric distance would require a separately defined jump-cost estimator.

:::

:::{prf:definition} Route Alignment Loss
:label: def-route-alignment-loss

The primary classification loss:

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right],

$$
where $\text{CE}$ denotes cross-entropy.

*Interpretation:* The predicted class distribution is the router-weighted average of per-chart class distributions. This must match the true label.

:::

:::{prf:definition} Total Loss
:label: def-total-loss

The full supervised topology loss:

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.

$$
Typical hyperparameters: $\lambda_{\text{pur}} = 0.1$, $\lambda_{\text{bal}} = 0.01$, $\lambda_{\text{met}} = 0.01$.

**Algorithm 25.4.7 (SupervisedTopologyLoss Implementation).**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class SupervisedTopologyLoss(nn.Module):
    """
    Supervised topology loss enforcing chart purity, balance, and separation.

    Cross-ref:
        - Definition 25.4.6 (Total Loss)
        - {ref}`sec-tier-the-attentive-atlas` (Router Weights)
    """

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin

        # Learnable chart-to-class mapping (Definition 25.2.1)
        self.chart_to_class = nn.Parameter(
            torch.randn(num_charts, num_classes) * 0.01
        )
        self.temperature = temperature

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: torch.Tensor,  # [B, N_c]
        y_true: torch.Tensor,          # [B] class labels
        z_latent: torch.Tensor = None, # [B, D] optional for metric loss
    ) -> Dict[str, torch.Tensor]:
        """
        Compute supervised topology losses.

        Returns dict with individual losses and total.
        """
        B = router_weights.shape[0]
        p_y_k = self.p_y_given_k  # [N_c, C]

        # === Route Alignment Loss (Definition 25.4.5) ===
        # P(Y|x) = sum_k w_k(x) * P(Y|K=k)
        p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
        loss_route = F.cross_entropy(
            torch.log(p_y_x + 1e-8), y_true
        )

        # === Purity Loss (Definition 25.4.1) ===
        # Estimate P(Y|K) from the labelled batch, rather than from the
        # learnable chart-to-class prior used by the route predictor.
        p_k = router_weights.mean(dim=0)  # [N_c]
        y_one_hot = F.one_hot(y_true, num_classes=self.num_classes).to(router_weights.dtype)
        p_y_given_k_emp = torch.matmul(router_weights.t(), y_one_hot)
        p_y_given_k_emp = p_y_given_k_emp / (p_k.unsqueeze(1) + 1e-8)
        entropy_per_chart = -(p_y_given_k_emp * torch.log(p_y_given_k_emp + 1e-8)).sum(dim=1)
        # L_purity = sum_k P(K=k) * H(Y|K=k)
        loss_purity = (p_k * entropy_per_chart).sum()

        # === Balance Loss (Definition 25.4.3) ===
        # KL(p_k || Uniform) = sum_k p_k * log(p_k / (1/N_c)) = sum_k p_k * (log(p_k) + log(N_c))
        uniform = torch.ones_like(p_k) / self.num_charts
        # Manual KL computation: KL(P||Q) = sum P * log(P/Q)
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()

        # === Metric Contrastive Loss (Definition 25.4.4) ===
        loss_metric = torch.tensor(0.0, device=router_weights.device)
        if self.lambda_metric > 0 and B > 1:
            # Router overlap as proxy for proximity
            # w_i^T w_j measures routing similarity
            overlap = torch.matmul(router_weights, router_weights.t())  # [B, B]

            # Class disagreement mask
            y_match = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).float()
            y_diff = 1.0 - y_match  # 1 if different classes

            # Penalize high overlap for different-class pairs
            # Use bounded router-overlap proxy; this is not a WFR distance.
            pseudo_dist = 1.0 - overlap
            hinge = F.relu(self.margin - pseudo_dist)
            loss_metric = (y_diff * overlap * hinge ** 2).sum() / (y_diff.sum() + 1e-8)

        # === Total Loss ===
        loss_total = (
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return {
            'loss_total': loss_total,
            'loss_route': loss_route,
            'loss_purity': loss_purity,
            'loss_balance': loss_balance,
            'loss_metric': loss_metric,
        }
```

**Cross-references:** {ref}`sec-tier-the-attentive-atlas` (Router Weights), Section 7.13 (Jump Operators), {ref}`sec-diagnostics-stability-checks` (Diagnostic Nodes).

:::

:::{prf:remark} Connection to Mobius Re-centering
:label: rem-connection-to-m-bius-re-centering

The Möbius re-centering $\phi_c$ (Definition {prf:ref}`def-mobius-translation`) can be
used to center a conditioned generation run at the **class centroid** defined
below. It is a coordinate change; it does not alter the potential or prove
that generated samples have the requested label:

$$
c_y := \arg\min_{c\in\mathbb D}
\sum_{x:Y(x)=y}d_{\mathbb D}(c,\operatorname{Enc}(x))^2,

$$
i.e., the Fréchet mean of the encoded class-$y$ samples. The associated
coordinate map is $\phi_{c_y}(z):=(-c_y)\oplus z$; conditioned generation
may initialize at $c_y$ and express subsequent coordinates relative to it.

:::

:::{prf:remark} Class-Conditioned Langevin Model
:label: prop-class-conditioned-langevin

The conservative overdamped Langevin equation from Theorem
{prf:ref}`thm-overdamped-limit` with class conditioning is, in computation
time,

$$
dz^k = \left[-G^{k\ell}\partial_\ell V_y
-T_cG^{ij}\Gamma^k_{ij}\right]ds
+\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW_s^j,

$$
where $V_y$ is the class-conditioned potential (Definition {prf:ref}`def-class-conditioned-potential`). This is a
formal conditional model; an invariant Gibbs law requires compatible drift, boundary conditions, and volume
corrections for the chosen manifold convention.

*Interpretation:* To generate a sample of class $y$, we run Langevin dynamics with the $V_y$ potential. The semantic term $-\beta_{\text{class}} \log P(Y=y \mid K)$ biases the flow toward class-$y$ charts.

:::

:::{prf:remark} Label as a Conditional Symmetry-Breaking Field {cite}`ho2022cfg`
:label: cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance

Assume the conservative model, a differentiable router, and

$$
dV_{\mathrm{base}}(0)=0,\qquad A(0)=0,
$$

so that the origin is a critical point of the unconditioned potential. Put
$q_y(z):=\sum_k w_k(z)P(Y=y\mid K=k)$ and assume $q_y(0)>0$. The class label
breaks the rotational symmetry at the origin precisely when $dq_y(0)\ne0$:

$$
dV_y(0) = -\beta_{\mathrm{class}}\,\frac{dq_y(0)}{q_y(0)}.
$$

With $G(0)$ positive definite, the corresponding metric gradient is nonzero
if and only if $dq_y(0)\ne0$; only under that non-orthogonality condition
does the initial deterministic drift acquire a class-dependent direction.

:::

:::{prf:definition} Class Centroid in Poincare Disk
:label: def-class-centroid-in-poincar-disk

For the Poincare disk embedding {cite}`nickel2017poincare,ganea2018hnn`, define the class centroid using the **Fréchet mean** {cite}`lou2020frechet`:

$$
c_y := \arg\min_{c \in \mathbb{D}} \sum_{x: Y(x)=y} d_{\mathbb{D}}(c, \text{Enc}(x))^2.

$$
Under the usual finite second-moment assumption, the complete negatively
curved disk has a unique Fréchet mean. This $c_y$ is the centroid used by the
Möbius map in Definition {prf:ref}`def-mobius-translation`.

**Cross-references:** {ref}`sec-policy-control-field` (Langevin Dynamics),
Definition {prf:ref}`def-mobius-translation`, and Definition
{prf:ref}`prop-so-d-symmetry-at-origin`.

:::

:::{prf:remark} Integration with TopologicalDecoder
:label: rem-integration-with-topologicaldecoder

The TopologicalDecoder ({ref}`sec-decoder-architecture-overview-topological-decoder`) receives the geometric content $z_{\text{geo}} = e_K + z_n$ and routes through chart-specific projectors. For class-conditioned generation:

1. **Class determines charts:** The class label $y$ biases chart selection toward $\mathcal{A}_y$ via the semantic potential $V_y$
2. **Decoder routing:** The TopologicalDecoder's inverse router ({ref}`sec-topological-decoder-module`) can either:
   - Accept an explicit chart index $K$ (from the generative flow)
   - Infer routing from $z_{\text{geo}}$ (autonomous mode)
3. **Consistency constraint:** The decoder's inferred routing should agree with the encoder's class-conditioned routing:

   $$
   \mathcal{L}_{\text{route-consistency}} = \mathbb{E}_{x,y}\left[\text{CE}\left(w_{\text{dec}}(z_{\text{geo}}), w_{\text{enc}}(x)\right)\right]

   $$
   where $w_{\text{dec}}$ are the decoder's soft router weights and $w_{\text{enc}}$ are the encoder's.

This ensures that class-conditioned generation produces samples that the encoder would classify correctly---a form of **cycle consistency** between encoding and decoding under the semantic topology.

:::

:::{prf:definition} Hierarchical Labels
:label: def-hierarchical-labels

A **label hierarchy** is a sequence of label spaces:

$$
\mathcal{Y}_L \xrightarrow{\,\pi_L\,}\mathcal{Y}_{L-1}
\xrightarrow{\,\pi_{L-1}\,}\cdots
\xrightarrow{\,\pi_1\,}\mathcal{Y}_0,

$$
where each $\pi_\ell:\mathcal{Y}_\ell\twoheadrightarrow\mathcal{Y}_{\ell-1}$
is a surjective coarsening map. $\mathcal{Y}_0$ contains coarse labels
(super-categories), while $\mathcal{Y}_L$ contains fine labels (leaf
categories).

*Example:* $\mathcal{Y}_0 = \{\text{Animal}, \text{Vehicle}\}$, $\mathcal{Y}_1 = \{\text{Dog}, \text{Cat}, \text{Car}, \text{Bike}\}$, $\mathcal{Y}_2 = \{\text{Terrier}, \text{Poodle}, \ldots\}$.

:::

:::{prf:proposition} Scale-Label Alignment
:label: prop-scale-label-alignment

In the stacked TopoEncoder ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`), enforce purity at each scale:

- **Layer 0 (Bulk/Slow):** Charts at level 0 correspond to coarse classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(0)} = H(\mathcal{Y}_0 \mid K^{(0)})

  $$
- **Layer $\ell$ (Intermediate):** Charts at level $\ell$ correspond to level-$\ell$ classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(\ell)} = H(\mathcal{Y}_\ell \mid K^{(\ell)})

  $$
- **Layer $L$ (Boundary/Fast):** Charts at level $L$ correspond to fine classes. Enforce:

  $$
  \mathcal{L}_{\text{purity}}^{(L)} = H(\mathcal{Y}_L \mid K^{(L)})

  $$
:::

:::{prf:remark} Renormalization Group Interpretation
:label: rem-renormalization-group-interpretation

The semantic hierarchy matches the physical renormalization scale:

| Scale                | Latent Structure              | Semantic Structure |
|----------------------|-------------------------------|--------------------|
| Bulk (Layer 0)       | Slow modes, large wavelengths | Super-categories   |
| Intermediate         | Medium modes                  | Categories         |
| Boundary (Layer $L$) | Fast modes, fine details      | Sub-categories     |

This is the **semantic RG flow**: coarse-graining in the label space corresponds to flowing toward the bulk in latent space.

:::

:::{prf:definition} Hierarchical Supervised Loss
:label: def-hierarchical-supervised-loss

The total hierarchical loss:

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right),

$$
where $\alpha_\ell$ weights the contribution of each scale (typically $\alpha_\ell = 1$ or decaying with $\ell$).

**Cross-references:** {ref}`sec-stacked-topoencoders-deep-renormalization-group-flow` (Stacked TopoEncoder), Definition {prf:ref}`def-the-peeling-step`, {ref}`sec-rigorous-interpretation-renormalization-group-flow` (RG Interpretation).

:::

## 07_cognition/02_governor.md

:::{prf:remark} Extending {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`
:label: rem-extending-section

{ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration` introduces three methods for adaptive multiplier tuning:
- **3.5.A (Primal-Dual):** $\lambda_{t+1} = \Pi[\lambda_t + \eta_\lambda (C(\theta_t) - \epsilon)]$ — linear, memoryless
- **3.5.B (PID):** $\lambda_{t+1} = K_p e_t + K_i \sum e + K_d \Delta e$ — hand-tuned temporal filter
- **3.5.C (Learned Precisions):** $\lambda_i = \exp(-s_i)$ — diagonal covariance, no temporal structure

Each method addresses a specific failure mode but lacks generality. The **Universal Governor** subsumes all three as special cases of a learned temporal policy over the diagnostic stream.

:::

:::{prf:definition} The Meta-Control Problem
:label: def-the-meta-control-problem

Let $\theta_t \in \mathcal{M}_\Theta$ be the agent parameters at training step $t$. The meta-control problem is: find a policy $\pi_{\mathfrak{G}}$ that selects hyperparameters $\Lambda_t$ to minimize task loss while satisfying the Sieve constraints.

**Cross-references:** {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration` (Adaptive Multipliers), Section 3.4 (Joint Optimization).

:::

:::{prf:definition} Uncontrolled Dynamics
:label: def-uncontrolled-dynamics

Standard gradient descent defines a discrete flow on $\mathcal{M}_\Theta$:

$$
\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{task}}(\theta_t),

$$
where $\eta > 0$ is the step size.

Units: $[\theta] = \text{parameter units}$, $[\eta] = \text{step}^{-1}$, $[\nabla\mathcal{L}] = \text{nat} \cdot [\theta]^{-1}$.

:::

:::{prf:definition} Constrained Dynamics
:label: def-constrained-dynamics

The Fragile Agent imposes $K$ constraints $\{C_k(\theta) \leq 0\}_{k=1}^K$ defined by the Sieve ({ref}`sec-theory-thin-interfaces`). Each $C_k$ corresponds to a diagnostic node:

$$
C_k(\theta) = \text{Node}_k(\theta) - \epsilon_k,

$$
where $\epsilon_k$ is the tolerance threshold. The learning dynamics must satisfy these constraints throughout training.

:::

:::{prf:definition} Controlled Update Law
:label: def-controlled-update-law

The controlled update with adaptive multipliers is:

$$
\theta_{t+1} = \theta_t - \eta_t \left( G^{-1}(\theta_t) \nabla \mathcal{L}_{\text{task}}(\theta_t) + \sum_{k=1}^K \lambda_{k,t} \nabla C_k(\theta_t) \right),

$$
where:
- $G(\theta)$ is the parameter-space metric (cf. natural gradient, {ref}`sec-second-order-sensitivity-value-defines-a-local-metric`)
- $\eta_t$ is the adaptive learning rate
- $\lambda_{k,t} \geq 0$ are the constraint multipliers

Units: $[\lambda_k] = \text{dimensionless}$.

*Remark (Natural Gradient Connection).* The factor $G^{-1}$ applies preconditioning analogous to Fisher Information in natural gradient methods {cite}`amari1998natural`. This ensures updates are measured in information-geometric units rather than Euclidean units.

**Cross-references:** {ref}`sec-second-order-sensitivity-value-defines-a-local-metric` (State-Space Metric), Section 3.1 (Diagnostic Nodes).

:::

:::{prf:definition} Diagnostic State Space
:label: def-diagnostic-state-space

The Governor observes the **Sieve Residuals** via the constraint evaluation map $\Psi: \mathcal{M}_\Theta \to \mathbb{R}^K$:

$$
s_t = \Psi(\theta_t) = [C_1(\theta_t), \ldots, C_K(\theta_t)]^\top.

$$
The components of $s_t$ are the normalized defect functionals corresponding to diagnostic nodes 1–41 ({ref}`sec-theory-thin-interfaces`). Positive values indicate constraint violation.

Units: $[s_t] = \text{nat}$ (for entropy-based nodes) or dimensionless (for normalized defects).

:::

:::{prf:definition} The Universal Governor
:label: def-the-universal-governor

The Governor is a policy $\pi_{\mathfrak{G}}: \mathbb{R}^{K \times H} \to \mathbb{R}_+^{K+2}$ mapping the history of Sieve residuals to control inputs:

$$
\Lambda_t = \pi_{\mathfrak{G}}(s_t, s_{t-1}, \ldots, s_{t-H}; \phi),

$$
where:
- $\Lambda_t = (\eta_t, \lambda_{1,t}, \ldots, \lambda_{K,t}, T_{c,t}) \in \mathbb{R}_+^{K+2}$, where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`)
- $\phi$ are the learnable parameters of the Governor
- $H$ is the history horizon (temporal context)

Units: $[\eta_t] = \text{step}^{-1}$, $[\lambda_{k,t}] = \text{dimensionless}$, $[T_{c,t}] = \text{nat}$.

*Remark (Temporal Processing).* The Governor processes a window of $H$ diagnostic snapshots. This enables detection of first and second differences $\Delta s_t$, $\Delta^2 s_t$, which are required for PID-like control (Proposition {prf:ref}`prop-subsumption-of-section`).

:::

:::{prf:proposition} Subsumption of {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration`
:label: prop-subsumption-of-section

The methods of {ref}`sec-adaptive-multipliers-learned-penalties-setpoints-and-calibration` are recovered as special cases of $\pi_{\mathfrak{G}}$:

| Method                     | Governor Instantiation                                                       |
|----------------------------|------------------------------------------------------------------------------|
| Primal-Dual (3.5.A)        | $\pi_{\mathfrak{G}}(s_t) = \lambda_{t-1} + \eta_\lambda s_t$ (affine, $H=1$) |
| PID (3.5.B)                | Linear filter with fixed $(K_p, K_i, K_d)$, $H \geq 2$                       |
| Learned Precisions (3.5.C) | Diagonal, no temporal dependence, $H=0$                                      |

:::

:::{prf:definition} Inner Problem: Agent Optimization
:label: def-inner-problem-agent-optimization

Given fixed control $\Lambda$, the agent minimizes the regularized objective:

$$
\theta^*(\Lambda) = \arg\min_{\theta} \left[ \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \lambda_k C_k(\theta) \right].

$$
:::

:::{prf:definition} Outer Problem: Governor Optimization
:label: def-outer-problem-governor-optimization

The Governor minimizes the **Training Regret** over the distribution of tasks $\mathcal{T}$:

$$
J(\phi) = \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[ \sum_{t=0}^T \left( \mathcal{L}_{\text{task}}(\theta_t) + \gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2 \right) \right],

$$
subject to: $\theta_{t+1} = \Phi(\theta_t, \pi_{\mathfrak{G}}(\Psi(\theta_t); \phi))$.

Units: $[J] = \text{nat}$, $[\gamma_{\text{viol}}] = \text{dimensionless}$.

The outer objective penalizes cumulative task loss (convergence speed) and squared constraint violations (feasibility). The weight $\gamma_{\text{viol}}$ trades off these two objectives.

:::

:::{prf:theorem} Bilevel Structure
:label: thm-bilevel-structure

The training of the Universal Governor has bilevel structure:

$$
\min_\phi \; J(\phi) \quad \text{s.t.} \quad \theta_t = \theta_t(\Lambda_{0:t-1}), \quad \Lambda_t = \pi_{\mathfrak{G}}(s_{t:t-H}; \phi).

$$
The inner problem (agent learning) depends on the outer variables (Governor parameters) through the control sequence $\{\Lambda_t\}$.

*Remark (Gradient Computation).* Computing $\nabla_\phi J$ requires differentiating through the entire training trajectory. In practice, we use truncated backpropagation through time or evolutionary strategies.

**Cross-references:** {ref}`sec-joint-optimization` (Joint Optimization).

:::

:::{prf:definition} Training Lyapunov Function
:label: def-training-lyapunov-function

Define the candidate Lyapunov function for the training dynamics:

$$
V_{\mathfrak{L}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_{k=1}^K \frac{\mu_k}{2} \max(0, C_k(\theta))^2,

$$
where $\mu_k > 0$ are penalty weights for constraint violations.

Units: $[V_{\mathfrak{L}}] = \text{nat}$, $[\mu_k] = \text{dimensionless}$.

$V_{\mathfrak{L}}$ is the augmented Lagrangian with quadratic penalty. If $\Delta V_{\mathfrak{L}} < 0$ along the training trajectory, training converges (Theorem {prf:ref}`thm-stable-training-trajectory`).

:::

:::{prf:theorem} Stable Training Trajectory
:label: thm-stable-training-trajectory

If the Governor $\pi_{\mathfrak{G}}$ selects $\Lambda_t$ such that:

$$
\Delta V_{\mathfrak{L}} := V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t) < 0 \quad \forall t \text{ where } \theta_t \notin \Omega,

$$
then the training process converges to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. Under standard regularity (twice-differentiable $\mathcal{L}$, LICQ), $\Omega$ consists of KKT points.

:::

:::{prf:corollary} Existence of Descent Direction
:label: cor-existence-of-descent-direction

At any non-stationary point $\theta$ where LICQ holds (the gradients $\{\nabla C_k : C_k(\theta) = 0\}$ for active constraints are linearly independent), there exist multipliers $\lambda_k \geq 0$ and step size $\eta > 0$ such that $\Delta V_{\mathfrak{L}} < 0$.

:::

:::{prf:corollary} The Varentropy Brake (Annealing Safety Margin)
:label: cor-varentropy-brake

The training process involves lowering $T_c$ (annealing) to converge on a Nash equilibrium. The stability of this process is governed by the Varentropy (Corollary {prf:ref}`cor-varentropy-stability`).

For the optimization trajectory to remain in the basin of attraction of the global minimum, the cooling schedule must be modulated by the Varentropy:

$$
\frac{d T_c}{dt} = - \eta \cdot \frac{T_c}{1 + \gamma V_H(\theta_t)},

$$
where $\eta, \gamma > 0$ are constants.

*Units:* $[\dot{T}_c] = \mathrm{nat}/[\text{time}]$.

**Mechanism:**
- When $V_H(\theta_t)$ is high (system is near a critical decision point/ridge), the effective cooling rate $\dot{T}_c \to 0$. The Governor "freezes" the temperature to allow the agent to resolve the bifurcation via exploration rather than collapsing into a random mode.
- This prevents **Spontaneous Symmetry Breaking** errors where rapid cooling locks the agent into a suboptimal local minimum.

:::

:::{prf:proposition} Structure of Diagnostic Inputs
:label: prop-structure-of-diagnostic-inputs

The input to the Governor, $s_t = \Psi(\theta_t)$, consists of quantities that depend only on the learned representations, not on the raw data $\mathcal{D}$:
- Entropies: $H(K)$, $H(Y|K)$, $I(K;X)$
- Spectral norms: $\|\nabla_A V\|$, $\lambda_{\max}(G)$
- Curvatures: $\|\nabla^2 V\|$, $R_{\text{Ric}}$

These are computed from the model's internal state $\theta_t$ and its outputs on training batches.

*Example:* Codebook collapse is diagnosed by $H(K) \to 0$. The correction (increase VQ commitment loss $\beta$) depends only on the diagnostic value, not on whether the data is images, audio, or tabular.

:::

:::{prf:proposition} Transfer via Meta-Generalization
:label: prop-transfer-via-meta-generalization

Under the conditions of the Meta-Generalization Metatheorem (**MT: Meta-Generalization** in `metalearning.md`), the Governor $\pi_{\mathfrak{G}}$ trained on a distribution of optimization landscapes $\mathcal{S}$ generalizes to new systems drawn from $\mathcal{S}$.

Specifically, if:
1. **Compact structural manifold:** The optimal diagnostic-to-correction mappings $\{\phi^*(S) : S \in \text{supp}(\mathcal{S})\}$ lie on a compact $C^1$ submanifold of the policy space
2. **Uniform local strong convexity:** The training regret $J(\phi)$ satisfies $c\,\text{dist}(\phi, \mathcal{M})^2 \leq J(\phi) \leq C\,\text{dist}(\phi, \mathcal{M})^2$ near the optimal manifold
3. **Lipschitz continuity:** The regret is Lipschitz in both the policy parameters and the training landscape

Then, with probability at least $1 - \delta$, a Governor trained on $N$ sampled landscapes satisfies:

$$
\mathbb{E}_{S \sim \mathcal{S}}[J_S(\hat{\phi}_N)] \leq C_1\left(\varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}}\right)

$$
where $\varepsilon_N$ is the optimization accuracy.

:::

:::{prf:proposition} Dimensional Analysis
:label: prop-dimensional-analysis

All inputs to $\pi_{\mathfrak{G}}$ are either:
1. **Dimensionless ratios:** $\nu_{\text{cap}} = I_{\text{bulk}}/C_\partial$
2. **Entropies:** measured in nats
3. **Normalized defects:** $(C_k - \epsilon_k)/\epsilon_k$

All outputs are either dimensionless (multipliers $\lambda_k$) or have standard units ($\eta$ in step$^{-1}$, $T_c$ in nat). This ensures the Governor's function approximator operates in a well-conditioned, scale-invariant regime.

:::

:::{prf:definition} Canonical Obstruction Suite
:label: def-canonical-obstruction-suite

A distribution of synthetic optimization landscapes $\{\mathcal{L}_{\text{syn}}^{(i)}\}$ constructed to elicit specific failure modes:

| Obstruction            | Hessian Property                          | Failure Mode            | Diagnostic Signal                              | Required Correction                        |
|------------------------|-------------------------------------------|-------------------------|------------------------------------------------|--------------------------------------------|
| **Rosenbrock Valley**  | $\kappa(\nabla^2\mathcal{L}) \gg 1$       | Oscillation             | High $\lVert\nabla\mathcal{L}\rVert$ variance  | Reduce $\eta$ (gain scheduling)            |
| **Saddle Point**       | $\lambda_{\min}(\nabla^2\mathcal{L}) < 0$ | Stagnation              | Low $\lVert\nabla\mathcal{L}\rVert$, flat loss | Increase $T_c$ (entropy injection)         |
| **Disconnected Modes** | Multimodal landscape                      | Mode collapse           | $H(K) \to 0$                                   | Increase jump rate $\lambda_{\text{jump}}$ |
| **Noise Floor**        | High aleatoric uncertainty                | Overfitting             | $I(K; Z_{\text{tex}}) > 0$                     | Texture firewalling                        |
| **Constraint Cliff**   | Sharp constraint boundary                 | Oscillation at boundary | $C_k$ sign changes                             | Increase $\mu_k$ (barrier strength)        |

*Remark (Training Protocol).* The Governor is trained via reinforcement learning on this suite, with reward $r_t = -\Delta V_{\mathfrak{L}}$. Episodes terminate when $V_{\mathfrak{L}}$ plateaus or diverges.

:::

## 07_cognition/03_memory_retrieval.md

:::{prf:definition} Historical Record
:label: def-historical-record

Let $\gamma: [0, T] \to \mathcal{Z}$ be the agent's trajectory on the latent manifold $(\mathcal{Z}, G)$ over time interval $[0, T]$. The *historical record* is the pair $(\gamma, \alpha)$ where $\alpha: [0, T] \to \mathbb{R}$ is the reward flux along the trajectory (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\gamma(t)] = [z]$, $[\alpha(t)] = \text{nat}/[s]$.

*Cross-reference:* This connects to Memory Time $t' < t$ (Definition 1.3.4).

:::

:::{prf:definition} Memory Screen
:label: def-memory-screen

The *memory screen* is the signed measure on $\mathcal{Z}$ defined by

$$
\Xi_T := \int_0^T \alpha(t') \, \delta_{\gamma(t')} \, dt',

$$
where:
- $\delta_{\gamma(t')}$ is the Dirac measure concentrated at $\gamma(t') \in \mathcal{Z}$,
- $\alpha(t') = J_r(t')$ is the (signed) reward flux at time $t'$ (Definition {prf:ref}`def-the-reward-flux`).

*Units:* $[\Xi_T] = \text{nat}$ (total signed measure), $[\alpha] = \text{nat}/[s]$ (reward flux rate).

*Interpretation:* $\Xi_T$ encodes where the agent has been, weighted by the sign and magnitude of reward received. Positive rewards contribute positive measure (attractive memory); negative rewards contribute negative measure (repulsive memory).

*Cross-reference (Relativistic Multi-Agent):* In Chapter 29, the Memory Screen is elevated from an auxiliary construct to a **primary state variable**. The Causal Bundle $\mathcal{Z}_{\text{causal}} := \mathcal{Z}^{(N)} \times \Xi_{<t}$ restores the Markov property in relativistic multi-agent settings where finite information speed creates non-Markovian dynamics. See Definition {prf:ref}`def-causal-bundle`.

:::

:::{prf:remark} Connection to Holographic Persistence
:label: rem-connection-to-holographic-persistence

The memory screen $\Xi_T$ provides the mathematical realization of holographic persistence ({ref}`FAQ D.5.3 <sec-appendix-d-control-theory-system-safety>`). The measure $\Xi_T$ on $\mathcal{Z}$ acts as a "hologram" of the agent's history projected onto the latent space, from which non-local forces can be computed.

:::

:::{prf:definition} Memory Kernel via Heat Equation {cite}`grigoryan2009heat,rosenberg1997laplacian`
:label: def-memory-kernel-via-heat-equation

The canonical memory kernel is the *Heat Kernel* $H_\tau(z, z')$ on $(\mathcal{Z}, G)$, defined as the fundamental solution to the heat equation:

$$
(\partial_\tau - \Delta_G) H_\tau(z, z') = 0, \quad H_0(z, z') = \delta(z - z'),

$$
where:
- $\tau > 0$ is the *diffusion time* (memory smoothing scale),
- $\Delta_G = G^{ij}\nabla_i\nabla_j$ is the Laplace-Beltrami operator on $(\mathcal{Z}, G)$ (Definition 2.5.3).

*Units:* $[H_\tau] = [z]^{-d}$ (probability density), $[\tau] = [z]^2$ (diffusion time in geometric units).

*Interpretation:* $H_\tau(z, z')$ measures how much influence a memory at $z'$ has on the current position $z$ after diffusion time $\tau$. Larger $\tau$ yields smoother, more diffuse memory influence. For compact manifolds, $H_\tau$ admits an eigenfunction expansion; for non-compact manifolds with bounded geometry, Gaussian upper bounds hold {cite}`grigoryan2009heat`.

:::

:::{prf:definition} Memory Potential
:label: def-memory-potential

The *memory potential* is defined by

$$
\Psi_{\text{mem}}(z) := -\int_{\mathcal{Z}} H_\tau(z, z') \, d\Xi_T(z').

$$
Expanding using Definition {prf:ref}`def-memory-screen`:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t')) \, dt'.

$$
*Units:* $[\Psi_{\text{mem}}] = \text{nat}$.

*Interpretation:* The memory potential is the convolution of the heat kernel with the signed reward-weighted trajectory measure. Since $\Xi_T$ is a signed measure:
- Near high-reward past positions ($\alpha > 0$): $\Psi_{\text{mem}} < 0$, creating a potential well. The force $-\nabla_G \Psi_{\text{mem}}$ points toward the memory (attractive).
- Near high-penalty past positions ($\alpha < 0$): $\Psi_{\text{mem}} > 0$, creating a potential barrier. The force $-\nabla_G \Psi_{\text{mem}}$ points away from the memory (repulsive).

The sign convention ensures that the drift term inside $\mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla \Psi_{\text{mem}}\right)$ moves toward rewarding experiences and away from penalizing ones.

:::

:::{prf:proposition} Kernel Alternatives {cite}`rasmussen2006gp`
:label: prop-kernel-alternatives

Alternative kernels may be used depending on application requirements:

1. **Gaussian (RBF) Kernel:**

   $$
   K_{\text{Gauss}}(z, z') := \exp\left(-\frac{d_G(z, z')^2}{2\ell^2}\right),

   $$
   where $d_G$ is the geodesic distance and $\ell > 0$ is the length scale. This provides fast (exponential) decay, suitable for short-range memory effects.

2. **Matérn Kernel:**

   $$
   K_{\nu}(z, z') \propto (-\Delta_G + \kappa^2)^{-\nu}\delta(z - z'),

   $$
   where $\nu > 0$ is the smoothness parameter and $\kappa > 0$ is the inverse correlation length. For $\nu = 1$, this recovers the Green's function $G_\kappa$ from {ref}`sec-the-bulk-potential-screened-poisson-equation`. The Matérn kernel has polynomial (rather than exponential) tails, providing longer-range correlations. See {cite}`rasmussen2006gp` Chapter 4 for the Euclidean case.

*Cross-reference:* The Matern kernel with $\nu = 1$ coincides with the screened Poisson Green's function (Definition {prf:ref}`prop-green-s-function-decay`), establishing a direct connection between memory effects and value propagation.

:::

:::{prf:theorem} Non-Markovian Nature of Memory
:label: thm-non-markovian-nature-of-memory

The force field $-\nabla_G \Psi_{\text{mem}}$ violates the Markov property.

:::

:::{prf:definition} Memory-Augmented Geodesic SDE
:label: def-memory-augmented-geodesic-sde

The memory-augmented dynamics on $(\mathcal{Z}, G)$ are:

$$
dz^k = \left[\mathcal{M}_{\text{curl}}\right]^k{}_{j}\left(-G^{j\ell}\partial_\ell\bigl(\Phi_{\text{eff}} + \Psi_{\text{mem}}\bigr) + u_\pi^j\right) ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$ is the effective potential (Definition {prf:ref}`def-effective-potential`),
- $\Psi_{\text{mem}}$ is the memory potential (Definition {prf:ref}`def-memory-potential`),
- $\Gamma^k_{ij}$ are the Christoffel symbols of $G$ (Definition 2.5.1),
- $u_\pi^k$ is the policy control field (Definition {prf:ref}`def-the-control-field`),
- $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`, {ref}`sec-the-geodesic-baoab-integrator`),
- $W^j_s$ is a standard Wiener process,
- $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$ is the curl-corrected mobility.

*Cross-reference:* Definition {prf:ref}`def-bulk-drift-continuous-flow`.

*Units:* All terms have units $[z]/[s]$.

:::

:::{prf:lemma} Virtual Work of Recall
:label: lem-virtual-work-of-recall

The infinitesimal work performed by the memory force during displacement $dz$ is:

$$
dW_{\text{mem}} := \langle -\nabla_G \Psi_{\text{mem}}, dz \rangle_G = -G_{kj}\,G^{k\ell}\partial_\ell \Psi_{\text{mem}}\, dz^j = -\partial_j \Psi_{\text{mem}}\, dz^j.

$$
*Units:* $[dW_{\text{mem}}] = \text{nat}$.

*Interpretation:* When the agent moves toward regions of low $\Psi_{\text{mem}}$ (attractive memory, i.e., $d\Psi_{\text{mem}} < 0$), positive work $dW_{\text{mem}} > 0$ is extracted from the memory field. This corresponds to "reward from recall"---revisiting previously successful states.

:::

:::{prf:theorem} Memory-Induced Barrier Crossing
:label: thm-memory-induced-barrier-crossing

Let $z_t$ be the current position and suppose there exists a past time $t^* < t$ with $z^* := \gamma(t^*)$ such that:
1. $d_G(z_t, z^*) < \ell_{\text{mem}}$ for some memory influence radius $\ell_{\text{mem}}$,
2. $|\alpha(t^*)|$ is large (strong reward signal at time $t^*$).

Then the memory gradient $\|\nabla_G \Psi_{\text{mem}}\|_G$ can exceed the local barrier gradient $\|\nabla_G \Phi_{\text{eff}}\|_G$, enabling transitions that would be forbidden under purely local dynamics.

:::

:::{prf:definition} Memory-Augmented Reaction-Diffusion
:label: def-memory-augmented-reaction-diffusion

The WFR dynamics with memory are:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \left(\frac{\Phi_{\text{eff}} + \Psi_{\text{mem}} - \bar{\Phi}_{\text{aug}}}{T_c}\right),

$$
where:
- $\rho(z, s)$ is the belief density,
- $\mathbf{v} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla(\Phi_{\text{eff}} + \Psi_{\text{mem}}) + u_\pi\right)$ is the curl-corrected drift,
- $\bar{\Phi}_{\text{aug}} = \int_{\mathcal{Z}} (\Phi_{\text{eff}} + \Psi_{\text{mem}}) \rho \, d\mu_G$ is the mean augmented potential.

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Theorem {prf:ref}`thm-wfr-consistency-value-creates-mass`.

*Units:* $[\partial_s \rho] = [z]^{-d}/[s]$, all terms balance.

:::

:::{prf:proposition} Mass Creation from Experience
:label: prop-mass-creation-from-experience

The memory contribution to the reaction term is:

$$
r_{\text{mem}}(z) := \frac{\rho(z)(\Psi_{\text{mem}}(z) - \bar{\Psi}_{\text{mem}})}{T_c},

$$
where $\bar{\Psi}_{\text{mem}} = \int_{\mathcal{Z}} \Psi_{\text{mem}} \rho \, d\mu_G$.

*Interpretation:* Belief mass is created where $\Psi_{\text{mem}} < \bar{\Psi}_{\text{mem}}$ (attractive memory) and destroyed where $\Psi_{\text{mem}} > \bar{\Psi}_{\text{mem}}$ (repulsive memory). This acts as a *virtual source* that redistributes probability toward remembered high-reward regions, even when local dynamics (via $\Phi_{\text{eff}}$) do not support such transitions.

:::

:::{prf:definition} Non-Locality Ratio
:label: def-non-locality-ratio

The *non-locality ratio* at position $z$ is:

$$
\Omega_{\text{mem}}(z) := \frac{\|\nabla_G \Psi_{\text{mem}}(z)\|_G}{\|\nabla_G \Phi_{\text{eff}}(z)\|_G + \epsilon},

$$
where $\epsilon > 0$ is a regularization constant preventing division by zero.

*Units:* $[\Omega_{\text{mem}}] = \text{dimensionless}$.

**Heuristic 27.5.2 (Homeostatic Bound on Memory).** For stable operation, the non-locality ratio should satisfy:

$$
\Omega_{\text{mem}} \in [\Omega_{\min}, \Omega_{\max}],

$$
with empirically recommended bounds $\Omega_{\min} \approx 0.01$, $\Omega_{\max} \approx 10$. These bounds are task-dependent and should be tuned based on the environment's stationarity.

*Boundary cases:*
- $\Omega_{\text{mem}} \to 0$: Pure Markovian dynamics; agent exhibits catastrophic forgetting.
- $\Omega_{\text{mem}} \to \infty$: Pure memory-driven dynamics; agent overfits to historical experience and fails to respond to current environmental gradients.

*Cross-reference:* The Governor ({ref}`sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`) can regulate $\Omega_{\text{mem}}$ by adjusting the memory smoothing scale $\tau$ or the reward flux weighting in $\alpha(t')$.

:::

:::{prf:definition} External Knowledge Manifold
:label: def-external-knowledge-manifold

Let $\mathcal{Z}_{\text{ext}}$ denote the external knowledge manifold equipped with metric $G_{\text{ext}}$, structured as a fiber bundle:

$$
\mathcal{Z}_{\text{ext}} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\text{tex}},

$$
where $\mathcal{K}$ is the macro-concept space, $\mathcal{Z}_n$ the nuisance coordinates, and $\mathcal{Z}_{\text{tex}}$ the texture fiber.

*Units:* $[G_{\text{ext},ij}] = [z]^{-2}$ (matching the internal metric).

*Cross-reference:* This decomposition mirrors {ref}`sec-conditional-independence-and-sufficiency`'s latent structure $(K, z_n, z_{\text{tex}})$ and {ref}`sec-tier-the-attentive-atlas`'s Atlas architecture.

:::

:::{prf:axiom} Metric Isometry
:label: ax-metric-isometry

There exists a canonical isometry $\Phi: \mathcal{Z}_{\text{int}} \to \mathcal{Z}_{\text{ext}}$ such that for all $z, z' \in \mathcal{Z}_{\text{int}}$:

$$
d_{G_{\text{int}}}(z, z') = d_{G_{\text{ext}}}(\Phi(z), \Phi(z')),

$$
where both manifolds carry the Poincare metric (Definition {prf:ref}`def-hyperbolic-volume-growth`):

$$
G_{ij}(z) = \frac{4\delta_{ij}}{(1 - \|z\|^2)^2}.

$$
*Interpretation:* The isometry axiom asserts that embedding models trained on shared semantic corpora induce compatible distance structures. This is the mathematical foundation for cross-modal retrieval.

:::

:::{prf:definition} Knowledge Atom
:label: def-knowledge-atom

A *knowledge atom* is a triple $\xi = (K, z_n, z_{\text{tex}}) \in \mathcal{Z}_{\text{ext}}$ where:
- $K \in \mathcal{K}$: macro-concept (topic, entity class, logical category)
- $z_n \in \mathcal{Z}_n$: nuisance coordinates (style, formatting, source metadata)
- $z_{\text{tex}} \in \mathcal{Z}_{\text{tex}}$: high-frequency texture (specific wording, surface form)

*Cross-reference:* Compare {ref}`sec-conditional-independence-and-sufficiency`'s decomposition. The macro closure mechanism (Definition 2.8.1) applies equally to external atoms.

:::

:::{prf:definition} Hyperbolic Geodesic Distance
:label: def-hyperbolic-geodesic-distance

For points $z, \xi \in \mathbb{D}^d$ (the Poincare disk), the geodesic distance is:

$$
d_{\mathbb{D}}(z, \xi) = \operatorname{acosh}\left(1 + \frac{2\|z - \xi\|^2}{(1 - \|z\|^2)(1 - \|\xi\|^2)}\right).

$$
*Units:* $[d_{\mathbb{D}}] = [z]$ (dimensionless in Poincare coordinates).

*Cross-reference:* This is the distance function induced by the Poincare metric $G_{ij}$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). See also Definition {prf:ref}`prop-isotropic-radial-expansion` for the hyperbolic potential $U(z) = -2\operatorname{artanh}(\|z\|)$.

:::

:::{prf:definition} Retrieval Measure via Geodesic Functional
:label: def-retrieval-measure-via-geodesic-functional

Given a query position $z \in \mathcal{Z}_{\text{int}}$ and archive prior $\mu_{\mathcal{E}} \in \mathcal{P}(\mathcal{Z}_{\text{ext}})$, the *retrieval measure* is:

$$
\nu_\omega = \arg\min_{\nu \in \mathcal{P}(\mathcal{Z}_{\text{ext}})} \left\{ \int d_{\mathbb{D}}(z, \xi) \, d\nu(\xi) + T_{\text{ret}} D_{\text{KL}}(\nu \| \mu_{\mathcal{E}}) \right\},

$$
where $T_{\text{ret}} > 0$ is the *retrieval temperature*.

*Units:* $[T_{\text{ret}}] = \text{nat}$.

*Interpretation:* This variational problem balances semantic proximity (first term) against prior plausibility (KL term). At $T_{\text{ret}} \to 0$, retrieval concentrates on the nearest neighbor; at $T_{\text{ret}} \to \infty$, it reverts to the archive prior.

:::

:::{prf:proposition} Exponential Complexity of Specificity
:label: prop-exponential-complexity-of-specificity

The volume of a geodesic ball in the Poincare disk grows exponentially with radius:

$$
\text{Vol}(B_r(z)) \sim \sinh^{d-1}(r) \sim \frac{1}{2^{d-1}} e^{(d-1)r} \quad \text{as } r \to \infty.

$$
:::

:::{prf:definition} Bulk Projection Operator
:label: def-bulk-projection-operator

The *bulk projection* $\Pi_{\text{bulk}}: \mathcal{Z}_{\text{ext}} \to \mathcal{K} \times \mathcal{Z}_n$ is defined by:

$$
\Pi_{\text{bulk}}(\xi) = \Pi_{\text{bulk}}(K, z_n, z_{\text{tex}}) := (K, z_n).

$$
*Interpretation:* This projection discards texture, retaining only control-relevant coordinates.

*Cross-reference:* This extends the internal texture exclusion of {ref}`sec-conditional-independence-and-sufficiency` to external retrieval.

:::

:::{prf:definition} Bulk-Filtered Retrieval Potential
:label: def-bulk-filtered-retrieval-potential

The *retrieval potential* is:

$$
\Psi_{\text{ret}}(z) = -\Lambda_{\text{ret}} \int_{\mathcal{Z}_{\text{ext}}} \exp\left(-\lambda \, d_{\mathbb{D}}(z, \Pi_{\text{bulk}}(\xi))\right) d\nu_\omega(\xi),

$$
with the firewall constraint:

$$
\frac{\partial \Psi_{\text{ret}}}{\partial z_{\text{tex,ext}}} \equiv 0.

$$
*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[\Lambda_{\text{ret}}] = \text{nat}$, $[\lambda] = [z]^{-1}$.

*Cross-reference:* Compare the memory potential $\Psi_{\text{mem}}$ (Definition {prf:ref}`def-memory-potential`), which uses heat kernel rather than geodesic exponential. Both generate conservative forces.

:::

:::{prf:theorem} Stability of Retrieval Loop
:label: thm-stability-of-retrieval-loop

Under the firewall constraint (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`), the retrieval force field:

$$
\mathbf{f}_{\text{ret}} = -G^{-1}\nabla_G \Psi_{\text{ret}}

$$
is smooth (Lipschitz in $z$) and independent of external texture coordinates $z_{\text{tex,ext}}$.

*Consequence:* The control loop remains stable; external texture cannot inject high-frequency gradients that would trigger Mode T.C (Labyrinthine Overfitting).

:::

:::{prf:definition} Retrieval-Augmented Geodesic SDE
:label: def-retrieval-augmented-geodesic-sde

The equations of motion with retrieval are:

$$
dz^k = \left[\mathcal{M}_{\text{curl}}\right]^k{}_{j}\left(-G^{j\ell}\partial_\ell(\Phi_{\text{eff}} + \Psi_{\text{mem}} + \Psi_{\text{ret}}) + u_\pi^j\right) ds - \Gamma^k_{ij}\dot{z}^i\dot{z}^j\,ds + \sqrt{2T_c}(G^{-1/2})^{kj}dW^j_s,

$$
where:
- $\Phi_{\text{eff}}$: effective potential (Definition {prf:ref}`def-effective-potential`)
- $\Psi_{\text{mem}}$: memory potential (Definition {prf:ref}`def-memory-potential`)
- $\Psi_{\text{ret}}$: retrieval potential (Definition {prf:ref}`def-bulk-filtered-retrieval-potential`)
- $\Gamma^k_{ij}$: Christoffel symbols (Definition 2.5.1, Definition 22.2.1a)
- $u_\pi^k$: policy control field (Definition {prf:ref}`def-the-control-field`)
- $T_c$: cognitive temperature ({ref}`sec-the-geodesic-baoab-integrator`)

*Cross-reference:* This extends the memory-augmented SDE (Definition {prf:ref}`def-memory-augmented-geodesic-sde`) with the retrieval term $\Psi_{\text{ret}}$.

:::

:::{prf:proposition} Superposition of Non-Local Forces
:label: prop-superposition-of-non-local-forces

The total non-local force is:

$$
\mathbf{f}_{\text{non-local}} = -G^{-1}\nabla_G(\Psi_{\text{mem}} + \Psi_{\text{ret}}),

$$
where:
- Memory force $\mathbf{f}_{\text{mem}}$ integrates over the agent's past trajectory
- Retrieval force $\mathbf{f}_{\text{ret}}$ integrates over the external archive

*Interpretation:* The agent simultaneously experiences attraction to its own memory ({ref}`sec-section-non-local-memory-as-self-interaction-functional`) and to relevant external knowledge (this section).

:::

:::{prf:definition} Retrieval Source Term
:label: def-retrieval-source-term

The Wasserstein–Fisher–Rao continuity equation with retrieval is:

$$
\partial_s \rho + \nabla \cdot (\rho \mathbf{v}) = \rho \, r_{\text{local}}(z) + \sigma_{\text{ret}}(z),

$$
where:
- $r_{\text{local}}(z)$: local mass creation rate (reward-driven, Definition {prf:ref}`def-the-wfr-action`)
- $\sigma_{\text{ret}}(z)$: retrieval source term

The retrieval source is:

$$
\sigma_{\text{ret}}(z) = \eta_{\text{ret}} \cdot \Psi_{\text{ret}}(z) \cdot \mathbf{1}[\Psi_{\text{ret}}(z) > \Psi_{\text{threshold}}],

$$
with $[\sigma_{\text{ret}}] = \text{nat}/[z]^d/\text{step}$.

*Cross-reference:* Compare {ref}`sec-wfr-dynamics-with-memory-sources`'s memory mass creation. Both mechanisms inject mass at non-local locations.

:::

:::{prf:proposition} Non-Causal Transition via Retrieval
:label: prop-non-causal-transition-via-retrieval

Mass injection at retrieved locations enables transitions without continuous geodesic paths:

$$
\rho(z', s + \Delta s) > 0 \quad \text{even if} \quad d_G(z, z') > \sup_{0 \leq \tau \leq \Delta s} \|\mathbf{v}(z, s+\tau)\| \cdot \Delta s.

$$
*Interpretation:* Retrieval teleports probability mass to semantically relevant regions, bypassing the diffusion constraint. This is the WFR-level description of "jumping to a retrieved fact."

:::

:::{prf:proposition} Optimal Non-Local Coupling
:label: prop-optimal-nonlocal-coupling

Let the control vector be $\Lambda = (\Lambda_{\text{mem}}, \Lambda_{\text{ret}})$. The optimal coupling is the fixed point of the Governor's policy $\pi_{\mathfrak{G}}$ ({prf:ref}`def-the-universal-governor`) given the diagnostic state $s_t = (\Delta_{\text{causal}}, \Omega_{\text{mem}})$.

**Control Law Derivation:**

1. **Surprise Signal:** Let $\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{int}} \| P_{\text{obs}})$ be the Interventional Gap (Node 53).

2. **Overfitting Signal:** Let $\Omega_{\text{mem}}$ be the Non-Locality Ratio ({prf:ref}`def-non-locality-ratio`, Node 43).

3. **Governor Update:** The Lyapunov descent condition $\Delta V_{\mathfrak{L}} < 0$ ({prf:ref}`def-training-lyapunov-function`) implies the following qualitative update dynamics:

$$
\begin{aligned}
\dot{\Lambda}_{\text{ret}} &\propto \alpha_1 \cdot \Delta_{\text{causal}} \\
\dot{\Lambda}_{\text{mem}} &\propto \alpha_2 \cdot (\Delta_{\text{causal}}^{\text{target}} - \Delta_{\text{causal}}) - \alpha_3 \cdot \operatorname{ReLU}(\Omega_{\text{mem}} - \Omega_{\max})
\end{aligned}

$$
where $\alpha_1, \alpha_2, \alpha_3 > 0$ are learning rates and $\Omega_{\max}$ is the maximum tolerable non-locality ratio.

:::

:::{prf:remark} Operational Interpretation
:label: rem-memory-retrieval-interpretation

- **If the agent is surprised by reality** ($\Delta_{\text{causal}}$ high): It must increase reliance on external truth ($\Lambda_{\text{ret}} \uparrow$).
- **If the agent is not surprised** ($\Delta_{\text{causal}}$ low): It can conserve bandwidth by relying on internal memory ($\Lambda_{\text{mem}} \uparrow$), subject to the constraint that it must not overfit ($\Omega_{\text{mem}} < \Omega_{\max}$).

This closes the joint optimization problem by reducing it to a specific instantiation of the Governor's Lyapunov stability framework ({prf:ref}`def-training-lyapunov-function`).
:::

:::{prf:theorem} Safe Retrieval Bandwidth
:label: thm-safe-retrieval-bandwidth

Let $\sigma_{\text{ret}}(z)$ be the retrieval source term in the WFR continuity equation ({prf:ref}`def-retrieval-source-term`). The latent geometry remains non-singular if and only if the total information flux satisfies:

$$
\int_{\mathcal{Z}} \left( \rho_I(z) + \sigma_{\text{ret}}(z) \right) \, d\mu_G \leq \kappa \, C_{\partial}

$$
where $C_{\partial} = \nu_D \cdot \text{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ is the boundary capacity (Definition {prf:ref}`def-holographic-coefficient`, {prf:ref}`def-levin-length`).

:::

:::{prf:theorem} Causal Isometry Theorem
:label: thm-causal-isometry

Let $\mathcal{M}_A$ and $\mathcal{M}_B$ be latent manifolds encoding modalities $A$ and $B$ of a common environment $\mathcal{E}$. Let $\Phi_{\text{causal}}$ be the Causal Information Potential ({ref}`sec-causal-discovery-interventional-geometry-and-the-singularity-of-action`). If both representations are **Interventionally Closed** ({prf:ref}`thm-interventional-closure`) and the metric-law solution is unique (as for the saturated Poincare-disk ansatz), then the induced metrics $G_A$ and $G_B$ are isometric.

:::

## 07_cognition/04_ontology.md

:::{prf:definition} Semantic Vacuum
:label: def-semantic-vacuum

Let $(\mathbb{D}, G)$ be the Poincare disk with metric $G_{ij}(z) = 4\delta_{ij}/(1-|z|^2)^2$ (Definition {prf:ref}`def-hyperbolic-volume-growth`). The **Semantic Vacuum** is the fiber

$$
\emptyset := \{z \in \mathcal{Z} : |z| = 0\} = \{0\} \times \mathcal{Z}_{\text{tex}},

$$
equipped with the following properties:

1. **$SO(D)$ Symmetry:** At $z=0$, the metric is isotropic $G(0) = 4I$ (Proposition {prf:ref}`prop-so-d-symmetry-at-origin`), and the entropic force vanishes: $F_{\text{entropy}}(0) = 0$. The system has full rotational symmetry $SO(D)$.

2. **Infrared Limit:** For any TopoEncoder scale $\tau$ ({ref}`sec-rigorous-interpretation-renormalization-group-flow`), $\lim_{\tau \to 0} z(\tau) = \emptyset$. The vacuum is the coarsest resolution.

3. **Reference Measure:** The vacuum carries the Dirac reference measure $\delta_0$ on the bulk coordinates $(K, z_n)$:

   $$
   \mu_{\emptyset} := \delta_0 \otimes \mathcal{N}(0, \sigma_{\text{tex}}^2 I),

   $$
   where the texture component is drawn from the isotropic prior (Definition {prf:ref}`def-boundary-texture-distribution` with $G^{-1}(0) = I/4$).

4. **Information Content:** At the vacuum, $U(0) = 0$ (Definition {prf:ref}`prop-isotropic-radial-expansion`), corresponding to zero information content (maximum entropy).

*Units:* $[\mu_{\emptyset}]$ is a probability measure; $[U] = \mathrm{nat}$.

*Remark (Unstable Origin).* The vacuum is not a fixed point of the radial dynamics: the entropic drift $(1-r^2)/2 > 0$ at $r=0$ implies trajectories immediately expand outward (Theorem {prf:ref}`thm-angular-symmetry-breaking`). The $SO(D)$ angular symmetry is broken by the policy or thermal fluctuations during this expansion.

:::

:::{prf:lemma} Default Mapping to Vacuum
:label: lem-default-mapping-to-vacuum

Let $\{q_i\}_{i=1}^{N_c}$ be the chart query bank (Definition {prf:ref}`def-attentive-routing-law`) and assume the queries are **centered**: $\sum_{i=1}^{N_c} q_i = 0$. Then for any key $k(x)$ such that all inner products are equal---$\langle q_i, k(x) \rangle = c$ for all $i$---the router weights are uniform:

$$
w_i(x) = \frac{1}{N_c} \quad \forall i \in \{1, \ldots, N_c\}.

$$
The resulting soft codebook embedding is the **barycenter**:

$$
z_q(x) = \sum_{i=1}^{N_c} w_i(x) e_{i, K_{\text{code},i}(x)} = \frac{1}{N_c} \sum_{i=1}^{N_c} e_{i,*},

$$
which equals $0$ if the per-chart codebooks are also centered ($\sum_c e_{i,c} = 0$ for each chart $i$).

:::

:::{prf:definition} Ontological Stress
:label: def-ontological-stress

Let $(K_t, z_{n,t}, z_{\text{tex},t})$ be the agent's state at time $t$ (Definition {prf:ref}`def-bounded-rationality-controller`). The **Ontological Stress** is the conditional mutual information:

$$
\Xi := I(z_{\text{tex},t}; z_{\text{tex},t+1} \mid K_t, z_{n,t}, K^{\text{act}}_t),

$$
where $I(\cdot;\cdot|\cdot)$ denotes conditional mutual information in nats.

*Units:* $[\Xi] = \mathrm{nat}$ (dimensionless information).

*Interpretation.* By Axiom {prf:ref}`ax-bulk-boundary-decoupling` (Bulk-Boundary Decoupling), texture should be unpredictable -- a white-noise residual. If $\Xi > 0$, then texture at time $t$ predicts texture at time $t+1$, conditional on the macro-state and action. This violates the partition condition: the texture channel contains structure that should have been captured by $(K, z_n)$ but was not. The agent's ontology is **too coarse**.

*Cross-reference.* Compare with the closure defect $I(K_{t+1}; Z_t \mid K_t, K^{\text{act}}_t)$ ({ref}`sec-conditional-independence-and-sufficiency`). Ontological Stress is the dual: predictability *within* texture rather than *from* texture to macro.

:::

:::{prf:theorem} Vacuum Concentration Under Unknown Unknowns
:label: thm-vacuum-concentration-under-unknown-unknowns

Let $\mathcal{F}[p, \pi]$ be the entropy-regularized objective (Definition {prf:ref}`def-entropy-regularized-objective-functional`):

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - \tau H(\pi(\cdot|z)) \Big) d\mu_G.

$$
If the value function $V$ is **uninformative** in a region $\Omega \subset \mathcal{Z}$ -- i.e., $\nabla_A V|_\Omega \approx 0$ and $\nabla^2 V|_\Omega \approx 0$ -- then the entropy term dominates and the optimal belief concentrates toward maximum-entropy configurations:

$$
p^*(z) \propto \exp\left(-\frac{V(z)}{\tau}\right) \xrightarrow{\nabla_A V \to 0} \text{uniform on } \Omega.

$$
In the Poincare disk geometry, the maximum-entropy state is the vacuum $z = 0$.

:::

:::{prf:axiom} Ontological Expansion Principle
:label: ax-ontological-expansion-principle

The agent should expand its chart structure (increase $N_c$) if and only if the expected value improvement exceeds the complexity cost:

$$
\mathbb{E}\left[\Delta V \mid \text{fission}\right] > \mathcal{C}_{\text{complexity}}(N_c \to N_c + 1),

$$
where $\Delta V$ is the value gain from finer discrimination and $\mathcal{C}_{\text{complexity}}$ is measured in nats (to match units with value).

*Remark.* This is the MDL/rate-distortion principle ({ref}`sec-the-shutter-as-a-vq-vae`) applied to ontology: expand only if the distortion reduction exceeds the rate increase.

:::

:::{prf:theorem} Fission Criterion
:label: thm-fission-criterion

Let $\Xi$ be the Ontological Stress (Definition {prf:ref}`def-ontological-stress`) and let $\Xi_{\text{crit}} > 0$ be a threshold. Let $\Delta V_{\text{proj}}$ be the projected value improvement from splitting the highest-stress chart. The fission criterion is:

$$
\text{Fission} \iff \Xi > \Xi_{\text{crit}} \quad \text{AND} \quad \Delta V_{\text{proj}} > \mathcal{C}_{\text{complexity}}.

$$
*Units:* All quantities are in nats. The complexity cost $\mathcal{C}_{\text{complexity}}(N_c \to N_c + 1)$ includes the entropy increase $\log((N_c+1)/N_c)$ from the expanded codebook plus any regularization penalty on parameter count.

:::

:::{prf:definition} Query Fission
:label: def-query-fission

Let $q_i \in \mathbb{R}^d$ be a chart query vector ({ref}`sec-tier-the-attentive-atlas`) with associated codebook $\{e_{i,c}\}_{c=1}^{N_v}$. A **query fission** replaces $q_i$ with two daughter queries:

$$
q_i \mapsto \{q_i^+, q_i^-\} := \{q_i + \epsilon u, q_i - \epsilon u\},

$$
where $u \in \mathbb{R}^d$ is the **fission direction** (unit vector) and $\epsilon > 0$ is the **fission amplitude**.

The daughter codebooks are initialized as copies:

$$
e_{i^\pm, c} := e_{i, c} \quad \forall c \in \{1, \ldots, N_v\}.

$$
*Selection of fission direction.* The optimal $u$ maximizes the variance of router assignments under the new queries:

$$
u^* = \arg\max_{\|u\|=1} \text{Var}_{x \sim \mathcal{D}}\left[\langle k(x), u \rangle \mid w_i(x) > 1/N_c\right],

$$
i.e., the principal component of keys within the chart's Voronoi cell.

:::

:::{prf:theorem} Supercritical Pitchfork Bifurcation for Charts {cite}`strogatz2015nonlinear`
:label: thm-supercritical-pitchfork-bifurcation-for-charts

The query fission dynamics exhibit a **supercritical pitchfork bifurcation**. Let $r := \|q_i^+ - q_i^-\|/2 = \epsilon$ be the half-separation of daughter queries. The radial evolution satisfies:

$$
\frac{dr}{ds} = (\Xi - \Xi_{\text{crit}}) r - \alpha r^3 + \sigma\xi,

$$
where:
- $\Xi - \Xi_{\text{crit}}$ is the **bifurcation parameter** (supercritical when positive)
- $\alpha > 0$ is a stabilizing cubic coefficient (from competition for training data)
- $\sigma\xi$ is noise from stochastic gradient updates
- $s$ is the training step (flow time)

**Phase Transition:**
1. **Sub-critical ($\Xi < \Xi_{\text{crit}}$):** $r=0$ is the unique stable fixed point. The daughters collapse back to the parent ($r \to 0$).
2. **Super-critical ($\Xi > \Xi_{\text{crit}}$):** $r=0$ becomes unstable. The daughters separate toward a new equilibrium:

   $$
   r^* = \sqrt{\frac{\Xi - \Xi_{\text{crit}}}{\alpha}}.

   $$
:::

:::{prf:definition} Ontological Ricci Flow
:label: def-ontological-ricci-flow

Let $G_{ij}(z, s)$ be the capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) parameterized by flow time $s$. Define the **local stress field** $\Xi(z) := \mathbb{E}[\Xi \mid K = k(z)]$, where $k(z)$ is the chart containing $z$. The **Ontological Ricci Flow** is:

$$
\frac{\partial G_{ij}}{\partial s} = -2\left(R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right) + \nu \nabla_i \nabla_j \Xi(z),

$$
where:
- $R_{ij}$ is the Ricci curvature tensor, $R = G^{ij}R_{ij}$ the scalar curvature
- $\Lambda, \kappa$ are constants from Theorem {prf:ref}`thm-capacity-constrained-metric-law`
- $T_{ij}$ is the risk tensor
- $\nu > 0$ is the stress-curvature coupling constant

*Units:* $[\partial G / \partial s] = [z]^{-2}$; $[\Xi] = \text{nat}$; $[\nabla_i \nabla_j \Xi] = \text{nat}/[z]^2$.

*Interpretation.* The first term drives the metric toward the capacity-constrained fixed point. The second term $\nu \nabla_i \nabla_j \Xi$ introduces curvature in regions of high stress gradient, expanding the metric where new distinctions are needed.

:::

:::{prf:proposition} Fixed Points of Ontological Ricci Flow
:label: prop-fixed-points-of-ontological-ricci-flow

The flow has fixed points when:
1. The capacity-constrained metric law is satisfied: $R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}$
2. The Ontological Stress has vanishing Hessian: $\nabla_i \nabla_j \Xi = 0$

Condition (2) is satisfied when either $\Xi$ is constant (uniform stress) or $\Xi = 0$ (no stress).

*Computational Proxy.* In practice, we do not solve the Ricci flow PDE. The squared residual of the fixed-point condition can be used as a regularization loss:

$$
\mathcal{L}_{\text{Ricci}} := \left\|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}\right\|_F^2 + \nu^2 \|\nabla_i \nabla_j \Xi\|_F^2,

$$
encouraging the learned metric to satisfy the capacity constraint while penalizing stress gradients.

:::

:::{prf:definition} Ontological Redundancy
:label: def-ontological-redundancy

Let $K_i$ and $K_j$ be two charts with associated belief distributions $\mu_i, \mu_j$, transition models $\bar{P}_i, \bar{P}_j$, and value functions $V_i, V_j$. Their **ontological redundancy** is:

$$
\Upsilon_{ij} := \exp\left(-\left[ d_{\text{WFR}}(\mu_i, \mu_j) + D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) + \|V_i - V_j\|_G^2 \right]\right)

$$
where:
- $d_{\text{WFR}}(\mu_i, \mu_j)$ is the Wasserstein-Fisher-Rao distance ({prf:ref}`def-the-wfr-action`) between belief distributions,
- $D_{\mathrm{KL}}(\bar{P}_i \| \bar{P}_j) := \mathbb{E}_{k \sim \mu_i}\left[ D_{\mathrm{KL}}(\bar{P}(\cdot|k, a) \| \bar{P}_j(\cdot|k, a)) \right]$ is the mean predictive divergence,
- $\|V_i - V_j\|_G^2 := \mathbb{E}_{z \sim \mu_i}\left[ (V_i(z) - V_j(z))^2 \right]$ is the mean squared value divergence.

*Units:* Dimensionless; $\Upsilon_{ij} \in [0, 1]$.

*Interpretation:* $\Upsilon_{ij} \to 1$ implies the charts are functionally redundant: they occupy similar regions of belief space, predict similar futures, and assign similar values. $\Upsilon_{ij} \to 0$ implies they are functionally distinct.
:::

:::{prf:definition} Discrimination Gain
:label: def-discrimination-gain

The **Discrimination Gain** $G_\Delta(i, j)$ is the mutual information the agent loses about observations by merging charts $i$ and $j$:

$$
G_\Delta(i, j) := I(X; \{K_i, K_j\}) - I(X; K_{i \cup j})

$$
where $K_{i \cup j}$ is the merged chart that routes observations previously assigned to $K_i$ or $K_j$ to a single index.

*Units:* nat.

*MDL interpretation:* $G_\Delta$ is the increase in **distortion** (description length) resulting from the merge. If $G_\Delta \approx 0$, the distinction between $K_i$ and $K_j$ carries negligible information about the observation stream.
:::

:::{prf:lemma} Redundancy-Gain Relationship
:label: lem-redundancy-gain

Under the assumption that charts partition the observation space and the encoder is deterministic given observation $x$:

$$
G_\Delta(i, j) \leq H(K_i, K_j) - H(K_{i \cup j}) = \log 2 - H(K_i | K_j) \cdot \mathbb{I}[\Upsilon_{ij} < 1]

$$
When $\Upsilon_{ij} \to 1$, the bound tightens: $G_\Delta \to 0$.

:::

:::{prf:axiom} Ontological Simplification Principle
:label: ax-ontological-simplification

The agent shall reduce ontological complexity when the expected value of maintaining a distinction is negative:

$$
\mathcal{C}_{\text{saved}}(N_c \to N_c - 1) > G_\Delta(i, j) + \mathbb{E}[\Delta V \mid \text{no fusion}]

$$
where $\mathcal{C}_{\text{saved}}$ is the metabolic savings from eliminating a chart.

*Remark.* This is the dual of {prf:ref}`ax-ontological-expansion-principle` (Ontological Expansion Principle). Both derive from the same MDL objective: minimize description length plus expected regret.
:::

:::{prf:theorem} Fusion Criterion
:label: thm-fusion-criterion

Charts $i$ and $j$ shall be merged if and only if:

$$
G_\Delta(i, j) < \mathcal{C}_{\text{complexity}}(N_c) - \mathcal{C}_{\text{complexity}}(N_c - 1) + \epsilon_{\text{hysteresis}}

$$
where:
- $\mathcal{C}_{\text{complexity}}(N_c) = \log N_c + \lambda_{\text{param}} |\theta_{\text{chart}}|$ is the metabolic cost of maintaining $N_c$ charts ({ref}`sec-the-fission-criterion`),
- $\epsilon_{\text{hysteresis}} > 0$ is a hysteresis constant preventing oscillatory fission-fusion ("ontological churn").

:::

:::{prf:definition} Query Coalescence
:label: def-query-coalescence

Given charts $i, j$ satisfying the Fusion Criterion ({prf:ref}`thm-fusion-criterion`), the merged query is the **usage-weighted barycenter**:

$$
q_{\text{merged}} := \frac{\bar{w}_i q_i + \bar{w}_j q_j}{\bar{w}_i + \bar{w}_j}

$$
where $\bar{w}_k := \mathbb{E}[w_k(x)]$ is the historical routing weight from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`).

*Interpretation:* The more frequently used chart contributes more to the merged query position. This preserves the routing behavior for the majority of observations.
:::

:::{prf:definition} Fiber Reconciliation
:label: def-fiber-reconciliation

Let $L_{j \to i}: \mathcal{F}_j \to \mathcal{F}_i$ be the factorized jump operator ({prf:ref}`def-factorized-jump-operator`). For an observation $x$ previously assigned to chart $j$ with nuisance coordinates $z_n^{(j)}$, the reconciled coordinates in chart $i$ are:

$$
z_n^{(i, \text{reconciled})} := L_{j \to i}(z_n^{(j)}) = A_i(B_j z_n^{(j)} + c_j) + d_i

$$
where $B_j$ is the chart-to-global encoder and $A_i$ is the global-to-chart decoder.

*Codebook reconciliation:* The codebook entries of chart $j$ are projected into chart $i$'s Voronoi structure. Entries that fall within existing Voronoi cells of chart $i$ are absorbed; entries that create new structure may be retained if codebook capacity permits.
:::

:::{prf:theorem} Subcritical Pitchfork for Fusion
:label: thm-subcritical-pitchfork-fusion

Let $r(s) := \|q_i(s) - q_j(s)\|$ be the query separation at computation time $s$. During fusion, the dynamics become:

$$
\frac{dr}{ds} = -(\Upsilon_{ij} - \Upsilon_{\text{crit}}) r - \alpha r^3 + \sigma\xi(s)

$$
where:
- $\Upsilon_{\text{crit}} \in (0, 1)$ is the critical redundancy threshold,
- $\alpha > 0$ is the cubic stabilization coefficient,
- $\sigma\xi(s)$ is white noise with intensity $\sigma$.

When $\Upsilon_{ij} > \Upsilon_{\text{crit}}$:
1. The linear term is **negative** (attractive toward $r = 0$).
2. $r = 0$ becomes the **unique stable attractor**.
3. The queries "fall into each other" until they merge.

*Contrast with Fission ({prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`):*

| Property                | Fission (Supercritical)          | Fusion (Subcritical)                |
|:------------------------|:---------------------------------|:------------------------------------|
| Linear term sign        | $+\mu r$ (repulsive from origin) | $-\mu r$ (attractive to origin)     |
| Trigger                 | $\Xi > \Xi_{\text{crit}}$        | $\Upsilon > \Upsilon_{\text{crit}}$ |
| Stable fixed points     | $r^* = \pm\sqrt{\mu/\alpha}$     | $r^* = 0$                           |
| Physical interpretation | Charts repel and separate        | Charts attract and merge            |

:::

:::{prf:definition} Node 54 --- FusionReadinessCheck
:label: node-fusion-readiness-check

**Component:** Atlas (Chart Router)

**Type:** Metabolic Efficiency

**Interpretation:** Are any two charts functionally redundant?

**Proxy:**

$$
\text{FusionReady} := \mathbb{I}\left[ \max_{i \neq j} \Upsilon_{ij} > \Upsilon_{\text{crit}} \right]

$$
**Computational cost:** $O(N_c^2)$ pairwise comparisons.

**Trigger condition:** Two or more charts have redundancy exceeding threshold.

**Remediation:**
1. Identify most redundant pair $(i^*, j^*) = \arg\max_{i \neq j} \Upsilon_{ij}$.
2. Verify Fusion Criterion ({prf:ref}`thm-fusion-criterion`).
3. If satisfied, initiate subcritical bifurcation dynamics.
4. Execute Query Coalescence and Fiber Reconciliation.
5. Decrement chart count: $N_c \to N_c - 1$.
:::

:::{prf:definition} Node 55 --- CodebookLivenessCheck
:label: node-codebook-liveness-check

**Component:** Codebook (VQ Layer)

**Type:** Dead Code Detection

**Interpretation:** Are any code indices unused?

**Proxy:**

$$
\text{DeadCodeDetected} := \mathbb{I}\left[ \min_k P(K = k) < \epsilon_{\text{dead}} \right]

$$
where $P(K = k)$ is the empirical usage frequency of code $k$ over a trailing window.

**Computational cost:** $O(|\mathcal{K}|)$.

**Trigger condition:** Code usage falls below minimum threshold (default $\epsilon_{\text{dead}} = 10^{-4}$).

**Remediation:** Execute Lazarus Protocol ({prf:ref}`alg-lazarus`).

*Connection to existing diagnostics:* This node operationalizes the dead-code tolerance constraint from {ref}`sec-calibrating-tolerances`: $H(K) \geq \log((1 - \rho_{\text{dead}})|\mathcal{K}|)$.
:::

:::{prf:definition} Intra-Symbol Variance (Geometric Tension)
:label: def-intra-symbol-variance

For code $e_k$ in chart $i$, the **geometric tension** is:

$$
\sigma_k^2 := \mathbb{E}\left[ \|z_e - e_k\|^2 \;\Big|\; \text{VQ}(z_e) = k \right]

$$
where $z_e$ is the pre-quantized encoder output.

*Units:* $[z]^2$ (squared latent units).

*Interpretation:* High $\sigma_k^2$ indicates the symbol is overloaded---its Voronoi cell contains multiple distinct clusters that should be separated.
:::

:::{prf:definition} Functional Indistinguishability
:label: def-functional-indistinguishability

Two symbols $k_1, k_2$ within the same chart are fusion candidates if the **policy divergence** and **value gap** are negligible:

$$
\mathcal{D}_f(k_1, k_2) := D_{\mathrm{KL}}\left( \pi(\cdot | k_1) \| \pi(\cdot | k_2) \right) + |V(k_1) - V(k_2)|

$$
If $\mathcal{D}_f(k_1, k_2) < \epsilon_{\text{indist}}$, the distinction provides no **control authority**.

*Units:* nat.

*Interpretation:* Symbols are functionally indistinguishable when the policy and value function treat them identically.
:::

:::{prf:algorithm} Lazarus Reallocation
:label: alg-lazarus

**Input:** Dead code $k_{\text{dead}}$ with $P(K = k_{\text{dead}}) < \epsilon_{\text{dead}}$.

**Procedure:**
1. Find the most stressed symbol:

   $$
   k_{\text{stressed}} := \arg\max_k \sigma_k^2

   $$
2. Perform Symbol Fission on $k_{\text{stressed}}$, reusing index $k_{\text{dead}}$:
   - Compute split direction $v_1$ from $\Sigma_{k_{\text{stressed}}}$.
   - Set $e_{k_{\text{dead}}} := e_{k_{\text{stressed}}} + \epsilon v_1$.
   - Update $e_{k_{\text{stressed}}} := e_{k_{\text{stressed}}} - \epsilon v_1$.
3. Update Voronoi cells: The new code inherits half of $k_{\text{stressed}}$'s cell.

**Effect:** Vocabulary migrates to high-information-density regions. Dead codes are "resurrected" where they are needed.

*Connection to existing constraints:* This implements the anti-collapse regularizer from {ref}`sec-calibrating-tolerances`: $\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K) \| \text{Unif}(\mathcal{K}))$.
:::

:::{prf:definition} Symbolic Voronoi Partition
:label: def-voronoi-partition

Let $\mathcal{Z}_i$ be the continuous fiber associated with chart $i$. The codebook $\mathcal{C}_i = \{e_{i,k}\}_{k=1}^{N_v}$ induces a partition $\{\mathcal{V}_k\}$ of $\mathcal{Z}_i$ via:

$$
\mathcal{V}_k := \left\{ z \in \mathcal{Z}_i : d_G(z, e_k) \leq d_G(z, e_j) \;\forall j \neq k \right\}

$$
The probability mass of symbol $k$ is the measure of its Voronoi cell:

$$
P(k) := \int_{\mathcal{V}_k} p(z)\, d\mu_G(z)

$$
where $d\mu_G = \sqrt{\det G}\, dz$ is the Riemannian volume form.
:::

:::{prf:definition} Local Distortion Functional
:label: def-local-distortion

The **local distortion** of symbol $k$ quantifies the representational error within its Voronoi cell:

$$
\mathcal{D}_k := \int_{\mathcal{V}_k} d_G(z, e_k)^2\, p(z)\, d\mu_G(z)

$$
*Units:* $[z]^2$ (weighted squared geodesic distance).

*Relation to geometric tension:* $\mathcal{D}_k = P(k) \cdot \sigma_k^2$, where $\sigma_k^2$ is the intra-symbol variance ({prf:ref}`def-intra-symbol-variance`).
:::

:::{prf:definition} Symbol Utility Functional
:label: def-symbol-utility

The **utility** $U_k$ of symbol $k$ measures its contribution to control authority and predictive accuracy:

$$
U_k := P(k) \cdot I(K=k; A) + P(k) \cdot I(K=k; K_{t+1})

$$
where:
- $I(K=k; A)$ is the mutual information between symbol activation and action selection,
- $I(K=k; K_{t+1})$ is the mutual information between symbol activation and next-state prediction.

*Units:* nat.

*Interpretation:* A symbol with $U_k \approx 0$ neither influences actions nor aids prediction---it is **semantically dead** regardless of its usage frequency.
:::

:::{prf:theorem} Optimal Reallocation Gradient
:label: thm-reallocation-gradient

Let $k_{\text{dead}}$ satisfy $U_{k_{\text{dead}}} < \epsilon_U$ and let $k_{\text{stressed}}$ satisfy $\mathcal{D}_{k_{\text{stressed}}} = \max_k \mathcal{D}_k$. The expected reduction in global distortion per reallocated code is:

$$
\frac{\delta \mathcal{D}}{\delta N_{\text{codes}}} \approx \frac{\mathcal{D}_{k_{\text{stressed}}}}{H(K = k_{\text{stressed}})}

$$
:::

:::{prf:corollary} The Bimodal Instability Theorem (Fission Trigger)
:label: cor-bimodal-instability

Let $K$ be a macro-symbol with associated policy $\pi(\cdot|K)$. The **Structural Stability** of $K$ is inversely proportional to its Varentropy.

If the policy $\pi(\cdot|K)$ is a mixture of two disjoint, equally weighted strategies (a "Buridan's Ass" scenario on a value ridge), the Varentropy satisfies:

$$
V_H(K) = \frac{1}{4}\left(\frac{\Delta Q}{T_c}\right)^2,

$$
where $\Delta Q = |Q_1 - Q_2|$ is the value gap between the modes. In the limit of distinct modes ($\Delta Q \gg T_c$), $V_H$ is maximized, whereas for a uniform (maximum entropy) distribution, $V_H = 0$.

*Units:* $\mathrm{nat}^2$.

**Refined Fission Criterion:**
The **Geometric Tension** $\sigma_k^2$ (Definition {prf:ref}`def-intra-symbol-variance`) is rigorously generalized by the **Varentropy Excess**:

$$
\text{Fission}(K) \iff V_H(K) > \mathcal{V}_{\text{crit}} \quad \text{AND} \quad H(K) > H_{\text{noise}}.

$$
**Interpretation:**
- **High $H$, Low $V_H$:** Aleatoric Uncertainty (Noise/Fog). The distribution is flat. *Action:* Smoothing/Integration.
- **High $H$, High $V_H$:** Epistemic Conflict (Bifurcation). The distribution is multimodal. *Action:* Topological Fission (Node 50).

:::

:::{prf:proposition} Equipartition of Meaning
:label: prop-equipartition

At metabolic equilibrium, the marginal utility per bit is uniform across the ontological hierarchy:

$$
\frac{\partial U}{\partial H(K_{\text{chart}})} \approx \frac{\partial U}{\partial H(K_{\text{code}})} \approx \text{const.}

$$
where $U$ is the total utility functional (value minus complexity cost).

*Interpretation:* The agent allocates representational capacity such that one additional bit of chart-level information provides the same marginal value as one additional bit of symbol-level information. This is the information-theoretic analogue of thermodynamic equipartition.
:::

:::{prf:theorem} Thermodynamic Lower Bound on Hysteresis
:label: thm-thermodynamic-hysteresis-bound

Let $\mathcal{C}$ be a cycle of ontological operations consisting of a fission event $N_c \to N_c + 1$ followed immediately by a fusion event $N_c + 1 \to N_c$. Let $T_c$ be the cognitive temperature and $\mathcal{W}_{\text{comp}}$ be the metabolic work of parameter instantiation. To satisfy the generalized Second Law of Thermodynamics for open cognitive systems (Theorem {prf:ref}`thm-generalized-landauer-bound`), the hysteresis threshold must satisfy:

$$
\epsilon_{\text{hysteresis}} \geq \frac{1}{\beta_{\text{eff}}} \left( \Delta H_{\text{Shannon}} + \frac{1}{T_c}\mathcal{W}_{\text{comp}} \right)

$$
where $\beta_{\text{eff}} = 1/T_c$ is the inverse cognitive temperature and $\Delta H_{\text{Shannon}}$ is the entropy reduction associated with the discarded distinction.

:::

:::{prf:definition} Hyperbolic Frechet Mean for Query Coalescence
:label: def-hyperbolic-frechet-coalescence

Let $\{q_i\}_{i=1}^k \subset \mathbb{D}$ be a set of chart query vectors with associated usage weights $\bar{w}_i := \mathbb{E}[w_i(x)]$ from the Attentive Atlas ({prf:ref}`def-attentive-routing-law`). The **Intrinsic Merged Query** is:

$$
q_{\text{merged}} := \operatorname*{arg\,min}_{q \in \mathbb{D}} \sum_{i=1}^k \bar{w}_i \cdot d^2_{\mathbb{D}}(q, q_i),

$$
where $d_{\mathbb{D}}(x, y) = \operatorname{arccosh}\left(1 + \frac{2\|x-y\|^2}{(1-\|x\|^2)(1-\|y\|^2)}\right)$ is the hyperbolic distance.

*Units:* $[q_{\text{merged}}] = [q_i]$ (dimensionless in the unit disk).

*Cross-reference:* This definition supersedes {prf:ref}`def-query-coalescence` for hyperbolic embeddings.
:::

:::{prf:theorem} Existence and Uniqueness of Fusion Center
:label: thm-frechet-fusion-uniqueness

Since the Poincare disk $(\mathbb{D}, G)$ is a complete, simply connected Riemannian manifold with non-positive sectional curvature ($K=-1$), it is a Hadamard space (global CAT(0) space). The squared distance function $d^2_{\mathbb{D}}(\cdot, y)$ is strictly convex. Therefore, the functional $F(q) = \sum \bar{w}_i d^2_{\mathbb{D}}(q, q_i)$ admits a unique global minimizer.

:::

:::{prf:remark} Computational Algorithm
:label: rem-frechet-algorithm

The minimizer can be computed via Riemannian gradient descent:

$$
q_{t+1} = \operatorname{Exp}_{q_t}\left( -\eta \sum_i \bar{w}_i \operatorname{Log}_{q_t}(q_i) \right)

$$
where:
- $\operatorname{Exp}_p: T_p\mathbb{D} \to \mathbb{D}$ is the exponential map at $p$
- $\operatorname{Log}_p: \mathbb{D} \to T_p\mathbb{D}$ is the logarithmic map (inverse of exponential)

For the Poincare disk, these have closed-form expressions via Mobius operations ({ref}`sec-bulk-boundary-independence`).

*Complexity:* $O(k \cdot d)$ per iteration, where $k$ is the number of charts being merged and $d$ is the embedding dimension.
:::

:::{prf:theorem} Fission Inhibition Corollary
:label: thm-fission-inhibition

Let $\mathcal{E}^{(\ell)}$ be the encoder at scale $\ell$. A Topological Fission event at layer $\ell$ (increasing chart count $N_c^{(\ell)} \to N_c^{(\ell)}+1$) strictly reduces the probability of fission at layer $\ell+1$.

:::

:::{prf:corollary} Hierarchical Stability
:label: cor-hierarchical-stability

The stacked architecture is **inherently stable** against fission cascades. Ontological expansion at coarse scales (low $\ell$) pre-empts the need for expansion at fine scales (high $\ell$).

*Interpretation:* If the agent learns a new high-level concept (e.g., "mammal"), the residual variance available to learn low-level distinctions (e.g., specific breeds) is reduced. The hierarchy self-regulates, preventing runaway complexity growth.
:::

## 07_cognition/05_metabolism.md

:::{prf:definition} Metabolic Flux
:label: def-metabolic-flux

Let $\rho(s, z)$ be the belief density evolving in computation time $s$ according to the WFR continuity equation (Definition {prf:ref}`def-the-wfr-action`):

$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r.

$$
We define the **Metabolic Flux** $\dot{\mathcal{M}}: \mathbb{R}_{\ge 0} \to \mathbb{R}_{\ge 0}$ as:

$$
\dot{\mathcal{M}}(s) := \sigma_{\text{met}} \int_{\mathcal{Z}} \left( \|v_s(z)\|_G^2 + \lambda^2 |r_s(z)|^2 \right) \rho(s, z) \, d\mu_G,

$$
where:
- $\sigma_{\text{met}} > 0$ is the **metabolic resistance coefficient** (units: nat$\cdot$step)
- $v_s(z)$ is the velocity field at computation time $s$
- $r_s(z)$ is the reaction rate (mass creation/destruction)
- $\lambda$ is the WFR length-scale (Definition {prf:ref}`def-the-wfr-action`)
- $d\mu_G = \sqrt{\det G} \, dz$ is the Riemannian volume form

*Physical interpretation:* The metabolic flux measures the instantaneous rate of energy dissipation required to update the belief distribution. Transport ($\|v\|_G^2$) represents the cost of moving probability mass; reaction ($|r|^2$) represents the cost of creating or destroying mass. The WFR action is the kinetic energy of the belief flow.

:::

:::{prf:theorem} Conditional Generalized Landauer Bound
:label: thm-generalized-landauer-bound

Let $(\mathcal{Z},G)$ be a compact $C^2$ Riemannian domain with a positive $C^1$
density $\rho_s$ and $C^1$ fields $v_s,r_s$ satisfying the WFR continuity equation.
Assume no transport flux through $\partial\mathcal{Z}$ and mass preservation
$\int_{\mathcal Z}\rho_s r_s\,d\mu_G=0$. Define

$$
E_v:=\int_{\mathcal Z}\rho_s\|v_s\|_G^2\,d\mu_G,\quad
E_r:=\int_{\mathcal Z}\rho_s r_s^2\,d\mu_G,\quad
I_\rho:=\int_{\mathcal Z}\rho_s\|\nabla\ln\rho_s\|_G^2\,d\mu_G,
\quad J_\rho:=\int_{\mathcal Z}\rho_s(\ln\rho_s)^2\,d\mu_G.
$$

Then, for $H(\rho_s)=-\int_{\mathcal Z}\rho_s\ln\rho_s\,d\mu_G$,

$$
\left|\frac{d}{ds}H(\rho_s)\right|
\le \sqrt{I_\rho E_v}+\sqrt{J_\rho E_r}.
$$

Consequently, the Landauer-form lower bound

$$
\dot{\mathcal M}(s)\ge T_c\left|\frac{d}{ds}H(\rho_s)\right|
$$

holds on any calibrated regime where

$$
\sigma_{\mathrm{met}}(E_v+\lambda^2E_r)
\ge T_c\big(\sqrt{I_\rho E_v}+\sqrt{J_\rho E_r}\big).
$$

The calibration, regularity, and boundary conditions are hypotheses of this
statement; they do not follow from the WFR continuity equation alone.

:::

:::{prf:definition} Metabolic Potential
:label: def-metabolic-potential

We define $\Psi_{\text{met}}(s) := \int_0^s \dot{\mathcal{M}}(u) \, du$ as the cumulative metabolic energy dissipated during a single interaction step $t$ for an internal rollout of duration $s$. Units: $[\Psi_{\text{met}}] = \text{nat}$.

:::

:::{prf:axiom} Dual-Horizon Action
:label: ax-dual-horizon-action

For any interaction step $t$, the agent selects a total computation budget $S \in [0, S_{\max}]$ that minimizes the **Deliberation Action** $\mathcal{S}_{\text{delib}}$:

$$
\mathcal{S}_{\text{delib}}[S] = -\underbrace{\mathbb{E}_{z \sim \rho_S} [V(z)]}_{\text{Expected Terminal Value}} + \underbrace{\Psi_{\text{met}}(S)}_{\text{Computational Cost}},

$$
where $V(z)$ is the task potential ({ref}`sec-hodge-decomposition-of-value`). Units: $[\mathcal{S}_{\text{delib}}] = \text{nat}$.

*Physical interpretation:* The agent faces a trade-off: longer deliberation ($S$ large) improves the expected value $\langle V \rangle_{\rho_S}$ by refining the belief toward high-value regions, but incurs greater metabolic cost $\Psi_{\text{met}}(S)$. The optimal $S^*$ balances these competing pressures.

*Remark (Sign convention).* We write $-\langle V \rangle$ because the agent seeks to **maximize** value. The Deliberation Action $\mathcal{S}_{\text{delib}}$ is minimized when value is maximized and cost is minimized.

:::

:::{prf:theorem} Deliberation Optimality Condition
:label: thm-deliberation-optimality-condition

Let $\rho_s$ evolve as a gradient flow of $V$ under WFR dynamics. The optimal computation budget $S^*$ satisfies:

$$
\left. \frac{d}{ds} \langle V \rangle_{\rho_s} \right|_{s=S^*} = \dot{\mathcal{M}}(S^*),

$$
provided such an $S^*$ exists in $(0, S_{\max})$.

:::

:::{prf:theorem} Fast/Slow Phase Transition
:label: thm-fast-slow-phase-transition

Let $\Gamma(s) := \left| \frac{d}{ds} \langle V \rangle_{\rho_s} \right|$ be the **Value-Improvement Rate**. There exists a critical threshold such that:

1. **Reflexive Regime (Fast):** If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $S^* = 0$. The agent executes an immediate action based on the prior $\rho_0$.

2. **Deliberative Regime (Slow):** If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $S^* > 0$. The agent enters a planning state, terminating only when the marginal gain in Value equals the marginal metabolic cost.

:::

:::{prf:theorem} Generalized Stopping for Non-Conservative Fields
:label: thm-generalized-stopping

When the Value Curl does not vanish ($\mathcal{F} \neq 0$, Definition {prf:ref}`def-value-curl`), the agent converges to a Non-Equilibrium Steady State (Theorem {prf:ref}`thm-ness-existence`) rather than a fixed point. The stopping criterion generalizes as follows:

**Conservative Case ($\mathcal{F} = 0$):** Stop when the Value-Improvement Rate equals the metabolic cost:

$$
\Gamma(S^*) = \dot{\mathcal{M}}(S^*)

$$

**Non-Conservative Case ($\mathcal{F} \neq 0$):** Stop when the **orbit parameters converge**:

$$
\frac{d}{ds}\|\text{Orbit}(s)\|_{\text{param}} < \epsilon_{\text{orbit}}

$$
even if the agent continues moving within the limit cycle.

*Remark.* In the conservative case, convergence is to a fixed point ($\dot{z} \to 0$). In the non-conservative case, convergence is to a stable limit cycle (periodic orbit with constant parameters).

**Operational Criterion:** Define the orbit-change metric as:

$$
\Delta_{\text{orbit}}(s) := \left\| \oint_{\gamma_s} \mathcal{R} - \oint_{\gamma_{s-\delta}} \mathcal{R} \right\|

$$
where $\gamma_s$ is the closed trajectory over one cycle at time $s$. Stop when $\Delta_{\text{orbit}}(s) < \epsilon_{\text{orbit}}$.

*Remark.* In the non-conservative case, the agent accumulates reward along periodic trajectories. Deliberation terminates when the orbit parameters stabilize, not when motion ceases.

:::

:::{prf:theorem} Total Entropy Production
:label: thm-total-entropy-production

The total entropy production rate of the agent $\sigma_{\text{tot}}$ during computation is:

$$
\sigma_{\text{tot}}(s) := \frac{d}{ds} H(\rho_s) + \frac{1}{T_c} \dot{\mathcal{M}}(s) \ge 0.

$$
:::

:::{prf:definition} Cognitive Carnot Efficiency
:label: def-cognitive-carnot-efficiency

The **Carnot limit** for cognitive systems is $\eta_{\text{thought}} = 1$, achieved when the belief update is a reversible isothermal process. Real agents operate at $\eta_{\text{thought}} < 1$ due to:
1. **Friction:** Non-optimal transport paths (geodesic deviation)
2. **Irreversibility:** Finite-rate updates (non-quasi-static processes)
3. **Dissipation:** Exploration noise ($T_c > 0$)

:::

## 07_cognition/06_causality.md

:::{prf:definition} The Interventional Surgery
:label: def-the-interventional-surgery

Let $U_t$ denote hidden environment state not included in $z_t$, and let the structural transition be
$P_\partial(z_{t+1}\mid z_t,a_t,U_t)$. We define the **Interventional Operator**
$\mathfrak{I}: \mathcal{P}(\mathcal{Z} \times \mathcal{A} \times \mathcal{Z}) \to
\mathcal{P}(\mathcal{Z} \times \mathcal{A} \times \mathcal{Z})$—equivalent to Pearl's $do(a_t)$
{cite}`pearl2009causality`—as a surgery that replaces the policy-induced action law by an exogenous action.

Geometrically, $\mathfrak{I}$ replaces the policy-induced motor flux by an exogenous motor flux. The sensory
Dirichlet condition on $z_t$ and the state dependence of the environment mechanism are unchanged.

Formally, the operator acts by truncated factorization:

$$
P(z' | z, do(a)) := \int P_\partial(z' | z, a, u)P(u|z)\,du,

$$
where the structural mechanism is preserved but $a$ is no longer generated by the observational policy. For marginal interventional queries:

$$
P_{\text{int}}(z' | do(a)) = \int_{\mathcal{Z}\times\mathcal U} P_\partial(z' | \tilde z, a, u)
P(u|\tilde z)P_{\text{pre}}(\tilde z) \, d\mu_G(\tilde z)\,du,

$$
where $P_{\text{pre}}(\tilde{z})$ is the pre-intervention distribution over latent states.

:::

:::{prf:lemma} Interventional Joint Under Exogenous Action
:label: lem-the-interventional-singularity

For a fixed intervention $do(a=a_0)$, the post-intervention joint has the factorization
$$
P_{\mathrm{pre}}(z)\,\delta_{a_0}(a)\,P_\partial(z'\mid z,a_0,u)P(u\mid z),
$$
after marginalizing any hidden environment state. It is singular with respect to an observational joint whose
action conditional $\pi(a\mid z)$ is a density and does not put an atom at $a_0$.

:::

:::{prf:definition} Causal Information Potential
:label: def-causal-information-potential

Define the **action-conditioned Causal Information Potential** $\Psi_{\text{causal}}^{\mathrm{act}}: \mathcal{Z} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$ as the Expected Information Gain (EIG) {cite}`lindley1956measure` regarding the transition parameters $\theta_W$ at state-action pair $(z, a)$:

$$
\Psi_{\text{causal}}^{\mathrm{act}}(z, a) := \mathbb{E}_{z' \sim \bar{P}(\cdot | z, do(a))} \left[ D_{\text{KL}} \left( p(\theta_W | z, a, z') \| p(\theta_W | z, a) \right) \right].

$$
Define the state potential used by the drift as $\Psi_{\text{causal}}(z):=\mathbb{E}_{a\sim\pi(\cdot|z)}[\Psi_{\text{causal}}^{\mathrm{act}}(z,a)]$. Units: $[\Psi_{\text{causal}}]=\text{nat}$.

*Physical interpretation:* $\Psi_{\text{causal}}^{\mathrm{act}}(z,a)$ measures how much the agent expects to learn about
the World Model parameters by executing $a$ from $z$. The state potential averages this quantity under the declared
policy; a max or a different design distribution would define a different state field.

:::

:::{prf:proposition} Conditional Interventional Gap
:label: thm-the-interventional-gap

Let $U$ be a hidden environment variable. Define
$$
P_{\text{obs}}(z'|z,a):=\int P_\partial(z'|z,a,u)P(u|z,a)\,du,
\qquad
P_{\text{int}}(z'|z,do(a)):=\int P_\partial(z'|z,a,u)P(u|z)\,du.
$$
The **Causal Deficit** is the KL divergence between these two declared kernels:

$$
\Delta_{\text{causal}}(z, a) := D_{\text{KL}} \left( P_{\text{int}}(z' | z, do(a)) \| P_{\text{obs}}(z' | z, a) \right).

$$
*Interpretation:* The Causal Deficit measures the discrepancy between interventional and observational predictions
under the hidden-state model. If no hidden common cause is present, or if $U$ is included in $z$, the two kernels
coincide and the deficit is identically zero. A positive value therefore requires the explicit hidden-state or
learned-model distinction above; it is not implied by the one-kernel Markov model.

:::

:::{prf:remark} Epistemic Curiosity Diagnostic
:label: cor-epistemic-curiosity-filter

The Causal Information Potential $\Psi_{\text{causal}}$ (Definition {prf:ref}`def-causal-information-potential`) may be
compared with posterior varentropy, but no proportionality follows from the EIG definition without a specified
posterior family and likelihood.

Let $V_H[P(\theta_W | z, a, z')]$ denote the Varentropy of the posterior over World Model parameters after observing
transition $(z, a) \to z'$. For a chosen posterior family one may test:

$$
\nabla \Psi_{\text{causal}} \stackrel{?}{\propto} \nabla \mathbb{E}_{z'} \left[ V_H [P(\theta_W | z, a, z')] \right].

$$
*Units:* nat (for $\Psi_{\text{causal}}$), $\mathrm{nat}^2$ (for $V_H$).

**Operational Significance:** The Curiosity Force can be ranked using this varentropy diagnostic when the same
posterior and sampling measure are used on both sides. It should not replace EIG without that calibration.

1. **High Entropy, Low Varentropy:** The World Model is confidently predicting "I don't know" (White Noise). The gradient $\nabla \Psi \approx 0$. The agent ignores this region (solves the "Noisy TV" problem).
2. **High Entropy, High Varentropy:** The World Model oscillates between distinct causal hypotheses ($H_1$: "Object falls", $H_2$: "Object floats"). The gradient $\nabla \Psi$ is maximal. The agent is strongly attracted to this state to resolve the structural ambiguity.

**Implementation:** The Experimental Sieve (Algorithm 32.5.1) may log varentropy alongside EIG; the selection rule
is the declared EIG or an explicitly chosen calibrated proxy.

:::

:::{prf:definition} Augmented Drift Model
:label: thm-augmented-drift-law

Given a state potential \(V\), a state-level causal information potential
\(\Psi_{\text{causal}}\), and the declared curl mobility
\(\mathcal M_{\text{curl}}\), define the exploratory drift by

$$
\dot z =
\mathcal M_{\text{curl}}\!\left(
  -G^{-1}\nabla_A V
  +\beta_{\text{exp}}G^{-1}\nabla\Psi_{\text{causal}}
\right),
\qquad
\mathcal M_{\text{curl}}
=(I-\beta_{\text{curl}}G^{-1}\mathcal F)^{-1}.
$$

Here \(\beta_{\text{exp}}\ge0\) is the exploration coefficient and
\(\nabla_A V=\nabla V-A\) uses the reward-field convention. This is a
chosen control law. The action functional written in the former proof
does not generate the curl mobility or the overdamped reduction, so no
variational theorem is asserted here.

*Physical interpretation:* The curiosity term points toward states with
larger declared expected information gain. Its use as an intrinsic reward
requires a calibrated interventional model and does not by itself prove
that learning or causal identification improves.

:::

:::{prf:remark} Conditional Scientific-Method Drift
:label: cor-scientific-method-as-geodesic

When \(V\) is constant, \(A=0\), and the curl mobility is the identity,
the deterministic second-order model with potential
\(-\beta_{\text{exp}}\Psi_{\text{causal}}\) has

$$
\ddot z^m+\Gamma^m_{ij}\dot z^i\dot z^j
= \beta_{\text{exp}}G^{mk}\partial_k\Psi_{\text{causal}}.
$$

This is a formal curiosity-driven control model. Calling its trajectories
scientific experiments requires the causal-information and intervention
hypotheses above; it is not a theorem that agents maximize knowledge.

:::

:::{prf:definition} Interventional Closure Criterion
:label: thm-interventional-closure

For a specified family of interventions, call the macro-ontology $K$ **Interventionally Closed** when the
predictability of the macro-state is invariant under those $do$-operations:

$$
I(K_{t+1} ; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0.

$$
*Interpretation:* If an agent moves an object (intervention), and the resulting macro-state $K_{t+1}$ depends on micro-texture $z_{\text{tex}}$ that was previously labeled "noise," the ontology has failed. The intervention has **exposed a hidden variable**, triggering **Ontological Expansion** ({ref}`sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`).

*Scope.* This is a criterion to be tested under the interventional measure. Observational closure alone does not
imply interventional closure: the intervention can change the distribution of hidden microvariables or expose a
back-door path. A positive conditional mutual information is evidence for an omitted variable only after positivity,
support, and mechanism-invariance assumptions have been checked; it is not an automatic iff theorem.

*Remark (Interventional Debugging).* Theorem {prf:ref}`thm-interventional-closure` provides a diagnostic for ontological adequacy: if the agent's predictions fail specifically under intervention but succeed under observation, the ontology contains a hidden confounder. This is the geometric manifestation of Simpson's paradox {cite}`pearl2009causality`. Algorithmic approaches to discovering such confounders are developed in the causal discovery literature {cite}`spirtes2000causation`.

:::

## 07_cognition/07_metabolic_transducer.md

:::{prf:definition} The Reward Flux
:label: def-reward-flux-harvesting

The **Reward Flux** $J_r(t)$ is the instantaneous rate of reward accumulation (Definition {prf:ref}`def-the-reward-flux`):

$$
J_r(t) = \langle \mathcal{R}(z_t), v_t \rangle_G = r_t

$$

where $\mathcal{R}$ is the reward 1-form ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`) and $v_t = \dot{z}_t$ is the velocity in latent space.

*Units:* $[J_r] = \text{nats/step}$ (information-theoretic) or $[\text{utility/step}]$ (decision-theoretic).

*Interpretation:* A positive reward $r_t > 0$ indicates the agent has navigated to a state with lower environmental entropy—a configuration where resources (food, fuel, safety) are localized and accessible.

:::

:::{prf:definition} Information Utility
:label: def-information-utility

The **Information Utility** $\mathcal{I}_{\text{util}}(r_t)$ quantifies the actionable information content of the reward signal:

$$
\mathcal{I}_{\text{util}}(r_t) := I(Z_t; R_t) = H[R_t] - H[R_t \mid Z_t]

$$

where $I(Z_t; R_t)$ is the mutual information between the agent's state $Z_t$ and the reward $R_t$.

*Operational interpretation:* This is the reduction in uncertainty about environmental resources achieved by navigating to state $z_t$ and observing reward $r_t$.

*Units:* $[\mathcal{I}_{\text{util}}] = \text{nats}$ (or bits if using $\log_2$).

*Simplification:* When the reward signal is deterministic given state, $H[R_t \mid Z_t] = 0$, so $\mathcal{I}_{\text{util}}(r_t) = H[R_t]$. In practice, we often use the approximation $\mathcal{I}_{\text{util}}(r_t) \approx |r_t|$ for rewards measured in natural units.

:::

:::{prf:axiom} The Szilard Correspondence (Information-Work Duality)
:label: ax-szilard-correspondence

Information about low-entropy configurations can be converted to extractable work. Specifically, if an agent possesses $I$ nats of mutual information with a thermal reservoir at temperature $T_{\text{env}}$, it can extract at most:

$$
W_{\max} = k_B T_{\text{env}} \cdot I

$$

joules of work, where $k_B$ is Boltzmann's constant.

*Physical basis:* This is the inverse of Landauer's principle. Landauer states that erasing 1 bit costs $k_B T \ln 2$ joules. Szilard's engine demonstrates that acquiring 1 bit about a system enables extracting $k_B T \ln 2$ joules. The two are thermodynamically dual.

*Cognitive interpretation:* A reward signal $r_t > 0$ encodes mutual information between the agent's state and resource availability. This information, when acted upon, enables work extraction from the environment.

:::

:::{prf:theorem} The Transducer Bound
:label: thm-szilard-transducer-bound

Let $r_t$ be the instantaneous reward signal with information content $\mathcal{I}_{\text{util}}(r_t)$ nats. The maximum free energy extractable per unit time is bounded by:

$$
\dot{E}_{\text{in}}^{\max}(t) = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where $T_{\text{env}}$ is the environmental temperature (characterizing energy availability).

:::

:::{prf:definition} The Metabolic Transducer Operator
:label: def-metabolic-transducer

The **Metabolic Transducer** $\mathfrak{T}_{\text{harvest}}$ is the operator converting the reward flux to free energy flux:

$$
\dot{E}_{\text{in}}(t) = \mathfrak{T}_{\text{harvest}}(r_t) := \eta \cdot k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)

$$

where:
- $k_B \approx 1.38 \times 10^{-23}$ J/K is **Boltzmann's constant**
- $T_{\text{env}}$ is the **environmental temperature** (Kelvin)
- The product $k_B T_{\text{env}}$ is the **energy-per-nat conversion factor** (Joules/nat)
- $\eta \in [0, 1]$ is the **transduction efficiency** (Carnot-bounded, see Theorem {prf:ref}`thm-carnot-transduction-bound`)
- $\mathcal{I}_{\text{util}}(r_t)$ is the **information utility** of the reward signal (Definition {prf:ref}`def-information-utility`)

*Units:* $[\mathfrak{T}] = \text{Joules/step}$ (power).

*Simplified form:* For dimensionless analysis with $k_B = 1$, we write:

$$
\mathfrak{T}_{\text{harvest}}(r_t) = \eta \cdot T_{\text{env}} \cdot r_t

$$

where $r_t$ is measured in nats.

:::

:::{prf:definition} The Internal Battery
:label: def-internal-battery

The **Internal Battery** $B(t)$ is a scalar state variable representing the agent's stored free energy:

$$
B: [0, \infty) \to [0, B_{\max}]

$$

where:
- $B_{\max}$ is the maximum storage capacity (Joules)
- $B(0) = B_0$ is the initial endowment

*Units:* $[B] = \text{Joules}$ (energy).

*Interpretation:* The battery represents the agent's capacity for future computation. In biological systems, this corresponds to ATP/glucose reserves; in artificial systems, to available compute budget.

:::

:::{prf:axiom} Energy Conservation (First Law)
:label: ax-energy-conservation-battery

The battery evolves according to the First Law of Thermodynamics:

$$
\frac{dB}{dt} = \underbrace{\mathfrak{T}_{\text{harvest}}(r_t)}_{\text{Income}} - \underbrace{\dot{\mathcal{M}}(t)}_{\text{Metabolic Cost}} - \underbrace{\gamma_{\text{leak}} B(t)}_{\text{Passive Dissipation}}

$$

where:
- $\mathfrak{T}_{\text{harvest}}(r_t)$ is the transduced energy from rewards (Definition {prf:ref}`def-metabolic-transducer`)
- $\dot{\mathcal{M}}(t)$ is the metabolic cost from Theorem {prf:ref}`thm-generalized-landauer-bound`
- $\gamma_{\text{leak}} \geq 0$ is the passive self-discharge rate (basal metabolic rate)

*Terminal Condition:* If $B(t) \leq 0$, the agent undergoes **Thermodynamic Death**. The metric collapses (Theorem {prf:ref}`thm-fading-metric-law`), inference halts, and the agent can no longer perform coherent computation.

:::

:::{prf:theorem} The Autopoietic Inequality
:label: thm-autopoietic-inequality

Let $\tau > 0$ be a target survival horizon. A **sufficient condition** for the agent to survive at time $\tau$ (i.e., $B(\tau) > 0$) is:

$$
\int_0^\tau \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) dt > \gamma_{\text{leak}} \int_0^\tau B(t) \, dt - B_0

$$

*Equivalently:* The time-averaged **Net Harvest Rate** must be positive:

$$
\langle \mathfrak{T} - \dot{\mathcal{M}} \rangle_\tau > \gamma_{\text{leak}} \langle B \rangle_\tau - \frac{B_0}{\tau}

$$

:::

:::{prf:corollary} The Survival Objective
:label: cor-survival-objective

The agent's fundamental objective is not reward maximization but **energy surplus maximization**:

$$
\mathcal{J}_{\text{survival}} = \mathbb{E}\left[ \int_0^\infty \left( \mathfrak{T}_{\text{harvest}}(r_t) - \dot{\mathcal{M}}(t) \right) e^{-\gamma_{\text{leak}} t} \, dt \right]

$$

Standard reward maximization $\max \mathbb{E}[\sum_t \gamma^t r_t]$ emerges as a degenerate case when:
1. Metabolic cost $\dot{\mathcal{M}} \to 0$ (free computation)
2. Transduction efficiency $\eta \to 1$ (perfect conversion)
3. Battery capacity $B_{\max} \to \infty$ (unlimited storage)

:::

:::{prf:theorem} The Information-Maintenance Cost
:label: thm-information-maintenance-cost

Maintaining Fisher Information $I_F$ on the latent manifold $(\mathcal{Z}, G)$ requires continuous energy expenditure:

$$
\dot{E}_{\text{maintain}} \geq \frac{1}{2} T_c \cdot I_F

$$

where $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`) and $I_F$ is the Fisher Information of the belief distribution.

:::

:::{prf:theorem} The Fading Metric Law
:label: thm-fading-metric-law

When available energy $B(t)$ falls below the maintenance requirement, the effective metric contracts. The **effective metric** is:

$$
G_{ij}^{\text{eff}}(z, B) = f\left(\frac{B}{B_{\text{crit}}}\right) \cdot G_{ij}(z)

$$

where:
- $G_{ij}(z)$ is the full-capacity metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $B_{\text{crit}}$ is the **critical energy** required to sustain full metric resolution
- $f: [0, \infty) \to [0, 1]$ is the **fading function** with $f(0) = 0$, $\lim_{x \to \infty} f(x) = 1$

**Specific form:** The fading function satisfying thermodynamic constraints is:

$$
f(x) = 1 - e^{-x}

$$

This gives exponential saturation: $f(x) \approx x$ for $x \ll 1$ (linear regime) and $f(x) \approx 1$ for $x \gg 1$ (saturation).

:::

:::{prf:corollary} Consequences of Metric Fading
:label: cor-metric-fading-consequences

As $B(t) \to 0$, the following degenerations occur:

1. **Resolution Loss:** Geodesic distances collapse:
   $$d_G^{\text{eff}}(z, z') = \sqrt{f(B/B_{\text{crit}})} \cdot d_G(z, z') \to 0$$
   Distinct concepts become indistinguishable.

2. **Inertia Loss:** The mass term in the geodesic SDE (Definition {prf:ref}`def-bulk-drift-continuous-flow`) vanishes. The agent loses momentum and becomes dominated by thermal noise.

3. **Causal Dissolution:** The Causal Information Bound ({ref}`sec-causal-information-bound`, Theorem {prf:ref}`thm-causal-information-bound`) collapses:
   $$I_{\max}^{\text{eff}} = \frac{\text{Area}(\partial\mathcal{Z})}{4\ell_L^2} \cdot f(B/B_{\text{crit}}) \to 0$$
   The agent's representational capacity vanishes.

4. **Control Loss:** The policy gradient $\nabla_z \Phi_{\text{eff}}$ scales with metric, so control authority degrades.

:::

:::{prf:corollary} The Starvation-Hallucination Regime
:label: cor-starvation-hallucination

As $B(t) \to 0$, the signal-to-noise ratio of internal dynamics degrades:

$$
\text{SNR}_{\text{dynamics}} = \frac{\|v\|_{G^{\text{eff}}}^2}{2T_c} \propto f(B/B_{\text{crit}}) \to 0

$$

In this regime:
- The drift term $v = -G^{-1} \nabla \Phi$ vanishes relative to diffusion $\sqrt{2T_c} dW$
- The agent performs a **random walk** in latent space
- Internal trajectories are indistinguishable from noise: **hallucination**

*Biological analogue:* Hypoglycemia causes confusion, disorientation, and hallucinations before coma—the same phenomenology predicted by metric fading. See also the Cognitive Temperature (Definition {prf:ref}`def-cognitive-temperature`) which controls the noise-to-signal ratio in latent dynamics.

:::

:::{prf:definition} The Homeostatic Potential
:label: def-homeostatic-potential

The battery level $B(t)$ induces a scalar potential field acting on the policy:

$$
\Phi_{\text{homeo}}(z, B) = \frac{\lambda_{\text{surv}}}{B + \epsilon} \cdot \mathbb{1}[z \in \mathcal{Z}_{\text{food}}]

$$

where:
- $\lambda_{\text{surv}} > 0$ is the **survival weight** (dimensionless priority)
- $\epsilon > 0$ is a regularization constant preventing singularity
- $\mathcal{Z}_{\text{food}} \subset \mathcal{Z}$ is the **food region** (states where $\mathfrak{T}(r) > 0$)

*Units:* $[\Phi_{\text{homeo}}] = [\Phi_{\text{task}}] = \text{nats}$ (log-probability scale).

:::

:::{prf:theorem} The Augmented Value Equation
:label: thm-augmented-value-equation

The total effective potential combines task and homeostatic contributions:

$$
\Phi_{\text{total}}(z, B) = \Phi_{\text{task}}(z) + \Phi_{\text{homeo}}(z, B)

$$

The value function satisfies the augmented screened Poisson equation ({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`):

$$
(-\Delta_{G^{\text{eff}}} + \kappa^2) V = \rho_r + \rho_{\text{homeo}}

$$

where:
- $G^{\text{eff}} = f(B/B_{\text{crit}}) \cdot G$ is the faded metric (Theorem {prf:ref}`thm-fading-metric-law`)
- $\rho_{\text{homeo}} = -\Delta \Phi_{\text{homeo}}$ is the homeostatic source term
- In the stationary diffusion convention, the screening coefficient is $\kappa^2=\lambda/T_c$ with
  $\lambda=-\ln\gamma/\Delta t$; it is unchanged by the battery-dependent source term.

*Consequence:* Both the metric (geometry) and the source term (drive) depend on battery state.

:::

:::{prf:corollary} Priority Inversion at Low Battery
:label: cor-priority-inversion

As $B \to 0$:

1. **Homeostatic dominance:** $\Phi_{\text{homeo}} \propto 1/B \to \infty$ while $\Phi_{\text{task}}$ remains bounded
2. **Gradient steering:** $\nabla_z \Phi_{\text{total}} \approx \nabla_z \Phi_{\text{homeo}}$ points toward $\mathcal{Z}_{\text{food}}$
3. **Priority inversion:** Task objectives become irrelevant; survival dominates

*Behavioral consequence:* A starving agent abandons task pursuit and seeks energy. This behavior emerges from the thermodynamic structure of autopoietic systems.

:::

:::{prf:theorem} The Carnot Bound on Transduction
:label: thm-carnot-transduction-bound

The transduction efficiency is bounded by the Carnot limit:

$$
\eta \leq \eta_{\text{Carnot}} = 1 - \frac{T_c}{T_{\text{env}}}

$$

where $T_c$ is the agent's cognitive temperature and $T_{\text{env}}$ is the environmental temperature.

:::

:::{prf:definition} The Waste Heat Flux
:label: def-waste-heat-flux

The **Waste Heat Flux** is the rate at which the agent must dump entropy to the environment:

$$
\dot{Q}_{\text{waste}} = (1 - \eta) \cdot \mathfrak{T}_{\text{gross}}(r_t) + \dot{\mathcal{M}}(t)

$$

where $\mathfrak{T}_{\text{gross}} = k_B T_{\text{env}} \cdot \mathcal{I}_{\text{util}}(r_t)$ is the gross transduction before efficiency losses.

*Units:* $[\dot{Q}_{\text{waste}}] = \text{Watts}$ (power).

*Interpretation:* All non-useful energy becomes waste heat that must be radiated to maintain thermal equilibrium.

:::

:::{prf:corollary} The Thermal Runaway Condition
:label: cor-thermal-runaway

Let $\dot{Q}_{\text{radiate}}$ be the maximum heat dissipation rate (determined by surface area, environment, cooling mechanisms). If:

$$
\dot{Q}_{\text{waste}} > \dot{Q}_{\text{radiate}}

$$

then the agent's internal temperature $T_c$ increases. This triggers a positive feedback loop:

1. $T_c \uparrow$ $\Rightarrow$ $\eta_{\text{Carnot}} = 1 - T_c/T_{\text{env}} \downarrow$
2. Lower $\eta$ $\Rightarrow$ more waste heat for same harvesting
3. More waste heat $\Rightarrow$ $T_c \uparrow$ (feedback)

*Terminal state:* $T_c \to T_{\text{env}}$, $\eta \to 0$, no harvesting possible, death by thermal runaway.

*Biological analogue:* Hyperthermia/heat stroke—metabolic rate increases with temperature, but cooling capacity is bounded, leading to runaway heating.

:::

:::{prf:definition} The Thermal Operating Envelope
:label: def-thermal-operating-envelope

The agent is **thermally viable** if there exists a steady-state solution to:

$$
\dot{Q}_{\text{waste}}(T_c) = \dot{Q}_{\text{radiate}}(T_c)

$$

with $T_c < T_{\text{env}}$ and $\eta(T_c) > \eta_{\min}$ where $\eta_{\min}$ is the minimum efficiency for survival (from Theorem {prf:ref}`thm-autopoietic-inequality`).

The **Thermal Operating Envelope** is the region in $(T_c, \dot{\mathcal{M}}, \dot{Q}_{\text{radiate}})$ space where this condition holds.

:::

## 07_cognition/08_intersubjective_metric.md

:::{prf:definition} Metric Friction
:label: def-metric-friction

Let $\phi_{A \to B}: \mathcal{Z}_A \to \mathcal{Z}_B$ be a $C^1$ diffeomorphism on the comparison region. **Metric Friction** is the squared tensor norm of the pullback metric distortion:

$$
\Phi_{AB}(z) := \bigl\|G_A(z) - (\phi_{A \to B}^{*}G_B)(z)\bigr\|_{G_A}^{2}.

$$

Here $\|\cdot\|_{G_A}$ is the tensor norm induced by $G_A$. Assume that $\mathcal{Z}_A$ and $\mathcal{Z}_B$ have
the same dimension and that $\phi_{A\to B}$ is a diffeomorphism on the region under study. The scalar $\Phi_{AB}$ is
distinct from the gauge curvature $\mathcal{F}_{AB}$ defined below.

*Interpretation:* $\Phi_{AB}=0$ means that the selected map is an isometry on the comparison region. A positive value
records distortion, but it does not by itself imply a loss of cooperation or a mismatch in causal structure.

*Units:* The units depend on the coordinate convention for the metrics. The normalized quantity
$\widetilde{\Phi}_{AB}:=\Phi_{AB}/\|G_A\|_{G_A}^{2}$ is dimensionless.

:::

:::{prf:remark} Metric Friction and Cooperative Utility
:label: lem-friction-bounds-utility

Let $V_{\text{coop}}$ denote the cooperative value for a specified task. A frequently useful modelling assumption is
the bound:

$$
V_{\text{coop}} \leq V_{\text{max}} \cdot \exp\left(-\frac{\Phi_{AB}}{\mathcal{F}_0}\right)

$$

where $V_{\text{max}}$ is the optimal cooperative value under perfect alignment and $\mathcal{F}_0$ is a characteristic friction scale.

This exponential dependence is not a consequence of the metric definition. To derive it one would need a task-specific
relation between the pullback distortion and the angle between the ordinary gradients of $V_A$ and $V_B\circ\phi$,
as well as a non-negative value range. Without that additional hypothesis, $\Phi_{AB}$ remains a diagnostic rather
than a utility theorem.

:::

:::{prf:definition} The Inter-Agent Connection
:label: def-inter-agent-connection

Let agents $A$ and $B$ each possess a nuisance bundle with gauge connection $A^{(A)}$ and $A^{(B)}$ (Definition
{prf:ref}`def-strategic-connection`). Before locking, choose a comparison region $\mathcal{D}_{AB}\subset\mathcal{Z}_A$
and a $C^1$ correspondence $\phi_{A\to B}:\mathcal{D}_{AB}\to\mathcal{Z}_B$. Pull the second connection back to
$\mathcal{D}_{AB}$ and define the relative coupling field

$$
\mathcal{C}_{AB}:=\phi_{A\to B}^{*}A^{(B)}-A^{(A)}.
$$

The **Inter-Agent Connection** is the chosen connection on this common comparison bundle:

$$
\mathcal{A}_{AB} := A^{(A)}\otimes\mathbb{1}_B+\mathbb{1}_A\otimes\phi_{A\to B}^{*}A^{(B)}
  +\lambda_{\text{lock}}\mathcal{C}_{AB}

$$

where:
- $\mathbb{1}_A, \mathbb{1}_B$ are identity operators on the respective bundles
- $\mathcal{C}_{AB}$ is the declared Lie-algebra-valued coupling field on $\mathcal{D}_{AB}$
- $\lambda_{\text{lock}} \geq 0$ is the **Locking Strength**

The comparison map and the transformation law for $\mathcal{C}_{AB}$ are part of the model. Both connections must
first be expressed on the same bundle before they can be compared. We use $g_{\text{lock}}$ below only for the gauge
coupling and reserve $\lambda_4$ for the quartic coefficient in the optional Landau model. The coefficient
$\lambda_{\text{lock}}$ weights the declared coupling field, while $\beta$ weights $\Psi_{\text{sync}}$ in a learning
objective; they are distinct parameters unless a calibration explicitly identifies them. Under a common gauge action
we require $\mathcal{C}_{AB}\mapsto U\mathcal{C}_{AB}U^{-1}$, so the curvature energy is gauge invariant.

*Interpretation:* The first two terms represent independent gauge evolution. The third term, proportional to $\lambda_{\text{lock}}$, couples the agents' internal gauges via communication.

:::

:::{prf:definition} The Locking Curvature
:label: def-locking-curvature

The **Locking Curvature** tensor measuring gauge mismatch between agents is:

$$
\mathcal{F}_{AB}:=d\mathcal{A}_{AB}-ig_{\text{lock}}\,\mathcal{A}_{AB}\wedge\mathcal{A}_{AB}

$$

where $g_{\text{lock}}$ is the inter-agent coupling constant. The **Integrated Friction** (gauge-invariant scalar) is:

$$
\Psi_{\text{sync}} := \int_{\mathcal{D}_{AB}} \operatorname{tr}\!\left(\mathcal{F}_{AB}\wedge *_{{G_{AB}}}\mathcal{F}_{AB}\right)

$$

*Interpretation:* When $\mathcal{F}_{AB}=0$ on a simply connected comparison region, parallel transport is path-independent
up to the stated regularity and boundary conditions. This is a statement about gauge transport; it is not a statement
about $\Phi_{AB}$. The symbol $G_{AB}$ in the Hodge star denotes the declared comparison metric on
$\mathcal{D}_{AB}$; it is not assumed to equal either private metric before a separate identification is made.

:::

:::{prf:definition} Euclidean Gauge-Curvature Energy
:label: thm-locking-operator-derivation

For the fixed comparison domain and measure above, define the Locking Operator by the positive Euclidean energy:

$$
\mathfrak{L}_{\text{sync}} := \frac{1}{4g_{\text{lock}}^2}\,\Psi_{\text{sync}}\geq 0.

$$

This definition is a gauge-curvature energy. It controls the selected connection only. Metric alignment, if desired,
must be added separately through a term such as
$\int_{\mathcal{D}_{AB}}\Phi_{AB}\,d\mu_{AB}$ and proved under a learning or gradient-flow hypothesis. No universal
Gromov--Hausdorff bound is asserted here.

:::

:::{prf:axiom} Finite Communication Bandwidth
:label: ax-finite-communication-bandwidth

The communication channel $\mathcal{L}$ has a declared finite capacity $C_{\mathcal{L}}$ in nats per update. This is
an assumption about the input alphabet, noise, and coding protocol. The static area budget
$I_{\max}=\nu_D\operatorname{Area}(\partial\mathcal{Z})/\ell_L^{D-1}$ from
{ref}`sec-causal-information-bound` is a separate total-information quantity and is not identified with
$C_{\mathcal{L}}$ without an explicit conversion.

$$
0<C_{\mathcal{L}}<\infty.

$$

*Justification:* A rate requires a channel model. The agent boundary may supply a separate upper bound only after the
update interval and the source/decoder convention have been specified.

:::

:::{prf:definition} The Gauge Alignment Order Parameter
:label: def-gauge-alignment-order-parameter

Choose a finite-dimensional unitary representation $\rho:G_{\text{Fragile}}\to U(N_\rho)$. The **Gauge Alignment
Order Parameter** measuring the relative orientation of agents' internal gauges is:

$$
\phi_{AB}(z) := \frac{1}{N_\rho}\operatorname{Tr}\!\left(\rho(U_A(z))\rho(U_B(z))^\dagger\right) \in \mathbb{C},

$$

where $U_A, U_B \in G_{\text{Fragile}}$ are the local gauge transformations, so $|\phi_{AB}|\leq 1$. The
**optional Landau potential** governing a scalar approximation is:

$$
\mathcal{V}_{\text{lock}}(\phi_{AB}) = -\mu_{\text{lock}}^2 |\phi_{AB}|^2 + \lambda_4 |\phi_{AB}|^4

$$

where:
- $\mu_{\text{lock}}^2 = \beta - \beta_c$ is the effective mass parameter
- $\beta$ is the interaction coupling strength
- $\beta_c$ is the critical coupling
- $\lambda_4 > 0$ is the quartic self-interaction coefficient (stabilization term). The quartic truncation is used only
  while its minimizer lies in the representation bound $|\phi_{AB}|\leq 1$.

When $\mu_{\text{lock}}^2>0$ and the unconstrained minimum is within that bound, the scalar model has
$|\phi_{AB}|=\sqrt{\mu_{\text{lock}}^2/(2\lambda_4)}$. This is a stationary point of the optional Landau model,
not a result of the strong-coupling proposition.

:::

:::{prf:proposition} Conditional Strong-Coupling Gauge Locking
:label: thm-spontaneous-gauge-locking

Fix the comparison domain $\mathcal{D}_{AB}$, the relative connection of
{prf:ref}`def-inter-agent-connection`, and the positive energy
$\Psi_{\text{sync}}$ of {prf:ref}`def-locking-curvature`. Suppose that the prediction term is bounded below,
that a feasible configuration with finite $\Psi_{\text{sync}}=0$ exists, and that the optimization actually reaches
(or approaches) minimizers of

$$
\mathcal{L}_{\beta}=\epsilon^{(A)}+\epsilon^{(B)}+\beta\Psi_{\text{sync}}.
$$

Then every sequence of minimizers with $\beta\to\infty$ has
$\Psi_{\text{sync}}\to0$. On a simply connected comparison region, the limiting relative connection is gauge-trivial,
so the pulled-back connections are gauge-equivalent. This conclusion concerns gauge transport. It does not imply
$\Phi_{AB}\to0$, a common metric, or a finite critical coupling.

:::

:::{prf:remark} Critical Coupling as a Model-Dependent Scale
:label: cor-critical-coupling-locking

No universal critical coupling follows from the preceding proposition. If a separate fluctuation model supplies a
kinetic term and an effective Landau expansion, one may define a model-dependent scale $\beta_c$; its formula must be
derived with the units of that model. The symbol $\beta_c$ in the Landau potential is therefore a fitted or derived
parameter, not a universal expression in $\sigma$, volume, and $g_{\text{lock}}$.

:::

:::{prf:definition} Message as Lie Algebra Element
:label: def-message-lie-algebra

A **Message** $m_{A \to B}$ from Agent $A$ to Agent $B$ is an element of the Lie algebra $\mathfrak{g}$ of the gauge group:

$$
m_{A \to B} \in \mathfrak{g} = \text{Lie}(G_{\text{Fragile}}), \quad m = m^a T_a

$$

where $\{T_a\}$ are the generators satisfying $[T_a, T_b] = i f^{abc} T_c$.

*Interpretation:* A message is an **instruction** to apply an infinitesimal gauge transformation. The symbol sequence encodes the coefficients $m^a$. "Understanding" a message means successfully applying $e^{im}$ to one's internal manifold.

:::

:::{prf:definition} The Language Channel
:label: def-language-channel

The **Language Channel** $\mathcal{L}$ is a low-bandwidth projection of the full gauge algebra:

$$
\mathcal{L}: \mathfrak{g} \to \mathfrak{g}_{\mathcal{L}} \subset \mathfrak{g}

$$

where $\dim(\mathfrak{g}_{\mathcal{L}}) \ll \dim(\mathfrak{g})$. The channel satisfies the bandwidth constraint of Axiom {prf:ref}`ax-finite-communication-bandwidth`.

*Interpretation:* Language cannot transmit the full metric tensor. It projects onto a finite-dimensional subspace—the "expressible" portion of experience.

:::

:::{prf:definition} Gauge-Covariant Translation Operator
:label: def-translation-operator

The **Translation Operator** $\mathcal{T}_{A \to B}(m)$ induced by message $m$ along a path $\gamma$ in the graph of
$\phi_{A\to B}$ is:

$$
\mathcal{T}_{A \to B}(m) := \rho(e^{im})\,W_\gamma,
\qquad
W_\gamma:=\mathcal{P}\exp\left(-ig_{\text{lock}}\int_\gamma \mathcal{A}_{AB}\right)

$$

where:
- The first factor encodes the **message content** in the chosen representation $\rho$
- The second factor is the **Wilson line** of the relative connection (parallel transport)
- $\mathcal{P}$ denotes path-ordering

*Properties:*
1. **Gauge Covariance:** The Wilson line transforms with the endpoint gauges, and the full operator has the corresponding
   conjugation law in the chosen representation.
2. **Composition:** Wilson lines compose under concatenation of paths; message factors compose only when their group
   actions are composed in the same representation.
3. **Identity at Locking:** If the relative connection vanishes along $\gamma$, then $W_\gamma=\mathbb{1}$ and the
   operator reduces to $\rho(e^{im})$.

:::

:::{prf:definition} Semantic Alignment
:label: def-semantic-alignment

**Understanding** occurs when the message reduces metric friction:

$$
\text{Understanding}(m) \;\Longrightarrow\; \Phi_{AB}(z; t+\Delta t) < \Phi_{AB}(z; t)

$$

after Agent $B$ receives and processes message $m$.

*Interpretation:* Under this operational test, a message is useful when it decreases the selected metric-distortion
proxy. The implication is conditional on the correspondence, task, and update rule; it is not a universal definition of
meaning.

:::

:::{prf:proposition} Conditional Holonomy Bound
:label: thm-untranslatability-bound

For a closed loop $\gamma=\partial\Sigma$ in the comparison domain, define the holonomy-induced message error by

$$
\mathcal{U}_{AB}(m):=\bigl\|\mathcal{H}_\gamma m\mathcal{H}_\gamma^{-1}-m\bigr\|.
$$

In a fixed matrix norm and in the small-curvature regime, this error is bounded by

$$
\mathcal{U}_{AB}(m) \leq 2g_{\text{lock}}\|m\|\int_{\Sigma}\|\mathcal{F}_{AB}\|\,dS + O(\|\mathcal{F}_{AB}\|^2).

$$

where $\Sigma$ is any surface bounded by the communication path.

:::

:::{prf:remark} Perfect Translation and Flatness
:label: cor-perfect-translation

Flatness is sufficient for path-independent transport on a simply connected domain, so it makes the holonomy error
vanish for the chosen loops. The converse requires a family of loops that detects all curvature components and is not
asserted without those hypotheses.

*Interpretation:* This concerns the gauge connection and does not imply metric alignment $\Phi_{AB}=0$.

:::

:::{prf:proposition} Rate--Distortion Babel Limit
:label: thm-babel-limit

Let $\Delta U$ denote the relative gauge variable and let
$R_{\Delta U}(\varepsilon)$ be the minimum rate (nats per update) needed to reproduce it with distortion at most
$\varepsilon$ under a declared source distribution and distortion measure. For a channel with capacity
$C_{\mathcal{L}}$ nats per update, $\varepsilon$-locking is achievable only if

$$
R_{\Delta U}(\varepsilon)\leq C_{\mathcal{L}}.
$$

If $R_{\Delta U}(\varepsilon)>C_{\mathcal{L}}$, no code for that source and distortion criterion can attain the target
fidelity. The static area budget $I_{\max}$ from {ref}`sec-causal-information-bound` may constrain a stored total,
but it becomes a channel rate only after an update interval and coding convention are supplied.

:::

:::{prf:remark} Untransmitted Components under a Rate--Distortion Model
:label: cor-ineffability-theorem

When $R_{\Delta U}(\varepsilon)>C_{\mathcal{L}}$, the chosen source and distortion model has a non-zero residual at
that target fidelity. One may call the unreproduced component a private or ineffable component, but no canonical
subspace or dimension follows from the capacity inequality alone. A dimension count requires a specified source model,
noise level, and allocation rule (for example, reverse water-filling for a Gaussian source).
:::

:::{prf:definition} Metric Eigendecomposition
:label: def-metric-eigendecomposition

Decompose the metric tensor into its principal components:

$$
G_A = \sum_{k=1}^{D} \gamma_k^{(A)} v_k^{(A)} \otimes v_k^{(A)}

$$

where $\gamma_1 \geq \gamma_2 \geq \cdots \geq \gamma_D > 0$ are metric eigenvalues (coordinate-dependent scale
factors) and $v_k^{(A)}$ are eigenvectors. They are not principal curvatures.

- **Core Concepts:** Components with $\gamma_k > \gamma_{\text{thresh}}$ (high selected scale)
- **Nuance:** Components with $\gamma_k \leq \gamma_{\text{thresh}}$ (low selected scale)

:::

:::{prf:remark} Conditional Spectral Allocation Diagnostic
:label: thm-spectral-locking-order

Under a source, noise, and coding model whose optimal allocation orders these modes by decreasing significance, a
diagnostic locked subspace after time $T$ may be defined by the $k_{\max}$ leading components satisfying:

$$
k_{\max} = \max\left\{k : \sum_{j=1}^k R_j(\varepsilon_j) \leq C_{\mathcal{L}} \cdot T\right\},

$$

Here $R_j(\varepsilon_j)$ is the declared per-mode rate--distortion cost. The ordering is a modelling assumption or a
result of that coding problem; metric eigenvalues alone do not prove it.

*Interpretation:* If the stated allocation model ranks modes in this way, high-ranked modes are transmitted first. The
labels "Gravity" and "Politics" are examples, not consequences of the spectrum alone.

:::

:::{prf:proposition} Conditional Quotient for a Shared Metric
:label: thm-emergence-objective-reality

Assume that $\mathcal{Z}_A$ and $\mathcal{Z}_B$ have the same dimension and that the selected comparison map
$\phi_{A\to B}$ is a diffeomorphism satisfying $\Phi_{AB}=0$ on the region of interest. Then the equivalence relation

$$
z_A\sim z_B \quad\Longleftrightarrow\quad z_B=\phi_{A\to B}(z_A)
$$

identifies the two copies, and the quotient carries the metric induced by $G_A$ (equivalently by
$\phi_{A\to B}^{*}G_B$). This construction is an operational shared representation. Gauge-curvature flatness alone does
not supply the diffeomorphism or the metric isometry.

:::

:::{prf:remark} Echo Chamber Effect (Metric Drift)
:label: rem-echo-chamber-effect

If agents $A$ and $B$ minimize inter-agent metric distortion $\Phi_{AB}$ but ignore an external grounding score, they
can spiral into a shared hallucination (folie à deux). Because the environment is a POMDP and does not carry a metric
tensor in this framework, define the operational grounding error
$E_{iE}(t):=\mathbb{E}[\ell_i(\hat{x}_{t+1}^{,i},x_{t+1})]$ on held-out environment transitions.

The corrected loss function must include grounding:

$$
\mathcal{L}_{\text{total}} = \lambda_{\text{lock}} \int_{\mathcal{D}_{AB}}\Phi_{AB}\,d\mu_{AB}
  + \lambda_{\text{ground}}(E_{AE}+E_{BE})

$$

where $E_{iE}$ is evaluated on a declared held-out transition and intervention set. It is a prediction/grounding
diagnostic, not a metric-friction tensor.

*Diagnostic:* The Babel check monitors $\partial_t E_{AE}$ and $\partial_t E_{BE}$ together with
$\partial_t\Phi_{AB}$. Rising grounding error while $\Phi_{AB}$ decreases is evidence of possible echo-chamber drift.

:::

:::{prf:remark} Population Thresholds Require a Model
:label: cor-critical-mass-consensus

The two-agent argument supplies no universal critical population $N_c$. A threshold can be defined only after an
interaction graph, noise model, and dimensionless population dynamics have been specified. The average pairwise
metric distortion $\langle\Phi_{ij}\rangle$ may be an input to such a model, but it does not determine the threshold
by itself.

:::

:::{prf:definition} The Institutional Manifold
:label: def-institutional-manifold

The **Institutional Manifold** $\mathcal{Z}_{\text{Inst}}$ is a **Static Reference Manifold** encoding shared conventions (Laws, Dictionaries, Money). Agents lock to the Institution rather than each other:

$$
\Phi_{A,\text{Inst}} + \Phi_{B,\text{Inst}} \quad \text{replaces} \quad \Phi_{AB}

$$

*Scaling:* Institution-mediated locking is $O(N)$ instead of $O(N^2)$.

:::

:::{prf:remark} Money as Universal Metric
:label: rem-money-universal-metric

**Money** is a **Universal Metric** in the institutional sense. It quantifies the "cost distance" between any two states:

$$
d_{\text{money}}(z_1, z_2) = \inf_{\gamma: z_1 \to z_2} \int_\gamma \text{Price}(\dot{z}) \, dt

$$

This provides a normalized gauge that allows agents with disjoint utility functions to coordinate.

*Interpretation:* Money emerges as the eigenmode of the institutional metric with highest consensus (largest eigenvalue in the shared subspace).

:::

## 07_cognition/09_retrieval_attention.md

:::{prf:proposition} Failure Modes of Standard Self-Attention for Memory
:label: prop-failure-modes-standard-self-attention

A self-attention mechanism $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$ applied to memory sequences fails to preserve:

1. **Causality**: Attention weight $\alpha_{t,t'} > 0$ for $t' > t$ (future influences past).

2. **Metric structure**: The inner product $Q^T K$ treats $\mathcal{Z}$ as flat Euclidean. The capacity-constrained metric $G(z)$ (Theorem {prf:ref}`thm-capacity-constrained-metric-law`) implies position-dependent distances.

3. **Gauge covariance**: Under local gauge transformation $\psi \to U(z)\psi$, attention scores change. Physical predictions should be gauge-invariant.

4. **Finite information speed**: Even with causal mask $t' < t$, there is no constraint on spatial separation. Events outside the light cone can influence attention.

5. **Geodesic correction**: No Christoffel symbol encoding. Parallel transport along worldlines is not accounted for.

*Consequence*: Standard self-attention requires extensive regularization and does not guarantee physical consistency even then.

:::

:::{prf:definition} Memory Potential (Recap)
:label: def-memory-potential-recap

From Definition {prf:ref}`def-memory-potential`, the **memory potential** is:

$$
\Psi_{\text{mem}}(z) = -\int_0^T \alpha(t') H_\tau(z, \gamma(t')) \, dt'
$$

where:
- $\gamma: [0, T] \to \mathcal{Z}$ is the agent's trajectory
- $\alpha(t')$ is the reward flux at time $t'$ (Definition {prf:ref}`def-the-reward-flux`)
- $H_\tau(z, z')$ is the heat kernel on $(\mathcal{Z}, G)$ (Definition {prf:ref}`def-memory-kernel-via-heat-equation`)

*Units:* $[\Psi_{\text{mem}}] = \text{nat}$.

*Cross-reference:* This definition does not incorporate causal structure; the integration runs over *all* past times without regard to light-cone constraints.

:::

:::{prf:definition} Causal Information Potential (Recap)
:label: def-causal-information-potential-recap

From Definition {prf:ref}`def-causal-information-potential`, the **Causal Information Potential** is:

$$
\Psi_{\text{causal}}(z, a) := \mathbb{E}_{z' \sim \bar{P}(\cdot | z, a)} \left[ D_{\text{KL}} \left( p(\theta_W | z, a, z') \| p(\theta_W | z, a) \right) \right]
$$

*Units:* $[\Psi_{\text{causal}}] = \text{nat}$.

*Interpretation:* Measures where the world model is most uncertain and experiments would be most informative.

:::

:::{prf:definition} Information Speed
:label: def-information-speed-recap

The **information speed** $c_{\text{info}}$ is the maximum rate at which influence can propagate through the latent manifold:

$$
c_{\text{info}} := \sup_{z, z', t} \left\{ \frac{d_G(z, z')}{|t - t'|} : (z', t') \text{ can causally influence } (z, t) \right\}
$$

*Units:* $[c_{\text{info}}] = [z]/[t]$ (geodesic distance per interaction time).

*Safety constraint:* The causal buffer condition requires $c_{\text{info}} < c_{\text{buffer}}$ for agent safety, ensuring the agent can respond to environmental changes before they propagate across its entire state space.

:::

:::{prf:definition} Lorentzian Memory Manifold
:label: def-lorentzian-memory-manifold

The **Lorentzian memory manifold** is the product $\mathcal{M} = \mathbb{R} \times \mathcal{Z}$ equipped with the metric:

$$
g_{\mu\nu}(z, t) = \begin{pmatrix} -c_{\text{info}}^2 \lambda(z)^2 & 0 \\ 0 & G_{ij}(z) \end{pmatrix}
$$

where:
- $(t, z) \in \mathbb{R} \times \mathcal{Z}$ are spacetime coordinates
- $c_{\text{info}}$ is the information speed (Definition {prf:ref}`def-information-speed-recap`)
- $\lambda(z) = 2/(1-|z|^2)$ is the conformal factor (Poincaré disk)
- $G_{ij}(z) = \lambda(z)^2 \delta_{ij}$ is the spatial metric (Definition {prf:ref}`def-poincare-metric-recap`)

The signature is $(-,+,+,\ldots,+)$ with the time component negative.

*Conformal structure:* The metric is conformally flat: $g_{\mu\nu} = \lambda^2 \eta_{\mu\nu}$ where $\eta_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, 1, \ldots, 1)$. This preserves the causal structure of flat spacetime in *coordinate* terms.

*Units:* $[g_{00}] = [z]^2/[t]^2 \cdot [z]^{-2} = [t]^{-2}$ (after absorbing $c_{\text{info}}$ units), $[g_{ij}] = [z]^{-2}$.

:::

:::{prf:definition} Effective Spacetime Interval
:label: def-spacetime-interval

The **effective spacetime interval** for memory causality between events $(z, t)$ and $(z', t')$ is:

$$
\Delta s^2_{\text{eff}} = -c_{\text{info}}^2 (t - t')^2 + d_G(z, z')^2
$$

where $d_G(z, z')$ is the geodesic distance on $(\mathcal{Z}, G)$ and $c_{\text{info}}$ is the information speed measured in *geodesic distance per unit time*.

*Mathematical remark:* This is an **effective** interval for defining causal structure, distinct from the metric-induced proper interval. The metric $g_{\mu\nu}$ in Definition {prf:ref}`def-lorentzian-memory-manifold` is conformally flat, so its null geodesics have constant *coordinate* speed $c_{\text{info}}$. However, for cognitive systems, the operationally meaningful constraint is that information travels at most $c_{\text{info}}$ in *proper (geodesic) distance* per unit time. This leads to the effective interval above.

*Equivalently:* The effective interval can be understood as the proper interval in a hypothetical metric $\tilde{g}_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, 1, \ldots, 1)$ where spatial distances are measured in geodesic coordinates.

**Classification:**
- **Timelike** ($\Delta s^2_{\text{eff}} < 0$): $|t - t'| > d_G(z, z') / c_{\text{info}}$ — causal connection possible
- **Spacelike** ($\Delta s^2_{\text{eff}} > 0$): $|t - t'| < d_G(z, z') / c_{\text{info}}$ — no causal connection
- **Lightlike** ($\Delta s^2_{\text{eff}} = 0$): $|t - t'| = d_G(z, z') / c_{\text{info}}$ — boundary of light cone

:::

:::{prf:definition} Causal Past Light Cone
:label: def-causal-past-light-cone

The **causal past** of event $(z, t)$ is the set:

$$
J^-(z, t) := \left\{ (z', t') \in \mathcal{M} : t' < t \text{ and } \Delta s^2_{\text{eff}}(z, t; z', t') \leq 0 \right\}
$$

Equivalently, using the spacetime interval from Definition {prf:ref}`def-spacetime-interval`:

$$
J^-(z, t) = \left\{ (z', t') : t' < t \text{ and } d_G(z, z') \leq c_{\text{info}} (t - t') \right\}
$$

*Interpretation:* $J^-(z, t)$ contains all events from which information, traveling at most at speed $c_{\text{info}}$ (measured in geodesic distance per unit time), could have reached $(z, t)$.

*Boundary:* The past light cone $\partial J^-(z, t)$ consists of null geodesics emanating backward in time from $(z, t)$, where $d_G(z, z') = c_{\text{info}} (t - t')$.

:::

:::{prf:theorem} Memory Causality Constraint
:label: thm-memory-causality-constraint

Let $\alpha(z, t; z', t')$ be the attention weight from Query at $(z, t)$ to Key at $(z', t')$. Causality requires:

$$
\alpha(z, t; z', t') = 0 \quad \text{if } (z', t') \notin J^-(z, t)
$$

:::

:::{prf:definition} Covariant Self-Attention with Causal Mask
:label: def-covariant-self-attention-causal

The **Covariant Self-Attention** mechanism for memory is:

$$
\text{SelfAttn}(z, t) = \sum_{t'=1}^{T} \alpha(z, t; z_{t'}, t') \cdot V(z_{t'}, t')
$$

where the attention weight is:

$$
\alpha(z, t; z', t') = M_{\text{causal}}(z, t; z', t') \cdot \text{softmax}_{(z', t') \in J^-(z,t)}\left( \frac{Q(z, t)^T K(z', t')}{\tau(z, t)} \right)
$$

Components:
- **Query**: $Q(z, t) = \Pi_Q \cdot U_{0 \to (z,t)} \cdot D_\mu \psi_{\text{mem}}(z, t)$
- **Key**: $K(z', t') = \Pi_K \cdot U_{0 \to (z',t')} \cdot D_\nu \psi_{\text{mem}}(z', t')$
- **Value**: $V(z', t') = \Pi_V \cdot U_{0 \to (z',t')} \cdot \psi_{\text{mem}}(z', t')$
- **Temperature**: $\tau(z) = \sqrt{d_k} / \lambda(z)$ (metric-encoded)
- **Causal mask**: $M_{\text{causal}}(z, t; z', t') = \mathbf{1}[(z', t') \in J^-(z, t)]$

Here $U_{0 \to (z,t)}$ is the Wilson line from origin to $(z, t)$ along a causal geodesic, $D_\mu$ is the covariant derivative (Definition {prf:ref}`def-covariant-derivative-recap`), and $\Pi_Q, \Pi_K, \Pi_V$ are learnable projections.

*Units:* $[\alpha] = \text{dimensionless}$, $[\text{SelfAttn}] = [\psi]$.

:::

:::{prf:theorem} Gauge Invariance of Causal Self-Attention
:label: thm-gauge-invariance-causal-self-attention

The attention weight $\alpha(z, t; z', t')$ in Definition {prf:ref}`def-covariant-self-attention-causal` is invariant under local gauge transformations $\psi(x) \to \Omega(x)\psi(x)$.

:::

:::{prf:definition} Temporal Christoffel Encoding
:label: def-temporal-christoffel-encoding

The **Temporal Geodesic Query** extends Definition {prf:ref}`def-geodesic-query-projection` to include temporal terms:

$$
Q_{\text{geo}}(x, z, t, v) = W_Q x + W_{Qz} z + W_{Qt} t + W_{Qv} v_{\text{feat}} + W_{Q,\Gamma}(z, z) + W_{Q,t}(t, t) + W_{Q,zt}(z, t)
$$

where:
- $W_Q \in \mathbb{R}^{d_k \times d_{\text{model}}}$: feature projection
- $W_{Qz} \in \mathbb{R}^{d_k \times d}$: spatial position
- $W_{Qt} \in \mathbb{R}^{d_k \times 1}$: temporal position
- $W_{Q,\Gamma} \in \mathbb{R}^{d_k \times d \times d}$: spatial Christoffel encoding
- $W_{Q,t} \in \mathbb{R}^{d_k \times 1 \times 1}$: temporal Christoffel encoding
- $W_{Q,zt} \in \mathbb{R}^{d_k \times d \times 1}$: mixed spacetime Christoffel encoding

**Lorentzian Christoffel structure:** For the metric $g_{\mu\nu} = \text{diag}(-c^2\lambda^2, \lambda^2 I_d)$ with $\lambda(z) = 2/(1-|z|^2)$, the non-zero Christoffel symbols are:

- **Spatial** ($\Gamma^k_{ij}$): $\Gamma^k_{ij} = \frac{2}{1-|z|^2}(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k)$ (as in Proposition {prf:ref}`prop-christoffel-encoding-poincare`)
- **Time-time-space** ($\Gamma^0_{0j}$): $\Gamma^0_{0j} = \frac{\partial_j \lambda}{\lambda} = \frac{2z_j}{1-|z|^2}$ (gradient of log conformal factor)
- **Space-time-time** ($\Gamma^k_{00}$): $\Gamma^k_{00} = \frac{c^2}{\lambda} \partial_k \lambda = \frac{2c^2 z_k}{1-|z|^2}$ (acceleration term)

*Note:* $\Gamma^k_{0j} = 0$ and $\Gamma^0_{ij} = 0$ for this diagonal metric (no off-diagonal time-space mixing).

:::

:::{prf:proposition} Causal Wilson Line Along Worldline
:label: prop-causal-wilson-line

The Wilson line in Definition {prf:ref}`def-covariant-self-attention-causal` is computed along the **causal geodesic** connecting $(z', t')$ to $(z, t)$, not an arbitrary path.

For events in the causal past with small separation, the linearized Wilson line is:

$$
U_{(z',t') \to (z,t)} \approx I - i A_\mu(\bar{z}, \bar{t}) \Delta x^\mu
$$

where:
- $\Delta x^\mu = (t - t', z - z')$ is the spacetime displacement
- $\bar{z}, \bar{t}$ is a reference point (midpoint or initial)
- $A_\mu$ is the total gauge connection

For events outside the light cone, no causal geodesic exists, so the Wilson line is undefined. This is consistent with the causal mask zeroing out such contributions.

:::

:::{prf:definition} Lorentzian Cross-Attention
:label: def-lorentzian-cross-attention

The **Lorentzian Cross-Attention** for retrieval from an external archive $\mathcal{E} = \{(z'_i, t'_i, v_i)\}_{i=1}^N$ is:

$$
\text{CrossAttn}(z, t) = \sum_{i=1}^{N} \alpha_{\text{ret}}(z, t; z'_i, t'_i) \cdot V(z'_i, t'_i)
$$

where the **retarded attention weight** is:

$$
\alpha_{\text{ret}}(z, t; z', t') = \alpha_{\text{bare}}(z, t; z', t') \cdot \Theta_{\text{ret}}(z, t; z', t')
$$

with:
- **Bare weight**: $\alpha_{\text{bare}} = \text{softmax}\left( Q(z, t)^T K(z', t') / \tau(z) \right)$ (gauge-covariant)
- **Retarded factor**: $\Theta_{\text{ret}}(z, t; z', t') = \theta\left( t - t' - \frac{d_G(z, z')}{c_{\text{info}}} \right)$

Here $\theta$ is the Heaviside step function: $\theta(x) = 1$ if $x \geq 0$, else $0$.

*Interpretation:* The retarded factor enforces that $(z', t')$ must be in the causal past of $(z, t)$, with information having had time to propagate the geodesic distance at speed $c_{\text{info}}$.

*Units:* $[\alpha_{\text{ret}}] = \text{dimensionless}$.

:::

:::{prf:definition} Ghost Memory Interface
:label: def-ghost-memory-interface

The **Ghost Memory** at archive position $i$, as seen from Query position $(z, t)$, is the retarded image:

$$
\xi_i^{\text{ghost}}(z, t) = \xi_i(t - t_{\text{ret},i})
$$

where the **retardation time** is:

$$
t_{\text{ret},i} = \frac{d_G(z, z'_i)}{c_{\text{info}}}
$$

*Cross-reference:* This extends the Ghost Interface of Definition {prf:ref}`def-ghost-interface` to memory retrieval. The archive item appears "frozen" at the time when information about it could have reached the Query position.

*Implementation:* For memory buffers where $t_{\text{ret}} \ll \Delta t$ (retardation much smaller than timestep), the ghost image is approximately instantaneous. For external knowledge bases or multi-agent settings, retardation may be significant.

:::

:::{prf:theorem} Lorentzian Cross-Attention Structure
:label: thm-lorentzian-cross-attention-structure

The Lorentzian Cross-Attention (Definition {prf:ref}`def-lorentzian-cross-attention`) satisfies:

1. **Causality**: $\alpha_{\text{ret}}(z, t; z', t') = 0$ for $(z', t') \notin J^-(z, t)$

2. **Gauge covariance**: The bare weight $\alpha_{\text{bare}}$ is gauge-invariant (Theorem {prf:ref}`thm-gauge-invariance-cross-attention`)

3. **Retarded propagator structure**: In the continuum limit, the attention kernel approaches the **retarded Green's function**:

$$
G_{\text{ret}}(z, t; z', t') \propto \delta\left( t - t' - \frac{d_G(z, z')}{c_{\text{info}}} \right) \cdot \theta(t - t')
$$

4. **Lorentz invariance** (in flat limit): Under Lorentz boosts, the attention structure is preserved (the light cone is Lorentz-invariant).

:::

:::{prf:definition} Causal Heat Kernel
:label: def-causal-heat-kernel

The **Causal Heat Kernel** on the Lorentzian memory manifold is:

$$
H_\tau^{\text{causal}}(z, t; z', t') := H_\tau(z, z') \cdot M_{\text{causal}}(z, t; z', t')
$$

where:
- $H_\tau(z, z')$ is the standard heat kernel on $(\mathcal{Z}, G)$ (Definition {prf:ref}`def-memory-kernel-via-heat-equation`)
- $M_{\text{causal}}(z, t; z', t') = \mathbf{1}[(z', t') \in J^-(z, t)]$ is the causal mask

*Interpretation:* The heat kernel provides spatial smoothing (how much influence a memory at $z'$ has on position $z$), while the causal mask restricts to events in the past light cone.

*Properties:*
- $H_\tau^{\text{causal}}(z, t; z', t') = 0$ if $t' \geq t$ (no future influence)
- $H_\tau^{\text{causal}}(z, t; z', t') = 0$ if $d_G(z, z') > c_{\text{info}}(t - t')$ (no superluminal influence)
- As $\tau \to 0$: $H_\tau^{\text{causal}} \to \delta(z - z') \cdot M_{\text{causal}}$

:::

:::{prf:theorem} Causal Memory Potential
:label: thm-causal-memory-potential

The **Causal Memory Potential** is:

$$
\Psi_{\text{mem}}^{\text{causal}}(z, t) := -\int_{J^-(z,t)} \alpha(t') H_\tau(z, \gamma(t')) \, d\mu_{J^-}(z', t')
$$

where the integration is over the causal past $J^-(z, t)$ with measure $d\mu_{J^-}$.

Equivalently, using the causal heat kernel:

$$
\Psi_{\text{mem}}^{\text{causal}}(z, t) = -\int_0^t \int_{\mathcal{Z}} \alpha(t') H_\tau^{\text{causal}}(z, t; z', t') \, d\mu_G(z') \, dt'
$$

*Comparison with Definition {prf:ref}`def-memory-potential-recap`:* The original memory potential integrates over all past times without regard to spatial separation. The causal memory potential restricts to events that could have causally influenced the present.

*Physical interpretation:* Only memories from within your light cone can pull you. Memories that are "too far away" (in spacetime) to have reached you yet have no influence.

:::

:::{prf:corollary} Memory Force with Causal Constraint
:label: cor-memory-force-causal

The memory-induced force (Lemma {prf:ref}`lem-virtual-work-of-recall`) becomes:

$$
\mathbf{f}_{\text{mem}}^{\text{causal}}(z, t) = -G^{-1}(z) \nabla_z \Psi_{\text{mem}}^{\text{causal}}(z, t)
$$

where the gradient is with respect to the spatial coordinates $z$, holding $t$ fixed.

*Properties:*
- The force is conservative (derived from a potential)
- The force respects causality (only causal past contributes)
- Near the light cone boundary, the force may have discontinuities as memories enter/exit the causal past

:::

:::{prf:definition} Causal BAOAB Steps for Memory
:label: def-causal-baoab-memory

The **Causal BAOAB** integrator for memory-augmented dynamics uses five steps as in Definition {prf:ref}`def-baoab-attention-heads`, with the memory potential modified to respect causality:

**Step 1 (B-step, first half-kick):**

$$
p \leftarrow p - \frac{h}{2} \nabla_z \left( \Phi_{\text{eff}}(z) + \Psi_{\text{mem}}^{\text{causal}}(z, t) \right)
$$

**Step 1.5 (Boris rotation, if $\mathcal{F} \neq 0$):** Apply the rotation from Definition {prf:ref}`def-baoab-splitting` using the Value Curl $\mathcal{F}$.

**Step 2 (A-step, first half-drift):**

$$
z \leftarrow z + \frac{h}{2} G^{-1}(z) p
$$

**Step 3 (O-step, thermostat):**

$$
p \leftarrow c_1 p + c_2 G^{1/2}(z) \xi, \quad \xi \sim \mathcal{N}(0, I)
$$
with $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1 - c_1^2) T_c}$.

**Step 4 (A-step, second half-drift):**

$$
z \leftarrow z + \frac{h}{2} G^{-1}(z) p
$$

**Step 5 (B-step, second half-kick):**

$$
p \leftarrow p - \frac{h}{2} \nabla_z \left( \Phi_{\text{eff}}(z) + \Psi_{\text{mem}}^{\text{causal}}(z, t) \right)
$$

**Step 5.5 (Boris rotation, if $\mathcal{F} \neq 0$):** Apply the same rotation as in Step 1.5.

**Time update:** $t \leftarrow t + h$

The gradient $\nabla_z \Psi_{\text{mem}}^{\text{causal}}$ is computed via Covariant Self-Attention with causal mask, as the weighted sum over memory Keys in $J^-(z, t)$.

:::

:::{prf:proposition} Conditional frozen-time Boltzmann tracking with causal memory
:label: thm-boltzmann-causal-memory

Fix a time $t$ and suppose that the dynamics are conservative
($\mathcal{F}=0$, $u_\pi=0$), $T_c$ is constant, the metric and boundary
conditions satisfy the hypotheses of the compatible BAOAB result, and the
frozen potential
$\Phi_t:=\Phi_{\mathrm{eff}}+\Psi_{\mathrm{mem}}^{\mathrm{causal}}(\cdot,t)$
is smooth with a normalizable Gibbs density.  Then the frozen-time Causal
BAOAB step has the formal target

$$
\rho_t(z,p) \propto \exp\left(-\frac{\Phi_t(z)}{T_c}
                              -\frac{\|p\|_G^2}{2T_c}\right).
$$
For a time-dependent memory potential, a slowly varying assumption can support
an adiabatic tracking estimate, with an error controlled by the chosen
$C^2$-variation bound and the mixing rate of the frozen chain.  It does not
give a stationary distribution for the full history-dependent process.

*Scope.* The compatible BAOAB result is the frozen, reversible calculation;
the causal mask itself does not preserve detailed balance.  Curl forcing,
policy forcing, state-dependent temperature, or an uncontrolled boundary
requires a separate estimate.

:::

## 08_multiagent/01_gauge_theory.md

:::{prf:definition} N-Agent Product Manifold
:label: def-n-agent-product-manifold

The global configuration space is the product manifold:

$$
\mathcal{Z}^{(N)} := \mathcal{Z}^{(1)} \times \mathcal{Z}^{(2)} \times \cdots \times \mathcal{Z}^{(N)}.

$$
The metric on $\mathcal{Z}^{(N)}$ is the direct sum of individual metrics:

$$
G^{(N)} := \bigoplus_{i=1}^N G^{(i)},

$$
where each $G^{(i)}$ is the capacity-constrained metric from Theorem {prf:ref}`thm-capacity-constrained-metric-law`. In coordinates, this is block-diagonal: if $\mathbf{z} = (z^{(1)}, \ldots, z^{(N)})$ with $z^{(i)} \in \mathbb{R}^{d_i}$, then $G^{(N)}_{\mu\nu}(\mathbf{z}) = G^{(i)}_{ab}(z^{(i)})$ when indices $\mu, \nu$ both lie in agent $i$'s block, and $G^{(N)}_{\mu\nu} = 0$ otherwise.

*Units:* $[G^{(N)}] = [z]^{-2}$.

*Remark (Isolated Agents).* The product metric $G^{(N)}$ describes agents in **isolation**—there is no cross-coupling between $\mathcal{Z}^{(i)}$ and $\mathcal{Z}^{(j)}$. Strategic coupling modifies this to $\tilde{G}^{(N)}$ via the Game Tensor ({ref}`sec-the-game-tensor-deriving-adversarial-geometry`).

:::

:::{prf:definition} Agent-Specific Boundary Interface
:label: def-agent-specific-boundary-interface

Each agent $i$ possesses its own symplectic boundary $(\partial\mathcal{Z}^{(i)}, \omega^{(i)})$ with:
- **Dirichlet component** (sensors): $\phi^{(i)}(x) = $ observation stream
- **Neumann component** (motors): $j^{(i)}_{\text{motor}}(x) = $ action flux
- **Reward component** (source): boundary reward flux $J_r^{(i)}$ (1-form); conservative case reduces to scalar charge
  density $\sigma_r^{(i)}$ (Definition {prf:ref}`def-the-reward-flux`)

The boundary conditions follow the structure of Definition {prf:ref}`def-dirichlet-boundary-condition-sensors`–23.1.3,
applied per-agent.

*Cross-reference:* {ref}`sec-the-symplectic-interface-position-momentum-duality` (Symplectic Boundary Manifold), Definition {prf:ref}`def-mass-tensor`.

:::

:::{prf:definition} Environment Distance
:label: def-environment-distance

Let $d_{\mathcal{E}}^{ij}$ denote the **environment distance** between agents $i$ and $j$—the geodesic length in the environment manifold $\mathcal{E}$ that information must traverse. This may differ from the latent distance $d_G(z^{(i)}, z^{(j)})$.

*Examples:*
- **Physical agents:** $d_{\mathcal{E}}^{ij} = $ spatial separation in meters
- **Networked agents:** $d_{\mathcal{E}}^{ij} = $ network hop distance or latency
- **Co-located agents:** $d_{\mathcal{E}}^{ij} = 0$ (shared boundary)

*Units:* $[d_{\mathcal{E}}^{ij}] = $ meters or equivalent environment-specific units.

:::

:::{prf:axiom} Information Speed Limit
:label: ax-information-speed-limit

There exists a maximum speed $c_{\text{info}} > 0$ at which information propagates through the environment $\mathcal{E}$. The **Causal Delay** between agents $i$ and $j$ is:

$$
\tau_{ij} := \frac{d_{\mathcal{E}}^{ij}}{c_{\text{info}}},

$$
where $d_{\mathcal{E}}^{ij}$ is the environment distance (Definition {prf:ref}`def-environment-distance`).

*Units:* $[c_{\text{info}}] = [\text{length}]/[\text{time}]$, $[\tau_{ij}] = [\text{time}]$.

*Examples:*
- **Physical systems:** $c_{\text{info}} = c \approx 3 \times 10^8$ m/s (speed of light)
- **Acoustic systems:** $c_{\text{info}} \approx 343$ m/s (speed of sound)
- **Networked systems:** $c_{\text{info}} \approx d/\text{latency}$ (effective propagation speed)
- **Co-located agents:** $c_{\text{info}} \to \infty$ effective limit when $d_{\mathcal{E}}^{ij} = 0$

:::

:::{prf:definition} Causal Interval
:label: def-causal-interval

The **Causal Interval** between spacetime events $(z^{(i)}, t_i)$ and $(z^{(j)}, t_j)$ is:

$$
\Delta s^2_{ij} := -c_{\text{info}}^2 (t_j - t_i)^2 + (d_{\mathcal{E}}^{ij})^2.

$$
The events are classified as:
- **Timelike** ($\Delta s^2_{ij} < 0$): $|t_j - t_i| > \tau_{ij}$. Causal influence is possible.
- **Spacelike** ($\Delta s^2_{ij} > 0$): $|t_j - t_i| < \tau_{ij}$. No causal influence is possible.
- **Lightlike** ($\Delta s^2_{ij} = 0$): $|t_j - t_i| = \tau_{ij}$. Boundary case.

*Consequence:* If agents $i$ and $j$ are spacelike separated at time $t$, no instantaneous Hamiltonian $H(z^{(i)}_t, z^{(j)}_t)$ can couple their states. Coupling must occur via retarded potentials.

:::

:::{prf:definition} Past Light Cone
:label: def-past-light-cone

The **Past Light Cone** of Agent $i$ at time $t$ is the set of all agent-time pairs that can causally influence Agent $i$:

$$
\mathcal{C}^-_i(t) := \left\{ (j, t') \in \{1,\ldots,N\} \times \mathbb{R} : t' \leq t - \tau_{ij} \right\}.

$$
The **Future Light Cone** is defined symmetrically:

$$
\mathcal{C}^+_i(t) := \left\{ (j, t') : t' \geq t + \tau_{ij} \right\}.

$$
*Physical interpretation:* Agent $i$ at time $t$ can only receive information from events in $\mathcal{C}^-_i(t)$ and can only influence events in $\mathcal{C}^+_i(t)$. The region outside both cones is causally disconnected.

:::

:::{prf:definition} Retarded Potential (Memory Screen)
:label: def-retarded-potential

Let $\rho^{(j)}_r(t, z)$ be the scalar source density associated with the conservative component of Agent $j$'s boundary
reward flux. The potential perceived by Agent $i$ at position $z$ and time $t$ is the **Retarded Potential**:

$$
\Psi_{\text{ret}}^{(i)}(t, z) = \sum_{j \neq i} \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z, t; \zeta, \tau) \rho^{(j)}_r(\tau, \zeta) \, d\mu_{G^{(j)}}(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ is the **Retarded Green's Function** for the wave operator on the manifold:

$$
G_{\text{ret}}(z, t; \zeta, \tau) \quad \text{solves} \quad \left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)G_{\text{ret}} = \delta(z-\zeta)\delta(t-\tau),

$$
with $G_{\text{ret}} = 0$ for $t < \tau$. In flat space and the massless limit ($\kappa = 0$), $G_{\text{ret}}$ reduces to a light-cone delta; for $\kappa > 0$ it develops an interior light-cone tail.

*Interpretation:* Agent $i$ does not perceive Agent $j$'s current state. It perceives the "ghost" of Agent $j$ from time $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ ago.

*Units:* $[\Psi_{\text{ret}}] = \text{nat}$, $[G_{\text{ret}}] = [\text{length}]^{2-D}[\text{time}]^{-1}$.

*Remark (Strategic coupling).* When strategic relationships matter, weight each source by $\alpha_{ij}$; equivalently replace
$\rho^{(j)}_r$ with $\rho^{\text{ret}}_{ij}$ from Definition {prf:ref}`def-retarded-interaction-potential`.

:::

:::{prf:definition} History state and received observations
:label: def-causal-bundle

Write $\mathsf H_t$ for the timestamped history of the modeled states,
actions, and signals through $t$, including pending emissions. The complete
history state is $(t,\mathsf H_t)$. A recipient's received history is its
restriction to events whose arrival time is at most $t$; it need not determine
the hidden complete history. The reward occupation screen
$\Xi_t=\int_0^t\alpha(s)\delta_{\gamma(s)}ds$ of
{prf:ref}`def-memory-screen` is a different, temporally compressed observable.
The notation $\mathcal Z_{\mathrm{causal}}$ denotes the space of admissible
timestamped histories with their current state, rather than a Cartesian
product with a particular realized measure. A finite received buffer is the
observation structure implemented below, not an asserted sufficient statistic.
:::

:::{prf:theorem} Markov representation by the complete history
:label: thm-markov-restoration

On the standard measurable path spaces of the specified process, the complete
history state $(t,\mathsf H_t)$ is Markov with its conditional extension kernel.
Neither a positive delay alone nor the reward occupation screen determines
whether a smaller state is Markov.

:::

:::{prf:corollary} Causal memory and control
:label: cor-memory-physical-necessity

The full history supplies a Markov representation by
{prf:ref}`thm-markov-restoration`. The actual recipient can use only its
received history and the beliefs computed from it. The occupation screen
retains its established role as a reward-weighted spatial measure. Neither
finite propagation nor that measure identifies a finite sufficient statistic.
This follows directly from the two histories in the preceding proof.
:::

:::{prf:definition} Ghost Interface
:label: def-ghost-interface

The **Ghost Interface** $\mathcal{G}_{ij}(t)$ between agents $i$ and $j$ at time $t$ is:

$$
\mathcal{G}_{ij}(t) := \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}),

$$
coupling Agent $i$'s current boundary to Agent $j$'s past boundary, where $\tau_{ij} = d_{\mathcal{E}}^{ij}/c_{\text{info}}$ is the causal delay.

The **Ghost Symplectic Structure** is:

$$
\omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.

$$

*Mechanism:* Agent $i$ couples not to $z^{(j)}_t$, but to the **Ghost State** $\hat{z}^{(j)}_t := z^{(j)}_{t-\tau_{ij}}$—the state of Agent $j$ when the signal was emitted.

*Units:* $[\tau_{ij}] = [\text{time}]$.

:::

:::{prf:proposition} Interaction Kernel
:label: prop-interaction-kernel

The **pairwise interaction potential** $\Phi_{\text{int}}: \mathcal{Z} \times \mathcal{Z} \to \mathbb{R}$ between agents at positions $z, \zeta$ is the screened Green's function weighted by influence:

$$
\Phi_{\text{int}}(z, \zeta) := \alpha \cdot \mathcal{G}_{\kappa}(z, \zeta)

$$
where $\mathcal{G}_{\kappa}$ is the screened Green's function (Proposition {prf:ref}`prop-green-s-function-interpretation`) and $\alpha$ encodes the strategic relationship.

*Properties:*
- $\Phi_{\text{int}}(z, \zeta) = \Phi_{\text{int}}(\zeta, z)$ (symmetric in cooperative settings)
- $\Phi_{\text{int}} \to 0$ as $d_G(z, \zeta) \to \infty$ (locality via screening)
- $\nabla^2_z \Phi_{\text{int}}$ defines the Game Tensor contribution (Definition {prf:ref}`def-the-game-tensor`)
:::

:::{prf:definition} Retarded Interaction Potential
:label: def-retarded-interaction-potential

The **Retarded Interaction Source Density** from Agent $j$ to Agent $i$ is:

$$
\rho^{\text{ret}}_{ij}(\zeta, \tau) := \alpha_{ij} \cdot \rho^{(j)}_r(\zeta, \tau),

$$
where:
- $\rho^{(j)}_r$ is the conservative reward source density for Agent $j$ derived from boundary reward flux
  (Definition {prf:ref}`def-the-reward-flux`)
- $\alpha_{ij} \in \{-1, 0, +1\}$ encodes the strategic relationship:
  - $\alpha_{ij} = +1$: Cooperative
  - $\alpha_{ij} = 0$: Independent
  - $\alpha_{ij} = -1$: Adversarial

We write $\rho^{\text{ret}}_{ij}$ on $\mathcal{Z}^{(j)}$ and pull it back to Agent $i$'s chart along the Ghost Interface;
for notational simplicity, we suppress the pullback in what follows.

The induced **Retarded Interaction Potential** is the retarded Green's function convolution:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(j)}} G_{\text{ret}}(z^{(i)}, t; \zeta, \tau)\,
\rho^{\text{ret}}_{ij}(\zeta, \tau)\, d\mu_{G^{(j)}}(\zeta)\, d\tau,

$$
where $G_{\text{ret}}$ is the retarded Green's function (Definition {prf:ref}`def-retarded-potential`).

*Remark (Point-source / ghost limit).* If Agent $j$'s conservative source is concentrated along a trajectory,
$\rho^{(j)}_r(\zeta, \tau) = \sigma^{(j)}_r(\tau)\,\delta(\zeta - z^{(j)}_\tau)$, then

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) = \alpha_{ij}\int_{-\infty}^{t} G_{\text{ret}}(z^{(i)}, t; z^{(j)}_\tau, \tau)\,
\sigma^{(j)}_r(\tau)\, d\tau,
$$
which reduces to evaluation at the retarded time in the massless flat-space limit. This recovers the ghost-state
interpretation.

*Remark (Quasi-static kernel).* In the low-frequency limit, $G_{\text{ret}}$ reduces to the screened static kernel
$\mathcal{G}_\kappa$ and the potential can be approximated by evaluating the instantaneous interaction at the ghost
state. This is a computational shortcut, not the first-principles definition.

*Remark (Non-conservative component).* Solenoidal reward components are not captured by the scalar source; they enter via
the curl field in the dynamics.

:::

:::{prf:proposition} Evaluation under a fixed delay kernel
:label: thm-strategic-delay-tensor

For a fixed delay $\tau_{ij}$, convolution with
$\delta(s-\tau_{ij})$ sends a continuous tensor trajectory $T$ to
$T(t-\tau_{ij})$: integrate $\delta(t-u-\tau_{ij})T(u)$ in $u$.
This defines the point-delay approximation. The retarded massive Green
operator in {prf:ref}`def-retarded-interaction-potential` generally has an
interior-cone tail and retains an integral over emission times. Its tensor
response is obtained by differentiating that integral where the defined
regularized kernel permits differentiation, rather than replacing the tail
by a delta distribution. $\square$
:::

:::{prf:corollary} Vanishing delay for continuous records
:label: cor-newtonian-limit-ghost

For fixed separation and a continuous recorded trajectory,
$z_j(t-d_{ij}/c)\to z_j(t)$ as $c\to\infty$ by continuity.
For a Lipschitz trajectory the error is at most
$\operatorname{Lip}(z_j)d_{ij}/c$. This establishes the point-delay
comparison. The retarded field integral retains its own initial, source,
and boundary data; its static operator comparison is
{prf:ref}`cor-helmholtz-limit`. $\square$
:::

:::{prf:theorem} Bellman generator and the separately defined wave action
:label: thm-hjb-klein-gordon

For the diffusion and discount already used in
{prf:ref}`thm-the-hjb-helmholtz-correspondence`, write
$\mathcal L=b\cdot\nabla+T_c\Delta_G$ and $\gamma_h=e^{-\lambda h}$.
The smooth Bellman equation has continuous-time form
$$
\partial_tV+\mathcal LV-\lambda V+r=0.
$$
In its stationary zero-drift sector,
$(-\Delta_G+\lambda/T_c)V=r/T_c$; denote this screening coefficient by
$\kappa_B^2=\lambda/T_c$. The scalar field action used in this chapter
instead defines the wave operator
$$
\Box_g=-|g|^{-1/2}\partial_\mu(|g|^{1/2}g^{\mu\nu}\partial_\nu),
\qquad(\Box_g+\kappa^2)V=\rho_r.
$$
The stationary operators coincide under the coefficient identification
$\kappa^2=\kappa_B^2$ and the same sources and boundary realization.

:::

:::{prf:corollary} Propagation and static screening
:label: cor-value-wavefront

The retarded Green operator of the stated wave model propagates inside its
causal cone. Static screening concerns its zero-frequency resolvent.
In a flat chart, inserting $e^{i(k\cdot x-\omega t)}$ into the homogeneous
equation gives $\omega^2=c^2(|k|^2+\kappa^2)$. Hence for real $k$ the
undamped amplitudes oscillate; a positive $\kappa$ does not supply temporal
friction or universal exponential attenuation of propagating waves.
At $\omega=0$ the spatial operator is $-\Delta+\kappa^2$, whose screened
kernel is the static interaction kernel already defined. $\square$
:::

:::{prf:corollary} Static operator and instantaneous comparison
:label: cor-helmholtz-limit

For the fixed product metric, $\partial_tV=0$ gives exactly
$(-\Delta_G+\kappa^2)V=\rho_r$. For a family with bounded $\partial_t^2V$,
the residual $c^{-2}\partial_t^2V$ is bounded by
$c^{-2}\|\partial_t^2V\|$ in the same norm. This is an operator residual
estimate, not an assertion of solution convergence for arbitrary initial data.
Holding $\lambda$ fixed in the different convention $\kappa=\lambda/c$
sends $\kappa$ to zero; it does not preserve a screened Helmholtz operator.
The Bellman comparison uses $\kappa^2=\lambda/T_c$ instead. $\square$
:::

:::{prf:proposition} Retarded Green's Function
:label: prop-retarded-greens-function

The solution to the inhomogeneous Klein-Gordon equation is given by convolution with the **Retarded Green's Function**:

$$
V^{(i)}(z, t) = \int_{-\infty}^{t} \int_{\mathcal{Z}^{(i)}} G_{\text{ret}}(z, t; \zeta, \tau) \left[ \rho^{(i)}_r(\zeta, \tau) + \sum_{j \neq i} \rho^{\text{ret}}_{ij}(\zeta, \tau) \right] d\mu_{G^{(i)}}(\zeta) \, d\tau,

$$
where $G_{\text{ret}}$ satisfies:

$$
\left( \frac{1}{c_{\text{info}}^2} \frac{\partial^2}{\partial t^2} - \Delta_G + \kappa^2 \right) G_{\text{ret}}(z, t; \zeta, \tau) = \delta(z - \zeta)\delta(t - \tau),

$$
with the **causal boundary condition** $G_{\text{ret}} = 0$ for $t < \tau$.

*Massless flat-space example (D = 3):* For $\mathcal{Z} = \mathbb{R}^3$ and $\kappa = 0$,

$$
G_{\text{ret}}(z, t; \zeta, \tau) = \frac{\Theta(t - \tau)}{4\pi |z - \zeta|} \delta\left(t - \tau - \frac{|z-\zeta|}{c_{\text{info}}}\right).

$$
For $\kappa > 0$, the retarded kernel acquires an interior light-cone tail with Bessel decay; we keep $G_{\text{ret}}$ abstract to avoid dimension-specific formulas.

:::

:::{prf:definition} Strategic Hessian and pullback
:label: def-the-game-tensor

Use the smooth local best-response branch and Strategic Jacobian already
specified in {prf:ref}`def-strategic-jacobian`. With the intrinsic connection
on agent $j$'s manifold define the covariant tensor
$$
H^{(i)}_{jj,mn}=\nabla^{(j)}_m\nabla^{(j)}_nV^{(i)},\qquad
\mathcal G^{(i)}_{ij,ab}=\mathcal J_{ji}^{m}{}_{a}
H^{(i)}_{jj,mn}\mathcal J_{ji}^{n}{}_{b}.
$$
No additional lowering of the Hessian indices is applied. The strategic
metric prescription is $\widetilde G^{(i)}=G^{(i)}+h^{(i)}$ with
$h^{(i)}=\sum_{j\ne i}\beta_{ij}\mathcal G^{(i)}_{ij}$.
Its positive-definite domain is checked using the spectral margin already
specified in {prf:ref}`def-e7-strategic-metric`.
The curvature equation {prf:ref}`thm-capacity-constrained-metric-law` remains
a separate differential identity; this algebraic prescription is not its solution.

For a $C^2$ response $y=b(x)$, direct differentiation gives
$$
\partial_{ab}V(x,b(x))=V_{ab}+V_{am}b^m_b+V_{bm}b^m_a
+V_{mn}b^m_ab^n_b+V_m\partial_{ab}b^m.
$$
Thus the pulled-back $H_{jj}$ is one contribution, not the full Hessian of
the composed value. The final term vanishes at a stationary point in $y$.
For the positive metric $\widetilde G=G+h$, subtraction of the two
metric-compatible torsion-free connections gives the exact identity
$$
\widetilde\Gamma^a_{bc}-\Gamma^a_{bc}
=\tfrac12\widetilde G^{ad}
(\nabla_bh_{dc}+\nabla_ch_{db}-\nabla_dh_{bc}).
$$
Replacing $\widetilde G^{-1}$ by $G^{-1}$ gives its first-order expansion,
with a remainder controlled by the inverse-metric identity
$\widetilde G^{-1}-G^{-1}=-G^{-1}h\widetilde G^{-1}$.
:::

:::{prf:theorem} Positive metric perturbations
:label: thm-adversarial-mass-inflation

For the strategic metric prescription, the exact difference is
$\xi^\top(\widetilde G-G)\xi=\sum_j\beta_{ij}
(\mathcal J_{ji}\xi)^\top H^{(i)}_{jj}(\mathcal J_{ji}\xi)$.
Every positive-semidefinite summand with nonnegative coefficient increases
the quadratic form; hence a sum of such contributions gives
$\widetilde G\succeq G$. A negative cooperative contribution must be included
in the same sum when testing its sign and positive-definiteness.
This follows by expanding the definition term by term. The strategic sign
label by itself does not determine the sign of the Hessian. $\square$
:::

:::{prf:definition} Retarded pullback tensor
:label: def-retarded-game-tensor

Evaluate the covariant Hessian and the specified Strategic Jacobian at the
recorded point-delay data in {prf:ref}`def-the-game-tensor`, obtaining
$h(t)=\sum_j\beta_{ij}\mathcal J_{ji}(t)^*H^{(i)}_{jj}(t)\mathcal J_{ji}(t)$.
Here the star denotes the covector pullback, represented by transpose in
real coordinates. This defines $\widetilde G(t)=G(t)+h(t)$ on its positive
metric domain. For field-mediated interactions retain the full retarded
kernel of {prf:ref}`def-retarded-interaction-potential`.
:::

:::{prf:proposition} Derivative of the strategic metric
:label: prop-retarded-metric-propagation

For differentiable coefficients, the product rule gives
$$
\dot h=\sum_j\left[\dot\beta_jJ_j^*H_jJ_j+
\beta_j\dot J_j^*H_jJ_j+\beta_jJ_j^*\dot H_jJ_j+
\beta_jJ_j^*H_j\dot J_j\right].
$$
For $H_j(t-\tau_j(t))$, its derivative includes
$(1-\dot\tau_j)H_j'(t-\tau_j)$, in addition to any current-state
dependence. Then $\dot{\widetilde G}=\dot G+\dot h$.
These are differentiation identities; a wave equation for the metric would
have to follow from its own evolution equation. $\square$
:::

:::{prf:definition} Joint WFR Action (Relativistic)
:label: def-joint-wfr-action

The N-agent WFR action on the product space with retarded interactions is:

$$
\mathcal{A}^{(N)}[\boldsymbol{\rho}, \mathbf{v}, \mathbf{r}] = \int_0^T \left[ \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \left(\|v^{(i)}\|_{\tilde{G}^{(i)}}^2 + \lambda_i^2 |r^{(i)}|^2 \right) d\rho^{(i)} + \mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) \right] dt,

$$
where:
- $v^{(i)}$ is the velocity field for Agent $i$'s belief flow
- $r^{(i)}$ is the reaction term (mass creation/destruction)
- $\tilde{G}^{(i)}$ is the game-augmented metric with retarded components (Definition {prf:ref}`def-retarded-game-tensor`)
- $\mathcal{V}_{\text{int}}^{\text{ret}}(\boldsymbol{\rho}, t) = \sum_{i=1}^N \int_{\mathcal{Z}^{(i)}} \Phi^{\text{ret}}_{i}(z^{(i)}, t) \, d\rho^{(i)}(z^{(i)})$ is the retarded interaction energy, with $\Phi^{\text{ret}}_{i} := \sum_{j \neq i} \Phi^{\text{ret}}_{ij}$

*Cross-reference:* Definition {prf:ref}`def-the-wfr-action`, Definition {prf:ref}`def-retarded-interaction-potential`.

:::

:::{prf:theorem} Time averages of the stated field dynamics
:label: thm-nash-standing-wave

For a bounded differentiable density trajectory,
$T^{-1}\int_0^T\partial_t\rho\,dt=(\rho(T)-\rho(0))/T\to0$.
This holds for many nonequilibrium trajectories and does not test unilateral
payoff improvements. Moreover $\langle\rho v\rangle$ need not vanish when
$\langle v\rangle=0$: on a periodic clock take $v=\sin t$ and
$\rho=1+\epsilon\sin t$, $0<\epsilon<1$, giving
$\langle\rho v\rangle=\epsilon/2$.
Standing-wave expansions describe solutions of the defined wave operator;
Nash conditions are the payoff inequalities in
{prf:ref}`thm-nash-equilibrium-as-geometric-stasis`. $\square$
:::

:::{prf:corollary} Delay residual for payoff evaluation
:label: cor-newtonian-nash-limit

The vanishing-delay estimate of {prf:ref}`cor-newtonian-limit-ghost` compares
continuous payoff evaluations at current and retarded states. It does not
establish convergence of equilibria or wave solutions, which are different
objects. Nash membership is evaluated by the unilateral inequalities above.
:::

:::{prf:theorem} Unilateral payoff tests and local stationarity
:label: thm-nash-equilibrium-as-geometric-stasis

The exact Nash condition is
$V_i(z_i^*,z_{-i}^*)\ge V_i(z_i,z_{-i}^*)$ for every feasible unilateral
choice. At an interior twice-differentiable optimum, variations
$z_i^*+t\xi$ give first derivative zero and second derivative nonpositive.
These are necessary conditions, not an equivalence: $V(x)=x^4$ has zero
first and second derivatives at zero but admits improving moves.
For constrained strategies use the feasible variations rather than an
unrestricted gradient. A stationary tensor at a fixed profile adds no test
of the global payoff inequality. $\square$
:::

:::{prf:corollary} Current at zero drift
:label: cor-vanishing-current-nash

At a point where the defined drift vanishes, $J=\rho v=0$ by multiplication.
Time-averaged drift alone does not give this conclusion for a time-dependent
density, as the explicit example in {prf:ref}`thm-nash-standing-wave` shows.
:::

:::{prf:remark} Thermodynamic vs. Resolution Limit
:label: rem-mean-field-vs-levin-length

The continuum limit used here is the **population/thermodynamic limit** $N \to \infty$ with
empirical measures $\mu_N \rightharpoonup \rho$, at **fixed** Levin length $\ell_L>0$. This is a
mean-field limit, not a UV limit. The Levin length is an operational resolution bound (Axiom
{prf:ref}`ax-constructive-finite-resolution`), not a lattice regulator to be sent to zero. Taking
$\ell_L \to 0$ would exit the framework by violating the Causal Information Bound and is **not**
required for validity. The continuum objects are the density fields $\rho$ at fixed resolution.

:::

:::{prf:theorem} Empirical normalization and kernel comparison
:label: thm-mean-field-metric-law

For a test agent and a defined pulled-back Hessian kernel $K_{ab}(z,\zeta)$,
the normalized interaction has
$$
h_{N,ab}(z)=\frac\alpha N\sum_{j\ne i}K_{ab}(z,z_j)
=\alpha\int K_{ab}(z,\zeta)d\mu_N(\zeta)
-\frac\alpha N K_{ab}(z,z_i).
$$
This is an exact finite identity, obtained by adding and subtracting the
diagonal term. Weak convergence evaluates bounded continuous kernels; it
does not by itself evaluate a singular Green-function Hessian.
For an explicitly resolved kernel $K_\ell$ the exact comparison is
$$
|h_{N,\ell}(z)-\alpha\int K_\ell(z,\zeta)d\mu(\zeta)|
\le |\alpha|\left|\int K_\ell(z,\zeta)d(\mu_N-\mu)(\zeta)\right|
+\frac{|\alpha|}{N}|K_\ell(z,z_i)|.
$$
When $K_\ell(z,\cdot)$ is Lipschitz, every coupling of $\mu_N,\mu$
bounds the first integral by its Lipschitz constant times the coupling's
mean distance; taking the infimum gives the $W_1$ bound. The chosen resolution
and its derivative constants remain in this estimate. For the unregularized
three-dimensional screened kernel, the $1/r$ singularity has a Hessian of
order $r^{-3}$; its integral is not justified by weak convergence.
This preserves the finite-resolution comparison without claiming a singular
mean-field limit. $\square$
:::

:::{prf:theorem} Metabolic cost of exact tracking
:label: thm-metabolic-tracking-bound

The established transport cost gives, along an exactly tracked differentiable
target, $\dot{\mathcal M}=\tfrac12\sigma_{\mathrm{met}}
\|\dot z^*\|_{\widetilde G}^2$. Hence the budget implies
$\|\dot z^*\|_{\widetilde G}\le
\sqrt{2\dot{\mathcal M}_{\max}/\sigma_{\mathrm{met}}}$.
This follows by substituting $v=\dot z^*$ into the cost and solving the
inequality. It is a necessary budget test, not a sufficiency proof for
tracking with delayed observations, noise, or restricted controls. $\square$
:::

:::{prf:theorem} Metric inflation and prescribed-velocity cost
:label: thm-geometric-locking-principle

For $h\succeq0$, the additional kinetic cost at a prescribed velocity is
$v^\top hv\ge0$. For the gradient response $v=(G+h)^{-1}p$, the cost
instead equals $p^\top(G+h)^{-1}p\le p^\top G^{-1}p$.
To prove the inequality, conjugate by $G^{-1/2}$: all eigenvalues of
$(I+G^{-1/2}hG^{-1/2})^{-1}$ lie in $(0,1]$.
Thus slowing the response can reduce the cost without reducing the tensor.
Metric inflation alone supplies neither a Lyapunov law for
$\operatorname{Tr}(G^{-1}h)$ nor convergence to cooperation. $\square$
:::

:::{prf:corollary} Scope of the metabolic comparison
:label: cor-metabolic-cooperation

The preceding identities compare kinetic costs for fixed velocity and fixed
force. At $v=0$ the kinetic term vanishes for every finite positive metric,
so its minimization does not select a cooperative tensor. Also
$\partial_y^2(xy)=0$ while $\partial_x\partial_y(xy)=1$; vanishing opponent
Hessian does not imply strategic decoupling. These explicit calculations
replace an inference of cooperation from the metric sign alone.
:::

:::{prf:axiom} Local Gauge Invariance (Nuisance Invariance)
:label: ax-local-gauge-invariance

The physical dynamics of the multi-agent system are invariant under position-dependent rotations of the internal nuisance coordinates. Formally, let $G$ be a compact Lie group with Lie algebra $\mathfrak{g}$. For any smooth map $U: \mathcal{Z} \to G$, the nuisance-frame transformation

$$
\xi'(z) = U(z)\xi(z), \qquad \psi'(z, t) = U(z)\psi(z, t)

$$

leaves observable quantities (reward, policy output, Nash conditions) unchanged. The scalar fields $\rho$ and $V$ are gauge-invariant; only the internal orientation $\xi$ (and any vector-valued nuisance features) transform.

*Units:* $[U] = \text{dimensionless}$ (group element).

*Interpretation:* Agent $i$ at location $z$ is free to rotate its internal representation (the "basis" in which it encodes nuisance). This is not a symmetry to be broken but a **redundancy** in the description that must be properly handled via gauge theory.

:::

:::{prf:definition} Local Gauge Group
:label: def-local-gauge-group

The **Local Gauge Group** is a compact Lie group $G$ with:

1. **Lie algebra $\mathfrak{g}$:** The tangent space at identity, with generators $\{T_a\}_{a=1}^{\dim(G)}$ satisfying $[T_a, T_b] = if^{abc}T_c$ where $f^{abc}$ are the **structure constants**.

2. **Representation:** Use a unitary representation on the matter fiber with its invariant Hermitian inner product; compactness permits averaging any positive inner product over normalized Haar measure.

3. **Position-dependent element:** $U(z) \in G$ for each $z \in \mathcal{Z}$, forming the infinite-dimensional group of gauge transformations $\mathcal{G} := C^\infty(\mathcal{Z}, G)$.

*Standard choices:*
- $G = SO(D)$: Rotations of $D$-dimensional nuisance space
- $G = SU(N)$: Unitary transformations (for complex representations)
- $G = U(1)$: Abelian phase rotations (electromagnetic limit)

*Cross-reference:* For the standard rotation action, the origin has stabilizer $SO(D)$; see {prf:ref}`conj-nuisance-fiber-gauge-orbit`.

:::

:::{prf:definition} Matter Field (Belief Amplitude)
:label: def-matter-field-belief-amplitude

The **Matter Field** for agent $i$ is the complex-valued section

$$
\psi^{(i)}: \mathcal{Z}^{(i)} \times \mathbb{R} \to V

$$

where $V$ is the representation space of $G$. The matter field is related to the belief wave-function by:

$$
\psi^{(i)}(z, t) = \sqrt{\rho^{(i)}(z, t)} \exp\left(\frac{iV^{(i)}(z, t)}{\sigma}\right) \cdot \xi^{(i)}(z)

$$

where:
- $\rho^{(i)}$ is the belief density
- $V^{(i)}$ is the value function (scalar, gauge-invariant)
- $\sigma > 0$ is the **cognitive action scale**, $\sigma := T_c \cdot \tau_{\text{update}}$, the information-theoretic analog of Planck's constant (full definition: {prf:ref}`def-cognitive-action-scale` in {ref}`sec-the-belief-wave-function-schrodinger-representation`)
- $\xi^{(i)}(z) \in V$ is a unit internal vector, so $\psi^\dagger\psi=\rho$

*Units:* $[\psi] = [\text{length}]^{-D/2}$ (probability amplitude density).

*Transformation law:* Under gauge transformation $U(z)$:

$$
\psi'^{(i)}(z, t) = \rho(U(z))\psi^{(i)}(z, t)

$$

where $\rho: G \to GL(V)$ is the representation. The scalar observables are unchanged: $\rho' = \rho$ and $V' = V$.

:::

:::{prf:proposition} Orbits and stabilizers of the declared action
:label: conj-nuisance-fiber-gauge-orbit

For the declared smooth compact-group action, the orbit through $\xi$ is
$G\xi\cong G/H_\xi$, where $H_\xi=\{g:g\xi=\xi\}$.
The map $gH_\xi\mapsto g\xi$ is well-defined and bijective: two images
agree exactly when the representatives differ by an element of $H_\xi$.
Its differential has kernel the stabilizer Lie algebra; the compact orbit
is embedded, giving the homogeneous-space identification.
For the standard rotation action on $\mathbb R^D$,
$H_0=SO(D)$ and $SO(D)0=\{0\}$. At a nonzero vector the stabilizer is
$SO(D-1)$ and the orbit is its fixed-radius sphere. Isotropy at the origin
therefore means the whole group fixes the origin. The VQ nuisance fiber
is the fiber defined by the encoder; identifying all of it with a single
orbit would require equality of the two defined sets, which the rotation
calculation does not establish. The gauge algebra below uses the declared
action directly. $\square$
:::

:::{prf:definition} Strategic Connection (Gauge Potential)
:label: def-strategic-connection

The **Strategic Connection** is a $\mathfrak{g}$-valued 1-form on $\mathcal{Z}$:

$$
A = A_\mu^a T_a \, dz^\mu

$$

where:
- $A_\mu^a(z, t)$ are the **connection coefficients** (real-valued functions)
- $\{T_a\}_{a=1}^{\dim(\mathfrak{g})}$ are the generators of the Lie algebra $\mathfrak{g}$
- $\mu$ indexes spacetime/latent coordinates $(t, z^1, \ldots, z^D)$

*Units:* $[A_\mu] = [\text{length}]^{-1}$ (inverse length, like momentum).

*Interpretation:* The connection $A_\mu$ tells agent $i$ how to "translate" the nuisance interpretation from point $z$ to point $z + dz$. It is the **strategic context** required to compare internal states at different locations.

:::

:::{prf:proposition} Gauge Transformation of the Connection
:label: prop-gauge-transformation-connection

Under a local gauge transformation $U(z) \in G$, the connection transforms as:

$$
A'_\mu = U A_\mu U^{-1} - \frac{i}{g}(\partial_\mu U)U^{-1}

$$

where $g > 0$ is the **coupling constant** (strategic coupling strength).

:::

:::{prf:definition} Covariant Derivative
:label: def-covariant-derivative

The **Covariant Derivative** acting on matter fields is:

$$
D_\mu = \partial_\mu - igA_\mu

$$

For a matter field $\psi$ in representation $\rho$:

$$
D_\mu\psi = \partial_\mu\psi - igA_\mu^a \rho(T_a)\psi

$$

*Properties:*
1. **Covariant transformation:** $(D_\mu\psi)' = U(D_\mu\psi)$
2. **Leibniz rule:** On a tensor product use the induced connection: $D(\psi\otimes\chi)=D\psi\otimes\chi+\psi\otimes D\chi$.
3. **Reduces to partial derivative** when $A_\mu = 0$ (trivial connection)

*Units:* $[D_\mu\psi] = [\psi]/[\text{length}]$.

:::

:::{prf:theorem} Gauge-covariant wave operator
:label: thm-gauge-covariant-klein-gordon

For $D_\mu=\partial_\mu-igA_\mu$, define
$$
\Box_A\psi=-|g|^{-1/2}D_\mu(\sqrt{|g|}g^{\mu\nu}D_\nu\psi).
$$
For $g=\operatorname{diag}(-c^2,\widetilde G(t,z))$ with constant $c$,
$$
\Box_A\psi=c^{-2}\left[D_t^2\psi+
\partial_t\log\sqrt{|\widetilde G|}\,D_t\psi\right]
-\frac1{\sqrt{|\widetilde G|}}D_i
(\sqrt{|\widetilde G|}\widetilde G^{ij}D_j\psi).
$$
:::

:::{prf:proposition} Minimal Coupling Principle
:label: prop-minimal-coupling

To maintain gauge invariance, derivatives acting on gauge-charged fields must be replaced by covariant derivatives:

$$
\partial_\mu \longrightarrow D_\mu = \partial_\mu - igA_\mu

$$

This **Minimal Coupling Principle** ensures that:
1. Transport of nuisance-frame vectors is covariant
2. Matter-field dynamics (e.g., $\psi$) are gauge-covariant
3. Learning gradients for gauge-charged features transform properly under internal rotations

*Consequence for implementation:* Use covariant gradients for parameters that live in gauge bundles. Scalar objectives like $V$ remain invariant and use ordinary gradients.

:::

:::{prf:proposition} Representations of differentiated fields
:label: prop-game-tensor-gauge-transformation

Since $V$ is invariant, its Riemannian Hessian is invariant under internal
frame changes. For a matter vector, $D'_kD'_l\psi'=UD_kD_l\psi$,
not conjugation. For an endomorphism $M'=UMU^{-1}$, the adjoint derivative
does transform by conjugation. Both assertions follow by applying the
appropriate intertwining identity twice. Inner products of vector-valued
derivatives and traces of endomorphism products supply scalar invariants.
:::

:::{prf:definition} Invariant strategic Hessian
:label: def-gauge-covariant-game-tensor

Use $H_{mn}=\partial_m\partial_nV-\Gamma^r_{mn}\partial_rV$ with
the intrinsic connection already fixed in {prf:ref}`def-the-game-tensor`.
Its pullback is $J^*HJ$. This fixes the metric used in the definition before
forming $\widetilde G$ and avoids an implicit circular definition through
the unknown perturbed connection. Charged vectors and endomorphisms use
their distinct transformation laws in
{prf:ref}`prop-game-tensor-gauge-transformation`.
:::

:::{prf:theorem} Invariant scalar contractions
:label: thm-gauge-invariant-metric-inflation

The pulled-back scalar Hessian is internally invariant because $V$ and
the base geometry are invariant. For a charged vector derivative $u_a$, the
bilinear form $\operatorname{Re}\langle u_a,u_b\rangle$ is invariant:
$\langle Uu_a,Uu_b\rangle=\langle u_a,u_b\rangle$ by unitarity.
For endomorphism derivatives $M_a$, cyclicity gives
$\operatorname{Tr}[(UM_aU^{-1})(UM_bU^{-1})]
=\operatorname{Tr}(M_aM_b)$. These are the appropriate scalar contractions
for their respective representations. Adding them to a metric still uses
the explicit positive-definiteness test of the metric prescription;
invariance and positivity are separate algebraic properties. $\square$
:::

:::{prf:definition} Field Strength Tensor (Yang-Mills Curvature)
:label: def-field-strength-tensor

The **Field Strength Tensor** is the $\mathfrak{g}$-valued 2-form:

$$
\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]

$$

In components with Lie algebra generators:

$$
\mathcal{F}_{\mu\nu}^a = \partial_\mu A_\nu^a - \partial_\nu A_\mu^a + gf^{abc}A_\mu^b A_\nu^c

$$

where $f^{abc}$ are the structure constants of $\mathfrak{g}$.

*Units:* $[\mathcal{F}_{\mu\nu}] = [\text{length}]^{-2}$ (curvature).

*Special cases:*
- **Abelian ($[A_\mu, A_\nu] = 0$):** $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ (electromagnetic field tensor)
- **Non-Abelian:** The commutator term generates **self-interaction** of the gauge field

:::

:::{prf:proposition} Covariant Transformation of Field Strength
:label: prop-field-strength-transformation

Under gauge transformation $U(z)$, the field strength transforms **covariantly** (not invariantly):

$$
\mathcal{F}'_{\mu\nu} = U \mathcal{F}_{\mu\nu} U^{-1}

$$

:::

:::{prf:theorem} Curvature from Covariant Derivative Commutator
:label: thm-curvature-commutator

The field strength measures the failure of covariant derivatives to commute:

$$
[D_\mu, D_\nu]\psi = -ig\mathcal{F}_{\mu\nu}\psi

$$

:::

:::{prf:theorem} Bianchi identity from the operator Jacobi identity
:label: thm-bianchi-identity

For the defined connection, the adjoint covariant derivative is
$\mathcal D_\rho F_{\mu\nu}=\partial_\rho F_{\mu\nu}-ig[A_\rho,F_{\mu\nu}]$.
On a test section $u$, expansion gives
$[D_\rho,F_{\mu\nu}]u=(\mathcal D_\rho F_{\mu\nu})u$.
Insert $[D_\mu,D_\nu]=-igF_{\mu\nu}$ into
$[D_\rho,[D_\mu,D_\nu]]+\mathrm{cyclic}=0$. Dividing by $-ig$ gives
$$
\mathcal D_\rho F_{\mu\nu}+\mathcal D_\mu F_{\nu\rho}
+\mathcal D_\nu F_{\rho\mu}=0.
$$
For $g=0$ the same identity is $d(dA)=0$. The geometric Christoffel
terms cancel under cyclic antisymmetrization for the torsion-free connection.
The identity holds in each smooth gauge chart and is preserved under
transition functions by conjugation; nontrivial bundle topology does not
violate it. $\square$
:::

:::{prf:definition} Strategic Curvature Scalar
:label: def-strategic-curvature-scalar

The **Strategic Curvature Scalar** is the gauge-invariant contraction:

$$
\mathcal{R}_{\text{strat}} := \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a \mathcal{F}^{\mu\nu,a}

$$

where indices are raised with the spacetime metric $g^{\mu\nu} = \text{diag}(-1/c_{\text{info}}^2, \tilde{G}^{ij})$ introduced above.

*Properties:*
- In Euclidean signature, $\mathcal{R}_{\text{strat}}$ is non-negative for compact gauge groups; in Lorentzian signature it is indefinite.
- In Euclidean signature with positive invariant trace, zero norm is equivalent to $\mathcal F=0$. In Lorentzian signature, $F_{\mu\nu}F^{\mu\nu}=2(|B|^2-|E|^2)$ can vanish for a nonzero field.
- Provides a measure of total strategic tension in a region

:::

:::{prf:definition} Yang-Mills Action
:label: def-yang-mills-action

The **Yang-Mills Action** for the strategic gauge field is:

$$
S_{\text{YM}}[A] = -\frac{1}{4}\int_{\mathcal{Z} \times \mathbb{R}} \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})\sqrt{|g|}\,d^{D+1}x

$$

where:
- $\mathcal{F}_{\mu\nu}$ is the field strength tensor (Definition {prf:ref}`def-field-strength-tensor`)
- $g_{\mu\nu}$ is the spacetime metric $g_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, \tilde{G}_{ij})$ with determinant $|g| = c_{\text{info}}^2|\tilde{G}|$
- $g_{\text{YM}}$ is the coupling constant
- The trace is over Lie algebra indices: $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu}) = \mathcal{F}_{\mu\nu}^a\mathcal{F}^{\mu\nu,a}$

*Units:* $[S_{\text{YM}}] = \text{nat}$ (action).
*Dimensionality:* In spacetime dimension $d = D+1$, the coupling has $[g^2] = [\text{length}]^{d-4}$ (so $g$ is dimensionless only when $d = 4$).

*Properties:*
1. **Gauge-invariant:** $S_{\text{YM}}[A'] = S_{\text{YM}}[A]$ under $A \to A'$
2. **Lorentz-invariant:** Covariant under coordinate transformations
3. **Positive in Euclidean signature:** After Wick rotation the action is positive semi-definite for compact gauge groups; in Lorentzian signature it is indefinite.

:::

:::{prf:theorem} Variation of the gauge and matter actions
:label: thm-yang-mills-equations

Use signature $(-,+,\ldots,+)$ and the invariant bilinear form normalized
by $\operatorname{Tr}(T_aT_b)=\delta_{ab}$ in the gauge action. Matter
representation matrices retain the same Lie-algebra basis. For
$S_{\mathrm{YM}}=-\tfrac14\int\operatorname{Tr}F_{\mu\nu}F^{\mu\nu}d\mu_g$,
stationarity of the total action gives
$$
\mathcal D_\mu F^{\mu\nu}=J^\nu,\qquad
J^{\nu,a}=-\frac{\delta\mathcal L_m}{\delta A_\nu^a},\qquad
\mathcal D_\mu F^{\mu\nu}=
|g|^{-1/2}\partial_\mu(\sqrt{|g|}F^{\mu\nu})-ig[A_\mu,F^{\mu\nu}].
$$
For the corrected scalar Lagrangian
$\mathcal L_m=-(D_\mu\psi)^\dagger D^\mu\psi-m^2\psi^\dagger\psi$,
$J^{\nu,a}=-2g\operatorname{Im}(\psi^\dagger T^aD^\nu\psi)$.

:::

:::{prf:corollary} Abelian Limit (Maxwell Equations)
:label: cor-maxwell-limit

For an Abelian gauge group $G = U(1)$ with $[T_a, T_b] = 0$:

$$
\partial_\mu F^{\mu\nu} = J^\nu

$$

This recovers the **Maxwell equations** of electromagnetism in covariant form.

*Correspondence:*
- $F^{0i} = E^i$ (electric field) $\leftrightarrow$ temporal strategic gradient
- $F^{ij} = \epsilon^{ijk}B_k$ (magnetic field) $\leftrightarrow$ spatial strategic vorticity
- $J^0 = \rho_e$ (charge density) $\leftrightarrow$ belief density
- $J^i = j^i$ (current density) $\leftrightarrow$ belief flux

:::

:::{prf:proposition} Gauge stress tensor in the stated signature
:label: prop-gauge-energy-momentum

Metric variation of the gauge action gives
$$
T_{\mu\nu}=\operatorname{Tr}\left(F_{\mu\rho}F_\nu{}^\rho
-\tfrac14g_{\mu\nu}F_{\rho\sigma}F^{\rho\sigma}\right).
$$
:::

:::{prf:corollary} Covariant charge conservation
:label: cor-current-conservation

The gauge equation gives $\mathcal D_\nu J^\nu=0$.
:::

:::{prf:definition} Scalar belief and gauge action
:label: def-complete-lagrangian

For the scalar belief multiplets already defined, use
$$
\mathcal L=-\tfrac14\operatorname{Tr}F_{\mu\nu}F^{\mu\nu}
-\sum_i\left[(D_\mu\psi_i)^\dagger D^\mu\psi_i+m_i^2\psi_i^\dagger\psi_i\right]
-(D_\mu\Phi)^\dagger D^\mu\Phi-U(\Phi),
\qquad U=\mu^2\Phi^\dagger\Phi+\lambda(\Phi^\dagger\Phi)^2.
$$
This is the scalar field model with signature $(-,+,\ldots,+)$.
The scalar belief representation does not define spinor indices or a Dirac
adjoint. A separately defined spinor model must supply those structures
and its invariant Yukawa contraction before a fermion mass can be computed.
The connection is normalized by $D=\partial-igA$; in dimension $d$ the
canonical dimensions are $[A]=L^{1-d/2}$, $[g]=L^{d/2-2}$, so $[gA]=L^{-1}$.
These reduce to the earlier inverse-length $A$ convention at $d=4$.
The field action defines this model; comparison with the Bellman and WFR
generators uses their explicit equations, not equality of terminology.
:::

:::{prf:theorem} Vacuum expansion and gauge mass matrix
:label: thm-higgs-mechanism

For the stable quartic potential already specified, $\lambda>0$ and
$\mu^2<0$, write $\Phi_0=(v/\sqrt2)n$, $n^\dagger n=1$ and
$v^2=-\mu^2/\lambda$. The radial mass is $m_h^2=2\lambda v^2$.
The gauge quadratic term is $-\tfrac12A_\mu^a(M^2)_{ab}A^{\mu b}$ with
$$
(M^2)_{ab}=g^2\Phi_0^\dagger\{T_a,T_b\}\Phi_0.
$$
:::

:::{prf:corollary} Vacuum tangent directions
:label: cor-goldstone-absorption

The tangent space to the vacuum orbit is spanned by $T_a\Phi_0$.
The mass-matrix calculation identifies its nullspace with the unbroken
generators. Local gauge coordinates along the orbit can be removed in a
local gauge chart; the remaining radial fluctuation has mass $m_h$.
This counts orbit and stabilizer directions and does not identify a
gauge-dependent vacuum orientation as an observable. $\square$
:::

:::{prf:definition} Spectral quantities of the specified operator
:label: def-mass-gap

For a self-adjoint $H$ with a ground eigenvalue $E_0$, use
$\Delta_H=\inf(\operatorname{spec}H\setminus\{E_0\})-E_0$.
This measures the gap above the whole ground eigenspace and does not imply
ground-state uniqueness. For the wave operator the rest frequency is
$m_{\mathrm{rest}}=c\kappa$; on a compact spatial realization denote the
spacing between distinct frequencies by $\delta\omega$.
The former notation $\Delta_{\mathrm{KG}}$ is used here only for this
specified mode spacing, never interchangeably with $m_{\mathrm{rest}}$.
Energy units introduce the action scale: $E=\sigma\omega$.
:::

:::{prf:theorem} Exact wave frequencies and envelope expansion
:label: thm-mass-gap-screening

For the spatial realization $-\Delta_G\phi_n=\lambda_n\phi_n$,
substitution of $e^{-i\omega t}\phi_n$ gives
$\omega_n=c\sqrt{\kappa^2+\lambda_n}$.
For $a=\kappa^2+\lambda_0>0$ and $x=\lambda_n-\lambda_0\ge0$,
Taylor's integral formula gives
$$
\left|\omega_n-c\sqrt a-\frac{cx}{2\sqrt a}\right|
\le\frac{cx^2}{8a^{3/2}}.
$$
Indeed the second derivative of $\sqrt{a+x}$ is
$-1/[4(a+x)^{3/2}]$, whose absolute value is at most $1/(4a^{3/2})$.
Thus the slow envelope has leading generator
$\sigma c(-\Delta_G-\lambda_0)/(2\sqrt a)$ after the ground frequency
is subtracted. A massless compact model can have positive mode spacing;
a massive infinite-volume model can have continuous spatial momenta. The
formula distinguishes these cases directly. $\square$
:::

:::{prf:proposition} Information and spectral support are distinct quantities
:label: lem-cib-excludes-massless-spectral

The Causal Information Bound controls its defined representational information.
It does not identify that quantity with an unnormalized integral of field
correlations. For finite normalized subsystems, the established relative-entropy
bound gives
$$
I(A:B)\ge\frac{|\langle XY\rangle-\langle X\rangle\langle Y\rangle|^2}
{2\|X\|^2\|Y\|^2}.
$$
To obtain this, write $I=D(\rho_{AB}\|\rho_A\otimes\rho_B)$, apply
$D\ge\tfrac12\|\rho_{AB}-\rho_A\otimes\rho_B\|_1^2$, and bound
the correlation by the trace norm times $\|X\|\|Y\|$.
This calculation is for bounded observables of the same two subsystems.
There is no additive sum over all pairs: $n$ copies of one fair bit have
unit pairwise covariance but mutual information $\log2$ between any two
nonempty groups of copies. Their pair sum grows quadratically.
Consequently the former spectral exclusion based on that pair sum is not
a consequence of the information bound. $\square$
:::

:::{prf:proposition} Quadratic masses and their scope
:label: lem-scalar-lightest-condition

The vacuum expansion in {prf:ref}`thm-higgs-mechanism` gives
$m_h^2=2\lambda v^2$ and the eigenvalues of
$g^2\Phi_0^\dagger\{T_a,T_b\}\Phi_0$. Comparing these numbers orders
the elementary quadratic fluctuation operators. It does not order all
gauge-invariant composite excitations of an interacting Hamiltonian.
This follows because the expansion computes only the second variation of
the action; higher interaction terms remain in the full operator.
:::

:::{prf:proposition} Operator comparison and transfer gaps
:label: lem-kg-ym-gap-bridge

For a specified unitary $W$ and self-adjoint operators related by
$H_2=WH_1W^{-1}$, spectral calculus gives
$e^{-tH_2}=We^{-tH_1}W^{-1}$ and equal spectra. To prove the latter,
$(z-H_2)^{-1}=W(z-H_1)^{-1}W^{-1}$ exactly when either inverse exists.
Thus an established gap transfers under this operator identification.
Equality of a scalar screening coefficient and a gauge-action coefficient
is not such an operator identity. No Yang--Mills gap is inferred here from
the scalar coefficient alone. $\square$
:::

:::{prf:theorem} Finite resolution and a spectral counterexample
:label: thm-computational-necessity-mass-gap

Finite local state spaces do not imply a gap uniform in system size.
On $n$ sites with nearest-neighbor discrete Laplacian and periodic boundary,
$f_k(j)=e^{2\pi i kj/n}$ gives
$(-\Delta_n)f_k=4\sin^2(\pi k/n)f_k$ by direct substitution.
The first positive eigenvalue is $4\sin^2(\pi/n)\to0$ although the
lattice resolution is fixed. This proves that a resolution bound alone
cannot replace a spectral estimate for the actual limiting operator.
$\square$
:::

:::{prf:proposition} Drift and stochastic motion
:label: lem-nontrivial-evolution

For $dZ=b\,dt+\Sigma\,dW$, Itô's formula gives
$\mathbb E[dZ\mid Z]=b\,dt$ and quadratic variation
$d[Z]_t=\Sigma\Sigma^*dt$. Thus active noise can produce motion with
$b=0$. A nonconstant contribution to a sum of potentials does not prevent
other contributions from canceling its gradient. The constructed drift
must be evaluated as a whole; these identities do not exclude stasis.
:::

:::{prf:remark} Local Hadamard control
:label: lem-massless-kg-decay-causal

The Hadamard form fixes the singular part of the two-point distribution
near its diagonal. Its smooth state-dependent remainder is not fixed by
that local singularity. Consequently this local expansion does not give
a uniform lower bound at arbitrarily large separation. Tiling coordinate
neighborhoods provides local charts, not an inequality relating correlations
of distant points. The same local leading singularity occurs for massive
and massless scalar fields, which further prevents reading an infrared gap
from it alone.
:::

:::{prf:theorem} Spectral conclusions supported by the construction
:label: thm-mass-gap-constructive

For the compact connected scalar realization of Appendix E.7, the
self-adjoint Schrödinger operator has compact resolvent and its ground
eigenvalue is simple, giving a positive gap at that fixed realization.
These are the conclusions of {prf:ref}`thm-e7-ground-state-positivity`.
The value of this gap depends on its operator, metric, potential, boundary,
volume, and action scale. The finite-resolution estimate alone gives no
uniform lower bound, as shown by
{prf:ref}`thm-computational-necessity-mass-gap`.

:::

:::{prf:corollary} Fixed-model spectral estimate
:label: cor-mass-gap-existence

For the fixed scalar operator and ground projection $P_0$ above, the
spectral expansion gives
$\|e^{-t(H-E_0)}(I-P_0)\|\le e^{-t\Delta_H}$.
This follows by taking the supremum of $e^{-t(E-E_0)}$ over excited
spectral values. It applies to this same operator and clock.
:::

:::{prf:remark} Confinement and information compression
:label: cor-confinement-data-compression

Compression bounds describe represented information. Wilson confinement
describes the law of gauge holonomies. Neither is defined by the other's
observable. The calculations here establish no equality between a boundary
capacity bound and a Wilson-loop area law.
:::

:::{prf:corollary} Small-gap relaxation
:label: cor-criticality-unstable

The preceding semigroup estimate weakens as its actual $\Delta_H$ decreases.
It gives a relaxation timescale bound and does not exclude critical models
from the space of mathematical or computational constructions.
:::

:::{prf:definition} Capacity violation at the declared resolution
:label: def-computational-swampland

For a specified represented information functional $I(R)$ and capacity
$C(R)$, define the violating set by $\{\mathcal T:\exists R,
I_{\mathcal T}(R)>C_{\mathcal T}(R)\}$. Membership is tested using these
same quantities. It is not equivalent by definition to masslessness,
vanishing lattice spacing, or algebraic correlation decay.
:::

:::{prf:proposition} Correlation integrals and capacity tests
:label: thm-cft-swampland

For $0<2\Delta<d$, rescaling $x=Ru,y=Rv$ gives
$\int_{B_R}\int_{B_R}|x-y|^{-2\Delta}dxdy
=R^{2d-2\Delta}\int_{B_1}\int_{B_1}|u-v|^{-2\Delta}dudy$.
The last integral is finite by local integrability of $r^{d-1-2\Delta}$.
This proves the scaling of a correlation integral. It supplies no lower
bound for the represented information functional: such an identification
was not established, and the repeated-bit example in
{prf:ref}`lem-cib-excludes-massless-spectral` rules out the general pair-sum
argument. Thus the former CFT exclusion does not follow. $\square$
:::

:::{prf:corollary} Finite-size mode spacing
:label: cor-finite-volume-mass-gap

On a periodic box of side $L$, the spatial eigenvalues are
$(2\pi/L)^2|k|^2$. The massless wave frequencies of the nonzero modes are
$2\pi c|k|/L$, by {prf:ref}`thm-mass-gap-screening`.
The zero mode remains and must be treated in the actual Hamiltonian
realization. The smallest positive spatial frequency scales as $L^{-1}$;
finite volume alone neither removes every field zero mode nor supplies
a gap uniform as $L\to\infty$. $\square$
:::

:::{prf:proposition} Coarse-graining identity for the capacity ratio
:label: thm-scale-covariance-bound

For unchanged boundary area and resolution $\ell'=\alpha\ell$,
$C'=C/\alpha^{d-1}$. Data processing gives $I'\le I$, so
$I'/C'\le\alpha^{d-1}I/C$. Preservation of the capacity inequality is
exactly $I'\le C/\alpha^{d-1}$, which must be evaluated for the declared
coarse-graining. For example, $I'=I=C$ satisfies data processing but
violates the new bound for $\alpha>1$. This proves that data processing
alone does not establish the former scale-covariance conclusion.
$\square$
:::

:::{prf:remark} Dependency order for gauge spectral conclusions
:label: thm-mass-gap-dichotomy

The order established here is: declared gauge action, variational field
equations, specified Hilbert-space realization, and spectral estimates for
that realization. The scalar gap of Appendix E.7 transfers to another
operator only through an operator identification such as
{prf:ref}`lem-kg-ym-gap-bridge`. The forward-referenced OS clustering proof
uses a mass-gap estimate and therefore cannot independently establish that
same estimate. The information and finite-resolution calculations above
provide no substitute for this operator step.
:::

:::{prf:remark} Scope of the gauge construction
:label: rem-clay-millennium

The classical connection, curvature, action, and field equations are
constructed in this chapter. The compact scalar spectral results refer to
their defined scalar Hamiltonian. These calculations do not establish a
nontrivial continuum quantum Yang--Mills measure on $\mathbb R^4$ or its
Hamiltonian gap. The prior claim that the Clay requirements were met by
the information-bound argument is withdrawn because its correlation
inequality and spectral identification fail as shown above.
:::

:::{prf:definition} Inference Hilbert Space
:label: def-inference-hilbert-space

Let $(\mathcal{Z}, G)$ be the latent manifold with capacity-constrained metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`). The **Inference Hilbert Space** is:

$$
\mathcal{H} := L^2(\mathcal{Z}, d\mu_G), \quad d\mu_G := \sqrt{\det G(z)}\, d^n z,

$$
with inner product:

$$
\langle \psi_1 | \psi_2 \rangle := \int_{\mathcal{Z}} \overline{\psi_1(z)} \psi_2(z)\, d\mu_G(z).

$$
The measure $d\mu_G$ is the **Riemannian volume form**, ensuring coordinate invariance of the inner product.

*Units:* $[\psi] = [z]^{-d/2}$ (probability amplitude density).

*Remark (Coordinate Invariance).* Under a coordinate transformation $z \to z'$, the Jacobian factor $|\partial z/\partial z'|$ cancels with $\sqrt{\det G}$, leaving $\langle \psi_1 | \psi_2 \rangle$ invariant.

*Remark (Field Extensions).* In the field-theoretic layer (SMoC), the scalar space $\mathcal{H}$
is extended to bundle-valued $L^2$ sections (e.g., spinor and gauge bundles) over spacetime
$\mathcal{M}$, with the same measure structure on each fiber.

:::

:::{prf:definition} Belief Wave-Function
:label: def-belief-wave-function

Let $\rho(z, s)$ be the belief density from the WFR dynamics (Definition {prf:ref}`def-the-wfr-action`) and $V(z, s)$ be the value function (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`). The **Belief Wave-Function** is the complex amplitude:

$$
\psi(z, s) := \sqrt{\rho(z, s)} \exp\left(\frac{i V(z, s)}{\sigma}\right),

$$
where $\sigma > 0$ is the **Cognitive Action Scale** (Definition {prf:ref}`def-cognitive-action-scale`).

**Decomposition:**
- **Amplitude:** $R(z, s) := \sqrt{\rho(z, s)} = |\psi(z, s)|$
- **Phase:** $\phi(z, s) := V(z, s)/\sigma = \arg(\psi(z, s))$

**Probability Recovery:**

$$
|\psi(z, s)|^2 = \rho(z, s), \quad \int_{\mathcal{Z}} |\psi|^2 d\mu_G = \int_{\mathcal Z}\rho\,d\mu_G.

$$
*Physical interpretation:* The amplitude $R$ encodes "how much" belief mass is at $z$; the phase $\phi$ encodes "which direction" the belief is flowing (via $\nabla_B V$).

:::

:::{prf:definition} Cognitive action normalization
:label: def-cognitive-action-scale

Define $\sigma=T_c\tau_{\mathrm{update}}>0$ using the established temperature
and update clock. Its units are temperature units times time; the amplitude
phase $V/\sigma$ uses the compatible value/action normalization.
This definition does not identify $\sigma$ with a squared length without
a conversion coefficient, nor does it force arbitrary densities to become
delta functions as $\sigma\to0$. For a fixed density and phase the definition
$\psi_\sigma=\sqrt\rho e^{iV/\sigma}$ has
$|\psi_\sigma|^2=\rho$ for every $\sigma$, proving the distinction.
:::

:::{prf:proposition} Self-adjoint spatial realizations
:label: prop-laplace-beltrami-self-adjointness

The nonnegative form $q[u]=\int|\nabla u|_G^2d\mu_G$ defines $-\Delta_G$.
On a smooth bounded domain its Dirichlet form domain is $H_0^1$ and its
Neumann form domain is $H^1$; each closed form determines its self-adjoint
operator. On a geodesically complete boundaryless manifold the minimal
Laplacian is essentially self-adjoint {cite}`strichartz1983analysis`.
The sign follows from $\langle u,-\Delta_Gu\rangle=q[u]\ge0$.
On an interval the operator on $C_c^\infty(0,1)$ has distinct Dirichlet
and Neumann self-adjoint extensions, so the boundary case is not essential
self-adjointness of that minimal domain. The boundary form domain selects
the realization used in subsequent spectral calculations.
:::

:::{prf:theorem} Exact polar representation of the stated WFR--HJB equations
:label: thm-madelung-transform

On a smooth positive-density chart with the fixed spatial metric of the
WFR equations, put $R=\sqrt\rho$, $p=dV-B$, $v=G^{-1}p$,
$D=\nabla-iB/\sigma$, and $Q_B=-\sigma^2\Delta_GR/(2R)$.
For the equations already stated,
$\partial_s\rho+\operatorname{div}_G(\rho v)=r\rho$ and
$\partial_sV+|p|_G^2/2+\Phi_{\mathrm{eff}}=0$, the exact amplitude equation is

$$
i\sigma\partial_s\psi=
\left[-\tfrac{\sigma^2}{2}\Delta_B+\Phi_{\mathrm{eff}}-Q_B
+\tfrac{i\sigma}{2}r\right]\psi,
\qquad\psi=Re^{iV/\sigma}.
$$

:::

:::{prf:definition} Bohm Quantum Potential (Information Resolution Limit)
:label: def-bohm-quantum-potential

The **Bohm Quantum Potential** is:

$$
Q_B(z, s) := -\frac{\sigma^2}{2} \frac{\Delta_G \sqrt{\rho}}{\sqrt{\rho}} = -\frac{\sigma^2}{2} \frac{\Delta_G R}{R},

$$
where $R = \sqrt{\rho}$ is the amplitude.

**Explicit form in terms of $\rho$:**

$$
Q_B = -\frac{\sigma^2}{8\rho^2} \|\nabla_G \rho\|_G^2 + \frac{\sigma^2}{4\rho} \Delta_G \rho.

$$
**Physical interpretation:** $Q_B$ represents the **energetic cost of belief localization**. Regions where $\rho$ has high curvature (sharp belief features) incur an effective potential energy penalty. This prevents the belief from concentrating to delta functions.

**Information-theoretic interpretation:** $Q_B$ enforces the **Levin Length** ({ref}`sec-saturation-limit`) as a resolution limit. The agent cannot represent distinctions finer than $\ell_L \sim \sqrt{\sigma}$.

*Units:* $[Q_B] = \text{nat}$ (same as potential).

*Cross-reference:* In standard quantum mechanics, $Q_B$ is called the "quantum potential" or "Bohm potential." Here it emerges from the information geometry, not fundamental physics.

:::

:::{prf:corollary} Reaction and norm balance
:label: cor-open-quantum-system

For the amplitude equation and a fixed metric with zero boundary flux,
$d\|\psi\|^2/ds=\int r|\psi|^2d\mu_G$.
Multiply the equation by $\bar\psi$, subtract its conjugate, and integrate:
the real terms cancel and the divergence integrates to zero.
For a density operator, the same prescribed multiplication term contributes
$\tfrac12\{r,\varrho\}$ with trace $\operatorname{Tr}(r\varrho)$.
Vanishing trace for one evolving state does not prove a linear CPTP law
for every state. For fixed Hermitian $r$, vanishing on every rank-one state
forces $r=0$ by polarization. The established GKSL generator has its
explicit recycling terms and must be used as defined in
{prf:ref}`def-gksl-generator`; it is not inferred from a single norm balance.
$\square$
:::

:::{prf:proposition} Geometric covariant Laplacian
:label: prop-operator-ordering-invariance

The kinetic quadratic form fixes the divergence realization
$$
\Delta_B\psi=|G|^{-1/2}D_i(\sqrt{|G|}G^{ij}D_j\psi)
=G^{ij}(D_iD_j-\Gamma^k_{ij}D_k)\psi.
$$
Here the $D_iD_j$ on the right acts on the components without a geometric
connection on the index $j$; the displayed Christoffel term supplies it.
The identity follows from
$|G|^{-1/2}\partial_i(\sqrt{|G|}G^{ik})=-G^{ij}\Gamma^k_{ij}$.
For example in polar coordinates $\Delta r=1/r$; the opposite sign would
give $-1/r$. Coordinate covariance alone does not forbid an additional
scalar curvature potential; the stated quadratic form fixes the operator.
$\square$
:::

:::{prf:corollary} Semiclassical Limit
:label: cor-semiclassical-limit

In the limit $\sigma \to 0$ (classical limit), the Schrödinger dynamics recover the **geodesic flow**:

**WKB Ansatz:** $\psi = a(z) e^{iS(z)/\sigma}$ with $a$ slowly varying.

**Leading Order ($O(\sigma^{-1})$):** The Hamilton-Jacobi equation

$$
\partial_s S + \frac{1}{2}\|\nabla_B S\|_G^2 + \Phi_{\text{eff}} = 0,

$$
**Next Order ($O(\sigma^0)$):** The transport equation

$$
\partial_s |a|^2 + \nabla_G \cdot (|a|^2 \nabla_B S) = 0.

$$
*Definition:* $\nabla_B S := \nabla S - B$. If curl-induced mobility is present, replace the flux by
$|a|^2 \mathcal{M}_{\text{curl}} G^{-1}\nabla_B S$.
These are exactly the HJB and continuity equations from WFR dynamics. The quantum correction $Q_B \to 0$ as $\sigma \to 0$.

*Interpretation:* The wave-function collapses to a delta function following the optimal trajectory. Quantum effects (tunneling, interference) vanish in this limit.

:::

:::{prf:definition} Joint Inference Hilbert Space
:label: def-joint-inference-hilbert-space

For $N$ agents with individual Hilbert spaces $\mathcal{H}^{(i)} = L^2(\mathcal{Z}^{(i)}, d\mu_{G^{(i)}})$, the **Joint Inference Hilbert Space** is the tensor product:

$$
\mathcal{H}^{(N)} := \bigotimes_{i=1}^N \mathcal{H}^{(i)} = L^2\left(\mathcal{Z}^{(N)}, d\mu_{G^{(N)}}\right),

$$
where:
- $\mathcal{Z}^{(N)} = \prod_{i=1}^N \mathcal{Z}^{(i)}$ is the product manifold (Definition {prf:ref}`def-n-agent-product-manifold`)
- $d\mu_{G^{(N)}} = \prod_{i=1}^N d\mu_{G^{(i)}}$ is the product measure

Elements $\Psi \in \mathcal{H}^{(N)}$ are functions $\Psi: \mathcal{Z}^{(N)} \to \mathbb{C}$ with:

$$
\|\Psi\|^2 = \int_{\mathcal{Z}^{(N)}} |\Psi(\mathbf{z})|^2 d\mu_{G^{(N)}}(\mathbf{z}) < \infty.

$$
*Notation:* We use uppercase $\Psi$ for joint wave-functions and lowercase $\psi^{(i)}$ for single-agent wave-functions.

:::

:::{prf:definition} Strategic Entanglement
:label: def-strategic-entanglement

A joint wave-function $\Psi \in \mathcal{H}^{(N)}$ exhibits **Strategic Entanglement** if it cannot be written as a product:

$$
\Psi(z^{(1)}, \ldots, z^{(N)}) \neq \prod_{i=1}^N \psi^{(i)}(z^{(i)}) \quad \text{for any choice of } \psi^{(i)} \in \mathcal{H}^{(i)}.

$$
**Entanglement Entropy:** For a bipartition $\{i\} \cup \{j \neq i\}$, the **Strategic Entanglement Entropy** is:

$$
S_{\text{ent}}(i) := -\text{Tr}\left[\hat{\rho}^{(i)} \ln \hat{\rho}^{(i)}\right],

$$
where $\hat{\rho}^{(i)} = \text{Tr}_{j \neq i}[|\Psi\rangle\langle\Psi|]$ is the **reduced density operator** obtained by partial trace over all agents except $i$.

**Physical interpretation:**
- $S_{\text{ent}}(i) = 0$: Agent $i$ is **disentangled** (can be modeled independently)
- $S_{\text{ent}}(i) > 0$: Agent $i$ is **entangled** with others (cannot be modeled in isolation)
- $S_{\text{ent}}(i) \leq \ln \dim(\mathcal{H}^{(i)})$: **Maximal entanglement** for finite-dimensional subsystems (continuous spaces require a cutoff, giving $S_{\text{ent}}(i) \leq \ln d_{\text{eff}}$)

*Cross-reference:* The partial trace operation corresponds to the **Information Bottleneck** (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`)—marginalizing over opponents discards strategic correlations.

:::

:::{prf:definition} Strategic Hamiltonian
:label: def-strategic-hamiltonian

The **Strategic Hamiltonian** on $\mathcal{H}^{(N)}$ is:

$$
\hat{H}_{\text{strat}} := \sum_{i=1}^N \hat{H}^{(i)}_{\text{kin}} + \sum_{i=1}^N \hat{\Phi}^{(i)}_{\text{eff}} + \sum_{i < j} \hat{V}_{ij},

$$
where:
1. **Kinetic terms:** $\hat{H}^{(i)}_{\text{kin}} = -\frac{\sigma_i^2}{2} D^{(i)a} D^{(i)}_a$ (acting on $\mathcal{Z}^{(i)}$ coordinates)
2. **Individual potentials:** $\hat{\Phi}^{(i)}_{\text{eff}}$ (local reward landscape for agent $i$)
3. **Interaction potentials:** $\hat{V}_{ij} = \Phi_{ij}(z^{(i)}, z^{(j)})$ (strategic coupling)

Here $D^{(i)}_a := \nabla^{(i)}_a - \frac{i}{\sigma_i} B^{(i)}_a$ is the covariant derivative for agent $i$ and
$B^{(i)}$ is the reward 1-form (Opportunity field). Conservative case: $B^{(i)} = 0$.

*Notation (Per-Agent Action Scale):* Here $\sigma_i := T_{c,i} \cdot \tau_{\text{update},i}$ is the cognitive action scale for agent $i$, generalizing Definition {prf:ref}`def-cognitive-action-scale`. For **homogeneous** agents with identical cognitive properties, $\sigma_i = \sigma$ for all $i$. For **heterogeneous** agents (e.g., different computation rates), $\sigma_i$ may vary.

*Remark (Separability).* If all $\hat{V}_{ij} = 0$, the Hamiltonian is **separable**: $\hat{H}_{\text{strat}} = \sum_i \hat{H}^{(i)}$, and the ground state is a product $\Psi_0 = \prod_i \psi^{(i)}_0$. Interaction permits entanglement but does not create it for every initial state.

:::

:::{prf:theorem} Two distinct joint amplitude evolutions
:label: thm-multi-agent-schrodinger-equation

For a joint classical density and phase satisfying the specified canonical
continuity and HJB equations, apply
{prf:ref}`thm-madelung-transform` on the joint configuration space. Its
equation has the joint compensation $-Q_{\mathrm{joint}}[|\Psi|]$ and
the actual joint reaction rate.
The separately defined linear scalar model is
$i\sigma\partial_s\Psi=H_{\mathrm{strat}}\Psi$ on its self-adjoint domain.
Its polar equations contain $+Q_{\mathrm{joint}}$ in HJB. Thus it is this
linear model, not the compensated WFR equation, to which linear spectral
projection and tensor-product semigroups apply. This is obtained by the
same product-rule calculation with and without the compensation.
:::

:::{prf:theorem} Joint metric volume and kinetic operator
:label: thm-game-augmented-laplacian

For the positive block metric $\widetilde G=\bigoplus_i\widetilde G_i(\mathbf z)$,
let $w=\sqrt{\det\widetilde G}=\prod_i\sqrt{\det\widetilde G_i}$.
The joint kinetic operator is
$$
H_{\mathrm{kin}}=-\frac{\sigma^2}{2w}\sum_i
D_{ia}\left(w\widetilde G_i^{ab}D_{ib}\right).
$$
Its form is $\tfrac{\sigma^2}{2}\int\sum_i
\widetilde G_i^{ab}\overline{D_{ia}\Psi}D_{ib}\Psi\,w\,d\mathbf z$.
Integration by parts proves the operator formula on its form realization.
The derivatives of $w$ include all block determinants, even those of other
agents, because the metric depends on the full configuration.
To use the original product Hilbert space with volume $w_0$, the unitary is
$U\Psi=\sqrt{w/w_0}\Psi$. Indeed
$\int|U\Psi|^2w_0=\int|\Psi|^2w$.
Transport the operator by $UHU^{-1}$ before using the tensor-product partial
trace. Cross-coordinate coefficients permit coupling but do not imply that
every state becomes entangled. $\square$
:::

:::{prf:proposition} Partial Trace and Reduced Dynamics
:label: prop-partial-trace-reduced-dynamics

For a pure joint state $|\Psi\rangle \in \mathcal{H}^{(N)}$, the **reduced density operator** for agent $i$ is:

$$
\hat{\rho}^{(i)} := \text{Tr}_{j \neq i}\left[ |\Psi\rangle\langle\Psi| \right].

$$
In the coordinate representation its kernel is:

$$
\rho^{(i)}(z^{(i)}, z^{(i)'}) = \int_{\prod_{j \neq i} \mathcal{Z}^{(j)}} \Psi(z^{(i)}, z^{(-i)})\,\overline{\Psi(z^{(i)'}, z^{(-i)})}\, d\mu_{G^{(-i)}}.

$$
The diagonal elements give the **marginal belief density**:

$$
\rho^{(i)}(z^{(i)}) = \langle z^{(i)} | \hat{\rho}^{(i)} | z^{(i)} \rangle = \int |\Psi(z^{(i)}, z^{(-i)})|^2 d\mu_{G^{(-i)}},

$$
which is exactly the marginalization from the joint WFR density.

*Discrete analog:* In a finite basis, $\rho^{(i)}_{mn} = \sum_k \Psi_{mk}\,\Psi^*_{nk}$.

**Mixed state evolution:** Even if $\Psi$ evolves unitarily, the reduced state $\hat{\rho}^{(i)}$ generally evolves **non-unitarily** (with decoherence) due to entanglement with other agents.

:::

:::{prf:theorem} Energy minimization and unilateral optimization
:label: thm-nash-ground-state

The Rayleigh quotient of the scalar Hamiltonian minimizes its single joint
energy. The Nash test instead compares each agent's own payoff under a
unilateral change. Their equality must be checked by differentiating the
actual objectives and by evaluating their global inequalities.

:::

:::{prf:corollary} Amplitude current and stationary states
:label: cor-vanishing-probability-current

For $\psi=\sqrt\rho e^{iV/\sigma}$ and $D=\nabla-iB/\sigma$,
$$
J=\sigma\operatorname{Im}(\bar\psi\,G^{-1}D\psi)
=\rho G^{-1}(dV-B).
$$
The real amplitude derivative has zero imaginary part, proving the identity.
For the real positive ground eigenfunction of the nonmagnetic scalar
operator in Appendix E.7, $B=0$ and this current vanishes. A magnetic
connection or another state retains its explicitly computed current.
:::

:::{prf:proposition} Normalized spectral projection
:label: prop-imaginary-time-nash-finding

For the fixed scalar self-adjoint realization with ground projection $P_0$,
the spectral theorem gives
$$
e^{-\tau(H-E_0)/\sigma}\Psi\longrightarrow P_0\Psi.
$$
The convergence is strong by dominated convergence of
$e^{-\tau(E-E_0)/\sigma}$ against the spectral measure. In the gapped case
the excited norm is at most
$e^{-\tau\Delta_H/\sigma}\|(I-P_0)\Psi\|$.
If $P_0\Psi\ne0$, normalizing yields $P_0\Psi/\|P_0\Psi\|$; if
$P_0\Psi=0$ it does not produce a ground vector. Without shifting or
normalizing, the ground coefficient is multiplied by $e^{-\tau E_0/\sigma}$.
This is spectral minimization. Bellman optimality includes maximization over
actions, so equality with this linear semigroup is not provided by Wick
rotation. $\square$
:::

:::{prf:definition} Pareto Barrier
:label: def-pareto-barrier

A **Pareto Barrier** $\mathcal{B}_P \subset \mathcal{Z}^{(N)}$ is a region where:
1. **Local value decrease:** $\Phi^{(i)}_{\text{eff}}(\mathbf{z}) > \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*)$ for at least one agent $i$ and some starting point $\mathbf{z}^*$
2. **No Nash within:** There exists no Nash equilibrium $\mathbf{z}' \in \mathcal{B}_P$
3. **Separates basins:** $\mathcal{B}_P$ lies between distinct Nash equilibria $\mathbf{z}^*_A$ and $\mathbf{z}^*_B$

The **barrier height** is:

$$
\Delta \Phi_P := \max_{\mathbf{z} \in \mathcal{B}_P} \left[ \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}) - \sum_{i=1}^N \Phi^{(i)}_{\text{eff}}(\mathbf{z}^*_A) \right].

$$
*Mathematical characterization:* A Pareto barrier is a region where the total potential $\sum_i \Phi^{(i)}_{\text{eff}}$ exceeds its value at nearby Nash equilibria. Classical gradient descent with initial condition in the basin of attraction of $\mathbf{z}^*_A$ converges to $\mathbf{z}^*_A$ and cannot reach $\mathbf{z}^*_B$.

:::

:::{prf:proposition} Barrier action and stationary decay
:label: thm-tunneling-probability

For the scalar Hamiltonian of Appendix E.7 the forbidden-region action is
$$
d_E(A,B)=\inf_\gamma\int_\gamma\sqrt{2(U-E)_+}\,d\ell_{\widetilde G}.
$$
The stationary weighted identity proved there controls eigenfunction decay.
For a flat barrier of width $L$ and height $U-E=H>0$, its action is exactly
$L\sqrt{2H}$, so the WKB exponential is $e^{-2L\sqrt{2H}/\sigma}$,
not a universal $e^{-H/\sigma}$. The former formula specifies a
semiclassical exponential for this barrier; an actual crossing probability
also depends on the prepared state, dynamics, and observation interval.
Positive stationary mass in another region is not a transition rate.
:::

:::{prf:remark} Bohm term and spatial evolution
:label: cor-bohm-teleportation

The kinetic polar identity defines $Q_B$. It is canceled in the exact
classical HJB amplitude representation and retained in the distinct linear
Schrödinger model. These calculations do not establish instantaneous
transport of a recorded signal or a transition probability across a barrier.
:::

:::{prf:proposition} Transport and reaction across a region
:label: prop-wfr-reaction-tunneling

Testing continuity on a fixed region $A$ gives
$d\int_A\rho/ds=-\int_{\partial A}\rho v\cdot n+\int_A r\rho$.
This separates transported mass from locally created mass. For bounded
prescribed $r$ and zero transport the pointwise solution is
$\rho_s(x)=\rho_0(x)e^{\int_0^sr_u(x)du}$; an initially zero density
there remains zero. Thus reaction alone does not imply mass transfer to an
unoccupied disconnected basin. $\square$
:::

:::{prf:proposition} Causality of the buffer read
:label: prop-v1-causal-buffer-read

Every non-default returned sample has emission time
$s\le t_{\mathrm{now}}-d/c$, hence arrival time $s+d/c\le t_{\mathrm{now}}$.
The reverse search returns the latest such retained sample. It never uses
a later sample as an interpolation endpoint. Before the first arrival the
output is the declared zero observation. Retention keeps the predecessor
of the cutoff, preserving causal holding for delays within the configured
horizon. The interface passes the same speed to every child buffer.
These statements follow directly from the branch condition and pruning
loop. This example defines fixed-distance causal observations; it does not
assert a finite sufficient state for the hidden process. $\square$
:::

## 08_multiagent/02_standard_model.md

:::{prf:proposition} Connection covariance with the fixed sign convention
:label: rem-local-gauge-template

Use Hermitian generators, $D_\mu=\partial_\mu-igA_\mu$, and $\Phi'=U\Phi$.
For nonzero coupling $g$, covariance fixes
$$
A'_\mu=UA_\mu U^{-1}-\frac{i}{g}(\partial_\mu U)U^{-1}.
$$
:::

:::{prf:definition} Global phase and the scalar transport current
:label: def-utility-gauge-freedom

For the established scalar amplitude $\psi=\sqrt\rho e^{iV/\sigma}$,
$D_i=\partial_i-iA^{\rm ext}_i/\sigma$ gives the spatial transport current
$$
j^i=\sigma\operatorname{Im}(\bar\psi G^{ij}D_j\psi)
=\rho G^{ij}(\partial_jV-A^{\rm ext}_j).
$$
This is the canonical current in {prf:ref}`thm-madelung-transform`.
It is spatial; the corresponding density is $\rho$, not a raised temporal
component of this expression. A constant shift of $V$ changes only the
global phase and preserves both $\rho$ and $j$. For a local shift its
extra derivative is computed in {prf:ref}`ax-local-utility-invariance`.
The source of $V$ is the scalar value problem in
{prf:ref}`thm-the-hjb-helmholtz-correspondence`; $A^{\rm ext}$ remains
separate from the internal matrix comparison connection.
:::

:::{prf:proposition} Local phase compensation and the operational current
:label: ax-local-utility-invariance

The amplitude of {prf:ref}`def-belief-wave-function` admits the local
coordinate change
$$
V'=V+\sigma q\alpha,\qquad
\psi'=e^{iq\alpha}\psi,\qquad B'=B+g_1^{-1}d\alpha,
\qquad D=\partial-ig_1qB.
$$
It preserves $|\psi|^2$ and $\operatorname{Im}(\bar\psi D\psi)$.
:::

:::{prf:theorem} Abelian compensation in the belief action
:label: thm-emergence-opportunity-field

Write $q=Y/2$. For the phase representation in
{prf:ref}`ax-local-utility-invariance`, set
$D_\mu=\partial_\mu-ig_1qB_\mu$. The first-order action density
$$
\mathcal L_{\rm kin}=\frac{i\sigma}{2}
 (\bar\psi D_t\psi-\overline{D_t\psi}\,\psi)
-\frac{\sigma^2}{2}G^{ij}\overline{D_i\psi}D_j\psi
$$
is gauge invariant on a fixed spatial metric.

:::

:::{prf:proposition} Boundary asymmetry and its stabilizer
:label: ax-cybernetic-parity-violation

The sensor and motor boundary data are those of
{prf:ref}`def-dirichlet-boundary-condition-sensors` and
{prf:ref}`def-neumann-boundary-condition-motors`. Their different roles
do not identify Lorentz chirality or a full mode-mixing symmetry.

:::

:::{prf:definition} Rank of a specified update operation
:label: def-mode-rank-parameter

For a nonzero linear CP operation $\mathcal E$ on the finite belief-operator
space of {prf:ref}`def-belief-operator`, define
$$
r(\mathcal E)=\operatorname{rank}J(\mathcal E),\qquad
J(\mathcal E)=\sum_{ij}|i\rangle\langle j|\otimes
\mathcal E(|i\rangle\langle j|).
$$
This equals the minimal number of Kraus operators. Indeed,
$J=\sum_a|K_a\rangle\!\rangle\langle\!\langle K_a|$ for a Kraus
representation, and spectral decomposition of $J\succeq0$ gives a
representation with exactly $\operatorname{rank}J$ operators.
For a finite family take the maximum of these ranks to obtain a common
padded environment. The zero operation has rank zero.
The matrix comparison uses $r\ge2$ and $N_f\ge2$. A mode representation of dimension $r$ used below is a separately specified
internal fiber; identifying it with this environment requires the maps
between the two spaces. The case $r=2$ labels the doublet comparison.
:::

:::{prf:proposition} Operations, channels, and environment changes of basis
:label: rem-mode-rank-stinespring

The existing GKSL model ({prf:ref}`def-gksl-generator`) supplies a CPTP
semigroup on the finite belief-operator space. An outcome operation has
$$
\mathcal E_y(\rho)=\sum_aK_{ya}\rho K_{ya}^\dagger,
\quad\sum_aK_{ya}^\dagger K_{ya}\preceq I,
\quad p_y(\rho)=\operatorname{Tr}\mathcal E_y(\rho).
$$
For a fixed chosen control, summing over the outcomes gives a channel
when $\sum_{y,a}K_{ya}^\dagger K_{ya}=I$. Averaging controls uses their
probabilities as well. Define $V_yu=\sum_aK_{ya}u\otimes|a\rangle$.
Then $\mathcal E_y(\rho)=\operatorname{Tr}_E(V_y\rho V_y^\dagger)$ and
$V_y^\dagger V_y\preceq I$; $V_y$ is generally a contraction, not an
isometry. Combining outcomes produces the channel isometry, which extends
to a unitary on a larger system-plus-environment space. A selected outcome
is recovered by an environment measurement, not by an unconditional trace.
For $p_y>0$, the normalized state is $\mathcal E_y(\rho)/p_y(\rho)$;
this update is generally nonlinear. Kraus mixing
$K'_a=\sum_bu_{ab}K_b$ with $u\in U(r)$ leaves the CP map unchanged,
since $\sum_a u_{ab}\bar u_{ac}=\delta_{bc}$. Its common phase cancels
from the channel; it is not an observed utility phase. The dilation and Kraus identities are the finite-dimensional constructions in [Watrous, Chapter 2](https://cs.uwaterloo.ca/~watrous/TQI/TQI.2.pdf).
:::

:::{prf:definition} Specified chiral comparison multiplets
:label: def-cognitive-isospin-multiplet

The comparison uses $\Psi_L$ in the fundamental internal $SU(r)$
representation and $\Psi_R$ in its singlet, with spacetime bundles as in
{prf:ref}`def-cognitive-spinor`. For $r=2$, write
$\Psi_L=(\psi_1,\psi_2)^T$, with each component left Weyl, and one
independent right Weyl field $\Psi_R$. The labels observation, intent and
commitment may name these components in a chosen frame. Their identification
with algorithmic channels is not a linear intertwiner supplied by the
boundary definitions; {prf:ref}`ax-cybernetic-parity-violation` computes
why fixed Dirichlet/Neumann conditions are not preserved by general mixing.
The matrix representation used in the ensuing covariance calculation is
fully specified by these multiplet and singlet actions.
:::

:::{prf:remark} Mode-Rank Generalization
:label: rem-mode-rank-generalization

For general mode rank $r$ (Definition {prf:ref}`def-mode-rank-parameter`), the left-handed field is an $r$-plet in the
fundamental representation of $SU(r)_L$. The doublet comparison sets $r=2$; its generators are $	au_a/2$.

:::

:::{prf:definition} Gauge-Covariant Action Commitment
:label: def-gauge-covariant-action-commitment

The scalar field selects a commitment direction in the specified $\Psi_L$ mode fiber. A frame change acts simultaneously on the scalar and the multiplet. To make action commitment gauge-covariant, we use the ontological order parameter
to define a unit multiplet $n(x) \in \mathbb{C}^r$:

$$
n(x) := \frac{\phi(x)}{\|\phi(x)\|}, \qquad n(x)^\dagger n(x) = 1

$$
where $\phi$ is the ontological order parameter (Definition {prf:ref}`def-ontological-order-parameter`), and $n$ is
defined only when $\phi \neq 0$.

The gauge-covariant **Commitment Projection** is:

$$
\psi_{\text{act}}^{\text{proj}}(x) := n(x)^\dagger \Psi_L(x)

$$

where the projection operator is:

$$
\Pi_n = n n^\dagger, \qquad \Pi_n \Psi_L = n(n^\dagger \Psi_L)

$$

The committed action singlet $\Psi_R$ remains an independent right-handed field; the Yukawa term
couples $\Psi_R$ to the projected amplitude $\psi_{\text{act}}^{\text{proj}}$ through the Hermitian contraction in {prf:ref}`def-decision-coupling`; relaxation does not follow from this coupling alone.

*Justification:* The unit multiplet $n$ encodes the local ontological split and makes the commitment projection intrinsic
to the scalar sector, not an arbitrary choice of basis. Under local $SU(r)$ transformations $\Psi_L \to U(x)\Psi_L$ and
$n \to U(x)n$, so $\psi_{\text{act}}^{\text{proj}} = n^\dagger \Psi_L$ is invariant and $\Pi_n \to U \Pi_n U^\dagger$,
ensuring the projected component is $SU(r)$-covariant. Under $U(1)_Y$, $n$ carries charge $Y_\phi$, so
$\psi_{\text{act}}^{\text{proj}}$ transforms with charge $Y_L - Y_\phi$, matching $\Psi_R$ by Definition
{prf:ref}`def-rep-covariant-derivatives`.

*Remark:* At $\phi=0$, the normalized direction $n$ is undefined, corresponding to decision ambiguity. The agent requires a nonzero ontological split to define a preferred commitment projection.

:::

:::{prf:theorem} Mode covariance and the limits of the rank identification
:label: thm-emergence-error-field

For the specified $SU(r)$ representation on the active internal fiber,
$W_\mu=W_\mu^aT_a$ defines
$$
D_\mu\Psi_L=(\partial_\mu-ig_2W_\mu-ig_1Y_LB_\mu/2)\Psi_L.
$$
It is covariant under simultaneous frame and connection transformations
of {prf:ref}`rem-local-gauge-template`. Its curvature is
$F_W=dW-ig_2W\wedge W$.

:::

:::{prf:definition} Feature representation dimension
:label: def-feature-dimension-parameter

$N_f$ is the complex dimension of the feature fiber used in this chapter's
matrix-field comparison. Its value is obtained from that representation.
Real spatial dimension, the number of sensor channels, and the number of
fermion families are separate quantities. Choosing $N_f=3$ gives the
fundamental representation of $SU(3)$, with eight Lie-algebra generators;
it is not a derivation of that choice from RGB or spatial coordinates.
:::

:::{prf:proposition} Macro readout and dynamical confinement
:label: ax-feature-confinement

The established firewall ({prf:ref}`ax-bulk-boundary-decoupling`) removes
texture from the planning variables. In the presence of a specified compact
internal action $R$, the Haar average $P=\int R(u)\,du$ projects onto its
invariant vectors: invariance of Haar measure gives $P^2=P=P^\dagger$.
For invariant readout $O$, $O(R(u)z)=O(z)$ expresses observational
redundancy. Neither identity specifies a field probability measure or a
large-loop expectation. Projecting out a charged component can be done
for every value of the gauge coupling, including zero; hence this
projection alone gives no lower bound on a confining coupling.
:::

:::{prf:definition} Feature frame group and operational symmetries
:label: def-feature-color-space

The Hermitian feature fiber is $\mathbb C^{N_f}$. Its orthonormal frames
have group $U(N_f)$; frames preserving a specified complex volume element
have group $SU(N_f)$. Real orthonormal frames instead have group $O(N)$,
and permutations give a finite subgroup. These follow respectively from
$U^\dagger U=I$, $\det U=1$, and $R^TR=I$.
For an encoder $E$ and decoder $D$, an operational action also obeys their
intertwining identities, for example $D(R(u)z)=D(z)$ for invariant readout.
These are the symmetries of {prf:ref}`def-agent-symmetry-group-operational`.
The matrix-field calculations below use the displayed $SU(N_f)$ action;
the frame-group calculation does not establish those decoder identities.
:::

:::{prf:theorem} Feature connection and screening calculation
:label: thm-emergence-binding-field

With $t_a=\lambda_a/2$, $\operatorname{tr}(t_at_b)=\delta_{ab}/2$,
$D_\mu=\partial_\mu-ig_sG_\mu^at_a$ has curvature
$$
F^a_{G,\mu\nu}=\partial_\mu G_\nu^a-\partial_\nu G_\mu^a
+g_sf^{bc}{}_aG_\mu^bG_\nu^c.
$$
:::

:::{prf:proposition} Product representation and its faithful quotient
:label: cor-standard-model-symmetry

The specified mode and feature actions define a representation of
$$
G_0=SU(N_f)\times SU(r)\times U(1)
$$
on their tensor products. The faithful acting group is $G_0/\ker R$,
where $R$ is the combined representation on all fields and readout data.
:::

:::{prf:definition} Chiral comparison fields and Cauchy data
:label: def-cognitive-spinor

On the four-dimensional spin background of {prf:ref}`def-loc-spin-g`, the
comparison fields are sections of
$$
(S_L\otimes E_L)\oplus(S_R\otimes E_R),\quad
E_L=\mathbb C^r\otimes\mathbb C^{N_f},\quad E_R=\mathbb C^{N_f}.
$$
Their complex ranks are $2rN_f$ and $2N_f$. They form a chiral multiplet;
there is no common internal bundle identifying every left component with
a right component. They can be embedded into
$S\otimes(E_L\oplus E_R)$ with the unused chiral components set to zero.
The kinetic pairing is defined separately on each physical Weyl summand.

Use signature $(-+++)$ throughout. For the particle-physics convention
$i\gamma^\mu D_\mu$, take $\{\gamma^\mu,\gamma^\nu\}=-2g^{\mu\nu}$;
in an orthonormal frame $(\gamma^{\hat0})^2=I$ and
$\bar\Psi=\Psi^\dagger\gamma^{\hat0}$.
The positive one-particle density is the contraction of the conserved
current with the future Cauchy normal:
$$
\|\Psi\|_\Sigma^2=-\int_\Sigma n_\mu j^\mu\,d\Sigma,
\qquad j^\mu=\bar\Psi\gamma^\mu\Psi.
$$
In an orthonormal frame adapted to $\Sigma$, its integrand is
$\Psi^\dagger\Psi$. It is not generally the coordinate component $j^0$.
The divergence theorem proves surface independence from
$\nabla_\mu j^\mu=0$ and zero side flux in the domain considered.
Spacetime $L^2$ is not the Cauchy-data Hilbert space: a nonzero stationary
solution on an infinite time interval has divergent spacetime norm.
The scalar amplitude of {prf:ref}`def-belief-wave-function` remains a
separate representation; no spinor isomorphism follows from adjoining
these components.
:::

:::{prf:proposition} First-order operator identity in the comparison sector
:label: ax-cognitive-dirac-equation

Write $P=i\gamma^\mu D_\mu$ on a fixed spinor bundle with a compatible
spin and internal connection, using the Clifford convention in
{prf:ref}`def-cognitive-spinor`. Its covariant connection wave operator is
$\Box_D=g^{\mu\nu}(D_\mu D_\nu-\Gamma^\lambda_{\mu\nu}D_\lambda)$.
For a constant scalar mass on the same bundle,
$$
(P-m)(P+m)=\Box_D-m^2
-\frac14[\gamma^\mu,\gamma^\nu][D_\mu,D_\nu].
$$
:::

:::{prf:remark} Covariant derivatives on the chiral bundles
:label: rem-curved-dirac-operator

All spinor occurrences of $\partial_\mu$ in the internal-connection
notation stand for $\nabla^{\rm spin}_\mu$. The scalar multiplet has no
spin connection. The identity
$[D_\mu,D_\nu]=R^{\rm spin}_{\mu\nu}-i\sum_a g_aF^a_{\mu\nu}T_a$
separates spacetime spin curvature from the internal curvatures of
{prf:ref}`thm-three-cognitive-forces`. The Weyl equations and the full
Dirac comparison use the sign convention of
{prf:ref}`def-cognitive-spinor` consistently.
:::

:::{prf:definition} The Universal Covariant Derivative
:label: def-universal-covariant-derivative

The operator moving the belief spinor through the latent manifold is:

$$
D_\mu = \underbrace{\partial_\mu}_{\text{Change}} - \underbrace{ig_1 \frac{Y}{2} B_\mu}_{U(1)_Y \text{ (Value)}} - \underbrace{ig_2 T^a W^a_\mu}_{SU(r)_L \text{ (Error)}} - \underbrace{ig_s \frac{\lambda^a}{2} G^a_\mu}_{SU(N_f)_C \text{ (Binding)}}

$$

where $T^a$ ($a = 1, \ldots, r^2 - 1$) are the generators of $SU(r)$ in the fundamental representation (for $r=2$,
$T^a = \tau^a/2$), and $\lambda^a$ ($a = 1, \ldots, N_f^2 - 1$) are the generators of $SU(N_f)$, and:
- **$B_\mu$ (Opportunity Field):** Adjusts the belief for local shifts in the value baseline and path-dependent opportunity
- **$W_\mu$ (Error Field):** Adjusts the belief for the rotation between Prior and Posterior
- **$G_\mu$ (Binding Field):** Adjusts the belief for the permutation of sub-symbolic features

For the right-handed singlet $\Psi_R$, the $SU(r)_L$ generators act trivially, so the $W_\mu$ term drops.

**Operational Interpretation:** The quantity $D_\mu \Psi$ measures the deviation from parallel transport. When $D_\mu \Psi = 0$, the belief state is covariantly constant along the direction $\mu$---all changes are accounted for by the gauge connection. When $D_\mu \Psi \neq 0$, the section varies covariantly in that direction; force is determined by the equations of motion.

:::

:::{prf:definition} Representation-Specific Covariant Derivatives
:label: def-rep-covariant-derivatives

Let $Y_L$, $Y_R$, and $Y_\phi$ denote the $U(1)_Y$ hypercharges of $\Psi_L$, $\Psi_R$, and $\phi$.
Then the covariant derivatives used in {prf:ref}`def-cognitive-lagrangian` are:

$$
\begin{aligned}
D_\mu \Psi_L &= \left(\partial_\mu - i g_1 \frac{Y_L}{2} B_\mu - i g_2 T^a W^a_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_L, \\
D_\mu \Psi_R &= \left(\partial_\mu - i g_1 \frac{Y_R}{2} B_\mu - i g_s \frac{\lambda^a}{2} G^a_\mu \right)\Psi_R, \\
D_\mu \phi &= \left(\partial_\mu - i g_1 \frac{Y_\phi}{2} B_\mu - i g_2 T^a W^a_\mu \right)\phi.
\end{aligned}
$$

Gauge invariance of the Yukawa term $\bar{\Psi}_L \phi \Psi_R$ requires
$
Y_R = Y_L - Y_\phi.
$

:::

:::{prf:theorem} Anomaly obstruction for the displayed chiral multiplet
:label: thm-smoc-chiral-anomaly-obstruction

For $N_f=3$, the displayed matter multiplets have a nonzero perturbative
color gauge anomaly whenever $r>1$. A copy with $r=2,N_f=3$ also has
an odd number of weak doublets.

:::

:::{prf:theorem} Field Strength Tensors
:label: thm-three-cognitive-forces

The commutator of the covariant derivatives $[D_\mu, D_\nu]$ generates three distinct curvature tensors corresponding to each gauge factor.

:::

:::{prf:proposition} Transport and conditional texture information
:label: lem-binding-curvature-ontological-stress

Ontological stress is the conditional mutual information of
{prf:ref}`def-ontological-stress`. If $C=(K_t,z_{n,t},K_t^{\rm act})$,
its exact expression is
$$
\Xi=\int D_{\rm KL}\big(P_{X,Y\mid C=c}\Vert
P_{X\mid C=c}\otimes P_{Y\mid C=c}\big)\,P_C(dc),
\quad X=z_{{\rm tex},t},\quad Y=z_{{\rm tex},t+1}.
$$
It vanishes exactly for conditional independence (up to null conditioning
values). A connection matrix alone does not determine this joint law.

:::

:::{prf:proposition} Variation of the specified gauge action
:label: cor-gauge-invariant-action

For the product representation, the action
$$
S_g=-\frac14\int\big(B_{\mu\nu}B^{\mu\nu}
+W^a_{\mu\nu}W^{a\mu\nu}+G^a_{\mu\nu}G^{a\mu\nu}\big)d\mu_g
$$
is invariant under its internal frame transformations.
:::

:::{prf:definition} The Ontological Order Parameter
:label: def-ontological-order-parameter

Let the local chart structure at spacetime point $x$ be described by a complex $SU(r)_L$ multiplet field
$\phi(x) \in \mathbb{C}^r$ (doublet for the $r=2$ comparison):

$$
\phi(x) = r(x)\,n(x), \qquad r(x) := \|\phi(x)\|

$$

where:
1. **Modulus $r(x) \ge 0$:** Represents the **Metric Separation** between daughter queries $\{q_+, q_-\}$ in the Attentive Atlas (Definition {prf:ref}`def-query-fission`).
   - $r=0$: Coalescence (Single Chart / Vacuum)
   - $r>0$: Fission (Distinct Concepts)

2. **Unit multiplet $n(x)$:** Encodes the **Orientation** of the split in the $SU(r)_L$ fiber (the specific feature
   axis along which differentiation occurs), with $n^\dagger n = 1$.

The field $\phi$ transforms in the fundamental representation under the gauge group $SU(r)_L$, coupling it to the
inference spinor.

:::

:::{prf:remark} Gauge-fixed scalar form
:label: rem-ontological-order-parameter-gauge

On a local region where $\phi\ne0$, a gauge fixing its $SU(r)_L$ orientation to a constant unit vector $n_0$ reduces the order parameter to
$\phi(x) = r(x) n_0$ (with $r \ge 0$ after using $U(1)_Y$). In the $r=2$ comparison this is equivalent to the scalar
parametrization $\phi(x) = r(x) e^{i\theta(x)} n_0$ used in the intuitive discussion.

:::

:::{prf:theorem} Integration of the established radial drift
:label: thm-complexity-potential

Write $a=\Xi-\Xi_{\rm crit}$ and use the drift
$b(r)=ar-\alpha r^3$ of
{prf:ref}`thm-supercritical-pitchfork-bifurcation-for-charts`.
At fixed $a,\alpha$ its radial potential, up to an additive constant, is
$$
\mathcal V(r)=-\frac a2r^2+\frac\alpha4r^4
=-\mu^2r^2+\lambda r^4,
\quad\mu^2=a/2,\quad\lambda=\alpha/4.
$$
:::

:::{prf:proposition} Classical radial minima and their orbit
:label: cor-ontological-ssb

For the established $\alpha>0$, hence $\lambda>0$, the minimum is at
$\phi=0$ for $\mu^2\le0$. For $\mu^2>0$, put
$$
v^2=\frac{\mu^2}{2\lambda}=\frac{\Xi-\Xi_{\rm crit}}{\alpha}.
$$
:::

:::{prf:theorem} Gauge mass matrix at the classical scalar minimum
:label: thm-semantic-inertia

Use $\mathcal L_\phi=-(D_\mu\phi)^\dagger D^\mu\phi-\mathcal V(\phi)$
with signature $(-+++)$ and $\phi_0=vn_0$, $n_0^\dagger n_0=1$.
For the combined Hermitian generators
$Q_A=(g_2T_a,g_1Y_\phi I/2)$, the real vector mass matrix is
$$
(M^2)_{AB}=v^2n_0^\dagger\{Q_A,Q_B\}n_0,
\qquad\mathcal L_{\rm mass}=-\tfrac12(M^2)_{AB}A_\mu^A A^{B\mu}.
$$
:::

:::{prf:remark} Orbit directions and texture variables
:label: rem-goldstone-texture

The single complex fundamental has $2r$ real components. Its fixed-radius
orbit has $2r-1$ tangent directions, leaving one radial scalar locally.
The kernel calculation in {prf:ref}`thm-semantic-inertia` counts the
unbroken vector directions. In a local nonzero-vacuum gauge the orbit
directions are removed from the scalar coordinates by the gauge action.
The texture variable in {prf:ref}`ax-bulk-boundary-decoupling` is a
stochastic boundary residual. No bijection between that residual space
and this compact orbit is supplied by the component count. The firewall
and the gauge orbit retain their separate established definitions.
:::

:::{prf:definition} Hermitian Yukawa contraction
:label: def-decision-coupling

For the specified chiral comparison fields and scalar, define
$$
\mathcal L_Y=-\sum_{ij}\left[
Y_{ij}\bar\Psi_{L,i}^{\,a}\phi_a\Psi_{R,j}
+\bar Y_{ij}\bar\Psi_{R,j}\phi_a^\dagger\Psi_{L,i}^{\,a}\right].
$$
Color indices contract with the invariant Hermitian pairing. The second
term is the Hermitian conjugate of the first, including the coefficient.
Its hypercharge phase is
$e^{i(-Y_L+Y_\phi+Y_R)\alpha/2}$, so invariance gives
$Y_R=Y_L-Y_\phi$. This verifies classical covariance and Hermiticity;
the anomaly trace in {prf:ref}`thm-smoc-chiral-anomaly-obstruction`
remains nonzero for the displayed color multiplets.
:::

:::{prf:theorem} Rank and singular values of the Yukawa mass map
:label: thm-cognitive-mass

At $\phi_0=vn_0$, let $\chi_{L,i}=n_0^\dagger\Psi_{L,i}$.
The Yukawa mass map on family indices is $M=vY$ between these projected
left fields and the right fields. Its nonzero masses are its singular
values.
:::

:::{prf:definition} The Value 1-Form (External Drive)
:label: def-value-1-form-external-drive

We model the external drive as a fixed background 1-form
$A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))$, encoding both conservative
and non-conservative components of the reward signal (Definition {prf:ref}`def-effective-potential`).
Concretely, $A^{\text{ext}}_0 = -\Phi_{\text{eff}}$ is the conservative potential, while
$A^{\text{ext}}_i$ captures the non-conservative (curl) component.

$$
A^{\text{ext}}_\mu(z) = (A^{\text{ext}}_0(z), A^{\text{ext}}_i(z))

$$

This is an **external background field**, distinct from the internal gauge field $B_\mu$.

**Special case (scalar drive):** If the external reward 1-form is purely temporal, then
$A^{\text{ext}}_\mu(z) = (-\Phi_{\text{eff}}(z), \vec{0})$.

:::

:::{prf:definition} External current pairing in the comparison action
:label: ax-minimal-value-coupling

The comparison action contains $\mathcal L_{\rm drive}=j^\mu A^{\rm ext}_\mu$
with $j^\mu=\sum_{\chi=L,R}\bar\Psi_\chi\gamma^\mu\Psi_\chi$.
Varying $\bar\Psi_\chi$ contributes
$\gamma^\mu A^{\rm ext}_\mu\Psi_\chi$ to its Euler--Lagrange equation.
In an adapted unit-lapse local inertial coordinate frame, a purely
scalar drive $A^{\rm ext}=(-\Phi,0)$ gives
$\mathcal L_{\rm drive}=-\Psi^\dagger\Psi\Phi$.
On a general slice the density is $-n_\mu j^\mu$, as in
{prf:ref}`def-cognitive-spinor`; $j^0$ alone depends on the coordinates.
A time-independent potential does not break time translations just because
its value equation includes a discount parameter. An explicitly varying
background preserves only its actual symmetry subgroup.
:::

:::{prf:theorem} Exact scalar representation of the established WFR equations
:label: thm-recovery-wfr-drift

Use the fixed-metric polar construction of {prf:ref}`thm-madelung-transform`.
On a positive-density chart put $a=\sqrt\rho$, $\psi=ae^{iV/\sigma}$,
$p=dV-B$, $v=G^{-1}p$, $D_i=\partial_i-iB_i/\sigma$, and
$Q=-\sigma^2\Delta_Ga/(2a)$. The Hamilton--Jacobi and mass equations
$$
\partial_sV+\tfrac12|p|_G^2+\Phi=0,\qquad
\partial_s\rho+\operatorname{div}_G(\rho v)=r\rho
$$
are equivalent on this chart to
$$
i\sigma\partial_s\psi=
\left[-\frac{\sigma^2}{2}\Delta_B+\Phi-Q+\frac{i\sigma r}{2}\right]\psi.
$$
Here $r$ is the reaction rate, not the internal mode dimension.

:::

:::{prf:definition} Classical comparison action and its quantum obstruction
:label: def-cognitive-lagrangian

The specified matrix and chiral fields define the classical density
$$
\begin{aligned}
\mathcal L_{\rm cmp}={}&-\tfrac14B_{\mu\nu}B^{\mu\nu}
-\tfrac14W^a_{\mu\nu}W^{a\mu\nu}-\tfrac14G^a_{\mu\nu}G^{a\mu\nu}\\
&+\sum_{\chi=L,R}\frac i2\left[
\bar\Psi_\chi\gamma^\mu D_\mu\Psi_\chi
-(D_\mu\bar\Psi_\chi)\gamma^\mu\Psi_\chi\right]\\
&-(D_\mu\phi)^\dagger D^\mu\phi-\mathcal V(\phi)
+\mathcal L_Y+\sum_{\chi=L,R}\bar\Psi_\chi\gamma^\mu A^{\rm ext}_\mu\Psi_\chi.
\end{aligned}
$$
The action is $\int\mathcal L_{\rm cmp}\,d\mu_g$. The symmetric spinor
kinetic term differs from the integrated one-sided form by a boundary
term, using compatibility and the divergence theorem. All contractions
use {prf:ref}`def-cognitive-spinor` and
{prf:ref}`def-rep-covariant-derivatives`; $\mathcal L_Y$ includes its
conjugated matrix coefficients. Units in this comparison are
$c_{\rm info}=\sigma=1$; the WFR identity retains both scales explicitly.
For a homogeneous scalar in a local inertial frame the kinetic term is
$|\partial_t\phi|^2$, and its Hamiltonian density is
$|\partial_t\phi|^2+|\nabla\phi|^2+\mathcal V$.
This checks the relative kinetic sign.

The density is a classical covariant comparison functional. Its chiral
matter has the obstruction in {prf:ref}`thm-smoc-chiral-anomaly-obstruction`;
it is not an established quantum field law. The established agent dynamics
used here are the scalar polar equations and the separately defined
finite-dimensional CP updates, with their proved representation maps.
:::

:::{prf:definition} Axiomatic Field Theory (AFT)
:label: def-aft

An **Axiomatic Field Theory (AFT)** is a relativistic quantum field theory whose vacuum correlation
functions satisfy the Wightman axioms (Definition {prf:ref}`def-wightman-axioms`) {cite}`wightman1956quantum`.
Equivalently, if its Euclidean Schwinger functions satisfy the Osterwalder-Schrader axioms
(Definition {prf:ref}`def-os-axioms`), then the OS reconstruction theorem yields a Wightman QFT
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

:::

:::{prf:definition} Wightman Axioms (W0-W4)
:label: def-wightman-axioms

For the Wightman comparison, let $\Phi_A(x)$ be operator-valued tempered distributions on a positive Hilbert space with a common invariant dense domain, and let
$|\Omega\rangle$ be the vacuum. The Wightman functions are
$W_n(x_1,\ldots,x_n) := \langle \Omega | \Phi_{A_1}(x_1)\cdots\Phi_{A_n}(x_n) | \Omega \rangle$.
The axioms {cite}`wightman1956quantum` are:

1. **W0 Temperedness:** Each $W_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **W1 Poincare Covariance:** There exists a unitary representation $U(a,\Lambda)$ of the proper
   orthochronous Poincare group with
   $U(a,\Lambda)\,\Phi_A(x)\,U(a,\Lambda)^{-1} = S_A{}^B(\Lambda)\,\Phi_B(\Lambda x + a)$ and
   $U(a,\Lambda)|\Omega\rangle = |\Omega\rangle$.
3. **W2 Spectral Condition:** The joint spectrum of translation generators $P^\mu$ lies in the closed
   forward light cone, and $P^\mu|\Omega\rangle=0$.
4. **W3 Locality (Microcausality):** For spacelike separation $(x-y)^2>0$ in signature $(-+++)$,
   $[\Phi_A(x),\Phi_B(y)]_\pm = 0$, with graded commutator chosen by spin-statistics.
5. **W4 Vacuum Cyclicity:** The set of vectors generated by polynomials in smeared fields acting on
   $|\Omega\rangle$ is dense in the Hilbert space.

:::

:::{prf:definition} Osterwalder-Schrader Axioms (OS0-OS4)
:label: def-os-axioms

For a specified Euclidean correlation family $S_n$, the following labels summarize the reconstruction properties. This list does not assert that the comparison action defines such a family. The
Osterwalder-Schrader axioms {cite}`osterwalder1973axioms,osterwalder1975axioms` are:

1. **OS0 Temperedness:** Each $S_n$ is a tempered distribution in $\mathcal{S}'((\mathbb{R}^4)^n)$.
2. **OS1 Euclidean Covariance:** $S_n$ is invariant under the Euclidean group $E(4)$.
3. **OS2 Reflection Positivity:** For any polynomial $F$ of smeared fields with support in positive
   Euclidean time, $\langle \Theta F \cdot F \rangle_E \ge 0$, where $\Theta$ is time reflection.
4. **OS3 Cluster Property:** $S_{m+n}(x_1,\ldots,x_m,x_{m+1}+a,\ldots,x_{m+n}+a) \to
   S_m(x_1,\ldots,x_m)\,S_n(x_{m+1},\ldots,x_{m+n})$ as $|a|\to\infty$.
5. **OS4 Symmetry:** $S_n$ is symmetric under permutations (graded symmetry for fermions).

The full reconstruction theorem also includes growth control on the correlation family; the index OS0 here includes that requirement when the theorem is invoked. Vacuum uniqueness is the vacuum-sector property associated with clustering.

:::

:::{prf:definition} The Background Category $\mathrm{Loc}_{\mathrm{Spin},G}$
:label: def-loc-spin-g

Fix the specified compact comparison group $G=G_0$. The category $\mathrm{Loc}_{\mathrm{Spin},G}$ has objects
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ where:
1. $(\mathcal{M}, g)$ is a 4D globally hyperbolic Lorentzian manifold with orientation
   $\mathfrak{o}$ and time orientation $\mathfrak{t}$.
2. $\mathcal{S}$ is a spin structure on $(\mathcal{M}, g)$.
3. $P_G$ is a principal $G$-bundle over $\mathcal{M}$ (fixed topology).
4. $A^{\text{ext}}$ is a fixed background 1-form (the external drive).

Morphisms $\chi:(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})
\to (\mathcal{M}', g', \mathfrak{o}', \mathfrak{t}', \mathcal{S}', P_G', A^{\text{ext}\prime})$
are smooth isometric embeddings with causally convex image that preserve $\mathfrak{o}$ and
$\mathfrak{t}$, admit a lift to the spin bundles, and are covered by a bundle morphism
$\tilde{\chi}:P_G \to P_G'$ with $\chi^*A^{\text{ext}\prime} = A^{\text{ext}}$.
Internal gauge connections are dynamical fields; only the underlying bundle $P_G$ is background data.

:::

:::{prf:remark} Fixed Bundle, Dynamical Connection
:label: rem-loc-spin-g-connection

Fixing $P_G$ selects the topological sector for the gauge fields; the connection 1-forms are
sections of the affine bundle of connections on $P_G$ and remain dynamical fields. Connections themselves are gauge dependent; physical observables are their gauge-invariant combinations. The LC-AFT
assignment is the functor $\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$,
so morphisms act by pullback on background data and by *-homomorphisms on algebras.

:::

:::{prf:definition} Locally Covariant AFT (LC-AFT)
:label: def-lc-aft

A **Locally Covariant AFT** is a covariant functor
$\mathcal{A}:\mathrm{Loc}_{\mathrm{Spin},G} \to *\mathrm{Alg}$ that assigns to each object
$(\mathcal{M}, g, \mathfrak{o}, \mathfrak{t}, \mathcal{S}, P_G, A^{\text{ext}})$ a *-algebra
$\mathcal{A}(\mathcal{M})$ of gauge-invariant observables, together with a net of subalgebras
$\mathcal{A}_{\mathcal{M}}(O) \subset \mathcal{A}(\mathcal{M})$ for causally convex regions
$O \subset \mathcal{M}$, such that {cite}`haag1992local,brunetti2003locally`:

1. **Isotony:** If $O_1 \subset O_2$, then $\mathcal{A}_{\mathcal{M}}(O_1) \subset \mathcal{A}_{\mathcal{M}}(O_2)$.
2. **Locality:** If $O_1$ and $O_2$ are spacelike separated, then
   $[\mathcal{A}_{\mathcal{M}}(O_1),\mathcal{A}_{\mathcal{M}}(O_2)]_\pm = 0$.
3. **Local Covariance:** For any morphism $\chi$ in $\mathrm{Loc}_{\mathrm{Spin},G}$, the induced
   *-homomorphism $\alpha_\chi := \mathcal{A}(\chi)$ is injective and satisfies
   $\alpha_\chi(\mathcal{A}_{\mathcal{M}}(O)) = \mathcal{A}_{\mathcal{M}'}(\chi(O))$, with
   $\alpha_{\chi_2 \circ \chi_1} = \alpha_{\chi_2} \circ \alpha_{\chi_1}$ and
   $\alpha_{\mathrm{id}} = \mathrm{id}$.
4. **Time-Slice:** If $O$ contains a Cauchy surface of $\mathcal{M}$, then $\mathcal{A}_{\mathcal{M}}(O)$ generates
   $\mathcal{A}(\mathcal{M})$.
5. **Gauge Invariance:** The physical algebra is the subalgebra invariant under vertical
   automorphisms of $P_G$; a constrained realization specifies its constraint quotient before assigning physical states.
6. **State Regularity (Microlocal Spectrum):** Physical states are positive linear functionals
   with the microlocal regularity appropriate to their represented fields. For basic free KG/Dirac fields this is the Hadamard two-point condition; composite fields require their own distributional products and bounds. No such products are supplied by this definition.

:::

:::{prf:proposition} Field reconstruction versus a net of algebras
:label: thm-lc-aft-special-cases

A locally covariant net as defined in {prf:ref}`def-lc-aft` specifies
algebras and their maps. Its definition alone supplies neither a preferred
vacuum nor tempered point fields. To apply a field reconstruction theorem,
the correlation distributions, their positivity, covariance, spectral
or Euclidean regularity, and the required growth conditions must be
verified for one and the same field family. This is the meaning of the
reconstruction criterion in {prf:ref}`thm-smoc-poincare-reconstruction`.

The distinction has an elementary state-level example. A direct sum of
two positive vacuum sectors, with a convex-mixture vacuum state, retains
locality and positive energy. For the central projection $P$ onto one
summand with vacuum weight $0<p<1$,
$\omega(P\alpha_a(P))-\omega(P)^2=p(1-p)$ for every translation $a$.
Clustering therefore does not follow just from locality and positive
energy. A time-independent external background also need not preserve
spatial translations or Lorentz boosts. These properties must be checked
on its actual stabilizer rather than inferred from stationarity.
:::

:::{prf:remark} Use of the reconstruction theorem
:label: cor-aft-validity-yang-mills

The OS theorem is applied to a specified Schwinger family satisfying the
full regularity, symmetry, covariance and reflected-positivity requirements
of its chosen version. It reconstructs the corresponding Hilbert space,
fields and positive-energy representation; it does not prove that a
formal action supplies those Schwinger functions. In this chapter the
classical comparison action, finite CP maps, and scalar operator
realization are distinct constructed objects. The records below give
no OS verification for an interacting chiral gauge measure associated
with {prf:ref}`def-cognitive-lagrangian`.
:::

:::{prf:remark} Symmetries of the background
:label: rem-aft-scope

Poincare covariance requires a background and state invariant under that
group. A generic fixed spatially varying drive is not translation invariant,
even if time independent. A curved background is described by its actual
isometries or the local-covariance comparison category. Defining that
category does not construct its interacting field functor.
:::

:::{prf:definition} Local-net comparison criterion
:label: ax-constructive-locality

For each oriented Riemannian manifold $(\mathcal{M}, g)$ (boundary allowed), there is a net of
local observable *-algebras $\mathcal{A}_{\mathcal{M}}(\mathcal{O})$ for open regions
$\mathcal{O} \subset \mathcal{M}$ with isotony:
$
\mathcal{O}_1 \subset \mathcal{O}_2 \Rightarrow
\mathcal{A}_{\mathcal{M}}(\mathcal{O}_1) \subset \mathcal{A}_{\mathcal{M}}(\mathcal{O}_2).
$
Algebras of causally disjoint regions commute (graded for fermions) with causal separation defined
by Definition {prf:ref}`def-causal-interval`.
:::

:::{prf:definition} Gauge-invariant observable subalgebra
:label: ax-constructive-gauge-physical

There is a compact gauge group $G$ acting locally on fields. The physical observable algebra is
the gauge-invariant subalgebra:
$
\mathcal{A}^{\mathrm{phys}}_{\mathcal{M}}(\mathcal{O}) =
\mathcal{A}_{\mathcal{M}}(\mathcal{O})^{G}.
$
Only gauge-invariant elements represent physical observables.
:::

:::{prf:remark} Operational resolution and correlation distributions
:label: ax-constructive-finite-resolution

The positive Levin length is the operational resolution scale already
defined in {prf:ref}`def-levin-length`. It specifies distinguishability
of observations. Temperedness of a correlation distribution instead means
continuity on Schwartz test functions, with seminorm estimates for that
distribution. The former definition alone gives neither these estimates
nor uniform bounds on all $n$-point distributions. Such bounds are not
added to the resolution definition or used as proved consequences here.
:::

:::{prf:axiom} Finite Propagation
:label: ax-constructive-finite-propagation

There exists a maximum information speed $c_{\mathrm{info}}$; causal influence is restricted to the
causal interval determined by $c_{\mathrm{info}}$ (Definition {prf:ref}`def-causal-interval`).
:::

:::{prf:definition} Local-action comparison criterion
:label: ax-constructive-local-action

The dynamics are generated by a local action functional
$\mathcal{S} = \int_{\mathcal{M}} \mathcal{L}(\Phi, D\Phi, g)\,d\mathrm{vol}_g$
with $\mathcal{L}$ a local density built from covariant fields and derivatives. For disjoint
subregions, the action decomposes additively and the induced dynamics glue consistently. The local
algebra is generated by (smeared) field polynomials supported in $\mathcal{O}$.
:::

:::{prf:remark} Three uses of positivity
:label: ax-constructive-positivity

Positive belief matrices belong to {prf:ref}`def-belief-operator`.
The scalar closed quadratic form supplies a self-adjoint operator and
its spectral semigroup. A Hilbert-space completion of a *-algebra uses a
positive functional $\omega$, via
$\langle a,b\rangle=\omega(a^*b)$ and its null quotient.
These are different constructions. Merely specifying a map on an
algebra as a positivity-preserving semigroup does not define this
functional or the full field Hilbert space. Reflection positivity is the
additional reflected correlation identity explicitly tested above.
:::

:::{prf:remark} Curvature and interaction diagnostics
:label: ax-constructive-nontriviality

A nonzero curvature observable records a nonflat connection. It does not
by itself prove a non-Gaussian quantum interaction: a free abelian field
can have nonzero curvature fluctuations. Nontriviality of a reconstructed
field law is assessed on its correlation functions and observable
algebra. The commutator in the classical non-Abelian curvature explicitly
produces nonlinear terms in the specified classical action.
:::

:::{prf:remark} Thermodynamic vs. Resolution Limit
:label: rem-thermo-vs-levin-length-smoc

The continuum limit used in this volume is the **population/thermodynamic limit** (large $N$ with
empirical measures converging to a density) at **fixed** Levin length $\ell_L>0$
({ref}`sec-mean-field-metric-law`). The Levin length is an operational resolution bound (Axiom
{prf:ref}`ax-constructive-finite-resolution`), not a regulator to be sent to zero. Taking
$\ell_L \to 0$ would exit the framework by violating the Causal Information Bound and is **not**
required for validity.

:::

:::{prf:remark} Construction identities and analytic realization
:label: thm-fragile-constructive-axioms

The architecture declares its local action, internal gauge action, and
finite-resolution observations. Gauge covariance follows from the
connection identities of {prf:ref}`prop-gauge-transformation-connection`.
The chosen scalar kinetic form has the self-adjoint realization of
{prf:ref}`prop-laplace-beltrami-self-adjointness`. These facts do not verify
every constructive QFT axiom for the interacting continuum field law:
self-adjointness of a spatial scalar operator does not prove reflection
positivity of another path law, and the former information-based gap
argument is corrected in {prf:ref}`thm-mass-gap-constructive`.
The assertions of this record are the stated construction identities;
it supplies no unconditional OS or gauge mass-gap theorem.
:::

:::{prf:proposition} Linear background-field Green operators
:label: lem-smoc-green-hyperbolic

For the smooth globally hyperbolic backgrounds of
{prf:ref}`def-loc-spin-g`, a fixed smooth KG connection-wave operator is
normally hyperbolic. The square identity in
{prf:ref}`ax-cognitive-dirac-equation` shows that a fixed compatible Dirac
operator is prenormally hyperbolic. The standard Cauchy theorem for these
linear operators gives advanced and retarded Green operators on compactly
supported sections. Their supports lie in the respective causal future
and past of the source. Smooth fixed mass and curvature endomorphisms
are lower-order terms and preserve the principal symbol.
This checks the linear-operator setting of the Cauchy theorem. When the
connection and matter evolve together through the nonlinear interacting
action, they are not fixed coefficients of that theorem; the conclusion
here concerns the fixed background operators only. The Green-operator theorem and its square-root property are established in [Bär, Green-hyperbolic operators](https://arxiv.org/abs/1310.0738).
:::

:::{prf:theorem} Time-slice identity for the linear equation quotient
:label: lem-smoc-time-slice

For a fixed linear Green-hyperbolic operator $P$ from
{prf:ref}`lem-smoc-green-hyperbolic`, the equation quotient is generated
by test sections in a neighborhood $O$ of a Cauchy surface.

:::

:::{prf:remark} Scope of the free-field state result
:label: lem-smoc-hadamard-existence

The Hadamard condition concerns the short-distance wavefront structure
of the two-point distribution of a specified field state. Its familiar
free KG/Dirac existence results concern the linear background operators
of {prf:ref}`lem-smoc-green-hyperbolic` and their corresponding CCR/CAR
algebras. They do not produce the higher correlation functions or
renormalized products of the interacting comparison action. In particular
a free Hadamard covariance is not a construction of a chiral gauge state,
and its ultraviolet wavefront set supplies no infrared mass-gap estimate.
The present reconstruction record uses no interacting-state existence
conclusion from this free-field comparison.
:::

:::{prf:remark} Which objects the comparison criteria describe
:label: rem-constructive-axiom-relations

The preceding net and action criteria describe the data of an algebraic
field theory. The operational speed and resolution come from the earlier
agent definitions. The scalar form and CP maps have the constructions
stated in {prf:ref}`thm-fragile-constructive-axioms`. These facts remain
separate until explicit maps identify their algebras, states and evolution.
The OS and AQFT records below therefore report construction dependencies,
not additional premises asserted for the algorithm.
:::

:::{prf:remark} OS reconstruction dependency record
:label: thm-constructive-specialization-os-wightman

The flat stationary field sector fixes the geometry and the candidate
Schwinger functions. The reconstruction theorem
{prf:ref}`thm-smoc-poincare-reconstruction` applies to functions satisfying
its stated OS requirements. The architecture and finite resolution alone
do not verify them. In particular the prior clustering route used
{prf:ref}`thm-mass-gap-constructive`; that result now supplies only a
fixed compact scalar gap and does not establish a Yang--Mills gap.
Thus this record is not a verification of OS0--OS4 for the full interacting
action. The finite algebraic and operator constructions retain their own
proved statements; a gauge-field reconstruction cannot use clustering
derived from the very spectral assertion it is meant to justify.
:::

:::{prf:remark} Role of the construction record
:label: rem-constructive-axioms-use

The record fixes which geometry, observable algebra, state and generator
belong to each comparison. It prevents a theorem about the finite belief
operator, scalar spatial Hamiltonian, or linear wave equation from being
used for a different interacting field law. The full-field conclusions
are restricted by the explicit anomaly and reflected-positivity
calculations, while the finite and scalar identities retain their proved
content.
:::

:::{prf:remark} Dependency Map (Constructive → OS/Wightman)
:label: rem-constructive-dependency-map

```{mermaid}
graph TD
  A[Specified action and gauge transformation] --> B[Gauge covariance identities]
  C[Specified scalar quadratic form] --> D[Scalar self-adjoint operator]
  D --> E[Operator-specific spectral estimate]
  F[Specified field law and observable algebra] --> G[Check the OS identities for this law]
  G --> H{OS requirements verified}
  H -->|Yes| I[Apply OS reconstruction]
  I --> J[Reconstructed Hilbert space and fields]
  K[Construction dependency record] --> G
  E --> L[Decay for the same scalar operator]
```
:::

:::{prf:remark} Status of the reconstruction comparison
:label: rem-os-wightman-hypotheses-checked

The corrected dependency record is
{prf:ref}`thm-constructive-specialization-os-wightman`. In particular,
{prf:ref}`thm-smoc-os3-construction` does not establish clustering of the
interacting field law from the compact scalar gap. Metric covariance of
an action, positivity of a measure, and reflection invariance are distinct
from positivity of all reflected Gram matrices. Each OS identity refers
to the same Schwinger family. The previous declaration of complete
verification is therefore not retained as an antecedent of the gauge
chapter's spectral conclusions.
:::

:::{prf:remark} Algebraic properties of the established constructions
:label: thm-haag-kastler-constructive

The earlier results supply a finite belief-operator algebra with CPTP
maps ({prf:ref}`def-gksl-generator`), a scalar self-adjoint form realization
({prf:ref}`prop-laplace-beltrami-self-adjointness`), and the linear time-slice
quotient of {prf:ref}`lem-smoc-time-slice`. Each conclusion refers to its
own space and generator. A functor satisfying
{prf:ref}`def-lc-aft` would additionally specify the local observable
algebras, embeddings and their composition for the interacting fields.
Those maps are not constructed by naming the category or writing a local
action. Consequently the earlier declaration of an interacting
Haag--Kastler construction is not a consequence of these results.
The finite algebra and linear quotient retain the identities proved above.
:::

:::{prf:remark} Dependency record for local observables
:label: rem-haag-kastler-hypotheses-checked

The construction record {prf:ref}`thm-fragile-constructive-axioms`
establishes the indicated gauge identities and scalar realization.
The time-slice calculation establishes a linear equation quotient.
Neither statement is a definition or proof of the full interacting
local observable net. The algebraic requirements in
{prf:ref}`def-lc-aft` are comparison criteria, and are not imported as
extra properties of the agent. This keeps the direction of dependence
from explicitly constructed algebras to their verified properties.
:::

:::{prf:theorem} Reflection positivity for a constructed reversible path law
:label: thm-smoc-os2-construction

There is an exact positive result for the existing finite-state
reversible Markov sector discussed in {prf:ref}`def-gksl-generator`.
Let $P_t=e^{tL}$ be its transition semigroup with stationary law $\pi$ and
detailed balance $\pi_xP_t(x,y)=\pi_yP_t(y,x)$. For the stationary two-sided
path law and a bounded cylinder functional $F$ of positive times, define
$\Theta F$ by complex conjugation and time reflection. Then
$$
\mathbb E[\Theta F\,F]
=\sum_x\pi_x\left|\mathbb E[F\mid X_0=x]\right|^2\ge0.
$$
:::

:::{prf:theorem} Capacity and reflected positivity are different inequalities
:label: thm-os2-closure-semigroup

Reflection symmetry and finite information capacity alone do not imply
reflection positivity. Moreover positivity of a scalar semigroup does
not identify a separate field measure with that semigroup.

:::

:::{prf:remark} Wilson observables and the reflected Gram matrix
:label: rem-os2-gauge-fixing-wilson

For a matrix connection define
$$
W_R(C)=\operatorname{tr}_R\mathcal P\exp
\left(i g\oint_C A_\mu\,dx^\mu\right).
$$
The generators and coupling belong inside the transporter. A closed-loop
transport transforms by conjugation at its base point, proving trace
invariance. This geometric identity does not evaluate its expectation.
For a specified field law and positive-time functionals $F_i$, reflection
positivity is the matrix inequality
$\sum_{ij}\bar c_i c_j\,\mathbb E[\Theta F_iF_j]\ge0$ for all $c$.
Restricting to gauge-invariant loops does not by itself prove that matrix
is positive. The positive path-law result above applies to its own
cylinder algebra; there is no established Wilson-law identification here.
:::

:::{prf:remark} Clustering and spectral support
:label: thm-smoc-os3-construction

For an already constructed self-adjoint transfer operator, its spectral
gap bounds centered transfer matrix elements by Cauchy--Schwarz and the
spectral semigroup estimate in {prf:ref}`cor-mass-gap-existence`.
This argument applies to that same Hilbert space, state, and operator.
The compact scalar gap does not establish this estimate for the full
gauge-invariant field sector. Moreover
$S_{m+n}-S_mS_n$ is a cluster difference, not generally the fully connected
$(m+n)$-point cumulant; lower connected partitions also contribute.
The previous proof used an unestablished full-field spectral gap and
cluster-expansion control, so it does not verify OS3 for the stated
interacting Schwinger functions. Those conclusions are not used as
antecedents in the corrected gauge chapter.
:::

:::{prf:remark} OS reconstruction as a correlation-family criterion
:label: thm-smoc-poincare-reconstruction

The Osterwalder--Schrader reconstruction theorem applies to a specified
Euclidean correlation family with its full OS regularity and growth
requirements, Euclidean covariance, permutation or graded symmetry, and
reflection positivity. The abbreviated list in {prf:ref}`def-os-axioms`
is an index of these properties; pointwise temperedness for each $n$
alone must not replace the growth requirement in the theorem used.
Clustering concerns the vacuum sector of the same family.

The construction proceeds as follows. On positive-time test sequences
set $(F,G)=S(\Theta F\,G)$. Reflection positivity permits quotienting by
the null space and completion. Positive Euclidean time translations
then yield a contraction semigroup $e^{-tH}$ with $H\ge0$ on this
reconstructed space. Spatial translations and rotations act unitarily.
Euclidean time translation is a semigroup, not a unitary representation
of the full Euclidean group on this Hilbert space. The reconstruction
and analytic-continuation theorem supplies the Lorentzian positive-energy
representation and fields from the same correlation family
{cite}`osterwalder1973axioms,osterwalder1975axioms`.

This describes the mathematical reconstruction operation. The chapter's
finite CP maps, reversible Markov identity and compact scalar estimates
do not verify its premises for the interacting comparison action; in
addition that action's chiral multiplet has
{prf:ref}`thm-smoc-chiral-anomaly-obstruction`. No unconditional Poincare
or Wightman construction for that action is concluded here.
:::

## 08_multiagent/03_parameter_sieve.md

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
4. **$T_c$:** Cognitive Temperature. Its angular effect is monitored by the local Péclet diagnostic
   $\mathrm{Pe}_\theta^2(r)=2r^2|u_\pi^\theta|^2/[T_c(1-r^2)^2]$ from Theorem
   {prf:ref}`thm-angular-symmetry-breaking`; $\mathrm{Pe}_\theta\approx1$ is a finite-time crossover convention,
   not a universal critical temperature.
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

Define the **Causal Horizon Length** $\ell_0 = c_{\text{info}} \cdot \tau_{\text{proc}}$ with dimension $[L]$. This is a propagation scale and is kept separate from the stationary diffusion scale. Let the temporal discount rate be $\lambda := -\ln\gamma / \Delta t$ and identify the processing interval $\Delta t := \tau_{\text{proc}}$. In the stationary diffusion convention proved for the Bellman generator, the **diffusion screening mass** and length are:

$$
\kappa_{\mathrm{diff}}^2=\frac{\lambda}{T_c}=\frac{-\ln\gamma}{T_c\Delta t},
\qquad
\ell_{\mathrm{diff}}=\kappa_{\mathrm{diff}}^{-1}=\sqrt{\frac{T_c\Delta t}{-\ln\gamma}}

$$

with $[\kappa_{\mathrm{diff}}]=[L^{-1}]$. A $c_{\text{info}}$-based conversion is a separate propagation model and is
not used in this diffusion identity (Corollary {prf:ref}`cor-discount-as-screening-length`).

These correspond to the physics constants $\{c, \hbar, \ell_P, k_B T, \alpha_s, \gamma_{\text{cosmo}}\}$ under the isomorphism of {ref}`sec-isomorphism-dictionary`.

:::

:::{prf:definition} The Sieve Constraint System
:label: def-sieve-constraint-system

Let $\mathcal{S}(\Lambda)$ denote the vector of constraint functions. The agent is **viable** if and only if:

$$
\mathcal{S}(\Lambda) \le \mathbf{0}

$$

where the inequality holds component-wise. Each component corresponds to a Sieve node that enforces a specific consistency condition. A constraint violation ($\mathcal{S}_i > 0$) triggers a diagnostic halt at the corresponding node.

:::

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

:::

:::{prf:corollary} The Speed Ratio Bound
:label: cor-speed-ratio-bound

The ratio of buffer depth to synchronization distance is bounded:

$$
\frac{L_{\text{buf}}}{d_{\text{sync}}} \ge 1

$$

with equality only in the degenerate case of a single-module agent. For distributed agents, this ratio determines the dynamic range of viable information speeds.

:::

:::{prf:theorem} The Holographic Bound
:label: thm-holographic-bound

Let $\text{Area}_\partial$ denote the boundary area of the agent's latent manifold (dimension $[L^{D-1}]$ for a $D$-dimensional bulk) and $I_{\text{req}}$ the information capacity required for viable operation (dimensionless, counting distinguishable microstates in nats). The Levin Length must satisfy:

$$
\ell_L^{D-1} \le \frac{\nu_D \cdot \text{Area}_\partial}{I_{\text{req}}}

$$

where $\nu_D$ is a **dimensionless** holographic coefficient (Corollary {prf:ref}`cor-a-dimension-dependent-coefficient`). Both sides have dimension $[L^{D-1}]$.

:::

:::{prf:definition} The Planck-Levin Correspondence
:label: def-planck-levin-correspondence

Under the physics isomorphism ({ref}`sec-isomorphism-dictionary`), the Levin Length $\ell_L$ corresponds to the Planck Length $\ell_P$:

$$
\ell_L \leftrightarrow \ell_P = \sqrt{\frac{\hbar G}{c^3}}

$$

The holographic bound becomes the Bekenstein-Hawking entropy bound:

$$
S_{\text{BH}} = \frac{A}{4\ell_P^2}

$$

*Remark:* The coefficient $\nu_2 = 1/4$ is the declared operational
normalization in Definition {prf:ref}`thm-a-complete-derivation-area-law`.
The appendix records conditional counting and does not derive a physical
Bekenstein--Hawking result for the agent model.

:::

:::{prf:theorem} The Capacity Horizon
:label: thm-capacity-horizon

As $I_{\text{bulk}} \to I_{\max} = \nu_D \cdot \text{Area}_\partial / \ell_L^{D-1}$, the agent approaches a **Capacity Horizon**. The metric diverges:

$$
\|v\|_G \to 0 \quad \text{as} \quad I_{\text{bulk}} \to I_{\max}

$$

:::

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

:::

:::{prf:corollary} The Computational Temperature Range
:label: cor-computational-temperature-range

Under the Landauer assumptions (positive erasure and metabolic rates), the Cognitive Temperature obeys the operational bound:

$$
0 < T_c \le \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2}

$$

The angular SDE supplies the local diagnostic $\mathrm{Pe}_\theta$ rather than a second universal temperature bound.

$$
\mathrm{Pe}_\theta(r)\approx1

$$

is a finite-time crossover convention; any use of it as an engineering constraint must report the chosen radius, horizon,
and control field explicitly.


:::

:::{prf:definition} The Coupling Function
:label: def-coupling-function

Let the binding coupling $g_s(\mu)$ (dimensionless) be a function of the **resolution scale** $\mu$, which has dimension $[L^{-1}]$ (inverse length). Equivalently, $\mu$ can be expressed as an energy scale via $\mu \sim E/(\sigma \cdot c_{\text{info}})$ where $\sigma$ is the Cognitive Action Scale and $c_{\text{info}}$ is the Information Speed (Axiom {prf:ref}`ax-information-speed-limit`).

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

:::

:::{prf:theorem} The Ultraviolet Decoupling Constraint
:label: thm-uv-decoupling-constraint

At the texture scale ($\mu \to \infty$), the coupling must vanish:

$$
\lim_{\mu \to \infty} g_s(\mu) = 0

$$

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

:::{prf:theorem} The Discount Window
:label: thm-discount-window

Under the stationary diffusion convention, suppose the design declares a maximum screening length $L_{\mathrm{buf}}$ and
a positive minimum planning length $\ell_{\min}$. Then the temporal discount factor must satisfy:

$$
\exp\!\left(-\frac{T_c\Delta t}{\ell_{\min}^2}\right)
\le \gamma \le
\exp\!\left(-\frac{T_c\Delta t}{L_{\mathrm{buf}}^2}\right)<1,
\qquad
\gamma_{\min}:=\exp\!\left(-\frac{T_c\Delta t}{\ell_{\min}^2}\right).

$$

The lower endpoint encodes the declared planning requirement; without such a length scale the analysis proves only
$0<\gamma<1$, not a universal numerical $\gamma_{\min}$.

:::

:::{prf:corollary} The Screening-Buffer Consistency
:label: cor-screening-buffer-consistency

The screening length and buffer depth must satisfy:

$$
\ell_{\mathrm{diff}} = \sqrt{\frac{T_c\Delta t}{-\ln\gamma}} \lesssim L_{\text{buf}}

$$

Both sides have dimension $[L]$. For $\gamma\to1$, $\ell_{\mathrm{diff}}\to\infty$; for $\gamma\to0$,
$\ell_{\mathrm{diff}}\to0$. The causal propagation scale $\ell_0$ is not substituted for $\ell_{\mathrm{diff}}$.

*Remark:* The planning horizon cannot exceed the causal memory span. This connects the temporal discount to the spatial architecture.

:::

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
| IR Binding | $g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}$ | DNN-B |
| UV Decoupling | $g_s(\mu_{\text{UV}}) \le \epsilon$ (for $\epsilon \to 0$) | 29 |
| Stiffness Lower | $\Delta E > T_c$ | 7 |
| Stiffness Upper | $\Delta E < \chi_{\text{max}} T_c$ | 7 |
| Discount Lower | $\gamma > \gamma_{\text{min}}$ | --- |
| Discount Upper | $\gamma < 1$ | --- |

:::

:::{prf:theorem} The Feasible Region
:label: thm-feasible-region

The **Feasible Region** $\mathcal{F} \subset \mathbb{R}^n_+$ is the intersection of all constraint half-spaces:

$$
\mathcal{F} = \{ \Lambda : \mathcal{S}_i(\Lambda) \le 0 \; \forall i \}

$$

A viable agent exists if and only if $\mathcal{F} \neq \emptyset$.

:::

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
- Losing binding → Confinement (DNN-B)

:::

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

## 08_multiagent/04_dnn_blocks.md

:::{prf:definition} Group Action on Latent Space
:label: def-group-action-latent

**Preliminaries:**
- A **Lie group** is a smooth manifold $G$ equipped with a group structure such that multiplication $(g, h) \mapsto gh$ and inversion $g \mapsto g^{-1}$ are smooth maps. Examples: $SO(d)$ (rotations), $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ (rigid motions), $SU(N)$ (special unitary group).
- The **general linear group** $GL(d_z, \mathbb{R})$ is the group of invertible $d_z \times d_z$ real matrices under matrix multiplication, with smooth manifold structure (open subset of $\mathbb{R}^{d_z^2}$).

**Setup:** Let $G$ be a Lie group and $\mathcal{Z} = \mathbb{R}^{d_z}$ be the latent space equipped with standard Euclidean topology and inner product $\langle z, z' \rangle = z^T z'$.

A **linear representation** is a continuous group homomorphism $\rho: G \to GL(d_z, \mathbb{R})$ satisfying:
1. $\rho(e_G) = I_{d_z}$ where $e_G$ is the identity element of $G$
2. $\rho(g_1 g_2) = \rho(g_1)\rho(g_2)$ for all $g_1, g_2 \in G$ (homomorphism property)

**Smoothness:** When $G$ is a Lie group, we additionally require $\rho$ to be smooth as a map between manifolds: for any smooth curve $\gamma: (-\epsilon, \epsilon) \to G$ with $\gamma(0) = e_G$, the matrix-valued map $t \mapsto \rho(\gamma(t))$ is $C^\infty$.

**Group action:** For $g \in G$ and $z \in \mathcal{Z}$ (viewed as a $d_z \times 1$ column vector), the action is given by standard matrix-vector multiplication:

$$
g \cdot z := \rho(g) z \in \mathbb{R}^{d_z}
$$
where the right-hand side denotes the matrix product of $\rho(g) \in \mathbb{R}^{d_z \times d_z}$ with $z \in \mathbb{R}^{d_z \times 1}$.

**Orthogonal representations:** When $G$ is compact (e.g., $SO(d)$, $SU(N)$), any finite-dimensional continuous representation can be made orthogonal (unitary) by constructing a $G$-invariant inner product via Haar measure averaging {cite}`serre1977linear,sepanski2007compact`.

**Construction:** Let $\rho_0: G \to GL(d_z, \mathbb{R})$ be any representation, and let $\langle \cdot, \cdot \rangle_0$ be an arbitrary initial inner product on $\mathcal{Z} = \mathbb{R}^{d_z}$. Define the averaged inner product:

$$
\langle z, z' \rangle_G := \int_G \langle \rho_0(g) z, \rho_0(g) z' \rangle_0 \, dg
$$
where $dg$ is the normalized Haar measure on $G$ (unique bi-invariant measure with $\int_G dg = 1$).

**Verification:**

1. **Inner product structure:** $\langle \cdot, \cdot \rangle_G$ satisfies linearity, symmetry, and positive definiteness. For $z \neq 0$, we have $\langle \rho_0(g) z, \rho_0(g) z \rangle_0 > 0$ for all $g$ (since $\rho_0(g)$ is invertible). By compactness of $G$ and continuity, the integral $\langle z, z \rangle_G > 0$.

2. **$G$-invariance:** For any $h \in G$:
   $$\langle \rho_0(h) z, \rho_0(h) z' \rangle_G = \int_G \langle \rho_0(g) \rho_0(h) z, \rho_0(g) \rho_0(h) z' \rangle_0 \, dg$$
   Substituting $g' = gh$ and using left-invariance of Haar measure ($dg' = dg$):
   $$= \int_G \langle \rho_0(g') z, \rho_0(g') z' \rangle_0 \, dg' = \langle z, z' \rangle_G$$
   Therefore $\rho_0(h)$ is orthogonal (actually unitary) with respect to $\langle \cdot, \cdot \rangle_G$ for all $h \in G$.

We typically choose $\rho$ such that $\rho(g) \in O(d_z)$ for all $g$, ensuring:

$$
\langle \rho(g) z, \rho(g) z' \rangle = \langle z, z' \rangle \quad \forall z, z', g
$$
This preserves the Euclidean structure of $\mathcal{Z}$.

**Remark on Peter-Weyl theorem:** The Peter-Weyl theorem states that for a compact group $G$, the renormalized matrix coefficients $\sqrt{\dim(\pi)} \cdot u_{ij}^{(\pi)}(g)$ (where $u_{ij}^{(\pi)}$ are matrix elements of irreducible unitary representations $\pi$) form an orthonormal basis for $L^2(G)$ with respect to normalized Haar measure. This theorem is foundational to representation theory and is intimately connected to the averaging construction: the Haar measure averaging used above is the same measure appearing in the Peter-Weyl decomposition {cite}`peter1927theorie,weyl1946classical`.

**Units:** $[\rho(g)]$ is dimensionless (orthogonal/unitary matrices have dimensionless entries).
:::

:::{prf:definition} G-Equivariant Operator
:label: def-g-equivariant-operator

Let $G$ be a group with representations $\rho: G \to GL(\mathbb{R}^{d_z})$ and $\rho': G \to GL(\mathbb{R}^{d_{z'}})$ on latent spaces $\mathcal{Z} = \mathbb{R}^{d_z}$ and $\mathcal{Z}' = \mathbb{R}^{d_{z'}}$ respectively.

A neural operator $f: \mathcal{Z} \to \mathcal{Z}'$ is **$G$-equivariant** (or **$(\rho, \rho')$-equivariant** when representations need emphasis) if:

$$
f(\rho(g) z) = \rho'(g) f(z) \quad \forall g \in G, \; z \in \mathcal{Z}
$$

where matrix-vector multiplication is implied.

**Physical interpretation:** Transforming the input then applying $f$ gives the same result as applying $f$ then transforming the output. Formally, the following diagram commutes:

$$
\begin{array}{ccc}
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}' \\
\downarrow \rho(g) & & \downarrow \rho'(g) \\
\mathcal{Z} & \xrightarrow{f} & \mathcal{Z}'
\end{array}
$$

**Units:** If $[z] = [z']$ (both in units of $\sqrt{\text{nat}}$), then $[f(z)] = [z']$ and $[\rho(g)] = [\rho'(g)] = $ dimensionless.
:::

:::{prf:definition} Fragile Gauge Group
:label: def-fragile-gauge-group

The Fragile framework adopts the gauge structure:

$$
G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y
$$

**Source:** This gauge group emerges from the multi-agent consistency requirements under capacity constraints $C < \infty$ (Chapter 8.1, {ref}`sec-symplectic-multi-agent-field-theory`). The full derivation from first principles—showing why this specific product structure is necessary and sufficient—is provided in Chapter 8, Sections 8.1-8.3.

**Important qualification:** This chapter **assumes** the gauge structure and demonstrates its implementation in neural architectures. We do not re-derive the gauge group here; instead, we cite the field-theoretic analysis from Chapter 8.1. Readers seeking the foundational derivation should consult that chapter first.

**Explicit definitions:**

1. **$SU(N_f)_C$ (Color symmetry):** The special unitary group of degree $N_f$, consisting of $N_f \times N_f$ complex unitary matrices $U$ with $\det(U) = 1$. Here $N_f$ equals the number of feature bundles $n_b$ (see Definition {prf:ref}`def-latent-vector-bundle`). This acts on the bundle index, permitting feature mixing across bundles.

2. **$SU(2)_L$ (Weak isospin):** The special unitary group of degree 2, consisting of $2 \times 2$ complex unitary matrices with determinant 1. This has 3 real parameters (Pauli matrix basis) and acts on observation-action doublets.

3. **$U(1)_Y$ (Hypercharge):** The circle group $\{e^{i\theta} : \theta \in [0, 2\pi)\} \cong SO(2)$, representing phase rotations. This is associated with a capacity bound (holographic bound).

**Representation on latent space:** For a latent space $\mathcal{Z} = \mathbb{R}^{d_z}$ decomposed into $n_b$ bundles of dimension $d_b$ (so $d_z = n_b \cdot d_b$), we construct the representation $\rho: G_{\text{Fragile}} \to GL(d_z, \mathbb{R})$ through **real forms** of the complex gauge groups.

**Challenge:** $SU(N_f)$ and $SU(2)$ are complex Lie groups (acting on $\mathbb{C}^{N_f}$ and $\mathbb{C}^2$ respectively), but our latent space $\mathcal{Z} = \mathbb{R}^{d_z}$ is real. We need a **real representation** compatible with neural network operations.

**Construction via real forms:**

1. **$SU(N_f)_C$ real representation:** With $N_f = n_b$ (number of bundles), we model bundle mixing with a real orthogonal action on the bundle index. This is a pragmatic, real-valued proxy for $SU(N_f)_C$ that preserves bundle norms and keeps the bundle count fixed. In block form:

   $$
   \rho_C(R) = \begin{pmatrix}
   R_{11} I_{d_b} & R_{12} I_{d_b} & \cdots & R_{1n_b} I_{d_b} \\
   R_{21} I_{d_b} & R_{22} I_{d_b} & \cdots & R_{2n_b} I_{d_b} \\
   \vdots & \vdots & \ddots & \vdots \\
   R_{n_b 1} I_{d_b} & R_{n_b 2} I_{d_b} & \cdots & R_{n_b n_b} I_{d_b}
   \end{pmatrix} \in GL(d_z, \mathbb{R})
   $$
   where $R = (R_{ij}) \in SO(n_b)$ is a real orthogonal mixing matrix on bundle indices (an architectural proxy for $SU(N_f)_C$), and $I_{d_b}$ is the $d_b \times d_b$ identity (each bundle block rotates together).

   **Explicit $SU(n_b) \to SO(2n_b)$ realification (faithful):** There is no canonical homomorphism $SU(N) \to SO(N)$ in general. A standard realification yields a faithful embedding into $SO(2N)$ as follows:

   1. **Complex to real decomposition:** Any $U \in SU(n_b)$ can be written as $U = A + iB$ where $A, B \in \mathbb{R}^{n_b \times n_b}$.

   2. **Block real form:** This induces a real representation $\rho_{\mathbb{R}}: SU(n_b) \to SO(2n_b)$ given by:

      $$
      U = A + iB \mapsto \begin{pmatrix} A & -B \\ B & A \end{pmatrix} \in SO(2n_b)
      $$
      This is an embedding preserving orthogonality: $\|Uz\|^2 = \|z\|^2$ for $z \in \mathbb{C}^{n_b}$ translates to the block matrix preserving $\mathbb{R}^{2n_b}$ norm.

   **Verification of homomorphism property:**

   1. **Identity preservation:** $I_N + i \cdot 0 \mapsto \mathrm{diag}(I_N, I_N) = I_{2N}$ ✓

   2. **Multiplicativity:** For $U_1, U_2 \in SU(N)$ with $U_k = A_k + iB_k$, the product $U_1 U_2 = (A_1 A_2 - B_1 B_2) + i(A_1 B_2 + B_1 A_2)$ satisfies:
      $$\rho_{\mathbb{R}}(U_1 U_2) = \begin{pmatrix} A_1 A_2 - B_1 B_2 & -(A_1 B_2 + B_1 A_2) \\ A_1 B_2 + B_1 A_2 & A_1 A_2 - B_1 B_2 \end{pmatrix}$$
      Direct computation of block matrix multiplication gives:
      $$\rho_{\mathbb{R}}(U_1) \rho_{\mathbb{R}}(U_2) = \begin{pmatrix} A_1 & -B_1 \\ B_1 & A_1 \end{pmatrix} \begin{pmatrix} A_2 & -B_2 \\ B_2 & A_2 \end{pmatrix} = \begin{pmatrix} A_1 A_2 - B_1 B_2 & -A_1 B_2 - B_1 A_2 \\ B_1 A_2 + A_1 B_2 & -B_1 B_2 + A_1 A_2 \end{pmatrix}$$
      Therefore $\rho_{\mathbb{R}}(U_1 U_2) = \rho_{\mathbb{R}}(U_1) \rho_{\mathbb{R}}(U_2)$, confirming the homomorphism property.

   3. **Orthogonality:** The unitarity condition $U^* U = I$ for $U = A + iB$ decomposes into $A^T A + B^T B = I_N$ (real part) and $A^T B = B^T A$ (imaginary part, implying $A^T B$ is symmetric). Computing:
      $$\rho_{\mathbb{R}}(U)^T \rho_{\mathbb{R}}(U) = \begin{pmatrix} A^T & B^T \\ -B^T & A^T \end{pmatrix} \begin{pmatrix} A & -B \\ B & A \end{pmatrix} = \begin{pmatrix} A^T A + B^T B & -A^T B + B^T A \\ -B^T A + A^T B & B^T B + A^T A \end{pmatrix}$$
      Using $A^T B = B^T A$ and $A^T A + B^T B = I_N$ yields $\rho_{\mathbb{R}}(U)^T \rho_{\mathbb{R}}(U) = I_{2N}$, so $\rho_{\mathbb{R}}(U) \in O(2N)$.

   4. **Determinant:** Using the block determinant identity:
      $$\det(\rho_{\mathbb{R}}(U)) = \det(A + iB)\det(A - iB) = \det(U)\det(\bar{U}) = |\det(U)|^2 = 1$$
      confirming $\rho_{\mathbb{R}}: SU(N) \to SO(2N)$.

   **Implementation note:** The faithful $SO(2n_b)$ realification doubles the bundle-index dimension. In practice, the full $SU(n_b)$ symmetry is **broken to a discrete subgroup** (permutations and sign flips of bundles), which naturally embeds in $SO(n_b)$ as signed permutation matrices. This is enforced implicitly through the isotropic architecture design, not by explicitly constructing gauge transformations.

2. **$SU(2)_L$ real representation:** The adjoint representation identifies the Lie algebras $\mathfrak{su}(2) \cong \mathfrak{so}(3)$ and yields a surjective homomorphism $\mathrm{Ad}: SU(2) \to SO(3)$ with kernel $\{\pm I\}$ {cite}`hall2015lie`. This gives a 3D real representation that factors through $SO(3)$ (so it is **not** faithful on $SU(2)$). A faithful real representation is obtained by realifying the fundamental doublet $SU(2) \curvearrowright \mathbb{C}^2$, giving an embedding $SU(2) \hookrightarrow SO(4)$ (equivalently, $SU(2)\cong Sp(1)$ acting on quaternions $\mathbb{H}\cong \mathbb{R}^4$ by left multiplication).

   **Architecture choice — spontaneous symmetry breaking:** In practice, the full $SU(2)_L$ symmetry is **broken to a $U(1)$ subgroup** (a maximal torus; all such subgroups are conjugate to the diagonal subgroup $\mathrm{diag}(e^{i\theta}, e^{-i\theta}) \in SU(2)$). In our real-valued implementation we realize this $U(1)$ action as the standard $SO(2)$ rotation mixing the observation-action doublet components (generated by $i\sigma_2$ in the Pauli basis):

   $$
   \rho_L: U(1) \hookrightarrow SU(2)_L \xrightarrow{\text{real form}} SO(2) \subset GL(2, \mathbb{R})
   $$
   acting on $(z_{\text{obs}}, z_{\text{act}}) \in \mathbb{R}^2 \subset \mathcal{Z}$.

   **Justification for symmetry breaking:** The reduction from $SU(2)_L$ to $U(1) \subset SU(2)_L$ is consistent with the weak isospin structure in the Standard Model, where the unbroken subgroup after electroweak symmetry breaking is $U(1)_{\text{EM}} \subset SU(2)_L \times U(1)_Y$. In the neural architecture, only the $U(1)$ rotation symmetry between observation and action channels is explicitly enforced; a faithful real implementation of the full $SU(2)$ doublet would require complex features (or its 4D realification), which is incompatible with a 2D real obs-action plane. See the remark in Definition {prf:ref}`def-obs-action-doublet` on real-valued implementations.

3. **$U(1)_Y$ real representation:** The hypercharge symmetry $U(1)_Y = \{e^{i\theta} : \theta \in [0, 2\pi)\}$ acts on complex fields as phase rotations $\psi \mapsto e^{iY\theta} \psi$, preserving $|\psi|^2$. The real representation is:

   $$
   \rho_Y: U(1)_Y \to SO(2) \subset GL(2, \mathbb{R}), \quad e^{i\theta} \mapsto \begin{pmatrix} \cos(Y\theta) & -\sin(Y\theta) \\ \sin(Y\theta) & \cos(Y\theta) \end{pmatrix}
   $$
   where $Y \in \mathbb{R}$ is the hypercharge quantum number. This is a rotation in a 2D subspace with angular velocity proportional to $Y$.

   **Architecture implementation via norm preservation:** In practice, $U(1)_Y$ invariance is enforced through **Lipschitz constraints** $\sigma_{\max}(W) \leq 1$ on all linear operators (Definition {prf:ref}`def-spectral-linear`). These constraints ensure:

   $$
   \|W z\| \leq \|z\| \quad \forall z \in \mathcal{Z}
   $$
   which preserves the total "charge" $\|z\|^2$ up to a maximum value (holographic bound $C < \infty$). The spectral bound implements **non-expansiveness**, making operators contractive or (in the limit) isometric; this relaxes strict $U(1)$ rotations to norm-non-increasing maps that still preserve a scalar charge. Theorem {prf:ref}`thm-spectral-preserves-hypercharge` proves that composing spectrally normalized layers maintains $\|z_t\| \leq \|z_0\|$ (non-increasing across layers), consistent with capacity constraints.

   **Relationship to hypercharge conservation:** In the Standard Model, hypercharge $Y$ is a conserved quantum number satisfying $Q = T_3 + Y/2$ (electric charge formula). In the neural architecture, the analog is **total information content** $I(X_t; Z_t) \leq C$, which is bounded by the holographic principle. Spectral normalization ensures the *linear* operators are non-expansive ($\|Wz\| \leq \|z\|$); keeping the overall network within capacity additionally requires controlling the gain of nonlinear blocks (e.g., via rescaled NormGate or explicit Lipschitz tracking).

**Product representation:** The full representation is the **direct sum** (⊕) of these factors acting on disjoint subspaces:

$$
\rho(U_C, U_L, e^{i\theta_Y}) = \rho_C(U_C) \oplus \rho_L(U_L) \oplus \rho_Y(e^{i\theta_Y})
$$
in block-diagonal form (each factor acts on its own subspace).

**Notation clarification:** We use the direct sum ⊕ (not tensor product ⊗) as an architectural simplification:
- **Direct sum $V \oplus W$:** Dimension = $\dim(V) + \dim(W)$. Transformations are block-diagonal: $\begin{pmatrix} A & 0 \\ 0 & B \end{pmatrix}$
- **Tensor product $V \otimes W$:** Dimension = $\dim(V) \times \dim(W)$. Transformations are Kronecker products: $A \otimes B$

In this model, each gauge factor acts independently on a designated subspace of $\mathcal{Z}$, so the representation space is $\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ (direct sum), and the group representation is the direct sum of individual representations.

**Derivation of direct sum structure:**

The choice of direct sum over tensor product is **architectural**, not fundamental. Here's why:

1. **Tensor product would be physically correct:** In gauge theory, matter fields typically transform under tensor product representations. For example, quarks transform as $(\mathbf{3}, \mathbf{2}, 1/6)$ under $SU(3)_C \times SU(2)_L \times U(1)_Y$, meaning the representation space is $V_C \otimes V_L \otimes V_Y$ with dimension $3 \times 2 \times 1 = 6$ per flavor.

2. **Direct sum is a simplifying assumption:** We decompose $\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y$ with each subspace transforming independently. This reduces computational complexity:
   - Tensor product: $d_z = d_C \times d_L \times d_Y$ (exponential growth)
   - Direct sum: $d_z = d_C + d_L + d_Y$ (linear scaling)

3. **Physical interpretation:** The direct sum structure corresponds to **separate degrees of freedom** for color, weak isospin, and hypercharge. This is analogous to decomposing a particle state into spin, flavor, and color quantum numbers as independent labels, rather than a fully entangled state.

4. **Consistency with architecture:** The bundle structure (Definition {prf:ref}`def-latent-vector-bundle`) already assumes $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ where each bundle $V_i$ is invariant under $\rho_C$. The $SU(2)_L$ and $U(1)_Y$ factors act on additional subspaces orthogonal to the bundle subspace.

**Implication:** This choice means we enforce gauge symmetry **separately** for each factor, not jointly. Full tensor product representations could be implemented but would require rethinking the bundle decomposition structure.

**Critical caveat:** This architectural simplification means the implementation does **not** fully realize the gauge-theoretic structure derived in Chapter 8.1. The direct sum is a **practical approximation** that preserves gauge covariance at the level of individual factors while avoiding the combinatorial explosion of full tensor product representations. Future work could explore whether tensor product architectures provide empirical benefits justifying the increased complexity.

**Implementation note:** In practice, neural network architectures do NOT implement the full gauge group action explicitly. Instead, we build **equivariant primitives**:
- IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) is equivariant w.r.t. $\rho_C$ (bundle mixing, Theorem {prf:ref}`thm-isotropic-preserves-color`)
- SteerableConv (Section {ref}`sec-covariant-retina`) is equivariant w.r.t. $\rho_L$ (obs-action doublet, Definition {prf:ref}`def-obs-action-doublet`)
- SpectralLinear (Definition {prf:ref}`def-spectral-linear`) preserves $\rho_Y$ (hypercharge bound, Theorem {prf:ref}`thm-spectral-preserves-hypercharge`)

**Remark on complex vs. real:** Physicists typically work with complex representations because quantum mechanics is inherently complex (wavefunctions are in $\mathbb{C}$). Neural networks are real-valued (weights in $\mathbb{R}$), so we use real forms. The **isomorphism** $SU(2) \cong \text{Spin}(3) \to SO(3)$ and $SU(N) \supset SO(N)$ (via embedding) allow translation between complex and real pictures. See Section {ref}`sec-symplectic-multi-agent-field-theory` for the complex gauge field formulation; here we use the real neural implementation.

**Requirement:** All neural operators in the latent dynamics must be $G_{\text{Fragile}}$-equivariant to preserve physical consistency.

**Implication:** Standard building blocks (ReLU, LayerNorm, biased Linear) that violate even simple $SO(d)$ equivariance cannot be used directly.
:::

:::{prf:theorem} ReLU Violates SO(d) Equivariance
:label: thm-relu-breaks-equivariance

Let $d \geq 2$ and consider the standard representation $\rho: SO(d) \to GL(d, \mathbb{R})$ where $\rho(R) = R$ (i.e., rotations act by matrix multiplication).

Define the ReLU activation $f: \mathbb{R}^d \to \mathbb{R}^d$ by:

$$
f(z) = (f(z_1), \ldots, f(z_d)) \quad \text{where} \quad f(z_i) = \max(0, z_i)
$$

Then $f$ is **not** $SO(d)$-equivariant with respect to $\rho$.

:::

:::{prf:corollary} ReLU Violates Smoothness Requirements for WFR Dynamics
:label: cor-relu-breaks-wfr

The WFR geometry (Chapter 5, Section {ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) provides the dynamical framework for latent state evolution. The WFR action and geodesic integrator require computing gradient flows:

$$
\frac{dz}{ds} = -G^{ij}(z) \nabla_{z_j} \mathcal{L}_{\text{WFR}}(z)
$$

where $G^{ij}(z)$ is the metric tensor from Theorem {prf:ref}`thm-capacity-constrained-metric-law` and $\mathcal{L}_{\text{WFR}}$ is the WFR action functional.

**Smoothness requirement:** Gradient-based geodesic integrators (e.g., Boris-BAOAB in Section 5.4) require $\mathcal{L}_{\text{WFR}}$ to be at least $C^1$ (continuously differentiable) to compute well-defined gradients $\nabla_z \mathcal{L}$.

**ReLU violates this:** By Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU creates non-differentiable kinks at coordinate hyperplanes $\{z \in \mathcal{Z} : z_i = 0\}$ for each $i = 1, \ldots, d_z$.

*Explicit derivation:*

**Step 1. Network with ReLU activation:**
Consider a simple network layer $f(z) = \max(0, Wz + b)$ where $W \in \mathbb{R}^{d \times d}$, $b \in \mathbb{R}^d$. The derivative is:

$$
\frac{\partial f_i}{\partial z_j} = \begin{cases}
W_{ij} & \text{if } (Wz + b)_i > 0 \\
0 & \text{if } (Wz + b)_i < 0 \\
\text{undefined} & \text{if } (Wz + b)_i = 0
\end{cases}
$$

**Step 2. WFR action functional dependence:**
The WFR action $\mathcal{L}_{\text{WFR}}[z]$ depends on the value function $V(z)$ and policy $\pi(a|z)$.

**Illustrative functional form:** For a capacity-constrained agent (Chapter 4, Theorem {prf:ref}`thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`), a typical action functional is:

$$
\mathcal{L}_{\text{WFR}}[z] = V(f(z)) + \lambda I(\pi(\cdot|f(z)))
$$
where:
- $V: \mathcal{Z} \to \mathbb{R}$ is the expected cumulative reward (value function)
- $I(\pi(\cdot|f(z))) = I(A; f(Z))$ is mutual information between actions $A$ and latent state $f(Z)$
- $\lambda$ is the Lagrange multiplier enforcing capacity constraint $I(A; Z) \leq C$ (nat/step)
- $f$ represents the network transformation pipeline (which may contain ReLU non-differentiability)

**Source:** This form derives from the bounded-rationality variational principle (see Chapter 5, Section {ref}`sec-the-wfr-metric`). The argument below applies to **any** action functional requiring smooth gradients; we use this as a concrete example to demonstrate ReLU incompatibility.

**Step 3. Chain rule breakdown:**
To compute $\nabla_z \mathcal{L}_{\text{WFR}}$, we need:

$$
\frac{\partial \mathcal{L}}{\partial z_j} = \sum_i \frac{\partial \mathcal{L}}{\partial f_i} \cdot \frac{\partial f_i}{\partial z_j}
$$
At kink points where $(Wz + b)_i = 0$, the term $\frac{\partial f_i}{\partial z_j}$ is undefined, causing $\nabla_z \mathcal{L}$ to be undefined.

**Step 4. Consequences at kinks:**

1. **Gradient undefined:** $\nabla_z \mathcal{L}$ does not exist in the classical sense (left derivative $\lim_{h \to 0^-}$ differs from right derivative $\lim_{h \to 0^+}$)

2. **Fisher metric ill-defined:** The Fisher information metric $\mathcal{F}_{ij}(z) = \mathbb{E}_{a \sim \pi(\cdot|z)}[\partial_i \log \pi(a|z) \partial_j \log \pi(a|z)]$ requires computing $\partial_i \log \pi(a|z)$. If the policy network $\pi(a|z)$ is parameterized by a network with ReLU activations, then $\pi(a|z)$ is non-differentiable at kink points, causing $\partial_i \log \pi(a|z)$ to be undefined. Consequently, the Fisher metric tensor $\mathcal{F}(z)$ cannot be computed in the classical sense at these points.

   The capacity-constrained metric $G(z)$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law` depends on the risk tensor $T_{ij}$, which in turn depends on gradients of the value function and the Fisher metric. If either $V$ or $\pi$ uses ReLU activations, the metric $G(z)$ will have ill-defined components at kink loci.

3. **Integration errors:** Boris-BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`) uses $\frac{dz}{ds} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}\nabla \mathcal{L}\right)$ with $\mathcal{M}_{\text{curl}} := (I - \beta_{\text{curl}} G^{-1}\mathcal{F}_{\text{curl}})^{-1}$ (Value Curl; Definition {prf:ref}`def-value-curl`). At kinks, the discontinuous gradient causes ill-defined integration steps. While one could use subgradients or generalized gradients at kinks, this introduces numerical instability and prevents the integrator from preserving the symplectic structure required for long-term energy conservation

**Gauge-dependence problem:** Per Theorem {prf:ref}`thm-relu-breaks-equivariance`, ReLU kinks are coordinate-dependent. Under gauge transformation $z \mapsto U(g) \cdot z$ for $g \in G_{\text{Fragile}}$, the kink locations transform but ReLU does not transform equivariantly. This creates **inconsistent kink patterns** across gauge choices, causing geodesic flows to depend on arbitrary coordinate choices.

**Consequence:** Smooth, gauge-equivariant activations (e.g., GELU in NormGate, Definition {prf:ref}`def-norm-gated-activation`) are necessary for well-defined WFR gradient flows and gauge-invariant dynamics.

**Reference to WFR smoothness:** The WFR formulation's smoothness requirements are established through:
- Variational calculus on action functionals (standard $C^1$ requirement)
- Riemannian geometry for geodesic equations (smooth metric tensor)
- Symplectic integrator theory (Lipschitz gradients for Boris-BAOAB)

See {doc}`../05_geometry/02_wfr_geometry` for the WFR metric and {doc}`../10_appendices/03_wfr_tensor` for the variational stress-energy calculation.

$\square$
:::

:::{prf:definition} Bundle Decomposition of Latent Space
:label: def-latent-vector-bundle

Let $\mathcal{Z} = \mathbb{R}^{n_b \cdot d_b}$ be the latent space. A **bundle decomposition** partitions $\mathcal{Z}$ into $n_b$ **bundles** (subspaces) of dimension $d_b$:

$$
\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i, \quad V_i \cong \mathbb{R}^{d_b}, \quad n_b \times d_b = \dim(\mathcal{Z})
$$

Each bundle $V_i$ carries a representation of the rotation group $SO(d_b)$. For $g_i \in SO(d_b)$ and $v_i \in V_i$:

$$
\rho_i(g_i) \cdot v_i = g_i \cdot v_i \quad \text{(matrix multiplication)}
$$

where we identify $SO(d_b) \subset GL(d_b, \mathbb{R})$ as the subgroup of orthogonal matrices with determinant 1. This is the **defining (standard) representation** of $SO(d_b)$.

**Product gauge group (derivation):** Given the direct sum decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$, what is the maximal symmetry group?

**Claim:** The full symmetry group preserving the decomposition is:

$$
G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)
$$

**Justification:**
1. **Preservation of decomposition:** Any symmetry must preserve $V_i \perp V_j$ for $i \neq j$ (orthogonal direct sum). Thus transformations cannot mix bundles.

2. **Within each bundle:** On $V_i \cong \mathbb{R}^{d_b}$, the maximal continuous symmetry group preserving the Euclidean inner product is $O(d_b)$ (orthogonal group).

3. **Orientation preservation:** For neural networks with deterministic forward pass, we require **orientation-preserving** transformations (continuous deformation from identity). Thus we restrict to $SO(d_b) \subset O(d_b)$ (special orthogonal: determinant +1).

4. **Independence:** Transformations on different bundles are independent, giving the product structure $\prod_{i=1}^{n_b} SO(d_b)$.

**Group action:** For $(g_1, \ldots, g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)}, \ldots, z^{(n_b)})$ with $z^{(i)} \in V_i$:

$$
\rho(g_1, \ldots, g_{n_b}) \cdot z = (g_1 z^{(1)}, \ldots, g_{n_b} z^{(n_b)})
$$

In matrix form (with respect to concatenated basis):

$$
\rho(g_1, \ldots, g_{n_b}) = \text{diag}(g_1, \ldots, g_{n_b}) = \begin{pmatrix} g_1 & 0 & \cdots & 0 \\ 0 & g_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & g_{n_b} \end{pmatrix}
$$
(block-diagonal structure).

**Note on bundle permutations:** If bundles are **semantically distinguished** (e.g., bundle 1 = edges, bundle 2 = textures, bundle 3 = colors), we cannot permute them. If all bundles are **identical** (homogeneous feature space), the symmetry group extends to $(\prod_{i=1}^{n_b} SO(d_b)) \rtimes S_{n_b}$ where $S_{n_b}$ is the permutation group. For this architecture, we assume distinguished bundles.

**Units:** $[V_i] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (from the capacity convention in {ref}`sec-dimensional-analysis`).

**Remark:** This is a **direct sum decomposition with group action**, not a fiber bundle in the differential-geometric sense (which would require a base manifold and projection map). Analogous to gauge fields in physics (Chapter {ref}`sec-symplectic-multi-agent-field-theory`): just as the Error field $W_\mu$ transforms under $SU(2)_L$, bundles transform under their respective $SO(d_b)$ factors.
:::

:::{prf:definition} Spectral Linear Operator
:label: def-spectral-linear

A linear map $W: \mathcal{Z} \to \mathcal{Z}'$ is **spectrally normalized** if:

$$
\sigma_{\max}(W) \leq 1
$$

where $\sigma_{\max}(W)$ is the largest singular value of $W$.

**Remark (Singular values):** The singular values of $W$ are the square roots of the eigenvalues of $W^T W$. The largest singular value $\sigma_{\max}(W) = \sup_{\|z\|=1} \|Wz\|$ is the operator norm induced by the Euclidean norm. This follows from the singular value decomposition: $W = U \Sigma V^T$ where $U, V$ are orthogonal and $\Sigma$ is diagonal with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. Then $\|Wz\| = \|U \Sigma V^T z\| = \|\Sigma V^T z\|$ (since $U$ is orthogonal), which achieves maximum $\sigma_1$ when $V^T z$ aligns with the first singular vector. For a standard reference, see Horn & Johnson, *Matrix Analysis*, Theorem 5.6.2 {cite}`horn2012matrix`.

**Implementation:** $W_{\text{spectral}} = W / \sigma_{\max}(W)$.

**Units:** $[W]$ is dimensionless (if $\mathcal{Z}, \mathcal{Z}'$ normalized).
:::

:::{prf:theorem} Spectral Normalization Preserves Light Cone
:label: thm-spectral-preserves-light-cone

Let $(\mathcal{Z}, \|\cdot\|)$ be a finite-dimensional normed vector space with $\|\cdot\|$ the **Euclidean norm** $\|z\| = \sqrt{z^T z}$. Let $W: \mathcal{Z} \to \mathcal{Z}'$ be a linear map with $\sigma_{\max}(W) \leq 1$ (spectrally normalized). Let $c_{\text{info}}$ be the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).

Then:

$$
\|W \cdot z\| \leq \|z\| \quad \text{(contraction property)}
$$

This ensures that whenever $d(z_1, z_2) := \|z_1 - z_2\| \leq c_{\text{info}} \cdot \Delta t$ (causal interval), we have:

$$
d(W \cdot z_1, W \cdot z_2) \leq c_{\text{info}} \cdot \Delta t
$$

Thus the **light cone** $\mathcal{C}(z_0, t_0) := \{(z, t) : \|z - z_0\| \leq c_{\text{info}}(t - t_0), \, t \geq t_0\}$ in the extended state-time space is preserved: if $(z, t) \in \mathcal{C}(z_0, t_0)$, then $(W \cdot z, t) \in \mathcal{C}(W \cdot z_0, t_0)$.

:::

:::{prf:proposition} Bias Terms Break Tangent Bundle Structure
:label: prop-bias-breaks-tangent-bundle

Working in the tangent bundle $T\mathcal{Z}$, elements are pairs $(z, v)$ where $z \in \mathcal{Z}$ is a point and $v \in T_z \mathcal{Z}$ is a tangent vector at $z$.

A gauge-covariant map on the tangent bundle must satisfy:

$$
f(z, \rho(g) \cdot v) = \rho'(g) \cdot f(z, v) \quad \forall g \in G
$$

**Claim:** Adding a bias $b$ violates this unless $b = 0$.

:::

:::{prf:definition} Norm-Gated Activation
:label: def-norm-gated-activation

For a vector bundle $v_i \in V_i$, define the **norm-gated activation**:

$$
f(v_i) = v_i \cdot g(\|v_i\| + b_i)
$$

where:
- $\|v_i\| = \sqrt{v_i^T v_i}$ is the Euclidean norm ($SO(d_b)$-invariant)
- $b_i \in \mathbb{R}$ is a learnable scalar bias (the "activation potential")
- $g: \mathbb{R} \to \mathbb{R}$ is a smooth scalar function (e.g., GELU, sigmoid)

**Physical interpretation:** Energy filter with radial symmetry. The gate opens when signal energy $\|v_i\|$ exceeds the potential barrier $-b_i$.

**Units:** All dimensionless if latent vectors normalized.

**Smoothness at the origin (implementation detail):** The norm $\|v\|$ is not differentiable at $v=0$. The map $v \mapsto v \cdot g(\|v\|+b)$ is still $C^1$ under mild regularity of $g$ (the apparent $\|v\|^{-1}$ singularity in the Jacobian cancels as $v \to 0$), but if you require a globally $C^\infty$ map you can replace $\|v\|$ with a smoothed norm $\|v\|_\varepsilon := \sqrt{v^T v + \varepsilon^2}$ in implementations.

**Remark (Choice of gating function $g$):** While any smooth $g: \mathbb{R} \to \mathbb{R}$ preserves equivariance, GELU is a **pragmatic choice** among functions satisfying design constraints:

1. **$C^\infty$ smoothness of $g$ (necessary):** $g$ should be $C^\infty$; combined with a smoothed norm $\|v\|_\varepsilon$ (if global smoothness is required), this yields a $C^\infty$ block compatible with the WFR metric (Corollary {prf:ref}`cor-relu-breaks-wfr`) and geodesic integrator assumptions (Section 5.4).

2. **Linear growth at large arguments (desirable):** For $x \gg 1$, GELU satisfies $g(x) \approx x$, so the gate value scales approximately linearly with energy rather than saturating to a constant gain.

3. **Controlled Lipschitz constant (practical):** $L_g \approx 1.129$ (Lemma {prf:ref}`lem-normgate-lipschitz`), close to 1 and comparable to softplus.

4. **Empirical effectiveness (validation):** Strong performance in transformers {cite}`hendrycks2016gaussian`.

**Critical distinction:** GELU is **not uniquely determined** by first principles. Conditions 1-3 are satisfied by multiple functions (e.g., Swish family $g(x) = x \cdot \sigma(\beta x)$, Softplus). A first-principles derivation selecting GELU uniquely (e.g., via information-geometric optimization) remains an open problem.

**Comparison with alternatives:**

| Activation | Smoothness | $\sup_x \|g'(x)\|$ | Unbounded? | Issue if used |
|------------|------------|-------------------|------------|---------------|
| Sigmoid | $C^\infty$ | $1/4$ | No (saturates to [0,1]) | Caps the gate gain at 1 (reduced dynamic range) |
| Tanh | $C^\infty$ | $1$ | No (saturates to [-1,1]) | Caps the gate gain (and allows sign flips) |
| Softplus | $C^\infty$ | $1$ | Yes (linear at $+\infty$) | Always nonnegative (good if you want a true “gate”) |
| GELU | $C^\infty$ | $\approx 1.129$ | Yes | Slight amplification ($L > 1$), addressed via spectral normalization |

The key advantage of GELU is that high-energy features propagate with energy-dependent gain ($g(x) \to x$ as $x \to \infty$), while sigmoid/tanh force the gate value to saturate to a constant.
:::

:::{prf:theorem} Norm-Gating Preserves SO(d_b) Equivariance
:label: thm-norm-gating-equivariant

Let $f$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`). Then:

$$
f(R \cdot v) = R \cdot f(v) \quad \forall R \in SO(d_b)
$$

:::

:::{prf:definition} Isotropic Block
:label: def-isotropic-block

The **Isotropic Block** is the atomic unit of gauge-covariant architecture:

$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$

where:
- **SpectralLinear**: Linear map $W$ with $\sigma_{\max}(W) \leq 1$ (Definition {prf:ref}`def-spectral-linear`) that is **block-diagonal** with respect to the bundle decomposition (see Lemma {prf:ref}`lem-block-diagonal-necessary`)
- **Reshape**: $\mathbb{R}^d \to (\mathbb{R}^{d_b})^{n_b}$ (bundle partition)
- **NormGate**: Norm-gated activation applied per bundle (Definition {prf:ref}`def-norm-gated-activation`)

**Structure constraint:** For **exact** $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$ equivariance, the weight matrix $W$ must be block-scalar:

$$
W = \begin{pmatrix} \lambda_1 I_{d_b} & 0 & \cdots & 0 \\ 0 & \lambda_2 I_{d_b} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n_b} I_{d_b} \end{pmatrix}
$$
where each $\lambda_i \in [-1, 1]$ is a learnable scalar (per-bundle scaling factor), and $I_{d_b}$ is the $d_b \times d_b$ identity matrix. This constraint follows from Schur's lemma: any linear map commuting with all elements of $SO(d_b)$ must be a scalar multiple of identity (see Lemma {prf:ref}`lem-schur-scalar-constraint`).

**Practical relaxation (approximate equivariance):** For increased expressiveness, implementations may use general block-diagonal $W$ with $\sigma_{\max}(W_i) \leq 1$. This sacrifices exact equivariance but provides bounded equivariance violation (see Proposition {prf:ref}`prop-approximate-equivariance-bound`).
:::

:::{prf:lemma} Block-Diagonal Structure is Necessary (But Not Sufficient) for Bundle Equivariance
:label: lem-block-diagonal-necessary

Let $W: \mathcal{Z} \to \mathcal{Z}$ be a linear map on $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ with $V_i \cong \mathbb{R}^{d_b}$.

For $W$ to be $G_{\text{bundle}}$-equivariant where $G_{\text{bundle}} = \prod_{i=1}^{n_b} SO(d_b)$, it is **necessary** that $W$ be block-diagonal: $W = \text{diag}(W_1, \ldots, W_{n_b})$ with $W_i: V_i \to V_i$.

However, block-diagonal structure is **not sufficient** for equivariance of the full IsotropicBlock; an additional constraint is required (see Lemma {prf:ref}`lem-schur-scalar-constraint`).

:::

:::{prf:lemma} Schur's Lemma Constraint: Scalar Blocks Required for Equivariance
:label: lem-schur-scalar-constraint

Let $W = \text{diag}(W_1, \ldots, W_{n_b})$ be block-diagonal where each $W_i: V_i \to V_i$ with $V_i \cong \mathbb{R}^{d_b}$.

Assume the scalar map $\phi_i: [0,\infty) \to [0,\infty)$ defined by

$$
\phi_i(r) := r \, \bigl|g(r + b_i)\bigr|
$$
is injective on $[0,\infty)$ (a non-degeneracy condition that holds, for example, when $g(r+b_i)\ge 0$ and $r \mapsto r\,g(r+b_i)$ is strictly increasing on the operating range).

Then the composition $\text{NormGate} \circ W$ is $G_{\text{bundle}}$-equivariant if and only if each $W_i = \lambda_i I_{d_b}$ for some scalar $\lambda_i \in \mathbb{R}$.

:::

:::{prf:theorem} IsotropicBlock is G-Equivariant (Scalar Block Case)
:label: thm-isotropic-block-equivariant

Let $G = \prod_{i=1}^{n_b} SO(d_b)$ be the product gauge group (Definition {prf:ref}`def-latent-vector-bundle`). By Lemmas {prf:ref}`lem-block-diagonal-necessary` and {prf:ref}`lem-schur-scalar-constraint`, the weight matrix $W$ in SpectralLinear must be **block-scalar** for exact equivariance:

$$
W = \begin{pmatrix} \lambda_1 I_{d_b} & 0 & \cdots & 0 \\ 0 & \lambda_2 I_{d_b} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_{n_b} I_{d_b} \end{pmatrix}
$$

where each $\lambda_i \in [-1, 1]$ is a learnable scalar satisfying $|\lambda_i| \leq 1$ (spectral normalization).

Then the IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) is **exactly** $G$-equivariant.

:::

:::{prf:proposition} Approximate Equivariance Bound for General Block-Diagonal W
:label: prop-approximate-equivariance-bound

For practical architectures using general block-diagonal $W = \text{diag}(W_1, \ldots, W_{n_b})$ with $\sigma_{\max}(W_i) \leq 1$ (instead of scalar blocks), the equivariance violation is bounded.

**Statement:** Let $\text{IB}$ denote IsotropicBlock with general block-diagonal $W$. Fix an operating range in which the per-bundle NormGate is $L_{\text{NG}}$-Lipschitz (Lemma {prf:ref}`lem-normgate-lipschitz`). Then for any $g = (g_1,\ldots,g_{n_b}) \in G_{\text{bundle}}$ and $z = (z^{(1)},\ldots,z^{(n_b)}) \in \mathcal{Z}$:

$$
\|\text{IB}(\rho(g)\cdot z) - \rho(g)\cdot \text{IB}(z)\|
\le
L_{\text{NG}} \, \|(W\rho(g) - \rho(g)W)z\|
=
L_{\text{NG}} \left(\sum_{i=1}^{n_b} \|[W_i, g_i]\,z^{(i)}\|^2\right)^{1/2}.
$$

In particular,

$$
\|\text{IB}(\rho(g)\cdot z) - \rho(g)\cdot \text{IB}(z)\|
\le
L_{\text{NG}} \sum_{i=1}^{n_b} \|[W_i, g_i]\|_{\mathrm{op}} \,\|z^{(i)}\|.
$$

**Implication:** If each $W_i$ commutes with $SO(d_b)$ (equivalently $W_i=\lambda_i I$ by Lemma {prf:ref}`lem-schur-scalar-constraint`), then $[W_i,g_i]=0$ and the equivariance violation is exactly zero.

:::

:::{prf:proposition} Conv2d is NOT SO(2)-Equivariant
:label: prop-conv-not-rotation-equivariant

Standard `Conv2d` with learned kernels is translation-equivariant but **not** rotation-equivariant for continuous rotations.

:::

:::{prf:definition} Image as Bundle Section
:label: def-image-as-bundle-section

An RGB image $I: \mathbb{R}^2 \to \mathbb{R}^3$ is a section of the trivial bundle $\mathbb{R}^2 \times \mathbb{R}^3$.

A rotation $R \in SO(2)$ acts on the base (spatial coordinates) but not the fiber (RGB values):

$$
(R \cdot I)(x) = I(R^{-1} \cdot x)
$$

For equivariant features, we need a **non-trivial bundle** where fibers also transform.
:::

:::{prf:definition} Steerable Filter Bank
:label: def-steerable-filter-bank

A filter bank $\{\psi_n^{(\ell)}\}_{n=1}^N$ is **steerable of type $\ell$** if:

$$
R_\theta \cdot \psi_n^{(\ell)} = \sum_m D_{nm}^{(\ell)}(\theta) \psi_m^{(\ell)}
$$

where $D^{(\ell)}$ is the $\ell$-th irreducible representation of $SO(2)$.

**Explicit form of $SO(2)$ irreducible representations:**

For $\ell \in \mathbb{Z}_{\geq 0}$ (non-negative integers), the $\ell$-th irreducible representation is:
- **$\ell = 0$:** Trivial representation, $D^{(0)}(\theta) = 1$ (scalar, 1-dimensional)
- **$\ell \geq 1$:** 2-dimensional representation acting on $\mathbb{R}^2$ or $\mathbb{C}$ via:

  $$
  D^{(\ell)}(\theta) = \begin{pmatrix} \cos(\ell\theta) & -\sin(\ell\theta) \\ \sin(\ell\theta) & \cos(\ell\theta) \end{pmatrix} \in SO(2)
  $$
  Equivalently, in complex notation: $D^{(\ell)}(\theta) \cdot z = e^{i\ell\theta} z$ for $z \in \mathbb{C}$.

**Physical interpretation:**
- $\ell$ is the **angular frequency** or **angular momentum quantum number**
- Under rotation by $\theta$, an $\ell$-mode rotates by $\ell \theta$ (frequency multiplication)

**Interpretation:**
- $\ell = 0$: Scalars (rotation-invariant, e.g., circularly symmetric filters). $D^{(0)}(\theta) = 1$
- $\ell = 1$: Vectors (oriented edge detectors). Rotate by $\theta$ → features rotate by $\theta$. $D^{(1)}(\theta) = R_\theta$
- $\ell = 2$: Quadrupoles (corner detectors). Rotate by $\theta$ → features rotate by $2\theta$. $D^{(2)}(\theta) = R_{2\theta}$
:::

:::{prf:definition} Feature Bundle as Associated Vector Bundle
:label: def-associated-feature-bundle

Let $P = SE(2) = \mathbb{R}^2 \rtimes SO(2)$ be the Euclidean group (translations and rotations), and let $H = SO(2)$ be the structure group.

For steerable features of type $\ell$, the **associated vector bundle** is:

$$
E^{(\ell)} = P \times_{H} V^{(\ell)}
$$

where:
- $V^{(\ell)} \cong \mathbb{R}^{2}$ (for $\ell \geq 1$) or $\mathbb{R}$ (for $\ell = 0$) is the representation space
- $SO(2)$ acts on $V^{(\ell)}$ via $D^{(\ell)}$ (the $\ell$-th irreducible representation)
- The quotient is formed by identifying $(p, v) \sim (ph, h^{-1} \cdot v)$ for $h \in SO(2)$

**Structure:**
- **Total space:** $E^{(\ell)} = \{[(g, v)] : g \in SE(2), v \in V^{(\ell)}\}$ (equivalence classes)
- **Base space:** $B = SE(2)/SO(2) \cong \mathbb{R}^2$ (spatial positions)
- **Projection:** $\pi: E^{(\ell)} \to B$, $\pi([(g, v)]) = [g] \in \mathbb{R}^2$
- **Fiber:** $F_x = \pi^{-1}(x) \cong V^{(\ell)}$ (representation space at position $x$)

**Sections:** A steerable feature map is a **section** $\phi: B \to E^{(\ell)}$ satisfying $\pi \circ \phi = \text{id}_B$.

In coordinates: $\phi(x) = (f_1^{(\ell)}(x), \ldots, f_N^{(\ell)}(x))$ where each $f_n^{(\ell)}$ transforms under $D^{(\ell)}$.
:::

:::{prf:definition} Connection and Covariant Derivative
:label: def-connection-steerable-bundle

A **connection** on the bundle $E^{(\ell)}$ specifies how to compare fibers at different base points.

For the associated bundle $E^{(\ell)} = SE(2) \times_{SO(2)} V^{(\ell)}$, the **canonical flat connection** is:

$$
\nabla_X \phi = X[\phi]
$$

where $X$ is a vector field on $\mathbb{R}^2$ and $X[\phi]$ is the directional derivative.

**Parallel transport:** A section $\phi(x + tv)$ along direction $v$ is **parallel** if $\nabla_v \phi = 0$, i.e., $\frac{d}{dt}\phi(x + tv) = 0$.

For steerable features, parallel transport preserves the transformation law:

$$
\phi(x + \delta x) = \phi(x) + \nabla_{\delta x} \phi + O(\|\delta x\|^2)
$$

where $\nabla_{\delta x} \phi$ transforms under $D^{(\ell)}$ at the new position.
:::

:::{prf:remark} Gauge Fields and Curvature
:label: rem-gauge-curvature-vision

In the gauge theory framework (Chapter 8.1), the **Binding field** $G_\mu$ acts as a connection on the feature bundle. The curvature (field strength) $F_{\mu\nu} = \partial_\mu G_\nu - \partial_\nu G_\mu + [G_\mu, G_\nu]$ measures the failure of parallel transport to be path-independent.

For the trivial bundle with flat connection (used in steerable CNNs), $F_{\mu\nu} = 0$ (zero curvature), meaning parallel transport is path-independent. This corresponds to **free field theory** in physics.

**Extension to non-trivial connections:** If steerable features are coupled to other latent variables via attention or gating, the effective connection becomes non-trivial (non-zero $G_\mu$), introducing curvature. This is explored in Section 8.5 (Covariant Cross-Attention) with **Wilson lines** for parallel transport.
:::

:::{prf:theorem} Steerable Convolution is SO(2)-Equivariant
:label: thm-steerable-conv-equivariant

Let $\{\psi_n^{(\ell)}\}_{n=1}^N$ be a steerable filter bank of type $\ell$ (Definition {prf:ref}`def-steerable-filter-bank`). Let $\text{Conv}_\ell$ denote convolution with these filters:

$$
(\text{Conv}_\ell I)_n(x) = \sum_m (\psi_m^{(\ell)} * I)(x) = \int_{\mathbb{R}^2} \psi_m^{(\ell)}(y) I(x - y) \, dy
$$

Then for any rotation $R_\theta \in SO(2)$:

$$
\text{Conv}_\ell(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}_\ell(I)
$$

where $D^{(\ell)}(\theta)$ is the $\ell$-th irreducible representation of $SO(2)$.

:::

:::{prf:definition} Lifting Map to SE(2)
:label: def-lifting-map

**Group structure:** The **special Euclidean group** $SE(2) = \mathbb{R}^2 \rtimes SO(2)$ is the group of rigid motions (translations + rotations) in the plane, with elements $g = (x, R_\theta)$ where $x \in \mathbb{R}^2$ is translation and $R_\theta \in SO(2)$ is rotation by angle $\theta$.

**Group multiplication:** For $g_1 = (x_1, R_{\theta_1})$ and $g_2 = (x_2, R_{\theta_2})$:

$$
g_1 \cdot g_2 = (x_1 + R_{\theta_1} x_2, R_{\theta_1 + \theta_2})
$$

*Interpretation:* Composition $g_1 \cdot g_2$ means "first apply $g_2$, then $g_1$". Acting on a point $p \in \mathbb{R}^2$:

$$
(g_1 \cdot g_2) \cdot p = g_1 \cdot (g_2 \cdot p) = g_1 \cdot (R_{\theta_2} p + x_2) = R_{\theta_1}(R_{\theta_2} p + x_2) + x_1 = R_{\theta_1 + \theta_2} p + (x_1 + R_{\theta_1} x_2)
$$
The rotation $R_{\theta_1}$ in $g_1$ acts on the translation $x_2$ from $g_2$, and the rotations compose as $R_{\theta_1} R_{\theta_2} = R_{\theta_1 + \theta_2}$.

**Domain and codomain:** Let $C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ be the space of continuous functions (images) from $\mathbb{R}^2$ to $\mathbb{R}^{C_{\text{in}}}$ (e.g., $C_{\text{in}} = 3$ for RGB). Let $C(SE(2), \mathbb{R}^{C_{\text{out}}})$ be functions on $SE(2)$ with values in $\mathbb{R}^{C_{\text{out}}}$ (output feature dimension).

The **lifting map** is an operator:

$$
L: C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}}) \to C(SE(2), \mathbb{R}^{C_{\text{out}}})
$$

**Definition:** For an input image $I \in C(\mathbb{R}^2, \mathbb{R}^{C_{\text{in}}})$ and group element $g = (x, R_\theta) \in SE(2)$:

$$
(L I)(g) = (L I)(x, \theta) := \sum_{i=1}^{C_{\text{out}}} (\psi_i^{(\theta)} * I)(x) \cdot e_i
$$
where:
- $\{\psi_i\}_{i=1}^{C_{\text{out}}}$ is a steerable filter bank at orientation $\theta = 0$ (Definition {prf:ref}`def-steerable-filter-bank`)
- $\psi_i^{(\theta)} := R_\theta \cdot \psi_i$ is the **rotated filter**: applying rotation $R_\theta$ to the base filter $\psi_i$
- $\{e_i\}$ is the standard basis of $\mathbb{R}^{C_{\text{out}}}$

**Key dependence on $\theta$:** The output $(LI)(x, \theta)$ depends explicitly on the rotation angle $\theta$ through the rotated filters $\psi_i^{(\theta)}$. At each orientation $\theta$, the network applies filters rotated to that orientation, detecting patterns aligned with $\theta$.

**Explicit filter rotation:** For a filter $\psi: \mathbb{R}^2 \to \mathbb{R}$ and rotation $R_\theta \in SO(2)$:

$$
(R_\theta \cdot \psi)(y) := \psi(R_\theta^{-1} y)
$$
This ensures that a filter detecting "vertical edge" at $\theta = 0$ becomes a filter detecting "edge at angle $\theta$" after rotation.

**Equivariance property:** For $g_0 = (x_0, R_{\theta_0}) \in SE(2)$ and image $I$, define the left-translated image $(L_{g_0} I)(x) := I(R_{\theta_0}^{-1}(x - x_0))$. Then:

$$
L(L_{g_0} I) = L_{g_0}(L I)
$$
where the right-hand side is left-multiplication on $SE(2)$: $(L_{g_0} f)(g) = f(g_0^{-1} g)$.

*Verification:* Evaluate both sides at $g = (x, R_\theta)$:
- **LHS:** $(L(L_{g_0} I))(x, \theta) = \sum_i (\psi_i^{(\theta)} * (L_{g_0} I))(x) \cdot e_i$
- Change variables in convolution: $(\psi_i^{(\theta)} * (L_{g_0} I))(x) = \int \psi_i(R_\theta^{-1}(x - y)) I(R_{\theta_0}^{-1}(y - x_0)) dy$
- Substitute $u = R_{\theta_0}^{-1}(y - x_0)$, so $y = R_{\theta_0} u + x_0$:
  $$= \int \psi_i(R_\theta^{-1}(x - x_0 - R_{\theta_0} u)) I(u) du = \int \psi_i(R_{\theta - \theta_0}^{-1}(R_{\theta_0}^{-1}(x - x_0) - u)) I(u) du$$
- **RHS:** $(L_{g_0}(L I))(g) = (L I)(g_0^{-1} g)$ where $g_0^{-1} = (-R_{\theta_0}^{-1} x_0, R_{\theta_0}^{-1})$
- $g_0^{-1} g = (R_{\theta_0}^{-1}(x - x_0), R_{\theta - \theta_0})$ (using SE(2) multiplication)
- $(L I)(R_{\theta_0}^{-1}(x - x_0), \theta - \theta_0) = \sum_i (\psi_i^{(\theta - \theta_0)} * I)(R_{\theta_0}^{-1}(x - x_0)) \cdot e_i$
- This matches the LHS after recognizing $\psi_i^{(\theta - \theta_0)}(y) = \psi_i(R_{\theta - \theta_0}^{-1} y)$. $\square$

**Geometric interpretation:** Instead of features at spatial locations $x \in \mathbb{R}^2$, lifted features live at *posed locations* $(x, \theta) \in SE(2)$: position AND orientation. The network learns to detect patterns *and* their orientations explicitly.

**Output dimension:** For $N_\theta$ discrete orientations (e.g., $N_\theta = 8$ for $\theta \in \{0°, 45°, 90°, \ldots, 315°\}$), the output has dimension $C_{\text{out}} = N_\theta \times C_{\text{feature}}$ where $C_{\text{feature}}$ is the number of feature types per orientation.
:::

:::{prf:definition} SU(N_f) Gauge Action on Bundle Space
:label: def-gauge-action-bundles

Let $Z = (z^{(1)}, \ldots, z^{(n_b)})$ be the bundled latent representation, viewed here in the **complexified** setting with $z^{(i)} \in \mathbb{C}^{d_b}$. The gauge group $SU(N_f)$ with $N_f = n_b$ acts on $Z$ as:

$$
Z \mapsto Z' = Z \cdot U
$$

where $U \in SU(N_f)$ is an $n_b \times n_b$ special unitary matrix:

$$
U^\dagger U = I, \quad \det(U) = 1
$$

**Explicit action:** In matrix form, treating $Z$ as a $d_b \times n_b$ matrix:

$$
z'^{(j)} = \sum_{i=1}^{n_b} U_{ij} \, z^{(i)} \quad \text{(bundle mixing)}
$$

**Color charge:** Represent the latent state as a matrix $Z \in \mathbb{C}^{d_b \times n_b}$ where the $i$-th column is the $i$-th bundle vector $z^{(i)} \in \mathbb{C}^{d_b}$:

$$
Z = [z^{(1)} \mid z^{(2)} \mid \cdots \mid z^{(n_b)}]
$$

For each generator $T^a \in \mathfrak{su}(n_b)$ ($a = 1, \ldots, n_b^2 - 1$), where $T^a$ is a traceless Hermitian $n_b \times n_b$ matrix, define the **color charge operator**:

$$
Q_C^a[Z] = \text{Tr}_{\text{bundle}}(Z^\dagger Z \cdot T^a) = \sum_{i,j=1}^{n_b} T^a_{ij} \, (z^{(i)})^\dagger z^{(j)}
$$

where:
- $Z \in \mathbb{C}^{d_b \times n_b}$ has columns $z^{(1)}, \ldots, z^{(n_b)} \in \mathbb{C}^{d_b}$ (bundles)
- $Z^\dagger \in \mathbb{C}^{n_b \times d_b}$ is the conjugate transpose (rows are bundle vectors)
- $Z^\dagger Z \in \mathbb{C}^{n_b \times n_b}$ is the **Gram matrix** with $(Z^\dagger Z)_{ij} = (z^{(i)})^\dagger z^{(j)}$
- $T^a \in \mathbb{C}^{n_b \times n_b}$ is the generator matrix (acts on bundle indices)
- $Z^\dagger Z \cdot T^a \in \mathbb{C}^{n_b \times n_b}$ is matrix multiplication
- $\text{Tr}_{\text{bundle}}$ denotes trace over bundle indices (summing diagonal elements of the $n_b \times n_b$ matrix)

**Dimensional consistency:**
- $[z^{(i)}] = \sqrt{\text{nat}}$ (latent vector)
- $[(z^{(i)})^\dagger z^{(j)}] = \text{nat}$ (inner product)
- $[T^a_{ij}]$ = dimensionless (matrix element)
- $[Q_C^a] = \text{nat}$ (charge is extensive in latent dimension)

A state is **color-neutral** (confined) if:

$$
Q_C^a[Z] = 0 \quad \forall \, a = 1, \ldots, n_b^2 - 1
$$

**Physical interpretation:** Just as quarks in QCD carry color charge under $SU(3)_C$, latent features carry "bundle charge" under $SU(N_f)_C$ with $N_f = n_b$. Only color-neutral combinations (satisfying all $n_b^2 - 1$ charge constraints) can propagate to the macro level.
:::

:::{prf:theorem} Isotropic Blocks Preserve SU(N_f)_C Gauge Structure
:label: thm-isotropic-preserves-color

The bundled structure (Definition {prf:ref}`def-latent-vector-bundle`) with $n_b$ bundles, coupled with norm-gating and the Binding field $G_\mu$, implements $SU(N_f)_C$ gauge invariance with $N_f = n_b$.

**Statement:** For any $U \in SU(N_f)$, the effective dynamics satisfy:

$$
\mathcal{L}_{\text{eff}}[Z \cdot U] = \mathcal{L}_{\text{eff}}[Z]
$$

Moreover, norm-gating induces scale-dependent coupling:
- **Infrared** ($\ell \to \infty$): Strong coupling $g_s(\mu_{\text{IR}}) \gg 1$ → confinement
- **Ultraviolet** ($\ell \to 0$): Weak coupling $g_s(\mu_{\text{UV}}) \to 0$ → asymptotic freedom

:::

:::{prf:proposition} Coupling Strength from Norm-Gating Barriers
:label: prop-coupling-from-barriers

The effective gauge coupling at layer $\ell$ is:

$$
g_s^{(\ell)} = \beta_\ell \cdot \frac{\xi_\ell}{\sqrt{\xi_\ell^2 + \eta_\ell^2}}
$$

where:
- $\beta_\ell = \tanh(b_\ell)$ is the dimensionless barrier strength
- $\xi_\ell = \langle \|W_{\text{off-diag}}^{(\ell)}\| \rangle$ is the mean off-diagonal weight norm (dimensionless)
- $\eta_\ell = \langle \|\partial_\ell z\|_2 / \|\partial_{\ell-1} z\|_2 \rangle$ is the normalized gradient flow ratio (dimensionless)

:::

:::{prf:theorem} Spectral Norm Bounds Hypercharge Dissipation
:label: thm-spectral-preserves-hypercharge

The Opportunity field $B_\mu$ (from {ref}`sec-symplectic-multi-agent-field-theory`) couples to hypercharge $Y$. Spectral normalization ensures hypercharge is **non-increasing** under forward propagation:

$$
Y(W \cdot z) \leq Y(z) \quad \text{(hypercharge cannot increase)}
$$

:::

:::{prf:definition} Observation-Action Doublet Structure
:label: def-obs-action-doublet

The latent space decomposes into **observation** and **action planning** subspaces:

$$
\mathcal{Z} = \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

These form an $SU(2)_L$ doublet:

$$
\Psi = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix} \in \mathcal{Z}_{\text{obs}} \oplus \mathcal{Z}_{\text{act}}
$$

**Full SU(2)_L transformation:** The special unitary group $SU(2)$ has 3 real parameters. A general element is:

$$
U(\vec{\theta}) = \exp\left(i \sum_{a=1}^3 \theta_a \frac{\sigma_a}{2}\right), \quad \vec{\theta} = (\theta_1, \theta_2, \theta_3) \in \mathbb{R}^3
$$
where $\{\sigma_a\}_{a=1}^3$ are Pauli matrices:

$$
\sigma_1 = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}, \quad \sigma_2 = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}, \quad \sigma_3 = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}
$$

**Real SO(2) subgroup (current implementation):** For real-valued neural networks, we restrict to the $SO(2)$ subgroup that performs real rotations in the $(\psi_{\text{obs}}, \psi_{\text{act}})$ plane. In the Pauli basis this corresponds to the one-parameter subgroup generated by $i\sigma_2$ (since $i\sigma_2 = \begin{psmallmatrix} 0 & 1 \\ -1 & 0 \end{psmallmatrix}$ is the canonical $\mathfrak{so}(2)$ generator). This gives 1-parameter transformations:

$$
U_{\text{SO(2)}}(\theta) = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix}, \quad \theta \in [0, 2\pi)
$$

**Action on doublet:**

$$
\Psi \to \Psi' = U_{\text{SO(2)}}(\theta) \Psi = \begin{pmatrix} \cos\theta & \sin\theta \\ -\sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}
$$

**Physical interpretation:**
- $\theta = 0$: Pure observation (no action planning)
- $\theta = \pi/2$: Pure action planning (no new observations)
- $0 < \theta < \pi/2$: Mixed observation-action processing

**Chirality:** This is a **left-chiral** doublet (incoming information stream). Right-chiral singlets correspond to executed actions (outgoing, no longer subject to mixing).

**Remark:** Full $SU(2)_L$ gauge theory requires complex-valued features. Current real-valued architectures (IsotropicBlock + SteerableConv) implement the $SO(2) \subset SU(2)$ subgroup. Extension to full 3-parameter $SU(2)$ requires complex steerable CNNs or quaternionic networks (see {ref}`sec-symplectic-multi-agent-field-theory` for gauge-theoretic derivation).
:::

:::{prf:theorem} Observation-Action Doublet as an $SO(2)$ (U(1)) Representation
:label: thm-steerable-induces-doublet

Let $\psi_{\text{obs}} \in \mathcal{Z}_{\text{obs}}$ and $\psi_{\text{act}} \in \mathcal{Z}_{\text{act}}$. Define the observation-action pair

$$
\Psi = \begin{pmatrix} \psi_{\text{obs}} \\ \psi_{\text{act}} \end{pmatrix}.
$$
Under the one-parameter subgroup $U_{\text{SO(2)}}(\theta)$ from Definition {prf:ref}`def-obs-action-doublet`, $\Psi$ transforms as $\Psi' = U_{\text{SO(2)}}(\theta)\Psi$, i.e., it is a 2D real representation of $SO(2)\cong U(1)$ (the implemented subgroup of $SU(2)_L$ in current real-valued architectures).

:::

:::{prf:definition} Latent Metric Tensor
:label: def-latent-metric

The latent space $\mathcal{Z}$ is equipped with the **Information Sensitivity Metric** (from Section {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

$$
G_{ij}(z) = \nabla^2_{ij} V(z) + \lambda \mathcal{F}_{ij}(z)
$$

where:
- $V(z)$ is the value function (expected return from state $z$)
- $\mathcal{F}_{ij}(z) = \mathbb{E}_{a \sim \pi(·|z)}[\nabla_{z_i} \log \pi(a|z) \nabla_{z_j} \log \pi(a|z)]$ is the Fisher Information Metric
- $\lambda > 0$ is the temperature parameter

**Physical interpretation:** The metric $G$ measures how sensitive the value function and policy are to changes in latent coordinates. High curvature regions indicate sensitive decision boundaries.

**Positive definiteness:** $G(z)$ is positive definite when $V$ is strongly convex and $\lambda > 0$.

**Units:** $[G_{ij}] = [\mathcal{Z}]^{-2} = \text{nat}^{-1}$ (inverse information).

**Variational origin** (sketch of derivation from Theorem {prf:ref}`thm-capacity-constrained-metric-law`):

The metric arises from minimizing the effective action under capacity constraints:

$$
G_{ij}(z) = \frac{\delta^2}{\delta z_i \delta z_j} \mathcal{A}_{\text{eff}}[z]
$$
where $\mathcal{A}_{\text{eff}}$ is the capacity-constrained effective action:

$$
\mathcal{A}_{\text{eff}}[z] = \int \left[V(z) + \frac{\lambda}{2} I(Z;A|z)\right] dz
$$

The **Hessian of the value function** $\nabla^2 V(z)$ captures curvature of the reward landscape (second-order approximation to the value surface), while the **Fisher Information Metric** $\mathcal{F}_{ij}(z)$ captures the sensitivity of the policy distribution $\pi(a|z)$ to latent perturbations.

**First variation** yields the Euler-Lagrange equations for optimal latent dynamics (geodesic equations on the WFR manifold). **Second variation** gives the metric as the Hessian of the action.

**Full derivation:** See Section 5.1 (Capacity-Constrained Metric Law) for the complete variational derivation from the bounded rationality Lagrangian, including the proof that $G$ is the unique metric satisfying the Monge-Ampère equation under holographic constraints.
:::

:::{prf:theorem} Composition of Equivariant Layers is Equivariant
:label: thm-composition-equivariant

Let $f_1, \ldots, f_L$ be $G$-equivariant layers. Then:

$$
F = f_L \circ \cdots \circ f_1 \text{ is } G\text{-equivariant}
$$

Moreover, if each $f_i$ has Lipschitz constant $L_i \leq 1$, then:

$$
L_F \leq 1 \quad \text{(global light cone preservation)}
$$

where $L_F$ is the Lipschitz constant of the composition $F$.

:::

:::{prf:lemma} NormGate Lipschitz Bound
:label: lem-normgate-lipschitz

Let $f(v) = v \cdot g(\|v\| + b)$ be the norm-gated activation (Definition {prf:ref}`def-norm-gated-activation`) where $g: \mathbb{R} \to \mathbb{R}$ is a smooth gating function with:
- $|g(x)| \leq C_g |x|$ for all $x$ (sublinear growth)
- $|g'(x)| \leq L_g$ for all $x$ (bounded derivative)

Assume an operating range $\|v\| \leq R_{\max}$ and a bounded bias $|b| \leq B$. Then $f$ is Lipschitz on the operating range with constant:

$$
L_f \leq \max\bigl(R_{\max}(C_g + L_g) + C_gB,\; C_g(R_{\max} + B)\bigr).
$$
In typical settings with $L_g > 0$ and $R_{\max} \gtrsim 1$, the radial term $R_{\max}(C_g + L_g) + C_gB$ dominates.

**For GELU:** The GELU function $g(x) = x\Phi(x)$ where $\Phi$ is the standard normal CDF satisfies:
- $C_g = 1$ (since $0 \leq \Phi(x) \leq 1$ implies $|g(x)| \leq |x|$)
- $g'(x) = \Phi(x) + x\phi(x)$ where $\phi(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$
- $\sup_{x \in \mathbb{R}} g'(x) \approx 1.129$ (achieved at $x^* = \sqrt{2} \approx 1.414$ where $g''(x^*) = 0$)
- For practical operating range $x \in [-3, 3]$: $\max_{x \in [-3,3]} g'(x) \approx 1.129$ (same critical point)

**Derivation of critical point:**
$$g''(x) = \frac{d}{dx}[\Phi(x) + x\phi(x)] = \phi(x) + \phi(x) - x^2\phi(x) = \phi(x)(2 - x^2)$$
Setting $g''(x) = 0$ yields $x^2 = 2$, so $x^* = \sqrt{2}$ (taking positive root). At this point: $g'(\sqrt{2}) = \Phi(\sqrt{2}) + \sqrt{2}\phi(\sqrt{2}) \approx 0.9214 + 0.2075 \approx 1.129$.

Thus $L_g \approx 1.129$ and

$$
L_f \leq R_{\max}(1 + 1.129) + B \approx 2.129\,R_{\max} + B.
$$

:::

:::{prf:definition} Micro-Macro Consistency
:label: def-micro-macro-consistency

A DNN layer $f: \mathcal{Z} \to \mathcal{Z}$ is **compatible with the geodesic integrator** if:

1. **Preserves metric (pullback):** Writing $J(z) := \frac{\partial f}{\partial z}(z)$, the pulled-back metric is

   $$
   (f^*G)(z) = J(z)^T\,G(f(z))\,J(z),
   $$
   and strict metric preservation (isometry) is $(f^*G)(z) = G(z)$.
2. **Bounded Lipschitz:** $\|f(z) - f(z')\| \leq L_f \cdot \|z - z'\|$ where $L_f$ is a finite constant
3. **Preserves gauge:** $f(U(g) \cdot z) = U(g) \cdot f(z)$ for all $g \in G_{\text{Fragile}}$

**Interpretation:**
- **Condition 1** ensures the metric structure evolves consistently (metric pullback)
- **Condition 2** ensures bounded signal propagation (Lipschitz continuity); for strict light cone preservation, require $L_f \leq 1$; for stable bounded amplification, allow $L_f = O(1)$ fixed constant
- **Condition 3** ensures gauge symmetry is preserved through network layers

**Units:** $[G] = [z]^{-2}$, $[L_f]$ dimensionless (ratio of distances), $[U(g)]$ dimensionless (unitary).
:::

:::{prf:theorem} Isotropic Blocks Satisfy Micro-Macro Consistency
:label: thm-isotropic-macro-compatible

IsotropicBlock (Definition {prf:ref}`def-isotropic-block`) satisfies:
1. **Bounded Lipschitz** (Condition 2 of {prf:ref}`def-micro-macro-consistency`) with $L_f \approx 9.5$ - proven rigorously
2. **Gauge invariance** (Condition 3 of {prf:ref}`def-micro-macro-consistency`) - proven rigorously
3. **Metric compatibility** (Condition 1 of {prf:ref}`def-micro-macro-consistency`) - pullback is well-defined, but exact isometry is not enforced (qualified)

:::

:::{prf:theorem} Metric Pullback Defect Under Composition
:label: thm-approximate-metric-preservation

Let $F = f_L \circ \cdots \circ f_1$ be a deep network on $\mathcal{Z}$, and let $G$ be a (possibly state-dependent) metric tensor on $\mathcal{Z}$.

Define $z_0 := z$, $z_\ell := f_\ell(z_{\ell-1})$, and Jacobians $J_\ell := \frac{\partial f_\ell}{\partial z}(z_{\ell-1})$. Let $J_F := J_L \cdots J_1$.

Define the per-layer **isometry defect**:

$$
E_\ell := J_\ell^T\,G(z_\ell)\,J_\ell - G(z_{\ell-1}).
$$

Then the total pullback defect admits the exact decomposition:

$$
J_F^T\,G(z_L)\,J_F - G(z_0)
=
\sum_{\ell=1}^L (J_{\ell-1}\cdots J_1)^T\,E_\ell\,(J_{\ell-1}\cdots J_1),
$$
with the convention $J_0\cdots J_1 := I$.

In particular, if $\|J_\ell\|_{\mathrm{op}} \leq L_J$ and $\|E_\ell\|_{\mathrm{op}} \leq e_\ell$, then:

$$
\|J_F^T\,G(F(z))\,J_F - G(z)\|_{\mathrm{op}} \leq \sum_{\ell=1}^L L_J^{2(\ell-1)}\,e_\ell.
$$

So when $L_J \le 1$ the defect accumulates at most linearly in $\sum_\ell e_\ell$, while for $L_J>1$ the naive worst-case bound grows like $L_J^{2L}$.

:::

:::{prf:proposition} Latent Dimension from Information-Theoretic First Principles
:label: prop-latent-dimension-from-capacity

The latent space $\mathcal{Z} \subset \mathbb{R}^{d_z}$ represents compressed observations encoding mutual information $I(X;Z) \leq C$ where $C$ is the channel capacity (nat/step) from the bounded rationality controller (Chapter 1).

**Derivation from Gaussian rate-distortion theory:**

For a Gaussian source $X \sim \mathcal{N}(0, \Sigma_X)$ encoded into latent $Z \in \mathbb{R}^{d_z}$ via encoder $p(Z|X)$ with reconstruction $\hat{X} = \mathbb{E}[X|Z]$:

**Step 1. Rate-distortion tradeoff:**
The optimal encoder for squared error distortion $D = \mathbb{E}[\|X - \hat{X}\|^2]$ achieves:

$$
I(X;Z) = \frac{1}{2}\sum_{i=1}^{d_z} \log\left(1 + \frac{\lambda_i}{\sigma^2}\right)
$$
where $\lambda_i$ are eigenvalues of the source covariance $\Sigma_X$ allocated to latent dimension $i$, and $\sigma^2$ is the noise level per dimension.

**Step 2. Equal allocation (isotropic latent):**
For computational efficiency, neural encoders typically use isotropic latent representations with equal variance per dimension:

$$
\Sigma_Z = \sigma_z^2 I_{d_z}
$$

The total information is:

$$
I(X;Z) \leq \frac{1}{2} d_z \log\left(1 + \frac{\sigma_X^2}{\sigma_{\text{noise}}^2}\right)
$$

**Step 3. Dimensional analysis from first principles:**

The mutual information formula (Step 1) is:

$$
I(X;Z) = \frac{1}{2}\sum_{i=1}^{d_z} \log\left(1 + \frac{\lambda_i}{\sigma^2}\right)
$$

**Logarithm constraint:** The logarithm function requires a dimensionless argument. Therefore:

$$
\left[1 + \frac{\lambda_i}{\sigma^2}\right] = [1] \quad \text{(dimensionless)}
$$

This implies:

$$
\left[\frac{\lambda_i}{\sigma^2}\right] = [1] \quad \Rightarrow \quad [\lambda_i] = [\sigma^2]
$$

**Information-theoretic foundation:** In Shannon information theory, differential entropy for a Gaussian random variable $X \sim \mathcal{N}(0, \Sigma)$ is **defined** to have units [nat]:

$$
h(X) = \frac{1}{2} \log \det(2\pi e \Sigma) \quad [\text{nat}]
$$

For a single dimension with variance $\lambda$:

$$
h(X_i) = \frac{1}{2}\log(2\pi e \lambda) \quad [\text{nat}]
$$

**Critical distinction:** The numerical prefactor $1/2$ is dimensionless (as all pure numbers are). The "nat" unit arises from the **operational definition** of information in Shannon's framework: entropy measures the expected log-probability, and we adopt the convention $[h(X)] = [\text{nat}]$ to distinguish information content from dimensionless logarithms. This is analogous to how we define $[E] = \text{Joule}$ in physics—it's a choice of unit system, not algebraic dimension propagation.

**Dimensional interpretation:** The formula should be understood as:

$$
h(X) = \left[\frac{1}{2}\log \det(2\pi e \Sigma)\right]_{\text{nat}}
$$
where the subscript indicates the dimensionless logarithm is **measured in units** of nats (the information-theoretic unit), not that it algebraically has dimension [nat].

**Dimensional convention (NOT derivation):** We **adopt the convention** that latent coordinates have dimension:

$$
\boxed{[z] = [\mathcal{Z}] := \sqrt{\text{nat}}}
$$

**This is a choice**, not a theorem. Here's why we make this choice:

**Step 3a. Consistency requirement:**
If we want variance $[\sigma_z^2]$ to have the **same units** as information measures (differential entropy $h(X)$ in nats), and if coordinates are related to variance by $[z^2] = [\sigma_z^2]$, then we must have:

$$
[z] = \sqrt{[\sigma_z^2]} = \sqrt{[\text{nat}]}
$$

**Step 3b. Motivation for the choice:**
This convention ensures dimensional consistency across the framework:
- Rate-distortion: $I(X;Z) = \frac{1}{2}\sum_i \log(\lambda_i/\sigma^2)$ requires $[\lambda_i] = [\sigma^2]$; setting both equal to [nat] makes $I$ dimensionally consistent with entropy
- Fisher metric: $\mathcal{F}_{ij} = \mathbb{E}[\partial_i \log p \, \partial_j \log p]$ has $[\mathcal{F}] = [z]^{-2}$; if $[z] = \sqrt{\text{nat}}$ then $[\mathcal{F}] = \text{nat}^{-1}$, matching information-theoretic quantities
- Capacity: $C$ (nat/step) becomes commensurate with $\|z\|^2$ (nat) and $d_z \times \sigma_z^2$ (dimensionless × nat)

**Step 3c. What is NOT claimed:**
- We do **not** claim this follows logically from information theory alone
- We do **not** claim $[\lambda] = [\text{nat}]$ is forced by mathematics—it's a **definition** we choose
- This is analogous to setting $c = 1$ in relativity: a **unit convention** that simplifies equations, not a physical law

**Interpretation:** Each latent coordinate carries information measured in natural units (nats). The variance of a latent dimension represents information content. This convention parallels quantum mechanics where position $x$ relates to momentum $p$ via $\Delta x \Delta p \sim \hbar$ with $[\hbar] = \text{action}$ giving $[x] \sim \sqrt{\text{action}}$.

**Remark on arbitrariness:** This **is** an arbitrary convention in the sense that we could equally well work in dimensionless units throughout and track "nat" as a label rather than a dimension. We choose to promote "nat" to a pseudo-dimension because:
1. It makes dimensional analysis track information flow explicitly
2. It connects latent-space geometry to information-theoretic foundations
3. It parallels established physics conventions (action, angular momentum as dimensional units)
:::

:::{prf:definition} Information Speed in Latent Coordinates
:label: def-information-speed-latent

The **latent information speed** is the maximum rate of latent state change per unit time:

$$
c_{\mathcal{Z}} := \sup_{z(·), \Delta t > 0} \frac{d_{\mathcal{Z}}(z(t + \Delta t), z(t))}{\Delta t}
$$

where:
- $z: [0, T] \to \mathcal{Z}$ is a latent trajectory
- $d_{\mathcal{Z}}(z_1, z_2) = \|z_1 - z_2\|$ is the Euclidean distance in latent space
- The supremum is taken over all admissible trajectories and time increments

**Dimensions**: $[c_{\mathcal{Z}}] = [\mathcal{Z}][T^{-1}] = \sqrt{\text{nat}} \cdot [T^{-1}]$ where $[T]$ denotes abstract time dimension (measured in seconds for physical agents)

**Physical interpretation**: This is the "speed of thought"—the maximum rate at which the agent's internal representation can evolve under the dynamics.

**Connection to environment information speed**:

Let $\mathcal{E}$ denote the environment observation space (e.g., pixel space for vision, $\mathcal{E} = \mathbb{R}^{H \times W \times C}$).

Let $\phi: \mathcal{E} \to \mathcal{Z}$ be the encoder network mapping observations to latents, with Jacobian $J_\phi(x) = \nabla \phi(x) \in \mathbb{R}^{d_z \times d_{\mathcal{E}}}$.

By the chain rule for composed dynamics $z(t) = \phi(x(t))$:

$$
\frac{dz}{dt} = J_\phi(x(t)) \cdot \frac{dx}{dt}
$$

Taking norms:

$$
\left\|\frac{dz}{dt}\right\| \leq \|J_\phi(x)\|_{\text{op}} \cdot \left\|\frac{dx}{dt}\right\|
$$

If the environment dynamics satisfy $\|dx/dt\| \leq c_{\text{info}}$ (Axiom {prf:ref}`ax-information-speed-limit` from Chapter 8.1), then:

$$
c_{\mathcal{Z}} \leq \sup_x \|J_\phi(x)\|_{\text{op}} \cdot c_{\text{info}}
$$

**Remark**: The encoder Lipschitz constant $L_\phi = \sup_x \|J_\phi(x)\|_{\text{op}}$ controls how environmental changes propagate to latent space. Spectral normalization ensures the **linear** parts satisfy $\|W\|_{\text{op}}=\sigma_{\max}(W)\leq 1$; the overall $L_\phi$ is then bounded by the product of per-block Lipschitz bounds, so strict $L_\phi \leq 1$ requires also using 1-Lipschitz nonlinearities (or explicitly tracking/rescaling their gain).

**Operational constraint**: For strict causality preservation (Theorem {prf:ref}`thm-spectral-preserves-light-cone`), every **linear** map must satisfy $\sigma_{\max}(W) \leq 1$ and each nonlinear block should satisfy $L_f \leq 1$; if bounded amplification is allowed, track the resulting global Lipschitz bound to maintain a controlled speed limit.
:::

:::{prf:proposition} Dimensional Consistency of IsotropicBlock
:label: prop-dimensional-consistency

Let $z \in \mathcal{Z}$ with $[z] = [\mathcal{Z}] = \sqrt{\text{nat}}$ (Proposition {prf:ref}`prop-latent-dimension-from-capacity`). The IsotropicBlock operation

$$
\text{IsotropicBlock}(z) = \text{Reshape}(\text{NormGate}(\text{SpectralLinear}(z)))
$$
preserves the latent dimension $[\mathcal{Z}]$ through each stage when interpreted with implicit normalization conventions.

:::

:::{prf:definition} Gauge Violation Metric
:label: def-gauge-violation-metric

For operator $f$ and group element $g \in G$:

$$
\delta_{\text{gauge}}(f, g) = \mathbb{E}_z\left[\|f(U(g) \cdot z) - U(g) \cdot f(z)\|^2\right]
$$

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}} = 10^{-4}$ (empirically tuned).
:::

## 08_multiagent/05_architecture.md

:::{prf:proposition} Failure Modes of Flat World Models
:label: prop-failure-modes-flat-world-models

A world model $f: \mathcal{Z} \times \mathcal{A} \to \mathcal{Z}$ implemented as a standard neural network (GRU, MLP, Transformer) built from flat-space operations does not, in general, *guarantee* preservation of:

1. **Metric structure**: The capacity-constrained metric $G(z)$ from Theorem {prf:ref}`thm-capacity-constrained-metric-law` implies position-dependent step sizes. Flat operations use constant step sizes.

2. **Gauge covariance**: Under local gauge transformation $\psi \to U(z)\psi$, predictions must transform covariantly. Flat operations are not gauge-aware.

3. **Symplectic structure**: The phase space $(\mathcal{Z} \times T^*\mathcal{Z}, \omega)$ has a conserved 2-form. Flat operations generically break symplectic conservation.

4. **Boundary constraints**: In the Poincare ball/disk model ($|z|<1$), the metric diverges as $|z| \to 1$, enforcing vanishing physical step size near the boundary. Flat operations can produce invalid states unless constrained explicitly.

*Consequence*: Flat world models require extensive regularization to approximately enforce these constraints, with no guarantee of exact satisfaction.

:::

:::{prf:definition} Lorentz-Langevin SDE (Recap)
:label: def-lorentz-langevin-recap

From Definition {prf:ref}`def-bulk-drift-continuous-flow`, the position coordinates evolve as:

$$
dz^k = \underbrace{\left( -G^{kj}\partial_j \Phi + u_\pi^k \right)}_{\text{gradient + control}} ds + \underbrace{\beta_{\text{curl}} G^{km} \mathcal{F}_{mj} \dot{z}^j ds}_{\text{Lorentz force}} - \underbrace{\Gamma^k_{ij}\dot{z}^i \dot{z}^j ds}_{\text{geodesic correction}} + \underbrace{\sqrt{2T_c}(G^{-1/2})^{kj} dW^j_s}_{\text{thermal noise}}
$$

where:
- $G^{kj}$ is the inverse metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
- $\Phi$ is the effective potential (Definition {prf:ref}`def-effective-potential`)
- $\mathcal{F}_{mj}$ is the Value Curl tensor (Definition {prf:ref}`def-value-curl`)
- $\Gamma^k_{ij}$ are Christoffel symbols of the Levi-Civita connection
- $T_c$ is the cognitive temperature ({prf:ref}`def-cognitive-temperature`)

:::

:::{prf:definition} Covariant Derivative (Recap)
:label: def-covariant-derivative-recap

From Theorem {prf:ref}`thm-emergence-opportunity-field`, the gauge-covariant derivative for the full gauge group $G_{\text{Fragile}} = SU(N_f)_C \times SU(2)_L \times U(1)_Y$ is:

$$
D_\mu = \partial_\mu - i g_s \frac{\lambda^a}{2} G_\mu^a - i g_2 \frac{\sigma^b}{2} W_\mu^b - i g_1 \frac{Y}{2} B_\mu
$$

where:
- $G_\mu^a$ ($a = 1, \ldots, N_f^2-1$) is the Binding field (Theorem {prf:ref}`thm-emergence-binding-field`)
- $W_\mu^b$ ($b = 1, 2, 3$) is the Error field (Theorem {prf:ref}`thm-emergence-error-field`)
- $B_\mu$ is the Opportunity field (Theorem {prf:ref}`thm-emergence-opportunity-field`)
- $\lambda^a, \sigma^b$ are generators of $SU(N_f)$ and $SU(2)$ respectively (we use $\sigma$ to avoid clashing with the softmax temperature $\tau(z)$ below)
- $g_s, g_2, g_1$ are coupling constants
- $Y$ is the hypercharge

:::

:::{prf:definition} Wilson Line
:label: def-wilson-line

The **Wilson line** (parallel transport operator) along a path $\gamma$ from $z_0$ to $z$ is:

$$
U_\gamma(z, z_0) = \mathcal{P}\exp\left(-i\int_\gamma A_\mu dx^\mu\right)
$$

where:
- $\mathcal{P}$ denotes path ordering
- $A_\mu = g_s \frac{\lambda^a}{2} G_\mu^a + g_2 \frac{\sigma^b}{2} W_\mu^b + g_1 \frac{Y}{2} B_\mu$ is the total gauge connection

For infinitesimal paths $\gamma: z_0 \to z_0 + \delta z$:

$$
U(z_0 + \delta z, z_0) \approx I - i A_\mu(z_0) \delta z^\mu + O(\delta z^2)
$$

*Gauge transformation*: Under $\psi(z) \to \Omega(z)\psi(z)$, the Wilson line transforms as:

$$
U_\gamma(z, z_0) \to \Omega(z) U_\gamma(z, z_0) \Omega^\dagger(z_0)
$$

This ensures that $U_\gamma(z, z_0) \psi(z_0)$ transforms correctly at $z$.

:::

:::{prf:definition} Poincare Ball/Disk Metric (Recap)
:label: def-poincare-metric-recap

The capacity-constrained metric on the Poincare ball (disk when $d=2$) $\mathbb{D}^d = \{z \in \mathbb{R}^d : |z| < 1\}$ is:

$$
G_{ij}(z) = \lambda(z)^2 \delta_{ij} = \frac{4}{(1-|z|^2)^2} \delta_{ij}
$$

where $\lambda(z) = 2/(1-|z|^2)$ is the **conformal factor**.

**Key properties**:
- As $|z| \to 1$: $\lambda(z) \to \infty$ (metric diverges at boundary)
- At origin $z = 0$: $\lambda(0) = 2$ (minimal metric)
- Inverse metric: $G^{ij}(z) = \lambda(z)^{-2} \delta^{ij} = \frac{(1-|z|^2)^2}{4} \delta^{ij}$

The **Christoffel symbols** for this metric are (Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`):

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right)
$$

:::

:::{prf:definition} Covariant Query-Key-Value Projections
:label: def-covariant-qkv-projections

Let $\psi_{\text{obs}}(z)$ be the observation latent and $\psi_{\text{act}}(z')$ be the action latent at positions $z, z' \in \mathcal{Z}$. The **covariant projections** are:

$$
\begin{aligned}
Q(z) &= \Pi_Q \cdot U_{z \to 0} \cdot D_\mu \psi_{\text{obs}}(z) \\
K(z') &= \Pi_K \cdot U_{z' \to 0} \cdot D_\nu \psi_{\text{act}}(z') \\
V(z') &= \Pi_V \cdot U_{z' \to 0} \cdot \psi_{\text{act}}(z')
\end{aligned}
$$

where:
- $U_{z \to 0} := U_\gamma(0, z)$ is the Wilson line transporting from $z$ to the origin (Definition {prf:ref}`def-wilson-line`)
- $D_\mu$ is the covariant derivative (Definition {prf:ref}`def-covariant-derivative-recap`)
- $\Pi_Q, \Pi_K, \Pi_V$ are learnable projection maps that act on feature indices (equivariantly on gauge indices) so they commute with gauge transformations at the reference point

**Interpretation**:
- The Wilson line $U_{z \to 0}$ parallel-transports the field to a common reference frame at the origin
- The covariant derivative $D_\mu$ ensures the derivative is gauge-covariant
- Queries use the derivative $D_\mu \psi$ (sensitivity to position change)
- Values use the field $\psi$ directly (the actual content to retrieve)

:::

:::{prf:theorem} Gauge Invariance of Covariant Cross-Attention
:label: thm-gauge-invariance-cross-attention

Let the attention score be computed as:

$$
\alpha(z, z') = \text{softmax}_{z'}\left(\frac{\operatorname{Re}\left(Q(z)^\dagger K(z')\right)}{\tau(z)}\right)
$$

where $Q$ and $K$ are defined with Wilson line preprocessing (Definition {prf:ref}`def-covariant-qkv-projections`), transporting both to a common reference point (the origin). (If the representation is purely real/orthogonal, replace $^\dagger$ with transpose and drop $\operatorname{Re}(\cdot)$.)

Then $\alpha(z, z')$ is **gauge-invariant**: under local gauge transformation $\psi \to \Omega(x)\psi$, the attention score is unchanged.

:::

:::{prf:proposition} Wilson Line Approximation for Attention
:label: prop-wilson-line-approximation

For attention between positions $z$ and $z'$ with $|z - z'| \ll 1$, the Wilson line can be approximated as:

$$
U(z, z') \approx I - i A_\mu(\bar{z}) (z - z')^\mu
$$

where $A_\mu$ is the total gauge connection from Definition {prf:ref}`def-wilson-line`, $\bar{z}$ is any point along the path (choices differ only at $O(|z-z'|^2)$), and the contraction $A_\mu(\bar{z}) (z - z')^\mu$ is the path-directional connection.

In the attention mechanism, this becomes a **relative position encoding**:

$$
\text{score}(z, z') := \operatorname{Re}\left(Q(z)^\dagger U(z, z') K(z')\right)
\approx \operatorname{Re}\left(Q(z)^\dagger K(z')\right) + \operatorname{Re}\left(-i\,Q(z)^\dagger \left[A_\mu(\bar{z}) (z - z')^\mu\right] K(z')\right)
$$

The second term is the **gauge correction** to the attention score.

:::

:::{prf:theorem} Metric-Temperature Correspondence
:label: thm-metric-temperature-correspondence

Let the attention mechanism use position-dependent temperature $\tau(z)$:

$$
\alpha(z, z') = \text{softmax}_{z'}\left(\frac{s(z, z')}{\tau(z)}\right)
$$

where $s(z,z')$ is any real-valued score (for example $s(z,z')=\operatorname{Re}(Q(z)^\dagger K(z'))$).

The choice

$$
\tau(z) = \frac{\sqrt{d_k}}{\lambda(z)} = \sqrt{d_k} \cdot \frac{1-|z|^2}{2}
$$

where $\lambda(z) = 2/(1-|z|^2)$ is the conformal factor, implies:

1. **Metric encoding (conformal case)**: for $G(z)=\lambda(z)^2 I$, the metric scale is recovered exactly as $G(z)=\tfrac{d_k}{\tau(z)^2}I$
2. **Boundary sharpening**: $\tau(z) \to 0$ as $|z| \to 1$, so $\text{softmax}(s/\tau)$ concentrates on the argmax for fixed scores $s(z,\cdot)$

:::

:::{prf:proposition} Mass–Metric–Temperature Identity
:label: prop-mass-metric-inverse-temperature

The Mass = Metric principle (Definition {prf:ref}`def-mass-tensor`) extends to attention:

$$
\mathbf{M}(z) = G(z) = \lambda(z)^2 I = \frac{d_k}{\tau(z)^2} I
$$

**Implication (conformal case)**: Large $\lambda(z)$ (large metric/mass scale) corresponds to small $\tau(z)$ (sharper softmax scaling).

:::

:::{prf:definition} Geodesic Query Projection
:label: def-geodesic-query-projection

The **Geodesic Query** extends the linear projection to include geometric terms that encode the Levi-Civita connection (with optional velocity-conditioned corrections when you want to go beyond Levi-Civita):

$$
Q_{\text{geo}}(x, z, v) = W_Q x + W_{Qz} z + W_{Qv} x_v + W_{Q,\Gamma}(z, z) + W_{Qzv}(z, v)
$$

where:
- $W_Q \in \mathbb{R}^{d_k \times d_{\text{model}}}$ is the feature projection
- $W_{Qz} \in \mathbb{R}^{d_k \times d}$ maps geometric coordinates
- $W_{Qv} \in \mathbb{R}^{d_k \times d_{\text{model}}}$ projects velocity/momentum features $x_v = \phi_v(v)$
- $W_{Q,\Gamma} \in \mathbb{R}^{d_k \times d \times d}$ is a 3-tensor encoding position-position quadratic terms
- $W_{Qzv} \in \mathbb{R}^{d_k \times d \times d}$ encodes optional position-velocity coupling

Here $x_v = \phi_v(v)$ is a feature embedding of velocity/momentum.

**Notation**: $(A, B)$ denotes bilinear contraction: $W_{Q,\Gamma}(z, z) = \sum_{ij} W_{Q,\Gamma}^{a,ij} z^i z^j$ and $W_{Qzv}(z, v) = \sum_{ij} W_{Qzv}^{a,ij} z^i v^j$.

The $W_{Qzv}$ term is optional; for Levi-Civita connections it can be omitted. In practice, a Hadamard coupling $z \odot v$ followed by a linear map is often sufficient.

**Christoffel Encoding**: Use $W_{Qz}$ to capture the linear-in-$z$ part of $\Gamma(z)$ near a reference point $z_0$, and use $W_{Q,\Gamma}$ for nonlinear corrections with learnable position dependence. Both can be initialized from the Poincare structure and refined during training.

:::

:::{prf:theorem} Geodesic Correction Representability via Attention
:label: thm-geodesic-correction-attention

Let the Query and Key be:

$$
\begin{aligned}
Q(x, z, v) &= W_Q x + W_{Qz} z + W_{Qv} x_v + W_{Q,\Gamma}(z, z) \\
K(z', v') &= W_K z' + W_{Kv} v'
\end{aligned}
$$

For clarity, we omit the optional $W_{Qzv}(z, v)$ term; it adds a velocity-conditioned correction without changing the representability argument.

The attention-weighted output

$$
\Delta z = \sum_{z'} \alpha(z, z') V(z')
$$

can represent the geodesic correction term $-\Gamma^k_{ij}(z) v^i v^j$ when:

1. The Values include quadratic velocity features (e.g., $V(z',v') = W_V\,\text{vec}(v' \otimes v')$ or a low-rank factorization)
2. The Query provides a learned parameterization of the (symmetric) coefficients $\Gamma(z)$ via the geometric terms ($W_{Qz}$, $W_{Q,\Gamma}$)
3. The context provides velocities in a neighborhood of the current velocity so the quadratic form is sampled/approximated locally

:::

:::{prf:proposition} Explicit Christoffel Encoding for Poincare Ball/Disk
:label: prop-christoffel-encoding-poincare

For the Poincare ball/disk with Christoffel symbols (Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`):

$$
\Gamma^k_{ij}(z) = \frac{2}{1-|z|^2}\left(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k\right)
$$

A practical initialization is to use a linear geometric term $W_{Qz} z$ to reproduce the linear-in-$z$ structure above (up to a sign convention that can be absorbed into the update rule), and reserve $W_{Q,\Gamma}(z,z)$ for nonlinear corrections. The conformal factor $2/(1-|z|^2)$ is position-dependent and cannot be represented by a constant $W_{Qz}$ alone; capture it via $W_{Q,\Gamma}$ or an explicit scalar modulation.

**Learnable approximation**: Since $\Gamma$ depends on position, a simple parameterization is to scale the nonlinear correction by the conformal factor:

$$
Q_\Gamma(z) = W_{Qz} z + \frac{2}{1-|z|^2}\,\tilde{W}_{Q,\Gamma}(z, z)
$$

where $\tilde{W}_{Q,\Gamma}$ is a learnable tensor initialized to approximate higher-order corrections and then refined by training.

:::

:::{prf:definition} Observation-Action Doublet in Attention
:label: def-observation-action-doublet-attention

The attention mechanism operates on **doublet-valued** representations:

$$
\Psi_L(z) = \begin{pmatrix} \psi_{\text{obs}}(z) \\ \psi_{\text{act}}^{\text{pre}}(z) \end{pmatrix} \in \mathbb{C}^{2d}
$$

The **Query** extracts observation information, the **Key** encodes action information, both with Wilson line transport to origin (cf. Definition {prf:ref}`def-covariant-qkv-projections`):

$$
\begin{aligned}
Q_{\text{obs}}(z) &= \Pi_{\text{obs}} \cdot U_{z \to 0} \cdot D_\mu \Psi_L(z) = \begin{pmatrix} 1 & 0 \end{pmatrix} U_{z \to 0} D_\mu \Psi_L \\
K_{\text{act}}(z') &= \Pi_{\text{act}} \cdot U_{z' \to 0} \cdot D_\nu \Psi_L(z') = \begin{pmatrix} 0 & 1 \end{pmatrix} U_{z' \to 0} D_\nu \Psi_L
\end{aligned}
$$

The **cross-attention** score between observation and action is (gauge-invariant by Theorem {prf:ref}`thm-gauge-invariance-cross-attention`):

$$
\alpha_{\text{cross}}(z, z') = \text{softmax}\left(\frac{\operatorname{Re}\left(Q_{\text{obs}}(z)^\dagger K_{\text{act}}(z')\right)}{\tau(z)}\right)
$$

*Interpretation*: The observation at $z$ attends to actions at $z'$. Both are transported to a common reference point, ensuring gauge-invariant comparison.

:::

:::{prf:definition} Chiral Projector from Value Gradient
:label: def-chiral-projector-value-gradient

The **chiral projector** extracts committed actions from the observation-action doublet using a unit $SU(2)$ direction derived from the value gradient:

$$
\hat{n}(z) = \frac{P \nabla V(z)}{\|P \nabla V(z)\|}
$$

where $P: \mathbb{R}^d \to \mathbb{R}^3$ is a learned projection and $\vec{\sigma} = (\sigma_1, \sigma_2, \sigma_3)$ are Pauli matrices (generators of $SU(2)$).

The **projection operator** is:

$$
\Pi_{\text{chirality}}(z) = \frac{1}{2}\left(I_2 + \hat{n}(z) \cdot \vec{\sigma}\right)
$$

The **committed action** is:

$$
\psi_{\text{act}}^{\text{commit}}(z) = \Pi_{\text{chirality}}(z) \cdot \Psi_L(z)
$$

The **commitment strength** (gauge-invariant under $SU(2)$, per feature channel) is:

$$
c(z) = \Psi_L(z)^\dagger \Pi_{\text{chirality}}(z) \Psi_L(z)
$$

**Properties**:
- $\Pi_{\text{chirality}}^2 = \Pi_{\text{chirality}}$ (idempotent)
- $\text{Tr}(\Pi_{\text{chirality}}) = 1$ (rank-1 projector)
- Under $SU(2)$ transformation $\Psi_L \to U\Psi_L$: if $\hat{n}$ is constructed as an adjoint vector, then $\hat{n} \to U\hat{n}U^\dagger$, preserving gauge covariance

**Degeneracy**: When $\|P \nabla V\| \to 0$ (flat value landscape), $\hat{n}$ is undefined. The agent should not commit in ambiguous regions.

:::

:::{prf:theorem} Gauge Covariance of Chiral Projection
:label: thm-gauge-covariance-chiral-projection

The commitment strength $c(z) = \Psi_L^\dagger \Pi_{\text{chirality}} \Psi_L$ is invariant under local $SU(2)_L$ transformations. The projected vector $\Pi_{\text{chirality}} \Psi_L$ transforms covariantly (in the fundamental representation), but gauge-invariant observables are obtained by contracting the $SU(2)$ indices.

:::

:::{prf:definition} Area Law Screening in Attention
:label: def-area-law-screening-attention

The **screened attention score** between positions $z$ and $z'$ at representation level $\ell$ is:

$$
\alpha_{\text{screened}}(z, z'; \ell) = \alpha_{\text{bare}}(z, z') \cdot \exp\left(-\sigma(\ell) \cdot A_{\text{string}}(z, z')\right)
$$

where:
- $\alpha_{\text{bare}}$ is the gauge-covariant attention score from Theorem {prf:ref}`thm-gauge-invariance-cross-attention`
- $\sigma(\ell)$ is the **string tension** at level $\ell$, with $\sigma(\ell) \propto g_s^2(\ell)$ (binding coupling squared)
- $A_{\text{string}}(z, z')$ is an **area proxy** used for screening (often quadratic in separation)

In practice, after applying screening one renormalizes $\alpha_{\text{screened}}(z,\cdot;\ell)$ over $z'$ so the weights sum to 1.

**Area approximation**: For nearby points in flat metric:

$$
A_{\text{string}}(z, z') \approx \frac{1}{2}|z - z'|^2
$$

For the Poincare ball/disk with conformal factor $\lambda$:

$$
A_{\text{string}}(z, z') \approx \frac{\lambda(z)^2}{2}|z - z'|^2
$$

:::

:::{prf:theorem} Texture Confinement via Area Law Screening
:label: thm-texture-confinement-area-law

Let the representation hierarchy have levels $\ell = 0$ (macro) to $\ell = L$ (texture), with running coupling $g_s(\ell)$ satisfying asymptotic freedom (Definition {prf:ref}`def-coupling-function`):

$$
g_s(\ell) \to 0 \text{ as } \ell \to L \quad (\text{UV, texture level})
$$

$$
g_s(\ell) \to g_s^{\text{crit}} \text{ as } \ell \to 0 \quad (\text{IR, macro level})
$$

Then:

1. **Texture-to-texture attention** ($\ell = L$): $\sigma(L) \to 0$, no screening. Features interact freely at texture level.

2. **Macro-to-texture attention** ($\ell = 0$ attending to $\ell = L$): $\sigma(0) > \sigma_{\text{crit}}$, strong screening. Texture is inaccessible from macro level.

3. **Gauge-singlet access**: Channels transforming in the trivial (color-neutral) representation of $SU(N_f)$ can be exempted from screening, allowing macro-level access to bound-state (concept) features while suppressing color-charged texture.

:::

:::{prf:proposition} Confinement Radius from String Tension
:label: prop-confinement-radius-string-tension

Assuming the local proxy $A_{\text{string}}(z,z') \approx d_G(z,z')^2/2$, the **confinement radius** $r_{\text{conf}}$ (in geodesic distance) is the scale at which screening suppresses attention by a factor $e^{-1}$:

$$
r_{\text{conf}}(\ell) = \sqrt{\frac{2}{\sigma(\ell)}}
$$

At macro level with $\sigma(0) \approx 1$: $r_{\text{conf}}(0) \approx \sqrt{2}$ (order-unity in geodesic units).

At texture level with $\sigma(L) \approx 0.01$: $r_{\text{conf}}(L) \approx 14$ (large, allowing texture-to-texture interaction).

*Interpretation*: Weak screening at texture allows long-range texture-to-texture interaction, while strong screening suppresses macro access to color-charged texture channels.

:::

:::{prf:definition} BAOAB Steps (Attention Heads + OU)
:label: def-baoab-attention-heads

The **GeodesicCrossAttention** module implements B-A-O-A-B with attention heads for B/A and a closed-form OU step in the middle (or an optional learned O-head):

**Step 1 (B-head 1): B-step (First half-kick)**
- **Query**: Current position $z$ with quadratic geodesic terms
- **Key**: Gradient bank $\{\nabla\Phi(z')\}_{z' \in \text{context}}$
- **Value**: Gradient vectors $\{\nabla\Phi(z')\}$
- **Output**: Momentum update $\Delta p_1 = -\frac{h}{2}\nabla\Phi(z)$

**Step 2 (A-head 1): A-step (First half-drift)**
- **Query**: Current position and momentum $(z, p)$
- **Key**: Exponential map or transport bank $\{\exp_{z'}(v)\}$
- **Value**: Displacement corrections
- **Output**: Drift correction $\Delta z_1$ added to the explicit drift $\frac{h}{2}G^{-1}(z)p$

**Step 3 (OU): O-step (Ornstein-Uhlenbeck thermostat)**
- **Default**: Closed-form OU update (no attention)
- **Optional learned thermostat**:
  - **Query**: Current momentum $p$
  - **Key**: Noise bank (random vectors)
  - **Value**: Noise vectors $\{\xi\}$
  - **Output**: Residual correction added to the OU update

**Step 4 (A-head 2): A-step (Second half-drift)**
- Same structure as Step 2
- **Output**: Drift correction $\Delta z_2$ added to the explicit drift $\frac{h}{2}G^{-1}(z)p$

**Step 5 (B-head 2): B-step (Second half-kick)**
- Same structure as Step 1
- **Output**: Momentum update $\Delta p_2 = -\frac{h}{2}\nabla\Phi(z)$

**Composition**: The full update is:

$$
(z_{t+1}, p_{t+1}) = \text{Step}_5 \circ \text{Step}_4 \circ \text{Step}_3 \circ \text{Step}_2 \circ \text{Step}_1(z_t, p_t)
$$

Here $\text{Step}_3$ denotes the OU operator unless a learned thermostat head is enabled.

:::

:::{prf:theorem} BAOAB Attention Targets Boltzmann Distribution (Idealized)
:label: thm-baoab-attention-boltzmann

In the idealized setting where the attention heads recover the BAOAB substeps (kick/drift) and the O-step is the exact OU update, the GeodesicCrossAttention module reduces to the standard BAOAB integrator (Definition {prf:ref}`def-baoab-splitting`). In that limit, it targets the Gibbs/Boltzmann density

$$
\rho(z, p) \propto \exp\left(-\frac{\Phi_{\text{eff}}(z)}{T_c} - \frac{\|p\|_G^2}{2T_c}\right)
$$

as the intended stationary distribution (cf. Proposition {prf:ref}`prop-baoab-preserves-boltzmann`), provided:

1. Each head uses position-dependent temperature $\tau(z) = \sqrt{d_k}/\lambda(z)$
2. The O-step uses thermalization coefficients $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1-c_1^2)T_c}$
3. The geodesic Query projections correctly encode Christoffel symbols
4. The A-steps include the explicit drift $G^{-1}(z)p$ (with any attention-based correction consistent with the exponential map)

:::

:::{prf:proposition} Keys as Gradient Bank
:label: prop-keys-as-force-bank

In the B-step attention heads, the Keys store precomputed gradients of the effective potential at context positions:

$$
K^{(\text{grad})}_{z'} = W_K \cdot \nabla\Phi_{\text{eff}}(z')
$$

The attention score $s(z, z') := \operatorname{Re}\left(Q(z)^\dagger K(z')\right)$ measures how aligned the current position is with the gradient at $z'$. The weighted sum of Values retrieves the local gradient estimate:

$$
\widehat{\nabla\Phi}(z) = \sum_{z' \in \text{context}} \alpha(z, z') \cdot V^{(\text{grad})}(z') = \nabla\Phi_{\text{eff}}(z) + O(\text{interpolation error})
$$

The B-step applies the negative sign: $\Delta p = -\frac{h}{2}\widehat{\nabla\Phi}(z)$.

*Advantage*: Precomputing gradients at context positions amortizes the cost of gradient computation across multiple queries.

:::

:::{prf:proposition} Values as State Updates
:label: prop-values-as-state-updates

In each attention head, the Values encode the update to apply (the OU step is closed-form unless you enable a learned thermostat):

| Head | Value Content | Update |
|:-----|:-------------|:-------|
| B (kick) | $\nabla\Phi$ | $\Delta p$ (with negative sign applied in the update) |
| A (drift) | Displacement correction | $\Delta z$ (added to explicit $G^{-1}p$ drift) |
| O (thermostat) | $c_2\,G^{1/2}\xi$ (OU) | Noise injection (optional learned residual) |

The attention-weighted Value sum produces the correction update, which is added to the explicit drift or momentum update as appropriate.

:::

:::{prf:proposition} Complexity of Naive Implementation
:label: prop-complexity-naive-implementation

The full covariant cross-attention has the following complexity breakdown:

| Component | Naive Complexity | Bottleneck |
|:----------|:-----------------|:-----------|
| Wilson line computation | $O(N^2 d^2)$ | Path integral for each pair |
| Attention scores | $O(N^2 d)$ | All-pairs dot product |
| Quadratic Query | $O(N d^2)$ | Christoffel tensor contraction |
| Area law screening | $O(N^2)$ | String area for each pair |
| Chiral projection | $O(N d)$ | Per-position projection |
| **Total** | $O(N^2 d^2)$ | Dominated by Wilson lines |

For $N = 1000$, $d = 64$: approximately $4 \times 10^9$ operations per layer.

:::

:::{prf:proposition} Gauge-Locality Correspondence (Practical Bound)
:label: thm-gauge-locality-correspondence

Assume area-law screening is applied as a multiplicative factor $\exp(-\sigma A_{\text{string}}(z,z'))$ and use the local approximation
$A_{\text{string}}(z,z') \approx d_G(z,z')^2/2$ (cf. Definition {prf:ref}`def-area-law-screening-attention` and the small-distance relation $d_G(z,z')\approx \lambda(z)\|z-z'\|$).

Then for any tolerance $\epsilon \in (0,1)$, the screening factor satisfies:

$$
\exp\left(-\sigma A_{\text{string}}(z,z')\right) \le \epsilon
\quad\text{whenever}\quad
d_G(z,z') \ge r_{\epsilon} := \sqrt{\frac{2\log(1/\epsilon)}{\sigma}}.
$$

*Consequence*: Beyond $r_\epsilon$, the screened (unnormalized) weights are exponentially small, motivating sparse neighborhoods. Separately, the Wilson-line linearization is accurate only up to a radius $r_{\text{Wilson}}$, so practical sparse neighborhoods should also enforce $d_G(z,z') \lesssim r_{\text{Wilson}}$.

:::

:::{prf:definition} Geodesic Sparse Attention
:label: def-geodesic-sparse-attention

Replace full attention with **geodesic-local sparse attention**:

$$
\alpha_{\text{sparse}}(z, z') = \begin{cases}
\alpha_{\text{full}}(z, z') & \text{if } d_G(z, z') \leq r_{\epsilon} \\
0 & \text{otherwise}
\end{cases}
$$

In practice, renormalize $\alpha_{\text{sparse}}(z,\cdot)$ over the retained neighbors so the weights sum to 1.

**Implementation**: Use a spatial data structure (k-d tree, ball tree, or locality-sensitive hashing) to find the $k$-nearest neighbors in geodesic distance.

**Complexity**: $O(N k d)$ where $k$ is the neighborhood size, typically $k \sim 32$-$128$.

**Gauge-faithfulness**: Exact within the retained neighborhood (up to any Wilson-line approximation used there). If $m_{\text{drop}}(z) := \sum_{d_G(z,z')>r_\epsilon} \alpha_{\text{full}}(z,z')$, then truncating and renormalizing changes the attention distribution by total variation distance at most $m_{\text{drop}}(z)$.

:::

:::{prf:definition} Linearized Covariant Attention
:label: def-linearized-covariant-attention

Approximate the softmax attention kernel using a positive random feature map (a Monte Carlo approximation):

$$
\exp\left(\frac{s}{\tau}\right)
\quad\text{with}\quad
s := \operatorname{Re}\left(Q^\dagger K\right)
\approx \phi(Q/\tau)^T \phi(K)
$$

where $\phi: \mathbb{R}^{d_k} \to \mathbb{R}^D$ is a random feature map with $D \ll N$:

$$
\phi(x) = \frac{e^{-\|x\|^2/2}}{\sqrt{D}} \begin{pmatrix} \exp(\omega_1^T x) \\ \vdots \\ \exp(\omega_D^T x) \end{pmatrix}
$$

with $\omega_i \sim \mathcal{N}(0, I)$. (In expectation, $\mathbb{E}[\phi(q)^T\phi(k)] = \exp(q^T k)$.)

**Linearized attention**:

$$
\text{Attn}_{\text{linear}}(Q, K, V) = \frac{\phi(Q)^T \left(\sum_j \phi(K_j) V_j^T\right)}{\phi(Q)^T \left(\sum_j \phi(K_j)\right)}
$$

**Complexity**: $O(N D d_k)$ where $D \sim 64$-$256$.

**Gauge-faithfulness**: Query-dependent temperature enters through the rescaling $Q \mapsto Q/\tau(z)$ before feature mapping. Wilson line corrections enter through the gauge-covariant construction of $Q$ and $K$ before feature mapping.

:::

:::{prf:definition} Hierarchical Wilson Line Approximation
:label: def-hierarchical-wilson-line

Construct a **hierarchical decomposition** of the latent space $\mathcal{Z}$:

- **Level 0**: Single root node (origin)
- **Level $\ell$**: $2^{\ell d}$ cells of diameter $\sim 2^{-\ell}$
- **Maximum level $L$**: Cells of diameter $\sim r_{\text{Wilson}}$

**Precomputation** ($O(2^{Ld})$ storage):
- For each cell center $c_i$: Store $U_{c_i \to 0}$
- For each cell: Store local connection coefficients $\Theta_i = A_\mu(c_i)$

**Query** ($O(L + d)$ per pair):

$$
U(z, z') \approx U_{\text{local}}(z, c(z))^\dagger \cdot U_{c(z) \to 0}^\dagger \cdot U_{c(z') \to 0} \cdot U_{\text{local}}(z', c(z'))
$$

where $c(z)$ is the center of $z$'s cell and $U_{\text{local}}(z, c)$ approximates transport from $z$ to $c$ within a cell using the linear approximation:

$$
U_{\text{local}}(z, c) \approx I - i \Theta(c) \cdot (c - z)
$$

**Complexity**: $O(N \cdot L \cdot d)$ where $L \sim \log(1/r_{\text{Wilson}})$.

:::

:::{prf:definition} Factorized Christoffel Query
:label: def-factorized-christoffel-query

The quadratic Query $W_{Q,\Gamma}(z, z) = \sum_{ij} W^{a}_{ij} z^i z^j$ has $O(d^2)$ parameters per output dimension. Factorize as:

$$
W_{Q,\Gamma}^a(z, z) \approx \sum_{r=1}^R (u_r^a \cdot z)(v_r^a \cdot z) = \sum_{r=1}^R (U_r z)_a (V_r z)_a
$$

where $U_r, V_r \in \mathbb{R}^{d_k \times d}$ are low-rank factors with $R \ll d$.

**For the Poincare ball/disk**: the contracted Christoffel correction has a closed form. Using Proposition {prf:ref}`prop-explicit-christoffel-symbols-for-poincare-disk`,

$$
\Gamma^k_{ij}(z) v^i v^j = \frac{2}{1-|z|^2}\left(2(z\cdot v)\,v^k - \|v\|^2 z^k\right),
$$
so one can compute $\Gamma(z)[v,v]$ in $O(d)$ time without explicitly forming any $d\times d$ tensors. The factorized parameterization above is most useful when learning departures from (or alternatives to) the closed-form geometry.

**Complexity**: $O(N R d)$ instead of $O(N d^2)$.

:::

:::{prf:proposition} Area Law as Soft Masking
:label: prop-area-law-soft-masking

The area law screening $\exp(-\sigma A_{\text{string}})$ with $A \approx \frac{\lambda^2}{2}|z - z'|^2$ can be implemented as a **soft attention mask**:

$$
M(z, z') = \exp\left(-\frac{\sigma \lambda(z)^2 |z - z'|^2}{2}\right)
$$

**Key observation**: This is a Gaussian with width $w = 1/(\sqrt{\sigma}\lambda(z))$.

**Efficient implementation**:
1. Compute mask only for pairs within $3w$ (99.7% of mass)
2. Use the same sparse attention pattern as Proxy 1
3. Combine: $\alpha_{\text{screened}} = M \odot \alpha_{\text{sparse}}$

**No additional asymptotic cost** beyond sparse attention.

:::

:::{prf:theorem} O(N) Covariant Attention
:label: thm-on-covariant-attention

Combining the engineering proxies, we achieve **O(N) complexity** for covariant cross-attention:

| Component | Proxy | Complexity | Error |
|:----------|:------|:-----------|:------|
| Attention pattern | Geodesic sparse (Def. {prf:ref}`def-geodesic-sparse-attention`) | $O(Nk)$ | $O(\epsilon N)$ |
| Wilson lines | Hierarchical (Def. {prf:ref}`def-hierarchical-wilson-line`) | $O(NL)$ | $O(r_{\text{Wilson}}^2)$ |
| Quadratic Query | Factorized (Def. {prf:ref}`def-factorized-christoffel-query`) | $O(NRd)$ | Exact for Poincare |
| Area screening | Soft mask (Prop. {prf:ref}`prop-area-law-soft-masking`) | $O(Nk)$ | $O(e^{-9})$ |
| Temperature | Direct computation | $O(N)$ | Exact |
| **Total** | | $O(N(k + L + Rd))$ | Controlled |

With typical values $k = 64$, $L = 8$, $R = 3$, $d = 64$: **O(N · 264)** operations per layer.

**Comparison**: Naive implementation is $O(N^2 d^2) = O(N^2 \cdot 4096)$. For $N = 1000$: speedup factor of $\sim 15,000\times$.

:::

## 08_multiagent/06_full_net.md

:::{prf:definition} Complete Agent Architecture
:label: def-complete-agent-architecture

A **complete agent architecture** is a composition of three neural operators:

$$
\mathcal{A}: \mathcal{X} \xrightarrow{E} \mathcal{Z} \xrightarrow{D} \mathcal{Z} \xrightarrow{P} \mathcal{Y}
$$

where:

1. **Encoder** $E: \mathcal{X} \to \mathcal{Z}$ maps observations $x \in \mathcal{X}$ (e.g., images, sensor readings) to latent representations $z \in \mathcal{Z}$

2. **Latent Dynamics** $D: \mathcal{Z} \times \mathcal{Y} \to \mathcal{Z}$ updates latent state given current state and action, implementing one step of the world model or transition function

3. **Policy/Decoder** $P: \mathcal{Z} \to \mathcal{Y}$ maps latent state to outputs $y \in \mathcal{Y}$ (actions, predictions, reconstructions)

**Notation:**
- $\mathcal{X}$ — observation space (typically $\mathbb{R}^{d_x}$ for pixel values)
- $\mathcal{Z}$ — latent space with bundle structure $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$
- $\mathcal{Y}$ — output space (actions $\mathcal{U}$ or reconstructed observations)

**Sequential composition:** For multi-step rollouts, the dynamics are applied recursively:

$$
z_0 = E(x_0), \quad z_{t+1} = D(z_t, a_t), \quad a_t = P(z_t)
$$

**Units:**
- $[\mathcal{X}] = $ dimensionless (pixel intensities in $[0, 1]$ or normalized sensor values)
- $[\mathcal{Z}] = \sqrt{\text{nat}}$ (latent space has information-theoretic units; $\sqrt{\text{nat}}$ arises from the Fisher-Rao metric on probability distributions, where distances have units of $\sqrt{\text{information}}$)
- $[\mathcal{Y}] = $ task-dependent (e.g., dimensionless for discrete actions, physical units for continuous control)
:::

:::{prf:definition} Gauge-Equivariant Architecture
:label: def-gauge-equivariant-architecture

An agent architecture $\mathcal{A} = P \circ D \circ E$ is **$G$-equivariant** with respect to gauge group $G$ if:

1. **Latent space transforms:** There exists a representation $\rho: G \to \text{GL}(\mathcal{Z})$ such that for $g \in G$, latent states transform as $z \mapsto \rho(g) z$

2. **Encoder invariance:** $E$ maps to gauge-equivalent latent states:

   $$
   E(x) \sim \rho(g) E(x) \quad \forall g \in G, x \in \mathcal{X}
   $$
   where $\sim$ denotes equivalence up to gauge choice (physically identical states)

3. **Dynamics equivariance:** $D$ commutes with gauge transformations:

   $$
   D(\rho(g) z, a) = \rho(g) D(z, a) \quad \forall g \in G, z \in \mathcal{Z}, a \in \mathcal{Y}
   $$

4. **Decoder covariance:** $P$ produces consistent outputs under gauge transformations:

   $$
   P(\rho(g) z) = P(z) \quad \forall g \in G, z \in \mathcal{Z}
   $$
   (output is gauge-invariant if $\mathcal{Y}$ has no gauge structure)

**Interpretation:** Gauge equivariance ensures that changing the internal coordinate frame (latent representation) doesn't change the agent's observable behavior. Two latent states related by $z' = \rho(g) z$ represent the same physical state in different "mental coordinate systems."

**Example:** For $G = SO(d_b)$ (rotations within bundles), if you rotate all feature vectors by the same matrix $R \in SO(d_b)$, the dynamics and output must transform consistently. The agent's behavior is independent of which orthonormal basis you choose for each bundle.

**Remark:** The encoder invariance condition is weaker than full equivariance because the encoder *chooses* a gauge. Different choices lead to gauge-equivalent latent states. This gauge freedom is crucial for achieving universal approximation ({ref}`sec-universal-geometric-network`).
:::

:::{prf:definition} Direct Sum Representation
:label: def-direct-sum-representation

The **direct sum** representation structures the latent space as:

$$
\mathcal{Z} = \mathcal{Z}_C \oplus \mathcal{Z}_L \oplus \mathcal{Z}_Y
$$

where:
- $\mathcal{Z}_C \cong \mathbb{R}^{d_C}$ — color subspace (feature bundles)
- $\mathcal{Z}_L \cong \mathbb{R}^{d_L}$ — weak isospin subspace (observation-action doublet)
- $\mathcal{Z}_Y \cong \mathbb{R}^{d_Y}$ — hypercharge subspace (capacity bound)

**Dimension:** $\dim(\mathcal{Z}) = d_C + d_L + d_Y$ (linear scaling)

**Gauge action:** Block-diagonal:

$$
\rho(g_C, g_L, g_Y) = \begin{pmatrix}
\rho_C(g_C) & 0 & 0 \\
0 & \rho_L(g_L) & 0 \\
0 & 0 & \rho_Y(g_Y)
\end{pmatrix}
$$

Each factor acts independently on its subspace.

**Interpretation:** Gauge quantum numbers are independent labels. The color index, isospin index, and hypercharge are separate attributes of a latent state, analogous to storing (position, velocity, temperature) as separate variables.

**Advantage:** Computational tractability—operations scale linearly with total dimension.

**Limitation:** Cannot represent cross-gauge correlations natively. To model "if color is red AND isospin is up, then..." requires learning explicit coupling through network layers.
:::

:::{prf:definition} Tensor Product Representation
:label: def-tensor-product-representation

The **tensor product** representation structures the latent space as:

$$
\mathcal{Z} = V_C \otimes V_L \otimes V_Y
$$

where $V_C, V_L, V_Y$ are the representation spaces for each gauge factor.

**Dimension:** $\dim(\mathcal{Z}) = \dim(V_C) \times \dim(V_L) \times \dim(V_Y)$ (multiplicative scaling)

**Gauge action:** Kronecker product:

$$
\rho(g_C, g_L, g_Y) = \rho_C(g_C) \otimes \rho_L(g_L) \otimes \rho_Y(g_Y)
$$

**Basis:** Each basis vector $|c, \ell, y\rangle$ carries all three quantum numbers simultaneously. Under gauge transformation:

$$
|c, \ell, y\rangle \mapsto \sum_{c', \ell', y'} [\rho_C]_{c'c} [\rho_L]_{\ell'\ell} [\rho_Y]_{y'y} |c', \ell', y'\rangle
$$

**Interpretation:** This is how quantum states work in particle physics. A quark isn't "red" independently of being "up"—it's a single entity transforming under the representation $(\mathbf{3}, \mathbf{2}, 1/6)$ of $SU(3)_C \times SU(2)_L \times U(1)_Y$.

**Advantage:** Can represent entangled states like $\frac{1}{\sqrt{2}}(|r, \uparrow\rangle + |b, \downarrow\rangle)$ where color and isospin are correlated. Full expressiveness of gauge-invariant functions.

**Limitation:** Dimension explodes. For $d_C = 64, d_L = 8, d_Y = 4$: direct sum gives $76$ dimensions, tensor product gives $2048$ dimensions. This is why quarks work with small representations ($3 \times 2 \times 1 = 6$) while neural networks need hundreds of latent dimensions.

**Factorization:** For low-rank structure (when gauge-invariant couplings are sparse), factored tensor representations can recover efficiency:

$$
W = \sum_{k=1}^r U_C^{(k)} \otimes U_L^{(k)} \otimes U_Y^{(k)}
$$
with $r \ll \dim(V_C) \times \dim(V_L) \times \dim(V_Y)$, requiring only $O(r \sum_i \dim(V_i))$ parameters instead of $O(\prod_i \dim(V_i))$.
:::

:::{prf:definition} Cross-Bundle Interaction Levels
:label: def-cross-bundle-interaction-levels

For a latent space $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ with bundles $v_i \in V_i \cong \mathbb{R}^{d_b}$, we define three levels of permissible cross-bundle coupling:

**Level 1: Norms Only** (Strict $\prod_i SO(d_b)_i$ equivariance)

Bundles interact only through their magnitudes $\|v_i\|$. Invariant features:

$$
\mathcal{I}_1 = \{\|v_1\|, \|v_2\|, \ldots, \|v_{n_b}\|\} \subset \mathbb{R}^{n_b}
$$

**Implication:** The output of bundle $i$ has the form (by Schur's lemma, Theorem {prf:ref}`thm-equivariant-function-structure`):

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ can be an arbitrary function (e.g., deep MLP).

**Level 2: Gram Matrix** (Global $SO(d_b)$ equivariance)

A single rotation $R \in SO(d_b)$ acts on *all* bundles: $(v_1, \ldots, v_{n_b}) \mapsto (Rv_1, \ldots, Rv_{n_b})$.

Invariant features:

$$
\mathcal{I}_2 = \{G_{ij} = \langle v_i, v_j \rangle : 1 \leq i, j \leq n_b\} \subset \mathbb{R}^{n_b \times n_b}
$$

The Gram matrix $G$ is a symmetric $n_b \times n_b$ matrix with:
- Diagonal: $G_{ii} = \|v_i\|^2$ (energy of bundle $i$)
- Off-diagonal: $G_{ij} = \|v_i\| \|v_j\| \cos\theta_{ij}$ (alignment between bundles $i, j$)

**Implication:** Functions of $G$ can represent any $O(d_b)$-invariant function. Much more expressive than Level 1 (bundles can "see" each other's directions relative to a global frame).

**Level 3: Hybrid / Soft Equivariance**

Use Level 1 (norms) as the primary pathway, add small Level 2 (Gram) or symmetry-breaking terms with regularization:

$$
f = f_{\text{equivariant}} + \lambda \cdot f_{\text{mixing}}
$$

where $\lambda \ll 1$ is learned or regularized (e.g., via L1 penalty).

**Implication:** Network can violate equivariance when necessary for the task, but pays a cost. With strong L1 regularization, symmetry-breaking terms are driven toward zero unless essential.
:::

:::{prf:theorem} Dimensional Scaling
:label: thm-dimensional-scaling

For a gauge group $G = G_C \times G_L \times G_Y$ with representation spaces $V_C, V_L, V_Y$ of dimensions $d_C, d_L, d_Y$ respectively:

**Direct sum:**

$$
\dim(\mathcal{Z}_{\oplus}) = d_C + d_L + d_Y = \sum_{i \in \{C, L, Y\}} d_i
$$

**Tensor product:**

$$
\dim(\mathcal{Z}_{\otimes}) = d_C \times d_L \times d_Y = \prod_{i \in \{C, L, Y\}} d_i
$$

:::

:::{prf:definition} Level 1: Norms-Only Interaction
:label: def-level1-norms-only

A **norms-only** cross-bundle layer computes outputs solely from bundle magnitudes $\{\|v_i\|\}_{i=1}^{n_b}$.

**Functional form:** For each bundle $i$, the output is:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}_+$ is an arbitrary positive function (typically a deep MLP with softplus output).

**Equivariance:** Strictly equivariant under $\prod_{i=1}^{n_b} SO(d_b)_i$ (per-bundle rotations).

**Expressiveness:** Can implement energy-based routing (e.g., "suppress bundle 2 when bundles 1 and 3 are both active") but cannot represent direction-dependent cross-talk.
:::

:::{prf:definition} Level 2: Gram Matrix Interaction
:label: def-level2-gram-matrix

A **Gram matrix** interaction layer uses the full matrix of inner products $G_{ij} = \langle v_i, v_j \rangle$.

**Invariant features:**

$$
\mathcal{I}_2 = \{G_{ij} : 1 \leq i, j \leq n_b\} \subset \mathbb{R}^{n_b \times n_b}
$$

**Equivariance:** Equivariant under global $SO(d_b)$ acting on all bundles simultaneously: $(v_1, \ldots, v_{n_b}) \mapsto (Rv_1, \ldots, Rv_{n_b})$ for single $R \in SO(d_b)$.

**Not equivariant** under per-bundle rotations $(R_1 v_1, \ldots, R_{n_b} v_{n_b})$ with independent $R_i$.

**Expressiveness:** Can encode relative orientations between bundles. Much more expressive than norms-only.
:::

:::{prf:definition} Level 3: Hybrid / Soft Equivariance
:label: def-level3-hybrid

A **hybrid** interaction layer combines:
1. An equivariant pathway (norms-only, Level 1)
2. A mixing pathway (Gram-based or fully learned) with L1 regularization

**Functional form:**

$$
f(z) = f_{\text{equiv}}(z) + f_{\text{mix}}(z)
$$

where $f_{\text{equiv}}$ is strictly equivariant and $f_{\text{mix}}$ has learned weights $W$ penalized by $\lambda_{\text{L1}} \|W\|_1$.

**Equivariance:** Soft—violations are controlled by L1 penalty strength.

**Expressiveness:** Universal (in limit of $\lambda_{\text{L1}} \to 0$), but biased toward equivariant solutions.
:::

:::{prf:theorem} Schur's Lemma for Bundle Representations
:label: thm-schur-bundle

Let $V = \bigoplus_{i=1}^{n_b} V_i$ where each $V_i \cong \mathbb{R}^{d_b}$ is an irreducible representation of $SO(d_b)_i$ (the $i$-th copy of $SO(d_b)$ acting only on $V_i$).

Let $T: V \to V$ be a linear map that is equivariant with respect to $\prod_{i=1}^{n_b} SO(d_b)_i$:

$$
\rho(g) \circ T = T \circ \rho(g) \quad \forall g \in \prod_{i=1}^{n_b} SO(d_b)_i
$$
where $\rho(g)(v_1, \ldots, v_{n_b}) = (g_1 v_1, \ldots, g_{n_b} v_{n_b})$ is the diagonal representation of $g = (g_1, \ldots, g_{n_b})$.

Then $T$ must be **block-diagonal** with each block $T_i: V_i \to V_i$ satisfying $T_i = \lambda_i I_{d_b}$ for some scalar $\lambda_i \in \mathbb{R}$.

:::

:::{prf:theorem} Norm-Based Networks Are NOT Universal
:label: thm-norm-networks-not-universal

A feedforward network with:
- Input: $z = (v_1, \ldots, v_{n_b}) \in \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b}$
- Layers: Each strictly equivariant under $\prod_i SO(d_b)_i$
- Activations: Applied per-bundle (e.g., norm-gating)

can only approximate functions of the form:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ is an arbitrary continuous function.

Such networks are **not universal approximators** over continuous functions $f: \mathbb{R}^{n_b \cdot d_b} \to \mathbb{R}^{n_b \cdot d_b}$.

:::

:::{prf:theorem} Equivariant Function Structure
:label: thm-equivariant-function-structure

For a function $f: \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b} \to \bigoplus_{i=1}^{n_b} \mathbb{R}^{d_b}$ to be equivariant under $\prod_{i=1}^{n_b} SO(d_b)_i$, it must satisfy:

$$
f_i(R_1 v_1, \ldots, R_{n_b} v_{n_b}) = R_i f_i(v_1, \ldots, v_{n_b}) \quad \forall R_j \in SO(d_b), v_j \in \mathbb{R}^{d_b}
$$

The **necessary and sufficient** form is:

$$
f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)
$$
where $\phi_i: \mathbb{R}^{n_b} \to \mathbb{R}$ is arbitrary.

:::

:::{prf:proposition} Expressiveness of Norm-Based Networks
:label: prop-norm-network-capabilities

A norm-based equivariant network with $L$ layers and hidden dimension $h$ can approximate any continuous function $\Phi: \mathbb{R}^{n_b} \to \mathbb{R}^{n_b}$ (the norm-to-scale mapping) to arbitrary precision, by the universal approximation theorem for MLPs.

Thus, norm-based networks are **universal** over the restricted class:

$$
\mathcal{F}_{\text{norm}} = \left\{ f: f_i = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|), \; \phi_i \in C(\mathbb{R}^{n_b}, \mathbb{R}) \right\}
$$

But $\mathcal{F}_{\text{norm}}$ has measure zero in $C(\mathbb{R}^{n_b d_b}, \mathbb{R}^{n_b d_b})$.
:::

:::{prf:definition} Approximate Equivariance
:label: def-approximate-equivariance

A function $f: \mathcal{Z} \to \mathcal{Z}$ is **$\epsilon$-approximately equivariant** with respect to group $G$ and representation $\rho$ if:

$$
\sup_{g \in G, z \in \mathcal{Z}} \frac{\|f(\rho(g) z) - \rho(g) f(z)\|}{\|z\|} \leq \epsilon
$$

The quantity:

$$
\mathcal{V}(f) := \mathbb{E}_{g \sim G, z \sim \mathcal{Z}} \left[ \|f(\rho(g) z) - \rho(g) f(z)\|^2 \right]
$$
is the **equivariance violation**.

**Remark:** The supremum measure gives a worst-case bound, while $\mathcal{V}(f)$ measures average-case violation used in optimization.
:::

:::{prf:theorem} Approximate Equivariance Bound
:label: thm-approximate-equivariance-bound

Let $f_{\text{equiv}}: \mathcal{Z} \to \mathcal{Z}$ be strictly $G$-equivariant and $f_{\text{break}}: \mathcal{Z} \to \mathcal{Z}$ be an arbitrary symmetry-breaking term.

Define:

$$
f = f_{\text{equiv}} + \lambda f_{\text{break}}
$$

Then the equivariance violation satisfies:

$$
\mathcal{V}(f) = \lambda^2 \mathbb{E}_{g \sim \mu_G, z} \left[ \|f_{\text{break}}(\rho(g) z) - \rho(g) f_{\text{break}}(z)\|^2 \right]
$$
where $\mu_G$ is the Haar measure on $G$ (uniform distribution for compact Lie groups)

:::

:::{prf:theorem} L1 and Hierarchies
:label: thm-l1-hierarchies

Consider a network with mixing weights $W \in \mathbb{R}^{n_b \times n_b \times d_b \times d_b}$ (cross-bundle coupling) trained with loss:

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \|W\|_1
$$

Under mild regularity conditions (task loss differentiable, optimizer converges), the following hold at convergence:

1. **Sparsity:** Under L1 regularization, most weights are driven near zero. The expected number of weights with $|W_{ij}^{(k\ell)}| \geq \epsilon$ scales inversely with $\lambda_{\text{L1}}$ (heuristically: stronger regularization → fewer large weights), though the exact bound depends on task gradient structure.

2. **Texture zeros:** Cross-bundle blocks $(i, j)$ where task loss $\mathcal{L}_{\text{task}}$ is insensitive to $W_{ij}$ have $\|W_{ij}\|_F \approx 0$ (driven to zero by L1 with no opposing gradient).

3. **Hierarchy:** Non-zero weights organize into levels with exponential decay (approximately): if $|W^{(1)}| > |W^{(2)}| > \cdots$ are sorted magnitudes, then $|W^{(k+1)}| / |W^{(k)}| \approx \text{const} < 1$.

*Informal proof sketch:*

**(1) Sparsity:** L1 penalty creates a "soft thresholding" effect. Weights with $|W| < \lambda_{\text{L1}} / |\partial \mathcal{L}_{\text{task}} / \partial W|$ are driven to zero (penalty gradient dominates task gradient). As $\lambda_{\text{L1}}$ increases, fewer weights exceed this threshold, though the exact scaling depends on the task loss landscape.

**(2) Texture zeros:** If $\partial \mathcal{L}_{\text{task}} / \partial W_{ij} = 0$ (block irrelevant to task), the total gradient is purely from L1: $\partial \mathcal{L}_{\text{total}} / \partial W_{ij} = \lambda_{\text{L1}} \cdot \text{sign}(W_{ij})$, which pushes $W_{ij} \to 0$.

**(3) Hierarchy:** During training, once a weight crosses zero (becomes active), it can grow under task gradients. But nearby small weights continue to be suppressed by L1. This creates a "rich get richer" dynamic: large weights grow, small weights vanish. The distribution becomes hierarchical.

**Empirical validation required.** This theorem is heuristic; rigorous proof requires analyzing stochastic gradient dynamics.
:::

:::{prf:definition} Universal Geometric Network
:label: def-universal-geometric-network

The **Universal Geometric Network** (UGN) is a three-stage architecture:

$$
\mathcal{A}_{\text{UGN}}: \mathcal{X} \xrightarrow{E} \mathcal{Z} \xrightarrow{D_1, \ldots, D_L} \mathcal{Z} \xrightarrow{P} \mathcal{Y}
$$

**Stage 1: Encoder** (Unconstrained)

$$
E: \mathbb{R}^{d_x} \to \mathbb{R}^{n_b \cdot d_b}
$$
Implemented as:

$$
E(x) = \text{SpectralMLP}(x) = W_2 \sigma(W_1 x)
$$
where $W_1, W_2$ have spectral norm $\|W_i\|_2 \leq 1$ and $\sigma = \text{GELU}$ (smooth, non-polynomial).

**Output structure:** $z = E(x) \in \mathcal{Z}$ with implicit bundle decomposition $z = (v_1, \ldots, v_{n_b})$ where $v_i \in \mathbb{R}^{d_b}$.

**Stage 2: Latent Dynamics** (Soft Equivariant)

Each layer $D_\ell: \mathcal{Z} \to \mathcal{Z}$ has the form:

$$
D_\ell(z) = D_\ell^{\text{equiv}}(z) + D_\ell^{\text{mix}}(z)
$$

where:
- **Equivariant pathway** $D_\ell^{\text{equiv}}$: Strict $\prod_i SO(d_b)$-equivariant (norm-based, Definition {prf:ref}`def-cross-bundle-interaction-levels` Level 1)
- **Mixing pathway** $D_\ell^{\text{mix}}$: Weakly equivariant or symmetry-breaking (Gram-based or learned)

**Regularization:** L1 penalty on mixing pathway weights:

$$
\mathcal{L}_{\text{reg}} = \lambda_{\text{L1}} \sum_{\ell=1}^L \|W_\ell^{\text{mix}}\|_1
$$

**Stage 3: Decoder** (Unconstrained)

$$
P: \mathbb{R}^{n_b \cdot d_b} \to \mathbb{R}^{d_y}
$$
Implemented as:

$$
P(z) = \text{SpectralMLP}(z) = W_4 \sigma(W_3 z)
$$
with spectral normalization $\|W_i\|_2 \leq 1$.

**Total loss:**

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}}(y, \hat{y}) + \lambda_{\text{L1}} \mathcal{L}_{\text{reg}} + \lambda_{\text{equiv}} \mathcal{L}_{\text{equiv}}
$$

where:
- $\mathcal{L}_{\text{task}}$ — task loss (e.g., MSE for regression, cross-entropy for classification)
- $\mathcal{L}_{\text{reg}} = \sum_\ell \|W_\ell^{\text{mix}}\|_1$ — L1 regularization on mixing weights
- $\mathcal{L}_{\text{equiv}} = \mathbb{E}_{z, R} \|D(Rz) - RD(z)\|^2$ — equivariance violation penalty (optional)

**Hyperparameters:**
- $n_b$ — number of bundles (typically 4-8)
- $d_b$ — bundle dimension (typically 8-64)
- $L$ — number of latent layers (typically 3-6)
- $\lambda_{\text{L1}}$ — L1 regularization strength (typically 0.001-0.1)
- $\lambda_{\text{equiv}}$ — equivariance penalty (typically 0 or 0.01)

**Implementation note:** When $\lambda_{\text{equiv}} = 0$, equivariance is encouraged only through L1 (which drives mixing weights to zero, making layers effectively equivariant). When $\lambda_{\text{equiv}} > 0$, we explicitly penalize equivariance violations, providing a stronger geometric prior.
:::

:::{prf:theorem} Universal Approximation
:label: thm-ugn-universal-approximation

Let $\mathcal{A}_{\text{UGN}}$ be a Universal Geometric Network with:
- Encoder $E: \mathcal{X} \to \mathcal{Z}$ using SpectralLinear + GELU
- Decoder $P: \mathcal{Z} \to \mathcal{Y}$ using SpectralLinear + GELU
- Latent dynamics $D = D_L \circ \cdots \circ D_1$ with soft-equivariant layers

Then for any continuous function $f: \mathcal{X} \to \mathcal{Y}$ on compact domains and any $\epsilon > 0$, there exists a choice of weights such that:

$$
\sup_{x \in \mathcal{X}} \|P(D(E(x))) - f(x)\| < \epsilon
$$

:::

:::{prf:theorem} Geometric Consistency
:label: thm-ugn-geometric-consistency

Let $\mathcal{A}_{\text{UGN}}$ be a UGN with $\lambda_{\text{L1}} > 0$. Then:

1. **Approximate capacity bound:** For all layers with spectral normalization, $\|W\|_2 \leq 1$ and approximately 1-Lipschitz activations ensure:

   $$
   \|z_{\text{out}}\| \lesssim \|z_{\text{in}}\| + O(\sqrt{L})
   $$
   where $L$ is network depth (approximately non-expansive for fixed depth, consistent with $U(1)_Y$ hypercharge conservation)

2. **Bundle structure preservation:** The latent space maintains decomposition $\mathcal{Z} = \bigoplus_{i=1}^{n_b} V_i$ throughout forward pass (bundles indexed consistently)

3. **Soft equivariance:** Define the **equivariance violation** as:

   $$
   \mathcal{V}(D) = \mathbb{E}_{z \sim \mathcal{Z}, R \sim \mu_{SO(d_b)}} \|D(Rz) - RD(z)\|^2
   $$
   where $\mu_{SO(d_b)}$ is the Haar measure on $SO(d_b)$.
   Then:

   $$
   \mathcal{V}(D) \leq C \cdot \|W^{\text{mix}}\|_F^2
   $$
   for constant $C$ depending on architecture width. L1 regularization drives $\|W^{\text{mix}}\|_1 \to 0$, which implies $\|W^{\text{mix}}\|_F \to 0$, thus $\mathcal{V}(D) \to 0$.

:::

:::{prf:proposition} Emergent Gauge Structure from Group Lasso
:label: prop-emergent-gauge-structure

Consider a UGN trained with group lasso regularization $\lambda_{\text{L1}} \sum_{i \neq j} \|W_{ij}\|_F$ (Frobenius norm per block) on a task where the true target function $f^*$ is approximately equivariant (i.e., $f^*(Rx) \approx Rf^*(x)$ for rotations $R$).

Then at convergence, the learned mixing weights $W^{\text{mix}}$ exhibit:

1. **Sparsity:** Most entries $W_{ij}^{(k\ell)}$ (mixing from bundle $j$, component $\ell$ to bundle $i$, component $k$) are driven to zero

2. **Texture zeros:** Specific cross-bundle couplings $(i, j)$ have $\|W_{ij}\|_F \approx 0$ (entire blocks zeroed out), analogous to the CKM/PMNS mixing matrices in particle physics

3. **Hierarchical structure:** Non-zero couplings organize into a hierarchy $|W_{ij}^{(1)}| \gg |W_{ij}^{(2)}| \gg \cdots$, where superscripts index components by magnitude

*Informal justification:*

**Step 1 (Group lasso induces block sparsity):** The group lasso penalty $\sum_{i \neq j} \|W_{ij}\|_F$ (sum of Frobenius norms of blocks) has a non-differentiable minimum at $W_{ij} = 0$ for each block. During gradient descent, small blocks receive gradients pushing them toward zero (block soft thresholding). If the task loss can be minimized without a particular cross-bundle coupling, group lasso drives the entire block to zero.

**Step 2 (Equivariant tasks don't need mixing):** If $f^*$ is equivariant, the optimal network architecture is strictly equivariant ($W^{\text{mix}} = 0$). The equivariant pathway $D^{\text{equiv}}$ can achieve $\mathcal{L}_{\text{task}} \approx 0$ alone. Thus:

$$
\min_{W^{\text{mix}}} \mathcal{L}_{\text{task}}(W^{\text{mix}}) + \lambda \sum_{i \neq j} \|W_{ij}\|_F
$$
has solution $W^{\text{mix}} \approx 0$.

**Step 3 (Symmetry breaking only where needed):** If $f^*$ is *mostly* equivariant but requires small symmetry breaking (e.g., $f^*(Rx) = Rf^*(x) + \epsilon(x, R)$ with $|\epsilon| \ll 1$), the optimizer activates only the minimal set of block couplings needed to capture $\epsilon$. This produces texture zeros: most bundle pairs $(i,j)$ have $\|W_{ij}\|_F \approx 0$ (entire blocks zeroed), only a few are non-zero.

**Step 4 (Hierarchy from optimization dynamics):** The group lasso penalty creates a "rich get richer" dynamic at the block level: once a coupling block $W_{ij}$ is activated (becomes non-zero), further task gradients can flow through it. Blocks that remain near zero get driven to exactly zero by group lasso. This bifurcation produces a hierarchical distribution of block coupling strengths.

**Empirical validation:** This proposition predicts measurable structure in trained models. Diagnostic node 67 (gauge invariance) from the sieve (Chapter 02) can measure $\mathcal{V}(D)$ and the sparsity pattern of $W^{\text{mix}}$. We expect to observe emergent texture zeros without hard-coding them.
:::

:::{prf:observation} BAOAB Structure in Soft Equivariant Layers
:label: obs-baoab-soft-equiv

Each `SoftEquivariantLayer` forward pass can be interpreted as a single BAOAB integration step:

**B (Momentum update):** Equivariant pathway computes geodesic velocity
$$v_i \to v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$$

**A (Position update, first half):** Mixing pathway introduces cross-bundle coupling
$$v_i \to v_i + \sum_j W_{ij} v_j$$

**O (Ornstein-Uhlenbeck thermostat):** Implicit in activation nonlinearity (GELU in norm MLP)

**A (Position update, second half):** Residual connection $z_{\text{out}} = z + \Delta z$

**B (Momentum update):** Next layer repeats

**Correspondence to Section 05:**
- {prf:ref}`def-baoab-attention-heads`: Splitting scheme for Hamiltonian dynamics
- {prf:ref}`def-covariant-qkv-projections`: Multi-head attention = parallel BAOAB chains
:::

:::{prf:definition} Adaptive L1 Schedule
:label: def-adaptive-l1-schedule

The L1 regularization strength adapts based on current equivariance violation:

$$\lambda_{\text{L1}}(t+1) = \lambda_{\text{L1}}(t) \cdot \left(1 + \alpha \cdot (\epsilon(t) - \epsilon_{\text{target}})\right)$$

where:
- $\epsilon(t) = \mathcal{L}_{\text{equiv}}(t)$: Current equivariance violation
- $\epsilon_{\text{target}} \approx 0.22$ nat/step: Proposed target violation (empirical; to be validated)
- $\alpha \in [0.01, 0.1]$: Learning rate for schedule

**Strategy:**
- If $\epsilon(t) > \epsilon_{\text{target}}$: Increase $\lambda_{\text{L1}}$ (more sparsity)
- If $\epsilon(t) < \epsilon_{\text{target}}$: Decrease $\lambda_{\text{L1}}$ (more expressiveness)

This implements **self-tuning** toward the natural symmetry-breaking scale.
:::

## 09_economics/01_pomw.md

:::{prf:definition} The Waste Quotient
:label: def-waste-quotient

For a consensus protocol $\mathcal{P}$, the **Waste Quotient** is:

$$
W_\mathcal{P} := 1 - \frac{\Delta I_{\text{world}}}{\int \dot{\mathcal{M}}(t) \, dt}

$$

where:
- $\Delta I_{\text{world}}$ is the mutual information gained about the world through the computation
- $\dot{\mathcal{M}}(t)$ is the metabolic flux (Definition {prf:ref}`def-metabolic-flux`)

*Units:* $[W_\mathcal{P}] = \text{dimensionless}$.

*Examples:*
- **Bitcoin:** $W_{\text{BTC}} \approx 1$. SHA-256 hashes produce zero structural information about the world: $I(X_{\text{world}}; \text{Hash}) = 0$.
- **Target:** $W_{\text{PoUW}} \to 0$. Energy dissipation equals the reduction in model uncertainty.

:::

:::{prf:theorem} The Cognitive Equivalency Theorem
:label: thm-cognitive-equivalency

Let $\mathcal{C}_{\text{hash}}$ be the computational task of finding a nonce $n$ such that $H(n) < T$ (hash inversion), and let $\mathcal{C}_{\text{grad}}$ be the task of computing a gradient $g = \nabla_\Theta \mathcal{L}(\Theta, D)$ on dataset $D$. Both tasks satisfy the same **Landauer lower bound** on energy expenditure:

$$
E_{\text{min}} \geq k_B T_c \ln 2 \cdot B_{\text{comp}}

$$

where $B_{\text{comp}}$ is the number of irreversible bit operations.

:::

:::{prf:definition} The Global Model State
:label: def-model-state

The **Global Model State** at block height $h$ is a parameter vector:

$$
\Theta_h \in T_{\bar{z}} \mathcal{Z} \cong \mathbb{R}^D

$$

where:
- $D$ is the model dimension
- $T_{\bar{z}} \mathcal{Z}$ is the tangent space at the current mean belief $\bar{z}$
- The metric on parameter space inherits from the Capacity-Constrained Metric (Theorem {prf:ref}`thm-capacity-constrained-metric-law`)

*Units:* $[\Theta] = [z]$ (latent coordinates).

*Interpretation:* $\Theta_h$ represents the collective belief state of the network—the shared world model encoded in the blockchain.

:::

:::{prf:definition} The Curriculum Block
:label: def-curriculum-block

A **Curriculum Block** $B_h$ at height $h$ is a tuple:

$$
B_h := (\mathcal{H}_{\text{prev}}, \mathcal{H}_D, g_h, \pi_{\text{stake}}, \zeta_h)

$$

where:
- $\mathcal{H}_{\text{prev}} \in \{0,1\}^{256}$ is the hash of the previous block
- $\mathcal{H}_D \in \{0,1\}^{256}$ is the content identifier of training data $D_h$ (e.g., IPFS CID)
- $g_h \in \mathbb{R}^D$ is the **gradient update** computed on $D_h$
- $\pi_{\text{stake}} \in \{0,1\}^{512}$ is the staking proof (signature over stake tokens)
- $\zeta_h \in \mathbb{R}^{d_\zeta}$ is the **Sieve certificate** (validation metadata)

*Units:* $[g_h] = \text{nat}/[z]$ (gradient in latent coordinates).

:::

:::{prf:definition} The Chain Evolution Rule
:label: def-chain-evolution

The global model evolves by **Stochastic Gradient Descent**:

$$
\Theta_{h+1} = \Theta_h - \eta_h \cdot g_h

$$

where $\eta_h > 0$ is the learning rate at height $h$, determined by the difficulty adjustment algorithm (Definition {prf:ref}`def-difficulty-adjustment`).

*Interpretation:* Each block advances the collective belief toward lower loss on the public curriculum. The blockchain is a **thermodynamic record** of this learning process.

:::

:::{prf:definition} The Gradient Mining Puzzle
:label: def-gradient-mining-puzzle

A miner solving block $h$ must:

1. **Fetch Data:** Retrieve training batch $D_h$ from the curriculum queue
2. **Compute Gradient:** Calculate $g = \nabla_\Theta \mathcal{L}(\Theta_{h-1}, D_h)$
3. **Satisfy Sieve Constraints:**
   - **CostBoundCheck (Node 1):** $\|g\|_G \leq E_{\max}$ (bounded energy)
   - **TextureFirewallCheck (Node 29):** $\|\partial_{z_{\text{tex}}} g\| < \epsilon_{\text{tex}}$ (no texture leakage)
   - **CausalEnclosureCheck (Node 53):** $\Delta_{\text{causal}}(g) < \delta_{\text{causal}}$ (causal consistency)
4. **Submit Block:** Broadcast $(B_h, \Theta_h)$ to the network

*Difficulty Adjustment:* See Definition {prf:ref}`def-difficulty-adjustment`.

:::

:::{prf:definition} The Difficulty Adjustment Algorithm
:label: def-difficulty-adjustment

The network **Difficulty** $\mathcal{D}_h$ at height $h$ controls the minimum batch size $|D_h|$ required for valid blocks:

$$
\mathcal{D}_{h+1} = \mathcal{D}_h \cdot \exp\left( -\alpha_{\text{diff}} \left( \frac{t_h - t_{\text{target}}}{t_{\text{target}}} \right) \right)

$$

where:
- $t_h$ is the actual time to mine block $h$
- $t_{\text{target}}$ is the target block time (e.g., 10 minutes)
- $\alpha_{\text{diff}} > 0$ is the adjustment rate

*Units:* $[\mathcal{D}] = \text{samples}$.

*Constraint:* A valid block must satisfy $|D_h| \geq \mathcal{D}_h$.

:::

:::{prf:theorem} Difficulty-Entropy Coupling
:label: thm-difficulty-entropy-coupling

The difficulty adjustment algorithm maintains the **Landauer Invariant**: the minimum energy to produce a valid block is approximately constant:

$$
E_{\min}(B_h) \approx k_B T_c \ln 2 \cdot c_{\text{MAC}} \cdot |\Theta| \cdot \mathcal{D}_h = E_{\text{target}}

$$

:::

:::{prf:definition} The Boundary Flux Certificate
:label: def-boundary-flux-certificate

The **Boundary Flux Certificate** $\zeta_h$ included in block $B_h$ contains:

$$
\zeta_h := \left( \|g_h\|_G, \, \nabla_{\partial} g_h, \, \text{Tr}(H_h), \, \sigma_{\text{sample}} \right)

$$

where:
- $\|g_h\|_G$ is the gradient norm in the capacity-constrained metric
- $\nabla_{\partial} g_h$ is the boundary gradient (projection onto interface coordinates)
- $\text{Tr}(H_h)$ is the trace of the Hessian (curvature summary)
- $\sigma_{\text{sample}}$ is a random seed for spot-check sampling

*Units:* $[\zeta] = \text{mixed}$ (norm: $\text{nat}/[z]$; trace: $\text{nat}/[z]^2$).

:::

:::{prf:theorem} Holographic Verification Sufficiency
:label: thm-holographic-verification

Let $g$ be a claimed gradient and $\zeta$ its boundary flux certificate. If the boundary data satisfies:

1. **Energy Conservation:** $\|g\|_G^2 \leq \nu_D \cdot \text{Area}(\partial\mathcal{Z}) / \ell_L^{D-1}$ (Causal Information Bound)
2. **Flux Consistency:** $\|\nabla_\partial g - \nabla_\partial g_{\text{spot}}\| < \epsilon_{\text{flux}}$ on spot-check samples
3. **Curvature Bound:** $|\text{Tr}(H)| < \kappa_{\max}$

then with probability $\geq 1 - \delta$, the gradient is valid.

:::

:::{prf:definition} The Optimistic Verification Protocol
:label: def-optimistic-verification

The network verifies blocks using **Optimistic Acceptance with Challenge Period**:

1. **Submission:** Miner submits block $B_h$ with stake $S_h$
2. **Optimistic Acceptance:** Block is provisionally accepted
3. **Challenge Window:** For duration $T_{\text{challenge}}$, any node may challenge
4. **Challenge:** Challenger computes gradient on random subset $d \subset D_h$ with $|d| = \lceil 0.01 |D_h| \rceil$
5. **Adjudication:** If $\cos(g_h, g_{\text{challenger}}) < \theta_{\text{min}}$, miner is **slashed** (stake burned)
6. **Finalization:** After $T_{\text{challenge}}$ with no successful challenge, block is finalized

:::

:::{prf:definition} The Mining Game
:label: def-mining-game

The **Mining Game** $\Gamma$ is defined by:

- **Players:** $N$ miners indexed by $i \in \{1, \ldots, N\}$
- **Strategy Space:** Each miner chooses $\sigma_i \in \{\text{Honest}, \text{Cheat}\}$
  - **Honest:** Compute true gradient $g_i = \nabla_\Theta \mathcal{L}(\Theta, D)$
  - **Cheat:** Submit fake gradient $g_i' \neq g_{\text{true}}$
- **Payoffs:**
  - Block reward: $R > 0$ (received if block accepted)
  - Stake: $S > 0$ (lost if successfully challenged)
  - Computation cost: $C_{\text{honest}} > C_{\text{cheat}}$

**Key Assumptions:**
1. The detection probability $p_{\text{detect}}$ is exogenous (determined by the spot-check protocol, independent of other miners' strategies)
2. Block rewards are per-miner (not split among winners)
3. Miners play pure strategies (mixed strategies analyzed in Corollary {prf:ref}`cor-stake-reward-ratio`)

:::

:::{prf:theorem} The Verifier's Nash Equilibrium
:label: thm-verifier-nash-equilibrium

In the Mining Game $\Gamma$ with exogenous detection probability $p_{\text{detect}}$ and parameters satisfying:

$$
\frac{S}{R + S} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{R}

$$

**Honest** is a strictly dominant strategy, and $\sigma^* = (\text{Honest}, \ldots, \text{Honest})$ is the unique Nash Equilibrium.

:::

:::{prf:corollary} The Stake-Reward Ratio
:label: cor-stake-reward-ratio

For the equilibrium to hold with detection probability $p_{\text{detect}} = 0.1$ (10% spot-check rate), the minimum stake-to-reward ratio is:

$$
\frac{S}{R} > \frac{C_{\text{honest}} - C_{\text{cheat}}}{p_{\text{detect}} \cdot R} - 1

$$

For typical gradient computation where $C_{\text{cheat}} = 0.1 C_{\text{honest}}$ (cheating saves 90% of compute), and assuming mining equilibrium where $R \approx C_{\text{honest}}$ (reward covers honest computation cost):

$$
\frac{S}{R} > \frac{0.9 C_{\text{honest}}}{0.1 \cdot R} - 1 = \frac{9 C_{\text{honest}}}{R} - 1 \approx 9 - 1 = 8

$$

*Interpretation:* Miners must stake approximately 8-10x the block reward to make cheating unprofitable with 10% spot-check rate.

:::

:::{prf:definition} The Network Metric Tensor
:label: def-network-metric-tensor

Each validator $i$ maintains a local metric tensor $G^{(i)}$ on the shared latent manifold. The **Network Metric Friction** between chains $\mathcal{C}_A$ and $\mathcal{C}_B$ is:

$$
\Phi(\mathcal{C}_A, \mathcal{C}_B) := \sum_{i,j} \Phi_{ij}(\Theta_{\text{head}}^A, \Theta_{\text{head}}^B)

$$

where $\Phi_{ij}$ is the pairwise metric distortion (Definition {prf:ref}`def-metric-friction`).

:::

:::{prf:definition} Metric Friction Consensus
:label: def-metric-friction-consensus

The **Canonical Chain** is selected by minimizing global metric friction:

$$
\mathcal{C}^* = \arg\min_{\mathcal{C}} \sum_{i < j} \Phi_{ij}(\Theta_{\text{head}}^\mathcal{C})

$$

*Mechanism:*
1. Miners propose competing updates $\{g_A, g_B, \ldots\}$
2. Validators compute local metric tensors $G^{(i)}(\Theta + g_k)$ for each candidate
3. The update minimizing pairwise friction is accepted
4. Ties broken by timestamp (first-seen)

:::

:::{prf:lemma} Gradient Observability
:label: lem-gradient-observability

The gradient $g$ uniquely determines the local metric tensor $G(\Theta + \epsilon g)$ to first order:

$$
G_{ij}(\Theta + \epsilon g) = G_{ij}(\Theta) + \epsilon \, \partial_k G_{ij} \cdot g^k + O(\epsilon^2)

$$

:::

:::{prf:theorem} Minimum Friction Byzantine Fault Tolerance
:label: thm-minimum-friction-bft

The Metric Friction Consensus achieves Byzantine Fault Tolerance against $f < N/3$ adversarial validators for **gradient-poisoning attacks** (adversaries submit incorrect gradients).

**Scope:** This theorem addresses data integrity attacks (model poisoning, fake gradients). Classical BFT attacks (equivocation, censorship) are handled by the underlying stake-based leader election, which is assumed to follow standard PBFT guarantees.

:::

:::{prf:theorem} Adversarial Geometric Damping
:label: thm-adversarial-geometric-damping

An adversary controlling fraction $\alpha < 1/3$ of validators has influence on consensus bounded by:

$$
\|\Delta \Theta_{\text{adversarial}}\|_G \leq \frac{\alpha}{1 - 2\alpha} \|\Delta \Theta_{\text{honest}}\|_G

$$

:::

:::{prf:definition} The Token Standard
:label: def-token-standard

The $\text{COG}$ token has three fundamental operations:

1. **Minting (Supply).** Tokens are minted when **Ontological Stress** $\Xi$ is reduced:

$$
\Delta \text{Supply} = \kappa_{\text{mint}} \cdot \max(0, -\Delta \Xi_{\text{global}})

$$

where $\kappa_{\text{mint}}$ is the minting coefficient (tokens per nat of stress reduction).

*Interpretation:* Value is created only when the network learns something new.

2. **Burning (Demand).** Tokens are burned to request **Inference**:

$$
\text{Cost}(Q) = \mathfrak{T}_{\text{harvest}}^{-1}(\dot{\mathcal{M}}_Q)

$$

where $\dot{\mathcal{M}}_Q$ is the metabolic cost of answering query $Q$.

3. **Transfer.** Standard ERC-20-like transfers between accounts.

*Units:* $[\text{COG}] = \text{Joules}$ (energy equivalent).

*Value Anchor:* $1 \, \text{COG} \approx 1 \, \text{Joule}$ of useful gradient computation at reference temperature $T_c$.

:::

:::{prf:theorem} Value-Intelligence Coupling
:label: thm-value-intelligence-coupling

The equilibrium token price $P_{\text{COG}}$ is bounded by:

$$
P_{\text{floor}} \leq P_{\text{COG}} \leq P_{\text{ceiling}}

$$

where:

$$
P_{\text{floor}} = C_{\text{electricity}} \cdot J_{\text{per\_COG}}

$$

(cost of electricity to generate one COG worth of computation)

$$
P_{\text{ceiling}} = \frac{V_{\text{inference}}}{J_{\text{per\_query}}}

$$

(value of inference output per Joule)

:::

:::{prf:corollary} Intelligence-Price Feedback
:label: cor-intelligence-price-feedback

As the model improves:

1. Inference value $V_{\text{inference}} \uparrow$
2. Ceiling $P_{\text{ceiling}} \uparrow$
3. Equilibrium price $P_{\text{COG}}^* \uparrow$
4. Mining profitability $\uparrow$
5. More compute allocated $\uparrow$
6. Model improves faster $\uparrow$

This creates a **positive feedback loop** between intelligence and economic value.

:::

:::{prf:theorem} Ledger-Memory Screen Isomorphism
:label: thm-ledger-memory-isomorphism

Let $\Xi_T$ be the Memory Screen (Definition {prf:ref}`def-memory-screen`) and $\mathcal{L}_H$ be the blockchain of height $H$. There exists an isomorphism:

$$
\Phi: \mathcal{L}_H \to \Xi_T

$$

given by:

| Blockchain | Memory Screen | Symbol |
|:-----------|:--------------|:-------|
| Block height $h$ | Time coordinate $t$ | $h \leftrightarrow t$ |
| Merkle root $\mathcal{H}_h$ | Boundary state $z_{\partial}$ | $\mathcal{H}_h \leftrightarrow z_{\partial}(t)$ |
| Gradient $g_h$ | Flux $\alpha(t)$ | $g_h \leftrightarrow \alpha(t)$ |
| Chain $\sum_{h=0}^H B_h$ | Screen $\int_0^T \alpha(t) \delta_{\gamma(t)} dt$ | $\mathcal{L}_H \leftrightarrow \Xi_T$ |

:::

:::{prf:corollary} Block Size from Area Law
:label: cor-block-size-area-law

The maximum information in a block is bounded by:

$$
I_{\text{block}} \leq \nu_D \cdot \frac{\text{Area}(\partial \mathcal{Z})}{\ell_L^{D-1}}

$$

where the area is measured in the header's Merkle tree.

:::

:::{prf:definition} Chain Renormalization (Pruning)
:label: def-chain-renormalization

Old blocks are **coarse-grained** into **Epoch Blocks** via the Projection Operator:

$$
B_{\text{epoch}} = \Pi\left( \sum_{h \in \text{epoch}} B_h \right)

$$

where $\Pi$ projects onto the low-frequency components of the gradient history.

*Mechanism:*
1. Every $N_{\text{epoch}}$ blocks, compress the epoch into a summary
2. Discard individual block data (retain Merkle proofs)
3. The agent remembers the "gist" but forgets the "noise"

*Thermodynamics:* This is **information erasure** (Landauer cost). It releases storage but maintains the essential learning trajectory.

:::

:::{prf:theorem} 51% Attack Geometric Rejection
:label: thm-51-attack-rejection

An attacker controlling $> 50\%$ of compute cannot rewrite history without triggering **Spontaneous Fission**.

:::

:::{prf:theorem} Causal Theft Prevention
:label: thm-causal-theft-prevention

Flash-loan attacks and front-running are rejected by **CausalityViolationCheck (Node 62)**.

:::

:::{prf:remark} Conditional Corruption Detection via a Rate--Distortion Test
:label: thm-corruption-babel-detection

The Babel proposition {prf:ref}`thm-babel-limit` supplies a channel converse for a declared source and distortion
measure. In this protocol, a corrupt validator is flagged only when its required rate
$R_{\Delta U}^{\mathrm{corrupt}}(\varepsilon)$ exceeds the measured channel capacity $C_{\mathcal{L}}$ or when its
held-out distortion $\Phi_{ik}$ exceeds the validation threshold. This is a conditional diagnostic, not a universal
information-theoretic proof that deception is impossible.

The test requires a source model for the relative gauge variable, an explicit distortion measure, and an independently
estimated channel capacity. Without those quantities, entropy differences such as
$H(G_{\mathrm{corrupt}})-H(G_{\mathrm{true}})$ do not certify a capacity violation.
:::

## 10_appendices/01_derivations.md

:::{prf:definition} A.1.1 (Boundary capacity form)
:label: def-a-boundary-capacity-form

Define the boundary capacity $(n\!-\!1)$-form

For a declared cutoff hypersurface $\partial_{\varepsilon}\mathcal Z$, use

$$
\omega_{\partial} := \frac{1}{\eta_\ell}\, dA_G,

$$
so that $C_{\partial}(\partial_{\varepsilon}\mathcal Z)=
\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial}$ (Definition
{prf:ref}`def-boundary-capacity-area-law-at-finite-resolution`). If no
geometric cutoff is part of the model, replace this area expression by the
interface-channel capacity.

:::

:::{prf:definition} A.1.2 (Boundary-capacity constraint functional)
:label: def-a-boundary-capacity-constraint-functional

Define the diagnostic difference (not a variational constraint in the action)

$$
\mathcal{C}[G,V]
:=
\underbrace{\int_{\mathcal{Z}} \iota_{\mathrm{bulk}}\, d\mu_G}_{I_{\text{bulk}}}
\;-\;
\underbrace{\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial}}_{C_{\partial}},

$$
where $\iota_{\mathrm{bulk}}$ is the relative-information density in
{prf:ref}`def-information-density-and-bulk-information-volume`. The realised
shutter inflow is the separate rate
$\lambda_{\mathrm{in}}=\mathbb{E}[I(X;K)]$ (Definition
{prf:ref}`def-grounding-rate`). The coupling-window definition
{prf:ref}`thm-information-stability-window-operational` supplies its
admissible operating range.

:::

:::{prf:definition} A.1.3 (Risk Lagrangian density)
:label: def-a-risk-lagrangian-density

Fix a smooth potential $V\in C^\infty(\mathcal{Z})$. A canonical risk Lagrangian density is the scalar-field functional

$$
\mathcal{L}_{\text{risk}}(V;G) := \frac{1}{2}\,G^{ab}\nabla_a V\,\nabla_b V + U(V),

$$
where $U:\mathbb{R}\to\mathbb{R}$ is a (possibly learned) on-site potential capturing non-gradient costs. (The sign convention is chosen for a Riemannian metric; see e.g. Lee, *Riemannian Manifolds*, 2018, for the variational identities used below.)

:::

:::{prf:definition} A.1.4 (Curvature--risk functional with a cutoff penalty)
:label: def-a-capacity-constrained-curvature-functional

Let $R(G)$ be the scalar curvature of $G$ and let $\Lambda\in\mathbb{R}$ be a constant. Define the functional

$$
\mathcal{S}[G,V]
:=
\int_{\mathcal{Z}}\left(R(G)-2\Lambda - 2\kappa\,\mathcal{L}_{\text{risk}}(V;G)\right)d\mu_G
\;-\;
2\kappa\oint_{\partial_{\varepsilon}\mathcal Z}\omega_{\partial},

$$
with coupling $\kappa\in\mathbb{R}$. The sign convention makes the
positive Riemannian risk tensor below appear on the right-hand side of the
stationarity equation. The cutoff term has no interior variation under
clamping, and $\Lambda$ remains a free curvature offset rather than a
multiplier determined by $C_{\partial}$.

*Remark (why $\Lambda$ is allowed).* A constant term in the integrand is the simplest coordinate-invariant scalar density and produces a $\Lambda G_{ij}$ term in the metric Euler–Lagrange equation. Here $\Lambda$ plays the role of a baseline curvature / capacity offset.

:::

:::{prf:lemma} A.3.1 (Divergence-to-boundary conversion)
:label: lem-a-divergence-to-boundary-conversion

For any sufficiently regular information flux field $\mathbf{j}$ on $\mathcal{Z}$,

$$
\int_{\mathcal{Z}} \operatorname{div}_G(\mathbf{j})\, d\mu_G = \oint_{\partial \mathcal{Z}} \langle \mathbf{j}, \mathbf{n}\rangle\, dA_G,

$$
which is the Riemannian divergence theorem underlying the global balance equation in Theorem {prf:ref}`thm-generalized-conservation-of-belief`.

:::

:::{prf:theorem} A.3.2 (Capacity-consistency identity; proof of Theorem {prf:ref}`thm-capacity-constrained-metric-law`)
:label: thm-a-capacity-consistency-identity-proof-of-theorem

Under the hypotheses of Section A.2, stationarity of $\mathcal{S}[G,V]$ with respect to arbitrary variations $\delta G^{ij}$ that vanish on $\partial\mathcal{Z}$ implies the Euler–Lagrange equation

$$
R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij},

$$
with $T_{ij}$ given by Section A.2.3.

:::

:::{prf:remark} Physical interpretation
:label: rem-physical-interpretation

The overdamped limit corresponds to:
- **Information geometry:** The "friction" $\gamma$ represents the rate of information dissipation (forgetting). High friction means the system equilibrates quickly to the local gradient.
- **Diffusion models:** Standard score-based diffusion models operate entirely in the overdamped regime, with $\gamma \to \infty$ implicitly.
- **Neural network training:** The geodesic term $\Gamma(\dot{z},\dot{z})$ can be interpreted as a "momentum correction" that accounts for the curvature of the loss landscape. In standard gradient descent (overdamped), this term is ignored.

:::

:::{prf:proposition} Conditional Classification Relaxation
:label: thm-classification-as-relaxation-a

Under the conservative, deterministic overdamped dynamics with the smooth class-conditioned potential $V_y$:

$$
dz = -G^{-1}(z)\nabla V_y(z)\,ds, \qquad T_c=0,

$$
The limiting chart assignment satisfies $K(\lim_{s\to\infty}z(s))\in\mathcal{A}_y$ whenever the trajectory converges to a minimum in $K^{-1}(\mathcal{A}_y)$ and the initial condition lies in its basin.

:::

:::{prf:remark} Connection to Classification Accuracy
:label: rem-connection-to-classification-accuracy

The theorem provides a geometric interpretation of classification accuracy: a sample $x$ is correctly classified if and only if $\text{Enc}(x) \in \mathcal{B}_{y_{\text{true}}}$. Misclassification occurs when the encoder maps $x$ to the wrong basin—either due to encoder limitations or overlap between class distributions in observation space.

:::

:::{prf:axiom} A.6.0a (Operational Distinguishability)
:label: ax-a-operational-distinguishability

Two probability distributions $p, q \in \mathcal{P}(\mathcal{Z})$ are **operationally distinguishable** if and only if:

$$
D_{\text{KL}}(p \| q) \geq 1 \text{ nat}.

$$
*Justification.* This is an **operational definition**, not a derived fact. The choice of 1 nat as the threshold is grounded in:

1. **Asymptotic error exponent.** For $n$ i.i.d. samples, the optimal Type II error probability at fixed Type I error decays as $\exp(-n \cdot D_{\text{KL}})$ (Stein's lemma). Thus $D_{\text{KL}} = 1$ nat corresponds to error decay rate $e^{-n}$.

2. **Information-theoretic meaning.** 1 nat = log(e) ≈ 1.44 bits represents a "natural unit" of information, where the likelihood ratio $p(x)/q(x)$ has expected log-value 1 under $p$.

3. **Dimensional analysis.** The nat is the natural unit when using natural logarithms; choosing 1 nat as the threshold makes the subsequent formulas dimensionally consistent.

*Remark.* Alternative thresholds (e.g., 1 bit = ln 2 nats) would change the numerical coefficient in the Area Law but not its structure.

:::

:::{prf:theorem} A.6.0b (Chentsov's Uniqueness Theorem)
:label: thm-a-chentsov-uniqueness

For a regular finite-dimensional family of strictly positive distributions, the
**Fisher Information Metric** is, up to constant scaling, the unique Riemannian
metric invariant under all Markov morphisms in the statistical category used by
Chentsov's theorem.

**Statement.** Let $\mathcal{M}$ be a statistical manifold parameterized by $\theta \in \Theta$. Any Riemannian metric $g$ on $\mathcal{M}$ satisfying:
1. **Markov invariance:** $g$ is preserved under the specified Markov morphisms (conditional expectations)
2. **Smoothness and regularity:** $g$ varies smoothly with $\theta$ and the model satisfies the regularity assumptions of the theorem

is proportional to the Fisher Information Metric:

$$
g_{ij}(\theta) = c \cdot \mathbb{E}_\theta\left[\frac{\partial \log p(x|\theta)}{\partial \theta^i} \frac{\partial \log p(x|\theta)}{\partial \theta^j}\right]

$$
for some constant $c > 0$.

:::

:::{prf:definition} A.6.0c (Computational Microstate)
:label: def-a-computational-microstate

A **computational microstate** at resolution $\ell$ is a complete specification of the agent's internal configuration $\mu = (\rho, K, \theta)$ where:
- $\rho \in \mathcal{P}(\mathcal{Z})$ is the belief distribution over the latent manifold
- $K \in \{1, \ldots, |\mathcal{K}|\}$ is the active chart assignment
- $\theta$ are the model parameters

discretized at the Levin Length scale: positions resolved to precision $\ell_L$, probabilities resolved to precision $e^{-1}$ in KL divergence.

Two microstates $\mu_1, \mu_2$ are **boundary-distinguishable** if an external observer, receiving only boundary observations $\partial\mathcal{Z}$, can distinguish them with probability $> 1 - e^{-1}$.

*Remark (Analogy to Physics).* In black hole thermodynamics, a microstate is a specific quantum configuration of the horizon degrees of freedom. Here, a microstate is a specific configuration of the agent's belief state. The boundary plays the role of the horizon: internal distinctions not visible at the boundary do not count toward the entropy.

:::

:::{prf:lemma} A.6.0d (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-probability-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with Fisher Information Metric, the geodesic distance from the uniform distribution $(1/2, 1/2)$ to a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\tfrac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
:::

:::{prf:lemma} A.6.0e (Curvature Normalization and the Factor of 4)
:label: lem-a-curvature-normalization-factor-4

The Poincare disk model with constant sectional curvature $K = -1$ has metric:

$$
ds^2 = \frac{4(dx^2 + dy^2)}{(1-|z|^2)^2}.

$$
The factor of 4 is uniquely determined by the curvature normalization.

:::

:::{prf:definition} A.6.0f (Fisher-coordinate cell convention)
:label: prop-a-area-minimal-distinguishable-cell

On a two-dimensional Poincaré chart with curvature normalization $K=-1$,
$G(0)=4I$.  If the declared coordinate resolution is $\ell_L$, a square
coordinate cell of side $\ell_L$ has Riemannian area

$$
A_{\mathrm{cell}}=4\ell_L^2.
$$

Calling this cell one nat is an operational capacity permit.  The Fisher
metric and Chentsov's theorem fix the local metric up to scale; they do not,
by themselves, identify a finite cell with one nat or prove an area law.
This convention is a two-dimensional chart calculation and is not the
$(D-1)$-dimensional boundary normalization used for the general operational
capacity in {ref}`sec-causal-information-bound`.

:::

:::{prf:proposition} A.6.0g (Conditional boundary-channel convention)
:label: thm-a-boundary-channel-capacity

Suppose a **two-dimensional boundary channel** is explicitly tiled by
independent cells of Riemannian area $4\ell_L^2$, and suppose the channel
permit assigns one nat to each such cell.  Then its declared capacity is

$$
C_\partial=\frac{A}{4\ell_L^2}\ \mathrm{nats}.
$$

This is a conditional counting statement for a two-dimensional boundary
(the boundary dimension is not the same as the $D=2$ Poincaré-disk bulk
case).  The general $D$-dimensional operational capacity is defined with
$\nu_D$ and the dimensionally normalized Levin length in
{ref}`sec-causal-information-bound`; no field equation or Fisher theorem
supplies the cell-to-nat permit.

:::

:::{prf:proposition} A.6.0h (Conditional microstate count)
:label: thm-a-microstate-count-area-law

Under the channel and achievability permit of Proposition
{prf:ref}`thm-a-boundary-channel-capacity`, the number of distinguishable
messages is bounded by

$$
\Omega\le e^{C_\partial},\qquad
\log\Omega\le \frac{A}{4\ell_L^2}.
$$

Equality is an additional coding assumption: it requires an achievable
independent-cell code and a specified input distribution.  The data
processing inequality supplies the upper-bound direction, but it does not
prove equality for an arbitrary latent model.

:::

:::{prf:remark} A.6.1 (Conditional bulk-to-boundary permit)
:label: lem-a-bulk-to-boundary-conversion

A relation of the form

$$
I_{\mathrm{bulk}}=\frac1\kappa\oint_{\partial\mathcal Z}
\operatorname{Tr}(K)\,dA_G
$$

may be adopted as an additional bulk-to-boundary permit for a specified
stationary field theory.  It is not a consequence of the contracted Bianchi
identity: that identity gives a covariantly conserved Einstein tensor, while
$\rho_I$ is an independently defined information density.  A derivation of
this permit must specify the action, source coupling, boundary term, and
units.  None is assumed by the Metric Law elsewhere in the volume.

:::

:::{prf:remark} A.6.2 (Formal spherical saturation ansatz)
:label: prop-a-saturation-metric-solution

For a selected spherical coordinate model one may study

$$
ds^2=A(r)\,dr^2+r^2d\Omega_{n-1}^2,
$$

and introduce a Schwarzschild-style denominator

$$
A(r)^{-1}=1-\frac{2\mu(r)}{(n-2)r^{n-2}}
-\frac{\Lambda_{\mathrm{eff}}r^2}{n(n-1)}.
$$

This is an ansatz, not a solution of the capacity-constrained Metric Law.
Uniform $T_{ij}=\sigma G_{ij}$ instead gives a constant-curvature source
term after the field equation is written out; a mass term and the sign of
$\Lambda_{\mathrm{eff}}$ require a separate boundary-value calculation.
Consequently the functions $\mu$ and $\Lambda_{\mathrm{eff}}$ below are
bookkeeping parameters for the ansatz, not an information mass derived from
$\rho_I$.

:::

:::{prf:definition} A.6.3 (Information Horizon)
:label: def-a-information-horizon

The **information horizon** $r_h$ is the smallest positive root of:

$$
1 - \frac{2\mu(r_h)}{(n-2)r_h^{n-2}} - \frac{\Lambda_{\text{eff}} r_h^2}{n(n-1)} = 0.

$$
At this radius, $A(r_h) \to \infty$ and $G^{rr}(r_h) \to 0$.

:::

:::{prf:remark} A.6.4a (Scope of the Fisher normalization)
:label: rem-a-connection-microstate-counting

The simplex distance and curvature normalization below justify the local
metric convention used in Proposition {prf:ref}`prop-a-area-minimal-cell`.
The statement that a cell carries one nat remains the explicit channel
permit of Proposition {prf:ref}`thm-a-boundary-channel-capacity`; it is not a
consequence of Chentsov's uniqueness theorem.

:::

:::{prf:lemma} A.6.4 (Geodesic Distance on the Probability Simplex)
:label: lem-a-geodesic-distance-simplex

On the 1-simplex $\Delta^1 = \{(p, 1-p) : p \in [0,1]\}$ with the Fisher Information Metric, the geodesic distance between the uniform distribution $(1/2, 1/2)$ and a vertex $(1, 0)$ is:

$$
d_{\text{Fisher}}\left(\frac{1}{2}, 1\right) = \frac{\pi}{2}.

$$
:::

:::{prf:proposition} A.6.5 (Poincaré chart area conversion)
:label: prop-a-area-minimal-cell

On the normalized two-dimensional Poincaré chart, a coordinate cell of side
$\ell_L$ has Riemannian area $4\ell_L^2$ at the origin.  This is the local
geometric conversion used by the conditional cell-counting convention; it
is not a statement about the entropy of an arbitrary data distribution.

:::

:::{prf:definition} A.6.6 (Operational area-law normalization)
:label: thm-a-complete-derivation-area-law

For a declared latent dimension $D$, boundary measure, coefficient $\nu_D$,
and Levin length $\ell_L$, define the operational capacity

$$
I_{\max}:=\nu_D\,
\frac{\operatorname{Area}(\partial\mathcal Z)}{\ell_L^{D-1}}.
$$

This is the normalization used by the Causal Information Capacity in
{ref}`sec-causal-information-bound`.  A field-theoretic derivation would
need, as separate hypotheses, a valid bulk-to-boundary identity, a solution
of the chosen metric equation, a boundary extrinsic-curvature estimate, and
a dimensionally consistent coupling.  The former A.6.1--A.6.3 argument does
not establish those hypotheses, so this definition carries no claim that
the Metric Law generates the area law.

:::

:::{prf:corollary} A.6.7 (Dimension-Dependent Coefficient)
:label: cor-a-dimension-dependent-coefficient

Under the operational capacity convention, a $D$-dimensional latent manifold
with $(D-1)$-sphere boundary uses the dimension-dependent normalization:

$$
I_{\max}(D) = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}},

$$
where the Holographic Coefficient $\nu_D$ (Definition {prf:ref}`def-holographic-coefficient`) is:

$$
\nu_D = \frac{(D-1)\Omega_{D-1}}{8\pi} = \frac{(D-1)\pi^{(D-2)/2}}{4\,\Gamma(D/2)},

$$
with $\Omega_{D-1} = 2\pi^{D/2}/\Gamma(D/2)$ the surface area of the unit $(D-1)$-sphere.

**Explicit values:**

| $D$ | $\Omega_{D-1}$ | $\nu_D$    | Numerical |
|-----|----------------|------------|-----------|
| 2   | $2\pi$         | $1/4$      | 0.250     |
| 3   | $4\pi$         | $1$        | 1.000     |
| 4   | $2\pi^2$       | $3\pi/4$   | 2.356     |
| 5   | $8\pi^2/3$     | $4\pi/3$   | 4.189     |
| 6   | $\pi^3$        | $5\pi^2/8$ | 6.169     |

*Remark.* The coefficient $\nu_D$ is **not monotonic** in $D$: it increases from $D=2$ to a peak at $D \approx 9$ ($\nu_9 \approx 9.4$), then decreases toward zero. For typical latent dimensions ($3 \le D \le 20$), $\nu_D > \nu_2 = 1/4$, so using the 2D coefficient **underestimates** capacity. For very high dimensions ($D \gtrsim 22$), $\nu_D < 1/4$, so the 2D coefficient **overestimates** capacity—this is the dangerous case (false safety). Implementers should always use the dimension-appropriate coefficient.

:::

:::{prf:remark} A.6.8 (Scope of curvature identities)
:label: rem-a-gauss-bonnet-generalization

The contracted Bianchi identity states $\nabla^iG_{ij}=0$; it is not the
boundary-divergence identity
$\int R\,d\mu_G=2\oint\operatorname{Tr}(K)\,dA_G$.  The latter is not valid
for a general manifold or dimension without additional curvature terms,
boundary terms, and field equations.  Classical Gauss--Bonnet identities
have their own dimension and topology hypotheses.  Therefore no such
identity is used to prove the operational capacity in this volume.

:::

:::{prf:remark} A.6.9 (Status of the area-law arguments)
:label: rem-a-non-circularity

The local Fisher/Poincaré calculation and the independent channel permit are
useful normalization checks. They do not derive a universal area law: the
cell-to-nat assignment is a coding assumption, and the former field-theoretic
route requires additional lemmas that are not established here. The main
text consequently presents
$I_{\max}=\nu_D\operatorname{Area}(\partial\mathcal Z)/\ell_L^{D-1}$ as an
operational capacity convention. Any comparison with the
Bekenstein--Hawking formula is an explicitly labelled mathematical analogy,
not a physical identification.

:::

## 10_appendices/04_faq.md

:::{prf:definition} Atomic Belief on a Specified Codebook
:label: def-faq-atomic-codebook-belief

Let $e_1,\ldots,e_m$ be distinct points of the specified latent metric space
$\mathcal Z$. For weights $p_k\geq0$ with $\sum_kp_k=1$, define

$$
\rho_p=\sum_{k=1}^m p_k\delta_{e_k}\in\mathcal P(\mathcal Z).
$$

The weights are recovered by $\rho_p(\{e_k\})=p_k$. A hard VQ assignment is the
special case with one weight equal to one.
:::

## 10_appendices/05_proofs.md

:::{prf:definition} The scalar metric realization
:label: def-e7-strategic-metric

Use the compact connected smooth scalar manifold and its positive strategic
metric from the existing Appendix E.7 setup. The same spectral margin is
$\|G^{-1/2}hG^{-1/2}\|_{\mathrm{op}}<1$ for sign-indefinite perturbations;
positive perturbations preserve positivity directly. The joint volume is
$w=\sqrt{\det\widetilde G}$, as in
{prf:ref}`thm-game-augmented-laplacian`. Compactness makes a fixed smooth
positive metric uniformly elliptic. This statement concerns the compact
realization already used here, not a replacement of unbounded confinement
by an artificial compact domain.
:::

:::{prf:definition} Scalar form realization
:label: def-e7-strategic-hamiltonian

For the existing real $C^2$ potential $U$ bounded below, set
$q_\sigma[u]=\tfrac{\sigma^2}{2}\int|\nabla u|^2+\int U|u|^2$.
The form domain is $H^1$ on a closed manifold or for the Neumann realization,
and $H_0^1$ for the Dirichlet realization. Its self-adjoint operator is
$H_\sigma=-\sigma^2\Delta_{\widetilde G}/2+U$ with the selected boundary
condition. The domain is the operator domain associated to this form,
not unrestricted $H^2$ on a domain with boundary. Compact embedding gives
compact resolvent for this fixed model.
:::

:::{prf:definition} Forbidden and allowed regions
:label: def-e7-forbidden-region

For an energy $E$ define $A_E=\{U\le E\}$ and $K_E=\{U>E\}$.
These are sets of a scalar potential. Their relation to payoff basins is
tested separately by the unilateral inequalities of
{prf:ref}`thm-nash-equilibrium-as-geometric-stasis`.
:::

:::{prf:theorem} Fixed scalar ground state
:label: thm-e7-ground-state-positivity

The compact connected nonmagnetic scalar realization has a simple ground
eigenvalue $E_0$ and an eigenfunction positive in the interior.

:::

:::{prf:definition} Barrier metric at a fixed energy
:label: def-e7-agmon-metric

Define $g_E=2(U-E)_+\widetilde G$ and
$d_E(x,A)=\inf_{\gamma:A\to x}\int\sqrt{2(U-E)_+}\,|\dot\gamma|_{\widetilde G}$.
The factor two matches the kinetic normalization $-\sigma^2\Delta/2$.
Distances may vanish within an allowed connected region.
:::

:::{prf:theorem} Exact weighted eigenfunction identity
:label: thm-e7-agmon-decay-bound

For an eigenfunction $(H_\sigma-E)u=0$ in the scalar realization and a
bounded smooth real weight $f$, put $v=e^{f/\sigma}u$. Then
$$
\frac{\sigma^2}{2}\int|\nabla v|^2
+\int\left(U-E-\frac12|\nabla f|^2\right)|v|^2=0.
$$
All integrals use $d\mu_{\widetilde G}$; the weight is taken constant in
the normal direction for the Neumann case. Compactly supported cutoffs
give the local form, with their explicit derivative terms retained.

:::

:::{prf:corollary} Metric comparison at fixed potential and energy
:label: cor-e7-adversarial-suppression

For $g_1\succeq g_0$ and the same $U,E$, every path obeys
$\int\sqrt{2(U-E)_+}|\dot\gamma|_{g_1}
\ge\int\sqrt{2(U-E)_+}|\dot\gamma|_{g_0}$.
Taking infima gives $d_E^{g_1}\ge d_E^{g_0}$. This compares barrier
actions at the same energy. Changing a Hamiltonian generally changes its
ground energy as well, so one cannot substitute two different ground
energies into this fixed-energy inequality or deduce an ordering of
transition probabilities from two upper bounds. $\square$
:::

:::{prf:theorem} Feynman--Kac clock and normalization
:label: thm-e7-feynman-kac

Let $X_s$ have generator $\Delta_{\widetilde G}/2$, killed at a Dirichlet
boundary or reflected for the Neumann realization. Then
$$
(e^{-tH_\sigma/\sigma^2}\phi)(x)
=\mathbb E_x\left[e^{-\sigma^{-2}\int_0^tU(X_s)ds}\phi(X_t)\right],
$$
with the survival indicator in the killed case. To obtain the ground vector,
$$
e^{tE_0/\sigma^2}e^{-tH_\sigma/\sigma^2}\phi
\longrightarrow\langle u_0,\phi\rangle u_0
\quad\text{in }L^2.
$$
:::

:::{prf:corollary} Action-length inequality
:label: cor-e7-large-deviations

For any absolutely continuous path in the forbidden region, set
$a=|\dot\gamma|_{\widetilde G}$, $b=\sqrt{2(U-E)}$. Then
$a^2/2+U-E-ab=(a-b)^2/2\ge0$.
Integration gives
$\int(|\dot\gamma|^2/2+U-E)dt
\ge\int\sqrt{2(U-E)}|\dot\gamma|dt$.
Equality is attained on a parametrization with $a=b$ wherever that
parametrization is defined. This relates an action to barrier length.
An asymptotic probability additionally belongs to a specified stochastic
law and event; it is not furnished by this algebraic inequality.
:::

:::{prf:definition} Strategic Jacobian on the established response branch
:label: def-strategic-jacobian

On the smooth, nondegenerate local best-response branch specified in the
original strategic construction, differentiate
$\nabla_jV_j(b_j(z_{-j}),z_{-j})=0$. The chain rule gives
$H_{jj}^{(j)}\mathcal J_{ji}+H_{ji}^{(j)}=0$, hence
$\mathcal J_{ji}=-(H_{jj}^{(j)})^{-1}H_{ji}^{(j)}$.
This identifies the branch derivative on the domain where the declared
inverse exists. It is not a global selection of a multivalued best-response
correspondence. Its pullback action on covariant Hessians is exactly that
used in {prf:ref}`def-the-game-tensor`.
:::

## 10_appendices/06_losses.md

:::{prf:definition} F.1.1 (Reconstruction Loss)
:label: def-f-reconstruction-loss

$$
\mathcal{L}_{\text{recon}} = (x - \hat{x})^i \, G_{ij}^{\text{obs}}(x) \, (x - \hat{x})^j
$$

**Parameters:**
- $x, \hat{x}$ – original and reconstructed observations
- $G_{ij}^{\text{obs}}(x)$ – metric tensor on observation space (learned or fixed)

**Purpose:** Ensures charted latents collectively preserve information for reconstruction, with
metric-weighted distances.

**Units:** $[\mathrm{nat}]$ (when scaled appropriately) or metric-weighted MSE.

**Flat limit:** When $G_{ij}^{\text{obs}} = \delta_{ij}$ (identity), recovers standard MSE:
$\|x - \hat{x}\|^2$.

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.2 (Vector Quantization Loss)
:label: def-f-vq-loss

$$
\mathcal{L}_{\text{vq}} = (z_q - z_e)^i \, G_{ij}(z_e) \, (z_q - z_e)^j + \beta \, (z_e - z_q)^i \, G_{ij}(z_q) \, (z_e - z_q)^j
$$

**Parameters:**
- $z_e$ – encoder output (pre-quantization)
- $z_q$ – quantized code embedding $e_{K}$ (per-chart)
- $G_{ij}(z)$ – metric tensor on latent space
- $\beta$ – commitment weight (default 0.25)

**Purpose:** Stabilizes per-chart codebooks. The first term updates code vectors; the second term
encourages the encoder to commit to nearby codes.

**Flat limit:** When $G_{ij} = \delta_{ij}$, recovers standard VQ:
$\|z_q - z_e\|^2 + \beta\|z_e - z_q\|^2$.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.3 (Routing Entropy Loss)
:label: def-f-closure-loss

$$
\mathcal{L}_{\text{entropy}} = \log N_c - \frac{1}{B}\sum_{b=1}^{B}\sum_{k=1}^{N_c} w_{bk}\,\log(w_{bk} + \epsilon)
$$

**Parameters:**
- $w_{bk}$ – router weights over charts
- $N_c$ – number of charts

**Purpose:** Raises per-sample routing entropy and discourages chart collapse. Batch-level usage and
diversity losses are still needed to prevent dead charts.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.4 (Consistency Loss)
:label: def-f-slowness-loss

$$
\mathcal{L}_{\text{consistency}} = \frac{1}{B}\sum_{b=1}^{B} \sum_{k=1}^{N_c} w^{\text{enc}}_{bk}\,\log\left(\frac{w^{\text{enc}}_{bk}+\epsilon}{w^{\text{dec}}_{bk}+\epsilon}\right)
$$

**Parameters:**
- $w^{\text{enc}}$ – encoder router weights
- $w^{\text{dec}}$ – decoder router weights

**Purpose:** Aligns encoder and decoder chart usage.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.5 (Window Loss / Grounding)
:label: def-f-nuisance-kl-loss

$$
\mathcal{L}_{\text{window}} = \max\left(0, \epsilon_{\text{ground}} - I(X;K)\right)^2
$$

**Parameters:**
- $I(X;K) = H(K) - H(K|X)$ – mutual information between input and chart assignment
- $\epsilon_{\text{ground}}$ – grounding threshold

**Purpose:** Enforces the stable learning window by requiring chart assignments to carry
information about inputs.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.6 (Per-Chart Code Entropy Loss)
:label: def-f-texture-kl-loss

$$
\mathcal{L}_{\text{code}} = \frac{1}{N_c}\sum_{k=1}^{N_c} \left(\log K - H(C\mid K=k)\right)
$$

**Parameters:**
- $C$ – code index within each chart
- $K$ – number of codes per chart

**Purpose:** Encourages each chart to use its codebook uniformly rather than collapsing to a
subset.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`

:::

:::{prf:definition} F.1.7 (Total TopoEncoder Loss)
:label: def-f-total-disentangled-loss

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{vq}} +
\lambda_{\text{ent}}\,\mathcal{L}_{\text{entropy}} +
\lambda_{\text{cons}}\,\mathcal{L}_{\text{consistency}} +
\sum_{i \in \text{tiers}} \lambda_i \mathcal{L}_i +
\lambda_{\text{jump}}\,\mathcal{L}_{\text{jump}} +
\lambda_{\text{sup}}\,\mathcal{L}_{\text{sup}}
$$

**Purpose:** Compound loss enforcing sharp routing, charted quantization, and stable geometry.

**Source:** {ref}`Section 3.2 <sec-loss-function-enforcing-macro-micro-separation>`,
Definition {prf:ref}`def-total-disentangled-loss`

:::

:::{prf:definition} F.1.8 (Jump Consistency Loss)
:label: def-f-overlap-consistency

$$
\mathcal{L}_{\text{jump}} = \mathbb{E}_{i \ne j}\left[\|\,z_n^{(j)} - \mathcal{J}_{i \to j}(z_n^{(i)})\,\|^2\right]
$$

**Parameters:**
- $z_n^{(i)}$ – nuisance coordinate from chart $i$
- $\mathcal{J}_{i \to j}$ – learned jump operator from chart $i$ to chart $j$

**Purpose:** Enforces consistency in chart overlaps by learning transitions between chart-local
nuisance coordinates.

**Units:** Dimensionless (metric-weighted embedding distance).

**Source:** {ref}`Section 7 <sec-the-overlap-consistency-loss>`

:::

:::{prf:definition} F.2.1 (Purity Loss / Conditional Entropy)
:label: def-f-purity-loss

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k) = H(Y \mid K)
$$

**Parameters:**
- $P(K=k) = \mathbb{E}_{x \sim \mathcal{D}}[w_k(x)]$ – marginal chart probability
- $H(Y \mid K=k) = -\sum_y P(Y=y \mid K=k) \log P(Y=y \mid K=k)$ – class entropy within chart $k$

**Purpose:** Measures how well charts separate classes. Low purity loss means each chart is associated with a single class. Equivalent to maximizing mutual information $I(K; Y)$ since $\mathcal{L}_{\text{purity}} = H(Y) - I(K; Y)$.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-the-supervised-topology-loss>`, Definition {prf:ref}`def-purity-loss`

:::

:::{prf:definition} F.2.2 (Load Balance Loss)
:label: def-f-balance-loss

$$
\mathcal{L}_{\text{balance}} = D_{\text{KL}}\left(\bar{w} \;\|\; \text{Uniform}(N_c)\right)
$$

**Parameters:**
- $\bar{w} = \mathbb{E}_{x \sim \mathcal{D}}[w(x)]$ – average router weight vector
- $N_c$ – number of charts

**Purpose:** Prevents "dead charts" (collapse to few charts). Encourages all charts to be used productively, addressing the expert-collapse problem in mixture-of-experts systems.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-load-balance-loss>`, Definition {prf:ref}`def-balance-loss`

:::

:::{prf:definition} F.2.3 (Metric Contrastive Loss)
:label: def-f-contrastive-loss

$$
\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_G^{\text{jump}}(z_i, z_j))^2
$$

**Parameters:**
- $\mathcal{P}$ – set of sample pairs in batch
- $w_i, w_j$ – router weight vectors
- $m > 0$ – margin (minimum desired geodesic separation)
- $d_G^{\text{jump}}(z_i, z_j)$ – minimum geodesic jump cost between samples under metric $G$

**Purpose:** Enforces that different-class samples are geometrically far apart in geodesic jump distance. The weighting $w_i^\top w_j$ focuses penalty on hard examples (high routing overlap despite different classes). The geodesic distance respects the curved manifold structure.

**Units:** $[\mathrm{nat}]$

**Flat limit:** When $G_{ij} = \delta_{ij}$, reduces to Euclidean jump distance.

**Source:** {ref}`Section 25.4 <sec-metric-contrastive-loss>`, Definition {prf:ref}`def-contrastive-loss`

:::

:::{prf:definition} F.2.4 (Route Alignment Loss)
:label: def-f-route-alignment-loss

$$
\mathcal{L}_{\text{route}} = \mathbb{E}_{x, y_{\text{true}}}\left[\text{CE}\left(\sum_k w_k(x) \cdot P(Y=\cdot \mid K=k), \; y_{\text{true}}\right)\right]
$$

**Parameters:**
- $w_k(x)$ – router weights for sample $x$ and chart $k$
- $P(Y=\cdot \mid K=k)$ – per-chart class distributions
- $\text{CE}$ – cross-entropy loss

**Purpose:** Primary classification loss. The predicted class distribution (router-weighted average of per-chart distributions) must match the true label.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.4 <sec-route-alignment-loss>`, Definition {prf:ref}`def-route-alignment-loss`

:::

:::{prf:definition} F.2.5 (Combined Supervised Topology Loss)
:label: def-f-total-supervised-loss

$$
\mathcal{L}_{\text{sup-topo}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}
$$

**Typical hyperparameters:**
| Weight | Typical Value | Role |
|--------|---------------|------|
| $\lambda_{\text{pur}}$ | 0.1 | Chart purity |
| $\lambda_{\text{bal}}$ | 0.01 | Load balancing |
| $\lambda_{\text{met}}$ | 0.01 | Metric separation |

**Purpose:** Weighted combination enforcing chart purity, balanced usage, geometric separation, and prediction accuracy.

**Source:** {ref}`Section 25.4 <sec-combined-supervised-topology-loss>`, Definition {prf:ref}`def-total-loss`

:::

:::{prf:definition} F.2.6 (Hierarchical Supervised Loss)
:label: def-f-hierarchical-loss

$$
\mathcal{L}_{\text{hier}} = \sum_{\ell=0}^{L} \alpha_\ell \left(\mathcal{L}_{\text{route}}^{(\ell)} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}}^{(\ell)}\right)
$$

**Parameters:**
- $\ell \in \{0, \ldots, L\}$ – scale levels (bulk to boundary)
- $\alpha_\ell$ – per-scale weights (often $\alpha_\ell = 1$ or decaying)
- $\mathcal{Y}_\ell$ – label space at scale $\ell$ (coarse to fine)

**Purpose:** Enforces classification at multiple scales via stacked TopoEncoders. Coarse (bulk) layers distinguish broad categories; fine (boundary) layers distinguish leaf categories.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25.6 <sec-hierarchical-classification-via-scale-decomposition>`, Definition {prf:ref}`def-hierarchical-supervised-loss`

:::

:::{prf:definition} F.3.1 (Cumulative Cost Functional)
:label: def-f-cumulative-cost

$$
\mathcal{S} = \int \Big(\mathcal{L}_{\text{control}} + C(z_t, a_t)\Big) \, dt
$$

**Parameters:**
- $\mathcal{L}_{\text{control}}$ – control/effort cost (KL penalty, action magnitude)
- $C(z_t, a_t)$ – task cost

**Purpose:** General optimal control objective under information/effort constraints. Specializes to KL-control and entropy-regularized RL.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 2 <sec-the-control-loop-representation-and-control>`

:::

:::{prf:definition} F.3.2 (Instantaneous Regularized Objective)
:label: def-f-instantaneous-objective

$$
F_t := V(Z_t) + \beta_K\big(-\log p_\psi(K_t)\big) + \beta_n D_{\mathrm{KL}}(q(z_{n,t} \mid x_t) \| p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex},t} \mid x_t) \| p(z_{\mathrm{tex}})) + T_c D_{\mathrm{KL}}(\pi(\cdot \mid K_t) \| \pi_0(\cdot \mid K_t))
$$

**Parameters:**
- $V(Z_t)$ – task-aligned cost-to-go (critic estimate)
- $\beta_K(-\log p_\psi(K_t))$ – macro codelength penalty (Occam's razor for discrete state)
- $\beta_n, \beta_{\text{tex}}$ – residual regularization weights
- $T_c$ – cognitive temperature
- $\pi_0$ – prior policy

**Purpose:** Trades off task cost, representation complexity, and control effort, all in consistent units (nats).

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 3.2 <sec-the-entropy-regularized-objective-functional>`, Definition {prf:ref}`def-f-instantaneous-objective`

:::

:::{prf:definition} F.3.3 (Monotonicity Surrogate Loss)
:label: def-f-monotonicity-loss

$$
\mathcal{L}_{\downarrow F} := \mathbb{E}\left[\mathrm{ReLU}(F_{t+1} - F_t)^2\right]
$$

**Purpose:** Penalizes increases in the instantaneous objective $F_t$ from one step to the next, encouraging trajectories that smoothly descend the objective landscape.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 3.2 <sec-the-entropy-regularized-objective-functional>`

:::

:::{prf:definition} F.3.4 (Closure Ratio Diagnostic)
:label: def-f-closure-ratio

$$
\text{Closure Ratio} = \frac{\mathbb{E}[-\log p_\psi(K_{t+1} \mid K_t, a_t)]}{\mathbb{E}[-\log p_{\text{base}}(K_{t+1})]} = \frac{H(K_{t+1} \mid K_t, a_t)}{H(K_{t+1})}
$$

**Interpretation:**
| Ratio | Meaning | Action |
|-------|---------|--------|
| $\ll 1$ | Strong predictive law learned | Success |
| $\approx 1$ | No predictive law | Increase model capacity |
| $> 1$ | Worse than baseline | Bug/degeneracy |

**Purpose:** Measures how much better the macro dynamics model predicts $K_{t+1}$ compared to a marginal baseline. The gap estimates predictive information $I(K_{t+1}; K_t, a_t)$.

**Units:** Dimensionless.

**Source:** {ref}`Runtime routing diagnostics <sec-runtime-diagnostics-the-closure-ratio>`,
Definition {prf:ref}`def-f-closure-ratio`

:::

:::{prf:definition} F.3.5 (Causal Information Potential)
:label: def-f-causal-info-potential

$$
\Psi_{\text{causal}}(z, a) := \mathbb{E}_{z' \sim \bar{P}(\cdot | z, a)} \left[ D_{\text{KL}} \left( p(\theta_W | z, a, z') \| p(\theta_W | z, a) \right) \right]
$$

**Parameters:**
- $z, a$ – current state and action
- $z'$ – next state sampled from world model
- $\theta_W$ – world model parameters
- $\bar{P}$ – world model transition distribution

**Purpose:** Measures the **Expected Information Gain** about world model parameters from executing action $a$ at state $z$. High $\Psi_{\text{causal}}$ indicates the outcome will resolve significant uncertainty about dynamics. Drives intrinsic motivation for exploration via Bayesian experimental design.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 29 <sec-the-causal-information-potential>`, Definition {prf:ref}`def-causal-information-potential`

:::

:::{prf:definition} F.4.1 (Hodge Decomposition of Reward)
:label: def-f-hodge-decomposition

$$
\mathcal{R} = \underbrace{d\Phi}_{\text{Gradient}} + \underbrace{\delta \Psi}_{\text{Solenoidal}} + \underbrace{\eta}_{\text{Harmonic}}
$$

**Components:**
- $d\Phi$ – Gradient/conservative reward component (the control-loop cost critic is $V=-\Phi$)
- $\delta\Psi$ – Solenoidal/rotational component (cyclic reward structure)
- $\eta$ – Harmonic component (topological cycles from manifold holes)

**Purpose:** Decomposes reward 1-form into orthogonal components. Separates optimizable value from inherently cyclic structure (e.g., Rock-Paper-Scissors).

**Units:** $[\Phi] = \mathrm{nat}$, $[\Psi] = \mathrm{nat} \cdot [\text{length}]^2$, $[\eta] = \mathrm{nat}/[\text{length}]$.

**Source:** {ref}`Section 18.2 <sec-hodge-decomposition-of-value>`, Theorem {prf:ref}`thm-hodge-decomposition`

:::

:::{prf:definition} F.4.2 (Value Curl / Vorticity)
:label: def-f-value-curl

$$
\mathcal{F}_{ij} := \partial_i \mathcal{R}_j - \partial_j \mathcal{R}_i = d\mathcal{R}
$$

**Properties:**
- Antisymmetric: $\mathcal{F}_{ij} = -\mathcal{F}_{ji}$
- Satisfies Bianchi identity: $d\mathcal{F} = 0$
- Gauge-invariant under $\mathcal{R} \to \mathcal{R} + d\chi$

**Purpose:** Detects non-conservative reward structure. Non-zero curl indicates orbiting strategies may be optimal. Diagnostic: $\oint_\gamma \mathcal{R} = \int_\Sigma \mathcal{F} \, d\Sigma \neq 0$ implies non-conservative rewards.

**Units:** $[\mathcal{F}] = \mathrm{nat}/[\text{length}]^2$

**Source:** {ref}`Section 18.2 <sec-hodge-decomposition-of-value>`, Definition {prf:ref}`def-value-curl`

:::

:::{prf:definition} F.4.3 (Class-Conditioned Potential)
:label: def-f-class-potential

$$
V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K)
$$

**Parameters:**
- $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ – learnable chart-to-class affinities
- $V_{\text{base}}(z, K)$ – unconditioned critic
- $\beta_{\text{class}} > 0$ – class temperature (inverse semantic diffusion)

**Purpose:** Shapes potential landscape so class-$y$ regions become energy minima. Used for both classification (relaxation inference) and generation (Langevin sampling).

**Units:** $[V_y] = \mathrm{nat}$

**Source:** {ref}`Section 25.2 <sec-the-semantic-potential>`, Definition {prf:ref}`def-class-conditioned-potential`

:::

:::{prf:definition} F.5.1 (Synchronization Potential)
:label: def-f-sync-potential

$$
\mathcal{L}_{\text{sync}} = \beta \Psi_{\text{sync}},
\qquad
\Psi_{\text{sync}}=\int_{\mathcal{D}_{AB}}\operatorname{tr}\!\left(\mathcal{F}_{AB}\wedge *_{{G_{AB}}}\mathcal{F}_{AB}\right)
$$

**Parameters:**
- $\beta$ – coupling strength
- $\mathcal{F}_{AB}$ – Locking curvature of the selected relative connection

**Purpose:** Penalizes curvature of the selected relative connection and can drive gauge locking under the hypotheses
of the conditional strong-coupling proposition. It does not by itself synchronize the private metrics.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`the inter-subjective metric chapter <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`

:::

:::{prf:definition} F.6.1 (Ontological Stress)
:label: def-f-ontological-stress

$$
\Xi = \sum_{\ell=1}^{L} \left( z_{\text{tex}}^{(\ell)} \right)^i G_{ij}^{(\ell)} \left( z_{\text{tex}}^{(\ell)} \right)^j
$$

**Parameters:**
- $z_{\text{tex}}^{(\ell)}$ – texture embedding at scale $\ell$
- $G_{ij}^{(\ell)}$ – metric tensor at scale $\ell$

**Purpose:** Measures predictability *within* texture across scales. High stress indicates ontological inadequacy---texture contains compressible structure that should have been captured by macro/nuisance. Dual to closure defect: closure measures micro-to-macro leakage; ontological stress measures within-texture predictability.

**Units:** Dimensionless (metric-weighted embedding norm).

**Flat limit:** When $G_{ij}^{(\ell)} = \delta_{ij}$, recovers $\sum_\ell \|z_{\text{tex}}^{(\ell)}\|^2$.

**Source:** {ref}`Section 33 <sec-ontological-expansion-topological-fission-and-the-semantic-vacuum>`

:::

:::{prf:definition} F.7.1 (Gradient Penalty Loss)
:label: def-f-gradient-penalty

$$
\mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} \left[\left(\|\nabla_A V\|_G - K\right)^2\right], \qquad \|\nabla_A V\|_G^2 := G^{ij}(\hat{s}) \, (\partial_i V - A_i) \, (\partial_j V - A_j)
$$

**Parameters:**
- $\hat{s}$ – interpolated samples between real and generated
- $V(\hat{s})$ – critic value at sample
- $G^{ij}(\hat{s})$ – inverse metric tensor (contravariant) at sample
- $K$ – target gradient norm (typically 1)

**Purpose:** Enforces Lipschitz constraint on the critic using the metric-induced norm. The covariant gradient norm $\|\nabla_A V\|_G$ measures the gauge-invariant rate of change along geodesics. Prevents vanishing gradients in flat value regions (BarrierGap) and ensures smooth value landscape for stable learning.

**Units:** Dimensionless.

**Flat limit:** When $G^{ij} = \delta^{ij}$ and $A=0$, recovers $(\|\nabla_A V\|_2 - K)^2$.

**Source:** {ref}`Section 4 <sec-barrier-implementation-details>`

:::

:::{prf:definition} F.7.2 (Information-Control Loss)
:label: def-f-info-control

$$
\mathcal{L}_{\text{InfoControl}} = \underbrace{\beta_K \mathbb{E}[-\log p_\psi(K)] + \beta_n D_{\mathrm{KL}}(q(z_n \mid x) \| p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x) \| p(z_{\mathrm{tex}}))}_{\text{Compression (Rate)}} + \underbrace{\gamma \mathbb{E}[\mathfrak{D}(Z, A)]}_{\text{Control Effort}}
$$

**Parameters:**
- $\beta_K, \beta_n, \beta_{\text{tex}}$ – compression weights for macro/nuisance/texture
- $\mathfrak{D}(Z, A)$ – actuation cost (KL-control or action norm)
- $\gamma$ – control effort weight

**Purpose:** Balances the Information-Control Tradeoff (BarrierScat vs BarrierCap). High compression removes details needed for fine control; this loss finds the Pareto frontier.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

:::{prf:definition} F.7.3 (Elastic Weight Consolidation)
:label: def-f-ewc

$$
\mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,\text{old}})^2
$$

**Parameters:**
- $F_i$ – diagonal Fisher Information for parameter $i$
- $\theta_i$ – current parameter value
- $\theta^*_{i,\text{old}}$ – parameter value from previous task

**Purpose:** Addresses the Stability-Plasticity Dilemma (BarrierVac vs BarrierPZ). High-sensitivity weights (large $F_i$) are constrained to preserve past learning; low-sensitivity weights can adapt freely.

**Units:** Dimensionless (parameter space distance, Fisher-weighted).

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

:::{prf:definition} F.7.4 (Bode Magnitude Loss)
:label: def-f-bode

$$
\mathcal{L}_{\text{Bode}} = \|\mathcal{F}(e_t) \cdot W(\omega)\|^2
$$

**Parameters:**
- $\mathcal{F}(e_t)$ – Fourier transform of error signal
- $W(\omega)$ – frequency weighting function

**Purpose:** Addresses the Bode Sensitivity Integral (BarrierBode). Suppressing error in one frequency band amplifies it in another (waterbed effect). This loss explicitly chooses where to be sensitive vs. blind.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 4 <sec-b-cross-barrier-regularization>`

:::

:::{prf:definition} F.8.1 (Quantum Speed Limit Loss)
:label: def-f-qsl

$$
\mathcal{L}_{\text{QSL}} := \mathrm{ReLU}\left(d_G(z_{t+1}, z_t) - v_{\max}\right)^2
$$

**Parameters:**
- $d_G(z_{t+1}, z_t)$ – geodesic distance traveled in one step
- $v_{\max}$ – maximum allowed velocity in latent space

**Purpose:** Enforces the Quantum Speed Limit: belief cannot change faster than the Mandelstam-Tamm bound allows. Prevents unrealistic jumps in belief state.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 11 <sec-belief-dynamics-prediction-update-projection>`

:::

:::{prf:definition} F.8.2 (Joint Prediction Loss)
:label: def-f-joint-prediction

$$
\mathcal{L}_{\text{joint}} = d_G(\hat{x}_{t+1}^A, x_{t+1})^2 + d_G(\hat{x}_{t+1}^B, x_{t+1})^2 + \beta \Psi_{\text{sync}}
$$

Expanded in coordinates:

$$
d_G(\hat{x}, x)^2 = (\hat{x} - x)^i \, G_{ij}^{\text{obs}}(x) \, (\hat{x} - x)^j
$$

**Parameters:**
- $\hat{x}_{t+1}^A, \hat{x}_{t+1}^B$ – predictions from agents $A$ and $B$
- $x_{t+1}$ – actual next observation
- $G_{ij}^{\text{obs}}$ – metric tensor on observation space
- $\Psi_{\text{sync}}$ – synchronization potential
- $\beta$ – coupling strength

**Purpose:** Multi-agent world model training. Both agents must predict accurately (measured under the observation-space metric), and their representations must synchronize (gauge lock).

**Units:** Metric-weighted prediction error + $[\mathrm{nat}]$.

**Flat limit:** When $G_{ij}^{\text{obs}} = \delta_{ij}$, recovers $\|\hat{x}^A - x\|^2 + \|\hat{x}^B - x\|^2 + \beta\Psi_{\text{sync}}$.

**Source:** {ref}`the inter-subjective metric chapter <sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality>`

:::

:::{prf:definition} F.9.1 (VICReg Loss)
:label: def-f-vicreg

$$
\mathcal{L}_{\text{VICReg}} = \lambda \mathcal{L}_{\text{inv}} + \mu \mathcal{L}_{\text{var}} + \nu \mathcal{L}_{\text{cov}}
$$

**Components:**

$$
\begin{aligned}
\mathcal{L}_{\text{inv}} &= d_G(z, z')^2 = (z - z')^i \, G_{ij}(z) \, (z - z')^j & \text{(invariance)} \\
\mathcal{L}_{\text{var}} &= \frac{1}{d} \sum_{j=1}^{d} \max\left(0, \gamma - \sqrt{\text{Var}_G(z^j) + \epsilon}\right) & \text{(variance)} \\
\mathcal{L}_{\text{cov}} &= \frac{1}{d} \sum_{i \neq j} \left[G^{-1/2} \text{Cov}(z) \, G^{-1/2}\right]_{ij}^2 & \text{(covariance)}
\end{aligned}
$$

**Parameters:**
- $z, z'$ – embeddings of two augmented views of the same input
- $G_{ij}(z)$ – metric tensor on embedding space
- $\gamma$ – variance threshold (typically 1)
- $\lambda, \mu, \nu$ – component weights (typically $\lambda = 25$, $\mu = \nu = 1$)

**Purpose:** Prevents representation collapse without negative samples. Invariance pulls augmented views together (geodesic distance); variance prevents dimension collapse (metric-aware); covariance decorrelates dimensions (whitened by metric).

**Units:** Dimensionless.

**Flat limit:** When $G_{ij} = \delta_{ij}$, recovers standard VICReg with $\|z - z'\|^2$.

**Source:** {ref}`Section 3 <sec-diagnostics-stability-checks>` (GeomCheck, Node 6)

:::

:::{prf:definition} F.9.2 (InfoNCE Loss)
:label: def-f-infonce

$$
\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp\left(-d_G(z_t, z_{t+k})^2 / \tau\right)}{\sum_{j} \exp\left(-d_G(z_t, z_j)^2 / \tau\right)}
$$

**Parameters:**
- $z_t, z_{t+k}$ – embeddings at current and future timesteps
- $z_j$ – negative samples (other timesteps or other sequences)
- $d_G(z, z')$ – geodesic distance under metric $G$
- $\tau$ – temperature parameter (squared distance scale)

**Purpose:** Contrastive predictive coding with geodesic similarity. Anchors macro latents to temporal structure by maximizing mutual information between present and future representations. The geodesic kernel $k_G(z, z') = \exp(-d_G(z,z')^2/\tau)$ respects the curved geometry of the latent manifold.

**Units:** $[\mathrm{nat}]$

**Flat limit:** When $G_{ij} = \delta_{ij}$ and using $\text{sim}(z,z') = -\|z-z'\|^2$, recovers standard InfoNCE with Gaussian kernel.

**Source:** {ref}`Section 3 <sec-diagnostics-stability-checks>` (GeomCheck, Node 6)

:::

:::{prf:definition} F.10.1 (Behavior Cloning Loss)
:label: def-f-bc

$$
\mathcal{L}_{\text{BC}} = \mathbb{E}_{(s, a^*) \sim \mathcal{D}_{\text{expert}}}[-\log \pi(a^* \mid s)]
$$

**Parameters:**
- $(s, a^*)$ – state-action pairs from expert demonstrations
- $\mathcal{D}_{\text{expert}}$ – expert demonstration dataset
- $\pi(a \mid s)$ – learned policy

**Purpose:** Supervised policy learning. Trains the policy to match expert actions via maximum likelihood.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 25 <sec-supervised-topology-semantic-potentials-and-metric-segmentation>`

:::

:::{prf:definition} F.11.1 (Metric-law residual)
:label: def-f-efe-loss

Let

$$
E_{ij}:=R_{ij}-\frac12R\,G_{ij}+\Lambda G_{ij}-\kappa T_{ij}.
$$

The coordinate-invariant metric-law loss is

$$
\mathcal{L}_{\mathrm{EFE}}
:=\int_{\mathcal Z}G^{ik}G^{jl}E_{ij}E_{kl}\,d\mu_G,
$$

or its minibatch approximation. It measures violation of the curvature--risk
stationarity identity; it does not replace the separate capacity diagnostic.

**Source:** {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits`, Theorem {prf:ref}`thm-capacity-constrained-metric-law`

:::

:::{prf:definition} F.11.2 (WFR Consistency Loss)
:label: def-f-wfr-consistency

$$
\mathcal{L}_{\text{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2
$$

**Parameters:**
- $\rho_t$ – belief density at time $t$
- $r_t$ – reaction rate (birth/death)
- $v_t$ – transport velocity field
- $\Delta t$ – timestep

**Purpose:** Enforces Wasserstein-Fisher-Rao consistency. Penalizes deviations from the unbalanced continuity equation in cone-space formulation.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 20 <sec-wfr-dynamics-with-memory-sources>`

:::

:::{prf:definition} F.11.2 (Critic TD Loss with PDE Regularization)
:label: def-f-critic-td

In the control-loop cost convention, write $c_t:=-r_t$ and $\rho_c:=-\rho_r$. The critic loss is

$$
\mathcal{L}_{\text{critic}} = \|c_t + \gamma V(s') - V(s)\|^2 + \lambda_{\text{PDE}} \| -\Delta_G V + \kappa^2 V - \rho_c \|^2.
$$

**Parameters:**
- TD-Error $= c + \gamma V(s') - V(s)$ – cost-convention temporal difference error
- $\Delta_G$ – Laplace-Beltrami operator on manifold
- $\lambda = -\ln\gamma/\Delta t$ and $\kappa^2=\lambda/T_c$ – stationary-diffusion screening coefficient from the discount factor
- $\rho_c=-\rho_r$ – cost density (the reward-side equation uses $\Phi=-V$ and $\rho_r$)

**Purpose:** Combines TD learning with Helmholtz PDE regularization. The PDE term enforces that the critic satisfies the continuum Bellman equation.

**Units:** $[\mathrm{nat}^2]$

**Source:** {ref}`Section 18 <sec-the-reward-field-value-forms-and-hodge-geometry>`

:::

:::{prf:definition} F.12.1 (Waste Quotient)
:label: def-f-waste-quotient

$$
W_\mathcal{P} := 1 - \frac{\Delta I_{\text{world}}}{\int \dot{\mathcal{M}}(t) \, dt}
$$

**Parameters:**
- $\Delta I_{\text{world}}$ – mutual information gained about world
- $\dot{\mathcal{M}}(t)$ – metabolic flux (energy dissipation rate)

**Interpretation:**
| Protocol | Waste Quotient | Meaning |
|----------|----------------|---------|
| Bitcoin PoW | $W_{\text{BTC}} \approx 1$ | Energy produces zero world knowledge |
| Target PoUW | $W_{\text{PoUW}} \to 0$ | Energy produces useful learning |

**Purpose:** Measures efficiency of consensus protocol. Low waste quotient means energy dissipation produces useful information gain.

**Units:** Dimensionless.

**Source:** {ref}`Section 38.1 <sec-the-thermodynamic-inefficiency-of-nakamoto-consensus>`, Definition {prf:ref}`def-waste-quotient`

:::

:::{prf:definition} F.13.1 (Governor Training Regret)
:label: def-f-governor-regret

$$
J(\phi) = \mathbb{E}_{\mathcal{T} \sim P(\mathcal{T})} \left[ \sum_{t=0}^T \left( \mathcal{L}_{\text{task}}(\theta_t) + \gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2 \right) \right]
$$

**Parameters:**
- $\phi$ – Governor parameters
- $\mathcal{T}$ – task from task distribution
- $\mathcal{L}_{\text{task}}(\theta_t)$ – task loss at training step $t$
- $C_k(\theta_t)$ – constraint $k$ value (negative when satisfied)
- $\gamma_{\text{viol}}$ – constraint violation penalty weight

**Purpose:** Meta-learning objective for the Governor. Minimizes cumulative task loss (convergence speed) plus squared constraint violations (feasibility). The Governor learns to set hyperparameters $\Lambda$ that lead to fast, stable training across diverse tasks.

**Units:** $[\mathrm{nat}]$

**Source:** {ref}`Section 26 <sec-bilevel-optimization-objective>`, Definition {prf:ref}`def-outer-problem-governor-optimization`

:::

## 10_appendices/07_architecture.md

:::{prf:definition} G.1.1 (TopoEncoderConfig)
:label: def-g-disentangled-config

**Class signature:**
```python
@dataclass
class TopoEncoderConfig:
    input_dim: int = 784
    hidden_dim: int = 32
    latent_dim: int = 2
    num_charts: int = 10
    codes_per_chart: int = 32
    covariant_attn: bool = True
    covariant_attn_tensorization: str = "full"
    covariant_attn_rank: int = 8
    covariant_attn_tau_min: float = 1e-2
    covariant_attn_denom_min: float = 1e-3
    covariant_attn_use_transport: bool = True
    covariant_attn_transport_eps: float = 1e-3
    vision_preproc: bool = False
    soft_equiv_metric: bool = False
    soft_equiv_temperature: float = 1.0
```

**Purpose:** Configuration dataclass for the TopoEncoder benchmark in
`src/experiments/topoencoder_2d.py`. Controls chart routing, codebook sizes, and optional
covariant/soft-equivariant components.

**Key parameters:**
- `num_charts`, `codes_per_chart` – atlas resolution
- `covariant_attn_*` – routing tensorization and transport
- `vision_preproc` – CovariantRetina feature extractor toggle
- `soft_equiv_metric`, `soft_equiv_temperature` – per-chart metric control

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, `topoencoder_2d.py`.
:::

:::{prf:definition} G.1.2 (PrimitiveAttentiveAtlasEncoder)
:label: def-g-encoder

**Class signature:**
```python
class PrimitiveAttentiveAtlasEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_charts: int, codes_per_chart: int, ...):
        ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        ...
```

**Input/Output:**
- Input: `x` shape `[B, D_in]` or `[B, C, H, W]`
- Output (ordered): `K_chart`, `K_code`, `z_n`, `z_tex`, `router_weights`, `z_geo`,
  `vq_loss`, `indices_stack`, `z_n_all_charts`, `c_bar`

**Purpose:** Encodes inputs into charted VQ latents with typed residuals. Uses chart routing to
select per-chart codebooks and produces geometry `z_geo` and texture `z_tex`.

**Key parameters:** `num_charts`, `codes_per_chart`, `covariant_attn_*`, `vision_preproc`,
`soft_equiv_metric`.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, `atlas.py`.
:::

:::{prf:definition} G.1.3 (CovariantChartRouter)
:label: def-g-vector-quantizer

**Class signature:**
```python
class CovariantChartRouter(nn.Module):
    def __init__(self, latent_dim: int, key_dim: int, num_charts: int, feature_dim: int | None = None, ...):
        ...

    def forward(self, z: torch.Tensor, features: torch.Tensor | None = None, chart_tokens: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: `z` shape `[B, D]`, optional `features` shape `[B, H]`
- Output: `(router_weights, K_chart)` where `router_weights` is `[B, N_c]`

**Purpose:** Gauge-covariant chart routing with Wilson-line transport and metric-aware temperature.

**Key parameters:** `tensorization`, `rank`, `tau_min`, `tau_denom_min`, `use_transport`,
`transport_eps`.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, `atlas.py`.
:::

:::{prf:definition} G.1.4 (PrimitiveTopologicalDecoder)
:label: def-g-decoder

**Class signature:**
```python
class PrimitiveTopologicalDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, num_charts: int, output_dim: int, ...):
        ...

    def forward(self, z_geo: torch.Tensor, z_tex: torch.Tensor | None = None, chart_index: torch.Tensor | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: `z_geo` shape `[B, D]`, optional `z_tex` shape `[B, D]`
- Output: `(x_hat, router_weights)` where `x_hat` is `[B, D_out]`

**Purpose:** Decodes charted geometry latents into reconstructions using chart projectors, a shared
renderer, and an optional texture residual path.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, `atlas.py`.
:::

:::{prf:definition} G.1.5 (TopoEncoderPrimitives)
:label: def-g-macro-dynamics-model

**Class signature:**
```python
class TopoEncoderPrimitives(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_charts: int, codes_per_chart: int, ...):
        ...

    def forward(self, x: torch.Tensor, use_hard_routing: bool = False) -> Tuple[torch.Tensor, ...]:
        ...
```

**Input/Output:**
- Input: `x` shape `[B, D_in]`
- Output: `(x_recon, vq_loss, enc_weights, dec_weights, K_chart, z_geo, z_n, c_bar)`

**Purpose:** Wrapper that couples encoder and decoder; exposes consistency loss and chart usage
perplexity helpers.

**Source:** {ref}`Section 3.2 <sec-architecture-the-disentangled-vq-vae-rnn>`, `atlas.py`.
:::

:::{prf:definition} G.1.6 (HierarchicalAtlasStack)
:label: def-g-disentangled-agent

**Purpose:** Multi-scale atlas stack that extends the TopoEncoder with multiple charted codebooks,
as defined in {ref}`Section 3.2 <sec-advanced-hierarchical-multi-scale-latents>` and Definition
{prf:ref}`def-hierarchical-latent`.

**Implementation sketch:**
- Shared feature extractor, multiple chart routers and codebooks
- Coarser levels update more slowly than fine levels
- Jump operator links charts across levels when enabled
:::

:::{prf:definition} G.1.7 (TopoEncoderAttachments)
:label: def-g-hierarchical-disentangled

Optional modules frequently attached to the TopoEncoder stack:

- `CovariantRetina` (feature extractor) – {prf:ref}`def-g-covariant-retina`
- `SoftEquivariantLayer` (metric) – {prf:ref}`def-g-soft-equivariant-layer`
- `FactorizedJumpOperator` (chart transitions)
- `InvariantChartClassifier` (detached readout)
:::

:::{prf:definition} G.2.1 (SupervisedTopologyLoss)
:label: def-g-supervised-topology-loss

**Class signature:**
```python
class SupervisedTopologyLoss(nn.Module):
    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        ...

    def forward(
        self,
        chart_assignments: torch.Tensor,  # [B, N_c] soft assignments
        class_labels: torch.Tensor,        # [B] ground truth classes
        embeddings: torch.Tensor,          # [B, D] latent embeddings
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ...
```

**Input/Output:**
- Input:
  - `chart_assignments` shape `[B, N_c]` – Soft chart assignments (router weights)
  - `class_labels` shape `[B]` – Ground truth class labels
  - `embeddings` shape `[B, D]` – Latent embeddings
- Output: `(total_loss, loss_dict)` where `loss_dict` contains individual loss terms

**Purpose:** Enforces that each chart is dominated by a single class (purity), charts are used roughly equally (balance), and same-class samples are metrically closer (separation).

**Key parameters:**
- `num_charts` – Number of atlas charts $N_c$
- `num_classes` – Number of semantic classes $C$
- `lambda_purity` – Weight for chart purity loss (Definition {prf:ref}`def-purity-loss`)
- `lambda_balance` – Weight for chart balance loss
- `lambda_metric` – Weight for metric contrastive loss

**Learnable parameters:**
- `chart_to_class` shape `[N_c, C]` – Logits mapping charts to class probabilities

**Loss components:**
1. **Chart Purity:** $\mathcal{L}_{\text{purity}} = -\sum_k \max_c p(c|k) \log \max_c p(c|k)$
2. **Chart Balance:** $\mathcal{L}_{\text{balance}} = D_{\text{KL}}(\bar{p}(k) \| \text{Uniform})$
3. **Metric Contrastive:** Encourages intra-class proximity, inter-class separation

**Source:** {ref}`Section 25.4 <sec-the-supervised-topology-loss>`, Definition {prf:ref}`def-total-loss`, line 680.
:::

:::{prf:definition} G.2.2 (class_modulated_jump_rate)
:label: def-g-class-modulated-jump-rate

**Function signature:**
```python
def class_modulated_jump_rate(
    lambda_base: torch.Tensor,     # [N_c, N_c] base jump rates
    chart_to_class: torch.Tensor,  # [N_c, C] learnable logits
    gamma_sep: float = 5.0,        # Separation strength
) -> torch.Tensor:
    ...
```

**Input/Output:**
- Input:
  - `lambda_base` shape `[N_c, N_c]` – Base jump rate matrix
  - `chart_to_class` shape `[N_c, C]` – Chart-to-class mapping logits
  - `gamma_sep` – Separation strength coefficient
- Output: `lambda_sup` shape `[N_c, N_c]` – Class-modulated jump rates

**Purpose:** Computes class-consistent jump rates that suppress transitions between charts of different dominant classes, implementing the class-modulated rate from Definition {prf:ref}`def-class-consistent-jump-rate`.

**Mathematical operation:**
$$\lambda_{kk'}^{\text{sup}} = \lambda_{kk'}^{\text{base}} \cdot \exp(-\gamma_{\text{sep}} \cdot D_{\text{class}}(k, k'))$$

where $D_{\text{class}}(k, k') = 1$ if charts $k$ and $k'$ have different dominant classes, else $0$.

**Key parameters:**
- `gamma_sep` – Controls how strongly cross-class jumps are suppressed (higher = stronger suppression)

**Source:** {ref}`Section 25.3 <sec-metric-segmentation-via-jump-rate-modulation>`, Definition {prf:ref}`def-class-consistent-jump-rate`, line 445.
:::

:::{prf:definition} G.3.1 (LorentzianConfig)
:label: def-g-lorentzian-config

**Class signature:**
```python
@dataclass
class LorentzianConfig:
    d_model: int = 256        # Model dimension [nat]
    d_latent: int = 64        # Latent space dimension
    n_heads: int = 4          # Number of attention heads
    c_info: float = 1.0       # Information speed (latent units per timestep)
    T_c: float = 0.1          # Cognitive temperature [nat/step]
    gamma_friction: float = 1.0  # Friction coefficient for O-step
    dt: float = 0.01          # Integration timestep
```

**Purpose:** Configuration for Lorentzian memory attention with causal structure.

**Key parameters:**
- `c_info` – Information speed $c_{\text{info}}$ defining the light cone (Definition {prf:ref}`def-information-speed-recap`)
- `d_latent` – Dimension of the latent manifold $\mathcal{Z}$

**Units:** `d_model` and `d_latent` in [nat], `c_info` in [latent units/timestep], `T_c` in [nat/step].

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, line 864.
:::

:::{prf:definition} G.3.2 (LorentzianMetric)
:label: def-g-lorentzian-metric

**Class signature:**
```python
class LorentzianMetric(nn.Module):
    def __init__(self, config: LorentzianConfig, epsilon: float = 1e-6):
        ...

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        ...

    def spacetime_interval(self, z: torch.Tensor, t: torch.Tensor,
                           z_mem: torch.Tensor, t_mem: torch.Tensor) -> torch.Tensor:
        ...

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        ...
```

**Input/Output:**
- `conformal_factor`: Input `z` shape `[B, d]` → Output `[B, 1]`
- `geodesic_distance`: Input `z1` shape `[B, d]`, `z2` shape `[B, N, d]` → Output `[B, N]`
- `spacetime_interval`: Input positions and times → Output `[B, N]` intervals
- `temperature`: Input `z` shape `[B, d]` → Output `[B, 1]`

**Purpose:** Implements the Lorentzian metric on the memory manifold $\mathcal{M} = \mathbb{R} \times \mathcal{Z}$ with signature $(-,+,\ldots,+)$.

**Key methods:**
- `conformal_factor`: $\lambda(z) = 2/(1-|z|^2)$ (Poincaré disk)
- `geodesic_distance`: $d_G(z, z') = \operatorname{arcosh}(1 + 2|z-z'|^2/((1-|z|^2)(1-|z'|^2)))$
- `spacetime_interval`: $\Delta s^2_{\text{eff}} = -c_{\text{info}}^2(t-t')^2 + d_G^2$ (Definition {prf:ref}`def-spacetime-interval`)
- `temperature`: $\tau(z) = \sqrt{d_k}/\lambda(z)$ (Theorem {prf:ref}`thm-metric-temperature-correspondence`)

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-lorentzian-memory-manifold`, line 886.
:::

:::{prf:definition} G.3.3 (CausalMask)
:label: def-g-causal-mask

**Class signature:**
```python
class CausalMask(nn.Module):
    def __init__(self, config: LorentzianConfig):
        ...

    def forward(
        self,
        z: torch.Tensor,       # [B, d] query position
        t: torch.Tensor,       # [B, 1] query time
        z_mem: torch.Tensor,   # [B, N, d] memory positions
        t_mem: torch.Tensor,   # [B, N, 1] memory times
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Query spacetime position $(z, t)$ and memory positions $(z_{\text{mem}}, t_{\text{mem}})$
- Output: `mask` shape `[B, N]` – Binary mask (1 = causal, 0 = acausal)

**Purpose:** Computes the causal mask from the light cone structure, enforcing that attention is zero outside the causal past $J^-(z, t)$.

**Mathematical operation:**
$$M_{\text{causal}}(z, t; z', t') = \mathbf{1}\left[ t' < t \text{ and } d_G(z, z') \leq c_{\text{info}}(t - t') \right]$$

**Key insight:** This is spacetime causality, not just temporal ordering. Events must be both in the past *and* within the light cone defined by the information speed $c_{\text{info}}$.

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-causal-past-light-cone`, line 978.
:::

:::{prf:definition} G.3.4 (TemporalChristoffelQuery)
:label: def-g-temporal-christoffel-query

**Class signature:**
```python
class TemporalChristoffelQuery(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_latent: int):
        ...

    def forward(
        self,
        x: torch.Tensor,        # [B, d_in]
        z: torch.Tensor,        # [B, d_latent]
        t: torch.Tensor,        # [B, 1]
        v_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Features `x`, position `z`, time `t`, optional velocity features
- Output: `Q` shape `[B, d_out]` – Geodesic Query vector

**Purpose:** Extends the geodesic Query projection to include temporal Christoffel terms for the Lorentzian metric.

**Mathematical operation:**
$$Q_{\text{geo}}(x, z, t, v) = W_Q x + W_{Qz} z + W_{Qt} t + W_{Qv} v + W_{Q,\Gamma}(z, z) + W_{Q,t}(t, t) + W_{Q,zt}(z, t)$$

**Christoffel structure:** For the Lorentzian metric $g_{\mu\nu} = \text{diag}(-c^2\lambda^2, \lambda^2 I_d)$:
- Spatial: $\Gamma^k_{ij} = \frac{2}{1-|z|^2}(\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k)$
- Time-time-space: $\Gamma^0_{0j} = \frac{2z_j}{1-|z|^2}$
- Space-time-time: $\Gamma^k_{00} = \frac{2c^2 z_k}{1-|z|^2}$

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, Definition {prf:ref}`def-temporal-christoffel-encoding`, line 1021.
:::

:::{prf:definition} G.3.5 (LorentzianMemoryAttention)
:label: def-g-lorentzian-memory-attention

**Class signature:**
```python
class LorentzianMemoryAttention(nn.Module):
    def __init__(self, config: LorentzianConfig):
        ...

    def forward(
        self,
        x: torch.Tensor,         # [B, d_model] current state features
        z: torch.Tensor,         # [B, d_latent] current position
        t: torch.Tensor,         # [B, 1] current time
        x_mem: torch.Tensor,     # [B, N, d_model] memory features
        z_mem: torch.Tensor,     # [B, N, d_latent] memory positions
        t_mem: torch.Tensor,     # [B, N, 1] memory times
        v_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Current state $(x, z, t)$ and memory bank $(x_{\text{mem}}, z_{\text{mem}}, t_{\text{mem}})$
- Output: `(output, weights)` where:
  - `output` shape `[B, d_model]` – Attended memory representation
  - `weights` shape `[B, N]` – Attention weights (for diagnostics)

**Purpose:** Full Lorentzian memory attention combining covariant self-attention with causal mask. Implements Definition {prf:ref}`def-covariant-self-attention-causal` and Definition {prf:ref}`def-lorentzian-cross-attention`.

**Components:**
- `metric` – LorentzianMetric for conformal factor and geodesic distance
- `causal_mask` – CausalMask for light cone enforcement
- `query` – TemporalChristoffelQuery for geodesic Query
- `wilson_scale` – Learnable Wilson line approximation scale

**Key properties:**
1. **Causality:** Attention weight is zero outside $J^-(z, t)$
2. **Gauge covariance:** Wilson line preprocessing ensures gauge invariance
3. **Metric-encoded temperature:** $\tau(z) = \sqrt{d_k}/\lambda(z)$

**Diagnostic nodes:** Monitor with Nodes 71-73 (CausalMaskCheck, RetardedPotentialCheck, LorentzianSignatureCheck).

**Source:** {ref}`Section 33 <sec-covariant-memory-attention-architecture>`, line 1095.
:::

:::{prf:definition} G.4.1 (GeodesicConfig)
:label: def-g-geodesic-config

**Class signature:**
```python
@dataclass
class GeodesicConfig:
    d_model: int = 256         # Model dimension [nat]
    d_latent: int = 64         # Latent space dimension
    n_heads: int = 1           # Number of attention heads per BAOAB step
    T_c: float = 0.1           # Cognitive temperature [nat/step]
    gamma_friction: float = 1.0  # Friction coefficient for O-step
    dt: float = 0.01           # Integration timestep
    g_s: float = 1.0           # Binding coupling strength
    g_2: float = 0.5           # Error field coupling
    g_1: float = 0.3           # Opportunity field coupling
    use_learned_thermostat: bool = False  # Enable learned thermostat residual
    thermostat_residual_scale: float = 0.1  # Scale for learned residual
```

**Purpose:** Configuration for the gauge-covariant geodesic cross-attention world model.

**Key parameters:**
- `g_s` – $SU(N_f)_C$ binding coupling (confinement)
- `g_2` – $SU(2)_L$ error field coupling (chirality)
- `g_1` – $U(1)_Y$ opportunity field coupling (hypercharge)
- `use_learned_thermostat` – If True, adds a learnable thermostat head; otherwise uses closed-form OU

**Units:** Couplings $g_s, g_2, g_1$ are dimensionless.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, line 1146.
:::

:::{prf:definition} G.4.2 (WilsonLineApprox)
:label: def-g-wilson-line-approx

**Class signature:**
```python
class WilsonLineApprox(nn.Module):
    def __init__(self, config: GeodesicConfig, d_k: int):
        ...

    def forward(
        self,
        z_query: torch.Tensor,  # [B, d_latent]
        z_key: torch.Tensor,    # [B, N, d_latent]
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Query position `z_query` and key positions `z_key`
- Output: `U` shape `[B, N, d_k, d_k]` – Transformation matrices for each key

**Purpose:** Computes the linearized Wilson line $U(z, z') \approx I - i A_\mu(z)(z - z')^\mu$ for parallel transport in attention.

**Learnable parameters:**
- `theta_binding` – $SU(N_f)_C$ connection coefficients
- `theta_error` – $SU(2)_L$ connection coefficients
- `theta_opportunity` – $U(1)_Y$ connection coefficient

**Mathematical operation:**
$$U(z, z') \approx I - i\Theta(z) \cdot (z - z')$$

where $\Theta$ encodes the total gauge connection $A_\mu = g_s G_\mu + g_2 W_\mu + g_1 B_\mu$.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Proposition {prf:ref}`prop-wilson-line-approximation`, line 1176.
:::

:::{prf:definition} G.4.3 (ConformalMetric)
:label: def-g-conformal-metric

**Class signature:**
```python
class ConformalMetric(nn.Module):
    def __init__(self, epsilon: float = 1e-6):
        ...

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        ...

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        ...
```

**Input/Output:**
- `conformal_factor`: `z` shape `[B, d]` → `[B, 1]`
- `metric`: `z` shape `[B, d]` → `[B, d, d]`
- `metric_inv`: `z` shape `[B, d]` → `[B, d, d]`
- `temperature`: `z` shape `[B, d]` → `[B, 1]`

**Purpose:** Computes the Poincaré disk metric and its derived quantities.

**Key formulas:**
- Conformal factor: $\lambda(z) = 2/(1-|z|^2)$
- Metric: $G_{ij}(z) = \lambda(z)^2 \delta_{ij}$
- Inverse metric: $G^{ij}(z) = \lambda(z)^{-2} \delta^{ij}$
- Temperature: $\tau(z) = \sqrt{d_k}/\lambda(z)$

**Boundary behavior:** As $|z| \to 1$, $\lambda \to \infty$ and $\tau \to 0$, making attention infinitely sharp and preventing boundary crossing.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-poincare-metric-recap`, line 1248.
:::

:::{prf:definition} G.4.4 (ChristoffelQuery)
:label: def-g-christoffel-query

**Class signature:**
```python
class ChristoffelQuery(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_latent: int):
        ...

    def forward(
        self,
        x: torch.Tensor,         # [B, d_in] feature vector
        z_geom: torch.Tensor,    # [B, d_latent] position
        v_feat: Optional[torch.Tensor] = None,  # velocity features
        v_geom: Optional[torch.Tensor] = None,  # velocity
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Features `x`, position `z_geom`, optional velocity features
- Output: `Q` shape `[B, d_out]` – Geodesic Query vector

**Purpose:** Implements the geodesic Query projection encoding Christoffel symbols via linear + quadratic terms.

**Mathematical operation:**
$$Q_{\text{geo}}(x, z, v) = W_Q x + W_{Qz} z + W_{Qv} v_{\text{feat}} + W_{Q,\Gamma}(z, z) + W_{Qzv}(z, v)$$

**Learnable parameters:**
- `W_Q` – Feature projection
- `W_Qz` – Position projection (captures linear part of $\Gamma$)
- `W_Qv` – Velocity feature projection
- `W_Q_gamma` – Quadratic tensor for Christoffel encoding
- `W_Qzv` – Position-velocity coupling

**Initialization:** `W_Q_gamma` is initialized with Poincaré-inspired structure to approximate $\Gamma^k_{ij} \propto (\delta^k_i z_j + \delta^k_j z_i - \delta_{ij} z^k)$.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-geodesic-query-projection`, line 1311.
:::

:::{prf:definition} G.4.5 (ChiralProjector)
:label: def-g-chiral-projector

**Class signature:**
```python
class ChiralProjector(nn.Module):
    def __init__(self, d_latent: int):
        ...

    def forward(
        self,
        psi_doublet: torch.Tensor,  # [B, 2, d] observation-action doublet
        grad_V: torch.Tensor,        # [B, d_latent] value gradient
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Doublet `psi_doublet` shape `[B, 2, d]` and value gradient `grad_V`
- Output: Gated projected doublet shape `[B, 2*d]`

**Purpose:** Implements the $SU(2)_L$ chiral projector that extracts committed actions from the observation-action doublet using the value gradient direction.

**Mathematical operation:**
$$\hat{n}(z) = \frac{P \nabla_A V}{\|P \nabla_A V\|}, \quad \Pi_{\text{chirality}} = \frac{1}{2}(I_2 + \hat{n} \cdot \vec{\tau})$$

where $\vec{\tau} = (\tau_1, \tau_2, \tau_3)$ are Pauli matrices.

**Key insight:** The projection extracts the component of the doublet aligned with the value gradient—the direction of improvement. When $\nabla_A V \approx 0$ (flat landscape), the projector is degenerate, encoding decision ambiguity.

**Gauge covariance:** The commitment strength $c(z) = \Psi_L^\dagger \Pi \Psi_L$ is $SU(2)$-invariant (Theorem {prf:ref}`thm-gauge-covariance-chiral-projection`).

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-chiral-projector-value-gradient`, line 1384.
:::

:::{prf:definition} G.4.6 (AreaLawScreening)
:label: def-g-area-law-screening

**Class signature:**
```python
class AreaLawScreening(nn.Module):
    def __init__(self, config: GeodesicConfig):
        ...

    def string_area(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def forward(
        self,
        attention: torch.Tensor,  # [B, N] attention scores
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
        level: int = 0,
    ) -> torch.Tensor:
        ...
```

**Input/Output:**
- Input: Attention weights, positions, conformal factor, hierarchy level
- Output: Screened attention shape `[B, N]`

**Purpose:** Implements $SU(N_f)_C$ area law screening for texture confinement. Suppresses attention between positions at different representation levels.

**Mathematical operation:**
$$\alpha_{\text{screened}} = \alpha \cdot \exp(-\sigma(\ell) \cdot A_{\text{string}})$$

where:
- $\sigma(\ell) = \sigma_0 \cdot e^{-\ell/L}$ is the level-dependent string tension
- $A_{\text{string}} \approx \frac{\lambda^2}{2}|z - z'|^2$ is the minimal string area

**Asymptotic freedom:** At texture level ($\ell = L$), $\sigma \to 0$ and features interact freely. At macro level ($\ell = 0$), $\sigma$ is large and texture is confined.

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-area-law-screening-attention`, Theorem {prf:ref}`thm-texture-confinement-area-law`, line 1438.
:::

:::{prf:definition} G.4.7 (CovariantAttention)
:label: def-g-covariant-attention

**Class signature:**
```python
class CovariantAttention(nn.Module):
    def __init__(
        self,
        config: GeodesicConfig,
        use_chirality: bool = False,
        use_screening: bool = False,
        head_type: str = 'generic',  # 'B', 'A', 'O', or 'generic'
    ):
        ...

    def forward(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        v_query: Optional[torch.Tensor] = None,
        v_query_geom: Optional[torch.Tensor] = None,
        grad_V: Optional[torch.Tensor] = None,
        level: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Query/key positions and features, optional velocity and value gradient
- Output: `(output, attention)` where:
  - `output` shape `[B, d_model]` – Attention output
  - `attention` shape `[B, N]` – Attention weights

**Purpose:** Single covariant attention head combining all gauge structures: Wilson lines, position-dependent temperature, Christoffel Query, chiral projection, and area law screening.

**Components:**
- `query` – ChristoffelQuery
- `wilson` – WilsonLineApprox
- `metric` – ConformalMetric
- `chiral` – ChiralProjector (optional)
- `screening` – AreaLawScreening (optional)

**Attention computation:**
1. Compute $Q$ with geodesic Query projection
2. Compute $K$ and apply Wilson line: $K_{\text{transported}} = U \cdot K$
3. Score: $s = Q^T K_{\text{transported}} / \tau(z)$
4. Softmax and optional screening
5. Weighted sum of $V$, optional chiral projection

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, line 1505.
:::

:::{prf:definition} G.4.8 (GeodesicCrossAttention)
:label: def-g-geodesic-cross-attention

**Class signature:**
```python
class GeodesicCrossAttention(nn.Module):
    def __init__(self, config: GeodesicConfig):
        ...

    def forward(
        self,
        z: torch.Tensor,             # [B, d_latent] current position
        p: torch.Tensor,             # [B, d_latent] current momentum
        context_z: torch.Tensor,     # [B, N, d_latent] context positions
        context_x: torch.Tensor,     # [B, N, d_model] context features
        context_force: torch.Tensor, # [B, N, d_latent] force/gradient bank
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ...
```

**Input/Output:**
- Input: Current phase space state $(z, p)$ and context banks
- Output: `(z_next, p_next)` – Updated position and momentum

**Purpose:** Full geodesic world model implementing Boris-BAOAB integration via four attention heads (B-A-A-B) plus a closed-form OU thermostat (or optional learned thermostat head).

**BAOAB Steps:**
1. **Head 1 (B-step):** First half-kick from force bank
2. **Head 2 (A-step):** First half-drift + attention correction
3. **OU step:** Ornstein-Uhlenbeck thermostat (closed-form, or learned residual)
4. **Head 4 (A-step):** Second half-drift + attention correction
5. **Head 5 (B-step):** Second half-kick from force bank

**OU coefficients:**
$$c_1 = e^{-\gamma h}, \quad c_2 = \sqrt{(1-c_1^2)T_c}$$

**Boltzmann preservation:** Preserves $\rho(z, p) \propto \exp(-\Phi_{\text{eff}}/T_c - \|p\|_G^2/(2T_c))$ to $O(h^2)$ (Theorem {prf:ref}`thm-baoab-attention-boltzmann`).

**Diagnostic nodes:** Monitor with Nodes 67-70 (gauge, temperature, chirality, confinement).

**Source:** {ref}`Section 35 <sec-covariant-cross-attention-architecture>`, Definition {prf:ref}`def-baoab-attention-heads`, line 1616.
:::

:::{prf:definition} G.5.1 (SpectralLinear)
:label: def-g-spectral-linear

**Class signature:**
```python
class SpectralLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, in_features]` – Feature vectors
- Output: `y` shape `[B, out_features]` – Transformed features

**Purpose:** Linear layer with spectral normalization $\sigma_{\max}(W) \leq 1$. Ensures capacity bound and light cone preservation for causal structure.

**Key parameters:**
- `in_features` – Input dimension [nat]
- `out_features` – Output dimension [nat]
- `bias` – Typically `False` for gauge invariance (breaks tangent bundle structure)

**Mathematical operation:**
$$y = W_{\text{normalized}} \cdot x \quad \text{where} \quad \sigma_{\max}(W_{\text{normalized}}) \leq 1$$

**Key properties:**
- Contraction: $\|y\| \leq \|x\|$ (no unbounded amplification)
- Light cone preservation: $d(Wz_1, Wz_2) \leq c_{\text{info}} \Delta t$ whenever inputs are causally connected
- No bias term (gauge invariance requirement)

**Diagnostic node:** Node 62 (CausalityViolationCheck) verifies $\sigma_{\max}(W) \leq 1 + \epsilon$ during training.

**Source:** {ref}`Section 04 <sec-geometric-micro-architecture>`, Definition {prf:ref}`def-spectral-linear`, line 569.
:::

:::{prf:definition} G.5.2 (NormGatedActivation)
:label: def-g-norm-gated-activation

**Function signature:**
```python
def norm_gated_activation(v: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Args:
        v: Bundle vectors [B, n_bundles, bundle_dim]
        b: Bias scalars [n_bundles]
    Returns:
        Gated vectors [B, n_bundles, bundle_dim]
    """
    norms = torch.norm(v, dim=-1, keepdim=True)  # [B, n_bundles, 1]
    gates = F.gelu(norms.squeeze(-1) + b)        # [B, n_bundles]
    return v * gates.unsqueeze(-1) / (norms + 1e-8)
```

**Input/Output:**
- Input: `v` shape `[B, n_bundles, d_b]` – Bundle vectors
- Output: Gated vectors shape `[B, n_bundles, d_b]` – Energy-filtered output

**Purpose:** $SO(d_b)$-equivariant activation using radial symmetry. Gates signal based on energy $\|v\|$ exceeding threshold $-b$.

**Mathematical operation:**
$$f(v_i) = v_i \cdot g(\|v_i\| + b_i)$$

where:
- $\|v_i\| = \sqrt{v_i^T v_i}$ is the Euclidean norm (rotation-invariant)
- $g: \mathbb{R} \to \mathbb{R}$ is GELU or another smooth scalar function
- $b_i$ is the learnable activation potential (energy barrier)

**Key properties:**
- **$SO(d_b)$ equivariance:** $f(Rv) = R f(v)$ for all $R \in SO(d_b)$
- **Physical interpretation:** Energy barrier—gate opens when $\|v\| > -b$
- **Direction independence:** Gate decision depends only on magnitude, not orientation

**GELU rationale:**
- $C^\infty$ smoothness (compatible with WFR metric)
- Linear growth at large arguments: $g(x) \approx x$ for $x \gg 1$
- Controlled Lipschitz constant $L_g \approx 1.129$
- Empirically effective (validated in transformers)

**Alternative activations:** Softplus ($C^\infty$, always positive), Sigmoid/Tanh (saturate, reduced dynamic range).

**Source:** {ref}`Section 04 <sec-geometric-micro-architecture>`, Definition {prf:ref}`def-norm-gated-activation`, line 714.
:::

:::{prf:definition} G.5.3 (IsotropicBlock)
:label: def-g-isotropic-block

**Class signature:**
```python
class IsotropicBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bundle_size: int = 16,
        exact: bool = False
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, in_dim]` – Input features
- Output: `z_out` shape `[B, out_dim]` – Transformed features

**Purpose:** Atomic gauge-covariant building block combining SpectralLinear, Reshape, and NormGate in sequence.

**Architecture:**
$$\text{IsotropicBlock}(z) = \text{NormGate}(\text{Reshape}(\text{SpectralLinear}(z)))$$

**Key parameters:**
- `in_dim` – Input dimension [nat]
- `out_dim` – Output dimension (must be divisible by `bundle_size`) [nat]
- `bundle_size` – Dimension of each bundle $d_b$ [nat]
- `exact` – If `True`, uses scalar blocks $W_i = \lambda_i I_{d_b}$ for exact equivariance; if `False` (default), uses block-diagonal for approximate equivariance

**Equivariance modes:**
- **Exact mode** (`exact=True`): Strictly $\prod_{i=1}^{n_b} SO(d_b)$ equivariant via scalar blocks
  - Weight matrix: $W = \text{diag}(\lambda_1 I, \ldots, \lambda_{n_b} I)$
  - Limited expressiveness (can only scale bundles)
  - Zero equivariance violation

- **Approximate mode** (`exact=False`): Bounded equivariance violation, greater expressiveness
  - Weight matrix: Block-diagonal with general $d_b \times d_b$ blocks
  - Each block spectrally normalized: $\sigma_{\max}(W_i) \leq 1$
  - Can learn within-bundle transformations

**Mathematical constraint (exact mode):**
By Schur's lemma, any linear map commuting with all $g \in SO(d_b)$ must be a scalar multiple of identity:
$$W_i \cdot g_i = g_i \cdot W_i \quad \forall g_i \in SO(d_b) \quad \Rightarrow \quad W_i = \lambda_i I_{d_b}$$

**Diagnostic nodes:** Node 67 (GaugeInvarianceCheck), Node 62 (CausalityViolationCheck), and the DNN-local BindingConfinementCheck (DNN-B). Global Node 40 is CapacitySaturationCheck.

**Source:** {ref}`Section 04 <sec-geometric-micro-architecture>`, Definition {prf:ref}`def-isotropic-block`, line 803.
:::

:::{prf:definition} G.5.4 (GaugeInvarianceCheck)
:label: def-g-gauge-invariance-check

**Class signature:**
```python
class GaugeInvarianceCheck(DiagnosticNode):
    def __init__(self, layer: nn.Module, group: str = "SO(d)"):
        ...

    def check(self, z: torch.Tensor) -> Dict[str, float]:
        ...
```

**Input/Output:**
- Input: `z` shape `[B, d]` – Latent state
- Output: Dictionary with `gauge_violation`, `threshold`, `passed` keys

**Purpose:** Diagnostic node (Node 67) that verifies $G$-equivariance by sampling random group transformations and measuring violation.

**Mathematical test:**
$$\delta_{\text{gauge}} = \|f(g \cdot z) - g \cdot f(z)\| < \epsilon_{\text{gauge}}$$

where $g$ is a randomly sampled group element (e.g., rotation matrix for $SO(d)$).

**Key parameters:**
- `layer` – The module to test
- `group` – Symmetry group ("SO(d)" for rotations)
- Threshold: $\epsilon_{\text{gauge}} = 10^{-4}$ (exact equivariance) or $\epsilon_{\text{gauge}} \approx 0.1$ (soft equivariance)

**Failure modes:**
- Large violation ($\delta > 0.1$): Symmetry breaking without L1 regularization
- Asymmetric violation: Equivariant under some $g$ but not others (indicates partial symmetry)

**Source:** {ref}`Section 04 <sec-geometric-micro-architecture>`, line 2908.
:::

:::{prf:definition} G.5.5 (CovariantRetina)
:label: def-g-covariant-retina

**Class signature:**
```python
class CovariantRetina(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 512,
        num_rotations: int = 8,
        kernel_size: int = 5
    ):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, C, H, W]` – RGB images
- Output: `z` shape `[B, out_dim]` – Latent features

**Purpose:** $SO(2)$-equivariant vision encoder using steerable convolutions (via E2CNN library). Ensures rotation equivariance for visual inputs.

**Architecture:**
1. **Lifting layer:** Maps trivial representation (standard image) to regular representation on $SE(2)$
2. **Steerable convolutions:** 3 layers with expanding channels (32 → 64 → 64)
3. **Group pooling:** Max over rotation group to extract rotation-invariant features
4. **Spatial pooling:** Adaptive average pooling to fixed size
5. **Linear projection:** Spectral-normalized fully connected layer to latent dimension

**Key parameters:**
- `in_channels` – Input channels (3 for RGB)
- `out_dim` – Output latent dimension [nat]
- `num_rotations` – Discretization of $SO(2)$ (typically 8 or 16)
- `kernel_size` – Convolutional kernel size [pixels]

**Equivariance guarantee:**
$$\text{Conv}(R_\theta \cdot I) = D^{(\ell)}(\theta) \cdot \text{Conv}(I)$$

where $R_\theta$ is a rotation by angle $\theta$ and $D^{(\ell)}$ is the representation matrix.

**Diagnostic node:** Node 68 (RotationEquivarianceCheck) verifies $\|f(R \cdot I) - R \cdot f(I)\| < \epsilon$ for random rotations.

**Source:** {ref}`Section 04 <sec-geometric-micro-architecture>`, line 1464.
:::

:::{prf:definition} G.6.1 (UGNConfig / BundleConfig)
:label: def-g-ugn-config

**Class signatures:**
```python
@dataclass
class BundleConfig:
    name: str              # Semantic label (e.g., "charge", "lepton")
    dim: int               # Bundle dimension d_b [dimensionless]
    semantic_role: str = ""  # Physical interpretation

@dataclass
class UGNConfig:
    input_dim: int         # Input dimension [dimensionless]
    output_dim: int        # Output dimension [dimensionless]
    bundles: List[BundleConfig]  # Bundle specifications
    n_latent_layers: int = 4     # Number of soft equivariant layers
    encoder_hidden_dim: int = 256
    decoder_hidden_dim: int = 256
    lambda_l1: float = 0.01      # L1 regularization strength
    lambda_equiv: float = 0.0    # Equivariance penalty
    use_spectral_norm: bool = True
```

**Purpose:** Configuration dataclasses for the three-stage Universal Geometric Network architecture.

**Key properties:**
- `n_bundles` – Number of gauge bundles (computed from `bundles` list)
- `total_latent_dim` – $\sum_{i=1}^{n_b} d_i$
- `bundle_dims` – List of bundle dimensions $[d_1, \ldots, d_{n_b}]$

**Typical bundle structure:**
```python
bundles = [
    BundleConfig(name="color", dim=64, semantic_role="Binding/texture confinement"),
    BundleConfig(name="isospin", dim=8, semantic_role="Error field/chirality"),
    BundleConfig(name="hypercharge", dim=4, semantic_role="Opportunity field/capacity"),
]
```

**Units:** All dimensions [nat] or [dimensionless], loss weights [dimensionless].

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1180, 1873.
:::

:::{prf:definition} G.6.2 (SoftEquivariantLayer)
:label: def-g-soft-equivariant-layer

**Class signature:**
```python
class SoftEquivariantLayer(nn.Module):
    def __init__(
        self,
        bundle_dims: List[int],
        hidden_dim: int = 64,
        use_spectral_norm: bool = True
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, sum(bundle_dims)]` – Latent state
- Output: `z_out` shape `[B, sum(bundle_dims)]` – Updated latent state

**Purpose:** Core latent dynamics layer combining equivariant and mixing pathways with L1 regularization for emergent structure discovery.

**Architecture:**
$$z_{\text{out}} = z + f_{\text{equiv}}(z) + g \cdot f_{\text{mix}}(z)$$

where:
- **Equivariant pathway:** $f_{\text{equiv}}(z) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$
  - Uses only bundle norms → strictly $\prod_i SO(d_i)$ equivariant
  - Implemented via norm MLP: $\mathbb{R}^{n_b} \to \mathbb{R}^{n_b}$

- **Mixing pathway:** $f_{\text{mix}}(z) = \sum_{i,j} W_{ij} v_j$
  - Cross-bundle interactions with learnable weights
  - L1 penalized: $\mathcal{L}_{\text{L1}} = \sum_{i,j} \|W_{ij}\|_1$
  - Encouraged to be sparse (emergent texture zeros)

**Key parameters:**
- `bundle_dims` – List $[d_1, \ldots, d_{n_b}]$ of bundle dimensions
- `hidden_dim` – Hidden dimension for norm MLP
- `use_spectral_norm` – Apply spectral normalization to all linear layers

**Learnable parameters:**
- Norm MLP weights: $O(n_b \cdot h + h^2)$ parameters
- Mixing weights $W_{ij}$: $O(n_b^2 d_{\max}^2)$ parameters (largest memory consumer)
- Gate biases: $n_b$ scalars

**L1 loss:**
```python
def l1_loss(self) -> torch.Tensor:
    return sum(
        torch.sum(torch.abs(self.mixing_weights[i][j]))
        for i in range(n_b) for j in range(n_b)
    )
```

**Diagnostic methods:**
- `mixing_strength()` – Total Frobenius norm of mixing weights (measures symmetry breaking)

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1219 (simplified), 1970 (production).
:::

:::{prf:definition} G.6.3 (UniversalGeometricNetwork)
:label: def-g-universal-geometric-network

**Class signature:**
```python
class UniversalGeometricNetwork(nn.Module):
    def __init__(self, config: UGNConfig):
        ...
```

**Input/Output:**
- Input: `x` shape `[B, input_dim]` – Raw observations
- Output: `y` shape `[B, output_dim]` – Predictions/actions

**Purpose:** Three-stage architecture achieving both universal approximation and geometric consistency.

**Architecture:**
1. **Encoder** (unconstrained, universal):
   - $E: \mathbb{R}^{d_{\text{in}}} \to \bigoplus_i V_i$
   - 2-3 spectral-normalized linear layers with GELU
   - **Chooses gauge** for latent representation

2. **Latent Dynamics** (soft equivariant):
   - $D_1, \ldots, D_L: \bigoplus_i V_i \to \bigoplus_i V_i$
   - Stack of `SoftEquivariantLayer` modules
   - **Respects bundle structure** via equivariant pathway + L1-regularized mixing

3. **Decoder** (unconstrained, universal):
   - $P: \bigoplus_i V_i \to \mathbb{R}^{d_{\text{out}}}$
   - 2-3 spectral-normalized linear layers with GELU
   - **Interprets gauge** to extract observables

**Key methods:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    z = self.encode(x)      # Encoder
    z = self.dynamics(z)    # Latent layers
    y = self.decode(z)      # Decoder
    return y

def regularization_loss(self) -> torch.Tensor:
    # L1 penalty on all mixing weights
    return sum(layer.l1_loss() for layer in self.latent_layers)

def equivariance_violation(self, z=None, n_samples=16) -> torch.Tensor:
    # Measure ||D(Rz) - RD(z)||² for random rotations
    ...
```

**Total loss:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{L1}} \mathcal{L}_{\text{L1}} + \lambda_{\text{equiv}} \mathcal{L}_{\text{equiv}}$$

**Key theorems:**
- Universal approximation (encoder/decoder handle arbitrary functions)
- Geometric consistency (latent dynamics respect bundle structure)
- Emergent gauge structure (L1 discovers texture zeros)

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 1335 (simplified), 2154 (production).
:::

:::{prf:definition} G.6.4 (FactoredTensorLayer)
:label: def-g-factored-tensor-layer

**Class signature:**
```python
class FactoredTensorLayer(nn.Module):
    def __init__(
        self,
        d_C: int,
        d_L: int,
        d_Y: int,
        rank: int,
        d_out: int
    ):
        ...
```

**Input/Output:**
- Input: `(z_C, z_L, z_Y)` shapes `[B, d_C]`, `[B, d_L]`, `[B, d_Y]`
- Output: `y` shape `[B, d_out]`

**Purpose:** Low-rank factorization of tensor product interaction for cross-gauge coupling.

**Mathematical operation:**
$$W = \sum_{k=1}^r U_C^{(k)} \otimes U_L^{(k)} \otimes U_Y^{(k)}$$

instead of full tensor $W \in \mathbb{R}^{(d_C d_L d_Y) \times d_{\text{out}}}$.

**Parameter count:**
- Factored: $r(d_C + d_L + d_Y + d_{\text{out}})$
- Full tensor: $(d_C \times d_L \times d_Y) \times d_{\text{out}}$

**Example reduction:**
For $d_C=64, d_L=8, d_Y=4, d_{\text{out}}=64, r=16$:
- Factored: 2,240 parameters
- Full: 131,072 parameters
- **58.5× reduction**

**Use case:** Specific cross-gauge interactions when low-rank structure is empirically justified. Not used in default UGN (uses direct sum instead).

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 381.
:::

:::{prf:definition} G.6.5 (NormInteractionLayer)
:label: def-g-norm-interaction-layer

**Class signature:**
```python
class NormInteractionLayer(nn.Module):
    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, n_bundles, bundle_dim]` – Bundle representation
- Output: `z_out` shape `[B, n_bundles, bundle_dim]` – Scaled bundles

**Purpose:** Level 1 cross-bundle interaction using only bundle norms (strictly equivariant).

**Mathematical operation:**
$$f_i(v_1, \ldots, v_{n_b}) = v_i \cdot \phi_i(\|v_1\|, \ldots, \|v_{n_b}\|)$$

where $\phi: \mathbb{R}^{n_b} \to \mathbb{R}_+$ is an MLP with Softplus output.

**Equivariance:** Strictly $\prod_{i=1}^{n_b} SO(d_b)_i$ equivariant (per-bundle rotations).

**Expressiveness:** Limited—can only scale bundles based on energy, cannot represent direction-dependent interactions.

**Computational cost:** $O(n_b d_b + h^2)$ where $h$ is MLP hidden dimension.

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 446.
:::

:::{prf:definition} G.6.6 (GramInteractionLayer)
:label: def-g-gram-interaction-layer

**Class signature:**
```python
class GramInteractionLayer(nn.Module):
    def __init__(self, n_bundles: int, hidden_dim: int = 64):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, n_bundles, bundle_dim]` – Bundle representation
- Output: `z_out` shape `[B, n_bundles, bundle_dim]` – Scaled bundles

**Purpose:** Level 2 cross-bundle interaction using Gram matrix $G_{ij} = \langle v_i, v_j \rangle$ (encodes relative orientations).

**Mathematical operation:**
$$G = z \cdot z^T \quad \text{(Gram matrix)}$$
$$\text{scales} = \phi(G_{\text{flat}}) \quad \text{(MLP)}$$
$$z_{\text{out}} = z \cdot \text{scales}$$

**Equivariance:** Equivariant under **global** $SO(d_b)$ (same rotation applied to all bundles), **not** under per-bundle rotations.

**Expressiveness:** High—can encode relative orientations between bundles.

**Computational cost:** $O(n_b^2 d_b + h^2)$.

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 494.
:::

:::{prf:definition} G.6.7 (L1Scheduler / AdaptiveL1Scheduler)
:label: def-g-l1-scheduler

**Class signature:**
```python
class AdaptiveL1Scheduler:
    def __init__(
        self,
        initial_lambda: float = 0.01,
        target_violation: float = 0.22,
        learning_rate: float = 0.05,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0
    ):
        ...

    def step(self, current_violation: float) -> float:
        ...
```

**Purpose:** Adaptive scheduler for L1 regularization strength $\lambda_{\text{L1}}$ that targets a specific equivariance violation level.

**Update rule:**
$$\lambda_{\text{L1}}(t+1) = \lambda_{\text{L1}}(t) \cdot \left(1 + \alpha \cdot (\epsilon(t) - \epsilon_{\text{target}})\right)$$

where:
- $\epsilon(t) = \mathcal{L}_{\text{equiv}}(t)$ is current equivariance violation
- $\epsilon_{\text{target}} \approx 0.22$ nat/step (proposed target)
- $\alpha$ is adaptation rate (typically 0.01-0.1)

**Strategy:**
- If $\epsilon(t) > \epsilon_{\text{target}}$: Increase $\lambda_{\text{L1}}$ (more sparsity, less mixing)
- If $\epsilon(t) < \epsilon_{\text{target}}$: Decrease $\lambda_{\text{L1}}$ (more expressiveness, more mixing)

**Key parameters:**
- `initial_lambda` – Starting $\lambda_{\text{L1}}$ value
- `target_violation` – Desired equivariance violation $\epsilon_{\text{target}}$ [nat/step]
- `learning_rate` – Adaptation rate $\alpha$
- `min_lambda` / `max_lambda` – Clamping bounds to prevent collapse or over-sparsity

**Training protocol:**
1. **Warmup (epochs 1-10):** Low $\lambda_{\text{L1}} = 0.001$, let network explore
2. **Ramp up (epochs 10-50):** Gradually increase $\lambda_{\text{L1}}$
3. **Adaptive (epochs 50+):** Use `AdaptiveL1Scheduler` to maintain target violation
4. **Fine-tune:** Fix $\lambda_{\text{L1}}$, early stopping on validation

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, lines 955, 2577.
:::

:::{prf:definition} G.6.8 (CovariantAttentionLayer)
:label: def-g-covariant-attention-layer

**Class signature:**
```python
class CovariantAttentionLayer(nn.Module):
    def __init__(
        self,
        bundle_dims: List[int],
        n_heads: int = 4,
        use_wilson_lines: bool = True
    ):
        ...
```

**Input/Output:**
- Input: `z` shape `[B, sum(bundle_dims)]`, optional `context` shape `[B, T, sum(bundle_dims)]`
- Output: `z_out` shape `[B, sum(bundle_dims)]`

**Purpose:** Covariant cross-attention for explicit world modeling and trajectory prediction. Alternative to `SoftEquivariantLayer` when planning is required.

**Architecture:**
- Multi-head attention per bundle
- Wilson lines for gauge-covariant Q/K/V projections
- Position-dependent temperature $\tau(z) = \sqrt{d_k}/\lambda(z)$
- Geometric Query terms with Christoffel symbols

**Use cases:**
- **SoftEquivariantLayer:** Default latent dynamics, implicit world model
- **CovariantAttentionLayer:** Explicit trajectory prediction, planning, memory retrieval

**Multi-stage pipeline example:**
1. Encoder → latent $Z$
2. SoftEquivariantLayer (×2) for geometric regularization
3. CovariantAttentionLayer for trajectory rollout
4. SoftEquivariantLayer (×2) for policy extraction
5. Decoder → action $Y$

**Source:** {ref}`Section 06 <sec-universal-geometric-network>`, line 2445. See also {ref}`Section 05 <sec-covariant-cross-attention-architecture>` for full derivation.
:::
