(sec-symplectic-multi-agent-field-theory)=
# Relativistic Symplectic Multi-Agent Field Theory

## TLDR

- Timestamped histories make delayed interactions explicit and support the history-state transition construction.
- The Bellman generator equation and the wave-action equation have distinct derivations; their stationary screened operators can be compared directly.
- Local frame changes give the gauge connection, covariant derivative, curvature, and variational field equations.
- The exact WFR amplitude equation contains a density-dependent Bohm compensation. Linear Hamiltonian methods are analyzed for their specified operators.
- Nash conditions, spectral gaps, and information bounds retain their own definitions and are related through explicit calculations.

## Roadmap

1. Construct delayed interfaces and their history state.
2. Derive interaction geometry and gauge field equations with consistent conventions.
3. Compare equilibrium tests, amplitude dynamics, and spectral calculations, then implement causal observation rules.

*Abstract.* We formulate multi-agent interaction on a product state space with timestamped causal histories. Retarded interfaces determine the observations available to each agent. We derive the controlled generator equation from Bellman recursion and the wave equation from its field action, and compare their stationary operators. Internal frame covariance yields the connection and curvature identities; variation of the action yields the matter currents and stress tensor. An explicit Madelung calculation reconstructs the WFR amplitude equation with its density-dependent compensation. Unilateral payoff tests, information estimates, and spectral results are kept attached to the mathematical objects for which they are proved.

(rb-relativistic-marl)=
:::{admonition} Researcher Bridge: Delayed observations and field dynamics
:class: info
A distributed agent acts on received messages and local observations. Timestamped histories encode which information is available. Field actions specify how modeled signals propagate; controlled generators specify how value evolves. Their relationship is established by comparing the equations and their coefficients.
:::

*Cross-references:* The screened reward operator is developed in
{ref}`sec-the-bulk-potential-screened-poisson-equation`; the occupation memory screen in
{ref}`sec-the-historical-manifold-and-memory-screen`. This chapter uses those constructions while distinguishing an ordered causal history from its occupation-measure compression.

*Literature:* Game theory {cite}`fudenberg1991game`; stochastic games {cite}`shapley1953stochastic`; multi-agent RL
{cite}`littman1994markov,lowe2017multi`; symplectic geometry {cite}`arnold1989mathematical`; retarded potentials
{cite}`jackson1999classical`.

:::{div} feynman-prose
A message tells an agent what another agent reported when the message was sent. The elapsed travel time matters whenever the sender can change before the receiver responds. This is the concrete purpose of the causal interface: make the available information explicit in the state and in the transition rule.

The electromagnetic analogy helps us picture delayed signals. The mathematical work comes from specifying which histories are retained, which observations are available, and which evolution equation transports the field.
:::

::::{admonition} Connection to RL #17: Independent PPO as Disconnected Sheaf
:class: note
:name: conn-rl-17
**The General Law (Fragile Agent):**
Multi-agent interaction is modeled via **Ghost Interfaces** (Definition {prf:ref}`def-ghost-interface`) connecting retarded boundary states:

$$
\mathcal{G}_{ij}(t) \subset \partial\mathcal{Z}^{(i)}(t) \times \partial\mathcal{Z}^{(j)}(t - \tau_{ij}), \quad \omega_{\mathcal{G},ij} := \omega^{(i)}(t) \oplus \omega^{(j)}(t - \tau_{ij})\big|_{\mathcal{G}_{ij}}.

$$
The **Game Tensor** $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) encodes strategic coupling with retarded components: how Agent $i$'s latent inertia changes due to Agent $j$'s past state.

**The Degenerate Limit:**
Set all interfaces $\mathcal{G}_{ij} = \emptyset$ (disconnect the sheaf). Each agent treats others as stationary noise.

**The Special Case (Standard RL - IPPO):**
Independent PPO {cite}`de2020independent` runs separate learners with shared scalar reward (conservative case):

$$
\pi^{(i)} = \arg\max_{\pi} \mathbb{E}\left[ \sum_t r^{(i)}_t(\mathbf{s}, \mathbf{a}) \right], \quad \text{treating } \pi^{(-i)} \text{ as fixed}.

$$
Each agent optimizes against a stationary environment—other agents are part of the "MDP noise."

**Result:** IPPO is the $\mathcal{G}_{ij} \to \emptyset$ limit where agents are **solipsistic**—they share a world but have no causal coupling.

**What the generalization offers:**
- **Causal structure**: Finite $c_{\text{info}}$ determines which events can influence which ({ref}`sec-the-failure-of-simultaneity`)
- **Ghost Interface**: Agents couple to retarded images, not instantaneous states ({ref}`sec-the-ghost-interface`)
- **Generator and wave equations**: Compare their derivations and stationary operators ({ref}`sec-the-hyperbolic-value-equation`)
- **Equilibrium tests**: Compare unilateral payoff optimality with field stationarity ({ref}`sec-relativistic-nash-equilibrium`)
- **Diagnostic nodes 46-48, 62**: Runtime monitoring including causality violation checks ({ref}`sec-yang-mills-action`)
::::

(sec-the-product-configuration-space)=
## The Product Configuration Space

:::{div} feynman-prose
The joint configuration records one state for each agent, so it lives in the product of their state spaces. In the uncoupled product metric, the squared length of a joint displacement is the sum of the individual squared lengths.

A block-diagonal metric has no cross terms between displacement blocks. Its blocks can nevertheless depend on other agents' coordinates. That dependence matters later: the joint volume element and differential operator must be computed from the full metric.
:::

Consider $N$ agents, each with an internal latent manifold $(\mathcal{Z}^{(i)}, G^{(i)})$ and a boundary interface
$B^{(i)} = (x^{(i)}, a^{(i)}, r^{(i)})$, where $r^{(i)}$ is a boundary reward sample (evaluation of the reward 1-form/flux;
scalar in the conservative case). The agents may be spatially distributed, with finite information propagation time
between them.

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
- **Dirichlet component** (sensors): $\phi^{(i)}(x)$ is the observation stream
- **Neumann component** (motors): $j^{(i)}_{\text{motor}}(x)$ is the action flux
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
- **Physical agents:** $d_{\mathcal{E}}^{ij}$ is the spatial separation in meters
- **Networked agents:** $d_{\mathcal{E}}^{ij}$ is the network hop distance or latency
- **Co-located agents:** $d_{\mathcal{E}}^{ij} = 0$ (shared boundary)

*Units:* $[d_{\mathcal{E}}^{ij}]$ is measured in meters or equivalent environment-specific units.

:::
(sec-the-failure-of-simultaneity)=
## The Failure of Simultaneity

:::{div} feynman-prose
A stationary screened equation describes a field after transients have settled. It does not say how a change reaches a distant observer. To describe that process, we must examine the actual evolution equation.

The Bellman expansion produces a first-order time equation involving the controlled generator. A wave action produces a second-order time equation. Both can have screened stationary solutions, but their transients differ. Finite signal speed specifies the allowed causal domain; it does not turn the Bellman generator into a wave operator.
:::

The standard HJB equation assumes the value $V(z)$ relaxes instantly across the manifold. This implies an infinite speed
of information propagation, violating the causal constraints of distributed systems. Here $V$ denotes the scalar
potential associated with the **conservative component** of the reward 1-form; non-conservative (curl) components
propagate as antisymmetric fields and appear as velocity-dependent forces rather than a scalar PDE.

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

:::{div} feynman-prose
The past cone of an event contains the emission events whose signals can reach it by the observation time. A receiver can use those messages. It cannot use a message that has not yet arrived.

Spacelike separation forbids a direct signal between the two events under this propagation rule. It does not forbid statistical correlation through a common past. Keeping these statements separate is essential when interpreting correlated agent histories.
:::

(pi-minkowski)=
::::{admonition} Physics Isomorphism: Minkowski Spacetime
:class: note

**In Physics:** Special relativity defines the causal structure via the Minkowski metric $ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2$. Events with $ds^2 < 0$ are timelike separated (causally connected); events with $ds^2 > 0$ are spacelike separated (causally disconnected) {cite}`jackson1999classical`.

**In Implementation:** The causal interval (Definition {prf:ref}`def-causal-interval`) induces a Lorentzian structure on the agent-time space:

$$
\Delta s^2_{ij} = -c_{\text{info}}^2 \Delta t^2 + d_{\mathcal{E}}^2.

$$

**Correspondence Table:**
| Special Relativity | Multi-Agent System |
|:-------------------|:-------------------|
| Speed of light $c$ | Information speed $c_{\text{info}}$ |
| Spatial distance | Environment distance $d_{\mathcal{E}}^{ij}$ |
| Past light cone | Causally accessible agent states |
| Spacelike separation | Instantaneously decoupled agents |
| Lorentz invariance | Causal consistency under frame changes |
::::

(sec-the-relativistic-state-restoring-markovianity)=
## The Relativistic State: Restoring Markovianity

:::{div} feynman-prose
Delay makes the choice of state important. Two systems with the same current positions can evolve differently when their pending messages differ. Retaining the ordered history makes those differences visible to the transition rule.

The reward-weighted memory screen is a compressed occupation measure: it records accumulated contributions at locations. It can discard when those contributions occurred and in what order. An ordered history and this occupation measure therefore play different roles. The Markov construction below uses the explicit history update; a compressed representation inherits it only through a proved sufficient-statistic identity.
:::

To recover a valid control problem under finite information speed, we must augment the state to include the field configuration within the past light cone.

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

*Proof.* The sigma-algebra generated by $(t,\mathsf H_t)$ contains the entire
modeled history through $t$. Let $K_{t,u}(h,\cdot)$ be the regular conditional
law of the extended history through $u$ given $\mathsf H_t=h$.
For a bounded history functional $F$,

$$
\mathbb E[F(\mathsf H_u)\mid\sigma(\mathsf H_s:s\le t)]
=\mathbb E[F(\mathsf H_u)\mid\mathsf H_t]
=K_{t,u}F(\mathsf H_t).
$$
The tower property gives $K_{t,u}=K_{t,v}K_{v,u}$ on realized histories.
Including $t$ in the state accounts for time-inhomogeneous coefficients.
This proves the representation without identifying a finite compression.
For the occupation screen take $\alpha=0$: every history has screen zero,
while histories with the same current position can have different
$z_{t-\tau}$ and therefore different delayed drifts. Conversely a delayed
signal with zero coupling leaves a Markov local process Markov. These examples
prove both limitations of the smaller-state claims. $\square$
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
(sec-the-ghost-interface)=
## The Ghost Interface: Asynchronous Coupling

:::{div} feynman-prose
The ghost interface records the information an agent can actually receive from another agent. A timestamp identifies when a message was emitted, and the propagation rule identifies when it becomes available.

This gives a practical interpretation to retardation. Every coupling must be evaluated from the available record. A predicted current state may be useful, but it is a prediction computed from that record, not a newly observed event.
:::

We replace the instantaneous coupling of boundary conditions with an asynchronous **Ghost Interface** that respects causal structure.

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
(pi-lienard-wiechert)=
::::{admonition} Physics Isomorphism: Liénard-Wiechert Potentials
:class: note

**In Physics:** The electromagnetic potentials of a moving charge are evaluated at the retarded time $t_{\text{ret}} = t - r/c$, not the current time. The Liénard-Wiechert potentials encode causality in classical electrodynamics {cite}`jackson1999classical`.

**In Implementation:** In the quasi-static approximation, the Ghost Interface evaluates strategic potentials at the
retarded time:

$$
\Phi^{\text{ret}}_{ij}(z^{(i)}, t) \approx \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau_{ij}}).

$$

**Correspondence Table:**
| Electrodynamics | Relativistic Agent |
|:----------------|:-------------------|
| Field equation $\square A^\mu = J^\mu$ | Value equation (conservative component) $\square_G V = \rho_r$ |
| Light speed $c$ | Information speed $c_{\text{info}}$ |
| Retarded time $t_{\text{ret}}$ | Ghost time $t - \tau_{ij}$ |
| Liénard-Wiechert potential | Retarded interaction potential |
| Radiation reaction | Strategic back-pressure |
::::



(sec-the-hyperbolic-value-equation)=
## The Hyperbolic Value Equation (Klein-Gordon)

:::{div} feynman-prose
The screened wave model has a second time derivative, so its initial state includes both a field and its initial rate of change. This is the additional dynamical structure supplied by its action. The Bellman generator instead evolves its value function through a first-order time equation.

Their static equations can be compared coefficient by coefficient. Screening controls the stationary spatial response; a positive mass term alone does not damp an oscillatory wave in time. The calculations below keep the static screening coefficient, propagation speed, and temporal evolution distinct.
:::

The controlled generator equation and the wave-action equation can share a stationary screened operator. The following calculation identifies their time derivatives and coefficients explicitly.

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

*Proof.* Generator consistency gives
$\mathbb E[V(Z_h,t+h)]=V+h(\partial_t+\mathcal L)V+o(h)$.
Insert this and $e^{-\lambda h}=1-\lambda h+o(h)$ into
$V=rh+e^{-\lambda h}\mathbb E[V(Z_h,t+h)]$, cancel $V$, and divide by $h$.
The $\partial_t^2V$ Taylor term has coefficient $h/2$ after division and
vanishes. Finite signal speed does not change that coefficient.
For the wave model vary

$$
S[V]=\int\left[-\tfrac12g^{\mu\nu}\partial_\mu V\partial_\nu V
-\tfrac12\kappa^2V^2+\rho_rV\right]\sqrt{|g|}\,dx.
$$
Integration by parts against a compactly supported variation $\eta$ gives
$\delta S=\int\eta[-\Box_gV-\kappa^2V+\rho_r]\sqrt{|g|}\,dx$.
Stationarity proves the field equation. For a fixed product metric
$g=\operatorname{diag}(-c^2,G)$, $\Box_g=c^{-2}\partial_t^2-\Delta_G$.
These are explicit equations for two defined evolutions; equality of their
stationary operators is the comparison established here. $\square$
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

(pi-klein-gordon)=
::::{admonition} Physics Isomorphism: Klein-Gordon Equation
:class: note

**In Physics:** The Klein-Gordon equation $(\square + m^2)\phi = \rho$ describes a relativistic scalar field with mass $m$. It reduces to the Helmholtz equation in the static limit {cite}`jackson1999classical`. (Sign convention: we use $\square_G = \frac{1}{c^2}\partial_t^2 - \Delta_G = -\frac{1}{\sqrt{|g|}}\partial_\mu(\sqrt{|g|}g^{\mu\nu}\partial_\nu)$.)

**In Implementation:** The scalar Value potential (conservative component) satisfies:

$$
\left(\frac{1}{c_{\text{info}}^2}\partial_t^2 - \Delta_G + \kappa^2\right)V = \rho_r

$$

**Correspondence Table:**
| Klein-Gordon (Physics) | Value Equation (Agent) |
|:-----------------------|:-----------------------|
| Scalar field $\phi$ | Value function $V$ |
| Mass parameter $m$ | Screening mass $\kappa$ |
| Source $\rho$ | Conservative reward density $\rho_r$ |
| D'Alembertian $\square$ | Manifold wave operator $\square_G$ |
| Static limit | Newtonian (Helmholtz) limit |
| Propagating modes | Value wavefronts |
::::
(sec-the-game-tensor-deriving-adversarial-geometry)=
## The Game Tensor: Relativistic Adversarial Geometry

:::{div} feynman-prose
Strategic sensitivity measures how a change in one agent's state affects another agent's objective or response. Its sign depends on the objective and the direction of variation. Adversarial interaction by itself does not make every Hessian positive.

Once a positive metric perturbation is identified, its cost has a simple interpretation: the same displacement costs more in the affected directions. Retardation changes which recorded state is used to evaluate that perturbation. The tensor indices and the full metric inverse then determine the resulting geometry.
:::

In an adversarial (zero-sum) game, Agent $j$ acts to minimize the value $V^{(i)}$ that Agent $i$ maximizes. Under relativistic constraints, the Game Tensor acquires retarded components that introduce strategic hysteresis.

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
$\xi^\top(\widetilde G-G)\xi=\sum_j\beta_{ij} (\mathcal J_{ji}\xi)^\top H^{(i)}_{jj}(\mathcal J_{ji}\xi)$.
Every positive-semidefinite summand with nonnegative coefficient increases
the quadratic form; hence a sum of such contributions gives
$\widetilde G\succeq G$. A negative cooperative contribution must be included
in the same sum when testing its sign and positive-definiteness.
This follows by expanding the definition term by term. The strategic sign
label by itself does not determine the sign of the Hessian. $\square$
:::
(rb-opponents-inertia)=
:::{admonition} Researcher Bridge: Opponents as Geometric Inertia
:class: info
In game-theoretic settings, adversarial opponents increase the effective **mass** (metric tensor eigenvalues) of the agent's latent space via the pulled-back Game Tensor $\mathcal{G}^{(i)}_{ij}$. This transforms strategic uncertainty into geometric inertia: the agent moves more slowly in contested regions because geodesic steps are more costly. Cooperation has the opposite effect—allies smooth the value landscape, reducing effective mass.
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
(sec-relativistic-nash-equilibrium)=
## Relativistic Nash Equilibrium (Standing Waves)

:::{div} feynman-prose
Nash equilibrium tests unilateral deviations: hold the other agents' strategies fixed and ask whether one agent can improve its payoff. A standing wave tests a field equation: ask whether its spatial pattern evolves with a single temporal frequency. These tests use different data.

A stationary density can also support circulating current, and a periodic density can have zero time-averaged change without satisfying every agent's optimization problem. We therefore compute field stationarity and unilateral payoff variations separately. A relation between them must appear in those calculations.
:::

We compare three tests: unilateral payoff optimality, stationarity of a density, and the eigenmode equation of a specified field operator. Finite propagation speed enters the delayed dynamics used in each test.

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
(sec-diagnostic-nodes-part-i)=
## Diagnostic Nodes 46–48, 62 (Multi-Agent Causality)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define monitors for multi-agent causal systems.

(node-46)=
**Node 46: GameTensorCheck**

| **#**  | **Name**            | **Component** | **Type**           | **Interpretation**                | **Proxy**                                                                     | **Cost**     |
|--------|---------------------|---------------|--------------------|-----------------------------------|-------------------------------------------------------------------------------|--------------|
| **46** | **GameTensorCheck** | Multi-Agent   | Strategic Coupling | Is strategic sensitivity bounded? | $\lVert\mathcal{G}_{ij}\rVert_F := \sqrt{\sum_{kl}(\mathcal{G}_{ij}^{kl})^2}$ | $O(N^2 d^2)$ |

**Interpretation:** Monitors the Frobenius norm of the Game Tensor between agent pairs. Large $\|\mathcal{G}_{ij}\|_F$ indicates high strategic interdependence, potentially leading to oscillatory dynamics or failure to converge.

**Threshold:** $\|\mathcal{G}_{ij}\|_F < \mathcal{G}_{\max}$ (implementation-dependent; typical default $\mathcal{G}_{\max} = 10 \cdot \|G^{(i)}\|_F$).

**Trigger conditions:**
- High GameTensorCheck: Agents are tightly coupled; small moves trigger large counter-moves.
- Remedy: Reduce coupling strength $\alpha_{\text{adv}}$; increase exploration temperature; consider decoupled training phases.

(node-47)=
**Node 47: NashResidualCheck**

| **#**  | **Name**              | **Component** | **Type**    | **Interpretation**                | **Proxy**                                                                                                       | **Cost** |
|--------|-----------------------|---------------|-------------|-----------------------------------|-----------------------------------------------------------------------------------------------------------------|----------|
| **47** | **NashResidualCheck** | Multi-Agent   | Equilibrium | Are agents near Nash equilibrium? | $\epsilon_{\text{Nash}} := \max_i \lVert(G^{(i)})^{-1}\nabla_{z^{(i)}} \Phi_{\text{eff}}^{(i)}\rVert_{G^{(i)}}$ | $O(N d)$ |

**Interpretation:** Measures the maximum deviation from the Nash stasis condition (Theorem {prf:ref}`thm-nash-equilibrium-as-geometric-stasis`, Condition 1). At equilibrium, $\epsilon_{\text{Nash}} = 0$.

**Threshold:** $\epsilon_{\text{Nash}} < \epsilon_{\text{Nash,tol}}$ (typical default $10^{-3}$).

If $\epsilon_{\text{Nash}} > 0$ but below threshold, the system is in a **transient non-equilibrated state**. This is expected during:
1. **Learning dynamics:** Agents are still adapting policies; gradients have not yet vanished.
2. **Environmental shift:** External conditions changed, invalidating previous equilibrium.
3. **Exploration phase:** Agents are deliberately perturbing away from equilibrium to discover better basins.

**Remediation:**
- If $\epsilon_{\text{Nash}}$ is decreasing: system is converging; no intervention needed.
- If $\epsilon_{\text{Nash}}$ is oscillating: potential limit cycle; reduce learning rates or add damping ($\gamma_{\text{damp}}$ in the joint SDE).
- If $\epsilon_{\text{Nash}}$ is increasing: instability detected; may indicate poorly conditioned Game Tensor. Check Node 46 for large $\|\mathcal{G}_{ij}\|_F$.

(node-48)=
**Node 48: RelativisticSymplecticCheck**

| **#**  | **Name**                      | **Component** | **Type**     | **Interpretation**                            | **Proxy**                                                                                                            | **Cost**   |
|--------|-------------------------------|---------------|--------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------|
| **48** | **RelativisticSymplecticCheck** | Multi-Agent   | Conservation | Is retarded flux balanced across Ghost Interface? | $\Delta_{\omega}^{\text{ret}} := \int_{t_1}^{t_2} \left\lvert \Phi_{\text{out}}(t) - \Phi_{\text{in}}(t + \tau_{ij}) \right\rvert dt$ | $O(N^2 d)$ |

**Interpretation:** Monitors symplectic flux conservation on the Ghost Interface (Definition {prf:ref}`def-ghost-interface`). Under relativistic constraints, we compare outflow at time $t$ with inflow at retarded time $t + \tau_{ij}$. Immediate conservation is impossible; **retarded conservation** is the appropriate measure.

**Threshold:** $\Delta_{\omega}^{\text{ret}} < \epsilon_{\omega}$ (typical default $10^{-4}$).

**Trigger conditions:**
- Positive RelativisticSymplecticCheck: Energy is leaking through non-conservative forces or causal inconsistency.
- **Remedy:** Check for unmodeled friction; verify causal buffer implementation; reduce timestep.

*Cross-reference:* This is the relativistic generalization of symplectic volume conservation to retarded interactions.

(node-62)=
**Node 62: CausalityViolationCheck**

| **#**  | **Name**                    | **Component** | **Type**   | **Interpretation**                                     | **Proxy**                                                                                          | **Cost** |
|--------|-----------------------------|---------------|------------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------|----------|
| **62** | **CausalityViolationCheck** | Multi-Agent   | Causality  | Did information arrive faster than $c_{\text{info}}$? | $\Delta_{\text{causal}} := \max_{i,j} \mathbb{I}\left[\Delta I(z^{(i)}_t; z^{(j)}_{t'}) > 0 \land t' > t - \tau_{ij}\right]$ | $O(N^2)$ |

**Interpretation:** Detects violations of the causal structure (Definition {prf:ref}`def-causal-interval`). Agent $i$ should have no mutual information with Agent $j$'s state at times $t' > t - \tau_{ij}$ (inside the future light cone).

**Threshold:** $\Delta_{\text{causal}} = 0$ (hard constraint: no superluminal information).

**Trigger conditions:**
- Positive CausalityViolationCheck: The simulation has leaked "ground truth" information that violates the light cone. This is a **fatal error** indicating:
  1. Incorrect causal buffer implementation
  2. Unmodeled fast communication channel
  3. Timing errors in boundary condition updates

- **Remedy:** Audit causal buffer; verify all inter-agent communication respects $\tau_{ij}$ delays; check for inadvertent global state sharing.

*Cross-reference:* This enforces the information speed limit (Axiom {prf:ref}`ax-information-speed-limit`).



(sec-summary-table-from-single-to-multi-agent)=
## Summary Table: Newtonian vs. Einsteinian Agent

**Table 29.9.1 (Newtonian vs. Relativistic Multi-Agent).**

| Feature | Newtonian ($c_{\text{info}} \to \infty$) | Relativistic ($c_{\text{info}} < \infty$) |
|:--------|:-----------------------------------------|:------------------------------------------|
| **Information Speed** | $\infty$ (Instantaneous) | Finite $c_{\text{info}}$ |
| **Value evolution** | Controlled generator equation | Controlled generator with delayed observations; wave equation for the specified field action |
| **State** | State variables of the instantaneous model | Current variables and ordered causal history |
| **Markov Property** | On $\mathcal{Z}^{(N)}$ | On Causal Bundle $\mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Interaction** | Synchronous Bridge $\mathcal{B}_{ij}$ | Asynchronous Ghost Interface $\mathcal{G}_{ij}$ |
| **Potential** | Instantaneous $\Phi_{ij}(z^{(i)}, z^{(j)}_t)$ | Retarded $\Phi^{\text{ret}}_{ij}(z^{(i)}, t)$ (quasi-static: $\approx \Phi_{ij}(z^{(i)}, z^{(j)}_{t-\tau})$) |
| **Game Tensor** | $\mathcal{G}_{ij}(z^{(j)}_t)$ | $\mathcal{G}_{ij}^{\text{ret}}(z^{(j)}_{t-\tau})$ |
| **Equilibrium** | Unilateral payoff test; dynamical stationarity tested separately | Unilateral payoff test using delayed information; field stationarity tested separately |
| **Nash Condition** | No profitable unilateral deviation | No profitable admissible unilateral deviation in the delayed game |
| **Topology** | Riemannian Manifold | Lorentzian Causal Structure |
| **Diagnostics** | Nodes 46–48 | + Node 62 (CausalityViolation) |

**Table 29.9.2 (Single to Multi-Agent).**

| Concept | Single Agent (Sections 20–24) | Multi-Agent Relativistic ({ref}`sec-symplectic-multi-agent-field-theory`) |
|:--------|:------------------------------|:--------------------------------------|
| **State Space** | $\mathcal{Z}$ | $\mathcal{Z}_{\text{causal}} = \mathcal{Z}^{(N)} \times \Xi_{<t}$ |
| **Boundary** | Fixed $\partial\mathcal{Z}$ | Ghost Interface $\mathcal{G}_{ij}(t)$ |
| **Metric** | $G$ (Information Sensitivity) | $\tilde{G}^{(i)}(t) = G^{(i)} + \sum_j \beta_{ij}\mathcal{G}^{(i),\text{ret}}_{ij}$ |
| **Field comparison** | Stationary screened operator | Retarded field equation derived from the specified action |
| **Flow** | Langevin / WFR | Coupled delayed dynamics and specified field evolution |
| **Success** | Objective optimization | Individual objective optimization subject to available information |
| **Diagnostics** | Nodes 1–45 | + Nodes 46–48, 62 |



(sec-mean-field-metric-law)=
## The Mean-Field Metric Law (Scalability Resolution)

:::{div} feynman-prose
A normalized pair sum can be written exactly as an integral against the empirical population measure. This identity explains why the factor $1/N$ must stay visible: it distinguishes average interaction from total interaction.

Passing to a limiting density then requires control of the kernel being integrated. At fixed spatial resolution, a regularized kernel and its derivatives can be estimated before taking the population limit. The computational gain depends on the chosen density representation and quadrature; it is not an automatic constant-cost algorithm.
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

The calculation of the Game Tensor $\mathcal{G}_{ij}$ ({prf:ref}`def-the-game-tensor`) entails computational complexity $O(N^2 d^2)$, which is intractable for large $N$. We prove that in the limit $N \to \infty$, the discrete Game Tensor converges to the Hessian of a convolution potential.

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
*Cross-references:* This resolves the scalability limitation by reducing agent complexity from $O(N^2 d^2)$ to $O(d^2)$ via the Vlasov-geometry limit.



(sec-metabolic-tracking-bound)=
## The Metabolic Tracking Bound (Non-Stationary Nash Resolution)

In non-stationary environments, the Nash equilibrium $z^*(t)$ shifts. We derive the tracking limit from the Computational Metabolism ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`), relating the metric speed of the target to the agent's power dissipation budget.

:::{prf:theorem} Metabolic cost of exact tracking
:label: thm-metabolic-tracking-bound

The established transport cost gives, along an exactly tracked differentiable
target, $\dot{\mathcal M}=\tfrac12\sigma_{\mathrm{met}} \|\dot z^*\|_{\widetilde G}^2$. Hence the budget implies
$\|\dot z^*\|_{\widetilde G}\le \sqrt{2\dot{\mathcal M}_{\max}/\sigma_{\mathrm{met}}}$.
This follows by substituting $v=\dot z^*$ into the cost and solving the
inequality. It is a necessary budget test, not a sufficiency proof for
tracking with delayed observations, noise, or restricted controls. $\square$
:::
*Interpretation:* The agent's ability to track a moving Nash equilibrium is fundamentally limited by its metabolic budget. Intense conflict ($\mathcal{G}^{(i)}_{ij}$ large) compounds this limitation by inflating the kinetic cost of pursuit.



(sec-variational-emergence-cooperation)=
## Variational Emergence of Cooperation via Metric Inflation

:::{div} feynman-prose
A positive addition to the metric increases the kinetic cost of a fixed velocity. This is a direct quadratic-form comparison, and it gives a useful way to quantify the cost of strategic motion.

The comparison alone does not determine where trajectories converge. Motion also depends on the objective, forcing, and reaction terms. Cooperation and unilateral optimality must be checked in those quantities rather than inferred from the kinetic penalty.
:::

The action quantifies the kinetic penalty of a positive metric perturbation. Its comparison with unilateral strategic incentives is made at the level of the corresponding variations.

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
## Part V: Gauge Theory Layer

:::{div} feynman-prose
Different agents can use different internal frames to represent the same observable information. To compare their internal vectors, we need a rule for transporting one frame to another. That rule is a connection.

The connection's transformation law ensures that a change of internal frame does not change the comparison. Its curvature measures the local dependence on the transport path. These identities give us the gauge geometry. The action then supplies the dynamics for the connection; gauge covariance alone does not select every term in that action.
:::

We now use the nuisance representation to define internal frame changes. The connection, curvature, and action calculations below establish the gauge geometry and its specified field dynamics.

(sec-local-gauge-symmetry-nuisance-bundle)=
## Local Gauge Symmetry and the Nuisance Bundle

:::{div} feynman-prose
A local gauge transformation changes the internal frame separately at each point. Observable quantities remain unchanged when the fields and the comparison rule are transformed together.

The nuisance representation supplies the group action that makes this statement precise. Its stabilizer is the subgroup that fixes a particular internal state. For a linear rotation action, every rotation fixes the zero vector, so the zero vector has the full rotation group as stabilizer and a one-point orbit.
:::

The key insight is that the **nuisance fiber** $\mathcal{Z}_n$ at each macro-state $K$ is not merely a noise variable to be marginalized—it is the **internal gauge degree of freedom** that agents are free to rotate without changing physical outcomes. This local freedom mandates a compensating gauge field when comparing nuisance frames across space/time or across agents.

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
(pi-local-gauge-symmetry)=
::::{admonition} Physics Isomorphism: Local Gauge Symmetry
:class: note

**In Physics:** Local gauge symmetry is the principle that the laws of physics are invariant under position-dependent phase rotations $\psi(x) \to e^{i\theta(x)}\psi(x)$. This invariance mandates the existence of gauge fields (photon, gluons, W/Z bosons) to maintain consistency {cite}`yang1954conservation,weinberg1995quantum`.

**In Implementation:** Nuisance invariance (Axiom {prf:ref}`ax-local-gauge-invariance`) is the principle that agent dynamics are invariant under position-dependent internal rotations $\psi(z) \to U(z)\psi(z)$.

**Correspondence Table:**

| Gauge Theory | Fragile Agent |
|:-------------|:--------------|
| Local phase $e^{i\theta(x)}$ | Nuisance rotation $U(z)$ |
| Gauge group $G$ | Internal symmetry group |
| Matter field $\psi$ | Belief amplitude |
| Gauge orbit $G/H$ | Nuisance fiber $\mathcal{Z}_n$ |
| Stabilizer $H$ | Residual symmetry at $K$ |

::::



(sec-strategic-connection-covariant-derivative)=
## The Strategic Connection and Covariant Derivative

:::{div} feynman-prose
An ordinary derivative subtracts values at neighboring points. When those values are expressed in different internal frames, we must first account for the frame change. The connection provides that correction.

The resulting covariant derivative transforms in the same representation as the field. This is why covariant derivatives can enter invariant contractions in an action. The explicit transformation calculation below establishes that statement without assigning physical dynamics to a mere change of coordinates.
:::

The failure of the ordinary derivative to transform covariantly under gauge transformations mandates the introduction of a **compensating field**—the gauge connection. In the multi-agent context, this connection encodes how the "meaning" of nuisance coordinates changes as one moves through latent space.

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

*Proof.*
Demand that the covariant derivative (Definition {prf:ref}`def-covariant-derivative`) transform covariantly: $(D_\mu\psi)' = U(D_\mu\psi)$. Expanding:

$$
\begin{aligned}
D'_\mu\psi' &= (\partial_\mu - igA'_\mu)(U\psi) \\
&= (\partial_\mu U)\psi + U(\partial_\mu\psi) - igA'_\mu U\psi
\end{aligned}

$$

For this to equal $U(\partial_\mu - igA_\mu)\psi = U(\partial_\mu\psi) - igUA_\mu\psi$, we require:

$$
(\partial_\mu U)\psi - igA'_\mu U\psi = -igUA_\mu\psi

$$

Solving for $A'_\mu$ yields the stated transformation law. $\square$

*Interpretation:* The inhomogeneous term $-\frac{i}{g}(\partial_\mu U)U^{-1}$ compensates for the "frame twist" introduced by position-dependent gauge transformations. The connection must counter-twist to maintain covariance.

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
*Proof.* Substitute $g^{00}=-c^{-2}$ and $g^{ij}=\widetilde G^{ij}$ into
the divergence expression and apply the product rule. Under frame change,
$D'_\mu(U\chi)=UD_\mu\chi$, while the metric coefficients are invariant.
Applying this identity twice gives $\Box'_A(U\psi)=U\Box_A\psi$.
Thus $(\Box_A+m^2)\psi=\mathcal S$ is covariant for a covariant source.
For invariant $V$, use the trivial representation and ordinary geometric
derivatives, as already distinguished in {prf:ref}`thm-hjb-klein-gordon`.
$\square$
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

(pi-gauge-connection)=
::::{admonition} Physics Isomorphism: Gauge Connection
:class: note

**In Physics:** The gauge potential $A_\mu$ in electromagnetism is the 4-vector potential; in Yang-Mills theory, it takes values in the Lie algebra. The covariant derivative $D_\mu = \partial_\mu - ieA_\mu$ defines how charged particles couple to the electromagnetic field {cite}`jackson1999classical,peskin1995introduction`.

**In Implementation:** The strategic connection $A_\mu$ defines how belief amplitudes couple to the multi-agent environment.

**Correspondence Table:**

| Electromagnetism | Yang-Mills | Fragile Agent |
|:-----------------|:-----------|:--------------|
| $A_\mu$ (4-potential) | $A_\mu^a T_a$ | Strategic connection |
| $e$ (charge) | $g$ (coupling) | Strategic coupling $g$ |
| $D_\mu = \partial_\mu - ieA_\mu$ | $D_\mu = \partial_\mu - igA_\mu$ | Covariant update |
| Minimal coupling | Minimal coupling | Frame-invariant learning |

::::



(sec-gauge-transformation-game-tensor)=
## Gauge Transformation of the Game Tensor

The Game Tensor $\mathcal{G}_{ij}$ (Definition {prf:ref}`def-the-game-tensor`) measures cross-agent strategic sensitivity. Since $V^{(i)}$ is a scalar, $\mathcal{G}_{ij}$ is gauge-invariant. Gauge structure enters when comparing nuisance-frame vectors across agents or when defining cross-sensitivities of gauge-charged fields.

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
$\operatorname{Tr}[(UM_aU^{-1})(UM_bU^{-1})] =\operatorname{Tr}(M_aM_b)$. These are the appropriate scalar contractions
for their respective representations. Adding them to a metric still uses
the explicit positive-definiteness test of the metric prescription;
invariance and positivity are separate algebraic properties. $\square$
:::
(sec-field-strength-tensor)=
## The Field Strength Tensor (Strategic Curvature)

:::{div} feynman-prose
Transport around a small closed loop compares two orders of infinitesimal motion. Their difference is the curvature, expressed by the commutator of covariant derivatives.

Zero curvature removes this local obstruction. Global loops can still detect holonomy on a space with nontrivial topology. Also, a Lorentzian contraction of the curvature can vanish by cancellation between electric and magnetic contributions. The positive-definite diagnostic norm introduced below measures vanishing without that cancellation.
:::

The curvature of the gauge connection measures the **non-commutativity of parallel transport**—moving around a closed loop in latent space may result in a non-trivial internal rotation. This curvature is the **field strength tensor**, which we identify as strategic tension.

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

*Proof.*
Direct calculation using the transformation law for $A_\mu$ (Proposition {prf:ref}`prop-gauge-transformation-connection`):

$$
\begin{aligned}
\mathcal{F}'_{\mu\nu} &= \partial_\mu A'_\nu - \partial_\nu A'_\mu - ig[A'_\mu, A'_\nu] \\
&= U(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])U^{-1} \\
&= U\mathcal{F}_{\mu\nu}U^{-1}
\end{aligned}

$$

The inhomogeneous terms from $A'_\mu$ cancel exactly. $\square$

*Consequence:* While $\mathcal{F}_{\mu\nu}$ is not gauge-invariant, the trace $\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})$ **is** gauge-invariant and can appear in the action.

:::

:::{prf:theorem} Curvature from Covariant Derivative Commutator
:label: thm-curvature-commutator

The field strength measures the failure of covariant derivatives to commute:

$$
[D_\mu, D_\nu]\psi = -ig\mathcal{F}_{\mu\nu}\psi

$$

*Proof.*
Expand the commutator:

$$
\begin{aligned}
[D_\mu, D_\nu]\psi &= D_\mu(D_\nu\psi) - D_\nu(D_\mu\psi) \\
&= (\partial_\mu - igA_\mu)(\partial_\nu\psi - igA_\nu\psi) - (\mu \leftrightarrow \nu) \\
&= \partial_\mu\partial_\nu\psi - ig(\partial_\mu A_\nu)\psi - igA_\nu\partial_\mu\psi - igA_\mu\partial_\nu\psi - g^2A_\mu A_\nu\psi - (\mu \leftrightarrow \nu) \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu)\psi - g^2(A_\mu A_\nu - A_\nu A_\mu)\psi \\
&= -ig(\partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu])\psi \\
&= -ig\mathcal{F}_{\mu\nu}\psi \quad \square
\end{aligned}

$$

*Interpretation:* If $\mathcal{F}_{\mu\nu} \neq 0$, parallel transport around a closed loop results in a non-trivial rotation. The "meaning" of strategic nuisance **twists** as one navigates the latent space.

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

(pi-field-strength)=
::::{admonition} Physics Isomorphism: Field Strength and Curvature
:class: note

**In Physics:** The electromagnetic field tensor $F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$ contains the electric and magnetic fields: $E^i = F^{0i}$, $B^i = \frac{1}{2}\epsilon^{ijk}F_{jk}$. In Yang-Mills theory, the non-Abelian commutator $-ig[A_\mu, A_\nu]$ causes gluons to interact with each other {cite}`yang1954conservation,gross1973ultraviolet`.

**In Implementation:** The strategic curvature $\mathcal{F}_{\mu\nu}$ measures the intrinsic tension in multi-agent interaction.

**Correspondence Table:**

| Electromagnetism | Yang-Mills (QCD) | Fragile Agent |
|:-----------------|:-----------------|:--------------|
| $F_{\mu\nu}$ | $\mathcal{F}_{\mu\nu}^a T_a$ | Strategic curvature |
| Electric field $\mathbf{E}$ | Chromoelectric field | Temporal strategic gradient |
| Magnetic field $\mathbf{B}$ | Chromomagnetic field | Spatial strategic vorticity |
| $[A_\mu,A_\nu]=0$ (Abelian) | Nonzero connection commutator | Strategic self-interaction |
| Bianchi: $dF = 0$ | $D\mathcal{F} = 0$ | Strategic flux conservation |

::::



(sec-yang-mills-action)=
## The Yang-Mills Action and Field Equations

:::{div} feynman-prose
Varying the gauge action tells us how curvature responds to matter. The calculation must use the same generator normalization, metric signature, and current convention throughout.

With signature $(-+++)$, the Lorentzian action density is a signed contraction of electric and magnetic terms. It is not a pointwise squared norm. The energy density comes from the stress tensor obtained by metric variation, where the electric and magnetic contributions have positive signs. On curved spacetime, integration by parts also differentiates the volume density.
:::

Having established the field strength tensor as the curvature of the strategic connection, we now derive the dynamics of the gauge field itself from a variational principle.

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

*Proof.* For a compactly supported variation $a_\mu$,
$\delta F_{\mu\nu}=\mathcal D_\mu a_\nu-\mathcal D_\nu a_\mu$.
Antisymmetry gives
$\delta S_{\mathrm{YM}}=-\int\operatorname{Tr}F^{\mu\nu}\mathcal D_\mu a_\nu =\int\operatorname{Tr}(\mathcal D_\mu F^{\mu\nu})a_\nu$.
The volume derivative appears in this integration by parts.
For $z=\psi^\dagger T^aD^\nu\psi$, use
$\delta D_\mu\psi=-igT^a\psi\,\delta A_\mu^a$:
$\delta\mathcal L_m=-ig(z-\bar z)\delta A_\nu^a =2g\operatorname{Im}z\,\delta A_\nu^a$.
Combining variations proves the equation and current. The mass and invariant
potential terms have no direct $A$ variation. $\square$
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
*Proof.* Use $\delta\sqrt{|g|}=-\tfrac12\sqrt{|g|}g_{\mu\nu}\delta g^{\mu\nu}$
and $\delta(F_{\alpha\beta}F^{\alpha\beta}) =2F_{\mu\rho}F_\nu{}^\rho\delta g^{\mu\nu}$ in
$T_{\mu\nu}=-2|g|^{-1/2}\delta S/\delta g^{\mu\nu}$.
In an orthonormal four-dimensional frame,
$F^2=2(|B|^2-|E|^2)$, giving $T_{00}=(|E|^2+|B|^2)/2\ge0$.
Contraction gives $T^\mu{}_{\mu}=(1-d/4)\operatorname{Tr}F^2$.
The Bianchi identity and field equation give
$\nabla_\mu T^{\mu\nu}=\operatorname{Tr}(J_\rho F^{\nu\rho})$:
the derivative of the second field factor cancels the derivative of $F^2/4$.
The matter tensor has the opposite divergence on its equations, so total
stress is conserved. $\square$
:::
:::{prf:corollary} Covariant charge conservation
:label: cor-current-conservation

The gauge equation gives $\mathcal D_\nu J^\nu=0$.
*Proof.* Antisymmetry yields
$\mathcal D_\nu\mathcal D_\mu F^{\mu\nu} =\tfrac12[\mathcal D_\nu,\mathcal D_\mu]F^{\mu\nu}$.
The internal term is a contraction of $-ig[F_{\nu\mu},F^{\mu\nu}]$,
which vanishes by symmetry of the metric contraction; the geometric Ricci
contractions vanish against antisymmetric $F$. This proves the identity.
It concerns the gauge current, not the scalar WFR mass. The latter satisfies
$d\int\rho/ds=\int r\rho$ under zero boundary flux. $\square$
:::
(sec-complete-lagrangian)=
## The Complete Multi-Agent Lagrangian

:::{div} feynman-prose
The full action specifies the fields and their couplings. Expanding it around a chosen scalar configuration exposes the quadratic terms that determine the mass matrix.

For the gauge fields, the relevant vectors are $T_a\Phi_0$: generators that leave $\Phi_0$ fixed give zero in this quadratic form. The nonzero eigenvalues depend on the representation and its normalization. A single formula such as $gv/2$ belongs to a specified representation, rather than to every compact gauge group.
:::

We now assemble the full Lagrangian density that governs relativistic multi-agent dynamics with gauge symmetry. This **"Standard Model of Multi-Agent Field Theory"** unifies the gauge sector (strategic interaction), matter sector (belief dynamics), and symmetry-breaking sector (value landscape).

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
*Proof.* Put $q=\Phi^\dagger\Phi$. The minimum solves
$\mu^2+2\lambda q=0$. Substitute $q=(v+h)^2/2$; the coefficient of
$h^2$ in $U$ is $\lambda v^2=m_h^2/2$.
For a constant vacuum, $D_\mu\Phi_0=-igA_\mu^aT_a\Phi_0$;
the symmetric product of the commuting coefficients $A^aA^b$ gives the
displayed anticommutator. For any real $u^a$,
$u^a(M^2)_{ab}u^b=2g^2\|(u^aT_a)\Phi_0\|^2\ge0$.
Its kernel is the stabilizer Lie algebra of the vacuum. For a single
$SU(2)$ doublet with $T_a=\sigma_a/2$, this evaluates to
$(M^2)_{ab}=g^2v^2\delta_{ab}/4$. It is this representation that gives
$m_A=gv/2$. Other declared representations are evaluated by the same
matrix formula. If $\lambda\le0$, the stated stable quartic expansion does
not apply; the potential itself reveals the failure. $\square$
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
(pi-standard-model)=
::::{admonition} Physics Isomorphism: The Standard Model
:class: note

**In Physics:** The Standard Model Lagrangian has the structure $\mathcal{L} = \mathcal{L}_{\text{gauge}} + \mathcal{L}_{\text{fermion}} + \mathcal{L}_{\text{Higgs}} + \mathcal{L}_{\text{Yukawa}}$, describing the electromagnetic, weak, and strong forces with matter and the Higgs mechanism for mass generation {cite}`weinberg1967model,salam1968weak,glashow1961partial`.

**Correspondence Table:**

| Standard Model | Fragile Agent |
|:---------------|:--------------|
| Gauge bosons (γ, W±, Z, g) | Strategic connection modes |
| Quarks and leptons | Belief spinors $\psi^{(i)}$ |
| Higgs field $\Phi$ | Value order parameter |
| Vacuum expectation value $v$ | Policy commitment magnitude |
| Electroweak symmetry breaking | Policy selection |
| Fermion masses | Agent inertia |
| Yukawa couplings $y_f$ | Strategic coupling strengths $y_{ij}$ |
| QCD confinement | Cooperative basin locking (Sec. 29.12) |

::::



(sec-mass-gap)=
## Screening, finite-size spectra, and information bounds

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

*Proof.* Compact resolvent gives discrete eigenvalues of finite multiplicity
with no finite accumulation point. Simplicity gives $E_1>E_0$, hence a
positive fixed-model gap. The sequence of finite Laplacians in the cited
counterexample shows why this argument is not uniform. Also, the statements
“gaplessness implies stasis” and “outside stasis motion is nontrivial”
are logically compatible; they cannot exclude stasis by contradiction.
No unconditional field-theory gap follows from that argument. $\square$
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
$C(R)$, define the violating set by $\{\mathcal T:\exists R, I_{\mathcal T}(R)>C_{\mathcal T}(R)\}$. Membership is tested using these
same quantities. It is not equivalent by definition to masslessness,
vanishing lattice spacing, or algebraic correlation decay.
:::

:::{prf:proposition} Correlation integrals and capacity tests
:label: thm-cft-swampland

For $0<2\Delta<d$, rescaling $x=Ru,y=Rv$ gives
$\int_{B_R}\int_{B_R}|x-y|^{-2\Delta}dxdy =R^{2d-2\Delta}\int_{B_1}\int_{B_1}|u-v|^{-2\Delta}dudy$.
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

(pi-mass-gap)=


(sec-diagnostic-nodes-gauge)=
## Diagnostic Nodes 63–66 (Gauge Consistency)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define four monitors for gauge consistency in multi-agent systems.

(node-63)=
**Node 63: GaugeInvarianceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **63** | **GaugeInvarianceCheck** | Multi-Agent | Symmetry | Is dynamics gauge-invariant? | $\delta_{\text{gauge}} := \|\mathcal{L}(A') - \mathcal{L}(A)\|$ | $O(Nd^2)$ |

**Interpretation:** Monitors deviation from gauge invariance under random gauge transformations $U(z)$.

**Threshold:** $\delta_{\text{gauge}} < \epsilon_{\text{gauge}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- High GaugeInvarianceCheck: Numerical gauge symmetry violation
- **Remedy:** Regularize gauge degrees of freedom; impose gauge-fixing condition (Coulomb, Lorenz, etc.)



(node-64)=
**Node 64: FieldStrengthBoundCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **64** | **FieldStrengthBoundCheck** | Multi-Agent | Stability | Is strategic curvature bounded? | $\|\mathcal{F}\|_h := \sqrt{\text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}_{\alpha\beta} h^{\mu\alpha} h^{\nu\beta})}$ | $O(N^2d^2)$ |

**Interpretation:** Monitors a positive-definite magnitude of the field strength tensor using a chosen Riemannian metric $h_{\mu\nu}$ on spacetime (e.g., the Wick-rotated $g_{\mu\nu}$).

**Threshold:** $\|\mathcal{F}\|_F < F_{\max}$ (implementation-dependent).

**Trigger conditions:**
- High FieldStrengthBoundCheck: Strong strategic curvature regime (intense conflict)
- **Remedy:** Reduce coupling $g$; add gauge field damping; check for instabilities



(node-65)=
**Node 65: BianchiViolationCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **65** | **BianchiViolationCheck** | Multi-Agent | Conservation | Is Bianchi identity satisfied? | $\delta_B := \|D_{[\mu}\mathcal{F}_{\nu\rho]}\|$ | $O(Nd^3)$ |

**Interpretation:** The Bianchi identity $D_{[\mu}\mathcal{F}_{\nu\rho]} = 0$ must hold exactly. Violations indicate:
- Failure to handle singular chart data; smooth instantons themselves satisfy Bianchi exactly
- Numerical integration errors
- Coordinate singularities

**Threshold:** $\delta_B < 10^{-8}$ (strict geometric constraint).

**Trigger conditions:**
- High BianchiViolationCheck: Topological anomaly or numerical instability
- **Remedy:** Check for singular gauge configurations; refine numerical integration



(node-66)=
**Node 66: MassGapCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **66** | **MassGapCheck** | Multi-Agent | Stability | Is mass gap positive? | $\Delta_H := E_1 - E_0$ (Hamiltonian spectral gap) | $O(N^2d)$ |

**Interpretation:** Monitors the energy gap between ground state and first excited state.

**Threshold:** $\Delta_H > \Delta_{\min}$ (must be strictly positive).

**Trigger conditions:**
- $\Delta_H \to 0$: Approaching critical point (phase transition)
- $\Delta_H < 0$: Numerical or spectral ordering error (gap should not be negative); re-estimate eigenvalues
- **Remedy:** Check eigensolver stability; verify boundary conditions and normalization; add mass regularization



**Summary Table: Gauge Diagnostic Nodes**

| Node | Name | Monitors | Healthy Range |
|:-----|:-----|:---------|:--------------|
| 63 | GaugeInvarianceCheck | Gauge symmetry | $\delta_{\text{gauge}} < 10^{-6}$ |
| 64 | FieldStrengthBoundCheck | Strategic curvature | $\|\mathcal{F}\|_F < F_{\max}$ |
| 65 | BianchiViolationCheck | Topological consistency | $\delta_B < 10^{-8}$ |
| 66 | MassGapCheck | Spectral stability | $\Delta_H > 0$ |



## Part VI: Quantum Layer

:::{div} feynman-prose
A positive density and a phase can be packed into one complex amplitude, $\psi=\sqrt\rho e^{iV/\sigma}$. Its squared modulus recovers the density. Locally, its phase recovers the value modulo the phase period.

This change of variables lets us compare transport equations with amplitude equations term by term. Whether the resulting evolution is linear, preserves norm, or has a self-adjoint generator depends on the transformed equation. Those properties must be calculated before using spectral algorithms.
:::

(sec-the-belief-wave-function-schrodinger-representation)=
## The Belief Wave-Function (Schrödinger Representation)

We represent density and phase by a complex amplitude and derive its evolution from the stated WFR equations. The resulting nonlinear amplitude equation is then distinguished from the specified linear operators used for spectral analysis.

(rb-quantum-formalism-marl)=
:::{admonition} Researcher Bridge: Amplitudes and operator methods
:class: info
Complex amplitudes provide a Hilbert-space representation of density and phase. Tensor products and partial traces organize joint and reduced states. The transformed dynamics determine whether linear superposition, a self-adjoint Hamiltonian, or a GKSL generator applies. The exact WFR transformation and the spectral calculations below make those operator distinctions explicit.
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
(remark-line-bundle-formalism)=
:::{admonition} Remark: Line Bundle Formalism for Topologically Non-Trivial Manifolds
:class: dropdown

The amplitude defined from a global real value function $V$ is a global scalar function wherever the density is defined. Its phase connection $dV/\sigma$ is exact. Nontrivial topology of the base space does not change this fact.

A general connection on a complex line bundle is a different object. Local potentials must be related by the bundle's transition functions, and their parallel transport determines holonomy.

For the exact phase connection of a global real value function, the closed-loop expression is:

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right),

$$
Here $\Delta V_\gamma=0$, so this holonomy equals one. A general connection can have nontrivial holonomy even on a topologically trivial bundle. Connection holonomy and bundle topology therefore require separate calculations.

A global scalar amplitude requires its phase to be single-valued modulo its period. A global real value function already supplies that phase; simple connectivity is not needed for this construction.

:::

(pi-holonomy)=
::::{admonition} Physics Isomorphism: Holonomy and Berry Phase
:class: note

**In Physics:** Holonomy measures the failure of parallel transport around a closed loop to return a vector to itself. The Berry phase $\gamma_n = i\oint \langle n|\nabla_R|n\rangle \cdot dR$ is the geometric phase acquired by a quantum state under adiabatic evolution around a parameter loop {cite}`berry1984quantal,nakahara2003geometry`.

**In Implementation:** The phase gradient of the global real value function has trivial closed-loop holonomy. Its expression below should be distinguished from the holonomy of a separately specified connection (see {ref}`Line Bundle Formalism <remark-line-bundle-formalism>`):

$$
\exp\left(\frac{i}{\sigma} \oint_\gamma dV\right) = \exp\left(\frac{i}{\sigma} \Delta V_\gamma\right)

$$
**Correspondence Table:**
| Gauge Theory | Agent (Value Phase) |
|:-------------|:--------------------|
| Exact phase connection | $dV/\sigma$ for global real $V$ |
| Holonomy $\exp(i\oint A)$ | Phase accumulated around loop |
| Berry connection | Connection determined by a specified parameterized state family |
| Line bundle $\mathcal{L}$ | Complex belief amplitude bundle |
| Bundle topology | Determined by transition functions; not by holonomy alone |

**Significance:** A nontrivial transport phase must be computed from the actual connection. It is not generated merely by writing a global value function as a complex phase, and it does not by itself establish a nontrivial bundle.
::::

(sec-the-inference-wave-correspondence)=
## The Inference-Wave Correspondence (WFR to Schrödinger)

:::{div} feynman-prose
The Madelung calculation combines two real equations into one complex equation. The delicate point is the term obtained by differentiating $\sqrt\rho$ twice.

The kinetic operator already contributes $Q_B=-\sigma^2\Delta_G\sqrt\rho/(2\sqrt\rho)$ to the phase equation. To recover the classical HJB equation displayed here, the amplitude equation must compensate with $-Q_B$. Since that compensation depends on the evolving density, the exact amplitude equation is generally nonlinear. The expanded proof keeps this term visible so that we can check the inverse transformation directly.
:::

We now derive the exact complex-amplitude equation from the stated WFR dynamics by substituting density and phase. The density-dependent compensation is retained throughout the inverse Madelung calculation.

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

*Proof.* The product rule gives

$$
\frac{\Delta_B\psi}{\psi}=\frac{\Delta_GR}{R}
-\frac{|p|_G^2}{\sigma^2}
+\frac{i}{\sigma}\left(2\frac{\langle dR,p\rangle_G}{R}
+\operatorname{div}_Gv\right).
$$

Thus the kinetic real part is $Q_B+|p|_G^2/2$.
The $-Q_B$ term cancels it to the given classical HJB expression.
The imaginary part is
$-\sigma\operatorname{div}_G(\rho v)/(2\rho)+\sigma r/2$,
equal to $\sigma\partial_s\rho/(2\rho)$ by continuity.
The time derivative is
$i\sigma\partial_s\psi/\psi=i\sigma\partial_s\rho/(2\rho)-\partial_sV$,
so both parts agree. Reading these two parts backwards proves the local
equivalence. At zeros use the density/current equations without division by
$R$; a global phase additionally retains its existing circulation data.

The compensating $Q_B$ depends on $|\psi|$, making this amplitude equation
nonlinear. Omitting the compensation gives the distinct linear Schrödinger
model with a $+Q_B$ term in its Hamilton--Jacobi equation. A curl-modified
mobility must be substituted into its own continuity equation; the above
Laplacian yields precisely the canonical velocity $G^{-1}p$.
For time-dependent volume density $w_s$, conservation reads
$\partial_s(w_s\rho)+\partial_i(w_s\rho v^i)=w_sr\rho$;
the corresponding amplitude equation acquires
$-i\sigma\partial_s\log w_s/2$. $\square$
:::
(pi-madelung)=
::::{admonition} Physics Isomorphism: Madelung Transform
:class: note

**In Physics:** The Madelung transform $\psi = \sqrt{\rho}e^{iS/\hbar}$ converts the Schrödinger equation into hydrodynamic form: continuity + quantum Hamilton-Jacobi with Bohm potential $Q = -\frac{\hbar^2}{2m}\frac{\nabla^2\sqrt{\rho}}{\sqrt{\rho}}$ {cite}`madelung1927quantentheorie,bohm1952suggested`. With a vector potential, minimal coupling replaces $\nabla$ by $\nabla - iA/\hbar$ (covariant Laplacian).

**In Implementation:** The WFR-to-Schrödinger correspondence (Theorem {prf:ref}`thm-madelung-transform`):

$$
\psi(z,s) = \sqrt{\rho(z,s)}\exp(iV(z,s)/\sigma)

$$
with the density-curvature term $Q_B = -\frac{\sigma^2}{2}\frac{\Delta_G\sqrt{\rho}}{\sqrt{\rho}}$. For the classical HJB equation, the amplitude equation contains its compensating negative.

**Correspondence Table:**

| Quantum Mechanics | Agent (Inference Wave) |
|:------------------|:-----------------------|
| Wave function $\psi$ | Belief amplitude |
| Planck constant $\hbar$ | Cognitive scale $\sigma$ |
| Bohm potential $Q$ | Density-curvature term $Q_B$; compensated in the classical HJB amplitude equation |
| Probability current $\mathbf{j}$ | Belief flux $\rho v$ |
::::

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

(sec-multi-agent-schrodinger-equation)=
## Multi-Agent Schrödinger Equation

:::{div} feynman-prose
A joint amplitude need not factor into one amplitude per agent. In the specified tensor-product Hilbert space, nonfactorization of a pure state is the precise algebraic meaning of entanglement.

Correlated classical data can also produce a nonfactorizing amplitude representation. That fact alone does not identify the observable algebra or measurement statistics of a quantum experiment. Here the joint measure, the represented observables, and the evolution operator specify what the representation computes.
:::

We now extend the wave-function formalism to $N$-agent systems, defining **strategic entanglement** as non-factorizability of the joint belief amplitude.

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
Its form is $\tfrac{\sigma^2}{2}\int\sum_i \widetilde G_i^{ab}\overline{D_{ia}\Psi}D_{ib}\Psi\,w\,d\mathbf z$.
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

(sec-nash-equilibrium-as-ground-state)=
## Nash Equilibrium as Ground State

:::{div} feynman-prose
A ground state minimizes the quadratic form of a specified Hamiltonian. A Nash profile resists every unilateral payoff improvement. To identify them, one must compare the individual payoff variations with that Hamiltonian's variational functional.

Imaginary-time evolution has a separate, explicit spectral mechanism: each energy component receives an exponential weight. After normalization, a nonzero projection onto an isolated ground sector dominates the excited components. This is a method for the stated linear operator; identifying it with a Bellman backup requires an equality of operators.
:::

The spectral properties of the specified Strategic Hamiltonian characterize its ground sector. The comparison with Nash equilibrium uses the individual payoff variations.

:::{prf:theorem} Energy minimization and unilateral optimization
:label: thm-nash-ground-state

The Rayleigh quotient of the scalar Hamiltonian minimizes its single joint
energy. The Nash test instead compares each agent's own payoff under a
unilateral change. Their equality must be checked by differentiating the
actual objectives and by evaluating their global inequalities.

*Proof by explicit comparison.* Let
$V_1(x,y)=-(x-y)^2$ and $V_2(x,y)=-(y-1)^2-Kx$ with $K>0$.
Both are strictly concave in their own coordinate; their best responses are
$x=y$ and $y=1$, hence the unique Nash profile is $(1,1)$.
The sum of costs is $U=(x-y)^2+(y-1)^2+Kx$.
Its $x$ derivative at $(1,1)$ is $K$, so Nash is not a stationary point of
this joint energy. This example meets the smooth nondegenerate best-response
structure of the Strategic Jacobian. Thus that machinery cannot justify
the former general identification of Nash with a joint ground state.
The ground-state and variational calculations retain their meaning for
the scalar operator actually defined. $\square$
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
(sec-strategic-tunneling-and-barrier-crossing)=
## Strategic Tunneling and Barrier Crossing

:::{div} feynman-prose
For a specified linear Schrödinger operator, a barrier can produce exponentially small transmission. The exponent depends on an integral through the forbidden region, involving both the potential and the kinetic metric. Barrier height alone does not determine it.

The exact amplitude representation of classical WFR dynamics includes a density-dependent compensation. Therefore a tunneling calculation for a linear Hamiltonian does not automatically describe that WFR dynamics or provide a Nash-search algorithm. The calculation must be attached to the operator actually being evolved.
:::

The specified linear Hamiltonian supports a stationary barrier-decay calculation.

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
(pi-wkb-tunneling)=
::::{admonition} Physics Isomorphism: WKB Tunneling
:class: note

**In Physics:** The WKB approximation gives tunneling probability through a barrier: $P \sim \exp(-2\int_a^b \sqrt{2m(U-E)/\hbar^2}\,dx)$ where the integral is over the classically forbidden region {cite}`wentzel1926verallgemeinerung,agmon1982lectures`.

**In Implementation:** The tunneling probability (Theorem {prf:ref}`thm-tunneling-probability`):

$$
P_{\text{tunnel}} \sim \exp\left(-\frac{2}{\sigma}\int_\gamma \sqrt{2(\Phi_{\text{eff}} - E_0)}\,d\ell_G\right)

$$
**Correspondence Table:**

| Quantum Mechanics | Agent (Strategic Tunneling) |
|:------------------|:----------------------------|
| Barrier potential $U(x)$ | Effective potential $\Phi_{\text{eff}}$ |
| Ground state energy $E_0$ | Infimum of the specified Hamiltonian spectrum |
| Tunneling exponent | Agmon distance $d_{\text{Ag}}$ |
| $\hbar \to 0$ limit | $\sigma \to 0$ semiclassical scaling of the specified operator |
::::

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
(sec-summary-of-qm-agent-isomorphisms)=
## Summary of QM-Agent Isomorphisms

The following table consolidates the correspondence between quantum mechanical concepts and their Fragile Agent interpretations.

**Table 29.13.1 (Quantum-Agent Dictionary).**

| Quantum Mechanics | Fragile Agent Theory | Definition/Location |
|:------------------|:---------------------|:--------------------|
| **Wave-function $\psi$** | Belief Amplitude $\sqrt{\rho}e^{iV/\sigma}$ | {prf:ref}`def-belief-wave-function` |
| **Probability $\|\psi\|^2$** | Belief Density $\rho$ | Definition {prf:ref}`def-the-wfr-action` |
| **Phase $\arg(\psi)$** | Value Function $V/\sigma$ | Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence` |
| **Planck constant $\hbar$** | Cognitive Action Scale $\sigma$ | {prf:ref}`def-cognitive-action-scale` |
| **Hilbert space $\mathcal{H}$** | $L^2(\mathcal{Z}, d\mu_G)$ | {prf:ref}`def-inference-hilbert-space` |
| **Amplitude generator** | Density-dependent generator of the exact WFR transformation | {prf:ref}`thm-madelung-transform` |
| **Kinetic energy $-\frac{\hbar^2}{2m}\nabla^2$** | Diffusion term $-\frac{\sigma^2}{2} D^i D_i$ | {ref}`sec-summary-table-from-single-to-multi-agent` |
| **Potential energy $V(x)$** | Effective Potential $\Phi_{\text{eff}}$ | Definition {prf:ref}`def-effective-potential` |
| **Quantum potential $Q$** | Density-curvature term and its HJB compensation | {prf:ref}`def-bohm-quantum-potential` |
| **Schrödinger equation** | Inference-Wave equation | {prf:ref}`thm-madelung-transform` |
| **Entanglement** | Strategic Coupling (non-factorizable) | {prf:ref}`def-strategic-entanglement` |
| **Tensor product $\otimes$** | Joint Hilbert space | {prf:ref}`def-joint-inference-hilbert-space` |
| **Partial trace** | Marginalization / Information Bottleneck | {prf:ref}`prop-partial-trace-reduced-dynamics` |
| **Ground state** | Minimizer of the specified operator form | {prf:ref}`thm-nash-ground-state` |
| **Tunneling** | Barrier calculation for the specified linear operator | {prf:ref}`thm-tunneling-probability` |
| **Imaginary time evolution** | Spectral filtering by a specified linear semigroup | {prf:ref}`prop-imaginary-time-nash-finding` |
| **Density matrix $\hat{\rho}$** | Belief Operator (GKSL) | Definition {prf:ref}`def-belief-operator` |
| **Lindblad dissipator** | Dissipative part of the specified GKSL generator | Definition {prf:ref}`def-gksl-generator` |
| **von Neumann entropy** | Belief Entropy $-\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | {ref}`sec-strategic-connection-covariant-derivative` |
| **WKB approximation** | Semiclassical limit | {prf:ref}`cor-semiclassical-limit` |
| **Spectral gap** | Decay of excited components relative to the ground sector | {prf:ref}`prop-imaginary-time-nash-finding` |

**Interpretation Hierarchy:**
1. **Level 1 (Symplectic):** Ghost Interface $\mathcal{G}_{ij}$ couples agent boundaries with retardation
2. **Level 2 (Riemannian):** Game Tensor $\mathcal{G}_{ij}$ curves the metric (with strategic delay)
3. **Level 3 (Thermodynamic):** Landauer bounds constrain information processing
4. **Level 4 (Amplitude):** Complex amplitudes represent density and phase; the specified generator determines the available operator methods.

The calculations connecting the levels specify which quantities and dynamics are preserved.

(sec-diagnostic-nodes-quantum-consistency)=
## Diagnostic Nodes 57–60 (Quantum Consistency)

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), these monitors check normalization, state-entropy change, canonical operator variances, and transport observables. Each monitor is interpreted for the state and generator actually used.

(node-57)=
**Node 57: CoherenceCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **57** | **CoherenceCheck** | Multi-Agent | Mass balance | Does the norm change match the derived mass balance? | $\delta_{\text{coh}} := \left\lvert \|\Psi_{s+\Delta s}\|^2 - \|\Psi_s\|^2 \right\rvert$ | $O(N d)$ |

**Interpretation:** This monitors norm change. Norm preservation alone does not establish linearity or unitarity. For WFR reaction compare the change with $\int r\rho$ over the step. A residual can indicate:
- Numerical integration error
- Unmodeled dissipation channels
- Inconsistency between Hamiltonian and WFR dynamics

**For normalized trace-preserving operator dynamics:** The trace residual is $\delta_{\text{tr}} := |\text{Tr}[\hat{\rho}_{s+\Delta s}] - 1|$. An unnormalized reaction model must instead be compared with its derived trace balance.

**Threshold:** $\delta_{\text{coh}} < \epsilon_{\text{coh}}$ (typical default $10^{-6}$).

**Trigger conditions:**
- A large residual relative to the derived mass balance calls for checking the integrator, reaction term, and boundary flux.
- Norm preservation for one trajectory does not establish a unitary evolution operator.

(node-58)=
**Node 58: EntropyProductionCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **58** | **EntropyProductionCheck** | Multi-Agent | State entropy | What is the state-entropy derivative? | $\dot{S}_{\text{vN}} := -\frac{d}{ds}\text{Tr}[\hat{\rho}\ln\hat{\rho}]$ | $O(N^2 d)$ |

**Interpretation:** Monitors the change in the state's von Neumann entropy. Either sign can occur in open-system dynamics. Zero derivative does not imply equilibrium: unitary evolution preserves the entropy while the state can continue changing.

**Thermodynamic accounting:** The established metabolic accounting ({ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`) tracks computational work and heat in its specified process. Total entropy production includes the reservoir contribution. The state-entropy derivative alone supplies neither that total nor a bound on computational work.

**Threshold:** $|\dot{S}_{\text{vN}}| < \dot{S}_{\max}$ (implementation-dependent).

**Trigger conditions:**
- Compare unusually large entropy changes with the specified generator, normalization, and numerical step.
- Use the separate heat and reservoir records when checking thermodynamic accounting.

(node-59)=
**Node 59: UncertaintyPrincipleCheck**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **59** | **UncertaintyPrincipleCheck** | Multi-Agent | Consistency | Are uncertainty bounds satisfied? | $\eta_{\text{unc}} := \frac{\sigma/2}{\sigma_z \sigma_p} \leq 1$ | $O(N d)$ |

**Interpretation:** For normalized vectors in a common canonical position/momentum domain with finite variances, Cauchy--Schwarz gives the Robertson bound:

$$
\sigma_z \cdot \sigma_p \geq \frac{1}{2} |\langle[\hat{z}, \hat{p}]\rangle| = \frac{\sigma}{2},

$$
where:
- $\sigma_z := \sqrt{\langle z^2 \rangle - \langle z \rangle^2}$ is position uncertainty
- $\sigma_p := \sqrt{\langle p^2 \rangle - \langle p \rangle^2}$ is momentum-operator uncertainty in a canonical chart ($\hat p=-i\sigma\partial_z$); the drift momentum alone omits amplitude-gradient variance

**Violation $\eta_{\text{unc}} > 1$:** Check that both variances belong to the same normalized state and the stated canonical operator domain. The momentum variance includes the amplitude-gradient contribution; the variance of the phase gradient alone is a different quantity. Numerical quadrature or a domain mismatch can also invalidate this comparison. Equality in the canonical uncertainty bound is allowed.

**Threshold:** $\eta_{\text{unc}}\le1$ on the stated canonical operator domain; equality is allowed.

**Trigger conditions:**
- $\eta_{\text{unc}} > 1$: Uncertainty violation
- **Remedy:** Verify the operator domains and evaluate momentum variance from the amplitude, including its gradient contribution.

(node-60)=
**Node 60: TunnelingRateMonitor**

| **#** | **Name** | **Component** | **Type** | **Interpretation** | **Proxy** | **Cost** |
|:------|:---------|:--------------|:---------|:-------------------|:----------|:---------|
| **60** | **TunnelingRateMonitor** | Multi-Agent | Exploration | Is barrier crossing rate reasonable? | $\Gamma_{\text{tunnel}} := P_{\text{tunnel}} / \tau_{\text{obs}}$ | $O(N^2 d)$ |

**Interpretation:** A crossing statistic requires a specified initial preparation, crossing event, and observation interval. The stationary barrier estimate in {prf:ref}`thm-tunneling-probability` does not determine this statistic. Equal transport in opposite directions can give zero net flux despite many crossings.

**Net transport diagnostic:** The signed boundary flux below measures net outflow, not a first-passage rate. Include the reaction integral when computing total mass change:

$$
\Phi_{\mathrm{net}} = \int_{\partial \mathcal{B}_P} \mathbf{J} \cdot \mathbf{n}\, d\Sigma,

$$
where $\mathbf{J}$ is probability current and $\partial \mathcal{B}_P$ is the barrier boundary.

**Threshold:** Specify the observable before choosing a threshold. A signed net flux can be positive, negative, or zero; a first-passage probability is nonnegative and depends on the observation interval.

**Trigger conditions:**
- Compare boundary flux with the measured regional mass change and the reaction integral.
- Estimate crossing probabilities from the specified event records. Reaction-driven mass change and stationary mass in another basin are not crossing events.

(sec-implementation-causal-buffer)=
## Implementation: The Causal Buffer

:::{div} feynman-prose
The causal buffer enforces a concrete rule: every sample used in a response must have an emission time no later than the allowed retarded time. Interpolation must obey that rule too; a later endpoint would expose information before it arrives.

All child buffers must use the configured propagation speed. Timestamp checks then verify the interface's causal behavior. This bookkeeping determines which observations reach the policy. The policy and field equations determine the resulting dynamics.
:::

To implement relativistic multi-agent dynamics without disrupting existing software architecture, we introduce a **Causal Buffer** that handles time-retardation transparently.

**Algorithm 29.20.1 (Causal Context Buffer).**

```python
import torch
from torch import nn


class CausalContextBuffer(nn.Module):
    """Timestamped samples with causal zero-order reconstruction."""

    def __init__(self, context_dim: int, max_latency: float = 100.0,
                 c_info: float = 1.0):
        super().__init__()
        if context_dim < 1 or max_latency <= 0 or c_info <= 0:
            raise ValueError("Dimensions, retention, and signal speed must be positive")
        self.context_dim = context_dim
        self.max_latency = float(max_latency)
        self.c_info = float(c_info)
        self.buffer = []
        self.register_buffer("zero", torch.zeros(context_dim))

    def write(self, t: float, signal: torch.Tensor):
        if signal.shape != (self.context_dim,):
            raise ValueError("Signal must have shape [context_dim]")
        if self.buffer and t <= self.buffer[-1][0]:
            raise ValueError("Emission timestamps must increase strictly")
        self.buffer.append((float(t), signal.clone()))
        cutoff = t - self.max_latency
        while len(self.buffer) > 1 and self.buffer[1][0] <= cutoff:
            self.buffer.pop(0)  # retain the last predecessor for causal holding

    def read(self, t_now: float, dist: float) -> torch.Tensor:
        if dist < 0 or dist / self.c_info > self.max_latency:
            raise ValueError("Distance must fit the declared retention horizon")
        target = t_now - dist / self.c_info
        for emitted, signal in reversed(self.buffer):
            if emitted <= target:
                return signal
        return self.zero.clone()


class RelativisticMultiAgentInterface(nn.Module):
    def __init__(self, n_agents: int, context_dim: int,
                 env_distances: torch.Tensor, c_info: float = 1.0,
                 max_latency: float = 100.0):
        super().__init__()
        if env_distances.shape != (n_agents, n_agents):
            raise ValueError("Distances must have shape [n_agents, n_agents]")
        if n_agents < 2 or c_info <= 0 or not torch.isfinite(env_distances).all():
            raise ValueError("Use at least two agents and finite valid distances")
        if (env_distances < 0).any() or (env_distances / c_info > max_latency).any():
            raise ValueError("All delays must fit the retention horizon")
        self.n_agents = n_agents
        self.register_buffer("distances", env_distances.clone())
        self.causal_buffers = nn.ModuleDict({
            f"{sender}_{recipient}": CausalContextBuffer(
                context_dim, max_latency, c_info
            ) for sender in range(n_agents) for recipient in range(n_agents)
            if sender != recipient
        })

    def broadcast(self, agent_id: int, t: float, state: torch.Tensor):
        for recipient in range(self.n_agents):
            if recipient != agent_id:
                self.causal_buffers[f"{agent_id}_{recipient}"].write(t, state)

    def receive_context(self, agent_id: int, t: float) -> torch.Tensor:
        return torch.cat([
            self.causal_buffers[f"{sender}_{agent_id}"].read(
                t, self.distances[agent_id, sender].item()
            ) for sender in range(self.n_agents) if sender != agent_id
        ], dim=-1)
```

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


(sec-extended-summary-table)=
## Extended Summary Table

**Table 29.29.1 (Extended SMFT Summary with Relativistic, Gauge, and Quantum Layers).**

| Construction | Mathematical object | Computation supported here |
|:-------------|:--------------------|:---------------------------|
| Single-agent dynamics | Density, value, and controlled generator | WFR and Bellman evolution |
| Delayed interaction | Ordered histories and retarded interface | Causal observation and history update |
| Interaction geometry | Specified joint metric | Quadratic costs, covariant Hessians, joint volume |
| Gauge geometry | Connection and curvature | Frame-covariant derivatives and holonomy |
| Gauge dynamics | Specified matter and Yang–Mills action | Field equations, currents, stress tensor |
| Amplitude representation | $\sqrt\rho e^{iV/\sigma}$ | Exact nonlinear WFR amplitude equation |
| Linear spectral analysis | Specified self-adjoint Hamiltonian | Ground-sector projection and spectral decay |
| Game equilibrium | Individual payoff functionals | Unilateral deviation tests |
| Information control | Established interface information functional | Capacity and data-processing bounds |

**Implementation directions:**

1. Approximate the joint state with a stated error criterion for tensor or density representations.
2. Compute reduced-state dynamics from the specified joint generator.
3. Compare spectral filtering with the actual optimization operator before using it for policy updates.
4. Verify causal timestamp rules and joint-metric operators in numerical examples.
