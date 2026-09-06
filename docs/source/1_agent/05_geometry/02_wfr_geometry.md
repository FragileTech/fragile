(sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces)=
# Wasserstein-Fisher-Rao Geometry: Unified Transport on Hybrid State Spaces

## TLDR

- When state is hybrid (discrete $K$ + continuous $z_n$), belief evolution requires a geometry that handles both **flow**
  (transport within charts) and **jumps** (reaction between charts).
- The Wasserstein–Fisher–Rao (WFR / Hellinger–Kantorovich) metric gives a **single variational principle** for this
  hybrid belief dynamics.
- WFR resolves “duct-tape” product-metric heuristics by pricing transport and reaction consistently, and it yields
  implementable consistency diagnostics (WFRCheck).
- Think operationally: belief is a **fluid** that can move and transform; WFR measures the cheapest way to do both.
- This chapter connects the metric law (capacity) to dynamics (geodesic jump-diffusion) and to the information bound
  (area-law-like limits).

## Roadmap

1. Why product metrics fail for hybrid state spaces.
2. Define WFR and interpret transport vs. reaction components.
3. Connect WFR to filtering/control objectives and to implementation diagnostics.

:::{div} feynman-prose
Let me tell you about one of the most elegant solutions I've ever seen to a problem that seems hopelessly messy at first.

Here's the situation. Our agent has an internal representation that mixes discrete and continuous parts. The discrete part says "what kind of situation is this?"---are we in the kitchen or the living room, is the object a cup or a bottle? The continuous part says "where exactly within that situation?"---the precise position, orientation, all the fine details.

Now, the standard approach is to handle these separately. You have some kind of graph or finite-state machine for the discrete part, and a Riemannian manifold for the continuous part, and you glue them together with duct tape and hope for the best. When does the agent "jump" from one discrete state to another? How do you compare paths that involve different combinations of jumping and moving? The whole thing becomes a computational and conceptual nightmare.

But there's a beautiful way out of this mess, and it comes from a surprising place: optimal transport theory. The key insight is deceptively simple: stop thinking about the agent's state as a *point* that moves around, and start thinking about it as a *distribution* that flows and transforms.
:::

The latent bundle $\mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}}$ ({ref}`Section 2.2a <sec-the-trinity-of-manifolds>`) combines a discrete macro-state $K$ with continuous nuisance coordinates $z_n$. The product metric $d_{\mathcal{K}} \oplus G_n$ (Definition 2.2.1) and the Sasaki-like warped metric ({ref}`Section 7.11.3 <sec-the-induced-riemannian-geometry>`) were heuristic constructions that treat the discrete and continuous components separately. These constructions are constrained by the agent's {prf:ref}`def-boundary-markov-blanket`.

(rb-distribution-shift)=
:::{admonition} Researcher Bridge: Handling Distribution Shift
:class: info
Standard Bayesian filters fail during "surprises" because they can't handle mass appearing or disappearing (Unbalanced Transport). The **Wasserstein-Fisher-Rao (WFR)** metric allows the agent's belief to both **flow** (smooth tracking) and **jump** (teleporting probability mass). This provides a unified variational principle for both continuous state-tracking and discrete hypothesis-switching.
:::

This section introduces the **Wasserstein-Fisher-Rao (WFR)** metric---also known as **Hellinger-Kantorovich** {cite}`chizat2018unbalanced,liero2018optimal`---which provides a rigorous, unified variational principle. The key insight is to treat the agent's internal state not as a *point* in $\mathcal{Z}$, but as a *measure* (belief state) $\rho_s \in \mathcal{M}^+(\mathcal{Z})$ evolving on the bundle.

(sec-motivation-the-failure-of-product-metrics)=
## Motivation: The Failure of Product Metrics

:::{div} feynman-prose
Before we dive into the solution, let's make sure we really understand the problem. Why doesn't the obvious approach work?

The obvious approach is what I call the "Sasaki-like construction"---you take the metric on the discrete part, you take the metric on the continuous part, and you combine them. It's like saying: the distance between two states is the discrete hop distance plus the continuous Riemannian distance.

This seems reasonable, but watch what happens when things get interesting.
:::

**The Problem with Sasaki-like Constructions.**

The metric tensor from {ref}`Section 7.11.3 <sec-the-induced-riemannian-geometry>` (where $\rho_{\text{depth}}$ denotes resolution depth, not density):

$$
ds^2 = d\rho_{\text{depth}}^2 + d\sigma_{\mathcal{K}}^2 + e^{-2\rho_{\text{depth}}}\|dz_n\|^2

$$
assumes a fixed point moving through the bundle. This creates two problems:

1. **Discontinuous Jumps:** When the agent transitions from chart $K_i$ to chart $K_j$, the metric provides no principled way to measure the "cost" of the jump versus continuous motion along an overlap.

2. **No Mass Conservation:** A point either is or isn't at a location. But the agent's *belief* can be partially in multiple charts simultaneously (soft routing, {ref}`Section 7.8 <sec-tier-the-attentive-atlas>`).

:::{div} feynman-prose
Let me make this very concrete. Suppose the agent is tracking an object that suddenly moves behind an occluder. The belief distribution should smoothly transition from "I'm pretty sure where it is" to "it could be in several places." But if we're tracking a *point*, we have to decide: does the point stay where it was, or does it jump? Neither option is right---the situation calls for a distribution that spreads out.

Or consider this: the agent is 90% confident it's in scenario A and 10% confident it's in scenario B. What's the "position" of that belief? There isn't one! You need a distribution.
:::

:::{admonition} The Core Problem
:class: warning feynman-added
Think of a particle versus a probability cloud. A particle has to be *somewhere*---it can move, but it can't be in two places at once. A probability cloud can spread, concentrate, flow, and even split. The agent's belief is fundamentally a cloud, not a particle. Treating it as a particle forces artificial discretization: when do you "switch" hypotheses? The WFR framework says: you don't have to choose. Mass can continuously redistribute.
:::

**The WFR Solution.**

The Wasserstein-Fisher-Rao metric resolves both issues by lifting dynamics to the space of measures $\mathcal{M}^+(\mathcal{Z})$. In this space:
- **Transport (Wasserstein):** Probability mass moves along continuous coordinates via the continuity equation.
- **Reaction (Fisher-Rao):** Probability mass is created/annihilated locally, enabling discrete chart transitions.

The metric determines the optimal path by minimizing the total cost: transport cost $\int\|v\|_G^2\,d\rho$ plus reaction cost $\int\lambda^2|r|^2\,d\rho$.

:::{div} feynman-prose
Here's the beautiful idea. Instead of asking "where is the agent's belief *point*?", we ask "what is the agent's belief *distribution*?" And instead of asking "how does the point move?", we ask "how does the distribution evolve?"

This distribution can do two things: it can *flow* (mass moves from here to there while conserving total probability) or it can *react* (mass appears or disappears locally). The first is what happens when you track a moving object. The second is what happens when you suddenly realize "wait, I was wrong about which scenario I'm in."

The WFR metric gives us a principled way to measure the "cost" of any combination of flowing and reacting. For
fixed endpoints, metric, and reaction scale, the Benamou--Brenier formulation is convex in the appropriate flux
variables such as $(\rho,\rho v,\rho r)$. That removes a combinatorial search from the variational problem, but the
choice of model, chart maps, and $\lambda$ remains a modelling decision.
:::

(sec-the-wfr-metric)=
## The WFR Metric (Benamou-Brenier Formulation)

:::{div} feynman-prose
Now let's get precise. The Benamou-Brenier formulation is a beautiful way to think about optimal transport: instead of asking "what's the cheapest way to rearrange mass from configuration A to configuration B?", you ask "what's the most efficient *process* that transforms A into B over time?"

It's like the difference between asking "what's the shortest path between two cities?" and asking "what's the most fuel-efficient way to drive between them, considering traffic and terrain?" The second question embeds the problem in time and lets you think about dynamics.
:::

Let $\rho(s, z)$ be a time-varying density on the latent bundle $\mathcal{Z}$, evolving in computation time $s$. The WFR distance is defined by the minimal action of a generalized continuity equation.

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

:::{div} feynman-prose
Let me unpack this piece by piece, because there's a lot going on.

The **unbalanced continuity equation** is the heart of the matter: $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$. On the left side, we have how the density changes with time ($\partial_s \rho$) plus how it flows due to velocity ($\nabla \cdot (\rho v)$). On the right side, we have the reaction term ($\rho r$)---if $r > 0$, mass is being created; if $r < 0$, mass is being destroyed.

In ordinary optimal transport, the right side is zero: mass is conserved, it just moves around. That's the "balanced" case. But we need the "unbalanced" case because when an agent switches hypotheses---goes from "I think it's scenario A" to "I think it's scenario B"---mass has to disappear from A and appear in B. That's not transport; that's reaction.

The **action functional** measures the total cost of a path. You pay for velocity (moving mass around) and you pay for reaction (creating or destroying mass). The parameter $\lambda$ sets the exchange rate: how much is one unit of transport worth compared to one unit of reaction?

And that **vector potential** term? It belongs to a separately driven action for situations where the reward
one-form has curl---where going around a loop need not return the same value. It is a linear bias on candidate paths,
not part of the squared WFR distance defined above. Only its curvature $\mathcal{F}=d\mathbf A$ is gauge invariant;
interpreting the bias as reward requires the declared field, units, and control reduction.
:::

:::{admonition} Example: Belief Update as WFR Flow
:class: feynman-added example

Imagine a robot tracking a ball. Initially, the belief is concentrated near position $x_0$. Then the ball moves quickly to $x_1$. What happens to the belief?

**Pure transport ($r = 0$):** The belief distribution flows smoothly from $x_0$ to $x_1$. This is what happens during normal tracking when the ball moves predictably.

**Pure reaction ($v = 0$):** The belief at $x_0$ shrinks while belief at $x_1$ grows. This is what happens during a "surprise"---the ball teleports (occlusion, fast motion), and rather than flowing smoothly, the belief essentially jumps.

**Mixed:** Usually both happen. The belief flows toward where you expect the ball to go, but also mass is transferred to alternative hypotheses ("maybe it bounced off something I didn't see").

The WFR metric finds the optimal mix. If $x_1$ is close to $x_0$, transport dominates (just track it). If $x_1$ is far away, reaction dominates (teleport the belief).
:::

(pi-wfr-metric)=
::::{admonition} Physics Isomorphism: Wasserstein-Fisher-Rao Geometry
:class: note

**In Physics:** The Wasserstein-Fisher-Rao (WFR) or Hellinger--Kantorovich metric is a distinguished cone-space/Benamou--Brenier construction that combines optimal transport with Hellinger reaction geometry. It is one member of a broader family of unbalanced transport metrics {cite}`liero2018optimal,chizat2018interpolating`.

**In Implementation:** The finite non-negative belief measure $\rho$ evolves under the WFR metric on $\mathcal{M}^+(\mathcal{Z})$:

$$
d_{\text{WFR}}^2(\rho_0, \rho_1) = \inf_{\rho, v, r} \int_0^1 \int_{\mathcal{Z}} \left( \|v\|_G^2 + \lambda^2 r^2 \right) \rho \, d\mu_G \, ds

$$
**Correspondence Table:**
| Optimal Transport | Agent (Belief Dynamics) |
|:------------------|:------------------------|
| Wasserstein distance $W_2$ | Transport cost for belief |
| Fisher-Rao distance | Information cost for reweighting |
| Transport velocity $v$ | Belief flow in $\mathcal{Z}$ |
| Reaction rate $r$ | Mass creation/annihilation |
| Benamou-Brenier formula | Dynamic formulation |
| Geodesic interpolation | Optimal belief transition |

**Significance:** WFR unifies transport and reaction on the cone of non-negative finite measures $\mathcal M^+(\mathcal Z)$. The pure-reaction restriction is the scaled Hellinger geometry on measures; its tangent tensor restricts to Fisher--Rao after fixing total mass, while a finite simplex requires an additional discrete-state restriction.
::::

:::{prf:remark} Units
:label: rem-units

$[v] = \text{length}/\text{time}$, $[r] = 1/\text{time}$, and $[\lambda] = \text{length}$ after taking the metric coordinates as length units. The ratio $\|v\|/(\lambda |r|)$ is a local cost ratio; the exact crossover for a pair of Dirac masses depends on the normalization of the Hellinger term.

:::

:::{div} feynman-prose
The units tell you something important. Velocity has units of length per time---that's obvious. Reaction rate has units of inverse time---it's a growth rate, like an interest rate. And $\lambda$, the crossover parameter, has units of length.

So how do we compare transport and reaction? The ratio $\|v\|/(\lambda |r|)$ is a useful local cost
indicator when both fields and their units have been fixed. It is not a universal decision rule: the pairwise
crossover for two measures also depends on mass normalization and the convention used for the Hellinger term.

The picture of "walking" versus "teleporting" is therefore a way to inspect a calibrated model. The variational
problem, rather than the slogan, decides which admissible combination is cheaper.
:::

(sec-transport-vs-reaction-components)=
## Transport vs. Reaction Components

:::{div} feynman-prose
Now let's look at the two mechanisms separately before understanding how they combine.
:::

The belief state $\rho_s$ evolves on the bundle $\mathcal{Z}$ via two mechanisms.

**1. Transport (Wasserstein Component):**
The density evolves via the continuity equation $\partial_s\rho + \nabla\cdot(\rho v) = 0$ along the continuous coordinates $z_n$. The transport cost is $\int \|v\|_G^2\, d\rho$. In the limit $r \to 0$, the dynamics reduce to the standard Wasserstein-2 ($W_2$) optimal transport on the Riemannian manifold.

**2. Reaction (Hellinger Component):**
The density undergoes local mass creation/annihilation via the source term $\rho r$. This corresponds to discrete chart transitions: mass decreases on Chart A ($r < 0$) and increases on Chart B ($r > 0$). The reaction cost is $\int \lambda^2|r|^2\, d\rho$. In the limit $v \to 0$, the induced distance is the scaled Hellinger geometry on $\mathcal M^+(\mathcal Z)$; its Fisher--Rao tensor appears only after restricting to fixed total mass (and, for a simplex, a finite chart register).

:::{div} feynman-prose
Here's a way to think about the difference.

**Transport** is like rearranging furniture in a room. You can slide the couch from here to there, but the couch is still the same couch, and the total amount of furniture is conserved. The cost depends on how far you move things and how heavy they are.

**Reaction** is like a chemical reaction. You put in reactants, you get out products. Mass isn't conserved locally---it appears and disappears. In our case, the "mass" is belief: probability assigned to different hypotheses.

Both mechanisms are doing something profound. Transport handles the question "how does my estimate of *position* change?" Reaction handles the question "how does my estimate of *what situation I'm in* change?" In a rich agent, both happen simultaneously.
:::

:::{admonition} The Two Extreme Cases
:class: feynman-added tip

| Limit | What dominates | Physical picture | Agent behavior |
|-------|---------------|------------------|----------------|
| $r \to 0$ | Transport | Incompressible fluid flow | Smooth tracking within a hypothesis |
| $v \to 0$ | Reaction | Chemical kinetics | Switching between hypotheses |
| General | Both | Compressible reactive flow | Tracking with hypothesis revision |

Most interesting agent behavior lives in the "general" regime. A robot tracking an object while considering alternative interpretations is doing both transport and reaction.
:::

**3. The Coupling Constant $\lambda$ (Reaction-Transport Crossover Scale):**

This parameter sets the characteristic length scale in the action. It does not by itself put the pairwise crossover at $\|z_A-z_B\|_G=\lambda$; the numerical threshold depends on the mass normalization and, for Dirac endpoints, on the cone-angle convention.

**Operational interpretation:** choose $\lambda$ as a declared WFR hyperparameter. It may be calibrated against a router-overlap scale, but the atlas overlap is not an intrinsic radius and is not identical to $\lambda$ by definition.

:::{div} feynman-prose
The parameter $\lambda$ sets the relative price of transport and reaction in the chosen action. The road and
airline picture is useful, but the crossover is determined by the full variational cost, mass normalization, and cone
convention, not by a universal distance equal to $\lambda$.

A router-overlap scale can help calibrate $\lambda$, but the two quantities are not identical by definition. Reaction
is a local source term on the measure cone, so a transition need not be a literal jump between two discrete charts; in
a finite chart register, the same mechanism can represent a change of hypothesis.
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

:::{div} feynman-prose
Why the injectivity radius? Because that's the scale at which the manifold's topology starts to matter. Within the injectivity radius, the space looks like flat Euclidean space---you can move in any direction without running into weird topological obstacles. Beyond the injectivity radius, paths might wrap around, and the geometry becomes nontrivial.

If you've never encountered the injectivity radius before, here's the intuition. Imagine you're on the surface of a sphere. From any point, you can draw geodesics (great circles) in all directions. How far can you go before some of those geodesics start meeting again? That distance is the injectivity radius. On a sphere of radius $R$, it's $\pi R$---the distance to the antipodal point.

In the agent's latent space, the injectivity radius tells you: how far can you transport mass along smooth geodesics before you have to worry about the discrete structure of the chart atlas?
:::

(sec-reconciling-discrete-and-continuous)=
## Reconciling Discrete and Continuous

:::{div} feynman-prose
Now comes the payoff. The WFR metric doesn't just let us handle discrete and continuous separately---it actually *unifies* them in a way that respects the structure of both.
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

*Proof sketch.* The cone-space representation lifts a pointwise measure to
$(z,2\lambda\sqrt{\rho})$ with the cone metric. With $r=0$ the projection is
the $W_2$ geodesic; with $v=0$ it is the Hellinger geodesic on
$\mathcal M^+(\mathcal Z)$. A finite Fisher--Rao simplex is the discrete
normalized special case. $\square$

:::

:::{div} feynman-prose
This is the useful picture: WFR contains transport and reaction in one variational problem. With $r=0$ it gives
the Wasserstein transport restriction; with $v=0$ it gives the Hellinger/Fisher--Rao-type cone geometry. A finite
Fisher--Rao simplex appears only after restricting to a finite normalized chart register. In a mixed regime the
minimizer can use both terms, subject to the stated endpoints and regularity assumptions.

The "cone-space representation" is a technical change of variables involving $\sqrt{\rho}$. It makes the local
formula easier to analyze, but it does not mean that every coordinate representation is linear or that supports move
smoothly in the ordinary Euclidean sense. The continuity is in the measure geometry.
:::

:::{admonition} Analogy: Highway vs. Airplane
:class: feynman-added note

Imagine you're in a landscape of cities connected by highways and airports.

- **Transport (Wasserstein):** Driving on highways. You can go anywhere, but it takes time proportional to distance.
- **Reaction (Fisher-Rao):** Taking a locally priced shortcut in weight, pictured as flying; in the full cone it can occur throughout the domain, while a finite chart register supplies the airport analogy.
- **WFR:** Finding the optimal combination. For short trips, drive. For long trips, drive to the nearest airport, fly, then drive from the destination airport.

The length scale $\lambda$ is like the crossover set by the relative price of driving and flying. In the actual
model it is a declared WFR parameter, possibly calibrated against chart geometry. Reaction is a local source term on
the whole measure cone; discrete "airports" are only an analogy for the finite chart-register specialization. The
variational problem chooses a combination only after the endpoints, metric, and admissible fields have been fixed.
:::

::::{admonition} Connection to RL #26: Distributional RL as a related limit
:class: note
:name: conn-rl-26
**The General Law (Fragile Agent):**
Belief states evolve on $\mathcal{M}^+(\mathcal{Z})$ via **Wasserstein-Fisher-Rao dynamics**:

$$
d^2_{\text{WFR}}(\rho_0, \rho_1) = \inf \int_0^1 \int_{\mathcal{Z}} \left( \|v_s\|_G^2 + \lambda^2 |r_s|^2 \right) d\rho_s\, ds

$$
subject to the unbalanced continuity equation $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$.

**The Degenerate Limit:**
Restrict to value distributions at single states (no spatial transport). Use Euclidean metric ($G \to I$).

**Related Standard-RL equation:**

$$
Z(s, a) \stackrel{D}{=} R + \gamma Z(S', A'), \quad Q(s,a) = \mathbb{E}[Z(s,a)]

$$
The distributional Bellman equation is a useful comparison, but it is not
derived from the reaction-only WFR action. C51, QR-DQN, and IQN add their own
projection or quantile objectives {cite}`bellemare2017c51,dabney2018qrdqn`.

**What the generalization offers:**
- **Unified transport-reaction**: WFR handles continuous flow (within charts) and discrete jumps (between charts) in one framework
- **Belief geometry**: The metric on $\mathcal{M}^+(\mathcal{Z})$ respects both $W_2$ (spatial) and Fisher-Rao (probabilistic)
- **Teleportation length**: $\lambda$ determines when transport beats reaction (Proposition {prf:ref}`prop-limiting-regimes`)
- **GKSL embedding**: Quantum-like master equations embed naturally ({ref}`Section 20.5 <sec-connection-to-gksl-master-equation>`)
::::

(sec-connection-to-gksl-master-equation)=
## Connection to GKSL / Master Equation ({ref}`Section 12.5 <sec-optional-operator-valued-belief-updates>`)

:::{div} feynman-prose
Now let me show you a connection that has rigorous mathematical foundations, but only after we restrict the
objects. A finite reversible classical master equation is a gradient flow in a discrete Wasserstein-type geometry.
A GKSL equation reduces to that setting when the density matrix stays diagonal and the Hamiltonian and jump operators
satisfy the stated hypotheses. That restricted classical limit is the connection; an arbitrary GKSL evolution is not
automatically a WFR gradient flow.
:::

The WFR framework connects rigorously to the GKSL (Lindblad) master equation via the **classical limit**. We state this precisely.

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

**Correspondence Table (Classical Limit):**

| GKSL Component (diagonal $\varrho$)                                                               | WFR Interpretation                                           |
|---------------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| $-i[H, \varrho]$ (Commutator)                                                                     | Vanishes when $H$ is diagonal in the belief basis            |
| $\sum_j \gamma_j(L_j \varrho L_j^\dagger - \frac{1}{2}\{L_j^\dagger L_j, \varrho\})$ (Dissipator) | Graph transport on $\mathcal K$ (Maas metric), distinct from pointwise WFR reaction |
| Probability conservation $\sum_k \dot{p}_k = 0$                                                   | Balanced reaction ($\int \rho r \, d\mu = 0$ globally)            |
| Jump operators $L_j$                                                                              | Transition kernels (where mass teleports to)                 |

:::{prf:remark} Full Quantum Case
:label: rem-full-quantum-wfr

For non-diagonal density matrices (quantum coherences), the appropriate geometric structure is the **quantum Wasserstein distance** of Carlen \& Maas {cite}`carlen2014wasserstein,carlen2017gradient`. The GKSL equation is the gradient flow of quantum relative entropy with respect to this metric. This framework handles coherences but is more complex than the classical WFR theory used here.

:::

:::{div} feynman-prose
Let me be precise about what is rigorous and what is not.

**Rigorous in the stated setting:** A finite reversible classical master equation is a gradient flow of relative
entropy in a discrete Wasserstein-type metric. Under the diagonal-density, diagonal-Hamiltonian, and
jump-preservation assumptions, the GKSL equation reduces to that master equation. This is a theorem, not a claim
about every Lindblad model.

**Rigorous but different:** For full quantum states with coherences, Carlen \& Maas constructed a quantum Wasserstein
framework in which suitable GKSL dynamics are gradient flows of quantum relative entropy. That space contains density
matrices, not classical probability measures. The continuous WFR results concern their own measure-valued variational
problem; they do not automatically convert a coherent GKSL evolution into the classical WFR problem.

**The practical upshot:** Use the classical WFR geometry when the belief state, rates, reversibility, and boundary
conditions meet its hypotheses. For coherent or nonreversible dynamics, retain the appropriate quantum or nonequilibrium
analysis instead of importing the WFR conclusion.
:::

:::{admonition} Why This Connection Matters
:class: feynman-added tip

The GKSL/Lindblad structure is useful when one actually has a quantum state and a GKSL generator. It then
supplies structural properties such as complete positivity, trace preservation for a trace-preserving generator, and
Markovian composition. A WFR implementation over classical densities does not inherit those guarantees merely by using
similar symbols. Its positivity, normalization, and Markov properties must be enforced or proved for the chosen
 discretization and update.
:::

(sec-the-unified-world-model)=
## The Unified World Model

:::{div} feynman-prose
Now let's see how this theory translates into something you can implement. A world-model component can expose a
velocity and a reaction rate in one interface, while a separate planner or loss trains those fields against a target.
Whether the macro predictor and micro dynamics share parameters is an architectural choice; the WFR definition does
not require one monolithic network.
:::

The WFR formulation enables a **single World Model** that predicts both transport and reaction, eliminating the need for separate "macro predictor" and "micro dynamics" modules.

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

:::{div} feynman-prose
Look at what this reference implementation exposes. The network outputs a velocity $v$ and a reaction rate $r$, and
a simple Euler step gives
- Position updates additively: $z' = z + v \cdot dt$
- Particle masses multiplicatively: $m' = m \cdot \exp(r \cdot dt)$

The multiplicative update is the local reaction law: $r>0$ grows a particle weight, $r<0$ shrinks it, and $r=0$
leaves that weight unchanged. It does not by itself conserve the total mass or implement the spatial divergence in the
continuity equation; normalization, boundary flux, and a discretized divergence must be supplied by the chosen solver.

A trained model may learn small reaction rates in predictable regions and larger rates near a chart transition, but
that behavior is a modelling hypothesis to validate. The code is an illustrative world-model component, not a proof
that the learned fields select the WFR minimizer or switch charts correctly.
:::

:::{admonition} Particle Filter Interpretation
:class: feynman-added note

You can use this implementation as a particle-filter analogy, provided you keep the distinction clear. Particles carry
weights and move with $v$; the local update $m' = m \cdot \exp(r \cdot dt)$ changes those weights. There is no literal
resampling in this code, and the particle approximation still needs normalization and a rule for representing the
spatial divergence.

The differentiable update makes end-to-end training possible for this component. It does not establish convergence to
the WFR geodesic, preservation of total probability, or consistency with a boundary-value problem; those are separate
numerical and analytic checks.
:::

(sec-scale-renormalization)=
## Scale Renormalization (Connection to {ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`)

:::{div} feynman-prose
Now here's something that makes physicists very happy: the WFR framework connects naturally to renormalization group ideas. If you're not familiar with the renormalization group, don't worry---the intuition is actually straightforward.

The idea is that physical systems often have structure at multiple scales. A turbulent fluid has large eddies containing smaller eddies containing even smaller eddies. An image has global composition, mid-level objects, and fine textures. And crucially, the "rules" at different scales might be different.

In our stacked TopoEncoder architecture, each layer corresponds to a different scale. The WFR metric applies at each scale, but with a scale-dependent parameter $\lambda$.
:::

For stacked TopoEncoders ({ref}`Section 7.12 <sec-stacked-topoencoders-deep-renormalization-group-flow>`), the WFR metric applies recursively with **scale-dependent coupling**.

Recall the WFR action:

$$
\mathcal{E}_{\mathrm{WFR}} = \int_0^1\!\int_{\mathcal Z} \left( \|v\|_G^2 + \lambda^2 |r|^2 \right) d\rho_s\,ds

$$
For a hierarchy of layers $\ell = 0, \ldots, L$:

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

:::{div} feynman-prose
This gives a useful way to discuss multi-scale representations, provided the measured residual scales support the
ordering. The schedule uses the cumulative factors $\Pi^{(\ell)}$, so a coarse layer is more transport-dominated or a
fine layer more reaction-dominated only when those factors and the metric normalization make it so. The words "IR" and
"UV" describe the intended interpretation; they do not prove an ordering for arbitrary encoders.

If the factors decrease with depth, a macro representation may carry a larger reaction scale and fine residuals a
smaller one. That can make coarse hypotheses harder to replace and fine details easier to revise. Check the learned
scales and the WFR residuals before assigning that physical picture to a particular model.
:::

:::{admonition} The Cosmological Constant Analogy
:class: feynman-added note

The correspondence with the cosmological constant $\Lambda$ is a controlled analogy. In general relativity, $\Lambda$
sets a baseline curvature. In this model, the same role can be assigned layer by layer only if the capacity-constrained
metric law is applied at each layer and the units and coupling have been declared.

Under that additional schedule, a smaller $\lambda^{(\ell)}$ corresponds to a larger nominal $\Lambda^{(\ell)}$
through $\Lambda^{(\ell)}\sim 1/(\lambda^{(\ell)})^2$. It does not by itself prove that the latent layer is more
curved, that reaction is preferred, or that the bulk and boundary have the stated ordering. Those are diagnostics to
check against the learned metric and admissible WFR paths.
:::

(sec-connection-to-einstein-equations)=
## Connection to Einstein Equations ({ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`)

:::{div} feynman-prose
We've been talking about the geometry of belief space as if it were fixed. Here is the careful version of the
connection. Varying the WFR action with the density, velocity, and reaction fields held fixed produces an auxiliary
stress tensor. It can be coupled to the metric-law equation only after an additional source identification and unit
conversion have been declared.

So the useful analogy with Einstein's equations is structural: one variational stress tensor can act as a source in a
separate metric equation. The WFR action alone does not determine the latent metric, and the metric-law theorem does
not automatically identify its Risk Tensor with this WFR tensor.
:::

The WFR dynamics provide an auxiliary variational stress tensor. The metric
law's $T_{ij}$ is the reward Risk Tensor in {ref}`Section 18
<sec-capacity-constrained-metric-law-geometry-from-interface-limits>`; the WFR
tensor is not that source unless an additional coupling and unit conversion
are declared.

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

*Proof sketch.* Vary $\mathcal{S}_{\mathrm{WFR}}$ with respect to $G^{ij}$ while holding
$(\rho,v,r)$ fixed. Use $\delta\|v\|_G^2=-v_i v_j\,\delta G^{ij}$ and
$\delta d\mu_G=-\tfrac12 G_{ij}\delta G^{ij}d\mu_G$, then collect terms to match
$\delta\mathcal{S}_{\mathrm{WFR}}=-\tfrac12\int T^{\mathrm{WFR}}_{ij}\delta G^{ij}d\mu_G\,ds$.
See {ref}`Appendix C <sec-appendix-c-wfr-stress-energy-tensor>` for the full derivation. Holding the belief measure fixed instead removes this pressure term and changes the mass variation, so the convention is part of the statement. $\square$

:::

:::{div} feynman-prose
Let me decode this. The auxiliary tensor $T_{ij}^{\mathrm{WFR}}$ records how belief mass, transport, and reaction enter the
variation of the selected WFR action. In relativity, a stress-energy tensor is a source in Einstein's equations; here the same
word describes a diagnostic with a different domain and derivation.

The result has a perfect-fluid-like algebraic form: a density times velocity-squared term and a pressure-like term. Calling it
"perfect fluid" describes that algebraic shape; it does not import relativistic matter dynamics.

The reaction contribution is worth noticing. High $r$ increases the pressure-like term in this auxiliary tensor. It can influence
curvature only after a separate metric-law coupling is declared and its hypotheses are checked; the WFR variation alone does not
make geometry curve.

What does this mean in practice? Regions of high belief dynamics can be flagged as carrying a larger auxiliary load. Whether they
become geometrically different from regions of certainty is a modeling or control response that must be measured, not a consequence
of this tensor alone.
:::

(pi-stress-energy)=
::::{admonition} Physics Isomorphism: Stress-Energy Tensor
:class: note

**In Physics:** The stress-energy tensor $T_{\mu\nu}$ is derived from the variation of the matter action with respect to the metric: $T_{\mu\nu} = -\frac{2}{\sqrt{-g}}\frac{\delta S_M}{\delta g^{\mu\nu}}$ {cite}`wald1984general`.

**In Implementation:** The auxiliary WFR tensor (Theorem {prf:ref}`thm-wfr-stress-energy-tensor-variational-form`) is:

$$
T^{\mathrm{WFR}}_{ij} = \rho v_i v_j + \frac{1}{2}\rho\left(\|v\|_G^2 + \lambda^2 r^2\right) G_{ij}

$$
derived from $\delta \mathcal{S}_{\text{WFR}}/\delta G^{ij}$ under the
density-fixed convention. It is an auxiliary perfect-fluid diagnostic, not the
Risk Tensor in the capacity metric law, with pressure $P = \frac{1}{2}\rho(\|v\|_G^2 + \lambda^2 r^2)$.

**Correspondence Table:**

| Field Theory | Agent (WFR) |
|:-------------|:------------|
| Matter density $\rho_m$ | Belief density $\rho$ |
| 4-velocity $u^\mu$ | Transport velocity $v^i$ |
| Pressure $p$ | Reaction pressure $\frac{\lambda^2}{2}\rho r^2$ |
| Rest mass density | WFR kinetic energy $\frac{1}{2}\rho\|v\|_G^2$ |
::::

**Implications:**
1. **High velocity ($v$):** Agent moves fast through a region, so the auxiliary stress contribution $T_{ij}$ can be
   large under the density-fixed convention. Any resulting curvature response or contraction is conditional on the
   selected metric-law coupling; the natural-gradient interpretation is a modeling interpretation, not a first-principles
   consequence of the WFR variation alone.

2. **High reaction ($r$):** Agent jumps frequently → $P_{\mathrm{react}}$ increases → capacity stress increases. This triggers the boundary-capacity constraint (Definition {prf:ref}`def-dpi-boundary-capacity-constraint`).

:::{div} feynman-prose
These implications deserve emphasis, with the convention in the theorem kept in view.

The variation says that, under the density-fixed convention, transport and reaction enter an auxiliary stress tensor and
that reaction contributes the nonnegative term $\tfrac12\lambda^2\rho r^2$ to its pressure component. It does not say
that high velocity has a universal curvature sign, that the latent space must contract, or that this tensor is already
the state-space natural-gradient metric.

A capacity response can be tested after a separate coupling to the metric law has been specified. High reaction may
then be a useful load indicator, but it is not by itself a proof of capacity saturation.
:::

**Consistency with existing losses:**

| Existing Loss                                    | WFR Interpretation                             | Status     |
|--------------------------------------------------|------------------------------------------------|------------|
| $\mathcal{L}_{\mathrm{pred}}$ (Prediction)       | Minimizing transport cost $\lVert v\rVert_G^2$ | Compatible |
| $\mathcal{L}_{\mathrm{closure}}$ (Macro closure) | Penalizing reaction $r$ in macro channel       | Compatible |
| Mass reaction                                     | $r<0$ annihilates belief mass; entropy monotonicity requires a separate gradient-flow hypothesis | Compatible |
| Capacity ($I < C$)                               | Metric curves to keep WFR path within budget   | Compatible |

:::{admonition} Why This Matters for Implementation
:class: feynman-added tip

The compatibility table is a map of possible correspondences, not an identity between losses. A prediction loss can
serve as a transport proxy only when its residual and units have been related to $\|v\|_G^2$. A closure loss can
penalize reaction only when the chart model defines reaction through that loss. Entropy monotonicity requires a separate
reversible or gradient-flow hypothesis.

Likewise, the capacity row does not say that WFR automatically curves the metric or enforces $I<C$. That requires the
capacity proxy, the metric-law coupling, and the boundary conditions to be specified. WFR can be added as a
variational layer or consistency regularizer; whether it replaces existing modules is an implementation choice to be
validated.
:::

(sec-comparison-sasaki-vs-wfr)=
## Comparison: Sasaki vs. WFR

:::{div} feynman-prose
Let me summarize the comparison between the old approach (Sasaki-like product metrics) and the new approach (WFR). This table tells the whole story.
:::

| Feature                     | Sasaki (Product Metric)          | WFR (Unbalanced Transport)             |
|-----------------------------|----------------------------------|----------------------------------------|
| **State representation**    | Fixed point                      | Probability mass / belief              |
| **Topology changes**        | Manual patching required         | Handled natively via $r$               |
| **Path type**               | "Walk then Jump" (discontinuous) | Smooth interpolation                   |
| **Optimization**            | Combinatorial + Gradient descent | Convex in $(\rho,\rho v,\rho r)$ with fixed endpoints and metric |
| **Theoretical consistency** | Ad-hoc construction              | Gradient flow under the stated reversibility hypotheses    |
| **Multi-scale**             | Separate metrics per scale       | Unified with scale-dependent $\lambda$ |

:::{div} feynman-prose
Every row in this table compares a different modelling choice. Let me highlight the useful distinction.

**Optimization**: A product construction may require an external search over chart sequences. For fixed WFR
endpoints, metric, and reaction scale, the flux formulation is convex and avoids that combinatorial enumeration. The
claim is about this constrained variational problem; learning the endpoints, chart maps, or metric is still generally
nonconvex.

**Path type**: WFR gives a continuous curve in the metric space of measures when the admissible action is finite. Its
transport and reaction components can trade off along that curve, but a coordinate trajectory or chart support need not
look smooth, and a numerical scheme can still introduce discontinuities.

**Theoretical consistency**: WFR is a specified variational metric with a well-defined cone construction under its
assumptions. Calling it "unique" without naming the class of metrics is too strong. Its usefulness comes from the
stated action, continuity equation, endpoints, and boundary conditions, not from a universal replacement theorem.
:::

(sec-implementation-wfr-consistency-loss)=
## Implementation: WFR Consistency Loss

:::{div} feynman-prose
Now let's get concrete about how to train models with this framework. The key idea is a **consistency loss** that
measures a time-discretized residual of the unbalanced continuity equation. It is a local diagnostic for the chosen
representation and discretization, not the WFR distance itself and not a substitute for solving the full boundary-value
problem.
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

:::{div} feynman-prose
Why work with $\sqrt{\rho}$ instead of $\rho$? This is the "cone-space" change of variables. It gives the local
identity used by the residual and often improves numerical conditioning, but it does not remove the need to define a
spatial discretization or boundary treatment.

The consistency loss compares the observed one-step change in $\sqrt{\rho}$ with the change predicted by the current
$(v,r)$ fields. A small value means those sampled transitions agree with the discretized equation. It does not prove
that the fields are a global WFR minimizer or that the density remains normalized.

The divergence term $\nabla \cdot (\rho v)$ is the delicate part computationally. Finite differences or automatic
differentiation can approximate it only after a grid, mesh, or differentiable spatial representation has been chosen;
the placeholder in the example intentionally omits that choice.
:::

:::{admonition} Implementation Notes
:class: feynman-added note

A few practical considerations:

1. **The $\epsilon$ stabilizer:** We add $\epsilon = 10^{-6}$ inside the square root to avoid division by zero when $\rho \approx 0$. This corresponds to a tiny uniform "background" belief.

2. **The divergence term:** The placeholder `div_rho_v = torch.zeros_like(rho_t)` in the code is a simplification. For a full implementation, you'd need to either:
   - Discretize the divergence using finite differences on a grid
   - Use automatic differentiation through a neural velocity field
   - Use a divergence-free parameterization and ignore this term

3. **Batching:** The loss is computed per-batch and averaged. You want to see it decrease during training, indicating that the world model's $(v, r)$ predictions are becoming more consistent with actual belief evolution.

4. **Scaling:** The loss magnitude depends on $dt$ and the scale of $\rho$. You may need to tune the loss weight relative to other training objectives.
:::

(sec-node-wfrcheck)=
## Auxiliary diagnostic: WFRCheck

:::{div} feynman-prose
Finally, we define a diagnostic node that monitors WFR consistency at runtime. This fits into the larger diagnostic framework described in Section 3.
:::

Following the diagnostic node convention ({ref}`Section 3.1 <sec-theory-thin-interfaces>`), we define:

| **ID**  | **Name**     | **Component**   | **Type**                 | **Interpretation**          | **Proxy**                    | **Cost** |
|--------|--------------|-----------------|--------------------------|-----------------------------|------------------------------|----------|
| **aux-WFR** | **WFRCheck** | **World Model** | **Dynamics Consistency** | Transport-Reaction balance? | $\mathcal{L}_{\mathrm{WFR}}$ | $O(BK)$  |

**Trigger conditions:**
- High $\mathcal{L}_{\mathrm{WFR}}$: World model's $(v, r)$ predictions violate continuity
- Remedy: Increase training on transitions; check for distribution shift

:::{div} feynman-prose
When should you worry about this diagnostic? A high WFRCheck residual means the sampled one-step prediction disagrees
with the discretized continuity equation. Insufficient training and distribution shift are two possibilities, but so are
an inaccurate divergence approximation, a mismatched time step, boundary leakage, or an inconsistent density convention.

Treat the number as a warning and inspect those choices before changing the model. A low residual supports local
one-step consistency for the samples and discretization used; it does not certify a global WFR geodesic, normalization,
or convergence of the learned world model.
:::

:::{admonition} Summary: What WFR Buys You
:class: feynman-added tip

Let me summarize the useful benefits of the WFR framework:

1. **One variational language:** Transport and reaction can be priced together for fixed endpoints, metric, and
   admissible fields. An implementation may still use separate modules for convenience.

2. **Declared jump cost:** The parameter $\lambda$ sets the relative scale of reaction and transport. It can be
   calibrated from geometry, but it is not derived from geometry without an additional calibration argument.

3. **Convex subproblem:** The flux formulation is convex for the fixed-data variational problem. Training a neural
   world model and learning a metric need not be convex.

4. **Careful physical analogies:** Fluid, thermodynamic, and quantum language can suggest useful constructions, while
   the corresponding physical identifications require their own hypotheses.

5. **A runtime residual:** WFRCheck monitors the selected discretization and samples. It is a diagnostic, not a proof of
   global consistency.

The core picture remains simple: let belief flow and react, charge for both, and check that the numerical path obeys
the continuity equation and the declared boundary conditions.
:::
