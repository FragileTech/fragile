(sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics)=
# Computational Metabolism: The Landauer Bound and Deliberation Dynamics

## TLDR

- Compute is not free: belief updates dissipate energy. This chapter gives a thermodynamic model of **deliberation
  dynamics**.
- Use Landauer’s principle to derive a **cost functional for computation time** and an optimal stopping condition:
  stop thinking when marginal value gain equals marginal metabolic cost.
- This yields an operational “fast vs. slow” regime switch (reflex versus deliberation) in compute allocation.
- Practical implication: track and regulate compute as part of control, not as an external budget afterthought.
- Connects metabolism to stability: excessive deliberation can be as harmful as insufficient compute.

## Roadmap

1. Thermodynamic framing (Landauer cost for information updates).
2. Derive the dual-horizon action and optimal compute allocation.
3. Operational regimes, crossover intuition, and implementation guidance.

*Abstract.* We establish a thermodynamic foundation for internal inference by coupling computation time $s$ to an
energetic cost functional. We model the agent as an open system where belief updates are dissipative processes. By
applying the conditional entropy-dissipation permit below to the Wasserstein-Fisher-Rao (WFR) flow, we show that the
optimal allocation of computation time $S^*$ emerges from the stationarity of a **Dual-Horizon Action**. The reflexive
(fast) and deliberative (slow) regimes form an operational crossover governed by the ratio of the task-gradient norm to
the metabolic dissipation rate; a thermodynamic phase transition would require an additional limiting argument.

(rb-thinking-fast-slow)=
:::{admonition} Researcher Bridge: Principled "Thinking Fast and Slow"
:class: info
Most agents spend the same amount of FLOPs on a trivial decision as a critical one. We use the **Landauer Bound** to assign a thermodynamic cost to information updates. The agent stops "deliberating" ($S^*$) exactly when the marginal gain in Value is outweighed by the metabolic cost of more compute. This derives "System 1 vs System 2" behavior from first principles.
:::

*Cross-references:* This section extends the WFR dynamics
({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) to account for the
thermodynamic cost of belief updates, building on the cognitive temperature framework
({ref}`sec-the-geodesic-baoab-integrator`) and the value potential
({ref}`sec-the-reward-field-value-forms-and-hodge-geometry`).

*Literature:* Landauer's principle {cite}`landauer1961irreversibility`; thermodynamics of computation
{cite}`bennett1982thermodynamics`; thermodynamics of information {cite}`parrondo2015thermodynamics`; dual-process theory
{cite}`kahneman2011thinking`; free energy principle {cite}`friston2010free`; information geometry
{cite}`amari2016information`.

:::{div} feynman-prose
Now here is a question that I think is absolutely fundamental, and yet most people building intelligent systems never even ask it: **How long should you think before you act?**

You see, in most of our theories about intelligent agents, we treat thinking as if it were free. The agent can compute for as long as it wants, refine its beliefs to arbitrary precision, and only then decide what to do. But that is not how the real world works. Thinking costs something. Every bit of computation burns energy. Every moment spent deliberating is a moment you are not acting, and the world keeps changing around you.

So there may be a sweet spot, some duration of thought where the modeled benefit of thinking more is balanced by the modeled cost
of additional computation. Whether an optimum exists, and whether it is interior, depends on the regularity, calibration, and
horizon assumptions of the chosen dynamics. Thermodynamics supplies an analogy and a conditional estimate; it does not by itself
guarantee an optimal stopping time.

The key insight comes from Landauer's Principle, which gives a minimum heat cost for logically irreversible erasure in an ideal
thermal setting. A belief update is not automatically a physical bit erasure. To use the analogy quantitatively, the WFR cost,
entropy convention, hardware, and temperature scale must be calibrated.

And here is the useful part: once a calibrated computation cost has been chosen, the question of "when to stop thinking" becomes a
variational problem. An interior minimizer satisfies a stationarity condition when the required differentiability and boundary
conditions hold. The resulting fast/slow split is a model-dependent allocation crossover, not automatically a phase transition
of a physical system.
:::



(sec-the-energetics-of-information-updates)=
## The Energetics of Information Updates

We begin by mapping the abstract WFR belief dynamics ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`) {cite}`chizat2018unbalanced,liero2018optimal` to physical dissipation via Landauer's Principle.

:::{div} feynman-prose
Before we dive into the formalism, let me explain what we are doing here in physical terms.

Imagine the agent's belief as a cloud of probability distributed over its latent space. When the agent thinks---when it processes information and updates its beliefs---this cloud moves and reshapes. Some probability mass flows from one region to another (that is the transport part). Some mass might appear or disappear (that is the reaction part, for when hypotheses are created or abandoned).

Now, all of this motion has a cost in the selected WFR model. The question is: when may that cost be interpreted as energy?

The metabolic flux we are about to define is an instantaneous dissipation proxy. Think of it as the agent's "caloric burn rate for
thinking" only after the coefficient and units have been calibrated. It has two components: one for moving probability around
(like dragging a weight across a floor), and one for creating or destroying probability mass (like building or demolishing a
house). The WFR geometry measures the selected costs; it does not by itself identify them with physical heat.
:::

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

:::{div} feynman-prose
Let me unpack what this definition is really saying.

The metabolic flux $\dot{\mathcal{M}}$ is an integral over all of latent space, weighted by the belief density $\rho$. This
weighting is crucial: the model charges update cost where probability mass is present. If some region of belief space is empty, the
integral assigns no cost there. Calling that quantity physical energy still requires the calibration of $\sigma_{\text{met}}$.

The two terms inside the integral are the transport cost and the reaction cost:

1. **Transport cost** $\|v\|_G^2$: This is the squared velocity, measured in the Riemannian metric $G$. The metric matters! Moving in directions the geometry says are "expensive" costs more than moving in "cheap" directions. This is like pushing a cart---it is easier to push it on a smooth floor than uphill through mud.

2. **Reaction cost** $\lambda^2 |r|^2$: This is the squared rate of mass creation or destruction, scaled by $\lambda^2$. Remember, $\lambda$ is the length scale where transport and reaction costs balance. If $\lambda$ is large, reactions are expensive relative to transport; if small, reactions are cheap.

The coefficient $\sigma_{\text{met}}$ is the "metabolic resistance." It can convert the abstract WFR kinetic quantity into
chosen energy units only after a hardware or simulator calibration. A high value then means that the modeled implementation pays
more per unit of belief update; it is not a universal biological constant.
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

*Proof.* Differentiating $H$, substituting the continuity equation, using the
no-flux condition and mass preservation, gives

$$
\frac{d}{ds}H = -\int_{\mathcal Z}\rho_s\langle\nabla\ln\rho_s,v_s\rangle_G\,d\mu_G
 -\int_{\mathcal Z}\rho_s r_s\ln\rho_s\,d\mu_G.
$$

Cauchy--Schwarz in $L^2(\rho_s d\mu_G)$ bounds the two terms by
$\sqrt{I_\rho E_v}$ and $\sqrt{J_\rho E_r}$, respectively. Their sum proves
the entropy-rate inequality, and the displayed calibration gives the final
Landauer-form inequality. This is an internal conditional estimate; the
classical Landauer principle is a physical analogy that requires its own
thermodynamic hypotheses. $\square$

*Remark (Landauer's Principle).* The classical Landauer bound states that erasing one bit of information requires dissipating at least $k_B T \ln 2$ joules of heat. The conditional theorem above becomes an information-geometric Landauer statement only when its calibration and normalization hypotheses hold.

:::

:::{div} feynman-prose
This estimate is useful, but its scope matters. I want to make sure you see exactly what has been shown.

Landauer's classical result concerns logically irreversible erasure of a physical bit coupled to a thermal reservoir. Under those
assumptions, resetting a bit requires at least $k_B T\ln 2$ of dissipated heat. A continuous belief entropy and the cognitive
scale $T_c$ are different mathematical objects until an interface and calibration identify them.

The calculation here starts from the WFR continuity equation. Differentiating $H(\rho_s)$ and integrating the transport term by
parts gives the displayed expression when the density and fields have the required regularity and the boundary flux vanishes. The
Cauchy--Schwarz step is valid under the corresponding integrability assumptions. Turning it into
$\dot{\mathcal M}\geq T_c|\dot H|$ additionally needs the stated gradient-flow or transport calibration, the analogous reaction
estimate, and consistent normalization of the metabolic coefficient.

Here is the intuitive picture. Your belief starts spread out (high entropy, uncertainty). As you think and process information, your
belief may concentrate (low entropy, certainty). In the calibrated model, that entropy change is paired with a WFR dissipation
budget. Without the calibration and no-flux hypotheses, concentration alone does not give a universal energy lower bound.

$T_c$ plays the role of a conversion factor in the selected gradient-flow model. At high cognitive temperature, the agent may explore
more freely; at low temperature, it may exploit what it knows. The statement that a given entropy reduction costs more at higher
$T_c$ is therefore conditional on this model and calibration, rather than a direct identification with a physical heat bath.
:::

:::{admonition} Example: The Cost of Certainty
:class: feynman-added tip

Suppose the agent starts with a uniform belief over 100 possible states (entropy $H = \ln 100 \approx 4.6$ nats) and wants to narrow down to just 10 possible states (entropy $H = \ln 10 \approx 2.3$ nats). If the regularity, no-flux, gradient-flow, reaction, and calibration hypotheses of the conditional estimate hold along this path, then:

The entropy reduction is $\Delta H \approx 2.3$ nats. The model's integrated dissipation estimate is then:

$$
\Psi_{\text{met}} \ge T_c \cdot |\Delta H| = 2.3 \, T_c \text{ nats}

$$

If $T_c = 1$, that is about 2.3 model energy units in the declared nat normalization. If $T_c = 0.1$ (a more "decisive" agent),
the estimate drops to 0.23 units. These numbers illustrate the calibrated inequality; they are not a hardware-independent energy
prediction.

The modeling tradeoff is clear: under the selected calibration, certainty has a dissipation price that depends on how "hot" the
thinking process is. The classical Landauer analogy should not be used beyond those hypotheses.
:::

(pi-landauer-principle)=
::::{admonition} Physics Isomorphism: Landauer's Principle
:class: note

**In Physics:** Erasing one bit of information requires dissipating at least $k_B T \ln 2$ joules of heat. More generally, reducing entropy by $\Delta S$ requires work $W \geq T|\Delta S|$ {cite}`landauer1961irreversibility,bennett1982thermodynamics`.

**In Implementation:** The generalized Landauer bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

$$
\dot{\mathcal{M}}(s) \geq T_c \left|\frac{d}{ds} H(\rho_s)\right|

$$
**Correspondence Table:**

| Thermodynamics | Agent (Metabolic) |
|:---------------|:------------------|
| Temperature $T$ | Cognitive temperature $T_c$ |
| Heat dissipation $\dot{Q}$ | Metabolic flux $\dot{\mathcal{M}}$ |
| Entropy $S$ | Belief entropy $H(\rho)$ |
| Boltzmann constant $k_B$ | 1 (nat units) |
| Work $W$ | Cumulative metabolic cost $\Psi_{\text{met}}$ |

**Consequence:** Thinking has irreducible thermodynamic cost. Deliberation stops when marginal value gain equals metabolic cost.
::::

:::{admonition} Connection to RL #14: Maximum Expected Utility as Zero-Temperature Limit
:class: note
:name: conn-rl-14
**The General Law (Fragile Agent):**
The agent optimizes a **Free Energy** objective that includes the metabolic cost of computation:

$$
\mathcal{F}[p, \pi] = \int_{\mathcal{Z}} p(z) \Big( V(z) - T_c H(\pi(\cdot|z)) \Big) d\mu_G - \Psi_{\text{met}}

$$
where $\Psi_{\text{met}} = \int_0^S \dot{\mathcal{M}}(s)\,ds$ is the cumulative metabolic energy. The agent stops thinking when marginal returns equal marginal costs.

**The Degenerate Limit:**
Set $T_c \to 0$ (computational temperature zero). Compute is free and infinite.

**The Special Case (Standard RL):**

$$
J(\pi) = \max_\pi \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r_t\right]

$$
This recovers standard **Maximum Expected Utility**---the objective used in DQN, PPO, SAC, etc.

**Result:** Standard RL ignores the thermodynamic cost of inference. The agent assumes it has infinite compute and can think forever. The Fragile Agent has an irreducible "cost of thinking" governed by the Landauer bound.

**What the generalization offers:**
- Principled stopping: deliberation ends when $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$ (marginal return = marginal cost)
- Fast/Slow phase transition: System 1 ($S^*=0$) vs System 2 ($S^*>0$) from first principles
- Landauer bound: $\dot{\mathcal{M}} \ge T_c |\dot{H}|$---thinking has irreducible thermodynamic cost
:::



(sec-the-metabolic-potential-and-deliberation-action)=
## The Metabolic Potential and Deliberation Action

We introduce the metabolic cost as a coordinate in the agent's extended state space.

:::{div} feynman-prose
Now we come to the central construction of this section. We have specified a modeled computation cost. The next question is: how
does the agent decide when to stop?

The answer is useful in its simplicity. We define an **action**---in the physicist's sense, not the agent's sense---that captures
the tradeoff between value gained and modeled dissipation. If a minimizer exists, the agent selects the computation time $S^*$ that
minimizes this action.

Think of it like this. Suppose you are trying to decide where to eat dinner. You could think about it for one second and pick
something adequate. Or you could spend an hour researching restaurants and find something excellent. But at some point, the
improvement in your dinner is not worth the additional time spent deciding. The Deliberation Action formalizes this tradeoff once
the value and dissipation terms have been specified.
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

:::{div} feynman-prose
Let me explain why we call this the "Dual-Horizon Action."

In ordinary reinforcement learning, there is one horizon: the time horizon over which the agent collects rewards in the world. The discount factor $\gamma$ controls how far into the future the agent looks.

But the Fragile Agent has a second horizon: the **computation horizon**. This is the internal time $s$ over which the agent thinks before acting. And just as the external horizon has a "discount" in the form of $\gamma$, the internal horizon has a "discount" in the form of metabolic cost.

The Deliberation Action captures both:
- **Expected Terminal Value** $\langle V \rangle_{\rho_S}$: This is what the agent expects to get from acting after thinking for time $S$. As $S$ increases, the belief $\rho_S$ concentrates on high-value regions, so this term increases (the minus sign means decreasing action, which is good).
- **Metabolic Cost** $\Psi_{\text{met}}(S)$: This is what the agent pays to think for time $S$. It increases with $S$.

The optimal $S^*$ is where these two competing effects balance. Think too little, and you leave value on the table. Think too much, and you waste energy chasing diminishing returns.
:::

:::{note}
:class: feynman-added

**Why "Action" and not "Loss"?**

In physics, the action is a functional whose stationary points give the equations of motion. This is the Principle of Least Action, one of the most powerful ideas in all of physics.

By framing deliberation as an action, we borrow variational calculus for a one-dimensional resource-allocation problem. An
interior $S^*$ can be characterized by stationarity when differentiability and boundary conditions permit it. This does not make
the agent a classical or quantum mechanical system, and it does not rule out a boundary optimum or a numerical optimization method.

The action formulation gives access to the calculus needed for the stated objective. Broader tools such as Euler--Lagrange or
Hamilton--Jacobi theory apply only when their hypotheses and the relevant dynamical structure have been supplied; no full Lagrangian
mechanics of cognition follows from the notation alone.
:::



(sec-optimal-deliberation-the-fast-slow-law)=
## Optimal Deliberation: The Fast/Slow Law

We now prove the existence of an optimal "stopping time" for internal thought.

:::{div} feynman-prose
This is where the bookkeeping becomes useful. Under the regularity, boundary, and calibration assumptions of the variational
problem, an interior minimizer satisfies a precise marginal-gain condition. Comparing the initial marginal gain with the initial
cost can then define fast and slow allocation regimes for this model.

That comparison may help organize the empirical distinction between fast and slow thinking, but it does not derive a universal
psychological transition or an exact physical phase boundary from first principles.
:::

:::{prf:theorem} Deliberation Optimality Condition
:label: thm-deliberation-optimality-condition

Let $\rho_s$ evolve as a gradient flow of $V$ under WFR dynamics. The optimal computation budget $S^*$ satisfies:

$$
\left. \frac{d}{ds} \langle V \rangle_{\rho_s} \right|_{s=S^*} = \dot{\mathcal{M}}(S^*),

$$
provided such an $S^*$ exists in $(0, S_{\max})$.

*Proof.* We seek to extremize $\mathcal{S}_{\text{delib}}$ with respect to the upper integration limit $S$. By the Leibniz Integral Rule and the definition of $\Psi_{\text{met}}$:

$$
\frac{d}{dS} \mathcal{S}_{\text{delib}} = -\frac{d}{dS} \langle V \rangle_{\rho_S} + \dot{\mathcal{M}}(S).

$$
The first term is the **Value-Improvement Rate**:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V(z) \partial_s \rho(S, z) \, d\mu_G.

$$
Applying the WFR continuity equation $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} V \cdot \rho r \, d\mu_G + \int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G.

$$
Integrating the divergence term by parts (assuming vanishing flux at $\partial\mathcal{Z}$):

$$
\int_{\mathcal{Z}} V (-\nabla \cdot (\rho v)) \, d\mu_G = \int_{\mathcal{Z}} \rho \langle \nabla_A V, v \rangle_G \, d\mu_G.

$$
For gradient flow dynamics, $v = -G^{-1} \nabla_A V$ (up to temperature scaling), so $\langle \nabla_A V, v \rangle_G = -\|\nabla_A V\|_G^2 \le 0$. Thus:

$$
\frac{d}{dS} \langle V \rangle_{\rho_S} = \int_{\mathcal{Z}} \rho \left( V r - \|\nabla_A V\|_G^2 \right) d\mu_G.

$$
The stationarity condition $\frac{d}{dS} \mathcal{S}_{\text{delib}} = 0$ yields the optimality condition. See {ref}`sec-appendix-e-rigorous-proof-sketches-for-ontological-and-metabolic-laws` for the full proof using the WFR adjoint operator. $\square$

*Physical interpretation:* The optimal stopping time $S^*$ is reached when the marginal gain in expected value (the "return on thinking") exactly equals the marginal metabolic cost (the "price of thinking"). At $S^*$, the agent has extracted all cost-effective information from deliberation.

:::

:::{div} feynman-prose
This optimality condition is wonderfully intuitive once you see it, provided we keep its hypotheses in view. If the calibrated
variational problem has an interior minimizer and the relevant derivatives exist, first-order stationarity says that the marginal
modeled value gain equals the marginal modeled dissipation. An optimum at the boundary need not satisfy this equality.

Let me give you an analogy. Imagine you are mining gold. Each hour of digging costs you some amount in effort (the modeled
dissipation), and each hour produces some amount of gold (the modeled value improvement). If the easy gold is extracted first, the
return per hour falls. At some point the gold you expect from another hour is worth no more than the effort of digging. That is the
allocation rule represented by the equation $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$.

Now, here is what makes this subtle. The value improvement rate
$\Gamma(s) = |\frac{d}{ds}\langle V \rangle|$ need not decay for an arbitrary task or flow; diminishing returns is an additional
modeling or empirical assumption. Likewise, $\dot{\mathcal{M}}(s)$ may be constant, increasing, or otherwise shaped by the chosen
cost and dynamics.

The intersection of these curves determines an interior $S^*$ only when continuity and the needed monotonicity, convexity, and
horizon conditions make that conclusion valid. Without them, the minimizer may be at $S=0$ or $S=S_{\max}$, or there may be
multiple stationary points.
:::

:::{prf:theorem} Fast/Slow Phase Transition
:label: thm-fast-slow-phase-transition

Let $\Gamma(s) := \left| \frac{d}{ds} \langle V \rangle_{\rho_s} \right|$ be the **Value-Improvement Rate**. There exists a critical threshold such that:

1. **Reflexive Regime (Fast):** If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $S^* = 0$. The agent executes an immediate action based on the prior $\rho_0$.

2. **Deliberative Regime (Slow):** If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $S^* > 0$. The agent enters a planning state, terminating only when the marginal gain in Value equals the marginal metabolic cost.

*Proof.* Consider the derivative of the Deliberation Action at $S = 0$:

$$
\left. \frac{d}{dS} \mathcal{S}_{\text{delib}} \right|_{S=0} = -\Gamma(0) + \dot{\mathcal{M}}(0).

$$
If $\Gamma(0) < \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} > 0$. Since $\mathcal{S}_{\text{delib}}$ is increasing at $S=0$ and we assume $\mathcal{S}_{\text{delib}}$ is convex (which holds when $\Gamma(s)$ is decreasing due to diminishing returns), the minimum occurs at the boundary $S^* = 0$.

If $\Gamma(0) > \dot{\mathcal{M}}(0)$, then $\frac{d}{dS} \mathcal{S}_{\text{delib}}|_{S=0} < 0$. The agent benefits from deliberation. As $s$ increases, $\Gamma(s)$ decreases (diminishing marginal returns on thinking) while $\dot{\mathcal{M}}(s)$ may increase or remain constant. The optimum $S^* > 0$ occurs when the curves cross: $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$. $\square$

*Remark (Dual-Process Theory).* Theorem {prf:ref}`thm-fast-slow-phase-transition` provides a first-principles derivation of Kahneman's "System 1 / System 2" dichotomy {cite}`kahneman2011thinking`. System 1 (reflexive) corresponds to $S^* = 0$; System 2 (deliberative) corresponds to $S^* > 0$. The transition is not a cognitive style but a phase transition governed by the ratio $\Gamma(0) / \dot{\mathcal{M}}(0)$.

:::

:::{div} feynman-prose
This is a useful classification, provided we read it as a statement about the selected optimization model. Under the theorem's
regularity and diminishing-returns assumptions, the initial comparison
$\Gamma(0)\mathrel{\lessgtr}\dot{\mathcal{M}}(0)$ identifies whether the modeled optimum is at the fast boundary or in a
deliberative region. It does not establish two physical phases of cognition.

The labels System 1 and System 2 are a helpful empirical analogy. A familiar task may have a small initial modeled gain, while a
novel task may have a larger one, but those statements require task-specific value estimates and a calibrated cost. The theorem
does not infer a human psychological mechanism from the labels.

The same modeled agent can therefore occupy different allocation regimes as the task, prior, horizon, or calibration changes. The
declared value and dissipation model supplies the decision rule; physics alone does not tell us which regime a real agent occupies.
:::

:::{admonition} Example: When to Think Fast vs. Slow
:class: feynman-added example

**Scenario 1: Catching a Ball**

You see a ball flying toward you. If the prior $\rho_0$ is already concentrated on the right action and the short horizon is
represented in the cost, the modeled value improvement from deliberation may be tiny---$\Gamma(0) \approx 0$. Under the theorem's
hypotheses, this gives $\Gamma(0) < \dot{\mathcal{M}}(0)$ and a reflexive allocation.

**Scenario 2: Buying a House**

You are considering a major purchase. If the prior $\rho_0$ is vague and the value model assigns substantial improvement to
examining omitted factors, then $\Gamma(0)$ may exceed the calibrated cost. Under the same hypotheses, the model favors
deliberation, perhaps over a longer horizon.

The same agent can therefore show different modeled behavior as the task and horizon change. The initial ratio is one input to that
decision; it does not by itself determine behavior independently of the stated value, cost, and regularity assumptions.
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

:::{div} feynman-prose
Now, here is a subtlety that most people miss.

Everything I said about stopping when marginal value equals marginal cost assumes the value field is **conservative**---meaning there is a single scalar value function $V(z)$, and moving around closed loops collects zero net reward.

But what if the value field has curl? What if there are cyclic preference structures, like rock-paper-scissors? Then a fixed point
need not describe the long-run behavior, and a recurrent orbit can be a more appropriate object. A nonzero curl by itself does not
prove that a limit cycle exists; that requires the additional dynamical assumptions behind the claimed steady state.

Does this mean the agent should think forever? No such conclusion follows. If the dynamics admit a stable orbit and its parameters
can be estimated, orbit stabilization can serve as an operational stopping diagnostic. The criterion changes because motion need not
cease, but existence of an optimal stopping time still requires the relevant objective and horizon assumptions.

Instead of waiting for the belief to stop moving, one may monitor whether an already observed orbit changes. A circulating belief can
have nearly constant shape and scale, but that does not establish that it is the best orbit or that further deliberation has no
value. Those are separate optimization claims.

This is a useful operational generalization: conservative and non-conservative models call for different diagnostics. The mathematics
must still supply the existence, stability, and calibration conditions before either diagnostic is treated as a theorem about a real
agent.
:::

:::{admonition} Connection to RL #15: UCB as Degenerate Thermodynamic VOI
:class: note
:name: conn-rl-15
**The General Law (Fragile Agent):**
The agent explores based on the **Thermodynamic Value of Information**:

$$
\text{VOI}(a) := \mathbb{E}[\Delta H(\rho) \mid a] - \frac{1}{T_c} \dot{\mathcal{M}}(a)

$$
Exploration is justified when the expected entropy reduction exceeds the metabolic cost.

**The Degenerate Limit:**
Assume a **single-state manifold** (no dynamics, stateless bandit). Use simplified Gaussian uncertainty.

**The Special Case (Multi-Armed Bandits):**

$$
a^* = \arg\max_a \left[ \hat{\mu}_a + c \sqrt{\frac{\ln t}{n_a}} \right]

$$
This recovers **UCB1 (Upper Confidence Bound)**. The exploration bonus $c\sqrt{\ln t / n_a}$ is the specific solution to the Landauer inequality for Gaussian arm distributions.

**Result:** UCB is the **thermodynamics of a single point**---exploration when there's no state, no dynamics, just uncertainty about arm means. The Fragile Agent generalizes to full manifold dynamics where exploration depends on local geometry.

**What the generalization offers:**
- State-dependent exploration: VOI varies with position $z$ on the manifold
- Geometric awareness: exploration bonus depends on local curvature $G(z)$
- Deliberation-aware: exploration trades off against computational cost $\dot{\mathcal{M}}$
:::



(sec-the-h-theorem-for-open-cognitive-systems)=
## The H-Theorem for Open Cognitive Systems

We reconcile computation with the Second Law of Thermodynamics {cite}`crooks1999entropy,parrondo2015thermodynamics`.

:::{div} feynman-prose
You might be wondering: how does all this relate to the Second Law of Thermodynamics? After all, when the agent reduces its belief entropy (becomes more certain), is not that a violation of the tendency for entropy to increase?

The answer, of course, is no. The Second Law applies to **closed** systems. The agent is an **open** system---it takes in energy (metabolic fuel) and uses that energy to reduce its internal entropy while increasing entropy elsewhere.

In the calibrated model, the conditional entropy-dissipation estimate lets us form a total-production residual: internal entropy change
plus the selected dissipation divided by the selected temperature scale. Under its regularity, no-flux, and calibration hypotheses,
that residual is non-negative. This is an open-system bookkeeping statement and an analogy to the Second Law; it is not a universal
claim about every learned update or every physical implementation.
:::

:::{prf:theorem} Total Entropy Production
:label: thm-total-entropy-production

The total entropy production rate of the agent $\sigma_{\text{tot}}$ during computation is:

$$
\sigma_{\text{tot}}(s) := \frac{d}{ds} H(\rho_s) + \frac{1}{T_c} \dot{\mathcal{M}}(s) \ge 0.

$$
*Proof.* Under the hypotheses of Theorem {prf:ref}`thm-generalized-landauer-bound`, $\dot{\mathcal{M}}(s) \ge T_c |\frac{d}{ds} H(\rho_s)|$. If $\frac{d}{ds} H < 0$ (entropy decreasing), then:

$$
\sigma_{\text{tot}} = \frac{dH}{ds} + \frac{\dot{\mathcal{M}}}{T_c} \ge \frac{dH}{ds} + \left| \frac{dH}{ds} \right| = \frac{dH}{ds} - \frac{dH}{ds} = 0.

$$
If $\frac{d}{ds} H \ge 0$, then $\sigma_{\text{tot}} \ge 0$ trivially since $\dot{\mathcal{M}} \ge 0$. $\square$

*Interpretation:* The agent can only reduce its internal uncertainty ($dH/ds < 0$) by dissipating metabolic energy ($\dot{\mathcal{M}} > 0$) {cite}`still2012thermodynamics`. This defines the **Efficiency of Thought**:

$$
\eta_{\text{thought}} := \frac{-T_c \cdot dH/ds}{\dot{\mathcal{M}}} \le 1.

$$
An agent is "thermodynamically fragile" if it requires high metabolic flux for low entropy reduction ($\eta_{\text{thought}} \ll 1$).

:::

:::{div} feynman-prose
This theorem is a cognitive analogue of the H-theorem. The non-negativity follows here from the conditional entropy-dissipation
estimate and its hypotheses, so it is best read as a consistency identity for the calibrated open-system model.

The efficiency of thought $\eta_{\text{thought}}$ measures how tightly an update approaches the selected lower bound. An efficiency of
1 means that the calibrated estimate is saturated; it does not by itself prove a physically reversible computation. An efficiency near
0 means that the modeled dissipation is large compared with the measured entropy decrease.

Values above 1 signal a mismatch with the assumptions, normalization, or estimators used by this model. They call for checking the
calibration and boundary bookkeeping; they are not, by themselves, evidence that a physical law has been violated.
:::

:::{prf:definition} Cognitive Carnot Efficiency
:label: def-cognitive-carnot-efficiency

The **Carnot limit** for cognitive systems is $\eta_{\text{thought}} = 1$, achieved when the belief update is a reversible isothermal process. Real agents operate at $\eta_{\text{thought}} < 1$ due to:
1. **Friction:** Non-optimal transport paths (geodesic deviation)
2. **Irreversibility:** Finite-rate updates (non-quasi-static processes)
3. **Dissipation:** Exploration noise ($T_c > 0$)

:::

:::{div} feynman-prose
Why do we call this the "Carnot efficiency"? The name is an analogy. Carnot's result concerns a heat engine between
thermodynamic reservoirs. Here $\eta_{\text{thought}}$ is a dimensionless ratio built from a calibrated entropy-rate estimate and a
modeled dissipation; the notation does not identify the agent with a heat engine or $T_c$ with a physical reservoir.

Within the stated model, $\eta_{\text{thought}}=1$ means that the conditional lower bound is saturated. The following sources of
loss are useful interpretations only when the corresponding geometry and dynamics have been specified:

1. **Friction:** If the WFR action is the relevant cost with fixed endpoints, a path longer than a minimizing geodesic can cost more.

2. **Irreversibility:** A finite-rate update can carry an additional cost when the model includes such a rate-dependent term. The
   quasi-static heat-engine analogy does not establish this term by itself.

3. **Exploration noise:** A stochastic update can change the entropy balance, and may counteract concentration, but the sign and size
   of that effect depend on the chosen dynamics and calibration.

This language helps diagnose where a calibrated implementation spends its modeled budget. It does not turn the analogy into a
universal physical limit.
:::

:::{warning}
:class: feynman-added

**On Thermodynamic Fragility**

An agent is "thermodynamically fragile" when its thinking efficiency $\eta_{\text{thought}}$ is low---it burns lots of energy to achieve only modest reductions in uncertainty.

This is dangerous for two reasons:

1. **Energy waste:** The agent depletes its metabolic budget quickly, potentially running out of "thinking fuel" when it matters most.

2. **Slow convergence:** Low efficiency means the agent takes longer to reach good beliefs. In time-critical situations, this can be fatal.

The diagnostic Node 51 (MetabolicEfficiencyCheck) monitors exactly this quantity. If $\eta_{\text{thought}}$ drops too low, the agent may be in "deliberative deadlock"---spinning its wheels without making progress.
:::



(sec-diagnostic-nodes-b)=
## Diagnostic Nodes 51--52

Following the diagnostic node convention ({ref}`sec-theory-thin-interfaces`), we define two new monitors for metabolic efficiency.

:::{div} feynman-prose
The theory is beautiful, but how do we know if an actual agent is behaving according to these principles? We need diagnostics---measurable quantities that tell us if things are working correctly.

Here we define two diagnostic nodes. The first checks whether the agent is getting a good modeled "return on investment" from its
thinking. The second checks consistency with the conditional entropy-dissipation estimate. A negative residual is a reason to inspect
the entropy estimator, calibration, boundary bookkeeping, or numerical solver; it is not by itself evidence that the physics of a
computation has been violated.
:::

(node-51)=
**Node 51: MetabolicEfficiencyCheck**

| **#**  | **Name**                     | **Component** | **Type**          | **Interpretation**             | **Proxy**                                                                                | **Cost** |
|--------|------------------------------|---------------|-------------------|--------------------------------|------------------------------------------------------------------------------------------|----------|
| **51** | **MetabolicEfficiencyCheck** | Solver        | Inference Economy | Is computation cost-effective? | $\eta_{\text{ROI}} := \frac{\lvert\Delta \langle V \rangle\rvert}{\Psi_{\text{met}}(S)}$ | $O(1)$   |

**Interpretation:** Monitors the **Return on Investment** of deliberation. High $\eta_{\text{ROI}}$ indicates efficient thinking; low $\eta_{\text{ROI}}$ indicates the agent is "daydreaming"---expending compute without improving terminal value.

**Threshold:** $\eta_{\text{ROI}} > \eta_{\text{min}}$ (typical default $\eta_{\text{min}} = 0.1$).

**Trigger conditions:**
- Low MetabolicEfficiencyCheck: The agent is in deliberative deadlock (Mode C.C: Decision Paralysis).
- **Remediation:** Apply **SurgCC** (time-boxing): force $S \le S_{\text{cap}}$ to bound deliberation.

(node-52)=
**Node 52: LandauerViolationCheck (EntropyProductionCheck)**

| **#**  | **Name**                   | **Component** | **Type**         | **Interpretation**                     | **Proxy**                                           | **Cost** |
|--------|----------------------------|---------------|------------------|----------------------------------------|-----------------------------------------------------|----------|
| **52** | **LandauerViolationCheck** | Dynamics      | Update Stability | Is the update thermodynamically valid? | $\delta_L := \dot{\mathcal{M}} + T_c \frac{dH}{ds}$ | $O(d)$   |

**Interpretation:** Monitors the Landauer bound (Theorem {prf:ref}`thm-generalized-landauer-bound`). A violation ($\delta_L < 0$) indicates entropy is decreasing faster than metabolic dissipation permits---a non-physical update.

**Threshold:** $\delta_L \ge -\epsilon_L$ (typical default $\epsilon_L = 10^{-4}$).

**Trigger conditions:**
- Negative LandauerViolationCheck: Non-physical belief update detected.
- **Cause:** Numerical errors in the WFR solver, unstable metric $G$, or incorrectly estimated entropy.
- **Remediation:** Reduce integration step size; verify metric positive-definiteness; check entropy estimator calibration.

*Cross-reference:* Node 52 extends the thermodynamic consistency checks of {ref}`sec-the-belief-evolution-cycle-perception-dreaming-action` (ThermoCycleCheck, Node 33) to the internal deliberation loop.

:::{div} feynman-prose
Let me say a word about the Landauer Violation Check, because it is unusual to have a diagnostic named after a physical bound.

In most simulations, a consistency condition is built into the update rule. Here the agent may learn or approximate its dynamics, so
the measured quantities can fall outside the calibrated estimate through numerical error, estimator bias, or a mismatch between the
implemented and assumed dynamics.

The diagnostic therefore asks whether the observed entropy decrease is compatible with the modeled metabolic expenditure. If the
residual is negative, check the entropy estimator, the metabolic coefficient, the no-flux assumption, and the WFR solver before
interpreting the result.

When it triggers, repair the underlying mismatch rather than hiding it with a threshold. The conditional bound is a property of the
declared model and calibration; it does not by itself certify that an implementation is physically impossible.
:::



(sec-summary-table-computational-thermodynamics)=
## Summary Table: Computational Thermodynamics

:::{div} feynman-prose
Let me wrap up by giving you the complete dictionary between thermodynamic concepts and their agent counterparts. This table is the Rosetta Stone for translating between physics and AI.
:::

**Table 31.6.1 (Computational Metabolism Summary).**

| Concept                | Thermodynamic Variable | Agent Implementation                                          |
|:-----------------------|:-----------------------|:--------------------------------------------------------------|
| **Energy**             | Gibbs Free Energy      | Task Potential $V(z)$                                         |
| **Heat**               | Metabolic Dissipation  | WFR Action $\dot{\mathcal{M}}$                                |
| **Work**               | Value Improvement      | Gradient Flux $\langle \nabla_A V, v \rangle_G$                 |
| **Equilibrium**        | $dG = 0$               | $S^*$ (Optimal Stopping)                                      |
| **Temperature**        | $T$                    | Cognitive Temperature $T_c$                                   |
| **Entropy Production** | $\sigma \ge 0$         | $\sigma_{\text{tot}} = \dot{H} + \dot{\mathcal{M}}/T_c \ge 0$ |

**Key Results:**
1. **Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):** $\dot{\mathcal{M}} \ge T_c |\dot{H}|$---thinking has a thermodynamic cost.
2. **Optimal Deliberation (Theorem {prf:ref}`thm-deliberation-optimality-condition`):** $S^*$ satisfies $\Gamma(S^*) = \dot{\mathcal{M}}(S^*)$---stop thinking when marginal returns equal marginal costs.
3. **Phase Transition (Theorem {prf:ref}`thm-fast-slow-phase-transition`):** Fast ($S^* = 0$) vs. Slow ($S^* > 0$) is determined by $\Gamma(0) \lessgtr \dot{\mathcal{M}}(0)$.

**Conclusion.** Computational Metabolism provides the "biological" limit for the Fragile Agent. By deriving $S^*$ from first principles, we transform the "Thinking Fast vs. Slow" heuristic into a rigorous physical law. The agent acts not when it is "ready," but when it is no longer metabolically efficient to continue refining its belief. This framework connects to the free energy principle {cite}`friston2010free` and active inference {cite}`friston2017active`, providing a thermodynamic foundation for bounded rationality.

:::{div} feynman-prose
And there you have it. We started with a simple question---how long should you think before you act?---and obtained a conditional,
operational account of deliberation.

The key points are:

1. **A modeled computation cost can be related to entropy change.** The Cauchy--Schwarz estimate and its Landauer form require the
   stated regularity, boundary, normalization, and calibration hypotheses; the classical physical principle is an analogy until the
   physical interface is supplied.

2. **An optimal thinking time minimizes a declared action.** For an interior minimizer, stationarity gives the marginal condition.
   Endpoint optima and multiple stationary points remain possible when the additional assumptions fail.

3. **Fast and slow thinking are allocation regimes in this model.** The initial marginal-gain comparison can classify those regimes
   under the theorem's hypotheses, but it does not establish a universal psychological or physical phase transition.

4. **The entropy-production check is conditional.** Its non-negativity follows from the calibrated estimate; a negative residual asks
   us to inspect the model, estimators, and numerics before drawing a physical conclusion.

The framework therefore constrains the declared model and provides useful diagnostics. Ignoring a measured computation cost can still
make deliberation wasteful, but a failed bound is evidence of a mismatch with the stated assumptions rather than proof of physical
impossibility.

For implementation, track calibrated dissipation and entropy estimates, record the boundary and normalization assumptions, and use
the stopping condition when its regularity and horizon hypotheses hold. Empirical validation is needed before transferring the result
to a real cognitive or hardware system.
:::
