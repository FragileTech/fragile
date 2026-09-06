(sec-parameter-space-sieve)=
# The Parameter Space Sieve: Deriving Fundamental Constants

## TLDR

- Treat “constants” as **feasibility constraints**: viable agents can only exist inside a region of parameter space where
  causality, capacity, thermodynamics, and stability all hold simultaneously.
- Collect operational constants into a single parameter vector $\Lambda$ and express the architecture as coupled
  inequalities (“the Sieve” in parameter space).
- Off-feasible-region behavior is categorical: violations produce causal incoherence, holographic overload, thermodynamic
  inconsistency, or ontological dissolution.
- This chapter is a synthesis node: it ties together metric law, WFR, belief dynamics, metabolism, and information
  bounds into one constraint system.
- Output: a checklist of constraints you can use to reason about scaling limits and “why the knobs can’t be arbitrary”.

## Roadmap

1. Formulate the parameter vector and the constraint system.
2. Derive/justify each inequality from earlier chapters.
3. Interpret the feasible region and what failure looks like outside it.

:::{div} feynman-prose
Here is an audacious question: Why is the speed of light what it is? Why is the fine structure constant approximately 1/137? For a century, physicists have treated these as brute facts---numbers you look up in a table, not numbers you derive.

This chapter takes a different view. We ask: What if these constants are not arbitrary? What if they are the *only* values that permit coherent agents to exist?

The logic is almost embarrassingly simple once you see it. An agent must satisfy certain consistency conditions---it cannot receive messages from its own future, it cannot store infinite information in finite space, it cannot think hotter than its energy budget permits. Each condition carves out a region in parameter space. The intersection of all these regions---the *feasible region*---is where viable agents can exist.

And here is the modest conclusion: any universe in which this kind of agent exists must sit inside the corresponding feasible region. Our universe supplies an example of an environment in which we can ask the question; that observation alone does not show that the Sieve selected its constants or that the region is unique.

This is not mysticism. It is constraint satisfaction. The same logic that tells you a bridge must be strong enough to hold its own weight tells you that a universe must have constants compatible with agency. We are going to derive those constraints.
:::

*Abstract.* This chapter derives the constraints on fundamental constants from cybernetic first principles. We formulate
the Sieve Architecture as a system of coupled inequalities that any viable agent must satisfy. The fundamental constants
$\Lambda = (c_{\text{info}}, \sigma, \ell_L, T_c, g_s, \gamma)$ are not free parameters but decision variables of a
constrained optimization problem. The physical universe exists within the **Feasible Region** where all constraints are
simultaneously satisfied. We prove that moving off this region triggers a Sieve violation: the agent either loses causal
coherence, exceeds its holographic bound, violates thermodynamic consistency, or suffers ontological dissolution.

*Cross-references:* This chapter synthesizes:
- {ref}`sec-capacity-constrained-metric-law-geometry-from-interface-limits` (Capacity-Constrained Metric Law)
- {ref}`sec-the-belief-wave-function-schrodinger-representation` (Cognitive Action Scale)
- {ref}`sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics` (Generalized Landauer Bound)
- {ref}`sec-causal-information-bound` (Causal Information Bound, Area Law)
- The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62)



(sec-sieve-formulation)=
## The Sieve Formulation: Agents as Constraint Satisfaction

:::{div} feynman-prose
Let me tell you how I think about this. Imagine you are designing a robot. You have knobs to turn: How fast should signals travel through its circuits? How fine-grained should its sensors be? How much energy should it burn per computation?

Now, you might think you can set these knobs however you like. But that is not true. If signals travel too slowly, different parts of the robot cannot coordinate---it becomes paralyzed. If signals travel too fast relative to the robot's memory, it gets confused about what happened when---it hallucinates causality violations. If sensors are too coarse, the robot cannot distinguish important states. If they are too fine, it runs out of memory. If it thinks too "hot" (explores too aggressively), it forgets faster than it can afford energetically.

Every knob has a viable range. Step outside that range, and the robot stops working---not gradually degrades, but *categorically fails*.

The Parameter Vector $\Lambda$ collects all these knobs into one mathematical object. The Sieve is the system of inequalities that says which settings work.
:::

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

:::{div} feynman-prose
Let me make sure you understand what each of these parameters means intuitively. The physics names below form a proposed dictionary; they are not identities supplied by the agent equations.

**Information speed** $c_{\text{info}}$ is how fast a signal can propagate through the agent's internal state. In your brain, this is related to axon conduction velocities. In a computer, it is the speed of electrical signals through wires. A proposed physical identification with $c$ requires a map of units and observables.

**Cognitive action scale** $\sigma$ sets the minimum distinguishable change in the agent's planning. Below this scale, different plans look identical under the chosen resolution. Calling it $\hbar$ is a conjectural physical identification, not a consequence of the notation.

**Levin length** $\ell_L$ is the smallest spatial scale the agent can resolve under the declared capacity convention. The associated cell count is model-dependent. A correspondence with $\ell_P$ (Planck length) is a proposed physical dictionary entry.

**Cognitive temperature** $T_c$ controls exploration versus exploitation. High temperature means the agent explores more broadly; low temperature means it sticks to known good options. Its relation to a physical $k_B T$ is a units-and-observables question.

**Binding coupling** $g_s$ determines how strongly features stick together to form objects. Too weak, and objects can fall apart into meaningless features; too strong, and everything can clump into one undifferentiated blob. Its proposed correspondence with $\alpha_s$ (the strong coupling) remains an interpretation to test.

**Discount factor** $\gamma$ determines how far into the future the agent plans. $\gamma \to 1$ means infinite planning horizon; $\gamma \to 0$ means totally myopic.

Each of these has a viable range. We are about to derive those ranges.
:::

:::{prf:definition} The Sieve Constraint System
:label: def-sieve-constraint-system

Let $\mathcal{S}(\Lambda)$ denote the vector of constraint functions. The agent is **viable** if and only if:

$$
\mathcal{S}(\Lambda) \le \mathbf{0}

$$

where the inequality holds component-wise. Each component corresponds to a Sieve node that enforces a specific consistency condition. A constraint violation ($\mathcal{S}_i > 0$) triggers a diagnostic halt at the corresponding node.

:::

:::{div} feynman-prose
The mathematical notation $\mathcal{S}(\Lambda) \le \mathbf{0}$ is compact but the idea is simple: you have a checklist of conditions, and *all* of them must be satisfied simultaneously.

Think of it like building codes. A building must satisfy fire safety *and* structural integrity *and* electrical codes *and* plumbing codes. Failing any single one makes the building non-viable. You do not get to trade off "a bit less fire-safe" for "a bit more structural integrity."

Same here: the declared agent model cannot call a configuration viable while one of its constraints is violated. Each constraint
points to a particular failure mode or diagnostic response. The consequence is model- and controller-dependent; a threshold
violation is not automatically catastrophic in every implementation.
:::



(sec-causal-consistency-constraint)=
## The Causal Consistency Constraint

:::{div} feynman-prose
Now we get to the first constraint, and it is a beautiful one. It says: information cannot travel too slow *or* too fast.

Too slow is obvious---if your left hand cannot tell your right hand what it is doing, you cannot coordinate. But too fast is subtler. If information travels so fast that you can receive messages from your own future, you get paradoxes. Your prediction depends on data you have not generated yet.

Both extremes are forbidden for the declared agent interface. There is a window of viable information speeds for that model; comparing it with a physical propagation speed requires a separate map of units, observables, and mechanisms.
:::

We derive the bounds on information speed from the requirements of buffer coherence and synchronization.

:::{prf:axiom} Causal Buffer Architecture
:label: ax-causal-buffer-architecture

Let the agent possess:
1. **$L_{\text{buf}}$:** Maximum buffer depth (spatial extent of causal memory)
2. **$\tau_{\text{proc}}$:** Minimum processing interval (temporal resolution)
3. **$d_{\text{sync}}$:** Minimum synchronization distance (coherence length)

These define the operational envelope within which the agent maintains consistent state updates.

:::

:::{div} feynman-prose
These three quantities define the agent's "size" in a causal sense.

**Buffer depth** $L_{\text{buf}}$ is how far into its past the agent can remember---literally, how much spatial extent its causal memory spans. Think of RAM in a computer, or working memory in a brain.

**Processing interval** $\tau_{\text{proc}}$ is the agent's "clock tick"---the minimum time between state updates. Below this timescale, the agent cannot distinguish "before" from "after."

**Synchronization distance** $d_{\text{sync}}$ is how far apart two modules can be while still coordinating their updates. If modules are farther apart than this, they cannot agree on a common present.

Now, here is the key insight: these three quantities constrain how fast information can travel.
:::

:::{prf:theorem} The Speed Window
:label: thm-speed-window

The information speed $c_{\text{info}}$ must satisfy the **Speed Window Inequality**:

$$
\frac{d_{\text{sync}}}{\tau_{\text{proc}}} \le c_{\text{info}} \le \frac{L_{\text{buf}}}{\tau_{\text{proc}}}

$$

*Proof.*

**Lower Bound (Node 2: ZenoCheck):**

Suppose $c_{\text{info}} < d_{\text{sync}}/\tau_{\text{proc}}$. Then information cannot traverse the synchronization distance within one processing cycle. By the Causal Interval (Definition {prf:ref}`def-causal-interval`), spacelike-separated modules cannot coordinate updates. The agent enters a **Zeno freeze**: each module waits indefinitely for signals that arrive too slowly. The belief update stalls, violating the continuity required by the WFR dynamics ({ref}`sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces`).

**Upper Bound (Node 62: CausalityViolationCheck):**

Suppose $c_{\text{info}} > L_{\text{buf}}/\tau_{\text{proc}}$. Then signals can traverse the entire buffer depth within one processing cycle. This creates **temporal aliasing**: the agent receives information about its own future state before that state is computed. By the Safe Retrieval Bandwidth (Theorem {prf:ref}`thm-safe-retrieval-bandwidth`), this constitutes a causal paradox—the agent's prediction depends on data it has not yet generated.

Node 62 enforces Theorem {prf:ref}`thm-causal-stasis`: the metric becomes singular at the boundary where causal violations would occur, preventing traversal.

$\square$

:::

:::{div} feynman-prose
Let me give you a physical picture of what is happening here.

**Lower bound (too slow):** Imagine a centipede trying to walk, but nerve signals travel so slowly that by the time leg 50 gets the "step now" signal, leg 1 has already taken three more steps. The legs cannot coordinate. In the declared synchronous model, the update can then stall: each leg waits for a coordination signal that arrives too late to be useful. This is the model's Zeno-freeze diagnostic, under the stated timing and continuity assumptions.

**Upper bound (too fast):** Now imagine the centipede's nerves are so fast that signals can travel the full length of its body in less time than it takes to complete one step. In a model that treats the buffer as a causal history, this can create temporal aliasing: leg 1 receives a state tagged as if leg 100 had already responded, although that update has not been computed. That is a consistency failure of the declared history model, not a proof that a physical signal literally came from the future.

The viable window is therefore: fast enough to coordinate, while respecting the history and processing conventions used by this agent model.

The speed of light can be compared with this window after the agent's length, time, and causal observables have been specified. The comparison is suggestive, but the Sieve calculation does not by itself show that $c$ was selected by agency or that the two systems share a physical mechanism.
:::

:::{note}
:class: feynman-added
The Speed Window theorem is an agent-level result. Under its declared buffer, processing, synchronization, and causal-interval
hypotheses, the model requires a finite nonzero $c_{\text{info}}$. Infinite or zero values violate those particular interface
assumptions. The theorem does not establish a cosmic speed limit, identify $c_{\text{info}}$ with the physical speed of light,
or show that the universe selected its constants for agency.
:::

:::{prf:corollary} The Speed Ratio Bound
:label: cor-speed-ratio-bound

The ratio of buffer depth to synchronization distance is bounded:

$$
\frac{L_{\text{buf}}}{d_{\text{sync}}} \ge 1

$$

with equality only in the degenerate case of a single-module agent. For distributed agents, this ratio determines the dynamic range of viable information speeds.

:::

:::{div} feynman-prose
This corollary has a nice interpretation: the agent's memory must extend at least as far as its coordination requirements. You cannot have modules that need to synchronize over distances larger than the memory buffer.

For a single-module agent (a point), buffer depth equals synchronization distance---there is nothing to synchronize with, and no history to store. For a distributed agent with many modules, the buffer must be deeper than the inter-module distances.
:::



(sec-holographic-stability-constraint)=
## The Holographic Stability Constraint

:::{div} feynman-prose
Here is a deep constraint that seems almost magical at first, until you see where it comes from.

Suppose you want to assign a capacity to some region of a model. Your first guess might be: volume. A bigger region, more information. The area-law convention used here makes a different assignment, but it applies only after the boundary, resolution, and channel have been declared.

Under that convention, the capacity is proportional to the area of the chosen boundary.

This borrows the language of the holographic principle, first discussed in black-hole physics. The information-theoretic intuition is that a finite retrieval channel can limit usable storage. Turning that intuition into an area-law bound requires the explicit counting model, units, dimensional normalization, and channel hypotheses; it is not a universal consequence of bandwidth alone.
:::

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

:::{div} feynman-prose
The declared capacity model assigns the relevant information budget through a boundary interface; it does not say that information literally lives only on boundaries.

Think about it this way. You have a room full of stuff. You want to know everything about what is in the room. But you can only look at the room through its walls (the boundary). How much can you learn?

You might think: if I make the walls higher resolution (smaller $\ell_L$), I can see more detail. True. But there is a limit. At some point, the walls themselves are packed so densely with information that you cannot read them fast enough. The bandwidth of reading limits the resolution of storage.

With a specified cell-counting model, the resulting operational capacity scales like boundary area divided by $\ell_L^{D-1}$. That is the model's area-law diagnostic; extending it to another geometry or a physical system requires new estimates.
:::

:::{admonition} Intuition: Why Area, Not Volume?
:class: feynman-added tip

Here is an interface-based intuition for the selected capacity convention. Suppose you try to pack information at density $\rho_I$
(bits per unit volume). The total information is $\rho_I \cdot V$. If the declared read channel is attached to a boundary of area
$A$ and its throughput is assumed to scale with $A$, access can become the bottleneck.

If $\rho_I \cdot V > \text{const} \cdot A$, the model predicts that the channel cannot read the memory within the chosen operational window.
Information that cannot be accessed is unusable for that controller.

With the extra cell-counting, dimensional, and throughput assumptions, this gives $\rho_I \lesssim A/V$ and total usable
information scaling like $A$. The scaling is an operational model for a declared interface; it is not a derivation of black-hole
entropy or a universal area law.

The black-hole comparison is a separate physical analogy. Its entropy-area relation requires its own gravitational and
semiclassical hypotheses.
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

:::{div} feynman-prose
Now here is something that should make you sit up. The Planck length is $\ell_P \approx 1.6 \times 10^{-35}$ meters. This is fantastically small---about $10^{20}$ times smaller than a proton. In this chapter it is a reference scale in the proposed physics dictionary, not an experimentally established value of the agent's resolution.

Why might a small resolution scale be useful? Under the declared capacity convention, smaller $\ell_L$ permits more cells per boundary measure and potentially a more detailed model. The Sieve does not show that evolution selected the physical Planck scale or that agency explains its value.

The physical statement that no distance below $\ell_P$ is meaningful belongs to a quantum-gravity hypothesis outside this agent model. Here $\ell_L>0$ is a declared resolution floor, and any identification with $\ell_P$ remains conjectural.
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

:::{div} feynman-prose
This result describes the selected metric ansatz as the declared capacity threshold is approached. It does not turn every area-law diagnostic into a universal horizon or guarantee that geometry will prevent an over-capacity state.

Under the radial, force, and boundary hypotheses of the cited result, a selected inverse metric component can tend to zero, so the corresponding radial update can slow. A full metric divergence, infinite distance, or exclusion of a forbidden region needs the additional estimates stated in that result.

The black-hole horizon is a useful analogy for that radial slowdown, but the agent calculation is not a black-hole solution. It does not imply freezing of all directions, an event horizon, or finite-proper-time inaccessibility.

The capacity expression is a declared operational model with conditional geometric consequences. The black-hole connection is a conjectural interpretation, not a consequence established by information theory alone.
:::



(sec-metabolic-viability-constraint)=
## The Metabolic Viability Constraint

:::{div} feynman-prose
Now we come to thermodynamics. Thinking costs energy. More precisely: under the usual thermal and logical-irreversibility assumptions, erasing information has a Landauer lower bound.

This is Landauer's principle, and it is one of the important results in the thermodynamics of computation. An erased bit contributes at least $k_B T \ln 2$ of heat in that idealized setting; a physical implementation may use reversible steps or additional dissipation, so the formula is not a complete energy model.

What does this have to do with cognitive temperature? Well, a "hotter" agent explores more possibilities, which means it forgets more possibilities (the ones it did not take). More forgetting, more energy. If the agent thinks hotter than it can afford, it starves.
:::

We derive the thermodynamic constraint on computational operations from the Generalized Landauer Bound.

:::{prf:definition} Metabolic Parameters
:label: def-metabolic-parameters

The agent possesses:
1. **$\dot{E}_{\text{met}}$:** Metabolic power budget (energy flux available for computation)
2. **$\dot{I}_{\text{erase}}$:** Information erasure rate (bits forgotten per unit time)
3. **$T_c$:** Cognitive Temperature (entropy-exploration tradeoff)

:::

:::{div} feynman-prose
These are the thermodynamic "books" of the agent.

**Metabolic power** $\dot{E}_{\text{met}}$ is the energy income---how many joules per second the agent can spend on computation. For a brain, this is about 20 watts. For a laptop, maybe 50 watts. For the universe as a whole... well, that is an interesting question.

**Erasure rate** $\dot{I}_{\text{erase}}$ is how fast the agent forgets. Every time you update your beliefs, you erase some old beliefs to make room for new ones. Every time you reject an option, you forget the rejected counterfactual.

**Cognitive temperature** $T_c$ sets the exploration-exploitation tradeoff. High temperature means "consider many options, choose randomly among good ones." Low temperature means "always pick the best-known option." Exploration requires considering (and then discarding) more possibilities, hence more erasure, hence more energy.
:::

:::{prf:theorem} The Landauer Constraint
:label: thm-landauer-constraint

The Cognitive Temperature must satisfy:

$$
T_c \le \frac{\dot{E}_{\text{met}}}{\dot{I}_{\text{erase}} \cdot \ln 2}

$$

where we use natural units with $k_B = 1$.

*Proof.*

**Step 1.** Under the regularity, no-flux, mass-preservation, and calibration hypotheses of the conditional Generalized Landauer Bound (Theorem {prf:ref}`thm-generalized-landauer-bound`):

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

:::{div} feynman-prose
Let me put this in everyday terms. Suppose your brain has a metabolic budget of 20 watts and you need to forget, say, $10^{10}$ bits per second to keep up with sensory input (a reasonable estimate for visual processing). Then:

$$
T_c \le \frac{20 \text{ J/s}}{10^{10} \text{ bits/s} \times 0.693} \approx 3 \times 10^{-9} \text{ J}

$$

This corresponds to about $7 \times 10^{14}$ kelvin in temperature units---room temperature is about $4 \times 10^{-21}$ joules, so the brain is operating *way* below the thermodynamic limit.

Why so far below? Because brains are not thermodynamically optimal computers. There is a lot of overhead. But the limit exists, and if you tried to build an agent that thinks "hotter" than this bound, it would starve.
:::

:::{warning}
:class: feynman-added
The Landauer expression is a lower bound for logically irreversible erasure under its ideal thermal assumptions. Here $T_c$ is the
agent's cognitive or exploration scale, not automatically the reservoir temperature $T$. Relating the two requires a calibrated
map of units, erasure rates, hardware, and protocol. The displayed operational constraint therefore does not by itself say that
an agent cannot explore cognitively hotter than every physical heat bath.
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

:::{div} feynman-prose
The general budget stated here is the Landauer-style bound under its thermal and logical-erasure assumptions. The angular SDE
supplies a second, local diagnostic rather than another universal temperature law.

The **Landauer bound** says: under the declared erasure and thermal model, a hotter cognitive schedule requires a larger energy
budget. If the schedule exceeds that budget, the selected implementation cannot sustain it; "starve" is shorthand for that model
failure, not a universal law about cognition.

The Péclet crossover $\mathrm{Pe}_\theta\approx1$ asks whether angular drift competes with thermal diffusion at a chosen radius and finite horizon. It can motivate an engineering setpoint, but it is not a universal bifurcation temperature or a proof that decisions become a random walk.

A viable agent must satisfy the Landauer inequality and report the radius, horizon, control field, and local geometry used for any Péclet-based decision diagnostic. The two checks concern different objects and should not be merged into a single temperature bound.
:::



(sec-hierarchical-coupling-constraint)=
## The Hierarchical Coupling Constraint

:::{div} feynman-prose
This section is about glue. How strongly should things stick together?

At the macro scale, you want strong glue. Objects should be stable---a chair should not fall apart into quarks while you are sitting on it. Features should bind into recognizable concepts.

At the micro scale, you want weak glue. Texture should not clump. Noise should remain noise, not spontaneously organize into spurious structure.

This has a useful structural resemblance to the QCD vocabulary of infrared binding and ultraviolet decoupling. The resemblance does not identify the feature coupling with the QCD beta function; the agent inequalities describe a design regime, and any physical correspondence remains conjectural.
:::

We derive the constraints on the binding coupling $g_s$ from the requirements of object permanence and texture decoupling.

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

:::{div} feynman-prose
The coupling $g_s(\mu)$ tells you how strongly features interact at scale $\mu$.

Think of the representation hierarchy: at the top (macro), you have abstract concepts---"chair," "face," "danger." At the bottom (micro), you have textures---pixel noise, high-frequency details. The coupling controls how strongly adjacent features attract each other at each level.

The scale derivative of a coupling describes how the chosen interaction changes as you zoom in or out. If that derivative is negative in the agent's convention, the interaction weakens toward fine scales; this has the shape of asymptotic freedom, but it is not a QCD beta-function calculation.

For viable agents, the desired schedule is strong binding at macro scale and weak coupling at micro scale.
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

**Step 3.** For $SU(N_f)$ gauge theory, this requires strong coupling $g_s > g_s^{\text{crit}}$ at large distances (Area Law, {ref}`sec-causal-information-bound`).

**Step 4.** If $g_s(\mu_{\text{IR}}) < g_s^{\text{crit}}$, features escape confinement—"color-charged" states propagate to the boundary $\partial\mathcal{Z}$. This violates the Observability Constraint (Definition {prf:ref}`def-boundary-markov-blanket`): the agent cannot form stable objects.

The DNN-local BindingConfinementCheck (DNN-B) enforces that only color-neutral bound states reach the macro-register; global Node 40 is the CapacitySaturationCheck.

$\square$

:::

:::{div} feynman-prose
The Infrared Binding Constraint says: at macro scale, features must stick together strongly enough to form stable objects.

Imagine building a tower from Lego bricks. If the bricks do not click firmly enough, the tower falls over. You cannot perceive "tower"---you just see a pile of loose bricks.

Same with cognitive features. If edge detectors and color patches do not bind strongly enough, you cannot perceive "cat"---you just see a soup of features. Object permanence requires strong binding at the scale where objects live.

In QCD, confinement is a dynamical statement about a quantum field theory. The agent-theoretic version is an architectural goal: feature interactions at concept scales should make bound representations useful and should suppress raw texture at the macro interface. The two mechanisms should not be identified without a map of states and observables.
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

**Step 3.** From the RG interpretation ({ref}`sec-stacked-topoencoders-deep-renormalization-group-flow`), the TopoEncoder implements coarse-graining. Residual coupling at the UV scale would prevent efficient compression—the Kolmogorov complexity of texture would diverge.

**Step 4.** Asymptotic freedom ($\beta < 0$) provides the required behavior: $g_s \to 0$ as $\mu \to \infty$.

Node 29 (TextureFirewallCheck) enforces this decoupling.

$\square$

:::

:::{div} feynman-prose
The UV Decoupling Constraint says: at texture scale, features must *not* stick together.

Why? Because texture is supposed to be disposable noise. If texture elements started binding, you would see spurious structure---faces in clouds, patterns in static. The compression algorithm would fail because "random noise" would actually contain structure that resists compression.

In QCD, asymptotic freedom is a statement about a particular renormalized quantum coupling. Here the desired decrease of feature coupling toward fine scales is an architectural schedule. It is analogous in shape, but it does not establish a QCD beta function or asymptotic freedom.

The agent-theoretic version: at texture scale, features should not interact. Only when you zoom out to concept scale should binding appear.
:::

:::{admonition} The Deep Connection to QCD
:class: feynman-added note

This is a comparison, not an identity. QCD's running coupling and confinement come from a particular renormalized quantum field
theory. The agent coupling $g_s(\mu)$ is an architectural schedule whose binding and decoupling properties must be checked in the
declared representation.

In QCD, the coupling $\alpha_s(\mu)$ runs with energy scale $\mu$. At $\mu = M_Z \approx 91$ GeV, $\alpha_s \approx 0.12$ (weak). At $\mu \approx \Lambda_{\text{QCD}} \approx 200$ MeV, $\alpha_s \to \infty$ (strong).

In the agent framework, the binding coupling $g_s(\mu)$ runs with representation scale $\mu$. At UV (texture), $g_s \to 0$ (decoupled). At IR (concepts), $g_s > g_s^{\text{crit}}$ (bound).

The two profiles may have the same sign or qualitative shape in a chosen comparison. That resemblance does not establish the same
beta function, physical consequence, or mathematical isomorphism. A physical identification would need a map of states,
observables, scales, and dynamics.
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

:::{div} feynman-prose
Now we come to a constraint about memory---specifically, about the tradeoff between remembering and learning.

In the selected metastable-state model, if memories are too fragile, thermal noise can erase them before the chosen observation
horizon. Stable beliefs then cannot be maintained by that implementation.

If they are too rigid, the same model can suppress updates on that horizon. The agent may then appear frozen in its initial state,
even though a different drive or a longer horizon could still permit transitions.

The viable regime is in between: stiff enough to resist the modeled noise, flexible enough to change when the modeled evidence and
control field supply enough drive. This is a Goldilocks design regime, whose location depends on the rates and horizon.
:::

We derive the constraint on the separation between adjacent energy levels that enables both memory stability and dynamic flexibility.

:::{prf:definition} The Stiffness Parameter
:label: def-stiffness-parameter

Let $\Delta E$ denote the characteristic energy gap between metastable states in the agent's latent manifold. Define the **Stiffness Ratio**:

$$
\chi = \frac{\Delta E}{T_c}

$$

This ratio determines the tradeoff between memory persistence and adaptability.

:::

:::{div} feynman-prose
The stiffness ratio $\chi$ is a useful dimensionless number for the selected metastable-state model.

**$\chi < 1$:** Energy barrier is smaller than thermal energy. In the corresponding Arrhenius model, transitions can be frequent
on the chosen observation horizon, so stable memory may be lost.

**$\chi \gg 1$:** Energy barrier is much larger than thermal energy. In that same model, transitions can be suppressed on the
chosen horizon, so updates may appear frozen; this does not prove that learning is impossible or that the state is eternally fixed.

**$\chi \sim 1$ to $\chi \sim 10$:** A possible Goldilocks range for the selected rates and horizon: beliefs can resist some
random fluctuations while still changing when evidence supplies enough drive.

Think of a marble in a bowl. If the bowl is too shallow (small $\chi$), thermal vibrations knock the marble out. If the bowl is too deep (large $\chi$), you cannot push the marble out even when you want to. You need a bowl of just the right depth.
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

:::{div} feynman-prose
The Stiffness Theorem is really about timescales.

A transition with $\chi = 10$ has probability $e^{-10} \approx 4 \times 10^{-5}$ per thermal fluctuation in the idealized
Arrhenius estimate. If fluctuations happen at rate $\nu$, the corresponding waiting-time estimate is $\sim e^{\chi}/\nu$.
For $\chi = 10$ and the illustrative choice $\nu = 10^{12}$ Hz, it is about $10^{-7}$ seconds; that number is a calibration
example, not a universal molecular rate.

For $\chi = 100$, the same Arrhenius estimate gives $e^{100}/\nu \approx 10^{31}$ seconds---longer than the chosen operational horizon. Under that model and rate estimate, the transition is effectively frozen for the application; it is not an assertion of eternal physical freezing.

So $\chi_{\text{max}}$ is really about: how long can this application wait for belief updates? A horizon such as $10^8$ seconds
and the rate above would give $\chi_{\text{max}} \approx \ln(10^8 \times 10^{12}) \approx 46$, but neither number is a
biological universal.

For chemical bonds at room temperature, an order-of-magnitude comparison may give $\chi \approx 500$. That is a different
timescale problem: chemical bonds are not supposed to flip during an agent's lifetime. The stiffness diagnostic applies to the
declared cognitive states and rate model, not automatically to structural matter.
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

:::{div} feynman-prose
Here is one of the most provocative comparisons. The fine-structure constant $\alpha \approx 1/137$ has puzzled physicists for a century. A cybernetic model can ask whether an analogous binding window would be compatible with chemistry, but that question is not yet a derivation of $\alpha$.

The Sieve supplies a conditional viability diagnostic, not an answer for the physical constant.

If $\alpha$ were changed in a specified atomic model, binding and reaction scales would change roughly like $\alpha^2$ in the simplest approximation. Whether chemistry becomes fragile requires solving that physical model, including other constants and reaction pathways.

Whether larger binding scales make reactions too slow likewise requires a model of the relevant activation pathways; the Sieve does not establish “no life” from $\alpha$ alone.

The observed value can be compared with such a window only after the physical model and its uncertainty ranges are fixed.

The agent calculation gives a quantitative conditional statement: if a chosen energy scale obeys $\chi = \Delta E/(k_B T) \propto \alpha^2 m_e c^2/(k_B T)$ and must satisfy $1 < \chi < \chi_{\text{max}}$, it produces an allowed interval. Mapping that interval to the observed $\alpha$ is a conjectural physical identification, not a conclusion of the Sieve alone.
:::



(sec-discount-screening-constraint)=
## The Temporal Screening Constraint

:::{div} feynman-prose
The last constraint is about planning horizons. How far into the future should the agent care about?

Zero horizon ($\gamma = 0$) means totally myopic: only the next timestep matters. Infinite horizon ($\gamma = 1$) means caring equally about all future times forever.

Both extremes are bad. Myopic agents stumble into avoidable long-term disasters. Infinite-horizon agents are paralyzed trying to consider all consequences unto eternity.

There is a viable window in between. And that window has beautiful connections to screening in field theory.
:::

We derive the constraint on the discount factor from the requirements of causal coherence and goal-directedness.

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

*Proof.*

**Upper Bound ($\gamma < 1$):**

**Step 1.** From the stationary-diffusion Bellman generator (Theorem {prf:ref}`thm-the-hjb-helmholtz-correspondence`), the Value function satisfies:

$$
( -\Delta_G + \lambda/T_c ) V = \rho_r/T_c

$$

where $\kappa_{\mathrm{diff}}^2:=\lambda/T_c=(-\ln\gamma)/(T_c\Delta t)$ has dimension $[L^{-2}]$.
The propagation scale $\ell_0=c_{\text{info}}\tau_{\text{proc}}$ is a separate quantity (Definition
{prf:ref}`def-agent-parameter-vector`).

**Step 2.** For $\gamma = 1$, we have $\lambda=\kappa_{\mathrm{diff}}=0$. The equation becomes Poisson's equation for the conservative
component:

$$
-\nabla^2 V = \rho_r

$$
where $\rho_r$ is the conservative reward source density (Definition {prf:ref}`def-the-reward-flux`).

For $D>2$, the Green's function decays as $1/r^{D-2}$ (long-range); for $D=2$ it grows logarithmically.

**Step 3.** Long-range value propagation violates locality: distant conservative reward sources dominate nearby
decisions. The agent cannot form local value gradients for navigation.

**Step 4.** From Corollary {prf:ref}`cor-discount-as-screening-length`, finite stationary screening
$\kappa_{\mathrm{diff}}>0$ (i.e., $\gamma<1$) is required for local goal-directedness.

**Lower Bound ($\gamma > \gamma_{\text{min}}$):**

**Step 5.** For $\gamma \to 0$, we have $-\ln\gamma \to \infty$, hence $\kappa_{\mathrm{diff}} \to \infty$. The **Screening Length** (dimension $[L]$):

$$
\ell_{\mathrm{diff}} = \frac{1}{\kappa_{\mathrm{diff}}}
 = \sqrt{\frac{T_c\Delta t}{-\ln\gamma}} \to 0

$$

**Step 6.** Zero screening length means the agent responds only to immediate conservative rewards—it has no planning
horizon.

**Step 7.** This violates the Causal Buffer requirement (Axiom {prf:ref}`ax-causal-buffer-architecture`): the agent must anticipate beyond its current timestep.

$\square$

:::

:::{div} feynman-prose
The analogy here is useful. The discount factor $\gamma$ creates a model-dependent screening scale $\kappa$ for the value field, which resembles the way a photon mass produces a screening length in a specified electromagnetic medium. In the stationary diffusion convention used for the value equation, set $\lambda=-\ln\gamma/\Delta t$ and obtain $\kappa^2=\lambda/T_c$, so $\ell=1/\kappa=\sqrt{T_c/\lambda}$. A separate propagation convention may introduce a rate divided by $c_{\text{info}}$; it must not be identified with this diffusion coefficient without an additional derivation. The value equation is not a photon-mass calculation, and the correspondence needs its own units and boundary hypotheses.

In electrostatics, the Coulomb potential is $V(r) \sim 1/r$---long range. In a superconductor, the photon gains a mass $m_\gamma$, and the potential becomes $V(r) \sim e^{-m_\gamma r}/r$---short range, decaying exponentially beyond the screening length $\ell = 1/m_\gamma$.

Same here in the flat, free-space reference calculation. With $\gamma = 1$ (no discounting), conservative reward sources solve a Poisson equation, so a source at distance $r$ contributes like $1/r^{D-2}$ for $D>2$ (logarithmically in $D=2$). A curved or bounded domain has a different Green kernel, and the agent's ability to focus must be checked there.

With $\gamma < 1$, the selected diffusion operator has screening. Its Green response is controlled by $\ell=\sqrt{T_c/\lambda}$ in the stationary convention, subject to the geometry and boundary conditions. The agent can focus on local goals only when that response estimate applies.

But if $\gamma$ is too small, the screening length is too short. The agent becomes myopic, unable to see past its nose. A deer with $\gamma = 0.1$ would walk into the lion's mouth because it only cares about the next few meters.
:::

:::{admonition} Intuition: Discounting as Screening
:class: feynman-added tip

Think of conservative reward sources as electric charges distributed in spacetime. This is a mathematical analogy, not a claim
that the value field is an electromagnetic field. The value function $V(z)$ is like the electrostatic potential---it tells you how
much "pull" you feel toward different states.

In a flat, free-space Green-kernel picture with no discounting ($\gamma = 1$), there is no exponential screening. Boundary
conditions and geometry can still affect the solution, and a conservative reward source a million steps away need not be treated
like one next step away by every model.

With discounting ($\gamma < 1$), distant charges are screened. Their contribution decays exponentially with distance. You
feel mostly the nearby conservative rewards.

In the stationary diffusion convention, the screening length $\ell=\sqrt{T_c/\lambda}$ sets the value-field scale, with $\lambda=-\ln\gamma/\Delta t$. A propagation-based length such as $c_{\text{info}}\tau_{\text{proc}}/|\ln\gamma|$ belongs to a separate convention and cannot be substituted for $\ell$ without matching the underlying generator and units. For $\gamma=0.99$, the numerical scale therefore depends on $T_c$ and $\Delta t$; it is not universally “about 100 steps.”
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

:::{div} feynman-prose
This corollary ties together time and space. Your planning horizon (temporal: how far ahead you think) is bounded by your memory depth (spatial: how much history you can hold), but the two lengths must first be expressed in the same convention.

Why? Because planning requires imagining future states, and imagining future states requires composing transitions from past experience. If your memory holds only 10 transitions, you cannot reliably plan 1000 steps ahead---you do not have the data to model that far.

In the stationary diffusion convention, compare $\ell=\sqrt{T_c/\lambda}$ with the effective buffer depth after matching units. A causal propagation length built from $c_{\text{info}}$ is a separate bookkeeping scale. Do not infer a planning or memory bound by substituting one for the other without a generator-level relation.

In practice, this means: if you build an agent with limited memory ($L_{\text{buf}}$ small), the value-field scale produced by the selected generator should be comparably short. A myopic agent with short memory can be internally consistent; an agent trying to plan forever with finite memory needs an additional state or approximation argument.
:::



(sec-sieve-eigenvalue-system)=
## The Sieve Eigenvalue System

:::{div} feynman-prose
Now we put all the constraints together. Each section above gave us one or two inequalities. This section collects them into a single system and asks: Is there any setting of the parameters that satisfies all constraints simultaneously?

This is the feasibility question, and it has a beautiful geometric interpretation: each constraint defines a half-space in parameter space, and the feasible region is the intersection of all these half-spaces. If the intersection is non-empty, viable agents can exist. If it is empty, the constraints are mutually incompatible---no agent can satisfy all of them.

Spoiler: the intersection is non-empty. We know this because we exist.
:::

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
| IR Binding | $g_s(\mu_{\text{IR}}) \ge g_s^{\text{crit}}$ | DNN-B |
| UV Decoupling | $g_s(\mu_{\text{UV}}) \le \epsilon$ (for $\epsilon \to 0$) | 29 |
| Stiffness Lower | $\Delta E > T_c$ | 7 |
| Stiffness Upper | $\Delta E < \chi_{\text{max}} T_c$ | 7 |
| Discount Lower | $\gamma > \gamma_{\text{min}}$ | --- |
| Discount Upper | $\gamma < 1$ | --- |

:::

:::{div} feynman-prose
Look at this table. Ten constraints, each coming from a different physical requirement.

Some come from causality: you cannot send signals faster than $L_{\text{buf}}/\tau_{\text{proc}}$ or slower than $d_{\text{sync}}/\tau_{\text{proc}}$.

Some come from information theory: under the declared boundary, resolution, and channel convention, the capacity diagnostic cannot exceed its stated budget. The holographic wording is a model-level analogy until those counting and normalization hypotheses are supplied.

Some come from thermodynamics: you cannot think hotter than Landauer allows.

Some come from stability: features must bind at macro scale, decouple at micro scale.

Some come from cognition: memories must be stable but updatable, planning horizons must be finite but nonzero.

These constraints are not independent. They form a coupled system. Changing one parameter affects which values of other parameters are viable.
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

:::{div} feynman-prose
The existence proof is almost embarrassingly simple: *we are here*. If the feasible region were empty, no agents could exist to ask the question.

For the declared agent model, a witnessed configuration can show that its feasible set is nonempty. Applying that witness to the
physical universe requires the proposed dictionary to map physical constants and observables to the agent quantities and to verify
every hypothesis. Merely observing that agents exist does not establish that the physical constants satisfy this particular Sieve.

The deeper question is: *why* does a chosen agent implementation sit inside its feasible region? The Sieve answers this only at
the level of the declared constraints: settings satisfying them are candidates for viability, while settings violating them fail
the model's requirements. It does not select a physical universe or prove that the region is unique.

The observer-selection analogy can still be useful, but it remains an interpretation of the model rather than a physical theorem.
:::

:::{note}
:class: feynman-added
The shape of the feasible region is a model-dependent question. It may be thin if several independent constraints are active, but
the present formulation does not prove low dimension, near-saturation, or a thin shell of viable universes. Those claims require
rank, regularity, and sampling or measure assumptions for the declared agent parameterization.

Such a geometry could offer a way to study apparent fine-tuning within the agent model. It does not show that physical constants
are tuned, that the physical universe lies in this region, or that a tuner is unnecessary.
:::



(sec-optimization-problem)=
## The Optimization Problem

:::{div} feynman-prose
Now we ask a more refined question. The feasible region $\mathcal{F}$ may contain many points---many settings of fundamental constants that permit viable agents. Which one do we observe?

The hypothesis is: the one that maximizes some objective, subject to the constraints. This is constrained optimization.

What is the objective? We propose it trades off two things: representational power (how much the agent can know about the world) versus computational cost (how much energy it takes to run the agent). More representation is good. Lower cost is good. You cannot maximize both, so you pick a tradeoff.
:::

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

:::{div} feynman-prose
The objective $\mathcal{J}$ makes economic sense. You want to know as much as possible ($I_{\text{bulk}}$ high) while spending as little as possible ($\mathcal{V}_{\text{metabolic}}$ low).

The parameter $\beta$ sets the exchange rate: how many bits of knowledge are you willing to give up to save one joule of energy? This depends on the environment. In a resource-scarce environment, $\beta$ is large (energy is precious). In a resource-rich environment, $\beta$ is small (burn energy freely for more knowledge).

An adaptive agent may optimize this objective subject to the Sieve constraints. Extending that optimization metaphor to biological evolution or to a selection over universes is an additional physical hypothesis, not a consequence of the agent model.
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

:::{div} feynman-prose
Here is the key insight, when the additional monotonicity and boundary-attainment assumptions of the optimization argument hold: an optimum can sit on the boundary, with at least one constraint active. Continuity and compactness alone establish existence, not boundary saturation.

Why might that happen? If the objective improves along a feasible direction from every interior point, you can push toward the boundary and do better—either increase representational power or decrease cost. Without that directional-improvement property, an interior optimum is possible.

This supplies one possible explanation for apparent fine-tuning, but it does not show that physical constants are selected by this objective. The “right up to the edge” reading is a conjectural interpretation of the optimization model.

A bridge engineer may size beams near a design boundary after choosing loads and safety factors. That is a useful analogy for the constrained optimization; it does not establish that the constants of physics were engineered by the Sieve.
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

:::{div} feynman-prose
A Pareto surface is the set of "you cannot improve one thing without making another thing worse." If you are on the Pareto surface, any movement either violates a constraint or trades off one objective against another.

The conjecture is that our universe may lie near such a surface when viewed through an agent-viability model. The Sieve proves an optimization problem for the declared constraints; it does not prove that observed fundamental constants are Pareto-optimal for agency.

This is a testable conjecture, not a result of the Sieve. If the proposed physical identification is correct, one could look for several independently defined constraints that are nearly saturated. Apparent fine-tuning observations would be evidence to analyze only after the feasible set, priors, units, and uncertainty model had been fixed.
:::



(sec-physics-isomorphism-constants)=
## Physics Isomorphism: The Standard Model Constants

:::{div} feynman-prose
Finally, we translate. Everything we have derived uses agent-theoretic language: information speed, cognitive temperature, binding coupling. This section records a proposed dictionary to familiar physics constants; the dictionary is an isomorphism claim to test, not a theorem supplied by the notation.
:::

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

:::{div} feynman-prose
Some of these correspondences are suggestive. Information speed $\leftrightarrow$ speed of light compares two finite-propagation scales, while cognitive action scale $\leftrightarrow$ Planck constant compares two resolution scales. A physical identification needs compatible units, state maps, and observables.

Others are more surprising. The proposed pairing of the binding coupling $g_s$ with the strong-force coupling $\alpha_s$ invites a comparison between QCD confinement and cognitive feature-binding. The mechanisms are not thereby the same; the pairing remains a conjectural physical interpretation.

The most provocative is the last row: the discount factor $\gamma$ is paired with a cosmological-horizon scale. This suggests a hypothesis about finite causal horizons and finite planning horizons. It is highly speculative, and the agent mathematics alone does not validate the physical identification.
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

:::{div} feynman-prose
Let me say this plainly. The claim of this chapter is:

**The laws of physics are what they are because they are the laws that permit agents to exist.**

Not designed for agents. Not fine-tuned by a creator. But *constrained* by the requirements of agency, and *optimized* for representational power per unit metabolic cost.

This is a strong conjecture. It might be wrong. It is testable only after the feasible set, priors, units, and observational comparison have been specified: constants comfortably inside the set would count against a saturation story, while apparent proximity to several boundaries would be evidence to investigate rather than proof.

Claims about current observations and criticality require a separate empirical analysis of the relevant physical models. The Sieve chapter does not establish that the fine-structure constant, cosmological constant, or strong coupling lie at its own boundaries.

The Sieve can derive constraints for the declared agent model. It does not derive the observed constants, their physical mechanisms, or a unique viable solution without an additional isomorphism theorem and data. Those identifications remain conjectural.
:::



(sec-summary-parameter-sieve)=
## Summary

:::{div} feynman-prose
Let me gather the threads.

We started with a question: why do the fundamental constants have the values they do? The traditional answers are either "they just do" (unsatisfying) or "they were fine-tuned for life" (mysterious).

The Sieve offers a third question: what parameter values are compatible with cybernetic viability? Any agent---biological or artificial---must satisfy the consistency conditions of its own model. These conditions carve out a feasible region in parameter space. Our existence supplies one physical example, but does not prove that observed constants were selected by this region.

We derived six families of constraints:
1. **Causal**: Information cannot travel too fast or too slow.
2. **Holographic**: Under a declared boundary and cell-counting convention, an operational storage diagnostic scales with boundary area.
3. **Metabolic**: Cognition costs energy; temperature is capped.
4. **Hierarchical**: Features must bind at macro scale, decouple at micro scale.
5. **Stiffness**: Memories must be stable but updatable.
6. **Temporal**: Planning horizons must be finite but nonzero.

Each constraint can correspond to a Sieve node that monitors the declared quantity at runtime. A threshold violation is a diagnostic event whose consequence depends on the controller; it is not automatically catastrophic.

The feasible region is the intersection of the declared constraint sets. An agent can be tested against that region. Whether our universe lies on a Pareto-optimal surface, and whether its constants maximize representational power per unit cost, are physical conjectures rather than results of this construction.

This is not a complete theory of everything. It does not tell you why the constraints exist or derive physical constants from first principles. It offers a framework for comparing coherent-agent requirements with physical models; the proposed physics isomorphism needs separate mathematical and empirical support.
:::

This chapter has derived the constraints on fundamental constants from cybernetic first principles:

1. **Causal Consistency** (§35.2): Information speed bounded by buffer architecture
2. **Holographic Stability** (§35.3): Levin length determines capacity via Area Law
3. **Metabolic Viability** (§35.4): Cognitive temperature bounded by Landauer limit
4. **Hierarchical Coupling** (§35.5): Binding at IR, decoupling at UV (asymptotic freedom)
5. **Stiffness Window** (§35.6): Energy gaps between memory and flexibility
6. **Temporal Screening** (§35.7): Discount factor enables local goal-directedness

The Sieve Architecture (Nodes 2, 7, 29, 40, 52, 56, 62) enforces these constraints at runtime. The fundamental constants of physics are the coordinates of the feasible region's Pareto-optimal surface.

**Key Result:** The laws of physics are not arbitrary but are the solution to a cybernetic optimization problem. The universe we observe is the one that supports viable agents—not because it was designed for us, but because agents can only exist in regions of parameter space where the Sieve constraints are satisfied.
