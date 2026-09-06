(sec-implementation-note-entropy-regularized-optimal-transport-bridge)=
# Implementation Note: Entropy-Regularized Optimal Transport Bridge

## TLDR

- The Schrödinger / entropy-regularized OT bridge is a **path-space view of KL control**: find the closest trajectory
  measure to a reference dynamics subject to boundary beliefs.
- Operationally: it gives you a principled way to connect priors/posteriors (or start/end distributions) without ad-hoc
  interpolation.
- Use it as an implementation tool: it explains why soft policy iteration behaves like a transport problem on
  trajectories.
- This chapter is optional but clarifies the coupling-window theorem by making “regularized control” geometrically
  explicit.

## Roadmap

1. Set up the bridge problem (reference dynamics + endpoint constraints).
2. Show the KL-control equivalence and what it means for implementation.

:::{div} feynman-prose
Before we get into the coupling-window criterion, I want to give you a beautiful piece of optional machinery. It's not required for what follows, but if you understand it, you'll see the whole story from a different angle---a path-space angle.

Here's the setup. You have a reference dynamics---say, your learned macro kernel $\bar{P}$, or a diffusion on your latent space. You have two "boundary conditions": a prior belief (where you think you are) and a posterior belief (where the observations say you are). The question is: what's the most natural way to connect these two?

The answer comes from optimal transport theory: find the path measure that's closest (in KL divergence) to your reference dynamics, subject to matching the boundary conditions. This is called a **Schrödinger bridge**, and it's the rigorous version of "most likely flow under entropy regularization."

Why should you care? KL-regularized control is the one-endpoint specialization of this path-space problem. Soft policy iteration and SAC fix the initial law while leaving the terminal law free; imposing a terminal marginal as well gives the full Schrödinger bridge. The distinction matters when the endpoint belief is part of the specification.
:::

(rb-kl-control-bridge)=
:::{admonition} Researcher Bridge: KL Control as a Schrödinger Bridge
:class: tip
Entropy-regularized control can be read as an optimal transport problem on trajectories. This is the same math behind soft policy iteration, just framed as a path-measure bridge.
:::

:::{div} feynman-prose
This is optional machinery, but it provides a clean path-space view of KL-regularized control and filtering.
:::

:::{admonition} The Schrödinger Bridge Picture
:class: feynman-added note

**Entropic bridge (Schrödinger bridge) viewpoint.** Given a reference dynamics (e.g. the macro kernel $\bar{P}$, or a continuous diffusion on $\mathcal{Z}_\mu$) and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals. This is the rigorous "most likely flow under entropy regularization" principle (entropic optimal transport) {cite}`cuturi2013sinkhorn,leonard2014schrodinger`.

In the Fragile Agent:
- the reference process is the world model's internal rollout,
- the boundary observations induce marginal constraints (via the shutter),
- the Sieve imposes feasibility constraints (via projections),

so each training update can be read as an entropic optimal transport step on belief trajectories.
:::



(sec-theorem-the-information-stability-threshold)=
## Theorem: The Information-Stability Threshold (Coupling Window)

:::{div} feynman-prose
Now we arrive at one of the central operating criteria: the **coupling window**. This is the measurable version of the Goldilocks principle I mentioned earlier. Your agent should be coupled to its boundary strongly enough to stay grounded in reality, while monitoring whether posterior uncertainty is becoming too diffuse.

Let me give you the physical picture first, then we'll state it precisely.

Imagine your agent as a spinning top. The boundary observations are like a hand that occasionally taps the top to keep it aligned. Too few taps, and the top wobbles off into some random orientation---this is **ungrounded inference**. Too many taps, too hard, and the top never settles into a stable spin at all---this is **symbol dispersion**.

The coupling window is the range of tap frequencies and strengths where the top remains aligned with the external reference while its posterior retains structure. Outside this operating range, a diagnostic or penalty calls for intervention; the definition itself is not a stability theorem.

What makes this criterion useful is that the quantities involved are **measurable**. We're not asking you to verify some abstract condition. We estimate the mutual information between observations and macro-states (the grounding signal), the entropy of the macro posterior (the mixing/dispersion signal), and their windowed rates. If a threshold is missed, a specific diagnostic or penalty identifies what needs attention.
:::

(rb-stable-learning-window)=
:::{admonition} Researcher Bridge: The Stable Learning Window
:class: warning
The coupling window is the stability region where representation and dynamics stay grounded. For RL readers, it plays the role of a learning-rate and discount range where updates contract rather than diverge.
:::

:::{div} feynman-prose
The coupling-window view gives an operational **window condition**: coupling must be strong enough to remain grounded (BoundaryCheck) while posterior uncertainty remains below the dispersion threshold. It is a monitored regime, not a standalone necessity or sufficiency theorem.

We state this as a rate balance rather than an ill-typed scalar comparison.
:::

:::{prf:definition} Grounding rate
:label: def-grounding-rate

Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). For a time window or minibatch, the *grounding rate* is the corresponding average information inflow per step:

$$
\lambda_{\text{in}} := \mathbb{E}[G_t].

$$
Units: $[\lambda_{\text{in}}]=\mathrm{nat/step}$.

:::

:::{div} feynman-prose
What is this $\lambda_{\text{in}}$, really? It's the time-window or minibatch average of how much your observations tell you about your macro-state. If $\lambda_{\text{in}}$ is high, the estimated boundary channel is informative about which macro-symbol is active. If it is low, the observations carry little estimated information about macro-state identity.

Think of it as the bandwidth of the "reality channel" into your agent. You need this channel to have enough capacity to correct drift in your internal model.
:::

:::{prf:definition} Mixing rate
:label: def-mixing-rate

Let $p_t\in\Delta^{|\mathcal K|-1}$ be the macro posterior from the belief update and set $S_t:=H(p_t)$. The *mixing rate* is the average positive growth of posterior uncertainty:

$$
\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+].

$$
Units: $[\lambda_{\text{mix}}]=\mathrm{nat/step}$.

:::

:::{div} feynman-prose
And $\lambda_{\text{mix}}$? That's the rate at which the **posterior** macro distribution is becoming more uncertain, averaged over the selected time window or minibatch. Some increase can be a legitimate consequence of exploration or ambiguous observations. The rate alone does not identify its cause, so interpret it together with the grounding signal and the update diagnostics before calling it a failure.

The $(\cdot)_+$ notation means we only count positive entropy changes. We're interested in how fast the distribution spreads, not how fast it concentrates.
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

:::{div} feynman-prose
Let me unpack this operational definition in plain language.

**The two inequalities** say:
1. **You must have grounding**: $I(X_t; K_t) \ge \epsilon_I$ means the estimated observation/macro coupling carries at least $\epsilon_I$ nats per sampled step. If this fails, you're flying blind---your observations aren't telling you much about which macro-state is active.

2. **You must monitor posterior dispersion**: $H(p_t) \le \log|\mathcal{K}| - \epsilon_H$ means the macro posterior is kept away from a uniform distribution. If this fails, the current belief is nearly indifferent among all macro-states; that is a posterior-dispersion signal, distinct from marginal code-usage entropy.

**The rate condition** $\lambda_{\text{in}}\ge\lambda_{\text{mix}}-\delta$ says the estimated grounding inflow should not fall more than the named slack $\delta$ below positive posterior-entropy growth. It is a local balance used for monitoring; it does not by itself imply long-time stability.

**The failure modes** are diagnostic gold:
- **Mode D.C (Decoupling)**: Your observations stopped being informative. Maybe your encoder broke. Maybe the environment changed in a way your shutter can't detect. Either way, you're ungrounded.
- **BarrierScat (Dispersion)**: Your posterior is spread over nearly everything. Maybe your updates are too aggressive. Maybe the evidence is ambiguous. Either way, the current belief is not selecting a coherent macro-state.
:::

:::{admonition} Why This Operational Criterion is Useful
:class: feynman-added tip

Notice what's special here: both conditions are **auditable at runtime**. You can estimate $I(X_t; K_t)$ from boundary samples and the encoder, and compute $H(p_t)$ from the current macro posterior. Marginal code-usage entropy $H(K_t)$ is a separate liveness statistic. No hidden environment state is required for the operational checks, although mutual-information estimates still require data and calibration.

This is by design. The definition is meant to be *operational*---something you can actually check, not a theoretical guarantee that requires omniscience to verify.
:::

::::{admonition} Connection to RL #9: Conservative Q-Learning as Soft Coupling Window
:class: note
:name: conn-rl-9
**The General Law (Fragile Agent):**
The **Coupling Window** (Definition {prf:ref}`thm-information-stability-window-operational`) specifies an information-flow operating range:

$$
\epsilon_I \le I(X_t; K_t) \quad \text{and} \quad H(p_t) \le \log|\mathcal{K}| - \epsilon_H.

$$
Node 13 (BoundaryCheck) gates on $I(X;K)>0$ at WARN/HALT level. The quantitative window is enforced by the differentiable penalty $\mathcal L_{\rm window}$ in the Sieve, so an offline-data violation is logged and penalized rather than described as an automatic halt.

**The Degenerate Limit:**
Replace the window constraint by a finite-weight penalty in the training objective, allowing a controlled trade-off between information regularization and task return.

**The Special Case (Standard RL - CQL):**
Conservative Q-Learning {cite}`kumar2020conservative` adds a penalty for out-of-distribution actions:

$$
\min_Q\; \alpha\,\mathbb{E}_{s\sim\mathcal D}\!\left[\log\!\sum_{a}\exp Q(s,a)-\mathbb{E}_{a\sim\hat\pi_\beta(\cdot\mid s)}Q(s,a)\right]
 +\frac12\,\mathbb{E}_{(s,a,s')\sim\mathcal D}\!\left[(Q(s,a)-\mathcal B^\pi\hat Q(s,a))^2\right].

$$
This softly penalizes overestimation on unseen actions but **does not prevent** the agent from taking them.

**Result:** A finite Lagrange multiplier turns a hard window constraint into a soft penalty. CQL is analogous in form---it penalizes unsupported value estimates---but it is not a limit of the coupling-window definition.

**What the generalization offers:**
- **Boundary gate**: Node 13 can warn or halt when the basic condition $I(X;K)>0$ fails; the quantitative window remains a soft training penalty
- **Auditable thresholds**: $I(X_t; K_t)$ is computed at runtime; failures are logged with specific diagnostic codes
- **Information-theoretic quantities**: the monitored thresholds use $I$ and entropy rather than a value-function surrogate; the DPI supplies upper bounds on $I$, not the lower grounding threshold
- **Bidirectional protection**: Both under-coupling (ungrounded) and over-coupling (dispersion) are detected and blocked
::::

:::{div} feynman-prose
The connection to Conservative Q-Learning is illuminating. CQL says: "penalize Q-values for actions you haven't seen data for." That is analogous to asking a learner not to be overconfident beyond its support. In the Fragile implementation, Node 13 can warn or halt when the basic grounding check fails, while the quantitative coupling window is enforced as a differentiable penalty. A finite penalty lets training continue while recording the violation.

Which response is appropriate depends on the intervention policy. A gate is available for the basic BoundaryCheck failure; finite penalties provide a softer response for quantitative threshold violations and make the trade-off visible during training.
:::


(sec-summary-unified-information-theoretic-control-view)=
## Summary: Unified Information-Theoretic Control View

:::{div} feynman-prose
Let me step back and show you the big picture. We've been building up a framework that unifies several different perspectives---geometry, boundaries, exploration, belief dynamics, optimality---under a single information-theoretic roof. Here's how the pieces fit together:
:::

:::{div} feynman-added
| Level | Formalism | Law |
| :--- | :--- | :--- |
| Geometry | Riemannian $(\mathcal{Z},G)$ | Distance measured by a sensitivity metric (Fisher/Hessian) |
| Boundary | Markov blanket $B_t$ ({prf:ref}`def-boundary-markov-blanket`) | Environment = boundary law $P_{\partial}$ ({ref}`sec-definitions-interaction-under-partial-observability`) |
| Exploration | Causal entropy / MaxEnt RL | Agent-controlled reachability pressure via policy path entropy on $\mathcal{K}$ ({ref}`sec-intrinsic-motivation-maximum-entropy-exploration`) |
| Belief Dynamics | Filtering + projection | Predict - update - project ({ref}`sec-belief-dynamics-prediction-update-projection`) |
| Optimality | Soft Bellman / log-normalizer | Under the finite-horizon deterministic-kernel convention, soft value = log-normalizer plus the stated uniform-prior offset; exploration gradient from causal path entropy ({ref}`sec-correspondence-table-filtering-control-template`) |
:::

:::{div} feynman-prose
Each row in this table is a different lens on the same underlying system:

**Geometry** tells you how to measure distances---not in Euclidean coordinates, but in a way that respects the sensitivity structure of your state space. Two points are "far apart" if small perturbations at one don't look like small perturbations at the other.

**Boundary** defines what "the environment" even means. It's not an object with hidden state you can peek at; it's a conditional law relating your actions to your observations. Everything you know about the world passes through this membrane.

**Exploration** is about keeping options open. The path entropy on $\mathcal{K}$ measures how many futures are reachable from where you are. Higher entropy means more freedom, more ability to adapt to unexpected circumstances.

**Belief Dynamics** is the engine that keeps your internal model synchronized with reality. Predict what you expect to see, update based on what you actually see, project away anything that violates your constraints.

**Optimality** ties it all together. In the finite-horizon deterministic-kernel setting, the soft Bellman value and the exponentially tilted path log-normalizer are dual descriptions, with the uniform-prior offset stated in the exploration chapter. Outside those hypotheses, entropy-regularized control remains useful but the exact path identity needs its own assumptions.

The Fragile Agent is a system that implements all of these layers, with explicit capacity limits and safety constraints woven throughout.
:::

:::{admonition} The Fragile Conclusion
:class: feynman-added important

The agent is a Bounded-Rationality Controller ({prf:ref}`def-bounded-rationality-controller`) with explicit information and stability constraints. The operational coupling window (Definition {prf:ref}`thm-information-stability-window-operational`) records a regime with grounding above $\epsilon_I$ and posterior entropy below $\log|\mathcal K|-\epsilon_H$, together with the stated rate slack. Outside it, the system reports under-coupling or excessive posterior mixing for intervention.

This is not a bug to be fixed; it's a feature to be monitored. The boundaries of the coupling window tell you exactly where your agent's competence ends.
:::
