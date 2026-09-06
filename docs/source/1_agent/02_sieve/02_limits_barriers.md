(sec-limits-barriers)=
(sec-4-limits-barriers-the-limits-of-control)=
# Limits: Barriers (The Limits of Control)

## TLDR

- Barriers are **hard limit surfaces**: failure modes you cannot “optimize through” with better tuning.
- Each barrier names a **mechanism** (actuator saturation, information limits, compute horizon, mixing traps, spectral
  gaps, etc.) and a corresponding **regularizer / intervention**.
- The “periodic table” is meant as a **diagnostic index**: when something breaks, identify the barrier and apply the
  matching remedy.
- Barriers complement the Sieve diagnostics: diagnostics tell you *what is happening*; barriers tell you *why it must
  happen* and what trade-off surface you are hitting.
- Expensive barriers (✗) often require approximations/offline checks; cheap barriers (✓) should run continuously.

## Roadmap

1. Catalog the barrier family and how to read the table.
2. Provide implementation notes: what to compute online vs. offline.
3. Connect barriers to the intervention chapter (what to do when a barrier activates).

:::{div} feynman-prose
Here is a question that ought to bother you: if we have a good policy, a good world model, and a good critic, why would the control loop ever fail? The answer is that some limits are built into the problem, while other rows below are operational barriers that identify a failing regime. A hard limit cannot be tuned away; an operational barrier tells you when the current design or approximation needs to change.

Suppose you are driving at 60 mph and an obstacle appears 10 feet ahead. No matter how perfect your reflexes, no matter how sophisticated your planning, you are going to hit it. The physics simply does not permit otherwise. That is a barrier - a hard limit imposed by the structure of the problem, not by your intelligence.

This section catalogs different ways a control system can hit such walls. Some are hard limits under explicit hypotheses; others are operational barriers or diagnostic proxies that mark a regime where the current design should stop, project, or change. Actuators, information, computation, and model class all matter. Understanding which kind of statement you are reading is part of using the table correctly.
:::

(rb-barriers-trust-regions)=
:::{admonition} Researcher Bridge: Barriers vs. Trust Regions
:class: warning
Standard RL uses trust regions, clipping, or penalty terms to avoid instability. Barriers are the formal limit surfaces those heuristics approximate. When a barrier activates, the correct response is to halt, project, or reshape updates rather than incur a soft penalty.
:::

Barriers represent the fundamental limits of the control loop.

:::{div} feynman-prose
The table below is a periodic table of failure modes. Each row describes a different way your control system can hit a wall. Here is how to read it:

- **Barrier ID**: A short name for this failure mode.
- **Bottleneck**: Which component (Policy, World Model, Critic, or VQ-VAE) gets stuck.
- **Limit**: The fundamental constraint being violated.
- **Mechanism**: Why things break down - the physical or computational reason.
- **Constraint / regularizer**: either a hard constraint or a differentiable loss used to stay away from this barrier.
- **Compute**: How expensive it is to monitor or enforce this constraint.

One detail prevents a common misdiagnosis: **BarrierScat** denotes the dispersion/saturation edge of the coupling window, where the macro posterior is too diffuse or $H(K)$ is too close to its upper limit. It is not codebook collapse; collapse is the opposite liveness problem. **Node 13 BoundaryCheck**, which is referenced in the diagnostics chapter, tests $I(X;K)>0$ as a non-collapse sanity check and does not by itself certify predictive grounding.

Do not memorize all of these. Use the table as a reference when something goes wrong. Ask: "Which barrier did I hit, under which hypotheses?" The answer tells you what to measure and what to fix.
:::

| Barrier ID         | Name                    | Bottleneck        | Limit                             | Mechanism                                                                                     | Constraint / Regularizer ($\mathcal{L}_{\text{barrier}}$)                                                             | Compute        |
|--------------------|-------------------------|-------------------|-----------------------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------|
| **BarrierSat**     | Saturation              | **Policy**        | **Actuator Saturation**           | Policy cannot output enough control authority to counter disturbance.                         | $a=F_{\text{max}}\tanh(u)$ componentwise, so $\lVert a\rVert_\infty\le F_{\text{max}}$ (architectural squashing; include the change-of-variables term when evaluating $\log\pi(a\mid z)$)                                                              | $O(BA)$ ✓      |
| **BarrierCausal**  | Causal Censor           | **World Model**   | **Computational Horizon**         | Failure happens faster than WM can predict/compute.                                           | $T_{\text{horizon}}$ (Discount Factor $\gamma < 1$)                                                                | $O(1)$ ✓       |
| **BarrierScat**    | Symbol Dispersion / Grounding | **VQ-VAE**        | **Grounding Loss**                | Symbol channel loses grounding or the macro posterior becomes too diffuse.                                | $\mathrm{ReLU}(\epsilon_I-I(X;K))^2 + \mathrm{ReLU}(H(p_t)-(\log\lvert\mathcal{K}\rvert-\epsilon_H))^2$ (Coupling-window penalty) | $O(B)$ ✓       |
| **BarrierTypeII**  | Type II Exclusion       | **Critic/Policy** | **Scaling Mismatch**              | $\beta_\pi>\alpha$ (Policy update scale outruns critic signal).                                   | $\max(0, \beta_\pi - \alpha)$ (Scaling Penalty)                                                                        | $O(P)$ ⚡       |
| **BarrierVac**     | Model Stability Limit   | **World Model**   | **Regime Stability**              | Operational mode is metastable; WM predicts collapse.                                         | $\Vert \nabla^2 V(z) \Vert$ (Hessian Regularization)                                                               | $O(BZ^2)$ ✗    |
| **BarrierCap**     | Capacity                | **Policy**        | **Fundamental Uncontrollability** | Unsafe region is too large for Policy to steer around.                                         | $V(z) \to \infty$ for $z \in \text{Bad}$ (Safe RL)                                                                 | $O(B)$ ⚡       |
| **BarrierGap**     | Spectral Gap            | **Critic**        | **Convergence Stagnation**        | Error surface is too flat ($\nabla_A V \approx 0$).                                             | $\max(0, \epsilon - \lVert \nabla_A V \rVert_G)$ (Stiffness)                                                             | $O(BZ)$ ✓      |
| **BarrierAction**  | Action Gap              | **Critic**        | **Cost Prohibitive**              | Correct move requires more cost budget ($V$) than affordable.                                 | $\Vert \nabla_\pi V(s, \pi) \Vert$ (Action Gradient)                                                               | $O(BAZ)$ ⚡     |
| **BarrierOmin**    | Lipschitz / Sensitivity               | **World Model**   | **Model Mismatch**                | World exhibits non-smooth or non-stationary structure outside the WM class.                   | $\lVert\nabla S_t\rVert$ (Lipschitz / sensitivity proxy)                                                              | $O(ZP_{WM})$ ⚡ |
| **BarrierMix**     | Mixing                  | **Policy**        | **Exploration Trap**              | Policy converges to a local minimum with insufficient state coverage.                                                            | $-H(\pi)$ (Entropy Bonus)                                                                                          | $O(BA)$ ✓      |
| **BarrierEpi**     | Epistemic               | **VQ-VAE/WM**     | **Information Overload**          | Environment ({prf:ref}`def-environment-as-generative-process`) complexity exceeds $\log\lvert\mathcal{K}\rvert$ and/or WM class; closure breaks. | $\mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{Sync}_{K-W}}$ (Distortion + Closure)                              | $O(BD)$ ✓      |
| **BarrierFreq**    | Frequency               | **World Model**   | **Loop Instability**              | Positive feedback causes oscillation amplification.                                           | $\Vert J_{WM} \Vert < 1$ (Jacobian Spectral Norm)                                                                  | $O(Z^2)$ ✗     |
| **BarrierBode**    | Bode Sensitivity        | **Policy**        | **Waterbed Effect**               | In an LTI single-loop approximation, suppressing error in one domain increases it in another.                                      | $\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \pi\sum_k\operatorname{Re}p_k$ over open-loop unstable poles (zero for a stable open loop of relative degree at least two)              | FFT ✗          |
| **BarrierInput**   | Input Stability         | **All**           | **Resource Exhaustion**           | Agent runs out of battery/compute/tokens.                                                     | $\text{Cost}(s) > \text{Budget}$ (Resource Penalty)                                                                | $O(B)$ ✓       |
| **BarrierVariety** | Requisite Variety       | **Policy**        | **Ashby's Deficit**               | Policy actuation capacity is sufficient for the disturbance process.                                                           | $\log|\mathcal K^{\mathrm{act}}| \ge H(D)$, where $D$ is the disturbance macro (Actuator Variety)                                                                    | $O(1)$ ✓       |
| **BarrierLock**    | Exclusion               | **World Model**   | **Hard-Coded Safety**             | Safety interlock successfully prevents illegal state.                                         | $\mathbb{I}(s \in \text{Forbidden}) \cdot \infty$                                                                  | $O(B)$ ✓       |

**Compute Legend:** ✓ Low (typically online) | ⚡ Moderate (often amortized/approximated) | ✗ High (often offline or coarse approximations)

:::{note}
:class: feynman-added
Notice the engineering distinction in the legend: barriers marked ✓ are cheap enough for online monitoring, while ✗ barriers usually need offline or coarse approximations. That mark describes compute cost, not danger or mathematical strength. A cheap check can still be decisive, and an expensive proxy can still miss a failure outside its assumptions.
:::

(sec-barrier-implementation-details)=
## Barrier Implementation Details

:::{div} feynman-prose
Knowing these barriers exist is nice, but what do we actually *do* about them? This section answers that in two parts.

First, single barriers - cases where one constraint is violated. These are the easier problems: clear culprit, clear fix. Actuator saturating? Squash the output. Critic too slow? Pause policy updates. Simple cause, simple cure.

The second part is where things get dangerous. Sometimes two barriers push against each other. Compressing your representation helps avoid one failure mode but makes another more likely. Stabilizing your world model helps in some ways but hurts plasticity. These are genuine dilemmas with no perfect solution, only trade-offs. The art of control design is navigating these trade-offs wisely.
:::

Implementing these barriers requires rigorous cybernetic engineering. We divide them into **Single-Barrier Limits** and **Cross-Barrier Dilemmas**.

(sec-a-single-barrier-enforcement)=
### A. Single-Barrier Enforcement (Hard Constraints)

:::{div} feynman-prose
The philosophy here matters: we are not punishing the system for approaching barriers; we are making it *structurally impossible* to violate them. Think of the difference between a "Do Not Enter" sign and a solid wall. The sign can be ignored; the wall cannot.

This is why we use `tanh` to squash policy outputs rather than penalizing large actions. The `tanh` function physically cannot output values outside $[-1, 1]$, no matter what the network learns. Build the constraint into the architecture, not the loss function.
:::

1.  **BarrierSat (Actuator Limit):**
    *   *Constraint:* $\lVert\pi(s)\rVert \le F_{\max}$.
    *   *Implementation:* **Squashing Function**. Sample an unconstrained action $u\sim\pi_\theta(\cdot\mid z)$ and set $a=F_{\max}\tanh(u)$ componentwise. Include the change-of-variables term $-\sum_i\log(1-\tanh^2u_i)$ in the action log-density; this enforces $\lVert a\rVert_\infty\le F_{\max}$ exactly.

2.  **BarrierTypeII (Scaling Mismatch):**
    *   *Constraint:* $\alpha > \beta_\pi$ (Critic is steeper than Policy).
    *   *Implementation:* **Two-Time-Scale Updating**.
        *   If $\text{Scale}(\text{Critic}) \le \text{Scale}(\text{Policy})$, **skip** the Policy update step ($k_\pi = 0$).
        *   Resume policy updates only when the Critic has re-established a valid gradient (restored a usable value landscape).

:::{div} feynman-prose
The critic tells the policy "go this way, it is better over there." But if the policy updates faster than the critic can evaluate, the policy runs blind - chasing stale gradients. It is like navigating by a map that is always one step behind where you actually are.

The fix requires discipline: when the critic falls behind, stop updating the policy. Let the critic catch up. Resume policy training only when you have reliable value estimates. Pausing feels wasteful, but training in the wrong direction is far more wasteful.
:::

3.  **BarrierOmin (Tameness):**
    *   *Constraint:* $\lVert S\rVert_{\mathrm{Lip}} \le K$.
    *   *Implementation:* **Spectral normalization** bounds the operator norm of each linear layer. With 1-Lipschitz activations, this upper-bounds the network Lipschitz constant by the product of per-layer spectral norms; choose per-layer caps so the implied global bound is $\le K$ {cite}`miyato2018spectral`.

:::{div} feynman-prose
What does "tame" or "o-minimal" mean? Roughly: the function cannot do anything too wild - no infinitely fast oscillations, no fractal behavior, no pathological surprises. A Lipschitz constraint says: change the input by a small amount, the output changes by at most $K$ times that amount.

Why care? Our world model predicts the future, and if predictions are too sensitive to small perturbations, they become useless. A tiny error in your state estimate explodes into a huge error in your predicted future. Spectral normalization bakes this smoothness constraint into the architecture by controlling the largest singular value of each weight matrix.
:::

4.  **BarrierGap (Spectral Gap):**
    *   *Constraint:* $\lVert\nabla_A V\rVert_G \ge \epsilon$ (No flat plateaus).
    *   *Implementation:* **Gradient Penalty**.

        $$
        \mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} [\max(0,\epsilon-\lVert\nabla_A V(\hat{s})\rVert_G)^2]

        $$
        Gradient-norm penalties discourage vanishing gradients on sampled points and help avoid large flat regions; they do not provide a global guarantee without additional assumptions {cite}`gulrajani2017improved`.

:::{div} feynman-prose
This barrier is about having a value landscape you can navigate. Imagine finding the lowest-cost direction while blindfolded---your only information is the local slope. If the covariant gradient is too small on the states you visit, you get little directional information; that is the local stiffness condition used by BarrierGap.

The one-sided gradient penalty penalizes slopes below the threshold. It can improve local responsiveness on sampled states, but it does not guarantee a global spectral gap or a well-behaved landscape without additional assumptions.
:::

(sec-b-cross-barrier-regularization)=
### B. Cross-Barrier Regularization (Cybernetic Dilemmas)

The most dangerous failures occur when barriers conflict. We model these as **Trade-off Functionals**:

:::{div} feynman-prose
Now we come to the genuinely hard problems. Each dilemma below represents a fundamental tension - you cannot satisfy both sides fully, so you must choose where on the trade-off curve to live. There is no "correct" answer; the right balance depends on your application.

These are not bugs you can fix with cleverness. They are like the uncertainty principle in quantum mechanics: a fundamental limit on what you can achieve simultaneously.
:::

1.  **The Information-Control Tradeoff (BarrierScat vs BarrierCap):**
    *   *Classes:* **Rate-Distortion Optimization.**
    *   *Conflict:* The posterior-dispersion edge of the coupling window can conflict with BarrierCap: bits added for controllability may increase posterior uncertainty while the policy is being adapted. Marginal code usage is monitored separately for liveness.
    *   *Regularization:*

        $$
        \mathcal{L}_{\text{InfoControl}}
        =
        \underbrace{\beta_K\,\mathbb{E}[-\log p_\psi(K)] + \beta_n D_{\mathrm{KL}}(q(z_n \mid x)\Vert p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x)\Vert p(z_{\mathrm{tex}}))}_{\text{Compression (Rate)}}
        +
        \underbrace{\lambda_{\mathfrak{D}}\,\mathbb{E}[\mathfrak{D}(Z,A)]}_{\text{Control Effort}}

        $$
        where {math}`\mathfrak{D}` is an actuation cost (e.g. KL-control to a prior {math}`\pi_0`, or a calibrated norm/penalty on actions).
    *   *Mechanism:* Use Lagrange multipliers to find the Pareto frontier. If control performance drops, decrease $\beta_K,\beta_n,\beta_{\mathrm{tex}}$ (allocate more bits to the shutter).

:::{div} feynman-prose
The dilemma in plain terms: you want to compress observations into a compact representation - this helps generalization and prevents overfitting to noise. But compression means throwing away information. Sometimes exactly what you threw away is what you needed for a fine control decision.

Think of a thermostat that only knows "hot" or "cold." Great for simple temperature control - you do not need five decimal places. But if you need to maintain a chemical reaction at exactly 37.2 degrees, that binary representation is catastrophically insufficient.

The loss function balances these concerns: the first term rewards compression, the second penalizes control effort. When control starts struggling, decrease the compression coefficients ($\beta$) to let more information through.
:::

2.  **BarrierVac vs. World-Model Plasticity (volatility scale $\gamma_{\mathrm{wm}}$ / forward-consistency drift, Node 5):**
    *   *Conflict:* A stable World Model (model stability limit) resists updating to new dynamics (plasticity / Zeno).
    *   *Regularization:* **Elastic Weight Consolidation (EWC)**.

        $$
        \mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,old})^2

        $$
    *   *Mechanism:* The Fisher Information Matrix $F_i$ quantifies parameter sensitivity. Updates are permitted for low-sensitivity weights while high-sensitivity (structurally important) weights are constrained.

:::{div} feynman-prose
This is catastrophic forgetting seen from control theory. Your world model needs stability - it should not wildly change predictions every time it sees new data. But it also needs plasticity - it should update when the world genuinely changes.

The trouble: how do you distinguish "the world changed" from "I saw noisy data"? Too much stability and you cannot adapt to genuine changes. Too much plasticity and you forget what you learned yesterday.

Elastic Weight Consolidation uses Fisher Information to identify which weights matter for things you already know, then penalizes changes to those weights more strongly. It says: "learn new things, but try not to break existing skills." The Fisher Information tells you which weights are load-bearing - changing them would damage existing capabilities most.
:::

3.  **The Sensitivity Integral (BarrierBode):**
    *   *Conflict:* Suppressing error in one frequency band amplifies it in another (Bode sensitivity integral constraint: $\int_{0}^{\infty} \log |S(j\omega)| d\omega = \pi\sum_k\operatorname{Re}p_k$ over open-loop unstable poles, hence $0$ for a stable open loop of relative degree at least two).
    *   *Regularization:* **Frequency-Weighted Cost**.

        $$
        \mathcal{L}_{\text{Bode}} = \lVert \mathcal{F}(e_t) \cdot W(\omega) \rVert^2

        $$
    *   *Mechanism:* Explicitly decide *where* to be blind. We penalize high-frequency errors heavily (instability) while accepting low-frequency drift (steady-state error), or vice versa.

:::{div} feynman-prose
This is my favorite dilemma because it comes from a beautiful theorem in classical control theory. In an LTI single-loop setting, with the stability and relative-degree hypotheses stated in the table, the Bode sensitivity integral fixes the signed area of $log|S(j\omega)|$ from the open-loop pole data. You cannot reduce sensitivity everywhere; you can only move it around within that theorem's scope.

Practically: suppose, within that LTI approximation, you build a controller that suppresses sensitivity in one frequency band. The waterbed effect says the integral constraint must be paid elsewhere, with the exact balance depending on unstable poles and relative degree. A nonlinear, time-varying neural policy is not covered automatically.

The question becomes: where do you want to be sensitive, where can you afford blindness? For a robot arm, you might care about high-frequency stability (no oscillations) but tolerate slow drift. For climate control, priorities might reverse. The frequency-weighted cost $W(\omega)$ encodes these priorities; it is a design regularizer, not a replacement for checking the Bode hypotheses.
:::
