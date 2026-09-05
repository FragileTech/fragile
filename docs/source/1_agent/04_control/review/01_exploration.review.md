# Mathematical Review: docs/source/1_agent/04_control/01_exploration.md

## Metadata
- Reviewed file: docs/source/1_agent/04_control/01_exploration.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (446 lines)
- Framework anchors (definitions/axioms/permits):
  - docs/source/1_agent/01_foundations/01_definitions.md:240 (action decomposition `a_t=(K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}})`)
  - docs/source/1_agent/01_foundations/02_control_loop.md:263-266 (macro codebook, `e_k`, VQ projection), 705-716 (value-curvature metric `G` at a latent point `z`), 1085-1103 (`def-causal-enclosure-condition`, macro kernel `\bar P`)
  - docs/source/1_agent/05_geometry/04_equations_motion.md:507 (`def-cognitive-temperature`)
  - docs/source/1_agent/04_control/03_coupling_window.md:43 (the volume's definition of the Schrödinger bridge)
  - In-chapter: `def-macro-path-distribution` (lines 65-85), `def-causal-path-entropy` (94-111), `def-maxent-rl-objective-on-macrostates` (159-177), `prop-soft-bellman-form-discrete-actions` (194-225), `thm-equivalence-of-entropy-regularized-control-forms-discrete-macro` (328-367)
  - Downstream users consulted: docs/source/1_agent/02_sieve/01_diagnostics.md:689, docs/source/1_agent/05_geometry/04_equations_motion.md:984

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 2
- Minor: 5
- Notes: 0
- Primary themes: (1) The main theorem claims an exact equivalence between discounted MaxEnt control / soft Bellman optimality and a single exponential tilt of the reference path measure. This fails for two independent reasons: the path tilt re-weights the stochastic macro kernel `\bar P` as well as the policy, and the path KL is undiscounted while the reward is discounted. Both failures were reproduced numerically on a small finite MDP by two independent scripts; equality holds only for deterministic `\bar P` and `\gamma=1`. (2) The chapter carefully distinguishes causal (policy-only) entropy `S_c` from Shannon path entropy, then does not respect that distinction in the theorem's KL-vs-entropy remark or in the naming of the exploration gradient. (3) The exploration gradient `\nabla_G S_c` is not well defined as stated, because `S_c` is a function on a finite set. (4) Several small notation issues (`H` as horizon and entropy; `a` vs `K^{\text{act}}`; "Schrödinger bridge" used for a problem without a terminal-marginal constraint; a missing additive constant in the log-normalizer identity).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Intro prose, Researcher Bridge, `def-exploration-gradient-metric-form` (31, 39, 130, 154) | Minor | Definition mismatch | Framework | this chapter | `S_c` described as "state-action path entropy" although it is policy-only entropy |
| E-002 | Kernel and objective displays, theorem hypothesis 1 (56-83, 159-170, 332) | Minor | Notation conflict | Framework | this chapter | `a\in\mathcal{A}` vs `K^{\text{act}}_t`, and finiteness of `\mathcal{A}`, clash with the upstream action decomposition |
| E-003 | `def-causal-path-entropy` (94-111) | Minor | Notation conflict | Framework | this chapter | `H` denotes both the horizon and the entropy operator in the same display |
| E-004 | `def-exploration-gradient-metric-form`, `def-exploration-gradient-covariant-form` (127-142, 304-313) | Moderate | Proof gap / omission | Framework | this chapter | `\nabla_G S_c(k,H;\pi)` requires a continuous extension of `S_c` that is never specified |
| E-005 | `thm-equivalence-...`, item 2 and variational identity (328-367); `conn-rl-22` (397-403) | Major | Invalid inference | Framework | this chapter | Path-space tilt of `P_0=\prod\pi_0\bar P` is not equivalent to soft Bellman optimality when `\bar P` is stochastic |
| E-006 | `thm-equivalence-...`, variational identity (340-363) | Major | Parameter inconsistency | Framework | this chapter | Undiscounted path KL against discounted reward; tilted optimum is time-inhomogeneous and differs from Forms 1/3 for `\gamma<1` |
| E-007 | `thm-equivalence-...`, remark after the identity (363) | Moderate | Invalid inference | Framework | this chapter | KL differs by a constant from causal entropy `S_c`, not from Shannon path entropy |
| E-008 | Theorem (351), "Why the Log-Normalizer Matters" (386), `conn-rl-22` (403) | Minor | Computational error | Framework | this chapter | `T_c\log Z = V^* - T_c H\log|\mathcal{A}|`; the additive constant is omitted |
| E-009 | `conn-rl-22` and following prose (403, 417, 426) | Minor | Conceptual | Framework | this chapter | Reward-tilted path measure called a "Schrödinger bridge" although no terminal marginal is prescribed |

## Detailed findings

### [E-001] `S_c` is named "state-action path entropy" but defined as causal (policy-only) entropy (was F-002)
- Location: lines 31, 39, 130, 154
- Severity: Minor
- Type: Definition mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 130: "Define the exploration gradient as the metric gradient of state-action path entropy: $\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi)$"; line 39: "we maximize the entropy of reachable macro state-action trajectories"; line 31: "The entropy of your future state-action trajectory distribution is a measure of that ability."
- Upstream anchor: in-chapter, lines 106-107: "Only policy randomness contributes; stochasticity in $\bar{P}$ does not add entropy credit."
- Why this is an error: By the chain rule applied to the path law of line 80, `\mathcal{H}(P_\pi(\cdot\mid k)) = S_c(k,H;\pi) + \sum_{h=0}^{H-1}\mathbb{E}_{P_\pi}[\mathcal{H}(\bar P(\cdot\mid K_{t+h},A_{t+h}))]`. The second term is nonzero whenever `\bar P` is stochastic (in a random 3-state, 2-action instance with `H=3`: Shannon path entropy 4.09 nats versus `S_c=1.45` nats). So the functional named in the definition's prose is not the functional in its display; the chapter's own point at lines 91, 106, 118-119 is that the two differ.
- Impact on downstream results: Feeds E-007 (the theorem remark that equates the KL term with Shannon path entropy up to a constant).
- Fix guidance:
  1. Line 130: write "the metric gradient of the causal path entropy $S_c$ (Definition `def-causal-path-entropy`)".
  2. Lines 31 and 39: say "the causal (policy-controlled) entropy of future macro state-action trajectories".
  3. Line 154: "We've defined causal path entropy ...".
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for "state-action path entropy" and "trajectory distribution" and confirm each remaining occurrence refers to `S_c` with the qualifier "causal".

### [E-002] Action symbol `a\in\mathcal{A}` vs `K^{\text{act}}`, and finiteness of `\mathcal{A}` (was F-007)
- Location: lines 56-83, 159-170, 332
- Severity: Minor
- Type: Notation conflict (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 59: "$\bar{P}(k'\mid k,a),\qquad k,k'\in\mathcal{K},\ a\in\mathcal{A}$"; line 167: "$\mathcal{R}(K_t,K^{\text{act}}_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))$"; line 332: "finite macro alphabet $\mathcal{K}$ and (for simplicity) finite action set $\mathcal{A}$".
- Upstream anchor: docs/source/1_agent/01_foundations/01_definitions.md:240: "actions decompose into structured components: $a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}})$ where $K^{\text{act}}_t$ is the discrete motor macro, $z_{n,\text{motor}}$ is motor nuisance (compliance), and $z_{\text{tex,motor}}$ is motor texture (tremor)."; docs/source/1_agent/01_foundations/02_control_loop.md:1088-1092 conditions causal enclosure on `(K_t,K^{\text{act}}_t)`.
- Why this is an error: Upstream, `a_t` is the full action with continuous residual components and `K^{\text{act}}_t` is its discrete macro; the enclosure kernel is conditioned on `K^{\text{act}}_t`. This chapter writes the kernel and the policy in `a\in\mathcal{A}`, the reward in `K^{\text{act}}_t`, mixes both inside the single objective at line 167, and assumes `\mathcal{A}` finite. If `\mathcal{A}` is the upstream action space it is not finite; if it is the range of `K^{\text{act}}`, this is never stated and the reward `\mathcal{R}(k,a)` must then be understood with motor residuals marginalized.
- Impact on downstream results: Determines whether the differential-entropy caveats (lines 83, 108-109) are needed and whether hypothesis 1 of the theorem is satisfiable in the framework.
- Fix guidance:
  1. After line 62 add: "In this chapter $\mathcal{A}$ denotes the finite motor-macro alphabet, the range of $K^{\text{act}}_t$, and we write $a\equiv K^{\text{act}}_t$; motor nuisance and texture are marginalized out of $\bar P$ and $\mathcal{R}$."
  2. Use one symbol consistently in lines 59-83, 167, 200, 345, 400.
- Required new assumptions/permits: the marginalization of motor residuals in `\bar P` and `\mathcal{R}` should be stated as an explicit modeling convention.
- Validation plan: check that every occurrence of `\mathcal{A}` in the chapter is compatible with a finite alphabet, and remove or re-scope the continuous-action caveats accordingly.

### [E-003] `H` used for both horizon and entropy operator (was F-001)
- Location: `def-causal-path-entropy`, lines 94-111
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 101-103: "$S_c(k,H;\pi) := \sum_{h=0}^{H-1} \mathbb{E}_{\xi\sim P_\pi(\cdot\mid k)} \left[ H\!\left(\pi(\cdot\mid K_{t+h})\right) \right]$"; line 109: "interpret $H(\pi(\cdot\mid k))$ as a differential entropy".
- Upstream anchor: in-chapter, line 170: "where $\mathcal{H}$ is Shannon entropy" (also lines 167, 200).
- Why this is an error: In one display `H\in\mathbb{N}` is the horizon (upper summation limit, second argument of `S_c`) and `H(\cdot)` is the entropy functional; the same functional is written `\mathcal{H}` in the formal statements that follow.
- Impact on downstream results: none mathematically; invites misreading.
- Fix guidance:
  1. Replace `H(\pi(\cdot\mid K_{t+h}))` by `\mathcal{H}(\pi(\cdot\mid K_{t+h}))` at line 103 and `H(\pi(\cdot\mid k))` by `\mathcal{H}(\pi(\cdot\mid k))` at line 109.
- Required new assumptions/permits: none.
- Validation plan: visual check of the rendered definition.

### [E-004] The exploration gradient `\nabla_G S_c(k,H;\pi)` is not a well-defined object (was F-003)
- Location: lines 127-142 and 304-313
- Severity: Moderate
- Type: Proof gap / omission (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 133: "$\mathbf{g}_{\text{expl}}(e_k) := T_c\ \nabla_G S_c(k,H;\pi)$"; line 136: "Operationally, gradients are taken through the continuous pre-quantization coordinates (straight-through VQ estimator); in the strictly symbolic limit, the gradient becomes a discrete preference ordering induced by $S_c(k,H;\pi)$."
- Upstream anchor: docs/source/1_agent/01_foundations/02_control_loop.md:263-266: "Let $\mathcal{K}=\{1,\dots,|\mathcal{K}|\}$ and let $\{e_k\}_{k\in\mathcal{K}}\subset\mathbb{R}^{d_m}$ be a learned codebook ... $K(x) := \arg\min_{k}\|z_e(x)-e_k\|_2^2$"; docs/source/1_agent/01_foundations/02_control_loop.md:705-716: the metric is defined "at a point $z$ in the latent space" from `\partial^2 V/\partial z_i\partial z_j`.
- Why this is an error: By lines 80 and 101-103, `S_c(k,H;\pi)` depends on `k` only through `\pi(\cdot\mid k)` and `\bar P(\cdot\mid k,a)`, both indexed by the discrete symbol. A function on a finite set has no gradient with respect to `e_k\in\mathbb{R}^{d_m}`, and the metric gradient `\nabla_G = G^{-1}\nabla_z` lives on the continuous latent. The straight-through estimator transports an already-existing gradient at the quantized code to `z_e(x)`; it does not define a gradient of a function that is constant on Voronoi cells. What is needed and not stated is a differentiable extension `\tilde S_c(z)` obtained by parameterizing `\pi(a\mid z)` and `\bar P(k'\mid z,a)` on the pre-quantization coordinate with `\tilde S_c(e_k)=S_c(k,H;\pi)`. The alternative at line 136 ("a discrete preference ordering") is an object of a different type, so the definition is not well typed as a single definition. The same object is restated verbatim at lines 304-313 without repair.
- Impact on downstream results: Line 351 ("Gradients of the log-normalizer therefore induce a well-defined exploration direction") and the proof sketch at line 365 inherit the gap. No other chapter in the volume references `\mathbf{g}_{\text{expl}}` or the two exploration-gradient labels directly, so the impact is local to this chapter.
- Fix guidance:
  1. State the continuous surrogate: assume `\pi_\theta(a\mid z)` and `\bar P_\phi(k'\mid z,a)` are smooth in `z\in\mathbb{R}^{d_m}` and define `\tilde S_c(z;H,\pi)` by the formula of `def-causal-path-entropy` with `k` replaced by `z` in the first step.
  2. Define `\mathbf{g}_{\text{expl}}(e_k) := T_c\,G^{-1}(e_k)\,\nabla_z\tilde S_c(z)\big|_{z=e_k}`.
  3. Move the "discrete preference ordering" alternative into a separate remark (it is a ranking of `k` by `S_c`, not a vector).
  4. Merge or cross-reference the two definitions (lines 127-142 and 304-313) so the object is defined once.
- Required new assumptions/permits: smoothness of the policy and kernel heads in the pre-quantization coordinate (already implicit in the straight-through training scheme).
- Validation plan: check that every later use of `\mathbf{g}_{\text{expl}}` treats it as a tangent vector at `e_k`, and that the surrogate reduces to `S_c` at the codes.

### [E-005] Path-space exponential tilt is not equivalent to soft Bellman optimality for stochastic `\bar P` (was F-004)
- Location: `thm-equivalence-of-entropy-regularized-control-forms-discrete-macro`, item 2 and the variational identity (lines 340-363); `conn-rl-22` (lines 397-403)
- Severity: Major
- Type: Invalid inference (secondary: Scope restriction)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 340-345: "The optimal controlled path law admits an exponential-family form relative to the reference measure induced by $\pi_0$ and $\bar{P}$: $P^*(\omega\mid K_t=k)\ \propto\ P_0(\omega \mid k)\,\exp\!\left(\frac{1}{T_c}\sum_{h=0}^{\infty}\gamma^h\,\mathcal{R}(K_{t+h},K^{\text{act}}_{t+h})\right)$"; lines 354-363: "$\log Z(k) = \sup_{P(\cdot\mid k)}\left\{\frac{1}{T_c}\,\mathbb{E}_{P}\!\left[\sum_{h=0}^{\infty}\gamma^h\,\mathcal{R}\right] -D_{\mathrm{KL}}(P(\cdot\mid k)\Vert P_0(\cdot\mid k))\right\}$, and the optimizer is exactly the exponentially tilted law $P^*$."
- Upstream anchor: the soft Bellman recursion it must match is in-chapter, lines 205-211: "$V^*(k) = T_c \log \sum_{a\in\mathcal{A}} \exp\!\left(\frac{1}{T_c}\left(\mathcal{R}(k,a)+\gamma\,\mathbb{E}_{k'\sim\bar{P}(\cdot\mid k,a)}[V^*(k')]\right)\right)$". Stochasticity of `\bar P` is a framework assumption: docs/source/1_agent/01_foundations/02_control_loop.md:1085-1103 defines it as a Markov kernel on `\mathcal{K}`; this chapter, line 91: "Stochasticity in $\bar{P}$ is not under the agent's control".
- Why this is an error: The supremum ranges over all path laws, and its optimizer `P^*\propto P_0\exp(\cdot)` tilts the whole path measure including the dynamics factors. Its induced transition law is `P^*(k_{h+1}\mid k_h,a_h,\dots)\propto\bar P(k_{h+1}\mid k_h,a_h)\,\mathbb{E}_{P_0}[e^{\text{future reward}/T_c}\mid k_{h+1}]`, which equals `\bar P` only when `\bar P(\cdot\mid k,a)` is a point mass. Hence `P^*` is not of the form `\pi\cdot\bar P` required by `def-macro-path-distribution`, and the policy it induces is not the `\pi^*` of Forms 1 and 3. The normalizer recursion of the tilt is `Z(k)=\sum_a\pi_0(a\mid k)e^{\mathcal{R}(k,a)/T_c}\,\mathbb{E}_{k'\sim\bar P}[Z_{\text{next}}(k')]` (a log-mean-exp of future values), whereas the soft Bellman recursion has `\exp(\gamma\,\mathbb{E}_{\bar P}[V^*]/T_c)`; by Jensen these differ for non-degenerate `\bar P`. The tilted form is the risk-seeking ("optimistic") control problem, not MaxEnt RL with fixed dynamics.
  Recomputation (two independent scripts, exact enumeration of paths, `|\mathcal{K}|=3`, `|\mathcal{A}|=2`, `H=3`, `T_c=1`, uniform `\pi_0`): maximum absolute difference between the soft-Bellman optimal policy at step 0 and the step-0 action marginal of the tilted path law is `2.45\times10^{-3}` (reviewer's seed, unit-scale rewards) and `6.6\times10^{-2}` (verifier's seed, rewards scaled by 3) for Dirichlet-random stochastic `\bar P` with `\gamma=1`, versus `10^{-16}` for deterministic `\bar P` with `\gamma=1`. The equivalence holds exactly only in the deterministic-dynamics case.
- Impact on downstream results: The "Consequence" of `prop-soft-bellman-form-discrete-actions` (lines 221-223), the sentence "the path-space log-normalizer is (up to scaling) the soft value" (line 351), the admonition at lines 383-391, and `conn-rl-22` (line 403) all rest on this identification. The theorem is cited from docs/source/1_agent/02_sieve/01_diagnostics.md:689 (KL-control regularizer, which uses the per-step policy KL and is therefore consistent with the corrected statement) and docs/source/1_agent/05_geometry/04_equations_motion.md:984 (cross-reference only).
- Fix guidance (choose one):
  1. Scope restriction: add the hypothesis "deterministic enclosure kernel `\bar P`" to item 2 and to the variational identity, and say so in `conn-rl-22`.
  2. Preferred: restrict the variational family to kernel-consistent laws `P=\pi\cdot\bar P`. Then `D_{\mathrm{KL}}(P\Vert P_0)=\sum_h\mathbb{E}_P[D_{\mathrm{KL}}(\pi(\cdot\mid K_h)\Vert\pi_0(\cdot\mid K_h))]`, the optimizer is `\pi^*\cdot\bar P` with `\pi^*` the soft-Bellman softmax, and the log-normalizer statement becomes "`T_c\log Z_\pi(k)=V^*(k)+\text{const}` where `Z_\pi` is the normalizer of the per-step policy tilt". Delete the displayed `P^*\propto P_0\exp(\cdot)` or label it as the deterministic-dynamics special case.
- Required new assumptions/permits: either deterministic `\bar P` (option 1) or an explicit restriction of the admissible path laws to those generated by a policy through the fixed kernel (option 2).
- Validation plan: rerun the enumeration check with the restricted family (or a deterministic kernel) and confirm the step-0 policies agree to machine precision; check that `conn-rl-22` and the admonition at lines 383-391 no longer claim the full-path tilt for stochastic dynamics.

### [E-006] Undiscounted path KL against discounted reward; tilted optimum is not stationary (was F-005)
- Location: `thm-equivalence-...`, lines 340-363
- Severity: Major
- Type: Parameter inconsistency (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 354-360 (identity with `\sum_h\gamma^h\mathcal{R}` and un-weighted `D_{\mathrm{KL}}(P\Vert P_0)`); line 348: "(For finite-horizon $H$, replace $\infty$ with $H-1$; the equivalence holds for any horizon.)"; line 363: "as $H \to \infty$, the finite-horizon solution converges to the stationary infinite-horizon optimum."
- Upstream anchor: in-chapter Form 1, line 167: "$\mathbb{E}_\pi\left[\sum_{t\ge 0}\gamma^t\left(\mathcal{R}(K_t,K^{\text{act}}_t) + T_c\,\mathcal{H}(\pi(\cdot\mid K_t))\right)\right]$" (entropy weighted by `\gamma^t`).
- Why this is an error: In Form 1 the entropy (equivalently, for uniform `\pi_0`, the per-step KL) carries the weight `\gamma^t`; in the variational identity the path KL is unweighted while the reward carries `\gamma^h`. For a kernel-consistent law the step-`h` term of the tilt Lagrangian is `\gamma^h\mathcal{R}/T_c-D_{\mathrm{KL}}(\pi_h\Vert\pi_0)`, i.e. a MaxEnt problem at effective temperature `T_c/\gamma^h`. The induced policy therefore relaxes to `\pi_0` as `h` grows: it is time-inhomogeneous (so "stationary infinite-horizon optimum" is wrong for the tilted law), and even for deterministic `\bar P` it differs from the stationary softmax `\pi^*` of Form 3 whenever `\gamma<1`.
  Recomputation (deterministic `\bar P`, `\gamma=0.5`, `H=3`): the tilted law's last-step conditional policy equals `\mathrm{softmax}(\gamma^{H-1}\mathcal{R}/T_c)` to `2\times10^{-16}` and differs from the soft-Bellman last-step policy `\mathrm{softmax}(\mathcal{R}/T_c)` by 0.22; the step-0 policies differ by `4.96\times10^{-3}` (reviewer's seed) and `8.8\times10^{-3}` (verifier's seed), versus `10^{-16}` at `\gamma=1`. The discount alone breaks the equivalence. (The infinite-horizon supremum itself is finite: the per-step KL of the optimizer decays like `\gamma^{2h}`, which is summable; the problem is the mismatch with Forms 1 and 3, not ill-posedness.)
- Impact on downstream results: Same as E-005. In addition, the claim that the finite-horizon exploration problem used by `def-causal-path-entropy` (finite `H`, undiscounted) and the discounted infinite-horizon objective share an optimal policy is unsupported, which bears on the "horizon as a knob" guidance at lines 445-446.
- Fix guidance (choose one):
  1. Set `\gamma=1` and finite `H` in Form 2. Then the correct Form 3 is the finite-horizon backward soft Bellman recursion with time-indexed `V_h^*` and `\pi_h^*`; drop "stationary".
  2. Keep `\gamma<1` and discount the KL: replace `D_{\mathrm{KL}}(P\Vert P_0)` by `\sum_h\gamma^h\,\mathbb{E}_P[D_{\mathrm{KL}}(\pi_h(\cdot\mid K_h)\Vert\pi_0(\cdot\mid K_h))]`. The optimizer is then a product of per-step tilts with step-dependent normalizers, not a single exponential tilt of the reference path measure; revise the displayed `P^*` and the "exponential-family form" wording accordingly.
- Required new assumptions/permits: none beyond choosing one of the two consistent conventions.
- Validation plan: with the chosen convention, rerun the enumeration check at `\gamma=0.5` and confirm agreement of the step-0 and last-step policies to machine precision.

### [E-007] KL differs by a constant from causal entropy `S_c`, not from Shannon path entropy (was F-006)
- Location: line 363
- Severity: Moderate
- Type: Invalid inference (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 363: "In the special case where $P_0$ is uniform (or treated as constant), the KL term differs from Shannon path entropy by an additive constant, recovering the standard 'maximize entropy subject to expected reward' view."
- Upstream anchor: in-chapter, line 348: "$P_0(\omega \mid k) := \prod_{h=0}^{\infty}\pi_0(K^{\text{act}}_{t+h}\mid K_{t+h})\,\bar{P}(K_{t+h+1}\mid K_{t+h},K^{\text{act}}_{t+h})$"; lines 106-107: "Only policy randomness contributes; stochasticity in $\bar{P}$ does not add entropy credit."
- Why this is an error: `P_0` is not uniform on `\Gamma_H(k)` unless `\bar P` is uniform. For a kernel-consistent `P_\pi=\pi\cdot\bar P` with uniform `\pi_0` and finite `H`, the `\bar P` factors cancel in `\log P_\pi-\log P_0`, so `D_{\mathrm{KL}}(P_\pi\Vert P_0)=\sum_h\mathbb{E}[\log\pi(A_h\mid K_h)-\log\pi_0(A_h\mid K_h)]=H\log|\mathcal{A}|-S_c(k,H;\pi)`. The KL therefore differs by a constant from the causal entropy, not from the Shannon path entropy `\mathcal{H}(P_\pi)=S_c+\sum_h\mathbb{E}_{P_\pi}[\mathcal{H}(\bar P(\cdot\mid K_h,A_h))]`, whose second term depends on `\pi` through the visited state-action pairs. Numerical check (stochastic `\bar P`, random `\pi`, `H=3`): `D_{\mathrm{KL}}=0.63085`, `H\log|\mathcal{A}|-S_c=0.63085`, Shannon path entropy `4.09`. In the infinite-horizon case the "constant" `H\log|\mathcal{A}|` diverges, so the remark only makes sense at finite `H` (see E-006). The correct statement is the one the chapter wants elsewhere: KL control maximizes expected reward plus `T_c\,S_c`, which is exactly why `def-causal-path-entropy` excludes dynamics randomness.
- Impact on downstream results: The identification "soft value = log-partition function" (lines 386, 419) and the reading of the exploration gradient as a gradient of the log-normalizer (lines 351, 365) depend on which entropy appears; using Shannon path entropy reintroduces the error of E-005.
- Fix guidance:
  1. Replace the sentence at line 363 by: "For uniform $\pi_0$ and finite $H$, $D_{\mathrm{KL}}(P_\pi\Vert P_0)=H\log|\mathcal{A}|-S_c(k,H;\pi)$, so the KL-regularized problem is 'maximize expected reward plus $T_c$ times the causal path entropy of Definition `def-causal-path-entropy`'."
  2. Restrict the supremum to kernel-consistent laws (E-005, option 2) so the identity applies.
  3. Align the proof sketch at line 365 ("maximize path entropy") with causal entropy.
- Required new assumptions/permits: none.
- Validation plan: symbolic check of the cancellation of `\bar P` factors; numerical check as above.

### [E-008] Log-normalizer equals the soft value only up to an additive constant as well as a scaling (was F-008)
- Location: lines 351, 386, 403
- Severity: Minor
- Type: Computational error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 386: "The log-normalizer $\log Z(k)$ in the variational identity is the soft value function $V^*(k)$ (up to a temperature-dependent scaling)."; line 403: "The path-space log-normalizer equals the soft value".
- Upstream anchor: in-chapter, lines 354-363 and 165-169.
- Why this is an error: In the setting where the identification holds (deterministic `\bar P` with `\gamma=1`, or the kernel-consistent family with uniform `\pi_0` and finite `H`), E-007 gives `T_c\log Z(k)=\sup_\pi\{\mathbb{E}[\sum_h\mathcal{R}]-T_c(H\log|\mathcal{A}|-S_c)\}=V^*_H(k)-T_c\,H\log|\mathcal{A}|`. Numerical check (deterministic `\bar P`, `\gamma=1`, `H=3`): `|T_c\log Z-(V^*_0-T_c H\log|\mathcal{A}|)|=8.9\times10^{-16}`. The constant is harmless for policies and gradients, but the stated equality is false, and line 390 proposes extracting values from `Z(k)`. In the infinite-horizon form with undiscounted KL the constant is infinite.
- Impact on downstream results: none for the optimal policy; matters wherever `\log Z` is used as a numerical value.
- Fix guidance:
  1. Write "$T_c\log Z(k)=V^*(k)-T_c\,H\log|\mathcal{A}|$ (uniform $\pi_0$, horizon $H$)", or define `Z` relative to the normalized reference measure so that the constant vanishes.
- Required new assumptions/permits: none.
- Validation plan: recompute on the deterministic example.

### [E-009] "Schrödinger bridge" is a misnomer for the reward-tilted path measure (was F-009)
- Location: lines 403, 417, 426
- Severity: Minor
- Type: Conceptual (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 403: "This is a **Schrödinger bridge** formulation."; line 417: "The optimal policy is a Schrödinger bridge between prior and reward-weighted measures".
- Upstream anchor: docs/source/1_agent/04_control/03_coupling_window.md:43: "Given a reference dynamics ... and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals."
- Why this is an error: The volume's own definition of a Schrödinger bridge is the KL projection of a reference path measure onto the set of path measures with prescribed initial and terminal marginals; its solution has the two-potential form `f(\omega_0)g(\omega_T)P_0`. The problem in this chapter fixes only the initial state and tilts by an additive path functional `\sum_h\gamma^h\mathcal{R}`; its solution is a Gibbs reweighting (KL control / linearly solvable control) with no terminal-marginal constraint. The description at line 426 ("the state-action path distribution closest to your reference that achieves a certain expected reward") is a Gibbs variational principle with a reward constraint, not a bridge.
- Impact on downstream results: framing only; no formal result depends on it. The coupling-window chapter uses the term correctly, so the two chapters currently disagree on what the term means.
- Fix guidance:
  1. Replace "Schrödinger bridge" at lines 403, 417, 426 by "KL-control (exponential-tilt / Gibbs) variational principle", or add the terminal-marginal constraint if a bridge is actually intended and cross-reference `03_coupling_window.md`.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for "bridge" and confirm each occurrence is either removed or accompanied by a terminal-marginal constraint.

## Scope restrictions and clarifications
- Item 2 of the equivalence theorem and the variational identity hold as stated only for deterministic `\bar P` and `\gamma=1` (finite `H`). For stochastic `\bar P` the admissible family must be restricted to kernel-consistent laws `\pi\cdot\bar P`; for `\gamma<1` the KL must be discounted or the horizon made finite with `\gamma=1`.
- The exploration gradient is defined only once a continuous surrogate `\tilde S_c(z)` of `S_c` on the pre-quantization coordinate is fixed.
- Throughout the chapter `\mathcal{A}` should be read as the finite motor-macro alphabet (range of `K^{\text{act}}`), which should be stated once.

## Open questions
- Which convention does the author intend for Form 2: finite horizon with `\gamma=1` (matching `def-causal-path-entropy`) or discounted infinite horizon with discounted KL (matching `def-maxent-rl-objective-on-macrostates`)? The fix to E-006 and the "horizon as a knob" guidance depend on the answer.
- Is the "risk-seeking" full-path tilt (log-mean-exp over `\bar P`) ever intended in the framework (for example as an optimistic exploration variant)? If so it deserves its own statement rather than being conflated with MaxEnt control.

## Rejected candidate findings
- None. All nine stage-1 findings were confirmed; three had their criterion or a sub-claim adjusted (F-004, F-005: criterion External changed to Framework; F-005: the "supremum not well-posed" sub-claim was dropped because the optimizer's KL is finite; F-009: criterion changed to Framework with an in-volume anchor).
