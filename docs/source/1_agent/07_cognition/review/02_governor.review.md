# Mathematical Review: docs/source/1_agent/07_cognition/02_governor.md

## Metadata
- Reviewed file: `docs/source/1_agent/07_cognition/02_governor.md`
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (957 lines)
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/02_control_loop.md:32` (notation table: $G(z)$ is a state-space metric, not a parameter-space metric); `:695-708` (`rb-beyond-adam`, `def-state-space-sensitivity-metric`)
  - `10_appendices/04_faq.md:39` (manifold separation $\mathcal Z$ vs. $\Theta$)
  - `02_sieve/01_diagnostics.md:815-900` (Method A projected dual ascent, Method B PID update, Method C learned precisions); `:203` (volatility exponent $\gamma$); `:156` (`sec-theory-thin-interfaces`)
  - `10_appendices/05_proofs.md:945-1008` (E.10, proof of `cor-varentropy-brake`)
  - `06_fields/02_reward_field.md:757-786` (`cor-varentropy-stability`, definition and units of $V_H$)
  - `06_fields/03_info_bound.md:636-690` (diagnostic node registry, Node 42 row)
  - `03_architecture/03_optimization.md:172-192` (`prop-varentropy-brake-discrete`)
  - `07_cognition/03_memory_retrieval.md:955-995` (`prop-optimal-nonlocal-coupling`, uses of the Governor)
  - `10_appendices/06_losses.md:848-856` (regret $J$ sourced from this chapter)
  - `03_architecture/02_disentangled_vae.md:259`, `05_geometry/04_equations_motion.md:347, 507` ($\lambda_{\text{jump}}$, `def-cognitive-temperature`)

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 7
- Minor: 9
- Notes: 1
- Primary themes:
  1. The convergence result (`thm-stable-training-trajectory`) imports LaSalle's principle without its hypotheses, defines the target set circularly, and identifies stationary points of a finite quadratic penalty with KKT points of the constrained problem; the descent-direction corollary inherits the defect. The merit function is also mislabelled an "augmented Lagrangian" although it has no multiplier term, which is exactly the term that would repair the KKT claim.
  2. The Varentropy Brake corollary states an ODE schedule that the cited appendix does not prove; E.10 establishes an inequality with a different $V_H$ scaling, and the "basin of the global minimum" claim is not derived anywhere.
  3. The controlled update law imports the state-space metric $G$ as a parameter-space metric, against the explicit disclaimers at its source, and the mixed preconditioning is dimensionally inconsistent; several unit declarations ($\eta$, $\gamma_{\text{viol}}$, $\mu_k$, the normalized defect) do not close.
  4. The "subsumption" proposition misdescribes the upstream Primal-Dual, PID and Learned-Precision methods relative to their definitions in the Sieve chapter.
  5. The training protocol asks the Governor to adjust quantities ($\mu_k$, $\lambda_{\text{jump}}$, texture firewall) outside its declared output space, and the RL reward $-\Delta V_{\mathfrak L}$ telescopes to a terminal objective rather than the stated regret $J$.

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | `rem-extending-section` 59-69; `prop-subsumption-of-section` 235-248 | Moderate | Definition mismatch | Framework | this chapter | Primal-Dual/PID/Learned Precisions misdescribed; subsumption not verified |
| E-002 | lines 75, 147, 944; code 778-857 | Minor | Citation / reference error | Framework | this chapter | Legacy "26.x.y" and "Section 3.4" numbers do not resolve |
| E-003 | `def-uncontrolled-dynamics` 91-104; 221; 669 | Minor | Dimensional mismatch | Framework | this chapter | $[\eta]=\mathrm{step}^{-1}$ inconsistent with the update law |
| E-004 | `def-controlled-update-law` 129-149 | Moderate | Definition mismatch | Framework | this chapter | State-space $G$ used as parameter-space metric; mixed preconditioning ill-typed |
| E-005 | `def-diagnostic-state-space` 195; `def-the-universal-governor` 210-223 | Minor | Notation conflict | Framework | this chapter | $\mathbb R^{K\times H}$ vs. $H+1$ snapshots; "nodes 1–41" |
| E-006 | `def-outer-problem-governor-optimization` 284-299; `def-training-lyapunov-function` 373-388; `prop-dimensional-analysis` 661-671 | Moderate | Dimensional mismatch | Framework | this chapter | $\mathrm{nat}^2$ penalties with dimensionless weights; normalized defect double-subtracts $\epsilon_k$ |
| E-007 | `def-outer-problem-governor-optimization` 293 | Minor | Notation conflict | Framework | this chapter | Outer constraint feeds the Governor only $\Psi(\theta_t)$, not the $H$-window |
| E-008 | line 320; `def-training-lyapunov-function` 386 | Minor | Miswording | External | this chapter | Quadratic penalty called an "augmented Lagrangian" |
| E-009 | `def-training-lyapunov-function` 386 | Minor | Miswording | External | this chapter | Definition asserts convergence from $\Delta V<0$ alone |
| E-010 | `thm-stable-training-trajectory` 408-421; 535; 911 | Major | Invalid inference | External | this chapter | LaSalle without hypotheses; circular $\Omega$; penalty stationary points are not KKT points |
| E-011 | `cor-existence-of-descent-direction` 437-446 | Moderate | Invalid inference | External | this chapter | Fails at infeasible minimizers of $V_{\mathfrak L}$; proof argues about the wrong function |
| E-012 | `cor-varentropy-brake` 456-477 | Major | Proof gap / omission | Framework | this chapter | Stated ODE and basin claim are not what E.10 proves |
| E-013 | `cor-varentropy-brake` 459-469 | Minor | Notation conflict | Framework | this chapter | $\gamma$, $\eta$ collisions; $V_H(\theta_t)$ aggregation; "Nash equilibrium" |
| E-014 | `pi-lyapunov` 537; Node 42 table 937; code 850-869 | Note | Notation conflict | Framework | this chapter | Three names/forms for one monitor |
| E-015 | `prop-structure-of-diagnostic-inputs` 583-595 | Minor | Miswording | Framework | this chapter | Listed inputs do depend on the data law; gradient norm called spectral norm |
| E-016 | `prop-transfer-via-meta-generalization` 610-641 | Moderate | External dependency | External | this chapter | Source absent; bound forces $J=0$ on $\mathcal M$; no complexity term |
| E-017 | `def-canonical-obstruction-suite` 699-705; Node 42 remedy 941 | Moderate | Algorithm mismatch | Framework | this chapter | Remedies use knobs outside $\Lambda_t$; online $\mu_k$ changes break $\Delta V_{\mathfrak L}$ |
| E-018 | `def-canonical-obstruction-suite` remark 707 | Moderate | Algorithm mismatch | Framework | this chapter | Reward $-\Delta V_{\mathfrak L}$ telescopes; does not implement $J$ |
| E-019 | Table 26.9.1, line 927 | Minor | Typo | Framework | this chapter | ReLU dropped from the regret |

## Detailed findings

### [E-001] Subsumption proposition misrepresents the upstream methods (was F-004)
- Location: `rem-extending-section` (lines 59-69) and `prop-subsumption-of-section` (lines 235-248)
- Severity: Moderate
- Type: Definition mismatch (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim: line 63 "3.5.A (Primal-Dual): $\lambda_{t+1} = \Pi[\lambda_t + \eta_\lambda (C(\theta_t) - \epsilon)]$ — linear, memoryless"; line 64 "3.5.B (PID): $\lambda_{t+1} = K_p e_t + K_i \sum e + K_d \Delta e$"; line 242 "Primal-Dual (3.5.A) | $\pi_{\mathfrak{G}}(s_t) = \lambda_{t-1} + \eta_\lambda s_t$ (affine, $H=1$)"; line 244 "Learned Precisions (3.5.C) | Diagonal, no temporal dependence, $H=0$"; line 246 "*Proof.* Direct verification. The Primal-Dual update is a memoryless affine map."
- Upstream anchor: `02_sieve/01_diagnostics.md:825-826` "$\lambda_i \leftarrow \Pi_{[0,\lambda_{\max}]}\!\left(\lambda_i + \eta_{\lambda}\,(\mathcal{C}_i(\theta)-\epsilon_i)\right)$"; `:866-878` "$\lambda_{t+1} = \Pi_{[\lambda_{\min},\lambda_{\max}]}\!\Big(\lambda_t + K_p e_t + K_i \sum_{t' \le t} e_{t'} + K_d(e_t-e_{t-1})\Big)$"; `:893-895` "introduces learnable $s_i = \log \sigma^2_i$. The effective weight is $\exp(-s_i)$".
- Why this is an error: Unrolling the dual step gives $\lambda_t=\Pi[\cdots\Pi[\lambda_0+\eta_\lambda s_0]\cdots+\eta_\lambda s_{t-1}]$, an integrator over the whole residual history; it is not a function of a finite window $s_{t:t-H}$ and the projection makes it non-affine. The table entry uses $\lambda_{t-1}$, which is not among the inputs declared at line 213 (residuals only). The PID formula at line 64 omits the $\lambda_t$ term and the projection present upstream, and its integral has unbounded horizon, so no finite $H$ recovers it without recurrent state. Learned precisions $\exp(-s_i)$ are parameters trained by gradient descent on the inner loss; they are not a function of the diagnostic residuals, so they are not an instance of $\pi_{\mathfrak G}(s_t)$ with $H=0$. The proof restates the claims without verifying them.
- Impact on downstream results: The subsumption claim is repeated in the summary table (line 928), the code docstring (lines 773-776), and `conn-rl-23`.
- Fix guidance:
  1. Give the policy an internal state: $(\Lambda_t,h_t)=\pi_{\mathfrak G}(s_t,h_{t-1};\phi)$ (the GRU already does this).
  2. Primal-Dual: $h_t=\Pi[h_{t-1}+\eta_\lambda s_t]$, $\Lambda_t=h_t$. PID: a linear state-space filter with state (integral, last error, $\lambda_t$) per constraint. Learned precisions: the constant policy $\Lambda_t\equiv\exp(-s)$ with $s$ trained by the inner objective, presented as a degenerate rather than a special case.
  3. Correct the PID formula at line 64 to match `01_diagnostics.md:866-878`.
- Required new assumptions/permits: none.
- Validation plan: Check each row of the table against the upstream formula by direct substitution.

### [E-002] Stale numeric cross-references (was F-015)
- Location: lines 75, 147, 944 ("Section 3.4", "Section 3.1", "Section 2.3"); code docstrings lines 778, 791, 800, 826, 832, 857
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim: line 778 "References: Definition 26.3.2, Proposition 26.3.3"; line 857 "(Definition 26.5.1)"; line 75 "Section 3.4 (Joint Optimization)".
- Upstream anchor: the chapter's results are labelled `def-*`, `prop-*`, not numbered; `sec-joint-optimization` is in `02_sieve/01_diagnostics.md`; `sec-the-bridge-rl-as-lyapunov-constrained-control` is in `01_foundations/02_control_loop.md`.
- Why this is an error: The numbers are legacy chapter numbers (the registry in `06_fields/03_info_bound.md:685` still uses "26.9"), but the built book has no "Definition 26.3.2" and the pointers cannot be followed. No mathematics depends on them.
- Impact on downstream results: none.
- Fix guidance: Replace with `{prf:ref}` labels (`def-the-universal-governor`, `prop-subsumption-of-section`, `def-controlled-update-law`, `def-training-lyapunov-function`) and `{ref}` section labels.
- Validation plan: Build the docs and confirm all references resolve.

### [E-003] Units of the step size are inconsistent with the update law (was F-001)
- Location: `def-uncontrolled-dynamics` (lines 91-104); repeated at lines 221 and 669
- Severity: Minor
- Type: Dimensional mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 97 "$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}_{\text{task}}(\theta_t)$"; line 102 "Units: $[\theta] = \text{parameter units}$, $[\eta] = \text{step}^{-1}$, $[\nabla\mathcal{L}] = \text{nat} \cdot [\theta]^{-1}$."
- Why this is an error: For $\eta\nabla\mathcal L$ to have units $[\theta]$ one needs $[\eta]=[\theta]^2\,\mathrm{nat}^{-1}$ (Euclidean case) or dimensionless (when $\eta$ multiplies a preconditioned gradient that already has units $[\theta]$). With $[\eta]=\mathrm{step}^{-1}$ the right-hand side cannot be added to $\theta_t$. The code (line 803, "η_t / η_base ∈ (0, 2)") outputs a dimensionless multiplier, which is the consistent reading.
- Impact on downstream results: The unit bookkeeping in `prop-dimensional-analysis` (line 669).
- Fix guidance: State $[\eta]=[\theta]^2\,\mathrm{nat}^{-1}$ (Euclidean) or dimensionless (natural-gradient), or make the Governor output the ratio $\eta_t/\eta_{\text{base}}$ as the code does and say so in the definition.
- Validation plan: Re-derive the units of every term in lines 97 and 135.

### [E-004] State-space metric $G$ imported as a parameter-space metric; update law dimensionally inconsistent (was F-002)
- Location: `def-controlled-update-law` (lines 129-149)
- Severity: Moderate
- Type: Definition mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: line 135 "$\theta_{t+1} = \theta_t - \eta_t \left( G^{-1}(\theta_t) \nabla \mathcal{L}_{\text{task}}(\theta_t) + \sum_{k=1}^K \lambda_{k,t} \nabla C_k(\theta_t) \right)$"; line 139 "$G(\theta)$ is the parameter-space metric (cf. natural gradient, {ref}`sec-second-order-sensitivity-value-defines-a-local-metric`)"; line 143 "Units: $[\lambda_k] = \text{dimensionless}$."
- Upstream anchor: `01_foundations/02_control_loop.md:32` "| $G(z)$ | State-space metric (sensitivity “mass” / preconditioner), not a parameter-space metric |"; `:698` "Standard optimizers like Adam or K-FAC use the Fisher Information of the **Parameter Manifold ($\Theta$)** to scale updates. ... The Fragile Agent introduces a metric $G$ on the **Latent State Manifold ($\mathcal{Z}$)**."; `10_appendices/04_faq.md:39` "The metric $G$ ... operates on the **state manifold** $\mathcal{Z}$ ..., not the parameter manifold $\Theta$".
- Why this is an error: (a) The chapter defines $G(\theta)$ on $\mathcal M_\Theta$ and cites as its source a section that defines $G$ on $\mathcal Z$ and whose notation table says it is not a parameter-space metric; no metric on $\mathcal M_\Theta$ is defined anywhere in Volume 1. (b) If $G$ were a parameter-space metric with $[G]=\mathrm{nat}\,[\theta]^{-2}$, then $G^{-1}\nabla\mathcal L$ has units $[\theta]$ while $\lambda_k\nabla C_k$ has units $\mathrm{nat}\,[\theta]^{-1}$ (dimensionless $\lambda_k$, $[C_k]=\mathrm{nat}$ by line 197); the two terms cannot be summed. (c) The natural gradient of the inner objective at line 273 is $G^{-1}(\nabla\mathcal L+\sum_k\lambda_k\nabla C_k)$, not the mixed form.
- Impact on downstream results: `cor-existence-of-descent-direction` (E-011); `07_cognition/03_memory_retrieval.md:961-989`, which instantiates the Governor's control law.
- Fix guidance:
  1. Introduce a parameter-space preconditioner the framework actually defines (e.g. the parameter Fisher $F_\theta$ with units $\mathrm{nat}\,[\theta]^{-2}$), or use the Euclidean update.
  2. Precondition both terms: $\theta_{t+1}=\theta_t-\eta_t F_\theta^{-1}\big(\nabla\mathcal L+\sum_k\lambda_k\nabla C_k\big)$.
  3. Remove or reword the cross-reference to `sec-second-order-sensitivity-value-defines-a-local-metric` as an analogy only.
- Required new assumptions/permits: a definition of $F_\theta$ (positive definite) if the preconditioned form is kept.
- Validation plan: Unit check of line 135 after the edit; confirm `03_memory_retrieval.md` still reads consistently.

### [E-005] Input dimension and node range (was F-003)
- Location: `def-diagnostic-state-space` (line 195); `def-the-universal-governor` (lines 210-223)
- Severity: Minor
- Type: Notation conflict (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim: line 210 "$\pi_{\mathfrak{G}}: \mathbb{R}^{K \times H} \to \mathbb{R}_+^{K+2}$"; line 213 "$\Lambda_t = \pi_{\mathfrak{G}}(s_t, s_{t-1}, \ldots, s_{t-H}; \phi)$"; line 195 "diagnostic nodes 1–41".
- Upstream anchor: `06_fields/03_info_bound.md:685` "| 42 | [GovernorStabilityCheck](#node-42) | 26.9 |" and rows through Node 47 and beyond; `07_cognition/03_memory_retrieval.md:956` has the Governor "acting on the diagnostic residuals of Nodes 43 and 53".
- Why this is an error: The argument list has $H+1$ entries, so the domain is $\mathbb R^{K\times(H+1)}$; with the stated domain the "$H=0$" case (line 244) has empty input. "Nodes 1–41" excludes the nodes that later chapters feed to the Governor, so $K$ is under-specified.
- Impact on downstream results: `prop-subsumption-of-section` case table; code docstring `[B, T, K]`.
- Fix guidance: Write $\pi_{\mathfrak G}:\mathbb R^{K\times(H+1)}\to\mathbb R_+^{K+2}$ (or index the window $s_{t-H+1:t}$); replace "nodes 1–41" by "the $K$ Sieve nodes used as constraints (see the registry in `sec-diagnostic-node-registry`)".
- Validation plan: Count arguments at line 213 against the domain.

### [E-006] Units of penalty terms; normalized defect formula (was F-005)
- Location: `def-outer-problem-governor-optimization` (lines 284-299); `def-training-lyapunov-function` (lines 373-388); `prop-dimensional-analysis` (lines 661-671)
- Severity: Moderate
- Type: Dimensional mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim: line 197 "Units: $[s_t] = \text{nat}$ (for entropy-based nodes) or dimensionless"; line 290-295 "$J(\phi) = \mathbb{E}[\sum_t(\mathcal{L}_{\text{task}} + \gamma_{\text{viol}} \sum_k \text{ReLU}(C_k)^2)]$ ... $[\gamma_{\text{viol}}] = \text{dimensionless}$"; line 379-384 "$V_{\mathfrak{L}} = \mathcal{L}_{\text{task}} + \sum_k \frac{\mu_k}{2} \max(0, C_k)^2$ ... $[\mu_k] = \text{dimensionless}$"; line 667 "Normalized defects: $(C_k - \epsilon_k)/\epsilon_k$".
- Upstream anchor: line 116 (this chapter) "$C_k(\theta) = \text{Node}_k(\theta) - \epsilon_k$".
- Why this is an error: For an entropy-based node $\mathrm{ReLU}(C_k)^2$ has units $\mathrm{nat}^2$ and cannot be added to a nat-valued loss with a dimensionless coefficient; $\gamma_{\text{viol}}$ and $\mu_k$ must carry $\mathrm{nat}^{-1}$ for such nodes, or all $C_k$ must first be made dimensionless. The normalized defect at line 667, with $C_k=\mathrm{Node}_k-\epsilon_k$, equals $(\mathrm{Node}_k-2\epsilon_k)/\epsilon_k$ (check: $\mathrm{Node}=3$, $\epsilon=1$ gives $C=2$ and $(C-\epsilon)/\epsilon=1$), so the threshold is subtracted twice and "positive = violated" shifts to $\mathrm{Node}_k>2\epsilon_k$; it is also undefined for $\epsilon_k=0$.
- Impact on downstream results: Node 42 proxy; summary table (lines 926-927); `10_appendices/06_losses.md:853`.
- Fix guidance:
  1. Define $\tilde C_k:=(\mathrm{Node}_k-\epsilon_k)/\max(\epsilon_k,\epsilon_{\min})$ once in `def-constrained-dynamics`.
  2. Use $\tilde C_k$ in $s_t$, $J$ and $V_{\mathfrak L}$; then $\gamma_{\text{viol}},\mu_k$ carry units of nat.
  3. Replace "$(C_k-\epsilon_k)/\epsilon_k$" by "$C_k/\epsilon_k$" or by $\tilde C_k$.
- Validation plan: Unit check of lines 290 and 379 after the edit.

### [E-007] Outer-problem constraint contradicts the history-window definition (was V-001)
- Location: `def-outer-problem-governor-optimization` (line 293)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim: line 293 "subject to: $\theta_{t+1} = \Phi(\theta_t, \pi_{\mathfrak{G}}(\Psi(\theta_t); \phi))$".
- Upstream anchor: line 213 "$\Lambda_t = \pi_{\mathfrak{G}}(s_t, s_{t-1}, \ldots, s_{t-H}; \phi)$"; line 329 "$\Lambda_t = \pi_{\mathfrak{G}}(s_{t:t-H}; \phi)$".
- Why this is an error: The constraint feeds the Governor only the current residual $\Psi(\theta_t)=s_t$, making the closed loop Markov in $\theta_t$ and removing the temporal behaviour (first and second differences, line 223) that the definition and `thm-bilevel-structure` rely on.
- Impact on downstream results: Consistency between `def-outer-problem-governor-optimization` and `thm-bilevel-structure`.
- Fix guidance: Write $\theta_{t+1}=\Phi(\theta_t,\pi_{\mathfrak G}(s_{t:t-H};\phi))$ with $s_\tau=\Psi(\theta_\tau)$.
- Validation plan: Compare lines 293 and 329 after the edit.

### [E-008] $V_{\mathfrak L}$ is called an augmented Lagrangian but has no multiplier term (was V-002)
- Location: line 320 (admonition "Why Squared Violations?"); `def-training-lyapunov-function` (line 386)
- Severity: Minor
- Type: Miswording (secondary: Definition mismatch)
- Criterion: External
- Origin: this chapter
- Claim: line 386 "$V_{\mathfrak{L}}$ is the augmented Lagrangian with quadratic penalty."; line 320 "This is exactly the logic behind augmented Lagrangian methods in constrained optimization."
- Why this is an error: An augmented Lagrangian for $C_k\le0$ contains a multiplier term, $\mathcal L+\sum_k[\lambda_kC_k^++\tfrac{\mu_k}{2}(C_k^+)^2]$ (or the equivalent shifted form), with dual updates on $\lambda_k$; $V_{\mathfrak L}$ at line 379 is a pure exterior quadratic penalty. The distinction is load-bearing here: the missing multiplier term is what would make stationary points of the merit function KKT points (E-010) and would connect $V_{\mathfrak L}$ to the $\lambda_k$ the Governor outputs.
- Impact on downstream results: E-010, E-011, E-017.
- Fix guidance: Call $V_{\mathfrak L}$ an "exterior quadratic penalty function", or add the multiplier term and the dual update $\lambda_k\leftarrow\max(0,\lambda_k+\mu_kC_k)$.
- Validation plan: Check the definition against `nocedal2006numerical` terminology, already cited in the chapter.

### [E-009] Definition asserts convergence from $\Delta V<0$ alone (was F-017)
- Location: `def-training-lyapunov-function` (line 386)
- Severity: Minor
- Type: Miswording (secondary: Invalid inference)
- Criterion: External
- Origin: this chapter
- Claim: line 386 "If $\Delta V_{\mathfrak{L}} < 0$ along the training trajectory, training converges (Theorem {prf:ref}`thm-stable-training-trajectory`)."
- Why this is an error: Strict decrease of a bounded-below sequence gives convergence of $V_{\mathfrak L}(\theta_t)$, not of $\theta_t$ (counterexample in E-010). The sentence sits inside a formal definition block and reads as a claim.
- Impact on downstream results: same as E-010.
- Fix guidance: "If $\Delta V_{\mathfrak L}<0$ and the sublevel set $\{V_{\mathfrak L}\le V_{\mathfrak L}(\theta_0)\}$ is compact, $\theta_t$ converges to the set of stationary points of $V_{\mathfrak L}$ (Theorem ...)."
- Validation plan: Align with the corrected theorem statement.

### [E-010] Stable Training Trajectory theorem: missing hypotheses, circular target set, penalty stationary points are not KKT points (was F-006)
- Location: `thm-stable-training-trajectory` (lines 408-421); echoed at line 535 ("LaSalle invariance | Convergence to KKT manifold") and line 911
- Severity: Major
- Type: Invalid inference (secondary: External dependency)
- Criterion: External
- Origin: this chapter
- Claim: lines 414-419 "$\Delta V_{\mathfrak{L}} := V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t) < 0 \quad \forall t \text{ where } \theta_t \notin \Omega$, then the training process converges to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. Under standard regularity (twice-differentiable $\mathcal{L}$, LICQ), $\Omega$ consists of KKT points. *Proof.* $V_{\mathfrak{L}}$ is bounded below by $\inf \mathcal{L}_{\text{task}}$. ... By LaSalle's invariance principle, trajectories converge to the largest invariant set $\Omega$ where $\Delta V_{\mathfrak{L}} = 0$. At points in $\Omega$, either (i) $\nabla \mathcal{L}_{\text{task}} = 0$ and all constraints are satisfied, or (ii) the trajectory is at a boundary where the gradient is balanced by constraint forces."
- Why this is an error:
  1. Missing hypotheses. Monotone decrease plus a lower bound yields convergence of the scalar $V_{\mathfrak L}(\theta_t)$ only. LaSalle's principle needs the trajectory to lie in a compact positively invariant set and $V$ and the update map to be continuous; none is assumed, and $\mathcal L_{\text{task}}$ is not assumed bounded below. Counterexample within the stated hypotheses: $K=0$, $\mathcal L(\theta)=e^{-\theta}$, $\theta_{t+1}=\theta_t+\eta e^{-\theta_t}$. Then $\Delta V<0$ for all $t$, yet $\theta_t\to\infty$ (increments are at least $\eta e^{-M}$ while $\theta_t\le M$; numerically, with $\eta=0.5$, $\theta\approx11.5$ after $2\times10^5$ steps and still rising) and no point has $\Delta V=0$.
  2. Circularity. $\Omega$ appears in the hypothesis (line 414) but is defined only in the conclusion (line 417).
  3. Penalty stationary points are not KKT points. Stationary points of $V_{\mathfrak L}$ satisfy $\nabla\mathcal L+\sum_k\mu_kC_k^+\nabla C_k=0$. If all $C_k\le0$ then $\nabla\mathcal L=0$; otherwise some $C_k>0$ and the point is infeasible. A constrained boundary optimum with $\nabla\mathcal L\ne0$ is never a stationary point of $V_{\mathfrak L}$, because $\mu_kC_k^+\nabla C_k=0$ on $C_k=0$; case (ii) of the proof cannot occur. Complementary slackness holds only in the limit $\mu_k\to\infty$. Twice-differentiability and LICQ do not change this. The controlled $\Lambda_t$ is never used in the proof.
- Impact on downstream results: The chapter's headline guarantee (lines 432, 911); `pi-lyapunov`; `conn-rl-23`; `07_cognition/03_memory_retrieval.md:969, 989`, which build on "the Lyapunov descent condition implies".
- Fix guidance:
  1. Add hypotheses: $\mathcal L_{\text{task}}$ bounded below and the sublevel set $\{V_{\mathfrak L}\le V_{\mathfrak L}(\theta_0)\}$ compact; $V_{\mathfrak L}$ and the controlled update map continuous.
  2. Define $\Omega$ first (fixed points of the controlled map, or stationary points of $V_{\mathfrak L}$) and require $\Delta V\le-\alpha(\mathrm{dist}(\theta_t,\Omega))$ for a class-$\mathcal K$ function $\alpha$.
  3. Replace "KKT points" by "stationary points of $V_{\mathfrak L}$", noting they approach KKT points as $\mu_k\to\infty$; or switch to the augmented Lagrangian of E-008, whose stationary points with the dual update are KKT points.
- Required new assumptions/permits: compactness of the initial sublevel set; continuity of $\Phi$.
- Framework-first proof sketch for the fix: with compact sublevel set and continuous $\Phi$, $\{\theta_t\}$ has a compact closure; the class-$\mathcal K$ decrease gives $\sum_t\alpha(\mathrm{dist}(\theta_t,\Omega))\le V(\theta_0)-\inf V<\infty$, hence $\mathrm{dist}(\theta_t,\Omega)\to0$ without invoking LaSalle.
- Validation plan: Re-run the $e^{-\theta}$ counterexample against the new hypotheses (it violates compactness); check that `03_memory_retrieval.md` uses only the retained conclusion.

### [E-011] Existence of descent direction fails at infeasible penalty minimizers (was F-007)
- Location: `cor-existence-of-descent-direction` (lines 437-446)
- Severity: Moderate
- Type: Invalid inference (secondary: Scope restriction)
- Criterion: External
- Origin: this chapter
- Claim: line 440 "At any non-stationary point $\theta$ where LICQ holds ..., there exist multipliers $\lambda_k \geq 0$ and step size $\eta > 0$ such that $\Delta V_{\mathfrak{L}} < 0$."; line 442 "*Proof.* At a non-KKT point, either (i) the unconstrained gradient $-\nabla \mathcal{L}_{\text{task}}$ points into the feasible region, giving descent, or (ii) some constraint is active with $\nabla C_k \neq 0$. Under LICQ, we can solve for $\lambda_k$ such that the projected gradient onto the feasible tangent cone is non-zero".
- Why this is an error: The statement says "non-stationary", the proof argues from "non-KKT". At a strict local minimizer $\theta^\dagger$ of $V_{\mathfrak L}$ with some $C_k(\theta^\dagger)>0$ (these exist for finite $\mu_k$, E-010(3)), $\theta^\dagger$ is not KKT (infeasible), yet $\Delta V_{\mathfrak L}\ge0$ for every small step, whatever $\lambda_k\ge0,\eta>0$ are chosen. The proof never computes $\Delta V_{\mathfrak L}$ for the update of line 135; the first-order condition is $\langle\nabla V_{\mathfrak L},G^{-1}\nabla\mathcal L+\sum_k\lambda_k\nabla C_k\rangle>0$, which with mixed preconditioning is not sign-definite even for $\lambda_k=\mu_kC_k^+$. LICQ plays no role for a penalty function.
- Impact on downstream results: The "existence guarantee" prose (lines 449-453); summary.
- Fix guidance:
  1. State the corollary for $\nabla V_{\mathfrak L}(\theta)\ne0$.
  2. Precondition both terms (E-004), choose $\lambda_k:=\mu_kC_k^+(\theta)$; then the step direction is $F_\theta^{-1}\nabla V_{\mathfrak L}$, the inner product is $\|\nabla V_{\mathfrak L}\|^2_{F_\theta^{-1}}>0$, and descent for small $\eta$ follows from $C^1$-smoothness of $V_{\mathfrak L}$.
  3. Drop LICQ.
- Validation plan: Verify the inner product identity symbolically after the edit.

### [E-012] Varentropy Brake: stated ODE and basin claim are not what E.10 proves (was F-008)
- Location: `cor-varentropy-brake` (lines 456-477)
- Severity: Major
- Type: Proof gap / omission (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter (proof in `10_appendices/05_proofs.md`)
- Claim: lines 461-467 "For the optimization trajectory to remain in the basin of attraction of the global minimum, the cooling schedule must be modulated by the Varentropy: $\frac{d T_c}{dt} = - \eta \cdot \frac{T_c}{1 + \gamma V_H(\theta_t)}$, where $\eta, \gamma > 0$ are constants."; line 473 "This prevents **Spontaneous Symmetry Breaking** errors where rapid cooling locks the agent into a suboptimal local minimum."; line 475 "*Proof:* See Appendix E.10."
- Upstream anchor: `10_appendices/05_proofs.md:947` "**Statement:** To maintain stability, the cooling rate must satisfy $|\dot{T}_c| \ll T_c / \sqrt{V_H}$."; `:990-1000` "$\left| \frac{ds}{dt} \right| \leq C \cdot \tau_{\text{relax}}^{-1}$ ... $\left| \frac{dT_c}{dt} \right| \leq C \frac{T_c}{\sqrt{V_H}}$"; `06_fields/02_reward_field.md:781-786` "$|\dot{T}_c| \ll \frac{T_c}{\sqrt{V_H(z)}}$".
- Why this is an error: (a) The appendix states and proves an inequality with $V_H^{-1/2}$ scaling; the corollary displays an equality (an ODE) with $(1+\gamma V_H)^{-1}$ scaling and the word "must". Nothing in E.10 derives the rational form or the constants. (b) The ODE respects the proved bound for all $V_H>0$ iff $\eta\le\min_{V>0}C(1+\gamma V)/\sqrt V=2C\sqrt\gamma$ (minimum at $V=1/\gamma$; numerically $3.4641=2\sqrt3$ for $\gamma=3$), a condition never stated; for large $V_H$ the ODE is one admissible schedule among many, not a necessary one. (c) E.10 mentions "stay in the basin of attraction" only as a gloss on the adiabatic hypothesis (Step 4) and "suboptimal metastable state" only in its concluding sentence; neither is derived, and "global minimum" does not appear. Quasi-static tracking of $\pi^*_{T_c}$ does not imply reaching a global minimizer. (d) The Step 4 bound $|ds/dt|\le C\tau_{\text{relax}}^{-1}$ is an assumption; the corollary is therefore a design rule, not a proved statement.
- Impact on downstream results: `03_architecture/03_optimization.md:172-190` ("The Varentropy Brake is stated and proved in `cor-varentropy-brake`") and its discrete schedule, whose second property carries the $(1+\gamma V_H)^{-1}$ form rather than the proved $V_H^{-1/2}$ bound; the Governor's $T_c$ head.
- Fix guidance:
  1. Restate the corollary as the inequality E.10 proves: under the adiabatic hypothesis, $|\dot T_c|\le C\,T_c/\sqrt{V_H}$.
  2. Present $\dot T_c=-\eta T_c/(1+\gamma V_H)$ as a design choice satisfying it when $\eta\le2C\sqrt\gamma$.
  3. Delete "basin of attraction of the global minimum" and the symmetry-breaking sentence, or add a separate result relating quasi-static annealing to basin selection.
- Required new assumptions/permits: the adiabatic bound $|ds/dt|\le C\tau_{\text{relax}}^{-1}$ made explicit as a hypothesis.
- Validation plan: Compare the restated corollary with E.10's Statement line; update `03_optimization.md` to cite the inequality.

### [E-013] Varentropy Brake: unit and notation inconsistencies (was F-009)
- Location: `cor-varentropy-brake` (lines 459-469)
- Severity: Minor
- Type: Notation conflict (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: line 459 "lowering $T_c$ (annealing) to converge on a Nash equilibrium"; line 464-467 "$\frac{d T_c}{dt} = - \eta \cdot \frac{T_c}{1 + \gamma V_H(\theta_t)}$, where $\eta, \gamma > 0$ are constants."
- Upstream anchor: `06_fields/02_reward_field.md:761-766` "Define the **Policy Varentropy** $V_H(z)$ as the variance of the surprisal under the Boltzmann policy ... *Units:* $\mathrm{nat}^2$."; `02_sieve/01_diagnostics.md:203` ($\gamma$ is the world-model volatility exponent).
- Why this is an error: $\gamma V_H$ must be dimensionless, so $[\gamma]=\mathrm{nat}^{-2}$, while $\gamma$ already names the volatility exponent and $\gamma_{\text{viol}}$ is used at line 290. $\eta$ here has units $\mathrm{time}^{-1}$ while $\eta_t$ (line 217) is the learning-rate component of the same control vector. $V_H$ is defined per state $z$ and written $V_H(\theta_t)$ with no aggregation. "Nash equilibrium" is not defined in this single-agent chapter; the target elsewhere is a KKT point.
- Impact on downstream results: `03_architecture/03_optimization.md` discrete schedule (which already uses $\eta_T$).
- Fix guidance: Rename to $\eta_T,\gamma_T$ with $[\gamma_T]=\mathrm{nat}^{-2}$; define $\bar V_H(\theta_t):=\mathbb E_{z\sim\text{batch}}[V_H(z;\theta_t)]$; replace "Nash equilibrium" by "a minimizer of $V_{\mathfrak L}$".
- Validation plan: Unit check of line 464.

### [E-014] Three names and forms for the Node 42 monitor (was F-016)
- Location: `pi-lyapunov` (line 537); Node 42 table (line 937); code (lines 850-869)
- Severity: Note
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim: line 537 "StabilityCheck monitors $\Delta\mathcal{L}_{\text{Lyap}}/\mathcal{L}_{\text{Lyap}}$."; line 937 "$\Delta V_{\mathfrak{L}} = V_{\mathfrak{L}}(\theta_{t+1}) - V_{\mathfrak{L}}(\theta_t)$"; line 857-865 `compute_lyapunov_descent` "Compute V_𝔏 for monitoring ... Returns: V_L: [B] Lyapunov function value".
- Why this is an error: $\mathcal L_{\text{Lyap}}$ vs. $V_{\mathfrak L}$, relative vs. absolute increment, and a method named "descent" that returns the level. The relative form is ill-defined when $V_{\mathfrak L}\le0$. Not a mathematical error.
- Impact on downstream results: none.
- Fix guidance: Use $V_{\mathfrak L}$ and the absolute increment everywhere; rename the method `compute_lyapunov_value` or return the difference.
- Validation plan: grep for `\mathcal{L}_{\text{Lyap}}`.

### [E-015] Diagnostic inputs do depend on the data law; gradient norm mislabelled (was F-013)
- Location: `prop-structure-of-diagnostic-inputs` (lines 583-595)
- Severity: Minor
- Type: Miswording (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim: line 586-591 "consists of quantities that depend only on the learned representations, not on the raw data $\mathcal{D}$: Entropies: $H(K)$, $H(Y|K)$, $I(K;X)$ ... Spectral norms: $\|\nabla_A V\|$, $\lambda_{\max}(G)$ ... These are computed from the model's internal state $\theta_t$ and its outputs on training batches."
- Why this is an error: $I(K;X)$ is a functional of the joint law of the raw observation and the code; $H(K)$ is the entropy of the encoder's push-forward of the data distribution; the last sentence concedes batch dependence. The defensible statement is invariance under relabelling of codebook entries and under the choice of raw-data representation. $\|\nabla_AV\|$ is a gradient norm.
- Impact on downstream results: Premise of `prop-transfer-via-meta-generalization`.
- Fix guidance: Replace "not on the raw data" by "only through representation-level statistics invariant to relabelling of $K$ and to the modality of $X$"; move $\|\nabla_AV\|$ to a "gradient norms" bullet.
- Validation plan: Re-read the proposition for consistency with line 591.

### [E-016] Meta-generalization bound: absent source, $J=0$ on the optimal manifold, no complexity term (was F-012)
- Location: `prop-transfer-via-meta-generalization` (lines 610-641)
- Severity: Moderate
- Type: External dependency (secondary: Proof gap / omission)
- Criterion: External
- Origin: this chapter
- Claim: line 613 "Under the conditions of the Meta-Generalization Metatheorem (**MT: Meta-Generalization** in `metalearning.md`)"; line 617 "$c\,\text{dist}(\phi, \mathcal{M})^2 \leq J(\phi) \leq C\,\text{dist}(\phi, \mathcal{M})^2$ near the optimal manifold"; line 623 "$\mathbb{E}_{S \sim \mathcal{S}}[J_S(\hat{\phi}_N)] \leq C_1\left(\varepsilon_N + \sqrt{\frac{\log(1/\delta)}{N}}\right)$".
- Upstream anchor: no file named `metalearning*` exists in the repository; line 638 (this chapter) calls the source "unpublished".
- Why this is an error: (i) The two-sided bound forces $J=0$ on $\mathcal M$, impossible for a cumulative task loss plus penalty (line 290) unless $\mathcal L_{\text{task}}$ vanishes along the whole trajectory; the object must be excess regret $J_S(\phi)-\inf J_S$, and the conclusion should bound excess regret. (ii) A uniform-convergence bound with a universal $C_1$ and no dependence on the policy class (covering numbers or Rademacher complexity of the GRU class) cannot hold for an arbitrary parametric class; step 2 of the sketch is true only with such a term. (iii) The source lies outside the book, so within the framework the result is an unproven import stated as a `prf:proposition`.
- Impact on downstream results: `sec-transfer-via-geometric-invariance`; `conn-rl-23` "Transfer via geometric invariants".
- Fix guidance:
  1. Replace $J$ by excess regret $\Delta J_S(\phi):=J_S(\phi)-\inf J_S$ in condition 2 and the conclusion.
  2. Add a complexity term, e.g. $C_1\big(\varepsilon_N+\mathfrak R_N(\Pi)+\sqrt{\log(1/\delta)/N}\big)$.
  3. Downgrade to a conjecture/assumption box until the metatheorem is included in the appendix with proof.
- Required new assumptions/permits: a bound on the Rademacher complexity of the Governor policy class.
- Validation plan: Check that the conclusion is dimensionally and logically consistent with $J\ge\inf J>0$.

### [E-017] Governor told to adjust quantities outside its output space (was F-010)
- Location: `def-canonical-obstruction-suite` table (lines 699-705); Node 42 remedy (line 941)
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim: line 703 "Increase jump rate $\lambda_{\text{jump}}$"; line 704 "Texture firewalling"; line 705 "Increase $\mu_k$ (barrier strength)"; line 941 "Remedy: Reduce learning rate; increase constraint penalties $\mu_k$".
- Upstream anchor: line 217 (this chapter) "$\Lambda_t = (\eta_t, \lambda_{1,t}, \ldots, \lambda_{K,t}, T_{c,t}) \in \mathbb{R}_+^{K+2}$"; line 382 "$\mu_k > 0$ are penalty weights"; `03_architecture/02_disentangled_vae.md:259` and `05_geometry/04_equations_motion.md:347` ($\lambda_{\text{jump}}$ is a loss weight / jump intensity).
- Why this is an error: The declared control vector contains $\eta_t$, the $\lambda_{k,t}$ and $T_{c,t}$ only. $\mu_k$ are the fixed weights of the candidate Lyapunov function; if changed between steps, $\Delta V_{\mathfrak L}$ becomes the difference of two different functions and `thm-stable-training-trajectory` and Node 42 lose their meaning (raising $\mu_k$ at a violated constraint increases $V_{\mathfrak L}$ and would itself trip Node 42). $\lambda_{\text{jump}}$ and texture firewalling are not components of $\Lambda_t$.
- Impact on downstream results: Meta-training protocol; Node 42 remediation.
- Fix guidance: Either enlarge $\Lambda_t$ and state that $\mu_k$ is held fixed within an episode, or map remedies onto existing outputs (Constraint Cliff: increase $\lambda_k$, decrease $\eta_t$; Disconnected Modes: increase $T_c$ or the $\lambda_k$ of the jump-consistency node; Noise Floor: increase the $\lambda_k$ of the texture-firewall node).
- Validation plan: Every remedy in the table should name a component of $\Lambda_t$.

### [E-018] RL reward $-\Delta V_{\mathfrak L}$ does not implement the stated regret (was F-011)
- Location: `def-canonical-obstruction-suite`, Remark (Training Protocol) (line 707) vs. `def-outer-problem-governor-optimization` (lines 287-297)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 707 "The Governor is trained via reinforcement learning on this suite, with reward $r_t = -\Delta V_{\mathfrak{L}}$."; line 297 "The outer objective penalizes cumulative task loss (convergence speed)".
- Why this is an error: Undiscounted, $\sum_{t=0}^{T-1}r_t=V_{\mathfrak L}(\theta_0)-V_{\mathfrak L}(\theta_T)$ (check: $V=(5,4,4,4,4,1)$ gives return $4$ while the trajectory sum is $22$). Maximizing it minimizes only the final penalized loss; a policy that stalls for $T-1$ steps then drops is optimal for the return and poor for $J$. $J$ also uses $\gamma_{\text{viol}}$ while $V_{\mathfrak L}$ uses $\mu_k/2$, with no stated relation.
- Impact on downstream results: Meta-training protocol; the summary claim that the Governor minimizes training regret.
- Fix guidance: Use $r_t=-\big(\mathcal L_{\text{task}}(\theta_t)+\gamma_{\text{viol}}\sum_k\mathrm{ReLU}(C_k(\theta_t))^2\big)$ so that $\sum_tr_t=-J$; or, for a shaped reward, $r_t=-V_{\mathfrak L}(\theta_{t+1})$ with $\mu_k:=2\gamma_{\text{viol}}$.
- Validation plan: Verify $\sum_tr_t=-J$ symbolically.

### [E-019] Summary table drops the ReLU from the regret (was F-014)
- Location: Table 26.9.1 (line 927)
- Severity: Minor
- Type: Typo (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: line 927 "$J(\phi) = \mathbb{E}[\sum_t \mathcal{L}_t + \gamma_{\text{viol}}\sum_k C_k^2]$"
- Upstream anchor: line 290 (this chapter) "$\gamma_{\text{viol}} \sum_{k=1}^K \text{ReLU}(C_k(\theta_t))^2$".
- Why this is an error: As written, satisfied constraints ($C_k<0$) are penalized as much as violated ones.
- Impact on downstream results: none beyond the table.
- Fix guidance: Write $\sum_k\mathrm{ReLU}(C_k)^2$.
- Validation plan: Compare with line 290.

## Scope restrictions and clarifications
- The Lyapunov results (E-009, E-010, E-011) hold, after correction, for stationary points of the penalty function $V_{\mathfrak L}$ under compactness of the initial sublevel set; convergence to KKT points of the constrained problem requires either $\mu_k\to\infty$ or the augmented-Lagrangian form with dual updates (E-008).
- The Varentropy Brake (E-012) is, as proved in E.10, an inequality under an assumed adiabatic bound; the displayed ODE is a design choice that satisfies it only when $\eta\le2C\sqrt\gamma$.
- The meta-generalization proposition (E-016) depends on a source outside the repository and should be read as a conjecture until the metatheorem is included.
- The unit conventions (E-003, E-006, E-013) assume entropy-based nodes are measured in nats; the fixes are simplest if all residuals are normalized to dimensionless form once, in `def-constrained-dynamics`.

## Open questions
- Should the Governor's control vector be enlarged to include $\mu_k$ (held fixed per episode), $\lambda_{\text{jump}}$ and the texture-firewall weight, or should the Obstruction Suite remedies be re-expressed in terms of the current $\Lambda_t$?
- Is the intended merit function the pure penalty $V_{\mathfrak L}$ or an augmented Lagrangian whose multipliers coincide with the $\lambda_k$ the Governor outputs? The latter would unify E-008, E-010, E-011, E-017 and E-018.
- Which parameter-space preconditioner (if any) is meant by $G^{-1}$ at line 135? The framework defines none.

## Rejected candidate findings
None. All seventeen stage-1 findings were confirmed; F-001 (criterion changed from External to Framework) and F-008 (evidence for point (c) corrected: E.10 does mention "basin of attraction" as a gloss, but never the global minimum) were kept with adjustments. Two findings were added by the verifier (E-007, E-008).
