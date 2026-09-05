# Mathematical Review: docs/source/1_agent/02_sieve/02_limits_barriers.md

## Metadata
- Reviewed file: docs/source/1_agent/02_sieve/02_limits_barriers.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (210 lines)
- Framework anchors (definitions/axioms/permits):
  - docs/source/1_agent/01_foundations/01_definitions.md:101-105 (split latent $Z_t\in\mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}}$), :124-128 (observation $x_t\in\mathcal{X}$), :236-246 (action vector, BarrierSat, Node 2)
  - docs/source/1_agent/01_foundations/02_control_loop.md:481-489 (macro-channel capacity $I(X;K)\le H(K)\le\log|\mathcal{K}|$), :498-502 (entropy-regularized free energy, stochastic policy)
  - docs/source/1_agent/02_sieve/01_diagnostics.md:93-108 (node table: Node 2 ZenoCheck, Node 9 TameCheck), :137-141 (Node 7, Node 9 criteria), :199-215 (scaling coefficients $\alpha,\beta_\pi,\gamma,\delta$ and the stability hierarchy), :264-284 (shutter losses, "anti-collapse"), :435-443 ($\mathcal{L}_{\text{Lip}}$), :545-553 (Lyapunov stiffness, Node 7), :705-722 (information-stability window penalty)
  - docs/source/1_agent/04_control/03_coupling_window.md:119-157 (`thm-information-stability-window-operational`; BarrierScat as dispersion)
  - docs/source/1_agent/04_control/01_exploration.md:174 (BarrierScat at maximal entropy)
  - docs/source/1_agent/02_sieve/03_failures_interventions.md:61-65 (Mode B.C, Ashby)
  - docs/source/1_agent/02_sieve/04_approximations.md:39-45 (BarrierBode proxy), :245-252 ($\mathcal{L}_{\text{tame}}$)
  - docs/source/1_agent/03_architecture/01_compute_tiers.md:66-82 (barrier tiers), :205-212 ($\max(0,\beta_\pi-\alpha)$)
  - docs/source/1_agent/10_appendices/06_losses.md:528-610 (F.7.1 gradient penalty, F.7.2 information-control, F.7.3 EWC, F.7.4 Bode)
  - docs/source/1_agent/10_appendices/04_faq.md:112-119, :652-658 (BarrierSat as hard clamp; hierarchy $\delta\ll\gamma\ll\alpha$)
  - Mechanical pre-pass: no dangling references or duplicate labels in this file; the `{prf:ref}` on line 63 resolves to 01_definitions.md:143.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 3
- Minor: 5
- Notes: 3
- Primary themes: (1) implementation recipes that do not deliver the guarantee stated next to them (BarrierSat mean-only squashing; BarrierGap two-sided penalty for a one-sided constraint); (2) table entries that contradict the framework's own definitions (BarrierVariety width inequality against the compression architecture; BarrierScat named after the opposite end of the coupling window; the stability-plasticity dilemma paired with a policy check); (3) an imported control-theory statement with the wrong hypothesis (Bode integral "= 0" under stable/minimum-phase); (4) symbol overloading ($\beta$, $\gamma$, $K$) against the scaling-coefficient conventions of 01_diagnostics.md.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Barrier table, column gloss and rows BarrierCausal/BarrierBode/BarrierLock (lines 45, 51, 54, 65, 68) | Note | Miswording | Framework | this chapter | "Regularization Factor" column, defined as a loss term, contains three non-losses |
| E-002 | Barrier table row BarrierSat (line 53); A.1 and preceding prose (lines 94-101) | Moderate | Algorithm mismatch (Dimensional mismatch) | Framework | this chapter | Mean-only tanh squashing does not make the norm constraint structurally impossible to violate for a stochastic vector policy; row says "Soft Clipping" |
| E-003 | Rows BarrierCausal/BarrierTypeII (lines 54, 56); A.2 (lines 104-106); B.1 (lines 157-166) | Minor | Notation conflict | Framework | this chapter | Bare $\beta$, two unrelated $\gamma$'s, and undefined $k_\pi$ against the $\alpha,\beta_\pi,\gamma,\delta$ conventions |
| E-004 | Row BarrierScat (line 55); B.1 conflict line (line 154) | Minor | Definition mismatch (Miswording) | Framework | this chapter | "Representation Collapse" / "anti-collapse" invert the upstream meaning of BarrierScat (dispersion) |
| E-005 | Row BarrierGap (line 59); A.4 (lines 125-133) | Minor | Algorithm mismatch (Notation conflict) | Framework | this chapter | One-sided constraint $\lVert\nabla_A V\rVert\ge\epsilon$ implemented as a two-sided pin to $K$ on the plain gradient |
| E-006 | Row BarrierOmin (line 61); A.3 (lines 115-122) | Minor | Conceptual (Definition mismatch) | External (naming); Framework (derivative order) | this chapter; upstream 02_sieve/01_diagnostics.md internally split | Lipschitz bound labelled o-minimality; first- vs second-derivative bound differs from Node 9 |
| E-007 | Row BarrierBode (line 65); B.3 (line 195); prose (lines 205-207) | Moderate | External dependency (Scope restriction) | External | this chapter | Bode integral claimed to equal 0 under stable/minimum-phase; correct hypothesis is stable plus relative degree at least two |
| E-008 | Row BarrierVariety (line 67) | Moderate | Conceptual (Dimensional mismatch) | Framework (also External) | this chapter | $\dim(Z)\ge\dim(\mathcal{X})$ forbids the shutter's compression and is not what requisite variety states |
| E-009 | Note after the table (line 74) | Note | Invalid inference | Framework | this chapter | Unsupported claim that the most dangerous barriers are the hardest to detect |
| E-010 | B.1 (lines 152-166) | Note | Conceptual | Framework | this chapter | Only the dispersion edge of the coupling window trades off against BarrierCap; the Pareto search is one-sided |
| E-011 | B.2 (lines 176-184) | Minor | Definition mismatch (Citation / reference error) | Framework | this chapter (06_losses.md:586 cites a nonexistent "BarrierPZ") | Stability-plasticity dilemma pairs a world-model barrier with the policy's ZenoCheck |

## Detailed findings

### [E-001] The "Regularization Factor" column contains entries that are not losses (was F-010)
- Location: Barrier table, column gloss (line 45), header (line 51), rows BarrierCausal (line 54), BarrierBode (line 65), BarrierLock (line 68)
- Severity: Note
- Type: Miswording (secondary: none)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 45 "**Regularization Factor**: A loss term to add to your training objective to stay away from this barrier."; line 54 "$T_{\text{horizon}}$ (Discount Factor $\gamma < 1$)"; line 65 "$\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.}$ (Bode sensitivity integral)"; line 68 "$\mathbb{I}(s \in \text{Forbidden}) \cdot \infty$".
- Upstream anchor: not applicable; the column is defined in this chapter.
- Why this is an error: a discount factor is an objective hyperparameter (its effective horizon $1/(1-\gamma)$ is unrelated to the compute latency the BarrierCausal row describes), the Bode integral is an identity, and an infinite indicator has no gradient. None can be "added to the training objective" as the gloss instructs; the actual Bode loss is $\mathcal{L}_{\text{Bode}}$ on line 199.
- Impact on downstream results: none mathematical; it weakens the "read the table as a recipe" instruction on line 48.
- Fix guidance:
  1. Rename the column "Constraint / Regularizer".
  2. Mark the three rows as constraints, or replace them: BarrierCausal by a latency budget $t_{\text{compute}}\le T_{\text{horizon}}$; BarrierBode by $\mathcal{L}_{\text{Bode}}$; BarrierLock by "hard projection".
- Required new assumptions/permits: none.
- Validation plan: re-read the column against line 45 and confirm every entry is either a differentiable loss or explicitly tagged as a constraint.

### [E-002] BarrierSat: mean-only tanh squashing does not make the norm constraint impossible to violate (was F-001)
- Location: Barrier table row BarrierSat (line 53); A. Single-Barrier Enforcement, prose and item 1 (lines 94-101)
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 53 "$\Vert \pi(s) \Vert < F_{\text{max}}$ (Soft Clipping)"; line 96 "The `tanh` function physically cannot output values outside $[-1, 1]$, no matter what the network learns."; line 100-101 "*Constraint:* $\lVert\pi(s)\rVert \le F_{\max}$. *Implementation:* **Squashing Function**. Use `tanh` on the policy mean: $\mu(z) = F_{max} \cdot \tanh(f_\theta(z))$. Do not rely on clipping losses alone; the architecture must be incapable of exceeding limits."
- Upstream anchor: 01_foundations/01_definitions.md:243 "**BarrierSat:** actuator saturation (finite control authority)." (name only); 01_foundations/02_control_loop.md:500 "$\mathcal{F}[p, \pi] = \int p(z)(V(z) - T_c H(\pi(\cdot|z))) d\mu_G$" (stochastic policy); this chapter line 62 "$-H(\pi)$ (Entropy Bonus)"; 01_definitions.md:238 "$a_t = (K^{\text{act}}_t, z_{n,\text{motor}}, z_{\text{tex,motor}})$" (vector action).
- Why this is an error: (i) The policy is stochastic throughout the book (entropy term in the free energy and in this chapter's BarrierMix row). Squashing only the mean leaves a sampled action $a=\mu+\sigma\epsilon$ from any unbounded-support distribution unbounded, so the architecture is not "incapable of exceeding limits"; the prose about tanh is true for a scalar deterministic output and is transferred to a setting where it does not apply. (ii) For an $A$-dimensional action, componentwise $|\mu_i|\le F_{\max}$ gives only $\lVert\mu\rVert_2\le\sqrt{A}\,F_{\max}$; the stated Euclidean constraint holds only if the norm is $\ell_\infty$. (iii) The table row labels the same barrier "(Soft Clipping)", contradicting the hard-wall philosophy of lines 94-96.
- Impact on downstream results: 03_architecture/01_compute_tiers.md:78 lists BarrierSat as "Built-in (tanh) / Zero runtime cost" and 10_appendices/04_faq.md:655 relies on BarrierSat as a hard clamp $a\in\mathcal{A}_{\text{safe}}$; both inherit the gap.
- Fix guidance:
  1. State the constraint per component ($\lVert\pi(s)\rVert_\infty\le F_{\max}$) or rescale to $F_{\max}/\sqrt{A}$ per component.
  2. Apply the squashing to the sample: $a=F_{\max}\tanh(u)$ with $u\sim\pi_\theta(\cdot\mid z)$, and include the change-of-variables term $-\sum_i\log(1-\tanh^2u_i)$ in $\log\pi(a\mid z)$.
  3. Note that the BarrierMix entropy bonus is then computed on the squashed distribution.
  4. Change "(Soft Clipping)" on line 53 to "(Architectural squashing)" or move soft clipping to a stated fallback.
- Required new assumptions/permits: the base distribution of $u$ must have a density (for the Jacobian term).
- Validation plan: sample actions from the trained policy and confirm $\max_i|a_i|\le F_{\max}$ holds exactly, not in expectation; check that the log-density used by the entropy term includes the Jacobian.

### [E-003] Scaling-coefficient notation conflicts with 01_diagnostics.md (was F-002)
- Location: Barrier table rows BarrierCausal (line 54) and BarrierTypeII (line 56); A.2 (lines 104-106); B.1 (lines 157-166)
- Severity: Minor
- Type: Notation conflict (secondary: none)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 56 "$\beta>\alpha$ (Policy update scale outruns critic signal)" and "$\max(0, \beta - \alpha)$"; line 54 "(Discount Factor $\gamma < 1$)"; line 104 "*Constraint:* $\alpha > \beta$"; line 106 "**skip** the Policy update step ($k_\pi = 0$)"; line 162 "$\underbrace{\gamma\,\mathbb{E}[\mathfrak{D}(Z,A)]}_{\text{Control Effort}}$".
- Upstream anchor: 02_sieve/01_diagnostics.md:201-204 (Critic "$\alpha$", Policy "$\beta_{\pi}$", World Model "Volatility scale $\gamma$", VQ-VAE "$\delta$"); :210 "$\delta \ll \gamma \ll \alpha,\qquad \beta_{\pi} \le \alpha$"; :215 "If $\beta_{\pi}>\alpha$, skip or shrink the policy update (BarrierTypeII; see Section 4.1)"; 03_architecture/01_compute_tiers.md:209 "$\lambda_{\text{scale}} \max(0, \beta_{\pi} - \alpha)$".
- Why this is an error: this chapter is the named implementation target of the upstream hierarchy, yet it drops the $\pi$ subscript on $\beta$ while using $\beta_K,\beta_n,\beta_{\mathrm{tex}}$ as compression weights a few lines later, and uses $\gamma$ for a discount (line 54) and for a control-effort weight (line 162) while upstream $\gamma$ is the world-model volatility scale in the same hierarchy. $k_\pi$ appears nowhere else in Volume 1 (grep). Line 104 is strict ($\alpha>\beta$) where upstream is non-strict ($\beta_\pi\le\alpha$).
- Impact on downstream results: confusion only; 10_appendices/06_losses.md:566-572 copies the $\beta_\ast,\gamma$ notation of B.1 and 04_faq.md:117 quotes the hierarchy with $\gamma$ as the WM scale.
- Fix guidance:
  1. Write $\beta_\pi$ in lines 56 and 104 and use the non-strict $\beta_\pi\le\alpha$.
  2. Rename the control-effort weight (e.g. $\lambda_{\mathfrak{D}}$) and the discount (e.g. $\gamma_{\text{disc}}$).
  3. Define $k_\pi$ as the number of policy-gradient steps per iteration, or delete the parenthetical.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter and 06_losses.md F.7.2 for bare `\beta` and `\gamma` after the edit.

### [E-004] BarrierScat named "Representation Collapse" and its rate term called "anti-collapse" (was F-003)
- Location: Barrier table row BarrierScat (line 55); B.1 conflict line (line 154)
- Severity: Minor
- Type: Definition mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 55 "**BarrierScat** | Representation Collapse | **VQ-VAE** | **Grounding Loss** | Symbol channel loses grounding; macrostates become noise-like. | $\mathrm{ReLU}(\epsilon-I(X;K))^2 + \mathrm{ReLU}(H(K)-(\log\lvert\mathcal{K}\rvert-\epsilon))^2$ (Window Penalty)"; line 154 "*Conflict:* High compression (anti-collapse) removes details needed for fine control (capacity/controllability)."
- Upstream anchor: 04_control/03_coupling_window.md:137 "If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion - symbol dispersion (BarrierScat)."; :155 "**BarrierScat (Dispersion)**: Your belief is spread over everything."; 04_control/01_exploration.md:174 "$T_c\to\infty$: $\pi$ approaches maximal entropy; behavior becomes overly random and may degrade grounding (BarrierScat)."; 02_sieve/01_diagnostics.md:266 "The most dramatic failure is "codebook collapse": the encoder uses only a handful of symbols"; :282 "$\lambda_{\text{use}} D_{\mathrm{KL}}(\hat{p}(K)\Vert \mathrm{Unif}(\mathcal{K}))$ ... anti-collapse (optional)".
- Why this is an error: upstream, BarrierScat is the saturation end of the coupling window ($H(K)\to\log|\mathcal{K}|$) and "collapse"/"anti-collapse" denote the opposite end (few codes used; the anti-collapse term raises $H(K)$ toward uniform). This chapter names the dispersion barrier "Representation Collapse" and calls the rate penalty $\beta_K\mathbb{E}[-\log p_\psi(K)]$, which is minimized by concentrating $K$ on high-probability codes and therefore lowers $H(K)$, "anti-collapse". Under the book's own vocabulary that term is anti-dispersion. The regularizer in the row is the correct two-sided window penalty of 01_diagnostics.md:710-717, so the operational recipe is right; only the labels point toward the wrong remedy.
- Impact on downstream results: 10_appendices/06_losses.md:566 repeats "High compression removes details needed for fine control" without the gloss; 03_architecture/01_compute_tiers.md:79 does not name collapse. No result changes.
- Fix guidance:
  1. Rename the row "Symbol Dispersion / Grounding Loss" (matching 03_coupling_window.md), or "Coupling-Window Violation" if both edges are intended.
  2. Replace "(anti-collapse)" on line 154 with "(anti-dispersion; keeps $H(K)\le\log|\mathcal{K}|-\epsilon$)".
- Required new assumptions/permits: none.
- Validation plan: grep Volume 1 for "BarrierScat" and confirm every occurrence reads as dispersion/saturation.

### [E-005] BarrierGap: one-sided constraint implemented as a two-sided gradient pin (was F-006)
- Location: Barrier table row BarrierGap (line 59); A.4 (lines 125-133)
- Severity: Minor
- Type: Algorithm mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 59 "$\max(0, \epsilon - \Vert \nabla_A V \Vert)$ (Stiffness)"; line 126 "*Constraint:* $\lVert\nabla_A V\rVert \ge \epsilon$ (No flat plateaus)."; line 130 "$\mathcal{L}_{GP} = \mathbb{E}_{\hat{s}} [(\lVert\nabla_{\hat{s}} V(\hat{s})\rVert - K)^2]$".
- Upstream anchor: 02_sieve/01_diagnostics.md:547-551 "$\mathcal{L}_{\text{Stiff}} = \max(0, \epsilon - \lVert\nabla_A V(z)\rVert)^2 + \lVert\nabla_A V(z)\rVert^2_{\text{reg}}$ ... The gradient $\nabla_A V$ must be non-zero (to drive the policy) but bounded"; 10_appendices/06_losses.md:533-548 (F.7.1) "$\mathcal{L}_{GP} = \mathbb{E}_{\hat{s}}[(\|\nabla_A V\|_G - K)^2]$ ... **Flat limit:** When $G^{ij} = \delta^{ij}$ and $A=0$, recovers $(\|\nabla_A V\|_2 - K)^2$."
- Why this is an error: the row and the stated constraint are a one-sided lower bound with threshold $\epsilon$ on the covariant gradient $\nabla_A V=\nabla V-A$. The implementation is (a) two-sided, pinning the norm to a target $K$ (an eikonal/Lipschitz condition); (b) written with the plain gradient, dropping the reward 1-form without the flat-limit assumption that F.7.1 states; (c) uses $K$ with no relation to $\epsilon$, while $K$ is also the Lipschitz cap in A.3 (line 116). The prose on line 138 concedes the penalty is two-sided, which confirms the mismatch with line 126.
- Impact on downstream results: 07_cognition/03_memory_retrieval.md:342 and 04_faq.md:112 cite BarrierGap qualitatively; only the recipe is affected.
- Fix guidance:
  1. Give the implementation as $\mathcal{L}_{GP}=\mathbb{E}_{\hat s}[\max(0,\epsilon-\lVert\nabla_A V(\hat s)\rVert_G)^2]$, matching the row and Node 7.
  2. If the two-sided WGAN-GP form is also wanted, present it as an additional upper bound with $K\ge\epsilon$ and state the flat-limit assumption $G=I$, $A=0$.
  3. Rename the target to avoid the clash with the Lipschitz cap $K$.
- Required new assumptions/permits: none beyond the explicit flat-limit assumption if the plain gradient is kept.
- Validation plan: check that the implemented loss is zero whenever $\lVert\nabla_A V\rVert\ge\epsilon$ (one-sided) and compare with 06_losses.md F.7.1.

### [E-006] BarrierOmin: Lipschitz bound labelled o-minimality, and derivative order differs from Node 9 (was F-005)
- Location: Barrier table row BarrierOmin (line 61); A.3 (lines 115-122)
- Severity: Minor
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: External (naming); Framework (derivative order)
- Origin: this chapter (naming and prose); upstream docs/source/1_agent/02_sieve/01_diagnostics.md is internally split on the derivative order
- Claim (verbatim): line 61 "$\Vert \nabla S_t \Vert$ for O-Minimality (Lipschitz)"; line 116 "*Constraint:* $\lVert S\rVert_{\mathrm{Lip}} \le K$."; line 120 "What does "tame" or "o-minimal" mean? Roughly: the function cannot do anything too wild - no infinitely fast oscillations, no fractal behavior, no pathological surprises. A Lipschitz constraint says: change the input by a small amount, the output changes by at most $K$ times that amount."
- Upstream anchor: 02_sieve/01_diagnostics.md:106 "**TameCheck** ... Dynamics Lipschitz-bounded? | $\Vert \nabla^2 S_t \Vert$ (Hessian Norm / Smoothness) | $O(Z^2 P_{WM})$"; :139 "**9 (Tameness)** | ... | Is $\lVert\nabla_z f\rVert_G < K$?"; :437-441 "**Lipschitz Constraint (BarrierOmin / Node 9):** $\mathcal{L}_{\text{Lip}} = \mathbb{E}[(\lVert S(z)-S(z')\rVert/\lVert z-z'\rVert - K)^+]^2$"; 02_sieve/04_approximations.md:249 "$\mathcal{L}_{\text{tame}} = \frac{\Vert \nabla_z S_t(z_1) - \nabla_z S_t(z_2) \Vert}{\Vert z_1 - z_2 \Vert + \epsilon}$".
- Why this is an error: (i) External: a Lipschitz bound is neither necessary nor sufficient for o-minimality. $f(x)=x^2\sin(1/x)$ on $[-1,1]$ has $f'(x)=2x\sin(1/x)-\cos(1/x)$, so $|f'|\le3$ (numerically about $1.33$), hence globally Lipschitz, yet it has infinitely many zeros in a bounded set and is not definable in any o-minimal structure; conversely $\sqrt{|x|}$ is semialgebraic but not Lipschitz at 0. Line 120 asserts exactly the implication that fails. (ii) Framework: this chapter bounds the first derivative $\Vert\nabla S_t\Vert$ (cost $O(ZP_{WM})$) whereas the Node 9 table entry (01_diagnostics.md:106) and the approximation (04_approximations.md:249) bound the second derivative; 01_diagnostics.md:139 and :437 use the first derivative. The book does not agree with itself on what "tameness" is operationally.
- Impact on downstream results: no proof in Volume 1 depends on genuine o-minimality; the damage is terminological plus the first/second-derivative inconsistency (03_architecture/01_compute_tiers.md:80 classifies BarrierOmin as "Specialized" without fixing the order).
- Fix guidance:
  1. Rename the barrier "Lipschitz / Sensitivity" or state that the Lipschitz and Hessian bounds are proxies motivated by, not equivalent to, tameness.
  2. Delete the clause claiming a Lipschitz constraint excludes infinitely fast oscillations.
  3. Choose one derivative order for Node 9 and use it in 01_diagnostics.md:106, :139, :437, 04_approximations.md:249 and this chapter.
- Required new assumptions/permits: none.
- Validation plan: cross-read the four upstream locations and this row after the edit; confirm one order and one cost.

### [E-007] Bode sensitivity integral stated to vanish under stable/minimum-phase assumptions (was F-008)
- Location: Barrier table row BarrierBode (line 65); B.3 (line 195); prose (lines 205-207)
- Severity: Moderate
- Type: External dependency (secondary: Scope restriction)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 195 "Bode sensitivity integral constraint: $\int_{0}^{\infty} \log |S(j\omega)| d\omega = \text{const.}$; equal to $0$ under standard stable/minimum-phase assumptions"; line 65 "$\int_{0}^{\infty} \log \lvert S(j\omega) \rvert d\omega = \text{const.}$ (Bode sensitivity integral)".
- Upstream anchor: none; the result is imported here. 02_sieve/04_approximations.md:41-45 only replaces it by a temporal gain-margin proxy; 10_appendices/06_losses.md:605 repeats the waterbed statement without the "= 0" claim.
- Why this is an error: for a continuous-time LTI loop with open-loop transfer function $L(s)$, $\int_0^\infty\ln|S(j\omega)|\,d\omega=\pi\sum_k\mathrm{Re}(p_k)-\tfrac{\pi}{2}\lim_{s\to\infty}sL(s)$, the sum over open-loop unstable poles. The integral is zero when $L$ is open-loop stable and has relative degree at least two. Minimum phase is not a hypothesis of this integral (right-half-plane zeros enter the Poisson-type constraints, not this area). Recomputation: for the stable, minimum-phase, relative-degree-one loop $L(s)=1/(s+1)$, $S=(s+1)/(s+2)$ and numerical quadrature gives $-1.5708=-\pi/2\neq0$; for $L(s)=1/(s+1)^2$ (relative degree two) the integral is $0$ to $10^{-13}$. The statement also silently assumes an LTI, continuous-time, single loop; for the sampled loop the book actually runs the integral is over $[0,\pi/\Delta t]$ and the waterbed is bounded.
- Impact on downstream results: none quantitative; no derivation uses the value 0. The prose on lines 205-207 ("Total volume is conserved") is a fair informal gloss of "const.".
- Fix guidance:
  1. Replace the parenthetical with "equal to $\pi\sum_k\mathrm{Re}\,p_k$ over open-loop unstable poles, hence $0$ when the open loop is stable and has relative degree at least two".
  2. Remove "minimum-phase".
  3. Add one clause noting the discrete-time form for a loop with step $\Delta t$.
- Required new assumptions/permits: LTI single-loop approximation of the closed loop, stated explicitly.
- Validation plan: numerical check of the integral for one relative-degree-one and one relative-degree-two stable plant (as above).

### [E-008] BarrierVariety regularizer $\dim(Z)\ge\dim(\mathcal{X})$ contradicts the compression architecture and requisite variety (was F-004)
- Location: Barrier table row BarrierVariety (line 67)
- Severity: Moderate
- Type: Conceptual (secondary: Dimensional mismatch)
- Criterion: Framework (also External)
- Origin: this chapter
- Claim (verbatim): line 67 "**BarrierVariety** | Requisite Variety | **Policy** | **Ashby's Deficit** | Policy states < Disturbance states. | $\dim(Z) \ge \dim(\mathcal{X})$ (Width Penalty) | $O(1)$ ✓".
- Upstream anchor: 01_foundations/01_definitions.md:103 "$Z_t := (K_t, z_{n,t}, z_{\mathrm{tex},t}) \in \mathcal{Z}=\mathcal{K}\times\mathcal{Z}_n\times\mathcal{Z}_{\mathrm{tex}}$"; :126 "$x_t\in\mathcal{X}$ is the observation (input sample)"; 01_foundations/02_control_loop.md:483-487 "$I(X;K)\le H(K)\le \log|\mathcal{K}|$ ... the macro channel is a bounded-rate symbolic memory with capacity at most $\log|\mathcal{K}|$ nats."; 02_sieve/03_failures_interventions.md:63 "**B.C** | Control Deficit | **Policy** | **Overwhelmed** | Disturbance more complex than controller (Ashby)."
- Why this is an error: (i) In the framework, $\mathcal{Z}$ is the compressed shutter output of $\mathcal{X}$; requiring $\dim(Z)\ge\dim(\mathcal{X})$ forbids the bottleneck that defines the VQ-VAE and every architecture in Chapter 3 (image observations with $\dim\mathcal{X}\sim10^4$-$10^5$ against latents of order $10^1$-$10^2$). (ii) Externally, requisite variety compares the variety of the regulator's responses (the action repertoire) with the variety of disturbances, in log-cardinality units; it is not a statement about the width of an internal latent versus the sensor dimension. The row's Mechanism column ("Policy states < Disturbance states") states the law correctly and is not what the regularizer encodes. (iii) The bottleneck is labelled "Policy" while the regularizer constrains the latent.
- Impact on downstream results: 03_architecture/01_compute_tiers.md:78 ("BarrierVariety | Built-in (tanh, dim)") and :69 ("Use a finite-dimensional latent space? Congratulations, you've limited representational variety.") inherit the latent-dimension reading, the second in the opposite direction. Mode B.C is unaffected.
- Fix guidance:
  1. Replace the regularizer by an actuator-side cardinality/entropy condition, e.g. $\log|\mathcal{K}^{\text{act}}|\ge H(D)$ where $D$ is the disturbance process (estimable from WM residual entropy) and $\mathcal{K}^{\text{act}}$ is the discrete motor macro of 01_definitions.md:238.
  2. Keep Bottleneck as "Policy" consistently; if a latent condition is really intended, change Bottleneck to "VQ-VAE" and write it as $\log|\mathcal{K}|\ge H(\text{disturbance macro})$, not a dimension inequality.
  3. Update 01_compute_tiers.md:69 and :78 to match.
- Required new assumptions/permits: an operational definition of disturbance variety $H(D)$ (or a proxy) in the framework.
- Validation plan: check that the new condition is satisfiable by the reference architectures in Chapter 3 and that Mode B.C in 03_failures_interventions.md maps to it.

### [E-009] Unsupported inference that the most dangerous barriers are the hardest to detect (was F-011)
- Location: note after the table (lines 72-75)
- Severity: Note
- Type: Invalid inference (secondary: none)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 74 "This is not a coincidence - the most dangerous barriers tend to be the hardest to detect."
- Upstream anchor: 03_architecture/01_compute_tiers.md:78-81 ranks the same barriers by cost only.
- Why this is an error: nothing in this chapter or upstream defines or measures "danger", and the cheap rows include the hard stops BarrierLock, BarrierSat and BarrierInput. The sentence presents a causal claim as a consequence of the table when the table supports no such ordering.
- Impact on downstream results: none.
- Fix guidance:
  1. Delete the sentence, or replace it with "some of the hardest-to-detect barriers (Bode, Freq, Vac) are also slow to manifest, which is why they need offline checks".
- Required new assumptions/permits: none.
- Validation plan: read-through.

### [E-010] The information-control trade-off is one-sided against the coupling window (was F-009)
- Location: B.1 (lines 152-166)
- Severity: Note
- Type: Conceptual (secondary: none)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 152-154 "**The Information-Control Tradeoff (BarrierScat vs BarrierCap):** ... *Conflict:* High compression (anti-collapse) removes details needed for fine control (capacity/controllability)."; line 166 "If control performance drops, decrease $\beta_K,\beta_n,\beta_{\mathrm{tex}}$ (allocate more bits to the shutter)."
- Upstream anchor: 04_control/03_coupling_window.md:121-128 "$\epsilon \le I(X_t;K_t)$ and $H(K_t)\le \log|\mathcal{K}|-\epsilon$"; this chapter line 55 (two-sided window penalty); line 58 (BarrierCap).
- Why this is a clarification: lowering the rate weights raises $I(X;K)$, which relieves the grounding edge of the window and BarrierCap together; it only pushes $H(K)$ toward the saturation edge. The genuine tension is therefore between the dispersion edge $H(K)\le\log|\mathcal{K}|-\epsilon$ and BarrierCap, and the Pareto search is one-sided. The functional on lines 157-164 is a standard rate-distortion Lagrangian and is fine. The misleading gloss "(anti-collapse)" is logged under E-004.
- Impact on downstream results: 10_appendices/06_losses.md:566 F.7.2 "Purpose" line; no quantitative result depends on it.
- Fix guidance:
  1. State the conflict as "the dispersion edge of the coupling window vs. BarrierCap: bits added for controllability push $H(K)$ toward saturation", and note that the lower edge is co-aligned with BarrierCap.
- Required new assumptions/permits: none.
- Validation plan: read-through against 03_coupling_window.md.

### [E-011] Stability-plasticity dilemma pairs a world-model barrier with the policy's ZenoCheck (was F-007)
- Location: B.2 (lines 176-184)
- Severity: Minor
- Type: Definition mismatch (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter (10_appendices/06_losses.md:586 cites a nonexistent "BarrierPZ" for the same dilemma)
- Claim (verbatim): line 176-177 "**The Stability-Plasticity Dilemma (BarrierVac vs ZenoCheck (Node 2)):** *Conflict:* A stable World Model (model stability limit) resists updating to new dynamics (plasticity / Zeno)."; line 181 "$\mathcal{L}_{\text{EWC}} = \sum_i F_i (\theta_i - \theta^*_{i,old})^2$".
- Upstream anchor: 02_sieve/01_diagnostics.md:95 "| **2** | **ZenoCheck ($\mathrm{Rec}_N$)** | **Policy** | **Action Frequency Limit** | Switching policies too fast? | $D_{\mathrm{KL}}(\pi_t \Vert \pi_{t-1})$ (Smoothness) |"; :147 "Take Node 2, the ZenoCheck ... The check asks: "Is the policy switching too fast?""; 10_appendices/06_losses.md:586 "Addresses the Stability-Plasticity Dilemma (BarrierVac vs BarrierPZ)" (no `BarrierPZ` label exists in Volume 1).
- Why this is an error: Node 2 is a policy check on action-switching frequency; it does not measure world-model plasticity, and the EWC penalty acts on world-model parameters $\theta$. The two named sides of the dilemma act on different components, so the pairing does not describe a trade-off in the sense of the section preamble (line 144). The side that competes with BarrierVac is world-model plasticity: the volatility scale $\gamma$ of 01_diagnostics.md:210-214 or Node 5 (forward-consistency drift). The EWC recipe itself is coherent, which is why this is a mislabel rather than a broken result.
- Impact on downstream results: 10_appendices/06_losses.md:576-590 (F.7.3) inherits the mispairing; no theorem depends on it.
- Fix guidance:
  1. Rephrase the heading as "BarrierVac vs. WM plasticity (volatility scale $\gamma$ / forward-consistency drift, Node 5)".
  2. If a Node 2 link is intended (policy chattering induced by an over-plastic WM), state that causal chain explicitly instead of naming Node 2 as the opposing barrier.
  3. Replace "BarrierPZ" in 06_losses.md:586 accordingly.
- Required new assumptions/permits: none.
- Validation plan: grep Volume 1 for "Stability-Plasticity" and "BarrierPZ" after the edit.

## Scope restrictions and clarifications
- The chapter is a catalogue and recipe list; none of its formulas is used as a step in a proof elsewhere in Volume 1. All findings therefore concern the correctness of individual recipes and their consistency with upstream definitions, not the validity of downstream theorems.
- The Bode integral (E-007) and the requisite-variety law (E-008) are external results imported without an internal permit; the fixes above state the hypotheses under which they apply.
- The BarrierVac row (line 57) prescribes a critic-Hessian regularizer while B.2 treats the same barrier with EWC on world-model parameters; this is not logged as an error because the two address different aspects (basin metastability vs. parameter retention), but the row could say so.

## Proposed edits (optional)
- Rename the "Regularization Factor" column and tag constraint-type entries (E-001).
- Rewrite A.1 with sample-level squashing and an $\ell_\infty$ constraint (E-002).
- Rewrite line 195 with the correct Bode hypothesis (E-007) and line 67 with an actuator-side variety condition (E-008).

## Open questions
- Should Node 9 / BarrierOmin be a first-derivative (Lipschitz) or second-derivative (Hessian) bound? The book currently uses both; the choice fixes the cost column in two chapters.
- Is BarrierVariety intended as a policy-side (actuator repertoire) or a representation-side (codebook capacity) condition? The row mixes the two.

## Rejected candidate findings
None. All eleven stage-1 findings were confirmed or kept with adjusted severity (F-003 Moderate to Minor; F-007 Moderate to Minor; F-009 Minor to Note; F-011 Minor to Note).
