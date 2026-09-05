# Mathematical Review: docs/source/1_agent/03_architecture/03_optimization.md

## Metadata
- Reviewed file: docs/source/1_agent/03_architecture/03_optimization.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (357 lines)
- Framework anchors (definitions/axioms/permits):
  - Local: A1-A5 (lines 69-82); `def-preconditioned-update`; `thm-preconditioned-descent`; `def-relative-trust-region`; `lem-trust-region-scaling`; `prop-varentropy-brake-discrete`; `def-gradient-alignment`; `prop-alignment-step-damping`; `prop-snr-gate`; `def-log-lr-conduction`; `prop-conduction-contracts`; `thm-optimizer-conditional-stability`
  - `docs/source/1_agent/07_cognition/02_governor.md` (`cor-varentropy-brake`, lines 455-470; Governor outputs line 217; pathology table lines 700-717)
  - `docs/source/1_agent/10_appendices/05_proofs.md` (E.10, lines 945-1012)
  - `docs/source/1_agent/06_fields/02_reward_field.md` (lines 784-790)
  - `docs/source/1_agent/02_sieve/01_diagnostics.md` (`sec-d-policy-regulation`, lines 562-576)
  - `docs/source/1_agent/10_appendices/04_faq.md` (lines 842-856), `docs/source/1_agent/11_implementation/02_world_model.md` (line 1183) for downstream use

## Executive summary
- Critical: 0
- Major: 1
- Moderate: 2
- Minor: 3
- Notes: 1
- Primary themes: Every local result in the chapter (preconditioned descent, trust-region scaling, discrete varentropy brake, SNR gate, log-LR conduction) is algebraically correct in isolation; all bounds were recomputed. The problems are in the composition. The combined Theorem `thm-optimizer-conditional-stability` claims expected descent under noise for the conjunction of the mechanisms, but noise-dependent step scaling (trust-region clipping and alignment damping driven by the noisy gradient) destroys the expected-descent argument; an explicit numerical counterexample satisfying A1-A3 and the SNR gate exhibits expected ascent. The SNR proposition tacitly needs the preconditioner to be independent of the current noise, which the recommended Adam surrogate violates. Per-group learning rates and conduction are not reconciled with the scalar-step descent theorem, and the order of application of the six mechanisms is unspecified. Three smaller items: a sign miswording in the trust-region proof, unsupported "oscillation brake" wording, and an inherited mismatch between the statement and the proof of the upstream varentropy-brake corollary. All cross-references resolve.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Sec. 2, proof of `lem-trust-region-scaling` (163) | Minor | Miswording | External | this chapter | Linear coefficient of the quadratic in $s$ is negative, not positive |
| E-002 | Sec. 3, implementation note (195-196); Sec. 7 (324-325) | Minor | Parameter inconsistency | Framework | upstream `07_cognition/02_governor.md`, `10_appendices/05_proofs.md` | "Obeys the adiabatic constraint" needs the unstated condition $\eta_T\le 2C\sqrt\gamma$ |
| E-003 | Sec. 3, `prop-varentropy-brake-discrete` (186); Sec. 7 (324-325) | Note | Scope restriction | Framework | this chapter | Monotone brake excludes Governor reheating steps |
| E-004 | Sec. 4 (200, 234-235); checklist item 4 (352) | Minor | Miswording | Framework | this chapter | Oscillation-suppression claims not supported by the proved statement; $m_t$ enters no update |
| E-005 | Sec. 5, `prop-snr-gate` proof (265-268); Sec. 8 (339) | Moderate | Proof gap / omission | External | this chapter | Cross term $\mathbb E[g_t^\top M_t\xi_t]$ vanishes only if $M_t$ is independent of $\xi_t$; Adam surrogate violates this |
| E-006 | Sec. 6 (281) and Sec. 7 (315-323) | Moderate | Proof gap / omission | Framework | this chapter | Per-group rates and conduction not reconciled with scalar-step descent; gate order unspecified |
| E-007 | Sec. 7, `thm-optimizer-conditional-stability` (313-326) | Major | Invalid inference | External | this chapter | Noise-dependent clipping/damping breaks expected descent; explicit counterexample |

## Detailed findings

### [E-001] Sign miswording in the trust-region scaling proof (was F-001)
- Location: Section 2, proof of Lemma `lem-trust-region-scaling`, lines 155-166
- Severity: Minor
- Type: Miswording (secondary: Typo)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): line 163: "The right-hand side is a quadratic in $s$ with positive linear term and nonnegative curvature. Since Theorem `thm-preconditioned-descent` guarantees descent at $s=1$, any smaller $s$ preserves nonincreasing $\mathcal{V}$."
- Upstream anchor: not applicable (defined in this chapter).
- Why this is an error: The quadratic is $\phi(s)=\mathcal V(\theta_t)-s\,g_t^\top d_t+\tfrac L2 s^2\|d_t\|^2$ with $g_t^\top d_t=\eta_t g_t^\top M_tg_t>0$ for SPD $M_t$, so the coefficient of $s$ is $-g_t^\top d_t<0$. A positive linear term with nonnegative curvature would give $\phi(s)\ge\mathcal V(\theta_t)$ for all $s\ge0$, the opposite of what is needed. The conclusion is correct: $\phi(s)-\mathcal V(\theta_t)=s\,(-g_t^\top d_t+\tfrac L2 s\|d_t\|^2)$, the bracket is nondecreasing in $s$ and is $\le0$ at $s=1$ by the theorem, hence $\le0$ on $[0,1]$.
- Impact on downstream results: None mathematically. The same argument is reused at line 231 (`prop-alignment-step-damping`) and cited by `10_appendices/04_faq.md:846`.
- Fix guidance:
  1. Replace the sentence at line 163 by: "The right-hand side minus $\mathcal V(\theta_t)$ equals $s\,(-g_t^\top d_t+\tfrac L2 s\|d_t\|^2)$; the bracket is nondecreasing in $s$ and is nonpositive at $s=1$ by Theorem `thm-preconditioned-descent`, hence nonpositive for all $s\in[0,1]$."
- Required new assumptions/permits: none.
- Validation plan: Re-read the proof; check that the wording at line 229-231 still refers to a valid argument.

### [E-002] "Obeys the adiabatic constraint" needs a hidden parameter condition; upstream corollary states an ODE but proves an inequality (was F-006)
- Location: Section 3, implementation note, lines 195-196; Section 7, lines 324-325
- Severity: Minor
- Type: Parameter inconsistency (secondary: Citation / reference error)
- Criterion: Framework
- Origin: upstream `docs/source/1_agent/07_cognition/02_governor.md` and `docs/source/1_agent/10_appendices/05_proofs.md` (inherited here)
- Claim (verbatim): lines 195-196: "this yields an explicit annealing schedule that satisfies the varentropy brake and prevents quenching near critical points"; lines 324-325: "the temperature schedule obeys the adiabatic (varentropy) constraint".
- Upstream anchor: `07_cognition/02_governor.md:461-465`: "the cooling schedule must be modulated by the Varentropy: $\frac{dT_c}{dt}=-\eta\cdot\frac{T_c}{1+\gamma V_H(\theta_t)}$". Its proof, `10_appendices/05_proofs.md:947`: "**Statement:** To maintain stability, the cooling rate must satisfy $|\dot T_c|\ll T_c/\sqrt{V_H}$", and `:1003-1005`: "$\left|\frac{dT_c}{dt}\right|\le C\frac{T_c}{\sqrt{V_H}}$". Same inequality form at `06_fields/02_reward_field.md:786`.
- Why this is an error: The discrete rule (line 182) is the forward-Euler discretisation of the upstream ODE, so relative to the statement of `cor-varentropy-brake` the chapter is faithful. Relative to what is actually proved upstream (the adiabatic inequality), the discrete schedule gives $|T_{t+1}-T_t|=\eta_TT_t/(1+\gamma V_H)$, and $\eta_T/(1+\gamma V)\le C/\sqrt V$ for all $V>0$ holds iff $\eta_T\le\min_{V>0}C(1+\gamma V)/\sqrt V=2C\sqrt\gamma$ (AM-GM; verified numerically, e.g. $C=1.3,\gamma=0.7$ gives minimum $2.1753=2C\sqrt\gamma$). The same condition is needed for the continuous ODE with constant $\eta$, so the root cause is the upstream statement/proof mismatch; the chapter inherits it and adds an unqualified "obeys". Separately, A4 (slow $\tau_t$) is compatible with a per-step multiplicative change of up to $\eta_T$ only if $\eta_T\ll1$, which is not stated.
- Impact on downstream results: Item 4 of Theorem `thm-optimizer-conditional-stability`; checklist item 3 (line 351).
- Fix guidance:
  1. In `prop-varentropy-brake-discrete` add the hypothesis $\eta_T\le2C\sqrt\gamma$ (with $C$ the relaxation constant of E.10), and state the conclusion as "$|T_{t+1}-T_t|\le C\,T_t/\sqrt{V_H(\theta_t)}$".
  2. Note that A4 requires $\eta_T\ll1$.
  3. Upstream: restate `cor-varentropy-brake` as the inequality its proof establishes, with the $1/(1+\gamma V_H)$ law as one admissible schedule.
- Required new assumptions/permits: $\eta_T\le2C\sqrt\gamma$; $\eta_T\ll1$.
- Validation plan: Check the AM-GM bound; check that E.10 defines $C$ explicitly enough for the condition to be meaningful.

### [E-003] Discrete brake is strictly monotone; guarantee excludes Governor reheating (was F-007)
- Location: Section 3, `prop-varentropy-brake-discrete`, lines 177-193; Section 7, lines 323-325
- Severity: Note
- Type: Scope restriction
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 186: "$0 < T_{t+1} \le T_t$ (temperature is positive and nonincreasing)"; lines 324-325: "the temperature schedule obeys the adiabatic (varentropy) constraint".
- Upstream anchor: `07_cognition/02_governor.md:702`: "Saddle Point ... Increase $T_c$ (entropy injection)"; `:716`: "Treatment: inject noise (increase temperature) to escape"; `:217` lists $T_{c,t}$ among the free Governor outputs.
- Why this is an error: Not an error in the stated proposition (with multiplier in $[1-\eta_T,1)$ the temperature is in fact strictly decreasing and decays geometrically for bounded $V_H$). But the Governor is allowed to raise $T_c$, and the chapter identifies $\tau_t\propto T_t$ (line 195), so during a reheating step neither the monotonicity claim nor the temperature clause of Section 7 applies, and $\eta_t\propto T_t$ grows, so the step-size bounds of Sections 2 and 5 must be re-checked.
- Impact on downstream results: Scope of Theorem `thm-optimizer-conditional-stability` item 4.
- Fix guidance:
  1. Add "during annealing phases (no Governor-initiated heating)" to the temperature clause of the theorem.
  2. Add a sentence that after any increase of $T_c$ the bounds of `thm-preconditioned-descent` and `prop-snr-gate` must be re-verified for the new $\eta_t$.
- Required new assumptions/permits: none.
- Validation plan: Cross-check with the Governor pathology table.

### [E-004] Alignment damping is said to suppress oscillation and prevent anti-gradient updates, but the analysed update never involves $m_t$ (was F-002)
- Location: Section 4, lines 198-235; checklist item 4, line 352
- Severity: Minor
- Type: Miswording (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 200: "Alignment damping prevents updates that oppose the local gradient"; lines 234-235: "reducing $\eta_t$ acts as a rigorous oscillation brake without changing the descent direction"; line 352: "Use alignment-triggered step damping to prevent momentum-induced oscillations."
- Upstream anchor: `02_sieve/01_diagnostics.md:573-574`: "The Zeno constraint ... prevents 'chattering'---rapid oscillation between strategies."
- Why this is an error: The only update analysed is $\theta_{t+1}=\theta_t-\eta_tM_tg_t$ (lines 45, 93), which is a descent direction for every SPD $M_t$; the momentum estimate $m_t$ of `def-gradient-alignment` enters no update equation. Proposition `prop-alignment-step-damping` therefore proves only that multiplying $\eta_t$ by $\rho\le1$ preserves the one-step descent bound (a special case of `lem-trust-region-scaling`). It proves nothing about oscillation. If the intended update were $\theta_{t+1}=\theta_t-\eta_tM_tm_t$, Theorem `thm-preconditioned-descent` would not apply because $m_t$ need not be a descent direction.
- Impact on downstream results: Item 3 of `thm-optimizer-conditional-stability` and checklist item 4 inherit an unproven oscillation claim. See also E-007 for the stochastic case.
- Fix guidance:
  1. Either state that the proposition is a step-size-shrinkage result whose only guarantee is preservation of the descent bound, and delete "prevents updates that oppose the local gradient" and "rigorous oscillation brake"; or
  2. analyse the momentum update $-\eta_tM_tm_t$ under an explicit alignment hypothesis $g_t^\top M_tm_t\ge c\|g_t\|^2$ and prove descent from it (then $a_t\ge0$ becomes a gating condition rather than a damping trigger).
- Required new assumptions/permits: for option 2, the alignment hypothesis.
- Validation plan: Confirm every occurrence of "oscillation" in Sections 4, 7, 8, 9 is backed by a proved statement.

### [E-005] SNR gate proof silently requires $M_t$ independent of the current noise; the recommended Adam surrogate violates this (was F-003)
- Location: Section 5, `prop-snr-gate` and proof, lines 242-269; Section 8, first bullet, line 339
- Severity: Moderate
- Type: Proof gap / omission (secondary: Algorithm mismatch)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 265-268: "Apply the smoothness bound with $d_t = \eta_t M_t \hat g_t$ and take conditional expectations. Use $\mathbb{E}[\hat g_t] = g_t$ and $\mathbb{E}[\|\hat g_t\|^2] = \|g_t\|^2 + \mathbb{E}[\|\xi_t\|^2] \le \|g_t\|^2 + \sigma^2$. Then apply the eigenvalue bounds from A2"; line 339: "Use Adam second-moment statistics as a diagonal proxy for the trust-region regulator (bounded SPD $M_t$) to satisfy A2."
- Upstream anchor: A2 and A3 are local (lines 72-76). A2 bounds only the spectrum of $M_t$; A3 only centres $\xi_t$ given $\theta_t$. Neither constrains the joint law of $(M_t,\xi_t)$.
- Why this is an error: After conditional expectation the first-order term is $-\eta_t\big(\mathbb E[g_t^\top M_tg_t\mid\theta_t]+\mathbb E[g_t^\top M_t\xi_t\mid\theta_t]\big)$. Reaching $-\eta_tm_{\min}\|g_t\|^2$ (line 250) requires the cross term to vanish, which holds if $M_t$ is measurable with respect to the information available before $\hat g_t$ is drawn, but not in general. Adam's $v_t=\beta_2v_{t-1}+(1-\beta_2)\hat g_t^{\odot2}$ contains the current $\hat g_t$, so $M_t=\mathrm{diag}(1/(\sqrt{\hat v_t}+\epsilon))$ is correlated with $\xi_t$ and the cross term can have either sign. In addition, the actual Adam direction is $M_t\hat m_t$, not $M_t\hat g_t$, so the surrogate also changes the update direction analysed here. The remainder of the bound is correct: $\|M_t\hat g_t\|^2\le m_{\max}^2\|\hat g_t\|^2$ pointwise, $\mathbb E\|\hat g_t\|^2=\|g_t\|^2+\mathbb E\|\xi_t\|^2$ by A3, and solving $-\eta m_{\min}\|g\|^2+\tfrac L2\eta^2m_{\max}^2(\|g\|^2+\sigma^2)\le0$ gives exactly lines 255-260.
- Impact on downstream results: Item 5 of `thm-optimizer-conditional-stability`; `10_appendices/04_faq.md:847-848` cites the noise-gating bound as established.
- Fix guidance:
  1. Add to A2 (or to the proposition): "$M_t$ is measurable with respect to $\sigma(\theta_t,\hat g_{t-1},\hat g_{t-2},\dots)$", and use it explicitly in the proof to kill the cross term.
  2. In Section 8 replace the Adam bullet by a lagged preconditioner $M_t=\mathrm{diag}(1/(\sqrt{\hat v_{t-1}}+\epsilon))$ with explicit clipping to $[m_{\min},m_{\max}]$, and note that the true Adam direction $M_t\hat m_t$ lies outside the analysed class.
  3. Alternatively add a lemma bounding $|\mathbb E[g_t^\top M_t\xi_t\mid\theta_t]|$ for the specific surrogate.
- Required new assumptions/permits: the measurability hypothesis in step 1.
- Validation plan: Re-derive line 250 with the cross term written out and check where the hypothesis is used.

### [E-006] Per-group learning rates and conduction are never reconciled with the scalar-step descent theorem or with the gate ordering (was F-005)
- Location: Section 6, lines 274-309; Section 7, `thm-optimizer-conditional-stability`, lines 313-326, items 1, 2, 5, 6
- Severity: Moderate
- Type: Proof gap / omission (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 281: "Let $\eta_i > 0$ be per-group learning rates"; lines 315-323: "Under A1--A5, with updates that apply: 1. preconditioned descent ..., 2. trust-region scaling ..., 5. SNR gating ..., and 6. log-LR conduction ..., the optimizer produces a nonincreasing Lyapunov objective in the deterministic case".
- Upstream anchor: not applicable; `thm-preconditioned-descent` (lines 98-113) uses a single scalar $\eta_t$; `lem-trust-region-scaling` scales the whole step by one scalar $s$.
- Why this is an error: (a) With per-group rates the update is $\theta_{t+1}=\theta_t-D_\eta M_tg_t$, $D_\eta=\mathrm{diag}(\eta_iI)$. Descent requires $g_t^\top D_\eta M_tg_t>0$, which can fail for a general SPD $M_t$: with $D_\eta=\mathrm{diag}(1,100)$ and $M=\begin{pmatrix}1&0.9\\0.9&1\end{pmatrix}$ (eigenvalues $0.1$ and $1.9$, so A2 holds), $\min_{\|g\|=1}g^\top D_\eta Mg\approx-16.7<0$ (recomputed numerically). The theorem covers the per-group case only if $M_t$ is block-diagonal with respect to the groups, with the step-size condition restated in terms of $\max_i\eta_i$; none of this is stated. (b) Conduction is the convex combination $x_i^+=(1-k)x_i+\tfrac k2(x_{i-1}+x_{i+1})$, so $\max_ix_i^+\le\max_ix_i$, but it can raise an individual group's rate above what the SNR gate or trust region prescribed for that group, and it invalidates a trust-region clip applied before it. (c) The theorem does not fix the order of the six mechanisms; "conduction, gates, step, clip" preserves the deterministic bound, "clip, then conduction" does not. The conduction proposition itself is correct: $\nabla E=Lx$, $x^+=x-\tfrac k2Lx$, $\lambda_{\max}(L)=2-2\cos(\pi(n-1)/n)<4$ (numerically $2,3,3.618,3.902$ for $n=2,3,5,10$), and $k/2\le1/2$ gives $|1-\tfrac k2\lambda|\le1$.
- Impact on downstream results: Deterministic half of `thm-optimizer-conditional-stability`; checklist items 2, 5, 6; `10_appendices/04_faq.md:850-855`.
- Fix guidance:
  1. State the per-group update explicitly and add the hypothesis "$M_t$ is block-diagonal with respect to the parameter groups", so that $D_\eta M_t$ is SPD with spectrum in $[\min_i\eta_i\,m_{\min},\max_i\eta_i\,m_{\max}]$; the descent condition becomes $\max_i\eta_i<2m_{\min}/(Lm_{\max}^2)$.
  2. Fix the order of application in the theorem: conduction on log-rates, then SNR/alignment gates (which only shrink), then the preconditioned step, then trust-region clipping last.
  3. Add the one-line lemma "$\max_ix_i^+\le\max_ix_i$ for $k\in[0,1]$" to justify that conduction cannot violate the global step-size bound.
- Required new assumptions/permits: block-diagonality of $M_t$ (satisfied by the diagonal surrogate of Section 8).
- Validation plan: Re-run the descent bound with $D_\eta M_t$ under the block-diagonal hypothesis; check each ordering in step 2 against Lemma `lem-trust-region-scaling`.

### [E-007] Combined theorem: noise-dependent step scaling breaks expected descent; explicit counterexample (was F-004)
- Location: Section 7, `thm-optimizer-conditional-stability`, lines 313-326, items 2, 3, 5 and the conclusion
- Severity: Major
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): lines 315-324: "Under A1--A5, with updates that apply: ... 2. trust-region scaling ..., 3. alignment-triggered step damping ..., 5. SNR gating ..., the optimizer produces a nonincreasing Lyapunov objective in the deterministic case and ensures expected descent under bounded noise."
- Upstream anchor: not applicable. `lem-trust-region-scaling` (line 147) is proved for the deterministic step $d_t=\eta_tM_tg_t$; `prop-snr-gate` (line 244) for the unscaled stochastic step $\eta_tM_t\hat g_t$. No statement covers a scaled stochastic step $s(\hat g_t)\,\eta_tM_t\hat g_t$.
- Why this is an error: In the stochastic case the trust-region factor $s_t=\min(1,\kappa(\|\theta_t\|+\epsilon_\theta)/\|\eta_tM_t\hat g_t\|)$ and the alignment trigger $a_t=\hat g_t^\top m_t<0$ are random variables correlated with $\xi_t$, so $\mathbb E[s_tg_t^\top M_t\hat g_t]\ne\bar s\,g_t^\top M_tg_t$ and the first-order descent term can be wiped out while the second-order term remains. Counterexample (independently recomputed): $\mathcal V(\theta)=\theta^2/2$ ($L=1$), $M_t=1$, $\theta_t=1$, $g_t=1$, $\xi_t=\pm3$ with probability $1/2$ each ($\mathbb E\xi_t=0$, $\sigma^2=9$, SNR $=1/9$). The SNR gate allows $\eta_t\le2\cdot\frac{1/9}{1+1/9}=0.2$; take $\eta_t=0.15$. Unscaled step: $\theta_{t+1}\in\{0.4,1.3\}$, $\mathbb E\mathcal V=0.4625<0.5$ (equal to the right-hand side of `prop-snr-gate`). With the relative trust region $\kappa=0.2$, $\epsilon_\theta=0$, the steps $0.6$ and $-0.3$ are clipped to $\pm0.2$, so $\theta_{t+1}\in\{0.8,1.2\}$ and $\mathbb E\mathcal V=0.52>0.5$: expected ascent. Also $\kappa=0.3\Rightarrow0.545$, $\kappa=0.1\Rightarrow0.505$, while $\kappa=0.5\Rightarrow0.485$ (descent), so the failure depends on the clip binding asymmetrically on the noise. Alignment damping with a trigger computed from $\hat g_t$ fails the same way: with $m_t=-1$, $\rho=0.3$, the trigger fires only for $\hat g_t=4$, giving $\theta_{t+1}\in\{0.82,1.3\}$ and $\mathbb E\mathcal V=0.5906>0.5$. Hence the conjunction of items 2, 3 and 5 does not yield expected descent, and the deterministic proofs of Lemma `lem-trust-region-scaling` and Proposition `prop-alignment-step-damping` do not transfer to the noisy setting.
- Impact on downstream results: The chapter's headline result and its use in `10_appendices/04_faq.md:844-855` (Governor stability answer) and `11_implementation/02_world_model.md:1183`. The deterministic half of the theorem is unaffected by this finding (but see E-006).
- Fix guidance:
  1. Restrict "expected descent under bounded noise" to the unscaled stochastic step of `prop-snr-gate` (with the measurability fix of E-005), and state that trust-region clipping and alignment damping are proved only in the deterministic regime.
  2. If a stochastic clipping guarantee is wanted, add a lemma under an extra hypothesis that makes the clip almost surely inactive or noise-independent, e.g. an a.s. noise bound $\|\xi_t\|\le B$ together with $\eta_tm_{\max}(\|g_t\|+B)\le\kappa(\|\theta_t\|+\epsilon_\theta)$, or a bound of the form $\mathbb E[\mathcal V(\theta_{t+1})]\le\mathcal V(\theta_t)-c\,\eta_t\|g_t\|^2+\text{(clip-bias term)}$ with the bias made explicit.
  3. For alignment damping in the stochastic case, compute the trigger from noise-independent quantities (e.g. $g_{t-1}^\top m_{t-1}$ or a lagged estimate), and say so.
- Required new assumptions/permits: for step 2, an almost-sure noise bound (A3 provides only a second-moment bound).
- Validation plan: Re-run the scalar counterexample against the restated theorem; verify the new lemma's hypothesis excludes it.

## Scope restrictions and clarifications
- All guarantees are one-step and local; no convergence-rate or global claims are made (the chapter says so at lines 328-332).
- The stochastic guarantee, once repaired, covers only the unscaled step with a preconditioner independent of the current noise.
- The temperature guarantee covers only annealing phases; Governor reheating is outside the proposition.
- Per-group rates are covered only for preconditioners block-diagonal with respect to the groups.

## Open questions
- Which quantity is $C$ in E.10, and is it available to the optimizer at run time so that $\eta_T\le2C\sqrt\gamma$ can be enforced?
- Should the chapter analyse the momentum update $-\eta_tM_tm_t$ at all, given that the recommended Adam surrogate uses the first-moment direction?
- In what order does the reference implementation apply conduction, gates, step, and clip? The theorem should mirror that order.

## Rejected candidate findings
None. All seven stage-1 findings were confirmed on independent recomputation.
