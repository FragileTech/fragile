# Mathematical Review: docs/source/1_agent/04_control/03_coupling_window.md

## Metadata
- Reviewed file: docs/source/1_agent/04_control/03_coupling_window.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (245 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/01_foundations/01_definitions.md:228-232` (internal state split $Z_t=(K_t,z_n,z_{\rm tex})$; Node 13: $I(X;K)>0$) and `:342` (permutation covariance of $K$)
  - `docs/source/1_agent/01_foundations/02_control_loop.md:1462` (Sieve condition citing this theorem)
  - `docs/source/1_agent/02_sieve/01_diagnostics.md:38,77` (halt semantics), `:97` (Node 4), `:111` (Node 13 row), `:700-719` (exploration loss, $\mathcal L_{\rm window}$)
  - `docs/source/1_agent/02_sieve/02_limits_barriers.md:55` (BarrierScat window penalty)
  - `docs/source/1_agent/04_control/02_belief_dynamics.md:228-244` (coupling dilemma), `:401-412` (NEPCheck), `:474` (macro belief $p_t(k)$)
  - `docs/source/1_agent/05_geometry/01_metric_law.md:61,145-148,372` (DPI / boundary-capacity constraint)
  - `docs/source/1_agent/10_appendices/01_derivations.md:48,838`; `docs/source/1_agent/10_appendices/04_faq.md:150-158`
  - `docs/references.bib:447,455,697` (`cuturi2013sinkhorn`, `leonard2014schrodinger`, `kumar2020conservative`)
  - Mechanical cross-reference report: no dangling references or duplicate labels in this file.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 4
- Minor: 6
- Notes: 1
- Primary themes:
  1. The central `prf:theorem` has an undefined antecedent, no proof, and an undefined relation $\gtrsim$; its provable content is a one-line consequence of $I(X;K)\le H(K)$ (E-004).
  2. The admissible threshold range $0<\epsilon<\log|\mathcal K|$ is infeasible on its upper half: $I\le H(K)$ forces $\epsilon\le\tfrac12\log|\mathcal K|$ (E-005). This range is inherited by two Sieve penalties.
  3. The dispersion clause is stated on the marginal symbol entropy $H(K_t)$, which does not detect grounding loss and, at the margin, fights balanced codebook utilisation; the chapter itself reads $H(K_t)$ sometimes as marginal usage entropy and sometimes as posterior belief entropy, and a downstream FAQ already cites the theorem with the opposite sign (E-006, E-011).
  4. The two rate definitions are written as expectations of deterministic functionals; "along typical trajectories" is ill-typed, and the mixing-rate formula does not implement its own definiendum (E-002, E-003).
  5. The "Connection to RL #9" box contains a garbled CQL objective, a false $\epsilon\to0$ limit, an unsupported DPI attribution, and describes the window as a Sieve halt when the Sieve implements it as a soft penalty (E-007 to E-010).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Bridge prose, line 27 | Note | Conceptual | External | this chapter | SAC / soft policy iteration equated with the two-marginal Schrödinger bridge; it is the one-endpoint KL-control specialisation. |
| E-002 | `def-grounding-rate`, lines 85-88; theorem, line 123; admonition, line 161 | Minor | Notation conflict | Framework | this chapter | $G_t=I(X_t;K_t)$ is a functional of the joint law, not a random variable; "along typical trajectories" is ill-typed; line 161 reads $H(K_t)$ as a belief entropy. |
| E-003 | `def-mixing-rate`, lines 104-107; prose 115-117 | Minor | Definition mismatch | Framework | this chapter | Formula counts all positive increments of $H(K_t)$; nothing implements "not attributable to purposeful exploration"; vanishes in any stationary regime. |
| E-004 | Theorem `thm-information-stability-window-operational`, lines 120-141 | Moderate | Proof gap / omission | Framework | this chapter | Undefined hypothesis "stable, grounded macrostates", no proof, undefined $\gtrsim$; cited as a theorem in seven other places. |
| E-005 | Theorem, lines 123-126 | Moderate | Parameter inconsistency | Framework | this chapter | Range $0<\epsilon<\log\lvert\mathcal K\rvert$ is empty for $\epsilon>\tfrac12\log\lvert\mathcal K\rvert$ because $I\le H(K)$. |
| E-006 | Theorem, lines 126, 137; prose 149, 161 | Moderate | Conceptual | Framework | this chapter | Marginal $H(K_t)$ bound does not detect grounding loss and caps achievable $I(X;K)$; marginal vs. posterior entropy conflated. |
| E-007 | Connection to RL #9, lines 176, 193 | Minor | Algorithm mismatch | Framework | this chapter | "The Sieve halts execution" on a window violation; the Sieve enforces the window by a squared-ReLU penalty, only Node 13 ($I>0$) is a gate. |
| E-008 | Connection to RL #9, line 185 | Minor | Notation conflict | External | this chapter | CQL display: outer $a\sim\mu$ shadowed by inner $\sum_a$; $\alpha$ missing; CQL($\mu$) and CQL($\mathcal H$) mixed. |
| E-009 | Connection to RL #9, lines 178-179, 190 | Moderate | Invalid inference | Framework | this chapter | "CQL is the $\epsilon\to0$ limit": as $\epsilon\to0$ both constraints become vacuous, not soft. |
| E-010 | Connection to RL #9, line 195 | Minor | Proof gap / omission | Framework | this chapter | "Derived from the Data Processing Inequality": no such derivation; the volume's DPI gives an upper bound on $I$, the wrong direction. |
| E-011 | Theorem, line 126, as cited by `10_appendices/04_faq.md:155` | Minor | Citation / reference error | Framework | upstream docs/source/1_agent/10_appendices/04_faq.md | FAQ states the theorem "requires $H(K)\approx\log\lvert\mathcal K\rvert$", the opposite of line 126, and attributes entropy monitoring to Node 4. |

## Detailed findings

### [E-001] Soft policy iteration is the one-endpoint KL-control specialisation, not the two-marginal bridge (was F-001)
- Location: feynman-prose after the Roadmap, lines 21-27; admonition "The Schrödinger Bridge Picture", lines 40-51
- Severity: Note
- Type: Conceptual (secondary: External dependency)
- Criterion: External
- Origin: this chapter
- Claim: line 27, "When you do soft policy iteration, when you use SAC, when you add entropy bonuses to your objective---you're implicitly solving a Schrödinger bridge problem."
- Upstream anchor: line 43 of this chapter, "Given a reference dynamics ... and two marginals (a prior belief and a boundary-conditioned posterior), the bridge problem finds the path measure closest in KL to the reference subject to matching the marginals" ({cite}`leonard2014schrodinger`, `docs/references.bib:455`).
- Why this is an error: the Schrödinger problem constrains both the initial and terminal marginals; KL-regularised control (soft policy iteration, SAC) fixes only the initial law and adds an expected path cost, leaving the terminal law free. The constraint sets and the optimality structure differ (a single backward potential vs. a forward/backward pair). The admonition on line 43 is correct; only the prose asserts the outright equivalence.
- Impact on downstream results: none; the section is explicitly optional (line 21). Logged so the equivalence is not later cited as a result.
- Fix guidance:
  1. On line 27 replace "you're implicitly solving a Schrödinger bridge problem" by "you're solving the one-endpoint (KL-control) specialisation of the bridge problem; pinning the terminal belief as well gives the full bridge."
- Required new assumptions/permits: none.
- Validation plan: reread lines 27 and 43 for consistency.

### [E-002] Grounding rate written as an expectation of a non-random functional; "along typical trajectories" is ill-typed (was F-002)
- Location: `def-grounding-rate`, lines 82-93; theorem statement, line 123; admonition, line 161
- Severity: Minor
- Type: Notation conflict (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim: lines 85-88, "Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13). The *grounding rate* is the average information inflow per step: $\lambda_{\text{in}} := \mathbb{E}[G_t]$."; line 123, "such that, along typical trajectories, $\epsilon \le I(X_t;K_t)$"; line 161, "You can compute $H(K_t)$ from your belief distribution."
- Upstream anchor: `01_foundations/01_definitions.md:232`, "Node 13 (BoundaryCheck): the channel is open in the only well-typed sense: $I(X;K)>0$"; `10_appendices/01_derivations.md:48`, "$I_{\text{bulk}}\approx \mathbb{E}[I(X;K)]$ (Node 13)"; `05_geometry/01_metric_law.md:145`, "$C_{\partial}\approx\mathbb{E}[I(X_t;K_t)]\le\log|\mathcal{K}|$".
- Why this is an error: $I(X_t;K_t)=D_{\rm KL}(P_{X_tK_t}\,\|\,P_{X_t}\otimes P_{K_t})$ is a deterministic number determined by the joint law at time $t$. The notation $\mathbb E[I(X;K)]$ recurs in the volume (derivations, metric law) and is defensible if $\mathbb E$ denotes a time or batch average; with that reading the definition is consistent with how the Sieve evaluates it (`02_sieve/01_diagnostics.md:711`, `02_belief_dynamics.md:401-406`). What cannot be repaired by that reading is the quantifier "along typical trajectories" on line 123, which ranges a non-random quantity over sample paths, and line 161, which reads $H(K_t)$ (defined on line 104 as the entropy of the random symbol) as the entropy of a belief.
- Impact on downstream results: the rate balance (line 132) and the "audited online" remark (line 139) presuppose per-step random readings; the Sieve uses batch scalars.
- Fix guidance:
  1. In `def-grounding-rate` state explicitly what $\mathbb E$ averages over (time window or minibatch), or replace $G_t$ by the pointwise information density $\iota(X_t;K_t)=\log\frac{p(K_t\mid X_t)}{p(K_t)}$ so that $\lambda_{\rm in}=\mathbb E[\iota]=I(X_t;K_t)$ is a genuine expectation.
  2. Delete "along typical trajectories" on line 123 (or, if the pointwise density is adopted, replace by an explicit time-average statement).
  3. On line 161 replace "from your belief distribution" by "from the empirical code-usage distribution" or, if the belief is meant, change the object (see E-006).
- Required new assumptions/permits: an AEP-type permit only if the pointwise-density route with "typical trajectories" is chosen.
- Validation plan: check that every use of $\lambda_{\rm in}$, $I(X_t;K_t)$ and $H(K_t)$ in this chapter names the same estimator as `02_sieve/01_diagnostics.md:711`.

### [E-003] Mixing-rate formula does not implement "not attributable to purposeful exploration" (was F-003)
- Location: `def-mixing-rate`, lines 101-112; prose, lines 114-118
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 104-107, "Let $S_t:=H(K_t)$ be the macro entropy. The *mixing rate* is the expected entropy growth not attributable to purposeful exploration: $\lambda_{\text{mix}} := \mathbb{E}[(S_{t+1}-S_t)_+]$."
- Upstream anchor: exploration objective for comparison, `02_sieve/01_diagnostics.md:700-706`, "$\mathcal L_{\text{expl}} := -\sum_{h} w_h\,S_c(K_t,h;\pi)$ ... a computable proxy is the entropy of the WM-predicted horizon marginals".
- Why this is an error: the display is the positive part of the total increment of $H(K_t)$; no term subtracts or conditions on exploration-driven growth, so the definiendum and the formula name different quantities. In addition, $S_t=H(K_t)$ is a deterministic sequence (E-002), so the outer expectation is vacuous, and in any stationary regime $S_{t+1}=S_t$ gives $\lambda_{\rm mix}=0$; the balance $\lambda_{\rm in}\gtrsim\lambda_{\rm mix}$ then reduces to $I>0$, i.e. to Node 13.
- Impact on downstream results: the interpretation on lines 115-117 and 151 ("spreading that happens *despite* your observations") depends on the exploration-adjusted reading.
- Fix guidance:
  1. Either delete "not attributable to purposeful exploration", or define $\lambda_{\rm mix}:=\mathbb E[(S_{t+1}-S_t)_+]-\Delta_{\rm expl}$ with an explicit exploration credit $\Delta_{\rm expl}$ (e.g. the entropy increase predicted by the world model under the exploration policy).
  2. If the balance is to carry content beyond Node 13, make $S_t$ a genuinely random per-step quantity, e.g. the posterior entropy $H(p_t)$ with $p_t$ the macro belief (`02_belief_dynamics.md:474`); see E-006.
- Required new assumptions/permits: none for the deletion; a definition of $\Delta_{\rm expl}$ for the second option.
- Validation plan: verify $\lambda_{\rm mix}$ is nonzero on a stationary synthetic run under the revised definition.

### [E-004] Theorem with undefined hypothesis, no proof, and undefined relation $\gtrsim$ (was F-006)
- Location: `thm-information-stability-window-operational`, lines 120-141
- Severity: Moderate
- Type: Proof gap / omission (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim: line 123, "A necessary condition for stable, grounded macrostates is the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that ..."; line 132, "$\lambda_{\text{in}} \gtrsim \lambda_{\text{mix}}$"; line 139, "strengthening it to a sufficient condition requires specifying the macro kernel class and a contraction inequality".
- Upstream anchor: "stable, grounded macrostates" occurs nowhere else in `docs/source/1_agent` (grep). The label is cited as a theorem at `01_foundations/02_control_loop.md:1462`, `02_sieve/01_diagnostics.md:709`, `04_control/02_belief_dynamics.md:233,239`, `intro_agent.md:356`, `05_geometry/01_metric_law.md:148`, `10_appendices/01_derivations.md:48,176`, `10_appendices/04_faq.md:155`.
- Why this is an error: a necessity claim $P\Rightarrow Q$ needs a definition of $P$ and an argument. Neither is present: the antecedent is undefined, no proof block follows, and the Remark discusses only why sufficiency is not claimed. As written, the two inequalities function as the definition of "grounded" and "non-dispersed" at threshold $\epsilon$, and $\gtrsim$ has no defined meaning. Nothing is proved, yet eight other places lean on the label as a theorem.
- Impact on downstream results: all citations listed above; lines 170 and 242 of this chapter.
- Fix guidance:
  1. Relabel the block as `prf:definition` (Coupling window at threshold $\epsilon$).
  2. Add a `prf:proposition` with proof for the provable content: for discrete $K_t$, $I(X_t;K_t)=H(K_t)-H(K_t\mid X_t)\le H(K_t)$; hence $I(X_t;K_t)\ge\epsilon$ implies $H(K_t)\ge\epsilon$ and $H(K_t\mid X_t)\le H(K_t)-\epsilon$.
  3. Replace $\gtrsim$ by an explicit inequality with a named slack, $\lambda_{\rm in}\ge\lambda_{\rm mix}-\delta$, or delete the clause until E-003 is resolved.
  4. If a genuine necessity theorem is wanted, define "grounded" (e.g. $H(K_t\mid X_{\le t})\le\delta$) and "stable" (e.g. $\sup_t H(p_t)\le\log|\mathcal K|-\epsilon$) and derive the window from them.
- Required new assumptions/permits: none for steps 1-3.
- Framework-first proof sketch for the fix: the chain rule $I(X;K)=H(K)-H(K\mid X)$ and $H(K\mid X)\ge0$ suffice.
- Validation plan: build the book and check every `{prf:ref}` to the label still resolves and reads correctly as a definition/proposition.

### [E-005] Admissible range $0<\epsilon<\log|\mathcal K|$ is infeasible on its upper half (was F-004)
- Location: theorem statement, lines 123-126
- Severity: Moderate
- Type: Parameter inconsistency (secondary: Computational error)
- Criterion: Framework
- Origin: this chapter
- Claim: lines 123-126, "the existence of constants $0<\epsilon<\log|\mathcal{K}|$ such that ... $\epsilon \le I(X_t;K_t) \quad\text{and}\quad H(K_t)\le \log|\mathcal{K}|-\epsilon$".
- Upstream anchor: the range is restated at `02_sieve/01_diagnostics.md:709`, "With thresholds $0<\epsilon<\log|\mathcal{K}|$", and the equal-margin penalty appears at `02_sieve/02_limits_barriers.md:55`; both cite this theorem as their source.
- Why this is an error: for discrete $K_t$, $I(X_t;K_t)\le H(K_t)$. Chaining the two displayed inequalities gives $\epsilon\le I\le H(K_t)\le\log|\mathcal K|-\epsilon$, so $2\epsilon\le\log|\mathcal K|$. For every $\epsilon\in(\tfrac12\log|\mathcal K|,\log|\mathcal K|)$ the window is empty. Recomputation with $|\mathcal K|=512$: $\log 512=6.238$, half $=3.119$; $\epsilon=3.0$ is feasible ($I\ge3$, $H\le3.238$); $\epsilon=3.2$ demands $I\ge3.2$ and $H\le3.038$, impossible; $\epsilon=4$ demands $I\ge4$ and $H\le2.238$, impossible. The equal-margin parametrisation also couples the lower bound on $I$ to the upper bound on $H$ for no stated reason.
- Impact on downstream results: $\mathcal L_{\rm window}$ (`02_sieve/01_diagnostics.md:709-719`), BarrierScat (`02_limits_barriers.md:55`), `01_foundations/02_control_loop.md:1462`; any calibration sweep of $\epsilon$ up to $\log|\mathcal K|$ spends half its range on an infeasible constraint.
- Fix guidance:
  1. State $0<\epsilon\le\tfrac12\log|\mathcal K|$; or
  2. Decouple the margins: $\epsilon_I\le I(X_t;K_t)$ and $H(K_t)\le\log|\mathcal K|-\epsilon_H$ with $\epsilon_I+\epsilon_H\le\log|\mathcal K|$.
  3. Propagate the chosen form to `02_sieve/01_diagnostics.md:709` and `02_sieve/02_limits_barriers.md:55`.
- Required new assumptions/permits: none.
- Validation plan: unit check that the feasible set is nonempty at the stated upper endpoint (take a deterministic encoder with $H(K)=\log|\mathcal K|-\epsilon_H$ and $I=H(K)$).

### [E-006] Dispersion clause stated on the marginal $H(K_t)$; marginal usage entropy and posterior entropy conflated (was F-005)
- Location: theorem, lines 126 and 137; prose, lines 149 and 161; rate balance, line 132
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim: line 126, "$H(K_t)\le \log|\mathcal{K}|-\epsilon$"; line 137, "If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion - symbol dispersion (BarrierScat)."; line 149, "your belief can't be spread uniformly over all macro-states"; line 161, "You can compute $H(K_t)$ from your belief distribution."
- Upstream anchor: `02_sieve/02_limits_barriers.md:55`, "BarrierScat | Representation Collapse | VQ-VAE | Grounding Loss | Symbol channel loses grounding; macrostates become noise-like."; `04_control/02_belief_dynamics.md:240`, "Over-coupling: noisy or overly aggressive updates drive mixing; the macro register loses stable structure"; `:474`, "Belief state $p_t(k)$ | Macro belief over $\mathcal K$"; `01_foundations/01_definitions.md:342`, "be covariant to symbol permutations $S_{|\mathcal K|}$"; `10_appendices/04_faq.md:153`, "This guarantees 100% codebook utilization."
- Why this is an error: (i) "macrostates become noise-like" means $K_t$ is nearly independent of $X_t$, i.e. $I(X_t;K_t)\approx0$, which is already the first inequality. The marginal entropy is silent on it: $H(K_t)=\log|\mathcal K|$ is compatible with $I=\log|\mathcal K|$ (deterministic, balanced encoder, the best-grounded case) and with $I=0$. (ii) Since $I\le H(K)$, the cap $H(K)\le\log|\mathcal K|-\epsilon$ caps achievable grounding at $\log|\mathcal K|-\epsilon$ and penalises balanced codebook usage, which the architecture otherwise promotes (permutation covariance; codebook liveness in the FAQ). (iii) Line 104 defines $H(K_t)$ as the entropy of the random symbol (marginal usage), whereas lines 149 and 161 read it as the entropy of the belief $p_t(\cdot)$ over $\mathcal K$. These are different objects with opposite health semantics: uniform usage is good, uniform posterior is dispersion. (iv) The balance on line 132 compares $I(X_t;K_t)=H(K_t)-H(K_t\mid X_t)$, a reduction in conditional entropy, with growth of the marginal $H(K_t)$; conditioning on $X_t$ does not lower $H(K_t)$, so the two sides do not act on the same ledger. The confusion is already producing contradictory downstream readings: `10_appendices/04_faq.md:155` states the theorem "requires $H(K)\approx\log|\mathcal K|$" (see E-011).
- Impact on downstream results: BarrierScat (`02_limits_barriers.md:55`), $\mathcal L_{\rm window}$ (`01_diagnostics.md:711-717`), `02_control_loop.md:1462`, the Fragile Conclusion (line 242), FAQ line 155. Enforcing the marginal cap during training will fight codebook utilisation.
- Fix guidance:
  1. Decide which object the dispersion clause is about. If grounding: replace the second inequality by $H(K_t\mid X_t)\le\log|\mathcal K|-\epsilon$ (equivalently $I(X_t;K_t)\ge H(K_t)-\log|\mathcal K|+\epsilon$). If belief stability: define $S_t:=H(p_t)$ with $p_t$ the macro posterior and say so in `def-mixing-rate` and on line 104.
  2. Rewrite lines 137, 149, 161 to name the chosen object consistently.
  3. With the belief reading, the balance $\lambda_{\rm in}\gtrsim\lambda_{\rm mix}$ compares information received with posterior-entropy growth, matching NEPCheck (`02_belief_dynamics.md:401-406`).
  4. Rename "over-coupling" on line 137 per `02_belief_dynamics.md:240` ("over-aggressive updating"), since a larger $I(X;K)$ by itself cannot raise $H(K\mid X)$.
  5. Propagate to `02_sieve/01_diagnostics.md:711`, `02_limits_barriers.md:55`, `02_control_loop.md:1462`, and the FAQ.
- Required new assumptions/permits: none; a definition of the macro posterior $p_t$ already exists (`02_belief_dynamics.md:474`).
- Validation plan: on a synthetic run with a deterministic balanced encoder ($I=H(K)=\log|\mathcal K|$), confirm the revised clause does not fire; on a run with an uninformative encoder ($I\approx0$, $H(K)=\log|\mathcal K|$), confirm it does.

### [E-007] "The Sieve halts execution" on a window violation contradicts the Sieve's soft enforcement (was F-007)
- Location: Connection to RL #9, lines 176 and 193
- Severity: Minor
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim: line 176, "If violated, the Sieve halts execution (BoundaryCheck failure)."; line 193, "BoundaryCheck halts execution when grounding fails---the agent cannot \"pay the fine\" and proceed".
- Upstream anchor: `02_sieve/01_diagnostics.md:709-719`, "Information–Stability Window ... $\mathcal L_{\text{window}} := \mathrm{ReLU}(\epsilon - I(X_t;K_t))^2 + \mathrm{ReLU}(H(K_t)-(\log|\mathcal K|-\epsilon))^2$. This is an explicit online enforcement of the coupling window"; `02_limits_barriers.md:55` (BarrierScat "Window Penalty"); Node 13 row `01_diagnostics.md:111`, "$I(X;K)$ (Symbolic MI $>0$)"; halt semantics `01_diagnostics.md:38,77`.
- Why this is an error: the Sieve's concrete enforcement of this theorem is a differentiable penalty added to the training loss, which is exactly the "pay the fine and proceed" mechanism the box says the framework forbids. Only Node 13's raw check $I(X;K)>0$ (no $\epsilon$, no $H(K)$ clause) is a gate. The contrast with CQL therefore collapses: both are soft penalties on a training objective, differing in the penalised quantity.
- Impact on downstream results: the rhetorical conclusion of the box and lines 199-202; weakens the "Result" claim (E-009).
- Fix guidance:
  1. Replace line 176 by: "Node 13 (BoundaryCheck) gates on $I(X;K)>0$ at WARN/HALT level; the $\epsilon$-window itself is enforced by the penalty $\mathcal L_{\rm window}$ (`02_sieve/01_diagnostics.md`)."
  2. Either rewrite the "Hard guarantees" bullet accordingly, or add the window thresholds to Node 13's intervention table so the claim becomes true.
- Required new assumptions/permits: none.
- Validation plan: cross-check the Node 13 row and the $\mathcal L_{\rm window}$ paragraph after the edit.

### [E-008] CQL objective is garbled (was F-008)
- Location: Connection to RL #9, displayed equation, line 185
- Severity: Minor
- Type: Notation conflict (secondary: Citation / reference error)
- Criterion: External
- Origin: this chapter
- Claim: line 185, "$\min_Q \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu}\left[\log \sum_a \exp Q(s,a)\right] - \mathbb{E}_{s,a \sim \mathcal{D}}[Q(s,a)] + \text{Bellman}.$"
- Upstream anchor: `docs/references.bib:697`, `kumar2020conservative`.
- Why this is an error: the integrand does not depend on the outer $a$, so $a\sim\mu$ is a dead variable shadowed by the bound $\sum_a$. In the cited paper the generic regulariser CQL($\mu$) is $\alpha(\mathbb E_{s\sim\mathcal D,a\sim\mu}[Q]-\mathbb E_{s\sim\mathcal D,a\sim\hat\pi_\beta}[Q])$, and CQL($\mathcal H$), obtained by optimising $\mu$ with an entropy regulariser, is $\alpha\,\mathbb E_{s\sim\mathcal D}[\log\sum_a\exp Q(s,a)-\mathbb E_{a\sim\hat\pi_\beta}[Q(s,a)]]$; once $\mu$ is optimised out there is no $\mu$ to sample. The temperature $\alpha$ is dropped, which matters in a box whose argument is about the strength of a soft penalty.
- Impact on downstream results: none beyond the expository box.
- Fix guidance:
  1. Replace the display by $\min_Q\ \alpha\,\mathbb E_{s\sim\mathcal D}\big[\log\sum_a\exp Q(s,a)-\mathbb E_{a\sim\hat\pi_\beta(\cdot\mid s)}[Q(s,a)]\big]+\tfrac12\,\mathbb E_{(s,a,s')\sim\mathcal D}\big[(Q-\mathcal B^\pi\hat Q)^2\big]$.
- Required new assumptions/permits: none.
- Validation plan: compare against Eq. (4) of the cited paper.

### [E-009] "CQL is the $\epsilon\to0$ limit" is false: the limit is vacuous, not soft (was F-009)
- Location: Connection to RL #9, "Degenerate Limit" lines 178-179 and "Result" line 190
- Severity: Moderate
- Type: Invalid inference (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim: line 190, "**Result:** CQL is the $\epsilon \to 0$ limit where coupling constraints become soft penalties rather than hard firewalls."
- Upstream anchor: theorem line 126; penalty form `02_sieve/01_diagnostics.md:711-717`.
- Why this is an error: $I(X_t;K_t)\ge0$ and $H(K_t)\le\log|\mathcal K|$ hold for every joint law, so as $\epsilon\to0$ both constraints $\epsilon\le I$ and $H\le\log|\mathcal K|-\epsilon$ become tautologies; the feasible set is the whole space and $\mathcal L_{\rm window}\to0$ identically. A hard constraint becomes a penalty by Lagrangian relaxation with a finite multiplier at fixed $\epsilon$ (which is what the Sieve already does), not by shrinking the threshold. Independently, no correspondence between $\epsilon$ and CQL's $\alpha$, or between $I(X;K)$ and $\log\sum_a\exp Q-\mathbb E_{\hat\pi_\beta}Q$, is given or apparent, so "special case" is not established.
- Impact on downstream results: the expository box and lines 199-202; the "Connection to RL" series presents itself as a set of exact degenerate limits, so a false limit undermines the series.
- Fix guidance:
  1. Replace line 190 by: "Relaxing the window constraint by a Lagrange multiplier (the Sieve's $\mathcal L_{\rm window}$) yields a soft penalty; CQL is analogous in form (a penalty against unsupported estimates), not a limit of the window theorem."
  2. Delete "$\epsilon\to0$" and rewrite lines 178-179 as "Replace the constraint by a penalty with finite weight".
- Required new assumptions/permits: none.
- Validation plan: reread the box for any remaining claim of an exact limit.

### [E-010] "Derived from the Data Processing Inequality" is unsupported (was F-010)
- Location: Connection to RL #9, bullet "Information-theoretic grounding", line 195
- Severity: Minor
- Type: Proof gap / omission (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim: line 195, "**Information-theoretic grounding**: The constraint is derived from the Data Processing Inequality, not a heuristic penalty term".
- Upstream anchor: the volume's DPI is `05_geometry/01_metric_law.md:61` (`def-dpi-boundary-capacity-constraint`), used at `:145-148`, "$C_{\partial}\approx\mathbb{E}[I(X_t;K_t)]\le\log|\mathcal{K}|$, which is exactly Node 13 (BoundaryCheck) and Theorem ...'s grounding condition", and at `:372`, "$I_{\text{bulk}} \le C_\partial \le \log|\mathcal{K}|$"; also `10_appendices/01_derivations.md:838`.
- Why this is an error: no DPI step appears in this chapter, and the theorem has no proof (E-004). The DPI in the volume yields upper bounds on mutual information ($I\le C_\partial\le\log|\mathcal K|$; along $X_t\to Z_t\to K_t$, $I(X_t;K_t)\le I(X_t;Z_t)$). Upper bounds cannot produce the lower bound $\epsilon\le I(X_t;K_t)$ and say nothing about $H(K_t)\le\log|\mathcal K|-\epsilon$. The metric-law passage that calls the DPI bound "exactly" the grounding condition is itself loose, but that is outside this chapter.
- Impact on downstream results: none mathematical; it misrepresents the status of the theorem.
- Fix guidance:
  1. Delete the bullet, or replace it with "the thresholds are information-theoretic quantities ($I$, $H$) rather than value-function penalties", which is what is true.
  2. If a DPI mention is kept, point it at `def-dpi-boundary-capacity-constraint` and state that it bounds $I$ from above only.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter for "Data Processing" after the edit.

### [E-011] Downstream FAQ cites the theorem for the opposite condition (added by verifier, V-001)
- Location: theorem line 126, as cited by `docs/source/1_agent/10_appendices/04_faq.md:155`
- Severity: Minor
- Type: Citation / reference error (secondary: Conceptual)
- Criterion: Framework
- Origin: upstream docs/source/1_agent/10_appendices/04_faq.md
- Claim: `04_faq.md:155`, "**Entropy monitoring.** Theorem {prf:ref}`thm-information-stability-window-operational` requires $H(K) \approx \log |\mathcal{K}|$. If entropy drops (collapse), **ScaleCheck (Node 4)** fails."
- Upstream anchor: this chapter, line 126, "$H(K_t)\le \log|\mathcal{K}|-\epsilon$" and line 137, "If $H(K)\approx \log|\mathcal{K}|$: over-coupling or dispersion"; Node 4 row `02_sieve/01_diagnostics.md:97`, "ScaleCheck ... Adaptation Scaling ... $\Vert\nabla\theta\Vert/\Vert\Delta S\Vert$".
- Why this is an error: the FAQ attributes to the theorem the condition $H(K)\approx\log|\mathcal K|$, which the theorem lists as the dispersion failure mode. It also names Node 4 as an entropy monitor, but Node 4 checks adaptation scaling. This is a direct symptom of the marginal-vs-posterior ambiguity in E-006: the FAQ is reading $H(K)$ as codebook-usage entropy (which liveness wants high), while the theorem bounds it from above.
- Impact on downstream results: a reader following the FAQ will tune the codebook in the direction the theorem forbids; the contradiction cannot be resolved until E-006 fixes the object.
- Fix guidance:
  1. Resolve E-006 first.
  2. In the FAQ replace the sentence by "Codebook-utilisation monitoring keeps the marginal usage entropy high (a liveness goal separate from the window theorem); the window theorem bounds [the chosen object] away from $\log|\mathcal K|$ from above", and name the node that actually monitors utilisation.
- Required new assumptions/permits: none.
- Validation plan: after the edits, grep the volume for `thm-information-stability-window-operational` and check each citation states the inequality with the correct direction.

## Scope restrictions and clarifications
- $\log|\mathcal K|$ is the maximum entropy of $K_t$ only for a discrete codebook of size $|\mathcal K|$ with a single active code per step; the whole window is stated for that setting.
- The section on the Schrödinger bridge (lines 1-51) is explicitly optional and carries no downstream dependency; E-001 is a clarity note only.
- The $\mathbb E[\cdot]$ notation in the two rate definitions is read here as a time or batch average, matching its use in `10_appendices/01_derivations.md:48` and `05_geometry/01_metric_law.md:145`; the residual issues are the quantifier "along typical trajectories" and the marginal/posterior conflation.
- The "Connection to RL #9" box is expository; E-007 to E-010 do not affect the formal content of the chapter but do affect how the theorem's status and enforcement are represented.

## Open questions
- Which object is the dispersion clause meant to constrain: the marginal code-usage entropy $H(K_t)$ or the posterior entropy $H(p_t)$ over $\mathcal K$? The answer determines the fixes for E-003, E-006 and E-011 and the sign of the FAQ's guidance.
- Is the $\epsilon$-window intended to be a gate (HALT) or a penalty? The chapter says gate; the Sieve implements a penalty (E-007).
- Should the two margins ($\epsilon_I$ on $I$, $\epsilon_H$ on $H$) be decoupled when E-005 is fixed, given they have different operational meanings?

## Rejected candidate findings
- None. All ten stage-1 findings were confirmed (eight) or kept with adjustments (F-002: type changed to Notation conflict and the definition itself accepted as a volume convention; F-010: upstream anchor corrected to the metric-law DPI definition). One finding was added by the verifier (E-011).
