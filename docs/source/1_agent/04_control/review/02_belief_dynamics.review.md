# Mathematical Review: docs/source/1_agent/04_control/02_belief_dynamics.md

## Metadata
- Reviewed file: docs/source/1_agent/04_control/02_belief_dynamics.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (481 lines)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/02_sieve/01_diagnostics.md` lines 93-124 (Sieve node table: Node 2 ZenoCheck, Node 13 BoundaryCheck, Node 22 MECCheck, Node 23 NEPCheck, Node 24 QSLCheck)
  - `docs/source/1_agent/02_sieve/03_failures_interventions.md:60` (Mode D.C, ungrounded inference)
  - `docs/source/1_agent/05_geometry/02_wfr_geometry.md` lines 106 (`def-the-wfr-action`), 355-412 (`thm-classical-master-equation-wfr`, `cor-gksl-classical-limit`, classical-limit correspondence table, `rem-full-quantum-wfr`)
  - `docs/source/1_agent/05_geometry/04_equations_motion.md` lines 248-262 (Hamiltonian structure) and 418-428 (`def-effective-potential`)
  - `docs/CLAUDE.md:218` (time-index convention: $t$ interaction, $s$ computation, $\tau$ scale)
  - `docs/_toc.yml` (chapter ordering)
  - Mechanical cross-reference pre-pass: no dangling references or duplicate labels in this file; all external labels used here resolve.

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 4
- Minor: 4
- Notes: 1
- Primary themes:
  1. The optional GKSL section identifies the Lindblad dissipator with Bayesian assimilation of a realized observation. Bayes conditioning and Sieve projection are nonlinear in the belief while the GKSL generator is linear, so the Node 22 consistency defect built on this identification is generically non-zero for a correct filter.
  2. The "classical limit" note in `def-gksl-generator` states two things that are false without extra hypotheses: the commutator does not vanish on diagonal states unless $H$ is diagonal, and the reduction to a reversible Markov jump process needs jump-type operators and detailed balance. The geometry chapter repeats the same statements.
  3. NEPCheck as written penalizes belief changes produced by the prediction step and by exact Bayes updates on individual samples, so it does not measure "ungrounded internal updating" as claimed.
  4. Several loose pointers: legacy section numerals, QSLCheck described as a generalization of ZenoCheck, and a claim that standard POMDP beliefs are "unbounded".
  The purely classical predict/update/project template and the worked example are correct (recomputed).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Feynman prose after Researcher Bridge, l.38; Over/Under Coupling, l.233 | Minor | Citation / reference error | Framework | this chapter | Legacy section numerals ("Sections 2-9", "Sections 3-6") point to the wrong parts of the book |
| E-002 | Optional: Operator-Valued Belief Updates, l.252, 285, 298; Master-Equation Consistency Defect, l.331-349 | Moderate | Conceptual (secondary: Algorithm mismatch) | Framework | this chapter | GKSL dissipator identified with Bayesian assimilation; conditioning is nonlinear, so the MEC defect cannot vanish for a correct filter |
| E-003 | `def-belief-operator` l.260; `def-gksl-generator` l.271; prose l.300 | Note | Proof gap / omission | External | this chapter | GKSL completeness stated without the semigroup hypothesis; $d$ never related to $|\mathcal K|$ |
| E-004 | `def-gksl-generator` l.274 vs. MEC defect l.343 | Minor | Notation conflict (secondary: Dimensional mismatch) | Framework | this chapter | Generator in computation time $s$, defect per interaction step $\Delta t$; no conversion stated |
| E-005 | `def-gksl-generator`, Note (WFR Correspondence), l.289 | Moderate | Computational error (secondary: Scope restriction) | Framework | this chapter (also `05_geometry/02_wfr_geometry.md:388,396`) | "The commutator term vanishes for diagonal states" is false unless $H$ is diagonal in the same basis |
| E-006 | `def-gksl-generator`, Note (WFR Correspondence), l.289 | Moderate | Scope restriction (secondary: External dependency) | External | this chapter and `05_geometry/02_wfr_geometry.md` | "Rigorously equivalent to a WFR gradient flow" drops diagonal invariance, detailed balance, and the Maas-metric/WFR identification |
| E-007 | Update vs Evidence (NEPCheck), l.389, 398-413 | Moderate | Conceptual (secondary: Parameter inconsistency) | Framework | this chapter and `02_sieve/01_diagnostics.md:121` | NEPCheck penalizes model-driven prediction and per-sample Bayes updates; evidence index mismatch |
| E-008 | Metric Speed Limit (QSLCheck), l.427 | Minor | Citation / reference error (secondary: Conceptual) | Framework | this chapter | QSLCheck (on $z_t$) is not a generalization of ZenoCheck (on $\pi_t$) |
| E-009 | Connection to RL #19, l.443, 455 | Minor | Miswording (secondary: Conceptual) | External | this chapter | "Standard POMDPs have unbounded continuous beliefs" is false for the cited finite-state setting |

## Detailed findings

### [E-001] Stale section numerals for the Sieve and geometry chapters (was F-001)
- Location: Feynman prose after the Researcher Bridge, l.38; "Over/Under Coupling" prose, l.233
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): l.38 "Sections 2-9 describe geometry, metrics, and effective macro dynamics."; l.233 "The Sieve (Sections 3-6) is the control layer that keeps the agent inside this window."
- Upstream anchor: `docs/_toc.yml` lines 12-17 (`02_sieve/*`), 27 (this chapter), 29-34 (`05_geometry/*`); `02_sieve/01_diagnostics.md:1` `(sec-diagnostics-stability-checks)=`.
- Why this is an error: The numerals come from a legacy single-document numbering. In the current book the geometry chapters follow this one, so "Sections 2-9 describe geometry" is not a valid back-reference, and the Sieve is Part II, not "Sections 3-6". A reader following the pointer lands in the wrong place.
- Impact on downstream results: none mathematical; navigation only.
- Fix guidance:
  1. Replace l.38 with "The geometry chapters (Part V) describe geometry, metrics, and effective macro dynamics" or a `{ref}` to `05_geometry/01_metric_law`.
  2. Replace "(Sections 3-6)" at l.233 with "({ref}`sec-diagnostics-stability-checks`)".
- Required new assumptions/permits: none.
- Validation plan: build the book and click the links.

### [E-002] GKSL dissipator identified with Bayesian assimilation; the MEC defect cannot be satisfied by a correct filter (was F-002)
- Location: Optional: Operator-Valued Belief Updates, l.252, 283-285, 298; Master-Equation Consistency Defect (Node 22), l.331-349
- Severity: Moderate (the stage-1 reviewer proposed Major; downgraded because the section is explicitly optional, Node 22 is marked not implemented in the Sieve table, no theorem depends on $\mathcal L_{\text{MEC}}$, and the fix is a local redefinition)
- Type: Conceptual (secondary: Algorithm mismatch)
- Criterion: Framework (the contradiction is between the chapter's own nonlinear update at l.111-113/164 and its own linear generator at l.273-278)
- Origin: this chapter (`02_sieve/01_diagnostics.md:120` imports the definition by reference)
- Claim (verbatim): l.285 "The dissipator is a structured way to represent **irreversible assimilation / disturbance** while preserving positivity and trace."; l.298 "This is where boundary information enters---where reality pokes holes in your internal model and forces corrections."; l.337-346 "If an implementation maintains an operator belief $\varrho_t$ and produces an empirical update $\varrho_{t+1}$ (e.g., after a boundary update + Sieve projection), then a **consistency defect** compares it to the GKSL-predicted infinitesimal update: $\mathcal{L}_{\text{MEC}} := \|(\varrho_{t+1}-\varrho_t)/\Delta t - \mathcal{L}_{\text{GKSL}}(\varrho_t)\|_F^2$"; l.333 "If this is small, the agent's belief dynamics are well-behaved. If it's large, something is wrong".
- Upstream anchor: `02_sieve/01_diagnostics.md:120` "| **22** | **MECCheck** | Belief / WM | CPTP Consistency | Operator update matches GKSL ({prf:ref}`def-gksl-generator`) form? | $\|(\varrho_{t+1}-\varrho_t)/\Delta t-\mathcal{L}_{\text{GKSL}}(\varrho_t)\|_F^2$ | $O(BZ^3)$ ✗ |".
- Why this is an error: The GKSL generator is linear in $\varrho$ and describes evolution averaged over unobserved environment outcomes. Assimilating a realized observation is conditioning: in the chapter's own classical case (l.111-113) $p_{t+1}=L\odot\tilde p/\langle L,\tilde p\rangle$, and the hard projection (l.164, 191) is mask-and-renormalize. Both maps are nonlinear because the normalizer depends on the state. Recomputation: with a fixed likelihood $L=(0.64,0.38,0.85)$ and two random beliefs $p_1,p_2$, the increment of the Bayes map at the midpoint differs from the average of the increments by $(0.0104,-0.0088,-0.0015)$, so no linear generator reproduces the Bayes increment on an open set of beliefs. Furthermore the outcome-average of the conditional update is the identity, $\mathbb E_x[p(\cdot\mid x)]=\tilde p$, so the Lindblad (ensemble) picture cannot contain assimilation at all; conditioning is the unravelling of a master equation (stochastic master equation / Belavkin filter), not the master equation. Hence the operational interpretation at l.285 and l.298 is wrong, and $\mathcal L_{\text{MEC}}$ as defined is generically non-zero for a correct Bayes filter with Sieve projection. The sentence at l.333 is therefore inverted for exactly the updates the chapter promotes: a correct implementation would raise Node 22 on essentially every step with informative evidence.
- Impact on downstream results: Node 22 in `02_sieve/01_diagnostics.md:120` and `06_fields/03_info_bound.md:663`; the Physics-Isomorphism box (l.324); "GKSL embedding" bullet in `05_geometry/02_wfr_geometry.md:359`; "Operator-valued updates" bullet at l.457. References to `def-gksl-generator` in `08_multiagent/*` and `07_cognition/01_supervised_topo.md:362` use only the GKSL form, not the assimilation identification, and are unaffected.
- Fix guidance:
  1. Restrict the GKSL generator to the prediction/decoherence step (unconditional evolution between observations). Rewrite l.285 as "irreversible disturbance / decoherence" and l.298 to say that boundary information enters through a separate, nonlinear instrument step $\varrho\mapsto M_x\varrho M_x^\dagger/\mathrm{Tr}(M_x\varrho M_x^\dagger)$.
  2. Redefine the defect on the predicted operator: $\mathcal L_{\text{MEC}}:=\|(\tilde\varrho_{t+1}-\varrho_t)/\Delta t-\mathcal L_{\text{GKSL}}(\varrho_t)\|_F^2$ where $\tilde\varrho_{t+1}$ is the pre-assimilation, pre-projection operator; mirror the change in `02_sieve/01_diagnostics.md:120` and `06_fields/03_info_bound.md:663`.
  3. Alternatively, if a realized observation is to be included, replace the generator by a stochastic master equation and state that only the outcome-averaged map is CPTP-linear.
- Required new assumptions/permits: a definition of the instrument (measurement operators $M_x$ with $\sum_x M_x^\dagger M_x=I$) if option 3 is taken.
- Validation plan: implement the classical diagonal case; check that with the redefined defect a Bayes filter whose prediction kernel is $\exp(\Delta t\,W)$ gives $\mathcal L_{\text{MEC}}=O(\Delta t^2)$, whereas the current definition gives an $O(1)$ residual.

### [E-003] Unstated hypotheses for the GKSL completeness claim and the relation between $d$ and $|\mathcal K|$ (was F-010)
- Location: `def-belief-operator` l.260; `def-gksl-generator` l.271; prose l.300
- Severity: Note
- Type: Proof gap / omission
- Criterion: External
- Origin: this chapter
- Claim (verbatim): l.271 "A continuous-time, Markovian, completely-positive trace-preserving (CPTP) evolution has a generator of the ... (GKSL) form"; l.300 "any Markovian, completely-positive, trace-preserving (CPTP) evolution can be written this way"; l.260 "Diagonal $\varrho_t$ reduces to a classical probability vector".
- Why this matters: The GKSL representation theorem is for norm-continuous, time-homogeneous CPTP semigroups in finite dimension; "Markovian" alone admits time-dependent generators. Since $d<\infty$ is given, adding "time-homogeneous (semigroup)" suffices. Separately, $d$ is never related to $|\mathcal K|$, yet l.260 presumes the diagonal basis is the macro basis.
- Impact on downstream results: none.
- Fix guidance:
  1. At l.271 write "A time-homogeneous, norm-continuous CPTP semigroup on $\mathbb C^{d\times d}$ has a generator of the GKSL form".
  2. In `def-belief-operator` state $d=|\mathcal K|$ (or describe how the feature basis embeds $\mathcal K$).
- Required new assumptions/permits: none.
- Validation plan: textual.

### [E-004] Generator written in computation time $s$, defect evaluated per interaction step $\Delta t$ (was F-005)
- Location: `def-gksl-generator` l.274 vs. Master-Equation Consistency Defect l.343
- Severity: Minor
- Type: Notation conflict (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): l.274 "$\frac{d\varrho}{ds} = -i[H,\varrho]+\dots$"; l.343-346 "$\left\|\frac{\varrho_{t+1}-\varrho_t}{\Delta t} - \mathcal{L}_{\text{GKSL}}(\varrho_t)\right\|_F^2$".
- Upstream anchor: `docs/CLAUDE.md:218` "Time indices: $t$ (interaction), $s$ (computation), $\tau$ (scale)".
- Why this is an error: The rates $\gamma_j$ and $H$ carry units of $1/s$ while the finite difference is per interaction step. Nothing in the chapter identifies the two clocks or states a conversion, so the two terms inside the norm are in different units.
- Impact on downstream results: Node 22 formula.
- Fix guidance:
  1. Write the generator in $t$, or state $\Delta s=\Delta t$ (or the conversion factor) next to the defect.
  2. Give units for $\gamma_j$ (step$^{-1}$).
- Required new assumptions/permits: none.
- Validation plan: dimensional check of the Node 22 formula.

### [E-005] "The commutator term vanishes for diagonal states" is false unless $H$ is diagonal in the same basis (was F-003)
- Location: `def-gksl-generator`, Note (WFR Correspondence), l.289
- Severity: Moderate
- Type: Computational error (secondary: Scope restriction)
- Criterion: Framework (verifiable by direct algebra on the chapter's own equation)
- Origin: this chapter; the same statement appears in `05_geometry/02_wfr_geometry.md:388,396` (`cor-gksl-classical-limit`), which comes later in the book
- Claim (verbatim): l.289 "The commutator term vanishes for diagonal states (no coherences to rotate)."
- Upstream anchor: `05_geometry/02_wfr_geometry.md:388` "The commutator term $-i[H, \varrho]$ vanishes identically for diagonal states."; `:396` "| $-i[H, \varrho]$ (Commutator) | **Vanishes** (no off-diagonal elements to rotate) |".
- Why this is an error: For $\varrho=\mathrm{diag}(p)$, $([H,\varrho])_{jk}=H_{jk}(p_k-p_j)$, which is non-zero whenever $H$ couples two levels with different populations. Recomputed: $H=\sigma_x$, $p=(0.9,0.1)$ gives $[H,\varrho]_{01}=-0.8$. The commutator vanishes on all diagonal states iff $H$ is diagonal in that basis. The classical reduction therefore needs the hypothesis "$H$ diagonal in the belief basis", stated in neither chapter.
- Impact on downstream results: `cor-gksl-classical-limit` in `05_geometry/02_wfr_geometry.md:378-391`; the classical-limit table there (l.396); the Physics-Isomorphism table here (l.315-322).
- Fix guidance:
  1. Replace the sentence by "If $H$ is diagonal in the belief basis, the commutator vanishes on diagonal states; otherwise it generates coherences and the diagonal subspace is not invariant."
  2. Add the same hypothesis to `cor-gksl-classical-limit` and its table upstream.
- Required new assumptions/permits: $[H,\Pi_{\text{diag}}]=0$ (or $H$ diagonal in the macro basis).
- Validation plan: symbolic check of $[H,\mathrm{diag}(p)]$ for a $2\times2$ example.

### [E-006] "Rigorously equivalent to a WFR gradient flow" drops the needed hypotheses (was F-004)
- Location: `def-gksl-generator`, Note (WFR Correspondence), l.289
- Severity: Moderate
- Type: Scope restriction (secondary: External dependency)
- Criterion: External
- Origin: this chapter (drops the detailed-balance hypothesis stated upstream) and `05_geometry/02_wfr_geometry.md` (diagonal-invariance gap, metric identification)
- Claim (verbatim): l.289 "In the **classical limit** (diagonal density matrix $\varrho = \mathrm{diag}(p)$), the GKSL generator reduces to a Markov jump process on the diagonal probabilities $p_k$. This classical master equation is **rigorously equivalent** to a gradient flow in the Wasserstein-Fisher-Rao metric".
- Upstream anchor: `05_geometry/02_wfr_geometry.md:369-374` (`thm-classical-master-equation-wfr`): "is the **gradient flow** of the relative entropy $H(p\|\pi)$ ... with respect to a discrete Wasserstein-type metric, where $\pi$ is the stationary distribution satisfying detailed balance"; `:381-389` (`cor-gksl-classical-limit`): "reduces to a classical master equation with rates $W_{jk}=\sum_\ell\gamma_\ell|\langle j|L_\ell|k\rangle|^2$".
- Why this is an error: (i) Diagonal invariance. For $\varrho=\mathrm{diag}(p)$, $(L\varrho L^\dagger)_{jk}=\sum_m L_{jm}\bar L_{km}p_m$ has off-diagonal entries unless each $L_\ell$ has at most one non-zero entry per column. Recomputed: $L=\tfrac1{\sqrt2}(|0\rangle+|1\rangle)\langle0|$, $\varrho=|0\rangle\langle0|$ gives $(L\varrho L^\dagger)_{01}=1/2$. The population equation with rates $W_{jk}$ is exact only at the instant the state is diagonal; coherences are generated and feed back, so the reduction to a closed Markov jump process holds only for jump operators $|j\rangle\langle k|$ (secular / Pauli master equation), not for the learned low-rank $L_j$ of l.378. (ii) Detailed balance. The cited Maas (2011) / Mielke (2011) theorem requires a reversible chain; upstream states this hypothesis, this chapter omits it and calls the equivalence "rigorous" for a generic GKSL-induced $W$, which need not be reversible. (iii) The metric in the cited theorem is a discrete transport metric on a graph; `def-the-wfr-action` (`02_wfr_geometry.md:106`) is a transport-plus-reaction metric on $\mathcal M^+(\mathcal Z)$. The identification is asserted, not derived in the book.
- Impact on downstream results: "GKSL embedding" bullet `05_geometry/02_wfr_geometry.md:359`; "quantum-like belief decoherence" bullet l.457; `07_cognition/01_supervised_topo.md:362` (base transition rate from the GKSL master equation).
- Fix guidance:
  1. Restate the note: "If $H$ is diagonal and the $L_\ell$ are jump operators $|j\rangle\langle k|$, diagonal states are invariant and GKSL reduces to the master equation with rates $W_{jk}$; if in addition $W$ satisfies detailed balance with respect to its stationary law, the dynamics is the gradient flow of relative entropy in the discrete transport metric of {cite}`maas2011gradient`."
  2. Drop "rigorously equivalent to a gradient flow in the WFR metric", or add a lemma relating the Maas metric on $\mathcal K$ to the reaction part of `def-the-wfr-action`.
- Required new assumptions/permits: jump-operator form of $L_\ell$; detailed balance $W_{jk}\pi_k=W_{kj}\pi_j$.
- Validation plan: numerically evolve a $2\times2$ GKSL equation with the non-jump $L$ above from a diagonal state and observe the growth of $|\varrho_{01}|$; check that with jump operators it stays zero.

### [E-007] NEPCheck penalizes model-driven prediction and per-sample Bayes updates; index mismatch (was F-007)
- Location: Update vs Evidence (NEPCheck), l.389, 398-413
- Severity: Moderate
- Type: Conceptual (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter (interpretive claims) and `02_sieve/01_diagnostics.md:121` (identical formula; also `06_fields/03_info_bound.md:664`)
- Claim (verbatim): l.404-406 "$\mathcal{L}_{\text{NEP}} := \mathrm{ReLU}\!\left(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t)\right)^2$"; l.410 "The KL-divergence from old belief to new belief is how much your mind changed. The mutual information $I(X_t;K_t)$ is how much evidence you received. If you changed your mind more than your evidence justified, that's a problem."; l.412 "it detects ungrounded internal updating relative to measured boundary coupling (Node 13)".
- Upstream anchor: `02_sieve/01_diagnostics.md:121` "| **23** | **NEPCheck** | Belief / Boundary | Update vs Evidence | Internal update supported by boundary info? | $\mathrm{ReLU}(D_{\mathrm{KL}}(p_{t+1}\Vert p_t)-I(X_t;K_t))^2$ |"; prediction step l.94, update l.111-113 (uses $x_{t+1}$).
- Why this is an error: (i) $D_{\mathrm{KL}}(p_{t+1}\Vert p_t)$ spans both the prediction step and the Bayes step. Recomputed: with $\bar P$ = swap on two symbols, $p_t=(0.9,0.1)$ and a flat likelihood, $p_{t+1}=(0.1,0.9)$ and $D_{\mathrm{KL}}=0.8\ln 9=1.758$ nat while $I(X_t;K_t)=0$; the check fires on a correct, model-consistent update with no evidence. If $\bar P$ moves mass onto a symbol with $p_t(k)=0$, the KL is $+\infty$. (ii) Even the Bayes step alone satisfies only $\mathbb E_x[D_{\mathrm{KL}}(p(\cdot\mid x)\Vert\tilde p)]=I$. Recomputed: $\tilde p=(0.8,0.2)$, $O(x\mid k)=\begin{pmatrix}0.9&0.1\\0.3&0.7\end{pmatrix}$ gives $I=0.145$ nat, but the realized KL for $x=1$ (probability $0.22$) is $0.450$ nat. Exact Bayes conditioning is penalized on a positive-probability set. (iii) The update at $t{+}1$ conditions on $x_{t+1}$, but the budget is $I(X_t;K_t)$ (wrong index) and is a marginal MI rather than the information the observation carries about $\tilde p_{t+1}$. The hedge "conservative audit metric" at l.412 does not rescue the claim at l.410.
- Impact on downstream results: Node 23 in the Sieve table; "Constraint enforcement" bullet l.456; `01_foundations/02_control_loop.md:1241` (NEPCheck as an update-unreliability proxy).
- Fix guidance:
  1. Audit the assimilation step only: $D_{\mathrm{KL}}(p_{t+1}\Vert\tilde p_{t+1})$ (the realized information gain), and state explicitly that the prediction-step change is exempt.
  2. Compare against an evidence quantity with matching index and in expectation, e.g. a running average so that the audited inequality is $\mathbb E[D_{\mathrm{KL}}(p_{t+1}\Vert\tilde p_{t+1})]\le I(X_{t+1};K_{t+1}\mid\text{past})$; or use the per-sample bound $D_{\mathrm{KL}}(p_{t+1}\Vert\tilde p_{t+1})\le\max_k\log L_{t+1}(k)-\log\langle L_{t+1},\tilde p_{t+1}\rangle$.
  3. Mirror the change in `02_sieve/01_diagnostics.md:121` and `06_fields/03_info_bound.md:664`.
- Required new assumptions/permits: none beyond the existing filtering model.
- Validation plan: simulate an HMM with the exact Bayes filter and confirm the redefined check has zero false-positive rate in expectation, while an injected ungrounded update (e.g. tempering the likelihood by an exponent $>1$) is detected.

### [E-008] QSLCheck described as a generalization of ZenoCheck, which constrains the policy, not the state (was F-008)
- Location: Metric Speed Limit (QSLCheck), l.427
- Severity: Minor
- Type: Citation / reference error (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): l.427 "This is a geometry-consistent generalization of KL-per-update constraints (ZenoCheck)."
- Upstream anchor: `02_sieve/01_diagnostics.md:95` "| **2** | **ZenoCheck** | **Policy** | Action Frequency Limit | Switching policies too fast? | $D_{\mathrm{KL}}(\pi_t\Vert\pi_{t-1})$ |"; `:122` "| **24** | **QSLCheck** | All | Update Speed Limit | Step too large in $d_G$? | $\mathrm{ReLU}(d_G(z_{t+1},z_t)-v_{\max})^2$ |".
- Why this is an error: ZenoCheck bounds the KL step of the policy $\pi_t$; QSLCheck bounds the metric step of the latent state $z_t$. Neither reduces to the other under any choice of metric. The KL-per-update analogue in belief space is Node 23's $D_{\mathrm{KL}}(p_{t+1}\Vert p_t)$ or a Fisher-Rao step on $p_t$.
- Impact on downstream results: none.
- Fix guidance:
  1. Replace with "This is the state-space counterpart of KL-per-update constraints (cf. ZenoCheck on the policy, NEPCheck on the belief)."
- Required new assumptions/permits: none.
- Validation plan: textual.

### [E-009] "Standard POMDPs have unbounded continuous beliefs" is false for the cited finite-state setting (was F-009)
- Location: Connection to RL #19, l.443, 455
- Severity: Minor
- Type: Miswording (secondary: Conceptual)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): l.443 "Remove the Sieve projections ... Use continuous beliefs without discrete macro-register."; l.455 "**Discrete auditable symbols**: $H(K)\le\log|\mathcal{K}|$ provides hard capacity bound; standard POMDPs have unbounded continuous beliefs".
- Upstream anchor: n/a (external reference `kaelbling1998planning`, finite $\mathcal S$); the chapter's own special case l.448 sums over a discrete $s$.
- Why this is an error: The cited POMDP has a finite state set; its belief $b\in\Delta^{|\mathcal S|-1}$ is compact and satisfies $H(b)\le\log|\mathcal S|$, exactly as $p_t\in\Delta^{|\mathcal K|-1}$ does here (l.83). The belief simplex is continuous in both settings, so the stated distinction does not exist, and the "degenerate limit" at l.443 contradicts the discrete-sum formula two lines later. The genuine difference is the Sieve projection, already listed in the preceding bullet.
- Impact on downstream results: none.
- Fix guidance:
  1. Delete "standard POMDPs have unbounded continuous beliefs" and the sentence "Use continuous beliefs without discrete macro-register."
  2. Either keep the capacity bound as shared with finite POMDPs or contrast explicitly with continuous-state POMDPs.
- Required new assumptions/permits: none.
- Validation plan: textual.

## Scope restrictions and clarifications
- The classical predict/update/project template (l.86-120), the worked example (l.128-171; recomputed: $p_{t+1}=(0.1316,0.8421,0.0263)$, $p'_{t+1}=(0.1351,0.8649,0)$), and the projection/reweighting definitions (l.185-217) are correct as stated.
- The operator-valued section is labelled optional and the TLDR advises treating the GKSL analogy as intuition; the findings E-002 to E-006 concern the concrete formulas and diagnostics that the section nevertheless defines and the Sieve table imports.
- The correspondence row "Hamiltonian $H$ | Effective potential $\Phi_{\text{eff}}$" (l.319) is consistent with the book's Hamiltonian $H(z,p)=\tfrac12G^{ij}p_ip_j+\Phi_{\text{eff}}(z)$ (`05_geometry/04_equations_motion.md:254-256`); citing that Hamiltonian in the row would make the reading unambiguous.

## Proposed edits (optional)
- Split the GKSL section into "unconditional evolution (GKSL, prediction/decoherence)" and "conditional assimilation (instrument step, nonlinear)", and let Node 22 audit only the former.
- Rewrite the WFR-correspondence note as a conditional statement with explicit hypotheses (diagonal $H$, jump-type $L_\ell$, detailed balance).
- Redefine NEPCheck on the assimilation step with matching time indices.

## Open questions
- Is Node 22 intended to audit the learned operator model (prediction) or the whole step including assimilation? The answer determines which of the two fixes in E-002 applies.
- Should NEPCheck be a per-step or a running-average diagnostic? Only the latter is sound as an information-budget inequality.

## Rejected candidate findings
- F-006 (Hamiltonian $H$ mapped to $\Phi_{\text{eff}}$ in the correspondence table): rejected. The book defines the conservative part of the latent dynamics by the Hamiltonian $H(z,p)=\tfrac12G^{ij}(z)p_ip_j+\Phi_{\text{eff}}(z)$ (`05_geometry/04_equations_motion.md:252-256`), so $\Phi_{\text{eff}}$ is the potential term of a conservative generator and the row is consistent with the framework; the overdamped gradient-flow reading is one limit, not the definition.
