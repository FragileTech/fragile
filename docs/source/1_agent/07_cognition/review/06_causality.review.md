# Mathematical Review: docs/source/1_agent/07_cognition/06_causality.md

## Metadata
- Reviewed file: docs/source/1_agent/07_cognition/06_causality.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (490 lines), including the appendix proofs E.5, E.6 and E.11 that the chapter delegates to
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/01_definitions.md`, sec-the-environment-is-an-input-output-law (environment as partially observed input-output law $P_\partial$)
  - `01_foundations/02_control_loop.md:141, 1063` (critic $V$ is a cost-to-go that decreases along controlled trajectories); `:1086-1108` (def-causal-enclosure-condition, def-closure-defect)
  - `02_sieve/01_diagnostics.md:206` (World Model volatility scale $\gamma$, dimensionless scalar)
  - `05_geometry/04_equations_motion.md:227` (def-bulk-drift-continuous-flow), `:396-425` (sec-the-unified-effective-potential), `:885-925` (sec-the-overdamped-limit, thm-overdamped-limit, $\mathcal{M}_\gamma$)
  - `06_fields/01_boundary_interface.md:90-125` (Dirichlet sensors, Neumann motors)
  - `06_fields/02_reward_field.md:196, 216, 357` ($A := \delta\Psi + \eta$, $\mathcal{F} = dA$, $\nabla_A V := \nabla V - A$)
  - `06_fields/03_info_bound.md:555-562, 694-698` (symbol table and node table entries for Chapter 32)
  - `07_cognition/03_memory_retrieval.md:961-987`, `09_economics/01_pomw.md:309, 368, 1306`, `10_appendices/04_faq.md:483, 612` (downstream users of $\Delta_{\text{causal}}$ and the curiosity force)
  - `10_appendices/05_proofs.md` E.5 (`:220-278`), E.6 (`:282-325`), E.9 (`:888`), E.11 (`:1012-1055`)
  - `10_appendices/02_parameters.md` (no entry for the Node 53 threshold or $\beta_{\text{exp}}$)

## Executive summary
- Critical: 0
- Major: 2
- Moderate: 6
- Minor: 7
- Notes: 0
- Primary themes:
  1. The two central objects of the chapter do not fit together. The interventional operator is defined by truncated factorization with the mechanism preserved, $P(z'|z,do(a)) := P(z'|z,a)$, and the Causal Deficit is then defined as the KL divergence between this interventional conditional and the observational conditional of the same kernel. Under the chapter's own definitions the deficit is identically zero, so the Interventional Gap theorem, Node 53, and the downstream Governor and PoUW checks have no content until a hidden confounder (or a learned-versus-true distinction) is made explicit.
  2. The Epistemic Curiosity Filter asserts a proportionality between the gradient of the expected information gain and the gradient of a varentropy. The statement uses the posterior varentropy over $\theta_W$, the appendix proof concludes with the predictive varentropy over $z'$, and the proportionality is false for both (explicit counterexamples in both directions).
  3. The variational proof of the Augmented Drift Law contains a sign flip in the curiosity term and an overdamped step that does not follow from a conservative action; terms $\nabla_A V$ and $\beta_{\text{curl}}\mathcal{F}$ are introduced without a generating term in the functional, and the cited section does not contain the overdamped limit. The theorem statement is the intended one; the defects are in the derivation and are shared with Appendix E.5.
  4. The Interventional Closure theorem is a definition dressed as an iff, and its proof sketch shows the interventional condition is inherited from causal enclosure, which would make the advertised ontology-expansion trigger impossible; the missing ingredient is a positivity hypothesis on the observational policy.
  5. Several notation and algorithm mismatches: $\Psi_{\text{causal}}$ on $\mathcal{Z}\times\mathcal{A}$ used as a field on $\mathcal{Z}$; $\gamma$ used as a spatial field; $\nabla_G V$ versus $\nabla_A V$; Algorithm 32.5.1 selects by $\Delta_{\text{causal}}$ rather than by the quantity the theory maximizes; threshold symbol and section number for Node 53 disagree with the master tables.

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | 32.1 Definition def-the-interventional-surgery, lines 85-106 | Moderate | Conceptual (Definition mismatch) | Framework | this chapter | Geometric gloss (Dirichlet clamp on $z_t$ removed, $z_{t+1}$ driven purely by $u_\pi$) describes a state intervention; the formal definition intervenes on $a$ only |
| E-002 | 32.1 Lemma lem-the-interventional-singularity, lines 107-116 | Moderate | Proof gap / omission (Conceptual) | Framework | this chapter | Lemma has no checkable content: undefined causal manifold and curvature; delta source in $z$ does not represent an intervention on $a$ |
| E-003 | 32.2 Definition def-causal-information-potential, lines 151-164; Theorem thm-augmented-drift-law, lines 289, 292 | Minor | Definition mismatch (Notation conflict) | Framework | this chapter | $\Psi_{\text{causal}}$ defined on $\mathcal{Z}\times\mathcal{A}$ but used as $\Psi_{\text{causal}}(z)$; $\gamma$ recalled but never used |
| E-004 | 32.2 Connection to RL #16, line 194 | Minor | Citation / reference error | Framework | this chapter | Placeholder "Definition **Def: Interventional Operator**" instead of a `{prf:ref}` |
| E-005 | 32.1 line 95 and 32.2 Theorem thm-the-interventional-gap, lines 207-226; Table 32.6.1 line 460 | Major | Conceptual (Definition mismatch) | Framework | this chapter | Under the chapter's definition of $do$, $\Delta_{\text{causal}} \equiv 0$; the confounding interpretation is unreachable |
| E-006 | 32.2 Corollary cor-epistemic-curiosity-filter, lines 227-250; Appendix E.11 | Major | Invalid inference (Definition mismatch) | Framework | this chapter and upstream `10_appendices/05_proofs.md` E.11 | Asserted proportionality between $\nabla\Psi_{\text{causal}}$ and a varentropy gradient is false; statement and proof use different random variables |
| E-007 | 32.3 prose, line 270 | Minor | Miswording | Framework | this chapter | "The utility force points uphill in the value landscape" contradicts $-G^{-1}\nabla V$ with $V$ a cost-to-go |
| E-008 | 32.3 Theorem thm-augmented-drift-law, statement line 285 | Minor | Notation conflict | Framework | this chapter | Statement uses undefined $\nabla_G V$; proof and Appendix E.5 use $\nabla_A V$; statement omits $\mathcal{M}_{\text{curl}}$ |
| E-009 | 32.3 Theorem thm-augmented-drift-law, proof lines 292-315 | Moderate | Computational error (Invalid inference) | Framework | this chapter and upstream `10_appendices/05_proofs.md` E.5 | Sign of the curiosity term flips between line 307 and line 313; the Lagrangian as written repels the agent from informative regions |
| E-010 | 32.3 Theorem thm-augmented-drift-law, proof lines 298-318 | Moderate | Invalid inference (Proof gap; Citation error) | Framework | this chapter and upstream `10_appendices/05_proofs.md` E.5 | $\nabla_A V$ and $\beta_{\text{curl}}\mathcal{F}$ appear without a generating term; overdamped limit of a conservative equation is invalid; wrong section cited |
| E-011 | 32.3 Corollary cor-scientific-method-as-geodesic, lines 323-332 | Minor | Invalid inference | Framework | this chapter | Curl term kept after assuming $A = 0$, although $\mathcal{F} = dA$ |
| E-012 | 32.4 Theorem thm-interventional-closure, lines 363-380; Appendix E.6 | Moderate | Invalid inference (Proof gap / omission) | Framework | this chapter and upstream `10_appendices/05_proofs.md` E.6 | Proof makes interventional closure automatic given enclosure, so the violation case is empty; needs an unstated positivity hypothesis |
| E-013 | 32.4 Example: Simpson's Paradox, line 387 | Minor | Miswording | External | this chapter | Overall comparison stated backwards (A 78 % vs B 83 %, attributed to A) |
| E-014 | 32.5 Algorithm 32.5.1, lines 413-420; line 246 | Moderate | Algorithm mismatch (Definition mismatch) | Framework | this chapter | Algorithm selects by $\Delta_{\text{causal}}$ (needs interventional data), corollary says varentropy, theory says $\Psi_{\text{causal}}$; $\gamma(z)$ undefined |
| E-015 | 32.5 Node 53, lines 422-441 | Minor | Notation conflict (Citation / reference error) | Framework | this chapter and `06_fields/03_info_bound.md:696` | Threshold symbol ($\Delta_{\max}$ vs $\delta_{\text{causal}}$), missing parameter entry, wrong section number in master table |

## Detailed findings

### [E-001] Geometric gloss of the surgery contradicts its truncated-factorization formula (was F-001)
- Location: Section 32.1, Definition def-the-interventional-surgery, lines 85-106; abstract lines 36-37; Table 32.6.1 line 460
- Severity: Moderate
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 90: "$\mathfrak{I}$ transforms the symplectic interface ... from a **Coupled Dirichlet state** (where $z_t$ is clamped by the observation $x_t$) to a **Forced Neumann state** (where $z_{t+1}$ is driven purely by the agent's internal motor impulse $u_\pi$)." Lines 95-98: "$P(z' | z, do(a)) := P(z' | z, a)$, where the structural mechanism $P(z' | z, a)$ is preserved but $a$ is no longer a function of $z$."
- Upstream anchor: `06_fields/01_boundary_interface.md:90-100`: "The sensory input stream $\phi(x)$ imposes a **Dirichlet** (position-clamping) condition on the belief density: $\rho_{\partial}^{\text{sense}}(q, t) = \delta(q - q_{\text{obs}}(t))$"; `:113-125`: "The motor output stream $A(x)$ imposes a **Neumann** (flux-clamping) condition: $\nabla \rho \cdot \mathbf{n}|_{\partial\mathcal{Z}_{\text{motor}}} = j_{\text{motor}}(p, t)$".
- Why this is an error: The formal definition cuts only the incoming edges to the action variable. The state $z_t$ remains an argument of the mechanism $P(z'|z,a)$, so the sensory Dirichlet clamp on $z_t$ is untouched and $z_{t+1}$ depends on $z_t$ exactly as before; it is not "driven purely by $u_\pi$". The gloss describes a hard intervention on the state (or the dreaming-mode reflective boundaries of `05_geometry/04_equations_motion.md:217`), which is a different operator from the one defined. Table 32.6.1 compounds the problem by listing $P(z'\mid z,a)$ as the observation operator, the same symbol the definition uses for the interventional kernel.
- Impact on downstream results: Lemma lem-the-interventional-singularity (E-002), the abstract, and the summary table rest on the gloss; E-005 is the formal consequence of using one symbol for both kernels.
- Fix guidance (step-by-step):
  1. Keep the formal definition and rewrite the gloss: "$do(a)$ replaces the policy-induced Neumann flux $j_{\text{motor}} = D_A(u_\pi(z))$ by an exogenous flux $j_{\text{motor}} = D_A(u)$ independent of $z$; the sensory Dirichlet condition is unchanged."
  2. If a state intervention is also intended, define $do(z = z_0)$ separately with its own truncated factorization $P(z'|do(z_0),a) = P(z'|z_0,a)$ and say which operator $\mathfrak{I}$ denotes.
  3. Rename the observation-row operator in Table 32.6.1 (for example $P_{\text{obs}}(z'\mid z,a)$) so that the two kernels carry distinct symbols.
- Required new assumptions/permits: none.
- Validation plan: Re-read lines 36-37, 90, 112 and 460 after the edit and check that every sentence describing what $do(a)$ removes refers to the edge $z\to a$ only.

### [E-002] Lemma "Interventional Singularity" has no derivable content (was F-002)
- Location: Section 32.1, Lemma lem-the-interventional-singularity, lines 107-116
- Severity: Moderate
- Type: Proof gap / omission (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 112: "Under passive observation, the agent samples from the equilibrium distribution $P_{\text{eq}}(z' | z, a)$ determined by the environment's Dirichlet boundary $\partial\mathcal{Z}$. ... In PDE terms, this corresponds to introducing a Dirac delta source $\delta(z - z_0)$ at the intervention point, creating a Green's function response that propagates through the causal graph. The 'singularity' is geometric: the intervention point has infinite curvature in the causal manifold because all causal arrows pointing into it are severed. $\square$"
- Upstream anchor: none applicable; the only imported objects are the operator of Definition def-the-interventional-surgery (action-variable surgery) and $P_\partial$ from `01_foundations/01_definitions.md`, sec-the-environment-is-an-input-output-law.
- Why this is an error: No causal manifold, metric on it, or curvature is defined anywhere in Volume 1, so "infinite curvature" is not a checkable statement. The intervention of the definition acts on $a$, not on a state $z_0$, so a source $\delta(z - z_0)$ in $\mathcal{Z}$ does not represent it (a hard intervention on $a$ is a delta $\delta(a - a_0)$ replacing the policy kernel, a Dirichlet-type clamp, not a source term). "Equilibrium distribution determined by the environment's Dirichlet boundary" conflates a transition kernel with a boundary condition of a PDE that is never written. The $\square$ closes a paragraph of metaphors.
- Impact on downstream results: rhetorical only (Summary item 1 "creating a singularity in the causal graph"). No later formula depends on it.
- Fix guidance (step-by-step):
  1. Downgrade to a Remark, or
  2. State a checkable lemma: "Under $do(a = a_0)$ the joint $P(z, a, z')$ is replaced by $P_{\text{pre}}(z)\,\delta_{a_0}(a)\,P(z'|z,a_0)$, which is singular (not absolutely continuous) with respect to the observational joint whenever $\pi(a_0|z)$ is a density," and prove it in two lines from the truncated factorization.
- Required new assumptions/permits: none.
- Validation plan: Each sentence in the proof should reference a defined object; delete any that does not.

### [E-003] $\Psi_{\text{causal}}$ defined on $\mathcal{Z}\times\mathcal{A}$ but used as a field on $\mathcal{Z}$; $\gamma$ recalled but unused (was F-003)
- Location: Section 32.2, Definition def-causal-information-potential, lines 151-164; Theorem thm-augmented-drift-law, lines 289, 292
- Severity: Minor
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 154: "$\Psi_{\text{causal}}: \mathcal{Z} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$"; line 289: "$\mathbf{f}_{\text{exp}} := G^{-1} \nabla_z \Psi_{\text{causal}}$"; line 292: "$\beta_{\text{exp}} \Psi_{\text{causal}}(z)$"; line 154: "Recall the World Model scaling coefficient $\gamma$".
- Upstream anchor: `02_sieve/01_diagnostics.md:206`: "| **World Model** | **Volatility scale** | $\gamma$ | dimensionless | **Dynamics non-stationarity / rollout volatility.** |"; `06_fields/03_info_bound.md:559`: "$\mathbf{f}_{\text{exp}}$ | Curiosity force | $G^{-1}\nabla\Psi_{\text{causal}}$".
- Why this is an error: The action functional and the force require a scalar field on $\mathcal{Z}$; the defined object depends on $a$ and no reduction ($\mathbb{E}_{a\sim\pi(\cdot|z)}$, $\max_a$, or evaluation at the executed action) is specified. Separately, $\gamma$ is invoked in the definition and plays no role in the displayed formula; upstream it is a single dimensionless training diagnostic, not a function on $\mathcal{Z}$.
- Impact on downstream results: thm-augmented-drift-law, cor-scientific-method-as-geodesic, Algorithm 32.5.1, `06_fields/03_info_bound.md:559`.
- Fix guidance (step-by-step):
  1. Add after line 157: "$\Psi_{\text{causal}}(z) := \mathbb{E}_{a\sim\pi(\cdot|z)}[\Psi_{\text{causal}}(z,a)]$" (or $\max_a$) and use this in the drift law.
  2. Delete the sentence recalling $\gamma$, or state explicitly how $\gamma$ enters (for example as a cheap proxy estimator for $\Psi_{\text{causal}}$).
- Required new assumptions/permits: none.
- Validation plan: Check that every occurrence of $\Psi_{\text{causal}}$ in Sections 32.3-32.5 has the declared arity.

### [E-004] Unresolved placeholder cross-reference (was F-004)
- Location: Section 32.2, Connection to RL #16, line 194
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "The $do(\cdot)$ operator distinguishes observations from interventions (Definition **Def: Interventional Operator**)"
- Upstream anchor: the intended target is `def-the-interventional-surgery` at line 86 of this chapter.
- Why this is an error: Bold placeholder text instead of a resolvable reference; the build does not check it, so the xref scanner cannot catch it.
- Impact on downstream results: none mathematical.
- Fix guidance: Replace with "(Definition {prf:ref}`def-the-interventional-surgery`)".
- Required new assumptions/permits: none.
- Validation plan: Build the docs and confirm the link resolves.

### [E-005] Under the chapter's definition of $do$, the Causal Deficit is identically zero (was F-005)
- Location: Section 32.1 line 95 versus Section 32.2, Theorem thm-the-interventional-gap, lines 207-226; Table 32.6.1 line 460
- Severity: Major
- Type: Conceptual (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 95-98: "$P(z' | z, do(a)) := P(z' | z, a)$, where the structural mechanism $P(z' | z, a)$ is preserved but $a$ is no longer a function of $z$." Lines 210-213: "Let $P_{\text{obs}}(z' | z, a)$ be the conditional density obtained via passive observation, and $P_{\text{int}}(z' | do(z, a))$ be the density under intervention. ... $\Delta_{\text{causal}}(z, a) := D_{\text{KL}}(P_{\text{int}}(z' | do(z, a)) \| P_{\text{obs}}(z' | z, a))$." Line 216: "If $\Delta_{\text{causal}} > 0$, the agent has mistaken a correlation for a causal link (confounding)".
- Upstream anchor: both objects are defined in this chapter. Downstream users: `07_cognition/03_memory_retrieval.md:965`: "Let $\Delta_{\text{causal}} = D_{\text{KL}}(P_{\text{int}} \| P_{\text{obs}})$ be the Interventional Gap (Node 53)"; `09_economics/01_pomw.md:309, 368, 1306`; `10_appendices/04_faq.md:483`.
- Why this is an error: The chapter has one transition kernel symbol, $P(z_{t+1}|z_t,a_t)$ (line 88). The definition identifies the interventional conditional with it and calls it the preserved mechanism. The observational conditional of $z'$ given $(z,a)$ is that same kernel in any model in which $z$ carries all common causes of $a$ and $z'$; the chapter introduces no other variable. Substituting, $\Delta_{\text{causal}}(z,a) = D_{\text{KL}}(P(z'|z,a)\,\|\,P(z'|z,a)) = 0$ for every $(z,a)$ and $\text{Vol}_{\text{ignorant}} = 0$ (line 221). Writing $do(z,a)$ rather than $do(a)$ in the theorem does not change this. The deficit can be positive only when a variable $u$ outside $z$ influences both $a$ and $z'$, in which case $P_{\text{obs}}(z'|z,a) = \int P(z'|z,a,u)P(u|z,a)\,du$ and $P_{\text{int}}(z'|z,do(a)) = \int P(z'|z,a,u)P(u|z)\,du$ differ, and the phrase "the structural mechanism $P(z'|z,a)$" is no longer well posed because $P(z'|z,a)$ is not invariant under intervention. The framework has a natural candidate for $u$ (the environment is a partially observed input-output law, `01_definitions.md`, sec-the-environment-is-an-input-output-law, so the environment's hidden state is not part of $z$), but the formal statements never use it; the Feynman prose at line 125 mentions a hidden $U$ informally only. The alternative reading, that $P_{\text{obs}}$ is the learned model $\bar P$ and $P_{\text{int}}$ the empirical interventional kernel, is not what line 210 says.
- Impact on downstream results: Node 53 (lines 422-441), the Governor coupling in `07_cognition/03_memory_retrieval.md:961-987`, the PoUW causal checks in `09_economics/01_pomw.md:309, 368, 1306`, Algorithm 32.5.1 step 2, Summary items 1-2. All inherit the symbol without a non-degenerate definition.
- Fix guidance (step-by-step):
  1. Option (a), make the confounder explicit: let the environment law be $P_\partial(z'|z,a,u)$ with an unobserved $u$ (hidden environment state and/or texture $z_{\text{tex}}$); define $P_{\text{obs}}(z'|z,a) := \int P_\partial(z'|z,a,u)\,P(u|z,a)\,du$ and $P_{\text{int}}(z'|z,do(a)) := \int P_\partial(z'|z,a,u)\,P(u|z)\,du$.
  2. Restate Definition def-the-interventional-surgery so the truncated-factorization identity $P(z'|z,do(a)) = P_{\text{obs}}(z'|z,a)$ is a consequence of causal sufficiency of $z$ (absence of $u$), not a definition.
  3. Option (b), if the intended quantity is an estimation gap: define $\Delta_{\text{causal}}$ as the divergence between the learned observational model $\bar P_{\text{obs}}$ and the empirical interventional kernel, and state confounding as one possible source rather than the interpretation.
  4. Update Table 32.6.1 (line 460) and the downstream symbol table entry (`06_fields/03_info_bound.md:560`) to match.
- Required new assumptions/permits: for option (a), an explicit statement that $z$ need not be causally sufficient (a hidden $u$ may exist); for option (b), a definition of the interventional estimator.
- Framework-first proof sketch for the fix: with option (a), $\Delta_{\text{causal}} = 0$ iff $P(u|z,a) = P(u|z)$ almost everywhere on the support of $P_\partial(\cdot|z,a,u)$, i.e. iff $a \perp u \mid z$; this is exactly the no-confounding condition and recovers the stated interpretation.
- Validation plan: After the edit, construct a two-state toy with a binary hidden $u$ affecting both $\pi$ and the transition and check numerically that $\Delta_{\text{causal}} > 0$; then check that it vanishes when $\pi$ is made independent of $u$.

### [E-006] Epistemic Curiosity Filter: the proportionality is false, and statement and proof use different random variables (was F-006)
- Location: Section 32.2, Corollary cor-epistemic-curiosity-filter, lines 227-250; proof in Appendix E.11 (`10_appendices/05_proofs.md:1012-1055`)
- Severity: Major
- Type: Invalid inference (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter (statement) and upstream `10_appendices/05_proofs.md` E.11 (proof)
- Claim (verbatim): lines 232-235: "Let $V_H[P(\theta_W | z, a, z')]$ denote the Varentropy of the posterior over World Model parameters ... Then: $\nabla \Psi_{\text{causal}} \propto \nabla \mathbb{E}_{z'} [ V_H [P(\theta_W | z, a, z')] ]$." Line 243: "**High Entropy, Low Varentropy:** ... The gradient $\nabla \Psi \approx 0$." Line 246: "The Experimental Sieve (Algorithm 32.5.1) selects interventions $do(a)$ that maximize the **Varentropy of the expected outcome distribution**."
- Upstream anchor: `05_proofs.md:1016`: "**Statement:** $\nabla \Psi_{\text{causal}} \propto \nabla \mathbb{E}_{z'} [ V_H[P(\theta_W | z, a, z')] ]$."; `:1023`: "$\text{EIG}(z, a) = I(\theta; z' | z, a) = H(z' | z, a) - \mathbb{E}_{\theta} [ H(z' | z, a, \theta) ]$"; `:1049-1051`: "maximizing EIG is functionally equivalent to maximizing the **Varentropy of the expected outcome**, provided the aleatoric noise floor is constant: $\nabla \Psi_{\text{causal}} \propto \nabla \mathrm{Var}_{z' \sim p(z'|z,a)} [ -\ln p(z'|z,a) ]$". Step 4 appeals to E.9 (`05_proofs.md:888`), which concerns a bimodal policy "with a background of $N-2$ negligible modes".
- Why this is an error: (i) Mismatch: the corollary's right-hand side is the varentropy of the posterior over $\theta_W$; the appendix concludes with the varentropy of the predictive distribution over $z'$. These live on different spaces and neither is shown proportional to the other. (ii) False for the predictive varentropy (recomputed): take $\theta_1$ = uniform on $\{1,2\}$, $\theta_2$ = uniform on $\{3,4\}$, prior $1/2$ each. The predictive is uniform on four outcomes, $H(z'|z,a) = \ln 4 = 1.3863$, $\mathbb{E}_\theta H(z'|z,a,\theta) = \ln 2$, so $\text{EIG} = \ln 2 = 0.6931 > 0$, while the predictive surprisal is constant and the predictive varentropy is $0$. This is the chapter's own case 2 (two distinct hypotheses) with zero varentropy. Conversely a known biased coin $(0.9, 0.1)$ has predictive varentropy $0.5402 - 0.3251^2 = 0.4345\ \text{nat}^2 > 0$ and $\text{EIG} = 0$. So EIG and predictive varentropy are neither proportional nor co-monotone. (iii) False for the posterior varentropy: $\mathrm{Var}_\theta[-\ln p(\theta|z,a,z')]$ measures the dispersion of the posterior's surprisal, not the change from prior to posterior; a posterior identical to the prior (EIG $= 0$) with prior $(0.9, 0.1)$ has varentropy $0.4345$. (iv) The E.9 appeal fails: counterexample (ii) is an equal-weight mixture of two disjoint hypotheses with zero varentropy, while E.9's bound relies on negligible background modes. (v) The proviso "provided the aleatoric noise floor is constant" is not among the corollary's hypotheses and is not sufficient (counterexample (ii) has constant aleatoric entropy $\ln 2$ under both hypotheses). The Noisy-TV conclusion at line 243 is true, but for the reason in E.11 Step 2 ($p(z'|z,a,\theta)$ independent of $\theta$ gives EIG $= 0$), not because of varentropy.
- Impact on downstream results: line 241 (weight $\mathbf{f}_{\text{exp}}$ by varentropy), line 246 (Experimental Sieve criterion), the Noisy-TV prose (lines 252-260), and the varentropy diagnostics in `07_cognition/04_ontology.md:1261-1266` that rely on the same idea.
- Fix guidance (step-by-step):
  1. Replace the proportionality with what is provable from the definition: $\Psi_{\text{causal}}(z,a) = I(\theta_W; z' \mid z,a) = H(z'|z,a) - \mathbb{E}_\theta[H(z'|z,a,\theta)]$ (epistemic = total minus aleatoric).
  2. State the Noisy-TV property as: "$\Psi_{\text{causal}}(z,a) = 0$ whenever $p(z'|z,a,\theta)$ does not depend on $\theta$."
  3. If a varentropy-based proxy is wanted, present it as a heuristic estimator with its failure cases, or restrict to a stated family (for example mixtures of well-separated unimodal hypotheses with a low-probability background) and prove a monotone relation as a lemma.
  4. Rewrite line 246 to say the Sieve maximizes $\Psi_{\text{causal}}$ (see E-014).
  5. Rewrite E.11 to prove the corrected statement.
- Required new assumptions/permits: none for steps 1-2; a stated hypothesis family for step 3.
- Framework-first proof sketch for the fix: the identity in step 1 is the standard decomposition of mutual information already written at `05_proofs.md:1023`; step 2 follows because the two entropies then coincide.
- Validation plan: Run the two counterexamples above against whatever replaces the corollary; both must give the correct EIG ordering.

### [E-007] Prose sign of the utility force (was V-002, added by verifier)
- Location: Section 32.3, Feynman prose, line 270
- Severity: Minor
- Type: Miswording
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "both forces are 'gradients'---they point uphill in their respective landscapes. The utility force points uphill in the value landscape. The curiosity force points uphill in the causal information potential landscape."
- Upstream anchor: `01_foundations/02_control_loop.md:141`: "**Value Function:** assigns a scalar cost-to-go/value to points in $Z$, representing risk/undesirability"; `:1063`: "The critic $V$ is a value/cost-to-go function that should decrease along controlled trajectories."
- Why this is an error: The utility force in the theorem is $-G^{-1}\nabla V$ (line 285) and $V$ is a cost-to-go, so the force points downhill in $V$. Only the curiosity force is an ascent direction. The sentence inverts the convention the chapter itself uses two paragraphs later and adds to the sign confusion in E-009.
- Impact on downstream results: none formal; misleads the reader about the sign convention.
- Fix guidance: "The utility force points downhill in the cost-to-go landscape $V$; the curiosity force points uphill in $\Psi_{\text{causal}}$. One descends cost, the other climbs information."
- Required new assumptions/permits: none.
- Validation plan: Check the sentence against line 285 after E-009 is fixed.

### [E-008] Undefined $\nabla_G V$ in the theorem statement, inconsistent with the proof (was V-001, added by verifier)
- Location: Section 32.3, Theorem thm-augmented-drift-law, statement line 285
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 285: "$F_{\text{total}} = \underbrace{-G^{-1} \nabla_G V}_{\text{Utility Force}} + \underbrace{\beta_{\text{exp}} \mathbf{f}_{\text{exp}}}_{\text{Curiosity Force}}$"; lines 313-316: "$\dot{z} = \mathcal{M}_{\text{curl}}(-G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}})$ ... Here $\nabla_A V := \nabla V - A$".
- Upstream anchor: `10_appendices/05_proofs.md:222`: "**Statement:** $F_{\text{total}} = -G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}}$." A search of Volume 1 for `\nabla_G V` finds uses at `05_geometry/04_equations_motion.md:459, 471`, `07_cognition/01_supervised_topo.md:896`, `01_foundations/02_control_loop.md:1501`, but no definition.
- Why this is an error: The statement and its proof are not the same formula: the statement has $\nabla_G V$ and no $\mathcal{M}_{\text{curl}}$; the proof concludes with $\nabla_A V$ inside $\mathcal{M}_{\text{curl}}$; the appendix statement has $\nabla_A V$ without $\mathcal{M}_{\text{curl}}$. If $\nabla_G$ denotes the Riemannian gradient $G^{-1}\nabla$, then $G^{-1}\nabla_G V$ applies $G^{-1}$ twice.
- Impact on downstream results: `06_fields/03_info_bound.md:559-560` and `10_appendices/04_faq.md:612` quote the force; implementers may pick either form.
- Fix guidance: Write the statement as $F_{\text{total}} = \mathcal{M}_{\text{curl}}\big(-G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}}\big)$ with $\nabla_A V := \nabla V - A$ declared in the statement, and make E.5's statement identical.
- Required new assumptions/permits: none.
- Validation plan: Diff the three displayed forms (line 285, line 313, `05_proofs.md:222`) after the edit.

### [E-009] Sign flip in the curiosity term of the variational proof (was F-007)
- Location: Section 32.3, Theorem thm-augmented-drift-law, proof lines 292-315; also Corollary cor-scientific-method-as-geodesic line 329
- Severity: Moderate (the stage-1 reviewer proposed Major; downgraded because the theorem statement and all downstream users carry the intended sign, and the fix is one sign in the functional)
- Type: Computational error (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter and upstream `10_appendices/05_proofs.md` E.5 (identical error)
- Claim (verbatim): line 292: "$\mathcal{S}_{\text{total}} = \int_0^T [ \frac{1}{2}\|\dot{z}\|_G^2 - V(z) - \beta_{\text{exp}} \Psi_{\text{causal}}(z) ] dt$"; line 307: "$\ddot{z}^m + \Gamma^m_{ij} \dot{z}^i \dot{z}^j = -G^{mk} (\nabla_A V)_k - \beta_{\text{exp}} G^{mk} \partial_k \Psi_{\text{causal}} + \dots$"; line 313: "$\dot{z} = \mathcal{M}_{\text{curl}}(-G^{-1}\nabla_A V + \beta_{\text{exp}} G^{-1}\nabla\Psi_{\text{causal}})$".
- Upstream anchor: `05_proofs.md:224`: "Lagrangian $L = \frac{1}{2}\|\dot{z}\|_G^2 - (V + \beta_{\text{exp}}\Psi_{\text{causal}})$"; `:268`: "$\ddot{z}^m + \Gamma^m_{ij}\dot{z}^i\dot{z}^j = -G^{mk}\partial_k V - \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}}$"; `:274`: "$\dot{z}^m = -G^{mk}\partial_k V + \beta_{\text{exp}} G^{mk}\partial_k \Psi_{\text{causal}}$".
- Why this is an error: With $L = T - U$ and $U = V + \beta_{\text{exp}}\Psi_{\text{causal}}$, $\partial L/\partial z^k = \tfrac12\partial_k G_{ij}\dot z^i\dot z^j - \partial_k V - \beta_{\text{exp}}\partial_k\Psi_{\text{causal}}$, so the Euler-Lagrange force is $-\nabla V - \beta_{\text{exp}}\nabla\Psi_{\text{causal}}$. Lines 295, 301 and 307 carry this sign correctly. Line 313 then writes $+\beta_{\text{exp}}$ with no intervening step, and line 329 does the same. With the functional as written, the agent descends $\Psi_{\text{causal}}$ and is pushed away from high-EIG regions, the opposite of the statement (line 285) and of every interpretive sentence (lines 320, 330, 467). Since $V$ is a cost (descended) and $\Psi_{\text{causal}}$ is to be climbed, they cannot enter the potential with the same sign.
- Impact on downstream results: The proof of the chapter's main theorem and the proof of cor-scientific-method-as-geodesic. The statement used downstream (`10_appendices/04_faq.md:612`, `06_fields/03_info_bound.md:559-560`, `07_cognition/07_metabolic_transducer.md`) is the intended one and is unaffected once the proof is corrected.
- Fix guidance (step-by-step):
  1. Change the functional to $\mathcal{S}_{\text{total}} = \int_0^T [\tfrac12\|\dot z\|_G^2 - V(z) + \beta_{\text{exp}}\Psi_{\text{causal}}(z)]\,dt$ (curiosity as negative potential energy).
  2. Propagate $+\beta_{\text{exp}}\partial_k\Psi_{\text{causal}}$ through lines 295, 301, 307 and through `05_proofs.md:224-268`.
  3. Line 313 and line 329 then follow with the intended sign.
- Required new assumptions/permits: none.
- Validation plan: Recompute the Euler-Lagrange equation for the flat-metric case $L = \tfrac12\dot z^2 - V + \beta\Psi$ and confirm $\ddot z = -V' + \beta\Psi'$.

### [E-010] Overdamped step is invalid; $\nabla_A V$ and $\beta_{\text{curl}}\mathcal{F}$ appear without a generating term; wrong section cited (was F-008)
- Location: Section 32.3, Theorem thm-augmented-drift-law, proof lines 298-318
- Severity: Moderate
- Type: Invalid inference (secondary: Proof gap / omission; Citation / reference error)
- Criterion: Framework
- Origin: this chapter (the $\nabla_A V$ and $\beta_{\text{curl}}$ insertion) and upstream `10_appendices/05_proofs.md` E.5 (frictionless overdamped limit and wrong pointer)
- Claim (verbatim): lines 298-301: "Expanding the left-hand side ...: $G_{kj}\ddot z^j + [ij,k]\dot z^i\dot z^j = -(\nabla_A V)_k - \beta_{\text{exp}}\partial_k\Psi_{\text{causal}} + \beta_{\text{curl}}\mathcal{F}_{kj}\dot z^j$."; lines 310-313: "In the overdamped limit ({ref}`sec-the-unified-effective-potential`), the acceleration term vanishes and the drift field is $\dot z = \mathcal{M}_{\text{curl}}(\dots)$".
- Upstream anchor: `05_geometry/04_equations_motion.md:909-925` (thm-overdamped-limit, in section labelled `sec-the-overdamped-limit` at `:885`): "$m\,\ddot z^k + \gamma\,\dot z^k - \beta_{\text{curl}} G^{km}\mathcal{F}_{mj}\dot z^j + G^{kj}\partial_j\Phi + \Gamma^k_{ij}\dot z^i\dot z^j = \dots$ In the limit $\gamma\to\infty$ ... $dz^k = [\mathcal{M}_\gamma]^k{}_j(-G^{j\ell}\partial_\ell\Phi)\,ds + \dots$", with `:899` "$\mathcal{M}_\gamma := (\gamma I - \beta_{\text{curl}} G^{-1}\mathcal{F})^{-1}$". `sec-the-unified-effective-potential` (`04_equations_motion.md:396-425`) defines $\Phi_{\text{eff}}$ and contains no overdamped limit. `05_proofs.md:270` has the same wrong pointer.
- Why this is an error: (i) The Euler-Lagrange equation of the stated functional (line 295) contains neither $A$ nor $\mathcal{F}$. Line 301 replaces $-\partial_k V$ by $-(\nabla_A V)_k$ and adds $+\beta_{\text{curl}}\mathcal{F}_{kj}\dot z^j$ without any term in the action that generates them; a velocity-dependent force requires a term linear in $\dot z$ (for example $\mathcal{R}_i\dot z^i$) in $L$, which is absent. (ii) Setting $\ddot z = 0$ in the conservative equation $\ddot z^m + \Gamma^m_{ij}\dot z^i\dot z^j = F^m$ gives $\Gamma^m_{ij}\dot z^i\dot z^j = F^m$, an algebraic constraint on the velocity, not $\dot z = F$. The first-order drift $\dot z = \mathcal{M}(F)$ requires a friction term $\gamma\dot z$ and the limit $\gamma\to\infty$ with rescaled time, both absent from the functional and the argument. (iii) The cited label does not contain the overdamped limit.
- Impact on downstream results: The proof of the main theorem; cor-scientific-method-as-geodesic reuses line 307.
- Fix guidance (step-by-step):
  1. Start from the book's Langevin equation (def-bulk-drift-continuous-flow, `04_equations_motion.md:227`) with $\Phi_{\text{eff}}$ replaced by $\Phi_{\text{eff}} - \beta_{\text{exp}}\Psi_{\text{causal}}$ and the reward 1-form $A$ included as in `06_fields/02_reward_field.md:357`.
  2. Invoke thm-overdamped-limit to obtain $\dot z = \mathcal{M}_\gamma(-G^{-1}\nabla_A V + \beta_{\text{exp}}G^{-1}\nabla\Psi_{\text{causal}})$; note that $\mathcal{M}_{\text{curl}}$ at line 315 is $\mathcal{M}_\gamma$ with $\gamma = 1$.
  3. Cite `sec-the-overdamped-limit` at line 310 and at `05_proofs.md:270`.
  4. Alternatively, present thm-augmented-drift-law as the definition of the augmented drift and drop the variational derivation.
- Required new assumptions/permits: friction $\gamma > 0$ and the overdamped regime, both already present in thm-overdamped-limit.
- Validation plan: Check that every term on the right-hand side of the final drift can be traced to a term in the starting equation.

### [E-011] Corollary keeps the curl term after setting $A = 0$ (was F-009)
- Location: Section 32.3, Corollary cor-scientific-method-as-geodesic, lines 323-332
- Severity: Minor
- Type: Invalid inference
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 328-330: "Setting $V = \text{const}$ and $A=0$ implies $\nabla_A V = 0$. The equation of motion reduces to $\ddot z^m + \Gamma^m_{ij}\dot z^i\dot z^j = \beta_{\text{exp}} G^{mk}\partial_k\Psi_{\text{causal}} + \beta_{\text{curl}} G^{mk}\mathcal{F}_{kj}\dot z^j$. The agent follows geodesics modified by the curiosity potential (and any value-curl drift)"
- Upstream anchor: `06_fields/02_reward_field.md:196`: "Define the non-exact component $A := \delta\Psi + \eta$, so $\mathcal{R} = d\Phi + A$ and $\mathcal{F} = dA$."; `:216`: "$\mathcal{F} := d\mathcal{R} = dA = d\delta\Psi$."
- Why this is an error: $A = 0$ forces $\mathcal{F} = dA = 0$, so the term $\beta_{\text{curl}}G^{mk}\mathcal{F}_{kj}\dot z^j$ and the phrase "any value-curl drift" cannot survive the hypothesis. The line also carries $+\beta_{\text{exp}}$ where line 307 has $-\beta_{\text{exp}}$ (E-009).
- Impact on downstream results: none beyond the corollary.
- Fix guidance: Either assume only $V = \text{const}$ (keeping $A$ and $\mathcal{F}$) or, if $A = 0$ is assumed, drop the curl term and the parenthetical.
- Required new assumptions/permits: none.
- Validation plan: Substitute $A = 0$ into the corrected line 307 and compare.

### [E-012] Interventional Closure: the condition is automatically inherited, making the advertised violation impossible; positivity hypothesis missing (was F-010)
- Location: Section 32.4, Theorem thm-interventional-closure, lines 363-380; Appendix E.6 (`10_appendices/05_proofs.md:282-325`)
- Severity: Moderate
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter and upstream `10_appendices/05_proofs.md` E.6
- Claim (verbatim): lines 366-369: "The macro-ontology $K$ is **Interventionally Closed** if and only if ... $I(K_{t+1} ; Z_{\text{micro}, t} | K_t, do(K^{\text{act}}_t)) = 0$"; line 376: "If the observational distribution is closed ($I = 0$), and the mechanism is invariant, the interventional distribution is necessarily closed. A violation ($I > 0$ under $do$) implies the existence of a back-door path through $Z_{\text{micro}}$ that was previously unobserved, necessitating a topological expansion of $K$".
- Upstream anchor: `01_foundations/02_control_loop.md:1086-1095`: "The macro-model requirement is the conditional independence $K_{t+1} \perp\!\!\!\perp (z_{n,t}, z_{\mathrm{tex},t}) \mid (K_t,K^{\text{act}}_t)$, equivalently ... $I(K_{t+1};z_{n,t},z_{\mathrm{tex},t}\mid K_t,K^{\text{act}}_t)=0$."; `05_proofs.md:308-318`: "If the observational distribution satisfies $I = 0$, then: $P(K_{t+1}|K_t,K^{\text{act}}_t) = P(K_{t+1}|K_t,K^{\text{act}}_t,Z_{\text{micro},t}) \quad \forall Z_{\text{micro},t}$. Since the mechanism is invariant under intervention ... Therefore, $I(\dots|K_t, do(K^{\text{act}}_t)) = 0$."
- Why this is an error: (i) The first half of the proof concludes that causal enclosure (the book's standing macro-model requirement) plus mechanism invariance implies interventional closure. If accepted, an enclosure-correct ontology can never violate the interventional condition and the trigger described in the next sentence and in the Interpretation (line 372) cannot fire. (ii) The inference is incomplete: $I = 0$ under $P$ gives the identity of conditionals only $P$-a.e. on the observational support of $(K_t, K^{\text{act}}_t, Z_{\text{micro},t})$. Observationally $K^{\text{act}} = \pi(Z_t)$ depends on $Z_{\text{micro}}$; under $do(K^{\text{act}} = k)$ the interventional measure puts mass on triples $(K_t, k, Z_{\text{micro}})$ with zero observational probability, where the identity says nothing. The correct statement is that interventional closure follows from observational enclosure if the observational policy has full support (positivity), and can fail on the off-support region otherwise. This hypothesis is stated nowhere; E.6's "Hypothesis: Let $\mathcal{M}$ be a Markov Blanket for $K$" is not it. Mechanism invariance itself also presupposes that $(K_t, Z_{\text{micro},t})$ contains all common causes of $K^{\text{act}}_t$ and $K_{t+1}$ (the same causal sufficiency issue as E-005). (iii) An iff whose right-hand side is the defining condition is a definition, not a theorem.
- Impact on downstream results: Ontological Expansion trigger (`07_cognition/04_ontology.md`), Node 53 remediation bullet 2 (line 439), Summary item 4, `09_economics/01_pomw.md:309, 368`.
- Fix guidance (step-by-step):
  1. Restate the displayed condition as Definition (Interventional Closure).
  2. Add Proposition: if causal enclosure holds under $P$, the mechanism $P(K_{t+1}|K_t,K^{\text{act}}_t,Z_{\text{micro},t})$ is invariant under $do$, and the observational policy satisfies positivity ($\pi(k^{\text{act}}|z) > 0$ for all $k^{\text{act}}$, $P$-a.e. $z$), then $K$ is interventionally closed.
  3. Add Corollary: a measured violation indicates either an off-support action regime (exploration has exposed new micro-dependence) or failure of observational enclosure itself; the ontology-expansion trigger applies to the former.
  4. Update E.6 accordingly.
- Required new assumptions/permits: positivity of the observational policy; causal sufficiency of $(K_t, Z_{\text{micro},t})$ for the mechanism.
- Framework-first proof sketch for the fix: under positivity every interventional triple is in the observational support, so the a.e. identity of conditionals transfers pointwise and the conditional mutual information under $P_{do}$ vanishes.
- Validation plan: Construct a three-variable toy ($K$, binary $Z_{\text{micro}}$, deterministic policy $K^{\text{act}} = Z_{\text{micro}}$) where the transition depends on $Z_{\text{micro}}$ only at the off-support action; check that observational $I = 0$ and interventional $I > 0$.

### [E-013] Simpson's-paradox example states the overall comparison backwards (was F-011)
- Location: Section 32.4, Example: Simpson's Paradox and Interventional Debugging, line 387
- Severity: Minor
- Type: Miswording
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "Looking at the overall data, Treatment A has a higher success rate: 78% vs 83%. So B is better, right?"
- Upstream anchor: none; self-contained classical example. Recomputed from the Charig et al. (1986) data: A small $81/87 = 93.1\%$, large $192/263 = 73.0\%$, overall $273/350 = 78.0\%$; B small $234/270 = 86.7\%$, large $55/80 = 68.8\%$, overall $289/350 = 82.6\%$.
- Why this is an error: B has the higher overall rate; the sentence attributes it to A while giving numbers that say the opposite. The following sentence and the rest of the example ("Treatment A is better in every stratum, but worse overall!") presuppose B higher overall. The stratum numbers are correct.
- Impact on downstream results: none mathematical; misstates the one fact the example exists to show.
- Fix guidance: "Treatment B has a higher overall success rate: 83% vs 78%. So B is better, right?"
- Required new assumptions/permits: none.
- Validation plan: Read the paragraph once more for consistency.

### [E-014] Algorithm 32.5.1 does not implement the theory it is said to implement (was F-012)
- Location: Section 32.5, Algorithm 32.5.1, lines 413-420; line 246
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 417-420: "1. **Monitor Volatility:** Track $\gamma(z)$ (World Model scaling coefficient) across the manifold. 2. **Generate Hypothesis:** Identify a region $U \subset \mathcal{Z}$ where $\Delta_{\text{causal}}$ is high. 3. **Execute do-operation:** Inject a Neumann impulse $u_\pi$ to drive the state into $U$. 4. **Update Kernel:** Correct $\bar{P}$ using the interventional feedback, reducing $\Psi_{\text{causal}}$." Line 246: "The Experimental Sieve (Algorithm 32.5.1) selects interventions $do(a)$ that maximize the **Varentropy of the expected outcome distribution**."
- Upstream anchor: `02_sieve/01_diagnostics.md:206` ($\gamma$ is a single dimensionless volatility scale, not a function of $z$). Chapter-internal: line 157 and line 285 make $\nabla\Psi_{\text{causal}}$ the exploration driver.
- Why this is an error: (i) Three different selection criteria are stated: $\Psi_{\text{causal}}$ (theory), varentropy (line 246), $\Delta_{\text{causal}}$ (step 2). $\Delta_{\text{causal}}$ is by definition a KL between the interventional and observational kernels at $(z,a)$, so evaluating it requires interventional samples at $(z,a)$; it cannot be the criterion used to decide where to intervene before any intervention has been performed there. $\Psi_{\text{causal}}$ can, since it is computed from the current posterior. (ii) Step 1 treats $\gamma$ as a field on $\mathcal{Z}$; no local version is defined anywhere. (iii) Step 3 is navigation to a region, not a $do$ on the action variable as defined at line 95; the experiment (which $a$ to force at $U$) is never specified. (iv) Step 4 mentions $\Psi_{\text{causal}}$, which the algorithm never evaluates.
- Impact on downstream results: Node 53 (lines 422-441) cites the algorithm; `09_economics/01_pomw.md:1488` links to this section for the PoUW causal check.
- Fix guidance (step-by-step):
  1. Step 1: compute $\Psi_{\text{causal}}(z,a)$ from the current posterior over $\theta_W$ (or a stated tractable estimator).
  2. Step 2: choose $(z^\star, a^\star) \in \arg\max \Psi_{\text{causal}}$ subject to the Sieve constraints.
  3. Step 3: navigate to $z^\star$ via the drift law and execute $do(a^\star)$.
  4. Step 4: update the posterior and record $\Delta_{\text{causal}}(z^\star, a^\star)$ for Node 53.
  5. Remove $\gamma(z)$ or define it; reconcile line 246 with the adopted criterion (see E-006).
- Required new assumptions/permits: a tractable estimator of $\Psi_{\text{causal}}$ if the posterior is not available in closed form.
- Validation plan: Check that each step consumes only quantities available at the time it runs.

### [E-015] Node 53: threshold symbol, default value and section number disagree with the master tables (was F-013)
- Location: Section 32.5, Node 53: InterventionalGapCheck, lines 422-441
- Severity: Minor
- Type: Notation conflict (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter (and `06_fields/03_info_bound.md:696` for the section number)
- Claim (verbatim): line 423: "**Node 53: InterventionalGapCheck (CausalEnclosureCheck)**"; line 431: "**Threshold:** $\Delta_{\text{causal}} < \Delta_{\max}$ (typical default $\Delta_{\max} = 0.5$ nat)."
- Upstream anchor: `06_fields/03_info_bound.md:696`: "| 53 | [CausalEnclosureCheck](#node-53) | 32.6 | $\Delta_{\text{causal}} < \delta_{\text{causal}}$ |"; `09_economics/01_pomw.md:309`: "**CausalEnclosureCheck (Node 53):** $\Delta_{\text{causal}}(g) < \delta_{\text{causal}}$"; `10_appendices/04_faq.md:483`: "**Node 53 (InterventionalGapCheck)** ({ref}`Section 32.5 <sec-implementation-the-experimental-sieve>`)". A search of `10_appendices/02_parameters.md` for "causal" returns nothing.
- Why this is an error: The threshold is $\Delta_{\max}$ here and $\delta_{\text{causal}}$ in the master node table and in PoUW; the 0.5 nat default appears only here and is absent from the parameter appendix; the master table places Node 53 in 32.6 while the node lives in 32.5 (32.6 is the summary table). The name CausalEnclosureCheck also collides semantically with the observational Closure Defect of `02_control_loop.md:1098-1108`, a different quantity.
- Impact on downstream results: implementers reading `01_pomw.md` and this chapter will use different symbols for the same threshold; the parameter table is incomplete.
- Fix guidance: Use $\delta_{\text{causal}}$ in both places; add it (with the 0.5 nat default) and $\beta_{\text{exp}}$ to `10_appendices/02_parameters.md`; correct the section column in `06_fields/03_info_bound.md:696` to 32.5.
- Required new assumptions/permits: none.
- Validation plan: grep for both symbols across Volume 1 after the edit.

## Scope restrictions and clarifications
- The interventional operator as defined applies to the action variable only. Any statement about intervening on states (the geometric gloss, the singularity lemma, "do(z,a)" in the theorem) needs a separately defined operator.
- The Causal Deficit is non-degenerate only when the agent's latent $z$ is not causally sufficient (a hidden environment state or texture variable influences both the action and the transition) or when it is defined as a learned-versus-empirical gap. The chapter should say which.
- The variational derivation of the drift law produces a conservative second-order equation; the first-order drift used everywhere else in the book requires the friction term and the overdamped limit of `05_geometry/04_equations_motion.md`, not a bare "acceleration vanishes" step.
- The Noisy-TV property holds because the epistemic term of the EIG vanishes when the likelihood is independent of $\theta_W$; it does not follow from, and is not equivalent to, a varentropy criterion.
- Interventional closure follows from observational enclosure only under positivity of the observational policy and causal sufficiency of $(K, Z_{\text{micro}})$; violations can occur only where those hypotheses fail.

## Open questions
- Is the intended $\Delta_{\text{causal}}$ a property of the environment (true confounding, option (a) in E-005) or of the learned model (estimation gap, option (b))? The downstream uses in `03_memory_retrieval.md` ("surprise signal") and `01_pomw.md` (gradient consistency check) read more naturally as option (b).
- Should $\Psi_{\text{causal}}$ enter the drift as $\mathbb{E}_{a\sim\pi}$ or $\max_a$? The choice changes whether the curiosity force depends on the current policy.
- Is the appeal to varentropy meant as an estimator of EIG (in which case a lemma restricting the hypothesis family is needed) or as an independent design criterion?

## Rejected candidate findings
- None. All thirteen stage-1 findings were confirmed; two were adjusted (F-006: criterion External changed to Framework; F-007: severity Major lowered to Moderate because the error is confined to the proof and the theorem statement is the intended, downstream-used one).
