# Mathematical Review: docs/source/1_agent/07_cognition/08_intersubjective_metric.md

## Metadata
- Reviewed file: docs/source/1_agent/07_cognition/08_intersubjective_metric.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1430 lines), including the implementation block and the diagnostic nodes
- Framework anchors (definitions/axioms/permits) consulted:
  - `08_multiagent/01_gauge_theory.md`: def-strategic-connection (1436-1451), prop-gauge-transformation-connection (1458-1464), prop-minimal-coupling (1558-1575), def-gauge-covariant-game-tensor (1619-1640), def-field-strength-tensor (1684-1700), def-yang-mills-action (1859-1880), def-joint-wfr-action (875-890), def-cognitive-action-scale (2956-2975)
  - `08_multiagent/02_standard_model.md`: def-cognitive-spinor (864), thm-three-cognitive-forces (1067), cor-ontological-ssb (1349-1375)
  - `06_fields/03_info_bound.md`: thm-causal-information-bound (255-262)
  - `06_fields/02_reward_field.md`: def-value-curl (211-218)
  - `intro_agent.md:42` ($G_{\text{Fragile}}$)
  - `09_economics/01_pomw.md`: def-network-metric-tensor (756-763), thm-metric-friction-bft proof Step 1 (810), thm-corruption-babel-detection (1335-1360)
  - `10_appendices/06_losses.md`: def-f-sync-potential (480-497), def-f-joint-prediction (636-665)
  - `10_appendices/02_parameters.md` (parameter table)
  - `08_multiagent/05_architecture.md:2111-2126`, `07_cognition/07_metabolic_transducer.md:1191,1214`, `07_cognition/09_retrieval_attention.md:1273` (node numbering)
  - Mechanical cross-reference report (duplicate labels)

## Executive summary
- Critical: 1
- Major: 2
- Moderate: 14
- Minor: 9
- Notes: 1
- Primary themes:
  1. The chapter's central chain, "minimise the Yang–Mills energy of the inter-agent connection, hence the inter-agent curvature vanishes, hence metric friction vanishes and the manifolds become isometric", does not hold. The Locking Curvature is a functional of the gauge connections only and is blind to the metrics $G_A, G_B$; flat connections with arbitrarily different metrics are an explicit counterexample. The symbol $\mathcal{F}_{AB}$ names both the scalar metric friction and the curvature 2-form, which is what makes the chain look like a tautology (E-007). The headline theorem on objective reality, the perfect-translation corollary and the PoMW consensus proof all consume this chain.
  2. The finite-$\beta_c$ phase transition is asserted rather than derived: the objective is silently swapped between statement and proof, no term in either objective competes with $\beta\Psi_{\text{sync}}$, the pure-gauge and gauge-fixing steps use Abelian formulas in a non-Abelian theory, the vacuum expectation value misses a factor $\sqrt2$, and $\beta_c$ and $N_c$ are quoted without derivation and with inconsistent units (E-010 through E-013, E-020).
  3. Three couplings ($\lambda_{\text{lock}}, \beta, g_{\text{lock}}$) are never related, and $g_{\text{lock}}$ is simultaneously the gauge coupling and the quartic Landau coefficient (E-003).
  4. The Babel Limit rebrands the static representational bound $I_{\max}$ as a Shannon capacity, multiplies the differential entropy of the metric tensor by the gauge-algebra dimension, and mixes rates with totals; PoMW's corruption-detection theorem inherits it (E-008, E-016).
  5. The domain of integration $\mathcal{Z}_{\text{shared}}$ is used before it exists (it is constructed only after locking) and appears with three different measures in the chapter and a fourth in the losses appendix (E-004, E-027).
  6. Implementation: the Procrustes rotation is transposed (numerically verified: residual 21.6 where the correct solution gives $10^{-14}$), and `check_babel_limit` returns "satisfied" independently of the metric whenever the mean log-eigenvalue is non-positive (E-023, E-025).
  7. Mechanical: labels `node-69` and `node-70` are defined in two chapters and the Sieve numbering 69/70 is used for three different pairs of diagnostics (E-026).

## Error log
| ID | Location (section, lines) | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Metric Friction, lem-friction-bounds-utility (102-116) | Moderate | Invalid inference | External (Framework for covariant gradient) | this chapter | Proof sketch invokes a non-existent exponential decay of cosine and an undefined covariant gradient of a gauge scalar |
| E-002 | def-inter-agent-connection (141-158) | Moderate | Proof gap / omission | Framework | this chapter | Coupling Connection $\mathcal{C}_{AB}$ never defined; index placement inconsistent |
| E-003 | def-locking-curvature (180-184), def-gauge-alignment-order-parameter (297-305), Step 11 (422), cor-critical-coupling-locking (455), 931, 963, 1119 | Moderate | Notation conflict | Framework | this chapter | $g_{\text{lock}}$ is both gauge coupling and quartic coefficient; $\lambda_{\text{lock}}, \beta, g_{\text{lock}}$ never related |
| E-004 | def-locking-curvature (184-189), Steps 4 and 9 (365-406), cor-critical-coupling-locking (455), thm-emergence-objective-reality (871-878) | Moderate | Circular reasoning | Framework | this chapter | $\mathcal{Z}_{\text{shared}}$ used as integration domain before it is constructed in the locked limit; measures $G_{\text{shared}}$, $G_{AB}$, $d\mu_G$ all differ; $2D$ vs $D$ dimensions |
| E-005 | thm-locking-operator-derivation (205-243), Step 7 (388) | Moderate | Definition mismatch | Framework (and External) | this chapter | Lorentzian minus sign kept on a Riemannian domain, so $\mathfrak{L}_{\text{sync}} \le 0$ and minimising it maximises curvature; $\mathfrak{L}_{\text{sync}}$ and $\Psi_{\text{sync}}$ conflated |
| E-006 | thm-locking-operator-derivation Step 1 (217) | Minor | Citation / reference error | Framework | this chapter | Spinor transformation law attributed to def-gauge-covariant-game-tensor |
| E-007 | thm-locking-operator-derivation Steps 3, 6 (221-239); thm-emergence-objective-reality (868-903); cor-perfect-translation (659-666); TLDR/Abstract (8-9, 32-39); Summary (1418-1428) | Critical | Invalid inference | Framework and External | this chapter | Vanishing Locking Curvature does not imply metric alignment; the $d_{\text{GH}}$ bound is false; $\mathcal{F}_{AB}$ overloaded |
| E-008 | ax-finite-communication-bandwidth (253-265), thm-babel-limit Step 1 (709-714) | Moderate | Definition mismatch | Framework | this chapter | Static $I_{\max}$ rebranded as channel capacity; $\partial\mathcal{L}$ undefined for a linear map; inequality/equality switch |
| E-009 | def-gauge-alignment-order-parameter (284-307), Step 11 (422) | Minor | Scope restriction | External | this chapter | $|\text{Tr}(U_A U_B^\dagger)| \le N$ bounds the order parameter; representation unspecified |
| E-010 | thm-spontaneous-gauge-locking (321-433) | Major | Proof gap / omission | Framework | this chapter | Objective swapped mid-proof; no competitor to $\beta\Psi_{\text{sync}}$; finite $\beta_c$ not derived; $U_i$ not varied |
| E-011 | thm-spontaneous-gauge-locking Steps 9-10 (399-415) | Moderate | Computational error | External and Framework | this chapter | Abelian pure-gauge and gauge-fixing formulas in a non-Abelian theory; missing $1/g$; 1-forms on different manifolds subtracted |
| E-012 | thm-spontaneous-gauge-locking Step 11 (419-425), 312 | Moderate | Computational error | External (contradicts cor-ontological-ssb) | this chapter | $v_{\text{lock}}$ off by $\sqrt2$ |
| E-013 | cor-critical-coupling-locking (449-463) | Moderate | Proof gap / omission | Framework | this chapter | $\beta_c$ has no derivation; the balanced kinetic term appears in no objective; units cannot match |
| E-014 | def-translation-operator (558-578) | Moderate | Definition mismatch | External and Framework | this chapter | Message factor is a $U(1)$ phase; Property 3 false; path between different manifolds |
| E-015 | thm-untranslatability-bound (606-647) | Moderate | Proof gap / omission | External and Framework | this chapter | $\mathcal{U}_{AB}(m)$ undefined; boundary integral against an area element |
| E-016 | thm-babel-limit (697-736), cor-ineffability-theorem (750-764) | Major | Conceptual | External and Framework | this chapter | $\dim\mathfrak{g} \cdot H(G_A)$ counts neither the gauge parameters nor the metric; rates vs totals; negative differential entropy |
| E-017 | def-metric-eigendecomposition (796-811) | Minor | Definition mismatch | External and Framework | this chapter | Metric eigenvalues called principal curvatures; $\sigma_k$ clashes with $\sigma$ |
| E-018 | thm-emergence-objective-reality Steps 2-4 (884-895), def-metric-friction (77-84) | Minor | Definition mismatch | External | this chapter | Coordinate-dependent clause in the equivalence relation; $\phi$ never assumed a diffeomorphism |
| E-019 | rem-echo-chamber-effect (923-939), Node 70 (1371-1377) | Minor | Proof gap / omission | Framework | this chapter | $\mathcal{F}_{AE}$ needs an environment metric that does not exist; Node 70 proxy stated two ways |
| E-020 | cor-critical-mass-consensus (957-971) | Moderate | Proof gap / omission | Framework | this chapter | $N_c$ stated without derivation and not dimensionless |
| E-021 | Kuramoto isomorphism (1107-1112) | Minor | Computational error | Framework | this chapter | Sign: the stated dynamics is gradient ascent on friction |
| E-022 | Implementation docstrings (1156, 1160, 1182, 1188, 1208, 1285, 1294, 1305) | Minor | Citation / reference error | Framework | this chapter | Stale "37.x" numbering that matches nothing in the chapter |
| E-023 | `compute_metric_friction` (1218-1224) | Moderate | Algorithm mismatch | External | this chapter | Procrustes rotation is the transpose of the minimiser; nonzero friction for isometric clouds |
| E-024 | `forward`, `compute_metric_friction` (1216, 1272-1297) | Note | Algorithm mismatch | Framework | this chapter | Procrustes step makes the learnable gauge transform inoperative; loss units are $[z]^2$, not nats |
| E-025 | `check_babel_limit` (1299-1326) | Moderate | Algorithm mismatch | Framework | this chapter | Returns "satisfied" independent of the metric when the mean log-eigenvalue is $\le 0$; implements neither stated formula |
| E-026 | Diagnostic Nodes 69-70 (1352-1353, 1368-1369), 937 | Minor | Citation / reference error | Framework | this chapter and `08_multiagent/05_architecture.md` | Duplicate MyST labels `node-69`, `node-70`; Sieve numbers 69/70 reused across three chapters |
| E-027 | `10_appendices/06_losses.md` def-f-sync-potential (480-497) | Minor | Definition mismatch | Framework | upstream `10_appendices/06_losses.md` | Appendix writes $\Psi_{\text{sync}}$ as a boundary integral with an area element, a fourth inconsistent domain |

## Detailed findings

### [E-001] Friction-bounds-utility lemma: the proof sketch does not produce the stated bound (was F-001)
- Location: The Solipsism Problem: Metric Friction, lem-friction-bounds-utility, lines 102-116
- Severity: Moderate
- Type: Invalid inference (secondary: Conceptual)
- Criterion: External (Framework for the covariant-gradient part)
- Origin: this chapter
- Claim (verbatim, lines 114): "the agents' covariant value gradients $\nabla_{A^{(A)}} V_A$ and $\nabla_{A^{(B)}} V_B$ (with $A^{(i)}$ the non-conservative component of agent $i$'s reward 1-form) misalign by an angle $\theta \propto \sqrt{\mathcal{F}_{AB}}$. The effective cooperative gradient is $|\nabla_{A^{(\text{coop})}} V_{\text{coop}}| = |\nabla_{A^{(A)}} V_A| \cos\theta$. Integrating the exponential decay of cosine near $\theta = \pi/2$ yields the bound."
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1573` (prop-minimal-coupling): "Scalar objectives like $V$ remain invariant and use ordinary gradients." `06_fields/02_reward_field.md:211-218` (def-value-curl): "$\mathcal{F} := d\mathcal{R} = dA = d\delta\Psi$", so $A$ is the co-exact part of the reward 1-form, not a connection.
- Why this is an error: Recomputed: $\cos\theta = \pi/2 - \theta + O((\theta-\pi/2)^3)$ near $\pi/2$ and $1 - \theta^2/2 + O(\theta^4)$ near $0$. There is no exponential decay to integrate, and no integration is specified. With $\theta \propto \sqrt{\mathcal{F}_{AB}}$ the natural bound is $V_{\text{coop}} \lesssim V_{\max}(1 - c\,\mathcal{F}_{AB})$, which matches $e^{-\mathcal{F}_{AB}/\mathcal{F}_0}$ only to first order and becomes negative rather than exponentially small once $\theta > \pi/2$. The proportionality $\theta \propto \sqrt{\mathcal{F}_{AB}}$ is not derived: $\mathcal{F}_{AB}$ compares two metrics through a pullback, not two value gradients on different manifolds. Finally, $V$ is a gauge scalar in the framework and takes ordinary gradients; a "covariant value gradient with respect to the reward 1-form" has no definition.
- Impact on downstream results: Local; the lemma is used rhetorically only. `09_economics/01_pomw.md:763` builds on def-metric-friction, not on this lemma.
- Fix guidance:
  1. Either restate the lemma as a modelling assumption ("we posit $V_{\text{coop}} \le V_{\max} e^{-\mathcal{F}_{AB}/\mathcal{F}_0}$") or
  2. Prove a first-order bound $V_{\text{coop}} \le V_{\max}(1 - c\,\tilde{\mathcal{F}}_{AB})$ with the angle defined explicitly between $\nabla V_A$ and $\phi^* \nabla V_B$ in the $G_A$ inner product.
  3. Remove the $\nabla_{A^{(i)}}$ notation.
- Required new assumptions/permits: an explicit relation between the pullback metric distortion and the angle between pulled-back gradients (e.g. a Lipschitz bound on $\nabla V_B \circ \phi$).
- Validation plan: check that the proven bound is consistent at $\mathcal{F}_{AB} = 0$ ($V_{\text{coop}} = V_{\max}$) and remains non-negative for all $\mathcal{F}_{AB}$.

### [E-002] The Coupling Connection $\mathcal{C}_{AB}$ is never defined (was F-002)
- Location: The Inter-Agent Connection, def-inter-agent-connection, lines 141-158
- Severity: Moderate
- Type: Proof gap / omission (secondary: Typo)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 147, 153): "$\mathcal{A}_{AB}^\mu(z_A, z_B) := A_\mu^{(A)}(z_A) \otimes \mathbb{1}_B + \mathbb{1}_A \otimes A_\mu^{(B)}(z_B) + \lambda_{\text{lock}} \mathcal{C}_{AB}^\mu$" and "$\mathcal{C}_{AB}^\mu$ is the **Coupling Connection** encoding the interaction".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1436-1451` (def-strategic-connection): "$A = A_\mu^a T_a \, dz^\mu$ ... $\mu$ indexes spacetime/latent coordinates $(t, z^1, \ldots, z^D)$", defined on a single agent's $\mathcal{Z}$.
- Why this is an error: $\mathcal{C}_{AB}$ appears only at lines 147 and 153 and is given no formula, no Lie-algebra value, no transformation law under $(U_A, U_B)$ and no relation to messages, yet it is the only term that can couple the two gauges: since $[A^{(A)} \otimes \mathbb{1}, \mathbb{1} \otimes A^{(B)}] = 0$, the tensor-sum part has curvature $\mathcal{F}^{(A)} \otimes \mathbb{1} + \mathbb{1} \otimes \mathcal{F}^{(B)}$, which carries no information about the relative gauge. The LHS has an upper index $\mu$ and the RHS a lower one, and which factor of the $2D$-dimensional product each $\mu$ ranges over is not specified.
- Impact on downstream results: def-locking-curvature, thm-locking-operator-derivation, thm-spontaneous-gauge-locking, cor-perfect-translation.
- Fix guidance:
  1. Define $\mathcal{C}_{AB}$ explicitly through the identification map, e.g. $\mathcal{C}_{AB} := \phi_{A \to B}^* A^{(B)} - A^{(A)}$, a $\mathfrak{g}$-valued 1-form on $\mathcal{Z}_A$ whose curvature measures relative gauge.
  2. State its transformation law under $(U_A, U_B)$.
  3. Fix the index placement and specify the coordinates on the product.
- Required new assumptions/permits: a map $\phi_{A \to B}$ regular enough to pull back 1-forms (at least $C^1$).
- Validation plan: verify that with the explicit $\mathcal{C}_{AB}$ the curvature vanishes exactly when $\phi^* A^{(B)}$ and $A^{(A)}$ are gauge equivalent.

### [E-003] $g_{\text{lock}}$ denotes two different constants; $\lambda_{\text{lock}}, \beta, g_{\text{lock}}$ are never related (was F-008)
- Location: def-locking-curvature (180, 184), def-gauge-alignment-order-parameter (297-305), Step 11 (422), cor-critical-coupling-locking (455), rem-echo-chamber-effect (931), cor-critical-mass-consensus (963), Kuramoto table (1119)
- Severity: Moderate
- Type: Notation conflict (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 184 "$g_{\text{lock}}$ is the inter-agent coupling constant" (inside $-ig_{\text{lock}}[\mathcal{A}, \mathcal{A}]$ and as $1/(4g_{\text{lock}}^2)$); lines 297-305 "$\mathcal{V}_{\text{lock}}(\phi_{AB}) = -\mu_{\text{lock}}^2 |\phi_{AB}|^2 + g_{\text{lock}} |\phi_{AB}|^4$ ... $g_{\text{lock}} > 0$ is the quartic self-interaction coefficient".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1684-1690` (gauge coupling multiplies the commutator); `08_multiagent/02_standard_model.md:1355` (cor-ontological-ssb uses a separate quartic $\lambda$: "$\mathcal{V}(\phi) = -\mu^2|\phi|^2 + \lambda|\phi|^4$").
- Why this is an error: A gauge coupling and a Landau quartic coefficient are different parameters with different dimensions; the chapter uses one symbol for both and then mixes them ($v_{\text{lock}} = \sqrt{(\beta - \beta_c)/g_{\text{lock}}}$ uses the quartic reading, $\beta_c = \sigma^2 \text{Vol}/(2 g_{\text{lock}}^2)$ and its interpretation at line 474 use the gauge reading). $\lambda_{\text{lock}}$ (strength of $\mathcal{C}_{AB}$ at 154, weight of $\mathcal{F}_{AB}$ at 931, coefficient in $N_c$ at 963) and $\beta$ (weight of $\Psi_{\text{sync}}$ at 327; "Locking Coefficient" at 1119) are both called the locking strength and never related. None of the three appears in `10_appendices/02_parameters.md`.
- Impact on downstream results: cor-critical-coupling-locking, cor-critical-mass-consensus, and the code's `coupling_strength` ("Lambda_lock coefficient", 1173), which multiplies metric friction rather than $\Psi_{\text{sync}}$.
- Fix guidance:
  1. Reserve $g_{\text{lock}}$ for the gauge coupling; introduce $\lambda_4$ for the quartic term.
  2. State whether $\beta$ and $\lambda_{\text{lock}}$ are the same parameter; if not, write the joint objective with both.
  3. Add all three to the parameter table with units.
- Required new assumptions/permits: none.
- Validation plan: dimensional check of every formula containing the three symbols after renaming.

### [E-004] $\mathcal{Z}_{\text{shared}}$ is used before it exists and is defined only in the locked limit (was F-003)
- Location: def-locking-curvature (184-189); Steps 4 and 9 of thm-spontaneous-gauge-locking (365-406); cor-critical-coupling-locking (455); definition at thm-emergence-objective-reality (871-878)
- Severity: Moderate
- Type: Circular reasoning (secondary: Definition mismatch, Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 187 "$\Psi_{\text{sync}} := \int_{\mathcal{Z}_{\text{shared}}} \text{Tr}(\mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB,\mu\nu}) \sqrt{|G_{\text{shared}}|}\, d^D z$"; lines 871-874 "In the limit of perfect locking ($\mathcal{F}_{AB} \to 0$), the private manifolds $\mathcal{Z}_A$ and $\mathcal{Z}_B$ collapse into a single **Quotient Manifold**: $\mathcal{Z}_{\text{shared}} := (\mathcal{Z}_A \sqcup \mathcal{Z}_B) / \sim_{\text{isometry}}$".
- Upstream anchor: none; $\mathcal{Z}_{\text{shared}}$ occurs nowhere else in Volume 1.
- Why this is an error: The potential that is supposed to drive locking is integrated over a manifold constructed only after locking is complete. Before locking there is no $\mathcal{Z}_{\text{shared}}$, so $\Psi_{\text{sync}}$, $\mathfrak{L}_{\text{sync}}$, the "simply-connected $\mathcal{Z}_{\text{shared}}$" hypothesis of Step 9 and $\text{Vol}(\mathcal{Z}_{\text{shared}})$ in $\beta_c$ are undefined precisely where they are used. The measure is written three ways ($\sqrt{|G_{\text{shared}}|}\,d^D z$ at 187, $\sqrt{|G_{AB}|}\,d^D z$ at 211, $d\mu_G$ at 368), and the losses appendix adds a fourth (E-027). The connection lives on the $2D$-dimensional product (144) but the integral is $D$-dimensional.
- Impact on downstream results: everything built on $\Psi_{\text{sync}}$.
- Fix guidance:
  1. Choose a pre-locking domain: $\mathcal{Z}_A$ with structure pulled back through $\phi_{A \to B}$, or the graph of $\phi$ inside $\mathcal{Z}_A \times \mathcal{Z}_B$ with the induced metric.
  2. Use one name for the measure.
  3. Rename the quotient in thm-emergence-objective-reality to avoid the collision.
- Required new assumptions/permits: existence and regularity of $\phi_{A \to B}$ before locking.
- Validation plan: confirm every occurrence of $\mathcal{Z}_{\text{shared}}$ refers to an object defined earlier in the text.

### [E-005] Sign and normalisation of the "Yang–Mills energy" (was F-004)
- Location: thm-locking-operator-derivation (205-243); Step 7 of thm-spontaneous-gauge-locking (388)
- Severity: Moderate
- Type: Definition mismatch (secondary: Computational error)
- Criterion: Framework (and External)
- Origin: this chapter
- Claim (verbatim, lines 211, 228, 232): "$\mathfrak{L}_{\text{sync}}(G_A, G_B) := -\frac{1}{4g_{\text{lock}}^2} \int_{\mathcal{Z}_{\text{shared}}} \text{Tr}(\mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB,\mu\nu}) \sqrt{|G_{AB}|}\, d^D z$"; "physical configurations minimize the integrated curvature squared"; "The Locking Operator generates a **Synchronizing Potential** $\Psi_{\text{sync}}$".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1859-1880` (def-yang-mills-action): "$S_{\text{YM}}[A] = -\frac{1}{4}\int_{\mathcal{Z} \times \mathbb{R}} \text{Tr}(\mathcal{F}_{\mu\nu}\mathcal{F}^{\mu\nu})\sqrt{|g|}\,d^{D+1}x$ ... $g_{\mu\nu} = \text{diag}(-c_{\text{info}}^2, \tilde{G}_{ij})$ ... **Positive in Euclidean signature:** After Wick rotation the action is positive semi-definite for compact gauge groups; in Lorentzian signature it is indefinite."
- Why this is an error: The upstream minus sign belongs to a Lorentzian action with a time direction. The chapter drops the time direction and integrates over a Riemannian latent domain, where $\text{Tr}(\mathcal{F}^{\mu\nu}\mathcal{F}_{\mu\nu}) = \sum_a \mathcal{F}^a_{\mu\nu}\mathcal{F}^{a\,\mu\nu} \ge 0$; hence $\mathfrak{L}_{\text{sync}} \le 0$ with maximum $0$ at flat connections, and "minimising" it (Step 4; Step 7 "energy minimum requires $\Psi_{\text{sync}} \to 0$") is self-contradictory. $\Psi_{\text{sync}}$ (187, non-negative) and $\mathfrak{L}_{\text{sync}} = -\Psi_{\text{sync}}/(4g^2_{\text{lock}})$ differ by a negative constant, yet the text treats one as "generating" the other; the joint loss (327) and `06_losses.md:485` both use $+\beta\Psi_{\text{sync}}$.
- Impact on downstream results: Step 7 of thm-spontaneous-gauge-locking; code comment "Locking loss (Theorem 37.1)".
- Fix guidance:
  1. Define $\mathfrak{L}_{\text{sync}} := +\frac{1}{4g^2_{\text{lock}}}\int \text{Tr}(\mathcal{F}^{\mu\nu}\mathcal{F}_{\mu\nu})\sqrt{|G|}\,d^D z = \Psi_{\text{sync}}/(4g^2_{\text{lock}})$ (Euclidean Yang–Mills energy).
  2. Use one symbol throughout.
- Required new assumptions/permits: none.
- Validation plan: check that $\mathfrak{L}_{\text{sync}} \ge 0$ with equality iff $\mathcal{F}_{AB} = 0$.

### [E-006] Wrong citation for the spinor transformation law (was F-006)
- Location: thm-locking-operator-derivation Step 1, line 217
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "By Definition {prf:ref}`def-gauge-covariant-game-tensor`, each agent's belief spinor $\psi^{(i)}$ transforms under local gauge $U^{(i)}(z) \in G_{\text{Fragile}}$."
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1619-1640` (def-gauge-covariant-game-tensor): "Because $V^{(i)}$ is a scalar, the gauge-consistent cross-sensitivity is the **Riemannian Hessian** ... $\tilde{\mathcal{G}}_{ij}^{kl}(z) := \nabla_k \nabla_l V^{(i)}$".
- Why this is an error: The cited definition concerns the Hessian of the scalar value function. The spinor and its transformation are def-cognitive-spinor (`02_standard_model.md:864`) and prop-gauge-transformation-connection (`01_gauge_theory.md:1458`).
- Impact on downstream results: none.
- Fix guidance: replace the reference.
- Required new assumptions/permits: none.
- Validation plan: build check on the reference.

### [E-007] Vanishing Locking Curvature does not imply metric alignment; the Gromov–Hausdorff bound is false (was F-005)
- Location: thm-locking-operator-derivation Steps 3 and 6 (221-239); inherited by thm-emergence-objective-reality (868-903), cor-perfect-translation (659-666), TLDR and Abstract (8-9, 32-39), Summary items 2, 3, 7 (1418-1428)
- Severity: Critical
- Type: Invalid inference (secondary: Notation conflict, External dependency)
- Criterion: Framework and External
- Origin: this chapter
- Claim (verbatim): line 221-224 "this curvature vanishes if and only if: $A_\mu^{(A)}(z) \sim A_\mu^{(B)}(z)$ (gauge equivalent)"; lines 232-239 "The Locking Operator generates a **Synchronizing Potential** $\Psi_{\text{sync}}$ that penalizes geometric disagreement. By comparison geometry, the local Gromov-Hausdorff distance satisfies: $d_{\text{GH}}(\mathcal{U}_A, \mathcal{U}_B) \leq C \cdot \|\mathcal{F}_{AB}\|^{1/2}$ for a universal constant $C > 0$. Thus $\mathfrak{L}_{\text{sync}}$ controls the metric alignment."; lines 882-891 "Perfect locking implies $\mathcal{F}_{AB}(z) = 0$ for all $z$. **Step 2.** By Definition {prf:ref}`def-metric-friction`, this means: $G_A(z) = \phi_{A \to B}^* G_B(\phi(z))$. The manifolds are isometric."; line 662-664 "Perfect translation ... if and only if the inter-agent curvature vanishes: $\mathcal{F}_{AB}^{\mu\nu} = 0$. *Interpretation:* This is equivalent to Spontaneous Gauge Locking. Perfect mutual understanding requires complete geometric alignment."; line 1420 "**Spontaneous Gauge Locking** ... proves that prediction error minimization forces geometric alignment".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1684-1690` (def-field-strength-tensor): "$\mathcal{F}_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu - ig[A_\mu, A_\nu]$", a functional of $A$ only. `08_multiagent/01_gauge_theory.md:1436-1451` (def-strategic-connection): $A$ is a $\mathfrak{g}$-valued 1-form on the nuisance bundle; `intro_agent.md:42`: $G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$, not a frame group.
- Why this is an error: (i) $\mathcal{F}_{AB}^{\mu\nu}$ (180) is built from $\mathcal{A}_{AB}$ (147), which contains $A^{(A)}, A^{(B)}, \mathcal{C}_{AB}$ and no metric; the gauge group has no frame-bundle factor, so the connection cannot encode $G$. Hence $\mathfrak{L}_{\text{sync}}(G_A, G_B)$ is not a function of the metrics, and nothing of the form "$\mathcal{F}_{AB} \to 0 \Rightarrow$ metric friction $\to 0$" can follow. Verified counterexample: take $A^{(A)} = A^{(B)} = 0$ and $\mathcal{C}_{AB} = 0$; then $\mathcal{F}_{AB}^{\mu\nu} \equiv 0$ and $\Psi_{\text{sync}} = 0$ for every metric on the domain (the metric enters only through index raising and the volume element, which cannot make a zero integrand nonzero), while $G_A = I_3$, $G_B = \text{diag}(4, 9, 16)$, $\phi = \text{id}$ give def-metric-friction $\mathcal{F}_{AB} = 298$ and $d_{\text{GH}} > 0$. No result of Riemannian comparison geometry bounds the Gromov–Hausdorff distance between two metric spaces by the curvature of an auxiliary principal-bundle connection; the "universal constant $C$" and exponent $1/2$ have no source in the framework or elsewhere. (ii) Step 3: for the tensor-sum connection the curvature splits as $\mathcal{F}^{(A)} \otimes \mathbb{1} + \mathbb{1} \otimes \mathcal{F}^{(B)} + \lambda_{\text{lock}}(\ldots)$ and vanishes iff each piece does; with $\mathcal{C}_{AB} = 0$ that means both connections are individually flat, which says nothing about gauge equivalence, and $A^{(A)}, A^{(B)}$ live on different manifolds and can only be compared after pulling back through $\phi$. (iii) The symbol $\mathcal{F}_{AB}$ is the scalar metric friction at line 80 and the $\mathfrak{g}$-valued 2-form at line 180; Steps 1-2 of thm-emergence-objective-reality read a curvature statement as a metric-friction statement, and cor-perfect-translation declares the two "equivalent". The overload is what makes the invalid chain look tautological.
- Impact on downstream results: the chapter's headline claims (TLDR 8-9, Abstract 32-39, Summary 2, 3, 7), thm-emergence-objective-reality, cor-perfect-translation, and `09_economics/01_pomw.md:810` ("By Theorem thm-spontaneous-gauge-locking ... $G^{(i)} \to G^{(j)}$"), which uses gauge locking as if it implied metric convergence.
- Fix guidance:
  1. Rename one of the two objects (e.g. $\Phi_{AB}$ for metric friction, $\mathcal{F}_{AB}^{\mu\nu}$ for curvature) and re-read every occurrence.
  2. Either (a) add a metric-coupling term to the objective, $\beta\Psi_{\text{sync}} + \beta_G \int \Phi_{AB}\, d\mu$, and prove metric convergence from the second term (the gauge term then aligns nuisance frames only); or (b) restrict all conclusions to alignment of the nuisance-bundle gauge and drop or mark as conjectures the $d_{\text{GH}}$, isometry and "objective reality" statements.
  3. Delete the comparison-geometry bound of Step 6 or replace it by a proven statement about whatever quantity is actually controlled.
  4. Update `09_economics/01_pomw.md:810` to cite the metric-coupling result once it exists.
- Required new assumptions/permits: for route (a), a permit that the metric-friction term is minimised by the agents' learning dynamics (a gradient-flow lemma on $\Phi_{AB}$ with respect to encoder parameters); for route (b), none.
- Framework-first proof sketch for the fix (route a): with $\Phi_{AB} = \|G_A - \phi^* G_B\|^2_{G_A}$ and the encoders trained by gradient descent on $\beta_G \int \Phi_{AB}$, $\Phi_{AB}$ is a Lyapunov function; its zero set is exactly the isometry condition of def-metric-friction, giving Steps 1-2 of thm-emergence-objective-reality honestly.
- Validation plan: apply the counterexample above to the revised objective; it must assign strictly positive loss.

### [E-008] Finite Communication Bandwidth axiom rebrands $I_{\max}$ as a channel capacity (was F-007)
- Location: ax-finite-communication-bandwidth (253-265); thm-babel-limit Step 1 (709-714)
- Severity: Moderate
- Type: Definition mismatch (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 256-259, 712): "The communication channel $\mathcal{L}$ between agents has finite Shannon capacity $C_{\mathcal{L}}$. By the Causal Information Bound ... $C_{\mathcal{L}} \leq \nu_D \cdot \frac{\text{Area}(\partial\mathcal{L})}{\ell_L^{D-1}}$" and "$C_{\mathcal{L}} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{L})}{\ell_L^{D-1}}$".
- Upstream anchor: `06_fields/03_info_bound.md:255-262` (thm-causal-information-bound): "the maximum information $I_{\max}$ that can be stably represented without the metric becoming singular is: $I_{\max} = \nu_D \cdot \frac{\text{Area}(\partial\mathcal{Z})}{\ell_L^{D-1}}$".
- Why this is an error: Upstream $I_{\max}$ is a total of stored nats for a latent manifold with boundary; it is not a rate or a channel capacity. The chapter applies it to $\mathcal{L}$, which is a linear projection $\mathfrak{g} \to \mathfrak{g}_{\mathcal{L}}$ (533-538) with no boundary, so $\text{Area}(\partial\mathcal{L})$ is meaningless; the axiom writes $\le$ and Step 1 writes $=$; and $C_{\mathcal{L}}$ is then compared with an "entropy rate" (700) and multiplied by $T$ (819), i.e. treated as nats per step. The three uses have inconsistent units.
- Impact on downstream results: thm-babel-limit, cor-ineffability-theorem, thm-spectral-locking-order, `09_economics/01_pomw.md:1337-1360`.
- Fix guidance:
  1. State the axiom as a genuine assumption "$C_{\mathcal{L}} < \infty$ (nats/step)".
  2. If a link to the area law is wanted, argue through the agent's boundary $\partial\mathcal{Z}$ and the per-step capacity of `01_foundations/02_control_loop.md`, with an explicit conversion from stored nats to nats/step.
- Required new assumptions/permits: an explicit channel model for $\mathcal{L}$ (input alphabet, noise) so that capacity is defined.
- Validation plan: unit check of every occurrence of $C_{\mathcal{L}}$.

### [E-009] The order parameter is bounded; representation unspecified (was F-012)
- Location: def-gauge-alignment-order-parameter (284-307); Step 11 (422)
- Severity: Minor
- Type: Scope restriction (secondary: Definition mismatch)
- Criterion: External
- Origin: this chapter
- Claim (verbatim, line 290): "$\phi_{AB}(z) := \text{Tr}(U_A(z) U_B^\dagger(z)) \in \mathbb{C}$ where $U_A, U_B \in G_{\text{Fragile}}$".
- Upstream anchor: `intro_agent.md:42`: "$G_{\text{Fragile}} = SU(N_f)_C \times SU(r)_L \times U(1)_Y$" (an abstract product group).
- Why this is an error: A trace requires a representation, which is not named. In any $N$-dimensional unitary representation $|\text{Tr}(U_A U_B^\dagger)| \le N$, with equality iff $U_A = e^{i\alpha} U_B$. The Landau minimum $v_{\text{lock}}$ of Step 11 grows without bound in $\beta$, so the quartic potential describes the order parameter only while $v_{\text{lock}} \le N$, i.e. $\beta - \beta_c \le 2 g N^2$.
- Impact on downstream results: Step 11, Kuramoto table.
- Fix guidance: define $\phi_{AB} := \frac{1}{N}\text{Tr}_\rho(U_A U_B^\dagger)$ in a named representation $\rho$ and state the validity range of the quartic truncation.
- Required new assumptions/permits: choice of $\rho$ (e.g. the spinor representation of def-cognitive-spinor).
- Validation plan: check $|\phi_{AB}| \le 1$ and that the stated VEV lies in range.

### [E-010] Spontaneous Gauge Locking: the finite-$\beta_c$ transition is not derived and the objective is swapped mid-proof (was F-009)
- Location: thm-spontaneous-gauge-locking (321-433)
- Severity: Major
- Type: Proof gap / omission (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 327 "$\mathcal{L}_{\text{joint}} = \|\hat{x}_{t+1}^A - x_{t+1}\|^2 + \|\hat{x}_{t+1}^B - x_{t+1}\|^2 + \beta \Psi_{\text{sync}}$"; line 375 "$\mathcal{A}_{\text{joint}} = \mathcal{A}_{\text{WFR}}^{(A)} + \mathcal{A}_{\text{WFR}}^{(B)} + \beta \Psi_{\text{sync}}$"; line 382 "$\frac{\delta \mathcal{A}_{\text{joint}}}{\delta A_\mu^{(i)}} = 0$"; line 417 "The transition from $\beta < \beta_c$ (unlocked) to $\beta > \beta_c$ (locked) is a continuous phase transition."
- Upstream anchor: `08_multiagent/01_gauge_theory.md:875-890` (def-joint-wfr-action): "$\mathcal{A}^{(N)}[\boldsymbol{\rho}, \mathbf{v}, \mathbf{r}] = \int_0^T [\sum_i \int (\|v^{(i)}\|^2_{\tilde{G}^{(i)}} + \lambda_i^2|r^{(i)}|^2)\,d\rho^{(i)} + \mathcal{V}^{\text{ret}}_{\text{int}}]\,dt$", a functional of $(\rho, v, r)$ with no $A_\mu$ dependence. `08_multiagent/02_standard_model.md:1349-1375` (cor-ontological-ssb) obtains its $\mu^2$ sign change from an explicit pitchfork equation.
- Why this is an error: (i) The statement minimises prediction errors, the proof varies WFR actions; the replacement is not justified. (ii) Neither the upstream WFR action nor $\|D^{(i)}(\psi^{(i)}) - x\|^2$ (350) is shown to depend on $A_\mu^{(i)}$, so $\delta\mathcal{A}_{\text{joint}}/\delta A^{(i)}_\mu = \beta\,\delta\Psi_{\text{sync}}/\delta A^{(i)}_\mu$; the Euler–Lagrange equations are source-free Yang–Mills for every $\beta > 0$ and $\mathcal{F} = 0$ is a global minimiser for every $\beta > 0$. With no competing term there is no threshold; the "$\beta \to \infty$ dominates" argument is vacuous and the finite $\beta_c$ of Step 11 cannot emerge. (iii) Step 11 restates the Landau potential postulated at 297-305 with $\mu^2_{\text{lock}} = \beta - \beta_c$ assumed; Steps 1-10 produce no effective potential for $\phi_{AB}$. (iv) Step 12 concludes about $\Delta U = U_A U_B^{-1}$, but $U_i$ are not dynamical variables of either objective. (v) The kinetic term $\sigma^2|\nabla\psi|^2$ used at 461 appears in neither objective.
- Impact on downstream results: cor-critical-coupling-locking, cor-critical-mass-consensus, Kuramoto isomorphism ($K_c \leftrightarrow \beta_c$), `09_economics/01_pomw.md:810`.
- Fix guidance:
  1. Choose one objective and include the terms that compete: a stochastic or entropic term for the relative gauge (e.g. a $\sigma^2$-weighted kinetic term for $\Delta U$) against $\beta\Psi_{\text{sync}}$.
  2. Derive the effective potential for $\phi_{AB}$ by a Landau expansion (integrate out fluctuations) to obtain $\mu^2_{\text{lock}}(\beta)$ and $\beta_c$.
  3. Alternatively, weaken the theorem to the $\beta \to \infty$ statement and mark the finite-$\beta_c$ transition as a conjecture.
- Required new assumptions/permits: a fluctuation model for $\Delta U$ (temperature $T_c$, diffusion on $G_{\text{Fragile}}$).
- Framework-first proof sketch for the fix: for $\Delta U = e^{i\theta}$ in the Abelian factor with energy $\beta \int |\nabla\theta|^2 - \ldots$ and thermal noise $T_c$, mean-field theory gives a Curie–Weiss potential for $\langle\cos\theta\rangle$ with $\mu^2 \propto \beta - \beta_c$, $\beta_c \propto T_c$.
- Validation plan: the derived $\beta_c$ must reproduce the dependence claimed in the interpretation at 465-477 and pass a units check.

### [E-011] Steps 9-10 use Abelian formulas in a non-Abelian theory (was F-010)
- Location: thm-spontaneous-gauge-locking Steps 9-10 (399-415)
- Severity: Moderate
- Type: Computational error (secondary: Conceptual)
- Criterion: External and Framework
- Origin: this chapter
- Claim (verbatim, lines 399-411): "For simply-connected $\mathcal{Z}_{\text{shared}}$, a flat connection is pure gauge: $A_\mu^{(A)}(z) - A_\mu^{(B)}(z) = \partial_\mu \chi(z)$ for some $\chi: \mathcal{Z} \to \mathfrak{g}$." "The gauge transformation $U_A \to U_A e^{-i\chi}$ absorbs the gradient term, yielding: $A_\mu^{(A)}(z) = A_\mu^{(B)}(z)$".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1458-1464` (prop-gauge-transformation-connection): "$A'_\mu = U A_\mu U^{-1} - \frac{i}{g}(\partial_\mu U)U^{-1}$"; `intro_agent.md:42` (non-Abelian $G_{\text{Fragile}}$).
- Why this is an error: For a non-Abelian group a flat connection is $-\frac{i}{g}(\partial U)U^{-1}$, not a gradient $\partial_\mu\chi$, and the difference of two connections is not a connection, so "flat implies the difference is pure gauge" is not meaningful; the correct statement is gauge equivalence $U^{-1}A^{(A)}U + \frac{i}{g}U^{-1}\partial U = A^{(B)}$. Under $U = e^{-i\chi}$ the connection changes by $UAU^{-1} - \frac{i}{g}(\partial U)U^{-1} \ne A - \partial\chi$ unless $\mathfrak{g}$ is Abelian and $g = 1$. $A^{(A)}$ is a form on $\mathcal{Z}_A$ and $A^{(B)}$ on $\mathcal{Z}_B$; their difference at "the same $z$" needs $\phi_{A \to B}$, never invoked.
- Impact on downstream results: Step 12 and the "residual global gauge freedom" interpretation.
- Fix guidance: replace Steps 9-10 by: "on a simply-connected domain a flat connection is gauge-trivial, $\mathcal{A} = -\frac{i}{g}(\partial U)U^{-1}$; with $\mathcal{C}_{AB} := \phi^*A^{(B)} - A^{(A)}$ (E-002), flatness of the relative connection means $\phi^*A^{(B)}$ and $A^{(A)}$ are gauge equivalent via a single $U_{AB}(z)$, and the residual freedom is global."
- Required new assumptions/permits: simple connectivity of the chosen pre-locking domain (E-004).
- Validation plan: verify the transformation law against prop-gauge-transformation-connection symbolically for $SU(2)$.

### [E-012] Vacuum expectation value off by $\sqrt2$ (was F-011)
- Location: thm-spontaneous-gauge-locking Step 11 (419-425); Mexican-hat admonition (312)
- Severity: Moderate
- Type: Computational error
- Criterion: External (Framework: contradicts the chapter's cited analogue)
- Origin: this chapter
- Claim (verbatim, line 422): "$v_{\text{lock}} = \sqrt{(\beta - \beta_c)/g_{\text{lock}}}$" for "$\mathcal{V}_{\text{lock}}(\phi_{AB}) = -\mu_{\text{lock}}^2 |\phi_{AB}|^2 + g_{\text{lock}} |\phi_{AB}|^4$, $\mu^2_{\text{lock}} = \beta - \beta_c$" (297-302).
- Upstream anchor: `08_multiagent/02_standard_model.md:1366` (cor-ontological-ssb): "$v = \langle|\phi|\rangle = \sqrt{\frac{\mu^2}{2\lambda}}$" for the same potential shape.
- Why this is an error: $\partial_{|\phi|}\mathcal{V} = -2\mu^2|\phi| + 4g|\phi|^3 = 0 \Rightarrow |\phi|^2 = \mu^2/(2g)$ (checked symbolically), so $v_{\text{lock}} = \sqrt{(\beta - \beta_c)/(2g_{\text{lock}})}$. The chapter states it is "analogous to Corollary cor-ontological-ssb", which has the correct factor.
- Impact on downstream results: Step 11 and the Kuramoto order-parameter analogy.
- Fix guidance: write $v_{\text{lock}} = \sqrt{(\beta - \beta_c)/(2\lambda_4)}$ with the quartic coefficient renamed per E-003.
- Required new assumptions/permits: none.
- Validation plan: differentiate the potential.

### [E-013] Critical coupling formula has no derivation and fails dimensional analysis (was F-013)
- Location: cor-critical-coupling-locking (449-463)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 455, 461): "$\beta_c = \frac{\sigma^2 \text{Vol}(\mathcal{Z}_{\text{shared}})}{2 g_{\text{lock}}^2}$ ... *Proof.* Balance the kinetic (diffusion) term $\sigma^2 |\nabla \psi|^2$ against the synchronization potential $\beta \Psi_{\text{sync}}$. The transition occurs when coupling energy equals the thermal fluctuation scale."
- Upstream anchor: `08_multiagent/01_gauge_theory.md:2956-2975` (def-cognitive-action-scale): "$\sigma := T_c \cdot \tau_{\text{update}}$ ... $[\sigma] = \text{nat} \cdot \text{step}$".
- Why this is an error: No term $\sigma^2|\nabla\psi|^2$ appears in $\mathcal{L}_{\text{joint}}$ or $\mathcal{A}_{\text{joint}}$ (E-010), no balance equation is written, and neither the factor 2 nor the $\text{Vol}$ or $g_{\text{lock}}^{-2}$ dependence is obtained from anything. Units: from (327) $[\beta] = [x]^2/[\Psi_{\text{sync}}]$, while the RHS has $(\text{nat} \cdot \text{step})^2 [z]^D / [g_{\text{lock}}]^2$, which cannot match for a $D$-independent $[g_{\text{lock}}]$. The interpretation at 474 reads $g_{\text{lock}}$ as the gauge coupling while Step 11 uses it as the quartic coefficient (E-003).
- Impact on downstream results: Kuramoto table ($K_c \leftrightarrow \beta_c$), cor-critical-mass-consensus.
- Fix guidance: derive $\beta_c$ from an explicit fluctuation term added to the objective via a mean-field computation (E-010), or demote the corollary to a scaling heuristic with consistent units.
- Required new assumptions/permits: as in E-010.
- Validation plan: units check; limiting cases $\sigma \to 0$ ($\beta_c \to 0$) and $\text{Vol} \to \infty$.

### [E-014] Translation Operator: the message factor is a scalar phase and Property 3 is false (was F-014)
- Location: def-translation-operator (558-578)
- Severity: Moderate
- Type: Definition mismatch (secondary: Computational error)
- Criterion: External and Framework
- Origin: this chapter
- Claim (verbatim, lines 564, 576): "$\mathcal{T}_{A \to B}(m) := \exp\left(-ig \int_{\gamma_{AB}} m^a A_\mu^a \, dz^\mu\right) \cdot \mathcal{P}\exp\left(-ig \int_{\gamma_{AB}} A_\mu \, dz^\mu\right)$" and "**Identity at Locking:** When $A^{(A)} = A^{(B)}$, reduces to pure message action".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:1436-1445` ($A = A_\mu^a T_a\,dz^\mu$, $\mathfrak{g}$-valued); def-message-lie-algebra (503-512, this chapter): "'Understanding' a message means successfully applying $e^{im}$".
- Why this is an error: (i) $m^a A_\mu^a\,dz^\mu$ contracts the Lie-algebra index and the form index and is a real scalar; its exponential is a $U(1)$ phase carrying no group-valued information about $m$, not the message action $e^{im}$. (ii) When $A^{(A)} = A^{(B)} = A \ne 0$ the Wilson line is not the identity, so $\mathcal{T}$ does not reduce to a pure message action; it does so only when the relative connection vanishes along $\gamma$. (iii) The endpoints of $\gamma_{AB}$ lie in different manifolds. (iv) Property 1 is the transformation law of a Wilson line, not of a product with a scalar phase; Property 2 fails for the message factor unless messages add.
- Impact on downstream results: thm-untranslatability-bound; the code comment "Implements $T_{A \to B}$".
- Fix guidance: define $\mathcal{T}_{A \to B}(m) := e^{im}\,W_\gamma$ with $W_\gamma := \mathcal{P}\exp(-ig\int_\gamma \mathcal{A})$ the Wilson line of the relative connection along a path in the graph of $\phi$; state "identity at locking" as $W_\gamma = \mathbb{1}$ when the relative connection vanishes.
- Required new assumptions/permits: the relative connection of E-002.
- Validation plan: check the three properties for the revised operator with $SU(2)$ test matrices.

### [E-015] Untranslatability Bound: $\mathcal{U}_{AB}(m)$ is undefined and the bound is ill-formed (was F-015)
- Location: thm-untranslatability-bound (606-647)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Computational error)
- Criterion: External and Framework
- Origin: this chapter
- Claim (verbatim, lines 612, 630): "$\mathcal{U}_{AB}(m) \leq \|m\| \cdot \oint_{\partial\Sigma} \|\mathcal{F}_{AB}\|_F \, dA$ where $\Sigma$ is any surface bounded by the communication path." "$\mathcal{H}_\gamma = \exp\left(-ig \int_\Sigma \mathcal{F}_{\mu\nu} \, dS^{\mu\nu}\right) + O(\mathcal{F}^2)$".
- Upstream anchor: none; $\mathcal{U}_{AB}$ first appears at line 609 without definition.
- Why this is an error: (i) The theorem bounds a quantity with no definition anywhere in the book; Step 4 suggests $\|m_{\text{received}} - m_{\text{sent}}\|$, but "received" is not defined (adjoint action $\mathcal{H} m \mathcal{H}^{-1}$?). (ii) Non-Abelian Stokes and the estimates of Step 5 give $\|\mathcal{H}_\gamma - \mathbb{1}\| \le g\int_\Sigma \|\mathcal{F}\|\,dS + O(\mathcal{F}^2)$, a surface integral with a factor $g$. The statement integrates over the 1-dimensional loop $\partial\Sigma$ against a 2-dimensional area element, which is not a well-defined integral, and drops $g$. (iii) If $m_{\text{received}} = \mathcal{H} m \mathcal{H}^{-1}$ then $\|\mathcal{H} m \mathcal{H}^{-1} - m\| \le 2\|m\|\,\|\mathcal{H} - \mathbb{1}\|$ in operator norm, so the constant of Step 4 is off by 2 unless a norm is fixed.
- Impact on downstream results: cor-perfect-translation.
- Fix guidance: define $\mathcal{U}_{AB}(m) := \|\mathcal{H}_\gamma m \mathcal{H}_\gamma^{-1} - m\|$ and state $\mathcal{U}_{AB}(m) \le 2g\|m\|\int_\Sigma \|\mathcal{F}_{AB}\|\,dS + O(\mathcal{F}^2)$.
- Required new assumptions/permits: small-curvature regime for the $O(\mathcal{F}^2)$ remainder.
- Validation plan: numerical check with an $SU(2)$ connection on a small square loop.

### [E-016] Babel Limit multiplies the entropy of the metric tensor by the gauge-algebra dimension and mixes rates with totals (was F-016)
- Location: thm-babel-limit (697-736); cor-ineffability-theorem (750-764)
- Severity: Major
- Type: Conceptual (secondary: Dimensional mismatch, External dependency)
- Criterion: External and Framework
- Origin: this chapter
- Claim (verbatim, lines 700-703, 718, 723, 730): "let $H(G_A)$ be the differential entropy rate of Agent $A$'s metric tensor. Complete gauge locking is achievable only if: $\dim(\mathfrak{g}) \cdot H(G_A) \leq C_{\mathcal{L}}$"; "The information required to specify the metric tensor $G_A$ at rate $r$ is $r \cdot H(G_A)$ nats per unit time."; "$I_{\text{required}} = \dim(\mathfrak{g}) \cdot H(G_A)$"; "$d_{\text{unlocked}} = \dim(\mathfrak{g}) - \lfloor C_{\mathcal{L}} / H(G_A) \rfloor$".
- Upstream anchor: `06_fields/03_info_bound.md:255-262` ($I_{\max}$ is total stored nats); `08_multiagent/01_gauge_theory.md:1436-1445` (gauge parameters are the $\dim\mathfrak{g}$ coefficients $A^a_\mu$, unrelated to the $D(D+1)/2$ components of $G$); `01_gauge_theory.md:1632` (the metric is built from gauge-invariant scalar quantities).
- Why this is an error: (i) Gauge locking as defined in this chapter is a condition on $\Delta U \in G_{\text{Fragile}}$, $\dim\mathfrak{g}$ parameters per point; the metric has $D(D+1)/2$ gauge-invariant components. Multiplying the entropy of $G_A$ by $\dim\mathfrak{g}$ counts neither object. (ii) Step 3 uses $r \cdot H(G_A)$ (so $H$ is nats per sample) and Step 4 uses $\dim\mathfrak{g} \cdot H(G_A)$ (so $H$ is nats per unit time); inconsistent, and neither is compared with a correctly dimensioned $C_{\mathcal{L}}$ (E-008). (iii) Differential entropy can be negative and is not the number of nats needed to specify a real parameter (that is infinite); the correct framework is rate–distortion. With $H(G_A) \le 0$ the floor in Step 5 and in cor-ineffability-theorem is negative or undefined. (iv) "Impossible by Shannon's theorem" needs a source with rate $R > C$; the source and its rate are never defined.
- Impact on downstream results: cor-ineffability-theorem, thm-spectral-locking-order (same undefined entropy), `check_babel_limit` (E-025), `09_economics/01_pomw.md:1335-1360` (thm-corruption-babel-detection applies the inequality verbatim).
- Fix guidance:
  1. State the theorem in rate–distortion form: let $R_{\Delta U}(\epsilon)$ be the rate (nats/step) needed to convey $\Delta U(z)$ to distortion $\epsilon$; for $\epsilon \to 0$ this is $\dim\mathfrak{g} \cdot \log(1/\epsilon) + \text{const}$, giving the $\dim\mathfrak{g}$ scaling honestly.
  2. $\epsilon$-locking is achievable only if $R_{\Delta U}(\epsilon) \le C_{\mathcal{L}}$.
  3. Derive $d_{\text{unlocked}}$ from reverse water-filling rather than a floor of an entropy ratio.
  4. Propagate the new statement to `01_pomw.md:1335-1360`.
- Required new assumptions/permits: a distortion measure on $G_{\text{Fragile}}$ and a source model for $\Delta U$.
- Validation plan: check the bound for a Gaussian source in one gauge direction against the known $R(\epsilon) = \frac12\log(\sigma^2/\epsilon)$.

### [E-017] Metric eigenvalues called principal curvatures; symbol clash (was F-017)
- Location: def-metric-eigendecomposition (796-811)
- Severity: Minor
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: External and Framework
- Origin: this chapter
- Claim (verbatim, line 806): "$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_D > 0$ are eigenvalues (principal curvatures)".
- Upstream anchor: `08_multiagent/01_gauge_theory.md:2956` (def-cognitive-action-scale) uses $\sigma$, as do lines 455 and 963 of this chapter; curvature in `05_geometry/01_metric_law.md` is $R_{ij}$, built from second derivatives of $G$.
- Why this is an error: Eigenvalues of the metric in a chart are coordinate-dependent scale factors, not principal curvatures (eigenvalues of a shape operator or of the Riemann tensor). The mislabel matters because thm-spectral-locking-order then equates high eigenvalue with "high-curvature (salient)" features (823).
- Impact on downstream results: interpretation of thm-spectral-locking-order.
- Fix guidance: call them "metric eigenvalues (Fisher-information scale factors)" and rename to $\gamma_k$.
- Required new assumptions/permits: none.
- Validation plan: none needed.

### [E-018] Objective-reality equivalence relation uses a coordinate-dependent clause (was F-018)
- Location: thm-emergence-objective-reality Steps 2-4 (884-895); def-metric-friction (77-84)
- Severity: Minor
- Type: Definition mismatch (secondary: Scope restriction)
- Criterion: External
- Origin: this chapter
- Claim (verbatim, line 893): "Define the equivalence relation: $z_A \sim z_B$ iff $\phi_{A \to B}(z_A) = z_B$ and $G_A(z_A) = G_B(z_B)$."
- Upstream anchor: none (def-metric-friction, line 77: "$\phi_{A \to B}$ ... the best-fit map").
- Why this is an error: The isometry condition already established in Step 2 is $G_A = \phi^* G_B$; the extra clause compares matrix components in two unrelated charts and is neither implied by nor generally compatible with the pullback condition. For the quotient to be a manifold with a well-defined metric, $\phi$ must be a diffeomorphism, never assumed. The notation $\phi^* G_B(\phi(z))$ double-evaluates the pullback, and the Frobenius norm of a $(0,2)$-tensor is not coordinate-invariant, so the magnitude of $\mathcal{F}_{AB}$ (and the Node 69 threshold) depends on the chart.
- Impact on downstream results: Node 69 threshold; the "hallucination shared by $N$ agents" interpretation.
- Fix guidance: assume $\phi$ is a diffeomorphism; drop the second clause; write $\mathcal{F}_{AB}(z) := \|G_A(z) - (\phi^* G_B)(z)\|^2_{G_A}$ with the $G_A$-induced tensor norm.
- Required new assumptions/permits: $\dim\mathcal{Z}_A = \dim\mathcal{Z}_B$ and $\phi$ a diffeomorphism.
- Validation plan: verify chart-independence of the revised friction under a coordinate change.

### [E-019] Environment friction $\mathcal{F}_{AE}$ is undefined; Node 70 proxy stated inconsistently (was F-019)
- Location: rem-echo-chamber-effect (923-939); Node 70 table (1371-1377)
- Severity: Minor
- Type: Proof gap / omission (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 935-937, 1373): "$\mathcal{F}_{iE}$ measures the friction between agent $i$ and the environment's causal structure. *Diagnostic:* Node 70 (BabelCheck) monitors $\partial \mathcal{F}_{AE}/\partial t$" versus the table "Proxy: $\partial \mathcal{F}_{AB} / \partial t$".
- Upstream anchor: `01_foundations/01_definitions.md` (the environment is a POMDP with hidden state and no metric tensor); def-metric-friction requires metrics on both sides.
- Why this is an error: The environment carries no $G_E$ in the framework, so $\mathcal{F}_{AE}$ is not an instance of def-metric-friction and the loss at 931 is not computable as written. The remark and the table disagree on what Node 70 monitors.
- Impact on downstream results: Node 70; the grounding recommendation.
- Fix guidance: define $\mathcal{F}_{iE}$ operationally (e.g. the prediction-error term already in $\mathcal{L}_{\text{joint}}$, or friction to the Fisher metric of $p(x'|x,a)$) and list both quantities in the Node 70 proxy column.
- Required new assumptions/permits: none beyond the chosen definition.
- Validation plan: none needed.

### [E-020] Critical Mass for Consensus: no derivation and not dimensionless (was F-020)
- Location: cor-critical-mass-consensus (957-971)
- Severity: Moderate
- Type: Proof gap / omission (secondary: Dimensional mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 963): "$N > N_c = \frac{\sigma^2}{\lambda_{\text{lock}} \cdot \langle \mathcal{F}_{ij} \rangle}$ where $\langle \mathcal{F}_{ij} \rangle$ is the average pairwise friction."
- Upstream anchor: `08_multiagent/01_gauge_theory.md:2975` ("$[\sigma] = \text{nat} \cdot \text{step}$"); def-metric-friction line 88 ("$[\mathcal{F}_{AB}] = [z]^{-4}$"); def-inter-agent-connection line 154 ($\lambda_{\text{lock}}$ dimensionless as a coefficient of a connection).
- Why this is an error: Labelled a corollary but follows from no preceding statement, with no proof. $N_c$ must be dimensionless, yet the RHS has units $(\text{nat} \cdot \text{step})^2 [z]^4 / [\lambda_{\text{lock}}]$. The verifier notes that the reviewer's further point, that the dependence on $\langle\mathcal{F}_{ij}\rangle$ is inverted relative to Kuramoto (where the threshold grows with heterogeneity), is plausible but not established: in the chapter's own loss (931) the restoring force scales with $\lambda_{\text{lock}}\mathcal{F}_{AB}$, so a coupling proportional to friction is a possible reading. This is recorded as an open question, not as part of the error.
- Impact on downstream results: prose on tribes versus civilisations; nothing formal.
- Fix guidance: derive $N_c$ from a mean-field (Kuramoto or Curie–Weiss) treatment of the locking dynamics, or remove the corollary.
- Required new assumptions/permits: a population model for pairwise interactions.
- Validation plan: units check and monotonicity check in $\sigma$, $\lambda_{\text{lock}}$, $\langle\mathcal{F}_{ij}\rangle$.

### [E-021] Kuramoto analogue is gradient ascent on friction (was F-021)
- Location: Physics Isomorphism: Kuramoto Model (1107-1112)
- Severity: Minor
- Type: Computational error (sign)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, line 1110): "$\frac{d\theta^{(i)}}{dt} = \omega^{(i)} + \beta \sum_{j \neq i} \nabla_\theta \mathcal{F}_{ij}$".
- Upstream anchor: none (locking minimises friction, lines 438-440 of this chapter).
- Why this is an error: Following $+\nabla_\theta\mathcal{F}_{ij}$ increases friction. The Kuramoto coupling $\frac{K}{N}\sum\sin(\theta_j - \theta_i) = -\frac{K}{N}\partial_{\theta_i}\sum_j(1 - \cos(\theta_j - \theta_i))$ is descent, so the analogue is $-\beta\sum_j \nabla_{\theta^{(i)}}\mathcal{F}_{ij}$.
- Impact on downstream results: none beyond the table.
- Fix guidance: insert the minus sign.
- Required new assumptions/permits: none.
- Validation plan: none needed.

### [E-022] Stale theorem numbers in docstrings (was F-024)
- Location: Implementation docstrings (1156, 1160, 1182, 1188, 1208, 1285, 1294, 1305)
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Implements the Locking Operator L_sync (Theorem 37.1)" ... "Locking Curvature (Definition 37.3)" ... "(Definition 37.4: Translation Operator)" ... "(Definition 37.5: Language Channel)" ... "Babel Limit is satisfied (Theorem 37.2)".
- Upstream anchor: none.
- Why this is an error: The chapter has no numbered environments. By order of appearance the definitions are 1 metric friction, 2 inter-agent connection, 3 locking curvature, 4 order parameter, 5 message, 6 language channel, 7 translation operator; theorems 1 locking operator, 2 spontaneous gauge locking, 3 untranslatability, 4 Babel limit. So "37.4: Translation Operator", "37.5: Language Channel" and "Theorem 37.2 (Babel Limit)" are wrong even under a sequential scheme.
- Impact on downstream results: none.
- Fix guidance: replace by label names (`thm-locking-operator-derivation`, `def-locking-curvature`, `def-translation-operator`, `def-language-channel`, `thm-babel-limit`).
- Required new assumptions/permits: none.
- Validation plan: none needed.

### [E-023] Procrustes alignment computes the transposed rotation (was F-022)
- Location: `compute_metric_friction` (1218-1224)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: External
- Origin: this chapter
- Claim (verbatim, lines 1220-1223): "# Solve: min_R ||z_a - z_b @ R||_F^2 s.t. R^T R = I / U, _, Vt = torch.linalg.svd(z_a.T @ z_b) / R = U @ Vt / z_b_aligned = z_b @ R".
- Upstream anchor: none.
- Why this is an error: The orthogonal Procrustes minimiser of $\|z_a - z_b R\|_F$ is $R = UV^\top$ with $z_b^\top z_a = U\Sigma V^\top$. The code decomposes $z_a^\top z_b = (z_b^\top z_a)^\top = V\Sigma U^\top$, so its `U @ Vt` equals $VU^\top = R^\top = R^{-1}$. Verified numerically (torch, float64, $B = 64$, $D = 5$, $z_b = z_a Q^\top$ with random orthogonal $Q$): the chapter's residual $\|z_a - z_b R\|_F = 21.61$ (MSE 1.46); `svd(z_b.T @ z_a)` gives $1.2 \times 10^{-14}$ and returns exactly $Q$; on a generic random $z_b$ the chapter's residual (24.62) exceeds the true minimum (22.58). So the reported friction is not the minimal distortion and is nonzero for isometric point clouds, contradicting def-metric-friction ($\mathcal{F}_{AB} = 0$ iff aligned) and making Node 69 fire spuriously.
- Impact on downstream results: `forward`, Node 69 proxy, any training on `loss`.
- Fix guidance: `U, _, Vt = torch.linalg.svd(z_b.T @ z_a); R = U @ Vt` (or keep the current SVD and use `R = (U @ Vt).T`).
- Required new assumptions/permits: none.
- Validation plan: unit test with `z_b = z_a @ Q.T` asserting friction below $10^{-10}$.

### [E-024] Procrustes step makes the learnable gauge transform inoperative; unit label (was F-025)
- Location: `forward` and `compute_metric_friction` (1216, 1272-1297)
- Severity: Note
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 1289-1292, 1216): "z_b_aligned = self.gauge_transform(agent_b_view) ... friction = self.compute_metric_friction(agent_a_view, z_b_aligned)" and "Returns: Scalar friction loss (nats)".
- Upstream anchor: none.
- Why this is an error: With `use_procrustes=True` the friction is $\min_R \|z_a - z_b W R\|_F^2$; for orthogonal $W$ the minimum over $R$ is independent of $W$ (since $WR$ ranges over the full orthogonal group), so the friction term gives `gauge_transform` no gradient along the orthogonal group and the module does not implement $\mathcal{T}_{A \to B}$ in an operative way. The returned MSE of latent coordinates has units $[z]^2$, not nats. (The verifier did not retain the reviewer's further claim that the loss rewards $\|W\| \to 0$; the MSE-optimal scale is positive when the clouds are correlated.)
- Impact on downstream results: none formal.
- Fix guidance: either drop the Procrustes step when `gauge_transform` is trained (with an orthogonality penalty), or drop `gauge_transform`; label the loss units as $[z]^2$.
- Required new assumptions/permits: none.
- Validation plan: check that the gradient of `loss` with respect to `gauge_transform.weight` is normal to the orthogonal group at initialisation.

### [E-025] `check_babel_limit` returns "satisfied" independent of the metric when the mean log-eigenvalue is non-positive (was F-023)
- Location: `check_babel_limit` (1299-1326)
- Severity: Moderate
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim, lines 1319-1324): "H_per_component = torch.log(eigenvalues + 1e-6).mean().item() / k_max = int(channel_capacity / max(H_per_component, 1e-6)) / k_max = min(k_max, self.latent_dim) / satisfied = (k_max >= self.gauge_dim)".
- Upstream anchor: thm-babel-limit (703): "$\dim(\mathfrak{g}) \cdot H(G_A) \le C_{\mathcal{L}}$"; thm-spectral-locking-order (819): "$k_{\max} = \max\{k : \sum_{j=1}^k H(\sigma_j v_j) \le C_{\mathcal{L}} \cdot T\}$".
- Why this is an error: Verified numerically: eigenvalues $(0.5, 0.8, 1.2, 2.0)$ give mean log-eigenvalue $-0.0102$; `max(H, 1e-6)` then yields `k_max = capacity / 1e-6`, clipped to `latent_dim`, and with `channel_capacity = 10`, `latent_dim = gauge_dim = 4` the function returns `(True, 4)`. Whenever the mean log-eigenvalue is $\le 0$ the check therefore passes for any capacity $\ge 10^{-6} \cdot$`gauge_dim`, i.e. for any realistic capacity, independent of the metric. Neither formula of the text is implemented (the code uses a mean of log-eigenvalues, not an entropy or a cumulative sum), `eigenvalues.flip(0)` is dead code, and `k_max` is capped by `latent_dim` but compared with `gauge_dim`, so with `gauge_dim > latent_dim` (e.g. 8 versus 4) the check can never pass (verified: returns `(False, 4)`).
- Impact on downstream results: Node 70 / Babel diagnostics if this module is used.
- Fix guidance: implement the cumulative rule of thm-spectral-locking-order with a well-defined non-negative per-component information (e.g. $\frac12\log(1 + \sigma_k/\sigma_{\text{noise}})$), return `k_max = ` number of components whose cumulative sum is $\le$ capacity, and set `satisfied = (k_max == latent_dim)`, or compare with $\dim\mathfrak{g}$ only after E-016 is resolved.
- Required new assumptions/permits: a noise scale $\sigma_{\text{noise}}$.
- Validation plan: unit tests with capacity 0 (must return `False, 0`) and capacity $\to \infty$ (must return `True, latent_dim`).

### [E-026] Duplicate labels `node-69`, `node-70` and inconsistent Sieve numbering (was F-026)
- Location: Diagnostic Nodes 69-70 (1352-1353, 1368-1369); line 937 "Node 70 (BabelCheck)"
- Severity: Minor
- Type: Citation / reference error (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter and `08_multiagent/05_architecture.md`
- Claim (verbatim): "(node-69)= **Node 69: MetricAlignmentCheck**" ... "(node-70)= **Node 70: BabelCheck**".
- Upstream anchor: `08_multiagent/05_architecture.md:2111-2112` "(node-69)= **Node 69: ChiralityViolationCheck**"; `:2125-2126` "(node-70)= **Node 70: TextureLeakageCheck**"; `07_cognition/07_metabolic_transducer.md:1191` "**Node 69: ThermalRunawayCheck**", `:1214` "**Node 70: MetricFadingCheck**"; `07_cognition/09_retrieval_attention.md:1273` "**Node 71: CausalMaskCheck**". Confirmed by the mechanical cross-reference report.
- Why this is an error: The MyST labels are defined in two files, so every `{ref}` to them is ambiguous. Independently, "Node 69" names three different diagnostics and "Node 70" three others; "Node 70 (BabelCheck)" at line 937 does not identify a unique node. The verifier downgraded this from Moderate to Minor because no mathematical content is affected and the fix is mechanical.
- Impact on downstream results: any cross-reference to `node-69`/`node-70`; the global node index.
- Fix guidance: give this chapter unique labels (e.g. `node-69-consensus`, `node-70-consensus`) and renumber so each diagnostic has a unique number in the global Sieve table.
- Required new assumptions/permits: none.
- Validation plan: the duplicate-label section of the cross-reference report must be empty for these labels.

### [E-027] Losses appendix defines $\Psi_{\text{sync}}$ with a fourth, inconsistent domain (added by verifier, V-001)
- Location: `10_appendices/06_losses.md`, def-f-sync-potential (480-497)
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: upstream `10_appendices/06_losses.md`
- Claim (verbatim, `06_losses.md:485`): "$\mathcal{L}_{\text{sync}} = \beta \Psi_{\text{sync}} = \beta \int_{\partial\Omega} \mathcal{F}_{AB}^{\mu\nu} \mathcal{F}_{AB\,\mu\nu} \, dA$"; `:491` "Drives gauge locking (synchronized metrics)."
- Upstream anchor: this chapter, lines 187, 211, 368 (volume integrals over $\mathcal{Z}_{\text{shared}}$ with three different measures).
- Why this is an error: The appendix writes the synchronisation potential as a boundary integral over an unspecified $\Omega$ with an area element, whereas the chapter uses a volume integral; the two cannot both be the definition. The appendix also restates the E-007 conflation ("synchronized metrics").
- Impact on downstream results: any implementation reading the appendix as the canonical loss table.
- Fix guidance: once E-004 fixes the domain and measure, copy the same expression into the appendix and change "synchronized metrics" to "aligned gauges".
- Required new assumptions/permits: none.
- Validation plan: textual comparison of the two definitions after the edit.

## Scope restrictions and clarifications
- Every formal result of the chapter that mentions metric alignment, isometry, Gromov–Hausdorff distance or "objective reality" is currently supported only for alignment of the nuisance-bundle gauge; metric convergence requires an additional coupling term (E-007).
- The phase-transition language (finite $\beta_c$, Landau potential, Kuramoto $K_c$) is a postulate, not a derived consequence of either stated objective (E-010, E-013, E-020).
- The Babel Limit and its corollaries should be read as heuristics until restated in rate–distortion form (E-008, E-016).
- The implementation block is illustrative; its Procrustes step and Babel check must be corrected before use as a diagnostic (E-023, E-025).

## Open questions
- Is $\mathcal{C}_{AB}$ intended to be the relative connection $\phi^* A^{(B)} - A^{(A)}$, or a genuinely new field sourced by messages? The answer determines whether E-002 and E-011 have a common fix.
- Are $\beta$ and $\lambda_{\text{lock}}$ the same parameter? If the joint loss carries both a curvature term and a metric-friction term, the chapter should say so explicitly and the appendix loss table should list both.
- In cor-critical-mass-consensus, should $N_c$ increase or decrease with average pairwise friction? A mean-field derivation would settle the sign.
- The dimension statement in Step 5 of thm-locking-operator-derivation ($[\mathfrak{L}_{\text{sync}}] = \text{nat}$ from the prefactor $1/(4g_{\text{lock}}^2)$ on a $D$-dimensional Riemannian integral) implies $[g^2_{\text{lock}}] = [\text{length}]^{D-4}/\text{nat}$, which differs from the $[g^2] = [\text{length}]^{d-4}$, $d = D+1$, of def-yang-mills-action; the author may wish to reconcile the two.

## Rejected candidate findings
None. All 26 stage-1 findings were confirmed; four were adjusted (F-020 partial demotion of the sign clause, F-023 wording, F-025 one sub-claim dropped, F-026 severity Moderate to Minor).
