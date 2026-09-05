# Mathematical Review: docs/source/1_agent/07_cognition/01_supervised_topo.md

## Metadata
- Reviewed file: docs/source/1_agent/07_cognition/01_supervised_topo.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (1205 lines), plus the appendix proof A.5 that the chapter delegates to
- Framework anchors (definitions/axioms/permits):
  - `01_foundations/02_control_loop.md:139-147` (discrete macro index $K$; covariant gradient $\nabla_A V := G^{-1}(dV - A)$)
  - `03_architecture/01_compute_tiers.md:1298` (`def-attentive-routing-law`, soft router weights $w_i(x)$); `:2354-2480` (`def-factorized-jump-operator`, transition maps $L_{i\to j}$)
  - `05_geometry/02_wfr_geometry.md:106-127` (`def-the-wfr-action`); `:374-381` (`thm-classical-master-equation-wfr`)
  - `05_geometry/03_holographic_gen.md:236-243` (`prop-so-d-symmetry-at-origin`); `:541-560` (`ax-bulk-boundary-decoupling`)
  - `05_geometry/04_equations_motion.md:418-427` (`def-effective-potential`, $V_{\text{critic}}$); `:513` (free energy $\Phi = E - T_c S$); `:641-683` (Möbius addition code); `:905-916` (`thm-overdamped-limit`)
  - `06_fields/01_boundary_interface.md:606-676` (`sec-the-context-space-unified-definition`, `def-context-instantiation-functor`, `thm-universal-context-structure`)
  - `07_cognition/03_memory_retrieval.md:696` (`sec-the-retrieval-texture-firewall`)
  - `10_appendices/01_derivations.md:417-527` (A.5, `thm-classification-as-relaxation-a` and proof)
  - `10_appendices/02_parameters.md:70,82,92`; `10_appendices/06_losses.md:201-241`
  - Node registries: `05_geometry/01_metric_law.md:325`, `06_fields/03_info_bound.md:683-684`, `08_multiagent/04_dnn_blocks.md:3163`, `08_multiagent/03_parameter_sieve.md:595`

## Executive summary
- Critical: 0
- Major: 4
- Moderate: 5
- Minor: 7
- Notes: 0
- Primary themes:
  1. The class-conditioned potential $V_y(z,K)$ depends on $z$ only through the hard chart index $K(z)$, so the "class-conditioned gradient flow" that the region-of-attraction definition, the relaxation theorem, the Langevin proposition and the symmetry-breaking corollary rely on has no label-dependent drift (E-003, E-014).
  2. The relaxation theorem asserts almost-sure convergence at positive temperature with finite barriers, which contradicts the escape rate its own appendix proof invokes; the Lyapunov sketch also applies the chain rule to the wrong gradient outside the conservative case (E-007, E-008).
  3. The "Effective Disconnection" proposition is false for the WFR distance it cites: that distance has a free reaction control, is bounded by a pure-reaction (Hellinger) path, and does not depend on $\gamma_{\text{sep}}$ (E-010).
  4. The implemented losses do not match the stated ones: the purity term is the entropy of the learnable $\Theta$ rows and never sees the labels, so the identity $\mathcal{L}_{\text{purity}} = H(Y \mid K)$ and the mutual-information duality fail for what is implemented; the metric term relies on an undefined $d_{\text{jump}}$ and its margin is inert at the default value (E-011, E-012).
  5. Several formal statements are internally inconsistent or ill-typed: transition charts cannot lie in two sub-atlases when $\epsilon_{\text{purity}} < 0.5$; a latent point is compared with a set of chart indices; the hierarchy surjections point the wrong way; Node 40 carries three unrelated identities; several cross-references point at the wrong object (E-002, E-005, E-015, E-016, E-001, E-013).

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Relationship to the Context-Conditioned Framework, lines 43, 64, 110 | Minor | Citation / reference error | Framework | this chapter | Section cites itself as the prior result; $\Phi_{\text{eff}} = -\log p(y\mid z)$ attributed to a theorem that does not contain it |
| E-002 | `prop-soft-injectivity`, lines 99-112 | Moderate | Definition mismatch (Invalid inference) | Framework | this chapter | $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ whenever $\epsilon_{\text{purity}} < 0.5$; "transition region" conditions describe charts in no sub-atlas; $H_{\text{transition}}$ undefined |
| E-003 | `def-class-conditioned-potential`, `def-region-of-attraction`, `thm-classification-as-relaxation`, lines 138-257 | Major | Conceptual (Proof gap) | Framework | this chapter | Semantic term is constant on each chart, so the class-conditioned flow equals the unconditioned flow |
| E-004 | "Why Logarithms?" box, line 192 | Minor | Conceptual | External | this chapter | $-\log P$ called "exactly the free energy"; it is $\beta(E - F)$, the surprisal |
| E-005 | `def-region-of-attraction`, lines 200-203 | Minor | Typo (Definition mismatch) | Framework | this chapter | $\lim \phi_t(z) \in \mathcal{A}_y$ compares a latent point with a set of chart indices; limit assumed to exist |
| E-006 | lines 207, 210, 233, 896 | Minor | Notation conflict | Framework | this chapter | $\nabla_A$ redefined without the metric, conflicting with the book convention; $\nabla_G$ undefined |
| E-007 | `thm-classification-as-relaxation`, lines 227-257 | Major | Invalid inference (Proof gap) | Framework (External secondary) | this chapter; mirrored in `10_appendices/01_derivations.md:419-527` | Almost-sure convergence at fixed $T_c > 0$ with finite barriers is false; "global minimum by construction" lacks a hypothesis on $\beta_{\text{class}}$; hypothesis 1 is tautological at $T_c \to 0$ |
| E-008 | proof sketch, lines 248-255 | Moderate | Computational error (Scope restriction) | Framework | this chapter | $\dot L = \nabla V_y \cdot \dot z$, not $\nabla_A V_y \cdot \dot z$; descent shown only for $A = 0$ |
| E-009 | `cor-inference-via-relaxation` Remark and "Two Paths" box, lines 273-307 | Moderate | Invalid inference | Framework | this chapter | Fast path is not the $T_c \to 0$, $s \to \infty$ limit of the relaxation path |
| E-010 | `prop-effective-disconnection`, lines 396-415; Node 41 proxy, line 1183 | Major | Invalid inference (Definition mismatch) | Framework | this chapter | $d_{\text{WFR}}$ of `def-the-wfr-action` is independent of $\gamma_{\text{sep}}$ and uniformly bounded; LHS ill-typed |
| E-011 | `def-purity-loss`, `prop-purity-information-duality`, code lines 748-754, box lines 838-844 | Major | Algorithm mismatch (Definition mismatch) | Framework | this chapter | Implemented purity loss is the label-free entropy of $\text{softmax}(\Theta)$ rows; identity with $H(Y\mid K)$ and MI duality fail for it |
| E-012 | `def-contrastive-loss`, lines 597-614; code lines 762-777 | Moderate | Algorithm mismatch (Proof gap) | Framework | this chapter | $d_{\text{jump}}$ undefined anywhere in Volume 1; code margin inert at default, loss is $\propto \sum (w_i^\top w_j)^3$ |
| E-013 | `rem-connection-to-m-bius-re-centering`, lines 869-880; `def-class-centroid-in-poincar-disk`, lines 935-948 | Minor | Citation / reference error (Notation conflict) | Framework | this chapter | $\phi_c$ anchored to an unrelated axiom and to a section without Möbius content; $c_y$ defined twice inconsistently |
| E-014 | `cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance`, lines 913-923; line 893 | Moderate | Invalid inference (Proof gap) | Framework | this chapter | $\nabla_A V_{\text{base}}(0) = 0$ unsupported; conditioned gradient is zero a.e. under hard $K$; Langevin equation cited to a proposition that does not contain it |
| E-015 | `def-hierarchical-labels`, lines 1008-1021 | Minor | Definition mismatch | Framework | this chapter | Surjections written coarse-to-fine; impossible for the definition's own example |
| E-016 | Node 40 and Node 41 headers and tables, lines 1149-1183 | Minor | Notation conflict (Citation / reference error) | Framework | this chapter and upstream registries | Node 40 is PurityCheck here, CapacitySaturationCheck in `01_metric_law.md`/`03_info_bound.md`, colour-confinement in `04_dnn_blocks.md` |

## Detailed findings

### [E-001] Context-conditioned framework is anchored to this chapter's own section (was F-001)
- Location: lines 43, 64 (`rem-extension-not-replacement`), 110
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 43: "This section **extends** the context-conditioned framework of {ref}`sec-relationship-to-the-context-conditioned-framework`"; line 64: "{ref}`sec-relationship-to-the-context-conditioned-framework` establishes classification as selecting a context $c \in \mathcal{Y}$ (the label space), with effective potential $\Phi_{\text{eff}} = -\log p(y|z)$ (Theorem {prf:ref}`thm-universal-context-structure`)."
- Upstream anchor: the label is defined at `07_cognition/01_supervised_topo.md:46` (this chapter). The content referred to lives at `06_fields/01_boundary_interface.md:606` (`sec-the-context-space-unified-definition`) and `:647-658` (`def-context-instantiation-functor`, table row "**Classification** | Label space $\mathcal{Y}$ | Class prediction $\hat{y}$ | $-\log p(y\mid z)$ (cross-entropy)"). `thm-universal-context-structure` (`:661-676`) states embedding, kick $u_\pi(0) = \tfrac14 e_c$ and softmax motor law, not $\Phi_{\text{eff}} = -\log p(y\mid z)$.
- Why this is an error: both references point at the section they sit in, so the "prior result" is cited circularly and the reader is never sent to the definition that contains the formula; the theorem citation is to the wrong object. Line 52 ("that's what Section 23.6 was about") and `05_geometry/03_holographic_gen.md:270` ("Section 23.6 <sec-relationship-to-the-context-conditioned-framework>") show the intended target is the Section 23.6 material in `06_fields/01_boundary_interface.md`.
- Impact on downstream results: none mathematically; the loop is book-wide (`03_holographic_gen.md:270`, `10_appendices/02_parameters.md:82,92` also route through this label).
- Fix guidance:
  1. At lines 43, 64 and 110 replace the reference with `{ref}`sec-the-context-space-unified-definition``.
  2. Replace "Theorem {prf:ref}`thm-universal-context-structure`" at line 64 with "Definition {prf:ref}`def-context-instantiation-functor`".
  3. Retarget `03_holographic_gen.md:270` to `sec-the-context-space-unified-definition` as well.
- Required new assumptions/permits: none.
- Validation plan: build the docs and confirm each reference resolves to `06_fields/01_boundary_interface.md`.

### [E-002] "Soft Injectivity" transition charts cannot exist under the purity definition (was F-002)
- Location: `prop-soft-injectivity`, lines 99-112, against `def-semantic-partition`, lines 76-89
- Severity: Moderate
- Type: Definition mismatch (secondary: Invalid inference)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 102-104: "The sub-atlases need not be disjoint. Charts in $\mathcal{A}_i \cap \mathcal{A}_j$ for $i \neq j$ are **transition regions** characterized by: 1. **Low purity:** $\max_y P(Y=y \mid K=k) < 1 - \epsilon_{\text{purity}}$ for all $y$ 2. **High entropy:** $H(Y \mid K=k) > H_{\text{transition}}$".
- Upstream anchor: line 82-85 (this chapter): "$\mathcal{A}_y := \{k \in \mathcal{K} : P(Y=y \mid K=k) > 1 - \epsilon_{\text{purity}}\}$, where $\epsilon_{\text{purity}} \in (0, 0.5)$".
- Why this is an error: with $\epsilon_{\text{purity}} < 0.5$ the threshold exceeds $1/2$, and $\sum_y P(Y=y\mid K=k) = 1$ forbids two classes both exceeding $1/2$; hence $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$ and the sub-atlases are always disjoint. Condition 1 ($\max_y P < 1 - \epsilon_{\text{purity}}$) characterises charts belonging to no sub-atlas, which is the opposite of being in an intersection; a chart in $\mathcal{A}_i \cap \mathcal{A}_j$ would have $P(Y=i\mid k) > 1 - \epsilon_{\text{purity}}$, contradicting condition 1. $H_{\text{transition}}$ is not defined anywhere in the chapter.
- Impact on downstream results: appendix Step 5 (`10_appendices/01_derivations.md:490`, "transition regions $\mathcal{A}_i \cap \mathcal{A}_j$") inherits the empty-set problem; the "Geography of Confusion" box and `rem-tunneling-as-anomaly-detection` describe transitions as intersections.
- Fix guidance:
  1. Define the transition set as the complement $\mathcal{T} := \mathcal{K} \setminus \bigcup_y \mathcal{A}_y = \{k : \max_y P(Y=y\mid K=k) \le 1 - \epsilon_{\text{purity}}\}$ and delete the sentence "The sub-atlases need not be disjoint."
  2. Either define $H_{\text{transition}}$ (for instance $H_{\text{transition}} := -(1-\epsilon)\log(1-\epsilon) - \epsilon\log\epsilon$, the entropy at the purity threshold for a two-class split) or drop condition 2.
  3. Update the appendix Step 5 wording to "transition charts $\mathcal{T}$".
- Required new assumptions/permits: none. (If genuinely overlapping sub-atlases are wanted, $\epsilon_{\text{purity}}$ must be allowed in $(0,1)$ and condition 1 rewritten.)
- Validation plan: check that every statement about transition charts is consistent with $\mathcal{T}$, in particular the saddle-point remark at line 108.

### [E-003] The semantic term is constant on each chart, so the class-conditioned flow has no class-dependent drift (was F-003)
- Location: `def-class-conditioned-potential` (lines 138-163), `def-region-of-attraction` (lines 195-215), `thm-classification-as-relaxation` (lines 227-257); also line 334, `prop-class-conditioned-langevin` line 901
- Severity: Major
- Type: Conceptual (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 144-148: "$V_y(z, K) := -\beta_{\text{class}} \log P(Y=y \mid K) + V_{\text{base}}(z, K)$ ... $P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$"; line 207: "$\dot{z} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z)\nabla_A V_y(z)\right)$"; line 213: "$\mathcal{B}_y$ is the set of initial conditions from which the deterministic gradient flow on $V_y$ converges to the class-$y$ region."; line 901: "The semantic term $-\beta_{\text{class}} \log P(Y=y \mid K)$ biases the flow toward class-$y$ charts."
- Upstream anchor: `01_foundations/02_control_loop.md:139`: "$K$ is a *discrete predictive latent*"; router weights are soft only at the routing stage, `03_architecture/01_compute_tiers.md:1298` (`def-attentive-routing-law`).
- Why this is an error: the semantic term depends on $z$ only through the discrete index $K = K(z)$ (the Lyapunov function at line 248 is $L(z) = V_y(z, K(z))$). It is piecewise constant, so $\nabla_z V_y(z, K(z)) = \nabla_z V_{\text{base}}(z, K(z))$ wherever the gradient exists. The ODE at line 207 and the SDE at line 233 therefore have the same drift for every $y$; the label never enters the equation of motion, and the flow generating the sets $\mathcal{B}_y$ is label-independent (the $\mathcal{B}_y$ are just the preimages of the $\mathcal{A}_y$ under the limit map of the $V_{\text{base}}$ flow). "Semantic gravity" (lines 133-135), "$V_y$ creates basins of attraction" (line 334) and "biases the flow toward class-$y$ charts" (line 901) are not properties of the object defined. The only place the label acts is through chart-boundary crossings (the jump channel, `def-class-consistent-jump-rate`), which the continuous flow does not model. The chapter's own `cor-inference-via-relaxation` step 2 relaxes "under neutral potential $V_{\text{base}}$", implicitly conceding the point. The appendix technical note (`10_appendices/01_derivations.md:505-511`, "$\sum_k w_k(z) V_y(z, k)$ is continuous") sketches a soft alternative that neither the definition nor the theorem uses.
- Impact on downstream results: `thm-classification-as-relaxation`, `cor-inference-via-relaxation`, `prop-class-conditioned-langevin`, `cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance` (E-014), RL Connection #21 ("Trajectories relax into class-specific basins via gradient flow"), Summary item 2, Appendix A.5.
- Fix guidance:
  1. Define the semantic potential through the soft router: $V_y(z) := -\beta_{\text{class}} \log \sum_k w_k(z)\, P(Y=y \mid K=k) + V_{\text{base}}(z)$, which is differentiable in $z$ and reduces to the current formula where routing is one-hot.
  2. Restate `def-region-of-attraction`, the theorem, the Langevin proposition and the symmetry-breaking corollary for this potential; its gradient is $-\beta_{\text{class}} \sum_k \nabla w_k(z) P(Y=y\mid k) / \sum_k w_k(z) P(Y=y\mid k)$, which is genuinely label-dependent.
  3. Alternatively, keep the hard form and reformulate the theorem as a statement about the jump-diffusion (continuous flow under $V_{\text{base}}$ plus class-modulated jumps), removing all "gradient flow on $V_y$" language.
- Required new assumptions/permits: differentiability of $w_k(z)$ in $z$ (holds for the softmax router).
- Validation plan: verify symbolically that $\nabla_z V_y \neq \nabla_z V_{\text{base}}$ for the new definition, and re-derive the Lyapunov computation (E-008) for it.

### [E-004] "$-\log P$ ... is exactly the free energy" misstates statistical physics (was F-009)
- Location: "Why Logarithms?" box, item 4, line 192
- Severity: Minor
- Type: Conceptual
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "In statistical physics, $-\log P$ at inverse temperature $\beta_{\text{ent}}$ is exactly the free energy."
- Upstream anchor: `05_geometry/04_equations_motion.md:513`: "**Free energy tradeoff:** The entropy-energy balance $\Phi = E - T_c S$".
- Why this is an error: for $P(x) = e^{-\beta E(x)}/Z$, $-\log P(x) = \beta E(x) + \log Z = \beta\,(E(x) - F)$ with $F = -\beta^{-1}\log Z$ (checked numerically). $-\log P$ is the dimensionless energy measured from the free energy, i.e. the surprisal, which is exactly what item 1 of the same box calls it; the free energy is the state-independent constant $F$.
- Impact on downstream results: none (explanatory box), but it mislabels the quantity $V_y$ is built from.
- Fix guidance:
  1. Replace item 4 with: "In statistical physics, $-\log P = \beta(E - F)$: the surprisal is the energy measured from the free energy $F = -\beta^{-1}\log Z$. The class temperature $\beta_{\text{class}}$ plays the role of $\beta$."
- Required new assumptions/permits: none.
- Validation plan: consistency with item 1 and with `04_equations_motion.md:513`.

### [E-005] Region of attraction compares a latent point with a set of chart indices (was F-004)
- Location: `def-region-of-attraction`, lines 200-203
- Severity: Minor
- Type: Typo (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 201: "$\mathcal{B}_y := \{z \in \mathcal{Z} : \lim_{t \to \infty} \phi_t(z) \in \mathcal{A}_y\}$".
- Upstream anchor: line 82 (this chapter): $\mathcal{A}_y \subset \mathcal{K}$.
- Why this is an error: $\phi_t(z) \in \mathcal{Z}$ while $\mathcal{A}_y \subset \mathcal{K}$, so the membership is ill-typed; the theorem at line 240 uses the well-typed $K(z(s)) \in \mathcal{A}_y$. The definition also presupposes that the limit exists, with no hypothesis excluding non-convergent trajectories of the curl-corrected flow when $\mathcal{F} \neq 0$. Appendix Step 5 (`10_appendices/01_derivations.md:496`) repeats "$\lim_{s\to\infty} z(s) \in \mathcal{A}_y$".
- Impact on downstream results: cosmetic for the theorem; the substantive issue with $\mathcal{B}_y$ is E-003.
- Fix guidance:
  1. Write $\mathcal{B}_y := \{z \in \mathcal{Z} : \omega(z) \subset K^{-1}(\mathcal{A}_y)\}$ with $\omega(z)$ the $\omega$-limit set, or "$\lim_{t\to\infty}\phi_t(z)$ exists and $K(\lim_{t\to\infty}\phi_t(z)) \in \mathcal{A}_y$".
  2. Apply the same correction in the appendix Step 5.
- Required new assumptions/permits: none.
- Validation plan: type-check every occurrence of $\mathcal{A}_y$ and $\mathcal{B}_y$ in the chapter and appendix.

### [E-006] $\nabla_A$ is redefined without the metric, conflicting with the book convention; $\nabla_G$ is undefined (was F-005)
- Location: lines 207, 210 (`def-region-of-attraction`), 233 (`thm-classification-as-relaxation`), 896 (`prop-class-conditioned-langevin`)
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter (the appendix copy at `10_appendices/01_derivations.md:427` repeats the local redefinition)
- Claim (verbatim): line 207: "$\dot{z} = \mathcal{M}_{\text{curl}}\!\left(-G^{-1}(z)\nabla_A V_y(z)\right)$"; line 210: "Here $\nabla_A V_y := \nabla V_y - A$"; line 896: "$dz = \mathcal{M}_{\text{curl}}\!\left(-\nabla_G V_y(z, K)\right) d\tau$".
- Upstream anchor: `01_foundations/02_control_loop.md:145-147`: "Define the covariant 1-form $d_A V := dV - A$; its metric-raised vector is $\nabla_A V := G^{-1}(dV - A)$ (coordinates: $(\nabla_A V)^i = G^{ij}(\partial_j V - A_j)$)".
- Why this is an error: in the book convention $\nabla_A V$ already contains $G^{-1}$, so $G^{-1}\nabla_A V$ applies the inverse metric twice; the chapter avoids this only by silently redefining $\nabla_A V$ as the 1-form $d_A V$. The Langevin proposition then uses a third symbol $\nabla_G V_y$ that is defined nowhere in Volume 1 and drops $A$, so the three displayed dynamics are not the same equation although the text treats them as such.
- Impact on downstream results: unit/shape bookkeeping only; no numeric consequence once a convention is fixed.
- Fix guidance:
  1. Write the drift as $-\mathcal{M}_{\text{curl}}\nabla_A V_y$ with the book's metric-raised $\nabla_A$ (equivalently $-\mathcal{M}_{\text{curl}}G^{-1} d_A V_y$) at lines 207 and 233, and delete the local redefinition at line 210 in favour of a pointer to `02_control_loop.md`.
  2. Use the identical expression at line 896.
  3. Mirror in `10_appendices/01_derivations.md:427`.
- Required new assumptions/permits: none.
- Validation plan: grep the chapter and appendix for `nabla_A` and `nabla_G` and confirm a single convention.

### [E-007] Almost-sure convergence at positive temperature contradicts the finite-barrier hypothesis; "global minimum by construction" needs a hypothesis on $\beta_{\text{class}}$ (was F-006)
- Location: `thm-classification-as-relaxation`, lines 227-257
- Severity: Major
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework (secondary: External, for the Kramers/Freidlin-Wentzell estimate the fix relies on)
- Origin: this chapter; mirrored in `10_appendices/01_derivations.md:419-527`
- Claim (verbatim): lines 239-246: "$\lim_{s \to \infty} K(z(s)) \in \mathcal{A}_y \quad \text{almost surely}$, provided: 1. $z(0) \in \mathcal{B}_y$ (initial condition in the basin) 2. $T_c$ is sufficiently small (low temperature limit) 3. The basins have positive measure and are separated by finite barriers"; line 255: "The class-$y$ region is the global minimum of $V_y$ by construction."
- Upstream anchor: `10_appendices/01_derivations.md:517-523`: "For small but positive $T_c$, standard results on diffusions in potential wells (Kramers' law) give the escape rate from basin $\mathcal{B}_y$: $\text{Rate}_{\text{escape}} \sim e^{-\Delta V / T_c}$ ... For $T_c \ll \Delta V$, escape is exponentially unlikely, ensuring practical convergence." Noise term `05_geometry/04_equations_motion.md:915`: "$\sqrt{2T_c}\,(G^{-1/2}(z))^{kj}\,dW^j_s$".
- Why this is an error: (a) For any fixed $T_c > 0$ the SDE at line 233 is a non-degenerate diffusion. With finite barriers the escape rate $e^{-\Delta V/T_c}$ is strictly positive, and a positive-recurrent diffusion re-enters every positive-measure basin (hypothesis 3) infinitely often; the discrete process $K(z(s))$ is therefore not eventually constant and $\lim_{s\to\infty} K(z(s))$ does not exist almost surely. "Sufficiently small $T_c$" cannot rescue an almost-sure statement about $s \to \infty$; only the $T_c \to 0$ limit (deterministic flow, where the conclusion restates the definition of $\mathcal{B}_y$) or a finite-horizon bound $\Pr[K(z(s)) \in \mathcal{A}_y\ \forall s \le S] \ge 1 - C\,S\,e^{-\Delta V/T_c}$ is true, which is what the appendix's own "practical convergence" sentence amounts to. (b) $V_y = -\beta_{\text{class}}\log P(Y=y\mid K) + V_{\text{base}}$. The semantic term is at most $-\beta_{\text{class}}\log(1-\epsilon_{\text{purity}})$ on $\mathcal{A}_y$, but $V_{\text{base}}$ is an arbitrary learned critic (`04_equations_motion.md:427`) that may be much lower on other charts; the class region is the global minimum only when $\beta_{\text{class}}[\log P_{\max} - \log(1-\epsilon_{\text{purity}})]$ exceeds the oscillation of $V_{\text{base}}$ across charts. No such hypothesis is stated; appendix Step 1 (`:441`) asserts the minimum from the semantic term alone. (c) Hypothesis 1 is stated in terms of $\mathcal{B}_y$, which is defined by convergence of the deterministic flow to $\mathcal{A}_y$; at $T_c \to 0$ the theorem asserts its own hypothesis.
- Impact on downstream results: `cor-inference-via-relaxation`, RL Connection #21, "Two Paths" box, `rem-connection-to-classification-accuracy` in the appendix, Node 41 rationale.
- Fix guidance:
  1. Split the statement: (i) for $T_c = 0$, $z(0) \in \mathcal{B}_y \Rightarrow K(z(s)) \in \mathcal{A}_y$ for all large $s$ (this is the definition of $\mathcal{B}_y$, so present it as such); (ii) for $T_c > 0$, a finite-horizon bound: for every $S > 0$, $\Pr[\exists s \le S : K(z(s)) \notin \mathcal{A}_y] \le C\,S\,e^{-\Delta V/T_c}$ under an explicit barrier-height hypothesis $\Delta V > 0$.
  2. Add the hypothesis $\beta_{\text{class}}\log\frac{1}{1-\epsilon_{\text{purity}}} \ge \sup_{K\in\mathcal{A}_y} V_{\text{base}} - \inf_{K\notin\mathcal{A}_y} V_{\text{base}} + \Delta V$ (or an equivalent) for the global-minimum claim, and state it in Appendix A.5 Step 1.
  3. Rewrite the Feynman paragraph at lines 260-270 accordingly ("stays in the basin for exponentially long times", not "will find the right class").
- Required new assumptions/permits: a barrier-height hypothesis $\Delta V > 0$; either an internal lemma giving the exit-time bound for the framework's SDE or an explicit external permit for Kramers/Freidlin-Wentzell.
- Validation plan: simulate the SDE on a two-well $V_{\text{base}}$ with a class term and confirm hopping at any fixed $T_c > 0$; check the horizon bound numerically.

### [E-008] Lyapunov sketch uses $\nabla_A V_y$ where the chain rule gives $\nabla V_y$; non-conservative case not covered (was F-007)
- Location: `thm-classification-as-relaxation`, proof sketch, lines 248-255
- Severity: Moderate
- Type: Computational error (secondary: Scope restriction)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 251: "$\frac{dL}{ds} = \nabla_A V_y \cdot \dot{z} = -\nabla_A V_y \cdot \mathcal{M}_{\text{curl}} G^{-1}\nabla_A V_y + \text{noise terms}$."; line 254: "The antisymmetric curl contribution in $\mathcal{M}_{\text{curl}}$ does no work, so it does not increase $L$."
- Upstream anchor: the appendix computes correctly, `10_appendices/01_derivations.md:463-475`: "Since $L = V_y$, we have $\nabla L = \nabla V_y$, so: $dL = -G^{-1}(\nabla V_y, \nabla_A V_y)\, ds + \ldots$ The first term is non-positive in the conservative case ($A=0$), reducing to $-\|\nabla V_y\|_G^2$."
- Why this is an error: $L(z) = V_y(z)$, so $\dot L = \nabla V_y \cdot \dot z$. With the chapter's $\nabla_A V_y = \nabla V_y - A$ this gives $\dot L = -(\nabla_A V_y + A)^\top \mathcal{M}_{\text{curl}}G^{-1}\nabla_A V_y$, whose sign is indefinite whenever $A \neq 0$. Recomputation: $\mathcal{M}_{\text{curl}}G^{-1} = (G - \beta_{\text{curl}}\mathcal{F})^{-1}$ and $v^\top(G-\beta_{\text{curl}}\mathcal{F})^{-1}v = u^\top G u > 0$ for $u = (G-\beta_{\text{curl}}\mathcal{F})^{-1}v$ (the antisymmetric part drops), so the curl claim is correct when the same vector sits on both sides; but the cross term $-A^\top(G-\beta_{\text{curl}}\mathcal{F})^{-1}\nabla_A V_y$ was positive in about 28% of random draws with $A \neq 0$. The sketch establishes descent only for $A = 0$, while the theorem is stated for general $A$ (line 210).
- Impact on downstream results: the theorem's hypotheses; Appendix A.5 Step 3 already restricts to $A = 0$ but the chapter statement does not.
- Fix guidance:
  1. Replace $\nabla_A V_y \cdot \dot z$ by $\nabla V_y \cdot \dot z$ at line 251.
  2. Add the hypothesis $A = 0$ (conservative reward field, `def-conservative-reward-field`) to the theorem, or add a small-circulation lemma $|A^\top\mathcal{M}_{\text{curl}}G^{-1}\nabla_A V_y| \le (1-\delta)\,\nabla_A V_y^\top\mathcal{M}_{\text{curl}}G^{-1}\nabla_A V_y$ under which descent still holds.
- Required new assumptions/permits: $A = 0$, or the small-circulation bound.
- Validation plan: symbolic check of $\dot L$ for $A \neq 0$ in two dimensions.

### [E-009] The fast path is not the $T_c \to 0$, $s \to \infty$ limit of the relaxation path (was F-008)
- Location: `cor-inference-via-relaxation`, Remark (Fast Path), lines 273-285; "Two Paths" box, lines 287-307
- Severity: Moderate
- Type: Invalid inference
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 281: "we often skip the relaxation and use direct readout: $\hat{y} = \arg\max_y \sum_k w_k(x) \cdot P(Y=y \mid K=k)$ ... The relaxation interpretation justifies this as the $T_c \to 0$, $s \to \infty$ limit."; line 304: "the router weights converge to indicators for the basin membership."
- Upstream anchor: `03_architecture/01_compute_tiers.md:1298` (`def-attentive-routing-law`): $w_i(x)$ is a softmax of the encoded query; no dynamics, no dependence on $T_c$ or $s$.
- Why this is an error: the relaxation path (lines 277-279) reads out $P(Y\mid K(z^*))$ at the equilibrium $z^*$ of the $V_{\text{base}}$ flow started from $z_0 = \text{Enc}(x)$; the fast path reads out a $w(x)$-weighted mixture at $z_0$. In the $T_c \to 0$, $s \to \infty$ limit the relaxation path gives $\arg\max_y P(Y=y\mid K(z^*))$, which equals the fast-path answer only if $z_0$ already lies in the interior of the chart containing $z^*$ and $w(z_0)$ is essentially one-hot there. Nothing in the limit moves $w(x)$; it is evaluated once, before any relaxation. For $z_0$ near a chart boundary, precisely where relaxation would matter, the two readouts differ. The claimed justification is an additional assumption ($z_0 \approx z^*$ in routing), not a limit.
- Impact on downstream results: motivates `def-route-alignment-loss` and the `SupervisedTopologyLoss` forward pass, which implement the fast path.
- Fix guidance:
  1. Replace the sentence at line 281 by: "The fast path coincides with the relaxed readout whenever $\text{Enc}(x)$ already routes (nearly) one-hot to the chart of $z^*$, i.e. when no chart boundary is crossed during relaxation; otherwise it is an approximation."
  2. Rewrite the "Why does the fast path work?" paragraph at line 304 to match, or define the fast path with router weights evaluated at $z^*$.
- Required new assumptions/permits: the no-boundary-crossing assumption, stated explicitly.
- Validation plan: construct a two-chart example with $z_0$ near the boundary and compare the two readouts.

### [E-010] Effective Disconnection is false for the WFR distance the chapter cites (was F-010)
- Location: `prop-effective-disconnection`, lines 396-415; Node 41 proxy, line 1183
- Severity: Major
- Type: Invalid inference (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 399-402: "As $\gamma_{\text{sep}} \to \infty$, the effective WFR distance between charts of different classes diverges: $d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \to \infty$"; line 405: "The WFR distance (Definition {prf:ref}`def-the-wfr-action`) involves minimizing over paths that may use both transport (continuous flow within charts) and reaction (jumps between charts)"; line 407: "pure transport paths have infinite cost"; line 411: "the reaction term penalizes staying in transition states waiting for rare jumps."
- Upstream anchor: `05_geometry/02_wfr_geometry.md:106-127` (`def-the-wfr-action`): "The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ is the infimum of $\mathcal{E}[\rho, v, r] = \int_0^1 \int_{\mathcal{Z}} \left( \|v_s(z)\|_G^2 + \lambda^2 |r_s(z)|^2 - 2\langle \mathbf{A}(z), v_s(z) \rangle \right) d\rho_s(z)\, ds$ subject to $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$ where ... $r_s(z) \in \mathbb{R}$ is the **reaction rate** (growth/decay of mass), $\lambda > 0$ is the **length-scale parameter**". Jump rates appear only in the discrete master-equation picture, `:374-381` (`thm-classical-master-equation-wfr`).
- Why this is an error: (a) In `def-the-wfr-action` the reaction rate $r$ is a free control, unconstrained by any inter-chart rate $\lambda_{i\to j}$, and $\gamma_{\text{sep}}$ does not appear in the functional, so $d_{\text{WFR}}$ cannot depend on it. (b) The distance is uniformly bounded: the pure-reaction path $\rho_s = ((1-s)\sqrt{\rho_0} + s\sqrt{\rho_1})^2$ with $v = 0$, $r = \partial_s \log \rho_s$ has cost $\int_0^1\!\!\int \lambda^2(\partial_s\rho_s)^2/\rho_s = 4\lambda^2\int(\sqrt{\rho_1}-\sqrt{\rho_0})^2 \le 4\lambda^2(\rho_0(\mathcal{Z}) + \rho_1(\mathcal{Z}))$ (numerically verified to $10^{-10}$), so $d_{\text{WFR}} \le 2\lambda\sqrt{\rho_0(\mathcal{Z}) + \rho_1(\mathcal{Z})}$ for all inputs and all $\gamma_{\text{sep}}$. (c) Sketch Step 1 is false on a connected latent manifold: mass can be transported continuously through intermediate charts. Step 2's "dwell times" is a kinetic notion with no counterpart in a static geodesic distance. (d) $d_{\text{WFR}}$ is a distance between measures on $\mathcal{Z}$, while $\mathcal{A}_y \subset \mathcal{K}$ is a set of chart indices, so the left-hand side is ill-typed.
- Impact on downstream results: central claim of the metric-segmentation section; justification for `rem-tunneling-as-anomaly-detection`, RL Connection #21 ("Classes are metrically separated"), Node 41's proxy $\min d_{\text{WFR}}(\mathcal{A}_{y_1},\mathcal{A}_{y_2})$, Summary item 3 ("effectively disconnecting different classes in the metric").
- Fix guidance:
  1. State the result for the discrete transport metric of `thm-classical-master-equation-wfr`, in which edge conductances are proportional to the rates: with $\lambda^{\text{sup}}_{i\to j} = \lambda^{(0)}_{i\to j}e^{-\gamma_{\text{sep}}D_{\text{class}}(i,j)}$, every path between charts of different classes uses at least one cross-class edge of conductance $\propto e^{-\gamma_{\text{sep}}}$, and the discrete Wasserstein distance between $\delta$-masses grows like conductance$^{-1/2} \sim e^{\gamma_{\text{sep}}/2}$, hence diverges.
  2. Replace $d_{\text{WFR}}(\mathcal{A}_{y_1},\mathcal{A}_{y_2})$ by $\min_{i\in\mathcal{A}_{y_1},\,j\in\mathcal{A}_{y_2}} \mathcal{W}_{\lambda^{\text{sup}}}(\delta_i,\delta_j)$ everywhere, including the Node 41 proxy at line 1183.
  3. Add a sentence stating that the continuous WFR distance of `def-the-wfr-action` is not affected by $\gamma_{\text{sep}}$.
- Required new assumptions/permits: an internal lemma for the scaling of the Maas/Mielke discrete distance with edge conductance (or an explicit external permit citing `maas2011gradient`).
- Validation plan: compute the discrete distance on a three-chart chain with one suppressed edge and check the $e^{\gamma_{\text{sep}}/2}$ growth.

### [E-011] Implemented purity loss is the entropy of the learnable $\Theta$, so $\mathcal{L}_{\text{purity}} = H(Y\mid K)$ and the MI duality fail for it (was F-011)
- Location: `def-purity-loss` (lines 500-515), `prop-purity-information-duality` (lines 527-538), Algorithm 25.4.7 (lines 748-754), "Anatomy" box (lines 838-844)
- Severity: Major
- Type: Algorithm mismatch (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 506-513: "$\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(Y \mid K=k)$ ... *Interpretation:* $\mathcal{L}_{\text{purity}} = H(Y \mid K)$"; line 530: "Minimizing $\mathcal{L}_{\text{purity}}$ is equivalent to maximizing the mutual information $I(K; Y)$"; code lines 750-754: `entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)` with `p_y_k = F.softmax(self.chart_to_class / self.temperature, dim=1)`; line 844: "This is exactly $H(Y|K)$."
- Upstream anchor: line 148 (this chapter): "$P(Y=y \mid K) = \text{softmax}(\Theta_{K,:})_y$ with learnable parameters $\Theta$".
- Why this is an error: the identity $H(Y) = H(Y\mid K) + I(K;Y)$ holds for the data conditional $P_{\text{data}}(Y\mid K)$. The code (and the definition, if $P(Y\mid K)$ is read as at line 148) uses the model rows $q_k = \text{softmax}(\Theta_{k,:})$. Then $\sum_k \bar w_k H(q_k)$ is label-free: `y_true` does not enter `loss_purity`, and the term is driven to zero by making every row of $\Theta$ one-hot, whichever class, creating no mutual information with $Y$. The true conditional entropy requires $\sum_k \bar w_k H(P_{\text{data}}(Y\mid K=k))$ with $P_{\text{data}}(Y=y\mid K=k) = \mathbb{E}[w_k(x)\mathbb{1}[Y=y]]/\mathbb{E}[w_k(x)]$, which the EMA remark (line 158) describes but the code does not compute. As implemented, $\mathcal{L}_{\text{purity}}$ is a confidence regulariser on $\Theta$, and `prop-purity-information-duality` does not apply to it. The same ambiguity affects Node 40's proxy (line 1158) and `def-hierarchical-supervised-loss`.
- Impact on downstream results: `def-total-loss`, `def-hierarchical-supervised-loss`, `prop-scale-label-alignment`, the Node 40 remedy ("increase $\lambda_{\text{pur}}$" merely sharpens $\Theta$), and Appendix A.5 Step 1.
- Fix guidance:
  1. In `def-purity-loss` state explicitly that $P(Y\mid K=k)$ is the data (batch or EMA) conditional, distinct from the model rows $\text{softmax}(\Theta_{k,:})$.
  2. In the code compute `p_yk_data = (router_weights.t() @ F.one_hot(y_true, C).float()) / (router_weights.sum(0, keepdim=True).t() + eps)` and take its row entropies weighted by `p_k`; gradients then flow through the router weights.
  3. If the model-entropy term is kept as a separate regulariser, rename it (e.g. $\mathcal{L}_{\text{conf}}$) and remove the "$= H(Y\mid K)$" and duality claims for it.
- Required new assumptions/permits: none.
- Validation plan: unit test that the corrected `loss_purity` changes when `y_true` is permuted and that it equals the batch $H(Y\mid K)$ for one-hot routing.

### [E-012] Metric contrastive loss: $d_{\text{jump}}$ is undefined and the code's margin is inert at its default (was F-012)
- Location: `def-contrastive-loss` (lines 597-614); Algorithm 25.4.7 metric block (lines 762-777); box line 853
- Severity: Moderate
- Type: Algorithm mismatch (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 603: "$\mathcal{L}_{\text{metric}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}: y_i \neq y_j} w_i^\top w_j \cdot \max(0, m - d_{\text{jump}}(z_i, z_j))^2$"; line 610: "$d_{\text{jump}}(z_i, z_j)$ is the minimum jump cost ({ref}`sec-factorized-jump-operators-efficient-chart-transitions`)"; code: `pseudo_dist = 1.0 - overlap`, `hinge = F.relu(self.margin - pseudo_dist)`, `margin: float = 1.0`, `loss_metric = (y_diff * overlap * hinge ** 2).sum() / (y_diff.sum() + 1e-8)`.
- Upstream anchor: `03_architecture/01_compute_tiers.md:2354-2480` defines only transition maps ("$L_{i \to j}(z) = A_j(B_i z + c_i) + d_j$", `def-factorized-jump-operator` at line 2436). No cost, length or distance $d_{\text{jump}}$ is defined there or anywhere else in `docs/source/1_agent` (grep; `10_appendices/06_losses.md:235` names an equally undefined $d_G^{\text{jump}}$).
- Why this is an error: (a) The formal definition rests on an object that does not exist in the framework; the reference given does not define it. (b) For simplex vectors $0 \le w_i^\top w_j \le 1$, so `pseudo_dist` $\in [0,1]$; with `margin=1.0` the hinge is `relu(overlap) = overlap` (verified) and the loss is $\sum_{y_i\neq y_j}(w_i^\top w_j)^3/\#\{y_i\neq y_j\}$. The margin never deactivates, so the "distances greater than $m$ contribute zero" property (line 621) is lost, and the quantity is unrelated to the displayed hinge on a jump distance. Normalisation also differs ($1/|\mathcal{P}|$ over all pairs versus over different-class ordered pairs). The box at line 853 presents this as a "simplification of the full jump-distance computation".
- Impact on downstream results: `def-total-loss`; Node 41 remedy "add metric contrastive loss".
- Fix guidance:
  1. Define $d_{\text{jump}}$ in the text, for example the within-chart geodesic distance $d_G$ plus, per jump $i \to j$, the cost $-\log(\lambda^{\text{sup}}_{i\to j}/\lambda_{\max})$ so that $\gamma_{\text{sep}}$ enters; or replace $d_{\text{jump}}$ by $d_G(z_i,z_j)$ using the `z_latent` argument the code accepts but never uses.
  2. If the overlap proxy is retained, define it in the text as $\mathcal{L}^{\text{proxy}}_{\text{metric}}$ with a margin $m < 1$ and the code's normalisation.
- Required new assumptions/permits: none beyond the chosen definition of $d_{\text{jump}}$.
- Validation plan: check that the loss vanishes for pairs beyond the margin under the new definition.

### [E-013] Möbius re-centering is anchored to an unrelated axiom and never defined; the class centroid is defined twice inconsistently (was F-013)
- Location: `rem-connection-to-m-bius-re-centering` (lines 869-880); `def-class-centroid-in-poincar-disk` (lines 935-948, cross-references line 946)
- Severity: Minor
- Type: Citation / reference error (secondary: Notation conflict)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 872-875: "The Mobius re-centering $\phi_c$ for conditioned generation (Definition {prf:ref}`ax-bulk-boundary-decoupling`) can be interpreted as centering at the **class centroid**: $c_y := \mathbb{E}_{x: Y(x)=y}[\text{Enc}(x)]$"; line 941: "$c_y := \arg\min_{c \in \mathbb{D}} \sum_{x: Y(x)=y} d_{\mathbb{D}}(c, \text{Enc}(x))^2$"; line 946: "{ref}`sec-the-retrieval-texture-firewall` (Mobius Re-centering)".
- Upstream anchor: `05_geometry/03_holographic_gen.md:541-560` (`ax-bulk-boundary-decoupling`): "The state decomposition $Z = (K, z_n, z_{\text{tex}})$ satisfies a **partition condition**" (an axiom; no Möbius map, no $\phi_c$). `07_cognition/03_memory_retrieval.md:696` (`sec-the-retrieval-texture-firewall`): no occurrence of Möbius/Mobius/re-centering. `10_appendices/02_parameters.md:70`: "$\phi_c$ | Möbius automorphism moving $c$ to origin ({ref}`Section 21.3 <sec-the-retrieval-texture-firewall>`)" points to the same section. Möbius translation exists only as code, `05_geometry/04_equations_motion.md:641-683`.
- Why this is an error: both references for $\phi_c$ are wrong (and an axiom is called a Definition), and the object is not formally defined in Volume 1. Independently, $c_y$ receives two different definitions: the Euclidean coordinate average (line 875) and the hyperbolic Fréchet mean (line 941). These do not coincide: for two points at Poincaré radius 0.9 and angles $\pm 30^\circ$ the Euclidean mean has radius 0.779 and the Fréchet mean radius 0.571 (recomputed). The remark uses the first, the definition the second.
- Impact on downstream results: description of class-conditioned generation; no proofs depend on it.
- Fix guidance:
  1. Add a definition $\phi_c(z) := (-c) \oplus z$ (Möbius translation taking $c$ to $0$) near `def-effective-potential` or in `05_geometry/03_holographic_gen.md`, and point line 872, line 946 and `02_parameters.md:70` to it.
  2. Delete the Euclidean $c_y$ at line 875 or label it $c_y^{\text{Eucl}}$ with a statement that the Fréchet mean of line 941 is the one used.
- Required new assumptions/permits: none.
- Validation plan: grep for $\phi_c$ and confirm a single definition and consistent pointers.

### [E-014] Symmetry-breaking corollary: $\nabla V_{\text{base}}(0) = 0$ is unsupported and the conditioned gradient vanishes under hard chart assignment (was F-014)
- Location: `cor-label-as-symmetry-breaking-field-cf-classifier-free-guidance` (lines 913-923); `prop-class-conditioned-langevin` line 893
- Severity: Moderate
- Type: Invalid inference (secondary: Proof gap / omission)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 918-919: "1. **Unconditioned:** $\nabla_A V_{\text{base}}(0) = 0$ (symmetric saddle) 2. **Conditioned:** $\nabla_A V_y(0) = -\beta_{\text{class}} \nabla_z \log P(Y=y \mid K(z))|_{z=0} \neq 0$"; line 893: "The generative Langevin equation ... (Definition {prf:ref}`prop-so-d-symmetry-at-origin`)".
- Upstream anchor: `05_geometry/03_holographic_gen.md:236-243` (`prop-so-d-symmetry-at-origin`): "At $z = 0$: 1. The metric is isotropic: $G(0) = 4I$ 2. The entropic force vanishes: $F_{\text{entropy}}(0) = 0$ 3. The system has full rotational symmetry $SO(D)$." The critic is a separate learned term, `05_geometry/04_equations_motion.md:420-427`. The overdamped Langevin SDE is `thm-overdamped-limit`, `04_equations_motion.md:905-916`.
- Why this is an error: (a) The upstream symmetry concerns $U(z) = -2\operatorname{artanh}|z|$ and the metric, not the learned $V_{\text{base}} = V_{\text{critic}}$; nothing in the framework forces $\nabla V_{\text{base}}(0) = 0$ or $A(0) = 0$, so item 1 is an unstated hypothesis. (b) Item 2 differentiates $\log P(Y=y\mid K(z))$ in $z$; with the hard index of `def-class-conditioned-potential` this is piecewise constant, so its gradient is $0$ wherever it exists and undefined on chart boundaries. The asserted "$\neq 0$" is unsupported as written (see E-003). Even with a soft router the gradient at the exact origin could vanish by symmetry if chart tokens are symmetric. (c) `prop-so-d-symmetry-at-origin` contains no Langevin equation.
- Impact on downstream results: the classifier-free-guidance analogy and `rem-integration-with-topologicaldecoder` item 1 ("class determines charts").
- Fix guidance:
  1. Use the soft semantic potential of E-003.
  2. State the hypothesis "$\nabla V_{\text{base}}(0) = 0$ and $A(0) = 0$ (the origin is a critical point of the unconditioned potential)" explicitly.
  3. Conclude $\nabla V_y(0) = -\beta_{\text{class}}\sum_k \nabla w_k(0)\,P(Y=y\mid K=k) / \sum_k w_k(0)P(Y=y\mid K=k)$, non-zero iff the routing gradient at $0$ is not orthogonal to the class-affinity vector, and state that condition.
  4. Cite `thm-overdamped-limit` for the Langevin equation at line 893.
- Required new assumptions/permits: the critical-point hypothesis at the origin; the non-orthogonality condition.
- Validation plan: compute $\nabla V_y(0)$ for a two-chart symmetric router and confirm the stated condition.

### [E-015] Label hierarchy surjections point from coarse to fine, which is impossible (was F-015)
- Location: `def-hierarchical-labels`, lines 1008-1021
- Severity: Minor
- Type: Definition mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): lines 1014-1019: "$\mathcal{Y}_0 \twoheadrightarrow \mathcal{Y}_1 \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_L$, where $\twoheadrightarrow$ denotes a surjection (coarsening). $\mathcal{Y}_0$ are coarse labels (super-categories), $\mathcal{Y}_L$ are fine labels (leaf categories). *Example:* $\mathcal{Y}_0 = \{\text{Animal}, \text{Vehicle}\}$, $\mathcal{Y}_1 = \{\text{Dog}, \text{Cat}, \text{Car}, \text{Bike}\}$".
- Upstream anchor: not applicable (defined in this chapter).
- Why this is an error: a surjection $\mathcal{Y}_0 \twoheadrightarrow \mathcal{Y}_1$ requires $|\mathcal{Y}_0| \ge |\mathcal{Y}_1|$; in the example $2 < 4$, so no such chain exists. Coarsening maps fine to coarse ("Each fine label maps to exactly one coarser label", line 1024), i.e. $\mathcal{Y}_L \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_0$. The intended object is unambiguous from the prose, and `prop-scale-label-alignment` and `def-hierarchical-supervised-loss` only index levels, so nothing downstream depends on the direction; this is why the severity is Minor.
- Impact on downstream results: none beyond the definition itself.
- Fix guidance:
  1. Write $\mathcal{Y}_L \twoheadrightarrow \mathcal{Y}_{L-1} \twoheadrightarrow \cdots \twoheadrightarrow \mathcal{Y}_0$ with coarsening maps $\pi_\ell : \mathcal{Y}_\ell \twoheadrightarrow \mathcal{Y}_{\ell-1}$.
- Required new assumptions/permits: none.
- Validation plan: check the example satisfies the reversed chain.

### [E-016] Node 40 is given three unrelated identities across the book (was F-016)
- Location: Node 40 header and table (lines 1149-1158); Node 41 header (line 1179)
- Severity: Minor
- Type: Notation conflict (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter and upstream (`05_geometry/01_metric_law.md:325`, `06_fields/03_info_bound.md:683`, `08_multiagent/04_dnn_blocks.md:3163`, `08_multiagent/03_parameter_sieve.md:595`)
- Claim (verbatim): line 1150: "**Node 40: PurityCheck (CapacitySaturationCheck)**"; line 1158: "| **40** | **PurityCheck** | **Router** | **Semantic Clustering** | Are charts class-pure? | $H(Y \mid K)$ | $O(BC)$ |"; line 1179: "**Node 41: ClassSeparationCheck (SupervisedTopologyChecks)**".
- Upstream anchor: `05_geometry/01_metric_law.md:325`: "| 40 | CapacitySaturationCheck | Bulk-boundary information ratio | $I_{\text{bulk}} / C_{\partial} > 1 - \epsilon$ |"; `06_fields/03_info_bound.md:683-684`: "| 40 | [CapacitySaturationCheck](#node-40) | 18.3 | $I_{\text{bulk}}/C_\partial$ |" and "| 41 | [SupervisedTopologyChecks](#node-41) | 25.4 |"; `08_multiagent/04_dnn_blocks.md:3163`: "| 40 | PurityCheck | $SU(N_f)_C$ confinement | Non-neutral bundles at macro boundary |"; `08_multiagent/03_parameter_sieve.md:595`: "Node 40 (PurityCheck) enforces that only color-neutral bound states reach the macro-register."
- Why this is an error: one node number is used for three diagnostics with unrelated proxies: $H(Y\mid K)$ (this chapter), $I_{\text{bulk}}/C_\partial$ (metric law, info bound), and colour neutrality $\sum_a |Q^a_C|^2$ (DNN blocks, parameter sieve). The registry row in `03_info_bound.md:683` links its CapacitySaturationCheck to `#node-40`, which resolves to this chapter's PurityCheck table with the wrong proxy. The parenthetical "(CapacitySaturationCheck)" at line 1150 records the collision without resolving it. The registry lists no ClassSeparationCheck for 41. The cost $O(BC)$ at line 1158 should be $O(BN_c + N_cC)$ for the quantity as implemented.
- Impact on downstream results: sieve bookkeeping only (Volume 1 claims 60 numbered nodes); no mathematical result depends on the number.
- Fix guidance:
  1. Give PurityCheck and ClassSeparationCheck distinct identifiers (e.g. 41a/41b under SupervisedTopologyChecks) and let Node 40 be CapacitySaturationCheck with anchor `sec-diagnostic-node-capacity-saturation`.
  2. Update `06_fields/03_info_bound.md:683`, `08_multiagent/04_dnn_blocks.md:3163` and `08_multiagent/03_parameter_sieve.md:595` accordingly (the confinement check needs its own number).
  3. Correct the cost entry to $O(BN_c + N_cC)$.
- Required new assumptions/permits: none.
- Validation plan: grep for "Node 40" and "node-40" across Volume 1 and confirm a single identity.

## Scope restrictions and clarifications
- The relaxation theorem (E-007) can be made true only in one of two forms: as a deterministic ($T_c = 0$) statement that restates the definition of $\mathcal{B}_y$, or as a finite-horizon probability bound under an explicit barrier-height hypothesis; an almost-sure statement at fixed positive temperature is not available.
- Every statement in the chapter that attributes label-dependent drift to $V_y$ (E-003, E-014, line 334, line 901) requires the soft-router form of the semantic potential; under the current hard-index form the label acts only through the jump channel.
- The "Effective Disconnection" result (E-010) is a statement about the discrete rate-weighted transport metric of `thm-classical-master-equation-wfr`, not about the continuous WFR distance of `def-the-wfr-action`; the Node 41 proxy must be redefined accordingly.
- The purity loss (E-011) and its information-theoretic interpretation are correct only for the data conditional $P_{\text{data}}(Y\mid K)$; the implemented quantity is a confidence regulariser on $\Theta$.
- Appendix A.5 (`10_appendices/01_derivations.md:417-527`) shares E-002, E-003, E-005, E-006 and E-007 and must be revised together with this chapter.

## Open questions
- Should the chapter adopt the soft semantic potential $-\beta_{\text{class}}\log\sum_k w_k(z)P(Y=y\mid K=k)$ (making all gradient statements meaningful) or reformulate the theory around the jump-diffusion with class-modulated rates? The two choices lead to different theorems and different losses.
- Is a divergence result for the discrete rate-weighted metric (E-010) enough for the anomaly-detection remark, or is a quantitative $e^{\gamma_{\text{sep}}/2}$ scaling needed downstream?
- Is $d_{\text{jump}}$ meant to depend on $\gamma_{\text{sep}}$ (so that the metric loss and the jump modulation are coupled), or is the geodesic $d_G$ intended, as `10_appendices/06_losses.md:235` suggests?
- Who owns node numbers 40 and 41 in the sieve registry, and should the confinement check in `08_multiagent` receive its own number?

## Rejected candidate findings
None. All sixteen stage-1 findings were confirmed; two were adjusted (F-006: criterion changed to Framework with External secondary; F-015: severity lowered to Minor and criterion changed to Framework). No findings were added by the verifier.
