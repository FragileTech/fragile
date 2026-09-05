# Mathematical Review: docs/source/1_agent/intro_agent.md

## Metadata
- Reviewed file: docs/source/1_agent/intro_agent.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (657 lines)
- Framework anchors (definitions/axioms/permits):
  - docs/source/1_agent/06_fields/02_reward_field.md:324-340 (thm-the-hjb-helmholtz-correspondence; discount rate lambda, screening mass kappa)
  - docs/source/1_agent/01_foundations/02_control_loop.md:584-607 (Lyapunov constraint with lambda_Lyap)
  - docs/source/1_agent/05_geometry/02_wfr_geometry.md:358 (teleportation length lambda)
  - docs/source/1_agent/02_sieve/01_diagnostics.md:38, 48, 109-115 (node count, PASS predicate, Nodes 13 and 15)
  - docs/source/1_agent/04_control/03_coupling_window.md:85 (grounding rate uses Node 13)
  - docs/source/1_agent/10_appendices/01_derivations.md:1066-1072, 1121-1134 (nu_D, remark A.6.8)
  - docs/source/1_agent/07_cognition/03_memory_retrieval.md:1045-1057 (thm-causal-isometry)
  - docs/source/1_agent/09_economics/01_pomw.md:499, 801-806 (holographic verification cost, thm-minimum-friction-bft)
  - docs/source/1_agent/10_appendices/04_faq.md:11 and headings D.1.1-D.13.3
  - docs/source/1_agent/03_architecture/01_compute_tiers.md:1818, 1982, 2375; 04_control/01_exploration.md:38 (rb- admonition names)
  - docs/source/1_agent/03_architecture/02_disentangled_vae.md:342, 365 (unlabeled headings targeted by two references)
  - Section labels at 05_geometry/01_metric_law.md:1, 06_fields/03_info_bound.md:1, 08_multiagent/01_gauge_theory.md:1
  - Connection box inventory: `grep -rhoE "Connection to RL[^:]*#\s*[0-9]+" docs/source/1_agent`

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 1
- Minor: 8
- Notes: 3
- Primary themes: The chapter is almost entirely navigational. Its one formal object, the RL Degeneracy Theorem (thm-rl-degeneracy), is internally inconsistent: three limits in the statement against five in the proof and conclusion, a proof that cites 37 Connection boxes when 36 exist (numbers 32, 33, 37 absent and 35 used twice), a "partition" of the 37 table rows that covers 28 memberships with one row doubled and ten rows omitted, and one table row whose limit contradicts the theorem's own hypothesis. All numeric formulae quoted from upstream (screening mass, holographic coefficient nu_2 = 1/4, detection probability, f < N/3, O(sqrt N) verification, the parameter vector, the gauge group) were recomputed or checked against their sources and are consistent. The remaining findings are a broken link, two dangling section targets, inconsistent section numbering in the Book Map, the symbol lambda used for four different quantities on one page, a few summaries that generalise upstream theorems beyond their stated scope, and two stale counts (Sieve nodes, FAQ objections).

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | TL;DR, PoUW security table (line 122); What Is Novel item 21(d) (line 397) | Minor | Scope restriction | Framework | this chapter | Minimum Friction BFT presented as generic Byzantine tolerance; upstream scopes it to gradient poisoning and assumes classical BFT from PBFT |
| E-002 | TL;DR, Researcher Bridge Index (line 165) | Minor | Citation / reference error | Framework | this chapter | `rb-world-models` has no definition in the volume |
| E-003 | How to Read (line 198); Document Map, Detailed Section Guide (lines 523-547) | Minor | Citation / reference error | Framework | this chapter | Same label numbered 18 and 19, and 24.5 and 33; Section 29 listed under Part VII though its file is in Part VIII; "Part I through Part VIII" against nine parts |
| E-004 | LLM-Assisted Exploration, example queries (line 238) | Note | Citation / reference error | Framework | this chapter | Coupling Window paired with Node 15; the theorem and the chapter's own row 9 use Node 13 |
| E-005 | Main Advantages item 5 (line 325); Section Guide Part II (line 507); TL;DR (lines 25, 60, 490) | Minor | Parameter inconsistency | Framework | this chapter (count inherited from upstream 02_sieve/01_diagnostics.md:38, 48) | "60" and "60+" used for the same object while node anchors run to node-73 with node-67..70 defined twice |
| E-006 | Main Advantages item 7 (line 330); What Is Novel items 6, 7 (lines 364, 367); Table 0.6.1 rows 8, 18 (lines 600, 610) | Minor | Notation conflict | Framework | this chapter | lambda denotes discount rate, teleportation length, penalty multiplier, and Lyapunov rate on one page |
| E-007 | What Is Novel item 12 (line 376); Physics Inspiration (line 419) | Note | Miswording (secondary: Citation / reference error) | Framework | upstream 10_appendices/01_derivations.md:1121 | Area Law described as derived "via generalized Gauss-Bonnet identity"; the appendix uses a divergence identity and says Gauss-Bonnet is not required, but titles the remark "Gauss-Bonnet Generalization" |
| E-008 | What Is Novel item 13 (line 377) | Note | Scope restriction | Framework | upstream 07_cognition/03_memory_retrieval.md:1045-1057 | Causal Isometry summary (like the upstream statement) omits the uniqueness hypothesis the upstream proof relies on |
| E-009 | Comparison Snapshot (line 430); Reading guide (line 442) | Minor | Citation / reference error | Framework | this chapter | `sec-differential-geometry-view-curvature-as-conditioning` and `sec-literature-connections` are not defined anywhere |
| E-010 | For Skeptical Readers (line 472); Document Map (lines 497, 560) | Minor | Citation / reference error (secondary: Parameter inconsistency) | Framework | upstream 10_appendices/04_faq.md:11, repeated here | FAQ said to contain forty objections; it contains fifty (D.1.1 to D.13.3) |
| E-011 | Standard RL as the Degenerate Limit, thm-rl-degeneracy and Table 0.6.1 (lines 571-653) | Moderate | Proof gap / omission (secondary: Parameter inconsistency) | Framework | this chapter | Statement has three limits, proof and conclusion five; 36 boxes not 37; class lists are not a partition; row 5 contradicts hypothesis 2 |
| E-012 | The Five Degeneracy Classes, item 1 (line 633); Table 0.6.1 rows 4, 31, 35, 36 | Minor | Miswording (secondary: Conceptual) | Framework | this chapter | Geometric class attributes to G -> I four rows whose table limits are discretisation, finite N, independence, or decoupling |

## Detailed findings

### [E-001] Minimum Friction BFT generalised beyond its stated scope (was F-010)
- Location: TL;DR, "Proof of Useful Work (PoUW) — Key Security Properties" table, line 122; "What Is Novel" item 21(d), line 397
- Severity: Minor
- Type: Scope restriction
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 122: "| **Byzantine validators** | Minimum Friction BFT (Theorem {prf:ref}`thm-minimum-friction-bft`) | Tolerates $< 1/3$ adversarial validators |"; line 397: "(d) Minimum Friction BFT achieving Byzantine tolerance via geometric coherence".
- Upstream anchor: docs/source/1_agent/09_economics/01_pomw.md:801-806: "The Metric Friction Consensus achieves Byzantine Fault Tolerance against $f < N/3$ adversarial validators for **gradient-poisoning attacks** (adversaries submit incorrect gradients). **Scope:** This theorem addresses data integrity attacks (model poisoning, fake gradients). Classical BFT attacks (equivocation, censorship) are handled by the underlying stake-based leader election, which is assumed to follow standard PBFT guarantees."
- Why this is an error: The upstream theorem covers one attack class and explicitly assumes classical Byzantine tolerance from an external PBFT layer. The chapter labels the attack vector generically as "Byzantine validators" and credits the tolerance to geometric coherence, which is the part upstream does not prove. The 1/3 threshold, the detection bound (pomw.md:615) and the O(sqrt N) verification cost (pomw.md:499) are consistent with upstream.
- Impact on downstream results: Security claims in the Part IX summaries; no derivation depends on it.
- Fix guidance:
  1. Change the attack-vector cell at line 122 to "Gradient-poisoning validators".
  2. Append to the guarantee cell: "(classical equivocation/censorship tolerance assumed from the underlying PBFT layer)".
  3. In item 21(d), replace "achieving Byzantine tolerance via geometric coherence" with "achieving tolerance to f < N/3 gradient-poisoning validators via geometric coherence".
- Required new assumptions/permits: none.
- Validation plan: re-read pomw.md:801-806 after the edit and confirm the chapter wording is a subset of the upstream claim.

### [E-002] One Researcher Bridge index entry points to an undefined label (was F-001)
- Location: TL;DR, "Researcher Bridge Index (Quick Links)", line 165
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 165: "| World Models with Typed Latents | {ref}`Link <rb-world-models>` |"
- Upstream anchor: no `(rb-world-models)=`, `:name: rb-world-models`, or `:label: rb-world-models` exists under docs/source/1_agent. The nearest bridge is "Researcher Bridge: World Models as Numerical Integrators" at docs/source/1_agent/08_multiagent/05_architecture.md:108, which has a different name and subject.
- Why this is an error: The link cannot resolve. The other four entries the stage-1 review flagged (rb-hyperbolic-hierarchy, rb-renormalization-resnets, rb-jump-operators, rb-maxent-exploration) do resolve: they are defined as `:name:` fields on admonitions at 03_architecture/01_compute_tiers.md:1818, 1982, 2375 and 04_control/01_exploration.md:38.
- Impact on downstream results: navigation only.
- Fix guidance:
  1. Locate the intended "World Models with Typed Latents" bridge (likely in 04_control/02_belief_dynamics.md or 03_architecture/02_disentangled_vae.md); if it exists, add `:name: rb-world-models` to it.
  2. Otherwise delete the row at line 165.
- Required new assumptions/permits: none.
- Validation plan: build the docs and confirm no unresolved-reference warning for `rb-world-models`.

### [E-003] Book Map numbering and Part assignments are inconsistent (was F-003)
- Location: "How to Read", line 198; "Detailed Section Guide", lines 523, 526, 534, 541, 545
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 523: "**{ref}`Section 18 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`**"; line 526: "**{ref}`Section 19 <sec-capacity-constrained-metric-law-geometry-from-interface-limits>`**"; line 534: "**{ref}`Section 24.5 <sec-causal-information-bound>`**"; line 545: "**{ref}`Section 33 <sec-causal-information-bound>`**"; line 541 (under Part VII): "**{ref}`Section 29 <sec-symplectic-multi-agent-field-theory>`**"; line 198: "Sequential reading from Part I through Part VIII"; line 486: "The document is organized into nine parts plus appendices".
- Upstream anchor: docs/source/1_agent/05_geometry/01_metric_law.md:1 `(sec-capacity-constrained-metric-law-geometry-from-interface-limits)=` (Part V per this chapter's TOC, line 270); docs/source/1_agent/06_fields/03_info_bound.md:1 `(sec-causal-information-bound)=` (Part VI, line 278); docs/source/1_agent/08_multiagent/01_gauge_theory.md:1 `(sec-symplectic-multi-agent-field-theory)=` (Part VIII, line 292; Modularity table line 216 "Multi-agent theory | Part VIII Ch. 01–03").
- Why this is an error: The same label carries two different section numbers in two places, and Section 29 is listed under Part VII although the chapter's own TOC and Modularity table put the file in Part VIII. The reading instruction omits Part IX, which the same chapter says exists.
- Impact on downstream results: Reader navigation; "Section N" numbers are used in cross-references throughout the volume, so duplicate numbering propagates confusion but no mathematics.
- Fix guidance:
  1. Delete the duplicate entry at line 526 (or renumber Part IV to end at Section 17 and keep Section 18 under Part V only).
  2. Delete the entry at line 534 (Section 24.5), keeping the info bound at Section 33 under the Part it actually belongs to, or move it to Part VI consistently.
  3. Move the Section 29 entry (line 541) under Part VIII and adjust the Part VII/VIII header ranges.
  4. Change "Part VIII" to "Part IX" at line 198.
- Required new assumptions/permits: none.
- Validation plan: check that each label appears exactly once in the Detailed Section Guide and that its Part matches the TOC at lines 248-309.

### [E-004] Example query pairs the Coupling Window with the wrong Sieve node (was F-011)
- Location: "LLM-Assisted Exploration", example queries, line 238
- Severity: Note
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "Explain how the Sieve Node 15 relates to the Coupling Window Theorem"
- Upstream anchor: docs/source/1_agent/04_control/03_coupling_window.md:85: "Let $G_t:=I(X_t;K_t)$ be the symbolic mutual information injected through the boundary (Node 13)"; docs/source/1_agent/02_sieve/01_diagnostics.md:111-113: Node 13 = BoundaryCheck, Node 15 = SNRCheck ("SNR < epsilon").
- Why this is an error: The window theorem is stated in terms of Node 13 (and H(K), Node 11). Node 15 is the boundary signal-to-noise check. The chapter's own Table 0.6.1 row 9 (line 601) correctly says "Coupling Window (Node 13)".
- Impact on downstream results: none.
- Fix guidance: replace "Node 15" with "Node 13".
- Required new assumptions/permits: none.
- Validation plan: none needed beyond the edit.

### [E-005] Sieve node count stated as both 60 and 60+ while the numbered set runs to 73 (was F-006)
- Location: "Main Advantages" item 5, line 325; Detailed Section Guide Part II, line 507; TL;DR lines 25 and 60; Document Map line 490
- Severity: Minor
- Type: Parameter inconsistency
- Criterion: Framework
- Origin: this chapter (count inherited from upstream docs/source/1_agent/02_sieve/01_diagnostics.md:38, 48)
- Claim (verbatim): line 325: "it decomposes into 60 explicit checks"; line 507: "The 60 diagnostic nodes"; line 25: "The Sieve (60+ Runtime Checks)"; line 60: "The Sieve: 60+ explicit monitors"; line 490: "the Sieve (60+ runtime diagnostics)".
- Upstream anchor: docs/source/1_agent/02_sieve/01_diagnostics.md:38: "Each of the 60 nodes is a mathematical contract"; :48: "PASS if all 60 diagnostics pass". Node anchors beyond 60: `(node-61)=` 06_fields/02_reward_field.md:1371; `(node-62)=`..`(node-66)=` 08_multiagent/01_gauge_theory.md:1047-2842; `(node-67)=`, `(node-68)=` at both 07_cognition/07_metabolic_transducer.md:1148, 1167 and 08_multiagent/05_architecture.md:2083, 2097; `(node-69)=`, `(node-70)=` at both 07_cognition/08_intersubjective_metric.md:1352, 1368 and 08_multiagent/05_architecture.md:2111, 2125; `(node-71)=`..`(node-73)=` 07_cognition/09_retrieval_attention.md:1272-1302.
- Why this is an error: The chapter uses two different counts for the same object, and the upstream PASS predicate is defined over "all 60 diagnostics" although the volume numbers checks up to 73, with four numbers each assigned to two different checks. Which checks constitute the gate is therefore ambiguous.
- Impact on downstream results: The PASS/BLOCK definition (01_diagnostics.md:48) and the "known compute cost" claim at line 325.
- Fix guidance:
  1. Resolve the duplicate node-67..70 labels upstream (rename one set, e.g. the 08_multiagent ones).
  2. Fix a single count N of gate nodes upstream in 01_diagnostics.md:38, 48 (either 60 core nodes with later nodes listed as extensions, or the full numbered set).
  3. Use one phrase in this chapter at lines 25, 60, 325, 490, 507 consistent with that choice.
- Required new assumptions/permits: none.
- Validation plan: grep for "60 " and "60+" in this chapter after the edit; confirm each `node-N` label is defined once.

### [E-006] The symbol lambda denotes four different quantities on one page (was F-007)
- Location: "Main Advantages" item 7, line 330; "What Is Novel" items 6 and 7, lines 364 and 367; Table 0.6.1 rows 8 and 18, lines 600 and 610
- Severity: Minor
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 330 and 367: "$\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln\gamma / \Delta t$"; line 364: "via the teleportation length $\lambda$"; line 600: "Soft $\lambda$-penalty"; line 610: "Neural Lyapunov Constraint $\dot{V} \le -\lambda V$".
- Upstream anchor: docs/source/1_agent/06_fields/02_reward_field.md:326: "Let the temporal discount rate be $\lambda := -\ln\gamma / \Delta t$"; docs/source/1_agent/05_geometry/02_wfr_geometry.md:358: "**Teleportation length**: $\lambda$ determines when transport beats reaction"; docs/source/1_agent/01_foundations/02_control_loop.md:598: "$\dot{V}(z) \le -\lambda_{\text{Lyap}} V(z), \quad [\lambda_{\text{Lyap}}] = s^{-1}$".
- Why this is an error: Upstream distinguishes the Lyapunov rate as lambda_Lyap; the chapter drops the subscript in row 18 and, on the same page, uses bare lambda for the discount rate (units 1/time), the WFR teleportation length (units length) and a Lagrange multiplier. Items 6 and 7 are three lines apart, so kappa = lambda / c_info reads as dimensionally wrong if lambda is taken from item 6.
- Impact on downstream results: none mathematically; each upstream chapter is internally consistent.
- Fix guidance:
  1. Write $\lambda_{\text{Lyap}}$ in row 18 (line 610).
  2. Write "teleportation length $\lambda_{\text{WFR}}$" (or name the symbol upstream and use it) at line 364.
  3. Write "soft Lagrange-multiplier penalty" at line 600.
  4. Keep bare lambda only for the discount rate (lines 330, 367).
- Required new assumptions/permits: none.
- Validation plan: search the chapter for `\lambda` and confirm each occurrence names one quantity.

### [E-007] Area Law described as derived via a "generalized Gauss-Bonnet identity" (was F-008)
- Location: "What Is Novel" item 12, line 376; "Physics Inspiration", line 419
- Severity: Note
- Type: Miswording (secondary: Citation / reference error)
- Criterion: Framework
- Origin: upstream docs/source/1_agent/10_appendices/01_derivations.md:1121 (the chapter repeats the upstream remark's title)
- Claim (verbatim): line 376: "Derived rigorously from the Capacity-Constrained Metric Law via generalized Gauss-Bonnet identity."; line 419: "the Area Law is proven from first principles in Appendix A.6 using the Capacity-Constrained Metric Law and generalized Gauss-Bonnet identity".
- Upstream anchor: docs/source/1_agent/10_appendices/01_derivations.md:1121-1134: ":::{prf:remark} A.6.8 (Gauss-Bonnet Generalization) ... uses the **Einstein tensor divergence identity** ... which is valid in **arbitrary dimension**. This is more general than the classical 2D Gauss-Bonnet theorem ... The Chern-Gauss-Bonnet theorem ... is not required here—we compute information capacity, not topology. The divergence theorem approach generalizes to any $D \geq 2$ without modification."
- Why this is an error: The tool named in the appendix is a divergence identity, and the appendix says Gauss-Bonnet is not required; calling it a "generalized Gauss-Bonnet identity" misidentifies the mechanism. The phrase, however, is the upstream remark's own title and gloss, so the chapter is summarising its source faithfully; the correction belongs upstream with the chapter following. The coefficient is correct: with Omega_1 = 2 pi, nu_2 = (2-1) * 2 pi / (8 pi) = 1/4, matching 01_derivations.md:1072.
- Impact on downstream results: reader expectations about the proof only.
- Fix guidance:
  1. Upstream: retitle remark A.6.8 (e.g. "Dimension-independence of the boundary conversion") and keep its content.
  2. Here: replace "via generalized Gauss-Bonnet identity" with "via the divergence identity used in Appendix A.6" at lines 376 and 419.
- Required new assumptions/permits: none.
- Validation plan: confirm the two chapter phrases match the retitled remark.

### [E-008] Causal Isometry summary omits the uniqueness hypothesis used in the upstream proof (was F-009)
- Location: "What Is Novel" item 13, line 377
- Severity: Note
- Type: Scope restriction
- Criterion: Framework
- Origin: upstream docs/source/1_agent/07_cognition/03_memory_retrieval.md:1045-1057
- Claim (verbatim): "The Causal Isometry Theorem ({prf:ref}`thm-causal-isometry`) proves that Interventionally Closed representations in different modalities induce isometric metrics, enabling principled cross-modal transfer."
- Upstream anchor: docs/source/1_agent/07_cognition/03_memory_retrieval.md:1048-1057: statement "If both representations are **Interventionally Closed** ..., then the induced metrics $G_A$ and $G_B$ are isometric."; proof step 4: "**Uniqueness:** Assuming the solution to the metric field equation is unique (guaranteed for the Poincare disk ansatz in the saturation limit), the geometries $G_A$ and $G_B$ are identical up to a diffeomorphism".
- Why this is an error: The upstream proof needs a hypothesis (uniqueness of the metric-law solution) that the upstream statement does not carry; the chapter reproduces the upstream statement, so the omission originates upstream. It is recorded here so the summary is updated once the upstream statement is repaired.
- Impact on downstream results: cross-modal retrieval justification in 07_cognition/03_memory_retrieval.md.
- Fix guidance:
  1. Upstream: add "under uniqueness of the metric-law solution (e.g. the saturated Poincare-disk ansatz)" to the theorem statement.
  2. Here: mirror the added hypothesis in item 13.
- Required new assumptions/permits: uniqueness of the metric field equation solution, to be stated upstream.
- Validation plan: compare item 13 with the revised upstream statement.

### [E-009] Two section references have no target (was F-002)
- Location: "Comparison Snapshot", row "Natural gradient / trust region", line 430; "Reading guide (connections by section)", line 442
- Severity: Minor
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 430: "{ref}`9.10 <sec-differential-geometry-view-curvature-as-conditioning>`"; line 442: "{ref}`9.7–9.9 <sec-literature-connections>`".
- Upstream anchor: no `(sec-differential-geometry-view-curvature-as-conditioning)=` or `(sec-literature-connections)=` exists under docs/source/1_agent. The intended headings exist without labels at docs/source/1_agent/03_architecture/02_disentangled_vae.md:342 "## Literature Connections (Mapping + Differences)" and :365 "## Differential-Geometry View (No Physics): Curvature as Conditioning". No heading auto-anchor setting is configured in docs/myst.yml or docs/_config.yml, and an auto-slug would not carry the `sec-` prefix in any case. The same first target is also used at 05_geometry/01_metric_law.md:40 and 10_appendices/04_faq.md:43.
- Why this is an error: both pointers fail to resolve; the "9.7-9.10" numbers do not correspond to any existing anchor.
- Impact on downstream results: navigation only.
- Fix guidance:
  1. Add `(sec-literature-connections)=` above 02_disentangled_vae.md:342 and `(sec-differential-geometry-view-curvature-as-conditioning)=` above :365.
  2. Update the displayed section numbers at lines 430 and 442 to the current numbering of those subsections.
- Required new assumptions/permits: none.
- Validation plan: docs build without unresolved-reference warnings for these two labels (this also repairs metric_law.md:40 and faq.md:43).

### [E-010] FAQ objection count stated as forty; the appendix contains fifty (added by verifier, V-001)
- Location: "For Skeptical Readers", line 472; Document Map, lines 497 and 560
- Severity: Minor
- Type: Citation / reference error (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: upstream docs/source/1_agent/10_appendices/04_faq.md:11, repeated in this chapter
- Claim (verbatim): line 472: "addresses forty such objections head-on"; line 497: "FAQ (40 objections)"; line 560: "FAQ—40 rigorous objections and responses".
- Upstream anchor: docs/source/1_agent/10_appendices/04_faq.md:11: "This appendix addresses forty rigorous objections". Headings D.1.1 through D.13.3 in the same file number 4+4+3+4+5+3+4+4+3+3+4+6+3 = 50 objections across thirteen themes.
- Why this is an error: the stated count is stale; the chapter's theme list (lines 473-479) also covers only D.1-D.5 of thirteen themes.
- Impact on downstream results: navigation only.
- Fix guidance:
  1. Replace "forty"/"40" with the actual count at lines 472, 497, 560 and at 04_faq.md:11.
  2. Optionally extend the theme bullets at lines 473-479 to mention D.6-D.13.
- Required new assumptions/permits: none.
- Validation plan: recount `### D.n.m` headings after any FAQ edit.

### [E-011] thm-rl-degeneracy: statement, proof, class partition, and reduction table are mutually inconsistent (was F-004)
- Location: "Standard RL as the Degenerate Limit", lines 571-653
- Severity: Moderate
- Type: Proof gap / omission (secondary: Parameter inconsistency)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 571: "Standard RL emerges from the Fragile Agent when three degeneracy conditions are imposed"; lines 576-580: "Standard Reinforcement Learning is recovered from the Fragile Agent framework under the joint limit: $\lim_{G \to I,\ |\mathcal{K}| \to \infty,\ \Xi_{\text{crit}} \to \infty}$ Fragile Agent"; line 583: "**Infinite Capacity** ($|\mathcal{K}| \to \infty$): No information bottleneck, continuous state space without quantization"; line 586: "*Proof.* Each of the 37 Connection boxes below demonstrates a specific reduction. The composite limit follows from the independence of the five degeneracy conditions. $\square$"; line 597 (row 5): "Tabular Q-Learning | VQ-VAE macro-register $K$ | $\lvert\mathcal{K}\rvert = \lvert\mathcal{S}\rvert$, encoder = identity"; line 653: "Standard RL and traditional blockchain consensus are recovered from the Fragile Agent under five degeneracy conditions: flat geometry, infinite capacity, disabled Sieve, zero metabolic cost, and useless consensus computation."
- Upstream anchor: defined in this chapter. Box inventory: `grep -rhoE "Connection to RL[^:]*#\s*[0-9]+" docs/source/1_agent` returns numbers {1..31, 34, 35, 35, 36}: 36 numbered boxes; #32, #33, #37 do not exist; #35 is used at both docs/source/1_agent/07_cognition/07_metabolic_transducer.md:764 ("Intrinsic Motivation as Battery-Independent Limit") and docs/source/1_agent/08_multiagent/05_architecture.md:396 ("Gauge-Covariant World Models"). No boxes under alternative names exist in 08_multiagent/02_standard_model.md, 03_parameter_sieve.md, or 09_economics/01_pomw.md.
- Why this is an error:
  1. Statement/proof mismatch: the theorem states three limits (also line 100 and line 571) but the proof appeals to "the five degeneracy conditions" and the Conclusion asserts five (adding T_c -> 0 and the consensus replacement). A theorem whose proof requires hypotheses absent from its statement is not proved as stated.
  2. Independence is asserted, not shown: nothing establishes that the limits commute. T_c -> 0 interacts with Xi_crit, since the hysteresis threshold is derived from Landauer/T_c (thm-thermodynamic-hysteresis-bound, cited at line 339).
  3. The proof cites "37 Connection boxes" but 36 exist; rows 32, 33, 37 of Table 0.6.1 have no supporting box, and two different reductions share #35.
  4. The class lists at lines 633-641 are not a partition of rows 1-37. Recomputed: class 1 = {1,2,3,4,18,20,24,27,30,31,35,36}, class 2 = {5,6,11,12,13,25,28,29}, class 3 = {8,9,12,19,23}, class 4 = {14,34}, class 5 = {37}; 28 memberships, row 12 in classes 2 and 3, rows {7, 10, 15, 16, 17, 21, 22, 26, 32, 33} in none. The composite argument does not cover ten rows.
  5. Row 5 recovers tabular Q-learning at |K| = |S| with an identity encoder, a finite-codebook special case, which is incompatible with hypothesis 2's gloss "continuous state space without quantization" under |K| -> infinity.
- Impact on downstream results: This is the organizing claim for every "Connection to RL #N" box and is repeated in the TL;DR (line 100) and the Conclusion (line 653). No later derivation uses it quantitatively, so Moderate rather than Major.
- Fix guidance:
  1. State the theorem with the five conditions actually used, or restrict both proof and Conclusion to the three-condition RL statement and treat rows 34-37 as separate remarks.
  2. Replace "independence" with the explicit claim "each row's reduction uses only the conditions of its class, so the limits may be taken in any order", or drop the sentence.
  3. Change "37 Connection boxes" to the true count; add boxes #32, #33, #37 at the sections the table cites (34.1, 35, Part IX) and renumber one of the two #35 boxes.
  4. Rewrite the class lists so every row 1-37 appears exactly once (assign rows 7, 10, 15, 16, 17, 21, 22, 26, 32, 33; keep row 12 in one class).
  5. Either drop "continuous state space" from hypothesis 2 or note that tabular RL is the finite-|K| identity-encoder special case rather than a |K| -> infinity limit.
- Required new assumptions/permits: none beyond making the hypotheses explicit.
- Framework-first proof sketch for the fix: with the five conditions stated and each row assigned to exactly one class, the "composite limit" is the statement that each row's reduction is a function of its own class's parameter only; commutation then follows trivially because the limits act on disjoint parameters, provided the T_c/Xi_crit dependence is either broken (state Xi_crit -> infinity before T_c -> 0) or acknowledged.
- Validation plan: re-run the box inventory grep and the class-membership count after the edit (expect 37 boxes with distinct numbers and 37 memberships).

### [E-012] Geometric Degeneracy class attributes to G -> I limits that are not flattenings (was F-005)
- Location: "The Five Degeneracy Classes", item 1, line 633; Table 0.6.1 rows 4, 31, 35, 36 (lines 596, 623, 627, 628)
- Severity: Minor
- Type: Miswording (secondary: Conceptual)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 633: "**Geometric Degeneracy** (Rows 1–4, 18, 20, 24, 27, 30–31, 35–36): Setting $G \to I$ flattens the manifold. ... the Helmholtz PDE becomes the Bellman recursion; ... mean-field limits reduce to finite-agent games; intersubjective metric loses geometric grounding; economic dynamics decouple from geometry."
- Upstream anchor: docs/source/1_agent/06_fields/02_reward_field.md:326-335 (thm-the-hjb-helmholtz-correspondence): "The Bellman condition ... approaches the following PDE in the limit $\Delta t \to 0$: $-\Delta_G V(z) + \kappa^2 V(z) = \rho_r(z)$ where $\Delta_G$ ... is the Laplace-Beltrami operator on the manifold $(\mathcal{Z}, G)$".
- Why this is an error: The table itself gives row 4's limit as "Discretize lattice", row 31 as "Finite N", row 35 as "Independent representations", row 36 as "Decouple economics from geometry"; none is G -> I. Upstream obtains the PDE from Bellman by a continuum limit on a manifold with arbitrary G, so the reverse recovery is a re-discretisation, valid for curved G, not a flattening.
- Impact on downstream results: only the narrative around thm-rl-degeneracy (E-011).
- Fix guidance:
  1. Reword item 1 so the G -> I sentence covers rows 1-3, 18, 20, 24, 27, 30 only.
  2. Move rows 4, 31, 35, 36 to a class named for their actual operations (discretisation / finite-N / decoupling), coordinated with the partition fix in E-011.
- Required new assumptions/permits: none.
- Validation plan: each row's class heading should name the same limit as the row's "Limit" column.

## Scope restrictions and clarifications
- The chapter is a front matter and positioning document; apart from thm-rl-degeneracy it defines no mathematical objects. All formulas it quotes (kappa = lambda / c_info with lambda = -ln gamma / Delta t; nu_D = (D-1) Omega_{D-1} / (8 pi) with nu_2 = 1/4; detection probability >= 1 - (1 - epsilon)^k; f < N/3; O(sqrt N) verification; Lambda = (c_info, sigma, ell_L, T_c, g_s, gamma); G_Fragile = SU(N_f)_C x SU(r)_L x U(1)_Y) were checked against their upstream sources and agree.
- Findings E-007, E-008 and E-010 originate upstream; they are logged here so the chapter's summaries are updated when the sources are fixed.
- The xref pre-pass used for this review does not index `:name:` labels on admonitions; the four Researcher Bridge links it reported as dangling at lines 161-163 and 166 resolve correctly.

## Open questions
- Which set of nodes constitutes the Sieve gate (60 core nodes, or the full numbered set through node-73) is a decision for 02_sieve/01_diagnostics.md; this chapter should follow it.
- Whether rows 32, 33 and 37 are meant to have Connection boxes (in 08_multiagent/02_standard_model.md, 03_parameter_sieve.md, and 09_economics/01_pomw.md) or should be dropped from the "37 boxes" count.

## Rejected candidate findings
- None. All eleven stage-1 findings were kept; F-001 was narrowed to a single dangling label, and F-008 and F-009 were downgraded to Notes with upstream origin.
