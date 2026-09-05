# Mathematical Review: docs/source/1_agent/02_sieve/03_failures_interventions.md

## Metadata
- Reviewed file: docs/source/1_agent/02_sieve/03_failures_interventions.md
- Review date: 2026-09-05
- Reviewer: Claude (stage-1 reviewer + independent verifier)
- Scope: full document (143 lines: TLDR, roadmap, failure-mode table, intervention table, commentary)
- Framework anchors (definitions/axioms/permits):
  - `docs/source/1_agent/01_foundations/02_control_loop.md:1478-1510` (`cor-boundary-filter-interpretation`, Modes B.E / B.D; HJB-interface coupling, Mode B.C)
  - `docs/source/1_agent/02_sieve/01_diagnostics.md:92-121` (Sieve node table: Nodes 2, 7c, 13, 14, 16, 20) and `:251-258` (regularizer-to-mode table)
  - `docs/source/1_agent/02_sieve/02_limits_barriers.md:160` (rate terms of the information-control objective)
  - `docs/source/1_agent/10_appendices/02_parameters.md` (parameter and threshold registry)
  - Cross-references to the mode codes in `03_architecture/01_compute_tiers.md:140,1901`, `04_control/02_belief_dynamics.md:229,241`, `04_control/03_coupling_window.md:136,154`, `07_cognition/03_memory_retrieval.md:571,752,756`, `07_cognition/05_metabolism.md:620`

## Executive summary
- Critical: 0
- Major: 0
- Moderate: 2
- Minor: 3
- Notes: 5
- Primary themes: The chapter is a taxonomy (two tables and commentary) with no proofs or derivations, so the defects are consistency defects rather than computational ones. (i) The intervention table contains a row for a mode, S.C, that is defined nowhere in Volume 1. (ii) The codes B.E and B.C are assigned different pathologies and different failed components in the control-loop chapter. (iii) The prose legend for the two-letter codes does not describe the codes actually used. (iv) Several intervention rows are internally inconsistent with the failure table (target component of SurgSE) or rely on thresholds introduced nowhere (SurgDC). The mechanical cross-reference pass reports no dangling references or duplicate labels for this file.

## Error log
| ID | Location | Severity | Type | Criterion | Origin | Short description |
|---|---|---|---|---|---|---|
| E-001 | Roadmap, lines 16-20 | Note | Citation / reference error | Framework | this chapter | Roadmap item 3 (failures vs Sieve nodes and barriers) has no corresponding section |
| E-002 | Prose legend, line 39 vs table lines 48-63 | Minor | Definition mismatch | Framework | this chapter | Two-letter code legend names three first letters; the table uses five, and the second-letter gloss does not decode the rows |
| E-003 | Failure table rows B.E, B.C, lines 61, 63 | Moderate | Definition mismatch | Framework | upstream 01_foundations/02_control_loop.md | B.E and B.C carry different meanings and failed components in the control-loop chapter |
| E-004 | Intervention table, SurgCE, line 112 | Note | Notation conflict | Framework | this chapter | Trust-region constraint uses an unspecified norm on policies |
| E-005 | Intervention table, SurgCD_Alt and SurgCD, lines 114, 117 | Note | Notation conflict | Framework | this chapter | Two surgeries target C.D under two names with no stated selector |
| E-006 | Intervention table, SurgSE, line 115 vs S.E, line 51 | Minor | Definition mismatch | Framework | this chapter | SurgSE target component (World Model) contradicts S.E failed component (Policy) |
| E-007 | Intervention table, SurgSC, line 116; prose line 131 | Moderate | Citation / reference error | Framework | this chapter | Target mode S.C is defined nowhere; the row has no trigger |
| E-008 | Intervention table, SurgDC, line 122 | Minor | Proof gap / omission | Framework | this chapter | Thresholds tau_n, tau_tex, tau_K are undefined and absent from the parameter appendix; canonical D.C detector not referenced |
| E-009 | Intervention table, SurgBE, line 124 | Note | Miswording | External | this chapter | "Saturation / Anti-Windup" mislabels a Lipschitz/gain bound |
| E-010 | Intervention table, SurgBD, line 125 vs B.D, line 62 | Note | Algorithm mismatch | Framework | this chapter | Replay addresses forgetting, which the B.D description does not mention |

## Detailed findings

### [E-001] Roadmap promises a section that does not exist (was F-001)
- Location: Roadmap, lines 16-20; TLDR, lines 13-14
- Severity: Note
- Type: Citation / reference error
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 20: "3. Worked interpretations: how failures relate to Sieve nodes and barrier surfaces."
- Upstream anchor: not applicable
- Why this is an error: The chapter contains the failure table (lines 48-63), the intervention table (lines 110-126), and commentary (lines 128-143). No section maps modes to Sieve nodes (`02_sieve/01_diagnostics.md:92-121`) or to barrier surfaces. The only node-level link is the SurgDE row's "Triggered by OscillateCheck / HolonomyCheck" (line 123). The partial mapping that does exist lives in `02_sieve/01_diagnostics.md:251-258` and `03_architecture/01_compute_tiers.md:140`.
- Impact on downstream results: None mathematically. Readers following the roadmap or the TLDR claim that the chapter "pairs with Diagnostics and Barriers" will not find the promised bridge.
- Fix guidance:
  1. Either add a short table (mode -> triggering node(s) -> barrier, if any) after the intervention table, drawing on `01_diagnostics.md:251-258`, or
  2. delete roadmap item 3 and soften the TLDR bullet at lines 13-14.
- Required new assumptions/permits: none.
- Validation plan: Confirm every roadmap item has a corresponding heading or table.

### [E-002] Prose legend for the two-letter codes does not match the codes used (was F-002)
- Location: line 39 vs failure table lines 48-63
- Severity: Minor
- Type: Definition mismatch (secondary: Miswording)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 39: "Each mode has a two-letter code: coordinates in "failure space." The first letter tells you the *type* of dynamics going wrong (D for dispersion, C for concentration, T for topology), and the second tells you the *direction* (spreading out, collapsing inward, or stuck)."
- Upstream anchor: not applicable; the table is defined in this chapter.
- Why this is an error: The table uses five first letters, not three: D, C, T and also S (S.E Subcritical-Equilib, S.D Struct-Dispersion) and B (B.E, B.D, B.C). The prose assigns no meaning to S or B. The second-letter gloss also fails to decode the rows: under any assignment of {D, E, C} to {spreading out, collapsing inward, stuck}, D.D ("Success (Convergence)") and C.D ("Mode Collapse") and T.D ("Glassy Freeze") share the same second letter but describe convergence, collapse, and freezing respectively; D.E is "Oscillatory" and C.C is "Event Accumulation (Zeno)". Several Standard Names (Glassy Freeze, Labyrinthine, Semantic Horizon, Sensitivity Expl., Resource Depletion, Control Deficit) are not of the form type-direction at all. The coordinate system described is therefore not the one used.
- Impact on downstream results: The codes are cited across the book as opaque identifiers (`03_architecture/01_compute_tiers.md:140,1901`, `04_control/02_belief_dynamics.md:229,241`, `04_control/03_coupling_window.md:136,154`, `07_cognition/03_memory_retrieval.md:571,752,756`, `07_cognition/05_metabolism.md:620`, `01_foundations/02_control_loop.md:1486-1487,1506`, `02_sieve/01_diagnostics.md:251-258,607,654`); none of them relies on the decoding. Impact is confined to the misleading explanation.
- Fix guidance:
  1. Either state the legend actually used (first letter in {D dispersion, C concentration, T topology, S structure/scaling, B boundary}; second letter in {D decay/dispersion, E escape/extension, C concentration/accumulation}) and adjust the Standard Name column so every row decomposes accordingly, or
  2. drop the claim that the letters are decodable coordinates and present the codes as labels.
- Required new assumptions/permits: none.
- Validation plan: Check each of the 14 rows against the stated legend.

### [E-003] Modes B.E and B.C are given different meanings in the control-loop chapter (was F-003)
- Location: failure table rows B.E and B.C, lines 61 and 63
- Severity: Moderate
- Type: Definition mismatch (secondary: Notation conflict)
- Criterion: Framework
- Origin: upstream `docs/source/1_agent/01_foundations/02_control_loop.md` (this chapter holds the defining table; the control-loop chapter uses the same codes with a different meaning)
- Claim (verbatim): line 61: "| **B.E** | Sensitivity Expl. | **Critic** | **Fragility** | Optimization for a single condition induces high sensitivity to perturbations. |"; line 63: "| **B.C** | Control Deficit | **Policy** | **Overwhelmed** | Disturbance more complex than controller (Ashby). |"
- Upstream anchor: `02_control_loop.md:1486`: "**Mode B.E (Injection):** Occurs when interface inflow exceeds the effective capacity of the manifold (Levin capacity), breaking the assumed operating regime."; `:1487`: "**Mode B.D (Starvation):** Occurs when interface inflow is too weak, causing the internal information volume to decay (catastrophic forgetting)."; `:1506`: "If the internal $V$ near the boundary does not match external feedback, the agent enters **Mode B.C (Control Deficit)**---its internal model may be self-consistent but poorly aligned with the task-relevant data stream."
- Why this is an error: The code B.E denotes two unrelated pathologies. Here (and in `02_sieve/01_diagnostics.md:254`, "Spectral (Lipschitz) barrier ... Mode B.E (fragility)") it is a Critic sensitivity failure remedied by spectral normalization (SurgBE, line 124). In the control-loop corollary it is a Boundary overload (inflow exceeding capacity, paired with B.D starvation as the two ends of the boundary-coupling range monitored by Nodes 13-16). The two have different failed components and different remedies. For B.C the name "Control Deficit" is shared, but the control-loop chapter describes a value/boundary alignment mismatch (what Node 16 AlignCheck, a Critic node, monitors), whereas this table assigns an Ashby requisite-variety deficit to the Policy with the remedy "Width Expansion". B.D is consistent in both places. The B.E conflict is unambiguous; the B.C conflict is a component mismatch (Critic vs Policy).
- Impact on downstream results: A reader following "Mode B.E" from `cor-boundary-filter-interpretation` to this chapter is directed to SurgBE (spectral normalization of the critic), which does not address input overload; the natural node for overload is InputSaturationCheck (Node 14). No theorem depends on the code, so the impact is operational rather than logical.
- Fix guidance:
  1. In `02_control_loop.md:1486`, stop reusing the code B.E for the overload case: either introduce a distinct code (for example "B.O (Overload)", tied to Node 14) or add an "Injection / Overload | Boundary" row to this table under a new code.
  2. Align the B.C wording in one of the two places: either "value-boundary misalignment | Critic | AlignCheck" or "requisite-variety deficit | Policy".
  3. Add a one-line cross-reference from the control-loop corollary to `sec-failure-modes`.
- Required new assumptions/permits: none.
- Validation plan: `grep -rn "Mode B\.[EDC]" docs/source/1_agent` and confirm each hit agrees with the table's component and pathology.

### [E-004] SurgCE trust-region constraint uses an unspecified norm on policies (was F-004)
- Location: intervention table, SurgCE, line 112
- Severity: Note
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "enforce $\Vert \pi_{new} - \pi_{old} \Vert < \delta$"
- Upstream anchor: `02_sieve/01_diagnostics.md:94` (Node 2 ZenoCheck): "$D_{\mathrm{KL}}(\pi_t \Vert \pi_{t-1})$ (Smoothness)"; `:258`: "natural-gradient / trust region with state metric $G$".
- Why this is an error: Policies are conditional distributions (or parameter vectors); a bare norm on their difference is undefined without naming the space and the norm. The book's own trust-region machinery uses KL divergence or the metric $G$. The row is a mechanism sketch, so this is a clarity issue only.
- Impact on downstream results: None.
- Fix guidance: Write the constraint as $D_{\mathrm{KL}}(\pi_{\text{old}}\Vert\pi_{\text{new}})<\delta$ (or a $d_G$-ball), matching Node 2 and the metric law.
- Required new assumptions/permits: none.
- Validation plan: Local read.

### [E-005] Two surgeries target C.D under two different names (was F-005)
- Location: intervention table, SurgCD_Alt (line 114) and SurgCD (line 117)
- Severity: Note
- Type: Notation conflict
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "| **SurgCD_Alt** | C.D (Obsession) | **Policy** | ..." and "| **SurgCD** | C.D (Collapse) | **Shutter** | ..."
- Upstream anchor: line 52 of this chapter: "| **C.D** | Conc-Dispersion | **Policy/Shutter** | **Mode Collapse / Obsession** | ..."
- Why this is an error: The failure table defines one mode C.D with two candidate failed components. The intervention table splits it by component and labels the halves "Obsession" and "Collapse" as if they were distinct modes, without a criterion for choosing between them. The reading rule (line 97) says the surgery ID is "Surg" plus the mode code, so two rows for one code is ambiguous about which trigger selects which.
- Impact on downstream results: None.
- Fix guidance: State the selector explicitly, for example "C.D with policy-entropy collapse (Nodes 7b/10) -> SurgCD_Alt; C.D with codebook or dead-fibre collapse (Nodes 3/11) -> SurgCD".
- Required new assumptions/permits: none.
- Validation plan: Local read.

### [E-006] SurgSE target component contradicts the failure table (was F-006)
- Location: intervention table, SurgSE, line 115 vs failure table row S.E, line 51
- Severity: Minor
- Type: Definition mismatch (secondary: Algorithm mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 115: "| **SurgSE** | S.E (Stumble) | **World Model** | **Curriculum Ease-off** | **Curriculum Learning:** Reduce Task Difficulty or Rewind to earlier level. |"; line 51: "| **S.E** | Subcritical-Equilib | **Policy** | **Curriculum Stumble** | Task difficulty increases faster than adaptation rate. |"
- Upstream anchor: not applicable.
- Why this is an error: The failure table localises S.E in the Policy; the intervention table says the surgery operates on the World Model; the mechanism (change the task schedule) operates on neither. By the chapter's own reading rule (line 101, "Target Component: which part of the system you are operating on"), the row is internally inconsistent. All other rows keep the target component within the failure row's component set.
- Impact on downstream results: None beyond this chapter.
- Fix guidance: Set the target component to "Policy (via task schedule)" or "Environment / curriculum", consistent with line 51.
- Required new assumptions/permits: none.
- Validation plan: Check each intervention row's target component against the corresponding failure row.

### [E-007] SurgSC targets a mode S.C that is defined nowhere (was F-007)
- Location: intervention table, SurgSC, line 116; prose, line 131
- Severity: Moderate
- Type: Citation / reference error (secondary: Definition mismatch)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 116: "| **SurgSC** | S.C (Instability) | **Critic** | **Parameter Freezing** | **Target Network Freeze:** Stop updating Target V; switch to slower exponential moving average. |"; line 131: "SurgSC freezes target networks."
- Upstream anchor: `grep -rn -E "\bS\.C\b|SurgSC" docs/source/1_agent` returns only these two lines. The failure table (lines 48-63) has fourteen rows and no S.C. `02_sieve/01_diagnostics.md:103` defines Node 7c "CheckSC ($\mathrm{SC}_{\partial c}$) | Critic | New Mode Viability | $\text{Var}(V(z'))$", a check name in the SC family, not a failure mode.
- Why this is an error: The chapter's rule (line 97) is that a surgery ID is "Surg" plus a failure-mode code from the table. S.C is not in the table, so SurgSC has no trigger and "S.C (Instability)" is a dangling reference. The researcher-bridge claim (line 91) that "each intervention is triggered by a specific diagnostic condition" is false for this row. Note that Node 7c is a Critic check with a variance-of-value signature, which is a natural trigger for freezing the target value network; the row is missing its mode, not its diagnostic.
- Impact on downstream results: The intervention cannot be dispatched by any diagnostic because no signature is defined for S.C. No other chapter references S.C.
- Fix guidance:
  1. Preferred: add an S.C row to the failure table, for example "S.C | Critic Instability | Critic | Target Chasing | Bootstrapped value target drifts faster than the critic tracks it", with Node 7c ($\text{Var}(V(z'))$) named as the signature; or
  2. retarget the row to C.E (Divergence / Blow-up, Policy/Critic) as a second C.E surgery (SurgCE_Alt), parallel to the SurgCD/SurgCD_Alt pair, and update line 131 accordingly.
- Required new assumptions/permits: none.
- Validation plan: After the edit, every Target Mode code in the intervention table must appear in the failure table.

### [E-008] SurgDC thresholds are undefined and the canonical D.C detector is not referenced (was F-008)
- Location: intervention table, SurgDC, line 122
- Severity: Minor
- Type: Proof gap / omission (secondary: Citation / reference error)
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): "if nuisance surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_n\mid x)\Vert p(z_n))>\tau_n$) and/or texture surprisal spikes (e.g. $D_{\mathrm{KL}}(q(z_{\mathrm{tex}}\mid x)\Vert p(z_{\mathrm{tex}}))>\tau_{\mathrm{tex}}$) and/or macro surprisal spikes (e.g. $-\log p_\psi(K)>\tau_K$), trigger fallback (safe stop)."
- Upstream anchor: the three surprisal terms are per-sample versions of the rate terms in `02_sieve/02_limits_barriers.md:160`: "$\beta_K\,\mathbb{E}[-\log p_\psi(K)] + \beta_n D_{\mathrm{KL}}(q(z_n \mid x)\Vert p(z_n)) + \beta_{\mathrm{tex}} D_{\mathrm{KL}}(q(z_{\mathrm{tex}} \mid x)\Vert p(z_{\mathrm{tex}}))$". The symbols $\tau_n$, $\tau_{\mathrm{tex}}$, $\tau_K$ appear nowhere else in Volume 1 and are absent from `10_appendices/02_parameters.md`, whose stated purpose (line 6) is to centralize thresholds so they are "auditable across chapters". The canonical D.C detector elsewhere is $I(X;K)$ (Node 13 BoundaryCheck, `01_diagnostics.md:111`; `04_control/03_coupling_window.md:136`).
- Why this is an error: The formulas are correct uses of the anchored objects (units: nats). But the intervention is stated as a hard trigger on three thresholds introduced nowhere: no units, no default range, no calibration rule, no appendix entry, although the chapter says interventions are "triggered by a specific diagnostic condition rather than manual hyperparameter tuning" (line 91). The relation between the per-sample surprisal test and the mutual-information criterion used for D.C in the coupling-window chapter is not stated.
- Impact on downstream results: Any implementation of SurgDC must invent the thresholds; readers of `04_control/03_coupling_window.md` see a different D.C criterion from this table.
- Fix guidance:
  1. Add $\tau_n$, $\tau_{\mathrm{tex}}$, $\tau_K$ to `10_appendices/02_parameters.md` (units: nat; suggested calibration: a high quantile of training-distribution surprisal).
  2. In the SurgDC row, cite Node 13 / the coupling window as the primary D.C detector and present the surprisal test as the per-sample fallback trigger.
- Required new assumptions/permits: none.
- Validation plan: `grep -n "tau_" 10_appendices/02_parameters.md` shows the three entries; the row cross-references Node 13.

### [E-009] "Saturation / Anti-Windup" label does not describe spectral normalization (was F-009)
- Location: intervention table, SurgBE, line 124
- Severity: Note
- Type: Miswording (secondary: Conceptual)
- Criterion: External
- Origin: this chapter
- Claim (verbatim): "| **SurgBE** | B.E (Fragile) | **Critic** | **Saturation / Anti-Windup** | **Spectral Normalization:** Constrain Lipschitz constant of $V(z)$. |"
- Upstream anchor: `02_sieve/01_diagnostics.md:254`: "**Spectral (Lipschitz) barrier** | gain / sensitivity drift | spectral norm constraints (per-layer) | Mode B.E (fragility) | Bounds local gain"; `:117` (Node 20 LipschitzCheck, "Gain Control").
- Why this is an error: Anti-windup is a remedy for integrator windup under actuator saturation (bounding the accumulated integral state); spectral normalization bounds the operator norm of a function approximator. The category label points to the wrong class of fix. The column is explicitly a loose "conceptual category" (line 103) and the mechanism column is correct, so nothing can be mis-implemented; hence Note.
- Impact on downstream results: None.
- Fix guidance: Relabel as "Gain / Lipschitz bound", consistent with Node 20 and `01_diagnostics.md:254`.
- Required new assumptions/permits: none.
- Validation plan: Local read.

### [E-010] SurgBD mechanism addresses forgetting, which the B.D description does not mention (was F-010)
- Location: intervention table, SurgBD, line 125 vs failure table row B.D, line 62
- Severity: Note
- Type: Algorithm mismatch
- Criterion: Framework
- Origin: this chapter
- Claim (verbatim): line 125: "| **SurgBD** | B.D (Starve) | **Boundary/Shutter** | **Replay Buffer / Reservoir** | **Experience Replay:** Train on historical buffers to prevent catastrophic forgetting. |"; line 62: "| **B.D** | Resource Depletion | **Boundary/Shutter** | **Starvation** | Input or power resources depleted. |"
- Upstream anchor: `01_foundations/02_control_loop.md:1487`: "**Mode B.D (Starvation):** Occurs when interface inflow is too weak, causing the internal information volume to decay (catastrophic forgetting)."
- Why this is an error: The link between resource depletion and catastrophic forgetting is made only in the control-loop chapter. Within this chapter the surgery treats a symptom the mode description does not name, and the "power" half of the description has no mitigation. The pairing is coherent once the upstream sentence is imported.
- Impact on downstream results: None.
- Fix guidance: Append "(internal information volume decays: catastrophic forgetting)" to the B.D description, mirroring `02_control_loop.md:1487`, and either drop "power" or add a separate mitigation for it.
- Required new assumptions/permits: none.
- Validation plan: Local read.

## Scope restrictions and clarifications
- The chapter contains no theorems, proofs, or numeric derivations; the review therefore concerns internal consistency of the two tables, their agreement with the reading rules stated in the chapter, and their agreement with the uses of the same codes elsewhere in Volume 1.
- The mode codes are used elsewhere in the book only as identifiers; none of the downstream references depends on the legend at line 39 or on the intervention table, so no downstream result is invalidated by any finding here.
- All labels defined in this file (`sec-failure-modes`, `sec-interventions`, `rb-rl-pathologies`, `rb-heuristic-fixes`) are unique, and the only inbound reference (`07_cognition/03_memory_retrieval.md:756`) resolves correctly.

## Open questions
- Is S.C intended as a genuine fifteenth mode (critic target instability, with Node 7c as its signature), or is SurgSC meant to be a second C.E surgery? The answer determines which fix to apply in E-007.
- Should the boundary overload case in `cor-boundary-filter-interpretation` receive its own code and row, or is it meant to be subsumed under B.C? The answer determines the minimal fix for E-003.

## Rejected candidate findings
None. All ten stage-1 findings were confirmed or adjusted; three were adjusted (E-007 fix guidance, E-008 error type, E-009 severity downgraded from Minor to Note).
