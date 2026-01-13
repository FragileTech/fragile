# Volume 1 (`docs/source/1_agent`) review and improvement report

Date: 2026-01-13

This report reviews the **Volume 1** lecture sources under `docs/source/1_agent/` and proposes improvements and concrete
fixes. It includes **at least 5 actionable items per file** (37 files total, including one notebook).

## Cross-cutting priorities (high impact)

1. **Fix misplaced section labels (`(sec-...)=`).** 24 files end with a *label-only* line that clearly belongs to the
   next lecture/section (for example, `01_definitions.md` ends with `sec-the-control-loop-representation-and-control`).
   These labels currently make `{ref}` links land at the end of the *previous* page, not at the actual section.
   - Recommendation: move each misplaced label to **immediately precede the heading it names**, and remove the trailing
     label-only line from the previous file.
   - Follow-up: consider patching the label-insertion tooling (`docs/add_subsection_labels.py`,
     `docs/convert_section_refs.py`) so this doesn’t regress.

2. **Add “TLDR / learning objectives” to every lecture.** None of the Volume 1 markdown files have a `## TLDR` section,
   while later volumes do. A short “why this matters + what you will learn” block will significantly improve onboarding
   and skimmability.

3. **Reduce extremely long lines.** Many files have hundreds of lines over 120 characters (some with ~900–1000 character
   lines). This makes pages harder to read on narrow screens and makes diffs difficult to review.

4. **Strengthen the theory→implementation bridge.** Volume 1 repeatedly defines diagnostics/barriers/losses and even
   includes code snippets, but many chapters would benefit from explicit “where this lives in code” pointers (or a clear
   statement that the content is currently conceptual).

5. **Improve navigation for long chapters.** Several lectures are 1k–3.5k lines. They need an explicit “roadmap” and
   (often) a split into multiple pages to keep the book usable.

## Per-file recommendations

### `docs/source/1_agent/01_foundations/01_definitions.md` — *Introduction: The Agent as a Bounded-Rationality Controller*

**Snapshot:** ~453 lines; `##` sections: 1; long lines >120: ~110; ends with stray label `sec-the-control-loop-representation-and-control`.

1. Add `## TLDR` with 3–5 bullets (Markov blanket/interface re-typing, environment as kernel, latent split
   `(K, z_n, z_tex)`), plus prerequisites and what later chapters build on this.
2. Fix labels: add a file-level anchor before the H1 (so “Definitions” can be referenced cleanly), and move the trailing
   `sec-the-control-loop-representation-and-control` label to the start of
   `docs/source/1_agent/01_foundations/02_control_loop.md`.
3. Break the single large “Definitions” section into multiple `##` sections (interface, environment law, observation
   decomposition, action decomposition, gauge/symmetry), and add a small symbol glossary.
4. Reduce “forward-reference load”: where the chapter references later sections, add a short inline definition first,
   then move the deeper cross-reference into a “Where to go next” box.
5. Add a concrete worked example mapping a familiar POMDP into the boundary variables and the macro/nuisance/texture
   split; a 1-page example will make the abstractions feel less “floating”.
6. Reflow long paragraphs and long list items to reduce >120-char lines (especially in enumerations and long prose).
7. Add explicit pointers to the code (if applicable): where the boundary interface, latent split, and checks are
   implemented (or state clearly that Volume 1 is currently specification-only).

### `docs/source/1_agent/01_foundations/02_control_loop.md` — *The Control Loop: Representation and Control*

**Snapshot:** ~1412 lines; `##` sections: 13; long lines >120: ~233; ends with stray label `sec-diagnostics-stability-checks`.

1. Add `## TLDR` + a short “roadmap” (1 paragraph explaining how the 13 sections fit together).
2. Fix labels: add `(sec-the-control-loop-representation-and-control)=` before the H1 (this label currently sits at the
   end of `01_definitions.md`), and move `sec-diagnostics-stability-checks` to
   `docs/source/1_agent/02_sieve/01_diagnostics.md`.
3. Add a single figure showing the control loop wiring (Shutter → WM/Critic/Policy) and where diagnostics/barriers
   attach; this would reduce repeated verbal explanations.
4. Add pseudocode for: (a) one training step, (b) one inference step, with callouts (“diagnostics here”, “barriers here”).
5. Add a compact symbol/notation table up front (especially clarifying `G(z)` vs `F(θ)` since the chapter warns about
   “category errors”).
6. Reduce width of wide tables (convert to `list-table` / smaller tables), and wrap lines for better rendering.
7. Add “implementation hooks” beside each major component (Shutter/WM/Critic/Policy): what data structures / modules a
   reader should expect to exist in code.

### `docs/source/1_agent/02_sieve/01_diagnostics.md` — *Diagnostics: Stability Checks (Monitors)*

**Snapshot:** ~1020 lines; `##` sections: 6; long lines >120: ~269; ends with stray label `sec-limits-barriers`.

1. Add `## TLDR` explaining how to use the chapter operationally (online checks vs amortized/offline checks).
2. Fix labels: add `(sec-diagnostics-stability-checks)=` before the H1 (this label currently sits at the end of
   `02_control_loop.md`), and move `sec-limits-barriers` to
   `docs/source/1_agent/02_sieve/02_limits_barriers.md`.
3. Make the big stability-check table maintainable: generate it from a single source-of-truth (YAML/JSON/Python) so node
   names, formulas, and compute costs don’t drift across docs/code.
4. Improve table usability: split into per-category subtables, or add a compact summary table plus per-node collapsible
   details.
5. Add a worked example: show a monitor pipeline that triggers WARN/HALT/KILL, and link to the interventions chapter.
6. Explicitly map each node to the failure modes and the “first-line” intervention(s) that respond to it.
7. Reduce long line widths in the giant table (table cells currently include very long expressions that are hard to read).

### `docs/source/1_agent/02_sieve/02_limits_barriers.md` — *Limits: Barriers (The Limits of Control)*

**Snapshot:** ~194 lines; `##` sections: 1; long lines >120: ~59; has top label `sec-4-limits-barriers-the-limits-of-control`; ends with stray label `sec-failure-modes`.

1. Add `## TLDR` summarizing what a “barrier” is and the correct response pattern (halt/project/reshape), in 5–7 lines.
2. Remove the trailing `sec-failure-modes` label and place `(sec-failure-modes)=` before the H1 of
   `docs/source/1_agent/02_sieve/03_failures_interventions.md`.
3. Split the barrier table into “online” vs “offline/infeasible” subsets, and add a one-sentence “observable symptom”
   for each barrier (what the practitioner should see first).
4. Standardize regularizer notation and units, and add explicit links to Appendix B for anything with nontrivial units.
5. Add a concrete scenario walkthrough (one barrier hit, what diagnostics spike, what the controller does next).
6. Cross-link explicitly to `docs/source/1_agent/02_sieve/04_approximations.md` for barriers whose exact form is
   infeasible.
7. Reflow long table cells so the “Mechanism” and “Regularization Factor” columns don’t become unreadable.

### `docs/source/1_agent/02_sieve/03_failures_interventions.md` — *Failure Modes (Observed Pathologies)*

**Snapshot:** ~126 lines; `##` sections: 1; long lines >120: ~55; ends with stray label `sec-computational-considerations`.

1. Add `## TLDR`: the failure code taxonomy in 6–10 bullets, plus “how to use this table when debugging”.
2. Add `(sec-failure-modes)=` before the H1 (currently the only “failure modes” anchor is placed in the previous file).
3. Move/remove the trailing `sec-computational-considerations` label (it should label
   `docs/source/1_agent/03_architecture/01_compute_tiers.md`).
4. Add a short legend table for the two-letter codes and ensure the legend matches the codes used in the tables.
5. Add a mapping column: “which diagnostics nodes detect this first?” (ties failure modes to the monitoring system).
6. Add 2–3 short “case studies” (mode collapse, ungrounded inference, oscillation) showing symptoms → diagnosis → fix.
7. Split long intervention table cells (some include long conditionals and multiple thresholds) to improve readability.

### `docs/source/1_agent/02_sieve/04_approximations.md` — *Infeasible Implementation Replacements*

**Snapshot:** ~546 lines; `##` sections: 6; code blocks: many; long lines >120: ~48; ends with stray label `sec-the-disentangled-variational-architecture-hierarchical-latent-separation`.

1. Add `## TLDR`: the “top 10 replacements” and the nodes/barriers they replace.
2. Add `(sec-infeasible-implementation-replacements)=` before the H1 (this label is currently stranded at the end of
   `docs/source/1_agent/03_architecture/01_compute_tiers.md`).
3. Move/remove the trailing `sec-the-disentangled-variational-architecture-hierarchical-latent-separation` label (it
   belongs at the start of `docs/source/1_agent/03_architecture/02_disentangled_vae.md`).
4. Make code blocks self-consistent and runnable: standardize imports, clarify tensor shapes, and annotate any
   `retain_graph=True` usage (common source of confusion).
5. For each approximation, explicitly state: what it detects, how it fails (false pos/neg), and recommended default
   hyperparameters (probes/horizons/thresholds).
6. Replace “copy-paste” code with doc-driven includes or links to canonical implementations in `src/fragile/...`, to avoid
   drift between docs and code.
7. Add a short performance note per approximation (what it costs per step, what can be amortized).

### `docs/source/1_agent/03_architecture/01_compute_tiers.md` — *Computational Considerations*

**Snapshot:** ~2678 lines; `##` sections: 13; `###` sections: many; long lines >120: ~252; ends with stray label `sec-infeasible-implementation-replacements`.

1. Add `## TLDR` + a decision tree: “If you have X compute and Y safety requirements, use Tier N”.
2. Add `(sec-computational-considerations)=` before the H1 (the label currently sits at the end of
   `docs/source/1_agent/02_sieve/03_failures_interventions.md`).
3. Move/remove trailing `sec-infeasible-implementation-replacements` label to `docs/source/1_agent/02_sieve/04_approximations.md`.
4. Split the file: cost summaries, synchronization losses, tier definitions, and architecture (Atlas/TopoEncoder) can be
   separate pages to keep navigation workable.
5. Make wide tables render well: convert the widest tables to `list-table` (better wrapping) and define common variables
   (`B, Z, A, H, ...`) once per file.
6. Consolidate code: either point to canonical implementations, or put one complete reference implementation in an
   appendix; avoid dozens of partial snippets.
7. Add explicit cross-links from each tier to the diagnostics/barriers it includes, and to the approximation chapter for
   anything marked infeasible.

### `docs/source/1_agent/03_architecture/02_disentangled_vae.md` — *The Disentangled Variational Architecture: Hierarchical Latent Separation*

**Snapshot:** ~1483 lines; `##` sections: 12; long lines >120: ~145; ends with stray label `sec-intrinsic-motivation-maximum-entropy-exploration`.

1. Add `## TLDR`: what is enforced (closure + texture blindness) and what to measure (closure ratio, grounding windows).
2. Add `(sec-the-disentangled-variational-architecture-hierarchical-latent-separation)=` before the H1 (this label is
   currently stranded at the end of `docs/source/1_agent/02_sieve/04_approximations.md`).
3. Move/remove trailing `sec-intrinsic-motivation-maximum-entropy-exploration` label (it belongs at the start of
   `docs/source/1_agent/04_control/01_exploration.md`).
4. Add a single architecture diagram showing which modules can “see” macro/nuisance/texture; readers should not infer
   this from prose alone.
5. Add a step-by-step training checklist (loss schedule, closure weight, evaluation) and a short ablation plan (“what to
   remove first if training is unstable”).
6. Make code blocks either complete and minimal, or replace them with references to code. The current code density is
   high and drift-prone.
7. Reduce long-line issues in tables/dictionaries; keep “mapping to literature” and “dictionary” tables readable on page.

### `docs/source/1_agent/04_control/01_exploration.md` — *Intrinsic Motivation: Maximum-Entropy Exploration*

**Snapshot:** ~401 lines; `##` sections: 5; long lines >120: ~66; ends with stray label `sec-implementation-note-entropy-regularized-optimal-transport-bridge`.

1. Add `## TLDR` and explicitly define “causal entropy” in one line, plus the practical knobs (temperature, horizon).
2. Add `(sec-intrinsic-motivation-maximum-entropy-exploration)=` before the H1 (currently the label is stranded at the end
   of `docs/source/1_agent/03_architecture/02_disentangled_vae.md`).
3. Move/remove trailing `sec-implementation-note-entropy-regularized-optimal-transport-bridge` label to
   `docs/source/1_agent/04_control/03_coupling_window.md`.
4. Add a toy worked example (bandit/gridworld) contrasting reward-only vs MaxEnt exploration with a plotted behavior
   difference.
5. Clarify parameter naming and disambiguate this chapter’s entropy regularization from “cognitive temperature” used
   later; add a symbol table.
6. Add pseudocode for the objective and gradient estimator, and connect explicitly to SAC-style updates.
7. Add a “failure modes” subsection: when max-entropy causes dithering/chattering and how the Zeno/oscillation checks
   intervene.

### `docs/source/1_agent/04_control/02_belief_dynamics.md` — *Belief Dynamics: Prediction, Update, Projection*

**Snapshot:** ~421 lines; `##` sections: 6; long lines >120: ~87; ends with stray label `sec-duality-of-exploration-and-soft-optimality`.

1. Add `## TLDR`: “predict–update–project” and where the Sieve projection happens.
2. Add a file-level label before the H1 (for stable cross-referencing), since this chapter is frequently referenced.
3. Move/remove trailing `sec-duality-of-exploration-and-soft-optimality` label: attach it to the matching heading in
   `docs/source/1_agent/04_control/01_exploration.md` instead.
4. Add a small discrete example with explicit belief vectors/matrices and one full update step, including a sieve
   reweighting/projection.
5. Add a notation block (symbols used in this chapter) to reduce context switching to `reference.md`.
6. If the GKSL/Lindblad analogy is retained, add a clear mapping and a disclaimer; otherwise move it to an appendix.
7. Reflow long lines in the correspondence table; keep it readable on narrow screens.

### `docs/source/1_agent/04_control/03_coupling_window.md` — *Implementation Note: Entropy-Regularized Optimal Transport Bridge*

**Snapshot:** ~232 lines; `##` sections: 2; long lines >120: ~47; ends with stray label `sec-capacity-constrained-metric-law-geometry-from-interface-limits`.

1. Add `## TLDR` that states the theorem’s operational meaning (“what you compute and what it tells you”).
2. Add `(sec-implementation-note-entropy-regularized-optimal-transport-bridge)=` before the H1 (currently stranded at the
   end of `docs/source/1_agent/04_control/01_exploration.md`).
3. Move/remove trailing `sec-capacity-constrained-metric-law-geometry-from-interface-limits` label to
   `docs/source/1_agent/05_geometry/01_metric_law.md`.
4. Expand assumptions: add a short checklist of conditions needed for the coupling-window result to hold.
5. Add a toy numerical example computing the window from synthetic distributions (and show how it changes a regularizer).
6. Cross-link to the diagnostics/barriers that depend on the window (over/under coupling, grounding failures).
7. Reflow theorem statements and long inline math to reduce horizontal overflow.

### `docs/source/1_agent/05_geometry/01_metric_law.md` — *Capacity-Constrained Metric Law: Geometry from Interface Limits*

**Snapshot:** ~389 lines; `##` sections: 3; long lines >120: ~80; ends with stray label `sec-conclusion`.

1. Add `## TLDR` and prerequisites (“what you need to know about Riemannian metrics/variational principles”).
2. Add `(sec-capacity-constrained-metric-law-geometry-from-interface-limits)=` before the H1 (currently stranded at the end
   of `docs/source/1_agent/04_control/03_coupling_window.md`).
3. Move/remove trailing `sec-conclusion` label: it should live in `docs/conclusions.md` (or wherever the Conclusion section
   actually is), not at the end of this lecture.
4. Add a diagram showing boundary/interface constraints inducing a bulk metric law (this chapter is highly geometric).
5. Make the “implementation hook” concrete: list required tensors/statistics to compute the residual loss
   `\\mathcal{L}_{cap-metric}` and link to Appendix B for units.
6. Add a small worked example (even 2D) illustrating how the stationarity condition constrains the geometry.
7. Wrap long lines in theorem statements and long explanatory paragraphs.

### `docs/source/1_agent/05_geometry/02_wfr_geometry.md` — *Wasserstein-Fisher-Rao Geometry: Unified Transport on Hybrid State Spaces*

**Snapshot:** ~869 lines; `##` sections: 11; long lines >120: ~157; ends with stray label `sec-radial-generation-entropic-drift-and-policy-control`.

1. Add `## TLDR` explaining WFR (transport + reaction) and why it’s the canonical geometry for hybrid belief states.
2. Add `(sec-wasserstein-fisher-rao-geometry-unified-transport-on-hybrid-state-spaces)=` near the top so “Section 20”
   references land here (right now that label is defined in `docs/conclusions.md` with no content, which misdirects links).
3. Move/remove trailing `sec-radial-generation-entropic-drift-and-policy-control` label to
   `docs/source/1_agent/05_geometry/03_holographic_gen.md`.
4. Add a worked example contrasting product metrics vs WFR on a discrete+continuous toy system (one jump + diffusion).
5. Add “how to compute” guidance (entropic OT / cone-space approximation) and a short note on computational complexity.
6. Add a notation section clarifying `\\lambda` (teleportation length), measure space conventions, and sign conventions.
7. Reduce long lines by reflowing dense prose and splitting wide equations into multiline aligned blocks.

### `docs/source/1_agent/05_geometry/03_holographic_gen.md` — *Radial Generation: Entropic Drift and Policy Control*

**Snapshot:** ~767 lines; `##` sections: 4; long lines >120: ~125; ends with stray label `sec-the-equations-of-motion-geodesic-jump-diffusion`.

1. Add `## TLDR` and a short “position in the story” (after WFR, before the full EoM).
2. Add `(sec-radial-generation-entropic-drift-and-policy-control)=` before the H1 (currently stranded at the end of
   `05_geometry/02_wfr_geometry.md`).
3. Move/remove trailing `sec-the-equations-of-motion-geodesic-jump-diffusion` label to
   `docs/source/1_agent/05_geometry/04_equations_motion.md`.
4. Add an algorithm box/pseudocode (“radial generation step”) and specify what quantities the agent must provide.
5. Add a visual: 1D/2D plot showing entropic drift vs policy drift; this will greatly improve intuition.
6. Clarify relationship to diffusion models / score matching vs RL rollouts; include a short “similarities/differences”.
7. Reflow long paragraphs and split long inline equations.

### `docs/source/1_agent/05_geometry/04_equations_motion.md` — *The Equations of Motion: Geodesic Jump-Diffusion*

**Snapshot:** ~1305 lines; `##` sections: 8; long lines >120: ~196; ends with stray label `sec-the-boundary-interface-symplectic-structure`.

1. Add `## TLDR` and a roadmap: state the final EoM early and then explain components; the file is long.
2. Add `(sec-the-equations-of-motion-geodesic-jump-diffusion)=` before the H1 (currently stranded at the end of
   `05_geometry/03_holographic_gen.md`).
3. Move/remove trailing `sec-the-boundary-interface-symplectic-structure` label to
   `docs/source/1_agent/06_fields/01_boundary_interface.md`.
4. Add a local symbol glossary (or a short “symbols used here” table) because this chapter combines many objects.
5. Move heavy derivations/proofs to the appendices and keep this chapter focused on statement + interpretation +
   implementation consequences.
6. Add a minimal simulation/example (even pseudocode) of the jump-diffusion to help readers validate intuition.
7. Reduce long-line count by wrapping prose and splitting wide equations.

### `docs/source/1_agent/06_fields/01_boundary_interface.md` — *The Boundary Interface: Symplectic Structure*

**Snapshot:** ~1173 lines; `##` sections: 8; long lines >120: ~135; ends with stray label `sec-the-reward-field-value-forms-and-hodge-geometry`.

1. Add `## TLDR` explaining the boundary interface, Dirichlet/Neumann duality, and what “symplectic” buys you.
2. Add `(sec-the-boundary-interface-symplectic-structure)=` before the H1 (currently stranded at the end of
   `05_geometry/04_equations_motion.md`).
3. Move/remove trailing `sec-the-reward-field-value-forms-and-hodge-geometry` label to
   `docs/source/1_agent/06_fields/02_reward_field.md`.
4. Add a schematic diagram of boundary variables and their duals (sensors vs actions) to reduce cognitive load.
5. Add a worked 1D example showing the boundary condition changing between waking/dreaming modes.
6. Add explicit implementation hooks: how to compute/monitor these boundary quantities in a deep RL system.
7. Wrap long lines and ensure the “Section 20” (WFR) references land on the correct WFR chapter anchor.

### `docs/source/1_agent/06_fields/02_reward_field.md` — *The Reward Field: Value Forms and Hodge Geometry*

**Snapshot:** ~1332 lines; `##` sections: 8; long lines >120: ~156; ends with stray label `sec-supervised-topology-semantic-potentials-and-metric-segmentation`.

1. Add `## TLDR` defining “reward 1-form”, “value potential”, and what Hodge decomposition means operationally.
2. Add `(sec-the-reward-field-value-forms-and-hodge-geometry)=` before the H1 (currently stranded at the end of
   `06_fields/01_boundary_interface.md`).
3. Move/remove trailing `sec-supervised-topology-semantic-potentials-and-metric-segmentation` label to
   `docs/source/1_agent/07_cognition/01_supervised_topo.md`.
4. Add an explicit RL example translating scalar reward into a value field/1-form and showing how it drives dynamics.
5. Add a “Hodge decomposition for practitioners” sidebar (exact/coexact/harmonic parts and how each appears in learning).
6. Add implementation notes: estimating gradients/forms with neural nets, regularizers, and diagnostics for debugging.
7. Reduce line widths (wrap long math and prose).

### `docs/source/1_agent/06_fields/03_info_bound.md` — *The Causal Information Bound*

**Snapshot:** ~670 lines; `##` sections: 12; long lines >120: ~244; no file-level `sec-*` label near top.

1. Add `## TLDR` stating the bound and its main consequence (Causal Stasis) in one paragraph.
2. Add `(sec-causal-information-bound)=` before the H1 (currently this label sits at the end of
   `docs/source/1_agent/07_cognition/06_causality.md`).
3. Add an early “units + scaling” subsection pointing to Appendix B; this chapter mixes area, nats, and length scales.
4. Add a numeric worked example computing `I_max` for toy parameters, and show how close one is to the stasis regime.
5. Ensure appendix references are as precise as possible (link directly to the A.6 subsection label for the area law).
6. Replace any ASCII/monospace “diagram” blocks with proper figures (Sphinx diagrams render more reliably).
7. Reflow long lines to reduce horizontal scrolling.

### `docs/source/1_agent/07_cognition/01_supervised_topo.md` — *Supervised Topology: Semantic Potentials and Metric Segmentation*

**Snapshot:** ~1181 lines; `##` sections: 7; long lines >120: ~215; ends with stray label `sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller`.

1. Add `## TLDR` (semantic potentials + what metric segmentation returns).
2. Add `(sec-supervised-topology-semantic-potentials-and-metric-segmentation)=` before the H1 (currently stranded at the
   end of `06_fields/02_reward_field.md`).
3. Move/remove trailing `sec-theory-of-meta-stability-the-universal-governor-as-homeostatic-controller` label to
   `docs/source/1_agent/07_cognition/02_governor.md`.
4. Add a worked supervised example (classification/metric learning) showing how potentials are built and used.
5. Add an explicit algorithm section: inputs, outputs, complexity, and how to evaluate segmentation quality.
6. Keep physics metaphors in `feynman-prose` blocks, but ensure the main text remains ML-first and operational.
7. Wrap long lines and add section summaries for navigation.

### `docs/source/1_agent/07_cognition/02_governor.md` — *Theory of Meta-Stability: The Universal Governor as Homeostatic Controller*

**Snapshot:** ~940 lines; `##` sections: 9; long lines >120: ~189; ends with stray label `sec-section-non-local-memory-as-self-interaction-functional`.

1. Add `## TLDR` describing what the Governor monitors and what it can change (and at what timescale).
2. Add a stable file-level label before the H1 (currently the “governor” label is stranded in the previous chapter).
3. Move/remove the trailing `sec-section-non-local-memory-as-self-interaction-functional` label: it should be attached to
   the memory chapter (and normalized—drop the extra `section-` prefix).
4. Add a block diagram: diagnostics → governor → parameter updates/halts; this will reduce repeated prose.
5. Add a “minimal implementation” recipe: what statistics to keep, how to set thresholds, and how to avoid governance
   oscillations (meta-Zeno behavior).
6. Add a correspondence table to classical control (PID/MPC/homeostasis) to anchor readers.
7. Reflow long lines and ensure every referenced theorem/definition resolves correctly via `prf:ref`.

### `docs/source/1_agent/07_cognition/03_memory_retrieval.md` — *27: Non-Local Memory as Self-Interaction Functional*

**Snapshot:** ~1039 lines; `##` sections: 17; long lines >120: ~215; ends with stray label `sec-symplectic-multi-agent-field-theory`.

1. Normalize the title: either remove the “27:” prefix or make numbering consistent across all Volume 1 lectures.
2. Add `## TLDR` summarizing the self-interaction view of memory and the main operational predictions.
3. Add/normalize a file-level `sec-*` label before the H1 (the current label intended for this section appears at the end
   of `07_cognition/02_governor.md`).
4. Move/remove trailing `sec-symplectic-multi-agent-field-theory` label to `docs/source/1_agent/08_multiagent/01_gauge_theory.md`.
5. Add a concrete implementation mapping: attention/retrieval from a replay buffer as a self-interaction kernel; include
   pseudocode and complexity.
6. Add failure modes + diagnostics: what “bad memory attractors” look like and how the system detects/remediates them.
7. Reduce long lines by splitting dense math/prose into aligned blocks and shorter paragraphs.

### `docs/source/1_agent/07_cognition/04_ontology.md` — *Ontological Expansion: Topological Fission and the Semantic Vacuum*

**Snapshot:** ~1471 lines; `##` sections: 17; long lines >120: ~284; ends with stray label `sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics`.

1. Add `## TLDR` defining ontological expansion/fission/vacuum in operational terms.
2. Add `(sec-ontological-expansion-topological-fission-and-the-semantic-vacuum)=` before the H1 (currently this label is
   stranded at the end of `docs/source/1_agent/08_multiagent/01_gauge_theory.md`).
3. Move/remove trailing `sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics` label to
   `docs/source/1_agent/07_cognition/05_metabolism.md`.
4. Split the chapter: conceptual overview vs formal criteria vs implementation triggers (it’s a 1.5k-line monolith).
5. Add a toy example of fission (single chart → two charts) with diagrams and a before/after metric picture.
6. Add “implementation hooks” (what triggers fission, how to implement safely, what to log/monitor).
7. Wrap long lines and add an in-file navigation aid (mini ToC / section summaries).

### `docs/source/1_agent/07_cognition/05_metabolism.md` — *Computational Metabolism: The Landauer Bound and Deliberation Dynamics*

**Snapshot:** ~671 lines; `##` sections: 6; long lines >120: ~138; ends with stray label `sec-causal-discovery-interventional-geometry-and-the-singularity-of-action`.

1. Add `## TLDR`: what metabolic cost is, what quantities are measurable, and what the bound constrains.
2. Add `(sec-computational-metabolism-the-landauer-bound-and-deliberation-dynamics)=` before the H1 (currently stranded at
   the end of `07_cognition/04_ontology.md`).
3. Move/remove trailing `sec-causal-discovery-interventional-geometry-and-the-singularity-of-action` label to
   `docs/source/1_agent/07_cognition/06_causality.md`.
4. Add a numeric worked example using Appendix B units (compute a budget and show where the inequality binds).
5. Clarify what is “modelled” vs “measured” vs “derived”; add a short “how to estimate in practice” subsection.
6. Add explicit links to governor actions (what changes when metabolic budget is exceeded).
7. Reflow long lines and long math expressions.

### `docs/source/1_agent/07_cognition/06_causality.md` — *Causal Discovery: Interventional Geometry and the Singularity of Action*

**Snapshot:** ~445 lines; `##` sections: 6; long lines >120: ~131; ends with stray label `sec-causal-information-bound`.

1. Add `## TLDR`: what is being discovered, and what “closure under intervention” means here.
2. Add `(sec-causal-discovery-interventional-geometry-and-the-singularity-of-action)=` before the H1 (currently stranded at
   the end of `07_cognition/05_metabolism.md`).
3. Move/remove trailing `sec-causal-information-bound` label to `docs/source/1_agent/06_fields/03_info_bound.md`.
4. Add a worked causal-graph example showing observational vs interventional distributions and how the method detects a
   missing variable.
5. Clarify when an action corresponds to a `do()` intervention vs conditioning; make assumptions explicit.
6. Add an implementation section listing practical invariance tests/statistics and how to integrate with diagnostics.
7. Reflow long lines; some paragraphs and equations exceed comfortable width.

### `docs/source/1_agent/07_cognition/07_metabolic_transducer.md` — *The Metabolic Transducer: Autopoiesis and the Szilard Engine*

**Snapshot:** ~1279 lines; `##` sections: 8; long lines >120: ~164; has file label `sec-the-metabolic-transducer-autopoiesis-and-the-szilard-engine`.

1. Add `## TLDR` + prerequisites at the top (the chapter is long and metaphor-heavy).
2. Reduce extreme line lengths (some lines are ~1000 chars) by wrapping prose and splitting long equations.
3. Add a schematic figure for the “transducer / Szilard engine” mapping (inputs, memory, work extraction).
4. Add concrete implementation hooks: what signals are measured, what loss terms are added, and what diagnostics to watch.
5. Add at least one toy example demonstrating “work” vs standard reward signal.
6. Strengthen cross-links to `09_economics/01_pomw.md` and Appendix B (units), since the story continues there.
7. Add short “section summaries” to improve navigation within the 1.2k-line chapter.

### `docs/source/1_agent/07_cognition/08_intersubjective_metric.md` — *The Inter-Subjective Metric: Gauge Locking and the Emergence of Objective Reality*

**Snapshot:** ~1394 lines; `##` sections: 12; `###` sections: 13; long lines >120: ~216; has file label `sec-the-inter-subjective-metric-gauge-locking-and-the-emergence-of-objective-reality`.

1. Add `## TLDR` defining gauge locking and the intersubjective metric in operational terms.
2. Reduce long-line extremes (max line length ~880) to improve rendering and diffs.
3. Add a worked example (two agents aligning their symbol systems) to ground the abstraction.
4. Add an explicit “dictionary” mapping terms here to those in `08_multiagent/01_gauge_theory.md` (shared vocabulary, but
   readers need a map).
5. Add a “failure modes + diagnostics” section: how misalignment manifests and what interventions exist.
6. Add a disclaimer/clarifier on what is metaphor vs literal claim to prevent misreadings.
7. Improve navigation with per-section summaries or a mini ToC (the file is long).

### `docs/source/1_agent/08_multiagent/01_gauge_theory.md` — *Relativistic Symplectic Multi-Agent Field Theory*

**Snapshot:** ~3573 lines; `##` sections: 31; long lines >120: ~546; ends with stray label `sec-ontological-expansion-topological-fission-and-the-semantic-vacuum`.

1. Add `## TLDR` + a reading guide (“minimal path”, “full path”); this is the longest chapter in Volume 1.
2. Add `(sec-symplectic-multi-agent-field-theory)=` before the H1 (currently stranded at the end of
   `docs/source/1_agent/07_cognition/03_memory_retrieval.md`).
3. Move/remove trailing `sec-ontological-expansion-topological-fission-and-the-semantic-vacuum` label to
   `docs/source/1_agent/07_cognition/04_ontology.md`.
4. Split into multiple pages (dictionary/prereqs, core field theory, applications, appendices) to make navigation viable.
5. Add a running toy model used throughout (2 agents, simple coupling) to make equations interpretable.
6. Reduce long-line count aggressively (wrap prose, break huge equations, avoid 500+ wide lines).
7. Add “implementation relevance” callouts: which parts map to concrete multi-agent learning losses and protocols.

### `docs/source/1_agent/08_multiagent/02_standard_model.md` — *The Standard Model of Cognition: Gauge-Theoretic Formulation*

**Snapshot:** ~1457 lines; `##` sections: 6; long lines >120: ~283; has file label `sec-standard-model-cognition`.

1. Add `## TLDR` summarizing the correspondence and what it claims to constrain/predict.
2. Add a “dictionary first” table early so readers don’t need to read physics-formalism before understanding the mapping.
3. Reduce extreme line lengths (~990) to improve readability.
4. Add a prerequisites section plus an alternate “no-physics” reading path focusing on control/geometry meaning.
5. Add one worked derivation showing how a stability/selection term emerges from the cognitive assumptions.
6. Strengthen links to the parameter sieve chapter + notebook (this chapter should feed into those concretely).
7. Add a short “limits of the analogy” section to reduce the risk of overinterpretation.

### `docs/source/1_agent/08_multiagent/03_parameter_sieve.md` — *The Parameter Space Sieve: Deriving Fundamental Constants*

**Snapshot:** ~1216 lines; `##` sections: 11; long lines >120: ~192; has file label `sec-parameter-space-sieve`.

1. Add `## TLDR`: list the constraints and the derived outputs in 6–10 bullets.
2. Add a consolidated “constraint checklist” table (formula, units, operational meaning, where computed in the notebook).
3. Add explicit links to `docs/source/1_agent/08_multiagent/04_constants_check.ipynb` in each constraint section.
4. Add a reproducibility box: which constants are fixed by SI, which are measured, which are hyperparameters/assumptions.
5. Add short “bridge summaries” between constraints (“what the last constraint ruled out / what remains”).
6. Wrap long lines and keep symbol usage consistent with Appendix B.
7. Add a limitations section mirroring the notebook’s “critical analysis” so the markdown chapter doesn’t overclaim.

### `docs/source/1_agent/08_multiagent/04_constants_check.ipynb` — *The Parameter Space Sieve: Computational Verification*

**Snapshot:** 76 cells (41 markdown, 35 code); well-structured headings.

1. Add a “How to run” cell at the top (dependencies, expected environment, version notes).
2. Ensure determinism/reproducibility: set seeds and note any version-sensitive calculations.
3. Keep outputs lightweight for docs builds: clear huge outputs, prefer saved figures with controlled sizes.
4. Add assertions / consistency checks so refactors don’t silently change derived ranges.
5. Add cross-links back to `docs/source/1_agent/08_multiagent/03_parameter_sieve.md` and forward to “critical analysis”.
6. Consider parameterizing the notebook (so ranges/assumptions can be changed without manual editing).
7. Add a final clean summary table intended to be copied into the markdown chapter.

### `docs/source/1_agent/09_economics/01_pomw.md` — *Proof of Useful Work: Cognitive Metabolism as Consensus*

**Snapshot:** ~1933 lines; `##` sections: 11; long lines >120: ~207; has file label `sec-proof-of-useful-work-cognitive-metabolism-as-consensus`.

1. Add `## TLDR` describing the protocol idea and how it differs from PoW/PoS.
2. Add a system architecture diagram (roles, messages, verification flow) to match the protocol narrative.
3. Clarify the security model (threats/assumptions) and separate what is proven vs proposed.
4. Add a “minimal spec” section: concrete state machine + message formats (so implementers have something to build).
5. Add explicit links back to the metabolism/transducer chapters and Appendix B (units) for consistency.
6. Add at least one worked example showing a small network reaching consensus and measuring “useful work”.
7. Reduce long-line issues and wide tables; add an in-file navigation aid for a ~2k-line chapter.

### `docs/source/1_agent/10_appendices/01_derivations.md` — *Appendix A: Full Derivations*

**Snapshot:** ~1174 lines; `##` sections: 7; long lines >120: ~126.

1. Fix the duplicate numbering: there are two headings named `## A.3 ...` (lines ~177 and ~216). Renumber the second and
   adjust subsequent numbering accordingly.
2. Add a short `## TLDR` stating what each A.* derivation supports in the main text.
3. Add a local “notation used here” block (or a strong pointer to Appendix B) at the start.
4. For each derivation section, state assumptions and starting equations explicitly before manipulating.
5. Add a boxed “result” at the end of each derivation, with a link back to where the result is first used.
6. Improve equation formatting (aligned multiline) for readability and verification.
7. Wrap long lines so derivations are readable in narrow displays.

### `docs/source/1_agent/10_appendices/02_parameters.md` — *Appendix B: Units, Parameters, and Coefficients*

**Snapshot:** ~96 lines; `##` sections: 3; long lines >120: ~69.

1. Add a short `## TLDR` explaining the unit conventions (nats/steps) and how to interpret the tables.
2. Reformat wide tables to avoid very long lines: split by category or use multi-line cells.
3. Add cross-links for overloaded symbols to the definitions/sections where each meaning is introduced.
4. Add default values where applicable and distinguish SI-defined constants vs measured vs chosen hyperparameters.
5. Add a “how to use this appendix” note listing common pitfalls (bits vs nats, time step symbols, etc.).
6. Consider generating this appendix from a single config/source-of-truth to avoid drift.
7. Expand the “Symbol Overload” subsection with concrete “this symbol means X in section Y” examples.

### `docs/source/1_agent/10_appendices/03_wfr_tensor.md` — *Appendix C: WFR Stress-Energy Tensor*

**Snapshot:** ~129 lines; `##` sections: 4; long lines >120: ~1.

1. Add a `## TLDR` stating the final stress-energy tensor form and what each term corresponds to.
2. Add explicit sign conventions and definition reminders (metric signature, variation conventions).
3. Add a “where used” section linking back to WFR geometry and equations-of-motion chapters.
4. Expand intermediate steps where derivations are currently condensed (to support verification).
5. Add sanity checks (dimensional analysis, limiting cases) for reader confidence.
6. Ensure notation matches the main text (symbols for metric, density, reaction term).
7. Keep the appendix self-contained: define all symbols locally or with precise links.

### `docs/source/1_agent/10_appendices/04_faq.md` — *Appendix D: Frequently Asked Questions*

**Snapshot:** ~938 lines; `##` sections: 13; many `###`; long lines >120: ~238.

1. Fix the broken `prf:ref`: `Theorem {prf:ref}`thm-rl-degeneracy`` is referenced but not defined anywhere in
   `docs/source` (it appears at `docs/source/1_agent/10_appendices/04_faq.md:384`).
2. Add a mini ToC at the top listing the FAQ sections/questions with links; this appendix is long.
3. Audit cross-references to “Section 0.6” and similar global sections: ensure they link to the correct file/label and
   prefer `{doc}` when the intent is “go to that chapter”, not “jump to a label”.
4. Standardize Q/A format: “short answer” first, then longer explanation, then pointers.
5. Move deeply technical answers into Appendix E or the reference file when they are effectively mini-lectures.
6. Reflow long lines and trim very wide math/code passages to keep the FAQ readable.
7. Add a “misconceptions” subsection clarifying the scope of physics analogies to prevent common misreadings.

### `docs/source/1_agent/10_appendices/05_proofs.md` — *Appendix E: Rigorous Proof Sketches for Ontological and Metabolic Laws*

**Snapshot:** ~2097 lines; `##` sections: 20; long lines >120: ~190.

1. Add `## TLDR` explaining what is proved vs sketched and what assumptions are imported as axioms.
2. Add an index table mapping each proof E.* to the theorem/corollary it supports in the main chapters.
3. Standardize proof-sketch structure (Statement → Assumptions → Proof idea → Key steps → Pointers).
4. Reduce long lines and ensure equation formatting is consistent (aligned blocks for multi-line derivations).
5. Add dependency notes per proof (which definitions/lemmas are required).
6. Consider splitting Appendix E into multiple appendices by theme (geometry, ontology, metabolism, multi-agent).
7. Add quick “sanity check / intuition” notes after each proof to help non-specialist readers.

### `docs/source/1_agent/reference.md` — *Agent reference (no proofs)*

**Snapshot:** ~10340 lines; `##` sections: 10 (category headers); long lines >120: ~1357; no explicit section labels.

1. Add a short `## TLDR` explaining how this file is generated, how to navigate it, and what it omits (“no proofs”).
2. Split into multiple reference pages (definitions/axioms/theorems/algorithms) with an index page; 10k lines is
   unwieldy.
3. Add anchors/labels for each category and (optionally) for entries so other chapters can deep-link into the reference.
4. Reduce long lines and wrap paragraphs; the reference is particularly affected by horizontal overflow.
5. Ensure the `{prf:ref}` links land on the correct statements in the main lectures (which requires fixing the misplaced
   `(sec-...)=` labels across Volume 1).
6. Add an “origin” field (file path + section) for each entry in addition to the `{prf:ref}` link, for fast context.
7. Add a de-duplication pass or a “canonical statement” mechanism to reduce size if many near-identical statements exist.

