# Hypostructure Volume 1 Improvement Report

Scope: docs/source/2_hypostructure (all files listed below). Each file has at least five concrete improvements or fixes.

## docs/source/2_hypostructure/intro_hypostructure.md
- Normalize the author name between the YAML front matter ("Guillem Duran-Ballester") and the byline ("Guillem Duran Ballester") so citations and site metadata are consistent.
- Fix the Part numbering in the Book Map and Quick Navigation ("Part XIX" and "Part XX" appear after Part IX/X); use a consistent Roman numeral sequence or drop numerals entirely in favor of section names.
- Clarify the "Five Axioms" section by explicitly calling them "five axiom families" and listing their member axioms so the count matches the text.
- Align the gate table (Node names and numbering) with the canonical gate specs in `docs/source/2_hypostructure/04_nodes/01_gate_nodes.md` to avoid mismatched terminology.
- Add one short worked example (e.g., 2D Navier-Stokes or a gradient flow) that maps thin objects to a few gates so the TL;DR reads as an executable overview.

## docs/source/2_hypostructure/template.md
- Fix the phase numbering ("Phase 0" appears after Phase 8) by renaming it "Post-Run" or moving it before Phase 1.
- Add a minimal filled-in example (toy system) so users see how to complete the template without guessing the level of detail.
- Convert the decision flowchart to Mermaid (or add a link to a diagram) so it is visually consistent with other chapters.
- Replace or link any hard-coded section references (e.g., "Section 8.C") with actual label references that exist in the book.
- Add a compact validation checklist at the top listing the required fields and minimum evidence per certificate type.

## docs/source/2_hypostructure/01_foundations/01_categorical.md
- Fix the hypothesis numbering gap (N1, N2, N3, N8, N9, N10, N11) by adding missing N4-N7 or renumbering the list.
- Add a concrete example of a hypostructure in the Set case (e.g., a classical PDE) to ground the categorical definition.
- Clarify what "flat connection" and the "exponential map" mean in a general cohesive topos (state existence assumptions or restrict to smooth settings).
- Include a small diagram that shows how Pi/flat/sharp act on a simple object (e.g., S^1) to make the modalities tangible.
- Rename or separate the "Small Object Argument" discussion to avoid confusion with Quillen's small object argument, even though a note already exists.

## docs/source/2_hypostructure/01_foundations/02_constructive.md
- Add a worked example of thin objects for one system (e.g., heat equation) so the inputs are not purely abstract.
- Clarify when the RCD upgrade is optional vs required, and list alternative conditions for non-RCD settings.
- Add a table that maps thin objects to the specific gate permits they enable, so readers see the derivation chain.
- Explain how to set or estimate the Levin limit M_sieve and what happens when it is exceeded.
- Ensure the cited metric-measure results appear in `references.bib` (RCD, Cheeger energy, log-Sobolev) and add missing entries if needed.

## docs/source/2_hypostructure/02_axioms/01_axiom_system.md
- Fix the Axiom SC prose: the explanation says "alpha > beta is supercritical" but earlier text uses it as subcritical; align the inequality and labels with the gate definition.
- Add a summary table of axioms with gate numbers, interface IDs, and failure modes to make the mapping explicit.
- Provide at least one concrete example per axiom family (Conservation, Duality, Symmetry, Topology, Boundary).
- Clarify the count by renaming "Five Axioms" to "Five Axiom Families" and listing sub-axioms explicitly.
- Add an explicit boundary axiom example (closed vs open systems) with references to Nodes 13-16.

## docs/source/2_hypostructure/03_sieve/01_structural.md
- Add a lookup table that maps each failure mode (C.E, T.D, etc.) to the corresponding gate/barrier/surgery nodes.
- Provide a static image or alt-text for the Mermaid diagram to support PDF builds or readers without Mermaid.
- Reconcile any node-count language (e.g., if "60 diagnostic nodes" appears later in the volume) with the actual 17-gate structure in this book.
- Formalize the "Excess/Deficiency/Complexity" axes with a short definition or criteria for classification.
- Add a brief example showing how the spectral-sequence interpretation maps to the actual gate ordering.

## docs/source/2_hypostructure/03_sieve/02_kernel.md
- Update the reference to "Part IV" if the Sieve diagram is actually defined in Part III.
- Replace `K^?` with the standard `K^{inc}` (or define `K^?` explicitly as an alias) to keep the certificate vocabulary consistent.
- Specify how context Gamma is updated at surgery re-entry (which certificates persist and which are invalidated).
- Add explicit outcome alphabets for special gates (CompactCheck, OscillateCheck) so they do not clash with barrier alphabets.
- Include short pseudocode for an epoch run to make the operational semantics implementable.

## docs/source/2_hypostructure/04_nodes/01_gate_nodes.md
- Rename Node 17 from "BarrierExclusion" to "Lock" or "LockCheck" to avoid mixing gate and barrier terminology.
- Ensure node names and IDs are consistent with other chapters (OscillateCheck, TameCheck, AlignCheck).
- Provide consistent Mermaid diagrams for all nodes or add a note explaining why some nodes omit diagrams.
- Add a mini diagram showing the 7a-7d subtree connections and re-entry targets to clarify the flow.
- Mark algorithmic pseudo-code sections as illustrative and list the assumptions (graph vs manifold inputs, sampling requirements).

## docs/source/2_hypostructure/04_nodes/02_barrier_nodes.md
- Standardize barrier certificate naming (e.g., `K_{BarrierSat}^{blk}`) and document the naming scheme.
- Add a summary table listing trigger gate, barrier outcome, and surgery re-entry target for each barrier.
- For special alphabet barriers (Scat, Gap), define how outcomes map to certificate types and routing.
- Clarify the meaning of "Weakest Precondition" with a concrete example and explicit non-circularity check.
- Add cross references from each barrier to any upgrade theorems that promote its blocked outcome.

## docs/source/2_hypostructure/04_nodes/03_surgery_nodes.md
- Add explicit admissibility predicates and required certificates for each surgery (not just in the schema).
- Ensure target node names match the exact gate labels/anchors used elsewhere, and fix any mismatches.
- Provide a progress measure (Type A or B) for every surgery so termination is explicit.
- Include a worked example (e.g., SurgCE) showing the state transformation and re-entry certificate construction.
- Add a list of preserved invariants per surgery to avoid accidental loss of key structure.

## docs/source/2_hypostructure/05_interfaces/01_gate_evaluator.md
- Harmonize the outcome alphabet (YES/NO/INC vs YES/NO with K_inc) with the kernel and node specs.
- Align modality notation with the rest of the book (Pi/flat/sharp vs int/flat/sharp), and add a brief equivalence note.
- Add a compact table mapping each interface to its gate number and definition label for quick lookup.
- Clarify the evaluation budget model and how resource exhaustion yields K_inc deterministically.
- Provide a concrete interface implementation example (e.g., D_E for a heat equation) to guide users.

## docs/source/2_hypostructure/05_interfaces/02_permits.md
- Distinguish gate verdicts (YES/NO) from barrier verdicts (Blocked/Breached) in the Weakest Precondition theorem.
- Add a dependency table for Tame, Rigidity, and Effectivity certificates with definition links.
- Provide a worked example showing an INC permit upgraded to YES using explicit prerequisites.
- Clarify the obligation ledger rules and link them to the template protocol.
- Ensure every permit mentioned links to its definition label to reduce cross-reference hunting.

## docs/source/2_hypostructure/05_interfaces/03_contracts.md
- Fix or define the "Mode V.D" reference (currently not aligned with the mode taxonomy elsewhere).
- Add a filled-in sample barrier contract and a sample surgery contract for clarity.
- Provide a contract validation checklist (non-circularity, re-entry, progress measure, scope).
- Clarify how "Scope" is expressed (type parameterization, explicit list) and how to document exclusions.
- Cross-link contracts to the concrete barrier/surgery specs to keep the atlas consistent.

## docs/source/2_hypostructure/06_modules/01_singularity.md
- Add explicit assumptions for the scaling limit in the Automation Guarantee (existence, regularity, symmetry).
- Provide pseudocode or a flow diagram for the factory pipeline (ProfileExtractor -> Admissibility -> SurgeryOperator).
- Add a table listing required permits per type (parabolic, dispersive, algorithmic, Markov).
- Include a short example of automated profile extraction for a well-known PDE.
- Clarify how automation failures are recorded (K_inc payloads and routing).

## docs/source/2_hypostructure/06_modules/02_equivalence.md
- Add concrete transport lemma definitions or links so def-transport-t1..t6 resolve.
- For each equivalence move, list required permits and the expected certificate payload.
- Provide a worked YES~ example showing equivalence + transport + gate certificate.
- Explain how comparability constants affect later thresholds (e.g., Lojasiewicz constants, capacity bounds).
- Add guidance for approximate or non-invertible moves (when to emit K_inc).

## docs/source/2_hypostructure/06_modules/03_lock.md
- Align the tactic count with other chapters (E1-E13 vs E1-E12/E1-E10) and update all references.
- Add a priority order or cost estimate for tactics to guide implementations.
- Provide a short example of a lock proof using one tactic (dimension mismatch) with explicit payload.
- Clarify whether the Lock is treated as a gate or a barrier and keep terminology consistent.
- Add a summary table listing each tactic, required permits, and failure conditions.

## docs/source/2_hypostructure/07_factories/01_metatheorems.md
- Make verifier outcomes consistent across the chapter (YES/NO vs YES/NO/INC) and align with kernel definitions.
- Update all references to the lock tactic list so E1-E13 is consistent with the lock module.
- Add a schema showing required inputs for each gate factory to make instantiation requirements explicit.
- Define or link the assumptions used in the decidability analysis (Rep-Constructive, Cert-Finite).
- Provide guidance on how to structure K_inc payloads and route them through barriers.

## docs/source/2_hypostructure/07_factories/02_instantiation.md
- Fix the "Part XII" reference to match the book map for this volume.
- Normalize certificate names in the Certificate Generator Library table to match definitions.
- Use the correct Lock certificate notation (K_Cat_Hom_blk) consistently in the unlock table.
- Add links from each table row to the exact gate/barrier definition labels.
- Clarify which of the eight user inputs are optional and how to handle closed vs open systems.

## docs/source/2_hypostructure/08_upgrades/01_instantaneous.md
- Add a generic upgrade rule template that specifies inputs, outputs, and ledger updates.
- For each upgrade theorem, state exactly which barrier certificate is required and where it is produced.
- Clarify how to record terminal upgrades (e.g., VICTORY) in the certificate chain.
- Add a failure mode note describing how to emit K_inc when bridge hypotheses are missing.
- Provide a mapping table from upgrade theorems to barrier definitions and literature.

## docs/source/2_hypostructure/08_upgrades/02_retroactive.md
- Define or link all certificates used in retroactive rules (K_Action_blk, K_Sym, K_CheckSC).
- Add a formal algorithm for retroactive upgrades that mirrors the ledger protocol in the template.
- Specify how multi-epoch provenance is recorded so retroactive upgrades are auditable.
- Add a non-circularity check for retroactive upgrades (late certs cannot depend on early ones).
- Include a worked timeline example (epoch 1 inc, epoch 2 strong cert, upgrade).

## docs/source/2_hypostructure/08_upgrades/03_stability.md
- Define the quantitative "Gap" and "Cap" measures used in Openness and link to their gate certificates.
- Provide definitions or links for the coupling certificates used in Product-Regularity (K_CouplingSmall, K_ACP).
- Clarify the topology and continuity assumptions on parameter space Theta and certificate functionals.
- Explain how shadowing certificates are generated in the Sieve (node vs external proof).
- Add a backend checklist (A/B/C) so users know which conditions to verify before applying.

## docs/source/2_hypostructure/09_mathematical/01_theorems.md
- Align the scaling inequality narrative with the axiom system and gate definitions (subcritical vs supercritical).
- Rename "LOCK-" theorem prefixes where they are gate predicates to avoid confusion with lock tactics.
- Add explicit links to the gate definition labels used in each theorem.
- Provide a concrete PDE example per theorem (NLS, Navier-Stokes, etc.).
- Audit node numbers (ErgoCheck, OscillateCheck) and align with gate specs.

## docs/source/2_hypostructure/09_mathematical/02_algebraic.md
- Align the scaling inequality in the Motivic Flow Principle with the rest of the book (alpha/beta consistency).
- Add Rigor Class declarations and bridge verification steps for each metatheorem.
- Include a worked example (e.g., smooth projective curve or abelian variety) to illustrate the motive assignment.
- Clarify the conditions under which the entropy-trace formula holds.
- Add cross references to the permits and their definitions used in the statements.

## docs/source/2_hypostructure/09_mathematical/03_cross_reference.md
- Replace or explain the "60 diagnostic nodes" claim so it matches the 17-gate structure in this volume.
- Audit all labels in the tables (`def-node-complex`, `mt-up-saturation-principle`, etc.) and fix broken references.
- Add missing entries for Nodes 13-17 and the 7a-7d subtree so the mapping is complete.
- Add a Rigor Class column to show which foundations are framework vs literature.
- Provide a short example showing how to trace a gate predicate to its theorem and literature source.

## docs/source/2_hypostructure/09_mathematical/04_taxonomy.md
- Define all certificate classes used in the tables (K_circ, K_re, K_ext, K_morph, K_hor) or link to definitions.
- Fix typos and naming inconsistencies (e.g., "Fadeev-Popov" -> "Faddeev-Popov").
- Add a legend that explains how to read each family/stratum cell.
- Provide a worked example of computing Structural DNA for a known system.
- Add cross links from the 7a-7d subtree to the gate node definitions.

## docs/source/2_hypostructure/09_mathematical/05_algorithmic.md
- Add an explicit notation alignment table for modalities (Pi/flat/sharp vs Disc/Gamma) and use one standard set.
- Clarify the statement and assumptions of the "Schreiber Structure Theorem" with precise citation.
- Justify or replace the entropy-based "polynomial time" definition and link it to CostCert in the bridge chapter.
- Provide examples for each algorithm class (climbers, propagators, alchemists, dividers, interference).
- Add a recipe for constructing each obstruction certificate from Sieve data.

## docs/source/2_hypostructure/09_mathematical/06_complexity_bridge.md
- Define the Fragile runtime step model explicitly so CostCert is checkable.
- Provide constructive translations for the four bridge theorems, with explicit polynomial overhead bounds.
- Clarify the notion of input size in the hypostructure model and how it maps to string length.
- Add a worked example (e.g., SAT verifier) to demonstrate the bridge in practice.
- Cross-reference representation assumptions (Rep_K) and how they are certified in the Sieve.

## docs/source/2_hypostructure/10_information_processing/01_metalearning.md
- Disambiguate K_A (defect functionals) vs certificate notation to avoid confusion.
- State conditions for existence of minimizers for the meta-objective (compactness/coercivity/convexity).
- Include a toy example that learns a parameterized axiom from data.
- Explain how boundary parameters are learned and how boundary defect is measured in practice.
- Add a short section on identifiability and regularization to avoid degenerate minima.

## docs/source/2_hypostructure/10_information_processing/02_fractal_gas.md
- Fix the incorrect path reference to the meta-learning chapter (use `10_information_processing/01_metalearning.md`).
- Rename the "TLDR" heading to "TL;DR" and add bullet points for faster scanning.
- Audit the file for statements missing Rigor Class or Status and mark them explicitly per the policy.
- Add pseudocode for the Fractal Gas update step (selection, cloning, mutation).
- Add a permit checklist showing which gates are required for each core theorem.

## docs/source/2_hypostructure/11_appendices/01_zfc.md
- Add a concrete example of translating a short certificate chain into ZFC statements.
- Fix or replace references to missing labels (e.g., def-higher-coherences) so links resolve.
- Provide a sample AC-dependency manifest so readers see how choice usage is recorded.
- Add a simple diagram for the translation pipeline (Hypostructure -> tau_0 -> ZFC).
- Clarify what information is lost under truncation and why it does not affect certificate validity.

## docs/source/2_hypostructure/11_appendices/02_notation.md
- Expand the interface identifier list to include all nodes (SC_partial c, TB_O, Bound_*, GC_T, etc.).
- Audit the references in the notation tables and replace any missing labels.
- Add definitions or references for K_circ, K_re, K_ext, K_morph, K_hor used in taxonomy tables.
- Include a small "symbol collisions" section listing overloaded symbols and disambiguations.
- Add modality symbols (Pi, int, flat, sharp, partial) to the notation index.

## docs/source/2_hypostructure/11_appendices/03_faq.md
- Add a table of contents and ensure the "40 questions" claim matches the actual count or update the claim.
- Audit cross references in answers and fix any missing labels.
- Add a concise "limitations and open problems" FAQ entry to balance the narrative.
- Provide a short glossary for repeated terms to reduce duplication in answers.
- Add a brief note on how to navigate the FAQ by theme (foundations, certificates, complexity, applications).

## docs/source/2_hypostructure/hypopermits_jb.md
- Add a note explaining that this file is a consolidated single-file version and link to the modular chapters.
- Normalize the author name to match the rest of the volume metadata.
- Fix the N1-N11 hypothesis numbering gaps (add missing items or renumber).
- Update internal references to match the Jupyter Book label structure.
- Add a short "how to use this file" section clarifying when to prefer the modular chapters.

## docs/source/2_hypostructure/references.bib
- Add missing DOIs/URLs for commonly cited entries to improve citation resolution.
- Normalize author name formatting and accents using consistent BibTeX escapes.
- Ensure all citations used in text (e.g., LaSalle76, HoTTBook) exist in the bibliography and add any missing entries.
- Standardize journal, publisher, and edition fields for consistent rendering.
- Add a short comment block for each new topic group (topos theory, algorithmic completeness) to mirror the document structure.
