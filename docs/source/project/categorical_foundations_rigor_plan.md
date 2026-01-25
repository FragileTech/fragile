# Categorical Foundations Rigorous Fix Plan

(sec-categorical-fix-plan-overview)=
## Overview and scope
This plan addresses the critical rigor failures identified in `docs/source/2_hypostructure/01_foundations/01_categorical.md`.
It focuses on formalization and proof structure, not stylistic edits. Each section below is a
step-by-step repair plan for one critical issue, with explicit deliverables and validation criteria.

Scope: critical issues only (AIT halting claims, Horizon Limit formalization, Initiality Lemma
smallness and existence, KRNL-Exclusion logic, holographic bound).

Guiding principles:
- Make every statement either a proved theorem with explicit hypotheses or an axiom/permit with a
  clear scope and provenance.
- Use consistent categorical foundations (1-topos vs (infty,1)-topos) and consistent logic
  (classical vs intuitionistic) across statements.
- Avoid claims that rely on unresolved conjectures or non-standard constructions unless they are
  explicitly marked as conditional.

(sec-fix-ait-halting-complexity)=
## Fix AIT halting-set complexity claims
Problem: The claim `K(K_n) = O(log n)` for the halting set is false, so the "liquid phase" and the
phase boundary narrative are incorrect as stated.

Plan:
1. Inventory current definitions and dependencies.
   - Locate and review `def-kolmogorov-complexity`, `def-algorithmic-phases`,
     `thm-sieve-thermo-correspondence`, and all references to `K(L_n)` or phase labels.
   - Record which complexity notion is used (plain, prefix-free, resource-bounded).
2. Fix the complexity model and notation.
   - Commit to a single complexity function for finite strings (prefer prefix-free `K`).
   - Define `L_n` precisely as the length-n prefix of the characteristic sequence of `L`.
   - Add a separate notion for "enumerator description length" if needed (e.g., `K_enum(L)`).
3. Replace incorrect claims with correct theorems.
   - Prove/quote: for decidable `L`, `K(L_n) <= K(n) + O(1)` (or `O(log n)`).
   - Prove/quote: for the halting set `K`, `K(K_n) >= n - O(1)` (Chaitin incompleteness).
   - Cite precise sources (Li-Vitanyi, Chaitin, Levin-Schnorr).
4. Rebuild or remove the "liquid phase" classification.
   - Option A (preferred): drop the liquid phase and state two regimes only:
     decidable (low `K(L_n)`), and random/incompressible (high `K(L_n)`).
   - Option B: keep a "liquid phase" but redefine it using `K_enum(L)` or a two-axis phase
     diagram (description length vs decidability). Make the axes explicit and non-conflicting.
5. Update theorem and proof text to match the corrected model.
   - Rewrite `thm-halting-ait-sieve-thermo` and its proof to remove the false `O(log n)` claim for
     c.e. sets.
   - Update all phase tables and explanatory prose to match the new model.
6. Validation criteria.
   - Every complexity claim references a precise definition and a cited theorem.
   - No statement implies `K(K_n) = O(log n)` or similar for c.e. sets.
   - The phase table uses only quantities that are well-defined and consistent.

Deliverables:
- Updated definitions for complexity and phase classification.
- Revised `thm-halting-ait-sieve-thermo` and its proof.
- Corrected narrative text around the phase boundary.

(sec-fix-horizon-limit)=
## Formalize the Horizon Limit theorem
Problem: The current statement uses `K(I)` for infinite sets and assumes a memory-bound proof that
does not follow from Kolmogorov complexity as defined.

Plan:
1. Fix the object of study.
   - Replace "problem `I subseteq N`" with a family of finite instances `I_n` or a decision
     function `f_n: {0,1}^n -> {0,1}`.
   - Define the Sieve input format and the decision task (e.g., "decide membership for length n").
2. Formalize the computational model.
   - Specify the Sieve as a finite-state machine or RAM model with memory `M` bits.
   - Define what "memory buffer" means (state size, work tape, or total storage).
3. Replace the statement with a correct finite-instance theorem.
   - Candidate statement: if `K(I_n) > M + c`, then no Sieve with memory `M` can reconstruct or
     decide `I_n` for all inputs of length `n`.
   - Use a counting argument: at most `2^M` distinct functions can be represented by `M`-bit
     machines, but there are `2^(2^n)` possible `n`-bit decision functions.
4. Update the proof accordingly.
   - Use standard incompressibility or counting arguments, not "store the entire set" claims.
   - Make the `HORIZON` verdict depend on the inability to represent the function for the given
     input length, not on `K(I)` for infinite sets.
5. Align with the AIT section.
   - Ensure `K(I_n)` or an equivalent finite complexity measure matches the corrected AIT model.
6. Validation criteria.
   - The theorem uses only finite objects with defined complexity.
   - The proof relies on a correct counting or information-theoretic argument.
   - No step requires computing a characteristic function for an undecidable set.

Deliverables:
- Revised statement and proof of `mt-krnl-horizon-limit`.
- Updated definitions for the Sieve memory model and input format.
- Updated references in phase classification sections.

(sec-fix-initiality-lemma)=
## Repair the Initiality Lemma and universal bad pattern existence
Problem: The current smallness/cofinality argument is invalid and the explicit constructions rely
on unresolved or ill-defined objects.

Plan:
1. Clarify the categorical setting.
   - Decide whether `Hypo_T` is a 1-category or an (infty,1)-category.
   - State required properties explicitly (locally presentable, accessible, complete/cocomplete).
2. Replace the "countable representative" argument.
   - Use a set-theoretic universe or accessible cardinal to define "smallness."
   - Show the class of germs is a set by bounding parameters in a fixed universe, not by
     separability or countability.
3. Implement a solution-set style existence proof.
   - Prove a solution set condition: every singularity pattern receives a morphism from a germ in
     a set `G_T` of representatives.
   - Use a standard construction: define `H_bad := colim_{g in G_T} H_g` in a cocomplete category.
   - Cite accessible category results (e.g., Adamek-Rosicky, Lurie).
4. Make cofinality precise or remove it.
   - If cofinality is used, state the indexing category and prove cofinality explicitly.
   - Otherwise, replace with the universal property of the colimit over the solution set.
5. Remove or reframe the explicit "construction by type" claims.
   - Move these to an "Examples (conditional)" subsection or remove entirely.
   - Only keep cases where a universal pattern is actually established in the literature.
6. Validation criteria.
   - The existence proof uses only well-defined set-theoretic or accessible category arguments.
   - No step claims countability without a proof.
   - The theorem is either proved under explicit hypotheses or downgraded to a conditional.

Deliverables:
- Revised Initiality Lemma statement with explicit hypotheses.
- Rewritten proof using solution set / accessible category arguments.
- Removal or qualification of unverifiable "by type" constructions.

(sec-fix-krnl-exclusion)=
## Repair KRNL-Exclusion logic in the topos setting
Problem: The proof treats Hom as a Set-valued presheaf and uses contrapositive reasoning not valid
in intuitionistic logic; the implication to `Rep_K` is not justified.

Plan:
1. Decide the logic level and mapping object.
   - If working in an (infty,1)-topos, use mapping spaces `Map(H_bad, H)` and apply
     propositional truncation `||Map||`.
   - If working in a 1-topos, use `Hom` sets but state whether the logic is Boolean.
2. Redefine the singularity predicate.
   - Define `Sing(H) := ||Map(H_bad, H)||` as a proposition (exists a morphism).
   - Define `Rep_K(T,Z)` explicitly as `not Sing(H(Z))`, or add an axiom that equates them.
3. Replace the contrapositive step.
   - If the topos is not Boolean, avoid `not (exists) -> Rep_K` unless `Rep_K` is definitional.
   - If Boolean logic is intended, add and justify a "Boolean/decidable" assumption for the
     relevant propositions.
4. Update the proof template.
   - Use Yoneda in the appropriate categorical level (Set or Spaces).
   - Keep the argument in the internal logic but avoid invalid negation steps.
5. Update hypotheses.
   - Add a hypothesis about truncation/decidability if needed.
   - Make N3 and N11 precise and consistent with the new definitions.
6. Validation criteria.
   - The proof uses only steps valid in the chosen logic.
   - `Rep_K` is derived by definition or by a stated logical principle.
   - Hom-emptiness is interpreted in the correct categorical level.

Deliverables:
- Revised KRNL-Exclusion statement and proof.
- Updated definitions for the singularity predicate and `Rep_K`.
- Clean hypothesis list with logic assumptions made explicit.

(sec-fix-holographic-bound)=
## Fix the holographic bound statement
Problem: The inequality `S_coh(X) <= C * chi(boundary X)` is false in general and is stated without
necessary hypotheses.

Plan:
1. Determine the intended geometric setting.
   - Specify dimension, compactness, orientation, boundary regularity, and singular locus model.
   - Define `X_sing` and `S_coh` precisely (base of logarithm, normalization).
2. Identify a correct theorem or downgrade to a permit/axiom.
   - If a real theorem is intended, locate one (e.g., Morse-theoretic bounds, Alexander duality,
     or bounds using Betti numbers) and state it with all hypotheses.
   - If no theorem supports the claim, reclassify it as an explicit axiom/permit tied to a
     physical model and mark it as such.
3. Fix sign and scaling issues.
   - Ensure `chi(boundary X)` is nonnegative or use `abs(chi)` and state the restriction.
   - Replace `log |pi_0|` with a monotone function that respects the allowed sign.
4. Update the Hypostructure definition block.
   - Move the bound under a clearly labeled axiom or permit.
   - Reference the exact lemma/theorem if it is proved elsewhere.
5. Validation criteria.
   - The bound is either proved under explicit hypotheses or labeled as an assumption.
   - All quantities are defined and dimensionally consistent.

Deliverables:
- Corrected holographic bound statement with explicit hypotheses or axiom label.
- Updated references to the bound and its proof or permit.

(sec-fix-integration)=
## Integration and consistency pass
Plan:
1. Cross-reference audit.
   - Search for all references to the changed statements and update or remove them.
   - Ensure labels and references are consistent after edits.
2. Glossary and definition alignment.
   - Update related definitions (complexity, singularity predicate, Rep_K, germ) in their
     canonical locations.
   - Ensure notation is consistent across the chapter and appendices.
3. Validation checklist.
   - Each critical issue is resolved according to its section's criteria.
   - No proof relies on intuitionistic contrapositive unless justified.
   - No claims depend on unresolved conjectures without being marked conditional.

Deliverables:
- Updated `docs/source/2_hypostructure/01_foundations/01_categorical.md`.
- Updated related definitions and references in supporting docs.
- A short change log describing the formal changes.
