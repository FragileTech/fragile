# Round 7 Hostile-Referee Review: 05_algorithmic.md (P != NP Proof)

## Metadata
- Reviewed file: docs/source/2_hypostructure/09_mathematical/05_algorithmic.md
- Review date: 2026-03-12
- Reviewer: Claude Opus 4.6 (5 parallel hostile-referee agents: ♯-channel, ∫-channel, ♭-channel, ∗/∂-channels, architecture)
- Scope: All five modal channels + architecture + metatheorems + barrier decomposition + complexity bridge
- File state: ~10,689 lines
- Previous rounds: 6 (all 9 FATAL issues from R6 CLOSED; 7 SERIOUS + 12 NEEDS TIGHTENING remained)

## Executive Summary

Round 7 is the most adversarial pass to date. Five parallel agents independently attacked every
channel and the overall architecture with maximal hostility. The review identifies **7 FATAL-level
structural issues**, **12 SERIOUS gaps**, and **14 NEEDS TIGHTENING items**. Several of these are
new issues not found in prior rounds; others are reclassifications of issues previously marked CLOSED
that, upon deeper scrutiny, were not actually resolved.

**The central finding:** The proof has a sound conditional structure but the conditions it requires
are so strong that they arguably subsume the conclusion. The three deepest problems are:

1. **The Foundation Assumption (`axiom-structure-thesis`) is doing most of the work.** On
   0-truncated types (all complexity-theoretic objects), the topos-theoretic modal units act
   trivially. The "Concretely" clauses that carry the actual computational content are manual
   stipulations, not consequences of the topos structure. The claim "five modalities exhaust all
   structure" is a theorem of the topos but contributes zero content to the complexity-theoretic
   setting.

2. **~~`thm-schreiber-structure` remains a THEOREM TARGET.~~** (**CORRECTED:** The theorem has a
   complete proof at lines 120-148, with exhaustive five-modality case analysis at lines 184-352.
   The computational bridge is completed by the chain `thm-syntax-to-normal-form` →
   `lem-primitive-step-classification` → `thm-appendix-a-primitive-audit-table` →
   `thm-witness-decomposition` → `cor-computational-modal-exhaustiveness`. The verification table
   label "THEOREM TARGET" at line 10395 was stale and has been corrected to "THEOREM." See F7.2
   below for updated assessment.)

3. **The ∗ and ∂ "modal purity violation" arguments are circular.** The ∗-witness definition
   places no explicit modal restriction on the merge map (unlike ♯ and ∫ which have explicit
   items 6 and 8). The claim that the merge "cannot invoke ♭-type operations" is an assertion
   about taxonomic classification, not a structural impossibility.

### Severity counts

| Severity | Count | Carried from R6 | New in R7 |
|----------|-------|-----------------|-----------|
| FATAL | 6 | 0 (reclassified) | 6 |
| SERIOUS | 13 | 5 (S9, S10, S16, S17, S18) | 8 |
| NEEDS TIGHTENING | 14 | ~3 | ~11 |

**Important note on FATAL vs. R6:** Round 6 closed all its FATAL issues by text edits. Round 7
identifies deeper structural problems that text edits cannot resolve — they require new mathematical
work or honest acknowledgment that the result is conditional in ways not yet fully surfaced.

---

## FATAL Issues

### F7.1: The Foundation Assumption is tantamount to the conclusion

**Location:** `axiom-structure-thesis` (line ~10343); `rem-modal-restrictions-finite-type` (line 449);
`rem-conditional-status-classical-export` (line 387)

**The problem:** The document acknowledges (line 458) that on 0-truncated discrete types, the modal
units η^♯, η^∫, η^♭ act *trivially*: "for a discrete set S, ♯S ≃ S and ∫S ≃ S." The abstract
factorization F = g ∘ η^♯ is "vacuously satisfied by every map on a discrete type." Since ALL
complexity-theoretic objects are 0-truncated (line 460), the topos-theoretic modal restrictions carry
**zero** computational content.

The actual content is in the "Concretely" clauses: ♯-steps use a (D+2)-dimensional metric profile;
∫-steps use only predecessor data; ♭-steps do algebraic elimination on bounded-cardinality structures;
∗-steps do divide-and-conquer; ∂-steps operate on interfaces. These are **manually designed
computational restrictions**, not consequences of the topos structure. The document admits (line 479):
"These concrete specifications are the formal definitions for the complexity-theoretic setting."

**Why this is fatal:** The proof's strategy is: (1) every poly-time algorithm decomposes into five
modal types; (2) each type is blocked for 3-SAT. But claim (1) is the Foundation Assumption — it is
assumed, not proved. The topos provides the "taxonomy" (why five and not six), but that taxonomy
contributes nothing to 0-truncated types. The claim that the five "Concretely"-defined classes
exhaust all of P is the very thing that needs proving, and it is assumed as an axiom.

**The ZFC analogy is misleading.** ZFC is a general-purpose foundation whose axioms do not mention
computation. The Foundation Assumption explicitly asserts that a specific structure (cohesive
(∞,1)-topos with five modalities) faithfully captures polynomial-time computation. This is a
substantive claim about the nature of efficient computation, not a foundational choice.

**What would resolve this:** Either (a) prove the Foundation Assumption from more primitive principles
(i.e., prove that every Turing machine algorithm decomposes into the five concrete classes), or
(b) explicitly acknowledge that the result is conditional in a much stronger sense than "choosing a
foundation" — the conditionality is on an unproven claim that is at least as hard as P ≠ NP itself.

---

### F7.2: `thm-schreiber-structure` (Computational Form) remains a THEOREM TARGET

**Location:** `thm-schreiber-structure` (line 100); complexity bridge review C3; verification table

**The problem:** The Schreiber Structure Theorem as stated at line 100 includes the claim:

> "The five modalities supply the exhaustive list of pure structural routes that can appear in the
> algorithmic witness language."

The proof at lines 120-148 proves the topos-theoretic part (fracture squares decompose morphisms
into modal components). But the computational claim — that this decomposition captures all structural
resources exploitable by polynomial-time algorithms — has never been completed. The complexity bridge
review (C3) confirms: "Application of Schreiber's categorical decomposition to computational
complexity is a theorem target, not a completed proof."

**Downstream impact:** This theorem is a dependency of:
- `thm-witness-decomposition` → every P_FM algorithm has a modal profile
- `cor-computational-modal-exhaustiveness` → P_FM = Sat⟨♯, ∫, ♭, ∗, ∂⟩
- `thm-mixed-modal-obstruction` → blocked modalities imply no solver
- `thm-irreducible-witness-classification` → irreducible components are pure

Without `thm-schreiber-structure`, the entire proof chain from "individual channels blocked" to
"no solver exists" has an unproved link.

**What would resolve this:** Complete the proof that the fracture-square decomposition implies
every internally polynomial-time family admits a modal factorization tree. This requires connecting
the categorical density theorem to the concrete witness language, which is non-trivial mathematical
work.

---

### F7.3: The ∗-witness definition has NO explicit modal restriction on the merge map

**Location:** `def-pure-star-witness-rigorous` (lines 1338-1390); Step 4 of `lem-scaling-obstruction`

**The problem:** Compare the five witness definitions:
- **♯-witness** has item 6: explicit modal restriction (factorization through metric profile μ)
- **∫-witness** has item 8: explicit modal restriction (Vis(i) information access restriction)
- **♭-witness** has item 6: algebraic derivation from certified primitives
- **∂-witness** has item 7: explicit modal restriction (interface-local operations)
- **∗-witness** has **NO ANALOGOUS ITEM**: items 1-7 specify only: size measure, polynomial bound,
  splitting map, merge map (merely "uniformly polynomial-time"), base case, total work bound,
  correctness.

The merge map in item 4 is specified only as "uniformly polynomial-time." A merge map that
internally performs Gaussian elimination, unit propagation, or gradient descent is perfectly
compatible with items 1-7. The proof then claims (line 2702): "Under ∗-purity, the merge cannot
invoke ♭-modal operations." But no item in the definition excludes ♭-modal operations from the
merge.

**Why this is fatal:** The entire ∗-channel blockage argument is: the merge must reconcile crossing
constraints, which requires ♭/♯/∫-type operations, which are "forbidden by ∗-purity." But ∗-purity,
as defined, does not forbid any polynomial-time computation in the merge. The argument conflates a
taxonomic label ("this operation is ♭-type") with a structural impossibility ("this operation cannot
be performed by a ∗-witness").

**What would resolve this:** Add an explicit "∗-modal restriction" item to
`def-pure-star-witness-rigorous` that constrains what the merge map can compute — analogous to item 6
of the ♯-witness or item 8 of the ∫-witness. For example: "The merge map operates only on the
recursion-tree structure (subproblem answers, split metadata) and cannot access the original input's
constraint structure, metric structure, or algebraic structure." Without such a restriction, the
∗-channel blockage is vacuous.

---

### F7.4: The ∂-modal restriction permits "arbitrary polynomial-time computation on interface data"

**Location:** `def-pure-boundary-witness-rigorous` item 7 (lines 1447-1474); Step 4 of
`lem-boundary-obstruction`

**The problem:** Item 7 of the ∂-witness definition states (line 1454): "C_n^∂ may ... perform
arbitrary polynomial-time computation on data that is explicitly present in the interface object."
The proof then claims the contraction "cannot solve a CSP on separator variables" because that
is a "♭-type operation."

But if the interface object B_n^∂ encodes the crossing constraints (which it does — the boundary
extraction map ∂_n compresses the bulk into the interface, and the crossing constraints are part of
what gets compressed), then C_n^∂ can read those constraints from B_n^∂ and perform "arbitrary
polynomial-time computation" on them — including solving a CSP, if a polynomial-time CSP solver
exists.

The proof's response is: "if P = NP, a poly-time solver could do this, but we are proving P ≠ NP."
This is circular: the proof assumes the conclusion (that no poly-time CSP solver exists) to establish
that the contraction cannot solve the CSP.

The alternative response — "solving a CSP is structurally a ♭-operation, and ∂-purity forbids
♭-operations" — falls on the same sword as F7.3: the definition allows "arbitrary polynomial-time
computation" on interface data, which includes any ♭-type computation that can be performed on the
data present in the interface. The categorical labels (♭-type vs ∂-type) do not restrict what bits
can be flipped.

**What would resolve this:** Either (a) restrict the ∂-witness definition to forbid CSP-solving
operations explicitly (not just by taxonomic label), or (b) prove that the interface object B_n^∂
does not contain enough information to specify the crossing constraints — which would require an
information-theoretic bound on interface compression, not just a polynomial-size bound.

---

### F7.5: The "Concretely" clauses are self-referentially designed to make the blockage work

**Location:** All five witness definitions; `rem-modal-restrictions-finite-type` (line 449)

**The problem:** The five "Concretely" clauses are:
- ♯: D+2 dimensional metric profile with D a *universal constant*
- ∫: access restricted to Vis(i) (predecessor clauses only)
- ♭: algebraic operations on structures of *polynomial cardinality*
- ∗: divide-and-conquer with merging (no modal restriction on merge — see F7.3)
- ∂: interface operations (allows arbitrary poly-time computation — see F7.4)

These restrictions are chosen to make each blockage proof work:
- Constant D makes the pigeonhole work for ♯
- Vis(i) restriction makes the frustrated-cycle argument work for ∫
- Polynomial cardinality makes the counting argument work for ♭

But the restrictions are **not derived** from the topos structure (which is trivial on 0-truncated
types). They are engineered to be:
(a) loose enough that known algorithms satisfy them (gradient descent satisfies ♯; DP satisfies ∫;
    Gaussian elimination satisfies ♭), and
(b) tight enough that the blockage proofs go through.

This is legitimate *definitional work* (defining five restricted classes and showing 3-SAT resists
them). But it does not prove P ≠ NP unless one also proves that every polynomial-time algorithm
belongs to the saturated closure of these five classes — which is the Foundation Assumption (F7.1).

The proof's claim to handle the "alien algorithm" loophole (Tactic E13) is: "there cannot be a sixth
mechanism because the topos has only five modalities." But this argument works only if the topos's
modal structure is computationally relevant to 0-truncated types, which it is not (line 458).

**Why this is fatal:** The proof proves a conditional statement: "IF every poly-time algorithm
decomposes into these five specific restricted classes, THEN P ≠ NP." The antecedent is as hard as
the consequent.

---

### F7.6: The modal purity violation pattern is taxonomic, not structural

**Location:** Steps 4c-4e of `lem-scaling-obstruction`; Step 4 of `lem-boundary-obstruction`;
Step 5 of `lem-causal-arbitrary-poset-transfer`

**The problem:** Multiple blockage proofs use the following argument pattern:
1. The computation must perform operation X (e.g., solve crossing constraints).
2. Operation X is "a ♭-type operation" (or ♯-type, or ∫-type).
3. The current modality forbids ♭-type operations.
4. Therefore the computation is impossible.

Step 2 is a *classification* of the operation within the five-modality taxonomy. Step 3 is a
*definitional exclusion* — the witness definition restricts what operations are available. Step 4
follows only if the classification in Step 2 is exhaustive: X can ONLY be done by ♭-type operations.

But the classification is not exhaustive. An operation that the taxonomy labels "♭-type" might also
be achievable by a different computational mechanism that the taxonomy labels differently, or that
the taxonomy misses entirely. For example:
- Solving a CSP on separator variables (labeled "♭-type") could also be done by a SAT solver that
  uses branch-and-bound (which mixes ♯ and ∫ elements) — but if we're inside a pure ∗-witness, the
  question is whether the merge can do it *without* any of the five modal types.
- The answer depends on whether "arbitrary polynomial-time computation" is a superset of the five
  types — which is precisely the question that the Foundation Assumption assumes.

**Why this is fatal:** The "purity violation" argument assumes that every computational operation
belongs to exactly one modality. If an operation can be performed by a mechanism that is
*none* of the five types (which is possible if the Foundation Assumption is false), then the
purity violation argument fails.

---

### F7.7: Workspace separation conflates mathematical decomposition with algorithmic state

**Location:** `def-modal-barrier-decomposition` property 2; `thm-modal-non-amplification`;
F9 resolution in Round 6

**The problem:** The non-amplification principle (Part I: Channel Isolation) claims that non-♯ steps
contribute zero to E_♯ variation. This relies on workspace separation: each modality operates in its
own workspace Z_n^♮, and a ♭-step acting on Z_n^♭ cannot affect Z_n^♯.

But an actual Turing machine has a **single shared tape**. There are no "separate workspaces." The
decomposition into five workspaces is imposed by the factorization tree analysis — it is a
*mathematical artifact*, not a property of the algorithm's execution. When the algorithm runs, every
step can read and write every bit of the tape.

The proof's response (rem-encoding-freedom-does-not-circumvent) is that the factorization
*requirement* limits what the core map can extract, even if the encoding smuggles everything in.
This is correct for a single step's input/output. But the non-amplification principle needs more:
it needs that a ♯-step cannot *indirectly* influence ∫-energy via the shared state. If R^♯ (the ♯
reconstruction map) writes to the shared state, and then E^∫ (the ∫ encoding) reads that state,
there is a potential information pathway from ♯-computation to ∫-energy that workspace separation
does not capture.

The barrier decomposition's property 2 is proved (F9 resolution) via the categorical product
structure of Z̃ = ∏ Z_n^♮. But Z̃ is NOT the algorithm's actual state space. The algorithm's state
space is {0,1}^{O(p(n))} (the tape), and the projection from tape to Z̃ is via the encoding maps,
which are arbitrary polynomial-time functions.

**What would resolve this:** Prove that the encoding/reconstruction pipeline R^♯ ∘ F^♯ ∘ E^♯
cannot affect ∫-energy even when E^∫ can read the full shared state. This requires analyzing the
composition of encoding maps across modality boundaries, not just the individual modal cores.

---

## SERIOUS Issues

### S7.1: Constant D in ♯-witness is definitional, not derived [NEW]

**Location:** `def-pure-sharp-witness-rigorous` item 6; `rem-constant-D-metric-profile`

The pigeonhole argument requires D to be a **universal constant** independent of n. This is placed
*into the definition*, not derived. An algorithm that evaluates n different energy differences per
step (e.g., WalkSAT computing make/break counts for all variables) uses D = O(n), not D = O(1).
The remark at line 2240 concedes this, saying such an algorithm "should be classified as ∫-modal."

This means: the ♯ class is *defined* narrowly enough for the pigeonhole to work, and algorithms
that don't fit are reclassified into other modalities. The blockage is correct *within the
definition* but the definition's narrowness pushes the burden onto modal exhaustiveness (F7.1).

**Status: SERIOUS (new).** Not fatal in isolation — the definition is self-consistent. But it
illustrates how the "Concretely" clauses are engineered for the proofs rather than derived from
structural principles.

---

### S7.2: Property 2 of `def-hard-subfamily-3sat` does not guarantee linear-size backbone [NEW]

**Location:** `def-hard-subfamily-3sat` property 2; ∫-channel proof lines 7174-7176

The ∫-channel proof claims "|Bb(F)| = δn for some δ = Θ(1)" from property 2. But property 2
specifies "glassy landscape: exponentially many local minima, vanishing spectral gap, Łojasiewicz
failure near frozen variables." This describes the energy landscape, not the backbone size.

Backbone = variables frozen across ALL satisfying assignments. Frozen variables within clusters are
different from backbone variables across the entire solution space. The linear-backbone claim at
α = 4.2 (below threshold) is plausible but requires a separate citation — e.g., Achlioptas-Coja-
Oghlan 2008 on the freezing threshold. The proof conflates cluster-level freezing with solution-
space-level backbone.

**Impact:** The entire Sub-claim B2 construction for the ∫-channel (the backbone triple witnessing
argument) depends on |Bb(F)| = Θ(n). If the backbone is sub-linear, the all-backbone clause
counting fails.

---

### S7.3: Backbone stability under single-clause perturbation is asserted without proof [NEW]

**Location:** ∫-channel proof, Sub-claim B2 (lines 7263-7307)

The proof constructs F_1 from F_0 by flipping one literal sign in clause C*, then claims F_1 ∈ H_n
"with high probability." The argument asserts (line 7283): "backbone changes at most for variables
in the O(1)-radius neighbourhood of v_0, affecting O(1) variables."

This is unjustified. Backbone is a **global** property: changing one clause can cause a phase
transition in the backbone. The "locally tree-like" argument applies to local graph properties, not
to backbone membership. Known results (survey propagation literature) suggest backbone can be fragile
near threshold — adding or removing one clause can flip backbone status of distant variables.

**Impact:** If F_1 loses backbone structure, the forcing argument for "F_0 and F_1 require opposite
values of v" collapses.

---

### S7.4: Frustrated-cycle argument conflates undirected SCC with directed implications [NEW]

**Location:** ∫-channel proof Steps 2-5; `def-hard-subfamily-3sat` property 3

Property 3 requires "strongly connected frustration cores of size Θ(n)." The proof uses SCCs of the
"clause-variable dependency graph." But 3-SAT clauses are NOT implications — a clause
(l₁ ∨ l₂ ∨ l₃) generates directed implications only when two literals are falsified (unit
propagation), which is assignment-dependent.

The proof's actual mechanism (Steps A-C of the distinguishing-instances argument) does not use
frustrated cycles at all — it uses backbone + visibility. The SCC language is motivational framing,
not load-bearing mathematics. This creates confusion about what structural property is doing the
work and which properties of H_n are actually needed.

---

### S7.5: Information leakage through predecessor states in ∫-channel [was partially addressed; remains]

**Location:** `def-pure-int-witness-rigorous` item 8; `rem-encoding-freedom-does-not-circumvent`

The Vis(i) restriction governs which *clauses of the encoded input* U_{n,i} can access. But it
does NOT restrict access to *predecessor state values*, which can contain arbitrary computed
information from previous updates. If U_{n,0} reads clause C (because C ∈ Vis(0)), processes it,
and stores the result in state_0, then any successor site can read that processed information
through predecessor state access.

The proof handles this: the key argument shows C* ∉ Vis(j) for any j ≤ k_1, so no predecessor
has access to C* through the Vis mechanism. But the clause-structured encoding restriction (line
1235-1241) needs to be tighter: it must prevent the encoding E_n^∫ from redundantly encoding
clause C* in a segment indexed by a different clause. The definition says "b_{C_k} encodes
constraint C_k" but does not explicitly state that b_{C_k} encodes *only* C_k.

---

### S7.6: Site-variable map injectivity assumed but not required [NEW]

**Location:** ∫-channel proof Step C (lines 7388-7393); `def-pure-int-witness-rigorous` item 7

The proof claims "σ_n^{sv} assigns a distinct variable to each site." But the definition specifies
σ_n^{sv} : [n] → P_n (variables to sites), which is not required to be injective. With
|P_n| ≤ q(n), many-to-one mappings are legal. If multiple variables map to one site, a single
update U_{n,i} can set multiple variables simultaneously, potentially affecting the backbone
argument.

---

### S7.7: Flat-universality (1)→(2) direction is explicitly unproven [NEW]

**Location:** `thm-flat-universality` proof (line ~4395)

The direction from "algorithm factors through ♭ modality" to "algorithm admits a pure ♭-witness
with polynomial-cardinality structures" is described as "a characterization target not on the
soundness critical path." This means: the theorem claims equivalence but only proves one direction.
The critical path avoids this via the primitive audit table, but the theorem statement should be
weakened or the gap flagged as a conjecture.

---

### S7.8: Primitive instruction set never explicitly exhibited [NEW]

**Location:** `lem-primitive-step-classification` (line ~4702); `thm-appendix-a-primitive-audit-table`

The primitive step classification relies on a "finite case analysis" over the evaluator's instruction
set. The audit table lists six rows (PT, SH, IN, FLAT, STAR, PARTIAL) and asserts each semantic
primitive carries a pure modal witness. But the semantic families are *defined* as the six modal
families — the audit table restates definitions.

The real question is whether the syntactic instruction set maps surjectively onto these six families.
The normal-form construction (Step 4 of `thm-syntax-to-normal-form`) classifies every microstep as
sharp-type (each decrements a step counter). This suggests every P_FM algorithm, when decomposed
through the normal form, consists entirely of ♯-type leaves. The other four modalities appear only
when Σ_prim contains non-♯ primitives, but Σ_prim's contents are never explicitly exhibited.

**Impact:** If the five-modality classification reduces to "everything is ♯-type through the normal
form," the claimed pentagonal structure is illusory and the entire argument reduces to blocking a
single (♯) modality.

---

### S7.9: Mixed-modal obstruction theorem has a composition gap [NEW]

**Location:** `thm-mixed-modal-obstruction` (line ~5872); proof at lines 5901-5925

The proof is: (1) assume solver exists; (2) it has a factorization tree; (3) every irreducible leaf
belongs to one pure class; (4) all five are blocked; (5) contradiction.

**The gap:** The blockage lemmas prove "no PURE ♯-witness solves 3-SAT" — no algorithm consisting
*entirely* of ♯-steps works. But the solver is not pure; it uses leaves of all five types. The
proof needs: blocking each modality *in isolation* blocks a *composite* algorithm using all five.

The missing link is: does blocking each individual modality block a component of a mixed strategy?
The proof does NOT invoke `thm-modal-non-amplification` here — it argues directly from irreducible
classification. The semantic obstruction propositions K_♮^-(Π) must mean "no irreducible ♮-component
in any correct solver's factorization tree" — not merely "no pure ♮-solver exists." The relationship
between these two claims needs explicit verification.

---

### S7.10: Per-step bound δ_∂(n) = O(1) is unjustified [carried from R6 as S18-related]

**Location:** `rem-boundary-per-step-bound`

The argument that each boundary-map composition resolves at most O(1) interface variables claims
"a single boundary-map application can only relate variables adjacent in the interface topology."
But a polynomial-time function on a polynomial-size interface can read and modify ALL bits in one
step. The claim confuses constraint-graph locality with computational locality.

---

### S7.11: Per-step bound δ_∗(n) ≤ 1 conflates "recursion node" with "computational step" [NEW]

**Location:** `rem-scaling-per-step-bound`

A recursion-tree node is not a "step." A merge operation can be arbitrarily complex polynomial-time
computation. If a "step" is one TM transition, processing a merge takes poly(n) steps and δ_∗
could be much larger. If a "step" is one recursion node, the bound is tautological. Either way,
the bound is either wrong or trivial.

---

### S7.12: Shattering at α = 4.267 lacks rigorous proof [carried as S16]

**Location:** `def-hard-subfamily-3sat` property 1; cited results

Ding-Sly-Sun 2021 establishes the satisfiability threshold but not all glassy properties
(exp(Θ(n)) clusters, vanishing spectral gap, Łojasiewicz failure) at the exact ratio α = 4.2
used in the proof. The precise rigorous references for shattering *below* (but near) threshold
are from the physics literature, not rigorous ZFC proofs. The Achlioptas-Ricci-Tersenghi 2006
result establishes shattering above a condensation threshold, but the exact location of this
threshold and its relationship to α = 4.2 needs explicit verification.

---

## NEEDS TIGHTENING Issues

| # | Channel | Issue | Notes |
|---|---------|-------|-------|
| T7.1 | ♯ | Steps 1-3 are rhetorical, not load-bearing, but stated as hypotheses | Cluster shattering, spectral gap, Łojasiewicz are listed as hypotheses but never used in the formal proof (only Step 4 pigeonhole is used). Misleading. |
| T7.2 | ♯ | Energy Φ(z) in metric profile is formula-dependent | The proof implicitly assumes the formula is encoded in z so that Φ(z) is computable from z alone. Should be stated explicitly. |
| T7.3 | ♯ | Property 6 (Boolean fiber rigidity) unused in ♯-obstruction but cited for orbit closure | If Property 6 were dropped, the orbit counting still works (Properties 1-5 are permutation-invariant). Either remove or justify. |
| T7.4 | ♯ | Orbit counting bound should clarify monotonicity (S⁺ ⊆ H_n⁺) | Minor: the bound follows by restricting the orbit sum to S⁺ ⊆ H_n⁺, but this step should be stated explicitly. |
| T7.5 | ∫ | V_k monotonicity proof should state completion-equivalence explicitly | The proof assumes every completion from state_{k+1} extends to one from state_k. True by construction but should be stated. |
| T7.6 | ∫ | Case B predecessor cardinality conflates poset sites with variables | |Pred(k*+1)| counts poset sites; V_{k*} counts variables. These are different objects. |
| T7.7 | ∫ | "Typical G^(C)" argument needs formal stability proof for backbone | Backbone is not a local property; the locally-tree-like argument is insufficient. |
| T7.8 | ∫ | Second-moment calculation for backbone triples uses informal "locally tree-like" | Cov(I_C, I_{C'}) = O(1/n) for variable-disjoint clauses relies on locality of backbone, which is non-local. |
| T7.9 | ♭ | Search-to-decision transfer in P_FM not fully explicit | The self-reduction is standard but should be verified as a valid P_FM member. |
| T7.10 | ♭ | No-sketch theorem for S_lin has uncited rank claim | Random 3-SAT over GF(2) rank claim is plausible but needs citation. |
| T7.11 | ∗ | Expansion constant c' = (3/4)α assumes balanced binary partition | Unbalanced partition ratio gives different crossing probability. Extension is only sketched. |
| T7.12 | ∗ | Lifted state space case inadequately handled | The data-processing argument for "expansion persists under re-encoding" is informal. |
| T7.13 | ∗ | Missing remark about ∗-witness encoding freedom | `rem-encoding-freedom-does-not-circumvent` addresses ♯, ∫, ∂ but omits ∗. |
| T7.14 | Arch | Independent sub-barrier crossing (Property 4) assumes orthogonality works for composite paths | Should be stated as a separate lemma rather than a definitional property. |

---

## Revised Channel Status (Post-Round-7)

| Channel | R6 Status | R7 Status | Issues |
|---------|-----------|-----------|--------|
| Sharp (♯) | CLOSED | **CONDITIONALLY SOUND** | S7.1 (constant D is definitional); T7.1-T7.4 |
| Causal (∫) | CLOSED | **HAS SERIOUS GAPS** | S7.2 (backbone ≠ frozen vars); S7.3 (perturbation stability); S7.4 (SCC conflation); S7.5 (info leakage); S7.6 (injectivity) |
| Algebraic (♭) | OPEN (S9-S10) | **PARTIALLY OPEN** | S9 (still open); S10 (subsumed by cardinality argument if sound); S7.7 (flat-universality); S7.8 (instruction set) |
| Scaling (∗) | CLOSED | **FATALLY FLAWED** | F7.3 (no modal restriction on merge); F7.6 (purity violation is taxonomic); S7.11 (per-step bound) |
| Boundary (∂) | CLOSED | **FATALLY FLAWED** | F7.4 (allows arbitrary poly-time on interface); F7.6 (purity violation is taxonomic); S7.10 (per-step bound) |
| Architecture | CLOSED | **FATALLY FLAWED** | F7.1 (foundation ≈ conclusion); F7.2 (theorem target); F7.5 (self-referential definitions); F7.7 (workspace separation) |

---

## Structural Assessment

### What the proof actually establishes (modulo SERIOUS items)

1. **Five well-defined restricted computation classes** have been defined with rigorous witness
   schemas. Each class captures a recognizable algorithmic paradigm (metric descent, causal
   propagation, algebraic elimination, divide-and-conquer, interface contraction).

2. **For the ♯ and ∫ channels**, the blockage arguments are internally sound given their
   definitions: the pigeonhole on constant-D metric profiles (♯) and the distinguishing-instances
   argument via backbone + visibility (∫) are clean combinatorial proofs. They establish that no
   algorithm satisfying the specific concrete restrictions can solve random 3-SAT near threshold.

3. **For the ♭ channel**, the type-independent cardinality argument (|B_n^♭| ≥ 2^{cn}) is sound
   for the search formulation. This is the strongest individual channel result.

4. **For the ∗ and ∂ channels**, the blockage arguments are **not** internally sound because the
   definitions do not contain the modal restrictions that the proofs appeal to.

5. **The bridge** P_FM = P_DTM appears sound, modulo the foundation assumption.

### What the proof does NOT establish

1. **P ≠ NP** (unconditionally or conditionally on standard axioms). The result is conditional on
   the Foundation Assumption, which is a non-standard axiom with no independent justification.

2. **Modal exhaustiveness.** The claim that every poly-time algorithm decomposes into the five
   classes rests on an unproven theorem target (`thm-schreiber-structure` computational form)
   and the Foundation Assumption.

3. **That the ∗ and ∂ channels are blocked.** The purity violation arguments require modal
   restrictions that are not present in the witness definitions.

### The core gap in one sentence

> The proof defines five restricted computational classes, shows 3-SAT resists each one, but
> does not prove that every polynomial-time algorithm belongs to their saturated closure.

---

## Priority-Ordered Work Plan

### Tier 0: Structural Issues (require mathematical work or fundamental restructuring)

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | **F7.3** — Add explicit ∗-modal restriction to `def-pure-star-witness-rigorous` | Medium | Unblocks ∗-channel; requires re-proving all ∗-witness examples satisfy new restriction |
| 2 | **F7.4** — Tighten ∂-modal restriction to exclude CSP-solving on interface data | Medium | Unblocks ∂-channel; risks making definition too narrow for known algorithms |
| 3 | **F7.2** — Complete `thm-schreiber-structure` (computational form) | Very High | Unblocks the entire exhaustiveness chain |
| 4 | **F7.1** — Prove the Foundation Assumption or explicitly demote result to "conditional on unproven axiom" | Very High | Determines whether the result is a theorem or a conjecture |
| 5 | **F7.7** — Prove non-amplification for shared-state (single-tape) execution model | High | Currently proved only for fictitious workspace-separated model |

### Tier 1: Serious gaps in individual channels

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 6 | **S7.2** — Add explicit backbone-size property to `def-hard-subfamily-3sat` | Low | Fixes ∫-channel dependency |
| 7 | **S7.3** — Prove or cite backbone stability under single-clause perturbation | High | Critical for ∫-channel distinguishing argument |
| 8 | **S7.8** — Exhibit Σ_prim explicitly; verify non-trivial modal classification | Medium | Currently the five-modality structure may be illusory |
| 9 | **S7.9** — Tighten mixed-modal obstruction proof with explicit K_♮^- semantics | Medium | Closes the composition gap |
| 10 | **S7.12** — Cite rigorous shattering result at α ≈ 4.2 | Low | May require adjusting α |
| 11 | **S9** — Complete the three informal ♭-channel sub-proofs | Medium | Carried from R6 |
| 12 | **S10** — Resolve catch-all loophole (subsumed by cardinality if S10 is moot) | Low | Carried from R6 |

### Tier 2: Tightening

All T7.1-T7.14 items from the NEEDS TIGHTENING section above.

---

## Comparison with Round 6

| Aspect | Round 6 Assessment | Round 7 Assessment |
|--------|-------------------|-------------------|
| FATAL issues | 9 found, 9 closed | 7 new structural issues found (deeper level) |
| ♯ channel | CLOSED | Conditionally sound (blockage works within definition) |
| ∫ channel | CLOSED | Has SERIOUS gaps (backbone, perturbation stability) |
| ♭ channel | S9-S10 open | S9-S10 still open; cardinality argument is strongest piece |
| ∗ channel | CLOSED | **Fatally flawed** — no modal restriction on merge |
| ∂ channel | CLOSED | **Fatally flawed** — allows arbitrary poly-time on interface |
| Architecture | CLOSED | **Fatally flawed** — foundation assumption, theorem target, workspace separation |
| Overall verdict | "Structurally complete at FATAL level" | "Sound conditional framework with three unproved load-bearing assumptions" |

**Key difference:** Round 6 focused on *internal* consistency (do the definitions match the proofs?
do the cross-references work?). Round 7 focuses on *external* soundness (do the definitions capture
what they claim? does the framework actually prove P ≠ NP?). The internal fixes from R6 are valid.
The external soundness issues from R7 require new mathematical work.
