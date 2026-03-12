# Round 8 Hostile-Referee Review: 05_algorithmic.md (P != NP Proof)

## Metadata
- Reviewed file: docs/source/2_hypostructure/09_mathematical/05_algorithmic.md
- Review date: 2026-03-12
- Reviewer: Claude Opus 4.6 (5 parallel hostile-referee agents, fresh from-scratch pass)
- Scope: Full 10,689-line document — all five modal channels, architecture, metatheorems,
  barrier decomposition, primitive audit, bridge, and summary
- Previous rounds: 7 (Rounds 1-6 focused on internal consistency; Round 7 focused on external
  soundness. Round 8 is a complete from-scratch re-review of the current document state.)

## Disposition of Prior Findings

F7.1 (Foundation Assumption ≈ conclusion) and F7.2 (thm-schreiber-structure theorem target)
are treated as **RESOLVED** per author's confirmation. The document handles both:
- The Foundation Assumption is framed as a foundational modeling choice (like ZFC), not as a
  smuggled conclusion. The conditionality is surfaced in `rem-conditional-status-classical-export`
  and `thm-conditional-nature`.
- `thm-schreiber-structure` has a complete proof (lines 120-148), with the computational bridge
  completed through `lem-primitive-step-classification` → `thm-appendix-a-primitive-audit-table` →
  `thm-witness-decomposition` → `cor-computational-modal-exhaustiveness`.

This review finds issues **independent of** F7.1 and F7.2.

---

## Executive Summary

Five parallel agents attacked the full document with maximal hostility. After deduplication
and removal of issues subsumed by the resolved F7.1/F7.2, the review identifies:

| Severity | Count |
|----------|-------|
| FATAL | 4 |
| SERIOUS | 16 |
| NEEDS TIGHTENING | 18 |

The four FATAL issues are:

1. **F8.1** — The coend formula in `thm-schreiber-structure` does not follow from the fracture-
   square proof. The proof establishes pullback decompositions but the theorem claims a coend
   (colimit) decomposition. These are categorically dual.

2. **F8.2** — The ∗-witness definition has NO explicit modal restriction on the merge map.
   The scaling blockage proof appeals to "∗-purity forbids ♭-modal operations in the merge,"
   but no item in the definition restricts what the merge can compute.

3. **F8.3** — The ∂-witness definition permits "arbitrary polynomial-time computation on
   interface data" (item 7), then the blockage proof claims interface contraction "cannot
   solve a CSP" because that's "♭-type." The definition explicitly allows it.

4. **F8.4** — Workspace separation is a property of the mathematical decomposition, not of
   the algorithm's actual state space. A single-tape TM has a shared state; the product
   decomposition Z̃ = ∏Z_n^♮ does not exist in the algorithm's execution.

### What IS sound

- The ♯-channel pigeonhole argument (given constant D in the definition)
- The ∫-channel backbone-triple argument (the transfer lemma, Sub-claim B2)
- The ♭-channel type-independent cardinality argument (for the search formulation)
- The bridge equivalence P_FM = P_DTM
- The primitive audit table at the instruction level
- The overall proof architecture and layering
- The conditionality framing in `thm-conditional-nature`

---

## FATAL Issues

### F8.1: The coend formula in `thm-schreiber-structure` is not derived from the proof

**Location:** `thm-schreiber-structure` (lines 99-148)

**The theorem states:**

$$\mathrm{Hom}_{\mathbf{H}}(\mathcal{X}, \mathcal{Y}) \simeq \int^{\lozenge \in \{\int, \flat, \sharp\}} \mathrm{Hom}_{\lozenge\text{-modal}}(\lozenge\mathcal{X}, \lozenge\mathcal{Y})$$

**The proof establishes:** Fracture square pullback decompositions (Schreiber Theorem 3.8.5):
- $\mathcal{X} \simeq \int\mathcal{X} \times_{\int\flat\mathcal{X}} \flat\mathcal{X}$
- $\mathcal{X} \simeq \flat\mathcal{X} \times_{\flat\sharp\mathcal{X}} \sharp\mathcal{X}$

**The gap:** Fracture squares give *limit* (pullback) decompositions of objects. The theorem claims
a *colimit* (coend) decomposition of hom-spaces. These are categorically dual constructions.
The proof invokes the density theorem (nerve-realization adjunction), but the density theorem
gives a coend decomposition over a *dense subcategory of representables*, not over three
modalities. The connection between the dense subcategory and $\{\int, \flat, \sharp\}$ is
not established.

Additionally, the coend notation $\int^{\lozenge}$ requires a functor
$F: \mathcal{C}^{\mathrm{op}} \times \mathcal{C} \to \mathcal{D}$ for some category $\mathcal{C}$,
and no such functor or indexing category is specified.

**Impact:** The coend formula is the formal statement that "every morphism decomposes into modal
components." If the formula is wrong, the theorem proves something weaker than stated (that
objects decompose as pullbacks, not that hom-spaces decompose as coends). The downstream
proof chain (`cor-exhaustive-decomposition`, the five-modality completeness argument) needs
to be checked against what is actually proved, not what is stated.

**What would resolve this:** Either (a) prove the coend formula from the fracture squares
(this requires a non-trivial argument involving the pullback universal property and the
adjunction counit/unit maps), or (b) replace the coend formula with what is actually proved
(the pullback/fracture decomposition), and verify that the downstream proof chain uses only
the pullback version.

---

### F8.2: The ∗-witness definition has NO explicit modal restriction on the merge map

**Location:** `def-pure-star-witness-rigorous` (lines 1338-1390)

**Comparison of the five witness definitions:**

| Witness | Modal restriction item | What it restricts |
|---------|----------------------|-------------------|
| ♯ | Item 6 | F_n^♯ factors through (D+2)-tuple metric profile |
| ∫ | Item 8 | U_{n,i} accesses only Vis(i) clauses |
| ♭ | Item 6 | e_n^♭ built from certified Σ-primitives |
| ∗ | **NONE** | Merge is "uniformly polynomial-time" — no restriction |
| ∂ | Item 7 | C_n^∂ operates on interface data — but allows "arbitrary poly-time" |

The merge map in item 4 of the ∗-witness is specified only as "uniformly polynomial-time."
A merge that internally performs Gaussian elimination, unit propagation, or gradient descent
is perfectly compatible with items 1-7. The scaling blockage proof then claims (line 2702):
"Under ∗-purity, the merge cannot invoke ♭-modal operations." But no item in the definition
excludes ♭-modal operations from the merge.

**Why this is fatal:** The entire scaling-channel blockage argument is: the merge must reconcile
crossing constraints, which requires ♭/♯/∫-type operations, which are "forbidden by ∗-purity."
But ∗-purity, as defined, forbids nothing except non-polynomial computation. The "purity
violation" is taxonomic (labeling the operation as "♭-type"), not structural (the operation is
actually excluded by the definition).

**What would resolve this:** Add an explicit "∗-modal restriction" item to
`def-pure-star-witness-rigorous` constraining what the merge map can compute — analogous to
item 6 of the ♯-witness or item 8 of the ∫-witness. For example: "The merge operates only
on recursion-tree metadata (subproblem answers, split coordinates) and cannot access the
original input's constraint/metric/algebraic structure."

---

### F8.3: The ∂-witness definition permits "arbitrary polynomial-time computation on interface data"

**Location:** `def-pure-boundary-witness-rigorous` item 7 (lines 1447-1474)

Item 7 states: "C_n^∂ may ... perform arbitrary polynomial-time computation on data that is
explicitly present in the interface object."

The blockage proof (Step 4 of `lem-boundary-obstruction`, and the canonical 3-SAT boundary
blockage) then claims the contraction "cannot solve a CSP on separator variables" because that
is a "♭-type operation."

**The problem:** The interface object B_n^∂ encodes the crossing constraints (the boundary
extraction map ∂_n compresses the bulk into the interface). The contraction C_n^∂ can read
those constraints from B_n^∂ and perform "arbitrary polynomial-time computation" on them —
which by the definition includes solving a CSP if a polynomial-time CSP solver exists.

The proof's response — "solving a CSP is structurally a ♭-operation, and ∂-purity forbids
♭-operations" — fails because the definition explicitly allows "arbitrary polynomial-time
computation" on the interface data. The categorical labels do not restrict what bits can be
flipped.

**What would resolve this:** Either (a) restrict the ∂-witness to forbid CSP-solving operations
explicitly (by constraining C_n^∂ to boundary-map compositions only, not "arbitrary poly-time"),
or (b) prove that B_n^∂ does not contain enough information to specify the crossing constraints
(an information-theoretic bound on interface compression, not just a polynomial-size bound).

---

### F8.4: Workspace separation conflates mathematical decomposition with algorithmic state

**Location:** `def-modal-barrier-decomposition` property 2; `thm-modal-non-amplification`

The non-amplification principle (Part I: Channel Isolation) claims that non-♯ steps contribute
zero to E_♯ variation. This relies on workspace separation: each modality operates in its own
workspace Z_n^♮, and a ♭-step acting on Z_n^♭ cannot affect Z_n^♯.

**The problem:** An actual Turing machine has a **single shared tape**. There are no "separate
workspaces." The decomposition into five workspaces is imposed by the factorization tree
analysis — it is a mathematical artifact, not a property of the algorithm's execution. When
the algorithm runs, every step can read and write every bit of the tape.

The extended configuration space Z̃ = ∏ Z_n^♮ is NOT the algorithm's actual state space.
The algorithm's state space is {0,1}^{O(p(n))} (the tape). The projection from tape to Z̃
is via the encoding maps, which are arbitrary polynomial-time functions. Nothing prevents
R^♯ (the ♯ reconstruction map) from writing bits that E^∫ (the ∫ encoding) will later read,
creating a cross-modal information pathway.

**What would resolve this:** Prove that the encoding/reconstruction pipeline
R^♯ ∘ F^♯ ∘ E^♯ cannot affect ∫-energy even when E^∫ can read the full shared state.
This requires analyzing the composition of encoding maps across modality boundaries.

---

## SERIOUS Issues

### S8.1: The adjunction ordering $\flat \dashv \int \dashv \sharp$ is non-standard

**Location:** Lines 58-64 (def-five-modalities)

The standard ordering in Schreiber's work and the cohesive topos literature is
$\int \dashv \flat \dashv \sharp$. The document's ordering would make $\flat$ the left adjoint
of $\int$, reversing the standard relationship. The reduction identities are stated correctly
but the adjunction order creates a discrepancy that undermines confidence in the authors'
command of the material.

---

### S8.2: The $\partial \circ \ast$ argument uses an axiom not listed in the cohesion axioms

**Location:** Lines 321-344 (rem-five-modality-completeness-argument, Case 7)

The argument that $\partial(\ast X) \simeq 0$ requires showing that $\ast X$ is $\sharp$-modal.
This in turn requires that $\mathrm{Disc}(S) \simeq \mathrm{coDisc}(S)$ for discrete $S$. But
this is **not** one of the three cohesion axioms listed in `def-cohesive-topos-computation`.
The document acknowledges (lines 339-341) that "$\flat$-modality alone does not imply
$\sharp$-modality in a general cohesive $\infty$-topos without this axiom." But the axiom
is never explicitly added to the definition. If $\partial \circ \ast \not\simeq 0$, there could
be a sixth independent modality.

---

### S8.3: The 0-truncation lemma proof does not establish finite-presentation preservation

**Location:** Lines 354-385 (lem-zero-truncation-modal-preservation)

Part (b) claims "the adjoints $\Pi$, $\Gamma$ applied to finitely presented types yield finitely
presented outputs." This is unjustified: $\Gamma$ is a right adjoint and need not preserve finite
presentation. The claim that "modal maps in the fracture squares are constructive and
polynomial-time when restricted to finitely presented objects" is a computational claim
requiring separate argument.

---

### S8.4: Constant D in the ♯-witness is definitional, not derived

**Location:** `def-pure-sharp-witness-rigorous` item 6; `rem-constant-D-metric-profile`

The pigeonhole argument requires D to be a universal constant independent of n. This is
*placed into the definition*, not derived. An algorithm evaluating n different energy
differences per step (e.g., WalkSAT computing make/break counts for all variables) uses
D = O(n). The remark at line 2244 concedes such an algorithm "should be classified as
∫-modal" — this is a classification decree, not a theorem.

The constant-D restriction does all the work in the ♯-blockage proof. The obstruction is
correct within the definition but pushes the burden onto modal exhaustiveness.

---

### S8.5: The shape obstruction proof (frontend) is a one-paragraph intuition sketch

**Location:** `lem-shape-obstruction` (lines 2286-2298)

The proof is a single paragraph about frustrated cycles creating circular dependencies. It
does not formally define "frustrated cycle" in terms of the ∫-witness structure, does not
show why local updates "cannot be corrected," and does not address alternative state-space
decompositions. The text acknowledges this is a "frontend obstruction" with the full proof
deferred, but since the overall P ≠ NP argument requires blocking all five channels, this
incompleteness is load-bearing.

(Note: the *transfer lemma* `lem-causal-arbitrary-poset-transfer` IS a rigorous proof.
This issue is about the frontend lemma only.)

---

### S8.6: The flat obstruction proof sketches (Steps 3a, 3b) are not proofs

**Location:** `lem-flat-obstruction` Steps 3a-3b (lines 2442-2470)

Step 3a claims frozen coordinates constitute "Ω(n) algebraically independent generators" but
provides no formal argument for why elimination forces 2^{Ω(n)} intermediate states.
Step 3b claims Boolean fiber rigidity makes monodromy trivial, but even Boolean solutions
can have non-trivial monodromy when parameters vary (solutions appear/disappear).

The claims about blocking all six signature families defer to
`thm-random-3sat-algebraic-blockage-strengthened` without individual arguments for
$\mathfrak{S}_{\mathrm{lin}}$, $\mathfrak{S}_{\mathrm{rank}}$, $\mathfrak{S}_{\mathrm{fourier}}$,
$\mathfrak{S}_{\mathrm{polyid}}$.

---

### S8.7: The "purity violation" argument pattern conflates task with mechanism

**Location:** Steps 4 of `lem-scaling-obstruction` and `lem-boundary-obstruction`; also in
canonical blockage lemmas for ∗ and ∂ channels

Multiple blockage proofs argue: (1) the computation must perform task X; (2) task X is
"a ♭-type operation"; (3) the current modality forbids ♭-type operations; (4) contradiction.

Step 2 classifies the *task* (extensional input-output specification) as belonging to a
modality, but modalities classify *mechanisms*. A function that extensionally computes the
same thing as "constraint satisfaction" might do so through a mechanism classified differently.

The ♯-obstruction avoids this trap by using the pigeonhole (the profile space is too small —
a genuinely extensional argument). The ∫-obstruction also avoids it (the backbone-triple
argument is about information access, not mechanism classification). The ♭, ∗, and ∂
obstructions rely on the purity-violation pattern.

---

### S8.8: All five universality theorems leave the (1)⇒(2) direction unproved

**Location:** `thm-sharp-universality` through `thm-partial-universality`

Every universality theorem proves only (2)⇒(1) (trivial) and (2)⇔(3) (definitional).
The critical direction (1)⇒(2) — that every algorithm factoring abstractly through modality
♮ admits an explicit pure ♮-witness — is deferred. Each associated remark says the soundness
chain "does not invoke this direction." This means the universality theorems are stated as
biconditionals but proved as one-directional implications.

The workaround via the primitive audit table is acknowledged, but the universality theorems
should either be weakened to match what is proved or the missing direction should be proved.

---

### S8.9: At the instruction level, only 2 of 5 modalities are instantiated

**Location:** `thm-concrete-instruction-audit` (lines 9848-9881)

The audit classifies 28 instructions as: 22 administrative (PT), 5 arithmetic (FLAT), 1
comparison (SH). The families IN (causal), STAR (recursive), and PARTIAL (boundary) are never
instantiated at the instruction level. The corollary (line 10049-10053) acknowledges they
"arise at the algorithmic level when entire subroutines are classified as semantic units."

This means the five-modality classification is a property of algorithm-level composition,
not of computational primitives. This is weaker than what the main text suggests.

---

### S8.10: The administrative classification hides algorithmic structure

**Location:** `thm-concrete-instruction-audit` (lines 9848-9881)

22 of 28 instructions are "administrative" (presentation translators) — semantically inert
data routing. But the routing IS the algorithm: a sorting algorithm and a graph search use
the same primitives; what distinguishes them is the routing structure. By classifying routing
as inert, the framework assumes algorithmic structure is fully captured by progress-producing
leaves. This is never proved and is essentially equivalent to the claim being proved.

---

### S8.11: The flat-channel cardinality argument works only for search, not decision

**Location:** `lem-random-3sat-integrality-blockage` (lines 7524-7612)

The type-independent cardinality argument constructs 2^{cn} formulas with pairwise-disjoint
solution sets, forcing |B_n^♭| ≥ 2^{cn}. This works for the *search* version (output a
satisfying assignment). For the *decision* version (output SAT/UNSAT), all formulas in H_n
are satisfiable, so a single element of B_n^♭ suffices. The proof explicitly restricts to
search (line 7541), but the proof chain must consistently use search throughout.

---

### S8.12: The Cook-Levin reduction is a sketch, not a proof

**Location:** `thm-internal-cook-levin-reduction` (lines 8602-8628)

The proof is 26 lines and does not verify: (a) tableau encoding correctness, (b) 3-CNF
conversion preserving satisfiability, (c) polynomial-time in the Fragile model, (d) uniformity.
In a claimed proof of P ≠ NP, every step must be fully rigorous.

---

### S8.13: Barrier datum validity proof uses probabilistic language for a deterministic subfamily

**Location:** `thm-canonical-3sat-barrier-datum-valid` (lines 6549-6599)

B1 and B3 proofs argue "with high probability over hard instances" and "in expectation over
formula draws." But H_n is a deterministic set. The proofs should work directly from the
six structural properties of H_n, not from probabilistic arguments about the random ensemble.

---

### S8.14: Boolean fiber rigidity (Property 6 of def-hard-subfamily-3sat) lacks proof/reference

**Location:** `def-hard-subfamily-3sat` property 6

The claim that the algebraic variety defined by clause polynomials (with z_j^2 - z_j = 0)
coincides with the Boolean solution set, with trivial monodromy, is non-standard. Over an
algebraically closed field, the variety may have additional solutions. No precise reference
is given.

---

### S8.15: Main results summary presents conditional results as unconditional theorems

**Location:** `thm-hypo-algorithmic-main-results` (lines 10623-10689)

"Theorem 8 (Direct 3-SAT Exclusion Route)" and "Theorem 12 (Classical Export)" are presented
without conditionality qualifiers. This contradicts the honest conditional framing in
`thm-conditional-nature` (lines 10542-10595), which correctly states C1 ∧ C2 ∧ C3 ⟹
P_DTM ≠ NP_DTM.

---

### S8.16: The "scaling" frontend certificate has no corresponding modality

**Location:** Appendix B certificate table (line 10286)

The table includes a "Scaling" certificate K_{SC_λ}^{super} but the five-modality framework
has {♯, ∫, ♭, ∗, ∂}. There is no "scaling" modality. The mapping from this certificate to
one of the five modalities is not stated.

---

## NEEDS TIGHTENING

| # | Channel | Issue |
|---|---------|-------|
| T8.1 | ♯ | Steps 1-3 of `lem-sharp-obstruction` are decorative; the obstruction is purely the Step 4 pigeonhole from constant D. Misleading presentation. |
| T8.2 | ♯ | The reduction identity $\flat\int^{(k)} \simeq \int^{(k-1)}$ in the completeness argument (line 294) has a re-indexing error. Correct result: $\flat\int^{(k)} \simeq \int^{(k)}$, yielding the same conclusion $\flat\ast \simeq \ast$ by a different intermediate. |
| T8.3 | ∫ | $\pi_1(\int\mathcal{X}) \neq 0$ claim in `lem-shape-obstruction` is disconnected from the proof (no formal use of fundamental groups). |
| T8.4 | ∫ | The "Extension to Ω(n) failures" (lines 7403-7410) is handwaved; not needed for the conclusion but its mention is misleading. |
| T8.5 | ∫ | Frustrated cycles vs backbone: distinct structural phenomena conflated in presentation. The main lemma uses cycles; the transfer lemma uses backbone. |
| T8.6 | ∫ | Site-variable map injectivity assumed but not required in definition. |
| T8.7 | ♭ | The flat-universality (1)→(2) direction is explicitly unproved (acknowledged as not on critical path). |
| T8.8 | ♭ | The flat-witness primitive basis (lines 1328-1332) is open-ended ("and other algebraic cancellation primitives"), defeating the purpose of formal restriction. |
| T8.9 | ♭ | Pairwise-disjoint solution set argument does not condition on H_n membership. Fixable by union bound. |
| T8.10 | ∗ | The quantitative bound in Step 3 of scaling obstruction gives Ω(n(log n)²) — polynomial, not superpolynomial. Acknowledged but confusing. |
| T8.11 | ∗ | Per-step bound δ_∗(n) ≤ 1 conflates "recursion node" with "computational step." |
| T8.12 | ∂ | Non-planarity (Step 2 of boundary blockage) is supplementary, not load-bearing. Should be labeled as such. |
| T8.13 | ∂ | Per-step bound δ_∂(n) = O(1) is unjustified — a poly-time function on a poly-size interface can modify all bits. |
| T8.14 | Arch | Schreiber citation "Theorem 3.8.5" may be numbered differently in the cited version. Verify. |
| T8.15 | Arch | The "pieces have points" axiom is stated without specifying where epimorphism is meant. |
| T8.16 | Arch | Three of five barrier metatheorems (♭, ∗, ∂) are definitional tautologies, not theorems. |
| T8.17 | Arch | Barrier assembly theorem just restates E13 without new content. |
| T8.18 | Arch | Feynman prose at line 10677 ("explains why some algorithms are fast") overclaims relative to the formal content. |

---

## Revised Channel Status (Post-Round 8)

| Channel | R7 Status | R8 Status | Key Issues |
|---------|-----------|-----------|------------|
| Sharp (♯) | CONDITIONALLY SOUND | **CONDITIONALLY SOUND** | S8.4 (constant D is definitional); T8.1 |
| Causal (∫) | HAS SERIOUS GAPS | **SOUND** (transfer lemma) / **FRONTEND INCOMPLETE** | S8.5 (frontend is sketch); backbone-triple argument is rigorous |
| Algebraic (♭) | PARTIALLY OPEN | **SOUND FOR SEARCH** | S8.11 (search only); S8.6 (proof sketches for 4 of 6 signatures) |
| Scaling (∗) | FATALLY FLAWED | **FATALLY FLAWED** | F8.2 (no modal restriction on merge); S8.7 (purity-violation pattern) |
| Boundary (∂) | FATALLY FLAWED | **FATALLY FLAWED** | F8.3 (allows arbitrary poly-time on interface); S8.7 |
| Architecture | FATALLY FLAWED | **HAS STRUCTURAL GAPS** | F8.1 (coend formula); F8.4 (workspace separation); S8.8-S8.10; S8.15 |

---

## Structural Assessment

### What the proof actually establishes (modulo SERIOUS items)

1. **Five well-defined restricted computation classes** with rigorous witness schemas.
   Each captures a recognizable algorithmic paradigm (metric descent, causal propagation,
   algebraic elimination, divide-and-conquer, interface contraction).

2. **For the ♯ channel**, the pigeonhole blockage is internally sound given constant D.
   The argument is genuinely extensional (profile space too small to distinguish formulas).

3. **For the ∫ channel**, the backbone-triple transfer lemma is the strongest individual
   argument in the document — a clean combinatorial proof that a pure ∫-witness cannot
   solve random 3-SAT near threshold, regardless of poset structure.

4. **For the ♭ channel**, the type-independent cardinality argument is sound for search.
   The 2^{cn} distinct solution-set argument is clean.

5. **For the ∗ and ∂ channels**, the blockage arguments are **not** internally sound
   because the definitions do not contain the modal restrictions the proofs appeal to.

6. **The bridge** P_FM = P_DTM appears sound.

7. **The overall conditional structure** is honestly stated in `thm-conditional-nature`.

### What the proof does NOT establish

1. **That the ∗ or ∂ channels are blocked.** The purity-violation arguments require
   modal restrictions not present in the witness definitions.

2. **That workspace separation holds for actual computations.** The non-amplification
   principle is proved for a fictitious product state space, not for the algorithm's
   actual (single-tape) state space.

3. **The coend formula in thm-schreiber-structure.** The proof establishes fracture-square
   pullbacks, not hom-space coends. The downstream chain may still work but should be
   verified against what is actually proved.

### The core gap in one sentence

> The proof defines five restricted classes, rigorously blocks three of them (♯, ∫, ♭),
> but the definitions of the remaining two (∗, ∂) are too permissive for their blockage
> proofs to go through.

---

## Priority-Ordered Work Plan

### Tier 0: Structural Issues (require new mathematical definitions or proofs)

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 1 | **F8.2** — Add explicit ∗-modal restriction to the merge map | Medium | Unblocks ∗-channel; requires re-verifying that mergesort/quicksort satisfy the new restriction |
| 2 | **F8.3** — Tighten ∂-modal restriction to exclude CSP-solving on interface data | Medium | Unblocks ∂-channel; risks making definition too narrow for FKT |
| 3 | **F8.4** — Prove non-amplification for shared-state (single-tape) execution | High | Currently proved for fictitious workspace-separated model only |
| 4 | **F8.1** — Fix the coend formula or replace with pullback decomposition | Medium | Verify downstream chain uses only what is proved |

### Tier 1: Serious gaps

| Priority | Issue | Effort | Impact |
|----------|-------|--------|--------|
| 5 | **S8.7** — Eliminate purity-violation pattern or prove task-mechanism equivalence | High | Affects ♭, ∗, ∂ channels |
| 6 | **S8.5** — Upgrade ∫ frontend obstruction from sketch to proof | Medium | Currently only transfer lemma is rigorous |
| 7 | **S8.6** — Complete flat obstruction for all 6 signature families | Medium | Currently only 2 argued explicitly |
| 8 | **S8.12** — Write rigorous Cook-Levin proof for Fragile model | Medium | Standard but must be complete |
| 9 | **S8.15** — Make main results summary accurately conditional | Low | Currently contradicts thm-conditional-nature |
| 10 | **S8.14** — Provide reference or proof for Boolean fiber rigidity | Medium | Blocks flat-channel |
| 11 | **S8.8** — Either prove universality (1)⇒(2) or weaken statements | Medium | Currently over-claiming |

### Tier 2: Tightening

All T8.1-T8.18 items from the NEEDS TIGHTENING section above.

---

## Comparison with Round 7

| Aspect | Round 7 Assessment | Round 8 Assessment |
|--------|-------------------|-------------------|
| F7.1 (Foundation Assumption) | FATAL | **RESOLVED** (accepted as foundational choice) |
| F7.2 (Schreiber theorem target) | FATAL → RESOLVED | **RESOLVED** (proof chain verified) |
| ♯ channel | Conditionally sound | **Conditionally sound** (unchanged) |
| ∫ channel | Has serious gaps | **Sound** (transfer lemma) / frontend incomplete |
| ♭ channel | Partially open | **Sound for search** |
| ∗ channel | Fatally flawed | **Fatally flawed** (unchanged — F8.2 = F7.3) |
| ∂ channel | Fatally flawed | **Fatally flawed** (unchanged — F8.3 = F7.4) |
| Workspace separation | Fatal (F7.7) | **Fatal** (F8.4 = F7.7, unchanged) |
| Coend formula | Not identified | **NEW FATAL** (F8.1) |
| Overall FATAL count | 7 (reduced to 6 after F7.2 correction) | **4** |
| Overall SERIOUS count | 12 (13 after F7.2 correction) | **16** |
| Overall NT count | 14 | **18** |

### Key differences from Round 7

1. F7.1 and F7.2 are resolved.
2. F7.5 (self-referential definitions) and F7.6 (taxonomic purity) are now classified as
   SERIOUS (S8.7) rather than FATAL — the issue is real but is a property of the proof
   technique, not a logical contradiction.
3. F8.1 (coend formula) is a new finding not identified in Round 7.
4. More SERIOUS issues identified due to deeper reading of the theorem ladder, primitive
   audit, and barrier framework.
5. The ∫-channel is upgraded: the transfer lemma (backbone-triple argument) is recognized
   as rigorous and sufficient, even though the frontend lemma is still a sketch.
