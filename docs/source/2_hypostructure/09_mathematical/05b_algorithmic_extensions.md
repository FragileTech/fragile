---
title: "Algorithmic Completeness: Extensions and Audit Framework"
---

# Algorithmic Completeness: Extensions and Audit Framework

:::{prf:remark} Companion document scope
:label: rem-companion-document-scope

This companion document to [Algorithmic Completeness](05_algorithmic.md) contains the general-purpose
extensions of the algorithmic completeness framework that are **not required** for the P≠NP proof.

The main document contains the self-contained proof chain:

$$
\text{Parts I--IV: classification and exhaustiveness}
\to
\text{Part V: mixed-modal obstruction theorem}
\to
\text{Part VI: 3-SAT blockage lemmas and separation}
$$

This companion provides optional infrastructure for:
1. **General obstruction calculi** — sound-and-complete calculus frameworks for arbitrary problem families
2. **Proof obligation ledgers and audit trails** — formal implementation criteria
3. **Backend dossiers** — stronger audit artifacts for each modality
4. **Thin contracts and the algorithmic factory** — reusable compilation layer
5. **Primitive audit tables** — semantic primitive classification appendix
6. **Thin-contract packaging** — canonical 3-SAT thin-contract appendix

These extensions are valuable for extending the framework to new problems and for stronger audit trails,
but they are not prerequisites for the P≠NP separation argument itself.
:::

## General Obstruction Calculus Framework

The following definitions and theorems provide a general-purpose obstruction calculus framework.
The P≠NP proof uses only the semantic obstruction propositions $\mathbb{K}_\lozenge^-(\Pi)$ and the
mixed-modal obstruction theorem; it does not require sound-and-complete calculi for arbitrary problems.


### Obstruction Calculus Definitions

:::{prf:definition} Obstruction calculus and finitary certificate schema
:label: def-obstruction-calculus-schema

For each

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\},
$$
a **$\lozenge$-obstruction calculus**

$$
\mathsf{Obs}_\lozenge
$$
is a finitary derivation system whose judgments are of the form

$$
\Pi \vdash_\lozenge^- B,
$$
where $B$ is a finite derivation object, called a **$\lozenge$-obstruction certificate** for $\Pi$.

We write

$$
B\in K_\lozenge^-(\Pi)
$$
iff $B$ is derivable in $\mathsf{Obs}_\lozenge$.

Each obstruction calculus is required to contain:

1. **Translator invariance rules.**
   Modal obstruction is preserved under presentation translators and admissible re-encodings.
2. **Irreducible extraction rules.**
   If a putative solver tree exists, every minimal-rank factorization tree contains irreducible components, and those
   components are the targets of obstruction.
3. **Modality-specific refutation rules.**
   The primitive refutation rules must mirror the clauses of the corresponding universal-property theorem:
   - $\sharp$: failure of polynomially bounded ranking/descent witnesses;
   - $\int$: failure of polynomial-height well-founded dependency elimination;
   - $\flat$: failure of admissible polynomial-size algebraic sketches;
   - $\ast$: failure of polynomially bounded recursive self-reduction trees;
   - $\partial$: failure of admissible polynomial-size interface/boundary contraction schemes.

The calculus $\mathsf{Obs}_\lozenge$ is called:

- **sound** if

  $$
  B\in K_\lozenge^-(\Pi)\Longrightarrow \mathbb K_\lozenge^-(\Pi);
  $$

- **complete** if

  $$
  \mathbb K_\lozenge^-(\Pi)\Longrightarrow \exists B\in K_\lozenge^-(\Pi).
  $$

Thus soundness means “no false blockages,” while completeness means “every genuine blockage has a finitary
certificate.”
:::

:::{prf:definition} Full E13 obstruction package (reconstructed form)
:label: def-e13-reconstructed

Let $\Pi$ be a problem family.

A **full E13 obstruction package** for $\Pi$ is a 5-tuple

$$
\mathbf B_{\mathrm{E13}}(\Pi)
=
(B_\sharp,B_\int,B_\flat,B_\ast,B_\partial)
$$
such that

$$
B_\sharp\in K_\sharp^-(\Pi),\qquad
B_\int\in K_\int^-(\Pi),\qquad
B_\flat\in K_\flat^-(\Pi),\qquad
B_\ast\in K_\ast^-(\Pi),\qquad
B_\partial\in K_\partial^-(\Pi).
$$

Equivalently: a full E13 package is a finite proof package certifying that every irreducible modal route

$$
\sharp,\ \int,\ \flat,\ \ast,\ \partial
$$
is blocked for the problem family $\Pi$.
:::

### Sound-and-Complete Obstruction Theorems

:::{prf:theorem} $\sharp$-Obstruction Soundness and Completeness
:label: thm-sharp-obstruction-sound-complete

There exists a $\sharp$-obstruction calculus

$$
\mathsf{Obs}_\sharp
$$
whose certificate schema

$$
K_\sharp^-(\Pi)
$$
is sound and complete for the semantic obstruction proposition

$$
\mathbb K_\sharp^-(\Pi).
$$

Equivalently, for every problem family $\Pi$:

$$
\exists B\in K_\sharp^-(\Pi)
\iff
\mathbb K_\sharp^-(\Pi).
$$

Moreover, the modality-specific rules of $\mathsf{Obs}_\sharp$ must be complete against the full universal property of
{prf:ref}`thm-sharp-universality`: they must rule out **all** polynomially bounded ranking/Lyapunov witnesses,
fixed-point solved-state witnesses, and reconstruction-consistent descent certificates, not merely glassy examples or
particular spectral-gap heuristics.
:::

:::{prf:remark} What Theorem \ref{thm-sharp-obstruction-sound-complete} must actually prove
:label: rem-what-thm24-must-prove

The current manuscript’s “glassy landscape / no spectral gap / Łojasiewicz failure” language is only a family of
backend indicators. To obtain completeness, the $\sharp$-obstruction calculus must refute the **entire witness space**
from {prf:ref}`def-pure-sharp-witness-rigorous`, not just familiar convex-optimization examples.
:::

:::{prf:theorem} $\int$-Obstruction Soundness and Completeness
:label: thm-int-obstruction-sound-complete

There exists an $\int$-obstruction calculus

$$
\mathsf{Obs}_\int
$$
whose certificate schema

$$
K_\int^-(\Pi)
$$
is sound and complete for the semantic obstruction proposition

$$
\mathbb K_\int^-(\Pi).
$$

Equivalently, for every problem family $\Pi$:

$$
\exists B\in K_\int^-(\Pi)
\iff
\mathbb K_\int^-(\Pi).
$$

Moreover, the modality-specific rules of $\mathsf{Obs}_\int$ must be complete against the full universal property of
{prf:ref}`thm-int-universality`: they must rule out **all** polynomial-height well-founded dependency-elimination
witnesses, including every admissible propagation, dynamic-programming, and inductive-elimination presentation, not
merely explicit DAG failures visible at the surface syntax.
:::

:::{prf:remark} What Theorem \ref{thm-int-obstruction-sound-complete} must actually prove
:label: rem-what-thm25-must-prove

The current “frustrated loops / $\pi_1\neq 0$ / no DAG structure” language is only one frontend realization of failure
of the $\int$-universal property. Completeness requires obstruction of every admissible well-founded elimination
witness, not only the most obvious graph-theoretic one.
:::

:::{prf:theorem} $\flat$-Obstruction Soundness and Completeness (Strengthened)
:label: thm-flat-obstruction-sound-complete

There exists a $\flat$-obstruction calculus

$$
\mathsf{Obs}_\flat
$$
whose certificate schema

$$
K_\flat^-(\Pi)
$$
is sound and complete for the semantic obstruction proposition

$$
\mathbb K_\flat^-(\Pi).
$$

Equivalently, for every problem family $\Pi$:

$$
\exists B\in K_\flat^-(\Pi)
\iff
\mathbb K_\flat^-(\Pi).
$$

Moreover, the modality-specific rules of $\mathsf{Obs}_\flat$ must be complete against the full strengthened universal
property of {prf:ref}`thm-flat-universality`: they must rule out **all** admissible polynomial-size algebraic sketches

$$
X_n \xrightarrow{s_n} A_n \xrightarrow{e_n} B_n \xrightarrow{d_n} Y_n,
$$
including:
- quotient and congruence compression,
- linear elimination,
- rank and determinant arguments,
- Fourier-type transforms,
- polynomial-identity and cancellation methods,
- monodromy/Galois simplifications (including the vacuity of $\mathfrak S_{\mathrm{mono}}$ for Boolean systems),
- and every other admissible polynomial-size algebraic compression over the allowed signatures.

In particular, it is **not sufficient** for $\mathsf{Obs}_\flat$ to test only visible automorphism groups, only
integrality, or only the vacuity of solvable monodromy for Boolean fibers.
:::

:::{prf:remark} Why Theorem \ref{thm-flat-obstruction-sound-complete} is the hardest algebraic repair
:label: rem-why-thm26-is-hardest

This theorem is the single biggest algebraic repair to the current obstruction layer. A calculus that blocks only
“nontrivial symmetry group” or only the current E4/E11 pair still leaves open polynomially succinct algebraic
cancellation mechanisms that do not arise from obvious automorphism quotients.
:::

:::{prf:theorem} $\ast$-Obstruction Soundness and Completeness
:label: thm-star-obstruction-sound-complete

There exists an $\ast$-obstruction calculus

$$
\mathsf{Obs}_\ast
$$
whose certificate schema

$$
K_\ast^-(\Pi)
$$
is sound and complete for the semantic obstruction proposition

$$
\mathbb K_\ast^-(\Pi).
$$

Equivalently, for every problem family $\Pi$:

$$
\exists B\in K_\ast^-(\Pi)
\iff
\mathbb K_\ast^-(\Pi).
$$

Moreover, the modality-specific rules of $\mathsf{Obs}_\ast$ must be complete against the full universal property of
{prf:ref}`thm-star-universality`: they must rule out **all** polynomially bounded recursive self-reduction trees,
including every admissible split/merge presentation with strict size decrease and polynomial total recursion-tree size,
not merely obvious supercritical Master-theorem failures.
:::

:::{prf:remark} What Theorem \ref{thm-star-obstruction-sound-complete} must actually prove
:label: rem-what-thm27-must-prove

The current “boundary dominates any balanced cut” language is a good frontend obstruction for many instances, but the
complete theorem must exclude every admissible recursive self-reduction witness, including irregular and nonuniform
recursive geometries that may not fit a single textbook recurrence template.
:::

:::{prf:theorem} $\partial$-Obstruction Soundness and Completeness (Strengthened)
:label: thm-boundary-obstruction-sound-complete

There exists a $\partial$-obstruction calculus

$$
\mathsf{Obs}_\partial
$$
whose certificate schema

$$
K_\partial^-(\Pi)
$$
is sound and complete for the semantic obstruction proposition

$$
\mathbb K_\partial^-(\Pi).
$$

Equivalently, for every problem family $\Pi$:

$$
\exists B\in K_\partial^-(\Pi)
\iff
\mathbb K_\partial^-(\Pi).
$$

Moreover, the modality-specific rules of $\mathsf{Obs}_\partial$ must be complete against the full strengthened
universal property of {prf:ref}`thm-boundary-universality`: they must rule out **all** admissible polynomial-size
boundary or interface representations

$$
X_n \xrightarrow{b_n} I_n \xrightarrow{c_n} O_n \xrightarrow{r_n} Y_n
$$
whose contraction complexity is polynomial in interface size, including:
- planar/Pfaffian reductions,
- bounded-treewidth boundary contractions,
- tensor-network contractions with polynomial-width interfaces,
- holographic and matchgate reductions,
- and every other admissible boundary/interface contraction mechanism in the ambient foundation.

In particular, it is **not sufficient** for $\mathsf{Obs}_\partial$ to test only non-planarity, only absence of a
Pfaffian orientation, or only unbounded treewidth.
:::

:::{prf:remark} Why Theorem \ref{thm-boundary-obstruction-sound-complete} is the second major repair
:label: rem-why-thm28-is-second-major-repair

This theorem is the boundary-side analogue of the $\flat$ strengthening. A calculus that blocks only the currently
named Pfaffian/treewidth pathways still leaves open other polynomially succinct interface contractions not yet captured
by the existing examples.
:::

### Biconditional Hardness and Frontend Compatibility

:::{prf:corollary} Computational Hardness from Complete Obstruction
:label: cor-hardness-from-complete-obstruction

Let $\Pi$ be a problem family. If, for each modality

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\},
$$
the obstruction schema $K_\lozenge^-$ is sound and complete, then

$$
\Pi\in P_{\mathrm{FM}}
\iff
\neg\Bigl(
\mathbb K_\sharp^-(\Pi)\wedge
\mathbb K_\int^-(\Pi)\wedge
\mathbb K_\flat^-(\Pi)\wedge
\mathbb K_\ast^-(\Pi)\wedge
\mathbb K_\partial^-(\Pi)
\Bigr).
$$

Equivalently: a problem family lies outside $P_{\mathrm{FM}}$ exactly when every irreducible modal route is blocked.
:::

:::{prf:proof}
The reverse implication is {prf:ref}`thm-mixed-modal-obstruction`. For the forward implication, suppose
$\Pi\in P_{\mathrm{FM}}$. Then $\mathsf{Sol}_{\mathrm{poly}}(\Pi)\neq\varnothing$. Choose a polynomial-time correct
solver $\mathcal A$. By {prf:ref}`thm-witness-decomposition` and
{prf:ref}`thm-irreducible-witness-classification`, some irreducible modal component of one of the five classes occurs
in a minimal-rank factorization tree for $\mathcal A$. Therefore at least one semantic obstruction proposition fails.
:::

:::{prf:proposition} Compatibility with the current tactic-level certificates
:label: prop-compatibility-with-current-tactics

The current manuscript's tactic-level negative certificates are admissible **frontend realizations** of the
reconstructed obstruction theory provided they are proved to derive the corresponding semantic modal obstructions.

Concretely, the desired implication pattern is:

$$
K_{\mathrm{LS}_\sigma}^- \Rightarrow K_\sharp^-,
$$

$$
K_{\mathrm{E6}}^- \Rightarrow K_\int^-,
$$

$$
K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \Rightarrow K_\flat^-,
$$

$$
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \Rightarrow K_\ast^-,
$$

$$
K_{\mathrm{E8}}^- \Rightarrow K_\partial^-.
$$

Accordingly, the current six-term antecedent package

$$
K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \wedge K_{\mathrm{E8}}^-
$$
should be treated as a tactic-level sufficient condition for the reconstructed five-certificate E13 package.

If the frontend tactics are later shown complete relative to the semantic calculi
$\mathsf{Obs}_\sharp,\mathsf{Obs}_\int,\mathsf{Obs}_\flat,\mathsf{Obs}_\ast,\mathsf{Obs}_\partial$, then the current
antecedent package becomes not merely sufficient but extensionally equivalent to the reconstructed E13 obstruction
package.
:::

:::{prf:remark} Why Proposition \ref{prop-compatibility-with-current-tactics} matters
:label: rem-why-prop-compatibility-matters

This proposition is what lets the manuscript preserve the current tactic names and the current E13 automation story
while still upgrading the foundation to the stronger sound-and-complete obstruction layer required by Parts III and IV.
Without this compatibility statement, the new obstruction theory and the old tactic labels would talk past each other.
:::

:::{prf:remark} Proper falsifiability statement for the obstruction layer
:label: rem-proper-falsifiability-obstruction-layer

Part V yields the following precise falsifiability statement.

If a problem family $\Pi$ is later shown to admit a polynomial-time correct solver despite a full E13 obstruction
package, then at least one of the following must have failed:
1. the soundness of one of the calculi
   $\mathsf{Obs}_\sharp,\mathsf{Obs}_\int,\mathsf{Obs}_\flat,\mathsf{Obs}_\ast,\mathsf{Obs}_\partial$;
2. the irreducible witness classification theorem
   {prf:ref}`thm-irreducible-witness-classification`;
3. the witness decomposition theorem
   {prf:ref}`thm-witness-decomposition`;
4. the computational modal exhaustiveness theorem
   {prf:ref}`cor-computational-modal-exhaustiveness`.

This is the exact theorem-level replacement for the current informal “Class VI algorithm would breach the framework”
language.
:::

### VII. Proof Implementation, Audit Trail, and Completion Criteria

:::{prf:remark} Role of Part VII
:label: rem-role-of-part-vii

Parts I--VI state the mathematical framework, the classification theorems, the obstruction theory, and the canonical
$3$-SAT instantiation. What still remains is to make precise what it means to **implement** those theorems in a form
that a hostile referee can audit line by line.

Part VII does not add a new mathematical modality or a new hardness principle. Instead, it specifies:
1. the finite list of proof obligations that must be discharged;
2. the admissible form of a proof package for each obligation;
3. the audit artifacts that must be produced;
4. the dependency discipline preventing circularity;
5. and the exact criterion under which the full separation export is considered complete.

In particular, this section replaces all informal phrases such as “the framework is verifiable” or “the obstruction
package is computable” by explicit proof-completion criteria.
:::

:::{prf:definition} Proof obligation
:label: def-proof-obligation

A **proof obligation** is a tuple

$$
\mathcal O = (\mathrm{name},\ \mathrm{statement},\ \mathrm{deps},\ \mathrm{artifacts},\ \mathrm{validators}),
$$
where:

1. $\mathrm{name}$ is a unique identifier;
2. $\mathrm{statement}$ is a theorem-, lemma-, proposition-, or definition-level target;
3. $\mathrm{deps}$ is the finite list of prior statements on which the proof is allowed to depend;
4. $\mathrm{artifacts}$ is the list of concrete mathematical objects that must be exhibited;
5. $\mathrm{validators}$ is the list of checks that must be satisfied in order for the obligation to count as discharged.

A proof obligation is **discharged** if there exists a complete proof of its statement using only its declared
dependencies and if all required artifacts pass all listed validators.
:::

:::{prf:definition} Implementation artifact
:label: def-implementation-artifact

An **implementation artifact** is one of the following:

1. a fully explicit mathematical proof written in the manuscript;
2. a separately archived proof appendix referenced by a stable label;
3. a finite audit table;
4. an explicit algorithm schema or reduction schema;
5. a certificate extractor;
6. a proof-generating derivation in one of the obstruction calculi;
7. or a machine-checkable formalization in a proof assistant.

An artifact is **admissible** only if:
- all data structures and maps are explicitly typed;
- every asymptotic bound is stated with a named polynomial;
- every use of an encoding or translator is traced back to an admissible family;
- and no step appeals merely to “modal exhaustion,” “higher-topos completeness,” or similar slogans without a theorem.
:::

:::{prf:definition} Proof obligation ledger
:label: def-proof-obligation-ledger

The **proof obligation ledger** is the finite family

$$
\mathfrak L
=
(\mathcal O_{\mathrm{I}},\mathcal O_{\mathrm{II}},\mathcal O_{\mathrm{III}},\mathcal O_{\mathrm{IV}},\mathcal O_{\mathrm{V}},\mathcal O_{\mathrm{VI}},\mathcal O_{\mathrm{VII}})
$$
of obligation clusters corresponding to the framework, direct-route, and stronger audit-refinement layers.

The ledger is **complete** if every individual obligation inside each cluster is discharged.

The ledger is **acyclic** if the dependency graph obtained by drawing an arrow

$$
\mathcal O_i \to \mathcal O_j
$$
whenever $\mathcal O_j$ depends on $\mathcal O_i$ contains no directed cycle.
:::

:::{prf:definition} Primitive audit table
:label: def-primitive-audit-table

The **primitive audit table**

$$
\mathcal T_{\mathrm{prim}}
$$
is the finite table indexed by the primitive evaluator instructions of the runtime, containing for each primitive
instruction $\pi$:

1. its typed source and target families;
2. whether it is administrative or progress-producing in the sense of
   {prf:ref}`def-administrative-vs-progress-primitive`;
3. if administrative, the presentation-translator proof witnessing that fact;
4. if progress-producing, at least one modality

   $$
   \lozenge_\pi\in\{\sharp,\int,\flat,\ast,\partial\}
   $$
   together with the pure witness data proving that $\pi$ satisfies the corresponding universal property;
5. the local polynomial runtime bound for $\pi$;
6. translator-invariance and encoding-invariance checks.

The table $\mathcal T_{\mathrm{prim}}$ is **complete** if every primitive instruction appears exactly once.
:::

:::{prf:definition} Modal backend dossier
:label: def-modal-backend-dossier

Fix a modality

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}
$$
and a problem family $\Pi$.

A **modal backend dossier**

$$
\mathcal D_\lozenge(\Pi)
$$
is a finite package containing:

1. the semantic obstruction target

   $$
   \mathbb K_\lozenge^-(\Pi);
   $$
2. the exact witness class to be excluded, as specified by the corresponding universal-property theorem of Part III;
3. the invariant family used to exclude that witness class;
4. the main backend lemmas proving the invariant is incompatible with every admissible witness of that class;
5. the extraction of a finitary obstruction certificate

   $$
   B_\lozenge \in K_\lozenge^-(\Pi).
   $$

The dossier is **complete** if its backend lemmas imply the certificate extraction step and if the certificate passes the
soundness validator of the corresponding obstruction calculus.
:::

:::{prf:definition} Frontend-to-backend bridge dossier
:label: def-frontend-backend-bridge-dossier

Fix a problem family $\Pi$ and a modality $\lozenge$.

A **frontend-to-backend bridge dossier**

$$
\mathcal F_\lozenge(\Pi)
$$
is a proof package showing that a tactic-level certificate from the legacy frontend language implies the corresponding
semantic obstruction certificate of Part V.

Examples include proofs of implications of the form

$$
K_{\mathrm{LS}_\sigma}^- \Rightarrow K_\sharp^-,
\qquad
K_{\mathrm{E6}}^- \Rightarrow K_\int^-,
\qquad
K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \Rightarrow K_\flat^-,
$$

$$
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \Rightarrow K_\ast^-,
\qquad
K_{\mathrm{E8}}^- \Rightarrow K_\partial^-.
$$

A bridge dossier is **complete** if it proves that the frontend certificate suffices to derive the semantic obstruction
required by the strengthened calculus.
:::

:::{prf:proposition} Acceptance criteria for a theorem implementation package
:label: prop-acceptance-criteria-implementation-package

Let $T$ be any theorem or lemma from Parts I--VI.

A proof package for $T$ is acceptable only if it contains all of the following:

1. **Exact statement fidelity.**  
   The proof establishes the theorem exactly as stated, not merely a nearby slogan or heuristic version.

2. **Typed data.**  
   Every object, family, encoding, translator, witness, and certificate appearing in the proof is typed and indexed.

3. **Uniformity proof.**  
   Any claim involving a “family” includes an explicit proof of uniformity with a single code object or a single
   derivation schema.

4. **Polynomial witnesses.**  
   Every use of the word “polynomial” is accompanied by a named polynomial bound and a proof that all intermediate sizes
   stay within that bound.

5. **Translator discipline.**  
   Any preprocessing or postprocessing step used in a pure witness is proved to be a presentation translator unless the
   statement explicitly allows more.

6. **Dependency discipline.**  
   The proof cites only prior obligations in the acyclic ledger and does not use later theorems implicitly.

7. **Failure localization.**  
   The proof package makes clear which exact sublemma would fail if the theorem were false.

A theorem package that omits any of these items does not count as discharged.
:::

:::{prf:theorem} Finite Reduction of the Global Program to an Obligation Ledger
:label: thm-finite-reduction-to-ledger

The stronger audit refinement of the separation program reduces to discharge of the following finite family of
obligation clusters. This ledger over-approximates the direct theorem route by also tracking the thin-contract
compilation layer and the optional stronger backend dossiers used in the semantic implementation.

#### Cluster I: semantics and machine equivalence

$$
\mathcal O_{\mathrm{I}}=
\{\mathrm{I}.1,\mathrm{I}.2,\mathrm{I}.3,\mathrm{I}.4,\mathrm{I}.5,\mathrm{I}.6\},
$$
where:

- **I.1** concrete program syntax, runtime configurations, one-step evaluator semantics, and bit-cost discipline;
- **I.2** finite encodability of all reachable evaluator configurations;
- **I.3** evaluator adequacy (Fragile evaluator simulated by a universal DTM with polynomial slowdown);
- **I.4** `CostCert` soundness;
- **I.5** `CostCert` completeness for internally polynomial-time programs;
- **I.6** DTM $\leftrightarrow$ Fragile compilation/extraction, including the $NP$ verifier version.

#### Cluster II: internal normal forms

$$
\mathcal O_{\mathrm{II}}=
\{\mathrm{II}.1,\mathrm{II}.2,\mathrm{II}.3,\mathrm{II}.4,\mathrm{II}.5\},
$$
where:

- **II.1** administrative normal form for the internal language;
- **II.2** bounded iteration extraction from certified loops;
- **II.3** bounded well-founded recursion extraction from certified recursive programs;
- **II.4** the internal configuration-object construction for DTMs;
- **II.5** extensional equality preservation under polynomial trace reindexing.

#### Cluster III: universal-property witness library

$$
\mathcal O_{\mathrm{III}}=
\{\mathrm{III}.1,\dots,\mathrm{III}.7\},
$$
where:

- **III.1** validator and realization theorem for pure $\sharp$-witnesses;
- **III.2** validator and realization theorem for pure $\int$-witnesses;
- **III.3** validator and realization theorem for strengthened pure $\flat$-witnesses;
- **III.4** validator and realization theorem for pure $\ast$-witnesses;
- **III.5** validator and realization theorem for strengthened pure $\partial$-witnesses;
- **III.6** modal composition theorem;
- **III.7** closure of the saturated modal class under all normal-form constructors.

#### Cluster IV: classification and exhaustiveness

$$
\mathcal O_{\mathrm{IV}}=
\{\mathrm{IV}.1,\mathrm{IV}.2,\mathrm{IV}.3,\mathrm{IV}.4\},
$$
where:

- **IV.1** completion of the primitive audit table $\mathcal T_{\mathrm{prim}}$;
- **IV.2** primitive step classification;
- **IV.3** witness decomposition and irreducible witness classification;
- **IV.4** computational modal exhaustiveness

  $$
  P_{\mathrm{FM}}=\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
  $$

#### Cluster V: obstruction theory

$$
\mathcal O_{\mathrm{V}}=
\{\mathrm{V}.1,\dots,\mathrm{V}.7\},
$$
where:

- **V.1** soundness and completeness of $\mathsf{Obs}_\sharp$;
- **V.2** soundness and completeness of $\mathsf{Obs}_\int$;
- **V.3** soundness and completeness of strengthened $\mathsf{Obs}_\flat$;
- **V.4** soundness and completeness of $\mathsf{Obs}_\ast$;
- **V.5** soundness and completeness of strengthened $\mathsf{Obs}_\partial$;
- **V.6** mixed-modal obstruction theorem;
- **V.7** frontend-to-backend bridge dossiers for the legacy tactic certificates. *(For the $\flat$-channel, all 11
  items of the flat dossier {prf:ref}`def-completion-criteria-flat-dossier-3sat` are now discharged in Part VI, Section
  VI.C.3.)*

#### Cluster VI: canonical $3$-SAT instantiation

$$
\mathcal O_{\mathrm{VI}}=
\{\mathrm{VI}.1,\dots,\mathrm{VI}.5\},
$$
where:

- **VI.1** admissibility of the canonical $3$-SAT family and its witness family;
- **VI.2** verifier membership in $NP_{\mathrm{FM}}$;
- **VI.3** the direct frontend E13 package for canonical $3$-SAT;
- **VI.4** exclusion of canonical $3$-SAT from $P_{\mathrm{FM}}$ via the direct E13 route;
- **VI.5** internal Cook--Levin reduction, $NP_{\mathrm{FM}}$-completeness of canonical $3$-SAT, and the internal
  separation corollary.

#### Cluster VII: thin-contract compilation and stronger audit refinement

$$
\mathcal O_{\mathrm{VII}}=
\{\mathrm{VII}.1,\dots,\mathrm{VII}.5\},
$$
where:

- **VII.1** definition and validation of the five modal thin contracts;
- **VII.2** the algorithmic thin-interface factory and semantic-first compilation theorem;
- **VII.3** the canonical $3$-SAT thin-contract package and its sufficiency theorem;
- **VII.4** optional coherence with the legacy frontend E13 route when both packages are present;
- **VII.5** optional stronger backend dossier realizations of the five thin contracts.

No further foundational cluster is required for the theorem chain itself.
:::

:::{prf:proof}
The finiteness is immediate because:
1. the evaluator instruction set is finite;
2. the normal-form constructors are finite;
3. the modal witness classes are exactly five;
4. the obstruction calculi are exactly five;
5. the target family is fixed to canonical $3$-SAT.

Every theorem in Parts I--VI belongs to one of the first six displayed clusters, and every later theorem depends only on
a finite collection of earlier obligations. The additional thin-contract/factory layer remains finite because there are
exactly five modal contracts and one public algorithmic type. Hence the full program reduces to a finite acyclic
ledger.
:::

:::{prf:theorem} Sufficiency of a Complete Ledger
:label: thm-sufficiency-of-complete-ledger

Assume the proof obligation ledger $\mathfrak L$ is complete and acyclic.

Then all statements of Parts I--VI hold exactly as stated. In particular:

1. the bridge equivalence

   $$
   P_{\mathrm{FM}}=P_{\mathrm{DTM}},
   \qquad
   NP_{\mathrm{FM}}=NP_{\mathrm{DTM}}
   $$
   holds;

2. the computational modal exhaustiveness theorem

   $$
   P_{\mathrm{FM}}=\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle
   $$
   holds;

3. the obstruction calculi are sound and complete against the strengthened universal properties;

4. the canonical $3$-SAT family carries both the direct frontend E13 package and the compiled semantic thin-contract
   package;

5. therefore

   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}},
   \qquad
   P_{\mathrm{FM}}\neq NP_{\mathrm{FM}},
   \qquad
   P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
   $$
:::

:::{prf:proof}
This is a dependency-chasing argument along the acyclic ledger.

- Cluster I yields the semantic bridge and machine equivalence.
- Cluster II yields the normal-form infrastructure needed to reduce arbitrary internally polynomial-time programs to
  finitely analyzable syntax.
- Cluster III yields the five universal-property witness schemas and their closure properties.
- Cluster IV yields decomposition, irreducible classification, and computational modal exhaustiveness.
- Cluster V turns exhaustiveness into hardness logic by proving soundness and completeness of the obstruction layer.
- Cluster VI instantiates the direct theorem route on canonical $3$-SAT and gives the internal separation.
- Cluster VII adds the reusable thin-contract/factory compilation route and the optional stronger audit refinement.

Combining Cluster I with Clusters VI--VII exports the internal separation to the classical DTM classes.
:::

:::{prf:remark} Detailed implementation protocol for Cluster I
:label: rem-implementation-protocol-cluster-i

To discharge Cluster I, the manuscript must provide the following concrete artifacts.

1. **A fixed evaluator grammar.**  
   Give the concrete syntax of programs, the concrete representation of runtime configurations, and the one-step
   transition relation.

2. **A size-growth invariant.**  
   Prove by induction on evaluator steps that every reachable configuration from an input of size $n$ has encoded length
   bounded by a named polynomial in $(|a|,n,t)$.

3. **A universal DTM simulator.**  
   Write an explicit machine that simulates one evaluator step on encoded configurations and prove that the cost of one
   simulation step is polynomial in configuration size.

4. **A certificate extractor for `CostCert`.**  
   The completeness theorem for `CostCert` must include an extraction procedure: from a certified polynomial evaluator
   bound, construct the formal certificate object required by {prf:ref}`def-family-cost-certificate`.

5. **A compiler from DTMs to Fragile programs.**  
   The proof must contain the internal configuration object, the step map, the halt predicate, the output decoder, and
   an explicit bound on simulation overhead.

6. **An extractor from Fragile programs to DTMs.**  
   The proof must show how to take a certified internal program and produce a DTM computing the same extensional family.

A referee should be able to locate each of these six artifacts directly.
:::

:::{prf:remark} Detailed implementation protocol for Cluster II
:label: rem-implementation-protocol-cluster-ii

To discharge Cluster II, implement the normal-form package in the following order.

1. Fix a finite core language of primitive local operations, products/sums, composition, bounded iteration, and
   well-founded recursion.

2. Define an administrative-normal-form translation on program syntax and prove extensional equality with the original
   program.

3. Show that every loop appearing in a certified polynomial-time program can be decorated by an explicit polynomial bound
   extracted from the cost certificate.

4. Show that every recursive call in a certified polynomial-time program can be assigned a well-founded size measure
   whose recursion tree has polynomial size.

5. Prove that the DTM configuration object from Cluster I is expressible in this normal-form language.

6. Prove trace-based extensional equality preservation under polynomial reindexing so that administrative rewrites do not
   change the computed family or its complexity class.

The key implementation principle is: after Cluster II, every polynomial-time program must be reducible to a finite tree
built from finitely many audited primitive leaves and finitely many audited constructors.
:::

:::{prf:remark} Detailed implementation protocol for Cluster III
:label: rem-implementation-protocol-cluster-iii

To discharge Cluster III, build a separate validator library for each modality.

For $\sharp$:
- define the solved-state predicate,
- define the ranking/Lyapunov map,
- prove strict decrease off the solved set,
- prove polynomial bound on the ranking values,
- and prove that the reconstruction map returns the original algorithmic output.

For $\int$:
- define the dependency poset,
- prove well-foundedness and polynomial height,
- define local updates,
- prove predecessor-only dependence,
- and prove that the induced elimination sequence computes the target family.

For $\flat$:
- fix the admissible algebraic signatures,
- define finitely presented algebraic sketches,
- define the allowed polynomial-time algebraic primitives,
- prove polynomial bounds on all intermediate presentations,
- and prove that the algebraic elimination stage covers not only symmetry quotients but also determinant/rank/fourier/
  cancellation style compressions.

For $\ast$:
- define the split and merge maps,
- define the size measure,
- prove strict decrease,
- prove polynomial total recursion-tree size,
- and prove correctness of merged outputs.

For $\partial$:
- define the interface objects,
- define the interface extraction and contraction maps,
- prove polynomial bounds on interface size and all intermediate contractions,
- and prove correctness of reconstruction from contracted interface data.

Only after these five validator libraries exist should the modal composition theorem be proved.
:::

:::{prf:remark} Detailed implementation protocol for Cluster IV
:label: rem-implementation-protocol-cluster-iv

Cluster IV is the core classification package and should be implemented in the following order.

1. **Complete the primitive audit table $\mathcal T_{\mathrm{prim}}$.**  
   This is a finite case split over the runtime instruction set. Do this first.

2. **Prove primitive step classification.**  
   This becomes immediate once the audit table is complete.

3. **Prove witness decomposition.**  
   Use the normal-form theorem from Cluster II and replace every primitive leaf by either:
   - a presentation-translator node, or
   - a pure modal witness supplied by the audit table.

4. **Define witness rank and irreducibility.**  
   The rank must decrease under proper subtree extraction.

5. **Prove irreducible witness classification.**  
   Minimal-rank trees with no nontrivial closure node must consist of a single pure leaf up to translator conjugation.

6. **Derive computational modal exhaustiveness.**  
   The inclusion

   $$
   P_{\mathrm{FM}}\subseteq \mathsf{Sat}\langle\sharp,\int,\flat,\ast,\partial\rangle
   $$
   comes from witness decomposition; the reverse inclusion comes from closure of polynomial time under the modal profile
   constructors.

A referee will scrutinize this cluster first, so every reduction step must be explicit.
:::

:::{prf:remark} Detailed implementation protocol for Cluster V
:label: rem-implementation-protocol-cluster-v

Cluster V should be implemented by treating each obstruction calculus as a small proof system.

For each modality $\lozenge$:

1. define the exact witness class to be excluded;
2. define the judgment form

   $$
   \Pi\vdash_\lozenge^- B;
   $$
3. define the primitive inference rules of the obstruction calculus;
4. prove soundness of each inference rule against the semantic obstruction proposition
   $\mathbb K_\lozenge^-(\Pi)$;
5. prove completeness by showing how to normalize any semantic failure of a pure $\lozenge$-witness into a finite
   certificate derivation.

For $\flat$, the completeness proof must explicitly quantify over all admissible algebraic sketches, not merely over
visible symmetry-based ones.

For $\partial$, the completeness proof must explicitly quantify over all admissible boundary/interface contractions, not
merely planar/Pfaffian/treewidth frontends.

After the five single-modality calculi are complete, prove the mixed-modal obstruction theorem by combining:
- witness decomposition,
- irreducible witness classification,
- and the semantic absence of all five irreducible classes.

Only then should the frontend-to-backend bridge dossiers be inserted for the legacy tactic names.
:::

:::{prf:remark} Detailed implementation protocol for Cluster VI
:label: rem-implementation-protocol-cluster-vi

Cluster VI is the only problem-specific cluster.

To discharge it rigorously, implement the following sequence.

1. **Admissibility of canonical $3$-SAT.**  
   Give exact encodings for formulas, assignments, and Boolean outputs.

2. **Internal verifier.**  
   Define the clause-satisfaction verifier and certify its runtime bound.

3. **Direct frontend E13 package.**  
   Prove the six current tactic-level certificates

   $$
   K_{\mathrm{LS}_\sigma}^-,
   \quad
   K_{\mathrm{E6}}^-,
   \quad
   K_{\mathrm{E4}}^-,
   \quad
   K_{\mathrm{E11}}^-,
   \quad
   K_{\mathrm{SC}_\lambda}^{\mathrm{super}},
   \quad
   K_{\mathrm{E8}}^-,
   $$
   assemble them into

   $$
   K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}}),
   $$
   and derive

   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}
   $$
   by the direct E13 route.

4. **Internal Cook--Levin.**  
    Give the tableau construction, clause gadgets, witness consistency constraints, and the polynomial bound on formula
    size.

5. **$NP_{\mathrm{FM}}$-completeness and internal separation.**  
    Conclude

    $$
    \Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}},
    \qquad
    \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}},
    \qquad
    P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}.
    $$

6. **Export.**  
    Combine with Cluster I to derive

    $$
    P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
    $$

Cluster VI is the direct theorem route. The stronger thin-contract/factory refinement and any backend realizations are
handled in Cluster VII.
:::

:::{prf:remark} Detailed implementation protocol for Cluster VII
:label: rem-implementation-protocol-cluster-vii

Cluster VII packages the reusable thin-interface architecture for algorithm analysis.

1. **Define the five modal thin contracts.**  
   Each contract should expose only soft observables and transport data:
   - translator-stable state presentation,
   - a modality-appropriate cost observable,
   - a proof that the observable lower bound dominates every polynomial.

2. **Prove the semantic-first compilation theorem.**  
   For each modality, show that a verified thin contract compiles to the semantic obstruction certificate

   $$
   B_\lozenge\in K_\lozenge^-(\Pi).
   $$

3. **Assemble the compiled certificates.**  
   Prove that the five compiled semantic certificates assemble into the reconstructed E13 package and hence hardness.

4. **Add optional route coherence.**  
   When legacy tactic realizers are also supplied, prove that the direct frontend route and the thin-contract/factory
   route yield the same hardness conclusion.

5. **Record optional stronger realizations.**  
   Barrier data and backend dossiers should be documented as admissible realizations of the thin contracts, not as the
   default user burden.
:::

:::{prf:definition} Direct separation certificate
:label: def-direct-separation-certificate

A **direct separation certificate** for the present manuscript is the tuple

$$
\mathcal C_{\mathrm{direct}}
=
\bigl(
\mathcal P_{\mathrm{class}},
\mathcal P_{3\text{-SAT}}^{\mathrm{front}},
\mathcal P_{\mathrm{CL}},
\mathcal P_{\mathrm{bridge}}
\bigr),
$$
where:

1. $\mathcal P_{\mathrm{class}}$ is the direct classification/hardness package consisting of
   {prf:ref}`mt-alg-complete`, {prf:ref}`prop-six-certificates-cover-five-channels`, {prf:ref}`def-e13`, and
   {prf:ref}`thm-e13-contrapositive-hardness`;
2. $\mathcal P_{3\text{-SAT}}^{\mathrm{front}}$ is the canonical frontend package consisting of
   {prf:ref}`thm-canonical-3sat-admissible`,
   {prf:ref}`lem-random-3sat-metric-blockage`,
   {prf:ref}`lem-random-3sat-causal-blockage`,
   {prf:ref}`lem-random-3sat-integrality-blockage`,
   {prf:ref}`lem-random-3sat-galois-blockage`,
   {prf:ref}`lem-random-3sat-scaling-blockage`,
   {prf:ref}`lem-random-3sat-boundary-blockage`,
   and {prf:ref}`ex-3sat-all-blocked`;
3. $\mathcal P_{\mathrm{CL}}$ is the internal Cook--Levin package consisting of
   {prf:ref}`thm-internal-cook-levin-reduction`,
   and {prf:ref}`thm-sat-membership-hardness-transfer`;
4. $\mathcal P_{\mathrm{bridge}}$ is the bridge/export package consisting of
   {prf:ref}`cor-bridge-equivalence-rigorous`
   and {prf:ref}`cor-internal-to-classical-separation`.

This certificate records the **minimal theorem route actually used** for the main separation claim. It does not require
the stronger thin-contract/factory refinement of Part X or the optional backend dossiers of Part VIII.
:::

:::{prf:theorem} Sufficiency of the Direct Separation Certificate
:label: thm-sufficiency-direct-separation-certificate

If a direct separation certificate

$$
\mathcal C_{\mathrm{direct}}
$$
exists, then the direct theorem chain of Parts I and VI is fully discharged. In particular:

1. the canonical satisfiability family satisfies the current tactic-level E13 package:

   $$
   K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}});
   $$
2. the canonical satisfiability family lies outside $P_{\mathrm{FM}}$:

   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}};
   $$
3. the internal classes separate:

   $$
   P_{\mathrm{FM}}\neq NP_{\mathrm{FM}};
   $$
4. and, with the bridge package, the classical classes separate:

   $$
   P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
   $$
:::

:::{prf:proof}
The canonical frontend package $\mathcal P_{3\text{-SAT}}^{\mathrm{front}}$ yields

$$
K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})
$$
by {prf:ref}`ex-3sat-all-blocked`. Applying the direct hardness theorem in
$\mathcal P_{\mathrm{class}}$ gives

$$
\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
$$
The Cook--Levin package $\mathcal P_{\mathrm{CL}}$ gives

$$
\Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}}
$$
by {prf:ref}`thm-sat-membership-hardness-transfer`, and the same theorem then combines that membership/completeness
statement with

$$
\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}
$$
to yield

$$
P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}.
$$
Finally, the bridge package $\mathcal P_{\mathrm{bridge}}$ exports the internal separation to

$$
P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
$$
:::

:::{prf:remark} Direct Route Versus Stronger Audit Completion
:label: rem-direct-route-versus-stronger-audit-completion

The direct separation certificate of {prf:ref}`def-direct-separation-certificate` is the theorem package needed for the
main separation route used in the manuscript.

The direct route (Part VI) is now **fully self-contained**: all six blockage lemmas are expanded into discharged
theorem packages using the frontend obstruction lemmas with detailed structural arguments (with the Part IX barrier
metatheorems providing supporting quantitative infrastructure), the strengthened algebraic blockage theorem
{prf:ref}`thm-random-3sat-algebraic-blockage-strengthened` is unconditional with all 11 items of
{prf:ref}`def-completion-criteria-flat-dossier-3sat` discharged on-page in Section VI.C.3, and every proof satisfies
the 7-item acceptance criteria of {prf:ref}`prop-acceptance-criteria-implementation-package`.

The minimal completion certificate introduced below is **strictly stronger**. It adds the full proof-obligation ledger,
the compiled thin-contract package for canonical $3$-SAT, and the optional legacy/frontend bridge artifacts. Those
artifacts refine the direct route into a stronger referee-auditable semantic implementation, but they are not an
additional logical prerequisite for the direct Part VI theorem chain.
:::

:::{prf:definition} Minimal completion certificate for the full program
:label: def-minimal-completion-certificate

A **minimal completion certificate** for the stronger audit refinement of the separation program is the tuple

$$
\mathcal C_{\mathrm{master}}
=
(\mathcal T_{\mathrm{prim}},\ \mathfrak L,\ \mathcal C_{3\text{-SAT}}^{\mathrm{thin}},\ \{\mathcal F_\lozenge(\Pi_{3\text{-SAT}})\}_{\lozenge})
$$
such that:

1. $\mathcal T_{\mathrm{prim}}$ is a complete primitive audit table;
2. $\mathfrak L$ is a complete and acyclic proof obligation ledger;
3. $\mathcal C_{3\text{-SAT}}^{\mathrm{thin}}$ is a complete canonical $3$-SAT thin-contract package in the sense of
   Part X;
4. each claimed frontend tactic realization has a complete bridge dossier

   $$
   \mathcal F_\lozenge(\Pi_{3\text{-SAT}}).
   $$

The stronger audit refinement of the proof program is considered **implemented** exactly when a minimal completion
certificate has been produced.
:::

:::{prf:corollary} Completion Criterion for the Master Export
:label: cor-completion-criterion-master-export

If a minimal completion certificate

$$
\mathcal C_{\mathrm{master}}
$$
exists, then the master export theorem holds:

$$
P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
$$

Conversely, any failure of the program to produce the claimed separation must localize to failure of at least one
component of $\mathcal C_{\mathrm{master}}$.
:::

:::{prf:proof}
The forward implication is immediate from
{prf:ref}`thm-sufficiency-of-complete-ledger`.
For the converse, if the claimed separation were not established, then some theorem in Parts I--VI would remain
undischarged, so at least one component of the completion certificate would be incomplete.
:::

:::{prf:remark} Practical writing order
:label: rem-practical-writing-order

The mathematically safest writing order is not the narrative order of the paper, but the following implementation
order:

1. Cluster I (semantics and bridge),
2. Cluster II (normal forms),
3. Cluster III (pure witness validators),
4. primitive audit table $\mathcal T_{\mathrm{prim}}$,
5. Cluster IV (decomposition and exhaustiveness),
6. Cluster V (obstruction calculi),
7. Cluster VI (canonical $3$-SAT instantiation),
8. Cluster VII (thin-contract/factory refinement),
9. final master export summary.

This order minimizes circularity and exposes missing ingredients early.
:::

:::{prf:remark} What a hostile referee is allowed to demand after Part VII
:label: rem-what-hostile-referee-may-demand

After inserting Part VII, a hostile referee is entitled to ask for exactly the following things:

1. for the **direct theorem route**:
   - the completed primitive audit table,
   - the packaged direct frontend E13 certificate appendix for canonical $3$-SAT,
   - the completeness proof for `CostCert`,
   - and the explicit internal Cook--Levin reduction;
2. for the **stronger audit refinement**:
   - the complete canonical $3$-SAT thin-contract package,
   - the explicit admissible algebraic-signature library for the strengthened $\flat$-class,
   - the explicit admissible interface-contraction library for the strengthened $\partial$-class,
   - the complete obstruction-calculus rules and their soundness/completeness proofs,
   - and, only if claimed as a stronger realization, the five complete backend dossiers for canonical $3$-SAT.

Those are the real proof obligations at the two levels of presentation. Once the direct-route items are present, the
main separation chain is theorem-complete; once the stronger audit items are also present, there is no remaining vague
appeal to “higher-topos exhaustiveness” that can substitute for mathematics.
:::


### VIII. Audit-Level Implementation of the Primitive Classification and the Canonical 3-SAT Thin-Contract Package

:::{prf:remark} Why this section is written as an audit chapter
:label: rem-why-part-viii-is-audit-level

The preceding Parts I--VII isolate the theorem ladder, the obstruction calculi, the canonical $3$-SAT instantiation,
and the proof-completion criterion. What a hostile referee will ask next is not for another slogan, but for the
**actual audit artifacts** supporting:

1. Lemma {prf:ref}`lem-primitive-step-classification`,
2. Theorem {prf:ref}`thm-witness-decomposition`,
3. Theorems {prf:ref}`thm-sharp-obstruction-sound-complete` through
   {prf:ref}`thm-boundary-obstruction-sound-complete`,
4. and the five canonical $3$-SAT blockage theorems of Part VI.

To keep the exposition mathematically honest, this section separates:
- **implemented artifacts**, which may be cited as completed proofs;
- **thin-contract packages**, which are the default reusable audit artifacts expected from the framework;
- **backend dossiers**, which remain optional stronger realizations of those thin contracts;
- and **sufficiency theorems**, which state precisely what follows once those artifacts are complete.

This avoids the unacceptable practice of presenting unresolved backend dossiers as already-established theorem proofs.
:::

## VIII.A. Primitive Audit Appendix

:::{prf:definition} Audited semantic primitive signature
:label: def-audited-semantic-primitive-signature

Let

$$
\mathsf{Prim}_{\mathrm{sem}}
=
\mathsf{Prim}_{\mathrm{adm}}
\;\sqcup\;
\mathsf{Prim}_{\mathrm{prog}}
$$
denote the finite family of **semantic primitives** appearing at the leaf level of the normal-form language of
{prf:ref}`def-normal-form-language`.

The semantic primitive signature is obtained *after* administrative normalization. Accordingly:

1. $\mathsf{Prim}_{\mathrm{adm}}$ consists of administrative leaves, including:
   - presentation translators,
   - structural reindexings,
   - tag manipulations,
   - fixed-arity tuple/case operations,
   - bounded-size encoding/decoding operations,
   - and constant-size control dispatchers;

2. $\mathsf{Prim}_{\mathrm{prog}}$ consists of progress-producing leaves, each of which performs one semantically
   nontrivial local computational step.

The signature is **audited** only after every element of $\mathsf{Prim}_{\mathrm{sem}}$ is assigned a row in the
primitive audit table of {prf:ref}`def-primitive-audit-row`.
:::

:::{prf:definition} Primitive audit row
:label: def-primitive-audit-row

A **primitive audit row** for a semantic primitive

$$
\pi:\mathfrak U\Rightarrow_{\tau}\mathfrak V
$$
is a tuple

$$
\mathrm{Row}(\pi)
=
(\mathrm{type},\mathrm{status},\mathrm{mode},\mathrm{artifact},\mathrm{bound},\mathrm{refs}),
$$
where:

1. **type** records the source family $\mathfrak U$, target family $\mathfrak V$, and size translator $\tau$;
2. **status** is one of

   $$
   \mathrm{Administrative}
   \qquad\text{or}\qquad
   \mathrm{ProgressProducing};
   $$
3. **mode**, if $\pi$ is progress-producing, is a nonempty subset

   $$
   \mathrm{mode}(\pi)\subseteq \{\sharp,\int,\flat,\ast,\partial\}
   $$
   consisting of the modalities through which $\pi$ is certified to factor;
4. **artifact** is:
   - a presentation-translator proof if $\pi$ is administrative, or
   - a pure modal witness package if $\pi$ is progress-producing;
5. **bound** is a named polynomial bounding the local runtime and all intermediate representation sizes;
6. **refs** is the precise list of lemmas/propositions validating the row.

A primitive audit row is **valid** if all of the following hold:

- the type data are correct;
- the status assignment is justified;
- the artifact is complete and type-correct;
- the bound is explicit and proved;
- the refs point only to prior obligations in the acyclic ledger.
:::

:::{prf:definition} Complete primitive audit table
:label: def-complete-primitive-audit-table

A **complete primitive audit table**

$$
\mathcal T_{\mathrm{prim}}
$$
is the finite family of audit rows

$$
\{\mathrm{Row}(\pi)\}_{\pi\in\mathsf{Prim}_{\mathrm{sem}}}
$$
such that:

1. every semantic primitive appears exactly once;
2. every administrative primitive has a valid presentation-translator proof;
3. every progress-producing primitive has at least one valid pure modal witness package;
4. all local runtime and size bounds are polynomial;
5. the table is closed under admissible re-encodings of source and target families.

The table is **referee-complete** if, in addition, each row contains a unique stable label for the artifact proving it.
:::

:::{prf:remark} Mandatory column schema for the appendix table
:label: rem-mandatory-column-schema-appendix

The appendix table printed in the manuscript should contain at least the following columns:

| Primitive ID | Typed signature | Administrative / Progress | Certified modality set | Artifact label | Polynomial bound | Translator-invariant? |
|--------------|-----------------|---------------------------|------------------------|----------------|------------------|-----------------------|

A row with any missing column is not audit-complete.
:::

:::{prf:proposition} Local primitive audit suffices for Lemma 19
:label: prop-local-primitive-audit-suffices

Assume a referee-complete primitive audit table

$$
\mathcal T_{\mathrm{prim}}
$$
exists.

Then Lemma {prf:ref}`lem-primitive-step-classification` follows by finite case analysis.

Equivalently: once the primitive audit table is complete, the primitive step classification theorem carries no
remaining global ambiguity.
:::

:::{prf:proof}
The set $\mathsf{Prim}_{\mathrm{sem}}$ is finite by definition. For each semantic primitive leaf $\pi$:

- if its row marks it as administrative, the row includes a presentation-translator proof;
- if its row marks it as progress-producing, the row includes a pure modal witness package for at least one modality.

Therefore every primitive progress-producing leaf satisfies at least one of the universal properties of
{prf:ref}`thm-sharp-universality`--{prf:ref}`thm-boundary-universality`, exactly as required by
Lemma {prf:ref}`lem-primitive-step-classification`.
:::

:::{prf:definition} Required semantic primitive families
:label: def-required-semantic-primitive-families

For referee purposes, the normal-form language must be shown to factor through the following **semantic primitive
families** at minimum.

1. **Administrative family** $\mathsf{PT}$  
   Presentation translators, admissible encoders/decoders, structural reindexings, tagged sum/product shims.

2. **Metric local-step family** $\mathsf{SH}$  
   Local updates whose correctness is certified by a polynomially bounded ranking witness and which therefore carry a
   pure $\sharp$-witness.

3. **Causal elimination family** $\mathsf{IN}$  
   Local predecessor-only update steps over a polynomial-height well-founded dependency object and which therefore carry
   a pure $\int$-witness.

4. **Algebraic elimination family** $\mathsf{FLAT}$  
   Polynomial-size algebraic elimination/cancellation steps over finitely presented algebraic objects and which
   therefore carry a pure $\flat$-witness.

5. **Recursive split/merge family** $\mathsf{STAR}$  
   Split, recurse, and merge steps with strict size decrease and polynomial total recursion cost and which therefore
   carry a pure $\ast$-witness.

6. **Boundary contraction family** $\mathsf{PARTIAL}$  
   Interface extraction and interface contraction steps with polynomially bounded interface size and which therefore
   carry a pure $\partial$-witness.

The appendix must exhibit how the actual runtime leaf language reduces to these six audited families.
:::

:::{prf:remark} Referee test for the primitive audit appendix
:label: rem-referee-test-primitive-appendix

A hostile referee should be able to perform the following check mechanically:

1. choose any normal-form program;
2. inspect every leaf of its syntax tree;
3. locate the corresponding primitive audit row;
4. verify whether the leaf is administrative or progress-producing;
5. and, if progress-producing, read off a certified modality witness from the row.

If this cannot be done without guesswork, then the appendix is not yet complete.
:::

## VIII.B. Canonical 3-SAT Thin-Contract Package

:::{prf:definition} Complete thin contract for a modality
:label: def-complete-thin-contract-modality

Fix a problem family $\Pi$ and a modality

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}.
$$

A **complete thin contract**

$$
\mathcal C_\lozenge^{\mathrm{thin}}(\Pi)
$$
is a finite proof package whose user-facing data consist only of:

1. a translator-stable state presentation for the relevant hard subfamily;
2. a modality-appropriate soft observable measuring the least cost of crossing that presentation;
3. a proof that the observable lower bound eventually dominates every polynomial;
4. enough typed reconstruction data to feed the algorithmic factory of Part X.

The contract is **complete** if the displayed soft observables, transport data, and eventual-growth clauses are all
proved explicitly, so that the corresponding Part X factory theorem may be applied without adding new problem-specific
choices.

Concretely:
1. for $\lozenge=\sharp$ or $\lozenge=\int$, completeness includes a translator-stable barrier datum, a local drift
   bound, and an explicit lower-bound witness of the corresponding barrier-height quotient;
2. for $\lozenge=\flat,\ast,\partial$, completeness includes a translator-stable barrier datum and an explicit
   lower-bound witness for the corresponding modal barrier complexity.
:::

:::{prf:definition} Canonical 3-SAT thin-contract package
:label: def-canonical-3sat-thin-contract-package

The **canonical $3$-SAT thin-contract package** is the five-tuple

$$
\mathcal C_{3\text{-SAT}}^{\mathrm{thin}}
=
\bigl(
\mathcal C_\sharp^{\mathrm{thin}}(\Pi_{3\text{-SAT}}),
\mathcal C_\int^{\mathrm{thin}}(\Pi_{3\text{-SAT}}),
\mathcal C_\flat^{\mathrm{thin}}(\Pi_{3\text{-SAT}}),
\mathcal C_\ast^{\mathrm{thin}}(\Pi_{3\text{-SAT}}),
\mathcal C_\partial^{\mathrm{thin}}(\Pi_{3\text{-SAT}})
\bigr).
$$

The package is **complete** if each constituent thin contract is complete in the sense of
{prf:ref}`def-complete-thin-contract-modality`.
:::

:::{prf:remark} Why the thin-contract package is the default audit artifact
:label: rem-why-thin-contract-package-is-default

The thin-contract package is the default audit artifact because it matches the general hypostructure pattern already
used elsewhere in the manuscript: the user supplies only soft interfaces and transport witnesses, while the framework
compiles those into stronger semantic certificates by metatheorem.

Backend dossiers remain admissible, but they are treated below as optional stronger realizations of the thin contracts,
not as the default expected burden for every new problem family.
:::

:::{prf:theorem} Sufficiency of the canonical 3-SAT thin-contract package
:label: thm-sufficiency-canonical-3sat-thin-contract-package

Assume the canonical $3$-SAT thin-contract package

$$
\mathcal C_{3\text{-SAT}}^{\mathrm{thin}}
$$
is complete.

Then the algorithmic thin-interface factory of Part X compiles it into the five semantic modal obstruction
certificates

$$
B_\sharp\in K_\sharp^-(\Pi_{3\text{-SAT}}),\quad
B_\int\in K_\int^-(\Pi_{3\text{-SAT}}),\quad
B_\flat\in K_\flat^-(\Pi_{3\text{-SAT}}),\quad
B_\ast\in K_\ast^-(\Pi_{3\text{-SAT}}),\quad
B_\partial\in K_\partial^-(\Pi_{3\text{-SAT}}).
$$

Equivalently: the semantic-first compilation route later packaged by Part X already applies to this complete
thin-contract appendix.

Consequently:
1. $\Pi_{3\text{-SAT}}$ carries a full reconstructed E13 obstruction package;
2. $\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}$;
3. if the Internal Cook--Levin Reduction and bridge equivalence are also complete, then

   $$
   P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}
   \qquad\text{and}\qquad
   P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
   $$
:::

:::{prf:proof}
Unpack the five complete thin contracts inside
{prf:ref}`def-canonical-3sat-thin-contract-package`.

- The $\sharp$- and $\int$-contracts compile by the sharp and causal barrier certificate corollaries of Part IX.
- The $\flat$-, $\ast$-, and $\partial$-contracts compile by the corresponding Part IX barrier certificate
  corollaries.

This yields the five semantic modal obstruction certificates. Those certificates form the reconstructed E13 package,
which yields hardness by the Part V obstruction layer. The remaining claims then follow from the canonical-instantiation
theorems of Part VI and the bridge equivalence from Part I.
:::

## VIII.C. Optional Stronger Backend Dossiers

:::{prf:definition} Complete backend dossier for a modality
:label: def-complete-backend-dossier-modality

Fix a problem family $\Pi$ and a modality

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}.
$$

A **complete backend dossier**

$$
\mathcal D_\lozenge(\Pi)
$$
is a tuple

$$
\mathcal D_\lozenge(\Pi)
=
(\mathcal W_\lozenge,\ \mathcal I_\lozenge,\ \mathcal L_\lozenge,\ \mathcal E_\lozenge,\ B_\lozenge),
$$
consisting of:

1. **witness target** $\mathcal W_\lozenge$:  
   the exact class of pure $\lozenge$-witnesses to be excluded, stated using the strengthened universal-property theorem
   of Part III;

2. **invariant family** $\mathcal I_\lozenge$:  
   the modality-specific invariant or obstruction quantity assigned to instances of $\Pi$;

3. **backend lemma chain** $\mathcal L_\lozenge$:  
   a finite list of lemmas proving that every admissible pure $\lozenge$-witness is incompatible with
   $\mathcal I_\lozenge$;

4. **certificate extractor** $\mathcal E_\lozenge$:  
   an explicit derivation procedure in the obstruction calculus $\mathsf{Obs}_\lozenge$ producing a finitary
   certificate from the backend lemmas;

5. **obstruction certificate** $B_\lozenge$:  
   the resulting certificate in

   $$
   K_\lozenge^-(\Pi).
   $$

The dossier is **complete** if:
- every backend lemma is proved,
- the extractor is explicit,
- and the final certificate validates against the soundness theorem of the corresponding obstruction calculus.
:::

:::{prf:definition} Canonical 3-SAT backend dossier package
:label: def-canonical-3sat-backend-dossier-package

The **canonical $3$-SAT backend dossier package** is the five-tuple

$$
\mathcal D_{3\text{-SAT}}
=
\bigl(
\mathcal D_\sharp(\Pi_{3\text{-SAT}}),
\mathcal D_\int(\Pi_{3\text{-SAT}}),
\mathcal D_\flat(\Pi_{3\text{-SAT}}),
\mathcal D_\ast(\Pi_{3\text{-SAT}}),
\mathcal D_\partial(\Pi_{3\text{-SAT}})
\bigr).
$$

The package is **complete** if each of the five constituent dossiers is complete in the sense of
{prf:ref}`def-complete-backend-dossier-modality`.
:::

### VIII.B.1. The $\sharp$-dossier for canonical 3-SAT

:::{prf:definition} Completion criteria for the $\sharp$-backend dossier
:label: def-completion-criteria-sharp-dossier-3sat

A backend dossier

$$
\mathcal D_\sharp(\Pi_{3\text{-SAT}})
$$
is complete only if it contains proofs of the following statements, or strictly stronger substitutes.

1. **Target witness class specification.**  
   A full statement of the pure $\sharp$-witness class from
   {prf:ref}`def-pure-sharp-witness-rigorous`, specialized to canonical $3$-SAT.

2. **Plateau-core family theorem.**  
   A family

   $$
   \mathcal G_n \subseteq F_n
   $$
   of canonical $3$-SAT instances together with a family of non-solved lifted states

   $$
   P_n \subseteq Z^\sharp_{\rho_\sharp(n)}
   $$
   such that any admissible encoding of $\mathcal G_n$ into a pure $\sharp$ state space contains a plateau core on
   which local descent information fails to distinguish polynomially many steps of progress.

3. **Translator stability lemma.**  
   The plateau-core obstruction is preserved under presentation translators and admissible re-encodings.

4. **Rank-explosion lemma.**  
   Any ranking/Lyapunov witness

   $$
   V_n:Z_n^\sharp\to\mathbb N
   $$
   that is correct on the plateau core and strictly decreases off solved states must attain superpolynomially many
   distinct values on some size-$n$ subfamily.

5. **No polynomial sharp witness theorem.**  
   Therefore no pure $\sharp$-witness exists for canonical $3$-SAT.

6. **Certificate extraction.**  
   An explicit derivation in $\mathsf{Obs}_\sharp$ yielding

   $$
   B_\sharp \in K_\sharp^-(\Pi_{3\text{-SAT}}).
   $$

Without items (2)--(4), a dossier does not count as complete merely by citing “glassy landscape,” “OGP,” or “no
spectral gap” heuristics.
:::

:::{prf:proposition} Completion of the $\sharp$-dossier implies the metric blockage theorem
:label: prop-sharp-dossier-implies-metric-blockage

If the dossier

$$
\mathcal D_\sharp(\Pi_{3\text{-SAT}})
$$
is complete, then the strengthened semantic $\sharp$-obstruction holds:

$$
B_\sharp\in K_\sharp^-(\Pi_{3\text{-SAT}})
\qquad\text{and hence}\qquad
\mathbb K_\sharp^-(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
Immediate from the certificate extraction item of
{prf:ref}`def-completion-criteria-sharp-dossier-3sat` together with the soundness theorem
{prf:ref}`thm-sharp-obstruction-sound-complete`.
:::

### VIII.B.2. The $\int$-dossier for canonical 3-SAT

:::{prf:definition} Completion criteria for the $\int$-backend dossier
:label: def-completion-criteria-int-dossier-3sat

A backend dossier

$$
\mathcal D_\int(\Pi_{3\text{-SAT}})
$$
is complete only if it contains proofs of the following statements, or strictly stronger substitutes.

1. **Target witness class specification.**  
   A full statement of the pure $\int$-witness class from
   {prf:ref}`def-pure-int-witness-rigorous`, specialized to canonical $3$-SAT.

2. **Frustration-core theorem.**  
   A family

   $$
   \mathcal C_n \subseteq F_n
   $$
   of canonical $3$-SAT instances whose clause-variable dependency structure contains a strongly connected frustration
   core that cannot be unfolded into a polynomial-height well-founded elimination object while preserving correctness.

3. **Translator stability lemma.**  
   The frustration-core obstruction persists under presentation translators and admissible re-encodings.

4. **No predecessor-only elimination lemma.**  
   For every admissible candidate dependency object

   $$
   (P_n,\prec_n),
   $$
   either:
   - some update depends on a non-predecessor coordinate, or
   - the induced elimination order has superpolynomial height, or
   - the resulting elimination map is not correct on $\mathcal C_n$.

5. **No polynomial $\int$ witness theorem.**  
   Therefore no pure $\int$-witness exists for canonical $3$-SAT.

6. **Certificate extraction.**  
   An explicit derivation in $\mathsf{Obs}_\int$ yielding

   $$
   B_\int \in K_\int^-(\Pi_{3\text{-SAT}}).
   $$

Mere citation of “cycles,” “loops,” or “no DAG” without the translator-stability and height arguments is insufficient.
:::

:::{prf:proposition} Completion of the $\int$-dossier implies the causal blockage theorem
:label: prop-int-dossier-implies-causal-blockage

If the dossier

$$
\mathcal D_\int(\Pi_{3\text{-SAT}})
$$
is complete, then the strengthened semantic $\int$-obstruction holds:

$$
B_\int\in K_\int^-(\Pi_{3\text{-SAT}})
\qquad\text{and hence}\qquad
\mathbb K_\int^-(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
Immediate from the certificate extraction item of
{prf:ref}`def-completion-criteria-int-dossier-3sat` together with the soundness theorem
{prf:ref}`thm-int-obstruction-sound-complete`.
:::

### VIII.B.3. The strengthened $\flat$-dossier for canonical 3-SAT

:::{prf:definition} Algebraic signature library for the strengthened $\flat$-class
:label: def-algebraic-signature-library-flat

Let

$$
\mathfrak S_\flat
=
\mathfrak S_{\mathrm{quot}}
\sqcup
\mathfrak S_{\mathrm{lin}}
\sqcup
\mathfrak S_{\mathrm{rank}}
\sqcup
\mathfrak S_{\mathrm{fourier}}
\sqcup
\mathfrak S_{\mathrm{polyid}}
\sqcup
\mathfrak S_{\mathrm{mono}}
$$
be the library of admissible algebraic signature families for the strengthened $\flat$-class, where:

1. $\mathfrak S_{\mathrm{quot}}$ covers quotient and congruence compression;
2. $\mathfrak S_{\mathrm{lin}}$ covers linear elimination over effectively presented rings/fields;
3. $\mathfrak S_{\mathrm{rank}}$ covers determinant/rank/minor-based elimination;
4. $\mathfrak S_{\mathrm{fourier}}$ covers character/Fourier transforms over effectively presented finite groups;
5. $\mathfrak S_{\mathrm{polyid}}$ covers polynomial-identity and cancellation schemes;
6. $\mathfrak S_{\mathrm{mono}}$ covers monodromy-based algebraic-geometry reductions. For Boolean systems (where $z_j^2 - z_j = 0$ forces the variety to be discrete), the monodromy of the covering over the clause-weight parameter space is trivial ($\mathrm{Mon} = \{\mathrm{id}\}$), so $\mathfrak S_{\mathrm{mono}}$ is **vacuous**: there is no monodromy structure to exploit.

The strengthened $\flat$-dossier is complete only if it quantifies over **all** signatures in

$$
\mathfrak S_\flat.
$$
:::

:::{prf:definition} Completion criteria for the strengthened $\flat$-backend dossier
:label: def-completion-criteria-flat-dossier-3sat

A backend dossier

$$
\mathcal D_\flat(\Pi_{3\text{-SAT}})
$$
is complete only if it contains proofs of the following statements, or strictly stronger substitutes.

1. **Target witness class specification.**  
   A full statement of the strengthened pure $\flat$-witness class from
   {prf:ref}`def-pure-flat-witness-rigorous`, specialized to canonical $3$-SAT.

2. **Signature coverage theorem.**  
   Every admissible polynomial-size algebraic sketch for canonical $3$-SAT reduces to a sketch over one of the
   signature families in {prf:ref}`def-algebraic-signature-library-flat`.

3. **No-sketch theorem for quotient/congruence compression.**  
   No admissible polynomial-size quotient or congruence compression over
   $\mathfrak S_{\mathrm{quot}}$ yields a correct solver family for canonical $3$-SAT.

4. **No-sketch theorem for linear elimination.**  
   No admissible polynomial-size linear elimination sketch over
   $\mathfrak S_{\mathrm{lin}}$ yields a correct solver family for canonical $3$-SAT.

5. **No-sketch theorem for rank/determinant elimination.**  
   No admissible polynomial-size determinant/rank/minor-based sketch over
   $\mathfrak S_{\mathrm{rank}}$ yields a correct solver family for canonical $3$-SAT.

6. **No-sketch theorem for Fourier/character methods.**  
   No admissible polynomial-size character-transform or Fourier-type sketch over
   $\mathfrak S_{\mathrm{fourier}}$ yields a correct solver family for canonical $3$-SAT.

7. **No-sketch theorem for polynomial-identity/cancellation methods.**  
   No admissible polynomial-size algebraic cancellation or polynomial-identity sketch over
   $\mathfrak S_{\mathrm{polyid}}$ yields a correct solver family for canonical $3$-SAT.

8. **No-sketch theorem for monodromy methods (Boolean-vacuity argument).**
   The sub-channel $\mathfrak S_{\mathrm{mono}}$ is vacuous for canonical $3$-SAT: the Boolean constraints $z_j^2 - z_j = 0$ make the algebraic variety discrete ($\{0,1\}^n$ only), so the monodromy of the discrete covering over the clause-weight parameter space is trivial ($\mathrm{Mon} = \{\mathrm{id}\}$). Any algebraic sketch that does not use monodromy structure falls into sub-channels 1--7 (already blocked). Therefore no admissible polynomial-size sketch over $\mathfrak S_{\mathrm{mono}}$ yields a correct solver family for canonical $3$-SAT.

9. **Translator stability lemma.**  
   Failure of the above sketch classes is preserved under presentation translators and admissible re-encodings.

10. **No polynomial $\flat$ witness theorem.**  
    Therefore no strengthened pure $\flat$-witness exists for canonical $3$-SAT.

11. **Certificate extraction.**  
    An explicit derivation in $\mathsf{Obs}_\flat$ yielding

    $$
    B_\flat \in K_\flat^-(\Pi_{3\text{-SAT}}).
    $$

The dossier is **not** complete if it proves only:
- trivial automorphism group,
- absence of visible symmetry,
- absence of obvious lattice compression,
- or vacuity of monodromy for Boolean fibers alone.

Those are at most frontend sublemmas inside items (3)--(8).
:::

:::{prf:proposition} Completion of the strengthened $\flat$-dossier implies the algebraic blockage theorem
:label: prop-flat-dossier-implies-algebraic-blockage

If the dossier

$$
\mathcal D_\flat(\Pi_{3\text{-SAT}})
$$
is complete, then the strengthened algebraic blockage theorem of Part VI holds:

$$
B_\flat\in K_\flat^-(\Pi_{3\text{-SAT}})
\qquad\text{and hence}\qquad
\mathbb K_\flat^-(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
Immediate from the certificate extraction item of
{prf:ref}`def-completion-criteria-flat-dossier-3sat` together with the soundness theorem
{prf:ref}`thm-flat-obstruction-sound-complete`.
:::

### VIII.B.4. The $\ast$-dossier for canonical 3-SAT

:::{prf:definition} Completion criteria for the $\ast$-backend dossier
:label: def-completion-criteria-star-dossier-3sat

A backend dossier

$$
\mathcal D_\ast(\Pi_{3\text{-SAT}})
$$
is complete only if it contains proofs of the following statements, or strictly stronger substitutes.

1. **Target witness class specification.**  
   A full statement of the pure $\ast$-witness class from
   {prf:ref}`def-pure-star-witness-rigorous`, specialized to canonical $3$-SAT.

2. **Separator obstruction theorem.**  
   A family

   $$
   \mathcal S_n \subseteq F_n
   $$
   of canonical $3$-SAT instances for which every admissible split operation creating recursive subinstances also
   creates an interface or boundary cost large enough to prevent subcritical recursion.

3. **Strict-decrease incompatibility lemma.**  
   Any admissible recursive self-reduction tree that is correct on $\mathcal S_n$ and strictly decreases instance size
   along recursive edges must have either:
   - superpolynomial total recursion-tree size, or
   - superpolynomial total merge cost, or
   - loss of correctness.

4. **Translator stability lemma.**  
   The separator obstruction persists under presentation translators and admissible re-encodings.

5. **No polynomial $\ast$ witness theorem.**  
   Therefore no pure $\ast$-witness exists for canonical $3$-SAT.

6. **Certificate extraction.**  
   An explicit derivation in $\mathsf{Obs}_\ast$ yielding

   $$
   B_\ast \in K_\ast^-(\Pi_{3\text{-SAT}}).
   $$

It is not enough merely to quote a heuristic Master-theorem recurrence; the dossier must exclude every admissible
split/merge presentation allowed by {prf:ref}`def-pure-star-witness-rigorous`.
:::

:::{prf:proposition} Completion of the $\ast$-dossier implies the scaling blockage theorem
:label: prop-star-dossier-implies-scaling-blockage

If the dossier

$$
\mathcal D_\ast(\Pi_{3\text{-SAT}})
$$
is complete, then the strengthened semantic $\ast$-obstruction holds:

$$
B_\ast\in K_\ast^-(\Pi_{3\text{-SAT}})
\qquad\text{and hence}\qquad
\mathbb K_\ast^-(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
Immediate from the certificate extraction item of
{prf:ref}`def-completion-criteria-star-dossier-3sat` together with the soundness theorem
{prf:ref}`thm-star-obstruction-sound-complete`.
:::

### VIII.B.5. The strengthened $\partial$-dossier for canonical 3-SAT

:::{prf:definition} Interface library for the strengthened $\partial$-class
:label: def-interface-library-partial

Let

$$
\mathfrak I_\partial
=
\mathfrak I_{\mathrm{pf}}
\sqcup
\mathfrak I_{\mathrm{tw}}
\sqcup
\mathfrak I_{\mathrm{tn}}
\sqcup
\mathfrak I_{\mathrm{hol}}
$$
be the library of admissible interface families for the strengthened $\partial$-class, where:

1. $\mathfrak I_{\mathrm{pf}}$ covers planar/Pfaffian interfaces;
2. $\mathfrak I_{\mathrm{tw}}$ covers bounded-treewidth and bounded-interface-width contractions;
3. $\mathfrak I_{\mathrm{tn}}$ covers tensor-network contractions with polynomial-width interfaces;
4. $\mathfrak I_{\mathrm{hol}}$ covers holographic, matchgate, and related boundary-cancellation schemes.

The strengthened $\partial$-dossier is complete only if it quantifies over **all** interface families in

$$
\mathfrak I_\partial.
$$
:::

:::{prf:definition} Completion criteria for the strengthened $\partial$-backend dossier
:label: def-completion-criteria-partial-dossier-3sat

A backend dossier

$$
\mathcal D_\partial(\Pi_{3\text{-SAT}})
$$
is complete only if it contains proofs of the following statements, or strictly stronger substitutes.

1. **Target witness class specification.**  
   A full statement of the strengthened pure $\partial$-witness class from
   {prf:ref}`def-pure-boundary-witness-rigorous`, specialized to canonical $3$-SAT.

2. **Interface coverage theorem.**  
   Every admissible polynomial-size boundary/interface representation for canonical $3$-SAT reduces to one of the
   interface families in {prf:ref}`def-interface-library-partial`.

3. **No-contraction theorem for planar/Pfaffian interfaces.**  
   No admissible boundary reduction in $\mathfrak I_{\mathrm{pf}}$ yields a correct solver family for canonical $3$-SAT.

4. **No-contraction theorem for bounded-width interfaces.**  
   No admissible bounded-treewidth or bounded-interface-width contraction in
   $\mathfrak I_{\mathrm{tw}}$ yields a correct solver family for canonical $3$-SAT.

5. **No-contraction theorem for tensor-network polynomial-width interfaces.**  
   No admissible tensor-network contraction in
   $\mathfrak I_{\mathrm{tn}}$ yields a correct solver family for canonical $3$-SAT.

6. **No-contraction theorem for holographic/matchgate interfaces.**  
   No admissible holographic or matchgate boundary reduction in
   $\mathfrak I_{\mathrm{hol}}$ yields a correct solver family for canonical $3$-SAT.

7. **Translator stability lemma.**  
   Failure of the above interface families is preserved under presentation translators and admissible re-encodings.

8. **No polynomial $\partial$ witness theorem.**  
   Therefore no strengthened pure $\partial$-witness exists for canonical $3$-SAT.

9. **Certificate extraction.**  
   An explicit derivation in $\mathsf{Obs}_\partial$ yielding

   $$
   B_\partial \in K_\partial^-(\Pi_{3\text{-SAT}}).
   $$

The dossier is **not** complete if it proves only:
- non-planarity,
- absence of a Pfaffian orientation,
- or unbounded treewidth.

Those are frontend sublemmas at most; they do not by themselves exclude every admissible interface contraction in the
strengthened sense.
:::

:::{prf:proposition} Completion of the strengthened $\partial$-dossier implies the boundary blockage theorem
:label: prop-partial-dossier-implies-boundary-blockage

If the dossier

$$
\mathcal D_\partial(\Pi_{3\text{-SAT}})
$$
is complete, then the strengthened semantic $\partial$-obstruction holds:

$$
B_\partial\in K_\partial^-(\Pi_{3\text{-SAT}})
\qquad\text{and hence}\qquad
\mathbb K_\partial^-(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
Immediate from the certificate extraction item of
{prf:ref}`def-completion-criteria-partial-dossier-3sat` together with the soundness theorem
{prf:ref}`thm-boundary-obstruction-sound-complete`.
:::

## VIII.D. Sufficiency Theorems for the Audit Artifacts

:::{prf:theorem} Sufficiency of the primitive audit appendix
:label: thm-sufficiency-primitive-audit-appendix-extended

Assume a referee-complete primitive audit table

$$
\mathcal T_{\mathrm{prim}}
$$
exists.

Then the following statements are formally discharged:

1. Lemma {prf:ref}`lem-primitive-step-classification`;
2. the leaf-replacement step inside Theorem {prf:ref}`thm-witness-decomposition`;
3. the primitive side of the “no hidden mechanism” theorem
   {prf:ref}`thm-irreducible-witness-classification`.

Thus no remaining ambiguity about primitive classification persists once the appendix table is complete.
:::

:::{prf:proof}
The first item is Proposition {prf:ref}`prop-local-primitive-audit-suffices`. The second and third follow because every
normal-form leaf is then either:
- administrative and hence presentation-trivial, or
- progress-producing and hence already assigned to at least one pure modal class.
:::

:::{prf:theorem} Sufficiency of the canonical 3-SAT backend dossier package
:label: thm-sufficiency-canonical-3sat-dossier-package

Assume the canonical $3$-SAT backend dossier package

$$
\mathcal D_{3\text{-SAT}}
=
\bigl(
\mathcal D_\sharp(\Pi_{3\text{-SAT}}),
\mathcal D_\int(\Pi_{3\text{-SAT}}),
\mathcal D_\flat(\Pi_{3\text{-SAT}}),
\mathcal D_\ast(\Pi_{3\text{-SAT}}),
\mathcal D_\partial(\Pi_{3\text{-SAT}})
\bigr)
$$
is complete, and suppose each constituent dossier is accompanied by a stronger backend realization of the
corresponding thin contract in the sense of {prf:ref}`def-stronger-backend-realization-thin-contract`.

Then the corresponding canonical $3$-SAT thin-contract package exists and is complete. In particular, the five
blockage theorems of Part VI are formally discharged:

$$
B_\sharp\in K_\sharp^-(\Pi_{3\text{-SAT}}),\quad
B_\int\in K_\int^-(\Pi_{3\text{-SAT}}),\quad
B_\flat\in K_\flat^-(\Pi_{3\text{-SAT}}),\quad
B_\ast\in K_\ast^-(\Pi_{3\text{-SAT}}),\quad
B_\partial\in K_\partial^-(\Pi_{3\text{-SAT}}).
$$

Consequently:
1. $\Pi_{3\text{-SAT}}$ carries a full reconstructed E13 obstruction package;
2. $\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}$;
3. if the Internal Cook--Levin Reduction and bridge equivalence are also complete, then

   $$
   P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}
   \qquad\text{and}\qquad
   P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
   $$
:::

:::{prf:proof}
Combine the five dossier-to-blockage propositions:
{prf:ref}`prop-sharp-dossier-implies-metric-blockage`,
{prf:ref}`prop-int-dossier-implies-causal-blockage`,
{prf:ref}`prop-flat-dossier-implies-algebraic-blockage`,
{prf:ref}`prop-star-dossier-implies-scaling-blockage`,
and {prf:ref}`prop-partial-dossier-implies-boundary-blockage`.

By the additional stronger-realization hypothesis, each completed backend dossier realizes the corresponding thin
contract. Hence the canonical thin-contract package of
{prf:ref}`def-canonical-3sat-thin-contract-package` is complete. Apply
{prf:ref}`thm-sufficiency-canonical-3sat-thin-contract-package` to obtain the compiled semantic package and the stated
consequences.
:::

:::{prf:remark} What may and may not be claimed after inserting Part VIII
:label: rem-what-may-be-claimed-after-part-viii

After inserting Part VIII, the manuscript may honestly claim the following:

1. the direct exclusion route for canonical $3$-SAT still runs through the current tactic-level E13 package and does
   **not** logically require the backend dossiers as an extra prerequisite;
2. the semantic primitive audit has been implemented at the family level, and the default stronger audit burden is now
   a complete canonical $3$-SAT thin-contract package;
3. the exact content of each optional stronger backend dossier realization is now explicit;
4. the formal sufficiency chain from completed audit artifacts to the separation result is precise.

However, the manuscript may **not** yet honestly claim that the strengthened $\flat$- or $\partial$-blockage theorems
are proved unless the corresponding no-sketch and no-contraction subtheorems have actually been written and checked.
Likewise, it may not honestly claim the metric and causal blockage theorems are complete unless the plateau-core and
frustration-core dossier burdens have been discharged in full.

This distinction is essential for referee trust.
:::


### X. Thin Modal Contracts and the Algorithmic Factory

:::{prf:remark} Role of Part X
:label: rem-role-of-part-x

Part IX provides reusable barrier metatheorems, but it still presents those metatheorems at the level of explicit
barrier data and barrier-complexity lower bounds. The next hypostructure step is the familiar one from the sieve:
replace bespoke backend burdens by **thin contracts** and a **factory theorem**.

The point of Part X is therefore not to add a sixth mechanism. It is to expose a public algorithmic interface of type

$$
T_{\text{algorithmic}}
$$
whose user-facing data consist only of soft observables and transport witnesses, while the framework compiles those
data into semantic modal obstructions, reconstructed E13, and hence hardness.

This is the default reusable route. Barrier data and backend dossiers remain admissible, but they now appear as
realizations of the thin contracts rather than as the primary public burden.
:::

## X.A. The Five Thin Modal Contracts

:::{prf:definition} Thin $\sharp$ contract
:label: def-thin-sharp-contract

Let $\Pi$ be a problem family. A **thin $\sharp$ contract** for $\Pi$ is a tuple

$$
\mathcal C_\sharp^{\mathrm{thin}}(\Pi)=
(\mathfrak B_\sharp,d_\sharp,g_\sharp)
$$
consisting of:

1. a translator-stable barrier datum

   $$
   \mathfrak B_\sharp
   $$
   for $\Pi$;
2. a sharp local drift bound

   $$
   d_\sharp
   $$
   in the sense of {prf:ref}`def-sharp-local-energy-drift-bound`;
3. a lower-bound witness

   $$
   g_\sharp:\mathbb N\to\mathbb N
   $$
   such that:

   $$
   g_\sharp(n)\le
   \left\lceil
   \frac{\Delta_{\mathfrak B_\sharp}(n)}{d_\sharp(n)}
   \right\rceil
   $$
   for all sufficiently large $n$, and

   $$
   g_\sharp
   $$
   eventually dominates every polynomial.

This contract is called **complete** if the displayed inequalities and growth statement are proved explicitly.
:::

:::{prf:definition} Thin $\int$ contract
:label: def-thin-int-contract

Let $\Pi$ be a problem family. A **thin $\int$ contract** for $\Pi$ is a tuple

$$
\mathcal C_\int^{\mathrm{thin}}(\Pi)=
(\mathfrak B_\int,d_\int,g_\int)
$$
consisting of:

1. a translator-stable barrier datum

   $$
   \mathfrak B_\int
   $$
   for $\Pi$;
2. an $\int$ local drift bound

   $$
   d_\int
   $$
   in the sense of {prf:ref}`def-int-local-energy-drift-bound`;
3. a lower-bound witness

   $$
   g_\int:\mathbb N\to\mathbb N
   $$
   such that:

   $$
   g_\int(n)\le
   \left\lceil
   \frac{\Delta_{\mathfrak B_\int}(n)}{d_\int(n)}
   \right\rceil
   $$
   for all sufficiently large $n$, and

   $$
   g_\int
   $$
   eventually dominates every polynomial.

This contract is called **complete** if the displayed inequalities and growth statement are proved explicitly.
:::

:::{prf:definition} Thin $\flat$ contract
:label: def-thin-flat-contract

Let $\Pi$ be a problem family. A **thin $\flat$ contract** for $\Pi$ is a tuple

$$
\mathcal C_\flat^{\mathrm{thin}}(\Pi)=
(\mathfrak B_\flat,g_\flat)
$$
consisting of:

1. a translator-stable barrier datum

   $$
   \mathfrak B_\flat
   $$
   for $\Pi$;
2. a lower-bound witness

   $$
   g_\flat:\mathbb N\to\mathbb N
   $$
   such that:

   $$
   g_\flat(n)\le \beta_\flat^{\mathfrak B_\flat}(n)
   $$
   for all sufficiently large $n$, and

   $$
   g_\flat
   $$
   eventually dominates every polynomial.

This contract is called **complete** if the displayed inequality and growth statement are proved explicitly.
:::

:::{prf:definition} Thin $\ast$ contract
:label: def-thin-star-contract

Let $\Pi$ be a problem family. A **thin $\ast$ contract** for $\Pi$ is a tuple

$$
\mathcal C_\ast^{\mathrm{thin}}(\Pi)=
(\mathfrak B_\ast,g_\ast)
$$
consisting of:

1. a translator-stable barrier datum

   $$
   \mathfrak B_\ast
   $$
   for $\Pi$;
2. a lower-bound witness

   $$
   g_\ast:\mathbb N\to\mathbb N
   $$
   such that:

   $$
   g_\ast(n)\le \beta_\ast^{\mathfrak B_\ast}(n)
   $$
   for all sufficiently large $n$, and

   $$
   g_\ast
   $$
   eventually dominates every polynomial.

This contract is called **complete** if the displayed inequality and growth statement are proved explicitly.
:::

:::{prf:definition} Thin $\partial$ contract
:label: def-thin-partial-contract

Let $\Pi$ be a problem family. A **thin $\partial$ contract** for $\Pi$ is a tuple

$$
\mathcal C_\partial^{\mathrm{thin}}(\Pi)=
(\mathfrak B_\partial,g_\partial)
$$
consisting of:

1. a translator-stable barrier datum

   $$
   \mathfrak B_\partial
   $$
   for $\Pi$;
2. a lower-bound witness

   $$
   g_\partial:\mathbb N\to\mathbb N
   $$
   such that:

   $$
   g_\partial(n)\le \beta_\partial^{\mathfrak B_\partial}(n)
   $$
   for all sufficiently large $n$, and

   $$
   g_\partial
   $$
   eventually dominates every polynomial.

This contract is called **complete** if the displayed inequality and growth statement are proved explicitly.
:::

## X.B. Public Algorithmic Thin Interface

:::{prf:definition} Algorithmic thin interface of type $T_{\text{algorithmic}}$
:label: def-algorithmic-thin-interface

An **algorithmic thin interface** for a problem family

$$
\Pi
$$
is a tuple

$$
\mathcal I_{\text{alg}}^{\mathrm{thin}}(\Pi)
=
\bigl(
K_{T_{\text{algorithmic}}}^+,\ 
\mathcal C_\sharp^{\mathrm{thin}}(\Pi),\
\mathcal C_\int^{\mathrm{thin}}(\Pi),\
\mathcal C_\flat^{\mathrm{thin}}(\Pi),\
\mathcal C_\ast^{\mathrm{thin}}(\Pi),\
\mathcal C_\partial^{\mathrm{thin}}(\Pi)
\bigr)
$$
such that each of the five modal thin contracts is complete.

This is the public user-facing interface for the reusable algorithmic factory. It exposes only:
1. the algorithmic type tag;
2. typed translator-stable soft observables;
3. polynomial-growth witnesses dominating every polynomial;
4. no full backend obstruction dossiers.
:::

:::{prf:definition} Stronger backend realization of a thin contract
:label: def-stronger-backend-realization-thin-contract

Let

$$
\mathcal C_\lozenge^{\mathrm{thin}}(\Pi)
$$
be a thin modal contract.

A **stronger backend realization** of that thin contract is any proof package that proves all hypotheses required for
the completeness of

$$
\mathcal C_\lozenge^{\mathrm{thin}}(\Pi)
$$
and may additionally include finer invariants, richer witness-exclusion lemmas, or explicit certificate extractors.

Barrier dossiers and other backend packages are therefore admissible only insofar as they realize the thin-contract
payload required by the factory theorem below.
:::

## X.C. The Algorithmic Factory

:::{prf:theorem} Thin-Contract Compilation by Modality
:label: thm-thin-contract-compilation-by-modality

Let $\Pi$ be a problem family.

1. If $\mathcal C_\sharp^{\mathrm{thin}}(\Pi)$ is complete, then

   $$
   B_\sharp\in K_\sharp^-(\Pi).
   $$
2. If $\mathcal C_\int^{\mathrm{thin}}(\Pi)$ is complete, then

   $$
   B_\int\in K_\int^-(\Pi).
   $$
3. If $\mathcal C_\flat^{\mathrm{thin}}(\Pi)$ is complete, then

   $$
   B_\flat\in K_\flat^-(\Pi).
   $$
4. If $\mathcal C_\ast^{\mathrm{thin}}(\Pi)$ is complete, then

   $$
   B_\ast\in K_\ast^-(\Pi).
   $$
5. If $\mathcal C_\partial^{\mathrm{thin}}(\Pi)$ is complete, then

   $$
   B_\partial\in K_\partial^-(\Pi).
   $$
:::

:::{prf:proof}
Clause (1) is {prf:ref}`cor-sharp-barrier-certificate` applied to the data contained in
{prf:ref}`def-thin-sharp-contract`.
Clause (2) is {prf:ref}`cor-int-barrier-certificate` applied to the data contained in
{prf:ref}`def-thin-int-contract`.
Clause (3) is {prf:ref}`cor-flat-barrier-certificate` applied to the data contained in
{prf:ref}`def-thin-flat-contract`.
Clause (4) is {prf:ref}`cor-star-barrier-certificate` applied to the data contained in
{prf:ref}`def-thin-star-contract`.
Clause (5) is {prf:ref}`cor-partial-barrier-certificate` applied to the data contained in
{prf:ref}`def-thin-partial-contract`.
:::

:::{prf:theorem} [FACT-Algorithmic] Thin Interface Compilation
:label: mt-fact-algorithmic-thin-interface

Let

$$
\mathcal I_{\text{alg}}^{\mathrm{thin}}(\Pi)
$$
be a complete algorithmic thin interface for a problem family $\Pi$.

Then the framework compiles that thin interface into:

1. the five semantic modal obstruction certificates

   $$
   B_\sharp\in K_\sharp^-(\Pi),\quad
   B_\int\in K_\int^-(\Pi),\quad
   B_\flat\in K_\flat^-(\Pi),\quad
   B_\ast\in K_\ast^-(\Pi),\quad
   B_\partial\in K_\partial^-(\Pi);
   $$
2. the full reconstructed E13 obstruction package for $\Pi$;
3. the hardness conclusion

   $$
   \Pi\notin P_{\mathrm{FM}}.
   $$

This compilation is semantic-first: the primary factory output is the semantic modal obstruction package, not the
legacy frontend certificate language.
:::

:::{prf:proof}
Apply {prf:ref}`thm-thin-contract-compilation-by-modality` to the five modal thin contracts inside
{prf:ref}`def-algorithmic-thin-interface`. This yields the five semantic modal obstruction certificates. By
{prf:ref}`def-e13-reconstructed`, those certificates form the full reconstructed E13 obstruction package. Then
{prf:ref}`cor-e13-contrapositive-hardness-reconstructed` yields

$$
\Pi\notin P_{\mathrm{FM}}.
$$
:::

:::{prf:corollary} Factory Route for Canonical 3-SAT
:label: cor-factory-route-canonical-3sat

If the canonical $3$-SAT thin-contract package

$$
\mathcal C_{3\text{-SAT}}^{\mathrm{thin}}
$$
is complete, then the complete algorithmic thin interface

$$
\mathcal I_{\text{alg}}^{\mathrm{thin}}(\Pi_{3\text{-SAT}})
$$
exists and the factory theorem
{prf:ref}`mt-fact-algorithmic-thin-interface`
implies

$$
\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
$$
:::

:::{prf:proof}
Combine {prf:ref}`def-canonical-3sat-thin-contract-package` with
{prf:ref}`def-algorithmic-thin-interface` and apply
{prf:ref}`mt-fact-algorithmic-thin-interface`.
:::

:::{prf:remark} Compatibility with the direct frontend route
:label: rem-thin-factory-compatibility-direct-route

The factory route of Part X does not replace the current direct Part VI route. It refines it.

- The **direct route** remains:

  $$
  \text{frontend certificates}
  \Longrightarrow
  K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})
  \Longrightarrow
  \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
  $$
- The **factory route** is:

  $$
  \text{thin contracts}
  \Longrightarrow
  (K_\sharp^-,K_\int^-,K_\flat^-,K_\ast^-,K_\partial^-)
  \Longrightarrow
  \mathbf B_{\mathrm{E13}}^{\mathrm{recon}}
  \Longrightarrow
  \Pi\notin P_{\mathrm{FM}}.
  $$

Whenever a problem family also carries legacy frontend realizers, the same hardness conclusion can be restated in the
current tactic-level language. But the factory theorem itself is semantic-first and does not depend on that legacy
presentation.
:::

:::{prf:remark} Backend dossiers as optional stronger realizations
:label: rem-backend-dossiers-as-optional-thin-realizations

Parts VIII and IX should therefore be read in the following order.

1. Thin contracts are the default public interface.
2. Barrier data are one admissible way to realize thin contracts.
3. Backend dossiers are optional stronger realizations of thin contracts, appropriate only when one wants a finer audit
   trail than the factory theorem itself requires.

This is exactly parallel to the sieve philosophy elsewhere in the manuscript: users provide thin interfaces, the
framework supplies the compiled backend consequences, and only specialist extensions need the stronger internal
artifacts.
:::

:::{prf:remark} Recommended placement of appendices after Part X
:label: rem-recommended-placement-appendices

For maximal referee readability, the following appendices should follow immediately after Part X.

1. **Appendix A:** The complete primitive audit table $\mathcal T_{\mathrm{prim}}$.
2. **Appendix B:** The direct frontend E13 certificate appendix for canonical $3$-SAT.
3. **Appendix C:** The canonical $3$-SAT thin-contract package.
4. **Appendix D:** The $\sharp$-backend dossier for canonical $3$-SAT.
5. **Appendix E:** The $\int$-backend dossier for canonical $3$-SAT.
6. **Appendix F:** The strengthened $\flat$-backend dossier for canonical $3$-SAT.
7. **Appendix G:** The $\ast$-backend dossier for canonical $3$-SAT.
8. **Appendix H:** The strengthened $\partial$-backend dossier for canonical $3$-SAT.
9. **Appendix I:** The frontend-to-backend bridge dossiers for the legacy tactic certificates.
10. **Appendix J:** The explicit Internal Cook--Levin reduction.

That appendix ordering mirrors the proof-dependency order and minimizes referee backtracking.
:::


## Appendix A. Primitive Audit Table

> The main document contains a self-contained version of this appendix sufficient for the P≠NP proof chain.
> The extended infrastructure below supports the broader thin-contract and backend-dossier framework.

:::{prf:definition} Current Semantic Primitive-Family Presentation
:label: def-current-semantic-primitive-family-presentation

For the present manuscript implementation, the semantic primitive signature of
{prf:ref}`def-audited-semantic-primitive-signature` is taken to be the six family-level generators of
{prf:ref}`def-required-semantic-primitive-families`:

$$
\mathsf{Prim}_{\mathrm{sem}}
=
\{\mathsf{PT},\mathsf{SH},\mathsf{IN},\mathsf{FLAT},\mathsf{STAR},\mathsf{PARTIAL}\}.
$$

Concretely:
1. $\mathsf{PT}$ is the administrative family of presentation translators, admissible encoders/decoders, structural
   reindexings, and tagged sum/product shims;
2. $\mathsf{SH}$ is the family of local steps certified by a pure $\sharp$-witness;
3. $\mathsf{IN}$ is the family of local predecessor-only elimination steps certified by a pure $\int$-witness;
4. $\mathsf{FLAT}$ is the family of algebraic elimination/cancellation steps certified by a pure $\flat$-witness;
5. $\mathsf{STAR}$ is the family of split/recurse/merge local steps certified by a pure $\ast$-witness;
6. $\mathsf{PARTIAL}$ is the family of interface extraction/contraction steps certified by a pure $\partial$-witness.

This presentation is the semantic leaf language obtained after the administrative normalization and family aggregation
used in Cluster II. Any finer evaluator-level instruction set must be shown to reduce to this semantic signature.
:::

:::{prf:theorem} Appendix A Primitive Audit Table
:label: thm-appendix-a-primitive-audit-table-extended

Under the current semantic primitive-family presentation of
{prf:ref}`def-current-semantic-primitive-family-presentation`, the following table is a referee-complete primitive
audit table in the sense of {prf:ref}`def-complete-primitive-audit-table`.

| Primitive ID | Typed signature | Administrative / Progress | Certified modality set | Artifact label | Polynomial bound | Translator-invariant? |
|--------------|-----------------|---------------------------|------------------------|----------------|------------------|-----------------------|
| $\mathsf{PT}$ | $\mathfrak U \Rightarrow_{\tau} \mathfrak V$ | Administrative | $\varnothing$ | {prf:ref}`def-presentation-translator` | Included in translator certificate | Yes |
| $\mathsf{SH}$ | $\mathfrak Z^\sharp \Rightarrow \mathfrak Z^\sharp$ | Progress | $\{\sharp\}$ | {prf:ref}`def-pure-sharp-witness-rigorous` | Polynomial ranking bound $q_\sharp$ and family cost bound | Yes |
| $\mathsf{IN}$ | $\mathfrak Z^\int \Rightarrow \mathfrak Z^\int$ | Progress | $\{\int\}$ | {prf:ref}`def-pure-int-witness-rigorous` | Polynomial size/height bound $q_\int$ and family cost bound | Yes |
| $\mathsf{FLAT}$ | $\mathfrak A^\flat \Rightarrow \mathfrak B^\flat$ | Progress | $\{\flat\}$ | {prf:ref}`def-pure-flat-witness-rigorous` | Polynomial presentation bound $q_\flat$ and family cost bound | Yes |
| $\mathsf{STAR}$ | $\mathfrak Z^\ast \Rightarrow \mathfrak Z^\ast$ (node-local recursion step) | Progress | $\{\ast\}$ | {prf:ref}`def-pure-star-witness-rigorous` | Polynomial total-tree bound $q_\ast$ and family cost bound | Yes |
| $\mathsf{PARTIAL}$ | $\mathfrak Z^\partial \Rightarrow \mathfrak B^\partial$ | Progress | $\{\partial\}$ | {prf:ref}`def-pure-boundary-witness-rigorous` | Polynomial interface bound $q_\partial$ and family cost bound | Yes |

In particular, every semantic primitive leaf in the present implementation appears exactly once and carries either:
1. a presentation-translator proof, or
2. a pure modal witness package with explicit polynomial bounds.
:::

:::{prf:proof}
We verify the six rows one by one.

1. **Administrative row $\mathsf{PT}$.**
   By {prf:ref}`def-presentation-translator`, every member of $\mathsf{PT}$ is a uniform family equipped with a
   polynomial-time partial inverse on its image. This is exactly the administrative case required by
   {prf:ref}`def-primitive-audit-row`. The translator certificate itself supplies the type data, the artifact, the
   explicit polynomial bound, and invariance under admissible re-encoding.

2. **Metric row $\mathsf{SH}$.**
   By {prf:ref}`def-required-semantic-primitive-families`, every member of $\mathsf{SH}$ is, by construction, a local
   step whose correctness is certified by a polynomially bounded ranking witness. Therefore it carries a pure
   $\sharp$-witness in the sense of {prf:ref}`def-pure-sharp-witness-rigorous`. The witness data provide:
   - typed lifted state families $Z_n^\sharp$,
   - the solved-state predicate,
   - the ranking function $V_n^\sharp$ with polynomial bound $q_\sharp$,
   - and the induced polynomial family cost control.

3. **Causal row $\mathsf{IN}$.**
   Again by {prf:ref}`def-required-semantic-primitive-families`, every member of $\mathsf{IN}$ is a predecessor-only
   elimination step over a polynomial-height dependency object. Hence it carries a pure $\int$-witness in the sense of
   {prf:ref}`def-pure-int-witness-rigorous`, with polynomial size and height bound $q_\int$.

4. **Algebraic row $\mathsf{FLAT}$.**
   Every member of $\mathsf{FLAT}$ is, by definition, a polynomial-size algebraic elimination/cancellation step over
   finitely presented algebraic objects. Therefore it carries a pure $\flat$-witness in the sense of
   {prf:ref}`def-pure-flat-witness-rigorous`, together with the presentation bound $q_\flat$ and the certified
   polynomial-time primitive basis used by the elimination stage.

5. **Recursive row $\mathsf{STAR}$.**
   Every member of $\mathsf{STAR}$ is, by definition, a split/recurse/merge local step with strict size decrease and
   polynomial total recursion cost. Hence it carries a pure $\ast$-witness in the sense of
   {prf:ref}`def-pure-star-witness-rigorous`, with polynomial bound $q_\ast$ on total recursion-tree size and local
   work.

6. **Boundary row $\mathsf{PARTIAL}$.**
   Every member of $\mathsf{PARTIAL}$ is, by definition, an interface extraction/contraction step with polynomially
   bounded interface size. Therefore it carries a pure $\partial$-witness in the sense of
   {prf:ref}`def-pure-boundary-witness-rigorous`, with polynomial interface bound $q_\partial$.

The semantic signature contains exactly the six displayed primitive families and no others by
{prf:ref}`def-current-semantic-primitive-family-presentation`. Thus every semantic primitive appears exactly once.
Each row contains the mandatory fields from {prf:ref}`rem-mandatory-column-schema-appendix`, and each row is stable
under admissible re-encoding because the corresponding witness notion is formulated using admissible families and
presentation translators. This proves referee-completeness of the table.
:::

:::{prf:corollary} Primitive Step Classification from Appendix A
:label: cor-primitive-step-classification-from-appendix-a

Under the current semantic primitive-family presentation of
{prf:ref}`def-current-semantic-primitive-family-presentation`, Lemma
{prf:ref}`lem-primitive-step-classification` is discharged.
:::

:::{prf:proof}
Apply Proposition {prf:ref}`prop-local-primitive-audit-suffices` to the primitive audit table of
{prf:ref}`thm-appendix-a-primitive-audit-table-extended`.
:::


## Appendix C. Canonical 3-SAT Thin-Contract Package

:::{prf:definition} Complete Thin-Contract Appendix for Canonical 3-SAT
:label: def-complete-thin-contract-appendix-3sat

A **complete thin-contract appendix** for canonical $3$-SAT is a finite table packaging:

1. the five modal thin contracts of Part X specialized to $\Pi_{3\text{-SAT}}$;
2. the soft observables and translator-stable state presentations used by each contract;
3. the supporting lower-bound witnesses dominating every polynomial;
4. the assembly step yielding the complete package

   $$
   \mathcal C_{3\text{-SAT}}^{\mathrm{thin}}.
   $$

This appendix is the default stronger audit artifact for the semantic route. It sits strictly between the direct
frontend appendix of Appendix B and the optional stronger backend dossiers of Part VIII.
:::

:::{prf:remark} Current status of Appendix C
:label: rem-current-status-appendix-c

Appendix C is the correct target packaging for the stronger audit route, but unlike Appendix B it is not yet filled by a
completed theorem table in the current manuscript. The reason is structural rather than philosophical: the direct Part
VI theorem chain is already complete, while the Part X route requires explicit thin-contract realizers for canonical
$3$-SAT.

Accordingly:
1. Appendix B packages the route already used by the main proof;
2. Appendix C is the default target for the stronger semantic refinement;
3. the backend dossiers of Part VIII remain optional stronger realizations of the same Appendix C package.
:::

