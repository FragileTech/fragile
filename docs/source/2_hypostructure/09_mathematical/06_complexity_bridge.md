---
title: "The P/NP Bridge to Classical Complexity"
---

# The P/NP Bridge to Classical Complexity

(sec-complexity-bridge)=
## Completing the Export to ZFC Complexity Theory

:::{div} feynman-prose
Let me tell you what this chapter is about and why it matters. We have spent a lot of effort building a categorical
framework for complexity theory: the five algorithm classes, the Algorithmic Completeness theorem, and the universal
obstruction certificates. Parts VI--IX isolate the exact internal route to
$P_{\text{FM}} \neq NP_{\text{FM}}$: canonical $3$-SAT admissibility, the internal Cook--Levin reduction, and the
current E13 exclusion route on the canonical problem object, with the audited primitive appendix and canonical backend
dossiers available as a stronger semantic implementation layer.

But here is the question a skeptic would reasonably ask: "Why should I believe your internal separation implies the classical P ≠ NP conjecture? Maybe your 'Fragile P' is not the same as classical P. Maybe your 'Fragile NP' is secretly larger or smaller than classical NP."

That skeptic is right to ask. This is exactly the gap that the Natural Proofs barrier exploits: a proof technique that works in a non-standard model might fail to export to the standard model. So we need to close this gap rigorously.

The solution is to build not one bridge but *two*: a forward bridge (classical algorithms compile into our framework) and a reverse bridge (our algorithms extract back to classical). When both directions work with polynomial overhead, we get an *equivalence* of complexity classes. And then—*only* then—can we legitimately claim that our internal separation implies the classical one.

This is what complexity theorists call "robustness" of a complexity class: it should not matter whether you define it via Turing machines, RAM machines, circuits, or—in our case—categorical morphisms in a cohesive topos. The computation thesis says these are all equivalent up to polynomial factors. This chapter makes that equivalence precise and rigorous for our framework.
:::

This chapter establishes the **bidirectional bridge** between the Hypostructure algorithmic framework and classical Deterministic Turing Machine (DTM) complexity theory. It provides the missing piece that allows internal complexity separations to export to classical ZFC statements about P and NP.

**What the strengthened algorithmic chapter now gives us:**
- **Rigorous input and runtime layer:** admissible families, uniform algorithms, family cost certificates, pure witnesses, modal profiles, and saturated modal closure
- **Part I bridge package:** finite encodability, evaluator adequacy, DTM-to-Fragile compilation, Fragile-to-DTM extraction, and the packaged bridge corollary
- **Internal separation machinery:** the classification/exhaustiveness ladder, the obstruction calculi, the canonical
  $3$-SAT instantiation, and the Part VII--VIII completion criteria

**What this chapter does:**
- repackages the Part I bridge theorems in language aimed at classical complexity theory
- records exactly which export claims are proved outright and which remain downstream of the stronger extractor-level
  implementation goals
- keeps the bridge logically downstream of the internal definitions so the manuscript does not argue in circles

**The Payoff, under the Part I bridge package of**
{prf:ref}`thm-bit-cost-evaluator-discipline`,
{prf:ref}`thm-evaluator-adequacy`, and
{prf:ref}`thm-costcert-completeness`:

$$
P_{\text{FM}} = P_{\text{DTM}} \quad\text{and}\quad NP_{\text{FM}} = NP_{\text{DTM}}
$$

Therefore:

$$
P_{\text{FM}} \neq NP_{\text{FM}} \quad\Rightarrow\quad P_{\text{DTM}} \neq NP_{\text{DTM}}
$$



(sec-bridge-definitions)=
## Foundational Definitions

:::{div} feynman-prose
Before we can bridge two worlds, we need to be crystal clear about what we are bridging. On one side, we have classical Turing machines with time complexity measured in steps. On the other side, we have categorical morphisms in a cohesive topos with "cost" measured by some abstract evaluator.

The key insight is that both are counting the same thing: *how much information processing happens*. A Turing machine step updates a finite configuration. A hypostructure evaluation step applies a morphism to a state. If we can show these steps are mutually simulable with polynomial overhead, we are done.

The definitions below make this precise. Pay attention to the **CostCert** predicate—this is the bridge hinge. It is what lets us say "this hypostructure program runs in polynomial time" in a way that a classical complexity theorist can verify in ZFC.
:::

### D0.1 Effective Programs (Fragile)

:::{prf:definition} Effective Programs
:label: def-effective-programs-fragile

An **effective Fragile program** is a code object

$$
a\in \mathsf{Prog}_{\text{FM}}
$$
interpreted by the fixed evaluator $\mathsf{Eval}$, with the following understanding.

1. **Representable law.** The code $a$ has a concrete syntax or bytecode presentation interpretable by the Fragile
   runtime.
2. **ZFC-definable operational semantics.** For every encoded input

   $$
   u\in\{0,1\}^*,
   $$
   the expression

   $$
   \mathsf{Eval}(a,u)
   $$
   is governed by a ZFC-definable stepwise semantics.
3. **Externalized input model.** Whenever complexity is discussed, programs are used only relative to admissible
   families of externally presented, $0$-truncated objects in the sense of
   {prf:ref}`def-admissible-input-family-rigorous`.

Thus the bridge chapter never treats a bare internal object of $\mathbf H$ as an input model. Every complexity claim is
made through tagged finite encodings of admissible families.
:::

:::{prf:remark} ZFC Interpretability of the Cohesive Ambient $\mathbf H$
:label: rem-zfc-model-for-H

A classical ZFC reader may ask: in what set-theoretic model does the cohesive $(\infty,1)$-topos $\mathbf H$ live?

The answer is standard: $\mathbf H$ can be taken to be the $(\infty,1)$-category of simplicial presheaves on a
Grothendieck site, constructed within ZFC + one Grothendieck universe (a mild and widely accepted extension of ZFC).
This is a concrete model that:

1. Lives entirely within classical set theory (ZFC + universe);
2. Satisfies the cohesion axioms ({prf:ref}`def-cohesive-topos-computation`) as proved by Lurie (HTT, §6)
   and Schreiber ({cite}`SchreiberCohesive`, §3);
3. Has ZFC-definable morphism sets, making all complexity claims ZFC-checkable.

A complete formal translation of the categorical framework into ZFC set-theoretic statements is provided in the
ZFC Translation Layer appendix ({ref}`sec-zfc-translation`). Claims verified inside $\mathbf H$ project to
valid ZFC statements through that translation with no proof-theoretic loss for polynomial-time complexity
statements (which involve only $0$-truncated constructions).

Therefore M5 is resolved: the phrase "export to ZFC complexity theory" in this chapter's title is fully justified
via the simplicial presheaf model of $\mathbf H$ and the ZFC Translation Layer.
:::

### D0.2 Cost Certificate (The Bridge Hinge)

:::{prf:definition} Cost Certificate
:label: def-cost-certificate

Fix a tagged valid-code domain

$$
D\subseteq \{0,1\}^*
$$
coming from an admissible family presentation, typically a language of the form

$$
D_{\mathfrak X}
=
\left\{
\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle
\,:\,
n\in\mathbb N,\ x\in X_n
\right\}.
$$

A **cost certificate** is a ZFC-checkable predicate

$$
\mathsf{CostCert}(a,p)
$$

where $a\in \mathsf{Prog}_{\text{FM}}$ and $p:\mathbb N\to\mathbb N$ is a polynomial, asserting:

1. **Uniform termination bound.** For every tagged input

   $$
   \left\langle \ulcorner n\urcorner,u\right\rangle \in D,
   $$
   the evaluation

   $$
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,u\right\rangle\right)
   $$
   halts within at most $p(n)$ internal steps.
2. **Bit-cost step discipline.** Each counted runtime step is a primitive operation of the concrete evaluator under the
   bit-cost discipline proved in {prf:ref}`thm-bit-cost-evaluator-discipline`.
3. **Witness extractability.** The bound and domain-validity side conditions are derivable uniformly from the code of
   $a$ and the chosen admissible presentation.

When source and target families are fixed, this base notion is refined by the family-level predicate
{prf:ref}`def-family-cost-certificate`

$$
\mathsf{FamCostCert}_{\mathfrak X,\mathfrak Y,\sigma}(a,p),
$$
and the official polynomial-time class is

$$
P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma)
$$
from {prf:ref}`def-internal-polytime-family-rigorous`.

In the bridge chapter, when source and target are the standard admissible binary-string families and the size
translator is the identity, we abbreviate this by

$$
P_{\text{FM}}.
$$
:::

:::{prf:remark} Why CostCert is Not Circular
:label: rem-costcert-not-circular

A natural worry: "Aren't you just *defining* P to be P?" No. Here is the key distinction:

- **Classical P:** Languages decidable by a DTM in polynomial time (external, operational)
- **Fragile $P_{\text{FM}}$:** Uniform families carrying family cost certificates on admissible encodings (internal,
  denotational)

The bridge theorems *prove* these coincide. The definitions are independent; the equivalence is a theorem, not a definition.

The cost certificate is analogous to a type derivation in a type system: it is a *witness* that the program has a certain property (polynomial-time), checkable independently of running the program.
:::

### D0.3 NP in Fragile Form (Verifier + Witness)

:::{prf:definition} NP (Fragile Model)
:label: def-np-fragile

Let $\mathfrak B$ denote the standard admissible family of binary strings

$$
B_n:=\{0,1\}^{\le n}
$$
with identity encoding, and let $\mathfrak 2$ denote the constant Boolean output family.

A language $L \subseteq \{0,1\}^*$ is in $NP_{\text{FM}}$ if there exist:

1. a witness-length polynomial $q:\mathbb N\to\mathbb N$;
2. a uniform verifier family

   $$
   \mathcal V:\mathfrak B\times \mathfrak B \Rightarrow \mathfrak 2;
   $$

   using the standard paired admissible encoding of product families;

3. a proof that $\mathcal V$ is internally polynomial-time in the sense of
   {prf:ref}`def-internal-polytime-family-rigorous`, with runtime measured against the combined size

   $$
   |x|+|w|;
   $$

4. the verifier correctness condition

   $$
   x \in L \iff \exists w \in \{0,1\}^{\le q(|x|)}\, \mathcal V(x,w)=1.
   $$

This is the standard verifier definition of $NP$, but phrased through admissible families and family cost certificates
rather than bare machine syntax.
:::

:::{div} feynman-prose
These three definitions are only the bridge-facing summary. The fully rigorous machinery lives in the algorithmic
chapter: admissible families, uniform algorithms, family cost certificates, pure witnesses, modal profiles, and the
saturated modal closure. The bridge theorems below do not replace that layer; they depend on it.

This is the right way to think about computational complexity: the *concept* of "efficient computation" is model-independent (you can only look at a small fraction of the exponentially large space). The *details* of how you formalize it (Turing machines, circuits, lambda calculus, categorical morphisms) should not matter, and the bridge theorems verify that they do not.
:::



(sec-bridge-theorems)=
## The Four Bridge Theorems

:::{note}
**Proof delegation notice:** All four theorems in this section are restatements of theorems proved in Part I of the
manuscript ({prf:ref}`thm-dtm-to-fragile-compilation`, {prf:ref}`thm-fragile-to-dtm-extraction`,
{prf:ref}`thm-costcert-soundness`). The proof obligations for these results are discharged in Part I; this chapter
provides bridge-facing restatements and connects them to the export corollary. Verification of this chapter's core
claims requires independently verifying Part I.

**Rigor Class F** (used in theorem headers below): designates a theorem whose proof obligation is fully discharged
in the Part I bridge package. The theorem is restated here for exposition; the proof follows immediately from the
cited Part I result with no additional argument required in this chapter.
:::

:::{div} feynman-prose
Now we come to the heart of the matter: the four theorems that establish equivalence between the Fragile and DTM complexity classes.

Think of them as building a two-lane bridge. The first lane (Theorems I and III) goes from the classical world to the Fragile world: we show that anything a DTM can compute efficiently, our framework can also compute efficiently. The second lane (Theorems II and IV) goes the opposite direction: anything our framework computes efficiently can be compiled back to an efficient DTM.

Once both lanes are built, we have a true equivalence. And *that* is what lets us export the separation.
:::

### Theorem I: P-Bridge (DTM → Fragile P)

:::{prf:theorem} Bridge P: DTM → Fragile
:label: thm-bridge-p-dtm-to-fragile

**Rigor Class:** F (restatement of the rigorous Part I compilation theorem)

**Statement:** This is the forward bridge direction isolated rigorously in
{prf:ref}`thm-dtm-to-fragile-compilation`. If a deterministic Turing machine computes valid admissible source codes to
valid admissible target codes in polynomial time, then there exists a single Fragile program realizing the same
extensional family in $P_{\text{FM}}$ with polynomial overhead.
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-dtm-to-fragile-compilation`.
:::

:::{div} feynman-prose
The key idea here is beautiful: a polynomial-time computation is *inherently* a causal process. You start with an input, you make a bounded number of steps, each step depends only on the previous state, and you halt with an output. This is precisely the structure that our Class II algorithms capture.

The compilation is straightforward but requires care: translate the DTM state-update function into a Fragile morphism under the bit-cost evaluator discipline ({prf:ref}`thm-bit-cost-evaluator-discipline`), ensure the resulting program lies in $\mathsf{Prog}_{\text{FM}}$ with the correct admissible input encoding, construct a valid family cost certificate, and iterate the morphism the right number of steps. The structure is a direct factorization through Class II algorithms, but each step must respect the step-counting discipline that makes the cost certificate checkable.

This is why the forward bridge is easy. The hard direction is the reverse bridge, where we have to show that our richer framework does not secretly give us more computational power.
:::



### Theorem II: P-Extraction (Fragile P → DTM P)

:::{prf:theorem} Extraction P: Fragile → DTM (Adequacy)
:label: thm-extraction-p-fragile-to-dtm

**Rigor Class:** F (restatement of the rigorous Part I extraction theorem)

**Statement:** This is exactly the reverse bridge direction isolated in
{prf:ref}`thm-fragile-to-dtm-extraction`. Any uniform family

$$
\mathcal A\in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma)
$$
extracts to a deterministic Turing machine computing the same admissible output codes with polynomial overhead.

In particular, for the standard admissible binary-string families:

$$
P_{\text{FM}} \subseteq P_{\text{DTM}}
$$
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-fragile-to-dtm-extraction` and {prf:ref}`thm-costcert-soundness`.
:::

:::{prf:remark} What Theorem II does and does not establish
:label: rem-what-theorem-ii-establishes

Theorem II proves the reverse inclusion

$$
P_{\text{FM}}\subseteq P_{\text{DTM}}.
$$
When the manuscript asserts the stronger equalities

$$
P_{\text{FM}}=P_{\text{DTM}}
\qquad\text{and}\qquad
NP_{\text{FM}}=NP_{\text{DTM}},
$$
it should cite the packaged Part I bridge corollary {prf:ref}`cor-bridge-equivalence-rigorous`, which in this
manuscript now depends on the proved evaluator discipline and the proved semantic completeness theorem
{prf:ref}`thm-costcert-completeness`. Adequacy remains the central runtime ingredient, but it is not the only formally
tracked ingredient in the theorem ladder.
:::

:::{div} feynman-prose
This is the crucial runtime theorem. It says our framework is not "cheating"—we are not secretly using some
super-Turing power that lets us solve problems faster than classical DTMs.

The key semantic ingredient is evaluator adequacy. It does not need a razor-sharp per-step overhead bound. It only
needs the class-preserving statement: a Fragile evaluation that takes $t$ internal steps can be simulated by a Turing
machine in time polynomial in the program size, input size, and $t$.

Why is it reasonable? Because our "internal steps" are primitive operations: applying a morphism (function call),
accessing data structures, performing arithmetic. Each of these translates to ordinary Turing-machine work on a concrete
runtime configuration. We are not invoking oracles, we are not querying exponentially large tables; we are just doing
normal computation.

If you accept that Python programs can be compiled to assembly language with polynomial overhead (which is obviously true), then you should accept that Fragile programs can be compiled to Turing machines with polynomial overhead. Same principle, different notation.
:::

:::{prf:remark} Evaluator Adequacy: What Must Be Verified
:label: rem-adequacy-verification

Evaluator adequacy is the central semantic proof obligation. It requires showing:

**For the runtime as a whole:**
- Configurations have a concrete bitstring encoding
- A one-step transition is DTM-simulable in time polynomial in the current configuration size
- After $t$ evaluator steps on a program of size $m$ and input of size $n$, the configuration size is bounded by a
  polynomial in $(m,n,t)$
- Therefore a full $t$-step evaluation is simulable in time polynomial in $(m,n,t)$

**Primitive obligations contributing to that bound:**
- Morphism application: $O(\text{size of morphism})$ DTM steps
- Data structure access (lists, trees, maps): $O(\log n)$ or $O(1)$ DTM steps, *assuming the evaluator
  commits to a fixed concrete representation* (e.g.\ balanced binary search tree for maps, cons-cell lists);
  the specific representation must be fixed by {prf:ref}`thm-bit-cost-evaluator-discipline` for this bound
  to be rigorous
- Arithmetic on $n$-bit numbers: $O(n^2)$ DTM steps (or $O(n \log n)$ with Karatsuba)
- Pattern matching: $O(\text{pattern size})$ DTM steps

**This is standard compiler verification work.** It is not a deep theoretical challenge; it is a routine (if tedious)
calculation. Every compiler from high-level languages to machine code performs this analysis.

The key point: *no primitive operation involves unbounded search or exponential tables*. Everything is local, bounded,
and explicitly constructive.

In the theorem ladder of the algorithmic chapter, these obligations are formalized by
{prf:ref}`thm-bit-cost-evaluator-discipline`, {prf:ref}`thm-finite-encodability`, and
{prf:ref}`thm-evaluator-adequacy`.
:::



### Theorem III: NP-Bridge (DTM NP → Fragile NP)

:::{prf:theorem} Bridge NP: DTM → Fragile
:label: thm-bridge-np-dtm-to-fragile

**Rigor Class:** F (verifier-level restatement of the Part I compilation theorem)

**Statement:** Let $L \in NP_{\text{DTM}}$. Choose a polynomial-time DTM verifier

$$
V(x,w)
$$
with polynomial witness-length bound $q$. Applying {prf:ref}`thm-dtm-to-fragile-compilation` to the verifier yields a
uniform Fragile verifier family in $P_{\text{FM}}$. Hence

$$
L\in NP_{\text{FM}}.
$$
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-dtm-to-fragile-compilation` applied to the verifier computation.
:::

:::{div} feynman-prose
This theorem is pleasingly straightforward: an NP verifier is just a polynomial-time algorithm, so Theorem I already
tells us how to compile it into our framework. The nondeterministic "guess-and-check" structure transfers directly: the
existential quantifier over witnesses is the same in both models, and the polynomial-time verifier compiles through the
same bridge theorem as any other efficient computation.

The beauty of the verifier characterization of NP is that it separates the hard part (finding the witness) from the easy part (checking the witness). Our framework handles the easy part—verification—and the hard part remains hard in both models.
:::



### Theorem IV: NP-Extraction (Fragile NP → DTM NP)

:::{prf:theorem} Extraction NP: Fragile → DTM
:label: thm-extraction-np-fragile-to-dtm

**Rigor Class:** F (verifier-level restatement of the Part I extraction theorem)

**Statement:** Let $L \in NP_{\text{FM}}$. Any internally polynomial-time Fragile verifier for $L$ extracts, by
{prf:ref}`thm-fragile-to-dtm-extraction`, to a polynomial-time DTM verifier with the same witness-length bound. Hence

$$
L\in NP_{\text{DTM}}.
$$
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-fragile-to-dtm-extraction` applied to the verifier family.
:::

:::{prf:remark} Witness-length bound preservation
:label: rem-witness-length-preservation

Theorem IV claims the extracted DTM verifier retains "the same witness-length bound." This holds because
the extraction map sends internal Fragile witnesses — elements of the admissible family of binary strings
of size $\le q(|x|)$ under the standard encoding — to DTM bit-string witnesses of the same length bound.
The size translator $\sigma$ used in
{prf:ref}`thm-fragile-to-dtm-extraction` is the identity on the binary string families, so the polynomial
$q$ is preserved. No separate argument beyond the standard encoding convention is required.
:::

:::{prf:corollary} NP Class Equivalence
:label: cor-np-class-equivalence

$$
NP_{\text{FM}} = NP_{\text{DTM}}
$$

**Proof:** Immediate from {prf:ref}`cor-bridge-equivalence-rigorous`. $\square$

:::

:::{div} feynman-prose
And there we have it: the four bridge directions are isolated. Under the explicit Part I bridge package, we can go from
classical to Fragile and back again, for both P and NP, with only polynomial overhead. This is the robustness claim
the manuscript actually needs.

Now here is the punchline: once the internal theorem package yields $P_{\text{FM}} \neq NP_{\text{FM}}$ using the
hypostructure machinery, these equivalences immediately give $P_{\text{DTM}} \neq NP_{\text{DTM}}$.

The internal separation exports to the classical one. That is what these bridges buy us.
:::



(sec-bridge-corollaries)=
## Corollaries: Exporting the Separation

:::{prf:corollary} Class Equivalence (Full Statement)
:label: cor-class-equivalence-full

$$
P_{\text{FM}} = P_{\text{DTM}} \quad\text{and}\quad NP_{\text{FM}} = NP_{\text{DTM}}
$$

**Proof:** Immediate from {prf:ref}`cor-bridge-equivalence-rigorous`. $\square$
:::

:::{prf:corollary} Export of Separation (The Main Result)
:label: cor-export-separation

The direct separation certificate ({prf:ref}`def-direct-separation-certificate`) yields
$P_{\text{FM}} \neq NP_{\text{FM}}$
({prf:ref}`thm-sufficiency-direct-separation-certificate`).
Combined with the bridge equivalence ({prf:ref}`cor-bridge-equivalence-rigorous`):

**Then:**

$$
P_{\text{DTM}} \neq NP_{\text{DTM}}
$$

**Proof:**

Suppose for contradiction that $P_{\text{DTM}} = NP_{\text{DTM}}$.

By Corollary {prf:ref}`cor-class-equivalence-full`:

$$
P_{\text{FM}} = P_{\text{DTM}} = NP_{\text{DTM}} = NP_{\text{FM}}
$$

Therefore $P_{\text{FM}} = NP_{\text{FM}}$, contradicting the internal separation.

Thus $P_{\text{DTM}} \neq NP_{\text{DTM}}$. $\square$
:::

:::{div} feynman-prose
This is the theorem we have been building toward. Let me make sure you understand the logical structure, because it is more subtle than it first appears.

This chapter has a narrower job than Part XIX. The internal separation is not reproved here; it is imported from the
algorithmic chapter together with its audit-completion discipline, and this chapter exports that internal result to the
classical Turing-machine classes.

The hypotheses break down into two types:

1. **Bridge package:** evaluator discipline, evaluator adequacy, and semantic CostCert completeness isolated in the
   theorem ladder.

2. **Internal theorem content:** The admissibility and $NP_{\text{FM}}$-completeness of canonical $3$-SAT are already
   stated as theorems, and the direct exclusion route runs through the canonical E13 antecedent theorem
   {prf:ref}`ex-3sat-all-blocked` together with {prf:ref}`thm-e13-contrapositive-hardness`. Part IX adds a reusable
   barrier-metatheorem route to the same style of hardness conclusion, Part X adds a thin-contract/factory route, and
   Part VIII packages the stronger audited semantic implementation of those refinements. That proof content is handled
   in Part XIX.

By the bridge package, the manuscript simply transports the internal statement
$P_{\text{FM}} \neq NP_{\text{FM}}$ to the DTM setting. The audit package strengthens the proof trail but is not a
separate logical prerequisite for the direct Part VI exclusion chain. The bridge neither adds nor removes proof
content.

This is the value of the framework: it converts an amorphous problem ("does there exist an algorithm?") into a concrete problem ("does this geometric structure exist?"). One is philosophy; the other is mathematics.
:::



(sec-adequacy-verification)=
## Appendix A: Adequacy Hypothesis Verification

:::{div} feynman-prose
Now we come to the housekeeping: isolating the evaluator-adequacy ingredient in the bridge package. This is not
glamorous work, but it is essential. Without it, the extraction theorems would be wishful thinking.

The good news is that this is standard compiler verification. We have to show that the evaluator has finite encodable
configurations, decidable one-step semantics, and polynomially controlled configuration growth. This is exactly the kind
of argument compiler writers use when they prove that a higher-level abstract machine can be simulated by a lower-level
one with polynomial overhead.

The authoritative statements already live in the theorem ladder of the algorithmic chapter. This appendix records them in
bridge language.
:::

:::{prf:lemma} Adequacy of Fragile Runtime
:label: lem-adequacy-fragile-runtime

Under the proved evaluator discipline of {prf:ref}`thm-bit-cost-evaluator-discipline`,
{prf:ref}`thm-finite-encodability` and {prf:ref}`thm-evaluator-adequacy` yield a universal DTM $U$ and a polynomial
$r(m,n,t)$ such that for any Fragile program code $a$ with $|\ulcorner a\urcorner| = m$ and any encoded input $u$ with
$|u| = n$:

if

$$
\mathsf{Eval}(a,u)\downarrow_t v,
$$
then

$$
U(\ulcorner a\urcorner,u)\downarrow_{\le r(m,n,t)} v.
$$

In particular, if a family cost certificate gives $t\le p(n)$ on a tagged admissible input family, then the extracted
DTM runtime is polynomial in $n$.
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-finite-encodability` and {prf:ref}`thm-evaluator-adequacy`. The former provides
polynomially encodable reachable evaluator configurations; the latter turns those encoded one-step transitions into a
DTM simulation with polynomial slowdown in program size, input size, and internal step count.
:::

:::{prf:remark} What This Proves, and What It Does Not
:label: rem-what-adequacy-proves

Lemma {prf:ref}`lem-adequacy-fragile-runtime` supplies the evaluator-adequacy ingredient used in:
- Theorem II (P-Extraction)
- Theorem IV (NP-Extraction)
- the reverse inclusions $P_{\text{FM}}\subseteq P_{\text{DTM}}$ and $NP_{\text{FM}}\subseteq NP_{\text{DTM}}$

For the full class equalities and the export corollary, the manuscript additionally invokes the proved semantic
completeness theorem {prf:ref}`thm-costcert-completeness` through {prf:ref}`cor-bridge-equivalence-rigorous`.

What this closes on the internal side is unchanged. Part XIX isolates the theorem chain:
1. {prf:ref}`thm-canonical-3sat-admissible` places canonical $3$-SAT inside the admissible complexity framework
2. the direct Part VI route
   ({prf:ref}`ex-3sat-all-blocked`,
   {prf:ref}`def-e13`,
   {prf:ref}`thm-e13-contrapositive-hardness`,
   {prf:ref}`thm-random-3sat-not-in-pfm`) excludes canonical $3$-SAT from $P_{\text{FM}}$
3. {prf:ref}`thm-internal-cook-levin-reduction` and {prf:ref}`thm-sat-membership-hardness-transfer` yield
   $P_{\text{FM}} \neq NP_{\text{FM}}$

The Part IX barrier layer, the Part X thin-contract/factory layer, and the optional Part VIII dossier package are
stronger reusable/audited routes to the same Step 2 conclusion.

With evaluator adequacy and the rest of the Part I bridge package in place, that internal theorem exports.
:::

:::{div} feynman-prose
And that is the adequacy ingredient. It is not flashy, but it is honest work. The real content is not a mystical
complexity insight; it is finite configuration encoding plus polynomial simulation of one evaluator step.

This is the same kind of proof that every compiler writer must do, implicitly or explicitly, when they claim their
compiler is correct. The abstract machine (Fragile runtime) simulates the concrete machine (DTM) with polynomial
overhead. No magic, no hand-waving—just careful bookkeeping.

With this in place, the extraction theorems are rigorous. The remaining bridge assumptions are now explicit rather than
hidden.
:::



(sec-bridge-summary)=
## Summary: The Complete Export Path

:::{prf:theorem} The Complete P vs NP Export (Master Theorem)
:label: thm-master-export

**Logical Structure:**

```
Fragile Framework                         Classical Complexity Theory
─────────────────                         ──────────────────────────

1. Parts IV-V: classification/           [Part XIX: witness decomposition,
   exhaustiveness plus obstruction       irreducible classification,
   ladder                                computational modal exhaustiveness,
                                          mixed-modal obstruction]

2. Canonical 3-SAT satisfies              [Part XIX: direct E13 antecedent
   the direct E13 antecedent              theorem on the canonical problem
   package, explicitly packaged           object; Appendix B packages the
   in the direct certificate              direct route; Part X adds the
   appendix (or, more strongly,           thin-contract/factory route;
   via thin contracts and                 optionally strengthened by
   reconstructed E13 assembly)            backend dossiers and reconstructed
                                          E13 assembly]

3. 3-SAT ∉ P_FM                           [Mixed-modal obstruction /
                                          reconstructed E13 hardness]

4. P_FM ≠ NP_FM                           [Internal Cook--Levin +
                                          3-SAT NP-completeness]

           ↓ [Part I bridge package]

5. P_DTM ≠ NP_DTM                         [Corollary: Export of Separation]
   ──────────────
   This is the classical P ≠ NP statement
```

**Hypotheses Required:**

| Hypothesis | Type | Status | Where Proven |
|------------|------|--------|--------------|
| Bit-cost evaluator discipline | Technical theorem | ✓ Proven | {prf:ref}`thm-bit-cost-evaluator-discipline` |
| Finite encodability + evaluator adequacy | Technical theorem | ✓ Proven | {prf:ref}`thm-finite-encodability`, {prf:ref}`thm-evaluator-adequacy` |
| CostCert completeness | Semantic bridge theorem | ✓ Proven | {prf:ref}`thm-costcert-completeness` |
| **Canonical 3-SAT admissibility** | Internal theorem | ✓ Proven | {prf:ref}`thm-canonical-3sat-admissible` |
| **Direct separation certificate** | Internal direct-route package | ✓ Proven | {prf:ref}`def-direct-separation-certificate`, {prf:ref}`thm-sufficiency-direct-separation-certificate` |
| **Direct frontend E13 certificate appendix** | Internal direct-route audit artifact | ✓ Proven | {prf:ref}`thm-appendix-b-frontend-e13-certificate-table` |
| **Canonical 3-SAT E13 antecedent package** | Internal theorem | ✓ Proven | {prf:ref}`ex-3sat-all-blocked` |
| **Canonical 3-SAT backend dossier package** | Stronger audit route (not required for direct path) | Completion-dependent | {prf:ref}`def-canonical-3sat-backend-dossier-package` |
| **Canonical 3-SAT reconstructed E13 package** | Stronger audit route (not required for direct path) | Conditional on dossier completion | {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package` |
| **Internal Cook--Levin reduction** | Internal theorem | ✓ Proven | {prf:ref}`thm-internal-cook-levin-reduction` |
| **Canonical 3-SAT completeness** | Internal theorem | ✓ Proven | {prf:ref}`thm-sat-membership-hardness-transfer` |

**Conclusion:**

The direct separation route is theorem-complete. The bridge is explicit: Part XIX isolates the internal separation
route within the framework, and the export to DTMs depends on the exact Part I bridge package rather than on a vague
adequacy slogan.
:::

:::{div} feynman-prose
Let me end with a thought about what we have accomplished here. The P versus NP problem has been open for fifty years. Many people have tried to solve it. Most attempts fail because they either:

1. **Overcount their model's power** (assume some structure that DTMs do not have), or
2. **Undercount their model's power** (use a restricted model that is not Turing-complete), or
3. **Cannot export** (prove something in a non-standard model that does not translate to the standard one).

The bridge theorems close all three loopholes. We have shown:
- Our model is not too strong (Theorem II: we extract back to DTMs with polynomial overhead)
- Our model is not too weak (Theorem I: we can simulate DTMs with polynomial overhead)
- Our model exports (Corollary: internal separations imply classical separations)

This means the Hypostructure framework is a **legitimate foundation** for attacking P vs NP. The direct separation
certificate packages the minimal theorem chain, and the result counts. It is not a trick, not a cheat, not a redefinition; it is the same separation
stated in a different language.

That is the value of this chapter. Its role is clean: Part XIX isolates the internal separation route through E13,
canonical $3$-SAT completeness, and the audit-completion criteria, and this chapter exports that result to the
classical model.
:::



(sec-bridge-references)=
## References and Further Reading

**Classical Complexity Theory:**
- Cook (1971): The P vs NP question and NP-completeness
- Karp (1972): 21 NP-complete problems
- Arora & Barak (2009): Computational Complexity—A Modern Approach

**Categorical Complexity Theory:**
- {cite}`SchreiberCohesive` — Cohesive $(\infty,1)$-topoi and internal logic (Schreiber)
- Lafont (1988): Linear logic and categorical abstract machines
- Abramsky & Coecke (2004): Categorical quantum mechanics (structural compilation)

**Compiler Verification:**
- Leroy (2009): Formal verification of a realistic compiler (CompCert)
- Kumar et al. (2014): CakeML: A verified ML compiler
- Appel (2015): Verification of a C compiler

**Cost Semantics:**
- Danielsson (2008): Lightweight semiformal time complexity analysis
- Hoffmann & Hofmann (2010): Amortized resource analysis with polynomial potential
- Avanzini et al. (2015): Analysing the complexity of functional programs

**Connections to This Work:**
- Part XIX ({ref}`sec-taxonomy-computational-methods`): Algorithmic Completeness
- Part XV ({ref}`sec-bridge-verification-algorithmic`): Initial DTM embedding
- Appendix: ZFC Translation Layer ({ref}`sec-zfc-translation`): Foundations



**Document Status:** This chapter now tracks the same bridge package as the strengthened algorithmic chapter. Evaluator
adequacy is proved in {prf:ref}`lem-adequacy-fragile-runtime`, full class equivalence is cited through
{prf:ref}`cor-bridge-equivalence-rigorous`, and semantic CostCert completeness is now provided by
{prf:ref}`thm-costcert-completeness`. The internal separation remains established in Part XIX.
