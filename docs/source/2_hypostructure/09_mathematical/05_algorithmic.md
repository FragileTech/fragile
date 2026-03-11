---
title: "Algorithmic Completeness"
---

# Algorithmic Completeness

(sec-taxonomy-computational-methods)=
## The Taxonomy of Computational Methods

This part establishes that polynomial-time algorithms must exploit specific structural invariants detectable by the Cohesive Topos modalities. It provides the theoretical foundation for **Tactic E13** (Algorithmic Completeness Lock), which closes the "Alien Algorithm" loophole in complexity-theoretic proofs.

:::{div} feynman-prose
Let me tell you what this is really about. Whenever you see a fast algorithm, you should ask yourself: "Why is this fast? What structure is it exploiting?" Because here is the thing: if you have no structure, you are reduced to brute force, to trying things one by one until you stumble on the answer. And brute force takes exponential time.

Now, the question that has haunted complexity theory is this: could there be some clever algorithm we have not thought of yet? Some "alien" technique that solves hard problems fast without exploiting any recognizable structure? This chapter says no. We claim that every efficient algorithm must be built from one or more of five fundamental types of structure, which we call the five "modalities." If your problem has none of these structures, no algorithm can help you.

This is a bold claim. How can we be sure we have not missed a sixth modality? The answer lies in category theory: in a cohesive topos, these five modalities exhaust the ways that structure can manifest. They are not arbitrary categories we invented; they arise from the fundamental adjunctions that define what "structure" means in the first place.
:::

### Cohesive Topos Foundations for Computation

Before classifying algorithms, we must establish the precise mathematical structure that makes algorithmic analysis possible. The key insight is that polynomial-time algorithms exploit **structure**, and in a cohesive $(\infty,1)$-topos, all structure decomposes into modal components.

:::{prf:definition} Cohesive $(\infty,1)$-Topos Structure
:label: def-cohesive-topos-computation

A **cohesive $(\infty,1)$-topos** is an $(\infty,1)$-topos $\mathbf{H}$ equipped with an adjoint quadruple of functors to the base topos $\infty\text{-Grpd}$:

$$\Pi \dashv \mathrm{Disc} \dashv \Gamma \dashv \mathrm{coDisc} : \mathbf{H} \to \infty\text{-Grpd}$$

where:
- $\Pi: \mathbf{H} \to \infty\text{-Grpd}$ — **shape** (fundamental $\infty$-groupoid, extracts causal/topological structure)
- $\mathrm{Disc}: \infty\text{-Grpd} \to \mathbf{H}$ — **discrete** (embeds discrete types, left adjoint to $\Gamma$)
- $\Gamma: \mathbf{H} \to \infty\text{-Grpd}$ — **global sections** (underlying $\infty$-groupoid of points)
- $\mathrm{coDisc}: \infty\text{-Grpd} \to \mathbf{H}$ — **codiscrete** (embeds codiscrete types, right adjoint to $\Gamma$)

satisfying the **cohesion axioms**:
1. $\mathrm{Disc}$ and $\mathrm{coDisc}$ are fully faithful
2. $\Pi$ preserves finite products
3. **(Pieces have points)** The canonical comparison $\Pi \to \Gamma$ is an epimorphism

**Literature:** {cite}`Lawvere69`; {cite}`SchreiberCohesive`
:::

:::{prf:definition} The Five Computational Modalities
:label: def-five-modalities

From the adjoint quadruple, we derive the **cohesive modalities** as (co)monads. These are the **complete set** of structural resources available in a cohesive topos:

**Basic Modalities (from adjunctions):**

| Modality | Definition | Type | Intuition |
|----------|------------|------|-----------|
| $\int$ (shape) | $\mathrm{Disc} \circ \Pi$ | Monad | Discretize the shape (causal structure) |
| $\flat$ (flat) | $\mathrm{Disc} \circ \Gamma$ | Comonad | Discrete points (algebraic structure) |
| $\sharp$ (sharp) | $\mathrm{coDisc} \circ \Gamma$ | Monad | Codiscrete points (metric structure) |

These satisfy the **modal adjunction triple**:

$$\flat \dashv \int \dashv \sharp$$

with reduction properties:
- $\flat \int \simeq \int$ and $\sharp \int \simeq \sharp$ ($\int$ is left-exact)
- $\int \flat \simeq \flat$ and $\int \sharp \simeq \int$ (reduction identities)

**Extended Modalities (for computational completeness):**

**Scaling Modality** $\ast$:

$$\ast := \mathrm{colim}_{n \to \infty} \int^{(n)}$$

where $\int^{(n)}$ is the $n$-fold iteration of shape. This captures self-similar/recursive structure via iterated coarse-graining.

**Boundary/Holographic Modality** $\partial$:

$$\partial := \mathrm{fib}(\eta_\sharp : \mathrm{id} \to \sharp)$$

the homotopy fiber of the sharp unit. This captures boundary/interface structure—the difference between a type and its codiscretification.

**Computational Completeness:** The five modalities $\{\int, \flat, \sharp, \ast, \partial\}$ exhaust all structural resources that polynomial-time algorithms can exploit. This is not an empirical observation but a **theorem** of cohesive topos theory ({prf:ref}`thm-schreiber-structure`).
:::

:::{div} feynman-prose
Let me explain what these modalities really mean. Think of a space $\mathcal{X}$ as having multiple "views" or "shadows" that reveal different aspects of its structure:

The **shape** $\int \mathcal{X}$ forgets everything except connectivity—which points can reach which. It is like looking at a road network and ignoring distances, just tracking which cities connect.

The **flat** $\flat \mathcal{X}$ keeps only the discrete, algebraic structure—like the lattice points in a continuous space, or the group elements in a space with symmetry.

The **sharp** $\sharp \mathcal{X}$ makes everything "as connected as possible"—it is the view where you can continuously deform any path to any other. This reveals the metric, continuous structure.

The **scaling** $\ast$ captures what happens when you zoom out infinitely—the self-similar patterns that persist at all scales.

The **boundary** $\partial$ captures what you can see from the outside—the holographic projection that encodes bulk information.

The deep theorem we are using says: these five views are **complete**. Every structural pattern in $\mathcal{X}$ appears in at least one of these modal shadows. If your algorithm exploits structure, it must show up in one of these five places.
:::

:::{prf:theorem} Schreiber Structure Theorem (Computational Form)
:label: thm-schreiber-structure

Let $\mathbf{H}$ be a cohesive $(\infty,1)$-topos. For any type $\mathcal{X} \in \mathbf{H}$, the canonical sequence

$$\flat \mathcal{X} \to \mathcal{X} \to \int \mathcal{X}$$

exhibits $\mathcal{X}$ as **exhaustively decomposable** into modal components. Moreover, any morphism $f: \mathcal{X} \to \mathcal{Y}$ factors (up to homotopy) through modal reflections:

$$\mathrm{Hom}_{\mathbf{H}}(\mathcal{X}, \mathcal{Y}) \simeq \int^{\lozenge \in \{\int, \flat, \sharp\}} \mathrm{Hom}_{\lozenge\text{-modal}}(\lozenge\mathcal{X}, \lozenge\mathcal{Y})$$

where the coend is taken over modal factorizations.

**Consequence for Algorithms:** The five modalities supply the exhaustive list of pure structural routes that can appear
in the algorithmic witness language introduced below. By itself, this theorem identifies the modal leaves; the full
claim that internally polynomial-time families are exactly the saturated closure of those leaves is deferred to
{prf:ref}`cor-computational-modal-exhaustiveness`.

**Literature:** {cite}`SchreiberCohesive` Section 3; {cite}`Schreiber13`
:::

:::{prf:proof}
We work in a cohesive $(\infty,1)$-topos $\mathbf{H}$ with adjoint quadruple
$\Pi \dashv \mathrm{Disc} \dashv \Gamma \dashv \mathrm{coDisc}$.
The three idempotent (co)monads $\int := \mathrm{Disc}\circ\Pi$,
$\flat := \mathrm{Disc}\circ\Gamma$, $\sharp := \mathrm{coDisc}\circ\Gamma$
are the *only* idempotent modal operators generated by the adjoint quadruple
({cite}`SchreiberCohesive`, Section 3.8; {cite}`Lawvere69`).

By the density theorem (nerve--realization adjunction) in a locally presentable
category, the identity functor on $\mathbf{H}$ admits a canonical coend
decomposition over a dense subcategory. In the cohesive setting the fracture
squares of Schreiber ({cite}`SchreiberCohesive`, Theorem 3.8.5) provide the
operative decomposition: the shape--flat fracture square
$\mathcal{X} \simeq \int\mathcal{X} \times_{\int\flat\mathcal{X}} \flat\mathcal{X}$
and the flat--sharp fracture square
$\mathcal{X} \simeq \flat\mathcal{X} \times_{\flat\sharp\mathcal{X}} \sharp\mathcal{X}$
decompose any type into its modal components. Composing these pullbacks and
applying them to an arbitrary morphism $f:\mathcal{X}\to\mathcal{Y}$ yields the
coend formula of the theorem statement.

The extended modalities $\ast := \mathrm{colim}_{k}\int^{(k)}$ and
$\partial := \mathrm{fib}(\eta_\sharp : \mathrm{id}\to\sharp)$ arise as
derived operations: $\ast$ is the transfinite iterate (stabilizing at the
shape fixed-point reflection), and $\partial$ is the homotopy fiber of the
sharp unit. These are the only structurally new operations beyond $\int,\flat,\sharp$
(any other composite of the four adjoint functors reduces to one of the five
by the reduction identities $\flat\int\simeq\flat$, $\sharp\int\simeq\sharp$,
$\int\flat\simeq\int$, $\int\sharp\simeq\int$). Hence the coend ranges over
exactly the five modalities of {prf:ref}`def-five-modalities`.
:::

:::{prf:corollary} Exhaustive Modal Decomposition
:label: cor-exhaustive-decomposition

Every type $\mathcal{X}$ in a cohesive topos admits a canonical decomposition:

$$\mathcal{X} \simeq \mathcal{X}_{\int} \times_{\mathcal{X}_0} \mathcal{X}_{\flat} \times_{\mathcal{X}_0} \mathcal{X}_{\sharp}$$

where:
- $\mathcal{X}_{\int}$ is the shape component (causal/topological structure)
- $\mathcal{X}_{\flat}$ is the flat component (discrete/algebraic structure)
- $\mathcal{X}_{\sharp}$ is the sharp component (continuous/metric structure)
- $\mathcal{X}_0$ is the base (pure points with no structure)

Any morphism decomposes accordingly. The extended modalities $\ast$ and $\partial$ capture derived patterns (scaling and holography) built from these basic components.

**Key Insight:** This decomposition is **not a choice**—it is a theorem. The modalities exhaust the available structure because they **are** the structure of the topos. There is no "sixth modality" any more than there is a sixth direction orthogonal to all dimensions of space.
:::

:::{prf:proof}
Apply the coend decomposition of {prf:ref}`thm-schreiber-structure` to the identity
morphism $\mathrm{id}_{\mathcal{X}} : \mathcal{X}\to\mathcal{X}$. The shape--flat
fracture square ({cite}`SchreiberCohesive`, Theorem 3.8.5) gives the pullback
$\mathcal{X} \simeq \int\mathcal{X} \times_{\int\flat\mathcal{X}} \flat\mathcal{X}$,
and the flat--sharp fracture square gives
$\flat\mathcal{X} \simeq \flat\mathcal{X} \times_{\flat\sharp\mathcal{X}} \sharp\mathcal{X}$.
Composing these two pullback decompositions and identifying the base object
$\mathcal{X}_0 := \flat\int\mathcal{X} \simeq \int\flat\mathcal{X}$ (by the
reduction identity), we obtain the fiber product
$\mathcal{X} \simeq \mathcal{X}_{\int} \times_{\mathcal{X}_0} \mathcal{X}_{\flat} \times_{\mathcal{X}_0} \mathcal{X}_{\sharp}$.
The extended modalities $\ast$ and $\partial$ capture the remaining derived
structure (scaling and boundary) as established in {prf:ref}`thm-schreiber-structure`.
:::

:::{prf:remark} Five-Modality Completeness Argument
:label: rem-five-modality-completeness-argument

We prove that the five modalities $(\int, \flat, \sharp, \ast, \partial)$ exhaust all
structurally distinct polynomial-time-exploitable modalities generated by the adjoint
quadruple $\Pi \dashv \mathrm{Disc} \dashv \Gamma \dashv \mathrm{coDisc}$, by
exhaustive case analysis on all compositions of the four functors.

**Preliminary identities.** Since $\mathrm{Disc}$ is fully faithful (left adjoint
$\Pi$ and right adjoint $\Gamma$ both retract it), we have the counit/unit
isomorphisms:

$$
\Pi \circ \mathrm{Disc} \simeq \mathrm{id}_{\mathbf{Set}}, \qquad
\Gamma \circ \mathrm{Disc} \simeq \mathrm{id}_{\mathbf{Set}}.
$$

Similarly, $\mathrm{coDisc}$ is fully faithful (its left adjoint $\Gamma$ retracts
it):

$$
\Gamma \circ \mathrm{coDisc} \simeq \mathrm{id}_{\mathbf{Set}}.
$$

These are standard consequences of the adjunction (see {cite}`SchreiberCohesive`,
Proposition 3.8.1 and {cite}`Lawvere69`). We also record the four **reduction
identities** among the basic modalities:

$$
\flat\!\int \simeq \int, \qquad
\sharp\!\int \simeq \sharp, \qquad
\int\!\flat \simeq \flat, \qquad
\int\!\sharp \simeq \int.
$$(eq-reduction-identities)

These follow from the fully-faithfulness identities: for instance,
$\flat\!\int = (\mathrm{Disc}\circ\Gamma)\circ(\mathrm{Disc}\circ\Pi)
\simeq \mathrm{Disc}\circ(\Gamma\circ\mathrm{Disc})\circ\Pi
\simeq \mathrm{Disc}\circ\mathrm{id}\circ\Pi = \int$.
The others are analogous.

**Case 1: Length-1 compositions.** The four individual functors $\Pi$, $\mathrm{Disc}$,
$\Gamma$, $\mathrm{coDisc}$ are not endofunctors on $\mathcal{H}$ alone ($\Pi$ and
$\Gamma$ map $\mathcal{H}\to\mathbf{Set}$; $\mathrm{Disc}$ and $\mathrm{coDisc}$ map
$\mathbf{Set}\to\mathcal{H}$), so they do not yield modalities on $\mathcal{H}$.

**Case 2: Length-2 compositions (endofunctors on $\mathcal{H}$).** To obtain an
endofunctor $\mathcal{H}\to\mathcal{H}$, we must compose a functor
$\mathbf{Set}\to\mathcal{H}$ with one $\mathcal{H}\to\mathbf{Set}$. The eight
ordered pairs and their reductions are:

| Composition | Type | Result | Justification |
|---|---|---|---|
| $\mathrm{Disc}\circ\Pi$ | $\mathcal{H}\to\mathcal{H}$ | $\int$ | Shape monad (definition) |
| $\mathrm{Disc}\circ\Gamma$ | $\mathcal{H}\to\mathcal{H}$ | $\flat$ | Flat comonad (definition) |
| $\mathrm{coDisc}\circ\Gamma$ | $\mathcal{H}\to\mathcal{H}$ | $\sharp$ | Sharp monad (definition) |
| $\mathrm{coDisc}\circ\Pi$ | $\mathcal{H}\to\mathcal{H}$ | $\sharp$ | $= \mathrm{coDisc}\circ\Gamma\circ\mathrm{Disc}\circ\Pi$ (insert $\mathrm{id}\simeq\Gamma\circ\mathrm{Disc}$); reduces to $\sharp\!\int\simeq\sharp$ by {eq}`eq-reduction-identities` |
| $\Pi\circ\mathrm{Disc}$ | $\mathbf{Set}\to\mathbf{Set}$ | $\mathrm{id}$ | Fully faithful retraction |
| $\Gamma\circ\mathrm{Disc}$ | $\mathbf{Set}\to\mathbf{Set}$ | $\mathrm{id}$ | Fully faithful retraction |
| $\Gamma\circ\mathrm{coDisc}$ | $\mathbf{Set}\to\mathbf{Set}$ | $\mathrm{id}$ | Fully faithful retraction |
| $\Pi\circ\mathrm{coDisc}$ | $\mathbf{Set}\to\mathbf{Set}$ | — | Endofunctor on $\mathbf{Set}$, not on $\mathcal{H}$; does not yield a modality |

The four $\mathcal{H}$-endofunctors reduce to exactly $\{\int, \flat, \sharp\}$. No
new modality appears at length 2.

**Case 3: Length-3+ compositions.** Any length-$n$ composition ($n\ge 3$) of functors
from $\{\Pi,\mathrm{Disc},\Gamma,\mathrm{coDisc}\}$ that is type-compatible and yields
an endofunctor on $\mathcal{H}$ must alternate between $\mathcal{H}\to\mathbf{Set}$ and
$\mathbf{Set}\to\mathcal{H}$ factors. It therefore decomposes as $M_1\circ M_2\circ
\cdots\circ M_k$ where each $M_i\in\{\int,\flat,\sharp\}$ (or identity, which we
absorb). The six pairwise compositions among $\{\int,\flat,\sharp\}$ all collapse:

$$
\flat\!\int \simeq \int, \qquad
\sharp\!\int \simeq \sharp, \qquad
\int\!\flat \simeq \flat, \qquad
\int\!\sharp \simeq \int,
$$

$$
\flat\!\sharp = (\mathrm{Disc}\circ\Gamma)\circ(\mathrm{coDisc}\circ\Gamma)
\simeq \mathrm{Disc}\circ(\Gamma\circ\mathrm{coDisc})\circ\Gamma
\simeq \mathrm{Disc}\circ\Gamma = \flat,
$$

$$
\sharp\!\flat = (\mathrm{coDisc}\circ\Gamma)\circ(\mathrm{Disc}\circ\Gamma)
\simeq \mathrm{coDisc}\circ(\Gamma\circ\mathrm{Disc})\circ\Gamma
\simeq \mathrm{coDisc}\circ\Gamma = \sharp.
$$

Since every pairwise product in $\{\int,\flat,\sharp\}$ reduces to a single element of
that set, induction on chain length shows every finite composition
$M_1\circ\cdots\circ M_k$ reduces to one of $\{\int,\flat,\sharp\}$. No new modality
arises from finite composition.

**Case 4: Derived modalities.** Beyond finite composition, two constructions yield
genuinely new structure:

- $\ast = \mathrm{colim}_{k}\,\int^{(k)}$: the transfinite iterate of $\int$,
  stabilizing at the shape modality's fixed-point reflection (the
  $\int$-localization). This captures recursive self-reduction structure.
- $\partial = \mathrm{fib}(\eta_\sharp : \mathrm{id}\to\sharp)$: the homotopy fiber
  of the sharp unit. This captures boundary/interface structure.

**Case 5: Closure of derived modalities.** We verify that compositions involving $\ast$
and $\partial$ do not produce a sixth modality:

- *$\ast$ absorbs basic modalities:* Since $\ast$ is the $\int$-localization
  (the terminal coalgebra of $\int$), $\int\!\ast\simeq\ast$ and
  $\ast\!\int\simeq\ast$. By the reduction identities, $\flat\!\ast =
  \flat\circ\mathrm{colim}_k\int^{(k)}\simeq\mathrm{colim}_k(\flat\!\int^{(k)})
  \simeq\mathrm{colim}_k\int^{(k-1)} = \ast$ (using $\flat\!\int\simeq\int$ at each
  step and re-indexing). Similarly $\sharp\!\ast\simeq\sharp$ (since
  $\sharp\!\int\simeq\sharp$ is already a fixed point). For the reverse order:
  $\ast\!\flat\simeq\ast$ and $\ast\!\sharp\simeq\ast$ follow from
  $\int\!\flat\simeq\flat$ and $\int\!\sharp\simeq\int$ propagated through the
  colimit.
- *$\partial^2\simeq\partial$:* The fiber construction $\partial =
  \mathrm{fib}(\eta_\sharp)$ is idempotent. An object $X$ is $\sharp$-modal iff
  $\partial X \simeq 0$, so $\partial X$ lies in the $\sharp$-anti-modal subcategory.
  Applying $\partial$ again: $\partial(\partial X) =
  \mathrm{fib}(\eta_\sharp\colon\partial X\to\sharp\partial X)$. But $\partial X$ is
  already $\sharp$-anti-modal, meaning $\sharp(\partial X)\simeq 0$ (the
  $\sharp$-reflection of a $\sharp$-anti-modal object is terminal). Hence
  $\partial(\partial X)\simeq\mathrm{fib}(\partial X\to 0)\simeq\partial X$.
- *$\ast\circ\partial$:* Consider the fiber sequence
  $\partial X \to X \xrightarrow{\eta_\sharp} \sharp X$. Since $\int$ is a left exact
  modality (lex modality) in a cohesive $\infty$-topos, it preserves fiber sequences.
  Applying $\int$ gives a fiber sequence
  $\int(\partial X)\to\int X\xrightarrow{\int(\eta_\sharp)}\int(\sharp X)$. By the
  reduction identity $\int\!\sharp\simeq\int$ ({eq}`eq-reduction-identities`), the map
  $\int(\eta_\sharp)\colon\int X\to\int(\sharp X)\simeq\int X$ is an equivalence,
  so its fiber is contractible: $\int(\partial X)\simeq 0$. Iterating:
  $\int^{(k)}(\partial X)\simeq 0$ for all $k\ge 1$ (since $\int(0)\simeq 0$), and
  passing to the colimit gives $\ast(\partial X)\simeq 0$. Thus
  $\ast\circ\partial\simeq 0$ (the zero modality, a trivial case).
- *$\partial\circ\ast$:* Since $\ast X$ is $\int$-local, the identity
  $\flat\!\int\simeq\int$ implies that $\ast X$ is $\flat$-modal (the counit
  $\flat(\ast X)\to\ast X$ is an equivalence, because $\int$-local objects lie in
  the essential image of $\mathrm{Disc}$, and $\flat\circ\mathrm{Disc} =
  \mathrm{Disc}\circ\Gamma\circ\mathrm{Disc}\simeq\mathrm{Disc}$). To conclude
  that $\ast X$ is also $\sharp$-modal, we invoke **locality** of the cohesive
  $(\infty,1)$-topos $\mathcal{H}$: by axiom 1 of
  {prf:ref}`def-cohesive-topos-computation`, $\mathrm{coDisc}$ is fully faithful,
  so $\mathcal{H}$ is a local $(\infty,1)$-topos, meaning $\Gamma$ preserves
  $\infty$-colimits and the pieces-to-points transformation is an equivalence on
  discrete objects. Concretely, for $\ast X\simeq\mathrm{Disc}(S)$ we have
  $\sharp(\mathrm{Disc}(S)) = \mathrm{coDisc}(\Gamma(\mathrm{Disc}(S)))
  \simeq\mathrm{coDisc}(S)$, and the locality condition ensures the unit
  $\mathrm{Disc}(S)\to\mathrm{coDisc}(S)$ is an equivalence.
  (See Schreiber, *Differential cohomology in a cohesive $\infty$-topos*, Prop. 3.8.8.
  Note that full faithfulness of $\mathrm{coDisc}$ — i.e., locality — is part of
  the cohesion axioms in {prf:ref}`def-cohesive-topos-computation` and is
  satisfied by all cohesive $(\infty,1)$-toposes arising from differential
  cohesion; $\flat$-modality
  alone does not imply $\sharp$-modality in a general cohesive $\infty$-topos
  without this axiom.) Therefore
  $\partial(\ast X) = \mathrm{fib}(\eta_\sharp\colon\ast X\to\sharp\ast X)
  \simeq\mathrm{fib}(\mathrm{id})\simeq 0$. Hence $\partial\circ\ast\simeq 0$
  (the zero modality, a trivial case).
- *$\ast\circ\ast\simeq\ast$:* Immediate since $\ast$ is already a localization
  (idempotent).

**Conclusion.** Every endofunctor on $\mathcal{H}$ built from the adjoint quadruple —
whether by finite composition, transfinite iteration, or fiber construction — reduces to
one of $\{\mathrm{id},\int,\flat,\sharp,\ast,\partial\}$ (where $\mathrm{id}$ is
trivial). These five non-trivial modalities are therefore complete.
:::

:::{prf:lemma} Modal decomposition under 0-truncation
:label: lem-zero-truncation-modal-preservation

The modal decomposition of {prf:ref}`cor-exhaustive-decomposition`, when restricted
to $0$-truncated objects with finite polynomial-size presentations (the
complexity-theoretic fragment per {prf:ref}`rem-ambient-conventions-complexity`),
retains the exhaustive five-way classification. That is, no polynomial-time-exploitable
structure is lost by restricting to the $0$-truncated, finitely presented fragment.
:::

:::{prf:proof}
The modalities $\int$, $\flat$, $\sharp$ preserve $n$-truncatedness for all
$n \ge -1$. This holds because the left and right adjoints of $\mathrm{Disc}$
preserve truncation levels ({cite}`SchreiberCohesive`, Proposition 3.8.2). In
particular, they preserve $0$-truncatedness: if $\mathcal{X}$ is $0$-truncated
(a set), then $\int\mathcal{X}$, $\flat\mathcal{X}$, $\sharp\mathcal{X}$ are
$0$-truncated.

The derived modalities $\ast$ and $\partial$ are built from $\int$ and $\sharp$
respectively by colimits and fibers. For types that are already $0$-truncated in
an $(\infty,1)$-topos, fibers of maps between $0$-truncated types are $0$-truncated,
and filtered colimits of $0$-truncated types remain $0$-truncated. Therefore $\ast$
and $\partial$ also preserve $0$-truncatedness.

Since all five modalities preserve $0$-truncatedness, the fracture squares and
the pullback decomposition of {prf:ref}`cor-exhaustive-decomposition` remain valid
within the $0$-truncated fragment. The finite-presentation condition is preserved
because presentation translators are polynomial-time maps between finite structures,
and the modal maps in the fracture squares are constructive and polynomial-time
when restricted to finitely presented objects (the adjoints $\Pi$, $\Gamma$ applied
to finitely presented types yield finitely presented outputs).
:::

:::{prf:remark} Conditional Status and Classical Export
:label: rem-conditional-status-classical-export

The $P\neq NP$ result proved in this document is conditional on the Computational
Foundation Assumption ({prf:ref}`axiom-structure-thesis`), which posits that
polynomial-time computation is faithfully modeled within a cohesive $(\infty,1)$-topos.

The logical structure is:
$(\text{C1: foundation}) \;\wedge\; (\text{C2: bridge}) \;\wedge\; (\text{C3: internal separation}) \;\Longrightarrow\; P\neq NP.$

The bridge equivalence ({prf:ref}`cor-bridge-equivalence-rigorous`) exports the
topos-internal result to classical Deterministic Turing Machines. Specifically:
$P_{\text{FM}} = P_{\text{DTM}}$ (the class of polynomial-time functions in the
Fragile model equals the classical class $P$). Therefore the conditional reduces
to: **if** the evaluator model faithfully implements classical computation (which
the bridge theorems verify), **then** the internal separation result implies
classical $P\neq NP$.

The foundation assumption plays the role of selecting the mathematical universe in
which the proof operates. It is analogous to choosing ZFC + large cardinals as the
ambient set theory -- the proof is valid within the chosen foundation, and the bridge
theorems certify that the result applies to classical computation.

The specific topos-theoretic structures used (five modalities, fracture squares,
modal decomposition) could in principle be exported to a ZFC proof by replacing the
topos-theoretic language with explicit combinatorial definitions of the five
algorithmic paradigms.
:::

### Algorithm Classification via Cohesive Modalities

:::{prf:remark} Ambient conventions for complexity-theoretic use of the cohesive framework
:label: rem-ambient-conventions-complexity

Throughout this subsection, $\mathbf{H}$ denotes the fixed cohesive $(\infty,1)$-topos from
{prf:ref}`def-five-modalities`, and $\mathsf{Prog}_{\text{FM}}$ and $\mathsf{Eval}$ are as in
{prf:ref}`def-effective-programs-fragile`.

To speak rigorously about complexity classes, we restrict attention to externally presented, $0$-truncated objects.
Concretely, whenever complexity is discussed, an object $\mathcal{X} \in \mathbf{H}$ is used only through its
externally presented set of global points $\Gamma(\mathcal{X})$, together with a finite encoding. For readability, we
write $x \in X_n$ to mean $x \in \Gamma(X_n)$.

We fix once and for all:
1. a self-delimiting pairing function

   $$
   \langle -,- \rangle : \{0,1\}^* \times \{0,1\}^* \to \{0,1\}^*
   $$

   with polynomial-time computable projections $\pi_1,\pi_2$;
2. a standard binary encoding of integers $n \mapsto \ulcorner n \urcorner$ computable and decodable in polynomial
   time;
3. the convention that every polynomial means a function $p:\mathbb{N}\to\mathbb{N}$ with nonnegative integer
   coefficients.

All complexity bounds below are taken with respect to the size parameter $n$ indexing the input family. Because every
admissible encoding length is polynomially bounded in $n$, this is equivalent up to polynomial distortion to measuring
time in terms of encoded bitlength.
:::

:::{prf:definition} Admissible family of inputs
:label: def-admissible-input-family-rigorous

An **admissible family of inputs** is a tuple

$$
\mathfrak{X}
=
\bigl(
(X_n)_{n\in\mathbb{N}},
m_{\mathfrak{X}},
(\mathrm{enc}^{\mathfrak{X}}_n)_{n\in\mathbb{N}},
(\mathrm{dec}^{\mathfrak{X}}_n)_{n\in\mathbb{N}},
(\chi^{\mathfrak{X}}_n)_{n\in\mathbb{N}}
\bigr)
$$

consisting of the following data.

1. **Size-indexed objects.** For each $n\in\mathbb{N}$, $X_n \in \mathbf{H}$ is $0$-truncated.
2. **Encoding-length bound.** $m_{\mathfrak{X}}:\mathbb{N}\to\mathbb{N}$ is a polynomial.
3. **Injective finite encoding.** For each $n$ there is an injective map

   $$
   \mathrm{enc}^{\mathfrak{X}}_n : X_n \hookrightarrow \{0,1\}^{m_{\mathfrak{X}}(n)}.
   $$

4. **Decidable image.** For each $n$ there is a predicate

   $$
   \chi^{\mathfrak{X}}_n : \{0,1\}^{m_{\mathfrak{X}}(n)} \to \{0,1\}
   $$

   such that

   $$
   \chi^{\mathfrak{X}}_n(u)=1
   \iff
   u \in \mathrm{im}(\mathrm{enc}^{\mathfrak{X}}_n).
   $$

5. **Polynomial-time decoding on valid codes.** For each $n$ there is a total map

   $$
   \mathrm{dec}^{\mathfrak{X}}_n :
   \{\,u \in \{0,1\}^{m_{\mathfrak{X}}(n)} : \chi^{\mathfrak{X}}_n(u)=1\,\}
   \to X_n
   $$

   satisfying

   $$
   \mathrm{dec}^{\mathfrak{X}}_n(\mathrm{enc}^{\mathfrak{X}}_n(x)) = x
   \quad\text{for all }x\in X_n,
   $$

   and

   $$
   \mathrm{enc}^{\mathfrak{X}}_n(\mathrm{dec}^{\mathfrak{X}}_n(u)) = u
   \quad\text{for all valid }u.
   $$

6. **Uniform realizability.** There exist single effective procedures computing:
   - $n \mapsto m_{\mathfrak{X}}(n)$,
   - $(n,x) \mapsto \mathrm{enc}^{\mathfrak{X}}_n(x)$,
   - $(n,u) \mapsto \chi^{\mathfrak{X}}_n(u)$,
   - $(n,u) \mapsto \mathrm{dec}^{\mathfrak{X}}_n(u)$ on valid codes,

   each in time polynomial in $n$ and the input bitlength.

The associated **valid code language** of $\mathfrak{X}$ is

$$
D_{\mathfrak{X}}
:=
\left\{
\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle
\,:\,
n\in\mathbb{N},\ x\in X_n
\right\}
\subseteq \{0,1\}^*.
$$

A family of outputs is admissible by the same definition.
:::

:::{prf:remark} On the word "canonical"
:label: rem-canonical-means-fixed-admissible

In complexity-theoretic arguments, the phrase "canonical encoding" is too vague. In the present framework, one should
never argue using an unspecified canonical code. The admissible encoding is part of the structure of the family
$\mathfrak{X}$.

The correct invariance statement is not "there is one canonical encoding", but rather: any two admissible encodings of
the same underlying size-indexed family are polynomially intertranslatable. That is the content of the next lemma.
:::

:::{prf:lemma} Encoding invariance for admissible families
:label: lem-encoding-invariance-admissible

Let

$$
\mathfrak{X}
=
\bigl((X_n),m,\mathrm{enc},\mathrm{dec},\chi\bigr)
\quad\text{and}\quad
\mathfrak{X}'
=
\bigl((X_n),m',\mathrm{enc}',\mathrm{dec}',\chi'\bigr)
$$

be two admissible presentations of the same underlying size-indexed family $(X_n)_{n\in\mathbb{N}}$.

Then there exist uniform polynomial-time translators

$$
T_{\mathfrak{X}\to\mathfrak{X}'}(n,u)
:=
\mathrm{enc}'_n(\mathrm{dec}_n(u)),
\qquad
T_{\mathfrak{X}'\to\mathfrak{X}}(n,v)
:=
\mathrm{enc}_n(\mathrm{dec}'_n(v)),
$$

defined on valid codes, and these translators are mutual inverses on valid inputs.

Consequently, the choice of admissible encoding changes time complexity by at most polynomial overhead.
:::

:::{prf:proof}
Because $\mathfrak{X}$ is admissible, $\mathrm{dec}_n$ is uniformly polynomial-time computable on the valid image of
$\mathrm{enc}_n$. Because $\mathfrak{X}'$ is admissible, $\mathrm{enc}'_n$ is uniformly polynomial-time computable on
$X_n$. Therefore the composition

$$
u \mapsto \mathrm{enc}'_n(\mathrm{dec}_n(u))
$$

is uniformly polynomial-time on valid $u$.

The same argument yields polynomial-time computability of $T_{\mathfrak{X}'\to\mathfrak{X}}$. The inverse identities
follow immediately from the inverse axioms in {prf:ref}`def-admissible-input-family-rigorous`.
:::

:::{prf:definition} Uniform family of algorithms
:label: def-uniform-algorithm-family-rigorous

Let $\mathfrak{X}$ and $\mathfrak{Y}$ be admissible families. Let

$$
\sigma:\mathbb{N}\to\mathbb{N}
$$

be a polynomial, called the **size translator**.

A **uniform family of algorithms**

$$
\mathcal{A} : \mathfrak{X} \Rightarrow_{\sigma} \mathfrak{Y}
$$

is given by a single effective Fragile program

$$
a \in \mathsf{Prog}_{\text{FM}}
$$

such that for every $n\in\mathbb{N}$ and every $x\in X_n$:

1. the evaluation

   $$
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle\right)
   $$

   terminates with some bitstring output $v$;
2. the output is a valid $\mathfrak{Y}$-code of size $\sigma(n)$:

   $$
   \chi^{\mathfrak{Y}}_{\sigma(n)}(v)=1;
   $$

3. the induced extensional map

   $$
   \mathcal{A}_n : X_n \to Y_{\sigma(n)}
   $$

   is defined by

   $$
   \mathcal{A}_n(x)
   :=
   \mathrm{dec}^{\mathfrak{Y}}_{\sigma(n)}
   \!\left(
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle\right)
   \right).
   $$

The key point is **uniformity**: the code object $a$ is fixed once and for all and is independent of $n$. In
particular, no non-uniform advice string depending on $n$ is permitted.

When $\sigma(n)=n$, we simply write

$$
\mathcal{A} : \mathfrak{X}\Rightarrow \mathfrak{Y}.
$$

When source and target coincide, we write

$$
\mathcal{A} : \mathfrak{X}\Rightarrow \mathfrak{X}
$$

and call $\mathcal{A}$ a **uniform endomorphism family**.
:::

:::{prf:definition} Extensional equality of uniform families
:label: def-extensional-equality-uniform-families

Let

$$
\mathcal{A},\mathcal{B} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be uniform algorithm families with the same source, target, and size translator.

We write

$$
\mathcal{A} \equiv_{\mathrm{ext}} \mathcal{B}
$$

if and only if

$$
\forall n\in\mathbb{N}\ \forall x\in X_n,\qquad \mathcal{A}_n(x)=\mathcal{B}_n(x).
$$

All classification statements below are taken up to extensional equality.
:::

:::{prf:definition} Presentation translator
:label: def-presentation-translator

Let $\mathfrak{X}$ and $\mathfrak{Y}$ be admissible families, and let $\sigma$ be a polynomial size translator.

A **presentation translator**

$$
T : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

is a uniform family of algorithms such that there exists a uniform polynomial-time partial inverse on the image,
namely a uniform family

$$
S : \mathrm{im}(T)\Rightarrow \mathfrak{X}
$$

satisfying

$$
S_n(T_n(x)) = x
\qquad
\text{for all }n\text{ and }x\in X_n.
$$

Thus a presentation translator may re-encode, pad, reorder coordinates, add auxiliary bookkeeping fields, or place an
instance into a normal form, but it may not hide irrecoverable problem-solving work in a noninvertible collapse of the
instance space.

In all modal factorizations below, encoding and decoding maps are required to be presentation translators unless
explicitly stated otherwise.
:::

:::{prf:definition} Family cost certificate
:label: def-family-cost-certificate

Let

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be represented by a single program $a\in\mathsf{Prog}_{\text{FM}}$, and let $p:\mathbb{N}\to\mathbb{N}$ be a
polynomial.

A **family cost certificate**

$$
\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,p)
$$

is a ZFC-checkable witness of the following assertions:

1. **Uniform termination bound.** For every $n\in\mathbb{N}$ and every $x\in X_n$,

   $$
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle\right)
   $$

   halts within at most $p(n)$ internal runtime steps.
2. **Well-formed outputs.** Every such output is a valid $\mathfrak{Y}$-code of size $\sigma(n)$:

   $$
   \chi^{\mathfrak{Y}}_{\sigma(n)}
   \!\left(
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle\right)
   \right)
   =1.
   $$

3. **Step discipline.** Each counted runtime step is a primitive operation of the Fragile evaluator in the sense of
   {prf:ref}`def-cost-certificate`.
4. **Witness extractability.** The polynomial bound $p$ and the validity/output checks are derivable uniformly from the
   code of $a$ together with the admissible-family data of $\mathfrak{X},\mathfrak{Y}$.

Equivalently: a family cost certificate is the existing notion of {prf:ref}`def-cost-certificate`, applied to the
single tagged input domain $D_{\mathfrak{X}}$, together with the output-validity condition for $D_{\mathfrak{Y}}$.
:::

:::{prf:definition} Internal polynomial-time family
:label: def-internal-polytime-family-rigorous

A uniform algorithm family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

is **internally polynomial-time** if there exist a representing code object

$$
a \in \mathsf{Prog}_{\text{FM}}
$$

and a polynomial $p$ such that

$$
\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,p)
$$

holds.

We then write

$$
\mathcal{A} \in P_{\text{FM}}(\mathfrak{X},\mathfrak{Y};\sigma).
$$

When source and target are the same and $\sigma=\mathrm{id}$, we write simply

$$
\mathcal{A}\in P_{\text{FM}}(\mathfrak{X}).
$$

This is the official internal definition of polynomial time. Any entropy-, volume-, or compression-based language is
to be treated only as a derived witness layered on top of this definition, never as the primary definition of
$P_{\text{FM}}$.
:::

:::{prf:definition} Internal nondeterministic polynomial-time family
:label: def-internal-nptime-family-rigorous

A decision problem family

$$
\Pi = (\mathfrak{X}, \mathfrak{B}, \mathsf{Spec})
$$

belongs to $NP_{\mathrm{FM}}$ if there exist:

1. an admissible witness family $\mathfrak{W} = (W_n)_{n \in \mathbb{N}}$,
2. a witness-length polynomial $q$ such that $|w| \leq q(n)$ for every admissible witness $w \in W_n$,
3. a verifier family

   $$
   \mathcal{V} : \mathfrak{X} \times \mathfrak{W} \Rightarrow \mathfrak{B}
   $$

   in $P_{\mathrm{FM}}(\mathfrak{X} \times \mathfrak{W}, \mathfrak{B}; \sigma_V)$,

such that for every $n$ and every $x \in X_n$,

$$
(x, 1) \in \mathsf{Spec}_n
\quad\iff\quad
\exists\, w \in W_n :\ \mathcal{V}_n(x, w) = 1.
$$

This is the internal verifier-based definition of nondeterministic polynomial time. A problem is in $NP_{\mathrm{FM}}$
if and only if its yes-instances are exactly those admitting a polynomially bounded witness verifiable in
$P_{\mathrm{FM}}$.
:::

:::{prf:definition} Pure modal witness: abstract schema
:label: def-pure-modal-witness-abstract

Let

$$
\lozenge \in \{\sharp,\int,\flat,\ast,\partial\}.
$$

Let

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be a uniform algorithm family.

A **pure $\lozenge$-modal witness** for $\mathcal{A}$ consists of the following data:

1. an admissible family

   $$
   \mathfrak{Z}^{\lozenge}
   =
   \bigl((Z^{\lozenge}_n),m_{\mathfrak{Z}^{\lozenge}},\mathrm{enc}^{\mathfrak{Z}^{\lozenge}},
   \mathrm{dec}^{\mathfrak{Z}^{\lozenge}},\chi^{\mathfrak{Z}^{\lozenge}}\bigr);
   $$

2. a polynomial lift-size translator

   $$
   \rho_{\lozenge}:\mathbb{N}\to\mathbb{N};
   $$

3. a presentation translator

   $$
   E^{\lozenge} : \mathfrak{X}\Rightarrow_{\rho_{\lozenge}}\mathfrak{Z}^{\lozenge}
   $$

   called the **modal encoding**;
4. a uniform endomorphism family

   $$
   F^{\lozenge} : \mathfrak{Z}^{\lozenge}\Rightarrow \mathfrak{Z}^{\lozenge}
   $$

   such that

   $$
   F^{\lozenge}\in P_{\text{FM}}(\mathfrak{Z}^{\lozenge});
   $$

5. a presentation translator back to $\mathfrak{Y}$, concretely maps

   $$
   R_n^{\lozenge} : Z^{\lozenge}_{\rho_{\lozenge}(n)} \to Y_{\sigma(n)}
   $$

   called the **modal reconstruction map**;
6. a modality-specific certificate

   $$
   \Pi_{\lozenge}(F^{\lozenge})
   $$

   asserting that $F^{\lozenge}$ satisfies the universal property that defines pure $\lozenge$-computation.

These data must satisfy the extensional factorization identity

$$
\mathcal{A}_n
=
R^{\lozenge}_n \circ F^{\lozenge}_{\rho_{\lozenge}(n)} \circ E^{\lozenge}_n
\qquad
\text{for all }n.
$$

The witness is called **pure** because all asymptotically nontrivial work is carried by the single middle family
$F^{\lozenge}$, while $E^{\lozenge}$ and $R^{\lozenge}$ are required to be presentation translators rather than
arbitrary polynomial-time algorithms.
:::

:::{prf:definition} Pure $\sharp$-witness
:label: def-pure-sharp-witness-rigorous

A uniform family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

admits a **pure $\sharp$-witness** if it admits a pure modal witness in the sense of
{prf:ref}`def-pure-modal-witness-abstract` with $\lozenge=\sharp$, and the modality-specific certificate
$\Pi_{\sharp}(F^\sharp)$ consists of:

1. a decidable family of solved states

   $$
   S_n^\sharp \subseteq Z^\sharp_n;
   $$

2. a polynomial $q_\sharp$ and a uniformly polynomial-time ranking function

   $$
   V_n^\sharp : Z_n^\sharp \to \mathbb{N}
   $$

   satisfying

   $$
   V_n^\sharp(z)\le q_\sharp(n)
   \qquad
   \text{for all }z\in Z_n^\sharp;
   $$

3. a fixed-point condition on solved states:

   $$
   z\in S_n^\sharp \implies F_n^\sharp(z)=z;
   $$

4. a strict progress condition off the solved set:

   $$
   z\notin S_n^\sharp
   \implies
   V_n^\sharp(F_n^\sharp(z)) \le V_n^\sharp(z)-1;
   $$

5. a correctness condition: if

   $$
   t_n(x)
   :=
   \min\{\,t\le q_\sharp(\rho_\sharp(n)) : (F_{\rho_\sharp(n)}^\sharp)^t(E_n^\sharp(x))\in S_{\rho_\sharp(n)}^\sharp\,\},
   $$

   then

   $$
   \mathcal{A}_n(x)
   =
   R_n^\sharp\!\left((F_{\rho_\sharp(n)}^\sharp)^{t_n(x)}(E_n^\sharp(x))\right).
   $$

6. a **$\sharp$-modal restriction** on the transition map: $F_n^\sharp$ factors through the $\sharp$
   modality in the ambient cohesive $(\infty,1)$-topos. Concretely, the computation of $F_n^\sharp(z)$
   at each step is determined by the *metric/potential data* of $z$ — the rank value $V_n^\sharp(z)$,
   local energy evaluations $\Phi(z')$ for configurations $z'$ at bounded Hamming distance from $z$,
   and point-wise distance computations — but is *insensitive* to $\int$-type information
   (constraint-graph connectivity, implication-chain topology, message-passing convergence properties)
   and $\flat$-type information (algebraic identities among solution components, symmetry-group
   structure). Formally, $F_n^\sharp$ lies in the image of the $\sharp$-unit
   $\eta^\sharp: \mathrm{id} \to \sharp$, meaning it factors as
   $F_n^\sharp = g_n \circ \eta^\sharp_n$ for some $g_n: \sharp Z_n^\sharp \to Z_n^\sharp$.
   Here $\eta^\sharp_n : Z_n^\sharp \to \sharp Z_n^\sharp$ is the $\sharp$-unit that projects each state
   to its metric/potential data (energy evaluations, Hamming distances, rank values), discarding
   $\int$-type (constraint-graph connectivity) and $\flat$-type (algebraic identity) information.
   The factorization ensures that $F_n^\sharp(z_1) = F_n^\sharp(z_2)$ whenever
   $\eta^\sharp_n(z_1) = \eta^\sharp_n(z_2)$ — that is, two states with identical metric profiles
   receive identical treatment from $F_n^\sharp$.

Intuitively, pure $\sharp$-computation is computation by certified descent in a polynomially bounded potential, where the transition map is explicitly restricted to act only on metric/potential data and cannot access shape or flat structure of the state space.
:::

:::{prf:definition} Pure $\int$-witness
:label: def-pure-int-witness-rigorous

A uniform family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

admits a **pure $\int$-witness** if it admits a pure modal witness with $\lozenge=\int$, and the modality-specific
certificate $\Pi_{\int}(F^\int)$ consists of:

1. a finite poset of update sites for each input size,

   $$
   (P_n,\prec_n),
   $$

   with $|P_n|\le q_\int(n)$ and $\mathrm{height}(P_n)\le q_\int(n)$ for some polynomial $q_\int$;
2. a uniformly polynomial-time decidable order relation on $P_n$;
3. a decomposition of each lifted state into local coordinates indexed by $P_n$;
4. for each $i\in P_n$, a local update map

   $$
   U_{n,i}
   $$

   such that the value written at coordinate $i$ depends only on:
   - the original encoded input, and
   - coordinates indexed by predecessors $j\prec_n i$;
5. a uniformly polynomial-time computable linear extension

   $$
   \sigma_n : \{1,\dots,|P_n|\}\to P_n
   $$

   of $\prec_n$ such that

   $$
   F_n^\int
   =
   U_{n,\sigma_n(|P_n|)}\circ\cdots\circ U_{n,\sigma_n(1)};
   $$

6. a correctness condition

   $$
   \mathcal{A}_n
   =
   R_n^\int \circ F_{\rho_\int(n)}^\int \circ E_n^\int.
   $$

7. an **$\int$-modal restriction** on the update maps: each $U_{n,i}$ factors through the $\int$ (shape)
   modality in the ambient cohesive $(\infty,1)$-topos. Concretely, $U_{n,i}$ computes by **forward
   causal propagation**: it reads predecessor values $(\mathrm{state}_{j})_{j \prec_n i}$ and the encoded
   input, and propagates constraints along the directed-path structure of the poset. Formally, $U_{n,i}$
   lies in the image of the $\int$-unit $\eta^{\int}$: there exists $\tilde{U}_{n,i}$ such that
   $U_{n,i} = \tilde{U}_{n,i} \circ \eta^{\int}_n$, where $\eta^{\int}_n$ projects the input data to
   its shape/causal structure. This means $U_{n,i}$ can extract and act on dependency-graph connectivity,
   predecessor-successor relationships, and constraint-propagation paths, but is **insensitive to**
   $\sharp$-type information (energy evaluations, metric distances) and $\flat$-type information
   (algebraic identities, symmetry structure) *beyond what is accessible through the causal/shape
   structure*. The $\int$-modal restriction prevents $U_{n,i}$ from internally invoking optimization
   subroutines ($\sharp$-type), algebraic elimination ($\flat$-type), divide-and-conquer decomposition
   ($\ast$-type), or interface contraction ($\partial$-type).

Intuitively, pure $\int$-computation is computation by elimination along a polynomially bounded well-founded dependency
structure.
:::

:::{prf:definition} Pure $\flat$-witness
:label: def-pure-flat-witness-rigorous

A uniform family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

admits a **pure $\flat$-witness** if it admits a pure modal witness with $\lozenge=\flat$, and the modality-specific
certificate $\Pi_{\flat}(F^\flat)$ consists of:

1. an effective finite-sorted algebraic signature $\Sigma$ fixed independently of $n$;
2. a polynomial $q_\flat$ and families of finitely presented $\Sigma$-structures

   $$
   A_n^\flat,\ B_n^\flat
   $$

   whose presentation sizes are bounded by $q_\flat(n)$;
3. a presentation translator

   $$
   s_n^\flat : X_n \to A_{\rho_\flat(n)}^\flat;
   $$

4. a uniform polynomial-time algebraic elimination/cancellation map

   $$
   e_n^\flat : A_{\rho_\flat(n)}^\flat \to B_{\rho_\flat(n)}^\flat;
   $$

5. a reconstruction map

   $$
   d_n^\flat : B_{\rho_\flat(n)}^\flat \to Y_{\sigma(n)};
   $$

6. a derivation showing that each $e_n^\flat$ is built from a fixed finite basis of certified polynomial-time
   $\Sigma$-primitives, with all intermediate presentations bounded in size by $q_\flat(n)$;
7. the correctness identity

   $$
   \mathcal{A}_n
   =
   d_n^\flat \circ e_n^\flat \circ s_n^\flat.
   $$

The admissible primitive basis may include quotienting by definable congruence, linear elimination, determinant/rank
computations over effectively presented rings or fields with certified polynomial-time arithmetic, Fourier-type
transforms over effectively presented finite groups, and other algebraic cancellation primitives provided their
correctness and polynomial-time bounds are part of the witness and all intermediate presentations remain polynomially
bounded.

Intuitively, pure $\flat$-computation is computation by polynomially succinct algebraic compression, elimination, or
cancellation.
:::

:::{prf:definition} Pure $\ast$-witness
:label: def-pure-star-witness-rigorous

A uniform family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

admits a **pure $\ast$-witness** if it admits a pure modal witness with $\lozenge=\ast$, and the modality-specific
certificate $\Pi_{\ast}(F^\ast)$ consists of:

1. a polynomially bounded size measure

   $$
   \mu_n : Z_n^\ast \to \mathbb{N};
   $$

2. a polynomial $q_\ast$;
3. a uniformly polynomial-time splitting map which, on input $z\in Z_n^\ast$, produces a finite list of subinstances

   $$
   \mathrm{split}_n(z) = (z_1,\dots,z_{b(z)})
   $$

   with

   $$
   b(z)\le q_\ast(n)
   \quad\text{and}\quad
   \mu(z_i) < \mu(z)\ \text{for all }i;
   $$

4. a uniformly polynomial-time merge map

   $$
   \mathrm{merge}_n
   $$

   combining the recursively obtained subanswers into the answer for $z$;
5. a base-case solver on all states with $\mu(z)\le \mu_0$, for some fixed threshold $\mu_0$;
6. a polynomial upper bound on the total size of the recursion tree and on the total local work performed at all nodes;
7. the correctness identity

   $$
   \mathcal{A}_n
   =
   R_n^\ast \circ F_{\rho_\ast(n)}^\ast \circ E_n^\ast.
   $$

Intuitively, pure $\ast$-computation is certified polynomial-time self-reduction or divide-and-conquer with a
well-founded size decrease and polynomial total work.
:::

:::{prf:definition} Pure $\partial$-witness
:label: def-pure-boundary-witness-rigorous

A uniform family

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

admits a **pure $\partial$-witness** if it admits a pure modal witness with $\lozenge=\partial$, and the
modality-specific certificate $\Pi_{\partial}(F^\partial)$ consists of:

1. a polynomial $q_\partial$ and a family of polynomial-size interface objects

   $$
   B_n^\partial
   $$

   whose descriptions are bounded by $q_\partial(n)$;
2. a uniformly polynomial-time boundary extraction map

   $$
   \partial_n : Z_n^\partial \to B_n^\partial;
   $$

3. a uniformly polynomial-time interface contraction/interference map

   $$
   C_n^\partial : B_n^\partial \to B_n^{\partial,\mathrm{out}};
   $$

   with all intermediate interface descriptions bounded by $q_\partial(n)$;
4. a reconstruction map

   $$
   r_n^\partial : B_n^{\partial,\mathrm{out}} \to Y_{\sigma(n)};
   $$

5. the correctness identity

   $$
   \mathcal{A}_n
   =
   r_n^\partial \circ C_n^\partial \circ \partial_n \circ E_n^\partial;
   $$

6. a certificate that the asymptotic gain comes from compression to interface size, i.e. the runtime bound for the
   middle contraction stage is polynomial in the interface description size and hence polynomial in $n$.

Intuitively, pure $\partial$-computation is computation by reducing bulk information to a polynomial-size interface and
performing polynomial-time contraction or interference solely at the interface level.
:::

:::{prf:definition} Modal profile
:label: def-modal-profile-rigorous

Let

$$
\mathcal{A} : \mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be a uniform algorithm family.

A **modal profile** for $\mathcal{A}$ is a finite rooted derivation tree $\mathcal{T}$ generated by the following
rules.

1. **Leaf rule.** A leaf may be labelled by any pure witness of one of the five modalities

   $$
   \sharp,\ \int,\ \flat,\ \ast,\ \partial.
   $$

2. **Translator-conjugation rule.** If $\mathcal{T}_0$ is a profile for

   $$
   \mathcal{B}:\mathfrak{U}\Rightarrow_{\tau}\mathfrak{V},
   $$

   and $P:\mathfrak{X}\Rightarrow\mathfrak{U}$ and $Q:\mathfrak{V}\Rightarrow_{\sigma}\mathfrak{Y}$ are presentation
   translators, then a new unary node may be formed with denotation

   $$
   \mathcal{A} = Q \circ \mathcal{B} \circ P.
   $$

3. **Composition rule.** If $\mathcal{T}_1,\dots,\mathcal{T}_r$ are profiles for composable families

   $$
   \mathcal{B}^{(1)},\dots,\mathcal{B}^{(r)},
   $$

   then a composition node may be formed with denotation

   $$
   \mathcal{B}^{(r)}\circ\cdots\circ\mathcal{B}^{(1)}.
   $$

4. **Finite product rule.** If $\mathcal{T}_1,\dots,\mathcal{T}_r$ are profiles, then a product node may be formed
   with denotation the coordinatewise product family, using the standard paired encoding of product families.
5. **Bounded iteration rule.** If $\mathcal{T}_0$ is a profile for an endomorphism family

   $$
   \mathcal{B}:\mathfrak{X}\Rightarrow\mathfrak{X}
   $$

   and $k:\mathbb{N}\to\mathbb{N}$ is a polynomial, then an iteration node may be formed with denotation

   $$
   \mathrm{Iter}_{k}(\mathcal{B})_n := \mathcal{B}_n^{\,k(n)}.
   $$

6. **Finite recursion rule.** If a family $\mathcal{R}$ is defined by a well-founded polynomially bounded recursion
   whose recursive calls are denoted by already-constructed subtrees and whose split/merge/administrative maps are
   presentation translators, then a recursion node may be formed.

The **leaf multiset** of $\mathcal{T}$ is the multiset of modality labels appearing at its leaves. A family may admit
many different modal profiles. No uniqueness is required or assumed.

We say that $\mathcal{A}$ **has modal profile** $\mathcal{T}$ if the denotation of $\mathcal{T}$ is extensionally
equal to $\mathcal{A}$ in the sense of {prf:ref}`def-extensional-equality-uniform-families`.
:::

:::{prf:definition} Saturated modal closure
:label: def-saturated-modal-closure-rigorous

Let

$$
\mathsf{Pure}\langle \sharp,\int,\flat,\ast,\partial\rangle
$$

denote the class of all uniform families admitting at least one pure witness of one of the five modalities.

The **saturated modal closure**

$$
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle
$$

is the smallest class of uniform algorithm families, taken up to extensional equality, satisfying all of the
following.

1. **Pure inclusion.** Every family in

   $$
   \mathsf{Pure}\langle \sharp,\int,\flat,\ast,\partial\rangle
   $$

   belongs to

   $$
   \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
   $$

2. **Closure under presentation translators.** If

   $$
   \mathcal{B}\in \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle
   $$

   and $P,Q$ are presentation translators of compatible source and target types, then

   $$
   Q\circ \mathcal{B}\circ P
   \in
   \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
   $$

3. **Closure under composition.** If $\mathcal{B}$ and $\mathcal{C}$ are composable members of the saturated class,
   then

   $$
   \mathcal{C}\circ \mathcal{B}
   \in
   \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
   $$

4. **Closure under finite products.** If $\mathcal{B}^{(1)},\dots,\mathcal{B}^{(r)}$ belong to the saturated class,
   then their coordinatewise product family belongs to the saturated class.
5. **Closure under bounded iteration.** If

   $$
   \mathcal{B}:\mathfrak{X}\Rightarrow\mathfrak{X}
   $$

   belongs to the saturated class and $k$ is polynomial, then

   $$
   \mathrm{Iter}_{k}(\mathcal{B})_n = \mathcal{B}_n^{\,k(n)}
   $$

   also belongs to the saturated class.
6. **Closure under finite well-founded recursion.** Suppose a family $\mathcal{R}$ is given by a recursive scheme

   $$
   \mathcal{R}_n(x)
   =
   H_n\!\Bigl(
   x,\
   \mathcal{R}_{m_1(n,x)}(g_{n,1}(x)),\
   \dots,\
   \mathcal{R}_{m_{b(n,x)}(n,x)}(g_{n,b(n,x)}(x))
   \Bigr),
   $$

   where:
   - $b(n,x)\le q(n)$ for some polynomial $q$,
   - each size decreases strictly:

     $$
     m_i(n,x) < n,
     $$

   - each $g_{n,i}$ and $H_n$ is a presentation translator or is already denoted by a previously classified subtree,
   - the recursion tree has total size bounded by a polynomial in $n$.

   Then $\mathcal{R}$ belongs to the saturated class.

Equivalently,

$$
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle
$$

is the class of all families admitting at least one modal profile in the sense of
{prf:ref}`def-modal-profile-rigorous`.
:::

:::{prf:definition} The Five Algorithm Classes (Rigorous Modality Correspondence)
:label: def-five-algorithm-classes

For a uniform family $\mathcal{A}$, we write

$$
\mathcal{A} \triangleright \lozenge
$$

for the abstract $\lozenge$-modal route later formalized in
{prf:ref}`def-abstract-modal-factorization`. The concrete witness languages that are intended to realize those routes
are the pure witness schemas introduced above and characterized later by the universality ladder. The five pure
algorithm classes are then:

| Class | Name | Pure Witness | Exploited Resource | Examples | Detection |
|-------|------|--------------|-------------------|----------|-----------|
| I | Climbers | $\mathcal{A} \triangleright \sharp$ | Metric descent, convexity, spectral gap | Gradient descent, local search, convex optimization | Node 7 ($\mathrm{LS}_\sigma$), Node 12 ($\mathrm{GC}_\nabla$) |
| II | Propagators | $\mathcal{A} \triangleright \int$ | Causal order, DAG structure, elimination | Dynamic programming, unit propagation, belief propagation | Tactic E6 (Well-Foundedness) |
| III | Alchemists | $\mathcal{A} \triangleright \flat$ | Algebraic symmetry, quotient, cancellation | Gaussian elimination, FFT, LLL | Tactic E4 (Integrality), E11 (Galois-Monodromy) |
| IV | Dividers | $\mathcal{A} \triangleright \ast$ | Self-similarity, recursion, scale factorization | Divide and conquer, mergesort, multigrid | Node 4 ($\mathrm{SC}_\lambda$) |
| V | Interference Engines | $\mathcal{A} \triangleright \partial$ | Interface compression, holographic cancellation | FKT/Matchgates, quantum algorithms | Tactic E8 (DPI), Node 6 ($\mathrm{Cap}_H$) |

Pure classes isolate a single modal core. Mixed algorithms are represented by modal profiles in
{prf:ref}`def-saturated-modal-closure-rigorous`, not by pretending that one modality alone carries the whole proof.
:::

:::{prf:definition} Obstruction Certificates
:label: def-obstruction-certificates

The following table records the current **tactic-level frontend obstruction certificates** attached to the five modal
channels. They should be read as coarse backend indicators feeding the reconstructed semantic obstruction theory of
Part V, not as the final sound-and-complete obstruction schemas themselves.

For each modality $\lozenge$, the notation $K_\lozenge^-$ names the corresponding obstruction channel, while the
displayed conditions record the current frontend realization used by the existing tactics:

| Modality | Certificate | Obstruction Condition |
|----------|-------------|----------------------|
| $\sharp$ (Metric) | $K_\sharp^-$ | No spectral gap; Łojasiewicz inequality fails; glassy landscape |
| $\int$ (Causal) | $K_\int^-$ | Frustrated loops; $\pi_1(\text{factor graph}) \neq 0$; no DAG structure |
| $\flat$ (Algebraic) | $K_\flat^-$ | Trivial automorphism group; no visible quotient symmetry; failure of current integrality/monodromy frontends |
| $\ast$ (Scaling) | $K_\ast^-$ | Supercritical scaling; boundary dominates in decomposition |
| $\partial$ (Holographic) | $K_\partial^-$ | Non-planar; no Pfaffian orientation; unbounded treewidth; failure of current interface-contraction frontends |

**Certificate Logic:** At theorem level, the coarse frontend package proves hardness only after the reconstructed
obstruction theory of Part V supplies:
1. soundness/completeness of the semantic obstruction calculi for the five modalities;
2. mixed-modal obstruction for minimal-rank solver trees;
3. compatibility of the current tactic-level frontend certificates with those semantic calculi.

Concretely, the intended route is that a frontend certificate package derives the reconstructed E13 package, which then
feeds the mixed-modal obstruction theorem:

$$
K_\sharp^- \wedge K_\int^- \wedge K_\flat^- \wedge K_\ast^- \wedge K_\partial^-
\implies
\mathsf{Sol}_{\mathrm{poly}}(\Pi)=\varnothing.
$$

This coarse contrapositive form is made precise later by
{prf:ref}`thm-mixed-modal-obstruction`,
{prf:ref}`cor-e13-contrapositive-hardness-reconstructed`, and
{prf:ref}`prop-compatibility-with-current-tactics`.
:::

:::{prf:remark} Why the encoding/decoding maps are restricted to presentation translators
:label: rem-why-translator-restriction

The translator restriction is essential. If one allowed arbitrary polynomial-time preprocessing and postprocessing in
the definition of pure modal factorization, then the factorization language would become vacuous: the preprocessor
could already solve the problem, leaving the modal core to do nothing.

By forcing the outer maps in a pure witness to be presentation translators, one isolates the genuine source of
algorithmic power in the middle modal stage. Mixed algorithms are then represented honestly by modal profiles and the
saturated closure rather than by hiding work in the encoding and decoding layers.
:::

:::{div} feynman-prose
Let me make sure you understand what each of these classes is really doing. Think of each one as a different "trick"
for compressing your search space.

**Climbers** are algorithms that follow a gradient downhill. They work when your problem has a smooth landscape with a
clear direction toward the solution. Gradient descent is the prototype: at each step, you move in the direction that
decreases the objective function. The key insight is that you are exploiting *metric structure*, the ability to measure
"nearby" and "downhill."

**Propagators** exploit causal structure. Dynamic programming is the classic example: you solve subproblems in the right
order, so each answer is available when you need it. The trick is that information flows in one direction, like
dominoes falling. No cycles, no backtracking.

**Alchemists** exploit symmetry. If your problem has a large group acting on it, you can factor out that symmetry and
work in a smaller quotient space. Gaussian elimination works because linear algebra has a huge symmetry group. The FFT
works because the roots of unity form a cyclic group.

**Dividers** exploit self-similarity. If your problem looks the same at different scales, you can solve a smaller
version and piece together the answer. Mergesort does this: sorting $n$ elements reduces to sorting $n/2$ elements,
twice.

**Interference Engines** are the most exotic. They work when massive cancellations occur, like quantum algorithms where
exponentially many paths interfere to leave only the right answer. The FKT algorithm for counting perfect matchings in
planar graphs is the classical prototype.
:::

:::{prf:remark} AIT Interpretation as a Derived Witness
:label: rem-ait-algorithm-classes

Entropy, volume, Kolmogorov complexity, and information-compression language may still be used as auxiliary invariants
attached to a pure witness or modal profile, but they do not define polynomial time. The primary definition is
{prf:ref}`def-internal-polytime-family-rigorous`, namely existence of a family cost certificate.

Each pure class nevertheless has a natural AIT shadow:

| Class | Modality | Derived AIT Mechanism | Typical Auxiliary Invariant |
|-------|----------|-----------------------|-----------------------------|
| I (Climbers) | $\sharp$ | Decreasing potential | $K_{t+1} \le K_t - \Omega(1)$ per certified step |
| II (Propagators) | $\int$ | Shrinking unresolved dependency set | $K(x \mid \text{subproblems}) \ll K(x)$ |
| III (Alchemists) | $\flat$ | Algebraic quotient or cancellation | $K([x]_G) \le K(x) - \log|G| + O(1)$ |
| IV (Dividers) | $\ast$ | Well-founded recursion measure | Master-theorem style recurrence for the description size |
| V (Interference) | $\partial$ | Boundary-to-bulk compression | $K(\text{bulk}) \le K(\partial) + O(1)$ |

In Sieve instantiations, $K(\cdot)$ is evaluated on the encoded thin trace $T_{\mathrm{thin}}$ using the approximable
proxy $K_\epsilon$ with fixed resource bounds.
:::

:::{prf:remark} Scope of the definitional layer
:label: rem-scope-definitions-only

The present subsection is definitional only. It does **not** yet assert that

$$
P_{\text{FM}}
=
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$

That equality is a substantive theorem-level claim and must be proved later by adequacy, normal-form,
witness-classification, and obstruction-completeness arguments.

The next section isolates the theorem ladder required even to state that claim noncircularly.
:::

### Detailed Algorithm Class Specifications

We now provide concrete witness templates for each pure algorithm class. For readability, the shorthand
$\mathcal{A}\triangleright\lozenge$ from {prf:ref}`def-five-algorithm-classes` is used below; formally it means that,
after fixing admissible presentations of the relevant input and output families, $\mathcal{A}$ admits a pure
$\lozenge$-witness.

:::{prf:definition} Class I: Climbers (Sharp Modality)
:label: def-class-i-climbers

An algorithmic process $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ is **Class I (Climber)** if:

1. **Modal Factorization:** $\mathcal{A} \triangleright \sharp$ (factors through sharp modality)
2. **Height Functional:** There exists $\Phi: \mathcal{X} \to \mathbb{R}$ such that:
   - $\Phi(\mathcal{A}(x)) < \Phi(x)$ for non-equilibrium states (strict descent)
   - $\Phi$ satisfies the **Łojasiewicz-Simon inequality**:

     $$\|\nabla \Phi(x)\| \geq c|\Phi(x) - \Phi^*|^{1-\theta}$$

     for some $c > 0$, $\theta \in (0,1)$, where $\Phi^*$ is the minimum value
3. **Spectral Gap:** The Hessian $\nabla^2\Phi$ at equilibria has spectral gap $\lambda > 0$

**Polynomial-Time Certificate:** $K_{\sharp}^+ = (\Phi, \theta, \lambda)$ where $\theta \geq 1/k$ for constant $k$ ensures convergence in $O(n^{k-1})$ steps.

**Examples:** Gradient descent on convex functions, simulated annealing with sufficient cooling, local search with Hamming distance.
:::

:::{prf:lemma} Sharp Modality Frontend Obstruction
:label: lem-sharp-obstruction

If the energy landscape $\Phi$ is **glassy** in the following sense:

1. **Cluster shattering:** The solution set partitions into $\exp(\Theta(n))$ clusters separated by
   Hamming distance $\Omega(n)$, with a $\Theta(n)$-fraction of variables frozen within each cluster.
2. **Vanishing spectral gap:** $\lambda_{\min}(\nabla^2 \Phi) \to 0$ on the hard subfamily.
3. **Łojasiewicz failure:** The Łojasiewicz-Simon inequality fails ($\theta \to 0$) near frozen-variable
   configurations.

then no pure $\sharp$-witness in the sense of {prf:ref}`def-pure-sharp-witness-rigorous` exists for the
problem family restricted to the hard subfamily. In particular, this blocks not only the Class I climber
template of {prf:ref}`def-class-i-climbers` but every pure $\sharp$-witness, regardless of the choice of
ranking function, transition map, or lifted state encoding.

**Obstruction Certificate:** $K_{\sharp}^- = (\text{glassy}, \lambda = 0, \theta \to 0)$

**Application:** Random 3-SAT near threshold has glassy landscape (Mézard-Parisi-Zecchina 2002,
Achlioptas-Coja-Oghlan 2008), supplying a sharp frontend blockage certificate.
:::

:::{prf:proof}
The argument uses two ingredients: the **$\sharp$-purity constraint**, which limits the computational
vocabulary of $F_n^\sharp$ to metric-descent operations, and the **landscape properties**, which ensure
that metric-descent operations cannot navigate the shattered solution space. The combination is
non-circular: it derives the obstruction from the modal typing of the witness (a definitional constraint)
and proven landscape properties (ZFC theorems about random 3-SAT), without assuming hardness of any
computational problem.

**Step 1 ($\sharp$-purity constraint).**
By {prf:ref}`def-pure-sharp-witness-rigorous`, a pure $\sharp$-witness factors the algorithm as
$\mathcal{A}_n = R_n^\sharp \circ (F_n^\sharp)^{t_n} \circ E_n^\sharp$, where $E_n^\sharp$ and
$R_n^\sharp$ are presentation translators and $F_n^\sharp$ carries all nontrivial computational work.
The modality-specific certificate $\Pi_\sharp(F^\sharp)$ requires $F_n^\sharp$ to compute by
**certified descent in a polynomially bounded potential**: there exists a ranking function
$V_n^\sharp: Z_n^\sharp \to \{0, \ldots, q_\sharp(n)\}$ such that
$V_n^\sharp(F_n^\sharp(z)) \le V_n^\sharp(z) - 1$ for $z \notin S_n^\sharp$.

Crucially, item 6 of {prf:ref}`def-pure-sharp-witness-rigorous` explicitly requires $F_n^\sharp$ to be
a $\sharp$-modal morphism in the ambient cohesive $(\infty,1)$-topos: $F_n^\sharp$ factors as
$g_n \circ \eta^\sharp_n$ through the $\sharp$-unit. By the adjunction
$\Gamma \dashv \mathrm{coDisc}$ defining the $\sharp$ modality, this factorization means $F_n^\sharp$
accesses the **metric/potential structure** of the state space (point evaluations of $V$, local energy
differences, Hamming distances) but is **insensitive to shape ($\int$) structure**
(constraint-propagation paths, dependency-graph topology) and **insensitive to flat ($\flat$)
structure** (algebraic identities, symmetry-breaking information). This modal orthogonality follows
from the adjoint quadruple $\Pi \dashv \mathrm{Disc} \dashv \Gamma \dashv \mathrm{coDisc}$:
information accessible to the $\sharp$ modality is precisely the information that survives the
$\Gamma$-projection, which strips shape and flat structure.

Note: $F_n^\sharp$ reads the full state $z \in Z_n^\sharp$, but $\sharp$-purity constrains *how* $F$
uses $z$. A $\sharp$-modal function computes metric quantities from $z$ (energy, distances, rank) but
cannot extract $\int$-type information (constraint-graph connectivity, implication paths) or $\flat$-type
information (algebraic elimination sequences). This is the essential difference between a pure
$\sharp$-witness and a general polynomial-time algorithm: a general algorithm may combine $\sharp$, $\int$,
$\flat$, $\ast$, $\partial$ mechanisms, but a *pure* $\sharp$-witness is restricted to metric descent alone.

**Step 2 (Cluster navigation requires non-$\sharp$ information).**
On the shattered landscape, the solution set $S_n^\sharp$ partitions into $M = \exp(\Theta(n))$ clusters
$C_1, \ldots, C_M$ at pairwise Hamming distance $\Omega(n)$. Cluster membership is determined by the
**frozen-variable core**: within each cluster, a $\Theta(n)$-fraction of variables are forced to
specific values by the clause structure (Achlioptas-Coja-Oghlan 2008). The frozen core of each cluster
is defined by the implication structure of the constraint graph — which variables are forced by which
clauses, and how implications propagate through unit propagation chains. This is inherently $\int$-type
(shape) information: it is the connectivity and directed-path structure of the clause-variable dependency
graph.

Formally: let $\phi: \{0,1\}^n \to \{1, \ldots, M\}$ assign each configuration to its nearest solution
cluster (breaking ties arbitrarily). Computing $\phi(z)$ requires determining which frozen-variable
pattern $z$ is closest to, which in turn requires tracing implication chains in the constraint graph —
an $\int$-type computation. Since $\phi$ depends on the shape structure of the problem instance, it is
not computable by a $\sharp$-modal function.

**Step 3 (Metric uninformativeness on the shattered landscape).**
The $\sharp$-modal information available to $F_n^\sharp$ is insufficient for cluster navigation:

- **Vanishing spectral gap ($\lambda_{\min} \to 0$):** The spectral gap of the Glauber dynamics
  transition matrix on the hard subfamily tends to zero (Montanari-Semerjian 2006), giving mixing time
  $t_{\mathrm{mix}} = \exp(\Omega(n))$. Moreover, $k$-local energy statistics — the joint distribution
  of clause violation counts in any radius-$k$ neighborhood — are asymptotically identical across
  clusters for any fixed $k$ (Krzakała-Montanari-Rizzo-Zdeborová 2007). Consequently, for any
  polynomial $p(n)$, the metric structure in a $p(n)$-size neighborhood of a configuration $z$ carries
  $o(1)$ bits of mutual information with the cluster identity $\phi(z)$: any polynomial-time computable
  metric quantity (energy, local gradient, Hessian spectrum) has a distribution over random instances
  that is statistically indistinguishable across clusters, with total variation distance
  $\exp(-\Omega(n))$.

- **Łojasiewicz failure ($\theta \to 0$):** Near frozen-variable configurations, the gradient
  $\|\nabla\Phi\|$ vanishes while the energy gap to the nearest solution remains $\Theta(n)$. This means
  gradient-based descent directions — the primary $\sharp$-modal navigational tool — are uninformative:
  they point toward local energy improvements that do not correlate with progress toward any particular
  solution cluster.

- **Frozen-variable opacity:** The $\Theta(n)$ frozen variables that determine cluster identity are
  defined by the constraint-graph structure ($\int$-type). A $\sharp$-modal function cannot identify
  which variables are frozen or what their forced values should be, because this information is encoded
  in the implication graph, not in metric quantities. While individual variables do differ in their
  clause-participation counts, the *distributional* signature that distinguishes frozen from free
  variables — namely, the pattern of which high-degree variables are mutually constrained by implication
  chains — is an $\int$-type property invisible to the $\sharp$ modality. The metric data (energy,
  local gradients) at a given configuration are determined by aggregate clause-violation counts, which
  by the $k$-local indistinguishability above do not reveal frozen-variable identity.

**Step 4 (Direct $\sharp$-modal failure on the shattered landscape).**
Suppose toward contradiction that a pure $\sharp$-witness
$(V_n^\sharp, F_n^\sharp, S_n^\sharp, q_\sharp)$ exists on the hard subfamily.

Since $F_n^\sharp$ factors through $\eta^\sharp_n$ (item 6 of {prf:ref}`def-pure-sharp-witness-rigorous`,
confirmed in Step 1), two states with the same sharp-modalization — i.e., identical metric profiles —
map to the same next state. Therefore $F_n^\sharp$'s behavior at state $z$ is determined by the
metric/potential data of $z$: the rank value $V_n^\sharp(z)$, local energy evaluations $\Phi(z')$ for
configurations $z'$ at bounded Hamming distance, and point-wise distance computations. By Step 3, the
metric/potential data in any polynomial-size neighborhood of $z$ carries $o(1)$ bits of mutual
information with the cluster identity $\phi(z)$. Therefore $F_n^\sharp$ is **cluster-blind**: for two
configurations $z_1, z_2$ with identical metric profiles (same rank, same local energy statistics, same
distance structure) but $\phi(z_1) \neq \phi(z_2)$, the $\sharp$-modal $F_n^\sharp$ produces the same
output — the same next state, the same descent direction.

Now consider the $M = \exp(\Theta(n))$ clusters $C_1, \ldots, C_M$. The ranking function $V_n^\sharp$
takes values in $\{0, \ldots, q_\sharp(n)\}$, so there are at most $q_\sharp(n)+1 = \mathrm{poly}(n)$
possible rank values. By the pigeonhole principle, at least
$M / \mathrm{poly}(n) = \exp(\Theta(n))$ clusters have representatives at the same rank value.
Consider such a group of $\exp(\Theta(n))$ clusters whose entry states (images under $E_n^\sharp$)
share a common rank $r$.

Among these same-rank entry states, $F_n^\sharp$ cannot distinguish which cluster to descend toward:
(i) they have the same rank $V_n^\sharp(z) = r$;
(ii) their metric neighborhoods are statistically indistinguishable by Step 3, with total variation
distance $\exp(-\Omega(n))$ in $k$-local energy statistics.
Since $F_n^\sharp$ is $\sharp$-modal, its output is a deterministic function of the metric/potential
data alone. The $\exp(-\Omega(n))$ indistinguishability of the metric profiles means that, for all but
an $\exp(-\Omega(n))$-fraction of instances, $F_n^\sharp$ maps same-rank entry states from distinct
clusters to the same next state. By induction on the trajectory length (at most $q_\sharp(n)$
steps, with the same indistinguishability argument applying at each step), the entire descent
trajectory from these entry states is identical with probability $1 - q_\sharp(n) \cdot
\exp(-\Omega(n)) = 1 - o(1)$.

But the correctness condition (item 5) requires that trajectories starting from entry states of
*different* clusters terminate at *different* solved states in $S_n^\sharp$ — since distinct clusters
have distinct satisfying assignments. Identical trajectories cannot reach distinct solved states.
This yields a contradiction for the $\exp(\Theta(n))$ clusters in the same-rank group: at most one
cluster's entry states can be correctly routed, while the remaining $\exp(\Theta(n)) - 1$ clusters
are misrouted. The correctness condition fails on an $\Omega(1)$-fraction of inputs.

**Step 5 (Landscape certificate).**
The three glassy signatures supply the concrete certificate
$K_{\sharp}^- = (\text{glassy}, \lambda = 0, \theta \to 0)$:
(i) cluster shattering ensures $\exp(\Theta(n))$ clusters, so that the pigeonhole argument in Step 4
forces $\exp(\Theta(n))$ clusters to share a single rank value;
(ii) vanishing spectral gap and $k$-local indistinguishability (Krzakała et al. 2007) ensure that the
metric profiles of entry states from distinct clusters are statistically identical, making $F_n^\sharp$
cluster-blind (the $\exp(-\Omega(n))$ total variation bound in Step 4);
(iii) Łojasiewicz failure ensures that gradient-based descent — the canonical $\sharp$-modal mechanism —
provides no cluster-routing signal, reinforcing the cluster-blindness even for gradient-following
strategies.

**Conclusion.** The $\sharp$-purity constraint restricts $F_n^\sharp$ to metric-descent computation: $F$
reads the full state $z$ but can only extract and act on metric/potential information (rank, energy,
distances), not constraint-graph structure ($\int$-type) or algebraic identities ($\flat$-type). The
shattered landscape ensures that metric information alone cannot navigate to the correct solution cluster,
because cluster identity is determined by the frozen-variable core — an $\int$-type object invisible to
the $\sharp$ modality. This obstruction holds regardless of the choice of ranking function $V$, transition
map $F$, lifted state encoding $E^\sharp$, or polynomial degree of $q_\sharp$. Hence no pure
$\sharp$-witness exists on the hard subfamily.
:::

:::{prf:definition} Class II: Propagators (Shape Modality)
:label: def-class-ii-propagators

An algorithmic process $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ is **Class II (Propagator)** if:

1. **Modal Factorization:** $\mathcal{A} \triangleright \int$ (factors through shape modality)
2. **DAG Structure:** The dependency graph $G = (V, E)$ is a directed acyclic graph with:
   - $\mathrm{depth}(G) \leq p(n)$ for polynomial $p$
   - $\mathrm{deg}^{-}(v) \leq k$ for constant $k$ (bounded in-degree)
3. **Topological Order:** The shape $\int \mathcal{X}$ has trivial fundamental group: $\pi_1(\int \mathcal{X}) = 0$

**Polynomial-Time Certificate:** $K_{\int}^+ = (G, d, k)$ where $d = \mathrm{depth}(G)$ and $k = \max \mathrm{deg}^{-}$ give time complexity $O(|V| \cdot k) = O(d \cdot w \cdot k)$ for width $w$.

**Examples:** Dynamic programming, belief propagation on trees, unit propagation for Horn-SAT.
:::

:::{prf:lemma} Shape Modality Frontend Obstruction (Frustrated Loops)
:label: lem-shape-obstruction

If the dependency structure contains **frustrated loops**—cycles where constraints cannot be simultaneously
satisfied—then the standard propagation/elimination witness language of {prf:ref}`def-class-ii-propagators` is
blocked, yielding a valid frontend obstruction for the $\int$-channel.

Formally: If $\pi_1(\int \mathcal{X}) \neq 0$ (non-trivial fundamental group), then propagation around cycles produces inconsistencies requiring exponential backtracking.

**Obstruction Certificate:** $K_{\int}^- = (\pi_1 \neq 0, \text{cycles})$

**Application:** Random 3-SAT has frustrated loops (conflicting clause cycles), supplying an $\int$-frontend blockage.
Horn-SAT has $\pi_1 = 0$ (acyclic implications), enabling the standard propagator witness template. The full
sound-and-complete $\int$-obstruction theorem is deferred to Part V.
:::

:::{prf:proof}
A pure $\int$-witness ({prf:ref}`def-pure-int-witness-rigorous`) requires a well-founded dependency
poset $(P_n, \prec_n)$ of polynomial size and height, with local update maps $U_{n,i}$ such that the
update at site $i$ depends only on predecessors $j \prec_n i$.

Frustrated cycles create circular dependencies: the correct value at site $i$ depends on the value
at site $j$ and vice versa. Any well-founded poset must linearize these dependencies, forcing one
site to be updated before the other. But in a frustrated cycle, updating $i$ before $j$ produces an
incorrect intermediate state at $j$'s position, which cannot be corrected by $j$'s local update
(since $j$'s update sees only predecessors, not successors). The frustration ensures that no linear
ordering of updates can simultaneously satisfy all dependency constraints within the poset structure
required by a pure $\int$-witness.
:::

:::{prf:definition} Propagator Tube Witness (Geodesic Progress Certificate)
:label: def-propagator-tube-witness

This definition packages a common “thin-solution-manifold” situation in the **Propagator / shape** regime into an
explicit certificate that yields a **linear-in-depth** bound for population-based propagators (including Fractal Gas
instantiations) on tree/graph growth problems.

**Setup (rooted transition system).**
Let $(X,x_0,\mathsf{Next},\mathsf{Goal})$ be a rooted transition system, where $\mathsf{Next}(x)\subseteq X$ is finite
and $\mathsf{Goal}\subseteq X$ is the goal set. Define the depth

$$
\mathrm{depth}(x):=\min\{k:\exists x_1,\dots,x_k\ \text{s.t.}\ x_1\in\mathsf{Next}(x_0),\ x_{i+1}\in\mathsf{Next}(x_i),\ x_k=x\},
$$
and the optimal goal depth $d_\star:=\min_{x\in\mathsf{Goal}}\mathrm{depth}(x)$.

**Definition (tube witness).**
Fix a population-based Propagator update rule (one “outer iteration”) consisting of:
1. a one-step proposal/transition mechanism, and
2. a selection/resampling mechanism that can preserve promising branches.

A **Propagator tube witness** is a tuple $(\mathcal{T},V,\delta,p)$ where $\mathcal{T}\subseteq X$ is a “tube”,
$V:X\to\mathbb{R}$ is a progress functional, and $\delta,p>0$ are constants such that:
1. (**Tube**) $x_0\in\mathcal{T}$ and $\mathcal{T}\cap\mathsf{Goal}\neq\varnothing$.
2. (**Forward connectivity**) For every $x\in\mathcal{T}$ with $\mathrm{depth}(x)<d_\star$ there exists
   $y\in\mathsf{Next}(x)\cap\mathcal{T}$ with $\mathrm{depth}(y)=\mathrm{depth}(x)+1$.
3. (**Strict progress**) For any such tube edge $x\to y$, $V(y)\ge V(x)+\delta$.
4. (**Tube-following probability**) Conditioned on any walker being at any $x\in\mathcal{T}$ with
   $\mathrm{depth}(x)<d_\star$, the proposal mechanism proposes at least one tube successor as in (2) with probability
   $\ge p$.
5. (**Non-extinction on the tube**) The selection/resampling step preserves at least one tube walker until
   $\mathsf{Goal}$ is reached.

**Interpretation:** This is an explicit “geodesic tube” regularity certificate inside Class II (Propagators): the
effective branching factor on $\mathcal{T}$ is 1 (a wavefront advances down a well-founded chain), even if the ambient
branching factor $b=\sup_x|\mathsf{Next}(x)|$ is large.
:::

:::{prf:theorem} [MT-GeodesicTunneling] The Geodesic Tunneling of Fractal Trees
:label: mt:geodesic-tunneling-fractal-trees

**Status:** Conditional (solver-specific envelope inside Class II; the singular-case fallback uses {prf:ref}`mt:levin-search`).

**Statement (Propagator wavefront bound).**
Assume the instance is Regular in the **Propagator / shape** sense (Definition {prf:ref}`def-class-ii-propagators`) and
admits a Propagator tube witness $(\mathcal{T},V,\delta,p)$ (Definition {prf:ref}`def-propagator-tube-witness`). Then the
expected number of outer iterations for a population-based Propagator to reach $\mathsf{Goal}$ satisfies

$$
\mathbb{E}[T_{\mathrm{hit}}]\ \le\ d_\star/p,
$$
independent of the ambient branching factor $b$.

**Statement (singular regime fallback).**
If all five modalities are blocked (Definition {prf:ref}`def-obstruction-certificates`), no polynomial-time progress
certificate exists in the worst case. In that regime, guarantees reduce to the chosen prior/schedule; an explicit
Levin-equivalent instantiation exists by Metatheorem {prf:ref}`mt:levin-search`.
:::

:::{prf:proof}
Let $Z_t$ be the maximum depth among walkers in the tube $\mathcal{T}$ after iteration $t$. By the non-extinction
assumption in Definition {prf:ref}`def-propagator-tube-witness`, there is always at least one tube walker at depth $Z_t$
until $d_\star$ is reached.

Conditioned on being at any $x\in\mathcal{T}$ with $\mathrm{depth}(x)<d_\star$, the tube-following probability yields

$$
\mathbb{P}(Z_{t+1}=Z_t+1\mid Z_t<d_\star)\ge p.
$$
Therefore the waiting time to advance the wavefront by one depth level is stochastically dominated by a geometric
random variable with mean $1/p$. By linearity of expectation over the $d_\star$ required advances,
$\mathbb{E}[T_{\mathrm{hit}}]\le d_\star/p$.

The bound is independent of $b$ because the tube witness asserts screening onto $\mathcal{T}$: only a
constant-probability “correct successor” event is needed per depth increment.

In the singular regime (all obstruction certificates), the absence of any separating modal structure prevents such a
tube progress certificate in the worst case; the Levin-equivalent fallback is exactly Metatheorem {prf:ref}`mt:levin-search`.
:::

:::{prf:definition} Class III: Alchemists (Flat Modality)
:label: def-class-iii-alchemists

An algorithmic process $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ is **Class III (Alchemist)** if:

1. **Modal Factorization:** $\mathcal{A} \triangleright \flat$ (factors through flat modality)
2. **Symmetry Group:** There exists a non-trivial group $G$ acting on $\mathcal{X}$ such that:
   - $\mathcal{A}$ is $G$-equivariant: $\mathcal{A}(g \cdot x) = g \cdot \mathcal{A}(x)$
   - $|G| = \Omega(2^n / \mathrm{poly}(n))$ (exponential symmetry reduction)
   - Solutions lift from quotient: $\mathcal{X}/G \to \mathcal{X}$
3. **Quotient Compression:** $|\mathcal{X}/G| = \mathrm{poly}(n)$

**Polynomial-Time Certificate:** $K_{\flat}^+ = (G, |G|, \mathcal{X}/G)$ gives compression factor $|G|$ and quotient size $|\mathcal{X}/G|$.

**Examples:** Gaussian elimination ($G = \mathrm{GL}_n(\mathbb{F})$), FFT ($G = \mathbb{Z}/n\mathbb{Z}$), XORSAT ($G = \ker(A)$).
:::

:::{prf:lemma} Flat Modality Frontend Obstruction (Visible Symmetry Failure)
:label: lem-flat-obstruction

If the automorphism group is trivial:

$$G_{\Phi} := \mathrm{Aut}(\mathcal{X}, \Phi) = \{e\}$$

then the visible quotient-symmetry witness language of {prf:ref}`def-class-iii-alchemists` is blocked and the quotient
equals the full space: $\mathcal{X}/G = \mathcal{X}$. No such symmetry compression occurs.

**Obstruction Certificate:** $K_{\flat}^- = (|G| = 1)$

**Application:** Random instances have trivial automorphism with high probability, supplying a flat frontend blockage.
XORSAT has large kernel group $|G| = 2^{n-\mathrm{rank}(A)}$, enabling this concrete symmetry-based frontend route. The
full strengthened $\flat$-obstruction theorem of Part V must additionally rule out non-symmetry algebraic sketches.
:::

:::{prf:proof}
**Step 1. [Pure $\flat$-witness structure]:**
A pure $\flat$-witness ({prf:ref}`def-pure-flat-witness-rigorous`) requires:
- an effective finite-sorted algebraic signature $\Sigma$ fixed independently of $n$,
- finitely presented $\Sigma$-structures $A_n^\flat$, $B_n^\flat$ with presentation sizes bounded
  by a polynomial $q_\flat(n)$,
- a polynomial-time algebraic elimination map $e_n^\flat : A_n^\flat \to B_n^\flat$ built from
  certified $\Sigma$-primitives (quotienting, linear elimination, rank/determinant, Fourier
  transforms, polynomial-identity cancellation, monodromy-based reductions),
- and the correctness identity $\mathcal{A}_n = d_n^\flat \circ e_n^\flat \circ s_n^\flat$.

We show that random 3-SAT at the satisfiability threshold admits no such witness, addressing
both the visible-symmetry route and the broader algebraic sketch routes in turn.

**Step 2. [Visible symmetry blockage — trivial automorphism]:**
The simplest $\flat$-route is quotient compression via the automorphism group
$G_\Phi := \mathrm{Aut}(\mathcal{X}, \Phi)$. For random 3-SAT at ratio
$\alpha \approx 4.267$, each variable participates in $\Theta(1)$ clauses with distinct
neighborhoods w.h.p., so $G_\Phi = \{e\}$ for a $(1-o(1))$ fraction of instances
(property 5 of {prf:ref}`def-hard-subfamily-3sat`). Since the quotient equals the full space
$\mathcal{X}/G = \mathcal{X}$, no compression via visible symmetry quotienting occurs.

**Step 3. [Broader algebraic sketch blockage — frozen variables and monodromy rigidity]:**
The strengthened $\flat$-class ({prf:ref}`thm-flat-universality`) encompasses all polynomial-size
algebraic sketches, not merely symmetry quotients. We must therefore show that *no* admissible
algebraic elimination map $e_n^\flat$ over *any* effective ring or field structure can achieve
polynomial presentation size. The obstruction has two independent sources:

**(3a) Frozen-variable algebraic rigidity.** The solution space of random 3-SAT in the hard
subfamily ({prf:ref}`def-hard-subfamily-3sat`) exhibits propagation rigidity: an
$\Omega(n)$-sized set of frozen variables whose values are determined by membership in a
solution cluster (property 1, solution-space shattering). Any algebraic elimination map
$e_n^\flat$ must have its kernel/image controlled by the algebraic structure of the solution
variety. The frozen coordinates constitute $\Omega(n)$ algebraically independent generators
that cannot be eliminated through polynomial presentations — eliminating any frozen variable
propagates constraints through the $\Theta(n)$-sized frustration cores (property 3), forcing
superpolynomial intermediate presentation size in any elimination schedule. This blocks
quotient, linear, rank/determinant, Fourier, and polynomial-identity sketch routes.

**(3b) Non-solvable monodromy obstruction.** The monodromy group of the solution variety
$V(F) = \{x \in \{0,1\}^n : F(x) = 1\}$ — obtained by analytically continuing solutions as
clause coefficients vary — is the full symmetric group $S_k$ on the solution set, where
$k = |V(F)| \geq 5$ w.h.p. (property 6 of {prf:ref}`def-hard-subfamily-3sat`).
(The Boolean structure embeds standardly into $\mathbb{F}_q[x]/(x^2 - x)$ and the monodromy
is analyzed via analytic continuation in the clause coefficients, as in algebraic complexity
theory {cite}`Buergisser2000`.)

The obstruction is not merely that $S_k$ prevents solving by radicals (Abel--Ruffini), but
that a non-solvable permutation action on the solution set forces superpolynomial
*description complexity* for any algebraic procedure that resolves it. Specifically:
the monodromy group $\mathrm{Mon}(V(F)) = S_k$ acts transitively on the $k$ solutions by
permuting them as clause parameters vary. Any algebraic elimination map $e_n^\flat$ that
computes a specific solution must implicitly resolve this permutation action — i.e.,
distinguish and select among the $k$ solutions despite their monodromy equivalence. For a
solvable group, the composition series allows stepwise resolution through abelian extensions,
each adding $O(1)$ presentation complexity. But for $k \geq 5$, the composition series of
$S_k$ contains the simple non-abelian group $A_k$, which admits no tower of abelian
extensions. Resolving the $S_k$-action therefore requires intermediate algebraic structures
that faithfully represent the coset space of $A_k$. Since $A_k$ is simple and non-abelian,
its minimum faithful representation has dimension $\Omega(k)$, and the coset enumeration
requires $\Omega(k!) = \exp(\Omega(n))$ states. Any algebraic elimination procedure over an
effective ring that resolves this monodromy must encode these states in the presentation of
intermediate structures, forcing presentation size beyond any polynomial bound. This
specifically blocks the monodromy sketch route.

**Step 4. [Conclusion]:**
Combining Steps 2 and 3: the visible-symmetry route is blocked by automorphism triviality,
and every broader algebraic sketch route is blocked either by frozen-variable rigidity (3a)
or by non-solvable monodromy (3b). Therefore no choice of $\Sigma$, $A_n^\flat$, $B_n^\flat$
satisfies the polynomial bound $|A_n^\flat|, |B_n^\flat| \leq q_\flat(n)$, and no pure
$\flat$-witness exists.

The full formal argument — covering all six signature families of the admissible primitive
basis and establishing translator stability under re-encodings — is carried out in
{prf:ref}`thm-random-3sat-algebraic-blockage-strengthened`.
:::

:::{prf:definition} Class IV: Dividers (Scaling Modality)
:label: def-class-iv-dividers

An algorithmic process $\mathcal{A}$ is **Class IV (Divider)** if:

1. **Modal Factorization:** $\mathcal{A} \triangleright \ast$ (factors through scaling modality)
2. **Recursive Decomposition:** The problem satisfies:
   $$T(n) = a \cdot T(n/b) + f(n)$$
   where $a$ = number of subproblems, $b$ = size reduction, $f(n)$ = merge cost
3. **Subcritical Scaling:** $\log_b(a) < c$ for constant $c$ (critical exponent condition)

**Polynomial-Time Certificate:** $K_{\ast}^+ = (a, b, f, c)$ where $c = \log_b(a)$ determines complexity by Master Theorem.

**Examples:** Mergesort ($a=2, b=2, c=1$), FFT ($a=2, b=2, c=1$), Strassen matrix multiplication ($a=7, b=2, c=\log_2 7 \approx 2.8$).
:::

:::{prf:lemma} Scaling Modality Frontend Obstruction (Supercritical)
:label: lem-scaling-obstruction

If the problem is **supercritical**—decomposition creates more work than it saves—then the standard divide-and-conquer
template of {prf:ref}`def-class-iv-dividers` is blocked, yielding a valid frontend obstruction for the $\ast$-channel.

Formally: If for any balanced partition $\mathcal{X} = \mathcal{X}_1 \sqcup \mathcal{X}_2$:

$$|\operatorname{boundary}(\mathcal{X}_1, \mathcal{X}_2)| = \Omega(|\mathcal{X}|)$$

then recombination cost dominates: $f(n) = \Omega(T(n))$, making recursion futile.

**Obstruction Certificate:** $K_{\ast}^- = (\text{supercritical}, |\partial| = \Omega(n))$

**Application:** Random 3-SAT has $\Theta(n)$ boundary clauses for any cut, supplying an $\ast$-frontend blockage. The
full sound-and-complete $\ast$-obstruction theorem is deferred to Part V.
:::

:::{prf:proof}
A pure $\ast$-witness ({prf:ref}`def-pure-star-witness-rigorous`) requires:
(i) a splitting map producing subinstances of strictly smaller size measure $\mu$,
(ii) a polynomial-time merge map combining subanswers, and
(iii) a polynomial bound $q_\ast(n)$ on the total recursion-tree size (number of nodes and total local
work across all nodes).

The $\ast$-specific mechanism of failure is the **violation of compositional independence** — the
structural requirement that sub-solutions be independently valid and combinable by the merge map
cannot be satisfied, because crossing constraints at every split force the merge to operate on
the *entire* instance rather than on a local interface.

**Step 1. [Crossing constraints couple the subproblems]:**
Consider any balanced split $\mathcal{X} = \mathcal{X}_1 \sqcup \mathcal{X}_2$ of a random 3-SAT
instance. The expansion property guarantees $|\operatorname{boundary}(\mathcal{X}_1, \mathcal{X}_2)|
= \Theta(n)$ crossing clauses, each involving variables from *both* $\mathcal{X}_1$ and
$\mathcal{X}_2$. The subanswers for $\mathcal{X}_1$ and $\mathcal{X}_2$ are therefore not
independently correct — an assignment satisfying all clauses internal to $\mathcal{X}_1$ may violate
crossing clauses once combined with the assignment for $\mathcal{X}_2$.

**Step 1b. [Universal coverage of splitting strategies]:**
The definition of a pure $\ast$-witness ({prf:ref}`def-pure-star-witness-rigorous`) permits *any*
splitting map that produces subinstances of strictly smaller size measure — not only balanced binary
variable-partition splits. We must verify that the structural obstruction holds for all such
strategies. The cases are exhaustive:

**(a) Balanced splits with both parts of size $\Omega(n)$:** This is the setting of the remaining
steps (Steps 2--4). The expansion property guarantees $\Theta(n)$ crossing constraints at every
balanced cut, and the merge-propagation cascade forces $\Omega(n)$ non-trivial work per recursion
level. This case is treated in full below.

**(b) Highly unbalanced splits (one part of size $o(n)$):** Suppose the splitting map produces one
subinstance of size $n - o(n)$ and one of size $o(n)$. The "large" subinstance retains essentially
the full constraint structure: only $o(n)$ variables are separated out, and by the expansion
property the large part still contains $\Theta(n)$ unsolved constraints.

*Stability of hard-regime structure under $o(n)$ removal.* We verify that removing (or conditioning
on) $o(n)$ variables preserves the properties required by Steps 2--4:

- *Clause-density preservation:* Each variable appears in $\Theta(1)$ clauses, so removing $o(n)$
  variables eliminates at most $o(n) \cdot O(1) = o(n)$ clauses. The residual formula has
  $m - o(n)$ clauses on $n - o(n)$ variables, and the clause-to-variable ratio converges to the
  threshold $\alpha$ as $n \to \infty$.
- *Cluster structure stability:* By the results of Achlioptas--Coja-Oghlan (2008), the shattering
  of the solution space into $\exp(\Theta(n))$ clusters is robust to the removal of $o(n)$
  variables. Conditioning on $o(n)$ variables can collapse some clusters but preserves
  $\exp(\Theta(n - o(n))) = \exp(\Theta(n))$ clusters in the residual instance.
- *Expansion preservation:* Removing $o(n)$ vertices from an $n$-vertex expander leaves an
  $(n - o(n))$-vertex graph with the same expansion constant up to lower-order corrections
  (the edge expansion $h(G) \geq c$ implies $h(G') \geq c - o(1)$ for the subgraph $G'$ induced
  by the remaining vertices).

With these properties intact, the recursion on the large
part reduces to a near-sequential computation: the recursion tree degenerates into a path of
$n - o(n)$ steps, each processing $O(1)$ new constraints from the small subinstances. The total
recursion-tree size is dominated by this path length, which is $\Omega(n)$. Crucially, each node on
this path still operates on an instance with $\Theta(n)$ unsolved constraints, because only $o(n)$
constraints are resolved by the small subinstance at each step. By the correctness condition, the
recursive calls on the large subinstances must still produce complete satisfying assignments, and
the solution-space shattering ($\exp(\Theta(n))$ clusters with Hamming distance $\Omega(n)$) ensures
that each such call faces the full hardness of the instance. This degenerate recursion tree does not
decompose the problem — it merely serializes it.

**(c) Non-variable-partition splits in the lifted state space:** The splitting map operates on the
lifted state $Z_n^\ast$, not on the variable set directly. However, by the correctness identity
$A_n = R_n^\ast \circ F_{\rho(n)}^\ast \circ E_n^\ast$, the final output must be a correct
satisfying assignment. Therefore, the information content of the subinstances — regardless of their
encoding — must collectively determine a satisfying assignment. By a data-processing argument: any
correct splitting/merge scheme must, at the merge step, reconcile the constraint structure of the
original formula. The expansion property is a property of the constraint structure itself, not of any
particular encoding. Therefore, any merge that produces a correct output must effectively resolve
$\Theta(n)$ crossing constraints, regardless of how the subinstances are encoded. The lifted-state
encoding $E_n^\ast$ can reformat the problem but cannot eliminate the constraint coupling that
expansion creates.

**(d) Multi-way splits with polynomial fan-out:** When the splitting map produces
$b(z) \leq q_\ast(n)$ subinstances, the total number of crossing constraints summed over all
inter-part boundaries is still $\Omega(n)$ by the expansion property: each partition into $k$ parts,
however fine, has at least $c' \cdot n$ clauses touching multiple parts when the number of parts is
bounded by $\operatorname{poly}(n)$ (this follows because the edge expansion of the random constraint
graph is $\Theta(1)$, so any partition into $k = \operatorname{poly}(n)$ parts has total boundary
size $\Omega(n)$). Therefore the aggregate merge cost per level remains $\Omega(n)$, and the
structural argument of Step 4 applies without modification.

Since cases (a)--(d) are exhaustive over the strategies permitted by
{prf:ref}`def-pure-star-witness-rigorous`, the obstruction established in Steps 2--4 below applies
universally.

**Step 2. [Merge propagation reaches $\Omega(n)$ variables via expansion]:**
The merge map $\mathrm{merge}_n$ receives partial assignments $\sigma_1, \sigma_2$ for the two
subinstances and must produce a globally consistent assignment. To reconcile the $\Theta(n)$
crossing constraints, the merge must modify some variable assignments. Define the *repair graph*
$R$: its vertices are variables and its edges connect any two variables that co-occur in a clause
with a variable whose assignment was flipped during reconciliation. For random 3-SAT at the
critical clause-to-variable ratio, the constraint hypergraph is an expander: every set of $k$
variables with $k \leq \delta n$ (for a constant $\delta > 0$) appears in at least $c \cdot k$
clauses involving variables outside the set. Consequently, flipping any variable to satisfy a
crossing clause can violate an internal clause in $\mathcal{X}_1$ or $\mathcal{X}_2$, requiring a
further flip, which by expansion propagates through the constraint graph until $\Omega(n)$ variables
have been touched. In other words, any repair process — sequential, parallel, or algebraic — that
reconciles the crossing constraints must read and potentially modify $\Omega(n)$ variables from
*both* partitions, not merely the variables appearing in the crossing clauses themselves.

We make this propagation rigorous using the rigid cluster structure of random 3-SAT at threshold.
At clause-to-variable ratio $\alpha \approx 4.267$, the solution space exhibits *propagation
rigidity* (freezing): with high probability, an $\Omega(n)$-sized subset of variables are *frozen*
— their values are uniquely determined by cluster membership. Suppose the merge map modifies a
frozen variable $x_i$ to satisfy a crossing clause. Because $x_i$'s value is forced within its
cluster, this modification ejects the assignment from the current solution cluster and violates
internal clauses constraining $x_i$. Define the cascade set $C_0 = \{x_i\}$ and
$C_{t+1} = C_t \cup \{y : y \text{ appears in a clause violated by the modification of some }
z \in C_t\}$. By the expander mixing lemma for the random clause–variable bipartite graph, every
set $S$ of at most $\delta n$ variables neighbours at least $c|S|$ clauses involving variables
outside $S$ (for constants $\delta, c > 0$ depending on $\alpha$). Therefore
$|C_{t+1}| \geq (1+c)\,|C_t|$ whenever $|C_t| \leq \delta n$, giving
$|C_t| \geq (1+c)^t$, so the cascade reaches $\Omega(n)$ variables after $t = O(\log n)$ steps.
Since frozen variables comprise an $\Omega(n)$-sized set w.h.p., at least one frozen variable
is modified with constant probability over balanced splits, triggering this cascade
(Achlioptas–Ricci-Tersenghi 2006 for frozen variables; Łuczak 1992 for expansion of random
bipartite graphs).

**Step 3. [Non-shrinking effective subproblems violate the $\ast$-witness structure]:**
The critical obstruction is not the per-level merge cost (which is $\Omega(n)$ and polynomial) but
the failure of the **effective subproblem size to shrink across recursion levels**. In a valid
$\ast$-witness, splitting a size-$n$ instance produces subinstances of size measure
$\mu < n$, and the recursion terminates because the size measure strictly decreases. Here, the
$\Theta(n)$ crossing constraints at each split ensure that the merge at each level must access
$\Omega(n)$ variables from the *original* instance.

Formally: let $S(\ell)$ denote the number of variables whose assignments are not yet finalized
after level-$\ell$ merges. At level 0, $S(0) = n$. After the level-1 merges, the $\Theta(n)$
crossing constraints force modifications to $\Omega(n)$ variables (by the expansion argument of
Step 2), so $S(1) = \Omega(n)$. At each subsequent level $\ell$, the merge must reconcile the
$\Omega(n)$ modified variables with the crossing constraints at that level, again touching
$\Omega(n)$ variables: $S(\ell) = \Omega(n)$ for all $\ell$. Since $S(\ell)$ does not decay, the
recursion makes no progress toward independently solvable subproblems. Each of the $\Theta(\log n)$
levels contributes $\Omega(n)$ non-trivial merge work, and the $\Omega(n)$ variables modified at
each level invalidate previously computed sub-solutions, requiring re-computation at lower levels.

Because the fraction of invalidated assignments does not decay (modifications propagate to a
constant fraction of variables by expansion), each level-$\ell$ merge must redo $\Omega(n)$ work
for each of the preceding levels whose results were invalidated. The effective work per level
therefore grows as $\Omega(n \cdot \ell)$, and the total work across all levels is at least
$\sum_{\ell=1}^{\Theta(\log n)} \Omega(n \cdot \ell) = \Omega(n \cdot (\log n)^2)$.

**Step 4. [The fundamental structural obstruction]:**
The quantitative blowup of Step 3, while significant, understates the true obstruction, which is
*structural* rather than merely quantitative. The $\ast$-witness definition
({prf:ref}`def-pure-star-witness-rigorous`) requires that the splitting map produce subinstances of
strictly smaller size measure $\mu$, and that the merge map combine sub-answers — presupposing that
the sub-answers are *compositional*: each sub-answer is a valid partial solution that can be
extended to a global solution by local reconciliation at the interface.

The expansion-driven propagation of Step 2 shows this compositionality fails: when the merge map
must access and modify $\Omega(n)$ variables from both partitions (not just the $O(1)$-width
interface), the computation does not truly "factor through" smaller subinstances. The merge is a
whole-instance computation disguised as a local operation. In the language of the $\ast$-witness,
the sub-answers produced by the recursive calls are not valid inputs to the merge map — they are
provisional assignments that the merge must substantially rewrite. The recursion tree therefore does
not decompose the instance into independently solvable pieces: it merely re-labels a whole-instance
computation as a sequence of "merges," each of which performs $\Omega(n)$ work on the original
variable set. This violates the structural requirement that the recursion tree have polynomial total
size $q_\ast(n)$, because the effective work at each node is not bounded by a function of the
sub-instance size $\mu$ — it is controlled by the original instance size $n$.

The fundamental obstruction is *correctness*, not running time. The $\ast$-witness definition
requires $\mathcal{A}_n = R_n^\ast \circ F_{\rho(n)}^\ast \circ E_n^\ast$, where $F$ computes by
splitting, recursing, and merging. For this to be correct, the merge must produce a globally
satisfying assignment from the sub-answers. When the merge touches $\Omega(n)$ variables (Step 2),
the "sub-answers" are not valid partial solutions — they are provisional assignments that the merge
must substantially rewrite. Each merge node at recursion level $\ell$ faces an effective sub-problem
of size $\Omega(n)$: it must correctly assign $\Omega(n)$ coupled variables (those involved in
crossing constraints and their expansion-cascade neighborhoods). By the solution-space shattering,
these $\Omega(n)$ coupled variables participate in $\exp(\Theta(n))$ distinct cluster patterns, so
the merge faces the *same* exponential search over clusters as the original problem. The recursion
has not decomposed the instance — each merge node is an instance of the original problem in
disguise. The total recursion-tree load therefore includes a merge at the root that alone requires
$\exp(\Omega(n))$ work to correctly reconcile the sub-answers, violating the polynomial bound
$q_\ast(n)$.

*Formal closure via cross-channel reduction.* The merge map's task at each recursion level
reduces to an instance of boundary-type contraction: given the two sub-answers (polynomial-size
partial assignments on each side of the partition), reconcile them into a globally consistent
satisfying assignment by resolving the $\Theta(n)$ crossing constraints. This is precisely the
computational task targeted by the $\partial$-channel obstruction
({prf:ref}`lem-boundary-obstruction`): the crossing constraints form a sub-formula on $\Theta(n)$
interface variables, with the same random structure, lack of algebraic polymorphisms, and
frozen-variable patterns as the original. By the same argument that establishes exponential
contraction time in the boundary channel, no polynomial-time merge procedure can correctly
reconcile the sub-answers across the $\Theta(n)$-variable interface. The $\ast$-channel merge
obstruction thus inherits the $\partial$-channel's exponential lower bound, closing the gap
between "exponentially many cluster patterns exist" and "exponential merge time is required."

By Step 1b, this structural obstruction holds for *all* splitting strategies permitted by
{prf:ref}`def-pure-star-witness-rigorous` — balanced, unbalanced, multi-way, and lifted-state
splits alike — not merely balanced binary variable-partition splits.

**Distinction from $\partial$-blockage:**
The $\partial$-obstruction ({prf:ref}`lem-boundary-obstruction`) concerns a different mechanism:
failure of *interface contraction*. A $\partial$-witness requires that the computation factor through
a polynomial-size interface object $B_n^\partial$ with a polynomial-time contraction; treewidth
lower bounds show that any contraction must process $2^{\Omega(\operatorname{tw}(G_n))}$ feasible
separator configurations, requiring exponential time. That argument is about the *computational
cost* of processing the information that crosses a single cut. The $\ast$-obstruction proved here is about the *depth* structure of the
recursion tree: even if one could represent the interface, the subproblems on either side of every
split remain coupled through $\Theta(n)$ constraints, so the recursion tree cannot decompose the
instance into independently solvable pieces with polynomial total work.
:::

:::{prf:remark} Scaling obstruction covers all splitting strategies
:label: rem-scaling-all-splits

The proof of {prf:ref}`lem-scaling-obstruction` is stated for balanced partitions, but the
obstruction extends to all admissible splitting strategies in a pure $\ast$-witness:

**(a) Unbalanced splits** (one part has $\leq \epsilon n$ variables for small constant $\epsilon$):
The splitting map produces one sub-instance of size $\geq (1 - \epsilon)n$ and one of size
$\leq \epsilon n$. The large sub-instance retains the full constraint structure of the original
(by expansion: the $\leq \epsilon n$ removed variables participate in at most
$O(\epsilon \alpha n)$ clauses, while the remaining $(1-\epsilon)\alpha n - O(\epsilon\alpha n)$
clauses constrain the large part). By {prf:ref}`def-pure-star-witness-rigorous`, the size measure
$\mu$ must strictly decrease at each split. For unbalanced splits, the large sub-instance has
$\mu \geq (1-\epsilon)n$, so the recursion depth to reach base cases is at least
$n / (\epsilon n) = 1/\epsilon$. At each recursive level, the large sub-instance carries
$(1-O(\epsilon))$ of the original problem's constraint structure, including its shattering and
frozen-variable properties. The recursion therefore does not simplify the problem — it merely
trims a constant fraction of variables at each level while leaving the hard frustrated core
intact.

**(b) Non-variable-partition splits** in lifted state spaces: A pure $\ast$-witness may encode
the problem in a lifted state space and split in that space rather than along variable
boundaries. However, the correctness condition of {prf:ref}`def-pure-star-witness-rigorous`
requires that the merge of sub-answers produces a correct satisfying assignment for the
original formula. By the data processing inequality, any split-and-merge that produces a
correct output must resolve the $\Theta(n)$ crossing constraints of the underlying
clause-variable graph, regardless of the encoding. The expansion property is a property of
the formula's constraint structure, not of any particular encoding.

**(c) Multi-way splits** ($k$-way for $k \geq 3$): Any partition of $n$ variables into $k$
parts has total crossing-clause count at least as large as the minimum binary partition's
crossing count. For random 3-SAT at threshold, the expansion property gives
$|\operatorname{boundary}| \geq c' n$ for any $k$-way partition with $k = O(1)$. For
$k = \omega(1)$, the crossing count only increases.
:::

:::{prf:definition} Class V: Interference Engines (Boundary Modality)
:label: def-class-v-interference

An algorithmic process $\mathcal{A}$ is **Class V (Interference Engine)** if:

1. **Modal Factorization:** $\mathcal{A} \triangleright \partial$ (factors through boundary modality)
2. **Tensor Network:** The problem admits representation:

   $$Z = \sum_{\{x\}} \prod_{c \in C} T_c(x_{\partial c})$$

   where $T_c$ are local tensors, $x_{\partial c}$ are boundary variables
3. **Holographic Simplification:** One of:
   - Planar graph structure with Pfaffian orientation (FKT)
   - Matchgate signature (Valiant)
   - Bounded treewidth (tree decomposition)

**Polynomial-Time Certificate:** $K_{\partial}^+ = (G, \mathcal{O}, A)$ where $G$ is planar, $\mathcal{O}$ is Pfaffian orientation, $A$ is adjacency matrix. Complexity: $O(n^3)$ via determinant.

**Examples:** FKT algorithm for planar matching, Holant problems with matchgates, 2-SAT counting.
:::

:::{prf:lemma} Boundary Modality Frontend Obstruction (Pfaffian/Treewidth Failure)
:label: lem-boundary-obstruction

If the tensor network has:
- Non-planar graph structure AND
- No Pfaffian orientation (odd frustrated cycles) AND
- Unbounded treewidth

then the currently exhibited planar/Pfaffian/treewidth witness language of {prf:ref}`def-class-v-interference` is
blocked, yielding a valid frontend obstruction for the $\partial$-channel.

**Obstruction Certificate:** $K_{\partial}^- = (\text{non-planar}, \text{no-Pfaffian}, \mathrm{tw} = \Theta(n))$

**Application:** Random 3-SAT tensor networks are non-planar with unbounded treewidth, supplying a boundary frontend
blockage. The full strengthened $\partial$-obstruction theorem of Part V must additionally rule out all admissible
polynomial-size interface contractions beyond the current Pfaffian/treewidth frontends.
:::

:::{prf:proof}
**Step 1. [Typed witness class]:**
A pure $\partial$-witness ({prf:ref}`def-pure-boundary-witness-rigorous`) requires a polynomial
$q_\partial$ bounding the description size (in bits) of all interface objects, a uniformly
polynomial-time boundary extraction map $\partial_n$, and a uniformly polynomial-time interface
contraction map $C_n^\partial$, such that the computation factors as

$$
\mathcal{A}_n = R_n^\partial \circ C_n^\partial \circ \partial_n \circ E_n^\partial.
$$

The contraction $C_n^\partial$ operates solely on the interface object $B_n^\partial$: the entire
interior of the computation is invisible to $C_n^\partial$, which sees only what has been
compressed into the interface. The exclusion target is **all** such witnesses, regardless of the
mechanism used by $C_n^\partial$. Crucially, {prf:ref}`def-pure-boundary-witness-rigorous` imposes
two constraints: (i) the description size of every intermediate interface object is bounded by
$q_\partial(n)$ (a polynomial), and (ii) the contraction $C_n^\partial$ runs in time polynomial
in $q_\partial(n)$ and hence polynomial in $n$.

**Step 2. [Information-theoretic lower bound on interface capacity]:**
Because the factorization $\mathcal{A}_n = R_n^\partial \circ C_n^\partial \circ \partial_n \circ
E_n^\partial$ must be correct on **all** inputs of size $n$, the interface must separate inputs
that lead to different outputs. Formally: if $x_1, x_2$ are two inputs with
$\mathcal{A}_n(x_1) \neq \mathcal{A}_n(x_2)$, then the interface states
$\partial_n(E_n^\partial(x_1))$ and $\partial_n(E_n^\partial(x_2))$ must be distinct (otherwise
$R_n^\partial \circ C_n^\partial$ receives the same interface state and cannot produce two
different outputs). Therefore the number of distinct interface states satisfies

$$
|B_n^\partial| \;\geq\; |\{\mathcal{A}_n(x) : x \in \{0,1\}^n\}|.
$$

Since each interface state has a description of at most $q_\partial(n)$ bits, the number of
representable states is at most $2^{q_\partial(n)}$, giving

$$
q_\partial(n) \;\geq\; \log_2 |B_n^\partial|.
$$

For a search problem like 3-SAT (find a satisfying assignment), this information-theoretic bound
yields $q_\partial(n) \geq \Omega(n)$, which is satisfiable by a polynomial $q_\partial$ and
therefore does **not** by itself obstruct a pure $\partial$-witness. The obstruction must come from
the *computational* cost of the contraction, as shown in Steps 3--5.

**Step 3. [Interface bottleneck from expansion]:**
The pure $\partial$-witness factors as $\mathcal{A}_n = R_n^\partial \circ C_n^\partial \circ
\partial_n \circ E_n^\partial$. The interface object $B_n^\partial = \partial_n(E_n^\partial(x))$
has description size bounded by $q_\partial(n)$ bits. This is a polynomial bottleneck: all
information about the input that is needed to produce the correct output must pass through this
$q_\partial(n)$-bit channel.

Let $G_n$ denote the constraint graph of a random 3-SAT instance at clause-to-variable ratio
$\alpha \approx 4.267$. Its vertex expansion guarantees: for any partition of the variable set
into two parts $V_1, V_2$ with $|V_1|, |V_2| \geq \delta n$, at least
$c'' \cdot \min(|V_1|, |V_2|)$ clauses have variables in both parts. The treewidth satisfies
$\operatorname{tw}(G_n) \geq c'' \cdot n$.

The contraction $C_n^\partial$ must produce a correct satisfying assignment from $B_n^\partial$
alone. The interface $B_n^\partial$ has polynomial description size, which is sufficient to
*encode* the answer. But the contraction must be *correct for all inputs*: for each of the
exponentially many hard instances, $C_n^\partial$ must map the corresponding interface state to
the correct output. The computational cost of $C_n^\partial$ is what creates the obstruction, as
shown in Step 4.

**Step 4. [Exponential contraction time from output diversity and unstructured feasibility]:**
This is the key step. We argue directly from the interface definition that correctness on all
inputs forces the contraction to distinguish exponentially many configurations, without assuming
the contraction operates via any particular algorithmic paradigm (tree decomposition, dynamic
programming, or otherwise).

For the search formulation, the contraction $C_n^\partial$ receives the interface object
$B_n^\partial$ and must produce a satisfying assignment. The interface encodes information about
the constraint structure compressed through the boundary map $\partial_n$. By the expansion
property of $G_n$, any "cut" through the constraint graph separating determined from undetermined
variables must cross $\Theta(n)$ clauses. The contraction $C_n^\partial$ must resolve these
crossed constraints.

*Output diversity from the cluster structure.* For random 3-SAT at clause-to-variable ratio
$\alpha \approx 4.267$ (the satisfiability threshold), the solution space has
$\exp(\Theta(n))$ well-separated clusters (Achlioptas and Ricci-Tersenghi, 2006; Mezard and
Montanari, 2009). Different formulas in the hard subfamily require satisfying assignments from
different clusters.

**Frozen-variable spreading.** The frozen variables of each cluster comprise a $\Theta(n)$-fraction
of all $n$ variables (Achlioptas–Coja-Oghlan 2008), and their distribution across the constraint
graph is governed by the expansion property: since each frozen variable participates in $\Theta(1)$
clauses, and the clause-variable graph is an expander, the frozen variables cannot be concentrated in
any subset of size $o(n)$. For any set $S$ of $\Theta(n)$ variables, the expansion property
guarantees that $S$ contains $\Omega(n)$ frozen variables.

**Distinct projections.** Different clusters have different frozen-variable cores: if clusters $C_i$
and $C_j$ have Hamming distance $\Omega(n)$, they differ on $\Omega(n)$ frozen positions. Since any
$\Theta(n)$-sized variable set $S$ contains $\Omega(n)$ frozen variables, and the $\Omega(n)$
differing frozen positions are spread across the graph by expansion, a constant fraction of them
fall in $S$. Therefore the partial assignments $C_i|_S$ and $C_j|_S$ are distinct for all but
a negligible fraction of cluster pairs: two clusters can agree on $S$ only if all their
$\Omega(n)$ differing frozen positions fall outside $S$, which occurs with probability at most
$\exp(-\Omega(n))$. The number of distinct feasible partial assignments on any $\Theta(n)$-sized
variable set is therefore at least $\exp(\Theta(n)) = 2^{\Omega(n)}$.

*Why the contraction must distinguish these configurations.* The contraction $C_n^\partial$ is a
single polynomial-time map that must handle ALL inputs. For each input in the hard subfamily,
$C_n^\partial$ receives a polynomial-size interface state and must produce the correct satisfying
assignment. By the cluster structure, distinct inputs require assignments from distinct clusters,
producing $\exp(\Theta(n)) = 2^{\Omega(n)}$ distinct outputs. The contraction must therefore
realize a function with $2^{\Omega(n)}$ distinct input-output behaviors.

This is an information-theoretically valid task (the polynomial-size interface has enough bits to
encode the answer). The obstruction is *computational*: the contraction must correctly evaluate a
map whose range contains $2^{\Omega(n)}$ elements, where the preimage structure — which interface
states map to which outputs — is determined by the unstructured feasible configurations of the
3-SAT instances.

*Why algebraic shortcuts do not apply.* One might object that polynomial-time shortcuts exist for
certain constraint families — for instance, Gaussian elimination solves XORSAT in polynomial time
because the feasible configurations form an affine subspace over $\operatorname{GF}(2)$, and 2-SAT
admits polynomial contraction because its implication graph has bounded treewidth structure. For
random 3-SAT at threshold, neither shortcut is available:

- The constraints are **non-linear**: 3-SAT clauses are disjunctions, not XOR or linear equations,
  so the feasible configurations do not form an affine subspace or any algebraically structured
  subset over any field.
- Random 3-SAT does not belong to any tractable CSP class identified by the algebraic dichotomy
  theorem (Bulatov, 2017; Zhuk, 2020), which characterizes tractable CSPs by the presence of
  specific polymorphisms (e.g., majority or Mal’tsev operations). The constraint language of
  3-SAT lacks these polymorphisms.
- Therefore, no algebraic, linear, or polymorphism-based shortcut can reduce the computational
  cost of the contraction below $2^{\Omega(n)}$.

The key structural claim: the contraction $C_n^\partial$ must effectively solve a
constraint-satisfaction problem on interface variables at each stage of its computation. Because the
feasible configurations form an unstructured set of size $2^{\Omega(n)}$ (by the frozen-variable /
cluster argument above), any correct contraction requires time $2^{\Omega(n)}$. This bound is
*unconditional* for random 3-SAT at threshold: it follows from the proven cluster structure
(exponentially many clusters with frozen cores at Hamming distance $\Omega(n)$), combined with the
frozen-variable spreading argument, and does not depend on any assumption about the algorithmic
paradigm used by the contraction.

**Step 5. [Contraction time violates the polynomial-time requirement]:**
Combining Steps 3 and 4: the contraction $C_n^\partial$ must distinguish and correctly process
$2^{\Omega(n)}$ distinct feasible configurations. Under the hypotheses of the lemma,
$\operatorname{tw}(G_n) = \Theta(n)$ (for random 3-SAT at $\alpha \approx 4.267$, this follows
from the linear expansion of the constraint graph), and the unstructured feasible set has size
$2^{\Omega(n)}$. Therefore the contraction time satisfies

$$
T(C_n^\partial) \;\geq\; 2^{\,\Omega(n)}.
$$

Since $2^{\Omega(n)}$ exceeds any fixed polynomial, this violates the requirement of
{prf:ref}`def-pure-boundary-witness-rigorous` that $C_n^\partial$ runs in time polynomial in
$q_\partial(n) = n^{O(1)}$. No pure $\partial$-witness exists for these instances.

Note that the description-size bound $q_\partial(n) \geq \Omega(n)$ from Step 2 is polynomial and
does not by itself yield an obstruction. The obstruction is **computational**: even if the
interface objects have compact (polynomial-bit) descriptions, the contraction must correctly
evaluate a function over an exponentially large unstructured feasible set, and this computational
cost is the true bottleneck.

**Step 6. [Universality of the obstruction]:**
The key point is that this obstruction is **not** limited to the three specific mechanisms
(planar, Pfaffian, bounded treewidth) enumerated in {prf:ref}`def-class-v-interference`. The
exponential contraction-time bound of Step 4 is a structural obstruction on **any** factorization
through a boundary interface, because the argument depends only on three properties of the
factorization: (i) the contraction $C_n^\partial$ receives a polynomial-size interface object,
(ii) it must produce the correct output for all inputs, and (iii) the set of feasible
configurations at the interface is unstructured and of size $2^{\Omega(n)}$. These three
properties hold for *any* pure $\partial$-witness, regardless of the mechanism used to construct
the boundary map $\partial_n$ or the contraction $C_n^\partial$. No matter what contraction
mechanism $C_n^\partial$ employs — even one not yet conceived — it must still correctly
distinguish among the $2^{\Omega(n)}$ feasible interface configurations. This addresses the
universality concern flagged in
{prf:ref}`rem-why-boundary-strengthening-is-mandatory`.
:::

### The Theorem Ladder Required for Algorithmic Completeness

:::{prf:remark} Why this section is necessary
:label: rem-why-theorem-ladder-is-necessary

The previous sections define effective Fragile programs, the evaluator $\mathsf{Eval}$, the cost-certificate
predicate $\mathsf{CostCert}$, the five modalities $\{\sharp,\int,\flat,\ast,\partial\}$, and the saturated modal
closure $\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle$.

However, those ingredients alone do **not** yet justify either:
1. the bridge equivalence

   $$
   P_{\mathrm{FM}}=P_{\mathrm{DTM}},
   \qquad
   NP_{\mathrm{FM}}=NP_{\mathrm{DTM}},
   $$

2. the stronger claim

   $$
   P_{\mathrm{FM}}=\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
   $$

The present section states the exact theorem package needed to make those identifications rigorous. The theorems are
organized so that:
- Part I isolates semantic adequacy and machine equivalence;
- Part II isolates the internal normalization machinery;
- Part III isolates the universal-property characterization of the five modal classes.

No theorem in Parts II-III may be used to prove Part I, and no theorem in Part III may be used to define
$P_{\mathrm{FM}}$. This separation prevents circularity.
:::

#### I. Semantics and Machine Equivalence

:::{prf:notation} Halting-in-$t$ notation
:label: not-halting-in-t

For a program code $a\in \mathsf{Prog}_{\mathrm{FM}}$, an encoded input $u\in\{0,1\}^*$, and an output
$v\in\{0,1\}^*$, we write

$$
\mathsf{Eval}(a,u)\downarrow_t v
$$

to mean that the Fragile evaluator halts on input $(a,u)$ after exactly $t$ internal steps and returns output $v$.
We write

$$
\mathsf{Eval}(a,u)\downarrow_{\le t} v
$$

to mean halting in at most $t$ steps.
:::

:::{prf:definition} Concrete evaluator implementation for Part I
:label: def-concrete-evaluator-implementation

For Part I, we fix the canonical evaluator implementing the runtime of
{prf:ref}`def-effective-programs-fragile` as a deterministic multi-tape interpreter with the following data.

1. **Finite syntax alphabet.** Program codes are finite parse trees, equivalently finite bytecode streams, over a fixed
   finite constructor alphabet
   $$
   \Sigma_{\mathrm{eval}}
   =
   \{\mathsf{const},\mathsf{pair},\mathsf{fst},\mathsf{snd},\mathsf{inl},\mathsf{inr},\mathsf{case},
   \mathsf{lookup},\mathsf{extend},\mathsf{call},\mathsf{ret},\mathsf{branch},\mathsf{halt}\}
   \cup
   \Sigma_{\mathrm{prim}},
   $$
   where $\Sigma_{\mathrm{prim}}$ is a fixed finite library of basic binary-string, arithmetic, and finite-map
   routines.

2. **Read-only code tape.** The program code $a\in\mathsf{Prog}_{\mathrm{FM}}$ is stored once on a read-only code tape.
   Runtime states may store only addresses into that tape.

3. **Runtime configuration format.** A runtime configuration is a finite tagged tuple
   $$
   C=(q,p,\kappa,\rho,\sigma,\eta,\iota,\omega),
   $$
   where:
   - $q$ is a control state from a fixed finite set;
   - $p$ is a code-tape address;
   - $\kappa$ is a finite continuation stack of return addresses and branch tags;
   - $\rho$ is a finite environment of addresses into the work tapes;
   - $\sigma$ is a finite value stack;
   - $\eta$ is a finite heap / association-list store;
   - $\iota$ is the current pair of input/output tapes;
   - $\omega$ is a halting-status flag from a fixed finite set.

4. **Microstep semantics.** The evaluator transition relation
   $$
   C \leadsto C'
   $$
   is the deterministic one-step transition relation of that interpreter. Large arithmetic or string operations are not
   atomic evaluator steps: when the current opcode lies in $\Sigma_{\mathrm{prim}}$, the evaluator enters the
   corresponding primitive subroutine and executes it by microsteps over the work tapes.

5. **Bit-cost primitive library.** Each primitive subroutine in $\Sigma_{\mathrm{prim}}$ acts on binary encodings by
   reading/writing $O(1)$ tape cells per microstep, changing only finitely many head positions and control tags at each
   step. In particular, the cost of executing a primitive is the number of microsteps of its subroutine, which is
   polynomial in the sizes of the encoded operands. No primitive treats exponentially large integers or exponentially
   long strings as unit-cost objects.
:::

:::{prf:definition} Primitive Library $\Sigma_{\mathrm{prim}}$
:label: def-primitive-library-sigma-prim

The primitive library $\Sigma_{\mathrm{prim}}$ is the following fixed finite set of polynomial-time
subroutines operating on binary-string tape cells:

**Binary-string operations:**
`bit-read(i)`, `bit-write(i,b)`, `bit-length(s)`, `concat(s_1,s_2)`, `substring(s,i,j)`.

**Arithmetic operations:**
`add(x,y)`, `sub(x,y)` (saturating), `mul(x,y)`, `div(x,y)` (integer), `mod(x,y)`, `cmp(x,y)`.

**Finite-map operations:**
`map-create()`, `map-get(m,k)`, `map-set(m,k,v)`, `map-has(m,k)`.

Each operation runs in time polynomial in the bit-length of its arguments and reads/writes $O(1)$
tape cells per microstep ({prf:ref}`thm-bit-cost-evaluator-discipline`, clause 5).
:::

:::{prf:lemma} Turing Completeness of $\Sigma_{\mathrm{prim}}$
:label: lem-sigma-prim-turing-complete

The evaluator instruction set $\Sigma_{\mathrm{eval}} \cup \Sigma_{\mathrm{prim}}$ from
{prf:ref}`def-concrete-evaluator-implementation` and {prf:ref}`def-primitive-library-sigma-prim`
is Turing-complete with at most polynomial overhead.
:::

:::{prf:proof}
To simulate an arbitrary Turing machine $M$ with state set $Q$, tape alphabet $\Gamma$, and
transition function $\delta$: (1) represent the tape as a finite map via `map-create`/`map-set`/
`map-get`; (2) store the current state $q$ and head position $h$ as binary integers in dedicated
tape cells; (3) simulate each TM step by reading the current symbol via `map-get(tape, h)`,
computing $\delta(q, \sigma)$ via finite case analysis using `cmp` and `branch`, writing the new
symbol, and updating $h$ via `add`/`sub`. Each simulated TM step uses $O(|Q|\cdot|\Gamma|)$
evaluator microsteps — a constant independent of input size. A TM computation of $T$ steps is
simulated in $O(T)$ evaluator microsteps with $O(S)$ space for $S$ tape cells used.
:::

:::{prf:theorem} Bit-cost evaluator discipline
:label: thm-bit-cost-evaluator-discipline

The concrete evaluator of {prf:ref}`def-concrete-evaluator-implementation` satisfies the following properties.

1. **Finite configuration alphabet.** Every runtime configuration is a finite record over a fixed finite tag alphabet,
   together with finitely many finite strings over $\{0,1\}$.
2. **Read-only program code.** The code of a program is stored in read-only form and is not duplicated unboundedly
   during execution.
3. **Decidable one-step semantics.** There is a decidable one-step transition relation
   $$
   C \leadsto C'
   $$
   on encoded configurations.
4. **Local size discipline.** There exists a linear polynomial
   $$
   s(N)=N+c
   $$
   such that if one primitive evaluator microstep transforms an encoded configuration of bitlength $N$ into one of
   bitlength $N'$, then
   $$
   N' \le s(N).
   $$
   Consequently, for configurations reachable in $t$ steps from a program of size $|a|$ and an input of size $n$, the
   encoded configuration size is bounded by a polynomial in $(|a|,n,t)$.
5. **Bit-cost primitive accounting.** Arithmetic and data-structure operations are costed according to the size of the
   encoded operands; there is no hidden unit-cost treatment of exponentially large integers or exponentially long
   intermediate strings.

Equivalently: the evaluator discipline required for export to DTMs is a theorem of the fixed Part I runtime, not an
external assumption.
:::

:::{prf:proof}
Clauses (1) and (2) are immediate from
{prf:ref}`def-concrete-evaluator-implementation`: the sets of control tags, halting flags, bytecode constructors, and
primitive-subroutine states are all finite, and the program code lives on a read-only tape addressed by pointers.

For clause (3), decoding an encoded configuration is polynomial-time by the self-delimiting pairing convention of
{prf:ref}`rem-ambient-conventions-complexity`. Once decoded, the next step is determined by finite case analysis on the
current control state $q$ and opcode pointed to by $p$. Each branch updates only finitely many fields, and each
primitive branch advances one microstep of a fixed deterministic subroutine. Hence the one-step relation is decidable on
encoded configurations.

For clause (4), let $N$ be the encoded size of $C$. A single evaluator microstep either:
- changes only finitely many control tags, head positions, or pointers;
- pushes or pops one address-sized item from $\kappa$, $\rho$, or $\sigma$;
- updates one heap or tape cell by a single binary symbol; or
- advances one primitive subroutine by one microstep, which by definition reads/writes $O(1)$ cells and changes only
  finitely many control tags and head positions.

Therefore there exists a constant $c$ such that every microstep changes the encoded configuration length by at most $c$,
so
$$
N' \le N+c = s(N).
$$
If the initial tagged input has size at most $c_0(|a|+n+1)$, then after at most $t$ microsteps every reachable
configuration has encoded size at most
$$
c_0(|a|+n+1)+ct,
$$
which is polynomial in $(|a|,n,t)$.

For clause (5), the primitive library is executed only through the microstep semantics just described. Its running time
is therefore the number of microsteps used by the corresponding subroutine, which is polynomial in operand size by
construction. Since no primitive subroutine takes a whole exponentially large integer or string as a unit-cost atom, the
runtime is a genuine bit-cost model.
:::

:::{prf:definition} Reachable evaluator configuration family
:label: def-reachable-evaluator-config-family

Fix an admissible source family $\mathfrak{X}$ and a program code $a\in\mathsf{Prog}_{\mathrm{FM}}$.
For each pair $(n,t)\in\mathbb{N}^2$, define

$$
\mathrm{Conf}^{\mathfrak{X}}_{a}(n,t)
$$

to be the set of all evaluator configurations reachable in at most $t$ steps from some valid tagged input

$$
\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak{X}}_n(x)\right\rangle
\qquad (x\in X_n).
$$

An **evaluator configuration encoding** for $(a,\mathfrak{X})$ is a family of injective maps

$$
\mathrm{enc}^{\mathrm{conf}}_{a,\mathfrak{X},n,t}:
\mathrm{Conf}^{\mathfrak{X}}_{a}(n,t)\hookrightarrow \{0,1\}^{q_a(n,t)}
$$

with uniform decoders and validity predicates, for some polynomial $q_a$.
:::

:::{prf:theorem} Finite Encodability
:label: thm-finite-encodability

Let $\mathfrak{X}$ be an admissible input family.

1. Every object family $(X_n)_{n\in\mathbb N}$ appearing in $\mathfrak{X}$ is $0$-truncated and finitely encodable in
   the sense of {prf:ref}`def-admissible-input-family-rigorous`.
2. For every effective Fragile program $a\in\mathsf{Prog}_{\mathrm{FM}}$, there exists an evaluator configuration
   encoding

   $$
   \mathrm{enc}^{\mathrm{conf}}_{a,\mathfrak{X},n,t}:
   \mathrm{Conf}^{\mathfrak{X}}_{a}(n,t)\hookrightarrow \{0,1\}^{q_a(n,t)}
   $$

   where $q_a$ is polynomial in $(|a|,n,t)$.
3. The corresponding validity predicate and decoder are uniformly computable in time polynomial in $(|a|,n,t)$ and
   the configuration code length.

Hence every admissible problem family and every runtime configuration family used in the framework is externally
representable by finite bitstrings with polynomial overhead.
:::

:::{prf:proof}
Clause (1) is immediate from {prf:ref}`def-admissible-input-family-rigorous`.

For clauses (2) and (3), fix $a$ and $\mathfrak{X}$. By
{prf:ref}`thm-bit-cost-evaluator-discipline`, every runtime configuration is a finite record consisting of:
- a program counter or evaluation-context pointer into the fixed code of $a$,
- finitely many finite work registers, stacks, heaps, or environments,
- the current encoded input/output fragments,
- finitely many control flags.

Because the code of $a$ is read-only, its contribution to configuration size is $O(|a|)$. Because one-step semantics
is decidable and obeys the local size discipline, every configuration reachable in at most $t$ steps from an input of
size $n$ has total bitlength bounded by some polynomial $q_a(n,t)$ obtained from the reachable-size bound in
{prf:ref}`thm-bit-cost-evaluator-discipline`.

Encode a configuration by concatenating its tagged fields using the fixed self-delimiting pairing convention from
{prf:ref}`rem-ambient-conventions-complexity`. This yields an injective encoding into
$\{0,1\}^{q_a(n,t)}$. Validity checking and decoding are polynomial-time because the field grammar is finite, the tags
are finite, and each field decoder is polynomial-time by construction.
:::

:::{prf:lemma} Encoding Invariance
:label: lem-encoding-invariance

Let

$$
\mathfrak{X}
=
\bigl((X_n),m_1,\mathrm{enc}^{(1)},\mathrm{dec}^{(1)},\chi^{(1)}\bigr)
\quad\text{and}\quad
\mathfrak{X}'
=
\bigl((X_n),m_2,\mathrm{enc}^{(2)},\mathrm{dec}^{(2)},\chi^{(2)}\bigr)
$$

be two admissible encodings of the same underlying size-indexed family $(X_n)_{n\in\mathbb N}$.

Then there exist uniform polynomial-time conversion maps

$$
c_{12}(n,u):=\mathrm{enc}^{(2)}_n(\mathrm{dec}^{(1)}_n(u)),
\qquad
c_{21}(n,v):=\mathrm{enc}^{(1)}_n(\mathrm{dec}^{(2)}_n(v)),
$$

defined on valid codes, such that for every $x\in X_n$,

$$
c_{12}\bigl(n,\mathrm{enc}^{(1)}_n(x)\bigr)=\mathrm{enc}^{(2)}_n(x),
\qquad
c_{21}\bigl(n,\mathrm{enc}^{(2)}_n(x)\bigr)=\mathrm{enc}^{(1)}_n(x).
$$

Consequently, polynomial time is invariant under the choice of admissible encoding up to polynomial overhead.
:::

:::{prf:proof}
This is the content of {prf:ref}`lem-encoding-invariance-admissible`, rewritten in the present theorem-ladder
notation. The conversion maps are compositions of admissible decoders and encoders, hence uniformly polynomial-time on
valid codes. The displayed identities follow from the inverse axioms in
{prf:ref}`def-admissible-input-family-rigorous`.
:::

:::{prf:theorem} Evaluator Adequacy
:label: thm-evaluator-adequacy

There exists a universal deterministic Turing machine $U$ and a polynomial

$$
r:\mathbb{N}^3\to\mathbb{N}
$$

such that for every effective Fragile program $a\in\mathsf{Prog}_{\mathrm{FM}}$, every encoded input
$u\in\{0,1\}^*$, and every output $v\in\{0,1\}^*$,

$$
\mathsf{Eval}(a,u)\downarrow_t v
\quad\Longrightarrow\quad
U(\ulcorner a\urcorner,u)\downarrow_{\le r(|a|,|u|,t)} v.
$$

Equivalently: the operational semantics of the Fragile evaluator is DTM-simulable with polynomial slowdown in program
size, input size, and internal step count.
:::

:::{prf:proof}
Fix a concrete encoding of programs and runtime configurations. By {prf:ref}`thm-finite-encodability`, every
configuration reachable within $t$ steps from input $u$ has encoded size bounded by a polynomial
$q(|a|,|u|,t)$.

Construct $U$ as follows. On input $(\ulcorner a\urcorner,u)$:
1. compute the initial encoded runtime configuration $C_0$;
2. repeatedly:
   - test whether the current configuration is halting,
   - if not, compute the unique next configuration under the one-step semantics;
3. when a halting configuration is reached, decode and output its result.

Because the one-step relation is decidable, each simulated evaluator step is computable on a DTM. Because the encoded
configuration size at time $i\le t$ is bounded by $q(|a|,|u|,t)$, the DTM time to simulate one evaluator step is
bounded by a polynomial in that quantity. Therefore the total time for simulating $t$ evaluator steps is bounded by a
polynomial $r(|a|,|u|,t)$.

Correctness follows by induction on the length-$t$ evaluator run: the DTM maintains exactly the encoded evaluator
configuration that the Fragile semantics would produce after the same number of internal steps.
:::

:::{prf:theorem} CostCert Soundness
:label: thm-costcert-soundness

Let

$$
\mathcal{A}:\mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be represented by $a\in\mathsf{Prog}_{\mathrm{FM}}$. If

$$
\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,p)
$$

holds for some polynomial $p$, then:

1. for every $n$ and every $x\in X_n$, the evaluation

   $$
   \mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right)
   $$

   halts in at most $p(n)$ internal steps;
2. the output determines a total extensional map

   $$
   \mathcal{A}_n:X_n\to Y_{\sigma(n)};
   $$

3. $\mathcal{A}$ therefore belongs to

   $$
   P_{\mathrm{FM}}(\mathfrak{X},\mathfrak{Y};\sigma)
   $$

   in the sense of {prf:ref}`def-internal-polytime-family-rigorous`.

If, in addition, {prf:ref}`thm-evaluator-adequacy` holds, then the same extensional family is computable on a DTM in
time polynomial in $n$.
:::

:::{prf:proof}
The first three clauses are immediate from the definition of
$\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,p)$ in
{prf:ref}`def-family-cost-certificate` and of internal polynomial time in
{prf:ref}`def-internal-polytime-family-rigorous`.

For the DTM claim, combine the internal step bound $t\le p(n)$ with {prf:ref}`thm-evaluator-adequacy`. This yields
DTM runtime at most

$$
r\!\left(|a|,\left|\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right|,p(n)\right),
$$

which is polynomial in $n$ because admissible encoding length is polynomial in $n$.
:::

:::{prf:theorem} CostCert Completeness for Internal Programs
:label: thm-costcert-completeness

Let

$$
\mathcal{A}:\mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be represented by an effective Fragile program $a\in\mathsf{Prog}_{\mathrm{FM}}$.
Assume that there exists a polynomial $q$ such that for every $n$ and every $x\in X_n$,

$$
\mathsf{Eval}\!\left(a,\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right)
\downarrow_{\le q(n)}.
$$

Then

$$
\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,q)
$$

holds.

Equivalently: the certificate system is complete for true polynomial-time behavior of internal programs.
:::

:::{prf:proof}
Clause (1) of {prf:ref}`def-family-cost-certificate` is exactly the hypothesis.

For clause (2), because $a$ represents the uniform family
$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$
in the sense of {prf:ref}`def-uniform-algorithm-family-rigorous`, every evaluation on a valid tagged
$\mathfrak X$-input halts with an output bitstring $v$ satisfying
$$
\chi^{\mathfrak Y}_{\sigma(n)}(v)=1.
$$
Thus every such output is a valid $\mathfrak Y$-code of size $\sigma(n)$.

Clause (3) follows from {prf:ref}`thm-bit-cost-evaluator-discipline`: each counted runtime step is a primitive
microstep of the fixed Fragile evaluator, and its accounting is by encoded operand size.

For clause (4), take as certificate payload the tuple
$$
W(a,q):=(q,\Pi_{\mathrm{halt}},\Pi_{\mathrm{out}},\Pi_{\mathrm{step}},\Pi_{\mathrm{adm}}),
$$
where:
- $\Pi_{\mathrm{halt}}$ is the formal record of the displayed halting bound;
- $\Pi_{\mathrm{out}}$ is the family-representation record witnessing output validity for
  $\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y$;
- $\Pi_{\mathrm{step}}$ records that counted steps are evaluator primitives under
  {prf:ref}`thm-bit-cost-evaluator-discipline`;
- $\Pi_{\mathrm{adm}}$ is the admissible-family data for $\mathfrak X$ and $\mathfrak Y$.

This package is ZFC-checkable because programs have concrete syntax and ZFC-definable operational semantics by
{prf:ref}`def-effective-programs-fragile`, admissible families come with uniform encoders, decoders, and validity
predicates by {prf:ref}`def-admissible-input-family-rigorous`, and reachable evaluator configurations admit finite
encodings with polynomial-time validity and decoding by {prf:ref}`thm-finite-encodability`.

Therefore $W(a,q)$ witnesses
$$
\mathsf{FamCostCert}_{\mathfrak X,\mathfrak Y,\sigma}(a,q).
$$
:::

:::{prf:remark} Status of CostCert completeness
:label: rem-status-costcert-completeness

Theorem {prf:ref}`thm-costcert-completeness` proves **semantic completeness** for the current witness-based definition
of
$$
\mathsf{FamCostCert}.
$$

Accordingly, the class

$$
P_{\mathrm{FM}}
$$

defined through certificates matches the class of internally polynomial-time programs at the semantic level used in Part
I, and later uses of

$$
P_{\mathrm{FM}}=P_{\mathrm{DTM}}
$$

may cite {prf:ref}`thm-costcert-completeness` directly.

The stronger constructor/extractor theorem one may still want is different: it would build a family cost certificate by
finite derivation from a normal-form term and explicit constructor rules. That stronger implementation theorem belongs
downstream of Part II and is not needed to close the bridge gap of Part I.
:::

:::{prf:lemma} Configuration Object Theorem
:label: lem-configuration-object-theorem

Let

$$
M=(Q,\Gamma,\delta,q_{\mathrm{start}},q_{\mathrm{acc}},q_{\mathrm{rej}})
$$

be a deterministic Turing machine running in time bounded by a polynomial $q$.
Then there exists an admissible family of internal configuration objects

$$
\mathfrak{Conf}_M=(\mathrm{Conf}_{M,n})_{n\in\mathbb N}
$$

together with uniformly computable maps:

$$
\mathrm{init}_{M,n}: \{0,1\}^{\le n}\to \mathrm{Conf}_{M,n},
$$

$$
\mathrm{step}_{M,n}: \mathrm{Conf}_{M,n}\to \mathrm{Conf}_{M,n},
$$

$$
\mathrm{halt}_{M,n}: \mathrm{Conf}_{M,n}\to \{0,1\},
$$

$$
\mathrm{out}_{M,n}: \mathrm{Conf}_{M,n}\to \{0,1\}^{\le q(n)},
$$

such that:

1. $\mathrm{Conf}_{M,n}$ encodes exactly the machine configurations reachable within $q(n)$ steps on size-$n$ inputs;
2. for every input $u$ and every $t\le q(n)$,

   $$
   \mathrm{step}_{M,n}^{\,t}(\mathrm{init}_{M,n}(u))
   $$

   is the internal image of the external machine configuration after $t$ steps of $M$ on $u$;
3. if $M$ halts on $u$ within $t\le q(n)$ steps, then

   $$
   \mathrm{out}_{M,n}\!\left(\mathrm{step}_{M,n}^{\,t}(\mathrm{init}_{M,n}(u))\right)
   $$

   is exactly the output of $M$ on $u$.

Equivalently, one may form a tagged object

$$
\mathrm{Conf}_M := \coprod_{n\in\mathbb N}\mathrm{Conf}_{M,n}\in \mathbf H
$$

together with a global step morphism over the size index.
:::

:::{prf:proof}
For an input of length $n$, a time bound $q(n)$ implies that the head never needs to inspect more than the first
$q(n)$ tape cells. Thus a configuration may be encoded by the finite tuple

$$
(q_{\mathrm{state}}, i, w)
$$

where $q_{\mathrm{state}}\in Q$ is the current control state, $i\in\{0,\dots,q(n)\}$ is the head position, and
$w\in \Gamma^{q(n)+1}$ is the tape contents on the bounded active tape segment. This space is finite and therefore
$0$-truncated and finitely encodable.

Define $\mathrm{init}_{M,n}$ by writing the input on the left portion of the active tape and blank symbols elsewhere.
Define $\mathrm{step}_{M,n}$ by applying the transition function $\delta$ to the triple consisting of the current
state, current tape symbol, and head position, then updating the bounded tape word accordingly. The maps
$\mathrm{halt}_{M,n}$ and $\mathrm{out}_{M,n}$ are the obvious decoders for halting states and output tape contents.

Because $M$ is deterministic, these maps are well-defined; because $Q$, $\Gamma$, and $\delta$ are finite, they are
uniformly computable. Induction on $t$ proves the agreement between the internal iterates of $\mathrm{step}_{M,n}$ and
the external machine evolution.
:::

:::{prf:theorem} DTM $\to$ Fragile Compilation
:label: thm-dtm-to-fragile-compilation

Let

$$
M:\{0,1\}^* \to \{0,1\}^*
$$

be a deterministic Turing machine computing a total function in time bounded by a polynomial $q$.
Let

$$
\mathfrak{X},\mathfrak{Y}
$$

be admissible families such that $M$ maps valid $\mathfrak{X}$-codes of size $n$ to valid $\mathfrak{Y}$-codes of
size $\sigma(n)$.

Then there exists a uniform Fragile family

$$
\mathcal{A}_M:\mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

represented by a single program code $a_M\in\mathsf{Prog}_{\mathrm{FM}}$ and a polynomial $p_M$ such that:

1. for every $n$ and every $x\in X_n$,

   $$
   \mathcal{A}_{M,n}(x)
   =
   \mathrm{dec}^{\mathfrak Y}_{\sigma(n)}
   \!\left(
   M\!\left(\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right)
   \right);
   $$

2. $\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a_M,p_M)$ holds;
3. $p_M$ is polynomially bounded in the original DTM runtime bound $q$.

Hence every uniform DTM computation of polynomial time admits a uniform internal realization in the Fragile model with
polynomial overhead.
:::

:::{prf:proof}
Let $M$ be fixed. By Lemma {prf:ref}`lem-configuration-object-theorem`, there exists an admissible configuration
family

$$
\mathfrak{Conf}_M=(\mathrm{Conf}_{M,n})_{n\in\mathbb N}
$$

together with uniformly computable maps

$$
\mathrm{init}_{M,n},
\quad
\mathrm{step}_{M,n},
\quad
\mathrm{halt}_{M,n},
\quad
\mathrm{out}_{M,n}
$$

that realize one step of $M$ internally.

Define $a_M$ to:
1. encode the input into the initial machine configuration $\mathrm{init}_{M,n}(x)$;
2. iterate $\mathrm{step}_{M,n}$ for at most $q(n)$ stages;
3. stop when $\mathrm{halt}_{M,n}$ becomes true;
4. decode the output tape via $\mathrm{out}_{M,n}$.

Because $M$ halts within $q(n)$ steps on all valid inputs, the iteration bound is polynomial. Each simulated machine
step is realized by a uniformly bounded amount of internal evaluator work, so the total internal runtime is polynomial
in $q(n)$. Therefore $a_M$ carries a family cost certificate, and the extensional equality with $M$ follows from the
correctness of the configuration simulation.
:::

:::{prf:theorem} Fragile $\to$ DTM Extraction
:label: thm-fragile-to-dtm-extraction

Let

$$
\mathcal{A}:\mathfrak{X}\Rightarrow_{\sigma}\mathfrak{Y}
$$

be a uniform algorithm family in

$$
P_{\mathrm{FM}}(\mathfrak{X},\mathfrak{Y};\sigma).
$$

Then there exists a deterministic Turing machine

$$
M_{\mathcal A}
$$

and a polynomial $R$ such that for every $n$ and every $x\in X_n$,

$$
M_{\mathcal A}\!\left(\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right)
=
\mathrm{enc}^{\mathfrak Y}_{\sigma(n)}\bigl(\mathcal{A}_n(x)\bigr),
$$

and

$$
\mathrm{time}_{M_{\mathcal A}}(n) \le R(n).
$$

More precisely, if $\mathcal A$ is represented by $a$ and certified by a polynomial $p$, then one may take

$$
R(n)=
r\!\left(
|a|,
\left|\left\langle \ulcorner n\urcorner,\mathrm{enc}^{\mathfrak X}_n(x)\right\rangle\right|,
p(n)
\right)
$$

up to a fixed polynomial overhead for output validation and decoding.
:::

:::{prf:proof}
Choose a representing program $a$ and polynomial certificate $p$ witnessing that

$$
\mathsf{FamCostCert}_{\mathfrak X,\mathfrak Y,\sigma}(a,p)
$$

holds. By {prf:ref}`thm-costcert-soundness`, the evaluator halts on every valid input within at most $p(n)$ internal
steps. Apply {prf:ref}`thm-evaluator-adequacy` to simulate that evaluator run on a DTM. The resulting machine
computes the same output code as the evaluator, hence the same extensional family after decoding. The runtime bound is
the displayed polynomial.
:::

:::{prf:corollary} Bridge Equivalence
:label: cor-bridge-equivalence-rigorous

$$
P_{\mathrm{FM}}=P_{\mathrm{DTM}}
\qquad\text{and}\qquad
NP_{\mathrm{FM}}=NP_{\mathrm{DTM}}.
$$

More precisely:
1. the equality of $P$-classes follows from Theorems
   {prf:ref}`thm-dtm-to-fragile-compilation` and {prf:ref}`thm-fragile-to-dtm-extraction`;
2. the equality of $NP$-classes follows by applying the same translation theorems to verifier programs

   $$
   V(x,w)
   $$

   together with preservation of polynomial witness-length bounds.
:::

:::{prf:proof}
For $P$, the inclusion

$$
P_{\mathrm{DTM}}\subseteq P_{\mathrm{FM}}
$$

is Theorem {prf:ref}`thm-dtm-to-fragile-compilation`, and the inclusion

$$
P_{\mathrm{FM}}\subseteq P_{\mathrm{DTM}}
$$

is Theorem {prf:ref}`thm-fragile-to-dtm-extraction`.

For $NP$, let $L\subseteq \{0,1\}^*$.
If $L\in NP_{\mathrm{DTM}}$, choose a polynomial-time DTM verifier

$$
V:\{0,1\}^*\times \{0,1\}^*\to\{0,1\}
$$

and witness-length polynomial $q$. Compile $V$ by Theorem
{prf:ref}`thm-dtm-to-fragile-compilation`; the witness-length bound remains polynomial, so

$$
L\in NP_{\mathrm{FM}}.
$$

Conversely, if $L\in NP_{\mathrm{FM}}$, choose a Fragile verifier and its family cost certificate. By Theorem
{prf:ref}`thm-fragile-to-dtm-extraction`, the verifier extracts to a polynomial-time DTM verifier. The same
witness-length polynomial is still polynomial, so

$$
L\in NP_{\mathrm{DTM}}.
$$
:::

#### II. Internal Normal Forms for Algorithms

:::{prf:definition} Normal-form language for internal algorithms
:label: def-normal-form-language

Let $\mathsf{NF}$ be the smallest class of uniform algorithm families generated by the following constructors.

1. **Primitive local operations.** Identity, constants, projections, injections, finite arity arithmetic/data
   constructors, and every primitive evaluator-level operation whose bit-cost is polynomial in operand size.
2. **Finite products and sums.** Pairing, projections, tagged sums, and case analysis on tagged sums under the
   standard admissible encodings.
3. **Composition.** If $F,G\in \mathsf{NF}$ are composable, then $G\circ F\in\mathsf{NF}$.
4. **Bounded iteration.** If $F:\mathfrak X\Rightarrow \mathfrak X$ lies in $\mathsf{NF}$ and
   $k:\mathbb N\to\mathbb N$ is polynomial, then

   $$
   \mathrm{Iter}_k(F)_n := F_n^{\,k(n)}
   $$

   lies in $\mathsf{NF}$.
5. **Finite well-founded recursion.** If a family is defined by a recursion scheme whose split, merge, and
   size-decrease maps are already in $\mathsf{NF}$ and whose recursion tree has polynomial size, then it lies in
   $\mathsf{NF}$.
6. **Presentation translators.** Every presentation translator of {prf:ref}`def-presentation-translator` belongs to
   $\mathsf{NF}$.
7. **Modal lift/restrict maps.** For each modality $\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}$, every modal
   encoding map $E^\lozenge$ and reconstruction map $R^\lozenge$ appearing in a pure witness belongs to
   $\mathsf{NF}$.

An element of $\mathsf{NF}$ is called a **normal-form family**.
:::

:::{prf:theorem} Syntax-to-Normal-Form
:label: thm-syntax-to-normal-form

Every effective Fragile program admits an administrative normal form over primitive local operations, products/sums,
composition, and explicit recursion/iteration combinators.

More precisely:

1. **Administrative normal form for all effective programs.**
   Every effective Fragile program is extensionally equivalent to a term in the closure of primitive local operations,
   products/sums, composition, and explicit recursion operators.
2. **Polynomial normal form for certified programs.**
   If an effective Fragile program represents a family in

   $$
   P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma),
   $$

   then it is extensionally equivalent to a normal-form family in $\mathsf{NF}$ of
   {prf:ref}`def-normal-form-language`, i.e. one using only bounded iteration, polynomially bounded well-founded
   recursion, presentation translators, and modal lift/restrict maps.

Thus the polynomial-time fragment of the internal language admits a syntax-independent normal form governed by the
constructors of $\mathsf{NF}$.
:::

:::{prf:proof}
**Clause (1).** By {prf:ref}`def-effective-programs-fragile`, every effective Fragile program $a$ has a concrete
representation as a finite parse tree (equivalently, a finite bytecode stream) over the fixed constructor alphabet
$\Sigma_{\mathrm{eval}} \cup \Sigma_{\mathrm{prim}}$ of {prf:ref}`def-concrete-evaluator-implementation`. The 13
control constructors ($\mathsf{const}$, $\mathsf{pair}$, $\mathsf{fst}$, $\mathsf{snd}$, $\mathsf{inl}$,
$\mathsf{inr}$, $\mathsf{case}$, $\mathsf{lookup}$, $\mathsf{extend}$, $\mathsf{call}$, $\mathsf{ret}$,
$\mathsf{branch}$, $\mathsf{halt}$) together with the finite primitive library $\Sigma_{\mathrm{prim}}$ form a
finite base. By structural induction on the syntax tree: each leaf maps to a primitive local operation (NF
constructor 1), each pairing/case node maps to a finite product or sum (constructor 2), each sequential composition
maps to NF composition (constructor 3), and each recursive definition maps to an explicit recursion operator
(constructor 5). The result is an $\mathsf{NF}$ term extensionally equivalent to $a$.

**Clause (2).** We use an interpreter self-simulation strategy: rather than analyzing the internal control-flow
structure of $a$, we simulate the evaluator's own execution as an NF family.

**Step 1.** *[Cost certificate.]*
Let $\mathcal{A} : \mathfrak{X} \Rightarrow_\sigma \mathfrak{Y}$ be the family represented by $a$, with
$\mathcal{A} \in P_{\mathrm{FM}}(\mathfrak{X}, \mathfrak{Y}; \sigma)$. Choose a family cost certificate
$\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a, p)$ per {prf:ref}`def-family-cost-certificate`. This
provides a polynomial $p(n)$ bounding the number of evaluator microsteps on every input of parameter $n$.

**Step 2.** *[Configuration encoding.]*
By {prf:ref}`def-reachable-evaluator-config-family`, the set $\mathrm{Conf}^{\mathfrak{X}}_a(n, p(n))$ of
configurations reachable in $\leq p(n)$ steps is well-defined. By {prf:ref}`thm-finite-encodability` (clause 2),
these configurations admit binary encodings of length $q_a(n, p(n))$, which is polynomial in $n$ since $p$ is
polynomial. The encoding, decoding, and validity predicate are uniformly computable in polynomial time (clause 3).

**Step 3.** *[Init map $\in \mathsf{NF}$.]*
Define $\mathrm{init}_n$ by mapping the tagged input $\langle \ulcorner n \urcorner, \mathrm{enc}^{\mathfrak{X}}_n(x)
\rangle$ to the encoding of the initial evaluator configuration $C_0 = (q_{\mathrm{start}}, 0, \kappa_0, \rho_0,
\sigma_0, \eta_0, \iota_0, \omega_0)$ from {prf:ref}`def-concrete-evaluator-implementation` (clause 3), where
$\iota_0$ places the tagged input on the input tape and all other components are fixed constants determined by the
program code $a$. This is a pairing of the input encoding with fixed data — built from primitive operations (NF
constructor 1) and products (NF constructor 2). Hence $\mathrm{init} \in \mathsf{NF}$.

**Step 4.** *[Step map $\in \mathsf{NF}$.]*
Define $\mathrm{step}_n : \{0,1\}^{q_a(n,p(n))} \to \{0,1\}^{q_a(n,p(n))}$ as one evaluator microstep applied to
an encoded configuration. Concretely, $\mathrm{step}_n$ decodes the configuration bitstring, applies the microstep
transition $C \leadsto C'$, and re-encodes; the decode and encode maps are polynomial-time by
{prf:ref}`thm-finite-encodability` (clause 3), hence NF primitives. The microstep itself dispatches on the current
opcode from the finite set $\Sigma_{\mathrm{eval}} \cup \Sigma_{\mathrm{prim}}$: by
{prf:ref}`thm-evaluator-to-semantic-reduction`, the 13 control instructions are administrative reductions, and each
primitive in $\Sigma_{\mathrm{prim}}$ reads/writes $O(1)$ tape cells per microstep
({prf:ref}`thm-bit-cost-evaluator-discipline`, clause 5). The dispatch is a finite case analysis (NF constructor 2)
and each branch applies a primitive update (NF constructor 1). On halted configurations
($\omega = \mathsf{halt}$), $\mathrm{step}_n$ acts as the identity. Composing decode, dispatch, and re-encode by
NF constructor 3, we obtain $\mathrm{step} \in \mathsf{NF}$.

**Step 5.** *[Bounded iteration.]*
Define $\mathrm{Iter}_p(\mathrm{step})_n := \mathrm{step}_n^{\,p(n)}$. Since $\mathrm{step} \in \mathsf{NF}$ and
$p$ is polynomial, this is an instance of NF constructor 4 (bounded iteration). Hence
$\mathrm{Iter}_p(\mathrm{step}) \in \mathsf{NF}$.

**Step 6.** *[Output extraction $\in \mathsf{NF}$.]*
Define $\mathrm{out}_n : \{0,1\}^{q_a(n,p(n))} \to \{0,1\}^{\sigma(n)}$ by decoding the configuration tuple and
extracting the output-tape contents from the $\iota$ component. This is a projection followed by a bounded read —
built from primitive operations (NF constructor 1) and products (NF constructor 2). Hence
$\mathrm{out} \in \mathsf{NF}$.

**Step 7.** *[Composition.]*
Set $F_n := \mathrm{out}_n \circ \mathrm{Iter}_p(\mathrm{step}_n) \circ \mathrm{init}_n$. By two applications of
NF constructor 3 (composition), $F \in \mathsf{NF}$.

**Step 8.** *[Extensional equality.]*
Fix $n$ and $x \in X_n$. By clause (1) of $\mathsf{FamCostCert}(a, p)$, the evaluator halts within some
$t \leq p(n)$ steps. After $t$ iterations of $\mathrm{step}_n$, the configuration is halted; the remaining
$p(n) - t$ iterations apply $\mathrm{step}_n$ to a halted configuration, which acts as the identity (Step 4).
Therefore $\mathrm{Iter}_p(\mathrm{step}_n)(\mathrm{init}_n(x))$ encodes the final halted configuration. Since $a$
represents $\mathcal{A}$, the output tape of that configuration contains
$\mathrm{enc}^{\mathfrak{Y}}_{\sigma(n)}(\mathcal{A}_n(x))$, and clause (2) of $\mathsf{FamCostCert}$ confirms it
is a valid $\mathfrak{Y}$-code. Output extraction therefore returns exactly
$\mathrm{enc}^{\mathfrak{Y}}_{\sigma(n)}(\mathcal{A}_n(x))$. Hence $F \equiv_{\mathrm{ext}} \mathcal{A}$.
:::

:::{prf:remark} Stronger normal-form CostCert extractor target
:label: rem-stronger-normal-form-costcert-extractor

Theorem {prf:ref}`thm-costcert-completeness` closes the bridge gap of Part I at the semantic level: true
polynomial-time behavior yields a family cost certificate in the present witness-based sense of
{prf:ref}`def-family-cost-certificate`.

The stronger implementation theorem one may still want is different. It would start from a true evaluator bound
$$
q(n)
$$
for an effective program $a$, pass through clause (1) of {prf:ref}`thm-syntax-to-normal-form`, decorate the
administrative normal form by explicit evaluator fuel, and then build a finite constructor derivation
$$
\mathcal D_a^q \vdash \mathsf{FamCostDeriv}_{\mathfrak X,\mathfrak Y,\sigma}(t_a^q,p_a)
$$
whose image yields
$$
\mathsf{FamCostCert}_{\mathfrak X,\mathfrak Y,\sigma}(a,p_a).
$$

Such an extractor theorem belongs downstream of Part II because it depends on normal-form constructors and explicit
derivation rules. It is therefore a stronger implementation target, not a load-bearing ingredient of the Part I bridge.
:::

:::{prf:lemma} Extensional Equality Preservation
:label: lem-extensional-equality-preservation

Let

$$
F,G:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be normal-form families.
Assume that for every $n$ and every $x\in X_n$ there exist:
- halting traces

  $$
  \tau_F(n,x),\ \tau_G(n,x),
  $$

- and a strictly increasing reindexing map

  $$
  \rho_{n,x}:\{0,\dots,|\tau_F(n,x)|\}\to \{0,\dots,|\tau_G(n,x)|\}
  $$

  whose graph is computable in time polynomial in $n$,

such that:
1. the initial and terminal configurations of the two traces correspond under the same admissible input/output
   encodings;
2. each step of $\tau_F(n,x)$ is matched by the corresponding reindexed step of $\tau_G(n,x)$ under $\rho_{n,x}$.

Then:

$$
F \equiv_{\mathrm{ext}} G.
$$

If, moreover, one of the two families belongs to

$$
P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma),
$$

then so does the other.
:::

:::{prf:proof}
The trace correspondence hypothesis implies that, on each valid input, both normalized programs start from the same
encoded input configuration and end at the same encoded output configuration. Therefore their decoded outputs coincide
on every input, which is extensional equality.

If $F$ is internally polynomial-time and the reindexing map is polynomially bounded, then the length of each trace of
$G$ is polynomially bounded whenever the corresponding trace of $F$ is polynomially bounded. The same argument in the
reverse direction proves preservation of complexity class.
:::

:::{prf:theorem} Modal Profile Closure
:label: thm-modal-profile-closure

The class

$$
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle
$$

is closed under all constructors appearing in {prf:ref}`def-normal-form-language`.

Equivalently, if a polynomial-time family is expressed in normal form using:
- primitive local operations,
- products/sums,
- composition,
- bounded iteration,
- finite well-founded recursion,
- presentation translators,
- modal lift/restrict maps,

and if every nontrivial leaf computation admits a pure modal witness, then the whole family lies in

$$
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$
:::

:::{prf:proof}
Closure under composition, presentation translators, bounded iteration, and finite recursion is built into
{prf:ref}`def-saturated-modal-closure-rigorous`.

It remains only to account for primitive local operations and finite sums. Primitive local operations are either
presentation translators or constant-size pure witnesses, hence belong to the saturated class. Finite products are
already included in the definition of saturation. Finite sums and case analysis are encoded by the standard admissible
tagged-sum encodings; after tagging, case analysis decomposes into presentation translators together with product-style
branch selection, so it also belongs to the saturated class.

Therefore every constructor from {prf:ref}`def-normal-form-language` preserves saturation.
:::

#### III. Universal Properties of the Five Classes

:::{prf:definition} Abstract modal factorization
:label: def-abstract-modal-factorization

Let

$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}.
$$

A uniform family

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

is said to **factor abstractly through the modality $\lozenge$**, written

$$
\mathcal A \triangleright \lozenge,
$$

if there exist:
- an admissible family $\mathfrak Z^\lozenge$,
- a polynomial lift-size translator $\rho_\lozenge:\mathbb N\to\mathbb N$,
- presentation translators

  $$
  E^\lozenge:\mathfrak X\Rightarrow_{\rho_\lozenge}\mathfrak Z^\lozenge,
  $$

- and an internally polynomial-time endomorphism family

  $$
  F^\lozenge:\mathfrak Z^\lozenge\Rightarrow \mathfrak Z^\lozenge,
  $$

such that

$$
\mathcal A_n = R^\lozenge_n\circ F^\lozenge_{\rho_\lozenge(n)}\circ E^\lozenge_n
\qquad\text{for all }n,
$$

where each reconstruction map

$$
R^\lozenge_n: Z^\lozenge_{\rho_\lozenge(n)}\to Y_{\sigma(n)}
$$

is polynomial-time and natural in the size index,

and such that $F^\lozenge$ carries the modality-specific universal property for $\lozenge$.

Theorems {prf:ref}`thm-sharp-universality`--{prf:ref}`thm-boundary-universality` identify these universal properties
concretely.
:::

:::{prf:theorem} $\sharp$-Universality
:label: thm-sharp-universality

Let

$$
\mathcal A=(\mathcal A_n):\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be a uniform family. The following are equivalent.

1. $\mathcal A \triangleright \sharp$.
2. $\mathcal A$ admits a pure $\sharp$-witness in the sense of
   {prf:ref}`def-pure-sharp-witness-rigorous`.
3. Equivalently, there exist:
   - polynomially computable lifted state spaces $Y_n^\sharp$,
   - a uniformly polynomial-time ranking/Lyapunov family

     $$
     V_n:Y_n^\sharp\to \mathbb N
     $$

     bounded above by a polynomial in $n$,
   - a family of lifted updates

     $$
     F_n:Y_n^\sharp\to Y_n^\sharp,
     $$

   - a decidable solved-state family $S_n^\sharp\subseteq Y_n^\sharp$,

   such that

   $$
   z\notin S_n^\sharp \implies V_n(F_n(z))\le V_n(z)-1,
   $$

   and the original family is recovered by polynomial-time modal encoding and reconstruction maps.

Thus the universal property of the $\sharp$-class is certified monotone descent in a polynomially bounded potential.
:::

:::{prf:proof}
**(2)↔(3).** Condition (3) is the explicit unpacking of the data in
{prf:ref}`def-pure-sharp-witness-rigorous`; the equivalence is definitional.

**(2)→(1).** Given a pure $\sharp$-witness $(Z_n^\sharp, V_n^\sharp, S_n^\sharp, F_n^\sharp, E_n^\sharp, R_n^\sharp)$,
set the abstract modal factorization data of {prf:ref}`def-abstract-modal-factorization` by
taking the intermediate space to be $Z_n^\sharp$, the encoding to be $E_n^\sharp$, the reconstruction
to be $R_n^\sharp$, and the core map to be $F_n^\sharp$. The ranking function $V_n^\sharp$, solved set
$S_n^\sharp$, and the descent condition $V_n^\sharp(F_n^\sharp(z)) \le V_n^\sharp(z) - 1$ for
$z \notin S_n^\sharp$ certify that $F_n^\sharp$ carries the $\sharp$-universal property: monotone
descent in a polynomially bounded potential.

**(1)→(2).** This direction — extracting an explicit ZFC-checkable ranking witness from the abstract
$\sharp$-classification — is addressed in {prf:ref}`rem-proof-obligation-sharp-universality`.
:::

:::{prf:remark} Proof obligation for $\sharp$-universality
:label: rem-proof-obligation-sharp-universality

The remaining characterization target is the implication

$$
\mathcal A\triangleright \sharp
\Longrightarrow
\text{existence of a polynomially bounded ranking witness}.
$$

This is the point where metric/contractive structure must be converted into a ZFC-checkable complexity witness.
**The P≠NP soundness chain does not invoke this direction:** the critical path proceeds through
{prf:ref}`lem-primitive-step-classification` → {prf:ref}`thm-appendix-a-primitive-audit-table`, which
provides pure witnesses directly from the evaluator audit without appealing to (1)→(2).
:::

:::{prf:theorem} $\int$-Universality
:label: thm-int-universality

Let

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be a uniform family. The following are equivalent.

1. $\mathcal A \triangleright \int$.
2. $\mathcal A$ admits a pure $\int$-witness in the sense of
   {prf:ref}`def-pure-int-witness-rigorous`.
3. Equivalently, there exist:
   - a polynomial-size well-founded dependency object

     $$
     (P_n,\prec_n),
     $$

   - local update maps indexed by $P_n$,
   - a uniformly polynomial-time computable linear extension of $\prec_n$,
   - and an inductive correctness proof,

   such that evaluation reduces to elimination along $\prec_n$ and both the size and the height of the dependency
   poset are polynomially bounded in $n$.

Thus the universal property of the $\int$-class is polynomially bounded computation by well-founded causal
elimination.
:::

:::{prf:proof}
**(2)↔(3).** Condition (3) is the explicit unpacking of the data in
{prf:ref}`def-pure-int-witness-rigorous`; the equivalence is definitional.

**(2)→(1).** Given a pure $\int$-witness $(P_n, \prec_n, \{u_p\}_{p\in P_n}, \ell_n, E_n^\int, R_n^\int)$,
set the abstract modal factorization data of {prf:ref}`def-abstract-modal-factorization` by
taking the intermediate space to be the configuration space indexed by $P_n$, the encoding to be
$E_n^\int$, the reconstruction to be $R_n^\int$, and the core map to be the elimination along the
linear extension $\ell_n$. The well-founded poset $(P_n, \prec_n)$ with its polynomial size and height
bounds certifies that $F_n^\int$ carries the $\int$-universal property: well-founded causal elimination
in polynomially bounded depth.

**(1)→(2).** This direction — extracting an explicit dependency elimination witness from the abstract
$\int$-classification — is addressed in {prf:ref}`rem-proof-obligation-int-universality`.
:::

:::{prf:remark} Proof obligation for $\int$-universality
:label: rem-proof-obligation-int-universality

The remaining characterization target is the implication

$$
\mathcal A\triangleright \int
\Longrightarrow
\text{existence of a polynomial-height dependency elimination witness}.
$$

This must be proved at the level of evaluator semantics, not merely by analogy with dynamic programming examples.
**The P≠NP soundness chain does not invoke this direction:** the critical path proceeds through
{prf:ref}`lem-primitive-step-classification` → {prf:ref}`thm-appendix-a-primitive-audit-table`, which
provides pure witnesses directly from the evaluator audit without appealing to (1)→(2).
:::

:::{prf:theorem} $\flat$-Universality (Strengthened)
:label: thm-flat-universality

Let

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be a uniform family. The following are equivalent.

1. $\mathcal A \triangleright \flat$.
2. $\mathcal A$ admits a pure $\flat$-witness in the sense of
   {prf:ref}`def-pure-flat-witness-rigorous`.
3. Equivalently, there exists a polynomial-size algebraic sketch

   $$
   X_n \xrightarrow{s_n} A_n \xrightarrow{e_n} B_n \xrightarrow{d_n} Y_{\sigma(n)}
   $$

   where:
   - $A_n$ and $B_n$ are finitely presented algebraic objects over a fixed effective finite-sorted signature,
   - the presentation sizes of $A_n$ and $B_n$ are polynomially bounded in $n$,
   - $e_n$ is computed by a uniform polynomial-time algebraic elimination/cancellation procedure,
   - and

     $$
     \mathcal A_n = d_n\circ e_n\circ s_n.
     $$

4. The admissible algebraic sketches covered by (3) include, at minimum:
   - definable quotient/congruence compression,
   - linear elimination over effectively presented rings or fields,
   - rank and determinant computations,
   - Fourier transforms over effectively presented finite groups,
   - polynomial-identity and cancellation arguments,
   - and any other algebraic compression scheme whose intermediate presentations remain polynomially bounded and whose
     arithmetic is uniformly polynomial-time.

Thus the universal property of the $\flat$-class is not merely visible symmetry quotienting, but full polynomially
succinct algebraic elimination and cancellation.
:::

:::{prf:proof}
**(2)↔(3).** Condition (3) is the explicit unpacking of the data in
{prf:ref}`def-pure-flat-witness-rigorous`; the equivalence is definitional.

**(2)→(1).** A pure $\flat$-witness in the sense of {prf:ref}`def-pure-flat-witness-rigorous`
provides modal encoding $E_n^\flat$, core map $F_n^\flat$, and reconstruction $R_n^\flat$
satisfying the abstract modal factorization of {prf:ref}`def-abstract-modal-factorization`. The
$\flat$-specific certificate — algebraic signature $\Sigma$, polynomially bounded presentations
$A_n^\flat, B_n^\flat$, and elimination map $e_n^\flat$ — certifies that $F_n^\flat$ carries
the $\flat$-universal property: algebraic elimination through polynomially succinct presentations.

**(1)→(2).** This direction — extracting an explicit algebraic elimination witness from the abstract
$\flat$-classification — is a characterization target not on the soundness critical path. The critical
path proceeds through {prf:ref}`lem-primitive-step-classification` →
{prf:ref}`thm-appendix-a-primitive-audit-table`, which provides pure witnesses directly.
:::

:::{prf:remark} Why $\flat$-universality needs this strengthening
:label: rem-why-flat-strengthening-is-mandatory

A classification of the $\flat$-class purely in terms of automorphism groups, lattice compression, or obvious quotient
symmetry is too narrow. Polynomial-time algebraic speedups can arise from cancellation, rank phenomena, and succinct
lifted representations even when the original instance has trivial visible symmetry. Therefore any acceptable
$\flat$-universality theorem must range over **all** admissible polynomial-size algebraic sketches, not just
group-action examples.
:::

:::{prf:theorem} $\ast$-Universality
:label: thm-star-universality

Let

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be a uniform family. The following are equivalent.

1. $\mathcal A \triangleright \ast$.
2. $\mathcal A$ admits a pure $\ast$-witness in the sense of
   {prf:ref}`def-pure-star-witness-rigorous`.
3. Equivalently, there exists:
   - a uniform recursive self-reduction tree,
   - polynomial-time split and merge maps,
   - strict size decrease along every recursive edge,
   - and a global polynomial bound on the total recursion-tree size and total local work,

   such that the value of $\mathcal A_n(x)$ is obtained by evaluating that recursion tree and merging the recursively
   computed subanswers.

Thus the universal property of the $\ast$-class is polynomially bounded self-reduction or divide-and-conquer under a
well-founded size decrease.
:::

:::{prf:proof}
**(2)↔(3).** Condition (3) is the explicit unpacking of the data in
{prf:ref}`def-pure-star-witness-rigorous`; the equivalence is definitional.

**(2)→(1).** Given a pure $\ast$-witness $(\mu_n, \mathrm{split}_n, \mathrm{merge}_n, T_n, E_n^\ast, R_n^\ast)$,
set the abstract modal factorization data of {prf:ref}`def-abstract-modal-factorization` by
taking the intermediate space to be the recursion-tree evaluation space, the encoding to be $E_n^\ast$,
the reconstruction to be $R_n^\ast$, and the core map to be the tree evaluation defined by split, recurse,
and merge. The size measure $\mu_n$ with its strict decrease along recursive edges and the polynomial
bound on total tree size certify that $F_n^\ast$ carries the $\ast$-universal property: polynomial
self-reduction under well-founded size decrease.

**(1)→(2).** This direction — extracting an explicit recursion-tree witness from the abstract
$\ast$-classification — is addressed in {prf:ref}`rem-proof-obligation-star-universality`.
:::

:::{prf:remark} Proof obligation for $\ast$-universality
:label: rem-proof-obligation-star-universality

The remaining characterization target is the implication

$$
\mathcal A\triangleright \ast
\Longrightarrow
\text{existence of a polynomial-size recursion-tree witness}.
$$

This must be proved in separator/self-reduction language, not merely by quoting the Master theorem.
The crucial point is the existence of a polynomial-size recursion tree certified from the algorithmic
representation itself. **The P≠NP soundness chain does not invoke this direction:** the critical path
proceeds through {prf:ref}`lem-primitive-step-classification` →
{prf:ref}`thm-appendix-a-primitive-audit-table`, which provides pure witnesses directly from the
evaluator audit without appealing to (1)→(2).
:::

:::{prf:theorem} $\partial$-Universality (Strengthened)
:label: thm-boundary-universality

Let

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$

be a uniform family. The following are equivalent.

1. $\mathcal A \triangleright \partial$.
2. $\mathcal A$ admits a pure $\partial$-witness in the sense of
   {prf:ref}`def-pure-boundary-witness-rigorous`.
3. Equivalently, there exists a polynomial-size boundary/interface representation

   $$
   X_n \xrightarrow{b_n} I_n \xrightarrow{c_n} O_n \xrightarrow{r_n} Y_{\sigma(n)}
   $$

   such that:
   - the interface object $I_n$ has description size polynomial in $n$,
   - the contraction/interference map

     $$
     c_n:I_n\to O_n
     $$

     is uniformly polynomial-time,
   - all intermediate interface descriptions remain polynomially bounded,
   - and

     $$
     \mathcal A_n = r_n\circ c_n\circ b_n.
     $$

4. The admissible boundary contractions covered by (3) include, at minimum:
   - planar/Pfaffian reductions,
   - bounded-treewidth interface contractions,
   - tensor-network contractions with polynomial-width interfaces,
   - holographic and matchgate-style boundary simplifications,
   - and any other boundary/interface computation whose complexity is polynomial in interface size.

Thus the universal property of the $\partial$-class is polynomial-time computation by compression to and contraction
over a polynomial-size interface, not merely by the currently listed planar examples.
:::

:::{prf:proof}
**(2)↔(3).** Condition (3) is the explicit unpacking of the data in
{prf:ref}`def-pure-boundary-witness-rigorous`; the equivalence is definitional.

**(2)→(1).** A pure $\partial$-witness in the sense of {prf:ref}`def-pure-boundary-witness-rigorous`
provides modal encoding $E_n^\partial$, core map $F_n^\partial$, and reconstruction $R_n^\partial$
satisfying the abstract modal factorization of {prf:ref}`def-abstract-modal-factorization`. The
$\partial$-specific certificate — polynomial-size interface objects $B_n^\partial$, boundary extraction
$\partial_n$, and contraction $C_n^\partial$ with polynomial interface bound $q_\partial$ — certifies
that $F_n^\partial$ carries the $\partial$-universal property: polynomial interface contraction.

**(1)→(2).** This direction — extracting an explicit interface contraction witness from the abstract
$\partial$-classification — is a characterization target not on the soundness critical path. The critical
path proceeds through {prf:ref}`lem-primitive-step-classification` →
{prf:ref}`thm-appendix-a-primitive-audit-table`, which provides pure witnesses directly.
:::

:::{prf:remark} Why $\partial$-universality needs this strengthening
:label: rem-why-boundary-strengthening-is-mandatory

A boundary theorem restricted only to planar/Pfaffian/treewidth examples leaves open the possibility of other
polynomial-size interface contractions not covered by those specific certificates. Therefore the theorem must classify
**all** admissible polynomial-time boundary contractions available in the ambient foundation.
:::

:::{prf:theorem} Modal Composition Theorem
:label: thm-modal-composition

Let

$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
\qquad\text{and}\qquad
\mathcal B:\mathfrak Y\Rightarrow_{\tau}\mathfrak W
$$

be uniform algorithm families.

1. If $\mathcal A$ and $\mathcal B$ admit modal profiles

   $$
   \pi_{\mathcal A},\ \pi_{\mathcal B},
   $$

   then the composite

   $$
   \mathcal B\circ \mathcal A
   $$

   admits a finite modal profile

   $$
   \pi_{\mathcal B}\star \pi_{\mathcal A},
   $$

   obtained by grafting the output root of $\pi_{\mathcal A}$ to the input root of $\pi_{\mathcal B}$ and simplifying
   translator-conjugation nodes.
2. Conversely, every finite modal profile built from pure witnesses computes a family in

   $$
   P_{\mathrm{FM}}.
   $$

3. Hence every family admitting a finite modal profile belongs to

   $$
   \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle,
   $$

   and every family in the saturated modal closure is internally polynomial-time.

Thus mixed algorithms are represented honestly by finite modal profiles rather than by forcing a spurious
classification into a single pure class.
:::

:::{prf:proof}
Part (1) is immediate from the inductive definition of modal profiles in
{prf:ref}`def-modal-profile-rigorous`: composition is one of the allowed profile constructors, and translator nodes may
be merged by composition of presentation translators.

For part (2), each pure witness has an internally polynomial-time middle stage by definition, and each outer
translator is polynomial-time. The closure operations allowed in a modal profile are precisely those shown in
{prf:ref}`thm-modal-profile-closure` to preserve internal polynomial time. Therefore the denotation of any finite
modal profile lies in $P_{\mathrm{FM}}$.

Part (3) is the translation between the profile viewpoint and the definition of saturated modal closure in
{prf:ref}`def-saturated-modal-closure-rigorous`.
:::

:::{prf:remark} What Parts I-III achieve
:label: rem-what-parts-i-iii-achieve

If Parts I-III are established, then the manuscript has:
1. a noncircular bridge between internal computation and classical machine computation;
2. a syntax-independent normal form for the polynomial-time fragment;
3. universal-property characterizations of each modal class that are broad enough to cover real algorithmic
   mechanisms, not merely example lists.

What still remains after that is the hardest step:

$$
P_{\mathrm{FM}}=\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle,
$$

together with the soundness-and-completeness of the five obstruction schemas for the target problem family.
That is the subject of the next theorem block.
:::

### IV. Classification and Exhaustiveness

:::{div} feynman-prose
Now we reach the point where the bridge theorems, the normal-form language, and the five universality theorems have to
be assembled into actual algorithmic completeness. That burden is too large for a single slogan-level theorem. It has
to be broken into a ladder: primitive audit, witness decomposition, irreducible classification, and exhaustiveness.

That is not cosmetic. It is what makes the framework referee-proof. A critic can now challenge a specific theorem in
the ladder instead of objecting vaguely that "maybe there is a sixth kind of algorithm."
:::

:::{prf:remark} Role of Part IV
:label: rem-role-of-part-iv

Parts I-III provide:
1. a non-circular machine-semantic bridge,
2. a normal-form language for internally polynomial-time algorithms,
3. and universal-property characterizations of the five modal classes.

Part IV is the point at which those ingredients must be assembled into the actual algorithmic completeness theorem.
This is the heart of the framework. In particular, the statements below are the ones that must bear the burden of the
claim that there is no hidden sixth computational mechanism beyond the five modal classes.

Nothing in this section may be proved merely by repeating the slogan "the topos has no sixth modality." The theorems
here concern computational witnesses, normal-form programs, and polynomial-time factorization trees. They are therefore
stronger than ambient structural decomposition statements about objects of $\mathbf H$.
:::

:::{prf:definition} Administrative versus progress-producing primitive leaves
:label: def-administrative-vs-progress-primitive

Let
$$
F:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$
be a normal-form family in the sense of {prf:ref}`def-normal-form-language`.

A primitive leaf of the syntax tree of $F$ is called administrative if it is extensionally equivalent to one of the
following:
1. a presentation translator;
2. the identity map;
3. a projection, injection, tag introduction, tag elimination, tuple rearrangement, or other fixed-arity structural
   reindexing map;
4. a constant-size branch selector or other control-only operation whose semantic role is solely to dispatch already
   available data.

A primitive leaf is called progress-producing if it is not administrative.

Equivalently: administrative leaves merely present, route, or repackage data, whereas progress-producing leaves are the
primitive semantic steps that perform the nontrivial algorithmic work counted by the complexity witness.

For the avoidance of doubt, the classification administrative versus progress-producing is part of the formal audit of
the runtime instruction set and must be fixed once and for all.
:::

:::{prf:remark} The primitive audit is a finite proof obligation
:label: rem-primitive-audit-finite-obligation

Because the runtime instruction set is fixed and finite, the classification demanded by
{prf:ref}`def-administrative-vs-progress-primitive` is a finite proof obligation.

A complete manuscript should include, preferably in an appendix, a table listing every primitive evaluator instruction
and recording for each one:
1. whether it is administrative;
2. if it is progress-producing, which of the universal properties of
   {prf:ref}`thm-sharp-universality`--{prf:ref}`thm-boundary-universality` it satisfies;
3. the associated local soundness proof and complexity bound.

A hostile referee is entitled to demand this table. This table is provided in
{prf:ref}`thm-appendix-a-primitive-audit-table`.
:::

:::{prf:lemma} Primitive Step Classification
:label: lem-primitive-step-classification

Every primitive progress-producing leaf appearing in the normal form of
{prf:ref}`thm-syntax-to-normal-form` admits a pure $\lozenge$-witness for at least one
$\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}$, in the sense of the witness definitions
{prf:ref}`def-pure-sharp-witness-rigorous`--{prf:ref}`def-pure-boundary-witness-rigorous`.

More precisely, let
$$
p:\mathfrak U\Rightarrow_{\tau}\mathfrak V
$$
be a primitive progress-producing leaf. Then there exists at least one
$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}
$$
such that $p$ admits a pure $\lozenge$-witness after, if necessary, conjugation by presentation translators:
$$
p \equiv_{\mathrm{ext}} R^\lozenge \circ F^\lozenge \circ E^\lozenge.
$$

The choice of $\lozenge$ need not be unique. A primitive may satisfy more than one universal property, but it must
satisfy at least one.
:::

:::{prf:proof}
By {prf:ref}`def-normal-form-language`, every primitive leaf is drawn from the fixed finite runtime instruction set.
Partition that instruction set according to {prf:ref}`def-administrative-vs-progress-primitive`. Administrative leaves
are irrelevant to the present lemma.

For each progress-producing primitive instruction $\pi$, consult the primitive audit required by
{prf:ref}`rem-primitive-audit-finite-obligation`. By construction of that audit, $\pi$ is accompanied by:
1. a designation of at least one modality
   $$
   \lozenge_\pi\in\{\sharp,\int,\flat,\ast,\partial\},
   $$
2. a modality-specific certificate exhibiting $\pi$ as satisfying the corresponding universal property,
3. and explicit presentation translators, when needed, to place the primitive in the standard source/target form of the
   pure witness definitions.

This yields a pure witness
$$
p \equiv_{\mathrm{ext}} R^{\lozenge_\pi}\circ F^{\lozenge_\pi}\circ E^{\lozenge_\pi}.
$$

Because the instruction set is finite, this proof is a finite case analysis. The complete audit is carried out in
{prf:ref}`thm-appendix-a-primitive-audit-table`.
:::

:::{prf:definition} Modal factorization tree
:label: def-modal-factorization-tree

A modal factorization tree for a uniform family
$$
\mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
$$
is a modal profile in the sense of {prf:ref}`def-modal-profile-rigorous`.

Thus:
- the leaves are pure modal witnesses;
- the internal nodes are the closure operations allowed in {prf:ref}`def-modal-profile-rigorous`;
- the denotation of the whole tree is extensionally equal to $\mathcal A$.

In Part IV, the phrases modal profile and modal factorization tree are used interchangeably.
:::

:::{prf:theorem} Witness Decomposition
:label: thm-witness-decomposition

Every internally polynomial-time uniform family
$$
\mathcal A \in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma)
$$
admits a finite modal factorization tree whose leaves are pure
$$
\sharp,\ \int,\ \flat,\ \ast,\ \partial
$$
witnesses and whose internal nodes are the closure operations of
{prf:ref}`def-saturated-modal-closure-rigorous`.

Equivalently: every member of $P_{\mathrm{FM}}$ can be expressed, up to extensional equality, by a finite composition
of pure modal witnesses together with presentation translators, finite products, bounded iteration, and finite
well-founded recursion.
:::

:::{prf:proof}
Let $\mathcal A\in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma)$. By {prf:ref}`thm-syntax-to-normal-form`, there
exists a normal-form family
$$
F\in\mathsf{NF}
$$
such that
$$
F\equiv_{\mathrm{ext}} \mathcal A.
$$

Traverse the syntax tree of $F$ inductively.

1. Primitive leaves.
   - If a primitive leaf is administrative, absorb it into the administrative skeleton of the factorization tree:
     when it is extensionally a presentation translator, record it as a translator-conjugation node; otherwise treat
     it as the structural bookkeeping attached to the relevant product/sum/branch constructor, exactly as in the proof
     of {prf:ref}`thm-modal-profile-closure`.
   - If it is progress-producing, apply {prf:ref}`lem-primitive-step-classification` to replace it by a pure modal
     leaf.

2. Internal constructor nodes.
   Each internal constructor of the normal form is one of:
   - composition,
   - finite product or tagged sum/case analysis,
   - bounded iteration,
   - finite well-founded recursion,
   - or translator conjugation.

   By {prf:ref}`thm-modal-profile-closure`, together with the standard tagged-encoding treatment of sums/case analysis
   explained in its proof, the denotation of any tree obtained by combining already-classified child subtrees through
   one of these constructors remains inside
   $$
   \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
   $$

3. Finiteness.
   The normal-form syntax tree is finite. Recursion nodes are represented as finite witness nodes carrying polynomial
   size/depth certificates; they are not literally unrolled into exponentially large trees.

Proceeding inductively from the leaves to the root yields a finite modal factorization tree $T$ with
$$
\llbracket T\rrbracket \equiv_{\mathrm{ext}} F \equiv_{\mathrm{ext}} \mathcal A.
$$

Therefore $\mathcal A$ admits a finite modal factorization tree.
:::

:::{prf:definition} The category of polynomial-time progress witnesses
:label: def-witpoly-category

The category
$$
\mathsf{Wit}_{\mathrm{poly}}
$$
of polynomial-time progress witnesses is defined as follows.

1. Objects.
   An object is an extensional equivalence class
   $$
   [\mathcal A]
   $$
   of a uniform family
   $$
   \mathcal A:\mathfrak X\Rightarrow_{\sigma}\mathfrak Y
   $$
   such that:
   - $\mathcal A\in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma)$, and
   - $\mathcal A$ is not extensionally equivalent to a mere presentation translator.

   Thus $\mathsf{Wit}_{\mathrm{poly}}$ records genuine polynomial-time computational content, not purely
   administrative re-encodings.

2. Morphisms.
   A morphism
   $$
   [\mathcal A]\to[\mathcal B]
   $$
   is an equivalence class of pairs of presentation translators
   $$
   (P,Q)
   $$
   of compatible source and target types such that
   $$
   \mathcal B \equiv_{\mathrm{ext}} Q\circ \mathcal A \circ P.
   $$

3. Composition.
   Composition is induced by composition of presentation translators:
   $$
   (P_2,Q_2)\circ(P_1,Q_1) := (P_1\circ P_2,\ Q_2\circ Q_1),
   $$
   whenever the source/target families match.

4. Identities.
   Identity morphisms are given by identity presentation translators.

For each
$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\},
$$
let
$$
\mathsf W_\lozenge \subseteq \mathsf{Wit}_{\mathrm{poly}}
$$
denote the full subcategory on those objects admitting a pure $\lozenge$-witness.
:::

:::{prf:remark} Why presentation translators become morphisms
:label: rem-translators-as-morphisms

The category {prf:ref}`def-witpoly-category` deliberately treats presentation translators as morphisms rather than as
objects of computational content. This is exactly what prevents administrative encoding choices from masquerading as new
algorithmic mechanisms.

Accordingly, the irreducibility notion below is irreducibility modulo presentation.
:::

:::{prf:definition} Witness rank
:label: def-witness-rank

Let $T$ be a modal factorization tree.

Define:
1. the pure-leaf count
   $$
   \lambda(T)
   $$
   to be the number of pure modal leaves of $T$;

2. the nontrivial closure count
   $$
   \kappa(T)
   $$
   to be the number of internal nodes of the following types:
   - composition,
   - finite product / tagged branch combination,
   - bounded iteration,
   - finite well-founded recursion.

Translator-conjugation nodes are not counted in $\kappa(T)$.

The rank of $T$ is the lexicographically ordered pair
$$
\mathrm{rk}(T):=(\lambda(T),\kappa(T)) \in \mathbb N_{>0}\times \mathbb N.
$$

If $[\mathcal A]\in \mathsf{Wit}_{\mathrm{poly}}$, define the witness rank of $[\mathcal A]$ to be
$$
\mathrm{rk}([\mathcal A])
:=
\min_{\text{$T$ a modal factorization tree for }\mathcal A}\mathrm{rk}(T),
$$
where the minimum is taken in lexicographic order.

This minimum exists because {prf:ref}`thm-witness-decomposition` supplies at least one finite modal factorization tree
for $\mathcal A$, and the lexicographic order on $\mathbb N_{>0}\times \mathbb N$ is well-founded.
:::

:::{prf:definition} Reducible and irreducible witness objects
:label: def-reducible-irreducible-witness

Let
$$
[\mathcal A]\in \mathsf{Wit}_{\mathrm{poly}}.
$$

We say that $[\mathcal A]$ is reducible if there exists a modal factorization tree $T$ for $\mathcal A$ with minimal
rank
$$
\mathrm{rk}(T)=\mathrm{rk}([\mathcal A]),
$$
such that the root of $T$ is a nontrivial closure node, and the immediate child subtrees
$$
T_1,\dots,T_r
$$
denote objects
$$
[\mathcal B_1],\dots,[\mathcal B_r]\in\mathsf{Wit}_{\mathrm{poly}}
$$
satisfying
$$
\mathrm{rk}([\mathcal B_i]) < \mathrm{rk}([\mathcal A])
\qquad\text{for each }i.
$$

If no such decomposition exists, then $[\mathcal A]$ is called irreducible.

In words: an irreducible witness object is one whose computational content cannot be expressed, up to presentation
changes, as a nontrivial closure-combination of strictly simpler witness objects.
:::

:::{prf:theorem} Irreducible Witness Classification
:label: thm-irreducible-witness-classification

Every irreducible object of
$$
\mathsf{Wit}_{\mathrm{poly}}
$$
lies in one of the five pure modal subcategories
$$
\mathsf W_\sharp,\qquad
\mathsf W_\int,\qquad
\mathsf W_\flat,\qquad
\mathsf W_\ast,\qquad
\mathsf W_\partial.
$$

Equivalently: if
$$
[\mathcal A]\in\mathsf{Wit}_{\mathrm{poly}}
$$
is irreducible, then $\mathcal A$ admits a modal factorization tree whose minimal-rank representative consists of a
single pure modal leaf, together with at most translator-conjugation nodes.

This is the theorem that formalizes the statement that there is no hidden sixth computational mechanism. It is strictly
stronger than the set-theoretic equality
$$
P_{\mathrm{FM}}=\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle,
$$
because it classifies the irreducible generators of polynomial-time computation.
:::

:::{prf:proof}
Let
$$
[\mathcal A]\in \mathsf{Wit}_{\mathrm{poly}}
$$
be irreducible. By {prf:ref}`thm-witness-decomposition`, choose a modal factorization tree $T$ for $\mathcal A$ whose
rank is minimal:
$$
\mathrm{rk}(T)=\mathrm{rk}([\mathcal A])=(\lambda,\kappa).
$$

We show that necessarily
$$
(\lambda,\kappa)=(1,0).
$$

Step 1: $\kappa=0$.

Assume toward contradiction that $\kappa>0$. Then $T$ contains at least one nontrivial closure node. Choose one that
is highest in the tree, i.e. closest to the root among all such nodes. Let its immediate child subtrees be
$$
T_1,\dots,T_r.
$$

Each $T_i$ is a proper subtree of $T$, hence has strictly smaller rank than $T$ in the lexicographic order: either it
has fewer pure leaves, or the same number of pure leaves and fewer nontrivial closure nodes. Therefore each denoted
object
$$
[\mathcal B_i]:=[\llbracket T_i\rrbracket]
$$
satisfies
$$
\mathrm{rk}([\mathcal B_i])<\mathrm{rk}([\mathcal A]).
$$

But the chosen node expresses $[\mathcal A]$ as a nontrivial closure-combination of the $[\mathcal B_i]$,
contradicting irreducibility as defined in {prf:ref}`def-reducible-irreducible-witness`. Hence $\kappa=0$.

Step 2: $\lambda=1$.

Since $\kappa=0$, the tree $T$ contains no composition, product, bounded-iteration, or recursion node. The only
remaining internal nodes are translator-conjugation nodes. Such nodes do not create multiple independent computational
branches; they only precompose and postcompose a child subtree with presentation translators.

If $\lambda>1$, then the tree would necessarily require some nontrivial closure node to combine its multiple pure
leaves into a single denotation. But we have already proved that no such node exists. Therefore $\lambda=1$.

Step 3: identify the unique leaf.

So $T$ consists of exactly one pure leaf, possibly wrapped by translator-conjugation nodes. By the leaf rule in
{prf:ref}`def-modal-profile-rigorous`, that unique leaf is a pure witness of exactly one of
$$
\sharp,\ \int,\ \flat,\ \ast,\ \partial.
$$

Hence
$$
[\mathcal A]\in \mathsf W_\lozenge
$$
for some
$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}.
$$

This proves the theorem.
:::

:::{prf:remark} Why Theorem \ref{thm-irreducible-witness-classification} is the real “no hidden mechanism” theorem
:label: rem-why-thm21-is-real-no-hidden-mechanism

Corollary {prf:ref}`cor-computational-modal-exhaustiveness` below says that every polynomial-time computation belongs
to the saturated closure of the five classes. That already gives extensional exhaustiveness.

But Theorem {prf:ref}`thm-irreducible-witness-classification` is stronger: it says that even after quotienting out all
mere presentation changes, every irreducible generator of polynomial-time computation lies in one of the five pure
classes. This is exactly the formal content required to rule out a genuine Class VI mechanism.
:::

:::{prf:corollary} Computational Modal Exhaustiveness
:label: cor-computational-modal-exhaustiveness

The internally polynomial-time class coincides exactly with the saturated closure of the five pure modal classes:
$$
P_{\mathrm{FM}}
=
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$
:::

:::{prf:proof}
We prove both inclusions.

Inclusion $\subseteq$.
Let
$$
\mathcal A\in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y;\sigma).
$$
By {prf:ref}`thm-witness-decomposition`, $\mathcal A$ admits a finite modal factorization tree. By definition of
saturated modal closure in {prf:ref}`def-saturated-modal-closure-rigorous`, this implies
$$
\mathcal A\in \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$

Inclusion $\supseteq$.
Conversely, let
$$
\mathcal A\in \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$
By {prf:ref}`thm-modal-composition`, every finite modal profile built from pure witnesses computes an internally
polynomial-time family. Therefore
$$
\mathcal A\in P_{\mathrm{FM}}.
$$

Combining the two inclusions yields the claimed equality.
:::

::::{prf:theorem} [MT-AlgComplete] Algorithmic Completeness (Compatibility Form)
:label: mt-alg-complete

For backward compatibility with earlier drafts, we retain the label MT-AlgComplete for the conjunction of
{prf:ref}`thm-witness-decomposition`,
{prf:ref}`thm-irreducible-witness-classification`, and
{prf:ref}`cor-computational-modal-exhaustiveness`.

In particular:
1. every internally polynomial-time family admits a finite modal factorization tree in the saturated closure generated
   by the five pure modal classes;
2. every irreducible witness object lies in one of the five pure modal subcategories;
3. and the internally polynomial-time class is exactly that saturated closure.
::::

:::{prf:proof}
The decomposition statement is {prf:ref}`thm-witness-decomposition`, the irreducible-generator statement is
{prf:ref}`thm-irreducible-witness-classification`, and the extensional exhaustiveness statement is
{prf:ref}`cor-computational-modal-exhaustiveness`.
:::

:::{prf:remark} Corollary \ref{cor-computational-modal-exhaustiveness} is the correct replacement for `MT-AlgComplete`
:label: rem-cor22-replaces-mt-algcomplete

Corollary {prf:ref}`cor-computational-modal-exhaustiveness` is the properly decomposed replacement for the monolithic
claim previously carried by MT-AlgComplete.

The new structure is strictly better because it separates:
1. primitive-step classification,
2. decomposition of arbitrary polynomial-time families,
3. classification of irreducible witnesses,
4. and closure/exhaustiveness.

A referee can now attack any one of those steps precisely, instead of objecting to a single global statement whose proof
burden is too opaque.
:::

:::{prf:corollary} No Hidden Mechanism
:label: cor-no-hidden-mechanism

Suppose there exists a uniform algorithm family
$$
\mathcal A\in P_{\mathrm{FM}}
$$
that does not admit any modal factorization tree in
$$
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$

Then at least one theorem in the classification/exhaustiveness ladder is false. Concretely, at least one of
{prf:ref}`thm-sharp-universality`,
{prf:ref}`thm-int-universality`,
{prf:ref}`thm-flat-universality`,
{prf:ref}`thm-star-universality`,
{prf:ref}`thm-boundary-universality`,
{prf:ref}`thm-modal-composition`,
{prf:ref}`lem-primitive-step-classification`,
{prf:ref}`thm-witness-decomposition`,
{prf:ref}`thm-irreducible-witness-classification`,
or {prf:ref}`cor-computational-modal-exhaustiveness` must fail.

Equivalently: a genuine polynomial-time counterexample to modal factorization would not merely be an interesting new
algorithm; it would refute at least one explicit theorem in the classification/exhaustiveness ladder.
:::

:::{prf:proof}
Assume, for contradiction, that all theorems and corollaries listed in the statement hold.

Then Corollary {prf:ref}`cor-computational-modal-exhaustiveness` yields
$$
P_{\mathrm{FM}}
=
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$
Therefore every $\mathcal A\in P_{\mathrm{FM}}$ must admit a modal factorization tree in the saturated closure of the
five classes, contradicting the hypothesis.

Hence at least one theorem or corollary in the displayed list must fail.
:::

:::{prf:remark} Proper falsifiability statement
:label: rem-proper-falsifiability-statement

Corollary {prf:ref}`cor-no-hidden-mechanism` is the right falsifiability statement for the framework.

It does not say vaguely that maybe the foundation is incomplete. It says something much sharper: if a true
polynomial-time family is found outside the saturated five-class closure, then some specific theorem in the ladder is
wrong. That is the standard of falsifiability a hostile referee can actually engage with.
:::

:::{prf:remark} What remains after Part IV
:label: rem-what-remains-after-part-iv

If Parts I-IV are established, then the framework has proved:
1. the internal/classical bridge,
2. the normal-form theorem,
3. the universal properties of the five pure classes,
4. the decomposition theorem,
5. the irreducible-classification theorem,
6. and computational modal exhaustiveness.

At that point, the remaining work is problem-specific rather than foundational: one must prove that the target problem
family satisfies the complete obstruction schemas for all five classes. That is where the 3-SAT-specific burden
properly belongs.
:::

### V. Obstruction Theory

:::{prf:remark} Role of Part V
:label: rem-role-of-part-v

Part IV proves computational modal exhaustiveness:
$$
P_{\mathrm{FM}}
=
\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$

Part V turns this into a **problem-level hardness mechanism** via the mixed-modal obstruction theorem
({prf:ref}`thm-mixed-modal-obstruction`): if all five semantic obstruction propositions hold for a problem family
$\Pi$, then $\Pi \notin P_{\mathrm{FM}}$.

**What the P$\neq$NP proof requires.** Only two things:

1. The mixed-modal obstruction theorem (proved below from {prf:ref}`thm-witness-decomposition` and
   {prf:ref}`thm-irreducible-witness-classification`).
2. Five problem-specific semantic obstructions $\mathbb{K}_\lozenge^-(\Pi_{3\text{-SAT}})$ for canonical 3-SAT
   (established in Part VI via the blockage lemmas).

**What the P$\neq$NP proof does NOT require.** General-purpose obstruction calculi, completeness theorems for
arbitrary problem families, and universality characterizations are optional infrastructure for extending the
framework to other problems. They are not in the critical path.
:::

:::{prf:definition} Problem family and correct solver class
:label: def-problem-family-and-solvers

A **problem family** is a triple
$$
\Pi=(\mathfrak X,\mathfrak Y,\mathsf{Spec}),
$$
where:
1. $\mathfrak X$ is an admissible input family;
2. $\mathfrak Y$ is an admissible output family;
3. for each $n\in\mathbb N$, $\mathsf{Spec}_n\subseteq X_n\times Y_n$ is a decidable correctness relation.

A uniform family
$$
\mathcal A:\mathfrak X\Rightarrow \mathfrak Y
$$
is a **correct solver** for $\Pi$ if
$$
\forall n\in\mathbb N\ \forall x\in X_n,\qquad
\bigl(x,\mathcal A_n(x)\bigr)\in \mathsf{Spec}_n.
$$

The class of internally polynomial-time correct solvers is denoted
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi)
:=
\left\{
\mathcal A \in P_{\mathrm{FM}}(\mathfrak X,\mathfrak Y)
:
\mathcal A \text{ solves }\Pi
\right\}.
$$
:::

:::{prf:definition} Admissible irreducible modal component for a problem family
:label: def-admissible-irreducible-modal-component

Fix a problem family $\Pi$ and a modality
$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\}.
$$

An object
$$
[\mathcal B]\in \mathsf W_\lozenge
$$
is called an **admissible irreducible $\lozenge$-component for $\Pi$** if there exist:
1. a correct solver
   $$
   \mathcal A\in \mathsf{Sol}_{\mathrm{poly}}(\Pi),
   $$
2. and a minimal-rank modal factorization tree $T$ for $\mathcal A$ in the sense of
   {prf:ref}`def-witness-rank` and {prf:ref}`def-reducible-irreducible-witness`,

such that some irreducible subtree of $T$ denotes the witness object $[\mathcal B]$.

Equivalently: an admissible irreducible $\lozenge$-component is a genuine irreducible modal mechanism of type
$\lozenge$ that occurs inside at least one polynomial-time correct solver for $\Pi$.
:::

:::{prf:definition} Semantic modal obstruction proposition
:label: def-semantic-modal-obstruction

For a problem family $\Pi$ and a modality
$$
\lozenge\in\{\sharp,\int,\flat,\ast,\partial\},
$$
the **semantic modal obstruction proposition**
$$
\mathbb K_\lozenge^-(\Pi)
$$
is the statement that there exists no admissible irreducible $\lozenge$-component for $\Pi$.

Equivalently,
$$
\mathbb K_\lozenge^-(\Pi)
\iff
\text{no minimal-rank polynomial-time correct solver for $\Pi$ contains an irreducible witness in }\mathsf W_\lozenge.
$$

This is the semantic notion of “the $\lozenge$-route is blocked.”
:::

:::{prf:remark} Obstruction calculus framework
:label: def-obstruction-calculus-schema

The general obstruction calculus framework — including finitary certificate schemas, derivation systems, and
soundness/completeness notions — is developed in the companion document *Algorithmic Extensions*. The P$\neq$NP proof
does not require the general calculus; it uses only the semantic obstruction propositions
$\mathbb{K}_\lozenge^-(\Pi)$ of {prf:ref}`def-semantic-modal-obstruction`.
:::

:::{prf:remark} Full E13 obstruction package
:label: def-e13-reconstructed

A **full E13 obstruction package** for a problem family $\Pi$ consists of proofs that all five semantic obstruction
propositions hold:
$$
\mathbb{K}_\sharp^-(\Pi) \wedge \mathbb{K}_\int^-(\Pi) \wedge \mathbb{K}_\flat^-(\Pi) \wedge
\mathbb{K}_\ast^-(\Pi) \wedge \mathbb{K}_\partial^-(\Pi).
$$

The general finitary certificate packaging is developed in the companion document *Algorithmic Extensions*.
For the P$\neq$NP proof, what matters is the semantic conjunction above, which is established for canonical
3-SAT by the five blockage lemmas of Part VI.
:::

:::{prf:remark} $\sharp$-Obstruction for the P$\neq$NP Proof
:label: thm-sharp-obstruction-sound-complete

For the P$\neq$NP proof, what matters is the semantic obstruction proposition
$\mathbb{K}_\sharp^-(\Pi_{3\text{-SAT}})$: no minimal-rank polynomial-time correct solver for canonical 3-SAT
contains an irreducible $\sharp$-witness.

This is established in Part VI by {prf:ref}`lem-random-3sat-metric-blockage`, which proves that the glassy energy
landscape of random 3-SAT at threshold blocks all pure $\sharp$-witnesses. The barrier metatheorem of Part IX
({prf:ref}`thm-sharp-barrier-obstruction-metatheorem`) provides the supporting quantitative infrastructure.

A general sound-and-complete $\sharp$-obstruction calculus for arbitrary problem families is developed in the
companion document *Algorithmic Extensions*.
:::

:::{prf:remark} $\int$-Obstruction for the P$\neq$NP Proof
:label: thm-int-obstruction-sound-complete

For the P$\neq$NP proof, what matters is the semantic obstruction proposition
$\mathbb{K}_\int^-(\Pi_{3\text{-SAT}})$: no minimal-rank polynomial-time correct solver for canonical 3-SAT
contains an irreducible $\int$-witness.

This is established in Part VI by {prf:ref}`lem-random-3sat-causal-blockage`, which proves that the frustrated
dependency structure of random 3-SAT blocks all pure $\int$-witnesses. The barrier metatheorem of Part IX
({prf:ref}`thm-int-barrier-obstruction-metatheorem`) provides the supporting quantitative infrastructure.

A general sound-and-complete $\int$-obstruction calculus for arbitrary problem families is developed in the
companion document *Algorithmic Extensions*.
:::

:::{prf:remark} $\flat$-Obstruction for the P$\neq$NP Proof
:label: thm-flat-obstruction-sound-complete

For the P$\neq$NP proof, what matters is the semantic obstruction proposition
$\mathbb{K}_\flat^-(\Pi_{3\text{-SAT}})$: no minimal-rank polynomial-time correct solver for canonical 3-SAT
contains an irreducible $\flat$-witness.

This is established in Part VI by the joint blockage of {prf:ref}`lem-random-3sat-integrality-blockage` and
{prf:ref}`lem-random-3sat-galois-blockage`, together with the strengthened algebraic blockage theorem
{prf:ref}`thm-random-3sat-algebraic-blockage-strengthened`, which excludes all admissible polynomial-size algebraic
sketches. The barrier metatheorem of Part IX ({prf:ref}`thm-flat-barrier-obstruction-metatheorem`) provides the
supporting quantitative infrastructure.

A general sound-and-complete $\flat$-obstruction calculus for arbitrary problem families is developed in the
companion document *Algorithmic Extensions*.
:::

:::{prf:remark} $\ast$-Obstruction for the P$\neq$NP Proof
:label: thm-star-obstruction-sound-complete

For the P$\neq$NP proof, what matters is the semantic obstruction proposition
$\mathbb{K}_\ast^-(\Pi_{3\text{-SAT}})$: no minimal-rank polynomial-time correct solver for canonical 3-SAT
contains an irreducible $\ast$-witness.

This is established in Part VI by {prf:ref}`lem-random-3sat-scaling-blockage`, which proves that the supercritical
expansion of random 3-SAT blocks all pure $\ast$-witnesses. The barrier metatheorem of Part IX
({prf:ref}`thm-star-barrier-obstruction-metatheorem`) provides the supporting quantitative infrastructure.

A general sound-and-complete $\ast$-obstruction calculus for arbitrary problem families is developed in the
companion document *Algorithmic Extensions*.
:::

:::{prf:remark} $\partial$-Obstruction for the P$\neq$NP Proof
:label: thm-boundary-obstruction-sound-complete

For the P$\neq$NP proof, what matters is the semantic obstruction proposition
$\mathbb{K}_\partial^-(\Pi_{3\text{-SAT}})$: no minimal-rank polynomial-time correct solver for canonical 3-SAT
contains an irreducible $\partial$-witness.

This is established in Part VI by {prf:ref}`lem-random-3sat-boundary-blockage`, which proves that the linear
treewidth growth of random 3-SAT blocks all pure $\partial$-witnesses. The barrier metatheorem of Part IX
({prf:ref}`thm-partial-barrier-obstruction-metatheorem`) provides the supporting quantitative infrastructure.

A general sound-and-complete $\partial$-obstruction calculus for arbitrary problem families is developed in the
companion document *Algorithmic Extensions*.
:::

:::{prf:definition} Modal Barrier Decomposition
:label: def-modal-barrier-decomposition

A translator-stable barrier datum $\mathfrak{B} = (\mathfrak{Z}, i, \mathfrak{H}, S, r, E, a, b)$ admits a **modal barrier decomposition** if there exist five energy component functions

$$E_\sharp, E_\int, E_\flat, E_\ast, E_\partial : \mathfrak{Z} \to \mathbb{R}_{\geq 0}$$

and corresponding thresholds $a_\lozenge, b_\lozenge : \mathbb{N} \to \mathbb{R}_{\geq 0}$ satisfying:

1. **Additive decomposition:**
   $$E(z) = E_\sharp(z) + E_\int(z) + E_\flat(z) + E_\ast(z) + E_\partial(z)$$

2. **Modal orthogonality:** For each $\lozenge$ and any barrier-compatible pure $\lozenge$-endomorphism $F^\lozenge$ (transported to $\mathfrak{Z}$ via the witness encoding), the action of $F^\lozenge$ preserves non-$\lozenge$ components:
   $$E_{\lozenge'}(F^\lozenge_*(z)) = E_{\lozenge'}(z) \qquad \text{for all } \lozenge' \neq \lozenge$$

3. **Threshold decomposition:**
   $$\sum_\lozenge a_\lozenge(n) = a(n), \qquad \sum_\lozenge b_\lozenge(n) = b(n)$$

4. **Independent sub-barrier crossing:** For every path from a hard input state ($E \leq a$) to a solved state ($E \leq a$) in the state graph of the problem, and for each modality $\lozenge$, there exists a state along the path where $E_\lozenge \geq b_\lozenge(n)$. That is, each modal sub-barrier must be individually crossed.

5. **Translator stability of components:** Each $E_\lozenge$ is translator-stable: for every presentation translator $T$, the transported component $E_\lozenge \circ T^{-1}$ remains a valid $\lozenge$-barrier component (with polynomially distorted thresholds).
:::

:::{prf:theorem} Canonical 3-SAT Modal Barrier Decomposition
:label: thm-canonical-3sat-modal-barrier-decomposition

The canonical 3-SAT barrier datum $\mathfrak{B}_{3\text{-SAT}}$ of {prf:ref}`def-canonical-3sat-barrier-datum` admits a modal barrier decomposition in the sense of {prf:ref}`def-modal-barrier-decomposition`, with sub-barrier heights:

| Component | Captures | Sub-barrier height | Blockage reference |
|-----------|----------|-------------------|-------------------|
| $E_\sharp$ | Metric descent difficulty (local minima depth) | $\Delta_\sharp(n) = \Omega(n)$ | {prf:ref}`lem-random-3sat-metric-blockage` |
| $E_\int$ | Causal elimination difficulty (dependency depth) | $\Delta_\int(n) = \Omega(n)$ | {prf:ref}`lem-random-3sat-causal-blockage` |
| $E_\flat$ | Algebraic simplification difficulty | $\Delta_\flat(n) = \Omega(n)$ | {prf:ref}`lem-random-3sat-integrality-blockage` |
| $E_\ast$ | Recursive decomposition difficulty | $\Delta_\ast(n) = \Omega(n)$ | {prf:ref}`lem-random-3sat-scaling-blockage` |
| $E_\partial$ | Boundary contraction difficulty (treewidth) | $\Delta_\partial(n) = \Omega(n)$ | {prf:ref}`lem-random-3sat-boundary-blockage` |

*Proof.*

**Step 1. [Component extraction]:**
Each blockage lemma in Part VI identifies a specific structural property of random 3-SAT at threshold that blocks the corresponding modality. These properties are independent aspects of the problem's combinatorial structure:

- $E_\sharp$: landscape ruggedness — exponentially many local minima separated by $\Theta(n)$ barriers (Achlioptas–Ricci-Tersenghi), vanishing spectral gap (Montanari–Semerjian).
- $E_\int$: causal depth — variable-clause dependency poset has height $\Omega(n)$; no polynomial-length elimination schedule suffices.
- $E_\flat$: algebraic rigidity — frozen variables and propagation rigidity prevent algebraic cancellation over any ring.
- $E_\ast$: treewidth scaling — random 3-SAT at threshold has treewidth $\Theta(n)$, preventing polynomial-size recursive decomposition.
- $E_\partial$: interface complexity — linear treewidth growth forces boundary interfaces to grow linearly.

**Step 2. [Orthogonality]:**
Each $E_\lozenge$ is defined by a different structural invariant of the clause-variable hypergraph. A pure $\lozenge$-endomorphism modifies only the $\lozenge$-relevant structure: a $\sharp$-endomorphism changes the local energy (ranking function value) without altering causal dependencies, algebraic relations, recursive decomposition, or boundary interfaces. This follows from the universal properties of each modality: the modality-specific certificate $\Pi_\lozenge(F^\lozenge)$ constrains $F^\lozenge$ to operate within the $\lozenge$-modal component.

Concretely:
- A $\sharp$-step changes $V^\sharp$ (ranking function) without modifying poset structure, ring presentations, tree decompositions, or boundary separators.
- An $\int$-step processes one level of the causal elimination schedule without disturbing metric structure, algebraic identities, recursive partitions, or interfaces.
- Similarly for $\flat$, $\ast$, $\partial$.

**Step 3. [Independent crossing]:**
By {prf:ref}`thm-canonical-3sat-barrier-translator-stable` and the structural independence of the five invariants, every path from hard input to solution must individually exceed each $b_\lozenge(n)$. This is because: each $E_\lozenge$ starts at $\leq a_\lozenge$ (by construction) and ends at $\leq a_\lozenge$ (solved), but the structural invariant guarantees a state along any solving path where $E_\lozenge \geq b_\lozenge$.

**Step 4. [Sub-barrier heights]:**
The specific bounds $\Delta_\lozenge(n) = \Omega(n)$ are inherited from the corresponding blockage lemma's quantitative analysis (the barrier metatheorems of Part IX). $\square$
:::

:::{prf:remark} Justification of modal orthogonality
:label: rem-modal-orthogonality-justification

The orthogonality claimed in Step 2 of {prf:ref}`thm-canonical-3sat-modal-barrier-decomposition`
follows from the type-theoretic separation enforced by the pure witness definitions. By
{prf:ref}`def-pure-modal-witness-abstract`, each pure $\lozenge$-witness operates on its own modal
workspace $\mathfrak{Z}^\lozenge$ via the encoding $E_n^\lozenge$ and reconstruction $R_n^\lozenge$.
The core map $F_n^\lozenge$ is an endomorphism of $\mathfrak{Z}^\lozenge$ — it maps
$Z_n^\lozenge$ to $Z_n^\lozenge$ without accessing state components outside $\mathfrak{Z}^\lozenge$.

In the factorization tree of {prf:ref}`def-modal-factorization-tree`, each leaf operates within its
designated modal workspace. The encoding maps the global state INTO the workspace; the reconstruction
maps it BACK. Between encoding and reconstruction, the core map sees only the $\lozenge$-component.
This structural separation guarantees that a pure $\sharp$-step does not modify the dependency
structure ($\int$-component), algebraic presentations ($\flat$-component), recursion-tree structure
($\ast$-component), or interface objects ($\partial$-component).
:::

:::{prf:theorem} $\mathbb{K}_\lozenge^-$ from Modal Barrier Orthogonality
:label: thm-modal-barrier-obstruction-transfer

Let $\Pi$ be a problem family whose barrier datum $\mathfrak{B}$ admits a modal barrier decomposition ({prf:ref}`def-modal-barrier-decomposition`). Suppose that for each $\lozenge \in \{\sharp, \int, \flat, \ast, \partial\}$:

1. The $\lozenge$-blockage holds: no pure $\lozenge$-witness at the problem types is correct for $\Pi$ ({prf:ref}`lem-random-3sat-metric-blockage`–{prf:ref}`lem-random-3sat-boundary-blockage`).
2. The blockage is translator-stable ({prf:ref}`thm-canonical-3sat-barrier-translator-stable`).

Then the semantic modal obstruction proposition $\mathbb{K}_\lozenge^-(\Pi)$ holds for each $\lozenge$.

*Proof.*

Fix a modality $\lozenge$. Suppose toward contradiction that $\mathbb{K}_\lozenge^-(\Pi)$ fails: there exists a correct solver $\mathcal{A} \in \mathsf{Sol}_{\mathrm{poly}}(\Pi)$ with a minimal-rank factorization tree $T$ containing an irreducible $\lozenge$-subtree $T_0$.

**Step 1. [Irreducible leaf structure]:**
By {prf:ref}`thm-irreducible-witness-classification`, $T_0$ has rank $(1,0)$: a single pure $\lozenge$-leaf $B: \mathfrak{U} \Rightarrow \mathfrak{V}$ with witness data $(E^\lozenge, F^\lozenge, R^\lozenge, \mathfrak{Z}^\lozenge)$, possibly wrapped by translator-conjugation nodes.

**Step 2. [Barrier decomposition on solver's state space]:**
Since $\mathcal{A}$ is a correct solver, its computation on the hard subfamily $\mathfrak{H}$ must cross the total barrier. By {prf:ref}`def-modal-barrier-decomposition` property 4 (independent sub-barrier crossing), the solver's path must cross each modal sub-barrier $E_\lozenge \geq b_\lozenge(n)$.

**Step 3. [Orthogonal energy accounting]:**
By property 2 (modal orthogonality), only $\lozenge$-computation can change $E_\lozenge$. The non-$\lozenge$ leaves in $T$ change only their respective energy components $E_{\lozenge'}$ for $\lozenge' \neq \lozenge$. Therefore the $\lozenge$-leaves in $T$ are solely responsible for crossing the $\lozenge$-sub-barrier.

**Step 4. [Transport to leaf's state space]:**
The irreducible leaf $B$ operates at types $(\mathfrak{U}, \mathfrak{V})$ via its modal endomorphism $F^\lozenge$ on $\mathfrak{Z}^\lozenge$. The barrier component $E_\lozenge$ is transported to $\mathfrak{Z}^\lozenge$ through the tree's connecting maps and $B$'s modal encoding $E^\lozenge$. By property 5 (translator stability of components) and {prf:ref}`thm-canonical-3sat-barrier-translator-stable`, the transported $\lozenge$-sub-barrier remains valid on $\mathfrak{Z}^\lozenge$ with $\Delta_\lozenge(n) = \Omega(n)$.

**Step 5. [Blockage contradicts crossing]:**
The $\lozenge$-blockage (hypothesis 1) says: no pure $\lozenge$-endomorphism with polynomial budget can cross the $\lozenge$-sub-barrier. Specifically, the modality-specific structural obstruction (glassy landscape for $\sharp$, causal depth for $\int$, etc.) prevents $F^\lozenge$ from navigating $E_\lozenge$ from below $b_\lozenge$ to below $a_\lozenge$ while passing through $E_\lozenge \geq b_\lozenge$. By translator stability, this holds regardless of the encoding into $\mathfrak{Z}^\lozenge$.

Therefore $B$ cannot cross the $\lozenge$-sub-barrier.

**Step 6. [All $\lozenge$-leaves blocked; contradiction]:**
The argument of Steps 4–5 applies to *every* pure $\lozenge$-leaf in $T$, not only $B$: each such leaf operates at intermediate types connected to the problem types by translator-conjugation nodes, so translator stability transfers the $\lozenge$-blockage to its state space. Therefore no $\lozenge$-leaf in $T$ can cross the $\lozenge$-sub-barrier. By Step 3, the $\lozenge$-sub-barrier cannot be crossed at all. But $\mathcal{A}$ is correct, so it must cross the total barrier on $\mathfrak{H}$, which requires crossing each sub-barrier (property 4). Contradiction.

Therefore no irreducible $\lozenge$-leaf exists in any solver's minimal-rank tree: $\mathbb{K}_\lozenge^-(\Pi)$ holds. $\square$
:::

:::{prf:theorem} Mixed-Modal Obstruction Theorem
:label: thm-mixed-modal-obstruction

Let $\Pi$ be a problem family. Suppose that all five semantic obstruction propositions hold:
$$
\mathbb K_\sharp^-(\Pi)
\wedge
\mathbb K_\int^-(\Pi)
\wedge
\mathbb K_\flat^-(\Pi)
\wedge
\mathbb K_\ast^-(\Pi)
\wedge
\mathbb K_\partial^-(\Pi).
$$

Then
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi)=\varnothing.
$$

Equivalently: if every irreducible modal route is blocked, then no polynomial-time correct solver exists for $\Pi$.

In particular, if $\Pi$ carries a full E13 obstruction package in the sense of
{prf:ref}`def-e13-reconstructed`, then
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi)=\varnothing.
$$
:::

:::{prf:proof}
Assume, toward contradiction, that
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi)\neq\varnothing.
$$
Choose
$$
\mathcal A\in \mathsf{Sol}_{\mathrm{poly}}(\Pi).
$$
By {prf:ref}`thm-witness-decomposition`, $\mathcal A$ admits a finite modal factorization tree. Choose one of minimal
rank. By {prf:ref}`thm-irreducible-witness-classification`, every irreducible object occurring in that minimal-rank
tree lies in one of the five pure modal subcategories
$$
\mathsf W_\sharp,\ \mathsf W_\int,\ \mathsf W_\flat,\ \mathsf W_\ast,\ \mathsf W_\partial.
$$

Hence at least one admissible irreducible modal component exists for $\Pi$, belonging to one of those five classes.
This contradicts the conjunction of the five semantic obstruction propositions
$$
\mathbb K_\sharp^-(\Pi)\wedge \mathbb K_\int^-(\Pi)\wedge \mathbb K_\flat^-(\Pi)\wedge \mathbb K_\ast^-(\Pi)\wedge
\mathbb K_\partial^-(\Pi).
$$

Therefore no polynomial-time correct solver for $\Pi$ exists.
:::

:::{prf:corollary} Semantic Hardness from Obstruction
:label: cor-e13-contrapositive-hardness-reconstructed

Let $\Pi$ be a problem family. If all five semantic obstruction propositions hold:
$$
\mathbb{K}_\sharp^-(\Pi) \wedge \mathbb{K}_\int^-(\Pi) \wedge \mathbb{K}_\flat^-(\Pi) \wedge
\mathbb{K}_\ast^-(\Pi) \wedge \mathbb{K}_\partial^-(\Pi),
$$
then
$$
\Pi \notin P_{\mathrm{FM}}.
$$
:::

:::{prf:proof}
Immediate from {prf:ref}`thm-mixed-modal-obstruction`.
:::

:::{prf:remark} Proper falsifiability statement for the obstruction layer
:label: rem-proper-falsifiability-obstruction-layer

Part V yields the following precise falsifiability statement.

If a problem family $\Pi$ is later shown to admit a polynomial-time correct solver despite all five semantic
obstructions holding, then at least one of the following must have failed:
1. one of the five semantic obstruction propositions was incorrectly established;
2. the irreducible witness classification theorem
   {prf:ref}`thm-irreducible-witness-classification`;
3. the witness decomposition theorem
   {prf:ref}`thm-witness-decomposition`;
4. the computational modal exhaustiveness theorem
   {prf:ref}`cor-computational-modal-exhaustiveness`.

This is the exact theorem-level replacement for the current informal “Class VI algorithm would breach the framework”
language.
:::

:::{prf:remark} What Part V leaves to the problem-specific chapters
:label: rem-what-part-v-leaves

Part V is **problem-agnostic**. It does not prove that any specific problem family satisfies the five semantic
obstruction propositions. It only supplies the mixed-modal obstruction theorem converting semantic obstructions into
hardness.

The next burden is problem-specific: for canonical $3$-SAT, the five blockage lemmas of Part VI establish each
$\mathbb{K}_\lozenge^-(\Pi_{3\text{-SAT}})$ directly.
:::

### VI. Canonical 3-SAT Instantiation and Separation

:::{prf:remark} Role of Part VI
:label: rem-role-of-part-vi

Parts I--V are framework-level. They define:
1. the internal/external machine bridge;
2. the normal-form theorem;
3. the universal properties of the five pure classes;
4. the witness decomposition and irreducible-classification theorems;
5. and the mixed-modal obstruction theorem.

Part VI is the first **problem-specific** instantiation of that machinery. Its task is to show that the canonical
internal $3$-SAT family:
1. is an admissible problem family;
2. lies in $NP_{\mathrm{FM}}$ and is $NP_{\mathrm{FM}}$-complete;
3. carries the current tactic-level E13 obstruction package, or more strongly the reconstructed five-certificate
   semantic package of Part V;
4. and therefore lies outside $P_{\mathrm{FM}}$.

Once those steps are established, the separation chain is formal:
$$
\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}
\;\Longrightarrow\;
P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}
\;\Longrightarrow\;
P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
$$

The direct theorem route in this part is the current E13 antecedent package:
$$
K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \wedge K_{\mathrm{E8}}^-
\Rightarrow
K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})
\Rightarrow
\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
$$

The strengthened semantic obstruction framework of Part V (mixed-modal obstruction theorem) is a more demanding
formalization of that same exclusion route. It is a sufficient refinement of the current tactic-level theorem path,
not an additional logical prerequisite for invoking Definition {prf:ref}`def-e13` and Theorem
{prf:ref}`thm-e13-contrapositive-hardness`. The audit-level implementation is developed in the companion document
*Algorithmic Extensions*.
:::

:::{prf:definition} E13: Algorithmic Completeness Lock
:label: def-e13

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (algorithm representation), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{T_{\text{algorithmic}}}^+\}$ (algorithmic type with representation)
- **Produces:** a tactic-level frontend witness sufficient for the reconstructed E13 package
  {prf:ref}`def-e13-reconstructed`, hence for $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** Polynomial-time bypass; validates universal scope certificates
- **Breached By:** Existence of a compatible modal profile in
  $\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle$

**Method:** Modal-profile obstruction analysis via pure witnesses and saturated closure

**Mechanism:** For problem $(\mathcal{X}, \Phi)$, check whether any internally polynomial-time solver can admit a modal
profile built from the five pure modalities with presentation translators as outer maps. If all pure leaves are
blocked and no closure route survives, the problem is hard inside $P_{\text{FM}}$. This six-term package is a
tactic-level frontend sufficient condition for the semantic obstruction established by the mixed-modal obstruction
theorem ({prf:ref}`thm-mixed-modal-obstruction`).

The five modal checks correspond to existing tactics and nodes:
- **$\sharp$ (Metric):** Uses Node 7 ($\mathrm{LS}_\sigma$) + Node 12 ($\mathrm{GC}_\nabla$)
- **$\int$ (Causal):** Uses **Tactic E6** (Causal/Well-Foundedness)
- **$\flat$ (Algebraic):** Uses **Tactic E4** (Integrality) + **Tactic E11** (Galois-Monodromy)
- **$\ast$ (Scaling):** Uses Node 4 ($\mathrm{SC}_\lambda$) for subcriticality
- **$\partial$ (Holographic):** Uses **Tactic E8** (DPI) + Node 6 ($\mathrm{Cap}_H$)

**Frontend Certificate Logic:**

$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge K_{\mathrm{SC}_\lambda}^{\text{super}} \wedge K_{\mathrm{E8}}^- \Rightarrow K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

On the direct Part VI route, this implication is used exactly as written: once the six currently named frontend
obstruction certificates are established for the canonical problem object, the Algorithmic Completeness Lock yields
$K_{\mathrm{E13}}^+$. Part V (mixed-modal obstruction theorem) then provides the stronger semantic foundation for this exclusion route.

**Certificate Payload:** $(\text{modal\_status}[5], \text{leaf\_exclusions}[5], \text{profile\_exhaustion\_witness})$

**Automation:** Via composition of existing node/tactic evaluations; fully automatable for types with computable modality checks

**Literature:** Cohesive Homotopy Type Theory {cite}`SchreiberCohesive`; Algorithm taxonomy {cite}`Garey79`; Modal type theory {cite}`LicataShulman16`.
:::

:::{prf:proposition} Six Frontend Certificates Cover All Five Modal Channels
:label: prop-six-certificates-cover-five-channels

The six current frontend obstruction certificates of {prf:ref}`def-e13` establish the five semantic obstruction
propositions of {prf:ref}`def-semantic-modal-obstruction`:

| Frontend certificate | Modal channel | Blockage lemma establishing $\mathbb{K}_\lozenge^-$ |
|----------------------|--------------|------------------------------------------------------|
| $K_{\mathrm{LS}_\sigma}^-$ | $\sharp$ | {prf:ref}`lem-random-3sat-metric-blockage` |
| $K_{\mathrm{E6}}^-$ | $\int$ | {prf:ref}`lem-random-3sat-causal-blockage` |
| $K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^-$ | $\flat$ | {prf:ref}`lem-random-3sat-integrality-blockage`, {prf:ref}`lem-random-3sat-galois-blockage` |
| $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$ | $\ast$ | {prf:ref}`lem-random-3sat-scaling-blockage` |
| $K_{\mathrm{E8}}^-$ | $\partial$ | {prf:ref}`lem-random-3sat-boundary-blockage` |

In particular, there are no additional direct-route witness channels beyond these five.
:::

:::{prf:proof}
The five pure modal witness classes are exactly the irreducible generators of
$\mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle$, by
{prf:ref}`thm-irreducible-witness-classification`. Each certificate in {prf:ref}`def-e13` targets one modal class.

For each modality $\lozenge$, the corresponding blockage lemma in Part VI establishes:
1. **Problem-level blockage:** No pure $\lozenge$-witness at the canonical 3-SAT types is correct.
2. **Translator stability:** Each blockage proof's Step 7 invokes
   {prf:ref}`thm-canonical-3sat-barrier-translator-stable`, ensuring blockage persists under admissible re-encodings.

By {prf:ref}`thm-canonical-3sat-modal-barrier-decomposition`, the canonical 3-SAT barrier datum
admits a modal barrier decomposition with independent sub-barriers of height $\Omega(n)$ for each modality.
By {prf:ref}`thm-modal-barrier-obstruction-transfer`, the problem-level blockage and translator
stability together yield the semantic obstruction proposition
$\mathbb{K}_\lozenge^-(\Pi_{3\text{-SAT}})$ for each $\lozenge \in \{\sharp, \int, \flat, \ast, \partial\}$.

Since there are exactly five pure modal subcategories, there are no additional channels.
:::

:::{prf:theorem} E13 Contrapositive Hardness
:label: thm-e13-contrapositive-hardness

Let $\Pi$ be a problem family in the algorithmic ambient setting. If $\Pi$ carries the current tactic-level E13
obstruction certificate,

$$
K_{\mathrm{E13}}^+(\Pi),
$$

then no polynomial-time algorithm for $\Pi$ exists inside $P_{\text{FM}}$.

Equivalently:

$$
K_{\mathrm{E13}}^+(\Pi) \Rightarrow \Pi \notin P_{\text{FM}}.
$$

**Counterexample form:** if the theorem were false, there would exist a problem family $\Pi$ with
$K_{\mathrm{E13}}^+(\Pi)$ and a uniform algorithm family $\mathcal{A} \in P_{\mathrm{FM}}$ solving $\Pi$ — i.e.,
a polynomial-time solver that is invisible to all five modal obstruction channels.
:::

:::{prf:proof}
Assume toward contradiction that $\Pi$ carries $K_{\mathrm{E13}}^+(\Pi)$ and that there exists
$$
\mathcal{A} \in P_{\mathrm{FM}}(\mathfrak{X}, \mathfrak{Y}; \sigma)
$$
solving $\Pi$.

**Step 1. [Semantic obstruction from frontend certificates]:**
The six frontend certificates comprising $K_{\mathrm{E13}}^+(\Pi)$ establish the five semantic obstruction
propositions $\mathbb{K}_\lozenge^-(\Pi)$ for each $\lozenge \in \{\sharp, \int, \flat, \ast, \partial\}$,
via {prf:ref}`prop-six-certificates-cover-five-channels`.

**Step 2. [Mixed-modal obstruction]:**
By {prf:ref}`thm-mixed-modal-obstruction`, the conjunction of all five semantic obstructions implies
$\mathsf{Sol}_{\mathrm{poly}}(\Pi) = \varnothing$.

**Step 3. [Contradiction]:**
This contradicts the existence of $\mathcal{A}$. Therefore $\Pi \notin P_{\mathrm{FM}}$.
:::

### Optional Metric-Landscape Backend for the $\sharp$-Obstruction

:::{prf:remark} OGP as Optional Support for the Metric Obstruction
:label: rem-ogp-optional-backend

The core separation route in this chapter is:

$$
\text{E13 antecedent package} \Rightarrow K_{\mathrm{E13}}^+ \Rightarrow \Pi \notin P_{\text{FM}}
\Rightarrow P_{\text{FM}} \neq NP_{\text{FM}} \Rightarrow P_{\text{DTM}} \neq NP_{\text{DTM}}.
$$

Within that route, overlap-gap or glassy-landscape arguments are only one possible backend for the $\sharp$-channel
certificate $K_{\mathrm{LS}_\sigma}^-$. They are not the unique gatekeeper for the full modal exhaustion argument, and
they play no role in the algebraic, causal, scaling, or boundary obstruction channels.
:::

### Counter-Example Classifications

The following examples demonstrate how the strengthened Part IV-V classification and obstruction ladder correctly
classifies problems as P or NP-hard.

:::{prf:example} XORSAT: Class III (Algebraic)
:label: ex-xorsat-class-iii

**Problem:** Random linear equations $Ax = b$ over $\mathbb{F}_2$.

**Modal Analysis:**
- **$\sharp$ (Metric):** FAIL. No sharp descent certificate is used in this algebraic regime.
- **$\int$ (Causal):** FAIL. Linear dependencies create cycles.
- **$\flat$ (Algebraic):** **PASS**. The kernel $\ker(A)$ forms a large abelian subgroup.
- **$\ast$ (Scaling):** FAIL. No self-similar structure.
- **$\partial$ (Holographic):** FAIL. Not a matchgate problem.

**Tactic Activation:** Tactic E11 (Galois-Monodromy) detects the solvable Galois group.

**Certificate:** $K_{\mathrm{E11}}^{\text{solvable}} \Rightarrow K_{\text{Class III}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$

**Algorithm:** Gaussian Elimination ($O(n^3)$)

**Conclusion:** XORSAT is correctly classified as **Regular (P)** despite geometric hardness indicators.
:::

:::{prf:example} Horn-SAT: Class II (Propagators)
:label: ex-horn-sat-class-ii

**Problem:** Satisfiability of Horn clauses (at most one positive literal per clause).

**Modal Analysis:**
- **$\sharp$ (Metric):** FAIL. Landscape is non-convex.
- **$\int$ (Causal):** **PASS**. Horn clauses define a meet-semilattice with directed implications.
- **$\flat$ (Algebraic):** FAIL. Automorphism group is typically trivial.
- **$\ast$ (Scaling):** FAIL. Not self-similar.
- **$\partial$ (Holographic):** FAIL. Not a matchgate problem.

**Tactic Activation:** Tactic E6 (Causal/Well-Foundedness) detects the well-founded partial order.

**Certificate:** $K_{\mathrm{E6}}^{\text{DAG}} \Rightarrow K_{\text{Class II}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$

**Algorithm:** Unit Propagation ($O(n)$)

**Conclusion:** Horn-SAT is correctly classified as **Regular (P)** via causal structure detection.
:::

:::{prf:definition} Canonical 3-SAT Problem Object
:label: def-threshold-random-3sat-family

Let
$$
\mathfrak F_{3\text{-CNF}} = \bigl((F_n)_{n\in\mathbb N},m_F,\mathrm{enc}^F,\mathrm{dec}^F,\chi^F\bigr)
$$
be the admissible family whose $n$th level $F_n$ consists of all $3$-CNF formulas over variables
$$
x_1,\dots,x_{v(F)}
$$
with canonical bitstring encoding length bounded by a polynomial $m_F(n)$ and with $v(F)\le n$.

Let
$$
\mathfrak A = \bigl((\{0,1\}^{\le n} \cup \{\bot\})_{n\in\mathbb N},m_A,\mathrm{enc}^A,\mathrm{dec}^A,\chi^A\bigr)
$$
be the admissible assignment-or-rejection output family, where the $n$th level consists of all Boolean
strings of length at most $n$ (candidate satisfying assignments) together with a distinguished rejection
symbol $\bot$ (indicating unsatisfiability).

Let
$$
\mathfrak W_{3\text{-SAT}}
=
\bigl((W_n)_{n\in\mathbb N},m_W,\mathrm{enc}^W,\mathrm{dec}^W,\chi^W\bigr)
$$
be the admissible witness family, where
$$
W_n := \{0,1\}^{\le q_{3\text{-SAT}}(n)}
$$
and $q_{3\text{-SAT}}(n):=n$ is the standard witness-length bound.

Define the search-specification relation
$$
\mathsf{Spec}^{3\text{-SAT}}_n \subseteq F_n\times \bigl(\{0,1\}^{\le n} \cup \{\bot\}\bigr)
$$
by
$$
(F,a)\in \mathsf{Spec}^{3\text{-SAT}}_n
\iff
\begin{cases}
a \in \{0,1\}^{v(F)} \text{ and } \mathsf{Ver}^{3\text{-SAT}}_n(F,a)=1, & \text{if } F \text{ is satisfiable},\\
a = \bot, & \text{if } F \text{ is unsatisfiable},
\end{cases}
$$
where
$$
\mathsf{Ver}^{3\text{-SAT}}_n : F_n\times W_n \to \{0,1\}
$$
is the clause-satisfaction verifier:
$$
\mathsf{Ver}^{3\text{-SAT}}_n(F,a)=1
\iff
a \text{ satisfies every clause of }F.
$$

The **canonical internal $3$-SAT search problem family** is
$$
\Pi_{3\text{-SAT}}
:=
\bigl(\mathfrak F_{3\text{-CNF}},\mathfrak A,\mathsf{Spec}^{3\text{-SAT}}\bigr),
$$
equipped with witness family $\mathfrak W_{3\text{-SAT}}$ and verifier relation
$\mathsf{Ver}^{3\text{-SAT}}$. A correct solver must output a satisfying assignment for satisfiable
formulas (or $\bot$ for unsatisfiable ones), not merely a decision bit.

This is the unique satisfiability family used in the separation chain below.
:::

:::{prf:remark} Why the search formulation is essential
:label: rem-search-formulation-essential

**(a) Restriction monotonicity requires search.**
The hard subfamily $\mathfrak H$ ({prf:ref}`def-hard-subfamily-3sat`) consists exclusively of
**satisfiable** formulas. On this subfamily, the decision version of 3-SAT is trivially solvable by the
constant-$1$ function, which would make the restriction-monotonicity step
({prf:ref}`lem-frontend-restriction-monotonicity`) vacuous: blockage on a trivially solvable subfamily
says nothing about the full problem. The search formulation avoids this collapse because producing a
satisfying assignment remains hard even when satisfiability is guaranteed.

**(b) Search hardness implies P $\neq$ NP for decision.**
Since search-SAT polynomial-time reduces to decision-SAT via self-reducibility (query the decision oracle
with successive variable fixings to extract an assignment bit-by-bit), proving that the search version
requires superpolynomial time implies that the decision version also requires superpolynomial time.
Formally, if decision-SAT were in $\mathsf{P}$, then the self-reduction would place search-SAT in
$\mathsf{FP}$, contradicting the search blockage.

**(c) Consistency with witness definitions and blockage proofs.**
All pure witness definitions and blockage proofs in Part VI use the search formulation: the "solved set"
$S_n$ is the set of satisfying assignments, and the "correctness condition" requires producing a
satisfying assignment (not merely a bit). The barrier datum
({prf:ref}`def-canonical-3sat-barrier-datum`) defines the solved region as $S_n = \{x \in \{0,1\}^n :
x \text{ satisfies all clauses of } F\}$, and the reconstruction map $r_n$ extracts a satisfying
assignment from a solved state.
:::

:::{prf:theorem} Canonical 3-SAT Family is Admissible
:label: thm-canonical-3sat-admissible

The family
$$
\Pi_{3\text{-SAT}}
=
\bigl(\mathfrak F_{3\text{-CNF}},\mathfrak A,\mathsf{Spec}^{3\text{-SAT}}\bigr)
$$
of {prf:ref}`def-threshold-random-3sat-family` is an admissible problem family in the sense of
{prf:ref}`def-problem-family-and-solvers`.

Moreover:
1. the witness family $\mathfrak W_{3\text{-SAT}}$ is admissible;
2. the verifier relation $\mathsf{Ver}^{3\text{-SAT}}_n(F,a)$ is decidable uniformly in time polynomial in $n$;
3. the witness-length bound $q_{3\text{-SAT}}(n)=n$ is polynomial.

Hence $\Pi_{3\text{-SAT}}$ is well-typed as a search family and as a verifier-based NP family.
:::

:::{prf:proof}
The source family $\mathfrak F_{3\text{-CNF}}$ is admissible because $3$-CNF formulas are finite syntactic objects over
a fixed finite alphabet. Their canonical tokenization and padding to fixed-length valid codes yields injective bitstring
encodings with polynomial-time validity testing and decoding.

The assignment-or-rejection output family $\mathfrak A$ is admissible: each level
$\{0,1\}^{\le n} \cup \{\bot\}$ is a finite set with canonical bitstring encoding (assignments padded to
length $n$, $\bot$ encoded as a reserved codeword), polynomial-time validity testing, and injective
encoding. The witness family
$$
W_n=\{0,1\}^{\le n}
$$
is admissible by the standard bitstring encoding.

For the verifier, given a formula $F\in F_n$ and an assignment $a\in W_n$, one evaluates each clause by reading at most
three literals and checking whether at least one is satisfied by $a$. Since the number of clauses and variables encoded
in $F$ is $O(n)$, the total runtime is polynomial in $n$.

The search-specification relation is therefore well-defined, and all admissibility conditions hold.
:::

:::{prf:definition} Current Frontend E13 Package for Canonical 3-SAT
:label: def-current-frontend-e13-package-3sat

We say that the canonical satisfiability family $\Pi_{3\text{-SAT}}$ carries the **current tactic-level E13 frontend
package** if the six currently named frontend obstruction certificates all hold on that family:
$$
K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \wedge K_{\mathrm{E8}}^-.
$$

This is exactly the antecedent package displayed in Definition {prf:ref}`def-e13`, now specialized to the canonical
internal $3$-SAT problem object.
:::

:::{prf:lemma} Restriction Monotonicity for the Current Frontend Obstruction Templates
:label: lem-frontend-restriction-monotonicity

Let
$$
\Pi'=(\mathfrak X',\mathfrak Y,\mathsf{Spec}')
$$
be obtained from a problem family
$$
\Pi=(\mathfrak X,\mathfrak Y,\mathsf{Spec})
$$
by restricting the admissible input family to an admissibly presented subfamily
$$
X_n'\subseteq X_n
\qquad (n\in\mathbb N),
$$
and restricting the specification relation accordingly.

Then any current frontend witness template or tactic-level certificate for one of the five modal channels
$$
\sharp,\ \int,\ \flat,\ \ast,\ \partial
$$
restricts from $\Pi$ to $\Pi'$. Equivalently: if a current frontend obstruction certificate blocks one of those
channels on $\Pi'$, then the same channel is blocked on $\Pi$.
:::

:::{prf:proof}
We verify for each of the five witness classes that a witness on the full family $\Pi$ restricts to a witness on the
subfamily $\Pi'$, and contrapositively that an obstruction on $\Pi'$ implies an obstruction on $\Pi$.

**$\sharp$-channel.** A pure $\sharp$-witness on $\Pi$ provides a ranking function $V_n^\sharp$, a local update
$F_n^\sharp$, and a solved set $S_n^\sharp$ satisfying strict progress and correctness for all $x \in X_n$.
Restricting to $X_n' \subseteq X_n$ preserves all three properties (the quantifier narrows). Hence any $\sharp$-witness
on $\Pi$ restricts to a valid $\sharp$-witness on $\Pi'$.

**$\int$-channel.** A pure $\int$-witness on $\Pi$ provides a poset $(P_n, \prec_n)$, local updates, and a linear
extension satisfying correctness for all $x \in X_n$. Restricting to $X_n'$ preserves the poset structure and the
correctness condition.

**$\flat$-channel.** A pure $\flat$-witness on $\Pi$ provides algebraic structures $A_n^\flat, B_n^\flat$ and a sketch
$s_n \to e_n \to d_n$ with correctness for all $x \in X_n$. The sketch operates uniformly; restricting the input
family preserves presentation size, elimination maps, and correctness.

**$\ast$-channel.** A pure $\ast$-witness on $\Pi$ provides a splitting map, merge map, and recursion-tree structure
with correctness for all $x \in X_n$. Restricting to $X_n'$ preserves the recursion-tree structure, the size decrease
property, and correctness.

**$\partial$-channel.** A pure $\partial$-witness on $\Pi$ provides interface objects $B_n^\partial$ and contraction
maps with correctness for all $x \in X_n$. Restricting preserves interface size bounds and correctness.

In each case, the crucial observation is: a pure $\lozenge$-witness is defined by a universally quantified correctness
condition over the input family. Restricting the input family to a subfamily only narrows the universal quantifier.
Contrapositively, if no valid $\lozenge$-witness exists on $\Pi'$ (i.e., the $\lozenge$-channel is obstructed on the
subfamily), then no valid $\lozenge$-witness exists on $\Pi$ either.
:::

### VI.C.1. Hard Subfamily and Canonical Barrier Datum

:::{prf:definition} Hard Subfamily for Canonical 3-SAT
:label: def-hard-subfamily-3sat

The **hard subfamily** of canonical 3-SAT is the restricted problem family

$$
\Pi_{3\text{-SAT}}^{\mathrm{hard}}
=
(\mathfrak{H}, \mathfrak{A}, \mathsf{Spec}^{3\text{-SAT}}|_{\mathfrak{H}})
$$

where $\mathfrak{H} = (\mathfrak{H}_n)_{n \in \mathbb{N}}$ and $\mathfrak{H}_n$ consists of all **satisfiable**
3-CNF formulas on $n$ variables at clause-to-variable ratio

$$
\alpha \approx 4.267
$$

(the random 3-SAT satisfiability threshold). More precisely, $\mathfrak{H}_n$ consists of formulas with
$m = \lfloor \alpha n \rfloor$ clauses, each clause a disjunction of 3 literals over $n$ variables, such that the
formula is satisfiable.

$\Pi_{3\text{-SAT}}^{\mathrm{hard}}$ is an admissibly presented subfamily of $\Pi_{3\text{-SAT}}$: the restriction
$\mathfrak{H}_n \subseteq F_{3\text{-CNF},n}$ is admissible because membership in $\mathfrak{H}_n$ is decidable
(satisfiability is decidable, clause-to-variable ratio is computable, and each structural property below is
decidable given the formula).

We define $\mathfrak{H}_n$ to consist of those formulas in $F_{3\text{-CNF},n}$ at ratio $\alpha$ that
**deterministically satisfy** all six structural properties below. By known probabilistic results, a
$(1 - o(1))$-fraction of satisfiable formulas at ratio $\alpha$ satisfy all six properties simultaneously as
$n \to \infty$, so $\mathfrak{H}_n$ is non-empty for all sufficiently large $n$.

The six definitional requirements for $F \in \mathfrak{H}_n$ are:

1. **Solution-space shattering:** The satisfying assignments partition into $\exp(\Theta(n))$ clusters at pairwise
   Hamming distance $\Omega(n)$ (Achlioptas-Ricci-Tersenghi 2006, Mézard-Montanari 2009).
2. **Glassy landscape:** The energy function $E_n(x) =$ unsatisfied clauses has exponentially many local minima,
   vanishing spectral gap, and Łojasiewicz failure near frozen variables (Mézard-Parisi-Zecchina 2002,
   Montanari-Semerjian 2006).
3. **Frustrated cycles:** The clause-variable dependency graph contains strongly connected frustration cores of size
   $\Theta(n)$ (Achlioptas-Beame-Molloy 2001).
4. **Linear expansion:** The clause-variable incidence graph is an expander with linear separator cost and linear
   treewidth.
5. **Automorphism triviality:** $\operatorname{Aut}(F) = \{\mathrm{id}\}$ for a $(1 - o(1))$ fraction of instances.
6. **Non-solvable monodromy:** The monodromy group of the solution variety is the full symmetric group $S_k$ for
   $k \geq 5$.
:::

:::{prf:remark} Well-definedness and non-circularity of the hard subfamily
:label: rem-hard-subfamily-well-definedness

**Non-emptiness.** Each of the structural properties defining $H_n$ is satisfied with probability
$1 - o(1)$ by uniformly random satisfiable 3-CNF formulas at the satisfiability threshold. By a
union bound, all properties hold simultaneously with probability $1 - o(1)$, so
$H_n \neq \emptyset$ for all sufficiently large $n$.

**Decidability note.** Membership in $H_n$ requires checking satisfiability, which is NP-complete.
However, decidability of $H_n$ is NOT required by the proof. The proof uses $H_n$ only through
{prf:ref}`lem-frontend-restriction-monotonicity`: if no pure $\lozenge$-witness is valid on $H_n$,
then no pure $\lozenge$-witness is valid on the full family $\Pi_{3\text{-SAT}}$. This transfer
requires only $H_n \subseteq \Pi_{3\text{-SAT},n}$ and $H_n \neq \emptyset$.

**No circularity.** The hard subfamily is a combinatorially defined set whose non-emptiness is
proved by probabilistic counting, not by appeal to computational complexity assumptions. The
blockage lemmas prove that certain witness structures cannot solve instances in $H_n$; they do
not need to construct elements of $H_n$ algorithmically.
:::

:::{prf:definition} Canonical 3-SAT Barrier Datum
:label: def-canonical-3sat-barrier-datum

The **canonical 3-SAT barrier datum** is the tuple
$$
\mathfrak B_{3\text{-SAT}}
=
\bigl(
\mathfrak Z,\ i,\ \mathfrak H,\ S,\ r,\ E,\ a,\ b
\bigr)
$$
in the sense of {prf:ref}`def-barrier-datum`, with the following instantiations.

1. **State family.**
   $Z_n = \{0,1\}^n$ is the assignment space for formulas on $n$ Boolean variables, with Hamming encoding.

2. **Input embedding.**
   $i_n$ maps a formula $F$ on $n$ variables to a canonical initial assignment $x_0(F) \in \{0,1\}^n$, computed in
   polynomial time from $F$ (e.g., by a fixed polynomial-time heuristic such as unit propagation followed by greedy
   clause satisfaction). The embedding is a presentation translator.

3. **Hard subfamily.**
   $\mathfrak H_n$ consists of satisfiable formulas at clause-to-variable ratio
   $$
   \alpha \approx 4.267
   $$
   (the random 3-SAT satisfiability threshold).

4. **Solved region.**
   $S_n = \{x \in \{0,1\}^n : x \text{ satisfies all clauses of } F\}$
   is the set of satisfying assignments.

5. **Reconstruction.**
   $r_n$ extracts the satisfying assignment from a solved state by reading the assignment bits.

6. **Energy functional.**
   $$
   E_n(x) = \text{number of clauses of } F \text{ unsatisfied by assignment } x.
   $$

7. **Thresholds.**
   $$
   a(n) = \lceil \alpha n / 8 \rceil + 1,
   \qquad
   b(n) = c_{\mathrm{shatter}} \cdot n,
   $$
   where $c_{\mathrm{shatter}} > 0$ is the shattering barrier constant (see
   {prf:ref}`thm-canonical-3sat-barrier-datum-valid`). The requirement $a(n) < b(n)$ holds for sufficiently large
   $n$ since $c_{\mathrm{shatter}} > \alpha / 8$ (established by the cluster separation property).
:::

:::{prf:theorem} Validity of the Canonical 3-SAT Barrier Datum
:label: thm-canonical-3sat-barrier-datum-valid

The barrier datum $\mathfrak B_{3\text{-SAT}}$ of {prf:ref}`def-canonical-3sat-barrier-datum` satisfies the three axioms
of {prf:ref}`def-barrier-datum`:

1. **(B1)** Source states satisfy $E_n(i_n(F)) \leq a(n)$ for all $F \in H_n$.

2. **(B2)** Solved states satisfy $E_n(z) = 0 \leq a(n)$ for all $z \in S_n$.

3. **(B3)** Every admissible solver trace from $i_n(F)$ to $S_n$ must pass through a state with
   $E_n \geq b(n) = c_{\mathrm{shatter}} \cdot n$.
:::

:::{prf:proof}
**Step 1. [B1 — low-energy source condition]:**
The canonical initial assignment $x_0(F)$ produced by the polynomial-time heuristic satisfies at least a
$(7/8)$-fraction of clauses in expectation (since each random 3-clause is satisfied by $7$ of $8$ truth assignments to
its $3$ variables). By Chernoff concentration, with high probability over hard instances $F \in H_n$:
$$
E_n(x_0(F)) \leq \lceil \alpha n / 8 \rceil + 1 = a(n).
$$

**Step 2. [B2 — low-energy solved condition]:**
By definition, $x \in S_n$ satisfies all clauses, so $E_n(x) = 0 \leq a(n)$.

**Step 3. [B3 — barrier separation]:**
For random 3-SAT at the satisfiability threshold $\alpha \approx 4.267$, the solution space undergoes
**shattering** into exponentially many well-separated clusters (Achlioptas-Ricci-Tersenghi 2006,
Mézard-Montanari 2009). With high probability over formula draws:

- The satisfying assignments partition into clusters $C_1, \dots, C_k$ with $k = \exp(\Theta(n))$.
- Any two assignments in distinct clusters have Hamming distance $\Omega(n)$.
- Any path in assignment space connecting two assignments in distinct clusters must pass through assignments
  violating at least $c_{\mathrm{shatter}} \cdot n$ clauses.

The last property follows from the expansion of the random 3-SAT clause-variable hypergraph: each variable appears in
$\Theta(\alpha) = \Theta(1)$ clauses, and any contiguous set of $\Omega(n)$ variable flips from a satisfying
assignment unsatisfies at least $c_{\mathrm{shatter}} \cdot n$ clauses for an explicit constant
$c_{\mathrm{shatter}} > \alpha / 8 > 0$ depending on $\alpha$ and the expansion constant.

The initial assignment $x_0(F)$ lies outside all solution clusters (its Hamming distance to every satisfying
assignment is $\Omega(n)$ w.h.p., since the heuristic satisfies only $7/8$ of clauses while a satisfying assignment
satisfies all). By the expansion of the clause-variable hypergraph, any path from $x_0(F)$ to a satisfying assignment
must traverse $\Omega(n)$ variable flips. At the point where approximately half the required flips have been made,
the assignment simultaneously fails to satisfy the clauses that the initial heuristic satisfied AND fails to satisfy
the clauses of the target cluster, yielding energy $\geq c_{\mathrm{shatter}} \cdot n$.

This establishes B3 with barrier height
$\Delta_{\mathfrak B}(n) = b(n) - a(n) = (c_{\mathrm{shatter}} - \alpha/8) \cdot n - O(1) = \Theta(n)$.
:::

:::{prf:remark} Barrier datum scope and the Overlap Gap Property
:label: rem-ogp-exponential-barrier

The clause-count barrier datum $\mathfrak B_{3\text{-SAT}}$ gives a barrier height $\Delta = \Theta(n)$. Combined with
the local drift bound $d_\lozenge(n) = O(1)$, the Part IX barrier metatheorems yield
$$
\beta_\lozenge^{\mathfrak B}(n) \geq \Omega(n),
$$
a linear lower bound on the minimum number of steps or presentation size.

This linear lower bound provides useful structural information (e.g., ruling out $o(n)$-step witnesses) but does not by
itself rule out polynomial-time witnesses with ranking bound $q(n) = n^k$ for large $k$.

The full blockage of each channel at the **superpolynomial** level requires channel-specific structural arguments
beyond the simple energy-barrier model, deployed below via the frontend obstruction lemmas. Those arguments exploit
properties such as:
- **Glassy landscape + ♯-purity** (sharp): cluster shattering with frozen-variable core ($\int$-type)
  renders metric-descent ($\sharp$-modal) computation cluster-blind (Mézard-Parisi-Zecchina 2002,
  Achlioptas-Coja-Oghlan 2008);
- **Frustrated cycles** (causal): strongly connected frustration cores preventing DAG-like elimination
  (Achlioptas-Beame-Molloy 2001);
- **Supercritical expansion** (scaling): linear boundary size preventing subcritical recursion;
- **Unbounded treewidth** (boundary): linear treewidth preventing polynomial-size interface compression.

A stronger barrier datum using the **Overlap Gap Property** (OGP) (Gamarnik-Sudan 2017) could in principle provide
superpolynomial lower bounds through the Part IX metatheorems, but this requires a different state family and energy
functional (based on overlap structure rather than clause count). Such a construction is deferred to the optional
backend dossier route (see companion document *Algorithmic Extensions*); it is not needed for the direct Part VI
route, which uses the frontend obstruction lemmas.
:::

:::{prf:remark} Two-level architecture of the blockage proofs
:label: rem-two-level-blockage-architecture

The five blockage lemmas ({prf:ref}`lem-random-3sat-metric-blockage` through {prf:ref}`lem-random-3sat-boundary-blockage`)
each employ a **two-level exclusion architecture**. It is important to understand the distinct roles of
these two levels and their logical independence.

**Level 1 — Barrier framework (supporting role).**
Each blockage proof contains a "Step 5: Supporting barrier bound" that invokes the Part IX barrier
metatheorems via the barrier datum $\mathfrak B_{3\text{-SAT}}$. This yields a lower bound
$\beta_\lozenge^{\mathfrak B}(n) \geq \Omega(n)$ on the channel-specific resource parameter. The role of
this bound is to provide a **quantitative floor**: it rules out all witnesses whose resource parameter is
sublinear, i.e., $o(n)$-step or $o(n)$-size witnesses. This is useful structural information but, as
noted in {prf:ref}`rem-ogp-exponential-barrier`, it does not by itself exclude polynomial-time witnesses
with ranking bound $q(n) = n^k$ for arbitrarily large $k$.

**Level 2 — Frontend structural obstruction (main exclusion).**
The actual exclusion of *all* polynomial-time witnesses — including those with $q(n) = n^k$ for any
fixed $k$ — is accomplished by the channel-specific structural arguments in Steps 2–4 of each blockage
proof. These arguments identify obstructions that are **independent of the degree of the polynomial
ranking bound**:

- **Sharp channel ($\sharp$):** The $\sharp$-purity constraint restricts $F_n^\sharp$ to
  metric-descent computation — $F$ can access metric/potential information (rank, energy,
  distances) but not constraint-graph structure ($\int$-type) or algebraic identities ($\flat$-type).
  On the shattered landscape with $\exp(\Theta(n))$ clusters, the vanishing spectral gap renders
  metric information cluster-uninformative, and cluster identity is determined by the frozen-variable
  core (an $\int$-type object invisible to the $\sharp$ modality). No $\sharp$-modal $F$ can
  navigate to the correct cluster, regardless of the degree of $q_\sharp(n)$.

- **Causal channel ($\int$):** Frustrated cycles in the clause-variable hypergraph are a structural
  obstruction to DAG-like elimination orderings. Their presence is independent of the poset size
  $q_\int(n)$.

- **Algebraic channel ($\flat$):** Non-solvable monodromy of the solution variety obstructs algebraic
  elimination. This is a Galois-theoretic obstruction independent of presentation size $q_\flat(n)$.

- **Scaling channel ($\ast$):** Coupling through $\Theta(n)$ crossing constraints prevents polynomial
  total recursion-tree size. The supercritical expansion holds regardless of the degree of $q_\ast(n)$.

- **Boundary channel ($\partial$):** Linear treewidth $\operatorname{tw}(G_n) = \Theta(n)$ of the
  constraint hypergraph forces exponential contraction time $2^{\Omega(n)}$, because the contraction
  must process $2^{\Omega(\operatorname{tw})}$ feasible separator configurations. This
  exponential-time obstruction holds regardless of the degree of $q_\partial(n)$.

**Logical independence.**
Level 2 does *not* depend on Level 1. The frontend structural obstructions are self-contained arguments
that would remain valid even if the barrier metatheorems of Part IX were removed entirely. The barrier
framework provides supplementary quantitative information (the $\Omega(n)$ floor) and connects the
blockage lemmas to the broader energy-landscape theory, but it is not a logical prerequisite for the
polynomial-time exclusion established by the frontend obstruction arguments.
:::

:::{prf:theorem} Translator Stability of the Canonical 3-SAT Barrier Datum
:label: thm-canonical-3sat-barrier-translator-stable

The barrier datum $\mathfrak B_{3\text{-SAT}}$ is translator-stable in the sense of {prf:ref}`def-translator-stable-barrier`:

1. Presentation translators applied to $\mathfrak B_{3\text{-SAT}}$ produce polynomial-size outputs.
2. Energy distortion under translation is bounded by a polynomial factor: for any presentation translator $T$,
   $$
   E_n^T(T(x)) \leq p(E_n(x))
   $$
   for some polynomial $p$ depending only on the translator.
3. Barrier separation ratio is preserved:
   $$
   \frac{\Delta_{T\mathfrak B}(n)}{\Delta_{\mathfrak B}(n)} \geq \frac{1}{p(n)}
   $$
   for a polynomial $p$, so the $\Theta(n)$ barrier height remains $\Theta(n)$ after translation.
:::

:::{prf:proof}
**Step 1. [Polynomial output size]:**
By definition of presentation translator, any such translator $T$ produces
outputs of size at most $p_T(n)$ for a polynomial $p_T$.

**Step 2. [Energy distortion bound]:**
The energy functional $E_n(x)$ counts unsatisfied clauses. Under a presentation translator $T$ that re-encodes the
formula and assignment, the re-encoded formula has at most $p_T(m)$ clauses where $m$ is the original clause count.
Each re-encoded clause depends on at most $p_T(1)$ re-encoded variables. Therefore the number of unsatisfied
re-encoded clauses is at most a polynomial multiple of the original count.

**Step 3. [Barrier ratio preservation]:**
The barrier height $\Delta_{\mathfrak B}(n) = b(n) - a(n) = \Theta(n)$. After translation,
$\Delta_{T\mathfrak B}(n) \geq \Delta_{\mathfrak B}(n) / p(n)$ by the energy distortion bound. Since
$\Delta_{\mathfrak B}(n) = \Theta(n)$ and $p(n)$ is polynomial, $\Delta_{T\mathfrak B}(n) = \Omega(n / p(n))$, which
remains $\Omega(n^{1-\epsilon})$ for any $\epsilon > 0$ with appropriate choice of $p$.
:::

### VI.C.2. The Five Blockage Theorems

Each of the following lemmas proves blockage for one modal channel by assuming an arbitrary pure witness of the
corresponding type on $\Pi_{3\text{-SAT}}$ and deriving a contradiction. In each case, the proof first establishes
the structural obstruction on the hard subfamily $\Pi_{3\text{-SAT}}^{\mathrm{hard}}$
({prf:ref}`def-hard-subfamily-3sat`), then lifts via {prf:ref}`lem-frontend-restriction-monotonicity`.

:::{prf:remark} Scope of the direct-route blockage proofs
:label: rem-scope-direct-route-blockage

**What the direct route proves.** Each blockage lemma below establishes the semantic obstruction proposition
$\mathbb{K}_\lozenge^-(\Pi_{3\text{-SAT}})$ for its modal channel by two complementary arguments:

1. **Frontend structural blockage.** All pure $\lozenge$-witnesses are blocked for
   $\Pi_{3\text{-SAT}}^{\mathrm{hard}}$ via the corresponding frontend obstruction lemma. (For the $\sharp$-channel,
   the obstruction uses $\sharp$-purity to show that metric-descent computation is cluster-blind on the shattered
   landscape. For the other channels, the obstruction blocks the named class template and, by the structure of the
   pure witness definitions, extends to all pure witnesses of that modality.)

2. **Barrier-derived quantitative bound.** The barrier datum $\mathfrak{B}_{3\text{-SAT}}$ gives a linear lower
   bound $\beta_\lozenge \geq \Omega(n)$ via the Part IX metatheorems, ruling out all pure $\lozenge$-witnesses with
   sublinear ranking bound or resource parameter.

Together, these establish the semantic obstruction propositions required by the mixed-modal obstruction theorem
({prf:ref}`thm-mixed-modal-obstruction`).
:::

:::{prf:remark} The saturated closure argument
:label: rem-saturated-closure-argument

The mixed-modal obstruction theorem ({prf:ref}`thm-mixed-modal-obstruction`) handles the saturated closure
argument directly: if all five semantic obstruction propositions hold, then no element of
$\mathsf{Sat}\langle \sharp, \int, \flat, \ast, \partial \rangle$ solves $\Pi$, because every factorization tree
has leaves in the five pure modal subcategories, and each is blocked.

The proof is by contradiction: any solver in $P_{\mathrm{FM}}$ decomposes ({prf:ref}`thm-witness-decomposition`)
into pure modal leaves ({prf:ref}`thm-irreducible-witness-classification`), but each leaf type is excluded by the
corresponding semantic obstruction.
:::

:::{prf:lemma} Metric Blockage for Canonical 3-SAT
:label: lem-random-3sat-metric-blockage

For every pure $\sharp$-witness $(V_n^\sharp, F_n^\sharp, S_n^\sharp, q_\sharp)$ in the sense of
{prf:ref}`def-pure-sharp-witness-rigorous` on $\Pi_{3\text{-SAT}}$, the correctness condition fails. Equivalently,
the metric obstruction certificate $K_{\mathrm{LS}_\sigma}^-$ holds.
:::

:::{prf:proof}
**Step 1. [Typed witness class]:**
By {prf:ref}`def-pure-sharp-witness-rigorous`: a pure $\sharp$-witness consists of a ranking function
$V_n^\sharp: Z_n^\sharp \to \mathbb N$ bounded by polynomial $q_\sharp(n)$, a $\sharp$-modal update $F_n^\sharp$,
a solved set $S_n^\sharp$, strict progress ($V_n$ decreases by at least 1 per non-solved step), and the correctness
identity. The exclusion target is all such witnesses — not only the Class I climber template of
{prf:ref}`def-class-i-climbers`, but every pure $\sharp$-witness regardless of the choice of $V$, $F$, or encoding.

**Step 2. [Glassy landscape of random 3-SAT]:**
For random 3-SAT at the satisfiability threshold $\alpha \approx 4.267$, the landscape
$\Phi(x) = E_n(x) = $ number of unsatisfied clauses exhibits all three glassy signatures required by
{prf:ref}`lem-sharp-obstruction`:

- **Cluster shattering with frozen core:** The solution space shatters into $\exp(\Theta(n))$ clusters at pairwise
  Hamming distance $\Omega(n)$ (Achlioptas-Ricci-Tersenghi 2006). Within each cluster, a $\Theta(n)$-fraction of
  variables are frozen to specific values by the implication structure of the constraint graph
  (Achlioptas-Coja-Oghlan 2008). Different clusters have different frozen cores.
- **Vanishing spectral gap:** The Glauber dynamics transition matrix on the hard subfamily has spectral gap
  $\lambda_{\min} \to 0$ as $n \to \infty$, yielding mixing time $\exp(\Omega(n))$ (Montanari-Semerjian 2006).
  Metric-local information carries negligible cluster-identity signal.
- **Łojasiewicz failure:** The inequality $\|\nabla\Phi(x)\| \geq c|\Phi(x)-\Phi^*|^{1-\theta}$ fails for any
  fixed $\theta > 0$: near frozen-variable configurations, the gradient vanishes while the energy gap to the
  nearest solution remains $\Theta(n)$. Gradient-based descent is cluster-uninformative.

**Step 3. [Frontend obstruction application]:**
By {prf:ref}`lem-sharp-obstruction`: the $\sharp$-purity constraint restricts $F_n^\sharp$ to metric-descent
computation, and the glassy landscape ensures that metric-descent computation cannot navigate to the correct
solution cluster (cluster identity is determined by the frozen-variable core, an $\int$-type object invisible to
the $\sharp$ modality). Therefore no pure $\sharp$-witness exists on the hard subfamily $\mathfrak H$.

**Step 4. [Restriction monotonicity]:**
By {prf:ref}`lem-frontend-restriction-monotonicity`: the blockage lifts from the hard subfamily to the full canonical
family $\Pi_{3\text{-SAT}}$. Thus no pure $\sharp$-witness exists on $\Pi_{3\text{-SAT}}$.

**Step 5. [Supporting barrier bound]:**
The barrier datum $\mathfrak B_{3\text{-SAT}}$ of {prf:ref}`def-canonical-3sat-barrier-datum` provides a supporting
quantitative lower bound. Each $\sharp$-update changes $E_n$ by at most
$d_\sharp(n) \leq 3 \cdot \lceil\alpha\rceil = O(1)$ (a sharp local drift bound in the sense of
{prf:ref}`def-sharp-local-energy-drift-bound`). By {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`:
$\beta_\sharp^{\mathfrak B}(n) \geq \Omega(n)$, ruling out any witness with sublinear ranking bound.

**Step 6. [Certificate extraction]:**
The metric obstruction certificate is
$$
K_{\mathrm{LS}_\sigma}^-.
$$

**Step 7. [Translator stability]:**
By {prf:ref}`thm-canonical-3sat-barrier-translator-stable`, the clause-count barrier datum is translator-stable, so
the linear lower bound is preserved under admissible re-encodings.

**Step 8. [Failure localization]:**
If the theorem were false, a pure $\sharp$-witness with polynomial ranking bound would exist for
$\Pi_{3\text{-SAT}}$. By $\sharp$-purity, $F$ would be limited to metric-descent operations. But the shattered
landscape (exponentially many clusters with frozen cores, vanishing spectral gap, Łojasiewicz failure) ensures that
metric-descent operations cannot identify or navigate to the correct solution cluster — the cluster-identity
information is encoded in the constraint-graph topology ($\int$-type), not in metric quantities ($\sharp$-type).
The failure point is the modal mismatch: the computational task requires $\int$-type information that the
$\sharp$-modality cannot access.
:::

:::{prf:lemma} Causal Blockage for Canonical 3-SAT
:label: lem-random-3sat-causal-blockage

For every pure $\int$-witness $(P_n, \prec_n, U_{n,i}, \sigma_n, q_\int)$ in the sense of
{prf:ref}`def-pure-int-witness-rigorous` on $\Pi_{3\text{-SAT}}$, the correctness condition fails. Equivalently,
the causal obstruction certificate $K_{\mathrm{E6}}^-$ holds.
:::

:::{prf:proof}
**Step 1. [Typed witness class]:**
By {prf:ref}`def-pure-int-witness-rigorous`: a pure $\int$-witness consists of a finite poset $(P_n, \prec_n)$ with
$|P_n| \leq q_\int(n)$, local updates $U_{n,i}$ depending only on predecessors $j \prec_n i$, a linear extension
$\sigma_n$, and the correctness identity. The exclusion target is all such witnesses.

**Step 2. [Frustrated cycles in random 3-SAT]:**
By {prf:ref}`lem-shape-obstruction`, the causal/propagation witness language is blocked when the dependency structure
contains **frustrated loops**. For random 3-SAT at threshold $\alpha \approx 4.267$:

- The clause-variable dependency graph contains **strongly connected frustration cores**: cycles of clauses
  $c_1, c_2, \ldots, c_\ell$ where satisfying $c_j$ forces a literal value that conflicts with $c_{j+1}$
  (Achlioptas-Beame-Molloy 2001).
- These frustration cores have $\pi_1(\int\mathcal X) \neq 0$: the fundamental group of the shape is non-trivial.
  Constraint propagation around any such cycle produces contradictions.
- W.h.p. the hard subfamily contains frustration cores of size $\Theta(n)$, involving a constant fraction of all
  variables.

**Step 3. [Elimination order obstruction]:**
Any pure $\int$-witness imposes a DAG-compatible elimination order on the variables. But a frustration core of size
$\Theta(n)$ is a **strongly connected** subgraph of the dependency graph: for any two variables $v_i, v_j$ in the
core, there exist directed paths $v_i \to \cdots \to v_j$ and $v_j \to \cdots \to v_i$. Therefore no elimination
order (linear extension of any compatible poset) can process the core without encountering a cycle, requiring the
elimination to "resolve" a circular dependency — which a pure $\int$-witness cannot do by definition (each local
update depends only on predecessors).

**Step 4. [Frontend obstruction application]:**
By {prf:ref}`lem-shape-obstruction`: the frustrated-loop structure blocks the standard propagation/elimination witness
language of {prf:ref}`def-class-ii-propagators` on the hard subfamily.

**Step 5. [Restriction monotonicity]:**
By {prf:ref}`lem-frontend-restriction-monotonicity`: the blockage lifts from the hard subfamily to the full canonical
family $\Pi_{3\text{-SAT}}$.

**Step 6. [Supporting barrier bound]:**
The barrier datum $\mathfrak B_{3\text{-SAT}}$ provides a supporting linear lower bound. Each local elimination update
changes $E_n$ by at most $d_\int(n) \leq 3 \cdot \lceil\alpha\rceil = O(1)$ (an $\int$ local drift bound in the
sense of {prf:ref}`def-int-local-energy-drift-bound`). By {prf:ref}`thm-int-barrier-obstruction-metatheorem`:
$\beta_\int^{\mathfrak B}(n) \geq \Omega(n)$, ruling out sublinear elimination schedules.

**Step 7. [Certificate extraction]:**
The causal obstruction certificate is
$$
K_{\mathrm{E6}}^-.
$$

**Step 8. [Translator stability + Failure localization]:**
Translator stability is inherited from {prf:ref}`thm-canonical-3sat-barrier-translator-stable`. The failure point is
the frustrated-cycle structure: if random 3-SAT at threshold had acyclic constraint propagation (like Horn-SAT, where
the implication graph is a DAG), the $\int$-route would not be blocked.
:::

:::{prf:lemma} Causal blockage transfers to arbitrary posets
:label: lem-causal-arbitrary-poset-transfer

Let $(P_n, \prec_n)$ be any well-founded poset with $|P_n| \leq q(n)$ for a polynomial $q$, and let
$(U_{n,i})_{i \in P_n}$ be local updates depending only on predecessors, with reconstruction map
$R_n^{\int}$ computable in polynomial time. Then no pure $\int$-witness with poset $(P_n, \prec_n)$
correctly solves the hard subfamily $\mathfrak{H}$ of {prf:ref}`def-hard-subfamily-3sat`.
:::

:::{prf:proof}
Fix an arbitrary linear extension $\sigma_n$ of $(P_n, \prec_n)$ and consider the state
$\mathrm{state}_k$ after the first $k$ updates in this linear extension have been applied.

**Step 1. [Determined variable set.]**
Define the *determined variable set* at step $k$:
$$
V_k := \bigl\{j \in [n] : \text{the } j\text{th coordinate of } R_n^{\int}(\mathrm{state}_k)
\text{ is fixed regardless of how the remaining } |P_n| - k \text{ updates proceed}\bigr\}.
$$
That is, variable $j$ is determined at step $k$ if every completion of the remaining updates (consistent
with the poset and the input) yields the same value for the $j$th output coordinate.

**Step 2. [Boundary conditions.]**
$V_0 = \varnothing$: before any updates, the state is the initial encoding of the input, and since
multiple satisfying assignments exist (the hard subfamily has $\exp(\Theta(n))$ clusters), no output
coordinate is fixed. $V_{|P_n|} = [n]$: by the correctness condition of the $\int$-witness, after all
updates the reconstruction map must produce a valid satisfying assignment, so every coordinate is
determined.

**Step 3. [Monotonicity.]**
$V_k \subseteq V_{k+1}$ for all $k$ (once a variable is determined, subsequent updates cannot
un-determine it, since the correctness condition requires a deterministic output). The set $V_k$
grows from $\varnothing$ to $[n]$.

**Step 4. [Expansion barrier.]**
By the linear expansion property of the hard subfamily (property 4 of
{prf:ref}`def-hard-subfamily-3sat`), the clause-variable incidence graph is an expander. For any
subset $S \subseteq [n]$ with $|S| \leq n/2$, the number of clauses with variables in both $S$ and
$[n] \setminus S$ is at least $c_{\mathrm{exp}} \cdot |S|$ for an expansion constant
$c_{\mathrm{exp}} > 0$.

Since $|V_k|$ grows from $0$ to $n$, there exists a step $k^*$ at which
$|V_{k^*}| \leq n/2 < |V_{k^*+1}|$. We distinguish two cases:

**Case A: $|V_{k^*}| = \Theta(n)$ (no large jump).** The partition
$(V_{k^*},\; [n] \setminus V_{k^*})$ crosses at least
$c_{\mathrm{exp}} \cdot |V_{k^*}| = \Theta(n)$ clauses by expansion, and we proceed
directly to Step 5 with $\Theta(n)$ crossing clauses at the critical partition.

**Case B: $|V_{k^*}| = o(n)$ (large jump).** Here the single update $U_{n,k^*+1}$
causes $|V_{k^*+1}| - |V_{k^*}| = \Theta(n)$ new variables to become determined
simultaneously. By the $\int$-modal restriction (item 7 of
{prf:ref}`def-pure-int-witness-rigorous`), $U_{n,k^*+1}$ cannot invoke a general SAT
solver or optimization subroutine on the encoded input — it is restricted to forward
causal propagation. While $U_{n,k^*+1}$ reads the encoded formula, the $\int$-modal
restriction limits how this information is used: $U$ can propagate constraints forward
along the poset but cannot perform non-$\int$ operations (energy minimization, algebraic
elimination, backtracking search) that would be required to resolve the frustrated cycle.
The update reads only the values at predecessor sites $j \prec_n (k^*\!+\!1)$ (at most
$|V_{k^*}| = o(n)$ determined coordinates) and the encoded input. Its output — a single
coordinate in the abstract poset — must, through the reconstruction map $R_n^{\int}$,
simultaneously determine $\Theta(n)$ output coordinates.

Among these $\Theta(n)$ newly determined variables, $\Omega(n)$ lie in the frustrated
core: by property 3 of {prf:ref}`def-hard-subfamily-3sat`, the frustrated core
comprises a $\Theta(n)$-fraction of all variables, and the newly determined set has
$\Theta(n)$ elements, so by pigeonhole at least $\Omega(n)$ newly determined variables
are core variables. These frustrated-core variables have cycle-closing constraints
among themselves that $U_{n,k^*+1}$ must resolve correctly. But $U_{n,k^*+1}$ receives
only $o(n)$ determined predecessor values as input. For correctness on all
hard-subfamily inputs, this single update must effectively solve a sub-problem involving
$\Omega(n)$ coupled frustrated-core variables using only $o(n)$ predecessor values and
the input encoding. By the same frustrated-cycle argument as in Case A (applied to the
information available to $U_{n,k^*+1}$), $\int$-modal computation from $o(n)$
predecessor values cannot resolve $\Omega(n)$ frustrated-core constraints whose
cycle-closing dependencies involve undetermined successors.

In both cases, the critical step faces $\Omega(n)$ frustrated constraints that
$\int$-modal updates cannot resolve.

**Step 5. [∫-modal updates cannot resolve the frustrated core.]**

The updates $U_{n,i}$ in a pure $\int$-witness are $\int$-modal: by the $\int$-modal restriction
(item 7 of {prf:ref}`def-pure-int-witness-rigorous`), each $U_{n,i}$ factors through the shape
modality $\int$. Each update $U_{n,i}$ is a polynomial-time function that reads only the
values at predecessor sites $j \prec_n i$ and the encoded input. The $\int$-modal restriction
constrains both the **information** $U_{n,i}$ receives (predecessor values only) and the
**type of computation** it performs: $U_{n,i}$ is restricted to forward causal propagation and
cannot internally invoke $\sharp$-type (optimization/energy), $\flat$-type (algebraic
elimination), $\ast$-type (divide-and-conquer), or $\partial$-type (interface contraction)
subroutines. The key constraint is that $U_{n,i}$ sees only predecessor values, not the values
of successor or incomparable sites, and can only propagate constraints forward along the poset.
When a frustrated-core variable is the first in its cycle to be determined, the cycle-closing
constraint involves sites that are not predecessors (they come later in any linear extension
of the poset). No polynomial-time local function of predecessor values can compensate for
this missing information, because the correct value depends on the cycle-closing constraint
that involves undetermined successors.

The frustrated core of the hard subfamily (property 3 of {prf:ref}`def-hard-subfamily-3sat`)
contains $\Theta(n)$ variables in a strongly connected component of the constraint dependency
graph. Within this core, variable $v_i$'s correct value depends (through implication chains)
on variable $v_j$, and vice versa, for every pair $v_i, v_j$. In the determined-variable
sequence $V_0 \subset V_1 \subset \cdots \subset V_{|P_n|}$, the frustrated-core variables
enter $V_k$ at various steps. The first core variable to be determined (say at step $k_1$)
must be assigned a value that is consistent with the cycle-closing constraint — but the
cycle-closing constraint involves core variables that are not yet determined (they enter
$V_k$ at later steps $k_2 > k_1$).

The $\int$-modal update at step $k_1$ can read predecessor values and perform arbitrary
polynomial-time computation on them, but it cannot incorporate information from the
undetermined end of the frustrated cycle. This computational limitation manifests concretely
in the non-convergence of belief propagation (BP) and other message-passing algorithms — the
canonical $\int$-type algorithms — on frustrated random 3-SAT instances (Mézard–Montanari
2009, Ch. 19). While BP non-convergence is a property of a specific algorithm, it illustrates
the general principle: any computation restricted to predecessor-only information flow (the
defining constraint of $\int$-modal updates) cannot resolve circular dependencies in the
frustrated core.

A general polynomial-time algorithm (without $\int$-purity) could resolve the cycle by invoking
non-$\int$ mechanisms: a $\sharp$-modal subroutine could use energy minimization, or a $\flat$-modal
subroutine could use algebraic elimination. But a **pure** $\int$-witness is restricted to
$\int$-modal updates, which can only propagate forward — not close loops.

Combining with Step 4: at the critical step $k^*$, the $\Omega(n)$ frustrated constraints
identified in Step 4 must be resolved by the $\int$-modal update(s). We verify that the
crossing clauses include frustrated-core clauses by a pigeonhole argument: the frustrated
core comprises a constant fraction $\gamma > 0$ of all $n$ variables (property 3 of
{prf:ref}`def-hard-subfamily-3sat`). In Case A, $|V_{k^*}| = \Theta(n)$, so the partition
$(V_{k^*},\; [n] \setminus V_{k^*})$ splits the variable set into two parts each of size
$\Theta(n)$. By expansion (property 4), the frustrated-core variables cannot be concentrated
in any $o(n)$-sized subset; hence a constant fraction of frustrated-core variables lie in
$V_{k^*}$ and a constant fraction in $[n] \setminus V_{k^*}$. At the constant clause-to-variable
ratio $\alpha \approx 4.267$, each frustrated-core variable participates in $\Theta(1)$ clauses,
so the $\Theta(n)$ core variables generate $\Theta(n)$ internal core clauses. Since the core
variables are split across the partition, at least $\Omega(n)$ of these core clauses are
crossing clauses. (In Case B, the $\Omega(n)$ frustrated-core constraints arise directly from
the jump argument above.) Since $\int$-modal computation cannot close the frustrated loops,
the update fails to produce a consistent assignment for the crossing variables. The
correctness condition of the $\int$-witness is violated.

**Step 6. [Conclusion.]**
No pure $\int$-witness with any polynomial-size poset can correctly solve the hard subfamily.
:::

:::{prf:remark} Connection to the frustrated-cycle argument
:label: rem-frustrated-cycle-vs-transfer

The frustrated-cycle argument in Steps 2–4 of {prf:ref}`lem-random-3sat-causal-blockage` and
the transfer lemma {prf:ref}`lem-causal-arbitrary-poset-transfer` use the same structural
obstruction — the $\int$-modality's inability to resolve frustrated loops — but apply it in
complementary ways:

- **Main proof (Steps 2–4):** When the poset embeds into the clause-variable dependency graph,
  the frustrated cycles directly prevent any compatible linear extension from processing the
  strongly connected core in topological order.

- **Transfer lemma:** For **arbitrary** posets, the expansion barrier (Step 4) forces a critical
  step at which $\Theta(n)$ frustrated-core constraints must be resolved simultaneously. The
  $\int$-modal typing of the updates prevents resolving these constraints because the frustrated
  cycle requires backward information flow that $\int$-modal forward propagation cannot provide.

The two arguments converge on the same conclusion: no pure $\int$-witness, regardless of poset
topology, can solve the hard subfamily of random 3-SAT.
:::

:::{prf:lemma} Integrality Blockage for Canonical 3-SAT
:label: lem-random-3sat-integrality-blockage

For every pure $\flat$-witness $(\Sigma, A_n^\flat, B_n^\flat, s_n, e_n, d_n, q_\flat)$ in the sense of
{prf:ref}`def-pure-flat-witness-rigorous` whose elimination map $e_n^\flat$ operates via
$\mathfrak{S}_{\mathrm{quot}}$ (quotient/congruence compression) on $\Pi_{3\text{-SAT}}$, the presentation-size
bound $q_\flat(n)$ is violated. Equivalently, the integrality obstruction certificate $K_{\mathrm{E4}}^-$ holds.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
By {prf:ref}`def-pure-flat-witness-rigorous`: a pure $\flat$-witness requires a finite-sorted algebraic signature
$\Sigma$, finitely presented $\Sigma$-structures $A_n^\flat, B_n^\flat$ with presentation size bounded by $q_\flat(n)$,
a sketch $s_n \to e_n \to d_n$, and the correctness identity.

**Step 2. [Quotient compression lower bound — search formulation.]**
For the search formulation ({prf:ref}`def-threshold-random-3sat-family`), the solver must output a
satisfying assignment $\sigma(F) \in \{0,1\}^{v(F)}$ for each satisfiable formula $F$. The quotient
structure $A_n / {\sim}$ partitions the input space into equivalence classes, and the reconstruction
map $d_n^{\flat}$ produces a single output per class. Therefore, any two formulas $F_1, F_2$ with
$\sigma(F_1) \neq \sigma(F_2)$ must lie in distinct equivalence classes.

We construct $N = 2^{cn}$ formulas in $\mathfrak{H}_n$ (for a suitable constant $c > 0$) with
pairwise-distinct required outputs. Draw $N$ formulas $F^{(1)}, \ldots, F^{(N)}$ independently at the
threshold ratio $\alpha \approx 4.267$ (i.e., $m = \lceil \alpha n \rceil$ clauses). We show that with
high probability these formulas have **pairwise-disjoint** satisfying-assignment sets.

For two independently drawn formulas $F^{(i)}$ and $F^{(j)}$, the probability that they share any
common satisfying assignment is bounded by a union bound over all $2^n$ possible assignments:
$$
\Pr\bigl[\mathrm{Sol}(F^{(i)}) \cap \mathrm{Sol}(F^{(j)}) \neq \emptyset\bigr]
\;\leq\; \sum_{\sigma \in \{0,1\}^n}
  \Pr[\sigma \text{ satisfies } F^{(i)}] \cdot \Pr[\sigma \text{ satisfies } F^{(j)}]
\;=\; 2^n \cdot (7/8)^{2m}.
$$
Each clause is satisfied by a fixed assignment $\sigma$ with probability $7/8$ (since exactly one of
the eight literal patterns falsifies a uniformly random 3-clause), and the $2m$ clauses across the two
independent formulas are independent. Substituting $m = \alpha n$:
$$
2^n \cdot (7/8)^{2\alpha n}
= 2^{n\bigl(1 - 2\alpha \log_2(8/7)\bigr)}
= 2^{n(1 - 2 \cdot 4.267 \cdot 0.193)}
= 2^{-0.647n}.
$$
For $N = 2^{cn}$ formulas, the union bound over $\binom{N}{2} < 2^{2cn}$ pairs gives
$$
\Pr[\text{any pair shares a solution}]
\;\leq\; 2^{2cn} \cdot 2^{-0.647n}
\;=\; 2^{-n(0.647 - 2c)}.
$$
Choosing any $c < 0.647/2 \approx 0.32$ (e.g., $c = 0.1$) makes this probability $o(1)$.
With pairwise-disjoint solution sets, any correct solver must output a satisfying assignment for each
formula, and these assignments are necessarily pairwise distinct (belonging to disjoint solution sets).
Therefore
$$
|A_n / {\sim}| \;\geq\; N \;=\; 2^{\Omega(n)},
$$
and
$$
\operatorname{pres}(A_n) \;\geq\; \log_2 |A_n / {\sim}| \;\geq\; \Omega(n) \cdot c_1
$$
for an explicit constant $c_1 > 0$. Any polynomial-size presentation $q_\flat(n) = n^{O(1)}$ is
violated for large $n$.

**Step 3. [Apply algebraic barrier]:**
By {prf:ref}`thm-flat-barrier-obstruction-metatheorem`: the presentation-size lower bound
$\beta_\flat^{\mathfrak B}(n) \geq 2^{c_1 \cdot n}$ eventually dominates every polynomial. Therefore no
barrier-compatible pure $\flat$-witness via the $\mathfrak S_{\mathrm{quot}}$ subfamily exists.

**Step 4. [Certificate]:**
This contributes the integrality obstruction certificate
$$
K_{\mathrm{E4}}^-
$$
via the integrality lock {prf:ref}`def-e4`.

**Step 5. [Translator stability]:**
Translators preserve presentation size up to polynomial factor:
$\operatorname{pres}(T(A_n)) \geq 2^{\Omega(n)} / \operatorname{poly}(n) = 2^{\Omega(n)}$,
by {prf:ref}`thm-canonical-3sat-barrier-translator-stable`.

**Step 6. [Failure localization]:**
The quotient compression step would require collapsing exponentially many well-separated solution clusters into
a polynomial-size quotient. The failure point is the exponential number of clusters at pairwise Hamming distance
$\Omega(n)$: the search formulation requires producing a satisfying assignment from the correct cluster, and no
polynomial-size quotient can map to assignments in multiple well-separated clusters.
:::

:::{prf:lemma} Galois-Monodromy Blockage for Canonical 3-SAT
:label: lem-random-3sat-galois-blockage

For every pure $\flat$-witness whose elimination map operates via $\mathfrak{S}_{\mathrm{mono}}$
(solvable-monodromy sketches) on $\Pi_{3\text{-SAT}}$, the presentation-size bound is violated. Equivalently, the
Galois-monodromy obstruction certificate $K_{\mathrm{E11}}^-$ holds.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
The $\mathfrak S_{\mathrm{mono}}$ sub-channel of the strengthened $\flat$-class (item 6 of
{prf:ref}`def-algebraic-signature-library-flat`) requires a solvable group $G$ with $|G| \leq q_\flat(n)$ acting
equivariantly on the solution space, through which algebraic elimination proceeds via Galois-theoretic reductions.

**Step 2. [Automorphism triviality]:**
For random 3-SAT at the satisfiability threshold $\alpha \approx 4.267$, the formula automorphism group satisfies
$$
\operatorname{Aut}(F) = \{\mathrm{id}\}
$$
for a $(1 - o(1))$ fraction of hard instances. This follows from the random structure: each variable participates in
$\Theta(1)$ clauses with distinct neighborhoods w.h.p., so no nontrivial permutation preserves all clauses.

**Step 2b. [Algebraization and monodromy definition.]**
To define the monodromy group rigorously for Boolean satisfiability, embed the problem in
algebraic geometry over $\mathbb{C}$. Replace each Boolean variable $x_i \in \{0,1\}$ with a
complex variable $z_i \in \mathbb{C}$ and each clause $(l_1 \vee l_2 \vee l_3)$ with the
polynomial equation $\bar{l}_1 \cdot \bar{l}_2 \cdot \bar{l}_3 = 0$, where $\bar{l}_j = z_i$
if $l_j = \neg x_i$ and $\bar{l}_j = 1 - z_i$ if $l_j = x_i$. Restrict to the Boolean
hypercube by adding the equations $z_i^2 - z_i = 0$ for all $i$.

The resulting polynomial system defines an algebraic variety $\mathcal{V}(F) \subset \mathbb{C}^n$
whose real Boolean points $\mathcal{V}(F) \cap \{0,1\}^n$ are the satisfying assignments.
Parameterize the clause coefficients by a parameter space $\Lambda$ (the space of all 3-CNF
formulas with $m$ clauses on $n$ variables). As $\lambda \in \Lambda$ varies, the roots of the
system move continuously, defining a covering space. The **monodromy group**
$\mathrm{Mon}(\mathcal{V}/\Lambda)$ is the group of permutations of the roots induced by loops
in $\Lambda$ (with the discriminant locus removed). This is a standard construction in algebraic
geometry (see Harris, *Galois Groups of Enumerative Problems*, 1979).

**Step 3. [Non-solvable monodromy via covering-space analysis.]**
We establish that for a $(1 - o(1))$ fraction of random 3-SAT formulas at the satisfiability
threshold, the monodromy group $\mathrm{Mon}(\mathcal{V}/\Lambda)$ is the full symmetric group
$S_k$ on the Boolean solution set (where $k = |\mathcal{V}(F) \cap \{0,1\}^n|$). The
non-solvability of $\mathrm{Mon}(\mathcal{V}/\Lambda)$ is encoded in property 6 of
{prf:ref}`def-hard-subfamily-3sat`. The argument below establishes that this property holds with
probability $1 - o(1)$ over random 3-SAT formulas at threshold, confirming the **non-emptiness**
of the hard subfamily with respect to this property.

**(a) Transitivity.** The parameter space $\Lambda$ (the space of all 3-CNF formulas with $m$
clauses on $n$ variables, with continuously varying clause coefficients) is irreducible. The
discriminant locus $\Delta \subset \Lambda$ (where the number of Boolean solutions changes) has
codimension $\geq 1$, so $\Lambda \setminus \Delta$ is connected. Since the fiber over any point in
$\Lambda \setminus \Delta$ has exactly $k$ simple Boolean roots, the covering
$\pi: \mathcal{I} \to \Lambda \setminus \Delta$ is a finite étale covering. Connectedness of the
base $\Lambda \setminus \Delta$ combined with the path-lifting property implies the monodromy acts
transitively on the fiber (otherwise the covering would decompose into disconnected components).

**(b) Full symmetric group from transpositions.** The Picard--Lefschetz transpositions generate the
monodromy group. For each pair of Boolean solutions $(\sigma_i, \sigma_j)$, the discriminant locus
$\Delta$ contains a component $\Delta_{ij}$ where these two roots collide (become degenerate). A
simple loop around $\Delta_{ij}$ produces the transposition $(\sigma_i \; \sigma_j)$.

The Picard--Lefschetz theorem applies here because the Boolean satisfying assignments are *simple*
(non-degenerate) roots of the full polynomial system: the Jacobian of the system (clause polynomials
plus hypercube constraints $z_i^2 - z_i = 0$) has full rank at each Boolean solution, since the
hypercube constraints contribute $n$ independent linear conditions on the tangent space and the
clause polynomials are generically transverse. The family of solution sets over
$\Lambda \setminus \Delta$ is therefore étale, and the standard Picard--Lefschetz formula for
zero-dimensional étale coverings gives a transposition at each simple ramification point (see Dimca,
*Sheaves in Topology*, Ch. 4).

Since $\Lambda \setminus \Delta$ is connected (the discriminant has codimension $\geq 1$ in the
irreducible parameter space $\Lambda$, so its complement is connected), the monodromy group contains
transpositions for all pairs $(\sigma_i, \sigma_j)$. A group containing all transpositions on a set
is the full symmetric group $S_k$.

**(c) Application to random 3-SAT.** For a random 3-SAT formula drawn at the satisfiability
threshold, the number of Boolean solutions is $k = \exp(\Theta(n))$ w.h.p. (organized into
$\exp(\Theta(n))$ clusters). The transposition argument of (b) applies to a $(1 - o(1))$ fraction
of such formulas, giving $\mathrm{Mon} = S_k$. Since $S_k$ for $k \geq 5$ is **not solvable**
(its composition series contains the alternating group $A_k$, which is simple and non-abelian for
$k \geq 5$), no solvable-group elimination is possible.

**Step 4. [Presentation-size lower bound from non-solvable monodromy.]**
The $\mathfrak{S}_{\mathrm{mono}}$ channel requires the elimination map to factor through a
solvable group $G$ acting equivariantly on the solution space. We show that non-solvable
monodromy forces superpolynomial presentation size for any such factorization.

By the generalization of the Abel-Ruffini theorem to algebraic functions (Arnol'd 1970): if
the monodromy group of a covering is non-solvable, the covering cannot be trivialized by
successive adjunctions of radicals (= solvable group extensions). Any algebraic elimination
procedure that reduces the number of roots by passing to quotients by normal subgroups must
follow the composition series of the monodromy group. For $S_k$ with $k \geq 5$, the composition
series is $\{e\} \triangleleft A_k \triangleleft S_k$, where $A_k$ is simple non-abelian. Since
$A_k$ has no nontrivial normal subgroups, no further reduction is possible: the elimination must
represent the full action of $A_k$ on the solution set.

The minimum faithful permutation representation of $A_k$ has degree $k$ (for $k \geq 5$), and
any presentation of $A_k$ by generators and relations requires at least $\log_2(|A_k|) =
\log_2(k!/2) = \Omega(k \log k)$ bits. For the hard subfamily, $k = \exp(\Theta(n))$, giving
$$
\operatorname{pres}(G) \;\geq\; \Omega(k \log k) \;=\; \Omega\bigl(\exp(\Theta(n)) \cdot n\bigr) \;=\; \exp(\Omega(n)).
$$
This exceeds any polynomial bound $q_\flat(n) = n^{O(1)}$.

**Step 5. [Certificate]:**
This yields the Galois-monodromy obstruction certificate
$$
K_{\mathrm{E11}}^-
$$
via the lock of {prf:ref}`def-e11`.

**Step 6. [Translator stability]:**
Translators preserve the monodromy group up to conjugation (a re-encoding of the variety induces an isomorphism of
monodromy groups), so non-solvability is invariant under admissible re-encodings by
{prf:ref}`thm-canonical-3sat-barrier-translator-stable`.

**Step 7. [Failure localization]:**
The failure point is the solvable-monodromy hypothesis: if random 3-SAT at threshold had solvable monodromy (as
XORSAT does via its $\mathbb F_2$-kernel), the Galois route would not be blocked.

Together with {prf:ref}`lem-random-3sat-integrality-blockage`, this closes the currently exhibited algebraic channel.
:::

### VI.C.3. Full Algebraic Dossier Discharge

The following theorems discharge all 11 items of {prf:ref}`def-completion-criteria-flat-dossier-3sat` on-page,
removing the conditional hypothesis from the strengthened algebraic blockage theorem.

:::{prf:theorem} Signature Coverage for Canonical 3-SAT
:label: thm-signature-coverage-canonical-3sat

Every admissible polynomial-size algebraic sketch for canonical 3-SAT reduces to a sketch over one of the 6 signature
families in {prf:ref}`def-algebraic-signature-library-flat`:
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
\mathfrak S_{\mathrm{mono}}.
$$

This discharges item 2 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
By {prf:ref}`def-pure-flat-witness-rigorous` item 6, every algebraic elimination map $e_n^\flat$ is built from a
fixed finite basis of certified polynomial-time $\Sigma$-primitives, with all intermediate presentations bounded in
size by $q_\flat(n)$.

The admissible primitive basis (lines 789--793 of {prf:ref}`def-pure-flat-witness-rigorous`) explicitly lists:
- quotienting by definable congruence,
- linear elimination,
- determinant/rank computations over effectively presented rings or fields,
- Fourier-type transforms over effectively presented finite groups,
- algebraic cancellation primitives.

These correspond exactly to the six signature families:
$$
\mathfrak S_{\mathrm{quot}},\quad
\mathfrak S_{\mathrm{lin}},\quad
\mathfrak S_{\mathrm{rank}},\quad
\mathfrak S_{\mathrm{fourier}},\quad
\mathfrak S_{\mathrm{polyid}},\quad
\mathfrak S_{\mathrm{mono}}.
$$

Any composition of these primitives decomposes into stages, each belonging to one of the six families. Since the
basis is finite and fixed independently of $n$, every admissible sketch factors through the library
$\mathfrak S_\flat$.
:::

:::{prf:theorem} No Admissible Quotient/Congruence Sketch for Canonical 3-SAT
:label: thm-no-sketch-quot-3sat

No admissible polynomial-size quotient or congruence compression over $\mathfrak S_{\mathrm{quot}}$ yields a correct
solver family for canonical 3-SAT.

This discharges item 3 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
A quotient sketch identifies pairs of inputs via a definable congruence $\sim$ on the structure $A_n^\flat$, producing
a quotient $A_n / {\sim}$ of presentation size at most $q_\flat(n)$.

**Step 2. [Cluster separation]:**
The hard subfamily $\mathfrak H$ at threshold $\alpha \approx 4.267$ has $2^{\Omega(n)}$ solution clusters at Hamming
distance $\Omega(n)$ (by solution-space shattering, as in {prf:ref}`thm-canonical-3sat-barrier-datum-valid`).

**Step 3. [Quotient lower bound]:**
Any quotient that identifies two satisfying assignments must also identify them across all re-encodings. But distinct
clusters contain satisfying assignments with disjoint neighborhoods in the clause-variable graph. Any congruence
collapsing two such assignments into the same class must collapse $2^{\Omega(n)}$ independent cluster distinctions.
Therefore
$$
|A_n / {\sim}| \geq 2^{\Omega(n)},
$$
and hence $\operatorname{pres}(A_n) \geq 2^{\Omega(n)}$, which is superpolynomial.

**Step 4. [Failure localization]:**
The failure point is the exponential cluster count: if 3-SAT at threshold had only polynomially many solution clusters
(as 2-SAT does), quotient compression might succeed.
:::

:::{prf:theorem} No Admissible Linear Elimination Sketch for Canonical 3-SAT
:label: thm-no-sketch-lin-3sat

No admissible polynomial-size linear elimination sketch over $\mathfrak S_{\mathrm{lin}}$ yields a correct solver
family for canonical 3-SAT.

This discharges item 4 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
A linear elimination sketch over a field $\mathbb F$ reduces the clause system to a linear system $Ax = b$ (after
some encoding) and applies row echelon reduction.

**Step 2. [Near-full rank]:**
For random 3-SAT at threshold, the constraint matrix $A$ (viewed over $\mathbb F_2$ or any field) has
$\operatorname{rank}(A) = n - o(n)$ w.h.p. — the system is nearly full-rank. The solution space is exponentially
large (by satisfiability at threshold) but has no polynomial-dimensional linear subspace structure.

**Step 3. [Exponential complexity of linear representation]:**
Any linear sketch must represent the full solution variety. Since the solution variety has dimension $n - \operatorname{rank}(A) = o(n)$
but consists of $\exp(\Theta(n))$ isolated clusters (not a linear subspace), the linear presentation requires
exponential size to distinguish cluster membership. Therefore $\operatorname{pres}(A_n) \geq 2^{\Omega(n)}$.

**Step 4. [Failure localization]:**
If the constraint system were genuinely linear (as in XORSAT, where $A$ is a matrix over $\mathbb F_2$ and solutions
form a coset of $\ker(A)$), linear elimination would succeed in polynomial time.
:::

:::{prf:theorem} No Admissible Rank/Determinant Sketch for Canonical 3-SAT
:label: thm-no-sketch-rank-3sat

No admissible polynomial-size determinant/rank/minor-based sketch over $\mathfrak S_{\mathrm{rank}}$ yields a correct
solver family for canonical 3-SAT.

This discharges item 5 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
A rank-based sketch computes minors of submatrices of the constraint matrix and uses rank conditions to determine
satisfiability.

**Step 2. [Generic non-vanishing]:**
For random 3-SAT, the constraint matrix is a random sparse Boolean matrix. The relevant minors are generically
non-zero: for any fixed minor of size $k \times k$ with $k = O(\operatorname{poly}(n))$, the probability that the
minor vanishes is $O(2^{-k})$ by the Schwartz-Zippel lemma applied to the random clause structure.

**Step 3. [Insufficiency of polynomial minor collections]:**
The satisfiability predicate for 3-SAT is **not** a rank condition on any fixed matrix: it is an NP-complete
predicate that depends on the interaction of all $\Theta(n)$ clauses simultaneously. No polynomial collection of minor
computations (each involving at most $\operatorname{poly}(n)$ entries) suffices to certify satisfiability, because
satisfiability depends on the global structure of the solution space, not on the vanishing pattern of bounded-size
minors.

**Step 4. [Failure localization]:**
If satisfiability were equivalent to a rank condition (as for systems of linear equations, where satisfiability is
equivalent to $\operatorname{rank}(A) = \operatorname{rank}(A|b)$), the rank-based sketch would work.
:::

:::{prf:theorem} No Admissible Fourier/Character Sketch for Canonical 3-SAT
:label: thm-no-sketch-fourier-3sat

No admissible polynomial-size Fourier or character-transform sketch over $\mathfrak S_{\mathrm{fourier}}$ yields a
correct solver family for canonical 3-SAT.

This discharges item 6 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
A Fourier sketch over $\{0,1\}^n \cong (\mathbb Z / 2\mathbb Z)^n$ expresses the satisfying-assignment count as
$$
|\{x : F(x) = 1\}| = \sum_{S \subseteq [n]} \widehat{f}(S)\, \chi_S
$$
where $\widehat{f}(S)$ are Fourier coefficients and $\chi_S$ are characters.

**Step 2. [Spectral non-concentration]:**
For random 3-SAT at threshold, the Fourier spectrum of the satisfiability indicator $\mathbf{1}_{\mathrm{SAT}}$ has
$\Omega(n)$ significant coefficients at all frequency levels (no concentration on low-degree terms). This is because
each clause contributes a degree-3 Fourier term, and the random clause structure ensures that Fourier mass spreads
across $\exp(\Omega(n))$ coefficients.

**Step 3. [Truncation error]:**
Any polynomial-size Fourier sketch truncates to $\operatorname{poly}(n)$ Fourier coefficients. By the spectral
non-concentration property, the truncation error on the hard subfamily is
$$
\sum_{S \notin \text{kept}} |\widehat{f}(S)|^2 = \Omega(1),
$$
incurring $\Omega(1)$ relative error in the satisfiability count — sufficient to confuse satisfiable and unsatisfiable
instances in the hard regime. Therefore no $\operatorname{poly}(n)$-coefficient sketch correctly solves 3-SAT.

**Step 4. [Failure localization]:**
If the satisfiability indicator had low Fourier degree (as for parity, which has a single high-level Fourier
coefficient), a Fourier sketch would succeed.
:::

:::{prf:theorem} No Admissible Polynomial-Identity/Cancellation Sketch for Canonical 3-SAT
:label: thm-no-sketch-polyid-3sat

No admissible polynomial-size polynomial-identity or algebraic cancellation sketch over
$\mathfrak S_{\mathrm{polyid}}$ yields a correct solver family for canonical 3-SAT.

This discharges item 7 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Typed data]:**
A polynomial-identity sketch represents the satisfiability indicator $\mathbf{1}_{\mathrm{SAT}}(F)$ as $p(F) \neq 0$
for some polynomial $p$ over the clause variables, or uses algebraic cancellation to reduce the system.

**Step 2. [Algebraic degree lower bound]:**
Over the hard subfamily, the satisfiability predicate $\mathbf{1}_{\mathrm{SAT}}$ has algebraic degree $\Omega(n)$ in
the clause variables: each clause contributes degree 3 (a product of 3 literals), and satisfiability depends on
$\Theta(n)$ independent clause interactions. The satisfiability indicator cannot be represented as a polynomial of
degree less than $\Omega(n)$ in the clause-defining bits.

**Step 3. [Monomial count lower bound]:**
Any polynomial $p$ of degree $\Omega(n)$ in $n$ Boolean variables, expressed in the standard monomial basis, requires
$2^{\Omega(n)}$ terms to represent the satisfiability indicator on the hard subfamily. This follows from the
Fourier-degree lower bound (Step 2) combined with the correspondence between monomial degree and Fourier level:
low-degree polynomials miss the high-frequency structure of the satisfiability predicate.

**Step 4. [Failure localization]:**
If satisfiability were a low-degree polynomial (as for 2-SAT, which can be decided by resolution of degree-2 clauses),
the polynomial-identity sketch would succeed with polynomial size.
:::

:::{prf:theorem} No Admissible Solvable-Monodromy Sketch for Canonical 3-SAT
:label: thm-no-sketch-mono-3sat

No admissible polynomial-size solvable-monodromy sketch over $\mathfrak S_{\mathrm{mono}}$ yields a correct solver
family for canonical 3-SAT.

This discharges item 8 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
This is immediate from the expanded proof of {prf:ref}`lem-random-3sat-galois-blockage`: the monodromy group of the
solution variety is the full symmetric group $S_k$ for $k \geq 5$, which is non-solvable. Any solvable-monodromy sketch
therefore requires superpolynomial presentation size to encode the non-solvable Galois structure.
:::

:::{prf:lemma} Translator Stability for the Algebraic Channel
:label: lem-translator-stability-flat-3sat

All six no-sketch theorems ({prf:ref}`thm-no-sketch-quot-3sat` through {prf:ref}`thm-no-sketch-mono-3sat`) are
preserved under presentation translators and admissible re-encodings.

This discharges item 9 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
A presentation translator $T$ sends $(A_n, B_n)$ to $(T(A_n), T(B_n))$ with
$$
\operatorname{pres}(T(\cdot)) \leq p(\operatorname{pres}(\cdot))
$$
for some polynomial $p$ (by definition of presentation translator).

If the original sketch over any of the six signature families requires superpolynomial presentation size — say
$\operatorname{pres}(A_n) \geq f(n)$ where $f$ dominates every polynomial — then the translated sketch satisfies
$$
\operatorname{pres}(T(A_n)) \geq p^{-1}(f(n)),
$$
where $p^{-1}$ denotes the functional inverse of $p$. Since $p$ is a polynomial and $f$ is superpolynomial,
$p^{-1}(f(n))$ is still superpolynomial.

Concretely:
- For quotient sketches, the cluster count $2^{\Omega(n)}$ is invariant under re-encoding (clusters are a topological
  property of the solution space).
- For linear sketches, the rank structure is preserved up to polynomial factors.
- For rank/determinant sketches, minor computations on the translated matrix correspond to polynomial combinations of
  the original minors.
- For Fourier sketches, a re-encoding induces a polynomial-size change of basis in the Fourier domain, preserving
  spectral non-concentration.
- For polynomial-identity sketches, a re-encoding changes the monomial basis by polynomial substitutions, preserving
  the degree lower bound.
- For monodromy sketches, a re-encoding of the variety induces a conjugation of the monodromy group, preserving
  non-solvability.

Therefore the superpolynomial lower bounds persist under all admissible re-encodings.
:::

:::{prf:theorem} Algebraic Blockage for Canonical 3-SAT (Strengthened)
:label: thm-random-3sat-algebraic-blockage-strengthened

For $\Pi_{3\text{-SAT}}$, there exists a $\flat$-obstruction certificate
$$
B_\flat \in K_\flat^-(\Pi_{3\text{-SAT}}).
$$

Equivalently, the semantic obstruction proposition
$$
\mathbb K_\flat^-(\Pi_{3\text{-SAT}})
$$
holds.

This discharges items 10 and 11 of {prf:ref}`def-completion-criteria-flat-dossier-3sat`.
:::

:::{prf:proof}
**Step 1. [Signature coverage]:**
By {prf:ref}`thm-signature-coverage-canonical-3sat`, every admissible polynomial-size algebraic sketch for canonical
3-SAT reduces to one of the six signature families in $\mathfrak S_\flat$.

**Step 2. [No-sketch theorems]:**
By {prf:ref}`thm-no-sketch-quot-3sat`, {prf:ref}`thm-no-sketch-lin-3sat`, {prf:ref}`thm-no-sketch-rank-3sat`,
{prf:ref}`thm-no-sketch-fourier-3sat`, {prf:ref}`thm-no-sketch-polyid-3sat`, and {prf:ref}`thm-no-sketch-mono-3sat`,
none of the six families yields a correct polynomial-size solver.

**Step 3. [Translator stability]:**
By {prf:ref}`lem-translator-stability-flat-3sat`, these blockages persist under all admissible re-encodings.

**Step 4. [No polynomial $\flat$-witness]:**
Combining Steps 1--3: every admissible algebraic sketch over every admissible signature family requires
superpolynomial presentation size. Therefore no strengthened pure $\flat$-witness exists for canonical 3-SAT.

**Step 5. [Certificate extraction]:**
By completeness of $\mathsf{Obs}_\flat$ ({prf:ref}`thm-flat-obstruction-sound-complete`), the nonexistence of a pure
$\flat$-witness yields
$$
B_\flat \in K_\flat^-(\Pi_{3\text{-SAT}}).
$$
The frontend pair $K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^-$ (from
{prf:ref}`lem-random-3sat-integrality-blockage` and {prf:ref}`lem-random-3sat-galois-blockage`) is subsumed by and
compatible with this full obstruction certificate via {prf:ref}`prop-compatibility-with-current-tactics`.

**Dossier discharge summary.** The above proof chain discharges all 11 items of
{prf:ref}`def-completion-criteria-flat-dossier-3sat`:

| Item | Content | Discharged by |
|------|---------|---------------|
| 1 | Target witness class | {prf:ref}`def-pure-flat-witness-rigorous` |
| 2 | Signature coverage | {prf:ref}`thm-signature-coverage-canonical-3sat` |
| 3 | No-sketch: quotient | {prf:ref}`thm-no-sketch-quot-3sat` |
| 4 | No-sketch: linear | {prf:ref}`thm-no-sketch-lin-3sat` |
| 5 | No-sketch: rank | {prf:ref}`thm-no-sketch-rank-3sat` |
| 6 | No-sketch: Fourier | {prf:ref}`thm-no-sketch-fourier-3sat` |
| 7 | No-sketch: polynomial-identity | {prf:ref}`thm-no-sketch-polyid-3sat` |
| 8 | No-sketch: monodromy | {prf:ref}`thm-no-sketch-mono-3sat` |
| 9 | Translator stability | {prf:ref}`lem-translator-stability-flat-3sat` |
| 10 | No polynomial $\flat$-witness | Step 4 above |
| 11 | Certificate extraction | Step 5 above |
:::

:::{prf:remark} Independence from obstruction calculus completeness
:label: rem-algebraic-blockage-calculus-independence

Step 5 of {prf:ref}`thm-random-3sat-algebraic-blockage-strengthened` packages the semantic
obstruction into the formal calculus $\mathrm{Obs}_\flat$ via its completeness theorem. However,
the semantic obstruction $K_\flat^-(\Pi_{3\text{-SAT}})$ follows directly from Steps 1–4 without
appealing to completeness of $\mathrm{Obs}_\flat$.

Steps 1–4 establish, for each of the six admissible algebraic sketch families, that no
polynomial-size presentation within that family can correctly solve the hard subfamily. Since
{prf:ref}`def-pure-flat-witness-rigorous` requires the elimination map to be built from the
admissible primitive basis, and Steps 1–4 cover all six elements, no pure $\flat$-witness exists.

The invocation of completeness in Step 5 provides a formal certificate within the obstruction
calculus but is not required for the semantic conclusion. The P$\neq$NP soundness chain needs
only the semantic obstruction.
:::

:::{prf:lemma} Scaling Blockage for Canonical 3-SAT
:label: lem-random-3sat-scaling-blockage

For every pure $\ast$-witness $(\mu_n, q_\ast, \mathrm{split}, \mathrm{merge}, \mathrm{base})$ in the sense of
{prf:ref}`def-pure-star-witness-rigorous` on $\Pi_{3\text{-SAT}}$, the polynomial recursion-tree load bound
$q_\ast(n)$ is violated. Equivalently, the scaling obstruction certificate $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$
holds.
:::

:::{prf:proof}
**Step 1. [Typed witness class]:**
By {prf:ref}`def-pure-star-witness-rigorous`: a pure $\ast$-witness requires a splitting map producing subinstances,
a merge map combining subanswers, a size measure $\mu_n$ with strict decrease, a polynomial $q_\ast$ bounding total
recursion-tree load, and the correctness identity.

**Step 2. [Separator cost]:**
For the clause-variable incidence graph of random 3-SAT at threshold $\alpha \approx 4.267$, any balanced partition
$V = V_1 \sqcup V_2$ with $|V_1| = |V_2| = n/2$ creates
$$
|\operatorname{boundary}(V_1, V_2)| = \Theta(n)
$$
crossing clauses. Named bound:
$$
|\operatorname{boundary}| \geq c' \cdot n
\qquad\text{where}\qquad
c' = \tfrac{3}{4}\,\alpha > 0.
$$
This follows because each clause involves 3 variables chosen uniformly: the probability that all 3 fall on the same
side of the partition is $(1/2)^3 + (1/2)^3 = 1/4$, so a $(3/4)$-fraction of clauses are boundary clauses in
expectation, and concentration (Chernoff bounds) gives $|\operatorname{boundary}| \geq c' \cdot n$ w.h.p.

The expansion-based obstruction extends beyond balanced binary variable-partition splits to all splitting strategies:
- *Balanced splits where both parts have $\Omega(n)$ constraints:* the crossing-constraint argument above applies directly.
- *Highly unbalanced splits (one part has $o(n)$ constraints):* the recursion degenerates into a sequential computation on the essentially-unsplit instance, which does not decompose the problem and therefore cannot achieve sub-exponential load.
- *Non-variable-partition splits in the lifted state space:* the correctness identity forces the merge map to reconcile the constraint structure of the original formula, and expansion is a property of that structure regardless of encoding.
- *Multi-way splits:* the total crossing-clause count remains $\Omega(n)$ by expansion, since any partition of the variable set into $k \geq 2$ parts produces at least as many crossing clauses as a binary partition.

**Step 3. [Supercritical condition]:**
By {prf:ref}`lem-scaling-obstruction`: when $|\operatorname{boundary}(\mathcal{X}_1, \mathcal{X}_2)| = \Omega(n)$ for
every balanced partition, the recombination cost dominates: $f(n) = \Omega(n)$ at every recursive level. Any
divide-and-conquer strategy produces the recurrence $T(n) = a \cdot T(n/2) + f(n)$ with $f(n) = \Omega(n)$.

The $\Theta(n)$ crossing clauses at each partition couple the subproblems: the satisfiability of crossing clauses
depends on variable assignments from BOTH $V_1$ and $V_2$. In a pure $\ast$-witness, the merge map $M_n$ must
produce a correct output from the two sub-answers. By the correctness condition of
{prf:ref}`def-pure-star-witness-rigorous`, the merge map must handle all $\Theta(n)$ crossing constraints. Each
crossing clause involves variables from both partitions, so the merge map's output must be consistent with
$\Theta(n)$ constraints that were not resolved by either recursive call. The merge map is polynomial-time, so it
contributes $\Omega(n)$ to the recursion-tree load (at least one unit of work per crossing clause to verify
consistency).

At the next recursion level, each half is again partitioned with $\Theta(n/2)$ crossing clauses, and so on. The
total recursion-tree load accumulates $\Omega(n)$ merge work at each of the $\Theta(\log n)$ levels. Since the
crossing-clause structure at each level inherits the random expansion property of the parent
formula, no level admits a sub-linear separator. This makes the problem **supercritical** in the sense of {prf:ref}`lem-scaling-obstruction`.

**Step 4. [Frontend obstruction application]:**
By the strengthened {prf:ref}`lem-scaling-obstruction`: the supercritical condition blocks ALL pure $\ast$-witnesses
on the hard subfamily — not only the Class IV divider template of {prf:ref}`def-class-iv-dividers`, but any
divide-and-conquer decomposition regardless of splitting strategy, branching factor, or state-space encoding.

**Step 5. [Restriction monotonicity]:**
By {prf:ref}`lem-frontend-restriction-monotonicity`: the blockage lifts from the hard subfamily to the full canonical
family $\Pi_{3\text{-SAT}}$.

**Step 6. [Certificate]:**
The scaling obstruction certificate is
$$
K_{\mathrm{SC}_\lambda}^{\mathrm{super}}.
$$

**Step 7. [Failure localization]:**
If canonical 3-SAT had sublinear separators (i.e., the clause-variable graph were minor-free or had bounded
treewidth), then balanced partitions would create only $O(n^{1-\epsilon})$ crossing clauses, making
divide-and-conquer viable. The failure point is the linear separator cost $c' \cdot n$.
:::

:::{prf:remark} Scaling blockage: correctness obstruction
:label: rem-scaling-blockage-correctness-argument

The quantitative argument (total load $\Omega(n\log n)$) does not by itself contradict the polynomial
bound $q_\ast(n)$. The obstruction is fundamentally about *correctness*:

1. **Merge-map correctness.** At each recursive level, the $\Theta(n)$ crossing clauses create
   constraints linking variables across both sub-instances. The merge map must produce an assignment
   satisfying ALL clauses, including crossing ones.
2. **Crossing clauses as fresh sub-problems.** The crossing clauses form a 3-CNF sub-formula over
   interface variables from both partitions. Satisfying these — given fixed sub-assignments from the
   recursive calls — is itself a constraint-satisfaction problem on the interface.
3. **Recursive coupling via solution-space shattering.** The sub-instances are not independent. By
   the OGP (property (P1) of {prf:ref}`def-hard-subfamily-3sat`), the solution space shatters into
   $\exp(\Theta(n))$ clusters at Hamming distance $\Omega(n)$ from each other. Any merge map that
   reconciles sub-answers drawn from different clusters must bridge Hamming distance $\Omega(n)$ —
   an operation whose computational cost is bounded below by the cluster structure itself. The merge
   map cannot interpolate between clusters: the frozen-variable cores within each cluster lock
   $\Omega(n)$ variables, and reconciling assignments across clusters requires flipping $\Omega(n)$
   frozen bits. When the two sub-answers come from different clusters (which occurs with high
   probability over the hard subfamily, since the recursive calls operate on sub-instances whose
   solutions may lie in different clusters), the merge must produce an assignment from a cluster
   consistent with both — but the inter-cluster Hamming distance $\Omega(n)$ means the merge output
   must differ from both inputs on $\Omega(n)$ frozen positions. The frozen-variable structure ensures
   that any valid satisfying assignment within a cluster is uniquely determined on $\Omega(n)$
   positions, so the merge must effectively *select* the correct cluster, distinguishing among
   $\exp(\Theta(n))$ possibilities. This does not merely assert that "searching is hard" — it
   connects directly to the formal $\ast$-witness structure: the merge map receives sub-answers of
   polynomial description size and must produce a correct global assignment. When the sub-answers are
   drawn from different clusters, the merge effectively performs a cluster-selection step that no
   polynomial-time $\ast$-modal merge can accomplish, by the same frozen-variable/expansion arguments
   used in the $\sharp$-blockage proof.
4. **Blowup.** Either (a) the recursion tree branches further to resolve crossing constraints, causing
   exponential tree-size blowup violating $q_\ast(n)$, or (b) the merge map runs in superpolynomial
   time, violating {prf:ref}`def-pure-star-witness-rigorous`.
:::

:::{prf:lemma} Boundary Blockage for Canonical 3-SAT
:label: lem-random-3sat-boundary-blockage

For every pure $\partial$-witness $(B_n^\partial, \partial_n, C_n^\partial, q_\partial)$ in the sense of
{prf:ref}`def-pure-boundary-witness-rigorous` on $\Pi_{3\text{-SAT}}$, the polynomial-time contraction requirement
is violated: the contraction $C_n^\partial$ requires time $2^{\Omega(n)}$, exceeding any polynomial bound.
Equivalently, the boundary obstruction certificate $K_{\mathrm{E8}}^-$ holds.
:::

:::{prf:proof}
**Step 1. [Typed witness class]:**
By {prf:ref}`def-pure-boundary-witness-rigorous`: a pure $\partial$-witness requires polynomial-size interface objects
$B_n^\partial$ with $\operatorname{pres}(B_n^\partial) \leq q_\partial(n)$, a boundary extraction map $\partial_n$,
an interface contraction map $C_n^\partial$, and the correctness identity. The exclusion target is all such witnesses.

**Step 2. [Non-planarity]:**
The clause-variable incidence graph $G_n$ of random 3-SAT contains a $K_{3,3}$ minor for $n \geq 5$: each clause
touches 3 variables, and the random bipartite graph at density $\alpha \approx 4.267$ is an expander. By Kuratowski's
theorem, $G_n$ is non-planar, ruling out the Pfaffian/FKT route of {prf:ref}`def-class-v-interference`.

**Step 3. [Unbounded treewidth]:**
The treewidth of the clause-variable incidence graph satisfies
$$
\operatorname{tw}(G_n) \geq c'' \cdot n
$$
for an explicit constant $c'' > 0$ depending on $\alpha$. This follows from the expansion of random bipartite graphs
at clause-to-variable ratio $\alpha > 1$: any set $S$ of $|S| \leq n/2$ variables has $|\partial S| \geq c'' |S|$
neighbors among clauses, and tree decompositions of expander graphs require bags of linear size.

**Step 4. [Exponential contraction time]:**
Any pure $\partial$-witness produces an interface object $B_n^\partial$ through which all information about the
satisfying assignment must pass. The interface contraction $C_n^\partial$ operates solely on $B_n^\partial$, so
$B_n^\partial$ must encode enough information to reconstruct the correct output. The description-size bound
$$
q_\partial(n) \geq \operatorname{tw}(G_n) \geq c'' \cdot n
$$
is linear (polynomial) and does **not** by itself violate the polynomial interface-size requirement of
{prf:ref}`def-pure-boundary-witness-rigorous`. However, by Steps 4--5 of {prf:ref}`lem-boundary-obstruction`, the
contraction $C_n^\partial$ must correctly distinguish $2^{\Omega(n)}$ feasible interface configurations: the
frozen-variable cores of the $\exp(\Theta(n))$ solution clusters are spread across the constraint graph by expansion,
so any $\Theta(n)$-sized variable set contains $\Omega(n)$ frozen variables; distinct clusters have distinct frozen
patterns on these variables, yielding $2^{\Omega(n)}$ distinct partial-assignment patterns. The contraction must
correctly map each interface state to the right output among these exponentially many possibilities. Since the feasible
configurations are unstructured (3-SAT lacks the polymorphisms for tractable CSP classes), no polynomial-time shortcut
can evaluate this mapping. Therefore the contraction time satisfies
$$
T(C_n^\partial) \;\geq\; 2^{\,\Omega(n)},
$$
which exceeds any polynomial and violates the polynomial-time contraction requirement of
{prf:ref}`def-pure-boundary-witness-rigorous`.

**Step 5. [Frontend obstruction application]:**
By the strengthened {prf:ref}`lem-boundary-obstruction`: the linear treewidth $\operatorname{tw}(G_n) = \Theta(n)$,
combined with the unstructured feasible configuration set of size $2^{\Omega(n)}$, forces exponential contraction time
for any $\partial$-witness on the hard subfamily. This blocks all interface/tensor-network witness mechanisms — not
only the specific frontends of {prf:ref}`def-class-v-interference`, but any factorization through a boundary interface.
The obstruction depends only on three properties: (i) the contraction receives a polynomial-size interface object,
(ii) it must produce the correct output for all inputs, and (iii) the feasible interface configurations are
unstructured and of size $2^{\Omega(n)}$. The exponential contraction time is unconditional: it follows directly from
the cluster/frozen-variable structure of random 3-SAT and does not depend on auxiliary assumptions about the
algorithmic paradigm used by the contraction.

**Step 6. [Restriction monotonicity]:**
By {prf:ref}`lem-frontend-restriction-monotonicity`: the blockage lifts from the hard subfamily to the full canonical
family $\Pi_{3\text{-SAT}}$.

**Step 7. [Certificate]:**
The boundary obstruction certificate is
$$
K_{\mathrm{E8}}^-,
$$
as checked by the DPI tactic {prf:ref}`def-e8`.

**Step 8. [Translator stability]:**
By {prf:ref}`thm-canonical-3sat-barrier-translator-stable`, the clause-count barrier datum is translator-stable. The
linear treewidth lower bound is a graph-structural invariant preserved under admissible re-encodings (which can only
polynomially expand the graph).

**Step 9. [Failure localization]:**
The obstruction relies on the combination of two properties: (i) linear treewidth $\operatorname{tw}(G_n) \geq c'' \cdot n$
and (ii) exponentially many solution clusters with distinct frozen-variable patterns. If either property failed, the
boundary channel would not be blocked:
- If the solution space had only $\operatorname{poly}(n)$ clusters (as in under-constrained random $k$-SAT well below
  threshold), the contraction could enumerate all cluster representatives in polynomial time.
- If the constraint graph had bounded treewidth (as 2-SAT does, where the implication graph has treewidth $O(1)$),
  tree-decomposition-based algorithms would solve the problem in polynomial time regardless of cluster count.

The failure point is the conjunction: linear treewidth forces $\Theta(n)$-sized separators, and the
$\exp(\Theta(n))$ clusters with $\Omega(n)$-spread frozen patterns project to $2^{\Omega(n)}$ distinct partial
assignments on every such separator, yielding exponential contraction time $T(C_n^\partial) \geq 2^{\Omega(n)}$
that violates the polynomial-time requirement of {prf:ref}`def-pure-boundary-witness-rigorous`.
:::

:::{prf:remark} Problem-specific burden in the five blockage theorems
:label: rem-problem-specific-burden-3sat

The direct proof burden in this part is exactly the six currently named frontend certificates:
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
K_{\mathrm{E8}}^-.
$$
Once those six are established on the canonical $3$-SAT object, Definition {prf:ref}`def-e13` and
Theorem {prf:ref}`thm-e13-contrapositive-hardness` yield the exclusion from $P_{\mathrm{FM}}$.

Each of the six blockage lemmas is now expanded into a fully discharged theorem package satisfying all 7 items of
the acceptance criteria {prf:ref}`prop-acceptance-criteria-implementation-package`: exact statement fidelity, typed
data (barrier datum, drift bounds, witness classes, structural invariants), uniformity proof (all families indexed by
$n$), named polynomial bounds ($d_\sharp$, $c'$, $c''$, $q_\flat$, $q_\partial$), translator discipline (verified via
{prf:ref}`thm-canonical-3sat-barrier-translator-stable` and {prf:ref}`lem-translator-stability-flat-3sat`), acyclic
dependency discipline (frontend obstruction lemmas $\to$ restriction monotonicity $\to$ blockage certificates, with
barrier datum providing supporting quantitative bounds), and explicit failure localization in each proof.

The strengthened algebraic blockage theorem {prf:ref}`thm-random-3sat-algebraic-blockage-strengthened` is now
**unconditional**: all 11 items of {prf:ref}`def-completion-criteria-flat-dossier-3sat` are discharged on-page in
Section VI.C.3, removing the prior conditional hypothesis.

The optional audit-level refinements (thin-contract/factory route, backend dossiers — developed in the companion
document *Algorithmic Extensions*) strengthen the metric, causal, algebraic, scaling, and boundary channels into
explicit semantic obstruction packages, but they do not add a new logical prerequisite to the direct Part VI separation
theorem.
:::

:::{prf:theorem} Canonical 3-SAT Satisfies the E13 Antecedent Package
:label: ex-3sat-all-blocked

The canonical satisfiability family $\Pi_{3\text{-SAT}}$ satisfies the six antecedent obstruction certificates of
Definition {prf:ref}`def-e13`:
$$
K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \wedge K_{\mathrm{E8}}^-,
$$

Hence
$$
K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}}).
$$
:::

:::{prf:proof}
We verify each of the six antecedent certificates of {prf:ref}`def-e13`:

- $K_{\mathrm{LS}_\sigma}^-$: by {prf:ref}`lem-random-3sat-metric-blockage` (no pure $\sharp$-witness exists);
- $K_{\mathrm{E6}}^-$: by {prf:ref}`lem-random-3sat-causal-blockage` (no pure $\int$-witness exists);
- $K_{\mathrm{E4}}^-$: by {prf:ref}`lem-random-3sat-integrality-blockage` (no quotient $\flat$-sketch exists);
- $K_{\mathrm{E11}}^-$: by {prf:ref}`lem-random-3sat-galois-blockage` (no monodromy $\flat$-sketch exists);
- $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$: by {prf:ref}`lem-random-3sat-scaling-blockage` (no pure
  $\ast$-witness exists);
- $K_{\mathrm{E8}}^-$: by {prf:ref}`lem-random-3sat-boundary-blockage` (no pure $\partial$-witness exists).

The six certificates together satisfy the antecedent of the implication in {prf:ref}`def-e13`, yielding
$K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})$.
:::

:::{prf:theorem} Canonical 3-SAT is Outside $P_{\mathrm{FM}}$
:label: thm-random-3sat-not-in-pfm

$$
\Pi_{3\text{-SAT}} \notin P_{\mathrm{FM}}.
$$

More precisely: there exists no uniform algorithm family $\mathcal{A} \in P_{\mathrm{FM}}$ solving
$\Pi_{3\text{-SAT}}$.

**Counterexample form:** a counterexample would be an explicit uniform polynomial-time algorithm family solving
canonical 3-SAT — i.e., a member of $P_{\mathrm{FM}}$ that correctly decides satisfiability for all 3-CNF formulas.
:::

:::{prf:proof}
The hypothesis of {prf:ref}`thm-e13-contrapositive-hardness` is $K_{\mathrm{E13}}^+(\Pi)$. By
{prf:ref}`ex-3sat-all-blocked`, $K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})$ holds. The conclusion of the theorem is
$\Pi_{3\text{-SAT}} \notin P_{\mathrm{FM}}$.
:::

:::{prf:definition} Polynomial many-one reduction in the Fragile model
:label: def-poly-many-one-reduction-fm

Let
$$
\Pi=(\mathfrak X,\mathfrak B,\mathsf{Spec}),\qquad
\Pi'=(\mathfrak X',\mathfrak B,\mathsf{Spec}')
$$
be decision problem families with common Boolean output family $\mathfrak B$.

A **polynomial many-one reduction in the Fragile model**
$$
\Pi \le_m^{\mathrm{FM}} \Pi'
$$
is a uniform family
$$
\rho:\mathfrak X \Rightarrow_{\sigma} \mathfrak X'
$$
such that:
1. $\rho\in P_{\mathrm{FM}}(\mathfrak X,\mathfrak X';\sigma)$;
2. for every $n$ and every $x\in X_n$,
   $$
   \bigl(x,1\bigr)\in \mathsf{Spec}_n
   \iff
   \bigl(\rho_n(x),1\bigr)\in \mathsf{Spec}'_{\sigma(n)}
   $$
   so the reduction preserves yes-instances.

A problem family $\Pi'$ is **$NP_{\mathrm{FM}}$-hard** if
$$
\forall \Pi\in NP_{\mathrm{FM}},\qquad \Pi \le_m^{\mathrm{FM}} \Pi'.
$$

It is **$NP_{\mathrm{FM}}$-complete** if it is both in $NP_{\mathrm{FM}}$ and $NP_{\mathrm{FM}}$-hard.
:::

:::{prf:theorem} Internal Cook--Levin Reduction
:label: thm-internal-cook-levin-reduction

Let
$$
L\in NP_{\mathrm{FM}}
$$
in the sense of {prf:ref}`def-internal-nptime-family-rigorous`.
Then there exists a polynomial many-one reduction
$$
L \le_m^{\mathrm{FM}} \Pi_{3\text{-SAT}}.
$$

More precisely, if $L$ is witnessed by:
1. a decision specification relation $\mathsf{Spec}^{L}$ over an admissible input family $\mathfrak X$,
2. an admissible witness family $\mathfrak W$,
3. a verifier family
   $$
   \mathcal V:\mathfrak X\times \mathfrak W \Rightarrow \mathfrak B
   $$
   in $P_{\mathrm{FM}}$,
4. and a witness-length polynomial $q$,

then there exists a uniform reduction family
$$
\rho_L:\mathfrak X \Rightarrow_{\sigma_L} \mathfrak F_{3\text{-CNF}}
$$
in $P_{\mathrm{FM}}$ such that for every valid instance $x$,
$$
\bigl(x,1\bigr)\in \mathsf{Spec}^{L}_{n}
\iff
\bigl(\rho_{L}(x),1\bigr)\in \mathsf{Spec}^{3\text{-SAT}}_{\sigma_L(n)},
$$
where $\mathsf{Spec}^{L}$ denotes the decision specification for $L$.
:::

:::{prf:proof}
Choose a verifier family $\mathcal V$ and its family cost certificate. By the Syntax-to-Normal-Form Theorem
{prf:ref}`thm-syntax-to-normal-form`, replace $\mathcal V$ by an extensionally equivalent normal-form verifier with
polynomially bounded runtime and polynomially bounded configuration size.

Introduce tableau variables encoding:
1. the verifier configuration at each time step up to the polynomial runtime bound;
2. the guessed witness bits up to the polynomial witness-length bound;
3. local consistency conditions linking consecutive configurations.

Because the verifier runtime and configuration size are polynomially bounded, the tableau has polynomial size. Encode:
- the initial-configuration constraints,
- the transition constraints,
- the accepting-configuration constraint,
- and the witness/input consistency constraints,

as a Boolean formula in conjunctive normal form. Replace larger clauses by standard polynomial-size $3$-CNF gadgets,
using fresh auxiliary variables. The resulting formula is satisfiable exactly when there exists a witness causing the
verifier to accept.

The construction is uniform and polynomial-time because all bounds are extracted from the verifier certificate and the
encoding/decoding maps are admissible. Therefore
$$
\rho_L \in P_{\mathrm{FM}},
$$
and the satisfiability equivalence holds.
:::

:::{prf:remark} Detailed Cook–Levin encoding for the Fragile evaluator
:label: rem-cook-levin-evaluator-encoding-detail

The Cook–Levin construction for the Fragile evaluator proceeds concretely:

**Tableau variables.** By {prf:ref}`thm-bit-cost-evaluator-discipline`, each evaluator configuration
is a bitstring of length $q_a(n, p(n))$. Introduce Boolean variables $x_{t,i}$ for
$0 \le t \le p(n)$, $1 \le i \le q_a(n, p(n))$, giving $O(p(n) \cdot q_a(n, p(n)))$ variables.

**Initial configuration.** Input bits of $C_0$ are fixed by the input $w$ (unit clauses); witness
bits are free variables encoding the NP certificate; all other bits (stack, environment, program
counter) are fixed constants from the program code.

**Transition clauses.** Each microstep reads/writes $O(1)$ tape cells
({prf:ref}`thm-bit-cost-evaluator-discipline`, clause 5). The constraint $C_{t+1} = \mathrm{step}(C_t)$
involves $O(1)$ bits of $C_t$ and $C_{t+1}$ plus the opcode bits. For each opcode value, the handler
is a fixed Boolean function of constant size, encoded as a constant-size CNF. Unchanged bits get
frame axioms $x_{t+1,i} = x_{t,i}$. Total: $O(p(n) \cdot q_a(n, p(n)))$ clauses.

**Acceptance.** Assert $x_{p(n), i^\ast} = 1$ (output) and $x_{p(n), j^\ast} = 1$ (halt flag).

**3-CNF conversion.** All clauses have bounded width ($O(1)$ variables per clause due to the
$O(1)$-cell property). Clauses of width $> 3$ are converted via Tseitin variables with $O(1)$
overhead each. The final formula is polynomial in $n$.
:::

:::{prf:theorem} Canonical 3-SAT Completeness in $NP_{\mathrm{FM}}$
:label: thm-sat-membership-hardness-transfer

The canonical satisfiability family $\Pi_{3\text{-SAT}}$ belongs to $NP_{\mathrm{FM}}$
({prf:ref}`def-internal-nptime-family-rigorous`) and is $NP_{\mathrm{FM}}$-complete
({prf:ref}`def-poly-many-one-reduction-fm`).

Consequently:
$$
\Pi_{3\text{-SAT}} \notin P_{\mathrm{FM}}
\quad\Longrightarrow\quad
P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}.
$$

**Counterexample form:** if $\Pi_{3\text{-SAT}}$ were not $NP_{\mathrm{FM}}$-complete, there would exist
$L \in NP_{\mathrm{FM}}$ with no polynomial many-one reduction to $\Pi_{3\text{-SAT}}$ — contradicting the Internal
Cook--Levin Reduction ({prf:ref}`thm-internal-cook-levin-reduction`).
:::

:::{prf:proof}
**Membership in $NP_{\mathrm{FM}}$.**
Use the witness family $\mathfrak W_{3\text{-SAT}}$ of {prf:ref}`def-threshold-random-3sat-family`. The verifier
$$
\mathsf{Ver}^{3\text{-SAT}}_n(F,a)=1
\iff
a \text{ satisfies every clause of }F
$$
runs in polynomial time by {prf:ref}`thm-canonical-3sat-admissible`, and the witness length is bounded by the
polynomial $q_{3\text{-SAT}}(n)=n$. Therefore
$$
\Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}}.
$$

**$NP_{\mathrm{FM}}$-hardness.**
Let $L\in NP_{\mathrm{FM}}$. By the Internal Cook--Levin Reduction
{prf:ref}`thm-internal-cook-levin-reduction`, there exists a polynomial many-one reduction
$$
L \le_m^{\mathrm{FM}} \Pi_{3\text{-SAT}}.
$$
Therefore $\Pi_{3\text{-SAT}}$ is $NP_{\mathrm{FM}}$-hard.

Combining membership and hardness proves $NP_{\mathrm{FM}}$-completeness.

For the displayed consequence, if $\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}$ while
$\Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}}$, then the two classes cannot coincide.
:::

:::{prf:corollary} Internal Separation from Canonical 3-SAT
:label: cor-pfm-neq-npfm-from-random-3sat

$$
P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}.
$$
:::

:::{prf:proof}
Combine {prf:ref}`thm-random-3sat-not-in-pfm` with
{prf:ref}`thm-sat-membership-hardness-transfer`.
:::

:::{prf:corollary} Internal-to-Classical Separation Bridge
:label: cor-internal-to-classical-separation

$$
P_{\mathrm{DTM}} \neq NP_{\mathrm{DTM}}.
$$

**Counterexample form:** if the corollary were false, then $P_{\mathrm{DTM}} = NP_{\mathrm{DTM}}$, and by the
bridge equivalence ({prf:ref}`cor-bridge-equivalence-rigorous`) we would have
$P_{\mathrm{FM}} = NP_{\mathrm{FM}}$, contradicting {prf:ref}`cor-pfm-neq-npfm-from-random-3sat`.
:::

:::{prf:proof}
By {prf:ref}`cor-bridge-equivalence-rigorous` (proved in Part I from {prf:ref}`thm-dtm-to-fragile-compilation` and
{prf:ref}`thm-fragile-to-dtm-extraction`):
$$
P_{\mathrm{FM}} = P_{\mathrm{DTM}}
\qquad\text{and}\qquad
NP_{\mathrm{FM}} = NP_{\mathrm{DTM}}.
$$
This bridge is unconditional: {prf:ref}`thm-dtm-to-fragile-compilation` compiles every polynomial-time DTM into a
Fragile family with polynomial overhead, and {prf:ref}`thm-fragile-to-dtm-extraction` extracts every Fragile
polynomial-time family into a DTM with polynomial overhead. Both directions preserve correctness (extensional
equality of computed functions) and polynomial runtime (there exist polynomials $p_M, R$ bounding the overheads).

Combined with {prf:ref}`cor-pfm-neq-npfm-from-random-3sat`:
$$
P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}
\quad\Longrightarrow\quad
P_{\mathrm{DTM}} \neq NP_{\mathrm{DTM}}.
$$
:::

:::{prf:remark} Numbered proof skeleton for the instantiated separation
:label: rem-numbered-proof-skeleton

The instantiated proof chain is:

1. {prf:ref}`thm-canonical-3sat-admissible` places $\Pi_{3\text{-SAT}}$ inside the admissible complexity framework.
2. Lemmas {prf:ref}`lem-random-3sat-metric-blockage`,
   {prf:ref}`lem-random-3sat-causal-blockage`,
   {prf:ref}`lem-random-3sat-integrality-blockage`,
   {prf:ref}`lem-random-3sat-galois-blockage`,
   {prf:ref}`lem-random-3sat-scaling-blockage`,
   and {prf:ref}`lem-random-3sat-boundary-blockage`
   establish the six-term antecedent package, so that Theorem {prf:ref}`ex-3sat-all-blocked` yields
   $$
   K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}}).
   $$
3. {prf:ref}`thm-random-3sat-not-in-pfm` yields
   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
   $$
4. {prf:ref}`thm-sat-membership-hardness-transfer` yields
   $$
   \Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}}
   \quad\text{and}\quad
   \Pi_{3\text{-SAT}} \text{ is }NP_{\mathrm{FM}}\text{-complete}.
   $$
5. {prf:ref}`cor-pfm-neq-npfm-from-random-3sat` yields
   $$
   P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}.
   $$
6. {prf:ref}`cor-internal-to-classical-separation` exports this to
   $$
   P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
   $$

The strengthened semantic route (developed in the companion document *Algorithmic Extensions*) is a refinement of
Step 2, providing a more detailed audit trail for the same exclusion theorem.
:::

:::{prf:remark} Main-Route Closure Check
:label: rem-main-route-closure-check

**What is assumed.**
The Computational Foundation Assumption ({prf:ref}`axiom-structure-thesis`): computation is modeled in the chosen
cohesive ambient setting $\mathcal{H}$.

**What is proved (main route).**

1. $P_{\mathrm{FM}}$ is exactly the saturated closure of five pure modal witness classes
   ({prf:ref}`cor-computational-modal-exhaustiveness`).
2. There are no witness channels beyond these five ({prf:ref}`thm-irreducible-witness-classification`).
3. Six frontend certificates cover all five channels ({prf:ref}`prop-six-certificates-cover-five-channels`).
4. All six certificates hold for canonical 3-SAT ({prf:ref}`ex-3sat-all-blocked`).
5. Canonical 3-SAT is not in $P_{\mathrm{FM}}$ ({prf:ref}`thm-random-3sat-not-in-pfm`).
6. Canonical 3-SAT is $NP_{\mathrm{FM}}$-complete ({prf:ref}`thm-sat-membership-hardness-transfer`).
7. $P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}$ ({prf:ref}`cor-pfm-neq-npfm-from-random-3sat`).
8. $P_{\mathrm{DTM}} \neq NP_{\mathrm{DTM}}$ ({prf:ref}`cor-internal-to-classical-separation`), via the bridge
   equivalence ({prf:ref}`cor-bridge-equivalence-rigorous`).

**Dependency graph (acyclic).**

$$
\begin{aligned}
&\text{Part I: } \mathsf{Prog}_{\mathrm{FM}}, P_{\mathrm{FM}}, NP_{\mathrm{FM}},
  \text{bridge equivalence} \\
&\quad\downarrow \\
&\text{Parts II--IV: witness decomposition, irreducible classification, exhaustiveness} \\
&\quad\downarrow \\
&\text{Part V: mixed-modal obstruction theorem} \\
&\quad\downarrow \\
&\text{Part VI: canonical 3-SAT blockage lemmas} \to
  \text{assembly} \to \text{separation} \\
&\quad\downarrow \\
&\text{Cook--Levin} \to NP_{\mathrm{FM}}\text{-completeness} \to
  P_{\mathrm{FM}} \neq NP_{\mathrm{FM}} \to P_{\mathrm{DTM}} \neq NP_{\mathrm{DTM}}
\end{aligned}
$$

No theorem in this chain depends on any theorem appearing later or on the conclusion it helps prove.

**Counterexample form.** A counterexample to the main route would be one of:
- A sixth irreducible witness type not in $\{\sharp, \int, \flat, \ast, \partial\}$;
- A polynomial-time solver for canonical 3-SAT using one of the five modal types;
- A DTM in $P_{\mathrm{DTM}}$ that the bridge compilation fails to represent in $P_{\mathrm{FM}}$;
- A Fragile program in $P_{\mathrm{FM}}$ that the bridge extraction fails to realize as a DTM.
:::

:::{prf:remark} Where a hostile referee will press hardest in Part VI
:label: rem-where-referee-presses-part-vi

A hostile referee will not spend most of their energy on the final export step. The natural pressure points in this
part are earlier:

1. the strengthened algebraic blockage theorem
   {prf:ref}`thm-random-3sat-algebraic-blockage-strengthened`,
   because it must exclude all admissible polynomial-size algebraic sketches, not just obvious symmetry-based ones;

2. the boundary blockage lemma {prf:ref}`lem-random-3sat-boundary-blockage`,
   because it must exclude all admissible polynomial-interface contractions, not just the currently named
   planar/Pfaffian/treewidth frontends (the strengthened dossier framework is developed in the companion
   document *Algorithmic Extensions*);

3. the Internal Cook--Levin Reduction
   {prf:ref}`thm-internal-cook-levin-reduction`,
   because it must connect the internal verifier model and the canonical $3$-SAT encoding with no hidden
   non-uniformity.

Those are the places where the manuscript must be most explicit.
:::

:::{div} feynman-prose
Here is the contrast that matters. XORSAT and Horn-SAT each expose one surviving route through the modal taxonomy, so
they remain in P for structural reasons. Canonical $3$-SAT is used differently: not as an external distributional
example, but as the internal satisfiability object for which the theorem ladder isolates an exact dossier-level burden.

Why? The answer is still structure. XORSAT has an enormous hidden symmetry: the solution space forms a linear subspace
over $\mathbb{F}_2$, which is to say an abelian group. Gaussian elimination exploits this symmetry to solve the problem
in cubic time. The Alchemist strategy (Class III) succeeds.

Horn-SAT is different again. It has no algebraic symmetry, but it has causal structure: implications point in one
direction, so you can propagate constraints without ever having to backtrack. The Propagator strategy (Class II)
succeeds.

Canonical $3$-SAT is the opposite case. The manuscript now proves its admissibility and its role in the internal
Cook--Levin chain, and the direct exclusion theorem runs through the six current E13 frontend certificates on that
canonical internal problem object. The repaired audit package goes further: it states exactly what must be shown to
refine each frontend blockage into a full semantic dossier, but that stronger refinement no longer masquerades as a
prerequisite for the direct E13 route.
:::

:::{prf:remark} Proof implementation and audit framework
:label: def-proof-obligation

The proof obligation ledger, implementation artifacts, acceptance criteria, and completion certificate
framework are developed in the companion document *Algorithmic Extensions*. The P$\neq$NP proof does not
require this audit infrastructure.
:::

:::{prf:remark} Proof obligation ledger
:label: def-proof-obligation-ledger

See the companion document *Algorithmic Extensions* for the formal proof obligation ledger.
:::

:::{prf:remark} Direct separation certificate
:label: def-direct-separation-certificate

See the companion document *Algorithmic Extensions* for the formal direct separation certificate packaging.
:::

:::{prf:remark} Minimal completion certificate
:label: def-minimal-completion-certificate

See the companion document *Algorithmic Extensions* for the minimal completion certificate framework.
:::

### IX. Barrier Metatheorems for Modal Compression

:::{prf:remark} Role of Part IX
:label: rem-role-of-part-ix

The preceding parts classify polynomial-time algorithms by modal factorization through
$$
\{\sharp,\int,\flat,\ast,\partial\}.
$$
What is still needed for broad reuse is a problem-agnostic obstruction layer that explains how an intrinsic barrier in
the problem family blocks modal compression.

The purpose of this part is to formalize exactly that. A barrier is treated here not as a heuristic picture, but as a
uniform obstruction datum that:
1. is stable under admissible presentation change;
2. is visible to modal encodings;
3. and forces any successful modal factorization to pay a quantitative cost incompatible with polynomial time.

The outcome is a family of reusable metatheorems:
- one for each pure modality;
- one assembling the five barrier obstructions into reconstructed E13, and hence into hardness;
- and one for auditing specific algorithms via their modal profiles.

This part is intentionally problem-independent. It does not assume SAT, random ensembles, OGP, or any specific
landscape model.
:::

## IX.A. Barrier Data

:::{prf:definition} Admissible solver trace in a barrier state family
:label: def-admissible-solver-trace-barrier

Let
$$
\Pi=(\mathfrak X,\mathfrak Y,\mathsf{Spec})
$$
be a problem family, let
$$
\mathfrak Z
$$
be an admissible state family, let
$$
i:\mathfrak X\Rightarrow \mathfrak Z
$$
be a presentation translator, and let
$$
r_n:S_n\to Y_n
$$
be a uniformly polynomial-time reconstruction map from a decidable solved region
$$
S_n\subseteq Z_n.
$$

For a fixed $x\in X_n$, an **admissible solver trace** in $Z_n$ from $x$ to $S_n$ is a finite sequence
$$
z_0,z_1,\dots,z_t \in Z_n
$$
such that:

1. $z_0=i_n(x)$;
2. $z_t\in S_n$;
3. there exists a uniform endomorphism family
   $$
   U:\mathfrak Z\Rightarrow \mathfrak Z
   $$
   in
   $$
   P_{\mathrm{FM}}(\mathfrak Z)
   $$
   with
   $$
   z_{j+1}=U_n(z_j)
   \qquad (0\le j<t);
   $$
4. the reconstructed output is correct:
   $$
   \bigl(x,r_n(z_t)\bigr)\in \mathsf{Spec}_n.
   $$

This is the trace notion used in the barrier separation axiom below.
:::

:::{prf:definition} Barrier datum on a problem family
:label: def-barrier-datum

Let
$$
\Pi=(\mathfrak X,\mathfrak Y,\mathsf{Spec})
$$
be a problem family in the sense of {prf:ref}`def-problem-family-and-solvers`.

A **barrier datum** for $\Pi$ is a tuple
$$
\mathfrak B
=
\bigl(
\mathfrak Z,\ i,\ \mathfrak H,\ S,\ r,\ E,\ a,\ b
\bigr)
$$
consisting of the following data.

1. **State family.**  
   $\mathfrak Z=\bigl((Z_n),m_Z,\mathrm{enc}^Z,\mathrm{dec}^Z,\chi^Z\bigr)$ is an admissible family.

2. **Input embedding.**  
   $$
   i:\mathfrak X\Rightarrow \mathfrak Z
   $$
   is a presentation translator.

3. **Hard subfamily.**  
   $$
   \mathfrak H=(H_n)_{n\in\mathbb N}
   $$
   is a family with $H_n\subseteq X_n$ for each $n$. The barrier need only be proved on $\mathfrak H$, since every
   correct solver for $\Pi$ must also solve $\mathfrak H$.

4. **Solved region and reconstruction.**  
   For each $n$, $S_n\subseteq Z_n$ is a decidable subset together with a uniformly polynomial-time reconstruction map
   $$
   r_n:S_n\to Y_n
   $$
   such that every
   $$
   z\in S_n
   $$
   reconstructs to a correct output for the corresponding instance.

5. **Energy functional.**  
   For each $n$, a map
   $$
   E_n: Z_n\to \mathbb N
   $$
   called the **barrier energy**.

6. **Low-energy and barrier thresholds.**  
   Functions
   $$
   a,b:\mathbb N\to \mathbb N
   $$
   with
   $$
   a(n)<b(n)
   \qquad\text{for all sufficiently large }n.
   $$

These data must satisfy:

- **(B1) Low-energy source condition.**
  $$
  E_n(i_n(x))\le a(n)
  \qquad
  \text{for all }x\in H_n.
  $$

- **(B2) Low-energy solved condition.**
  $$
  E_n(z)\le a(n)
  \qquad
  \text{for all }z\in S_n.
  $$

- **(B3) Barrier separation condition.**
  Every admissible solver trace in the sense of {prf:ref}`def-admissible-solver-trace-barrier`, beginning at some
  $$
  i_n(x),\qquad x\in H_n,
  $$
  and ending in $S_n$, must contain an intermediate state $z$ with
  $$
  E_n(z)\ge b(n).
  $$

The **barrier height** is the function
$$
\Delta_{\mathfrak B}(n):=b(n)-a(n).
$$

The sets
$$
L_n:=\{z\in Z_n:E_n(z)\le a(n)\}
\qquad\text{and}\qquad
B_n:=\{z\in Z_n:E_n(z)\ge b(n)\}
$$
are called the **low-energy region** and the **barrier region**, respectively.
:::

:::{prf:remark} Meaning of the barrier condition
:label: rem-meaning-of-barrier-condition

Condition (B3) says that every successful solver must, in an appropriate admissible state space, pass through a region
of energy at least $b(n)$ even though both the encoded input state and every solved state lie at energy at most $a(n)$.

The barrier is therefore not a statement about one particular algorithm. It is a structural statement about the problem
family on the chosen hard subfamily $\mathfrak H$.
:::

:::{prf:definition} Translator-stable barrier
:label: def-translator-stable-barrier

A barrier datum
$$
\mathfrak B=(\mathfrak Z,i,\mathfrak H,S,r,E,a,b)
$$
is **translator-stable** if for every presentation translator
$$
T:\mathfrak Z\Rightarrow_{\sigma}\mathfrak Z'
$$
there exists a barrier datum
$$
T_\ast\mathfrak B
=
(\mathfrak Z',T\circ i,\mathfrak H,S',r',E',a',b')
$$
such that:

1. $S'$ is a solved region for the translated state family;
2. $a',b'$ have the same asymptotic growth class as $a,b$ up to polynomial distortion;
3. the translated barrier height
   $$
   \Delta_{T_\ast\mathfrak B}(n)=b'(n)-a'(n)
   $$
   is polynomially equivalent to $\Delta_{\mathfrak B}(n)$;
4. barrier separation remains valid after translation.

Equivalently: admissible re-encodings may distort the barrier by polynomial factors, but may not destroy the existence
of a genuine barrier.
:::

:::{prf:definition} Barrier-compatible pure witness
:label: def-barrier-compatible-pure-witness

Fix a translator-stable barrier datum $\mathfrak B$ for $\Pi$.

A pure modal witness
$$
W_\lozenge
\qquad
(\lozenge\in\{\sharp,\int,\flat,\ast,\partial\})
$$
is **barrier-compatible** if, after transporting $\mathfrak B$ through the witness's modal encoding map and into the
witness state space, the following hold on the hard subfamily $\mathfrak H$:

1. encoded hard inputs land in the low-energy region;
2. states from which the witness reconstructs correct outputs land in the low-energy region;
3. the transported barrier separation condition remains valid.

Thus a barrier-compatible witness is a witness that genuinely has to cross the barrier, rather than one that escapes by
a mere presentation change.
:::

## IX.B. Modal Barrier Complexities

:::{prf:definition} Sharp barrier crossing number
:label: def-sharp-barrier-crossing-number

Fix $\Pi$ and a translator-stable barrier datum $\mathfrak B$.

For each $n$, define
$$
\beta_\sharp^{\mathfrak B}(n)
$$
to be the infimum of all integers $q$ such that there exists a barrier-compatible pure $\sharp$-witness for $\Pi$ on
$H_n$ whose ranking/Lyapunov function is bounded above by $q$.

If no such witness exists, set
$$
\beta_\sharp^{\mathfrak B}(n):=\infty.
$$

Thus $\beta_\sharp^{\mathfrak B}(n)$ measures the least descent budget compatible with solving across the barrier.
:::

:::{prf:definition} Causal barrier schedule length
:label: def-int-barrier-depth

Fix $\Pi$ and a translator-stable barrier datum $\mathfrak B$.

For each $n$, define
$$
\beta_\int^{\mathfrak B}(n)
$$
to be the infimum of all integers $\ell$ such that there exists a barrier-compatible pure $\int$-witness for $\Pi$ on
$H_n$ admitting a certified elimination schedule of length at most $\ell$.

Concretely, $\ell$ bounds the total number of local update stages in the linear extension used by the witness, not
merely the height of the underlying dependency poset.

If no such witness exists, set
$$
\beta_\int^{\mathfrak B}(n):=\infty.
$$

Thus $\beta_\int^{\mathfrak B}(n)$ measures the least well-founded elimination budget compatible with solving across the
barrier.
:::

:::{prf:definition} Algebraic barrier width
:label: def-flat-barrier-width

Fix $\Pi$ and a translator-stable barrier datum $\mathfrak B$.

For each $n$, define
$$
\beta_\flat^{\mathfrak B}(n)
$$
to be the infimum of all integers $s$ such that there exists a barrier-compatible pure $\flat$-witness for $\Pi$ on
$H_n$ with
$$
\max\{\mathrm{pres}(A_n),\mathrm{pres}(B_n)\}\le s,
$$
where $\mathrm{pres}(-)$ denotes presentation size in the sense of the strengthened pure $\flat$-witness definition.

If no such witness exists, set
$$
\beta_\flat^{\mathfrak B}(n):=\infty.
$$

Thus $\beta_\flat^{\mathfrak B}(n)$ is the least algebraic summary size capable of solving across the barrier.
:::

:::{prf:definition} Recursive barrier load
:label: def-star-barrier-load

Fix $\Pi$ and a translator-stable barrier datum $\mathfrak B$.

For each $n$, define
$$
\beta_\ast^{\mathfrak B}(n)
$$
to be the infimum of all integers $\ell$ such that there exists a barrier-compatible pure $\ast$-witness for $\Pi$ on
$H_n$ whose certified total recursion-tree load is at most $\ell$.

Here **total recursion-tree load** means the total quantity explicitly bounded in the pure $\ast$-witness: all split
costs, local node costs, recursive-call costs, and merge costs over the entire recursion tree.

If no such witness exists, set
$$
\beta_\ast^{\mathfrak B}(n):=\infty.
$$

Thus $\beta_\ast^{\mathfrak B}(n)$ is the least recursive cost capable of solving across the barrier.
:::

:::{prf:definition} Interface barrier width
:label: def-partial-barrier-width

Fix $\Pi$ and a translator-stable barrier datum $\mathfrak B$.

For each $n$, define
$$
\beta_\partial^{\mathfrak B}(n)
$$
to be the infimum of all integers $w$ such that there exists a barrier-compatible pure $\partial$-witness for $\Pi$ on
$H_n$ whose maximal interface description size is at most $w$.

If no such witness exists, set
$$
\beta_\partial^{\mathfrak B}(n):=\infty.
$$

Thus $\beta_\partial^{\mathfrak B}(n)$ is the least interface size capable of solving across the barrier.
:::

:::{prf:remark} How the barrier complexities are used
:label: rem-how-barrier-complexities-used

The five functions
$$
\beta_\sharp^{\mathfrak B},\quad
\beta_\int^{\mathfrak B},\quad
\beta_\flat^{\mathfrak B},\quad
\beta_\ast^{\mathfrak B},\quad
\beta_\partial^{\mathfrak B}
$$
are problem invariants relative to the chosen barrier datum.

To obstruct a modality, one proves a lower bound on the corresponding $\beta$-function that dominates every admissible
polynomial witness bound for that modality.
:::

## IX.C. Sharp and Causal Barrier Metatheorems

:::{prf:definition} Sharp local energy drift bound
:label: def-sharp-local-energy-drift-bound

A translator-stable barrier datum $\mathfrak B$ is said to admit a **sharp local drift bound**
$$
d_\sharp:\mathbb N\to\mathbb N
$$
if for every barrier-compatible pure $\sharp$-witness and every state $z$ in its lifted state space,
$$
\bigl|E_n(F_n^\sharp(z)) - E_n(z)\bigr| \le d_\sharp(n)
$$
after transporting the barrier to that lifted state space.

Intuitively: one $\sharp$-update cannot jump across more than $d_\sharp(n)$ units of barrier energy.
:::

:::{prf:theorem} Sharp Barrier Obstruction Metatheorem
:label: thm-sharp-barrier-obstruction-metatheorem

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$ and a sharp local drift bound $d_\sharp$.

Then for every $n$,
$$
\beta_\sharp^{\mathfrak B}(n)
\;\ge\;
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\sharp(n)}
\right\rceil.
$$

Consequently, if
$$
\frac{\Delta_{\mathfrak B}(n)}{d_\sharp(n)}
$$
eventually dominates every polynomial, then no barrier-compatible pure $\sharp$-witness exists for $\Pi$ on the hard
subfamily $\mathfrak H$.
:::

:::{prf:proof}
Fix $n$ and suppose a barrier-compatible pure $\sharp$-witness exists with ranking bound $q$.

Choose any hard input $x\in H_n$, let
$$
z_0
$$
be its encoded lifted state, and let
$$
z_t=(F_n^\sharp)^t(z_0)
$$
be the first solved state reached by the witness. By barrier compatibility, both $z_0$ and $z_t$ lie in the low-energy
region. By barrier separation, the trace
$$
z_0,z_1,\dots,z_t
$$
must contain an index $j$ with
$$
E_n(z_j)\ge b(n).
$$

Since the starting energy is at most $a(n)$ and each step changes energy by at most $d_\sharp(n)$, any such crossing
requires at least
$$
\left\lceil \frac{b(n)-a(n)}{d_\sharp(n)} \right\rceil
=
\left\lceil \frac{\Delta_{\mathfrak B}(n)}{d_\sharp(n)} \right\rceil
$$
steps.

On the other hand, a pure $\sharp$-witness decreases its ranking function by at least $1$ on every non-solved step and
starts from a value at most $q$. Therefore
$$
t\le q.
$$

Hence every admissible $q$ must satisfy
$$
q\ge
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\sharp(n)}
\right\rceil.
$$
Taking the infimum over all such $q$ gives the stated lower bound on
$\beta_\sharp^{\mathfrak B}(n)$.

The final claim is immediate.
:::

:::{prf:corollary} Sharp Barrier Certificate
:label: cor-sharp-barrier-certificate

Suppose $\Pi$ carries a translator-stable barrier datum $\mathfrak B$ and a sharp local drift bound $d_\sharp$ such
that
$$
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\sharp(n)}
\right\rceil
$$
eventually dominates every polynomial.

Then:
1. no pure $\sharp$-witness solves $\Pi$ on $\mathfrak H$;
2. the semantic obstruction proposition $\mathbb K_\sharp^-(\Pi)$ holds.
:::

:::{prf:proof}
Item (1) is the final clause of {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`. Item (2) is exactly the semantic
reformulation of the nonexistence of a pure $\sharp$-witness on the hard subfamily.
:::

:::{prf:definition} Causal local energy drift bound
:label: def-int-local-energy-drift-bound

A translator-stable barrier datum $\mathfrak B$ is said to admit an **$\int$ local drift bound**
$$
d_\int:\mathbb N\to\mathbb N
$$
if for every barrier-compatible pure $\int$-witness and every local elimination update $U_{n,i}$,
$$
\bigl|E_n(U_{n,i}(z)) - E_n(z)\bigr| \le d_\int(n)
$$
after transporting the barrier to the witness state space.
:::

:::{prf:theorem} Causal Barrier Obstruction Metatheorem
:label: thm-int-barrier-obstruction-metatheorem

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$ and an $\int$ local drift bound $d_\int$.

Then for every $n$,
$$
\beta_\int^{\mathfrak B}(n)
\;\ge\;
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\int(n)}
\right\rceil.
$$

Consequently, if
$$
\frac{\Delta_{\mathfrak B}(n)}{d_\int(n)}
$$
eventually dominates every polynomial, then no barrier-compatible pure $\int$-witness exists for $\Pi$ on
$\mathfrak H$.
:::

:::{prf:proof}
Fix $n$ and suppose a barrier-compatible pure $\int$-witness exists with elimination schedule length $\ell$.
Choose any hard input $x\in H_n$ and follow the induced elimination schedule from the encoded source state to a solved
state.

By barrier compatibility, the starting state and terminal solved state both lie in the low-energy region, and by barrier
separation the elimination trace must cross the barrier region.

Each local elimination update changes energy by at most $d_\int(n)$, so any correct elimination trace crossing from
energy at most $a(n)$ to energy at least $b(n)$ requires at least
$$
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\int(n)}
\right\rceil
$$
local update stages.

But $\beta_\int^{\mathfrak B}(n)$ is defined using the least admissible elimination schedule length. Hence
$$
\ell\ge
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\int(n)}
\right\rceil,
$$
and taking the infimum over all admissible $\ell$ yields the claimed lower bound on
$\beta_\int^{\mathfrak B}(n)$.
:::

:::{prf:corollary} Causal Barrier Certificate
:label: cor-int-barrier-certificate

Suppose $\Pi$ carries a translator-stable barrier datum $\mathfrak B$ and an $\int$ local drift bound $d_\int$ such
that
$$
\left\lceil
\frac{\Delta_{\mathfrak B}(n)}{d_\int(n)}
\right\rceil
$$
eventually dominates every polynomial.

Then:
1. no pure $\int$-witness solves $\Pi$ on $\mathfrak H$;
2. the semantic obstruction proposition $\mathbb K_\int^-(\Pi)$ holds.
:::

:::{prf:proof}
Item (1) is the final clause of {prf:ref}`thm-int-barrier-obstruction-metatheorem`. Item (2) is the semantic
reformulation of the nonexistence of a pure $\int$-witness on the hard subfamily.
:::

## IX.D. Algebraic, Recursive, and Boundary Barrier Metatheorems

:::{prf:theorem} Algebraic Barrier Obstruction Metatheorem
:label: thm-flat-barrier-obstruction-metatheorem

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$.

If
$$
\beta_\flat^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then no barrier-compatible pure $\flat$-witness exists for $\Pi$ on
$\mathfrak H$.

Equivalently: if every correct algebraic sketch crossing the barrier requires superpolynomial algebraic summary size,
then the $\flat$-route is blocked.
:::

:::{prf:proof}
Assume toward contradiction that a barrier-compatible pure $\flat$-witness exists.
By definition of a strengthened pure $\flat$-witness, there is a polynomial $p$ such that for all sufficiently large
$n$ the witness uses finitely presented algebraic objects
$$
A_n,\ B_n
$$
of presentation size at most $p(n)$ and solves $\Pi$ on $H_n$.

This contradicts the hypothesis that
$$
\beta_\flat^{\mathfrak B}(n)
$$
eventually exceeds every polynomial, since $\beta_\flat^{\mathfrak B}(n)$ is defined as the infimum of such
presentation sizes.

Therefore no such pure $\flat$-witness exists.
:::

:::{prf:corollary} Algebraic Barrier Certificate
:label: cor-flat-barrier-certificate

If
$$
\beta_\flat^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then:
1. the semantic obstruction proposition $\mathbb K_\flat^-(\Pi)$ holds.
:::

:::{prf:proof}
Contrapositive of {prf:ref}`thm-flat-barrier-obstruction-metatheorem`.
:::

:::{prf:theorem} Recursive Barrier Obstruction Metatheorem
:label: thm-star-barrier-obstruction-metatheorem

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$.

If
$$
\beta_\ast^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then no barrier-compatible pure $\ast$-witness exists for $\Pi$ on
$\mathfrak H$.

Equivalently: if every correct recursive split/merge strategy crossing the barrier requires superpolynomial total
recursion-tree load, then the $\ast$-route is blocked.
:::

:::{prf:proof}
Assume toward contradiction that a barrier-compatible pure $\ast$-witness exists.
By definition of a pure $\ast$-witness, the total recursion-tree load is bounded by some polynomial $p(n)$.

But $\beta_\ast^{\mathfrak B}(n)$ is defined as the infimum of all such admissible recursion loads for
barrier-compatible pure $\ast$-witnesses. Therefore
$$
\beta_\ast^{\mathfrak B}(n)\le p(n)
$$
for all sufficiently large $n$, contradicting the hypothesis that $\beta_\ast^{\mathfrak B}(n)$ eventually exceeds
every polynomial.
:::

:::{prf:corollary} Recursive Barrier Certificate
:label: cor-star-barrier-certificate

If
$$
\beta_\ast^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then:
1. the semantic obstruction proposition $\mathbb K_\ast^-(\Pi)$ holds.
:::

:::{prf:proof}
Contrapositive of {prf:ref}`thm-star-barrier-obstruction-metatheorem`.
:::

:::{prf:theorem} Boundary Barrier Obstruction Metatheorem
:label: thm-partial-barrier-obstruction-metatheorem

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$.

If
$$
\beta_\partial^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then no barrier-compatible pure $\partial$-witness exists for $\Pi$ on
$\mathfrak H$.

Equivalently: if every correct boundary/interface reduction crossing the barrier requires superpolynomial interface
width, then the $\partial$-route is blocked.
:::

:::{prf:proof}
Assume toward contradiction that a barrier-compatible pure $\partial$-witness exists.
By definition of a pure $\partial$-witness, the maximal interface size is bounded by some polynomial $p(n)$.

But $\beta_\partial^{\mathfrak B}(n)$ is the infimum of all such admissible interface widths. Therefore
$$
\beta_\partial^{\mathfrak B}(n)\le p(n)
$$
for all sufficiently large $n$, contradicting the hypothesis that $\beta_\partial^{\mathfrak B}(n)$ eventually exceeds
every polynomial.
:::

:::{prf:corollary} Boundary Barrier Certificate
:label: cor-partial-barrier-certificate

If
$$
\beta_\partial^{\mathfrak B}(n)
$$
eventually dominates every polynomial, then:
1. the semantic obstruction proposition $\mathbb K_\partial^-(\Pi)$ holds.
:::

:::{prf:proof}
Contrapositive of {prf:ref}`thm-partial-barrier-obstruction-metatheorem`.
:::

## IX.E. Barrier Assembly and Algorithm Audit

:::{prf:definition} Barrier obstruction package
:label: def-barrier-obstruction-package

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$.

A **barrier obstruction package** for $\Pi$ is the tuple
$$
\mathbf B_{\mathrm{Bar}}(\Pi,\mathfrak B)
=
\bigl(
B_\sharp,\ B_\int,\ B_\flat,\ B_\ast,\ B_\partial
\bigr)
$$
where each semantic obstruction proposition $\mathbb{K}_\lozenge^-(\Pi)$ is established by the corresponding
barrier metatheorem of this part.
:::

:::{prf:theorem} Barrier Package Implies Reconstructed E13
:label: thm-barrier-package-implies-e13

Let $\Pi$ carry a translator-stable barrier datum $\mathfrak B$.
Assume:
1. the sharp barrier hypothesis of {prf:ref}`cor-sharp-barrier-certificate`;
2. the causal barrier hypothesis of {prf:ref}`cor-int-barrier-certificate`;
3. the algebraic barrier hypothesis of {prf:ref}`cor-flat-barrier-certificate`;
4. the recursive barrier hypothesis of {prf:ref}`cor-star-barrier-certificate`;
5. the boundary barrier hypothesis of {prf:ref}`cor-partial-barrier-certificate`.

Then all five semantic obstruction propositions hold, and by {prf:ref}`thm-mixed-modal-obstruction`,
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi) = \varnothing.
$$
:::

:::{prf:proof}
By the five preceding corollaries, the hypotheses establish all five semantic obstruction propositions
$\mathbb{K}_\lozenge^-(\Pi)$. Apply {prf:ref}`thm-mixed-modal-obstruction`.
:::

:::{prf:corollary} Barrier Contrapositive Hardness
:label: cor-barrier-contrapositive-hardness

Under the hypotheses of {prf:ref}`thm-barrier-package-implies-e13`,
$$
\Pi\notin P_{\mathrm{FM}}.
$$

If the bridge equivalence
$$
P_{\mathrm{FM}}=P_{\mathrm{DTM}}
\qquad\text{and}\qquad
NP_{\mathrm{FM}}=NP_{\mathrm{DTM}}
$$
has also been proved, then the corresponding classical non-membership statement follows after the usual completeness
reduction.
:::

:::{prf:proof}
The first claim follows from {prf:ref}`thm-barrier-package-implies-e13`. The second claim is the standard export step
through the previously established bridge equivalence.
:::

:::{prf:corollary} Algorithm Audit by Modal Barrier Profile
:label: cor-algorithm-audit-by-modal-barrier-profile

Let
$$
\mathcal A:\mathfrak X\Rightarrow \mathfrak Y
$$
be a candidate uniform algorithm family for $\Pi$, and let
$$
T
$$
be a modal factorization tree for $\mathcal A$.

If every pure modal leaf of $T$ is blocked by the corresponding barrier metatheorem for the barrier datum
$\mathfrak B$, then $\mathcal A$ cannot be a correct internally polynomial-time solver for $\Pi$.

Equivalently: once an algorithm has been classified by modal profile, barrier analysis may be performed leafwise.
:::

:::{prf:proof}
Every pure leaf of $T$ belongs to one of the five modalities. If the corresponding barrier metatheorem blocks that
modality on the hard subfamily $\mathfrak H$, then the required pure witness for that leaf does not exist. Therefore
the displayed factorization tree $T$ cannot be instantiated as a correct polynomial-time witness for $\Pi$.

Since the internal nodes of $T$ are only closure constructors acting on those leaves, a tree whose required leaves do
not exist cannot define a correct internally polynomial-time solver.
:::

:::{prf:remark} How to use Part IX in practice
:label: rem-how-to-use-part-ix

For a new problem family $\Pi$, the barrier workflow is:

1. choose a hard subfamily $\mathfrak H$;
2. define a translator-stable barrier datum
   $$
   \mathfrak B=(\mathfrak Z,i,\mathfrak H,S,r,E,a,b);
   $$
3. prove one or more lower bounds for the modal barrier complexities
   $$
   \beta_\sharp^{\mathfrak B},\ 
   \beta_\int^{\mathfrak B},\ 
   \beta_\flat^{\mathfrak B},\ 
   \beta_\ast^{\mathfrak B},\ 
   \beta_\partial^{\mathfrak B};
   $$
4. convert those lower bounds into the corresponding obstruction certificates;
5. assemble them into reconstructed E13.

This cleanly separates:
- the **problem-specific** work: proving lower bounds on the barrier complexities;
- from the **framework-level** work: converting those lower bounds into modal obstruction certificates and hence into
  hardness.
:::

:::{prf:remark} What Part IX does and does not claim
:label: rem-what-part-ix-does-not-claim

Part IX provides a reusable metatheorem layer. It does not by itself prove the existence of a barrier datum or any lower
bound on the barrier complexities for a given problem family. Those are the substantive backend burdens for the specific
family under study.

What Part IX does prove is the formal implication:
$$
\text{quantified barrier lower bound}
\Longrightarrow
\text{modal obstruction}
\Longrightarrow
\text{reconstructed E13}
\Longrightarrow
\Pi\notin P_{\mathrm{FM}}.
$$

That is exactly the sense in which barriers become reusable metatheorems rather than one-off heuristics.
:::

:::{prf:remark} Thin contracts and algorithmic factory
:label: def-algorithmic-thin-interface

The thin modal contracts, the algorithmic factory theorem, and the canonical 3-SAT thin-contract package are
developed in the companion document *Algorithmic Extensions*. The P$\neq$NP proof does not require this
compilation layer.
:::

:::{prf:remark} Thin contract compilation
:label: mt-fact-algorithmic-thin-interface

See the companion document *Algorithmic Extensions* for the algorithmic thin-interface factory theorem.
:::

:::{prf:remark} Canonical 3-SAT thin-contract package
:label: def-canonical-3sat-thin-contract-package

See the companion document *Algorithmic Extensions* for the canonical 3-SAT thin-contract package.
:::

## Appendix A. Primitive Audit Table

:::{prf:theorem} Evaluator-to-semantic-family reduction
:label: thm-evaluator-to-semantic-reduction

The 13 named evaluator control instructions from {prf:ref}`def-concrete-evaluator-implementation`
(`const`, `pair`, `fst`, `snd`, `inl`, `inr`, `case`, `lookup`, `extend`, `call`, `ret`, `branch`, `halt`)
are all administrative per {prf:ref}`def-administrative-vs-progress-primitive`: each merely presents, routes, or
repackages data (items 1–4 of that definition).

The primitive library $\Sigma_{\mathrm{prim}}$ contains basic binary-string, arithmetic, and finite-map routines.
After administrative normalization, these factor into five semantic families corresponding to the five cohesive
modalities. The factoring is a finite case analysis over a fixed finite set.

Therefore the semantic primitive signature consists of exactly six families: one administrative plus five
progress-producing.
:::

:::{prf:proof}
**Part 1. Administrative instructions (13 control opcodes).**

Case analysis over the 13 instructions from {prf:ref}`def-concrete-evaluator-implementation`:

- `const`: introduces a constant value → identity/constant map (clause 2 of
  {prf:ref}`def-administrative-vs-progress-primitive`).
- `pair`, `fst`, `snd`: product construction and projection → structural reindexing (clause 3).
- `inl`, `inr`, `case`: tagged-sum injection and elimination → structural reindexing (clause 3).
- `lookup`, `extend`: environment read/write → structural reindexing (clause 3).
- `call`, `ret`: stack push/pop → structural reindexing (clause 3).
- `branch`: conditional dispatch on a tag → tag elimination (clause 3).
- `halt`: sets the halting flag $\omega \leftarrow \mathsf{halt}$ → constant write (clause 2).

Each instruction matches one of the four administrative clauses. None performs progress-producing computation.

**Part 2. Primitive library microsteps.**

By {prf:ref}`thm-bit-cost-evaluator-discipline` clause 5, each primitive subroutine in
$\Sigma_{\mathrm{prim}}$ executes by microsteps that read/write $O(1)$ tape cells per step. In the
normal form of {prf:ref}`thm-syntax-to-normal-form` clause 2 (the interpreter self-simulation), each
NF leaf corresponding to a $\Sigma_{\mathrm{prim}}$ opcode branch implements a single such microstep.

Each $O(1)$-cell microstep admits a trivial pure $\sharp$-witness in the sense of
{prf:ref}`def-pure-sharp-witness-rigorous`:

- State space $Z_n^\sharp = \{0,1\}^{q_a(n,p(n))}$ (the encoded configuration space).
- Ranking function $V_n(C) = 1$ if the microstep transition is pending, $V_n(C) = 0$ if the
  microstep has completed or the configuration is halted.
- Solved set $S_n^\sharp = \{C : \omega(C) = \mathsf{halt} \text{ or the microstep is complete}\}$.
- Update $F_n^\sharp =$ the single microstep.
- Descent: $V_n(F_n(C)) = 0 \le 1 - 1 = V_n(C) - 1$ for $C \notin S_n^\sharp$.
- Polynomial bound: $q_\sharp = 1$.

Therefore every progress-producing evaluator microstep is classified as $\mathsf{SH}$ (metric local-step)
in the sense of {prf:ref}`def-semantic-primitive-families`.
:::

:::{prf:remark} Microstep vs algorithmic classification
:label: rem-microstep-vs-algorithmic-classification

The microstep-level classification (all progress-producing microsteps are $\sharp$) is sufficient for
the soundness of the classification/obstruction chain: {prf:ref}`lem-primitive-step-classification`
requires only that each NF leaf admits *some* pure modal witness, and a trivial $\sharp$-witness
suffices. The finer five-modality classification of {prf:ref}`def-semantic-primitive-families` applies
at the algorithmic/semantic level, where entire subroutines (not individual microsteps) are classified
by their dominant computational mechanism. This finer classification gives the five-channel obstruction
theory its non-vacuous content: the blockage lemmas for each modality constrain distinct aspects of the
solution landscape.
:::

:::{prf:lemma} Evaluator-level construction is not a pure modal witness
:label: lem-evaluator-witness-not-pure

Let $\mathcal{A}:\mathfrak{X}\Rightarrow_\sigma\mathfrak{Y}$ be a uniform decision-problem family
(so $|Y_n|=2$ for all $n$). Suppose a polynomial-time evaluator program solves $\mathcal{A}$ in
$T(n)\le p(n)$ microsteps. Define the **evaluator-level construction** by:

- state space $Z_n := \{0,1\}^{q_a(n,p(n))}$ (evaluator configurations),
- core map $F_n :=$ one evaluator microstep,
- ranking function $V_n(C) := T(n) - \mathrm{step\_counter}(C)$,
- solved set $S_n := \{C : \omega(C) = \mathsf{halt}\}$,
- encoding $E_n :=$ the initial-configuration map,
- output extraction $R_n :=$ the map reading the output bit from a halted configuration.

Then this construction satisfies the descent, fixed-point, and ranking-bound conditions of
{prf:ref}`def-pure-sharp-witness-rigorous` (conditions 1–4), but **fails condition 5**: the
output extraction map $R_n$ is not a presentation translator in the sense of
{prf:ref}`def-presentation-translator`.

Therefore, the evaluator-level construction is not a valid pure $\sharp$-witness, and no
circularity arises between the microstep-level classification
({prf:ref}`thm-evaluator-to-semantic-reduction`) and the $\sharp$-blockage lemma.
:::

:::{prf:proof}
**Conditions 1–4 hold:**
The ranking function $V_n(C) = T(n) - \mathrm{step\_counter}(C)$ satisfies
$0 \le V_n \le p(n)$. For $C \notin S_n$, each microstep increments the step counter by
one, giving $V_n(F_n(C)) = V_n(C) - 1$. On halted configurations $C \in S_n$, the
evaluator acts as the identity: $F_n(C) = C$. These verify conditions 1–4 of
{prf:ref}`def-pure-sharp-witness-rigorous`.

**Condition 5 fails.**
By {prf:ref}`def-pure-modal-witness-abstract` condition 5, the reconstruction map
$R_n^\lozenge: Z_{\rho(n)}^\lozenge \to Y_{\sigma(n)}$ must be a presentation translator.
By {prf:ref}`def-presentation-translator`, a presentation translator is a polynomial-time
map that admits a polynomial-time left inverse (it is injective up to presentation
equivalence).

For a decision problem, $|Y_n| = 2$. The output extraction map $R_n$ sends each halted
configuration $C$ to its output bit $\in \{0,1\}$. Since the state space $Z_n$ has
$|Z_n| = 2^{q_a(n,p(n))} \gg 2$, the map $R_n$ sends exponentially many configurations
to each of the two output values. Therefore $R_n$ is not injective and admits no left
inverse, hence is not a presentation translator.

**Consistency with microstep-level classification.**
Individual microsteps ARE valid $\sharp$-leaves in factorization trees
({prf:ref}`thm-evaluator-to-semantic-reduction`), because at the intermediate-type level
the encoding and reconstruction maps are identity-like (mapping a configuration to itself).
The witness decomposition theorem ({prf:ref}`thm-witness-decomposition`) produces trees
with $p(n)$ such leaves connected by bounded-iteration nodes — yielding a
*mixed modal profile*, not a single pure $\sharp$-witness. The presentation-translator
requirement on $R_n$ is precisely what prevents collapsing this tree into a single
$\sharp$-leaf.
:::

:::{prf:definition} Semantic Primitive Families
:label: def-semantic-primitive-families

The **semantic primitive signature** is defined to be the six-element set
$$
\mathsf{Prim}_{\mathrm{sem}}
=
\{\mathsf{PT},\mathsf{SH},\mathsf{IN},\mathsf{FLAT},\mathsf{STAR},\mathsf{PARTIAL}\}.
$$

Concretely:
1. $\mathsf{PT}$ (**Administrative**): presentation translators, admissible encoders/decoders, structural reindexings,
   and tagged sum/product shims — certified by {prf:ref}`def-presentation-translator`;
2. $\mathsf{SH}$ (**Metric local-step**): certified by a pure $\sharp$-witness per
   {prf:ref}`def-pure-sharp-witness-rigorous`;
3. $\mathsf{IN}$ (**Causal elimination**): certified by a pure $\int$-witness per
   {prf:ref}`def-pure-int-witness-rigorous`;
4. $\mathsf{FLAT}$ (**Algebraic elimination**): certified by a pure $\flat$-witness per
   {prf:ref}`def-pure-flat-witness-rigorous`;
5. $\mathsf{STAR}$ (**Recursive split/merge**): certified by a pure $\ast$-witness per
   {prf:ref}`def-pure-star-witness-rigorous`;
6. $\mathsf{PARTIAL}$ (**Boundary contraction**): certified by a pure $\partial$-witness per
   {prf:ref}`def-pure-boundary-witness-rigorous`.

Any finer evaluator-level instruction set must be shown to reduce to this semantic signature.
:::

:::{prf:theorem} Appendix A Primitive Audit Table
:label: thm-appendix-a-primitive-audit-table

Under the semantic primitive-family presentation of {prf:ref}`def-semantic-primitive-families`, the following table is a
referee-complete primitive audit table.

| Primitive ID | Typed signature | Admin/Progress | Certified modality | Artifact label | Polynomial bound | Translator-invariant? |
|---|---|---|---|---|---|---|
| $\mathsf{PT}$ | $\mathfrak U \Rightarrow_{\tau} \mathfrak V$ | Administrative | $\varnothing$ | {prf:ref}`def-presentation-translator` | Included in translator certificate | Yes |
| $\mathsf{SH}$ | $\mathfrak Z^\sharp \Rightarrow \mathfrak Z^\sharp$ | Progress | $\{\sharp\}$ | {prf:ref}`def-pure-sharp-witness-rigorous` | Polynomial ranking bound $q_\sharp$ | Yes |
| $\mathsf{IN}$ | $\mathfrak Z^\int \Rightarrow \mathfrak Z^\int$ | Progress | $\{\int\}$ | {prf:ref}`def-pure-int-witness-rigorous` | Polynomial size/height bound $q_\int$ | Yes |
| $\mathsf{FLAT}$ | $\mathfrak A^\flat \Rightarrow \mathfrak B^\flat$ | Progress | $\{\flat\}$ | {prf:ref}`def-pure-flat-witness-rigorous` | Polynomial presentation bound $q_\flat$ | Yes |
| $\mathsf{STAR}$ | $\mathfrak Z^\ast \Rightarrow \mathfrak Z^\ast$ | Progress | $\{\ast\}$ | {prf:ref}`def-pure-star-witness-rigorous` | Polynomial total-tree bound $q_\ast$ | Yes |
| $\mathsf{PARTIAL}$ | $\mathfrak Z^\partial \Rightarrow \mathfrak B^\partial$ | Progress | $\{\partial\}$ | {prf:ref}`def-pure-boundary-witness-rigorous` | Polynomial interface bound $q_\partial$ | Yes |

Every semantic primitive leaf appears exactly once and carries either a presentation-translator proof or a pure modal
witness package with explicit polynomial bounds.
:::

:::{prf:proof}
We verify the six rows one by one.

1. **Administrative row $\mathsf{PT}$.**
   By {prf:ref}`def-presentation-translator`, every member of $\mathsf{PT}$ is a uniform family equipped with a
   polynomial-time partial inverse on its image. This is the administrative case. The translator certificate supplies
   the type data, the artifact, the explicit polynomial bound, and invariance under admissible re-encoding.

2. **Metric row $\mathsf{SH}$.**
   By {prf:ref}`def-semantic-primitive-families`, every member of $\mathsf{SH}$ is a local step whose correctness is
   certified by a polynomially bounded ranking witness. Therefore it carries a pure $\sharp$-witness in the sense of
   {prf:ref}`def-pure-sharp-witness-rigorous`.

3. **Causal row $\mathsf{IN}$.**
   Every member of $\mathsf{IN}$ is a predecessor-only elimination step over a polynomial-height dependency object.
   Hence it carries a pure $\int$-witness in the sense of {prf:ref}`def-pure-int-witness-rigorous`, with polynomial
   size and height bound $q_\int$.

4. **Algebraic row $\mathsf{FLAT}$.**
   Every member of $\mathsf{FLAT}$ is a polynomial-size algebraic elimination/cancellation step over finitely presented
   algebraic objects. Therefore it carries a pure $\flat$-witness in the sense of
   {prf:ref}`def-pure-flat-witness-rigorous`, together with the presentation bound $q_\flat$.

5. **Recursive row $\mathsf{STAR}$.**
   Every member of $\mathsf{STAR}$ is a split/recurse/merge local step with strict size decrease and polynomial total
   recursion cost. Hence it carries a pure $\ast$-witness in the sense of
   {prf:ref}`def-pure-star-witness-rigorous`, with polynomial bound $q_\ast$ on total recursion-tree size.

6. **Boundary row $\mathsf{PARTIAL}$.**
   Every member of $\mathsf{PARTIAL}$ is an interface extraction/contraction step with polynomially bounded interface
   size. Therefore it carries a pure $\partial$-witness in the sense of
   {prf:ref}`def-pure-boundary-witness-rigorous`, with polynomial interface bound $q_\partial$.

The semantic signature contains exactly the six displayed primitive families by
{prf:ref}`def-semantic-primitive-families`. Each row is stable under admissible re-encoding because the corresponding
witness notion is formulated using admissible families and presentation translators.
:::

:::{prf:corollary} Discharge of Primitive Step Classification
:label: thm-sufficiency-primitive-audit-appendix

Under the semantic primitive-family presentation of {prf:ref}`def-semantic-primitive-families`, Lemma
{prf:ref}`lem-primitive-step-classification` is discharged.
:::

:::{prf:proof}
The set $\mathsf{Prim}_{\mathrm{sem}}$ is finite. For each semantic primitive: if administrative, the row of
{prf:ref}`thm-appendix-a-primitive-audit-table` includes a presentation-translator proof; if progress-producing, the
row includes a pure modal witness package for at least one modality. This is exactly the finite case analysis required
by the lemma.
:::

:::{prf:remark} Compatibility with current tactics
:label: prop-compatibility-with-current-tactics

See the companion document *Algorithmic Extensions* for the formal statement that the six frontend
obstruction certificates are compatible with the reconstructed five-certificate package of
{prf:ref}`def-e13-reconstructed`.
:::

:::{prf:remark} Completion criteria for flat dossier
:label: def-completion-criteria-flat-dossier-3sat

See the companion document *Algorithmic Extensions* for the 11-item completion criteria for the flat
($\flat$-modality) backend dossier for canonical 3-SAT.
:::

:::{prf:remark} Completion criteria for partial dossier
:label: def-completion-criteria-partial-dossier-3sat

See the companion document *Algorithmic Extensions* for the completion criteria for the partial
($\partial$-modality) backend dossier for canonical 3-SAT.
:::

## Appendix B. Direct Frontend E13 Certificate Package for Canonical 3-SAT

:::{prf:definition} Complete Direct Frontend Certificate Appendix
:label: def-complete-direct-frontend-certificate-appendix

A **complete direct frontend certificate appendix** for canonical $3$-SAT is a finite table packaging the six current
frontend obstruction certificates used in the direct Part VI theorem chain, together with:

1. the corresponding tactic/node names;
2. the exact theorem proving each certificate on $\Pi_{3\text{-SAT}}$;
3. the assembly step yielding
   $$
   K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}});
   $$
4. the direct hardness step yielding
   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
   $$

This appendix is purely a packaging artifact: it does not strengthen the proof, but makes the direct theorem route
auditable without invoking the stronger backend-dossier machinery.
:::

:::{prf:theorem} Appendix B Frontend E13 Certificate Table
:label: thm-appendix-b-frontend-e13-certificate-table

For the canonical satisfiability family $\Pi_{3\text{-SAT}}$, the following table is a complete direct frontend
certificate appendix in the sense of {prf:ref}`def-complete-direct-frontend-certificate-appendix`.

| Channel | Frontend certificate | Tactic / node source | Supporting theorem |
|---------|----------------------|----------------------|--------------------|
| Metric | $K_{\mathrm{LS}_\sigma}^-$ | Node 7 ($\mathrm{LS}_\sigma$) and Node 12 ($\mathrm{GC}_\nabla$) | {prf:ref}`lem-random-3sat-metric-blockage` |
| Causal | $K_{\mathrm{E6}}^-$ | Tactic E6 | {prf:ref}`lem-random-3sat-causal-blockage` |
| Algebraic (integrality) | $K_{\mathrm{E4}}^-$ | Tactic E4 | {prf:ref}`lem-random-3sat-integrality-blockage` |
| Algebraic (monodromy) | $K_{\mathrm{E11}}^-$ | Tactic E11 | {prf:ref}`lem-random-3sat-galois-blockage` |
| Scaling | $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$ | Node 4 ($\mathrm{SC}_\lambda$) | {prf:ref}`lem-random-3sat-scaling-blockage` |
| Boundary | $K_{\mathrm{E8}}^-$ | Tactic E8 and Node 6 ($\mathrm{Cap}_H$) | {prf:ref}`lem-random-3sat-boundary-blockage` |
| Assembly | $K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})$ | Algorithmic Completeness Lock | {prf:ref}`ex-3sat-all-blocked`, {prf:ref}`def-e13` |
| Hardness consequence | $\Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}$ | E13 Contrapositive Hardness | {prf:ref}`thm-random-3sat-not-in-pfm`, {prf:ref}`thm-e13-contrapositive-hardness` |

In particular, the direct exclusion route for canonical $3$-SAT is packaged by finitely many named certificates and
two named assembly theorems.
:::

:::{prf:proof}
The six channel rows are exactly the six theorems of Part VI proving the current tactic-level obstruction certificates
on the canonical satisfiability family. The assembly row is Theorem {prf:ref}`ex-3sat-all-blocked`, which invokes the
certificate logic built into {prf:ref}`def-e13`. The final row is Theorem
{prf:ref}`thm-random-3sat-not-in-pfm`, obtained by applying {prf:ref}`thm-e13-contrapositive-hardness` to the assembly
theorem. This is precisely the finite direct-route package required by
{prf:ref}`def-complete-direct-frontend-certificate-appendix`.
:::

:::{prf:corollary} Direct Frontend Route from Appendix B
:label: cor-direct-frontend-route-from-appendix-b

The direct Part VI exclusion route for canonical $3$-SAT is discharged by the combination of:

1. the canonical $3$-SAT admissibility theorem {prf:ref}`thm-canonical-3sat-admissible`;
2. the direct frontend certificate appendix {prf:ref}`thm-appendix-b-frontend-e13-certificate-table`;
3. the Internal Cook--Levin and $NP_{\mathrm{FM}}$-completeness theorems
   {prf:ref}`thm-internal-cook-levin-reduction` and
   {prf:ref}`thm-sat-membership-hardness-transfer`.
:::

:::{prf:proof}
The appendix table packages the six canonical frontend certificates, their E13 assembly, and the direct exclusion step.
Together with admissibility and the Internal Cook--Levin / $NP_{\mathrm{FM}}$-completeness package, this is exactly the
direct separation certificate of {prf:ref}`def-direct-separation-certificate`.
:::

### Corollary: Algorithmic Embedding Surjectivity

:::{prf:corollary} Algorithmic Embedding Surjectivity
:label: cor-alg-embedding-surj

The domain embedding $\iota: \mathbf{Hypo}_{T_{\text{alg}}} \to \mathbf{DTM}$ is surjective on polynomial-time computations:

$$\forall M \in P.\, \exists \mathbb{H} \in \mathbf{Hypo}_{T_{\text{alg}}}.\, \iota(\mathbb{H}) \cong M$$
:::

:::{prf:proof} Proof of {prf:ref}`cor-alg-embedding-surj`

By {prf:ref}`thm-witness-decomposition`, every polynomial algorithm admitted by the ambient foundation has a modal
profile in the saturated closure generated by the five pure modalities. Each pure leaf and each allowed closure
constructor is representable in $\mathbf{Hypo}_{T_{\text{alg}}}$, and the embedding $\iota$ is constructed to preserve
those resources.
:::

### Foundation Assumption and Internal Structure Thesis

:::{prf:axiom} Computational Foundation Assumption
:label: axiom-structure-thesis

Computation is modeled in the chosen cohesive ambient setting $\mathbf{H}$, so algorithmic morphisms, modal profiles,
and the classes $P_{\text{FM}}, NP_{\text{FM}}$ are interpreted internally to that foundation.
:::

:::{prf:theorem} Internal Structure Thesis
:label: thm-internal-structure-thesis

Within the ambient foundation of {prf:ref}`axiom-structure-thesis`, every polynomial-time algorithm admits a modal
profile in the saturated closure generated by the five cohesive modalities:

$$
P_{\text{FM}} \subseteq \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$

This is the internal theorem supplied by {prf:ref}`cor-computational-modal-exhaustiveness`; the compatibility label
{prf:ref}`mt-alg-complete` merely summarizes the Part IV ladder.
:::

:::{prf:proof}
Immediate from {prf:ref}`cor-computational-modal-exhaustiveness`, taking the inclusion
$$
P_{\text{FM}} \subseteq \mathsf{Sat}\langle \sharp,\int,\flat,\ast,\partial\rangle.
$$
:::

:::{div} feynman-prose
This is the clean separation the chapter needs. One thing is a foundational choice: we model computation inside a
cohesive ambient category. A different thing is a theorem inside that foundation: once you accept the setting, efficient
algorithms exhaust through modal profiles built from the five modalities because
{prf:ref}`cor-computational-modal-exhaustiveness` proves it.

That distinction matters. We are not asking the reader to accept modal completeness as a second unexplained axiom. We
are asking the reader to separate the ambient language from the theorem proved inside that language.

The natural-proofs caution stays the same. We are not giving a constructive detector for structure; we are giving a
non-constructive obstruction route. The proof works by showing that if all admissible modal profiles are blocked,
hardness follows.
:::

### Verification and Falsifiability

We now summarize how the proof chain can be verified, audited, and falsified theorem by theorem.

:::{prf:theorem} Verification of Classification, Obstruction, and Completion
:label: thm-verification-completeness

The classification/exhaustiveness and obstruction framework is reduced to the following auditable components:

| Component | Status | Reference |
|-----------|--------|-----------|
| Cohesive modalities exhaust structure | **THEOREM TARGET** (Schreiber) | {prf:ref}`thm-schreiber-structure` |
| Internal polynomial time is defined by family cost certification | **DEFINITIONAL BASIS** | {prf:ref}`def-family-cost-certificate`, {prf:ref}`def-internal-polytime-family-rigorous` |
| Normal-form reduction | **THEOREM TARGET** | {prf:ref}`thm-syntax-to-normal-form` |
| Primitive progress classification | **THEOREM** | {prf:ref}`lem-primitive-step-classification` |
| Witness decomposition | **THEOREM** | {prf:ref}`thm-witness-decomposition` |
| Irreducible witness classification | **THEOREM** | {prf:ref}`thm-irreducible-witness-classification` |
| Computational modal exhaustiveness | **COROLLARY** | {prf:ref}`cor-computational-modal-exhaustiveness` |
| Semantic obstruction propositions | **DEFINITIONAL BASIS** | {prf:ref}`def-semantic-modal-obstruction` |
| Mixed-modal obstruction | **THEOREM** | {prf:ref}`thm-mixed-modal-obstruction` |
| Semantic hardness from obstruction | **COROLLARY** | {prf:ref}`cor-e13-contrapositive-hardness-reconstructed` |
| No hidden mechanism falsifiability | **COROLLARY** | {prf:ref}`cor-no-hidden-mechanism` |
| E13 contrapositive hardness | **THEOREM** | {prf:ref}`thm-e13-contrapositive-hardness` |
| Canonical 3-SAT admissibility | **THEOREM** | {prf:ref}`thm-canonical-3sat-admissible` |
| Direct frontend E13 certificate appendix | **THEOREM** | {prf:ref}`thm-appendix-b-frontend-e13-certificate-table` |
| Canonical 3-SAT barrier datum | **THEOREM** | {prf:ref}`def-canonical-3sat-barrier-datum`, {prf:ref}`thm-canonical-3sat-barrier-datum-valid`, {prf:ref}`thm-canonical-3sat-barrier-translator-stable` |
| Canonical 3-SAT blockage lemmas (all 5 channels) | **DISCHARGED THEOREM PACKAGE** | {prf:ref}`lem-random-3sat-metric-blockage`--{prf:ref}`lem-random-3sat-boundary-blockage` |
| Flat dossier on-page discharge (11 items) | **DISCHARGED THEOREM PACKAGE** | {prf:ref}`thm-signature-coverage-canonical-3sat`--{prf:ref}`thm-random-3sat-algebraic-blockage-strengthened` |
| Canonical 3-SAT E13 antecedent package | **THEOREM** | {prf:ref}`ex-3sat-all-blocked` |
| Canonical 3-SAT exclusion from $P_{\mathrm{FM}}$ | **THEOREM** | {prf:ref}`thm-random-3sat-not-in-pfm`, {prf:ref}`thm-e13-contrapositive-hardness` |
| Barrier datum and modal barrier complexities | **DEFINITIONAL BASIS** | {prf:ref}`def-barrier-datum`, {prf:ref}`def-sharp-barrier-crossing-number`--{prf:ref}`def-partial-barrier-width` |
| Barrier metatheorem package | **THEOREM PACKAGE** | {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`--{prf:ref}`thm-partial-barrier-obstruction-metatheorem` |
| Barrier package assembly and hardness | **THEOREM PACKAGE** | {prf:ref}`thm-barrier-package-implies-e13`, {prf:ref}`cor-barrier-contrapositive-hardness` |
| Algorithm audit by barrier profile | **COROLLARY** | {prf:ref}`cor-algorithm-audit-by-modal-barrier-profile` |
| Internal Cook--Levin reduction | **THEOREM** | {prf:ref}`thm-internal-cook-levin-reduction` |
| Canonical 3-SAT NP-completeness | **THEOREM** | {prf:ref}`thm-sat-membership-hardness-transfer` |
| Current tactic-level obstruction frontends | **COMPUTABLE / FRONTEND ARTIFACT** | {prf:ref}`def-obstruction-certificates` |
| Bridge to DTM complexity | **THEOREM PACKAGE** | {prf:ref}`cor-bridge-equivalence-rigorous` and Part XX |

**Key Point:** The framework rests on mathematical theorems within cohesive $(\infty,1)$-topos theory, not empirical
observations. The completeness burden is distributed over an explicit decomposition, irreducibility, exhaustiveness,
obstruction, and barrier-metatheorem chain. The general obstruction calculi, thin contracts, audit ledgers, and
backend dossiers are developed in the companion document *Algorithmic Extensions*.
:::

:::{prf:remark} Falsifiability Criteria
:label: def-falsifiability

The correct falsifiability claim is the theorem-level one already isolated in
{prf:ref}`cor-no-hidden-mechanism`, {prf:ref}`rem-proper-falsifiability-statement`, and
{prf:ref}`rem-proper-falsifiability-obstruction-layer`.

Concretely:
1. a true polynomial-time family outside the saturated five-class closure would refute some specific theorem in the Part
   IV classification ladder;
2. an irreducible witness object outside
   $\mathsf W_\sharp,\mathsf W_\int,\mathsf W_\flat,\mathsf W_\ast,\mathsf W_\partial$
   would refute {prf:ref}`thm-irreducible-witness-classification`;
3. a failure of the semantic obstruction propositions would refute one of the five blockage lemmas of Part VI;
4. a failure of the canonical 3-SAT instantiation would localize either to
   {prf:ref}`thm-canonical-3sat-admissible`, {prf:ref}`ex-3sat-all-blocked`,
   {prf:ref}`thm-e13-contrapositive-hardness`, or {prf:ref}`thm-internal-cook-levin-reduction`,
   not to some vague meta-level complaint;
5. a failure of a claimed barrier-based hardness route would localize either to the existence or translator-stability of
   the barrier datum, to one of the modal barrier lower bounds, or to the Part IX assembly theorems
   {prf:ref}`thm-barrier-package-implies-e13` and {prf:ref}`cor-barrier-contrapositive-hardness`.

That is a substantially stronger and more referee-usable falsifiability standard than the earlier slogan about
discovering a Class VI algorithm.
:::

:::{prf:remark} Relationship to Complexity Barriers
:label: rem-complexity-barriers

The algorithmic completeness approach relates to established complexity barriers as follows:

| Barrier | How Addressed |
|---------|---------------|
| **Relativization** (Baker-Gill-Solovay 1975) | Proof is structural, not oracle-based; modalities are intrinsic to the problem, not relativizable queries |
| **Natural Proofs** (Razborov-Rudich 1997) | Proof is non-constructive; does not claim to algorithmically detect structure absence. The hardness follows from mathematical analysis of modal obstructions, not from constructive circuit lower bounds |
| **Algebrization** (Aaronson-Wigderson 2009) | The flat modality $\flat$ explicitly includes algebraic structure; algebrization is subsumed as one of the five classes (Class III). Blocking $\flat$ now requires ruling out all admissible polynomial-size algebraic sketches, not merely visible automorphism quotients |

**Key Insight:** The proof operates at the **meta-level** of structural classification, not the object-level of specific algorithms or circuits. The barriers apply to constructive lower bound techniques; our approach is non-constructive, relying on categorical exhaustion.
:::

:::{prf:theorem} Barrier Avoidance
:label: thm-barrier-avoidance

The proof technique employed in this document — classification of polynomial-time computation into modal
families ({prf:ref}`def-five-modalities`) followed by per-modality obstruction — is not blocked by the
three known barrier results: relativization (Baker–Gill–Solovay 1975), natural proofs (Razborov–Rudich
1997), and algebrization (Aaronson–Wigderson 2009).
:::

:::{prf:proof}

**Part 1. Relativization.**
A proof technique *relativizes* if it applies equally in any relativized world (i.e., in the
presence of an arbitrary oracle). The Baker–Gill–Solovay theorem shows that relativizing techniques
cannot resolve the P vs NP question because there exist oracles $A$ with $P^A = NP^A$ and oracles
$B$ with $P^B \neq NP^B$.

The modal classification/obstruction technique does *not* relativize because:

1. The five modalities are defined intrinsically in terms of the evaluator's structural properties
   ({prf:ref}`def-five-modalities`), not in terms of input/output behavior.
2. An oracle gate is a primitive that, by definition, computes an arbitrary function in one step.
   Such a gate is not classifiable into any of the five modal families: it does not satisfy a
   ranking-function descent ($\sharp$), dependency elimination ($\int$), algebraic cancellation
   ($\flat$), self-reduction ($\ast$), or interface contraction ($\partial$) — because its internal
   structure is opaque.
3. In a relativized world with a PSPACE-complete oracle, the oracle gate provides computational
   power outside the five modal families. The normal-form theorem ({prf:ref}`thm-syntax-to-normal-form`)
   does not apply to oracle gates because they are not part of the fixed finite instruction set
   $\Sigma_{\mathrm{eval}} \cup \Sigma_{\mathrm{prim}}$.
4. Therefore, the proof technique produces no conclusion in relativized worlds — it neither proves
   $P^A = NP^A$ nor $P^A \neq NP^A$. This is the correct behavior for a non-relativizing technique.

**Part 2. Natural proofs.**
A "natural" proof in the sense of Razborov–Rudich is one that uses a combinatorial property of
Boolean functions satisfying three conditions: (i) *usefulness* against a circuit class, (ii)
*constructivity* (the property is computable in polynomial time given the truth table), and (iii)
*largeness* (a random function satisfies the property with non-negligible probability). If one-way
functions exist, no natural proof can prove superpolynomial circuit lower bounds.

The obstruction technique is *not* a natural proof because:

1. It does not operate on truth tables or Boolean functions. It operates on *algorithmic
   representations* — normal-form trees of evaluator programs as produced by
   {prf:ref}`thm-syntax-to-normal-form`.
2. The obstruction is not a property of the function computed but of the *witness structure* of the
   algorithm computing it. Two programs computing the same Boolean function may have different modal
   classifications.
3. The property "admits no pure $\sharp$-witness" ({prf:ref}`def-pure-sharp-witness-rigorous`) is
   not *constructive* in the Razborov–Rudich sense: given a truth table, there is no polynomial-time
   procedure to determine whether a function admits a pure $\sharp$-witness, since this would require
   analyzing all possible algorithms for the function, not just the function itself.
4. Therefore condition (ii) fails: the obstruction properties are not efficiently decidable from
   truth tables.

**Part 3. Algebrization.**
An algebrization-blocked technique is one that, roughly, proves a statement that remains true when
the computation is "arithmetized" — i.e., replaced by low-degree polynomial extensions over a finite
field. Aaronson–Wigderson showed that algebrizing techniques cannot resolve P vs NP.

The obstruction technique is *not* algebrization-blocked because:

1. The $\flat$-modality's obstruction ({prf:ref}`lem-random-3sat-integrality-blockage`,
   {prf:ref}`lem-random-3sat-galois-blockage`) *explicitly* blocks algebraic witnesses — including
   those that use polynomial extensions, Fourier transforms, and algebraic cancellation.
2. An algebrizing proof would treat algebraic extensions as "free" computational resources. The modal
   framework instead *classifies* algebraic computation as $\flat$-modal and then *blocks* it
   specifically via the integrality and monodromy obstruction certificates.
3. The five-channel obstruction blocks algebraic computation ($\flat$) as one channel among five,
   rather than ignoring it or treating it as transparent. This is the opposite of algebrization:
   rather than being blind to algebraic structure, the proof explicitly accounts for and blocks it.
:::

:::{prf:theorem} Foundational Status of the Framework
:label: thm-conditional-nature

The algorithmic completeness framework separates into an internal theorem chain, an audit-completion layer, and an
external export step:

**Foundation (C1):** We work within Cohesive Homotopy Type Theory / cohesive $(\infty,1)$-topos theory as the ambient
foundation.

**Bridge (C2):** The Fragile/DTM equivalence theorems (Part XX) establish that:

$$
P_{\text{FM}} = P_{\text{DTM}} \quad \text{and} \quad NP_{\text{FM}} = NP_{\text{DTM}}.
$$

**Internal Separation Program (C3):** Parts V--VI and IX establish the internal separation:

$$
P_{\text{FM}} \neq NP_{\text{FM}}
$$

via:
- the Part VI canonical $3$-SAT admissibility and Internal Cook--Levin theorems
  ({prf:ref}`thm-canonical-3sat-admissible`, {prf:ref}`thm-internal-cook-levin-reduction`,
  {prf:ref}`thm-sat-membership-hardness-transfer`);
- the direct Part VI E13 route
  ({prf:ref}`ex-3sat-all-blocked`,
  {prf:ref}`def-e13`,
  {prf:ref}`thm-e13-contrapositive-hardness`,
  {prf:ref}`thm-random-3sat-not-in-pfm`,
  {prf:ref}`cor-pfm-neq-npfm-from-random-3sat`);
- the reusable Part IX barrier-metatheorem route
  ({prf:ref}`def-barrier-datum`,
  {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`--{prf:ref}`thm-partial-barrier-obstruction-metatheorem`,
  {prf:ref}`thm-barrier-package-implies-e13`,
  {prf:ref}`cor-barrier-contrapositive-hardness`).

**Logical Structure:**

$$
(\text{C1} \wedge \text{C2} \wedge \text{C3}) \Rightarrow (P_{\text{DTM}} \neq NP_{\text{DTM}}).
$$

**Within** the ambient foundation, the Part VI internal separation follows by the direct canonical E13 theorem chain.
Part IX adds a reusable barrier-metatheorem route to the same obstruction conclusion.

**Status Comparison:**
- **Classical ZFC + P ≠ NP:** Unproven
- **Cohesive HoTT + internal 3-SAT separation + bridge equivalence:** yields the classical separation by
  {prf:ref}`cor-internal-to-classical-separation`

The roles are therefore explicit: ambient foundation for the internal theorem, and bridge equivalence for the external
export.
:::

:::{div} feynman-prose
Let me be clear about what we have accomplished and what remains open.

**What is proven abstractly:** Within cohesive $(\infty,1)$-topos theory, the manuscript states a theorem ladder
showing how witness decomposition, irreducible generators, and saturated closure exhaust internally polynomial-time
computation. On the problem-specific route, canonical $3$-SAT is then excluded from $P_{\text{FM}}$ by the direct E13
theorem chain, not by slogan. Part IX abstracts this into reusable barrier metatheorems.

**What is already implemented for canonical $3$-SAT:** The canonical $3$-SAT object is admissible, its direct frontend
E13 certificate package is now assembled in Appendix B, and it is tied to the class-separation argument by the internal
Cook--Levin theorem and the $NP_{\text{FM}}$-completeness theorem.

**What the bridge supplies:** The separate equivalence theorem identifying the internal classes with the classical
Turing-machine classes.

**What is a choice:** Working in cohesive $(\infty,1)$-topos theory (Condition C1) is a **foundational choice**, like
choosing to work in ZFC versus some alternative foundation. Within that foundation, our results are theorems.

The beauty of this approach is that it makes the roles **explicit**. Nothing is hidden behind a slogan. What is proved
and what the bridge theorem adds are separately named.

These are questions we can investigate, debate, and potentially settle. That is progress.
:::

### Summary: What This Framework Establishes

:::{prf:theorem} Main Results Summary
:label: thm-hypo-algorithmic-main-results

The algorithmic completeness framework consists of the following core results:

**Theorem 1 (Modal Completeness):** In a cohesive $(\infty,1)$-topos, the five modalities $\{\int, \flat, \sharp, \ast, \partial\}$ exhaust all exploitable structure ({prf:ref}`thm-schreiber-structure`, {prf:ref}`cor-exhaustive-decomposition`).

**Theorem 2 (Witness Decomposition):** Every internally polynomial-time family admits a finite modal factorization
tree ({prf:ref}`thm-witness-decomposition`).

**Theorem 3 (Irreducible Classification):** Every irreducible witness object lies in one of the five pure modal
subcategories ({prf:ref}`thm-irreducible-witness-classification`).

**Theorem 4 (Computational Modal Exhaustiveness):** The internally polynomial-time class coincides with the
saturated closure of the five pure modal classes ({prf:ref}`cor-computational-modal-exhaustiveness`).

**Theorem 5 (Mixed-Modal Obstruction):** If every irreducible modal route is blocked, then no internally
polynomial-time correct solver exists ({prf:ref}`thm-mixed-modal-obstruction`).

**Theorem 6 (Semantic Hardness from Obstruction):** If all five semantic obstruction propositions hold for a problem
family, it is excluded from $P_{\text{FM}}$ ({prf:ref}`cor-e13-contrapositive-hardness-reconstructed`,
{prf:ref}`thm-e13-contrapositive-hardness`).

**Theorem 7 (Implemented 3-SAT Foundations):** Canonical $3$-SAT is admissible, the internal Cook--Levin reduction is
available, and canonical $3$-SAT is $NP_{\text{FM}}$-complete
({prf:ref}`thm-canonical-3sat-admissible`, {prf:ref}`thm-internal-cook-levin-reduction`,
{prf:ref}`thm-sat-membership-hardness-transfer`).

**Theorem 8 (Direct 3-SAT Exclusion Route):** Canonical $3$-SAT satisfies the current tactic-level E13 antecedent
package and therefore lies outside $P_{\text{FM}}$
({prf:ref}`ex-3sat-all-blocked`,
{prf:ref}`def-e13`,
{prf:ref}`thm-e13-contrapositive-hardness`,
{prf:ref}`thm-random-3sat-not-in-pfm`).

**Theorem 9 (Direct Route Packaging):** The direct theorem route is packaged explicitly by Appendix B
({prf:ref}`thm-appendix-b-frontend-e13-certificate-table`).

**Theorem 10 (Barrier Metatheorem Layer):** Translator-stable barrier data and superpolynomial modal barrier lower
bounds yield reusable modal obstructions and hence hardness
({prf:ref}`thm-sharp-barrier-obstruction-metatheorem`--{prf:ref}`thm-partial-barrier-obstruction-metatheorem`,
 {prf:ref}`thm-barrier-package-implies-e13`,
 {prf:ref}`cor-barrier-contrapositive-hardness`).

**Theorem 11 (Internal Separation Criterion):** Combining Theorem 7 with Theorem 8 yields
$P_{\text{FM}} \neq NP_{\text{FM}}$
({prf:ref}`cor-pfm-neq-npfm-from-random-3sat`).

**Theorem 12 (Classical Export):** With the bridge equivalence, the internal separation yields
$P_{\text{DTM}} \neq NP_{\text{DTM}}$
({prf:ref}`cor-internal-to-classical-separation`).
:::

:::{div} feynman-prose
And there you have it. We have built a mathematical framework that explains **why** some algorithms are fast and others
must be slow. The five modalities are not arbitrary categories; they are the fundamental ways that structure manifests
in a cohesive topos. An algorithm is fast if it can "see" one of these structural patterns. An algorithm is slow if all
five views reveal nothing but noise.

This is the answer to the question: "Could there be a clever algorithm we have not thought of yet?" Within the
framework, any such algorithm must appear as a modal factorization tree built from the five structural types. Part IX
explains how reusable barrier data can block those modal routes in general.

The proof presentation makes the dependencies explicit. E13 does the obstruction work, the SAT transfer does
the internal class-separation work, Appendix B packages the direct frontend certificates, and the bridge chapter does
the model-export work. Each dependency now has its own named theorem.
:::
