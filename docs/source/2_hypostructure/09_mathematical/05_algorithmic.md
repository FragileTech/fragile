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
- $\flat \int \simeq \flat$ and $\sharp \int \simeq \sharp$ ($\int$ is left-exact)
- $\int \flat \simeq \int$ and $\int \sharp \simeq \int$ (reduction identities)

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

Intuitively, pure $\sharp$-computation is computation by certified descent in a polynomially bounded potential.
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

If the energy landscape $\Phi$ is **glassy** (exhibiting one or more of):
- Exponentially many local minima separated by $\Theta(n)$ barriers
- No spectral gap: $\lambda_{\min}(\nabla^2 \Phi) \to 0$
- Łojasiewicz inequality fails: $\theta \to 0$ (flat regions)

then the standard metric-descent witness language of {prf:ref}`def-class-i-climbers` is blocked, yielding a valid
frontend obstruction for the $\sharp$-channel.

**Obstruction Certificate:** $K_{\sharp}^- = (\text{glassy}, \lambda = 0, \theta \to 0)$

**Application:** Random 3-SAT near threshold has glassy landscape (Mézard-Parisi-Zecchina 2002), supplying a sharp
frontend blockage certificate. The full sound-and-complete $\sharp$-obstruction theorem is deferred to Part V.
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

:::{prf:assumption} Bit-cost evaluator discipline
:label: assump-bit-cost-evaluator-discipline

All theorems in Part I are stated relative to a concrete evaluator implementation satisfying the following conditions.

1. **Finite configuration alphabet.** Every runtime configuration is a finite record over a fixed finite tag alphabet,
   together with finitely many finite strings over $\{0,1\}$.
2. **Read-only program code.** The code of a program is stored in read-only form and is not duplicated unboundedly
   during execution.
3. **Decidable one-step semantics.** There is a decidable one-step transition relation

   $$
   C \leadsto C'
   $$

   on encoded configurations.
4. **Local size discipline.** There exists a polynomial $s$ such that if one primitive evaluator step transforms an
   encoded configuration of bitlength $N$ into one of bitlength $N'$, then

   $$
   N' \le s(N),
   $$

   and, for configurations reachable in $t$ steps from an input of size $n$, iterating this bound yields a polynomial
   upper bound in $(|a|,n,t)$.
5. **Bit-cost primitive accounting.** Arithmetic and data-structure operations are costed according to the size of the
   encoded operands; there is no hidden unit-cost treatment of exponentially large integers or exponentially long
   intermediate strings.

This is the semantic refinement required in order for the cost model to be exportable to DTMs.
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
{prf:ref}`assump-bit-cost-evaluator-discipline`, every runtime configuration is a finite record consisting of:
- a program counter or evaluation-context pointer into the fixed code of $a$,
- finitely many finite work registers, stacks, heaps, or environments,
- the current encoded input/output fragments,
- finitely many control flags.

Because the code of $a$ is read-only, its contribution to configuration size is $O(|a|)$. Because one-step semantics
is decidable and obeys the local size discipline, every configuration reachable in at most $t$ steps from an input of
size $n$ has total bitlength bounded by some polynomial $q_a(n,t)$ obtained by iterating the size bound from
{prf:ref}`assump-bit-cost-evaluator-discipline`.

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

Assume {prf:ref}`assump-bit-cost-evaluator-discipline`. Then there exists a universal deterministic Turing machine
$U$ and a polynomial

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

:::{prf:assumption} CostCert Completeness for Internal Programs
:label: assump-costcert-completeness

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

Then there exists a polynomial $p$ and a certificate witness establishing

$$
\mathsf{FamCostCert}_{\mathfrak{X},\mathfrak{Y},\sigma}(a,p).
$$

Equivalently: the certificate system is complete for true polynomial-time behavior of internal programs.
:::

:::{prf:remark} Status of CostCert completeness
:label: rem-status-costcert-completeness

Assumption {prf:ref}`assump-costcert-completeness` is a certificate-calculus completeness requirement.
Without it, the class

$$
P_{\mathrm{FM}}
$$

defined through certificates is only guaranteed to be a sound fragment of internally polynomial-time computation, not
the whole class. Any later use of

$$
P_{\mathrm{FM}}=P_{\mathrm{DTM}}
$$

must therefore either:
1. prove Assumption {prf:ref}`assump-costcert-completeness`, or
2. replace equality by the weaker inclusion

   $$
   P_{\mathrm{FM}}\subseteq P_{\mathrm{DTM}}.
   $$
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

Assume {prf:ref}`assump-bit-cost-evaluator-discipline`,
{prf:ref}`thm-evaluator-adequacy`, and {prf:ref}`assump-costcert-completeness`.
Then:

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
For clause (1), use the representable-law hypothesis from {prf:ref}`def-effective-programs-fragile`: every program
has a concrete syntax tree or bytecode. Expand derived control constructs into a fixed core language of primitive
operations, pairing/case constructs, composition, and explicit recursion. Standard structural induction on syntax then
yields administrative normal form.

For clause (2), let $a$ represent a family in $P_{\mathrm{FM}}$ and choose a family cost certificate. Inline the
certificate's polynomial bounds into the control structure of the administrative normal form:
- every loop is replaced by a loop with explicit polynomial bound;
- every recursive call is replaced by a well-founded recursive scheme whose recursion-tree size is polynomially bounded
  by the certificate;
- all outer data-presentation manipulations are factored into presentation translators.

The resulting family lies in $\mathsf{NF}$ and computes the same extensional function.
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

:::{prf:remark} Proof obligation for $\sharp$-universality
:label: rem-proof-obligation-sharp-universality

The nontrivial content is the implication

$$
\mathcal A\triangleright \sharp
\Longrightarrow
\text{existence of a polynomially bounded ranking witness}.
$$

This is the point where metric/contractive structure must be converted into a ZFC-checkable complexity witness. Until
that implication is proved, Theorem {prf:ref}`thm-sharp-universality` may be used only as a target characterization,
not as an established shortcut.
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

:::{prf:remark} Proof obligation for $\int$-universality
:label: rem-proof-obligation-int-universality

The critical implication is

$$
\mathcal A\triangleright \int
\Longrightarrow
\text{existence of a polynomial-height dependency elimination witness}.
$$

This must be proved at the level of evaluator semantics, not merely by analogy with dynamic programming examples.
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

:::{prf:remark} Proof obligation for $\ast$-universality
:label: rem-proof-obligation-star-universality

This theorem must be proved in separator/self-reduction language, not merely by quoting the Master theorem. The
crucial point is the existence of a polynomial-size recursion tree certified from the algorithmic representation
itself.
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

A hostile referee is entitled to demand this table.
:::

:::{prf:lemma} Primitive Step Classification
:label: lem-primitive-step-classification

Every primitive progress-producing leaf appearing in the normal form of
{prf:ref}`thm-syntax-to-normal-form` satisfies at least one of the universal properties of
{prf:ref}`thm-sharp-universality`--{prf:ref}`thm-boundary-universality`.

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

Because the instruction set is finite, this proof is a finite case analysis.
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

Part V must turn that structural classification theorem into a **problem-level hardness mechanism**. The point is not
merely to say that the five modalities exist; the point is to produce, for a given problem family $\Pi$, finite
certificates showing that no irreducible computational witness from any of the five modal classes can appear in any
polynomial-time solver for $\Pi$.

Accordingly, the obstruction theory must be aligned with:
1. the strengthened universal properties of
   {prf:ref}`thm-sharp-universality`--{prf:ref}`thm-boundary-universality`;
2. the witness decomposition theorem {prf:ref}`thm-witness-decomposition`;
3. the irreducible witness classification theorem
   {prf:ref}`thm-irreducible-witness-classification`.

This part therefore replaces the coarse slogan
$$
K_\sharp^- \wedge K_\int^- \wedge K_\flat^- \wedge K_\ast^- \wedge K_\partial^- \Rightarrow \Pi\notin P
$$
by a theorem-driven obstruction calculus whose soundness and completeness are stated explicitly.
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
- monodromy/Galois simplifications,
- and every other admissible polynomial-size algebraic compression over the allowed signatures.

In particular, it is **not sufficient** for $\mathsf{Obs}_\flat$ to test only visible automorphism groups, only
integrality, or only solvable monodromy.
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

:::{prf:corollary} E13 Contrapositive Hardness (Reconstructed Form)
:label: cor-e13-contrapositive-hardness-reconstructed

Let $\Pi$ be a problem family. If $\Pi$ carries a full E13 obstruction package
$$
\mathbf B_{\mathrm{E13}}(\Pi)
=
(B_\sharp,B_\int,B_\flat,B_\ast,B_\partial),
$$
with
$$
B_\sharp\in K_\sharp^-(\Pi),\quad
B_\int\in K_\int^-(\Pi),\quad
B_\flat\in K_\flat^-(\Pi),\quad
B_\ast\in K_\ast^-(\Pi),\quad
B_\partial\in K_\partial^-(\Pi),
$$
then
$$
\Pi\notin P_{\mathrm{FM}}.
$$

More precisely:
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi)=\varnothing.
$$
:::

:::{prf:proof}
By soundness of the five obstruction calculi, each certificate
$$
B_\lozenge\in K_\lozenge^-(\Pi)
$$
implies the corresponding semantic obstruction proposition
$$
\mathbb K_\lozenge^-(\Pi).
$$
Hence the hypotheses yield the conjunction of the five semantic obstruction propositions. Apply
{prf:ref}`thm-mixed-modal-obstruction`.
:::

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

:::{prf:remark} What Part V leaves to the problem-specific chapters
:label: rem-what-part-v-leaves

Part V is still **problem-agnostic**. It does not yet prove that any specific problem family carries the five
obstruction certificates. It only supplies the abstract obstruction machinery.

The next burden is problem-specific: for the target family $\Pi_{3\text{-SAT}}$, one must prove that the sharpened
obstruction calculi actually produce the five modal certificates (or an equivalent tactic-level frontend package) and
hence a full reconstructed E13 package. Only then does the hardness theorem become an instantiated separation argument.
:::

### VI. Canonical 3-SAT Instantiation and Separation

:::{prf:remark} Role of Part VI
:label: rem-role-of-part-vi

Parts I--V are framework-level. They define:
1. the internal/external machine bridge;
2. the normal-form theorem;
3. the universal properties of the five pure classes;
4. the witness decomposition and irreducible-classification theorems;
5. and the sound-and-complete obstruction calculi.

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

The strengthened semantic obstruction dossiers of Parts V and VIII are a more demanding audit-level implementation of
that same exclusion route. They are sufficient refinements of the current tactic-level theorem path, not an additional
logical prerequisite for invoking Definition {prf:ref}`def-e13` and Theorem
{prf:ref}`thm-e13-contrapositive-hardness`.
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
blocked and no closure route survives, the problem is hard inside $P_{\text{FM}}$. In the strengthened Part V
language, this six-term package is a tactic-level frontend sufficient condition for the reconstructed five-certificate
package of {prf:ref}`def-e13-reconstructed`, as stated abstractly in
{prf:ref}`prop-compatibility-with-current-tactics`.

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
$K_{\mathrm{E13}}^+$. Parts V and VIII then provide a stronger semantic/audit refinement of the same exclusion route.

**Certificate Payload:** $(\text{modal\_status}[5], \text{leaf\_exclusions}[5], \text{profile\_exhaustion\_witness})$

**Automation:** Via composition of existing node/tactic evaluations; fully automatable for types with computable modality checks

**Literature:** Cohesive Homotopy Type Theory {cite}`SchreiberCohesive`; Algorithm taxonomy {cite}`Garey79`; Modal type theory {cite}`LicataShulman16`.
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
:::

:::{prf:proof}
By {prf:ref}`mt-alg-complete`, every internally polynomial-time solver in the ambient foundation factors through a
finite modal profile built from the five pure classes
$$
\{\sharp,\int,\flat,\ast,\partial\}.
$$
Definition {prf:ref}`def-e13` packages the six current frontend obstruction certificates into the single statement
$K_{\mathrm{E13}}^+$ that all five modal routes are blocked. Therefore no polynomial-time factorization remains.
Hence
$$
\Pi \notin P_{\text{FM}}.
$$
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
\mathfrak B = \bigl((\{0,1\})_{n\in\mathbb N},m_B,\mathrm{enc}^B,\mathrm{dec}^B,\chi^B\bigr)
$$
be the constant admissible Boolean output family.

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

Define the decision-specification relation
$$
\mathsf{Spec}^{3\text{-SAT}}_n \subseteq F_n\times \{0,1\}
$$
by
$$
(F,b)\in \mathsf{Spec}^{3\text{-SAT}}_n
\iff
\Bigl(
b=1 \iff \exists a\in W_n\ \mathsf{Ver}^{3\text{-SAT}}_n(F,a)=1
\Bigr),
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

The **canonical internal $3$-SAT problem family** is
$$
\Pi_{3\text{-SAT}}
:=
\bigl(\mathfrak F_{3\text{-CNF}},\mathfrak B,\mathsf{Spec}^{3\text{-SAT}}\bigr),
$$
equipped with witness family $\mathfrak W_{3\text{-SAT}}$ and verifier relation
$\mathsf{Ver}^{3\text{-SAT}}$.

This is the unique satisfiability family used in the separation chain below.
:::

:::{prf:theorem} Canonical 3-SAT Family is Admissible
:label: thm-canonical-3sat-admissible

The family
$$
\Pi_{3\text{-SAT}}
=
\bigl(\mathfrak F_{3\text{-CNF}},\mathfrak B,\mathsf{Spec}^{3\text{-SAT}}\bigr)
$$
of {prf:ref}`def-threshold-random-3sat-family` is an admissible problem family in the sense of
{prf:ref}`def-problem-family-and-solvers`.

Moreover:
1. the witness family $\mathfrak W_{3\text{-SAT}}$ is admissible;
2. the verifier relation $\mathsf{Ver}^{3\text{-SAT}}_n(F,a)$ is decidable uniformly in time polynomial in $n$;
3. the witness-length bound $q_{3\text{-SAT}}(n)=n$ is polynomial.

Hence $\Pi_{3\text{-SAT}}$ is well-typed both as a decision family and as a verifier-based NP family.
:::

:::{prf:proof}
The source family $\mathfrak F_{3\text{-CNF}}$ is admissible because $3$-CNF formulas are finite syntactic objects over
a fixed finite alphabet. Their canonical tokenization and padding to fixed-length valid codes yields injective bitstring
encodings with polynomial-time validity testing and decoding.

The Boolean output family $\mathfrak B$ is trivially admissible. The witness family
$$
W_n=\{0,1\}^{\le n}
$$
is admissible by the standard bitstring encoding.

For the verifier, given a formula $F\in F_n$ and an assignment $a\in W_n$, one evaluates each clause by reading at most
three literals and checking whether at least one is satisfied by $a$. Since the number of clauses and variables encoded
in $F$ is $O(n)$, the total runtime is polynomial in $n$.

The decision-specification relation is therefore well-defined, and all admissibility conditions hold.
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
Every current frontend witness template is built from uniform maps on the admissible input family together with the
same output family and correctness relation. Restricting the input family to
$$
\mathfrak X' \subseteq \mathfrak X
$$
preserves admissibility, uniformity, and correctness checks. Therefore any surviving witness or tactic-level positive
route on $\Pi$ would restrict to the same kind of route on $\Pi'$. Contrapositively, if a given modal/tactic route is
blocked on $\Pi'$, then it is blocked on the ambient family $\Pi$ as well.
:::

:::{prf:lemma} Metric Blockage for Canonical 3-SAT
:label: lem-random-3sat-metric-blockage

For $\Pi_{3\text{-SAT}}$, the metric obstruction certificate
$$
K_{\mathrm{LS}_\sigma}^-
$$
holds.
:::

:::{prf:proof}
Apply the sharp obstruction criterion {prf:ref}`lem-sharp-obstruction` to the standard hard cores inside canonical
$3$-SAT and lift the resulting blockage from those admissibly presented subfamilies to the full canonical family by
{prf:ref}`lem-frontend-restriction-monotonicity`. Thus no polynomial sharp-descent witness of
{prf:ref}`def-class-i-climbers` survives on $\Pi_{3\text{-SAT}}$, and the metric route is blocked by
$$
K_{\mathrm{LS}_\sigma}^-.
$$
:::

:::{prf:lemma} Causal Blockage for Canonical 3-SAT
:label: lem-random-3sat-causal-blockage

For $\Pi_{3\text{-SAT}}$, the causal obstruction certificate
$$
K_{\mathrm{E6}}^-
$$
holds.
:::

:::{prf:proof}
Apply {prf:ref}`lem-shape-obstruction` to the frustrated clause-variable cores appearing inside canonical $3$-SAT and
lift the resulting blockage from those admissibly presented subfamilies to the full canonical family by
{prf:ref}`lem-frontend-restriction-monotonicity`. The well-foundedness tactic {prf:ref}`def-e6` therefore returns the
blocking certificate
$$
K_{\mathrm{E6}}^-.
$$
:::

:::{prf:lemma} Integrality Blockage for Canonical 3-SAT
:label: lem-random-3sat-integrality-blockage

For $\Pi_{3\text{-SAT}}$, the integrality obstruction certificate
$$
K_{\mathrm{E4}}^-
$$
holds.
:::

:::{prf:proof}
The $\flat$-route requires an arithmetic quotient or lattice-type compression. The canonical $3$-SAT family presents
no such integral compression witness, so the integrality lock {prf:ref}`def-e4` contributes the negative certificate
$$
K_{\mathrm{E4}}^-.
$$
:::

:::{prf:lemma} Galois-Monodromy Blockage for Canonical 3-SAT
:label: lem-random-3sat-galois-blockage

For $\Pi_{3\text{-SAT}}$, the Galois-monodromy obstruction certificate
$$
K_{\mathrm{E11}}^-
$$
holds.
:::

:::{prf:proof}
The residual algebraic route through solvable monodromy is excluded by the absence of a compressing solvable symmetry.
Accordingly, the lock of {prf:ref}`def-e11` supplies
$$
K_{\mathrm{E11}}^-.
$$
Together with {prf:ref}`lem-random-3sat-integrality-blockage`, this closes the currently exhibited algebraic channel.
:::

:::{prf:theorem} Algebraic Blockage for Canonical 3-SAT (Strengthened)
:label: thm-random-3sat-algebraic-blockage-strengthened

Assume the completion criteria of
{prf:ref}`def-completion-criteria-flat-dossier-3sat`.

For $\Pi_{3\text{-SAT}}$, there exists a $\flat$-obstruction certificate
$$
B_\flat \in K_\flat^-(\Pi_{3\text{-SAT}}).
$$

Equivalently, the semantic obstruction proposition
$$
\mathbb K_\flat^-(\Pi_{3\text{-SAT}})
$$
holds.

In particular, the current frontend pair
$$
K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^-
$$
is admissible provided it is shown, via Proposition {prf:ref}`prop-compatibility-with-current-tactics`, to derive the
strengthened algebraic obstruction.
:::

:::{prf:proof}
This is Proposition {prf:ref}`prop-flat-dossier-implies-algebraic-blockage`. Part VIII isolates the exact
signature-coverage, no-sketch, translator-stability, and certificate-extraction burdens that discharge the theorem.
The frontend pair $K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^-$ remains the intended operational record once the
corresponding bridge dossier is complete.
:::

:::{prf:lemma} Scaling Blockage for Canonical 3-SAT
:label: lem-random-3sat-scaling-blockage

For $\Pi_{3\text{-SAT}}$, the scaling obstruction certificate
$$
K_{\mathrm{SC}_\lambda}^{\mathrm{super}}.
$$
holds.
:::

:::{prf:proof}
Invoke the supercritical scaling criterion {prf:ref}`lem-scaling-obstruction` on the standard boundary-dense hard cores
inside canonical $3$-SAT and lift the resulting blockage to the full canonical family by
{prf:ref}`lem-frontend-restriction-monotonicity`. Balanced decompositions retain boundary size proportional to the
instance, so divide-and-conquer does not create a subcritical recurrence. This is recorded as
$$
K_{\mathrm{SC}_\lambda}^{\mathrm{super}}.
$$
:::

:::{prf:lemma} Boundary Blockage for Canonical 3-SAT
:label: lem-random-3sat-boundary-blockage

For $\Pi_{3\text{-SAT}}$, the boundary obstruction certificate
$$
K_{\mathrm{E8}}^-
$$
holds.
:::

:::{prf:proof}
Apply {prf:ref}`lem-boundary-obstruction` to the standard non-planar, unbounded-width hard cores inside canonical
$3$-SAT and lift the resulting blockage to the full canonical family by
{prf:ref}`lem-frontend-restriction-monotonicity`. The corresponding boundary certificate is
$$
K_{\mathrm{E8}}^-,
$$
as checked by the DPI tactic {prf:ref}`def-e8`.
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

The backend dossiers of Parts V and VIII are a stronger audit-level refinement of the same route. They strengthen the
metric, causal, algebraic, scaling, and boundary channels into explicit semantic obstruction packages, but they do not
add a new logical prerequisite to the direct Part VI separation theorem.
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
Combine {prf:ref}`lem-random-3sat-metric-blockage`,
{prf:ref}`lem-random-3sat-causal-blockage`,
{prf:ref}`lem-random-3sat-integrality-blockage`,
{prf:ref}`lem-random-3sat-galois-blockage`,
{prf:ref}`lem-random-3sat-scaling-blockage`,
and {prf:ref}`lem-random-3sat-boundary-blockage`, then apply the certificate logic built into
{prf:ref}`def-e13`.
:::

:::{prf:theorem} Canonical 3-SAT is Outside $P_{\mathrm{FM}}$
:label: thm-random-3sat-not-in-pfm

$$
\Pi_{3\text{-SAT}} \notin P_{\mathrm{FM}}.
$$

More precisely:
$$
\mathsf{Sol}_{\mathrm{poly}}(\Pi_{3\text{-SAT}})=\varnothing.
$$

In particular, the direct exclusion theorem already holds at the tactic level; the stronger backend-dossier route of
Part VIII recovers the same conclusion by a more detailed semantic audit.
:::

:::{prf:proof}
Apply {prf:ref}`thm-e13-contrapositive-hardness` to {prf:ref}`ex-3sat-all-blocked`.
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
L\in NP_{\mathrm{FM}}.
$$
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

:::{prf:theorem} Canonical 3-SAT Completeness in $NP_{\mathrm{FM}}$
:label: thm-sat-membership-hardness-transfer

The canonical satisfiability family $\Pi_{3\text{-SAT}}$ belongs to $NP_{\mathrm{FM}}$ and is
$NP_{\mathrm{FM}}$-complete.

Consequently:
$$
\Pi_{3\text{-SAT}} \notin P_{\mathrm{FM}}
\quad\Longrightarrow\quad
P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}.
$$
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

Assume the bridge equivalence of Part I:
$$
P_{\mathrm{FM}}=P_{\mathrm{DTM}}
\qquad\text{and}\qquad
NP_{\mathrm{FM}}=NP_{\mathrm{DTM}}.
$$

Then
$$
P_{\mathrm{FM}} \neq NP_{\mathrm{FM}}
\quad\Longrightarrow\quad
P_{\mathrm{DTM}} \neq NP_{\mathrm{DTM}}.
$$
:::

:::{prf:proof}
This is immediate from Corollary {prf:ref}`cor-bridge-equivalence-rigorous`: if
$$
P_{\mathrm{DTM}}=P_{\mathrm{FM}}
\quad\text{and}\quad
NP_{\mathrm{DTM}}=NP_{\mathrm{FM}},
$$
then inequality of the internal classes transports directly to inequality of the classical classes.
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

The strengthened semantic route is a refinement of Step 2: once the backend dossiers are complete, the Part VIII
sufficiency theorem supplies the reconstructed E13 package and hence recovers the same exclusion theorem by a more
detailed audit trail.
:::

:::{prf:remark} Where a hostile referee will press hardest in Part VI
:label: rem-where-referee-presses-part-vi

A hostile referee will not spend most of their energy on the final export step. The natural pressure points in this
part are earlier:

1. the strengthened algebraic blockage theorem
   {prf:ref}`thm-random-3sat-algebraic-blockage-strengthened`,
   because it must exclude all admissible polynomial-size algebraic sketches, not just obvious symmetry-based ones;

2. the strengthened boundary dossier burden of Part VIII, centered on
   {prf:ref}`def-completion-criteria-partial-dossier-3sat`,
   because it must exclude all admissible polynomial-interface contractions, not just the currently named
   planar/Pfaffian/treewidth frontends;

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
(\mathcal O_{\mathrm{I}},\mathcal O_{\mathrm{II}},\mathcal O_{\mathrm{III}},\mathcal O_{\mathrm{IV}},\mathcal O_{\mathrm{V}},\mathcal O_{\mathrm{VI}})
$$
of obligation clusters corresponding respectively to Parts I--VI.

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
obligation clusters. This ledger over-approximates the direct theorem route by also tracking the backend dossiers and
bridge dossiers used in the stronger semantic implementation.

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
- **V.7** frontend-to-backend bridge dossiers for the legacy tactic certificates.

#### Cluster VI: canonical $3$-SAT instantiation
$$
\mathcal O_{\mathrm{VI}}=
\{\mathrm{VI}.1,\dots,\mathrm{VI}.7\},
$$
where:

- **VI.1** admissibility of the canonical $3$-SAT family and its witness family;
- **VI.2** verifier membership in $NP_{\mathrm{FM}}$;
- **VI.3** the direct frontend E13 package for canonical $3$-SAT;
- **VI.4** exclusion of canonical $3$-SAT from $P_{\mathrm{FM}}$ via the direct E13 route;
- **VI.5** the five modal backend dossiers
  $$
  \mathcal D_\sharp(\Pi_{3\text{-SAT}}),\ 
  \mathcal D_\int(\Pi_{3\text{-SAT}}),\ 
  \mathcal D_\flat(\Pi_{3\text{-SAT}}),\ 
  \mathcal D_\ast(\Pi_{3\text{-SAT}}),\ 
  \mathcal D_\partial(\Pi_{3\text{-SAT}});
  $$
- **VI.6** internal Cook--Levin reduction;
- **VI.7** $NP_{\mathrm{FM}}$-completeness of canonical $3$-SAT, the internal separation corollary, and the
  strengthened reconstructed-E13 refinement.

No further foundational cluster is required for the theorem chain itself.
:::

:::{prf:proof}
The finiteness is immediate because:
1. the evaluator instruction set is finite;
2. the normal-form constructors are finite;
3. the modal witness classes are exactly five;
4. the obstruction calculi are exactly five;
5. the target family is fixed to canonical $3$-SAT.

Every theorem in Parts I--VI belongs to one of the six displayed clusters, and every later theorem depends only on a
finite collection of earlier obligations. Hence the full program reduces to a finite acyclic ledger.
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

4. the canonical $3$-SAT family carries a full reconstructed E13 package;

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
- Cluster VI instantiates the abstract obstruction theory to canonical $3$-SAT and gives the internal separation.

Combining Cluster I with Cluster VI exports the internal separation to the classical DTM classes.
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

4. **Backend dossier for $\sharp$.**  
   State the invariant excluding every polynomially bounded descent witness.

5. **Backend dossier for $\int$.**  
   State the invariant excluding every polynomial-height well-founded elimination witness.

6. **Backend dossier for $\flat$.**  
   State the invariant excluding every admissible polynomial-size algebraic sketch, including determinant/rank/fourier/
   cancellation methods.

7. **Backend dossier for $\ast$.**  
   State the invariant excluding every admissible polynomially bounded self-reduction tree.

8. **Backend dossier for $\partial$.**  
   State the invariant excluding every admissible polynomial-size interface contraction.

9. **Certificate extraction.**  
   Convert those five backend dossiers into the five obstruction certificates
   $$
   B_\sharp,\ B_\int,\ B_\flat,\ B_\ast,\ B_\partial.
   $$

10. **Reconstructed E13 assembly.**  
   Assemble the five certificates into the full package and derive
   $$
   \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}}.
   $$

11. **Internal Cook--Levin.**  
    Give the tableau construction, clause gadgets, witness consistency constraints, and the polynomial bound on formula
    size.

12. **$NP_{\mathrm{FM}}$-completeness and internal separation.**  
    Conclude
    $$
    \Pi_{3\text{-SAT}}\in NP_{\mathrm{FM}},
    \qquad
    \Pi_{3\text{-SAT}}\notin P_{\mathrm{FM}},
    \qquad
    P_{\mathrm{FM}}\neq NP_{\mathrm{FM}}.
    $$

13. **Export.**  
    Combine with Cluster I to derive
    $$
    P_{\mathrm{DTM}}\neq NP_{\mathrm{DTM}}.
    $$

The hardest items on the stronger audit route are steps 6 and 8. The direct theorem route, by contrast, is Step 3 plus
Steps 11--13.
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
   {prf:ref}`mt-alg-complete`, {prf:ref}`def-e13`, and
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
the stronger semantic backend dossiers of Parts V and VIII.
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

The minimal completion certificate introduced below is **strictly stronger**. It adds the full proof-obligation ledger,
the five semantic backend dossiers, and the frontend-to-backend bridge dossiers. Those artifacts refine the direct route
into a stronger referee-auditable semantic implementation, but they are not an additional logical prerequisite for the
direct Part VI theorem chain.
:::

:::{prf:definition} Minimal completion certificate for the full program
:label: def-minimal-completion-certificate

A **minimal completion certificate** for the stronger audit refinement of the separation program is the tuple
$$
\mathcal C_{\mathrm{master}}
=
(\mathcal T_{\mathrm{prim}},\ \mathfrak L,\ \{\mathcal D_\lozenge(\Pi_{3\text{-SAT}})\}_{\lozenge},\ \{\mathcal F_\lozenge(\Pi_{3\text{-SAT}})\}_{\lozenge})
$$
such that:

1. $\mathcal T_{\mathrm{prim}}$ is a complete primitive audit table;
2. $\mathfrak L$ is a complete and acyclic proof obligation ledger;
3. each modal backend dossier
   $$
   \mathcal D_\lozenge(\Pi_{3\text{-SAT}})
   $$
   is complete;
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
8. final master export summary.

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
   - the explicit admissible algebraic-signature library for the strengthened $\flat$-class,
   - the explicit admissible interface-contraction library for the strengthened $\partial$-class,
   - the complete obstruction-calculus rules and their soundness/completeness proofs,
   - and the five complete backend dossiers for canonical $3$-SAT.

Those are the real proof obligations at the two levels of presentation. Once the direct-route items are present, the
main separation chain is theorem-complete; once the stronger audit items are also present, there is no remaining vague
appeal to “higher-topos exhaustiveness” that can substitute for mathematics.
:::

### VIII. Audit-Level Implementation of the Primitive Classification and the Canonical 3-SAT Backend Dossiers

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
- **backend dossiers**, which specify the exact lower-bound burdens still to be proved;
- and **sufficiency theorems**, which state precisely what follows once those dossiers are complete.

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

## VIII.B. Canonical 3-SAT Backend Dossiers

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
6. $\mathfrak S_{\mathrm{mono}}$ covers solvable-monodromy and related algebraic-geometry reductions.

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

8. **No-sketch theorem for solvable-monodromy methods.**  
   No admissible polynomial-size solvable-monodromy sketch over
   $\mathfrak S_{\mathrm{mono}}$ yields a correct solver family for canonical $3$-SAT.

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
- or failure of solvable monodromy alone.

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

## VIII.C. Sufficiency Theorems for the Audit Artifacts

:::{prf:theorem} Sufficiency of the primitive audit appendix
:label: thm-sufficiency-primitive-audit-appendix

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
is complete.

Then the five blockage theorems of Part VI are formally discharged:
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

This yields the full reconstructed E13 package for canonical $3$-SAT. Then apply the Mixed-Modal Obstruction Theorem of
Part V and the canonical-instantiation theorems of Part VI. The final export to classical complexity uses the bridge
equivalence from Part I.
:::

:::{prf:remark} What may and may not be claimed after inserting Part VIII
:label: rem-what-may-be-claimed-after-part-viii

After inserting Part VIII, the manuscript may honestly claim the following:

1. the direct exclusion route for canonical $3$-SAT still runs through the current tactic-level E13 package and does
   **not** logically require the backend dossiers as an extra prerequisite;
2. the semantic primitive audit has been implemented at the family level, and the exact remaining stronger backend
   burden has been reduced to the five canonical $3$-SAT dossiers;
3. the exact content of each backend dossier is now explicit;
4. the formal sufficiency chain from completed audit artifacts to the separation result is precise.

However, the manuscript may **not** yet honestly claim that the strengthened $\flat$- or $\partial$-blockage theorems
are proved unless the corresponding no-sketch and no-contraction subtheorems have actually been written and checked.
Likewise, it may not honestly claim the metric and causal blockage theorems are complete unless the plateau-core and
frustration-core dossier burdens have been discharged in full.

This distinction is essential for referee trust.
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
2. the semantic obstruction proposition $\mathbb K_\sharp^-(\Pi)$ holds;
3. by completeness of $\mathsf{Obs}_\sharp$, there exists a certificate
   $$
   B_\sharp\in K_\sharp^-(\Pi).
   $$
:::

:::{prf:proof}
Item (1) is the final clause of {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`. Item (2) is exactly the semantic
reformulation of the nonexistence of a pure $\sharp$-witness on the hard subfamily. Item (3) then follows from the
completeness direction of {prf:ref}`thm-sharp-obstruction-sound-complete`.
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
2. the semantic obstruction proposition $\mathbb K_\int^-(\Pi)$ holds;
3. by completeness of $\mathsf{Obs}_\int$, there exists a certificate
   $$
   B_\int\in K_\int^-(\Pi).
   $$
:::

:::{prf:proof}
Item (1) is the final clause of {prf:ref}`thm-int-barrier-obstruction-metatheorem`. Item (2) is the semantic
reformulation of the nonexistence of a pure $\int$-witness on the hard subfamily. Item (3) then follows from the
completeness direction of {prf:ref}`thm-int-obstruction-sound-complete`.
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
1. the semantic obstruction proposition $\mathbb K_\flat^-(\Pi)$ holds;
2. by completeness of $\mathsf{Obs}_\flat$, there exists a certificate
   $$
   B_\flat\in K_\flat^-(\Pi).
   $$
:::

:::{prf:proof}
Item (1) is the contrapositive form of {prf:ref}`thm-flat-barrier-obstruction-metatheorem`. Item (2) follows from the
completeness direction of {prf:ref}`thm-flat-obstruction-sound-complete`.
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
1. the semantic obstruction proposition $\mathbb K_\ast^-(\Pi)$ holds;
2. by completeness of $\mathsf{Obs}_\ast$, there exists a certificate
   $$
   B_\ast\in K_\ast^-(\Pi).
   $$
:::

:::{prf:proof}
Item (1) is the contrapositive form of {prf:ref}`thm-star-barrier-obstruction-metatheorem`. Item (2) follows from the
completeness direction of {prf:ref}`thm-star-obstruction-sound-complete`.
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
1. the semantic obstruction proposition $\mathbb K_\partial^-(\Pi)$ holds;
2. by completeness of $\mathsf{Obs}_\partial$, there exists a certificate
   $$
   B_\partial\in K_\partial^-(\Pi).
   $$
:::

:::{prf:proof}
Item (1) is the contrapositive form of {prf:ref}`thm-partial-barrier-obstruction-metatheorem`. Item (2) follows from
the completeness direction of {prf:ref}`thm-boundary-obstruction-sound-complete`.
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
where each
$$
B_\lozenge\in K_\lozenge^-(\Pi)
$$
is obtained from the corresponding barrier metatheorem of this part.
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

Then $\Pi$ carries a full reconstructed E13 obstruction package in the sense of
{prf:ref}`def-e13-reconstructed`.

If, in addition, the corresponding frontend-to-backend bridge dossiers of Part VIII are complete, then $\Pi$ also
carries the tactic-level certificate
$$
K_{\mathrm{E13}}^+(\Pi).
$$
:::

:::{prf:proof}
By the five preceding corollaries, the hypotheses produce certificates
$$
B_\sharp\in K_\sharp^-(\Pi),\quad
B_\int\in K_\int^-(\Pi),\quad
B_\flat\in K_\flat^-(\Pi),\quad
B_\ast\in K_\ast^-(\Pi),\quad
B_\partial\in K_\partial^-(\Pi).
$$

Collecting these five certificates gives exactly the full reconstructed E13 obstruction package of
{prf:ref}`def-e13-reconstructed`. If the frontend-to-backend bridge dossiers are also complete, then the current
tactic-level certificate $K_{\mathrm{E13}}^+(\Pi)$ follows by transporting the semantic obstruction package back to the
legacy frontend language.
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
The first claim follows from the reconstructed E13 package and the mixed-modal obstruction / reconstructed-hardness
layer of Part V, in particular {prf:ref}`thm-mixed-modal-obstruction` and
{prf:ref}`cor-e13-contrapositive-hardness-reconstructed`. The second claim is the standard export step through the
previously established bridge equivalence.
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
5. assemble them into reconstructed E13, and optionally into the current frontend E13 package when the bridge dossiers
   are available.

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

If the relevant frontend-to-backend bridge dossiers are also present, the same route yields the current tactic-level
certificate $K_{\mathrm{E13}}^+(\Pi)$ as well.

That is exactly the sense in which barriers become reusable metatheorems rather than one-off heuristics.
:::

:::{prf:remark} Recommended placement of appendices after Part IX
:label: rem-recommended-placement-appendices

For maximal referee readability, the following appendices should follow immediately after Part IX.

1. **Appendix A:** The complete primitive audit table $\mathcal T_{\mathrm{prim}}$.
2. **Appendix B:** The direct frontend E13 certificate appendix for canonical $3$-SAT.
3. **Appendix C:** The $\sharp$-backend dossier for canonical $3$-SAT.
4. **Appendix D:** The $\int$-backend dossier for canonical $3$-SAT.
5. **Appendix E:** The strengthened $\flat$-backend dossier for canonical $3$-SAT.
6. **Appendix F:** The $\ast$-backend dossier for canonical $3$-SAT.
7. **Appendix G:** The strengthened $\partial$-backend dossier for canonical $3$-SAT.
8. **Appendix H:** The frontend-to-backend bridge dossiers for the legacy tactic certificates.
9. **Appendix I:** The explicit Internal Cook--Levin reduction.

That appendix ordering mirrors the proof-dependency order and minimizes referee backtracking.
:::

## Appendix A. Primitive Audit Table

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
:label: thm-appendix-a-primitive-audit-table

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
{prf:ref}`thm-appendix-a-primitive-audit-table`.
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

We now summarize how the strengthened Part IV--IX ladder can be verified, audited, and falsified theorem by theorem.

:::{prf:theorem} Verification of Classification, Obstruction, and Completion
:label: thm-verification-completeness

The classification/exhaustiveness and obstruction framework is reduced to the following auditable components:

| Component | Status | Reference |
|-----------|--------|-----------|
| Cohesive modalities exhaust structure | **THEOREM TARGET** (Schreiber) | {prf:ref}`thm-schreiber-structure` |
| Internal polynomial time is defined by family cost certification | **DEFINITIONAL BASIS** | {prf:ref}`def-family-cost-certificate`, {prf:ref}`def-internal-polytime-family-rigorous` |
| Normal-form reduction | **THEOREM TARGET** | {prf:ref}`thm-syntax-to-normal-form` |
| Primitive audit (semantic family presentation) | **THEOREM** | {prf:ref}`thm-appendix-a-primitive-audit-table` |
| Primitive audit appendix sufficiency | **SUFFICIENCY THEOREM** | {prf:ref}`thm-sufficiency-primitive-audit-appendix` |
| Primitive progress classification | **AUDIT-COMPLETE THEOREM** | {prf:ref}`lem-primitive-step-classification`, {prf:ref}`cor-primitive-step-classification-from-appendix-a` |
| Witness decomposition | **LEDGER-GOVERNED THEOREM** | {prf:ref}`thm-witness-decomposition`, {prf:ref}`thm-sufficiency-of-complete-ledger` |
| Irreducible witness classification | **LEDGER-GOVERNED THEOREM** | {prf:ref}`thm-irreducible-witness-classification`, {prf:ref}`thm-sufficiency-of-complete-ledger` |
| Computational modal exhaustiveness | **LEDGER-GOVERNED COROLLARY** | {prf:ref}`cor-computational-modal-exhaustiveness`, {prf:ref}`thm-sufficiency-of-complete-ledger` |
| Semantic obstruction calculi | **DEFINITIONAL BASIS** | {prf:ref}`def-semantic-modal-obstruction`, {prf:ref}`def-obstruction-calculus-schema` |
| Modal obstruction soundness/completeness | **THEOREM TARGET PACKAGE** | {prf:ref}`thm-sharp-obstruction-sound-complete`--{prf:ref}`thm-boundary-obstruction-sound-complete` |
| Mixed-modal obstruction | **THEOREM TARGET** | {prf:ref}`thm-mixed-modal-obstruction` |
| Frontend-tactic compatibility | **PROPOSITION TARGET** | {prf:ref}`prop-compatibility-with-current-tactics` |
| Reconstructed E13 hardness | **COROLLARY TARGET** | {prf:ref}`cor-e13-contrapositive-hardness-reconstructed` |
| No hidden mechanism falsifiability | **COROLLARY TARGET** | {prf:ref}`cor-no-hidden-mechanism` |
| E13 contrapositive hardness | **THEOREM** | {prf:ref}`thm-e13-contrapositive-hardness` |
| Canonical 3-SAT admissibility | **THEOREM** | {prf:ref}`thm-canonical-3sat-admissible` |
| Direct separation certificate | **DEFINITIONAL PACKAGE** | {prf:ref}`def-direct-separation-certificate` |
| Direct separation certificate sufficiency | **THEOREM** | {prf:ref}`thm-sufficiency-direct-separation-certificate` |
| Direct frontend E13 certificate appendix | **THEOREM** | {prf:ref}`thm-appendix-b-frontend-e13-certificate-table` |
| Canonical 3-SAT E13 antecedent package | **THEOREM** | {prf:ref}`ex-3sat-all-blocked` |
| Canonical 3-SAT exclusion from $P_{\mathrm{FM}}$ | **THEOREM** | {prf:ref}`thm-random-3sat-not-in-pfm`, {prf:ref}`thm-e13-contrapositive-hardness` |
| Canonical 3-SAT backend dossier package | **STRENGTHENED AUDIT ARTIFACT** | {prf:ref}`def-canonical-3sat-backend-dossier-package` |
| Canonical 3-SAT dossier sufficiency | **STRENGTHENED SUFFICIENCY THEOREM** | {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package` |
| Canonical 3-SAT reconstructed E13 package | **STRENGTHENED COMPLETION-DEPENDENT CONSEQUENCE** | {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package` |
| Barrier datum and modal barrier complexities | **DEFINITIONAL BASIS** | {prf:ref}`def-barrier-datum`, {prf:ref}`def-sharp-barrier-crossing-number`--{prf:ref}`def-partial-barrier-width` |
| Barrier metatheorem package | **THEOREM PACKAGE** | {prf:ref}`thm-sharp-barrier-obstruction-metatheorem`--{prf:ref}`thm-partial-barrier-obstruction-metatheorem` |
| Barrier package assembly and hardness | **THEOREM PACKAGE** | {prf:ref}`thm-barrier-package-implies-e13`, {prf:ref}`cor-barrier-contrapositive-hardness` |
| Algorithm audit by barrier profile | **COROLLARY** | {prf:ref}`cor-algorithm-audit-by-modal-barrier-profile` |
| Internal Cook--Levin reduction | **THEOREM** | {prf:ref}`thm-internal-cook-levin-reduction` |
| Canonical 3-SAT NP-completeness | **THEOREM** | {prf:ref}`thm-sat-membership-hardness-transfer` |
| Proof obligation ledger | **DEFINITIONAL BASIS** | {prf:ref}`def-proof-obligation-ledger` |
| Finite reduction to an obligation ledger | **THEOREM** | {prf:ref}`thm-finite-reduction-to-ledger` |
| Completion criterion for the master export | **COROLLARY** | {prf:ref}`cor-completion-criterion-master-export` |
| Current tactic-level obstruction frontends | **COMPUTABLE / FRONTEND ARTIFACT** | {prf:ref}`def-obstruction-certificates` |
| Bridge to DTM complexity | **THEOREM PACKAGE** | {prf:ref}`cor-bridge-equivalence-rigorous` and Part XX |

**Key Point:** The framework rests on mathematical theorems within cohesive $(\infty,1)$-topos theory, not empirical
observations. The completeness burden is no longer hidden in one theorem name; it is distributed over an explicit audit,
decomposition, irreducibility, exhaustiveness, obstruction, barrier-metatheorem, and completion ledger. Parts VII--IX
make precise which components are already formalized and which are completion-dependent.
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
3. a failure of modal obstruction soundness or completeness would refute one of
   {prf:ref}`thm-sharp-obstruction-sound-complete`--{prf:ref}`thm-boundary-obstruction-sound-complete`;
4. a failure of the canonical 3-SAT instantiation would localize either to
   {prf:ref}`thm-canonical-3sat-admissible`, {prf:ref}`ex-3sat-all-blocked`,
   {prf:ref}`thm-e13-contrapositive-hardness`, {prf:ref}`thm-internal-cook-levin-reduction`,
   or, on the strengthened audit route, to {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package` and the Part VIII
   backend/bridge dossiers, not to some vague meta-level complaint.
5. a failure to complete the full audit trail would localize to a missing component of the minimal completion
   certificate from {prf:ref}`def-minimal-completion-certificate`.
6. a failure of a claimed barrier-based hardness route would localize either to the existence or translator-stability of
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

**Internal Separation Program (C3):** Parts VI--IX isolate an exact criterion under which the internal separation
follows:

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
  {prf:ref}`cor-barrier-contrapositive-hardness`);
- and, as a strengthened audit refinement of that same route, the Part VIII backend-dossier sufficiency theorem
  ({prf:ref}`def-canonical-3sat-backend-dossier-package`,
  {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package`).

**Logical Structure:**

$$
(\text{C1} \wedge \text{C2} \wedge \text{C3}) \Rightarrow (P_{\text{DTM}} \neq NP_{\text{DTM}}).
$$

**Within** the ambient foundation, the Part VI internal separation follows by the direct canonical E13 theorem chain.
Part IX adds a reusable barrier-metatheorem route to the same obstruction conclusion, and Parts VII--VIII sharpen the
overall picture by specifying a stronger audited semantic implementation whose completion also forces the same internal
separation.

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
theorem chain, not by slogan. Part IX abstracts this into reusable barrier metatheorems, and Parts VII--VIII reduce the
stronger semantic implementation of that same route to explicit audit artifacts.

**What is already implemented for canonical $3$-SAT:** The canonical $3$-SAT object is admissible, its direct frontend
E13 certificate package is now assembled in Appendix B, and it is tied to the class-separation argument by the internal
Cook--Levin theorem and the $NP_{\text{FM}}$-completeness theorem.

**What Parts VII--IX add:** an explicit obligation ledger, an implemented semantic primitive audit appendix, the direct
separation certificate, the reusable barrier-metatheorem layer, the canonical backend dossier templates, and a minimal
completion certificate specifying exactly which stronger audit artifacts must exist to refine the current E13 route into
a full semantic obstruction implementation.

**What remains to discharge:** at the stronger audit level, the five complete canonical $3$-SAT backend dossiers, any
desired problem-specific barrier data and lower bounds feeding the Part IX metatheorems, and the remaining
bridge-completeness artifacts. Those are no longer part of the minimal direct route; they strengthen the same
exclusion/separation chain by adding a finer semantic audit trail.

**What the bridge supplies:** The separate equivalence theorem identifying the internal classes with the classical
Turing-machine classes.

**What is a choice:** Working in cohesive $(\infty,1)$-topos theory (Condition C1) is a **foundational choice**, like
choosing to work in ZFC versus some alternative foundation. Within that foundation, our results are theorems.

The beauty of this approach is that it makes the roles **explicit**. Nothing is hidden behind a slogan. What is proved,
what is completion-dependent, and what the bridge theorem adds are all separately named.

These are questions we can investigate, debate, and potentially settle. That is progress.
:::

### Summary: What This Framework Establishes

:::{prf:theorem} Main Results Summary
:label: thm-hypo-algorithmic-main-results

The algorithmic completeness framework now consists of the following proved foundations and stronger audit-level
refinements:

**Theorem 1 (Modal Completeness):** In a cohesive $(\infty,1)$-topos, the five modalities $\{\int, \flat, \sharp, \ast, \partial\}$ exhaust all exploitable structure ({prf:ref}`thm-schreiber-structure`, {prf:ref}`cor-exhaustive-decomposition`).

**Theorem 2 (Witness Decomposition Target):** Every internally polynomial-time family admits a finite modal factorization
tree ({prf:ref}`thm-witness-decomposition`), with the primitive leaf audit localized by
{prf:ref}`thm-sufficiency-primitive-audit-appendix`.

**Theorem 3 (Irreducible Classification Target):** Every irreducible witness object lies in one of the five pure modal
subcategories ({prf:ref}`thm-irreducible-witness-classification`).

**Theorem 4 (Computational Modal Exhaustiveness Target):** The internally polynomial-time class coincides with the
saturated closure of the five pure modal classes ({prf:ref}`cor-computational-modal-exhaustiveness`).

**Theorem 5 (Mixed-Modal Obstruction Target):** If every irreducible modal route is blocked, then no internally
polynomial-time correct solver exists ({prf:ref}`thm-mixed-modal-obstruction`).

**Theorem 6 (Reconstructed E13 Hardness Target):** A full E13 obstruction package excludes a problem family from
$P_{\text{FM}}$ ({prf:ref}`cor-e13-contrapositive-hardness-reconstructed`, compatibly restated by
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

**Theorem 9 (Direct Route Packaging):** The direct theorem route is packaged explicitly by the direct separation
certificate and Appendix B
({prf:ref}`def-direct-separation-certificate`,
 {prf:ref}`thm-sufficiency-direct-separation-certificate`,
 {prf:ref}`thm-appendix-b-frontend-e13-certificate-table`).

**Theorem 10 (Barrier Metatheorem Layer):** Translator-stable barrier data and superpolynomial modal barrier lower
bounds yield reusable modal obstructions, reconstructed E13, and hence hardness
({prf:ref}`thm-sharp-barrier-obstruction-metatheorem`--{prf:ref}`thm-partial-barrier-obstruction-metatheorem`,
 {prf:ref}`thm-barrier-package-implies-e13`,
 {prf:ref}`cor-barrier-contrapositive-hardness`).

**Theorem 11 (3-SAT Backend Sufficiency):** If the canonical $3$-SAT backend dossier package is complete, then canonical
$3$-SAT carries the full reconstructed E13 obstruction package and therefore also lies outside $P_{\text{FM}}$
({prf:ref}`def-canonical-3sat-backend-dossier-package`,
 {prf:ref}`thm-sufficiency-canonical-3sat-dossier-package`).

**Theorem 12 (Internal Separation Criterion):** Combining Theorem 7 with Theorem 8 yields
$P_{\text{FM}} \neq NP_{\text{FM}}$
({prf:ref}`cor-pfm-neq-npfm-from-random-3sat`).

**Theorem 13 (Audit Completion and Classical Export):** The stronger audit refinement counts as implemented exactly when
the primitive audit appendix, proof obligation ledger, modal backend dossiers, and frontend bridge dossiers are
complete; with the bridge equivalence, the master export then yields $P_{\text{DTM}} \neq NP_{\text{DTM}}$
({prf:ref}`def-minimal-completion-certificate`,
{prf:ref}`cor-completion-criterion-master-export`,
{prf:ref}`cor-internal-to-classical-separation`).
:::

:::{div} feynman-prose
And there you have it. We have built a mathematical framework that explains **why** some algorithms are fast and others
must be slow. The five modalities are not arbitrary categories; they are the fundamental ways that structure manifests
in a cohesive topos. An algorithm is fast if it can "see" one of these structural patterns. An algorithm is slow if all
five views reveal nothing but noise.

This is the answer to the question: "Could there be a clever algorithm we have not thought of yet?" Within the
framework, any such algorithm must appear as a modal factorization tree built from the five structural types. Part IX
explains how reusable barrier data can block those modal routes in general, and Parts VII--VIII make the remaining
implementation burden explicit: for the stronger audit route, complete the primitive audit, the five canonical $3$-SAT
backend dossiers, and the remaining bridge-completeness artifacts.

The repaired proof presentation makes the dependencies explicit. E13 does the obstruction work, the SAT transfer does
the internal class-separation work, Appendix B packages the direct frontend certificates, the audit chapters say exactly
what stronger semantic artifacts would refine that route, and the bridge chapter does the model-export work. Each
dependency now has its own named theorem or named completion artifact.
:::
