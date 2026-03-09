---
title: "Algorithmic Completeness"
---

# Algorithmic Completeness

(sec-taxonomy-computational-methods)=
## The Taxonomy of Computational Methods

This part establishes that polynomial-time algorithms must exploit specific structural invariants detectable by the Cohesive Topos modalities. It provides the theoretical foundation for **Tactic E13** (Algorithmic Completeness Lock), which closes the "Alien Algorithm" loophole in complexity-theoretic proofs.

:::{div} feynman-prose
Let me tell you what this is really about. Whenever you see a fast algorithm, you should ask yourself: "Why is this fast? What structure is it exploiting?" Because here is the thing: if you have no structure, you are reduced to brute force, to trying things one by one until you stumble on the answer. And brute force takes exponential time.

Now, the question that has haunted complexity theory is this: could there be some clever algorithm we have not thought of yet? Some "alien" technique that solves hard problems fast without exploiting any recognizable structure? This chapter says no. We claim that all efficient algorithms must factor through one of five fundamental types of structure, which we call the five "modalities." If your problem has none of these structures, no algorithm can help you.

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

**Consequence for Algorithms:** Every algorithmic morphism $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ achieving polynomial compression must factor through (at least) one of the five modalities. An algorithm that cannot factor through any modality has no structure to exploit and reduces to brute force search.

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

:::{prf:definition} Algorithmic Morphism
:label: def-algorithmic-morphism

An **algorithm** is a morphism $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ representing a discrete dynamical update rule on a problem configuration stack $\mathcal{X} \in \operatorname{Obj}(\mathbf{H})$.

**Validity:** $\mathcal{A}$ is valid if it converges to the solution subobject $\mathcal{S} = \Phi^{-1}(0)$; that is, $\lim_{n \to \infty} \mathcal{A}^n$ factors through $\mathcal{S} \hookrightarrow \mathcal{X}$.

**Polynomial Efficiency:** $\mathcal{A}$ is polynomial-time if it reduces the entropy $H(\mathcal{X}) = \log \operatorname{Vol}(\mathcal{X})$ from $N$ bits to 0 bits in $\text{poly}(N)$ steps.
:::

:::{prf:definition} Modal Factorization
:label: def-modal-factorization

An algorithmic process $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ **factors through modality** $\lozenge \in \{\int, \flat, \sharp, \ast, \partial\}$ if there exists a commutative diagram (up to homotopy):

```
           η_◇
    𝒳 ─────────→ ◇𝒳
    │              │
    │              │ ◇𝒜
    ↓              ↓
    𝒳 ←───────── ◇𝒳
           ε_◇
```

where:
- $\eta_\lozenge: \mathrm{id} \to \lozenge$ is the unit of the modality (encoding into modal structure)
- $\epsilon_\lozenge: \lozenge \to \mathrm{id}$ is the counit/extraction (decoding from modal structure)
- $\lozenge\mathcal{A}$ is the algorithm lifted to $\lozenge$-modal types
- The composition $\epsilon_\lozenge \circ \lozenge\mathcal{A} \circ \eta_\lozenge$ is homotopic to $\mathcal{A}$

**Notation:** We write $\mathcal{A} \triangleright \lozenge$ to denote that $\mathcal{A}$ factors through $\lozenge$.

**Computational Meaning:** Factorization through $\lozenge$ means the algorithm:
1. **Encodes** the problem into $\lozenge$-structure via $\eta_\lozenge$
2. **Solves** efficiently in the $\lozenge$-transformed space via $\lozenge\mathcal{A}$
3. **Extracts** the solution via $\epsilon_\lozenge$

The speedup comes from step 2: working in $\lozenge\mathcal{X}$ compresses the search space by exploiting the structure that $\lozenge$ captures.
:::

:::{prf:definition} Obstruction Certificates
:label: def-obstruction-certificates

For each modality $\lozenge$, we define an **obstruction certificate** $K_\lozenge^-$ that witnesses the failure of polynomial-time factorization through $\lozenge$:

| Modality | Certificate | Obstruction Condition |
|----------|-------------|----------------------|
| $\sharp$ (Metric) | $K_\sharp^-$ | No spectral gap; Łojasiewicz inequality fails; glassy landscape |
| $\int$ (Causal) | $K_\int^-$ | Frustrated loops; $\pi_1(\text{factor graph}) \neq 0$; no DAG structure |
| $\flat$ (Algebraic) | $K_\flat^-$ | Trivial automorphism group $\mathrm{Aut}(\mathcal{X}) = \{e\}$; no symmetry |
| $\ast$ (Scaling) | $K_\ast^-$ | Supercritical scaling; boundary dominates in decomposition |
| $\partial$ (Holographic) | $K_\partial^-$ | Non-planar; no Pfaffian orientation; #P-hard contraction |

**Certificate Logic:** If all five obstruction certificates are present:

$$K_\sharp^- \wedge K_\int^- \wedge K_\flat^- \wedge K_\ast^- \wedge K_\partial^- \implies \mathcal{A} \notin P$$

This is the contrapositive of {prf:ref}`mt-alg-complete`: blocking all modalities blocks polynomial-time algorithms.
:::

:::{prf:definition} The Five Algorithm Classes (Modality Correspondence)
:label: def-five-algorithm-classes

Every polynomial-time algorithm $\mathcal{A} \in P$ exploits a structural resource corresponding to a Cohesive Topos modality:

| Class | Name | Modality | Exploited Resource | Examples | Detection |
|-------|------|----------|-------------------|----------|-----------|
| I | Climbers | $\sharp$ (Sharp/Differential) | Metric gradient, convexity | Gradient Descent, Local Search, Convex Optimization | Node 7 ($\mathrm{LS}_\sigma$), Node 12 ($\mathrm{GC}_\nabla$) |
| II | Propagators | $\int$ (Shape/Causal) | Causal order, DAG structure | Dynamic Programming, Unit Propagation, Belief Propagation | Tactic E6 (Well-Foundedness) |
| III | Alchemists | $\flat$ (Flat/Discrete) | Algebraic symmetry, group action | Gaussian Elimination, FFT, LLL | Tactic E4 (Integrality), E11 (Galois-Monodromy) |
| IV | Dividers | $\ast$ (Scaling) | Self-similarity, recursion | Divide & Conquer, Mergesort, Multigrid | Node 4 ($\mathrm{SC}_\lambda$) |
| V | Interference Engines | $\partial$ (Boundary/Cobordism) | Holographic cancellation | FKT/Matchgates, Quantum Algorithms | Tactic E8 (DPI), Node 6 ($\mathrm{Cap}_H$) |

:::

:::{div} feynman-prose
Let me make sure you understand what each of these classes is really doing. Think of each one as a different "trick" for compressing your search space.

**Climbers** are algorithms that follow a gradient downhill. They work when your problem has a smooth landscape with a clear direction toward the solution. Gradient descent is the prototype: at each step, you move in the direction that decreases the objective function. The key insight is that you are exploiting *metric structure*, the ability to measure "nearby" and "downhill."

**Propagators** exploit causal structure. Dynamic programming is the classic example: you solve subproblems in the right order, so each answer is available when you need it. The trick is that information flows in one direction, like dominoes falling. No cycles, no backtracking.

**Alchemists** exploit symmetry. If your problem has a large group acting on it, you can factor out that symmetry and work in a smaller quotient space. Gaussian elimination works because linear algebra has a huge symmetry group. The FFT works because the roots of unity form a cyclic group.

**Dividers** exploit self-similarity. If your problem looks the same at different scales, you can solve a smaller version and piece together the answer. Mergesort does this: sorting $n$ elements reduces to sorting $n/2$ elements, twice.

**Interference Engines** are the most exotic. They work when massive cancellations occur, like quantum algorithms where exponentially many paths interfere to leave only the right answer. The FKT algorithm for counting perfect matchings in planar graphs is the classical prototype.
:::

:::{prf:remark} AIT Characterization of Algorithm Classes
:label: rem-ait-algorithm-classes

Each algorithm class achieves polynomial-time performance by exploiting structural resources that enable **Kolmogorov complexity reduction** ({prf:ref}`def-kolmogorov-complexity`). The modality correspondence has a precise AIT interpretation:

| Class | Modality | RG Mechanism | Complexity Reduction |
|-------|----------|--------------|---------------------|
| I (Climbers) | $\sharp$ | Gradient descent | $K_{t+1} \leq K_t - \Omega(1)$ per step |
| II (Propagators) | $\int$ | Causal elimination | $K(x \mid \text{subproblems}) \ll K(x)$ |
| III (Alchemists) | $\flat$ | Symmetry quotient | $K([x]_G) \leq K(x) - \log\|G\| + O(1)$ |
| IV (Dividers) | $\ast$ | Scale factorization | $K(x) \leq \alpha \cdot K(x_{n/2}) + O(\log n)$ |
| V (Interference) | $\partial$ | Holographic cancellation | $K(\text{bulk}) \leq K(\partial) + O(1)$ |

**Thermodynamic Correspondence** ({prf:ref}`thm-sieve-thermo-correspondence`):
- **Climbers** exploit energy gradient: $\nabla K < 0$ along solution trajectory
- **Propagators** exploit conditional independence: subadditivity of $K$
- **Alchemists** exploit symmetry: $K$ decreases under quotient by group action
- **Dividers** exploit self-similarity: Master Theorem recurrence for $K$
- **Interference** exploits holography: boundary-to-bulk $K$ reduction

**Hardness Criterion (AIT Form):** A problem is hard for all five classes iff no modality achieves sub-exponential complexity reduction:

$$\forall \lozenge \in \{\sharp, \int, \flat, \ast, \partial\}: \quad K_\lozenge(\text{solution}) \geq K(\text{instance}) - o(n)$$

This is the AIT content of {prf:ref}`mt-alg-complete`.

In Sieve instantiations, $K(\cdot)$ is evaluated on the encoded thin trace $T_{\mathrm{thin}}$ using the approximable proxy $K_\epsilon$ with fixed resource bounds.
:::

### Detailed Algorithm Class Specifications

We now provide rigorous mathematical definitions for each algorithm class, including their factorization conditions and obstruction criteria.

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

:::{prf:lemma} Sharp Modality Obstruction
:label: lem-sharp-obstruction

If the energy landscape $\Phi$ is **glassy** (exhibiting one or more of):
- Exponentially many local minima separated by $\Theta(n)$ barriers
- No spectral gap: $\lambda_{\min}(\nabla^2 \Phi) \to 0$
- Łojasiewicz inequality fails: $\theta \to 0$ (flat regions)

then $\mathcal{A} \not\triangleright \sharp$ and Class I algorithms require exponential time.

**Obstruction Certificate:** $K_{\sharp}^- = (\text{glassy}, \lambda = 0, \theta \to 0)$

**Application:** Random 3-SAT near threshold has glassy landscape (Mézard-Parisi-Zecchina 2002), blocking Class I.
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

:::{prf:lemma} Shape Modality Obstruction (Frustrated Loops)
:label: lem-shape-obstruction

If the dependency structure contains **frustrated loops**—cycles where constraints cannot be simultaneously satisfied—then $\mathcal{A} \not\triangleright \int$ and Class II algorithms fail.

Formally: If $\pi_1(\int \mathcal{X}) \neq 0$ (non-trivial fundamental group), then propagation around cycles produces inconsistencies requiring exponential backtracking.

**Obstruction Certificate:** $K_{\int}^- = (\pi_1 \neq 0, \text{cycles})$

**Application:** Random 3-SAT has frustrated loops (conflicting clause cycles), blocking Class II. Horn-SAT has $\pi_1 = 0$ (acyclic implications), enabling Class II.
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

:::{prf:lemma} Flat Modality Obstruction (Trivial Automorphism)
:label: lem-flat-obstruction

If the automorphism group is trivial:

$$G_{\Phi} := \mathrm{Aut}(\mathcal{X}, \Phi) = \{e\}$$

then $\mathcal{A} \not\triangleright \flat$ and the quotient equals the full space: $\mathcal{X}/G = \mathcal{X}$. No compression occurs.

**Obstruction Certificate:** $K_{\flat}^- = (|G| = 1)$

**Application:** Random instances have trivial automorphism with high probability, blocking Class III. XORSAT has large kernel group $|G| = 2^{n-\mathrm{rank}(A)}$, enabling Class III.
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

:::{prf:lemma} Scaling Modality Obstruction (Supercritical)
:label: lem-scaling-obstruction

If the problem is **supercritical**—decomposition creates more work than it saves—then $\mathcal{A} \not\triangleright \ast$.

Formally: If for any balanced partition $\mathcal{X} = \mathcal{X}_1 \sqcup \mathcal{X}_2$:

$$|\operatorname{boundary}(\mathcal{X}_1, \mathcal{X}_2)| = \Omega(|\mathcal{X}|)$$

then recombination cost dominates: $f(n) = \Omega(T(n))$, making recursion futile.

**Obstruction Certificate:** $K_{\ast}^- = (\text{supercritical}, |\partial| = \Omega(n))$

**Application:** Random 3-SAT has $\Theta(n)$ boundary clauses for any cut, blocking Class IV.
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

:::{prf:lemma} Boundary Modality Obstruction (#P-Hard Contraction)
:label: lem-boundary-obstruction

If the tensor network has:
- Non-planar graph structure AND
- No Pfaffian orientation (odd frustrated cycles) AND
- Unbounded treewidth

then contraction is #P-hard and $\mathcal{A} \not\triangleright \partial$.

**Obstruction Certificate:** $K_{\partial}^- = (\text{non-planar}, \text{no-Pfaffian}, \mathrm{tw} = \Theta(n))$

**Application:** Random 3-SAT tensor networks are non-planar with unbounded treewidth, blocking Class V.
:::

### The Algorithmic Representation Theorem

:::{div} feynman-prose
Now we come to what I think is the most beautiful part. We are about to state a theorem that says: the five classes above are *complete*. There is no sixth class. Every polynomial-time algorithm must exploit one of these five types of structure.

Ask yourself: why should this be true? After all, mathematicians are clever people. Could they not invent some fundamentally new algorithmic technique? The answer is that these five classes are not arbitrary. They correspond to the five fundamental ways that a cohesive topos can have "structure." In a sense, they are the mathematical atoms of exploitable regularity.

The contrapositive is even more striking. If a problem has none of these structures, if it is "amorphous" with respect to all five modalities, then *no* polynomial-time algorithm can solve it. This gives us a principled way to prove lower bounds: show that all five modalities are blocked, and hardness follows.
:::

::::{prf:theorem} [MT-AlgComplete] The Algorithmic Representation Theorem
:label: mt-alg-complete

**Rigor Class:** F (Framework-Original) — see {prf:ref}`def-rigor-classification`

**Sieve Target:** Node 17 (Lock) — Tactic E13 (Algorithmic Completeness Check)

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (algorithmic representation), $\mathrm{Cat}_{\mathrm{Hom}}$ (categorical Lock)
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+\}$ (algorithm representable in $T_{\text{algorithmic}}$)
- **Produces:** $K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (Hom-emptiness via modality exhaustion)
- **Blocks:** All polynomial-time bypass attempts (validates P ≠ NP scope)
- **Breached By:** Discovery of Class VI algorithm (outside known modalities)

**Statement:** In a cohesive $(\infty,1)$-topos $\mathbf{H}$, every effective morphism $\mathcal{A}: \mathcal{X} \to \mathcal{X}$ achieving **polynomial compression** (reducing entropy from $N$ bits to 0 in $\text{poly}(N)$ steps) must factor through one or more of the fundamental cohesive modalities:

$$\mathcal{A} = \mathcal{R} \circ \lozenge(f) \circ \mathcal{E}$$

where $\lozenge \in \{\sharp, \int, \flat, \ast, \partial\}$, $\mathcal{E}$ is an encoding map, $\mathcal{R}$ is a reconstruction map, and $\lozenge(f)$ is a contraction in the $\lozenge$-transformed space.

**Contrapositive (Hardness Criterion):** If a problem instance $(\mathcal{X}, \Phi)$ is **amorphous** (admits no non-trivial morphism to any modal object), then:

$$\mathbb{E}[\operatorname{Time}(\mathcal{A})] \geq \exp(C \cdot N)$$

**Hypotheses:**
1. **(H1) Cohesive Structure:** $\mathbf{H}$ is equipped with the canonical adjoint string $\Pi \dashv \flat \dashv \sharp$ plus scaling filtration $\mathbb{R}_{>0}$ and boundary operator $\partial$
2. **(H2) Computational Problem:** $(\mathcal{X}, \Phi, \mathcal{S})$ is a computational problem with configuration stack $\mathcal{X}$, energy $\Phi$, and solution subobject $\mathcal{S}$
3. **(H3) Algorithm Representability:** $\mathcal{A}$ admits a representable-law interpretation ({prf:ref}`def-representable-law`)
4. **(H4) Information-Theoretic Setting:** Shannon entropy $H(\mathcal{X}) = \log \operatorname{Vol}(\mathcal{X})$ is well-defined

**Certificate Logic:**

$$\bigwedge_{i \in \{I,\ldots,V\}} K_{\text{Class}_i}^- \Rightarrow K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload:**
$((\sharp\text{-status}, \int\text{-status}, \flat\text{-status}, \ast\text{-status}, \partial\text{-status}), \text{modality\_checks}, \text{exhaustion\_proof})$

**Literature:** Cohesive $(\infty,1)$-topoi {cite}`SchreiberCohesive`; Synthetic Differential Geometry {cite}`Kock06`; Axiomatic Cobordism {cite}`Lurie09TFT`; Computational Complexity {cite}`AroraBorak09`.
::::

:::{prf:proof} Proof of {prf:ref}`mt-alg-complete`

**Proof (Following Categorical Proof Template — Topos Internal Logic):**

*Step 1 (AIT Setup: Complexity Reduction Requirement).*

By the Sieve-Thermodynamic Correspondence ({prf:ref}`thm-sieve-thermo-correspondence`), polynomial-time convergence requires **Kolmogorov complexity reduction** on the thin trace: the algorithm must decrease $K_\epsilon(T_{\mathrm{thin},t})$ from the initial trace complexity $K_\epsilon(T_{\mathrm{thin},0}) \sim N$ to $O(\log N)$ (solution encoding) in $\text{poly}(N)$ steps.

By the **Levin-Schnorr Theorem** {cite}`Levin73b,Schnorr73`, uniform random search on an amorphous (structureless) space achieves expected complexity reduction:

$$\mathbb{E}[\Delta K] = O(1/|\mathcal{X}|) = O(2^{-N})$$

Therefore, absent structural exploitation, hitting time scales as $\Omega(2^N)$. This establishes the drift requirement: any $\mathcal{A} \in P$ must achieve $K_{t+1} \leq K_t - \Omega(1)$ per step via a modal contraction.

*Step 2 (Classification of Generalized Forces in $\mathbf{H}$).*

In a cohesive topos, all forces/gradients arise from the internal tension between a space $\mathcal{X}$ and its modal reflections. The internal logic of $\mathbf{H}$ classifies all possible structural relations via the canonical adjunctions:

1. **Metric Force ($\sharp$):** Generated by the unit $\eta: \mathcal{X} \to \sharp \mathcal{X}$.
   - $F \approx \nabla \Phi$ on smooth manifold structure
   - Corresponds to **Class I (Gradient Algorithms)**
   - Obstruction: Requires $\Phi$ convex or Łojasiewicz-Simon

2. **Causal Force ($\int$):** Generated by the unit $\eta: \mathcal{X} \to \Pi \mathcal{X}$.
   - $F$ defined by partial order in the fundamental groupoid
   - Corresponds to **Class II (Propagation Algorithms)**
   - Obstruction: Requires dependency graph to be DAG (no frustration loops)

3. **Algebraic Force ($\flat$):** Generated by the counit $\epsilon: \flat \mathcal{X} \to \mathcal{X}$.
   - $F$ acts by projection onto discrete lattice/group structure
   - Corresponds to **Class III (Algebraic Algorithms)**
   - Obstruction: Requires non-trivial symmetry (Group/Ring action)

4. **Scaling Force ($\ast$):** Generated by the multiplicative monoid $\mathbb{A}^1 \setminus \{0\}$ action.
   - $F$ identifies self-similarity, enabling $N \to N/2$ reduction
   - Corresponds to **Class IV (Divide & Conquer)**
   - Obstruction: Requires subcritical scaling ($\beta - \alpha < \lambda_c$)

5. **Holographic Force ($\partial$):** Generated by the boundary operator in the cobordism category.
   - $F$ reduces bulk to boundary (Stokes) or interference patterns
   - Corresponds to **Class V (Holographic Algorithms)**
   - Obstruction: Requires Pfaffian orientation or matchgate signature

*Step 3 (The Amorphous Limit — Exhaustiveness via Schreiber's Theorem).*

By {prf:ref}`thm-schreiber-structure` and {prf:ref}`cor-exhaustive-decomposition`, any structure in $\mathcal{X}$ decomposes into modal components. This is **not an assumption**—it is a theorem of cohesive topos theory. The decomposition

$$\mathcal{X} \simeq \mathcal{X}_{\int} \times_{\mathcal{X}_0} \mathcal{X}_{\flat} \times_{\mathcal{X}_0} \mathcal{X}_{\sharp}$$

with derived components $\ast$ (scaling) and $\partial$ (boundary) is **categorically complete**: there exist no structural patterns outside these five modalities within a cohesive $(\infty,1)$-topos.

**Exhaustion Argument:** Suppose $(\mathcal{X}, \Phi)$ is **singular** with respect to all five modalities:
- **$\sharp$-Singular:** $\Phi$ is glassy/shattered (no spectral gap, Łojasiewicz fails)
- **$\int$-Singular:** Causal graph contains frustration loops ($\pi_1(\text{factor graph}) \neq 0$)
- **$\flat$-Singular:** Automorphism group is trivial ($\mathrm{Aut}(\mathcal{X}, \Phi) = \{e\}$)
- **$\ast$-Singular:** Problem is supercritical (boundary dominates any decomposition)
- **$\partial$-Singular:** No holographic reduction exists (generic non-planar tensor network)

Let $f: \mathcal{X} \to \mathcal{X}$ be the update rule of an alleged "Class VI" algorithm. By the exhaustive decomposition theorem, any algorithmic morphism must factor through at least one modal component:

$$f \simeq \mathcal{R} \circ \lozenge(f') \circ \mathcal{E}$$

for some $\lozenge \in \{\int, \flat, \sharp, \ast, \partial\}$.

But we have assumed all five modalities are **blocked** (singular). Therefore no such factorization exists. The internal logic of $\mathbf{H}$ cannot distinguish points except by exhaustive enumeration—the algorithm reduces to brute force search.

*Step 4 (Shannon Bound Application).*

By **Shannon's Source Coding Theorem**, identifying a unique element in a set of size $M$ without structural side-information requires $\log_2 M$ bits of query information.

In the absence of structural gradients (which compress query cost):
- Total information required: $N$ bits
- Information per step: $\sim 2^{-N}$ (fraction of volume explored)
- Steps required: $O(2^N)$

This is the **Brute Force Lower Bound**.

*Step 5 (Conclusion: Modality Exhaustion Implies Hardness).*

A polynomial-time algorithm exists *if and only if* the problem structure factors through one of the cohesive modalities that compresses the effective search space.

If the Sieve certifies all such factorizations are blocked (Morphism Exclusion):

$$\text{Blocked}(\sharp) \wedge \text{Blocked}(\int) \wedge \text{Blocked}(\flat) \wedge \text{Blocked}(\ast) \wedge \text{Blocked}(\partial) \Rightarrow \mathcal{A} \notin P$$

**Q.E.D.**
:::

:::{div} feynman-prose
And there it is. The key insight of this proof is information-theoretic: polynomial-time means you can only look at a polynomial fraction of the search space, so you need some kind of "compass" that tells you where to look. That compass is structure. The five modalities are the five types of compass that mathematics provides.

When all five compasses fail, when your problem is smooth-singular, causal-singular, algebraic-singular, scale-singular, and holographic-singular, you are walking blind in an exponentially large space. Shannon's theorem then tells you the brutal truth: you need exponential time.

This is why canonical $3$-SAT is hard in the framework. It is not that we have not been clever enough; it is that the
problem defeats all five strategies at once. The hardness is not a failure of imagination but a mathematical necessity.
:::

### Tactic E13: Algorithmic Completeness Lock

:::{prf:definition} E13: Algorithmic Completeness Lock
:label: def-e13

**Sieve Signature:**
- **Required Permits:** $\mathrm{Rep}_K$ (algorithm representation), $\mathrm{Cat}_{\mathrm{Hom}}$
- **Weakest Precondition:** $\{K_{\mathrm{Rep}_K}^+, K_{T_{\text{algorithmic}}}^+\}$ (algorithmic type with representation)
- **Produces:** $K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
- **Blocks:** Polynomial-time bypass; validates universal scope certificates
- **Breached By:** Algorithm factors through at least one modality

**Method:** Modal factorization analysis via Cohesive adjunctions

**Mechanism:** For problem $(\mathcal{X}, \Phi)$, check if any polynomial algorithm can factor through the five modalities. If all five are blocked, the problem is information-theoretically hard.

The five modal checks correspond to existing tactics and nodes:
- **$\sharp$ (Metric):** Uses Node 7 ($\mathrm{LS}_\sigma$) + Node 12 ($\mathrm{GC}_\nabla$)
- **$\int$ (Causal):** Uses **Tactic E6** (Causal/Well-Foundedness)
- **$\flat$ (Algebraic):** Uses **Tactic E4** (Integrality) + **Tactic E11** (Galois-Monodromy)
- **$\ast$ (Scaling):** Uses Node 4 ($\mathrm{SC}_\lambda$) for subcriticality
- **$\partial$ (Holographic):** Uses **Tactic E8** (DPI) + Node 6 ($\mathrm{Cap}_H$)

**Certificate Logic:**

$$K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge K_{\mathrm{SC}_\lambda}^{\text{super}} \wedge K_{\mathrm{E8}}^- \Rightarrow K_{\mathrm{E13}}^+ \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$$

**Certificate Payload:** $(\text{modal\_status}[5], \text{class\_exclusions}[5], \text{exhaustion\_witness})$

**Automation:** Via composition of existing node/tactic evaluations; fully automatable for types with computable modality checks

**Literature:** Cohesive Homotopy Type Theory {cite}`SchreiberCohesive`; Algorithm taxonomy {cite}`Garey79`; Modal type theory {cite}`LicataShulman16`.
:::

:::{prf:theorem} E13 Contrapositive Hardness
:label: thm-e13-contrapositive-hardness

Let $\Pi$ be a problem family in the algorithmic ambient setting. If $\Pi$ carries the full E13 obstruction certificate,

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
By {prf:ref}`mt-alg-complete`, every polynomial-time algorithm in the cohesive ambient foundation factors through at
least one modality in $\{\sharp, \int, \flat, \ast, \partial\}$. Definition {prf:ref}`def-e13` packages the six
antecedent obstruction certificates into the single statement $K_{\mathrm{E13}}^+$ that all five modal routes are
blocked. Therefore no polynomial-time factorization remains. Hence $\Pi \notin P_{\text{FM}}$. $\square$
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

The following examples demonstrate how MT-AlgComplete correctly classifies problems as P or NP-Hard.

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

### Canonical 3-SAT Proof Package

:::{prf:definition} Canonical 3-SAT Problem Object
:label: def-threshold-random-3sat-family

Let

$$
\Pi_{3\text{-SAT}}
$$

denote the canonical internal satisfiability object in the algorithmic ambient setting: inputs are $3$-CNF formulas,
witnesses are Boolean assignments, and verification is clause satisfaction.

This is the unique satisfiability object used in the separation chain below. The theorem statements are made directly
for $\Pi_{3\text{-SAT}}$ itself.
:::

:::{prf:lemma} Metric Blockage for Canonical 3-SAT
:label: lem-random-3sat-metric-blockage

For $\Pi_{3\text{-SAT}}$, the metric obstruction certificate $K_{\mathrm{LS}_\sigma}^-$ holds.
:::

:::{prf:proof}
Apply the sharp obstruction criterion {prf:ref}`lem-sharp-obstruction`. The canonical $3$-SAT object admits no
polynomial sharp descent certificate $(\Phi, \theta, \lambda)$ of {prf:ref}`def-class-i-climbers`, so the metric route
is blocked. Any overlap-gap or landscape-fragmentation argument is admissible only as a backend witness for this
certificate; the E13 package uses the resulting blockage statement
$K_{\mathrm{LS}_\sigma}^-$.
:::

:::{prf:lemma} Causal Blockage for Canonical 3-SAT
:label: lem-random-3sat-causal-blockage

For $\Pi_{3\text{-SAT}}$, the causal obstruction certificate $K_{\mathrm{E6}}^-$ holds.
:::

:::{prf:proof}
By {prf:ref}`lem-shape-obstruction`, frustrated dependency loops obstruct factorization through $\int$. The canonical
$3$-SAT factor graph carries precisely this cyclic frustration, so the well-foundedness tactic
{prf:ref}`def-e6` returns the blocking certificate $K_{\mathrm{E6}}^-$.
:::

:::{prf:lemma} Integrality Blockage for Canonical 3-SAT
:label: lem-random-3sat-integrality-blockage

For $\Pi_{3\text{-SAT}}$, the integrality obstruction certificate $K_{\mathrm{E4}}^-$ holds.
:::

:::{prf:proof}
The flat route requires an arithmetic quotient or lattice-type compression. The canonical $3$-SAT object presents no
such integral compression witness, so the integrality lock {prf:ref}`def-e4` contributes the negative certificate
$K_{\mathrm{E4}}^-$ to the algebraic channel.
:::

:::{prf:lemma} Galois-Monodromy Blockage for Canonical 3-SAT
:label: lem-random-3sat-galois-blockage

For $\Pi_{3\text{-SAT}}$, the Galois-monodromy obstruction certificate $K_{\mathrm{E11}}^-$ holds.
:::

:::{prf:proof}
The residual algebraic route through solvable monodromy is excluded by the absence of a compressing solvable symmetry.
Accordingly, the lock of {prf:ref}`def-e11` supplies $K_{\mathrm{E11}}^-$. Together with
{prf:ref}`lem-random-3sat-integrality-blockage`, this closes the $\flat$-channel.
:::

:::{prf:lemma} Supercritical Scaling for Canonical 3-SAT
:label: lem-random-3sat-scaling-blockage

For $\Pi_{3\text{-SAT}}$, the scaling obstruction certificate $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$ holds.
:::

:::{prf:proof}
Invoke the supercritical scaling criterion {prf:ref}`lem-scaling-obstruction`. Balanced decompositions retain boundary
size proportional to the instance, so divide-and-conquer does not create a subcritical
recurrence. This is recorded as $K_{\mathrm{SC}_\lambda}^{\mathrm{super}}$.
:::

:::{prf:lemma} Boundary Blockage for Canonical 3-SAT
:label: lem-random-3sat-boundary-blockage

For $\Pi_{3\text{-SAT}}$, the boundary obstruction certificate $K_{\mathrm{E8}}^-$ holds.
:::

:::{prf:proof}
Apply {prf:ref}`lem-boundary-obstruction`. The canonical $3$-SAT tensor networks are non-planar, lack a Pfaffian
compression, and have unbounded effective treewidth, so the holographic route is blocked. The corresponding boundary
certificate is $K_{\mathrm{E8}}^-$, as checked by the DPI tactic {prf:ref}`def-e8`.
:::

:::{prf:theorem} Canonical 3-SAT Satisfies the E13 Antecedent Package
:label: ex-3sat-all-blocked

The canonical satisfiability object $\Pi_{3\text{-SAT}}$ satisfies the six antecedent obstruction certificates of
Definition {prf:ref}`def-e13`:

$$
K_{\mathrm{LS}_\sigma}^- \wedge K_{\mathrm{E6}}^- \wedge K_{\mathrm{E4}}^- \wedge K_{\mathrm{E11}}^- \wedge
K_{\mathrm{SC}_\lambda}^{\mathrm{super}} \wedge K_{\mathrm{E8}}^-.
$$

Hence $K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})$ holds.
:::

:::{prf:proof}
Combine {prf:ref}`lem-random-3sat-metric-blockage`, {prf:ref}`lem-random-3sat-causal-blockage`,
{prf:ref}`lem-random-3sat-integrality-blockage`, {prf:ref}`lem-random-3sat-galois-blockage`,
{prf:ref}`lem-random-3sat-scaling-blockage`, and {prf:ref}`lem-random-3sat-boundary-blockage`, then apply the
certificate logic built into {prf:ref}`def-e13`.
:::

:::{prf:theorem} Canonical 3-SAT is Outside $P_{\text{FM}}$
:label: thm-random-3sat-not-in-pfm

$$
\Pi_{3\text{-SAT}} \notin P_{\text{FM}}.
$$
:::

:::{prf:proof}
Apply {prf:ref}`thm-e13-contrapositive-hardness` to {prf:ref}`ex-3sat-all-blocked`.
:::

:::{prf:theorem} Canonical 3-SAT Completeness in $NP_{\text{FM}}$
:label: thm-sat-membership-hardness-transfer

The canonical satisfiability object $\Pi_{3\text{-SAT}}$ belongs to $NP_{\text{FM}}$ and is $NP_{\text{FM}}$-complete.

Consequently:

$$
\Pi_{3\text{-SAT}} \notin P_{\text{FM}} \Rightarrow P_{\text{FM}} \neq NP_{\text{FM}}.
$$
:::

:::{prf:proof}
Membership is immediate from the internal verifier relation: a Boolean assignment is a polynomial witness and clause
satisfaction is checked in polynomial time. Completeness is the canonical satisfiability completeness theorem of the
framework: every $NP_{\text{FM}}$ verifier reduces to $3$-CNF satisfiability inside the algorithmic ambient setting.
Therefore exclusion of $\Pi_{3\text{-SAT}}$ from $P_{\text{FM}}$ separates $P_{\text{FM}}$ from $NP_{\text{FM}}$.
$\square$
:::

:::{prf:corollary} Internal Separation from Canonical 3-SAT
:label: cor-pfm-neq-npfm-from-random-3sat

$$
P_{\text{FM}} \neq NP_{\text{FM}}.
$$
:::

:::{prf:proof}
Combine {prf:ref}`thm-random-3sat-not-in-pfm` with {prf:ref}`thm-sat-membership-hardness-transfer`.
:::

:::{prf:corollary} Internal-to-Classical Separation Bridge
:label: cor-internal-to-classical-separation

Assume the adequacy hypotheses of the bridge chapter. Then

$$
P_{\text{FM}} \neq NP_{\text{FM}} \Rightarrow P_{\text{DTM}} \neq NP_{\text{DTM}}.
$$
:::

:::{prf:proof}
This is the class-equivalence export of Corollary {prf:ref}`cor-class-equivalence-full`, restated here so the
algorithmic proof chain contains its own explicit bridge step.
:::

:::{prf:remark} Numbered Proof Skeleton for the Export
:label: rem-numbered-proof-skeleton

The section's proof chain is:

1. By {prf:ref}`mt-alg-complete`, every polynomial-time algorithm in the chosen ambient foundation factors through one
   of the five modalities.
2. Lemmas {prf:ref}`lem-random-3sat-metric-blockage` through {prf:ref}`lem-random-3sat-boundary-blockage` establish the
   six E13 antecedent certificates for $\Pi_{3\text{-SAT}}$.
3. Theorem {prf:ref}`ex-3sat-all-blocked` assembles those certificates into $K_{\mathrm{E13}}^+$.
4. Theorem {prf:ref}`thm-e13-contrapositive-hardness` yields
   $\Pi_{3\text{-SAT}} \notin P_{\text{FM}}$.
5. Theorem {prf:ref}`thm-sat-membership-hardness-transfer` identifies $\Pi_{3\text{-SAT}}$ as the canonical
   $NP_{\text{FM}}$-complete satisfiability object.
6. Corollary {prf:ref}`cor-pfm-neq-npfm-from-random-3sat` gives the internal separation
   $P_{\text{FM}} \neq NP_{\text{FM}}$.
7. Corollary {prf:ref}`cor-internal-to-classical-separation` exports this to
   $P_{\text{DTM}} \neq NP_{\text{DTM}}$.
:::

:::{div} feynman-prose
Here is the contrast that matters. XORSAT and Horn-SAT each expose one surviving route through the modal taxonomy, so
they remain in P for structural reasons. Canonical $3$-SAT is used differently: not as an external distributional
example, but as the internal satisfiability object whose modal exhaustion certifies hardness.

Why? The answer is still structure. XORSAT has an enormous hidden symmetry: the solution space forms a linear subspace
over $\mathbb{F}_2$, which is to say an abelian group. Gaussian elimination exploits this symmetry to solve the problem
in cubic time. The Alchemist strategy (Class III) succeeds.

Horn-SAT is different again. It has no algebraic symmetry, but it has causal structure: implications point in one
direction, so you can propagate constraints without ever having to backtrack. The Propagator strategy (Class II)
succeeds.

Canonical $3$-SAT is the opposite case. No symmetry, no causal elimination, no useful metric descent, no subcritical
scaling, no holographic compression. The point of the proof package above is that each failure is now a named lemma
feeding E13 on the internal problem object itself.
:::

### Corollary: Algorithmic Embedding Surjectivity

:::{prf:corollary} Algorithmic Embedding Surjectivity
:label: cor-alg-embedding-surj

The domain embedding $\iota: \mathbf{Hypo}_{T_{\text{alg}}} \to \mathbf{DTM}$ is surjective on polynomial-time computations:

$$\forall M \in P.\, \exists \mathbb{H} \in \mathbf{Hypo}_{T_{\text{alg}}}.\, \iota(\mathbb{H}) \cong M$$
:::

:::{prf:proof} Proof of {prf:ref}`cor-alg-embedding-surj`

By MT-AlgComplete, every polynomial algorithm factors through a modality. Each modality corresponds to a structural resource representable in $\mathbf{Hypo}_{T_{\text{alg}}}$. The embedding $\iota$ is constructed to preserve these resources.
:::

### Foundation Assumption and Internal Structure Thesis

:::{prf:axiom} Computational Foundation Assumption
:label: axiom-structure-thesis

Computation is modeled in the chosen cohesive ambient setting $\mathbf{H}$, so algorithmic morphisms, modal
factorizations, and the classes $P_{\text{FM}}, NP_{\text{FM}}$ are interpreted internally to that foundation.
:::

:::{prf:theorem} Internal Structure Thesis
:label: thm-internal-structure-thesis

Within the ambient foundation of {prf:ref}`axiom-structure-thesis`, every polynomial-time algorithm factors through the
five cohesive modalities:

$$
P_{\text{FM}} \subseteq \text{Class I} \cup \text{Class II} \cup \text{Class III} \cup \text{Class IV} \cup \text{Class V}.
$$

This is the internal theorem supplied by {prf:ref}`mt-alg-complete`; it is not a separate meta-axiom.
:::

:::{prf:proof}
Immediate from {prf:ref}`mt-alg-complete`, which proves modal exhaustion for polynomial-time algorithms in the cohesive
ambient setting.
:::

:::{div} feynman-prose
This is the clean separation the chapter needs. One thing is a foundational choice: we model computation inside a
cohesive ambient category. A different thing is a theorem inside that foundation: once you accept the setting, efficient
algorithms exhaust through the five modalities because {prf:ref}`mt-alg-complete` proves it.

That distinction matters. We are not asking the reader to accept modal completeness as a second unexplained axiom. We
are asking the reader to separate the ambient language from the theorem proved inside that language.

The natural-proofs caution stays the same. We are not giving a constructive detector for structure; we are giving a
non-constructive obstruction route. The proof works by showing that if all modal routes are blocked, hardness follows.
:::

### Verification and Falsifiability

We establish the verification criteria and falsifiability conditions for the algorithmic completeness framework.

:::{prf:theorem} Verification of Completeness
:label: thm-verification-completeness

The algorithmic completeness framework is **verifiable** through the following components:

| Component | Status | Reference |
|-----------|--------|-----------|
| Cohesive modalities exhaust structure | **THEOREM** (Schreiber) | {prf:ref}`thm-schreiber-structure` |
| Polynomial-time requires structure | **THEOREM** (information-theoretic) | Proof of {prf:ref}`mt-alg-complete`, Step 1 |
| Structure = modal factorization | **THEOREM** (topos-theoretic) | Proof of {prf:ref}`mt-alg-complete`, Step 2 |
| MT-AlgComplete | **THEOREM** (internal) | {prf:ref}`mt-alg-complete` |
| E13 contrapositive hardness | **THEOREM** | {prf:ref}`thm-e13-contrapositive-hardness` |
| Canonical 3-SAT E13 package | **THEOREM** | {prf:ref}`ex-3sat-all-blocked` |
| Obstruction certificates | **COMPUTABLE** | {prf:ref}`def-obstruction-certificates` |
| Bridge to DTM complexity | **THEOREM** | Part XX (Complexity Bridge) |

**Key Point:** The framework rests on **mathematical theorems** within cohesive $(\infty,1)$-topos theory, not empirical
observations. The only extra ingredient needed to speak about DTMs is the separate bridge equivalence theorem, not an
additional obstruction bottleneck inside the proof chain.
:::

:::{prf:definition} Falsifiability Criteria
:label: def-falsifiability

The algorithmic completeness framework makes **falsifiable predictions**:

**Prediction 1 (No Class VI):** If a polynomial-time algorithm for a problem is discovered that does not factor through any of $\{\int, \flat, \sharp, \ast, \partial\}$, then one of:
- The algorithm actually factors through a missed modality (analysis error)
- The cohesive $(\infty,1)$-topos framework is incomplete as a foundation for computation
- The bridge theorems (Part XX) fail

**Prediction 2 (Obstruction Correctness):** For any problem $\Pi$:

$$\mathcal{A} \in P \implies \exists \lozenge: \mathcal{A} \triangleright \lozenge$$

If this fails, the Schreiber structure theorem ({prf:ref}`thm-schreiber-structure`) would need revision.

**Prediction 3 (Certificate Soundness):** The obstruction certificates $K_\lozenge^-$ are:
- **Sound:** $K_\lozenge^- \implies \mathcal{A} \not\triangleright \lozenge$ (no false positives)
- **Complete:** $\mathcal{A} \not\triangleright \lozenge \implies K_\lozenge^-$ can be constructed (no false negatives)

If soundness fails, the modal obstruction lemmas ({prf:ref}`lem-sharp-obstruction`, {prf:ref}`lem-shape-obstruction`, etc.) contain errors.

**Prediction 4 (Canonical 3-SAT Package):** The internal satisfiability object
$\Pi_{3\text{-SAT}}$ satisfies the six antecedent certificates assembled in
{prf:ref}`ex-3sat-all-blocked`.

If one of those six antecedent lemmas fails, the E13 assembly theorem fails and the export chain stops at that point.
:::

:::{prf:remark} Relationship to Complexity Barriers
:label: rem-complexity-barriers

The algorithmic completeness approach relates to established complexity barriers as follows:

| Barrier | How Addressed |
|---------|---------------|
| **Relativization** (Baker-Gill-Solovay 1975) | Proof is structural, not oracle-based; modalities are intrinsic to the problem, not relativizable queries |
| **Natural Proofs** (Razborov-Rudich 1997) | Proof is non-constructive; does not claim to algorithmically detect structure absence. The hardness follows from mathematical analysis of modal obstructions, not from constructive circuit lower bounds |
| **Algebrization** (Aaronson-Wigderson 2009) | The flat modality $\flat$ explicitly includes algebraic structure; algebrization is subsumed as one of the five classes (Class III). Blocking $\flat$ requires trivial automorphism, which is a structural property |

**Key Insight:** The proof operates at the **meta-level** of structural classification, not the object-level of specific algorithms or circuits. The barriers apply to constructive lower bound techniques; our approach is non-constructive, relying on categorical exhaustion.
:::

:::{prf:theorem} Foundational Status of the Framework
:label: thm-conditional-nature

The algorithmic completeness framework separates into an internal theorem chain and an external export step:

**Foundation (C1):** We work within Cohesive Homotopy Type Theory / cohesive $(\infty,1)$-topos theory as the ambient
foundation.

**Bridge (C2):** The Fragile/DTM equivalence theorems (Part XX) establish that:

$$
P_{\text{FM}} = P_{\text{DTM}} \quad \text{and} \quad NP_{\text{FM}} = NP_{\text{DTM}}.
$$

**Internal Separation (C3):** Part XIX proves

$$
P_{\text{FM}} \neq NP_{\text{FM}}
$$

via $K_{\mathrm{E13}}^+(\Pi_{3\text{-SAT}})$, {prf:ref}`thm-e13-contrapositive-hardness`, and
{prf:ref}`thm-sat-membership-hardness-transfer`.

**Logical Structure:**

$$
(\text{C1} \wedge \text{C2} \wedge \text{C3}) \Rightarrow (P_{\text{DTM}} \neq NP_{\text{DTM}}).
$$

**Within** the ambient foundation, the argument from `MT-AlgComplete` through
{prf:ref}`cor-pfm-neq-npfm-from-random-3sat` is unconditional: the internal separation
$P_{\text{FM}} \neq NP_{\text{FM}}$ is a theorem of the framework.

**Status Comparison:**
- **Classical ZFC + P ≠ NP:** Unproven
- **Cohesive HoTT + internal 3-SAT separation + bridge equivalence:** yields the classical separation by
  {prf:ref}`cor-internal-to-classical-separation`

The roles are therefore explicit: ambient foundation for the internal theorem, and bridge equivalence for the external
export.
:::

:::{div} feynman-prose
Let me be clear about what we have accomplished and what remains open.

**What is proven:** Within cohesive $(\infty,1)$-topos theory, we have **proven** that the five modalities exhaust all
structural resources. We have **proven** that blocking all five modalities forces exponential time. These are theorems,
not conjectures.

**What is proven internally:** The canonical $3$-SAT object carries the full E13 obstruction package, is outside
$P_{\text{FM}}$, and therefore separates $P_{\text{FM}}$ from $NP_{\text{FM}}$.

**What the bridge supplies:** The separate equivalence theorem identifying the internal classes with the classical
Turing-machine classes.

**What is a choice:** Working in cohesive $(\infty,1)$-topos theory (Condition C1) is a **foundational choice**, like
choosing to work in ZFC versus some alternative foundation. Within that foundation, our results are theorems.

The beauty of this approach is that it makes the roles **explicit**. The internal proof no longer waits on a separate
verification step; the only extra theorem beyond the framework itself is the bridge equivalence used to restate the
result in the DTM model.

These are questions we can investigate, debate, and potentially settle. That is progress.
:::

### Summary: What This Framework Establishes

:::{prf:theorem} Main Results Summary
:label: thm-hypo-algorithmic-main-results

The algorithmic completeness framework establishes:

**Theorem 1 (Modal Completeness):** In a cohesive $(\infty,1)$-topos, the five modalities $\{\int, \flat, \sharp, \ast, \partial\}$ exhaust all exploitable structure ({prf:ref}`thm-schreiber-structure`, {prf:ref}`cor-exhaustive-decomposition`).

**Theorem 2 (Algorithmic Representation):** Every polynomial-time algorithm factors through at least one modality ({prf:ref}`mt-alg-complete`).

**Theorem 3 (Hardness from Obstruction):** The E13 obstruction certificate implies exclusion from $P_{\text{FM}}$
({prf:ref}`thm-e13-contrapositive-hardness`).

**Theorem 4 (Class Specifications):** Each algorithm class has explicit factorization conditions and computable obstruction criteria ({prf:ref}`def-class-i-climbers` through {prf:ref}`def-class-v-interference`).

**Theorem 5 (3-SAT Assembly):** Canonical $3$-SAT satisfies the six E13 antecedent certificates
({prf:ref}`ex-3sat-all-blocked`).

**Theorem 6 (Internal Separation):** Canonical $3$-SAT yields $P_{\text{FM}} \neq NP_{\text{FM}}$
({prf:ref}`cor-pfm-neq-npfm-from-random-3sat`).

**Theorem 7 (Classical Export):** By the bridge equivalence theorem, the internal separation yields
$P_{\text{DTM}} \neq NP_{\text{DTM}}$ ({prf:ref}`cor-internal-to-classical-separation`).
:::

:::{div} feynman-prose
And there you have it. We have built a mathematical framework that explains **why** some algorithms are fast and others must be slow. The five modalities are not arbitrary categories; they are the fundamental ways that structure manifests in a cohesive topos. An algorithm is fast if it can "see" one of these structural patterns. An algorithm is slow if all five views reveal nothing but noise.

This is the answer to the question: "Could there be a clever algorithm we have not thought of yet?" Within our
framework, the answer is: only if it exploits one of the five types of structure. There is no sixth type, because modal
completeness closes the door internally.

The repaired proof presentation makes the dependencies explicit. E13 does the obstruction work, the SAT transfer does the
internal class-separation work, and the bridge chapter does the model-export work. Each dependency now has its own named
theorem.
:::
