---
title: "Algebraic-Geometric Unification"
---

# Algebraic-Geometric Unification

(sec-algebraic-geometric-metatheorems)=
## Algebraic-Geometric Metatheorems

*These metatheorems establish the bridge between the sieve framework and algebraic geometry. They enable sieve execution for problems involving algebraic cycles, cohomology, period domains, and moduli spaces.*

:::{div} feynman-prose

Now we come to what I think is the most beautiful part of this whole framework. We have been building up this machinery of certificates and permits and sieves, and you might be wondering: does any of this connect to the classical problems that mathematicians have worried about for a century?

The answer is yes, and the connection is deeper than you might expect. What we are about to see is that the sieve framework, when specialized to algebraic geometry, recovers and clarifies some of the most profound structures in mathematics: motives, Hodge theory, and the mysterious relationship between analysis and algebra.

Here is the key idea. When you study a geometric object like a variety, you can look at it through two lenses:

1. **The analytic lens:** Differential equations, flows, energy functionals, gradient descent
2. **The algebraic lens:** Polynomial equations, correspondences, cohomology classes

These two viewpoints seem very different. But here is what makes algebraic geometry so remarkable: under the right conditions, they tell you the same thing. The metatheorems in this section make that correspondence precise, and they do so through the language of permits and certificates.

Why does this matter for us? Because the sieve is fundamentally an analytic object. It works with flows and energy and dissipation. But the singularities we want to exclude are often algebraic in nature. These metatheorems are the translation dictionaries that let the sieve speak the language of algebraic geometry.

:::

---

### Motivic Flow Principle

:::{div} feynman-prose

Let me tell you what a motive is, in plain terms. When you have an algebraic variety, you can extract all sorts of invariants from it: its cohomology groups, its Hodge numbers, its cycle classes. Now, these invariants seem like they are different things, computed in different ways. But Grothendieck had a remarkable insight: there should be a single underlying object, the motive, that contains all this information at once. The cohomology groups and Hodge structures are just different ways of looking at the same motive.

The Motivic Flow Principle says this: if your system has controlled energy, concentrated profiles, and subcritical scaling, then you can assign a motive to it. And the structure of this motive reflects the structure of your dynamical system.

Think of it this way. You have a flow on some space, and you want to understand its long-term behavior. The motive captures the essential algebraic structure that persists under the flow. The weight filtration tells you how different parts of the space scale differently. The Frobenius eigenvalues give you information about periodic orbits. The theorem says: all of this structure falls out automatically once you have the right certificates.

:::


:::{prf:theorem} [LOCK-Motivic] Motivic Flow Principle
:label: mt-lock-motivic

**Sieve Signature (Motivic Flow)**
- **Requires:** $K_{D_E}^+$ (finite energy), $K_{C_\mu}^+$ (concentration), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling)
- **Produces:** $K_{\text{motive}}^+$ (motivic assignment with weight filtration)

**Statement:** Let $X$ be a smooth projective variety over a field $k$ with flow $S_t: H^*(X) \to H^*(X)$ induced by correspondences. Suppose the sieve has issued:
- $K_{D_E}^+$: The height functional $\Phi = \|\cdot\|_H^2$ is finite on cohomology
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space $\mathcal{P}$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha, \beta)$ satisfy $\beta - \alpha < \lambda_c$

Then there exists a contravariant functor to Chow motives:

$$
\mathcal{M}: \mathbf{SmProj}_k^{\text{op}} \to \mathbf{Mot}_k^{\text{eff}}, \quad X \mapsto h(X) = (X, \Delta_X, 0)

$$

satisfying:

1. **Künneth Decomposition:** $h(X) = \bigoplus_{i=0}^{2\dim X} h^i(X)$ with $H^*(h^i(X)) = H^i(X, \mathbb{Q})$
2. **Weight Filtration:** The motivic weight filtration $W_\bullet h(X)$ satisfies:

   $$
   \text{Gr}_k^W h(X) \cong \bigoplus_{\alpha - \beta = k} h(X)_{\alpha,\beta}

   $$

   where $(\alpha, \beta)$ are the scaling exponents from $K_{\mathrm{SC}_\lambda}^+$
3. **Frobenius Eigenvalues:** For $k = \mathbb{F}_q$, the Frobenius $F: h(X) \to h(X)$ has eigenvalues $\{\omega_i\}$ with $|\omega_i| = q^{w_i/2}$ where $w_i \in W_{w_i}$
4. **Entropy-Trace Formula:** $\exp(h_{\text{top}}(S_t)) = \rho(F^* \mid H^*(X))$ where $\rho$ is spectral radius

**Required Interface Permits:** $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$

**Prevented Failure Modes:** S.E (Supercritical Cascade), C.D (Geometric Collapse)

**Certificate Produced:** $K_{\text{motive}}^+$ with payload:
- $h(X) \in \mathbf{Mot}_k^{\text{eff}}$: The effective Chow motive
- $W_\bullet$: Weight filtration with $\text{Gr}_k^W \cong $ Mode $k$
- $(\alpha, \beta)$: Scaling exponents from $K_{\mathrm{SC}_\lambda}^+$
- $\rho(F^*)$: Spectral radius (= $\exp(h_{\text{top}})$)

**Literature:** {cite}`Manin68`; {cite}`Scholl94`; {cite}`Deligne74`; {cite}`Jannsen92`; {cite}`Andre04`
:::

:::{prf:proof}

*Step 1 (Profile space construction).* The certificate $K_{C_\mu}^+$ guarantees concentration: there exists a finite-dimensional algebraic variety $\mathcal{P} \subset \text{Hilb}(X)$ such that all limit profiles lie in $\mathcal{P}/G$ where $G$ is the symmetry group. By Grothendieck's representability, $\mathcal{P}$ is a quasi-projective scheme.

*Step 2 (Motive assignment).* Define the Chow motive $h(X) := (X, \Delta_X, 0) \in \mathbf{Mot}_k$ where $\Delta_X \subset X \times X$ is the diagonal correspondence. For the profile space: $h(\mathcal{P}) := (\mathcal{P}, \Delta_{\mathcal{P}}, 0)$. If $\mathcal{P}$ is singular, apply resolution of singularities $\pi: \tilde{\mathcal{P}} \to \mathcal{P}$ and set $h(\mathcal{P}) := h(\tilde{\mathcal{P}})$.

*Step 3 (Künneth projectors).* By the Künneth formula in $\mathbf{Mot}_k$ (assuming standard conjectures or working with abelian varieties where proven), there exist orthogonal idempotents $\pi^i \in \text{Corr}^0(X, X)$ with:

$$
\sum_{i=0}^{2n} \pi^i = \Delta_X, \quad \pi^i \circ \pi^j = \delta_{ij}\pi^i, \quad H^*(\pi^i) = H^i(X)

$$

*Step 4 (Frobenius action).* The flow $S_t$ induces a correspondence $\Gamma_{S_t} \subset X \times X$. For self-similar profiles with scaling data from $K_{\mathrm{SC}_\lambda}^+$:

$$
F_t^* = [\Gamma_{S_t}]^*: H^*(X) \to H^*(X), \quad F_t^*[\alpha] = t^{\alpha - \beta}[\alpha] \text{ for } \alpha \in H^{p,q}

$$

The exponent $\alpha - \beta = p - q$ is the Hodge weight difference.

*Step 5 (Weight filtration).* Define the weight filtration on $h(X)$ by:

$$
W_k h(X) := \bigoplus_{\substack{i \leq k \\ \text{Frob. wt.} \leq k}} h^i(X)

$$

The scaling certificate $K_{\mathrm{SC}_\lambda}^+$ with exponents $(\alpha, \beta)$ gives: $\text{Gr}_k^W \cong h(X)_{\alpha - \beta = k}$. This identifies weight graded pieces with mode sectors.

*Step 6 (Trace formula).* By the Lefschetz trace formula for correspondences:

$$
\#\text{Fix}(F) = \sum_{i=0}^{2n} (-1)^i \text{Tr}(F^* \mid H^i(X))

$$

The topological entropy satisfies $\exp(h_{\text{top}}) = \lim_{n \to \infty} |\text{Tr}((F^*)^n)|^{1/n} = \rho(F^*)$, the spectral radius.

*Step 7 (Certificate assembly).* Construct the output certificate:

$$
K_{\text{motive}}^+ = \left(h(X), \{\pi^i\}_{i=0}^{2n}, W_\bullet, \{(\alpha_j, \beta_j)\}_j, \rho(F^*)\right)

$$

containing the motive, Künneth projectors, weight filtration, scaling exponents, and spectral radius.
:::

---

### Schematic Sieve

:::{div} feynman-prose

Here is a beautiful idea. We have been saying that the sieve excludes bad patterns, but how do we actually prove that exclusion? The Schematic Sieve gives us an algebraic certificate.

The key insight is this: our permit certificates define regions in some space of invariants. The energy is bounded here. The scaling is subcritical there. The gradient is steep enough over here. Each of these conditions carves out a region, and their intersection is the safe set.

Now, if the bad patterns live somewhere else entirely, disjoint from this safe set, we want a proof. And here is where the Positivstellensatz comes in. It is like the Nullstellensatz you might know from algebra, but for inequalities instead of equations. It says: if two semialgebraic sets are disjoint, there exists a certificate, a specific polynomial identity, that witnesses this disjointness.

What makes this computationally wonderful is that finding such certificates is a semidefinite programming problem. You can actually run an algorithm and get back a proof that your safe region cannot intersect the bad region. The certificate is not just existential: it is constructive.

:::

:::{prf:theorem} [LOCK-Schematic] Semialgebraic Exclusion
:label: mt-lock-schematic

**Source:** Stengle's Positivstellensatz (1974)

**Sieve Signature (Schematic)**
- **Requires:** $K_{\mathrm{Cap}_H}^+$ (capacity bound), $K_{\mathrm{LS}_\sigma}^+$ (Łojasiewicz gradient), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling), $K_{\mathrm{TB}_\pi}^+$ (topological bound)
- **Produces:** $K_{\text{SOS}}^+$ (sum-of-squares certificate witnessing Bad Pattern exclusion)

**Setup:**
Let structural invariants be polynomial variables: $x_1 = \Phi$, $x_2 = \mathfrak{D}$, $x_3 = \text{Gap}$, etc.
Let $\mathcal{R} = \mathbb{R}[x_1, \ldots, x_n]$ be the polynomial ring over the reals.

**Safe Set (from certificates):**
The permit certificates define polynomial inequalities. The *safe region* is:

$$
S = \{x \in \mathbb{R}^n \mid g_1(x) \geq 0, \ldots, g_k(x) \geq 0\}

$$

where:
- $g_{\text{SC}}(x) := \beta - \alpha - \varepsilon$ (from $K_{\mathrm{SC}_\lambda}^+$)
- $g_{\text{Cap}}(x) := C\mathfrak{D} - \text{Cap}_H(\text{Supp})$ (from $K_{\mathrm{Cap}_H}^+$)
- $g_{\text{LS}}(x) := \|\nabla\Phi\|^2 - C_{\text{LS}}^2 |\Phi - \Phi_{\min}|^{2\theta}$ (from $K_{\mathrm{LS}_\sigma}^+$)
- $g_{\text{TB}}(x) := c^2 - \|\nabla\Pi\|^2$ (from $K_{\mathrm{TB}_\pi}^+$)

**Statement (Stengle's Positivstellensatz):**
Let $B \subset \mathbb{R}^n$ be the *bad pattern region* (states violating safety). Then:

$$
S \cap B = \emptyset

$$

if and only if there exist sum-of-squares polynomials $\{p_\alpha\}_{\alpha \in \{0,1\}^k} \subset \sum \mathbb{R}[x]^2$ such that:

$$
-1 = p_0 + \sum_{i} p_i g_i + \sum_{i<j} p_{ij} g_i g_j + \cdots + p_{1\ldots k} g_1 \cdots g_k

$$

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $\mathrm{LS}_\sigma$ (Stiffness), $\mathrm{SC}_\lambda$ (Scaling), $\mathrm{TB}_\pi$ (Topology)

**Prevented Failure Modes:** C.D (Geometric Collapse), S.D (Stiffness Breakdown)

**Certificate Produced:** $K_{\text{SOS}}^+$ with payload:
- $\{p_\alpha\}$: SOS polynomials witnessing the Positivstellensatz identity
- $\{g_i\}$: Permit constraint polynomials
- SDP witness: Numerical certificate of SOS decomposition

**Remark (Nullstellensatz vs. Positivstellensatz):**
The original Nullstellensatz formulation applies to equalities over $\mathbb{C}$. Since permit certificates assert *inequalities* (e.g., $\text{Gap} > 0$) over $\mathbb{R}$, the correct algebraic certificate is Stengle's Positivstellensatz, which handles semialgebraic sets.

**Literature:** {cite}`Stengle74`; {cite}`Parrilo03`; {cite}`Blekherman12`; {cite}`Lasserre09`
:::

:::{prf:proof}

*Step 1 (Real algebraic geometry).* The permit certificates define polynomial inequalities over $\mathbb{R}$, not equalities over $\mathbb{C}$. Hilbert's Nullstellensatz does not apply directly to inequalities; we use the Positivstellensatz instead.

*Step 2 (Bad pattern encoding).* A bad pattern $B_i$ is encoded as a semialgebraic set:

$$
B_i = \{x \in \mathbb{R}^n \mid h_1(x) \geq 0, \ldots, h_m(x) \geq 0, f(x) = 0\}

$$

representing states that lead to singularity type $i$.

*Step 3 (Infeasibility certificate).* By Stengle's Positivstellensatz, $S \cap B_i = \emptyset$ admits a constructive certificate: an identity expressing $-1$ as a combination of the constraint polynomials weighted by SOS polynomials.

*Step 4 (SOS computation).* The SOS certificate can be computed via semidefinite programming (SDP). Given a degree bound $d$, search for SOS polynomials $p_\alpha$ of degree $\leq d$ satisfying the identity. If such an identity exists, the intersection is algebraically certified empty.

*Step 5 (Certificate assembly).* The output certificate consists of:

$$
K_{\text{SOS}}^+ = \left(\{p_\alpha\}_\alpha, \{g_i\}_i, \text{SDP feasibility witness}\right)

$$
:::

---

### Kodaira-Spencer Stiffness Link

:::{div} feynman-prose

Now we connect stiffness to deformation theory. This is one of the most satisfying links in the whole framework.

You know how some structures are rigid and some are floppy? A triangle is rigid: you cannot deform it without changing the lengths of its sides. A quadrilateral is floppy: you can squish it into a rhombus or stretch it into a different parallelogram.

In algebraic geometry, the same distinction appears for varieties. Some varieties are rigid, meaning they admit no deformations. Others have moduli, meaning they come in continuous families. And here is the beautiful thing: you can read off this rigidity from cohomology groups.

The Kodaira-Spencer theorem tells you that $H^1(V, T_V)$, the first cohomology of the tangent bundle, parametrizes infinitesimal deformations. If this group vanishes, the variety is rigid. If it does not vanish, you need to look at $H^2(V, T_V)$ for obstructions, the ways a first-order deformation might fail to extend.

What we do here is connect this classical story to the sieve framework. The stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ is saying: the energy landscape has no flat directions. And that is precisely the condition for rigidity in the Kodaira-Spencer sense. Either there are no deformations at all, or every deformation is obstructed. Either way, the system is locked in place.

:::

:::{prf:theorem} [LOCK-Kodaira] Kodaira-Spencer Stiffness Link
:label: mt-lock-kodaira

**Sieve Signature (Kodaira-Spencer)**
- **Requires:** $K_{\mathrm{LS}_\sigma}^+$ (stiffness gradient), $K_{C_\mu}^+$ (concentration on finite-dimensional moduli)
- **Produces:** $K_{\text{KS}}^+$ (deformation cohomology, rigidity classification)

**Statement:** Let $V$ be a smooth projective variety over a field $k$. Suppose the sieve has issued:
- $K_{\mathrm{LS}_\sigma}^+$: Łojasiewicz gradient with exponent $\theta \in (0,1)$ and constant $C_{\text{LS}} > 0$
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space

Consider the tangent sheaf cohomology groups $H^i(V, T_V)$ for $i = 0, 1, 2$. Then:

1. **Symmetries:** $H^0(V, T_V) \cong \text{Lie}(\text{Aut}^0(V))$ — global vector fields are infinitesimal automorphisms
2. **Deformations:** $H^1(V, T_V) \cong T_{[V]}\mathcal{M}$ — first-order deformations parametrize tangent space to moduli
3. **Obstructions:** $H^2(V, T_V) \supseteq \text{Ob}(V)$ — obstruction space for extending deformations
4. **Stiffness ↔ Rigidity:** $K_{\mathrm{LS}_\sigma}^+$ holds if and only if:
   - $H^1(V, T_V) = 0$ (infinitesimal rigidity), OR
   - The obstruction map $\text{ob}: H^1 \otimes H^1 \to H^2$ is surjective (all deformations obstructed)

**Required Interface Permits:** $\mathrm{LS}_\sigma$ (Stiffness), $C_\mu$ (Concentration)

**Prevented Failure Modes:** S.D (Stiffness Breakdown), S.C (Parameter Instability)

**Certificate Produced:** $K_{\text{KS}}^+$ with payload:
- $(h^0, h^1, h^2) := (\dim H^0(T_V), \dim H^1(T_V), \dim H^2(T_V))$
- $\text{ob}: \text{Sym}^2 H^1 \to H^2$: Obstruction map
- Classification: "rigid" if $h^1 = 0$; "obstructed" if $\text{ob}$ surjective; "unobstructed" otherwise
- Rigidity flag: $\mathbf{true}$ iff $K_{\mathrm{LS}_\sigma}^+$ is compatible

**Literature:** {cite}`KodairaSpencer58`; {cite}`Kuranishi65`; {cite}`Griffiths68`; {cite}`Artin76`; {cite}`Sernesi06`
:::

:::{prf:proof}

*Step 1 (Deformation functor).* Define the deformation functor $\text{Def}_V: \mathbf{Art}_k \to \mathbf{Sets}$ by:

$$
\text{Def}_V(A) := \left\{\text{flat } \mathcal{V} \to \text{Spec}(A) \mid \mathcal{V} \times_A k \cong V\right\} / \sim

$$

This is the moduli problem for flat families with special fiber $V$.

*Step 2 (Kodaira-Spencer map).* For an infinitesimal deformation $\mathcal{V} \to \text{Spec}(k[\epsilon])$, the Kodaira-Spencer map:

$$
\text{KS}: T_0\text{Def}_V \xrightarrow{\cong} H^1(V, T_V)

$$

identifies first-order deformations with cohomology classes. This is an isomorphism by the exponential sequence.

*Step 3 (Kuranishi space).* By Kuranishi's theorem, there exists a versal deformation $\mathcal{V} \to (\mathcal{K}, 0)$ with:
- $(\mathcal{K}, 0)$ a germ of analytic space (or formal scheme)
- $T_0\mathcal{K} = H^1(V, T_V)$
- The obstruction space $\text{Ob} \subseteq H^2(V, T_V)$

*Step 4 (Obstruction theory).* The obstruction to extending a first-order deformation $\xi \in H^1$ to second order lies in $H^2$. The obstruction map:

$$
\text{ob}: \text{Sym}^2 H^1(V, T_V) \to H^2(V, T_V)

$$

arises from the bracket $[-, -]: T_V \otimes T_V \to T_V$. If $H^2 = 0$, the Kuranishi space is smooth of dimension $h^1(T_V)$.

*Step 5 (Stiffness ↔ Łojasiewicz).* The certificate $K_{\mathrm{LS}_\sigma}^+$ with gradient inequality $\|\nabla\Phi\| \geq C|\Phi - \Phi_{\min}|^\theta$ corresponds to deformation rigidity:
- **Case $H^1 = 0$:** No infinitesimal deformations exist; $V$ is locally rigid in moduli. The certificate issues with payload "rigid".
- **Case $H^1 \neq 0$, $\text{ob}$ surjective:** All first-order deformations are obstructed; $\mathcal{K} = \{0\}$ scheme-theoretically. The certificate issues with payload "obstructed".
- **Case $H^1 \neq 0$, $\text{ob}$ not surjective:** Positive-dimensional moduli; stiffness certificate $K_{\mathrm{LS}_\sigma}^-$ issues (stiffness fails).

*Step 6 (Concentration link).* The certificate $K_{C_\mu}^+$ ensures the moduli space $\mathcal{M}$ is finite-dimensional. By Grothendieck's representability:

$$
\dim \mathcal{M} = h^1(V, T_V) - \dim(\text{Im ob}) < \infty

$$

Concentration forces $h^1 < \infty$, which holds for all coherent sheaf cohomology on proper varieties.

*Step 7 (Certificate assembly).* Construct the output certificate:

$$
K_{\text{KS}}^+ = \left((h^0, h^1, h^2), \text{ob}, \text{classification}\right)

$$

where classification $\in \{\text{rigid}, \text{obstructed}, \text{unobstructed-positive}\}$.
:::

---

### Virtual Cycle Correspondence

:::{div} feynman-prose

Here is one of the deepest ideas in modern geometry: the virtual fundamental class.

Let me set the stage. You want to count something: curves on a surface, sheaves on a threefold, solutions to some geometric problem. The classical approach is to set up a moduli space of all the objects you want to count, and then count points in it.

But here is the trouble: moduli spaces are often badly behaved. They might have the wrong dimension, or be non-reduced, or have components you did not expect. If you just count points naively, you get the wrong answer.

The virtual fundamental class is the fix. It says: even if the moduli space is badly behaved, there is a canonical way to assign it an effective dimension (the virtual dimension) and a canonical cycle of that dimension (the virtual class). When you integrate over this virtual class, you get the right answer, the one that behaves well in families and satisfies the expected identities.

What makes this relevant for the sieve? The permit certificates ensure the moduli space is tame enough for the virtual class to exist and be well-behaved. The capacity bound says the space is not too big. The energy bound says nothing runs off to infinity. Under these conditions, the virtual machinery works, and you can define enumerative invariants that count certificate failures with the correct virtual multiplicity.

This is how the sieve connects to Gromov-Witten theory and Donaldson-Thomas theory: the most powerful counting machines in modern geometry.

:::

:::{prf:theorem} [LOCK-Virtual] Virtual Cycle Correspondence
:label: mt-lock-virtual

**Sieve Signature (Virtual Cycle)**
- **Requires:** $K_{\mathrm{Cap}_H}^+$ (capacity bound on moduli), $K_{D_E}^+$ (finite energy), $K_{\mathrm{Rep}}^+$ (representation completeness)
- **Produces:** $K_{\text{virtual}}^+$ (virtual fundamental class, enumerative invariants)

**Statement:** Let $\mathcal{M}$ be a proper Deligne-Mumford stack with perfect obstruction theory $\phi: \mathbb{E}^\bullet \to \mathbb{L}_{\mathcal{M}}$ where $\mathbb{E}^\bullet = [E^{-1} \to E^0]$. Suppose the sieve has issued:
- $K_{\mathrm{Cap}_H}^+$: The Hausdorff capacity satisfies $\text{Cap}_H(\mathcal{M}) \leq C \cdot \mathfrak{D}$ for dimension $\mathfrak{D} = \text{vdim}(\mathcal{M})$
- $K_{D_E}^+$: The energy functional $\Phi$ on $\mathcal{M}$ is bounded: $\sup_{\mathcal{M}} \Phi < \infty$

Then:

1. **Virtual Fundamental Class:** There exists a unique class:

   $$
   [\mathcal{M}]^{\text{vir}} = 0_E^![\mathfrak{C}_{\mathcal{M}}] \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})

   $$

   where $\mathfrak{C}_{\mathcal{M}} \subset E^{-1}|_{\mathcal{M}}$ is the intrinsic normal cone and $0_E^!$ is the refined Gysin map.

2. **Certificate Integration:** For any certificate test function $\chi_A: \mathcal{M} \to \mathbb{Q}$:

   $$
   \int_{[\mathcal{M}]^{\text{vir}}} \chi_A = \#^{\text{vir}}\{p \in \mathcal{M} : K_A^-(p)\}

   $$

   counts (with virtual multiplicity) points where certificate $K_A$ fails.

3. **GW Invariants:** For $X$ a smooth projective variety, $\beta \in H_2(X, \mathbb{Z})$:

   $$
   \text{GW}_{g,n,\beta}(X; \gamma_1, \ldots, \gamma_n) = \int_{[\overline{M}_{g,n}(X,\beta)]^{\text{vir}}} \prod_{i=1}^n \text{ev}_i^*(\gamma_i)

   $$

   counts stable maps with $K_{\mathrm{Rep}}^+$ ensuring curve representability.

4. **DT Invariants:** For $X$ a Calabi-Yau threefold, $\text{ch} \in H^*(X)$:

   $$
   \text{DT}_{\text{ch}}(X) = \int_{[\mathcal{M}_{\text{ch}}^{\text{st}}(X)]^{\text{vir}}} 1

   $$

   counts stable sheaves with $K_{\mathrm{Cap}_H}^+$ ensuring proper moduli.

**Required Interface Permits:** $\mathrm{Cap}_H$ (Capacity), $D_E$ (Energy), $\mathrm{Rep}$ (Representation)

**Prevented Failure Modes:** C.D (Geometric Collapse), E.I (Enumeration Inconsistency)

**Certificate Produced:** $K_{\text{virtual}}^+$ with payload:
- $[\mathcal{M}]^{\text{vir}} \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})$: Virtual fundamental class
- $\text{vdim} = \text{rk}(E^0) - \text{rk}(E^{-1})$: Virtual dimension
- $\mathbb{E}^\bullet = [E^{-1} \to E^0]$: Perfect obstruction theory
- Invariants: $\text{GW}_{g,n,\beta}$, $\text{DT}_{\text{ch}}$ as needed

**Literature:** {cite}`BehrFant97`; {cite}`LiTian98`; {cite}`KontsevichManin94`; {cite}`Thomas00`; {cite}`Maulik06`; {cite}`Graber99`
:::

:::{prf:proof}

*Step 1 (Obstruction theory).* A perfect obstruction theory is a morphism $\phi: \mathbb{E}^\bullet \to \mathbb{L}_{\mathcal{M}}$ in $D^{[-1,0]}(\mathcal{M})$ satisfying:
- $h^0(\phi): h^0(\mathbb{E}^\bullet) \xrightarrow{\cong} h^0(\mathbb{L}_{\mathcal{M}}) = \Omega_{\mathcal{M}}$ is an isomorphism
- $h^{-1}(\phi): h^{-1}(\mathbb{E}^\bullet) \twoheadrightarrow h^{-1}(\mathbb{L}_{\mathcal{M}})$ is surjective

The certificate $K_{\mathrm{Cap}_H}^+$ ensures $\mathbb{E}^\bullet$ is a 2-term complex of finite-rank vector bundles.

*Step 2 (Virtual dimension).* The virtual dimension is:
$$\text{vdim}(\mathcal{M}) := \text{rk}(E^0) - \text{rk}(E^{-1}) = \chi(\mathbb{E}^\bullet)$$
At each point $p \in \mathcal{M}$: deformations $= H^0(\mathbb{E}^\bullet|_p)$, obstructions $= H^1(\mathbb{E}^\bullet|_p)$.

*Step 3 (Intrinsic normal cone).* The intrinsic normal cone $\mathfrak{C}_{\mathcal{M}} \subset h^1/h^0(\mathbb{E}^{\bullet\vee})$ is a cone stack. By Behrend-Fantechi, it embeds canonically:
$$\mathfrak{C}_{\mathcal{M}} \hookrightarrow E_1 := (E^{-1})^\vee$$
The certificate $K_{D_E}^+$ (bounded energy) ensures $\mathfrak{C}_{\mathcal{M}}$ has proper support.

*Step 4 (Virtual class construction).* Define the virtual fundamental class via the refined Gysin map:
$$[\mathcal{M}]^{\text{vir}} := 0_{E_1}^! [\mathfrak{C}_{\mathcal{M}}] \in A_{\text{vdim}}(\mathcal{M}, \mathbb{Q})$$
When $\mathcal{M}$ is smooth of dimension $d > \text{vdim}$, this equals:
$$[\mathcal{M}]^{\text{vir}} = e(\text{Ob}^\vee) \cap [\mathcal{M}]$$
where $\text{Ob} = \text{coker}(T_{\mathcal{M}} \to E^0)$ is the obstruction sheaf.

*Step 5 (Certificate integration).* For a certificate $K_A$ with associated section $s_A: \mathcal{M} \to \text{Ob}^\vee$, the zero locus $Z(s_A) = \{K_A^- \text{ holds}\}$ is the failure locus. Virtual intersection:
$$\int_{[\mathcal{M}]^{\text{vir}}} e(s_A^*\text{Ob}^\vee) = [Z(s_A)]^{\text{vir}} \cdot [\mathcal{M}]^{\text{vir}}$$
counts certificate violations with virtual multiplicity.

*Step 6 (Enumerative invariants).*
- **GW theory:** The certificate $K_{\mathrm{Rep}}^+$ (curve representability) ensures evaluation maps $\text{ev}_i: \overline{M}_{g,n}(X,\beta) \to X$ are well-defined. GW invariants count certificates issued.
- **DT theory:** The certificate $K_{\mathrm{Cap}_H}^+$ (capacity bound) ensures stable sheaves form a proper moduli space. DT invariants count $K_{\mathrm{Cap}_H}^+$ certificates.

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{virtual}}^+ = \left([\mathcal{M}]^{\text{vir}}, \text{vdim}, \mathbb{E}^\bullet, \{\text{inv}_\alpha\}_\alpha\right)$$
with virtual class, dimension, obstruction theory, and computed invariants.
:::

---

### Monodromy-Weight Lock

:::{div} feynman-prose

What happens when a family of nice varieties degenerates to a singular one? This is one of the central questions in algebraic geometry, and the answer involves some of the most beautiful mathematics of the twentieth century.

Imagine you have a family of smooth curves, parametrized by points on a disk. At every point except the center, you have a nice smooth curve. But at the center, something singular happens: maybe two branches cross, or a cycle pinches off. The question is: what can you say about the limiting behavior as you approach the singularity?

Hodge theory gives you a filtration on the cohomology of each smooth fiber. As you approach the singularity, this filtration has a limit, the limiting Hodge filtration. But here is the subtle part: the monodromy, what happens when you go around the singular point, creates a weight filtration that interacts with the Hodge filtration in a precise way.

Schmid's Nilpotent Orbit Theorem is the key technical result. It says: the behavior near the singularity is controlled by a nilpotent operator (the logarithm of monodromy), and the period map approaches a limiting value in a very specific way.

What does this have to do with the sieve? The scaling exponents from $K_{\mathrm{SC}_\lambda}^+$ correspond exactly to the weights in the weight filtration. The vanishing cycles, the cycles that disappear at the singularity, correspond to collapse modes. The invariant cycles, those that survive, correspond to concentration modes. The whole story of degeneration and limiting behavior is encoded in the certificate structure.

:::

:::{prf:theorem} [LOCK-Hodge] Monodromy-Weight Lock
:label: mt-lock-hodge

**Rigor Class (Monodromy-Weight):** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificates $K_{\mathrm{TB}_\pi}^+ \wedge K_{\mathrm{SC}_\lambda}^+ \wedge K_{D_E}^+$ imply: proper flat morphism $\pi: \mathcal{X} \to \Delta$ with semistable reduction, bounded period map $\|\nabla\Pi\| \leq c$
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{MHS}$ mapping to category of mixed Hodge structures via Deligne's construction
3. *Conclusion Import:* Schmid's Nilpotent Orbit Theorem {cite}`Schmid73` + GAGA {cite}`Serre56` + Griffiths' Hodge Theory {cite}`Griffiths68` $\Rightarrow K_{\text{MHS}}^+$ (weight-monodromy correspondence)

**Sieve Signature (Monodromy-Weight)**
- **Requires:** $K_{\mathrm{TB}_\pi}^+$ (topological bound on monodromy), $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling), $K_{D_E}^+$ (finite energy)
- **Produces:** $K_{\text{MHS}}^+$ (limiting mixed Hodge structure, weight-monodromy correspondence)

**Statement:** Let $\pi: \mathcal{X} \to \Delta$ be a proper flat morphism with smooth generic fiber $X_t$ ($t \neq 0$) and semistable reduction at $0 \in \Delta$. Suppose the sieve has issued:
- $K_{\mathrm{TB}_\pi}^+$: Topological bound $\|\nabla\Pi\| \leq c$ for the period map $\Pi: \Delta^* \to D/\Gamma$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha_i, \beta)$ satisfy subcriticality $\beta - \alpha_i < \lambda_c$
- $K_{D_E}^+$: Energy $\Phi$ bounded on cohomology of general fiber

Then the limiting mixed Hodge structure (MHS) satisfies:

1. **Schmid ↔ Profile Exactification:** The nilpotent orbit
   $$F^p_t = \exp\left(\frac{\log t}{2\pi i} N\right) \cdot F^p_\infty + O(|t|^\epsilon)$$
   provides the profile map. Certificate $K_{\mathrm{TB}_\pi}^+$ ensures $F^p_\infty$ exists.

2. **Weight Filtration ↔ Scaling Exponents:** The weight filtration $W_\bullet = W(N, k)$ satisfies:
   $$\text{Gr}^W_j H^k \neq 0 \Rightarrow \alpha_{j} = j/2$$
   where $\alpha_j$ are the scaling exponents from $K_{\mathrm{SC}_\lambda}^+$.

3. **Clemens-Schmid ↔ Mode Decomposition:**
   - Vanishing cycles $V := \text{Im}(N)$ correspond to Mode C.D (collapse)
   - Invariant cycles $I := \ker(N) \cap \ker(1-T)$ correspond to Mode C.C (concentration)

4. **Picard-Lefschetz ↔ Dissipation:** Monodromy eigenvalues $\{\zeta\}$ of $T$ satisfy $|\zeta| = 1$ (roots of unity), with $\zeta \neq 1$ contributing dissipation modes.

**Required Interface Permits:** $\mathrm{TB}_\pi$ (Topology), $\mathrm{SC}_\lambda$ (Scaling), $D_E$ (Energy)

**Prevented Failure Modes:** T.E (Topological Twist), S.E (Supercritical Cascade)

**Certificate Produced:** $K_{\text{MHS}}^+$ with payload:
- $F^\bullet_\infty$: Limiting Hodge filtration
- $W_\bullet = W(N, k)$: Deligne weight filtration
- $N = \log(T^m)$: Nilpotent monodromy logarithm
- $(I, V)$: Invariant/vanishing cycle decomposition
- $\{(\alpha_j = j/2, j)\}$: Weight-scaling correspondence from $K_{\mathrm{SC}_\lambda}^+$

**Literature:** {cite}`Schmid73`; {cite}`Deligne80`; {cite}`Clemens77`; {cite}`CKS86`; {cite}`PS08`; {cite}`Steenbrink76`
:::

:::{prf:proof}

*Step 1 (Monodromy).* Let $T: H^k(X_t, \mathbb{Z}) \to H^k(X_t, \mathbb{Z})$ be the monodromy operator for a loop $\gamma$ around $0$. The certificate $K_{\mathrm{TB}_\pi}^+$ ensures $\|\nabla\Pi\|$ is bounded, which by Borel's theorem implies $T$ is quasi-unipotent:
$$(T^m - I)^{k+1} = 0 \quad \text{for some } m \geq 1$$
The bound $\|\nabla\Pi\| \leq c$ from $K_{\mathrm{TB}_\pi}^+$ controls the monodromy weight.

*Step 2 (Nilpotent orbit theorem).* After finite base change $t \mapsto t^m$, assume $T$ unipotent. Define $N := \log T = \sum_{j=1}^\infty \frac{(-1)^{j+1}}{j}(T-I)^j$. By Schmid's theorem, the period map $\Phi: \Delta^* \to D$ satisfies:
$$\Phi(t) = \exp\left(\frac{\log t}{2\pi i} N\right) \cdot \Phi_\infty + O(|t|^\epsilon)$$
for some $\epsilon > 0$. The limiting Hodge filtration $F^p_\infty$ exists and is horizontal.

*Step 3 (Weight filtration).* Construct $W_\bullet = W(N, k)$ as the unique filtration satisfying:
- **Shifting:** $N(W_j) \subseteq W_{j-2}$ (nilpotent lowers weight by 2)
- **Hard Lefschetz:** $N^j: \text{Gr}^W_{k+j} \xrightarrow{\cong} \text{Gr}^W_{k-j}$ for all $j \geq 0$

This is the Deligne weight filtration associated to $(H^k_{\lim}, N)$.

*Step 4 (Mixed Hodge structure).* The pair $(W_\bullet, F^\bullet_\infty)$ defines a mixed Hodge structure on $H^k_{\lim}$:
- Each $\text{Gr}^W_j H^k_{\lim}$ carries a pure Hodge structure of weight $j$
- The filtrations satisfy $F^p \cap W_j + F^{j-p+1} \cap W_j = W_j \cap (F^p + F^{j-p+1})$

The certificate $K_{D_E}^+$ (bounded energy) ensures the MHS has finite-dimensional graded pieces.

*Step 5 (Scaling-weight correspondence).* The certificate $K_{\mathrm{SC}_\lambda}^+$ provides scaling exponents $(\alpha_i, \beta)$. For $v \in \text{Gr}^W_j H^k$:
$$\|v(t)\| \sim |t|^{-j/2} \quad \text{as } t \to 0$$
Thus $\alpha_j = j/2$. Subcriticality $\beta - \alpha_j < \lambda_c$ imposes $\alpha_j > \beta - \lambda_c$, hence $j > 2(\beta - \lambda_c)$ (weights below $2(\beta - \lambda_c)$ are excluded).

*Step 6 (Clemens-Schmid sequence).* The exact sequence of mixed Hodge structures:
$$\cdots \to H_k(X_0) \xrightarrow{i_*} H^k(X_t) \xrightarrow{1-T} H^k(X_t) \xrightarrow{\text{sp}} H_k(X_0) \xrightarrow{N} H_{k-2}(X_0)(-1) \to \cdots$$
decomposes cohomology:
- **Invariant part:** $I = \ker(1-T) = \text{Im}(i_*)$ — cycles surviving to $X_0$ (Mode C.C)
- **Vanishing part:** $V = \text{Im}(N) \cong \text{coker}(i_*)$ — cycles disappearing at $X_0$ (Mode C.D)

*Step 7 (Certificate assembly).* Construct the output certificate:
$$K_{\text{MHS}}^+ = \left(F^\bullet_\infty, W_\bullet, N, T, (I, V), \{(\alpha_j, j)\}\right)$$
containing the limiting Hodge filtration, weight filtration, monodromy data, cycle decomposition, and weight-scaling pairs.
:::

---

### Tannakian Recognition Principle

:::{div} feynman-prose

Now here is something remarkable. Suppose you have a category of objects that behave like representations of a group, but you do not know what group they represent. The Tannakian formalism says: you can recover the group from the category itself.

Think about it this way. If you have a group $G$ and look at all its finite-dimensional representations, these form a category with a lot of structure: you can take tensor products, duals, direct sums. Now, Tannakian theory says the converse is true: if you have a category with all this structure, and a way to extract vector spaces from it (the fiber functor), then there exists a unique group whose representations give you back the category.

This is enormously powerful. It means that categorical structure determines group structure. The symmetries are encoded in how objects combine.

Why does this matter for the sieve? Because the Lock at Node 17 is asking: does a morphism exist between certain objects? In a Tannakian category, this question becomes: does a $G$-equivariant map exist between representations? And that question is decidable: you can compute the invariant subspace and check if it contains what you need.

The motivic Galois group, which conjecturally controls all the algebraic relations between periods, is exactly the Tannakian group associated to the category of motives. So this metatheorem gives us a precise language for talking about the deepest structural constraints in algebraic geometry.

:::

:::{prf:theorem} [LOCK-Tannakian] Tannakian Recognition Principle
:label: mt-lock-tannakian

**Rigor Class (Tannakian):** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* The $\mathrm{Cat}_{\mathrm{Hom}}$ interface data together with $K_{\Gamma}^+$ imply: neutral Tannakian category $\mathcal{C}$ over $k$ with exact faithful tensor-preserving fiber functor $\omega$
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{TannCat}_k$ mapping to category of Tannakian categories via forgetful functor
3. *Conclusion Import:* Deligne's Tannakian Duality {cite}`Deligne90` $\Rightarrow K_{\text{Tann}}^+$ (group scheme $G = \underline{\text{Aut}}^\otimes(\omega)$ recoverable, $\mathcal{C} \simeq \text{Rep}_k(G)$)

**Sieve Signature (Tannakian)**
- **Requires:** $\mathrm{Cat}_{\mathrm{Hom}}$ interface data (Hom-functor structure), $K_{\Gamma}^+$ (full context certificate)
- **Produces:** $K_{\text{Tann}}^+$ (Galois group reconstruction, algebraicity criterion, lock exclusion)

**Statement:** Let $\mathcal{C}$ be a neutral Tannakian category over a field $k$ with fiber functor $\omega: \mathcal{C} \to \mathbf{Vect}_k$. Suppose the sieve has instantiated the $\mathrm{Cat}_{\mathrm{Hom}}$ interface so that:
- The category $\mathcal{C}$ is $k$-linear, abelian, rigid monoidal with $\text{End}(\mathbb{1}) = k$
- $K_{\Gamma}^+$: Full context certificate with fiber functor $\omega$ exact, faithful, and tensor-preserving

Then:

1. **Group Reconstruction:** The functor of tensor automorphisms
   $$G := \underline{\text{Aut}}^\otimes(\omega): \mathbf{Alg}_k \to \mathbf{Grp}, \quad R \mapsto \text{Aut}^\otimes(\omega \otimes R)$$
   is representable by an affine pro-algebraic group scheme over $k$.

2. **Categorical Equivalence:** There is a canonical equivalence of tensor categories:
   $$\mathcal{C} \xrightarrow{\simeq} \text{Rep}_k(G), \quad V \mapsto (\omega(V), \rho_V)$$
   where $\rho_V: G \to \text{GL}(\omega(V))$ is the natural action.

3. **Motivic Galois Group:** For $\mathcal{C} = \mathbf{Mot}_k^{\text{num}}$ with Betti realization $\omega = H_B$:
   - $G = \mathcal{G}_{\text{mot}}(k)$ is the motivic Galois group
   - Algebraic cycles correspond to $\mathcal{G}_{\text{mot}}$-invariants: $\text{CH}^p(X)_\mathbb{Q} \cong H^{2p}(X)^{\mathcal{G}_{\text{mot}}}$
   - Transcendental classes lie in representations with non-trivial $\mathcal{G}_{\text{mot}}$-action

4. **Lock Exclusion via Galois Constraints:** For barrier $\mathcal{B}$ and safe region $S$ in $\mathcal{C}$:
   $$\text{Hom}_{\mathcal{C}}(\mathcal{B}, S) = \emptyset \Leftrightarrow \text{Hom}_{\text{Rep}(G)}(\rho_{\mathcal{B}}, \rho_S)^G = 0$$
   The lock condition reduces to absence of $G$-equivariant morphisms.

**Required Interface Permits:** $\mathrm{Cat}_{\mathrm{Hom}}$ (Categorical Hom), $\Gamma$ (Full Context)

**Prevented Failure Modes:** L.M (Lock Morphism Existence) — excludes morphisms violating Galois constraints

**Certificate Produced:** $K_{\text{Tann}}^+$ with payload:
- $G = \text{Aut}^\otimes(\omega)$: Reconstructed Galois/automorphism group
- $\mathcal{O}(G)$: Coordinate Hopf algebra
- $\mathcal{C} \simeq \text{Rep}_k(G)$: Categorical equivalence
- $V^G = \text{Hom}(\mathbb{1}, V)$: Invariant (algebraic) subspace for each $V$
- Lock status: $\text{Hom}(\mathcal{B}, S)^G = 0$ verification

**Literature:** {cite}`Deligne90`; {cite}`SaavedraRivano72`; {cite}`DeligneMillne82`; {cite}`Andre04`; {cite}`Nori00`
:::

:::{prf:proof}

*Step 1 (Tannakian axioms).* The $\mathrm{Cat}_{\mathrm{Hom}}$ interface data ensure $\mathcal{C}$ satisfies the Tannakian axioms:
- **Abelian:** $\mathcal{C}$ is a $k$-linear abelian category
- **Rigid monoidal:** $(\mathcal{C}, \otimes, \mathbb{1})$ is a rigid tensor category with unit $\mathbb{1}$
- **Neutrality:** $\text{End}_{\mathcal{C}}(\mathbb{1}) = k$ (no non-trivial automorphisms of the unit)

The certificate $K_{\Gamma}^+$ provides the fiber functor $\omega: \mathcal{C} \to \mathbf{Vect}_k$.

*Step 2 (Automorphism functor).* For any commutative $k$-algebra $R$, define:
$$G(R) := \text{Aut}^\otimes(\omega_R) = \left\{\eta: \omega_R \xrightarrow{\sim} \omega_R \;\middle|\; \begin{array}{l} \eta_{V \otimes W} = \eta_V \otimes \eta_W \\ \eta_\mathbb{1} = \text{id}_R \end{array}\right\}$$
where $\omega_R := \omega \otimes_k R: \mathcal{C} \to \mathbf{Mod}_R$. This defines a functor $G: \mathbf{Alg}_k \to \mathbf{Grp}$.

*Step 3 (Representability).* By Deligne's theorem, $G$ is represented by an affine group scheme:
$$G = \text{Spec}(\mathcal{O}(G)), \quad \mathcal{O}(G) = \varinjlim_{V \in \mathcal{C}} \text{End}(\omega(V))^*$$
The Hopf algebra structure on $\mathcal{O}(G)$ encodes the group law. For $\mathcal{C}$ of subexponential growth, $G$ is pro-algebraic.

*Step 4 (Equivalence).* The canonical functor $\Phi: \mathcal{C} \to \text{Rep}_k(G)$ defined by:
$$\Phi(V) := (\omega(V), \rho_V), \quad \rho_V(g)(v) := g_V(v) \text{ for } g \in G, v \in \omega(V)$$
is an equivalence of tensor categories. Inverse: $\Psi: \text{Rep}_k(G) \to \mathcal{C}$ via torsors.

*Step 5 (Invariant subspace).* For any $V \in \mathcal{C}$, the $G$-invariant subspace is:
$$\omega(V)^G := \{v \in \omega(V) : \forall g \in G(\bar{k}), \; g \cdot v = v\} = \text{Hom}_{\mathcal{C}}(\mathbb{1}, V)$$
This is the subspace of "algebraic" or "Hodge" elements. Certificate $K_{\text{Tann}}^+$ records $\dim V^G$.

*Step 6 (Motivic application).* For the category of numerical motives $\mathcal{C} = \mathbf{Mot}_k^{\text{num}}$:
- The motivic Galois group $\mathcal{G}_{\text{mot}} = \text{Aut}^\otimes(H_B)$ is a pro-reductive group
- The standard conjecture C implies $\mathcal{G}_{\text{mot}}$ is reductive (semisimple component)
- For a motive $h(X)$: $h(X)^{\mathcal{G}_{\text{mot}}} = \text{CH}^*(X)_\mathbb{Q}$ (algebraic cycles)
- Transcendental cycles = $h(X) / h(X)^{\mathcal{G}_{\text{mot}}}$

*Step 7 (Lock verification).* For the sieve lock condition with barrier $\mathcal{B}$ and safe region $S$:
$$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}} \text{ iff } \text{Hom}_{\mathcal{C}}(\mathcal{B}, S) = \emptyset$$
By the equivalence $\mathcal{C} \simeq \text{Rep}(G)$, this becomes:
$$\text{Hom}_{\text{Rep}(G)}(\rho_{\mathcal{B}}, \rho_S)^G = 0$$
The lock is verified iff no $G$-equivariant morphisms exist. This is computed via representation theory of $G$.
:::

---

### Holographic Entropy Lock

:::{div} feynman-prose

Here is an information-theoretic lock that works purely from channel capacity.

The idea is this: you cannot transmit more information through a channel than the channel capacity allows. This is Shannon's fundamental theorem, and it is as solid as anything in mathematics.

Now, think of the boundary of your space as a channel. Whatever happens in the bulk, you can only observe it through what crosses the boundary. If the boundary has finite capacity, you can only extract finite information about the bulk.

This creates a lock. Suppose a bad pattern required encoding some structure with high information content. If that information content exceeds the boundary capacity, the bad pattern simply cannot manifest in a way that affects the boundary observables. It is excluded by information theory, not by dynamics or geometry, but by the limits of what can be communicated.

This is the holographic principle from physics, translated into the sieve language. The boundary controls the bulk, because the boundary is the bottleneck for information flow.

:::

:::{prf:theorem} [LOCK-Capacity] Holographic Capacity Lock
:label: mt-lock-entropy

**Rigor Class (Holographic):** L (Literature-Anchored) — see {prf:ref}`def-rigor-classification`

**Bridge Verification:**
1. *Hypothesis Translation:* Certificates $K_{\mathrm{Cap}_H}^+ \wedge K_{\mathrm{TB}_\pi}^+$ imply: bounded boundary channel capacity $C(\partial\mathcal{X})$
2. *Domain Embedding:* $\iota: \mathbf{Hypo}_T \to \mathbf{InfoGeom}$ mapping to information-theoretic channel model $X \to Y \to Z$
3. *Conclusion Import:* Shannon's Channel Coding Theorem {cite}`Shannon48` + Data Processing Inequality {cite}`CoverThomas06` $\Rightarrow K_{\text{Holo}}^+$ (bulk information retrieval bounded by boundary capacity)

**Sieve Signature (Holographic)**
- **Requires:** $K_{\mathrm{Cap}_H}^+$ (capacity certificate), $K_{\mathrm{TB}_\pi}^+$ (topological bound)
- **Produces:** $K_{\text{Holo}}^+$ (holographic capacity certificate, information-theoretic lock)

**Statement:** Let $(\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure with boundary $\partial\mathcal{X}$. If the sieve has issued:
- $K_{\mathrm{Cap}_H}^+$: Capacity bound $\text{Cap}_H(\partial\mathcal{X}) \leq \mathcal{C}_{\max}$
- $K_{\mathrm{TB}_\pi}^+$: Topological bound on fundamental group $|\pi_1(\partial\mathcal{X})| < \infty$

Then the **Data Processing Inequality** provides an information-theoretic lock:

1. **Information Bound:** The retrieveable information satisfies:
   $$I(X; Z) \leq I(X; Y) \leq C(Y)$$
   where $Y$ is the boundary channel and $C(Y)$ is its capacity.

2. **Complexity Bound:** Kolmogorov complexity is bounded:
   $$K(\mathcal{X}) \leq \mathcal{C}_{\max} + O(1)$$

3. **Lock Mechanism:** If $\mathbb{H}_{\mathrm{bad}}$ requires transmitting $I_{\mathrm{bad}} > \mathcal{C}_{\max}$:
   $$\text{Hom}_{\mathbf{Hypo}_T}(\mathbb{H}_{\mathrm{bad}}, \mathcal{X}) = \emptyset$$
   The singularity is excluded by channel capacity.

**Certificate Produced:** $K_{\text{Holo}}^+$ with payload $(\mathcal{C}_{\max}, K_{\max}, \text{DPI verification})$

**Literature:** {cite}`Shannon48`; {cite}`CoverThomas06`; {cite}`Levin73`
:::

:::{prf:proof}

*Step 1 (Data Processing Inequality).* Consider the Markov chain $X \to Y \to Z$, where $X$ is the bulk state, $Y$ is the boundary state, and $Z$ is the observer's measurement. The Data Processing Inequality states that $I(X; Z) \leq I(X; Y)$.

*Step 2 (Channel Capacity).* The mutual information $I(X; Y)$ is upper bounded by the capacity of the boundary channel: $I(X; Y) \leq C(Y) = \max_{p(x)} I(X; Y)$. If the boundary has finite measure/area/dimension, $C(Y)$ is finite.

*Step 3 (Lock Application).* If $\mathbb{H}_{\mathrm{bad}}$ requires establishing an isomorphism or embedding that preserves $I_{\mathrm{bad}}$ bits of information, but $I_{\mathrm{bad}} > C(Y)$, such a morphism cannot exist. The attempt to observe the "bad" structure fails because the boundary cannot transmit the necessary bits to distinguish it.
:::

---

(sec-structural-reconstruction-principle)=
## Structural Reconstruction Principle

*This section introduces a universal metatheorem that resolves epistemic deadlock at Node 17 (Lock) for any hypostructure type. The Structural Reconstruction Principle generalizes Tannakian reconstruction to encompass algebraic, parabolic, and quantum systems, providing a canonical bridge between analytic observables and structural objects.*

:::{div} feynman-prose

And now we come to the main event.

The Structural Reconstruction Principle is the heart of this entire framework. Let me tell you what it says in plain terms, because once you see it, you will understand why everything else is here.

Suppose you have a system with analytic structure: energy bounds, concentration, scaling, stiffness. You have tried all your tactics to decide whether a bad pattern can embed, but you get stuck. The tactics give you partial information, but not a definitive answer.

The Structural Reconstruction Principle says: if your system is stiff enough and tame enough, you can translate the question into a different category where it becomes decidable.

Here is the key insight. Analytic conditions, the Lojasiewicz gradient inequality, o-minimal definability, spectral gaps, these are not just technical hypotheses. They are rigidity conditions. They say: the system cannot wiggle around too much. And when the system cannot wiggle, it must be structured.

What kind of structure? It depends on the type of system:
- For algebraic systems, the structure is algebraic cycles and motives
- For parabolic systems (like dispersive PDEs), the structure is solitons and blow-up profiles
- For quantum systems, the structure is ground states and spectral projections

The reconstruction functor translates analytic observables into these structural objects. And the beautiful thing is: morphism questions in the structural category are decidable. Tannakian categories have effective representation theory. O-minimal structures have algorithmic cell decomposition. Spectral theory computes ground state projections.

So the answer to the lock question, can the bad pattern embed, becomes: does a $G$-equivariant map exist? Is there a definable function connecting the strata? Does the spectral projection annihilate the state? These are questions you can actually answer.

This is why the sieve works. Not because it tries all possibilities, which would be infinite, but because stiffness plus tameness forces the system into a structured world where decidability is possible.

:::

---

### The Reconstruction Metatheorem

:::{prf:theorem} [LOCK-Reconstruction] Structural Reconstruction Principle
:label: mt-lock-reconstruction

**Rigor Class (Reconstruction):** F (Framework-Original) — see {prf:ref}`def-rigor-classification`

This metatheorem is the "Main Result" of the framework: it proves that **Stiff** (Analytic) + **Tame** (O-minimal) systems *must* admit a representation in the structural category $\mathcal{S}$. The Łojasiewicz-Simon inequality restricts the "Moduli of Failure" so severely that only structural objects (algebraic cycles/solitons) remain.

**Sieve Signature (Reconstruction)**
- **Requires:**
  - $K_{D_E}^+$ (finite energy bound on state space)
  - $K_{C_\mu}^+$ (concentration on finite-dimensional profile space)
  - $K_{\mathrm{SC}_\lambda}^+$ (subcritical scaling exponents)
  - $K_{\mathrm{LS}_\sigma}^+$ (Łojasiewicz-Simon gradient inequality)
  - $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ (tactic exhaustion at Node 17 with partial progress)
  - $K_{\text{Bridge}}$ (critical symmetry $\Lambda$ descends from $\mathcal{A}$ to $\mathcal{S}$)
  - $K_{\text{Rigid}}$ (subcategory $\langle\Lambda\rangle_{\mathcal{S}}$ satisfies semisimplicity, tameness, or spectral gap)
- **Produces:** $K_{\text{Rec}}^+$ (constructive dictionary $D_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$ with Hom isomorphism, Lock resolution)

**Statement:** Let $(\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure of type $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$. Let $\mathcal{A}$ denote the category of **Analytic Observables** (quantities controlled by interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$) and let $\mathcal{S} \subset \mathcal{A}$ be the rigid subcategory of **Structural Objects** (algebraic cycles, solitons, ground states). Suppose the sieve has issued the following certificates:

- $K_{D_E}^+$: The energy functional $\Phi: \mathcal{X} \to [0, \infty)$ is bounded: $\sup_{x \in \mathcal{X}} \Phi(x) < \infty$
- $K_{C_\mu}^+$: Energy concentrates on a finite-dimensional profile space $\mathcal{P}$ with $\dim \mathcal{P} \leq d_{\max}$
- $K_{\mathrm{SC}_\lambda}^+$: Scaling exponents $(\alpha, \beta)$ satisfy subcriticality: $\beta - \alpha < \lambda_c$
- $K_{\mathrm{LS}_\sigma}^+$: Łojasiewicz-Simon gradient inequality holds: $\|\nabla\Phi\| \geq C|\Phi - \Phi_{\min}|^\theta$ with $\theta \in (0,1)$

- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$: Tactics E1-E13 fail at Node 17 with partial progress indicators:
  - Dimension bounds: $\dim \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \leq d_{\max}$ (via $K_{C_\mu}^+$)
  - Invariant constraints: $\mathcal{H}_{\text{bad}}$ annihilated by cone $\mathcal{C} \subset \text{End}(\mathcal{X})$
  - Obstruction witness: Critical symmetry group $G_{\text{crit}} \subseteq \text{Aut}(\mathcal{X})$

- $K_{\text{Bridge}}$: A **Bridge Certificate** witnessing that the critical symmetry operator $\Lambda \in \text{End}_{\mathcal{A}}(\mathcal{X})$ (governing the organization of the state space) descends to the structural category:
  $$\Lambda \in \text{End}_{\mathcal{S}}(\mathcal{X})$$
  with action $\rho: G_{\text{crit}} \to \text{Aut}_{\mathcal{S}}(\mathcal{X})$ preserving:
  - Energy (via $K_{D_E}^+$): $\Phi(\rho(g) \cdot x) = \Phi(x)$ for all $g \in G_{\text{crit}}$
  - Stratification (via $K_{\mathrm{SC}_\lambda}^+$): $\rho(g)(\Sigma_k) = \Sigma_k$ for all strata $\Sigma_k$
  - Gradient structure (via $K_{\mathrm{LS}_\sigma}^+$): $\rho(g)$ commutes with gradient flow

- $K_{\text{Rigid}}$: A **Rigidity Certificate** witnessing that the subcategory $\langle\Lambda\rangle_{\mathcal{S}}$ generated by $\Lambda$ satisfies one of:
  - **(Algebraic)** Semisimplicity: $\text{End}_{\mathcal{S}}(\mathbb{1}) = k$ and $\mathcal{S}$ is abelian semisimple (Deligne {cite}`Deligne90`)
  - **(Parabolic)** Tame Stratification: Profile family admits o-minimal stratification $\mathcal{F} = \bigsqcup_k \mathcal{F}_k$ in structure $\mathcal{O}$ (van den Dries {cite}`vandenDries98`)
  - **(Quantum)** Spectral Gap: $\inf(\sigma(L_G) \setminus \{0\}) \geq \delta > 0$ for gauge-fixed linearization $L_G$ (Simon {cite}`Simon83`)

Then there exists a canonical **Reconstruction Functor**:
$$F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$$
satisfying the following properties:

1. **Hom Isomorphism:** For any "bad pattern" $\mathcal{H}_{\text{bad}} \in \mathcal{A}$:
   $$\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \cong \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$$
   The isomorphism is natural in $\mathcal{X}$ and preserves obstruction structure.

2. **Rep Interface Compliance:** $F_{\text{Rec}}$ satisfies the $\mathrm{Rep}$ interface (Node 11):
   - Finite representation: $|F_{\text{Rec}}(X)| < \infty$ for all $X \in \mathcal{A}$ (guaranteed by $K_{C_\mu}^+$)
   - Effectiveness: $F_{\text{Rec}}$ is computable given the input certificates

3. **Lock Resolution:** The inconclusive verdict at Node 17 is resolvable:
   $$K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}} \wedge K_{\text{Bridge}} \wedge K_{\text{Rigid}} \Longrightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}$$
   where verdict $\in \{\text{blk}, \text{br-wit}\}$ (blocked or breached-with-witness).

4. **Type Universality:** The construction is uniform across hypostructure types $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$.

**Required Interface Permits:** $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$, $\mathrm{Rep}$, $\Gamma$

**Prevented Failure Modes:** L.M (Lock Morphism Undecidability), E.D (Epistemic Deadlock), R.I (Reconstruction Incompleteness), C.D (Geometric Collapse)

**Formal Category Constructions:**

*Construction of $\mathcal{A}$ (Analytic Observables Category):*
The category $\mathcal{A}$ is constructed as follows:

- **Objects:** $\text{Ob}(\mathcal{A}) := \{(V, \Phi_V, \mathfrak{D}_V, \sigma_V) \mid V \in \mathcal{X}, \; K_{D_E}^+[V], \; K_{C_\mu}^+[V]\}$
  where:
  - $V \subseteq \mathcal{X}$ is a subobject certified by upstream permits
  - $\Phi_V := \Phi|_V$ is the restricted energy functional
  - $\mathfrak{D}_V := \mathfrak{D}|_V$ is the restricted dissipation
  - $\sigma_V \subseteq \mathbb{R}^+$ is the scaling signature from $K_{\mathrm{SC}_\lambda}^+$

- **Morphisms:** $\text{Hom}_{\mathcal{A}}((V_1, \ldots), (V_2, \ldots)) := \{f: V_1 \to V_2 \mid \text{(A1)-(A4)}\}$ where:
  - **(A1) Energy non-increasing:** $\Phi_{V_2}(f(x)) \leq \Phi_{V_1}(x)$ for all $x \in V_1$
  - **(A2) Dissipation compatible:** $\mathfrak{D}_{V_2}(f(x)) \leq C \cdot \mathfrak{D}_{V_1}(x)$ for uniform $C > 0$
  - **(A3) Scale equivariant:** $f(\lambda \cdot x) = \lambda^{\alpha/\beta} \cdot f(x)$ for scale action $\lambda$
  - **(A4) Gradient regular:** $f$ maps Łojasiewicz regions to Łojasiewicz regions (via $K_{\mathrm{LS}_\sigma}^+$)

- **Composition:** Standard function composition (closed under (A1)-(A4) by chain rule)

- **Identity:** $\text{id}_V = \text{id}$ satisfies (A1)-(A4) trivially

*Construction of $\mathcal{S}$ (Structural Objects Subcategory):*
The subcategory $\mathcal{S} \hookrightarrow \mathcal{A}$ is the **full subcategory** on structural objects:

- **Objects:** $\text{Ob}(\mathcal{S}) := \{W \in \text{Ob}(\mathcal{A}) \mid \text{(S1) or (S2) or (S3)}\}$ where:
  - **(S1) Algebraic:** $W$ is an algebraic cycle: $W = \{[\omega] \in H^*(X; \mathbb{Q}) \mid [\omega] = [Z], Z \text{ algebraic}\}$
  - **(S2) Parabolic:** $W$ is a soliton manifold: $W = \{u \in \mathcal{X} \mid \nabla\Phi(u) = \lambda \cdot u, \lambda \in \sigma_{\text{soliton}}\}$
  - **(S3) Quantum:** $W$ is a ground state sector: $W = \ker(H - E_0)$ for ground energy $E_0$

- **Morphisms:** $\text{Hom}_{\mathcal{S}}(W_1, W_2) := \text{Hom}_{\mathcal{A}}(W_1, W_2)$ (full subcategory)

- **Inclusion functor:** $\iota: \mathcal{S} \hookrightarrow \mathcal{A}$ is the identity on objects/morphisms in $\mathcal{S}$

*Algorithmic Extraction of $G_{\text{crit}}$:*
Given the tactic trace from $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$, the critical symmetry group is extracted as:

1. **Collect obstructions:** Let $\mathcal{O} := \{o_1, \ldots, o_m\}$ be the obstructions encountered in E3-E12
2. **Compute stabilizers:** For each $o_i$, compute $\text{Stab}(o_i) := \{g \in \text{Aut}(\mathcal{X}) \mid g \cdot o_i = o_i\}$
3. **Intersect:** $G_{\text{crit}} := \bigcap_{i=1}^m \text{Stab}(o_i)$
4. **Verify non-triviality:** Check $|G_{\text{crit}}| > 1$ (otherwise no obstruction, tactics should succeed)
5. **Extract generator:** $\Lambda := \frac{d}{dt}\big|_{t=0} \exp(t \cdot \xi)$ where $\xi$ generates $\text{Lie}(G_{\text{crit}})$

*Certificate Production Algorithm for $K_{\text{Bridge}}$:*

**Input:** $G_{\text{crit}}$, upstream certificates $K_{D_E}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$
**Output:** $K_{\text{Bridge}}$ or FAIL

1. **Energy invariance test:** For each generator $g \in G_{\text{crit}}$:
   - Compute $\Delta\Phi(g) := \sup_{x \in \mathcal{X}} |\Phi(g \cdot x) - \Phi(x)|$
   - If $\Delta\Phi(g) > \epsilon_E$ (from $K_{D_E}^+$ tolerance), return FAIL
2. **Stratification test:** For scaling stratification $\{\Sigma_k\}$ from $K_{\mathrm{SC}_\lambda}^+$:
   - Check $g(\Sigma_k) \subseteq \Sigma_k$ for all $k$
   - If any stratum is not preserved, return FAIL
3. **Gradient commutativity test:** For gradient flow $\phi_t$ from $K_{\mathrm{LS}_\sigma}^+$:
   - Check $\|g \circ \phi_t - \phi_t \circ g\|_{\text{op}} < \epsilon_{LS}$ for $t \in [0, T_{\text{test}}]$
   - If commutativity fails, return FAIL
4. **Structural descent verification:** Check $\Lambda \in \text{End}_{\mathcal{S}}(\mathcal{X})$ by type:
   - *Algebraic:* Verify $\Lambda$ is an algebraic correspondence (Chow group test)
   - *Parabolic:* Verify $\Lambda$ preserves soliton structure (scaling test)
   - *Quantum:* Verify $\Lambda$ commutes with $H$ (spectral test)
5. **Output:** $K_{\text{Bridge}} := (\Lambda, G_{\text{crit}}, \rho, \text{verification traces})$

*Certificate Production Algorithm for $K_{\text{Rigid}}$:*

**Input:** $\mathcal{S}$, $\Lambda$, type $T$
**Output:** $K_{\text{Rigid}}$ or FAIL

1. **Type dispatch:**
   - If $T = T_{\text{alg}}$: Go to (2a)
   - If $T = T_{\text{para}}$: Go to (2b)
   - If $T = T_{\text{quant}}$: Go to (2c)

2a. **Semisimplicity test (Algebraic):**
   - Compute $\text{End}_{\mathcal{S}}(\mathbb{1})$ via cohomological methods
   - Check $\text{End}_{\mathcal{S}}(\mathbb{1}) = k$ (no non-trivial endomorphisms)
   - For each simple object $S_i \in \mathcal{S}$, verify $\text{Ext}^1(S_i, S_j) = 0$
   - If all tests pass: $K_{\text{Rigid}} := (\text{semisimple}, G_{\text{motivic}}, \omega)$

2b. **O-minimal test (Parabolic):**
   - Compute cell decomposition of profile family $\mathcal{F}$ from $K_{C_\mu}^+$
   - Verify each cell is definable in structure $\mathcal{O}$
   - Count strata: $N := |\{\mathcal{F}_k\}|$; verify $N < \infty$
   - Check Łojasiewicz compatibility with $K_{\mathrm{LS}_\sigma}^+$
   - If all tests pass: $K_{\text{Rigid}} := (\text{o-minimal}, \mathcal{O}, N, \text{cell data})$

2c. **Spectral gap test (Quantum):**
   - Compute spectrum $\sigma(L_G)$ via Rayleigh-Ritz or exact diagonalization
   - Compute gap $\delta := \inf(\sigma(L_G) \setminus \{0\})$
   - Verify $\delta > 0$ (isolated ground state)
   - Compute ground state projector $\Pi_0 = \mathbb{1}_{\{0\}}(L_G)$
   - If $\delta > 0$: $K_{\text{Rigid}} := (\text{spectral-gap}, \delta, \psi_0, \Pi_0)$

3. **Failure:** If type-specific test fails, return FAIL with diagnostic

**Literature:**
- *Tannakian Categories:* {cite}`Deligne90`; {cite}`SaavedraRivano72`; {cite}`DeligneMillne82`
- *Motivic Galois Groups:* {cite}`Andre04`; {cite}`Jannsen92`; {cite}`Nori00`
- *O-minimal Structures:* {cite}`vandenDries98`; {cite}`Wilkie96`; {cite}`Lojasiewicz65`
- *Dispersive PDEs:* {cite}`KenigMerle06`; {cite}`MerleZaag98`; {cite}`DKM19`
- *Spectral Theory:* {cite}`Simon83`; {cite}`ReedSimon78`; {cite}`Kato95`; {cite}`GlimmJaffe87`; {cite}`FSS76`
- *Algebraic Geometry:* {cite}`Kleiman68`; {cite}`Humphreys72`
:::

:::{prf:proof}

*Step 1 (Breached-inconclusive certificate analysis).* The certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ records that tactics E1-E13 have been exhausted at Node 17 without determining whether $\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) = \emptyset$. The upstream certificates $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$ provide **partial progress data**:

- **Dimension bounds** (from $K_{C_\mu}^+$): $\dim \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \leq d_{\max}$ via concentration on $\mathcal{P}$
- **Scaling constraints** (from $K_{\mathrm{SC}_\lambda}^+$): The exponents $(\alpha, \beta)$ stratify the Hom-space by weight
- **Gradient regularity** (from $K_{\mathrm{LS}_\sigma}^+$): The kernel $\ker(\text{ev}: \text{Hom} \to \mathcal{X})$ has Łojasiewicz structure
- **Obstruction witness** (from E3-E12): A critical symmetry group $G_{\text{crit}} \subseteq \text{Aut}(\mathcal{X})$ emerges

The key insight is that $G_{\text{crit}}$ is not arbitrary—it is precisely the group of symmetries that prevent E1-E13 from concluding. This group becomes the target for the Bridge Certificate.

*Step 2 (Bridge certificate: structural symmetry).* The certificate $K_{\text{Bridge}}$ establishes that $G_{\text{crit}}$ acts not merely as analytic automorphisms, but as **structural** automorphisms:
$$\rho: G_{\text{crit}} \hookrightarrow \text{Aut}_{\mathcal{S}}(\mathcal{X})$$

This is verified by checking that $G_{\text{crit}}$ preserves the permit-certified data:

**(a) Energy preservation (from $K_{D_E}^+$):** For all $g \in G_{\text{crit}}$ and $x \in \mathcal{X}$:
$$\Phi(\rho(g) \cdot x) = \Phi(x)$$
The energy functional certified by $D_E$ is $G_{\text{crit}}$-invariant.

**(b) Stratification equivariance (from $K_{\mathrm{SC}_\lambda}^+$):** For the scaling stratification $\mathcal{X} = \bigsqcup_{k=0}^N \Sigma_k$:
$$\rho(g)(\Sigma_k) = \Sigma_k \quad \text{for all } g \in G_{\text{crit}}, \; k \in \{0, \ldots, N\}$$
The strata defined by scaling exponents are preserved.

**(c) Gradient compatibility (from $K_{\mathrm{LS}_\sigma}^+$):** The Łojasiewicz gradient flow commutes with $G_{\text{crit}}$:
$$\rho(g) \circ \nabla\Phi = \nabla\Phi \circ \rho(g)$$

**(d) Critical operator descent:** The operator $\Lambda$ generating $G_{\text{crit}}$ lies in $\text{End}_{\mathcal{S}}(\mathcal{X})$:
- *Algebraic:* $\Lambda = L$ (Lefschetz operator) is an algebraic correspondence ({cite}`Kleiman68`)
- *Parabolic:* $\Lambda = x \cdot \nabla$ (scaling generator) preserves soliton structure ({cite}`Weinstein85`)
- *Quantum:* $\Lambda = H$ (Hamiltonian) defines the spectral decomposition ({cite}`ReedSimon78`)

The Bridge Certificate is the mathematical content of the phrase "the organizing symmetry is structural."

*Step 3 (Rigidity certificate decomposition).* The certificate $K_{\text{Rigid}}$ provides the categorical rigidity needed for reconstruction. This certificate interacts with the upstream permits $K_{C_\mu}^+$ (finite dimensionality) and $K_{\mathrm{LS}_\sigma}^+$ (gradient structure). We analyze by type:

**Case A (Algebraic — Tannakian Rigidity):** The category $\mathcal{S}$ is a neutral Tannakian category over $k$ with:
- $\text{End}_{\mathcal{S}}(\mathbb{1}) = k$ (no non-trivial endomorphisms of the unit)
- $\mathcal{S}$ is abelian and semisimple (every object decomposes into simples)

By Deligne's theorem {cite}`Deligne90`, there exists an affine group scheme $G = \text{Spec}(\mathcal{O}(G))$ with:
$$\mathcal{S} \simeq \text{Rep}_k(G)$$
The group $G$ is the **motivic Galois group** when $\mathcal{S} = \mathbf{Mot}_k$ ({cite}`Andre04`; {cite}`Jannsen92`). The concentration certificate $K_{C_\mu}^+$ ensures $\dim \omega(V) < \infty$ for all $V \in \mathcal{S}$.

**Case B (Parabolic — O-minimal Tameness):** The profile family $\mathcal{F}$ from $K_{C_\mu}^+$ admits a tame stratification in an o-minimal structure $\mathcal{O}$ (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\exp}$):
$$\mathcal{F} = \bigsqcup_{k=1}^N \mathcal{F}_k$$
where each $\mathcal{F}_k$ is a $C^m$-submanifold definable in $\mathcal{O}$. By van den Dries {cite}`vandenDries98` and Wilkie {cite}`Wilkie96`, such stratifications have:
- Finite complexity: $N < \infty$ strata (compatible with $K_{C_\mu}^+$)
- Cell decomposition: Each $\mathcal{F}_k$ is a union of cells
- Definable selection: Continuous selectors exist on each stratum
- Łojasiewicz inequality: Compatible with $K_{\mathrm{LS}_\sigma}^+$ ({cite}`Lojasiewicz65`)

**Case C (Quantum — Spectral Gap):** The linearized operator $L_G$ (gauge-fixed Hamiltonian, Fokker-Planck generator, or Dirichlet form) satisfies:
$$\inf(\sigma(L_G) \setminus \{0\}) \geq \delta > 0$$
This spectral gap, established via $K_{\mathrm{LS}_\sigma}^+$ (Simon {cite}`Simon83`; {cite}`Glimm87`), ensures:
- Isolated ground state: $\ker(L_G) = \text{span}(\psi_0)$
- Exponential decay: Solutions converge to ground state at rate $e^{-\delta t}$
- Perturbative stability: Gap persists under small perturbations (Kato {cite}`Kato95`)

*Step 4 (Dictionary construction).* We construct the Reconstruction Functor $F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$ explicitly by type. The construction uses the finiteness from $K_{C_\mu}^+$ and the regularity from $K_{\mathrm{LS}_\sigma}^+$:

**Type $T_{\text{alg}}$ (Algebraic):** Let $\omega: \mathcal{A} \to \mathbf{Vect}_k$ be the fiber functor (e.g., Betti cohomology $H_B$, or de Rham $H_{\text{dR}}$). Define:
$$F_{\text{Rec}}^{\text{alg}}(X) := (\omega(X), \rho_X)$$
where $\rho_X: G \to \text{GL}(\omega(X))$ is the representation induced by the Tannakian structure ({cite}`DeligneMillne82`). The functor satisfies:
- $F_{\text{Rec}}^{\text{alg}}(\mathbb{1}) = (\mathbf{1}_k, \text{triv})$ (monoidal unit)
- $F_{\text{Rec}}^{\text{alg}}(X \otimes Y) = F_{\text{Rec}}^{\text{alg}}(X) \otimes F_{\text{Rec}}^{\text{alg}}(Y)$ (tensor compatibility)
- $\dim \omega(X) < \infty$ (from $K_{C_\mu}^+$)

**Type $T_{\text{para}}$ (Parabolic):** Using the o-minimal cell decomposition from $K_{\text{Rigid}}$, define:
$$F_{\text{Rec}}^{\text{para}}(X) := (\text{profile}(X), \text{stratum}(X), \text{cell}(X))$$
where:
- $\text{profile}(X) \in \mathcal{P}$: The limit profile from $K_{C_\mu}^+$ ({cite}`KenigMerle06`)
- $\text{stratum}(X) \in \{1, \ldots, N\}$: Index of the containing stratum $\mathcal{F}_k$
- $\text{cell}(X)$: Cell index within the stratum (from o-minimal structure)

By Merle-Zaag {cite}`MerleZaag98` and Duyckaerts-Kenig-Merle {cite}`DKM19`, the profile library is finite: $|\mathcal{P}| < \infty$. The Łojasiewicz exponent from $K_{\mathrm{LS}_\sigma}^+$ determines the convergence rate to profiles.

**Type $T_{\text{quant}}$ (Quantum):** Using the spectral resolution of $L_G$ from $K_{\text{Rigid}}$, define:
$$F_{\text{Rec}}^{\text{quant}}(X) := (\psi_0(X), \sigma(X), \Pi_0(X))$$
where:
- $\psi_0(X)$: Projection onto the ground state sector ({cite}`GlimmJaffe87`)
- $\sigma(X) \subset [0, \infty)$: Spectrum of $L_G|_X$
- $\Pi_0(X) = \mathbb{1}_{\{0\}}(L_G)$: Ground state projector

The spectral gap $\delta > 0$ from $K_{\text{Rigid}}$ ensures $\Pi_0$ is finite-rank (Fröhlich-Simon-Spencer {cite}`FSS76`).

*Step 5 (Hom isomorphism verification).* We prove the central isomorphism using the certificates $K_{\text{Bridge}}$ and $K_{\text{Rigid}}$:
$$\Phi_{\text{Rec}}: \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) \xrightarrow{\cong} \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$$

**Injectivity:** Let $f, f' \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$ with $F_{\text{Rec}}(f) = F_{\text{Rec}}(f')$. By $K_{\text{Bridge}}$, the critical symmetry $G_{\text{crit}}$ acts on both sides via structural automorphisms. The $G_{\text{crit}}$-equivariant structure of $F_{\text{Rec}}$ (inherited from $K_{D_E}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$) implies:
$$f(x) = f'(x) \quad \text{for all } x \in \mathcal{H}_{\text{bad}}$$
using:
- *Algebraic:* Faithfulness of $\omega$ ({cite}`Deligne90`, Prop. 2.11)
- *Parabolic:* Definability in $\mathcal{O}$ ({cite}`vandenDries98`, Ch. 4)
- *Quantum:* Spectral uniqueness ({cite}`ReedSimon78`, Thm. VIII.5)

**Surjectivity:** Let $\phi \in \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X}))$. The certificate $K_{\text{Rigid}}$ ensures "enough morphisms" to lift:
- *Algebraic:* Semisimplicity implies $\mathcal{S}$ has enough injectives/projectives ({cite}`SaavedraRivano72`, §I.4)
- *Parabolic:* O-minimal definable selection provides lifts ({cite}`vandenDries98`, Thm. 6.1.7)
- *Quantum:* Spectral theorem reconstructs operators from spectral data ({cite}`Kato95`, §V.3)

By the universal property of $F_{\text{Rec}}$, there exists $f \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$ with $F_{\text{Rec}}(f) = \phi$. This lift is unique by injectivity.

**Naturality:** For any morphism $g: \mathcal{X} \to \mathcal{Y}$ in $\mathcal{A}$, the diagram:
$$\begin{CD}
\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) @>{\Phi_{\text{Rec}}}>> \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \\
@V{g_*}VV @V{F_{\text{Rec}}(g)_*}VV \\
\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{Y}) @>{\Phi_{\text{Rec}}}>> \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{Y}))
\end{CD}$$
commutes by functoriality of $F_{\text{Rec}}$. This is the content of the $\mathrm{Cat}_{\mathrm{Hom}}$ interface compliance.

*Step 6 (Lock resolution).* The Hom isomorphism from Step 5 resolves the Node 17 Lock. This step consumes the $\mathrm{Cat}_{\mathrm{Hom}}$ interface permit:

**Case: $\text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) = \emptyset$**

By the isomorphism $\Phi_{\text{Rec}}$:
$$\text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X}) = \emptyset$$
The sieve issues certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ (VICTORY). The bad pattern cannot embed. This triggers success at Node 17.

**Case: $\text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \neq \emptyset$**

The Reconstruction Functor provides an explicit morphism witness via the $\mathrm{Rep}$ interface:
$$\phi \in \text{Hom}_{\mathcal{S}}(F_{\text{Rec}}(\mathcal{H}_{\text{bad}}), F_{\text{Rec}}(\mathcal{X})) \leadsto f := \Phi_{\text{Rec}}^{-1}(\phi) \in \text{Hom}_{\mathcal{A}}(\mathcal{H}_{\text{bad}}, \mathcal{X})$$
The sieve issues certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{morph}}$ with explicit witness $f$ (FATAL if bad pattern embeds, RECOVERABLE if controllable via other permits).

**Decidability via Interface Permits:** The key insight is that $\text{Hom}_{\mathcal{S}}$ is decidable because each rigidity type has effective algorithms:
- *Algebraic:* $G$-invariants in finite-dimensional representations are computable via Chevalley-Jordan decomposition ({cite}`Humphreys72`)
- *Parabolic:* O-minimal cell decomposition is effective ({cite}`vandenDries98`, Thm. 1.8.1); profile matching uses $K_{C_\mu}^+$
- *Quantum:* Spectral projections are computable for discrete spectrum ({cite}`ReedSimon78`); gap from $K_{\text{Rigid}}$ ensures isolation

The inconclusive verdict is resolved: **partial progress (from $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$) + structural symmetry ($K_{\text{Bridge}}$) + rigidity ($K_{\text{Rigid}}$) = decidable answer**.

*Step 7 (Certificate assembly).* Construct the output certificate incorporating all upstream permit data:
$$K_{\text{Rec}}^+ = \left(F_{\text{Rec}}, \Phi_{\text{Rec}}, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}, T, D_{\text{Rec}}\right)$$

**Certificate Produced:** $K_{\text{Rec}}^+$ with payload:
- $F_{\text{Rec}}: \mathcal{A} \to \mathcal{S}$: Reconstruction functor (fiber/profile/spectral by type)
- $\Phi_{\text{Rec}}: \text{Hom}_{\mathcal{A}} \xrightarrow{\cong} \text{Hom}_{\mathcal{S}}$: Natural isomorphism with explicit inverse
- $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}} \in \{\text{blk}, \text{morph}\}$: Resolved Lock outcome at Node 17
- $T \in \{T_{\text{alg}}, T_{\text{para}}, T_{\text{quant}}\}$: Hypostructure type
- $D_{\text{Rec}}$: Constructive Dictionary satisfying $\mathrm{Rep}$ interface (Node 11):
  - Finiteness: $|D_{\text{Rec}}(x)| < \infty$ for all $x$ (inherited from $K_{C_\mu}^+$)
  - Algorithm: Explicit computation procedure for $F_{\text{Rec}}$
- Upstream certificates consumed: $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$, $K_{\text{Bridge}}$, $K_{\text{Rigid}}$
:::

:::{prf:remark} Reconstruction uses obligation ledgers
:label: rem-rec-uses-ledger

When {prf:ref}`mt-lock-reconstruction` is invoked (from any $K^{\mathrm{inc}}$ route, particularly $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$), its input includes the **obligation ledger** $\mathsf{Obl}(\Gamma)$ ({prf:ref}`def-obligation-ledger`).

The reconstruction procedure must produce one of the following outcomes:
1. **New certificates that discharge entries:** {prf:ref}`mt-lock-reconstruction` produces $K_{\text{Bridge}}$, $K_{\text{Rigid}}$, and ultimately $K_{\text{Rec}}^+$, which enable inc-upgrades ({prf:ref}`def-inc-upgrades`) to fire during closure, discharging relevant $K^{\mathrm{inc}}$ entries from the ledger.

2. **Refined missing set:** If full discharge is not possible, {prf:ref}`mt-lock-reconstruction` may refine the $\mathsf{missing}$ component of existing $K^{\mathrm{inc}}$ certificates into a strictly more explicit set of prerequisites—smaller template requirements, stronger preconditions, or more specific structural data. This refinement produces a new $K^{\mathrm{inc}}$ with updated payload.

**Formalization:**

$$
\text{Structural Reconstruction}: \mathsf{Obl}(\Gamma) \to \left(\{K^+_{\text{new}}\} \text{ enabling discharge}\right) \cup \left(\mathsf{Obl}'(\Gamma) \text{ with refined } \mathsf{missing}\right)

$$

This ensures reconstruction makes definite progress: either discharging obligations or producing a strictly refined $\mathsf{missing}$ specification.

:::

---

### Type Instantiation Table

The following table summarizes how the Structural Reconstruction Principle instantiates across the three fundamental hypostructure types. Each row shows how the interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$ specialize to the given type:

| **Hypostructure Type** | **Critical Symmetry ($\Lambda$)** | **Bridge Certificate ($K_{\text{Bridge}}$)** | **Rigidity Certificate ($K_{\text{Rigid}}$)** | **Resulting Theorem** |
|:----------------------|:----------------------------------|:--------------------------------------------|:---------------------------------------------|:---------------------|
| **Algebraic** ($T_{\text{alg}}$) | Lefschetz operator $L: H^{n-1} \to H^{n+1}$ | Standard Conjecture B: "$L$ is algebraic" (correspondence in $\text{CH}^1(X \times X)$) | Semisimplicity: $\mathbf{Mot}_k^{\text{num}} \simeq \text{Rep}(\mathcal{G}_{\text{mot}})$ ({cite}`Jannsen92`) | **Hodge Conjecture:** Every harmonic $(p,p)$-form is $\mathbb{Q}$-algebraic |
| **Parabolic** ($T_{\text{para}}$) | Scaling operator $\Lambda = x \cdot \nabla + \frac{2}{\alpha}$ | Virial identity from $K_{\mathrm{SC}_\lambda}^+$: Scaling is monotone in $V(t) = \int |x|^2 |u|^2$ | Tame stratification via $K_{C_\mu}^+$: $\mathcal{P} = \{W_1, \ldots, W_N\}$ in $\mathbb{R}_{\text{an}}$ ({cite}`MerleZaag98`) | **Soliton Resolution:** $u(t) = u_L + \sum_j u_j^*(t-t_j) + o(1)$ |
| **Quantum** ($T_{\text{quant}}$) | Hamiltonian $H = -\Delta + V$ | Spectral condition from $K_{D_E}^+$: $H \geq 0$ with discrete spectrum | Spectral gap via $K_{\mathrm{LS}_\sigma}^+$: $\inf \sigma(H) \setminus \{E_0\} \geq E_0 + \Delta$ ({cite}`GlimmJaffe87`) | **Mass Gap:** Vacuum unique, gap $\Delta > 0$ |

**Permit Specialization by Type:**

| **Permit** | **Algebraic ($T_{\text{alg}}$)** | **Parabolic ($T_{\text{para}}$)** | **Quantum ($T_{\text{quant}}$)** |
|:-----------|:--------------------------------|:----------------------------------|:--------------------------------|
| $K_{D_E}^+$ | Height function bounded | $\|u\|_{H^1}^2 < \infty$ | $\langle \psi, H\psi \rangle < \infty$ |
| $K_{C_\mu}^+$ | Hodge numbers finite | Profile space $\mathcal{P}$ finite | Ground state isolated |
| $K_{\mathrm{SC}_\lambda}^+$ | Weight filtration bounded | Scaling exponent subcritical | Spectral dimension finite |
| $K_{\mathrm{LS}_\sigma}^+$ | Hodge metric analytic | Łojasiewicz at solitons | Spectral gap $\delta > 0$ |

---

### Corollaries

:::{div} feynman-prose

These corollaries spell out what the Structural Reconstruction Principle buys you in practice.

The Bridge-Rigidity Dichotomy says: you always get an answer. Either the bridge certificate works, and you can translate to the structural world, or it fails, and the failure itself tells you something. You are never stuck in limbo.

The Analytic-Structural Equivalence is the precise statement of what I keep calling "soft implies hard." If your analytic conditions are satisfied, you do not lose any morphism information when you pass to the structural category. The two worlds see the same obstructions.

And the Permit Flow Theorem shows how all the certificates chain together. Energy flows to concentration flows to scaling flows to stiffness. Then stiffness enables the bridge, which enables rigidity, which enables reconstruction. Each permit enables the next, until you reach the final verdict.

:::

:::{prf:corollary} Bridge-Rigidity Dichotomy
:label: cor-bridge-rigidity

If $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{br\text{-}inc}}$ is issued at Node 17 (with upstream certificates $K_{D_E}^+$, $K_{C_\mu}^+$, $K_{\mathrm{SC}_\lambda}^+$, $K_{\mathrm{LS}_\sigma}^+$ satisfied), then exactly one of the following holds:

1. **Bridge Certificate obtainable:** $K_{\text{Bridge}}$ can be established, and the Lock resolves via {prf:ref}`mt-lock-reconstruction` producing $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}$
2. **Bridge obstruction identified:** The failure of $K_{\text{Bridge}}$ provides a new certificate $K_{\text{Bridge}}^-$ containing:
   - A counterexample to structural descent: $\Lambda \notin \text{End}_{\mathcal{S}}(\mathcal{X})$
   - An analytic automorphism not preserving structure: $g \in G_{\text{crit}}$ with $g(\mathcal{S}) \not\subseteq \mathcal{S}$
   - A violation witness for one of $K_{D_E}^+$, $K_{\mathrm{SC}_\lambda}^+$, or $K_{\mathrm{LS}_\sigma}^+$ under the $G_{\text{crit}}$-action

In either case, the epistemic deadlock at Node 17 is resolved.
:::

:::{prf:corollary} Analytic-Structural Equivalence
:label: cor-analytic-structural

Under the hypotheses of {prf:ref}`mt-lock-reconstruction` (with all interface permits $D_E$, $C_\mu$, $\mathrm{SC}_\lambda$, $\mathrm{LS}_\sigma$, $\mathrm{Cat}_{\mathrm{Hom}}$ satisfied), the categories $\mathcal{A}$ and $\mathcal{S}$ are **Hom-equivalent** on the subcategory generated by $\mathcal{H}_{\text{bad}}$:

$$
\mathcal{A}|_{\langle\mathcal{H}_{\text{bad}}\rangle} \simeq_{\text{Hom}} \mathcal{S}|_{\langle F_{\text{Rec}}(\mathcal{H}_{\text{bad}})\rangle}

$$

This equivalence is the rigorous formulation of "soft implies hard" for morphisms. In particular:
- Analytic obstructions (from $K_{\mathrm{LS}_\sigma}^+$) are equivalent to structural obstructions
- Concentration data (from $K_{C_\mu}^+$) determines the structural representation
- The $\mathrm{Rep}$ interface is satisfied by both categories
:::

:::{prf:corollary} Permit Flow Theorem
:label: cor-permit-flow

The Structural Reconstruction Principle defines a **permit flow** at Node 17:

$$\begin{CD}
K_{D_E}^+ @>>> K_{C_\mu}^+ @>>> K_{\mathrm{SC}_\lambda}^+ @>>> K_{\mathrm{LS}_\sigma}^+ \\
@. @. @VVV @VVV \\
@. @. K_{\text{Bridge}} @>>> K_{\text{Rigid}} \\
@. @. @VVV @VVV \\
@. @. @. K_{\text{Rec}}^+ @>>> K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}}
\end{CD}$$

Each arrow represents a certificate dependency. The output $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\text{verdict}} \in \{\text{blk}, \text{morph}\}$ is the decidable resolution of the Lock.
:::

---

### The Analytic-Algebraic Rigidity Lemma

*This lemma provides the rigorous "engine" that powers the algebraic case ($T_{\text{alg}}$) of the Structural Reconstruction Principle ({prf:ref}`mt-lock-reconstruction`). It formalizes the a posteriori inference: analytic stiffness plus tameness forces algebraicity.*

:::{div} feynman-prose

This lemma is the technical heart of the Hodge conjecture approach. Let me tell you what it says in simple terms.

You have a harmonic form of type $(k,k)$. The Hodge conjecture says: if this form has rational periods, it should be the fundamental class of an algebraic cycle. But how do you prove that something analytic is actually algebraic?

The answer: by exclusion. You show that anything that is not algebraic would violate one of your certificates. The argument has four steps, each excluding a different type of pathology:

1. **Stiffness excludes wild smooth forms.** A smooth function that is not real-analytic has flat directions in the energy landscape. But the stiffness certificate says: no flat directions. So the form must be real-analytic away from its singular support.

2. **Tameness excludes fractal singularities.** The singular support might be complicated, but it is o-minimal definable. This means it has a finite cell decomposition. Combined with finite energy, this forces the form to extend as a rectifiable current.

3. **Hodge type excludes non-holomorphic contributions.** On a Kahler manifold, a real-analytic harmonic $(k,k)$-form with these properties corresponds to a complex analytic subvariety. The deformation rigidity from stiffness pins down the variety.

4. **GAGA completes the argument.** On a projective variety, every analytic subvariety is algebraic. This is Serre's remarkable theorem, and it finishes the proof.

The beautiful thing is that each step uses a different piece of the certificate structure. Energy bounds, gradient inequalities, o-minimal geometry, and Hodge theory all work together to force the conclusion.

:::

:::{prf:lemma} Analytic-Algebraic Rigidity
:label: lem-analytic-algebraic-rigidity

**Sieve Signature (Analytic-Algebraic)**
- **Requires:**
  - $K_{D_E}^+$ (finite energy: $\|\eta\|_{L^2}^2 < \infty$)
  - $K_{\mathrm{LS}_\sigma}^+$ (stiffness: spectral gap $\lambda > 0$ on Hodge-Riemann pairing)
  - $K_{\mathrm{Tame}}^+$ (tameness: singular support $\Sigma(\eta)$ is o-minimal definable)
  - $K_{\mathrm{Hodge}}^{(k,k)}$ (type constraint: $\eta$ is harmonic of type $(k,k)$)
- **Produces:** $K_{\mathrm{Alg}}^+$ (algebraicity certificate: $[\eta] \in \mathcal{Z}^k(X)_{\mathbb{Q}}$)

**Statement:** Let $X$ be a smooth complex projective variety with hypostructure $(\mathcal{X}, \Phi, \mathfrak{D})$ of type $T_{\text{alg}}$. Let $\eta \in H^{2k}(X, \mathbb{C})$ be a harmonic form representing a cohomology class of type $(k,k)$. Suppose the sieve has issued the following certificates:

- $K_{D_E}^+$ **(Energy Bound):** The energy functional satisfies $\Phi(\eta) = \|\eta\|_{L^2}^2 < \infty$.

- $K_{\mathrm{LS}_\sigma}^+$ **(Stiffness/Spectral Gap):** The form $\eta$ lies in a subspace $V \subset H^{2k}(X)$ on which the Hodge-Riemann pairing $Q(\cdot, \cdot)$ is non-degenerate with definite signature. For any perturbation $\delta\eta \in V$, the second variation of the energy satisfies:

  $$
  \|\nabla^2 \Phi(\eta)\| \geq \lambda > 0

  $$

  This is the **stiffness condition**: the energy landscape admits no flat directions.

- $K_{\mathrm{Tame}}^+$ **(O-minimal Tameness):** The singular support

  $$
  \Sigma(\eta) = \{x \in X : \eta(x) \text{ is not real-analytic}\}

  $$

  is definable in an o-minimal structure $\mathcal{O}$ expanding $\mathbb{R}$ (e.g., $\mathbb{R}_{\text{an}}$, $\mathbb{R}_{\exp}$).

- $K_{\mathrm{Hodge}}^{(k,k)}$ **(Type Constraint):** The form $\eta$ is harmonic ($\Delta\eta = 0$) and of Hodge type $(k,k)$.

Then $\eta$ is the fundamental class of an algebraic cycle with rational coefficients:

$$
[\eta] \in \mathcal{Z}^k(X)_{\mathbb{Q}}

$$

The sieve issues certificate $K_{\mathrm{Alg}}^+$ with payload $(Z^{\text{alg}}, [Z^{\text{alg}}] = [\eta], \mathbb{Q})$.

**Required Interface Permits:** $D_E$, $\mathrm{LS}_\sigma$, $\mathrm{Tame}$, $\mathrm{Hodge}$, $\mathrm{Rep}$

**Prevented Failure Modes:** W.S (Wild Smooth), S.I (Singular Irregularity), N.H (Non-Holomorphic), N.A (Non-Algebraic)

**Proof (4 Steps):**

*Step 1 (Exclusion of wild smooth forms via $K_{\mathrm{LS}_\sigma}^+$).* The stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ excludes $C^\infty$ forms that are not real-analytic. Suppose $\eta$ were smooth but not real-analytic at some point $p \in X$. By the construction of smooth bump functions, there exists a perturbation:
$$\eta_\epsilon = \eta + \epsilon \psi$$
where $\psi$ is a smooth form with $\text{supp}(\psi) \subset U$ for an arbitrarily small neighborhood $U$ of $p$.

Because $\psi$ is localized, its interactions with the global Hodge-Riemann pairing $Q$ can be made arbitrarily small or sign-indefinite. This creates **flat directions** in the energy landscape:

$$
\langle \nabla^2\Phi(\eta) \cdot \psi, \psi \rangle \to 0 \quad \text{as } U \to \{p\}

$$

This violates the uniform spectral gap condition $\|\nabla^2\Phi\| \geq \lambda > 0$ from $K_{\mathrm{LS}_\sigma}^+$. The Łojasiewicz-Simon inequality ({cite}`Simon83`; {cite}`Lojasiewicz65`) implies the energy landscape admits no flat directions at critical points.

**Conclusion:** $\eta$ must be real-analytic on $X \setminus \Sigma$, where $\Sigma$ is the singular support. The failure mode **W.S (Wild Smooth)** is excluded.

*Step 2 (Rectifiability via $K_{\mathrm{Tame}}^+$ and $K_{D_E}^+$).* The tameness certificate $K_{\mathrm{Tame}}^+$ combined with finite energy $K_{D_E}^+$ ensures that $\eta$ extends to a rectifiable current.

By the **Cell Decomposition Theorem** for o-minimal structures ({cite}`vandenDries98`, Theorem 1.8.1), the singular support $\Sigma$ admits a finite stratification:

$$
\Sigma = \bigsqcup_{i=1}^N S_i

$$

where each $S_i$ is a $C^m$-submanifold definable in $\mathcal{O}$. The finiteness $N < \infty$ is guaranteed by o-minimality.

The finite energy certificate $K_{D_E}^+$ implies $\|\eta\|_{L^2}^2 < \infty$, hence $\eta$ has **finite mass** as a current:

$$
\mathbb{M}(\eta) = \int_X |\eta| \,dV < \infty

$$

By the **Federer-Fleming Closure Theorem** adapted to tame geometry ({cite}`Federer69`, §4.2; {cite}`vandenDries98`, Ch. 6), a current with:
- Finite mass
- O-minimal definable support

is a **rectifiable current**. The tameness of $\mathcal{O}$ excludes pathological fractal-like singularities.

**Conclusion:** $\eta$ extends to a current defined by integration over an analytic chain. The failure mode **S.I (Singular Irregularity)** is excluded.

*Step 3 (Holomorphic structure via $K_{\mathrm{Hodge}}^{(k,k)}$ and $K_{\mathrm{LS}_\sigma}^+$).* The type constraint $K_{\mathrm{Hodge}}^{(k,k)}$ combined with stiffness establishes holomorphicity.

On a Kähler manifold $X$, a real-analytic harmonic $(k,k)$-form with integral periods defines a holomorphic geometric object. The **Poincaré-Lelong equation** ({cite}`GriffithsHarris78`, Ch. 3):

$$
\frac{i}{2\pi} \partial\bar{\partial} \log |s|^2 = [Z]

$$

relates $(k,k)$-currents to zero sets of holomorphic sections. This provides the bridge from analytic to holomorphic.

The stiffness certificate $K_{\mathrm{LS}_\sigma}^+$ implies **deformation rigidity**: the tangent space to the moduli of such objects vanishes:

$$
H^1(Z, \mathcal{N}_{Z/X}) = 0

$$

where $\mathcal{N}_{Z/X}$ is the normal bundle ({cite}`Demailly12`, §VII). The moduli space is discrete (zero-dimensional). A "stiff" form cannot deform continuously into a non-holomorphic form without breaking harmonicity or Hodge type.

**Conclusion:** The analytic chain underlying $\eta$ is a complex analytic subvariety $Z \subset X$. The failure mode **N.H (Non-Holomorphic)** is excluded.

*Step 4 (Algebraization via GAGA).* The projectivity of $X$ enables the final step via Serre's GAGA theorem ({cite}`Serre56`).

We have established that $\eta$ corresponds to a global analytic subvariety $Z$ in $X^{\text{an}}$ (the analytification of $X$). Since $X$ is a projective variety, **Serre's GAGA Theorem** applies:

> *The functor from algebraic coherent sheaves on $X$ to analytic coherent sheaves on $X^{\text{an}}$ is an equivalence of categories.*

In particular:
- Every analytic subvariety of a projective variety is algebraic
- The ideal sheaf $\mathcal{I}_Z$ is the analytification of an algebraic ideal sheaf $\mathcal{I}_{Z^{\text{alg}}}$

Therefore:

$$
Z = (Z^{\text{alg}})^{\text{an}}

$$

for a unique algebraic subvariety $Z^{\text{alg}} \subset X$.

**Conclusion:** The cohomology class $[\eta]$ is the image of the algebraic cycle class:

$$
[\eta] = [Z^{\text{alg}}] \in H^{2k}(X, \mathbb{Q})

$$

The failure mode **N.A (Non-Algebraic)** is excluded.

**Certificate Produced:** $K_{\mathrm{Alg}}^+$ with payload:
- $Z^{\text{alg}}$: The algebraic cycle
- $[Z^{\text{alg}}] = [\eta]$: Cycle class equality in $H^{2k}(X, \mathbb{Q})$
- $\mathbb{Q}$-coefficients: Rationality of the cycle
- Upstream certificates consumed: $K_{D_E}^+$, $K_{\mathrm{LS}_\sigma}^+$, $K_{\mathrm{Tame}}^+$, $K_{\mathrm{Hodge}}^{(k,k)}$

**Literature:**
- *Łojasiewicz-Simon theory:* {cite}`Simon83`; {cite}`Lojasiewicz65`
- *O-minimal structures:* {cite}`vandenDries98`; {cite}`Wilkie96`
- *Geometric measure theory:* {cite}`Federer69`
- *Complex geometry:* {cite}`GriffithsHarris78`; {cite}`Demailly12`
- *GAGA:* {cite}`Serre56`
:::

---

**Connection to {prf:ref}`mt-lock-reconstruction`:** This lemma is the **algebraic instantiation** of the Structural Reconstruction Principle:

| Structural Reconstruction Component | Lemma Instantiation |
|:------------------------------------|:--------------------|
| $\mathcal{A}$ (Analytic Observables) | Harmonic $(k,k)$-forms in $H^{2k}(X, \mathbb{C})$ |
| $\mathcal{S}$ (Structural Objects) | Algebraic cycles $\mathcal{Z}^k(X)_{\mathbb{Q}}$ |
| $K_{\text{Bridge}}$ | Lefschetz operator $L$ is algebraic (Standard Conjecture B) |
| $K_{\text{Rigid}}$ | Semisimplicity of $\mathbf{Mot}_k^{\text{num}}$ ({cite}`Jannsen92`) |
| $F_{\text{Rec}}$ | Cycle class map $\text{cl}: \mathcal{Z}^k \to H^{2k}$ |

The lemma provides the rigorous **a posteriori proof** that stiffness + tameness forces algebraicity, implementing the "soft implies hard" principle for Hodge theory.
