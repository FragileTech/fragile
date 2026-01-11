(sec-equivalence-transport)=
# Part X: Equivalence and Transport

:::{div} feynman-prose

Now we come to something that is, frankly, one of the most useful tricks in all of mathematics and physics. Here is the situation: you have a problem that is difficult to analyze directly, but if you could just look at it from a different angle---in a different coordinate system, or after quotienting out some symmetry, or using a different metric---then suddenly everything becomes clear. The question is: how do you transfer your hard-won results back to the original problem?

This is what equivalence moves and transport lemmas are all about. Think of it this way: suppose you want to prove that a certain quantity is bounded. You cannot prove it directly. But you notice that your system has a symmetry---maybe rotational invariance---and if you work on the quotient space where you have identified all the rotations, the proof becomes straightforward. The equivalence move takes you from the original space to the quotient; the transport lemma tells you how to carry the bound back.

The key insight is that these moves are not approximations. They are exact transformations with explicit comparability bounds. When you go from $\Phi$ to $\tilde{\Phi}$, you know exactly how the values relate: within factors of $C_1$ and $C_2$. This precision is what makes the whole machinery work. You are not losing information; you are reorganizing it.

:::

:::{prf:remark} Naming convention

This part defines **equivalence moves** ({prf:ref}`def-equiv-symmetry`--{prf:ref}`def-equiv-bridge`) and **transport lemmas** ({prf:ref}`def-transport-t1`--{prf:ref}`def-transport-t6`). These are distinct from the **Lock tactics** ({prf:ref}`def-e1`--{prf:ref}`def-e10`) defined in {ref}`the Lock Exclusion Tactics section <sec-lock-exclusion-tactics>`. The `Eq` prefix distinguishes equivalence moves from Lock tactics.

:::

(sec-equivalence-library)=
## Equivalence Library

:::{div} feynman-prose

Let me tell you what we mean by an "admissible" equivalence move, because this word is doing real work. Not every transformation qualifies. You might transform your problem into something that looks simpler but loses essential information, or transforms it into something that is not actually equivalent in the ways that matter.

An admissible move has to satisfy three conditions, and each one earns its place. First, you need comparability bounds: the new Lyapunov functional $\tilde{\Phi}$ and dissipation $\tilde{\mathfrak{D}}$ must be bounded above and below by the originals, with explicit constants. This is crucial because our whole machinery depends on these functionals. If you cannot control how they change, you cannot transport your results.

Second, you need structural preservation: the interfaces, the permits, the whole architecture must carry over. You are not allowed to cheat by destroying the structure that makes the sieve work. Third, you need a certificate: a formal record that the equivalence was established correctly. This certificate becomes part of your proof artifact.

:::

:::{prf:definition} Admissible equivalence move
:label: def-equiv-move

An **admissible equivalence move** for type $T$ is a transformation $(x, \Phi, \mathfrak{D}) \mapsto (\tilde{x}, \tilde{\Phi}, \tilde{\mathfrak{D}})$ with:
1. **Comparability bounds**: Constants $C_1, C_2 > 0$ with
   $$\begin{aligned}
   C_1 \Phi(x) &\leq \tilde{\Phi}(\tilde{x}) \leq C_2 \Phi(x) \\
   C_1 \mathfrak{D}(x) &\leq \tilde{\mathfrak{D}}(\tilde{x}) \leq C_2 \mathfrak{D}(x)
   \end{aligned}$$
2. **Structural preservation**: Interface permits preserved
3. **Certificate production**: Equivalence certificate $K_{\text{equiv}}$

:::

---

### Standard Equivalence Moves

:::{div} feynman-prose

Now let me show you the standard repertoire of equivalence moves. These are the moves you will use over and over again. Each one corresponds to a different kind of simplification, and understanding when to apply each one is part of the art.

**Symmetry quotient (Eq1)** is perhaps the most intuitive. If your system has a symmetry group $G$ acting on it---say, rotational symmetry---then many distinct states $x$ are really "the same" as far as the dynamics care. By passing to the quotient $X/G$, you work with equivalence classes $[x]_G$ instead of individual points. The Lyapunov functional on the quotient is the infimum over the orbit, which is the right notion because if any point in the orbit has low energy, they all do (up to the symmetry).

**Metric deformation (Eq2)** is the tool you reach for when the standard metric makes your analysis difficult, but some equivalent metric makes it tractable. This is the essence of hypocoercivity: you cannot prove your bound in the original metric, but by tilting the metric slightly, the estimate goes through.

**Conjugacy (Eq3)** lets you work with a simpler dynamical system that is related to yours by a change of variables. If $h: X \to Y$ is invertible and your flow $S_t$ on $X$ corresponds to $\tilde{S}_t = h \circ S_t \circ h^{-1}$ on $Y$, then results on $Y$ transport back to $X$.

**Surgery identification (Eq4)** handles the case where you have excised some bad region $E$ and want to work only on the complement. Outside the excision, the two objects agree.

**Analytic-hypostructure bridge (Eq5)** is the most subtle. It connects the classical analytical world (solutions $u$ to PDEs) with our hypostructure framework (states $x$ in the sieve). The maps $\mathcal{H}$ and $\mathcal{A}$ go back and forth, with controlled bounds.

:::

:::{prf:definition} Eq1: Symmetry quotient
:label: def-equiv-symmetry

For symmetry group $G$ acting on $X$:
$$\tilde{x} = [x]_G \in X/G$$
Comparability: $\Phi([x]_G) = \inf_{g \in G} \Phi(g \cdot x)$ (coercivity modulo $G$)

:::

:::{prf:definition} Eq2: Metric deformation (Hypocoercivity)
:label: def-equiv-metric

Replace metric $d$ with equivalent metric $\tilde{d}$:
$$C_1 d(x, y) \leq \tilde{d}(x, y) \leq C_2 d(x, y)$$
Used when direct LS fails but deformed LS holds.

:::

:::{prf:definition} Eq3: Conjugacy
:label: def-equiv-conjugacy

For invertible $h: X \to Y$:
$$\tilde{S}_t = h \circ S_t \circ h^{-1}$$
Comparability: $\Phi_Y(h(x)) \sim \Phi_X(x)$

:::

:::{prf:definition} Eq4: Surgery identification
:label: def-equiv-surgery-id

Outside excision region $E$:
$$x|_{X \setminus E} = x'|_{X \setminus E}$$
Transport across surgery boundary.

:::

:::{prf:definition} Eq5: Analytic-hypostructure bridge
:label: def-equiv-bridge

Between classical solution $u$ and hypostructure state $x$:
$$x = \mathcal{H}(u), \quad u = \mathcal{A}(x)$$
with inverse bounds.

:::

---

(sec-yes-tilde-permits)=
## YES$^\sim$ Permits

:::{div} feynman-prose

Here is where the machinery pays off. Suppose you have a predicate $P_i$ that you need to certify, but you cannot prove it directly on your original object $x$. However, you can:

1. Transform to an equivalent object $\tilde{x}$ (via one of the equivalence moves)
2. Prove $P_i$ on $\tilde{x}$ (getting a YES certificate there)
3. Transport the result back

The YES$^\sim$ ("YES-tilde") certificate formalizes this indirect proof. It is a triple containing: the equivalence certificate (proving $x \sim \tilde{x}$), the transport lemma certificate (explaining how results move between them), and the actual YES certificate on the transformed object.

The beautiful part is that our metatheorems cannot tell the difference. When a metatheorem asks for a YES certificate on predicate $P_i$, you can hand it a YES$^\sim$ certificate instead, and it works just the same. This is what we mean by "accepting YES$^\sim$"---the metatheorem's conclusion follows just as surely whether you give it a direct proof or an indirect one via equivalence.

This is not cheating. The equivalence moves preserve all the structure that matters. A bound proven on the quotient space really does imply a bound on the original space, once you account for the comparability constants.

:::

:::{prf:definition} YES$^\sim$ certificate
:label: def-yes-tilde-cert

A **YES$^\sim$ certificate** for predicate $P_i$ is a triple:
$$K_i^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_i^+[\tilde{x}])$$
where:
- $K_{\text{equiv}}$: Certifies $x \sim_{\mathrm{Eq}} \tilde{x}$ for some equivalence move {prf:ref}`def-equiv-symmetry`--{prf:ref}`def-equiv-bridge`
- $K_{\text{transport}}$: Transport lemma certificate (from {prf:ref}`def-transport-t1`--{prf:ref}`def-transport-t6`)
- $K_i^+[\tilde{x}]$: YES certificate for $P_i$ on the equivalent object $\tilde{x}$

:::

:::{prf:definition} YES$^\sim$ acceptance
:label: def-yes-tilde-accept

A metatheorem $\mathcal{M}$ **accepts YES$^\sim$** if:
$$\mathcal{M}(K_{I_1}, \ldots, K_{I_i}^{\sim}, \ldots, K_{I_n}) = \mathcal{M}(K_{I_1}, \ldots, K_{I_i}^+, \ldots, K_{I_n})$$
That is, YES$^\sim$ certificates may substitute for YES certificates in the metatheorem's preconditions.

:::

---

(sec-transport-toolkit)=
## Transport Toolkit

:::{div} feynman-prose

Now we need to be precise about how results actually transport between equivalent objects. This is where the comparability bounds earn their keep.

**Inequality transport (T1)** is the most basic: if you have proved that $\tilde{\Phi}(\tilde{x}) \leq E$ on the transformed space, what can you say about $\Phi(x)$ on the original? The answer uses the lower comparability bound: since $C_1 \Phi(x) \leq \tilde{\Phi}(\tilde{x})$, you get $\Phi(x) \leq E/C_1$. You pick up a factor, but you get your bound.

**Integral transport (T2)** works similarly for integrated quantities like total dissipation. The upper comparability bound controls how much the integral can grow.

**Quotient transport (T3)** handles the symmetry case. If the predicate holds on the equivalence class $[x]_G$, when does it hold on the original point $x$? You need an additional "orbit bound" to control behavior within the orbit.

**Metric equivalence transport (T4)** is essential for hypocoercivity. If you have proved an LS inequality in the deformed metric $\tilde{d}$, what do you get in the original metric $d$? The constant degrades by a factor of $C_2$, but the qualitative structure survives.

**Conjugacy transport (T5)** tells you that invariants---quantities that do not change under the dynamics---transport trivially under conjugacy. If $\tau$ is an invariant for $S_t$, then $\tilde{\tau} = \tau \circ h^{-1}$ is an invariant for $\tilde{S}_t$.

**Surgery transport (T6)** is the simplest: outside the excised region, everything is literally the same, so all certificates transfer verbatim.

:::

:::{prf:definition} T1: Inequality transport
:label: def-transport-t1

Under comparability $C_1 \Phi \leq \tilde{\Phi} \leq C_2 \Phi$:
$$\tilde{\Phi}(\tilde{x}) \leq E \Rightarrow \Phi(x) \leq E/C_1$$

:::

:::{prf:definition} T2: Integral transport
:label: def-transport-t2

Under dissipation comparability:
$$\int \tilde{\mathfrak{D}} \leq C_2 \int \mathfrak{D}$$

:::

:::{prf:definition} T3: Quotient transport
:label: def-transport-t3

For $G$-quotient with coercivity:
$$P_i(x) \Leftarrow P_i([x]_G) \wedge \text{(orbit bound)}$$

:::

:::{prf:definition} T4: Metric equivalence transport
:label: def-transport-t4

LS inequality transports under equivalent metrics:
$$\text{LS}_{\tilde{d}}(\theta, C) \Rightarrow \text{LS}_d(\theta, C/C_2)$$

:::

:::{prf:definition} T5: Conjugacy transport
:label: def-transport-t5

Invariants transport under conjugacy:
$$\tau(x) = \tilde{\tau}(h(x))$$

:::

:::{prf:definition} T6: Surgery identification transport
:label: def-transport-t6

Outside excision, all certificates transfer:
$$K[x|_{X \setminus E}] = K[x'|_{X \setminus E}]$$

:::

---

(sec-promotion-system)=
## Promotion System

:::{div} feynman-prose

Now let me tell you about something subtle but important. Sometimes you cannot prove a predicate directly, and you cannot find a nice equivalence move either. But you can prove something weaker: a "blocked" certificate that says the predicate holds conditionally, or under certain assumptions.

The promotion system is machinery for upgrading these weaker certificates to full YES certificates, once additional information becomes available.

**Immediate promotion** works when the blocked certificate plus earlier certificates together imply the predicate. Think of it this way: maybe you could not prove $P_i$ by itself, but once you have established $P_1, \ldots, P_{i-1}$, those earlier results combine with your blocked certificate to give you $P_i$ after all. This is not circular---you are using the sieve order.

**A-posteriori promotion** is trickier: here you use certificates that come later in the sieve. This sounds suspicious---are we going backwards in time? But it is legitimate because these are not causal dependencies; they are logical ones. Once the full sieve has run and you have the final collection of certificates, you can look back and realize that some of your blocked certificates can now be promoted.

The **promotion closure** is the mathematical formalization: take your initial certificate set, close under all promotion rules, and iterate until you reach a fixed point. This closure includes everything that can be derived.

**Replay** is the final step: once you have the closure, you re-run the sieve logic to see what stronger fingerprint you can now achieve. The same inputs, but with more certificates available, may yield a better outcome.

:::

:::{prf:definition} Immediate promotion
:label: def-promotion-immediate

Rules using only past/current certificates:

**Barrier-to-YES**: If blocked certificate plus earlier certificates imply the predicate:
$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j < i} K_j^+ \Rightarrow K_i^+$$

Example: $K_{\text{Cap}}^{\mathrm{blk}}$ (singular set measure zero) plus $K_{\text{SC}}^+$ (subcritical) may together imply $K_{\text{Geom}}^+$.

:::

:::{prf:definition} A-posteriori promotion
:label: def-promotion-aposteriori

Rules using later certificates:

$$K_i^{\mathrm{blk}} \wedge \bigwedge_{j > i} K_j^+ \Rightarrow K_i^+$$

Example: Full Lock passage ($K_{\mathrm{Cat}_{\mathrm{Hom}}}^+$) may retroactively promote earlier blocked certificates to full YES.

:::

:::{prf:definition} Promotion closure
:label: def-promotion-closure

The **promotion closure** $\mathrm{Cl}(\Gamma)$ is the least fixed point:
$$\Gamma_0 = \Gamma, \quad \Gamma_{n+1} = \Gamma_n \cup \{K : \text{promoted or inc-upgraded from } \Gamma_n\}$$
$$\mathrm{Cl}(\Gamma) = \bigcup_n \Gamma_n$$

This includes both blocked-certificate promotions ({prf:ref}`def-promotion-permits`) and inconclusive-certificate upgrades ({prf:ref}`def-inc-upgrades`).

:::

:::{prf:definition} Replay semantics
:label: def-replay

Given final context $\Gamma_{\text{final}}$, the **replay** is a re-execution of the sieve under $\mathrm{Cl}(\Gamma_{\text{final}})$, potentially yielding a different (stronger) fingerprint.

:::
