# Fractal Set as Causal Set

**Prerequisites**: {doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set`, Causal Set Theory ({cite}`BombelliLeeEtAl87,Sorkin05`)

---

## TLDR

*Notation: $(E, \prec_{\mathrm{CST}})$ = Fractal Set with CST causal order; $\prec_{\mathrm{LC}}$ = geometric light-cone order; BLMS = Bombelli-Lee-Meyer-Sorkin axioms; $d = \dim \mathcal{X}$, $D = d + 1$ (spacetime dimension for CST formulas); $\rho_{\mathrm{adaptive}}$ = QSD sampling density; $\sqrt{\det g}$ = Riemannian volume element.*

**The Fractal Set is a Valid Causal Set**: The episode set $E$ with CST ordering $\prec_{\mathrm{CST}}$ satisfies all three BLMS axioms (irreflexivity, transitivity, local finiteness), making the Fractal Set a rigorous causal set in the sense of quantum gravity research.

**Adaptive Sprinkling Innovation**: Unlike standard Poisson sprinkling with constant spacetime density, QSD sampling yields an inhomogeneous density $\rho_{\mathrm{adaptive}}(x, t) \propto \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T}$, adapting resolution to the learned geometry (higher-weight regions get more episodes).

**Causal Set Machinery Applies**: With causal set status established, CST tools (d'Alembertian, dimension estimators, curvature measures) apply in full, using the geometric light-cone order and adaptive-density formulas given here.

---

(sec-cst-intro)=
## Introduction

:::{div} feynman-prose
Here is a beautiful connection that was waiting to be discovered. Causal set theory is one of the leading approaches to quantum gravity—the idea that spacetime is fundamentally discrete, made up of a locally finite (typically countable) set of events with a partial ordering that encodes causal structure. The program was launched by Bombelli, Lee, Meyer, and Sorkin in 1987 {cite}`BombelliLeeEtAl87`, and it has developed into a sophisticated mathematical framework (see {cite}`Sorkin05`).

Now, the Fractal Set is also a discrete structure with a causal ordering. Episodes are events; CST edges encode causal precedence. The question is: does the Fractal Set satisfy the axioms of a causal set? If it does, then the mathematical machinery of causal set theory—developed over decades by quantum gravity researchers—becomes available to us, and the adaptive-density versions are spelled out explicitly here.

The answer is yes. The Fractal Set is a valid causal set. But it is more than that: it is an *adaptive* causal set, where the sampling density automatically adjusts to local geometry. This goes beyond the standard Lorentz-invariant Poisson construction, which fixes a constant density.
:::

Causal set theory (CST) posits that spacetime is fundamentally discrete: a locally finite (typically countable) collection of events with a partial ordering encoding causal relationships. The Fractal Set, defined in {doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set`, provides exactly such a structure via its CST edges ({prf:ref}`def-fractal-set-cst-edges`) and CST axioms ({prf:ref}`def-fractal-set-cst-axioms`). This chapter establishes that:

1. The Fractal Set satisfies all BLMS axioms for causal sets
2. QSD sampling provides adaptive (not uniform) sprinkling
3. CST mathematical machinery applies, with explicit adaptive-density formulas

---

(sec-cst-axioms)=
## Causal Set Theory: Axioms and Framework

### Standard Causal Set Definition

:::{div} feynman-prose
Let me tell you what causal set theory is really about. The idea is almost embarrassingly simple, which is often a sign that it might be right.

Here is the question: What is spacetime made of? Einstein taught us that spacetime is not a fixed stage on which physics plays out—it is itself a dynamical thing, curved by matter and energy. But even in general relativity, spacetime is still a *continuum*. Between any two points, there are infinitely many other points. And here is the problem: when you try to combine general relativity with quantum mechanics, infinities show up everywhere. Divergences. Renormalization nightmares. The math is trying to tell us something.

Causal set theory takes a bold step. It says: maybe spacetime is not continuous at all. Maybe at the tiniest scales—near the Planck length, $10^{-35}$ meters—spacetime is actually made of *discrete events*. A finite number of points in any bounded region. No continuum, no infinities.

But here is the clever part. You cannot just sprinkle points at random and call it spacetime. You need *structure*. And the structure you need is *causality*—the relationship of "this event can influence that event." In relativity, this is determined by light cones. Event A can influence event B only if a signal traveling at or below light speed can get from A to B.

So a causal set is just this: a locally finite collection of events, plus a relation that tells you which events can causally influence which other events. That is all. From this minimal structure, the hope is that the entire fabric of spacetime—dimension, curvature, topology—can be *recovered* in the large-scale limit.

The three axioms you are about to see are just the mathematical way of saying: "You cannot be your own ancestor" (irreflexivity), "If A caused B and B caused C, then A caused C" (transitivity), and "Only finitely many things can happen between any two events" (local finiteness). These are not arbitrary mathematical conditions—they are the minimum requirements for anything deserving the name "causality."
:::

:::{prf:definition} Causal Set ({cite}`BombelliLeeEtAl87`)
:label: def-causal-set-blms

A **causal set** $(C, \prec)$ is a locally finite partially ordered set satisfying:

**Axiom CS1 (Irreflexivity)**: For all $e \in C$, $e \not\prec e$

**Axiom CS2 (Transitivity)**: For all $e_1, e_2, e_3 \in C$, if $e_1 \prec e_2$ and $e_2 \prec e_3$, then $e_1 \prec e_3$

**Axiom CS3 (Local Finiteness)**: For all $e_1, e_2 \in C$, the set $\{e \in C : e_1 \prec e \prec e_2\}$ is finite

**Physical interpretation**:
- Elements $e \in C$ represent spacetime events
- $e_1 \prec e_2$ means "$e_1$ causally precedes $e_2$" (inside future light cone)
- Local finiteness = finite events in any causal interval (discreteness)
:::

### Poisson Sprinkling (Standard Construction)

:::{div} feynman-prose
Now comes the question that every discrete theory must face: how do you go from the discrete to the continuous? If spacetime is really made of discrete events, we had better be able to recover ordinary smooth spacetime in some limit—otherwise the theory would contradict everything we know about physics at everyday scales.

The standard approach is called *Poisson sprinkling*. The idea is beautifully simple. Take a region of classical spacetime—a chunk of Minkowski space, say—and throw points into it at random, with a fixed density. Then define the causal order by using the light cones of the original spacetime: point A precedes point B if A is in the causal past of B.

Think of it like this. Imagine you have a piece of paper (representing 2D spacetime) and you close your eyes and drop grains of sand on it uniformly at random. The positions where the grains land are your causal set elements. Two grains are causally related if one is in the "light cone" of the other.

Why Poisson? Because a Poisson process gives you *Lorentz invariance for free* {cite}`BombelliLeeEtAl87,Sorkin05`. Here is the beautiful thing: if you boost to a different reference frame, the Poisson sprinkling looks exactly the same statistically. The uniform density is Lorentz-invariant. This is crucial—you want the fundamental discreteness to not pick out any preferred frame.

But here is the limitation. Uniform density means you put the same number of points per unit volume everywhere. In flat spacetime, this is fine. But in curved spacetime, "interesting" things happen in high-curvature regions—near black holes, at the Big Bang—and these are precisely the regions where you might want more resolution, more points, to capture the physics accurately. Uniform sprinkling is blind to curvature. It treats boring flat regions and exciting curved regions exactly the same.

This is where the Fractal Set will improve on the standard construction.
:::

:::{prf:definition} Poisson Sprinkling
:label: def-poisson-sprinkling-cst

Given a $D$-dimensional Lorentzian manifold $(M, g_{\mu\nu})$ with volume element $dV = \sqrt{-\det g} \, d^D x$, a **Poisson sprinkling** with constant density $\rho_0$ is ({cite}`BombelliLeeEtAl87,Sorkin05`):

1. **Sample count**: Draw $N \sim \mathrm{Poisson}(\rho_0 V_{\mathrm{total}})$

2. **Sample points**: Conditional on $N$, draw $\{x_i\}$ i.i.d. with density
   $p(x) = \sqrt{-\det g(x)} / V_{\mathrm{total}}$

3. **Define order**: $e_i \prec e_j$ iff $x_i$ is in the causal past of $x_j$

**Property**: Expected number of elements in causal interval $I(e_1, e_2)$ is $\mathbb{E}[|I|] = \rho_0 \cdot V_{\mathrm{Lorentz}}(I)$.
:::

**Limitation**: Uniform density $\rho_0 = \mathrm{const}$ does not adapt to local geometry:
- Over-sampling in flat regions (wasteful)
- Under-sampling in curved regions (loss of information)

An inhomogeneous density generally breaks global Lorentz invariance; in the Fractal Set this tradeoff is intentional because the target geometry and sampling measure are emergent and algorithm-defined.

---

(sec-fractal-causal-set)=
## Fractal Set as Adaptive Causal Set

:::{div} feynman-prose
Here is where we connect two different worlds. On one side, we have causal set theory—a deep idea from quantum gravity about the discrete structure of spacetime. On the other side, we have the Fractal Set—a data structure that records the complete history of an optimization algorithm. It turns out they are the same thing.

Well, almost the same. The Fractal Set is a causal set, but it is a *better* causal set. Let me explain what I mean by that.

In standard causal set theory, you sprinkle points uniformly. Every region of spacetime gets the same point density, whether it is flat and boring or curved and interesting. This is democratic but wasteful. You end up with too many points in regions where nothing much is happening, and not enough points in regions where the geometry is rich and complex.

The Fractal Set does something smarter. Its points—the episodes generated by the Adaptive Gas—are distributed according to a density that *adapts* to the local geometry. The density goes like $\sqrt{\det g} \cdot e^{-U_{\mathrm{eff}}/T}$. The first factor, $\sqrt{\det g}$, is the Riemannian volume element—it accounts for how the metric stretches or shrinks volumes. The second factor, $e^{-U_{\mathrm{eff}}/T}$, is a Boltzmann weight that concentrates points where the effective potential is low—that is, where the "interesting" dynamics happens.

This is not something we had to put in by hand. It emerges automatically from the QSD (Quasi-Stationary Distribution) of the stochastic dynamics. The algorithm naturally explores more densely the regions that matter.

Think of it this way. If you wanted to draw a picture of a mountain, you would use more ink on the parts where the terrain changes rapidly—the ridges, the valleys—and less ink on the flat plains. The Fractal Set does exactly this, but for spacetime. It puts more "events" where the geometry is interesting, and fewer where it is bland.
:::

### Causal Order on Episodes

To make the growth process concrete, the CST edges form a layered, directed tree as spacetime evolves.

:::{figure} figures/cst-growth-tree.svg
:alt: CST tree growth across timesteps with directed edges.
:width: 95%
:align: center

CST tree growth across timesteps. Each layer is a timestep; directed edges encode causal precedence.
:::

:::{prf:definition} Causal Order on Fractal Set
:label: def-fractal-causal-order

Let episodes be nodes $e = n_{i,t}$ and let $E_{\mathrm{CST}}$ be the CST edge set
({prf:ref}`def-fractal-set-cst-edges`). The canonical causal order is the transitive closure:

$$
e_i \prec_{\mathrm{CST}} e_j \quad \iff \quad \exists \text{ directed CST path from } e_i \text{ to } e_j .
$$

To connect with light cones, define the time-indexed metric
$g_t(x) = H(x, S(t)) + \epsilon_\Sigma I$ ({prf:ref}`def-adaptive-diffusion-tensor-latent`) and,
for any CST path $\gamma = (e_0, \ldots, e_m)$ with consecutive CST edges at times $t_k$ and
displacements $\Delta x_k$, define its length

$$
L_g(\gamma) := \sum_{k=0}^{m-1} \|\Delta x_k\|_{g_{t_k}} .
$$

The induced (directed) path length on episodes is

$$
d_g(e_i, e_j) := \inf_{\gamma: e_i \to e_j} L_g(\gamma),
$$
with $d_g(e_i, e_j) = \infty$ if no CST path exists. Define the instantaneous propagation bound

$$
c_{\mathrm{eff}}(t_k) := \max_{(e \to e') \in E_{\mathrm{CST}} \text{ at } t_k}
\frac{\|\Delta x\|_{g_{t_k}}}{\Delta t_k} ,
$$
where $\Delta t_k := t_{k+1} - t_k$.

The maximum exists because each timestep has finitely many CST edges; if a timestep is empty,
set $c_{\mathrm{eff}}(t)=0$.

By construction of the kinetic step, the drift velocity is squashed by
$\psi_v$ ({prf:ref}`def-latent-velocity-squashing`), so each CST edge satisfies
$\|\Delta x\|_{g_{t_k}} / \Delta t_k \le V_{\mathrm{alg}}$. Hence
$c_{\mathrm{eff}}(t_k) \le V_{\mathrm{alg}}$ provides an algorithm-defined speed limit.
Let $c:=V_{\mathrm{alg}}$.

Define the **geometric path length**. Let $t_-:=\min(t_i,t_j)$ and $t_+:=\max(t_i,t_j)$. Then

$$
d_{\mathrm{geo}}(e_i, e_j) := \inf_{\gamma} \int_{t_-}^{t_+} \|\dot{x}(t)\|_{g_t}\,dt,
$$
where the infimum is over $C^1$ curves $\gamma:[t_-,t_+]\to\mathcal{X}$ with
$\gamma(t_i)=x_i$ and $\gamma(t_j)=x_j$; if no such curve exists, set
$d_{\mathrm{geo}}=\infty$.

Define the geometric (light-cone) order

$$
e_i \prec_{\mathrm{LC}} e_j \quad \iff \quad t_i < t_j
\;\wedge\; d_{\mathrm{geo}}(e_i, e_j) \leq \int_{t_i}^{t_j} c\, dt = c\,(t_j-t_i) .
$$

For discrete timesteps, interpret the integral as $c\sum_k \Delta t_k$.
In practice, $d_{\mathrm{geo}}$ is computed from the reconstructed $g_R$ on each slice:
either by a geodesic solver in the continuum lift, or by the IG-graph shortest-path
distance with edge lengths induced by $g_R$ (which converges to $d_{\mathrm{geo}}$ by
{prf:ref}`thm:induced-riemannian-structure` and {prf:ref}`mt:cheeger-gradient`).

**Physical meaning**: $e_i \prec_{\mathrm{LC}} e_j$ iff information from $e_i$ can causally
influence $e_j$.

**Compatibility**: Proposition {prf:ref}`prop-fractal-causal-order-equivalence` shows
that $\prec_{\mathrm{CST}}$ is an order-embedding into $\prec_{\mathrm{LC}}$ and, on
CST-connected pairs, $d_g$ converges to the realized trajectory length and upper-bounds
$d_{\mathrm{geo}}$.
Geometric estimators below (volume and dimension) use $\prec_{\mathrm{LC}}$ to match the
continuum causal structure; $\prec_{\mathrm{CST}}$ remains the algorithmic ancestry order.
:::

:::{note}
CST edges encode continuous timelike evolution only. When a walker clones
($c(n_{i,t})\neq\bot$), the new node $n_{i,t}$ starts a **new** CST path; clone ancestry is
recorded in $E_{\mathrm{clone}}$ ({prf:ref}`def-fractal-set-clone-ancestry`) and IA edges, and is
excluded from $d_g$ and $\prec_{\mathrm{CST}}$.
:::

:::{div} feynman-prose
Look at this definition carefully. It says that one episode "precedes" another if (1) it happens earlier in time, AND (2) it is close enough in space that a signal could travel between them.

The second condition is the light cone constraint. If two events are too far apart in space, no signal—not even light—can travel between them in the available time. They are "spacelike separated," to use the relativity jargon. They cannot influence each other.

The effective speed $c_{\mathrm{eff}}$ is like a "speed limit" for information propagation in this system. It plays the role that the speed of light plays in relativity. Nothing can travel faster, so nothing can causally connect events that are outside each other's "cones." We use the algorithmic cap $c=V_{\mathrm{alg}}$ for the geometric order, with $c_{\mathrm{eff}}(t)\le c$ by construction.

Here is the physical picture. Draw a diagram with time going up and space going sideways. At each event, draw a cone opening upward—this is the "future light cone." Only events inside (or on) this cone can be influenced by the original event. The geometric causal order $e_i \prec_{\mathrm{LC}} e_j$ means exactly that $e_j$ is inside the future cone of $e_i$.
:::

:::{figure} figures/cst-light-cone.svg
:alt: Light cone diagram showing causal and spacelike events.
:width: 90%
:align: center

Light-cone causal order. Events inside the cone are causally related; events outside are spacelike.
:::

:::{prf:proposition} Graph-Light-Cone Compatibility
:label: prop-fractal-causal-order-equivalence

The CST order $\prec_{\mathrm{CST}}$ is an order-embedding into the geometric light-cone order
$\prec_{\mathrm{LC}}$. Moreover, on CST-connected pairs the graph distance converges to the
length of the realized trajectory and upper-bounds $d_{\mathrm{geo}}$; thus the orders are
compatible on realized ancestry but no new relations between distinct lineages are inferred.

*Proof.* For any CST edge, $\|\Delta x\|_{g_{t_k}} / \Delta t_k \le c$ by the velocity squashing
map $\psi_v$ ({prf:ref}`def-latent-velocity-squashing`), so the piecewise-geodesic interpolation
of a CST path is a future-directed causal curve with speed $\le c$. Hence
$e_i \prec_{\mathrm{CST}} e_j \Rightarrow e_i \prec_{\mathrm{LC}} e_j$.

For CST-connected pairs, let $h:=\max_k \Delta t_k$ and interpolate the CST path by a $C^1$
curve in the continuum lift. Expansion Adjunction and the continuum injection
({prf:ref}`thm-expansion-adjunction`, {prf:ref}`mt:continuum-injection`) imply the discrete
length $d_g(e_i,e_j)$ converges to the length of that realized trajectory as $h\to 0$.
Since $d_{\mathrm{geo}}$ is the infimum over all $C^1$ curves, we have
$d_{\mathrm{geo}}(e_i,e_j) \le \lim_{h\to 0} d_g(e_i,e_j)$ on CST-connected pairs. Thus the
Lorentzian order induced by $g=-c^2 dt^2+g_R$ is compatible with $\prec_{\mathrm{CST}}$ on the
realized episode set; no additional relations are inferred between distinct lineages. $\square$
:::

### QSD Sampling = Adaptive Sprinkling

:::{prf:theorem} Fractal Set Episodes Follow Adaptive Density
:label: thm-fractal-adaptive-sprinkling

Episodes generated by the Adaptive Gas are distributed according to:

$$
\rho_{\mathrm{adaptive}}(x, t) = \frac{1}{Z(t)} \sqrt{\det g(x, t)} \exp\left(-\frac{U_{\mathrm{eff}}(x, t)}{T}\right)
$$

where $g(x, t) = H(x, S(t)) + \epsilon_\Sigma I$ is the emergent Riemannian metric
({prf:ref}`def-adaptive-diffusion-tensor-latent`) and
$Z(t) = \int \sqrt{\det g(x, t)} \exp\left(-\frac{U_{\mathrm{eff}}(x, t)}{T}\right) dx$.

This specifies the instantaneous spatial marginal of the QSD; episodes need not form an independent
Poisson process. For a time window, the spacetime intensity is
$\lambda(t, x) = r(t)\, \rho_{\mathrm{adaptive}}(x, t)$, where $r(t)$ is the episode rate
reconstructed from CST edges. For discrete timesteps $\{t_k\}$ with step sizes
$\Delta t_k = t_{k+1} - t_k$, define

$$
E_{\mathrm{CST}}(t_k) := \{(e \to e') \in E_{\mathrm{CST}} : t(e) = t_k\}
$$
and

$$
r(t_k) := |E_{\mathrm{CST}}(t_k)| / \Delta t_k .
$$
Equivalently, since each alive walker contributes exactly one CST edge per step,
$r(t_k) = |\{e = n_{i,t_k}\}| / \Delta t_k$. For continuous $t$, take $r(t)$ to be the
piecewise-constant interpolation.

**Window convention**: We use half-open time windows $[t_0, t_1)$ so each episode at time
$t_k \in [t_0, t_1)$ contributes exactly one CST edge. For discrete steps, this gives
$N = |E| = \sum_k |E_{\mathrm{CST}}(t_k)|$ and $R = \int_{t_0}^{t_1} r(t)\,dt = N$.

**Comparison with Poisson sprinkling**:

| Standard CST | Fractal Set |
|:------------|:------------|
| Density $\rho = \mathrm{const}$ | Density $\rho(x, t) \propto \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T}$ |
| Uniform sampling | Adaptive sampling |
| Ad-hoc choice of $\rho$ | Automatic from QSD |
:::

:::{admonition} Conformal vs. General Geometry
:class: note

Standard CST order+counting fixes only the conformal class, so the usual "Minkowski trick" applies only when
$g$ is conformally flat. In that case, a uniform sprinkling in $(M,g)$ appears as a non-uniform density
$\rho \propto \Omega^D$ in flat coordinates, consistent with the adaptive density above. The
density-corrected Laplacian (see {prf:ref}`def-density-corrected-laplacian`) removes this coordinate bias.
For general geometries with nonzero Weyl curvature, there is no global flattening; instead we use the
emergent metric $g_R$ (from {prf:ref}`def-adaptive-diffusion-tensor-latent`) and the QSD density
({prf:ref}`thm-fractal-adaptive-sprinkling`) together with LSI mixing
({prf:ref}`thm-n-uniform-lsi-exchangeable`) to keep the continuum limits coordinate-invariant.
:::

:::{figure} figures/adaptive-sprinkling.svg
:alt: Comparison of uniform and adaptive sprinkling densities.
:width: 95%
:align: center

Uniform versus adaptive sprinkling. Adaptive density concentrates episodes where the geometry and QSD weight are high.
:::

:::{dropdown} 📖 Hypostructure Reference
:icon: book

**Rigor Class:** L (Literature-Imported)

**Permits:** $\mathrm{TB}_\rho$ (Node 10 ErgoCheck)

**Hypostructure connection:** The QSD spatial marginal is established in the meta-learning framework via the Expansion Adjunction ({prf:ref}`thm-expansion-adjunction`). The adaptive density emerges from the Stratonovich interpretation of the SDE ({prf:ref}`def-fractal-set-sde`), which preserves the Riemannian volume measure $\sqrt{\det g} \, dx$.

**References:**
- State space definition: {prf:ref}`def:state-space-fg`
- Algorithmic space: {prf:ref}`def-algorithmic-space-generic`
:::

### Framework Lift: CST Operators Are Computable

:::{prf:proposition} Framework Lift for CST Operators
:label: prop-fractal-cst-framework-lift

By lossless reconstruction ({prf:ref}`thm-fractal-set-lossless`), all quantities in the
reconstruction target set ({prf:ref}`def-fractal-set-reconstruction-targets`) are recoverable
at episode locations, including trajectories and the adaptive diffusion tensor
$\Sigma_{\mathrm{reg}}$ from the SDE ({prf:ref}`def-fractal-set-sde`). The emergent metric is
defined by $\Sigma_{\mathrm{reg}}$ via {prf:ref}`def-adaptive-diffusion-tensor-latent`, so $g$
and the spatial geodesic distance $d_{g_R}$ (hence the spacetime proxy $d_{\mathrm{geo}}$) are
available on the same support. Expansion Adjunction
({prf:ref}`thm-expansion-adjunction`) and Lock Closure
({prf:ref}`mt:fractal-gas-lock-closure`) lift the discrete causal order to the continuum
limit. Therefore the adaptive-density CST operators in this chapter are well-defined and
directly computable on the Fractal Set.
:::

:::{seealso}
:class: feynman-added
- {doc}`/source/3_fractal_gas/3_fitness_manifold/01_emergent_geometry`: Emergent metric from adaptive diffusion
- {doc}`/source/3_fractal_gas/3_fitness_manifold/02_scutoid_spacetime`: Scutoid tessellation and plaquette structure
- {doc}`/source/3_fractal_gas/3_fitness_manifold/03_curvature_gravity`: Curvature from discrete holonomy
- {doc}`/source/3_fractal_gas/3_fitness_manifold/04_field_equations`: Field equations and pressure dynamics
:::

---

(sec-axiom-verification)=
## Verification of Causal Set Axioms

:::{div} feynman-prose
Now we come to the heart of the matter: proving that the Fractal Set actually *is* a causal set. This is not just a formal exercise. If we can prove this, then decades of mathematical machinery developed by quantum gravity researchers becomes available to us. All the tools for extracting geometry from causal structure—dimension estimators, curvature measures, wave equations—can be applied using the adaptive-density formulas given below.

The proof is surprisingly simple, which is always a good sign. The three axioms—irreflexivity, transitivity, and local finiteness—each follow from basic properties of time and space. Let me give you the intuition before we dive into the formalism.

**Irreflexivity** says you cannot be your own cause. Why is this true? Because the causal order requires $t_i < t_j$—strict inequality. An event cannot happen before itself. This is so obvious it almost seems silly to state, but that is the point: the axioms of causality are just formal statements of things that are obviously true about time.

**Transitivity** says that if A causes B and B causes C, then A causes C. This follows from chaining causal steps: a causal path from A to B concatenated with a causal path from B to C is a causal path from A to C.

**Local finiteness** says that only finitely many events can happen between any two events. This is where discreteness enters. Between A and C, only a finite set of discrete timesteps can occur, and each timestep contains finitely many episodes. Done.

You see how the structure of spacetime—time ordering, spatial distances, bounded regions—automatically gives you causality. The math is just making explicit what is already implicit in the physics.
:::

:::{prf:theorem} Fractal Set is a Valid Causal Set
:label: thm-fractal-is-causal-set

The Fractal Set $\mathcal{F} = (E, \prec_{\mathrm{CST}})$ satisfies all BLMS axioms.

**Rigor Class:** F (Framework-Original)

**Permits:** $\mathrm{TB}_\pi$ (Node 8), $\mathrm{TB}_O$ (Node 9)
:::

:::{prf:proof}
We verify each axiom:

**Axiom CS1 (Irreflexivity):** If $e_i \prec_{\mathrm{CST}} e_i$, there is a directed CST path from $e_i$ to
itself. But each CST edge strictly increases time, so no directed path can return to its
starting node. Hence $e_i \not\prec_{\mathrm{CST}} e_i$. ✓

**Axiom CS2 (Transitivity):** If $e_1 \prec_{\mathrm{CST}} e_2$ and $e_2 \prec_{\mathrm{CST}} e_3$, then there are CST paths
from $e_1$ to $e_2$ and from $e_2$ to $e_3$. Concatenating them gives a CST path from $e_1$ to
$e_3$, so $e_1 \prec_{\mathrm{CST}} e_3$. ✓

**Axiom CS3 (Local Finiteness):** If $e_1 \prec_{\mathrm{CST}} e_2$, then $t_1 < t_2$. Any episode with
$e_1 \prec_{\mathrm{CST}} e \prec_{\mathrm{CST}} e_2$ must lie at a timestep in $\{t_1+1, \ldots, t_2-1\}$, and each timestep
contains finitely many episodes (bounded by the walker count). Hence $|I(e_1, e_2)| < \infty$. ✓

$\square$
:::

:::{dropdown} 📖 ZFC Proof (Classical Verification)
:icon: book

**Classical Verification (ZFC):**

Working in Grothendieck universe $\mathcal{U}$, the Fractal Set $(E, \prec_{\mathrm{CST}})$ is a finite partially ordered set (poset) where:

1. $E \in V_\mathcal{U}$ is a finite set of episodes
2. $\prec_{\mathrm{CST}} \subseteq E \times E$ is a binary relation

**CS1 (Irreflexivity):** $\prec_{\mathrm{CST}}$ is the transitive closure of CST edges, each of
which increases time. Thus no directed path can return to its start, so $e_i \not\prec_{\mathrm{CST}} e_i$.

**CS2 (Transitivity):** By construction, the transitive closure of any relation is transitive.

**CS3 (Local Finiteness):** If $e_1 \prec_{\mathrm{CST}} e_2$, then $t_1 < t_2$, and any $e$ with
$e_1 \prec_{\mathrm{CST}} e \prec_{\mathrm{CST}} e_2$ lies on a timestep between $t_1$ and $t_2$. Each timestep has finitely
many episodes (bounded by the walker count), so the interval is finite.

All axioms verified using only ZFC set theory and real analysis. $\square$
:::

---

(sec-faithful-discretization)=
## Faithful Discretization and Manifoldlikeness

:::{div} feynman-prose
Proving the Fractal Set is a causal set is only half the battle. The other half is showing that it is a *good* discretization of the underlying continuous spacetime. This is the "manifoldlikeness" question: does the discrete structure look like a smooth manifold when you zoom out?

The key idea is *volume matching*. If you have a faithful discretization, the number of points in any region should be proportional to the volume of that region. More precisely, if you count the episodes in some domain $\Omega$, divide by the total number of episodes, you should get something close to the volume fraction $V(\Omega)/V_{\mathrm{total}}$.

But here is where the adaptive density makes things interesting. In ordinary Poisson sprinkling, the number of points is proportional to the *coordinate* volume. In the Fractal Set, the number of points is proportional to the *weighted* volume—the integral of $\sqrt{\det g} \cdot e^{-U_{\mathrm{eff}}/T}$. This means more points in regions where the metric is stretched (high $\sqrt{\det g}$) and where the effective potential is low (high $e^{-U_{\mathrm{eff}}/T}$).

Why is this better? Because it automatically concentrates resolution where it matters. If you are trying to compute geometric quantities—curvature, for instance—you need more samples in regions where the geometry varies rapidly. The adaptive density provides exactly this. It is as if the discretization has "learned" where to pay attention.

The Myrheim-Meyer dimension estimator is a beautiful piece of mathematical machinery that extracts the dimension of spacetime from pure causal order {cite}`Myrheim1978,Meyer1988`. You count the fraction of pairs that are causally related, and this fraction depends on the dimension. In two dimensions, more pairs are causally related than in four dimensions—because light cones are "wider" in lower dimensions. The Fractal Set inherits this machinery: count the ordering fraction, and you recover the dimension.
:::

### Volume Matching

:::{prf:theorem} Fractal Set Provides Faithful Discretization
:label: thm-fractal-faithful-embedding

The Fractal Set faithfully discretizes the emergent Riemannian manifold $(\mathcal{X}, g)$
with respect to the QSD-weighted measure defined by the Adaptive Gas dynamics
({prf:ref}`def-fractal-set-sde`):

**Volume Matching**: For a half-open time window $[t_0, t_1)$, let $E$ be the set of **alive
episodes** ($s(n)=1$) with $t\in[t_0,t_1)$ and $N:=|E|$. Condition on $N$ episodes and rate $r(t)$;
the episode count in a spatial region $\Omega$ satisfies:

$$
\mathbb{E}\left[\frac{|E \cap ([t_0,t_1)\times \Omega)|}{N}\right]
= \frac{1}{R} \int_{t_0}^{t_1} \frac{r(t)}{Z(t)}
\int_{\Omega} \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T} \, dx \, dt
$$

with $R = \int_{t_0}^{t_1} r(t)\, dt$ (for discrete steps, define
$r(t_k):=|E_{\mathrm{CST}}(t_k)|/\Delta t_k$ with $E_{\mathrm{CST}}(t_k)$ the set of CST edges
starting at $t_k$; then $R=\sum_k r(t_k)\Delta t_k=\sum_k |E_{\mathrm{CST}}(t_k)|=N$).
The expectation is conditional on $N$ (or asymptotic as $N \to \infty$). Variance scales as
$O(1/N)$ under standard mixing/ergodicity. (To recover
geometric spacetime volume
$\int_{t_0}^{t_1}\!\int_\Omega \sqrt{\det g(x,t)} \, dx \, dt$, reweight by
$e^{U_{\mathrm{eff}}(x,t)/T} \, Z(t) / r(t)$ and multiply by $R$.)

**Distance Estimation**: Let $L_{\mathrm{chain}}(e_i,e_j)$ be the length (number of CST edges)
of the longest chain from $e_i$ to $e_j$, and define the algorithmic time separation
$\tau_{\mathrm{alg}}(e_i,e_j):=\sum_{e\in\text{chain}}\Delta t(e)$ (so $\tau_{\mathrm{alg}} =
L_{\mathrm{chain}}\Delta t$ when $\Delta t$ is constant). Fix a calibration constant
$\kappa_\tau>0$ relating Lorentzian proper time to algorithmic time, i.e.
$\tau_{\mathrm{prop}}=\kappa_\tau\,\tau_{\mathrm{alg}}$. Then timelike distances are estimated by
$\kappa_\tau\,\tau_{\mathrm{alg}}$; in particular, if algorithmic time is chosen to equal proper
time, set $\kappa_\tau=1$ to recover the usual longest-chain estimate. Spatial distances follow
from reconstructed trajectories and IG-edge geometry ({prf:ref}`thm-fractal-set-trajectory`,
{prf:ref}`def-fractal-set-ig-edges`).

**Dimension Estimation**: The Myrheim-Meyer estimator converges when computed with the
geometric light-cone order $\prec_{\mathrm{LC}}$ and the adaptive-density correction described
below; an equivalent implementation is to compute it on local windows of approximately
constant density:

$$
d_{\mathrm{MM}} \xrightarrow{N \to \infty} D = \dim \mathcal{X} + 1
$$
:::

### Advantages over Poisson Sprinkling

:::{prf:proposition} Adaptive Sprinkling Improves Geometric Fidelity
:label: prop-adaptive-vs-poisson

Compared to uniform Poisson sprinkling, the Fractal Set achieves:

1. **Better coverage**: Episodes concentrate where the QSD weight is higher (large $\sqrt{\det g}$ or low $U_{\mathrm{eff}}$)

2. **Lower variance for QSD-weighted observables**: Estimates aligned with the adaptive measure are more sample-efficient than uniform sprinkling

3. **Automatic adaptation**: No ad-hoc density choices; $\rho$ emerges from QSD
:::

---

(sec-cst-machinery)=
## Causal Set Mathematical Machinery

:::{div} feynman-prose
Now comes the payoff. Having established that the Fractal Set is a valid causal set, we inherit the toolbox that causal set theorists have developed over the past four decades, now specialized to the adaptive density. These are not just abstract mathematical constructions—they are the discrete versions of the most important objects in physics: volume, the wave equation, dimension, curvature.

Let me give you the philosophy first. In continuum physics, we are used to differential operators—gradients, Laplacians, d'Alembertians. These are defined in terms of infinitesimal limits: "how does the field change as you move an infinitesimally small distance?" But on a causal set, there are no infinitesimals. Space is grainy. So how do you define a derivative?

The answer is: you use *nonlocal* operators that average over many nearby points. Benincasa and
Dowker showed that for uniform Poisson sprinkling, a carefully weighted nonlocal sum converges to
the ordinary d'Alembertian. Here we construct the QSD-weighted operator generated by the Fractal
Gas itself. It is covariant by construction and reduces to Benincasa-Dowker in the uniform limit.

This is genuinely remarkable. You start with a completely discrete structure—just a locally finite set of points with an ordering—and you can compute the wave equation, the curvature, the dimension. All of classical and quantum field theory on curved spacetime becomes accessible, not despite the discreteness but *through* it.

The dimension estimator is my favorite example. It counts the fraction of pairs that are
causally related in the geometric order $\prec_{\mathrm{LC}}$. Once that order is fixed by the
emergent metric and speed cap, it is just counting—no further coordinate or tangent data are
needed. And yet from this single number, you can read off the dimension of spacetime.
:::

With the Fractal Set established as a valid causal set, standard CST tools apply in their
adaptive-density form, and all required inputs are reconstructible from the framework. We use
$d = \dim \mathcal{X}$ for spatial slices and $D = d + 1$ for the spacetime dimension in CST
formulas.

### Causal Set Volume Element

:::{prf:definition} Causal Set Volume Fraction (Adaptive Measure, LC Order)
:label: def-cst-volume

Let $r(t)$ be the episode rate and $\rho_{\mathrm{adaptive}}(x, t)$ the instantaneous spatial
QSD marginal. Define the spacetime measure
$d\mu_{\mathrm{adaptive}}(t, x) := r(t)\,\rho_{\mathrm{adaptive}}(x, t)\, dt\, dx$ and let
$E$ denote the **alive episodes** ($s(n)=1$) in the chosen half-open time window $[t_0, t_1)$
with $N = |E|$. Define the geometric (light-cone) past
$J^-_{\mathrm{LC}}(e) := \{e' \in E : e' \prec_{\mathrm{LC}} e\}$ and use the same notation for
its continuum counterpart in $M$. The **adaptive causal set volume fraction** of $e \in E$ is:

$$
V_{\mathrm{adaptive}}(e) := \frac{1}{N} \sum_{e' \in E} \mathbb{1}_{e' \prec_{\mathrm{LC}} e}
$$

This is normalized by $N$, so $0 \le V_{\mathrm{adaptive}}(e) \le 1$. The unnormalized adaptive
volume is $\mu_{\mathrm{adaptive}}(J^-_{\mathrm{LC}}(e))$, and in the continuum limit
$\mu_{\mathrm{adaptive}}(J^-_{\mathrm{LC}}(e)) = R\,V_{\mathrm{adaptive}}(e)$ with

$$
\mu_{\mathrm{adaptive}}(J^-_{\mathrm{LC}}(e)) = \int_{J^-_{\mathrm{LC}}(e)} \frac{r(t)}{Z(t)} \sqrt{\det g(x, t)}
\, e^{-U_{\mathrm{eff}}(x, t)/T} \, dt \, dx, \quad R = \int r(t)\, dt .
$$
For discrete steps, define $r(t_k):=|E_{\mathrm{CST}}(t_k)|/\Delta t_k$ with $E_{\mathrm{CST}}(t_k)$
the set of CST edges starting at $t_k$. Each alive episode at $t_k$ has exactly one outgoing
CST edge, so $R=\sum_k r(t_k)\Delta t_k=\sum_k |E_{\mathrm{CST}}(t_k)|=N$ for $[t_0,t_1)$.

**Geometric volume recovery**: Reweight by the Boltzmann factor to undo the QSD bias:

$$
V_g(e) := \frac{1}{N} \sum_{e' \in E,\, e' \prec_{\mathrm{LC}} e}
\frac{Z(t_{e'})}{r(t_{e'})} \exp\!\left(\frac{U_{\mathrm{eff}}(x_{e'}, t_{e'})}{T}\right),
\quad \mathbb{E}[V_g(e)] = \frac{1}{R} \int_{J^-_{\mathrm{LC}}(e)} \sqrt{\det g(x, t)} \, dt \, dx .
$$
This is likewise a normalized geometric volume fraction; multiply by $R$ to recover the
geometric volume.
:::

### Fractal Gas Nonlocal d'Alembertian and Action (Full-Rigor)

:::{div} feynman-prose
The d'Alembertian $\Box_g = g^{\mu\nu}\nabla_\mu\nabla_\nu$ is the wave operator—it tells you how waves propagate through spacetime. With signature $(-,+,\ldots,+)$ and $g=-c^2 dt^2 + g_R$, this is $\Box_g = -c^{-2}\partial_t^2 + \Delta_{g_R}$ up to the usual lower-order terms when $g_R$ depends on $t$. In the continuum, it is defined using second derivatives. But on a discrete set, there are no derivatives. How do you take a derivative when there is no notion of "infinitesimally close"?

Our answer is to build the operator from the **process itself**. We already know the sampling law: episodes follow the QSD with density proportional to $\sqrt{\det g}\,e^{-U_{\mathrm{eff}}/T}$. We already know the causal structure: CST edges give the order, and the algorithm enforces a speed limit $c = V_{\mathrm{alg}}$. So we can define a nonlocal causal operator whose expectation is taken with respect to the QSD-weighted measure, and whose kernel is chosen to match the continuum d'Alembertian.

This avoids any reliance on Poisson sprinkling. The operator is covariant by construction, because it is built from the emergent metric and the QSD measure that the dynamics actually produces.
:::

:::{prf:assumption} Fractal Gas Continuum Hypotheses
:label: assm-fractal-gas-nonlocal

We assume:

**A1 (Geometry)**: The continuum lift is a globally hyperbolic spacetime
$M=[t_0,t_1]\times \mathcal{X}$ with Lorentzian metric
$g = -c^2 dt^2 + g_R$, where $c=V_{\mathrm{alg}}$ and
$g_R$ is a $C^4$ Riemannian metric on $\mathcal{X}$ with uniform ellipticity.

**A2 (Smooth fields)**: $U_{\mathrm{eff}}(x,t)$, $r(t)$, $Z(t)$, and $g_R(x,t)$ are $C^4$ and
bounded with bounded derivatives up to order 4 on the window, with $r(t), Z(t)$ bounded away
from $0$.

**A3 (QSD sampling)**: The episode process is time-inhomogeneous with intensity
$\lambda(t,x)=r(t)\rho_{\mathrm{adaptive}}(x,t)$, and conditional on each time slice $t$ the
spatial law is QSD with density
$\rho_{\mathrm{adaptive}}(x,t) \propto \sqrt{\det g_R(x,t)}\,e^{-U_{\mathrm{eff}}(x,t)/T}$.
We assume slice-wise ergodicity on the window.

**A4 (Mixing)**: The conditional episode process on each time slice satisfies an LSI with
constant $\kappa>0$ uniform in $t$, implying exponential mixing and a law of large numbers
for bounded Lipschitz functionals uniformly over the window.

**A5 (Kernel + algorithmic cutoff)**: $K\in C^2_c([0,1])$. Let $\rho$ be the localization
scale and $\varepsilon_c$ the coherence scale ({doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set`). Define the algorithmic
locality radius $R_{\mathrm{loc}} := \min(\rho,\varepsilon_c)$ and the light-crossing time
$T_{\mathrm{loc}} := \min(t_1-t_0, R_{\mathrm{loc}}/c)$. For $\varepsilon>0$, let

$$
J_{\mathrm{alg}}^{(\varepsilon)} := \{\xi:\tau(\xi)\in[0,\varepsilon],\; |\xi^0|\le T_{\mathrm{loc}},\; \|\xi\|\le R_{\mathrm{loc}}\}
$$
be the algorithmic double cone in tangent Minkowski space, with
$\tau(\xi):=\sqrt{c^2(\xi^0)^2-\|\xi\|^2}$ and $c=V_{\mathrm{alg}}$. For the rescaling step
we work in units with $c=1$ (equivalently rescale the time coordinate by $c$) so that all
components of $\xi$ have length units. Then set
$\xi=\varepsilon \zeta$, the unit cone

$$
\widehat{J}_{\mathrm{alg}} := \{\zeta:\tau(\zeta)\in[0,1],\; |\zeta^0|\le T_{\mathrm{loc}}/\varepsilon,\; \|\zeta\|\le R_{\mathrm{loc}}/\varepsilon\}
$$
is fixed once we identify $\varepsilon := R_{\mathrm{loc}}$ (A6), so the moment conditions are
imposed on $\widehat{J}_{\mathrm{alg}}$ with the scaled kernel $K(\tau(\zeta))$.
For $\varepsilon$ small enough that $R_{\mathrm{loc}}/c \le t_1-t_0$, we have
$T_{\mathrm{loc}}=R_{\mathrm{loc}}/c$ and thus $|\zeta^0|\le 1/c$ (i.e., $|\zeta^0|\le 1$ in
$c=1$ units), $\|\zeta\|\le 1$.
The moment conditions are:

$$
M_0 := \int_{\widehat{J}_{\mathrm{alg}}} K(\tau(\zeta))\,d\zeta = 0,\qquad
M_2^{\mu\nu} := \int_{\widehat{J}_{\mathrm{alg}}} K(\tau(\zeta))\,\zeta^\mu\zeta^\nu\,d\zeta = 2m_2\, g^{\mu\nu},
$$
with $m_2>0$. The cutoff is symmetric, so odd moments vanish. (These conditions can be enforced
by choosing $K$ with signed weights.)

**A6 (Scaling)**: $\varepsilon\to 0$, $N\to\infty$, and $N\varepsilon^{D+4}\to\infty$.
:::

### Discharging the Fractal Gas Continuum Hypotheses (Implementation Plan)

This section records the concrete, Volume 3-only steps that replace A1-A6 with
internal certificates and metatheorems. No algorithm changes and no new
assumptions are introduced; each item names the exact proof objects to cite.

#### A1 (Geometry): globally hyperbolic Lorentzian manifold

1. **Continuum lift from QSD sampling**: Use propagation of chaos
   ({prf:ref}`thm-propagation-chaos-qsd`) plus the mean-field PDE
   ({prf:ref}`thm-mean-field-equation`) to obtain a deterministic continuum limit
   on spatial slices, with a time parameter inherited from episode ordering.
2. **Induced spatial metric**: Use the emergent metric from the adaptive diffusion
   tensor ({prf:ref}`def-adaptive-diffusion-tensor-latent`) and the induced
   Riemannian structure certificate ({prf:ref}`thm:induced-riemannian-structure`)
   to define $g_R$ on each slice. Uniform ellipticity is guaranteed by the
   diffusion floor $\epsilon_\Sigma$ (see {prf:ref}`def-adaptive-diffusion-tensor-latent`).
3. **Continuum injection and Cheeger gradient**: Invoke the metatheorems
   {prf:ref}`mt:continuum-injection`, {prf:ref}`mt:emergent-continuum`, and
   {prf:ref}`mt:cheeger-gradient` to promote the discrete IG geometry to a
   $C^2$ Riemannian metric compatible with the algorithmic distance.
4. **Lorentzian signature from causal order**: Use CST order and the algorithmic
   time function $t(e)$ to define the product metric
   $g=-c^2 dt^2 + g_R$ with $c=V_{\mathrm{alg}}$, and verify that CST edges
   are timelike while IG edges are spacelike by construction of the speed limit
   (Sections 1-2 of this chapter).
5. **Global hyperbolicity**: Show causal diamonds are compact on the window
   $[t_0,t_1]$ using the confining envelope from the decorated Gibbs structure
   ({prf:ref}`thm-decorated-gibbs` in {doc}`/source/3_fractal_gas/appendices/07_discrete_qsd`) and the Safe Harbor
   barrier mechanisms ({doc}`/source/3_fractal_gas/appendices/03_cloning`, {doc}`/source/3_fractal_gas/appendices/07_discrete_qsd`). This yields a
   Cauchy foliation by constant-$t$ slices.

#### A2 (Smooth fields): $U_{\mathrm{eff}}$, $r(t)$, $Z(t)$, $g_R$

1. **$U_{\mathrm{eff}}$ regularity**: Express $U_{\mathrm{eff}}$ via the
   mean-field fitness potential and decorated Gibbs envelope
   ({doc}`/source/3_fractal_gas/appendices/07_discrete_qsd`, {prf:ref}`thm-decorated-gibbs`). Use the fitness
   pipeline smoothness certificate in {doc}`../1_the_algorithm/02_fractal_gas_latent`
   (composition of $C^2$ primitives + Gaussian smoothing) and hypoelliptic
   regularity from {doc}`/source/3_fractal_gas/appendices/09_propagation_chaos` to upgrade to $C^4$ on the alive core.
2. **$g_R$ smoothness**: Start from the adaptive diffusion tensor
   ({prf:ref}`def-adaptive-diffusion-tensor-latent`) and the Lipschitz continuity
   result ({prf:ref}`prop-lipschitz-diffusion-latent` in
   {doc}`../3_fitness_manifold/01_emergent_geometry`). Combine with the
   $C^4$ regularity of $V_{\mathrm{fit}}$ (from Step 1) to lift $g_R$ to $C^4$.
3. **$r(t)$ and $Z(t)$ smoothness**: Define $r(t)$ and $Z(t)$ as time-marginals of
   the QSD density (Section 2.2 here). Use mass conservation
   ({prf:ref}`thm-mass-conservation` in {doc}`/source/3_fractal_gas/appendices/08_mean_field`) and differentiation
   under the integral sign with $C^4$ density to show $r, Z \in C^4([t_0,t_1])$.
4. **Uniform derivative bounds**: Use the confining envelope and bounded core
   (Safe Harbor + decorated Gibbs) to bound derivatives uniformly on the window.

#### A3 (QSD sampling): stationarity and density form

1. **Existence/uniqueness**: Use {doc}`/source/3_fractal_gas/appendices/06_convergence` (finite-$N$ QSD) plus
   {doc}`/source/3_fractal_gas/appendices/09_propagation_chaos` (mean-field limit) to identify the QSD density
   $\rho_{\mathrm{adaptive}}(\cdot,t)$ on each time slice.
2. **Gibbs structure**: Apply {prf:ref}`thm-decorated-gibbs` to write
   $\rho_0(x) \propto e^{-U_{\mathrm{eff}}/T_{\mathrm{sys}}}\,\Xi(x)$ and match
   the QSD density used in the volume measure reweighting
   ({prf:ref}`def-cst-volume`).
3. **Ergodicity**: Combine exchangeability and uniqueness
   ({prf:ref}`thm-qsd-exchangeability` in {doc}`/source/3_fractal_gas/appendices/12_qsd_exchangeability_theory`)
   with LSI mixing (A4) to obtain slice-wise ergodic sampling on the window.

#### A4 (Mixing): LSI and concentration

1. **N-uniform LSI**: Use {prf:ref}`thm-n-uniform-lsi-exchangeable`
   ({doc}`/source/3_fractal_gas/appendices/12_qsd_exchangeability_theory`) and the KL convergence proof
   ({prf:ref}`thm-kl-convergence-euclidean` in {doc}`/source/3_fractal_gas/appendices/15_kl_convergence`) to
   obtain an LSI constant independent of $N$ on the alive core.
2. **Hypocoercive smoothing**: Use {doc}`/source/3_fractal_gas/appendices/10_kl_hypocoercive` to pass LSI from
   the discrete chain to the continuous-time generator.
3. **LLN and exponential mixing**: Apply the standard LSI $\Rightarrow$
   hypercontractivity $\Rightarrow$ exponential decay of correlations pathway
   to justify the QSD-weighted law of large numbers invoked in the d'Alembertian
   consistency proof (Step 1 of {prf:ref}`thm-cst-fractal-dalembertian-consistency`).

#### A5 (Kernel): moment conditions for $K$

1. **Explicit construction**: Choose a $C^2$ bump $\phi \in C^2_c([0,1])$ and set
   $K(s)=a\,\phi(s)+b\,s^2\phi(s)$. Solve the two linear constraints
   $M_0=0$ and $M_2^{\mu\nu}=2m_2 g^{\mu\nu}$ for $(a,b)$, guaranteeing $m_2>0$.
2. **Symmetry check**: The cutoff is invariant under spatial rotations and time
   reversal, so odd moments vanish and $M_2^{\mu\nu}$ is diagonal with equal
   spatial components; $K$ is then chosen to match $2m_2 g^{\mu\nu}$.
3. **Kernel deployment**: Record $(a,b)$ and $m_2$ as analytic constants of the
   estimator; no algorithmic change is required because $K$ appears only in the
   analytic reconstruction operator.

#### A6 (Scaling): $\varepsilon \to 0$, $N\to\infty$, $N\varepsilon^{D+4}\to\infty$

1. **Bias/variance balance**: From
   {prf:ref}`thm-cst-fractal-dalembertian-consistency`,
   $\text{Bias}=O(\varepsilon^2)$ and $\text{Var}=O((N\varepsilon^{D+2})^{-1})$.
   The condition $N\varepsilon^{D+4}\to\infty$ forces both terms to vanish.
2. **Sampling density from QSD**: Use the full-support guarantee and mean-field
   error certificate in {doc}`../1_the_algorithm/02_fractal_gas_latent` to express
   the effective sampling radius $\varepsilon(N)$ as an explicit function of $N$
   (invert the $O(N^{-1/2})$ mean-field rate). This yields $\varepsilon(N)\to 0$.
3. **Parameter identification**: Set $\varepsilon := R_{\mathrm{loc}}=\min(\rho,\varepsilon_c)$
   using the localization/coherence scales from the Fractal Set parameters
   ({doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set`), and take the continuum limit with $\rho,\varepsilon_c \to 0$
   while keeping macroscopic scales fixed.

:::{prf:definition} Interior Episodes and Boundary Bias
:label: def-fractal-gas-interior-episodes

Let $\mathcal{X}_{\mathrm{core}}\subset\mathcal{X}$ be the compact alive core guaranteed by the
Safe Harbor/confining envelope (Section 2; {doc}`/source/3_fractal_gas/appendices/07_discrete_qsd`). For time-dependent metrics,
write $\mathrm{dist}_{g_{R,t}}(x,\partial\mathcal{X}_{\mathrm{core}})$ for the slice-wise
geodesic boundary distance induced by $g_R(\cdot,t)$ (set to $+\infty$ if
$\partial\mathcal{X}_{\mathrm{core}}=\varnothing$). Define

$$
E_{\mathrm{int}} := \{e=(x_e,t_e)\in E:\; t_e\in[t_0+T_{\mathrm{loc}},\,t_1-T_{\mathrm{loc}}),\;
\mathrm{dist}_{g_{R,t_e}}(x_e,\partial\mathcal{X}_{\mathrm{core}})\ge R_{\mathrm{loc}}\},
$$

and $E_{\mathrm{bdy}}:=E\setminus E_{\mathrm{int}}$. For Lipschitz boundaries,

$$
\frac{|E_{\mathrm{bdy}}|}{N}=O\!\left(\frac{T_{\mathrm{loc}}}{t_1-t_0}+\frac{R_{\mathrm{loc}}}{L_{\mathrm{core}}}\right),
\quad L_{\mathrm{core}}:=\sup_{t\in[t_0,t_1]}\mathrm{diam}_{g_{R,t}}(\mathcal{X}_{\mathrm{core}}),
$$
so boundary bias in normalized sums is $O(T_{\mathrm{loc}}+R_{\mathrm{loc}})=O(\varepsilon)$ under A6.
All localized sums and estimators below are defined for $e\in E_{\mathrm{int}}$.
:::

:::{prf:definition} Fractal Gas d'Alembertian (QSD-Weighted Nonlocal Operator)
:label: def-cst-fractal-dalembertian

Let $E_{\mathrm{int}}$ be the interior episodes from {prf:ref}`def-fractal-gas-interior-episodes`.
For $e\in E_{\mathrm{int}}$ and episodes $e'$ with $e'\prec_{\mathrm{LC}} e$ or
$e\prec_{\mathrm{LC}} e'$, define the proper-time proxy

$$
\tau(e', e) := \sqrt{c^2 (t_e - t_{e'})^2 - d_{\mathrm{geo}}(e', e)^2}, \qquad c := V_{\mathrm{alg}}.
$$

Let $w_{\mathrm{geo}}(e') := \frac{Z(t_{e'})}{r(t_{e'})}
\exp\!\left(\frac{U_{\mathrm{eff}}(x_{e'}, t_{e'})}{T}\right)$
be the geometric reweighting from {prf:ref}`def-cst-volume`.
Let $R:=\int_{t_0}^{t_1} r(t)\,dt$ (for half-open windows, $R=N$).
Define the **localized** two-sided causal neighborhood

$$
J_{\mathrm{loc}}^\pm(e):=\{e' \in E:\, e'\prec_{\mathrm{LC}} e \text{ or } e\prec_{\mathrm{LC}} e',\;
|t_e-t_{e'}|\le T_{\mathrm{loc}},\; d_{\mathrm{geo}}(e',e)\le R_{\mathrm{loc}}\},
$$
using the algorithmic cutoffs $T_{\mathrm{loc}}, R_{\mathrm{loc}}$ from A5. This removes
far light-cone contributions and makes the kernel moments finite by construction. In practice,
$d_{\mathrm{geo}}$ is computed from the reconstructed $g_R$ or via IG-graph shortest paths
with edge lengths induced by $g_R$.

The **Fractal Gas d'Alembertian** acting on $f:E\to\mathbb{R}$ is

$$
(\Box_{\mathrm{FG}} f)(e) := \frac{1}{m_2\,\varepsilon^{D+2}} \cdot \frac{R}{N}
\sum_{e' \in J_{\mathrm{loc}}^\pm(e)} w_{\mathrm{geo}}(e')\,K\!\left(\frac{\tau(e', e)}{\varepsilon}\right)
\bigl(f(e')-f(e)\bigr).
$$

This operator is covariant under coordinate changes (all quantities are geometric) and vanishes on constants.
:::

:::{prf:definition} Fractal Gas Scalar Action
:label: def-cst-fractal-action

Let $\mu_{\mathrm{geo}}(e) := \frac{R}{N} w_{\mathrm{geo}}(e)$ be the discrete geometric volume weight
(so $\mu_{\mathrm{geo}}=w_{\mathrm{geo}}$ on half-open windows).
The **Fractal Gas scalar action** for a field $f$ is

$$
S_{\mathrm{FG}}[f] := \frac{1}{2}\sum_{e\in E_{\mathrm{int}}} \mu_{\mathrm{geo}}(e)\, f(e)\, (\Box_{\mathrm{FG}} f)(e).
$$
:::

:::{prf:theorem} Continuum Consistency (Fractal Gas d'Alembertian)
:label: thm-cst-fractal-dalembertian-consistency

Assume {prf:ref}`assm-fractal-gas-nonlocal`. Let $f\in C^4_c(M)$. Then for each episode
$e=(x,t)\in E_{\mathrm{int}}$,

1. **Bias**:

$$
\mathbb{E}\big[(\Box_{\mathrm{FG}} f)(e)\,\big|\,e=(x,t)\big]
= \Box_g f(x,t) + O(\varepsilon^2).
$$

2. **Variance**:

$$
\mathrm{Var}\big[(\Box_{\mathrm{FG}} f)(e)\,\big|\,e\big] \le \frac{C}{N\,\varepsilon^{D+2}}
$$
for a constant $C$ depending on $K$, $f$, and the window.

3. **Consistency**: Under scaling A6, $(\Box_{\mathrm{FG}} f)(e)\to \Box_g f(x,t)$ in probability.

4. **Action limit**:

$$
S_{\mathrm{FG}}[f] \;\xrightarrow[N\to\infty]{\varepsilon\to 0}\;
\frac{1}{2}\int_M f\,\Box_g f \, d\mathrm{vol}_g .
$$

*Proof.*

**Step 1 (QSD-weighted LLN).**
By {prf:ref}`def-cst-volume`, the reweighting $w_{\mathrm{geo}}$ converts the QSD marginal to the
geometric measure $d\mathrm{vol}_g$. Under A3-A4, conditional on each time slice $t$, empirical
QSD-weighted sums of bounded Lipschitz functions converge in mean and probability to their
geometric expectations uniformly over the window. Therefore, conditional on $e=(x,t)$ (and $N$),
the prefactor $R/N$ converts the QSD-weighted discrete sum into the corresponding geometric
integral over $J_{\mathrm{loc}}^\pm(e)$ up to $O(N^{-1/2})$ fluctuations. Boundary-layer
contributions are $O(\varepsilon)$ by {prf:ref}`def-fractal-gas-interior-episodes`.

**Step 2 (Local expansion).**
Work in normal coordinates for $g$ at $(x,t)$ and write $\xi^\mu$ for the coordinate difference.
Expand

$$
f(x+\xi) = f(x) + \partial_\mu f(x)\,\xi^\mu + \frac{1}{2}\partial_\mu\partial_\nu f(x)\,\xi^\mu\xi^\nu
 + O(\|\xi\|^3).
$$
Because the kernel is two-sided and $K$ depends only on $\tau(\xi)/\varepsilon$, the odd moments vanish.
Using A5 and the change of variables $\xi=\varepsilon\zeta$, the second-moment term yields

$$
\frac{1}{m_2\varepsilon^{D+2}} \int_{J_{\mathrm{alg}}^{(\varepsilon)}} K(\tau(\xi)/\varepsilon)\,
\frac{1}{2}\partial_\mu\partial_\nu f(x)\,\xi^\mu\xi^\nu\, d\xi
 = \Box_g f(x) + O(\varepsilon^2),
$$
establishing the bias statement.

**Step 3 (Variance).**
The summand is bounded by $\|K\|_\infty \|f\|_{C^1}\varepsilon^{-D-1}$ on its support.
Mixing from A4 implies concentration of empirical averages, giving
$\mathrm{Var} = O((N\varepsilon^{D+2})^{-1})$.

**Step 4 (Consistency and action limit).**
Combining Steps 1-3 with A6 gives convergence in probability of $\Box_{\mathrm{FG}} f$.
The action limit follows by dominated convergence and the boundedness of $\mu_{\mathrm{geo}}$.
$\square$
:::

:::{note}
In the uniform Poisson limit with constant density and a kernel that reproduces the Benincasa-Dowker
coefficients, $\Box_{\mathrm{FG}}$ reduces to the classical $\Box_{\mathrm{BD}}$ operator.
:::

### Dimension and Curvature Estimation

:::{div} feynman-prose
Here is something that should make you sit up. We are going to measure the dimension of spacetime—an integer!—by counting pairs and computing a fraction. Once the geometric causal order is fixed, it is just counting—no further coordinates, metrics, or calculus are needed.

The idea is simple but profound. Take any two points in a causal set and ask: are they causally related? That is, is one in the past of the other? In some pairs the answer is yes; in others, no (they are spacelike separated). The fraction of pairs that are causally related turns out to depend on the dimension of the spacetime.

Why? Think about light cones. In two dimensions (1 time + 1 space), the light cone is just two lines. Half of spacetime is causally related to you. In four dimensions (1 time + 3 space), the light cone is a narrow cone in a vast 4-dimensional space. A much smaller fraction of points are causally related.

Myrheim and Meyer worked out the exact relationship {cite}`Myrheim1978,Meyer1988`. For a causal set uniformly sprinkled in $D$-dimensional Minkowski space, the ordering fraction converges to a specific function of $D$. Invert this function, and you can read off the dimension from the ordering fraction.

For curvature, the approach is similar but more subtle. We build QSD-weighted **proper-time
neighborhoods** around a single event (not order intervals, which would be chains in a tree),
and we **localize** them using the algorithmic scales $\rho$ and $\varepsilon_c$ so the
neighborhoods are compact and first-principles. The primary estimator uses the geometric
light-cone order and $d_{\mathrm{geo}}$ from the reconstructed $g_R$. We also record a
trajectory-only diagnostic along each CST lineage based on realized path length; it is useful
for comparisons but is not claimed to converge to $R_g$ without additional geodesicity or
augmented-order assumptions. In the uniform Poisson limit the coefficients reduce to
Benincasa-Dowker.

This is the ultimate vindication of the causal set program: once the causal order of the
emergent manifold is fixed, geometry—dimension, curvature, volume—can be recovered from
counting causal relationships.
:::

:::{prf:definition} Myrheim-Meyer Dimension Estimator ({cite}`Myrheim1978,Meyer1988`)
:label: def-myrheim-meyer

The spacetime dimension is estimated from the ordering fraction, using the geometric
light-cone order $\prec_{\mathrm{LC}}$ defined above.

$$
r := \frac{C_2}{\binom{N}{2}} = \frac{|\{(e_i, e_j) : e_i \prec_{\mathrm{LC}} e_j\}|}{N(N-1)/2}
$$

For a causal set faithfully embedded in $D$-dimensional Minkowski space:

$$
r \xrightarrow{N \to \infty} \frac{\Gamma(D+1) \Gamma(D/2)}{2 \Gamma(3D/2)}
$$

The **Myrheim-Meyer estimator** inverts this relation to obtain $d_{\mathrm{MM}}$ from the
observed ordering fraction $r$. For the Fractal Set, $d_{\mathrm{MM}}$ estimates $D$, so the
spatial dimension is $d = D - 1$.

For adaptive density, an explicit correction is:

$$
w_{\mathrm{geo}}(e) := \frac{Z(t_e)}{r(t_e)} \exp\!\left(\frac{U_{\mathrm{eff}}(x_e,t_e)}{T}\right),
$$

$$
r_w := \frac{\sum_{i<j} w_{\mathrm{geo}}(e_i) w_{\mathrm{geo}}(e_j)\,
\mathbb{1}_{e_i\prec_{\mathrm{LC}} e_j \text{ or } e_j\prec_{\mathrm{LC}} e_i}}
{\sum_{i<j} w_{\mathrm{geo}}(e_i) w_{\mathrm{geo}}(e_j)}.
$$
In the uniform limit, $w_{\mathrm{geo}}\equiv 1$ and $r_w=r$.

An equivalent local-window estimator avoids explicit weights: for each interior event
$e\in E_{\mathrm{int}}$,
let $W(e):=\{e'\in E:\,|t_e-t_{e'}|\le T_{\mathrm{loc}},\; d_{\mathrm{geo}}(e',e)\le R_{\mathrm{loc}}\}$
using the reconstructed $d_{\mathrm{geo}}$ (via spinor-stored trajectories and $g_R$, or the
IG-graph shortest-path approximation noted above), and set

$$
r_e:=\frac{|\{(e_i,e_j)\in W(e)^2:\,e_i\prec_{\mathrm{LC}} e_j\}|}{\binom{|W(e)|}{2}}.
$$
With $R_{\mathrm{loc}},T_{\mathrm{loc}}\to 0$ and $|W(e)|\to\infty$ under the scaling in A6,
$r_e\to r$ and the same inversion yields $d_{\mathrm{MM}}$; this uses the local flatness of
$g$ on windows whose diameter shrinks with $R_{\mathrm{loc}}$.
:::

:::{prf:definition} Proper-Time Neighborhoods (Geometric and Trajectory)
:label: def-fractal-gas-proper-time-neighborhoods

Let $g = -c^2 dt^2 + g_R$ with $c=V_{\mathrm{alg}}$. For $e\in E_{\mathrm{int}}$ and two events
$e'=(x_{e'},t_{e'})$ and $e=(x_e,t_e)$, write
$t_-=\min(t_e,t_{e'})$, $t_+=\max(t_e,t_{e'})$. Define the **geometric path length**

$$
d_{\mathrm{geo}}(e',e) := \inf_{\gamma} \int_{t_-}^{t_+} \|\dot{x}(t)\|_{g_t}\,dt,
$$
where the infimum is over $C^1$ curves $\gamma:t\mapsto x(t)$ with $\gamma(t_{e'})=x_{e'}$,
$\gamma(t_e)=x_e$. Define the **geometric proper-time proxy**

$$
\tau_g(e',e) := \sqrt{c^2(t_e-t_{e'})^2 - d_{\mathrm{geo}}(e',e)^2}.
$$
When $g_R$ is time-independent this equals the Lorentzian proper time; for time-dependent
$g_R$ we use it as a local proxy in the small-$\varepsilon$ regime.

Let $R_{\mathrm{loc}}$ and $T_{\mathrm{loc}}$ be the algorithmic cutoffs from A5. The
**geometric proper-time neighborhood** is

$$
J_{g,\mathrm{loc}}^\pm(e;\varepsilon):=\{y\in M:\,0<\tau_g(y,e)\le \varepsilon,\;
|t(y)-t_e|\le T_{\mathrm{loc}},\; d_{\mathrm{geo}}(y,e)\le R_{\mathrm{loc}}\}.
$$

For a trajectory-only diagnostic, use the graph path length $d_g$ from
{prf:ref}`def-fractal-causal-order` and define the symmetric directed distance
$d_g^\pm(e',e) := \min(d_g(e',e), d_g(e,e'))$ (finite only for CST-comparable pairs). Then
define the **trajectory proper-time proxy**

$$
\tau_{\mathrm{traj}}(e',e) := \sqrt{c^2(t_e-t_{e'})^2 - d_g^\pm(e',e)^2},
$$

with $\tau_{\mathrm{traj}}(e',e)$ real iff $d_g^\pm(e',e)\le c|t_e-t_{e'}|$. The
**trajectory proper-time neighborhood** is

$$
J_{\mathrm{traj,loc}}^\pm(e;\varepsilon) :=
\{e' \in E:\, e'\prec_{\mathrm{CST}} e \text{ or } e\prec_{\mathrm{CST}} e',\;
0<\tau_{\mathrm{traj}}(e',e)\le \varepsilon,\;
|t_e-t_{e'}|\le T_{\mathrm{loc}},\; d_g^\pm(e',e)\le R_{\mathrm{loc}}\}.
$$

The geometric neighborhood is used for curvature estimation; the trajectory neighborhood is a
lineage-only diagnostic and does not approximate light-cone neighborhoods across lineages.
Both neighborhoods use only existing episodes; no new points are sampled.
:::

:::{prf:definition} Fractal Gas Curvature Estimators
:label: def-fractal-gas-curvature

Let $K_R \in C^2_c([0,1])$ be a curvature kernel. In normal coordinates at $e=(x,t)\in E_{\mathrm{int}}$,
assume the expansion

$$
\frac{1}{\varepsilon^D}\int_{J_{g,\mathrm{loc}}^\pm(e;\varepsilon)} K_R\!\left(\frac{\tau_g(y,e)}{\varepsilon}\right)
\, d\mathrm{vol}_g(y) = M_0^{(R)} + M_R\,R_g(x,t)\,\varepsilon^2 + O(\varepsilon^3),
$$
with $M_R \ne 0$. Here $M_0^{(R)}$ and $M_R$ are dimension-dependent constants determined by
 $K_R$ (computed in flat space). The algorithmic cutoffs make these moments finite; their
values depend on $(R_{\mathrm{loc}}, T_{\mathrm{loc}})$ (hence on $\varepsilon$ through the
identification in A6) but not on $N$.

Define the **geometric Fractal Gas curvature estimator**:

$$
\widehat{R}_{\mathrm{FG}}^{(g)}(e) :=
\frac{1}{M_R\,\varepsilon^2}
\left[
\frac{R}{N\,\varepsilon^D}\sum_{e' \in E \cap J_{g,\mathrm{loc}}^\pm(e;\varepsilon)} w_{\mathrm{geo}}(e')\,
K_R\!\left(\frac{\tau_g(e',e)}{\varepsilon}\right)
- M_0^{(R)}
\right].
$$

Define the **trajectory curvature diagnostic**:

$$
\widehat{R}_{\mathrm{FG}}^{(\mathrm{traj})}(e) :=
\frac{1}{M_R\,\varepsilon^2}
\left[
\frac{R}{N\,\varepsilon^D}\sum_{e' \in J_{\mathrm{traj,loc}}^\pm(e;\varepsilon)} w_{\mathrm{geo}}(e')\,
K_R\!\left(\frac{\tau_{\mathrm{traj}}(e',e)}{\varepsilon}\right)
- M_0^{(R)}
\right].
$$

This diagnostic uses only realized CST worldlines and is intended for comparison against the
geometric estimator; it is not claimed to converge to $R_g$ without additional geodesicity
assumptions. On the half-open window, $R = \int r(t)\,dt = N$ so the prefactor simplifies to
$\varepsilon^{-D}$.
:::

:::{prf:theorem} Ricci Scalar from Fractal Gas (Geometric Estimator)
:label: thm-fractal-gas-ricci

Assume {prf:ref}`assm-fractal-gas-nonlocal` and the expansion in
{prf:ref}`def-fractal-gas-curvature`. Then for $e\in E_{\mathrm{int}}$,
$\widehat{R}_{\mathrm{FG}}^{(g)}(e) \to R_g(x,t)$ in probability as $\varepsilon\to 0$,
$N\to\infty$.

Define the Fractal Gas Einstein-Hilbert action

$$
S_{\mathrm{FG}}^{(g)} := \frac{1}{2\kappa_D}\sum_{e\in E_{\mathrm{int}}}
\mu_{\mathrm{geo}}(e)\,\widehat{R}_{\mathrm{FG}}^{(g)}(e).
$$

Then $S_{\mathrm{FG}}^{(g)}$ converges to
$(2\kappa_D)^{-1}\int_M R_g\, d\mathrm{vol}_g$. In the uniform Poisson limit with a kernel
matching the Benincasa-Dowker coefficients, this action reduces to $S_{\mathrm{BD}}$
({cite}`BenincasaDowker2010`).
:::

:::{admonition} Comparison with Classical CST
:class: note

- **Benincasa–Dowker (BD) limit**: Under uniform Poisson sprinkling and a kernel that matches
  BD coefficients, the Fractal Gas estimators reduce to the classical BD curvature estimator
  and action.
- **Interval counts vs neighborhoods**: Classical BD curvature uses interval counts $N_k$ on
  order intervals $I(p,q)$. In a tree, $I(p,q)$ collapses to a chain, so we use proper-time
  neighborhoods around a single event instead; the Poisson limit reproduces BD weights for the
  geometric estimator.
- **Dimension checks**: The Myrheim–Meyer estimator is unchanged; adaptive density requires
  local windows or reweighting, and in the uniform limit it matches standard CST results.
- **Sanity tests**: On Minkowski sprinklings, the geometric estimator should recover $R=0$ and
  the ordering fraction; deviations quantify kernel choice and sampling bias.
- **Cross-method comparison**: Later in `3_fitness_manifold`, compare these estimates to
  holonomy/plaquette curvature to validate the emergent geometry.
:::

---

(sec-physical-consequences)=
## Physical Consequences

:::{div} feynman-prose
Now we get to the really exciting part—the physical implications. Having established that the Fractal Set is a valid causal set with adaptive sprinkling, what can we actually *do* with it?

In quantum mechanics, you compute probabilities by summing over all possible paths a particle could take, weighted by a phase factor $e^{iS}$ where $S$ is the action. Feynman taught us this in the 1940s. The same idea extends to quantum gravity: instead of summing over paths of a particle, you sum over *geometries*—all possible shapes spacetime could have.

But here is the problem. What does it mean to "sum over all geometries"? In the continuum, this is almost impossibly hard. There are infinitely many ways to warp and curve a manifold. How do you put a measure on this space? What does the integral even mean?

Causal set theory provides an answer: sum over all causal sets {cite}`Sorkin05`. The space of finite causal sets is discrete. Counting is well-defined. There is no ultraviolet divergence because the number of points is finite. The sum is, in principle, tractable.

The Fractal Set does something extra. In ordinary causal set quantum gravity, you sum over all causal sets with equal weight—every causal set counts the same. But this is arbitrary. Why should a causal set representing flat empty space have the same weight as one representing a black hole? The QSD provides a *physical* measure. Causal sets that arise from the Adaptive Gas dynamics—sets that represent physically sensible configurations—get weighted by $\mathcal{P}_{\mathrm{QSD}}$. Unphysical configurations get suppressed.

This is like the difference between uniform and Boltzmann sampling. In statistical mechanics, you do not count all configurations equally—you weight them by $e^{-E/kT}$. The QSD provides the analogous weighting for quantum gravity.
:::

### Quantum Gravity Path Integral

:::{admonition} Path Integral Formulation
:class: important

The causal set path integral for quantum gravity:

$$
Z_{\mathrm{Fractal}} = \sum_{\mathcal{F} \in \mathcal{F}_N} e^{iS_{\mathrm{CST}}(\mathcal{F})} \cdot \mathcal{P}_{\mathrm{QSD}}(\mathcal{F})
$$

**Key advantage**: The QSD provides a **physically motivated measure** on the space of causal sets, replacing ad-hoc uniform measures.
:::

### Observable Predictions

:::{div} feynman-prose
Can we actually test any of this? The scale of quantum gravity—the Planck length, $10^{-35}$ meters—is so far below anything we can probe directly that you might think the whole enterprise is purely theoretical.

But there are loopholes. The key observation is that quantum gravity effects, even if tiny, can *accumulate* over cosmological distances. A photon traveling billions of light-years from a distant gamma-ray burst might experience tiny Planck-scale fluctuations at every step. Over billions of steps, these add up to something potentially measurable.

The predictions fall into two categories. First, there is a discreteness scale—the average spacing between episodes. This is like the lattice spacing in a crystal, but for spacetime itself. Second, there are modified dispersion relations. In ordinary physics, the energy and momentum of a particle are related by $E^2 = p^2 c^2 + m^2 c^4$. Discreteness adds correction terms proportional to $E/E_{\mathrm{Planck}}$. At low energies, these are negligible. At ultra-high energies—cosmic rays, gamma-ray bursts—they might be detectable.

The current observational bounds are beginning to nibble at the interesting parameter space. We have not seen Lorentz violation yet, which rules out some models. But the Fractal Set makes specific predictions about the form of these corrections. Future observations could either confirm them or rule them out. This is real physics, not just mathematics.
:::

:::{prf:proposition} Testable Predictions
:label: prop-cst-predictions

The Fractal Set causal structure leads to observable consequences:

1. **Discreteness scale**: Average proper distance between episodes (with $V_{\mathrm{total}}$ the spacetime volume of the window):

$$
\ell_{\mathrm{eff}} \approx \left(\frac{V_{\mathrm{total}}}{N}\right)^{1/D}, \quad
\ell(t, x) \sim \lambda(t, x)^{-1/D}, \;\; \lambda(t, x) = r(t)\, \rho_{\mathrm{adaptive}}(x, t)
$$

2. **Modified dispersion relations**: High-energy particles experience corrections:

$$
E^2 = p^2 c^2 + m^2 c^4 + \eta_1 \frac{E^3}{E_{\mathrm{Planck}}} + \eta_2 \frac{E^4}{E_{\mathrm{Planck}}^2} + \ldots
$$
where $E_{\mathrm{Planck}} = \sqrt{\hbar c^5 / G}$ and $\eta_i$ are $O(1)$ coefficients.

3. **Lorentz violation bounds**: Observable in cosmic rays, gamma-ray bursts, ultra-high-energy neutrinos
:::

### Connection to Loop Quantum Gravity

:::{div} feynman-prose
One question that always comes up: how does this relate to Loop Quantum Gravity, the other major approach to quantum gravity?

LQG and causal set theory are different programs with different starting points. LQG starts from general relativity and quantizes it directly; the result is a spin network—a graph where nodes carry quantum numbers (spins) and edges carry group-theoretic labels. Causal set theory starts from discreteness and causality; the manifold is supposed to emerge in the large-scale limit.

The Fractal Set sits somewhere in between. It has nodes (like spin networks) and directed edges (like causal sets). The CST edges are timelike—they encode evolution. The IG edges are spacelike—they encode spatial coupling. There is an intriguing structural similarity.

The key difference is that the Fractal Set is classical and stochastic, not quantum. It arises from a Monte Carlo algorithm, not from a path integral over quantum states. But the structural parallels suggest that the same mathematical objects keep appearing, regardless of which door you enter the building through. Perhaps this is telling us something deep about the nature of quantum spacetime.
:::

:::{admonition} Relation to LQG
:class: note

| Fractal Set | Loop Quantum Gravity |
|:------------|:---------------------|
| Episodes $e \in E$ | Nodes of spin network |
| CST edges $e_1 \prec_{\mathrm{CST}} e_2$ | Links with SU(2) labels |
| IG edges $e_i \sim e_j$ | Gauge connections |
| Adaptive density $\rho \propto \sqrt{\det g}$ | Quantum geometry operators |

**Key difference**: Fractal Set is classical + stochastic; LQG is quantum from the start.
:::

---

## Summary

**Main Results**:

1. ✅ **Fractal Set is a causal set**: Satisfies all BLMS axioms (Theorem {prf:ref}`thm-fractal-is-causal-set`)

2. ✅ **Adaptive sprinkling**: QSD sampling with $\rho(x, t) \propto \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T}$ provides geometry-aware resolution

3. ✅ **CST machinery applies**: d'Alembertian, volume elements, and dimension/curvature estimators apply in their adaptive-density form

4. ✅ **Physical implications**: Foundation for quantum gravity calculations on emergent spacetime

---

## References

### Causal Set Theory
1. Bombelli, L., Lee, J., Meyer, D., & Sorkin, R.D. (1987) "Space-Time as a Causal Set", *Phys. Rev. Lett.* **59**, 521 {cite}`BombelliLeeEtAl87`
2. Sorkin, R.D. (2005) "Causal Sets: Discrete Gravity", in *Lectures on Quantum Gravity*, Springer {cite}`Sorkin05`
3. Benincasa, D.M.T. & Dowker, F. (2010) "The Scalar Curvature of a Causal Set", *Phys. Rev. Lett.* **104**, 181301 {cite}`BenincasaDowker2010`
4. Myrheim, J. (1978) "Statistical Geometry" {cite}`Myrheim1978`
5. Meyer, D.A. (1988) *The Dimension of Causal Sets* (PhD thesis) {cite}`Meyer1988`

### Framework Documents
6. {doc}`/source/3_fractal_gas/2_fractal_set/01_fractal_set` — Fractal Set definition and structure
7. {prf:ref}`def-fractal-set-cst-edges` — CST edge definition
8. {prf:ref}`def-fractal-set-cst-axioms` — CST axioms
9. {prf:ref}`mt:fractal-gas-lock-closure` — Lock Closure for Fractal Gas (Hypostructure)
10. {prf:ref}`thm-fractal-set-lossless` — Lossless reconstruction theorem
11. {prf:ref}`def-fractal-set-reconstruction-targets` — Reconstruction target set
12. {prf:ref}`def-adaptive-diffusion-tensor-latent` — Adaptive diffusion tensor and emergent metric
13. {prf:ref}`thm-expansion-adjunction` — Expansion Adjunction (framework lift)
