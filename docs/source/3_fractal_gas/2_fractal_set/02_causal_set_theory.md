# Fractal Set as Causal Set

**Prerequisites**: {doc}`01_fractal_set`, Causal Set Theory ({cite}`BombelliLeeEtAl87,Sorkin05`)

---

## TLDR

*Notation: $(E, \prec)$ = Fractal Set with causal order; BLMS = Bombelli-Lee-Meyer-Sorkin axioms; $d = \dim \mathcal{X}$, $D = d + 1$ (spacetime dimension for CST formulas); $\rho_{\mathrm{adaptive}}$ = QSD sampling density; $\sqrt{\det g}$ = Riemannian volume element.*

**The Fractal Set is a Valid Causal Set**: The episode set $E$ with CST ordering $\prec$ satisfies all three BLMS axioms (irreflexivity, transitivity, local finiteness), making the Fractal Set a rigorous causal set in the sense of quantum gravity research.

**Adaptive Sprinkling Innovation**: Unlike standard Poisson sprinkling with constant spacetime density, QSD sampling yields an inhomogeneous density $\rho_{\mathrm{adaptive}}(x, t) \propto \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T}$, adapting resolution to the learned geometry (higher-weight regions get more episodes).

**Causal Set Machinery Applies**: With causal set status established, CST tools (d'Alembertian, dimension estimators, curvature measures) apply in full, using the adaptive-density formulas given here.

---

(sec-cst-intro)=
## Introduction

:::{div} feynman-prose
Here is a beautiful connection that was waiting to be discovered. Causal set theory is one of the leading approaches to quantum gravity—the idea that spacetime is fundamentally discrete, made up of a locally finite (typically countable) set of events with a partial ordering that encodes causal structure. The program was launched by Bombelli, Lee, Meyer, and Sorkin in 1987 {cite}`BombelliLeeEtAl87`, and it has developed into a sophisticated mathematical framework (see {cite}`Sorkin05`).

Now, the Fractal Set is also a discrete structure with a causal ordering. Episodes are events; CST edges encode causal precedence. The question is: does the Fractal Set satisfy the axioms of a causal set? If it does, then the mathematical machinery of causal set theory—developed over decades by quantum gravity researchers—becomes available to us, and the adaptive-density versions are spelled out explicitly here.

The answer is yes. The Fractal Set is a valid causal set. But it is more than that: it is an *adaptive* causal set, where the sampling density automatically adjusts to local geometry. This goes beyond the standard Lorentz-invariant Poisson construction, which fixes a constant density.
:::

Causal set theory (CST) posits that spacetime is fundamentally discrete: a locally finite (typically countable) collection of events with a partial ordering encoding causal relationships. The Fractal Set, defined in {doc}`01_fractal_set`, provides exactly such a structure via its CST edges ({prf:ref}`def-fractal-set-cst-edges`) and CST axioms ({prf:ref}`def-fractal-set-cst-axioms`). This chapter establishes that:

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
\frac{\|\Delta x\|_{g_{t_k}}}{\Delta t} ,
$$

The maximum exists because each timestep has finitely many CST edges; if a timestep is empty,
set $c_{\mathrm{eff}}(t)=0$.

Define the geometric (light-cone) order

$$
e_i \prec_{\mathrm{LC}} e_j \quad \iff \quad t_i < t_j
\;\wedge\; d_g(e_i, e_j) \leq \int_{t_i}^{t_j} c_{\mathrm{eff}}(t)\, dt .
$$

For discrete timesteps, interpret the integral as $\sum_k c_{\mathrm{eff}}(t_k)\,\Delta t_k$.

**Physical meaning**: $e_i \prec e_j$ iff information from $e_i$ can causally influence $e_j$.

**Equivalence**: Proposition {prf:ref}`prop-fractal-causal-order-equivalence` shows
$\prec_{\mathrm{CST}} = \prec_{\mathrm{LC}}$, so the geometric form can be used throughout.
:::

:::{div} feynman-prose
Look at this definition carefully. It says that one episode "precedes" another if (1) it happens earlier in time, AND (2) it is close enough in space that a signal could travel between them.

The second condition is the light cone constraint. If two events are too far apart in space, no signal—not even light—can travel between them in the available time. They are "spacelike separated," to use the relativity jargon. They cannot influence each other.

The effective speed $c_{\mathrm{eff}}$ is like a "speed limit" for information propagation in this system. It plays the role that the speed of light plays in relativity. Nothing can travel faster, so nothing can causally connect events that are outside each other's "cones."

Here is the physical picture. Draw a diagram with time going up and space going sideways. At each event, draw a cone opening upward—this is the "future light cone." Only events inside (or on) this cone can be influenced by the original event. The causal order $e_i \prec e_j$ means exactly that $e_j$ is inside the future cone of $e_i$.
:::

:::{figure} figures/cst-light-cone.svg
:alt: Light cone diagram showing causal and spacelike events.
:width: 90%
:align: center

Light-cone causal order. Events inside the cone are causally related; events outside are spacelike.
:::

:::{prf:proposition} Graph-Light-Cone Equivalence
:label: prop-fractal-causal-order-equivalence

The graph order $\prec_{\mathrm{CST}}$ equals the geometric order $\prec_{\mathrm{LC}}$.

*Proof.* For any CST edge, the definition of $c_{\mathrm{eff}}(t)$ gives
$\|\Delta x\|_{g_t} \leq c_{\mathrm{eff}}(t)\,\Delta t$. Hence any CST path $\gamma$ satisfies
$L_g(\gamma) \leq \int_{t_i}^{t_j} c_{\mathrm{eff}}(t)\, dt$, so
$e_i \prec_{\mathrm{CST}} e_j \Rightarrow e_i \prec_{\mathrm{LC}} e_j$. Conversely,
if $e_i \prec_{\mathrm{LC}} e_j$, then $d_g(e_i, e_j) \leq \int_{t_i}^{t_j} c_{\mathrm{eff}}(t)\, dt$.
Since $d_g = \infty$ when no CST path exists, the inequality implies a CST path exists and
$e_i \prec_{\mathrm{LC}} e_j \Rightarrow e_i \prec_{\mathrm{CST}} e_j$. Thus the orders coincide.
By Expansion Adjunction and Lock Closure, $d_g$ converges to the geodesic distance of the
emergent metric in the continuum lift, so the light-cone criterion matches the manifold
causal order without assuming a stationary metric or uniform episode rate. $\square$
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

**Comparison with Poisson sprinkling**:

| Standard CST | Fractal Set |
|:------------|:------------|
| Density $\rho = \mathrm{const}$ | Density $\rho(x, t) \propto \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T}$ |
| Uniform sampling | Adaptive sampling |
| Ad-hoc choice of $\rho$ | Automatic from QSD |
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
- Algorithmic space: {prf:ref}`def:algorithmic-space-fg`
:::

### Framework Lift: CST Operators Are Computable

:::{prf:proposition} Framework Lift for CST Operators
:label: prop-fractal-cst-framework-lift

By lossless reconstruction ({prf:ref}`thm-fractal-set-lossless`), all quantities in the
reconstruction target set ({prf:ref}`def-fractal-set-reconstruction-targets`) are recoverable
at episode locations, including trajectories and the adaptive diffusion tensor
$\Sigma_{\mathrm{reg}}$ from the SDE ({prf:ref}`def-fractal-set-sde`). The emergent metric is
defined by $\Sigma_{\mathrm{reg}}$ via {prf:ref}`def-adaptive-diffusion-tensor-latent`, so $g$
and $d_g$ are available on the same support. Expansion Adjunction
({prf:ref}`thm-expansion-adjunction`) and Lock Closure
({prf:ref}`mt:fractal-gas-lock-closure`) lift the discrete causal order to the continuum
limit. Therefore the adaptive-density CST operators in this chapter are well-defined and
directly computable on the Fractal Set.
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

**Axiom CS1 (Irreflexivity):** If $e_i \prec e_i$, there is a directed CST path from $e_i$ to
itself. But each CST edge strictly increases time, so no directed path can return to its
starting node. Hence $e_i \not\prec e_i$. ✓

**Axiom CS2 (Transitivity):** If $e_1 \prec e_2$ and $e_2 \prec e_3$, then there are CST paths
from $e_1$ to $e_2$ and from $e_2$ to $e_3$. Concatenating them gives a CST path from $e_1$ to
$e_3$, so $e_1 \prec e_3$. ✓

**Axiom CS3 (Local Finiteness):** If $e_1 \prec e_2$, then $t_1 < t_2$. Any episode with
$e_1 \prec e \prec e_2$ must lie at a timestep in $\{t_1+1, \ldots, t_2-1\}$, and each timestep
contains finitely many episodes (bounded by the walker count). Hence $|I(e_1, e_2)| < \infty$. ✓

$\square$
:::

:::{dropdown} 📖 ZFC Proof (Classical Verification)
:icon: book

**Classical Verification (ZFC):**

Working in Grothendieck universe $\mathcal{U}$, the Fractal Set $(E, \prec)$ is a finite partially ordered set (poset) where:

1. $E \in V_\mathcal{U}$ is a finite set of episodes
2. $\prec \subseteq E \times E$ is a binary relation

**CS1 (Irreflexivity):** $\prec_{\mathrm{CST}}$ is the transitive closure of CST edges, each of
which increases time. Thus no directed path can return to its start, so $e_i \not\prec e_i$.

**CS2 (Transitivity):** By construction, the transitive closure of any relation is transitive.

**CS3 (Local Finiteness):** If $e_1 \prec e_2$, then $t_1 < t_2$, and any $e$ with
$e_1 \prec e \prec e_2$ lies on a timestep between $t_1$ and $t_2$. Each timestep has finitely
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

**Volume Matching**: For a time window $[t_0, t_1]$, condition on $N$ episodes and rate $r(t)$,
the episode count in a spatial region $\Omega$ satisfies:

$$
\mathbb{E}\left[\frac{|E \cap ([t_0,t_1]\times \Omega)|}{N}\right]
= \frac{1}{R} \int_{t_0}^{t_1} \frac{r(t)}{Z(t)}
\int_{\Omega} \sqrt{\det g(x, t)} \, e^{-U_{\mathrm{eff}}(x, t)/T} \, dx \, dt
$$

with $R = \int_{t_0}^{t_1} r(t)\, dt$ (for discrete steps, $R = \sum_k |E_{\mathrm{CST}}(t_k)|$).
The expectation is conditional on $N$ (or asymptotic as $N \to \infty$). Variance scales as
$O(1/N)$ under standard mixing/ergodicity. (To recover
geometric spacetime volume
$\int_{t_0}^{t_1}\!\int_\Omega \sqrt{\det g(x,t)} \, dx \, dt$, reweight by
$e^{U_{\mathrm{eff}}(x,t)/T} \, Z(t) / r(t)$ and multiply by $R$.)

**Distance Estimation**: Timelike distances are estimated by longest-chain length; spatial
distances follow from reconstructed trajectories and IG-edge geometry
({prf:ref}`thm-fractal-set-trajectory`, {prf:ref}`def-fractal-set-ig-edges`).

**Dimension Estimation**: The Myrheim-Meyer estimator converges after applying the
adaptive-density correction described below; an equivalent implementation is to compute it
on local windows of approximately constant density:

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

The answer is: you use *nonlocal* operators that average over many nearby points. The brilliant insight of Benincasa and Dowker is that you can construct a discrete operator that looks wildly nonlocal—it sums over all points in the causal past with complicated dimension-dependent coefficients—but in the continuum limit, this operator converges to the ordinary d'Alembertian. The graininess washes out, and smooth physics emerges.

This is genuinely remarkable. You start with a completely discrete structure—just a locally finite set of points with an ordering—and you can compute the wave equation, the curvature, the dimension. All of classical and quantum field theory on curved spacetime becomes accessible, not despite the discreteness but *through* it.

The dimension estimator is my favorite example. It counts nothing but the fraction of pairs that are causally related. That is all. No coordinates, no metric, no tangent spaces. Just counting. And yet from this single number, you can read off the dimension of spacetime. This is causal set theory at its purest: geometry from order alone.
:::

With the Fractal Set established as a valid causal set, standard CST tools apply in their
adaptive-density form, and all required inputs are reconstructible from the framework. We use
$d = \dim \mathcal{X}$ for spatial slices and $D = d + 1$ for the spacetime dimension in CST
formulas.

### Causal Set Volume Element

:::{prf:definition} Causal Set Volume (Adaptive Measure)
:label: def-cst-volume

Let $r(t)$ be the episode rate and $\rho_{\mathrm{adaptive}}(x, t)$ the instantaneous spatial
QSD marginal. Define the spacetime measure
$d\mu_{\mathrm{adaptive}}(t, x) := r(t)\,\rho_{\mathrm{adaptive}}(x, t)\, dt\, dx$ and let
$E$ denote the episodes in the chosen time window with $N = |E|$. The **adaptive causal set volume** of
$e \in E$ is:

$$
V_{\mathrm{adaptive}}(e) := \frac{1}{N} \sum_{e' \in E} \mathbb{1}_{e' \prec e}
$$

**Continuum limit**: (conditional on $N$, or asymptotically) $V_{\mathrm{adaptive}}(e) \to \mu_{\mathrm{adaptive}}(J^-(e)) / R$ with

$$
\mu_{\mathrm{adaptive}}(J^-(e)) = \int_{J^-(e)} \frac{r(t)}{Z(t)} \sqrt{\det g(x, t)}
\, e^{-U_{\mathrm{eff}}(x, t)/T} \, dt \, dx, \quad R = \int r(t)\, dt
\;(\text{discrete } R = \sum_k |E_{\mathrm{CST}}(t_k)|).
$$

**Geometric volume recovery**: Reweight by the Boltzmann factor to undo the QSD bias:

$$
V_g(e) := \frac{1}{N} \sum_{e' \in E,\, e' \prec e}
\frac{Z(t_{e'})}{r(t_{e'})} \exp\!\left(\frac{U_{\mathrm{eff}}(x_{e'}, t_{e'})}{T}\right),
\quad \mathbb{E}[V_g(e)] = \frac{1}{R} \int_{J^-(e)} \sqrt{\det g(x, t)} \, dt \, dx .
$$
:::

### Discrete d'Alembertian (Benincasa-Dowker Operator)

:::{div} feynman-prose
The d'Alembertian $\Box = \partial_t^2 - \nabla^2$ is the wave operator—it tells you how waves propagate through spacetime. In the continuum, it is defined using second derivatives. But on a discrete set, there are no derivatives. How do you take a derivative when there is no notion of "infinitesimally close"?

Benincasa and Dowker found the answer. The key insight is that derivatives are really about *averaging*. The second derivative of a function at a point tells you how the function compares to its average in a small neighborhood. If the function is higher than the average of its neighbors, the second derivative is negative (the function is curved downward); if lower, the second derivative is positive.

On a causal set, you can define "neighborhood" using the causal structure. The points in your causal past—the ones that can influence you—are your neighbors. But here is the subtle part: you need to weight these points carefully. Points that are one step away in the causal past should contribute differently than points that are two steps away.

The Benincasa-Dowker operator does exactly this. It sums the function values at all points in the causal past, weighted by dimension-dependent coefficients that depend on how many intermediate points exist. The formula looks complicated, but it has a deep structural reason: these are precisely the weights that make the discrete operator converge to the continuum d'Alembertian as you take more and more points {cite}`BenincasaDowker2010`.

The beautiful part is that this works in any dimension. The coefficients $C_k^{(D)}$ change with dimension, but the structure is the same. Count the intervals, weight them properly, and you get waves.
:::

:::{prf:definition} Discrete d'Alembertian on Fractal Set (Benincasa-Dowker)
:label: def-cst-dalembertian

The **Benincasa-Dowker d'Alembertian** acting on $f: E \to \mathbb{R}$ in $D$ dimensions is:

$$
(\Box_{\mathrm{BD}} f)(e) := \frac{1}{\ell_D^2} \left( -\alpha_D f(e) + \sum_{k=0}^{n_D} C_k^{(D)} \sum_{\substack{e' \prec e \\ |I(e', e)| = k}} f(e') \right)
$$

where:
- $\ell_D = \rho^{-1/D}$ is the discreteness scale for sprinkling intensity $\rho$ (use local spacetime intensity $\lambda(t, x)$ for inhomogeneous sprinkling)
- $\alpha_D$, $C_k^{(D)}$ are dimension-dependent coefficients (including the standard normalization; see {cite}`BenincasaDowker2010` for explicit values)
- $|I(e', e)|$ is the number of elements in the causal interval between $e'$ and $e$

**Convergence** ({cite}`BenincasaDowker2010`): For uniform Poisson sprinkling and smooth functions:

$$
\lim_{N \to \infty} \mathbb{E}[(\Box_{\mathrm{BD}} f)(e_i)] = (\Box_g f)(x_i) + O(\ell_D^2)
$$
where $\Box_g = g^{\mu\nu}\nabla_\mu\nabla_\nu$ is the continuum d'Alembertian for the
induced Lorentzian metric (with spatial part $g$ and time coordinate $t$).

For adaptive density, use the local-density corrected operator (the required $\lambda(t, x)$ is
reconstructible from the episode data).
:::

### Dimension and Curvature Estimation

:::{div} feynman-prose
Here is something that should make you sit up. We are going to measure the dimension of spacetime—an integer!—by counting pairs and computing a fraction. No coordinates. No metric. No calculus. Just counting.

The idea is simple but profound. Take any two points in a causal set and ask: are they causally related? That is, is one in the past of the other? In some pairs the answer is yes; in others, no (they are spacelike separated). The fraction of pairs that are causally related turns out to depend on the dimension of the spacetime.

Why? Think about light cones. In two dimensions (1 time + 1 space), the light cone is just two lines. Half of spacetime is causally related to you. In four dimensions (1 time + 3 space), the light cone is a narrow cone in a vast 4-dimensional space. A much smaller fraction of points are causally related.

Myrheim and Meyer worked out the exact relationship {cite}`Myrheim1978,Meyer1988`. For a causal set uniformly sprinkled in $D$-dimensional Minkowski space, the ordering fraction converges to a specific function of $D$. Invert this function, and you can read off the dimension from the ordering fraction.

For curvature, the approach is similar but more subtle. The Benincasa-Dowker action counts intervals with specific weights. In flat space, these counts have certain expected values. Curvature perturbs these expectations in a predictable way. By measuring the deviation from flatness, you can extract the Ricci scalar—the simplest measure of how curved spacetime is.

This is the ultimate vindication of the causal set program: geometry is not primary. Order is primary. Geometry—dimension, curvature, volume—emerges from counting causal relationships.
:::

:::{prf:definition} Myrheim-Meyer Dimension Estimator ({cite}`Myrheim1978,Meyer1988`)
:label: def-myrheim-meyer

The spacetime dimension is estimated from the ordering fraction:

$$
r := \frac{C_2}{\binom{N}{2}} = \frac{|\{(e_i, e_j) : e_i \prec e_j\}|}{N(N-1)/2}
$$

For a causal set faithfully embedded in $D$-dimensional Minkowski space:

$$
r \xrightarrow{N \to \infty} \frac{\Gamma(D+1) \Gamma(D/2)}{2 \Gamma(3D/2)}
$$

The **Myrheim-Meyer estimator** inverts this relation to obtain $d_{\mathrm{MM}}$ from the observed ordering fraction $r$. For the Fractal Set, $d_{\mathrm{MM}}$ estimates $D$, so the spatial dimension is $d = D - 1$. For adaptive density, compute $r$ in local windows or apply density reweighting using the reconstructed $\rho_{\mathrm{adaptive}}$.
:::

:::{prf:proposition} Ricci Scalar from Causal Set
:label: prop-ricci-cst

The Ricci scalar curvature is estimated via the **Benincasa-Dowker action** ({cite}`BenincasaDowker2010`):

For a small causal diamond $\mathcal{A}(p, q)$ with $N$ elements:

$$
S_{\mathrm{BD}}[\mathcal{A}] = \frac{\hbar}{\ell_D^{D-2}} \left( \alpha_D N - \sum_{k=0}^{n_D} \beta_k^{(D)} N_k \right)
$$
where $N_k$ counts $k$-element intervals and $\ell_D$ is the discreteness scale.

**Curvature extraction**: In the continuum limit:

$$
\lim_{\ell \to 0} \frac{S_{\mathrm{BD}}[\mathcal{A}]}{V(\mathcal{A})} = \kappa_D \, R + O(\ell^2)
$$
where $R$ is the Ricci scalar of the induced Lorentzian metric, $V(\mathcal{A})$ is the spacetime volume, and the prefactor
is a known dimension-dependent constant (see {cite}`BenincasaDowker2010`).
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
| CST edges $e_1 \prec e_2$ | Links with SU(2) labels |
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
6. {doc}`01_fractal_set` — Fractal Set definition and structure
7. {prf:ref}`def-fractal-set-cst-edges` — CST edge definition
8. {prf:ref}`def-fractal-set-cst-axioms` — CST axioms
9. {prf:ref}`mt:fractal-gas-lock-closure` — Lock Closure for Fractal Gas (Hypostructure)
10. {prf:ref}`thm-fractal-set-lossless` — Lossless reconstruction theorem
11. {prf:ref}`def-fractal-set-reconstruction-targets` — Reconstruction target set
12. {prf:ref}`def-adaptive-diffusion-tensor-latent` — Adaptive diffusion tensor and emergent metric
13. {prf:ref}`thm-expansion-adjunction` — Expansion Adjunction (framework lift)
