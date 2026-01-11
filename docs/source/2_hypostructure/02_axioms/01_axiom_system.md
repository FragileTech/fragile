# Part III: The Axiom System

:::{div} feynman-prose
Now we come to the heart of the matter. Everything we have built so far---the categorical foundations, the hypostructure object, the fixed-point principle---has been preparation. Here is where we actually say what it *means* for a dynamical system to be well-behaved.

You might think: "Well, I know what regular means---no singularities, solutions exist for all time, the usual PDE stuff." And you would be partly right. But here is the thing that makes the Sieve approach different: we are not going to just *assume* regularity and see what follows. We are going to identify *exactly* what conditions must hold for regularity to be possible, and then *check* those conditions systematically.

Think of it this way. A pilot does not just assume the plane will fly. Before takeoff, there is a checklist: fuel levels, control surfaces, instruments, communications. Each item on the checklist corresponds to a specific failure mode that has, at some point in aviation history, caused a crash. The axioms you are about to see are the theoretical physicist's pre-flight checklist. Each one corresponds to a specific way that dynamical systems can go wrong.

The beautiful thing is that these axioms are not arbitrary. They emerge naturally from asking: "What could possibly prevent a system from evolving smoothly forever?" The answers turn out to fall into five categories---Conservation, Duality, Symmetry, Topology, and Boundary---and each category tells a different story about how systems can fail.

Let me walk you through each one, and I promise you that by the end, you will see that these constraints are not just technically necessary---they are *obvious* once you understand what they are protecting against.
:::

The Hypostructure $\mathbb{H}$ is valid if it satisfies the following structural constraints (Axioms). In the operational Sieve, these are verified as **Interface Permits** at the corresponding nodes.

(sec-conservation-constraints)=
## Conservation Constraints

:::{div} feynman-prose
The first family of axioms deals with conservation---the idea that some quantities cannot grow without bound or accumulate infinitely fast. This is physics at its most fundamental: energy does not come from nowhere, and infinitely many things cannot happen in finite time.

Why should we care? Because the two most common ways for PDEs to blow up are: (1) some quantity like energy goes to infinity, and (2) infinitely many discrete events pile up at a single moment (Zeno's paradox, but for differential equations). If we can rule out both of these, we have already eliminated the most obvious failure modes.
:::

:::{prf:axiom} Axiom D (Dissipation)
:label: ax-dissipation

The energy-dissipation inequality holds:

$$
\Phi(S_t x) + \int_0^t \mathfrak{D}(S_s x) \, ds \leq \Phi(x)
$$

**Enforced by:** {prf:ref}`def-node-energy` --- Certificate $K_{D_E}^+$
:::

:::{div} feynman-prose
Here is the picture you should have in your mind for Axiom D. Imagine energy as water in a bathtub. The dissipation $\mathfrak{D}$ is the drain---it is how fast water flows out. The inequality says: the water level now, plus everything that went down the drain, cannot exceed what you started with. No water appears from nowhere.

This is the Second Law of Thermodynamics in disguise. In a closed system, useful energy always decreases (or at best stays constant). If this inequality ever fails, you have found a perpetual motion machine---and we know those do not exist.

The $\mathfrak{D}$ term is crucial. It is not enough to say energy is bounded. We need to know *where the energy goes*. The dissipation functional tells us: it goes into heat, into friction, into the irreversible background noise of the universe. Without tracking dissipation, you cannot rule out the possibility that energy concentrates into smaller and smaller regions until it blows up locally even while staying globally bounded.
:::

:::{prf:axiom} Axiom Rec (Recovery)
:label: ax-recovery

Discrete events are finite: $N(J) < \infty$ for any bounded interval $J$.

**Enforced by:** {prf:ref}`def-node-zeno` --- Certificate $K_{\text{Rec}_N}^+$
:::

:::{div} feynman-prose
Axiom Rec is Zeno's paradox taken seriously. Remember Zeno? He argued that to cross a room, you must first go halfway, then half of what remains, then half again, and so on forever. Since you can never complete infinitely many tasks, motion is impossible.

Of course, motion *is* possible---the sum $\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \cdots$ converges to 1. But here is the thing: that argument only works because each step takes proportionally less time. If each step took the same amount of time, you really could not cross the room.

In dynamical systems, discrete events---jumps, collisions, phase transitions---are like Zeno's steps. If infinitely many of them pile up in finite time, the system loses predictability at that moment. Everything before it leads to a single instant where infinitely many things happen, and you cannot say what comes next.

Axiom Rec says: this cannot happen. In any bounded time interval, only finitely many discrete events occur. This sounds obvious, but it is surprisingly easy to violate. Think of a bouncing ball with no energy loss---it bounces infinitely many times before coming to rest (in the idealized model). Real balls lose energy, so they eventually stop bouncing entirely. The axiom forces us to include that energy loss.
:::

(sec-duality-constraints)=
## Duality Constraints

:::{div} feynman-prose
The duality constraints are more subtle. They deal with what happens when you zoom in or zoom out---does the system look similar at different scales? And when energy stays bounded, does it actually *go* somewhere definite, or does it spread out and disappear?

These questions connect to some of the deepest ideas in modern physics: renormalization, scale invariance, the infrared and ultraviolet behavior of field theories. But the basic idea is simple enough. When you have a bounded sequence of states, you want to be able to extract a limit. If you cannot---if things keep escaping to infinity or spreading thinner and thinner forever---then you have lost control of the dynamics.
:::

:::{prf:axiom} Axiom C (Compactness)
:label: ax-compactness

Bounded energy sequences admit convergent subsequences modulo the symmetry group $G$:

$$
\sup_n \Phi(u_n) < \infty \implies \exists (n_k), \, g_k \in G: \, g_k \cdot u_{n_k} \to u_\infty
$$

**Enforced by:** {prf:ref}`def-node-compact` --- Certificate $K_{C_\mu}^+$ (or dispersion via $K_{C_\mu}^-$)
:::

:::{div} feynman-prose
Axiom C is about *catching* things. Imagine you have a sequence of photographs of a ball. In each photo, the ball has roughly the same size (bounded energy), but it might be in different positions. Compactness says: if you are allowed to re-center each photo (that is what the symmetry group $G$ does), you can find a subsequence where the ball converges to a definite position.

Why "modulo $G$"? Because many physical systems have symmetries. A wave packet traveling to the right looks the same as one traveling to the left, just shifted. If you insist on convergence without shifting, you might fail even when the physics is perfectly well-behaved---the wave packet just keeps moving. But if you allow yourself to "follow" the packet by translating, you see that it maintains its shape.

The alternative to concentration is *dispersion*: the energy spreads out uniformly over larger and larger regions until there is nothing left at any particular place. Dispersion is actually a *good* outcome---it means the system does not blow up, it just fades away. The Sieve accepts dispersion as an alternative to concentration. Both are fine; what is not fine is energy that neither concentrates nor disperses but oscillates forever without settling down.
:::

:::{prf:axiom} Axiom SC (Scaling)
:label: ax-scaling

Dissipation scales faster than time: $\alpha > \beta$, where $\alpha$ is the energy scaling dimension and $\beta$ is the dissipation scaling dimension.

**Enforced by:** {prf:ref}`def-node-scale` --- Certificate $K_{SC_\lambda}^+$
:::

:::{div} feynman-prose
Axiom SC is the *subcriticality* condition, and it is where dimensional analysis becomes powerful. Here is the idea: when you zoom in on a system, both energy and dissipation change. Energy might go as $\lambda^\alpha$ when you zoom by factor $\lambda$, while dissipation goes as $\lambda^\beta$.

If $\alpha > \beta$, then at small scales, energy dominates dissipation. This is *bad* for regularity---it means zooming in amplifies problems rather than smoothing them out. The system is *supercritical*.

If $\alpha < \beta$, dissipation wins at small scales. This is *good*---small-scale features get damped out. The system is *subcritical*.

The boundary case $\alpha = \beta$ is *critical*, and that is where life gets interesting. Critical systems often have beautiful scale-invariant structures---fractals, conformal field theories, critical phenomena at phase transitions. But they are delicate. Proving regularity at criticality requires special techniques.

The physical intuition: think of turbulence. In three dimensions, the Navier-Stokes equations are supercritical---the energy cascade to small scales is too fast for viscosity to control. This is why the Clay Millennium Prize for Navier-Stokes remains unclaimed. In two dimensions, the equations are subcritical, and we have global regularity. Same equations, different dimensions, different scaling, completely different behavior.
:::

(sec-symmetry-constraints)=
## Symmetry Constraints

:::{div} feynman-prose
The symmetry constraints might seem like the odd ones out---what do symmetries have to do with blow-up? But here is the deep connection: symmetries create *degeneracies*, and degeneracies mean you can get stuck.

Think of a ball on a perfectly flat table. It is in equilibrium everywhere. Give it a tiny push, and it rolls forever without settling down. Now add a gentle bowl shape to the table. Suddenly there is a unique equilibrium at the bottom, and any perturbation brings the ball back. The bowl's curvature breaks the translation symmetry and creates *stiffness*.

In infinite dimensions, this problem becomes critical. If your energy landscape has flat directions---zero modes, Goldstone bosons, whatever you want to call them---the system can slide along those directions indefinitely. The Łojasiewicz-Simon inequality quantifies stiffness: it says equilibria are isolated (or at least discrete) in a precise sense.
:::

:::{prf:axiom} Axiom LS (Stiffness)
:label: ax-stiffness

The Łojasiewicz-Simon inequality holds near equilibria, ensuring a mass gap:

$$
\inf \sigma(L) > 0
$$

where $L$ is the linearized operator at equilibrium.

**Enforced by:** {prf:ref}`def-node-stiffness` --- Certificate $K_{LS_\sigma}^+$
:::

:::{div} feynman-prose
The mass gap $\inf \sigma(L) > 0$ says: the smallest eigenvalue of the linearized operator is bounded away from zero. No arbitrarily soft modes. No infinitely gentle perturbations that cost zero energy.

Why does this matter for regularity? Because if there are zero modes, small perturbations can grow without bound in that direction. The system might not blow up dramatically, but it drifts away forever, never settling down. The Sieve needs convergence to equilibrium, and convergence needs stiffness.

This is why the Yang-Mills mass gap problem is so important. If the Yang-Mills theory has a mass gap, then quantum chromodynamics (the theory of quarks and gluons) is mathematically consistent at low energies. If not, the theory might be sick in some subtle way. The Sieve would catch this: no mass gap means Axiom LS fails, and we route to Mode S.D (Stiffness Breakdown).
:::

:::{prf:axiom} Axiom GC (Gradient Consistency)
:label: ax-gradient-consistency

Gauge invariance and metric compatibility: the control $T(u)$ matches the disturbance $d$.

**Enforced by:** {prf:ref}`def-node-align` --- Certificate $K_{GC_T}^+$
:::

:::{div} feynman-prose
Axiom GC is about *consistency between different descriptions of the same physics*. In gauge theories, you can describe the same physical state using different mathematical representatives---different choices of gauge. The physics should not depend on your choice.

But here is the subtle point: your dynamics should respect this redundancy. If you have a control input $T(u)$ trying to counteract a disturbance $d$, they should be "speaking the same language." If $d$ is gauge-covariant and $T(u)$ is not, you get misalignment: the control thinks it is canceling the disturbance, but from a gauge-invariant perspective, it is actually making things worse.

This becomes critical in field theories with local symmetries. The electromagnetic field is gauge-invariant. If your numerical method for solving Maxwell's equations introduces gauge-dependent artifacts, those artifacts can grow and contaminate the solution. Axiom GC catches this: it ensures your control matches your disturbance in a gauge-consistent way.
:::

(sec-topology-constraints)=
## Topology Constraints

:::{div} feynman-prose
The topology constraints are where things get mathematically interesting. Topology is the study of properties that do not change under continuous deformation---you can stretch, bend, and twist, but you cannot tear or glue. A coffee cup and a donut are topologically the same (both have one hole), but neither is the same as a sphere (no holes).

Why does topology matter for dynamics? Because topological features are *protected*. If your system has to change topology to reach equilibrium, it must pay a price. That price is measured by an action gap---the minimum energy required to tunnel from one topological sector to another. If the energy is below the gap, the topology cannot change, and the system is trapped in its current sector.

This has profound implications. It means some singularities are topologically inevitable: if you start in the wrong sector, no amount of smooth evolution will save you. You need surgery---a controlled topology change---to escape. The Sieve handles this by routing topologically obstructed states to surgery nodes.
:::

:::{prf:axiom} Axiom TB (Topological Background)
:label: ax-topology

Topological sectors are separated by an action gap:

$$
[\pi] \in \pi_0(\mathcal{C})_{\mathrm{acc}} \implies E < S_{\min} + \Delta
$$

**Enforced by:** {prf:ref}`def-node-topo` --- Certificate $K_{TB_\pi}^+$
:::

:::{div} feynman-prose
Here is the picture: configuration space is divided into disconnected regions (topological sectors), like islands in an ocean. To get from one island to another, you have to swim through the ocean---and swimming costs energy $S_{\min}$.

Axiom TB says: you can only access sectors where your energy exceeds the swimming cost. If $E < S_{\min} + \Delta$, you are stuck on your island. The $\Delta$ is a safety margin; it accounts for quantum tunneling and thermal fluctuations that let you cross small barriers.

This is how instantons work in quantum field theory. An instanton is a path through configuration space that changes topology---like swimming between islands. The action of the instanton measures the swimming cost. In the semiclassical approximation, the transition rate goes like $e^{-S_{\mathrm{instanton}}/\hbar}$---exponentially suppressed when the action is large.

For regularity, we want the system to stay on its island. If it can access dangerous sectors with wild topology, bad things happen. Axiom TB bounds the accessible sectors.
:::

:::{prf:axiom} Axiom Cap (Capacity)
:label: ax-capacity

Capacity density bounds prevent concentration on thin sets:

$$
\operatorname{codim}(S) \geq 2 \implies \operatorname{Cap}_H(S) = 0
$$

**Enforced by:** {prf:ref}`def-node-geom` --- Certificate $K_{\text{Cap}_H}^+$
:::

:::{div} feynman-prose
Axiom Cap is about the *size* of singularities. Even if singularities exist, they can be harmless if they are small enough. "Small" here means codimension at least 2---the singular set is two dimensions smaller than the ambient space.

Why codimension 2? Think about it in three dimensions. A point has codimension 3: it is zero-dimensional in 3D space. A line has codimension 2. A surface has codimension 1. Now, if you have a random path in 3D, the probability of hitting a point is zero. The probability of hitting a line is also zero (you have to aim exactly). But a surface will generically be crossed.

For PDEs, the same logic applies. If singularities live on a set of codimension 2 or more, generic trajectories avoid them. The capacity $\operatorname{Cap}_H$ makes this precise: it measures how hard it is to "see" a set from a potential-theoretic viewpoint. Sets of codimension $\geq 2$ have zero capacity---you cannot charge them up in a meaningful way.

This is why isolated singularities are often removable: they are points, hence codimension $n$ where $n \geq 3$ typically. But singularities along curves or surfaces are more dangerous.
:::

:::{div} feynman-prose
Now we come to something beautiful: the Tits Alternative. This is one of those theorems that says "there are only a few kinds of geometry in the world." Not just one or two---but the classification is *finite*. And each kind of geometry corresponds to a different phase of matter.

Let me tell you what this is really about. When you have a discrete structure---a network, a graph, something you can count---it has a *growth rate*. How many points can you reach in $r$ steps? If the answer is roughly $r^d$ for some dimension $d$, you have polynomial growth. This is the "crystal" phase: orderly, predictable, finite-dimensional.

But there are other possibilities. Hyperbolic structures grow exponentially, but in a very organized way---like a branching tree. These are the "liquid" phase: infinite but still compressible, still describable by finite rules. And then there are expanders---networks where everything is connected to everything else so efficiently that information spreads instantly. These are the "gas" phase: too chaotic to compress.

The Tits Alternative says: *every discrete group is either virtually nilpotent (crystal) or contains a free subgroup (liquid/gas)*. There is no fourth option. This is profound. It means the Sieve's classification is not just convenient---it is *mathematically exhaustive*.
:::

:::{prf:axiom} Axiom Geom (Geometric Structure License --- Tits Alternative)
:label: ax-geom-tits

The Thin Kernel's simplicial complex $K$ must satisfy the **Discrete Tits Alternative**: it admits either polynomial growth (Euclidean/Nilpotent), hyperbolic structure (Logic/Free Groups), or is a CAT(0) space (Higher-Rank Lattices).

**Predicate**:

$$
P_{\mathrm{Geom}}(K) := (\operatorname{Growth}(K) \leq \operatorname{Poly}(d)) \lor (\delta_{\mathrm{hyp}}(K) < \infty) \lor (\operatorname{Cone}(K) \in \operatorname{Buildings})
$$

**Operational Check** (Node 7c):
1. **Polynomial Growth Test**: If ball growth satisfies $|B_r(x)| \sim r^d$ for some $d < \infty$, emit $K_{\mathrm{Geom}}^{+}(\text{Poly})$. *(Euclidean/Nilpotent structures)*
2. **Hyperbolic Test**: Compute Gromov $\delta$-hyperbolicity constant. If $\delta < \epsilon \cdot \operatorname{diam}(K)$ for small $\epsilon$, emit $K_{\mathrm{Geom}}^{+}(\text{Hyp})$. *(Logic trees/Free groups)*
3. **CAT(0) Test**: Check triangle comparison inequality $d^2(m,x) \leq \frac{1}{2}d^2(y,x) + \frac{1}{2}d^2(z,x) - \frac{1}{4}d^2(y,z)$ for all triangles. If satisfied, emit $K_{\mathrm{Geom}}^{+}(\text{CAT0})$. *(Higher-rank lattices/Yang-Mills)*

**Rejection Mode**:
If all three tests fail (exponential growth AND fat triangles AND no building structure), the object is an **Expander Graph** (thermalized, no coherent structure). Emit $K_{\mathrm{Geom}}^{-}$ and route to **Mode D.D (Dispersion)** unless rescued by Spectral Resonance ({prf:ref}`ax-spectral-resonance`).

**Physical Interpretation**:
The Tits Alternative is the **universal dichotomy** for discrete geometric structures:
- **Polynomial/CAT(0)**: Structured (Crystal phase) → Admits finite description
- **Hyperbolic**: Critical (Liquid phase) → Admits logical encoding (infinite but compressible)
- **Expander**: Thermal (Gas phase) → No compressible structure (unless arithmetically constrained)

**Certificate**:

$$
K_{\mathrm{Geom}}^{+} = (\operatorname{GrowthType} \in \{\text{Poly}, \text{Hyp}, \text{CAT0}\}, \, \text{evidence}, \, \text{parameters})
$$

**Literature:** {cite}`Tits72` (Tits Alternative); {cite}`Gromov87` (Hyperbolic groups); {cite}`BridsonHaefliger99` (CAT(0) geometry); {cite}`Lubotzky94` (Expander graphs)

**Enforced by:** Node 7c (Geometric Structure Check) --- Certificate $K_{\mathrm{Geom}}^{\pm}$
:::

:::{div} feynman-prose
But wait---what about expanders? They fail all three tests. Does that mean they are hopeless? Not quite. There is one more chance for redemption: Spectral Resonance.
:::

:::{prf:axiom} Axiom Spec (Spectral Resonance --- Arithmetic Rescue)
:label: ax-spectral-resonance

An object **rejected** by {prf:ref}`ax-geom-tits` as an Expander (thermal chaos) is **re-admitted** if it exhibits **Spectral Rigidity** --- non-decaying Bragg peaks indicating hidden arithmetic structure.

**Predicate**:
Let $\rho(\lambda)$ be the spectral density of states for the combinatorial Laplacian $\Delta_K$. Define the **Structure Factor**:

$$
S(t) := \left|\int e^{i\lambda t} \rho(\lambda) \, d\lambda\right|^2
$$

The object passes the **Spectral Resonance Test** if:

$$
\exists \{p_i\}_{i=1}^N : \, \lim_{T \to \infty} \frac{1}{T} \int_0^T S(t) \, dt > \eta_{\mathrm{noise}}
$$

where $\{p_i\}$ are **quasi-periods** (resonances) and $\eta_{\mathrm{noise}}$ is the random matrix theory baseline.

**Operational Check** (Node 7d):
1. **Eigenvalue Computation**: Compute spectrum $\operatorname{spec}(\Delta_K) = \{\lambda_i\}$
2. **Level Spacing Statistics**: Compute nearest-neighbor spacing distribution $P(s)$
   - **Poisson** $P(s) \sim e^{-s}$: Random (Gas phase) → Fail
   - **GUE/GOE** $P(s) \sim s^\beta e^{-cs^2}$: Quantum chaos (Critical) → Pass if Trace Formula detected
3. **Trace Formula Detection**: Check for periodic orbit formula:

$$
\rho(\lambda) = \rho_{\mathrm{Weyl}}(\lambda) + \sum_{\gamma \text{ periodic}} A_\gamma \cos(\lambda \ell_\gamma)
$$

   If present, emit $K_{\mathrm{Spec}}^{+}(\text{ArithmeticChaos})$

**Physical Interpretation**:
This distinguishes:
- **Arithmetic Chaos** (e.g., Riemann zeros, Quantum graphs, Maass forms): Expander-like growth BUT spectral correlations follow number-theoretic laws → **Admits arithmetic encoding**
- **Thermal Chaos** (Random matrices, generic expanders): No long-range spectral correlations → **Truly random (Gas phase)**

**Connection to Number Theory**:
The **Selberg Trace Formula** and **Explicit Formula** for the Riemann zeta function are instances of spectral resonance:

$$
\psi(x) = x - \sum_\rho \frac{x^\rho}{\rho} - \log(2\pi)
$$

where $\rho$ are the non-trivial zeros. The zeros exhibit GUE statistics (quantum chaos) but are **arithmetically structured**.

**Certificate**:

$$
K_{\mathrm{Spec}}^{+} = (\operatorname{LevelStatistics} = \text{GUE/GOE}, \, \operatorname{TraceFormula}, \, \{p_i\}, \, \eta_{\mathrm{signal}}/\eta_{\mathrm{noise}})
$$

**Literature:** {cite}`Selberg56` (Trace formula); {cite}`MontgomeryOdlyzko73` (Pair correlation conjecture); {cite}`Sarnak95` (Quantum chaos); {cite}`KatzSarnak99` (Random matrix theory)

**Enforced by:** Node 7d (Spectral Resonance Check) --- Certificate $K_{\mathrm{Spec}}^{\pm}$
:::

:::{div} feynman-prose
Here is the beautiful idea behind Axiom Spec. Some systems look random at first glance but have hidden order. The Riemann zeros are the paradigmatic example: they seem scattered chaotically along the critical line, but their statistics match the eigenvalues of random matrices with exquisite precision. This is not randomness---it is *arithmetic chaos*.

The spectral resonance test looks at the eigenvalues of the Laplacian and asks: do they show long-range correlations? If the level spacings follow a Poisson distribution, you have genuine randomness---the gas phase. But if they follow GUE (Gaussian Unitary Ensemble) statistics, you have quantum chaos with hidden structure.

The clincher is the trace formula. In arithmetic chaos, the eigenvalue density contains oscillations at discrete frequencies corresponding to periodic orbits. These oscillations are like Bragg peaks in crystallography---they reveal underlying order. If the trace formula holds, the expander is not truly thermal; it is arithmetically structured and admits a finite description.

This is why the Sieve can handle objects like quantum graphs and arithmetic surfaces: they fail the naive geometry test but pass the spectral test. The combination of Tits + Spectral gives complete coverage of discrete structures.
:::

:::{prf:remark} Universal Coverage via Tits + Spectral
:label: rem-universal-coverage

The combination of {prf:ref}`ax-geom-tits` and {prf:ref}`ax-spectral-resonance` achieves **universal coverage** of discrete structures:

| Structure Class | Tits Test | Spectral Test | Verdict | Example |
|----------------|-----------|---------------|---------|---------|
| Euclidean/Nilpotent | ✓ Poly | N/A | **REGULAR** (Crystal) | $\mathbb{Z}^d$, Heisenberg group |
| Hyperbolic/Free | ✓ Hyp | N/A | **PARTIAL** (Liquid) | Free groups, Logic trees |
| Higher-Rank/CAT(0) | ✓ CAT0 | N/A | **REGULAR** (Structured) | $SL(n,\mathbb{Z})$ ($n \geq 3$), Buildings |
| Arithmetic Chaos | ✗ Expander | ✓ Resonance | **PARTIAL** (Arithmetic) | Riemann zeros, Quantum graphs |
| Thermal Chaos | ✗ Expander | ✗ Random | **HORIZON** (Gas) | Random matrices, Generic expanders |

This resolves the **completeness gap**: every discrete structure routes to exactly one verdict.
:::

(sec-boundary-constraints)=
## Boundary Constraints

:::{div} feynman-prose
The final family of axioms deals with boundaries---how the system couples to its environment. This is where we stop pretending our system is isolated and acknowledge that real systems exchange energy, information, and matter with the outside world.

Think about it: almost every interesting system is open. Your body exchanges heat, food, and waste with the environment. A computer exchanges electrical energy and data with its surroundings. Even the universe, if it has a boundary (cosmological horizon?), has flux through that boundary.

The boundary axioms encode three fundamental requirements: (1) the boundary is not trivial---the system is genuinely open, not just a closed system with a pretend boundary; (2) the flux through the boundary is bounded---you cannot pump in infinite energy; and (3) the input is sufficient---the system has enough resources to keep operating. Violate any of these, and you get pathological behavior: overload, starvation, or misalignment between what you are trying to control and what you are actually controlling.
:::

The Boundary Constraints enforce coupling between bulk dynamics and environmental interface via the Thin Interface $\partial^{\mathrm{thin}} = (\mathcal{B}, \operatorname{Tr}, \mathcal{J}, \mathcal{R})$.

:::{prf:axiom} Axiom Bound (Input/Output Coupling)
:label: ax-boundary

The system's boundary morphisms satisfy:
- $\mathbf{Bound}_\partial$: $\operatorname{Tr}: \mathcal{X} \to \mathcal{B}$ is not an equivalence (open system) --- {prf:ref}`def-node-boundary`
- $\mathbf{Bound}_B$: $\mathcal{J}$ factors through a bounded subobject $\mathcal{J}: \mathcal{B} \to \underline{[-M, M]}$ --- {prf:ref}`def-node-overload`
- $\mathbf{Bound}_{\Sigma}$: The integral $\int_0^T \mathcal{J}_{\mathrm{in}}$ exists as a morphism in $\operatorname{Hom}(\mathbf{1}, \underline{\mathbb{R}}_{\geq r_{\min}})$ --- {prf:ref}`def-node-starve`
- $\mathbf{Bound}_{\mathcal{R}}$: The **reinjection diagram** commutes:

$$
\mathcal{J}_{\mathrm{out}} \simeq \mathcal{J}_{\mathrm{in}} \circ \mathcal{R} \quad \text{in } \operatorname{Hom}_{\mathcal{E}}(\mathcal{B}, \underline{\mathbb{R}})
$$

**Enforced by:** {prf:ref}`def-node-boundary`, {prf:ref}`def-node-overload`, {prf:ref}`def-node-starve`, {prf:ref}`def-node-align`
:::

:::{div} feynman-prose
Let me unpack these four boundary conditions. They are really about different failure modes:

**$\mathbf{Bound}_\partial$ (Non-trivial boundary):** The trace map is not an isomorphism. This sounds technical, but it just means the boundary actually matters. If $\operatorname{Tr}$ were an equivalence, the boundary would contain the same information as the bulk, and you would have a closed system pretending to be open.

**$\mathbf{Bound}_B$ (Bounded flux):** The flux through the boundary is bounded by some $M$. You cannot inject infinite energy per unit time. Without this, you could blow up any system just by overloading it from outside---hardly a failure of the internal dynamics.

**$\mathbf{Bound}_\Sigma$ (Sufficient input):** The total input over time must be at least $r_{\min}$. This prevents starvation. A robot with no battery cannot execute its control law, no matter how clever.

**$\mathbf{Bound}_\mathcal{R}$ (Reinjection consistency):** What goes out comes back in, via the reinjection map $\mathcal{R}$. This is about conservation: if you have a Fleming-Viot process where particles are killed at the boundary and reborn elsewhere, the total mass should be conserved. The reinjection diagram formalizes this.
:::

:::{prf:remark} Reinjection Boundaries (Fleming-Viot)
:label: rem-reinjection

When $\mathcal{R} \not\simeq 0$, the boundary acts as a **non-local transport morphism** rather than an absorbing terminal object. This captures:
- **Fleming-Viot processes:** The reinjection factors through the **probability monad** $\mathcal{P}: \mathcal{E} \to \mathcal{E}$
- **McKean-Vlasov dynamics:** $\mathcal{R}$ depends on global sections $\Gamma(\mathcal{X}, \mathcal{O}_\mu)$
- **Piecewise Deterministic Markov Processes:** $\mathcal{R}$ is a morphism in the Kleisli category of the probability monad

The Sieve verifies regularity by checking **Axiom Rec** at the boundary:
1. **{prf:ref}`def-node-boundary`:** Detects that $\mathcal{J} \neq 0$ (non-trivial exit flux)
2. **{prf:ref}`def-node-starve`:** Verifies $\mathcal{R}$ preserves the **total mass section** ($K_{\mathrm{Mass}}^+$)

Categorically, this defines a **non-local boundary condition** as a span:

$$
\mathcal{X} \xleftarrow{\operatorname{Tr}} \mathcal{B} \xrightarrow{\mathcal{R}} \mathcal{P}(\mathcal{X})
$$

The resulting integro-differential structure is tamed by **Axiom C** applied to the Wasserstein $\infty$-stack $\mathcal{P}_2(\mathcal{X})$.
:::

:::{div} feynman-prose
And there you have it---the complete axiom system. Let me step back and tell you what we have accomplished.

We started by asking: what could possibly go wrong with a dynamical system? The answer came in five flavors: Conservation (energy blow-up, event accumulation), Duality (failure to concentrate or disperse, wrong scaling), Symmetry (soft modes, gauge inconsistency), Topology (inaccessible sectors, fat singularities, unclassifiable geometry), and Boundary (overload, starvation, misalignment).

Each axiom is a *guard* against one specific failure mode. Pass all the guards, and you have proven regularity. Fail a guard, and you know exactly what went wrong and where to look for trouble.

The remarkable thing is that this list is *complete*. Not complete in the sense that we can prove everything---some problems are genuinely undecidable, and we route those to HORIZON. But complete in the sense that every way a well-posed dynamical system can fail is captured by some axiom. The Tits Alternative and Spectral Resonance together cover all discrete structures. The capacity and topology axioms cover all geometric singularities. The boundary axioms cover all coupling pathologies.

This is the power of the axiomatic approach: instead of proving regularity from scratch for each new system, you identify which axioms are satisfied and inherit the general theory. The Sieve is just the systematic machine for checking these axioms and routing systems to the appropriate verdict.

Now you understand the blueprint. What remains is to see how the Sieve implements these checks in practice---and that is what the next part of this story is about.
:::

:::{admonition} Quick Reference: The Axiom Checklist
:class: feynman-added tip

| Axiom | What It Checks | Failure Mode If Violated |
|-------|----------------|--------------------------|
| **D** (Dissipation) | Energy decreases or stays bounded | C.E: Energy blow-up |
| **Rec** (Recovery) | Finite events in finite time | C.C: Zeno/event accumulation |
| **C** (Compactness) | Bounded energy $\Rightarrow$ convergence | C.D: Geometric collapse |
| **SC** (Scaling) | Subcritical dimension | S.E: Supercritical cascade |
| **LS** (Stiffness) | Spectral gap at equilibria | S.D: Stiffness breakdown |
| **GC** (Gradient Consistency) | Gauge-compatible dynamics | B.C: Misalignment |
| **TB** (Topological Background) | Accessible sectors bounded | T.E: Topological twist |
| **Cap** (Capacity) | Singularities have zero capacity | C.D: Fat singularity |
| **Geom** (Tits Alternative) | Classifiable discrete geometry | D.D: Thermal chaos |
| **Spec** (Spectral Resonance) | Arithmetic structure if expander | D.D: Thermal chaos |
| **Bound** (Boundary) | Bounded flux, sufficient input | B.E/B.D: Overload/Starvation |

This is your pre-flight checklist. Every system must pass every applicable axiom to certify REGULAR.
:::
