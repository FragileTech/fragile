---
title: "Permits and Preconditions"
---

# Permits and Preconditions

(sec-weakest-precondition)=
## The Weakest Precondition Principle

:::{div} feynman-prose
Here is the key idea that makes the whole Sieve machinery work, and I want you to understand it viscerally because it changes how you think about proving regularity.

Traditionally, when mathematicians attack a PDE problem, they assume smoothness up front and then try to prove it doesn't break. This is backwards. It is like asking "will this bridge hold?" while assuming the bridge is infinitely strong. What you really want is a machine where you pour in the minimal ingredients---the local checks, the interface predicates---and the machine tells you whether regularity holds or fails.

That is exactly what we have here. You do not prove that solutions are smooth. You do not classify singularities in advance. You implement a few computable checks---scaling exponents, dimension estimates, Lojasiewicz constants---and then you run the Sieve. The verdict emerges from the computation itself. If it says YES, you have regularity. If it says NO, you get a certificate explaining exactly what went wrong and where.

This is Dijkstra's insight applied to analysis: the weakest precondition that guarantees a postcondition is precisely what you compute, not what you assume.
:::

The interface formalism of {prf:ref}`def-interface-permit` embodies a fundamental design principle: **regularity is an output, not an input**.

:::{prf:theorem} [RESOLVE-WeakestPre] Weakest Precondition Principle
:label: mt-resolve-weakest-pre

To instantiate the Structural Sieve for a dynamical system, users need only:

1. **Map Types**: Define the state space $X$, height functional $\Phi$, dissipation $\mathfrak{D}$, and symmetry group $G$.

2. **Implement Interfaces**: Provide computable formulas for each interface predicate $\mathcal{P}_n$ relevant to the problem:
   - Scaling exponents $\alpha, \beta$ (for $\mathrm{SC}_\lambda$)
   - Dimension estimates $\dim(\Sigma)$ (for $\mathrm{Cap}_H$)
   - Łojasiewicz exponent $\theta$ (for $\mathrm{LS}_\sigma$)
   - Topological invariant $\tau$ (for $\mathrm{TB}_\pi$)
   - etc.

3. **Run the Sieve**: Execute the Structural Sieve algorithm ({prf:ref}`def-sieve-functor`).

**The Sieve automatically determines regularity.** Users do not need to:
- Prove global existence a priori
- Assume solutions are smooth
- Know where singularities occur
- Classify all possible blow-up profiles in advance

The verdict $\mathcal{V} \in \{\text{YES}, \text{NO}, \text{Blocked}\}$ emerges from the certificate-driven computation. NO verdicts carry typed certificates ($K^{\mathrm{wit}}$ or $K^{\mathrm{inc}}$) distinguishing refutation from inconclusiveness.

**Literature:** Dijkstra's weakest precondition calculus {cite}`Dijkstra76`; predicate transformer semantics {cite}`Back80`.
:::

:::{prf:remark} Computational Semantics
:label: rem-computational-semantics

The Weakest Precondition Principle gives the framework its **operational semantics**:

| User Provides | Sieve Computes |
|---------------|----------------|
| Interface implementations | Node verdicts |
| Type mappings | Barrier certificates |
| Local predicates | Global regularity/singularity |
| Computable checks | Certificate chains |

This is analogous to how a type checker requires type annotations but derives type safety, or how a SAT solver requires a formula but derives satisfiability.
:::

:::{prf:corollary} Separation of Concerns
:label: cor-separation-concerns

The interface formalism separates:
- **Domain expertise** (implementing $\mathcal{P}_n$ for specific PDEs)
- **Framework logic** (the Sieve algorithm and metatheorems)
- **Certificate verification** (checking that certificates satisfy their specifications)

A researcher can contribute a new interface implementation without understanding the full Sieve machinery, and the framework can be extended with new metatheorems without modifying existing implementations.
:::

:::{div} feynman-prose
Let me put this differently. Think of a type checker in programming. You write type annotations on your functions. You do not prove that your program is type-safe---the type checker does that for you, automatically. And when it fails, it tells you exactly where the types do not match.

The Sieve works the same way. Domain expertise goes into implementing the local predicates---you know your PDE, you know what scaling exponent to compute, you know what dimension estimate is relevant. Framework logic is the Sieve algorithm itself, which chains these local checks into global conclusions. Certificate verification is automatic: you can replay the proof and check that every step follows.

This separation is not just elegant. It is practical. A geometer working on Ricci flow does not need to understand how the Sieve handles dispersive equations. She implements her interface predicates, runs the Sieve, gets her certificate. The framework holds it all together.
:::

---

(sec-rigidity-recovery)=
## Rigidity & Recovery Metatheorems

:::{div} feynman-prose
Now we come to what I think is one of the most satisfying parts of this whole framework. We have seen that the Sieve can tell you whether regularity holds. But what if you want more? What if you want to know not just that things do not blow up, but exactly how they converge? What if you want canonical structures, uniqueness, explicit formulas?

That is where these strengthening metatheorems come in. Think of them as upgrades you can unlock. The base Sieve gives you a verdict. But if your problem has additional structure---if singularities are "tame" enough to classify, if you have enough rigidity for canonical Lyapunov functions, if everything is computable---then you get stronger conclusions automatically.

It is like a video game. Beat the first level and you can play the game. Beat it with enough style points and you unlock bonus content. Here, the "style points" are structural certificates---tameness, rigidity, effectivity---and the bonus content is uniqueness theorems, convergence rates, algorithmic decidability.
:::

This section defines **strengthening metatheorems** that upgrade soft interface permits to stronger analytical guarantees. These metatheorems provide:
- **Tame recovery** ($K_{\text{Tame}}$): Singularity classification and surgery admissibility
- **Rigidity** ($K_{\text{Rigid}}$): Canonical Lyapunov functions and convergence rates
- **Effectivity** ($K_{\text{Eff}}$): Algorithmic decidability of lock exclusion

---

### Tame Singularity and Recovery

**Purpose**: Guarantee that the "singularity middle-game" is **uniformly solvable**:
- profiles can be classified (finite or tame),
- surgery admissibility can be certified (possibly up to equivalence),
- recovery steps can be executed systematically.

#### Profile Classification Trichotomy

The Tame Certificate ($K_{\text{Tame}}$) requires a verified metatheorem that, given a profile $V$, outputs exactly one of:

1. **Finite canonical library**: $K_{\mathrm{lib}}: \{V_1,\dots,V_N\}$ and membership certificate $K_{\mathrm{can}}: V\sim V_i$

2. **Tame stratification**: $K_{\mathrm{strat}}$: finite stratification $\bigsqcup_{k=1}^K \mathcal P_k\subseteq\mathbb R^{d_k}$ and classifier $K_{\mathrm{class}}$

3. **NO certificate (wild or inconclusive)**: $K_{\mathrm{Surg}}^{\mathrm{wild}}$ or $K_{\mathrm{Surg}}^{\mathrm{inc}}$: classification not possible under current Rep/definability regime (wildness witness or method exhaustion)

**Requirement**: For types with $K_{\text{Tame}}$, outcomes (1) or (2) always occur for admissible profiles (i.e., classification failure is ruled out).

#### Surgery Admissibility Trichotomy

The Tame Certificate requires a verified metatheorem that, given Surgery Data $(\Sigma, V, \lambda(t), \mathrm{Cap}(\Sigma))$, outputs exactly one of:

1. **Admissible**: $K_{\mathrm{adm}}$: canonical + codim bound + cap bound (as in {prf:ref}`mt-resolve-admissibility`)

2. **Admissible up to equivalence**: $K_{\mathrm{adm}^\sim}$: after an admissible equivalence move (YES$^\sim$), the singularity becomes admissible

3. **Not admissible**: $K_{\neg\mathrm{adm}}$: explicit reason (cap too large, codim too small, classification failure)

**Requirement**: For types with $K_{\text{Tame}}$, if a singularity is encountered, it is either admissible (1) or admissible up to equivalence (2). The "not admissible" case becomes a certified rare/horizon mode for types without the Tame Certificate.

#### Structural Surgery Availability

The Tame Certificate includes the premise needed to invoke the **Structural Surgery Principle** uniformly:
- surgery operator exists when admissible,
- flow extends past surgery time,
- finite surgeries on finite windows (or well-founded complexity decrease).

**Certificate produced**: $K_{\text{Tame}}$ with `HasRecoveryEngine(T)`.

---

### Rigidity and Canonical Energy

**Purpose**: Upgrade from "we can recover" to "we can prove uniqueness, convergence, and canonical Lyapunov structure."

The Rigidity Certificate ($K_{\text{Rigid}}$) enables Perelman-like monotone functionals, canonical Lyapunov uniqueness, and rate results.

#### Local Stiffness Core Regime

With the Rigidity Certificate, any time the system is in the safe neighborhood $U$, the stiffness check can be made **core**, not merely "blocked":
- either StiffnessCheck = YES directly,
- or BarrierGap(Blocked) plus standard promotion rules yields LS-YES (often with $\theta=1/2$ when a gap exists).

**Certificate produced**: $K_{\text{Rigid}}$ with `LS-Core`.

#### Canonical Lyapunov Functional is available

With the Rigidity Certificate, the conditions needed for canonical Lyapunov recovery hold in the relevant regime:
- existence of a Lyapunov-like progress function (already in base level),
- plus rigidity/local structure sufficient for **uniqueness up to monotone reparameterization**.

**Level-up certificate**: `Lyapunov-Canonical`.

#### Quantitative convergence upgrades

With the Rigidity Certificate, you may state and reuse quantitative convergence metatheorems:
- exponential or polynomial rates (depending on LS exponent),
- uniqueness of limit objects (no wandering among equilibria),
- stability of attractor structure.

**Level-up certificate**: `Rates`.

#### Stratified Lyapunov across surgery times

With the Rigidity Certificate, even with surgeries, you can define a **piecewise / stratified Lyapunov**:
- decreasing between surgeries,
- jumps bounded by a certified jump rule,
- global progress because surgery count is finite or complexity decreases.

**Level-up certificate**: `Lyapunov-Stratified`.

#### Lyapunov Existence via Value Function

With the Rigidity Certificate, with validated interface permits $D_E$, $C_\mu$, and $\mathrm{LS}_\sigma$, a canonical Lyapunov functional exists and can be explicitly constructed.

**Level-up certificate:** `Lyapunov-Existence`.

:::{div} feynman-prose
Now, here is something beautiful. When you have the right certificates in place, not only does a Lyapunov functional exist---it is essentially unique, and we can write it down explicitly.

The construction is wonderfully natural. Ask yourself: how far is my current state from equilibrium? Not in terms of Euclidean distance, but in terms of what it costs to get there. The Lyapunov functional measures exactly this---the minimal accumulated dissipation needed to reach the safe manifold, plus the height once you arrive.

Think of it like measuring distance on a mountain by the energy you expend hiking, not by the straight-line path a bird would fly. The "effort distance" to the valley bottom is what matters for the dynamics, and that is precisely what the Lyapunov functional captures.

The uniqueness part is remarkable: any other functional with these properties is just a monotone relabeling of this one. There is only one natural notion of "how far from equilibrium" given the dissipation structure.
:::

:::{prf:theorem} [KRNL-Lyapunov] Canonical Lyapunov Functional
:label: mt-krnl-lyapunov

**[Sieve Signature: Canonical Lyapunov]**
- **Requires:** $K_{D_E}^+$ AND $K_{C_\mu}^+$ AND $K_{\mathrm{LS}_\sigma}^+$
- **Produces:** $K_{\mathcal{L}}^+$ (Lyapunov functional exists)
- **Output:** Canonical loss $\mathcal{L}$ = optimal-transport cost to equilibrium

**Statement:** Given a hypostructure $\mathcal{H} = (\mathcal{X}, \Phi, \mathfrak{D}, G)$ with validated interface permits for dissipation ($D_E$ with $C=0$), compactness ($C_\mu$), and local stiffness ($\mathrm{LS}_\sigma$), there exists a canonical Lyapunov functional $\mathcal{L}: \mathcal{X} \to \mathbb{R} \cup \{\infty\}$ with the following properties:

1. **Monotonicity:** Along any trajectory $u(t) = S_t x$, $t \mapsto \mathcal{L}(u(t))$ is nonincreasing and strictly decreasing whenever $u(t) \notin M$.

2. **Stability:** $\mathcal{L}$ attains its minimum precisely on $M$: $\mathcal{L}(x) = \mathcal{L}_{\min}$ iff $x \in M$.

3. **Height Equivalence:** $\mathcal{L}(x) - \mathcal{L}_{\min} \asymp (\Phi(x) - \Phi_{\min})$ on energy sublevels.

4. **Uniqueness:** Any other Lyapunov functional $\Psi$ with these properties satisfies $\Psi = f \circ \mathcal{L}$ for some monotone $f$.

**Explicit Construction (Value Function):**

$$
\mathcal{L}(x) := \inf\left\{\Phi(y) + \mathcal{C}(x \to y) : y \in M\right\}

$$

where the infimal cost is:

$$
\mathcal{C}(x \to y) := \inf\left\{\int_0^T \mathfrak{D}(S_s x) \, ds : S_T x = y, T < \infty\right\}

$$

**Certificate Produced:** $K_{\mathcal{L}}^+ = (\mathcal{L}, M, \Phi_{\min}, \mathcal{C})$

**Literature:** {cite}`AmbrosioGigliSavare08,Villani09`
:::

:::{prf:proof}

*Step 1 (Well-definedness).* Define $\mathcal{L}$ via inf-convolution as above. The functional is well-defined since $\mathfrak{D} \geq 0$ implies $\mathcal{C} \geq 0$. By the **direct method of calculus of variations** ({cite}`Dacorogna08` Chapter 3): $C_\mu$ provides compactness of sublevel sets, and $\Phi + \mathcal{C}$ is lower semicontinuous (as sum of l.s.c. functions). Therefore the infimum is attained at some $y^* \in M$.

*Step 2 (Monotonicity).* For $z = S_t x$, subadditivity gives $\mathcal{C}(x \to y) \leq \mathcal{C}(x \to z) + \mathcal{C}(z \to y)$ for any $y \in M$. Taking infimum over $y$:

$$
\mathcal{L}(x) \leq \int_0^t \mathfrak{D}(S_s x)\, ds + \mathcal{L}(S_t x)

$$

Hence $\mathcal{L}(S_t x) \leq \mathcal{L}(x) - \int_0^t \mathfrak{D}(S_s x)\, ds \leq \mathcal{L}(x)$, with strict inequality when $\mathfrak{D}(S_s x) > 0$ for $s$ in a set of positive measure.

*Step 3 (Minimum on M).* By $\mathrm{LS}_\sigma$, points in $M$ are critical: $\nabla\Phi|_M = 0$, hence the flow is stationary and $\mathfrak{D}|_M = 0$ (since $\mathfrak{D}$ measures instantaneous dissipation rate). Thus $\mathcal{C}(y \to y) = 0$ for $y \in M$, giving $\mathcal{L}(y) = \Phi(y) = \Phi_{\min}$. Conversely, if $x \notin M$, the flow eventually dissipates energy, so $\mathcal{L}(x) > \Phi_{\min}$.

*Step 4 (Height Equivalence).* **Upper bound:** By definition, $\mathcal{L}(x) \leq \Phi(y^*) + \mathcal{C}(x \to y^*)$ for optimal $y^* \in M$. The dissipation inequality from $D_E$ bounds $\mathcal{C}(x \to y^*) \lesssim \Phi(x) - \Phi(y^*) = \Phi(x) - \Phi_{\min}$, so $\mathcal{L}(x) - \Phi_{\min} \lesssim \Phi(x) - \Phi_{\min}$. **Lower bound:** The Łojasiewicz-Simon inequality from $\mathrm{LS}_\sigma$ gives $\|\nabla\Phi(x)\| \geq c|\Phi(x) - \Phi_{\min}|^{1-\theta}$ near $M$. **Integration argument:** Along the gradient flow $\dot{x} = -\nabla\Phi(x)$, we have $\frac{d}{dt}\Phi(x(t)) = -\|\nabla\Phi\|^2 \leq -c^2|\Phi - \Phi_{\min}|^{2(1-\theta)}$. Integrating, $\Phi(x(t)) - \Phi_{\min} \lesssim t^{-1/(2\theta-1)}$ for $\theta < 1$. The arc length satisfies $\int_0^\infty \|\dot{x}\| dt = \int_0^\infty \|\nabla\Phi\| dt < \infty$ by the finite dissipation, giving $\mathrm{dist}(x, M) \lesssim (\Phi(x) - \Phi_{\min})^\theta$. Hence $\mathcal{L}(x) - \Phi_{\min} \gtrsim \mathrm{dist}(x, M)^{1/\theta}$.

*Step 5 (Uniqueness).* Let $\Psi$ satisfy the same properties. Since both $\mathcal{L}$ and $\Psi$ are strictly decreasing along non-equilibrium trajectories and constant on $M$, each level set $\{\mathcal{L} = c\}$ maps to a unique value under $\Psi$. Define $f: \mathrm{Im}(\mathcal{L}) \to \mathbb{R}$ by $f(\mathcal{L}(x)) := \Psi(x)$. This is well-defined (constant on level sets) and monotone (both decrease along flow). Hence $\Psi = f \circ \mathcal{L}$.
:::

#### Action Reconstruction via Jacobi Metric

When gradient consistency ($\mathrm{GC}_\nabla$) is additionally validated, the Lyapunov functional becomes explicitly computable as geodesic distance in a conformally scaled metric.

**Level-up certificate:** `Lyapunov-Jacobi`.

:::{div} feynman-prose
And here is the geometric punchline. If you have gradient consistency---meaning dissipation equals squared velocity along the flow---then the Lyapunov functional is nothing but geodesic distance in a warped geometry.

Picture your state space with the original metric. Now imagine stretching the geometry: where dissipation is high, distances become longer; where dissipation is low, they stay about the same. This is the Jacobi metric---a conformal rescaling by the dissipation itself.

In this warped geometry, the Lyapunov value of any state is simply its distance to the safe manifold. The gradient flow follows geodesics in the Jacobi metric. This is not a metaphor; it is an exact identification. The abstract "cost to reach equilibrium" becomes a concrete "length of shortest path."

This connection to Riemannian geometry is powerful because it imports the entire machinery of differential geometry---geodesics, curvature, cut loci---into the analysis of dynamical systems.
:::

:::{prf:theorem} [KRNL-Jacobi] Action Reconstruction
:label: mt-krnl-jacobi

**[Sieve Signature: Jacobi Metric]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}_\nabla}^+$
- **Produces:** $K_{\text{Jacobi}}^+$ (Jacobi metric reconstruction)
- **Output:** $\mathcal{L}(x) = \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$

**Statement:** Let $\mathcal{H}$ satisfy interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$ on a metric space $(\mathcal{X}, g)$. Then the canonical Lyapunov functional is explicitly the **minimal geodesic action** from $x$ to the safe manifold $M$ with respect to the **Jacobi metric**:

$$
g_{\mathfrak{D}} := \mathfrak{D} \cdot g \quad \text{(conformal scaling by dissipation)}

$$

**Explicit Formula:**

$$
\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: x \to M} \int_0^1 \sqrt{\mathfrak{D}(\gamma(s))} \cdot \|\dot{\gamma}(s)\|_g \, ds

$$

**Simplified Form:**

$$
\mathcal{L}(x) = \Phi_{\min} + \mathrm{dist}_{g_{\mathfrak{D}}}(x, M)

$$

**Certificate Produced:** $K_{\text{Jacobi}}^+ = (g_{\mathfrak{D}}, \mathrm{dist}_{g_{\mathfrak{D}}}, M)$

**Literature:** {cite}`Mielke16,AmbrosioGigliSavare08`
:::

:::{prf:proof}

*Step 1 (Gradient Consistency).* Interface permit $\mathrm{GC}_\nabla$ asserts: along gradient flow $\dot{u} = -\nabla_g \Phi$, we have $\|\dot{u}(t)\|_g^2 = \mathfrak{D}(u(t))$. This identifies dissipation with squared velocity.

*Step 2 (Jacobi Length Formula).* For the Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$, the length element is $ds_{g_{\mathfrak{D}}} = \sqrt{\mathfrak{D}} \cdot ds_g$. Hence for any curve $\gamma$:

$$
\mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \int_0^T \|\dot{\gamma}(t)\|_{g_{\mathfrak{D}}} \, dt = \int_0^T \sqrt{\mathfrak{D}(\gamma(t))} \|\dot{\gamma}(t)\|_g \, dt

$$

*Step 3 (Flow Paths Have Optimal Length).* Along gradient flow $u(t) = S_t x$, by Step 1: $\sqrt{\mathfrak{D}(u(t))} \|\dot{u}(t)\|_g = \sqrt{\mathfrak{D}} \cdot \sqrt{\mathfrak{D}} = \mathfrak{D}(u(t))$. Integrating:

$$
\mathrm{Length}_{g_{\mathfrak{D}}}(u|_{[0,T]}) = \int_0^T \mathfrak{D}(u(t))\, dt = \mathcal{C}(x \to u(T))

$$

Thus Jacobi length equals accumulated cost, and by {prf:ref}`mt-krnl-lyapunov`, gradient flow achieves the infimal cost to $M$.

*Step 4 (Distance Identification).* The infimal Jacobi length from $x$ to $M$ equals $\mathrm{dist}_{g_{\mathfrak{D}}}(x, M)$. By Step 3 and {prf:ref}`mt-krnl-lyapunov`:

$$
\mathrm{dist}_{g_{\mathfrak{D}}}(x, M) = \inf_{\gamma: x \to M} \mathrm{Length}_{g_{\mathfrak{D}}}(\gamma) = \mathcal{C}(x \to M) = \mathcal{L}(x) - \Phi_{\min}

$$

*Step 5 (Lyapunov Verification).* Along flow: $\frac{d}{dt}\mathcal{L}(u(t)) = \frac{d}{dt}\mathrm{dist}_{g_{\mathfrak{D}}}(u(t), M) + 0 = -\|\dot{u}(t)\|_{g_{\mathfrak{D}}} = -\mathfrak{D}(u(t)) \leq 0$, confirming monotone decay.
:::

#### Hamilton-Jacobi PDE Characterization

The Lyapunov functional satisfies a static Hamilton-Jacobi equation, providing a PDE route to explicit computation.

**Level-up certificate:** `Lyapunov-HJ`.

:::{prf:theorem} [KRNL-HamiltonJacobi] Hamilton-Jacobi Characterization
:label: mt-krnl-hamilton-jacobi

**[Sieve Signature: Hamilton-Jacobi PDE]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}_\nabla}^+$
- **Produces:** $K_{\text{HJ}}^+$ (Hamilton-Jacobi PDE characterization)
- **Output:** $\|\nabla_g \mathcal{L}\|_g^2 = \mathfrak{D}$ with $\mathcal{L}|_M = \Phi_{\min}$

**Statement:** Under interface permits $D_E$, $\mathrm{LS}_\sigma$, and $\mathrm{GC}_\nabla$, the Lyapunov functional $\mathcal{L}(x)$ is the unique viscosity solution to the static **Hamilton-Jacobi equation**:

$$
\|\nabla_g \mathcal{L}(x)\|_g^2 = \mathfrak{D}(x)

$$

subject to the boundary condition $\mathcal{L}(x) = \Phi_{\min}$ for $x \in M$.

**Conformal Transformation Identity:**
For conformal scaling $\tilde{g} = \phi \cdot g$ with $\phi > 0$:
- Inverse metric: $\tilde{g}^{-1} = \phi^{-1} g^{-1}$
- Gradient: $\nabla_{\tilde{g}} f = \tilde{g}^{-1}(df, \cdot) = \phi^{-1} \nabla_g f$
- Norm squared: $\|\nabla_{\tilde{g}} f\|_{\tilde{g}}^2 = \tilde{g}(\nabla_{\tilde{g}} f, \nabla_{\tilde{g}} f) = \phi \cdot \phi^{-2} \|\nabla_g f\|_g^2 = \phi^{-1}\|\nabla_g f\|_g^2$

For Jacobi metric $g_{\mathfrak{D}} = \mathfrak{D} \cdot g$, setting $\phi = \mathfrak{D}$:

$$
\|\nabla_{g_{\mathfrak{D}}} f\|_{g_{\mathfrak{D}}}^2 = \mathfrak{D}^{-1} \|\nabla_g f\|_g^2

$$

**Certificate Produced:** $K_{\text{HJ}}^+ = (\mathcal{L}, \nabla_g \mathcal{L}, \mathfrak{D})$

**Literature:** {cite}`Evans10,CrandallLions83`
:::

:::{prf:proof}

*Step 1 (Eikonal for Distance).* In any Riemannian manifold, the distance function $d_M(x) = \mathrm{dist}(x, M)$ satisfies the eikonal equation $\|\nabla d_M\| = 1$ almost everywhere (away from cut locus). For $g_{\mathfrak{D}}$:

$$
\|\nabla_{g_{\mathfrak{D}}} d_M^{g_{\mathfrak{D}}}\|_{g_{\mathfrak{D}}} = 1

$$

where $d_M^{g_{\mathfrak{D}}} = \mathrm{dist}_{g_{\mathfrak{D}}}(\cdot, M)$.

*Step 2 (Apply Conformal Identity).* Using the transformation with $\phi = \mathfrak{D}$:

$$
1 = \|\nabla_{g_{\mathfrak{D}}} d_M^{g_{\mathfrak{D}}}\|_{g_{\mathfrak{D}}}^2 = \mathfrak{D}^{-1} \|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2

$$

Hence $\|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2 = \mathfrak{D}$.

*Step 3 (Identification with $\mathcal{L}$).* By {prf:ref}`mt-krnl-jacobi`: $\mathcal{L}(x) = \Phi_{\min} + d_M^{g_{\mathfrak{D}}}(x)$. Since $\nabla_g \Phi_{\min} = 0$:

$$
\|\nabla_g \mathcal{L}\|_g^2 = \|\nabla_g d_M^{g_{\mathfrak{D}}}\|_g^2 = \mathfrak{D}

$$

with boundary condition $\mathcal{L}|_M = \Phi_{\min}$.

*Step 4 (Viscosity Uniqueness).* The Hamilton-Jacobi equation $\|\nabla_g u\|_g^2 = \mathfrak{D}$ with $u|_M = \Phi_{\min}$ has a unique viscosity solution by standard theory {cite}`CrandallLions83`. Since $\mathcal{L}$ satisfies this equation a.e. and is Lipschitz, it is the viscosity solution.
:::

#### Extended Action Reconstruction (Metric Spaces)

For non-Riemannian settings (Wasserstein spaces, discrete graphs), the reconstruction extends using metric slope.

**Level-up certificate:** `Lyapunov-Metric`.

:::{prf:definition} Metric Slope
:label: def-metric-slope

The **metric slope** of $\Phi$ at $u \in \mathcal{X}$ is:

$$
|\partial \Phi|(u) := \limsup_{v \to u} \frac{(\Phi(u) - \Phi(v))^+}{d(u, v)}

$$

where $(a)^+ := \max(a, 0)$. This generalizes $\|\nabla \Phi\|$ to metric spaces.
:::

:::{prf:definition} Generalized Gradient Consistency ($\mathrm{GC}'_\nabla$)
:label: def-gc-prime

Interface permit $\mathrm{GC}'_\nabla$ (dissipation-slope equality) holds if along any metric gradient flow trajectory:

$$
\mathfrak{D}(u(t)) = |\partial \Phi|^2(u(t))

$$

This extends $\mathrm{GC}_\nabla$ from Riemannian to general metric spaces.
:::

:::{prf:theorem} [KRNL-MetricAction] Extended Action Reconstruction
:label: mt-krnl-metric-action

**[Sieve Signature: Metric Action]**
- **Requires:** $K_{D_E}^+$ AND $K_{\mathrm{LS}_\sigma}^+$ AND $K_{\mathrm{GC}'_\nabla}^+$
- **Produces:** $K_{\mathcal{L}}^{\text{metric}}$ (Lyapunov on metric spaces)
- **Extends:** Riemannian → Wasserstein → Discrete

**Statement:** Under interface permit $\mathrm{GC}'_\nabla$ (dissipation-slope equality), the reconstruction theorems extend to general metric spaces. The Lyapunov functional satisfies:

$$
\mathcal{L}(x) = \Phi_{\min} + \inf_{\gamma: M \to x} \int_0^1 |\partial \Phi|(\gamma(s)) \cdot |\dot{\gamma}|(s) \, ds

$$

where $|\dot{\gamma}|$ denotes the metric derivative and the infimum ranges over all absolutely continuous curves from the safe manifold $M$ to $x$.

**Applications:**

| Setting | State Space | Height $\Phi$ | Dissipation $\mathfrak{D}$ | Metric Slope |
|---------|-------------|---------------|---------------------------|--------------|
| Wasserstein | $(\mathcal{P}_2(\mathbb{R}^n), W_2)$ | Entropy $H(\rho)$ | Fisher info $I(\rho)$ | $\sqrt{I(\rho)}$ |
| Discrete | Prob. on graph $V$ | Rel. entropy $H(\mu\|\pi)$ | Dirichlet form | Discrete Otto |

**Certificate Produced:** $K_{\mathcal{L}}^{\text{metric}} = (\mathcal{L}, |\partial\Phi|, d)$

**Literature:** {cite}`AmbrosioGigliSavare08,Maas11,Mielke11`
:::

:::{prf:proof}

*Step 1 (Metric Derivative Identity).* For absolutely continuous curves $\gamma: [0,1] \to \mathcal{X}$, the metric derivative is $|\dot{\gamma}|(s) := \lim_{h \to 0} d(\gamma(s+h), \gamma(s))/|h|$. By $\mathrm{GC}'_\nabla$, along gradient flow curves: $|\dot{u}|(t)^2 = \mathfrak{D}(u(t)) = |\partial\Phi|^2(u(t))$, hence $|\dot{u}|(t) = |\partial\Phi|(u(t))$.

*Step 2 (Action = Cost).* The action functional $\int_0^1 |\partial\Phi|(\gamma) \cdot |\dot{\gamma}|\, ds$ generalizes Jacobi length. For gradient flow $u(t)$:

$$
\int_0^T |\partial\Phi|(u(t)) \cdot |\dot{u}|(t)\, dt = \int_0^T |\partial\Phi|^2(u(t))\, dt = \int_0^T \mathfrak{D}(u(t))\, dt = \mathcal{C}(x \to u(T))

$$

By the Energy-Dissipation-Identity (EDI) in metric gradient flow theory {cite}`AmbrosioGigliSavare08`, this equals $\Phi(x) - \Phi(u(T))$ for curves of maximal slope.

*Step 3 (Infimum Attained).* The infimum over curves from $M$ to $x$ is attained by the (time-reversed) gradient flow, giving:

$$
\mathcal{L}(x) - \Phi_{\min} = \inf_{\gamma: M \to x} \int_0^1 |\partial\Phi|(\gamma) \cdot |\dot{\gamma}|\, ds

$$

This extends {prf:ref}`mt-krnl-jacobi` to non-smooth settings where $|\partial\Phi|$ replaces $\|\nabla\Phi\|_g$.
:::

---

#### Soft Local Tower Globalization

:::{div} feynman-prose
Now we tackle something that sounds impossible at first: how do you prove global statements about a system by looking only at local data? The answer is tower structures, and the key insight is that "local in scale" can imply "global in behavior."

Imagine a tower of increasingly fine resolutions---like looking at a fluid first at the scale of meters, then centimeters, then millimeters. At each scale you have local data: how much energy dissipates, how states relate across scales, what invariants characterize the configuration. The question is: if your local data is well-behaved at every scale, does the tower converge to something sensible at infinite resolution?

The metatheorem says yes, under checkable conditions. If dissipation is summable with exponential weights, if scale coherence holds so that energy jumps are controlled by local contributions, if local invariants determine local energy---then the global asymptotic structure is completely fixed. No runaway modes, no supercritical growth, no surprises at infinity.

This is the engine behind many deep results: Iwasawa theory in arithmetic, renormalization in physics, multiscale analysis in PDEs. Local-to-global, powered by structural certificates.
:::

For **tower hypostructures** (multiscale systems), local data at each scale determines global asymptotic behavior.

:::{prf:definition} Tower Hypostructure
:label: def-tower-hypostructure

A **tower hypostructure** is a tuple $\mathbb{H} = (X_t, S_{t \to s}, \Phi, \mathfrak{D})$ where:
- $t \in \mathbb{N}$ or $t \in \mathbb{R}_+$ is a **scale index**
- $X_t$ is the state space at level $t$
- $S_{t \to s}: X_t \to X_s$ (for $s < t$) are **scale transition maps** compatible with the semiflow
- $\Phi(t)$ is the height/energy at level $t$
- $\mathfrak{D}(t)$ is the dissipation increment at level $t$
:::

:::{prf:definition} Tower Interface Permits
:label: def-tower-permits

The following **tower-specific interface permits** extend the standard permits to multiscale settings:

| Permit | Name | Question | Certificate |
|--------|------|----------|-------------|
| $C_\mu^{\mathrm{tower}}$ | SliceCompact | Is $\{\Phi(t) \leq B\}$ compact mod symmetries for each scale? | $K_{C_\mu^{\mathrm{tower}}}^{\pm}$ |
| $D_E^{\mathrm{tower}}$ | SubcritDissip | Is $\sum_t w(t)\mathfrak{D}(t) < \infty$ for $w(t) \sim e^{-\alpha t}$? | $K_{D_E^{\mathrm{tower}}}^{\pm}$ |
| $\mathrm{SC}_\lambda^{\mathrm{tower}}$ | ScaleCohere | Is $\Phi(t_2) - \Phi(t_1) = \sum_u L(u) + o(1)$? | $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^{\pm}$ |
| $\mathrm{Rep}_K^{\mathrm{tower}}$ | LocalRecon | Is $\Phi(t)$ determined by local invariants $\{I_\alpha(t)\}$? | $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^{\pm}$ |

**$C_\mu^{\mathrm{tower}}$ (Compactness on slices):** For each bounded interval of scales and each $B > 0$, the sublevel set $\{X_t : \Phi(t) \leq B\}$ is compact or finite modulo symmetries.

**$D_E^{\mathrm{tower}}$ (Subcritical dissipation):** There exists $\alpha > 0$ and weight $w(t) \sim e^{-\alpha t}$ (or $p^{-\alpha t}$ for $p$-adic towers) such that:

$$
\sum_t w(t) \mathfrak{D}(t) < \infty

$$

**$\mathrm{SC}_\lambda^{\mathrm{tower}}$ (Scale coherence):** For any $t_1 < t_2$:

$$
\Phi(t_2) - \Phi(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + o(1)

$$

where each $L(u)$ is a **local contribution** determined by level $u$ data, and $o(1)$ is uniformly bounded.

**$\mathrm{Rep}_K^{\mathrm{tower}}$ (Soft local reconstruction):** For each scale $t$, the energy $\Phi(t)$ is determined (up to bounded, summable error) by **local invariants** $\{I_\alpha(t)\}_{\alpha \in A}$ at scale $t$:

$$
\Phi(t) = F(\{I_\alpha(t)\}_\alpha) + O(1)

$$
:::

::::{prf:theorem} [RESOLVE-Tower] Soft Local Tower Globalization
:label: mt-resolve-tower

**Sieve Signature: Tower Globalization**
- *Weakest Precondition:* $K_{C_\mu^{\mathrm{tower}}}^+$, $K_{D_E^{\mathrm{tower}}}^+$, $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$, $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^+$
- *Produces:* $K_{\mathrm{Global}}^+$ (global asymptotic structure)
- *Invalidated By:* Local-global obstruction

**Setup.** Let $\mathbb{H} = (X_t, S_{t \to s}, \Phi, \mathfrak{D})$ be a tower hypostructure with the following interface permits certified:
1. $C_\mu^{\mathrm{tower}}$: Compactness/finiteness on slices
2. $D_E^{\mathrm{tower}}$: Subcritical dissipation with weight $w(t) \sim e^{-\alpha t}$
3. $\mathrm{SC}_\lambda^{\mathrm{tower}}$: Scale coherence
4. $\mathrm{Rep}_K^{\mathrm{tower}}$: Soft local reconstruction

**Conclusion (Soft Local Tower Globalization):**

**(1)** The tower admits a **globally consistent asymptotic hypostructure**:

$$
X_\infty = \varprojlim X_t

$$

**(2)** The asymptotic behavior of $\Phi$ and the defect structure of $X_\infty$ is **completely determined** by the collection of local reconstruction invariants from $\mathrm{Rep}_K^{\mathrm{tower}}$.

**(3)** No supercritical growth or uncontrolled accumulation can occur: every supercritical mode violates the $D_E^{\mathrm{tower}}$ permit.

**Certificate Produced:** $K_{\mathrm{Global}}^+ = (X_\infty, \Phi_\infty, \{I_\alpha(\infty)\}_\alpha)$

**Usage:** Applies to multiscale analytic towers (fluid dynamics, gauge theories), Iwasawa towers in arithmetic, RG flows (holographic or analytic), complexity hierarchies, spectral sequences/filtrations.
::::

:::{prf:proof}

*Step 1 (Existence of limit).* By $K_{C_\mu^{\mathrm{tower}}}^+$, the spaces $\{X_t\}$ at each level are precompact modulo symmetries. The transition maps $S_{t \to s}$ are compatible by the semiflow property. By $K_{D_E^{\mathrm{tower}}}^+$, the total dissipation is finite:

$$
\sum_t w(t) \mathfrak{D}(t) < \infty

$$

This implies $\mathfrak{D}(t) \to 0$ as $t \to \infty$ (otherwise the weighted sum diverges). Hence dynamics becomes increasingly frozen.

*Step 2 (Asymptotic consistency).* By $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$:

$$
\Phi(t_2) - \Phi(t_1) = \sum_{u=t_1}^{t_2-1} L(u) + O(1)

$$

The $O(1)$ error bound is **uniform** in $t_1, t_2$: by scale coherence, the error comes from boundary terms at the scale interfaces, and there are only $O(1)$ such boundaries per unit interval (the interface permit quantifies this). Taking $t_2 \to \infty$ and using finite dissipation from Step 1:

$$
\Phi(\infty) - \Phi(t_1) = \sum_{u=t_1}^{\infty} L(u) + O(1)

$$

The sum converges absolutely: $|L(u)| \leq C \cdot \mathfrak{D}(u)$ by the scale-coherence permit, and $\sum_u \mathfrak{D}(u) < \infty$ by Step 1. Thus $\Phi(\infty)$ is well-defined.

*Step 3 (Local determination).* By $K_{\mathrm{Rep}_K^{\mathrm{tower}}}^+$:

$$
\Phi(t) = F(\{I_\alpha(t)\}_\alpha) + O(1)

$$

for local invariants $\{I_\alpha(t)\}$. Taking $t \to \infty$: local invariants stabilize (by finite dissipation) to limiting values $I_\alpha(\infty)$. Therefore:

$$
\Phi(\infty) = F(\{I_\alpha(\infty)\}_\alpha) + O(1)

$$

The asymptotic height is completely determined by the asymptotic local data.

*Step 4 (Exclusion of supercritical growth).* Suppose supercritical growth at scale $t_0$: $\Phi(t_0+n) - \Phi(t_0) \gtrsim n^\gamma$ for some $\gamma > 0$. By $K_{\mathrm{SC}_\lambda^{\mathrm{tower}}}^+$, this growth reflects in the local contributions. But then:

$$
\sum_t w(t)\mathfrak{D}(t) \geq \sum_{u=t_0}^\infty e^{-\alpha u} \cdot u^{\gamma-1} = \infty

$$

for any $\gamma > 0$, contradicting $K_{D_E^{\mathrm{tower}}}^+$.

*Step 5 (Defect inheritance).* The limit $X_\infty$ inherits the hypostructure:
- Height functional: $\Phi_\infty(x_\infty) := \lim_{t\to\infty}\Phi(x_t)$
- Dissipation: $\mathfrak{D}_\infty \equiv 0$ (frozen dynamics at infinity)
- Constraints: any violation at $X_\infty$ propagates back to finite levels, contradicting the permits.
:::

---

#### Obstruction Capacity Collapse

:::{div} feynman-prose
Here is a pattern that shows up everywhere, and once you see it, you cannot unsee it.

In many problems, there is a sector of "obstructions"---things that prevent solutions from existing or being unique. The Tate-Shafarevich group in arithmetic. Cohomological obstructions in topology. Singular sets in PDEs. The question is always: how big can this obstruction sector be?

What this metatheorem says is remarkable: under structural conditions on the hypostructure, the obstruction sector must be finite. Not bounded, not controlled---finite. The obstructions cannot accumulate indefinitely because accumulation would violate the subcritical dissipation bound.

This is the "Sha Killer" strategy. You do not attack obstructions one by one. You prove they must be finite by showing that an infinite obstruction sector would violate structural invariants that you have already certified. The counting is automatic once you have the right certificates.

Think of it as a conservation law for obstructions: the total "obstruction weight" is finite because each obstruction costs dissipation, and total dissipation is bounded.
:::

The "Sha Killer" --- proves obstruction sectors (like Tate-Shafarevich groups) are finite. Analogous to Cartan's Theorems A/B for coherent sheaf cohomology.

:::{prf:definition} Obstruction Interface Permits
:label: def-obstruction-permits

The following **obstruction-specific interface permits** extend the standard permits to obstruction sectors $\mathcal{O} \subset \mathcal{X}$:

| Permit | Name | Question | Certificate |
|--------|------|----------|-------------|
| $\mathrm{TB}_\pi^{\mathcal{O}} + \mathrm{LS}_\sigma^{\mathcal{O}}$ | ObsDuality | Is $\langle\cdot,\cdot\rangle_{\mathcal{O}}$ non-degenerate? | $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}\pm}$ |
| $C_\mu^{\mathcal{O}} + \mathrm{Cap}_H^{\mathcal{O}}$ | ObsHeight | Does $H_{\mathcal{O}}$ have compact sublevel sets? | $K_{C+\mathrm{Cap}}^{\mathcal{O}\pm}$ |
| $\mathrm{SC}_\lambda^{\mathcal{O}}$ | ObsSubcrit | Is $\sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty$? | $K_{\mathrm{SC}_\lambda}^{\mathcal{O}\pm}$ |
| $D_E^{\mathcal{O}}$ | ObsDissip | Is $\mathfrak{D}_{\mathcal{O}}$ subcritical? | $K_{D_E}^{\mathcal{O}\pm}$ |

**$\mathrm{TB}_\pi^{\mathcal{O}} + \mathrm{LS}_\sigma^{\mathcal{O}}$ (Duality/Stiffness on obstruction):** The obstruction sector admits a non-degenerate invariant pairing $\langle \cdot, \cdot \rangle_{\mathcal{O}}: \mathcal{O} \times \mathcal{O} \to A$ compatible with the hypostructure flow.

**$C_\mu^{\mathcal{O}} + \mathrm{Cap}_H^{\mathcal{O}}$ (Obstruction height):** There exists a functional $H_{\mathcal{O}}: \mathcal{O} \to \mathbb{R}_{\geq 0}$ such that:
- Sublevel sets $\{x : H_{\mathcal{O}}(x) \leq B\}$ are finite/compact
- $H_{\mathcal{O}}(x) = 0 \Leftrightarrow x$ is trivial obstruction

**$\mathrm{SC}_\lambda^{\mathcal{O}}$ (Subcritical accumulation):** Under any tower or scale decomposition:

$$
\sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty

$$

**$D_E^{\mathcal{O}}$ (Subcritical obstruction dissipation):** The obstruction defect $\mathfrak{D}_{\mathcal{O}}$ grows strictly slower than structural permits allow for infinite accumulation.
::::

::::{prf:theorem} [RESOLVE-Obstruction] Obstruction Capacity Collapse
:label: mt-resolve-obstruction

**Sieve Signature: Obstruction Collapse**
- *Weakest Precondition:* $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$, $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$, $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$, $K_{D_E}^{\mathcal{O}+}$
- *Produces:* $K_{\mathrm{Obs}}^{\mathrm{finite}}$ (obstruction sector is finite)
- *Invalidated By:* Infinite obstruction accumulation

**Setup.** Let $\mathbb{H} = (\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure with distinguished obstruction sector $\mathcal{O} \subset \mathcal{X}$. Assume all obstruction interface permits are certified.

**Conclusion (Obstruction Capacity Collapse):**

**(1)** The obstruction sector $\mathcal{O}$ is **finite-dimensional/finite** in the appropriate sense.

**(2)** No infinite obstruction or runaway obstruction mode can exist.

**(3)** Any nonzero obstruction must appear in strictly controlled, finitely many directions, each of which is structurally detectable.

**Certificate Produced:** $K_{\mathrm{Obs}}^{\mathrm{finite}} = (\mathcal{O}_{\text{tot}}, \dim(\mathcal{O}_{\text{tot}}), H_{\mathcal{O}})$

**Usage:** Applies to Tate-Shafarevich groups, torsors/cohomological obstructions, exceptional energy concentrations in PDEs, forbidden degrees in complexity theory, anomalous configurations in gauge theory.

**Literature:** Cartan's Theorems A/B for coherent cohomology {cite}`CartanSerre53`; finiteness of Tate-Shafarevich {cite}`Kolyvagin90`; {cite}`Rubin00`; obstruction theory {cite}`Steenrod51`.
::::

:::{prf:proof}

*Step 1 (Finiteness at each scale).* Fix a scale $t$. By $K_{C+\mathrm{Cap}}^{\mathcal{O}+}$, the sublevel set $\mathcal{O}_t^{\leq B} := \{x \in \mathcal{O}_t : H_{\mathcal{O}}(x) \leq B\}$ is finite or compact for each $B > 0$.

*Step 2 (Uniform bound on obstruction count).* By $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$, the weighted sum:

$$
S := \sum_t w(t) \sum_{x \in \mathcal{O}_t} H_{\mathcal{O}}(x) < \infty

$$

For each $t$, let $N_t := |\{x \in \mathcal{O}_t : H_{\mathcal{O}}(x) \geq \varepsilon\}|$ count non-trivial obstructions. Then:

$$
S \geq \sum_t w(t) \cdot N_t \cdot \varepsilon

$$

Since $S < \infty$ and $w(t) > 0$, we have $\sum_t w(t) N_t < \infty$, implying $N_t \to 0$ as $t \to \infty$.

*Step 3 (Global finiteness).* The total obstruction $\mathcal{O}_{\text{tot}} := \bigcup_t \mathcal{O}_t$ has contributions from only finitely many scales (Step 2), each finite by Step 1. Hence $\mathcal{O}_{\text{tot}}$ is finite-dimensional.

*Step 4 (No runaway modes).* Suppose a runaway obstruction exists: $(x_n) \subset \mathcal{O}$ with $H_{\mathcal{O}}(x_n) \to \infty$. By $K_{D_E}^{\mathcal{O}+}$:

$$
\mathfrak{D}_{\mathcal{O}}(x_n) \leq C \cdot H_{\mathcal{O}}(x_n)^{1-\delta}

$$

for some $\delta > 0$. But accumulating such obstructions requires $\sum_n H_{\mathcal{O}}(x_n) = \infty$, contradicting $K_{\mathrm{SC}_\lambda}^{\mathcal{O}+}$.

*Step 5 (Structural detectability).* By $K_{\mathrm{TB}+\mathrm{LS}}^{\mathcal{O}+}$, the pairing is non-degenerate: any non-trivial $x \in \mathcal{O}$ has $\langle x, y \rangle_{\mathcal{O}} \neq 0$ for some $y$. Combined with $H_{\mathcal{O}}$, obstructions are localized to specific directions with quantifiable pairing contributions.
:::

---

#### Stiff Pairing / No Null Directions

:::{div} feynman-prose
Now here is a worry you might have: what if there are hidden degrees of freedom where problems can lurk? What if your decomposition into "free sector" and "obstruction sector" misses something?

This metatheorem says: if you have the right structural certificates, there is no place to hide. Every direction in your state space is either free (where the pairing lets you move) or obstruction (where the pairing pins you down). There is no mysterious "null sector" where singularities could accumulate undetected.

The proof is beautifully algebraic. The pairing gives you a map from states to dual states. Non-degeneracy means this map is injective on the parts you care about. Gradient consistency ensures that flat directions of energy match flat directions of the pairing. Put these together and the null sector must be trivial---any direction orthogonal to the free sector must lie in the obstruction sector, and within the obstruction sector the pairing is non-degenerate.

This is the "no ghost sectors" guarantee. Once you have stiffness, you can be sure your accounting is complete.
:::

Guarantees no hidden "ghost sectors" where singularities can hide. All degrees of freedom are accounted for by free components + obstructions.

::::{prf:theorem} [KRNL-StiffPairing] Stiff Pairing / No Null Directions
:label: mt-krnl-stiff-pairing

**Sieve Signature: Stiff Pairing**
- *Weakest Precondition:* $K_{\mathrm{LS}_\sigma}^+$, $K_{\mathrm{TB}_\pi}^+$, $K_{\mathrm{GC}_\nabla}^+$
- *Produces:* $K_{\mathrm{Stiff}}^+$ (no null directions)
- *Invalidated By:* Hidden degeneracy

**Setup.** Let $\mathbb{H} = (\mathcal{X}, \Phi, \mathfrak{D})$ be a hypostructure with bilinear pairing $\langle \cdot, \cdot \rangle : \mathcal{X} \times \mathcal{X} \to F$ such that:
- $\Phi$ is generated by this pairing (via $\mathrm{GC}_\nabla$)
- $\mathrm{LS}_\sigma$ holds (local stiffness)

Let $\mathcal{X} = X_{\mathrm{free}} \oplus X_{\mathrm{obs}} \oplus X_{\mathrm{rest}}$ be a decomposition into free sector, obstruction sector, and possible null sector.

**Hypotheses:**
1. $K_{\mathrm{LS}_\sigma}^+ + K_{\mathrm{TB}_\pi}^+$: $\langle \cdot, \cdot \rangle$ is non-degenerate on $X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$
2. $K_{\mathrm{GC}_\nabla}^+$: Flat directions for $\Phi$ are flat directions for the pairing
3. Any vector orthogonal to $X_{\mathrm{free}}$ lies in $X_{\mathrm{obs}}$

**Conclusion (Stiffness / No Null Directions):**

**(1)** There is **no** $X_{\mathrm{rest}}$: $\mathcal{X} = X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$

**(2)** All degrees of freedom are accounted for by free components + obstructions.

**(3)** No hidden degeneracies or "null modes" exist.

**Certificate Produced:** $K_{\mathrm{Stiff}}^+ = (X_{\mathrm{free}}, X_{\mathrm{obs}}, \langle\cdot,\cdot\rangle)$

**Usage:** Applies to Selmer groups with p-adic height, Hodge-theoretic intersection forms, gauge-theory BRST pairings, PDE energy inner products, complexity gradients.

**Literature:** Selmer groups and p-adic heights {cite}`MazurTate83`; {cite}`Nekovar06`; Hodge theory {cite}`GriffithsHarris78`; BRST cohomology {cite}`HenneauxTeitelboim92`; non-degenerate pairings {cite}`Serre62`.
::::

:::{prf:proof}

*Step 1 (Pairing structure).* The pairing induces $\Psi: \mathcal{X} \to \mathcal{X}^*$, $\Psi(x)(y) := \langle x, y \rangle$. By $K_{\mathrm{LS}_\sigma}^+ + K_{\mathrm{TB}_\pi}^+$, this map is injective on $X_{\mathrm{free}} \oplus X_{\mathrm{obs}}$.

*Step 2 (Radical characterization).* Define $\mathrm{rad}(\langle \cdot, \cdot \rangle) := \{x \in \mathcal{X} : \langle x, y \rangle = 0 \text{ for all } y\}$. Any radical element is orthogonal to $X_{\mathrm{free}}$, hence lies in $X_{\mathrm{obs}}$ by hypothesis.

*Step 3 (Radical within obstruction).* If $x \in \mathrm{rad}$, then $x \in X_{\mathrm{obs}}$ (Step 2). Within $X_{\mathrm{obs}}$, the pairing is non-degenerate, so $\langle x, y \rangle = 0$ for all $y \in X_{\mathrm{obs}}$ implies $x = 0$.

*Step 4 (No null sector).* Suppose $X_{\mathrm{rest}} \neq 0$ with nonzero $z \in X_{\mathrm{rest}}$.
- *Case (a):* $z \in \mathrm{rad} \Rightarrow z = 0$ (Step 3), contradiction.
- *Case (b):* $z \notin \mathrm{rad} \Rightarrow \exists y$ with $\langle z, y \rangle \neq 0$. But $z$ orthogonal to $X_{\mathrm{free}}$ implies $z \in X_{\mathrm{obs}}$, and $X_{\mathrm{obs}} \cap X_{\mathrm{rest}} = \{0\}$, so $z = 0$, contradiction.

*Step 5 (Gradient consistency).* By $K_{\mathrm{GC}_\nabla}^+$, flat directions of $\Phi$ correspond to flat directions of the pairing. Since the pairing has trivial radical, $\Phi$ has no hidden flat directions. Therefore $X_{\mathrm{rest}} = 0$.
:::

---

### Effective (Algorithmic) Proof

:::{div} feynman-prose
Now we come to the final upgrade, and it is the one that computer scientists will love most.

Everything we have done so far is mathematically rigorous, but in principle it could involve steps that are not computable. Non-constructive existence proofs. Undecidable membership tests. Infinite searches. If you want to actually run this machinery on a computer, you need more.

The Effective Certificate says: for this problem type, everything terminates. Certificates live in a finite language. Closure operations complete in finite time. Lock tactics reduce to decidable fragments like SMT or linear arithmetic. The proof backend becomes not just sound but executable.

This is where the Sieve stops being pure mathematics and becomes a theorem prover. You feed in the problem description. The machine runs the Sieve. If it certifies regularity, you have a machine-checkable proof. If it finds an obstruction, you have a machine-verifiable counterexample. The human's job becomes implementing the interface predicates correctly and interpreting the output---the heavy lifting happens automatically.

This is not science fiction. For the right problem types, this is achievable. The challenge is identifying which fragments of which theories admit effective certificate languages.
:::

**Purpose**: Turn the sieve into an **effective conjecture prover**:
- certificates are finite,
- closure terminates,
- lock tactics are decidable or semi-decidable.

The Effective Certificate ($K_{\text{Eff}}$) provides **Rep + computability**.

#### Finite Certificate Language

The Effective Certificate assumes the certificate schemas used by the run live in a **finite or bounded-description language** for the chosen type $T$:
- bounded precision, bounded term size, finite invariant basis, etc.

This ensures:
- promotion closure $\mathrm{Cl}(\Gamma)$ terminates (or is effectively computable),
- replay is decidable.

**Level-up certificate**: `Cert-Finite(T)`.

#### Rep is constructive

Rep is not merely ``exists,'' but comes with:
- a concrete dictionary $D$,
- verifiers for invariants and morphism constraints,
- and an explicit representation of the bad pattern object.

**Level-up certificate**: `Rep-Constructive`.

#### Lock backend tactics are effective

E1--E10 tactics become effective procedures:
- dimension checks, invariant mismatch checks, positivity/integrality constraints, functional equations,
- in a decidable fragment (SMT/linear arithmetic/rewrite systems).

Outcome:
- either an explicit morphism witness,
- or a checkable Hom-emptiness certificate.

**Level-up certificate**: `Lock-Decidable` or `Lock-SemiDecidable`.

#### Decidable/Effective classification (optional)

If you also assume effective profile stratification (Tame Certificate) and effective transport toolkit:
- classification of profiles becomes decidable within the type $T$,
- admissibility and surgery selection can be automated.

**Level-up certificate**: `Classification-Decidable`.

---

### Relationship between Certificate Levels (Summary)

- **Base level**: You can run the sieve; you can classify failures; you can recover when surgery is certified; horizon modes are explicit.
- **$K_{\text{Tame}}$**: You can *systematically classify profiles* and *systematically recover* from singularities for admissible types.
- **$K_{\text{Rigid}}$**: You can derive *canonical Lyapunov structure*, *uniqueness*, and *rates* (including stratified Lyapunov across surgery).
- **$K_{\text{Eff}}$**: You can make large parts of the engine *algorithmic/decidable* (proof backend becomes executable).

---

## Witness Certificates (Auxiliary Bounds)

These are **derived payload certificates** used to interface with analytic modules. They are
*not* standalone gate permits; instead they are produced from existing permits by
metatheorems (or supplied explicitly when derivation is impossible).

:::{prf:definition} Witness Certificates for Uniform Bounds
:label: def-witness-certificates-bounds

| Certificate | Meaning | Payload |
|---|---|---|
| $K_{D_{\max}}^+$ | Bounded algorithmic diameter on the alive core | $(D_{\max},\ \text{support/diameter proof})$ |
| $K_{\rho_{\max}}^+$ | Uniform upper bound on invariant/QSD density | $(\rho_{\max},\ \text{density bound proof})$ |

**Scope:** Each witness is tied to a specified time window or alive core region.

**Typical derivation (Fractal Gas):**
- $K_{D_{\max}}^+$ from $C_\mu^+$ and boundary/overload/starve permits
  ($\mathrm{Bound}_\partial$, $\mathrm{Bound}_B$, $\mathrm{Bound}_\Sigma$).
- $K_{\rho_{\max}}^+$ from $K_{\mathrm{TB}_\rho}^+$ (mixing), $K_{\mathrm{Cap}_H}^+$ (non-collapse),
  and $K_{D_E}^+$ (energy confinement).

If a problem class cannot derive these witnesses from its thin interfaces, the witnesses
must be **supplied explicitly** as extra permits.
:::

:::{prf:remark} Usage Pattern
:label: rem-witness-usage

Witness certificates appear as prerequisites in **bridge-verification** metatheorems
(e.g., Gevrey admissibility). They are never used to *replace* the sieve; they only
certify that an independent analytic proof has its hypotheses satisfied.
:::

### How to Use These Certificates in Theorem Statements

Every strong metatheorem should be written as:
- **Minimal preconditions**: certificates available at base level (works broadly, weaker conclusion).
- **Upgrade preconditions**: additional certificates ($K_{\text{Tame}}$/$K_{\text{Rigid}}$/$K_{\text{Eff}}$), yielding stronger conclusions.

**Example schema**:
- "Soft Lyapunov exists" (base level)
- "Canonical Lyapunov unique up to reparam" ($K_{\text{Rigid}}$)
- "Explicit reconstruction (HJ/Jacobi)" ($K_{\text{Rigid}}$ + $K_{\text{GC}}$ certificates)
- "Algorithmic lock proof (E1--E10 decidable)" ($K_{\text{Eff}}$)

---

(sec-type-system)=
## The Type System

:::{div} feynman-prose
Now, here is something practical. We have all this machinery---Sieve, permits, certificates, metatheorems. But how do you actually use it on a specific problem?

The answer is types. A type is a recipe that tells you: for this class of problems, here is the structure you can assume, here are the moves you are allowed to make, here are the tools available, and here is what happens when you hit the edge of what is provable.

Think of types as "problem templates." Parabolic PDEs are one type. Dispersive equations are another. Gradient flows on metric spaces. Markov processes. Each type comes with its own toolkit, its own natural barriers, its own library of known profiles.

When you face a new problem, you first classify it by type. Then you know immediately what interface predicates to implement, what barriers to expect, what profile library to consult. The type system is what makes the framework usable in practice---it packages all the structural knowledge about a class of problems into a reusable template.
:::

:::{prf:definition} Problem type
:label: def-problem-type

A **type** $T$ is a class of dynamical systems sharing:
1. Standard structure (local well-posedness, energy inequality form)
2. Admissible equivalence moves
3. Applicable toolkit factories
4. Expected horizon outcomes when Rep/definability fails

:::

---

### Type Catalog

:::{div} feynman-prose
Let me show you the zoo. Each type below is a complete "problem template" that packages everything you need: the structure of the evolution, the form of energy and dissipation, what moves are allowed, what barriers typically arise, and what profiles are known.

You will notice a pattern: each type has evolved its own toolkit because each type has its own characteristic difficulties. Parabolic equations develop singularities where curvature concentrates---so the toolkit includes surgery. Dispersive equations scatter to infinity unless concentration occurs---so the toolkit includes concentration-compactness. Gradient flows on metric spaces may lack smoothness---so the toolkit uses metric slopes instead of gradients.

This is not mere catalog-keeping. Each type represents decades of hard-won insight about a class of problems, distilled into a format the Sieve can use. When you instantiate the Sieve for a specific equation, you are leveraging this accumulated knowledge.
:::

:::{prf:definition} Type $T_{\text{parabolic}}$
:label: def-type-parabolic

**Parabolic PDE / Geometric Flows**

**Structure**:
- Evolution: $\partial_t u = \Delta u + F(u, \nabla u)$ or geometric analog
- Energy: $\Phi(u) = \int |\nabla u|^2 + V(u)$
- Dissipation: $\mathfrak{D}(u) = \int |\partial_t u|^2$ or $\int |\nabla^2 u|^2$

**Equivalence moves**: Symmetry quotient, metric deformation, Ricci/mean curvature surgery

**Standard barriers**: Saturation, Type II (via monotonicity formulas), Capacity (epsilon-regularity)

**Profile library**: Solitons, shrinkers, translators, ancient solutions

:::

:::{prf:definition} Type $T_{\text{dispersive}}$
:label: def-type-dispersive

**Dispersive PDE / Scattering**

**Structure**:
- Evolution: $i\partial_t u = \Delta u + |u|^{p-1}u$ or wave equation
- Energy: $\Phi(u) = \int |\nabla u|^2 + |u|^{p+1}$
- Dispersion: Strichartz estimates

**Equivalence moves**: Galilean/Lorentz symmetry, concentration-compactness

**Standard barriers**: Scattering (Benign), Type II (Kenig-Merle), Capacity

**Profile library**: Ground states, traveling waves, blow-up profiles

:::

:::{prf:definition} Type $T_{\text{metricGF}}$
:label: def-type-metricgf

**Metric Gradient Flows**

**Structure**:
- Evolution: Curves of maximal slope in metric spaces
- Energy: Lower semicontinuous functional
- Dissipation: Metric derivative squared

**Equivalence moves**: Metric equivalence, Wasserstein transport

**Standard barriers**: EVI (Evolution Variational Inequality), Geodesic convexity

:::

:::{prf:definition} Type $T_{\text{Markov}}$
:label: def-type-markov

**Diffusions / Markov Semigroups**

**Structure**:
- Evolution: $\partial_t \mu = L^* \mu$ (Fokker-Planck)
- Energy: Free energy / entropy
- Dissipation: Fisher information

**Equivalence moves**: Time-reversal, detailed balance conjugacy

**Standard barriers**: Log-Sobolev, Poincare, Mixing times

:::

:::{prf:definition} Type $T_{\text{algorithmic}}$
:label: def-type-algorithmic

**Computational / Iterative Systems**

**Structure**:
- Evolution: $x_{n+1} = F(x_n)$ or continuous-time analog
- Energy: Loss function / Lyapunov
- Dissipation: Per-step progress

**Equivalence moves**: Conjugacy, preconditioning

**Standard barriers**: Convergence rate, Complexity bounds

:::

:::{prf:definition} Representable Set (Algorithmic States)
:label: def-representable-set-algorithmic

For any algorithm $\mathcal{A}$ with configuration $q_t$ at time $t$, the **representable set** is:

$$
\mathcal{R}(q_t) := \{x \in \{0,1\}^n : x \text{ is explicitly encoded or computable from } q_t \text{ in } O(1)\}

$$

The **capacity** of state $q_t$ is:

$$
\mathrm{Cap}(q_t) := |\mathcal{R}(q_t)|

$$

**Polynomial capacity bound:** An algorithm $\mathcal{A}$ satisfies $K_{\mathrm{Cap}}^{\mathrm{poly}}$ if:

$$
\forall t, \forall q_t: \mathrm{Cap}(q_t) \leq \mathrm{poly}(n)

$$

This holds for all polynomial-time algorithms by definition (tape length bound).
:::

:::{prf:definition} Representable-Law Semantics
:label: def-representable-law

For configuration $q_t$ of any algorithm $\mathcal{A}$, the **representable induced law** is:

$$
\mu_{q_t} := \mathrm{Unif}(\mathcal{R}(q_t))

$$

**Certificate:** $K_{\mu \leftarrow \mathcal{R}}^+ := (\mathrm{supp}(\mu_{q_t}) \subseteq \mathcal{R}(q_t))$

**Semantic content:** "State laws are supported on the representable set." This makes "in support => representable now" true by construction.

**Justification:** This replaces the "induced distribution over future outputs" semantics with a semantics tied to the current state's explicit content. The key insight is that what an algorithm "knows" at time $t$ is precisely what it can compute from its current configuration in $O(1)$ time.
:::
