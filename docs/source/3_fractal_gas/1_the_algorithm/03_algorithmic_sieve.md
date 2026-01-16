# The Algorithmic Sieve: Rigorous Parameter Constraints for Convergence

## TLDR

**Three-Layer Constraint Synthesis**: This chapter derives parameter bounds for Fractal Gas convergence by combining three complementary analysis layers: appendix proofs (QSD structure, error bounds, contraction rates), algorithm bounds (fitness ranges, cloning scores, Doeblin floors), and hypostructure certificates (17-node systematic verification). The result is a clear separation between *rigorous constraints* and *heuristic tuning rules*.

**Rigorous vs. Heuristic Constraints**: The sieve yields provable constraints (companion minorization, LSI noise floor, and the acoustic limit under Appendix 15 assumptions) and separates them from heuristic tuning rules (phase-balance targets and timestep stability). This keeps formal guarantees clean while still providing practical guidance.

**Quantitative Convergence Rate**: The total convergence rate follows the bottleneck principle
$\kappa_{\text{total}} = \min(\kappa_x,\kappa_v,\kappa_W,\kappa_b)\,(1-\epsilon_{\text{coupling}})$, with an irreducible $O(1/\sqrt{N})$ error floor from finite population effects. Defaults should be *checked* against the bounds rather than assumed to satisfy them.

**Theory-First Parameter Selection**: The chapter provides a conservative parameter recipe and labels which steps are rigorous and which are heuristic. Bounds are classified by rigor: mean-field/QSD structure is asymptotic, Wasserstein contraction and error bounds are provable under stated assumptions, and hypocoercive rates remain conjectured.

## Introduction

:::{div} feynman-prose
Now, here is something every engineer learns the hard way: a beautiful algorithm that diverges is worthless. You can have the most elegant mathematical framework in the world, but if your particles fly off to infinity, or your cloning rates explode, or your population collapses to a single point—well, you have nothing.

So the question is: *which parameter settings actually work?* Not in a hand-wavy "these seem reasonable" sense, but in a rigorous "I can prove this converges" sense.

The trouble is, there are three different sources telling us about constraints, and they do not quite agree. The appendix analysis gives us rate formulas, but some with incomplete derivations. The algorithm bounds give us exact local conditions, but those are necessary, not sufficient. And the hypostructure framework—that is where the systematic verification happens.

What we are going to do in this chapter is show you how to combine all three sources into a single, coherent system of constraints. Think of it as building a sieve: each layer catches different problems, and only parameters that pass through *all* the layers are guaranteed to work.
:::

This chapter derives **parameter constraints** for the Fractal Gas algorithm by synthesizing results from three analysis layers. Where {doc}`01_algorithm_intuition` explains what the algorithm does and {doc}`02_fractal_gas_latent` provides the formal machinery, this chapter answers the practical question: given a problem, what parameter values are *provably sufficient* for convergence, and which are *heuristic tuning rules*?

The approach is systematic rather than ad hoc. We separate the constraints that are explicitly derived from appendices or algorithm definitions from rules-of-thumb used for tuning. Only the former enter the formal guarantees.

:::{admonition} Philosophy: Theory Takes Precedence
:class: important

The theoretical constraints derived here define what parameter values are **permissible**. The code implementation must satisfy these bounds—not the reverse. If defaults violate theory, defaults must change.
:::

### Proof Status Classification

Before proceeding, we classify the theoretical results by rigor:

**Rigorously Proven (under stated assumptions)**:
1. **Quantitative error bounds** (Appendix 13): Observable error $O(1/\sqrt{N})$ with explicit constants.
2. **Propagation of chaos** (Appendix 09): Mean-field convergence rate.
3. **Foster-Lyapunov structure** (Appendix 06): Drift conditions and bottleneck-rate formula.
4. **Wasserstein contraction** (Appendix 04): N-uniform $\kappa_W > 0$ with explicit lower bounds.

**Asymptotic / Model-Based (mean-field scaling)**:
1. **QSD structure** (Appendix 07): $\rho_{\text{clone}}(z) \propto R(z)^{\gamma_{\text{eff}}}$ in the mean-field/linear-regime approximation.

**Conjectured / Heuristic**:
1. **Hypocoercive rate** (Appendix 10): $\Lambda \approx \gamma \rho_{\text{LSI}}/M^2$ — derivation incomplete.
2. **Bounded density ratio** (Appendix 11): $M^2$ bound deferred.

(sec-three-layer-hierarchy)=
## Three-Layer Bound Hierarchy

:::{div} feynman-prose
Let me explain what we are up against here. Imagine you are building a bridge. The materials engineer says "the steel beams must hold X tons." The structural engineer says "the span requires Y inches of clearance." The safety inspector says "you need Z factor of safety." Each expert is correct in their own domain, but no single expert tells you the complete story.

Our situation is similar. We have three "experts":

1. **The Appendix Analysis** — This is like the materials engineer. It tells us how fast things converge in the ideal continuous limit, with proofs we can trust. But the constants are sometimes "order one," which is maddening when you need actual numbers.

2. **The Algorithm Bounds** — This is like the structural engineer. It gives us exact formulas for fitness bounds, cloning scores, and kernel weights. These are *necessary* conditions—if you violate them, you definitely fail. But satisfying them does not guarantee success.

3. **The Hypostructure Certificates** — This is the safety inspector who checks that everything works together. The 17 gates form a systematic verification that all the pieces actually fit.

The beautiful thing is how these three layers complement each other. The appendix gives us the forms of the bounds. The algorithm gives us the exact constants. And the hypostructure gives us the logical machinery to combine them.
:::

### Layer 1: Appendix Analysis (Proven + Asymptotic)

The appendices provide rigorously proven convergence results and explicit mean-field scaling laws. We use the former as hard constraints and the latter as asymptotic guidance.

:::{prf:theorem} QSD Structure (from Appendix 07)
:label: thm-alg-sieve-qsd-structure

In the mean-field limit, and in the linear-response regime of the fitness sigmoid (Appendix 07), the cloning equilibrium density has the form:

$$
\rho_{\text{clone}}(z) = \frac{1}{Z} R(z)^{\gamma_{\text{eff}}}
$$

where the **concentration exponent** is:

$$
\gamma_{\text{eff}} = \frac{\alpha D}{\beta}
$$

This corresponds to an effective **cloning temperature**:

$$
T_{\text{clone}} = \frac{\beta}{\alpha D}
$$

*Proof*: See Appendix 07, Theorem 2.4 (derived from iso-fitness principle and mean-field limit of algorithmic distance). $\square$
:::

:::{div} feynman-prose
Now let me tell you what this QSD structure is really saying. The quasi-stationary distribution (QSD) is what your swarm of particles settles into after a long time—not the *absolute* equilibrium, but the equilibrium *given that particles have not died*.

The formula $\rho_{\text{clone}}(z) = R(z)^{\gamma_{\text{eff}}}/Z$ looks complicated, but here is the physical picture: regions of high reward $R(z)$ get exponentially more particles. The exponent $\gamma_{\text{eff}} = \alpha D/\beta$ controls how "greedy" the concentration is.

- If $\gamma_{\text{eff}}$ is large, particles pile up aggressively at the best spots.
- If $\gamma_{\text{eff}}$ is small, the distribution stays more uniform.

The beautiful thing is that this exponent directly connects the algorithm parameters ($\alpha$, $\beta$) to the concentration behavior. This is not a hand-wavy correspondence—it is an exact relationship in the mean-field limit.
:::

:::{prf:theorem} Quantitative Error Bounds (from Appendix 13)
:label: thm-alg-sieve-error-bounds

For any Lipschitz observable $\phi$ with constant $L_\phi$, the mean-field approximation error satisfies:

$$
\left| \mathbb{E}_{\nu_N^{\text{QSD}}} \left[ \frac{1}{N} \sum_{i=1}^N \phi(z_i) \right] - \int_\Omega \phi(z) \rho_0(z) \, dz \right| \leq \frac{C_{\text{obs}} \cdot L_\phi}{\sqrt{N}}
$$

where $C_{\text{obs}} = \sqrt{C_{\text{var}} + C_{\text{dep}} \cdot C_{\text{int}}}$ with:
- $C_{\text{var}}$: variance of $\rho_0$
- $C_{\text{dep}}$: dependence constant
- $C_{\text{int}} = \lambda \cdot L_{\log \rho_0} \cdot \text{diam}(\Omega)$: interaction complexity

The **LSI constant** is:

$$
\lambda_{\text{LSI}} = \frac{\gamma \cdot \kappa_{\text{conf}} \cdot \kappa_W \cdot \delta^2}{C_0}
$$

*Proof*: See Appendix 13, Lemmas 3.1-3.4 (Fournier-Guillin bound combined with N-uniform LSI). $\square$
:::

:::{div} feynman-prose
This error bound is crucial, so let me make sure you understand what it is saying. When you run the algorithm with $N$ particles, you are approximating a continuous distribution with a finite sample. The question is: how good is this approximation?

The answer is $O(1/\sqrt{N})$—the same scaling you get from the central limit theorem. If you want ten times better accuracy, you need one hundred times more particles. This is the tyranny of Monte Carlo methods.

But here is the key insight: this is an *error floor*, not a convergence rate. No matter how long you run the algorithm, you cannot do better than $C/\sqrt{N}$. The finite population creates irreducible statistical noise. To reduce this noise, you must increase $N$—there is no free lunch.

The other piece—the LSI constant $\lambda_{\text{LSI}}$—tells you how fast you *approach* this floor. A larger LSI constant means faster convergence, which depends on the friction $\gamma$, the confinement $\kappa_{\text{conf}}$, and the geometry of the interaction $\kappa_W$.
:::

:::{prf:theorem} Wasserstein Contraction (from Appendix 04)
:label: thm-alg-sieve-wasserstein-contraction

The cloning operator induces N-uniform Wasserstein-2 contraction:

$$
W_2^2(\Psi_{\text{clone}}(\mu_1), \Psi_{\text{clone}}(\mu_2)) \leq (1 - \kappa_W) W_2^2(\mu_1, \mu_2) + C_W
$$

where $\kappa_W > 0$ is **independent of $N$**. A conservative explicit form (Appendix 04) is

$$
\kappa_W = c_{\text{dom}} \cdot p_u(\varepsilon)\, c_{\text{geom}}\, c_{\text{sep}}(\varepsilon),
$$

with $p_u(\varepsilon)>0$ (cloning pressure), $c_{\text{geom}}>0$ (geometric constant), and
$c_{\text{sep}}(\varepsilon)>0$ (cluster-separation constant), all N-uniform under the stated assumptions.

*Proof*: See Appendix 04, Theorem 6.1 (cluster-level analysis avoiding $q_{\min} \sim 1/N!$ obstruction). $\square$
:::

:::{div} feynman-prose
The Wasserstein contraction is perhaps the most subtle result here, so let me explain why it matters.

Wasserstein distance measures how "far apart" two probability distributions are in terms of *transport cost*—how much work it takes to move the mass from one distribution to the other. If the cloning operator contracts Wasserstein distance, it means that after cloning, any two swarm configurations become more similar.

The key phrase is "N-uniform": the contraction rate $\kappa_W$ does not depend on $N$. This was not obvious! A naive analysis might suggest that with more particles, the worst-case contraction gets worse (because there are more ways for particles to be arranged pathologically). The proof in Appendix 04 shows that by analyzing clusters rather than individual particles, we avoid this trap.

The practical consequence: adding more particles helps accuracy (reducing the $1/\sqrt{N}$ error floor) without hurting the convergence speed.
:::

:::{admonition} Conjectured Result: Hypocoercive Rate
:class: warning

The following result appears in Appendix 10 but has incomplete derivation:

$$
\Lambda_{\text{hypo}} \approx \frac{\gamma \cdot \rho_{\text{LSI}}}{M^2}
$$

where $M = \sup \|\nabla_x^2 V_{\text{eff}}\|$ is the Hessian bound. **Use as heuristic guidance only.**
:::

### Layer 2: Algorithm Bounds (Exact Formulas)

:::{div} feynman-prose
Now we come to something more concrete. The algorithm document gives us actual numbers—no "order one" hand-waving. These are the bounds we can compute from the code.

The key insight is that the algorithm has many knobs: reward exponent $\alpha$, diversity exponent $\beta$, positivity floor $\eta$, logistic bound $A$, and so on. Each of these parameters has explicit upper and lower bounds that come from inspecting what the code actually does.

These bounds are *necessary*—if you violate them, something definitely breaks. But they are not *sufficient*—satisfying all of them does not magically guarantee convergence. That is why we need the third layer to tie everything together.
:::

The algorithm document provides exact, computable bounds.

:::{prf:definition} Fitness Bounds
:label: def-alg-sieve-fitness-bounds

The fitness potential $V_{\text{fit}}$ is bounded by:

$$
V_{\min} := \eta^{\alpha+\beta} \leq V_{\text{fit}} \leq (A+\eta)^{\alpha+\beta} =: V_{\max}
$$

**Parameter definitions**:
- $\eta > 0$: positivity floor (prevents $V_{\text{fit}} = 0$)
- $A > 0$: logistic bound on reward signal
- $\alpha, \beta \geq 0$: reward/diversity exponents

**Default values** ($\alpha = \beta = 1$, $\eta = 0.1$, $A = 2.0$):

$$
V_{\min} = 0.01, \quad V_{\max} = 4.41
$$
:::

:::{prf:definition} Cloning Score Bound
:label: def-alg-sieve-cloning-score

The cloning score satisfies:

$$
|S_i| \leq S_{\max} := \frac{V_{\max} - V_{\min}}{V_{\min} + \varepsilon_{\text{clone}}}
$$

**Default value** ($\varepsilon_{\text{clone}} = 0.01$):

$$
S_{\max} = \frac{4.41 - 0.01}{0.01 + 0.01} = 220
$$

:::{div} feynman-prose
Here is something important: that $S_{\max} = 220$ looks scary, but it is a *worst case*. In practice, the cloning score is much smaller because not every particle is at the extremes.

The distinction between worst-case bounds and expected behavior matters for stability analysis. If you design your friction coefficient assuming every particle achieves the maximum cloning score, you will end up with $\gamma$ much larger than necessary. That makes the algorithm overdamped—it converges, but slowly.

The smarter approach is to use expected values when analyzing typical behavior and worst-case bounds only for hard safety guarantees.
:::

:::

:::{prf:proposition} Doeblin Floor for Softmax Kernel
:label: prop-alg-sieve-doeblin-softmax

For the softmax companion kernel with unnormalized weights

$$
w_{ij} = \exp\!\left(-\frac{d_{ij}^2}{2\varepsilon^2}\right), \quad j\neq i,
$$

the companion selection probability

$$
P_i(j) = \frac{w_{ij}}{\sum_{k \neq i} w_{ik}}
$$

satisfies the minorization bound

$$
P_i(j) \geq \frac{m_\varepsilon}{n_{\mathrm{alive}} - 1},
$$

where $m_\varepsilon = \exp(-D_{\text{alg}}^2/(2\varepsilon^2))$ and $D_{\text{alg}} = \sqrt{D_z^2 + \lambda_{\text{alg}} D_v^2}$ is the algorithmic diameter on the alive core.

*Proof*: Each weight is at least $m_\varepsilon$ and at most $1$, so $\sum_{k\neq i} w_{ik} \le n_{\mathrm{alive}} - 1$. Hence $P_i(j)\ge m_\varepsilon/(n_{\mathrm{alive}}-1)$. $\square$
:::

:::{div} feynman-prose
The Doeblin condition is really about preventing isolation. In a population of walkers, you need some mechanism for communication—particles that are far apart must still be able to exchange information through cloning.

The softmax kernel makes this work: even the most distant particle has *some* probability of being selected as a companion. The probability is exponentially small (that is the $\exp(-D^2/2\varepsilon^2)$ factor), but it is not zero. And that tiny probability is what prevents the swarm from fragmenting into isolated clusters that never talk to each other.

This is the mathematical essence of ergodicity: from any state, you can eventually reach any other state. Without it, the Markov chain could get stuck in local pockets forever.
:::

### Layer 3: Hypostructure Certificates (Systematic Verification)

:::{div} feynman-prose
Now comes the clever part. We have all these individual constraints from the appendix and algorithm analyses. But how do we know they are consistent? How do we know that satisfying constraint A does not violate constraint B?

The hypostructure framework is essentially a logic engine. Each "gate" represents a constraint that must be satisfied. When a gate is satisfied, it produces a "certificate"—a formal proof object that says "yes, this constraint is met." The certificates can then trigger other gates in a chain reaction.

Think of it like dominoes. You verify gate 1, which produces certificate $K_1^+$. That certificate enables gate 5 to fire, producing certificate $K_5^+$. And so on, until either all gates are satisfied or you hit a failure.

The key insight is that this propagation *terminates*. After at most 17 iterations, you have either proved everything works or identified exactly which constraint is violated. There is no infinite regress, no circular dependencies—just a clean logical chain.
:::

The hypostructure framework provides systematic verification via 17 gate nodes.

:::{prf:definition} Certificate Propagation
:label: def-alg-sieve-certificate-prop

The **certificate closure** operation computes all logical consequences:

```python
def closure(Gamma: set[Certificate]) -> set[Certificate]:
    """Compute certificate closure via promotion rules."""
    changed = True
    while changed:
        changed = False
        for gate in GATES_1_TO_17:
            if gate.can_fire(Gamma) and gate.certificate not in Gamma:
                Gamma.add(gate.fire(Gamma))
                changed = True
    return Gamma
```

**Termination**: Closure terminates in at most 17 iterations (each gate fires at most once by monotonicity).
:::

(sec-master-constraints)=
## Master Constraint System

:::{div} feynman-prose
Here is where the rubber meets the road. We organize constraints into five themes, each tied to a distinct failure mode. Some are rigorous (B, C, E); others are heuristic tuning rules (A, D):

1. **Phase Control** — Are we in the liquid phase where optimization actually happens, or have we frozen solid or evaporated into a diffuse gas?

2. **Acoustic Stability** — Is the friction strong enough to smooth out the "shocks" from cloning events, or does the system develop instabilities?

3. **Doeblin Minorization** — Can distant particles still talk to each other, or has the swarm fragmented into isolated islands?

4. **Timestep (CFL)** — Is the discretization fine enough to capture the dynamics accurately, or are we losing information at every step?

5. **Noise Injection** — Is there enough jitter to explore the space, or too much so that we lose precision?

Constraints **B**, **C**, and **E** are derived in the appendices (under stated assumptions). Constraints **A** and **D**
are **heuristic tuning rules**: they guide practical stability but are not part of the formal convergence proof. The art
is finding parameter settings that satisfy the rigorous bounds while using the heuristic ones to improve performance.
:::

We now derive the five constraint themes and label which ones are rigorous.

### A. Phase Control (Thermal Balance)

:::{div} feynman-prose
Let me tell you about phase transitions, because that is really what is happening here.

In physics, when you cool water, it goes from gas to liquid to solid. Each phase has qualitatively different behavior. Gas fills its container uniformly. Liquid flows but has definite volume. Solid holds its shape.

The Fractal Gas has analogous phases:

- **Gas phase** ($\Gamma \gg 1$): The kinetic temperature dominates. Particles bounce around exploring the whole space uniformly. Great for coverage, terrible for optimization—nothing ever concentrates anywhere.

- **Crystal phase** ($\Gamma \ll 1$): The cloning temperature dominates. Particles lock onto local optima and stay there. The swarm finds *a* solution quickly but cannot explore alternatives.

- **Liquid phase** ($\Gamma \approx 1$): The sweet spot. Enough kinetic energy to explore, enough cloning pressure to concentrate on good regions. This is where optimization actually happens.

The phase control parameter $\Gamma$ is the ratio of kinetic to cloning temperature. Keeping it near 1 keeps the system in the liquid phase.
:::

:::{prf:definition} Phase Control Parameter
:label: def-alg-sieve-phase-parameter

The **phase control parameter** balances kinetic and cloning temperatures:

$$
\Gamma := \frac{T_{\text{kin}}}{T_{\text{clone}}} = \frac{\sigma_v^2}{2\gamma} \cdot \frac{\alpha D}{\beta}
$$

where:
- $T_{\text{kin}} = \sigma_v^2/(2\gamma)$: kinetic temperature (fluctuation-dissipation)
- $T_{\text{clone}} = \beta/(\alpha D)$: cloning temperature (from {prf:ref}`thm-alg-sieve-qsd-structure`)
- $D$: effective dimension (phase space dimension if $\lambda_{\text{alg}} > 0$, else spatial dimension)
:::

:::{admonition} Heuristic: Phase Balance
:class: feynman-added tip

A practical target range is:

$$
\boxed{\Gamma \in [0.5, 2.0]}
$$

**Derivation** (empirical with theoretical motivation):

The phase parameter $\Gamma$ controls the balance between exploration (kinetic diffusion) and exploitation (cloning concentration). From numerical experiments and mean-field analysis:

- **$\Gamma < 0.5$**: Cloning dominates kinetics; swarm collapses to local optima (crystal phase)
- **$\Gamma \in [0.5, 2.0]$**: Balanced regime; effective optimization with maintained diversity (liquid phase)
- **$\Gamma > 2.0$**: Kinetics dominates cloning; swarm diffuses uniformly (gas phase)

The boundaries $[0.5, 2.0]$ are **empirical** operating ranges. Mean-field analysis suggests critical points near $\Gamma \approx 0.3$ and $\Gamma \approx 1.7$; the interval $[0.5, 2.0]$ provides practical safety margins around those transitions.

**Physical interpretation**:
- $\Gamma \gg 2$: **Gas phase** — high entropy, uniform coverage, slow optimization
- $\Gamma \approx 1$: **Liquid phase** — balanced exploitation/exploration, optimal
- $\Gamma \ll 0.5$: **Crystal phase** — locked in local minima, poor exploration

:::

### B. Acoustic Stability (Friction Bound)

:::{div} feynman-prose
Here is something subtle that trips people up. Cloning is not a smooth process—it is a *jump*. One moment you have particle $i$ at position $z_i$; the next moment you have two particles at slightly different positions. That is a discontinuity.

Now, in fluid dynamics, when you have a shock wave, you need viscosity to smooth it out. Otherwise the discontinuity steepens and the numerics blow up. The same thing happens here: cloning creates "shocks" in the particle distribution, and friction plays the role of viscosity.

If the friction is too low relative to the cloning rate, these shocks accumulate faster than they dissipate. The distribution develops instabilities—oscillations, fragmentation, eventual divergence. That is the "acoustic" instability (named by analogy with sound waves in compressible fluids).

The Appendix 15 acoustic limit gives an explicit lower bound on $\gamma$ in terms of the Hessian bound, the cloning rate, and the Dobrushin constant. It says: the damping rate must exceed the shock production rate.
:::

:::{prf:proposition} Friction Lower Bound
:label: prop-alg-sieve-friction-bound

The friction coefficient must satisfy the **explicit acoustic limit** (Appendix 15):

$$
\boxed{\gamma > \gamma_* := \frac{c_2 M^2}{c_1 \lambda} + \frac{C_{\text{Dob}}\,\nu_{\text{clone}}}{c_1 \kappa_W}}
$$

where:
- $M = \sup_x \|\nabla^2 U(x)\|$ is the Hessian bound of the effective potential,
- $\lambda>0$ is the position–velocity coupling parameter in the hypocoercive carré du champ,
- $C_{\text{Dob}}$ is the Dobrushin constant controlling cloning perturbations,
- $\nu_{\text{clone}}$ is the cloning rate (expected clones per unit time),
- $\kappa_W$ is the Wasserstein contraction rate, and
- $c_1, c_2$ are the hypocoercivity constants from Appendix 15.

This is the rigorous friction lower bound used in the LSI/acoustic-limit analysis; numerical substitutes should be marked as heuristics.

$\square$
:::

### C. Doeblin Minorization (Kernel Scale)

:::{div} feynman-prose
The Doeblin condition is one of those beautiful results from probability theory that has a simple physical meaning: from anywhere, you can get anywhere else.

More precisely, it says that no matter how far apart two particles are, there is always *some* minimum probability that they can become companions. This prevents the population from fragmenting into isolated clusters that never exchange genetic material.

The kernel scale $\varepsilon$ controls this. If $\varepsilon$ is too small, distant particles have negligible companion probability. The softmax kernel gives weight $\exp(-d^2/2\varepsilon^2)$ to a particle at distance $d$. When $d = D_{\text{alg}}$ (the maximum possible distance), this weight is $\exp(-D_{\text{alg}}^2/2\varepsilon^2)$. For this to be non-negligible, we need $\varepsilon$ large enough.

The formula $\varepsilon \geq D_{\text{alg}}/\sqrt{2\ln((n_{\mathrm{alive}}-1)/p_{\min})}$ comes from inverting this requirement: given a target minimum probability $p_{\min}$, how large must $\varepsilon$ be?
:::

:::{prf:proposition} Kernel Scale Bound
:label: prop-alg-sieve-kernel-bound

The companion kernel scale must satisfy:

$$
\boxed{\varepsilon \geq \frac{D_{\text{alg}}}{\sqrt{2 \ln((n_{\mathrm{alive}}-1)/p_{\min,\text{target}})}}}
$$

**Derivation** (from softmax Doeblin condition):

From {prf:ref}`prop-alg-sieve-doeblin-softmax`, the minimum companion probability is:

$$
P_{\min} \geq \frac{\exp(-D_{\text{alg}}^2/(2\varepsilon^2))}{n_{\mathrm{alive}} - 1}
$$

For the Doeblin condition to yield meaningful mixing, we need $P_{\min} \geq p_{\min,\text{target}}$:

$$
\frac{\exp(-D_{\text{alg}}^2/(2\varepsilon^2))}{n_{\mathrm{alive}} - 1} \geq p_{\min,\text{target}}
$$

Solving for $\varepsilon$:

$$
-\frac{D_{\text{alg}}^2}{2\varepsilon^2} \geq \ln(p_{\min,\text{target}}(N-1))
$$

$$
 \varepsilon^2 \geq \frac{D_{\text{alg}}^2}{2 \ln((n_{\mathrm{alive}}-1)/p_{\min,\text{target}})}.
$$

This is a **sufficient** bound derived from the minorization floor. Defaults should be checked against the chosen
$p_{\min,\text{target}}$ rather than assumed to satisfy it.

$\square$
:::

### D. Timestep Constraint (CFL-like)

:::{div} feynman-prose
Anyone who has done numerical computation knows about timestep stability. If you make the timestep too large, oscillations build up and the solution explodes. This is not a bug—it is a fundamental limitation of discrete approximations to continuous dynamics.

The CFL condition (named after Courant, Friedrichs, and Lewy) is the classic stability criterion for wave equations: the numerical "information speed" must exceed the physical wave speed. For our Langevin dynamics with friction $\gamma$ and potential curvature $\omega$, the analogous condition is $h < 2/\omega$.

Why? Because if the timestep is larger than a half-period of the fastest oscillation, the integrator cannot resolve that oscillation. It sees aliased garbage instead. The velocity Verlet and BAOAB integrators both inherit this limitation.

The practical bound $h < 0.1$ is conservative—it provides margin for nonlinear effects and multi-scale interactions that the linearized analysis does not capture.
:::

:::{admonition} Heuristic: Timestep Stability
:class: feynman-added tip

For numerical stability, a conservative guideline is:

$$
\boxed{h < \min\left(\frac{2}{\omega}, 0.1\right)}
$$

where $\omega = \sqrt{\lambda_{\max}(\nabla^2 U)}$ is the maximum eigenfrequency of the potential. This bound is a
discretization guideline, not a formal requirement of the convergence proof.

**Derivation**:

1. **BAOAB weak error**: The BAOAB integrator has weak error $O(h^2)$ per unit time. For total error $< \varepsilon_{\text{disc}}$ over time $T$:

$$
h < \sqrt{\varepsilon_{\text{disc}} / T} \cdot C_{\text{BAOAB}}
$$

2. **Velocity Verlet stability**: For a harmonic oscillator with frequency $\omega$, the Verlet method requires:

$$
\omega h < 2 \quad \implies \quad h < \frac{2}{\omega}
$$

3. **Practical bound**: For strongly confined systems ($\omega \sim 10$), this gives $h < 0.2$. The conservative bound $h < 0.1$ provides safety margin.

**BAOAB discrete-time consistency**:

For BAOAB with coefficients $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1-c_1^2)T_c}$:

The noise injection variance per step is $c_2^2 = (1 - e^{-2\gamma h})T_c$.

For small $h$: $c_2^2 \approx 2\gamma h \cdot T_c$ (first-order expansion).

The **stationary velocity distribution** has temperature $T_c$ (matching the continuous limit), but the per-step noise variance scales with $h$. This is correct: smaller timesteps inject less noise per step but take more steps, yielding the same equilibrium.

:::

### E. Noise Injection (LSI Spectral Gap)

:::{div} feynman-prose
Here is a tension you need to understand. When we clone a particle, we add a small amount of noise—that is the "jitter" $\sigma_x$. Too little jitter, and cloned particles stay too close together, leading to genealogical collapse (everyone becomes a clone of one ancestor). Too much jitter, and we lose precision—particles diffuse away from good regions.

The Log-Sobolev Inequality (LSI) gives us the lower bound. It says that for the algorithm to mix properly, the jitter must provide enough randomization to overcome the correlations introduced by cloning. The formula involves the geometric constants $\kappa_{\text{conf}}$ and $\kappa_W$ and the interaction complexity $C_0$.

But there is also an upper bound: $\sigma_x \leq \varepsilon$. The jitter should not exceed the kernel scale, or you are essentially teleporting particles randomly instead of performing local perturbations.

In practice, these bounds often conflict. When they do, you clip $\sigma_x = \varepsilon$ and accept that convergence will be dominated by Wasserstein contraction rather than LSI mixing.
:::

:::{prf:proposition} Jitter Lower Bound
:label: prop-alg-sieve-jitter-bound

The cloning jitter must satisfy:

$$
\boxed{\sigma_x^2 \geq \frac{\lambda_{\text{target}} \cdot C_0}{\gamma \cdot \kappa_{\text{conf}} \cdot \kappa_W}}
$$

**Derivation** (from LSI constant requirement):

From {prf:ref}`thm-alg-sieve-error-bounds`, the LSI constant is:

$$
\lambda_{\text{LSI}} = \frac{\gamma \cdot \kappa_{\text{conf}} \cdot \kappa_W \cdot \delta^2}{C_0}
$$

where $\delta^2 = \sigma_x^2$ is the jitter variance.

For KL convergence at rate $\lambda_{\text{target}}$, we need $\lambda_{\text{LSI}} \geq \lambda_{\text{target}}$:

$$
\sigma_x^2 \geq \frac{\lambda_{\text{target}} \cdot C_0}{\gamma \cdot \kappa_{\text{conf}} \cdot \kappa_W}
$$

**Upper bound**: $\sigma_x \leq \varepsilon$ (jitter must not exceed kernel scale to maintain locality).

:::{note}
**Bound Tension**: For typical parameters, the LSI lower bound may exceed the locality upper bound. In this case, the algorithm clips $\sigma_x = \varepsilon$, and the effective LSI convergence rate is slower than $\lambda_{\text{target}}$. The actual convergence is then dominated by the Wasserstein contraction rate $\kappa_W$ rather than the LSI rate.
:::

$\square$
:::

(sec-quantitative-bounds)=
## Quantitative Bounds Table

:::{prf:definition} Valid Parameter Ranges (Theory-Derived)
:label: def-alg-sieve-parameter-table

| Parameter | Symbol | Unit | Lower Bound (derived) | Upper Bound | Recommended Default | Status |
|-----------|--------|------|----------------------|-------------|---------------------|--------|
| Population | $N$ | [count] | $\geq 2$ (for mixing) | $\infty$ | 50 | rigorous |
| Kernel scale | $\varepsilon$ | [distance] | $D_{\text{alg}} / \sqrt{2\ln((n_{\mathrm{alive}}-1)/p_{\min})}$ | $\infty$ | 0.1 (check vs. $p_{\min}$) | rigorous (sufficient) |
| Friction | $\gamma$ | [1/time] | $\gamma_*=\frac{c_2 M^2}{c_1 \lambda} + \frac{C_{\text{Dob}}\nu_{\text{clone}}}{c_1 \kappa_W}$ | — | 1.0 (check) | rigorous (Appendix 15) |
| Temperature | $T_c$ | [dimensionless] | $> 0$ | $\infty$ | 1.0 | rigorous |
| Timestep | $h$ | [time] | $> 0$ | heuristic: $\min(2/\omega, 0.1)$ | 0.01 | heuristic |
| Cloning jitter | $\sigma_x$ | [distance] | $\sqrt{\lambda_{\text{target}} C_0 / (\gamma \kappa_{\text{conf}} \kappa_W)}$ | heuristic: $\le \varepsilon$ | 0.1 (check) | rigorous lower / heuristic upper |
| Reward exponent | $\alpha$ | [dimensionless] | $\geq 0$ | — | 1.0 | rigorous |
| Diversity exponent | $\beta$ | [dimensionless] | $> 0$ | — | 1.0 | rigorous |
| Positivity floor | $\eta$ | [dimensionless] | $> 0$ | — | 0.1 | rigorous |
| Logistic bound | $A$ | [dimensionless] | $> 0$ | $\infty$ | 2.0 | rigorous |
| Clone regularizer | $\varepsilon_{\text{clone}}$ | [dimensionless] | $> 0$ | — | 0.01 | rigorous |
| Max clone prob | $p_{\max}$ | [probability] | $> 0$ | $1$ | 1.0 | rigorous |
:::

(sec-convergence-rate)=
## Convergence Rate Formula

:::{div} feynman-prose
Now we come to what everyone really wants to know: *how fast does this thing converge?*

The answer is simultaneously simple and subtle. Simple because there is a single number $\kappa_{\text{total}}$ that tells you the exponential convergence rate. Subtle because that number is the minimum of several competing mechanisms, each of which could be the bottleneck.

Think of it like a chain: the strength of the chain is determined by the weakest link. The convergence rate is determined by the slowest mechanism:

- Position contraction $\kappa_x$ (selection + jitter)
- Velocity contraction $\kappa_v$ (OU friction)
- Wasserstein contraction $\kappa_W$ (companion geometry + cloning)
- Boundary contraction $\kappa_b$ (killing/revival)

Whichever of these is smallest becomes your bottleneck. And on top of that, you have the $O(1/\sqrt{N})$ error floor from finite population—a fundamental limit you cannot beat without adding more particles.

The practical implication: if your algorithm is converging slowly, figure out which mechanism is limiting. Then either accept the rate or change the relevant parameters.
:::

:::{prf:theorem} Total Convergence Rate (Rigorous)
:label: thm-alg-sieve-total-rate-rigorous

The discrete-time convergence to the QSD occurs exponentially with rate:

$$
\kappa_{\text{total}} = \min(\kappa_x,\kappa_v,\kappa_W,\kappa_b)\,(1-\epsilon_{\text{coupling}})
$$

where the component rates are defined in Appendix 06 and implemented in `src/fragile/fractalai/convergence_bounds.py`.

**Finite-N correction**: The mean-field approximation error (Appendix 09, 13) adds an $O(1/\sqrt{N})$ error floor:

$$
\|\mu_N - \rho_0\|_{\text{TV}} \leq e^{-\kappa_{\text{total}} t} + \frac{C_{\text{chaos}}}{\sqrt{N}}
$$

**Stability requirement**: The acoustic stability margin must be positive for convergence:

$$
\gamma > \gamma_*
$$

This is a **necessary condition**, not an additional rate contribution.

*Proof*:
- $\kappa_W$: Structural contraction from companion geometry (Appendix 04)
- $\kappa_x,\kappa_v,\kappa_b$: component rates from the Lyapunov decomposition (Appendix 06)
- The $O(1/\sqrt{N})$ term is an additive error, not a rate correction (Appendix 13, Theorem 4.2)

The minimum component rate (after coupling penalty) determines the asymptotic rate. $\square$
:::

:::{admonition} Conjectured Enhancement (Heuristic)
:class: warning

The hypocoercive term

$$
\kappa_{\text{hypo}} \approx \frac{\gamma \rho_{\text{LSI}}}{M^2}
$$

appears in Appendix 10 but the derivation has gaps. If valid, this would add to the convergence rate. **Treat as heuristic guidance only.**
:::

:::{div} feynman-prose
Let me unpack what that mixing time really means. If you want error of 1% (that is, $\varepsilon = 0.01$), you need on the order of $\ln(100)$ relaxation times once you plug in your computed $\kappa_{\text{total}}$ and the equilibrium constant $C_{\text{total}}$.

The $O(1/\sqrt{N})$ floor still matters: no matter how long you run, finite $N$ creates an irreducible error. To reduce that floor, you must increase $N$.
:::

:::{prf:corollary} Mixing Time
:label: cor-alg-sieve-mixing-time

The mixing time to reach error $\varepsilon$ (beyond the finite-$N$ floor) is:

$$
T_{\text{mix}}(\varepsilon) = \frac{1}{\kappa_{\text{total}}} \ln\left(\frac{V_{\text{init}}\kappa_{\text{total}}}{\varepsilon\, C_{\text{total}}}\right)
$$

This is the formula implemented in `T_mix` (Appendix 06 / `convergence_bounds.py`). When $V_{\text{init}}$ and $C_{\text{total}}$ are $O(1)$, the simplified $\ln(1/\varepsilon)$ scaling is a good approximation.

**Note**: The achievable error is limited by the finite-$N$ floor $C_{\text{chaos}}/\sqrt{N}$.
:::

(sec-parameter-selection)=
## Parameter Selection Checklist

:::{div} feynman-prose
Now I want to show you something practical. All these theoretical bounds are useless if you cannot turn them into actual parameter values. So here is a checklist that does exactly that.

The idea is simple: given your problem specification (Hessian bound $M^2$, dimension $d$, target error $\varepsilon$, population size $N$), compute parameters that satisfy the rigorous constraints and flag the heuristic choices. The checklist walks through each constraint, applies the bound, and checks that everything is consistent.

This is not machine learning—it is just arithmetic applied carefully. Each parameter depends on the others, so you have to compute them in a sensible order. Start with the bounds that do not depend on the rest (minorization and fitness ranges), then friction (acoustic limit), then noise floor, and finally the heuristic tuning rules.

The checklist below is a conservative template. It separates rigorous bounds from heuristic choices so you can see which knobs are certified and which are tuning rules.
:::

:::{prf:algorithm} Parameter Selection Checklist (Template)
:label: alg-alg-sieve-parameter-selection

**Inputs**: problem constants $(M^2, d, N)$, kernel target $p_{\min}$, LSI target $\lambda_{\text{target}}$, and acoustic-limit constants $(c_1, c_2, \lambda, C_{\text{Dob}}, \nu_{\text{clone}}, \kappa_W)$ from Appendix 15 or profiling.

**Steps**:
1. Compute fitness bounds $V_{\min}=\eta^{\alpha+\beta}$ and $V_{\max}=(A+\eta)^{\alpha+\beta}$.
2. Choose a target minorization floor $p_{\min}$ and set
   $\varepsilon \ge D_{\text{alg}} / \sqrt{2\ln((n_{\mathrm{alive}}-1)/p_{\min})}$.
3. **Acoustic limit (rigorous)**: compute
   $\gamma_* = \frac{c_2 M^2}{c_1 \lambda} + \frac{C_{\text{Dob}}\,\nu_{\text{clone}}}{c_1 \kappa_W}$ and choose $\gamma \ge \gamma_*.$
4. **LSI noise floor (rigorous)**: choose
   $\sigma_x^2 \ge \lambda_{\text{target}} C_0/(\gamma\kappa_{\text{conf}}\kappa_W)$.
   If $\sigma_x > \varepsilon$, note that the LSI target is not achievable and convergence is dominated by $\kappa_W$.
5. **Heuristics**: pick a small $h$ for BAOAB stability (e.g., $h<2/\omega$) and, if using phase balance, target $\Gamma\approx 1$.
6. Once component rates $(\kappa_x,\kappa_v,\kappa_W,\kappa_b)$ are available, compute
   $\kappa_{\text{total}}=\min(\kappa_x,\kappa_v,\kappa_W,\kappa_b)\,(1-\epsilon_{\text{coupling}})$ and report the finite-$N$ error floor $\sim 1/\sqrt{N}$.

**Output**: a parameter set that satisfies the rigorous bounds, plus heuristic choices explicitly flagged as such.
:::



(sec-17-node-verification)=
## Verification: 17-Node Sieve Instantiation

:::{div} feynman-prose
Finally, let me show you how all these constraints get verified systematically. The 17-node sieve is not just a list of things to check—it is a logical structure that ensures completeness.

Each node represents a potential failure mode. Node 1 checks energy bounds. Node 10 checks ergodicity. Node 17 checks for deadlocks. When all 17 nodes pass, you have *proved* that the algorithm cannot fail in any of the ways we know about.

The beauty is that this is not a black box. Each node has an explicit predicate and produces a concrete certificate. If something fails, you know exactly which constraint was violated and can trace back to the parameter that caused it.

This is what distinguishes rigorous parameter selection from trial-and-error tuning. We are not hoping the parameters work—we are *proving* they work.
:::

The tight bounds are certified by the 17-node sieve:

```{raw} html
<figure class="sieve-diagram">
  <svg
    width="960"
    height="520"
    viewBox="0 0 960 520"
    role="img"
    aria-labelledby="sieveTitle sieveDesc"
    xmlns="http://www.w3.org/2000/svg"
    shape-rendering="geometricPrecision"
  >
    <title id="sieveTitle">17-node verification sieve</title>
    <desc id="sieveDesc">
      Seventeen labeled nodes arranged in four rows, showing the verification checkpoints
      from energy and recovery through alignment and lock.
    </desc>
    <defs>
      <linearGradient id="sieveBg" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%" stop-color="#f4f0e8" />
        <stop offset="100%" stop-color="#e6edf6" />
      </linearGradient>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.18" />
      </filter>
      <style>
        .sieve-title {
          font: 600 22px/1.2 "IBM Plex Sans", "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
          fill: #1b2430;
          letter-spacing: 0.4px;
        }
        .sieve-divider {
          stroke: #cbd5e1;
          stroke-width: 1;
        }
        .node rect {
          fill: #ffffff;
          stroke: #94a3b8;
          stroke-width: 1.2;
        }
        .node text {
          font-family: "IBM Plex Sans", "Source Sans 3", "Helvetica Neue", Arial, sans-serif;
          text-anchor: middle;
        }
        .node-num {
          font-size: 12px;
          fill: #475569;
          font-weight: 600;
        }
        .node-label {
          font-size: 14px;
          fill: #111827;
          font-weight: 600;
          letter-spacing: 0.2px;
        }
      </style>
    </defs>
    <rect width="960" height="520" rx="18" fill="url(#sieveBg)" />
    <text class="sieve-title" x="480" y="38">17-Node Verification Sieve</text>
    <line class="sieve-divider" x1="60" y1="167" x2="900" y2="167" />
    <line class="sieve-divider" x1="60" y1="277" x2="900" y2="277" />
    <line class="sieve-divider" x1="60" y1="387" x2="900" y2="387" />

    <g class="node">
      <rect x="150" y="80" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="225" y="104">Node 1</text>
      <text class="node-label" x="225" y="126">Energy</text>
    </g>
    <g class="node">
      <rect x="320" y="80" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="395" y="104">Node 2</text>
      <text class="node-label" x="395" y="126">Recovery</text>
    </g>
    <g class="node">
      <rect x="490" y="80" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="565" y="104">Node 3</text>
      <text class="node-label" x="565" y="126">Confinement</text>
    </g>
    <g class="node">
      <rect x="660" y="80" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="735" y="104">Node 4</text>
      <text class="node-label" x="735" y="126">Scaling</text>
    </g>

    <g class="node">
      <rect x="65" y="190" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="140" y="214">Node 5</text>
      <text class="node-label" x="140" y="236">Parameters</text>
    </g>
    <g class="node">
      <rect x="235" y="190" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="310" y="214">Node 6</text>
      <text class="node-label" x="310" y="236">Capacity</text>
    </g>
    <g class="node">
      <rect x="405" y="190" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="480" y="214">Node 7</text>
      <text class="node-label" x="480" y="236">Analyticity</text>
    </g>
    <g class="node">
      <rect x="575" y="190" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="650" y="214">Node 8</text>
      <text class="node-label" x="650" y="236">Topology</text>
    </g>
    <g class="node">
      <rect x="745" y="190" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="820" y="214">Node 9</text>
      <text class="node-label" x="820" y="236">Tameness</text>
    </g>

    <g class="node">
      <rect x="150" y="300" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="225" y="324">Node 10</text>
      <text class="node-label" x="225" y="346">Ergodicity</text>
    </g>
    <g class="node">
      <rect x="320" y="300" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="395" y="324">Node 11</text>
      <text class="node-label" x="395" y="346">Complexity</text>
    </g>
    <g class="node">
      <rect x="490" y="300" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="565" y="324">Node 12</text>
      <text class="node-label" x="565" y="346">Oscillation</text>
    </g>
    <g class="node">
      <rect x="660" y="300" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="735" y="324">Node 13</text>
      <text class="node-label" x="735" y="346">Boundary</text>
    </g>

    <g class="node">
      <rect x="150" y="410" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="225" y="434">Node 14</text>
      <text class="node-label" x="225" y="456">Overload</text>
    </g>
    <g class="node">
      <rect x="320" y="410" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="395" y="434">Node 15</text>
      <text class="node-label" x="395" y="456">Starvation</text>
    </g>
    <g class="node">
      <rect x="490" y="410" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="565" y="434">Node 16</text>
      <text class="node-label" x="565" y="456">Alignment</text>
    </g>
    <g class="node">
      <rect x="660" y="410" width="150" height="64" rx="12" filter="url(#softShadow)" />
      <text class="node-num" x="735" y="434">Node 17</text>
      <text class="node-label" x="735" y="456">Lock</text>
    </g>
  </svg>
  <figcaption>Structured layout of the 17 verification nodes used by the sieve.</figcaption>
</figure>
```

| Node | Constraint | Bound Verified |
|------|------------|----------------|
| 1 (Energy) | $\Phi \in [0, V_{\max}]$ | $V_{\max} = 4.41$ |
| 2 (Recovery) | Bad set finite | Cloning repairs |
| 3 (Confinement) | $S_N$ symmetry | Permutation invariance |
| 4 (Scaling) | Critical/controlled by confinement | BarrierTypeII + Foster-Lyapunov |
| 5 (Parameters) | Constants fixed | Table values |
| 6 (Capacity) | Bad set capacity | Finite |
| 7 (Analyticity) | $C^2$ regularity | Bounded derivatives |
| 8 (Topology) | Single sector | Connected ball |
| 9 (Tameness) | O-minimal | Definable |
| 10 (Ergodicity) | Doeblin + hypoelliptic | $m_\varepsilon > 0$, $T_c > 0$ |
| 11 (Complexity) | Finite precision | Float64 |
| 12 (Oscillation) | Bounded | Alive core |
| 13 (Boundary) | Open + killing | Recovery via cloning |
| 14 (Overload) | Controlled | Thermostat |
| 15 (Starvation) | QSD conditioning | $n_{\text{alive}} \geq 1$ |
| 16 (Alignment) | Selection pressure | Mean fitness increase |
| 17 (Lock) | Pattern blocked | Invariant mismatch |

## Summary

:::{div} feynman-prose
Let me step back and tell you what we have accomplished here.

We started with three separate analyses—appendix proofs, algorithm bounds, hypostructure certificates—each giving partial information about parameter constraints. By combining them systematically, we separated **rigorous constraints** (acoustic limit, Doeblin minorization, LSI noise floor) from **heuristic tuning rules** (phase balance, timestep stability).

Defaults are useful starting points, but they must be checked against problem-specific constants such as $D_{\text{alg}}$, $\kappa_W$, and the acoustic-limit $\gamma_*$. We also made clear which statements are proven, which are asymptotic, and which remain conjectured.
:::

This chapter derived parameter constraints for the Fractal Gas by:

1. **Using proven appendix results** (04, 06, 09, 13) for contraction, Lyapunov structure, and error bounds
2. **Translating algorithm definitions** into explicit bounds (fitness ranges, cloning scores, minorization floors)
3. **Separating rigorous constraints from heuristics** (phase balance and timestep stability)
4. **Providing a conservative selection checklist** with explicit dependencies
5. **Clearly marking asymptotic and conjectured results** (mean-field QSD scaling, hypocoercive rate)

Defaults should be validated against the bounds above rather than assumed to satisfy them.

## References

```{bibliography}
:filter: docname in docnames
```
