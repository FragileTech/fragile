---
title: "Stability and Composition Metatheorems"
---

# Stability & Composition Metatheorems

(sec-perturbation-and-coupling)=
## Perturbation and Coupling

The **Stability and Composition Metatheorems** govern how Sieve verdicts extend from individual systems to families of systems (perturbations) and coupled systems (compositions). These theorems answer the fundamental questions:

1. *"If my model is slightly wrong, does the proof hold?"* (Stability)
2. *"Can I build a regular system out of regular parts?"* (Composition)

These metatheorems are **universal**: they apply to any valid Hypostructure because they operate on the Certificate algebra, not the underlying physics.

:::{div} feynman-prose
Here is something that worries every honest physicist: *What if my model is wrong?*

Not catastrophically wrong---we would notice that. I mean subtly wrong. Your viscosity coefficient is off by 2%. Your boundary conditions are idealized. Your numerical simulation has round-off error. Does your entire proof collapse?

The metatheorems in this section say: *No, it does not.* If your proof has enough margin---if you are not balanced on a knife-edge---then small errors do not matter. This is not hand-waving. It is a precise mathematical statement about the *topology* of the space of proofs.

The key insight is this: the Sieve operates on certificates, not on the underlying physics directly. And certificates that pass with strict inequalities form an *open set*. Small perturbations preserve openness. This is the implicit function theorem doing its quiet, powerful work.
:::

---

(sec-openness-of-regularity)=
### Openness of Regularity

:::{div} feynman-prose
Let me tell you what "openness" really means here. Imagine you have certified a system as globally regular. You have your certificate $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$ in hand. Now someone comes along and says, "But your parameter $\theta_0$ is only known to 6 decimal places. What about $\theta_0 + 0.000001$?"

The openness theorem says: *Relax.* If your barriers are strict---if Gap is not merely positive but *bounded away from zero*---then there exists a whole neighborhood of parameters where the proof still works. You do not need to redo anything. The certificate algebra is continuous.

Think of it like a mountain pass. If you are at the very top of the saddle point, the slightest wind might blow you either way. But if you are safely past the pass, in the valley, a small breeze changes nothing essential about your descent.
:::

:::{prf:theorem} [KRNL-Openness] Openness of Regularity
:label: mt-krnl-openness

**Source:** Dynamical Systems (Morse-Smale Stability) / Geometric Analysis.

**Hypotheses.** Let $\mathcal{H}(\theta_0)$ be a Hypostructure depending on parameters $\theta \in \Theta$ (a topological space). Assume:
1. Global Regularity at $\theta_0$: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}(\theta_0)$
2. Strict barriers: $\mathrm{Gap}(\theta_0) > \epsilon$, $\mathrm{Cap}(\theta_0) < \delta$ for some $\epsilon, \delta > 0$
3. Continuous dependence: the certificate functionals are continuous in $\theta$

**Statement:** The set of Globally Regular Hypostructures is **open** in the parameter topology. There exists a neighborhood $U \ni \theta_0$ such that $\forall \theta \in U$, $\mathcal{H}(\theta)$ is also Globally Regular.

**Certificate Logic:**

$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}(\theta_0) \wedge (\mathrm{Gap} > \epsilon) \wedge (\mathrm{Cap} < \delta) \Rightarrow \exists U: \forall \theta \in U, K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}(\theta)

$$

**Use:** Validates that the proof is robust to small modeling errors or physical noise.

**Literature:** {cite}`Smale67`; {cite}`PalisdeMelo82`; {cite}`Robinson99`
:::

:::{prf:proof}

Strict inequalities define open sets. The Morse-Smale stability theorem (Palis and de Melo, 1982) states that structurally stable systems form an open set. The key is non-degeneracy: if all eigenvalues are strictly away from zero and all capacities are strictly bounded, small perturbations preserve these properties. This is the implicit function theorem applied to the certificate functionals.
:::

---

(sec-shadowing-metatheorem)=
### Shadowing Metatheorem

:::{div} feynman-prose
Now here is something that should give you great comfort if you ever run numerical simulations.

Every computer makes errors. Your floating-point arithmetic rounds off. Your time-stepper introduces small drift. After a million steps, your computed trajectory $\{y_n\}$ has accumulated countless tiny mistakes. It is not the *true* orbit---it is what we call a "pseudo-orbit." At each step, you are almost solving the equation, but not quite.

The question is: does this matter? Is your simulation telling you anything about the real dynamics?

The Shadowing Theorem says: *Yes, if you have hyperbolicity.* When your system has a spectral gap $\lambda > 0$---meaning errors contract exponentially fast---then something remarkable happens. There exists a *genuine* trajectory of the true system that stays within distance $\delta$ of your computed pseudo-orbit. Your simulation may not be following the exact trajectory you think it is, but it *is* following some true trajectory. The error $\delta$ goes like $\varepsilon/\lambda$: small numerical error divided by contraction rate.

This is how numerical analysis gets upgraded to rigorous existence proofs. You compute a solution, verify hyperbolicity, and the Shadowing Theorem hands you a mathematical guarantee that a real solution exists nearby.
:::

:::{prf:theorem} [KRNL-Shadowing] Shadowing Metatheorem
:label: mt-krnl-shadowing

**Source:** Hyperbolic Dynamics (Anosov Shadowing Lemma).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A Stiffness certificate: $K_{\mathrm{LS}_\sigma}^+$ with spectral gap $\lambda > 0$
2. A numerical pseudo-orbit: $\{y_n\}$ with $d(f(y_n), y_{n+1}) < \varepsilon$ for all $n$
3. Hyperbolicity: the tangent map $Df$ has exponential dichotomy

**Statement:** For every $\varepsilon$-pseudo-orbit (numerical simulation), there exists a true orbit $\{x_n\}$ that $\delta(\varepsilon)$-shadows it: $d(x_n, y_n) < \delta(\varepsilon)$ for all $n$. The shadowing distance satisfies $\delta(\varepsilon) = O(\varepsilon/\lambda)$.

**Certificate Logic:**

$$
K_{\mathrm{LS}_\sigma}^+ \wedge K_{\text{pseudo}}^{\varepsilon} \Rightarrow K_{\text{true}}^{\delta(\varepsilon)}

$$

**Use:** Upgrades a high-precision **Numerical Simulation** into a rigorous **Existence Proof** for a nearby solution (essential for $T_{\text{algorithmic}}$).

**Literature:** {cite}`Anosov67`; {cite}`Bowen75`; {cite}`Palmer88`
:::

:::{prf:proof}

The Anosov shadowing lemma (1967) states that uniformly hyperbolic systems have the shadowing property. The spectral gap $\lambda$ controls the contraction rate, and the shadowing distance is $\delta \sim \varepsilon/\lambda$. Bowen (1975) extended this to Axiom A systems. Palmer (1988) gave a proof via the contraction mapping theorem on sequence spaces.
:::

---

(sec-weak-strong-uniqueness)=
### Weak-Strong Uniqueness

:::{div} feynman-prose
Here is something that causes genuine anxiety in PDE theory: *non-uniqueness of weak solutions.*

When you solve a differential equation by the "weak" method---using compactness arguments, extracting subsequences, passing to limits---you get a solution, but you have no guarantee it is the *only* solution. Maybe there are infinitely many weak solutions, and you just happened to grab one. Maybe the physics you are trying to model has fundamentally ambiguous predictions.

The Weak-Strong Uniqueness principle resolves this anxiety. It says: *If a strong solution exists, it is unique, and all weak solutions must agree with it.*

What makes a solution "strong"? Local stiffness---the $K_{\mathrm{LS}_\sigma}^+$ certificate. A strong solution has enough regularity that you can control its behavior precisely. It does not wobble or branch.

The theorem says that weak solutions cannot escape from strong ones. If you start at the same initial data, and a strong solution exists even locally, then the weak solution you constructed by compactness methods must coincide with it. The weak solution may be hiding extra regularity you did not expect. It has no freedom to branch off in a different direction.

This is profoundly reassuring. Your compactness construction did not introduce spurious solutions. The physics is unambiguous wherever regularity holds.
:::

:::{prf:theorem} [KRNL-WeakStrong] Weak-Strong Uniqueness
:label: mt-krnl-weak-strong

**Source:** PDE Theory (Serrin/Prodi-Serrin Criteria).

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. A "Weak" solution $u_w$ constructed via concentration-compactness ($K_{C_\mu}$)
2. A "Strong" local solution $u_s$ with Stiffness ($K_{\mathrm{LS}_\sigma}^+$) on $[0, T]$
3. Both solutions have the same initial data: $u_w(0) = u_s(0)$

**Statement:** If a "Strong" solution exists on $[0, T]$, it is unique. Any "Weak" solution constructed via Compactness/Surgery must coincide with the Strong solution almost everywhere: $u_w = u_s$ a.e. on $[0, T] \times \Omega$.

**Certificate Logic:**

$$
K_{C_\mu}^{\text{weak}} \wedge K_{\mathrm{LS}_\sigma}^{\text{strong}} \Rightarrow K_{\text{unique}}

$$

**Use:** Resolves the "Non-Uniqueness" anxiety in weak solutions. If you can prove stiffness locally, the weak solution cannot branch off.

**Literature:** {cite}`Serrin63`; {cite}`Lions96`; {cite}`Prodi59`
:::

:::{prf:proof}

The weak-strong uniqueness principle uses energy estimates. If $v = u_w - u_s$, then $\frac{d}{dt}\|v\|^2 \leq C\|v\|^2 \cdot \|u_s\|_{X}$ for an appropriate norm $X$. If $u_s \in L^p([0,T]; X)$ (Serrin class), Gronwall's inequality gives $\|v(t)\| = 0$. For Navier-Stokes, $X = L^r$ with $\frac{2}{p} + \frac{3}{r} = 1$, $r > 3$ (Serrin, 1963; Lions, 1996).
:::

---

(sec-product-regularity-metatheorem)=
### Product-Regularity Metatheorem

:::{div} feynman-prose
And now we come to what I think is the most practically important result in this section: *Can you build a safe system out of safe parts?*

This matters enormously for engineering. Suppose you have verified that your neural network is well-behaved. And you have verified that your physics engine is stable. Now you couple them together---the neural net controls the physics simulation. Is the combined system safe?

Without a theorem, you would have to verify the composite system from scratch. All your previous work would be useless. The modular verification dream would collapse.

The Product-Regularity Metatheorem says: *Yes, you can compose.* If you have Lock certificates for both components, and the coupling is "weak enough" in a precise sense, then the product system inherits global regularity.

What does "weak enough" mean? The theorem gives three alternative backends---three different ways to verify that your coupling is sufficiently tame:

**Backend A** says the coupling should be *subcritical*: it should scale slower than either component's critical exponent. Think of it as: the interaction term is always dominated by the intrinsic dynamics.

**Backend B** uses semigroup theory: if each component generates a nice evolution operator, and the coupling is a bounded (or relatively bounded) perturbation, the combined system is still nice.

**Backend C** is energy-based: if the coupling cannot pump energy into the system faster than dissipation removes it, you are safe.

Each backend accommodates different proof styles. Use whichever matches your problem best. The certificate algebra handles the logic; you just need to verify one of the three coupling conditions.
:::

:::{prf:theorem} [LOCK-Product] Product-Regularity
:label: mt-lock-product

**Sieve Signature:**
- **Required Permits (Alternative Backends):**
  - **Backend A:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^B \wedge K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+$ (Subcritical Scaling + Coupling Control)
  - **Backend B:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^B \wedge K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+$ (Semigroup + Perturbation + ACP)
  - **Backend C:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^B \wedge K_{\mathrm{LS}_\sigma}^{\text{abs}}$ (Energy + Absorbability)
- **Weakest Precondition:** $\{K_{\mathrm{Cat}_{\mathrm{Hom}}}^A, K_{\mathrm{Cat}_{\mathrm{Hom}}}^B\}$ (component regularity certified)
- **Produces:** $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{A \times B}$ (product system globally regular)
- **Blocks:** All failure modes on product space
- **Breached By:** Strong coupling exceeding perturbation bounds

**Context:** Product systems arise when composing verified components (e.g., Neural Net + Physics Engine, multi-scale PDE systems, coupled oscillators). The principle of **modular verification** requires that certified components remain certified under weak coupling.

**Certificate Logic:**

$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^A \wedge K_{\mathrm{Cat}_{\mathrm{Hom}}}^B \wedge \left((K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+) \vee (K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+) \vee K_{\mathrm{LS}_\sigma}^{\text{abs}}\right) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{A \times B}

$$

:::

:::{prf:proof}

#### Backend A: Subcritical Scaling

**Hypotheses:**
1. Component Hypostructures $\mathcal{H}_A = (\mathcal{X}_A, \Phi_A, \mathfrak{D}_A)$ and $\mathcal{H}_B = (\mathcal{X}_B, \Phi_B, \mathfrak{D}_B)$
2. Lock certificates: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A$ and $K_{\mathrm{Cat}_{\mathrm{Hom}}}^B$ (global regularity for each)
3. Coupling term $\Phi_{\text{int}}: \mathcal{X}_A \times \mathcal{X}_B \to \mathbb{R}$ with scaling exponent $\alpha_{\text{int}}$
4. **Subcritical condition:** $\alpha_{\text{int}} < \min(\alpha_c^A, \alpha_c^B)$
5. **Coupling control** (permit $K_{\mathrm{CouplingSmall}}^+$, {prf:ref}`def-permit-couplingsmall`): Dissipation domination constants $\lambda_A, \lambda_B > 0$ with $\mathfrak{D}_i \geq \lambda_i E_i$, and energy absorbability $|\dot{E}_{\text{int}}| \leq \varepsilon(E_A + E_B) + C_\varepsilon$ for some $\varepsilon < \min(\lambda_A, \lambda_B)$

**Certificate:** $K_{\mathrm{SC}_\lambda}^{\text{sub}} \wedge K_{\mathrm{CouplingSmall}}^+ = (\alpha_{\text{int}}, \alpha_c^A, \alpha_c^B, \delta, \lambda_A, \lambda_B, \varepsilon, \text{absorbability witness})$

(proof-mt-lock-product-backend-a)=
**Proof (5 Steps):**

*Step 1 (Scaling Structure).* Define the scaling action $\lambda \cdot (x_A, x_B) = (\lambda^{a_A} x_A, \lambda^{a_B} x_B)$ where $a_A, a_B$ are the homogeneity weights. The total height functional transforms as:

$$
\Phi_{\text{tot}}(\lambda \cdot x) = \lambda^{\alpha_A} \Phi_A(x_A) + \lambda^{\alpha_B} \Phi_B(x_B) + \lambda^{\alpha_{\text{int}}} \Phi_{\text{int}}(x_A, x_B)

$$

*Step 2 (Subcritical Dominance).* Since $\alpha_{\text{int}} < \min(\alpha_c^A, \alpha_c^B)$, the interaction term is asymptotically subdominant. For large $\lambda$:

$$
|\Phi_{\text{int}}(\lambda \cdot x)| \leq C \lambda^{\alpha_{\text{int}}} = o(\lambda^{\alpha_c})

$$

The interaction cannot drive blow-up faster than the natural scaling.

*Step 3 (Decoupled Barrier Transfer).* The Lock certificates $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A, K_{\mathrm{Cat}_{\mathrm{Hom}}}^B$ provide a priori bounds:

$$
\|u_A(t)\|_{\mathcal{X}_A} \leq M_A, \quad \|u_B(t)\|_{\mathcal{X}_B} \leq M_B \quad \forall t \geq 0

$$

Under subcritical coupling, these bounds persist with at most polynomial growth correction.

*Step 4 (Energy Control).* The total energy $E_{\text{tot}} = E_A + E_B + E_{\text{int}}$ satisfies:

$$
\frac{d}{dt} E_{\text{tot}} \leq -\mathfrak{D}_A - \mathfrak{D}_B + |\dot{E}_{\text{int}}|

$$

where $\mathfrak{D}_A, \mathfrak{D}_B \geq 0$ are the dissipation rates (energy loss per unit time). Subcriticality implies $|\dot{E}_{\text{int}}| \leq \varepsilon (E_A + E_B) + C_\varepsilon$ for any $\varepsilon > 0$. Choosing $\varepsilon$ small enough that $\varepsilon < \min(\lambda_A, \lambda_B)$ (where $\mathfrak{D}_i \geq \lambda_i E_i$), the dissipation dominates the interaction.

*Step 5 (Grönwall Closure + Global Existence).* Standard Grönwall inequality closes the estimate. **Product local well-posedness** follows from standard semilinear theory: component LWP (guaranteed by the Lock certificates $K_{\mathrm{Cat}_{\mathrm{Hom}}}^A, K_{\mathrm{Cat}_{\mathrm{Hom}}}^B$) extends to the product system under Lipschitz coupling with subcritical growth (Hypotheses 3-4). Combined with the uniform energy bound from Step 4, global existence follows: no singularity can form in the product space.

**Literature:** Scaling analysis {cite}`Tao06`; subcritical perturbation {cite}`CazenaveSemilinear03`

---

#### Backend B: Semigroup + Perturbation Theory

**Hypotheses:**
1. Each component generates a $C_0$-semigroup: $T_A(t) = e^{tA_A}$ on $\mathcal{X}_A$, $T_B(t) = e^{tA_B}$ on $\mathcal{X}_B$
2. Global bounds: $\|T_A(t)\| \leq M_A e^{\omega_A t}$, $\|T_B(t)\| \leq M_B e^{\omega_B t}$ with $\omega_A, \omega_B \leq 0$ (dissipative)
3. Coupling operator $B: D(A_A) \times D(A_B) \to \mathcal{X}_A \times \mathcal{X}_B$ is either:
   - (i) **Bounded:** $\|B\| < \infty$, or
   - (ii) **$A$-relatively bounded:** $\|Bx\| \leq a\|(A_A \oplus A_B)x\| + b\|x\|$ with $a < 1$
4. Lock certificates translate to: trajectories remain in generator domain
5. **Abstract Cauchy Problem formulation** (permit $K_{\mathrm{ACP}}^+$, {prf:ref}`def-permit-acp`): The product dynamics are represented by the abstract Cauchy problem $\dot{u} = Au$, $u(0) = u_0$ on state space $X = \mathcal{X}_A \times \mathcal{X}_B$ with generator $A = A_A \oplus A_B + B$ and domain $D(A) \supseteq D(A_A) \times D(A_B)$

**Certificate:** $K_{D_E}^{\text{pert}} \wedge K_{\mathrm{ACP}}^+ = (A_A, A_B, B, \text{perturbation type}, X, D(A), \text{mild/strong equivalence})$

(proof-mt-lock-product-backend-b)=
**Proof (5 Steps):**

*Step 1 (Product Semigroup).* On $\mathcal{X} = \mathcal{X}_A \times \mathcal{X}_B$, the uncoupled generator $A_0 = A_A \oplus A_B$ generates $T_0(t) = T_A(t) \times T_B(t)$ with:

$$
\|T_0(t)\| \leq M_A M_B e^{\max(\omega_A, \omega_B) t}

$$

*Step 2 (Perturbation Classification).* The total generator is $A = A_0 + B$ where $B$ represents coupling. By hypothesis, $B$ is either bounded or relatively bounded with bound $< 1$.

*Step 3 (Perturbation Theorem Application).*
- If $B$ bounded: **Bounded Perturbation Theorem** (Pazy, Theorem 3.1.1) yields $A$ generates $C_0$-semigroup.
- If $B$ relatively bounded with $a < 1$: **Relatively Bounded Perturbation** (Engel-Nagel, III.2.10) yields same.

*Step 4 (A Priori Bounds from Lock).* The Lock certificates provide:

$$
\sup_{t \in [0,T]} \|(u_A(t), u_B(t))\|_{D(A_0)} < \infty

$$

Standard semigroup theory: if $u(t) \in D(A)$ initially and $A$ generates $C_0$-semigroup, solution exists globally.

*Step 5 (Conclusion).* The perturbed semigroup $e^{tA}$ is globally defined on $\mathcal{X}_A \times \mathcal{X}_B$. No finite-time blow-up.

**Literature:** Semigroup theory {cite}`EngelNagel00`; perturbation of generators {cite}`Pazy83`; coupled parabolic systems {cite}`Cardanobile10`

---

#### Backend C: Energy + Absorbability

**Hypotheses:**
1. Coercive Lyapunov/energy functionals $E_A: \mathcal{X}_A \to \mathbb{R}$, $E_B: \mathcal{X}_B \to \mathbb{R}$:

   $$
   E_A(u) \geq c_A \|u\|_{\mathcal{X}_A}^p - C_A, \quad E_B(v) \geq c_B \|v\|_{\mathcal{X}_B}^q - C_B

   $$

2. Dissipation structure from Lock certificates:

   $$
   \frac{d}{dt} E_A \leq -\lambda_A E_A + d_A, \quad \frac{d}{dt} E_B \leq -\lambda_B E_B + d_B

   $$

3. **Absorbability condition:** The coupling contribution to energy evolution satisfies:

   $$
   \left|\frac{d}{dt}\Phi_{\text{int}}(u(t), v(t))\right| \leq \varepsilon (E_A(u) + E_B(v)) + C_\varepsilon

   $$

   for some $\varepsilon < \min(\lambda_A, \lambda_B)$. (This bounds the *rate* of energy exchange, not the potential itself.)

**Certificate:** $K_{\mathrm{LS}_\sigma}^{\text{abs}} = (E_A, E_B, \lambda_A, \lambda_B, \varepsilon, \text{absorbability witness})$

(proof-mt-lock-product-backend-c)=
**Proof (5 Steps):**

*Step 1 (Total Energy Construction).* Define $E_{\text{tot}} = E_A + E_B$. By coercivity:

$$
E_{\text{tot}}(u, v) \geq c_{\min}(\|u\|^p + \|v\|^q) - C_{\max}

$$

This controls the product norm.

*Step 2 (Energy Evolution).* The time derivative:

$$
\frac{d}{dt} E_{\text{tot}} = \frac{d}{dt} E_A + \frac{d}{dt} E_B + \underbrace{\text{coupling contribution}}_{\leq \varepsilon E_{\text{tot}} + C_\varepsilon}

$$

*Step 3 (Grönwall Closure).* Combining dissipation and absorbability:

$$
\frac{d}{dt} E_{\text{tot}} \leq -(\lambda_{\min} - \varepsilon) E_{\text{tot}} + C

$$

where $\lambda_{\min} = \min(\lambda_A, \lambda_B)$. Since $\varepsilon < \lambda_{\min}$, the coefficient is negative.

*Step 4 (Global Bound).* Standard Grönwall inequality:

$$
E_{\text{tot}}(t) \leq E_{\text{tot}}(0) e^{-(\lambda_{\min} - \varepsilon)t} + \frac{C}{\lambda_{\min} - \varepsilon}

$$

Bounded uniformly in time.

*Step 5 (Conclusion + Global Existence).* Coercivity translates energy bound to norm bound. **Product local well-posedness** follows from standard energy-space theory: the coercive energy bounds (Hypothesis 1) provide control of the state space norms, and the Lipschitz coupling control implicit in the absorbability condition (Hypothesis 3) ensures local existence extends from components to the product. Combined with the uniform bound from Step 4, global existence follows.

**Literature:** Grönwall inequalities {cite}`Gronwall19`; energy methods {cite}`Lions69`; dissipative systems {cite}`Temam97`

---

**Backend Selection Logic:**

| Backend | Required Certificates | Best For |
|:-------:|:--------------------:|:--------:|
| A | $K_{\mathrm{SC}_\lambda}^{\text{sub}}$ (subcritical exponent) | Scaling-critical PDEs, dispersive equations |
| B | $K_{D_E}^{\text{pert}}$ (semigroup perturbation) | Linear/semilinear PDEs, evolution systems |
| C | $K_{\mathrm{LS}_\sigma}^{\text{abs}}$ (energy absorbability) | Dissipative systems, thermodynamic applications |

**Use:** Allows building complex Hypostructures by verifying components and coupling separately. The three backends accommodate different proof styles: scaling-based (A), operator-theoretic (B), and energy-based (C).

:::

---

(sec-subsystem-inheritance)=
### Subsystem Inheritance

:::{div} feynman-prose
Here is the flip side of composition: *inheritance.*

Product-Regularity asks whether combining systems preserves regularity. Subsystem Inheritance asks the reverse: if the big system is safe, are smaller pieces automatically safe?

The answer is yes, and the reason is almost tautological once you see it. An invariant subsystem is a piece of state space that the dynamics never leave. If you start in the subsystem, you stay in the subsystem forever. It is like a room with no doors leading out.

Now suppose the parent system $\mathcal{H}$ has the Lock certificate: no singularities can form anywhere in the full state space. The subsystem $\mathcal{S}$ is contained in that state space. A singularity in $\mathcal{S}$ would be a singularity in $\mathcal{H}$. But we just said there are no singularities in $\mathcal{H}$. Contradiction. Done.

This is useful in practice. Sometimes the full system is easier to analyze than the restricted case. For example, general 3D fluid dynamics might be simpler to prove regular (if we could) than the special case of axisymmetric flow. Once you have the general result, all symmetric restrictions inherit it for free.
:::

:::{prf:theorem} [KRNL-Subsystem] Subsystem Inheritance
:label: mt-krnl-subsystem

**Source:** Invariant Manifold Theory.

**Hypotheses.** Let $\mathcal{H}$ be a Hypostructure with:
1. Global Regularity: $K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}$
2. An invariant subsystem $\mathcal{S} \subset \mathcal{H}$: if $x(0) \in \mathcal{S}$, then $x(t) \in \mathcal{S}$ for all $t$
3. The subsystem inherits the Hypostructure: $\mathcal{H}|_{\mathcal{S}} = (\mathcal{S}, \Phi|_{\mathcal{S}}, \mathfrak{D}|_{\mathcal{S}}, G|_{\mathcal{S}})$

**Statement:** Regularity is hereditary. If the parent system $\mathcal{H}$ admits no singularities (Lock Blocked), then no invariant subsystem $\mathcal{S} \subset \mathcal{H}$ can develop a singularity.

**Certificate Logic:**

$$
K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}(\mathcal{H}) \wedge (\mathcal{S} \subset \mathcal{H} \text{ invariant}) \Rightarrow K_{\mathrm{Cat}_{\mathrm{Hom}}}^{\mathrm{blk}}(\mathcal{S})

$$

**Use:** Proves safety for restricted dynamics (e.g., "If the general 3D fluid is safe, the axisymmetric flow is also safe").

**Literature:** {cite}`Fenichel71`; {cite}`HirschPughShub77`; {cite}`Wiggins94`
:::

:::{prf:proof}

If $\mathcal{S}$ developed a singularity, it would correspond to a morphism $\phi: \mathcal{B}_{\text{univ}} \to \mathcal{S} \hookrightarrow \mathcal{H}$. But this contradicts $\mathrm{Hom}(\mathcal{B}_{\text{univ}}, \mathcal{H}) = \emptyset$. The Fenichel invariant manifold theorem (1971) shows that normally hyperbolic invariant manifolds persist under perturbation; combined with Hirsch-Pugh-Shub (1977), the restriction maintains all regularity properties.
:::
