(sec-cosmological-constants)=
# Cosmological Constants and Regime Transitions

**Prerequisites:** {doc}`03_curvature_gravity`, {doc}`05_holography`

---

## TLDR

*Notation: $\Lambda_{\mathrm{holo}}$ = holographic boundary vacuum; $\Lambda_{\mathrm{bulk}}$ = bulk QSD vacuum; $\Lambda_{\mathrm{eff}}$ = effective exploration vacuum; QSD = quasi-stationary distribution.*

**Three Distinct Cosmological Constants**: The "cosmological constant" is not one number but three: (1) $\Lambda_{\mathrm{holo}} < 0$ from IG pressure at horizons (always negative, yields AdS boundary), (2) $\Lambda_{\mathrm{bulk}} = 0$ at QSD equilibrium (no expansion when stationary), and (3) $\Lambda_{\mathrm{eff}} > 0$ possible during exploration phases (drives observed cosmic expansion).

**Boundary vs Bulk Resolution**: The apparent contradiction between "AdS from holography" and "dS from observations" dissolves when you recognize they measure different geometric quantities. Holographic calculations probe boundary vacuum (surface integral), while cosmological observations measure bulk dynamics (volume integral). These are mathematically distinct operations.

**Exploration Drives Expansion**: Positive effective cosmological constant arises from non-equilibrium exploration dynamics. When walkers spread uniformly rather than cluster on fitness peaks, the defocusing geometry creates negative Ricci curvature along worldlines, which the Raychaudhuri equation converts into sustained positive expansion.

**Dark Energy Reframed**: The observed $\Lambda_{\mathrm{obs}} \approx 10^{-52}$ m$^{-2}$ is not mysterious vacuum energy but exploration pressure---a dynamical signature of a universe that has not yet reached QSD equilibrium. The "cosmological constant problem" becomes a different question: Why is the universe so close to equilibrium?

---

(sec-cosmology-intro)=
## Introduction

:::{div} feynman-prose
Let me tell you about one of the most confusing topics in all of physics: the cosmological constant. Or rather, the cosmological *constants*---because there are three of them, and they measure completely different things.

Now, you might think there is only one cosmological constant. That is what Einstein introduced, that is what drives the accelerated expansion of our universe, and that is what everyone argues about. But here is the thing that confused me for months until I finally understood it: when physicists talk about "the cosmological constant," they are often talking about three different physical quantities without realizing it.

This confusion has real consequences. People ask: "Is the cosmological constant positive or negative?" And the honest answer is: "Which one?" The holographic boundary? Always negative. The bulk at equilibrium? Zero. The effective bulk during expansion? Can be positive. These are not different guesses about the same number---they are measurements of different things.

In this chapter, we are going to untangle this mess. We will define all three vacuum energies precisely, understand why they are different physical quantities, and see how the Latent Fractal Gas framework naturally accommodates all of them. The resolution of apparent contradictions is not that someone made a mistake---it is that the question was ambiguous.

Here is the beautiful thing: once you understand this, the "cosmological constant problem" looks completely different. The question is not "Why is the cosmological constant so small?" It is "Which cosmological constant are you asking about, and what physical quantity does it measure?"
:::

(sec-three-lambda-problem)=
## The Three Lambda Problem

:::{div} feynman-prose
Ask yourself: what does the cosmological constant actually measure?

The standard textbook answer is "vacuum energy density." But vacuum energy density *where*? Measured *how*? These are not pedantic questions. They have different answers depending on what you mean.

Let me give you an analogy. Imagine a water droplet floating in space. You could ask about three different "pressures":

1. **Surface pressure:** The surface tension at the boundary of the droplet. This is a boundary effect, measured per unit area, and it pulls inward (negative pressure convention).

2. **Interior pressure:** The hydrostatic pressure inside the droplet. If the droplet is in equilibrium, this is constant throughout and balances the surface tension.

3. **Expansion pressure:** If the droplet is evaporating or growing, there is a dynamic pressure associated with the volume change.

These are all called "pressure," but they measure different physical quantities. You cannot add them up or compare them directly. The surface pressure has units of force per length; the interior pressure has units of force per area; the expansion pressure is a rate of change.

The cosmological constant situation is exactly analogous. We have three different Lambdas measuring three different things:

1. **Holographic boundary Lambda** ($\Lambda_{\mathrm{holo}}$): Vacuum energy at the boundary/horizon of a localized system. This is what the IG pressure measures.

2. **Bulk equilibrium Lambda** ($\Lambda_{\mathrm{bulk}}$): Vacuum energy in the interior when the system is at quasi-stationary equilibrium.

3. **Effective expansion Lambda** ($\Lambda_{\mathrm{eff}}$): The effective cosmological constant driving bulk dynamics when the system is *not* at equilibrium.

Let us define each one precisely.
:::

(sec-three-vacuum-energies)=
## The Three Vacuum Energies

### Holographic Boundary Vacuum

:::{prf:definition} Holographic Boundary Vacuum Energy
:label: def-holographic-boundary-vacuum

The **holographic boundary vacuum energy** $\Lambda_{\mathrm{holo}}$ is measured by the IG pressure at spatial horizons. For a localized system with characteristic length scale $L$ and horizon area $A_H$:

$$
\Lambda_{\mathrm{holo}} = \frac{8\pi G_N}{c^2}\frac{\Pi_{\mathrm{IG}}}{L}
$$

where the IG pressure from {prf:ref}`thm-holographic-pressure` is:

$$
\Pi_{\mathrm{IG}}(L) = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{8dL^2} < 0
$$

**Properties:**
1. **Always negative:** $\Lambda_{\mathrm{holo}} < 0$ for all $\varepsilon_c > 0$
2. **Boundary measurement:** Computed from jump Hamiltonian derivative with respect to horizon area
3. **AdS geometry:** Negative $\Lambda_{\mathrm{holo}}$ implies Anti-de Sitter boundary structure

**Physical interpretation:** The IG correlation network at the boundary acts as a surface tension, pulling inward. This is the vacuum structure *at the horizon*, not in the bulk.
:::

:::{div} feynman-prose
The holographic boundary vacuum is always negative. Let me make sure you understand why this is a mathematical theorem, not an approximation.

The IG correlation kernel $K_\varepsilon(z,z')$ is a Gaussian---positive everywhere. When you compute the jump Hamiltonian derivative with respect to horizon area, you are asking: how much does the correlation energy change when you stretch the boundary?

Stretching the boundary separates correlated pairs. Separated pairs have lower correlation (Gaussian decays with distance). Lower correlation means higher energy. Therefore, stretching costs energy. Therefore, the derivative is positive. Therefore, the pressure (negative derivative) is negative.

This argument holds for any positive correlation kernel, not just Gaussians. As long as correlations decay with distance, stretching costs energy, and the holographic pressure is negative. This is as rigorous as surface tension being positive for liquids---it follows from the attractive nature of the interaction.
:::

### Bulk QSD Vacuum

:::{prf:definition} Bulk QSD Vacuum Energy
:label: def-bulk-qsd-vacuum

The **bulk QSD vacuum energy** $\Lambda_{\mathrm{bulk}}$ is determined by the QSD equilibrium condition. At quasi-stationary equilibrium:

$$
\nabla_\mu T^{\mu\nu} = 0
$$

where $T^{\mu\nu}$ is the effective stress-energy tensor ({prf:ref}`def-effective-stress-energy`).

**Result:** For a spatially confined system at QSD equilibrium:

$$
\Lambda_{\mathrm{bulk}} = 0
$$

**Conditions for this result:**
1. **Spatial confinement:** Walkers restricted to bounded domain $\mathcal{X}$
2. **QSD equilibrium:** $\partial_t \rho = 0$ (density stationary)
3. **No bulk currents:** $J^\mu = 0$ (no net flow)
4. **Thermal balance:** Velocity distribution is local Maxwellian

**Physical interpretation:** At equilibrium, the bulk spacetime has no net expansion or contraction. The vacuum energy density is zero because there is no source driving dynamics.
:::

:::{prf:theorem} Vanishing Bulk Vacuum at QSD
:label: thm-vanishing-bulk-vacuum

When the Latent Fractal Gas reaches quasi-stationary equilibrium, the bulk cosmological constant vanishes:

$$
\Lambda_{\mathrm{bulk}}^{(\mathrm{QSD})} = 0
$$

*Proof.*

**Step 1. QSD implies stationarity.**

At QSD, the walker density satisfies $\partial_t \rho = 0$. The one-point statistics are time-independent.

**Step 2. Stress-energy conservation.**

The effective stress-energy tensor satisfies

$$
\nabla_\mu T^{\mu\nu} = J^\nu
$$

where $J^\nu$ is the source term from non-equilibrium effects.

**Step 3. Source vanishes at equilibrium.**

At QSD:
- **Thermal equilibrium:** $J^0 = 0$ (energy density is stationary)
- **Force balance:** $J^i = 0$ (no net momentum flow)

This follows from detailed balance of the reversible BAOAB diffusion kernel at QSD equilibrium. Away from equilibrium, the cloning step is dissipative and this argument does not apply.

**Step 4. Einstein equations with vanishing source.**

The field equations become

$$
G_{\mu\nu} + \Lambda_{\mathrm{bulk}} g_{\mu\nu} = \kappa T_{\mu\nu}
$$

with $\nabla_\mu T^{\mu\nu} = 0$.

For a spatially homogeneous QSD in a confined domain, the Einstein tensor satisfies $G_{\mu\nu} = \kappa T_{\mu\nu}$ with $\Lambda_{\mathrm{bulk}} = 0$.

$\square$
:::

:::{div} feynman-prose
The vanishing of the bulk vacuum at equilibrium is not surprising once you think about it. The cosmological constant drives expansion or contraction. If the system is at equilibrium---not expanding, not contracting---then the effective Lambda must be zero.

But notice the key assumption: the system must be *at equilibrium*. Our universe is manifestly not at equilibrium---it is expanding. So this theorem does not contradict observations. It just tells us that cosmological expansion requires non-equilibrium dynamics.
:::

### Effective Exploration Vacuum

:::{prf:definition} Effective Exploration Vacuum Energy
:label: def-effective-exploration-vacuum

The **effective exploration vacuum energy** $\Lambda_{\mathrm{eff}}$ arises from non-equilibrium bulk dynamics. When the system is not at QSD:

$$
\Lambda_{\mathrm{eff}} = \Lambda_{\mathrm{bulk}}^{(\mathrm{QSD})} + \Lambda_{\mathrm{exploration}} = 0 + \Lambda_{\mathrm{exploration}}
$$

where $\Lambda_{\mathrm{exploration}}$ is determined by the source term $J^\mu \neq 0$ in the modified field equations:

$$
G_{\mu\nu} + \Lambda_{\mathrm{eff}} g_{\mu\nu} = \kappa T_{\mu\nu} + \kappa (J_\mu u_\nu + J_\nu u_\mu)
$$

where the exploration current $J^\mu$ couples symmetrically to the 4-velocity $u_\nu$ to form a proper rank-2 tensor.

**Properties:**
1. **Sign depends on dynamics:** $\Lambda_{\mathrm{eff}}$ can be positive, negative, or zero
2. **Exploration-dominated:** When walkers are spreading (exploration phase), $\Lambda_{\mathrm{eff}} > 0$ is possible
3. **Exploitation-dominated:** When walkers are converging (exploitation phase), $\Lambda_{\mathrm{eff}} \leq 0$

**Physical interpretation:** The effective cosmological constant measures how far the system is from equilibrium. Positive $\Lambda_{\mathrm{eff}}$ corresponds to expansion-driving dynamics.
:::

:::{div} feynman-prose
This is where things get interesting. The effective Lambda is not a fixed constant---it is a dynamical quantity that depends on the state of the system. During exploration phases, walkers spread out, creating defocusing geometry and positive effective vacuum energy. During exploitation phases, walkers converge on fitness peaks, creating focusing geometry and non-positive effective vacuum energy.

Think about it this way. The cosmological constant in our universe is measured to be positive---about $10^{-52}$ m$^{-2}$. This means the universe is *not* at equilibrium. It is still exploring, still spreading out. The positive Lambda is a signature of this ongoing exploration.

If the universe ever reached QSD equilibrium---walkers clustered on the fitness peaks, no net expansion or contraction---the effective Lambda would be zero. But that has not happened yet. The universe is still far from equilibrium, and the observed positive Lambda tells us how far.
:::

(sec-why-different)=
## Why They Are Different Physical Quantities

:::{div} feynman-prose
Now let me explain why these three Lambdas are genuinely different, not just different names for the same thing.

The key is geometry. The holographic Lambda is measured at the *boundary*---it is a surface quantity. The bulk Lambdas are measured in the *interior*---they are volume quantities. These are mathematically distinct operations, like taking a surface integral versus a volume integral.
:::

:::{prf:proposition} Geometric Distinction: Boundary vs Bulk
:label: prop-geometric-distinction

The three vacuum energies correspond to geometrically distinct measurements:

**Holographic $\Lambda_{\mathrm{holo}}$:** Boundary integral measurement

$$
\Lambda_{\mathrm{holo}} \sim \frac{1}{A_H} \frac{\partial}{\partial A_H} \iint_{H \times \mathcal{Z}} K_\varepsilon(z,z') \rho(z) \rho(z') \, dz \, dz'
$$

This is a double integral with one point on the horizon $H$ and one in the bulk.

**Bulk $\Lambda_{\mathrm{bulk}}$:** Volume integral measurement

$$
\Lambda_{\mathrm{bulk}} \sim \frac{1}{V} \int_{\mathcal{X}} (\text{vacuum energy density}) \, dV
$$

This is a single integral over the entire bulk volume.

**Effective $\Lambda_{\mathrm{eff}}$:** Dynamical measurement

From the $(d+1)$-dimensional Friedmann equations (rearranged to solve for $\Lambda$ in a flat, matter-dominated universe):

$$
\Lambda_{\mathrm{eff}} = \frac{d(d-1)}{2} H^2 - (d-1)\frac{\ddot{a}}{a} - \frac{8\pi G_N}{d-1} \rho_{\mathrm{matter}}
$$
where $H = \dot{a}/a$ is the Hubble parameter, $\ddot{a}/a$ is the deceleration term, and $d$ is the spatial dimension.

This is computed from the expansion rate and acceleration.

**Key point:** These are distinct geometric operations that can have different values simultaneously.
:::

:::{admonition} Table: Boundary vs Bulk Comparison
:class: feynman-added note

| Property | Holographic (Boundary) | Bulk (Volume) |
|----------|------------------------|---------------|
| **Dimension** | $(d-1)$-dimensional horizon | $d$-dimensional volume |
| **Data source** | IG correlations across horizon | Stress-energy tensor in bulk |
| **Measurement** | Jump Hamiltonian derivative | Field equation solution |
| **Sign at QSD** | Always negative | Zero |
| **Physical analog** | Surface tension of droplet | Bulk pressure in equilibrium |
| **AdS/CFT role** | Boundary CFT vacuum | Bulk gravity vacuum |
:::

:::{div} feynman-prose
The liquid droplet analogy is worth developing further.

In a water droplet, surface tension acts at the boundary and has nothing to do with the internal pressure. You can have high surface tension (strongly curved boundary) with zero internal pressure gradient (hydrostatic equilibrium). Or you can have low surface tension but high internal pressure (a bubble about to burst).

The IG correlation network is the surface tension of spacetime. It creates negative pressure at boundaries because stretching correlations costs energy. This has nothing to do with what happens in the bulk.

The bulk pressure depends on the dynamics *inside*. At equilibrium, there is no net force, and the effective cosmological constant is zero. Out of equilibrium, there can be net expansion or contraction, and the effective Lambda is nonzero.

The mistake people make is assuming that the boundary pressure determines the bulk dynamics. It does not. They are coupled---what happens at the boundary affects what happens in the bulk---but they are not the same thing.
:::

(sec-uv-regime)=
## UV Regime: Short-Range Correlations

:::{div} feynman-prose
Now let us analyze what happens in different regimes, starting with the UV (short correlation length) limit. This is where the holographic results are cleanest and AdS geometry emerges most clearly.
:::

:::{prf:definition} UV Regime
:label: def-uv-regime

The **UV regime** is characterized by:

$$
\varepsilon_c \ll L
$$

where $\varepsilon_c$ is the IG correlation length and $L$ is the system size.

**Physical characteristics:**
- Correlations are short-range (decay quickly with distance)
- Elastic pressure dominates radiation pressure
- Frequency gap $\omega_0 \gg k_B T_{\mathrm{eff}}$
- Mode occupation is exponentially suppressed

**Consequence:** From {prf:ref}`thm-pressure-regimes`, elastic pressure dominates and total pressure is negative.
:::

:::{prf:theorem} AdS Boundary in UV Regime
:label: thm-ads-boundary-uv

In the UV regime ($\varepsilon_c \ll L$), the holographic boundary geometry is always Anti-de Sitter:

$$
\Lambda_{\mathrm{holo}} = \frac{8\pi G_N}{c^2}\frac{\Pi_{\mathrm{IG}}}{L} < 0 \quad \forall \varepsilon_c > 0
$$

**Explicit formula:**

$$
\Lambda_{\mathrm{holo}} = -\frac{\pi G_N C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{d c^2 L^3}
$$

**Properties:**
1. **Universal negativity:** Holds for all positive $\varepsilon_c$, not just UV regime
2. **Scaling:** $|\Lambda_{\mathrm{holo}}| \propto \varepsilon_c^{d+2} / L^3$
3. **AdS radius:** $L_{\mathrm{AdS}}^2 = -d(d-1)/(2\Lambda_{\mathrm{holo}}) > 0$

*Proof.*

This follows directly from {prf:ref}`thm-holographic-pressure`. The IG pressure is

$$
\Pi_{\mathrm{IG}}(L) = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{8dL^2} < 0.
$$

The holographic Lambda is

$$
\Lambda_{\mathrm{holo}} = \frac{8\pi G_N}{c^2}\frac{\Pi_{\mathrm{IG}}}{L} = -\frac{\pi G_N C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{d c^2 L^3} < 0.
$$

$\square$
:::

:::{div} feynman-prose
This theorem is remarkable in its universality. No matter what the correlation length is, no matter what the system size is, the holographic boundary is always AdS. There is no parameter regime where you get de Sitter boundary geometry from the IG pressure calculation.

This makes physical sense. The IG is an attractive correlation network---walkers that are correlated tend to have similar properties. Attractive networks create surface tension, and surface tension creates negative pressure. You cannot get positive pressure from an attractive network.

The connection to AdS/CFT is now clear. The AdS/CFT correspondence was discovered in string theory, where Anti-de Sitter bulk geometry emerges from conformal field theory boundary dynamics. Here we are deriving the same structure from optimization: the IG correlation network plays the role of the CFT, and its attractive nature guarantees AdS boundary geometry.
:::

(sec-qsd-vs-exploration)=
## QSD Equilibrium vs Exploration Phase

:::{div} feynman-prose
Now we come to the critical distinction: equilibrium versus non-equilibrium dynamics. The quasi-stationary distribution (QSD) represents equilibrium---walkers have settled into their long-time statistical behavior. But the universe is not at equilibrium. It is still evolving, still expanding, still exploring.

Let me define both regimes precisely and show how they lead to different effective Lambdas.
:::

:::{prf:definition} QSD Regime (Exploitation)
:label: def-qsd-regime

The **QSD regime** is characterized by:

$$
\partial_t \rho = 0, \quad J^\mu = 0
$$

**Physical characteristics:**
- Walker density is stationary
- No net bulk currents
- Walkers clustered on fitness peaks
- Exploitation dominates exploration
- Positive Ricci curvature (geodesic focusing toward peaks)

**Geometric consequences:**
- Bulk vacuum: $\Lambda_{\mathrm{bulk}} = 0$
- Raychaudhuri focusing: $R_{\mu\nu}u^\mu u^\nu > 0$
- Expansion: $\theta \to 0$ (no net expansion)

**Physical examples:**
- Optimization algorithm at convergence
- Galaxy clusters (virially relaxed)
- Black hole interior at equilibrium
:::

:::{prf:definition} Exploration Regime
:label: def-exploration-regime

The **exploration regime** is characterized by:

$$
\partial_t \rho \neq 0, \quad J^\mu \neq 0
$$

**Physical characteristics:**
- Walker density is evolving
- Net bulk currents present
- Walkers spreading uniformly
- Exploration dominates exploitation
- Negative or zero Ricci curvature (geodesic defocusing)

**Geometric consequences:**
- Effective vacuum: $\Lambda_{\mathrm{eff}} > 0$ possible
- Raychaudhuri defocusing: $R_{\mu\nu}u^\mu u^\nu \leq 0$
- Expansion: $\theta > 0$ (volume growth)

**Physical examples:**
- Early-universe inflation
- Dark energy era
- Monte Carlo exploration phase
:::

:::{admonition} Table: QSD vs Exploration Regimes
:class: feynman-added note

| Property | QSD (Exploitation) | Exploration |
|----------|-------------------|-------------|
| **Density** | $\partial_t \rho = 0$ | $\partial_t \rho \neq 0$ |
| **Source term** | $J^\mu = 0$ | $J^\mu \neq 0$ |
| **Walker behavior** | Clustered on peaks | Spreading uniformly |
| **Ricci curvature** | $R_{\mu\nu}u^\mu u^\nu > 0$ (focusing) | $R_{\mu\nu}u^\mu u^\nu \leq 0$ (defocusing) |
| **Bulk Lambda** | $\Lambda_{\mathrm{bulk}} = 0$ | $\Lambda_{\mathrm{eff}} > 0$ possible |
| **Expansion** | $\theta \to 0$ | $\theta > 0$ sustained |
| **Geometry** | Static or AdS near boundaries | Expanding (FRW or dS) |
:::

(sec-raychaudhuri-expansion)=
## The Raychaudhuri Equation and Cosmic Expansion

:::{div} feynman-prose
The Raychaudhuri equation from {prf:ref}`thm-raychaudhuri-scutoid` tells us how bundles of geodesics evolve. In the context of cosmology, these geodesics are the worldlines of comoving observers (or walkers), and the expansion scalar $\theta$ is essentially the Hubble parameter.

Let me show you how exploration dynamics can drive sustained expansion.
:::

:::{prf:theorem} Exploration-Driven Expansion
:label: thm-exploration-expansion

During the exploration-dominated regime, the Raychaudhuri equation allows sustained positive expansion:

$$
\frac{d\theta}{d\tau} = -\frac{1}{d}\theta^2 - \sigma_{\mu\nu}\sigma^{\mu\nu} + \omega_{\mu\nu}\omega^{\mu\nu} - R_{\mu\nu}u^\mu u^\nu
$$

**Exploration conditions:**
1. **Defocusing curvature:** $R_{\mu\nu}u^\mu u^\nu < 0$ (walkers spreading, not focusing)
2. **Low shear:** $\sigma_{\mu\nu}\sigma^{\mu\nu} \approx 0$ (isotropic expansion)
3. **Vorticity permitted:** $\omega_{\mu\nu}\omega^{\mu\nu} \geq 0$

**Result:** With $R_{\mu\nu}u^\mu u^\nu < 0$, the equation becomes

$$
\frac{d\theta}{d\tau} = -\frac{1}{d}\theta^2 + |R_{\mu\nu}u^\mu u^\nu| + \omega_{\mu\nu}\omega^{\mu\nu}
$$

For sufficiently negative Ricci curvature, $\theta > 0$ can be sustained or even grow.

*Proof sketch.*

The key term is $-R_{\mu\nu}u^\mu u^\nu$. In the exploitation (QSD) regime, walkers focus on fitness peaks, creating positive Ricci curvature along geodesics. With $\sigma_{\mu\nu}\sigma^{\mu\nu} \geq 0$ and $R_{\mu\nu}u^\mu u^\nu > 0$, this causes $d\theta/d\tau < 0$ (focusing).

In the exploration regime, walkers spread uniformly on a flat or saddle-like fitness landscape. The effective Ricci curvature becomes negative (defocusing). Now $-R_{\mu\nu}u^\mu u^\nu > 0$, which can overcome the $-\theta^2/d$ term and drive $\theta > 0$.

The expansion is sustained as long as the exploration phase continues.

$\square$
:::

:::{prf:theorem} Bulk Can Be de Sitter During Exploration
:label: thm-bulk-can-be-ds

In the exploration-dominated regime, the bulk effective cosmological constant can be positive:

$$
\Lambda_{\mathrm{eff}} > 0
$$

leading to de Sitter-like expanding geometry.

**Mechanism:**
1. Walkers undergo volumetric spreading (exploration)
2. Defocusing geometry creates negative Ricci curvature along worldlines
3. Modified field equations with source term $J^\mu \neq 0$
4. Effective positive vacuum energy drives expansion

**Quantitative relation:**

$$
\Lambda_{\mathrm{eff}} \propto -R_{\mu\nu}u^\mu u^\nu \cdot L^2
$$

where $R_{\mu\nu}u^\mu u^\nu < 0$ during exploration.

**Status:** Mechanism established qualitatively. Quantitative calculation requires solving non-equilibrium McKean-Vlasov PDE.
:::

:::{div} feynman-prose
Here is the physical picture. Imagine the walkers are exploring a very flat fitness landscape---no peaks to focus on, no valleys to avoid. What do they do? They spread out uniformly, because there is no reason to go anywhere in particular.

This uniform spreading creates defocusing geometry. Nearby geodesics diverge rather than converge. The Ricci curvature along worldlines is negative. And negative Ricci curvature drives expansion through the Raychaudhuri equation.

This is exactly what "dark energy" looks like. A universe that is not focused on any particular attractor, still searching, still spreading out. The positive cosmological constant measures how far we are from finding the fitness peak.

The beautiful thing is that this does not contradict the holographic result. The boundary is still AdS (negative Lambda from IG pressure). But the bulk can be expanding (positive effective Lambda from exploration dynamics). Boundary and bulk are different places, and different things can happen there.
:::

(sec-closure-theory)=
## Closure Theory and Computational Coarse-Graining

:::{div} feynman-prose
Now let me connect this to a deeper mathematical structure: closure theory. This is the information-theoretic framework for understanding when coarse-grained descriptions preserve predictive power.

The question is: when we go from microscopic walker dynamics to macroscopic cosmological evolution, do we lose information that matters? Or is the coarse-graining "closed" in the sense that macro predicts macro as well as micro does?
:::

:::{prf:definition} Epsilon-Machine
:label: def-epsilon-machine

An **epsilon-machine** is the minimal sufficient statistic for prediction. Given a stochastic process $\{X_t\}$:

**Causal equivalence:** Two pasts $\overleftarrow{x}$ and $\overleftarrow{x}'$ are causally equivalent if they induce identical conditional distributions over futures:

$$
P(\overrightarrow{X} \mid \overleftarrow{X} = \overleftarrow{x}) = P(\overrightarrow{X} \mid \overleftarrow{X} = \overleftarrow{x}')
$$

**Causal state:** The equivalence class $[\overleftarrow{x}]_\varepsilon$ is a causal state $\sigma \in \Sigma_\varepsilon$.

**Epsilon-machine:** The pair $(\Sigma_\varepsilon, T_\varepsilon)$ where $\Sigma_\varepsilon$ is the set of causal states and $T_\varepsilon$ is the transition function.

**Optimality:** The epsilon-machine achieves optimal prediction with minimal state complexity.
:::

:::{prf:definition} Information Closure
:label: def-information-closure-cosmo

A coarse-graining $f: X \to Y$ satisfies **information closure** if the macroscopic process predicts itself as well from macro-data as from micro-data:

$$
I(\overrightarrow{Y}_t ; \overleftarrow{Y}_t) = I(\overrightarrow{Y}_t ; \overleftarrow{X}_t)
$$

**Interpretation:** All micro-information relevant to macro-futures is captured by macro-pasts.
:::

:::{prf:definition} Computational Closure
:label: def-computational-closure-cosmo

A coarse-graining $f: X \to Y$ satisfies **computational closure** if the macro epsilon-machine is a coarse-graining of the micro epsilon-machine.

**Formal condition:** There exists a projection $\pi: \Sigma_\varepsilon^{(X)} \to \Sigma_\varepsilon^{(Y)}$ such that:

$$
\pi([\overleftarrow{x}]_\varepsilon^{(X)}) = [f(\overleftarrow{x})]_\varepsilon^{(Y)}
$$

**Interpretation:** Macro causal states are aggregations of micro causal states.
:::

:::{prf:definition} Causal Closure
:label: def-causal-closure-cosmo

A coarse-graining $f: X \to Y$ satisfies **causal closure** if causal equivalence at the micro level is preserved at the macro level:

$$
[\overleftarrow{x}]_\varepsilon^{(X)} = [\overleftarrow{x}']_\varepsilon^{(X)} \implies [f(\overleftarrow{x})]_\varepsilon^{(Y)} = [f(\overleftarrow{x}')]_\varepsilon^{(Y)}
$$

**Interpretation:** Pasts that are indistinguishable for predicting micro-futures remain indistinguishable for predicting macro-futures. The causal structure is preserved under coarse-graining.
:::

:::{prf:theorem} Closure Equivalence
:label: thm-closure-equivalence-cosmo

For any coarse-graining:

$$
\text{Information Closure} \iff \text{Causal Closure}
$$

Furthermore, for spatial coarse-grainings (aggregating spatially local variables):

$$
\text{Information Closure} \implies \text{Computational Closure}
$$

*Proof sketch.*

**Equivalence of Information and Causal Closure.**

($\Rightarrow$) If information closure holds, then $I(\overrightarrow{Y}_t ; \overleftarrow{Y}_t) = I(\overrightarrow{Y}_t ; \overleftarrow{X}_t)$. This means all predictive information about macro-futures is contained in macro-pasts. Therefore, micro-pasts that differ only in ways irrelevant to macro-futures map to the same macro causal state, establishing causal closure.

($\Leftarrow$) If causal closure holds, micro-pasts with identical macro-projections induce identical macro-future distributions. By the data processing inequality, no additional predictive information is lost when passing from $\overleftarrow{X}_t$ to $\overleftarrow{Y}_t$, establishing information closure.

**Spatial coarse-grainings imply computational closure.**

For spatially local aggregations, the epsilon-machine structure respects locality: if micro causal states $\sigma_1, \sigma_2$ project to the same macro state under $f$, their transition probabilities also project consistently. This follows from the Markov property of spatially local dynamics.

$\square$

**Application to cosmology:** The renormalization group flow that takes us from micro (walker dynamics) to macro (Friedmann equations) is an instance of computational closure when it preserves physical predictions.
:::

:::{div} feynman-prose
Here is why closure theory matters for cosmology. When we write down the Friedmann equations for the expansion of the universe, we are making an enormous coarse-graining. We are going from $10^{80}$ particles to a few numbers: scale factor, Hubble parameter, density, pressure.

The question is: does this coarse-graining preserve predictive power? Can we predict macro-futures (will the universe expand forever?) from macro-pasts (what is the current expansion rate?) without needing micro-details?

Closure theory says: yes, if and only if the coarse-graining satisfies information closure. And for spatial coarse-grainings like the homogeneous approximation, information closure implies computational closure.

This is a rigorous justification for using effective field theory in cosmology. The reason the Friedmann equations work is that homogeneous coarse-graining satisfies closure. The macro dynamics are self-contained.

But here is the subtle point: closure can fail. Near phase transitions, near critical points, near singularities---the macro description may lose information that matters for macro-futures. When closure fails, the effective description breaks down, and you need to go back to micro-dynamics.

This might be relevant for the very early universe (inflation) or the very late universe (heat death). The closure theory framework gives us rigorous criteria for when our effective cosmological descriptions are trustworthy.
:::

(sec-cosmological-observations)=
## Connection to Cosmological Observations

:::{div} feynman-prose
Now let us connect this framework to what we actually observe in our universe. The key observation is the accelerated expansion, discovered in 1998 through supernova observations.
:::

:::{prf:proposition} Observed Cosmological Constant
:label: prop-observed-lambda

The observed cosmological constant is

$$
\Lambda_{\mathrm{obs}} \approx 1.1 \times 10^{-52} \, \text{m}^{-2} > 0.
$$

**This is a bulk measurement**, not a boundary measurement. It is extracted from:
1. Supernova luminosity distances (Riess 1998, Perlmutter 1999)
2. CMB angular power spectrum (Planck 2018)
3. Large-scale structure growth suppression

**Interpretation in Latent Fractal Gas framework:**
- $\Lambda_{\mathrm{obs}} = \Lambda_{\mathrm{eff}} > 0$ (effective bulk vacuum during exploration)
- Universe is NOT at QSD equilibrium
- Universe is in exploration-dominated phase
- Positive Lambda = residual exploration pressure
:::

:::{prf:proposition} Dark Energy as Exploration Pressure
:label: prop-dark-energy-exploration

Dark energy is the bulk manifestation of exploration dynamics:

$$
\rho_{\mathrm{DE}} = \frac{\Lambda_{\mathrm{eff}} c^2}{8\pi G_N}.
$$

**Physical interpretation:**
- Dark energy density measures "distance from QSD"
- Larger $\rho_{\mathrm{DE}}$ means farther from equilibrium
- As universe approaches QSD: $\Lambda_{\mathrm{eff}} \to 0$
- Heat death = QSD equilibrium = no more exploration

**Equation of state:**

$$
w = \frac{P}{\rho} = -1 + \mathcal{O}\left(\frac{1}{\Lambda_{\mathrm{eff}} L^2}\right).
$$

The $w \approx -1$ equation of state emerges from the exploration dynamics, not from vacuum energy in the traditional sense.
:::

:::{admonition} Table: Three Lambdas and Observations
:class: feynman-added note

| Lambda | Value | Measurement | Physical Meaning |
|--------|-------|-------------|------------------|
| $\Lambda_{\mathrm{holo}}$ | $< 0$ | IG pressure at horizons | Boundary surface tension |
| $\Lambda_{\mathrm{bulk}}^{(\mathrm{QSD})}$ | $= 0$ | Equilibrium field equations | No expansion at QSD |
| $\Lambda_{\mathrm{eff}}$ | $\approx 10^{-52}$ m$^{-2}$ | Supernova, CMB | Exploration-driven expansion |
:::

:::{div} feynman-prose
The observed positive cosmological constant fits perfectly into this framework. It is not the holographic Lambda (that is always negative). It is not the equilibrium bulk Lambda (that is zero). It is the effective Lambda from exploration dynamics.

The universe is exploring. It has not found the fitness peak yet. The positive Lambda tells us how actively it is searching.

This reframes the "cosmological constant problem." The traditional problem is: why is $\Lambda_{\mathrm{obs}}$ so small compared to particle physics predictions? But that comparison is confused. The particle physics predictions are computing something like $\Lambda_{\mathrm{holo}}$ (vacuum energy at short distances, boundary-type calculation). The observations are measuring $\Lambda_{\mathrm{eff}}$ (bulk expansion rate). These are different quantities!

The "smallness" of $\Lambda_{\mathrm{obs}}$ is not a fine-tuning problem. It is telling us that the universe is *close* to QSD equilibrium, but not quite there. The exploration pressure is weak, but nonzero. This is consistent with a universe that has been evolving for 13.8 billion years---long enough to settle down significantly, but not long enough to reach equilibrium.
:::

(sec-de-sitter-conjecture)=
## Resolution of the de Sitter Conjecture

:::{div} feynman-prose
There is a famous debate in string theory called the "de Sitter conjecture"---whether de Sitter space (positive cosmological constant) can arise from consistent quantum gravity. Let me show you how our framework resolves this.
:::

:::{prf:theorem} Resolution of de Sitter Question
:label: thm-de-sitter-resolution

The apparent tension between "AdS from holography" and "dS from observations" is resolved by recognizing they measure different quantities:

**Holographic boundary:** Always AdS

$$
\Lambda_{\mathrm{holo}} < 0 \quad \text{(proven rigorously)}
$$

**Bulk at QSD:** Zero

$$
\Lambda_{\mathrm{bulk}}^{(\mathrm{QSD})} = 0 \quad \text{(proven rigorously)}
$$

**Bulk during exploration:** Can be positive

$$
\Lambda_{\mathrm{eff}} > 0 \quad \text{(mechanism established)}
$$

**Status summary:**
- AdS boundary: PROVEN ({prf:ref}`thm-ads-boundary-uv`)
- Zero bulk at equilibrium: PROVEN ({prf:ref}`thm-vanishing-bulk-vacuum`)
- Positive effective bulk during exploration: MECHANISM ESTABLISHED ({prf:ref}`thm-bulk-can-be-ds`), quantitative calculation pending

**Resolution:** The de Sitter conjecture was asking about boundary vacuum (where AdS is proven). Observations measure bulk vacuum during non-equilibrium (where dS is possible). No contradiction.
:::

:::{div} feynman-prose
The de Sitter conjecture debate was asking the wrong question. Or rather, asking a question without specifying which Lambda.

The holographic calculations in string theory (and our IG pressure calculation) give negative Lambda at boundaries. This is correct and robust. The cosmological observations give positive Lambda in the bulk. This is also correct. They are measuring different things.

The confusion arose because people assumed there was only one cosmological constant. Once you recognize there are three, the apparent contradiction dissolves.

Let me be precise about what is proven and what is not:
- **Proven:** Holographic boundary is AdS ($\Lambda_{\mathrm{holo}} < 0$)
- **Proven:** Bulk at equilibrium has zero vacuum energy ($\Lambda_{\mathrm{bulk}} = 0$)
- **Established mechanism:** Bulk during exploration can have positive effective vacuum energy ($\Lambda_{\mathrm{eff}} > 0$)
- **Not yet calculated:** Quantitative value of $\Lambda_{\mathrm{eff}}$ from exploration parameters

The third point is not a rigorous theorem yet. We know the mechanism (defocusing geometry from exploration), but computing the exact value of $\Lambda_{\mathrm{eff}}$ requires solving the non-equilibrium McKean-Vlasov equations, which is technically challenging.
:::

(sec-physical-interpretation)=
## Physical Interpretation

:::{div} feynman-prose
Let me step back and tell you what all this means. The vacuum is not a fixed backdrop. It is an emergent attractor of search dynamics.
:::

:::{prf:remark} Vacuum as Algorithmic Attractor
:label: rem-vacuum-algorithmic

The vacuum state of spacetime is not fundamental but emergent:

**Vacuum = QSD attractor**

The "vacuum" is the quasi-stationary distribution that the swarm approaches over long times. Different fitness landscapes lead to different QSD attractors, hence different "vacua."

**Cosmological constant = distance from equilibrium**

$\Lambda_{\mathrm{eff}}$ measures how far the system is from its QSD attractor. Positive Lambda means ongoing exploration; zero Lambda means equilibrium reached.

**Dark energy = residual exploration pressure**

The observed dark energy is the dynamical signature of a universe that has not yet found its fitness peak.
:::

:::{prf:remark} Exploration vs Exploitation as Cosmic Dynamics
:label: rem-exploration-exploitation-cosmic

The exploration-exploitation tradeoff has cosmic manifestations:

**Exploration-dominated (early universe)**
- Inflation: Rapid exploration of vacuum structure
- $\Lambda_{\mathrm{eff}} \gg 0$: Strong expansion-driving pressure
- Defocusing geometry: Walkers spread exponentially
- Result: Flat, homogeneous universe

**Exploitation-dominated (late universe)**
- Structure formation: Walkers cluster on fitness peaks
- $\Lambda_{\mathrm{eff}} \to 0$: Weak expansion pressure
- Focusing geometry: Matter clusters into galaxies
- Result: Hierarchical structure

**Current era**
- Mixture: Both exploration (cosmic expansion) and exploitation (structure formation)
- $\Lambda_{\mathrm{eff}} \approx 10^{-52}$ m$^{-2}$: Small but positive
- Competition: Dark energy vs gravitational collapse
- Result: Accelerating expansion with galaxy-scale clustering
:::

:::{prf:remark} Cosmological Constant Problem Reframed
:label: rem-cc-problem-reframed

The traditional "cosmological constant problem" asks:

> Why is $\Lambda$ so much smaller than particle physics predicts?

The Latent Fractal Gas framework reframes this as:

> Why is the universe so close to QSD equilibrium?

**Traditional problem:** $\Lambda_{\mathrm{obs}} / \Lambda_{\mathrm{QFT}} \sim 10^{-120}$ (unnatural fine-tuning)

**Reframed question:** Why has the universe evolved so close to equilibrium in 13.8 billion years?

**Possible answer:** Selection effects. Universes far from equilibrium (large $\Lambda_{\mathrm{eff}}$) expand too fast for structure. Universes at equilibrium ($\Lambda_{\mathrm{eff}} = 0$) might collapse or freeze. Observers exist in the "Goldilocks" zone of exploration---close enough to equilibrium for structure, far enough for expansion.

This is not a complete solution, but it shows how the problem looks different in this framework.
:::

:::{prf:remark} Multiverse as Different QSD Attractors
:label: rem-multiverse-qsd

Different "vacua" in the string landscape may correspond to different QSD attractors:

**String landscape interpretation:**
- Each vacuum = different fitness landscape
- Each cosmological constant = distance from that landscape's QSD
- Transitions between vacua = transitions between QSD attractors

**Observable implications:**
- Our vacuum is one QSD attractor among many
- Anthropic selection favors vacua with suitable $\Lambda_{\mathrm{eff}}$
- Bubble nucleation = walker "tunneling" to different attractor

**Status:** Speculative interpretation, not proven from the framework.
:::

(sec-conclusions-cosmology)=
## Conclusions

:::{div} feynman-prose
Let me summarize what we have accomplished in this chapter.

We started with a confusion: the cosmological constant seems to be negative (from holographic calculations), zero (from equilibrium arguments), and positive (from observations). How can one number have three values?

The resolution is that these are three different numbers measuring three different things:

1. **Holographic boundary Lambda** ($\Lambda_{\mathrm{holo}} < 0$): Surface tension at horizons, always negative, creates AdS boundary geometry.

2. **Bulk equilibrium Lambda** ($\Lambda_{\mathrm{bulk}} = 0$): Vacuum energy in the interior at QSD, zero because there is no expansion at equilibrium.

3. **Effective exploration Lambda** ($\Lambda_{\mathrm{eff}} > 0$ possible): Dynamical vacuum during non-equilibrium, positive when exploration drives expansion.

The key theorems are:
- **{prf:ref}`thm-ads-boundary-uv`:** Holographic boundary is always AdS (proven)
- **{prf:ref}`thm-vanishing-bulk-vacuum`:** Bulk vacuum is zero at QSD (proven)
- **{prf:ref}`thm-bulk-can-be-ds`:** Bulk can be dS during exploration (mechanism established)

The cosmological implications are profound:
- **Dark energy** is not mysterious vacuum energy---it is exploration pressure from a universe still searching for equilibrium
- **The cosmological constant problem** is reframed as a question about distance from equilibrium
- **AdS/CFT** is confirmed at boundaries; observations measure bulk non-equilibrium

The closure theory connection shows that our effective cosmological descriptions (Friedmann equations) are valid when computational closure is satisfied---macro predicts macro as well as micro does.

What remains open is the quantitative calculation of $\Lambda_{\mathrm{eff}}$ from exploration parameters. This requires solving non-equilibrium dynamics, which is technically challenging. But the mechanism is clear, and the qualitative predictions match observations.

The deep insight is this: the vacuum is algorithmic. It is not a fixed backdrop but an emergent attractor of optimization dynamics. Different fitness landscapes produce different vacua, and the cosmological constant measures how far we are from finding the attractor.

The universe is searching. It has not found the answer yet. And the tiny positive cosmological constant is the signature of that ongoing search.
:::

:::{admonition} Key Takeaways
:class: tip

**The Three Cosmological Constants:**

| Lambda | Formula | Sign | Physical Meaning |
|--------|---------|------|------------------|
| $\Lambda_{\mathrm{holo}}$ | $\frac{8\pi G_N}{c^2}\frac{\Pi_{\mathrm{IG}}}{L}$ | $< 0$ always | Boundary surface tension |
| $\Lambda_{\mathrm{bulk}}^{(\mathrm{QSD})}$ | From $\nabla_\mu T^{\mu\nu} = 0$ | $= 0$ | Equilibrium vacuum |
| $\Lambda_{\mathrm{eff}}$ | From non-equilibrium $J^\mu \neq 0$ | Can be $> 0$ | Exploration pressure |

**Regime Summary:**

| Regime | Walker Behavior | Curvature | Bulk Lambda | Geometry |
|--------|-----------------|-----------|-------------|----------|
| QSD | Clustered on peaks | $R_{\mu\nu}u^\mu u^\nu > 0$ | $= 0$ | Static/AdS boundary |
| Exploration | Spreading uniformly | $R_{\mu\nu}u^\mu u^\nu < 0$ | $> 0$ possible | Expanding (FRW/dS) |

**Key Theorems:**

1. **AdS Boundary** ({prf:ref}`thm-ads-boundary-uv`): $\Lambda_{\mathrm{holo}} < 0$ always (PROVEN)
2. **Zero Bulk at QSD** ({prf:ref}`thm-vanishing-bulk-vacuum`): $\Lambda_{\mathrm{bulk}} = 0$ (PROVEN)
3. **Positive Bulk during Exploration** ({prf:ref}`thm-bulk-can-be-ds`): $\Lambda_{\mathrm{eff}} > 0$ (MECHANISM ESTABLISHED)
4. **Closure Equivalence** ({prf:ref}`thm-closure-equivalence-cosmo`): Information closure $\iff$ Causal closure

**Physical Interpretations:**

- **Dark energy** = Residual exploration pressure
- **Cosmological constant problem** = Why so close to equilibrium?
- **De Sitter conjecture** = Resolved (boundary vs bulk distinction)
- **Multiverse** = Different QSD attractors
:::

(sec-symbols-cosmology)=
## Table of Symbols

| Symbol | Definition | Reference |
|--------|------------|-----------|
| $\Lambda_{\mathrm{holo}}$ | Holographic boundary vacuum energy | {prf:ref}`def-holographic-boundary-vacuum` |
| $\Lambda_{\mathrm{bulk}}$ | Bulk QSD vacuum energy | {prf:ref}`def-bulk-qsd-vacuum` |
| $\Lambda_{\mathrm{eff}}$ | Effective exploration vacuum energy | {prf:ref}`def-effective-exploration-vacuum` |
| $\Pi_{\mathrm{IG}}$ | IG pressure at horizon | {prf:ref}`thm-holographic-pressure` |
| $\varepsilon_c$ | IG correlation length | {prf:ref}`def-ig-structure` |
| $\rho_0$ | Uniform walker density | {prf:ref}`thm-elastic-pressure` |
| $J^\mu$ | Non-equilibrium source term | {prf:ref}`def-effective-exploration-vacuum` |
| $\theta$ | Expansion scalar | {prf:ref}`def-kinematic-decomposition` |
| $R_{\mu\nu}$ | Ricci tensor | {prf:ref}`def-ricci-tensor-scalar` |
| $\sigma_{\mu\nu}$ | Shear tensor | {prf:ref}`def-kinematic-decomposition` |
| $\omega_{\mu\nu}$ | Vorticity tensor | {prf:ref}`def-kinematic-decomposition` |
| $\Sigma_\varepsilon$ | Causal states (epsilon-machine) | {prf:ref}`def-epsilon-machine` |
| $T_\varepsilon$ | Epsilon-machine transition function | {prf:ref}`def-epsilon-machine` |
| $I(A;B)$ | Mutual information | {prf:ref}`def-information-closure-cosmo` |
| $L_{\mathrm{AdS}}$ | AdS radius | {prf:ref}`thm-ads-boundary-uv` |
| $T_{\mathrm{eff}}$ | Effective temperature | {prf:ref}`thm-qsd-gibbs` |
| $w$ | Equation of state parameter | {prf:ref}`prop-dark-energy-exploration` |
| QSD | Quasi-stationary distribution | {prf:ref}`def-qsd-regime` |

---

## References

### Framework Documents

- {doc}`03_curvature_gravity` --- Emergent curvature and the Raychaudhuri equation
- {doc}`05_holography` --- Holographic structure, IG pressure, and thermodynamics
- {doc}`04_field_equations` --- Field equations and pressure regimes

### External References

This chapter draws on standard results from:

- **Cosmology:** Friedmann equations, cosmological constant, dark energy observations
- **General relativity:** Raychaudhuri equation, de Sitter and Anti-de Sitter spacetimes
- **Computational mechanics:** Epsilon-machines, information closure, causal states
- **Quantum gravity:** de Sitter conjecture, string landscape, holographic vacuum structure
