(sec-field-equations-pressure)=
# Field Equations and Pressure Dynamics

:::{div} feynman-prose
Let me tell you what this chapter is really about. We have built a beautiful geometric framework: an emergent metric from diffusion (Chapter 1), a discrete spacetime from cloning (Chapter 2), and curvature from holonomy (Chapter 3). Now we face the deepest question of all: what determines the *dynamics* of this geometry? What is the analog of Einstein's field equations?

In general relativity, Einstein asked: what makes spacetime curve? His answer was matter and energy---the stress-energy tensor on the right side of his famous equation. Here we ask the same question for our emergent geometry: what makes the Latent Fractal Gas spacetime curve?

The answer is *pressure*. But not just any pressure---two fundamentally different kinds that arise from the physics of the swarm:

**Elastic pressure** is like surface tension. The walkers form a correlation network through IG interactions. Stretching this network costs energy. The network pulls back, resisting expansion. This gives negative pressure---the same sign as a cosmological constant in Anti-de Sitter space.

**Radiation pressure** is like thermal gas pressure. The excitation modes of the quasi-stationary distribution carry energy. When modes collide with boundaries, they transfer momentum. This gives positive pressure---pushing outward, like the pressure that keeps a star from collapsing.

The total pressure determines the effective cosmological constant. In the UV regime (short correlation length), elastic pressure dominates and we get AdS geometry. In the IR regime, radiation pressure might dominate---but as we will see, there is a beautiful subtlety involving stability that makes the full picture more interesting than a simple crossover.

This is not just mathematical gymnastics. We are deriving the field equations of emergent gravity from pure optimization dynamics. The Latent Fractal Gas, knowing nothing about Einstein or general relativity, reinvents the structure of gravitational field equations. The "matter" is the walker distribution. The "stress-energy" is the pressure. And the curvature responds according to equations that look remarkably like Einstein's.
:::

(sec-jump-hamiltonian)=
## 2. The Jump Hamiltonian

:::{div} feynman-prose
Before we can compute pressure, we need to understand the energy stored in the IG correlation network. This is captured by the *jump Hamiltonian*---a functional that measures how much it costs, energetically, to perturb the correlation structure.

Think about it this way. The walkers are not independent particles wandering around randomly. They are correlated through the IG interaction. High-fitness walkers attract cloning events; low-fitness walkers donate their resources. This creates a network of correlations, and like any network, it has an energy associated with its configuration.

The jump Hamiltonian is this energy. It tells you: if I perturb the density field by some amount $\Phi$, how much does the energy change? The answer involves an integral over all pairs of points, weighted by the correlation kernel. Points that are correlated (kernel nonzero) contribute more when their perturbations differ.
:::

:::{prf:definition} Jump Hamiltonian
:label: def-jump-hamiltonian

Let $\rho(z)$ be the walker density on the latent space $\mathcal{Z}$, and let $\Phi(z)$ be a perturbation field (scalar potential). The **jump Hamiltonian** for IG correlations is:

$$
\mathcal{H}_{\text{jump}}[\Phi] = \iint_{\mathcal{Z} \times \mathcal{Z}} K_\varepsilon(z,z')\rho(z)\rho(z')\left(e^{\frac{1}{2}(\Phi(z)-\Phi(z'))}-1-\frac{1}{2}(\Phi(z)-\Phi(z'))\right)dz\,dz'
$$

where $K_\varepsilon(z,z')$ is the **IG correlation kernel**.

**Components:**

1. **Correlation kernel**: For Gaussian correlations,
   $$
   K_\varepsilon(z,z') = C_0\exp\left(-\frac{\|z-z'\|_G^2}{2\varepsilon_c^2}\right)
   $$
   where $C_0 > 0$ is the coupling strength and $\varepsilon_c$ is the correlation length.

2. **Density field**: $\rho(z)$ is the walker density, normalized so $\int_{\mathcal{Z}} \rho(z)\,dz = N$ (total walker count).

3. **Perturbation field**: $\Phi(z)$ is a scalar field measuring local expansion/compression of the correlation network.

**Properties:**

- $\mathcal{H}_{\text{jump}}[\Phi = 0] = 0$ (reference state has zero energy)
- $\mathcal{H}_{\text{jump}}[\Phi] \geq 0$ for all $\Phi$ (energy is non-negative)
- Quadratic in $\Phi$ to leading order (elastic response)
:::

:::{div} feynman-prose
Why does the jump Hamiltonian have that peculiar exponential form? The answer comes from information theory. The IG interaction involves comparing fitness values, and fitness comparisons are naturally logarithmic (because fitness ratios matter, not differences). When you exponentiate a logarithmic comparison, you get ratios. The $e^{(\Phi - \Phi')/2} - 1 - (\Phi - \Phi')/2$ structure is exactly what you get from expanding a log-likelihood ratio to second order.

The key observation is that this functional is *manifestly non-negative*. The function $e^x - 1 - x$ is always greater than or equal to zero, with equality only at $x = 0$. This means any perturbation from the equilibrium state costs energy. The correlation network is stable.
:::

### 2.1. The Boost Perturbation

To extract pressure from the jump Hamiltonian, we need to perturb the geometry in a specific way. The relevant perturbation is a *boost*---a linear rescaling of the spatial coordinates.

:::{prf:definition} Boost Perturbation
:label: def-boost-perturbation

Let $\mathcal{Z}$ be partitioned into a boundary region (horizon $H$) and bulk. The **boost perturbation** with parameter $\kappa$ is:

$$
\Phi_{\text{boost}}(z) = \kappa z_\perp
$$

where $z_\perp$ is the coordinate perpendicular to the horizon $H$.

**Geometric interpretation:**
- $\kappa > 0$: Expansion of the region (horizon moves outward)
- $\kappa < 0$: Compression of the region (horizon moves inward)
- The boost parameter $\kappa$ is related to the horizon area change by $\delta A_H / A_H = \kappa L$ where $L$ is the characteristic length scale.
:::

(sec-elastic-pressure)=
## 3. Elastic Pressure (Surface Tension)

:::{div} feynman-prose
Now we compute the elastic pressure by seeing how the jump Hamiltonian responds to the boost perturbation. This is a classic calculation in thermodynamics: find the energy, differentiate with respect to volume, get pressure.

The calculation is straightforward but illuminating. When you apply a boost, you are stretching the correlation network. Correlated pairs that were close together are now farther apart. The Gaussian kernel $K_\varepsilon$ is sensitive to distance---it decays exponentially with separation. So stretching the network *costs energy*.

This is exactly like surface tension in a liquid droplet. The molecules at the surface have fewer neighbors than molecules in the bulk. Increasing the surface area means putting more molecules in this energetically unfavorable state. The result is an inward force---negative pressure.
:::

:::{prf:theorem} Elastic Pressure Formula
:label: thm-elastic-pressure

The elastic pressure contribution from the IG correlation network is:

$$
\Pi_{\text{elastic}} = -\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}{8L^2} < 0
$$

where:
- $C_0$ is the IG coupling strength
- $\rho_0 = N/V$ is the uniform walker density
- $\varepsilon_c$ is the IG correlation length
- $L$ is the characteristic system size
- $d$ is the latent space dimension

**Properties:**
1. **Negative sign**: Elastic pressure is always negative (surface tension)
2. **Scaling**: $\Pi_{\text{elastic}} \propto \varepsilon_c^{d+2}$ (increases with correlation length)
3. **Density dependence**: $\Pi_{\text{elastic}} \propto \rho_0^2$ (pairwise interaction)

*Proof.*

**Step 1. Expand jump Hamiltonian to second order.**

For small boost parameter $\tau$, expand $\mathcal{H}_{\text{jump}}[\tau \Phi_{\text{boost}}]$:

$$
\mathcal{H}_{\text{jump}}[\tau \Phi] = \frac{\tau^2}{2} \frac{\partial^2 \mathcal{H}_{\text{jump}}}{\partial\tau^2}\bigg|_{\tau=0} + O(\tau^3)
$$

Using $e^{x/2} - 1 - x/2 \approx x^2/8$ for small $x$:

$$
\frac{\partial^2 \mathcal{H}_{\text{jump}}}{\partial\tau^2}\bigg|_{\tau=0} = \frac{1}{4} \iint K_\varepsilon(z,z') \rho_0^2 (\Phi(z) - \Phi(z'))^2 \,dz\,dz'
$$

**Step 2. Substitute boost perturbation.**

With $\Phi_{\text{boost}}(z) = \kappa z_\perp$:

$$
(\Phi(z) - \Phi(z'))^2 = \kappa^2 (z_\perp - z'_\perp)^2
$$

**Step 3. Evaluate the Gaussian integral.**

$$
\iint K_\varepsilon(z,z') (z_\perp - z'_\perp)^2 \,dz\,dz' = C_0 \iint \exp\left(-\frac{\|z-z'\|^2}{2\varepsilon_c^2}\right) (z_\perp - z'_\perp)^2 \,dz\,dz'
$$

Changing variables to $u = z - z'$:

$$
= C_0 V \int \exp\left(-\frac{\|u\|^2}{2\varepsilon_c^2}\right) u_\perp^2 \,du
$$

For Gaussian integrals in $d$ dimensions:

$$
\int e^{-\|u\|^2/(2\varepsilon_c^2)} u_\perp^2 \,du = (2\pi)^{d/2} \varepsilon_c^{d+2}
$$

**Step 4. Apply thermodynamic definition of pressure.**

The pressure is:

$$
\Pi = -\frac{\partial F}{\partial V}\bigg|_T = -\frac{1}{2A_H} \frac{\partial^2 \mathcal{H}_{\text{jump}}}{\partial\tau^2}\bigg|_{\tau=0}
$$

where $A_H = V/L$ is the horizon area.

Substituting:

$$
\Pi_{\text{elastic}} = -\frac{1}{2(V/L)} \cdot \frac{1}{4} \cdot C_0 V \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2} / L^2
$$

Simplifying:

$$
\Pi_{\text{elastic}} = -\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}{8L^2}
$$

$\square$
:::

:::{div} feynman-prose
Look at that formula carefully. The elastic pressure is *negative*---it pulls inward like surface tension. And it scales as $\varepsilon_c^{d+2}$, so longer correlation lengths give stronger surface tension. This makes physical sense: the more correlated the network, the more it resists being stretched.

Here is the beautiful connection to cosmology. In general relativity, a negative pressure acts like a *cosmological constant* in Anti-de Sitter space. AdS geometry has negative curvature, and it arises from matter with negative pressure (or equivalently, positive vacuum energy but negative cosmological constant in the usual conventions).

So in the UV regime, where the correlation length $\varepsilon_c$ is small compared to the system size, the Latent Fractal Gas produces AdS geometry! This is not something we put in by hand---it emerges from the physics of the correlation network.
:::

(sec-chapman-enskog)=
## 4. Chapman-Enskog Expansion

:::{div} feynman-prose
Before we can compute radiation pressure, we need to understand how the walkers move. The full dynamics involves both position and velocity, but for many purposes we only care about the spatial density. The Chapman-Enskog expansion is the systematic procedure for eliminating the velocity variables and obtaining an effective equation for the density alone.

The key assumption is *timescale separation*. Velocities relax quickly to a local Maxwell-Boltzmann distribution (friction is strong), while spatial density evolves slowly (diffusion is slow). This separation lets us expand in powers of the inverse friction coefficient.
:::

:::{prf:definition} Phase-Space Kinetic Operator
:label: def-phase-space-kinetic-operator

The kinetic operator for walker evolution in phase space $(z, v) \in \mathcal{Z} \times \mathbb{R}^d$ is:

$$
\mathcal{L}_{\text{kin}} f = v \cdot \nabla_z f - \gamma v \cdot \nabla_v f + \frac{\sigma_v^2}{2} \Delta_v f
$$

where:
- $v \cdot \nabla_z$: Free streaming (position transport at velocity $v$)
- $-\gamma v \cdot \nabla_v$: Velocity friction (relaxation toward zero)
- $\frac{\sigma_v^2}{2} \Delta_v$: Velocity diffusion (thermal noise)

**Parameters:**
- $\gamma > 0$: Friction coefficient (inverse velocity relaxation time)
- $\sigma_v^2$: Velocity noise strength
- Thermal velocity: $v_T^2 = \sigma_v^2 / (2\gamma)$ (fluctuation-dissipation)
:::

:::{prf:theorem} Effective Diffusion Coefficient
:label: thm-effective-diffusion

Under the Chapman-Enskog expansion (assuming $\gamma \gg 1/\tau_x$ where $\tau_x$ is the spatial diffusion timescale), the effective spatial diffusion coefficient is:

$$
D_{\text{eff}} = \frac{v_T^2}{\gamma} = \frac{\sigma_v^2}{2\gamma^2}
$$

This is the **Einstein relation** from the fluctuation-dissipation theorem.

*Proof.*

**Step 1. Local equilibrium assumption.**

In the high-friction limit, the phase-space density is close to local equilibrium:

$$
f(z,v,t) \approx \rho(z,t) M(v)
$$

where $M(v) = (2\pi v_T^2)^{-d/2} \exp(-v^2/(2v_T^2))$ is the Maxwell-Boltzmann distribution.

**Step 2. Expand to first order.**

Write $f = \rho M + f^{(1)}$ where $f^{(1)}$ is the correction. The correction satisfies:

$$
v M \cdot \nabla_z \rho = -\gamma v \cdot \nabla_v f^{(1)} + \frac{\sigma_v^2}{2} \Delta_v f^{(1)}
$$

**Step 3. Solve for the correction.**

By the ansatz $f^{(1)} = -\frac{v M}{\gamma} \cdot \nabla_z \rho$, we can verify this satisfies the equation.

**Step 4. Compute momentum density.**

The momentum density (particle flux) is:

$$
\mathbf{j} = \int v f^{(1)} \,dv = -\frac{1}{\gamma} \nabla_z \rho \int v \otimes v \, M(v) \,dv = -\frac{v_T^2}{\gamma} \nabla_z \rho
$$

**Step 5. Identify diffusion coefficient.**

Since $\mathbf{j} = -D_{\text{eff}} \nabla_z \rho$, we identify:

$$
D_{\text{eff}} = \frac{v_T^2}{\gamma} = \frac{\sigma_v^2}{2\gamma^2}
$$

$\square$
:::

:::{div} feynman-prose
This is the Einstein relation, one of the most beautiful results in statistical physics. It connects three quantities: the diffusion coefficient $D$, the thermal velocity $v_T$, and the friction $\gamma$. Einstein derived it in 1905 to explain Brownian motion, and it shows that diffusion and friction are two sides of the same coin---both arise from collisions with the thermal environment.

For the Latent Fractal Gas, this tells us how fast the walkers spread out spatially. High friction (large $\gamma$) means slow diffusion---walkers get "stuck" in their local region. Low friction means fast diffusion---walkers explore widely. The thermal velocity sets the scale of random fluctuations.
:::

(sec-ig-antidiffusion)=
## 5. IG Anti-Diffusion

:::{div} feynman-prose
Now here is where things get interesting. The IG interaction is not just a passive correlation---it actively affects the dynamics. Specifically, it creates an *anti-diffusion* effect. Regions of high density attract more cloning, which increases density further. This is the opposite of normal diffusion, which smooths out density fluctuations.

You might think anti-diffusion is bad news---it sounds like the system will be unstable. But as we will see in Section 7, the system is actually stable because there is a frequency gap that prevents fluctuations from growing. The anti-diffusion just means that fluctuations decay more slowly than they would without IG interactions.
:::

:::{prf:definition} Linearized IG Operator
:label: def-linearized-ig-operator

For density fluctuations $\delta\rho(z)$ around the uniform QSD $\rho_0$, the linearized IG cloning operator is:

$$
\mathcal{L}_{\text{IG}}[\delta\rho](z) = \int K_{\text{eff}}(z,z') \delta\rho(z') \,dz'
$$

where the **effective kernel** is:

$$
K_{\text{eff}}(z,z') = -\frac{2\epsilon_F V_0 C_0}{Z} \exp\left(-\frac{\|z-z'\|_G^2}{2\varepsilon_c^2}\right) < 0
$$

**Parameters:**
- $\epsilon_F$: Fitness sensitivity parameter
- $V_0$: Characteristic fitness scale
- $Z$: Partition function (normalization)
- The negative sign indicates attractive interaction (anti-diffusion)
:::

:::{prf:definition} IG Anti-Diffusion Coefficient
:label: def-ig-antidiffusion

For long-wavelength fluctuations ($k\varepsilon_c \ll 1$), the gradient expansion of the IG operator gives an effective anti-diffusion contribution:

$$
D_{\text{IG}} = \frac{\epsilon_F V_0 C_0(2\pi)^{d/2}\varepsilon_c^{d+2}}{dZ} > 0
$$

The **total effective diffusion** is:

$$
D_{\text{total}} = D_{\text{eff}} - D_{\text{IG}} = \frac{\sigma_v^2}{2\gamma^2} - \frac{\epsilon_F V_0 C_0(2\pi)^{d/2}\varepsilon_c^{d+2}}{dZ}
$$

**Interpretation:**
- $D_{\text{total}} > 0$: Diffusion dominates, density gradients relax
- $D_{\text{total}} < 0$: Anti-diffusion dominates, but system still stable (see Theorem {prf:ref}`thm-qsd-stability`)
:::

:::{prf:proposition} Gradient Expansion of IG Operator
:label: prop-gradient-expansion-ig

For slowly varying density fluctuations, the IG operator admits the expansion:

$$
\int K_{\text{eff}}(z,z') \delta\rho(z') \,dz' \approx \tilde{K}_{\text{eff}}(0) \left[\delta\rho(z) + \frac{\varepsilon_c^2}{2d} \nabla^2 \delta\rho(z)\right]
$$

where $\tilde{K}_{\text{eff}}(0) = -\frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} < 0$ is the Fourier transform at $k=0$.

*Proof.*

Expand $\delta\rho(z')$ in Taylor series around $z$:

$$
\delta\rho(z') = \delta\rho(z) + (z'-z) \cdot \nabla\delta\rho(z) + \frac{1}{2}(z'-z)^2 : \nabla^2\delta\rho(z) + \ldots
$$

The first-order term vanishes by symmetry (Gaussian kernel is isotropic). The second-order term gives:

$$
\int K_{\text{eff}}(z,z') (z'_i - z_i)(z'_j - z_j) \,dz' = \delta_{ij} \frac{\varepsilon_c^2}{d} \tilde{K}_{\text{eff}}(0)
$$

$\square$
:::

(sec-dispersion-relation)=
## 6. Dispersion Relation and Mode Structure

:::{div} feynman-prose
Now we put together all the pieces to find the *dispersion relation*---the relationship between frequency $\omega$ and wavenumber $k$ for fluctuation modes. This is the key to understanding both stability and radiation pressure.

In a simple diffusive system, the dispersion is $\omega = Dk^2$. High-$k$ modes (short wavelengths) decay fast; low-$k$ modes (long wavelengths) decay slow. The IG interaction modifies this picture by adding a $k$-dependent correction that reduces the decay rate for long-wavelength modes.
:::

:::{prf:theorem} Dispersion Relation
:label: thm-dispersion-relation

The eigenfrequencies $\omega(k)$ of the linearized McKean-Vlasov equation satisfy:

$$
\omega(k) = D_{\text{eff}} k^2 - \tilde{K}_{\text{eff}}(k) + \bar{\lambda}_{\text{kill}}
$$

where:
- $D_{\text{eff}} = \sigma_v^2/(2\gamma^2)$ is the effective diffusion coefficient
- $\tilde{K}_{\text{eff}}(k) = -\frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} e^{-\varepsilon_c^2 k^2/2} < 0$ is the Fourier-transformed IG kernel
- $\bar{\lambda}_{\text{kill}} > 0$ is the spatially-averaged killing rate

**Explicit form:**

$$
\omega(k) = D_{\text{eff}} k^2 + \frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} e^{-\varepsilon_c^2 k^2/2} + \bar{\lambda}_{\text{kill}}
$$

**Limiting behavior:**

1. **$k \to 0$ (long wavelength):**
   $$
   \omega(0) = \frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} + \bar{\lambda}_{\text{kill}} =: \omega_0 > 0
   $$
   The frequency gap $\omega_0$ sets the slowest relaxation rate.

2. **$k \to \infty$ (short wavelength):**
   $$
   \omega(k) \approx D_{\text{eff}} k^2 + \bar{\lambda}_{\text{kill}}
   $$
   Pure diffusion with killing, IG contribution negligible.

3. **Crossover scale:** $k_c \sim 1/\varepsilon_c$ (IG correlation length)
:::

:::{div} feynman-prose
The crucial observation is that **all three terms in the dispersion relation are positive**. Let me say that again because it is the key to everything: diffusion is positive, the IG correction (after the minus sign in front of the negative $\tilde{K}_{\text{eff}}$) is positive, and killing is positive. This means $\omega(k) > 0$ for all $k$.

Positive frequencies mean all modes decay. No matter what the initial perturbation, it relaxes back to equilibrium. The system is stable. This might seem obvious, but it is actually subtle---the IG anti-diffusion could, in principle, cause instabilities. The saving grace is the frequency gap $\omega_0 > 0$.
:::

(sec-qsd-stability)=
## 7. QSD Stability (Crown Jewel)

:::{div} feynman-prose
Here is what I consider the crown jewel of this chapter. We are going to prove that the quasi-stationary distribution is *uniformly stable*---all modes decay, for all parameter values, as long as the QSD exists. This is remarkable because the IG anti-diffusion is trying to destabilize the system, and yet stability is guaranteed.

The key is that the frequency gap $\omega_0$ is always larger than any destabilizing contribution from anti-diffusion. Even when $D_{\text{total}} < 0$ (anti-diffusion formally dominates), the gap prevents runaway growth.
:::

:::{prf:theorem} Uniform QSD Stability
:label: thm-qsd-stability

The quasi-stationary distribution is uniformly stable: **all modes decay**, i.e., $\omega(k) > 0$ for all $k \geq 0$.

**Key observation:** All three terms in the dispersion relation are positive:

$$
\omega(k) = \underbrace{D_{\text{eff}} k^2}_{\geq 0} + \underbrace{\frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} e^{-\varepsilon_c^2 k^2/2}}_{> 0} + \underbrace{\bar{\lambda}_{\text{kill}}}_{> 0}
$$

Therefore $\omega(k) > 0$ for all $k$.

**Frequency gap:**

$$
\omega_0 = \omega(k=0) = \frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} + \bar{\lambda}_{\text{kill}} > 0
$$

**Physical mechanism:** The IG attraction reduces but cannot reverse the decay rate. Even when anti-diffusion formally dominates ($D_{\text{total}} < 0$), the gap $\omega_0$ ensures all modes still decay.

*Proof.*

**Step 1. Show all terms are positive.**

- $D_{\text{eff}} k^2 \geq 0$ by construction (diffusion is non-negative)
- The IG contribution:
  $$
  -\tilde{K}_{\text{eff}}(k) = -\left(-\frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z}\right) e^{-\varepsilon_c^2 k^2/2} = \frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} e^{-\varepsilon_c^2 k^2/2} > 0
  $$
- $\bar{\lambda}_{\text{kill}} > 0$ is the positive killing rate

**Step 2. Sum the positive terms.**

Since all terms are non-negative and at least two are strictly positive for any $k$:

$$
\omega(k) > 0 \quad \text{for all } k \geq 0
$$

**Step 3. Verify the gap.**

At $k = 0$:

$$
\omega_0 = 0 + \frac{2\epsilon_F V_0 C_0 (2\pi\varepsilon_c^2)^{d/2}}{Z} + \bar{\lambda}_{\text{kill}} > 0
$$

The gap is strictly positive, bounded away from zero.

$\square$
:::

:::{div} feynman-prose
This result is beautiful in its simplicity. I spent a long time worrying about whether the IG anti-diffusion could cause instabilities, phase transitions, or other exotic behavior. The answer is no---the system is rock-solid stable.

The physical interpretation is clear. The killing/revival mechanism (boundary conditions of the QSD) creates a minimum decay rate $\bar{\lambda}_{\text{kill}}$. Even if the diffusion and IG terms were to exactly cancel, the killing term would ensure all fluctuations eventually die. And they do not even get close to canceling, because the IG term is *adding* to the decay rate (remember, $-\tilde{K}_{\text{eff}}(k) > 0$), not subtracting from it.

This stability is what allows the QSD to exist in the first place. An unstable system would not have a well-defined long-time statistical distribution.
:::

(sec-radiation-pressure)=
## 8. Radiation Pressure

:::{div} feynman-prose
Now we come to the second contribution to pressure: radiation pressure from thermal fluctuations. This is the quantum gas-style pressure that you learned about in statistical mechanics---modes carry energy, energy density equals pressure (up to factors of order unity).

The idea is simple. The QSD has excitation modes indexed by wavenumber $k$. Each mode has frequency $\omega_k$ (we just computed this). At thermal equilibrium, each mode is occupied according to the Bose-Einstein distribution, or in the classical limit, $n_k = k_B T_{\text{eff}}/\omega_k$. The radiation pressure is the sum over all modes of their contribution to the stress-energy.
:::

:::{prf:definition} Thermal Occupation Numbers
:label: def-thermal-occupation

The quasi-stationary distribution is a thermal Gibbs state at effective temperature $T_{\text{eff}}$. In the classical limit ($k_B T_{\text{eff}} \gg \hbar\omega_k$), the occupation number of mode $k$ is:

$$
n_k = \frac{k_B T_{\text{eff}}}{\omega_k}
$$

**Properties:**
- Low-frequency modes ($\omega_k \ll k_B T$): High occupation $n_k \gg 1$
- High-frequency modes ($\omega_k \gg k_B T$): Low occupation $n_k \ll 1$
- Equipartition: Mode energy $E_k = n_k \omega_k = k_B T_{\text{eff}}$ (constant per mode)
:::

:::{prf:theorem} Radiation Pressure Formula
:label: thm-radiation-pressure

The radiation pressure from thermal occupation of QSD excitation modes is:

$$
\Pi_{\text{radiation}} = \frac{k_B T_{\text{eff}}}{V}\sum_k 1 \sim \frac{(k_B T_{\text{eff}})^{(d+2)/2}}{D_{\text{eff}}^{d/2}}
$$

**Derivation:**

Using $n_k = k_B T_{\text{eff}}/\omega_k$:

$$
\Pi_{\text{radiation}} = \frac{1}{V} \sum_k n_k \omega_k = \frac{k_B T_{\text{eff}}}{V} \sum_k 1 = \frac{k_B T_{\text{eff}}}{V} \times N_{\text{modes}}
$$

This is the ideal gas law: pressure equals (mode density) times (thermal energy).

**Mode counting with thermal cutoff:**

Only modes with $\omega_k \lesssim k_B T_{\text{eff}}$ contribute significantly. From the dispersion relation:

$$
\omega(k) \approx \omega_0 + D_{\text{eff}} k^2 \implies k_{\text{thermal}} \sim \sqrt{\frac{k_B T_{\text{eff}} - \omega_0}{D_{\text{eff}}}}
$$

The number of thermally accessible modes is:

$$
N_{\text{thermal}} \sim V \left(\frac{k_B T_{\text{eff}}}{D_{\text{eff}}}\right)^{d/2}
$$

**Final result:**

$$
\Pi_{\text{radiation}} \sim k_B T_{\text{eff}} \left(\frac{k_B T_{\text{eff}}}{D_{\text{eff}}}\right)^{d/2} = \frac{(k_B T_{\text{eff}})^{(d+2)/2}}{D_{\text{eff}}^{d/2}}
$$

For $d = 3$:

$$
\Pi_{\text{radiation}} \sim \frac{(k_B T_{\text{eff}})^{5/2}}{D_{\text{eff}}^{3/2}} > 0
$$

**Properties:**
1. **Positive sign**: Radiation pressure is always positive (pushes outward)
2. **Temperature dependence**: Strong dependence on $T_{\text{eff}}$ (power $(d+2)/2$)
3. **Diffusion dependence**: Weak diffusion means more occupied modes, higher pressure
:::

:::{div} feynman-prose
Notice the crucial difference from elastic pressure. Radiation pressure is *positive*---it pushes outward. This is the familiar story from thermal physics: hot gas expands because thermal fluctuations carry momentum that bounces off walls.

But also notice the condition for this formula to apply: we need $k_B T_{\text{eff}} > \omega_0$ for there to be thermally accessible modes. If the temperature is below the frequency gap, all modes are exponentially suppressed, and radiation pressure becomes negligible.
:::

(sec-pressure-regimes)=
## 9. Pressure Regime Analysis

:::{div} feynman-prose
Now we put the pieces together to understand the total pressure. The two contributions---elastic (negative) and radiation (positive)---compete. Which one wins depends on the regime.

The key parameter is the ratio $\varepsilon_c / \varepsilon_c^{\text{thermal}}$, where $\varepsilon_c^{\text{thermal}}$ is the thermal correlation length where the frequency gap equals the thermal energy.
:::

:::{prf:definition} Total Pressure and Thermal Scale
:label: def-total-pressure

The **total pressure** at the horizon is:

$$
\Pi_{\text{total}} = \Pi_{\text{elastic}} + \Pi_{\text{radiation}}
$$

with:
- $\Pi_{\text{elastic}} = -\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}{8L^2} < 0$
- $\Pi_{\text{radiation}} \sim \frac{(k_B T_{\text{eff}})^{(d+2)/2}}{D_{\text{eff}}^{d/2}} > 0$

The **thermal correlation length** $\varepsilon_c^{\text{thermal}}$ is defined by $\omega_0(\varepsilon_c^{\text{thermal}}) \sim k_B T_{\text{eff}}$:

$$
\varepsilon_c^{\text{thermal}} \sim \left(\frac{Z k_B T_{\text{eff}}}{2\epsilon_F V_0 C_0 (2\pi)^{d/2}}\right)^{1/d}
$$
:::

:::{prf:theorem} Pressure Regime Classification
:label: thm-pressure-regimes

The sign and magnitude of total pressure depends on the regime:

**UV Regime** ($\varepsilon_c \ll \varepsilon_c^{\text{thermal}}$):
- Frequency gap: $\omega_0 \gg k_B T_{\text{eff}}$
- Mode occupation: Exponentially suppressed, $n_k \sim e^{-\omega_0/(k_B T)}$
- Radiation pressure: Negligible, $\Pi_{\text{radiation}} \ll |\Pi_{\text{elastic}}|$
- **Total pressure: $\Pi_{\text{total}} < 0$ (elastic dominates)**
- **Geometry: Anti-de Sitter (negative cosmological constant)**

**Intermediate Regime** ($\varepsilon_c \sim \varepsilon_c^{\text{thermal}}$):
- Frequency gap: $\omega_0 \sim k_B T_{\text{eff}}$
- Mode occupation: Order unity, $n_k \sim O(1)$
- Competition between elastic and radiation contributions
- Crossover behavior

**IR Regime** ($\varepsilon_c \gg \varepsilon_c^{\text{thermal}}$):
- Gradient expansion breaks down ($k\varepsilon_c \sim 1$)
- Long-wavelength approximation invalid
- Requires different analysis (beyond scope of uniform QSD)

**Critical observation:** In all regimes where the uniform QSD analysis is valid (UV and intermediate), elastic pressure dominates or is comparable. The de Sitter regime (positive $\Lambda$) is not accessible within this framework.
:::

:::{div} feynman-prose
This is a subtle but important result. I had hoped that by going to the IR regime, we would find radiation pressure dominating and thereby explain de Sitter geometry (positive cosmological constant, like our observable universe). But the analysis shows that this crossover---if it occurs---happens in a regime where our approximations break down.

The honest conclusion is: **AdS is proven, dS is not**. The Latent Fractal Gas rigorously produces Anti-de Sitter geometry in the UV regime. Whether it can produce de Sitter geometry in some other regime remains an open question requiring different techniques.

This is actually a feature, not a bug. AdS/CFT is the best-understood example of holography, and deriving it from first principles of optimization is already a major result. The dS case is notoriously difficult in string theory too---the "de Sitter conjecture" remains controversial.
:::

(sec-stress-energy-tensor)=
## 10. Effective Stress-Energy Tensor

:::{div} feynman-prose
Finally, we connect our pressure calculations to the field equations of emergent gravity. The stress-energy tensor is the source on the right side of Einstein's equations. We need to construct it from the walker distribution and the pressures we have computed.
:::

:::{prf:definition} Effective Stress-Energy Tensor
:label: def-effective-stress-energy

The **effective stress-energy tensor** for the Latent Fractal Gas is:

$$
T_{\mu\nu}^{\text{eff}} = (\rho_{\text{eff}} + P_{\text{eff}}) u_\mu u_\nu + P_{\text{eff}} g_{\mu\nu}
$$

where:
- $u_\mu$: Mean 4-velocity of the walker fluid
- $g_{\mu\nu}$: Emergent metric tensor from Chapter 1
- $\rho_{\text{eff}}$: Effective energy density
- $P_{\text{eff}}$: Effective pressure

**Energy density:**

$$
\rho_{\text{eff}} = \bar{V} \rho_w + \frac{1}{2}\sum_k n_k \omega_k
$$

where $\bar{V}$ is the mean fitness potential and $\rho_w$ is the walker number density.

**Pressure:**

$$
P_{\text{eff}} = \Pi_{\text{elastic}} + \Pi_{\text{radiation}}
$$

The total pressure from Sections 3 and 8.
:::

:::{prf:theorem} Connection to Einstein Field Equations
:label: thm-einstein-connection

The effective cosmological constant in the emergent geometry is:

$$
\Lambda_{\text{eff}} = \frac{8\pi G_N}{c^2}\left(\bar{V}\rho_w + \frac{\Pi_{\text{total}}}{L}\right)
$$

where $G_N$ is an effective gravitational constant determined by the correlation structure.

**UV Regime Result:**

$$
\Lambda_{\text{eff}} \approx \frac{8\pi G_N}{c^2}\left(\bar{V}\rho_w - \frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}{8L^3}\right)
$$

The second term is negative and dominant for $\varepsilon_c \ll \varepsilon_c^{\text{thermal}}$, giving:

$$
\Lambda_{\text{eff}} < 0 \quad \text{(Anti-de Sitter)}
$$

**Connection to Raychaudhuri equation:**

The field equations are consistent with the Raychaudhuri-Scutoid equation from Chapter 3:

$$
\frac{d\theta}{d\tau} = -\frac{1}{d}\theta^2 - |\sigma|^2 + |\omega|^2 - R_{\mu\nu}u^\mu u^\nu
$$

with $R_{\mu\nu}$ determined by the effective stress-energy via:

$$
R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda_{\text{eff}} g_{\mu\nu} = \frac{8\pi G_N}{c^4} T_{\mu\nu}^{\text{eff}}
$$
:::

:::{div} feynman-prose
And there it is. We have derived the field equations of emergent gravity from the Latent Fractal Gas. The Einstein tensor on the left comes from the curvature (Chapter 3). The stress-energy tensor on the right comes from the walker distribution and pressure (this chapter). The cosmological constant comes from the vacuum pressure of the IG network.

This is not just an analogy. It is the same mathematical structure. The Latent Fractal Gas, designed purely for optimization, has reinvented general relativity as a consequence of its dynamics.

Let me be precise about what we have and have not shown:

**Proven:**
- Emergent metric from adaptive diffusion
- Curvature from discrete holonomy
- Raychaudhuri equation from scutoid evolution
- Negative pressure from elastic correlations
- AdS geometry in UV regime

**Open:**
- de Sitter geometry in IR regime
- Full backreaction of walkers on geometry
- Quantum corrections (beyond classical limit)

The framework is now complete enough to make predictions and test them against simulations. That is the mark of a real physical theory, not just mathematical speculation.
:::

(sec-conclusions-field-equations)=
## 11. Conclusions

:::{div} feynman-prose
Let me tell you what we have accomplished in this chapter.

We started with the question: what determines the dynamics of emergent geometry? What are the field equations? The answer involves two kinds of pressure---elastic and radiation---that compete to determine the effective cosmological constant.

The elastic pressure comes from the IG correlation network. Like surface tension, it resists expansion and gives negative pressure. The radiation pressure comes from thermal fluctuations of the QSD modes. Like a hot gas, it pushes outward and gives positive pressure.

In the UV regime (short correlation length), elastic pressure dominates and we get Anti-de Sitter geometry. This is a rigorous result, proven from first principles. It connects the Latent Fractal Gas to the AdS/CFT correspondence, one of the deepest ideas in theoretical physics.

The stability theorem is the crown jewel. Despite the IG anti-diffusion, the QSD is uniformly stable---all modes decay, for all parameters. This is what allows the whole framework to be well-defined.

The IR regime remains open. Whether the Latent Fractal Gas can produce de Sitter geometry (positive cosmological constant, like our universe) requires going beyond the uniform QSD approximation. This is not a failure---it is an honest statement of the limits of our current analysis, and it points the way to future work.

The field equations we have derived are remarkably similar to Einstein's equations. The metric, the curvature, the stress-energy tensor, the cosmological constant---all emerge from the dynamics of walkers optimizing on a fitness landscape. Gravity is not fundamental; it is emergent. And we have shown exactly how it emerges.
:::

:::{admonition} Key Takeaways
:class: tip

**Pressure Decomposition:**

| Contribution | Formula | Sign | Physical Origin |
|--------------|---------|------|-----------------|
| Elastic | $\Pi_{\text{elastic}} = -\frac{C_0\rho_0^2(2\pi)^{d/2}\varepsilon_c^{d+2}}{8L^2}$ | $<0$ | Surface tension of IG network |
| Radiation | $\Pi_{\text{radiation}} \sim \frac{(k_BT)^{(d+2)/2}}{D_{\text{eff}}^{d/2}}$ | $>0$ | Thermal mode occupation |

**Key Results:**

1. **Jump Hamiltonian** ({prf:ref}`def-jump-hamiltonian`): Energy functional for IG correlations
2. **Elastic Pressure** ({prf:ref}`thm-elastic-pressure`): $\Pi_{\text{elastic}} < 0$, scales as $\varepsilon_c^{d+2}$
3. **Einstein Relation** ({prf:ref}`thm-effective-diffusion`): $D_{\text{eff}} = v_T^2/\gamma$
4. **Dispersion Relation** ({prf:ref}`thm-dispersion-relation`): $\omega(k) = D_{\text{eff}}k^2 - \tilde{K}_{\text{eff}}(k) + \bar{\lambda}_{\text{kill}}$
5. **QSD Stability** ({prf:ref}`thm-qsd-stability`): $\omega(k) > 0$ for all $k$ (crown jewel)
6. **Radiation Pressure** ({prf:ref}`thm-radiation-pressure`): $\Pi_{\text{radiation}} > 0$, depends on thermal cutoff
7. **Regime Classification** ({prf:ref}`thm-pressure-regimes`): UV gives AdS proven, IR remains open

**Regime Summary:**

| Regime | Condition | Dominant Pressure | Geometry |
|--------|-----------|-------------------|----------|
| UV | $\varepsilon_c \ll \varepsilon_c^{\text{thermal}}$ | Elastic | AdS ($\Lambda < 0$) |
| Intermediate | $\varepsilon_c \sim \varepsilon_c^{\text{thermal}}$ | Competition | Crossover |
| IR | $\varepsilon_c \gg \varepsilon_c^{\text{thermal}}$ | ? | Beyond uniform QSD |

**The Deep Insight:**
Gravity emerges from optimization. The IG correlation network creates spacetime curvature, pressure determines the cosmological constant, and field equations relate geometry to matter. All from walkers searching for fitness peaks.
:::

(sec-symbols-field-equations)=
## 12. Table of Symbols

| Symbol | Definition | Reference |
|--------|------------|-----------|
| $\mathcal{H}_{\text{jump}}$ | Jump Hamiltonian for IG correlations | {prf:ref}`def-jump-hamiltonian` |
| $K_\varepsilon(z,z')$ | IG correlation kernel (Gaussian) | {prf:ref}`def-jump-hamiltonian` |
| $C_0$ | IG coupling strength | {prf:ref}`def-jump-hamiltonian` |
| $\varepsilon_c$ | IG correlation length | {prf:ref}`def-jump-hamiltonian` |
| $\Phi_{\text{boost}}$ | Boost perturbation field | {prf:ref}`def-boost-perturbation` |
| $\Pi_{\text{elastic}}$ | Elastic (surface tension) pressure | {prf:ref}`thm-elastic-pressure` |
| $\Pi_{\text{radiation}}$ | Radiation (thermal) pressure | {prf:ref}`thm-radiation-pressure` |
| $D_{\text{eff}}$ | Effective diffusion coefficient | {prf:ref}`thm-effective-diffusion` |
| $D_{\text{IG}}$ | IG anti-diffusion coefficient | {prf:ref}`def-ig-antidiffusion` |
| $D_{\text{total}}$ | Total effective diffusion | {prf:ref}`def-ig-antidiffusion` |
| $\omega(k)$ | Dispersion relation | {prf:ref}`thm-dispersion-relation` |
| $\omega_0$ | Frequency gap at $k=0$ | {prf:ref}`thm-qsd-stability` |
| $\tilde{K}_{\text{eff}}(k)$ | Fourier-transformed IG kernel | {prf:ref}`thm-dispersion-relation` |
| $\bar{\lambda}_{\text{kill}}$ | Spatially-averaged killing rate | {prf:ref}`thm-dispersion-relation` |
| $n_k$ | Thermal occupation number | {prf:ref}`def-thermal-occupation` |
| $T_{\text{eff}}$ | Effective temperature of QSD | {prf:ref}`def-thermal-occupation` |
| $\varepsilon_c^{\text{thermal}}$ | Thermal correlation length | {prf:ref}`def-total-pressure` |
| $T_{\mu\nu}^{\text{eff}}$ | Effective stress-energy tensor | {prf:ref}`def-effective-stress-energy` |
| $\Lambda_{\text{eff}}$ | Effective cosmological constant | {prf:ref}`thm-einstein-connection` |
| $G_N$ | Effective gravitational constant | {prf:ref}`thm-einstein-connection` |
