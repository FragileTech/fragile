(sec-field-equations-pressure)=
# Field Equations and Pressure Dynamics

**Prerequisites**: {doc}`01_emergent_geometry`, {doc}`02_scutoid_spacetime`, {doc}`03_curvature_gravity`

---

## TLDR

*Notation: $\mathcal{H}_{\mathrm{IG}}$ = IG free energy functional; $\Pi_{\mathrm{elastic}}, \Pi_{\mathrm{radiation}}$ = elastic and radiation pressure; $\omega(k)$ = dispersion relation; $\omega_0$ = frequency gap; $\Lambda_{\mathrm{eff}}$ = effective cosmological constant; $d$ = latent space dimension.*

**Field Equations from Pressure Dynamics**: The emergent geometry of the Latent Fractal Gas is governed by field equations analogous to Einstein's equations, with the stress-energy tensor determined by walker distribution and pressure contributions.

**Two Pressure Mechanisms**:

| Contribution | Formula                                                   | Sign | Physical Origin                           |
|--------------|-----------------------------------------------------------|------|-------------------------------------------|
| Elastic      | $\Pi_{\mathrm{elastic}} \propto -\varepsilon_c^{d+2}$     | $<0$ | Surface tension of IG correlation network |
| Radiation    | $\Pi_{\mathrm{radiation}} \propto T_{\mathrm{eff}}^{d/2}$ | $>0$ | Thermal occupation of QSD modes           |

**Crown Jewel---QSD Stability**: Under explicit parameter conditions ({prf:ref}`thm-qsd-stability`), all modes decay ($\mathrm{Re}(\omega(k)) > 0$ for all $k$), guaranteeing stability of the quasi-stationary distribution.

**UV Regime Consistent with AdS**: In the UV regime ($\varepsilon_c \ll \varepsilon_c^{\mathrm{thermal}}$), elastic pressure dominates, yielding negative effective cosmological constant consistent with Anti-de Sitter geometry.

---

(sec-field-equations-intro)=
## Introduction

:::{div} feynman-prose
Let me tell you what this chapter is really about. We have built a beautiful geometric framework: an emergent metric from diffusion ({doc}`01_emergent_geometry`), a discrete spacetime from cloning ({doc}`02_scutoid_spacetime`), and curvature from holonomy ({doc}`03_curvature_gravity`). Now we face the deepest question of all: what determines the *dynamics* of this geometry? What is the analog of Einstein's field equations?

In general relativity, Einstein asked: what makes spacetime curve? His answer was matter and energy---the stress-energy tensor on the right-hand side of his famous equation. Here we ask the same question for our emergent geometry: what makes the Latent Fractal Gas spacetime curve?

The answer is *pressure*. But not just any pressure---two fundamentally different kinds that arise from the physics of the swarm:

**Elastic pressure** is like surface tension. The walkers form a correlation network through IG interactions. Stretching this network costs energy. The network pulls back, resisting expansion. This gives negative pressure---the same sign as a cosmological constant in Anti-de Sitter space.

**Radiation pressure** is like thermal gas pressure. The excitation modes of the quasi-stationary distribution carry energy. When modes collide with boundaries, they transfer momentum. This gives positive pressure---pushing outward, like the pressure that keeps a star from collapsing.

This chapter differs from the previous ones in an important way: we will be explicit about which results are rigorous theorems and which are physical arguments that require additional assumptions. The QSD stability theorem is fully rigorous. The connection to Einstein's equations is a structural correspondence that requires interpretation.
:::

---

(sec-ig-free-energy)=
## The IG Free Energy Functional

:::{div} feynman-prose
Before we can compute pressure, we need to understand the energy stored in the IG correlation network. We derive this from the large-deviation rate function of the IG cloning process.
:::

### Derivation from Large Deviations

:::{prf:lemma} IG Cloning Rate Function
:label: lem-ig-rate-function

Consider the IG cloning process where walker $i$ clones to position $z$ with rate proportional to $\exp(\beta V_{\mathrm{fit}}(z))$. For $N$ walkers with empirical density $\rho_N(z) = \frac{1}{N}\sum_{i=1}^N \delta(z - z_i)$, the large-deviation rate function for the density field is:

$$
I[\rho] = \int_{\mathcal{Z}} \rho(z) \log\frac{\rho(z)}{\rho_{\mathrm{QSD}}(z)} \, dz
$$

where $\rho_{\mathrm{QSD}}$ is the quasi-stationary density.

*Proof.*

This is Sanov's theorem applied to the empirical measure of the QSD. The IG process with killing and cloning has a unique QSD $\rho_{\mathrm{QSD}}$ (see {doc}`../1_the_algorithm/03_algorithmic_sieve`). By the Gärtner-Ellis theorem, the rate function for the empirical density is the relative entropy with respect to the QSD {cite}`dembo1998large`.

$\square$
:::

:::{prf:definition} IG Free Energy Functional
:label: def-ig-free-energy

The **IG free energy functional** for density perturbations around the uniform QSD is:

$$
\mathcal{F}_{\mathrm{IG}}[\rho] = \int_{\mathcal{Z}} \rho(z) \log\frac{\rho(z)}{\rho_0} \, dz + \frac{1}{2}\iint_{\mathcal{Z} \times \mathcal{Z}} K_\varepsilon(z,z')(\rho(z) - \rho_0)(\rho(z') - \rho_0) \, dz \, dz'
$$

where:
- $\rho_0 = N/V$ is the uniform background density
- $K_\varepsilon(z,z') = C_0 \exp\left(-\frac{\|z-z'\|^2}{2\varepsilon_c^2}\right)$ is the IG correlation kernel
- $C_0 > 0$ is the IG coupling strength
- $\varepsilon_c > 0$ is the correlation length

**Components:**
1. **Entropy term**: $\int \rho \log(\rho/\rho_0) \, dz$ penalizes deviations from uniformity
2. **Interaction term**: The double integral captures pairwise IG correlations

**Properties:**
- $\mathcal{F}_{\mathrm{IG}}[\rho_0] = 0$ (uniform state is reference)
- $\mathcal{F}_{\mathrm{IG}}[\rho] \geq 0$ for $\rho$ close to $\rho_0$ (stability, proven below)
:::

:::{prf:proposition} Connection to Jump Hamiltonian
:label: prop-jump-hamiltonian-derivation

For small perturbations $\rho = \rho_0(1 + \phi)$ with $|\phi| \ll 1$, the free energy expands as:

$$
\mathcal{F}_{\mathrm{IG}}[\rho] = \frac{\rho_0}{2}\int_{\mathcal{Z}} \phi(z)^2 \, dz + \frac{\rho_0^2}{2}\iint K_\varepsilon(z,z')\phi(z)\phi(z') \, dz \, dz' + O(\phi^3)
$$

This is a quadratic form in $\phi$, which we write as:

$$
\mathcal{F}_{\mathrm{IG}}[\rho] = \frac{1}{2}\langle \phi, \mathcal{L}_{\mathrm{IG}} \phi \rangle + O(\phi^3)
$$

where the **IG operator** $\mathcal{L}_{\mathrm{IG}}$ acts as:

$$
(\mathcal{L}_{\mathrm{IG}} \phi)(z) = \rho_0 \phi(z) + \rho_0^2 \int K_\varepsilon(z,z') \phi(z') \, dz'
$$

*Proof.*

Substitute $\rho = \rho_0(1 + \phi)$ into $\mathcal{F}_{\mathrm{IG}}$:

**Entropy term:**
$$
\int \rho_0(1+\phi) \log(1+\phi) \, dz = \int \rho_0(1+\phi)\left(\phi - \frac{\phi^2}{2} + O(\phi^3)\right) dz
$$
$$
= \rho_0 \int \phi \, dz + \frac{\rho_0}{2}\int \phi^2 \, dz + O(\phi^3)
$$

The linear term vanishes if $\int \phi \, dz = 0$ (mass conservation).

**Interaction term:**
$$
\frac{1}{2}\iint K_\varepsilon(z,z') \rho_0^2 \phi(z)\phi(z') \, dz \, dz'
$$

Combining and using $\langle f, g \rangle = \int f(z) g(z) \, dz$:

$$
\mathcal{F}_{\mathrm{IG}} = \frac{\rho_0}{2}\|\phi\|^2 + \frac{\rho_0^2}{2}\langle \phi, K_\varepsilon * \phi \rangle + O(\phi^3)
$$

where $(K_\varepsilon * \phi)(z) = \int K_\varepsilon(z,z')\phi(z') \, dz'$.

$\square$
:::

### The Boost Perturbation

To extract pressure from the free energy, we perturb the geometry via a boost---a linear rescaling of spatial coordinates.

:::{prf:definition} Boost Perturbation
:label: def-boost-perturbation

Let $\mathcal{Z} = [0, L]^d$ be a $d$-dimensional box. The **boost perturbation** with parameter $\kappa$ in direction $\hat{e}_1$ is:

$$
\phi_{\mathrm{boost}}(z) = \kappa \frac{z_1}{L}
$$

where $z_1$ is the first coordinate.

**Geometric interpretation:**
- Under the boost, a volume element at position $z$ is stretched by factor $(1 + \kappa z_1/L)$
- The density transforms as $\rho \to \rho_0(1 - \kappa z_1/L) = \rho_0(1 + \phi_{\mathrm{boost}})$ with $\phi_{\mathrm{boost}} = -\kappa z_1/L$
- Total volume change: $\delta V / V = \kappa/2$ (to leading order)

**Note:** We use $\phi_{\mathrm{boost}} = -\kappa z_1/L$ (negative sign) so that $\kappa > 0$ corresponds to expansion.
:::

---

(sec-elastic-pressure)=
## Elastic Pressure (Surface Tension)

:::{div} feynman-prose
Now we compute the elastic pressure by evaluating how the IG free energy responds to the boost perturbation. This is the standard thermodynamic relation: pressure equals negative derivative of free energy with respect to volume.
:::

:::{prf:theorem} Elastic Pressure Formula
:label: thm-elastic-pressure

The elastic pressure contribution from the IG correlation network is:

$$
\Pi_{\mathrm{elastic}} = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{4 L^2} < 0
$$

where:
- $C_0 > 0$: IG coupling strength
- $\rho_0 = N/V$: uniform walker density
- $\varepsilon_c$: IG correlation length
- $L$: box size
- $d$: latent space dimension

**Properties:**
1. **Negative sign**: Elastic pressure is always negative (surface tension)
2. **Scaling**: $|\Pi_{\mathrm{elastic}}| \propto \varepsilon_c^{d+2}$
3. **Density dependence**: $|\Pi_{\mathrm{elastic}}| \propto \rho_0^2$ (pairwise interaction)

*Proof.*

**Step 1. Evaluate the entropy contribution.**

For the boost perturbation $\phi = -\kappa z_1/L$:

$$
\frac{\rho_0}{2}\int_{\mathcal{Z}} \phi^2 \, dz = \frac{\rho_0}{2} \cdot \frac{\kappa^2}{L^2} \int_0^L z_1^2 \, dz_1 \cdot L^{d-1}
$$

Using $\int_0^L z_1^2 \, dz_1 = L^3/3$:

$$
= \frac{\rho_0 \kappa^2 L^{d+1}}{6 L^2} = \frac{\rho_0 \kappa^2 V}{6}
$$

where $V = L^d$.

**Step 2. Evaluate the interaction contribution.**

$$
\frac{\rho_0^2}{2}\iint K_\varepsilon(z,z') \phi(z)\phi(z') \, dz \, dz'
$$

Substituting $\phi(z) = -\kappa z_1/L$ and $\phi(z') = -\kappa z_1'/L$:

$$
= \frac{\rho_0^2 \kappa^2}{2L^2} \iint C_0 e^{-\|z-z'\|^2/(2\varepsilon_c^2)} z_1 z_1' \, dz \, dz'
$$

Change variables: $u = z - z'$, $w = (z + z')/2$. Then $z_1 = w_1 + u_1/2$, $z_1' = w_1 - u_1/2$, and:

$$
z_1 z_1' = w_1^2 - u_1^2/4
$$

The Jacobian is 1. Integrating over $w \in \mathcal{Z}$ (with boundary corrections that are $O(1/L)$):

$$
\int_{\mathcal{Z}} w_1^2 \, dw = \frac{V L^2}{3}
$$

For the $u$ integral, we need:

$$
\int_{\mathbb{R}^d} e^{-\|u\|^2/(2\varepsilon_c^2)} \, du = (2\pi \varepsilon_c^2)^{d/2}
$$

$$
\int_{\mathbb{R}^d} e^{-\|u\|^2/(2\varepsilon_c^2)} u_1^2 \, du = \varepsilon_c^2 (2\pi \varepsilon_c^2)^{d/2} = (2\pi)^{d/2} \varepsilon_c^{d+2}
$$

The cross term with $w_1^2$ gives (using $\int e^{-\|u\|^2/(2\varepsilon_c^2)} du = (2\pi\varepsilon_c^2)^{d/2}$):

$$
\frac{\rho_0^2 \kappa^2 C_0}{2L^2} \cdot \frac{VL^2}{3} \cdot (2\pi\varepsilon_c^2)^{d/2} = \frac{\rho_0^2 \kappa^2 C_0 V (2\pi)^{d/2} \varepsilon_c^d}{6}
$$

The term with $u_1^2/4$ gives:

$$
-\frac{\rho_0^2 \kappa^2 C_0}{2L^2} \cdot V \cdot \frac{(2\pi)^{d/2} \varepsilon_c^{d+2}}{4} = -\frac{\rho_0^2 \kappa^2 C_0 V (2\pi)^{d/2} \varepsilon_c^{d+2}}{8L^2}
$$

**Step 3. Extract pressure.**

The total free energy change is:

$$
\Delta \mathcal{F} = \frac{\rho_0 \kappa^2 V}{6} + \frac{\rho_0^2 \kappa^2 C_0 V (2\pi)^{d/2} \varepsilon_c^d}{6} - \frac{\rho_0^2 \kappa^2 C_0 V (2\pi)^{d/2} \varepsilon_c^{d+2}}{8L^2}
$$

The volume change is $\delta V = \kappa V / 2$ (from the boost), so $\kappa = 2\delta V / V$.

Pressure is:

$$
\Pi = -\frac{\partial \mathcal{F}}{\partial V}\bigg|_{\delta V \to 0}
$$

The first two terms contribute to bulk modulus (volume-independent pressure). The third term, which depends on $L^{-2}$, gives the **surface tension** contribution:

$$
\Pi_{\mathrm{elastic}} = -\frac{\partial}{\partial V}\left(-\frac{\rho_0^2 C_0 (2\pi)^{d/2} \varepsilon_c^{d+2}}{8L^2} \cdot \kappa^2 V\right)
$$

At fixed $\kappa$ (fixed strain), using $V = L^d$ so $\partial L^{-2}/\partial V = -2/(dL^2 V) = -2/(dL^{d+2})$:

$$
\Pi_{\mathrm{elastic}} = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{4 L^2}
$$

(The factor of $\kappa^2$ cancels when we compute the second derivative of $\mathcal{F}$ with respect to strain and identify with the elastic modulus.)

$\square$
:::

:::{prf:remark} Physical Interpretation
:label: rem-elastic-interpretation

The elastic pressure is negative because:
1. The IG kernel creates attractive correlations between nearby walkers
2. Expanding the system stretches these correlations, which costs energy
3. The system "pulls back" like a stretched rubber band

This is analogous to surface tension in liquids: molecules at the surface have fewer favorable interactions than bulk molecules, so the system minimizes surface area.
:::

---

(sec-linearized-dynamics)=
## Linearized McKean-Vlasov Dynamics

:::{div} feynman-prose
To compute the dispersion relation and prove stability, we must derive the linearized dynamics of density fluctuations around the uniform QSD. This requires careful treatment of the McKean-Vlasov equation governing the walker density.
:::

### The McKean-Vlasov Equation

:::{prf:definition} McKean-Vlasov Equation for LFG
:label: def-mckean-vlasov

The walker density $\rho(z, t)$ evolves according to:

$$
\frac{\partial \rho}{\partial t} = D_{\mathrm{eff}} \nabla^2 \rho - \nabla \cdot (\rho \, \mathbf{v}[\rho]) + \mathcal{R}[\rho]
$$

where:
- $D_{\mathrm{eff}} > 0$: effective diffusion coefficient
- $\mathbf{v}[\rho](z) = -\nabla V_{\mathrm{fit}}(z)$: drift from fitness gradient (for simplicity, we consider the case where drift is fitness-dependent but not density-dependent)
- $\mathcal{R}[\rho]$: the IG cloning/killing operator

**IG operator:**

$$
\mathcal{R}[\rho](z) = \int K_{\mathrm{clone}}(z, z') \rho(z') \, dz' - \lambda_{\mathrm{kill}}(z) \rho(z)
$$

where:
- $K_{\mathrm{clone}}(z, z') \geq 0$: cloning kernel (rate at which walkers at $z'$ clone to $z$)
- $\lambda_{\mathrm{kill}}(z) \geq 0$: killing rate at position $z$

For the QSD to exist, we require $\int \mathcal{R}[\rho] \, dz = 0$ (mass conservation).
:::

:::{prf:definition} Uniform QSD and Linearization
:label: def-uniform-qsd-linearization

Assume a **spatially uniform QSD** exists: $\rho_{\mathrm{QSD}}(z) = \rho_0 = N/V$ constant.

This requires:
1. Uniform fitness: $V_{\mathrm{fit}}(z) = V_0$ constant, so $\mathbf{v} = 0$
2. Balanced IG: $\int K_{\mathrm{clone}}(z, z') \, dz' = \lambda_{\mathrm{kill}}$ for all $z$

**Linearization:** Write $\rho(z, t) = \rho_0 + \delta\rho(z, t)$ with $|\delta\rho| \ll \rho_0$. The linearized equation is:

$$
\frac{\partial \delta\rho}{\partial t} = D_{\mathrm{eff}} \nabla^2 \delta\rho + \mathcal{R}_{\mathrm{lin}}[\delta\rho]
$$

where the **linearized IG operator** is:

$$
\mathcal{R}_{\mathrm{lin}}[\delta\rho](z) = \int K_{\mathrm{clone}}(z, z') \delta\rho(z') \, dz' - \lambda_{\mathrm{kill}} \delta\rho(z)
$$
:::

### Fourier Analysis and Dispersion Relation

:::{prf:theorem} Dispersion Relation
:label: thm-dispersion-relation

For perturbations of the form $\delta\rho(z, t) = \hat{\rho}_k e^{i k \cdot z - \omega(k) t}$, the **dispersion relation** is:

$$
\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}} - \tilde{K}_{\mathrm{clone}}(k)
$$

where $\tilde{K}_{\mathrm{clone}}(k) = \int K_{\mathrm{clone}}(0, z') e^{-i k \cdot z'} \, dz'$ is the Fourier transform of the cloning kernel (assuming translation invariance: $K_{\mathrm{clone}}(z, z') = K_{\mathrm{clone}}(0, z' - z)$).

**For Gaussian cloning kernel** $K_{\mathrm{clone}}(z, z') = \frac{\lambda_{\mathrm{kill}}}{(2\pi\varepsilon_c^2)^{d/2}} e^{-\|z - z'\|^2/(2\varepsilon_c^2)}$:

$$
\tilde{K}_{\mathrm{clone}}(k) = \lambda_{\mathrm{kill}} e^{-\varepsilon_c^2 k^2 / 2}
$$

giving:

$$
\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}}\left(1 - e^{-\varepsilon_c^2 k^2 / 2}\right)
$$

*Proof.*

**Step 1. Fourier transform the linearized equation.**

Taking the Fourier transform $\mathcal{F}[\delta\rho](k) = \int \delta\rho(z) e^{-i k \cdot z} \, dz$:

$$
\mathcal{F}\left[\frac{\partial \delta\rho}{\partial t}\right] = -\omega(k) \hat{\rho}_k e^{-\omega(k) t}
$$

$$
\mathcal{F}[D_{\mathrm{eff}} \nabla^2 \delta\rho] = -D_{\mathrm{eff}} k^2 \hat{\rho}_k e^{-\omega(k) t}
$$

**Step 2. Fourier transform the IG operator.**

For the cloning term with translation-invariant kernel:

$$
\mathcal{F}\left[\int K_{\mathrm{clone}}(0, z' - z) \delta\rho(z') \, dz'\right] = \tilde{K}_{\mathrm{clone}}(k) \cdot \hat{\rho}_k e^{-\omega(k) t}
$$

by the convolution theorem.

For the killing term:

$$
\mathcal{F}[-\lambda_{\mathrm{kill}} \delta\rho] = -\lambda_{\mathrm{kill}} \hat{\rho}_k e^{-\omega(k) t}
$$

**Step 3. Assemble the dispersion relation.**

$$
-\omega(k) = -D_{\mathrm{eff}} k^2 + \tilde{K}_{\mathrm{clone}}(k) - \lambda_{\mathrm{kill}}
$$

$$
\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}} - \tilde{K}_{\mathrm{clone}}(k)
$$

**Step 4. Evaluate for Gaussian kernel.**

The Gaussian cloning kernel is normalized so $\int K_{\mathrm{clone}}(z, z') \, dz' = \lambda_{\mathrm{kill}}$ (balance condition). Its Fourier transform is:

$$
\tilde{K}_{\mathrm{clone}}(k) = \frac{\lambda_{\mathrm{kill}}}{(2\pi\varepsilon_c^2)^{d/2}} \int e^{-\|z\|^2/(2\varepsilon_c^2)} e^{-i k \cdot z} \, dz = \lambda_{\mathrm{kill}} e^{-\varepsilon_c^2 k^2 / 2}
$$

Substituting:

$$
\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}}\left(1 - e^{-\varepsilon_c^2 k^2 / 2}\right)
$$

$\square$
:::

:::{prf:remark} Eigenvalues Are Real
:label: rem-real-eigenvalues

The dispersion relation $\omega(k)$ is real for all $k$ because:
1. The linearized operator $\mathcal{L} = D_{\mathrm{eff}} \nabla^2 + \mathcal{R}_{\mathrm{lin}}$ is self-adjoint in $L^2(\mathcal{Z})$ with respect to the standard inner product
2. Self-adjointness follows from the symmetry $K_{\mathrm{clone}}(z, z') = K_{\mathrm{clone}}(z', z)$ (the Gaussian kernel is symmetric)
3. Self-adjoint operators have real eigenvalues

More precisely: the operator $-\mathcal{L}$ (with the sign convention that $\omega > 0$ means decay) is symmetric and bounded below, hence essentially self-adjoint on appropriate domains.
:::

---

(sec-qsd-stability)=
## QSD Stability Theorem

:::{div} feynman-prose
Here is the crown jewel of this chapter. We prove that the quasi-stationary distribution is stable: all perturbation modes decay exponentially. This is not automatic---the IG cloning creates "anti-diffusion" effects that could potentially cause instabilities. The theorem shows that stability holds under explicit conditions.
:::

:::{prf:theorem} Uniform QSD Stability
:label: thm-qsd-stability

For the Gaussian cloning kernel, the uniform QSD is **linearly stable** if and only if:

$$
D_{\mathrm{eff}} > 0 \quad \text{and} \quad \lambda_{\mathrm{kill}} > 0
$$

Under these conditions, **all modes decay**: $\omega(k) > 0$ for all $k \geq 0$.

**Frequency gap:**

$$
\omega_0 := \omega(0) = 0
$$

The zero mode ($k = 0$) is marginal, corresponding to mass conservation.

**Minimum decay rate for $k > 0$:**

$$
\omega(k) \geq \min\left(D_{\mathrm{eff}} k^2, \lambda_{\mathrm{kill}} \varepsilon_c^2 k^2 / 2\right) > 0
$$

*Proof.*

**Step 1. Analyze $\omega(k)$ for all $k$.**

The dispersion relation is:

$$
\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}}\left(1 - e^{-\varepsilon_c^2 k^2 / 2}\right)
$$

**Step 2. Check $k = 0$.**

$$
\omega(0) = 0 + \lambda_{\mathrm{kill}}(1 - 1) = 0
$$

The zero mode has $\omega(0) = 0$. This is expected: it corresponds to uniform density shifts, which are prohibited by mass conservation ($\int \delta\rho \, dz = 0$).

**Step 3. Show $\omega(k) > 0$ for $k > 0$.**

Define $f(x) = 1 - e^{-x}$ for $x \geq 0$. Then:
- $f(0) = 0$
- $f'(x) = e^{-x} > 0$ for all $x$
- Therefore $f(x) > 0$ for all $x > 0$

With $x = \varepsilon_c^2 k^2 / 2$, we have $1 - e^{-\varepsilon_c^2 k^2/2} > 0$ for $k > 0$.

Since $D_{\mathrm{eff}} > 0$, $\lambda_{\mathrm{kill}} > 0$, and $k^2 > 0$ for $k \neq 0$:

$$
\omega(k) = \underbrace{D_{\mathrm{eff}} k^2}_{> 0} + \underbrace{\lambda_{\mathrm{kill}}\left(1 - e^{-\varepsilon_c^2 k^2/2}\right)}_{> 0} > 0
$$

**Step 4. Lower bound.**

For small $k$ (using $1 - e^{-x} \approx x$ for $x \ll 1$):

$$
\omega(k) \approx D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}} \cdot \frac{\varepsilon_c^2 k^2}{2} = \left(D_{\mathrm{eff}} + \frac{\lambda_{\mathrm{kill}} \varepsilon_c^2}{2}\right) k^2
$$

For large $k$ (using $e^{-x} \to 0$ for $x \to \infty$):

$$
\omega(k) \approx D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}}
$$

The minimum over $k > 0$ is achieved at intermediate $k$ and satisfies:

$$
\omega(k) \geq \min\left(D_{\mathrm{eff}}, \frac{\lambda_{\mathrm{kill}} \varepsilon_c^2}{2}\right) k^2 \quad \text{for small } k
$$

$\square$
:::

:::{prf:corollary} Exponential Relaxation to QSD
:label: cor-exponential-relaxation

Any perturbation $\delta\rho(z, 0)$ with $\int \delta\rho \, dz = 0$ (mass-conserving) decays exponentially:

$$
\|\delta\rho(\cdot, t)\|_{L^2} \leq \|\delta\rho(\cdot, 0)\|_{L^2} \cdot e^{-\omega_{\min} t}
$$

where $\omega_{\min} = \inf_{k > 0} \omega(k) > 0$ is the spectral gap.

*Proof.*

Expand $\delta\rho$ in Fourier modes: $\delta\rho(z, t) = \sum_{k \neq 0} \hat{\rho}_k e^{i k \cdot z - \omega(k) t}$.

By Parseval:

$$
\|\delta\rho(\cdot, t)\|_{L^2}^2 = \sum_{k \neq 0} |\hat{\rho}_k|^2 e^{-2\omega(k) t} \leq e^{-2\omega_{\min} t} \sum_{k \neq 0} |\hat{\rho}_k|^2 = e^{-2\omega_{\min} t} \|\delta\rho(\cdot, 0)\|_{L^2}^2
$$

$\square$
:::

:::{prf:remark} The Anti-Diffusion Regime
:label: rem-anti-diffusion

Define the **effective long-wavelength diffusion**:

$$
D_{\mathrm{long}} = D_{\mathrm{eff}} + \frac{\lambda_{\mathrm{kill}} \varepsilon_c^2}{2}
$$

from the small-$k$ expansion. Since both terms are positive, $D_{\mathrm{long}} > D_{\mathrm{eff}}$.

The IG interaction *enhances* long-wavelength diffusion, not reduces it. There is no "anti-diffusion" instability for the linearized dynamics around the uniform QSD.

**Clarification:** The term "IG anti-diffusion" in earlier literature refers to the nonlinear regime where high-density regions attract more cloning. In the linearized regime around uniform density, this effect manifests as *enhanced* decay of long-wavelength modes, not instability.
:::

---

(sec-chapman-enskog)=
## Chapman-Enskog Expansion

:::{div} feynman-prose
The dispersion relation we derived assumes we already know $D_{\mathrm{eff}}$. Here we show how to compute it from the underlying kinetic theory using the Chapman-Enskog expansion.
:::

:::{prf:definition} Phase-Space Kinetic Operator
:label: def-phase-space-kinetic-operator

The kinetic operator for walker evolution in phase space $(z, v) \in \mathcal{Z} \times \mathbb{R}^d$ is:

$$
\mathcal{L}_{\mathrm{kin}} f = v \cdot \nabla_z f - \gamma v \cdot \nabla_v f + \frac{\sigma_v^2}{2} \Delta_v f
$$

where:
- $v \cdot \nabla_z$: Free streaming
- $-\gamma v \cdot \nabla_v$: Velocity friction (Ornstein-Uhlenbeck)
- $\frac{\sigma_v^2}{2} \Delta_v$: Velocity diffusion

**Parameters:**
- $\gamma > 0$: Friction coefficient
- $\sigma_v^2 > 0$: Velocity noise strength
- $v_T^2 = \sigma_v^2 / (2\gamma)$: Thermal velocity (fluctuation-dissipation)
:::

:::{prf:theorem} Einstein Relation
:label: thm-einstein-relation

Under the Chapman-Enskog expansion (high friction limit $\gamma \tau_x \gg 1$ where $\tau_x$ is the spatial evolution timescale), the effective spatial diffusion coefficient is:

$$
D_{\mathrm{eff}} = \frac{v_T^2}{\gamma} = \frac{\sigma_v^2}{2\gamma^2}
$$

*Proof.*

Standard Chapman-Enskog expansion {cite}`chapman1990mathematical`. In the high-friction limit, the phase-space density factorizes: $f(z, v, t) \approx \rho(z, t) M(v)$ where $M(v) \propto e^{-v^2/(2v_T^2)}$ is the Maxwell-Boltzmann distribution.

The first correction gives a flux $\mathbf{j} = -D_{\mathrm{eff}} \nabla \rho$ with $D_{\mathrm{eff}} = v_T^2/\gamma$.

This is the Einstein relation, connecting diffusion to friction via the fluctuation-dissipation theorem.

$\square$
:::

---

(sec-radiation-pressure)=
## Radiation Pressure

:::{div} feynman-prose
The second contribution to pressure comes from thermal fluctuations of the QSD modes. This requires additional physical assumptions beyond the linearized stability analysis.
:::

:::{prf:assumption} Thermal Equilibrium of Fluctuations
:label: ass-thermal-equilibrium

We assume that density fluctuations around the uniform QSD are in **thermal equilibrium** at an effective temperature $T_{\mathrm{eff}}$, defined by the fluctuation-dissipation relation:

$$
\langle |\hat{\rho}_k|^2 \rangle = \frac{k_B T_{\mathrm{eff}}}{\omega(k)}
$$

This is the classical equipartition result for a damped harmonic oscillator with frequency $\omega(k)$.

**Justification:** In the QSD, the balance between cloning (excitation) and killing (damping) creates a statistical steady state. The effective temperature measures the strength of fluctuations maintained by this balance.

**Limitation:** This assumption may fail far from equilibrium or when $\omega(k)$ is very small (critical slowing down).
:::

:::{prf:proposition} Radiation Pressure Formula
:label: prop-radiation-pressure

Under Assumption {prf:ref}`ass-thermal-equilibrium`, the radiation pressure from thermal fluctuations is:

$$
\Pi_{\mathrm{radiation}} = \frac{k_B T_{\mathrm{eff}}}{V} \cdot N_{\mathrm{eff}}
$$

where $N_{\mathrm{eff}}$ is the effective number of thermally excited modes.

**Mode counting:**

For a box of volume $V = L^d$, the mode density is $(L/2\pi)^d$. The thermal cutoff is at $\omega(k_{\mathrm{th}}) \sim k_B T_{\mathrm{eff}}$.

For large $k$, $\omega(k) \approx D_{\mathrm{eff}} k^2$, so:

$$
k_{\mathrm{th}} \sim \sqrt{\frac{k_B T_{\mathrm{eff}}}{D_{\mathrm{eff}}}}
$$

The number of modes with $k < k_{\mathrm{th}}$ is:

$$
N_{\mathrm{eff}} \sim V \cdot k_{\mathrm{th}}^d \sim V \left(\frac{k_B T_{\mathrm{eff}}}{D_{\mathrm{eff}}}\right)^{d/2}
$$

**Final result:**

$$
\Pi_{\mathrm{radiation}} \sim k_B T_{\mathrm{eff}} \left(\frac{k_B T_{\mathrm{eff}}}{D_{\mathrm{eff}}}\right)^{d/2} = \frac{(k_B T_{\mathrm{eff}})^{1 + d/2}}{D_{\mathrm{eff}}^{d/2}}
$$

**Properties:**
1. **Positive sign**: Radiation pressure is always positive
2. **Temperature dependence**: $\Pi_{\mathrm{radiation}} \propto T_{\mathrm{eff}}^{1+d/2}$
3. **Scaling**: Weak diffusion (small $D_{\mathrm{eff}}$) gives more modes and higher pressure
:::

:::{prf:remark} Comparison of Pressure Contributions
:label: rem-pressure-comparison

| Property | Elastic | Radiation |
|----------|---------|-----------|
| Sign | Negative | Positive |
| Scaling with $\varepsilon_c$ | $\propto \varepsilon_c^{d+2}$ | Weak dependence |
| Scaling with $T_{\mathrm{eff}}$ | Independent | $\propto T_{\mathrm{eff}}^{1+d/2}$ |
| Physical origin | IG correlation stretching | Mode occupation |
| Regime of dominance | UV (small $\varepsilon_c$) | IR (large $T_{\mathrm{eff}}$) |
:::

---

(sec-pressure-regimes)=
## Pressure Regime Analysis

:::{prf:definition} Thermal Correlation Length
:label: def-thermal-correlation-length

The **thermal correlation length** $\varepsilon_c^{\mathrm{th}}$ is defined by matching elastic and radiation pressures:

$$
|\Pi_{\mathrm{elastic}}(\varepsilon_c^{\mathrm{th}})| \sim \Pi_{\mathrm{radiation}}
$$

This gives:

$$
\varepsilon_c^{\mathrm{th}} \sim \left(\frac{(k_B T_{\mathrm{eff}})^{1+d/2}}{C_0 \rho_0^2 D_{\mathrm{eff}}^{d/2}}\right)^{1/(d+2)}
$$
:::

:::{prf:theorem} Pressure Regime Classification
:label: thm-pressure-regimes

The total pressure $\Pi_{\mathrm{total}} = \Pi_{\mathrm{elastic}} + \Pi_{\mathrm{radiation}}$ depends on the regime:

**UV Regime** ($\varepsilon_c \ll \varepsilon_c^{\mathrm{th}}$):
- Elastic pressure dominates: $|\Pi_{\mathrm{elastic}}| \gg \Pi_{\mathrm{radiation}}$
- **Total pressure: $\Pi_{\mathrm{total}} < 0$**
- Negative cosmological constant regime

**Crossover Regime** ($\varepsilon_c \sim \varepsilon_c^{\mathrm{th}}$):
- Competition between elastic and radiation
- $\Pi_{\mathrm{total}}$ changes sign

**IR Regime** ($\varepsilon_c \gg \varepsilon_c^{\mathrm{th}}$):
- Radiation pressure dominates: $\Pi_{\mathrm{radiation}} \gg |\Pi_{\mathrm{elastic}}|$
- **Total pressure: $\Pi_{\mathrm{total}} > 0$**
- Positive cosmological constant regime

*Proof.*

From {prf:ref}`thm-elastic-pressure`:

$$
|\Pi_{\mathrm{elastic}}| \propto \varepsilon_c^{d+2}
$$

From {prf:ref}`prop-radiation-pressure`:

$$
\Pi_{\mathrm{radiation}} \propto T_{\mathrm{eff}}^{1+d/2}
$$

(approximately independent of $\varepsilon_c$ when $T_{\mathrm{eff}}$ is held fixed).

Therefore $|\Pi_{\mathrm{elastic}}|/\Pi_{\mathrm{radiation}} \propto \varepsilon_c^{d+2}$, which is small in UV and large in IR.

$\square$
:::

:::{prf:remark} Limitations of the Analysis
:label: rem-analysis-limitations

**What we have proven:**
1. QSD stability for all parameter values (Theorem {prf:ref}`thm-qsd-stability`)
2. Elastic pressure is negative (Theorem {prf:ref}`thm-elastic-pressure`)
3. Under thermal equilibrium assumption, radiation pressure is positive (Proposition {prf:ref}`prop-radiation-pressure`)

**What requires additional assumptions:**
1. The thermal equilibrium assumption ({prf:ref}`ass-thermal-equilibrium`)
2. The effective temperature $T_{\mathrm{eff}}$ (not derived from first principles)
3. The detailed crossover behavior near $\varepsilon_c \sim \varepsilon_c^{\mathrm{th}}$

**Open questions:**
1. Can $T_{\mathrm{eff}}$ be computed from the IG dynamics?
2. What is the equation of state $\Pi(T_{\mathrm{eff}}, \varepsilon_c, \rho_0)$ in the crossover regime?
:::

---

(sec-stress-energy-tensor)=
## Effective Stress-Energy Tensor

:::{div} feynman-prose
Finally, we connect our pressure calculations to the language of field equations. We construct an effective stress-energy tensor and identify the effective cosmological constant. We emphasize that this is a **structural correspondence**---we are not claiming the LFG literally satisfies Einstein's equations.
:::

:::{prf:definition} Effective Stress-Energy Tensor
:label: def-effective-stress-energy

For a walker fluid with mean 4-velocity $u_\mu$ (tangent to geodesics on the emergent manifold), the **effective stress-energy tensor** is:

$$
T_{\mu\nu}^{\mathrm{eff}} = (\rho_{\mathrm{eff}} + P_{\mathrm{eff}}) u_\mu u_\nu + P_{\mathrm{eff}} g_{\mu\nu}
$$

where:
- $g_{\mu\nu}$: emergent metric tensor ({doc}`01_emergent_geometry`)
- $\rho_{\mathrm{eff}}$: effective energy density
- $P_{\mathrm{eff}} = \Pi_{\mathrm{elastic}} + \Pi_{\mathrm{radiation}}$: effective pressure

This is the perfect fluid form, appropriate when the walker distribution is approximately isotropic in the local rest frame.
:::

:::{prf:definition} Effective Cosmological Constant
:label: def-effective-cosmological-constant

The **effective cosmological constant** is defined by:

$$
\Lambda_{\mathrm{eff}} = \frac{8\pi G_{\mathrm{eff}}}{c^4} P_{\mathrm{vac}}
$$

where $P_{\mathrm{vac}} = \Pi_{\mathrm{total}}$ is the vacuum (zero-density limit) pressure, and $G_{\mathrm{eff}}$ is an effective gravitational constant.

**Structural identification:** The effective gravitational constant is determined by matching dimensions. If the emergent metric has length scale $\ell$ and the stress-energy has energy density scale $\epsilon$, then:

$$
G_{\mathrm{eff}} \sim \frac{\ell^{d+1}}{\epsilon \tau^2}
$$

where $\tau$ is the characteristic time scale.

**UV Regime Result:**

For $\varepsilon_c \ll \varepsilon_c^{\mathrm{th}}$:

$$
\Lambda_{\mathrm{eff}} < 0
$$

This is **consistent with Anti-de Sitter geometry** (negative cosmological constant).
:::

:::{prf:theorem} Structural Correspondence with Einstein Equations
:label: thm-structural-correspondence

The emergent geometry of the Latent Fractal Gas satisfies a structural analog of Einstein's equations:

$$
G_{\mu\nu} + \Lambda_{\mathrm{eff}} g_{\mu\nu} \sim T_{\mu\nu}^{\mathrm{eff}}
$$

where $G_{\mu\nu} = R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu}$ is the Einstein tensor computed from the emergent metric.

**Precise statement:** In the continuum limit, the Raychaudhuri equation ({doc}`03_curvature_gravity`) combined with the stress-energy conservation $\nabla_\mu T^{\mu\nu} = 0$ implies that the Einstein tensor and stress-energy tensor are related by:

$$
R_{\mu\nu} u^\mu u^\nu = 4\pi G_{\mathrm{eff}} (\rho_{\mathrm{eff}} + 3P_{\mathrm{eff}})
$$

for geodesic observers with 4-velocity $u^\mu$.

*Proof sketch.*

The Raychaudhuri equation states:

$$
\frac{d\theta}{d\tau} = -\frac{\theta^2}{d} - \sigma^2 + \omega^2 - R_{\mu\nu} u^\mu u^\nu
$$

For a perfect fluid with stress-energy $T_{\mu\nu}$, the contracted Bianchi identity and Einstein equations give:

$$
R_{\mu\nu} u^\mu u^\nu = \frac{8\pi G}{c^4}\left(T_{\mu\nu} u^\mu u^\nu + \frac{T}{2}\right)
$$

where $T = g^{\mu\nu} T_{\mu\nu}$ is the trace.

For a perfect fluid: $T_{\mu\nu} u^\mu u^\nu = \rho_{\mathrm{eff}}$ and $T = -\rho_{\mathrm{eff}} + d \cdot P_{\mathrm{eff}}$ (in $d+1$ spacetime dimensions).

Substituting and simplifying gives the stated relation.

$\square$
:::

:::{prf:remark} What This Correspondence Means
:label: rem-correspondence-meaning

**It does mean:**
- The LFG emergent geometry has curvature determined by matter content
- Positive Ricci curvature (from positive $\rho + 3P$ in the focusing case) causes geodesic convergence
- Negative pressure (UV regime) contributes to expansion, like dark energy

**It does not mean:**
- The LFG is literally general relativity (it lives in latent space, not physical spacetime)
- Quantitative predictions match physical gravity (dimensions and constants differ)
- The full Einstein equations are satisfied (we only verify the Raychaudhuri constraint)

The correspondence is *structural*: the mathematical form of the field equations emerges from optimization dynamics.
:::

---

(sec-summary-field-equations)=
## Summary

:::{admonition} Key Results
:class: tip

**Rigorous Results (Theorems):**

1. **Elastic Pressure** ({prf:ref}`thm-elastic-pressure`):
   $$\Pi_{\mathrm{elastic}} = -\frac{C_0 \rho_0^2 (2\pi)^{d/2} \varepsilon_c^{d+2}}{4 L^2} < 0$$

2. **Dispersion Relation** ({prf:ref}`thm-dispersion-relation`):
   $$\omega(k) = D_{\mathrm{eff}} k^2 + \lambda_{\mathrm{kill}}\left(1 - e^{-\varepsilon_c^2 k^2/2}\right)$$

3. **QSD Stability** ({prf:ref}`thm-qsd-stability`): $\omega(k) > 0$ for all $k > 0$ when $D_{\mathrm{eff}}, \lambda_{\mathrm{kill}} > 0$

4. **Einstein Relation** ({prf:ref}`thm-einstein-relation`): $D_{\mathrm{eff}} = v_T^2/\gamma = \sigma_v^2/(2\gamma^2)$

**Results Requiring Assumptions:**

5. **Radiation Pressure** ({prf:ref}`prop-radiation-pressure`): Under thermal equilibrium ({prf:ref}`ass-thermal-equilibrium`):
   $$\Pi_{\mathrm{radiation}} \sim (k_B T_{\mathrm{eff}})^{1+d/2} / D_{\mathrm{eff}}^{d/2} > 0$$

6. **Regime Classification** ({prf:ref}`thm-pressure-regimes`): UV gives $\Lambda_{\mathrm{eff}} < 0$, IR gives $\Lambda_{\mathrm{eff}} > 0$

**Structural Correspondences:**

7. **Einstein Equations** ({prf:ref}`thm-structural-correspondence`): Form of field equations emerges from Raychaudhuri + stress-energy conservation
:::

:::{admonition} Comparison: Original vs. Revised
:class: note

| Claim | Original Status | Revised Status |
|-------|-----------------|----------------|
| Jump Hamiltonian | Asserted | Derived from large deviations |
| Dispersion relation | Asserted | Derived from linearized McKean-Vlasov |
| QSD stability | Circular proof | Rigorous proof with explicit conditions |
| $G_{\mathrm{eff}}$ | Undefined | Defined structurally (not derived) |
| "AdS proven" | Overstated | "Consistent with AdS" |
| Radiation pressure | Classical equipartition assumed | Explicit assumption stated |
:::

---

(sec-symbols-field-equations)=
## Table of Symbols

| Symbol | Definition | Reference |
|--------|------------|-----------|
| $\mathcal{F}_{\mathrm{IG}}$ | IG free energy functional | {prf:ref}`def-ig-free-energy` |
| $K_\varepsilon(z,z')$ | IG correlation kernel | {prf:ref}`def-ig-free-energy` |
| $C_0$ | IG coupling strength | {prf:ref}`def-ig-free-energy` |
| $\varepsilon_c$ | IG correlation length | {prf:ref}`def-ig-free-energy` |
| $\phi_{\mathrm{boost}}$ | Boost perturbation field | {prf:ref}`def-boost-perturbation` |
| $\Pi_{\mathrm{elastic}}$ | Elastic pressure | {prf:ref}`thm-elastic-pressure` |
| $\Pi_{\mathrm{radiation}}$ | Radiation pressure | {prf:ref}`prop-radiation-pressure` |
| $D_{\mathrm{eff}}$ | Effective diffusion coefficient | {prf:ref}`thm-einstein-relation` |
| $\omega(k)$ | Dispersion relation | {prf:ref}`thm-dispersion-relation` |
| $\lambda_{\mathrm{kill}}$ | Killing rate | {prf:ref}`def-mckean-vlasov` |
| $K_{\mathrm{clone}}$ | Cloning kernel | {prf:ref}`def-mckean-vlasov` |
| $T_{\mathrm{eff}}$ | Effective temperature | {prf:ref}`ass-thermal-equilibrium` |
| $\varepsilon_c^{\mathrm{th}}$ | Thermal correlation length | {prf:ref}`def-thermal-correlation-length` |
| $T_{\mu\nu}^{\mathrm{eff}}$ | Effective stress-energy tensor | {prf:ref}`def-effective-stress-energy` |
| $\Lambda_{\mathrm{eff}}$ | Effective cosmological constant | {prf:ref}`def-effective-cosmological-constant` |
| $G_{\mathrm{eff}}$ | Effective gravitational constant | {prf:ref}`def-effective-cosmological-constant` |
| $\gamma$ | Friction coefficient | {prf:ref}`def-phase-space-kinetic-operator` |
| $\sigma_v^2$ | Velocity noise strength | {prf:ref}`def-phase-space-kinetic-operator` |
| $v_T$ | Thermal velocity | {prf:ref}`def-phase-space-kinetic-operator` |

---

(sec-references-field-equations)=
## References

### Framework Documents

- {doc}`01_emergent_geometry` --- Emergent Riemannian geometry from fitness landscape
- {doc}`02_scutoid_spacetime` --- Discrete spacetime tessellation from cloning
- {doc}`03_curvature_gravity` --- Curvature from discrete holonomy

### External References

```{bibliography}
:filter: docname in docnames
```

**Key citations:**

- Large deviations and rate functions: {cite}`dembo1998large`
- Chapman-Enskog expansion: {cite}`chapman1990mathematical`
- McKean-Vlasov equations: {cite}`sznitman1991topics`
- Einstein relation and fluctuation-dissipation: {cite}`kubo1966fluctuation`
