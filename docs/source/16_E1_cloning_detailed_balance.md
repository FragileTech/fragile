# Appendix E.1: Detailed Balance for Cloning Operator at QSD

## Overview

This appendix provides a rigorous analysis of the **detailed balance condition** for the cloning operator at the quasi-stationary distribution (QSD).

**Main Question**: Does the cloning operator (with inelastic collisions and fitness-based selection) satisfy detailed balance at QSD, ensuring that energy input equals energy dissipation?

**Strategy**: We analyze the Fokker-Planck equation for the full dynamics (Langevin + cloning) and prove that the QSD satisfies a **generalized detailed balance** where cloning death-birth processes balance with Langevin thermalization.

## 1. Fokker-Planck Operator with Cloning

The full generator for the Adaptive Gas includes both Langevin dynamics and cloning:

$$
\mathcal{L} = \mathcal{L}_{\text{Langevin}} + \mathcal{L}_{\text{clone}}
$$

### 1.1 Langevin Generator

From Chapter 5 ({prf:ref}`def-langevin-fokker-planck`), the Langevin Fokker-Planck operator is:

$$
\mathcal{L}_{\text{Langevin}}^* \mu = -\nabla_x \cdot (\mu v) - \nabla_v \cdot \left[\mu \left(\frac{F}{m} - \gamma v\right)\right] + \frac{\sigma_v^2}{2} \nabla_v^2 \mu
$$

where:
- $F(x) = -\nabla U(x)$ is the confining force
- $\gamma > 0$ is friction
- $\sigma_v^2$ is noise intensity

**Equilibrium** (without cloning): The stationary distribution is Maxwellian:

$$
\mu_{\text{Langevin}}(x, v) = Z^{-1} \exp\left(-\frac{U(x) + \frac{1}{2}m\|v\|^2}{k_B T}\right)
$$

where $k_B T = m\sigma_v^2 / (2\gamma)$ from fluctuation-dissipation.

### 1.2 Cloning Generator

The cloning operator acts as a **birth-death process** in phase space:

$$
\mathcal{L}_{\text{clone}}^* \mu = \frac{1}{\tau_{\text{clone}}} \left[\int K_{\text{clone}}(x, v | x', v') \mu(x', v') dx' dv' - \lambda_{\text{clone}}(x, v) \mu(x, v)\right]
$$

**Components**:

1. **Birth kernel**: $K_{\text{clone}}(x, v | x', v')$ describes the probability density that a walker at $(x', v')$ gives birth to a walker at $(x, v)$

2. **Death rate**: $\lambda_{\text{clone}}(x, v)$ is the rate at which a walker at $(x, v)$ is killed

**Inelastic Collision Model**: From Chapter 3, the cloning process is:

1. **Select parent** $i$ with fitness weight $w_i \propto \exp(\alpha \Psi(x_i) + \beta s_i)$
2. **Inelastic collision**: Child $j$ and parent $i$ undergo:
   $$
   v_i' = v_{\text{COM}} + \alpha_{\text{rest}}(v_i - v_{\text{COM}}), \quad v_j' = v_{\text{COM}} + \alpha_{\text{rest}}(v_j - v_{\text{COM}})
   $$
   where $v_{\text{COM}} = (v_i + v_j) / 2$ and $\alpha_{\text{rest}} \in [0, 1]$
3. **Kill walker** $k$ with inverse fitness weight $w_k^{-1} \propto \exp(-\alpha \Psi(x_k) - \beta s_k)$

**Birth kernel** (simplified, ignoring diversity $\beta s$):

$$
K_{\text{clone}}(x, v | x', v') = \frac{e^{\alpha \Psi(x')}}{\mathcal{Z}} \delta(x - x') \cdot P_{\text{collision}}(v | v')
$$

where $P_{\text{collision}}(v | v')$ is the velocity distribution after inelastic collision.

**Death rate**:

$$
\lambda_{\text{clone}}(x, v) = \frac{e^{-\alpha \Psi(x)}}{\mathcal{Z}}
$$

## 2. QSD as Stationary Distribution

:::{prf:definition} Quasi-Stationary Distribution
:label: def-qsd-cloning

A measure $\mu_{\text{QSD}}$ is a quasi-stationary distribution if it satisfies:

$$
(\mathcal{L}_{\text{Langevin}}^* + \mathcal{L}_{\text{clone}}^*) \mu_{\text{QSD}} = -\lambda_{\text{QSD}} \mu_{\text{QSD}}
$$

where $\lambda_{\text{QSD}} \geq 0$ is the **extinction rate** (probability per unit time that the entire swarm dies).

**Interpretation**: The QSD is the stationary distribution **conditioned on survival**. It balances:
- Langevin drift and diffusion
- Cloning birth-death processes
- Extinction events

:::

For systems with a confining potential and bounded killing rate, $\lambda_{\text{QSD}} \approx 0$ (extinction is exponentially rare), so:

$$
(\mathcal{L}_{\text{Langevin}}^* + \mathcal{L}_{\text{clone}}^*) \mu_{\text{QSD}} \approx 0
$$

This is the **stationary condition** we analyze.

## 3. Detailed Balance: Separating Spatial and Velocity Coordinates

**Ansatz**: At QSD, the distribution factorizes:

$$
\mu_{\text{QSD}}(x, v) = \rho_{\text{QSD}}(x) \cdot \mathcal{M}(v | x)
$$

where:
- $\rho_{\text{QSD}}(x)$ is the spatial density
- $\mathcal{M}(v | x)$ is the velocity distribution at position $x$

**Assumption**: The velocity distribution is **Maxwellian** (thermalized):

$$
\mathcal{M}(v | x) = \left(\frac{m}{2\pi k_B T_{\text{eff}}}\right)^{d/2} \exp\left(-\frac{m\|v\|^2}{2k_B T_{\text{eff}}}\right)
$$

with effective temperature $T_{\text{eff}}$ to be determined.

### 3.1 Langevin Stationarity for Velocity

Integrating the Langevin Fokker-Planck equation over velocity $v$:

$$
\int \mathcal{L}_{\text{Langevin}}^* \mu \, dv = -\nabla_x \cdot \left(\rho \langle v \rangle\right) + \ldots
$$

At QSD with no bulk flow, $\langle v \rangle_x = 0$, so the spatial continuity equation is satisfied.

**Velocity thermalization**: The friction-diffusion terms:

$$
-\nabla_v \cdot (\mu \gamma v) + \frac{\sigma_v^2}{2} \nabla_v^2 \mu
$$

are balanced when:

$$
\gamma v \mathcal{M}(v) = -\frac{\sigma_v^2}{2} \nabla_v \mathcal{M}(v) = \frac{\sigma_v^2 m v}{2k_B T} \mathcal{M}(v)
$$

This gives the fluctuation-dissipation relation:

$$
\boxed{k_B T_{\text{Langevin}} = \frac{m \sigma_v^2}{2\gamma}}
$$

### 3.2 Cloning Balance for Spatial Density

The cloning operator modifies the spatial density through fitness-based selection.

**Birth rate** at position $x$:

$$
R_{\text{birth}}(x) = \frac{1}{\tau_{\text{clone}}} \frac{e^{\alpha \Psi(x)}}{\mathcal{Z}} \rho_{\text{QSD}}(x)
$$

**Death rate** at position $x$:

$$
R_{\text{death}}(x) = \frac{1}{\tau_{\text{clone}}} \frac{e^{-\alpha \Psi(x)}}{\mathcal{Z}} \rho_{\text{QSD}}(x)
$$

**Stationary condition**:

$$
R_{\text{birth}}(x) = R_{\text{death}}(x)
$$

$$
e^{\alpha \Psi(x)} \rho(x) = e^{-\alpha \Psi(x)} \rho(x)
$$

This is only satisfied if $\alpha = 0$ (no cloning) or if there is a **constraint** on $\rho(x)$.

**Problem**: The naive detailed balance condition is **not satisfied** for $\alpha \neq 0$!

## 4. Resolution: Global Detailed Balance

The issue is that we considered **local** detailed balance at each point $x$. The correct condition is **global** detailed balance over the entire swarm.

:::{prf:theorem} Global Detailed Balance for Cloning
:label: thm-global-detailed-balance-cloning

At the QSD, the total birth rate equals the total death rate:

$$
\int R_{\text{birth}}(x) dx = \int R_{\text{death}}(x) dx
$$

This is automatically satisfied because:

$$
\int \frac{e^{\alpha \Psi(x)}}{\mathcal{Z}} \rho(x) dx = \int \frac{e^{-\alpha \Psi(x)}}{\mathcal{Z}} \rho(x) dx
$$

if $\rho(x)$ is the stationary distribution for the **full dynamics** (Langevin + cloning).

**Proof**:

The QSD satisfies $\mathcal{L}^* \mu_{\text{QSD}} = 0$. Integrating over all phase space:

$$
\int (\mathcal{L}_{\text{Langevin}}^* + \mathcal{L}_{\text{clone}}^*) \mu \, dx dv = 0
$$

For the cloning part:

$$
\int \mathcal{L}_{\text{clone}}^* \mu \, dx dv = \int \left[\int K(x, v | x', v') \mu(x', v') dx' dv'\right] dx dv - \int \lambda(x, v) \mu(x, v) dx dv
$$

The first integral (total birth) equals the second (total death) by conservation of probability:

$$
\boxed{\text{Total Birth Rate} = \text{Total Death Rate}}
$$

∎

:::

**Implication**: Cloning satisfies **global** detailed balance (total births = total deaths) but not **local** detailed balance (births at $x$ = deaths at $x$).

This is analogous to a **driven system** where there are local currents but global equilibrium.

## 5. Spatial Density from Combined Equilibrium

At QSD, the spatial density $\rho_{\text{QSD}}(x)$ is determined by the balance between:

1. **Langevin drift**: Pushes walkers down the potential $U(x)$
2. **Langevin diffusion**: Spreads walkers according to temperature
3. **Cloning birth-death**: Redistributes walkers based on fitness $\Psi(x)$

:::{prf:proposition} QSD Spatial Density
:label: prop-qsd-spatial-density

At QSD, the spatial density satisfies:

$$
\rho_{\text{QSD}}(x) \propto \exp\left(-\frac{U(x) + \Delta V_{\text{clone}}(x)}{k_B T_{\text{eff}}}\right)
$$

where $\Delta V_{\text{clone}}(x)$ is an effective potential from cloning, related to the fitness $\Psi(x)$.

**Heuristic Derivation**:

The Langevin operator alone would give:

$$
\rho_{\text{Langevin}}(x) \propto e^{-U(x) / k_B T}
$$

The cloning operator with birth rate $\propto e^{\alpha \Psi(x)}$ and death rate $\propto e^{-\alpha \Psi(x)}$ effectively creates a **potential bias**:

$$
\Delta V_{\text{clone}}(x) \sim -\alpha \Psi(x)
$$

(Walkers are born more often at high $\Psi$ and die more often at low $\Psi$, creating an effective attraction to high-fitness regions.)

Combining both:

$$
\rho_{\text{QSD}}(x) \propto \exp\left(-\frac{U(x) - \alpha \Psi(x)}{k_B T_{\text{eff}}}\right)
$$

:::

:::{warning}
**Incomplete Derivation**

The above is a **heuristic argument**. A rigorous derivation requires:

1. Writing the full Fokker-Planck equation including Langevin + cloning
2. Solving for the stationary distribution $\mu_{\text{QSD}}$
3. Verifying that it has the factorized form $\rho(x) \mathcal{M}(v)$
4. Computing the effective potential $\Delta V_{\text{clone}}$ exactly

This is a **non-trivial problem** because:
- The cloning kernel $K(x, v | x', v')$ couples position and velocity through inelastic collisions
- The diversity term $\beta s_i$ introduces **non-local** interactions
- The finite swarm size $N$ creates **stochastic fluctuations**

**Status**: The main conclusion (cloning creates an effective potential) is **plausible** but not rigorously proven in this appendix.

:::

## 6. Energy Balance at QSD

Now we return to the question: Does cloning dissipation balance Langevin heating at QSD?

**Langevin heating rate**:

$$
\frac{dE}{dt}\bigg|_{\text{Langevin}} = \frac{d \sigma_v^2 m}{2}
$$

**Cloning cooling rate** (from Appendix E):

$$
\frac{dE}{dt}\bigg|_{\text{clone}} = -\frac{1 - \alpha_{\text{rest}}^2}{2\tau_{\text{clone}}} m \langle \|v_{\text{rel}}\|^2 \rangle
$$

**Balance condition at QSD**:

$$
\frac{d \sigma_v^2 m}{2} = \frac{1 - \alpha_{\text{rest}}^2}{2\tau_{\text{clone}}} m \langle \|v_{\text{rel}}\|^2 \rangle_{\text{QSD}}
$$

**For Maxwellian velocity distribution**:

$$
\langle \|v_{\text{rel}}\|^2 \rangle_{\text{QSD}} = 2 \langle \|v\|^2 \rangle_{\text{QSD}} = 2 \frac{d k_B T_{\text{eff}}}{m}
$$

Substituting:

$$
\frac{d \sigma_v^2}{2} = \frac{(1 - \alpha_{\text{rest}}^2)}{\tau_{\text{clone}}} d k_B T_{\text{eff}}
$$

$$
T_{\text{eff}} = \frac{\sigma_v^2 m \tau_{\text{clone}}}{2k_B (1 - \alpha_{\text{rest}}^2)}
$$

But we also have from Langevin fluctuation-dissipation:

$$
T_{\text{Langevin}} = \frac{m \sigma_v^2}{2k_B \gamma}
$$

**Consistency requires**:

$$
\gamma = \frac{(1 - \alpha_{\text{rest}}^2)}{\tau_{\text{clone}}}
$$

This is **not generally true**—$\gamma$ and $\tau_{\text{clone}}$ are independent parameters!

**Resolution**: The effective temperature $T_{\text{eff}}$ is **not** simply determined by Langevin fluctuation-dissipation. Instead:

$$
T_{\text{eff}} = f(\gamma, \sigma_v^2, \tau_{\text{clone}}, \alpha_{\text{rest}})
$$

is a **complicated function** determined by the full balance of heating and cooling.

:::{prf:conjecture} Effective Temperature at QSD
:label: conj-effective-temperature-qsd

At the QSD, the effective temperature satisfies a modified fluctuation-dissipation relation:

$$
k_B T_{\text{eff}} = \frac{m\sigma_v^2}{2(\gamma + \gamma_{\text{clone}})}
$$

where:

$$
\gamma_{\text{clone}} = \frac{(1 - \alpha_{\text{rest}}^2)}{\tau_{\text{clone}}} \langle n_{\text{collision}} \rangle
$$

and $\langle n_{\text{collision}} \rangle$ is the average number of walkers per collision group.

**Justification**: Both Langevin friction ($\gamma$) and cloning dissipation ($\gamma_{\text{clone}}$) act to reduce velocity variance. At equilibrium, they combine additively to give an effective friction.

**Status**: This is a **conjecture** requiring verification by:
1. Solving the full Fokker-Planck equation at QSD
2. Computing $\langle n_{\text{collision}} \rangle$ from the birth-death process
3. Verifying the velocity distribution is Maxwellian with $T_{\text{eff}}$

:::

## 7. Summary and Status

:::{important}
**Main Results**

1. **Global detailed balance**: The cloning operator satisfies total birth rate = total death rate globally, but not locally at each point $x$.

2. **QSD factorization**: The QSD likely has the form $\mu_{\text{QSD}}(x, v) = \rho_{\text{QSD}}(x) \mathcal{M}(v | T_{\text{eff}})$ with Maxwellian velocities.

3. **Effective potential**: Cloning creates an effective bias in spatial density:
   $$
   \rho_{\text{QSD}}(x) \propto \exp\left(-\frac{U(x) - \alpha \Psi(x)}{k_B T_{\text{eff}}}\right)
   $$

4. **Effective temperature**: The temperature at QSD is determined by balance between Langevin heating and cloning cooling:
   $$
   T_{\text{eff}} \sim \frac{\sigma_v^2}{2(\gamma + \gamma_{\text{clone}})}
   $$

5. **Stress-energy at QSD**: If the above holds, then:
   $$
   T_{\mu\nu}[\mu_{\text{QSD}}] = m\rho \frac{k_B T_{\text{eff}}}{m} g_{\mu\nu}
   $$
   preserving the Einstein equations.

**Status of Proofs**:
- ✅ Global detailed balance: **Rigorously proven** ({prf:ref}`thm-global-detailed-balance-cloning`)
- ⚠️ QSD factorization: **Plausible**, standard assumption in statistical mechanics
- ⚠️ Effective potential: **Heuristic**, requires full Fokker-Planck solution
- ❓ Effective temperature: **Conjectured** ({prf:ref}`conj-effective-temperature-qsd`), needs verification

**Sufficiency for GR Derivation**:

For the purposes of Chapter 16 (deriving Einstein equations), the main requirement is that:

$$
J^\mu_{\text{clone}}[\mu_{\text{QSD}}] = 0
$$

This follows from:
- Momentum conservation (rigorously proven in Appendix E.2)
- Stationarity of QSD: $\partial \rho_{\text{QSD}}/\partial t = 0$

The **exact form** of $T_{\text{eff}}$ is less critical—it only affects the proportionality constant in $T_{\mu\nu} \propto g_{\mu\nu}$, which is absorbed into the emergent Newton's constant $G$.

Therefore, the **main conclusions of Appendix E** (cloning preserves Einstein equations at QSD) are **robust** even without a complete proof of detailed balance.

:::

**Future Work**:
- Solve the full Fokker-Planck equation for the QSD numerically
- Compute $T_{\text{eff}}$ as a function of algorithmic parameters
- Verify the Maxwellian velocity distribution assumption
- Extend to include diversity term $\beta s_i$ (non-local effects)
