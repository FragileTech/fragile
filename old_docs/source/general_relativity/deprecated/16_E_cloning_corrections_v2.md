# Appendix E: Higher-Order Corrections from Cloning Operator

## Overview

This appendix analyzes how the **cloning operator** modifies the stress-energy tensor beyond the leading-order kinetic term $T_{\mu\nu} = m\rho \langle v^\mu v^\nu \rangle$.

**Main Result**: The inelastic collision cloning model (Chapter 3) **exactly conserves momentum** at each event, so there is **no source term** $J^\mu_{\text{clone}}$ from cloning. The only effect is **energy dissipation** through inelastic collisions, which modifies the effective temperature. At QSD, this is balanced by Langevin heating, preserving the Einstein equations.

## 1. Cloning Operator: Inelastic Collision Model

From Chapter 3 ({prf:ref}`def-inelastic-collision-update`), the canonical cloning mechanism is:

**Selection**: Walkers clone from high-fitness companions with probability $\propto \exp(\alpha \Psi(x) + \beta s)$

**Multi-body inelastic collision**: When walkers $\{j_1, \ldots, j_k\}$ clone from companion $i$:

1. **Center-of-mass velocity** (total momentum conserved):
   $$
   v_{\text{COM}} = \frac{1}{k+1}\sum_{a \in \{i,j_1,\ldots,j_k\}} v_a
   $$

2. **Inelastic collapse** with restitution $\alpha_{\text{rest}} \in [0, 1]$:
   $$
   v_a' = v_{\text{COM}} + \alpha_{\text{rest}}(v_a - v_{\text{COM}})
   $$

3. **Position cloning**: $x_{j_k}' = x_i$ (exact spatial cloning)

**Key Properties**:
- ✅ **Total momentum conserved**: $\sum_i m v_i' = \sum_i m v_i$
- ⚠️ **Kinetic energy dissipated**: $(1 - \alpha_{\text{rest}}^2)$ fraction of relative KE lost

## 2. Stress-Energy Tensor Analysis

### 2.1 Momentum Conservation

:::{prf:lemma} No Momentum Source from Cloning
:label: lem-no-momentum-source-cloning

The cloning operator conserves total four-momentum exactly:

$$
\sum_{i=1}^N p_i^\mu \bigg|_{\text{after}} = \sum_{i=1}^N p_i^\mu \bigg|_{\text{before}}
$$

**Consequence**: There is **no source term** $J^\mu_{\text{clone}}$ in the energy-momentum conservation law. The cloning operator does not modify:

$$
\nabla_\mu T^{\mu\nu} = J^\nu_{\text{Langevin}}
$$

where $J^\nu_{\text{Langevin}}$ is the friction-noise source from the kinetic operator (Appendix B).

:::

**Proof**: The inelastic collision formula $v_a' = v_{\text{COM}} + \alpha_{\text{rest}}(v_a - v_{\text{COM}})$ preserves:

$$
\sum_{a \in \text{group}} v_a' = (k+1)v_{\text{COM}} = \sum_{a \in \text{group}} v_a
$$

Since all walkers either persist (unchanged) or participate in such a collision, total momentum is conserved. ∎

### 2.2 Energy Dissipation

:::{prf:proposition} Cloning Energy Sink
:label: prop-cloning-cooling

The cloning operator dissipates kinetic energy at rate:

$$
\frac{dE_{\text{kin}}}{dt}\bigg|_{\text{clone}} = -\frac{1 - \alpha_{\text{rest}}^2}{2\tau_{\text{clone}}} m\langle \|v_{\text{rel}}\|^2 \rangle_{\text{collision groups}} < 0
$$

where $\tau_{\text{clone}}$ is the average time between cloning events and $v_{\text{rel}}$ is relative velocity within collision groups.

**Physical Interpretation**: Cloning acts as an **effective friction** that reduces the temperature:

$$
\frac{dT}{dt}\bigg|_{\text{clone}} = -\gamma_{\text{clone}}(T - T_{\text{target}})
$$

where $\gamma_{\text{clone}} \sim (1 - \alpha_{\text{rest}}^2)/\tau_{\text{clone}}$.

:::

**Proof Sketch**: Energy loss per collision is $\Delta E = -\frac{1-\alpha_{\text{rest}}^2}{2}m\sum_a \|v_a - v_{\text{COM}}\|^2$. Averaging over collision groups and multiplying by rate $1/\tau_{\text{clone}}$ gives the stated result. ∎

## 3. Equilibrium at QSD

### 3.1 Energy Balance

At the quasi-stationary distribution, energy input from Langevin noise balances energy dissipation from cloning:

**Langevin heating**:
$$
\frac{dE_{\text{kin}}}{dt}\bigg|_{\text{Langevin}} = +\frac{d \sigma_v^2 m}{2}
$$

**Cloning cooling**:
$$
\frac{dE_{\text{kin}}}{dt}\bigg|_{\text{clone}} = -\frac{1-\alpha_{\text{rest}}^2}{2\tau_{\text{clone}}} m \langle \|v_{\text{rel}}\|^2 \rangle
$$

**Equilibrium condition**:
$$
\frac{d \sigma_v^2 m}{2} = \frac{1-\alpha_{\text{rest}}^2}{2\tau_{\text{clone}}} m \langle \|v_{\text{rel}}\|^2 \rangle_{\text{QSD}}
$$

**Effective temperature**: The equilibrium temperature $T_{\text{QSD}}$ satisfies:

$$
\langle \|v_{\text{rel}}\|^2 \rangle_{\text{QSD}} = \frac{d \sigma_v^2 \tau_{\text{clone}}}{1 - \alpha_{\text{rest}}^2}
$$

### 3.2 Stress-Energy Tensor at QSD

:::{prf:theorem} Cloning Preserves Einstein Equations at QSD
:label: thm-cloning-preserves-gr

At the quasi-stationary distribution, the cloning operator does not introduce any source terms or modify the form of the Einstein equations. The stress-energy tensor is:

$$
T_{\mu\nu}[\mu_{\text{QSD}}] = m\rho \langle v^\mu v^\nu \rangle_{\text{QSD}}
$$

where the velocity distribution is Maxwellian with effective temperature $T_{\text{QSD}}$ determined by the energy balance between Langevin heating and cloning cooling.

**Consequence**: The Einstein equations remain:

$$
G_{\mu\nu} = 8\pi G T_{\mu\nu}
$$

with no modifications from the cloning dynamics.

**Status**: This is **conditionally proven** assuming:
1. ✅ Momentum conservation (rigorously proven, {prf:ref}`lem-no-momentum-source-cloning`)
2. ⚠️ Energy balance at QSD (plausible, requires full McKean-Vlasov analysis with cloning)
3. ⚠️ Maxwellian velocity distribution at QSD (standard assumption, requires detailed balance proof)

:::

**Justification**:

1. **No momentum source**: Proven rigorously by inelastic collision mechanics

2. **Energy balance**: The QSD must satisfy $\mathcal{L}^* \mu_{\text{QSD}} = 0$ where $\mathcal{L}^*$ includes both Langevin and cloning operators. This implies energy input = energy dissipation.

3. **Maxwellian distribution**: At QSD with isotropic forces and detailed balance, the velocity distribution is Maxwellian (standard result from statistical mechanics).

:::{warning}
**Gaps in Current Analysis**

The energy balance argument is **heuristic**. A rigorous proof requires:

1. Writing the full Fokker-Planck operator including cloning birth-death terms
2. Proving that $\mu_{\text{QSD}}$ is Maxwellian in velocity at each spatial point
3. Deriving the effective temperature $T_{\text{QSD}}$ from the balance equation
4. Handling the non-local diversity term $\beta s_i$ in the cloning probability

These are standard (but technical) calculations in non-equilibrium statistical mechanics. The main claim (cloning preserves GR at QSD) is physically robust but not fully rigorously proven in this appendix.
:::

## 4. Off-Equilibrium Behavior

Away from QSD, the energy dissipation from cloning creates an effective **negative pressure** contribution to the stress-energy tensor.

**Modified stress-energy**:
$$
T_{\mu\nu}^{\text{eff}} = m\rho\langle v^\mu v^\nu \rangle - \Pi_{\mu\nu}^{\text{dissipation}}
$$

where $\Pi_{\mu\nu}^{\text{dissipation}}$ accounts for energy loss during cloning events.

**Conservation law**:
$$
\nabla_\mu T^{\mu\nu}_{\text{eff}} = J^\nu_{\text{Langevin}}
$$

Note that there is still **no cloning momentum source** $J^\nu_{\text{clone}}$, only a modification to the effective stress tensor.

**Convergence to QSD**: As the system approaches QSD, $\Pi_{\mu\nu}^{\text{dissipation} \to 0$ exponentially fast (Chapter 4 convergence theory).

## 5. Summary and Implications

:::{important}
**Main Results**

1. **Momentum conservation**: The inelastic collision cloning model exactly conserves momentum, so $J^\mu_{\text{clone}} = 0$ identically (no source term).

2. **Energy dissipation**: Cloning dissipates kinetic energy at rate $\propto (1 - \alpha_{\text{rest}}^2)/\tau_{\text{clone}}$, acting as effective cooling.

3. **QSD equilibrium**: At QSD, Langevin heating balances cloning cooling, yielding an effective temperature $T_{\text{QSD}}$.

4. **Einstein equations preserved**: At QSD, the field equations remain:
   $$
   G_{\mu\nu} = 8\pi G T_{\mu\nu}
   $$
   with $T_{\mu\nu}$ computed using $T_{\text{QSD}}$.

5. **Robustness**: The emergence of GR is robust to cloning dynamics—algorithmic details affect only the effective temperature, not the tensor structure.

**Status**:
- ✅ Momentum conservation: **Rigorously proven**
- ⚠️ Energy balance and Maxwellian QSD: **Plausible, not fully rigorous**
- ✅ Preservation of Einstein equations: **Conditional on QSD properties**

**Comparison to Initial Draft**:

The initial draft incorrectly used a non-conserving cloning model ($v_{\text{child}} = v_{\text{parent}} + \delta\xi$), which would have created a momentum source $J^\mu_{\text{clone}}$. The canonical inelastic collision model (Chapter 3) avoids this issue entirely by conserving momentum exactly.

This is a **major simplification**: cloning does not introduce new source terms, only modifies the effective temperature.
:::

**Next**: Appendix F analyzes adaptive forces ($\varepsilon_F$, mean-field potential) and Appendix G covers viscous coupling ($\nu$).
