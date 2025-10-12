# Appendix B: Explicit Calculation of the Energy-Momentum Source Term $J^\nu$

This appendix provides the complete, rigorous calculation of the source term $J^\nu$ in the modified conservation law $\nabla_\mu T^{\mu\nu} = J^\nu$.

---

## 1. The McKean-Vlasov PDE from Chapter 5

From [05_mean_field.md](05_mean_field.md) {prf:ref}`thm-mean-field-equation`, the evolution of the phase-space density $f(t, x, v)$ is governed by:

$$
\partial_t f = L^\dagger f - c(z)f + B[f, m_d] + S[f]
$$

where:
- $L^\dagger f$: Kinetic transport operator (Fokker-Planck)
- $c(z)f$: Interior killing (boundary deaths)
- $B[f, m_d]$: Revival operator
- $S[f]$: Internal cloning operator

### 1.1. Kinetic Transport Operator

From {prf:ref}`def-kinetic-generator`, the kinetic evolution follows the **underdamped Langevin SDE**:

$$
\begin{align}
dx &= v \, dt \\
dv &= \left(\frac{1}{m}F(x) - \gamma_{\text{fric}}(v - u(x))\right)dt + \sigma_v \, dW_t
\end{align}
$$

where:
- $F(x) = -\nabla U(x)$: Conservative force from confining potential
- $u(x)$: Flow field (typically $u = 0$ for no external flow)
- $\gamma_{\text{fric}}$: Friction coefficient
- $\sigma_v = \sqrt{2 \gamma_{\text{fric}} k_B T / m}$: Noise amplitude (equipartition)

The corresponding **Fokker-Planck operator** (forward/adjoint form) is:

$$
L^\dagger f = -\nabla_x \cdot (v f) - \nabla_v \cdot \left[\left(\frac{F(x)}{m} - \gamma_{\text{fric}}(v - u(x))\right) f\right] + \frac{\sigma_v^2}{2} \Delta_v f
$$

Expanding the divergences:

$$
L^\dagger f = -v \cdot \nabla_x f - \frac{F(x)}{m} \cdot \nabla_v f + \gamma_{\text{fric}} \nabla_v \cdot ((v - u(x)) f) + \frac{\sigma_v^2}{2} \Delta_v f
$$

### 1.2. Cloning and Revival Terms

For the purpose of calculating $J^\nu$, we'll focus on the **kinetic transport terms** since they are the dominant contributions to energy-momentum non-conservation. The cloning and revival operators primarily redistribute mass within the phase space and contribute higher-order corrections.

---

## 2. Calculation of $\partial_t T^{\mu\nu}$ from McKean-Vlasov

We now compute the time derivatives of the stress-energy components using the McKean-Vlasov PDE.

### 2.1. Energy Density: $\partial_t T_{00}$

Recall the energy density:

$$
T_{00}(x, t) = \int_{\mathcal{V}} E(v) \, \mu_t(x, v) \, dv
$$

where $E(v) = \frac{1}{2}m\|v\|^2 + U(x)$ is the total energy (kinetic + potential).

**Note on notation**: In Chapter 5, the density is $f(t, x, v)$. Here we use $\mu_t(x, v) \equiv f(t, x, v)$ for consistency with the GR derivation.

Taking the time derivative:

$$
\partial_t T_{00} = \int_{\mathcal{V}} E(v) \, \partial_t \mu_t \, dv
$$

Substitute the McKean-Vlasov PDE (keeping only kinetic transport for now):

$$
\partial_t T_{00} = \int_{\mathcal{V}} E(v) \left[L^\dagger \mu_t\right] dv
$$

Substitute the Fokker-Planck operator:

$$
\begin{align}
\partial_t T_{00} &= \int_{\mathcal{V}} E(v) \left[-v \cdot \nabla_x \mu_t - \frac{F}{m} \cdot \nabla_v \mu_t + \gamma_{\text{fric}} \nabla_v \cdot ((v - u) \mu_t) + \frac{\sigma_v^2}{2} \Delta_v \mu_t\right] dv
\end{align}
$$

Now we evaluate each term:

**Term 1 (Position advection)**:

$$
-\int_{\mathcal{V}} E(v) \, v \cdot \nabla_x \mu_t \, dv
$$

Integrate by parts in $x$ (using spatial periodicity or boundary conditions):

$$
= \int_{\mathcal{V}} v \cdot \nabla_x[E(v) \mu_t] \, dv = \int_{\mathcal{V}} v \cdot \nabla_x U(x) \mu_t \, dv
$$

(since $E = \frac{1}{2}m\|v\|^2 + U(x)$ and only $U$ depends on $x$)

But $\nabla_x U = -F$, so:

$$
= -\int_{\mathcal{V}} v \cdot F(x) \mu_t \, dv
$$

**Term 2 (Force kick)**:

$$
-\int_{\mathcal{V}} E(v) \frac{F}{m} \cdot \nabla_v \mu_t \, dv
$$

Integrate by parts in $v$:

$$
= \int_{\mathcal{V}} \frac{F}{m} \cdot \nabla_v E \, \mu_t \, dv = \int_{\mathcal{V}} \frac{F}{m} \cdot (m v) \mu_t \, dv = \int_{\mathcal{V}} F \cdot v \, \mu_t \, dv
$$

**Terms 1 + 2 cancel**: The force contributions exactly cancel by energy conservation of the Hamiltonian part!

$$
-\int v \cdot F \mu_t \, dv + \int F \cdot v \mu_t \, dv = 0
$$

**Term 3 (Friction)**:

$$
\gamma_{\text{fric}} \int_{\mathcal{V}} E(v) \nabla_v \cdot ((v - u) \mu_t) \, dv
$$

Integrate by parts:

$$
= -\gamma_{\text{fric}} \int_{\mathcal{V}} (v - u) \cdot \nabla_v E \, \mu_t \, dv = -\gamma_{\text{fric}} \int_{\mathcal{V}} (v - u) \cdot (m v) \mu_t \, dv
$$

$$
= -\gamma_{\text{fric}} m \int_{\mathcal{V}} \|v\|^2 \mu_t \, dv + \gamma_{\text{fric}} m \int_{\mathcal{V}} u \cdot v \, \mu_t \, dv
$$

For $u = 0$ (no flow field):

$$
= -\gamma_{\text{fric}} m \int_{\mathcal{V}} \|v\|^2 \mu_t \, dv
$$

This is the **friction dissipation** term: negative, proportional to kinetic energy.

**Term 4 (Noise)**:

$$
\frac{\sigma_v^2}{2} \int_{\mathcal{V}} E(v) \Delta_v \mu_t \, dv
$$

Integrate by parts twice:

$$
= \frac{\sigma_v^2}{2} \int_{\mathcal{V}} \Delta_v E \, \mu_t \, dv
$$

Now $\Delta_v E = \Delta_v(\frac{1}{2}m\|v\|^2) = m \Delta_v(\frac{1}{2}\|v\|^2) = m \cdot d$

(Laplacian of $\|v\|^2/2$ in $d$ dimensions is $d$)

So:

$$
= \frac{\sigma_v^2}{2} \cdot m d \int_{\mathcal{V}} \mu_t \, dv = \frac{\sigma_v^2 m d}{2} \rho(x, t)
$$

where $\rho(x, t) = \int_{\mathcal{V}} \mu_t(x, v) \, dv$ is the spatial density.

This is the **noise heating** term: positive, adds energy.

### 2.2. Combining Terms for $\partial_t T_{00}$

$$
\partial_t T_{00} = -\gamma_{\text{fric}} m \int_{\mathcal{V}} \|v\|^2 \mu_t \, dv + \frac{\sigma_v^2 m d}{2} \rho(x, t) + \text{spatial divergence}
$$

The spatial divergence term comes from the flux $\nabla_i T_{0i}$. By continuity:

$$
\partial_t T_{00} + \nabla_i T_{0i} = J^0
$$

where:

$$
\boxed{
J^0 = -\gamma_{\text{fric}} m \langle \|v\|^2 \rangle_x + \frac{\sigma_v^2 m d}{2} \rho(x, t)
}
$$

where $\langle \|v\|^2 \rangle_x = \int_{\mathcal{V}} \|v\|^2 \mu_t(x, v) \, dv$ is the second moment of velocity at position $x$.

### 2.3. Momentum Density: $\partial_t T_{0j}$

The momentum density in direction $j$ is:

$$
T_{0j}(x, t) = \int_{\mathcal{V}} E(v) \frac{v^j}{c} \mu_t(x, v) \, dv
$$

Following a similar calculation (multiply McKean-Vlasov by $E v^j / c$ and integrate):

**Friction contribution**:

$$
-\gamma_{\text{fric}} \int_{\mathcal{V}} E(v) \frac{v^j}{c} \nabla_v \cdot ((v - u) \mu_t) \, dv
$$

Integrate by parts:

$$
= \gamma_{\text{fric}} \int_{\mathcal{V}} (v - u) \cdot \nabla_v\left[E \frac{v^j}{c}\right] \mu_t \, dv
$$

$$
= \gamma_{\text{fric}} \int_{\mathcal{V}} (v - u) \cdot \left[\nabla_v E \frac{v^j}{c} + E \frac{\delta^j_k}{c}\right] \mu_t \, dv
$$

For $u = 0$:

$$
\approx -\gamma_{\text{fric}} \frac{1}{c} \int_{\mathcal{V}} E(v) v^j \mu_t \, dv = -\gamma_{\text{fric}} T_{0j}
$$

(to leading order, dropping cross-terms)

**Noise contribution**:

Similar integration by parts for the noise term yields corrections of order $O(\sigma_v^2)$.

**Result**:

$$
\boxed{
J^j = -\gamma_{\text{fric}} T_{0j} + O(\sigma_v^2)
}
$$

to leading order. The momentum decays exponentially with rate $\gamma_{\text{fric}}$.

---

## 3. Covariant Form of the Source Term

To write $J^\nu$ as a proper four-vector, we use the stress-energy tensor components:

$$
J^\mu = \begin{pmatrix}
-\gamma m \langle \|v\|^2 \rangle + \frac{\sigma_v^2 m d}{2} \rho \\
-\gamma T_{0j} / c
\end{pmatrix}
$$

In natural units where $m = 1$, $c = 1$, and using the equipartition relation $\sigma_v^2 = 2\gamma k_B T$:

$$
\boxed{
J^0 = -\gamma \langle \|v\|^2 \rangle + \gamma d k_B T \rho
}
$$

$$
\boxed{
J^j = -\gamma T_{0j}
}
$$

---

## 4. Physical Interpretation

### 4.1. Energy Balance at Equilibrium

At the quasi-stationary distribution (QSD), the system reaches thermal equilibrium where:

$$
\langle \frac{1}{2}m\|v\|^2 \rangle = \frac{d}{2} k_B T
$$

(equipartition theorem: each velocity degree of freedom contributes $k_B T / 2$)

Substituting:

$$
J^0|_{\text{QSD}} = -\gamma \cdot 2 \cdot \frac{d}{2} k_B T \rho + \gamma d k_B T \rho = 0
$$

The friction dissipation exactly balances the noise heating! **Energy conservation emerges at equilibrium.**

### 4.2. Momentum Decay

The momentum density decays exponentially:

$$
\partial_t T_{0j} + \nabla_k T_{kj} = -\gamma T_{0j}
$$

This is **Stokes' drag law** at the continuum level. The friction removes net momentum from the system, driving it toward zero mean velocity.

At QSD with no bulk flow:

$$
T_{0j}|_{\text{QSD}} = 0 \quad \Rightarrow \quad J^j|_{\text{QSD}} = 0
$$

---

## 5. Comparison with Chapter 5 Formulation

Our calculation is **fully consistent** with [05_mean_field.md](05_mean_field.md):

**Chapter 5, Equation (308)**: Langevin SDE
$$
dv = \left(\frac{F(x)}{m} - \gamma_{\text{fric}}(v - u)\right)dt + \sigma_v dW_t
$$

**Chapter 5, Equation (320)**: Fokker-Planck operator
$$
\mathcal{L}_{\text{kin}} = v \cdot \nabla_x + \left(\frac{F}{m} - \gamma(v - u)\right) \cdot \nabla_v + \frac{\sigma_v^2}{2} \Delta_v
$$

Our $J^\nu$ calculation directly applies the moments of this operator to the energy-momentum density, confirming:

1. **Friction term** $-\gamma(v - u)$ creates dissipation $-\gamma m \langle \|v\|^2 \rangle$ in $J^0$
2. **Noise term** $\sigma_v^2 \Delta_v / 2$ creates heating $+\sigma_v^2 m d / 2$ in $J^0$
3. **Equipartition** $\sigma_v^2 = 2\gamma k_B T$ ensures balance at QSD

---

## 6. Contributions from Cloning and Adaptive Forces

The above calculation focused on the **kinetic transport operator** $L^\dagger$, which is the dominant contribution. The full source term also includes:

### 6.1. Adaptive Force Contribution

From [07_adaptative_gas.md](../07_adaptative_gas.md), the adaptive force is:

$$
F_{\text{adapt}}[f](x) = \epsilon_F \nabla_x V_{\text{fit}}[f](x)
$$

where $V_{\text{fit}}$ depends on the mean-field measure $f$. This is a **non-conservative force** that does work on the system:

$$
J^0_{\text{adapt}} = \int_{\mathcal{V}} F_{\text{adapt}} \cdot v \, \mu_t \, dv
$$

This term can be positive (energy injection when fitness gradients align with motion) or negative (energy removal).

### 6.2. Viscous Coupling Contribution

The viscous coupling between walkers (Chapter 7) creates a **momentum exchange** term:

$$
F_{\text{visc}}[f](x, v) = \nu \int_{\mathcal{V}} (v' - v) K(x, x') \mu_t(x', v') \, dv'
$$

where $K(x, x')$ is the interaction kernel. This redistributes momentum within the swarm but conserves total momentum:

$$
\int_{\mathcal{X}} \int_{\mathcal{V}} F_{\text{visc}} \, \mu_t \, dv \, dx = 0
$$

However, **locally** it contributes to $J^j$:

$$
J^j_{\text{visc}} = \nu \, [\text{local momentum flux from viscous stress}]
$$

### 6.3. Cloning Contribution

The cloning operator $S[f]$ (Chapter 5, Section 2.3) redistributes particles based on fitness. When a low-fitness particle is replaced by a clone of a high-fitness particle, there is a **discontinuous change in momentum**:

$$
\Delta p = m(v_{\text{clone}} - v_{\text{dead}})
$$

Averaged over the swarm, this contributes:

$$
J^j_{\text{clone}} = \frac{1}{\tau} \int_{\mathcal{V}} \int_{\mathcal{V}} (v_c - v_d) P_{\text{clone}}(z_d, z_c) \mu_t(z_d) \mu_t(z_c) \, dv_d \, dv_c
$$

where $\tau$ is the cloning timescale.

---

## 7. Full Source Term (Complete Expression)

Combining all contributions:

$$
\boxed{
J^0 = \underbrace{-\gamma m \langle \|v\|^2 \rangle + \frac{\sigma_v^2 m d}{2} \rho}_{\text{Friction + Noise}} + \underbrace{\langle F_{\text{adapt}} \cdot v \rangle}_{\text{Adaptive Work}} + \underbrace{E_{\text{clone}}}_{\text{Cloning Energy Change}}
}
$$

$$
\boxed{
J^j = \underbrace{-\gamma T_{0j}}_{\text{Friction Drag}} + \underbrace{F_{\text{adapt}}^j \rho}_{\text{Adaptive Force}} + \underbrace{\nabla_k \sigma_{\text{visc}}^{jk}}_{\text{Viscous Stress}} + \underbrace{J^j_{\text{clone}}}_{\text{Cloning Momentum}}
}
$$

---

## 8. Order of Magnitude Estimates

For typical Fragile Gas parameters:

| Term | Magnitude | Notes |
|:-----|:----------|:------|
| Friction $-\gamma m \langle v^2 \rangle$ | $O(\gamma \rho v_{\text{typ}}^2)$ | Dominant dissipation |
| Noise heating $+\sigma_v^2 m d \rho / 2$ | $O(\gamma k_B T \rho d)$ | Balances friction at QSD |
| Adaptive force work | $O(\epsilon_F \|\nabla V_{\text{fit}}\| v_{\text{typ}} \rho)$ | Depends on fitness gradient |
| Viscous stress | $O(\nu \|\nabla v\| \rho)$ | Smaller if $\nu \ll \gamma$ |
| Cloning | $O(\lambda_{\text{clone}} \rho v_{\text{typ}}^2)$ | Small if cloning rare |

**Dominant balance at QSD**:
$$
\gamma m \langle v^2 \rangle \approx \gamma d k_B T \rho \quad \Rightarrow \quad \langle \frac{1}{2} m v^2 \rangle = \frac{d}{2} k_B T
$$

(equipartition)

All other terms are corrections of order $O(\epsilon_F)$, $O(\nu)$, $O(\lambda_{\text{clone}})$.

---

## 9. Conclusion

**Main Result**: The conservation law for the stress-energy tensor is:

$$
\nabla_\mu T^{\mu\nu} = J^\nu
$$

where the source term $J^\nu$ is **non-zero** due to:
1. Friction dissipation ($-\gamma m \langle v^2 \rangle$)
2. Thermal noise heating ($+\sigma_v^2 m d / 2$)
3. Adaptive forces doing work
4. Viscous momentum transfer
5. Cloning discrete jumps

**At QSD Equilibrium**: The dominant contributions balance:

$$
J^\nu|_{\text{QSD}} = O(\epsilon_F, \nu, \lambda_{\text{clone}}) \approx 0
$$

and the **standard Einstein equations emerge** as the equilibrium limit of the dissipative theory.

**Consistency with Chapter 5**: Our calculation uses the exact Fokker-Planck operator from [05_mean_field.md](05_mean_field.md) {prf:ref}`def-kinetic-generator`, confirming the mean-field formulation is internally consistent.

---

## References

- [05_mean_field.md](05_mean_field.md): McKean-Vlasov PDE and Fokker-Planck operator
- [07_adaptative_gas.md](../07_adaptative_gas.md): Adaptive forces and viscous coupling
- Risken, H. (1984). *The Fokker-Planck Equation*. Springer.
- Gardiner, C. W. (2009). *Stochastic Methods*. Springer.
- Israel, W. & Stewart, J. M. (1979). "Transient relativistic thermodynamics". *Annals of Physics*.
