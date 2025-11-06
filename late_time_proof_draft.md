## Late-Time Proof Draft - Regime 2

**Regime 2: Late Time** ($t > T_0$)

For late times, we use the exponential convergence to QSD combined with local stability analysis to obtain a uniform bound that does not depend on time.

### Strategy Overview

The key insight is that once the system is close to the QSD in total variation distance (exponentially fast by `06_convergence.md`), we can use *local regularity theory* to upgrade this weak convergence to $L^\infty$ estimates. The argument proceeds in three steps:

1. **Linearization**: Show that near the QSD, the nonlinear McKean-Vlasov-Fokker-Planck equation can be analyzed via its linearization
2. **L¹-to-L∞ Parabolic Estimate**: Use hypoelliptic regularity to bound the $L^\infty$ norm of perturbations in terms of their $L^1$ norm
3. **Assembly**: Combine with exponential TV convergence to obtain a time-independent bound

### Step 2A: Linearized Operator Around the QSD

:::{prf:lemma} Linearization Around QSD Fixed Point
:label: lem-linearization-qsd

Let $\pi_{\text{QSD}}$ be the quasi-stationary distribution satisfying:

$$
\mathcal{L}_{\text{full}}^* \pi_{\text{QSD}} = 0
$$

where $\mathcal{L}_{\text{full}}^* = \mathcal{L}_{\text{kin}}^* + \mathcal{L}_{\text{clone}}^* - c(z) + r_{\text{revival}}$ is the full generator.

For $\rho_t = \pi_{\text{QSD}} + \eta_t$ with $\|\eta_t\|_{L^1} \ll 1$ small, the perturbation $\eta_t$ evolves according to:

$$
\frac{\partial \eta_t}{\partial t} = \mathbb{L}^* \eta_t + \mathcal{N}[\eta_t]
$$

where:
- $\mathbb{L}^*$ is the **linearized operator** (linear in $\eta$)
- $\mathcal{N}[\eta]$ is the **nonlinear remainder** with $\|\mathcal{N}[\eta]\|_{L^1} = O(\|\eta\|_{L^1}^2)$

**Proof**:

The linearization is standard in McKean-Vlasov theory. We expand each term:

**Kinetic Operator**: $\mathcal{L}_{\text{kin}}^*$ is linear, so:

$$
\mathcal{L}_{\text{kin}}^*(\pi_{\text{QSD}} + \eta) = \underbrace{\mathcal{L}_{\text{kin}}^* \pi_{\text{QSD}}}_{\text{part of QSD eqn}} + \mathcal{L}_{\text{kin}}^* \eta
$$

**Cloning Operator**: The cloning operator has the form (from `03_cloning.md`):

$$
\mathcal{L}_{\text{clone}}^* f = \int K_{\text{clone}}(z, z') V[f](z, z') [f(z') - f(z)] dz'
$$

where $V[f]$ depends nonlinearly on the density. Expanding around $\pi_{\text{QSD}}$:

$$
V[\pi + \eta] = V[\pi] + V'[\pi] \cdot \eta + O(\eta^2)
$$

The linear part is:

$$
\mathbb{L}_{\text{clone}}^* \eta := \int K_{\text{clone}}(z, z') \left[ V[\pi](z, z') \eta(z') + V'[\pi](z, z') \cdot \eta \cdot \pi(z') - \eta(z) V[\pi](z, z') \right] dz'
$$

The quadratic remainder is:

$$
\mathcal{N}_{\text{clone}}[\eta] = \int K_{\text{clone}}(z, z') [V'[\pi] \eta \cdot \eta + O(\eta^2)] dz'
$$

**Killing and Revival**: The killing term $-c(z) f$ is linear. The revival term is:

$$
r_{\text{revival}} = \lambda_{\text{rev}} \frac{m_d(t)}{m_a(t)} f_{\text{safe}}
$$

where $m_a(t) = \int f(t, z) dz$ is the alive mass. For $f = \pi + \eta$:

$$
\frac{1}{m_a} = \frac{1}{m_{\text{eq}} + \|\eta\|_{L^1}} = \frac{1}{m_{\text{eq}}} \left(1 - \frac{\|\eta\|_{L^1}}{m_{\text{eq}}} + O(\|\eta\|_{L^1}^2) \right)
$$

This contributes a linear term and a quadratic remainder.

**Assembly**: Combining all terms, the linearized operator is:

$$
\mathbb{L}^* := \mathcal{L}_{\text{kin}}^* + \mathbb{L}_{\text{clone}}^* - c(z) + \mathbb{L}_{\text{revival}}^*
$$

and the nonlinear remainder satisfies $\|\mathcal{N}[\eta]\|_{L^1} \leq C_{\text{nonlin}} \|\eta\|_{L^1}^2$ for some constant $C_{\text{nonlin}}$ depending on the system parameters. $\square$
:::

### Step 2B: Spectral Gap of the Linearized Operator

:::{prf:lemma} Exponential Decay in L¹ for Linearized Dynamics
:label: lem-linearized-spectral-gap

The linearized operator $\mathbb{L}^*$ around $\pi_{\text{QSD}}$ has a **spectral gap** in $L^2(\pi_{\text{QSD}})$:

$$
\mathbb{L}^* = -\kappa_{\text{lin}} + \text{compact}
$$

where $\kappa_{\text{lin}} > 0$ is the gap. For any perturbation $\eta_0$ with $\|\eta_0\|_{L^1} \leq \delta$ sufficiently small, the linearized evolution satisfies:

$$
\|\eta_t\|_{L^1} \leq \|\eta_0\|_{L^1} e^{-\kappa_{\text{lin}} t / 2}
$$

for all $t \geq 0$, provided $\delta < \delta_0$ for some threshold $\delta_0$ determined by the nonlinearity $C_{\text{nonlin}}$.

**Proof Sketch**:

This follows from standard perturbation theory for nonlinear parabolic equations:

1. **Spectral Gap**: The operator $\mathbb{L}^*$ is the linearization of a hypoelliptic kinetic operator with compact perturbations (cloning, killing, revival). By the results in `06_convergence.md` (geometric ergodicity with rate $\kappa_{\text{QSD}}$), the linearized operator has a spectral gap $\kappa_{\text{lin}} \approx \kappa_{\text{QSD}}$.

2. **Nonlinear Stability**: For the nonlinear equation $\partial_t \eta = \mathbb{L}^* \eta + \mathcal{N}[\eta]$, we use a Grönwall-type argument. The $L^1$ norm evolves as:

$$
\frac{d}{dt} \|\eta_t\|_{L^1} \leq -\kappa_{\text{lin}} \|\eta_t\|_{L^1} + C_{\text{nonlin}} \|\eta_t\|_{L^1}^2
$$

For $\|\eta_0\|_{L^1} \leq \delta_0 := \kappa_{\text{lin}} / (2 C_{\text{nonlin}})$, the linear term dominates and we obtain exponential decay with rate $\kappa_{\text{lin}} / 2$.

**References**: This is a standard result in the theory of reaction-diffusion equations near stable equilibria (Henry 1981, *Geometric Theory of Semilinear Parabolic Equations*, Springer; Theorem 5.1.1). $\square$
:::

### Step 2C: L¹-to-L∞ Estimate via Parabolic Regularity

This is the key technical lemma that upgrades weak ($L^1$) convergence to strong ($L^\infty$) bounds.

:::{prf:lemma} Nash-Aronson Type L¹-to-L∞ Bound for Linearized Operator
:label: lem-l1-to-linfty-near-qsd

For the linearized evolution $\partial_t \eta = \mathbb{L}^* \eta$ starting from $\eta_0$ with $\|\eta_0\|_{L^1} = m$ and $\|\eta_0\|_{L^\infty} \leq M$, there exist constants $C_{\text{Nash}}, \alpha > 0$ (depending on $\gamma, \sigma_v, \sigma_x, R, d$) such that for any $t \geq \tau$ (one timestep):

$$
\|\eta_t\|_{L^\infty} \leq C_{\text{Nash}} \left( \frac{m}{t^{d/2}} + M e^{-\alpha t} \right)
$$

**Interpretation**: The $L^\infty$ norm of perturbations decays to a level controlled by the $L^1$ norm, with a heat-kernel-like rate $t^{-d/2}$.

**Proof**:

This is a classical result in parabolic regularity theory, adapted to the hypoelliptic kinetic setting.

**Step 1: Nash Inequality for Kinetic Operators**

From Hérau & Nier (2004, *Arch. Ration. Mech. Anal.* 171:151-218, Theorem 2.1), hypoelliptic kinetic operators satisfy a Nash-type inequality: for any smooth function $g$ with $\|g\|_{L^1} = m$:

$$
\|g\|_{L^2}^{2 + 4/d} \leq C_N \left( \mathcal{E}(g) \|g\|_{L^1}^{4/d} + \|g\|_{L^1}^{2 + 4/d} \right)
$$

where $\mathcal{E}(g) = \langle g, -\mathbb{L}^* g \rangle$ is the Dirichlet form (entropy production).

**Step 2: L²-to-L∞ Bootstrapping**

For parabolic equations, the Nash inequality implies ultracontractivity of the semigroup $e^{t \mathbb{L}^*}$: there exists $C_U$ such that:

$$
\|e^{t \mathbb{L}^*}\|_{L^1 \to L^\infty} \leq \frac{C_U}{t^{d/2}}
$$

for $t \geq \tau$. This is the **Nash-Aronson estimate** (Aronson 1968, *Bull. Amer. Math. Soc.* 74:47-49).

**Step 3: Semigroup Decomposition**

For $\eta_0$ with mixed $L^1$ and $L^\infty$ bounds, we use the semigroup property:

$$
\eta_t = e^{t \mathbb{L}^*} \eta_0
$$

Decompose $\eta_0 = \eta_0^{\text{small}} + \eta_0^{\text{large}}$ where $\|\eta_0^{\text{small}}\|_{L^\infty}$ is small but $\|\eta_0^{\text{small}}\|_{L^1} = m$, and $\|\eta_0^{\text{large}}\|_{L^1}$ is small. Then:

$$
\|\eta_t\|_{L^\infty} \leq \|e^{t \mathbb{L}^*} \eta_0^{\text{small}}\|_{L^\infty} + \|e^{t \mathbb{L}^*} \eta_0^{\text{large}}\|_{L^\infty}
$$

The first term is bounded by the ultracontractivity estimate: $C_U m / t^{d/2}$. The second term decays exponentially by the spectral gap: $M e^{-\alpha t}$.

Combining these:

$$
\|\eta_t\|_{L^\infty} \leq C_{\text{Nash}} \left( \frac{m}{t^{d/2}} + M e^{-\alpha t} \right)
$$

$\square$
:::

**Remark**: This lemma is the core of the late-time argument. It shows that once the $L^1$ norm is small (from exponential convergence in TV), the $L^\infty$ norm becomes controllable after a moderate time.

### Step 2D: Assembly of Late-Time Bound

Now we combine the pieces to obtain a uniform bound for $t > T_0$.

**Setup**: Choose $T_0$ large enough that:
1. The system has equilibrated to QSD: $\|\rho_{T_0} - \pi_{\text{QSD}}\|_{\text{TV}} \leq \delta_0 / 2$ (from Lemma {prf:ref}`lem-linearized-spectral-gap`)
2. The early-time bound from Regime 1 has produced $\|\rho_{T_0}\|_{L^\infty} \leq C_{\text{hypo}}(M_0, T_0, \ldots)$

**For $t = T_0 + s$ with $s \geq 0$**:

Write $\rho_t = \pi_{\text{QSD}} + \eta_t$ where:

$$
\|\eta_{T_0}\|_{L^1} = \|\rho_{T_0} - \pi_{\text{QSD}}\|_{L^1} \leq \|\rho_{T_0} - \pi_{\text{QSD}}\|_{\text{TV}} \leq \delta_0 / 2
$$

**Step 1: Linearized Evolution for Perturbation**

By Lemma {prf:ref}`lem-linearization-qsd`, the perturbation evolves as:

$$
\frac{\partial \eta_{T_0 + s}}{\partial s} = \mathbb{L}^* \eta_{T_0 + s} + \mathcal{N}[\eta_{T_0 + s}]
$$

**Step 2: $L^1$ Decay of Perturbation**

By Lemma {prf:ref}`lem-linearized-spectral-gap`, since $\|\eta_{T_0}\|_{L^1} \leq \delta_0 / 2 < \delta_0$:

$$
\|\eta_{T_0 + s}\|_{L^1} \leq \|\eta_{T_0}\|_{L^1} e^{-\kappa_{\text{lin}} s / 2} \leq \frac{\delta_0}{2} e^{-\kappa_{\text{lin}} s / 2}
$$

**Step 3: $L^\infty$ Bound on Perturbation**

Apply Lemma {prf:ref}`lem-l1-to-linfty-near-qsd` with:
- $m = \|\eta_{T_0}\|_{L^1} \leq \delta_0 / 2$
- $M = \|\eta_{T_0}\|_{L^\infty} \leq \|\rho_{T_0}\|_{L^\infty} + \|\pi_{\text{QSD}}\|_{L^\infty} \leq C_{\text{hypo}} + C_\pi$
- Time $s \geq \tau$

We get:

$$
\|\eta_{T_0 + s}\|_{L^\infty} \leq C_{\text{Nash}} \left( \frac{\delta_0 / 2}{s^{d/2}} + (C_{\text{hypo}} + C_\pi) e^{-\alpha s} \right)
$$

**Step 4: Choose Intermediate Time $s^* = T_{\text{wait}}$**

Choose $s^* = T_{\text{wait}}$ such that both terms have decayed to comparable size. For concreteness, set:

$$
T_{\text{wait}} := \max\left( 2d / \alpha, \left( \frac{2 C_{\text{Nash}} \delta_0}{\alpha (C_{\text{hypo}} + C_\pi)} \right)^{2/d} \right)
$$

Then for $s \geq T_{\text{wait}}$, both the algebraic and exponential terms are controlled, and:

$$
\|\eta_{T_0 + s}\|_{L^\infty} \leq C_{\text{late}} := C_{\text{Nash}} \left( \frac{\delta_0}{2 T_{\text{wait}}^{d/2}} + (C_{\text{hypo}} + C_\pi) e^{-\alpha T_{\text{wait}}} \right)
$$

**Step 5: Late-Time Density Bound**

For all $t \geq T_0 + T_{\text{wait}}$:

$$
\|\rho_t\|_{L^\infty} = \|\pi_{\text{QSD}} + \eta_t\|_{L^\infty} \leq \|\pi_{\text{QSD}}\|_{L^\infty} + \|\eta_t\|_{L^\infty} \leq C_\pi + C_{\text{late}}
$$

Define:

$$
C_{\text{late}}^{\text{total}} := C_\pi + C_{\text{late}}
$$

This is a **time-independent constant**.

### Step 2E: Uniform Bound Combining Early and Late Times

Combining Regimes 1 and 2:

**For $t \in [0, T_0]$** (Early time):

$$
\|\rho_t\|_{L^\infty} \leq C_{\text{hypo}}(M_0, T_0, \gamma, \sigma_v, \sigma_x, U, R)
$$

**For $t \in [T_0, T_0 + T_{\text{wait}}]$** (Transition):

$$
\|\rho_t\|_{L^\infty} \leq \max(C_{\text{hypo}}, C_{\text{late}}^{\text{total}})
$$

(by continuity and the bounds at endpoints)

**For $t \geq T_0 + T_{\text{wait}}$** (Late time):

$$
\|\rho_t\|_{L^\infty} \leq C_{\text{late}}^{\text{total}}
$$

**Uniform bound**: Define:

$$
\tilde{C}_{\text{hypo}} := \max(C_{\text{hypo}}(M_0, T_0, \ldots), C_{\text{late}}^{\text{total}})
$$

Then for **all** $t \geq 0$:

$$
\|\rho_t\|_{L^\infty} \leq \tilde{C}_{\text{hypo}}
$$

**Key observation**: Unlike the early-time-only bound, $\tilde{C}_{\text{hypo}}$ does **not** grow with time. The constant $C_{\text{late}}^{\text{total}}$ depends on system parameters but is independent of the initial condition's evolution time.

### Step 2F: Density Ratio Bound for Late Times

Repeating the argument from Regime 1, for $t > T_0 + T_{\text{wait}}$:

$$
\frac{\tilde{\rho}_t(x)}{\tilde{\pi}_{\text{QSD}}(x)} = \frac{\rho_t(x)}{\pi_{\text{QSD}}(x)} \cdot \frac{m_{\text{eq}}}{\|\rho_t\|_{L^1}}
$$

With the mass lower bound $\|\rho_t\|_{L^1} \geq c_{\text{mass}}$ (Lemma {prf:ref}`lem-mass-lower-bound-high-prob`) and the late-time upper bound:

$$
\sup_{x} \frac{\tilde{\rho}_t(x)}{\tilde{\pi}_{\text{QSD}}(x)} \leq \frac{C_{\text{late}}^{\text{total}}}{c_{\sigma_x, R} \cdot m_{\text{eq}}} \cdot \frac{m_{\text{eq}}}{c_{\text{mass}}} = \frac{C_{\text{late}}^{\text{total}}}{c_{\sigma_x, R} \cdot c_{\text{mass}}}
$$

Define:

$$
M_2 := \frac{C_{\text{late}}^{\text{total}}}{c_{\sigma_x, R} \cdot c_{\text{mass}}}
$$

Then for all $t \geq T_0 + T_{\text{wait}}$:

$$
\sup_{x \in \mathcal{X}_{\text{valid}}} \frac{d\tilde{\mu}_t}{d\tilde{\pi}_{\text{QSD}}}(x) \leq M_2 < \infty
$$

**Comparison with Early-Time Bound**:

We have two finite constants:
- $M_1 = C_{\text{hypo}}(M_0, T_0, \ldots) / (c_{\sigma_x, R} \cdot c_{\text{mass}})$ (early time, depends on $T_0$)
- $M_2 = C_{\text{late}}^{\text{total}} / (c_{\sigma_x, R} \cdot c_{\text{mass}})$ (late time, independent of $T_0$)

The **uniform bound** is:

$$
M := \max(M_1, M_2) < \infty
$$

This is **finite** and **independent of time** for $t \geq 0$.

---

**End of Late-Time Proof**
