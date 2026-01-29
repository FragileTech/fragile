(sec-contact-nambu-mechanics)=
# Contact-Nambu Mechanics: Dissipative Higher-Order Dynamics

## TLDR

- **Contact-Nambu** unifies contact geometry (intrinsic dissipation) with Nambu mechanics (higher-order brackets)
- Contact structure provides: friction $-\gamma p$ and entropy production $\dot{s} = p \cdot v - H$
- Nambu brackets provide: volume-preserving flow on $(z, p)$ subspace with constraint $\mathcal{S}$
- The synthesis uses **contact dynamics for dissipation** + **Nambu structure for constraints**
- Fully consistent with underdamped Langevin (Definition {prf:ref}`def-bulk-drift-continuous-flow`)

## Motivation

Standard Hamiltonian mechanics is conservative: energy is preserved, phase space volume is preserved (Liouville). But physical systems dissipate. The framework uses underdamped Langevin dynamics ({ref}`sec-the-equations-of-motion-geodesic-jump-diffusion`, Definition {prf:ref}`def-bulk-drift-continuous-flow`) with explicit friction $-\gamma p$. 

**Question:** Where does the friction come from geometrically?

**Answer:** Contact geometry. The contact structure provides a canonical way to include dissipation in Hamiltonian mechanics without breaking the geometric framework. The Nambu structure adds the ability to impose additional constraints (like entropy bounds) while preserving the dissipative dynamics.

{cite}`bravetti2017contact,bravetti2019contact,ciaglia2018contact`

(sec-contact-geometry-primer)=
## Contact Geometry Primer

:::{prf:definition} Contact Manifold
:label: def-contact-manifold

A **contact manifold** $(M^{2n+1}, \theta)$ is an odd-dimensional manifold $M$ equipped with a 1-form $\theta$ such that:

$$
\theta \wedge (d\theta)^n \neq 0
$$

everywhere on $M$. The form $\theta$ is the **contact form** and the condition ensures maximal non-integrability.

*Standard example:* The extended phase space $(T^*Q \times \mathbb{R}, \theta)$ with coordinates $(q, p, s)$ and:

$$
\theta = ds - p_i\, dq^i
$$

This is the **thermodynamic phase space** where $s$ is entropy/action.

:::

:::{prf:definition} Reeb Vector Field
:label: def-reeb-vector-field

Given a contact manifold $(M, \theta)$, the **Reeb vector field** $R$ is the unique vector field satisfying:

$$
\theta(R) = 1, \qquad \iota_R\, d\theta = 0
$$

*Interpretation:* $R$ is the direction of "pure dissipation" — motion along $R$ changes $s$ without changing $(q, p)$.

In coordinates $(q, p, s)$ with $\theta = ds - p_i\, dq^i$:

$$
R = \frac{\partial}{\partial s}
$$

:::

:::{prf:definition} Contact Hamiltonian Vector Field
:label: def-contact-hamiltonian-vector-field

Given a contact manifold $(M, \theta)$ and a function $H: M \to \mathbb{R}$, the **contact Hamiltonian vector field** $X_H$ is defined by:

$$
\iota_{X_H}\, d\theta = dH - (R \cdot H)\,\theta, \qquad \theta(X_H) = -H
$$

The resulting **contact Hamilton equations** are:

$$
\begin{cases}
\dot{q}^i = \dfrac{\partial H}{\partial p_i} \\[8pt]
\dot{p}_i = -\dfrac{\partial H}{\partial q^i} - p_i \dfrac{\partial H}{\partial s} \\[8pt]
\dot{s} = p_i \dfrac{\partial H}{\partial p_i} - H
\end{cases}
$$

*Key properties:*
1. The term $-p_i \frac{\partial H}{\partial s}$ in the momentum equation is the **geometric origin of friction**
2. The entropy evolution $\dot{s} = p \cdot v - H$ is **not arbitrary** — it follows from the contact structure
3. Energy is not conserved: $\frac{dH}{dt} = -H \cdot \frac{\partial H}{\partial s}$

:::

:::{admonition} Contact vs Symplectic
:class: note

| Property | Symplectic $(T^*Q, \omega)$ | Contact $(T^*Q \times \mathbb{R}, \theta)$ |
|----------|---------------------------|------------------------------------------|
| Dimension | $2n$ (even) | $2n+1$ (odd) |
| Structure | 2-form $\omega$ | 1-form $\theta$ |
| Hamiltonian | Conserved | Not conserved |
| Volume | Preserved (Liouville) | Contracted/expanded |
| Physics | Conservative | Dissipative |
| Momentum eq. | $\dot{p} = -\partial_q H$ | $\dot{p} = -\partial_q H - p\, \partial_s H$ |
| Entropy eq. | N/A | $\dot{s} = p \cdot \partial_p H - H$ |

:::

(sec-nambu-mechanics-review)=
## Nambu Mechanics Review

:::{prf:definition} Nambu Bracket
:label: def-nambu-bracket

A **Nambu $n$-bracket** on a manifold $M$ is an $n$-linear, skew-symmetric map:

$$
\{-, \ldots, -\}: C^\infty(M)^n \to C^\infty(M)
$$

satisfying:

1. **Skew-symmetry:** $\{f_1, \ldots, f_n\}$ changes sign under transposition of any two arguments

2. **Leibniz rule:** 

$$
\{f_1 g, f_2, \ldots, f_n\} = f_1\{g, f_2, \ldots, f_n\} + g\{f_1, f_2, \ldots, f_n\}
$$

3. **Fundamental identity (FI):**

$$
\{f_1, \ldots, f_{n-1}, \{g_1, \ldots, g_n\}\} = \sum_{i=1}^n \{g_1, \ldots, \{f_1, \ldots, f_{n-1}, g_i\}, \ldots, g_n\}
$$

:::

:::{prf:definition} Poisson-Nambu Structure
:label: def-poisson-nambu-structure

On a manifold $M^{2n}$ with Casimir function $C: M \to \mathbb{R}$, the **Poisson-Nambu bracket** is:

$$
\{f, g\}_C := \{f, g, C\}_{\text{Nambu}}
$$

where $\{-, -, -\}_{\text{Nambu}}$ is a Nambu 3-bracket. This defines a Poisson bracket on each level set $C^{-1}(c)$.

*Key property:* The Casimir $C$ is automatically conserved: $\{C, g\}_C = \{C, g, C\} = 0$ by skew-symmetry.

:::

:::{prf:proposition} Nambu Conserves All Arguments
:label: prop-nambu-conserves-hamiltonians

For the Nambu evolution $\dot{f} = \{f, H_1, H_2\}$:

$$
\dot{H}_1 = \{H_1, H_1, H_2\} = 0, \qquad \dot{H}_2 = \{H_1, H_2, H_2\} = 0
$$

by skew-symmetry. This is why **pure Nambu mechanics cannot describe dissipation**.

:::

(sec-contact-nambu-synthesis)=
## Contact-Nambu Synthesis

The key insight is that contact geometry and Nambu mechanics serve **different roles**:
- **Contact:** Provides dissipation (friction, entropy production)
- **Nambu:** Provides constraint preservation (Casimirs, level sets)

We combine them by using contact dynamics as the primary structure, with Nambu constraints restricting the flow.

:::{prf:definition} Contact-Nambu System
:label: def-contact-nambu-system

A **Contact-Nambu system** $(M^{2n+1}, \theta, H, C)$ consists of:
- A contact manifold $(M, \theta)$ with coordinates $(q^i, p_i, s)$
- A contact Hamiltonian $H: M \to \mathbb{R}$
- A Casimir constraint $C: M \to \mathbb{R}$ (optional)

The **Contact-Nambu evolution** is the contact Hamiltonian flow (Definition {prf:ref}`def-contact-hamiltonian-vector-field`) restricted to level sets of $C$ when present.

*Interpretation:*
- Contact structure → dissipation ($-p \cdot \partial_s H$ in momentum, $p \cdot v - H$ for entropy)
- Casimir constraint → conserved quantity (e.g., total angular momentum, topological charge)

:::

:::{prf:theorem} Contact-Nambu Equations of Motion
:label: thm-contact-nambu-equations

For the Contact-Nambu system with $H(q, p, s) = K(q, p) + U(q) + \gamma s$ where $K = \frac{1}{2}G^{ij}p_i p_j$:

$$
\begin{cases}
\dot{q}^i = G^{ij} p_j \\[6pt]
\dot{p}_i = -\partial_i U - \gamma p_i \\[6pt]
\dot{s} = G^{ij} p_i p_j - K - U - \gamma s = K - U - \gamma s
\end{cases}
$$

*Proof.* Direct application of Definition {prf:ref}`def-contact-hamiltonian-vector-field`:
- $\dot{q}^i = \partial H / \partial p_i = G^{ij} p_j$ ✓
- $\dot{p}_i = -\partial H / \partial q^i - p_i \cdot \partial H / \partial s = -\partial_i U - \gamma p_i$ ✓
- $\dot{s} = p_i \cdot \partial H / \partial p_i - H = p_i G^{ij} p_j - (K + U + \gamma s) = 2K - K - U - \gamma s = K - U - \gamma s$ ✓

$\square$

:::

:::{prf:corollary} Energy Dissipation Rate
:label: cor-contact-energy-dissipation

For the Contact-Nambu system:

$$
\frac{dH}{dt} = -H \cdot \gamma = -\gamma(K + U + \gamma s)
$$

The mechanical energy $E = K + U$ evolves as:

$$
\frac{dE}{dt} = \frac{d(K + U)}{dt} = -2\gamma K
$$

Energy dissipates at twice the kinetic energy times friction coefficient.

:::

(sec-contact-nambu-langevin)=
## Contact-Nambu Formulation of Underdamped Langevin

We now show that the underdamped Langevin equation (Definition {prf:ref}`def-bulk-drift-continuous-flow`) is precisely a Contact-Nambu system on the Poincaré disk.

:::{prf:definition} Contact Phase Space for Langevin
:label: def-contact-phase-space-langevin

The **contact phase space** for the agent is:

$$
M = T^*\mathbb{D}^d \times \mathbb{R} \cong \{(z, p, s) : z \in \mathbb{D}^d, p \in T_z^*\mathbb{D}^d, s \in \mathbb{R}\}
$$

with contact form:

$$
\theta = ds - G_{ij}(z)\, p^i\, dz^j
$$

where $G_{ij}(z) = \frac{4\delta_{ij}}{(1-|z|^2)^2}$ is the Poincaré metric.

The contact Hamiltonian is:

$$
H(z, p, s) = \underbrace{\frac{1}{2}G^{ij}(z) p_i p_j}_{K} + \underbrace{\Phi_{\text{eff}}(z)}_{U} + \gamma s
$$

:::

:::{prf:theorem} Langevin from Contact Hamiltonian
:label: thm-langevin-from-contact

The contact Hamilton equations for $H = K + U + \gamma s$ on $(T^*\mathbb{D}^d \times \mathbb{R}, \theta)$ yield:

$$
\begin{cases}
\dot{z}^k = G^{kj}(z)\, p_j \\[6pt]
\dot{p}_k = -\partial_k \Phi_{\text{eff}} - \gamma\, p_k - \Gamma^m_{k\ell}\, G^{\ell j}\, p_j\, p_m \\[6pt]
\dot{s} = K - \Phi_{\text{eff}} - \gamma s
\end{cases}
$$

The first two equations match Definition {prf:ref}`def-bulk-drift-continuous-flow` exactly.

*Proof.* 
**Position:** $\dot{z}^k = \partial H / \partial p_k = G^{kj} p_j$ ✓

**Momentum:** On a Riemannian manifold, the contact Hamilton equation includes the Christoffel connection:

$$
\dot{p}_k = -\partial_k H - p_k \cdot \partial_s H - \Gamma^m_{k\ell} G^{\ell j} p_j p_m = -\partial_k \Phi_{\text{eff}} - \gamma p_k - \Gamma^m_{k\ell} G^{\ell j} p_j p_m
$$
✓

**Entropy:** $\dot{s} = p_k G^{kj} p_j - H = 2K - (K + U + \gamma s) = K - U - \gamma s$ ✓

$\square$

:::

:::{admonition} Geometric Origin of Friction
:class: important

The friction term $-\gamma p$ emerges **canonically** from the contact structure:

1. The contact form $\theta = ds - p\, dq$ defines the geometry
2. The Hamiltonian $H = K + U + \gamma s$ couples energy to entropy coordinate
3. The contact Hamilton equations automatically give $\dot{p} = -\partial_q H - p \cdot \partial_s H = -\partial_q H - \gamma p$

**No ad-hoc terms are added.** The friction coefficient $\gamma$ appears because we chose $H$ linear in $s$.

:::

:::{prf:definition} Thermodynamic Interpretation of Entropy Evolution
:label: def-thermodynamic-entropy

The contact entropy evolution $\dot{s} = K - U - \gamma s$ has clear thermodynamic meaning:

- **$K$ term:** Kinetic energy converted to heat (increases entropy)
- **$-U$ term:** Potential energy release/absorption  
- **$-\gamma s$ term:** Approach to equilibrium (entropy relaxation)

At equilibrium ($\dot{s} = 0$): $s_{\text{eq}} = (K - U)/\gamma$

For the Langevin system with thermal noise, the **heat dissipation rate** is:

$$
\dot{Q} = 2\gamma K = \gamma G^{ij} p_i p_j = \gamma |v|_G^2
$$

This is the rate of energy lost to the heat bath, matching the thermodynamic expectation.

:::

(sec-contact-nambu-stochastic)=
## Stochastic Contact-Nambu Dynamics

To recover the full Langevin equation with noise, we add thermal fluctuations. The noise amplitude is determined by the fluctuation-dissipation relation.

:::{prf:definition} Stochastic Contact Hamiltonian Equation
:label: def-stochastic-contact-hamiltonian

The **stochastic contact Hamiltonian equation** is:

$$
\begin{cases}
dz^k = G^{kj} p_j\, dt \\[6pt]
dp_k = \left(-\partial_k\Phi_{\text{eff}} - \gamma p_k - \Gamma^m_{k\ell} G^{\ell j} p_j p_m + u_{\pi,k}\right) dt + \sqrt{2\gamma T_c}\, (G^{1/2})_{kj}\, dW^j \\[6pt]
ds = (K - \Phi_{\text{eff}} - \gamma s)\, dt
\end{cases}
$$

where:
- $W^j$ is a standard Wiener process
- $T_c > 0$ is the cognitive temperature
- $u_{\pi,k}$ is the control field from policy
- The noise coefficient $\sqrt{2\gamma T_c}$ satisfies **fluctuation-dissipation**

This is precisely Definition {prf:ref}`def-bulk-drift-continuous-flow` with explicit entropy tracking.

:::

:::{prf:theorem} Fluctuation-Dissipation Relation
:label: thm-fluctuation-dissipation

The noise amplitude $\sigma = \sqrt{2\gamma T_c}$ is uniquely determined by requiring the stationary distribution to be Boltzmann:

$$
\rho_*(z, p) \propto \exp\left(-\frac{K(z,p) + \Phi_{\text{eff}}(z)}{T_c}\right)
$$

*Proof.* The Fokker-Planck equation for the $(z, p)$ marginal:

$$
\partial_t \rho = -\nabla_z \cdot (\dot{z}\, \rho) - \nabla_p \cdot (\dot{p}\, \rho) + \frac{1}{2}\nabla_p \cdot (D \nabla_p \rho)
$$

has stationary solution $\rho_* \propto e^{-H/T_c}$ iff:

$$
D_{ij} = 2\gamma T_c\, G_{ij}
$$

This is Einstein's fluctuation-dissipation relation. $\square$

:::

(sec-contact-nambu-ness)=
## Non-Equilibrium Steady States

When the effective potential $\Phi_{\text{eff}}$ has non-conservative components (Value Curl $\mathcal{F} \neq 0$), the system reaches a **non-equilibrium steady state (NESS)**.

:::{prf:definition} Contact Hamiltonian with Value Curl
:label: def-contact-value-curl

When the reward field has curl, we add minimal coupling:

$$
H(z, p, s) = \frac{1}{2}G^{ij} p_i p_j + \Phi_{\text{eff}}(z) + \gamma s + \beta_{\text{curl}}\, \mathcal{A}_i(z)\, G^{ij}\, p_j
$$

where $\mathcal{F}_{ij} = \partial_i \mathcal{A}_j - \partial_j \mathcal{A}_i$ is the Value Curl.

The momentum equation becomes:

$$
\dot{p}_k = -\partial_k\Phi_{\text{eff}} - \gamma p_k + \beta_{\text{curl}}\, \mathcal{F}_{kj}\, G^{j\ell}\, p_\ell + \text{(geodesic terms)}
$$

The Lorentz-like force $\mathcal{F} \cdot v$ causes spiraling trajectories.

:::

:::{prf:theorem} NESS Entropy Production
:label: thm-ness-entropy-production

In a NESS with $\mathcal{F} \neq 0$, the average entropy production rate is:

$$
\langle \dot{S}_{\text{prod}} \rangle = \frac{\gamma}{T_c} \langle |v|_G^2 \rangle_{\text{NESS}} > 0
$$

The system continuously dissipates energy while being driven by the rotational Value Curl force.

:::

(sec-contact-nambu-integrator)=
## Contact-Nambu BAOAB Integrator

:::{prf:definition} Contact BAOAB
:label: def-contact-baoab

The **Contact BAOAB** integrator for Definition {prf:ref}`def-stochastic-contact-hamiltonian`:

1. **B** (half kick): $p \leftarrow p - \frac{h}{2}\nabla\Phi_{\text{eff}}$ + Boris rotation if $\mathcal{F} \neq 0$
   
2. **A** (half drift): 
   - $z \leftarrow \operatorname{Exp}_z\left(\frac{h}{2} G^{-1} p\right)$
   - $s \leftarrow s + \frac{h}{2}(K - \Phi_{\text{eff}} - \gamma s)$

3. **O** (thermostat): $p \leftarrow c_1 p + c_2\, G^{1/2}\, \xi$ where $c_1 = e^{-\gamma h}$, $c_2 = \sqrt{(1-c_1^2)T_c}$

4. **A** (half drift): same as step 2

5. **B** (half kick): same as step 1

*Remark:* The O-step handles friction and noise together via the exact Ornstein-Uhlenbeck solution.

:::

**Algorithm (Contact BAOAB Implementation):**

```python
import torch
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ContactState:
    """State of the Contact Hamiltonian integrator."""
    z: torch.Tensor          # [B, d] latent position (Poincaré disk)
    p: torch.Tensor          # [B, d] covariant momentum
    s: torch.Tensor          # [B] entropy coordinate
    t: float                 # computation time


def poincare_metric_inv(z: torch.Tensor) -> torch.Tensor:
    """Inverse Poincaré metric G^{ij} = (1-|z|²)²/4 δ^{ij}."""
    B, d = z.shape
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    inv_conformal = (1.0 - r_sq + 1e-8) ** 2 / 4.0
    return inv_conformal.unsqueeze(-1) * torch.eye(d, device=z.device).expand(B, d, d)


def poincare_exp_map(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Exponential map on Poincaré disk: Exp_z(v)."""
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - r_sq + 1e-8)
    v_norm = torch.sqrt((v ** 2).sum(dim=-1, keepdim=True) + 1e-8) * lambda_z
    v_dir = v / (torch.sqrt((v ** 2).sum(dim=-1, keepdim=True)) + 1e-8)
    magnitude = torch.tanh(v_norm / 2.0)
    w = magnitude * v_dir
    
    # Möbius addition z ⊕ w
    z_sq, w_sq = (z**2).sum(-1, True), (w**2).sum(-1, True)
    z_dot_w = (z * w).sum(-1, True)
    num = (1 + 2*z_dot_w + w_sq) * z + (1 - z_sq) * w
    return num / (1 + 2*z_dot_w + z_sq * w_sq + 1e-8)


def christoffel_contraction(z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Christoffel contraction Γ^k_{ij}v^iv^j for Poincaré disk."""
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    v_sq = (v ** 2).sum(dim=-1, keepdim=True)
    z_dot_v = (z * v).sum(dim=-1, keepdim=True)
    denom = 1.0 - r_sq + 1e-8
    return (4.0 / denom) * z_dot_v * v - (2.0 / denom) * v_sq * z


def contact_baoab_step(
    state: ContactState,
    grad_Phi: torch.Tensor,           # [B, d] gradient of effective potential
    Phi_eff: torch.Tensor,            # [B] effective potential value
    u_pi: Optional[torch.Tensor],     # [B, d] control field (optional)
    T_c: float,                       # cognitive temperature
    gamma: float,                     # friction coefficient
    h: float,                         # time step
    F: Optional[torch.Tensor] = None, # [B, d, d] Value Curl tensor
    beta_curl: float = 0.0,
) -> ContactState:
    """
    Contact BAOAB integrator.
    
    Implements stochastic contact Hamiltonian dynamics:
    - Friction from contact structure: -γp in O-step
    - Entropy evolution from contact Hamilton equation
    - Fluctuation-dissipation balanced noise
    """
    z, p, s = state.z, state.p, state.s
    B, d = z.shape
    device = z.device
    
    if u_pi is None:
        u_pi = torch.zeros_like(p)
    
    # O-step coefficients
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c) if T_c > 0 else 0.0
    
    # ===== B-step: half kick =====
    p = p - (h / 2) * (grad_Phi - u_pi)
    
    # Boris rotation for Value Curl (angle depends on field magnitude, not force)
    if beta_curl > 0 and F is not None and d == 2:
        # F is antisymmetric in 2D, so |F| = |F_{01}| = |F_{10}|
        F_magnitude = torch.abs(F[:, 0, 1]).unsqueeze(-1)  # [B, 1]
        angle = (h / 2) * beta_curl * F_magnitude  # cyclotron-like rotation
        cos_a, sin_a = torch.cos(angle).squeeze(-1), torch.sin(angle).squeeze(-1)
        p = torch.stack([cos_a*p[...,0] - sin_a*p[...,1], 
                         sin_a*p[...,0] + cos_a*p[...,1]], dim=-1)
    
    # ===== A-step: half drift + entropy =====
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    geodesic_corr = christoffel_contraction(z, velocity)
    z = poincare_exp_map(z, (h/2) * (velocity - (h/4)*geodesic_corr))
    
    # Kinetic energy K = (1/2) p^T G^{-1} p
    K = 0.5 * (p * velocity).sum(dim=-1)
    
    # Entropy: ds = (K - Φ - γs) dt  [contact Hamilton equation]
    s = s + (h / 2) * (K - Phi_eff - gamma * s)
    
    # ===== O-step: thermostat =====
    r_sq = (z ** 2).sum(dim=-1, keepdim=True)
    G_sqrt = 2.0 / (1.0 - r_sq + 1e-8)
    p = c1 * p + c2 * G_sqrt * torch.randn(B, d, device=device)
    
    # ===== A-step: half drift + entropy =====
    G_inv = poincare_metric_inv(z)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    geodesic_corr = christoffel_contraction(z, velocity)
    z = poincare_exp_map(z, (h/2) * (velocity - (h/4)*geodesic_corr))
    
    K = 0.5 * (p * velocity).sum(dim=-1)
    s = s + (h / 2) * (K - Phi_eff - gamma * s)
    
    # ===== B-step: half kick =====
    p = p - (h / 2) * (grad_Phi - u_pi)
    
    if beta_curl > 0 and F is not None and d == 2:
        F_magnitude = torch.abs(F[:, 0, 1]).unsqueeze(-1)
        angle = (h / 2) * beta_curl * F_magnitude
        cos_a, sin_a = torch.cos(angle).squeeze(-1), torch.sin(angle).squeeze(-1)
        p = torch.stack([cos_a*p[...,0] - sin_a*p[...,1], 
                         sin_a*p[...,0] + cos_a*p[...,1]], dim=-1)
    
    # Project to disk interior
    z_norm = torch.sqrt((z ** 2).sum(-1, True))
    z = torch.where(z_norm > 0.999, z * 0.999 / z_norm, z)
    
    return ContactState(z=z, p=p, s=s, t=state.t + h)
```

:::

(sec-contact-nambu-summary)=
## Summary: Contact-Nambu Framework

:::{admonition} Framework Correspondence
:class: important

| Framework Component | Contact-Nambu Correspondence |
|---------------------|------------------------------|
| Underdamped Langevin (Def. {prf:ref}`def-bulk-drift-continuous-flow`) | Stochastic Contact Hamiltonian (Def. {prf:ref}`def-stochastic-contact-hamiltonian`) |
| Friction $-\gamma p$ | Contact term $-p \cdot \partial_s H$ with $H = K + U + \gamma s$ |
| Effective potential $\Phi_{\text{eff}}$ | Potential $U$ in Hamiltonian |
| Cognitive temperature $T_c$ | Fluctuation-dissipation: $\sigma = \sqrt{2\gamma T_c}$ |
| Value Curl $\mathcal{F}$ | Minimal coupling $\mathcal{A} \cdot p$ in Hamiltonian |
| BAOAB integrator | Contact BAOAB (Def. {prf:ref}`def-contact-baoab`) |
| Entropy production | Contact equation $\dot{s} = p \cdot v - H$ |

:::

:::{prf:definition} The Geometric Hierarchy
:label: def-geometric-hierarchy

From most to least restrictive:

1. **Hamiltonian** (symplectic): $\dot{f} = \{f, H\}$, energy conserved
2. **Nambu**: $\dot{f} = \{f, H_1, H_2\}$, two Casimirs conserved
3. **Contact**: $\dot{f} = X_H(f)$, intrinsic dissipation via $\partial_s H$
4. **Contact + Casimir**: Contact dynamics with conserved constraints ← **Our framework**
5. **Stochastic Contact**: Add fluctuation-dissipation noise

The agent framework uses **stochastic contact Hamiltonian dynamics** — the geometric formulation of underdamped Langevin.

:::

:::{admonition} Why Contact Geometry?
:class: tip

**What contact geometry provides:**
1. Friction $-\gamma p$ from the term $-p \cdot \partial_s H$ (no ad-hoc addition)
2. Entropy evolution $\dot{s} = p \cdot v - H$ (thermodynamically consistent)
3. Energy dissipation $\dot{E} = -2\gamma K$ (heat loss to bath)
4. Natural coupling to thermodynamics (temperature, equilibrium)

**What must still be specified:**
1. The noise amplitude (via fluctuation-dissipation: $\sigma = \sqrt{2\gamma T_c}$)
2. The temperature $T_c$ (determines equilibrium distribution)
3. The potential $\Phi_{\text{eff}}$ (encodes rewards/control)

**Connection to Nambu:** When additional conserved quantities (Casimirs) are needed, they can be imposed as constraints on the contact flow. The Nambu structure provides a systematic way to do this while preserving the dissipative dynamics.

:::

## References

- {cite}`bravetti2017contact` — Contact Hamiltonian mechanics
- {cite}`bravetti2019contact` — Contact geometry and thermodynamics  
- {cite}`ciaglia2018contact` — Contact manifolds and information geometry
- {cite}`nambu1973generalized` — Original Nambu mechanics paper
- {cite}`takhtajan1994foundation` — Mathematical foundations of Nambu mechanics
- {cite}`leimkuhler2016computation` — BAOAB integrator
