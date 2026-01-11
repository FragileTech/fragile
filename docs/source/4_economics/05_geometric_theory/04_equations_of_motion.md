# Market Equations of Motion: Portfolio Geodesics and Jump-Diffusion

:::{admonition} Researcher Bridge: Continuous-Time Portfolio Dynamics
:class: info
:name: rb-portfolio-dynamics

If you're familiar with Hamiltonian mechanics or Langevin dynamics, the portfolio equations of motion follow the same structure. The **position** is the portfolio allocation, the **momentum** is the trading velocity, and the **mass** is the risk metric. The geodesic equation ensures that the portfolio follows minimum-risk paths in the curved geometry of return space.

This is the continuous-time limit of discrete rebalancing, with the Christoffel symbols capturing cross-asset correlation effects that are invisible in single-asset models.
:::

The portfolio follows a **geodesic jump-diffusion** on the risk manifold. This section derives the complete equations of motion for portfolio evolution, including:
1. **Geodesic drift:** Minimum-risk paths on the curved Ruppeiner manifold
2. **Return gradient:** Move toward high risk-adjusted returns
3. **Alpha signal:** Policy-induced trades
4. **Regime jumps:** Discrete transitions between investment regimes
5. **Market noise:** Stochastic diffusion from market fluctuations

(sec-position-inertia)=
## Position Inertia: Mass = Risk Metric

**Definition 27.1.1 (Position Inertia Tensor).** The **position inertia** is the Ruppeiner risk metric:
$$
\mathbf{M}(w) := G(w).
$$

**Operational consequences:**
- **High-risk positions** (large $G$) have large inertia → smaller rebalancing per unit signal.
- **Low-risk positions** (small $G$) have small inertia → larger rebalancing allowed.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Mass tensor $\mathbf{M}(z)$ | Position inertia |
| Kinetic energy $\frac{1}{2}\mathbf{M}\|\dot{z}\|^2$ | Trading cost / market impact |
| Potential $\Phi_{\text{eff}}$ | Risk-adjusted return landscape |
| Christoffel symbols $\Gamma^k_{ij}$ | Cross-asset correlation corrections |

**Risk-Metric Coupling (Market Natural Gradient):**
$$
\text{High risk } T_{ij} \;\Rightarrow\; \text{Large } G_{ij} \;\Rightarrow\; \text{Large } \mathbf{M}_{ij} \;\Rightarrow\; \text{Reduced trade size}
$$

## Portfolio Jump-Diffusion SDE

**Definition 27.2.1 (Portfolio Geodesic SDE).** The portfolio weights $w^k$ evolve according to:
$$
dw^k = \underbrace{\left( -G^{kj}\partial_j \Phi_{\text{risk}} + u_\pi^k \right)}_{\text{Drift (signal + policy)}} ds - \underbrace{\Gamma^k_{ij}\dot{w}^i \dot{w}^j\,ds}_{\text{Correlation correction}} + \underbrace{\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s}_{\text{Market noise}},
$$
where:
- $\Phi_{\text{risk}}$ is the risk-adjusted return potential,
- $u_\pi^k$ is the alpha signal (policy control),
- $\Gamma^k_{ij}$ are Christoffel symbols of the risk metric (correlation structure),
- $T_c$ is the market temperature (volatility scaling).

**Three-Force Decomposition:**
1. **Return gradient:** $-G^{-1}\nabla\Phi_{\text{risk}}$ — move toward high risk-adjusted returns.
2. **Alpha signal:** $u_\pi$ — policy-induced trades (momentum, value, etc.).
3. **Correlation correction:** $-\Gamma(\dot{w},\dot{w})$ — adjusts for cross-asset dependencies.

::::{admonition} Physics Isomorphism: Hamiltonian Mechanics on Portfolio Space
:class: note
:name: pi-hamiltonian-portfolio

**In Physics:** Hamiltonian mechanics describes particle motion via position $q$ and conjugate momentum $p$. The **second-order** geodesic equation with dissipation is:
$$\ddot{q}^k + \Gamma^k_{ij}\dot{q}^i\dot{q}^j = -\gamma \dot{q}^k + \text{force}^k$$

In the **overdamped limit** ($\gamma \to \infty$), this reduces to **first-order** gradient flow:
$$\dot{q}^k = -M^{-1,kj}\partial_j V + \text{noise}$$

**In Markets:** Portfolio evolution follows the overdamped (first-order) limit, since transaction costs dominate inertia:
$$\dot{w}^k = -G^{kj}\partial_j \Phi_{\text{risk}} + u_\pi^k + \text{noise}$$

The BAOAB integrator (Section 27.5) handles the intermediate regime where momentum effects matter.

**Correspondence Table:**

| Classical Mechanics | Market (Portfolio Dynamics) |
|:-------------------|:---------------------------|
| Position $q$ | Portfolio weights $w$ |
| Momentum $p = M\dot{q}$ | Trading velocity $p = G\dot{w}$ |
| Mass tensor $M$ | Risk metric $G$ (position inertia) |
| Potential energy $V(q)$ | Risk-adjusted return potential $\Phi_{\text{risk}}$ |
| Kinetic energy $\frac{1}{2}p^T M^{-1} p$ | Trading cost $\frac{1}{2}\dot{w}^T G \dot{w}$ |
| Friction $\gamma$ | Transaction cost coefficient |
| External force | Alpha signal $u_\pi$ |
| Christoffel symbols $\Gamma^k_{ij}$ | Cross-asset correlation corrections |
| Overdamped limit | Standard portfolio optimization |

**The geodesic principle:** In the absence of external forces, portfolios follow geodesics (minimum-risk paths) on the Ruppeiner manifold. In the overdamped limit, these become gradient flows.
::::

::::{note} Connection to Standard Finance #20: CAPM as Degenerate Geodesic
**The General Law (Fragile Market):**
Portfolio weights evolve via **geodesic jump-diffusion**:
$$
dw^k = \left( -G^{kj}\partial_j \Phi_{\text{risk}} + u_\pi^k \right) ds - \Gamma^k_{ij}\dot{w}^i \dot{w}^j\,ds + \sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s
$$
with Christoffel symbols $\Gamma^k_{ij}$ encoding cross-asset correlations.

**The Degenerate Limit:**
Flat geometry ($G \to I$, $\Gamma \to 0$). Single-period optimization ($\dot{w} \to 0$). Mean-variance potential.

**The Special Case (CAPM):**
$$
\mathbb{E}[R_i] - R_f = \beta_i (\mathbb{E}[R_m] - R_f), \quad \beta_i = \frac{\text{Cov}(R_i, R_m)}{\text{Var}(R_m)}
$$
This recovers the **Capital Asset Pricing Model** in the limit of:
- Flat geometry ($G \to \Sigma$, constant covariance)
- No geodesic corrections ($\Gamma \to 0$)
- Equilibrium ($\dot{w} = 0$)
- Single-period ($T \to 0$)

**What the generalization offers:**
- **Curved geometry**: Risk metric $G$ varies across portfolio space—diversification benefits are state-dependent
- **Path dependency**: Christoffel symbols correct for multi-asset rebalancing effects
- **Continuous dynamics**: BAOAB integrator provides stable numerical evolution
- **Regime awareness**: Jump process handles structural breaks
::::

(sec-regime-jump-process)=
## Regime Jump Process

**Definition 27.3.1 (Regime Jump Intensity).** The intensity of jumping from regime $i$ to regime $j$ is:
$$
\lambda_{i \to j}(w) = \lambda_0 \cdot \exp\left(\beta \cdot \left( V_j(w) - V_i(w) - c_{\text{switch}} \right) \right),
$$
where:
- $V_i, V_j$ are regime-specific value functions,
- $c_{\text{switch}}$ is the regime transition cost,
- $\beta$ is inverse temperature (sharpness).

**Interpretation:** Regime transitions occur when $V_j(w) - V_i(w) > c_{\text{switch}}$, with rate exponentially increasing in the value differential.

## Effective Return Potential

**Definition 27.4.1 (Effective Return Potential).** The unified potential is:
$$
\Phi_{\text{risk}}(w, K) = \alpha \cdot U(w) + (1 - \alpha) \cdot V_{\text{alpha}}(w, K) + \gamma_{\text{risk}} \cdot \Psi_{\text{risk}}(w),
$$
where:
- $U(w)$ is the information potential (spread compression),
- $V_{\text{alpha}}(w, K)$ is the alpha signal (expected returns),
- $\Psi_{\text{risk}}(w) = \frac{1}{2}\text{tr}(T_{ij} G^{ij})$ is risk concentration.

| $\alpha$ | Behavior | Strategy Type |
|----------|----------|---------------|
| $\alpha = 1$ | Pure liquidity provision | Market making |
| $\alpha = 0$ | Pure alpha capture | Directional trading |
| $\alpha = 0.5$ | Balanced | Mixed strategy |

## BAOAB Portfolio Integrator

**Algorithm 27.5.1 (Portfolio BAOAB Step).**

```python
def portfolio_baoab_step(
    w: torch.Tensor,        # Portfolio weights [B, N_assets]
    p: torch.Tensor,        # Momentum (trading velocity) [B, N_assets]
    regime: torch.Tensor,   # Regime index [B]
    grad_Phi: torch.Tensor, # Return gradient [B, N_assets]
    u_alpha: torch.Tensor,  # Alpha signal [B, N_assets]
    G: torch.Tensor,        # Risk metric [B, N, N]
    T_c: float,             # Market temperature
    gamma: float,           # Friction (transaction costs)
    h: float,               # Time step
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Portfolio BAOAB integrator with geodesic corrections.

    B-A-O-A-B splitting:
    - B: Momentum kick from returns + alpha
    - A: Position drift (portfolio update)
    - O: Market noise (Ornstein-Uhlenbeck)
    """
    c1 = math.exp(-gamma * h)
    c2 = math.sqrt((1 - c1**2) * T_c)

    # B-step: half kick
    total_force = grad_Phi - u_alpha
    p = p - (h / 2) * total_force

    # A-step: half drift
    G_inv = torch.linalg.inv(G)
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    w = w + (h / 2) * velocity

    # O-step: market noise
    G_sqrt = torch.linalg.cholesky(G)
    xi = torch.randn_like(p)
    p = c1 * p + c2 * torch.einsum('bij,bj->bi', G_sqrt, xi)

    # A-step: half drift
    velocity = torch.einsum('bij,bj->bi', G_inv, p)
    w = w + (h / 2) * velocity

    # B-step: half kick
    p = p - (h / 2) * total_force

    # Normalize to simplex
    w = F.softmax(w, dim=-1)

    return w, p
```

## Market Dynamics Diagnostics

Following the diagnostic node convention (Section 7), we define the geodesic consistency gate:

:::{prf:definition} Gate43 Specification
:label: def-gate43-specification

**Predicate:** Portfolio trajectory satisfies geodesic equation.
$$
P_{43} : \quad \left\|\ddot{w}^k + \Gamma^k_{ij}\dot{w}^i\dot{w}^j + G^{kj}\partial_j\Phi_{\text{risk}} - u_\pi^k\right\|_G \le \epsilon_{\text{geo}},
$$
where $\epsilon_{\text{geo}}$ is the geodesic tolerance.

**Market interpretation:** The portfolio is following a minimum-risk path given the current return landscape and alpha signals.

**Observable metrics:**
- Geodesic residual $\|\text{EOM violation}\|_G$
- Path curvature $\kappa = \|\ddot{w} + \Gamma(\dot{w},\dot{w})\|$
- Alpha contribution $\|u_\pi\|$
- Return gradient magnitude $\|G^{-1}\nabla\Phi\|$

**Certificate format:**
$$
K_{43}^+ = (\text{residual}, \kappa, \|u_\pi\|, \|\nabla\Phi\|, \text{regime})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{geo}} = \lambda_{43} \cdot \left\|\ddot{w} + \Gamma(\dot{w},\dot{w}) + G^{-1}\nabla\Phi - u_\pi\right\|_G^2
$$
:::

**Node GateGeodesic: Geodesic Consistency Check**

| **#**  | **Name**         | **Component**    | **Type**                | **Interpretation**          | **Proxy**                                                                | **Cost**  |
|--------|------------------|------------------|-------------------------|-----------------------------|--------------------------------------------------------------------------|-----------|
| **Gate43** | **GeodesicCheck** | Portfolio Model  | Trajectory Consistency  | Is portfolio path geodesic? | $\|\ddot{w} + \Gamma(\dot{w},\dot{w}) + G^{-1}\nabla\Phi - u_\pi\|_G$ | $O(BN^2)$ |

**Trigger conditions:**
- High geodesic residual: Portfolio is not following the optimal path.
- Remedy: Check for alpha signal miscalibration; verify risk metric consistency.

---

