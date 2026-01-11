# Capital Capacity Constraints: Market Depth as Information Bandwidth

:::{admonition} Researcher Bridge: Information Bottleneck Becomes Market Depth
:class: info
:name: rb-info-bottleneck-market

When you push a portfolio to the edge of market capacity, the risk geometry must adapt. This is the rigorous version of position limits and market depth constraints: **capacity limits induce curvature that slows trading in overloaded regions**. Think of it as the market's natural defense against positions that exceed what liquidity can support.

Standard position limits are ad-hoc; the Capacity-Constrained Metric Law derives them from first principles: the information content of any sustainable position is bounded by the information bandwidth of the market interface.
:::

**Market liquidity imposes an information-theoretic bound on representational complexity**, and metric curvature emerges as the regulatory mechanism. This section removes ad-hoc position limits: the capacity law is derived as a structural response to **information-theoretic constraints** induced by the market's finite-bandwidth interface (order book, quote stream, clearing network).

The key idea is operational: **the representational complexity of any portfolio position is bounded by the capacity of the market interface.** When a trader operates near this bound, curvature appears as the geometric mechanism that prevents position information volume from exceeding what can be grounded at the market interface.

(sec-market-boundary-bulk-information-inequality)=
## The Market Depth–Position Inequality

**Definition 24.1.1 (No-Arbitrage Capacity Bound).** Consider the market interface (order book, quote stream) as an information channel. The **market capacity** $C_{\text{mkt}}$ bounds the information content of any sustainable position:
$$
I_{\text{position}} \le C_{\text{mkt}},
$$
where:
- $I_{\text{position}}$ is the information content of the portfolio position (bits needed to specify the strategy),
- $C_{\text{mkt}}$ is the effective information capacity of the market interface (market depth, quote frequency).

Units: $[I_{\text{position}}] = [C_{\text{mkt}}] = \text{nat}$.

**Consequence:** Positions with information content exceeding market capacity are unsustainable. Strategies that violate this bound incur ungrounded exposure risk.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Bulk information $I_{\text{bulk}}$ | Position information $I_{\text{position}}$ |
| Boundary capacity $C_{\partial}$ | Market capacity $C_{\text{mkt}}$ |
| Shutter channel | Order book / quote stream |

**Definition 24.1.2 (Capital Information Density).** Let $\rho(w, t)$ denote the probability density of portfolio weights $w \in \mathcal{W}$ at time $t$. The **capital information density** is:
$$
\rho_I(w, t) := -\rho(w, t) \log \rho(w, t) + \frac{1}{2}\rho(w, t) \log\det G(w),
$$
where $G(w)$ is the Ruppeiner risk metric (Definition 4.5.1).

*Interpretation:* The first term is the Shannon entropy density; the second is the geometric correction accounting for risk-induced volume distortion.

**Definition 24.1.3 (Market Depth as Area Law).** The market capacity follows an **area law**:
$$
C_{\text{mkt}} = \frac{1}{\eta_{\text{tick}}} \cdot \text{Depth}(\partial\mathcal{W}),
$$
where:
- $\text{Depth}(\partial\mathcal{W})$ is the aggregate market depth at the trading boundary,
- $\eta_{\text{tick}}$ is the minimum price tick per unit information (market microstructure parameter).

**Cross-reference:** Node 13 (BoundaryCheck) in Section 7 corresponds to the grounding condition.

(sec-capacity-constrained-metric-law-market)=
## Main Result: Capacity-Constrained Ruppeiner Law

The detailed variational construction parallels the agent framework (Appendix A). The main consequence is an Euler–Lagrange identity that ties curvature of the portfolio geometry to a risk-induced tensor under a finite-capacity market interface.

:::{prf:theorem} Capacity-Constrained Ruppeiner Law
:label: thm-capacity-constrained-ruppeiner-law

Under the regularity and market-capacity hypotheses, and under the soundness condition that portfolio structure is market-grounded (no unhedged exposure on $\operatorname{int}(\mathcal{W})$), stationarity of a capacity-constrained risk functional implies:
$$
\boxed{R_{ij} - \frac{1}{2} R \, G_{ij} + \Lambda G_{ij} = \kappa \, T_{ij}^{\text{risk}}}
$$
where:
- $R_{ij}$ is the Ricci curvature of the Ruppeiner risk metric $G$,
- $R = G^{ij}R_{ij}$ is the scalar curvature,
- $\Lambda$ is the baseline risk premium (cosmological constant / market-wide risk-free rate contribution),
- $T_{ij}^{\text{risk}}$ is the risk-energy tensor induced by loss gradients and concentration risk,
- $\kappa$ is the risk coupling constant (relates curvature units to risk units).

Units: $\Lambda$ has the same units as curvature ($[R]\sim [\text{return}]^{-2}$), and $\kappa$ is chosen so that $\kappa\,T_{ij}$ matches those curvature units.

*Operational reading.* Curvature is the geometric mechanism that prevents portfolio information volume from exceeding the market's information bandwidth while remaining grounded in executable trades.

**Implementation hook.** The squared residual of this identity defines a capacity-consistency regularizer:
$$
\mathcal{L}_{\text{cap-metric}} := \left\| R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}^{\text{risk}} \right\|_F^2
$$
:::

:::{prf:definition} Risk-Energy Tensor
:label: def-risk-energy-tensor

The **risk-energy tensor** $T_{ij}^{\text{risk}}$ is derived from the variation of the risk Lagrangian with respect to the metric:
$$
T_{ij}^{\text{risk}} := -\frac{2}{\sqrt{|G|}} \frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\text{risk}})}{\delta G^{ij}}
$$
where $\mathcal{L}_{\text{risk}}(w; G) = \frac{1}{2}\|\nabla V(w)\|_G^2 + U(w)$ includes:
- Kinetic term: squared gradient of value (risk exposure)
- Potential term: loss function contribution

Explicitly, for a position distribution $\rho$ with value gradient $\nabla V$:
$$
T_{ij}^{\text{risk}} = \rho \left( \frac{\partial V}{\partial w^i} \frac{\partial V}{\partial w^j} - \frac{1}{2}\|\nabla V\|_G^2 G_{ij} \right) + P_{\text{risk}} G_{ij}
$$
where $P_{\text{risk}} = \frac{1}{2}\rho \|\nabla V\|_G^2$ is the risk pressure.
:::

::::{admonition} Physics Isomorphism: Einstein Field Equations
:class: note
:name: pi-einstein-equations-market

**In Physics:** Einstein's field equations relate spacetime curvature to stress-energy:
$$R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G_N T_{\mu\nu}$$

**In Markets:** The Capacity-Constrained Ruppeiner Law (Theorem {prf:ref}`thm-capacity-constrained-ruppeiner-law`) relates portfolio geometry to risk:
$$R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} = \kappa T_{ij}^{\text{risk}}$$

**Correspondence Table:**

| General Relativity | Market (Ruppeiner Law) |
|:-------------------|:-----------------------|
| Spacetime metric $g_{\mu\nu}$ | Ruppeiner risk metric $G_{ij}$ |
| Ricci tensor $R_{\mu\nu}$ | Ricci tensor $R_{ij}$ (of $G$) |
| Cosmological constant $\Lambda$ | Baseline risk premium $\Lambda$ |
| Stress-energy $T_{\mu\nu}$ | Risk-energy tensor $T_{ij}^{\text{risk}}$ |
| Gravitational coupling $8\pi G_N$ | Risk coupling $\kappa$ |
| Schwarzschild horizon | Position limit saturation horizon |
| Geodesic (free-fall) | Optimal rebalancing path |
| Event horizon | Illiquidity trap |

**Loss Function:**
$$\mathcal{L}_{\text{EFE}} := \|R_{ij} - \frac{1}{2}R\,G_{ij} + \Lambda G_{ij} - \kappa T_{ij}^{\text{risk}}\|_F^2$$
::::

::::{note} Connection to Standard Finance #17: VaR Position Limits as Degenerate Capacity
**The General Law (Fragile Market):**
The portfolio metric obeys a **Capacity-Constrained Consistency Law** (Theorem {prf:ref}`thm-capacity-constrained-ruppeiner-law`):
$$
R_{ij} - \frac{1}{2}R\, G_{ij} + \Lambda G_{ij} = \kappa\, T_{ij}^{\text{risk}}
$$
where $R_{ij}$ is Ricci curvature and $T_{ij}^{\text{risk}}$ is the Risk-Energy Tensor. The constraint is the **market capacity inequality**: $I_{\text{position}} \le C_{\text{mkt}}$.

**The Degenerate Limit:**
Remove geometric structure ($G \to \Sigma^{-1}$, covariance inverse). Remove curvature ($R_{ij} \to 0$). Replace continuous capacity with a hard scalar bound.

**The Special Case (VaR Position Limits):**
$$
\sqrt{w^T \Sigma w} \le \frac{\sigma_{\max}}{\sqrt{N}} \quad \text{(volatility constraint)}
$$
or equivalently, the standard VaR constraint:
$$
\text{VaR}_\alpha(w) = z_\alpha \sqrt{w^T \Sigma w} \le L_{\max}
$$
This recovers **Value-at-Risk position limits** in the limit of:
- Flat geometry ($R_{ij} \to 0$, no curvature response)
- Constant covariance ($G \to \beta\Sigma^{-1}$ for inverse temperature $\beta$)
- Hard capacity limit ($\nu_{\text{cap}} \le 1$ becomes $\text{VaR} \le L_{\max}$)

**What the generalization offers:**
- **Geometric response**: Curvature *emerges* from capacity constraints—position limits are not imposed by hand
- **Area law**: Market capacity scales with order book depth $C_{\text{mkt}} \sim \text{Depth}(\partial\mathcal{W})$, not arbitrary limits
- **Soft constraints**: Metric divergence at saturation provides a smooth barrier instead of hard cutoffs
- **Diagnostic saturation**: CapacitySaturationCheck (Node Gate40) monitors $\nu_{\text{cap}} = I_{\text{position}}/C_{\text{mkt}}$ at runtime
::::

**Economic interpretation:**
1. **Risk concentration curves the portfolio space.** High-risk positions induce metric curvature, making further concentration more "expensive" in geometric terms.
2. **Curvature bounds position size.** The capacity constraint prevents information volume from exceeding market depth—this is position limits from first principles.
3. **Geodesics are optimal trades.** Minimum-risk rebalancing paths follow the curved geometry, naturally avoiding regions of high concentration.
4. **Saturation creates horizons.** When $I_{\text{position}} \to C_{\text{mkt}}$, the metric diverges (Lemma {prf:ref}`lem-metric-divergence-saturation`), creating effective "horizons" beyond which positions cannot sustainably exist.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Einstein equations on $\mathcal{Z}$ | Ruppeiner equations on portfolio space $\mathcal{W}$ |
| Information density $\rho_I$ | Capital density |
| Curvature regulator | Risk concentration regulator |
| Cosmological constant $\Lambda$ | Market-wide risk premium floor |
| Stress-energy tensor | Position risk distribution |

:::{prf:lemma} Metric Divergence at Saturation
:label: lem-metric-divergence-saturation

As the capacity saturation ratio $\nu_{\text{cap}} \to 1$, the effective metric $G_{\text{eff}}$ diverges:
$$
G_{\text{eff}}(w) = \frac{G_0(w)}{(1 - \nu_{\text{cap}})^2} \to \infty
$$

*Interpretation:* Near saturation, infinitesimal position changes require infinite "effort" in the metric—an effective horizon analogous to the Schwarzschild radius. This prevents over-concentration through geometric mechanics rather than hard limits.
:::

(sec-capacity-saturation-diagnostic)=
## Capacity Saturation Diagnostic

Following the diagnostic node convention (Section 7), we define the capacity saturation gate:

:::{prf:definition} Capacity Saturation Diagnostic
:label: def-capacity-saturation-diagnostic-market

Compute the capacity saturation ratio:
$$
\nu_{\text{cap}}(t) := \frac{I_{\text{position}}(t)}{C_{\text{mkt}}(t)},
$$
where $I_{\text{position}}(t) = \int_{\mathcal{W}} \rho_I(w,t)\, d\mu_G$ per Definition 24.1.2.

*Interpretation:*
- $\nu_{\text{cap}} \ll 1$: Under-utilized capacity; capital may be suboptimally deployed (too conservative).
- $\nu_{\text{cap}} \approx 1$: Operating at capacity limit; geometry must regulate to prevent overflow.
- $\nu_{\text{cap}} > 1$: **Violation** of the market capacity constraint; indicates ungrounded positions requiring immediate deleveraging.

*Cross-reference:* When $\nu_{\text{cap}} > 1$, the curvature correction (Theorem {prf:ref}`thm-capacity-constrained-ruppeiner-law`) is insufficient. This triggers geometric reflow—the metric $G$ must increase $|G|$ (expand risk volume) to bring $I_{\text{position}}$ back within bounds.
:::

**Node GateCapacitySat: Capacity Saturation Check**

| **#**  | **Name**           | **Component** | **Type**                | **Interpretation**             | **Proxy**                                     | **Cost** |
|--------|--------------------|---------------|-------------------------|--------------------------------|-----------------------------------------------|----------|
| **Gate40** | **CapacitySatCheck** | Risk Manager  | Capacity Monitoring     | Is position within market depth? | $\nu_{\text{cap}} = I_{\text{position}}/C_{\text{mkt}}$ | $O(N_{\text{assets}})$ |

:::{prf:definition} Gate40 Specification
:label: def-gate40-specification

**Predicate:** Position information is bounded by market capacity.
$$
P_{40} : \quad \nu_{\text{cap}}(t) := \frac{I_{\text{position}}(t)}{C_{\text{mkt}}(t)} \le 1 - \epsilon_{\text{buffer}},
$$
where $\epsilon_{\text{buffer}} > 0$ is a safety margin.

**Market interpretation:** The portfolio's representational complexity does not exceed what the market can liquidate.

**Observable metrics:**
- Capacity ratio: $\nu_{\text{cap}}$
- Position entropy: $H(w) = -\sum_i w_i \log w_i$
- Market depth utilization: $\sum_i |w_i| / \text{Depth}_i$
- Illiquidity premium: excess return required for capacity-constrained positions

**Certificate format:**
$$
K_{40}^+ = (\nu_{\text{cap}}, H(w), \text{depth utilization}, \text{timestamp})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{cap-sat}} = \lambda_{40} \cdot \max(0, \nu_{\text{cap}} - (1 - \epsilon_{\text{buffer}}))^2
$$
:::

| $\nu_{\text{cap}}$ | Interpretation | Action |
|-------------------|----------------|--------|
| $\ll 1$ | Under-utilized capacity | Review for suboptimal capital deployment |
| $\approx 0.7$–$0.9$ | Healthy utilization | Normal operations |
| $\approx 0.9$–$1.0$ | Operating at capacity | Risk metric regulates; constrain new positions |
| $> 1$ | **Violation** | Ungrounded position; deleveraging required |

## Implementation: Capacity Monitoring

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class CapacityConfig:
    """Configuration for capacity-constrained risk monitoring."""
    n_assets: int = 100           # Number of assets
    epsilon_buffer: float = 0.1   # Safety buffer (10%)
    lambda_cap: float = 1.0       # Capacity saturation loss weight
    eta_tick: float = 0.01        # Information per tick (nats/tick)


class CapacitySaturationMonitor(nn.Module):
    """
    Monitor capacity saturation ratio ν_cap = I_position / C_mkt.

    Implements Gate40 (CapacitySatCheck) from the Market Sieve.
    """

    def __init__(self, config: CapacityConfig):
        super().__init__()
        self.config = config

        # Learnable market depth model (can be replaced with empirical depth)
        self.depth_estimator = nn.Sequential(
            nn.Linear(config.n_assets, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # Ensure positive depth
        )

    def compute_position_information(
        self,
        weights: torch.Tensor,      # [B, N] portfolio weights
        risk_metric: torch.Tensor,  # [B, N, N] Ruppeiner metric G
    ) -> torch.Tensor:
        """
        Compute information content of position.

        I_position = -∑_i w_i log w_i + (1/2) log det G

        Returns:
            I_position: [B] information content in nats
        """
        B, N = weights.shape
        eps = 1e-8

        # Shannon entropy term: -∑ w log w
        entropy = -torch.sum(
            weights * torch.log(weights.clamp(min=eps)),
            dim=-1
        )  # [B]

        # Geometric correction: (1/2) log det G
        # Use log-det for numerical stability
        log_det_G = torch.logdet(risk_metric + eps * torch.eye(N, device=risk_metric.device))  # [B]
        geo_correction = 0.5 * log_det_G

        return entropy + geo_correction

    def compute_market_capacity(
        self,
        market_depth: torch.Tensor,  # [B, N] depth at each asset
    ) -> torch.Tensor:
        """
        Compute market capacity via area law.

        C_mkt = (1/η_tick) × Depth(∂W)

        Returns:
            C_mkt: [B] market capacity in nats
        """
        # Aggregate depth (sum over assets)
        total_depth = market_depth.sum(dim=-1)  # [B]

        # Area law: capacity = depth / tick_size
        capacity = total_depth / self.config.eta_tick

        return capacity

    def forward(
        self,
        weights: torch.Tensor,       # [B, N] portfolio weights
        risk_metric: torch.Tensor,   # [B, N, N] Ruppeiner metric
        market_depth: torch.Tensor,  # [B, N] market depth per asset
    ) -> Dict[str, torch.Tensor]:
        """
        Compute capacity saturation ratio and loss.

        Returns dict with:
            - nu_cap: Capacity saturation ratio
            - I_position: Position information
            - C_mkt: Market capacity
            - loss: Capacity saturation loss
            - certificate: Gate40 certificate
        """
        # Compute information content
        I_position = self.compute_position_information(weights, risk_metric)

        # Compute market capacity
        C_mkt = self.compute_market_capacity(market_depth)

        # Saturation ratio
        nu_cap = I_position / (C_mkt + 1e-8)

        # Loss: penalize exceeding (1 - ε_buffer)
        threshold = 1.0 - self.config.epsilon_buffer
        violation = torch.relu(nu_cap - threshold)
        loss = self.config.lambda_cap * (violation ** 2).mean()

        # Certificate
        certificate = {
            'nu_cap': nu_cap,
            'I_position': I_position,
            'C_mkt': C_mkt,
            'entropy': -torch.sum(weights * torch.log(weights.clamp(min=1e-8)), dim=-1),
        }

        return {
            'nu_cap': nu_cap,
            'I_position': I_position,
            'C_mkt': C_mkt,
            'loss': loss,
            'certificate': certificate,
            'gate_passed': (nu_cap <= threshold).all(),
        }


def compute_ruppeiner_metric(
    returns: torch.Tensor,  # [T, B, N] return time series
    beta: float = 1.0,      # Inverse temperature (risk aversion)
) -> torch.Tensor:
    """
    Compute Ruppeiner metric from return covariance.

    G_ij = β × Cov(r_i, r_j) = β × E[(r_i - μ_i)(r_j - μ_j)]

    This is the thermodynamic metric on the space of portfolios.
    """
    T, B, N = returns.shape

    # Mean returns
    mu = returns.mean(dim=0)  # [B, N]

    # Covariance (empirical)
    centered = returns - mu.unsqueeze(0)  # [T, B, N]
    cov = torch.einsum('tbi,tbj->bij', centered, centered) / (T - 1)  # [B, N, N]

    # Ruppeiner metric = β × Σ
    G = beta * cov

    return G
```

## Area Law and Holographic Bound

:::{prf:theorem} Area Law for Market Capacity
:label: thm-area-law-market-capacity

The maximum information $I_{\max}$ that can be sustainably maintained in a portfolio region $\Omega \subseteq \mathcal{W}$ is bounded by the "area" of its market interface:
$$
I_{\max}(\Omega) \le \frac{\text{Area}_G(\partial\Omega)}{4\ell_{\text{tick}}^2}
$$
where:
- $\text{Area}_G(\partial\Omega) = \oint_{\partial\Omega} d^{N-1}\sigma_G$ is the $(N-1)$-dimensional boundary area in the Ruppeiner metric,
- $\ell_{\text{tick}}$ is the minimum tick resolution (Planck-like scale for markets).

*Proof sketch.* Apply the Bekenstein-like bound to the market entropy functional. The boundary area measures the information bandwidth of the order book interface; the tick resolution sets the quantum of market information. $\square$
:::

**Market interpretation:** Large portfolios require proportionally large order book depth to be sustainable. This is why institutional investors cannot simply scale up retail strategies—they hit the holographic bound where position information exceeds market interface capacity.

**Cross-reference:** This extends the solvency gates from Section 7.3.

---

