# Portfolio Transport Geometry: Wasserstein-Fisher-Rao for Rebalancing

:::{admonition} Researcher Bridge: Handling Regime Shifts in Portfolio Rebalancing
:class: info
:name: rb-regime-shift-rebalancing

Standard Bayesian portfolio models fail during regime shifts because they can't handle probability mass appearing or disappearing—what optimal transport calls "unbalanced transport." The **Wasserstein-Fisher-Rao (WFR)** metric allows the portfolio's allocation distribution to both **flow** (continuous rebalancing) and **jump** (discrete regime switches). This provides a unified variational principle for both systematic rebalancing and tactical allocation shifts.

If you've worked with mean-variance optimization, WFR generalizes it to handle non-stationary regimes and tactical overlays in a single framework.
:::

The portfolio bundle $\mathcal{W} = \mathcal{K} \times \mathcal{W}_n$ combines a discrete regime state $K$ (risk-on/risk-off, sector allocation) with continuous asset weights $w$. The WFR metric unifies **continuous rebalancing** (Wasserstein transport) with **discrete regime transitions** (Fisher-Rao reaction).

The key insight is to treat the portfolio not as a fixed allocation $w$, but as a **measure** (allocation distribution) $\rho_s \in \mathcal{M}^+(\mathcal{W})$ evolving on the portfolio space.

(sec-wfr-failure-product-metrics)=
## Motivation: The Failure of Product Metrics

**The Problem with Traditional Portfolio Metrics.**

Standard portfolio optimization assumes a single regime with static covariance. When regimes shift (2008 crisis, COVID crash), the optimizer must either:
1. Wait for enough data to estimate new parameters (too slow), or
2. Make a discontinuous jump to the new allocation (ignoring path costs).

The metric induced by mean-variance optimization:
$$
ds^2 = dw^T \Sigma^{-1} dw
$$
assumes a fixed covariance matrix $\Sigma$. This creates two problems:

1. **Discontinuous Regime Jumps:** When transitioning from "risk-on" ($K_1$) to "risk-off" ($K_2$), the metric provides no principled way to measure the "cost" of the jump versus gradual rotation.

2. **No Mass Conservation Flexibility:** A point allocation either is or isn't at a location. But the portfolio manager's *belief* can be partially in multiple regimes simultaneously (ensemble models, scenario analysis).

**The WFR Solution:**

The Wasserstein-Fisher-Rao metric resolves both issues by lifting dynamics to the space of measures $\mathcal{M}^+(\mathcal{W})$. In this space:
- **Transport (Wasserstein):** Allocation probability flows along continuous weights via the continuity equation.
- **Reaction (Fisher-Rao):** Allocation probability is created/annihilated locally, enabling discrete regime transitions.

The metric determines the optimal rebalancing path by minimizing the total cost: transport cost (transaction costs) plus reaction cost (regime switch costs).

(sec-wfr-rebalancing-action)=
## The WFR Metric (Benamou-Brenier Formulation)

Let $\rho(s, w)$ be a time-varying allocation density on the portfolio space $\mathcal{W}$, evolving in trading time $s$. The WFR distance is defined by the minimal action of a generalized continuity equation.

:::{prf:definition} The WFR Rebalancing Action
:label: def-wfr-rebalancing-action

The squared WFR distance $d^2_{\mathrm{WFR}}(\rho_0, \rho_1)$ between portfolio distributions is the infimum of the energy functional:
$$
\mathcal{E}[\rho, v, r] = \int_0^1 \int_{\mathcal{W}} \left( \underbrace{\|v_s(w)\|_G^2}_{\text{Transaction Cost}} + \underbrace{\lambda^2 |r_s(w)|^2}_{\text{Regime Switch Cost}} \right) d\rho_s(w) \, ds
$$
subject to the **Unbalanced Continuity Equation**:
$$
\partial_s \rho + \nabla \cdot (\rho v) = \rho r
$$

where:
- $v_s(w) \in T_w\mathcal{W}$ is the **rebalancing velocity** (continuous portfolio drift)
- $r_s(w) \in \mathbb{R}$ is the **regime transition rate** (growth/decay of allocation probability)
- $\lambda > 0$ is the **rebalancing granularity** balancing transaction costs and regime switch costs
- $G$ is the Ruppeiner risk metric on portfolio space (Section 4.5)

*Forward reference (Boundary Conditions).* Section 28 specifies how boundary conditions on $\partial\mathcal{W}$ (order book interface) constrain the WFR dynamics: **Trading hours** impose execution constraints; **Overnight** allows unconstrained internal rebalancing.
:::

::::{admonition} Physics Isomorphism: Wasserstein-Fisher-Rao Geometry
:class: note
:name: pi-wfr-geometry-market

**In Physics:** The Wasserstein-Fisher-Rao (WFR) metric on probability measures combines optimal transport (Wasserstein) with information geometry (Fisher-Rao). It is the unique metric allowing both mass transport and creation/annihilation.

**In Markets:** The allocation density $\rho$ evolves under the WFR metric on $\mathcal{P}(\mathcal{W})$:
$$d_{\text{WFR}}^2(\rho_0, \rho_1) = \inf_{\rho, v, r} \int_0^1 \int_{\mathcal{W}} \left( \|v\|_G^2 + \lambda^2 r^2 \right) \rho \, d\mu_G \, dt$$

**Correspondence Table:**

| Optimal Transport | Market (Portfolio Rebalancing) |
|:------------------|:-------------------------------|
| Wasserstein distance $W_2$ | Transaction cost for rebalancing |
| Fisher-Rao distance | Regime switch / tactical shift cost |
| Transport velocity $v$ | Rebalancing rate (shares/day) |
| Reaction rate $r$ | Regime probability change rate |
| Benamou-Brenier formula | Dynamic rebalancing formulation |
| Geodesic interpolation | Optimal rebalancing path |
| Mass conservation | Dollar conservation (within regime) |
| Mass creation/annihilation | Regime allocation shifts |

**Significance:** WFR unifies systematic rebalancing (transport) and tactical allocation shifts (reaction) in a single Riemannian geometry.
::::

:::{prf:remark} Units for Portfolio WFR
:label: rem-units-portfolio-wfr

When $w$ represents **portfolio weights** (dimensionless, summing to 1):
- $[v] = 1/\text{time}$ (rate of weight change)
- $[r] = 1/\text{time}$ (regime transition rate)
- $[\lambda] = 1$ (dimensionless crossover scale)

When $w$ represents **dollar positions**:
- $[v] = \text{dollars}/\text{time}$ (trading rate)
- $[r] = 1/\text{time}$ (regime transition rate)
- $[\lambda] = \text{dollars}$ (minimum efficient trade size)

In both cases, the ratio $\|v\|_G/(\lambda |r|)$ determines whether continuous rebalancing or discrete regime switching dominates.
:::

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Transport velocity $v$ | Continuous rebalancing rate |
| Reaction rate $r$ | Regime probability transition rate |
| Length scale $\lambda$ | Rebalancing granularity / minimum trade size |
| Mass $\rho$ | Allocation probability / belief |
| Teleportation | Instantaneous regime switch |
| Geodesic | Minimum-cost rebalancing path |

(sec-transport-reaction-markets)=
## Transport vs. Reaction Components

The allocation distribution $\rho_s$ evolves on the portfolio space $\mathcal{W}$ via two mechanisms.

**1. Transport (Wasserstein Component):**

The density evolves via the continuity equation $\partial_s\rho + \nabla\cdot(\rho v) = 0$ along the continuous asset weights. The transport cost is $\int \|v\|_G^2\, d\rho$. In the limit $r \to 0$, the dynamics reduce to the standard Wasserstein-2 ($W_2$) optimal transport on the Ruppeiner manifold.

$$
\partial_s \rho + \nabla \cdot (\rho v) = 0
$$

- **Market interpretation:** Gradual rebalancing (dollar-cost averaging, systematic rotation, TWAP execution).
- **Cost:** Transaction costs, market impact proportional to $\|v\|^2$.
- **Regime:** Same investment regime, different weights.
- **Example:** Rebalancing from 60/40 stocks/bonds to 55/45 within a "moderate risk" regime.

**2. Reaction (Fisher-Rao Component):**

The density undergoes local mass creation/annihilation via the source term $\rho r$. This corresponds to discrete regime transitions: allocation probability decreases on Regime A ($r < 0$) and increases on Regime B ($r > 0$). The reaction cost is $\int \lambda^2|r|^2\, d\rho$. In the limit $v \to 0$, the dynamics reduce to the Fisher-Rao metric on the probability simplex $\Delta^{|\mathcal{K}|}$.

$$
\partial_s \rho = \rho r
$$

- **Market interpretation:** Regime switches (risk-on to risk-off, sector rotation, tactical allocation overlays).
- **Cost:** Opportunity cost, execution risk, regime transition costs, tracking error.
- **Regime:** Different investment regime, probability mass redistributed.
- **Example:** Switching from "growth" regime to "defensive" regime during market stress.

**3. The Crossover Scale $\lambda$ (Transaction-Reallocation Tradeoff):**

This parameter defines the characteristic dollar value at which transaction costs exceed reallocation costs:
- If rebalancing distance $< \lambda$: Transport preferred (execute gradual rebalancing)
- If rebalancing distance $> \lambda$: Reaction preferred (discrete regime switch)

:::{prf:definition} Canonical Crossover Scale
:label: def-canonical-crossover-scale

Let $G$ be the Ruppeiner metric on $\mathcal{W}$. The canonical choice for $\lambda$ is the **minimum efficient trade size**:
$$
\lambda := \min_{w \in \mathcal{W}} \text{MES}(w),
$$
where $\text{MES}(w)$ is the minimum efficient trade size at allocation $w$—the smallest trade where transaction costs are dominated by market impact rather than fixed costs.

*Default value.* If MES is unknown, a practical default is:
$$
\lambda_{\text{default}} = \sqrt{\frac{\text{ADV} \times \text{spread}}{\text{impact coeff}}} \approx \$50\text{k}–\$500\text{k}
$$
depending on market liquidity.
:::

:::{prf:proposition} Limiting Regimes for Portfolio Rebalancing
:label: prop-limiting-regimes-portfolio

The WFR metric seamlessly unifies systematic and tactical rebalancing:

1. **Systematic Rebalancing (Flow):** When rebalancing within a regime, $r \approx 0$. The dynamics are dominated by $\nabla \cdot (\rho v)$, and the metric reduces to $W_2$ (Wasserstein-2). This recovers the Riemannian manifold structure of the Ruppeiner geometry.

2. **Tactical Allocation (Jump):** When the rebalancing requires regime change, transport becomes expensive (large tracking error). It becomes cheaper to use the source term $r$:
   - $r < 0$ on the old regime (reducing allocation probability)
   - $r > 0$ on the new regime (increasing allocation probability)
   This recovers the **Fisher-Rao metric** on the discrete regime simplex $\Delta^{|\mathcal{K}|}$.

3. **Mixed Regime (Overlay):** In regime overlaps (e.g., transition periods), both $v$ and $r$ are active. The optimal path smoothly interpolates between systematic and tactical.

*Proof sketch.* The cone-space representation of WFR (lifting $\rho$ to $(\sqrt{\rho}, \sqrt{\rho} \cdot w)$) shows that the WFR geodesic projects to a $W_2$ geodesic when $r = 0$, and to a Fisher-Rao geodesic when $v = 0$. $\square$
:::

::::{note} Connection to Standard Finance #18: Mean-Variance as Degenerate WFR
**The General Law (Fragile Market):**
Allocation distributions evolve on $\mathcal{M}^+(\mathcal{W})$ via **Wasserstein-Fisher-Rao dynamics**:
$$
d^2_{\text{WFR}}(\rho_0, \rho_1) = \inf \int_0^1 \int_{\mathcal{W}} \left( \|v_s\|_G^2 + \lambda^2 |r_s|^2 \right) d\rho_s\, ds
$$
subject to the unbalanced continuity equation $\partial_s \rho + \nabla \cdot (\rho v) = \rho r$.

**The Degenerate Limit:**
Fix regime ($r = 0$). Use covariance-inverse metric ($G \to \Sigma^{-1}$). Restrict to point masses (deterministic allocation).

**The Special Case (Mean-Variance Optimization):**
$$
\min_w \frac{1}{2} w^T \Sigma w - \mu^T w \quad \text{s.t.} \quad \mathbf{1}^T w = 1
$$
This recovers **Markowitz portfolio optimization** in the limit of:
- Single regime ($|\mathcal{K}| = 1$)
- Point allocation ($\rho = \delta_w$)
- Quadratic transaction cost proxy ($\|v\|^2 \approx w^T\Sigma w$)

**What the generalization offers:**
- **Unified transport-reaction**: WFR handles systematic rebalancing (within regimes) and tactical shifts (between regimes) in one framework
- **Allocation geometry**: The metric on $\mathcal{M}^+(\mathcal{W})$ respects both $W_2$ (transaction costs) and Fisher-Rao (regime switch costs)
- **Trade granularity**: $\lambda$ determines when rebalancing beats regime switching (Proposition {prf:ref}`prop-limiting-regimes-portfolio`)
- **Robust optimization**: Distribution over allocations enables ensemble methods and scenario analysis
::::

## WFR Portfolio World Model

**Definition 25.4.1 (WFR Portfolio Dynamics).** The policy outputs a generalized velocity $(v, r)$ to minimize WFR path length to the target allocation (goal portfolio).

```python
class WFRPortfolioModel(nn.Module):
    """
    WFR-based portfolio dynamics model.

    Predicts continuous rebalancing (transport) and
    regime transitions (reaction) in a unified framework.
    """

    def __init__(self, n_assets: int, n_regimes: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = n_assets + n_regimes + 1  # weights + regime_probs + risk_budget

        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Transport velocity (continuous rebalancing)
        self.head_v = nn.Linear(hidden_dim, n_assets)

        # Reaction rate (regime transitions)
        self.head_r = nn.Linear(hidden_dim, n_regimes)

    def forward(self, w: torch.Tensor, regime: torch.Tensor,
                risk_budget: torch.Tensor, dt: float = 0.1):
        """
        Predict next portfolio state via WFR dynamics.

        Returns:
            w_next: Next portfolio weights
            regime_next: Next regime probabilities
            v: Rebalancing velocity
            r: Regime transition rate
        """
        inp = torch.cat([w, regime, risk_budget], dim=-1)
        feat = self.dynamics_net(inp)

        v = self.head_v(feat)  # Continuous rebalancing
        r = self.head_r(feat)  # Regime transition

        # Transport update: w' = w + v * dt
        w_next = w + v * dt
        w_next = F.softmax(w_next, dim=-1)  # Normalize

        # Reaction update: regime' = regime * exp(r * dt)
        regime_next = regime * torch.exp(r * dt)
        regime_next = regime_next / regime_next.sum(dim=-1, keepdim=True)

        return w_next, regime_next, v, r
```

## WFR Stress-Energy Tensor

The WFR dynamics provide the **stress-energy tensor** $T_{ij}^{\text{WFR}}$ that drives curvature in Theorem {prf:ref}`thm-capacity-constrained-ruppeiner-law`.

:::{prf:theorem} WFR Stress-Energy Tensor for Portfolios
:label: thm-wfr-stress-energy-tensor-portfolio

Let the WFR rebalancing action be
$$
\mathcal{S}_{\mathrm{WFR}} = \frac{1}{2}\int_0^T\int_{\mathcal{W}} \rho\left(\|v\|_G^2+\lambda^2 r^2\right)\,d\mu_G\,ds,
$$
with continuity equation $\partial_s\rho+\nabla\!\cdot(\rho v)=\rho r$.

Define
$$
T_{ij}^{\text{WFR}} := -\frac{2}{\sqrt{|G|}}\frac{\delta(\sqrt{|G|}\,\mathcal{L}_{\mathrm{WFR}})}{\delta G^{ij}} \quad\text{(holding }\rho,v,r\text{ fixed).}
$$

Then
$$
T_{ij}^{\text{WFR}} = \rho\,v_i v_j + P_{\text{reb}}\,G_{ij}, \qquad P_{\text{reb}} = \frac{1}{2}\,\rho\left(\|v\|_G^2+\lambda^2 r^2\right),
$$
which is the perfect-fluid form with regime switching contributing an additive pressure term $P_{\text{react}}=\tfrac{1}{2}\lambda^2\rho r^2$.
:::

**Market Implications:**
1. **High rebalancing velocity ($v$):** Portfolio changes fast → $T_{ij}^{\text{WFR}}$ large → curvature increases → transaction costs rise nonlinearly. This is the **market impact** effect derived from first principles.

2. **High regime switching ($r$):** Portfolio jumps frequently → $P_{\text{react}}$ increases → capacity stress increases. This triggers the capacity constraint (Section 24).

## WFR Consistency Diagnostic

Following the diagnostic node convention (Section 7), we define the WFR consistency gate:

:::{prf:definition} WFR Consistency Loss
:label: def-wfr-consistency-loss-portfolio

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:
$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2
$$

This penalizes deviations from the unbalanced continuity equation.
:::

**Node GateWFR: WFR Consistency Check**

| **#**  | **Name**     | **Component**     | **Type**                 | **Interpretation**             | **Proxy**                    | **Cost** |
|--------|--------------|-------------------|--------------------------|--------------------------------|------------------------------|----------|
| **Gate41** | **WFRCheck** | **Portfolio Model** | **Dynamics Consistency** | Transport-Reaction balance? | $\mathcal{L}_{\mathrm{WFR}}$ | $O(BK)$  |

:::{prf:definition} Gate41 Specification
:label: def-gate41-specification

**Predicate:** Portfolio dynamics satisfy the unbalanced continuity equation.
$$
P_{41} : \quad \mathcal{L}_{\mathrm{WFR}} \le \epsilon_{\text{WFR}},
$$
where $\epsilon_{\text{WFR}}$ is the consistency tolerance.

**Market interpretation:** The portfolio model's rebalancing and regime switching predictions are internally consistent.

**Observable metrics:**
- WFR consistency loss $\mathcal{L}_{\mathrm{WFR}}$
- Transport-reaction ratio $\|v\|^2 / (\lambda^2 r^2)$
- Mass conservation error $|\int \rho_{t+1} - \int \rho_t \cdot e^{r \Delta t}|$

**Certificate format:**
$$
K_{41}^+ = (\mathcal{L}_{\mathrm{WFR}}, \|v\|_{\text{avg}}, |r|_{\text{avg}}, \text{mass error})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{Gate41}} = \lambda_{41} \cdot \mathcal{L}_{\mathrm{WFR}}
$$
:::

**Trigger conditions:**
- High $\mathcal{L}_{\text{WFR}}$: Portfolio model's $(v, r)$ predictions violate continuity.
- Remedy: Increase training on regime transitions; check for distribution shift in market conditions.

```python
def compute_wfr_consistency_loss(
    rho_t: torch.Tensor,       # [B, K] allocation belief over regimes at time t
    rho_t1: torch.Tensor,      # [B, K] allocation belief over regimes at time t+1
    v_t: torch.Tensor,         # [B, K, N] rebalancing velocity per regime
    r_t: torch.Tensor,         # [B, K] regime transition rate
    dt: float = 1.0,           # Time step (e.g., 1 day)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute WFR consistency loss for portfolio rebalancing.

    Penalizes violation of unbalanced continuity equation:
    ∂_s ρ + ∇·(ρv) = ρr

    Args:
        rho_t: Allocation probability at time t [B, K]
        rho_t1: Allocation probability at time t+1 [B, K]
        v_t: Rebalancing velocity per regime [B, K, N]
        r_t: Regime transition rate [B, K]
        dt: Time step
        eps: Numerical stability

    Returns:
        WFR consistency loss (scalar)
    """
    sqrt_rho_t = torch.sqrt(rho_t + eps)
    sqrt_rho_t1 = torch.sqrt(rho_t1 + eps)

    # Approximate divergence term (simplified: use velocity magnitude)
    # In full implementation, compute ∇·(ρv) via autodiff
    div_rho_v = torch.zeros_like(rho_t)  # Placeholder

    # Predicted change in sqrt(rho) from continuity equation
    # d/ds sqrt(rho) = (rho*r - div(rho*v)) / (2*sqrt(rho))
    predicted_delta = (dt / (2 * sqrt_rho_t + eps)) * (rho_t * r_t - div_rho_v)

    # Actual change
    actual_delta = sqrt_rho_t1 - sqrt_rho_t

    # L2 loss
    loss = ((actual_delta - predicted_delta) ** 2).mean()

    return loss
```

## Comparison: Traditional vs. WFR Rebalancing

| Feature                     | Traditional (Markowitz)          | WFR (Unbalanced Transport)             |
|-----------------------------|----------------------------------|----------------------------------------|
| **State representation**    | Fixed allocation $w$             | Allocation distribution $\rho(w, K)$   |
| **Regime changes**          | Manual patching / rolling windows| Handled natively via reaction $r$      |
| **Path type**               | "Rebalance then Switch"          | Smooth interpolation                   |
| **Optimization**            | Quadratic program                | Convex (generalized geodesics)         |
| **Theoretical consistency** | Ad-hoc covariance estimation     | Gradient flow of entropy (rigorous)    |
| **Multi-scale**             | Separate models per horizon      | Unified with scale-dependent $\lambda$ |
| **Transaction costs**       | Added post-hoc                   | Integrated via transport cost $\|v\|^2$|

---

