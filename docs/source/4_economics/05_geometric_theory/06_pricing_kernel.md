# Value Field and Pricing Kernel: Discounted Cash Flow as Screened Poisson

:::{admonition} Researcher Bridge: Value as a Smooth Field (PINN for Pricing)
:class: info
:name: rb-value-pinn-pricing

If you know physics-informed neural networks (PINNs), the pricing kernel is exactly this: learning a value function that satisfies the Helmholtz PDE. The screened Poisson equation $(-\Delta + \kappa^2)V = \rho$ is the steady-state of the heat equation with decay—exactly what discounted cash flow represents.

The discount rate $\gamma$ becomes the screening mass $\kappa = -\ln(\gamma)$; the investment horizon is $\ell = 1/\kappa$. Higher discount rates (short horizons) mean faster decay of future cash flows.
:::

The **pricing kernel** is the Green's function of a screened Poisson equation, and the **discount rate** is the screening mass. This section provides a rigorous PDE formulation for asset valuation that:
1. **Unifies DCF with field theory:** Cash flows are sources; NPV is the potential field
2. **Characterizes investment horizon:** Screening length $\ell = 1/\kappa$ determines the effective valuation range
3. **Enables conformal coupling:** Value curvature feeds back into the risk metric

(sec-reward-as-cash-flow)=
## Reward as Cash Flow

**Definition 29.1.1 (Cash Flow as Source Term).** The cash flow stream (dividends, coupons) acts as a scalar source:
$$
\sigma_{\text{cf}}(t, w) = \sum_{t' < t} \text{CF}_{t'} \cdot \delta(t - t') \cdot \delta(w - w_{t'}),
$$
where $\text{CF}_t$ is the cash flow at time $t$.

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Reward flux $J_r$ | Cash flow stream |
| Boundary charge $\sigma_r$ | Dividend/coupon payments |
| Potential $V(z)$ | Net present value $\text{NPV}(w)$ |
| Screening mass $\kappa$ | Discount rate |

## Pricing Kernel as Screened Poisson Solver

**Theorem 29.2.1 (DCF as Helmholtz Equation).** The net present value $V(w)$ satisfies the **screened Poisson equation**:
$$
\boxed{-\Delta_G V(w) + \kappa^2 V(w) = \rho_{\text{cf}}(w)}
$$
where:
- $\Delta_G$ is the Laplace-Beltrami operator on the risk manifold,
- $\kappa = -\ln(\gamma)/\Delta t$ is the screening mass (discount rate),
- $\rho_{\text{cf}}$ is the cash flow density.

**Proof sketch:** The Bellman equation $V(w) = \mathbb{E}[\text{CF} + \gamma V(w')]$ approaches the Helmholtz PDE in the continuous limit. $\square$

## Discount as Screening Length

**Corollary 29.3.1 (Investment Horizon as Screening Length).**
$$
\ell_{\text{horizon}} = \frac{1}{\kappa} = \frac{\Delta t}{-\ln\gamma}.
$$

| Discount $\gamma$ | Screening Mass $\kappa$ | Horizon $\ell$ | Interpretation |
|-------------------|-------------------------|----------------|----------------|
| $\gamma \to 1$ | $\kappa \to 0$ | $\ell \to \infty$ | Long-term investor |
| $\gamma = 0.99$ | $\kappa \approx 0.01$ | $\ell \approx 100$ | Standard DCF |
| $\gamma = 0.9$ | $\kappa \approx 0.1$ | $\ell \approx 10$ | Short-term trader |
| $\gamma \to 0$ | $\kappa \to \infty$ | $\ell \to 0$ | Myopic (day trader) |

## Value-Risk Conformal Coupling

**Definition 29.4.1 (Value-Metric Feedback).** High-value-curvature regions induce metric distortion:
$$
\tilde{G}_{ij}(w) = \Omega^2(w) \cdot G_{ij}(w),
$$
where:
$$
\Omega(w) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(w)\|_{\text{op}}.
$$

**Operational effect:**
- **Flat value landscape:** Default risk metric applies.
- **High curvature (decision boundary):** Metric expands, reducing position velocity.
- **Saddle regions:** Moderate metric expansion.

## Pricing Kernel Implementation

```python
class PricingKernel(nn.Module):
    """
    Pricing kernel as Helmholtz equation solver.

    Maps cash flow sources to net present value via screened Poisson.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 256, gamma: float = 0.99):
        super().__init__()
        self.kappa = -math.log(gamma)  # Screening mass

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Compute NPV at portfolio position w."""
        return self.net(w)

    def helmholtz_loss(self, w: torch.Tensor, w_next: torch.Tensor,
                        cf: torch.Tensor, gamma: float) -> torch.Tensor:
        """
        Enforce Bellman/Helmholtz consistency.

        V(w) = cf + gamma * V(w')
        """
        V = self(w)
        V_next = self(w_next).detach()
        td_error = V - (cf + gamma * V_next)
        return td_error.pow(2).mean()
```

::::{admonition} Physics Isomorphism: Helmholtz/Yukawa Equation
:class: note
:name: pi-helmholtz-yukawa-market

**In Physics:** The screened Poisson (Helmholtz) equation describes electrostatic potentials with screening (Debye-Hückel) or meson exchange (Yukawa):
$$(-\nabla^2 + \kappa^2)\phi = \rho$$

**In Markets:** The pricing kernel satisfies the same equation:
$$(-\Delta_G + \kappa^2)V = \rho_{\text{cf}}$$

**Correspondence Table:**

| Field Theory | Market (Pricing Kernel) |
|:-------------|:-----------------------|
| Potential $\phi$ | Net present value $V$ |
| Charge density $\rho$ | Cash flow density $\rho_{\text{cf}}$ |
| Screening mass $\kappa$ | Discount rate $\kappa = -\ln(\gamma)$ |
| Screening length $\ell = 1/\kappa$ | Investment horizon |
| Green's function $G(x, x')$ | Pricing kernel $K(w, w')$ |
| Yukawa decay $e^{-\kappa r}/r$ | Discounted cash flow |
| Debye screening | Time value of money |

**The Green's function interpretation:**
$$V(w) = \int K(w, w') \rho_{\text{cf}}(w') dw'$$
where $K(w, w') = e^{-\kappa d_G(w, w')} / d_G(w, w')$ is the Yukawa-like propagator on the Ruppeiner manifold.
::::

::::{note} Connection to Standard Finance #22: DCF as Degenerate Helmholtz
**The General Law (Fragile Market):**
Net present value satisfies the **Helmholtz equation** on the risk manifold:
$$
-\Delta_G V(w) + \kappa^2 V(w) = \rho_{\text{cf}}(w)
$$
where $\kappa = -\ln(\gamma)$ is the screening mass and $\rho_{\text{cf}}$ is the cash flow density.

**The Degenerate Limit:**
Flat geometry ($G \to I$, $\Delta_G \to \nabla^2$). Discrete cash flows. Constant discount rate.

**The Special Case (Discounted Cash Flow):**
$$
V = \sum_{t=1}^{T} \frac{\text{CF}_t}{(1+r)^t}
$$
This recovers **standard DCF valuation** in the limit of:
- Flat portfolio space ($G \to I$)
- Discrete time and cash flows
- Constant discount rate $r$

**What the generalization offers:**
- **Geometric structure**: Cash flows are "charges" on the risk manifold; NPV is the potential field
- **Spatial screening**: Value influence decays with geometric (not just temporal) distance
- **Conformal coupling**: High value curvature distorts the risk metric
- **PINN training**: Neural pricing kernel learns Helmholtz-consistent valuations
::::

## HJB-Helmholtz Correspondence

:::{prf:theorem} HJB-Helmholtz Correspondence
:label: thm-hjb-helmholtz-correspondence

In the continuous-time limit with ergodic dynamics, the Hamilton-Jacobi-Bellman equation
$$
0 = \max_a \left[ r(w, a) + \mathcal{L}_a V(w) - \kappa V(w) \right]
$$
reduces to the Helmholtz equation (Theorem 29.2.1) when the policy is fixed:
$$
-\Delta_G V + \kappa^2 V = \rho_{\text{cf}}
$$

*Proof sketch.* Under the Riemannian measure induced by $G$, the generator $\mathcal{L}$ of the diffusion process reduces to the Laplace-Beltrami operator $\Delta_G$ plus lower-order drift terms. The discount term $-\kappa V$ becomes the screening mass. $\square$
:::

## Value Field Diagnostics

Following the diagnostic node convention (Section 7), we define the Helmholtz consistency gate:

:::{prf:definition} Gate45 Specification
:label: def-gate45-specification

**Predicate:** Value function satisfies the Helmholtz equation.
$$
P_{45} : \quad \|-\Delta_G V + \kappa^2 V - \rho_{\text{cf}}\| \le \epsilon_{\text{helm}},
$$
where $\epsilon_{\text{helm}}$ is the PDE residual tolerance.

**Market interpretation:** The pricing kernel correctly discounts future cash flows.

**Observable metrics:**
- Helmholtz residual $\|-\Delta_G V + \kappa^2 V - \rho_{\text{cf}}\|$
- TD error (discrete approximation)
- Screening length consistency
- Value gradient magnitude $\|\nabla V\|$

**Certificate format:**
$$
K_{45}^+ = (\text{residual}, \text{TD error}, \ell_{\text{effective}}, \|\nabla V\|)
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{helm}} = \lambda_{45} \cdot \|-\Delta_G V + \kappa^2 V - \rho_{\text{cf}}\|^2
$$
:::

**Node GateHelmholtz: Helmholtz Residual Check**

| **#**  | **Name**           | **Component**    | **Type**           | **Interpretation**          | **Proxy**                                           | **Cost** |
|--------|--------------------|-----------------|--------------------|-----------------------------|----------------------------------------------------|----------|
| **Gate45** | **HelmholtzCheck** | Pricing Kernel  | PDE Consistency    | Is DCF equation satisfied?  | $\|-\Delta_G V + \kappa^2 V - \rho_{\text{cf}}\|$ | $O(BD)$  |

**Trigger conditions:**
- High Helmholtz residual: Pricing kernel does not satisfy the DCF PDE.
- Remedy: Increase PINN training iterations; check discount rate consistency.

---

