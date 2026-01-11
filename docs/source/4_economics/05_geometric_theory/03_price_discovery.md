# Price Discovery Dynamics: Entropic Drift and Market Steering

:::{admonition} Researcher Bridge: Diffusion-Style Price Discovery with Market Maker Control
:class: info
:name: rb-diffusion-price-discovery

If you know diffusion or score-based generative models, the radial expansion here is the price discovery flow. The market maker's control field is the drift term that steers price evolution toward informed valuations.

Think of price discovery as the market "generating" a price from noise (prior) to signal (quote). The entropic drift is the natural tendency of markets to compress spreads; the control field is how informed traders steer that process.
:::

**Price discovery** is the expansion from maximum uncertainty (prior) to information revelation (quoted price). This section establishes the following unification: by identifying the **market maker** as the source of initial direction selection, we merge Market Microstructure and Information Theory into a single variational operation:
- **Microstructure:** The market maker compresses spreads and improves prices.
- **Information Theory:** The market aggregates private information into public prices.
- Both contribute to the drift term in the price discovery SDE.

(sec-price-discovery-radial-expansion)=
## Hyperbolic Volume and Price Discovery

Consider the price state as a particle in the Poincaré disk $\mathbb{D} = \{z \in \mathbb{C} : |z| < 1\}$. The number of distinguishable price levels (precision) grows exponentially with radius $r$.

:::{prf:definition} Price Manifold Boundary and Interior
:label: def-price-manifold-boundary-interior

- **Interior ($\mathring{\mathcal{Z}}$):** The latent price space where mid-prices evolve. $z=0$ is maximum entropy (uninformed prior).
- **Boundary ($\partial\mathcal{Z}$):** The market interface where quotes are revealed. $|z| \to 1$ is minimum entropy (perfect price revelation).
- **Price Discovery:** The process of moving from interior to boundary.
:::

| Geometric Concept | Market Interpretation |
|-------------------|----------------------|
| Origin $z=0$ | Maximum uncertainty (wide spreads, prior only) |
| Boundary $|z|\to 1$ | Price revelation (tight spreads, informed quote) |
| Entropic drift | Natural spread compression over time |
| Policy control field $u_\pi$ | Market maker steering / order flow pressure |
| Radial coordinate $r$ | Information content / price precision |
| Angular coordinate $\theta$ | Price direction (up/down) |

:::{prf:definition} Information Content of Price
:label: def-information-content-price

The hyperbolic distance from origin represents information content:
$$
I_{\text{price}}(z) := d_{\mathbb{D}}(0, z) = 2 \operatorname{artanh}(|z|).
$$

This measures how much information the market has revealed about the fundamental price.
:::

| $|z|$ | $I_{\text{price}}$ | Market Interpretation |
|-------|-------------------|----------------------|
| $0$ | $0$ | Prior only (no market information) |
| $0.5$ | $1.1$ nat | Moderate price discovery |
| $0.9$ | $2.9$ nat | High price precision |
| $\to 1$ | $\to \infty$ | Perfect price revelation |

## Entropic Spread Compression

**Definition 26.2.1 (Entropic Drift in Markets).** In the absence of order flow, prices experience an **entropic drift** toward revelation:
$$
\dot{r} = \frac{1 - r^2}{2},
$$
which integrates to:
$$
r(\tau) = \tanh(\tau/2).
$$

**Interpretation:** In the absence of order flow, the system evolves toward the boundary at this rate. The entropic drift represents the baseline price discovery rate.

**Definition 26.2.2 (Market Maker Control Field).** The market maker (or informed trader) provides a **control field**:
$$
u_{\text{mm}}(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi}[a],
$$
which breaks rotational symmetry at the origin, selecting a preferred direction for price evolution.

| Control Field | Market Interpretation |
|--------------|----------------------|
| $u_{\text{mm}} = 0$ | Uninformed trading (random walk) |
| $u_{\text{mm}} \neq 0$ | Informed trading (directional pressure) |
| $u_{\text{mm}} \cdot \hat{r} > 0$ | Accelerated price discovery |
| $u_{\text{mm}} \cdot \hat{r} < 0$ | Price discovery inhibition |

## Bid-Ask Separation as Partition Condition

**Axiom 26.3.1 (Bid-Ask Decoupling).** The state decomposition $(K, z_n, z_{\text{tex}})$ maps to:
- **Interior (price process):** Mid-price trajectory $z(\tau)$ evolves on the pricing manifold.
- **Boundary (microstructure):** Bid-ask spread $z_{\text{tex}}$ is sampled at the interface.

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}, \lambda_{\text{jump}}, u_\pi \right] = 0
$$

**Consequence:** Mid-price dynamics are independent of microstructure noise. Spread fluctuations decouple from the fundamental price discovery process.

**Definition 26.3.2 (Microstructure Noise Distribution).** At the market interface:
$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma_{\text{spread}}(z)),
$$
where:
$$
\Sigma_{\text{spread}}(z) = \sigma_{\text{spread}}^2 \cdot G^{-1}(z).
$$

**Scaling:** Near the origin (wide spreads), microstructure noise variance is large. Near the boundary (tight spreads), noise is suppressed by the metric.

## Price Discovery SDE

:::{prf:definition} Price Discovery Stochastic Differential Equation
:label: def-price-discovery-sde

The complete price discovery dynamics are:
$$
dz = \underbrace{\left[ v_{\text{drift}}(z) + u_{\pi}(z) \right]}_{\text{Deterministic}} ds + \underbrace{\sigma(z) \, dW_s}_{\text{Stochastic}}
$$
where:
- $v_{\text{drift}}(z) = \frac{1-|z|^2}{2} \hat{r}$ is the entropic drift (spread compression),
- $u_{\pi}(z)$ is the control field (market maker / order flow),
- $\sigma(z) = \sigma_0 (1-|z|^2)$ is the state-dependent volatility (Poincaré scaling),
- $dW_s$ is standard Brownian motion.

**Market interpretation:** Prices evolve via:
1. Natural spread compression toward tighter quotes (entropic drift)
2. Informed trading pressure from market makers (control field)
3. Noise from uninformed trading (diffusion term)
:::

::::{admonition} Physics Isomorphism: Spontaneous Symmetry Breaking
:class: note
:name: pi-symmetry-breaking-market

**In Physics:** At phase transitions, systems spontaneously break symmetry: a symmetric high-temperature state (e.g., paramagnetic) transitions to an ordered low-temperature state (e.g., ferromagnetic) by selecting a specific direction.

**In Markets:** At the origin ($z=0$), all price directions are equivalent (rotational symmetry). The control field $u_\pi$ breaks this symmetry, selecting a direction for price discovery:

**Correspondence Table:**

| Statistical Physics | Market (Price Discovery) |
|:-------------------|:-------------------------|
| High-temperature phase | Pre-trade (wide spreads) |
| Low-temperature phase | Post-trade (tight spreads) |
| Order parameter | Price direction $\theta$ |
| Symmetry-breaking field | Market maker control $u_\pi$ |
| Goldstone mode | Price drift after revelation |
| Domain wall | Bid-ask spread |

**The pitchfork bifurcation:** At $z=0$, the angular component $\theta$ is undetermined. The control field resolves this degeneracy:
$$
\theta = \arg(u_\pi(0))
$$
::::

::::{note} Connection to Standard Finance #19: Efficient Markets as Degenerate Entropic Drift
**The General Law (Fragile Market):**
Price states evolve via **radial generation** on the Poincaré disk:
$$
dz = \left[ v_{\text{drift}}(z) + u_{\pi}(z) \right] ds + \sigma(z) \, dW_s
$$
The control field $u_\pi$ steers price discovery; without it, prices follow entropic drift.

**The Degenerate Limit:**
Set control field to zero ($u_\pi = 0$). Flatten geometry ($\mathbb{D} \to \mathbb{R}$). Use constant volatility ($\sigma(z) \to \sigma$).

**The Special Case (Efficient Market Hypothesis):**
$$
dp = \mu \, dt + \sigma \, dW_t
$$
This recovers **Geometric Brownian Motion** in the limit of:
- No informed trading ($u_\pi = 0$)
- Flat price space ($\mathbb{D} \to \mathbb{R}$)
- Constant volatility ($\sigma(z) \to \sigma$)

**What the generalization offers:**
- **Geometric structure**: Price space is hyperbolic, not Euclidean—extreme prices are harder to reach
- **Information revelation**: Radial coordinate tracks price discovery progress
- **Market maker influence**: Control field captures informed trading impact
- **Microstructure effects**: Spread compression emerges from entropic drift
::::

## Price Discovery Diagnostic

Following the diagnostic node convention (Section 7), we define the price discovery gate:

:::{prf:definition} Gate42 Specification
:label: def-gate42-specification

**Predicate:** Price discovery has progressed sufficiently.
$$
P_{42} : \quad |z_{\text{final}}| \ge R_{\text{cutoff}},
$$
where $R_{\text{cutoff}}$ is the minimum acceptable price precision threshold.

**Market interpretation:** The market has revealed sufficient price information for trading.

**Observable metrics:**
- Radial coordinate $|z|$ (information content)
- Spread compression rate $\dot{r}$
- Control field magnitude $\|u_\pi\|$
- Time to discovery $\tau_{\text{disc}}$

**Certificate format:**
$$
K_{42}^+ = (|z_{\text{final}}|, I_{\text{price}}, \tau_{\text{disc}}, \text{spread})
$$

**Loss contribution:**
$$
\mathcal{L}_{\text{disc}} = \lambda_{42} \cdot \max(0, R_{\text{cutoff}} - |z_{\text{final}}|)^2
$$
:::

**Node GatePriceDisc: Price Discovery Check**

| **#**  | **Name**           | **Component** | **Type**              | **Interpretation**           | **Proxy**                                     | **Cost** |
|--------|--------------------|---------------|----------------------|------------------------------|-----------------------------------------------|----------|
| **Gate42** | **PriceDiscCheck** | Market Model  | Discovery Validity   | Did price discovery converge? | $\mathbb{I}(|z_{\text{final}}| \ge R_{\text{cutoff}})$ | $O(B)$ |

**Trigger conditions:**
- Low PriceDiscCheck: Price discovery incomplete (wide spreads persist).
- Remedy: Increase trading horizon; check for liquidity constraints.

## Implementation: Price Discovery Module

```python
import torch
import torch.nn as nn
from typing import Tuple


class PriceDiscoveryModule(nn.Module):
    """
    Price discovery dynamics on the Poincaré disk.

    Models price evolution from prior (origin) to revelation (boundary)
    via entropic drift + market maker control field.
    """

    def __init__(
        self,
        dim: int = 2,           # Price space dimension
        sigma_0: float = 0.1,   # Base volatility
        hidden_dim: int = 64,   # Control field network hidden dim
    ):
        super().__init__()
        self.dim = dim
        self.sigma_0 = sigma_0

        # Control field network (market maker / order flow model)
        self.control_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def entropic_drift(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute entropic drift: v_drift = (1 - |z|²)/2 × ẑ

        Args:
            z: Price state [B, D]

        Returns:
            Drift velocity [B, D]
        """
        r = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-6)
        z_hat = z / r  # Unit radial direction
        scale = (1 - r**2) / 2
        return scale * z_hat

    def control_field(self, z: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        """
        Compute control field from market maker / order flow.

        Args:
            z: Price state [B, D]
            context: Optional conditioning (e.g., order flow imbalance)

        Returns:
            Control velocity [B, D]
        """
        # Simple case: network predicts control from state
        u = self.control_net(z)

        # Project to ensure we stay in disk (Poincaré constraint)
        r = torch.norm(z, dim=-1, keepdim=True)
        max_u = (1 - r**2) / 2  # Maximum velocity at this radius
        u_norm = torch.norm(u, dim=-1, keepdim=True).clamp(min=1e-6)
        u = u * torch.clamp(max_u / u_norm, max=1.0)

        return u

    def volatility(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent volatility: σ(z) = σ₀(1 - |z|²)

        Args:
            z: Price state [B, D]

        Returns:
            Volatility [B, 1]
        """
        r = torch.norm(z, dim=-1, keepdim=True)
        return self.sigma_0 * (1 - r**2)

    def step(
        self,
        z: torch.Tensor,
        dt: float = 0.01,
        noise: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Single Euler-Maruyama step of price discovery.

        Args:
            z: Current price state [B, D]
            dt: Time step
            noise: Optional pre-sampled noise [B, D]

        Returns:
            z_next: Next price state [B, D]
            info: Dictionary with intermediate values
        """
        B, D = z.shape

        # Compute drift components
        v_drift = self.entropic_drift(z)
        u_pi = self.control_field(z)
        sigma = self.volatility(z)

        # Sample noise if not provided
        if noise is None:
            noise = torch.randn_like(z)

        # Euler-Maruyama step
        dz = (v_drift + u_pi) * dt + sigma * torch.sqrt(torch.tensor(dt)) * noise
        z_next = z + dz

        # Project back to disk (ensure |z| < 1)
        r_next = torch.norm(z_next, dim=-1, keepdim=True)
        z_next = z_next * torch.clamp(0.999 / r_next, max=1.0)

        # Compute information content
        I_price = 2 * torch.atanh(torch.norm(z_next, dim=-1))

        return z_next, {
            'v_drift': v_drift,
            'u_pi': u_pi,
            'sigma': sigma,
            'I_price': I_price,
        }

    def forward(
        self,
        z_0: torch.Tensor,
        n_steps: int = 100,
        dt: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full price discovery trajectory.

        Args:
            z_0: Initial price state [B, D]
            n_steps: Number of steps
            dt: Time step

        Returns:
            z_final: Final price state [B, D]
            trajectory: Full trajectory [n_steps, B, D]
        """
        trajectory = [z_0]
        z = z_0

        for _ in range(n_steps):
            z, _ = self.step(z, dt)
            trajectory.append(z)

        return z, torch.stack(trajectory, dim=0)
```

---

