# Representation and Information Constraints

## Macro Register and Residual Channels

The market maintains a discrete macro register $K_t$ (regime, liquidity state, policy state). A valid $K_t$ must satisfy:
- **Capacity:** $H(K_t) \le \log |\mathcal{K}|$.
- **Grounding:** $I(B_t; K_t) > 0$.
- **Closure:** future price dynamics conditional on $K_t$ are stable across regimes.

## Filtering and Belief Update

Let $q(K_t \mid B_{\le t})$ be the market belief over regimes. A consistent update requires:
$$
D_{KL}(q_{t+1} \Vert q_t) \le I(B_{t+1}; K_{t+1}),
$$
so that belief updates are supported by boundary information.

## Information Cost and Pricing

Define an information cost $\mathcal{I}_t := D_{KL}(q_t \Vert p_0)$ relative to a prior $p_0$ {cite}`kullback1951information,cover2006elements`. Pricing that ignores $\mathcal{I}_t$ violates the thermoeconomic free-energy principle and will be rejected by the Sieve (BoundaryCheck + AlignCheck).

## State-Space Metric and Trust Region

Let $z_t$ denote the continuous part of the market state (inventory, funding spreads, liquidity coordinates). Define the state-space Fisher metric
$$
G_t := \mathbb{E}_t[\nabla_{z} \log p(B_{t+1}\mid z_t) \nabla_{z} \log p(B_{t+1}\mid z_t)^\top].
$$
Updates must satisfy a trust-region constraint
$$
d_{G_t}(z_{t+1}, z_t) \le v_{\max},
$$
ensuring the market does not move faster than its own information capacity (Node 7 and Node 12).

## Coupling Window

For a window length $W$, define the boundary coupling condition
$$
0 < I(B_{t:t+W}; K_t) < \log|\mathcal{K}|.
$$
The lower bound prevents starvation (Node 15), the upper bound prevents overload (Node 14).

---

