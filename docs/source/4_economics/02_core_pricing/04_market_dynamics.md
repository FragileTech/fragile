# Market Dynamics and Control

The market is a collection of bounded-rational controllers that trade under constraints and costs.

## Entropy-Regularized Portfolio Choice

Agent $j$ solves:
$$
\min_{\pi_j} \; \mathbb{E}\left[ C_j(\pi_j) + \alpha_j D_{KL}(\pi_j \Vert \pi_j^0) \right],
$$
where $C_j$ is expected cost and $\pi_j^0$ is a prior allocation. The solution is exponential-family:
$$
\pi_j(a) \propto \pi_j^0(a) \exp(-C_j(a)/\alpha_j).
$$
This is the **thermodynamic logit** and links risk aversion to temperature {cite}`todorov2009efficient,kappen2005path`.

## Market Clearing as Fixed Point

Let $D_t(p)$ be aggregate demand and $S_t(p)$ aggregate supply. Clearing requires:
$$
D_t(p_t) = S_t(p_t).
$$
Under permits, the clearing price $p_t$ is a fixed point of the trading dynamics.

## Stability via Lyapunov Potential

Define a Lyapunov functional $L$ (aggregate risk plus costs). Stability requires:
$$
\Delta L \le 0
$$
in normal regimes. Persistent positive drift violates the **Stiffness and Oscillation permits**.

---

