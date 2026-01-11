# Risk Measures and Thermodynamic Duality

## Coherent Risk Measures

A risk measure $\rho$ is coherent if {cite}`artzner1999coherent`:
- monotone,
- subadditive,
- positive homogeneous,
- translation invariant.

::::{note} Connection to Standard Finance #10: CVaR/Expected Shortfall as Degenerate Coherent Risk
**The General Law (Fragile Market):**
Risk is measured by a **robust expectation** over an uncertainty set $\mathcal{Q}$ of probability measures:
$$
\rho(X) = \sup_{Q \in \mathcal{Q}} \left\{ \mathbb{E}_Q[-X] - \alpha(Q) \right\}
$$
where $\alpha(Q)$ is a penalty function encoding model uncertainty. This is the **dual representation** of coherent risk measures.

**The Degenerate Limit:**
Fix the uncertainty set to $\mathcal{Q}_\alpha = \{Q : Q \ll P, \frac{dQ}{dP} \le 1/\alpha\}$. Use zero penalty $\alpha(Q) = 0$ inside the set. Assume single-period, scalar loss.

**The Special Case (Conditional Value-at-Risk / Expected Shortfall):**
$$
\text{CVaR}_\alpha(X) = \mathbb{E}[X \mid X \le \text{VaR}_\alpha(X)] = \frac{1}{\alpha} \int_0^\alpha \text{VaR}_u(X) \, du
$$
This is the average loss in the worst $\alpha$-fraction of scenarios.

**What the generalization offers:**
- **Model uncertainty**: The penalty $\alpha(Q)$ quantifies confidence in each scenario model
- **Dynamic consistency**: Fragile risk measures extend to multi-period with proper conditioning
- **Thermodynamic grounding**: Risk = free energy = expected loss + entropy penalty
- **Coherence by construction**: The dual representation guarantees all four coherence axioms
::::

## Entropic Risk and Free Energy

The entropic risk of payoff $X$ at risk aversion $\alpha$ is
$$
\rho_{\alpha}(X) = \frac{1}{\alpha} \log \mathbb{E}[e^{\alpha X}].
$$
This is a **free-energy functional** and corresponds to exponential utility {cite}`follmer2011stochastic`.

## Dual Form (Relative Entropy)

Entropic risk has dual form
$$
\rho_{\alpha}(X) = \sup_{Q \ll P} \left( \mathbb{E}_Q[X] - \frac{1}{\alpha} D_{KL}(Q \Vert P) \right),
$$
which is the thermoeconomic principle: value equals expected payoff minus information cost {cite}`kullback1951information,follmer2011stochastic`.

---

