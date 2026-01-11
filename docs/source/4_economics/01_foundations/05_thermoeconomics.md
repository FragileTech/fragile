# Thermoeconomic Foundations

We model pricing as a thermodynamic system with capital as energy, uncertainty as entropy, and risk aversion as temperature.

## Economic Energy, Entropy, and Free Energy

Let $U_t$ be total market wealth (mark-to-market), $S_t$ be informational entropy, and $T_t$ be risk temperature (inverse risk aversion).

**Definition 4.1.1 (Free energy).**
$$
F_t := U_t - T_t S_t.
$$
$F_t$ is the **extractable value** after accounting for uncertainty. In equilibrium, pricing minimizes expected free energy subject to constraints.
This is the standard MaxEnt free-energy form under information constraints {cite}`jaynes1957information,cover2006elements`.

::::{note} Connection to Standard Finance #1: Exponential Utility as Degenerate Free Energy
**The General Law (Fragile Market):**
Agents minimize expected **free energy** $F = U - TS$, where $U$ is expected wealth, $S$ is entropy, and $T$ is risk temperature:
$$
\min_\pi \; \mathbb{E}[F_\pi] = \min_\pi \; \left\{ \mathbb{E}[-R_\pi] + T \cdot H[\pi] \right\}
$$
This is entropy-regularized optimization with $T$ controlling exploration vs. exploitation.

**The Degenerate Limit:**
Fix entropy form to KL divergence. Single-period decision. Scalar wealth. No constraints beyond budget.

**The Special Case (Exponential Utility / CARA):**
$$
U(W) = -\exp(-\alpha W), \quad \text{CE} = -\frac{1}{\alpha} \log \mathbb{E}[\exp(-\alpha W)]
$$
The certainty equivalent is a cumulant generating function; $\alpha = 1/T$ is risk aversion.

**What the generalization offers:**
- **Thermodynamic consistency**: Free energy connects utility to entropy production
- **Dynamic extension**: Multi-period with proper time discounting via $\beta = 1/T$
- **Constraint handling**: The SDF factor $\xi_{t+1}$ incorporates collateral, funding, and default constraints
- **Information-theoretic grounding**: Entropy $S$ measures market uncertainty, not just portfolio randomness
::::

## First Law (Capital Flow)

**Definition 4.2.1 (Capital balance).**
$$
\Delta U = \Delta W + \Delta Q - \Delta D,
$$
where:
- $\Delta W$ is work done by trading (rebalancing gains),
- $\Delta Q$ is external inflow (income, dividends, funding),
- $\Delta D$ is dissipation (transaction costs, slippage, defaults).

## Second Law (Entropy Production)

**Definition 4.3.1 (Entropy production).**
$$
\Delta S \ge \Delta S_{\text{info}} + \Delta S_{\text{friction}},
$$
with strictly positive entropy production when trading costs and information loss are nonzero.

**Interpretation.** In the absence of friction, entropy production can be zero (reversible pricing). With frictions, arbitrage extraction generates entropy and dissipates profit.

## Pricing Kernel as a Thermodynamic Factor

**Definition 4.4.1 (Thermoeconomic SDF).** For payoff $X_{t+1}$,
$$
P_t(X_{t+1}) = \mathbb{E}_t[ m_{t+1} X_{t+1} ],
$$
where the stochastic discount factor $m_{t+1}$ has the form
$$
m_{t+1} = \exp(-\beta_t \Delta F_{t+1}) \cdot \xi_{t+1},
$$
with $\beta_t = 1/T_t$, and $\xi_{t+1}$ captures constraints (collateral, funding, default).

This makes discounting a **free-energy penalty** plus constraint adjustments.
It is consistent with standard SDF-based pricing when $\xi_{t+1} \equiv 1$ {cite}`cochrane2005asset`.

## Ruppeiner Geometry for Markets: The Risk Metric Tensor

The **Ruppeiner metric** measures thermodynamic distance between market states, where "distance" reflects the difficulty of arbitraging between them.

**Definition 4.5.1 (Ruppeiner Metric Tensor).** The market risk metric is the Hessian of entropy:
$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{1}{T} \frac{\partial^2 F}{\partial z^i \partial z^j},
$$
where:
- $z = (z^1, \ldots, z^n)$ are market state coordinates (e.g., log-prices, volatilities, spreads),
- $S$ is the market entropy (uncertainty about future prices),
- $F$ is free energy (risk-adjusted value),
- $T$ is risk temperature (inverse risk aversion).

**Proposition 4.5.2 (Metric Components for Standard Markets).** For a market with log-returns $r_i$ and covariance $\Sigma_{ij}$:
$$
G_{ij} = \frac{1}{T} \Sigma^{-1}_{ij}.
$$
High covariance = low metric distance (easy to arbitrage); low covariance = high metric distance (hard to hedge).

**Definition 4.5.3 (Thermodynamic Distance).** The distance between market states $z$ and $z'$ is:
$$
d_G(z, z') := \int_0^1 \sqrt{G_{ij}(z(\tau)) \dot{z}^i(\tau) \dot{z}^j(\tau)} \, d\tau,
$$
minimized over paths $z(\tau)$ from $z$ to $z'$.

**Market interpretation:** $d_G$ measures the **minimum risk** required to move a portfolio from state $z$ to state $z'$.

::::{note} Connection to Standard Finance #2: Markowitz Mean-Variance as Degenerate Ruppeiner
**The General Law (Fragile Market):**
The Ruppeiner risk metric $G_{ij}(z)$ is the **Hessian of entropy** on market state space:
$$
G_{ij}(z) = -\frac{\partial^2 S}{\partial z^i \partial z^j}
$$
Portfolio risk is measured by geodesic distance: $d_G(z, z') = \int \sqrt{G_{ij} \dot{z}^i \dot{z}^j} \, d\tau$.

**The Degenerate Limit:**
Assume Gaussian returns. Fix the metric to be state-independent ($G_{ij} \to \text{const}$). Restrict to single-period optimization. Ignore path-dependence.

**The Special Case (Markowitz Mean-Variance Optimization):**
$$
\min_w \; w^T \Sigma w \quad \text{s.t.} \quad w^T \mu = \mu_{\text{target}}, \quad w^T \mathbf{1} = 1
$$
The efficient frontier is the set of minimum-variance portfolios for each target return.

**What the generalization offers:**
- **State-dependent risk**: $G_{ij}(z)$ varies across market regimes—covariance is not constant
- **Non-Gaussian returns**: Entropy Hessian captures higher moments and tail dependencies
- **Path-dependent optimization**: Geodesic distance accounts for transaction costs and market impact along the rebalancing path
- **Thermodynamic consistency**: Risk metric derived from entropy ensures no arbitrage in the information-theoretic sense
::::

## Market Phase Transitions: Crystal, Liquid, Gas

Markets exhibit **three thermodynamic phases** with distinct pricing behavior.

**Definition 4.6.1 (Market Phases).**

| Phase | Entropy | Structure | Price Behavior | Examples |
|-------|---------|-----------|----------------|----------|
| **Crystal** | Low | Ordered, predictable | Prices at fundamental value | Government bonds at par, pegged FX |
| **Liquid** | Medium | Structured randomness | Efficient pricing with noise | Normal equity markets, active FX |
| **Gas** | High | Chaotic, unpredictable | Prices disconnected from fundamentals | Flash crashes, speculative bubbles |

**Definition 4.6.2 (Phase Order Parameter).** The market phase is characterized by:
$$
\Psi := \frac{H(K_t)}{\log |\mathcal{K}|} \in [0, 1],
$$
where $H(K_t)$ is the entropy of the regime distribution.
- $\Psi \approx 0$: Crystal phase (one dominant regime).
- $\Psi \approx 0.5$: Liquid phase (moderate uncertainty).
- $\Psi \approx 1$: Gas phase (maximum uncertainty, all regimes equiprobable).

**Theorem 4.6.3 (Phase Transition Detection).** A phase transition occurs at time $t^*$ if:
$$
\left| \frac{d\Psi}{dt} \right|_{t=t^*} > \Psi_{\text{crit}},
$$
where $\Psi_{\text{crit}}$ is a threshold (typically calibrated to VIX spikes or spread blowouts).

**Definition 4.6.4 (Critical Exponents).** Near a phase transition, observables scale as:
$$
\text{Volatility} \sim |T - T_c|^{-\gamma}, \quad \text{Correlation length} \sim |T - T_c|^{-\nu},
$$
where $T_c$ is the critical temperature and $\gamma, \nu$ are critical exponents.

## Scaling Exponents: The Four Market Temperatures

We track **four scaling exponents** that characterize market dynamics.

**Definition 4.7.1 (Market Scaling Exponents).**

| Exponent | Symbol | Meaning | Observable Proxy |
|----------|--------|---------|------------------|
| **Risk Temperature** | $\alpha$ | Curvature of value landscape | $\sqrt{\mathbb{E}[(\nabla V)^2]}$ |
| **Volatility Temperature** | $\beta$ | Plasticity of price dynamics | Realized vol / Implied vol |
| **Liquidity Temperature** | $\gamma$ | Fluidity of capital flows | Bid-ask spread inverse |
| **Leverage Temperature** | $\delta$ | Amplification of positions | Aggregate leverage ratio |

**Operational extraction:** Using Adam-style optimizer statistics,
$$
\alpha \approx \log_{10}\left( \sqrt{\mathbb{E}[v_t^{\text{critic}}]} + \epsilon \right),
$$
where $v_t$ is the second moment estimate of risk gradients.

**Definition 4.7.2 (Temperature Hierarchy).** For stable markets, the scaling exponents must satisfy:
$$
\alpha > \beta > \gamma > \delta,
$$
meaning:
1. Risk perception ($\alpha$) must dominate volatility ($\beta$).
2. Volatility ($\beta$) must dominate liquidity effects ($\gamma$).
3. Liquidity ($\gamma$) must dominate leverage amplification ($\delta$).

**Violation → Instability:**
- $\beta > \alpha$: Volatility outpaces risk perception → Mode C.E (blow-up).
- $\gamma > \beta$: Liquidity dominates volatility → Mode T.D (frozen market).
- $\delta > \gamma$: Leverage dominates liquidity → Mode S.E (leverage spiral).

## Einstein Equations of Finance: Curvature = Risk

We formulate the **Einstein field equations** for market dynamics, where curvature encodes risk.

**Definition 4.8.1 (Market Einstein Tensor).** Define the Einstein tensor:
$$
\mathcal{G}_{ij} := R_{ij} - \frac{1}{2} R \, G_{ij},
$$
where $R_{ij}$ is the Ricci curvature and $R = G^{ij} R_{ij}$ is the scalar curvature.

**Definition 4.8.2 (Risk-Energy Tensor).** The risk-energy tensor is:
$$
\mathcal{T}_{ij} := \frac{\partial \Phi}{\partial z^i} \frac{\partial \Phi}{\partial z^j} - \frac{1}{2} G_{ij} |\nabla \Phi|^2_G + \Lambda G_{ij},
$$
where $\Phi$ is the risk potential and $\Lambda$ is a "cosmological constant" (baseline risk premium).

**Theorem 4.8.3 (Market Einstein Equations).** In equilibrium, curvature and risk satisfy:
$$
\mathcal{G}_{ij} = \kappa \mathcal{T}_{ij},
$$
where $\kappa$ is the coupling constant (market-specific).

**Interpretation:**
- **Risk (mass-energy) curves the market state space.**
- **Curvature determines how portfolios "fall" toward equilibrium.**
- **Geodesics are optimal trading paths.**

**Corollary 4.8.4 (No-Arbitrage as Flatness).** If $\mathcal{T}_{ij} = 0$ (no risk concentration), then $\mathcal{G}_{ij} = 0$ (flat space), and all paths are equivalent—no arbitrage opportunities.

::::{note} Connection to Standard Finance #3: Sharpe Ratio as Degenerate Risk Premium
**The General Law (Fragile Market):**
Risk concentration is encoded in the **Einstein field equations** of the risk manifold:
$$
\mathcal{G}_{ij} = R_{ij} - \frac{1}{2}R\, G_{ij} = \kappa\, \mathcal{T}_{ij}
$$
The risk-energy tensor $\mathcal{T}_{ij}$ sources curvature; equilibrium risk premia compensate for local curvature.

**The Degenerate Limit:**
Flat portfolio space ($\mathcal{G}_{ij} = 0$). Single risk factor. Single-period evaluation. Constant covariance.

**The Special Case (Sharpe Ratio):**
$$
\text{SR} = \frac{\mathbb{E}[R_p] - R_f}{\sigma_p} = \frac{\mu - r_f}{\sqrt{w^T \Sigma w}}
$$
In flat space, the risk premium is linear in volatility; the Sharpe ratio is the slope of the capital market line.

**What the generalization offers:**
- **Curvature = risk concentration**: High $\mathcal{G}_{ij}$ indicates risk clustering requiring higher premia
- **State-dependent risk premia**: Returns vary with market state, not just volatility
- **Arbitrage detection**: $\mathcal{G}_{ij} \neq \kappa \mathcal{T}_{ij}$ signals mispricing (field equations violated)
- **Geodesic arbitrage**: Arbitrage paths are geodesics in flat regions where $\mathcal{G}_{ij} = 0$
::::

## Geodesic Portfolio Flow: Natural Gradient Investing

Optimal portfolio dynamics follow **geodesics** on the risk manifold.

**Definition 4.9.1 (Geodesic Equation for Portfolios).** A portfolio path $w(t)$ is geodesic if:
$$
\frac{d^2 w^i}{dt^2} + \Gamma^i_{jk} \frac{dw^j}{dt} \frac{dw^k}{dt} = 0,
$$
where $\Gamma^i_{jk}$ are Christoffel symbols derived from $G_{ij}$.

**Proposition 4.9.2 (Natural Gradient Update).** The optimal portfolio update is:
$$
\Delta w = -\eta \, G^{-1} \nabla_w \Phi,
$$
where $G^{-1}$ is the inverse metric and $\nabla_w \Phi$ is the risk gradient.

**Market interpretation:** Natural gradient adjusts position sizes based on local risk curvature:
- **High curvature (risky region):** Small position changes.
- **Low curvature (safe region):** Larger position changes allowed.

**Definition 4.9.3 (Covariant Portfolio Dissipation).** The dissipation rate along a portfolio path is:
$$
\mathfrak{D}_{\text{geo}} := \left\langle \nabla_w V, \dot{w} \right\rangle_G = G_{ij} \frac{\partial V}{\partial w^i} \dot{w}^j,
$$
where $V$ is the value function and $\langle \cdot, \cdot \rangle_G$ is the inner product under the Ruppeiner metric.

**Theorem 4.9.4 (Geodesic Optimality).** Among all self-financing paths from $w_0$ to $w_T$, the geodesic minimizes total transaction cost:
$$
\mathcal{C}[w] = \int_0^T \sqrt{G_{ij}(w) \dot{w}^i \dot{w}^j} \, dt.
$$

::::{note} Connection to Standard Finance #4: Natural Gradient as Degenerate Geodesic Flow
**The General Law (Fragile Market):**
Portfolio updates follow **natural gradient descent** on the risk manifold:
$$
\Delta w = -\eta \, G^{-1}(w) \nabla_w \Phi(w)
$$
where $G^{-1}$ is the inverse Ruppeiner metric and $\Phi$ is the risk potential. Geodesics minimize transaction cost.

**The Degenerate Limit:**
Use covariance inverse as metric ($G \to \Sigma^{-1}$). Linear risk potential ($\Phi \to -\mu^T w$). Single-period. Ignore Christoffel corrections.

**The Special Case (Mean-Variance Gradient):**
$$
\Delta w = -\eta \, \Sigma \nabla_w \left( \frac{1}{2} w^T \Sigma w - \lambda \mu^T w \right) = \eta \left( \lambda \Sigma \mu - \Sigma^2 w \right)
$$
This is gradient descent toward the efficient frontier with covariance scaling.

**What the generalization offers:**
- **Path-dependent cost**: Geodesic length $\mathcal{C}[w]$ accounts for transaction costs along the rebalancing path
- **Curvature corrections**: Christoffel symbols $\Gamma^i_{jk}$ encode cross-asset effects invisible in linear models
- **Natural step size**: Metric $G$ determines locally optimal step size—large steps in safe regions, small steps in risky regions
- **Riemannian optimization**: Converges faster than Euclidean gradient descent on curved manifolds
::::

## Landauer Bound for Trading: Information-Theoretic Costs

Trading incurs an **irreducible information-theoretic cost** bounded by Landauer's principle.

**Theorem 4.10.1 (Landauer Bound for Markets).** Any trade that erases $\Delta I$ bits of market information must dissipate at least:
$$
\Delta Q \ge k_B T \ln(2) \cdot \Delta I,
$$
where $k_B T$ is thermal energy (in market context: risk temperature × volatility).

**Market interpretation:** Information processing (price discovery, order matching) has a **minimum energy cost**. This is why:
1. Market making is not free—spread compensates for information processing.
2. High-frequency trading requires proportionally high infrastructure investment.
3. "Free" information is impossible; all price signals cost someone.

**Definition 4.10.2 (Information-Theoretic Spread).** The minimum bid-ask spread is:
$$
s_{\min} = \frac{k_B T \ln(2)}{V_{\text{avg}}} \cdot H(K_t),
$$
where $V_{\text{avg}}$ is average trade volume and $H(K_t)$ is regime entropy.

**Corollary 4.10.3 (Efficient Market Bound).** In an efficient market, the actual spread satisfies:
$$
s_{\text{actual}} \ge s_{\min},
$$
with equality only in the theoretical limit of zero noise and infinite liquidity.

::::{note} Connection to Standard Finance #5: Kyle Lambda as Degenerate Landauer Bound
**The General Law (Fragile Market):**
Market impact is bounded by the **Landauer principle**—information erasure has minimum cost:
$$
\Delta Q \ge k_B T \ln(2) \cdot \Delta I
$$
The bid-ask spread $s_{\min} \propto H(K_t) / V_{\text{avg}}$ scales with regime entropy and inversely with volume.

**The Degenerate Limit:**
Linear price impact. Gaussian information structure. Single informed trader. Continuous trading.

**The Special Case (Kyle Lambda):**
$$
\Delta p = \lambda \cdot Q, \quad \lambda = \frac{\sigma_v}{2\sigma_u}
$$
where $\lambda$ is the Kyle lambda, $\sigma_v$ is fundamental volatility, $\sigma_u$ is noise trader volatility, and $Q$ is order flow. Price impact is linear in order size.

**What the generalization offers:**
- **Information-theoretic grounding**: Spread compensates for entropy reduction (price discovery)
- **Regime-dependent impact**: $H(K_t)$ varies—impact is higher in uncertain regimes
- **Fundamental lower bound**: Landauer bound sets minimum cost; actual costs may exceed this
- **Volume scaling**: Impact decreases with liquidity $V_{\text{avg}}$—markets are more efficient when active
- **Physical consistency**: Connects to thermodynamic principles; no free lunch in information processing
::::

## Log-Sobolev Inequality and Market Concentration

The **Log-Sobolev inequality** connects entropy to concentration—markets with good LSI constants have predictable price distributions.

**Definition 4.11.1 (Market Log-Sobolev Constant).** The LSI constant $\rho_{\text{LSI}}$ satisfies:
$$
\text{Ent}_{\mathfrak{m}}(f^2) \le \frac{2}{\rho_{\text{LSI}}} \int |\nabla f|^2 \, d\mathfrak{m},
$$
for all smooth $f$ with $\int f^2 d\mathfrak{m} = 1$.

**Market interpretation:**
- **Large $\rho_{\text{LSI}}$:** Prices concentrate tightly around equilibrium.
- **Small $\rho_{\text{LSI}}$:** Prices are dispersed; fat tails and extreme events are common.

**Proposition 4.11.2 (LSI and VaR).** The Value-at-Risk at confidence $\alpha$ satisfies:
$$
\text{VaR}_{\alpha} \le \mu + \sigma \sqrt{\frac{2}{\rho_{\text{LSI}}} \ln\left(\frac{1}{1-\alpha}\right)},
$$
where $\mu, \sigma$ are mean and standard deviation.

## Wasserstein Distance and Regime Shifts

Regime shifts are measured by **Wasserstein distance** between price distributions.

**Definition 4.12.1 (Regime Transition Cost).** The cost of transitioning from regime $K$ to regime $K'$ is:
$$
W_2(\mu_K, \mu_{K'}) := \left( \inf_{\pi \in \Pi(\mu_K, \mu_{K'})} \int d(x, y)^2 \, d\pi(x, y) \right)^{1/2},
$$
where $\Pi(\mu_K, \mu_{K'})$ is the set of couplings.

**Proposition 4.12.2 (Regime Transition Warning).** A regime transition is imminent when:
$$
\frac{d}{dt} W_2(\mu_t, \mu_K) < -\epsilon_{\text{trans}},
$$
where $\mu_K$ is the current regime distribution and $\epsilon_{\text{trans}}$ is a threshold.

---

