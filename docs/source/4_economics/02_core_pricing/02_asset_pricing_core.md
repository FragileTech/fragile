# Asset Pricing Core

We now lay out the classical pricing machinery in the hypostructure language.

## Probability Space and Assets

Let $(\Omega, \mathcal{F}, (\mathcal{F}_t)_{t\ge 0}, \mathbb{P})$ be a filtered probability space.
- Asset $i$ has price process $S_t^i$ and dividend stream $D_t^i$.
- The money market account $B_t$ satisfies $dB_t = r_t B_t dt$ (or $B_{t+1} = (1+r_t) B_t$).

A trading strategy $\theta_t$ is **self-financing** if changes in value come only from asset returns, not external infusion.

## No-Arbitrage and the SDF

**Definition 6.2.1 (No-arbitrage).** There is no self-financing strategy with zero cost and nonnegative payoff that is positive with positive probability.

**Theorem 6.2.2 (SDF existence).** Under standard regularity (locally bounded prices, NFLVR), there exists a strictly positive process $M_t$ such that for all assets {cite}`harrison1979martingales,harrison1981martingales,delbaen1994ftap`:
$$
S_t^i = \mathbb{E}_t[ M_{t+1} (S_{t+1}^i + D_{t+1}^i) ].
$$
$M_t$ is the **stochastic discount factor**.

::::{note} Connection to Standard Finance #6: Fundamental Theorem of Asset Pricing (FTAP) as Degenerate SDF Existence
**The General Law (Fragile Market):**
The SDF $M_t$ emerges from the **market hypostructure** as a fixed-point of the self-consistency equation:
$$
M_t = \mathcal{T}[M_{t+1}, G, \mathcal{K}]
$$
where $\mathcal{T}$ is the permit-constrained transition operator, $G$ is the risk metric, and $\mathcal{K}$ is the regime codebook.

**The Degenerate Limit:**
Assume complete markets. Assume a representative agent. Remove permit constraints (all gates pass). Ignore transaction costs and funding frictions.

**The Special Case (Harrison-Pliska FTAP):**
$$
\text{No arbitrage} \iff \exists M_t > 0 : \mathbb{E}_t[M_{t+1} R_{t+1}^i] = 1 \;\forall i
$$
Equivalently, there exists a risk-neutral measure $\mathbb{Q}$ under which discounted prices are martingales.

**What the generalization offers:**
- **Heterogeneous agents**: SDF reflects aggregation over diverse beliefs and constraints, not a single representative agent
- **Incomplete markets**: Multiple SDFs are possible; the framework selects via minimum entropy or free energy
- **Friction-aware**: Transaction costs, funding spreads, and capacity constraints are built into the permit system
- **Regime-dependent**: SDF varies across market regimes $K \in \mathcal{K}$, capturing structural breaks
::::

## Risk-Neutral Measure

Define the Radon-Nikodym derivative
$$
\frac{d\mathbb{Q}}{d\mathbb{P}} \propto M_T.
$$
Then discounted prices are martingales under $\mathbb{Q}$ {cite}`harrison1981martingales,duffie2001dynamic`:
$$
\frac{S_t^i}{B_t} = \mathbb{E}_t^{\mathbb{Q}}\left[\frac{S_T^i + D_T^i}{B_T}\right].
$$

## Equilibrium Pricing (Consumption-Based)

If a representative agent with utility $U$ consumes $C_t$, then
$$
M_{t+1} = \beta \frac{U'(C_{t+1})}{U'(C_t)},
$$
and the Euler equation implies for any asset return $R_{t+1}^i$:
$$
\mathbb{E}_t[M_{t+1} R_{t+1}^i] = 1.
$$
This is the consumption-based Euler condition {cite}`lucas1978asset,breeden1979intertemporal`.

::::{note} Connection to Standard Finance #7: Lucas Asset Pricing as Degenerate Thermoeconomic Equilibrium
**The General Law (Fragile Market):**
The SDF emerges from **free energy minimization** across heterogeneous agents:
$$
M_{t+1} = \exp(-\beta \Delta F_{t+1}) \cdot \xi_{t+1}
$$
where $\Delta F = \Delta U - T\Delta S$ is the change in free energy and $\xi$ captures constraints.

**The Degenerate Limit:**
Representative agent. Power utility $U(C) = C^{1-\gamma}/(1-\gamma)$. Complete markets. No constraints ($\xi = 1$).

**The Special Case (Lucas 1978 / CCAPM):**
$$
M_{t+1} = \beta \frac{U'(C_{t+1})}{U'(C_t)} = \beta \left(\frac{C_{t+1}}{C_t}\right)^{-\gamma}
$$
The Euler equation $\mathbb{E}_t[M_{t+1} R_{t+1}] = 1$ determines asset prices.

**What the generalization offers:**
- **Heterogeneous agents**: Aggregation over diverse preferences and constraints
- **Information entropy**: $S$ captures uncertainty about future states beyond consumption risk
- **Constraint-aware**: $\xi_{t+1}$ incorporates collateral, leverage, and funding constraints
- **Regime-dependent**: SDF varies with market regime $K_t$
::::

## Factor Structure and Risk Premia

Assume $M_{t+1}$ is affine in factors $F_{t+1}$:
$$
M_{t+1} = a_t - b_t^\top F_{t+1}.
$$
Then expected excess returns satisfy
$$
\mathbb{E}[R_{t+1}^i - R_f] = \beta_i^\top \lambda,
$$
where $\beta_i$ is asset exposure and $\lambda$ is factor price of risk.
Empirical factor structures are documented in {cite}`fama1993common,hansen1991implications,cochrane2005asset`.

::::{note} Connection to Standard Finance #8: APT/Fama-French as Degenerate Factor SDF
**The General Law (Fragile Market):**
The SDF is a **nonlinear function** of regime state and factors:
$$
M_{t+1} = f(K_{t+1}, F_{t+1}; \theta)
$$
where $K$ is the regime codebook and $F$ are observable factors. The function $f$ is learned from market data.

**The Degenerate Limit:**
Linear SDF ($M = a - b^T F$). Constant factor loadings. Time-invariant risk premia. No regime dependence.

**The Special Case (APT / Fama-French):**
$$
\mathbb{E}[R_i - R_f] = \beta_{i,\text{MKT}} \lambda_{\text{MKT}} + \beta_{i,\text{SMB}} \lambda_{\text{SMB}} + \beta_{i,\text{HML}} \lambda_{\text{HML}} + \ldots
$$
Linear factor model with constant betas and risk premia $\lambda$.

**What the generalization offers:**
- **Nonlinear factor dependence**: $f(K, F)$ captures interactions and nonlinearities
- **Regime-dependent loadings**: $\beta_i(K)$ varies across market regimes
- **Time-varying risk premia**: $\lambda(K_t)$ adapts to market conditions
- **Latent factors**: Regime codebook $K$ captures factors not directly observable
::::

## Term Structure

Zero-coupon bond price for maturity $T$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u du\right)\right].
$$
HJM and affine models fit into this SDF framework with $r_t$ and $M_t$ jointly specified {cite}`heath1992bond,vasicek1977equilibrium,cox1985theory,duffie2001dynamic`.

::::{note} Connection to Standard Finance #9: Vasicek/CIR as Degenerate Term Structure Field
**The General Law (Fragile Market):**
The yield curve is a **field** on the regime manifold:
$$
y(t, T; K) = -\frac{1}{T-t} \log P(t, T; K)
$$
where $P(t,T;K)$ is the bond price conditional on regime $K$. The field satisfies consistency conditions from the SDF.

**The Degenerate Limit:**
Single regime. Affine short rate dynamics. Time-homogeneous coefficients. No jumps.

**The Special Case (Vasicek / CIR):**
$$
dr_t = \kappa(\theta - r_t) dt + \sigma r_t^\beta dW_t
$$
with $\beta = 0$ (Vasicek) or $\beta = 1/2$ (CIR). Bond prices are exponential-affine: $P(t,T) = A(T-t) e^{-B(T-t) r_t}$.

**What the generalization offers:**
- **Regime-dependent dynamics**: $\kappa, \theta, \sigma$ vary with market regime $K$
- **Jump processes**: Rate can jump at regime transitions
- **Multi-factor extension**: Yield curve field captures level, slope, curvature as regime-dependent factors
- **Consistency with SDF**: Term structure automatically satisfies no-arbitrage via the market hypostructure
::::

## Incomplete Markets and Bounds

When markets are incomplete, SDFs are not unique. Let $\mathcal{M}$ be the admissible SDF set. Then:
$$
\inf_{M \in \mathcal{M}} \mathbb{E}_t[M X] \le P_t(X) \le \sup_{M \in \mathcal{M}} \mathbb{E}_t[M X].
$$
A canonical choice is the **minimal entropy martingale measure**, consistent with the free-energy principle {cite}`frittelli2000entropy`.

## Transaction Costs and Funding Frictions

With proportional costs $\kappa$ and funding spread $s_t$, the no-arbitrage price interval for $X$ widens:
$$
P_t^{\text{bid}}(X) \le P_t(X) \le P_t^{\text{ask}}(X),
$$
with bounds computed from super- and sub-hedging costs. These bounds are enforced by the **Liquidity and Funding Barriers** in the Sieve.

---

