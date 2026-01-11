# Asset Class Pricing (Comprehensive)

This section provides complete pricing specifications for all 12 major asset classes. Each class includes:
- SDF-based pricing derivation
- Complete permit checklist (relevant gates and barriers)
- Asset-specific failure mode mapping
- Risk geometry (curvature, geodesics)
- Stress test scenario

---

## Risk-Free and Government Bonds

:::{prf:definition} Government Bond Pricing Framework
:label: def-govbond-pricing

**Fundamental equation.** For a zero-coupon bond paying 1 at maturity $T$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u \, du\right)\right] = \mathbb{E}_t^{\mathbb{P}}\left[M_T / M_t\right],
$$
where $r_u$ is the short rate and $M_t$ is the stochastic discount factor.

**Affine term structure.** Under affine models (Vasicek, CIR, multi-factor):
$$
P(t,T) = \exp\left(A(t,T) - B(t,T) \cdot X_t\right),
$$
where $X_t$ is the state vector (short rate, slope, curvature factors).

**Duration and convexity geometry.** Define the risk metric:
$$
g^{\text{bond}}_{ij} = \frac{\partial^2 \log P}{\partial X_i \partial X_j} = B_i B_j - \frac{\partial B_i}{\partial X_j}.
$$
This is the **Fisher information metric** on the bond manifold.

**Geodesic portfolio path.** Duration-neutral rebalancing follows geodesics:
$$
\ddot{w}^k + \Gamma^k_{ij} \dot{w}^i \dot{w}^j = 0,
$$
where $w$ is the portfolio weight vector and $\Gamma$ is the Christoffel symbol from $g^{\text{bond}}$.
:::

**Permit checklist:**
- Node 1 (Solvency): Government credit risk (for non-AAA sovereigns)
- Node 5 (Stationarity): Interest rate regime stability
- Node 6 (Capacity): Treasury market depth
- Node 8 (Connectivity): Dealer network functionality
- Node 11 (Representation): Yield curve model adequacy
- BarrierInput: Central bank data feed integrity
- BarrierLiq: On-the-run vs. off-the-run liquidity

**Failure mode mapping:**
- T.D (Frozen Market): Treasury market stress (March 2020)
- B.E (External Shock): Fed policy surprise
- S.D (Flat Vol): Yield curve control regimes

**Stress test scenario:** Fed surprise 100bp hike
- BarrierOmin: Check for gap risk in long-duration positions
- BarrierBode: Duration hedge increases convexity exposure
- Expected response: Price interval widens; switch to rolling auctions

::::{note} Connection to Standard Finance #11: Affine Term Structure as Degenerate Bond Field
**The General Law (Fragile Market):**
Bond prices form a **field** on the state-regime manifold:
$$
P(t, T; X, K) = \mathbb{E}_t^{\mathbb{Q}(K)}\left[\exp\left(-\int_t^T r_u \, du\right) \mid X_t, K_t\right]
$$
The risk metric $g^{\text{bond}}_{ij} = \partial^2 \log P / \partial X_i \partial X_j$ is the Fisher information on the bond manifold.

**The Degenerate Limit:**
Single regime ($|K| = 1$). Affine state dynamics. Time-homogeneous coefficients. No jumps at regime transitions.

**The Special Case (Affine Term Structure Models):**
$$
P(t,T) = \exp(A(\tau) - B(\tau) \cdot X_t), \quad \tau = T - t
$$
where $(A, B)$ solve Riccati ODEs. This includes Vasicek, CIR, and multi-factor models.

**What the generalization offers:**
- **Regime-dependent term structure**: Yield curve shape varies with market regime $K$
- **Jump risk pricing**: Regime transitions induce discontinuous yield moves
- **Geometric hedging**: Duration-neutral rebalancing follows geodesics on bond manifold
- **Non-affine extensions**: Nonlinear $P(X, K)$ captures empirical yield curve features
::::

---

## Inflation-Linked Bonds (TIPS, Linkers)

:::{prf:definition} Inflation-Linked Bond Pricing
:label: def-tips-pricing

**Real vs. nominal decomposition.** Let $I_t$ be the price index. The real bond price:
$$
P^{\text{real}}(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[\exp\left(-\int_t^T r_u^{\text{real}} \, du\right)\right],
$$
where $r^{\text{real}}_t = r_t - \pi_t$ with $\pi_t$ the instantaneous inflation rate.

**Breakeven inflation.** The breakeven rate $\text{BE}(t,T)$ satisfies:
$$
P^{\text{nom}}(t,T) = P^{\text{real}}(t,T) \cdot \exp\left(-\text{BE}(t,T)(T-t)\right).
$$

**Inflation risk premium.** The difference between breakeven and expected inflation:
$$
\text{IRP}(t,T) = \text{BE}(t,T) - \mathbb{E}_t[\bar{\pi}_{t,T}] = -\frac{\text{Cov}_t(M_T, I_T)}{M_t P^{\text{nom}}(t,T)}.
$$

**Risk geometry.** The inflation-linked bond manifold has metric:
$$
g^{\text{TIPS}}_{ij} = g^{\text{real}}_{ij} + g^{\text{inflation}}_{ij} + 2 \cdot \text{cross-term}_{ij},
$$
capturing real rate risk, inflation risk, and their correlation.
:::

**Permit checklist:**
- Node 1 (Solvency): Sovereign real credit risk
- Node 5 (Stationarity): Inflation regime stability
- Node 11 (Representation): Inflation model adequacy (seasonal adjustment)
- BarrierInput: CPI data integrity and publication schedule
- BarrierCausal: Indexation lag (3-month typical)
- BarrierRef: Reference index definition changes

**Failure mode mapping:**
- D.C (Fundamental Uncertainty): Inflation regime change (1970s, 2021-22)
- S.C (Parameter Drift): Correlation breakdown between breakeven and realized
- B.E (External Shock): Commodity price shock affecting CPI

**Stress test scenario:** CPI methodology change
- BarrierRef triggers: Index definition no longer comparable
- BarrierCausal: Historical comparisons invalid
- Expected response: Widen price bounds; mark as model uncertainty

---

## Equities

:::{prf:definition} Equity Pricing Framework
:label: def-equity-pricing

**Dividend discount model.** Stock price equals discounted expected dividends:
$$
S_t = \mathbb{E}_t\left[\sum_{u>t} M_u D_u\right] = \mathbb{E}_t\left[\int_t^\infty M_u D_u \, du\right].
$$

**Risk premium decomposition.** The equity risk premium:
$$
\mathbb{E}_t[R_{t+1}] - r_t = -\text{Cov}_t\left(\frac{M_{t+1}}{M_t}, R_{t+1}\right) = \gamma_t \cdot \text{Cov}_t(R_{t+1}, \Delta c_{t+1}),
$$
where $\gamma_t$ is risk aversion and $c$ is consumption (CCAPM form).

**Factor model embedding.** In factor space:
$$
\mathbb{E}_t[R_i] - r_t = \sum_k \beta_{ik} \lambda_k,
$$
where $\beta_{ik}$ is exposure to factor $k$ and $\lambda_k$ is the factor risk premium.

**Risk geometry (Sharpe manifold).** Define the metric on equity space:
$$
g^{\text{eq}}_{ij} = \frac{1}{\sigma_i \sigma_j} \left(\rho_{ij} - \frac{\mu_i - r}{\sigma_i} \cdot \frac{\mu_j - r}{\sigma_j} \cdot \frac{1}{\text{SR}^2_{\text{max}}}\right),
$$
where SR$_{\text{max}}$ is the maximum Sharpe ratio. Geodesics are **efficient portfolio paths**.

**Natural gradient update.** Portfolio optimization via:
$$
w_{t+1} = w_t - \eta \cdot (g^{\text{eq}})^{-1} \nabla_w \mathcal{L},
$$
where $\mathcal{L}$ is the risk-adjusted loss.
:::

**Permit checklist:**
- Node 1 (Solvency): Corporate credit/bankruptcy risk
- Node 2 (Turnover): Trading volume adequacy
- Node 3 (Leverage): Margin requirements, short interest
- Node 5 (Stationarity): Factor regime stability
- Node 6 (Capacity): Market cap, float
- Node 7 (Stiffness): Mean reversion in valuations
- Node 8 (Connectivity): Exchange connectivity, dark pools
- Node 11 (Representation): Factor model adequacy
- Node 12 (Oscillation): Momentum vs. mean-reversion balance
- BarrierOmin: Flash crash protection
- BarrierSat: Position limits
- BarrierFreq: HFT monitoring

**Failure mode mapping:**
- D.E (Boom-Bust): Equity bubbles (dot-com, meme stocks)
- T.E (Flash Crash): May 2010, August 2015
- S.E (Supercritical Leverage): Margin debt spikes
- C.D (Too-Big-to-Fail): Index concentration

**Stress test scenario:** Factor rotation (growth → value)
- Node 5 triggers: Regime change detected
- Node 12 monitors: Oscillation amplitude
- BarrierBode: Factor hedge introduces sector exposure
- Expected response: Increase model uncertainty; reduce position sizing

---

## Commodities

:::{prf:definition} Commodity Pricing Framework
:label: def-commodity-pricing

**Spot-futures relationship.** Futures price under cost-of-carry:
$$
F_{t,T} = S_t \exp\left((r_t + c_t - y_t)(T-t)\right),
$$
where $c_t$ is storage cost and $y_t$ is convenience yield.

**Convenience yield dynamics.** Convenience yield reflects inventory scarcity:
$$
y_t = y_0 + \kappa(\bar{y} - y_t) dt + \sigma_y dW_t^y + \text{jump}(\text{inventory shock}).
$$

**Backwardation vs. contango regimes.** Market regime $K_t \in \{\text{backwardation}, \text{contango}\}$:
$$
K_t = \begin{cases}
\text{backwardation} & \text{if } y_t > r_t + c_t \\
\text{contango} & \text{if } y_t < r_t + c_t
\end{cases}
$$

**Risk geometry.** Commodity manifold with inventory state:
$$
g^{\text{comm}}_{ij} = \begin{pmatrix} \sigma_S^2 & \rho_{Sy}\sigma_S\sigma_y \\ \rho_{Sy}\sigma_S\sigma_y & \sigma_y^2 \end{pmatrix}
$$
Portfolio roll strategy follows geodesics on this manifold.

**Physical vs. financial convergence.** At delivery:
$$
\lim_{t \to T} F_{t,T} = S_T \quad \text{(physical settlement)}.
$$
This is enforced by arbitrage but requires storage/delivery capacity.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty risk (OTC), clearinghouse risk (exchange)
- Node 5 (Stationarity): Regime stability (backwardation/contango)
- Node 6 (Capacity): Storage capacity, delivery infrastructure
- Node 9 (Tameness): Price limit compliance
- Node 14 (Coupling): Spot-futures basis tracking
- BarrierInput: Inventory data, weather data
- BarrierRef: Benchmark price integrity (Brent, WTI, etc.)
- BarrierGap: Roll gap risk at expiry

**Failure mode mapping:**
- T.E (Flash Crash): Oil flash crash (April 2020 negative prices)
- D.E (Boom-Bust): Commodity supercycles
- B.E (External Shock): Geopolitical supply disruption
- T.C (Complexity): Physical vs. paper market divergence

**Stress test scenario:** Negative oil prices (April 2020 style)
- BarrierOmin: Price floor breach (negative prices possible)
- BarrierGap: Roll to next contract fails
- Node 6 triggers: Storage capacity exhausted
- Expected response: Halt physical delivery; cash settlement only

---

## Foreign Exchange

:::{prf:definition} FX Pricing Framework
:label: def-fx-pricing

**Covered interest parity (CIP).** Forward rate determined by interest differential:
$$
F_{t,T} = S_t \cdot \frac{B^{\text{dom}}_t(T)}{B^{\text{for}}_t(T)} = S_t \exp\left((r^{\text{dom}}_t - r^{\text{for}}_t)(T-t)\right).
$$

**CIP deviations (cross-currency basis).** Actual forward deviates due to funding constraints:
$$
F_{t,T}^{\text{actual}} = F_{t,T}^{\text{CIP}} \cdot \exp(-\text{basis}_t \cdot (T-t)),
$$
where basis reflects dollar funding premium.

**Uncovered interest parity (UIP).** Expected spot change:
$$
\mathbb{E}_t[S_{T}] = S_t \exp\left((r^{\text{dom}}_t - r^{\text{for}}_t)(T-t)\right) + \text{risk premium}.
$$
UIP failure is the **carry trade premium**.

**Triangle arbitrage.** For currencies A, B, C:
$$
S_{A/B} \times S_{B/C} \times S_{C/A} = 1.
$$
Deviations are arbitrage opportunities or market stress indicators.

**Risk geometry.** FX space forms a Lie group (currency ratios):
$$
g^{\text{FX}}_{ij} = \sigma_i \sigma_j \rho_{ij},
$$
with natural group structure for cross rates.
:::

**Permit checklist:**
- Node 1 (Solvency): Sovereign default risk (EM currencies)
- Node 3 (Leverage): Margin requirements, leverage limits
- Node 5 (Stationarity): Interest rate regime, carry regime
- Node 8 (Connectivity): Dealer network, ECN access
- Node 10 (Mixing): Market maker activity
- Node 14 (Coupling): CIP/UIP relationship
- BarrierLiq: Liquidity in crosses vs. majors
- BarrierRef: Benchmark fixings (WM/Reuters)
- BarrierCausal: Time zone gaps

**Failure mode mapping:**
- D.D (Dispersion Success): Carry trade crowding
- S.E (Supercritical Leverage): Carry unwind (JPY 2024)
- B.E (External Shock): EM currency crisis
- T.E (Flash Crash): GBP October 2016

**Stress test scenario:** G10 carry unwind
- Node 3 triggers: Leverage across carry trades
- D.D → S.E cascade: Crowded carry → forced deleveraging
- BarrierOmin: Gap risk in JPY crosses
- Expected response: Reduce leverage; widen stops; hedge with vol

---

## Credit and Defaultable Bonds

:::{prf:definition} Credit Pricing Framework
:label: def-credit-pricing

**Intensity-based model.** With hazard rate $\lambda_t$ and recovery $R$:
$$
P(t,T) = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T (r_u + \lambda_u) du}\right] + R \cdot \mathbb{E}_t^{\mathbb{Q}}\left[\int_t^T \lambda_u e^{-\int_t^u (r_s + \lambda_s) ds} du\right].
$$

**Credit spread decomposition.** Spread $s_t = \lambda_t (1-R) + \text{liquidity premium} + \text{risk premium}$:
$$
s_t = s^{\text{default}}_t + s^{\text{liquidity}}_t + s^{\text{risk}}_t.
$$

**Structural model (Merton).** Equity as call option on firm value:
$$
E_t = V_t N(d_1) - D e^{-rT} N(d_2),
$$
where $V_t$ is firm value and $D$ is debt face value.

**Distance to default.** Probability of default proxy:
$$
\text{DD}_t = \frac{\log(V_t/D) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}.
$$

**Risk geometry.** Credit manifold with coordinates (spread, duration, recovery):
$$
g^{\text{credit}}_{ij} = \begin{pmatrix} \sigma_s^2 & \rho_{sd} & \rho_{sr} \\ \rho_{sd} & \sigma_d^2 & \rho_{dr} \\ \rho_{sr} & \rho_{dr} & \sigma_r^2 \end{pmatrix}
$$
Geodesics represent constant-risk-adjusted credit curves.
:::

**Permit checklist:**
- Node 1 (Solvency): Default risk (primary concern)
- Node 2 (Turnover): Bond market liquidity (often poor)
- Node 5 (Stationarity): Credit cycle regime
- Node 6 (Capacity): New issue absorption
- Node 7 (Stiffness): Mean reversion in spreads
- Node 11 (Representation): Model adequacy (structural vs. intensity)
- BarrierSat: Concentration limits
- BarrierGap: Credit event gap risk
- BarrierInput: Financial statement data, rating actions

**Failure mode mapping:**
- C.E (Default Cascade): Contagion across credits
- C.D (Too-Big-to-Fail): Single-issuer concentration
- D.E (Boom-Bust): Credit cycle (spread compression → blow-out)
- T.D (Frozen Market): High-yield market freeze

**Stress test scenario:** IG → HY downgrade wave
- C.E triggers: Fallen angels create forced selling
- BarrierSat: Mandate constraints (IG-only funds)
- T.D risk: HY market cannot absorb supply
- Expected response: Pre-position for fallen angel risk; diversify by rating

::::{note} Connection to Standard Finance #12: Merton Model as Degenerate Default Boundary
**The General Law (Fragile Market):**
Default is a **barrier crossing** in the credit manifold:
$$
\tau_{\text{default}} = \inf\{t : V_t \le D(t; K_t)\}
$$
where $V_t$ is firm value and $D(t; K)$ is the regime-dependent default boundary. Credit spreads reflect the hitting probability.

**The Degenerate Limit:**
GBM firm value dynamics. Constant debt barrier. Single regime. No jumps.

**The Special Case (Merton 1974):**
$$
E_t = V_t N(d_1) - D e^{-rT} N(d_2), \quad \text{DD}_t = \frac{\log(V_t/D) + (\mu - \sigma^2/2)T}{\sigma\sqrt{T}}
$$
Equity is a call option on firm value; distance-to-default measures solvency.

**What the generalization offers:**
- **Regime-dependent boundaries**: Default barrier $D(t; K)$ varies with credit cycle
- **Jump processes**: Firm value can jump at regime transitions (sudden deterioration)
- **Contagion**: Default of one firm affects others via the regime codebook $K$
- **Recovery dynamics**: Recovery rate is regime-dependent, not constant
- **Credit manifold geometry**: Geodesics on $(s, d, R)$ space for portfolio hedging
::::

---

## Options and Derivatives

:::{prf:definition} Option Pricing Framework
:label: def-option-pricing

**Risk-neutral pricing.** European option value:
$$
V_t = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T r_u du} \cdot \text{Payoff}(S_T)\right].
$$

**Black-Scholes-Merton.** Under GBM dynamics ($dS = rS dt + \sigma S dW$):
$$
C_t = S_t N(d_1) - K e^{-r(T-t)} N(d_2), \quad d_{1,2} = \frac{\log(S/K) + (r \pm \sigma^2/2)(T-t)}{\sigma\sqrt{T-t}}.
$$

**Greeks as geometry.** The Greeks form a covariant structure:
$$
\Delta = \frac{\partial V}{\partial S}, \quad \Gamma = \frac{\partial^2 V}{\partial S^2}, \quad \Theta = \frac{\partial V}{\partial t}, \quad \text{Vega} = \frac{\partial V}{\partial \sigma}.
$$

**Volatility surface.** Implied vol $\sigma^{\text{imp}}(K,T)$ encodes market expectations. Surface dynamics:
$$
d\sigma^{\text{imp}} = \alpha dt + \xi dW^{\sigma},
$$
with constraints from no-arbitrage (Gatheral conditions).

**Risk metric on vol surface.** Curvature of the vol surface:
$$
R^{\text{vol}} = \frac{\partial^2 \sigma}{\partial K^2} - \frac{1}{T}\frac{\partial^2 \sigma}{\partial T^2},
$$
with high curvature indicating pricing stress.

**Replication and hedging.** Dynamic hedge portfolio:
$$
\Pi_t = V_t - \Delta_t S_t,
$$
requires continuous rebalancing. Market impact creates **hedging friction**.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty risk (OTC)
- Node 4 (Scale): Position sizing relative to gamma
- Node 5 (Stationarity): Vol regime stability
- Node 6 (Capacity): Liquidity at strikes/tenors
- Node 7 (Stiffness): Vol mean reversion
- Node 9 (Tameness): Tail risk bounds
- Node 10 (Mixing): Market maker activity
- Node 11 (Representation): Model adequacy (local vol, stochastic vol)
- Node 12 (Oscillation): Pin risk near expiry
- BarrierTypeII: Vol-of-vol crisis
- BarrierGap: Gap risk (discrete hedging)
- BarrierFreq: Gamma scalping frequency

**Failure mode mapping:**
- D.D (Dispersion Success): Vol selling crowding
- S.E (Supercritical Leverage): Gamma exposure × vol spike
- T.E (Flash Crash): Delta hedging cascade
- C.C (HFT Instability): Option market making at high frequency

**Stress test scenario:** Vol spike + liquidity withdrawal
- BarrierTypeII triggers: Vol-of-vol exceeds threshold
- Node 10 fails: Market makers pull quotes
- BarrierGap: Discrete hedging creates realized vs. implied gap
- Expected response: Reduce gamma exposure; accept wider bid-ask

::::{note} Connection to Standard Finance #13: Black-Scholes-Merton as Degenerate Feynman-Kac
**The General Law (Fragile Market):**
Option value satisfies the **Helmholtz equation on the risk manifold**:
$$
\left(-\Delta_G + \kappa^2\right) V = \rho_{\text{payoff}}
$$
where $\Delta_G$ is the Laplace-Beltrami operator on the Ruppeiner metric, $\kappa$ is the discount rate, and $\rho_{\text{payoff}}$ is the payoff density. The Greeks emerge as covariant derivatives: $\Delta = \nabla_S V$, $\Gamma = \nabla^2_S V$.

**The Degenerate Limit:**
Assume flat risk manifold ($G_{ij} \to \delta_{ij}$). Constant volatility ($\sigma = \text{const}$). Complete market (unique risk-neutral measure). No transaction costs or market impact.

**The Special Case (Black-Scholes-Merton PDE):**
$$
\frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + rS\frac{\partial V}{\partial S} - rV = 0
$$
Yields the explicit formula: $C = S N(d_1) - Ke^{-rT}N(d_2)$.

**What the generalization offers:**
- **Stochastic volatility**: Risk metric $G$ varies with market state—vol smile emerges naturally
- **Jump processes**: Helmholtz equation extends to jump-diffusion via non-local operators
- **Market incompleteness**: Multiple hedging strategies; optimal hedge minimizes residual risk in $G$-norm
- **Market impact**: Hedging friction enters via boundary conditions on $\partial\mathcal{W}$
- **Greeks as geometry**: Second-order Greeks (Gamma, Volga, Vanna) are curvature components of $G$
::::

---

## Volatility Products

:::{prf:definition} Volatility Product Pricing
:label: def-vol-pricing

**Variance swap.** Fair strike for variance swap:
$$
K_{\text{var}} = \mathbb{E}_t^{\mathbb{Q}}\left[\frac{1}{T}\int_t^T \sigma_u^2 du\right] = \frac{2}{T}\int_0^\infty \frac{C(K) + P(K)}{K^2} dK,
$$
derived from static replication via log contract.

**VIX definition.** VIX index approximates 30-day implied variance:
$$
\text{VIX}^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i),
$$
where $Q(K_i)$ are out-of-money option prices.

**Vol-of-vol dynamics.** VIX follows mean-reverting jump-diffusion:
$$
d(\text{VIX}) = \kappa(\bar{v} - \text{VIX})dt + \xi \cdot \text{VIX}^{\beta} dW + J dN,
$$
with jumps $J$ and Poisson process $N$.

**Volatility term structure.** VIX futures curve:
$$
F^{\text{VIX}}_{t,T} = \mathbb{E}_t^{\mathbb{Q}}[\text{VIX}_T],
$$
typically in contango (upward sloping) due to variance risk premium.

**Risk geometry.** Vol space is positively curved (vol bounded below by zero):
$$
g^{\text{vol}}_{ij} = \frac{\partial^2}{\partial \sigma_i \partial \sigma_j}\log p(\sigma),
$$
where $p(\sigma)$ is the vol distribution. Non-Euclidean distances matter for vol trading.
:::

**Permit checklist:**
- Node 1 (Solvency): Extreme loss potential (short vol)
- Node 3 (Leverage): Leveraged vol ETPs
- Node 4 (Scale): Position size vs. market vol
- Node 5 (Stationarity): Vol regime stability
- Node 7 (Stiffness): Vol mean reversion strength
- Node 9 (Tameness): Tail risk in vol distribution
- Node 12 (Oscillation): Vol clustering
- BarrierTypeII: Vol-of-vol crisis (primary concern)
- BarrierVac: Regime instability
- BarrierOmin: Gap risk in vol products

**Failure mode mapping:**
- D.D (Dispersion Success): Short vol crowding (XIV blowup)
- S.E (Supercritical Leverage): Leveraged vol ETP cascade
- T.E (Flash Crash): VIX spike (Volmageddon February 2018)
- D.E (Boom-Bust): Vol compression → explosion cycle

**Stress test scenario:** Volmageddon replay
- BarrierTypeII triggers: Vol-of-vol extreme
- S.E activates: Leveraged products force rebalancing
- D.D → S.E cascade: Crowded short vol → forced covering
- Expected response: Position limits on vol ETPs; dynamic margin

::::{note} Connection to Standard Finance #14: VIX as Degenerate Volatility Field
**The General Law (Fragile Market):**
Volatility is a **field** on the regime manifold with its own geometric structure:
$$
\sigma^2(K, S, T) = \mathbb{E}_t^{\mathbb{Q}(K)}\left[\frac{1}{T-t}\int_t^T \sigma_u^2 du \mid S_t, K_t\right]
$$
The vol surface metric $g^{\text{vol}}_{ij}$ captures information geometry on the space of volatility distributions.

**The Degenerate Limit:**
Single regime. Constant vol-of-vol. No vol jumps. Static replication via log contract.

**The Special Case (VIX / Variance Swaps):**
$$
\text{VIX}^2 = \frac{2}{T}\sum_i \frac{\Delta K_i}{K_i^2} e^{rT} Q(K_i), \quad K_{\text{var}} = \mathbb{E}^{\mathbb{Q}}\left[\frac{1}{T}\int_0^T \sigma_t^2 dt\right]
$$
VIX is the square root of the 30-day variance swap strike.

**What the generalization offers:**
- **Regime-dependent vol**: $\sigma^2(K)$ varies discretely with market regime
- **Vol-of-vol geometry**: $g^{\text{vol}}$ captures curvature of vol distribution
- **Jump risk in vol**: Regime transitions induce vol discontinuities (Volmageddon)
- **Variance risk premium**: Regime-aware decomposition of realized vs. implied variance
- **Vol surface consistency**: No-arbitrage constraints from the market hypostructure
::::

---

## Real Assets (Real Estate, Infrastructure)

:::{prf:definition} Real Asset Pricing Framework
:label: def-real-asset-pricing

**Discounted cash flow with illiquidity.** Real asset value:
$$
P_t = \mathbb{E}_t\left[\sum_{u>t} M_u (D_u - \iota_u)\right],
$$
where $\iota_u$ is the illiquidity discount (option value of liquidity foregone).

**Cap rate model.** Property value from net operating income:
$$
P_t = \frac{\text{NOI}_t}{\text{cap rate}_t}, \quad \text{cap rate}_t = r_t + \text{risk premium}_t - g_t,
$$
where $g_t$ is expected growth.

**Appraisal smoothing.** Reported values are smoothed:
$$
\hat{P}_t = \alpha P_t + (1-\alpha) \hat{P}_{t-1},
$$
creating **artificial autocorrelation** and understated volatility.

**NAV discount/premium.** REITs trade at:
$$
\text{Price}_t = \text{NAV}_t \times (1 + \text{discount/premium}_t),
$$
with discount reflecting liquidity premium, governance, and leverage.

**Illiquidity as option.** The illiquidity cost is a **real option**:
$$
\iota_t = \mathbb{E}_t\left[\max\left(0, V^{\text{liquid}}_\tau - V^{\text{illiquid}}_\tau\right)\right],
$$
where $\tau$ is the (uncertain) time of forced sale.
:::

**Permit checklist:**
- Node 1 (Solvency): Underlying tenant/asset credit
- Node 2 (Turnover): Transaction frequency (very low)
- Node 5 (Stationarity): Real estate cycle regime
- Node 6 (Capacity): Market depth for transactions
- Node 11 (Representation): Appraisal methodology adequacy
- BarrierInput: Valuation data quality and frequency
- BarrierCausal: Appraisal lag (quarterly typical)
- BarrierGap: Transaction gap (illiquidity)

**Failure mode mapping:**
- T.D (Frozen Market): Real estate transaction freeze (2008-09)
- D.E (Boom-Bust): Property cycles
- B.D (Liquidity Starvation): Redemption pressure on open-end funds
- D.C (Fundamental Uncertainty): Valuation during market stress

**Stress test scenario:** Open-end real estate fund redemption wave
- BarrierInput triggers: Stale NAV vs. market reality
- B.D activates: Redemptions exceed liquidity
- T.D risk: Fund gates, no transactions to price
- Expected response: Gate redemptions; mark to distressed sale levels

---

## Crypto and Digital Assets

:::{prf:definition} Crypto Asset Pricing Framework
:label: def-crypto-pricing

**Network value model.** Token value from network economics:
$$
S_t = \mathbb{E}_t\left[\sum_{u>t} M_u (\text{fees}_u + \text{staking}_u + \text{MEV}_u)\right],
$$
where MEV is maximal extractable value.

**Metcalfe's Law approximation.** Network value scales with users:
$$
V_t \propto n_t^{\alpha}, \quad \alpha \in [1.5, 2],
$$
where $n_t$ is active addresses/users.

**Staking yield.** Proof-of-stake yield:
$$
y^{\text{stake}}_t = \frac{\text{block rewards}_t + \text{tips}_t}{\text{staked amount}_t} - \text{slashing risk}_t.
$$

**Oracle dependency.** DeFi prices depend on oracles:
$$
P^{\text{DeFi}}_t = f(\text{Oracle}_t), \quad \text{Oracle}_t = \text{median}(\text{reporters}).
$$
Oracle manipulation creates **reference barrier risk**.

**Cross-chain arbitrage.** Price consistency across chains:
$$
P^{\text{Chain A}}_t = P^{\text{Chain B}}_t + \text{bridge cost}_t + \text{latency premium}_t.
$$
:::

**Permit checklist:**
- Node 1 (Solvency): Protocol security, smart contract risk
- Node 2 (Turnover): On-chain vs. CEX volume
- Node 3 (Leverage): DeFi leverage (liquidation cascades)
- Node 5 (Stationarity): Protocol upgrade stability
- Node 8 (Connectivity): Cross-chain bridges, CEX connectivity
- Node 10 (Mixing): MEV, front-running
- Node 11 (Representation): Valuation model adequacy
- BarrierRef: Oracle integrity (critical)
- BarrierInput: Blockchain data feed
- BarrierLiq: DEX vs. CEX liquidity fragmentation

**Failure mode mapping:**
- C.E (Default Cascade): DeFi liquidation cascade
- T.E (Flash Crash): Crypto flash crashes (common)
- C.C (HFT Instability): MEV extraction instability
- B.C (Agency Misalignment): Insider trading, rug pulls
- T.C (Complexity): Smart contract composability failure

**Stress test scenario:** Oracle manipulation attack
- BarrierRef triggers: Oracle reports manipulated price
- C.E cascade: Liquidations cascade from bad prices
- Node 8 fails: Bridge exploits compound losses
- Expected response: Circuit breakers on oracle deviation; multi-source oracles

---

## Private Equity and Venture Capital

:::{prf:definition} Private Equity Pricing Framework
:label: def-pe-pricing

**Stochastic exit model.** PE value with random exit:
$$
P_t = \mathbb{E}_t\left[M_\tau \cdot X_\tau\right],
$$
where $\tau$ is exit time (IPO, sale, failure) and $X_\tau$ is exit value.

**J-curve dynamics.** Fund NAV follows J-curve pattern:
$$
\text{NAV}_t = \text{Invested}_t - \text{Fees}_t + \text{Appreciation}_t,
$$
with early years showing fees > appreciation.

**Multiple expansion.** Value creation decomposition:
$$
\text{Return} = \text{Revenue growth} + \text{Margin expansion} + \text{Multiple expansion} + \text{Leverage}.
$$

**Secondary pricing.** Secondary market trades at:
$$
P^{\text{secondary}}_t = \text{NAV}_t \times (1 - \text{discount}_t),
$$
with discount reflecting illiquidity and information asymmetry.

**Wide bounds from incomplete markets.** Valuation is **interval-valued**:
$$
P_t \in [P^{\text{down}}_t, P^{\text{up}}_t],
$$
where bounds reflect scenario analysis, not point estimate.
:::

**Permit checklist:**
- Node 1 (Solvency): Portfolio company credit risk
- Node 2 (Turnover): Secondary market (limited)
- Node 5 (Stationarity): Exit market regime
- Node 6 (Capacity): Exit capacity (IPO window)
- Node 11 (Representation): Valuation methodology
- Node 17 (Lock): Contractual lock-up periods
- BarrierInput: Portfolio company data (quarterly, delayed)
- BarrierCausal: Reporting lag
- BarrierVariety: Incomplete hedging (unhedgeable)

**Failure mode mapping:**
- T.D (Frozen Market): IPO market closure
- D.E (Boom-Bust): VC cycle (2021 → 2022)
- D.C (Fundamental Uncertainty): Startup valuation
- B.D (Liquidity Starvation): LP distribution pressure

**Stress test scenario:** IPO market closure
- Node 6 triggers: Exit capacity zero
- T.D activates: No price discovery
- BarrierVariety: Cannot hedge, must hold
- Expected response: Mark to distressed comps; extend holding periods

---

## Structured Products

:::{prf:definition} Structured Product Pricing Framework
:label: def-structured-pricing

**Path-dependent pricing.** General structured product value:
$$
V_t = \mathbb{E}_t^{\mathbb{Q}}\left[e^{-\int_t^T r_u du} \cdot \text{Payoff}(\text{Path}_{t:T})\right],
$$
where payoff depends on full price path.

**Barrier option example.** Down-and-out call:
$$
V_t = C^{\text{vanilla}}_t - \left(\frac{S_t}{H}\right)^{2\lambda} C^{\text{vanilla}}_t(S \to H^2/S_t),
$$
where $H$ is barrier and $\lambda = (r - q - \sigma^2/2)/\sigma^2$.

**Correlation products.** Basket option value depends on correlation:
$$
V_t = f(\rho_{ij}), \quad \frac{\partial V}{\partial \rho_{ij}} = \text{correlation vega}.
$$

**CVA/DVA adjustment.** Credit valuation adjustment:
$$
V^{\text{adjusted}}_t = V^{\text{clean}}_t - \text{CVA}_t + \text{DVA}_t,
$$
accounting for counterparty and own default.

**Funding valuation adjustment (FVA).** Collateral costs:
$$
V^{\text{funded}}_t = V^{\text{adjusted}}_t - \text{FVA}_t,
$$
where FVA reflects funding spread on uncollateralized portion.
:::

**Permit checklist:**
- Node 1 (Solvency): Counterparty credit (critical)
- Node 3 (Leverage): Embedded leverage
- Node 4 (Scale): Position size vs. underlying liquidity
- Node 5 (Stationarity): Correlation regime stability
- Node 6 (Capacity): Hedging capacity for exotic risks
- Node 9 (Tameness): Tail risk (barriers, autocallables)
- Node 11 (Representation): Model risk (correlation, path)
- Node 14 (Coupling): Basis risk between hedge and product
- BarrierGap: Barrier breach gap risk
- BarrierVariety: Incomplete hedging of exotics
- BarrierBode: Hedging one Greek worsens another

**Failure mode mapping:**
- T.C (Complexity): CDO-squared opacity (2008)
- D.C (Fundamental Uncertainty): Correlation smile stress
- S.E (Supercritical Leverage): Autocallable barrier breach cascade
- B.C (Agency Misalignment): Suitability failures

**Stress test scenario:** Correlation spike + barrier breach
- BarrierGap triggers: Barrier breached in gap
- Node 11 fails: Correlation model breaks down
- BarrierVariety: Cannot hedge correlation exposure
- Expected response: Mark to scenario analysis; reserve for model risk

---

