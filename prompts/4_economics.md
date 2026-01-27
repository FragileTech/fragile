## 01_foundations/01_fragile_market.md

:::{prf:axiom} A1: Bounded Rationality
:label: axiom-bounded-rationality

All market agents operate under finite information capacity and computational constraints:

$$
I(a_t; Z_t) \le C_{\text{agent}} < \infty,

$$
where $a_t$ is agent action, $Z_t$ is market state, and $C_{\text{agent}}$ is agent channel capacity.
:::

:::{prf:axiom} A2: Thermodynamic Consistency
:label: axiom-thermo-consistency

Market dynamics obey the laws of thermodynamics:
1. **First law (Conservation):** Capital is conserved modulo external flows and dissipation.
2. **Second law (Entropy):** Entropy production is non-negative; $\Delta S \ge 0$ for isolated systems.
3. **Third law (Irreversibility):** Finite-time transactions have irreducible friction cost.
:::

:::{prf:axiom} A3: No-Arbitrage
:label: axiom-no-arbitrage

In the absence of barrier breaches, there exists no self-financing strategy yielding positive return with zero risk:

$$
\nexists \theta : V_0(\theta) = 0, \; V_T(\theta) \ge 0 \; \text{a.s.}, \; \mathbb{P}(V_T(\theta) > 0) > 0.

$$
:::

:::{prf:axiom} A4: Positive SDF
:label: axiom-positive-sdf

There exists a strictly positive stochastic discount factor $M_t > 0$ such that:

$$
p_t = \mathbb{E}_t[M_{t+1} \cdot \text{Payoff}_{t+1}].

$$
:::

:::{prf:axiom} A5: Information Grounding
:label: axiom-info-grounding

Prices must be coupled to observable boundary data:

$$
I(p_t; B_t) > 0,

$$
where $B_t$ is the boundary signal (quotes, flows, news).
:::

:::{prf:axiom} A6: Finite Complexity
:label: axiom-finite-complexity

The market state has bounded Kolmogorov complexity:

$$
K(Z_t) \le K_{\max} < \infty.

$$
:::

:::{prf:axiom} A7: Permit Completeness
:label: axiom-permit-completeness

Every market failure mode is detectable by at least one gate or barrier:

$$
\forall \text{ failure } F, \; \exists \text{ permit } P : P(F) = \text{FAIL}.

$$
:::

## 01_foundations/02_market_controller.md

:::{prf:definition} Market controller
:label: def-market-controller

The market has internal state

$$
Z_t := (K_t, Z_{n,t}, Z_{\mathrm{tex},t}) \in \mathcal{Z} = \mathcal{K} \times \mathcal{Z}_n \times \mathcal{Z}_{\mathrm{tex}},

$$
where:
- $K_t$ is a **discrete macro state** (regimes, liquidity state, risk-on/off),
- $Z_{n,t}$ is **structured nuisance** (microstructure, seasonal effects, inventory),
- $Z_{\mathrm{tex},t}$ is **texture residual** (high-frequency noise, idiosyncratic features).
:::

:::{prf:definition} Boundary / market interface
:label: def-market-boundary

The boundary variables at time $t$ are:

$$
B_t := (x_t, y_t, p_t, d_t, f_t, m_t, a_t),

$$
where:
- $x_t$ is public information (macro data, news),
- $y_t$ is microstructure data (order flow, quotes, depth),
- $p_t$ are observed prices,
- $d_t$ are observed cash flows (dividends, coupons, funding),
- $f_t$ are funding and collateral rates,
- $m_t$ are margin and constraint signals,
- $a_t$ is the aggregate action (net demand / rebalancing flow).
:::

:::{prf:definition} Market as input-output law
:label: def-market-environment-kernel

The market environment is a conditional law of future boundary signals given boundary history and actions:

$$
P_{\partial}(B_{t+1} \mid B_{\le t}, a_{\le t}).

$$
In the Markov case this reduces to $P_{\partial}(B_{t+1} \mid B_t, a_t)$, but the interpretation is the same: **pricing and stability depend only on observable boundary signals**.
:::

:::{prf:definition} Market symmetry group
:label: def-market-symmetry-group

A minimal symmetry group is

$$
\mathcal{G}_{\mathbb{M}} := G_{\text{numeraire}} \times S_{|\mathcal{A}|} \times G_{\text{measure}} \times G_{\text{unit}},

$$
where:
- $G_{\text{numeraire}}$ is positive scaling of the unit of account,
- $S_{|\mathcal{A}|}$ permutes asset labels,
- $G_{\text{measure}}$ is change of measure equivalent under the SDF,
- $G_{\text{unit}}$ rescales data units (volatility, notional).
:::

:::{prf:definition} Cohesive market topos
:label: def-cohesive-market

The market topos $\mathcal{E}_{\text{mkt}}$ is a cohesive $(\infty,1)$-topos equipped with the adjoint quadruple:

$$
\Pi \dashv \flat \dashv \sharp \dashv \text{coDisc} : \mathcal{E}_{\text{mkt}} \to \infty\text{-Grpd},

$$
where:
- **$\Pi$ (Shape):** extracts the homotopy type of market configurations (e.g., connected components of trading networks, fundamental group of arbitrage cycles),
- **$\flat$ (Flat/Discrete):** embeds constant sheaves; distinguishes pointwise (spot) pricing from derived structures,
- **$\sharp$ (Sharp/Codiscrete):** contractible path spaces; enables continuous deformation of pricing strategies.
:::

:::{prf:definition} Market object in the cohesive topos
:label: def-market-object-in-topos

A market configuration is an object $\mathcal{M} \in \mathcal{E}_{\text{mkt}}$ such that:

$$
\pi_0(\mathcal{M}) = \text{market regimes (discrete states)}, \quad \pi_1(\mathcal{M}) = \text{arbitrage cycles (gauge symmetries)},

$$
$$
\pi_n(\mathcal{M}) = \text{higher anomalies and obstructions for } n \ge 2.

$$
:::

:::{prf:remark} Why category theory?
:label: rem-why-category-theory

The categorical framing provides:
1. **Universality:** pricing theorems become natural transformations, not ad-hoc formulas.
2. **Compositionality:** complex instruments are built from simpler ones via colimits.
3. **Invariance:** gauge-independent statements are morphisms in the topos.
:::

:::{prf:definition} Wealth functor
:label: def-wealth-functor

Define the wealth functional as a derived functor:

$$
\Phi_{\bullet} : \mathcal{E}_{\text{mkt}} \to \text{Ch}(\mathbb{R}),

$$
where $\text{Ch}(\mathbb{R})$ is the derived category of real-valued chain complexes. The degree-$n$ component $\Phi_n$ measures:
- **$\Phi_0$:** Mark-to-market value (0th homology = direct valuation).
- **$\Phi_1$:** Contingent claims and options (1st homology = linear exposure).
- **$\Phi_2$:** Convexity and gamma exposure (2nd homology = curvature risk).
- **$\Phi_n$:** Higher-order Greeks and exotic path dependencies.
:::

:::{prf:definition} Euler characteristic of a portfolio
:label: def-portfolio-euler-characteristic

For a truncated exposure complex (up to order $N$), define the Euler characteristic:

$$
\chi_N(\Phi_{\bullet}) := \sum_{n=0}^{N} (-1)^n \operatorname{rank}(\Phi_n).

$$
Heuristically: $n=0$ corresponds to linear cashflows (NPV-like), $n=1$ to optionality, $n=2$ to
convexity, and higher $n$ to higher-order Greeks/path dependence.
:::

:::{prf:remark} Cohomological pricing (heuristic)
:label: rem-cohomological-pricing

This cohomological packaging is an accounting device: it organizes exposures by “order” and makes
gauge covariance easier to state. The rigorous invariance used throughout the volume is the
numéraire/measure-change covariance of valuation ({prf:ref}`thm-mkt-equivariance`).
:::

:::{prf:definition} Shape modality ($\Pi$) for markets
:label: def-market-shape-modality

$$
\Pi(\mathcal{M}) = \text{homotopy type of market configuration space}.

$$
- **Application:** Detects whether two market states are "topologically equivalent" (connected by continuous deformation) or "topologically distinct" (separated by phase transition).
- **Observable:** Number of connected components = number of distinct regimes.
:::

:::{prf:definition} Flat modality ($\flat$) for markets
:label: def-market-flat-modality

$$
\flat(\mathcal{M}) = \text{discrete/pointwise evaluation of prices}.

$$
- **Application:** Spot prices, mark-to-market, instantaneous valuation.
- **Contrast with $\sharp$:** $\flat$ ignores path dependence; $\sharp$ includes it.
:::

:::{prf:definition} Sharp modality ($\sharp$) for markets
:label: def-market-sharp-modality

$$
\sharp(\mathcal{M}) = \text{contractible deformation space of price paths}.

$$
- **Application:** Path-dependent options (Asian, barrier, lookback), accumulated dividends, accrued interest.
- **Mathematical structure:** $\sharp(\mathcal{M})$ has trivial homotopy groups—all paths are equivalent up to endpoints.
:::

:::{prf:proposition} Modal decomposition of pricing
:label: prop-market-modal-decomposition

Any pricing functional $P$ decomposes as:

$$
P = P_{\flat} + P_{\sharp - \flat} + P_{\Pi},

$$
where:
- $P_{\flat}$ is the spot/intrinsic value,
- $P_{\sharp - \flat}$ is the path-dependent premium (time value, optionality),
- $P_{\Pi}$ is the topological risk premium (regime/crisis premium).
:::

:::{prf:definition} Ruppeiner risk metric (state space)
:label: def-ruppeiner-risk-metric

The state-space metric is:

$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{\partial^2 F}{\partial z^i \partial z^j} \cdot \frac{1}{T},

$$
where $S$ is entropy, $F$ is free energy, and $T$ is risk temperature. This measures the **thermodynamic distance** between market states.
:::

:::{prf:definition} Agent taxonomy (market roles)
:label: def-market-agent-taxonomy

| Agent Type | Objective | Time Horizon | Key Constraint |
|------------|-----------|--------------|----------------|
| **Market Maker** | Minimize inventory risk | Intraday | Spread ≥ cost |
| **Arbitrageur** | Exploit mispricings | Seconds–days | Capital limits |
| **Hedger** | Minimize variance | Weeks–years | Basis risk |
| **Speculator** | Maximize expected return | Days–months | Drawdown limits |
| **Index Fund** | Track benchmark | Continuous | Tracking error |
| **Central Bank** | Stability | Permanent | Political mandate |
:::

:::{prf:definition} Aggregate market dynamics
:label: def-market-aggregate-dynamics

The market evolution $S_t$ is the composition of agent-level dynamics:

$$
S_t = \bigcirc_{j \in \mathcal{J}} S_t^{(j)},

$$
where $\mathcal{J}$ indexes active agents and $\bigcirc$ denotes composition under market clearing.
:::

## 01_foundations/04_market_hypostructure.md

:::{prf:definition} Market hypostructure
:label: def-market-hypostructure

A market hypostructure is a tuple

$$
\mathbb{H}_{\text{mkt}} = (\mathcal{X}, \nabla, \Phi_{\bullet}, \tau, \partial_{\bullet}),

$$
where:
1. **State stack $\mathcal{X}$:** the configuration stack of balance sheets, contracts, positions, and market microstructure.
2. **Connection $\nabla$:** time evolution under trading, settlement, and policy constraints.
3. **Potential $\Phi_{\bullet}$:** a thermoeconomic potential encoding total utility, risk, and costs.
4. **Truncation structure $\tau$:** market constraints (capital, leverage, liquidity, information capacity, topology of the trading network).
5. **Boundary morphism $\partial_{\bullet}$:** restriction to the market interface $B_t$.

**Interpretation.** $\mathbb{H}_{\text{mkt}}$ is the **object on which pricing lives**. Prices are not intrinsic; they are sections of boundary data consistent with $\nabla$ and $\Phi_{\bullet}$ under $\tau$.
:::

:::{prf:definition} Self-consistent market
:label: def-market-self-consistent

A market trajectory is self-consistent if the evolution $S_t$ preserves all permits and converges
to a state where pricing is internally and externally consistent.
:::

:::{prf:definition} Thin market kernel
:label: def-thin-kernel

A thin market kernel is a quintuple:

$$
\mathcal{T}_{\text{mkt}} = (\mathcal{X}^{\text{thin}}, \Phi^{\text{thin}}, \mathfrak{D}^{\text{thin}}, G^{\text{thin}}, \partial^{\text{thin}}),

$$
where each component has explicit market interpretation:

**Component 1: Arena $\mathcal{X}^{\text{thin}} = (X, d, \mathfrak{m})$**
- **$X$:** Polish metric space of market configurations (positions, prices, balances).
- **$d$:** Distance function; typically $d(x, x') = \|p - p'\|_2 + \lambda \|\theta - \theta'\|_{\text{param}}$.
- **$\mathfrak{m}$:** Reference measure; empirical distribution of historical market states.

**Component 2: Potential $\Phi^{\text{thin}} = (\Phi, \alpha_{\Phi})$**
- **$\Phi : X \to \mathbb{R}_{\ge 0}$:** Total market risk functional (e.g., aggregate VaR, expected shortfall).
- **$\alpha_{\Phi}$:** Scaling dimension; for equity markets, typically $\alpha_{\Phi} \approx 2$ (quadratic risk).

**Component 3: Dissipation $\mathfrak{D}^{\text{thin}} = (\mathfrak{D}, \beta_{\mathfrak{D}})$**
- **$\mathfrak{D} : X \times X \to \mathbb{R}_{\ge 0}$:** Transaction cost and friction functional.
- **$\beta_{\mathfrak{D}}$:** Scaling dimension; typically $\beta_{\mathfrak{D}} = 1$ (linear in volume) or $3/2$ (with impact).

**Component 4: Symmetry $G^{\text{thin}}$**
- Symmetry group acting on $\mathcal{X}$; at minimum, $G^{\text{thin}} = G_{\text{numeraire}} \times S_{|\mathcal{A}|}$.

**Component 5: Boundary $\partial^{\text{thin}} = (\mathcal{B}, \mathrm{Tr}, \mathcal{J}, \mathcal{R})$**
- **$\mathcal{B}$:** Boundary data space (observed prices, flows, news).
- **$\mathrm{Tr} : X \to \mathcal{B}$:** Trace map projecting bulk state to boundary observables.
- **$\mathcal{J}$:** Boundary flux (order flow, capital flow).
- **$\mathcal{R}$:** Risk signal (VIX, credit spreads, funding stress).
:::

:::{prf:theorem} Thin kernel sufficiency
:label: thm-thin-kernel-sufficiency

Given a thin market kernel $\mathcal{T}_{\text{mkt}}$, the Sieve constructor $F_{\text{Sieve}}$
produces a full market hypostructure:

$$
F_{\text{Sieve}}(\mathcal{T}_{\text{mkt}}) = \mathbb{H}_{\text{mkt}}.

$$

*Proof sketch.* This is a direct market reading of the Thin-to-Hypo promotion principle: given Thin
Kernel Objects ({prf:ref}`def-thin-objects`), the Expansion Adjunction ({prf:ref}`thm-expansion-adjunction`)
promotes thin kernels to hypostructures by systematically lifting discrete certificates into the
ambient cohesive setting. $\square$
:::

:::{prf:definition} Market RCD condition
:label: def-market-rcd

The market state space $(\mathcal{X}, d, \mathfrak{m})$ satisfies $\mathrm{RCD}(K, N)$ if:
1. **Ricci curvature bounded below:** $\mathrm{Ric} \ge K$ (market has limited "negative curvature" / instability).
2. **Dimension bounded above:** $\dim \le N$ (finite degrees of freedom).
:::

:::{prf:proposition} RCD contraction (modeling condition)
:label: prop-market-rcd-contraction

If a market distribution $\mu_t$ evolves as a $K$-contractive semigroup in $(\mathcal{X}, W_2)$ (for
example, as a gradient flow on an $\mathrm{RCD}(K,N)$ space with $K>0$), then:

$$
W_2(\mu_t, \mu_{\infty}) \le e^{-Kt} W_2(\mu_0, \mu_{\infty}),

$$
where $W_2$ is Wasserstein-2 distance and $\mu_{\infty}$ is the equilibrium distribution.

*Market implication:* Treat $K$ as an effective “price discovery stiffness”: larger $K$ implies
faster mixing/convergence in the space of market states.
:::

:::{prf:definition} Market Cheeger energy
:label: def-market-cheeger-energy

$$
\mathrm{Ch}(f | \mathfrak{m}) := \frac{1}{2} \inf \left\{ \liminf_{n \to \infty} \int_X |\nabla f_n|^2 \, d\mathfrak{m} : f_n \to f \text{ in } L^2 \right\}.

$$
:::

:::{prf:proposition} Liquidity as a Cheeger constant
:label: prop-liquidity-cheeger-constant

The market liquidity index is:

$$
\mathcal{L}_{\text{mkt}} := \inf_{A : 0 < \mathfrak{m}(A) < 1} \frac{\text{Per}(A)}{\min(\mathfrak{m}(A), 1 - \mathfrak{m}(A))},

$$
where $\text{Per}(A)$ is the perimeter of set $A$ in the metric-measure space.

Low $\mathcal{L}_{\text{mkt}}$ indicates "bottlenecks" where capital cannot flow freely—liquidity traps.
:::

## 01_foundations/05_thermoeconomics.md

:::{prf:definition} Free energy
:label: def-free-energy

$$
F_t := U_t - T_t S_t.

$$
$F_t$ is the **extractable value** after accounting for uncertainty. In equilibrium, pricing minimizes expected free energy subject to constraints.
This is the standard MaxEnt free-energy form under information constraints {cite}`jaynes1957information,cover2006elements`.
:::

:::{prf:definition} Capital balance
:label: def-capital-balance

$$
\Delta U = \Delta W + \Delta Q - \Delta D,

$$
where:
- $\Delta W$ is work done by trading (rebalancing gains),
- $\Delta Q$ is external inflow (income, dividends, funding),
- $\Delta D$ is dissipation (transaction costs, slippage, defaults).
:::

:::{prf:definition} Entropy production
:label: def-entropy-production

$$
\Delta S \ge \Delta S_{\text{info}} + \Delta S_{\text{friction}},

$$
with strictly positive entropy production when trading costs and information loss are nonzero.
:::

:::{prf:definition} Thermoeconomic SDF
:label: def-thermo-sdf

For payoff $X_{t+1}$,

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
:::

:::{prf:definition} Ruppeiner metric tensor (markets)
:label: def-ruppeiner-market

The market risk metric is the Hessian of entropy:

$$
G_{ij}(z) := -\frac{\partial^2 S}{\partial z^i \partial z^j} = \frac{1}{T} \frac{\partial^2 F}{\partial z^i \partial z^j},

$$
where:
- $z = (z^1, \ldots, z^n)$ are market state coordinates (e.g., log-prices, volatilities, spreads),
- $S$ is the market entropy (uncertainty about future prices),
- $F$ is free energy (risk-adjusted value),
- $T$ is risk temperature (inverse risk aversion).
:::

:::{prf:proposition} Metric components for Gaussian markets
:label: prop-ruppeiner-gaussian-market

For a market with log-returns $r_i$ and covariance $\Sigma_{ij}$:

$$
G_{ij} = \frac{1}{T} \Sigma^{-1}_{ij}.

$$
High covariance = low metric distance (easy to arbitrage); low covariance = high metric distance (hard to hedge).
:::

:::{prf:definition} Thermodynamic distance
:label: def-thermodynamic-distance

The distance between market states $z$ and $z'$ is:

$$
d_G(z, z') := \int_0^1 \sqrt{G_{ij}(z(\tau)) \dot{z}^i(\tau) \dot{z}^j(\tau)} \, d\tau,

$$
minimized over paths $z(\tau)$ from $z$ to $z'$.
:::

:::{prf:definition} Market phases
:label: def-market-phase

| Phase | Entropy | Structure | Price Behavior | Examples |
|-------|---------|-----------|----------------|----------|
| **Crystal** | Low | Ordered, predictable | Prices at fundamental value | Government bonds at par, pegged FX |
| **Liquid** | Medium | Structured randomness | Efficient pricing with noise | Normal equity markets, active FX |
| **Gas** | High | Chaotic, unpredictable | Prices disconnected from fundamentals | Flash crashes, speculative bubbles |
:::

:::{prf:definition} Phase order parameter
:label: def-market-phase-order-parameter

The market phase is characterized by:

$$
\Psi := \frac{H(K_t)}{\log |\mathcal{K}|} \in [0, 1],

$$
where $H(K_t)$ is the entropy of the regime distribution.
- $\Psi \approx 0$: Crystal phase (one dominant regime).
- $\Psi \approx 0.5$: Liquid phase (moderate uncertainty).
- $\Psi \approx 1$: Gas phase (maximum uncertainty, all regimes equiprobable).
:::

:::{prf:definition} Phase transition detection rule
:label: def-market-phase-transition-detection

A phase transition occurs at time $t^*$ if:

$$
\left| \frac{d\Psi}{dt} \right|_{t=t^*} > \Psi_{\text{crit}},

$$
where $\Psi_{\text{crit}}$ is a threshold (typically calibrated to VIX spikes or spread blowouts).
:::

:::{prf:definition} Critical exponents
:label: def-market-critical-exponents

Near a phase transition, observables scale as:

$$
\text{Volatility} \sim |T - T_c|^{-\gamma}, \quad \text{Correlation length} \sim |T - T_c|^{-\nu},

$$
where $T_c$ is the critical temperature and $\gamma, \nu$ are critical exponents.
:::

:::{prf:definition} Market scaling exponents
:label: def-market-scaling-exponents

| Exponent | Symbol | Meaning | Observable Proxy |
|----------|--------|---------|------------------|
| **Risk Temperature** | $\alpha$ | Curvature of value landscape | $\sqrt{\mathbb{E}[(\nabla V)^2]}$ |
| **Volatility Temperature** | $\beta$ | Plasticity of price dynamics | Realized vol / Implied vol |
| **Liquidity Temperature** | $\gamma$ | Fluidity of capital flows | Bid-ask spread inverse |
| **Leverage Temperature** | $\delta$ | Amplification of positions | Aggregate leverage ratio |
:::

:::{prf:assumption} Temperature hierarchy (stability condition)
:label: asm-temperature-hierarchy

For stable markets, the scaling exponents must satisfy:

$$
\alpha > \beta > \gamma > \delta,

$$
meaning:
1. Risk perception ($\alpha$) must dominate volatility ($\beta$).
2. Volatility ($\beta$) must dominate liquidity effects ($\gamma$).
3. Liquidity ($\gamma$) must dominate leverage amplification ($\delta$).
:::

:::{prf:definition} Market Einstein tensor
:label: def-market-einstein-tensor

Define the Einstein tensor:

$$
\mathcal{G}_{ij} := R_{ij} - \frac{1}{2} R \, G_{ij},

$$
where $R_{ij}$ is the Ricci curvature and $R = G^{ij} R_{ij}$ is the scalar curvature.
:::

:::{prf:definition} Risk-energy tensor
:label: def-risk-energy-tensor

The risk-energy tensor is:

$$
\mathcal{T}_{ij} := \frac{\partial \Phi}{\partial z^i} \frac{\partial \Phi}{\partial z^j} - \frac{1}{2} G_{ij} |\nabla \Phi|^2_G + \Lambda G_{ij},

$$
where $\Phi$ is the risk potential and $\Lambda$ is a "cosmological constant" (baseline risk premium).
:::

:::{prf:assumption} Market Einstein equations (closure)
:label: asm-market-einstein-equations

In equilibrium, curvature and risk satisfy:

$$
\mathcal{G}_{ij} = \kappa \mathcal{T}_{ij},

$$
where $\kappa$ is the coupling constant (market-specific).
:::

:::{prf:remark} No-arbitrage as flatness (heuristic)
:label: rem-no-arbitrage-flatness

If $\mathcal{T}_{ij} = 0$ (no risk concentration), then $\mathcal{G}_{ij} = 0$ (flat space), and all
paths are equivalent—no arbitrage opportunities.
:::

:::{prf:definition} Geodesic equation for portfolios
:label: def-portfolio-geodesic-equation

A portfolio path $w(t)$ is geodesic if:

$$
\frac{d^2 w^i}{dt^2} + \Gamma^i_{jk} \frac{dw^j}{dt} \frac{dw^k}{dt} = 0,

$$
where $\Gamma^i_{jk}$ are Christoffel symbols derived from $G_{ij}$.
:::

:::{prf:proposition} Natural gradient update
:label: prop-natural-gradient-portfolio-update

The optimal portfolio update is:

$$
\Delta w = -\eta \, G^{-1} \nabla_w \Phi,

$$
where $G^{-1}$ is the inverse metric and $\nabla_w \Phi$ is the risk gradient.
:::

:::{prf:definition} Covariant portfolio dissipation
:label: def-covariant-portfolio-dissipation

The dissipation rate along a portfolio path is:

$$
\mathfrak{D}_{\text{geo}} := \left\langle \nabla_w V, \dot{w} \right\rangle_G = G_{ij} \frac{\partial V}{\partial w^i} \dot{w}^j,

$$
where $V$ is the value function and $\langle \cdot, \cdot \rangle_G$ is the inner product under the Ruppeiner metric.
:::

:::{prf:assumption} Geodesic optimality (transaction cost model)
:label: asm-geodesic-optimality

Among all self-financing paths from $w_0$ to $w_T$, the geodesic minimizes total transaction cost:

$$
\mathcal{C}[w] = \int_0^T \sqrt{G_{ij}(w) \dot{w}^i \dot{w}^j} \, dt.

$$
:::

:::{prf:assumption} Landauer-style lower bound (market information processing)
:label: asm-landauer-bound-markets

Any trade that erases $\Delta I$ bits of market information must dissipate at least:

$$
\Delta Q \ge k_B T \ln(2) \cdot \Delta I,

$$
where $k_B T$ is thermal energy (in market context: risk temperature × volatility).
:::

:::{prf:definition} Information-theoretic spread
:label: def-information-theoretic-spread

The minimum bid-ask spread is:

$$
s_{\min} = \frac{k_B T \ln(2)}{V_{\text{avg}}} \cdot H(K_t),

$$
where $V_{\text{avg}}$ is average trade volume and $H(K_t)$ is regime entropy.
:::

:::{prf:remark} Efficient market bound (heuristic)
:label: rem-efficient-market-spread-bound

In an efficient market, the actual spread satisfies:

$$
s_{\text{actual}} \ge s_{\min},

$$
with equality only in the theoretical limit of zero noise and infinite liquidity.
:::

:::{prf:definition} Market Log-Sobolev constant
:label: def-market-log-sobolev-constant

The LSI constant $\rho_{\text{LSI}}$ satisfies:

$$
\text{Ent}_{\mathfrak{m}}(f^2) \le \frac{2}{\rho_{\text{LSI}}} \int |\nabla f|^2 \, d\mathfrak{m},

$$
for all smooth $f$ with $\int f^2 d\mathfrak{m} = 1$.
:::

:::{prf:remark} LSI and VaR (concentration-style bound)
:label: rem-lsi-var

The Value-at-Risk at confidence $\alpha$ satisfies a concentration-style upper bound of the form:

$$
\text{VaR}_{\alpha} \le \mu + \sigma \sqrt{\frac{2}{\rho_{\text{LSI}}} \ln\left(\frac{1}{1-\alpha}\right)},

$$
where $\mu, \sigma$ are mean and standard deviation.
:::

:::{prf:definition} Regime transition cost (Wasserstein-2)
:label: def-regime-transition-cost

The cost of transitioning from regime $K$ to regime $K'$ is:

$$
W_2(\mu_K, \mu_{K'}) := \left( \inf_{\pi \in \Pi(\mu_K, \mu_{K'})} \int d(x, y)^2 \, d\pi(x, y) \right)^{1/2},

$$
where $\Pi(\mu_K, \mu_{K'})$ is the set of couplings.
:::

:::{prf:definition} Regime transition warning rule
:label: def-regime-transition-warning

A regime transition is imminent when:

$$
\frac{d}{dt} W_2(\mu_t, \mu_K) < -\epsilon_{\text{trans}},

$$
where $\mu_K$ is the current regime distribution and $\epsilon_{\text{trans}}$ is a threshold.
:::

## 02_core_pricing/02_asset_pricing_core.md

:::{prf:definition} No-arbitrage
:label: def-no-arbitrage

There is no self-financing strategy with zero cost and nonnegative payoff that is positive with
positive probability.

See also Axiom {prf:ref}`axiom-no-arbitrage` for the permit-gated version used throughout the
volume.
:::

:::{prf:theorem} SDF existence (FTAP form)
:label: thm-sdf-existence

Under standard regularity (locally bounded prices, NFLVR), there exists a strictly positive
process $M_t$ such that for all assets {cite}`harrison1979martingales,harrison1981martingales,delbaen1994ftap`:

$$
S_t^i = \mathbb{E}_t[ M_{t+1} (S_{t+1}^i + D_{t+1}^i) ].

$$
$M_t$ is the **stochastic discount factor**.
:::

## 02_core_pricing/03_market_sieve.md

:::{prf:definition} Gate permits
:label: def-market-gate-permits
For each gate $i$, the outcome alphabet is $\{\text{YES}, \text{NO}\}$ with certificates:
- $K_i^+$ (YES): the predicate $P_i$ holds on the current state or window.
- $K_i^-$ (NO): the predicate fails or cannot be certified.
:::

:::{prf:definition} Barrier permits
:label: def-market-barrier-permits
For each barrier, the outcome alphabet is $\{\text{Blocked}, \text{Breached}\}$ with certificates:
- $K^{\mathrm{blk}}$: the barrier holds; proceed.
- $K^{\mathrm{br}}$: the barrier fails; enter defense mode.
:::

:::{prf:definition} Surgery permits
:label: def-market-surgery-permits
A surgery outputs a re-entry certificate

$$
K^{\mathrm{re}} = (D_S, x', \pi)

$$
where $D_S$ is the intervention data, $x'$ is the post-surgery state, and $\pi$ proves the next gate's precondition.
:::

:::{prf:definition} YES-tilde permits (equivalence)
:label: def-market-yes-tilde
A YES$^\sim$ permit allows acceptance up to equivalence (e.g., numeraire change):

$$
K_i^{\sim} = (K_{\text{equiv}}, K_{\text{transport}}, K_i^+[\tilde{x}]).

$$
:::

:::{prf:definition} Promotion permits
:label: def-market-promotion
Blocked certificates may be promoted to YES if other certificates imply the original predicate:

$$
K_i^{\mathrm{blk}} \wedge \bigwedge_j K_j^+ \Rightarrow K_i^+.

$$
Promotions may be immediate (past-only) or a-posteriori (future-enabled).
:::

:::{prf:definition} Inconclusive upgrade permits
:label: def-market-inc-upgrade
If a NO certificate is due to missing prerequisites, it can be upgraded when those prerequisites are later supplied:

$$
K_P^{\mathrm{inc}} \wedge \bigwedge_{j \in J} K_j^+ \Rightarrow K_P^+.

$$
:::

:::{prf:definition} Market Sieve loss
:label: def-market-sieve-loss

For learning, calibration, or monitoring, it is useful to aggregate permit outcomes into a single
scalar objective:

$$
\mathcal{L}_{\text{Sieve}}
:=
\sum_{i \in \mathcal{G}} \lambda_i\,\ell_i
\;+\;
\sum_{b \in \mathcal{B}} \Lambda_b\,\mathbf{1}[b\ \text{breached}],

$$
where $\ell_i=0$ when gate $i$ is certified YES and $\ell_i>0$ measures the gate violation
magnitude, while barrier breaches are treated as high-penalty events.
:::

:::{prf:definition} Node 1 Specification
:label: def-node1-solvency

**Predicate:** Total mark-to-market losses are bounded by available capital.

$$
P_1 : \quad \Phi(x_t) := \text{VaR}_{\alpha}(L_t) \le C_t - \epsilon_{\text{buffer}},

$$
where $L_t$ is the loss distribution, $C_t$ is available capital, and $\epsilon_{\text{buffer}}$ is a safety margin.

**Market interpretation:** The market participant (or aggregate) can absorb expected losses without insolvency.

**Observable metrics:**
- Capital ratio: $\rho_{\text{cap}} := C_t / \text{RWA}_t$
- VaR breach count over rolling window
- Distance to default: $\text{DD}_t := (\mu_t - D_t) / \sigma_t$

**Certificate format:**

$$
K_1^+ = (\text{VaR}_{\alpha}, C_t, \rho_{\text{cap}}, \text{timestamp})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{solvency}} = \lambda_1 \cdot \max(0, \Phi(x_t) - C_t + \epsilon_{\text{buffer}})^2

$$
:::

:::{prf:definition} Node 2 Specification
:label: def-node2-turnover

**Predicate:** Trading activity is bounded; no infinite-frequency switching.

$$
P_2 : \quad N_t := \sum_{s \le t} \mathbb{I}[\text{trade at } s] \le N_{\max}(t),

$$
where $N_{\max}(t)$ is a time-dependent bound (e.g., $N_{\max}(t) = \kappa \cdot t$).

**Market interpretation:** Prevents "Zeno paradox" where infinite trades occur in finite time. Detects HFT instability and quote stuffing.

**Observable metrics:**
- Turnover rate: $\tau_t := \text{Volume}_t / \text{AUM}_t$
- Order-to-trade ratio
- Cancel rate: fraction of orders cancelled before execution

**Certificate format:**

$$
K_2^+ = (\tau_t, \text{O2T ratio}, \text{cancel rate}, \text{window})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{zeno}} = \lambda_2 \cdot D_{KL}(\pi_t \| \pi_{t-1})

$$
:::

:::{prf:definition} Node 3 Specification
:label: def-node3-compactness

**Predicate:** Positions and leverage are bounded; no concentration blow-up.

$$
P_3 : \quad \|w_t\|_{\infty} \le w_{\max} \quad \text{and} \quad \text{Lev}_t := \frac{\sum_i |w_t^i|}{\text{NAV}_t} \le L_{\max}.

$$

**Market interpretation:** Energy (capital at risk) concentrates or disperses but doesn't escape to infinity. Detects dangerous position concentrations.

**Observable metrics:**
- Gross leverage: $\text{Lev}_t$
- Herfindahl index: $\text{HHI}_t := \sum_i (w_t^i / \sum_j |w_t^j|)^2$
- Maximum single position as fraction of NAV

**Certificate format:**

$$
K_3^+ = (\text{Lev}_t, \text{HHI}_t, w_{\max}, \|w_t\|_{\infty})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{compact}} = \lambda_3 \cdot \max(0, \text{Lev}_t - L_{\max})^2 + \lambda_3' \cdot \text{HHI}_t

$$
:::

:::{prf:definition} Node 4 Specification
:label: def-node4-scale

**Predicate:** Model parameters evolve slower than the market adapts.

$$
P_4 : \quad \|\nabla_t \theta_t\|^2 \le \epsilon_{\text{drift}} \cdot \|\nabla_\theta \mathcal{L}\|^2,

$$
where $\theta_t$ are model parameters and $\nabla_t$ is the time derivative.

**Market interpretation:** The pricing model is not chasing noise. Parameters should be stable relative to signal.

**Observable metrics:**
- Parameter volatility: $\sigma(\theta)$ over rolling window
- Signal-to-noise ratio of parameter updates
- Autocorrelation of parameter changes

**Certificate format:**

$$
K_4^+ = (\|\nabla_t \theta_t\|, \text{SNR}_{\theta}, \text{AC}_{\theta})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{scale}} = \lambda_4 \cdot \|\theta_t - \theta_{t-1}\|^2 / \|\nabla_\theta \mathcal{L}\|^2

$$
:::

:::{prf:definition} Node 5 Specification
:label: def-node5-stationarity

**Predicate:** The current regime is statistically stationary.

$$
P_5 : \quad \text{ADF}(r_{t-W:t}) < \text{crit}_{\alpha} \quad \text{or} \quad \text{KPSS}(r_{t-W:t}) > \text{crit}_{\alpha}',

$$
where ADF is Augmented Dickey-Fuller and KPSS is Kwiatkowski-Phillips-Schmidt-Shin test.

**Market interpretation:** Price dynamics are stable within the current regime; no structural breaks.

**Observable metrics:**
- ADF test statistic
- KPSS test statistic
- Chow test for structural breaks
- Rolling mean/variance stability

**Certificate format:**

$$
K_5^+ = (\text{ADF}_t, \text{KPSS}_t, \text{break count}, \text{window})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{stat}} = \lambda_5 \cdot \text{ReLU}(\text{ADF}_t - \text{crit}_{\alpha})

$$
:::

:::{prf:definition} Node 6 Specification
:label: def-node6-capacity

**Predicate:** Market depth supports the information content of the state.

$$
P_6 : \quad I(B_t; K_t) \le \mathcal{C}_{\text{channel}} := \log_2(1 + \text{SNR}_{\text{depth}}),

$$
where $\mathcal{C}_{\text{channel}}$ is the information capacity of the order book.

**Market interpretation:** The market can transmit enough information to support the complexity of the current regime.

**Observable metrics:**
- Order book depth at multiple levels
- Effective spread
- Kyle's lambda (price impact coefficient)
- Information ratio of order flow

**Certificate format:**

$$
K_6^+ = (I(B_t; K_t), \mathcal{C}_{\text{channel}}, \text{depth}, \lambda_{\text{Kyle}})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{cap}} = \lambda_6 \cdot \mathcal{L}_{\text{InfoNCE}}(z_t, z_{t+1})

$$
:::

:::{prf:definition} Node 7 Specification
:label: def-node7-stiffness

**Predicate:** The value gradient is non-vanishing; price discovery is possible.

$$
P_7 : \quad \|\nabla_z V(z_t)\| \ge \epsilon_{\text{stiff}} > 0.

$$

**Market interpretation:** Prices respond to information. A flat value landscape means prices are stuck (no signal).

**Observable metrics:**
- Value gradient norm
- Price impact: $\Delta p / \Delta q$
- Łojasiewicz exponent estimate
- Bid-ask spread sensitivity

**Certificate format:**

$$
K_7^+ = (\|\nabla V\|, \text{impact}, \text{LS exponent})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{stiff}} = \lambda_7 \cdot \max(0, \epsilon_{\text{stiff}} - \|\nabla_z V\|)^2

$$
:::

:::{prf:definition} Node 7a Specification
:label: def-node7a-bifurcation

**Predicate:** The system is not at a bifurcation point.

$$
P_{7a} : \quad |\det(J_S(z_t))| \ge \epsilon_{\text{bif}},

$$
where $J_S$ is the Jacobian of the market dynamics.

**Market interpretation:** Small perturbations don't cause qualitative regime changes. Near bifurcation, tiny shocks can flip the market between states.

**Observable metrics:**
- Jacobian determinant (estimated via finite differences)
- Eigenvalue clustering near zero
- Sensitivity of regime probabilities to shocks

**Certificate format:**

$$
K_{7a}^+ = (|\det J_S|, \lambda_{\min}(J_S), \text{sensitivity})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{bif}} = \lambda_{7a} \cdot \text{Var}(\nabla_z S_t)

$$
:::

:::{prf:definition} Node 7b Specification
:label: def-node7b-alternatives

**Predicate:** Multiple viable trading strategies exist.

$$
P_{7b} : \quad H(\pi_t) \ge \epsilon_{\text{ent}},

$$
where $H(\pi_t)$ is the entropy of the policy/strategy distribution.

**Market interpretation:** The market isn't locked into a single strategy. Diversity of approaches provides resilience.

**Observable metrics:**
- Policy entropy
- Number of active strategies (above threshold)
- Strategy correlation matrix rank

**Certificate format:**

$$
K_{7b}^+ = (H(\pi_t), \text{active count}, \text{rank}(\Sigma_{\text{strat}}))

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{alt}} = -\lambda_{7b} \cdot H(\pi_t)

$$
:::

:::{prf:definition} Node 7c Specification
:label: def-node7c-newregime

**Predicate:** After a regime switch, the new regime is stable.

$$
P_{7c} : \quad \text{Var}(V(z_{t+1:t+W}) | K_{t+1} = k') \le \sigma_{\text{stable}}^2.

$$

**Market interpretation:** When the market transitions to a new regime, prices stabilize quickly rather than continuing to fluctuate wildly.

**Observable metrics:**
- Post-transition variance
- Time to stabilization
- Return autocorrelation decay

**Certificate format:**

$$
K_{7c}^+ = (\text{Var}_{post}, \tau_{\text{stable}}, \text{AC decay rate})

$$
:::

:::{prf:definition} Node 7d Specification
:label: def-node7d-switching

**Predicate:** The cost of switching strategies is affordable.

$$
P_{7d} : \quad |V(\pi') - V(\pi)| - B_{\text{switch}} \le \text{Budget}_{\text{switch}}.

$$

**Market interpretation:** Transitioning between strategies doesn't consume excessive capital in transaction costs.

**Observable metrics:**
- Estimated rebalancing cost
- Turnover required for strategy change
- Slippage estimate

**Certificate format:**

$$
K_{7d}^+ = (\text{switch cost}, \text{turnover}, \text{slippage})

$$
:::

:::{prf:definition} Node 8 Specification
:label: def-node8-connectivity

**Predicate:** The trading/clearing network is connected.

$$
P_8 : \quad \text{connected}(G_{\text{clearing}}) = \text{True},

$$
where $G_{\text{clearing}}$ is the graph of counterparty relationships.

**Market interpretation:** All market participants can reach each other for settlement. Disconnection indicates fragmentation or clearing failure.

**Observable metrics:**
- Graph connectivity (strongly/weakly connected components)
- Average path length
- Clustering coefficient
- Central counterparty coverage

**Certificate format:**

$$
K_8^+ = (\text{num components}, \text{avg path}, \text{clustering}, \text{CCP coverage})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{conn}} = \lambda_8 \cdot (\text{num components} - 1)

$$
:::

:::{prf:definition} Node 9 Specification
:label: def-node9-tameness

**Predicate:** Pricing functions are smooth; no discontinuities.

$$
P_9 : \quad \|\nabla^2 P(z)\|_{\text{op}} \le \Gamma_{\max},

$$
where $\nabla^2 P$ is the Hessian (gamma) of the pricing function.

**Market interpretation:** Prices respond smoothly to state changes. Jumps in gamma indicate potential for discontinuous repricing.

**Observable metrics:**
- Gamma bounds across instruments
- Convexity measures
- Jump frequency in pricing
- Lipschitz constant estimate

**Certificate format:**

$$
K_9^+ = (\Gamma_{\max}, \text{convexity}, \text{Lip const}, \text{jump count})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{tame}} = \lambda_9 \cdot \|\nabla^2_z S_t\|^2

$$
:::

:::{prf:definition} Node 10 Specification
:label: def-node10-mixing

**Predicate:** The market explores all regimes adequately.

$$
P_{10} : \quad \min_k \hat{p}(K_t = k) \ge p_{\min},

$$
where $\hat{p}$ is the empirical regime frequency.

**Market interpretation:** The market doesn't get stuck in one regime. Adequate exploration means all states have been tested.

**Observable metrics:**
- Regime visit frequencies
- Mixing time estimate
- Ergodic ratio
- Time since last regime $k$ visit

**Certificate format:**

$$
K_{10}^+ = (\min_k \hat{p}(k), \tau_{\text{mix}}, \text{ergodic ratio})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{mix}} = -\lambda_{10} \cdot H(\pi_t)

$$
:::

:::{prf:definition} Node 11 Specification
:label: def-node11-representation

**Predicate:** Regime complexity is within capacity.

$$
P_{11} : \quad H(K_t) \le \log |\mathcal{K}| - \epsilon_{\text{margin}}.

$$

**Market interpretation:** The number of active regimes doesn't exceed what the market can distinguish. Prevents "hallucinated" regimes.

**Observable metrics:**
- Regime entropy $H(K_t)$
- Effective number of regimes: $\exp(H(K_t))$
- Rate utilization: $H(K_t) / \log |\mathcal{K}|$

**Certificate format:**

$$
K_{11}^+ = (H(K_t), |\mathcal{K}|_{\text{eff}}, \text{utilization})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{rep}} = \lambda_{11} \cdot H(q(K|x))

$$
:::

:::{prf:definition} Node 12 Specification
:label: def-node12-oscillation

**Predicate:** No persistent oscillatory patterns (boom-bust cycles).

$$
P_{12} : \quad \|z_t - z_{t-2}\| \ge \epsilon_{\text{osc}} \quad \text{or} \quad \text{FFT}(z_{t-W:t}) \text{ has no dominant frequency}.

$$

**Market interpretation:** Prevents period-2 limit cycles where the market ping-pongs between states.

**Observable metrics:**
- Period-2 autocorrelation
- Spectral peak detection
- Holonomy around closed paths
- Boom-bust indicator

**Certificate format:**

$$
K_{12}^+ = (\text{AC}_2, \text{spectral peak}, \text{holonomy})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{osc}} = \lambda_{12} \cdot \|z_t - z_{t-2}\|^{-2}

$$
:::

:::{prf:definition} Node 13 Specification
:label: def-node13-boundary

**Predicate:** Prices are grounded in observable data.

$$
P_{13} : \quad I(B_t; K_t) > \epsilon_{\text{ground}}.

$$

**Market interpretation:** Internal regime beliefs are supported by external evidence. Prevents "ungrounded inference" where prices disconnect from fundamentals.

**Observable metrics:**
- Mutual information $I(B_t; K_t)$
- Boundary-bulk correlation
- Forecast error relative to boundary data

**Certificate format:**

$$
K_{13}^+ = (I(B_t; K_t), \rho_{\text{boundary-bulk}}, \text{forecast error})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{bound}} = \lambda_{13} \cdot \text{ReLU}(D_{KL}(p_{t+1} \| p_t) - I(B_t; K_t))^2

$$
:::

:::{prf:definition} Node 14 Specification
:label: def-node14-overload

**Predicate:** Data channels are not saturated.

$$
P_{14} : \quad \text{Quote outage rate} \le \epsilon_{\text{outage}} \quad \text{and} \quad \text{Spread}_t \le s_{\max}.

$$

**Market interpretation:** The market infrastructure can handle the data load. Overload causes stale prices and failed executions.

**Observable metrics:**
- Quote outage frequency
- Message queue depth
- Latency spikes
- Spread blowouts

**Certificate format:**

$$
K_{14}^+ = (\text{outage rate}, \text{queue depth}, \text{latency p99}, \text{spread})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{overload}} = \lambda_{14} \cdot \mathbb{I}(\|x_t\| > x_{\max})

$$
:::

:::{prf:definition} Node 15 Specification
:label: def-node15-starvation

**Predicate:** Sufficient data is available for pricing.

$$
P_{15} : \quad \text{SNR}_t \ge \epsilon_{\text{SNR}} \quad \text{and} \quad \text{Volume}_t \ge V_{\min}.

$$

**Market interpretation:** The market has enough trading activity to produce meaningful prices. Starvation = illiquidity.

**Observable metrics:**
- Signal-to-noise ratio
- Trading volume
- Time since last trade
- Quote staleness

**Certificate format:**

$$
K_{15}^+ = (\text{SNR}_t, \text{Volume}_t, \text{staleness})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{starve}} = \lambda_{15} \cdot \text{ReLU}(\epsilon_{\text{SNR}} - \text{SNR}_t)

$$
:::

:::{prf:definition} Node 16 Specification
:label: def-node16-alignment

**Predicate:** Short-term incentives align with long-term objectives.

$$
P_{16} : \quad \|V_{\text{proxy}} - V_{\text{true}}\| \le \epsilon_{\text{align}}.

$$

**Market interpretation:** Trading signals (proxy) actually predict long-term value (true). Misalignment causes agency problems.

**Observable metrics:**
- Proxy-true value correlation
- Funding rate vs. risk signal divergence
- Agent objective alignment score

**Certificate format:**

$$
K_{16}^+ = (\|V_{\text{proxy}} - V_{\text{true}}\|, \rho_{\text{align}}, \text{divergence})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{align}} = \lambda_{16} \cdot \|V_{\text{proxy}} - V_{\text{true}}\|^2

$$
:::

:::{prf:definition} Node 17 Specification
:label: def-node17-lock

**Predicate:** No structural arbitrage exists.

$$
P_{17} : \quad \text{Hom}(\mathbb{H}_{\text{arb}}, \mathbb{H}_{\text{mkt}}) = \emptyset,

$$
where $\mathbb{H}_{\text{arb}}$ is the universal arbitrage pattern.

**Market interpretation:** There is no way to extract guaranteed profit without risk. This is the **final lock** that validates the entire pricing structure.

**Observable metrics:**
- Arbitrage cycle detection (graph algorithms)
- Put-call parity violations
- Cross-market price discrepancies
- Negative basis detection

**Certificate format:**

$$
K_{17}^+ = (\text{arb cycle count} = 0, \text{max parity violation}, \text{basis})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{lock}} = \infty \cdot \mathbb{I}(\text{arbitrage detected})

$$

**Implementation note:** This is a **hard constraint**. Any arbitrage detection immediately invalidates the pricing model.
:::

:::{prf:definition} Node 18: Symmetry Check
:label: def-node18-symmetry

**Predicate:** Pricing is invariant under gauge transformations.

$$
P_{18} : \quad \mathbb{E}_g[D_{KL}(q(K|x) \| q(K|g \cdot x))] \le \epsilon_{\text{sym}}.

$$

**Market interpretation:** Changing numeraire or relabeling assets doesn't change fundamental valuations.
:::

:::{prf:definition} Node 19: Disentanglement Check
:label: def-node19-disentangle

**Predicate:** Macro and micro factors are separated.

$$
P_{19} : \quad \|\text{Cov}(z_{\text{macro}}, z_{\text{micro}})\|_F^2 \le \epsilon_{\text{dis}}.

$$

**Market interpretation:** Regime state (macro) is not contaminated by noise (micro). Clean separation enables robust pricing.
:::

:::{prf:definition} Node 20: Lipschitz Check
:label: def-node20-lipschitz

**Predicate:** All operators have bounded Lipschitz constants.

$$
P_{20} : \quad \max_{\ell} \sigma(W_{\ell}) \le L_{\max},

$$
where $\sigma(W_{\ell})$ is the spectral norm of layer $\ell$.

**Market interpretation:** Small input changes produce small output changes. No explosive sensitivity.
:::

:::{prf:definition} Node 21: Symplectic Check (for Hamiltonian markets)
:label: def-node21-symplectic

**Predicate:** Market dynamics preserve phase space volume.

$$
P_{21} : \quad \|J_S^T J J_S - J\|_F^2 \le \epsilon_{\text{symp}},

$$
where $J$ is the symplectic form.

**Market interpretation:** For markets with Hamiltonian structure (e.g., order book dynamics), volume preservation ensures no information loss.
:::

:::{prf:definition} BarrierSat Specification
:label: def-barrier-sat

**Condition:** Position sizes hit hard limits.

$$
\text{Breached} \iff \exists i : |w_t^i| \ge w_{\max}^i - \epsilon.

$$

**Market context:** Regulatory position limits, exchange limits, risk limits.

**Trigger observables:**
- Position size relative to limit
- Utilization rate: $|w| / w_{\max}$
- Time at limit

**Defense action:**
1. Cap new orders in breached direction
2. Allow only reducing trades
3. Notify risk management

**Re-entry condition:** $|w_t^i| < 0.9 \cdot w_{\max}^i$ for all $i$.
:::

:::{prf:definition} BarrierTypeII Specification
:label: def-barrier-typeii

**Condition:** Volatility temperature exceeds risk temperature.

$$
\text{Breached} \iff \beta_t > \alpha_t,

$$
where $\beta$ = volatility scaling, $\alpha$ = risk perception scaling.

**Market context:** The market is moving faster than participants can assess risk. Classic "vol-of-vol" crisis.

**Trigger observables:**
- Ratio $\beta_t / \alpha_t$
- VIX / realized vol divergence
- Model uncertainty spikes

**Defense action:**
1. Freeze portfolio updates (skip policy step)
2. Widen confidence intervals
3. Increase haircuts/margins

**Re-entry condition:** $\alpha_t > \beta_t + \epsilon_{\text{buffer}}$ for $\tau_{\text{stable}}$ periods.
:::

:::{prf:definition} BarrierGap Specification
:label: def-barrier-gap

**Condition:** Bid-ask spread exceeds normal bounds.

$$
\text{Breached} \iff s_t := p_{\text{ask}} - p_{\text{bid}} > s_{\max}.

$$

**Market context:** Liquidity withdrawal, market maker failure, or extreme uncertainty.

**Trigger observables:**
- Spread as multiple of normal
- Depth at best quotes
- Time since last fill

**Defense action:**
1. Price using mid ± conservative spread
2. Mark positions at worst-case (bid for longs, ask for shorts)
3. Reject market orders; limit orders only

**Re-entry condition:** $s_t < s_{\text{normal}}$ for $\tau_{\text{recover}}$ periods.
:::

:::{prf:definition} BarrierOmin Specification
:label: def-barrier-omin

**Condition:** Price velocity exceeds physical limits.

$$
\text{Breached} \iff \left| \frac{dp_t}{dt} \right| > v_{\max}.

$$

**Market context:** Flash crash, fat-finger error, algorithmic feedback loop.

**Trigger observables:**
- Price change per unit time
- Cumulative intraday move
- Deviation from fair value

**Defense action:**
1. **Circuit breaker:** halt trading for $\tau_{\text{halt}}$
2. Cancel outstanding orders
3. Re-open with auction mechanism

**Re-entry condition:** Successful auction clears within bounds.
:::

:::{prf:definition} BarrierCausal Specification
:label: def-barrier-causal

**Condition:** Prediction horizon exceeds model validity.

$$
\text{Breached} \iff \tau_{\text{forecast}} > \tau_{\text{model validity}}.

$$

**Market context:** Trying to price long-dated instruments with short-term models.

**Trigger observables:**
- Forecast horizon vs. training window
- Out-of-sample error growth
- Model confidence decay

**Defense action:**
1. Shorten effective forecast horizon
2. Increase uncertainty bands exponentially with horizon
3. Use unconditional (prior) distribution for long horizons

**Re-entry condition:** Model retrained or horizon reduced.
:::

:::{prf:definition} BarrierScat Specification
:label: def-barrier-scat

**Condition:** Boundary-bulk coupling collapses.

$$
\text{Breached} \iff I(B_t; K_t) < \epsilon_{\text{min}} \quad \text{or} \quad H(K_t) \approx \log |\mathcal{K}|.

$$

**Market context:** Prices become disconnected from fundamentals; regimes become indistinguishable.

**Trigger observables:**
- Mutual information estimate
- Regime entropy (too high = dispersion)
- Cross-venue price divergence

**Defense action:**
1. Consolidate to primary venue
2. Reduce regime model complexity
3. Use simple (robust) pricing models

**Re-entry condition:** $I(B_t; K_t) > 2 \epsilon_{\text{min}}$ sustained.
:::

:::{prf:definition} BarrierMix Specification
:label: def-barrier-mix

**Condition:** Strategy diversity collapses.

$$
\text{Breached} \iff H(\pi_t) < \epsilon_{\text{ent}}.

$$

**Market context:** Everyone is on the same trade; crowded positions create systemic risk.

**Trigger observables:**
- Strategy entropy
- Correlation of flows across participants
- Short interest concentration

**Defense action:**
1. Inject exploration noise
2. Increase contrarian signal weight
3. Reduce position size in crowded trades

**Re-entry condition:** $H(\pi_t) > 2\epsilon_{\text{ent}}$.
:::

:::{prf:definition} BarrierCap Specification
:label: def-barrier-cap

**Condition:** No available action can improve the situation.

$$
\text{Breached} \iff \forall a \in \mathcal{A} : V(S(z_t, a)) \ge V(z_t).

$$

**Market context:** Stuck in a bad state with no exit. The "doom loop."

**Trigger observables:**
- All actions worsen value
- Controllability Gramian near-singular
- No hedge available at any price

**Defense action:**
1. Accept current position (no trading)
2. Seek external intervention (liquidity injection)
3. Invoke surgery (bailout protocol)

**Re-entry condition:** At least one improving action becomes available.
:::

:::{prf:definition} BarrierVac Specification
:label: def-barrier-vac

**Condition:** System is at or near a bifurcation point.

$$
\text{Breached} \iff |\det(J_S)| < \epsilon_{\text{bif}}.

$$

**Market context:** Small shocks can cause large regime changes. Metastability.

**Trigger observables:**
- Jacobian determinant
- Critical slowing down (autocorrelation spike)
- Variance spike

**Defense action:**
1. Widen all uncertainty bands
2. Prepare for both regime outcomes
3. Reduce leverage preemptively

**Re-entry condition:** $|\det(J_S)| > 2\epsilon_{\text{bif}}$ sustained.
:::

:::{prf:definition} BarrierFreq Specification
:label: def-barrier-freq

**Condition:** Closed-loop system exhibits resonance.

$$
\text{Breached} \iff \|J_{\text{feedback}}\|_{\text{spectral}} \ge 1.

$$

**Market context:** HFT algorithms create feedback loops; quote flickering; mini-flash crashes.

**Trigger observables:**
- Quote update frequency
- Price oscillation frequency
- Spectral power at characteristic frequencies

**Defense action:**
1. Rate-limit order updates
2. Introduce minimum quote lifetime
3. Damping via wider spreads

**Re-entry condition:** Oscillation amplitude decays below threshold.
:::

:::{prf:definition} BarrierEpi Specification
:label: def-barrier-epi

**Condition:** Information channel is saturated.

$$
\text{Breached} \iff I_{\text{received}} > \mathcal{C}_{\text{channel}}.

$$

**Market context:** Too many signals; processing capacity exceeded; system cannot keep up.

**Trigger observables:**
- Message queue depth
- Processing latency
- Dropped message rate

**Defense action:**
1. Throttle incoming data
2. Prioritize critical feeds
3. Use cached/delayed data with discounts

**Re-entry condition:** Queue depth returns to normal.
:::

:::{prf:definition} BarrierAction Specification
:label: def-barrier-action

**Condition:** Desired trade cannot be executed.

$$
\text{Breached} \iff \text{ExecutionCost}(a_t) > \text{Budget}_{\text{exec}}.

$$

**Market context:** Market impact too high; no counterparty available; settlement failure.

**Trigger observables:**
- Estimated market impact
- Fill rate on orders
- Settlement failures

**Defense action:**
1. Queue order for later execution
2. Break into smaller pieces (TWAP/VWAP)
3. Accept partial fill or cancel

**Re-entry condition:** Execution cost falls within budget.
:::

:::{prf:definition} BarrierInput Specification
:label: def-barrier-input

**Condition:** Insufficient market data for pricing.

$$
\text{Breached} \iff \text{Volume}_t < V_{\min} \quad \text{or} \quad \text{Age}(\text{last quote}) > \tau_{\text{stale}}.

$$

**Market context:** Illiquid market; exchange outage; data feed failure.

**Trigger observables:**
- Time since last trade
- Quote staleness
- Data feed status

**Defense action:**
1. Use last known price with uncertainty premium
2. Interpolate from related instruments
3. Mark as "indicative only"

**Re-entry condition:** Fresh data arrives.
:::

:::{prf:definition} BarrierVariety Specification
:label: def-barrier-variety

**Condition:** Hedge dimensionality insufficient.

$$
\text{Breached} \iff \dim(\text{Hedge Space}) < \dim(\text{Risk Space}).

$$

**Market context:** Incomplete market; basis risk cannot be eliminated.

**Trigger observables:**
- Rank of hedge instrument covariance
- Residual risk after best hedge
- Basis spread volatility

**Defense action:**
1. Accept residual (unhedgeable) risk
2. Price in risk premium for incompleteness
3. Reduce exposure to unhedgeable component

**Re-entry condition:** New hedge instruments become available or exposure reduced.
:::

:::{prf:definition} BarrierBode Specification
:label: def-barrier-bode

**Condition:** Reducing one risk increases another (control theory waterbed).

$$
\text{Breached} \iff \int_0^\infty \log |S(j\omega)| d\omega > 0,

$$
where $S$ is the sensitivity function.

**Market context:** Hedging one risk (e.g., delta) amplifies another (e.g., gamma, vega).

**Trigger observables:**
- Cross-Greek sensitivity
- Hedge effectiveness vs. new risk introduction
- Portfolio sensitivity integral

**Defense action:**
1. Balanced multi-objective hedging
2. Accept tradeoff explicitly
3. Use robust (worst-case) hedging

**Re-entry condition:** Sensitivity integral returns to acceptable range.
:::

:::{prf:definition} BarrierLock Specification
:label: def-barrier-lock

**Condition:** Legal or regulatory limit breached.

$$
\text{Breached} \iff x_t \in \mathcal{X}_{\text{forbidden}}.

$$

**Market context:** Exceeding position limits, capital requirements, or other hard regulatory constraints.

**Trigger observables:**
- Regulatory metric values
- Distance to regulatory threshold
- Compliance flags

**Defense action:**
1. **Mandatory:** Cease prohibited activity immediately
2. Report to compliance
3. Execute remediation plan

**Re-entry condition:** Explicit regulatory clearance or metric returns to compliant range.

**Implementation note:** This barrier has **infinite penalty**. Unlike other barriers, there is no discretion—breach requires immediate action.
:::

:::{prf:definition} BarrierLiq Specification
:label: def-barrier-liq

**Condition:** Market liquidity falls below operational threshold.

$$
\text{Breached} \iff \text{Spread}_t > s_{\max} \quad \text{or} \quad \text{Depth}_t < d_{\min}.

$$

**Market context:** Illiquidity crisis where normal market making withdraws. Bid-ask spreads explode, depth vanishes.

**Trigger observables:**
- Bid-ask spread (absolute and relative)
- Order book depth at multiple levels
- Time between trades
- Quote update frequency

**Defense action:**
1. Widen position limits (reduce trading)
2. Switch to interval pricing
3. Mark positions to conservative estimate
4. Notify risk management

**Re-entry condition:** Spread < $0.8 \cdot s_{\max}$ and Depth > $1.2 \cdot d_{\min}$ for sustained period.
:::

:::{prf:definition} BarrierLev Specification
:label: def-barrier-lev

**Condition:** Aggregate leverage exceeds safe bounds.

$$
\text{Breached} \iff \text{Lev}_t := \frac{\text{Gross Exposure}_t}{\text{NAV}_t} > L_{\max}.

$$

**Market context:** Excessive leverage creates forced deleveraging risk. When markets move against leveraged positions, margin calls cascade.

**Trigger observables:**
- Gross leverage ratio
- Net leverage ratio
- Margin utilization
- Funding rate

**Defense action:**
1. Halt new position increases
2. Begin orderly deleveraging
3. Increase margin reserves
4. Monitor counterparty exposure

**Re-entry condition:** Leverage < $0.9 \cdot L_{\max}$ with stable margin.
:::

:::{prf:definition} BarrierRef Specification
:label: def-barrier-ref

**Condition:** Reference price deviates significantly from consensus or shows manipulation signs.

$$
\text{Breached} \iff |p_{\text{ref}} - p_{\text{consensus}}| > \delta_{\text{ref}} \cdot p_{\text{consensus}}.

$$

**Market context:** Oracle attacks, benchmark manipulation, stale reference prices. Critical for derivatives and DeFi protocols that depend on external price feeds.

**Trigger observables:**
- Reference price deviation from median of sources
- Time since last update
- Cross-source disagreement
- Historical volatility of reference

**Defense action:**
1. Reject outlier reference prices
2. Fall back to backup oracle
3. Use time-weighted average (TWAP)
4. Pause operations if no reliable reference

**Re-entry condition:** Reference price within $0.5 \cdot \delta_{\text{ref}}$ of consensus from multiple independent sources.
:::

:::{prf:definition} BarrierDef Specification
:label: def-barrier-def

**Condition:** Credit event or default affects portfolio.

$$
\text{Breached} \iff \exists i : \text{Issuer}_i \in \text{Default State}.

$$

**Market context:** Counterparty default, issuer bankruptcy, credit event (failure to pay, restructuring). Triggers recovery process and cascade risk assessment.

**Trigger observables:**
- Credit event notices
- CDS auction triggers
- Rating downgrades to D
- Missed payment notifications

**Defense action:**
1. Freeze affected positions
2. Assess recovery value
3. Check for cascade exposure
4. Invoke credit event protocols

**Re-entry condition:** Recovery process complete and value realized; no residual exposure to defaulted entity.
:::

## 02_core_pricing/06_asset_classes.md

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

## 03_implementation/04_failure_modes.md

:::{prf:definition} Failure Mode C.E
:label: def-failure-ce

**Mathematical signature:**

$$
\frac{d(\text{Defaults})}{dt} > \lambda_{\text{crit}} \cdot \text{Defaults},

$$
where the default rate exceeds the critical branching factor $\lambda_{\text{crit}} > 1$, producing exponential growth.

**Interpretation:** One default triggers multiple defaults. The system exhibits **supercritical branching**: each failure causes more than one subsequent failure on average.

**Market examples:**
- 2008 financial crisis: Lehman default → money market freeze → bank run cascade
- Sovereign debt contagion: Greece → Portugal → Spain
- Crypto exchange collapse: FTX → Alameda → lending platforms

**Observable signatures:**
- CDS spreads widen exponentially
- Interbank lending freezes
- Correlation spike across unrelated credits
- Recovery rates collapse

**Violated permits:** Node 1 (Solvency), Node 2 (Turnover), BarrierSat

**Intervention class:** SurgCE (Bailout/Recapitalization)
:::

:::{prf:definition} Failure Mode C.D
:label: def-failure-cd

**Mathematical signature:**

$$
\text{HHI} = \sum_j s_j^2 > \text{HHI}_{\text{crit}},

$$
where market share concentration exceeds critical threshold, creating systemic nodes.

**Interpretation:** Wealth/risk concentrates in too few entities. The system becomes **fragile by concentration**—a single node failure is catastrophic.

**Market examples:**
- "Too-big-to-fail" banks (2008)
- Dominant market makers (Knight Capital, Archegos)
- Index concentration (FAANG in S&P 500)
- Stablecoin concentration (USDT dominance)

**Observable signatures:**
- Herfindahl index above threshold
- Single-name CDS dominating index CDS
- Correlation asymmetry (one name moves all)
- Implicit government guarantee priced in

**Violated permits:** Node 3 (Leverage balance), BarrierSat

**Intervention class:** SurgCD (Forced Deleveraging/Breakup)
:::

:::{prf:definition} Failure Mode C.C
:label: def-failure-cc

**Mathematical signature:**

$$
\tau_{\text{trade}} < \tau_{\text{settle}} \implies \text{Zeno regime},

$$
where trade frequency exceeds settlement/clearing capacity.

**Interpretation:** Trading approaches **Zeno's paradox**: infinite trades in finite time, but settlement cannot keep up. The market produces trades faster than it can reconcile them.

**Market examples:**
- Flash crashes from HFT feedback loops
- Quote stuffing overwhelming exchanges
- Latency arbitrage creating phantom liquidity
- MEV extraction in DeFi creating ordering games

**Observable signatures:**
- Message-to-trade ratio exploding
- Quote flickering (sub-millisecond updates)
- Latency variance increasing
- Settlement queue growing

**Violated permits:** Node 10 (Mixing), BarrierFreq

**Intervention class:** SurgCC (Circuit Breakers/Speed Bumps)
:::

:::{prf:definition} Failure Mode T.E
:label: def-failure-te

**Mathematical signature:**

$$
\left|\frac{dp}{dt}\right| > v_{\text{max}} \quad \text{for} \quad \Delta t < \tau_{\text{human}},

$$
where price velocity exceeds any historical precedent on timescales faster than human reaction.

**Interpretation:** The market **falls through itself**—prices move so fast that intermediate liquidity providers cannot react, creating a liquidity vacuum.

**Market examples:**
- May 2010 Flash Crash (Dow -9% in minutes)
- August 2015 ETF dislocations
- October 2016 GBP flash crash
- Crypto flash crashes (20%+ in minutes)

**Observable signatures:**
- Bid-ask spread explosion
- Stub quotes getting hit
- Stop-loss cascade
- Rapid partial recovery

**Violated permits:** Node 8 (Connectivity), BarrierOmin

**Intervention class:** SurgTE (Trading Halt/Auction)
:::

:::{prf:definition} Failure Mode T.D
:label: def-failure-td

**Mathematical signature:**

$$
\text{Volume}_t < \epsilon \cdot \text{Volume}_{\text{normal}} \quad \text{for} \quad t > \tau_{\text{freeze}},

$$
where trading volume collapses below operational threshold.

**Interpretation:** The market **goes silent**—bid and ask exist but nobody trades. Liquidity providers withdraw rather than reveal information or take risk.

**Market examples:**
- 2008 interbank lending freeze
- Emerging market currency crises (no bid)
- Off-the-run Treasury illiquidity
- Distressed credit no-trade zones

**Observable signatures:**
- Zero or near-zero volume
- Bid-ask quotes but no prints
- Price discovery halted
- Stale marks persisting

**Violated permits:** Node 6 (Capacity), Node 9 (Tameness), BarrierInput

**Intervention class:** SurgTD (Market Maker of Last Resort)
:::

:::{prf:definition} Failure Mode T.C
:label: def-failure-tc

**Mathematical signature:**

$$
K(\text{Market State}) > K_{\text{observable}},

$$
where Kolmogorov complexity of the true market state exceeds observable data's descriptive capacity.

**Interpretation:** The market becomes **undecidable locally**—no participant can determine the true state from available information. The topology of dependencies is too complex to model.

**Market examples:**
- CDO-squared pricing collapse (2008)
- Cross-exchange arbitrage with hidden order books
- DeFi composability leading to unforeseen interactions
- Regulatory arbitrage creating invisible risk transfers

**Observable signatures:**
- Model disagreement across participants
- Basis trades failing unexpectedly
- Correlation breakdown in "hedged" positions
- Audit failures revealing hidden exposures

**Violated permits:** Node 11 (Representation), BarrierEpi

**Intervention class:** Simplification mandate, Position transparency
:::

:::{prf:definition} Failure Mode D.E
:label: def-failure-de

**Mathematical signature:**

$$
\ddot{x}_t + \omega^2 x_t \approx 0 \quad \text{with} \quad |\dot{x}_t| \gg \sigma_{\text{normal}},

$$
where the system exhibits undamped oscillation with growing amplitude.

**Interpretation:** The market **oscillates destructively**—leverage builds during boom, then unwinds catastrophically in bust. The dual forces of greed and fear fail to balance.

**Market examples:**
- Dot-com bubble and crash (1995-2002)
- Housing bubble and crash (2003-2009)
- Crypto boom-bust cycles (2017, 2021)
- Commodity supercycles

**Observable signatures:**
- Valuation ratios at historical extremes
- Leverage building during appreciation
- "New paradigm" narratives
- Rapid sentiment reversal

**Violated permits:** Node 12 (Oscillation), Node 7 (Stiffness), BarrierTypeII

**Intervention class:** Counter-cyclical capital buffers, Macroprudential policy
:::

:::{prf:definition} Failure Mode D.D
:label: def-failure-dd

**Mathematical signature:**

$$
\text{Var}(\text{Returns}) \to 0 \quad \text{but} \quad \text{Skew} \to -\infty,

$$
where apparent stability masks extreme tail risk.

**Interpretation:** The market appears **too stable**—volatility selling is profitable, spreads compress, risk seems to have vanished. But the stability is purchased by tail risk accumulation.

**Market examples:**
- Short volatility strategy blowups (XIV, 2018)
- Carry trade unwinds
- Convergence trades gone wrong (LTCM)
- "Picking up pennies in front of a steamroller"

**Observable signatures:**
- Volatility at historical lows
- Option skew extremely negative
- Carry strategies crowded
- Correlation rising despite low vol

**Violated permits:** Node 4 (Scale), Node 5 (Stationarity), BarrierVac

**Intervention class:** Stress testing, Tail risk disclosure
:::

:::{prf:definition} Failure Mode D.C
:label: def-failure-dc

**Mathematical signature:**

$$
\text{Var}_{\text{model}}(X) = \infty \quad \text{or} \quad P(\text{model correct}) < p_{\text{min}},

$$
where no finite-variance model can describe the asset's behavior.

**Interpretation:** The market faces **Knightian uncertainty**—not risk (measurable probability) but genuine uncertainty (unknown unknowns). Pricing is fundamentally undecidable.

**Market examples:**
- Early-stage venture capital (no comparable)
- Pandemic pricing (March 2020)
- Regulatory cliff events (Brexit vote)
- Novel asset classes (first bitcoin pricing)

**Observable signatures:**
- Wide bid-ask spreads
- Option prices not fitting any model
- Expert disagreement
- "Price discovery" language used

**Violated permits:** Node 11 (Representation), BarrierEpi, BarrierCausal

**Intervention class:** Accept uncertainty explicitly, Use interval pricing
:::

:::{prf:definition} Failure Mode S.E
:label: def-failure-se

**Mathematical signature:**

$$
\text{Leverage} \cdot \text{Volatility} > L_{\text{crit}},

$$
where the product of leverage and volatility exceeds the Kelly criterion bound.

**Interpretation:** The market is **overleveraged for its volatility regime**. When volatility spikes, forced deleveraging creates selling pressure that raises volatility further—a positive feedback loop.

**Market examples:**
- LTCM (1998): 25:1 leverage meets volatility spike
- Archegos (2021): Concentrated leveraged positions
- Crypto margin cascades
- VaR-based deleveraging spirals

**Observable signatures:**
- Margin utilization at extremes
- Prime broker exposure concentration
- Funding rates elevated
- Vol-of-vol spiking

**Violated permits:** Node 3 (Leverage), BarrierLev, BarrierSat

**Intervention class:** SurgSE (Regulatory Capital Injection/Margin Holiday)
:::

:::{prf:definition} Failure Mode S.D
:label: def-failure-sd

**Mathematical signature:**

$$
\frac{d\sigma}{dS} \to 0 \quad \text{while} \quad \sigma \to \sigma_{\min},

$$
where volatility becomes unresponsive and artificially suppressed.

**Interpretation:** The market loses its **natural warning system**. Central bank intervention or structural changes suppress volatility, but risk accumulates invisibly until sudden release.

**Market examples:**
- "Greenspan put" → "Fed put" vol suppression
- European sovereign spreads pre-2010 (uniform despite divergent fundamentals)
- Vol targeting strategies forcing vol down
- Implicit government guarantees flattening credit spreads

**Observable signatures:**
- Realized vol << implied vol for extended period
- VIX term structure in contango
- Correlation with vol selling flows
- Sudden regime breaks when policy changes

**Violated permits:** Node 7 (Stiffness), Node 5 (Stationarity), BarrierVac

**Intervention class:** SurgSD (Volatility Injection/Policy Taper)
:::

:::{prf:definition} Failure Mode S.C
:label: def-failure-sc

**Mathematical signature:**

$$
\theta_t \neq \theta_{t+\Delta t} \quad \text{on calibration timescale},

$$
where model parameters drift faster than models can be recalibrated.

**Interpretation:** The market exhibits **non-stationarity**. Models calibrated yesterday fail today because the underlying generating process has changed, but the change is not observable in real-time.

**Market examples:**
- Factor investing regime shifts
- Correlation breakdown during stress
- Term structure model failure
- Machine learning model decay

**Observable signatures:**
- Calibration residuals growing
- Backtest-to-live performance gap
- Hedging effectiveness declining
- P&L attribution unexplained

**Violated permits:** Node 5 (Stationarity), Node 11 (Representation), BarrierCausal

**Intervention class:** Ensemble models, Regime-aware calibration
:::

:::{prf:definition} Failure Mode B.E
:label: def-failure-be

**Mathematical signature:**

$$
\|\text{Input}_t - \text{Input}_{t-1}\|_2 > \delta_{\text{shock}},

$$
where an exogenous input exceeds the market's absorption capacity.

**Interpretation:** The market receives an **external shock** that exceeds its capacity to absorb. The shock originates outside the financial system (war, pandemic, natural disaster) but propagates through it.

**Market examples:**
- COVID-19 market crash (March 2020)
- 9/11 market closure and reopening
- Oil embargo (1973)
- Fukushima disaster → Japanese equities

**Observable signatures:**
- News-driven gap opens
- Cross-asset correlation spikes to 1
- Safe haven flows dominate
- Normal trading patterns suspended

**Violated permits:** BarrierInput, Node 15 (Starvation), $\mathrm{Bound}_\partial$

**Intervention class:** Coordinated policy response, Market closure if needed
:::

:::{prf:definition} Failure Mode B.D
:label: def-failure-bd

**Mathematical signature:**

$$
\text{Inflow}_t < \text{Required Flow}_t \quad \text{for} \quad t > \tau_{\text{starve}},

$$
where the market receives insufficient capital/data/liquidity to function.

**Interpretation:** The market **starves**—not from a shock but from gradual withdrawal. Capital leaves, liquidity providers exit, data feeds degrade. Death by a thousand cuts.

**Market examples:**
- Capital flight from emerging markets
- Dealer balance sheet constraints post-regulation
- Repo market strains (September 2019)
- Stablecoin redemption pressure

**Observable signatures:**
- Persistent outflows
- Market depth declining
- Bid-ask spreads widening gradually
- Prime broker credit tightening

**Violated permits:** Node 15 (Starvation), BarrierInput, Node 6 (Capacity)

**Intervention class:** SurgBD (Liquidity Injection/Quantitative Easing)
:::

:::{prf:definition} Failure Mode B.C
:label: def-failure-bc

**Mathematical signature:**

$$
\nabla_a U_{\text{agent}} \cdot \nabla_a U_{\text{principal}} < 0,

$$
where agent incentives point opposite to principal/social welfare.

**Interpretation:** The market exhibits **principal-agent failure**. Market participants optimize their own objective, but this optimization harms the system or their clients.

**Market examples:**
- Rating agency conflicts (paid by issuers)
- Fund manager incentives (AUM vs. returns)
- Market maker payment for order flow
- Auditor independence failures

**Observable signatures:**
- Persistent mispricings in one direction
- Information asymmetry exploitation
- Governance scandals
- Regulatory enforcement actions

**Violated permits:** Node 16 (Alignment), BarrierMix, BarrierCap

**Intervention class:** SurgBC (Incentive Realignment/Regulation)
:::

:::{prf:proposition} Failure Cascade Paths
:label: prop-failure-cascade

Common cascade sequences:
1. **S.E → C.E**: Leverage crisis → Default cascade (LTCM, 2008)
2. **T.E → C.C**: Flash crash → HFT instability (May 2010)
3. **B.E → T.D**: External shock → Frozen market (COVID March 2020)
4. **D.E → S.E → C.E**: Bubble → Overleveraging → Cascade (housing crisis)
5. **B.C → D.D → S.E**: Misalignment → Hidden risk → Leverage blowup (2008 CDOs)
:::

:::{prf:proposition} Conservation of Risk
:label: prop-risk-conservation

Interventions in one cell shift risk to adjacent cells:

$$
\sum_{i,j} R_{ij} = \text{const} \quad \text{(up to dissipation)},

$$
where $R_{ij}$ is risk intensity in cell $(i,j)$.

**Examples:**
- Bailing out C.E (default cascade) increases moral hazard in C.D (too-big-to-fail)
- Suppressing S.D (flat vol) accumulates pressure for D.E (boom-bust)
- Circuit breakers (T.E) can concentrate risk into T.D (frozen market)
:::

:::{prf:definition} Severity Levels
:label: def-severity-levels

Each failure mode is classified by severity:

**Level 1 (Warning):** Metrics approaching threshold. Defense: monitoring upgrade.

**Level 2 (Alert):** Threshold breached but contained. Defense: position reduction, hedging.

**Level 3 (Crisis):** Multiple barriers breached, cascade risk. Defense: surgery invocation.

**Level 4 (Systemic):** Cross-market contagion, regulatory intervention required. Defense: coordinated policy response.

Severity formula:

$$
\text{Severity} = \max_i \left( \frac{\text{Metric}_i - \text{Threshold}_i}{\text{Threshold}_i} \right) \times \text{Cascade Factor}.

$$
:::

## 03_implementation/05_surgery_contracts.md

:::{prf:definition} Surgery Contract Template
:label: def-surgery-template

A surgery contract $\text{Surg}_X$ consists of:
1. **Trigger condition:** When the surgery is invoked
2. **Preconditions:** What must be true before surgery
3. **Actions:** The intervention steps
4. **Postconditions:** What must be true after surgery
5. **Re-entry certificate:** Proof that the system can resume normal operation
6. **Side effects:** Risk shifts to other failure modes
:::

:::{prf:definition} SurgCE Contract
:label: def-surg-ce

**Trigger:** Failure mode C.E (Default Cascade) at Level 3+.

**Preconditions:**
- Default cascade confirmed (branching factor $> 1$)
- Systemic importance threshold exceeded
- Market-based solutions exhausted

**Actions:**
1. Identify systemically important defaulting entities
2. Inject capital via equity purchase, loan guarantee, or direct transfer
3. Provide liquidity backstop to counterparties
4. Impose conditions (management change, dividend restrictions)
5. Establish resolution framework for orderly unwinding

**Postconditions:**
- Branching factor $< 1$ (cascade halted)
- Key entities solvent
- Interbank/funding markets functional

**Re-entry certificate:** $K^{\text{re}}_{\text{CE}}$ = (Solvency restored, Funding normalized, Capital plan approved)

**Side effects:**
- Increases C.D risk (moral hazard, too-big-to-fail reinforced)
- May trigger B.C (agency misalignment from implicit guarantees)
- Fiscal cost creates boundary constraint

**Historical examples:** TARP (2008), EU bank recapitalizations, FTX/Alameda aftermath discussions
:::

:::{prf:definition} SurgCD Contract
:label: def-surg-cd

**Trigger:** Failure mode C.D (Too-Big-to-Fail) at Level 3+.

**Preconditions:**
- Concentration metrics (HHI) exceed critical threshold
- Single entity failure would trigger C.E cascade
- Voluntary deleveraging insufficient

**Actions:**
1. Mandate capital raise or asset sales
2. Enforce position limits
3. If necessary, mandate structural separation (Glass-Steagall style)
4. Create resolution plan ("living will")
5. Increase capital requirements for systemic entities

**Postconditions:**
- HHI below critical threshold
- No single entity systemically critical
- Credible resolution plans in place

**Re-entry certificate:** $K^{\text{re}}_{\text{CD}}$ = (Concentration reduced, Resolution plans filed, Capital buffers adequate)

**Side effects:**
- May reduce market efficiency (economies of scale lost)
- Could trigger T.D (reduced market making capacity)
- Regulatory arbitrage risk

**Historical examples:** Volcker Rule, Dodd-Frank systemic designations, UK ring-fencing
:::

:::{prf:definition} SurgCC Contract
:label: def-surg-cc

**Trigger:** Failure mode C.C (HFT Instability) at Level 2+.

**Preconditions:**
- Trade frequency exceeding settlement capacity
- Quote-to-trade ratio explosive
- Latency arbitrage creating instability

**Actions:**
1. Implement market-wide circuit breakers (price limits)
2. Introduce minimum resting times for quotes
3. Apply speed bumps (intentional delays)
4. Enforce batch auctions at intervals
5. Increase messaging fees for excessive quoting

**Postconditions:**
- Trade frequency within settlement capacity
- Quote quality improved (quote-to-trade ratio normalized)
- Price continuity restored

**Re-entry certificate:** $K^{\text{re}}_{\text{CC}}$ = (Trading velocity bounded, Message rates stable, Settlement current)

**Side effects:**
- May reduce liquidity (HFT provides some liquidity)
- Could shift activity to less-regulated venues
- May increase T.D risk (slower price discovery)

**Historical examples:** NYSE circuit breakers, IEX speed bump, EU minimum tick sizes
:::

:::{prf:definition} SurgTE Contract
:label: def-surg-te

**Trigger:** Failure mode T.E (Flash Crash) at Level 3+.

**Preconditions:**
- Price move exceeds velocity threshold
- Liquidity vacuum detected
- Normal market making suspended

**Actions:**
1. Halt trading immediately
2. Cancel clearly erroneous trades
3. Accumulate orders during halt
4. Conduct single-price auction to reopen
5. Gradually restore continuous trading

**Postconditions:**
- Price within reasonable range
- Bid-ask spread normalized
- Liquidity providers re-engaged

**Re-entry certificate:** $K^{\text{re}}_{\text{TE}}$ = (Auction completed, Price discovery validated, Market makers present)

**Side effects:**
- Halts transfer risk to correlated markets
- May trigger T.D if halt extends
- Uncertainty during halt period

**Historical examples:** May 2010 trade cancellations, single-stock circuit breakers, crypto exchange halts
:::

:::{prf:definition} SurgTD Contract
:label: def-surg-td

**Trigger:** Failure mode T.D (Frozen Market) at Level 3+.

**Preconditions:**
- Trading volume collapsed
- Normal market makers withdrawn
- Price discovery halted

**Actions:**
1. Central entity (central bank, exchange, or designated institution) posts two-sided quotes
2. Provide backstop liquidity at wide but finite spreads
3. Accept losses as cost of market function
4. Gradually narrow spreads as private liquidity returns
5. Exit once private market making resumes

**Postconditions:**
- Two-sided market exists
- Trades can execute
- Price discovery resumed

**Re-entry certificate:** $K^{\text{re}}_{\text{TD}}$ = (Bid and ask present, Volume above minimum, Private makers returning)

**Side effects:**
- Moral hazard for market makers
- Central entity takes mark-to-market risk
- May crowd out private liquidity if not exited promptly

**Historical examples:** Fed commercial paper facility (2008), ECB bond purchases, Treasury buybacks during stress
:::

:::{prf:definition} SurgSE Contract
:label: def-surg-se

**Trigger:** Failure mode S.E (Supercritical Leverage) at Level 3+.

**Preconditions:**
- Leverage × volatility product exceeds critical bound
- Forced deleveraging creating feedback loop
- Margin calls cascading

**Actions:**
1. Temporarily reduce margin requirements
2. Extend margin call deadlines
3. Provide emergency credit lines to clearinghouses
4. Coordinate orderly position reduction
5. Increase margin requirements gradually once stable

**Postconditions:**
- Leverage × volatility below critical bound
- Forced selling halted
- Clearinghouse solvent

**Re-entry certificate:** $K^{\text{re}}_{\text{SE}}$ = (Margin calls current, Leverage reduced, Volatility subsiding)

**Side effects:**
- Moral hazard (leverage may rebuild faster)
- Regulatory credibility affected
- May shift risk to B.D (liquidity starvation if credit tight)

**Historical examples:** Exchange margin reductions during stress, Fed lending to clearinghouses, coordinated bank credit lines
:::

:::{prf:definition} SurgSD Contract
:label: def-surg-sd

**Trigger:** Failure mode S.D (Flat Volatility) at Level 2+ (preventive).

**Preconditions:**
- Volatility suppressed for extended period
- Policy intervention identified as cause
- Tail risk accumulating invisibly

**Actions:**
1. Signal policy normalization (taper talk)
2. Reduce intervention gradually
3. Allow volatility to rise naturally
4. Stress test for higher vol regime
5. Monitor for D.E (boom-bust) emergence

**Postconditions:**
- Volatility at historically normal levels
- Risk pricing restored
- Policy intervention reduced

**Re-entry certificate:** $K^{\text{re}}_{\text{SD}}$ = (Volatility normalized, Policy stance communicated, Stress tests passed)

**Side effects:**
- May trigger D.E or S.E as suppressed volatility releases
- Market adjustment costs
- Communication risk (taper tantrum)

**Historical examples:** Fed taper (2013), ECB policy normalization attempts, BOJ yield curve control adjustments
:::

:::{prf:definition} SurgBC Contract
:label: def-surg-bc

**Trigger:** Failure mode B.C (Agency Misalignment) at Level 3+.

**Preconditions:**
- Systematic misalignment identified
- Market-based correction insufficient
- Harm to principals/system documented

**Actions:**
1. Regulatory intervention (new rules, enforcement)
2. Mandate disclosure of conflicts
3. Restructure compensation (clawbacks, deferred pay)
4. Strengthen fiduciary duties
5. Create alignment mechanisms (skin in the game requirements)

**Postconditions:**
- Agent incentives aligned with principals
- Conflicts disclosed and managed
- Enforcement mechanism operational

**Re-entry certificate:** $K^{\text{re}}_{\text{BC}}$ = (Rules in effect, Compliance verified, Alignment metrics improved)

**Side effects:**
- Compliance costs
- May reduce market participation (activity shifts elsewhere)
- Regulatory capture risk

**Historical examples:** Dodd-Frank compensation rules, MiFID II inducement rules, rating agency regulation
:::

:::{prf:definition} Multi-Surgery Protocol
:label: def-multi-surgery

**Priority ordering:**
1. SurgTE (halt trading if flash crash—immediate safety)
2. SurgCE (stop default cascade—systemic risk)
3. SurgSE (margin relief—prevent cascade amplification)
4. SurgTD (restore liquidity—enable price discovery)
5. SurgCC, SurgCD, SurgSD, SurgBC (structural—can wait for stability)

**Coordination rules:**
- Never invoke SurgSD during active crisis (would amplify)
- SurgCE and SurgCD conflict (bailout vs. breakup)—choose based on urgency
- SurgTE and SurgTD are sequential (halt → maker of last resort → reopen)

**Exit sequencing:**
- Surgeries should exit in reverse order of invocation
- Each exit requires the re-entry certificate of later surgeries
- Central bank facilities should be last to exit
:::

## 03_implementation/06_metatheorems.md

:::{prf:theorem} Market Consistency Theorem (MKT-Consistency)
:label: thm-mkt-consistency

Self-consistent prices arise whenever payoffs depend on prices (funding, margin, endogenous
constraints, feedback through order flow). In that setting, “pricing” is a **fixed point problem**.

Let $K \subset \mathbb{R}^n$ be a non-empty, compact, convex set of admissible price vectors (after
imposing conventions and barrier constraints). Let $\mathcal{M}:K \to K$ be a continuous pricing
operator, for example

$$
\mathcal{M}(p) := \mathbb{E}^{\mathbb{Q}}_t\!\left[M_{t,t+1}\,X_{t+1}(p)\right],

$$
where $M_{t,t+1}>0$ is an SDF and $X_{t+1}(p)$ is the (possibly price-dependent) payoff.

Then $\mathcal{M}$ admits at least one fixed point $p^*\in K$ such that $\mathcal{M}(p^*)=p^*$.

Moreover, if $\mathcal{M}$ is a contraction on $K$, the fixed point is unique and iteration
$p^{(n+1)}=\mathcal{M}(p^{(n)})$ converges to $p^*$.

*Proof.* Existence follows from Brouwer's fixed point theorem applied to the continuous map
$\mathcal{M}:K\to K$. The contraction statement is Banach's fixed point theorem. $\square$
:::

:::{prf:corollary} Permit interpretation
:label: cor-permit-interpretation

A pricing model satisfies MKT-Consistency iff all Sieve gates pass. Gate failures indicate
inconsistency.
:::

:::{prf:lemma} Contraction for Unique Fixed Point
:label: lem-contraction-unique

If the market operator $\mathcal{M}$ is a **contraction** with Lipschitz constant $L < 1$:

$$
\|\mathcal{M}(p) - \mathcal{M}(q)\| \le L \|p - q\|,

$$
then the fixed point $p^*$ is **unique** and iteration converges geometrically: $\|p^{(n)} - p^*\| \le L^n \|p^{(0)} - p^*\|$.
:::

:::{prf:remark} Constructive Fixed Point
:label: rem-constructive-fp

The fixed point is **constructively obtained** via iteration:

$$
p^{(n+1)} = \mathcal{M}(p^{(n)}),

$$
converging under contraction conditions. The rate of convergence indicates **pricing stability**:
- Fast convergence → stable, well-identified prices
- Slow convergence → fragile, sensitive to perturbations
- Non-convergence → inconsistent pricing (barrier breach)
:::

:::{prf:theorem} Market Exclusion Theorem (MKT-Exclusion)
:label: thm-mkt-exclusion

Assume a one-period (or discrete-time finite-dimensional) market where the set $C$ of terminal
payoffs attainable from zero initial cost is a convex cone that is closed in the relevant
topology. Then the following are equivalent:

1. (**No-arbitrage**) $C \cap L^0_+ = \{0\}$.
2. (**Separation**) There exists a strictly positive linear pricing functional $\pi$ such that
   $\pi(X) \le 0$ for all $X \in C$ and $\pi(Y) > 0$ for all $Y \in L^0_+ \setminus \{0\}$.

In particular, $\pi$ induces a strictly positive state-price density / stochastic discount factor.

*Proof sketch.* If $C$ is closed and $C \cap L^0_+ = \{0\}$, the Hahn–Banach / separating hyperplane
theorem yields a continuous linear functional that separates $C$ from the positive cone. Strict
positivity of the separator gives a state-price density. The converse is immediate. For
continuous-time semimartingale models, replace the separation step by NFLVR and apply FTAP
{cite}`delbaen1994ftap`. $\square$
:::

:::{prf:remark} Permit interpretation
:label: rem-mkt-exclusion-permits

Operationally, “no-arbitrage” is enforced by a bundle of Sieve permits: solvency/funding/liquidity
barriers rule out strategies that are profitable only in the frictionless abstraction; grounding
and tameness gates rule out model-generated mispricings that are not executable.
:::

:::{prf:remark} Basis trades as near-arbitrages
:label: rem-basis-near-arbitrage

Basis trades are near-arbitrages in the frictionless limit: the price discrepancy is real, but
the strategy is not admissible once funding, liquidity, and settlement constraints are included.
In Sieve terms, the trade is blocked by a barrier rather than refuting the pricing functional.
:::

:::{prf:theorem} Market Trichotomy Theorem (MKT-Trichotomy)
:label: thm-mkt-trichotomy

The Sieve is a runtime protocol: given a state estimate and boundary data, it decides whether a
pricing statement is admissible. The trichotomy is therefore a statement about the **protocol
output**, not a claim that real markets have only three dynamical behaviors.

Fix a set of gate permits and barrier permits (the Market Sieve). Define the protocol outcome at a
time $t$ as follows:

1. **Equilibrium (E):** no barrier is breached and the pricing operator residual is small,
   $\|\mathcal{M}(p_t)-p_t\|\le \varepsilon_{\mathrm{fp}}$, with all critical gates certified.
2. **Crisis (C):** at least one *operational* barrier is breached (solvency, funding, liquidity,
   settlement), requiring a surgery before prices are accepted.
3. **Horizon (H):** an *epistemic* barrier is breached (information overload) or a critical gate
   cannot be certified with available boundary data/capacity; pricing must be interval-valued or
   the model must abstain.

Assuming permit completeness (Axiom {prf:ref}`axiom-permit-completeness`) and a fixed partition of
barriers into operational vs. epistemic, exactly one of $\{E,C,H\}$ is returned.

*Proof.* By construction, the protocol first checks barrier certificates. If any operational
barrier is breached it returns $C$; if any epistemic barrier is breached it returns $H$. If no
barrier is breached, it checks the critical gates and the fixed-point residual; if they pass it
returns $E$, otherwise it returns $H$. The cases are mutually exclusive by the case split. $\square$
:::

:::{prf:remark} Crisis as Temporary State
:label: rem-crisis-temporary

Crisis (C) is **transient by design**—surgery contracts exist to return the system to an admissible
region. Horizon (H) is “absorbing” for point-valued pricing at the current resolution, but it may
resolve after (i) new information arrives, (ii) time passes and the horizon shortens, or (iii) the
model class is upgraded.
:::

:::{prf:theorem} Market Equivariance Theorem (MKT-Equivariance)
:label: thm-mkt-equivariance

Pricing statements should be **gauge-covariant**: they must not depend on arbitrary choices of unit
system, asset relabeling, or numéraire.

Let $P_t(X_T)$ be a valuation functional represented by an SDF:

$$
P_t(X_T) = \mathbb{E}_t\!\left[M_{t,T}\,X_T\right], \qquad M_{t,T} > 0.

$$

Then the following invariances hold for admissible gauge transformations:

1. (**Scale / unit covariance**) For $\lambda>0$, $P_t(\lambda X_T)=\lambda P_t(X_T)$.
2. (**Relabeling**) For any asset permutation $\sigma$, relabeling payoffs relabels prices.
3. (**Numéraire invariance**) Let $N_t>0$ be a strictly positive traded numéraire. Then there
   exists an equivalent measure $\mathbb{Q}^N$ such that for all payoffs $X_T$,

   $$
   P_t(X_T) = N_t\,\mathbb{E}_t^{\mathbb{Q}^N}\!\left[\frac{X_T}{N_T}\right].

   $$

*Proof sketch.* The first two items follow from linearity of expectation. For the third, take the
money-market account $B_t$ as baseline numéraire, and define

$$
\left.\frac{d\mathbb{Q}^N}{d\mathbb{Q}^B}\right|_{\mathcal{F}_T}
=
\frac{(N_T/B_T)}{(N_0/B_0)}.

$$
Then Bayes' rule gives $P_t/N_t=\mathbb{E}_t^{\mathbb{Q}^N}[X_T/N_T]$. $\square$

**Permit interpretation:** scale/unit covariance is checked by the scale gates; numéraire changes
are allowed by YES$^\sim$ permits; relabeling invariance is a symmetry sanity check in diagnostics.
:::

:::{prf:remark} Relative value as symmetry diagnostics
:label: rem-symmetry-relative-value

Observed deviations from gauge-covariant pricing (e.g., cross-currency basis, on/off-the-run
spreads, dual-listed share discrepancies) are *relative value signals*. In the full model they are
not “free arbitrage”: the Sieve typically identifies the missing certificate as a funding,
liquidity, or settlement barrier.
:::

:::{prf:theorem} Horizon Limit Theorem (MKT-HorizonLimit)
:label: thm-mkt-horizon

The market has an **effective horizon** beyond which point-valued pricing claims cannot be
certified at a chosen tolerance: uncertainty amplification eventually dominates available
information.

Assume the priced quantity is sensitive in the sense that, within a given regime, there is an
effective Lyapunov exponent $\lambda_{\max}>0$ such that two admissible boundary-consistent initial
conditions can generate trajectories $p_t,\tilde p_t$ with

$$
\|p_t-\tilde{p}_t\| \gtrsim \epsilon_{\text{in}}\,e^{\lambda_{\max} t},

$$
where $\epsilon_{\text{in}}>0$ lower-bounds irreducible input uncertainty.

Let $\epsilon_{\text{out}}>0$ be the tolerated pricing error. Then no point-valued pricing claim at
tolerance $\epsilon_{\text{out}}$ can be certified for horizons $t>T^*$, where

$$
T^* = \frac{1}{\lambda_{\max}} \ln\left(\frac{\epsilon_{\text{out}}}{\epsilon_{\text{in}}}\right).

$$

Beyond $T^*$, the correct output is an interval (or a Horizon/abstain certificate), not a single
price.

*Proof.* Two boundary-consistent initial states within $\epsilon_{\text{in}}$ cannot be reliably
distinguished. Under the assumed sensitivity, their induced price predictions at horizon $t$ differ
by at least $\epsilon_{\text{in}}e^{\lambda_{\max} t}$. If this exceeds $\epsilon_{\text{out}}$, no
algorithm can guarantee a point estimate within tolerance for all admissible states. $\square$
:::

:::{prf:remark} Permit interpretation
:label: rem-horizon-permit-interpretation

The Horizon outcome is monitored by representation/grounding gates and by the epistemic barrier
(`BarrierEpi` in the Market Sieve). In finance terms, it triggers widened uncertainty sets, stress
scenarios, or model abstention.
:::

:::{prf:lemma} Horizon estimation in practice
:label: lem-horizon-practice

As an order-of-magnitude illustration, take:
- $\epsilon_{\text{in}}=10^{-3}$ (0.1% input uncertainty)
- $\epsilon_{\text{out}}=10^{-1}$ (10% tolerated error)
- $\lambda_{\max}=0.02$ per day

Then:

$$
T^* \approx \frac{1}{0.02} \ln(100) \approx 230 \text{ trading days} \approx 1 \text{ year}.

$$

This is not a universal constant; it is a diagnostic derived from the chosen tolerance and the
effective instability of the regime.
:::

:::{prf:remark} Practical implications
:label: rem-horizon-practical

The Horizon Limit implies:
- Long-term valuation is fundamentally **interval-valued**, not point-valued
- Long-tenor derivatives require **model uncertainty quantification**
- Planning under long horizons should use **scenario families**, not point forecasts
- Any model claiming arbitrary precision must ship explicit uncertainty certificates (or it is out
  of envelope)
:::

:::{prf:remark} Scope of the metatheorems
:label: rem-metatheorem-scope

The five metatheorems are the top-level constraints this volume treats as first-class:
fixed-point consistency, no-arbitrage separation, Sieve routing (E/C/H), gauge covariance, and an
explicit epistemic horizon. They are not asserted as a logically complete characterization of all
possible finance models; they delimit the modeling envelope we can audit and operationalize.
:::

## 03_implementation/07_algorithmic_pricing.md

:::{prf:definition} Price Complexity
:label: def-price-complexity

The **Kolmogorov complexity** of a price series is defined on a finite encoding (for example:
tick-binned log-returns, or a fixed-precision discretization). For such an encoding
$x_{1:T}$, the Kolmogorov complexity is the length of the shortest program that generates it:

$$
K(x_{1:T}) = \min_{\text{program } \pi} \{|\pi| : U(\pi) = x_{1:T}\},

$$
where $U$ is a universal Turing machine and $|\pi|$ is program length in bits.

**Interpretation:** $K(p)$ measures the **intrinsic information content** of prices—how much description is needed to specify them exactly.
:::

:::{prf:proposition} Complexity vs. entropy (typical sequences)
:label: prop-compress-bounds

For any probabilistic model $P$ over encoded price strings, there exists a constant $c_P$ such
that:

$$
K(x_{1:T}) \le -\log P(x_{1:T}) + c_P.

$$
In particular, if $x_{1:T}$ is typical under a stationary source with entropy rate $H$, then
$K(x_{1:T}) \approx T H$ up to sublinear terms.

**Market phases by complexity:**
- **Crystal phase (efficient):** $K(p) \approx K(\text{random})$ — prices are incompressible
- **Liquid phase (predictable):** $K(p) < K(\text{random}) - \epsilon$ — structure exists
- **Gas phase (chaotic):** $K(p) \approx K(\text{random})$ but structure is emergent
:::

:::{prf:definition} Market Complexity Phases
:label: def-complexity-phases

**Crystal Phase (Efficient Markets):**
- Prices reflect all available information instantly
- $K(\text{price} | \text{info}) \approx 0$
- No profitable prediction possible
- Corresponds to: Liquid, competitive markets with low barriers

**Liquid Phase (Arbitrageable Markets):**
- Prices reflect most information with friction
- $0 < K(\text{price} | \text{info}) < K_{\text{barrier}}$
- Prediction profitable after costs
- Corresponds to: Markets with execution costs, information asymmetry

**Gas Phase (Random/Chaotic Markets):**
- Prices disconnected from information
- $K(\text{price} | \text{info}) \approx K(\text{price})$
- No systematic relationship to fundamentals
- Corresponds to: Crisis, bubble, or nascent markets

**Phase boundaries:**
- Crystal ↔ Liquid: Execution cost threshold
- Liquid ↔ Gas: Information capacity threshold
- Gas ↔ Crystal: Crisis resolution / market maturation
:::

:::{prf:proposition} Phase Detection
:label: prop-phase-detection

Phase can be detected via the **compression ratio**:

$$
\rho = \frac{K(p_{1:T})}{T \cdot H_0},

$$
where $H_0$ is the entropy of uniform prices.

- $\rho \approx 1$: Crystal or Gas phase
- $\rho < 1 - \epsilon$: Liquid phase (exploitable structure)

Distinguishing Crystal from Gas requires **external information tests**.
:::

:::{prf:definition} Price Discovery Depth
:label: def-price-depth

The **computational depth** of a price is the time required to compute it from fundamentals:

$$
\text{Depth}(p) = \min_{\pi : U(\pi) = p} \{\text{runtime}(\pi)\}.

$$

**Interpretation:** Deep prices require complex computation; shallow prices are easily derived.
:::

:::{prf:remark} Depth–complexity tradeoff (heuristic)
:label: rem-depth-complexity

In market prediction, there is often an empirical tradeoff: low-description (“simple”) strategies
may require deep computation to discover and validate, while high-complexity signals can be
computationally shallow but hard to specify and robustify. This motivates monitoring both
compression/structure and runtime/latency budgets.
:::

:::{prf:remark} Levin-style prediction bound (heuristic)
:label: rem-levin-market

As a modeling heuristic, one may treat “surprisal” as being controlled by algorithmic complexity
via a Solomonoff/Levin-style prior:

$$
\mathbb{P}(\text{price series } p) \propto 2^{-K(p)},

$$
and then combine it with the thermodynamic cost of computation (Landauer) to motivate a
compute–accuracy tradeoff of the form:

$$
\mathbb{E}[\text{error}] \geq \frac{k_B T_{\text{market}}}{E_{\text{computation}}},

$$
where $T_{\text{market}}$ is market temperature (risk tolerance) and $E_{\text{computation}}$ is energy expended on prediction.

**Implications:**
- **No free lunch:** Better prediction requires more computation/energy.
- **Thermodynamic consistency:** Market efficiency has physical foundations.
- **HFT limits:** Speed requires energy; there's a speed-energy tradeoff.

This is not a theorem about real markets; it is a closure that links the “bounded rationality”
stance to operational compute budgets.
:::

:::{prf:definition} Algorithmic Efficiency
:label: def-alg-efficiency

A market is **algorithmically efficient** at level $\epsilon$ if:

$$
K(p_{t+1} | p_{1:t}, \text{public info}) > K(p_{t+1}) - \epsilon.

$$

**Interpretation:** Future prices are nearly as complex given history as they are unconditionally—history provides minimal compression.
:::

:::{prf:proposition} Efficiency Hierarchy
:label: prop-efficiency-hierarchy

Market efficiency levels map to algorithmic notions:

1. **Weak efficiency:** $K(p_{t+1} | p_{1:t}) \approx K(p_{t+1})$ — price history uninformative
2. **Semi-strong efficiency:** $K(p_{t+1} | p_{1:t}, \text{public}) \approx K(p_{t+1})$ — public info reflected
3. **Strong efficiency:** $K(p_{t+1} | \text{all info}) \approx K(p_{t+1})$ — all info reflected

**Permit mapping:**
- Node 11 (Representation) tracks deviations from efficiency
- BarrierEpi triggers when complexity analysis shows exploitable structure
- Liquid phase markets are semi-strong efficient with friction
:::

:::{prf:definition} Proof-Carrying Prices
:label: def-proof-price

A **proof-carrying price** is a tuple $(p, \pi)$ where:
- $p$ is the price
- $\pi$ is a certificate/proof that $p$ satisfies required properties

The verification function $V(p, \pi) \in \{\text{ACCEPT}, \text{REJECT}\}$ runs in polynomial time.
:::

:::{prf:proposition} Sieve as Proof System
:label: prop-sieve-proof

The [Market Sieve](../02_core_pricing/03_market_sieve.md) implements a proof system for prices:
- **Prover:** Market dynamics generating prices
- **Verifier:** Sieve gates checking permits
- **Certificate:** Gate passage record $K = (K_1, \ldots, K_{21})$
- **Soundness:** Invalid prices fail some gate (completeness of gates)
- **Completeness:** Valid prices pass all gates

The certificate size is:

$$
|K| = O(\text{number of gates} \times \log(\text{precision})) = O(21 \times 64) = O(1344 \text{ bits}).

$$
:::

:::{prf:definition} Sieve Computational Complexity
:label: def-sieve-complexity

The computational cost of running the Market Sieve:

**Per-gate costs:**
- Node 1-2 (Conservation): $O(n)$ where $n$ = number of positions
- Node 3-5 (Duality): $O(n)$ for leverage/scale checks
- Node 6-7 (Geometry): $O(n \log n)$ for capacity/stiffness
- Node 8-10 (Topology): $O(n^2)$ worst case for connectivity (typically $O(n \log n)$ with sparse structure)
- Node 11-12 (Epistemics): $O(m)$ where $m$ = model parameters
- Node 13-17 (Extended): $O(n)$ each

**Total Sieve cost:**

$$
T_{\text{Sieve}} = O(n^2 + m),

$$
with typical sparsity allowing $O(n \log n + m)$.

**Barrier monitoring:**
- Per barrier: $O(1)$ to $O(n)$ depending on barrier type
- 20 barriers: $O(n)$ total

**Full pricing loop overhead:**

$$
\text{Overhead} = \frac{T_{\text{Sieve}}}{T_{\text{Pricing}}} \approx 2-5\%,

$$
for typical portfolios with $n \sim 1000$ positions.
:::

## 04_applications/03_calibration.md

:::{prf:remark} Calibration Philosophy
:label: rem-calibration-philosophy

Thresholds should be set to achieve:
1. **High recall for catastrophic events** (never miss a crisis)
2. **Acceptable precision for warnings** (tolerate some false alarms)
3. **Regime-dependent adjustment** (tighter in stress, looser in calm)
4. **Asset-class specificity** (equities differ from bonds)
:::

:::{prf:definition} Regime-Adjusted Threshold
:label: def-regime-threshold

For base threshold $\tau_0$ and regime $K$, the adjusted threshold is:

$$
\tau_K = \tau_0 \cdot \phi_K,

$$
where $\phi_K$ is the regime adjustment factor:

| Regime | $\phi_K$ | Interpretation |
|--------|----------|----------------|
| Risk-On | 1.2 | Looser thresholds |
| Neutral | 1.0 | Base thresholds |
| Risk-Off | 0.8 | Tighter thresholds |
| Crisis | 0.5 | Much tighter |
| Recovery | 0.9 | Slightly tight |
:::

## 04_applications/04_risk_attribution.md

:::{prf:definition} Risk Attribution Decomposition
:label: def-risk-attribution

Total portfolio risk $\sigma^2$ decomposes hierarchically:

$$
\sigma^2 = \underbrace{\sigma^2_{\text{sys}}}_{\text{Systematic}} + \underbrace{\sigma^2_{\text{idio}}}_{\text{Idiosyncratic}} + \underbrace{\sigma^2_{\text{regime}}}_{\text{Regime}} + \underbrace{\sigma^2_{\text{barrier}}}_{\text{Barrier}}.

$$
:::

:::{prf:definition} Gate Risk Contribution
:label: def-gate-risk

For gate $i$ with current value $v_i$ and threshold $\tau_i$, the gate risk contribution is:

$$
R_i = w_i \cdot \max\left(0, \frac{v_i - \tau_i^{\text{warn}}}{\tau_i^{\text{crit}} - \tau_i^{\text{warn}}}\right)^2,

$$
where $w_i$ is the weight representing potential loss if gate $i$ fails.
:::

:::{prf:definition} Failure Mode Risk
:label: def-fm-risk

For failure mode $F$ with probability $p_F$ and severity $s_F$:

$$
R_F = p_F \cdot s_F \cdot \mathbb{E}[\text{Loss} \mid F],

$$
where $p_F$ is estimated from gate/barrier states.
:::

## 04_applications/05_backtesting.md

:::{prf:remark} Backtesting Goals
:label: rem-backtest-goals

The backtesting framework aims to:
1. **Validate gate thresholds** against historical crises
2. **Measure false positive/negative rates** for barriers
3. **Test surgery effectiveness** on historical interventions
4. **Calibrate regime detection** accuracy
5. **Estimate economic value** of the Sieve
:::

:::{prf:definition} Sieve Performance Metrics
:label: def-backtest-metrics

For a set of $N$ historical events, the Sieve is evaluated on:

1. **Detection rate:** $DR = \frac{\text{Events where any gate/barrier triggered}}{\text{Total events}}$

2. **Early warning rate:** $EW = \frac{\text{Events with trigger} \ge 5 \text{ days before peak}}{\text{Total events}}$

3. **False positive rate:** $FPR = \frac{\text{Triggers in non-event periods}}{\text{Total non-event days}}$

4. **Failure mode accuracy:** $FMA = \frac{\text{Correctly identified failure modes}}{\text{Actual failure modes}}$

5. **Economic value:** $EV = \sum_{\text{events}} (\text{Loss avoided by early exit}) - (\text{Opportunity cost of false exits})$
:::

## 05_geometric_theory/01_capital_capacity.md

:::{prf:definition} No-arbitrage capacity bound
:label: def-no-arbitrage-capacity-bound

Consider the market interface (order book, quote stream) as an information channel. The **market
capacity** $C_{\text{mkt}}$ bounds the information content of any sustainable position:

$$
I_{\text{position}} \le C_{\text{mkt}},

$$
where:
- $I_{\text{position}}$ is the information content of the portfolio position (bits needed to specify the strategy),
- $C_{\text{mkt}}$ is the effective information capacity of the market interface (market depth, quote frequency).

Units: $[I_{\text{position}}] = [C_{\text{mkt}}] = \text{nat}$.

**Consequence:** Positions with information content exceeding market capacity are unsustainable. Strategies that violate this bound incur ungrounded exposure risk.
:::

:::{prf:definition} Capital information density
:label: def-capital-information-density

Let $\rho(w, t)$ denote the probability density of portfolio weights $w \in \mathcal{W}$ at time
$t$. The **capital information density** is:

$$
\rho_I(w, t) := -\rho(w, t) \log \rho(w, t) + \frac{1}{2}\rho(w, t) \log\det G(w),

$$
where $G(w)$ is the Ruppeiner market metric ({prf:ref}`def-ruppeiner-market`).

*Interpretation:* The first term is the Shannon entropy density; the second is the geometric correction accounting for risk-induced volume distortion.
:::

:::{prf:definition} Market depth as an area law
:label: def-market-depth-area-law

The market capacity follows an **area law**:

$$
C_{\text{mkt}} = \frac{1}{\eta_{\text{tick}}} \cdot \text{Depth}(\partial\mathcal{W}),

$$
where:
- $\text{Depth}(\partial\mathcal{W})$ is the aggregate market depth at the trading boundary,
- $\eta_{\text{tick}}$ is the minimum price tick per unit information (market microstructure parameter).
:::

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

:::{prf:definition} Risk-energy tensor (capacity-law form)
:label: def-capacity-risk-energy-tensor

This is the capacity-law specialization of the market risk-energy tensor
({prf:ref}`def-risk-energy-tensor`).

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

:::{prf:lemma} Metric Divergence at Saturation
:label: lem-metric-divergence-saturation

As the capacity saturation ratio $\nu_{\text{cap}} \to 1$, the effective metric $G_{\text{eff}}$ diverges:

$$
G_{\text{eff}}(w) = \frac{G_0(w)}{(1 - \nu_{\text{cap}})^2} \to \infty

$$

*Interpretation:* Near saturation, infinitesimal position changes require infinite "effort" in the metric—an effective horizon analogous to the Schwarzschild radius. This prevents over-concentration through geometric mechanics rather than hard limits.
:::

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

:::{prf:conjecture} Area Law for Market Capacity (holographic closure)
:label: conj-area-law-market-capacity

The maximum information $I_{\max}$ that can be sustainably maintained in a portfolio region $\Omega \subseteq \mathcal{W}$ is bounded by the "area" of its market interface:

$$
I_{\max}(\Omega) \le \frac{\text{Area}_G(\partial\Omega)}{4\ell_{\text{tick}}^2}

$$
where:
- $\text{Area}_G(\partial\Omega) = \oint_{\partial\Omega} d^{N-1}\sigma_G$ is the $(N-1)$-dimensional boundary area in the Ruppeiner metric,
- $\ell_{\text{tick}}$ is the minimum tick resolution (Planck-like scale for markets).

*Motivation.* This is a physics-inspired closure: the boundary area measures executable information
bandwidth (order book interface), while tick resolution sets the minimum distinguishable unit of
price information. It should be treated as an empirical scaling hypothesis, not as a theorem of
standard finance.
:::

## 05_geometric_theory/02_wfr_transport.md

:::{prf:definition} Wasserstein-2 distance
:label: def-wasserstein-2

Let $(\mathcal{W}, d_G)$ be the portfolio space equipped with a ground metric $d_G$ induced by the
risk geometry. For probability measures $\mu,\nu$ on $\mathcal{W}$, the Wasserstein-2 distance is:

$$
W_2^2(\mu,\nu) := \inf_{\pi \in \Pi(\mu,\nu)} \int_{\mathcal{W}\times\mathcal{W}} d_G(w,w')^2\,d\pi(w,w'),

$$
where $\Pi(\mu,\nu)$ is the set of couplings of $\mu$ and $\nu$.
:::

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
- $G$ is the Ruppeiner risk metric on portfolio space ({prf:ref}`def-ruppeiner-market`)

*Forward reference (Boundary Conditions).* {prf:ref}`def-dirichlet-bc-price-quotes` and
{prf:ref}`def-neumann-bc-order-submission` specify how boundary conditions on $\partial\mathcal{W}$
(order book interface) constrain WFR dynamics: **trading hours** impose execution constraints;
**overnight** allows unconstrained internal rebalancing.
:::

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

:::{prf:definition} WFR portfolio dynamics
:label: def-wfr-portfolio-dynamics

The policy outputs a generalized velocity $(v, r)$ to minimize WFR path length to the target
allocation (goal portfolio).
:::

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

:::{prf:definition} WFR Consistency Loss
:label: def-wfr-consistency-loss-portfolio

The cone-space representation linearizes WFR locally. From $\partial_s \rho = \rho r - \nabla \cdot (\rho v)$ and $u = \sqrt{\rho}$, we have $\partial_s u = \frac{\rho r - \nabla \cdot (\rho v)}{2\sqrt{\rho}}$. Define the consistency loss:

$$
\mathcal{L}_{\mathrm{WFR}} = \left\| \sqrt{\rho_{t+1}} - \sqrt{\rho_t} - \frac{\Delta t}{2\sqrt{\rho_t}}\left(\rho_t r_t - \nabla \cdot (\rho_t v_t)\right) \right\|_{L^2}^2

$$

This penalizes deviations from the unbalanced continuity equation.
:::

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

## 05_geometric_theory/03_price_discovery.md

:::{prf:definition} Price Manifold Boundary and Interior
:label: def-price-manifold-boundary-interior

- **Interior ($\mathring{\mathcal{Z}}$):** The latent price space where mid-prices evolve. $z=0$ is maximum entropy (uninformed prior).
- **Boundary ($\partial\mathcal{Z}$):** The market interface where quotes are revealed. $|z| \to 1$ is minimum entropy (perfect price revelation).
- **Price Discovery:** The process of moving from interior to boundary.
:::

:::{prf:definition} Information Content of Price
:label: def-information-content-price

The hyperbolic distance from origin represents information content:

$$
I_{\text{price}}(z) := d_{\mathbb{D}}(0, z) = 2 \operatorname{artanh}(|z|).

$$

This measures how much information the market has revealed about the fundamental price.
:::

:::{prf:definition} Entropic drift in markets
:label: def-entropic-drift-markets

In the absence of order flow, prices experience an **entropic drift** toward revelation:

$$
\dot{r} = \frac{1 - r^2}{2},

$$
which integrates to:

$$
r(\tau) = \tanh(\tau/2).

$$
:::

:::{prf:definition} Market maker control field
:label: def-market-maker-control-field

The market maker (or informed trader) provides a **control field**:

$$
u_{\text{mm}}(z) = G^{-1}(z) \cdot \mathbb{E}_{a \sim \pi}[a],

$$
which breaks rotational symmetry at the origin, selecting a preferred direction for price evolution.
:::

:::{prf:axiom} Bid-ask decoupling
:label: ax-bid-ask-decoupling

The state decomposition $(K, z_n, z_{\text{tex}})$ maps to:
- **Interior (price process):** Mid-price trajectory $z(\tau)$ evolves on the pricing manifold.
- **Boundary (microstructure):** Bid-ask spread $z_{\text{tex}}$ is sampled at the interface.

$$
\frac{\partial}{\partial z_{\text{tex}}} \left[ \dot{z}, \lambda_{\text{jump}}, u_\pi \right] = 0

$$

**Consequence:** Mid-price dynamics are independent of microstructure noise. Spread fluctuations decouple from the fundamental price discovery process.
:::

:::{prf:definition} Microstructure noise distribution
:label: def-microstructure-noise-distribution

At the market interface:

$$
z_{\text{tex}} \sim \mathcal{N}(0, \Sigma_{\text{spread}}(z)),

$$
where:

$$
\Sigma_{\text{spread}}(z) = \sigma_{\text{spread}}^2 \cdot G^{-1}(z).

$$

**Scaling:** Near the origin (wide spreads), microstructure noise variance is large. Near the boundary (tight spreads), noise is suppressed by the metric.
:::

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

## 05_geometric_theory/04_equations_of_motion.md

:::{prf:definition} Position inertia tensor
:label: def-position-inertia-tensor

The **position inertia** is the Ruppeiner risk metric:

$$
\mathbf{M}(w) := G(w).

$$
:::

:::{prf:definition} Portfolio geodesic SDE
:label: def-portfolio-geodesic-sde

The portfolio weights $w^k$ evolve according to:

$$
dw^k = \underbrace{\left( -G^{kj}\partial_j \Phi_{\text{risk}} + u_\pi^k \right)}_{\text{Drift (signal + policy)}} ds - \underbrace{\Gamma^k_{ij}\dot{w}^i \dot{w}^j\,ds}_{\text{Correlation correction}} + \underbrace{\sqrt{2T_c}\,(G^{-1/2})^{kj}\,dW^j_s}_{\text{Market noise}},

$$
where:
- $\Phi_{\text{risk}}$ is the risk-adjusted return potential,
- $u_\pi^k$ is the alpha signal (policy control),
- $\Gamma^k_{ij}$ are Christoffel symbols of the risk metric (correlation structure),
- $T_c$ is the market temperature (volatility scaling).

**Three-Force Decomposition:**
1. **Return gradient:** $-G^{-1}\nabla\Phi_{\text{risk}}$ — move toward high risk-adjusted returns.
2. **Alpha signal:** $u_\pi$ — policy-induced trades (momentum, value, etc.).
3. **Correlation correction:** $-\Gamma(\dot{w},\dot{w})$ — adjusts for cross-asset dependencies.
:::

:::{prf:definition} Regime jump intensity
:label: def-regime-jump-intensity

The intensity of jumping from regime $i$ to regime $j$ is:

$$
\lambda_{i \to j}(w) = \lambda_0 \cdot \exp\left(\beta \cdot \left( V_j(w) - V_i(w) - c_{\text{switch}} \right) \right),

$$
where:
- $V_i, V_j$ are regime-specific value functions,
- $c_{\text{switch}}$ is the regime transition cost,
- $\beta$ is inverse temperature (sharpness).

**Interpretation:** Regime transitions occur when $V_j(w) - V_i(w) > c_{\text{switch}}$, with rate exponentially increasing in the value differential.
:::

:::{prf:definition} Risk-adjusted return potential
:label: def-risk-adjusted-return-potential

The unified potential is:

$$
\Phi_{\text{risk}}(w, K) = \alpha \cdot U(w) + (1 - \alpha) \cdot V_{\text{alpha}}(w, K) + \gamma_{\text{risk}} \cdot \Psi_{\text{risk}}(w),

$$
where:
- $U(w)$ is the information potential (spread compression),
- $V_{\text{alpha}}(w, K)$ is the alpha signal (expected returns),
- $\Psi_{\text{risk}}(w) = \frac{1}{2}\text{tr}(T_{ij} G^{ij})$ is risk concentration.

| $\alpha$ | Behavior | Strategy Type |
|----------|----------|---------------|
| $\alpha = 1$ | Pure liquidity provision | Market making |
| $\alpha = 0$ | Pure alpha capture | Directional trading |
| $\alpha = 0.5$ | Balanced | Mixed strategy |
:::

:::{prf:definition} Gate43 Specification
:label: def-gate43-specification

**Predicate:** Portfolio trajectory satisfies geodesic equation.

$$
P_{43} : \quad \left\|\ddot{w}^k + \Gamma^k_{ij}\dot{w}^i\dot{w}^j + G^{kj}\partial_j\Phi_{\text{risk}} - u_\pi^k\right\|_G \le \epsilon_{\text{geo}},

$$
where $\epsilon_{\text{geo}}$ is the geodesic tolerance.

**Market interpretation:** The portfolio is following a minimum-risk path given the current return landscape and alpha signals.

**Observable metrics:**
- Geodesic residual $\|\text{EOM violation}\|_G$
- Path curvature $\kappa = \|\ddot{w} + \Gamma(\dot{w},\dot{w})\|$
- Alpha contribution $\|u_\pi\|$
- Return gradient magnitude $\|G^{-1}\nabla\Phi\|$

**Certificate format:**

$$
K_{43}^+ = (\text{residual}, \kappa, \|u_\pi\|, \|\nabla\Phi\|, \text{regime})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{geo}} = \lambda_{43} \cdot \left\|\ddot{w} + \Gamma(\dot{w},\dot{w}) + G^{-1}\nabla\Phi - u_\pi\right\|_G^2

$$
:::

## 05_geometric_theory/05_market_interface.md

:::{prf:definition} Symplectic market interface
:label: def-symplectic-market-interface

The market interface is a symplectic manifold $(\partial\mathcal{W}, \omega)$ with:
- $q \in \mathcal{Q}$ is the **price coordinate** (mark-to-market values),
- $p \in T^*_q\mathcal{Q}$ is the **flow coordinate** (order flow, trading velocity).

The symplectic form is:

$$
\omega = \sum_{i=1}^n dq^i \wedge dp_i.

$$
:::

:::{prf:definition} Dirichlet boundary condition (price quotes)
:label: def-dirichlet-bc-price-quotes

Market prices impose position-clamping:

$$
q_{\partial}^{\text{quote}}(t) = q_{\text{mid}}(t),

$$
where $q_{\text{mid}}$ is the observable mid-price. This clamps the **configuration** of the portfolio.
:::

:::{prf:definition} Neumann boundary condition (order submission)
:label: def-neumann-bc-order-submission

Trading imposes flux-clamping:

$$
\nabla_n q \cdot \mathbf{n} \big|_{\partial\mathcal{W}} = j_{\text{trade}}(p, t),

$$
where $j_{\text{trade}}$ is the order flow determined by the trading strategy.
:::

:::{prf:definition} Trading cycle phases
:label: def-trading-cycle-phases

| Phase | Process | Information Flow | Entropy Change |
|-------|---------|------------------|----------------|
| **I. Observation** | Price compression | Market data → portfolio state | $\Delta S < 0$ |
| **II. Simulation** | Internal risk analysis | No external exchange | $\Delta S = 0$ (isentropic) |
| **III. Execution** | Order expansion | Trading signal → order flow | $\Delta S > 0$ |
:::

:::{prf:assumption} Market Carnot efficiency (information-to-profit bound)
:label: asm-market-carnot-efficiency

The efficiency of converting market information to trading profits is bounded:

$$
\eta = \frac{I(A_t; K_t)}{I(X_t; K_t)} \le 1 - \frac{T_{\text{exec}}}{T_{\text{obs}}},

$$
where $T_{\text{exec}}$ and $T_{\text{obs}}$ are effective temperatures at execution and observation interfaces.
:::

:::{prf:definition} Active trading mode
:label: def-active-trading-mode

$$
\rho_{\partial}^{\text{quote}}(w, t) = \delta(w - w_{\text{target}}(t)) \quad \text{(Dirichlet)},

$$
$$
\nabla_n \rho \cdot \mathbf{n} = j_{\text{trade}}(u_\pi) \quad \text{(Neumann)}.

$$
:::

:::{prf:definition} Closed-system simulation mode
:label: def-closed-system-simulation-mode

$$
\nabla_n \rho \cdot \mathbf{n} = 0 \quad \text{(Reflective)}.

$$
The system is closed—no trading, pure risk simulation.
:::

:::{prf:definition} Market context space
:label: def-market-context-space

The context $c \in \mathcal{C}$ determines the trading objective:

| Task | Context $c$ | Output | Potential $\Phi_{\text{eff}}$ |
|------|-------------|--------|-------------------------------|
| **Alpha Capture** | Signal space | Trade direction | $V_{\text{alpha}}(w, K)$ |
| **Risk Management** | Risk budget | Hedge ratio | $-\log p(\text{safe}|w)$ |
| **Execution** | Target portfolio | Order sequence | $-\log p(\text{fill}|w, \text{target})$ |
:::

:::{prf:definition} Gate44 Specification
:label: def-gate44-specification

**Predicate:** Quote and trade boundary conditions are symplectically compatible.

$$
P_{44} : \quad \|\omega(j_{\text{quote}}, j_{\text{trade}})\| \le \epsilon_{\text{symp}},

$$
where $\omega$ is the symplectic form and $j_{\text{quote}}, j_{\text{trade}}$ are the quote and trade flows.

**Market interpretation:** The trading activity is consistent with the quoted market structure.

**Observable metrics:**
- Symplectic residual $\|\omega(j_q, j_t)\|$
- Price impact $\|q_{\text{exec}} - q_{\text{quote}}\|$
- Fill rate $\text{P}(\text{fill} | \text{order})$
- Spread capture $(\text{bid} + \text{ask})/2 - q_{\text{mid}}$

**Certificate format:**

$$
K_{44}^+ = (\|\omega\|, \text{impact}, \text{fill rate}, \text{spread})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{symp}} = \lambda_{44} \cdot \|\omega(j_{\text{quote}}, j_{\text{trade}})\|^2

$$
:::

## 05_geometric_theory/06_pricing_kernel.md

:::{prf:definition} Cash flow as source term
:label: def-cash-flow-source-term

The cash flow stream (dividends, coupons) acts as a scalar source:

$$
\sigma_{\text{cf}}(t, w) = \sum_{t' < t} \text{CF}_{t'} \cdot \delta(t - t') \cdot \delta(w - w_{t'}),

$$
where $\text{CF}_t$ is the cash flow at time $t$.
:::

:::{prf:theorem} DCF as Helmholtz equation
:label: thm-dcf-helmholtz-equation

The net present value $V(w)$ satisfies the **screened Poisson equation**:

$$
\boxed{-\Delta_G V(w) + \kappa^2 V(w) = \rho_{\text{cf}}(w)}

$$
where:
- $\Delta_G$ is the Laplace-Beltrami operator on the risk manifold,
- $\kappa = \lambda / c_{\text{info}}$ with $\lambda = -\ln(\gamma)/\Delta t$ (natural units: $\kappa = -\ln\gamma$) is the screening mass (discount factor),
- $\rho_{\text{cf}}$ is the cash flow density.

**Proof sketch:** The Bellman equation $V(w) = \mathbb{E}[\text{CF} + \gamma V(w')]$ approaches the Helmholtz PDE in the continuous limit. $\square$
:::

:::{prf:corollary} Investment horizon as screening length
:label: cor-investment-horizon-screening-length

$$
\ell_{\text{horizon}} = \frac{1}{\kappa} = \frac{c_{\text{info}} \Delta t}{-\ln\gamma}.

$$
:::

:::{prf:definition} Value-metric feedback
:label: def-value-metric-feedback

High-value-curvature regions induce metric distortion:

$$
\tilde{G}_{ij}(w) = \Omega^2(w) \cdot G_{ij}(w),

$$
where:

$$
\Omega(w) = 1 + \alpha_{\text{conf}} \cdot \|\nabla^2_G V(w)\|_{\text{op}}.

$$
:::

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

## 05_geometric_theory/07_sector_classification.md

:::{prf:definition} Sector partition
:label: def-sector-partition

Let $\mathcal{Y} = \{\text{Tech}, \text{Finance}, \text{Healthcare}, \ldots\}$ be sector labels. The
sector induces a partition of the regime atlas:

$$
\mathcal{A}_y := \{k \in \mathcal{K} : P(\text{Sector}=y \mid K=k) > 1 - \epsilon_{\text{purity}}\}.

$$
:::

:::{prf:definition} Sector risk premium potential
:label: def-sector-risk-premium-potential

$$
V_{\text{sector}}(w, K) := -\beta_{\text{sector}} \log P(\text{Sector}=y \mid K) + V_{\text{base}}(w, K),

$$
where:
- $P(\text{Sector}=y \mid K)$ is the sector probability given regime,
- $\beta_{\text{sector}}$ is the sector temperature (concentration preference).
:::

:::{prf:definition} Sector allocation basin
:label: def-sector-allocation-basin

The **allocation basin** for sector $y$ is:

$$
\mathcal{B}_y := \{w \in \mathcal{W} : \lim_{t \to \infty} \phi_t(w) \in \mathcal{A}_y\},

$$
where $\phi_t$ is the flow of $\dot{w} = -G^{-1}(w)\nabla V_y(w)$.
:::

:::{prf:proposition} Sector rotation as relaxation
:label: prop-sector-rotation-relaxation

Consider the zero-temperature sector rotation dynamics:

$$
\dot{w} = -G^{-1}(w)\nabla V_y(w).

$$
If $V_y$ is $C^1$ and the trajectory remains in a compact sublevel set, then $V_y(w(s))$ decreases
monotonically and the trajectory converges to the set of critical points of $V_y$ (LaSalle
invariance principle). In particular, the dynamics relax to a local minimum in the corresponding
allocation basin.

With a small-temperature overdamped Langevin perturbation, trajectories do not converge almost
surely but spend most time near the minima of $V_y$; this is the sense in which “sector rotation”
is a relaxation process.
:::

:::{prf:definition} Sector-modulated regime transition
:label: def-sector-modulated-regime-transition

Modify regime transition rates:

$$
\lambda_{i \to j}^{\text{sector}} := \lambda_{i \to j}^{(0)} \cdot \exp\left(-\gamma_{\text{sep}} \cdot D_{\text{sector}}(i, j)\right),

$$
where $D_{\text{sector}}(i, j) = \mathbb{I}[\text{Sector}(i) \neq \text{Sector}(j)]$.
:::

:::{prf:definition} Sector purity loss
:label: def-sector-purity-loss

$$
\mathcal{L}_{\text{purity}} = \sum_{k=1}^{N_c} P(K=k) \cdot H(\text{Sector} \mid K=k).

$$
:::

:::{prf:definition} Sector rotation loss
:label: def-sector-rotation-loss

$$
\mathcal{L}_{\text{sector}} = \mathcal{L}_{\text{route}} + \lambda_{\text{pur}} \mathcal{L}_{\text{purity}} + \lambda_{\text{bal}} \mathcal{L}_{\text{balance}} + \lambda_{\text{met}} \mathcal{L}_{\text{metric}}.

$$
:::

:::{prf:definition} Gate46 Specification
:label: def-gate46-specification

**Predicate:** Regimes are sector-pure (low entropy).

$$
P_{46} : \quad H(\text{Sector} \mid K) \le \epsilon_{\text{purity}},

$$
where $H(\text{Sector} \mid K)$ is the conditional entropy of sector given regime.

**Market interpretation:** Each regime corresponds cleanly to a sector—no mixed-sector regimes.

**Observable metrics:**
- Sector purity $1 - H(\text{Sector} \mid K)$
- Cross-sector exposure per regime
- Sector membership probabilities

**Certificate format:**

$$
K_{46}^+ = (H(\text{Sector} \mid K), \text{purity}, \text{cross-exposure})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{purity}} = \lambda_{46} \cdot H(\text{Sector} \mid K)

$$
:::

:::{prf:definition} Gate47 Specification
:label: def-gate47-specification

**Predicate:** Sectors are metrically separated.

$$
P_{47} : \quad \min_{y_1 \neq y_2} d_{\text{WFR}}(\mathcal{A}_{y_1}, \mathcal{A}_{y_2}) \ge \epsilon_{\text{sep}},

$$
where $d_{\text{WFR}}$ is the WFR distance between sector allocation basins.

**Market interpretation:** Different sectors are geometrically far apart—no easy transitions.

**Observable metrics:**
- Minimum inter-sector WFR distance
- Average intra-sector distance
- Sector separation ratio

**Certificate format:**

$$
K_{47}^+ = (d_{\min}, d_{\text{intra}}, \text{sep ratio})

$$

**Loss contribution:**

$$
\mathcal{L}_{\text{sep}} = \lambda_{47} \cdot \max(0, \epsilon_{\text{sep}} - d_{\min})^2

$$
:::
