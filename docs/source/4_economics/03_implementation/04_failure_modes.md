# Market Failure Mode Taxonomy

Markets fail in structured ways. The **Failure Mode Taxonomy** classifies all market pathologies into a 3×5 grid indexed by:
- **Structural domain** (row): Conservation (C), Topology (T), Duality (D), Symmetry (S), Boundary (B)
- **Failure type** (column): Explosive (E), Degenerative (D), Computational (C)

This taxonomy is **complete** in the sense that every market failure routes through exactly one cell, and **conserved** in the sense that interventions in one cell can shift risk to adjacent cells but cannot eliminate it.

## Taxonomy Overview

```{list-table} Market Failure Mode Grid
:header-rows: 1
:name: failure-mode-grid

* - Domain
  - Explosive (E)
  - Degenerative (D)
  - Computational (C)
* - Conservation (C)
  - C.E: Default Cascade
  - C.D: Too-Big-to-Fail
  - C.C: HFT Instability
* - Topology (T)
  - T.E: Flash Crash
  - T.D: Frozen Market
  - T.C: Complexity Crisis
* - Duality (D)
  - D.E: Boom-Bust Cycle
  - D.D: Dispersion Success
  - D.C: Fundamental Uncertainty
* - Symmetry (S)
  - S.E: Supercritical Leverage
  - S.D: Flat Volatility
  - S.C: Parameter Drift
* - Boundary (B)
  - B.E: External Shock
  - B.D: Liquidity Starvation
  - B.C: Agency Misalignment
```

## Conservation Failures (Row C)

Conservation failures violate **solvency, turnover, or resource constraints**—the market's equivalent of mass-energy conservation.

---

### C.E: Default Cascade (Explosive Conservation)

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

---

### C.D: Too-Big-to-Fail (Degenerative Conservation)

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

---

### C.C: HFT Instability (Computational Conservation)

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

---

## Topology Failures (Row T)

Topology failures affect **market connectivity, reachability, and structural integrity**—the graph of who can trade with whom.

---

### T.E: Flash Crash (Explosive Topology)

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

---

### T.D: Frozen Market (Degenerative Topology)

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

---

### T.C: Complexity Crisis (Computational Topology)

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

---

## Duality Failures (Row D)

Duality failures involve **oscillation, balance, and the observer-system relationship**—the market's feedback loops.

---

### D.E: Boom-Bust Cycle (Explosive Duality)

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

---

### D.D: Dispersion Success (Degenerative Duality)

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

---

### D.C: Fundamental Uncertainty (Computational Duality)

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

---

## Symmetry Failures (Row S)

Symmetry failures involve **scaling, leverage, and invariance properties**—the market's dimensional consistency.

---

### S.E: Supercritical Leverage (Explosive Symmetry)

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

---

### S.D: Flat Volatility Trap (Degenerative Symmetry)

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

---

### S.C: Parameter Drift (Computational Symmetry)

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

---

## Boundary Failures (Row B)

Boundary failures involve **external interactions, data flows, and agent incentives**—the market's interface with the outside world.

---

### B.E: External Shock (Explosive Boundary)

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

---

### B.D: Liquidity Starvation (Degenerative Boundary)

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

---

### B.C: Agency Misalignment (Computational Boundary)

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

---

## Failure Mode Interactions

Failures rarely occur in isolation. The taxonomy reveals **interaction patterns**:

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

## Failure Mode Detection

Each failure mode has a **signature vector** of observable metrics:

```{list-table} Failure Mode Detection Signatures
:header-rows: 1
:name: failure-detection

* - Mode
  - Primary Signal
  - Secondary Signals
  - Lead Time
* - C.E
  - CDS spread acceleration
  - Interbank rate, correlation spike
  - Hours to days
* - C.D
  - HHI index
  - Single-name dominance
  - Months to years
* - C.C
  - Message/trade ratio
  - Latency variance
  - Milliseconds
* - T.E
  - Price velocity
  - Spread explosion
  - Seconds
* - T.D
  - Volume collapse
  - Quote staleness
  - Hours to days
* - T.C
  - Model disagreement
  - Basis unexplained
  - Days to weeks
* - D.E
  - Valuation extremes
  - Sentiment indicators
  - Months
* - D.D
  - Vol at lows, skew extreme
  - Carry crowding
  - Weeks to months
* - D.C
  - Spread width
  - Expert variance
  - Persistent
* - S.E
  - Leverage × vol product
  - Margin utilization
  - Days
* - S.D
  - Vol suppression duration
  - VIX term structure
  - Months
* - S.C
  - Calibration residuals
  - Hedge effectiveness
  - Days to weeks
* - B.E
  - News shock
  - Gap opens
  - Immediate
* - B.D
  - Flow data
  - Depth metrics
  - Weeks
* - B.C
  - Governance signals
  - Regulatory actions
  - Months to years
```

## Failure Mode Severity Classification

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

---

