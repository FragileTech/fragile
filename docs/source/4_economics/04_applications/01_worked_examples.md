# Worked Examples

This section provides complete worked examples demonstrating the Market Hypostructure in action.

## Bond Pricing Under Fed Surprise

**Scenario:** Unexpected 100bp Fed rate hike during Treasury auction.

**Initial State:**
- 10-year Treasury yield: 4.00%
- Portfolio: Long $100M 10Y duration
- Duration: 8.5 years
- Convexity: 75

**Step 1: Pre-shock Sieve Check**
```
Gate 1 (Solvency): PASS - Equity positive
Gate 5 (Stationarity): PASS - No drift detected
Gate 6 (Capacity): PASS - Position within depth
Gate 11 (Representation): PASS - Model fits yield curve
All barriers: CLEAR
Certificate: VALID, confidence 95%
```

**Step 2: Shock Application**
```
Fed announces surprise 100bp hike
New yield: 5.00%
Price change: -8.5 × 1.00% + 0.5 × 75 × (1.00%)² ≈ -8.1%
P&L: -$8.1M
```

**Step 3: Post-shock Sieve Check**
```
Gate 1 (Solvency): PASS - Still positive equity
Gate 5 (Stationarity): FAIL - Drift > threshold
Gate 7 (Stiffness): WARNING - Mean reversion uncertain
BarrierOmin: WARNING - Price velocity elevated
BarrierInput: PASS - Data feeds normal

Certificate: BOUNDED
Price bounds: [91.0, 92.5]
Failure mode: B.E (External Shock)
Recommended action: Widen bid-ask, reduce duration
```

**Step 4: Recovery Path**
```
T+1 day: Markets stabilize
Gate 5: PASS - New regime established
BarrierOmin: CLEAR
Certificate: VALID
```

**Key insight:** The Sieve correctly identified the external shock (B.E) and widened price bounds during transition. No false positive on default cascade (C.E) since solvency maintained.

---

## Options During Vol Spike (Volmageddon)

**Scenario:** VIX spikes from 12 to 50 in one session (February 2018 replay).

**Initial State:**
- Short $10M vega in SPX options
- VIX: 12
- Short vol ETPs: Leveraged 2x
- Realized vol: 8%

**Step 1: Pre-spike State**
```
Gate 3 (Leverage): WARNING - High leverage (8x effective)
Gate 7 (Stiffness): PASS - Vol mean reversion assumed
BarrierTypeII: CLEAR - Vol-of-vol normal
BarrierVac: CLEAR - Regime stable

Certificate: BOUNDED (due to leverage warning)
Failure mode monitoring: D.D (Dispersion Success)
```

**Step 2: Vol Spike Sequence**
```
T+0: VIX 12 → 20
  BarrierTypeII: WARNING
  Gate 3: FAIL (leverage + vol product > threshold)

T+1 hour: VIX 20 → 35
  BarrierTypeII: BREACHED
  BarrierOmin: WARNING (price velocity)
  Failure mode: D.D → S.E cascade

T+2 hours: VIX 35 → 50
  Certificate: INVALID
  Multiple barriers: BREACHED
  Surgery triggered: SurgSE (margin relief)
```

**Step 3: Defense Actions**
```
BarrierTypeII defense: Reduce gamma exposure by 50%
BarrierSat defense: Scale positions to limit
SurgSE: Extend margin call deadline

Post-defense state:
  Positions: Reduced 60%
  Leverage: 3x (from 8x)
  Certificate: BOUNDED
```

**Step 4: Post-Crisis Analysis**
```
Root cause: D.D (crowded short vol) → S.E (leverage crisis)
Cascade path: D.D → S.E → (near C.E avoided by intervention)
Recovery time: 3 days to CLEAR status
Loss attribution:
  - Vol move: 70%
  - Liquidity cost: 20%
  - Forced selling: 10%
```

---

## Credit Default Cascade (2008 Style)

**Scenario:** Major financial institution default triggering counterparty concerns.

**Initial State:**
- Investment grade portfolio: $500M
- Single-name concentration: 15% in one issuer
- CDS hedge: 50% notional
- HHI index: 0.08 (moderate concentration)

**Step 1: Pre-default**
```
Gate 1 (Solvency): PASS
Gate 3 (Leverage): PASS (low leverage)
BarrierSat: WARNING (concentration at 15%)
Node 14 (Coupling): WARNING (CDS-bond basis elevated)

Certificate: BOUNDED
Monitoring: C.D (Too-Big-to-Fail)
```

**Step 2: Default Event**
```
T+0: Lehman-equivalent defaults
Concentrated position: -75% overnight
CDS hedge: +60% (partial offset)
Net loss: -$11.25M

Gate 1: FAIL (solvency impaired for that position)
BarrierGap: BREACHED (credit event gap)
Failure mode: C.E activated
```

**Step 3: Cascade Propagation**
```
T+1 day: Counterparty concerns spread
  CDS spreads: +200bp across IG
  Interbank: Freeze beginning

Certificate trace:
  C.E metrics: Branching factor = 1.2 > 1.0 (supercritical)
  T.D metrics: Volume down 80%

Surgery trigger: SurgCE (bailout/backstop)
```

**Step 4: Intervention**
```
SurgCE applied:
  - Fed liquidity facility announced
  - Counterparty guarantees
  - Funding normalized

Post-intervention:
  Gate 1: PASS (guarantee counts as capital)
  Branching factor: 0.7 < 1.0 (subcritical)
  Certificate: BOUNDED

Recovery:
  T+30 days: C.E cleared
  T+90 days: All barriers CLEAR
```

**Key insight:** The Sieve detected C.D risk (concentration) pre-event, correctly identified C.E cascade post-event, and tracked intervention effectiveness.

---

## Crypto Oracle Attack

**Scenario:** Chainlink oracle reports manipulated price for DeFi lending protocol.

**Initial State:**
- ETH collateral: $50M
- Borrowed stablecoins: $30M (60% LTV)
- Oracle: Chainlink ETH/USD
- Health factor: 1.67

**Step 1: Normal Operation**
```
Gate 1 (Solvency): PASS
Node 8 (Connectivity): PASS (oracle functional)
BarrierRef: CLEAR (oracle deviation < 1%)

Certificate: VALID
```

**Step 2: Oracle Manipulation**
```
T+0: Oracle reports ETH = $500 (actual: $2000)
  BarrierRef: BREACHED (deviation 75%)
  Immediate effect: Apparent LTV = 240%
  Protocol triggers liquidation

Certificate: INVALID
Failure mode: C.E triggered by false data
```

**Step 3: Cascade Effects**
```
Liquidation cascade:
  T+0: Protocol attempts to sell $50M ETH at $500
  T+1 block: MEV bots front-run
  T+2 blocks: Market absorbs selling
  Actual execution: $1800 average

Additional damage:
  - Protocol TVL drops 40%
  - Cross-protocol contagion via composability

Failure modes active:
  - C.E (liquidation cascade)
  - T.C (composability complexity)
  - B.C (oracle incentive misalignment)
```

**Step 4: Defense & Recovery**
```
Defense actions:
  BarrierRef defense: Reject outlier prices
  Circuit breaker: Pause protocol
  Multi-oracle: Require 3/5 consensus

Post-incident:
  Oracle source: Expanded to 5 providers
  Deviation threshold: Tightened to 5%
  Time delay: 10-minute TWAP required

Certificate recovery:
  T+2 days: BarrierRef CLEAR with new design
  T+7 days: Full certificate VALID
```

---

## Cross-Asset Contagion

**Scenario:** Emerging market crisis spreading across asset classes.

**Initial State:**
- EM equity: $100M
- EM local currency bonds: $50M
- USD/EM FX hedge: 50%
- Commodity exposure (oil): $25M
- Correlation assumption: 0.3 across assets

**Step 1: Initial Shock (EM Equities)**
```
T+0: EM political crisis
EM equities: -15%
Initial loss: $15M

Gate 5 (Stationarity): FAIL (regime break)
Failure mode: B.E (External Shock)
```

**Step 2: FX Contagion**
```
T+1 day: EM currencies depreciate 10%
FX hedge: Partially offsets
Net FX loss: $2.5M (after hedge)

Correlation observation:
  Actual EM eq/FX correlation: 0.8 (vs 0.3 assumed)

Gate 11 (Representation): FAIL
Node 14 (Coupling): FAIL (basis blowout)
```

**Step 3: Bond Market Impact**
```
T+2 days: EM bond spreads widen 300bp
Local bonds: -20% (duration + FX + spread)
Loss: $10M

T.D emerging: EM bond liquidity drying up
Certificate: INVALID (multiple gate failures)
```

**Step 4: Commodity Spillover**
```
T+3 days: Oil drops 10% on global growth fears
Commodity loss: $2.5M

Total portfolio loss: $30M (17% of initial $175M)

Failure mode progression:
  B.E (shock) → B.D (EM liquidity starvation) →
  T.D (frozen EM bonds) → D.E (correlation spike)
```

**Step 5: Multi-Barrier Coordination**
```
Active barriers:
  BarrierInput (EM data quality degraded)
  BarrierGap (illiquidity gaps)
  BarrierVariety (hedge incomplete)

Surgery coordination (Section 15.10):
  Priority 1: Reduce EM exposure (BarrierSat defense)
  Priority 2: Accept illiquidity (no forced selling)
  Priority 3: Mark to conservative (interval pricing)

Recovery timeline:
  T+7 days: Volatility subsides
  T+14 days: Liquidity returning
  T+30 days: Certificate BOUNDED
  T+60 days: Certificate VALID (new correlation model)
```

**Key insight:** The multi-barrier coordination protocol prevented forced selling at distressed prices. Interval pricing preserved capital while acknowledging uncertainty.

---

