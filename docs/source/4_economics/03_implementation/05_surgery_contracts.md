# Surgery Contracts (Market Interventions)

When a failure mode reaches Level 3 or above, the market requires **surgery**—a structured intervention that repairs the violation and returns the system to a safe state. Each surgery is a **contract** specifying preconditions, actions, and postconditions.

## Surgery Contract Structure

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

---

## SurgCE: Bailout and Recapitalization

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

---

## SurgCD: Forced Deleveraging and Breakup

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

---

## SurgCC: Circuit Breakers and Speed Controls

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

---

## SurgTE: Trading Halt and Price Auction

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

---

## SurgTD: Market Maker of Last Resort

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

---

## SurgSE: Emergency Margin Relief

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

---

## SurgSD: Volatility Injection

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

---

## SurgBC: Incentive Realignment

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

---

## Surgery Coordination

When multiple surgeries are needed simultaneously:

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

---

