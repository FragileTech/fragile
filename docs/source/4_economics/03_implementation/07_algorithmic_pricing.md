# Algorithmic Pricing Theory

This section develops the **information-theoretic** and **computational** aspects of market pricing, connecting to algorithmic complexity and the physics of computation.

## Kolmogorov Complexity of Prices

:::{prf:definition} Price Complexity
:label: def-price-complexity

The **Kolmogorov complexity** of a price series $\{p_t\}_{t=1}^T$ is the length of the shortest program that generates it:
$$
K(p_{1:T}) = \min_{\text{program } \pi} \{|\pi| : U(\pi) = p_{1:T}\},
$$
where $U$ is a universal Turing machine and $|\pi|$ is program length in bits.

**Interpretation:** $K(p)$ measures the **intrinsic information content** of prices—how much description is needed to specify them exactly.
:::

:::{prf:proposition} Compressibility Bounds
:label: prop-compress-bounds

Price series satisfy complexity bounds:
$$
K(p_{1:T}) \leq H(p_{1:T}) + O(\log T),
$$
where $H$ is the Shannon entropy rate.

**Market phases by complexity:**
- **Crystal phase (efficient):** $K(p) \approx K(\text{random})$ — prices are incompressible
- **Liquid phase (predictable):** $K(p) < K(\text{random}) - \epsilon$ — structure exists
- **Gas phase (chaotic):** $K(p) \approx K(\text{random})$ but structure is emergent
:::

---

## Three Pricing Phases

Markets exhibit **phase transitions** in their complexity characteristics:

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

---

## Computational Depth of Price Discovery

:::{prf:definition} Price Discovery Depth
:label: def-price-depth

The **computational depth** of a price is the time required to compute it from fundamentals:
$$
\text{Depth}(p) = \min_{\pi : U(\pi) = p} \{\text{runtime}(\pi)\}.
$$

**Interpretation:** Deep prices require complex computation; shallow prices are easily derived.
:::

:::{prf:proposition} Depth-Complexity Tradeoff
:label: prop-depth-complexity

Prices exhibit a tradeoff between specification complexity and computational depth:
$$
K(p) \cdot \text{Depth}(p) \geq \Omega(\text{Information Content}(p)).
$$

**Implications:**
- Simple prices (low $K$) require deep computation to discover
- Complex prices (high $K$) may be computationally shallow but hard to specify
- Arbitrage opportunities have high depth (hard to find) but low complexity once found
:::

---

## Levin Limit for Markets

:::{prf:theorem} Levin Market Limit
:label: thm-levin-market

There exists a **thermodynamic bound** on market prediction analogous to Levin's universal prior:
$$
\mathbb{P}(\text{price series } p) \propto 2^{-K(p)},
$$
and the expected prediction error satisfies:
$$
\mathbb{E}[\text{error}] \geq \frac{k_B T_{\text{market}}}{E_{\text{computation}}},
$$
where $T_{\text{market}}$ is market temperature (risk tolerance) and $E_{\text{computation}}$ is energy expended on prediction.

**Proof sketch:**
1. By Levin's universal prior, probability is bounded by Kolmogorov complexity.
2. Prediction error is lower-bounded by Bayesian optimal (Levin prior).
3. Computation requires energy (Landauer bound): $E \geq k_B T \ln 2$ per bit erased.
4. Market temperature scales energy costs; combining gives the bound.

**Implications:**
- **No free lunch:** Better prediction requires more computation/energy.
- **Thermodynamic consistency:** Market efficiency has physical foundations.
- **HFT limits:** Speed requires energy; there's a speed-energy tradeoff.
:::

---

## Algorithmic Information and Market Efficiency

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

::::{note} Connection to Standard Finance #16: EMH as Degenerate Algorithmic Efficiency
**The General Law (Fragile Market):**
Market efficiency is characterized by **Kolmogorov complexity** of prices:
$$
\eta_{\text{eff}} = 1 - \frac{K(p_{t+1} \mid \mathcal{I}_t)}{K(p_{t+1})}
$$
where $K(\cdot)$ is algorithmic complexity. Three phases: Crystal ($\eta \approx 0$, predictable), Liquid ($\eta \approx 0.5$, semi-efficient), Gas ($\eta \approx 1$, random).

**The Degenerate Limit:**
Infinite computational power. Instantaneous information processing. No complexity bounds. No regime structure.

**The Special Case (Efficient Market Hypothesis):**
$$
\mathbb{E}[r_{t+1} \mid \mathcal{I}_t] = r_f \quad \text{(weak/semi-strong/strong form)}
$$
Prices fully reflect available information; returns are unpredictable conditional on public information.

**What the generalization offers:**
- **Computational bounds**: Market efficiency limited by collective computational power (Levin limit)
- **Phase structure**: Markets transition between crystal (inefficient), liquid (semi-efficient), gas (random)
- **Algorithmic arbitrage**: Exploitable structure = low conditional complexity = computable prediction
- **Information hierarchy**: Weak, semi-strong, strong EMH as complexity conditions
- **Regime-aware**: Efficiency varies with market regime $K$—not a fixed property
::::

---

## Price as Proof

:::{prf:definition} Proof-Carrying Prices
:label: def-proof-price

A **proof-carrying price** is a tuple $(p, \pi)$ where:
- $p$ is the price
- $\pi$ is a certificate/proof that $p$ satisfies required properties

The verification function $V(p, \pi) \in \{\text{ACCEPT}, \text{REJECT}\}$ runs in polynomial time.
:::

:::{prf:proposition} Sieve as Proof System
:label: prop-sieve-proof

The Market Sieve (Section 7) implements a proof system for prices:
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

---

## Computational Cost Analysis

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

---

