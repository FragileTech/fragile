# The Market Sieve: Permits and Certificates

The market Sieve is the operational protocol that determines whether pricing statements are valid in the current regime. It follows the permit vocabulary of `docs/source/hypopermits_jb.md`.

## Permit Vocabulary

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

## Gate Permits (Core Checks)

Each gate outputs YES ($K_i^+$) or NO ($K_i^-$). NO is conservative. Below we provide **full specifications** for all 21 market gate nodes.

### Summary Table

| Node | Permit | Market Check | Interpretation | Example observable |
|---|---|---|---|---|
| 1 | $D_E$ | Solvency / budget | Total losses bounded | aggregate VaR, capital ratio |
| 2 | $\mathrm{Rec}_N$ | Turnover limit | No chattering trades | turnover rate, cancel ratio |
| 3 | $C_\mu$ | Compactness | Leverage and positions bounded | leverage, position size |
| 4 | $\mathrm{SC}_\lambda$ | Scale stability | Parameters not drifting too fast | vol of parameters |
| 5 | $\mathrm{SC}_{\partial c}$ | Stationarity | model drift tolerable | regime drift rate |
| 6 | $\mathrm{Cap}_H$ | Information capacity | market depth supports state | depth, spread, order flow |
| 7 | $\mathrm{LS}_\sigma$ | Stiffness | price impact bounded | impact vs size |
| 7a | $\mathrm{Bif}$ | Bifurcation | no regime instability | Jacobian determinant |
| 7b | $\mathrm{Sym}$ | Alternatives | multiple strategies exist | policy entropy |
| 7c | $\mathrm{SC}_{\text{new}}$ | New regime stability | new mode stable | variance after switch |
| 7d | $\mathrm{TB}_{\text{switch}}$ | Switching cost | transition affordable | switch cost vs budget |
| 8 | $\mathrm{TB}_\pi$ | Connectivity | clearing network connected | graph connectivity |
| 9 | $\mathrm{TB}_O$ | Tameness | pricing function smooth | gamma bounds, convexity |
| 10 | $\mathrm{TB}_\rho$ | Mixing | regime exploration adequate | regime transition counts |
| 11 | $\mathrm{Rep}_K$ | Representation | regime complexity within budget | $H(K)$ vs $\log|\mathcal{K}|$ |
| 12 | $\mathrm{GC}_\nabla$ | Oscillation | no endogenous cycles | boom-bust indicator |
| 13 | $\mathrm{Bound}_\partial$ | Boundary coupling | prices grounded in data | $I(B;K)$ |
| 14 | $\mathrm{Bound}_B$ | Overload | data channel saturated | quote outages, spreads |
| 15 | $\mathrm{Bound}_\Sigma$ | Starvation | insufficient data | thin trading, stale prices |
| 16 | $\mathrm{GC}_T$ | Alignment | incentives match constraints | funding vs risk signals |
| 17 | $\mathrm{Cat}_{\mathrm{Hom}}$ | Lock | no structural arbitrage | arbitrage cycle detection |

---

### Node 1: Solvency Check ($D_E$) — Conservation of Capital

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

---

### Node 2: Turnover Check ($\mathrm{Rec}_N$) — No Zeno Trading

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

---

### Node 3: Compactness Check ($C_\mu$) — Leverage Bounds

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

---

### Node 4: Scale Stability Check ($\mathrm{SC}_\lambda$) — Parameter Drift

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

---

### Node 5: Stationarity Check ($\mathrm{SC}_{\partial c}$) — Regime Stability

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

---

### Node 6: Information Capacity Check ($\mathrm{Cap}_H$) — Market Depth

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

---

### Node 7: Stiffness Check ($\mathrm{LS}_\sigma$) — Price Impact Bounds

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

---

### Node 7a: Bifurcation Check ($\mathrm{Bif}$) — Regime Stability

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

---

### Node 7b: Alternatives Check ($\mathrm{Sym}$) — Strategy Diversity

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

---

### Node 7c: New Regime Stability ($\mathrm{SC}_{\text{new}}$)

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

---

### Node 7d: Switching Cost Check ($\mathrm{TB}_{\text{switch}}$)

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

---

### Node 8: Connectivity Check ($\mathrm{TB}_\pi$) — Network Topology

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

---

### Node 9: Tameness Check ($\mathrm{TB}_O$) — Pricing Smoothness

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

---

### Node 10: Mixing Check ($\mathrm{TB}_\rho$) — Regime Exploration

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

---

### Node 11: Representation Check ($\mathrm{Rep}_K$) — Regime Complexity

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

---

### Node 12: Oscillation Check ($\mathrm{GC}_\nabla$) — No Endogenous Cycles

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

---

### Node 13: Boundary Coupling Check ($\mathrm{Bound}_\partial$) — Price Grounding

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

---

### Node 14: Overload Check ($\mathrm{Bound}_B$) — Data Saturation

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

---

### Node 15: Starvation Check ($\mathrm{Bound}_\Sigma$) — Data Sufficiency

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

---

### Node 16: Alignment Check ($\mathrm{GC}_T$) — Incentive Consistency

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

---

### Node 17: Lock Check ($\mathrm{Cat}_{\mathrm{Hom}}$) — No Structural Arbitrage

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

---

### Nodes 18-21: Extended Checks

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

## Barrier Permits (Failure Defense)

Barriers return **Blocked** ($K^{\mathrm{blk}}$) or **Breached** ($K^{\mathrm{br}}$). When breached, pricing enters **defense mode**: conservative bounds, reduced position limits, or suspension.

### Summary Table: 20 Market Barriers

| Barrier | Category | Meaning | Trigger | Defense |
|---------|----------|---------|---------|---------|
| BarrierSat | Position | Actuator saturation | Position hits hard limit | Cap positions |
| BarrierTypeII | Scaling | Vol-of-vol crisis | $\beta > \alpha$ | Freeze updates |
| BarrierGap | Liquidity | Price discontinuity | Spread > threshold | Widen quotes |
| BarrierOmin | Dynamics | Flash crash | $\|dp/dt\| > \text{limit}$ | Circuit breaker |
| BarrierCausal | Information | Prediction lag | Forecast horizon exceeded | Shorten horizon |
| BarrierScat | Representation | Market fragmentation | $I(B;K) \to 0$ | Consolidate |
| BarrierMix | Diversity | Herding | $H(\pi) \to 0$ | Inject noise |
| BarrierCap | Control | Uncontrollability | No hedge exists | Reduce exposure |
| BarrierVac | Stability | Regime vacuum | Bifurcation detected | Stabilize |
| BarrierFreq | Oscillation | HFT instability | Resonance detected | Rate limit |
| BarrierEpi | Information | Overload | Channel saturated | Throttle |
| BarrierAction | Execution | Trade impossible | Execution fails | Queue/cancel |
| BarrierInput | Resources | Data starvation | No quotes available | Use stale + discount |
| BarrierVariety | Hedging | Incomplete market | Hedge unavailable | Accept residual |
| BarrierBode | Tradeoff | Risk waterbed | Suppress one, amplify another | Balanced response |
| BarrierLock | Regulatory | Hard limit | Legal/regulatory breach | Mandatory stop |
| BarrierLiq | Liquidity | Illiquidity crisis | Spread/depth threshold | Interval pricing |
| BarrierLev | Leverage | Excess leverage | Leverage > limit | Deleveraging |
| BarrierRef | Reference | Price integrity | Oracle deviation | Fallback oracle |
| BarrierDef | Credit | Default event | Credit event trigger | Recovery protocol |

---

### BarrierSat: Position Saturation

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

---

### BarrierTypeII: Scaling Hierarchy Violation

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

---

### BarrierGap: Liquidity Gap

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

---

### BarrierOmin: Flash Crash Detection

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

---

### BarrierCausal: Information Horizon Exceeded

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

---

### BarrierScat: Market Fragmentation

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

---

### BarrierMix: Herding / Mode Collapse

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

---

### BarrierCap: Uncontrollability

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

---

### BarrierVac: Regime Instability / Bifurcation

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

---

### BarrierFreq: HFT Oscillation / Resonance

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

---

### BarrierEpi: Information Overload

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

---

### BarrierAction: Execution Impossibility

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

---

### BarrierInput: Data Starvation

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

---

### BarrierVariety: Hedging Impossibility (Ashby)

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

---

### BarrierBode: Risk Waterbed Effect

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

---

### BarrierLock: Regulatory Hard Stop

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

---

### BarrierLiq: Liquidity Threshold

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

---

### BarrierLev: Leverage Threshold

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

---

### BarrierRef: Reference Price Integrity

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

---

### BarrierDef: Default/Credit Event

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

---

### Barrier Interaction: Multi-Barrier Coordination

When multiple barriers breach simultaneously, the market enters **crisis mode**:

**Priority ordering:**
1. BarrierLock (always first—legal requirement)
2. BarrierOmin (safety—prevent crash continuation)
3. BarrierSat, BarrierCap (position management)
4. All others (risk management)

**Cascade detection:** If $\ge 3$ barriers breach within $\tau_{\text{cascade}}$, invoke **SurgeryMode** (Section 7.6).

## Edge Validity and Determinism

**Edge validity.** An edge $N_1 \xrightarrow{o} N_2$ is valid iff the certificate $K_o$ implies the precondition of $N_2$.

**Determinism policy.** Any UNKNOWN check is treated as NO. This routes execution to barrier defenses and guarantees conservative pricing.

## Promotions and Inconclusive Upgrades

Promotion and inc-upgrade rules are applied during context closure: the market aggregates certificates until no more promotions are possible. This makes pricing conclusions **monotone** with respect to evidence.

## Surgery (Interventions)

Surgery nodes are **market interventions** that repair violations and re-enter the Sieve:
- circuit breakers,
- margin calls and position reductions,
- central bank liquidity,
- temporary price bands or auctions.

A surgery outputs a re-entry certificate $K^{\mathrm{re}}$ that proves preconditions for the next gate.


