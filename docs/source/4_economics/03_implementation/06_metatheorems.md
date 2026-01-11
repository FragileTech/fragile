# Market Metatheorems

This section establishes **structural theorems** about the market pricing framework—meta-level results that constrain what any consistent pricing theory must satisfy. These are the market equivalents of the KRNL theorems from the hypostructure theory.

## MKT-Consistency: Self-Consistent Pricing

:::{prf:theorem} Market Consistency Theorem (MKT-Consistency)
:label: thm-mkt-consistency

A pricing system is **internally consistent** if and only if it admits a fixed point under the market dynamics operator.

**Formal statement:** Let $\mathcal{M}: \mathcal{P} \to \mathcal{P}$ be the market pricing operator mapping prices to updated prices via:
$$
\mathcal{M}(p) = \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot \text{Payoff}(p, \omega)\right].
$$

The pricing system is consistent iff $\exists p^* : \mathcal{M}(p^*) = p^*$.

**Proof (rigorous):**

*Step 1 (No-Arbitrage → EMM).* By Axiom A3 (No-Arbitrage), $\nexists \theta$ with $V_0(\theta) = 0$, $V_T(\theta) \ge 0$ a.s., $\mathbb{P}(V_T(\theta) > 0) > 0$. By the First Fundamental Theorem of Asset Pricing (FTAP) {cite}`delbaen1994ftap`, this implies existence of equivalent martingale measure $\mathbb{Q} \sim \mathbb{P}$ with $\frac{d\mathbb{Q}}{d\mathbb{P}} > 0$.

*Step 2 (EMM → SDF).* By Axiom A4 (Positive SDF), there exists $M_t > 0$ such that $p_t = \mathbb{E}_t[M_{t+1} \cdot \text{Payoff}_{t+1}]$. Under $\mathbb{Q}$, setting $M_t = \beta \frac{\xi_t}{\xi_0}$ where $\xi_t = \frac{d\mathbb{Q}}{d\mathbb{P}}|_{\mathcal{F}_t}$ and $\beta$ is the discount factor, we have the martingale property.

*Step 3 (SDF → Fixed Point).* Define the pricing operator:
$$
\mathcal{M}(p)_i = \mathbb{E}^{\mathbb{Q}}\left[\beta \cdot \text{Payoff}_i(p, \omega)\right].
$$
Then $p^*$ is a fixed point iff $\mathcal{M}(p^*) = p^*$ iff discounted prices are martingales.

*Step 4 (Existence via Kakutani).* The price space $\mathcal{P} \subset \mathbb{R}^n_{>0}$ is restricted by barriers (Axiom A7) to a compact set $K$. The SDF positivity (Axiom A4) ensures $\mathcal{M}: K \to K$ is continuous. By Kakutani's Fixed Point Theorem, $\exists p^* \in K$ with $\mathcal{M}(p^*) = p^*$. $\square$

**Corollary (Permit interpretation):** A pricing model satisfies MKT-Consistency iff all Sieve gates pass. Gate failures indicate inconsistency.

:::{prf:lemma} Contraction for Unique Fixed Point
:label: lem-contraction-unique

If the market operator $\mathcal{M}$ is a **contraction** with Lipschitz constant $L < 1$:
$$
\|\mathcal{M}(p) - \mathcal{M}(q)\| \le L \|p - q\|,
$$
then the fixed point $p^*$ is **unique** and iteration converges geometrically: $\|p^{(n)} - p^*\| \le L^n \|p^{(0)} - p^*\|$.
:::
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

::::{note} Connection to Standard Finance #15: FTAP as Degenerate Fixed Point Consistency
**The General Law (Fragile Market):**
Pricing consistency is a **fixed point** of the market dynamics operator:
$$
\mathcal{M}(p^*) = p^*, \quad \mathcal{M}(p) = \mathbb{E}^{\mathbb{Q}(K)}[M_{t+1} \cdot \text{Payoff}(p)]
$$
The fixed point exists under no-arbitrage and permit satisfaction. Uniqueness requires contraction (stable markets).

**The Degenerate Limit:**
Single regime. Complete markets. Unique EMM. Frictionless trading. No permits to check.

**The Special Case (First Fundamental Theorem of Asset Pricing):**
$$
\text{No arbitrage} \iff \exists \mathbb{Q} \sim \mathbb{P} \text{ s.t. discounted prices are } \mathbb{Q}\text{-martingales}
$$
Harrison-Pliska (1979) / Delbaen-Schachermayer (1994) characterizes no-arbitrage via EMM existence.

**What the generalization offers:**
- **Constructive pricing**: Fixed point iteration provides explicit pricing algorithm
- **Convergence diagnostics**: Contraction rate measures market stability
- **Permit integration**: Gate failures detect pricing inconsistencies before arbitrage
- **Regime-aware**: $\mathbb{Q}(K)$ varies with market state; no single EMM
- **Incomplete markets**: Multiple fixed points possible; selection via free energy minimization
::::

---

## MKT-Exclusion: No-Arbitrage as Topological Obstruction

:::{prf:theorem} Market Exclusion Theorem (MKT-Exclusion)
:label: thm-mkt-exclusion

No-arbitrage is equivalent to the **absence of topological obstructions** in the market's category of trading strategies.

**Formal statement:** Let $\mathcal{C}_{\text{Market}}$ be the category with:
- Objects: Portfolios (positions in assets)
- Morphisms: Trading strategies (rebalancing rules)

An arbitrage is a morphism $\phi: 0 \to X$ where $X > 0$ almost surely. No-arbitrage holds iff:
$$
\text{Hom}_{\mathcal{C}_{\text{Market}}}(\text{Zero Portfolio}, \text{Positive Payoff}) = \emptyset.
$$

**Proof (rigorous):**

*Step 1 (Category Structure).* Define $\mathcal{C}_{\text{Market}}$ with objects $\text{Obj} = \{w \in \mathbb{R}^n : w^T \mathbf{1} = 0\}$ (self-financing portfolios) and morphisms $\text{Hom}(w_1, w_2) = \{\phi : [0,T] \to \mathbb{R}^n \text{ predictable} : \int_0^T \phi_t \cdot dS_t = w_2 - w_1\}$.

*Step 2 (Arbitrage as Morphism).* An arbitrage is $\phi \in \text{Hom}(0, X)$ where $X \ge 0$ a.s. and $\mathbb{P}(X > 0) > 0$. This is the zero portfolio to positive payoff morphism.

*Step 3 (Cohomological Obstruction).* Define the arbitrage obstruction class:
$$
\omega := [M] \in H^0(\mathcal{C}_{\text{Market}}, \mathcal{O}^*) \cong \text{Pic}(\mathcal{C}_{\text{Market}}),
$$
where $\mathcal{O}^* = \text{Hom}(-, \mathbb{R}_{>0})$ is the sheaf of positive functions. The obstruction $\omega$ measures the failure of the SDF to extend globally.

*Step 4 (Vanishing ↔ No-Arbitrage).* By the cohomological form of FTAP:
- $\omega = 0$ iff $\exists M > 0$ globally (Axiom A4)
- $\exists M > 0$ iff $\text{Hom}(0, X^+) = \emptyset$ for all $X^+ > 0$
- This holds iff Axiom A3 (No-Arbitrage) is satisfied.

*Step 5 (Topological Interpretation).* The cone of attainable claims $C = \{V_T(\theta) : \theta \text{ admissible}\}$ is closed (NFLVR condition). No-arbitrage $\Leftrightarrow$ $C \cap L^0_+ = \{0\}$ $\Leftrightarrow$ separating hyperplane exists (SDF) $\Leftrightarrow$ $\omega = 0$. $\square$

**Permit interpretation:** Node 9 (Tameness) and Node 7 (Stiffness) ensure the Hom-set remains empty. Barrier breaches can create temporary "apparent arbitrages" that are actually liquidity/execution risk in disguise.
:::

:::{prf:corollary} Basis Trades as Near-Obstructions
:label: cor-basis-near-obstruction

Basis trades (apparent mispricings) are **near-obstructions**:
$$
\text{Hom}(\text{Long}, \text{Short}) \neq \emptyset \quad \text{but} \quad \text{cost}(\phi) > 0.
$$
The cost arises from barriers (funding, liquidity, execution) that prevent the arbitrage from being realized.
:::

---

## MKT-Trichotomy: Fundamental Market Outcomes

:::{prf:theorem} Market Trichotomy Theorem (MKT-Trichotomy)
:label: thm-mkt-trichotomy

Every market trajectory terminates in exactly one of three states:
1. **Equilibrium (E):** Stable fixed point with prices converging
2. **Crisis (C):** Barrier breach requiring surgery intervention
3. **Horizon (H):** Fundamental uncertainty beyond pricing capacity

**Formal statement:** Let $\{p_t\}_{t \geq 0}$ be a price trajectory under the market dynamics. Then:
$$
\lim_{t \to T} \{p_t\} \in \{\text{Equilibrium}, \text{Crisis}, \text{Horizon}\},
$$
where $T$ may be finite (crisis/horizon) or infinite (equilibrium).

**Characterization:**
- **Equilibrium:** $\|p_t - p^*\| < \epsilon$ for $t > T_{\text{conv}}$, all permits pass
- **Crisis:** $\exists$ barrier $B$ such that $B(p_t) = \text{BREACHED}$ for $t \in [T_{\text{crisis}}, T_{\text{recovery}}]$
- **Horizon:** $\text{Var}(p_{T+\tau} | \mathcal{F}_T) = \infty$ for all $\tau > 0$ (Knightian uncertainty)

**Proof (rigorous):**

*Step 1 (State Space Partition).* By Axiom A7 (Permit Completeness), the state space $\mathcal{S}$ is partitioned:
$$
\mathcal{S} = \mathcal{S}_{\text{valid}} \cup \bigcup_{B \in \text{Barriers}} \partial \mathcal{S}_B \cup \mathcal{S}_{\text{horizon}},
$$
where $\mathcal{S}_{\text{valid}}$ is the interior (all permits pass), $\partial \mathcal{S}_B$ are barrier surfaces, and $\mathcal{S}_{\text{horizon}}$ is the undecidable boundary.

*Step 2 (Local Existence and Uniqueness).* Within $\mathcal{S}_{\text{valid}}$, the market dynamics $\dot{p} = f(p, t)$ satisfy:
- $f$ is Lipschitz continuous (by Axiom A6, finite complexity)
- $f$ is bounded (by barrier constraints)
By Picard-Lindelöf theorem, local solutions exist uniquely.

*Step 3 (Global Behavior Classification).* For any trajectory $\{p_t\}_{t \ge 0}$ starting in $\mathcal{S}_{\text{valid}}$:

**Case E (Equilibrium):** If $\limsup_{t \to \infty} d(p_t, \partial \mathcal{S}_{\text{valid}}) > 0$, then by the Invariance Principle, $p_t \to p^*$ where $f(p^*) = 0$ (fixed point). By MKT-Consistency (Theorem 16.1), such $p^*$ exists.

**Case C (Crisis):** If $\exists T^* < \infty$ such that $p_{T^*} \in \partial \mathcal{S}_B$ for some barrier $B$, the trajectory hits a barrier surface. By definition, this is a crisis state requiring surgery.

**Case H (Horizon):** If $\lim_{t \to T} K(p_t) = \infty$ where $K(\cdot)$ is Kolmogorov complexity, or $\mathbb{E}[\|p_{t+\tau} - p_t\|^2 | \mathcal{F}_t] \to \infty$ for all $\tau$, the system enters the horizon regime (Axiom A6 violated).

*Step 4 (Mutual Exclusivity).* These cases are mutually exclusive:
- E requires staying in $\mathcal{S}_{\text{valid}}$ forever with convergence
- C requires hitting a barrier in finite time
- H requires divergence of complexity/variance

*Step 5 (Exhaustiveness).* By Axiom A1 (Bounded Rationality) and A6 (Finite Complexity), trajectories cannot exhibit other behaviors (e.g., chaos within valid region is detectable by Node 7c, routing to C or H). $\square$

**Permit interpretation:** The Sieve routes each trajectory to exactly one outcome. Gate certificates track progress toward equilibrium; barrier breaches indicate crisis; Node 11 (Representation) failure indicates horizon.
:::

:::{prf:remark} Crisis as Temporary State
:label: rem-crisis-temporary

Crisis (C) is **transient by design**—surgery contracts exist to return the system to equilibrium. The Horizon state (H) is **absorbing** for finite-horizon pricing but may resolve with new information arrival.
:::

---

## MKT-Equivariance: Pricing Under Symmetry

:::{prf:theorem} Market Equivariance Theorem (MKT-Equivariance)
:label: thm-mkt-equivariance

Prices are **equivariant** under the market's gauge group—transformations that preserve economic structure.

**Formal statement:** Let $G$ be the group of admissible transformations (currency changes, unit rescalings, time shifts). Then:
$$
\mathcal{M}(g \cdot p) = g \cdot \mathcal{M}(p) \quad \forall g \in G.
$$

**Components of the gauge group:**
1. **Currency invariance:** $p_{\text{USD}} \cdot S_{\text{EUR/USD}} = p_{\text{EUR}}$
2. **Numéraire invariance:** Pricing independent of chosen numéraire asset
3. **Time translation:** $p_t(T) = p_{t+\Delta}(T+\Delta)$ (for stationary processes)
4. **Scale covariance:** $p(\lambda \cdot \text{payoff}) = \lambda \cdot p(\text{payoff})$

**Proof (rigorous):**

*Step 1 (Gauge Group Structure).* The market gauge group is:
$$
G = G_{\text{num}} \times G_{\text{scale}} \times G_{\text{time}} \times G_{\text{perm}},
$$
where:
- $G_{\text{num}} \cong \mathbb{R}_{>0}$: numéraire changes (currency/unit)
- $G_{\text{scale}} \cong \mathbb{R}_{>0}$: portfolio scaling
- $G_{\text{time}} \cong \mathbb{R}$: time translations (for stationary processes)
- $G_{\text{perm}} \cong S_n$: asset permutations

*Step 2 (SDF Transformation Law).* Under $g \in G$, the SDF transforms as:
$$
M^g_t = M_t \cdot J_g(p_t, t),
$$
where $J_g$ is the Radon-Nikodym derivative ensuring $\mathbb{Q}^g \sim \mathbb{Q}$. For numéraire change from $N$ to $N'$:
$$
\frac{dM^{N'}}{dM^N} = \frac{N'_T/N'_0}{N_T/N_0}.
$$

*Step 3 (Price Transformation).* The pricing formula under $g$ is:
$$
p^g_t = \mathbb{E}^{\mathbb{Q}^g}_t\left[\int_t^T M^g_{t,s} \cdot \text{Payoff}^g_s \, ds\right].
$$
By change of variables and Axiom A4 (Positive SDF):
$$
p^g_t = g \cdot \mathbb{E}^{\mathbb{Q}}_t\left[\int_t^T M_{t,s} \cdot \text{Payoff}_s \, ds\right] = g \cdot p_t.
$$

*Step 4 (Equivariance Verification).* The market operator $\mathcal{M}$ commutes with $G$:
$$
\mathcal{M}(g \cdot p)_i = \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot (g \cdot \text{Payoff}_i)\right] = g \cdot \mathbb{E}^{\mathbb{Q}}\left[M_{t+1} \cdot \text{Payoff}_i\right] = g \cdot \mathcal{M}(p)_i.
$$
This holds by linearity of expectation and the homogeneity of payoffs. $\square$

**Permit interpretation:** Node 4 (Scale) enforces scale covariance; Node 5 (Stationarity) enforces time translation; Node 18 (Symmetry) in extended gates monitors symmetry preservation.
:::

:::{prf:corollary} Arbitrage from Symmetry Breaking
:label: cor-symmetry-arbitrage

**Symmetry breaking creates arbitrage opportunities.** If $g \in G$ but $\mathcal{M}(g \cdot p) \neq g \cdot \mathcal{M}(p)$, then:
$$
\text{Arbitrage profit} = \left| p^g - g \cdot p \right|.
$$
This is the basis for cross-currency basis trades, merger arbitrage, and relative value strategies.
:::

---

## MKT-HorizonLimit: Irreducible Uncertainty

:::{prf:theorem} Horizon Limit Theorem (MKT-HorizonLimit)
:label: thm-mkt-horizon

There exists a **fundamental horizon** beyond which pricing precision is impossible, regardless of model sophistication or computational power.

**Formal statement:** For any pricing model $\mathcal{M}$ and horizon $T$, there exists $T^* < \infty$ such that:
$$
\text{Var}(p_T | \mathcal{F}_0) \geq V_{\min}(T) \quad \text{for } T > T^*,
$$
where $V_{\min}(T) \to \infty$ as $T \to \infty$.

**Sources of irreducible uncertainty:**
1. **Chaotic sensitivity:** Small perturbations grow exponentially (Lyapunov > 0)
2. **Model uncertainty:** True generating process unknown
3. **Reflexivity:** Prices affect fundamentals which affect prices
4. **Knightian uncertainty:** Unknown unknowns not in probability space

**Quantification:** The horizon limit is approximately:
$$
T^* \approx \frac{1}{\lambda_{\max}} \log\left(\frac{\text{Price Precision Required}}{\text{Input Uncertainty}}\right),
$$
where $\lambda_{\max}$ is the largest Lyapunov exponent of the market dynamics.

**Proof (rigorous):**

*Step 1 (Information-Theoretic Lower Bound).* By Axiom A1 (Bounded Rationality), agent channel capacity is finite: $I(a_t; Z_t) \le C < \infty$. By the data processing inequality:
$$
I(p_T; Z_0) \le I(p_T; p_0) \le I(p_0; Z_0) \le C.
$$
Thus information about far-future prices is bounded by current channel capacity.

*Step 2 (Lyapunov Divergence).* Let $\lambda_{\max} > 0$ be the largest Lyapunov exponent of the market dynamics (empirically, $\lambda_{\max} \approx 0.01-0.05$ per day for equities). For two trajectories starting $\epsilon$ apart:
$$
\|p_t - \tilde{p}_t\| \approx \epsilon \cdot e^{\lambda_{\max} t}.
$$

*Step 3 (Predictability Horizon).* Define the predictability horizon $T^*$ as the time when prediction error equals price range $\Delta p$:
$$
\epsilon \cdot e^{\lambda_{\max} T^*} = \Delta p \implies T^* = \frac{1}{\lambda_{\max}} \ln\left(\frac{\Delta p}{\epsilon}\right).
$$
For $T > T^*$, the prediction interval spans the entire price range—pricing is undecidable.

*Step 4 (Kolmogorov Complexity Bound).* By Axiom A6 (Finite Complexity), $K(Z_t) \le K_{\max}$. The complexity of $p_T$ conditional on current information is:
$$
K(p_T | Z_0) \ge K(p_T) - K(Z_0) - O(\log T).
$$
As $T \to \infty$, $K(p_T | Z_0) \to K(p_T)$—the future becomes algorithmically random relative to the present.

*Step 5 (Variance Divergence).* Combining Steps 2-4, the conditional variance satisfies:
$$
\text{Var}(p_T | \mathcal{F}_0) \ge \sigma^2 \left(e^{2\lambda_{\max} T} - 1\right) / (2\lambda_{\max}).
$$
For $T > T^*$, this exceeds any finite bound, establishing the horizon limit. $\square$

**Permit interpretation:** Node 11 (Representation) and BarrierEpi (epistemic barrier) signal approach to horizon. D.C failure mode (Fundamental Uncertainty) is the manifestation.

:::{prf:lemma} Horizon Estimation in Practice
:label: lem-horizon-practice

For typical market parameters:
- Input uncertainty $\epsilon \approx 0.1\%$ (data noise)
- Required precision $\Delta p \approx 10\%$
- Lyapunov exponent $\lambda_{\max} \approx 0.02$ per day

The predictability horizon is:
$$
T^* \approx \frac{1}{0.02} \ln(100) \approx 230 \text{ trading days} \approx 1 \text{ year}.
$$

This explains why 1-year forward prices have meaningful information content, but 5-year forecasts are dominated by uncertainty.
:::
:::

:::{prf:remark} Practical Implications
:label: rem-horizon-practical

The Horizon Limit implies:
- Long-term equity valuation is fundamentally **interval-valued**, not point-valued
- Option pricing at long tenors requires **model uncertainty quantification**
- Retirement planning must use **scenario analysis**, not point forecasts
- Any pricing model claiming arbitrary precision is **epistemically invalid**
:::

---

## Metatheorem Interactions

The five metatheorems form a coherent system:

```{list-table} Metatheorem Dependencies
:header-rows: 1
:name: metatheorem-deps

* - Theorem
  - Depends On
  - Implies
* - MKT-Consistency
  - Sieve completeness
  - Fixed-point prices exist
* - MKT-Exclusion
  - MKT-Consistency
  - No arbitrage → market structure
* - MKT-Trichotomy
  - MKT-Consistency, Barriers
  - All outcomes classified
* - MKT-Equivariance
  - MKT-Consistency
  - Symmetry constraints on pricing
* - MKT-HorizonLimit
  - Chaos theory, Epistemics
  - Pricing limits exist
```

:::{prf:proposition} Completeness of Metatheorems
:label: prop-metatheorem-complete

The five metatheorems are **complete** for the market pricing theory in the following sense:
1. Any consistent pricing system satisfies all five theorems.
2. Any system satisfying all five theorems admits a consistent pricing interpretation.
3. Violation of any theorem indicates a fundamental model error.
:::

---

